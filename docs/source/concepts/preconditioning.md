# Solvers and preconditioners

Every inverse in MRX is a Krylov solve on matrix-free operators with a
matrix-free preconditioner: preconditioned CG for SPD operators, MINRES for
symmetric indefinite ones (`mrx/solvers.py`). No matrix is factorised, and
nothing larger than the dense polar core is stored.

## 1. Which solver for which operator

The sequence exposes each solve as the `solve` method of an operator,
implemented in `mrx/operators.py`. Each solves in the spaces of the object it
is called on, `seq` (Dirichlet) or `seq.free`.

| solve | entry point | solver | preconditioner |
|---|---|---|---|
| `M_k u = f` | `seq.M[k].solve(f)` | PCG | the mass atom |
| `L_0 u = f` | `seq.L[0].solve(f)` | PCG on `S_0`, kernel deflated | the k = 0 Laplacian atom |
| `L_k u = f`, k = 1, 2 | `seq.L[k].solve(f)` | the Hodge split: three SPD PCGs | the Laplacian atoms of levels k and k-1 |
| `L_3 u = f` | `seq.L[3].solve(f)` and `seq.leray(v, k=2)` | saddle-point MINRES | the k = 3 Laplacian atom, the k = 2 mass atom |
| `(M_k + eps L_k) u = f` | `seq.shifted(k, eps).solve(f)` | two SPD PCGs | the shifted-stiffness atoms |

Every solve takes a warm start `guess` and returns the signed iteration
count with `return_info=True`. `seq.M[k].precondition(r)` and
`seq.L[k].precondition(r)` apply the atoms themselves.

**Hodge split** (k = 1, 2). With `G = G_{k-1}` the incidence matrix,
`G^T S_k = 0` splits off the exact part:

```
S_{k-1} g = G^T b,   S_k x_perp = b - M_k G g,   S_{k-1} a = M_{k-1} g - G^T M_k x_perp,   x = x_perp + G a
```

`S_k` alone is singular on the exact forms, so the
middle solve runs on `S_k + M_k G W G^T M_k` (`W` the level-(k-1) mass
atom), which has the same exact-orthogonal solution for any SPD `W`.

**Saddle point** (k = 3). With `S_3 = 0` there is nothing to split:

```
| 0       D_2  | | u     |   | f |
| D_2^T  -M_2  | | sigma | = | 0 |
```

whose Schur complement is `L_3`. `sigma = M_2^{-1} D_2^T u` is the weak
gradient of `u`, which the Leray projection removes. The preconditioner is
block-diagonal.

**Shifted solve.** `G_k G_{k-1} = 0` gives exactly

```
(M_k + eps L_k)^{-1} = (M_k + eps S_k)^{-1} - eps G_{k-1} (M_{k-1} + eps S_{k-1})^{-1} G_{k-1}^T
```

that is, two SPD solves. This is the resistive step and the velocity smoothing.

**Leray projection.** `seq.leray(v, k=2)` removes the gradient part of a
Dirichlet 2-form by the k = 3 saddle solve. `seq.leray(v, k=1)` splits a
Dirichlet 1-form into `F_w + grad p_w` with `p_w = 0` on the wall by one
k = 0 Dirichlet solve, and `seq.free.leray(v, k=1)` does the same with the
free spaces (`dp/dn = v . n`).

There is no Krylov solve inside a Krylov solve. Where the weak term
`M_k G M_{k-1}^{-1} G^T M_k` of `L_k` sits inside an iteration, the mass
preconditioner replaces `M_{k-1}^{-1}` (`apply_laplacian_approx`). In the
Newton Hessian a fixed Chebyshev polynomial in the mass atom replaces
`M_1^{-1}` ([Relaxation](relaxation.md)). Every
unshifted solve deflates the harmonic forms. Every solve stops on its true
residual in the mass-atom norm ([Precision](precision.md)).

## 2. The atoms

`seq.build_preconditioners()` builds a mass atom (`MetricLumpingMass`) and
a Laplacian atom (`MetricLumpingLaplacian`) for every degree `k`, on the
Dirichlet and on the free spaces (`mrx/metric_lumping.py`). Each is block Jacobi over two uncoupled blocks:

**Bulk.** The tensor-product rows, per vector component. The Laplacian is
approximated by the Kronecker sum

```
A_c = K_r (x) M_t (x) M_z + M_r (x) K_t (x) M_z + M_r (x) M_t (x) K_z
```

where `(x)` is the Kronecker product, with unweighted 1-D masses and 1-D
stiffnesses that carry the axis mean of the metric weight `g^{aa} det DPhi`
(metric lumping), the component factor kept
exactly as a diagonal sandwich. `A_c` is inverted by fast
diagonalisation: three 1-D generalised eigenproblems at build time, three
small dense products and a pointwise divide per apply. The mass atom is a
single Kronecker product inside a diagonal sandwich that reproduces
`diag(M_k)` exactly. Under a natural condition at `r = 1` the surface term
of the weak block enters the radial stiffness as a rank-one update, scaled
by `PRODUCTION_BC_SCALE`.

**Core.** The polar rows ([polar.md](polar.md)) are not tensor-product
functions. The operator is probed on them in the residual precision, in
batches of `PROBE_BATCH` rows per call, and the block is inverted densely,
eigenvalues below `CORE_TOL` relative to the largest dropped.

**Shifted stiffness.** The same atom preconditions `M_k + eps S_k` for an
`eps` known only at the solve: the bulk Kronecker terms are shifted, and
the core pair `(M_k, S_k)` is diagonalised once so the block is
`V diag(1 / (1 + eps mu)) V^T`.

On a half-period sequence the parity views reduce the base atoms
(`ReducedAtom`, `X^T P X` with the view's expansion `X`).

**The apply.** The three vector components of a 1- or 2-form have slightly
different grid sizes. The apply pads them to one common size and treats
them together, so each step runs once instead of once per component. The
index shuffling around the blocks (picking out each component's entries,
the parity views, putting the result back in order) is worked out once when
the atom is built and done in a single lookup on the way in and one on the
way out. On a GPU every small operation costs a few microseconds no matter
how little it computes, so doing fewer of them halved the cost of an apply
(`docs/research/performance.md`).

## 3. Building and invalidation

```python
seq.set_map(Phi)                # installs the geometry, drops seq.operators
seq.build_preconditioners()     # both atoms for every k, Dirichlet and free
compute_nullspaces(seq)         # the harmonic forms, onto the bundle
```

A missing atom raises at the solve. Nothing is built on demand. The atom
payloads are `eqx.Module` pytrees of device arrays applied by one
module-level jitted function, so a rebuild for a new geometry of the same
discretisation reuses the compiled program. `test/test_poisson.py` checks
the Laplacian solves against manufactured solutions on li383, k = 0..3.
