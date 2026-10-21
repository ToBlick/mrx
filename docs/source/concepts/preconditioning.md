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
| `M_k u = f` | `seq.M[k].solve(f)` | PCG | the slab model (k = 1, 2), the mass atom (k = 0, 3) |
| `L_0 u = f` | `seq.L[0].solve(f)` | PCG on `S_0`, kernel deflated | the k = 0 Laplacian atom |
| `L_k u = f`, k = 1, 2 | `seq.L[k].solve(f)` | the Hodge split: three PCGs on the stiffnesses | the Laplacian atoms of levels k and k-1 |
| `S_k u = b`, `b = G_k^T w` | `seq.S[k].solve(b)` | PCG on the singular `S_k` | the Laplacian atom of level k |
| `L_3 u = f` | `seq.L[3].solve(f)` and `seq.leray(v, k=2)` | the Hodge split at k = 3: two PCGs on `S_2`, one gauge on `S_1` | the Laplacian atoms of levels 2 and 1 |
| `(M_k + eps L_k) u = f` | `seq.shifted(k, eps).solve(f)` | three SPD PCGs, the Hodge split with `delta = 1 / eps` | the shifted-stiffness atoms plus the Laplacian atom of level k - 1 on gradients |
| `(M_k + eps S_k) u = f` | `seq.shifted_stiffness(k, eps).solve(f)` | one PCG on `M_k + eps S_k` | the same |

Every solve takes a warm start `guess` and returns the signed iteration
count with `return_info=True`. `seq.M[k].precondition(r)` and
`seq.L[k].precondition(r)` apply the atoms themselves.

**Hodge split** (k = 1, 2). With `G = G_{k-1}` the incidence matrix,
`G^T S_k = 0` splits off the exact part:

```
S_{k-1} g = G^T b,   S_k x_perp = b - M_k G g,   S_{k-1} a = M_{k-1} g - G^T M_k x_perp,   x = x_perp + G a
```

Each is a solve of the stiffness alone, singular for degrees 1 and 2. It
converges only as far as its right-hand side is consistent. That of
`x_perp` is consistent up to the residual of `g`, which leaves an exact
form in it of the size of the `S_{k-1}^{-1}` norm of that residual. So `g`
stops where this is a thirtieth of the tolerance relative to `b`. For
k = 2 that of `a` is consistent only as far as `g` is co-exact, and `g` is
gauged by an `S_0` solve. In a later pass `b` is a residual whose co-exact
part can be far smaller than what `g` leaves, so `x_perp` stops at a third
of the tolerance relative to `b` and not to its own right-hand side. The
residual of `a` counts through `M_k G`, which weights its fine modes by up
to the square root of the largest eigenvalue of `S_{k-1}` against
`M_{k-1}`: on li383 5 times for a random right-hand side and up to 70
times for a smooth one, growing like `1 / h`. So the solve for `a` runs in
two legs. The first stops at a third of the tolerance in its own norm. The
ratio between that norm and the norm in which the residual counts is read
off its residual and sets the tolerance of a second solve of that residual.
One pass of the split then meets the tolerance. The `S_2` multigrid serves
the middle solve of `L_2`.

**Part solves.** The solution of `L_k x = b` is the sum of a co-exact and an exact form, and a right-hand
side that has only one of them needs fewer solves. `seq.L[k].solve(b, part=...)` says which:

```
coexact (b = G_k^T w):       S_k x = b,   then a gauge solve of degree k - 1
exact (b = M_k G_{k-1} c):   S_{k-1} g = G^T b,   g gauged,   S_{k-1} a = M_{k-1} g,   x = G a
```

For a co-exact right-hand side the first solve of the split is zero and conjugate gradients run on the
singular `S_k` alone. They converge because `b` lies exactly in its range, and each iteration drops the two
mass applies and the atom of the exact-form term. The solve is stopped and refined like every other one
([Precision](precision.md)). `seq.S[k].solve(b)` leaves out the gauge solve. What comes back is then the
solution plus a closed form that nothing controls, of which only `G_k x` is determined. This is what the
force potential, the vector potential of the helicity, the harmonic 2-form and the vacuum field of the
shape optimisation use, since only `G_k x` enters them.

The stiffness solve needs a consistent right-hand side. In float32 `G_k^T w` is formed from a lifted vector
([Precision](precision.md)), since computed in float32 it is inconsistent at about `1e-7`. A right-hand side that
is only approximately in the range of `G_k^T`, such as the middle one of the Hodge split, makes the iteration
diverge. The second solve of the exact part, `S_{k-1} a = M_{k-1} g`, is consistent only as far as `g` is co-exact.
So `g` is gauged first by one solve of degree `k - 2`. That solve stops relative to `M_{k-1} g` and not to its own
right-hand side, which weights the high modes up to `h^-1` more. Without the gauge the second solve diverges. A
0-form solution has no exact part and a 3-form solution no co-exact part. For k = 3 the exact part is the whole
solve.

**k = 3.** `S_3 = 0`, so the middle solve of the split has nothing to do and is left out. What remains is

```
S_2 z = G_2^T b,   z gauged by S_1,   S_2 a = M_2 z,   u = G_2 a
```

The second unknown of the outer loop is `sigma = z = M_2^{-1} G_2^T M_3 u`, the weak gradient of `u`, and the
residual the loop judges is that of the saddle-point system of `(u, sigma)`. On li383 graded (48,64,24) it takes
116 iterations and 40 ms, against 565 and 289 ms for MINRES on the saddle-point system, which it replaces.

**Shifted solve.** This is the resistive step and the velocity smoothing.
Divided by `eps`, `M_k + eps L_k` is `delta M_k + L_k` with `delta = 1 / eps`,
and one routine solves both it and `L_k` itself (`delta = 0`). With
`G = G_{k-1}`:

```
(delta M_{k-1} + S_{k-1}) z = G^T b
(delta M_k + S_k) x_perp    = b - M_k G z
(delta M_{k-1} + S_{k-1}) a = M_{k-1} z - G^T M_k x_perp
x = x_perp + G a
```

These are three SPD solves, and at `delta = 0` they are the Hodge split above.
Because `G^T S_k = 0` and `G^T M_k G = S_{k-1}`, `x` solves the system
whatever the error of `z`. That error enters the second and the third
right-hand side so that it cancels. The third solve also takes up the
gradient part `G^T r` of the residual `r` of the second, which `S_k` does
not see. `z` stops at a tenth of the tolerance on its own residual, `x_perp`
at a third relative to `b`, and `a` in the two legs above, and the
repetition judges the result on the residual of the posed equation.
`delta = 0` is decided when the solve is compiled: the Laplacian forms no
mass term, and its singular solves remove the harmonic forms and need the
consistent right-hand sides above. With
two solves, `x = x_k - eps G z`, nothing cancelled the error of `z`. It
reached the residual multiplied by `eps`, and above `eps` of about 0.1 the
solve did not converge.

`seq.shifted_stiffness(k, eps).solve(b)` is the one solve of
`M_k + eps S_k`. It is the solution for a `b` orthogonal to the closed forms.
For any other `b` it differs from the solution by an exact form, so `G_k x`
is the same. The velocity smoothing needs only that.

**Leray projection.** `seq.leray(v, k=2)` removes the gradient part of a
Dirichlet 2-form by the k = 3 split: the part removed is its `sigma`. The
upper block of the residual the loop judges is the divergence residual, so
for a force that is 5 % of `v` the divergence of the result is 2.5e-9 of it,
against 9e-5 by the saddle-point MINRES this replaced, in a sixth of the time
(li383 graded (48,64,24)). `seq.leray(v, k=1)` splits a
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
exactly as a diagonal sandwich. The sandwich also gives the model the
angular profile of the stiffness diagonal, the square root of its ratio
to the model's own diagonal along each ring. The lumped weights miss most
along the last rings near `r = 1` on shaped cross-sections, and the
profile cuts the co-exact `S_1` iterations by a third on li383. `A_c` is inverted by fast
diagonalisation: three 1-D generalised eigenproblems at build time, three
small dense products and a pointwise divide per apply. The mass atom is a
single Kronecker product inside a diagonal sandwich that reproduces
`diag(M_k)` exactly. Under a natural condition at `r = 1` the surface term
of the weak block enters the radial stiffness as a rank-one update, scaled
by `PRODUCTION_BC_SCALE`.

**Core.** The polar rows ([polar.md](polar.md)) are not tensor-product
functions. The operator is probed on them in
batches of `PROBE_BATCH` rows per call, and the block is inverted densely.
It is positive definite, and the build raises if it is not.

**Shifted stiffness.** The same atom preconditions `M_k + eps S_k` for an
`eps` known only at the solve: the bulk Kronecker terms are shifted, and
the core pair `(M_k, S_k)` is diagonalised once so the block is
`V diag(1 / (1 + eps mu)) V^T`. The Kronecker terms model the whole
Laplacian `L_k`, while `S_k` vanishes on gradients. On gradients the operator is
`M_k`, and in the `M`-orthogonal Hodge decomposition it is exactly
`G^T M_k G = S_{k-1}` there for every `eps`. So the solve adds the Laplacian
atom of level k - 1 on gradients, `G_{k-1} P_{k-1} G_{k-1}^T`. Without it the
atom is too small on gradients by `1 / (1 + eps lambda)`, and the iteration
count grows with `eps`.

On a half-period sequence the parity views reduce the base atoms
(`ReducedAtom`, `X^T P X` with the view's expansion `X`).

**Graded spaces.** On a graded space ([polar.md](polar.md)) a component has
one Kronecker block per theta count, a level. Rings of two levels overlap
near the border between them, so the model couples the levels there. The
last few rings of each level that overlap the next one are the strips.
Without the strips the levels are uncoupled, and the model is inverted
exactly by block elimination. The atom solves the levels without their
strips, solves a small dense system on the strips, and solves the levels
again. That dense system is built once, from the level solves themselves.
For the Laplacian it is one block per toroidal mode, because all levels
of a component share the 1-D zeta matrices. For the mass it is a single
block. The Laplacian also averages the angular metric weights per level,
over the radial band of that level, instead of over the whole radius. The
zeta averages of the bands keep one common shape, so the levels still
share the zeta eigenbasis. Together this halves the iterations of the
Laplacian solves on li383 (48,64,24). The shifted-stiffness apply keeps
the levels uncoupled, because the dense system would depend on `eps`. A
uniform space has a single level and no strips.

**The slab mass model.** On shaped cross-sections the gradients of `r` and
`theta` are far from orthogonal, and the mass atom loses the coupling
between the radial and the poloidal components that this skew creates. Its
condition number on `M_1` and `M_2` is then about 50. The slab model
(`SlabMass`, `mrx/slab_mass.py`) keeps that coupling. It works on zeta
slabs, the degrees of freedom of one toroidal spline index. The block of
`M_k` inside each slab is used exactly and factored. Neighbouring slabs are
coupled through the 1-D toroidal mass alone. Inside a slab the factor is
block tridiagonal in groups of `p` rings, so an apply is one sweep inward
and one sweep outward over the rings, for all slabs at once. The radial and
poloidal components form one group and the toroidal component another, and
the coupling between the two groups is dropped. The model is built by
probing the assembled mass, once per geometry. The full factor would store
about `7 n_theta` numbers per degree of freedom. Its entries fall off fast
in theta, so it is cut to a band of a few theta cells (`THETA_HALF_WIDTH`)
and keeps a fixed number of entries per degree of freedom, still many times
more than the atom. The build never forms the full factor either. The probe
keeps the mass on its own theta band. The factor is computed one group of
rings at a time and cut to the band before the next group, so the memory of
the build also grows only like the number of degrees of freedom.
It is the preconditioner of the mass solves of `k = 1, 2` by default
(`mass_atom="slab"`, `"lumping"` keeps the atom). A mass solve stops in the
norm of its own preconditioner, the slab's for `k = 1, 2`. The atom stays the
norm of the other solves, every `M^{-1}` inside an iteration, and
`seq.M[k].precondition`.
The mass polynomial of the Newton system needs an apply that is symmetric to
round-off, which the float32 slab factors are not: with them the Hessian was
symmetric only to 2.5e-5 (li383 graded (48,64,24), where its degree fell from 37
to 9). The scalar masses keep the atom, which is already close to exact there.

**Multigrid for S_2.** The stiffness of the 2-forms, `S_2 = G_2^T M_3 G_2`,
sees the metric only through the scalar `1 / J` of the 3-form mass, so it
has none of the skew that limits the atoms. A geometric V-cycle with one
damped Jacobi step before and after the coarse correction solves it in
about 10 iterations on every mesh measured (`StiffnessMultigrid`,
`mrx/multigrid.py`). The levels are coarser sequences on the same map, and
the transfers are the knot insertions of the coarse splines, exact on
graded meshes. `seq.build_preconditioners(laplacian_atom="multigrid")` uses
it for the co-closed `S_2` solves on the Dirichlet 2-forms. These are
`seq.S[2].solve(b)`, both stiffness solves of the exact part of
`L_3` and the middle solve of the Hodge split of `L_2`.

**The apply.** The three vector components of a 1- or 2-form have slightly
different grid sizes. The apply pads them to one common size and treats
them together, so each step runs once instead of once per component. The
index shuffling around the blocks (picking out each component's entries,
the parity views, putting the result back in order) is worked out once when
the atom is built and done in a single lookup on the way in and one on the
way out. On a GPU every small operation costs a few microseconds no matter
how little it computes, so doing fewer of them halved the cost of an apply.

## 3. Building and invalidation

```python
seq.set_geometry_from_map(Phi)                # installs the geometry, drops seq.operators
seq.build_preconditioners()     # both atoms for every k, Dirichlet and free
compute_nullspaces(seq)         # the harmonic forms, onto the bundle
```

A missing atom raises at the solve. Nothing is built on demand. The atom
payloads are `eqx.Module` pytrees of device arrays applied by one
module-level jitted function, so a rebuild for a new geometry of the same
discretisation reuses the compiled program. `test/test_poisson.py` checks
the Laplacian solves against manufactured solutions on li383, k = 0..3.
