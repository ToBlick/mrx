# Solvers and preconditioners

Every inverse in MRX is a Krylov solve on callable matvecs with a callable
preconditioner. No matrix is factorised; nothing larger than a dense polar
core is stored. This page says which solver and which preconditioner each
operator uses, and what the two preconditioner atoms are. The measurements
behind the choices are in `docs/research/preconditioner_technical_note_source.md`
and `preconditioner_lessons.md`.

## 1. Which solver for which operator

Solvers are in `mrx/solvers.py`; the wiring is in `mrx/operators.py`.
Every solve runs under the refinement loop of
[precision.md](precision.md): the Krylov iteration in the working
precision, the true residual in float64.

| solve | entry point | solver | preconditioner |
|---|---|---|---|
| `M_k u = f` | `apply_inverse_mass_matrix` | `solve_singular_cg` | the mass atom |
| `L_0 u = f` | `apply_inverse_laplacian`, k=0 | `solve_singular_cg`, harmonic mode deflated | the Laplacian atom |
| `L_k u = f`, k=1,2 | `apply_inverse_laplacian_hodge` | the Hodge split: PCG on the SPD strong stiffnesses `S_k + M_k D W D^T M_k` (`W` the mass atom of level k-1), the exact part from a `(k-1)`-level solve, one more to close; no saddle system, no mass inverse | the Laplacian atoms of levels k and k-1 |
| `L_3 u = f` | `apply_inverse_laplacian_saddle` | `solve_saddle_point_minres` on `[[0, D_2], [D_2^T, -M_2]]` (`S_3 = 0`, nothing to split) | upper: the k=3 Laplacian atom; lower: the k=2 mass atom |
| `(L_k + eps M_k) u = f` | `apply_inverse_shifted_laplacian` | k=0 CG, k>=1 the saddle MINRES; nothing deflated | as above |
| `(M_k + eps L_k) u = f` | `apply_inverse_mass_plus_eps_laplace_matrix` | two SPD CG solves, `M_k + eps S_k` and `M_{k-1} + eps S_{k-1}`, through the split identity `(M_k + eps S_k)^-1 - eps D_{k-1} (M_{k-1} + eps S_{k-1})^-1 D_{k-1}^T` (exact, from `D_k D_{k-1} = 0`) | the shifted-stiffness form of the Laplacian atom, `(M^ + eps S^)^-1` |

The saddle system for k >= 1 is

```
| K_k + eps M_k    D_{k-1}  | | u |   | f |
| D_{k-1}^T       -M_{k-1}  | | s | = | 0 |
```

whose Schur complement is `L_k` itself; the lower unknown `s = M_{k-1}^-1
D_{k-1}^T u` is the weak codifferential of the solution, which the Leray
projection uses as its gradient part. MINRES needs an SPD preconditioner on
each block, block-diagonal, never coupled through a Schur complement; the
harmonic forms are deflated from the upper block only, since a harmonic
`v` has `D^T v = 0` and the saddle matrix's nullspace is `(v, 0)`.

There is no Krylov solve inside a Krylov solve. The weak term `D_{k-1}
M_{k-1}^{-1} D_{k-1}^T` of `L_k` is applied with the mass *preconditioner*
in place of `M_{k-1}^{-1}` (`apply_laplacian_approx`) wherever it sits
inside an iteration -- the operator the Laplacian atoms probe their polar
core with, and the hat operator of the Hodge split, whose exact-orthogonal
solution does not depend on `W`. The mass preconditioner is therefore part
of the operator at k >= 1, not only part of the solve.

## 2. One preconditioner per solve

`build_preconditioners` builds the mass atom (`MetricLumpingMass`) and the
Laplacian atom (`MetricLumpingLaplacian`) for every `(k, BC)` onto the
bundle `seq.operators` (`mass_lumping`, `laplacian_lumping`, keyed `(k,
dirichlet)`), and every solve through the sequence uses the atom of its own
`(k, BC)`. A missing atom raises at the solve; nothing is built on demand
and nothing is substituted.

## 3. The Laplacian atom: `MetricLumpingLaplacian`

`mrx/metric_lumping_laplacian.py`. Block Jacobi with two kinds of block,
applied independently.

**Bulk.** For each vector component `c` of `V^k`, the diagonal block of `L_k`
on the tensor-product rows is approximated by a three-term Kronecker sum

```
A_c = K_r ⊗ M_t ⊗ M_z + M_r ⊗ K_t ⊗ M_z + M_r ⊗ M_t ⊗ K_z
```

with unweighted 1D masses `M_a` and 1D stiffnesses `K_a` that carry the
metric weight averaged over the other two axes (`component_factors`; the
*bundled* mean `<g^{aa} J>`, since `g^{tt} J ~ 1/r` is integrable where
`g^{tt}` alone is not). On a derivative axis of the component the stiffness
is that of the derivative splines themselves, from their tabulated
derivatives (`seq.dd_basis_jk`). The component factor `m_k / J` is pulled
out as a diagonal similarity `D^{1/2} A_c D^{1/2}` (`component_diagonal`).
`A_c` is inverted exactly by fast diagonalisation: three 1D generalised
eigenproblems at build time (`_simultaneous_diagonalize_pair`), then three
small dense products and a pointwise divide per apply (`_fd_apply_3d`).
Cost per apply is `O(N (n_r + n_t + n_z))`; storage is `O(n^2)` per axis.
Requirement: `n_r >= p + 2`; a one-element radial mesh has no separable atom.

**Core.** The polar rows, where the extraction fuses a ring of raw functions,
are not tensor-product functions. `core_rows` lists them; `probe_core_block`
forms `L_k` on those rows by one operator apply per row, on the
residual-precision sequence, and `_dense_symmetric_inverse` inverts the
block by `eigh`, dropping eigenvalues below `CORE_TOL` (4096 machine
epsilons of the residual precision) relative to the largest. Probed in the
working precision the cut-off zeroed real modes of the k=1 core and made the
preconditioner blind to their residual. The shifted-stiffness form of the
atom needs `(M_k + eps S_k)^{-1}` on the same rows for an `eps` known only
at the solve: the pair `(M_k, S_k)` on the core is diagonalised once at
build and the block is `V diag(1 / (1 + eps mu)) V^T`, two small matmuls
per solve.

**Natural boundary term.** Under a free condition at `r = 1` the weak
block's integration by parts leaves a surface term `alpha (e e^T) ⊗ M_t ⊗
M_z` with `e` the one-hot derivative-spline trace, the shape of the first
Kronecker term, so it merges into `K_r` as a rank-one update at no cost, on
the components whose radial axis is a derivative axis. `alpha` is the face
average of the component's mass weight (`_face_alpha`) times
`PRODUCTION_BC_SCALE = 3.0`: the exact surface integral is a penalty on the
normal trace and the atom wants it closer to the hard `u_r = 0` limit; 3.0
is inside the flat optimum of a 24-cell sweep
(`docs/research/natural_bc_coefficient_handoff.md`). Under Dirichlet the
term is zero.

## 4. The mass preconditioner: `MetricLumpingMass`

Same file, same shape, simpler algebra: a mass is a single Kronecker product,
so the bulk inverse is three 1D dense solves inside the diagonal sandwich
`Lam` that reproduces `diag(M_k)` exactly (`_kron_mass_model_1d`). The polar
core is probed with `apply_mass_matrix` on the residual-precision sequence
and inverted densely; there is no pseudoinverse of the extraction anywhere.
`apply_in(dtype)` is the atom as part of an operator of that precision (the
weak term of the hat Laplacian on the float64 view).

## 5. Building and invalidation

```python
seq.set_map(F)                  # installs the geometry, drops seq.operators
seq.build_preconditioners()     # a fresh bundle: both atoms for every (k, BC)
compute_nullspaces(seq)         # the harmonic forms, onto the bundle
```

`set_geometry` drops the whole bundle because everything on it factorises
the old metric; there is no cache to invalidate. The atom payloads are
`eqx.Module` pytrees built eagerly at construction, with one jitted apply
per tree structure, so a rebuild for a new geometry of the same
discretisation reuses the compiled program.

## 6. Not in production

Multigrid, Chebyshev or Richardson acceleration, CP fits, HX auxiliary-space
transfers, dense outer-ring probes, the Fourier coarse correction and the
per-DoF Jacobi baselines were measured and are not used
(`docs/research/preconditioner_lessons.md`); the research code lives on
branch `greville-prod` under `mrx/experimental/`.

`test/test_poisson.py` pins the iteration counts of the production
preconditioners on the session fixture, all eight `(k, BC)` Laplacians
against manufactured solutions. Rank alternatives by total time, not
iterations; iteration counts move by about 1% between runs, and only a
two-digit percentage is a result.
