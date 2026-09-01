# Solvers and preconditioners

Every inverse in MRX is a Krylov solve on callable matvecs with a callable
preconditioner. No matrix is factorised; nothing larger than a dense polar
core is stored. This page says which solver and which preconditioner each
operator uses, and what the production preconditioner `metric_lumping` is.
The measurements behind the choices are in
`docs/research/preconditioner_technical_note_source.md`.

## 1. Which solver for which operator

Solvers are in `mrx/solvers.py`; the wiring is in `mrx/operators.py`.

| solve | entry point | solver | preconditioner |
|---|---|---|---|
| `M_k u = f` | `apply_inverse_mass_matrix` | `solve_singular_cg` | mass, `metric_lumping` |
| `L_0 u = f` | `apply_inverse_laplacian`, k=0 | `solve_singular_cg`, harmonic mode deflated | Laplacian atom |
| `L_k u = f`, k=1,2,3 | `apply_inverse_laplacian` | `solve_saddle_point_minres` | lower: mass `metric_lumping` of degree k-1; upper: Laplacian atom |
| `(L_k + eps M_k) u = f` | `apply_inverse_shifted_laplacian` | as above; nothing deflated | as above, plus a `1/eps` harmonic coarse correction when the harmonic vector exists |
| `(M_k + eps L_k) u = f` | `apply_inverse_mass_plus_eps_laplace_matrix` | CG for k=0, saddle MINRES otherwise | mass `metric_lumping` on both blocks |

The saddle system for k >= 1 is

```
| K_k + eps M_k    D_{k-1}  | | u |   | f |
| D_{k-1}^T       -M_{k-1}  | | s | = | 0 |
```

whose Schur complement is `L_k` itself. MINRES needs an SPD preconditioner
on each block; `solve_saddle_point_minres` takes `precond_upper` and
`precond_lower` as callables. Every solver takes `tol=None`, which is
`mrx.sqrt_eps()`, and reports `info = -k` after `k` iterations on convergence
and `+k` on failure.

There is no Krylov solve inside a Krylov solve. The weak term
`D_{k-1} M_{k-1}^{-1} D_{k-1}^T` of `L_k` is applied with the mass
*preconditioner* in place of `M_{k-1}^{-1}` (`apply_laplacian_approx`).
The mass preconditioner is therefore part of the operator at k >= 1, not
only part of the solve, and changing it changes `L_k`.

## 2. Kinds

`MassPreconditionerSpec(kind=...)` in `mrx/preconditioners.py` names a
preconditioner. The live kinds are:

| kind | mass | Laplacian |
|---|---|---|
| `'none'` | identity | identity |
| `'jacobi'` | `1/diag(E M_k E^T)`, probed | `1/diag(L_k)`, probed through `apply_laplacian_approx` |
| `'metric_lumping'` | `MetricLumpingMass` | `MetricLumpingLaplacian` |
| `'auto'` | `'metric_lumping'` | the atom when the bundle has it for this `(k, BC)`, otherwise `'none'` |

`'auto'` never substitutes. Nothing is built on demand: `build_preconditioners`
builds both atoms for every requested `(k, BC)` onto the bundle, a missing
mass atom raises at the solve, and a missing Laplacian atom makes the
default specs (`_materialize_default_saddle_preconditioner`,
`_materialize_default_scalar_hodge_preconditioner`) pick `'none'`, so an
unbuilt preconditioner runs unpreconditioned -- visibly slow -- instead of
on a different one. `apply_laplacian_preconditioner(kind='auto')`, the bare
apply, warns and applies the identity in that case. The Jacobi option is
built only on request, `build_preconditioners(jacobi=True)`: one-hot probes
of the applies themselves -- `O(n_k)` applies per `(k, BC)` -- store
`1/diag(E M_k E^T)`, `1/diag(L_k)` and, for the saddle solves, `1/diag` of
the approximate Schur operator `S_k + D B D^T` on the bundle
(`operators.mass_jacobi`, `laplacian_jacobi`, `schur_jacobi`).

A saddle solve is specified by `SaddlePointPreconditionerSpec(mass, schur,
coupled)` with `schur = SchurPreconditionerSpec(inner, outer)`: `mass` is the
lower block, `inner` stands in for `M_{k-1}^{-1}` inside the weak term, and
`outer` preconditions `L_k`. Production is `mass = inner = 'metric_lumping'`,
`outer = 'metric_lumping'`, `coupled = False`. `outer = 'jacobi'` applies the
probed Schur diagonal; `outer = 'none'` is the default of the spec object so
that a missing build fails visibly.

## 3. The Laplacian atom: `MetricLumpingLaplacian`

`mrx/metric_lumping_laplacian.py`. One instance per `(k, dirichlet)`, built by
`assemble_metric_lumping_laplacian_preconditioner(seq, ops)` and stored in
the dict `seq._metric_lumping_laplacian`. `MetricLumpingLaplacian.apply(x)`
is one jitted call on a flattened pytree payload.

Block Jacobi with two kinds of block:

**Bulk.** For each vector component `c` of `V^k`, the diagonal block of `L_k`
on the tensor-product rows is approximated by a three-term Kronecker sum

```
A_c = K_r ⊗ M_t ⊗ M_z + M_r ⊗ K_t ⊗ M_z + M_r ⊗ M_t ⊗ K_z
```

with unweighted 1D masses `M_a` and 1D stiffnesses `K_a` that carry the
metric weight averaged over the other two axes (`component_factors`). On a
derivative axis the stiffness is `Ktilde`, the 1D stiffness of the derivative
splines. The component factor `m_k / J` is pulled out as a diagonal
similarity `D^{1/2} A_c D^{1/2}` (`component_diagonal`). `A_c` is inverted exactly by fast diagonalisation:
three 1D generalised eigenproblems at build time
(`_simultaneous_diagonalize_pair`), then three small dense products and a
pointwise divide per apply (`_fd_apply_3d`). Cost per apply is
`O(N (n_r + n_t + n_z))`; storage is `O(n^2)` per axis.

**Core.** The polar rows, where the extraction fuses a ring of raw functions,
are not tensor-product functions. `core_rows` lists them; `probe_core_block`
forms `L_k` on those rows by one operator apply per row;
`_dense_symmetric_inverse` inverts the block on device by `eigh`, dropping
eigenvalues below `CORE_TOL` relative to the largest. Bulk and core are
applied independently; they are not coupled through a Schur complement.

**Natural boundary term.** Under a free condition at `r = 1` the weak block's
integration by parts leaves a surface term `alpha (e e^T) ⊗ M_t ⊗ M_z` with
`e` the one-hot derivative-spline trace, the shape of the first Kronecker
term. It merges into `K_r` as a rank-one update at no cost, on the components
whose radial axis is a derivative axis (`trace_components`: none at k=0, `r`
at k=1, `theta, zeta` at k=2, the single component at k=3). The coefficient
`alpha` is derived (`bc_entry="ibpd"`); it is multiplied by
`PRODUCTION_BC_SCALE = 3.0`, a measured balance point, not a derived factor.
`build_preconditioners(bc_scale=...)` overrides the constant; there is no
environment variable. Under Dirichlet the term is zero.

**Why the free 1D ends are the right natural conditions on curved maps.**
The free Hodge Laplacian at k=1 imposes `u.n = 0` and `curl u x n = 0` at
`r = 1`. On a flat map these collapse onto the 1D factors: `u.n = 0` pins
`u_r` on the face, so its tangential derivatives vanish there, and
`curl u x n = (d_r u_t - d_t u_r, d_r u_z - d_z u_r) = 0` reduces to
`d_r u_t = d_r u_z = 0` -- exactly the natural conditions of the free-end 1D
stiffnesses on the primal-axis components. The collapse survives curvature,
for two metric-independent reasons and one approximation:

1. *The curl half is metric-free.* In the logical covariant components the
   discretization stores, curl is the exterior derivative:
   `(du)_{rt} = d_r u_t - d_t u_r`, plain antisymmetrized partials with no
   metric factors and no Christoffel symbols, on any map.
2. *Positive weights do not change a natural condition.* The curved
   conditions carry metric factors, e.g.
   `g^rr g^tt J (d_r u_t - d_t u_r) = 0` on the face, but
   `w(1) u'(1) = 0 <=> u'(1) = 0` for `w > 0`: the weighted 1D operators
   impose the same condition. Only the *strength* of the penalized trace
   depends on the weight, which is what `_face_alpha` averages.
3. *The metric enters only through face orthogonality.* `u.n = 0` means
   `u^r = g^rr u_r + g^rt u_t + g^rz u_z = 0`, which is the logical
   statement `u_r = 0` -- the trace the rank-one term penalizes -- exactly
   when `g^rt = g^rz = 0` at the face. Exact on the cylinder and toroid; on
   W7-X it is the same orthogonal-metric approximation the bulk atom makes
   everywhere (`component_factors`), so the boundary adds nothing new.

At k=0 there is nothing to penalize at all: the codifferential of a 0-form
is zero, so `L_0 = D_0^T M_1 D_0` has no weak half, no integration by parts
and no surface term; the Neumann condition is genuinely natural for
free-end splines. At k=2 the same collapse runs with the roles swapped (the
penalized trace is `w x n` on the two derivative components, the primal
component keeps its free end); k=3 penalizes the full trace. See
`trace_components` in `mrx/metric_lumping_laplacian.py`.

One caveat: the collapse uses `u_r = 0` *pointwise* on the face, but the
free discrete problem enforces `u.n = 0` by a penalty, not by removing a
DOF. The residual `d_t u_r` contamination of the tangential conditions is
part of why the exact surface integral is not the kappa-best scale and the
measured `PRODUCTION_BC_SCALE` pushes toward the hard `u_r = 0` limit.

Requirement: `n_r >= p + 2`; a one-element radial mesh has no separable atom.

## 4. The mass preconditioner: `MetricLumpingMass`

Same file, same shape, simpler algebra: a mass is a single Kronecker product,
so the bulk inverse is three 1D dense solves with no fast diagonalisation
(`_kron_mass_model_1d`, `_apply_mass_payload`). The polar core is probed
with `apply_mass_matrix` and inverted densely; there is no pseudoinverse of
the extraction anywhere. Built by `assemble_mass_metric_lumping_preconditioner`
(inside `build_preconditioners`) and stored on `operators.mass_lumping`,
keyed `(k, dirichlet)`; nothing builds one on first use.

## 5. Building and invalidation

```python
seq.set_map(F)                  # installs the geometry, drops seq.operators
seq.build_preconditioners()     # a fresh bundle: both atoms for every (k, BC)
```

`seq.set_map_and_preconditioners(F)` is the two calls in one. `set_geometry`
drops the whole bundle because everything on it factorises the old metric;
there is no cache to invalidate, and the harmonic forms (also on the bundle)
are recomputed with `compute_nullspaces`.

The atom payloads are `eqx.Module` pytrees (`_LumpPayload`, `_MassPayload`)
built eagerly at construction, with one jitted apply per tree structure, so
a rebuild for a new geometry of the same discretisation reuses the compiled
program.

## 6. Cut-offs

All rank and structure cut-offs are multiples of the working-precision
epsilon (`mrx.eps`), so they scale with `MRX_DTYPE` (see
[precision.md](precision.md)).

| constant | value | gates |
|---|---|---|
| `CORE_TOL` | `eps(4096)` | eigenvalues of the probed core treated as zero |
| `PSEUDOINVERSE_TOL` | `eps(2^25)` | singular-value floor in `_symmetric_pseudoinverse` |
| `PROJECTOR_SVD_TOL` | `eps(2^19)` | rank cut of the extraction projector |
| `PROJECTOR_PLANE_TOL` | `eps(2^22)` | per-zeta-plane block equality |
| `BLOCK_DIAGONAL_TOL` | `eps(2^12)` | block-diagonality of the Gram matrix |

Endpoint nudges away from a clamped knot are `sqrt_eps() * h`.
`PROBE_BATCH_SIZE = 8` rows per `lax.map` batch when probing a diagonal.

## 7. Not in production

Research code (Chebyshev smoothers, the modal-radial atom, the coarse correction) lives on branch `greville-prod` under `mrx/experimental/`, not here:
`chebyshev.py` (polynomial acceleration), `metric_lumping_coarse.py` (the
truncated-Fourier coarse correction `CoarseCorrectedMetricLumping`), `modal_radial.py`.
Multigrid, HX auxiliary-space transfers, CP rank fits, dense outer-ring probes,
and the per-DoF Jacobi baselines are measured and not used; the
verdicts are in `docs/research/preconditioner_lessons.md`.

## 8. Measuring

- `scripts/poisson_study.py`: all eight `(k, BC)` Hodge-Laplacian solves on
  the toroid through the production `'auto'` dispatch, with the nullspace
  iteration counts, the true residuals and the solve iteration counts per
  resolution. `n=[8] p=3` is the smoke run.
- `test/test_poisson.py` pins the iteration counts of the production
  preconditioners on the session fixture: all eight `(k, BC)` Laplacians
  against manufactured solutions.

Rank alternatives by total time, not iterations: every arm costs the same per
iteration, so build cost decides. Iteration counts move by about 1% between
runs; only a two-digit percentage is a result.
