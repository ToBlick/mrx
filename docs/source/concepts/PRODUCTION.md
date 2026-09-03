# What runs today

One page. Every identifier exists in `mrx/` or `scripts/`. The reasoning is in
[preconditioning.md](preconditioning.md); the measurements are in
`docs/research/`.

## Sequence and operators

`build_sequence(geometry, ns, p)` in `mrx/geometry.py` is the production
build (`geometry` is `toroid`, `cylinder`, `rot-ellipse`, or the path of a
GVEC export): `DeRhamSequence(ns, (p,)*3, p + 1, ("clamped", "periodic", "periodic"),
polar=True, betti_numbers=(1, 1, 0, 0))`, `set_map`, then
`build_preconditioners()`: the Jacobi and metric-lumped mass and Laplacian
preconditioners for `k = 0..3` and both boundary conditions, on a fresh
bundle. Mass matrices are never stored; every operator is a matrix-free apply
(`mass_core_apply`, `apply_incidence_matrix`, `apply_derivative_matrix`,
`apply_stiffness`, `apply_laplacian`).

## Preconditioners

| solve | preconditioner | code |
|---|---|---|
| mass, all k | `kind='metric_lumping'`: separable Kronecker bulk, polar core probed and inverted densely | `MetricLumpingMass` |
| Laplacian, k = 0..3, free and Dirichlet | `kind='metric_lumping'`: per-component Kronecker-sum atom by fast diagonalisation, dense polar core, rank-one natural-BC term | `MetricLumpingLaplacian` |

Kinds: `none`, `jacobi`, `metric_lumping`, `auto`. `auto` resolves to
`metric_lumping` for the mass, always; for a Laplacian it uses the atom when
`build_preconditioners` has built it for that `(k, BC)` and `none` otherwise.
`jacobi` is the probed diagonal, built only by
`build_preconditioners(jacobi=True)`; it is never substituted.

Saddle solves (k >= 1): `mass = inner = outer = 'metric_lumping'`,
`coupled = False`.

`PRODUCTION_BC_SCALE = 3.0` in `mrx/metric_lumping_laplacian.py` multiplies
the natural-BC coefficient; `build_preconditioners(bc_scale=...)` is the only
override (no environment variable). `bc_entry="ibpd"` is the default and the
production configuration; pass nothing.

Not in production: multigrid, Chebyshev or Richardson acceleration, CP fits,
HX transfers, dense outer-ring probes, the Fourier coarse correction
(`mrx/experimental/` on branch `greville-prod`).

## Solvers

- k=0 Laplacian and every mass: `solve_singular_cg`, harmonic mode deflated
  on the unshifted problem.
- k >= 1 Laplacian: `solve_saddle_point_minres` on `[[K_k, D], [D^T, -M_{k-1}]]`.
- Shifted problems do not deflate; free k >= 1 adds the `1/eps` harmonic
  coarse correction when the vector exists.
- `M_k + eps L_k` (velocity smoothing, resistive step): two SPD CG solves
  through the split identity `(M_k + eps S_k)^-1 - eps D_{k-1} (M_{k-1} +
  eps S_{k-1})^-1 D_{k-1}^T`; no saddle system. Preconditioner: the
  shifted-stiffness atom, `(M^ + eps S^)^-1` from the Laplacian atom's
  strong-half terms (`MetricLumpingLaplacian.shifted_stiffness_apply`).
- No Krylov solve inside a Krylov solve: the weak term uses the mass
  preconditioner as `M^{-1}` (`apply_laplacian_approx`).
- Harmonic forms: `compute_nullspaces` (direct, `b2 = 0`) or
  `compute_nullspaces_iterative`.

## Tolerances and precision

- `MRX_DTYPE` selects `float64` (default) or `float32`;
  `jax_default_matmul_precision` is `"highest"`.
- Solver tolerance `tol=None` is `mrx.sqrt_eps()` (`1.5e-8` in float64)
  everywhere (`DeRhamSequence(tol=...)`); `scripts/poisson_study.py --tol`
  defaults to `1e-9`, the tolerance of the archived convergence numbers.
- Cut-offs are multiples of `mrx.eps`: `CORE_TOL`, `PSEUDOINVERSE_TOL`,
  `PROJECTOR_SVD_TOL`, `PROJECTOR_PLANE_TOL`, `BLOCK_DIAGONAL_TOL`.
- `maxiter = 10_000` per solve.

## Quadrature

`q = p + 1` Gauss points per knot span, passed by every entry point
(`build_sequence`, `build_gvec_map`, `scripts/poisson_study.py`,
`test/conftest.py`).

## Maps

Analytic maps are callables. A GVEC state or VMEC wout becomes a polar
spline map on the sequence's own 0-form space with the coefficients built
from the series coefficients, mode by mode, no evaluation grid
(`series_spline_dofs`, `build_gvec_map` in `mrx/gvec.py`); an analytic map
is fitted by `seq.interpolate(f, 0)` on the same space
(`greville_interpolate_map` in `mrx/geometry.py`). No
reference mass matrix.
Geometry lives on `SequenceGeometry` as `metric_jkl`, `metric_inv_jkl` and
`jacobian_j`, built once per map; every mass weight is an elementwise product
of those, memoised per degree in the element layout. `DF` is not stored --
`load(frame='phys')` recomputes it at load time.

## Relaxation

`scripts/relax.py --geometry <GVEC state or VMEC wout>`: `--ns 8,16,16`, `--p 2`,
`--maxiter 2000`, `--precision float32`, `--ic clebsch`, `--method lbfgs --history 1`,
`--history 3`, `--dt-mode linesearch`, `--cfl 0.5`; stops when the mean of
the relative force residual over `--floor-steps 100` steps is below
`--floor-tol 1e-3`; one method per run; output `relax.json` and `B.h5`.
Each step is operator-split (Lie): ideal transport, then implicit resistive
diffusion. Details in [relaxation.md](relaxation.md).

## Traps

- `frame='ref'` in `load` and `interpolate` takes `g ω / J`, not the primal
  components `ω`. Push forward and use `frame='phys'`.
- Nothing is built on first use. `set_map` drops the operator bundle;
  rebuild with `build_preconditioners` (or `set_map_and_preconditioners`)
  and recompute the harmonic forms.
- Run something real after every merge. A renamed function whose caller lives
  on the other branch merges green and dies at setup.
