# What runs today

One page. Every identifier exists in `mrx/` or `scripts/`. The reasoning is in
[preconditioning.md](preconditioning.md); the measurements are in
`docs/research/`.

## Sequence and operators

`build_sequence(geometry, ns, p)` in `mrx/geometries.py` is the production
build: `DeRhamSequence(ns, (p,)*3, p + 1, ("clamped", "periodic", "periodic"),
polar=True, betti_numbers=(1, 1, 0, 0))`, `evaluate_1d()`, `set_map`, then
`assemble_incidence_operators`, `assemble_mass_jacobi_preconditioner`,
`assemble_metric_lumping_laplacian_preconditioner` for `k = 0..3` and both
boundary conditions, `warm_mass_preconditioner_cache`, `set_operators`.
Mass matrices are never stored; every operator is a matrix-free apply
(`mass_core_apply`, `apply_incidence_matrix`, `apply_derivative_matrix`,
`apply_stiffness`, `apply_hodge_laplacian`).

## Preconditioners

| solve | preconditioner | code |
|---|---|---|
| mass, all k | `kind='metric_lumping'`: separable Kronecker bulk, polar core probed and inverted densely | `MetricLumpingMass` |
| Laplacian, k = 0..3, free and Dirichlet | `kind='metric_lumping'`: per-component Kronecker-sum atom by fast diagonalisation, dense polar core, rank-one natural-BC term | `MetricLumpingLaplacian` |

Kinds: `none`, `jacobi`, `metric_lumping`, `auto`. `auto` resolves to
`metric_lumping` for the mass, always; for a Laplacian it uses the atom when
`assemble_metric_lumping_laplacian_preconditioner` has built it for that
`(k, BC)` and `none` otherwise. It never substitutes `jacobi`.

Saddle solves (k >= 1): `mass = inner = outer = 'metric_lumping'`,
`coupled = False`. `outer = 'jacobi'` is the comparison baseline.

`PRODUCTION_BC_SCALE = 3.0` in `mrx/metric_lumping_laplacian.py` multiplies
the natural-BC coefficient; `MRX_BJ_BC_SCALE` overrides it; an explicit
`bc_scale` argument beats both. `bc_entry="ibpd"`, `ktilde_mode="honest"`,
`lumped="diag"` are the defaults and the production configuration; pass
nothing.

Not in production: multigrid, Chebyshev or Richardson acceleration, CP fits,
HX transfers, `outer_rings`, the Fourier coarse correction
(`mrx/experimental/`).

## Solvers

- k=0 Laplacian and every mass: `solve_singular_cg`, harmonic mode deflated
  on the unshifted problem.
- k >= 1 Laplacian: `solve_saddle_point_minres` on `[[K_k, D], [D^T, -M_{k-1}]]`.
- Shifted problems do not deflate; free k >= 1 adds the `1/eps` harmonic
  coarse correction when the vector exists.
- No Krylov solve inside a Krylov solve: the weak term uses the mass
  preconditioner as `M^{-1}` (`apply_hodge_laplacian_approx`).
- Harmonic forms: `compute_nullspaces` (direct, `b2 = 0`) or
  `compute_nullspaces_iterative`.

## Tolerances and precision

- `MRX_DTYPE` selects `float64` (default) or `float32`;
  `jax_default_matmul_precision` is `"highest"`.
- Solver tolerance `tol=None` is `mrx.sqrt_eps()` (`1.5e-8` in float64)
  everywhere: `DeRhamSequence(tol=...)`, `NumericsConfig.solver_tol`.
  `conf/config_poisson_test.yaml` pins `1e-9` for the archived numbers.
- Cut-offs are multiples of `mrx.eps`: `CORE_TOL`, `PSEUDOINVERSE_TOL`,
  `PROJECTOR_SVD_TOL`, `PROJECTOR_PLANE_TOL`, `BLOCK_DIAGONAL_TOL`.
- `maxiter = 10_000` per solve.

## Quadrature

`q = p + 1` Gauss points per knot span, passed by every entry point
(`build_sequence`, `build_gvec_map`, the Poisson scripts with
`quad_order_offset: 0`, `test/conftest.py`).

## Maps

Analytic maps are callables. Data maps are fitted by `seq.interpolate(f, 0)`
on a polar map sequence (`build_gvec_map`, `build_w7x_map` in `mrx/gvec.py`;
`greville_interpolate_map` in `mrx/geometry.py`). No reference mass matrix.
Geometry lives on `SequenceGeometry` as `DF_jkl` and `jacobian_j` only; the
metric, its inverse and every mass weight are formed from those on demand.

## Relaxation

`scripts/relax.py`: `--method cg`, `--dt-mode linesearch`, `--ic logical`,
`--floor-tol 10*eps`, `--floor-window 100`; one method per run; output
`relax.json` and `B.h5`. Details in [relaxation.md](relaxation.md).

## Traps

- `frame='ref'` in `load` and `interpolate` takes `g ω / J`, not the primal
  components `ω`. Push forward and use `frame='phys'`.
- Any new traced entry point that solves must call
  `warm_mass_preconditioner_cache` first; the mass preconditioner build is
  host-side.
- `set_map` drops the Laplacian atoms; rebuild with `build_preconditioners`
  or use `set_map_and_preconditioners`.
- Run something real after every merge. A renamed function whose caller lives
  on the other branch merges green and dies at setup.
