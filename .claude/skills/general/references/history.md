# Settled decisions and removed methods

What was decided and why, so it is not proposed again. Details and numbers are in `docs/research/`
(`preconditioners.md`, `solvers_precision.md`, `relaxation.md`, `performance.md`, `geometry_vacuum.md`,
`poincare_islands.md`), each with a "decided against / removed" section. Reopen a decision only when asked.

## Solvers and preconditioners

- **Metric-lumping atoms** (`mrx/metric_lumping.py`) are the base family, `PRODUCTION_BC_SCALE = 3.0` for
  every p. On top: the slab model for `M_1`, `M_2` (default since 2026-10-18, the atom loses the r-theta
  coupling on shaped sections) and the `S_2` multigrid (opt-in). Removed: tensor/raw_kron/Greville/rank>1
  atoms, k = 0 multigrid (lost to the atom in wall time), the slab model for `S_0` (dense in theta, n^4
  storage), per-mode factorisations (memory), Jacobi closed forms.
- **Hodge split for L_1, L_2** (stiffness PCGs). **L_3 and Leray by the split at k = 3** (2026-10-18),
  replacing saddle MINRES (116 against 565 iterations on li383 graded (48,64,24)).
- **Shifted solves** `(M + eps L)^-1` as three SPD solves (the error of the first cancels).
- **Harmonic forms** by a direct Hodge decomposition of closed seeds. The inverse-iteration polish was
  harmful and is gone.
- **Precision:** float64 is the default, float32 runs with double-single solutions and residuals for devices
  without float64. The former mixed mode and its residual dtype are gone. Matmul precision "highest" (H100
  float32 matmul is TF32 otherwise).

## Relaxation

- **Gauss-Newton direction** (2026-10-18): the Gauss-Newton part of the Hessian by MINRES with reorthogonalised
  Lanczos vectors, tol 0.1, maxiter 300, `M_1` solves inside the operator to 1e-6, parallel penalty 3 x strain.
  It replaced Newton-MR on the full Hessian with the `MassChebyshev` polynomial for `M_1^-1`, which stalled on
  graded meshes. Same steps as block Gauss-Newton under flexible GCR at 5-11 % less time.
- Earlier removals: trust-region Newton-CG (2x slower), CG-Steihaug (fails on W7-X's indefinite Hessian),
  forcing-term variants and a MINRES stall exit (worse than a fixed tol), dt floor and cap, `dt = dt*` instead of
  the cap 1 (later floors, opens islands on li383), Newton smoothing, fallbacks, Krylov recycling (cheaper
  iterations, no better steps).
- **L-BFGS removed** (2026-09-17). `--descent.method {newton,gradient}`, gradient = the smoothed force.
- **Helicity correction, implicit midpoint, auxiliary fields** were removed from the dev branch (on branch
  `paper` only).
- **Potential route** for the force (no pressure solve per step). Leray route once per chunk for `p`.
- **Velocity smoothing** `(M + mu L)^-1`, `mu = C h_r^2`: same floor, 8-16x fewer steps for the gradient
  method.
- **Initial field** from the Clebsch potential `B = dA'` in closed form from the file's series. The gridded
  and logical-profile routes are deleted (the logical-profile IC made W7-X chaotic).
- **Seed amplitude** from the closed-form energy criterion, not by hand.
- `--budget.floor-tol` well below the expected floor. Floored runs are not continued.
- Tried for kernels and dropped (PR #27): Pallas, Mosaic-GPU (WGMMA needs TF32) and CUDA FFI mass kernels
  (no win at 32^3, atomics). Records: `docs/research/performance.md`.

## Geometry and layout

- One evaluator (de Boor) for spline points and tables, so `d/dx` at r = 1 is correct (fixed 2026-09-28).
- Quadrature grid r-major, `p + 1` points in r and `p` in the angles (2026-10-18, 1.5-2x faster applies).
  Topology on the host once, geometry on the device.
- Maps always positive, angles right-handed (2026-09-30). `load(frame='ref')` removed.
- Half-period quadrature + parity views for stellarator symmetry (`symmetry=stellarator|field-period|none`).
- Graded polar spaces (2026-10-17): fewer theta cells near the axis, the map on uniform theta. Resolution
  `(n, 2n, 2n)` and cubes are not used for results. No p = 5. W7-X at p = 2.
- `FrameMap` beside `CylindricalMap` reads GVEC G-frame states.
- Library compacted 2026-09-28: subpackages `mrx.equilibria`, `mrx.relaxation` (`newton` was `hessian`,
  `config` was `relax_config`), `mrx.diagnostics`.

## Project landscape

- `main` = the public release, write-protected. Work goes on branches.
- Branch `paper`: paper tooling and paper-only options.
