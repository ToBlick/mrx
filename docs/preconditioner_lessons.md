# Preconditioner lessons — settled findings and dead ends (as of 2026-07-07)

Distilled from session memory so it travels with the repo. Each item is settled
empirically; don't re-derive or retry without new evidence. Evidence pointers are
to committed scripts / gitignored `outputs/` on the cluster.

## Run conventions

- **Always run preconditioner sweeps at p=3** (user default; lower p hides the
  high-p separability failures, higher p is not the production target).
- **Never validate a preconditioner by a κ / spectral check alone** — the
  radial_dense atom passed a dense κ-check and then stalled at ~1e-6 in real
  free-BC CG on curved geometries. Gate on CG-to-1e-10 solves, both BCs.

## Laplacian / stiffness path

- **rank>1 is DEAD for the tensor-Hodge Laplacian atoms** (k=0 `L_0` etc.):
  Lynch fast-diagonalization is exact only at rank 1; at rank>1 a single
  isolated outlier eigenvalue of `smoother∘L_0` appears (23–124× gap to an
  otherwise-unchanged active spectrum [~0.53, 1.65]) and collapses λmin → κ 1e5,
  Chebyshev degree explosion, OOM. Refuted fixes (do not repeat): positivity/NTF,
  bulk-mode deflation/pseudo-inverse, "CP-ALS is broken". The clean fix, if ever
  needed: deflate the ONE smallest eigenpair of the composite and tune Chebyshev
  on [λ₂, λmax]. Production stays rank-1. (Mass path is healthy and *improves*
  with rank — the defect is stiffness/Hodge-specific.)
- **radial_dense (rank-2 radial split) k=0 atom breaks on CURVATURE, not free BC**:
  exact on cylinder even free-BC; stalls on toroid free-BC, NaNs on W7-X. The
  rank-1 FD atom is the robust default.
- **k=0 channel weights are clean power laws on ALL geometries**
  (`α_rr ~ r`, `α_θθ ~ 1/r`, `α_ζζ ~ r`; `scripts/debug/laplacian_radial_profiles.py`,
  doc §D3 of `preconditioner_plan.md`): two different radial functions
  multiplying a mass matrix cannot share one pencil → the radial direction is
  forced dense for any *exact* FD atom; every cheap 2-of-3 "radial-pencil"
  shortcut was tested and is WORSE than scalar grev-const (the dropped θθ
  off-diagonal is first-order precisely because 1/r vs r are maximally
  different). Angular spread decides strategy: cylinder 0%, toroid ~24%,
  W7-X ~60% (angular is first-order on W7-X; no radial-only method suffices
  there).
- **Metric-weight separability rule (analytic):** 1/r-type weights are rank-1
  (easy); r·R and r·R⁻¹-type weights are θ-coupled (hard). The radial factor is
  always low-rank; θ-ζ is the coupled plane. Predicts which blocks need
  cross-section treatment before running anything.
- **k=3 sideways transfer (V3→V0) is not viable** — no cheap interpolation
  between those spaces; recurse through k=2 instead.

## Mass path

- **Block-Chebyshev polish on the tensor MASS regressed** and the default was
  reverted to `bcheb=0` (validated 2026-06-25): each polish step costs ≈6 full
  mass matvecs per apply, so bcheb=3 lost ~10× on the wall despite ~10× fewer
  iterations; `ischur` (inner-block coupling) is never worth it. With bcheb=0
  the tensor mass beats Jacobi on wall even on W7-X.
- **Lumped-L/U block-SGS for k=1/2 mass coupling regressed** (2026-07-07, see
  `mass_coupling_preconditioner_handoff.md` for the full table): ~0.1–0.17
  lump error in the off-diagonal blocks makes SGS sweeps ADD error (k=1:
  291–350 it vs 75–80 baseline). Next lever is the support-integrated diagonal
  lump, not coupling.

## Geometry / evaluation traps

- **Spline-map DF is singular at ρ=1** (det(DF)=0 at the outer knot):
  pushforward/metric evaluation AT ρ=1 gives inf; quad-point-only checks never
  see it. Evaluate at 1−ε (the greville scripts clip to `1e-7`).
- **Flat quad-point ordering is (θ, r, ζ), theta-major** — meshgrid
  `indexing='xy'` swap; see the NOTE/TODO comments at
  `mrx/operators.py:_reshape_quadrature_scalar_field` and `mrx/quadrature.py`.
  Reshape to `(ny, nx, nz)` then transpose `(1, 0, 2)` for (r, θ, ζ) fields.
- **W7-X (cluster only):** build via greville+RGI bridge, nfp=5; set
  `XLA_PYTHON_CLIENT_PREALLOCATE=false W7X_MAP_BATCH=128` or the map fit OOMs;
  Cartesian B components are ζ-quasiperiodic (de-rotate per field period before
  interpolating — `docs/w7x_vacuum_bfield_handoff.md`). Dominant metric
  coupling for preconditioning is ρ-θ (~0.49 normalized), not θ-ζ.

## Active threads (2026-07-07)

- k=0 Laplacian geometric MG: `docs/laplacian_mg_k0_plan.md` (phase-0 done,
  transfers near the axis are the blocker).
- k=1/2 mass coupling: `docs/mass_coupling_preconditioner_handoff.md`
  (SGS dead; support-integrated diagonal lump next).
