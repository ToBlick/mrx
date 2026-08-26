> **Status:** current
> **Read this for:** settled findings, dead ends and traps of the preconditioner campaign, in priority order
> **Do not read for:** the construction; that is preconditioner_technical_note_source.md

# Preconditioner lessons — settled findings and dead ends (as of 2026-08-25)

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
  `preconditioner_lessons.md` for the full table): ~0.1–0.17
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

## The 2026-08 arc: what shipped and what died

This section replaces sixteen working-log handoffs from 2026-08-13 to 2026-08-25
(the pivot, the natural-BC sweep, the audit series, the surgery-Schur plan). Their
conclusions are here; the narrative of how they were reached is not, deliberately.

- **There is now exactly ONE preconditioner besides `none` and `jacobi`:
  `metric_lumping`.** Small dense core at the polar axis, Kronecker tail elsewhere
  (`M ~ Lam (A_r x A_t x A_z) Lam`, with the core the part that differs). It is the
  production mass preconditioner AND, as `schur.outer`, the production k>=1
  Laplacian preconditioner. Named `block` / `block_jacobi` until 2026-08-25.
- **`raw_kron` was deleted 2026-08-25**, after the A/B that could only be run once
  (`result_2026-08-25_schur_probe_ab.md`): six converged cells, five favouring the
  atom by 2.4-16.6% with the largest gain on W7-X, one at +0.6% inside a MEASURED
  0.1-0.3% run-to-run noise floor. The old `tensor` (CP/ALS) stack died earlier the
  same week. Do not reintroduce a second mass kind to "compare against" without
  reading that note first.
- **Never soft-substitute a preconditioner.** `_materialize_default_saddle_preconditioner`
  resolves to the atom when assembled and `'none'` otherwise — never a probed jacobi
  diagonal. Probe-building a fallback is how the relaxation loop ran its innermost
  solve on a diagonal for months with nobody noticing: a substituted preconditioner
  does not fail, it just gets slower, which is invisible. Running unpreconditioned is
  visible. "You get what you built."
- **The boundary penalty is settled: ship only `alpha_k = <m_k sqrt(g^rr)> / <m_k/J> / h`
  with `s = 3`.** The `product` / `halves` / `matrixwise` / `product_bare_h`
  conventions are gone. A0 @ 0.10 and A5 @ 2.83 are equivalent (+0.8%); rank by TOTAL
  iterations, not worst case. "Prefer 0.05 at p>=5" was wrong.
- **The natural-BC coefficient question is closed, and the answer is not the obvious
  one:** alpha is the best NORM fit to L's boundary block but NOT the best
  PRECONDITIONER of it. The gap is within-ring angular/cross-component coupling the
  atom drops. The DtN hypothesis was refuted. A local ring-block match predicts the
  scale with no solve. Deliverable = the scalar term at the corrected scale, rank-1
  and free, capturing 66-74% of the available gain.
- **Geometric MG for the k=0 Laplacian is a research branch, not production.** The
  production route is the metric-lumped atom swapped into the existing thin-core
  preconditioner. Deflation, eps-shifts, truncation and fdhel-v1 were all refuted;
  a raw-atom Schur rebuild floors CG (indefinite).
- **2-D ring atoms work INNER only.** Outer rings need the dense probe — the DtN
  behaviour there is nonlocal.
- **Ranking rule: TOTAL TIME, not iteration count**, and only a TWO-DIGIT percent
  worsening is worth investigating at all. Run-to-run variation here is 0.1-0.3%.

## Active threads (2026-08-25)

- **Even-p histopolation identity** (`handoff_2026-08-25_histopolation.md`): at odd p
  the projectors are exact at every k and both BCs (~5e-16); at even p, k>=1 is not a
  projector (7e-2 to 1.3e-1) and the cause is UNKNOWN. Two mechanisms refuted with
  evidence — the extraction (k=3 has `E E^T = I` to 0.000 and fails anyway) and
  quadrature exactness (splitting spans at knots improved accuracy and made the
  identity slightly worse). Next experiment is cheap and needs no GPU: compare
  `moments(D_i)` against `H[:, i]` column by column in 1-D at p=2.
- **k=1/2 mass coupling**: SGS is dead; the support-integrated diagonal lump is the
  next lever, not coupling.
- **Deferred follow-ups from the raw_kron deletion**: the `outer='none'` branch still
  builds a Schur apply it discards; `verify_block_jacobi.py` keeps its old name
  because 22 scripts import `build_sequence` from it.
