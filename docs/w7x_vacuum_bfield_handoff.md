# Vacuum B-field: projection & vacuum-field handoff

Script: `scripts/debug/w7x_vacuum_bfield_project.py` (notebook-style `# %%` cells,
argparse-driven, runs on any conforming h5).
Verification script: `scripts/debug/quasr_covcontra_verify.py` (co/contra diagnostics).

## Goal

Load a vacuum equilibrium sampled on a logical tensor grid, build the stellarator
map on MRX splines, and **project B onto the 2-form space V2**, reporting the error
at the stored nodes. Second objective: compute the vacuum field directly as a
harmonic 2-form.

## Input h5 layout (generic)

- Logical tensor grid over one field period, C-order, ζ fastest.
- `eval_points` (N,3): logical (ρ,θ,ζ) ∈ [0,1); `R`,`Z` (N,); `B` (N,3) Cartesian [T].
- attrs: `n_rho`, `n_theta`, `n_zeta`, `nfp` (or `precomputed_nr/ntheta/nzeta` in the
  simsopt files — the loader accepts either).
- Tested: `data/W7X-vacuum.h5` (32³, nfp=5), `data/quasr_0009983.h5` (50³, nfp=2,
  also carries native `B_cov`/`B_contra` — see §warnings), and the simsopt files
  `data/quasrXXXX_simsopt_B.h5` (8×16×8) — but see the ⚠ 0044970 section below.
- Run: `python scripts/debug/w7x_vacuum_bfield_project.py --h5 <file> [--nfp N] [--stride S]`

## The recipe (validated: quasr mean rel ~1.2%, seam-free)

1. **Interpolatory tensor B-splines for R, Z** — per-axis square collocation
   (`_solve_tensor_collocation_axis`), factorized. `n_basis = n_data` per axis.
2. **Build the map with the orientation that gives det(DF) > 0.**
   `F = (R·cos a, sign·R·sin a, Z)`, `a = 2π·ζ/nfp`. Compute `det(DF)` with
   `sign=+1`; if negative, use `sign=−1` (flipping the toroidal sign is a
   reflection → flips `det` sign). **This is the ONLY orientation requirement** —
   the projection below is self-consistent in DF, so absolute frame handedness is
   irrelevant, but `J<0` makes M2 indefinite → CG returns **NaN**. Auto-detected;
   W7-X lands on `sign=−1` (MRX's stock `stellarator_map` convention), quasr on
   `sign=+1`. **Do not hardcode `−sin`.**
3. **Covariant proxy `ω = DFᵀ B` on the data grid** — built from raw Cartesian B
   and our analytic (factorized) DF. This is the reference 2-form pull that
   `load(frame='ref')` pairs against, and crucially it is **periodic** even though
   Cartesian B is not: `(R·DF₀)ᵀ(R·B₀) = DF₀ᵀB₀`, so the ζ-rotation cancels.
   Interpolating `ω` is therefore **seam-free**.
4. **Project onto V2**: `mrx.io.load_grid_field(ω, seq, 2, frame='ref')` → `M2⁻¹`.

### Why DFᵀB, not the file's B_cov (the metric-factor saga)

- `load(frame='ref')` is a metric **dual**: for k=2 you feed the *covariant*
  components; the recovered DOFs are `J·B_contra` (MRX's Piola k=2 pushforward is
  `DF·ω/J`, so the 2-form ref proxy = `J·DF⁻¹B`). Co/contra roles **swap** vs naive
  intuition (k=1 feeds contravariant).
- The file's native `B_cov`/`B_contra` are in **GVEC coordinate normalization**
  (θ∈[0,2π], ζ∈[0,2π/nfp] radians) while MRX uses [0,1] — a diagonal factor
  `S_cov = diag(1, 2π, 2π/nfp)`. The ζ factor (`2π/nfp`) is clean and confirmed.
  **But** `B_cov` (geometric `B·eᵢ`) and `B_contra` (GVEC-native `B_contra_t/z`)
  use *inconsistent* θ conventions (cov·contra ≠ 1 on θ; ζ is fine), so no single
  diagonal makes both work — feeding the file's fields is a trap.
- Building `ω = DFᵀB` ourselves sidesteps all of it: exact, needs only Cartesian B
  + our DF, and self-consistent (pushforward uses the same DF). The file's co/contra
  are useful only as diagnostics (they exposed the `J<0` bug and confirmed
  `DFᵀB ≈ S_cov·B_cov` to 4.3%).

### `load_grid_field` (new library helper, `mrx/io.py`)

Factorized, interpolatory-spline analogue of `seq.load(callable)` for grid-sampled
data. Fits the interpolatory spline (per-axis solve), evaluates it at `seq`'s quad
grid via `_tp_evaluate` (three 1D contractions — `O(N_q(n1+n2+n3))`, no pointwise
`lax.map` / per-quad basis sweep), applies the k-form frame pullback + weight
(mirrors `projectors.load`; `frame='phys'` reuses stored `seq.DF_jkl`, no jacfwd),
integrates → **dual load vector**. Validated bit-identical to `seq.load(callable)`.
Use it (not `project_sampled_field`, which is linear-RGI with an O(h²) bias).

## Performance notes

- The **projection** (fit → `load_grid_field` → `M2⁻¹`) is fully factorized/cheap.
- The **error diagnostic** (evaluating the reconstructed 2-form at all N data nodes
  to compute mean/max rel) is inherently `O(N_data)`. At 50³ = 125k nodes this
  dominates → **use `--stride 2` on quasr** (per-axis subsample keeps the tensor
  grid; the projection quality is set by `--ns`, not the eval stride).
- `fit_seq` is built with quad order `FIT_Q=1` (its quadrature is never used — only
  its 1D bases / `e0`); a full `2·FIT_P` rule at 50³ would allocate a ~2.7e7-point
  3D quad mesh in the constructor.

## Geometry storage refactor (mrx/geometry.py)

`SequenceGeometry` now stores **DF as the primitive** (`DF_jkl` + `metric_inv_jkl`
+ `jacobian_j`); `metric_jkl` is a contraction property. `seq.DF_jkl` exposed;
`projectors.load` (frame='phys') and `io.project_sampled_field` reuse it instead of
recomputing `jax.jacfwd` over the quad grid. Kept resident (~19·N_q) on purpose —
the matrix-free matvec builds (`build_matrixfree_mass_apply`) read the metric on the
hot path, so recompute-from-coeffs (O(N)) is a bad trade. See memory
`geometry-df-primitive-storage`. Open: `quad.x` (the 3D N_q×3 mesh) is build-time
only and could be made lazy from the 1D points.

## Vacuum field = harmonic 2-form (unchanged, `--no-harmonic` to skip)

k=2 DBC harmonic 2-form (dim 1 on betti=(1,1,0,0)), seeded by logical dζ=(0,0,1)→V2
(`M2⁻¹` supplies the metric; geometry-robust, no 1/R assumption). Needs
`assemble_incidence_operators` + `assemble_schur_jacobi_preconditioner(ks=(2,),
dirichlet_variants=(True,))`. On W7-X the IC was near-harmonic (2 inverse-iteration
steps), matching stored B to ~0.7% — better than projecting sampled B, because it
lives in the logical frame and never samples the Cartesian ζ=0 seam.

## Single-file interface + harmonic-vs-reference comparison (2026-07-02)

The script now takes **one** self-contained h5 via `--h5`; geometry (R,Z,nfp) **and**
the field B are read from that single file. It handles both formats transparently:
old GVEC (`quasr_XXXX.h5`, 50³, attrs `n_rho/n_theta/n_zeta`, `B` Cartesian) and new
simsopt (`quasrXXXX_simsopt_B.h5`, 8×16×8, attrs `precomputed_nr/ntheta/nzeta`).
Because R,Z and B come from the same file they are self-consistent by construction —
**no cross-file frame reconciliation**, which is what previously masked the bug below.

Both harmonic forms are solved and compared to the file's B (best-fit-scaled,
xyz frame): the **2-form (k=2 DBC)** and the **1-form (k=1 no-DBC)**. `--harmonic-tol`
(default 1e-6) sets `||L v||`. Per-`[ref]`-line diagnostics (pure numpy):
`|B|·R` CoV + `corr(|B|,1/R)` (magnitude/1-R structure, catches scrambled rows) and
**cylindrical `frac toroidal/radial/vertical`** in the map frame `Φ=sign·2π·ζ/nfp`.
`err_line` now prints p50/75/90/95/99/max percentiles + the (ρ,θ,ζ) of max error, and
stashes every comparison's per-point arrays in `B_ERRORS`.

## ⚠ quasr0044970_simsopt_B.h5 is mis-generated — B rotated one field period off R,Z

Diagnosed 2026-07-02 (all on CPU, no solve). Symptoms and proof:
- Harmonics match the file's own field to ~1.3% for **0009983** but **88%** for
  **0044970** (both k=1 and k=2, so it's the data not the solver).
- `[RZ]` passes (~1% of span) for both — R,Z are **rotation-invariant**, so they
  cannot see a z-rotation of B. Do **not** trust R,Z agreement as a frame check.
- `|B|·R` CoV and `corr(|B|,1/R)` match the GVEC field for both → B is a legit
  `|B|∝1/R` vacuum magnitude, **not** row-scrambled.
- **Cylindrical decomposition is the tell**: everything is ~99% toroidal *except*
  0044970 simsopt, which is **85% radial / 49% toroidal**. The offset that restores
  it to ~98% toroidal is exactly `δ = −2π/nfp = −120°` = **one field period**.

Root cause: the Biot–Savart B was evaluated at points whose toroidal angle is offset
one field period from the ζ→Φ convention of the R,Z stored beside it — i.e. **B and
(R,Z) refer to different physical points within the file**. A rigid z-rotation
commutes with the cylindrical frame map, so **no comparison frame (Cartesian,
cylindrical, or flux/logical `DF⁻¹`) can undo it** — the fix must be upstream.

**Fix (upstream, not in this script):** regenerate the simsopt B by evaluating
Biot–Savart at the `(x,y,z)` reconstructed from *this file's own* R,Z and ζ (same
`Φ=2π·ζ/nfp`), so B and (R,Z) are the same point. Acceptance check: `frac radial`
must be tiny (`frac toroidal ≈ 0.99`) at δ=0 — regardless of what `[RZ]` says.

Note **0009983 is a separate, benign case**: 99% toroidal at δ=0, just a global
toroidal-sign/`flip_zeta` convention (absorbed by best-fit scale); its ~22% is
residual poloidal GVEC-vs-Biot–Savart difference, plausibly real, not this bug.

## Open items

1. Full-resolution (`--stride 1`) confirmation of the quasr ~1% once affordable
   (or factorize the diagnostic pushforward: reuse the analytic DF/J instead of
   `Pushforward`'s jacfwd, though the polar 2-form eval stays pointwise).
2. Upstream: replace `_toroidal_vacuum_field` in `nullspace.py:_initial_guesses`
   (physical 1/R e_ζ, toroid-only) with the logical-(0,0,1) construction.
3. If it recurs, sort out GVEC's native `B_contra` θ-normalization for a k=1/V1
   projection (V2 doesn't need it).

## Memory

`geometry-df-primitive-storage`, `interpolatory-spline-vs-linear-rgi`,
`logical-dzeta-vacuum-ic`, `w7x-cartesian-bfield-zeta-quasiperiodic`,
`bfield-frame-via-flux-surface-tangency`, `spline-map-DF-singular-at-r1`.
