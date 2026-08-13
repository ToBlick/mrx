# Handoff — 2026-08-13: k=0 MG preconditioner moves to GPU / cluster

Single entry point for the k=0 Laplacian multigrid line. Supersedes and
folds in the four stacked addenda of the retired
`handoff_2026-07-09_polar_c2_mg.md` (removed; all technical detail lives on in
`docs/laplacian_mg_k0_plan.md` (running log + theory), `docs/dev/k0_massprecond_surgery_findings.md`
(the dense-κ studies), and `docs/polar_pole_regularity.md` (C⁰/C¹/C² theory + convergence)).

## CLUSTER RESULTS + DECISIONS (2026-08-13, same day) — READ THIS FIRST

The full experiment matrix below RAN (all CSVs under
`outputs/laplacian_mg_k0/2026-08-13/`), plus a same-day pivot to a
single-level program. Verdicts:

**PRODUCTION DECISION: no MG. Swap the k=0 tensor-hodge atom weights
fd→fdbund instead.** Single-level fdbund + the EXISTING production thin-core
Schur factors beats the production baseline (`sl-*` rows, dir `07-39-14` +
the 4-geometry validation of jobs 16131589–98, all thin-core `--schur fd`,
`sl-fd` == baseline exactly on every cell). VALIDATED iteration counts
(baseline → sl-fdbund, dbc | free at the largest size per geometry):
- W7-X (16,32,32):  80 → 62 | 117 → 85   (wall-clock 1.29× | 1.39×)
- toroid (24,48,24): 41 → 29 | 59 → 36   (1.45× | 1.64×)
- cerfon (16,32,16): 47 → 34 | 62 → 39   (1.41× | 1.61×)
- rot-ellipse (16,32,16): 67 → 68 | 84 → 73  (tie | 1.17×; worst cell
  12³ dbc −10%, the one regression anywhere)
MG's best (2-level fat-core+anchor fdbund) wins only free-BC wall-clock at
1.05–1.4× while losing dbc everywhere, at ~2k lines of production
complexity — shelved as a research branch.

**WIRED + VERIFIED (same day): production changes are in.**
`_assemble_k0_greville_bulk_factors` now defaults to the fdbund atom
(bundled `<g^{aa}J>` per-axis weighted 1D stiffnesses via
`_k0_bundled_axis_profiles`, D = 1, alpha = 1); `MRX_K0_ATOM=fd` reverts to
the collocated atom. `_symmetric_pseudoinverse` is positive-part
(negative/sub-cutoff eigenvalues dropped instead of inverted by magnitude)
— required for the swap, not just hygiene: the production core Schur is
REBUILT through the bulk atom (`ass − C0ᵀ·B_bulk·C0`).
**Schur-probe gotcha found by verification:** probing that rebuild with the
fdbund bulk itself FLOORS W7-X CG at ~1e-2 both BCs (fdbund's flat theta
profile overestimates `A_bb^{-1}` near the axis → indefinite rebuilt core);
production therefore probes the Schur with the COLLOCATED fd atom and
applies fdbund at runtime (the validated configuration). Verified
end-to-end (job 16136365): W7-X (12,24,24) baseline 54 dbc / 79 free at
1e-11 (was 64/100), toroid (16,32,16) 24/30 (was 32/45), and
`MRX_K0_ATOM=fd` reproduces the old numbers exactly.
**k=1 P_A bundled (same recipe, one level up):** the curl-curl channel fits
in `_assemble_k1_curlcurl_regular_tensor_model` now default to
deterministic mean-field rank-1 bundled factors
(`_bundled_rank1_mass_factors`: cross-axis quad-mean profiles of the
`beta_aa = g_aa/J` channels, xi_1 radial cutoff, 1/mean² normalization —
exact on rank-1 weights) instead of CP-ALS; `MRX_K1_ATOM=cp` reverts.

**k=1 three-arm W7-X benchmark result (jobs 16139235–41 + toroid controls
16139355/56; logs `outputs/k1_pa_compare/2026-08-13/`):**
- **W7-X: the saddle P_A+P_B STALLS in every arm** — pre-campaign,
  k0-swap-only, full-bundled are indistinguishable (floors 1e-4..1e-7,
  4/4 fails, both BCs, both sizes; jacobi converges dbc in 949/1509 it).
  So the k=1 W7-X wall is NOT the k=0 solves in P_B/Π and NOT rank-1
  channel-weight quality — consistent with the 2026-06 FFT diagnosis
  (β_θθ/β_rr need 4–7 angular modes; no rank-1 atom reaches it). The
  stall-not-slow signature points at preconditioner INCONSISTENCY.
  **Probe result (jobs 16141984/85, eigencount before the PSD chop): the
  rebuilt surgery Schur is VIOLENTLY indefinite on BOTH geometries** —
  toroid 23/40 negative (min −256 vs max +1.1, neg mass 5×), W7-X 47/80
  (min −81 vs max +2.1, neg mass 7.7×). So indefiniteness alone does not
  discriminate W7-X (toroid converges anyway); the magnitude convicts the
  bulk surrogate B of OVERSHOOTING A_bb^{-1} by orders of magnitude in
  core-coupled directions — mechanism: the curl-curl blocks' near-null
  own-axis modes are floor-regularized in the pinv and 1/floor-amplified
  (the 2026-06 suspect). That same overshooting B is the solve-time bulk
  apply; toroid survives because the amplified directions are gradients
  (P_B/leak-damped), W7-X plausibly not. Next build: the PROFILE atom —
  k=0-fdbund-STYLE k=1 terms (own-axis profiles in the K's, UNWEIGHTED
  masses; e.g. A_rr ≈ M⊗K_θ[p(β_ζζ)]⊗M + M⊗M⊗K_ζ[p(β_θθ)]) — exactly
  Lynch-invertible (conflict-free pencils, unlike the rank-1 product fits
  whose same-axis weighted masses force modal-diagonal chopping), with the
  block nullspace analytic → zeroed exactly, non-overestimating B by
  construction. Converts stall→slow if the overshoot is the stall; then
  tier-2 (dense θ-ζ block per r-mode, the 2026-06 design) prices the
  expressiveness ceiling: W7-X β_θθ/β_rr need 4-7 coupled angular modes
  that NO weight-separability-rank-1 atom (CP, bundled, or otherwise)
  can represent.
  **Profile-atom result (MRX_K1_ATOM=profile, jobs 16142545/46): NEW BEST
  k=1 toroid atom** — dbc 237 it (vs 385 bundled / 418 CP / 522 jacobi,
  and a dbc WALL win vs jacobi 336 vs 350 ms), free 531 it: the exact
  k=0-style construction (own-axis profiles in the K's, unweighted masses,
  exact pencils, analytic null zeroed) strictly dominates the rank-1
  product atoms. **W7-X still stalls (~1e-5) → the ENTIRE atom family is
  exonerated**: CP / bundled / profile stall identically. And the rebuilt
  surgery Schur stays violently indefinite EVEN on toroid with the exact
  profile atom (min −567, neg mass 4.8×) while toroid converges fine →
  indefinite-rebuild is generic to inexact-B Schur rebuilds and NOT the
  killer. Surviving suspects, in order: (1) numerical SYMMETRY violation
  of the assembled upper preconditioner (MINRES + slightly nonsymmetric P
  = stall-at-floor signature; probe: random-vector <Pu,v> vs <u,Pv> on
  W7-X vs toroid — few applies, no solves); (2) under-degreed fixed
  Chebyshev L0^{-1} in P_B/Π at W7-X's κ; (3) the k=1 surgery
  extraction/couplings on the W7-X polar structure.
  **Symmetry probe (jobs 16142789/90): EXONERATED** — rel_asym ~1e-15
  everywhere incl. W7-X. **Null(P) floor test (16142917/18): the chopped
  surgery-Schur directions floored at 1/λmax instead of zeroed** — toroid
  unharmed (242 vs 237 it), W7-X still maxiter BUT the residual floor
  moved 2.7× (4.4e-5 → 1.6e-5): a true null(P) stall would not move.
  **Reframed diagnosis: not a stall — extreme κ(P·Ŝ) on W7-X**; 2000 its
  is simply the far tail of a convergent process (fits all data: three
  atoms alike, toroid fine, jacobi's moderate-κ grind converges).
  **DENSE P·Ŝ SPECTRUM (--dense-ps-spectrum, job 16144512): CASE CLOSED —
  it is the huge-κ continuum.** κ_eff: toroid dbc 425 (→ the observed 237
  MINRES its), toroid free 1.7e6 (paired low-mode ladder 2×4.9e-4, 2×7e-3,
  2×5.5e-2 — a THIN deflatable set, ~6 modes would cut κ ~1000×: cheap
  separate win for toroid free), W7-X dbc 7.6e6, free 7.9e7 — seven
  decades, 3397/4176 modes < 1e-4·max = a CONTINUUM (not deflatable),
  plus extreme top outliers (32817 vs 7034 vs 3433 dbc). √κ·log ≈ many
  thousands of its → 2000 lands at ~1e-5: the "stall" was an honest crawl.
  P itself strictly positive on both geometries (floors did their job).
  ⇒ Only structural enlargement changes this: vector-FD per-mode 3×3
  (channel-sparsity: diag block = 2 channels, off-diag (a,b) = single
  channel c∉{a,b} as ONE Kronecker term of C_a⊗C_b⊗M_c[p_c] — the
  existing --pa-block-vector-fd-true-basis / radial_banded models are the
  half-built ladder; pair them with PROFILE weights + per-mode grad-div
  regularization instead of the rel=1e-2 floors), then tier-2 dense θ-ζ
  per r-mode.
  **P_A atom ladder, measured (8³-class, toroid dbc/free its | W7-X dbc
  floor @2000):** profile 237/531 | 1.6e-5; greville-D (=diagonal-scaling
  sandwich D=√(β_bb·β_cc), the June greville P_A via MRX_K1_PA_GREVILLE=1;
  its old W7-X NaN is GONE post-fixes) 303/439 | 1.5e-6; radial_banded
  395/806 | 4.0e-7; vector_fd_true_basis 516/925 | 6.8e-6 (legacy floors
  implicated); **rank1 (MRX_K1_ATOM=rank1) 280/479 | 9.7e-8 — best W7-X
  crawler by ~450× over the original.** rank1 = full rank-1 weights on
  EVERY axis inverted EXACTLY: a two-term block has two 1D matrices per
  axis, any SPD pair is a generalized pencil (mass-mass included) → wire
  each axis's two term matrices as (reference, operator) and the modal
  diagonals are exact; denominator λ_θ(j)+μ(i)λ_ζ(k). (Tobias's theorem —
  corrects the earlier proportionality claim, which applies only to ≥3
  matrices/axis, i.e. k=0's 3-term case. k=1's blocks are structurally
  luckier than k=0.) No toroid winner across BCs (profile dbc, greville-D
  free) → hybrid rank1+D is the indicated next atom. Coupled-exact theory:
  keeping the C_a⊗C_bᵀ⊗M off-diagonals is exactly invertible per tensor
  mode via the paired de Rham eigenbases (gV^N = V^D√Λ) but shrinks the
  weight class to ONE shared rank-1 field × channel scalars (per-channel
  coupled 2×2 pieces are individually exact at full rank-1 — a
  channel-Schwarz middle path exists). Extended-maxiter rank1 W7-X run
  = the convergence test.
  **CONVERGENCE ACHIEVED (job 16146220, maxiter 8000): rank1 CONVERGES on
  W7-X** — dbc 2908 it to 4.7e-10, free ~7500 to 4.0e-10: the first
  converging tensor P_A+P_B on W7-X ever. Not yet competitive (jacobi dbc
  948 it at ~half the per-it cost → ~6× wall gap); the residual κ is the
  coupled θ-ζ content + dropped C-terms. NEXT BUILD (spec final): the
  coupled-exact atom with PAIRWISE-shared weights — the coupling constraint
  is NOT one global field: per derivative axis a, only the TWO channels
  differentiating along a must share their a-factor w_a (spatial pairwise
  average, no geometric means); own-axis factors m_c and scalars α_c stay
  FREE (pencil (M^N[m_a], K[w_a]) per axis; W=gV^N drags M^D[w_a]→I and
  C[w_a]→Λ^{1/2} along automatically). Weight class:
  β_cc ≈ α_c·m_c(x_c)·w_a(x_a)·w_b(x_b). Per-mode 3×3 block
  [[α_ζλ_θ+α_θλ_ζ, −α_ζ s_r s_θ, −α_θ s_r s_ζ], [·, α_ζλ_r+α_rλ_ζ,
  −α_r s_θ s_ζ], [·,·, α_θλ_r+α_rλ_θ]] with analytic null (s_r,s_θ,s_ζ)
  = the mode's gradient — regularize with +σ·vvᵀ (grad-div surrogate,
  physics not floors) or exact per-mode pinv. On W7-X the pairing is kind:
  β_θθ's θ-profile and β_rr's r-profile land in FREE own-axis slots; the
  only contested shared slot (w_ζ between the two hard channels) carries
  small marginals. Edge modes (λ=0 columns, periodic D-complement) are the
  fiddly part. Alternatives ranked after it: hybrid rank1+D (toroid free
  says D helps), channel-Schwarz, tier-2.
  **Shared-slot policy (Tobias): DO NOT MIX metric factors — default the
  shared derivative-axis slots to CONSTANTS**, i.e. the no-mixing
  coupled-exact class β_cc ≈ α_c·m_c(x_c) (own-axis profiles only,
  unweighted ladders; pencil (M^N[m_c], K) per axis; per-mode 3×3 and
  analytic null unchanged). Rationale: a fitted compromise profile
  mis-weights BOTH channels multiplicatively and can be worse per channel
  than a constant; constants are each channel's honest L² projection onto
  "no information along this axis". On W7-X the free own-axis slots keep
  each hard channel's hardest direction (β_θθ's bean θ-profile, β_rr's
  radial profile). Implement the pairwise machinery with the shared slots
  as a knob (w_a=1 default = no mixing; fitted pairwise mean kept only as
  an A/B arm expected to lose).
- **Toroid control: bundled P_A is safe and slightly better** (dbc 418→385,
  free 687→663 it vs CP-ALS) → `MRX_K1_ATOM=bundled` default stands. BUT
  the tensor path's edge over jacobi shrinks with resolution (doc headline
  96-vs-386 it at 6,12,4 → 385-vs-522 and a wall-clock LOSS at 8,16,8):
  k=1 h-scaling needs its own investigation, independent of W7-X.
- Benchmark harness fix that unblocked all of this: three
  `_get_schur_diaginv(..., 'diag')` call sites predated the
  `tensor_probe`-only mode consolidation → always-None → RuntimeError.
- Parked, fully designed, blocked on the stall fix: div-div auxiliary
  replacement for P_B (BC-FLIPPED capped k=2 div-div atom + collocated
  Greville proxy transfer Π₁₂ = J·g⁻¹, NO projection — live with the
  overlap; L2+mass-precond transfer as fallback arm). Removes all L₀
  solves from the k=1 apply; linear (not squared) equivalence constants.

**Experiment results (matrix A–E + follow-ons):**
- A/atom: **fdbund adopted** (decision bullet below). fd λmax prediction 8.7
  MISSED (9.81 measured at 16,32,32) + fd stalled twice on free BC.
- B/h-flatness: **C² == fat-core to all digits on every geometry/size**
  (C² = cheaper realization, confirmed at scale) but **C² does NOT restore
  h-flatness by itself** — h-flatness is ATOM-dependent: fdbund+surgery is
  h-flat on toroid/cerfon, still grows ×1.2–1.5 on rot-ellipse/W7-X (the
  θ-ζ-coupled geometries). Baselines grow ×1.62–1.68/doubling everywhere.
- C/depth: 3-level is benign WITH surgery (fat-core+anchor 24³ toroid:
  λmax 2.63, 7–9 it) — the unanchored/plain-C¹ 3-level elevation (λmax 4.03)
  was the missing surgery, NOT the recursion and NOT the ξ₁ ladder
  (anchored-vs-not at 3 levels: bit-identical null result). But 3-level
  LOSES the W7-X free-BC crossover 2-level wins (0.356 vs baseline 0.314;
  2-level 0.305) — depth adds cost, not value, at these sizes.
- E/wall-clock: with fat-core+anchor, MG(fdbund) crosses over on free BC on
  toroid, cerfon AND W7-X (0.305 vs 0.317 at 16,32,32) — but never on dbc,
  and the per-iteration cost RATIO worsens with size (GPU favors the
  baseline's single apply). λmax(SA) ≈ 1.7/ξ₁ + shaping CONFIRMED on W7-X:
  fat-core drops λmax 5.91→3.73 and auto-m 4→3.
- New atoms (same-day): **fdhel v1 INVALID everywhere** (λmax 19–38; the
  naive shear leaves the ζ-term weighted g^ζζ while the helical derivative
  carries ρ²g^θθ — needs the sheared-metric weight
  g^ζζ+2ρg^θζ+ρ²g^θθ = "v2", future work). Helical-spread diagnostic
  (`scripts/debug/helical_weight_spread.py`): W7-X θθ spread 0.51→0.26 at
  ρ*=-0.55 (real but partial; rot-ellipse only ×1.2). **fdslab == fdbund in
  MG** (no gain); as a smoother it's sound (SPD, mild truncation tax
  λ 1.51→1.66 on toroid).
- Single-level program (`--levels 1`, atom-as-preconditioner in the
  envelope): sl-fd+production-Schur == baseline EXACTLY (path validated).
  **All fat-core single-level arms FLOOR at rel≈1e-2**: the rebuilt Schur
  `ass − C0ᵀ·S·C0` with a RAW-atom S goes indefinite → PSD chop →
  inconsistent preconditioner (V-cycle S ≈ A⁻¹ is why MG-mode rebuild
  works). Fat-core single-level is BLOCKED on a properly probed coupling.
  eps-shift (`--sl-eps-frac`) and truncated-pinv (`--sl-trunc-frac`) both
  REFUTED (mistargeted at atom eigenvalues; τ also floors as predicted).
- Deflation: **dead end.** Low-tail census (extended `--spectrum-diag`):
  ZERO modes below 0.03·λmax; κ_eff 12.6 dbc / 18.7 free (fat-core fdbund,
  W7-X 12,24,24) — smeared spectrum, no tail; single-level counts are near
  their spectral limit, the lever is κ itself (= the fdbund swap).

**Follow-on queue (non-MG):** (1) fdbund swap in production
`mrx/operators.py` atom assembly after the 4-geometry validation
(jobs 16131589–98); (2) `_symmetric_pseudoinverse` fix; (3) fat-core/C²
single-level, blocked on exact fat-core Schur coupling (probe with real
solves at setup, or extend production assembly; C² preferred — 3× smaller
core — once free-BC nullspace plumbing exists); (4) fdhel-v2 sheared-metric
ζ-weight only if W7-X κ still hurts after (1); (5) bundled weights for the
k≥1 saddle atoms.

**Gotcha added the hard way:** the launcher stamp is second-resolution —
two launches in the same second share an output dir and CLOBBER each
other's sbatch log (CSV appends survive). Sleep 2 between launches or add
`$$` to STAMP.

## Where the work stands — the local phase is DONE

The k=0 MG prototype (`scripts/debug/laplacian_mg_k0.py`) and the C⁰/C¹/C²
polar surgery are committed on `greville-prod` (NOT merged to main). Every
question answerable on a laptop CPU at 8³ has been answered; what remains is
**scale** — h-refinement, true multilevel, W7-X, and wall-clock — all of
which need the GPU. Concretely:

1. **Smoother question CLOSED (both directions).** Point-Jacobi is out
   (47–180 CG its vs 5–15 for the fd-family at ~equal ms/it — A-applies
   dominate — and it even loses to the single-level baseline; cause is the
   bulk-wide polar anisotropy `g^{θθ}~1/r²`, the classic point-smoother
   failure). And nothing fancier than the separable fd/fdbund atom pays: the
   off-diagonal `g^{rθ}` lever is refuted (dropping ALL off-diagonals costs
   κ 1.5→2.6 helical vs atoms at ~12); the real residual is
   averaging/separability of the *diagonal* weights, and the best
   production-shaped fix (2D-per-ζ-mode solves) buys only ~1.5× at real cost.
   ⇒ **fd-family separable atom + exact core surgery is the sweet spot.**
   (`k0_massprecond_surgery_findings.md`; `run_mg_k0_jacobi_ab.sh`.)

2. **C²-on-axis == fat-core R=1, but cheaper (priority-1, DONE).**
   `--polar-order {0,1,2}` is wired into the prototype. C² reproduces the
   fat-core λmax(S·A) to all printed digits on every geometry/atom (8³ dbc,
   fd): toroid 1.61, cerfon 2.28, rot-ellipse 3.55 — because both start the
   bulk window at ring 3 — but with **fewer DOFs** (n=560 vs 664) and a **~3×
   smaller Schur core** (6·nz=48 vs fat 3·nz+nt·nz=152): C² *eliminates* the
   near-axis freedom rather than solving it exactly. The C⁰ control (window
   ring 1, λmax 3.9–7.3) confirms the spread tracks exact-region *extent*, not
   pole smoothness per se. ⇒ **C² is the production-preferred realization of
   the fat-core gain**; fat-core stays the C¹-compatible fallback (k≥1,
   free-BC). (`run_mg_k0_polar_order_ab.sh`,
   `outputs/laplacian_mg_k0/polar_order_ab_20260805/`.)

3. **h-scaling (C¹) is close but not textbook-flat — and the fix is
   predicted, not measured.** With the auto-m window rule
   (`--cheb-lo 0.85 --auto-m`, m~n^{1/4}), MG(fd) grows ×1.4 dbc / ×1.2 free
   over 8³→16³ (toroid) vs baseline ×1.7, and the MG:baseline lead *widens*
   with size (2.8×→3.75×). Residual non-flatness is NOT the smoother (damping
   matched) — leading suspects are the **rediscretized non-Galerkin coarse
   operator** and the fine/coarse ξ₁-gap annulus. Crucially,
   `λmax(S·A) ≈ 1.7/ξ₁ + h-independent-shaping`, and fat-core/C² removes
   exactly the `1.7/ξ₁` axis part → **C² should restore near h-flatness**.
   This is the single most valuable unmeasured claim (see experiment B).

## Cluster experiment matrix (priority order)

Driver: `slurm/job_laplacian_mg_k0.sh` (one sbatch/geometry on `gpu-h100s` /
`extremedata`). It now takes an `EXTRA_ARGS` env passthrough and defaults
`SMOOTHERS=fd,fdbund` (fdax retired — fdbund is the adopted alternative). All
knobs are env vars; presets below.

**A. W7-X reconfirm + atom decision + λmax prediction.** The motivating
regime. Reconfirm (12,24,24) baseline 64/99 vs MG(fd) 18/23; add (16,32,32).
`fd` vs `fdbund` is the *open atom decision* (locally a tie; theory says
fdbund pulls ahead only on strong anisotropy = W7-X). Falsifiable check: **fd
λmax ≈ 8.7 (dbc) at (16,32,32)** — if far off, the axis/shaping factorization
misses an interaction. `--spectrum-diag` gives the definitive mode geography.
```
GEOMETRIES=w7x NS_LIST="12,24,24 16,32,32" SMOOTHERS=fd,fdbund bash slurm/job_laplacian_mg_k0.sh
```

**B. fat-core/C² h-sweep — does C² restore h-flatness? (highest-value new run.)**
All C²/fat-core data is 8³ only. Sweep 8³→16³ and check whether MG(C²) is
h-flat where MG(C¹) grew ×1.4. If yes, C² fixes what auto-m only papered over.
```
GEOMETRIES="toroid cerfon rotating_ellipse" NS_LIST="8,16,8 12,24,12 16,32,16" \
  EXTRA_ARGS="--polar-order 2 --anchor-xi1 --bc dbc" bash slurm/job_laplacian_mg_k0.sh
# baseline/C¹ reference arm: EXTRA_ARGS="--fat-core 1 --anchor-xi1" (same window, C¹ layout)
```
`--bc dbc` is REQUIRED with `--polar-order 2` — the launcher's default `--bc both`
makes every invocation SystemExit at startup, and its `|| true` hides that as a
fast-COMPLETED job (first launch 2026-08-13 died this way in 58 s; relaunched
under `outputs/laplacian_mg_k0/2026-08-13/03-42-10`).

**C. True multilevel scaling — completely untested.** *Every* result to date
is two-level with a dense *exact* pseudoinverse coarse solve (deliberately, to
remove the coarse-solve confound). So current iteration counts are best-case;
a real 3+ level V-cycle with the non-Galerkin rediscretized coarse operator
could degrade. Run (24,48,24) at `LEVELS=3`.
```
GEOMETRIES=toroid NS_LIST="24,48,24" LEVELS=3 COARSEN="2,2,2" bash slurm/job_laplacian_mg_k0.sh
```

**D. m=1 smoothing A/B on the κ≈1.6 C² configs.** C²/fat-core drops
axisymmetric geometries to κ≈1.9, where a single damped atom apply (m=1) may
suffice → 3 A-units/cycle, break-even ~3×. Needs a one-line auto-m floor 2→1
in the script (`m = max(1, ...)`) behind a flag before this can run.

**E. GPU wall-clock crossover.** At every size reachable so far AND at W7-X
(12,24,24) on the cluster, the single-level baseline still wins wall-clock —
the 3–4× iteration lead hasn't beaten the m=4 per-cycle A-apply penalty. The
crossover is expected only at larger W7-X on GPU (baseline iteration growth +
GPU utilization). This is *the* open question; A/B/C feed it.

## Gotchas / invariants (carry these to the cluster)

- **Env for W7-X:** `XLA_PYTHON_CLIENT_PREALLOCATE=false W7X_MAP_BATCH=128`
  (the recursive-spline map jacfwd OOMs under full vmap; batched lax.map).
  Needs the fitted map data in gitignored `data/W7-X.h5`.
- **`--polar-order != 1` is dbc-only, no production baseline.** The shipped FD
  preconditioner warm-up (`assemble_laplacian_operators`) hardcodes the C¹
  polar layout, so order≠1 uses incidence-only assembly; free-BC needs the
  nullspace vectors that assembly fills → not yet supported for order≠1.
  Threading invariant: `ring0 = fat_core + (order-1)` drives ALL bulk-window
  indexing (atoms/transfers/profiles/diagonal).
  **Decision point (2026-08-13): the dbc-only limit is plumbing, not math** —
  the C² constraint lives at the axis, the BC at the wall; free BC only lacks
  its nullspace vectors under the order-2 layout. Do NOT fix preemptively:
  fat-core is the C¹-compatible spectral twin and covers free BC today.
  Decide from experiment B: (i) C² separates from fat-core at scale (smaller
  core → wall-clock win) → add free-BC support: cheap route = represent the
  constant mode directly in the C²-reduced basis (k=0 nullspace is analytic);
  proper route = thread `ring0` through `assemble_laplacian_operators`
  (wanted for production wiring anyway). (ii) They stay identical (as at 8³)
  → ship fat-core for both BCs, keep C² as a dbc-only DOF optimization or
  drop it — the limitation becomes moot. Either way the free-BC arm needs its
  own validation pass (W7-X free stalls + the PSD pseudoinverse trap cluster
  there); dbc results don't transfer on faith.
- **PSD pseudoinverse trap.** Any rebuilt Schur (free BC, strong
  preconditioner) can land the analytic null direction slightly negative;
  `_symmetric_pseudoinverse` inverts by magnitude WITH sign → −O(10³)
  Rayleigh. The prototype uses `_psd_pseudoinverse` (positive part only).
  **Production `mrx/preconditioners.py:_symmetric_pseudoinverse` carries the
  same latent trap — fix when wiring production.**
- **fdax is retired** (fdbund strictly better: bundled `⟨g^{aa}J⟩` milder than
  bare `g^{aa}`, cheaper D=1, better-motivated).
- **ATOM DECISION (2026-08-13): fdbund is the default going forward** (Tobias's
  call, on motivation: the bundled per-axis average `⟨g^{aa}J⟩` keeps the g–J
  correlation — `g^{θθ}J ~ 1/r` vs the divergent bare `1/r²` — which is the
  principled compression; fd's collocated `J·(∏g^{aa})^{1/3}` is the ad-hoc
  one). First cluster data agrees: toroid equal-or-fewer its, free-BC 7 vs 11
  (and an fd free-BC stall at 16³, rel_res 1.3e-4), ~3× cheaper smoother
  setup. fd stays available as the A/B reference; the W7-X run (experiment A)
  is now confirmation, not decision.
- **Background/長 runs:** local Bash tasks die at 10 min — the cluster is
  precisely to escape this; use the sbatch driver, not nohup.

## Rerun / repro

- Local sanity (~2.5 min CPU): `python scripts/debug/laplacian_mg_k0.py --geometry cylinder --ns 8 16 8 --two-level-check`
- Local C² smoke: `python scripts/debug/laplacian_mg_k0.py --geometry toroid --ns 6 12 6 --smoothers fd --cheb-lo 0.85 --auto-m --two-level-check --bc dbc --polar-order 2`
- Dense-κ studies (laptop): `scripts/debug/verify_hodge_massprecond_k0.py` (off-diag ladder), `scripts/debug/run_mg_k0_jacobi_ab.sh`, `scripts/debug/run_mg_k0_polar_order_ab.sh`.
- CSV merge on cluster: `awk 'FNR==1&&NR!=1{next}1' outputs/laplacian_mg_k0/<stamp>/mg_*.csv > .../merged.csv`

## Backlog (not cluster-gated)

C² de Rham for k≥1 (Bernstein-identity rework of E¹/E², few hundred lines);
per-ζ-plane ξ for stellarator maps; `get_xi2` map-adapted C² (currently
circle-only); production wiring (PSD pseudoinverse, r_scale=0.5, auto-m rule,
coarsest Cholesky + explicit deflation instead of thresholding); full-space MG
experiment (`--envelope full`, kills the Schur probe + W precompute) — design
sketched in the plan doc.
