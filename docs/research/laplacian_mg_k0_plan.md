> **STATUS: RESEARCH / EXPERIMENTAL — superseded for production by `docs/PRODUCTION.md`.**

# Geometric Multigrid Preconditioner for the k=0 Laplacian (prototype)

**Status (2026-08-13): LOCAL PHASE DONE — moving to GPU/cluster.** Entry point
is now `docs/dev/handoff_2026-08-13_gpu_cluster.md` (retires the four-addendum
`handoff_2026-07-09_polar_c2_mg.md`). Everything answerable at 8³ on a laptop
is settled: the smoother question is CLOSED (jacobi out, off-diagonal lever
refuted, fd/fdbund + exact core surgery is the sweet spot —
`preconditioner_lessons.md`); `--polar-order {0,1,2}` is
wired and **C² reproduces the fat-core R=1 λmax exactly with fewer DOFs + a 3×
smaller core** (priority-1 done). What remains is SCALE — the cluster
experiment matrix (W7-X atom decision + λmax≈8.7 check; the fat-core/C²
h-flatness sweep; true 3-level scaling; GPU wall-clock crossover) is in the
handoff. The running log below is preserved for derivation detail.

**Status (2026-07-07, session 2):** Phase-0 debugging done on cylinder 8×16×8
(CPU). The original suspects were CLEARED; the real defect was found and
fixed; equal-area radial knot grading added. Iteration counts (dbc/free):

- **Transfer defect cleared.** `--p-fix {rownorm,lump}` both restore
  `P_const_err` 0.63 → ~1e-15 (free) / 5e-3 (dbc outer-layer, legitimate) but
  iterations DON'T move (jacobi 65/103, fdax 78/98 vs unfixed 62/94, 62/77).
  rownorm kept as default (mathematically right, free). Not the bottleneck.
- **`--levels 1` diagnostic:** Chebyshev-only PCG (jacobi 92/132, fdax
  87/109) ≈ two-level MG → the coarse correction contributed ~nothing; the
  problem was spectrum coverage, not transfers.
- **`--cheb-window 16`:** dbc modest gain (57/66); free jacobi STALLED
  (3000 it, rel 6e-7). Window widening is not the fix and wide windows are
  fragile under free BC.
- **ROOT CAUSE (fdax): quadrature-divergent profile average.** The cross-axis
  mean of `g^θθ ~ 1/r²` over r∈(0,1) is dominated by the innermost Gauss
  points (∫r⁻² diverges) → θ-stiffness weight inflated by ~an order of
  magnitude → interior modes of S·A squashed to tiny λ → Chebyshev window
  missed them. FIXED (no knob): radial integration in the fdax cross-axis
  means starts at the first interior breakpoint ξ₁ (the surgery element is
  core-handled by the Schur envelope anyway). Effect on the UNIFORM grid:
  **MG(fdax) 78/98 → 17/21, beating baseline 24/26**; lam_max 1.23 → 5.25.
- **Equal-area radial knots** (`--r-scale`, default 0.5; breakpoints
  `linspace**r_scale` via the existing `DeRhamSequence(r_scale=…)`, now
  threaded through `build_sequence`): cells ~equal disk area, first
  breakpoint 1/n_el → 1/√n_el (0.2 → 0.45 at 5 elements) → tames the
  near-axis 1/r² anisotropy AND helps the production baseline: baseline
  24/26 → 18/19; **MG(fd) 6/6; MG(fdax) 9 (dbc)**.
- **MG(jacobi) stays bad** (71/139 at r_scale 0.5) — a pointwise diagonal
  atom + upper-window Chebyshev cannot cover this spectrum regardless of
  transfers/window. That answers the smoother comparison: the 1D
  eigendecomposed atoms (fd/fdax) are the way. CLOSED with data under the
  auto rule too (cylinder 12³: jacobi 58/86 at m=4/5 vs fd/fdax 7–14/10–16
  under the identical rule): the diagonal doesn't COMPRESS the near-axis
  anisotropy into a coverable band, and the left-behind low-λ modes are
  geometrically oscillatory — invisible to the coarse space. Atom problem,
  not window problem.
- **RESOLVED: free-BC fdax SPD failure was the SIGNED pseudoinverse on the
  rebuilt Schur, NOT the eigenvalue estimate.** Dense probe of S·A (fdax,
  free): λ ⊂ [0.41, 3.46], Lanczos est 3.81 — a valid bound (power iteration
  would change nothing; top of spectrum is a cluster 3.46/3.41/3.41…, gap
  ratio 0.98, where Lanczos ≫ power anyway). The true free-BC core Schur is
  singular-PSD (constant null vector); the rebuild `ass − C0ᵀ B_mg C0` puts
  that direction at ± tiny, and a STRONG B_mg (λ(SA) up to 3.46 > 1) can land
  it slightly negative — `_symmetric_pseudoinverse` inverts by magnitude
  WITH SIGN (`preconditioners.py:1910`) → −O(10³) Rayleigh. The old weak
  fdax masked this latent trap (production FD under-approximates → PSD by
  luck). FIX: `_psd_pseudoinverse` in the script (invert only the positive
  part) for the Schur rebuild + coarsest solve. ⚠ production
  `_symmetric_pseudoinverse` carries the same trap for any future strong
  preconditioner — fix in `mrx/` when wiring production.
- **Phase 0 PASSES (all gates, both BCs), cylinder 8×16×8, r_scale 0.5:**
  baseline 18/19 · MG(fd) 6/7 · **MG(fdax) 9/10** · MG(jacobi) 71/139;
  sym_err ~1e-17, min_rayleigh ∈ [15.6, 56.6]. At ~6.9 A-units/it (m=2) vs
  baseline 1.4, fdax is ~2× iteration reduction — below the 5× break-even at
  this size; the h-sweep decides (MG wins if h-flat while baseline grows).
**Phase 1 (cylinder h-sweep, r_scale 0.5) + Phase 2 (toroid), dbc/free:**

| ns | baseline | MG(fd) | MG(fdax) | fdax λmax(SA) |
|----|----------|--------|----------|---------------|
| (8,16,8)   | 18/19 | 6/7   | 9/10  | 3.81 |
| (12,24,12) | 25/26 | 10/11 | 14/16 | 5.40 |
| (16,32,16) | 31/32 | 14/14 | 19/21 | 6.75 |

- (m=2, `--cheb-window 4`.) MG grows ×2.1 over the sweep vs baseline ×1.7 —
  NOT h-flat with a RELATIVE window. Cause identified: **λmax(S·A) ≈ 1.7/ξ₁
  ~ √n_el** (2.2/3.0/3.6 vs 3.81/5.40/6.75) — the structural cross-axis g^θθ
  residual widens as the first element shrinks — so the relative window's
  lower edge rises (0.95→1.7) and a growing band of O(1) modes gets neither
  smoother nor coarse-grid treatment.
- **Window design rule (verified in two steps):** pin the lower edge at an
  ABSOLUTE O(1) value (≈0.85 ⇒ window κ = λmax/0.85) and grow the Chebyshev
  degree like m ≈ √κ (i.e. m ~ n^{1/4}: 3 at 8³, 4 at 16³). With the edge
  pinned but m=3 fixed: fd 4/4→8/8, fdax 6/7→12/14 — still ×2 because
  T₃-damping degrades 10.8→4.6 as κ 4.5→8; **m=4 at κ=8 (T₄≈9.5, matched
  damping): fd 6/7, fdax 10/12** ⇒ growth ×1.5–1.6 vs baseline ×1.7. Not
  fully h-flat (target ≤1.3×) — the residual growth is NOT the smoother
  (damping matched); leading suspects: the non-Galerkin rediscretized coarse
  operator, and the fine/coarse ξ₁ gap annulus (fine 0.277 vs coarse 0.38 at
  16³) invisible to the coarse space. MG holds a ~2.7× iteration lead that
  erodes only slowly; cylinder wall-clock still favors baseline (m=4 ⇒ 9
  A-applies/cycle).
- **Toroid (12,24,12), all SPD gates pass, no free-BC stall:** baseline
  26/38, MG(fd) 10/15, MG(fdax) 14/18. Same counts as the cylinder at equal
  ns ⇒ mild curvature is not the driver; near-axis anisotropy dominates.
  Baseline degrades under free BC (38) while MG holds ⇒ MG's margin widens
  exactly where the production FD weakens (the W7-X direction).
- Historical note: the free-jacobi `--cheb-window 16` stall predates the PSD
  pseudoinverse fix and matches that bug's signature (indefinite envelope →
  CG stall at 1e-7); wide windows are not per-se indicted.
- **DECISION: the coarse operator stays REDISCRETIZED** (build on the
  coarser grid, reuse the fine map) — cleaner and production-friendly (no
  dense Galerkin PᵀAP products at scale). The variational-inconsistency
  share of the residual ×1.5–1.6 h-growth is accepted unless the toroid
  sweep says otherwise.
- Window/degree rule wired into the script: `--cheb-lo 0.85 --auto-m`
  (per-level κ = λmax/cheb_lo, m = max(2, round(1.414·√κ)); CSV `m` column
  reports the actual fine-level m).
**Toroid h-sweep (auto rule `--cheb-lo 0.85 --auto-m`), dbc/free:**

| ns | baseline | MG(fd) | MG(fdax) | fdax κ/m |
|----|----------|--------|----------|----------|
| (8,16,8)   | 19/28 | 7/11  | 6/10  | 4.5/3 |
| (12,24,12) | 26/38 | 7/11  | 8/10  | 6.4/4 |
| (16,32,16) | 32/45 | 10/13 | 10/12 | 7.9/4 |

- Growth 8³→16³: baseline ×1.7/×1.6; MG(fdax) ×1.7 dbc (from a base of 6) /
  **×1.2 free**; MG(fd) ×1.4/×1.2. Free BC meets the ≤1.3× criterion; dbc
  nominally misses on tiny absolute counts. **The MG:baseline iteration lead
  WIDENS with size (2.8–3.2× at 8³ → 3.2–3.75× at 16³)** — the right trend
  for W7-X, where the baseline is weakest. The auto rule halved fdax's 12³
  counts vs the old relative window (14/18 → 8/10).
- Wall-clock (CPU, these sizes) still favors the baseline (fdax ~345 ms/it
  vs ~50; lead 3.2–3.75× < the ~5–9× break-even at m=4). The wall-clock
  verdict needs W7-X-scale on GPU.
**W7-X first results (2026-07-08, cluster, (12,24,24), auto rule), dbc/free:**
baseline **64/99** (the motivating degradation, finally visible) · MG(fd)
**18/23** (κ=8.5/8.8, m=4) · MG(fdax) 19/23 (κ=11.7, m=5). All SPD gates
pass on the fitted map + graded knots; `--zeta-diag`: a_zz cv=0.10, low
dominant ζ-modes ⇒ ζ-coarsening 24→12 is safe; P_const_err 3.5e-3/2e-15.
- **ATOM DECISION FORMING (superseded 2026-07-09: reopened by the
  fat-core sweep — see below): fd (production 1/3 atom) wins or ties on ALL
  THREE geometries** — on W7-X fdax has LARGER spread (11.7 vs 8.5 ⇒ m=5 vs
  m=4, ~20% dearer per apply) at equal iterations. Structural reason: W7-X
  variation is HELICAL (θ − nfp·ζ jointly); the 1D marginals fdax uses wash
  out on the (θ,ζ)-diagonal, while fd's collocated pointwise-3D D tracks it
  locally. fdax's own-axis-variation premise does not hold on stellarators.
- Wall-clock at (12,24,24): baseline still 1.8–2.8× faster (MG ~10 ms/it vs
  ~1; iteration lead 3.6–4.3× < the ~10× m=4 break-even). Watch (16,32,32):
  baseline growth + GPU utilization vs MG flatness decides the crossover
  trajectory ((24,48,24)/3-level next if short).
- **What dominates the W7-X spread — axis (1/r²) vs shaping (θ–ζ)?** Both,
  comparably, at (12,24,24). Decomposition from same-ξ₁ cross-geometry
  λmax(S_fd·A): cylinder 3.1 ≈ toroid 3.2 (shared polar axis part) vs W7-X
  7.2 ⇒ shaping multiplies ×~2.3 (fdax: 5.4→9.9, ×1.8). Scaling differs:
  axis part grows ~1.7/ξ₁ with refinement; shaping part is h-independent.
  **Falsifiable prediction for the (16,32,32) sweep row: fd λmax ≈ 8.7
  (dbc)** — if far off, the factorization misses an interaction. Definitive
  measurement = `--spectrum-diag` mode (dense S·A probe at (12,24,24),
  5760², ~265 MB, H100-ok): top-eigenvector radial energy profile + (m,n)
  Fourier content (axis-dominated: innermost-r, θ at grid limit; shaping:
  mid-radius, helical m ≈ nfp·n correlation). IMPLEMENTED 2026-07-09
  (`--spectrum-diag`: S^½AS^½ eigh on the fine bulk, cap nb≤6000; reports
  dense λmax vs Lanczos, per-mode r-peak, inner-2-ring energy, top (m,n)).
- **Anchored-ξ₁ effect size (expectation):** freezes only the axis factor —
  ~×1.2 in κ at (16,·,·), ~×1.5 at (24,·,·) ⇒ 10–25% cost via m~√κ at
  practical sizes; asymptotically √n_el. Does NOT touch the ×2.3 shaping
  residual. The MG-level-consistency benefits may outweigh the κ itself.
- **C²-on-axis surgery idea (Tobias):** constrain the 3rd radial ring into
  the core (3 → 6 polar functions/ζ-plane; Schur 72→144, trivial) ⇒ bulk
  vanishes to 3rd order at the axis ⇒ the worst 1/r²-vs-average mismatch
  the atom sees moves OUTWARD — same term anchoring attacks, but through
  the function space at fixed mesh. Real cost = spline math: C² polar
  extraction needs map-dependent (axis-Hessian) compatibility conditions.
  **MEASURE BEFORE BUILDING — "fat core" emulation:** fold ring 2 into the
  exactly-solved Schur core in the prototype (core 3nz → 3nz + nt·nz, bulk
  window starts at ring 3 — pure indexing, C¹ extraction untouched). If
  W7-X κ drops materially ⇒ C² pays; if not, math saved. IMPLEMENTED +
  MEASURED locally 2026-07-09 (`--fat-core`; forces `--schur rebuild`,
  envelope probes ass+C0 in one lax.map pass, PSD pseudoinverse; all SPD
  gates pass) — results in the local sweep section below.

**Local fat-core + spectrum sweep (2026-07-09, laptop CPU, (8,16,8), auto
rule `--cheb-lo 0.85 --auto-m`, r_scale 0.5; runs/CSV in
`outputs/laplacian_mg_k0/local_fatcore_20260709/`, driver
`scripts/debug/run_mg_k0_local_fatcore.sh`).** New geometry `cerfon` =
Cerfon–Freidberg one-size-fits-all map wired into `build_sequence`
(κ=1.7, α=0.4 ⇒ triangularity 0.39; NON-diagonal metric, |g^rθ|max ≈
1.7·g^rr_max) — axisymmetric shaping. `rotating_ellipse` (κ=1.5, nfp=3) =
the laptop helical proxy. λmax(S·A) dbc, plain → fat (iterations dbc/free
plain → fat in parens; baseline degrades toroid 19/28 → cerfon 28/37 →
rot-ellipse 39/46 while MG holds — the W7-X trend reproduced locally):

| geometry | fd λmax | fdax λmax | fd it | fdax it |
|----------|---------|-----------|-------|---------|
| cylinder | 3.1→1.49 | 3.81→(n/a) | 6/7→5/5 (m=2,w=4) | — |
| toroid | 2.33→1.61 | 3.81→1.83 | 7/11→5/7 | 6/10→4/8 |
| cerfon | 3.13→2.28 | 4.02→1.96 | 9/12→10/13 (m 3→2, −25%/it) | 9/12→8/11 |
| rot-ellipse | 4.36→3.54 | 7.29→3.39 | 13/15→12/14 | 13/14→10/12 |

1. **FAT CORE PAYS, biggest exactly where the spread was worst:** fdax
   helical 7.29→3.39 (×2.15), fdax shaped 4.02→1.96, cylinder-axis fd
   3.1→1.49. The C²-emulation premise holds: exact core treatment of ring
   2 absorbs the dominant axis mismatch. NOTE fat-core is not only an
   emulation — it is a WORKING configuration as-is (core 3nz+nt·nz; Schur
   still trivial at prototype sizes). Its real cost is the W probe (nt·nz
   extra V-cycles at setup: 4–5 s vs 1.7 s at 8³, →1152 V-cycles at
   (24,48,24)) — THAT is what the C² surgery (6/ζ-plane core) buys back.
2. **ATOM DECISION REOPENED: post-fat fdax wins or ties on all three
   local geometries** (toroid 4/8, cerfon 8/11, rot-ellipse 10/12 — beats
   fd everywhere but toroid-free). The W7-X "fd wins" conclusion was
   drawn WITHOUT fat-core; the fdax helical washout penalty apparently
   lives mostly in the axis-shaping INTERACTION (plain fdax rot-ellipse
   top modes: ring 0, E[rings 0-1]=0.99 AND (m,n)=(±7–8, 4) grid-corner
   helical) which the fat core removes. Re-run the W7-X atom comparison
   with `--fat-core` when cluster access returns; expect fdax κ 11.7 to
   drop ~×2 (→ ~5–6) vs fd 8.5→(less).
3. **Spectrum-diag mode geography (axis vs shaping, confirmed):**
   cylinder-fat: top modes at innermost kept ring, m=nt/2, n≈0 — pure
   axis remnant. cerfon-fat fd: modes move to the OUTER boundary (r≈1,
   inner-ring energy 0.01) — the irreducible shaping residual; cerfon-fat
   fdax keeps a small axis remnant instead (own-axis θ-averaging captures
   axisymmetric shaping — fdax premise holds when variation is NOT
   helical). rot-ellipse: both atoms' survivors are grid-corner (m=8,n=4)
   helical modes — fd's at the shaped edge, fdax's at the axis.
4. **Residual after fat is the h-independent shaping share:** κ ≈ 1.9
   (toroid) / 2.3–2.7 (cerfon) / 4.0–4.3 (rot-ellipse) at cheb_lo 0.85 ⇒
   m=2–3 everywhere. At κ ≈ 2 a SINGLE damped atom apply (m=1, T₁ ≈ 3
   band damping) becomes viable — worth an `--auto-m` floor 2→1 A/B on
   the fat configuration (3 A-units/cycle ⇒ break-even ~3×; the answer to
   "why Chebyshev at all" is quantitatively "κ > ~2.5", and fat-core puts
   axisymmetric geometries below that).
5. **Ring count (`--fat-core R` now an int knob): R=2 confirms diminishing
   returns** (λmax dbc, R=1→R=2): toroid fd 1.61→1.42, fdax 1.83→1.32
   (it 4/3); rot-ellipse fd 3.54→3.38, fdax 3.39→2.66 (it 9-11) —
   ≈ the √((k+1)/k) equal-area prediction; the helical shaping residual
   survives any R. Decision: **R=1 + anchored-ξ₁** is the design point
   ("core footprint ~ coarsest first element" is anchoring in disguise:
   rings cost 2^{L-1}−1·nt·nz core DOFs + W-probe V-cycles; anchoring the
   knots gives level-identical cores at R=1 for free).
6. **NEW atom `fdbund`** (per-axis quad-means of the BUNDLED g^aa·J,
   D = 1 — answers "average g·J instead of g + collocated J?"): keeps the
   g–J correlation (θ-weight ~ 1/r not 1/r²) but loses pointwise J
   tracking. Measured: plain toroid λmax 2.32 (7/7 at m=2, beats fdax
   free 10); plain rot-ellipse 3.83 (vs fdax 7.29!), 13/15 at m=3 not
   m=4; with fat+anchor: toroid 1.51, **5/5 — best free-BC toroid row**;
   BUT rot-ellipse dbc 15 it (worst) — the predicted low-λ axis tail
   (flat-vs-r mismatch in the r/ζ terms) bites exactly on helical+dbc.
   Two-sided as theorized; candidate default for axisymmetric, keep
   fdax/fd for stellarators pending a W7-X row.
7. **Anchored-ξ₁ implemented** (`DeRhamSequence(r_breakpoints=…)` in mrx +
   `--anchor-xi1`: coarse levels keep the FINE ξ₁, outer coarse elements
   equal-area over [ξ₁,1]; all levels anchor to the finest). Measured at
   8³ two-level: **P_const_err 5e-3/1.2e-2 → 7.8e-16 both BCs** (the
   axis-side transfer defect is GONE — coarse near-axis functions become
   fine-representable); fine-level λmax unchanged (consistency: it's a
   fine-grid quantity); iterations neutral (±1) at this size — the payoff
   (level-consistency, h-flatness) needs the 12³/16³ levels-3 sweep on
   cluster. All SPD gates pass. (API note: `r_breakpoints` was reworked to
   `DeRhamSequence(knots=(T_r, T_θ, T_ζ))` — three optional FULL knot
   vectors, padding implied by axis regularity; regression-identical.)
8. **get_xi GENERALIZED to map-adapted ξ (thesis Eq. 5.7–5.9)**:
   `get_xi(nt, ring1)` = barycentric weights of the ACTUAL ring-1 control
   points w.r.t. the τ-triangle (τ per Eq. 5.9 ⇒ weights ∈ [0,1], PoU
   exact); `ring1_control_points()` extracts them from the Greville
   interpolant of a poloidal map; `DeRhamSequence(polar_ring1=…)`,
   prototype `--xi-adapt` (axisymmetric only — ζ-dependent maps need
   per-ζ-plane ξ = a PolarExtractionOperator refactor, deferred).
   ring1=None reproduces the old circle EXACTLY (1e-16); our old get_xi
   is the circle specialization, exact iff ∂F/∂r|axis is pure m=±1
   (includes ellipses; cerfon/W7-X are not). Cerfon A/B: **solver-
   invariant by construction** — measured identical baseline 28/37, λmax
   3.13/4.02, it 9/12: the BULK block never sees ξ (bulk rows of E are
   identity; the ξ-mixed polar functions are core, solved exactly by the
   Schur envelope). The payoff is DISCRETIZATION accuracy near the axis
   on shaped maps (max weight shift 0.24 on cerfon) — demonstrate with a
   manufactured-solution L2 study, not iteration counts.

**C²-on-axis surgery IMPLEMENTED for k=0 — full theory/validation
write-up in `docs/polar_pole_regularity.md` (jet-matching derivation,
the product-of-splines obstruction, collocated-C² vs the Toshniwal
multi-degree exact route, C⁰/C¹/C² family, code map).** (2026-07-09,
`DeRhamSequence(polar_order=2)`, `get_xi2`, verification
`scripts/debug/verify_c2_polar.py`, convergence
`scripts/debug/poisson_k0_c2_convergence.py`).** Derivation = jet matching
against the degenerate spline map: ring-2 condition
`c₂(χ) = q₀ + q₁·ΔP₂(χ) + ρ·ΔP₁(χ)ᵀQΔP₁(χ)`, ρ = 2N₁'(0)²/N₂''(0) — the
Hessian enters through the RING-1 offsets with a knot-dependent scale, NOT
by evaluating q at ring-2 control points. The quadratic term is a product
of splines (degree 2p in χ) ⇒ EXACT C² w.r.t. the discrete map is
impossible in the fixed tensor space — that is the precise content of the
thesis' "in principle possible" and why implementations stop at C¹.
Resolution: COLLOCATED C² (quadratic term sampled at the Greville angles) —
the same sampled-coefficient class as C¹'s own spline-sampled pole jets.
6 polar functions/plane = quadratic Bernstein jets on the C¹ control
triangle (PoU exact on every ring; V⁰_C² ⊂ V⁰_C¹ by construction, verified
3e-16). k≥1 extractions stay C¹ (guarded) — the k=0 pipeline is fully
consistent because apply_stiffness sandwiches the TENSOR incidence + mass
between E⁰'s only. VERIFIED: Taylor-remainder at the pole order 3.00
(random C² element) vs <2 (generic C¹ element) — a genuine 2-jet.
Poisson k=0 toroid dbc p=3, manufactured solutions with poloidal m=0/1/2
(m=2 = the ring-2-critical case), source by AD through the metric,
n∈{4,6,8,12}: **C¹ and C² L2 errors IDENTICAL to all printed digits at
every n; final rates 4.44/4.32/4.39 both orders (O(h⁴) ✓)**; C² has
10–16% fewer DOFs and consistently fewer jacobi-CG iterations (the
removed near-axis DOFs were the stiff ones). ⇒ the C² constraint costs
NOTHING in approximation and delivers the permanently-small exact core
(6nz vs fat-core's 3nz+nt·nz) — next: wire `--polar-order 2` into the MG
prototype and measure κ vs fat-core R=1.

- Local phases 0–2 are DONE. Remaining: W7-X (16,32,32) rows + legacy-m
  reference + (24,48,24) levels-3 + full CSV sweeps. jacobi is
  answered — drop from default smoother list. Production wiring notes: use
  the PSD pseudoinverse (`_psd_pseudoinverse`) for any rebuilt Schur; carry
  `r_scale=0.5` + the fdax ξ₁ profile cutoff + `--cheb-lo/--auto-m` rule.
- **AGREED NEXT EXPERIMENT — full-space MG (drop the Schur envelope):**
  V-cycle on the WHOLE extracted operator (core included), knob
  `--envelope {schur,full}` for A/B on cylinder+toroid 8³/12³. Rationale:
  the punctured-disk near-kernel pathology (k=1 section below) is an
  artifact of the bulk cutout — the full operator's nullspace is ANALYTIC
  (constant / BETTI harmonics); also kills the Schur probe + W precompute
  (dominant setup cost) and the rebuilt-Schur fragility class; per-iteration
  cost ~unchanged (bulk apply already calls the full apply_stiffness).
  Design: additive block atom S = blockdiag(ass⁻¹, S_bulk) (core dense
  3nz×3nz per level) inside the same Chebyshev; **full-space transfers via
  the extraction operators: P = E_f · T · E_cᵀ** (extend coarse extracted
  DOFs to the coarse tensor grid, plain tensor-product 1D prolongation T
  over the FULL radial basis — no window slicing, no core special-casing,
  no rownorm — then re-extract on the fine level; keeps the cross terms
  coarse-bulk → fine-core; R = Pᵀ); coarsest = dense full-operator probe +
  the analytic free-BC null handled by deflation-or-shift (implementation
  detail), Cholesky/eigh once — no magnitude thresholds anywhere. Open
  empirical question: does λ(S·A) stay tame with the core-bulk coupling
  handled iteratively? Fallback if the axis degrades: symmetric
  core-exact / bulk-Chebyshev multiplicative smoother.
  Supporting changes (useful independently): DeRhamSequence accepts a
  CUSTOM radial knot vector (not just r_scale) so coarse levels can ANCHOR
  ξ₁ (identical core footprint across levels — rediscretized coarse op
  agrees with fine exactly where coupling is stiffest) — DONE 2026-07-09
  (`r_breakpoints=`); precompute the per-level surgery ξ's (get_xi at each
  coarse nt) — minor.
  **Anchored-ξ₁ rationale (agreed: do not refine the center; the surgery is
  always just the 3 polar bases per ζ-plane):** one knob r₀ = fixed first
  breakpoint buys simultaneously (1) fixed physical surgery footprint across
  h-refinement AND MG levels, (2) fine/coarse core geometric identity, (3) a
  stationary fdax averaging domain, and (4) — HYPOTHESIS, untested — h-flat
  atom spread: κ is set by the true-vs-averaged weight mismatch at r ≈ ξ₁
  (frequency-independent since both θ-energies scale as m²); cylinder sweep
  gives λmax·ξ₁ ≈ 1.70/1.80/1.87 (≈constant, weak upward creep unexplained;
  W7-X shaping adds an h-independent cross-axis residual on top). NOTE the
  current sweep does NOT anchor: each resolution re-grades equal-area, so
  ξ₁ = 1/√n_el shrinks as it refines (the fdax profile cutoff tracks each
  grid's own ξ₁).
- **Coarsest-solve cleanup (agreed, not yet implemented):** the coarsest
  BULK block is nonsingular (core=0 ⇒ bulk functions vanish to 2nd order at
  the axis — a point-Dirichlet-like subspace condition with no kernel
  capacity for scalars) ⇒ plain Cholesky, no pseudoinverse/threshold. The
  only genuinely singular object is the FINE-level rebuilt Schur (free BC,
  analytic null 1_c) ⇒ deflate it explicitly instead of thresholding. Outer
  CG deflation does NOT protect the envelope (both observed failures
  happened with it active).
- **⚠ k=1 (vector Laplacian) warning:** the bulk cutout turns the solid
  torus (b₁=1, b₂=0) into a thickened torus SHELL (b₁=2, b₂=1). Near-null
  count of the k=1 bulk block (Hodge: free/Neumann ↔ b₁ tunnels, dbc ↔ b₂
  cavities): **free: 2** — the θ-loop around the removed core (mollified
  `dθ = θ̂/r`) + the toroidal ζ-loop (exact harmonic of the FULL operator,
  deflated by the BETTI machinery there, but perturbed into a bulk-block
  near-null by the ring constraint); **dbc: 1** — the cavity field `∇u`,
  u harmonic and constant per boundary torus (~`r̂/r`, θ-independent).
  These have Rayleigh quotient ~1/log(1/ξ₁): h-dependent and decaying only
  LOGARITHMICALLY — no threshold separates them cleanly at any resolution.
  The k=0 recipe does not generalize: at the coarsest level COMPUTE the
  1–2 near-kernel vectors (small dense eigensolve; seeds: dθ, ∇log r, ζ̂),
  deflate explicitly, and ensure the transfers reproduce them across levels.

Laptop note: cylinder/toroid/rotating_ellipse run on CPU out of the box; w7x
needs the fitted map data (gitignored `data/`) and a GPU — cluster only.
(Currently NO slurm/GPU access — small local CPU tests only; the sweep
phases 3–4 wait until cluster access returns.)
Repro: `python scripts/debug/laplacian_mg_k0.py --geometry cylinder --ns 8 16 8 --two-level-check`
(~2.5 min CPU).

## Context

The k=0 scalar Laplacian `K₀ = G₀ᵀ M₁ G₀` is solved by PCG with a single-level greville
fast-diagonalization (FD) preconditioner (`_assemble_k0_greville_bulk_factors`,
`mrx/operators.py:1380`, inside the core/bulk Schur envelope at `operators.py:1536/1603`).
Two problems: (1) single-level → h-dependent CG iteration growth; (2) a **structural
obstruction**: any single-sandwich FD form forces the θ- and ζ-summands to share their
radial weight profile, but the true profiles differ by ~r² (`w_θθ ~ 1/r` vs `w_ζζ ~ r`) —
so the FD atom's metric handling is *necessarily* some averaging of the g-factors.

Plan: a **geometric multigrid V-cycle** on the bulk (h-independence), with a **three-way
smoother comparison** that measures how the metric should enter the smoother:

- **jacobi** — Chebyshev over `diag(K₀)`: locally exact for all three `g_aa·J` weights, no
  separability assumption.
- **fd** — Chebyshev over the production atom as-is: `D = J·(g^rr g^θθ g^ζζ)^{1/3}` at
  Greville points + global per-term scalars α (the "1/3 version").
- **fdax** — NEW axis-averaged FD: `D = J` at Greville points; per axis, generalized
  eigendecomp of the pair `(M_a, K_a[ḡ^aa(x_a)])` with `ḡ^aa(x_a)` = quad-weighted mean of
  `g^aa` over the other two axes; denom `λ_r+λ_θ+λ_ζ`, **no α scalars**. Captures each
  g-factor's variation along its own axis (g_rr in r, g_θθ in θ, g_ζζ in ζ — the W7-X
  shaping/helical variation); the cross-axis part (e.g. g^θθ ~ 1/r²) is the measured
  residual — structurally unavoidable. Exactly diagonalizable: one mass + one weighted
  stiffness per axis.

Decisions made: full 3D coarsening default with per-axis knob; prototype script only
(production wiring deferred); single-level production FD stays as the baseline to beat.

## Files

- **NEW** `scripts/debug/laplacian_mg_k0.py` — the prototype (all logic in-script; no
  `mrx/` changes this pass).
- **NEW** `slurm/job_laplacian_mg_k0.sh` — per-geometry sbatch on `gpu-h100s` /
  `extremedata` (W7-X: `XLA_PYTHON_CLIENT_PREALLOCATE=false W7X_MAP_BATCH=128`), CSVs to
  `outputs/laplacian_mg_k0/<stamp>/`.

## Architecture

MG acts only on the **bulk** block `A = A_bb` (embed bulk vector with core=0 →
`apply_stiffness(seq, ops, ·, 0, dirichlet)` → slice `[core:]`; core = `3·nz`,
`nr_bulk = nr − 2 − int(dirichlet)`, radial window **starts at 2 always**; the dirichlet
flag drops the LAST radial function — slice `P_r_full[2:2+nr_bulk_f, 2:2+nr_bulk_c]`).

### Levels (rediscretized, geometry reused)
Coarse level = new `DeRhamSequence(ns_c, (p,p,p), 2p, TYPES, polar=True)` +
`evaluate_1d()` + `assemble_reference_mass_matrix()` + `set_map(fine_seq.map)` (reuses the
fitted W7-X map; jacfwd only at coarse quad points, ~1/8 cost) +
`assemble_incidence_operators(ks=(0,))` + `assemble_laplacian_operators(ks=(0,))` + eager
`apply_stiffness` warm-up per BC (mass-core cache / tracer guard). Recipe:
`scripts/benchmark/benchmark_k0_rank_geometries.py:200-219`.

`coarsen_ns` (knob `--coarsen fr,ft,fz`, default `2,2,2`; `--levels`, default 2):
- periodic θ/ζ: `n_c = n_f // f`, floor `max(4, p+1)`; W7-X ζ floor `2·nfp` (checked
  empirically by a `--zeta-diag` mode: count oscillation periods of `a_zz(r₀,θ₀,ζ)`).
- clamped r: halve elements — `e_c = max(2, ceil((n_f−p)/2))`, `n_c = e_c + p`, floor
  `n_c ≥ 5` (dbc). Non-nesting is fine (transfers are quasi-interpolation).

### Transfers (tensor-product 1D, R = Pᵀ)
`P_axis = solve(C_ff, C_cf)` where `C_ff = fine.collocation_matrix(fine_greville)` (square,
Schoenberg–Whitney) and `C_cf = coarse.collocation_matrix(fine_greville)`
(`spline_bases.py:136`; periodic wrap already handled). Exact injection on nested periodic
axes; O(h^{p+1}) quasi-interpolation on the non-nested radial axis. Radial slicing per the
window rule above. Apply = 3 einsums (mirror `_fd_apply_3d` style). Known benign defect:
window truncation breaks exact constant reproduction in a ~p-wide layer at the axis —
report `‖P·1_c − 1_b‖_∞` per level (`P_const_err`), accept unless Phase-1 shows otherwise.

### Smoothers (`--smoother {jacobi,fd,fdax}`)
All three wrapped in symmetric Chebyshev on `[λ_max/window, λ_max]` (`--cheb-window 4`,
`--smooth-steps m` default 2): λ_max per level from the existing Lanczos helper
(`_estimate_chebyshev_lanczos_bounds_apply`, `preconditioners.py:902`) with
constant-mode deflation via `orthogonal_vectors` for free BC; **discard the Lanczos λ_min**.
Local ~30-line copy of the Chebyshev body (`preconditioners.py:1757`) returning
`(x, residual)` — saves 1 A-apply per smoothing pass.
- jacobi: `diag(K₀)` bulk slice via the `diag_EAET_direct` pattern used by
  `assemble_mass_jacobi_preconditioner` (`operators.py:1707`) on the sparse
  `G₀ᵀ M₁_sp G₀` product (reuse the scalar-hodge 'jacobi' kind if already wired).
- fd: the level's `_assemble_k0_greville_bulk_factors` + `_fd_apply_3d` (eps=0 null
  threshold active).
- fdax: in-script builder using `_assemble_weighted_1d_stiffness` + 1D generalized
  eigendecomp (`_assemble_1d_fd_eigendecomp`, `operators.py:2715`); profiles `ḡ^aa(x_a)`
  from the resident quad-point metric; D = J collocated at the 0-form Greville grid.

### V-cycle (symmetric, SPD)
```
V(r): x1, r1 = smooth(r)          # pre, m A-applies (residual free from cheb loop)
      ec = coarse_solve(Pᵀ r1)
      c  = P ec;  r2 = r1 − A c   # 1 A-apply
      dx, _ = smooth(r2)          # post, m A-applies
      return x1 + c + dx
```
2m+1 fine A-applies per cycle. SPD: `B = 2T − TAT + (I−TA) P B_c Pᵀ (I−AT)` with T a
symmetric Chebyshev polynomial (λ(TA) ⊂ (0,2) by the window), R=Pᵀ, B_c SPD → B PD.
Numeric gates: symmetry error < 1e−12, positive Rayleigh quotients on random vectors.

### Coarsest solve (`--coarse-solve {dense,fd,fdax}`, default dense)
Dense probe of the coarsest bulk operator (sequential `lax.map`, n_c coarse applies at
setup) + Cholesky. Exact — removes the coarse-solve confound from the smoother comparison.
n_c is small by construction (~0.3–3k DOFs; ≤80 MB dense worst case).

### Schur envelope (rebuilt, one V-cycle per apply)
Probe `ass` (3nz×3nz) and `C0` (n_bulk×3nz) exactly as production; then precompute
`W = B_mg C0` (3nz V-cycles, once) → `schur = sym(ass − C0ᵀ W)` →
`schur_inv = _symmetric_pseudoinverse`. Apply: `y = V(rhs_b)`;
`z = schur_inv (rhs_c − C0ᵀ y)`; `x_b = y − W z` — **one V-cycle per preconditioner
apply** (not two), and the Schur rebuild is free given W. Knob `--schur {rebuild,fd}`
(reusing the FD-based schur_inv is SPD-safe — the envelope is SPD for any SPD S⁻¹, B).

### Outer solve & measurement
`solve_singular_cg` with `vs = _nullspace_vectors(ops, 0, dirichlet)` (constant, free BC).
`jax.jit` around the whole solve closure (operators captured, compile over rhs); warm +
timed runs so ms/it is marginal cost (compile-overhead lesson from the mass benchmark).
Baseline = production `assemble_tensor_laplacian_preconditioner` + tensor-hodge apply.
`--levels 1` degenerates to Chebyshev-smoother-only PCG — free single-level comparison of
the three smoothers before MG enters.

## Cost model (expectations)

MG-PCG per iteration ≈ 6.9 fine-A units (m=2) / 4.4 (m=1) vs baseline ≈ 1.4 →
**break-even ≈ 5× (m=2) / 3× (m=1) iteration reduction**. Expect MG to lose wall-clock at
(8,16,8) and win at ≥(16,32,16) if h-flat. Setup (probes, W, Lanczos, coarse geometry)
reported separately in the CSV (`setup_*` columns).

## Validation matrix

`{cylinder, toroid, w7x} × {dbc, free} × ns {(8,16,8),(12,24,12),(16,32,16),(24,48,24)}
× smoother {jacobi, fd, fdax} × m {1,2}` at p=3, vs baseline; W7-X additionally
`--coarsen {2,2,2 (ζ-floored), 2,2,1}`; (24,48,24) additionally `--levels {2,3}`.

Success: (1) MG iters grow ≤ ~1.3× across the h-sweep (vs baseline growth); (2) wall-clock
beats baseline at (16,32,16)+; (3) no free-BC stall on toroid/W7-X (recomputed residual
consistent); (4) SPD checks pass. The smoother comparison answers: does local-exact
Jacobi beat both FD averagings, and how much does axis-averaging (fdax) recover vs the
1/3-version (fd)?

## Phasing

0. Script skeleton + level/transfer builders; `--two-level-check` on cylinder (8,16,8):
   SPD asserts, `P_const_err`, nested-periodic P exactness.
1. Two-level cylinder, dbc+free, all three smoothers, m∈{1,2} — gate: SPD pass, h-flat.
2. Toroid dbc+free — first curved-geometry + free-BC-stall test.
3. W7-X on GPU (`--zeta-diag` first; coarsen-policy sweep).
4. (24,48,24), levels 2 vs 3, `--schur rebuild` vs `fd`; full CSV for the final table.

Production wiring into `operators.py` (equinox factor leaves, `cp_kwargs={"mg_levels":…}`)
is **deferred** until Phases 1–4 validate.

## Verification

- Local quick check (CPU ok): `python scripts/debug/laplacian_mg_k0.py --geometry cylinder
  --ns 8 16 8 --two-level-check` → SPD asserts pass, MG-PCG converges, iters printed.
- Sweep: `bash slurm/job_laplacian_mg_k0.sh` (per-geometry jobs on gpu-h100s); compare
  `outputs/laplacian_mg_k0/<stamp>/*.csv`: MG vs baseline iters across ns (h-flatness),
  ms/it, setup costs, smoother ranking.
