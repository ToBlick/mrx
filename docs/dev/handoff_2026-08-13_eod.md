# Handoff — 2026-08-13 end of day: k=0 shipped, k=1 wall broken, coupled atom wins toroid

One-day arc from "run the cluster experiment matrix" to a new k=1 atom
family. Running detail (every experiment, number, and dead end) is in
`handoff_2026-08-13_gpu_cluster.md`; this is the clean summary + queue.
Branch `greville-prod`, 11 commits `981326a..3b754f4` + the coupled-atom
commit(s) after; NOT pushed, NOT merged to main.

## TL;DR

1. **k=0 PRODUCTION SHIPPED:** fdbund atom swap (`MRX_K0_ATOM=fd` reverts)
   + positive-part `_symmetric_pseudoinverse` + fd-probed Schur. Verified:
   W7-X 64→54 dbc / 100→79 free; toroid 32→24 / 45→30; revert
   bit-identical. **MG: shelved** (research branch; wins only free-BC
   1.05–1.4× for ~2k lines).
2. **k=1 W7-X WALL BROKEN:** the historical stall = extreme κ(P·Ŝ), no
   bug. `MRX_K1_ATOM=rank1` (full rank-1 weights, exact per-axis pencils)
   converges W7-X dbc 2908 it / free ~7500 — first ever. Still ~6× jacobi
   wall there.
3. **NEW: coupled atom (`MODELS=coupled` / `--pa-block-coupled`,
   `scripts/benchmark/k1_coupled_atom.py`): first k=1 preconditioner to
   beat jacobi on BOTH BCs, iterations AND wall (toroid: dbc 203 it/277 ms
   vs jacobi 522/349; free 297/532 vs 743/814).** W7-X: coupled HALVES
   rank1 — dbc 1614 it @3.1e-10 (was 2908), free 3817 (was ~7500) — best
   atom on every cell measured; still 3.2×/2.4× jacobi wall there (jacobi
   948/2509 at ~half the per-it cost). The remaining ~3× is the measured
   top-outlier-deflation territory (queue item 2).
4. **Jacobi is done evolving** (dense probe: compact spectrum, κ 1.7e3
   dbc / 1.3e4 free on W7-X, no deflatable tail, no block structure worth
   the cost). **The cheap win is on the tensor side: top-outlier
   deflation** — rank1's κ=8e5 is carried by ~50 top modes (dbc; ~8 free)
   → one fast Lanczos + ~50-vector projection ≈ jacobi-κ territory at
   tensor iteration counts.

## The k=1 atom ladder (8³-class; toroid dbc/free its | W7-X dbc @2000)

| atom | toroid | W7-X floor | note |
| --- | --- | --- | --- |
| CP-ALS (old default) | 418/687 | ~1e-5 stall | legacy |
| bundled (`MRX_K1_ATOM=bundled`, new default) | 385/663 | ~1e-5 | rank-1 mean-field fits |
| profile (`=profile`) | 237/531 | 1.6e-5 | k=0-style, exact Lynch |
| greville-D (`MRX_K1_PA_GREVILLE=1`) | 303/**439** | 1.5e-6 | D=√(β_bb·β_cc) sandwich; June NaN fixed |
| radial_banded | 395/806 | 4.0e-7 | coupled r, legacy floors |
| rank1 (`=rank1`) | 280/479 | **9.7e-8 → CONVERGES** (2908 @8000) | Tobias's pencil theorem |
| **coupled** (`MODELS=coupled`) | **203/297** | **1614 @3e-10 (free 3817)** | C-terms exact, no-mixing weights |

## The coupled atom (what finally worked and why)

Per-tensor-mode 3×3 exact inverse keeping the inter-component C-terms:
- Channel sparsity (Tobias): diagonal block = 2 channels, off-diagonal
  (a,b) = single channel c∉{a,b} as ONE Kronecker term C_a⊗C_bᵀ⊗M_c.
- Paired de Rham eigenbases: pencil (M^N[m_c], K) per axis; W = gV^N gives
  V^D and modal cross factor √λ EXACTLY (banded incidence + de
  Rham-compatible windows: radial N-start 2 / D-start 1; radial D-window
  is ONE LARGER than N — complements fill dead λ-slots, remainder appended
  with zero-padded V^N columns → partner-less modes decouple naturally).
- **No-mixing weights (policy):** β_cc ≈ m_c(own-axis marginal, with
  magnitude, ξ₁ cutoff); derivative ladders UNWEIGHTED; never blend two
  metrics into one fitted profile (pairwise-shared variant exists in
  theory, kept only as an expected-to-lose A/B).
- Per-mode symbol B with analytic gradient null (s_r,s_θ,s_ζ); grad-div
  surrogate σ·n̂n̂ᵀ (σ=λ_r+λ_θ+λ_ζ) → SPD, batched 3×3 inv. Physics
  regularisation — ZERO floors (floors were the k=0-diagnosed disease).
- Apply = 6 einsum transforms + one batched 3×3 matvec; jit gotcha: the
  state pytree is TRACED through the solve — no int leaves, derive shapes
  from matrix .shape.

## Next-session queue (data-ranked)

1. Read the W7-X coupled result; if strong, wire coupled as the k=1 P_A
   default and rerun the 12,24,24 benchmark.
2. **rank1/coupled + top-outlier deflation** (~50 largest Ritz vectors,
   one Lanczos at setup): the measured shortest path to beating jacobi
   wall on W7-X dbc; composes with any atom.
3. Hybrid + D (greville-D won toroid free before coupled — check whether
   coupled already subsumes it; if not, D-sandwich over the coupled atom).
4. k=1 h-scaling investigation (tensor edge over jacobi shrank with
   resolution pre-coupled — recheck with coupled).
5. k=0 leftovers: fat-core/C² single-level (blocked on exact coupling
   probe), C² free-BC plumbing decision, toroid-free 6-mode deflation.

## Knobs added today

- `MRX_K0_ATOM` = fdbund(default)|fd — production k=0 atom.
- `MRX_K1_ATOM` = bundled(default)|cp|profile|rank1 — k=1 channel fits.
- `MRX_K1_PA_GREVILLE=1` — greville D-sandwich P_A.
- benchmark: `--pa-block-coupled` (launcher `MODELS=coupled`),
  `--dense-ps-spectrum` (per-method κ census), `--sl-eps-frac`,
  `--sl-trunc-frac`; MG prototype: `--levels 1` single-level mode,
  fdhel/fdslab atoms, low-tail spectrum census.
- Launcher gotchas: per-geometry shape args now built in (cerfon/rot-ell
  degenerate without them); second-resolution STAMP collision (sleep 2
  between launches); `--polar-order 2` requires `--bc dbc`.

## Dead ends (measured, do not revisit without new evidence)

MG in production; deflation of k=0 single-level (no tail); eps-shift and
truncated-pinv regularisation; fdhel-v1 (needs sheared-metric ζ-weight);
fdslab-in-MG; jacobi improvements (block/deflation — compact spectrum);
CP-ALS rank>1 stiffness fits; geometric-mean shared weights (no-mixing
policy); global-shared-w coupled class (superseded by pairwise theory,
itself defaulted to constants).

## 2026-08-14 addendum: gradient diagnostics close the k=1 single-level story

- Gradient-RHS probe: NULL (identical its to random) — P·Ŝ mixes subspaces.
- Grad-energy eigenmode decomposition (`--dense-ps-spectrum` now prints it):
  coupled atom's top outliers are PURE gradient (frac 1.00, λ to 8.6e4);
  bottom/bulk modes MIXED (0.5–0.6).
- σ A/B (`MRX_K1_COUPLED_SIGMA` lin|sq|inf, default lin): sq = identical
  dbc (1612), worse free; inf = FLOORS at 1e-2 (P_B cannot own gradients;
  the atom's gradient response is load-bearing). ⇒ top outliers are
  cosmetic for MINRES; the W7-X gap lives in the MIXED bulk.
- VERDICT: coupled atom is optimal in its class; the mixed smooth bulk is
  coarse-grid territory ⇒ **k=1 MG revival is the path** (economics invert
  at k=1: baseline 1600–3800 its). Smoother = Chebyshev-wrapped coupled
  atom (σ=lin); Hiptmair G₀-pass and the BC-flipped div-div third leg kept
  as smoother options (mixed-mode evidence favors trying the 3-space form);
  new code needed = per-component commuting transfers only.

## 2026-08-14: THE k=1 ANSWER — exact L0 in P_B/Pi (Tobias's block-diagonalization argument)

`MRX_K1_L0INV = dense | ns2` (make_apply_routines hook, projector-consistent
L0). W7-X 8,16,16, coupled atom, its (dbc | free):
single-apply L0 1614|3817 → ns2 826|2419 → **dense 154|172 @1e-11 —
3.3×/9.5× FASTER THAN JACOBI WALL (272/414 ms vs 907/3938).** The whole
W7-X k=1 wall was L0 fidelity inside P_B and the projector (the default was
ONE tensor-atom apply); with exact L0 the M1-orthogonal splitting
block-diagonalizes Ŝ and gradients sit at λ=1 exactly. Fidelity need is
STEEP (ns2 only halves) ⇒ production = DENSE L0 factorization (n0 is the
small space: 1.3k here, ~7k at 12,24,24 — one Cholesky, n0² matvec/apply)
now; FIXED symmetric V-cycles of the k=0 MG as the h-scalable successor
(stationary+symmetric = legal inside P_B; V-cycle count O(1) in h vs
Chebyshev degree ~√κ) — the shelved k=0 MG's true purpose. Next: wire
dense-L0 as a production option, validate 12,24,24, then k=2 recursion.

## NEXT BUILD (spec'd, Tobias 2026-08-14): P_A + P12 divdiv P12^T -- no L0 anywhere

Replace P_B/Pi entirely: P = P_A(coupled) + P12^T . B_div . P12, NO
projection (live with overlap), NO L0 solves.
- **P12 (V1 -> V2 proxy transfer, Greville collocation):** pointwise
  v2 = J g^{-1} v1 at each V2 component's Greville grid. 9 component-pair
  blocks, each = diag((J g^{-1})_ab at grid_a) x (E^ab_r ⊗ E^ab_t ⊗ E^ab_z)
  where E = 1D spline-evaluation (collocation) matrices of the V1
  component-b bases at the V2 component-a Greville abscissae (endpoint
  eps clip 1e-7 for the spline map). Adjoint = transpose. Fallback arm:
  L2 transfer with the k=2 mass preconditioner as approximate M2^{-1}.
- **B_div (k=2 div-div atom, built P_A-like = the coupled recipe at k=2):**
  single channel (weight 1/J, no-mixing marginal m(1/J) if any); per
  tensor mode the div-div symbol is RANK-1: B = t t^T with
  t = (s_r, s_t, s_z) in the k=2 paired ladders -> per-mode CAPPED pinv
  (invert t-direction, zero t-perp = the curls; the cap IS the pinv here)
  or + curl-curl surrogate sigma*(I - t t^T/|t|^2) for SPD-nonsingular.
- **BC FLIP (Tobias):** the k=1 weak div with essential BCs pairs with the
  STRONG k=2 div-div of the OPPOSITE BC -- index the auxiliary payload
  with `not dirichlet` or the spectrum mismatches at the wall.
- Wire as a P_B alternative flag (--pb-divdiv) beside MRX_K1_L0INV arms;
  compare on all four geometries vs dense-L0 (154-172-class counts) and
  mg2. Expected win: zero L0 solves, one collocation each way + one
  batched rank-1-pinv per apply -- the cheapest gradient machinery
  conceivable if the transfer holds on W7-X.

STATUS at close: final4 job's last cell (W7-X mg2 auto-m) still computing
-- read outputs/k1_pa_compare/dense_spec/final4.log. m=1 arm complete
(W7-X 204/546 -- even the crudest MG-L0 beats jacobi 4.6x its). All wiring
knobs: MRX_K1_L0INV=dense|ns2|mg{1,2}, MRX_K1_MG_M, MRX_K1_COUPLED_SIGMA.

### P12 implementation breadcrumbs (started)
- 1D eval: `SplineBasis.collocation_matrix(points)` (mrx/spline_bases.py:136).
- Metric weight at V2 grids: (J g^{-1})_ab = J*minv_ab from
  `compute_geometry_terms(seq.map, pts)` (k0 greville pattern, eps-clip 1e-7).
- V2 comp shapes: _k2_regular_component_shapes -> r:(N_r,D_t,D_z) etc.;
  V1: r:(D_r,N_t,N_z) etc. OPEN: locate the 1D D-basis OBJECTS (N 1D bases
  = seq.basis_0.Λ[axis]; D 1D analogs — check seq.basis_1/basis_2
  attributes or construct SplineBasis(n,p-1,type)).
- Module: scripts/benchmark/k1_p12_divdiv.py; wire as --pb-divdiv beside
  the L0INV arms; B_div per-mode rank-1 ttᵀ capped pinv, BC-flipped.

### FINAL four-geometry table (complete, job 16182715)
its (dbc/free), coupled P_A: cylinder dense 80/83 mg2 86/95; toroid 87/88,
89/94; rot-ell 90/91, 92/97; **W7-X dense 155/172, mg2 156/247** — the
MG-powered L0 matches exact-L0 to ~1 it on W7-X dbc. Jacobi: 403-948 dbc /
530-2504 free. Architecture validated end-to-end on all four geometries;
mg2 wall = unjitted prototype glue only.

## P12 div-div four-geometry verdict (job 16184910) + k=3 surgery

P = raw P_A + Pi21 B_div Pi21^T (no L0, no projection):
- cylinder 272/278, toroid 361/363, rot-ell 329/331 its @1e-11 — beats
  jacobi on BOTH BCs everywhere axisymmetric-class (wall too), trails the
  L0-based reference ~1.8×. THE TRANSFER WORKS.
- **W7-X: STALLS at 0.63/0.48 @8000** — the floor ≈ the curl fraction of a
  random RHS ⇒ scale imbalance: the g/J transfer weights inflate the P12
  term by orders on W7-X, the sum goes effectively gradient-only, curls
  unpreconditioned. FIX (v2): Lanczos-normalize the two terms' preconditioned
  λmax at setup (standard additive calibration, skipped in v1). Then the
  greville-B_div arm (MRX_K1_BDIV=greville, wired) as quality upgrade.
- k=3: transfer T = M^{-1}C (Galerkin, mass-precond legs); rank-deficient by
  the POLAR-EXTRACTION mismatch (V0-extracted loses ~2 n_t n_z − 3 n_z axis
  DOFs that V3 keeps) → 0.54 floor. AXIS-CORE SURGERY implemented
  (MRX_K3_CORE=R: dense-probed leading-R-ring block, positive-part pinv
  with 1/λmax floor, added to the transfer) — 4-geometry test in flight
  (job 16185831).

## k=3 transfer: the missing-subspace ladder (2026-08-14, cylinder)

Pure transfer P_3 = T L0^-1 T* floor: 0.54 (bare) -> 2.4e-2 (axis surgery
MRX_K3_CORE=2; R-scan 2-5 flat => axis fully captured at R=2) -> 3.8e-3
(+wall ring MRX_K3_WALL=1: the dbc-dropped outer V0 function's territory,
6x drop, 96 its and descending faster than jacobi-to-tolerance). axis0
projector arm = NO-OP (zeroing V0 cores can't extend the reachable set;
Tobias's constraint idea needs the re-solved axis-Dirichlet L0 if ever).
Remaining ~4e-3 residue: candidates = 2nd wall ring, harmonic path --
same probe cycle. Verdict: the k=3<->k=0 duality WORKS once each
structural subspace (axis extraction/J-constraint deficit + BC-flip wall
deficit) gets its small probed core; pure-transfer descent rate beats
jacobi decisively (96 its to 3.8e-3 vs 193 to 1e-10).

## PRODUCTION POLICY (Tobias, 2026-08-14 — supersedes all preconditioner
## recommendations above for production use)

- **Masses (all k): Kronecker/tensor preconditioners.** Measured excellent;
  no change.
- **k=0 Laplacian: the shipped tensor-hodge preconditioner stands** (fdbund
  swap, verified 4-geometry, MRX_K0_ATOM=fd reverts).
- **k=1, k=2, k=3 Laplacians: Schur-outer JACOBI.** Rationale: nothing
  beats jacobi in production-trustworthy wall-clock on W7-X today. The
  coupled-atom + dense-L0 result (154/172 it, 3-9x prototype wall win) is
  real but prototype-grade; MG-L0 wall numbers were unjitted glue; P12
  fails W7-X pending term-scale calibration; k=3 transfer still floored.
  Jacobi = robustness + zero new complexity. Block/line-jacobi and
  Schwarz measured or argued unproductive; Vanka untested (same cost
  family).

**Shelf (validated research, with reopen conditions):**
1. k=1 coupled P_A + dense/MG L0 in P_B/Pi — the math risk is RETIRED
   (geometry-independent ~80-172 it, all four geometries). Reopen when k=1
   solves bottleneck production: the remaining work is jit + production
   wiring of dense-L0 (n0 small), not research.
2. P12 div-div (no-L0): wins 3/4 geometries; needs Lanczos term-scale
   calibration for W7-X. 3. k=3 transfer: subspace ladder documented;
   surgical route at 3.8e-3 (best); UNEXTRACTED-tensor route REFUTED as
   built (floor 0.62 survives the window fix AND the xi1-cutoff fix --
   suspect: conditioning of the square N-at-D-greville collocation
   pairing; check via SVD of the three E matrices before reviving). 4. MG-as-L0:
   iteration-validated (== dense within ~10%); wall needs jitting.

## Relaxation-class ledger (2026-08-14, closing the standard-methods question)

- **l1-jacobi (row-norm realization): REFUTED** — trails plain jacobi 10%
  (toroid dbc 583 vs 522) to 30% (W7-X free 3267 vs 2508) everywhere;
  minor robustness edge at the tolerance boundary only. Exact-Kronecker l1
  not worth building against a 30% deficit. (True row-l1 IS exactly
  computable here via Kronecker factorization of |1D| row sums, recorded
  for reference.)
- **mass-as-Laplacian-preconditioner: REFUTED** — toroid dbc 3767 it vs
  jacobi 522 (7×); metric-blindness on the stiffness side, as theorized.
- **MG(mass smoother): REFUTED by the window measurement alone** — the
  auto-m rule printed smoother window κ ≈ 9.7e5 (W7-X) / 5.9e5 (toroid),
  demanding m ≈ 1390/1090 Chebyshev steps per pass (vs the fdbund atom's
  κ 3–7, m 2–4): the ENTIRE metric contrast lands in the mass smoother's
  window. Jobs cancelled after the κ print; the number is the finding.
  (Mass smoothing's spectral-element niche = p-robustness with smooth
  coefficients — the polar metric is the opposite regime.)
⇒ Relaxation/smoother ledger CLOSED: plain jacobi is measured-optimal in
its class at k≥1; the k=0 tensor-hodge and (shelved) atom family remain
the only things that beat it, and they do so by absorbing the metric.
