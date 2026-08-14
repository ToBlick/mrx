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
