# Handoff — 2026-07-09 session: MG fat-core/anchoring/atoms + C⁰/C¹/C² polar surgery

## Addendum — 2026-07-24 session: off-diagonal lever REFUTED; smoother-question endgame

Dense model-matrix ladder (`verify_hodge_massprecond_k0.py`, new section in
`docs/dev/k0_massprecond_surgery_findings.md`, logs
`outputs/k0_offdiag_ladder/`):

- **The 07-22 "open lever" (off-diagonal `g^{rθ}`) is REFUTED.** Dropping ALL
  off-diagonals costs κ 1.5→2.6 (helical ns=6); the atoms sit at ~12. Adding a
  ζ-avg `g^{rθ}` cross to fdbund gains nothing (12.05 vs 12.01).
- **Real lever: averaging/separability error of the DIAGONAL weights**, on
  helical specifically the ζ-variation (ζ-avg 2D-per-mode candidate: 8.0 vs
  pointwise-diag 2.6 vs fdbund 12). The atoms' h-growth is averaging error
  (pointwise rungs nearly h-stable). No practical single-level atom upgrade
  pays → atom-sophistication line CLOSED; remaining spread belongs to the
  V-cycle coarse correction (or per-ζ-slab block-Jacobi, unmeasured — only
  worth it if the smoother A/B shows a gap).
- **Smoother endgame DECIDED: jacobi is OUT, smoother question CLOSED.**
  A/B (`scripts/debug/run_mg_k0_jacobi_ab.sh`, results
  `outputs/laplacian_mg_k0/jacobi_ab_20260724/`): MG(jacobi) needs 47–180 its
  vs MG(fd/fdbund) 5–15 at ~equal ms/it (A-applies dominate) — and even loses
  to the single-level baseline. Window tuning is flat in total A-applies.
  Cause: bulk-wide polar anisotropy (g^tt ~ 1/r², ratio 1..36), the classic
  point-smoother failure; the fd atoms' averaged θ-stiffness captures it.
  With the ladder (fancier ≤1.5× κ at real cost) ⇒ fd-family separable atom +
  exact core surgery is the sweet spot from BOTH directions. fd vs fdbund in
  MG: tie (atom switch stays W7-X-gated). MG-vs-baseline payoff stays the
  fine-h/W7-X (cluster) question: baseline wins wall-clock at 8³ (~3 vs ~20
  ms/it) but its its grow (19→40 dbc) while MG(fd) stays ~5–15.

## Addendum — 2026-07-22 session: k=0 metric handling, axis surgery, atom choice

Dense condition-number study (`scripts/debug/verify_hodge_massprecond_k0.py`,
full write-up `docs/dev/k0_massprecond_surgery_findings.md`). Key conclusions:

- **Production `apply_laplacian_preconditioner(k=0, tensor)` = `fd`-bulk + exact
  `3·nz` polar-core Schur surgery.** The bulk is *literally* the `fd` atom
  (geomean `D`, unweighted atoms). The core surgery (not a better bulk model) is
  what gives it `kappa 2–17`. Framework cross-check: reconstructed `fd+surgery`
  reproduces the production column to a few %.
- **Metric-lumping resolves by region:** off-axis lumping works (anisotropy
  bounded); on-axis (`g^tt ~ 1/r²`) is not lumped but solved *exactly* by the
  core surgery. The bulk metric model is **second-order** to the surgery.
- **The "B" idea (route K₀⁻¹ through the mass preconditioner) is CLOSED.**
  `B = L₀⁻¹ G₀ᵀ P₁ G₀ L₀⁻¹` is bulk-global curl-leaky (regular-decomposition
  constant), `kappa ~ 100–1000`, grows with h. **Axis surgery does NOT rescue it**
  (393→391) — its error is bulk-global, not axis-local. The `M_id`-weighted
  variant `B'` (reference-domain FEM Laplacian `L' = G₀ᵀ M_id G₀`, separable →
  FD-exact) improves B 2.6–6× and flattens it across geometry, but stays
  `kappa ~ 100`, still dominated. Do not pursue routing through the mass matrix.
- **`fdbund` beats `fdax` (adopt); `fdbund` vs production `fd` is DEFERRED to a
  real W7-X run — do NOT switch the shipped atom yet.** With exact core surgery
  fixed, `fdbund > fdax` on every geometry (bundled `g^tt J ~ 1/r` milder than bare
  `g^tt ~ 1/r²`). `fdbund` is **better-motivated** (represents each directional
  weight; geomean is one isotropic scalar that gets *none* right pointwise) — BUT
  the local evidence does not support switching `fd`: `fd` wins on toroid
  (2.41 vs 2.53) AND on the helical rotating-ellipse (10.87 vs 12.01, the closest
  local W7-X proxy); `fdbund` wins only on cerfon (4.43 vs 4.70). The
  rotating-ellipse ellipticity proxy **cannot reach strong anisotropy** — the map
  folds at `kappa=2` (min J≤0), valid ceiling `kappa=1.5` is only mildly
  anisotropic (raw κ≈969 ≈ toroid). So the strong-anisotropy payoff is UNCONFIRMED
  and the one helical point leans to `fd`. **Gate on a real W7-X (cluster) run**
  (to be done later); if `fdbund` wins there, change
  `_assemble_k0_greville_bulk_factors` (`mrx/operators.py`) to per-axis
  bundled-average weighted stiffnesses (`D=1`) + production convergence tests.
- **Open lever for the shaped/helical `kappa` growth:** the off-diagonal
  `g^{rθ}` (turned on by shaping/helicity) is discarded by *every* current bulk
  atom (fd/fdax/fdbund keep only the diagonal). Pursue it **directly** in the atom
  (off-diagonal coupling, or the rank-2 CP "radial_dense" path), surgery fixed —
  NOT via `M₁` (which reaches it only bundled with the fatal leakage).
  **(REFUTED 2026-07-24 — see the addendum above: the off-diagonal is cheap to
  drop; the real error is diagonal-weight averaging.)**

---

**One-paragraph summary.** Local (laptop CPU, 8³) measurement day for the
k=0 MG line, followed by implementing the C²-on-axis surgery. Fat-core
(ring 2 folded into the exact Schur core) absorbs the dominant axis share
of the smoother-atom spread on all geometries (κ up to ×2.15 down);
anchored-ξ₁ kills the axis-side transfer defect (P_const_err 5e-3 → 8e-16);
the atom decision is REOPENED (post-fat fdax wins or ties everywhere
locally — the W7-X "fd wins" call predates fat-core); the g·J-bundled atom
(`fdbund`) wins on axisymmetric, loses on helical dbc. Then the C² polar
extraction was derived (jet matching; Hessian enters via ring 1 × ρ =
2N₁′(0)²/N₂″(0)), implemented as collocated-C² (exact C² at fixed degree p
is impossible — product-of-splines obstruction; Toshniwal et al. resolve
it with angular degree ≥ 6, we collocate at Greville angles), verified
(pole Taylor order exactly 3.00; V_C² ⊂ V_C¹ at 1e-16), and
convergence-validated (L2 errors IDENTICAL to C¹ to all digits, rates
~4.4, on m=0/1/2 manufactured solutions incl. the ring-2-critical m=2).
C⁰ came free (`polar_order=0`) and also converges at full order.

## Read these first

1. `docs/laplacian_mg_k0_plan.md` — the living plan; today's results are
   in the "Local fat-core + spectrum sweep (2026-07-09)" section (items
   1–8) and the C² block below it.
2. `docs/polar_pole_regularity.md` — full C⁰/C¹/C² theory + validation
   write-up (derivation, obstruction, deviation-from-paper rationale,
   code map, convergence table).
3. Outputs/logs/CSV: `outputs/laplacian_mg_k0/local_fatcore_20260709/`
   (per-run logs, `results.csv`, convergence logs).

## Working tree state (ALL UNCOMMITTED)

Suggested commit split:

1. **mrx core**: `mrx/derham_sequence.py` (knots=(T_r,T_θ,T_ζ) API,
   polar_ring1, polar_order∈{0,1,2}), `mrx/extraction_operators.py`
   (get_xi generalized to map-adapted ring1 + ring1_control_points;
   NEW get_xi2; PolarExtractionOperator generalized to (n_polar,
   ring_depth) with k≥1 guarded to C¹), `mrx/operators.py` (grad-stencil
   skip guard for polar_order≠1).
2. **validation scripts**: `scripts/debug/verify_c2_polar.py`,
   `scripts/debug/poisson_k0_c2_convergence.py` (both rerunnable, exit
   nonzero on failure).
3. **MG prototype**: `scripts/debug/laplacian_mg_k0.py` (`--fat-core R`,
   `--spectrum-diag`, `--anchor-xi1`, `--xi-adapt`, `fdbund` atom, cerfon
   wiring), `scripts/benchmark/benchmark_graddiv_k1_preconditioner.py`
   (cerfon geometry, polar_ring1 passthrough),
   `scripts/debug/run_mg_k0_local_fatcore.sh` (sweep driver).
4. **docs**: `docs/laplacian_mg_k0_plan.md`,
   `docs/polar_pole_regularity.md`, this handoff.

## Key numbers to remember (8³, auto rule, dbc/free)

- Fat-core R=1 λmax(S·A): toroid fd 2.33→1.61, fdax 3.81→1.83; cerfon
  fd 3.13→2.28, fdax 4.02→1.96 (fdax now BEATS fd on shaped axisym);
  rot-ellipse fd 4.36→3.54, fdax 7.29→3.39 (helical washout was mostly
  the axis interaction). R=2: diminishing (~√((k+1)/k)).
- fdbund: toroid fat+anch 1.51 (5/5 free — best row); rot-ellipse dbc
  15 it (worst) — low-λ tail on helical, geometry-dependent option.
- Anchoring: P_const_err → 7.8e-16; iteration-neutral at 8³ two-level.
- xi-adapt: solver-invariant BY CONSTRUCTION (bulk block never sees ξ);
  payoff = near-axis accuracy (unmeasured, needs MMS study).
- C² Poisson: C¹/C² L2 identical to all digits, rates 4.44/4.32/4.39
  (m0/m1/m2); C² 10–16% fewer DOFs, fewer CG its; C⁰ same order.

## Next-actions queue (priority order)

1. **Wire `--polar-order {0,1,2}` into the MG prototype** and A/B κ
   against fat-core R=1 (toroid/cerfon/rot-ellipse, 8³). Needs: bulk
   window starts at ring `1+order` (reuse the ring0 threading — C² core
   ≙ window start 3, same as fat R=1), envelope probes the 6nz core
   (ass+C0 probe path already exists from fat-core), production
   `k0_tensor_hodge_precond` NOT usable as baseline arm for order≠1
   (hardcodes 3nz core) — baseline can stay C¹-order runs. C⁰ arm =
   control for "core size vs pole regularity" in the spread.
2. **m=1 smoothing A/B** on fat/C² configs (auto-m floor 2→1, one-line):
   at κ ≈ 2, a single damped atom apply may suffice → 3 A-units/cycle,
   break-even ~3×.
3. **Cluster (when access returns)**: W7-X rows with `--fat-core 1` —
   re-decide the atom (expect fdax κ 11.7 → ~5–6); the (16,32,32) rows +
   λmax ≈ 8.7 prediction check; full-space MG experiment (kills the
   W-probe, makes bigger cores ~free).
4. **MMS accuracy study for xi-adapt** (cerfon, near-axis L2/order vs
   circle-ξ) — the claim that map-adapted ξ buys accuracy is untested.
5. Deferred/backlog: C² de Rham for k≥1 (Bernstein-identity rework of
   E¹/E², few hundred lines); per-ζ-plane ξ (stellarator map adaptation);
   `get_xi2(ring1, ring2)` map-adapted C² (currently circle-only —
   NotImplementedError guard when polar_ring1 given); production wiring
   notes in the plan doc (PSD pseudoinverse, r_scale=0.5, auto window
   rule, coarsest-Cholesky + explicit deflation).

## Rerun commands

- MG sweep: `bash scripts/debug/run_mg_k0_local_fatcore.sh` (~15 min);
  single run: `python scripts/debug/laplacian_mg_k0.py --geometry cerfon
  --kappa 1.7 --alpha 0.4 --ns 8 16 8 --smoothers fd,fdax --cheb-lo 0.85
  --auto-m --two-level-check [--fat-core 1] [--anchor-xi1] [--xi-adapt]
  [--spectrum-diag]`
- C² checks: `python scripts/debug/verify_c2_polar.py` (~1 min);
  `python scripts/debug/poisson_k0_c2_convergence.py --ns 4 6 8 12
  --orders 0,1,2` (~45 min full, `--ns 4 6 --cases m2` for a smoke).

## Gotchas rediscovered today

- Background Bash tasks are killed at 10 min — long sweeps need
  nohup+disown with their own log file.
- `assemble_laplacian_operators` eagerly warms the production FD
  preconditioner → crashes on polar_order≠1 sequences; assemble only the
  incidence (`assemble_incidence_operators(ks=(0,))`) for C²/C⁰ runs.
- `solve_singular_cg` takes `vs=[]`, not `vs=None`.
- P_const_err check reduces over an empty array when the bulk radial
  window ≤ p (fixed: skip + nan).
