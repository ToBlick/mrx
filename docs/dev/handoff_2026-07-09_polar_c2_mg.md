# Handoff — 2026-07-09 session: MG fat-core/anchoring/atoms + C⁰/C¹/C² polar surgery

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
