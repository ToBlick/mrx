# Rerunning the paper's experiments on the released code: what changes, what it costs (2026-09-11)

Tobias: "make a list of every experiment in the paper, what would be
different if we reran it today and what it would cost approx." Companion of
`paper_experiments_audit_2026-09-11.md` (the inventory and verdicts) and
`runtimes_rescale_2026-09-11.md` (today's per-step rates).

## What "today" means

The released configuration differs from the runs behind the paper in:

1. **Smoothing before the L-BFGS combination** (hard-coded). The paper's
   sweeps smoothed after it. Lower floor (velocity table: 2.08e-8 against
   5.72e-8 at 5000 steps, m = 1), different early transient, same helicity
   drift class.
2. **Potential velocity route** by default for L-BFGS (one k=1 Hodge solve
   instead of the Leray saddle and the shifted k=2 solve). Same descent to
   three digits (paper's own A/B), 26-30% cheaper per step with smoothing.
   Rerunning L-BFGS with it is the right call: it is what a reader gets.
3. **Faster solver core** (tpu-support merge): 1.4-1.9x per step at n <= 32,
   more at the hard rows once verified. Plus the refinement fix e680ab4
   (the early stop that left solves unconverged at n = 48, p >= 4,
   tol 1e-6 is gone); every rerun must be on or after e680ab4.
4. **Squared residual recorded directly**, floor 1e-8 in those units; the
   paper's conversions disappear.
5. **Newton is the default method** (100 steps, chunk 20, 300 MINRES): the
   sweeps pass `--method lbfgs --steps N --chunk 500`. **Preconditioner:
   the harmonic atom** (Tobias, 2026-09-11 16:00: "harmonic preconditioner
   reaches the floor faster -- that is what we want"; it reaches the floor
   in a fifth of the steps, then descends past it with the residual
   climbing). kappa = 1 (the newton session's arms: kappa = 1e-2 never
   reaches 1e-8 at (16,32,32), kappa = 1 is at 5.8e-9 by step 6; the
   window below 1e-8 is steps 6-10). To become the default in the code on
   Tobias's direct confirmation to the newton session; the Newton rows
   C1-C3 run with it, and their steps-to-floor and times shrink
   accordingly. kappa = 1 has never run at (32,64,64); its 100-step run
   there (newton session) is the confirmation the finest row needs.
   **No floor stop for the paper's runs** (Tobias, 16:30): the chunk-mean
   criterion misses a five-step window, chunk 5 is silly for Newton, an
   in-loop stop is a production question for later. Every rerun uses
   `--floor-tol 0` and runs to its step budget; the tables take the
   per-step trace (minimum and its step, steps and time to the
   thresholds), as `newton_table.py` already does.
6. **Explicit helicity correction available** (newton worktree 04f5cfe,
   `--helicity-correction true`, off by default): the explicit step
   conserves the discrete helicity to 1e-15 in float64. Changes what the
   helicity table can say (a fifth column, or the midpoint rows retired).
   **Decided (Tobias, 17:00): the midpoint scheme is omitted from the
   paper.** The helicity table becomes explicit x {mixed, float64} x
   {with, without the auxiliary field} plus the explicit step with the
   correction; the scheme option, the Picard sentence and the midpoint
   proposition go from the text. The four midpoint arms of the 2026-09-11
   rerun batch had finished (9-14 min each) before the decision; their
   records stay in `paper_rerun_2026-09-11/hel_mp_*` unused.
7. **Knot lists per axis** replace `--r-refine` (the refined mesh of the
   mesh-refinement figure is a breakpoint list now).
8. **Vacuum: one construction** (direct Hodge decomposition, Rayleigh
   1e-21) for every rung; the dual-form gap diagnostic should be off.
9. **Large meshes**: n = 48 needs 160 GB and about 31 min of scan compile
   on the current code (the paper's needed 20), n = 64 needs 320 GB; both
   need `--map-batch 8192`.

Rates used below (s/step, today, potential route where the paper row is
L-BFGS): (16,32,32) p=2 0.30; (12,24,24) 0.17; (24,48,48) 0.95;
(32,64,64) 2.2; n = 48 ~9; n = 64 ~25 (the last two are the paper's rates
divided by 2.5 and unverified); p=1 0.15, p=3 0.60, p=4 1.2, p=5 2.7;
order 0 at 16^3 0.22; float64 order 1 0.50, order 0 0.30; plain float32
0.09 / 0.05; Newton (16,32,32) 11.1, (24,48,48) 23.6, (32,64,64) ~42
(held); setup 3-10 min per job, compile 1-5 min except n >= 48.
Costs are GPU hours on one H100, setup and compile included, tracing of
Poincare sections included where the figure needs them (~0.2 h per set of
five planes at 16^3).

## The list

| # | experiment (paper item) | runs | different if rerun today | cost today | note |
|---|---|---|---|---|---|
| A1 | Analytic vacuum convergence (fig:analytic_vacuum, rates) | 104 rungs, p = 1..4, n = 4..64, two routes | Numbers identical to the digits shown (same spaces, same solves in float64); one construction throughout. | 2-3 h (solve-only; the diagnostics that made the old jobs hours are off) | Not needed for correctness; rerun only for uniform provenance. |
| A2 | VMEC vacuum convergence, high-res reference (fig:vmec_vacuum, 4.5e-5 floor) | 24 rungs, p = 2..4 | Floor unchanged (the three largest rungs already are the current construction). Timings become quotable: 19 s harmonic form at the floor rung. | 1.5 h | Same. |
| A3 | Low-res VMEC reference (8.4e-5 plateau) | 38 rungs | Unchanged. | 1.5 h | Could be dropped from the rerun; the sentence stands on the old data. |
| A4 | QA vacuum |B| figure and Poincare section | 1 solve + traces | Unchanged field; resolution gets recorded this time. | 0.3 h | Cheap, fixes the unrecorded resolution. |
| B1 | Floor study: 3 precisions x 2 smoothing orders (fig:precision_and_gamma_sweep_F, wall table, "33%") | 6 arms x 20 000 steps | Order-1 arms: smooth-first + potential route, lower floor, likely still coincide across precisions; order-0 arms: identical physics (verified 1.00), cheaper. New wall table. | 8 h (f64 2.8 + 1.7, mixed 1.7 + 1.2, f32 0.5 + 0.3) | Moderate. The float32 order-0 divergence is a property, will recur. |
| B2 | Smoothing-constant sweep (tab:epsilon_sweep) | 6 arms x 5000 steps | Optimum may shift with smooth-first (the sweep was done smooth-last); s/step column becomes meaningful (today's code). | 2.5 h | Moderate; the table's claim "flat optimum 0.02-0.064" should be rechecked under smooth-first. |
| B3 | h sweep n = 12, 16, 24, 32 (fig:n_sweep, helicity panel, Newton-table L-BFGS rows) | 20 000 / 18 000 / 10 000 / 5000 steps | Lower floors (smooth-first), potential route, new wall axis. The n^alpha of the cost changes (2.65-ish). | 8.5 h (0.9 + 1.5 + 2.6 + 3.1) | Moderate. |
| B4 | h sweep n = 48, 64 | 1140 / 390 steps | As B3, plus the compile and memory caveats (9). | 5-9 h (n48 ~3.5 + 1 compile; n64 ~3 + compile, unverified rates) | EXPENSIVE and uncertain; consider 500 / 200 steps. |
| B5 | p sweep p = 1, 3, 4, 5 (fig:p-sweep) | 62 000 / 10 000 / 5000 / 2500 steps | As B3; p = 4, 5 rates held until verified on e680ab4. | 8 h (2.6 + 1.7 + 1.7 + 1.9) | Moderate; p = 1 is the long one (62 000 steps). |
| B6 | m sweep m = 0, 5 (fig:lbfgs_sweep) | 18 000 / 17 000 steps | m = 0 identical physics (no combination to reorder); m = 5 smooth-first. | 3 h | Cheap. |
| B7 | tol sweep 1e-6, 1e-10 (fig:tol_sweep, spike story) | 32 000 / 14 500 steps | The 1e-6 spike at ~8500 steps may or may not recur under smooth-first + potential route (the potential route's k=1 Hodge solve at tol 1e-6 is a different failure mode than the saddle's); the 1e-10 arm as B3. | 2.8 h (1.3 + 1.5) | Moderate; the spike paragraph would need rewriting if it does not recur. |
| B8 | Velocity table (tab:velocity_choices) | 8 arms x 5000 | Smooth-last row gone; the four potential rows are today's default, the four Leray rows are the comparison. | 3 h (or 1.5 h potential rows only) | Cheap. |
| B9 | Helicity preservation table (tab:helicity_preservation) | 8 arms x 1000 steps (low-res reference file, order 0) | Same numbers if run as before; with `--helicity-correction true` the explicit rows drop to ~1e-15 (float64), which changes the table's message and may retire the midpoint. Should also be run on the 1.4m file for consistency. | 1.5 h (+1.5 h with the correction rows) | Cheap; decision on the table's content needed first. |
| B10 | Mesh refinement, three meshes, one solve at 5000 (fig:mesh_refinement, "2%") | 3 arms x 10 000 steps, (16,32,32), (32,32,32), (32,32,32) refined | Smooth-first, potential route, smoothing constant 0.02 instead of 0.064, refined mesh by knot list; widths after the solve should agree as before. | 6 h (0.85 + 2.5 + 2.5) + traces | Moderate. |
| B11 | Reconnection ladder (fig:ladder_relaxation_trace, sections, 3-D title figure, widths 0.45/1.2/3.0 h_r) | 40 000 steps, 4 solves | Smooth-first, potential route; the solves' doses unchanged (helicity accounting is route-independent); widths should track sqrt(dH) as before. | 3.5 h + 0.5 h traces | Moderate. |
| B12 | Seeded arms (fig:seeded_poincare_sections, tab:seeded) + island-pressure figures | 2 arms x 10 000 steps + 1 unseeded (the anchor) | Smooth-first, potential route; floors lower; widths +-5% claim rechecked; force-profile and pressure-map figures from the new end states. | 2 h + 0.5 h traces + 0.3 h profiles | Cheap. |
| C1 | Newton convergence table and figure, (16,32,32) | 220 steps (49 to the floor) | Unchanged method (Newton smoothed first already); only the times. Verified today to 1-5%. | 0.7 h | Cheap; could run only to the floor + 20 (0.3 h). |
| C2 | Newton (24,48,48) | 680 steps (419 to the floor) | As C1. | 4.5 h (2.8 h to the floor) | EXPENSIVE. |
| C3 | Newton (32,64,64) | 480 steps (325 to the floor) | As C1 once e680ab4 is confirmed stall-free; the 3/5-chain observation should hold. | 6 h (4 h to the floor) + compile | EXPENSIVE. |
| C4 | Newton floor Poincare sections | 3 trace sets | Unchanged. | 0.5 h | Cheap. |

Not experiments: the codes table, the landscape sketch, the diagnostics
definitions, the hyperparameter table (edit by hand).

## Totals

- Cheap, do regardless (A4, B6, B8, B9, B12, C1, C4): ~11 h.
- Moderate (B1, B2, B3, B5, B7, B10, B11): ~39 h.
- Expensive (B4, C2, C3): ~16-20 h, uncertain at n = 48/64.
- Vacuum provenance (A1-A3): ~5 h, optional.

Everything: ~75 h of GPU. Everything but the expensive three: ~55 h.
Sequencing: the cheap block first on e680ab4 (also settles the potential
route's floors), then B3/B5/B7 whose numbers the text quotes most, then
the rest. The L-BFGS sweeps stay in the paper as they are meant to; they
just run the released method.
