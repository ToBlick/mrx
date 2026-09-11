# The paper's experiments, mapped to their runs, and what is still reliable (2026-09-11)

Tobias: "compose a list of all experiments that were run for the paper, and
which numbers are still reliable and which are not." Source
`outputs/figures_2026-09/resubmit-4.tex` (mtime 2026-09-11 10:05), the run
records under `outputs/` of the main checkout and of the `vmec-axis-guard`
and `newton` worktrees. Inventory by a search agent (49 entries, every
figure, table and quoted number, with the relax.json settings); the
verdicts are mine, from `runtimes_rescale_2026-09-11.md` and the
re-measurement of today.

Reference settings ("REF") of the relaxation studies: li383
`data/wout_li383_1.4m.nc`, (16,32,32), p = 2, mixed precision (tol 1e-8),
L-BFGS m = 1, explicit, smoothing order 1 at 0.02/n_r^2, CFL 0.5, Leray
route, chunk 500, no floor. Every REF run smoothed AFTER the L-BFGS
combination ("smooth last", `smooth_first` absent); the released code
smooths first.

## Three facts that decide most verdicts

1. **Physics results stand.** Residuals, energies, helicities, widths,
   rates and Poincare sections were produced by the code of their date;
   nothing that happened to the code since changes a record. Where the
   current head was checked against those records it reproduces them
   (order-0 rows, Newton (16,32,32), the smooth-first Leray arm), so the
   old runs were not victims of a bug the head fixed.
2. **Every timing is stale.** The solver core got 1.4-1.9x faster on
   2026-09-10 (the CG carry and refinement changes of the tpu-support
   merge); per-row factors measured today are in
   `runtimes_rescale_2026-09-11.md`. Rows at (32,64,64), n = 48, p = 4, 5
   and tol 1e-6 are held: at the head of this morning (b0f8d38) the
   refinement's early stop (a395a65) leaves solves unconverged there;
   removed in e680ab4, reruns pending.
3. **The method described is not the method run.** The h/p/m/tol sweeps,
   the floor study, the smoothing sweep, the seeded arms, the ladder, the
   mesh-refinement arms and the L-BFGS rows of the Newton table all
   smoothed last. Only the velocity table's "smooth first" rows and
   today's runs smooth first. If the paper restricts to one form, it must
   say the sweeps ran the other one, or those studies are rerun.

## The list

Verdict codes: **OK** reliable as printed; **T** timing, stale, rescale;
**M** method description differs from the run (numbers stand); **X**
inconsistency or unverifiable; **?** cannot be checked here.

| # | item | run(s) on disk | settings as run | verdict and what to change |
|---|---|---|---|---|
| 1 | fig:ncsx_3d_sections (title figure) | `figures_2026-09/mesh3d/`, sections of the ladder run `reconnect_2026-09/ladder/h16_p2` | REF, 40 000 steps, four resistive solves | OK. Caption says "intentional reconnection": it is the ladder's end state. |
| 6 | fig:vacuum-qa-bmag | `qa_vacuum/figures/vacuum_qa_Bmag.pdf` (2026-09-08) | caption (32,64,32) p=3; the folder's tex snippet says (12,24,12) p=3 | X: the resolution of the plotted field is not recorded; generator `scripts/plot_vacuum_qa.py` gone. State whichever you plotted. |
| 7 | fig:vacuum-qa-poincare | `qa_vacuum/figures/poincare/poincare_zeta0.25.pdf` (2026-09-08) | no run record | ? : sections exist, tracer settings and the field's resolution not recorded. |
| 8-10 | analytic vacuum: DoF counts, rates 0.96/1.97/3.02/4.07 and 0.94/1.97/2.93/3.92, fig:analytic_vacuum | `analytic_vacuum_pscan/*/analytic_vacuum.json`, `figures_2026-09/vacuum_analytic_convergence.json` | QA lowres map, coil field, n = 8..64, p = 1..4, float64 | OK for the rates (json fits 0.956/1.975/3.017/3.889 and 0.942/1.973/2.931/3.925: the k=1 p=4 rate is 3.89 in the json against 4.07 in the caption, from the n=24/28 rows that differ between json and `tables.tex`). X: "8x16x168" is a typo for 8x16x8; the figure file itself is not on disk here. No timing quoted. |
| 11-13 | VMEC vacuum: floors 8e-5 / 5e-5, fig:vmec_vacuum caption 4.5e-5 and 8.4e-5 | `qa_vacuum/`, `qa_vacuum_highres/rung_*/`, `figures_2026-09/vacuum_vmec_convergence.json` | float64; rungs <= 33x66x33 by the 2026-09-01 construction (inverse-iteration polish, Rayleigh 1e-11), the three largest by the direct construction (1e-21) | OK: D is a property of the space, both constructions reach the same floor (4.33e-5 / 4.46e-5 / 4.61e-5 at the largest rungs, 4.76e-5 mean). Today's re-solve of the floor rungs on the head: harmonic form in 19 s, force residual 1e-26. The figure file is not on disk here. |
| 17 | NCSX parameters, initial residual 1.6e-2 | `ladder_widths.json` ic.resid 0.0157 (unsquared) | | OK; the 1.6e-2 is the UNSQUARED residual while the paper's tables are squared (2.5e-4). Say which. VMEC-file numbers not checked. |
| 18 | reference configuration sentence | `sweeps_2026-09/h/h16_p2` | REF | M: add "smoothing applied after the L-BFGS combination" or rerun. |
| 19 | line 1244 "timings include diagnostics, 1-5%" | | | X: the opposite; the stepping wall excludes the diagnostics and includes the first chunk's compile, 1-4% (Newton), <1% (L-BFGS). |
| 20 | fig:precision_and_gamma_sweep_F (floor study) | `floor_study/{f64,mixed,f32}_{g1,g0}` | REF x precision x order, 20 000 steps | OK (physics). M for the order-1 arms (smooth last). The order-0 rows are reproduced by the head to 1.00. |
| 21-22 | smoothing constant 0.02, tab:epsilon_sweep | `mu_sweep/{g0,c*}` | REF, 5000 steps, order 1 at c/n_r^2 | OK for residual, energy, helicity. T for the s/step column (0.40-1.11): not re-measured except c = 0.02 (0.69 -> 0.41); drop the column or rerun the sweep's timing. M (smooth last). |
| 23-24 | "mixed gains 33%", tab:precision_and_gamma_sweep_wall | floor study walls, `tol/h16_p2_tol1e-10` | | T: today 0.67 / 0.49 / 0.41 / 0.13 (order 1), 0.41 / -- / 0.25 / 0.06 (order 0); the gain is 27%. |
| 25 | helicity bound 1e-5 across sweeps, p=1 1e-4, tol 1e-6 +3e-5 | `sweeps_*.json` | | OK. |
| 26 | (64,128,128), 1e6 DoFs | `sweeps_2026-09/h/h64_p2` | 390 steps | OK (3.03e6 k=2 DoFs). |
| 27 | fig:n_sweep, s/step 0.33...62.1 | `sweeps_2026-09/h/*` | REF, n = 12..64 | OK traces; T: today 0.229 / 0.411 / 1.29 / 2.98 for n = 12..32, n = 48 and 64 held (n=48 measured 5.3 on the faulty stop, invalid). M. The x axis of this figure is wall time: regenerated with today's rates in `figs/` of the rescale note. |
| 28 | fig:p-sweep, s/step 0.21...4.92 | `sweeps_2026-09/p/*` | REF, p = 1..5 | OK traces; T: 0.197 / 0.411 / 0.814 for p = 1..3; p = 4, 5 held. M. |
| 29 | cost ~ n^3.1, (p+1)^2.8, 1.2-2.2 us per quadrature point, Jacobi-PCG counts ~ n^1.3-1.6, p^2-2.3 | no source file found for any of the four | | X and T: the exponents and the per-point cost have no record; refit today n^2.65 (12..32), (p+1)^2.6, 0.8-1.5 us. The iteration-count exponents were not found anywhere; the h-scaling note has kappa ~ n^1.7. Either recover the source or drop the sentence. |
| 30 | fig:mesh_refinement, widths agree to 2% | `li383_pulse/reconnect_l5_{h16,h32u,h32r}_p2_g1` | (16,32,32) / (32,32,32) / (32,32,32) radially refined, smoothing 0.064/n_r^2 (pre-09-05 constant), one solve at 5000, 10 000 steps | OK for the widths (0.049 / 0.050 / 0.050 after the solve). M twice: smooth last AND the older smoothing constant; the radial refinement was `--r-refine`, now knot lists. Figure file not on disk here (nearest `li383_pulse/figures/mesh_ladder_noaxes/mesh_2d`). |
| 31 | fig:lbfgs_sweep, s/step 0.65 / 0.64 / 0.69 | `sweeps_2026-09/m/*` | REF, m = 0, 1, 5 | OK traces; T: 0.43 / 0.41 / 0.44. M. |
| 32 | Sec 6.3, tab:helicity_preservation | `midpoint_sweep/{ex,mp}_r16_{f32,f64}_{Hd,bonly}` | **`wout_li383_low_res_reference.nc`** (not the 1.4m file), (16,32,32) p=2, **smoothing order 0**, 1000 steps, floor 1e-4 | OK numbers. X: caption says only (16,32,32) p=2; the geometry file and the absence of smoothing differ from REF and should be stated. No timing. Note the head now has an explicit helicity correction (newton worktree 04f5cfe) that makes the explicit step conserve to 1e-15 too; this table predates it. |
| 33-34 | potential-route sentences, tab:velocity_choices | `newton/.../potential_relax/*`, `sweeps h16_p2`, `m0`, `mu_sweep/g0` | REF, 5000 steps | OK physics. T: today Leray/potential 0.411/0.303 (m=1, g=1), 0.428/0.303 (m=0), 0.248/0.224 (m=1, g=0), 0.227/0.224 (m=0, g=0); saving 26% / 29% with smoothing, 10% / 1% without (was 30% / 20% / none). The "smooth last" row goes if one form is kept; the remaining Leray m=0 g=1 row and the m=1 g=0 row are smooth-last runs (m=0 has no combination to smooth after, so only the g=1 rows care). |
| 35-36 | fig:tol_sweep, s/step 0.33 / 0.64 / 0.81, the tol 1e-6 spike story | `sweeps_2026-09/tol/*`, `tol/h16_p2_tol1e-6/spike/` | REF x tol | OK physics (the spike, widths 0.43/0.35/0.29 h_r, 3.6e-5 H0, 600x). T: today 0.193 (1e-6, held: not reproduced on the faulty stop) / 0.411 / 0.486. X: the right panel (dE, dH vs step) has no generated figure on disk; only dH exists. M. |
| 37 | resistive rate formula, width ~ sqrt(dH), beta drop 5-7 per 1% | `helicity_eta_derivation.tex`, ladder records | | OK (ratios 5.6-7.4 in the ladder). |
| 38-39 | fig:ladder_relaxation_trace, fig:ladder_poincare_sections, widths 0.45 -> 1.2 -> 3.0 h_r | `reconnect_2026-09/ladder/h16_p2` | REF, 40 000 steps, solves at 8k/16k/24k/32k for 2.5% (spent 2.25-2.32%) | OK. M. No timing quoted (0.594 s/step in the record). |
| 40 | Sec 7.2 island pressure numbers | `figures_2026-09/{force_profile,pressure_map}_s{61,51}_e1e-2`, seeded arms | seeded arms at step 10 000 | OK. X: `\ref{fig:force-profile}` and `\ref{fig:pressure-map}` have no figure environment in the tex (the PDFs exist). |
| 42-43 | fig:seeded_poincare_sections, tab:seeded | `seeded_2026-09/s61_e1e-2`, `s51_e1e-2`; unseeded = the sweep anchor h16_p2 (first 10 000 steps; `seeded_2026-09/unseeded/relax.json` is a dangling symlink) | REF + seed, 10 000 steps | OK physics (2.27 / 2.93 / 8.36 e-8; widths +1.9% / +4.7%, i.e. within the "+-5%"). T: the seeded_table.tex s/step column 0.654 / 0.655 / 0.629 -> 0.41 for all, or drop it. M. |
| 44 | Newton settings sentence | Newton records | tol 0.1, 300 MINRES, dt cap 1, Laplacian atom | OK. |
| 45 | fig:newton (residual vs wall time) | `newton_convergence_figure.py`, arms below | | OK shape and the "one tenth" ratio (rate-independent); T: x axis stale, regenerated with today's rates for 16 and 24; 32 held. |
| 46 | tab:newton_convergence | L-BFGS: `sweeps h16/h24/h32`; Newton: `newton_relax/n0_from0`, `h24_newton_from0(+cont,+cont/cont)`, `h32_newton_from0(+cont)` | Newton: (16,32,32) from a converted step-0 checkpoint, 300 MINRES, chunk 20, float32-refined = mixed | OK: steps, floors, steps-to-threshold, dH. T: every s/step and time column; rescaled table in the rescale note (16 and 24 usable; the (32,64,64) row of both methods held). X: (24,48,48) Newton s/step 38.2 in the tex, 38.4 in the generated file. The L-BFGS rows smoothed last (M) while the Newton rows smoothed first (Newton's fallback), which is fine to state. |
| 47 | fig:newton_poincare_sections | `figures_2026-09/newton_floors/*` from checkpoints 60 / 420 / 340 | | OK; note the sections are of the chunk checkpoints nearest the floor steps 49 / 419 / 325, not of the floor steps themselves. |
| 49 | tab:hyperparameters | code defaults | | X: Newton row "(0.1, 100, 1)": the MINRES budget is 300 (100 is the relaxation-step default); chunk default is 500 for L-BFGS and 20 for Newton; "restart none" fine. Picard counts 2 / 4 match `midpoint_sweep/summary.md`. |

Items 2-5, 14-16, 48: no experiment (literature table, code description,
landscape sketch, diagnostics definitions, future work).

## Figure files the tex includes that are not on disk here

`analytic_vacuum_convergence.pdf`, `vmec_vacuum_convergence.pdf`,
`degree_sweep.pdf` (the h sweep), `p_sweep.pdf`, `lbfgs_sweep.pdf`,
`tol_sweep.pdf`, `tol_sweep_dE_dH.pdf`, `mesh_2d_bare.pdf`. Presumably in
the paper repository's `figs/final/`; the sweep panels correspond to the
four panels of `sweeps_wall_residual_10step` and of `sweeps_helicity`
(no dE panel exists anywhere). Cannot be checked from here.

## What to do, by effort

- No runs: fix 19, 29 (drop or refit), 40 (labels), 49 (table), 8 (typo),
  17 (squared vs not), 32 (state the file and order 0), 6 (state the
  resolution), 46 (38.4); state the smoothing order of the sweeps (18);
  replace the timing numbers of 23-24, 27, 28, 31, 33-35, 42-43, 46 with
  today's for the verified rows; drop the epsilon-sweep and seeded s/step
  columns or mark them "smooth last, 2026-09-05 code".
- After the no-stop reruns (about 2.7 GPU h): fill the held rows
  ((32,64,64) both methods, n = 48, p = 4, 5, tol 1e-6) and the growth
  laws of 29.
- Only if the paper must run one smoothing form throughout: the sweeps,
  floor study, seeded arms and ladder are 60-100 GPU h to redo; the
  alternative is one sentence.
