# Research record

Campaign handoffs, plans, measurements and refuted approaches. Nothing here
describes what runs today; that is `docs/source/concepts/relaxation.md` and the rest of `docs/source/`. Every file starts
with a three-line block: its status, what to read it for, and what not to
read it for. Read the block before the file.

Open items across all campaigns are indexed once, in `OPEN.md`.

Every `scripts/debug/...`, `scripts/benchmark/...`, `slurm/job_...` and `conf/...` path in
these notes refers to branch `greville-prod`, commit 53a71ed; those files are not on the
clean branch.

## Preconditioner campaign

Canonical:

- `preconditioner_technical_note_source.md` — the production preconditioner: construction, derivation of the natural-BC coefficient, refutations, all measurements (its 0.10 scale is superseded by `s_scale_2026-08-25.md`).
- `preconditioner_lessons.md` — settled findings, dead ends and traps, in priority order.
- `s_scale_2026-08-25.md` — why `PRODUCTION_BC_SCALE = 3.0`.
- `result_2026-08-25_schur_probe_ab.md` — the irreproducible probe measurement cited from `mrx/preconditioners.py`.
- `audit_2026-08-25_production.md` — the open shelf of preconditioner and infrastructure experiments.
- `precond_h_scaling_2026-09-02.md` — the atom's iteration count vs resolution, all `(k, BC)`, toroid vs QA: not h-independent anywhere, k>=1 on QA loses equivalence with n; `bc_scale` n-dependence and Lanczos localisation (open). Also the k=1 free harmonic-form fix (0c3aa4d) that exposed it.
- `li383_sweep_results_2026-09-02.md` — every li383 relaxation number (2026-08 sweep, current-reader reruns, the seeded (6,1)/(5,1) island arms), which arm backs which paper figure; the reference sets the residual floor.

Superseded chain, oldest first (each replaced by the next; kept for the reasoning):

- `preconditioner_plan.md` -> `mass_preconditioner_pivot.md` -> `mass_preconditioners.md` -> `tensor_preconditioners.md` -> `preconditioner_technical_note_source.md`.
- `natural_bc_coefficient_handoff.md` (day-by-day record, 2916 lines) -> distilled into the technical note.
- `laplacian_mg_k0_plan.md` -> `handoff_2026-08-13_gpu_cluster.md` -> superseded by the 2026-08-22 stack; multigrid is shelved.
- `hiptmair_xu_preconditioner.md` — HX/AMS at k>=1, shelved; verdicts in the lessons file.
- `production_simplification_plan.md`, `sweep_plan_2026-08-24.md`, `TODO_2026-08-24_precond_audit.md`, `status_2026-08-25_precond_prod.md` — executed; outcome in `audit_2026-08-25_production.md`.

## Relaxation and initial conditions

- `descent_method_2026-08-26.md` — CG vs L-BFGS m=1..10 on the Clebsch IC: one trajectory to within noise (historical: the CG arm was deleted 2026-08-28, the L-BFGS memory 2026-09-17; the descent is gradient descent on the smoothed force, Newton the method).
- `relaxation_ic_2026-08-25.md` — canonical: ICs from logical profiles or GVEC scalars, closed-form helicity, the two silent traps.
- `handoff_2026-08-25_relaxation_prelim.md` — campaign narrative (2757 lines); section 34 is the sweep shelf.
- `relaxation_results_table.md` — generated on greville-prod (53a71ed) from the `out/relax_prelim` archive; static.

## GVEC and vacuum fields

- `handoff_2026-08-25_gvec_ic.md` — session record; its deliverable became `docs/source/concepts/gvec_mrx_interface.md`.
- `w7x_vacuum_bfield_handoff.md` — vacuum-field projection recipe and frame traps.
- `gvec_h5_vacuum_comparison.md` — MRX vacuum field versus GVEC h5; corrects the previous file on the simsopt exports.
- `qa_vacuum_convergence_2026-08-28.md` — VMEC QA wout vs the discrete harmonic form by resolution: O(h^p) in the bulk with no floor to 5e-5, the residual is the axis; scale fit = flux match (theorem).
- `analytic_map_2026-08-28.md` — the polar spline map of a GVEC state / VMEC wout built from the series coefficients (L2 projection, closed-form angular symbols), no evaluation grid; interpolant measured and dropped; axis and wall analysis.

## Poincare

- `handoff_2026-08-24_poincare.md` — tracer physics; stands unchanged.
- `handoff_2026-08-25_poincare_plotter.md` — plotter and relaxed-state tracer.

## Convergence and bugs

- `handoff_2026-08-25_poisson_convergence.md` — the eight Poisson cases; `nbc_k1` order ~3.2 is open.
- `poisson_convergence_submitit_bug.md` — superseded; its hypothesis was refuted by finding 8 of the previous file.
- `handoff_2026-08-24_harmonic_k1_free.md` — the k=1 free harmonic form and the saddle-outer diagnosis; fix unmerged (`OPEN.md` 2.1).
- `handoff_2026-08-25_histopolation.md` — resolved; leftovers in `OPEN.md` 3.5.

## September 2026 (Newton, the paper's reruns)

- `newton_second_variation_2026-09-06.md` — Newton on the second variation: the operator, the potential form, the harmonic atom.
- `harmonic_atom_and_helicity_correction_2026-09-11.md` — the harmonic atom from the VMEC field, kappa/budget sweeps, the regularised line search, Powell's restart, the helicity correction.
- `hessian_spectrum_2026-09-17.md` — the Hessian's soft end is a field-aligned near-null continuum (u ~ fB), not resonant outliers; the parallel-flow penalty H + alpha M_par (`--newton-parallel-penalty`).
- `harmonic_atom_floor_and_smoothing_2026-09-17.md` — kappa = 3 was LM damping of that continuum; the computed strain floor + alpha reaches 7-20x below the September floor with intact surfaces; the regularised search, dt floor, step cap > 1 and Newton-direction smoothing are inert or negative; candidate default pending Tobias.
- `descent_block_m0_2026-09-17.md` — the paper's descent block rerun as gradient descent (m = 0); the L-BFGS memory removed from the code and the paper; what changed in the tables and figures.
- `paper_experiments_audit_2026-09-11.md`, `paper_rerun_plan_2026-09-11.md`, `paper_rerun_results_2026-09-11.md` — the paper's runs audited, replanned and rerun on the released code.
- `handoff_2026-09-03_chunked_relaxation_loop.md`, `handoff_2026-09-04_relax_cli_prune.md`, `handoff_2026-09-09.md` — the September handoffs (the last one carries the running status log).
