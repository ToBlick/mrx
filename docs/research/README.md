# Research record

Campaign handoffs, plans, measurements and refuted approaches. Nothing here
describes what runs today; that is `docs/PRODUCTION.md`. Every file starts
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

Superseded chain, oldest first (each replaced by the next; kept for the reasoning):

- `preconditioner_plan.md` -> `mass_preconditioner_pivot.md` -> `mass_preconditioners.md` -> `tensor_preconditioners.md` -> `preconditioner_technical_note_source.md`.
- `natural_bc_coefficient_handoff.md` (day-by-day record, 2916 lines) -> distilled into the technical note.
- `laplacian_mg_k0_plan.md` -> `handoff_2026-08-13_gpu_cluster.md` -> superseded by the 2026-08-22 stack; multigrid is shelved.
- `hiptmair_xu_preconditioner.md` — HX/AMS at k>=1, shelved; verdicts in the lessons file.
- `production_simplification_plan.md`, `sweep_plan_2026-08-24.md`, `TODO_2026-08-24_precond_audit.md`, `status_2026-08-25_precond_prod.md` — executed; outcome in `audit_2026-08-25_production.md`.

## Relaxation and initial conditions

- `descent_method_2026-08-26.md` — CG vs L-BFGS m=1..10 on the Clebsch IC: one trajectory to within noise; CG stays the default.
- `relaxation_ic_2026-08-25.md` — canonical: ICs from logical profiles or GVEC scalars, closed-form helicity, the two silent traps.
- `handoff_2026-08-25_relaxation_prelim.md` — campaign narrative (2757 lines); section 34 is the sweep shelf.
- `relaxation_results_table.md` — generated on greville-prod (53a71ed) from the `out/relax_prelim` archive; static.

## GVEC and vacuum fields

- `handoff_2026-08-25_gvec_ic.md` — session record; its deliverable became `docs/gvec_mrx_interface.md`.
- `w7x_vacuum_bfield_handoff.md` — vacuum-field projection recipe and frame traps.
- `gvec_h5_vacuum_comparison.md` — MRX vacuum field versus GVEC h5; corrects the previous file on the simsopt exports.

## Poincare

- `handoff_2026-08-24_poincare.md` — tracer physics; stands unchanged.
- `handoff_2026-08-25_poincare_plotter.md` — plotter and relaxed-state tracer.

## Convergence and bugs

- `handoff_2026-08-25_poisson_convergence.md` — the eight Poisson cases; `nbc_k1` order ~3.2 is open.
- `poisson_convergence_submitit_bug.md` — superseded; its hypothesis was refuted by finding 8 of the previous file.
- `handoff_2026-08-24_harmonic_k1_free.md` — the k=1 free harmonic form and the saddle-outer diagnosis; fix unmerged (`OPEN.md` 2.1).
- `handoff_2026-08-25_histopolation.md` — resolved; leftovers in `OPEN.md` 3.5.
