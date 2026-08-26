# Open issues — index, as of 2026-08-26

One entry point for everything outstanding. Each item is one line plus a
pointer; the detail stays where it was written. If you fix something, delete
its line here and update its source.

Ordered by what bites first.

## 1. Live on the production line

**1.1 Run something real after every merge.** A renamed function whose caller
lives on the other branch produces no textual conflict, merges green, and dies
at setup. Git's conflict detection is textual; this defect is semantic. It was
found by a validation run dying, not by knowing to check. *Rule:* one real run
after any merge; grepping the call sites is the response once you know.

**1.2 The k=3 Laplacian apply is slower after the one-write apply.** At
`(16,32,16)` the k=3 `MetricLumpingLaplacian` apply went 47 -> 60 us (+21..28%)
while k=1,2 went -15%; the k=0..2 sizes and all masses are flat or faster. Fewer
ops, so an XLA fusion or dispatch artefact; unexplained. *Detail:*
`scripts/benchmark/precond_build_apply.py --hlo`; phase-1 preconditioner report.

**1.3 The k=1 mass apply is +6..12% after the static selector fast path**
(`(8,16,8)` 61 -> 68 us, `(16,32,16)` 60 -> 64 us; one dispatch is ~7 us), where
the first A/B showed 0. Same script.

**1.4 `float32` identity tests.** Tests that check identities at `1e-12`
(quadrature exactness, partition of unity, `d ∘ d`, autodiff vs
`DerivativeSpline`) fail in float32 by `1e-7..1e-6` with nothing diverged. Their
tolerances are hard-coded float64 constants and must be expressed through
`mrx.eps`. The suite is float64 until then. *Detail:* phase-1 misc report §E.

**1.5 `n_r >= p + 2` is not a named error.** Below it `component_factors`'s
eigenvalue scale goes non-finite and numpy raises `LinAlgError` from inside
`eigvals`. *Detail:* `assemble_metric_lumping_laplacian_preconditioner`
docstring; `handoff_2026-08-25_poisson_convergence.md` Stage D.

## 2. Committed but not merged

**2.1 S1, the harmonic-form preconditioner switch** (`precond-api`, commit
`ec0c4f7`). Numerically correct and verified live (W7-X p=5 k=1 free
6.6e-09 -> 4.7e-24 through the production path). It was blocked on the lazy
payload memo, which is fixed (eager payload since 2026-08-26), so it is
unblocked and unmerged. *Detail:* `handoff_2026-08-24_harmonic_k1_free.md`.

**2.2 `worktree-poisson-k1`, 5 commits** — rebased duplicates of merged work.
Deletable.

## 3. Open bugs and unfinished measurements

**3.1 `nbc_k1` converges at ~3.2, not 4.** Under-integration eliminated
(`dbc_k2` reaches ~4.4 at the same quadrature). Cheapest next step needs no
solve: the projection test on `omega_1`, which is what settled k=2. *Detail:*
`handoff_2026-08-25_poisson_convergence.md` §5.

**3.2 The `k2_dbc` harmonic polish stops at 0 iterations with residual
3.5e-4 at n=16** in the preconditioned Poisson baseline. The inverse-iteration
polish is gated on `relL2_direct`; no run has isolated polish from the
preconditioner at k=1,2. *Detail:* `handoff_2026-08-25_relaxation_prelim.md`
§34 A3; `handoff_2026-08-24_harmonic_k1_free.md`.

**3.3 No relaxation arm has been run to a floor.** The criterion exists
(`scripts/relax.py --floor-tol`, replayed in `test/test_relax_floor.py`) and no
arm has been re-run with it, so every h- and p-refinement claim is still open.
*Detail:* `handoff_2026-08-25_relaxation_prelim.md` §34 B1.

**3.4 The p-refinement result is confounded and withdrawn.** Its p=2 and p=4
arms predate the even-p quadrature fix. ~1.6 GPU-h re-measures both. *Detail:*
§33.2 and §34 A1 of the same file.

**3.5 Histopolation leftovers.** `test/test_projectors.py` accuracy tolerances
are still the `< 1.0` placeholders; `frame='phys'` at k=3 in `interpolate` is
rejected rather than defined; `mrx/metric_lumping_laplacian.py` still
attributes the 4-6x free-vs-dbc iteration lag to `det(DF) = 0` at the last
knot, which the 2026-08-25 measurement refuted (the zero was an autodiff
artefact, removed by the local-support evaluator). Treat the lag as
unexplained. *Detail:* `handoff_2026-08-25_histopolation.md` §9.

**3.6 Real pressure has never been exercised in the Poincaré tracer.** The
p panel was proven with a synthetic p. *Detail:*
`handoff_2026-08-25_poincare_plotter.md` §3.2.

**3.7 `poincare_vacuum.py` unverified since its refactor**, and neither
Poincaré verification is reproducible from the repo. Re-run it through
`slurm/run.sh`.

**3.8 Hydra multirun not exercised** after the `MRX_DTYPE` / `solver_tol`
plumbing; single runs verified. `MRX_ROOT` must be set for a multirun.
*Detail:* `slurm/README.md`.

## 4. Where the folding time goes

Production logs show XLA constant-folding alarms individually exceeding 2 s
at res16 in a real relaxation run; the microbenchmark at that resolution
measured 533 ms total. The warning column is currently unmeasurable.
*Detail:* `audit_2026-08-25_production.md` items 1 and 2.

## 5. The two shelves

Runnable experiments with question, decision, cost and null-result meaning:

- Preconditioners / infrastructure: `audit_2026-08-25_production.md`, 9
  items; item 7 first (what fraction of a k>=1 iteration is the
  preconditioner apply — never measured, the denominator of every comparison).
- Relaxation: `handoff_2026-08-25_relaxation_prelim.md` §34, 11 items, B1
  first (3.3 above).

## 6. Left undone, not questions

- `except Exception: pass` in `warm_mass_preconditioner_cache`
  (`mrx/operators.py`): with one mass kind left it hides genuine build
  failures.
- The `outer='none'` branch of `apply_inverse_shifted_hodge_laplacian` still
  builds a Schur apply it discards (`_build_schur_apply_from_saddle_preconditioner`
  in the `else` branch); ~0.2 s per run, a clarity argument.
- `mrx/geometries.py` `build_sequence` and `scripts/relax.py` docstrings say
  "quadrature `2p`"; the code passes `p + 1`.
- `mrx/metric_lumping_laplacian.py` module docstring says `bc_entry="exact"`
  is the default; every signature defaults to `"ibpd"`.
- `polar_order = 0` works but the `DeRhamSequence` docstring and error message
  say `1 or 2`.
- `verify_block_jacobi.py` keeps its name and re-exports `build_sequence`
  from `mrx.geometries` because 22 scripts import it there.
- `get_smallest_ev_pair` in `mrx/solvers.py` has no production caller
  (re-exported by `mrx/utils.py`, used by `test/deprecated`).

## 7. Candidates for deletion (decision pending)

From the phase-1 misc report §F; nothing deleted.

- `mrx/circulation.py` (SIMSOPT normalisation; callers
  `scripts/harmonic/eval_boundary_circulation.py`,
  `scripts/plotting/harmonic_nullspace_geometry.py`): move to `scripts/harmonic`.
- `conf/config_relax_from_nfs.yaml`, `RelaxFromNFSConfig`, `RelaxStellConfig`,
  `conf/experiment/{test,video,eta_sweep,convergence}.yaml`: for nonexistent
  `relax_from_nfs.py` / `relax_stell.py`; delete.
- `conf/config_mc_poisson.yaml`, `MCPoissonConfig`: for nonexistent
  `scripts/dice/mc_poisson.py`; delete.
- `scripts/config_scripts/mass_preconditioner_submit.py`,
  `conf/config_mass_preconditioner.yaml` (tensor era); delete.
- `scripts/harmonic/run_harmonic_nullspace.slurm` (hard-coded account);
  delete or rewrite.
- `scripts/deprecated/*` (4 tensor/Chebyshev demos); delete.
- `scripts/interactive/*` (11 cell scripts): delete. `projection_test.py` is
  dead against the current API. (`debug_poisson_convergence.py` was deleted
  2026-08-26: it imported the removed tensor mass preconditioner and called
  the removed `assemble_hodge_laplacian_tp`, so it could not run.)
- `scripts/plotting`: `force_multiplot`, `helix_plots`, `iter_plots`,
  `solovev_*`, `stell_plots`, `plot_mass_sweep_results`, `w7x_matrix_fill`,
  `w7x_saddle_fill`, `plot_poisson_{results,sweep}` unreferenced; delete.
  Keep `harmonic_nullspace_{geometry,plots}`, `plot_harmonic_nullspace_saved`,
  `load_results`, `plot_poisson_convergence`, `poincare_plots`.
- Untracked slurm wraps calling missing scripts (26 of 57); delete.
- `scripts/debug` (85 files): finished-campaign diagnostics (greville mass/CP
  era, k0 MG and atom campaign, BC-alpha campaign, payload probes,
  harmonic-form campaign except `nullspace_jacobi_ab` until 2.1 lands, C2
  polar, Poisson frame probes, W7-X data verification) to archive or delete.
  Still useful: `verify_block_jacobi`, `verify_default_preconditioners`,
  `w7x_geometry`, `poincare_vacuum/relaxed/replot/render_check/converge/pullback_check`,
  `relax_results_table`, `relax_plot_traces`, `analytic_ic_verify`,
  `lambda_warmstart`, `greville_mass_coupling_ceiling`, `torus_map_plot_check`.
- Scripts referencing deleted surgery functions:
  `scripts/interactive/inspect_tensor_mass_defaults.py`,
  `scripts/interactive/screen_vlp_neumann_eta.py`,
  `scripts/benchmark/benchmark_graddiv_k1_preconditioner.py`,
  `scripts/debug/greville_bulk_precond.py`.
- `mrx/spline_geometry.py`: a 14-line re-export shim of `mrx/geometry.py`.
