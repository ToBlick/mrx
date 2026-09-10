# Simplify sweep, 2026-09-05

**Merged 2026-09-10, in part.** The branch was cut at ae7293b; by the time
it was merged into `static-dynamic-refactor` the solver core, the sequence,
the relaxation loop, the Poincaré tracer and plotter and their docs pages had
been rewritten (Newton, trace/plot split, PR #20). Tobias's rule for the
merge: take what still applies and is easy, ignore what this note says about
modules rewritten since. What went in: the deletions in `mrx/__init__`
(no star imports), `gvec.py` (no `sp` threaded through the readers, no
`sign` on `build_gvec_map`, `block_knots`, `radial_design`), `nullspace.py`,
`differential_forms.py` (`jacobian_determinant`; the `derivative_axes` method
is read by `preconditioners.py`), `quadrature.py` (`spectral_quad`,
`select_quadrature`), `precision.py` (`solve_tol` takes no factor),
`preconditioners.py`, `spline_bases.py`, `mappings.py`, `geometry.py`,
`vmec.py`, `initial_conditions.py`, `plotstyle.py`; the tests that went with
them; `docs/source/concepts.md`, `concepts/PRODUCTION.md`,
`slurm/README.md` removed, `concepts/manufactured_solutions.md` moved to `docs/research/`; the API
index trimmed to the caller-facing modules; the concept pages
`architecture`, `polar`, `preconditioning`, `testing_strategy`,
`gvec_mrx_interface`, `cluster`, `faq`, `getting_started` as rewritten
here. Kept at the current version, so the paragraphs below about them do
NOT describe the tree: `derham_sequence.py`, `mass.py`,
`metric_lumping_laplacian.py` (still owns its own `derivative_axes(k, c)`),
`operators.py` (`extraction`, `_require_bundle`, the assembler keywords),
`projectors.py`, `extraction_operators.py` (the boundary-DoF family stays),
`plotting.py`, `poincare.py`, `relaxation.py`, `solvers.py`,
`scripts/relax.py`, `scripts/plot_relaxation.py`, the tutorials,
`docs/source/relaxation.md`, `tutorials.md`, `concepts/mass.md`,
`concepts/precision.md`, `concepts/relaxation.md`. `scripts/poincare_relax.py`
is gone on both sides (replaced by `poincare_trace.py` + `poincare_plot.py`).

Six read-only passes over `origin/static-dynamic-refactor` at ae7293b
(solver core; sequence and spaces; geometry, readers and tests; relaxation,
driver and tutorials; Poincaré and plotting; the docs), each reading its
slice in full and grepping the rest for callers, then applied on branch
`worktree-simplify-sweep` in five commits (four for the code and the docs pages, one for `concepts/relaxation.md`). The rule: delete what has no
consumer anywhere -- library, tests, the shipped scripts (`relax.py`,
`plot_relaxation.py`, `poincare_relax.py`, the tutorials) AND the study
scripts -- and list the rest here for Tobias.

## Done

**Library (~900 lines).** The never-set knobs of the atoms (`bc_scale`,
`bc_entry`, `core_tol`; `PRODUCTION_BC_SCALE` is the one number),
`trace_components`, the `eps == 0` arms of the shifted solve, MINRES's
optional-argument defaults, the `'constant'` basis type nothing constructs,
the boundary-DoF family (`bc_lift`, `apply_bc_mass_correction` -- which
read an attribute that no longer existed -- `E_bc`, `n_bc`,
`bc_extraction_op`, `load(bc=)`), `apply_weak_div`,
`apply_mass_plus_eps_laplace_matrix`, `set_map_and_preconditioners`, the
`tol`/`maxiter` parameters of the sequence's inverse applies, `r_scale`,
`n_inner`, the `compute_nullspaces` method, the star imports of
`mrx/__init__` (`import mrx` no longer imports matplotlib's neighbours),
`sp` threaded through the GVEC readers (the parsed block carries its
knots), `R_fn`/`Z_fn`, `SplineMap`'s write-only fields, `State.B_nplus1`
(a per-step temporary that doubled every checkpoint's field),
`lbfgs_sy`, `picard_restarts`, `trace["gain"]`, `chunk_runner`'s unread
step index and `extra` hook, the unreachable scheme branch, the
physical-profile branch of `render_section` (`midplane_crossings`,
`surface_label`, `--profile-coord`), `plot_torus`'s multi-period
rendering, the `batch_size` thread through the tracer, `logical_field`'s
k=1 arm, the pinned parameters of `seed_from_axis`, `trace_and_classify`,
`plot_twin_axis`, `plot_crossections_separate`, `EVENT`, `DPI`.

**One owner of the derivative-axes rule:** `DifferentialForm.derivative_axes(c)`,
read by `mass.py`, the atoms, `_form_comp_info` and the histopolation
(four spellings before). The residual view builds its extraction through
the constructor's own `_build_extractions`. The output permutation of a
Laplacian atom is computed once, not twice.

**Two bugs.** `@house_style()` had landed on a rotation helper instead of
`plot_torus` (b8701a1), so the torus figures drew unstyled; the snapshots
movie pinned the profile panel's x-limits in metres on a logical-r axis.

**Copies replaced by calls.** `poincare.trace_sections` +
`plotting.section_figure` for the Poincaré loop the tutorials wrote three
times (and the dead `poincare.section_figure`); `plotting.torus_grids` for
the cut/surface grids written five times; `poincare_relax.py` archives
`sections.npz` and renders from it on one path (a fresh trace and
`--from-npz` alike), saving through `plotting.save_figure` (the same
PGF writer was in the script too). The tutorials: tutorial 2 takes the
current from `apply_weak_curl` instead of a force evaluation, 3 and 5 no
longer compute a weak pressure nothing read, 3 evaluates `initial_state`
once, 4 and 5 share nothing by copy any more.

**Driver.** The checks the library makes anyway, `params["start_step"]`
and the second parse of the geometry file for its kind are gone.

**Docs (2,660 -> ~1,600 lines).** `concepts/PRODUCTION.md` (a third copy of
five pages, wrong in nine places), `concepts.md` (a hand-written sidebar)
and `slurm/README.md` (the twin of `cluster.md`) deleted;
`manufactured_solutions.md` to `docs/research`; eight internal modules out
of the API tree; every page without its dated measurements and its copies
of other pages. Wrong identifiers fixed: the k=1,2 Laplacian solver (the
Hodge split, not MINRES) on four pages, the float32 default on
`getting_started`, `SOLVE_TOL`, `_sumfact_kernel`,
`operators.laplacian_lumping`, `inner_tol`, `--eps`, the five standing
planes, `force_floor_reached`, `test_weak_pressure.py`, `<out>/reconnect/<k>/`.

**History notes** ("until 2026-..", "it used to", measured tables in
docstrings) reduced to their load-bearing sentence throughout; the
measurements stay in `docs/research`.

## For Tobias to decide (kept, used only by study scripts)

| symbol | lines | used by |
|---|---|---|
| `nullspace.compute_nullspaces_iterative`, `find_nullspace_vectors`, `estimate_spectral_gap`, `_initial_guesses`, `_logical_constant_seed` | ~340 | `poisson_study.py`, `vacuum_convergence.py`, `analytic_vacuum.py`, `quad_order_equivalence.py`; the `b2 > 0` route |
| `apply_laplacian_preconditioner` (+ the sequence method) | 10 | `quad_order_equivalence.py` |
| `DeRhamSequence.get_operators`, `set_operators`, `nullspace`, `apply_strong_div` | 12 | `poisson_study.py`, `vacuum_convergence.py`, `analytic_vacuum.py`, `map_projection_study.py` (and tutorial 2 for `get_operators`) |
| `plotstyle.figsize`, `COLUMN_WIDTH`, `TEXT_WIDTH`, `PANEL_ASPECT`, `arm_style`, `CYCLE`, `DASHES` | 30 | `midpoint_figures.py` (the paper style) |
| `gvec.evaluate` | 8 | `test_readers.py` only (the oracle of the synthetic state) |
| `metric_lumping_laplacian._fd_stiffness_degree0` | 75 | the p=1 stand-in; every mesh in the repo has p >= 2 |
| `test/manufactured.py`'s `frame='phys'` arm | 20 | `poisson_study.py` |

Study scripts this sweep broke (they call removed signatures):
`vacuum_convergence.py` (`section_figure`, `logical_field(..., 2, True)`,
`nullspace`), `map_projection_study.py` (`StateField(vector=)`, `sp`
arguments), `poisson_study.py` (`set_operators` still exists;
`compute_nullspaces_iterative` still exists), `li383_pub_figures.py`
(`render_section` keyword changes, `P_COLOR` from `mrx.plotting`),
`plot_mesh.py` (`save_figure` fine; `build_gvec_map` info keys). They are
paper tooling and were not touched.

**Two plotters.** `plot_relaxation.py` (the torus pressure figures and the
trace) and `poincare_relax.py` (the sections) both read a run directory;
the shipped set says ONE plotting driver. They share the run rebuild
(~12 lines each) and nothing else; merging them is a CLI decision, not a
cleanup.

## Verification

`pytest test` (refined float32, the production configuration) as a GPU
job from the worktree after the library commit and at the head: 49 passed
both times (job 17940368, 9:05). An end-to-end smoke of every shipped
script at small settings (job 17940369, 32 min): the five tutorials
(`(8,16,8)` / `(8,12,12)` p=2, two outer x ten inner steps), `relax.py`
on the analytic torus with `--reconnect-every 10` (two checkpoints, one
resistive step, a restart), `plot_relaxation.py` and `poincare_relax.py`
on the tutorial-3 run including `--from-npz`, `--pressure strong` and
`--fields snapshots` on the torus run. All eleven steps passed.
