# Release review: self-sweep findings (2026-08-27)

The file-by-file sweep for the three things the review targets — legacy code
we no longer use, legacy compatibility we can drop, simplifications — run
after the static/dynamic refactor on branch `static-dynamic-refactor`
(`2be817c`). Nothing here has been changed; every item is a finding awaiting
a decision. Line numbers are as of `2be817c`.

## Dead code (zero references from any live code, script or test)

1. **`mrx/projectors.py:588` `BoundaryProjector`** — ~260 lines, headed by two
   `# TODO: requires testing still`; the surface-integral boundary load.
   Nothing constructs it. (Related but *live*: `load(bc=True)`/`E_bc` — the
   non-homogeneous-BC facility we decided to keep; only `BoundaryProjector`
   itself is dead.)
2. **`mrx/differential_forms.py` `Pullback`** — no caller anywhere (its
   sibling `Pushforward` is used by poincare/plotting/tutorials); also
   **`jacobian_determinant`** (one-liner, no caller).
3. **`mrx/operators.py:193` `_assemble_k0_greville_bulk_factors`** — only
   reference is a docstring mention in `metric_lumping_laplacian.py:543`; the
   lumping atoms build their factors via `component_factors`.
4. **`mrx/nullspace.py`**: `estimate_spectral_gap` (722), `generic_rayleigh`
   (266), `exact_derivative_residual` (274) — research diagnostics with no
   caller (`harmonic_rayleigh` *is* used, by the vacuum-field tutorial).
5. **`mrx/poincare.py:840` `seed_line`** — no caller; production seeding is
   `seed_from_axis`. One stale mention in `docs/source/relaxation.md:207`.
6. **`test/random_fields.py`** — the whole module (Besov random-field
   generators): no test imports it.
7. The previously tagged ORPHANs still standing:
   `mappings.one_size_fits_all_map`, `mappings.extend_map_nfp`,
   `geometry.greville_interpolate_stellarator_map` (one stale doc mention),
   `initial_conditions.metric_coefficients` ("currently not pursued" per its
   own comment), `solvers.backtracking_line_search` (nothing calls it — the
   relaxation has its own analytic line search; the module docstring still
   advertises it in its first line).

## Options only ever used at one value (candidates to hard-wire)

8. **`relaxation.TimeStepper.stochastic`** + `apply_noise` +
   `State.noise_level`/`key` — `stochastic=True` is set nowhere (not even
   `relax.py`); the PRNG key threading through `State`/`relaxation_step`/
   `relaxation_loop` exists only for this.
9. **`relaxation` L-BFGS**: `DescentMethod.LBFGS`, `_lbfgs_direction`, and
   the `State` fields `s_history, y_history, Ms_history, My_history,
   lbfgs_sy` are reachable via `relax.py --method lbfgs`, but the 2026-08-26
   descent study concluded "CG default; L-BFGS brings nothing on this
   problem". If that stands, the whole L-BFGS arm plus five `State` fields
   can go.
10. **`mappings.stellarator_map(flip_zeta=...)`** — threaded through two
    signatures, never passed `True` by any caller.
11. **`metric_lumping_laplacian` research knobs**: `ktilde_mode="roundtrip"`,
    the `lumped` scalar/off variants, `extra_rings`, `outer_rings` — no
    non-default use outside one test docstring; each is a
    measured-and-lost alternative per its own comments. Same file:
    **`MRX_BJ_BC_SCALE` env var** (`_resolve_bc_scale`) — the last env knob
    in the package.
12. **`operators._coerce_scalar_hodge_preconditioner`** — near-duplicate of
    `_coerce_mass_preconditioner_spec` with a materialize branch; it also
    returns bare strings that the builder re-coerces. Foldable into one
    coercer.

## Simplifications / warts

13. **`solvers.solve_singular_cg(..., vs=[])`** — mutable default argument;
    harmless today (never appended) but a classic trap.
14. **`solvers.py` module docstring** still opens with "and a backtracking
    line search" (the orphan); should name the three entry points that
    matter (`solve_singular_cg`, `solve_saddle_point_minres`,
    `preconditioned_cg`).
15. **`quadrature.QuadratureRule` `indexing='xy'` axis-swap TODO** (line 52):
    the flat quad ordering is (θ, r, ζ) and every consumer carries a
    reshape+transpose to compensate (`_reshape_quadrature_*`, the element
    layout, the CP factor order). The TODO itself says it is a coordinated
    migration; it is *the* remaining structural wart of the package. Also the
    class docstring is the old auto-generated style.
16. **`relaxation.State`** — 25 fields incl. `p`, `p_v`, `A`, and the giant
    `Literal[...]` in `update_field`; with L-BFGS and noise gone it shrinks
    by ~8 fields, and `update_field`'s Literal could be dropped for a plain
    `eqx.tree_at`.
17. **`poincare.py`** — 991 lines mixing the tracer (trace/rhs/classify),
    the figure code (`render_section` is ~260 lines), and seeding; moving
    `render_section` to `plotting.py` (or a `poincare/` split) would match
    the one-module-one-job pattern. `step_convergence`/`iota_convergence`/
    `escaped_mask` are internal-only — could be underscored.
18. **`spline_bases.py` per-basis-function API** (`SplineBasis.__getitem__`,
    `TensorBasis.evaluate/__call__/__getitem__`,
    `DerivativeSpline.__getitem__`) has no grep-visible `basis[i]`-style
    callers; the live paths are `evaluate_local`/`contract` and the jitted
    tables. Same story as `DifferentialForm.evaluate`. Also
    `TensorBasis`/`QuadratureRule`/`DifferentialForm` still carry the old
    boilerplate docstrings ("A class for handling…").
19. **`gvec.py` §3 `stride=`** on `build_gvec_map`/`load_gvec_grids`
    (grid-subsampling knob) — no script passes it since the closed-form
    route; candidate to drop with the next h5-route slim-down.
20. **`nullspace.py` `use_coarse`** arm of `find_nullspace_vectors` defaults
    False and only the iterative route flips it internally; the whole
    `use_harmonic_coarse` machinery in `operators.py`
    (`_shifted_harmonic_coarse_vector/ready/wrap`, ~80 lines) is exercised
    only via that path plus the eps>0 free-BC default — worth one deliberate
    decision: either it is load-bearing for `compute_nullspaces_iterative`
    (keep, document) or it can go with a measured re-run of the nullspace
    gate.
21. **`scripts/trim_gvec_export.py`** — with the closed-form route it only
    serves h5-only sources; its docstring still describes the
    pre-closed-form world.
22. **Doc leftovers**: `docs/source/relaxation.md:207` (`seed_line`),
    `docs/source/concepts/architecture.md` mention of
    `greville_interpolate_stellarator_map`.

## Suggested ranking

1–7 (pure deletions, ~700 lines), 8–9 (relaxation slimming — decision on
L-BFGS), 11 (lumping knobs + last env var), 13, 15 (the big one, as its own
project), 16–18.

## Decisions and status (2026-08-28)

Decided by Tobias, executed on `static-dynamic-refactor` (commits `5e3b3ee`
items 1-5, `c25657c` items 7-12):

| # | decision | done |
|---|---|---|
| 1 | remove `BoundaryProjector`; `load(bc=True)`/`E_bc` (lifting) stays the non-homogeneous-BC route | yes |
| 2 | `Pullback` (and `jacobian_determinant`) stay | -- |
| 3 | remove `_assemble_k0_greville_bulk_factors` (+ the five helpers only it used) | yes |
| 4 | `harmonic_rayleigh` and `estimate_spectral_gap` stay; `generic_rayleigh` and `exact_derivative_residual` removed; `compute_nullspaces` now prints, per form, the Rayleigh quotient and (k=1 free, k=2 dbc) `lambda_1` from 5 sweeps of inverse iteration deflated against the stored forms | yes |
| 5 | `seed_line` removed; the logical chart is a fourth panel of the relaxation figure (subagent on branch `poincare-panels` polishing it) | yes |
| 6 | `test/random_fields.py` stays (preconditioner benchmarks) | -- |
| 7 | `one_size_fits_all_map`, `metric_coefficients`, `backtracking_line_search` removed; `extend_map_nfp` stays, REWRITTEN (it recomputed the toroidal angle from zeta -- wrong for any map whose angle is not exactly linear in zeta -- now rotates the period map's own output; seam test added); `greville_interpolate_stellarator_map` pending Tobias's call | yes |
| 8 | noise arm removed (`stochastic`, `apply_noise`, `State.noise_level/key`, `noise_schedule`, the PRNG threading) | yes |
| 9 | L-BFGS stays; the four secant arrays are now `(0, n)` and untouched under cg/gradient (they were `(m, n)` and rolled every step whatever the method) | yes |
| 10 | `flip_zeta` stays | -- |
| 11 | `ktilde_mode`, `lumped`, `extra_rings`, `outer_rings`, `MRX_BJ_BC_SCALE` removed; `bc_scale` stays explicit, default `PRODUCTION_BC_SCALE`, exposed as `build_preconditioners(bc_scale=...)` | yes |
| 12 | `_coerce_scalar_hodge_preconditioner` now delegates to `_coerce_mass_preconditioner_spec` | yes |
| 13-22 | not yet decided | -- |
