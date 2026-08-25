# Production audit — open problems, 2026-08-25

Read-only audit of `mrx/` at `greville-prod` **14b3671**. Compiled by the production
cleanup agent, independently spot-verified by the coordinator. **Nothing here was
changed.** This is a worklist, not a record — delete this file once the items are
resolved or explicitly declined.

Every claim below was checked against the tree. Where the two of us disagreed, the
resolution is stated inline.

---

## A. Needs a decision

These are not typing. Each one is a judgement about what the project wants.

### A1. `mrx/relaxation_deprecated.py` — 832 lines, **zero code**, and two live drivers import from it

    $ grep -vE '^\s*#|^\s*$' mrx/relaxation_deprecated.py | wc -l
    0
    $ grep -cE '^(def |class )' mrx/relaxation_deprecated.py
    0

Importers, all top-level and unguarded:

    scripts/config_scripts/relax_from_nfs.py:31
    scripts/config_scripts/relax_stell.py:36
    test/deprecated/integration_tests/test_z_pinch.py:11    (deprecated, fine)

**Neither `config_scripts` driver can import.** That directory is not deprecated, and
`slurm/job_relax_from_nfs.sh` is tracked with documented multirun usage pointing at
one of them. `relax_from_nfs.py` is the driver behind `conf/config_relax_from_nfs.yaml`
— the p=4 config.

**Dated:** commit `9bf888b` "all of the refactors" (2026-06-06) emptied the module and
left both imports in the *same commit*. Broken ~2.5 months, unnoticed.

**Decision:** delete the module and both scripts, or rewrite them against
`mrx/relaxation.py`. The relaxation work has since moved to
`scripts/debug/relax_prelim.py`. Given "scripts are cheap, production stays clean",
deletion is the default — but these were real drivers once.

### A2. 356 lines of CP/ALS/NTF in `mrx/preconditioners.py`, reachable only from debug scripts

    _mode_unfold_3tensor                         475     2 lines
    _cp_als_3tensor                              501    68
    _greedy_cp_terms                             684    54
    _cp_ntf_3tensor                              740    72
    _ntf_terms                                   814    19
    _build_diagonal_tensor_block_factors         963    78
    _build_mass_referenced_tensor_block_factors 1413    63   (zero refs anywhere)

The chain's only entry point is `_build_diagonal_tensor_block_factors`, imported by
exactly two files — `scripts/debug/greville_bulk_precond_k0.py` and
`greville_bulk_speed.py`. **Nothing in `mrx/`, nothing in `test/`.** ~12% of the file.

Carries the `MRX_CP_GREEDY` env knob, whose stated purpose is restoring "the legacy
unconstrained greedy rank-1 ALS fit for A/B comparison" — structurally the same
one-meaningful-setting knob as `MRX_MASS_KIND`, retired 2026-08-25.

This is the CP/ALS stack superseded by metric lumping.

**Decision:** do those two greville-bulk scripts still matter?

### A3. `MRX_BJ_BC_SCALE` silently overrides an explicit argument

`mrx/metric_lumping_laplacian.py:364`:

    def _resolve_bc_scale(bc_scale=None):
        env = os.environ.get("MRX_BJ_BC_SCALE")
        if env is not None:
            return float(env)          # wins over the caller's argument
        return PRODUCTION_BC_SCALE if bc_scale is None else float(bc_scale)

A caller passing `bc_scale=2.0` is silently ignored when the variable is set. The
documented rationale is genuine — sweep harnesses always set it so recorded arms keep
their meaning — but **a leftover export in a shell silently changes production
numerics of the production Laplacian preconditioner**, and nothing reports it. This is
the hidden-factor class covered by the standing "metric factors must be explicit" rule.

Four debug scripts set it: `block_jacobi_spectrum`, `bench_real_solves`,
`bc_schur_effective`, `verify_block_jacobi`. Two pass no argument and would be
unaffected by a change.

**Decision:** flip the precedence so an explicit argument wins, or keep it and log
loudly when the env path is taken.

### A4. `assemble_mass_metric_lumping_preconditioner` — public API nothing calls

`mrx/operators.py:3031`. Public eager-assembly entry point; zero in-repo callers. A
user could reasonably call it. Deserves a judgement rather than the dead-code
treatment.

---

## B. Mechanical — no decision required

### B1. `mrx/solvers.py:597` — the return-convention trap, THIRD instance

    Returns:
        info: 0 if converged, >0 otherwise.

This is `solve_saddle_point_minres`, the **production k>=1 saddle solver**. It returns
`info` verbatim from `minres`, which uses the signed convention
(`jnp.where(converged_final, -k_final, k_final)`). Lines 217 and 402 of the same file
were corrected on 2026-08-24 and 08-25, each carrying a note that the stale version
"caused a converged solve to be read as a failure". Line 597 was missed both times.

**Latent, not live** — every consumer was checked (`operators.py:3782`, `:3919`, four
sites in `benchmark_graddiv`, `bench_real_solves`); none tests `info == 0`. The next
person to write that gets a silent wrong answer.

One docstring line. **Highest cost-of-leaving / cost-of-fixing on the list.**

### B2. `mrx/operators.py:363` — `K0TensorHodgePreconditionerFactors`, dead class

42 lines, zero references in `mrx/`, `test/` or `scripts/`. A survivor of the Stage B
tensor deletion; its own comments describe an apply that no longer exists.

### B3. Legacy alias `'hodge_laplacian'`

`mrx/operators.py:359` advertises it; the `match` at 326 and 347 dispatches
`case "hodge_laplacian" | "laplacian"`. **Exactly one user:** `mrx/assembly.py:618`.
Repoint it to `'laplacian'` and drop the alias — two lines, textbook no-zombie.

### B4. `mrx/relaxation.py:407` — a comment describing a defect that is fixed

Warns that `apply_incidence_matrix`'s docstring prefers the mass-projected form. That
docstring was corrected on 2026-08-25 (`derham_sequence.py` now reads "PREFER THIS
FORM" and explains the correction). The comment now sends readers chasing a
contradiction that no longer exists.

Its cited line number has also drifted: `apply_incidence_matrix` is at **2036**, not
2039. **Bare line-number citations in comments rot** — worth a convention of citing
symbols rather than lines.

### B5. Leftovers from the Stage C deletion, reported by the agent that made them

* `mrx/preconditioners.py:1550` `_metric_lumping_block_apply` — **zero references**.
  Its only caller, `apply_mass_raw_kron_preconditioner`, was deleted in Stage C.
  `ruff --select F` does not flag an unused module-level function.
* `mrx/preconditioners.py:1559` — `build_mass_metric_lumping_factors` still opens
  *"Build the raw_kron mass preconditioner factors for `M_k`"*. Function renamed, first
  line not.

### B6. ~21 zero-live-reference top-level symbols

    88  mrx/assembly.py:511          build_neighbors
    68  mrx/assembly.py:441          assemble_sparse
    43  mrx/solvers.py:55            newton_solver
    29  mrx/mappings.py:200          invert_map
    21  mrx/operators.py:3031        assemble_mass_metric_lumping_preconditioner  (see A4)
    16  mrx/operators.py:2949        _build_nested_iterative_preconditioner_apply
    14  mrx/mappings.py:184          approx_inverse_map
    14  mrx/io.py:83                 epoch_time
    12  mrx/operators.py:2934        _normalize_recursive_scalar_leaf_spec
     8  mrx/assembly.py:611          assemble_dense_hodge_laplacian
     8  mrx/preconditioners.py:301   set_mass_tensor
    + 6 smaller in preconditioners.py

**Verification note.** A coordinator spot-check found references to `build_neighbors`,
`assemble_sparse` and `newton_solver` that the sweep reported as zero. On resolution,
**all are in `test/deprecated/`**, which is gitignored and outside the live suite. The
zero-live-reference claim stands; deleting these breaks deprecated tests only.

---

## C. Negative results — recorded so nobody re-checks them

* **No permanently-skipped tests remain.** One `pytest.skip` in the live suite
  (`test_sequence.py:137`) and it is structural — "no harmonic forms for this
  (k, dirichlet)".
* **No consumer misreads the signed `info` convention** anywhere in `mrx/`, `test/`,
  or non-deprecated `scripts/`.
* **Accept-list / dispatch agreement is sound outside preconditioner kinds.** Checked:
  frames (`io.py:368`, `projectors.py:267,365`), `ktilde_mode`, `rescale`, operator
  kinds. All dispatch what they accept. The only alias is B3.

---

## D. Coverage — stated honestly

This is **a pattern-directed sweep of the whole tree plus a deep read of the
preconditioner/solver stack. It is not a line-by-line audit of `mrx/`.**

**Read properly:** `solvers.py`, the preconditioner regions of `operators.py` and
`preconditioners.py`, `metric_lumping_laplacian.py` around the BC scale,
`relaxation.py` around the incidence swap, `differential_forms.py:418-450`,
`relaxation_deprecated.py` (entirely — it is all comments).

**Grepped and spot-read only:** `derham_sequence.py`, `nullspace.py`, `projectors.py`,
`io.py`, `mappings.py`, `assembly.py`.

**Not opened at all:** `extraction_operators.py` (52 KB), `local_assembly.py` (38 KB),
`plotting.py`, `geometry.py`, `spline_bases.py`, `quadrature.py`, `circulation.py`,
`io_nfs_map.py`, `config.py`, `poincare.py`, `experimental/*`.

The eight defect classes below were searched everywhere. **Classes nobody has named
yet would only have been caught in the parts that were read.** The largest unexamined
surface is `extraction_operators.py`, which is also where the polar Gram correction
that makes `d.d` exact lives.

### Modules with no direct test import

Transitively exercised, so not "untested" — but nothing targets them:
`extraction_operators.py`, `local_assembly.py`, `assembly.py`, `poincare.py`,
`circulation.py`, `io_nfs_map.py`, `plotting.py`, `config.py`, `utils.py`,
`spline_geometry.py`.

---

## E. The defect classes searched for

Each was found by accident during 2026-08-25 and then searched for deliberately.

1. **Stale docstrings contradicting the code.** Three found that day, each cost real
   work: `minres` claiming "0 if converged"; two preconditioner resolvers justifying
   `raw_kron` with a reason `C1` had invalidated; `apply_incidence_matrix` preferring
   the mass-projected form after the Gram correction made the incidence form exact.
   **Highest-yield category.**
2. **Accept-lists not matching dispatch**, in either direction.
3. **Checks that cannot distinguish "condition met" from "check failed."**
4. **Defensive code over guaranteed-positive quantities.** Note the distinction:
   keying on `sy > 0` (does a usable pair exist) is structural and fine; keying on
   `yy > 1e-30` (is this positive thing positive enough) is defensive and is not.
5. **Zombie code and aliases** kept so old scripts keep working.
6. **Dead code** — zero callers, unreachable branches, unpassed parameters.
7. **Coverage in appearance only** — permanently-skipped tests, assertions that
   cannot fail; and the inverse, production invariants with no test.
8. **Error and log text naming things that no longer exist.**

## F. Recommendation

The **dead-code AST sweep was the highest-yield single tool** — 21 symbols, including
two created that same morning, which `ruff` does not flag. It is ~60 lines and could
run in CI. That is the one item here that would prevent recurrence rather than fix an
instance.
