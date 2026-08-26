# Open issues — index, as of 2026-08-26 (misc cleanup applied)

**One entry point for everything outstanding.** Each item is one line plus a pointer;
the detail stays where it was written. If you fix something, delete its line here as
well as updating its source.

Ordered by what bites first, not by subsystem.

---

## 1. Live on the production line

**1.1 The `_flat` memo can capture tracers.** `metric_lumping_laplacian.py:1016`
memoises the flattened payload onto an atom instance on FIRST APPLY. Instances live
in a dict on `seq`, and `torus_seq` is `scope="session"` — one object for the whole
suite. Any code path whose first apply happens inside a `lax` body stashes tracers on
a long-lived object; the failure then surfaces in an unrelated later test as
`UnexpectedTracerError`.
LATENT today — no current path does it. **S1 creates one, which is how it was found.**
Fix (written up, untried): build the payload eagerly at construction. Construction is
already hoisted above the trace and the build is host-side numpy, so this deletes the
lazy path rather than guarding it. *Detail:* `audit_2026-08-25_production.md` item 9.
**Fixing this makes S1 mergeable as a side effect. It is not an S1 problem.**

**1.2 Worktree jobs and `PYTHONPATH`.** `mrx` is an editable install pinned to the main
checkout, so a job without `PYTHONPATH` silently tests the wrong library; `python -m`
and `python script.py` resolve differently, so print `mrx.__file__`. `slurm/run.sh`
exports it and prints `mrx from:` first (see `slurm/README.md`); the five tracked wraps
are untracked since 2026-08-26 and their invocations live in the scripts' docstrings.
*Detail:* memory `worktree-jobs-need-pythonpath`.
**Hydra:** the submitit launcher does not export PYTHONPATH on its own; the configs
under `conf/` now add `export PYTHONPATH=${oc.env:MRX_ROOT}` to the launcher setup, so
`MRX_ROOT` must be set for a multirun. Do NOT force `hydra/launcher=basic` — hydra then
rejects the submitit keys while composing and every study dies in ~1 s.

**1.4 A green merge tells you nothing; run something real afterwards.**
greville-prod renamed `assemble_block_jacobi_laplacian_preconditioner`; a caller on
another branch used the old name. Caller and definition lived in files only one side
touched, so NO CONFLICT SURFACED, the merge reported success, the conflict resolution
was also green, and every run would have died at setup. Git's conflict detection is
textual; this defect is semantic.
It was found by a validation run launched for an unrelated reason dying at setup —
**not** by knowing to check. So the rule is not "grep the callers after a rename",
which is the after-the-fact tidy version: it is **run something real after any merge,
because both green signals are silent on this class.** Grepping the call sites is the
right RESPONSE once you know; it is not what finds it.

**1.5 `test_projectors` is ~76% of pytest wall time**, one parametrisation 9.7 min
alone. A 1 h walltime gets cancelled at 61% and looks like a failure when nothing
failed. Budget 2 h.

## 2. Committed but not merged

**2.1 S1 — the harmonic-form preconditioner switch.** `precond-api`, commit `ec0c4f7`,
marked DO NOT MERGE. Numerically correct and verified live (W7-X p=5 k=1 free goes
6.632e-09 -> 4.674e-24 through the production path), but its gate is RED: it triggers
1.1 above. Blocked on 1.1, not on itself.

**2.2 ~~`relaxation-prelim`~~ — MERGED 2026-08-26.** Validated post-merge: full suite
235 passed / 1 skipped / 0 failed, Poisson reproduces to recorded precision.

**2.3 `worktree-poisson-k1`, 5 commits** — rebased duplicates of already-merged work.
Deletable.

## 3. Open bugs

**3.1 `nbc_k1` converges at ~3.3, not 4.** Under-integration ELIMINATED (`dbc_k2`
reaches ~4.4 at the same `quad_order_offset: 0`). Cheapest next step needs no solve:
run the projection test on `omega_1`, which is exactly what settled k=2.
*Detail:* `handoff_2026-08-25_poisson_convergence.md` §5.

**3.2 ~~Nothing in the relaxation campaign ever floored~~ — CRITERION LANDED 2026-08-26.**
`scripts/relax.py` (which replaces `relax_prelim.py`) stops when the windowed relative
energy decrease per step falls below `--floor-tol` (default `1e3*eps`); replayed on the
archived traces in `test/test_relax_floor.py` (S10 fires at ~1660, S07/C1 never). No
arm has been RE-RUN to a floor yet, so the h-refinement claims are still open.
*Detail:* relaxation handoff §34, item **P0**.

**3.3 The p-refinement result is confounded and withdrawn.** Its p=2 and p=4 arms
predate the even-p quadrature fix, and they are the two best-looking points in the
sweep. ~1.6 GPU-h re-measures both and restores the p axis. *Detail:* §33.2.

**3.4 Real pressure has never been exercised in the Poincaré tracer.** The p panel was
proven with a synthetic p, labelled as such. A real flux-function p collapses each
stripe to a point, which makes stripe width a free diagnostic.
*Detail:* `handoff_2026-08-25_poincare_plotter.md` §3.2.

**3.5 Histopolation leftovers.** Accuracy tolerances are still vacuous `< 1.0`
placeholders and should be set from measured values now that the operators are exact
at both parities; `frame='phys'` at k=3 is unresolved rather than defined; the
closed-last-piece convention is offered and not applied because it touches every
assembly path. *Detail:* `handoff_2026-08-25_histopolation.md` §9.

**3.6 `poincare_vacuum.py` unverified since its refactor**, and neither Poincaré
verification is reproducible from the repo — `slurm/job_*` is gitignored and
`job_poincare_render_check.sh` was never committed. Its job also ran without
`PYTHONPATH`, so it verified the main checkout rather than the branch. Re-run it.

**3.7 The polish prediction is unresolved.** Fixing the jacobi pinning should make the
inverse-iteration polish neutral-or-helpful at k=1,2, but no run isolated polish from
the preconditioner. Consistent, not demonstrated.

## 4. Where the folding time goes — the sharpest open question

Production logs show XLA constant-folding alarms individually exceeding **2 s** at
res16 in a real relaxation run. The microbenchmark at that same resolution measured
**533 ms total**. Those cannot both describe the same work, and the microbenchmark is a
LOWER BOUND, not an estimate. Prerequisite: the fd-2 capture returns 0 bytes at every
resolution, so the warning column is currently unmeasurable.
*Detail:* `audit_2026-08-25_production.md` items 1 and 2.

## 5. The two shelves

Both hold runnable experiments — question, what it decides, cheapest experiment,
GPU-hours, and what a null result means. Do not add an entry that cannot answer the
last one.

* **Preconditioners / infrastructure** — `audit_2026-08-25_production.md`, 9 items.
  The one to run first is item 7: *what fraction of a k>=1 iteration IS the
  preconditioner apply?* Never measured, ~0.5 GPU-h, and it is the denominator of every
  preconditioner comparison anyone will run here. Its null result can reopen a decision
  already taken.
* **Relaxation** — `handoff_2026-08-25_relaxation_prelim.md` §34, 11 items, P0 first.

## 6. Left undone, not questions

`except Exception: pass` in the cache warmer (`operators.py:3477`); the `outer='none'`
branch still builds a Schur apply it discards (`operators.py:2871-2885`, ~0.2 s/run, a
clarity argument not a performance one). `verify_block_jacobi.py` keeps its name and
re-exports `build_sequence` from `mrx.geometries` because 22 scripts import it there.
