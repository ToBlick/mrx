# Production audit, 2026-08-25 — what shipped, and the shelf of open questions

The audit's worklist is **done**: A1, A2, A3, A4 and all of B are shipped
(`b58bfaf`, `bcd750b`, `5f4f609`, `13e39e5`, `c6e9343`, `bb7676a`, `4ce889c`,
`4aec4fc`, `1de98c4`). ~5,900 lines of dead code and stale claims removed, plus
the eqx payload refactor and `build_preconditioners`. What remains is this
shelf.

**These are experiments, not unknowns.** Each says what it decides, how to
decide it, and what a null result means — that last line is the one people skip
and the one that stops a sweep being run twice. **Class A** restores or confirms
something already half-known; **Class B** opens new territory.

**Adding an entry: write the null-result line first.** If the answer changing
nothing leaves you with nothing to write there, the entry should not exist. Two
candidates were dropped from this list on that test, and it only bites once you
are forced to say what the answer would MEAN rather than what it would be.

**Gating a change: diff the failures BY NAME. The count is not evidence.**
Three baselines on 2026-08-25 — 76bf5f3, c57e8c8, fded2d0 — each reported
exactly **nine** failures, and the three sets are NOT the same nine. Fixes
landed and new controls appeared in the same merges, and the total happened to
return to nine every time. "Still nine" would have read as stability on three
separate occasions and been wrong on all three. A count that keeps matching for
different reasons is worse than one that moves, because nothing prompts you to
look. `slurm/job_pytest_baseline.sh` builds a baseline from any ref in a
throwaway worktree; diff the sorted `FAILED` lines.

Nothing here is launched. Tobias, 2026-08-25: *"We are not launching anything
more, but we can collect ideas and open questions for future sweeps."*

---

## 1. Where does the folding time actually go? — **Class B, sharpest open item**

**Q.** Why does a real relaxation run at res16 emit eight XLA
`Constant folding an instruction is taking > 2s` alarms when the microbenchmark
at the same resolution measures 533 ms of compile in total?

**Decides.** What preconditioner setup costs in production. Those two numbers
differ by a large factor and cannot both describe the same work, so the real
cost of the constants path is currently unknown — the microbenchmark is a
**lower bound, not an estimate**, and must not be quoted as one.

**Experiment.** Instrument a real `relax_prelim` launch at res16 — not a
microbenchmark — with **item 2 fixed first**. Log `mrx.__file__`, every fd-2
line, and per-compile wall time. Compare against
`scripts/debug/precond_const_vs_param_sweep.py` at the same resolution. The
likely suspects are a different code path, more preconditioners built than the
one the benchmark builds, or `gamma > 0`.

**Cost.** ~1 GPU-hour (estimate); one launch plus analysis.

**Null result.** If the real run also shows ~533 ms and no alarms, then the
alarms come from something other than the preconditioner payload and the
question moves to whatever else that job compiles — still progress, and it
retires a lead rather than leaving it open.

## 2. Fix the fd-2 capture — **Class A, prerequisite for item 1**

**Q.** Why did `capture_fd2` return **0 bytes at every resolution** while the
same class of message appears in production logs?

**Decides.** Whether the XLA-warning column in any sweep means anything. Until
this works, that column reads UNKNOWN, never "no".

**Experiment.** No GPU needed. Establish a positive control first: emit a known
message on fd 2 from C++-level code inside the context and assert it is
captured. Then check whether absl/XLA writes to fd 2 at all under these settings
(it may go to a log file, or need `--logtostderr` / `TF_CPP_MIN_LOG_LEVEL`), and
whether JAX's logging is initialised before the redirect takes effect.

**Cost.** ~0 GPU-hours; a local iteration.

**Null result.** If a control message IS captured and XLA's still is not, then
XLA is not writing to fd 2 here and the capture approach is wrong rather than
broken — switch to reading the job's stderr file after the fact.

## 3. S1's missing fluke guard: (3, dbc) — **Class A, bounded**

**Q.** Does the metric_lumping atom also beat jacobi on the harmonic form at
k=3 dbc, the one harmonic-bearing cell the S1 sweep never covered?

**Decides.** Whether S1's k>=1 result rests on more than k=1 free and k=2 dbc.
(1, dbc) and (2, free) were chosen as guards and turned out to have **no
harmonic form** — `_n_vectors` with betti (1,1,0,0) — so the guard is missing,
not merely uninformative.

**Experiment.** `scripts/debug/nullspace_jacobi_ab.py` already does this; add
`(3, False)`... `(3, True)` to its `cells` list and run W7-X p=3,4,5. Gate on
`rayleigh/generic` with the absolute floor, not the ratio.

**Cost.** ~1 GPU-hour (estimate).

**Null result.** If (3, dbc) is comparable, the S1 conclusion stands on its two
cells and no more; it does not weaken them, it just fails to widen them.

## 4. Confirm the k=0 shifted-fit guard removal — **Class A**

**Q.** Does the atom's advantage on `S_0 + eps M_0` hold at p=5 on both
geometries, as it did at p=3 and p=4?

**Decides.** Whether `operators.py:3232`'s `eps != 0` guard comes off. Its
stated reason — "how the atom fits the shifted operator is unmeasured" — is
already refuted 6/6, but the removal has not landed and the highest p is where
the atom's advantage was narrowest (1.29 vs 1.40 on W7-X p=5).

**Experiment.** Part 2 of `nullspace_jacobi_ab.py` at p=5,6 on toroid and W7-X.
Measuring around the guard, never through it.

**Cost.** ~1 GPU-hour (estimate).

**Null result.** If the atom loses at p>=5, the guard STAYS and its comment is
upgraded from "unmeasured" to "measured, and here is why not" — a strictly
better resting state than today either way.

## 5. What does `outer='none'` waste? — **Class A, cheap**

**Q.** How much setup time is spent building a Schur apply that
`outer='none'` immediately discards?

**Decides.** Whether to hoist that construction into its consuming branch.
`31ef58f` did this for `outer='block'` only; the `else` branch still builds and
throws away, and `'none'` is the production path when the atom is not assembled.

**Experiment.** `scripts/debug/schur_inner_waste.py` already times this shape;
point it at `outer='none'` across k=1,2,3 and both BCs.

**Cost.** ~0.5 GPU-hours (estimate).

**Null result.** If it is a few ms, close the item — it is a clarity fix worth
doing only if someone is in that code anyway.

## 6. Re-derive the weak-term diagonal bound for metric_lumping — **Class B**

**Q.** What IS the correct closed-form weak diagonal now that `M^-1` is
metric_lumping rather than raw_kron?

**Decides.** Whether `kind='jacobi'` at k>=1 is worth keeping as a baseline at
all. `build_weak_term_raw_diagonal` is calibrated for a preconditioner that no
longer exists; against metric_lumping its error grows to **22% median / 114%
max** (k=1 dbc, spline toroid 8,16,8 p=2). The test that pinned this was
deleted with raw_kron because its gate could never open again.

**Experiment.** Model the new mass rather than widening a bound: derive the
Kronecker model of the metric_lumping `M^-1`, rebuild the closed form, and
re-measure against `_probed_laplacian_diaginv`, which is exact and unaffected.

**Cost.** Unbounded — this is derivation, not a sweep. Flagged as Class B for
that reason.

**Null result.** If no closed form is materially better, then `kind='jacobi'`
at k>=1 is permanently a rough baseline and should be documented as one rather
than presented as a modelled diagonal.

## 7. What fraction of an iteration IS the preconditioner apply? — **Class A, and it should have been item 0**

**Q.** In a real k>=1 solve, what share of per-iteration wall time is the
preconditioner apply versus the operator apply?

**Decides.** The denominator for every future preconditioner comparison. Today's
entire constants-vs-parameters investigation — four jobs, two retractions — used
the preconditioner apply as its own baseline and produced break-even figures
that were arithmetically correct and practically meaningless.

Tobias settled it from knowing his own code: *"the preconditioner apply is not
the dominating cost, the dominating cost is in fact applying the operator
itself."* He was right — **and a correct judgement is not the same as a known
quantity.** There is no number. The next person comparing preconditioners will
either re-derive his judgement or, far more likely, do exactly what we did and
use the preconditioner apply as its own denominator without noticing. **Half a
GPU-hour buys the number that makes that mistake impossible.**

**Experiment.** Time `apply_hodge_laplacian_approx` against the preconditioner
apply inside one MINRES iteration, k=1,2, W7-X, p=3 and 5. Report the ratio.

**Cost.** ~0.5 GPU-hours (estimate).

**Null result.** If the two are comparable, then preconditioner-apply cost DOES
matter and the constants-vs-parameters trade should be **reopened** with the
right denominator — which is exactly the case this measurement exists to detect.
Note this is the one entry on the shelf that can **overturn a decision already
taken**, which is worth knowing before deciding it is not urgent.

## 8. `nbc_k1` converges at ~3.3, not 4 — **Class A, carried from `Poisson convergence`**

**Q.** Is the k=1 natural-BC order deficit in the projection rather than the
solve?

**Decides.** Where to look. Under-integration is already eliminated.

**Experiment.** The projection test on `omega_1` — **no solve required** — the
same move that settled k=2.

**Cost.** ~0.5 GPU-hours (estimate). Not this agent's thread; carried so it is
on one shelf.

**Null result.** If the projection is clean at order 4, the deficit is in the
solve or the BC treatment and the search narrows to those.

---

## Not questions — things simply left undone

* `warm_mass_preconditioner_cache`'s `except Exception: pass` still swallows
  build failures. Measured harmless today (8/8 pairs build on toroid and W7-X),
  and `build_preconditioners` declines to inherit it, but the original is
  untouched. Removing it is a behaviour change, not a measurement.
* `scripts/debug/verify_block_jacobi.py` keeps its pre-rename name; 22 scripts
  import `build_sequence` from it.
* `job_poincare.sh`, `job_laplacian_mg_k0.sh`, `job_mass_coupling_ceiling.sh`
  set no `PYTHONPATH` — live library-provenance traps. See
  [[worktree-jobs-need-pythonpath]]; printing `mrx.__file__` at the top of every
  wrap is the fix, because "does the wrap set PYTHONPATH" is not a sufficient
  audit question when `python script.py` and `python -m` resolve differently.
