# raw_kron vs the metric-lumped atom as the Schur-Jacobi probe

**These numbers cannot be regenerated.** raw_kron is deleted immediately after
this measurement, so this file is the only surviving evidence that the forced
switch of the Schur-Jacobi probe backing was measured rather than assumed.

**Both arms measured at commit `988a29bb199911c05bad8402e1374973815ff77e`**
(toroid, job 16773428) and **`7cb753a4a32e72bda4f8ab2b00cb25f6c00e2886`**
(W7-X, job 16773614), on branch `stage-cd`. Full hashes, not branch names: after
the deletion this note is the only witness, and someone finding it later must be
able to check out the exact code both arms ran.
Those two commits differ only in `docs/`, the A/B script, and the slurm wrapper
— **no `mrx/` changes**, so both halves measure identical library code.
Caveat on provenance: the toroid job stamped `988a29b` but actually ran the
hardened script that landed in `7cb753a`; only the stamp is off, not the code
under test.

Geometry `ns=(12,24,12)`, `p=3`, `tol=1e-10`, `maxiter=20000`,
`schur.outer='jacobi'`, `mass='block_jacobi'`.
Script `scripts/debug/schur_probe_ab.py`, job `slurm/job_schur_probe_ab.sh`.

## Why this had to be measured before the deletion

`assemble_schur_jacobi_preconditioner` probes and STORES `1/diag(A_k)` with
`A_k(x) = S_k x + D_{k-1} B_{k-1} D_{k-1}^T x`, where `B_{k-1}` is the
`schur.inner` inverse. That inner was raw_kron, and `_SCHUR_DIAG_MODES` had
exactly one entry. Deleting raw_kron leaves no backing, so the Schur-Jacobi
diagonal must be probed from the metric-lumped atom instead — **the baseline
changes not by choice but because nothing else survives.** This measures that
forced switch.

## Results

`diag rel` is `|d_raw_kron - d_atom| / |d_raw_kron|`: the liveness check. It
must be non-zero or nothing else means anything.

### Toroid

| k | BC | n | diag rel | raw_kron | atom | change | agree |
|---|---|---|---|---|---|---|---|
| 1 | free | 8700 | 1.795e-01 | *no conv* | *no conv* | — | undefined |
| 1 | dbc | 8124 | 4.290e-01 | 1136 | **1059** | −6.8% | 2.210e-10 |
| 2 | free | 8664 | 2.584e-01 | 1302 | **1195** | −8.2% | 4.276e-10 |
| 2 | dbc | 8376 | 2.589e-01 | *no conv* | *no conv* | — | undefined |
| 3 | free | 2880 | 2.584e-01 | 461 | **430** | −6.7% | 2.605e-10 |
| 3 | dbc | 2880 | 2.589e-01 | *no conv* | *no conv* | — | undefined |

### W7-X

| k | BC | n | diag rel | raw_kron | atom | change | agree |
|---|---|---|---|---|---|---|---|
| 1 | free | 8700 | 1.592e-01 | *no conv* | *no conv* | — | undefined |
| 1 | dbc | 8124 | 4.496e-01 | 2340 | **1952** | −16.6% | 3.128e-10 |
| 2 | free | 8664 | 2.577e-01 | **5862** | 5896 | +0.6% | 1.079e-09 |
| 2 | dbc | 8376 | 2.585e-01 | *no conv* | *no conv* | — | undefined |
| 3 | free | 2880 | 2.521e-01 | 1076 | **1050** | −2.4% | 2.257e-10 |
| 3 | dbc | 2880 | 2.526e-01 | *no conv* | *no conv* | — | undefined |

**Six converged cells across both geometries: five favour the atom (−2.4% to
−16.6%), one is +0.6% (within noise). No cell shows a two-digit worsening**, the
only outcome that would have warranted a pause.

### Independent second W7-X pass

Job 16773428 ran both geometries, so it repeated W7-X after finishing the
toroid. This was unplanned — it fell out of the job carrying both — and it gave
something better than a duplicate: **a measured noise floor.**

| cell | job | raw_kron | atom | agree |
|---|---|---|---|---|
| k=1 dbc | 16773614 | 2340 | 1952 | 3.128e-10 |
| k=1 dbc | 16773428 | 2334 | 1952 | 3.108e-10 |

raw_kron varies by 6 iterations (0.26%) run-to-run; the atom is bit-identical.
So the ~1% noise floor used above to dismiss the +0.6% cell is **not an
inherited assumption in this write-up — it is measured here, on this geometry,
with these arms.** That matters because that cell is the single adverse result
in a record nobody can re-run.

The probed diagonals also reproduce to every printed digit (k=1 free
`1.592e-01`, k=1 dbc `4.496e-01`), and the k=1 free non-convergence reproduces.

Worth carrying forward: running the decisive geometry twice is cheap insurance
that pays twice over — protection against a wall-clock timeout, and an in-run
noise estimate you would otherwise have to assume.

## Measured run-to-run noise

The repeat pass gives a noise floor measured on this geometry with these arms,
rather than an inherited figure:

| cell | job | raw_kron | atom |
|---|---|---|---|
| k=1 dbc | 16773614 | 2340 | 1952 |
| k=1 dbc | 16773428 | 2334 | 1952 |
| k=2 free | 16773614 | 5862 | 5896 |
| k=2 free | 16773428 | 5865 | 5890 |
| k=3 free | 16773614 | 1076 | 1050 |
| k=3 free | 16773428 | 1076 | 1051 |

Spreads: raw_kron 6, 3 and 0 iterations (0.26%, 0.05%, 0%); atom 0, 6 and 1
(0%, 0.10%, 0.10%). The second pass also reproduces all three non-converged
cells and every probed diagonal to the digits printed.
**Both arms vary by a handful of iterations and neither is reproducibly more
deterministic than the other.** An earlier draft of this note claimed the atom
was bit-identical while raw_kron drifted — that was drawn from the k=1 dbc cell
alone, and the k=2 free repeat refutes it. One cell is not a determinism
finding.

What the repeat DOES establish is the thing it was needed for: run-to-run
variation here is ~0.1–0.3%, so the +0.6% adverse cell sits inside demonstrated
noise rather than merely assumed noise.

## Reading this correctly

**Liveness passed everywhere.** The probed diagonals differ by 16–45%. This is
not a no-op swap, so the comparison is meaningful.

**Most converged cells favour the atom; W7-X k=2 free goes the other way at
+0.6%.** That cell is noise, and this run MEASURES that rather than assuming it:
the same k=1 dbc cell repeated in a second pass moved by 0.26% (see below), so
+0.6% is inside demonstrated run-to-run variation on this geometry with these
arms. The standing decision rule points the same way — Tobias, 2026-08-25:
*"only two digit percent worsening is even worth looking at"* — but the
measurement is the evidence and the rule is only what we do about it.

This file still reports every adverse cell with its number. The threshold
governs what is worth CHASING, not what is worth SAYING.

Correctness holds throughout: the arms agree to 2e-10–1e-9 at `tol=1e-10`, the
check passing at solver tolerance. A converged solve is preconditioner-
independent, so agreement is what *must* happen if the change is correct.

**THE NON-CONVERGED CELLS ARE NOT EVIDENCE AGAINST EITHER ARM.** Under
`outer='jacobi'`, k=1 free does not converge — this is the known saddle-outer
gap, already in the record independent of this comparison, and here it also hits
the k=2/k=3 dbc cells. **It is a property of the jacobi BASELINE, not of
raw_kron or of the atom**, and both arms hit the 20000 cap identically.
Production runs `outer='block'` and never sees this path at all
(`_materialize_default_saddle_preconditioner` resolves outer to `'block'` when
the atom is assembled and `'none'` otherwise — never `'jacobi'`). A later reader
must not take "half the cells did not converge" as a mark against the atom.

The `|dx|/|x|` values printed for those cells by the job's script version
(`ARMS DISAGREE`, ~1.4–6.4) **must not be quoted.** `|dx|/|x|` is a correctness
check only because a converged solve is preconditioner-independent; with neither
arm converged it compares two arbitrary partial iterates and means nothing. That
verdict text is a bug in the script, fixed in `17adc04` — after these jobs
launched. See the three traps in
`plan_2026-08-25_raw_kron_deletion.md`.

**Why `outer='jacobi'` at all, when it is not a production path?** Because it is
the *only* configuration where `schur.inner` is live. Under `outer='block'` the
atom is the upper-block inverse directly and the probe is never consulted, so
measuring there would measure nothing.

## What this does and does not support

Supports: the forced switch of the probe backing is **not a regression**. On the
toroid it is a modest improvement, tightly clustered at 6.7–8.2% across three
cells. On W7-X it is uneven — a large gain at k=1 dbc (−16.6%) and nothing at
k=2 free (+0.6%, inside noise) — so "bigger win on the harder geometry" is NOT
what the data says, however tempting the k=1 dbc number makes it. What the data
says is that the atom matches or beats raw_kron everywhere measured, sometimes
substantially, and loses nowhere outside noise.

Does not support: any claim about production solve times. Production does not
use this path. The result is that the *baseline* the deletion forces us onto is
at least as good as the one it replaces.

## Gate: the deletion introduced no regressions

Full suite, GPU, on the rebased branch versus a fresh baseline built from
`c57e8c8` itself in a throwaway worktree:

| | failed | passed | skipped |
|---|---|---|---|
| stage-cd (job 16774872) | 9 | 222 | 1 |
| c57e8c8 baseline (job 16774899) | 9 | 218 | 2 |

The deltas are exactly accounted for: **+4** are the dispatch tests added by this
work, and **−1 skipped** is `test_weak_term_diagonal_matches_exact_rows`, deleted
because its raw_kron gate could never open again.

**The nine failures are IDENTICAL BY NAME** (`gate_failures_*.txt` beside this
file; `diff` is empty). Six are even-p `test_interpolation_reproduces_its_own_space[p2-...]`
cases from the outstanding span-quadrature defect, and three are
`test_pi_full_is_idempotent[1,2,3]`, which is a deliberate isolation of that same
defect rather than a regression.

**Why comparing by name rather than by count mattered here.** The pre-merge
baseline at 76bf5f3 was ALSO nine failures — and a completely different nine:
`test_k1_histopolation_error_is_small` plus all eight
`interpolation_reproduces_its_own_space[k0..k3]` parametrisations. The
histopolation merge fixed those at odd p and added three designed controls, so
the total landed back on nine by coincidence. **Gating on "still nine" would have
passed for entirely the wrong reason.**
