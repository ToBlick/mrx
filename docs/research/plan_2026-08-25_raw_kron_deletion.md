# Deleting raw_kron, and what has to survive it

Ruling (Tobias, 2026-08-25): *"run the A/B first, then delete raw_kron (of course
keep plumbing that you still need for metric_lumping)"*, and separately
*"let us name the new default production preconditioner ... metric_lumping.
Then you can retire all the other names or rename things that are common
functionality and must be kept to metric_lumping"*.

So: **A/B → record → delete + rename → repoint call sites**, as four commits.
The A/B must stay reproducible against the pre-deletion SHA, so it cannot be
squashed with the deletion.

## 0. Why the A/B had to come first

`assemble_schur_jacobi_preconditioner` probes and STORES `1/diag(A_k)` with

    A_k(x) = S_k x + D_{k-1} B_{k-1} D_{k-1}^T x

and `B_{k-1}` is the `schur.inner` inverse. Today that inner is raw_kron, and
`_SCHUR_DIAG_MODES` has exactly one entry. Delete raw_kron and there is no
backing left, so the Schur-Jacobi diagonal must be probed from the metric-lumped
atom instead. **The jacobi baseline changes not by choice but because nothing
else survives.** That is the change the A/B measures, and after the deletion it
can never be measured again.

## 1. What the probe path actually does

Worth writing down, because it is not what the names suggest:

`_build_schur_probe_apply` validates its `mode` token against
`_SCHUR_DIAG_MODES` and then **ignores it**, delegating to
`_build_schur_apply_from_saddle_preconditioner`, which reads
`saddle_preconditioner.schur.inner`. The mode string is pure gatekeeping.

The only place the mode genuinely selects a backing is
`assemble_schur_jacobi_preconditioner`, which builds a `dummy_spec` from it.
**That one line is what the deletion must change** — not a missing dispatch.

## 1b. Three traps, for whoever runs the next A/B here

Only the first is obvious, and each survives the guards for the ones before it.
A bad number in a reproducible experiment is an inconvenience; a bad number in
an UNREPEATABLE one is permanent misinformation, which is why all three are
guarded rather than watched for.

**1. No-op SWAP.** The change never took effect and both arms are the same
operator. Guard: assert the two probed diagonals DIFFER before believing
anything downstream. (Measured here: they differ by 18-43%, so the swap is
decisively live.)

**2. No-op COMPARISON.** The swap took effect, but `outer='jacobi'` calls
`_build_schur_outer_jacobi_diaginv` with `allow_stored_tensor_diaginv=True`, so
a preassembled mode-matched diagonal is reused by BOTH arms and the iteration
counts are unrelated to either preconditioner. Guard: assert no stored diagonal
is present before the merit runs. Survives guard 1 -- the swap really was live.

**3. INVALID SOLVE.** Both arms are genuinely different AND genuinely measured,
but neither converged, so `|dx|/|x|` compares two arbitrary partial iterates and
means nothing. Survives guards 1 AND 2. `|dx|/|x|` is a correctness check ONLY
because a converged solve is preconditioner-independent; the moment convergence
fails, that premise is gone. Guard: judge agreement only when BOTH arms report
convergence, and report the true `||Lx - b||/||b||` per arm, which stays
meaningful either way.

Trap 3 was live in this very run and nearly reached the permanent record: k=1
free on the toroid ran both arms to the 20000 cap and the first version of the
script printed `|dx|/|x| = 1.454e+01 *** ARMS DISAGREE ***`. That is not a
disagreement between preconditioners; it is two unconverged iterates.

`scripts/debug/schur_probe_ab.py` guards all three as of `17adc04`. Note the
in-flight runs at `988a29b`/`7cb753a` predate the trap-3 fix, so their
non-converged cells carry the misleading verdict TEXT and must be annotated
rather than quoted.

**The k=1 free non-convergence is NOT caused by either arm.** It is the known
saddle-outer gap: under `outer='jacobi'`, k=1 free does not converge, which is
already in the record independent of this comparison.

## 2. Plumbing that SURVIVES, under the metric_lumping name

The chain that feeds the atom's weak-term diagonal:

    build_weak_term_raw_diagonal
      -> _weak_term_kron_terms / _weak_term_taylor_parts / _weak_term_exact_parts
        -> _weak_term_raw_terms
          -> build_mass_raw_kron_factors        <-- load-bearing, KEEP

`build_mass_raw_kron_factors` supplies the 1-D Kronecker shapes and inverses the
atom is built on. `block_jacobi_laplacian.py:900` says it outright: *"raw_kron
is already half of this shape -- M ~ Lam (A_r x A_t x A_z) Lam ... What changes
is the CORE"*. The Kronecker sandwich is shared; only the core differs.

Keep and rename: `build_mass_raw_kron_factors`, its `RawKronMassFactors`
container, and the helpers it needs (`_raw_kron_diff_flags`,
`_raw_kron_block_apply`, the `(E E^T)^-1` builder).

Rule for anything ambiguous: **keep and rename rather than delete.** A wrongly
kept helper is a cleanup task; a wrongly deleted one is a bug.

Note this is not in tension with the no-zombie rule in section 3. Keeping shared
plumbing under a NEW name is correct; what is forbidden is keeping the OLD name
pointing at it so untouched callers keep working.

## 3. What GOES — the raw_kron selection surface

* `kind='raw_kron'` from every accept-list, and its dispatch branches
  (`_build_operator_preconditioner_apply`,
  `_build_schur_apply_from_saddle_preconditioner`)
* `_raw_kron_factors_for` as a *mass preconditioner* path
* `apply_mass_raw_kron_preconditioner`, `build_mass_raw_kron_preconditioner`
* `SequenceOperators.raw_kron` cache field
* `MRX_MASS_KIND` — its whole purpose was flipping between the two arms for an
  honest A/B. With one arm left it is a knob with one setting. The reason it
  existed belongs in this document, which is why it is written here.
* `_get_schur_diaginv`'s back-compat branch keyed on `'tensor_probe'`, a mode
  `_coerce_schur_diag_mode` now rejects before it can be reached. Dead since D0,
  and a textbook instance of the rule below.

**Standing rule (Tobias, 2026-08-25), which governs all of the above:** *"I do
not like this zombie code where we have aliases in the code to keep old scripts
alive. Scripts are cheap and quickly rewritten and updated for new API.
Production should be CLEAN. You see where this mess has brought us."*

So: no aliases, no deprecated-but-accepted kind strings, no compatibility mode
added so the six call sites keep working untouched. Delete fully, repoint every
caller in the same change. The dispatch test in
`test/test_preconditioner_kind_dispatch.py` is the mechanical form of this rule.

## 4. Two defaults that stop being silent, which is the point

* `MassPreconditionerSpec.kind = 'raw_kron'` while
  `default_mass_preconditioner()` returns `block_jacobi`. Every bare
  `MassPreconditionerSpec()` silently carries raw_kron today. After the deletion
  that default names a kind that does not exist, so the inconsistency becomes a
  hard error instead of a quiet divergence. Fix in the same commit.
* `schur_diag_mode: str = 'raw_kron_probe'` — same shape, same commit. Name the
  surviving mode for what it actually is.

## 5. The six call sites

They pass `schur_diag_mode='tensor_probe'` and currently fail loudly. Repoint
them **after** the deletion, and annotate each one: the baseline they measure is
NOT the one their comments describe. `benchmark_overnight_sweep.py:569`'s claim
that *"k=1, k=2, k=3 share an IDENTICAL jacobi baseline"* stays true — all three
still share one baseline — but the baseline itself differs from the pre-Stage-B
one. Say so at the site, or the next reader compares across the boundary.

## 6. Open, not for me to decide unilaterally

`_warm_mass_preconditioner_caches` (operators.py ~3530) wraps its builds in
`except Exception: pass`. After the deletion its `kinds` set is a singleton and
the if/elif collapses to one branch — at which point a swallowed exception means
a genuine build failure is invisible, which is exactly the failure mode the
standing no-defensive-code rule exists to prevent. Removing the swallow could
also turn an unsupported (k, BC) into a hard failure during *warming* rather
than at the real call site, which is a behaviour change. **Flag to Tobias
rather than fold into the deletion.**
