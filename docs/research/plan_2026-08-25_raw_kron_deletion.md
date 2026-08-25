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

## 1b. Two liveness traps, for whoever runs the next A/B here

Both must be guarded, and the second one is the nastier because it survives a
correct swap:

* **No-op SWAP.** The change fails to take effect and the two arms are the same
  preconditioner. Guard: assert the two probed diagonals DIFFER before believing
  anything downstream.
* **No-op COMPARISON.** The swap works, but `outer='jacobi'` calls
  `_build_schur_outer_jacobi_diaginv` with `allow_stored_tensor_diaginv=True`, so
  if a mode-matched diagonal was preassembled BOTH arms silently reuse that one
  stored vector. Iteration counts then match for a reason unrelated to either
  preconditioner. Guard: assert no stored diagonal is present before the merit
  runs.

Only the first is obvious. `scripts/debug/schur_probe_ab.py` asserts both.

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
