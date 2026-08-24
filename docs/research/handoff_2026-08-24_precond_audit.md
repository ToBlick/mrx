# Handoff 2026-08-24 (evening) — preconditioner audit, prod dead-code prune, and the S5 gate result

Picks up from `handoff_2026-08-24_harmonic_k1_free.md` (morning: the two bugs)
and `sweep_plan_2026-08-24.md` (the re-measurement plan). Two things landed
this evening: **the S5 nullspace gate came back and it is decisive**, and the
production package was audited and pruned. Action list lives in
`TODO_2026-08-24_precond_audit.md`; this document is the state of play.

**Read alongside `handoff_2026-08-24_session.md`** (committed 17:14 by the
parallel session). Two points of contact, neither a conflict:

* It reaches the same conclusion about inverse iteration from the other end —
  *"one sweep replaces the vector with the shifted solve's output, so it can
  never beat that solve's accuracy"*, with the k=1 free shifted solve measured
  at ~1e-04. §1.2 below supplies the CAUSE of that ceiling: the shifted solve
  is pinned to `schur.outer='jacobi'` and its validator rejects `'block'`.
  Independent routes, same answer.
* It still quotes the p=2/3/5 degradation table (1.7e-01 at W7-X p=5). §1.1
  retires that table. Where the two disagree, this one is later.

**Provenance of the numbers in §1:** the `invit_rerun` jobs launched at 16:18;
`6ab7117` ("terminate inverse iteration on the Rayleigh quotient") landed at
16:53. So the table below was produced by the PRE-`6ab7117` `nullspace.py`,
which terminated on `||L_k v||`. That changes when polish stops, not the
`relL2_direct` column and not the mechanism — but re-run before quoting the
polished column as current.

---

## 1. S5 gate: RESOLVED, and it overturns this morning's diagnosis

`outputs/invit_rerun/2026-08-24/16-18-02/`, six geometries, 12x24x12, **p=5**,
inner tol 1e-13, 100 sweeps.

```
geom          k bc       direct   polished  verdict      s
hegna         0 free   1.36e-16   1.57e-16  same       0.0
hegna         1 free   4.03e-05   7.09e-05  same     176.1
hegna         2 dbc    3.18e-11   1.10e-06  WORSE    144.4
hegna         3 dbc    7.36e-12   2.74e-13  better    61.5
quasr44970    0 free   8.88e-16   1.09e-15  same       0.0
quasr44970    1 free   1.49e-10   6.14e-05  WORSE    176.5
quasr44970    2 dbc    1.08e-10   4.95e-07  WORSE    144.2
quasr44970    3 dbc    2.20e-11   2.16e-12  better    61.1
quasr9983     0 free   1.09e-15   1.09e-15  same       0.0
quasr9983     1 free   6.11e-11   5.27e-08  WORSE    176.7
quasr9983     2 dbc    1.08e-10   6.49e-11  same     205.3
quasr9983     3 dbc    4.46e-11   3.37e-12  better    34.5
rot-ellipse   0 free   3.85e-16   1.57e-16  same       0.0
rot-ellipse   1 free   8.80e-11   2.79e-10  same     176.1
rot-ellipse   2 dbc    6.50e-11   1.70e-11  same     143.4
rot-ellipse   3 dbc    6.89e-12   9.98e-13  same      40.7
toroid        0 free   3.14e-16   3.85e-16  same       0.0
toroid        1 free   5.77e-11   2.99e-13  better   104.5
toroid        2 dbc    1.19e-12   1.28e-12  same     132.9
toroid        3 dbc    4.93e-12   7.43e-13  same      17.2
w7x           0 free   3.68e-16   3.28e-16  same       0.0
w7x           1 free   1.99e-11   1.28e-04  WORSE    176.0
w7x           2 dbc    4.82e-11   1.52e-06  WORSE    205.5
w7x           3 dbc    7.32e-12   6.13e-13  better    42.7
```

### 1.1 The harmonic form is FIXED — it never degraded with `p`

W7-X k=1 free at p=5: **1.7e-01 this morning, 1.99e-11 now.** Ten orders of
magnitude, and rot-ellipse p=5 went 3.0e-02 -> 8.8e-11. Five of six geometries
now sit at 1e-10..1e-11.

So this morning's headline — *"degrades with p, not with the device"* and
*"at p>=3 the solve floors and more accuracy is unobtainable"* — was an
artefact. The solve floored because it had no mass preconditioner
(`_materialize_default_mass_preconditioner`'s `_tensor_available` gate) and a
per-DoF-diagonal Schur outer. Fix both and the floor is gone. **Retire that
table**; do not carry it into the paper.

The one survivor: **hegna k=1 free is still 4.0e-05**, alone among the six and
unmoved by polish. That is now the only nullspace-quality question open, and it
is a geometry question, not a `p` question.

### 1.2 Inverse-iteration polish is HARMFUL at k=1,2 — and here is the cause

Polish never improves k=1 or k=2 except on the toroid; it is 5-7 orders WORSE
on the two hardest cells (w7x k=1 free 1.99e-11 -> 1.28e-04, w7x k=2 dbc
4.82e-11 -> 1.52e-06), and it costs 140-205 s per cell to get there. At k=3 it
helps in 5 of 6; at k=0 it is a no-op.

`handoff_2026-08-24_session.md` states the shape of this independently: a sweep
replaces the vector with the shifted solve's output, so it can never beat that
solve, and at k=1 free that solve reaches only ~1e-04. What follows is why the
ceiling is 1e-04 rather than 1e-11.

**Mechanism, found in the audit:**
`nullspace._nullspace_shifted_preconditioner(k>=1)` hard-codes
`schur.outer='jacobi'`, and `_validate_nullspace_shifted_preconditioner`
*raises* on `kind='block'`. So the direct route goes through
`apply_inverse_hodge_laplacian` and gets the block atom, while the polish
route's inner shifted solves are pinned to the per-DoF diagonal. Polish drags a
good direct vector down to what its own weaker inner solve can support — which
is exactly the k=1/k=2 cells where the morning's ledger showed jacobi stalling.

`slurm/job_invit_rerun.sh` says *"compute_nullspaces now assembles the block
atom, so find_nullspace_vectors picks it up automatically"*. The assembly
happens; the pickup does not, because this spec overrides it.

**Consequence for the sweeps:** S5 is satisfied by the DIRECT route alone at
p=5. Gate S1/S2/S4 on `relL2_direct`, and do not run polish for k=1,2. Fixing
the pin (item 3.2 in the TODO) is a measurement of its own, not a prerequisite.

### 1.3 What this retires

- The p=2/3/5 degradation table in `handoff_2026-08-24_harmonic_k1_free.md`.
- "Not fixable by inverse-iteration polish ... it walks AWAY." True, and now
  explained: it walks away because its inner solve is the unfixed one.
- The `tensor_preconditioners.md` §8 claim that inverse iteration seeded from
  the direct vector reaches ~3e-24 independent of `h`. Not reproduced at p=5 on
  any shaped geometry with the current wiring.

---

## 2. Production package audit and prune

Full findings and priority order: **`docs/research/TODO_2026-08-24_precond_audit.md`**.
Summary of what changed on disk (uncommitted at handoff time).

### 2.1 What is in prod, verified against code

Mass (all k) `block_jacobi`; Laplacian k=0..3 `block` at
`PRODUCTION_BC_SCALE = 3.0`; k>=1 saddle = block_jacobi lower + `block` outer
when assembled (else `jacobi` **with a RuntimeWarning**) + raw_kron inner;
k=0 deflated CG, k>=1 saddle MINRES with harmonic deflation.

Two paths did **not** get the swaps: the timestep/diffusion solve and the
nullspace polish. Both are flagged below.

### 2.2 Dead code removed — 1441 deletions / 87 insertions, 11 modules

Every deletion was a module-level def with **zero** references anywhere in
`mrx/` (experimental included), `test/`, `scripts/`, `docs/`, `papers/` or
`slurm/`, iterated to a fixpoint so cascades went with it.

- Preconditioner stack (~700 lines): the CP/NTF prior-fit cluster in
  `operators.py`, the entrywise raw_kron trio, `assemble_mass_raw_kron_preconditioner`,
  the prior-term machinery and `_assemble_shared_modal_basis` in
  `preconditioners.py`, the `set_mass_rtzblock_*` tombstones.
- The k=0 modal-radial atom: `assemble_k0_blockdiag_preconditioner` was
  referenced only from a docstring, which orphaned its bulk-factor builder and
  left the `modal_*` fields with no producer — so the `if factors.modal_W is
  not None` branch had been **dead by dataflow** already. Numbers survive in
  `mass_preconditioner_pivot.md` §7 and `mrx/experimental/modal_radial.py`.
- Outside the stack (~500 lines): `mrx/utils.py` is now only its re-export
  shim — `run_relaxation_loop` and `update_config` were already **broken**
  (F821 on `norm_2` / `DEVICE_PRESETS`) — plus dead defs in `assembly.py`,
  `circulation.py`, `io.py`, `plotting.py`, `projectors.py`, `relaxation.py`,
  `nullspace.py`, `block_jacobi_laplacian.py`.

Verification: every module parses; ruff **125 -> 117 findings, zero new**; the
fixpoint scan reports 0 dead module-level defs; a cross-module import check
shows no new breakage vs HEAD; `slurm/job_deadcode_tests.sh` runs an import
sweep (prod + all experimental) plus the full `pytest test/` on a GPU node.

### 2.3 A trap that will recur

`ruff --fix` deleted `_core_size` from `operators.py`'s import block as F401 —
but `mrx/experimental/k0_core_schur.py` and two debug scripts import it
**through** `mrx.operators`. The re-export is load-bearing and lint cannot see
it; the job caught it as an `ImportError` on the first run. Now `# noqa: F401`
with a comment. **`run_ruff` auto-fixes, so run a cross-module import check
after any `--fix` pass.**

---

## 3. Open, in priority order

1. **The production timestep solve never got either 2026-08 swap.**
   `relaxation.py:240` -> `apply_inverse_mass_plus_eps_laplace_matrix` ->
   `_build_diffusion_preconditioner_apply`, whose `valid_kinds` are
   `('none','jacobi','tensor')`. It cannot accept `block_jacobi`, and its k>=1
   saddle upper block never sees the block atom. This is the production hot
   path and it is unmeasured.
2. **The `jacobi` baseline is mis-calibrated and it is S1's comparison arm.**
   `build_weak_term_diagonal` is still calibrated for raw_kron (known
   regression in `PRODUCTION.md`), so jacobi costs 1-10% more than it should —
   which flatters the block atom by up to 10% in the headline table.
   Recalibrate, or state the bias in the paper.
3. **The convergence study runs the retired stack.**
   `scripts/config_scripts/test_torus_poisson_all_k_sparse.py` hard-codes
   `kind='tensor', surgery_schur=True`; the eight per-k scripts do the same.
   Same class of error as the two found this morning. These scripts are also
   the only thing keeping `kind='tensor'` and the CP machinery reachable — if
   they are repointed, that path can leave `mrx/preconditioners.py`.
4. **Wire `block` as the inverse-iteration shifted outer** and re-measure §1.2.
   Note the shift is `S_k + eps M_k`, not `L_k`, so the atom's fit there wants
   measuring rather than assuming.
5. **Collapse the two disagreeing "default saddle preconditioner" functions**
   (`preconditioners.default_saddle_preconditioner` vs
   `operators._materialize_default_saddle_preconditioner`). Root cause is that
   the field default `MassPreconditionerSpec.kind = 'raw_kron'` never moved.
6. **`bc_scale = 3.0` and its `[2,4]` basin** are unsupported, not wrong — the
   basin was measured on the approx operator with an unpreconditioned lower
   block. Sweep S2 settles it.

`docs/PRODUCTION.md` is stamped 2026-08-22 and predates both of today's fixes;
it needs a revision pass once 1 and 4 are decided.

---

## 4. State at handoff

- **Regression: GREEN.** Job `16729137` ran the import sweep (prod + every
  `mrx/experimental` module) plus the full `pytest test/` on a GPU node and
  finished `COMPLETED, ExitCode 0:0` in 13:50 —
  `outputs/deadcode_tests/2026-08-24/17-03-38/deadcode.log`. The wrap runs under
  `set -euo pipefail`, so exit 0 IS the pass signal; there is no "N passed" line
  because the command passes `-q` twice, which suppresses the summary. One
  skip, four xfails, no failures.
- The harness is `slurm/job_deadcode_tests.sh`. It is NOT committed —
  `.gitignore` carries `slurm/job_*` and only four legacy scripts are tracked —
  so it lives on disk only. Re-run it after any further prune.
- Nothing was changed that alters numerical behaviour: the prune removed only
  unreferenced defs, and the six stale docstrings corrected in place are
  comments. The two behavioural staleness items (§3.1, §3.4) were deliberately
  left alone and documented at the call sites instead.

Committed on `greville-prod`: the prune and the docstring corrections in one
commit, these two documents in another.
