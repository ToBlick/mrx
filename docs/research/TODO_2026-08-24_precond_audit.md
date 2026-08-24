# TODO — preconditioner audit, 2026-08-24

What is actually in production, what was deleted as dead, and what is stale and
still needs a decision. Written after the two 2026-08-24 fixes (the mass
preconditioner gate and the saddle `schur.outer`), so it supersedes the
"what runs" sections of `docs/PRODUCTION.md` where the two disagree.

---

## 1. What is in prod, verified against the code

| solve | preconditioner | resolved by |
| --- | --- | --- |
| Mass, all k | `block_jacobi` (lazy, memoised on geometry identity) | `preconditioners.default_mass_preconditioner`, `MRX_MASS_KIND` overrides |
| Laplacian k=0..3 | `block` — `BlockJacobiLaplacian`, `PRODUCTION_BC_SCALE = 3.0` | `apply_laplacian_preconditioner(kind='auto')`, needs `assemble_block_jacobi_laplacian_preconditioner` first, else falls back to `jacobi` |
| k>=1 saddle (Laplacian) | lower = `block_jacobi`; `schur.outer` = `block` when assembled, else `jacobi` **with a RuntimeWarning**; `schur.inner` = `raw_kron` (unused when outer is block) | `operators._materialize_default_saddle_preconditioner` |
| k=0 solve | deflated CG, condensed | — |
| k>=1 solve | saddle MINRES + harmonic deflation | — |
| **timestep / diffusion** `(M + eps L)` | **`jacobi`** — see item 3.1 | `operators._coerce_diffusion_preconditioner_spec` |
| **nullspace inverse iteration**, k>=1 | lower = `block_jacobi`, `schur.outer` = **`jacobi`** — see item 3.2 | `nullspace._nullspace_shifted_preconditioner` |

Env knobs that survive in prod `mrx/`: `MRX_MASS_KIND`, `MRX_BJ_BC_SCALE`,
`MRX_LAPLACIAN_DIAG_{SPLIT,RESCALE,EXACT_RINGS,PROBE}`, `MRX_WEAK_EXACT_MAXDIM`,
`MRX_CP_GREEDY`. The last four only bite on paths that are no longer production
(`kind='jacobi'` weak-term diagonal, and the retired CP fit).

---

## 2. Done today — dead code removed from prod

1441 deletions against 87 insertions across 11 modules in `mrx/` (excluding
`mrx/experimental/`). The insertions are docstring corrections; every deletion
was a module-level def with **zero** references anywhere in `mrx/`
(experimental included), `test/`, `scripts/`, `docs/`, `papers/` or `slurm/`.
Iterated to a fixpoint, so the cascades went too.

- **Preconditioner stack (~700 lines).** The CP/NTF prior-fit cluster in
  `operators.py` (`_fit_positive_rank*`, the six `_project_tensor_to_*` and
  `_solve_rank_coupled_projection`), the dense-matrix helpers around it,
  `assemble_mass_raw_kron_preconditioner` + `_raw_kron_available`, the
  entrywise raw_kron trio (`raw_kron_entry`, `build_raw_kron_pinv_columns`,
  `raw_kron_extracted_entry`), the prior-term machinery in
  `preconditioners.py`, `_assemble_shared_modal_basis`, `_schur_blocks`,
  `tensor_mass_rank_for_degree`, and the `set_mass_rtzblock_*` tombstones.
- **The k=0 modal-radial atom.** `assemble_k0_blockdiag_preconditioner` was
  referenced only from a docstring; deleting it orphaned
  `_assemble_k0_modal_radial_bulk_factors` and `_k0_radial_profiles`, and left
  the `modal_*` fields of `K0TensorHodgePreconditionerFactors` with no producer
  — so `if factors.modal_W is not None` in
  `_apply_k0_tensor_hodge_bulk_inverse` had been **dead by dataflow** already.
  Fields, apply and branch all removed. Numbers live in
  `docs/research/mass_preconditioner_pivot.md` §7 and
  `mrx/experimental/modal_radial.py`.
- **Outside the stack (~500 lines).** `mrx/utils.py` reduced to its re-export
  shim (`run_relaxation_loop` and `update_config` were already **broken** —
  F821 on `norm_2` and `DEVICE_PRESETS`), plus dead functions in
  `assembly.py`, `circulation.py`, `io.py`, `plotting.py`, `projectors.py`,
  `relaxation.py`, `nullspace.py`, `block_jacobi_laplacian.py`.

Checks: every `mrx/` module parses; `ruff` findings **125 -> 117** with **zero**
new ones; the fixpoint scan reports 0 dead module-level defs remaining; a
cross-module import check (every `from mrx.x import y` still resolves) shows no
new breakage against HEAD; `slurm/job_deadcode_tests.sh` runs the import sweep
+ full `pytest test/` on a GPU node.

### 2.1 The trap this hit, worth a permanent note

`ruff --fix` deleted `_core_size` from `operators.py`'s import block as F401 —
but `mrx/experimental/k0_core_schur.py` and two debug scripts import it **from
`mrx.operators`**. The re-export is load-bearing and lint cannot see it. It is
now marked `# noqa: F401` with a comment. `run_ruff` auto-fixes, so this will
recur: **run the cross-module import check after any `--fix` pass**, not just
ruff.

---

## 3. Flagged, NOT changed — these need your decision

### 3.1 The production timestep solve never got either 2026-08 swap — **top item**

`relaxation.py:240` -> `apply_inverse_mass_plus_eps_laplace_matrix` ->
`_build_diffusion_preconditioner_apply`, whose `valid_kinds` are
`('none', 'jacobi', 'tensor')`. It **cannot accept `block_jacobi`**, and at
k>=1 the saddle upper block is built by the same function, so it never sees the
block Laplacian atom either.

So the path `PRODUCTION.md` calls the production timestep solve runs on the
per-DoF diagonal, while the pure-Laplacian path it says "collapses to one-off
solves" is the one that got the 2.5x. Its own docstring still says
"diffusion preconditioners currently use the same mass-side defaults as the
other inverse paths: Jacobi and tensor" — accurate, and no longer what the
other inverse paths do. Cost is unmeasured. This is where production time is
actually spent.

### 3.2 Inverse iteration is pinned to `schur.outer='jacobi'`, and *rejects* `block`

`nullspace._nullspace_shifted_preconditioner(k>=1)` hard-codes the jacobi
outer, and `_validate_nullspace_shifted_preconditioner` raises on
`kind='block'`. `find_nullspace_vectors` therefore does **not** inherit the
assembled atom — it overrides it.

This matters right now: `slurm/job_invit_rerun.sh` (launched 16:18, the S5
gate) carries the comment *"compute_nullspaces now assembles the block atom, so
find_nullspace_vectors picks it up automatically"*. The assembly happens; the
pickup does not. The **direct** route does go through
`apply_inverse_hodge_laplacian` and does get `block`; only the inverse-iteration
polish is pinned. Those are exactly the two routes the S5 gate compares, so
read its output with that in mind.

Left unchanged on purpose: the shifted operator is `S_k + eps M_k`, not `L_k`,
so the atom's fit there wants measuring, and flipping it mid-sweep would
invalidate the S5 numbers. The rejection list also still names the retired
`tensor` kind at k=0 and omits `block`. In-code note added at the site.

### 3.3 The paper's convergence study runs the retired stack

`scripts/config_scripts/test_torus_poisson_all_k_sparse.py` (2026-08-14, the
all-eight-(k,BC) study) hard-codes

```python
mass=MassPreconditionerSpec(kind='tensor', surgery_schur=True)
schur=SchurPreconditionerSpec(inner=MassPreconditionerSpec(kind='tensor'))
```

plus `assemble_tensor_mass_preconditioner(..., rank=1, cp_kwargs=...)` — the
stack retired on 2026-08-17 and replaced again on 2026-08-22. The eight per-k
scripts beside it (2026-06-17) do the same. This is the same class of error as
the two found this morning: the script does not solve what production solves.

These scripts are also the **only** thing keeping `kind='tensor'` and the CP
machinery reachable at all. Decide: repoint them at the production defaults
(and the tensor path becomes deletable), or keep them and mark clearly that
their preconditioner is not production.

### 3.4 `default_saddle_preconditioner()` does not describe the default saddle preconditioner

Two functions, one name-shape, disagreeing content:

- `preconditioners.default_saddle_preconditioner()` -> bare
  `SaddlePointPreconditionerSpec()` = mass `raw_kron`, inner `raw_kron`, outer
  `jacobi`.
- `operators._materialize_default_saddle_preconditioner()` -> the real default
  = mass `block_jacobi`, outer `block`.

The first is used **only** to pick which kinds `warm_mass_preconditioner_cache`
warms. Harmless today (it over-warms), actively misleading to read. Root cause
is that the field default `MassPreconditionerSpec.kind = 'raw_kron'` never
moved when `default_mass_preconditioner()` did — so every bare
`MassPreconditionerSpec()` is still raw_kron.

### 3.5 The `jacobi` baseline is mis-calibrated, and it is the paper's baseline

Already in `PRODUCTION.md` as a KNOWN REGRESSION: `build_weak_term_diagonal` is
still calibrated for the raw_kron mass, so `kind='jacobi'` costs 1-10% more
than it used to, and `test_weak_term_diagonal_matches_exact_rows` skips unless
the mass is raw_kron. That was a footnote when jacobi was production. It is not
one now: **jacobi is the comparison arm in sweep S1**, so a mis-calibrated
baseline flatters the block atom by up to 10%. Recalibrate or state the bias.

### 3.6 `bc_scale = 3.0` and its basin need re-measuring

`PRODUCTION_BC_SCALE = 3.0` with "basin flat over [2,4]" — but that basin was
measured on `apply_hodge_laplacian_approx` with an unpreconditioned lower
block, i.e. the wrong operator (see `sweep_plan_2026-08-24.md` S2). The value
is not wrong, it is *unsupported*. Sweep S2 is the fix.

### 3.7 Stale claims fixed in place today (no behaviour change)

`MassPreconditionerSpec`'s kind comment (said raw_kron was production and
block_jacobi was not the default), `_mass_block_jacobi_for`'s "NOT YET THE
DEFAULT", `apply_inverse_mass_matrix_ops`'s "the default is tensor when
assembled", `_resolve_legacy_mass_preconditioner`'s raw_kron comment,
`_assemble_k0_tensor_hodge_preconditioner`'s "Production k=0 Laplacian
preconditioner", and the nullspace comment claiming a raw_kron lower mass.

Still stale, deliberately left because fixing the words would hide the
behaviour question they sit on: `_materialize_default_scalar_hodge_preconditioner`
("Jacobi at every k as of 2026-08-18") and `_coerce_diffusion_preconditioner_spec`
("Production default: tensor for the scalar k=0 Laplacian") — both belong to
item 3.1.

---

## 4. Order

1. **3.1** — measure the timestep solve with the block atom + block_jacobi
   mass. It is the production hot path and nothing has been measured there.
2. **3.5** — recalibrate the jacobi weak-term diagonal, or state the bias,
   before S1 becomes a paper table.
3. **3.3** — repoint or relabel the convergence scripts; if repointed, the
   `tensor` kind and the CP machinery can leave `mrx/preconditioners.py`.
4. **3.2** — measure `block` as the inverse-iteration outer, after the S5 gate
   lands.
5. **3.4** — collapse the two "default saddle preconditioner" functions.
6. **3.6** — falls out of sweep S2.

`docs/PRODUCTION.md` needs a revision pass once 3.1 and 3.2 are decided; it is
stamped 2026-08-22 and predates both of today's fixes.
