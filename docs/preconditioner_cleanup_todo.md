# Preconditioner Cleanup TODO

Companion to [`preconditioner_primer.md`](preconditioner_primer.md). Two
sections: dead code that can be removed, and oddities/changes to consider.

## Status (resume here)

The tensor mass preconditioner has been compacted to **rank-1 only**, with
optional Richardson sweeps that default to 0:

- `TensorDiagonalBlockInverseFactors` no longer carries `modal_basis_*`,
  `modal_r/t/z`, or `modal_denom`. Only the rank-1 direct-Kronecker path
  (`direct_inv_r/t/z`) remains.
- `_apply_tensor_diagonal_block_preconditioner` is rank-1-only and raises
  if `direct_inv_r is None`.
- `_build_diagonal_tensor_block_factors` forces `rank = 1` at entry and
  raises if more than one separable term is produced. The modal/multirank
  branch (`_assemble_shared_modal_basis`, `eigh` per axis,
  `modal_denom`) is gone.
- `_apply_tensor_diagonal_block` keeps the optional true-block Richardson
  loop (`true_block_apply` callable threaded through
  `_apply_tensor_exact_block`) plus the omega autotune via Lanczos. With
  the static `richardson_steps=0` default the loop is constant-folded
  out under jit.
- `assemble_all_operators` now passes `k{0,1,2,3}_rank=1`.
- `_major_radius_prior_terms` is called with `rank=1` at all six sites
  (the 2x rank-inflation bug from earlier in the multiplicative path).

What this buys us in benchmarks: rank-1 production matches the old
modal multirank path (~13 iters at k=1, ~10 at k=2) at ~1/3 the per-apply
wall time, because both now run the same direct-Kronecker rank-1
smoother. Per-block Richardson sweeps were measured to be a no-op (omega
autotunes to ~1, spectrum already clustered around 1) except for k=3
rank=3 which saved one iter. Detailed numbers in the conversation
transcript at
`/home/tblickhan/.vscode-server/data/User/workspaceStorage/820340d02833ca378f5d79bf88c8c3f9-1/GitHub.copilot-chat/transcripts/284a9347-7d05-4ea1-a9e1-cd1483ebe6c4.jsonl`.

### Next session pickup

1. **Test the compacted code.** Cluster was down at handover so this is
   the first thing to do:
   ```
   pytest test/test_operators.py -k "tensor or mass_preconditioner or k2_schur" -x --tb=short
   ```
   The fixture builds at `rank=3` but each per-degree call now hard-forces
   `rank=1`; tests should still pass since the behaviour at `rank=1`
   matches the previous `len(term_matrices)==1` branch.
2. **Prod-vs-Jacobi-vs-Chebyshev mass benchmark** — the user-requested
   next step. Mirror `scripts/benchmark_richardson_vs_modal.py`: build
   sequence + bare mass operators once, then for each `k` run
   `solve_singular_cg(apply_mass_matrix, rhs, ..., precond_matvec=...)`
   with three `precond_matvec` choices:
   - production tensor: `apply_mass_tensor_preconditioner_ops`
   - Jacobi: `apply_mass_jacobi_preconditioner_ops`
   - Chebyshev (degree 3 over Jacobi-preconditioned mass): existing
     helper in `mrx/preconditioners.py` (search for
     `_build_chebyshev_apply_preconditioner` callers).
   Report avg/max iters and wall time per (k, strategy) cell over a
   small RHS bank. Sweep at least one `(ns, p)` and one geometry.
3. **Touch up dead-code references in scripts** if they break after the
   modal-field removal: `scripts/debug_tensor_forward_models.py`,
   `scripts/interactive/debug_mass_block_spectra.py`,
   `scripts/interactive/screen_vlp_neumann_eta.py` reference the now-gone
   `term_r/t/z` and `modal_basis_r` attributes. Either delete those
   scripts (they are debug-only) or update them to use the rank-1
   `direct_inv_*` fields.
4. Then proceed with the dead-code list below.

## Dead Code To Remove

- [ ] **`mrx/relaxation_deprecated.py`** — entire file is commented out
      (verified with `grep -nE '^[^#]'` → no matches). Only reference is
      `test/deprecated/integration_tests/test_z_pinch.py`, itself in the
      deprecated tree.
- [ ] **`mrx/utils.run_relaxation_loop`** — imports `DescentMethod`,
      `MRXHessian`, `TimeStepper` from `relaxation_deprecated`, would
      `ImportError` if called. Zero callers. Live `mrx/relaxation.py`
      already supplies the modern equivalents.
- [ ] **Legacy `k = 0` fast-diagonalization Hodge stack** in
      [`mrx/operators.py`](../../mrx/operators.py):
  - [ ] `assemble_tensor_hodge_preconditioner` (also: drop the call from
        `assemble_all_operators`)
  - [ ] `_fd_apply_3d`, `_fd_apply_full`, `apply_hodge_kron_preconditioner`
  - [ ] `_fd_hodge_scales_K`
  - [ ] `SequenceOperators` fields: `fd_V_p_{r,t,z}`,
        `fd_lam_p_{r,t,z}`, `dd0_fd_scale_K`
  - Production `k = 0` Hodge uses `k0_tensor_hodge_precond` from
        `update_hodge_operator`, not these.
- [ ] **Duplicate helpers in `mrx/operators.py` and
      `mrx/preconditioners.py`** — collapse to one (keep the
      `preconditioners.py` versions, import from there):
  - [ ] `_estimate_preconditioned_max_eigenvalue_apply`
        (operators.py:3327, preconditioners.py:893)
  - [ ] `_estimate_chebyshev_lanczos_bounds_apply`
        (operators.py:3371, preconditioners.py:930) — signatures already
        diverge; bug-prone.
  - [ ] `_build_chebyshev_apply_preconditioner`
        (operators.py:3660, preconditioners.py:1456)
- [ ] **`apply_mass_rtzblock_preconditioner`**
      (`mrx/preconditioners.py:2790`) — defined, no callers.
- [ ] **Debug-only forward-model helpers** — keep only if the named debug
      scripts are still in use; otherwise drop with the script:
  - [ ] `K2TensorDivDivForwardModel`, `_apply_k2_divdiv_*`,
        `_assemble_k2_divdiv_*` (`scripts/debug_k2_divdiv_forward_model.py`)
  - [ ] `_apply_k0_tensor_hodge_forward_model` (debug only)
  - [ ] `apply_mass_tensor_forward_model_ops`,
        `apply_mass_tensor_forward_model` (debug only)

## Oddities / Changes To Consider

- [ ] **Stated direction vs. wired default mismatch.** ~~The "rank-1
      backbone + Richardson smoother" path is implemented
      (`fit_strategy="split"`, `split_backbone_inv_*`, the Richardson loop
      in `_apply_tensor_diagonal_block`) but `assemble_all_operators` only
      sets per-degree `rank=2` and leaves `richardson_steps=0`,
      `fit_strategy="multiplicative"`. Production runtime is the
      shared-modal multirank inverse, not rank-1 + correction.~~
      **Resolved**: tensor builder is now rank-1-only; modal/multirank
      retired; `assemble_all_operators` passes `k{0..3}_rank=1`. Open
      sub-task: decide whether `fit_strategy="split"` should become the
      default (currently `"multiplicative"`) — at rank 1 the two paths
      produce the same single Kronecker, but `"split"` populates
      `split_backbone_inv_*` which is required for the Richardson omega
      autotune to fire.
- [ ] **Krylov-in-Krylov in `mrx/relaxation.py:apply_diffusion`.** It
      builds `apply_A(x) = M_2 x + η · seq.apply_hodge_laplacian(x, 2, ...)`
      and wraps it in an outer `solve_singular_cg`, while
      `apply_hodge_laplacian` for `k ≥ 1` runs an inner CG to full
      tolerance. Replace with `apply_inverse_mass_plus_eps_laplace_matrix`
      or use `apply_hodge_laplacian_approx`.
- [ ] **Builder defaults `k1_inner_schur=True`, `k2_inner_schur=True`** in
      `build_mass_tensor_preconditioner` contradict the practical
      guidance in [`mass_preconditioners.md`](../mass_preconditioners.md).
      Either flip defaults to `False` or override in
      `assemble_all_operators` so behaviour does not depend on entrypoint.
- [ ] **Confusing API:** `build_mass_tensor_preconditioner(full_matrix,
      ...)` immediately does `del full_matrix`. Drop the parameter and
      update `assemble_tensor_mass_preconditioner` callers.
- [ ] **Saddle outer = `jacobi` by default** while Chebyshev is the
      stated second baseline. Decide explicitly: making
      `default_saddle_preconditioner().schur.outer` = `chebyshev` (or
      `richardson` with cached spectral bounds) may be the better
      default. At minimum document why `jacobi` was chosen.
- [ ] **Eager assembly of legacy FD Hodge.** `assemble_all_operators`
      calls `assemble_tensor_hodge_preconditioner` even though no
      production solve reads its outputs. Real GPU memory wasted on 3D
      runs. Resolved by the dead-code removal above.
- [ ] **`apply_hodge_laplacian` docstring openly says "inner `M_{k-1}^{-1}`
      solves use CG ... to full solver tolerance".** Either rename it
      (`apply_hodge_laplacian_exact`) and mark not-for-Krylov, or
      replace its dangerous call sites
      (`apply_mass_plus_eps_laplace_matrix`,
      `relaxation.apply_diffusion`) with `apply_hodge_laplacian_approx`.
      The `mrx/nullspace.py` uses are residual-norm probes and fine.
- [ ] **Rank knob duplication.** ~~`MassPreconditionerSpec` carries
      top-level `rank` plus per-degree `k{0,1,2,3}_rank`, resolved by
      `_tensor_mass_rank`. `assemble_all_operators` then overrides via
      `cp_kwargs={'k0_rank': 2, ...}` while passing `rank=1`. Works, but
      easy to misread. Consider a single `mass_ranks: dict[int, int]`
      entry.~~ **Effectively obsolete:** the builder now hard-forces
      `rank=1`, so all the rank knobs are decorative. Either remove
      them entirely (`rank`, `k{0..3}_rank`, `tensor_mass_rank_for_degree`,
      `_tensor_mass_rank`) or leave them in place as a stub for a future
      multi-Kronecker path. Dropping is recommended.
