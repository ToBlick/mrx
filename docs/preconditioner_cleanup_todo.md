# Preconditioner Cleanup TODO

Companion to [`preconditioner_primer.md`](preconditioner_primer.md). Two
sections: dead code that can be removed, and oddities/changes to consider.

## Status (resume here)

The tensor mass preconditioner is multirank again:

- `_build_diagonal_tensor_block_factors` now uses `_greedy_cp_terms` on the
      mapped diagonal coefficient field.
- Rank `1` is the exact direct-Kronecker inverse via `direct_inv_{r,t,z}`.
- Rank `2` is exact Lynch fast-diagonalization on the leading two Kronecker
      terms via `_build_kron_sum_fd_factors`.
- Rank `>= 3` reuses that FD basis and adds only projected diagonal
      contributions from the extra terms. This is approximate but keeps the
      same six-einsum apply as rank `2`.
- `_apply_tensor_diagonal_block_preconditioner` therefore has both the direct
      path and the FD path active again.
- The active assembled bulk path treats the coefficient fields as black boxes:
      `radial_baseline`, `prior_terms`, and `fit_strategy` are accepted for API
      compatibility but ignored inside `_build_diagonal_tensor_block_factors`.
- `assemble_all_operators` now uses eager mass rank
      `k0 = k1 = k2 = k3 = 3`.

Current reading: the mass preconditioners are essentially done. The remaining
mass work is mostly policy and validation work rather than new algorithmic
design:

- keep a few larger regression benchmarks around so the practical winner stays
      visible,
- and remove stale API knobs that no longer affect the assembled tensor route.

Latest free-boundary mass sweep worth remembering (`ns = (16, 32, 16)`, `p = 3`,
`inner_schur = off`):

- `k = 0` and `k = 3` still improved materially from rank `2` to rank `3`,
      and on this larger case `k = 0` actually looked worse at rank `2` than at
      rank `1`,
- `k = 1` and `k = 2` also now look better at rank `3` than at rank `2`, with
      rank `4` giving little extra,
- Provisional reading: rank `3` is now the leading candidate for the next mass
      default update.

Current default policy:

- stiffness (`k = 0, 1, 2`) stays at rank `1`,
- mass (`k = 0, 1, 2, 3`) now defaults to rank `3`.

The scalar `k = 0` Hodge / stiffness preconditioner is now built by the new
FD-based bulk builder `_assemble_k0_stiffness_fd_bulk_factors`:

- it fits `alpha_rr = J g^{rr}`, `alpha_thetatheta = J g^{θθ}`, and
      `alpha_zetazeta = J g^{ζζ}` separately,
- builds the leading modal basis from the rank-1 terms,
- and for higher ranks reuses that basis while adding only projected diagonal
      corrections to the additive denominator.

The old prior-based/operator-aware `k = 0` bulk builder stack in
`mrx/operators.py` has been removed.

Recent smoke checks on the rotating ellipse (`eps = 0.3`) gave about `13`
iterations for the `dbc` scalar stiffness solve and about `15` for the free
solve with the tensor preconditioner, versus about `115` for Jacobi on the
`dbc` case. Higher ranks do change the assembled bulk denominator, but on this
test geometry they do not materially change the iteration count.

Current reading: `k = 0` stiffness will stay on the tensor route. The open work
there is no longer a search for a different preconditioner family, but better
testing to decide how good the current tensor model really is across geometry,
resolution, and boundary-condition choices.

Recent k=`0` multirank checks sharpen that reading:

- higher ranks improve the stored forward model substantially,
- but on dense DBC `P K` spectra the solver-relevant gain is small,
- so the present default remains stiffness rank `1` even though the fitted
      higher-rank model is more accurate as a forward surrogate.

One note for future revisits: a bulk-local Chebyshev correction on the exact
bulk operator, using the rank-1 tensor inverse as the smoother, is still a
principled fallback option. But earlier experiments of that flavor did not pay
off in wall-clock time; it was better to let the outer CG iterations absorb the
remaining bulk error. So keep it as a future escape hatch, not as the current
plan.

The higher-form standalone stiffness preconditioner has been debugged far
enough to separate the basis/nullspace issues from the earlier small-resolution
k=`1` pathology:

- The k=`1`/k=`2` block reference masses now use the true unweighted block
      masses from the design table (`M` or `M^(d)`), not CP-weighted averages.
- The active-axis modal bases now use the canonical unweighted 1D stiffness
      operators `K = G^T M G`; inactive axes remain mass-orthonormal. The
      scalar k=`0` path was left untouched.
- The interactive stiffness notebook now checks canonical active-axis spectra,
      dense nullspaces, projected pure-stiffness solves, and dense `P K`
      spectra on the positive subspace.
- In the harmonic-free cases (`k=1` with Dirichlet, `k=2` without Dirichlet),
      the deflated nullspaces now match the dense generalized kernels:
      - k=`1`, DBC: dense null dim = exact dim = deflated dim = `44`
      - k=`2`, free: dense null dim = exact dim = deflated dim = `136`
- The pure-stiffness RHS projection is not the remaining problem. In the
      harmonic-free k=`1` DBC case, the projected trial vector, RHS, error,
      and residual all have negligible overlap with the deflated kernel.
- The k=`2` free standalone tensor stiffness preconditioner is healthy on the
      same small rotating-ellipse case: about `19` CG iterations with residual
      and solution error both at about `1e-9`.
- On the smaller `ns = (4, 8, 4)` test, the k=`1` DBC standalone tensor
      stiffness preconditioner stalled even though the nullspace handling was
      correct. That case now looks like a low-bulk edge case rather than a
      general algebra failure: at that resolution the radial bulk had only one
      basis left.
- Dense positive-subspace diagnostics show the current k=`1` standalone
      preconditioner can become a bad CG preconditioner on that tiny-bulk case:
      - `P K` is far from the identity,
      - `P K` has large symmetry defect,
      - `P K` has negative real eigenvalues,
      - the symmetrized `P` has a negative mode on the positive subspace.
- Enabling `k1_inner_schur` materially improves k=`1` numerically, but it is
      too expensive for production and still does not make the standalone
      operator fully SPD. Treat it as a diagnostic endpoint, not the fix.
- New dense bulk-only audits show the k=`1` bulk inverse is not the main
      source of the indefiniteness. On the bulk positive subspace, both the
      default diagonal bulk path and the inner-Schur bulk path remain SPD
      enough: the minimum real eigenvalue stays positive and the symmetrized
      bulk preconditioner stays positive.
- New exact-vs-approx outer-Schur audits isolate the remaining failure to the
      k=`1` surgery Schur *operator*, not to the matrix inversion step:
      - the raw surgery block `A_ss` is SPD,
      - the exact Schur complement `S_exact = A_ss - A_sb A_bb^+ A_bs` is PSD
        up to roundoff, with the expected near-null modes,
      - the approximate Schur assembled with the production bulk inverse is
        strongly indefinite,
      - the stored `schur_inv` matches the inverse/pseudoinverse of that bad
        approximate Schur to roundoff, so the inversion is faithful.
- The k=`1` surgery block is the combined `theta_surgery ⊕ zeta_surgery`
      block. On the current test this expands to `8` theta-surgery dofs and
      `12` zeta-surgery dofs. The dominant Schur defect is in the
      theta-surgery block: its relative error is much larger than the zeta
      block error, while the theta-zeta coupling error is secondary.
- Re-running the same notebook workflow at `ns = (6, 8, 4)` changes the
      conclusion materially:
      - `k=1`, DBC now converges in about `29` iterations with relative
        residual about `5.4e-9` and relative solution error about `2.5e-8`,
      - `k=2`, free still converges cleanly (about `25` iterations, residual
        about `4.9e-9`, solution error about `7.7e-9`),
      - the CP fits on the mapped diagonal channels are not exact, but are
        good enough for the standalone solves at this resolution,
      - so the earlier `k=1` failure at `ns = (4, 8, 4)` should be treated as
        a low-resolution / tiny-bulk Schur edge case, not as a blocking bug in
        the general `k=1` standalone stiffness path.

Current reading: the standalone `k=0,1,2` stiffness ingredients are now in a
good enough state to move on. The small `k=1` failure at `ns = (4, 8, 4)` was
useful diagnostically, but the clean `ns = (6, 8, 4)` solves suggest it is an
edge case of the tiny-bulk outer Schur approximation rather than the main
blocker for this task.

The active higher-form strategy is no longer to keep tuning standalone
`k=1` / `k=2` stiffness blocks or Schur-outers as the end state. HX/AMS
auxiliary-space experiments were explored and are now archived (see
`docs/hiptmair_xu_preconditioner.md`).

In that light, the recent saddle-point benchmark split should be read as a
transition diagnostic, not as the final higher-form design:

- The mixed solves are benchmarked on the actual MINRES block system rather
      than on standalone `curl-curl` / `div-div` blocks.
- The current **reference baseline** is deliberately allowed to cheat a
      little:
      - lower mass block = Jacobi,
      - Schur outer = `exact_jacobi`, meaning its diagonal is probed from the
        true Schur using actual lower `M^{-1}` applies during setup,
      - use this only as a reference line, not as the production route.
- The current **production-oriented comparison** is:
      - lower mass block = tensor,
      - Schur inner = tensor mass apply,
      - Schur outer = Chebyshev wrapped around the approximate Schur
        `S + D M_precond D^T`,
      - no inner exact solves inside the outer Krylov loop,
      - but this should now be treated as an interim comparison point rather
        than the planned final preconditioner for `k = 1, 2, 3`.
- For now the safe benchmark cases remain `k=1` with Dirichlet and `k=2`
      without Dirichlet, because the nullspace / harmonic path outside those
      cases is not cleaned up yet.
- The standalone higher-form stiffness scripts stay useful as local
      diagnostics, but they are no longer the main production target for this
      task.

Recent benchmark state:

- The larger mass benchmark at `ns = (12, 16, 8)` with all `dbc = False`
      confirms the main lower-block decision: even rank-1 tensor mass is
      already clearly better than Jacobi and Jacobi-Chebyshev across
      `k = 0, 1, 2, 3`.
- The standalone scalar `k=0` stiffness benchmark also looks healthy at both
      moderate and larger resolutions, so the scalar path is not the current
      blocker.
- The first full saddle-point smoke on `k=1`, DBC shows the intended split
      between the two comparison strategies:
      - baseline exact-Jacobi is cheap per iteration but can stall,
      - tensor + Chebyshev is more expensive per iteration but converges
        cleanly on the same case.

### Next session pickup

1. Treat HX/AMS content as archived diagnostics unless explicitly re-opened.
2. Keep production higher-form solves on the saddle MINRES + Jacobi Schur
   baseline while other cleanup items are completed.
3. Continue nullspace and regression cleanup independent of HX.
4. **Run focused regression tests after the recent mass/stiffness/saddle changes.**
   ```
   pytest test/test_operators.py -k "tensor or mass_preconditioner or k2_schur" -x --tb=short
   ```
5. **Strengthen `k = 0` stiffness validation.** Add a small benchmark or Ritz /
      condition-number diagnostic so the scalar tensor route is tested on more
      than the current smoke cases.
6. **Optionally keep one tiny-bulk note in mind.** If low resolutions such as
      `ns = (4, 8, 4)` must be production-supported, revisit the `k=1` outer
      Schur approximation there. Otherwise treat it as a diagnostic edge case.
7. **Keep validating the new eager mass default.** The code now defaults all
      four mass blocks to rank `2`; keep one or two larger regression
      benchmarks around so that choice stays justified.
8. **Measure why scalar k=0 rank changes do not move CG counts much.** The
      assembled bulk denominator changes with rank on the rotating ellipse, but
      the iteration counts stayed flat. A small Ritz-value or condition-number
      diagnostic on the preconditioned operator would explain whether the extra
      terms are simply too spectrally weak after diagonal truncation.
9. **Use the current saddle benchmarks as diagnostics only.** Keep
      exact-Jacobi and tensor+Chebyshev mixed runs as reference lines; treat HX
      experiments as archived.
10. Then proceed with the dead-code list below.

## Dead Code To Remove

- [ ] **`mrx/relaxation_deprecated.py`** — entire file is commented out
      (verified with `grep -nE '^[^#]'` → no matches). Only reference is
      `test/deprecated/integration_tests/test_z_pinch.py`, itself in the
      deprecated tree.
- [ ] **`mrx/utils.run_relaxation_loop`** — imports `DescentMethod`,
      `MRXHessian`, `TimeStepper` from `relaxation_deprecated`, would
      `ImportError` if called. Zero callers. Live `mrx/relaxation.py`
      already supplies the modern equivalents.
- [ ] **Remaining legacy `k = 0` fast-diagonalization Hodge stack** in
      [`mrx/operators.py`](../../mrx/operators.py):
  - [x] old prior-based/operator-aware bulk builders removed
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

- [x] **Docs/default mismatch on mass rank.** The intended production policy is
      now explicit: mass defaults to rank `2` for all four degrees, while
      stiffness stays at rank `1`.
- [x] **Krylov-in-Krylov in `mrx/relaxation.py:apply_diffusion`.** No longer exists as a method; `apply_inverse_mass_plus_eps_laplace_matrix` on `DeRhamSequence` is the correct replacement.
      builds `apply_A(x) = M_2 x + η · seq.apply_laplacian(x, 2, ...)`
      and wraps it in an outer `solve_singular_cg`, while
      `apply_laplacian` for `k ≥ 1` runs an inner CG to full
      tolerance. Replace with `apply_inverse_mass_plus_eps_laplace_matrix`
      or use `apply_laplacian_approx`.
- [ ] **`fit_strategy` is still user-visible but inactive in the assembled
      mass block path.** `build_mass_tensor_preconditioner` still stores and
      threads it through, but `_build_diagonal_tensor_block_factors` now
      explicitly ignores it. Either remove it from the public options or make
      it meaningful again.
- [ ] **Confusing API:** `build_mass_tensor_preconditioner(full_matrix,
      ...)` immediately does `del full_matrix`. Drop the parameter and
      update `assemble_tensor_mass_preconditioner` callers.
- [ ] **Higher-form strategy note cleanup.** The current saddle code default is
      still `schur.outer = jacobi`; HX/AMS is now archived and should not be
      described as the active next step in production docs.
- [ ] **Possible future k=0 bulk Chebyshev revisit.** A bulk-local Chebyshev
      correction using the true bulk operator and the rank-1 tensor inverse as
      smoother is still conceptually clean, but earlier experiments of this
      kind did not beat just letting the outer CG handle the residual. Revisit
      only if a harder geometry shows rank-1 tensor is no longer enough.
- [ ] **Eager assembly of legacy FD Hodge.** `assemble_all_operators`
      still calls `assemble_tensor_hodge_preconditioner` even though the
      production scalar solve reads `k0_tensor_hodge_precond` instead.
      Real GPU memory is still being spent on the old payload.
- [ ] **`apply_laplacian` docstring openly says "inner `M_{k-1}^{-1}`
      solves use CG ... to full solver tolerance".** Either rename it
      (`apply_laplacian_exact`) and mark not-for-Krylov, or
      replace its dangerous call sites
      (`apply_mass_plus_eps_laplace_matrix`,
      `relaxation.apply_diffusion`) with `apply_laplacian_approx`.
      The `mrx/nullspace.py` uses are residual-norm probes and fine.
- [ ] **Rank knob duplication.** `MassPreconditionerSpec` still carries a
      top-level `rank` plus per-degree `k{0,1,2,3}_rank`, resolved by
      `_tensor_mass_rank`. That machinery matters again now that multirank is
      live, but it is still easy to misread. A single per-degree rank map would
      be clearer.
