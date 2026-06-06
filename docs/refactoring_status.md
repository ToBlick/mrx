# Source and Test Refactoring Status

Tracks the cleanup sweep of `mrx/` and `test/` on the `sparse` branch.
Last updated: 2026-06-05 (session 5).

---

## Source files (`mrx/`)

| File | Status | Notes |
|---|---|---|
| `operators.py` | ✅ cleaned | `_assemble_unweighted_1d_stiffness` removed; 6 call sites now use `_assemble_weighted_1d_stiffness` directly. `_prod3` replaces 12× `int(jnp.prod(...))`. Projection block `(2,1)`/`(1,2)` merged. Docstrings trimmed to Google style. `g0/g1/g2` type annotations corrected from `jsparse.BCSR` to `_MatrixFreeIncidence`. Stale "BCSR matvecs" wording removed from `apply_derivative_matrix` and `apply_stiffness` docstrings. Seven `# TODO: remove` markers placed on the functions listed below. |
| `geometry.py` | ✅ cleaned | `grad_1d` lifted from `assembly.py`. |
| `assembly.py` | 🔴 delete | Pure BCSR/dense assembly. Superseded entirely by matrix-free mass (`build_matrixfree_mass_apply`) and `_MatrixFreeIncidence`. No callers outside deprecated paths. |
| `local_assembly.py` | 🔴 delete | Element-wise quadrature assembly. Superseded by sum-factorized matrix-free apply in `operators.py`. Fixed `evaluate_basis_local` to handle `type="constant"` (was raising `NotImplementedError`). |
| `relaxation_deprecated.py` | 🔴 delete | Entire file is commented out. Only reference is `test/deprecated/integration_tests/test_z_pinch.py` (itself deprecated). |
| `spline_geometry.py` | 🔴 delete | Module-level docstring says deprecated. Superseded by `geometry.py`. |
| `derham_sequence.py` | ✅ cleaned | Stale imports `evaluate_at_xq_deprecated` / `integrate_against_deprecated` removed; `cross_product_projection_deprecated` commented-out block (~120 lines) deleted. `cross_product_projection` → `cross_product_load`; `pressure_projection` → `pressure_load`. 12 `l*` projector attributes (`l0/l1/l2/l3`, `*_dbc`, `*_bc`) replaced by `load(f, k, dirichlet, bc)` and `interpolate(f, k, dirichlet)` methods that delegate to the pure functions in `projectors.py`. |
| `utils.py` | ✅ cleaned | All utility functions migrated to owning modules (`differential_forms.py`, `quadrature.py`, `solvers.py`, `preconditioners.py`, `relaxation.py`). `utils.py` is now a thin shim of re-exports for backward compatibility. `evaluate_at_xq_deprecated`, `integrate_against_deprecated` still present but have no live callers. `run_relaxation_loop` uses lazy import of `append_to_trace_dict` to break circular import. |
| `preconditioners.py` | ✅ cleaned | `build_mass_jacobi_pair` rewritten to use `diag_matvec` probing (no `mass_sp` BCSR argument). `diag_EAET` / `diag_EAET_direct` imports removed. `_mass_diaginv` in `operators.py` updated to probe via `mass_core_apply` + `diag_matvec`; BCSR fallback (`diag_EAET_direct`) and its import removed. No-op stubs (`set_mass_rtzblock_factor`, `invalidate_mass_rtzblock`) kept as-is. See `preconditioner_cleanup_todo.md` for remaining dead-code targets (`assemble_tensor_hodge_preconditioner`, duplicate Chebyshev/Lanczos helpers, debug-only forward-model code). |
| `relaxation.py` | ✅ no action | `apply_diffusion` no longer exists as a method; `apply_inverse_mass_plus_eps_laplace_matrix` on `DeRhamSequence` is the correct replacement and is already used everywhere. |
| `nullspace.py` | ✅ cleaned | `get_saddle_point_nullspaces`: Python for-loop over rows replaced with `jax.vmap(_lower)`. `get_stiffness_nullspace`: removed redundant `0.5*(A+A.T)` symmetrisation of the already-symmetric gram matrix; flattened the no-op `for v in exact_basis: basis.append(v)` loop to `basis = list(exact_basis)`. |
| `solvers.py` | ✅ cleaned | `picard_solver`: state now carries `fz = f(z)` so `f` is called once per iteration (not twice); removed `lax.cond` post-loop re-evaluation and nested function docstrings. `preconditioned_cg`: removed no-op `z0.copy()`; deduplicated `jnp.dot(p, Ap)` into local `pAp`. `minres`: replaced fragile positional tuple state with `_MinresState(NamedTuple)`; `cond_fn` and final extraction now use named fields. `solve_saddle_point_minres`: replaced for-loop projections and `inner_upper/lower` helpers with vmap-based vectorised projections. |
| `projectors.py` | ✅ cleaned | `Projector` class deleted. Replaced by two pure module-level functions: `load(seq, f, k, dirichlet, bc)` (matrix-free dual load vector assembly via `jax.lax.map` + `integrate_against`) and `interpolate(seq, f, k, dirichlet)` (Greville collocation for k=0; histopolation for k=1,2,3). Collocation/histopolation matrices built lazily; `# TODO: cache` notes added. `BoundaryProjector` and `surface_integral` untouched. Polar paths still raise `NotImplementedError`. |
| `extraction_operators.py` | ✅ cleaned | **`PolarExtractionOperator`**: removed dead element-by-element dispatch methods (`_vector_index`, `_element`, `_inner_zeroform`, `_outer_zeroform`, `inner_oneform_r`, `inner_oneform_θ`, `_threeform`); superseded by `build_extraction()` which fills COO triplets via explicit NumPy loops (one-time O(n) setup). **`BoundaryOperator`**: `assemble_sparse` renamed to `build_extraction`; `sparse_matrix` wrapper removed. `assemble()` / `matrix()` / `__array__()` were already gone. Both `build_extraction` methods produce a `MatrixFreeExtraction` (gather/scatter apply — no matrix stored or multiplied). All 8 call sites in `derham_sequence.py` updated. |
| `io.py` | ✅ cleaned | Removed `# %%` cell marker (line 1) and unused `from mrx.differential_forms import Pushforward` import. |
| `plotting.py` | ✅ cleaned | Fixed 4 bugs: (1) duplicate `jax.lax.map(F, _x)` in `get_2d_grids` — second call overwrote `_y` after `_y1/y2/y3` were already sliced from the first; (2) `integrate_fieldlines` used stale attribute names `seq.Lambda_2/0`, `seq.E2/0`, `seq.F` — updated to `seq.basis_2/0`, `seq.e2/0`, `seq.map`; (3) `trace_plot` called `ax2.legend(...)` twice — first combined call (with `ax2_top` handles) was immediately overwritten by a second call with only local handles; removed the dead second call; (4) stripped three `# %%` notebook cell markers. No tests needed. |
| `config.py` | ✅ no action | Pure Hydra dataclasses + ConfigStore registration; no dead code. |
| `mappings.py` | ✅ no action | Clean; `rotating_ellipse_map` etc. are stable. |
| `quadrature.py` | ✅ no action | Stable, no known issues. |
| `spline_bases.py` | ✅ no action | Stable, well-tested. |
| `differential_forms.py` | ✅ no action | Stable. |

### `operators.py` — pending `# TODO: remove` targets

These functions are marked in source and should be removed once
`assembly.py` and `local_assembly.py` are gone:

| Function / symbol | Line | Reason |
|---|---|---|
| `_assemble_mass_block` | 1816 | BCSR assembly, superseded by matrix-free |
| `update_mass_operator` | 1825 | wrapper around the above |
| `assemble_mass_operators` | 1872 | wrapper around the above |
| `assemble_tensor_hodge_preconditioner` | 2838 | deprecated no-op shim; production uses `update_hodge_operator` |
| `_assemble_derivative_block` | 3178 | no-op validator, wraps `_MatrixFreeIncidence` build |
| `update_derivative_operator` | 3192 | no-op validator path |
| `assemble_all_dense_operators` | 3927 | debug/dense path only |

Additional dead code in `operators.py` (tracked in `preconditioner_cleanup_todo.md`):
`_fd_apply_3d`, `_fd_apply_full`, `apply_hodge_kron_preconditioner`,
`_fd_hodge_scales_K`, and related `SequenceOperators` fields
(`fd_V_p_{r,t,z}`, `fd_lam_p_{r,t,z}`, `dd0_fd_scale_K`).

Duplicate helpers that should be collapsed (keep `preconditioners.py` versions):
`_estimate_preconditioned_max_eigenvalue_apply`,
`_estimate_chebyshev_lanczos_bounds_apply`,
`_build_chebyshev_apply_preconditioner`.

---

## Test files (`test/`)

| File | Status | Notes |
|---|---|---|
| `test_operators.py` | ✅ complete | Mass tests (identity + rotating-ellipse), Hodge Laplacians (k=0–3 × free/DBC), de Rham complex d²=0, strong/weak derivative consistency, incidence topology, nullspace Betti-number checks, Poisson k=0 analytical convergence, Hodge preconditioner SPD + acceleration. All use session-scoped `torus_seq`. |
| `test_solvers.py` | ✅ new / complete | Tests all 5 public solvers: `preconditioned_cg`, `solve_singular_cg`, `minres`, `solve_saddle_point_minres`, `picard_solver`. Module-level SPD system (N=24), Jacobi preconditioner. |
| `test_preconditioners.py` | ✅ refactored | Migrated from module-level `_SEQ`/`_OPS` globals (axisymmetric, rank-1) to session-scoped `torus_seq` + `precond_jit` fixtures (full 3D, rank-3). Tests: symmetry, SPD, CG iteration reduction, round-trip accuracy. All 48 tests share one assembly. |
| `test_nullspace.py` | ✅ new / complete | Tests: `_n_vectors` torus Betti table, `init_nullspaces` shapes/zeros, `get_nullspace` raises when uninitialised, harmonicity `‖Lv‖ ≤ 10·tol`, mass-orthonormality (Gram = I), saddle-point lower block `M_{k-1}w = D_{k-1}^T v`, stiffness-nullspace kernel membership, stiffness-nullspace mass-orthonormality. Uses `torus_seq`; stiffness bases cached on `torus_seq.stiffness_null`. |
| `test_extraction_operators.py` | ✅ no action | Stable. |
| `test_geometry.py` | ✅ no action | Stable. |
| `test_sequence.py` | ✅ cleaned | `Projector` import removed; all tests now use `seq.load(...)` and `seq.interpolate(...)`. Two polar tests marked `xfail`. New `test_cross_product_load_1_1_1_l2_convergence` (p=1,2). **New Poisson integration tests** (k=0–3, all analytical): `test_poisson_k0_matches_analytical` (DBC, scalar), `test_poisson_k3_matches_analytical` (NBC, 3-form recovered via `Pushforward`), `test_poisson_k1_matches_analytical` (DBC, e_zeta 1-form), `test_poisson_k2_matches_analytical` (NBC, e_zeta 2-form). |
| `test_spline_bases.py` | ✅ no action | Stable, well-tested. |
| `test_quadrature.py` | ✅ no action | Stable. |
| `test_projectors.py` | ✅ cleaned | All `proj_seq.l0/l1/l2/l3` calls replaced with `seq.load(f, k)` and `seq.interpolate(f, k)`. Four polar tests `xfail(raises=NotImplementedError, strict=True)`. Physical L2 error metric uses `Pushforward`. |
| `random_fields.py` | ✅ new | `build_random_besov_function`, `build_random_besov_rhs_batch` test helpers. |
| `conftest.py` | ✅ updated | `torus_seq`: session-scoped, full 3D `(8,16,8)` p=3 spline-projected torus with nullspaces + `stiffness_null` attribute. `precond_jit`: session-scoped dict of JIT-compiled + warmed-up preconditioner applies keyed by `(label, k, dbc)`. `n_dofs`, `build_dense` helpers exported. |
| `test/deprecated/` | 🔴 delete (whole tree) | Mirrors the deprecated source files. Safe to remove once `assembly.py`, `local_assembly.py`, and `relaxation_deprecated.py` are gone. |

### Deleted
- `test/test_laplacians.py` — merged into `test_operators.py`.
