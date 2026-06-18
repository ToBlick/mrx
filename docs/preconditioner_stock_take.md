# Preconditioner Stock-Take

A compact map of the preconditioning landscape in MRX as it actually exists in
the code today, plus a concrete plan for the comprehensive ablation study
requested.

This note is intentionally a snapshot: it points at the canonical
implementation files, marks docs that are redundant or stale, and lists the
hyperparameters each route exposes.

---

## 1. What Code Actually Exposes

### 1.1 User-facing spec objects

All in [mrx/preconditioners.py](../mrx/preconditioners.py):

- `MassPreconditionerSpec(kind, ...)` — single-block (mass-like) spec; also
  reused for the scalar `k=0` Hodge slot and for Schur-inner / Schur-outer
  slots.
- `SchurPreconditionerSpec(inner, outer)` — pair of `MassPreconditionerSpec`s
  used inside the saddle path.
- `SaddlePointPreconditionerSpec(mass, schur, coupled)` — the full
  saddle-point spec.

### 1.2 Admitted `kind` values per slot

Verified against the validators in `mrx/operators.py`:

| Slot | Valid `kind` values |
|---|---|
| Mass / single-block (`_build_operator_preconditioner_apply`) | `none`, `jacobi`, `richardson`, `chebyshev`, `tensor` |
| `schur.inner` (saddle) | `tensor` only (must be terminal: no `surgery_schur`, no `smoother`) |
| `schur.outer` (saddle) | `none`, `jacobi`, `richardson`, `chebyshev`, `exact_jacobi` |
| Scalar `k=0` Hodge | `MassPreconditionerSpec` (same kinds as mass) |

Notes:

- `tensor` is **rejected** as `schur.outer` because the sandwich
  `D M_tensor^{-1} D^T` is no longer a tensor mass block.
- Polynomial mass preconditioners (`richardson`, `chebyshev`) currently
  require `tensor` as their inner smoother (validated by
  `_validate_inner_tensor_only_spec`). Building Chebyshev/Jacobi-on-Jacobi
  for benchmarking is done *outside* the public API by hand-wiring
  `_estimate_chebyshev_lanczos_bounds_apply` and
  `_build_chebyshev_apply_preconditioner` — that is exactly what
  [scripts/benchmark_mass_preconditioners.py](../scripts/benchmark_mass_preconditioners.py)
  already does.
- `exact_jacobi` is a benchmark-only outer that probes the diagonal of the
  true Schur using exact lower mass solves.

### 1.3 Solver wiring

| Problem | Solver | Builder |
|---|---|---|
| `M_k u = f` | `solve_singular_cg` (CG) | `_build_mass_preconditioner_apply` |
| `L_0 u = f` | `solve_singular_cg` | `_build_scalar_hodge_preconditioner_apply` |
| `(L_0 + ε M_0) u = f` | `solve_singular_cg` | `_build_diffusion_preconditioner_apply` |
| `L_k u = f`, `k ∈ {1,2,3}` | `solve_saddle_point_minres` | `_coerce_saddle_preconditioner_spec` + `_build_schur_apply_from_saddle_preconditioner` |
| `(L_k + ε M_k) u = f`, `k ≥ 1` | `solve_saddle_point_minres` | same |

---

## 2. Production Default Policy

| Solve | Default |
|---|---|
| All four `M_k` | `tensor`, eager rank `3` per degree (`k0_rank=k1_rank=k2_rank=k3_rank=3` in `TensorMassPreconditioner`) |
| `L_0` (scalar) | `tensor` (FD-based scalar Hodge bulk + nullspace deflation) |
| `L_k`, `k≥1` (saddle) | `mass=tensor`, `schur.inner=tensor`, `schur.outer=jacobi`, `coupled=False` |
| Stiffness / `k=0` Hodge fallback rank | `1` |
| `k=1` / `k=2` mass `inner_schur` toggle | `False` (off) for wall-clock |

HX/AMS auxiliary-space work is now treated as archived diagnostics
([docs/hiptmair_xu_preconditioner.md](hiptmair_xu_preconditioner.md)).
For production reliability, keep Schur-outer Jacobi as the baseline.

---

## 3. Hyperparameters Per Route

### 3.1 `kind='jacobi'`

None. Pulls the assembled diagonal inverse.

### 3.2 `kind='tensor'` (mass / scalar `k=0` Hodge)

Top-level on `TensorMassPreconditioner`:

- `rank` and per-degree `k{0,1,2,3}_rank` (per-degree wins; resolved by
  `tensor_mass_rank_for_degree`)
- `cp_maxiter`, `cp_tol`, `cp_ridge` — greedy CP fit knobs
- `fit_strategy` — accepted but **inactive** in the assembled bulk path
  (cleanup target).
- `block_chebyshev_steps`, `block_lanczos_iterations`,
  `block_lanczos_max_eig_inflation`, `block_lanczos_min_eig_deflation`,
  `block_lanczos_min_eig_floor_fraction` — tuning for the optional inner
  tensor-block Chebyshev correction
- `richardson_steps`, `richardson_omega` — optional Richardson correction on
  top of the tensor inverse (default `richardson_steps=0`, i.e. plain tensor
  apply)
- `surgery_schur_pinv_tol`

`assemble_tensor_mass_preconditioner` `cp_kwargs`:

- `k1_inner_schur` (default `True` in code, `False` in benchmarks)
- `k2_inner_schur` (default `True` in code, `False` in benchmarks)

`MassPreconditionerSpec` (acting as a tensor mass slot):

- `kind='tensor'`, `surgery_schur=True/False` (whether to wrap the surgery
  Schur around any non-tensor inner kind)

### 3.3 `kind='chebyshev'` / `kind='richardson'`

On `MassPreconditionerSpec`:

- `steps` — polynomial degree
- `power_iterations` — for Lanczos bound estimate
- `min_eig_fraction` — floor for `λ_min`
- `damping_safety` (richardson)
- `lanczos_iterations`, `lanczos_max_eig_inflation`,
  `lanczos_min_eig_deflation`, `lanczos_min_eig_floor_fraction`
- `smoother` — required terminal `MassPreconditionerSpec` (enforced
  `kind='tensor'` by `_validate_inner_tensor_only_spec` in the public path)

### 3.4 `kind='exact_jacobi'` (Schur outer only)

Probes the diagonal of the assembled Schur using exact lower-mass solves at
setup. No runtime knobs beyond a warning context.

### 3.5 Saddle path

`SaddlePointPreconditionerSpec`:

- `mass: MassPreconditionerSpec` (any of jacobi/tensor/poly with tensor
  inner smoother)
- `schur.inner: MassPreconditionerSpec` (tensor only at present)
- `schur.outer: MassPreconditionerSpec` (none/jacobi/richardson/chebyshev/exact_jacobi)
- `coupled: bool` (currently unused in the solve path; coupled completion is
  not the active route)

---

## 4. Existing Benchmark Scripts (and what they cover)

In [scripts/](../scripts):

- [benchmark_mass_preconditioners.py](../scripts/benchmark_mass_preconditioners.py)
  — production flagship. Compares `tensor` (any rank list) vs `jacobi` vs
  `chebyshev(jacobi-inner)` for `M_k`. Sweeps over `--ranks`, has
  `--rotating-eps`, `--rotating-kappa`, `--rotating-r0`, `--rotating-nfp`,
  `--ns`, `--p`, `--cheb-steps`, `--free`, `--no-inner-schur`, Besov RHS
  smoothness `s` baked into `BESOV_RHS_KWARGS` (currently fixed `s=1.0`).
- [benchmark_saddle_hodge_preconditioners.py](../scripts/benchmark_saddle_hodge_preconditioners.py)
  — saddle MINRES `L_k`. Two strategies hard-coded: `baseline` =
  `(jacobi, tensor, exact_jacobi)`; `tensor_chebyshev` =
  `(tensor, tensor, chebyshev)`. Only `k1_dbc` and `k2_free` (harmonic-free
  cases). Sweeps `--ranks`, `--cheb-steps`, geometry knobs as above.
- [benchmark_stiffness_preconditioners.py](../scripts/benchmark_stiffness_preconditioners.py)
  — standalone stiffness blocks (`k=0,1,2`) with `solve_singular_cg` after
  nullspace deflation. Tensor stiffness preconditioner
  (`apply_stiffness_tensor_preconditioner`) vs Jacobi vs Chebyshev. **Note:**
  this exercises the standalone tensor stiffness block, which is *not* the
  production path for `k=1,2,3` (those go through saddle MINRES). Useful as
  a diagnostic only.
- [benchmark_preconditioners.py](../scripts/benchmark_preconditioners.py) —
  older Laplacian smoke benchmark on a torus. Largely superseded by
  the three above. Candidate for retirement.
- [benchmark_mass_k0_rank_sweep.py](../scripts/benchmark_mass_k0_rank_sweep.py),
  [benchmark_richardson_vs_modal.py](../scripts/benchmark_richardson_vs_modal.py),
  [benchmark_schur_bulk_endpoints.py](../scripts/benchmark_schur_bulk_endpoints.py),
  [benchmark_extraction_matvecs.py](../scripts/benchmark_extraction_matvecs.py)
  — narrow studies; keep as-is, not the right hooks for a comprehensive
  ablation.

Interactive demos:

- [scripts/interactive/mass_preconditioner_demo.py](../scripts/interactive/mass_preconditioner_demo.py)
  — broad mass + scalar `k=0` Hodge demo with plots.
- [scripts/interactive/laplacian_preconditioner_demo.py](../scripts/interactive/laplacian_preconditioner_demo.py)
  — saddle Hodge demo over the supported saddle slices.

Shared RHS generator: `mrx.utils.build_random_besov_rhs_batch` with knobs
`s` (smoothness exponent), `upper_limit`, `num_modes`, `scale`,
`smoothness_margin`, `normalization_samples`. **`s` is the right knob for
the "rhs smoothness" axis** of the requested study.

---

## 5. Documentation Audit

| File | Keep / merge / drop |
|---|---|
| [docs/preconditioner_primer.md](preconditioner_primer.md) | **Keep** as the canonical production overview. |
| [docs/mass_preconditioners.md](mass_preconditioners.md) | **Compact**. §1–§4 are still current; §5 (analytic-priors / Richardson plan), §6 (eps sweep), §7 (final) are speculation/historical and can move under `docs/dev/`. |
| [docs/preconditioner_cleanup_todo.md](preconditioner_cleanup_todo.md) | **Compact**. Most of the long status preamble duplicates `preconditioner_primer.md`; only the dead-code list and the small "Oddities" list are unique. |
| [docs/hiptmair_xu_preconditioner.md](hiptmair_xu_preconditioner.md) | Keep — consolidated HX/AMS postmortem and archive. |
| [docs/dev/iterative_solver_primer.md](dev/iterative_solver_primer.md) | **Keep** but de-duplicate against `preconditioner_primer.md` §1. Sections 4–6 already overlap heavily. |
| [docs/dev/tensor_preconditioner_primer.md](dev/tensor_preconditioner_primer.md) | Keep — the only narrow note on the tensor mechanism. |
| [docs/dev/laplacian_preconditioner_notes.md](dev/laplacian_preconditioner_notes.md) | Keep, narrow scope. |
| [docs/dev/operator_aware_scalar_stiffness_fit.md](dev/operator_aware_scalar_stiffness_fit.md) | **Stale** — describes the prior-based scalar `k=0` builder that has since been removed and replaced by the FD-based path. Move to a `docs/dev/archive/` or drop. |
| [docs/dev/higher_form_hodge_tensor_plan.md](dev/higher_form_hodge_tensor_plan.md) | Historical diagnostic note; not active production direction. |
| [docs/dev/surgery_schur_refactor_plan.md](dev/surgery_schur_refactor_plan.md) | Likely landed; review and drop if so. |
| [docs/dev/tensor_debug_findings.md](dev/tensor_debug_findings.md) | Keep as a debug log. |
| [docs/dev/benchmark_artifact_diagnostics_report.md](dev/benchmark_artifact_diagnostics_report.md) | Keep, narrow. |
| [docs/dev/testing_strategy.md](dev/testing_strategy.md) | Keep. |

The biggest concrete redundancy: the saddle/`k≥1` and mass-rank narrative is
told three times (`preconditioner_primer.md`, `mass_preconditioners.md`,
`preconditioner_cleanup_todo.md`). Recommend collapsing to:

- **`docs/preconditioner_primer.md`** — production policy and admitted kinds
  (one source of truth).
- **`docs/preconditioner_hyperparameters.md`** (new, or section §3 of this
  note) — flat table of every knob each kind exposes.
- **`docs/preconditioner_cleanup_todo.md`** — keep only the dead-code and
  oddity bullets; drop the long "Status (resume here)" preamble.
- Move historical narrative (rank sweeps, eps sweeps, prior experiments)
  into `docs/dev/history/`.

---

## 6. Code Consolidation Targets

From the cleanup TODO plus what was visible during this stock-take:

1. **Duplicate helpers** between `mrx/operators.py` and `mrx/preconditioners.py`:
   - `_estimate_preconditioned_max_eigenvalue_apply`
   - `_estimate_chebyshev_lanczos_bounds_apply`
   - `_build_chebyshev_apply_preconditioner`

   Pick the `preconditioners.py` versions; have `operators.py` import.

2. **Dead module:** `mrx/relaxation_deprecated.py` (entirely commented out;
   only referenced by `mrx/utils.run_relaxation_loop`, which is itself dead).

3. **Legacy `k=0` FD Hodge stack** in `mrx/operators.py`:
   `assemble_tensor_hodge_preconditioner`, `_fd_apply_3d`, `_fd_apply_full`,
   `apply_hodge_kron_preconditioner`, `_fd_hodge_scales_K`, and the
   `SequenceOperators` fields `fd_V_p_{r,t,z}`, `fd_lam_p_{r,t,z}`,
   `dd0_fd_scale_K`. Production `k=0` Hodge already uses
   `k0_tensor_hodge_precond`.

4. **Inactive knob:** `MassPreconditionerSpec.fit_strategy` /
   `TensorMassPreconditioner.fit_strategy` — accepted but ignored by the
   assembled bulk path. Either remove or re-enable.

5. **Confusing API:** `build_mass_tensor_preconditioner(full_matrix, ...)`
   immediately discards `full_matrix`. Drop the parameter.

6. **Rank knob duplication:** `MassPreconditionerSpec` carries top-level
   `rank` plus per-degree `k{0..3}_rank`. Replace with a single
   per-degree map.

7. **Krylov-in-Krylov hazard:** `relaxation.apply_diffusion` calls
   `apply_laplacian` (which itself runs an inner CG). Switch to
   `apply_laplacian_approx` / `apply_inverse_mass_plus_eps_laplace_matrix`.

8. **Unused public:** `apply_mass_rtzblock_preconditioner` in
   `mrx/preconditioners.py`.

---

## 7. Plan For The Requested Ablation Study

Goal axes from the user request:

- **Baselines**: whole-matrix Jacobi, whole-matrix Chebyshev (Jacobi-inner).
- **Tensor + Chebyshev correction** on bulk blocks (i.e., turn on the inner
  tensor-block Chebyshev / Richardson correction that already exists via
  `block_chebyshev_steps` and `richardson_steps`).
- **Sweeps**: resolution `ns`, spline order `p`, RHS smoothness `s` (Besov
  exponent), aspect ratio `rotating-eps`.

### 7.1 Existing infrastructure that already covers most of it

- `benchmark_mass_preconditioners.py` already supports `tensor` vs `jacobi`
  vs `chebyshev(jacobi)` for mass solves with all four geometry knobs.
  **Missing**: a flag to expose `--rhs-s` for `BESOV_RHS_KWARGS["s"]`, and
  a flag to enable per-degree tensor-block Chebyshev correction
  (`block_chebyshev_steps > 0`).
- `benchmark_saddle_hodge_preconditioners.py` already supports
  `baseline`/`tensor_chebyshev` for `k1_dbc, k2_free`. **Missing**: a third
  strategy `whole_jacobi` (and `whole_chebyshev`) that uses Jacobi/Chebyshev
  on the lower mass block *and* a Jacobi/Chebyshev whole-matrix outer — i.e.
  the "no-tensor anywhere" baseline implied by the request. Also missing:
  `k3_dbc`/`k3_free` cases, currently restricted by harmonic-handling
  caveats.
- `benchmark_stiffness_preconditioners.py` is orthogonal to this study (it
  benchmarks standalone stiffness blocks, which are not the production
  saddle path).

### 7.2 Recommended consolidation: one driver + a YAML matrix

A single `scripts/benchmark_preconditioner_sweep.py` driver that:

1. accepts a sweep config (resolutions × `p` × `eps` × `s` × strategy list),
2. dispatches to the existing per-`k` benchmark functions (so we don't
   rewrite the solve loops),
3. emits one tidy CSV/JSON per row with the columns:
   `(problem, k, dirichlet, ns, p, rotating_eps, rhs_s, strategy, rank,
   cheb_steps, avg_iters, max_iters, avg_ms, failures, residual)`.

Strategies to support (named, fixed, well-defined):

| Name | Mass slot | `schur.inner` | `schur.outer` | Notes |
|---|---|---|---|---|
| `whole_jacobi` | `jacobi` | `tensor` | `jacobi` | current `schur.outer=jacobi` default |
| `whole_chebyshev` | `jacobi` | `tensor` | `chebyshev` | with Jacobi inner smoother on outer block |
| `tensor_baseline` | `tensor` | `tensor` | `jacobi` | current production saddle default |
| `tensor_outer_chebyshev` | `tensor` | `tensor` | `chebyshev` | already exists as `tensor_chebyshev` |
| `tensor_bulk_chebyshev` | `tensor` (with `block_chebyshev_steps>0`) | `tensor` | `chebyshev` | the new "Chebyshev on top of tensor for bulk" the user asked for |
| `reference_exact_jacobi` | `jacobi` | `tensor` | `exact_jacobi` | reference upper-bound (probes truth) |

For mass-only solves, drop the `schur` columns; the same `mass` strategy
column suffices.

### 7.3 Concrete action items (smallest viable first)

1. Add `--rhs-s` to `benchmark_mass_preconditioners.py` and
   `benchmark_saddle_hodge_preconditioners.py` (one-line plumbing into
   `BESOV_RHS_KWARGS`).
2. Add a `--block-cheb-steps` flag to `benchmark_mass_preconditioners.py`
   that threads into `TensorMassPreconditioner.block_chebyshev_steps`.
3. Add `whole_jacobi` and `whole_chebyshev` strategies to
   `benchmark_saddle_hodge_preconditioners.py` (small change: build the
   lower preconditioner from `MassPreconditionerSpec(kind='jacobi')` and
   the outer from a hand-wired Chebyshev-on-Jacobi like
   `benchmark_mass_preconditioners.py` already does).
4. Write a thin sweep runner that calls those scripts with a matrix of
   parameter combinations and aggregates output.
5. Once the sweep is tidy, retire `benchmark_preconditioners.py` (the older
   torus Hodge benchmark) which is now strictly subsumed.

These four changes give a comprehensive ablation matrix on top of the
existing scripts without forking the solve paths.

---

## 8. TL;DR

- Three preconditioner families exist in code: **`jacobi`, `tensor`, polynomial
  (`richardson`/`chebyshev`)**, plus the saddle-only **`exact_jacobi`** outer.
- `tensor` is the production default for all four mass blocks (rank 3) and
  for scalar `k=0` Hodge (rank 1).
- For `k≥1` the production is saddle MINRES with
  `(mass=tensor, schur.inner=tensor, schur.outer=jacobi)`. The strategic
  production reliability baseline remains Schur-outer Jacobi.
- Hyperparameters per kind are listed in §3.
- The requested ablation needs **only ~3 small flag additions** to the
  existing benchmark scripts plus a thin sweep driver — no new solve
  machinery.
- Three docs duplicate the same status narrative
  (`preconditioner_primer.md` / `mass_preconditioners.md` /
  `preconditioner_cleanup_todo.md`). Recommended collapse in §5.
- Concrete dead-code targets are in §6 (most already tracked in the cleanup
  TODO).

---

## 9. Convergence benchmark (2026-06-05, `sparse` branch)

Toroidal domain, `epsilon=1/3`, `ns=(n, 2n, n)`, `cg_tol=1e-12`.
Error = relative physical L² norm.

### k=0 DBC (tensor Laplacian preconditioner)

| p | n=8 | n=12 | n=16 | rate |
|---|-----|------|------|------|
| 1 | 4.58e-02 (11) | 1.83e-02 (12) | 9.77e-03 (13) | 2.2 |
| 2 | 4.12e-03 (10) | 1.04e-03 (12) | 4.12e-04 (12) | 3.3 |
| 3 | 4.77e-04 (11) | 7.88e-05 (12) | 2.35e-05 (12) | 4.3 |

Iteration counts ~10–13, n- and p-independent. Rates = p+1. ✓

### k=1 DBC, k=3 NBC (Schur Jacobi preconditioner)

| p | k | n=8 | n=12 | rate |
|---|---|-----|------|------|
| 1 | 1 | 2.64e-01 (155) | 1.69e-01 (295) | 1.1 |
| 2 | 1 | 2.19e-02 (110) | 7.84e-03 (201) | 2.5 |
| 1 | 3 | 3.47e-01 (106) | 2.12e-01 (198) | 1.2 |
| 2 | 3 | 4.83e-02 (66)  | 1.80e-02 (112) | 2.4 |

Rates = p+1. ✓  Iteration counts ~10–20× higher than k=0 — Schur Jacobi is
weak.  HX/AMS auxiliary-space attempts are archived; Jacobi remains the
practical baseline.
