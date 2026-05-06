# Benchmark Artifact Diagnostics Report

## Summary

The interactive benchmark scripts now persist both configured hyperparameters and inferred runtime diagnostics in their artifact JSON output.

Affected scripts:

- `scripts/interactive/mass_preconditioner_demo.py`
- `scripts/interactive/laplacian_preconditioner_demo.py`

Supporting runtime changes:

- `mrx/preconditioners.py`
- `mrx/operators.py`

## What Is Stored

Top-level run metadata still records the full configured experiment state, including geometry, resolution, spline order, tensor settings, and benchmark-selection knobs.

In addition, each benchmark row written via `asdict(report)` now includes an optional `diagnostics` object.

### Mass Demo Diagnostics

For mass benchmarks, `diagnostics` may contain:

- `tuning`
  - `method`: `richardson` or `chebyshev`
  - `source`: either `estimated_runtime` or `dense_exact_jacobi`
  - `smoother`: the smoother actually used for the estimate, such as `tensor` or `jacobi`
  - `max_eig`: estimated or exact upper spectral bound
  - `omega`: Richardson damping parameter when applicable
  - `min_eig`: Chebyshev lower spectral bound when applicable
- `tensor_fit`
  - fit-quality summaries extracted from the tensor block factors
  - only the fit residual-style values are stored:
    - `relative_error`
    - `final_delta`

For `EXACT_HYPERPARAMS=True`, plain Jacobi-smoothed polynomial mass runs store the dense-exact tuning values actually used by the benchmark path.

### Laplace Demo Diagnostics

For scalar, saddle, and diffusion benchmarks, `diagnostics` follows the same basic pattern.

Possible fields are:

- `tuning`
  - inferred Richardson or Chebyshev parameters for scalar and diffusion preconditioners
- `mass`
  - lower-block mass diagnostics for saddle-point preconditioners
- `schur_outer`
  - inferred outer Schur Richardson or Chebyshev parameters when the outer Schur preconditioner is iterative
  - a marker-only record when the outer Schur preconditioner is `exact_jacobi`
- `tensor_fit`
  - tensor fit-quality values for any tensor-backed lower block involved in the run

For scalar `k=0` Hodge/Laplacian cases, the inferred tuning is computed against the shifted or unshifted scalar Hodge operator, using the same estimator routines as the operator implementation.

## Tensor Fit Policy

The tensor diagnostics intentionally do **not** store CP rank or iteration count.

Only fit-quality values are persisted:

- `cp_relative_error`
- `cp_final_delta`

These are attached to `TensorDiagonalBlockInverseFactors` in `mrx/preconditioners.py` and then serialized through the benchmark reports.

## Implementation Notes

- The demos reuse the same internal spectral estimators as the runtime preconditioner builders:
  - `_estimate_preconditioned_max_eigenvalue_apply`
  - `_estimate_chebyshev_lanczos_bounds_apply`
- The Laplace demo diagnostics use `_build_schur_operator_apply` so the saved Schur tuning matches the actual saddle-point preconditioner path.
- The tensor-Hodge CP-ALS callsite in `mrx/operators.py` was updated to match the expanded `_cp_als_3tensor(...)` return signature.