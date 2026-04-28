# Mass Preconditioners

This note summarizes the current production picture for the mass
preconditioners across all form degrees `k = 0, 1, 2, 3`.

The shared design principle is simple:

- keep the correct extracted-space block structure for the degree,
- approximate the geometric coefficient fields rather than the assembled
  inverse,
- build tensor-product or low-rank sums of tensor-product block models from
  those coefficient fields,
- and use dense Schur solves only where the extracted structure forces a small
  surgery or core block.

The degree-specific debug scripts remain the regression harnesses. This note is
meant to explain the ideas behind the production preconditioners, not to keep a
full research diary.

## 1. Common Geometric Picture

On the mapped domain, the weak mass matrices use the standard pull-back
weights

$$
W^0 = J,
\qquad
W^1 = J g^{-1},
\qquad
W^2 = g / J,
\qquad
W^3 = 1 / J.
$$

The production tensor route keeps only the diagonal coefficient fields that
belong to the diagonal tensor blocks:

- `k = 0`: `J`,
- `k = 1`: `J g^{rr}`, `J g^{theta theta}`, `J g^{zeta zeta}`,
- `k = 2`: `g_rr / J`, `g_theta theta / J`, `g_zeta zeta / J`,
- `k = 3`: `1 / J`.

These fields are sampled on the tensor quadrature grid and fitted by low-rank
CP decompositions. Each rank-1 term assembles one weighted tensor-product mass
contribution, so a rank-`R` fit becomes a short sum of Kronecker-like blocks.

For `R = 1`, the inverse apply is just a tensor-product inverse. For `R > 1`,
the current structured inverse uses the shared-modal fast-diagonalisation path
rather than storing dense 3-D inverse blocks.

So the compression target is the geometry field, not the inverse operator.

## 2. Baselines

Two simpler baselines still exist and remain useful as references:

- `jacobi`: inverse diagonal of the extracted mass matrix,
- `kronecker`: legacy reference-product inverse with per-component averaged
  geometric scaling.

The tensor path is now the preferred production route for all four form
degrees `k = 0, 1, 2, 3`. The baselines remain useful for fallback and
benchmarking, but the production rollout itself is complete.

## 3. Degree-Specific Structures

### `k = 0`

`k = 0` is the scalar Schur case.

- The extracted matrix splits into a small core block and one bulk tensor block.
- The production preconditioner uses a dense Schur solve on the core.
- The bulk inverse is a tensor-diagonal inverse built from a CP fit of `J`.

So the structure is:

- small dense core Schur,
- one scalar tensor bulk inverse.

### `k = 1`

`k = 1` is the most complicated mass case.

- The extracted `theta` and `zeta` components contain surgery rows.
- The correct structure is an outer surgery Schur plus an inner Schur on the
  coupled `(r, theta_bulk)` block.
- The diagonal bulk inverse drop-ins are modeled from CP fits of
  `J g^{rr}`, `J g^{theta theta}`, and `J g^{zeta zeta}`.

So the structure is:

- outer surgery Schur,
- inner coupled Schur on `(r, theta_bulk)`,
- tensor-diagonal inverse models for the diagonal bulk blocks.

### `k = 2`

`k = 2` mirrors `k = 1`, but the extracted surgery structure is simpler.

- The extracted `r` component contributes the small surgery block.
- The remaining `r_bulk`, `theta`, and `zeta` blocks stay in the bulk.
- The bulk inverse models come from CP fits of
  `g_rr / J`, `g_theta theta / J`, and `g_zeta zeta / J`.

So the structure is:

- one smaller outer Schur block than in `k = 1`,
- three tensor-diagonal bulk inverse blocks.

### `k = 3`

`k = 3` is the second scalar case.

- There is no extracted-space Schur split.
- The extracted matrix is one scalar tensor block.
- The production inverse apply is the tensor-diagonal inverse built from a CP
  fit of `1 / J`.

So the structure is:

- no surgery,
- no Schur,
- direct scalar tensor inverse.

## 4. Why This Route Won

Older debug paths tried to compress inverse-like objects more directly. The
current route survived because it preserves the right abstraction boundary:

- the extracted-space Schur logic stays exact,
- only the diagonal tensor blocks are approximated,
- the approximation acts on the coefficient field instead of the dense inverse,
- and the stored data scales like 1-D factors and small Schur blocks rather
  than dense per-mode 3-D inverses.

That is the main unifying idea across all four degrees.

## 5. Practical Summary

- `k = 0`: scalar Schur plus tensor bulk inverse.
- `k = 1`: outer surgery Schur plus inner coupled Schur plus tensor bulk
  inverses.
- `k = 2`: simpler outer Schur plus tensor bulk inverses.
- `k = 3`: direct scalar tensor inverse.

So the common theme is not a single matrix pattern. The common theme is a
coefficient-first tensor compression wrapped around the correct block structure
for each degree.

## 6. Polynomial Tuning Policy

The polynomial mass preconditioners now store their parameter-generation
hyperparameters directly on `MassPreconditionerSpec`.

For Richardson, the active tuning fields are:

- `steps`,
- `power_iterations`,
- `damping_safety`.

The implementation still uses the existing power-iteration estimate for the
largest relevant eigenvalue and then sets

$$
\omega \approx \frac{\text{damping\_safety}}{\lambda_{\max}}.
$$

So Richardson remains the cheap setup path and is also the path used inside
nullspace inverse iteration.

For Chebyshev, the active tuning fields are:

- `steps`,
- `lanczos_iterations`,
- `lanczos_max_eig_inflation`,
- `lanczos_min_eig_deflation`,
- `lanczos_min_eig_floor_fraction`.

The current production policy is:

- estimate the upper and lower bounds from a short Lanczos run on the active
  preconditioned operator,
- inflate the upper Ritz value by `lanczos_max_eig_inflation`,
- deflate the smallest positive Ritz value by
  `lanczos_min_eig_deflation`,
- and keep the lower guard above
  `lanczos_min_eig_floor_fraction * lambda_max_used`.

So `min_eig_fraction` remains the legacy heuristic field, but the active
Chebyshev setup path now prefers the stored guarded-Lanczos policy instead.

## 7. Current Production Guidance

- Use `jacobi` as the minimal fallback.
- Use the tensor path as the preferred production mass preconditioner.
- Keep the degree-specific debug scripts as regression harnesses:
  - [scripts/debug_k0_mass_surgery.py](/scratch/tblickhan/mrx/scripts/debug_k0_mass_surgery.py)
  - [scripts/debug_k1_mass_surgery.py](/scratch/tblickhan/mrx/scripts/debug_k1_mass_surgery.py)
  - [scripts/debug_k2_mass_surgery.py](/scratch/tblickhan/mrx/scripts/debug_k2_mass_surgery.py)
  - [scripts/debug_k3_mass_surgery.py](/scratch/tblickhan/mrx/scripts/debug_k3_mass_surgery.py)
- Treat rank `3` as the practical reference point in benchmarks, since it has
  typically captured most of the gain while keeping setup moderate.

## 8. Status

The mass-preconditioner program is structurally finished:

- `k = 0`: scalar core-plus-bulk Schur,
- `k = 1`: outer surgery Schur plus inner coupled Schur,
- `k = 2`: outer surgery Schur with three tensor bulk blocks,
- `k = 3`: direct scalar tensor inverse.

So the remaining questions are no longer about missing algebraic cases. They
are tuning and robustness questions:

- how robust the current CP-ALS/shared-modal route is on geometry families far
  from the present benchmark set,
- whether rank selection should remain fixed by the caller or become more
  automatic again,
- whether future geometry pipelines should expose tensor-structured metric
  fields upstream instead of recovering them from quadrature samples,
- and how much the guarded-Lanczos Chebyshev tuning should eventually be made
  user-facing versus remaining an internal default policy.

Those are optimization questions. The main structural choice is already made,
and the production tensor path should now be treated as available across all
form degrees.