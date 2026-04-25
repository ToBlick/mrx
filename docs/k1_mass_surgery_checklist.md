# k=1 Polar Mass Surgery Checklist

This document tracks the experimental debug path for the polar `k=1` mass
matrix. The goal is to understand the extracted-space surgery rows before
designing a `k=1` structured mass preconditioner.

All work stays in debug scripts until the block structure is clear on small
problems.

## Current Status Snapshot

- The surgery structure is now established and the outer surgery Schur plus
  inner coupled `(r, theta_bulk)` Schur split remains the right model.
- The old full-rank `rt|z` family is still the best diagnostic ceiling among
  the currently tested fully structured debug paths, but it is not a good
  production endpoint because it stores dense per-mode inverses.
- The metric-factor route is the right compression target:
  - direct CP-ALS fits on `J g^{rr}`, `J g^{theta theta}`, `J g^{zeta zeta}`
    are accurate at low rank,
  - `A_rr` is already excellent at rank `3`,
  - `A_thetatheta` is acceptable at rank `3`,
  - `A_zetazeta` is excellent at rank `3`.
- A dense-inverse metric-rank-3 debug model gave the best observed benchmark,
  but that inverse path is not structurally acceptable.
- A rank-3 Neumann inverse around the dominant term was not good enough.
- A rank-1 exact metric-Kronecker inverse is viable as a cheap fallback:
  it is clearly better than `jacobi` and production `kronecker`, but still
  worse than the best current `rt|z` diagnostics.

## Remaining TODOs

- [ ] Replace the debug-only dense inverse of the low-rank metric-Kronecker
  sum by a genuinely structured inverse for `R > 1`.
- [ ] Test the shared-modal fast-diagonalisation idea for low-rank metric
  Kronecker sums.
- [ ] Revisit overlap-weighted / separable coefficient models only if the
  low-rank metric route stalls.
- [ ] Keep the benchmark path honest by separating mixed-only applies from
  debug-only dense truth construction when the metric model is promoted.
- [ ] Do not promote any `k=1` production change until the structured
  low-rank metric inverse beats or matches the current best debug `rt|z`
  benchmark without large dense storage.

## Current Facts

- The extracted `r` component is a pure selector/permutation block.
- The extracted `theta` component has a surgery block consisting of its first
  `2 * n_z` rows.
- Those `theta` surgery rows mix raw `r` and raw `theta` components.
- The extracted `zeta` component has a surgery block consisting of its first
  `3 * d_z` rows.
- Those `zeta` surgery rows are scalar-like and only touch the raw `zeta`
  component, analogous to the `k=0` scalar core.
- The `k=1` component bookkeeping bug in `n1_3` / `n1_3_dbc` is fixed.
- After reordering, the true surgery block is still only
  `theta_surgery ∪ zeta_surgery`; it is small and is not the dominant dense
  cost.
- The outer surgery Schur split and the inner coupled `(r, theta_bulk)` Schur
  split are still the right structural model.
- The main remaining modeling problem is the drop-in inverse for the diagonal
  tensor blocks inside `A_bb`:
  - `A_rr`
  - `A_thetatheta`
  - `A_zetazeta`

## Current Conclusions

- The current full-rank `rt|z` drop-in is a useful diagnostic model, but it is
  not the preferred production direction for `k=1`.
- Naive z-rank truncation inside the vector-valued `rt|z` model is not a valid
  compression strategy for `k=1`.
- In the current debug implementation, truncating the z rank removes the
  discarded z-complement entirely, so the result is not just a weaker
  preconditioner; it is a structurally incomplete inverse model.
- The rank-ablation study therefore does not support promotion of a truncated
  vector-valued `rt|z` basis.
- The real asymptotic dense cost is in the diagonal bulk inverse drop-ins, not
  in the small square surgery block.
- In particular, the current `rt|z` diagonal-block model still stores dense
  per-mode `rt` inverses. That is the main reason it is not a good production
  endpoint for `k=1` even when the Schur structure itself is correct.
- The right object to compress is the geometric coefficient field inside the
  tensor block, not the inverse operator after assembly.
- Rank-3 metric-factor fits are already good enough at the block level to make
  the inversion of the resulting Kronecker sum the only serious remaining
  design problem.

## Preferred Direction For `A_bb`

- Keep the Schur structure unchanged:
  - outer surgery Schur on the reordered matrix,
  - inner Schur on the coupled `(r, theta_bulk)` block,
  - only replace the drop-ins for
    - `A_rr^{-1}`
    - `A_thetatheta^{-1}`
    - `A_zetazeta^{-1}`.
- Treat each diagonal bulk block as a geometric-weighted tensor-product mass
  matrix.
- Approximate the geometric weight, not the inverse subspace.
- The intended structured replacement is a Kronecker-style inverse built from
  one-dimensional weighted mass matrices.
- For the `k=1` diagonal bulk blocks, the natural coefficient fields are
  - `alpha_rr = J g^{rr}`
  - `alpha_thetatheta = J g^{theta theta}`
  - `alpha_zetazeta = J g^{zeta zeta}`.
- More generally, after surgery each diagonal block should be viewed as a sum
  of tensor-product contributions induced by the metric entries, not as a
  single scalar-weighted tensor mass matrix.
- Because `metric_inv_jkl` and `jacobian_j` are already available on the full
  tensor quadrature grid, the preferred first low-rank fit is directly on the
  coefficient 3-tensors
  - `J g^{rr}`
  - `J g^{theta theta}`
  - `J g^{zeta zeta}`
  rather than on the assembled block entries.

## Tensor-Block Inversion Notes

For one diagonal block, think of the matrix as

`A = integral alpha(r, theta, zeta) * basis_r * basis_theta * basis_zeta`.

The goal is to replace `alpha(r, theta, zeta)` by a separable or nearly
separable surrogate so that the block becomes a tensor product again.

### First Approximation: Constant Average Per Block

- Replace the geometric factor by one scalar average for each block:
  - `alpha_rr -> c_rr`
  - `alpha_thetatheta -> c_thetatheta`
  - `alpha_zetazeta -> c_zetazeta`
- This gives
  - `A_rr ~= c_rr * (M_r ⊗ M_theta ⊗ M_zeta)`
  - and analogously for `A_thetatheta` and `A_zetazeta`.
- This is the cheapest production-shaped tensor surrogate and should be tested
  first.

### Better Approximation: Separable One-Dimensional Averages

- Replace each block coefficient by a product of one-dimensional factors,
  for example
  - `alpha_rr(r, theta, zeta) ~= a_r^(rr)(r) * a_theta^(rr)(theta) * a_zeta^(rr)(zeta)`
  - and similarly for the other two diagonal blocks.
- Then each block becomes a Kronecker product of weighted one-dimensional mass
  matrices.
- This preserves the current Schur logic while producing a genuine tensor
  inverse apply.

### Next Approximation: Low-Rank Metric Factor Decomposition

- Instead of approximating each block coefficient by one scalar or by one
  separable product, approximate it by a short separable expansion,
  for example
  - `alpha_rr(r, theta, zeta) ~= sum_{ell=1}^R a_r_ell(r) a_theta_ell(theta) a_zeta_ell(zeta)`
  - and analogously for `A_thetatheta` and `A_zetazeta`.
- Then each diagonal block becomes a sum of Kronecker products of weighted
  one-dimensional mass matrices.
- This is the correct extension of the current Kronecker idea: rank-1 is the
  current scalar-average model, while `R > 1` gives a structured sum of
  Kronecker products.
- This is preferable to truncating the inverse itself, because the compressed
  object is the metric coefficient field rather than the already assembled
  inverse operator.
- The first implementation path should be a pure-JAX CP-ALS fit on the metric
  coefficient tensor itself.
- Once that fit is available, each recovered rank-1 term yields one weighted
  Kronecker product contribution to the diagonal tensor block.

### Basis-Attached Coefficient Sampling

- A practical intermediate model is to attach one sampled diagonal coefficient
  value to each tensor-product basis function, for each diagonal block.
- Greville points are the most natural basis-attached sample locations.
- For off-diagonal matrix entries, use the arithmetic average of the two basis
  values as the stand-in coefficient:
  - `alpha_ij ~= (alpha_i + alpha_j) / 2`.
- This gives a symmetric local coefficient model that is exact on the diagonal
  within the sampled representation.
- This model does not produce a single pure Kronecker factorization by itself,
  but it is still a meaningful structured approximation and can be used as a
  data source for a later separable fit.

### Important Detail: Use Overlap-Weighted Averages

- Plain volume averaging is the cheapest starting point, but it ignores where
  the basis-function overlaps are concentrated.
- A better approximation averages the coefficient against overlap densities,
  not against raw volume.
- The natural aggregate overlap weights are
  - `rho_r(r) = sum_i B_i(r)^2`
  - `rho_theta(theta) = sum_j T_j(theta)^2`
  - `rho_zeta(zeta) = sum_k Z_k(zeta)^2`.
- These emphasize the regions where basis overlaps are strongest.
- The practical separable fit is then built from one-dimensional marginals of
  `alpha` using these overlap weights.
- This gives a structured way to approximate the coefficient field near the
  effective overlap peak without fitting each matrix entry separately.

## What Not To Do

- Do not change the Schur structure just to accommodate a new diagonal-block
  inverse model.
- Do not interpret the failed z-rank truncation experiment as evidence that the
  Schur model is wrong.
- Do not promote a truncated vector-valued `rt|z` model as a production
  preconditioner for `k=1`.
- Do not spend additional effort compressing the inverse before trying to
  compress the metric coefficient field itself.

## If Only Matrix Entries Are Available

- This is now the fallback route, not the preferred route.
- If the metric coefficient field is unavailable and only the assembled block
  entries `A_ij` are available, the right target is the nearest low-rank sum of
  Kronecker products.
- Reshape the assembled tensor block using its product indexing,
  for example
  - rows: `(i_r, i_theta, i_z)`
  - cols: `(j_r, j_theta, j_z)`.
- Then seek an approximation of the form
  - `A ~= sum_{ell=1}^R Kr_ell ⊗ Ktheta_ell ⊗ Kz_ell`.
- The rank-1 version can be found by a Kronecker SVD / matrix rearrangement
  argument.
- Higher ranks can be found by alternating least squares or repeated residual
  deflation on the rearranged matrix.
- This matrix-only route is less interpretable than fitting the metric factor
  itself, but it is still a viable fallback when only `A_ij` is available.

## Experimental Defaults

- Use the rotating-ellipse map.
- Start with small resolutions such as `n = (4, 4, 4)`.
- Start with spline order `p = 2`.
- Validate both `dirichlet=False` and `dirichlet=True`.
- Compare extracted dense truth against `E_1 M_{1,raw} E_1^T`.

## Stage 0: Row Families

- [x] Confirm the extracted `k=1` component split:
  - `r`
  - `theta`
  - `zeta`
- [x] Confirm the surgery row sets:
  - `theta_surgery = first 2 * n_z rows of theta`
  - `zeta_surgery = first 3 * d_z rows of zeta`
- [x] Confirm the remaining rows are ordinary selector/permutation rows.

Gate:

- [x] Dense row-support counts match the operator formulas exactly.

## Stage 1: Support Structure

- [x] Measure row support counts in the dense extraction matrix `E_1`.
- [x] Confirm `theta_surgery` rows touch raw `r` and raw `theta`, but not raw `zeta`.
- [x] Confirm `zeta_surgery` rows touch only raw `zeta`.
- [x] Confirm bulk rows have single-entry support.

Gate:

- [x] Support summaries match the expected local formulas.

## Stage 2: Extracted Mass Block Split

- [x] Build the dense extracted mass matrix `A = E_1 M_{1,raw} E_1^T`.
- [x] Verify it matches the production extracted dense mass matrix.
- [x] Reorder or slice the matrix into
  - `r`
  - `theta_surgery`
  - `theta_bulk`
  - `zeta_surgery`
  - `zeta_bulk`
- [x] Record Frobenius norms of the induced block matrix.
- [x] Build an explicit permutation that moves surgery rows/columns to the front.
- [x] Inspect the reordered block matrix with ordering
  - `theta_surgery`
  - `zeta_surgery`
  - `r`
  - `theta_bulk`
  - `zeta_bulk`

Gate:

- [x] `||A - E_1 M_{1,raw} E_1^T||_max` is near machine precision.
- [x] The reordered surgery-vs-bulk split is visually and numerically clear.

## Stage 3: Dense Schur Prototype

- [x] Split the reordered matrix into
  - `surgery = theta_surgery ∪ zeta_surgery`
  - `bulk = r ∪ theta_bulk ∪ zeta_bulk`
- [x] Build the dense Schur complement against the bulk block.
- [x] Verify the dense block inverse built from that Schur complement matches the full dense inverse.
- [x] Record the surgery-block size for both BC choices.

Gate:

- [x] Dense Schur-complement apply matches dense inverse apply to tight tolerance.

## Stage 4: Identify The New Difficulty

- [x] Check whether the `zeta_surgery` coupling behaves like the scalar `k=0` core.
- [x] Check whether the genuinely new feature is only the `theta_surgery` block.
- [x] Decide whether the right abstraction is:
  - a Schur-complement split around all surgery rows, or
  - a bulk tensor model plus a smaller coupled correction.

Status:

- The new difficulty is not the small surgery square itself.
- The real remaining problem is the diagonal bulk inverse drop-ins inside the
  otherwise correct Schur structure.

## Stage 5: First Bulk Inverse Prototype On `A_rr`

- [x] Extract the `A_rr` diagonal bulk block from `A_bb`.
- [x] Treat `A_rr` as a tensor block with shape `((dr-1), nt, nz)`.
- [x] Test the same `rt|z` block-inverse construction used in the scalar `k=0` bulk.
- [x] Compare the resulting approximate inverse apply against the exact dense `A_rr^{-1}`.
- [x] Record whether `A_rr` is already well captured by this first tensor model.

Status:

- `A_rr` is captured reasonably well by the full-rank diagnostic `rt|z` model.
- This stage is useful as a structural diagnostic, but not as the preferred
  production approximation.
- Rank-1 and rank-3 metric-factor models were also tested here.
- Rank `3` is excellent at the block level; rank `1` is coarse but still
  usable as a cheap exact-product fallback.

## Stage 6: Coupled `(r, theta_bulk)` Bulk Model

- [x] Compare the exact dense inverse of the `(r, theta_bulk)` block against a pure block-diagonal model.
- [x] Separate coupling error from `A_rr` modeling error by comparing:
  - `diag(A_rr^{-1}, A_thetatheta^{-1})`
  - `diag(\widetilde A_rr^{-1}, A_thetatheta^{-1})`
- [x] Build a Schur model on the coupled `(r, theta_bulk)` block using:
  - exact `A_rr^{-1}`
  - `rt|z` approximate `A_rr^{-1}`
- [x] Record whether explicit `r-theta` coupling removes the large residual seen in the block-diagonal model.

Status:

- Explicit `(r, theta_bulk)` coupling is the main structural improvement.
- The Schur structure here should be kept.
- Future work in this stage is about swapping better drop-ins for the diagonal
  block inverses, not changing the coupled-Schur pattern.

## Stage 7: Reassemble The Full Bulk Block `A_bb`

- [x] Lift the coupled `(r, theta_bulk)` model back into the full bulk block.
- [x] Keep `zeta_bulk` separate at first and measure the ignored `(rt) <-> zeta_bulk` coupling.
- [x] Compare the exact dense `A_bb^{-1}` against:
  - exact coupled-`rt` Schur + exact `A_zeta zeta^{-1}`
  - approximate coupled-`rt` Schur + exact `A_zeta zeta^{-1}`
- [x] Decide whether `zeta_bulk` can remain a separate cheap block or whether it needs to join the Schur model.

Status:

- `A_zetazeta` itself is fairly well behaved as a diagonal tensor block.
- Adding the `(rt) <-> zeta_bulk` coupling gives a ceiling model, but the best
  production-shaped next step is still to improve the diagonal tensor-block
  drop-ins first.

## Stage 8: Outer Surgery Schur Prototype

- [x] Use the new `A_bb^{-1}` model inside the outer surgery Schur split on the full reordered extracted matrix.
- [x] Compare against the exact dense full inverse with:
  - exact bulk apply
  - approximate bulk apply
- [x] Record whether the remaining approximation error is dominated by the bulk model or by the outer surgery coupling.

Status:

- The outer surgery coupling is not the bottleneck.
- The remaining performance/design problem is the quality and structure of the
  diagonal bulk drop-ins.

## Stage 9: Ceiling Check With `rt-zeta` Coupling

- [x] Add one more Schur layer inside `A_bb` to include the `(rt) <-> zeta_bulk` coupling.
- [x] Compare the current full prototype against this coupled-bulk ceiling model.
- [x] Decide whether the extra coupling is worth the additional setup and bookkeeping.

Status:

- The coupled-`zeta` model remains a useful ceiling check.
- It should not replace the simpler production-shaped path unless its extra
  cost clearly pays for itself.

## Stage 10a: Replace The Diagonal Bulk Drop-Ins

- [x] Keep the current outer surgery Schur and inner `(r, theta_bulk)` Schur structure fixed.
- [ ] Replace the current diagonal `rt|z` drop-ins for
  - `A_rr^{-1}`
  - `A_thetatheta^{-1}`
  - `A_zetazeta^{-1}`
  by tensor-product inverse surrogates based on averaged geometric weights.
- [ ] Test two coefficient models for each diagonal block:
  - constant average,
  - separable one-dimensional overlap-weighted averages.
- [ ] Test a basis-attached Greville-sampled coefficient model with symmetric
  pair averaging for off-diagonal entries.
- [x] Test low-rank separable decompositions of the diagonal metric factors,
  producing sums of Kronecker products.
- [x] Start with a pure-JAX CP-ALS fit of the quadrature-grid coefficient
  tensors
  - `J g^{rr}`
  - `J g^{theta theta}`
  - `J g^{zeta zeta}`
  and measure reconstruction error by rank before assembling any new block
  surrogates.
- [ ] Test a shared-modal fast-diagonalisation inverse for low-rank metric
  Kronecker sums by diagonalising all rank terms approximately in one common
  per-axis basis.
- [ ] If tensor-product maps become the dominant geometry path later, revisit
  metric storage and consider making the grid-shaped metric layout the primary
  representation, with flattened quadrature views derived only for assembly.
- [x] Compare both variants against the current full-rank diagnostic `rt|z` drop-ins.
- [x] Benchmark the resulting full preconditioner with the same `k=1` harness.

Status:

- The metric-factor route is now the primary direction.
- Dense-inverse rank-3 metric surrogates gave the best debug benchmark, but
  that inversion path is not acceptable structurally.
- Rank-3 Neumann inversion of the Kronecker sum was not good enough.
- Rank-1 exact Kronecker inversion is a viable cheap fallback, but it does not
  match the current best `rt|z` diagnostic benchmark.
- The main remaining todo is therefore no longer the metric fit itself; it is
  the structured inverse of the low-rank Kronecker sum.

Gate:

- [ ] A tensor-product drop-in matches or improves the current production-shaped
  lowrank benchmark without relying on z-rank truncation.

## Stage 10: CG Benchmark

- [x] Wrap the hand-built `k=1` model as a preconditioner matvec in the original ordering.
- [x] Benchmark against production `jacobi` and `kronecker` on Gaussian RHS.
- [x] Compare the simpler separated-`zeta` model against the coupled-`zeta` ceiling model.

Status:

- Current benchmark ordering is roughly:
  - best debug path: dense-inverse metric-rank-3,
  - then the full-rank `rt|z` surgery models,
  - then metric-rank-1 exact,
  - then production `kronecker`,
  - then `jacobi`.
- Because the dense-inverse metric-rank-3 path is only a debug ceiling, the
  real remaining benchmark question is whether a structured `R > 1` inverse
  can beat the current `rt|z` diagnostic path.

## Stage 11: Promotion Criteria

- [ ] Do not change production `k=1` mass preconditioners until the surgery split is clear.
- [ ] Keep the interactive debug script as the regression harness after promotion.

Status:

- The surgery split is clear enough to proceed with debug-only structured
  inverse experiments.
- Production promotion is still blocked on the remaining structured inverse
  todo in Stage 10a.