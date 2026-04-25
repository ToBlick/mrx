# k=0 Polar FD Block Checklist

This document tracks the experimental implementation path for a boundary-aware
polar scalar FD preconditioner. All work stays in debug scripts until the
checks below pass on small problems.

## Experimental Defaults

- Use an analytical toroidal map.
- Default spline degrees: `p = 2` in all directions.
- Default resolutions: `n = (4, 5, 3)`.
- Validate both `dirichlet=False` and `dirichlet=True`.
- Compare every structured construction against dense truth on the extracted
  space.

## Stage 0: Baseline

- [x] Record current `k=0` FD vs Jacobi behavior with the existing debug script.
- [x] Keep the existing no-DBC path as a regression target.
- [x] Do not change production `operators.py` until the experimental script passes.

## Stage 1: Extracted-Space Block Split

- [x] Build the dense extracted operator `A = E A_full E^T`.
- [x] Split extracted DoFs into:
  - `core`: first `3 * n_z` rows from the polar fused block
  - `bulk`: remaining rows
- [x] Verify block slicing and reconstruction exactly reproduce `A`.
- [x] Verify the same split works for both DBC and no-DBC.

Verified on the experimental mass operator for `n = (4, 5, 3)` and `p = 2`.

Gate:

- [x] `||A - A_block_reconstructed||_max` is near machine precision.

## Stage 2: Bulk Restriction and Tensor Structure

- [x] Verify the exact bulk restriction identity
  - `A_bb = E_bulk A_full E_bulk^T`
- [x] Confirm that this bulk restriction check is exact for both DBC and no-DBC.
- [ ] Build the 1D reference scalar mass and stiffness matrices.
- [ ] Restrict the radial 1D matrices to the outer bulk indices.
- [ ] Reconstruct the bulk block as a Kronecker sum:
  - `K_r_bulk ⊗ M_t ⊗ M_z`
  - `M_r_bulk ⊗ K_t ⊗ M_z`
  - `M_r_bulk ⊗ M_t ⊗ K_z`
- [ ] Compare that tensor bulk operator against the dense extracted bulk block.

Note:
The exact bulk restriction check is now validated. The tensor-factorization
check remains open and is a separate structural claim, not a topological one.

Gate:

- [x] `||A_bb - E_bulk A_full E_bulk^T||_max` is near machine precision.
- [ ] `||A_bb - A_bb_tensor||_max` is near machine precision.

Mass-operator structural result on the analytical toroidal map:

- [x] The best single-`M_z` projection of the extracted mass bulk block,
  `A_bb ~= A_rtheta ⊗ M_z`, has very low exact Kronecker rank in its reduced
  `A_rtheta` factor on the experimental case.
  - `dirichlet=False`: exact rank `2`
  - `dirichlet=True`: exact rank `1`

This is a statement about the projected reduced factor `A_rtheta`, not about
the full bulk block `A_bb` on general geometries.

## Stage 3: Core and Coupling z-Separability

- [x] Check whether the core block can be written as:
  - `K_pol ⊗ M_z + M_pol ⊗ K_z`
- [x] Check whether the core-bulk coupling can be written as:
  - `B_0 ⊗ M_z + B_1 ⊗ K_z`
- [x] Solve for the small coefficient blocks from the dense truth and reconstruct.

Verified first on the mass operator, where this reduces to pure `M_z`
separability, and previously also observed on the stiffness experiment.

Gate:

- [x] `||A_cc - A_cc_sep||_max` is near machine precision.
- [x] `||A_cb - A_cb_sep||_max` is near machine precision.

## Stage 4: Dense Block Inverse

- [x] Build the block inverse using the Schur complement on the extracted space.
- [x] For the mass operator, compare against the direct dense inverse for both BCs.
- [x] Validate repeated block-inverse applies against direct dense solves.
- [ ] For stiffness, compare against a lightly shifted dense solve or the exact
  pseudoinverse on the range.

Gate:

- [x] On the mass operator, block inverse apply matches dense truth to tight tolerance.
- [ ] On the stiffness operator, block inverse apply matches dense truth to tight tolerance.

## Stage 5: Structured Inverse Prototype

- [x] For the mass operator, replace the dense bulk solve with the exact
  structured bulk inverse `A_bb^{-1} = A_rt^{-1} ⊗ M_z^{-1}`.
- [x] Keep the core solve dense and small.
- [x] Precompute the interface term `U = A_bb^{-1} A_bc`.
- [x] Precompute the Schur complement `S = A_cc - A_cb U`.
- [x] Verify the structured mass apply matches the dense block inverse.
- [ ] Replace the dense bulk solve with FD on `A_bb` for the stiffness/final
  target operator.

Verified on the experimental mass operator for both `dirichlet=False` and
`dirichlet=True`, with repeated structured applies matching the dense inverse
to roundoff.

Additional structural finding:

- [x] On the experimental mass operator, the bulk `r-θ` block admits an exact
  low-rank Kronecker decomposition.
- [x] Compare the current production-style scalar bulk surrogate against
  truncated low-rank bulk surrogates on the experimental mass block.
- [ ] Exploit that low-rank Kronecker structure in a scalable bulk-solve
  prototype, rather than storing the dense `A_rt^{-1}`.

Observed on the analytical toroidal map:

- The current production-style scalar bulk surrogate is far too crude as a
  bulk inverse model on the extracted mass block.
- A rank-1 low-rank surrogate improves the bulk operator error, but still
  gives very poor inverse-apply accuracy.
- A rank-2 low-rank surrogate recovers the experimental mass bulk block to
  roundoff for both `dirichlet=False` and `dirichlet=True`.

Observed on the rotating-ellipse map:

- The extracted-space block split remains exact, but the single-`M_z` bulk
  model is no longer accurate enough.
- The fitted single-`M_z` bulk error is small in absolute terms but is large
  enough to destroy the bulk inverse and the Schur complement.
- The reported rank-2 recovery applies only to the reduced factor in the best
  single-`M_z` projection, not to the true bulk block `A_bb`.
- The true bulk mass block `A_bb`, viewed as a 3-tensor with modes
  `(r_bulk^2, theta^2, zeta^2)`, has exact multilinear Tucker rank
  `(2, 2, 2)` for both `dirichlet=False` and `dirichlet=True`.
- The current CP-ALS experiment is numerically unstable and does not resolve
  the CP rank; this does not contradict the exact Tucker-rank result.
- The extracted `k=3` mass operator (`W^3 = 1/J`), viewed as a 3-tensor with
  modes `(r_3^2, theta_3^2, zeta_3^2)`, is identical for `dirichlet=False`
  and `dirichlet=True` on this test and is only approximately low
  multilinear rank.
- Among the three two-way matricizations, `rt|z` is the most compressible for
  `W^3`, with relative Frobenius errors about `2.4e-4` at rank `2` and
  `1.8e-6` at rank `3`; the `rz|t` and `tz|r` splits decay slightly more
  slowly but still reach about `1e-6` relative error by rank `4`.
- The `W^3` Tucker decay is similarly fast but not exact at low rank:
  `(2,2,2)` gives about `2.0e-3` relative Frobenius error,
  `(3,3,3)` gives about `5.5e-5`, and `(4,4,4)` gives about `1.3e-6`.
- A build-time hierarchical `rt|z` then `r|t` compression of the scalar
  quadrature weights has now been validated directly against the exact
  extracted sparse mass apply without constructing a dense full 3D matrix.
  - For `k=0`, both BC choices use outer rank `2` and inner ranks `(2, 2)`,
    giving roundoff-level apply errors.
  - For `k=3`, both BC choices use outer rank `2` and inner ranks `(3, 5)` at
    the current tolerances, giving about `3e-4` relative apply error.

Interpretation:

- The scalar averaged geometry model is not an adequate proxy for the bulk
  inverse on this toroidal test.
- The exact low-rank recovery is specific to the toroidal analytical map, so
  the next check must move to a `θ-ζ` coupled geometry rather than promoting
  the toroidal rank-2 result directly.
- On a `θ-ζ` coupled geometry, the first failing assumption is the single-`M_z`
  bulk model itself, before the reduced `r-θ` rank becomes the limiting issue.
- For the rotating-ellipse mass bulk block, the right exact statement is an
  exact Tucker `(2,2,2)` structure, not an exact single-`M_z` model and not
  currently an established CP-rank statement.
- For rotating-ellipse `W^3`, the right statement is weaker: the operator is
  strongly compressible and appears approximately low multilinear rank, but it
  does not show the same exact low-rank closure as `W^0` in the current tests.
- For scalar preconditioning, the right implementation path is therefore
  hierarchical compression of the quadrature weights, not dense operator SVD.

Next experimental step:

- [x] Exploit the exact Tucker `(2,2,2)` structure of the rotating-ellipse
  mass bulk block in a structured build-time apply prototype.
- [x] Check whether the analogous metric family for `k=3` retains low
  multilinear rank or only approximate low multilinear rank.
- [ ] Check whether the analogous metric families for `k=1` and `k=2` retain
  low multilinear rank or only approximate low multilinear rank.
- [ ] Promote the scalar hierarchical `rt|z` then `r|t` compression from the
  debug harness into the production preconditioner assembly path.

Gate:

- [x] On the mass operator, structured apply error against the dense block
  inverse is small on both BCs.
- [ ] On the stiffness/final target operator, structured apply error against
  the dense block inverse is small on both BCs.

## Stage 6: Solver-Level Checks

- [ ] Compare current `apply_hodge_kron_preconditioner` against the structured
  experimental apply.
- [ ] Confirm symmetry numerically: `<x, P y> == <y, P x>`.
- [ ] Confirm positivity on DBC and on the no-DBC range.
- [ ] Re-run the existing benchmark script and compare iteration counts.

Success criteria:

- No-DBC behavior does not regress materially.
- DBC no longer shows the current refinement crossover.

## Promotion Criteria

- [ ] Only move logic into production code after Stages 1 through 6 pass.
- [ ] Keep the experimental script as a regression harness even after promotion.