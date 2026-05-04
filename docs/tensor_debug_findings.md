# Tensor Debug Findings

## Scope

This note summarizes the dense debugging session around:

- the `k=0` tensor Hodge / Laplacian preconditioner,
- the `k=0` tensor mass preconditioner,
- the `k=1` tensor mass preconditioner, especially the `M1` bulk `rt` block.

The main diagnostic driver was the interactive script:

- `scripts/interactive/debug_k0_tensor_hodge_preconditioner.py`

The debugging goal changed over the session. The early question was whether the production tensor preconditioners were applying the wrong algebra. The later question was whether the remaining `M1` error came from omitted metric couplings, insufficient CP rank, or from the inverse model used inside the tensor block factors.

## Production Bugs That Were Fixed

### `k=0` tensor mass apply

The original direct `k=0` tensor mass apply used only a block-diagonal surrogate on the surgery/bulk split. That was wrong.

It now uses the surgery-plus-bulk Schur structure:

- apply the bulk tensor inverse,
- form the surgery Schur solve,
- apply the correction back to the bulk.

This is now implemented in `mrx/preconditioners.py` by storing and using `schur_inv` inside `K0TensorMassPreconditionerFactors`.

### `k=1` tensor mass apply

The original direct `k=1` tensor mass apply was also too weak. It treated the `r`, `theta`, and `zeta` blocks too independently and did not match the routed Schur structure.

It now uses two coupled levels:

1. an inner Schur solve on the bulk `rt = (r, theta)` block,
2. an outer Schur solve on the surgery-vs-bulk split.

In other words, the current `k=1` production tensor mass apply is not just a blockwise diagonal inverse. It is a nested Schur solve on the approximate tensor model.

### Public `k=0` tensor Hodge route

The public `kind="tensor"` route for the scalar Hodge / Laplacian preconditioner was still going through the older fast-diagonalized reference helper instead of the assembled surgery-plus-Schur tensor Hodge model.

That route now aliases the assembled `k=0` tensor Hodge apply.

## What The Dense Diagnostics Established

### `M0`

`M0` is in good shape.

Representative dense diagnostics from the latest run:

- `||P_tensor - M0^{-1}|| / ||M0^{-1}|| = 1.150e-02`
- `||M0 P_tensor - I|| / ||I|| = 1.645e-02`
- `||M0_bb(model) - M0_bb(exact)|| / ||M0_bb(exact)|| = 6.982e-03`
- `||P_M0 - A_model^{-1}|| / ||A_model^{-1}|| = 3.611e-14`
- `||A_model P_M0 - I|| / ||I|| = 4.949e-14`

Interpretation:

- the production `M0` apply matches the inverse of its assembled coupled model to machine precision,
- the remaining error is the usual model error, not an application or Schur-coupling bug.

### `M1`

The key `M1` conclusions are sharper.

Representative dense diagnostics from the latest run:

- `||P_tensor - M1^{-1}|| / ||M1^{-1}|| = 2.493e-02`
- `||M1 P_tensor - I|| / ||I|| = 4.840e-01`
- `||P_M1 - A_model^{-1}|| / ||A_model^{-1}|| = 5.841e-14`
- `||A_model P_M1 - I|| / ||I|| = 2.793e-14`
- `||P_M1 - A_schur^{-1}|| / ||A_schur^{-1}|| = 5.931e-14`
- `||A_model(parts) - A_schur|| / ||A_schur|| = 2.509e-14`

Interpretation:

- the production `M1` apply matches the inverse of its own assembled model to machine precision,
- the nested Schur logic is therefore correct,
- the remaining error is entirely in the quality of the modeled bulk operator.

### `k=0` tensor Hodge

The scalar Hodge preconditioner also matches the inverse of its assembled tensor model to machine precision.

Representative diagnostics from the latest run:

- `||P_tensor - K^+|| / ||K^+|| = 2.432e-02`
- `||K P_tensor - P_range|| / ||P_range|| = 1.251e-01`
- `||A_bb(model) - A_bb(exact)|| / ||A_bb(exact)|| = 3.635e-02`
- `||S(model) - S(exact)|| / ||S(exact)|| = 4.925e-03`
- `||P_tensor - H_model^{-1}|| / ||H_model^{-1}|| = 6.219e-15`
- `||H_model P_tensor - I|| / ||I|| = 6.217e-15`

Interpretation:

- the remaining scalar Hodge error is model error,
- not a mismatch between the production apply and the assembled tensor Hodge model.

## What Was Ruled Out For `M1`

The later part of the session focused on the `M1` bulk `rt` mismatch.

The following explanations were tested and effectively ruled out as the dominant cause.

### Not a Schur-apply bug

This was ruled out by the machine-precision agreement between:

- `P_M1` and `A_model^{-1}`,
- `P_M1` and `A_schur^{-1}`,
- `A_model(parts)` and `A_schur`.

So the current production path is doing the intended nested Schur solve on its approximate model.

### Not a CP field-fit problem

The fitted metric-field residuals are tiny:

- `cp(alpha_rr) relative error = 1.067e-04`
- `cp(alpha_thetatheta) relative error = 8.658e-05`
- `cp(alpha_zetazeta) relative error = 1.327e-04`

So the diagonal metric fields themselves are being fit accurately.

### Not primarily omitted metric off-diagonal entries

The metric off-diagonal field norms are not large enough to explain the observed `13%` `rt` error by themselves:

- `||J g^01|| / max(||J g^ii||) = 5.273e-02`
- `||J g^02|| / max(||J g^ii||) = 1.212e-02`
- `||J g^12|| / max(||J g^ii||) = 2.412e-03`

Also, the exact diagonal-only `rt` model is already quite close to the exact `rt` block:

- `||M1_rt(diag-exact) - M1_rt(exact)|| / ||M1_rt(exact)|| = 5.503e-03`

That is far smaller than the `13%` error in the production `rt` model.

## What The `M1` Evidence Now Points To

The dominant error appears to come from the inverse surrogate used inside the tensor diagonal block factors, not from the high-level Schur structure and not from the diagonal metric ansatz itself.

The strongest evidence is:

- `||M1_rr(diag-exact) - M1_rr(exact)|| / ||M1_rr(exact)|| = 2.335e-16`
- `||M1_rr(model) - M1_rr(diag-exact)|| / ||M1_rr(diag-exact)|| = 1.308e-01`
- `||S_theta(diag-exact) - S_theta(exact)|| / ||S_theta(exact)|| = 4.423e-02`
- `||S_theta(model) - S_theta(diag-exact)|| / ||S_theta(diag-exact)|| = 1.028e-01`
- `||M1_rt(diag-exact) - M1_rt(exact)|| / ||M1_rt(exact)|| = 5.503e-03`
- `||M1_rt(model) - M1_rt(diag-exact)|| / ||M1_rt(diag-exact)|| = 1.301e-01`

Interpretation:

- the exact diagonal-only block construction is already very accurate for `rr` and reasonably accurate for the full `rt` block,
- the large error enters when those diagonal blocks are inverted through the tensor inverse machinery,
- so the weak link is the multi-term inverse approximation inside `TensorDiagonalBlockInverseFactors`.

At present, the likely bottleneck is the shared-modal inverse surrogate used when the CP fit has more than one term.

## Current Best Summary

After the fixes in this session, the situation is:

- `k=0` mass: production apply is algebraically correct and the remaining error is model quality,
- `k=1` mass: production apply is algebraically correct and the remaining large error comes from the tensor inverse approximation inside the block factors,
- `k=0` Hodge: production apply is algebraically correct and the remaining error is model quality.

The most important negative result is that the current `M1` issue is not explained by:

- missing Schur coupling,
- insufficient CP fit quality,
- or omitted metric off-diagonal fields alone.

## Recommended Next Diagnostic

The clean next check is a debug-only rebuild of the `M1` model where:

- the exact diagonal-only `rr`, `theta`, and `zeta` blocks are assembled densely,
- those blocks are inverted densely,
- the same nested Schur structure is then rebuilt from those dense inverses.

If that collapses the `M1` bulk error from roughly `1.3e-1` toward the much smaller diagonal-only model errors, then the next production work should focus on improving the multi-term tensor inverse in `mrx/preconditioners.py`, not on changing the Schur logic.

## Alternative Plan: Best Rank-1 Kronecker Surrogate

Another clean follow-up is to stop approximating the inverse of a multi-term Kronecker sum term-by-term and instead approximate the full sum itself by a single best Kronecker product.

Suppose a block has the form

$$
M \approx \sum_{i=1}^{r} A_i \otimes B_i
$$

where the sum may come from the CP rank expansion or, more generally, from several coupled terms in the modeled block. The proposal is to replace that whole sum by

$$
M \approx \widetilde{A} \otimes \widetilde{B},
\qquad
M^{-1} \approx \widetilde{A}^{-1} \otimes \widetilde{B}^{-1}.
$$

This is attractive because it has no tuning parameters and gives a deterministic best rank-1 Kronecker approximation in the Van Loan-Pitsianis sense.

### Deterministic Construction

For each term, vectorize the factors:

$$
a_i = \mathrm{vec}(A_i),
\qquad
b_i = \mathrm{vec}(B_i).
$$

Then form the reshuffled matrix

$$
R = \sum_{i=1}^{r} a_i b_i^T.
$$

Compute its singular value decomposition

$$
R = U \Sigma V^T.
$$

Let $\sigma_1$ be the largest singular value with corresponding singular vectors $u_1$ and $v_1$. Reshape these back into matrices to define the best rank-1 surrogate:

$$
\widetilde{A} = \mathrm{unvec}(\sigma_1 u_1),
\qquad
\widetilde{B} = \mathrm{unvec}(\sigma_1 v_1).
$$

Then use

$$
M^{-1} \approx \widetilde{A}^{-1} \otimes \widetilde{B}^{-1}.
$$

### Why This Is Worth Testing

- It avoids the shared-modal multi-term inverse surrogate entirely.
- It compresses the full modeled sum before inversion rather than trying to invert a sum of several Kronecker terms directly.
- It gives a single obvious diagonalization anchor once the best rank-1 surrogate is formed.
- It is especially natural in the scalar case, where the modeled operator already behaves like one tensor block on one tensor-product space.

### Cheap Screening Test

This idea is easy to evaluate before any serious implementation work.

The main diagnostic is the singular value decay of the reshuffled matrix $R$.

- If $\sigma_1$ clearly dominates, then a single Kronecker surrogate is likely a good approximation and is worth testing in the dense debug path.
- If the decay is slow, then a rank-1 surrogate is probably too aggressive and this route is unlikely to beat the current multi-term model.

So the first check should be:

1. form the reshuffled matrix for the modeled block,
2. inspect its singular value decay,
3. only if the leading singular value is dominant, build the dense rank-1 Kronecker surrogate and compare it against the current block model and inverse model.

This gives a cheap go/no-go diagnostic for whether the best single-Kronecker approximation is competitive for the problematic `M1` blocks.

## Temporary Caveat: Exact-Block Chebyshev Prototype Is Dense

There is a temporary implementation caveat from the recent exact-block Chebyshev experiment.

Not all densification in the tensor mass path is new. The tensor preconditioner assembly was already being fed by dense full matrices through `jnp.asarray(sp.todense())` in `mrx/operators.py` before this latest experiment.

What is new in the current prototype is that the exact-block Chebyshev attempt adds further dense data on top of that:

- dense extracted subblocks such as `arr_block`, `theta_block`, `zeta_block`, and `r_bulk_block` are stored in the tensor factor payloads,
- those dense extracted block matrices are then used directly during the production tensor apply path for the current Chebyshev prototype,
- the benchmark helper in `scripts/interactive/mass_preconditioner_demo.py` is also intentionally dense, which is acceptable there, but the production route should not stay in this form.

So the current status is:

1. the broad full-matrix densification predated this experiment,
2. the extra exact-block dense storage and exact-block dense apply logic were added only for the recent Chebyshev prototype,
3. this should be treated as temporary and cleaned up later.

The intended long-term direction is:

- keep dense exact-block logic only in benchmark/debug surfaces,
- remove the extra dense exact-block storage from the production tensor preconditioner payloads,
- if exact-block Chebyshev survives benchmarking, rebuild it around restricted matvecs and stored spectral bounds rather than stored dense block matrices.

## Alternative Plan: VLP + Neumann Hybrid

Another possible follow-up does not rely on Chebyshev at all.

The idea is to combine two deterministic ingredients:

1. compress only the dominant base block to a single Kronecker term using a Van Loan-Pitsianis style rank-1 surrogate,
2. recover the lost structure through a fixed-order Neumann expansion.

The intended use case is when the dominant base block is itself already a short sum of Kronecker products and therefore no longer has a trivial inverse.

### Setup

Suppose the modeled block is decomposed as

$$
M = T_0 + E_{\mathrm{ALS}},
$$

where the dominant part is a short sum

$$
T_0 = X_1 + X_2 + X_3,
\qquad
X_j = A_j \otimes B_j \otimes C_j,
$$

and $E_{\mathrm{ALS}}$ collects the remaining ALS terms.

### Step 1: Compress Only The Dominant Base

Apply the hierarchical Van Loan-Pitsianis compression only to the three terms in $T_0$ to build a rank-1 surrogate

$$
\widetilde{T}_0 = \widetilde{A} \otimes \widetilde{B} \otimes \widetilde{C}.
$$

This gives a base whose inverse is trivial:

$$
\widetilde{T}_0^{-1} = \widetilde{A}^{-1} \otimes \widetilde{B}^{-1} \otimes \widetilde{C}^{-1}.
$$

### Step 2: Fold The Compression Defect Into The Perturbation

Because $\widetilde{T}_0$ is only an approximation of $T_0$, define the compression defect

$$
\Delta T = T_0 - \widetilde{T}_0.
$$

Then combine that defect with the remaining ALS terms:

$$
E_{\mathrm{total}} = \Delta T + E_{\mathrm{ALS}}.
$$

Now the modeled matrix is written exactly as

$$
M = \widetilde{T}_0 + E_{\mathrm{total}}.
$$

The key point is that $E_{\mathrm{total}}$ is still just a sum of Kronecker terms. Nothing has been discarded; the compression error is explicitly tracked.

### Step 3: Use A Fixed-Order Neumann Inverse

With $\widetilde{T}_0$ as the invertible anchor, apply a truncated Neumann expansion:

$$
M^{-1}
\approx
\widetilde{T}_0^{-1}
- \widetilde{T}_0^{-1} E_{\mathrm{total}} \widetilde{T}_0^{-1}
+ \widetilde{T}_0^{-1} E_{\mathrm{total}} \widetilde{T}_0^{-1} E_{\mathrm{total}} \widetilde{T}_0^{-1}
- \cdots
$$

This is attractive because it avoids Krylov/Chebyshev tuning and logically rebuilds the original three-term base through the perturbation series rather than discarding it.

### Why This Is Structurally Appealing

- It keeps a trivially invertible rank-1 anchor.
- It does not throw away the mismatch between the original 3-term base and the compressed anchor.
- It incorporates the remaining ALS terms in the same perturbation machinery.
- It stays in tensor/Kronecker algebra throughout, at least formally.

### Practical Caveat

This route is deterministic, but it still requires a chosen truncation order for the Neumann series. So it avoids Chebyshev-specific tuning, but it is not literally parameter-free.

### Cheap Screening Test

The first thing to inspect is again a compression diagnostic on the dominant base block alone.

1. Form the hierarchical reshuffled operator for the 3-term base $T_0$.
2. Check the singular value decay of that reshuffled matrix.
3. Only if the rank-1 surrogate for $T_0$ looks strong enough, build $\widetilde{T}_0$ and test the first few fixed-order Neumann truncations in the dense debug path.

If the rank-1 surrogate for the dominant base is already weak, then this hybrid route is unlikely to be competitive.

## Benchmark Plan: Block-Inversion Comparison Ladder

The intended benchmark set should stay narrow and compare only the following five solver/preconditioner choices for the same test problems:

1. Jacobi on the whole matrix.
2. Chebyshev on the whole matrix with Jacobi as the smoother.
3. Schur decomposition with Chebyshev on the blocks and Jacobi as the block smoother.
4. Schur decomposition with tensor inverses on the blocks.
5. Schur decomposition with Chebyshev on the blocks and tensor inverses as the block smoother.

The point of this comparison ladder is to isolate where the gain is actually coming from:

- whole-matrix polynomial acceleration versus blockwise polynomial acceleration,
- Schur decomposition alone versus Schur plus tensor structure,
- tensor blocks used directly versus tensor blocks used only as smoothers inside Chebyshev.

So the benchmark focus should remain on these five cases rather than broadening into many loosely related variants.

## Interpretation Of The Dense Spectrum Results

The dense toy-case spectrum screens give a clear structural message.

### Raw Conditioning

For the representative toy case that was checked, the raw block condition numbers were roughly:

- `M0 bulk`: `3.5e+02`
- `M1 arr`: `2.2e+03`
- `M1 theta`: `1.4e+02`
- `M1 zeta`: `1.4e+02`

So the hard block is `M1 arr`. The scalar bulk is moderate, and the `theta` / `zeta` vector blocks are comparatively easy.

### What The Preconditioned Spectra Show

The more important result is the preconditioned spectrum clustering.

- Jacobi remains poor everywhere and mostly reflects the raw conditioning.
- The tensor block preconditioners already cluster all block spectra very tightly.
- The exact inverse of the modeled tensor block clusters them even more tightly, often essentially to the identity.

Representative outcomes from the same toy case were:

- `M0 bulk`: tensor rank-1 already gives condition number about `1.17`, while the exact inverse of the modeled rank-2 or rank-4 block is essentially exact.
- `M1 zeta`: tensor rank-4 gives condition number about `1.02`, and the exact inverse of the modeled rank-4 block is essentially exact.
- `M1 theta`: tensor rank-4 gives condition number about `1.44`, while the exact inverse of the modeled rank-4 block is about `1.01`.
- `M1 arr`: tensor rank-4 still gives a respectable condition number about `1.86`, but the exact inverse of the modeled rank-4 block is about `1.02`.

### Main Interpretation

This separates two questions very cleanly:

1. Is the tensor model good enough?
2. Is the inverse surrogate for that tensor model good enough?

For the difficult `M1 arr` block, the answer is:

- the tensor model itself is already very good,
- the remaining weakness is in the approximate inverse of the multi-term tensor block, not in the block model.

That is exactly what the gap between `tensor-rk` and `model-inv-rk` means.

- `model-inv-rk` tests the quality of the modeled block itself by inverting that model densely.
- `tensor-rk` tests the actual production-style tensor inverse surrogate.

When `model-inv-rk` is excellent but `tensor-rk` is noticeably worse, the model is not the bottleneck anymore. The inverse surrogate is.

### Practical Consequence

The current evidence points to the following engineering conclusion.

- `M0` is essentially solved by low-rank tensor structure.
- `M1 zeta` is also essentially solved.
- `M1 theta` is model-easy but inverse-moderate.
- `M1 arr` is the main remaining difficulty, and that difficulty lies primarily in how to invert a multi-term tensor model.

So the immediate next production path should not be to keep chasing better tensor fits for `M1`. The remaining design question is whether the current tensor inverse is already strong enough to use directly inside the outer Krylov solver, or whether it is still worth wrapping it inside block Chebyshev.

## Updated View Of The Tensor Smoother

The later dense spectrum screens changed the conclusion above in an important way.

### What The New Tensor-Smoother Screens Show

After the `k=3` basis fix, the tensor inverse itself already flattens the block spectra very strongly.

Representative tensor-preconditioned condition numbers from the latest toy-case screen were:

- `M0 bulk`: about `1.05`
- `M1 arr`: about `1.86`
- `M1 theta`: about `1.44`
- `M1 zeta`: about `1.02`
- `M2 r_bulk`: about `1.42`
- `M2 theta`: about `1.23`
- `M2 zeta`: about `1.02`
- `M3`: about `1.06`

This is qualitatively different from Jacobi. The tensor block is not acting like a weak smoother anymore. It is already an aggressive spectrum flattener.

### What The Robust Rank-1 Alternatives Show

The later comparisons against:

- hierarchical VLP rank-1 anchors,
- grouped `rt|z` rank-1 anchors,
- and direct no-split rank-1 ALS on the known Kronecker-sum terms,

all pointed to the same conclusion.

- The split choice is not the main issue; all VLP splits landed on essentially the same rank-1 anchor.
- The no-split rank-1 ALS fit lands on essentially the same quality as the VLP anchor.
- Those robust rank-1 routes are competitive on easy blocks and on `M1 arr`, but they are materially worse on `M2 theta`, `M2 zeta`, and `M3`.

So the robust rank-1 alternatives do not change the overall design picture. They provide a simpler fallback, but they do not outperform the current multi-term tensor inverse.

### Is Block Chebyshev Still Worth It?

This is now the key practical question.

If the tensor-preconditioned block already has condition number below about `2`, then a further inner Chebyshev iteration can only deliver a limited incremental gain. It may still reduce the block error slightly, but the spectrum is already clustered enough that the outer Krylov solver should see a very benign operator.

That suggests the following working conclusion.

- `Schur + tensor blocks + outer CG` should now be treated as the primary baseline, not as an intermediate fallback.
- `Schur + Chebyshev(blocks, tensor smoother)` is still worth benchmarking, but mainly to test whether a few cheap inner polynomial steps reduce outer iterations enough to justify the added complexity.
- For blocks already near condition `1.0` to `1.3`, the answer may well be no.
- The only place where block Chebyshev still has a plausible case is the weaker tail of the current tensor inverse, especially `M1 arr` and secondarily `M1 theta`.

## Current Production Direction

The current production direction is now narrower than the earlier exploration paths.

### Restrict Production Tensor Blocks To Rank 1

For production, the tensor mass blocks should be restricted to rank-1 tensor smoothers.

The main reason is that the rank-1 path is the only one that keeps the block apply genuinely simple and fully tensor-separable.

- the modeled block is one tensor-product term,
- the inverse apply is one tensor-product inverse,
- there is no need for shared modal diagonalization,
- there is no need for dense surrogate inverses,
- and the dense debug screens show that this rank-1 path is already quite strong.

In the latest dense rank-1 screens, `tensor-r1` and `model-inv-r1` agree to displayed precision on every tested block. That means the production tensor apply is already matching the exact inverse of its own rank-1 surrogate model in the cases being screened.

So the current production baseline should be:

- Schur decomposition at the outer block level,
- rank-1 tensor block smoothers inside those Schur blocks,
- outer CG as the main Krylov solver.

### Chebyshev Is Now A Wrapper Option, Not The Base Design

If the rank-1 tensor smoother is not sufficient by itself on the hardest blocks, the next production option is to wrap it in block Chebyshev rather than to return to more complicated multi-term inverse models.

The dense bound checks are encouraging here.

For the tensor smoother, the production Lanczos/Chebyshev bound routine gives estimates that are close to the true dense extremal eigenvalues of the preconditioned block operator, with the remaining gap largely explained by the deliberate safety factors:

- `lambda_max` is consistently close to the inflated `1.1 *` true maximum,
- `lambda_min` is consistently close to the deflated `0.85 *` true minimum.

So, for the tensor smoother, the current production eigenvalue estimator appears good enough to support Chebyshev.

By contrast, the same lower-bound estimator is much less informative for Jacobi. The Jacobi `lambda_min` estimates sit well above the true left edge of the dense spectrum, so Jacobi should not be treated as the main target for calibrated block Chebyshev in this design thread.

### What We Are Not Taking Forward

The production path should not currently depend on any of the following:

- shared simultaneous diagonalization of multi-term tensor blocks,
- dense exact inverses of surrogate block models,
- exact-torus reference inverses,
- or other dense debug-only constructions used only to screen ideas.

Those remain useful as diagnostics and upper bounds, but they are no longer the primary engineering direction.

### Practical Decision

The practical production decision is therefore:

1. keep the production tensor mass smoother path rank 1,
2. use that rank-1 tensor apply directly as the default block smoother/preconditioner,
3. benchmark block Chebyshev on top of that rank-1 tensor smoother for the harder blocks,
4. only add the Chebyshev wrapper where it reduces outer solve cost enough to justify the extra applies.

So the design focus has shifted away from building stronger dense surrogate inverses and toward answering a much narrower question:

- is rank-1 tensor smoothing already sufficient,
- and where, if anywhere, is a small Chebyshev wrapper on top of that rank-1 smoother actually worth it?

In other words, once the tensor inverse flattens the block spectrum this strongly, the default expectation should shift. The burden of proof is now on block Chebyshev to show that it beats going straight to the outer CG.

### Updated Benchmark Priority

The benchmark ladder should therefore still be run, but the interpretation should be updated.

- The most important comparison is now `Schur + tensor blocks` versus `Schur + Chebyshev(blocks, tensor smoother)`.
- The purpose is no longer to rescue a weak tensor smoother.
- The purpose is to decide whether a very strong tensor smoother should be used directly, or only as the inner accelerator inside a short Chebyshev polynomial.

At the moment, the dense evidence suggests that direct outer CG on top of `Schur + tensor blocks` is likely the simpler default, with block Chebyshev needing to earn its keep by measurable iteration savings.

## Harder-Geometry Update

The later harder-geometry screen makes the production choice clearer.

### Raw Conditioning In The Harder Case

Representative raw block condition numbers in the harder setup were roughly:

- `M0 bulk`: `8.0e+03`
- `M1 arr`: `1.9e+04`
- `M1 theta`: `5.8e+03`
- `M1 zeta`: `3.4e+03`
- `M2 r_bulk`: `2.5e+03`
- `M2 theta`: `8.2e+03`
- `M2 zeta`: `4.1e+03`
- `M3`: `1.7e+03`

So the harder geometry did exactly what it should do diagnostically: it separated genuinely robust inverse strategies from ones that only look acceptable on the easier toy case.

### Tensor Inverse Versus Rank-1 Compression In The Harder Case

In that harder screen, the production-style tensor inverse still flattened the spectra strongly:

- `M0 bulk`: tensor rank-4 about `1.17`
- `M1 arr`: tensor rank-4 about `1.62`
- `M1 theta`: tensor rank-4 about `1.89`
- `M1 zeta`: tensor rank-4 about `1.22`
- `M2 r_bulk`: tensor rank-4 about `2.58`
- `M2 theta`: tensor rank-4 about `1.40`
- `M2 zeta`: tensor rank-4 about `1.11`
- `M3`: tensor rank-4 about `1.12`

By contrast, all rank-1 routes remained much weaker on the hard blocks:

- `M1 arr`: rank-1 compression about `7.1`
- `M1 theta`: rank-1 compression about `4.7`
- `M2 r_bulk`: rank-1 compression about `4.8`
- `M2 theta`: rank-1 compression about `5.9`
- `M3`: rank-1 compression about `2.1`

This harder case is the strongest current argument for the production choice.

- On the easy case, rank-1 compression looked plausible on several blocks.
- On the harder case, it is clearly not competitive as a universal inverse strategy.
- The current simultaneous-diagonalization-style tensor inverse still behaves well on every tested block.

So if one production path is needed across the board, the current tensor inverse should remain the default. The robust rank-1 alternatives are now best understood as fallback options, not as replacements.

### Where The Remaining Tensor-Inverse Gap Still Lives

The harder case also sharpens where the inverse surrogate is still leaving performance on the table.

- `M2 r_bulk`: tensor rank-4 about `2.58`, but model-inverse rank-4 about `1.18`
- `M1 theta`: tensor rank-4 about `1.89`, but model-inverse rank-4 about `1.12`
- `M1 arr`: tensor rank-4 about `1.62`, but model-inverse rank-4 about `1.18`

So the current tensor inverse is already good enough to be useful in production, but there is still headroom in the inverse surrogate on a few harder blocks, especially `M2 r_bulk` and secondarily `M1 theta` and `M1 arr`.

## Apply-Cost Comparison: Tensor Inverse Versus Rank-1 VLP

The cost tradeoff is real, but it is not arbitrarily large.

### Current Tensor Inverse Apply

The current tensor block apply for multi-term factors works by:

1. applying three forward modal transforms,
2. building the modal denominator by summing over all retained Kronecker terms,
3. dividing pointwise in modal space,
4. applying three backward modal transforms.

Ignoring optional Richardson refinement, this is therefore:

- six dense tensor-product transforms,
- plus one `R`-term accumulation over the modal tensor grid,
- plus one pointwise divide.

### Rank-1 VLP / Rank-1 ALS Apply

Once a rank-1 Kronecker anchor has been formed and inverted, the apply reduces to a single separable inverse:

- one radial factor apply,
- one poloidal factor apply,
- one toroidal factor apply.

That is just three tensor-product transforms.

### Practical Cost Estimate

So, at the block-apply level, rank-1 VLP is cheaper by roughly a factor of two in the dominant transform count:

- rank-1 inverse: about `3` tensor contractions,
- current tensor inverse: about `6` tensor contractions,
- with some extra overhead from summing the `R` modal diagonal contributions.

So a reasonable mental model is:

- rank-1 VLP is roughly `2x` cheaper per block apply,
- perhaps modestly more than `2x` when the retained rank is larger and the modal denominator assembly is not negligible,
- but not an order-of-magnitude cheaper.

The harder-case spectrum data should therefore be interpreted through that lens.

- If a block only degrades marginally under rank-1 compression, a `~2x` cheaper apply may be worth it.
- If a block jumps from condition `~1.4-1.9` to `~4.5-7.1`, then the cheaper apply is probably not worth the outer-iteration penalty.

### Current Production Recommendation

Given the harder-geometry results, the best current production recommendation is:

- keep the current simultaneous-diagonalization-style tensor inverse as the default production block inverse,
- benchmark `Schur + tensor blocks + outer CG` first,
- treat block Chebyshev as an optional refinement that must justify its extra block applications,
- and treat rank-1 VLP only as a fallback for cases where simplicity or memory pressure matters more than iteration quality.

## Rank-Expanded Geometry And Diagonalization Structure

The current note has mostly discussed the scalar mass-style block models, where each retained CP term gives one pure tensor-product factor

$$
M^{(m)} = M_r^{(m)} \otimes M_\theta^{(m)} \otimes M_\zeta^{(m)}.
$$

For that case, the only difficulty is how to invert a short sum of such tensor products.

That is not the whole story, though. The vectorial mass blocks `M1` and `M2` already carry the same directional mixed-basis structure that motivated the later diagonalization discussion. The scalar stiffness case is only the cleanest place where that structure collapses to a true Kronecker sum on one tensor-product space.

To avoid overloading notation, in this section `D` should be read as “the 1-D factor assembled on the derivative spline space along that axis”. In the scalar stiffness case that really is a stiffness factor; in the vectorial mass case it is the 1-D mass factor on the derivative spline space.

The common directional pattern is therefore

$$
D \otimes M \otimes M
+
M \otimes D \otimes M
+
M \otimes M \otimes D.
$$

That structure is worth writing down explicitly, because it suggests a better diagonalization strategy than the current shared-basis heuristic.

### Vectorial Mass `M1` And `M2` Already Have The Mixed-Basis Pattern

For the diagonal metric model of `M1`, the three component blocks already use different tensor-product spaces:

- `arr`: derivative in `r`, primal in `theta`, primal in `zeta`,
- `theta`: primal in `r`, derivative in `theta`, primal in `zeta`,
- `zeta`: primal in `r`, primal in `theta`, derivative in `zeta`.

So after CP expansion of the three diagonal metric fields, the diagonal-block model has the form

$$
M_{1,\mathrm{diag}}
\approx
\operatorname{blkdiag}\!\Bigl(
\sum_{m=1}^{R_r} D_r^{(r,m)} \otimes M_\theta^{(r,m)} \otimes M_\zeta^{(r,m)},
\sum_{m=1}^{R_\theta} M_r^{(\theta,m)} \otimes D_\theta^{(\theta,m)} \otimes M_\zeta^{(\theta,m)},
\sum_{m=1}^{R_\zeta} M_r^{(\zeta,m)} \otimes M_\theta^{(\zeta,m)} \otimes D_\zeta^{(\zeta,m)}
\Bigr).
$$

Likewise, for `M2` the component spaces are

- `r_bulk`: primal in `r`, derivative in `theta`, derivative in `zeta`,
- `theta`: derivative in `r`, primal in `theta`, derivative in `zeta`,
- `zeta`: derivative in `r`, derivative in `theta`, primal in `zeta`.

So the diagonal-block model has the form

$$
M_{2,\mathrm{diag}}
\approx
\operatorname{blkdiag}\!\Bigl(
\sum_{m=1}^{R_r} M_r^{(r,m)} \otimes D_\theta^{(r,m)} \otimes D_\zeta^{(r,m)},
\sum_{m=1}^{R_\theta} D_r^{(\theta,m)} \otimes M_\theta^{(\theta,m)} \otimes D_\zeta^{(\theta,m)},
\sum_{m=1}^{R_\zeta} D_r^{(\zeta,m)} \otimes D_\theta^{(\zeta,m)} \otimes M_\zeta^{(\zeta,m)}
\Bigr).
$$

So the pushback is correct: `M1` and `M2` are not generic pure-mass tensor blocks. They already have the same directional `D/M` structure, just componentwise rather than as one scalar Kronecker sum on a single tensor-product space.

The current production tensor preconditioner already exploits this partially by building `arr`, `theta`, and `zeta` factors separately with the corresponding derivative/primal basis choices. What it does not yet do is align the retained geometric ranks across those directional families.

### Exact Form After Rank Expansion

For a scalar diagonal second-order block, the modeled operator is built from the three diagonal mapped coefficient fields

$$
\alpha_{rr},\qquad \alpha_{\theta\theta},\qquad \alpha_{\zeta\zeta}.
$$

In the current code path for the scalar tensor Hodge bulk, these are fit independently by CP expansions:

$$
\alpha_{rr} \approx \sum_{m=1}^{R_r} \omega_{r,m}\, a_{r,m} \otimes b_{r,m} \otimes c_{r,m},
$$

$$
\alpha_{\theta\theta} \approx \sum_{m=1}^{R_\theta} \omega_{\theta,m}\, a_{\theta,m} \otimes b_{\theta,m} \otimes c_{\theta,m},
$$

$$
\alpha_{\zeta\zeta} \approx \sum_{m=1}^{R_\zeta} \omega_{\zeta,m}\, a_{\zeta,m} \otimes b_{\zeta,m} \otimes c_{\zeta,m}.
$$

After assembling the 1-D weighted factors, the modeled bulk matrix is therefore

$$
A_{\mathrm{model}}
= \sum_{m=1}^{R_r}
D_r^{(r,m)} \otimes M_\theta^{(r,m)} \otimes M_\zeta^{(r,m)}
+ \sum_{m=1}^{R_\theta}
M_r^{(\theta,m)} \otimes D_\theta^{(\theta,m)} \otimes M_\zeta^{(\theta,m)}
+ \sum_{m=1}^{R_\zeta}
M_r^{(\zeta,m)} \otimes M_\theta^{(\zeta,m)} \otimes D_\zeta^{(\zeta,m)}.
$$

So, strictly speaking, the current implementation is not yet in the aligned-rank form “one rank gives one `DMM + MDM + MMD` block”. It is a sum over directional families, each with its own CP expansion.

That said, your proposed viewpoint is the natural next simplification.

The current production direction for the scalar `k=0` tensor Hodge route is now
to use that simplification at rank 1 only:

$$
\alpha_{rr} \approx c_r w,
\qquad
\alpha_{\theta\theta} \approx c_\theta w,
\qquad
\alpha_{\zeta\zeta} \approx c_\zeta w,
$$

with one shared separable scalar field

$$
w(\theta,r,\zeta) \approx b_\theta(\theta) b_r(r) b_\zeta(\zeta),
$$

followed by a 1-D generalized eigendecomposition on each axis for the weighted
mass/stiffness pairs. This keeps the outer Schur structure aligned with the
mass-preconditioner refactor: only the surgery slice stays dense, while the
bulk inverse is applied through fast diagonalization and the core/bulk coupling
is applied by sparse stiffness matvecs rather than stored dense coupling blocks.

If the three diagonal geometric fields are expanded in a shared rank basis, or are post-processed into aligned dominant ranks, then one retained rank would contribute exactly one operator of the form

$$
A^{(m)}
= D_r^{(m)} \otimes M_\theta^{(m)} \otimes M_\zeta^{(m)}
+ M_r^{(m)} \otimes D_\theta^{(m)} \otimes M_\zeta^{(m)}
+ M_r^{(m)} \otimes M_\theta^{(m)} \otimes D_\zeta^{(m)}.
$$

Then the full modeled block would be

$$
A_{\mathrm{model}} \approx \sum_{m=1}^{R} A^{(m)}.
$$

This is the form that makes the diagonalization question much cleaner.

It is also important to separate two cases.

- For scalar stiffness-style blocks, this aligned-rank form is a genuine Kronecker sum on one coefficient space.
- For vectorial mass `M1` and `M2`, the analogous structure is a direct sum across components of tensor blocks with different derivative/primal axis choices.

So the same rank-alignment idea applies in both settings, but the exact modal algebra is simpler in the scalar stiffness case than in the vectorial mass case.

### Why One Rank Is Special

For one fixed rank `m`, suppose the three mass factors are SPD and define Cholesky factors

$$
M_r^{(m)} = L_r^{(m)} {L_r^{(m)}}^T,
\qquad
M_\theta^{(m)} = L_\theta^{(m)} {L_\theta^{(m)}}^T,
\qquad
M_\zeta^{(m)} = L_\zeta^{(m)} {L_\zeta^{(m)}}^T.
$$

Whiten the directional stiffness pieces in the mass metric:

$$
\widehat D_r^{(m)} = {L_r^{(m)}}^{-1} D_r^{(m)} {L_r^{(m)}}^{-T},
$$

$$
\widehat D_\theta^{(m)} = {L_\theta^{(m)}}^{-1} D_\theta^{(m)} {L_\theta^{(m)}}^{-T},
$$

$$
\widehat D_\zeta^{(m)} = {L_\zeta^{(m)}}^{-1} D_\zeta^{(m)} {L_\zeta^{(m)}}^{-T}.
$$

Then the whole rank-`m` block factors as

$$
A^{(m)}
= (L_r^{(m)} \otimes L_\theta^{(m)} \otimes L_\zeta^{(m)})
\Bigl(
\widehat D_r^{(m)} \otimes I \otimes I
+ I \otimes \widehat D_\theta^{(m)} \otimes I
+ I \otimes I \otimes \widehat D_\zeta^{(m)}
\Bigr)
(L_r^{(m)} \otimes L_\theta^{(m)} \otimes L_\zeta^{(m)})^T.
$$

Now each whitened 1-D operator can be diagonalized independently:

$$
\widehat D_r^{(m)} = Q_r^{(m)} \Lambda_r^{(m)} {Q_r^{(m)}}^T,
\qquad
\widehat D_\theta^{(m)} = Q_\theta^{(m)} \Lambda_\theta^{(m)} {Q_\theta^{(m)}}^T,
\qquad
\widehat D_\zeta^{(m)} = Q_\zeta^{(m)} \Lambda_\zeta^{(m)} {Q_\zeta^{(m)}}^T.
$$

So one rank is exactly diagonalized by the tensor-product basis

$$
V_r^{(m)} = {L_r^{(m)}}^{-T} Q_r^{(m)},
\qquad
V_\theta^{(m)} = {L_\theta^{(m)}}^{-T} Q_\theta^{(m)},
\qquad
V_\zeta^{(m)} = {L_\zeta^{(m)}}^{-T} Q_\zeta^{(m)},
$$

with modal denominator

$$
\lambda_{r,a}^{(m)} + \lambda_{\theta,b}^{(m)} + \lambda_{\zeta,c}^{(m)}.
$$

This is the main structural point: within one retained geometric rank, the problem is not a generic multi-term tensor inverse. It is a separable Kronecker sum in the mass metric, and that part can be diagonalized exactly.

### Where The Actual Difficulty Starts

The hard part is not one rank. The hard part is combining several ranks when the mass factors and therefore the natural modal bases differ from rank to rank.

If the leading ranks were all perfectly aligned, we could keep one common basis and simply sum their modal denominators. But in general, rank `m` and rank `n` come with different

$$
M_r^{(m)},\ M_\theta^{(m)},\ M_\zeta^{(m)},
\qquad \text{and hence different } V_r^{(m)}, V_\theta^{(m)}, V_\zeta^{(m)}.
$$

So across ranks, the question is really how to choose one useful basis family that respects the dominant ranks without trying to diagonalize every small tail term equally well.

### Diagonalization Strategies Worth Considering

The current shared-modal implementation effectively chooses one basis per axis by diagonalizing a weighted average of the whitened 1-D matrices. That is cheap, but it throws away the fact that one rank is already an exactly diagonalizable Kronecker sum.

The more structured options are the following.

#### 1. Dominant-Rank Anchor

Pick the leading retained rank, or the leading few ranks, and build the modal basis from them alone.

- For one dominant rank, use its exact mass-metric diagonalization.
- Treat the lower ranks as perturbations in that basis.
- Recover the tail by Richardson, Neumann, or low-order Chebyshev correction.

This matches the observed rank decay and is the simplest way to make “important ranks matter more” in a hard sense rather than through heuristic scalar weights.

#### 2. Weighted Average In The Mass Metric

For each axis, whiten the directional operators in a reference mass metric and diagonalize

$$
\overline D_d = \frac{\sum_m \omega_m^{(d)} \widehat D_d^{(m)}}{\sum_m \omega_m^{(d)}},
\qquad d \in \{r,\theta,\zeta\}.
$$

This is the natural generalization of the current shared-basis idea to the `DMM + MDM + MMD` viewpoint.

The key design variable is the weight choice. Better candidates than raw Frobenius size are:

- CP amplitude of the rank,
- complementary-factor size, meaning how much the other two axes amplify this rank,
- or an inverse-aware score measuring how much this rank contributes to the small modal denominators.

#### 3. Weighted Approximate Joint Diagonalization

Instead of diagonalizing only the weighted mean, solve the actual weighted approximate joint diagonalization problem

$$
\min_{Q_d^T Q_d = I}
\sum_m \omega_m^{(d)}
\left\| \operatorname{offdiag}\bigl(Q_d^T \widehat D_d^{(m)} Q_d\bigr) \right\|_F^2.
$$

This is the cleanest shared-basis formulation when several ranks matter. It tells the dominant ranks to be nearly diagonal simultaneously rather than hoping that the eigenvectors of their average are good enough.

#### 4. Split Head And Tail

Use exact per-rank diagonalization on the first few large ranks and compress only the tail into a cheaper shared basis or a rank-1 correction.

In other words,

$$
A_{\mathrm{model}} \approx \sum_{m=1}^{R_{\mathrm{head}}} A^{(m)} + A_{\mathrm{tail}}.
$$

This makes the large ranks exact and forces only the decaying remainder into an approximate shared representation.

#### 5. Shared-Rank Refit Of The Geometric Fields

If we want the `DMM + MDM + MMD` structure to hold literally rank-by-rank, then the three diagonal geometric fields should be fit in a shared CP basis rather than independently.

That means using one common set of rank vectors and one triplet of per-rank amplitudes, so that the `rr`, `theta theta`, and `zeta zeta` pieces are aligned by construction. Then rank `m` is genuinely one separable Kronecker sum, and the “diagonalize one rank exactly, combine across ranks approximately” strategy becomes fully natural.

### Practical Recommendation

If this direction is pursued, the cleanest order of attack is:

1. write the stiffness-style blocks explicitly in the aligned-rank `DMM + MDM + MMD` form,
2. test the dominant-rank-anchor strategy first,
3. then compare it against weighted average and weighted approximate joint diagonalization across ranks,
4. and only after that decide whether a full shared-rank refit of the geometric fields is worth the extra setup complexity.

The reason to start there is that one rank is already structurally solved. The real algorithmic question is only how much of the decaying cross-rank tail must be represented in the common basis before the outer Krylov solver stops caring.

### First Shared-Rank Prototype Result

A first debug-only prototype was added to `scripts/interactive/debug_mass_block_spectra.py` that does the simplest possible shared-rank experiment for `M1` and `M2`.

- Stack the three diagonal metric fields into a single 4-tensor with one component axis.
- Fit one shared CP rank across that stacked tensor.
- Rebuild dense block models for the three component blocks from those shared spatial factors.
- Compare the exact inverse of those aligned-rank dense block models against the existing independently-fitted dense block models.

On the small rank-1 screen, the result was mixed rather than uniformly positive.

- `M1 arr`: slight improvement, from condition about `3.09` to about `3.06`.
- `M1 theta`: meaningful improvement, from about `2.10` to about `1.89`.
- `M1 zeta`: clear regression, from about `1.24` to about `2.77`.
- `M2 r_bulk`: essentially neutral, from about `1.98` to about `2.04`.
- `M2 theta`: severe regression, from about `2.42` to about `12.6`.
- `M2 zeta`: essentially neutral, about `1.65` in both cases.

So the naive shared-rank idea is not enough by itself. Aligning the ranks across components can help some blocks, but it can also force the wrong compromise on others.

The most likely interpretation is that the important issue is not merely “shared rank across all three diagonal fields”, but rather how the shared basis is weighted and which blocks are allowed to dominate that compromise.

That sharpens the next iteration.

- A global unweighted shared-rank fit across all three component fields is too blunt.
- The next plausible variants are a dominant-rank anchor, a weighted shared-rank fit, or a head/tail split where only the leading ranks are aligned and the tail remains componentwise.
- The `M2 theta` failure is especially informative: it shows that some blocks are sensitive enough that they should not be forced into the same shared geometric factors as the others without a weighting rule that reflects their importance.

### Torus-Scaled Shared-Rank Rank-1 Result

The next debug-only variant multiplied in the known toroidal radial scaling before fitting the shared rank-1 correction.

- For `M1`, the shared-rank correction was fit after factoring out the radial baseline `(r, 1/r, r)`.
- For `M2`, the shared-rank correction was fit after factoring out `(1/r, r, 1/r)`.

This was still only a rank-1 shared correction, but the torus-informed scaling changed the picture materially.

Compared against the plain shared-rank rank-1 fit:

- `M1 arr`: essentially unchanged, about `3.06` to `3.09`.
- `M1 theta`: essentially unchanged, about `1.89` to `1.88`.
- `M1 zeta`: improved, about `2.77` down to about `2.45`, but still much worse than the independent rank-1 fit.
- `M2 r_bulk`: regressed slightly, about `2.04` up to about `2.38`.
- `M2 theta`: improved dramatically, about `12.6` down to about `4.08`, but still worse than the independent rank-1 fit at about `2.42`.
- `M2 zeta`: improved clearly, about `1.65` down to about `1.35`, now beating the independent rank-1 fit.

So the torus prior is clearly helping. The bad failures of the naive shared-rank fit were not just telling us that shared ranks are wrong; they were also telling us that the fit was being asked to learn an obvious radial scaling from scratch.

At the same time, this result also says that the torus-scaled shared rank-1 model is not yet a uniformly better replacement for the independent rank-1 fit.

- It helps significantly on `M2 theta` and `M2 zeta`.
- It helps somewhat on `M1 zeta`.
- It is neutral on `M1 arr` and `M1 theta`.
- It still does not beat the existing independent rank-1 fit on the hardest remaining vector-mass blocks.

The updated interpretation is therefore:

- the torus-informed multiplicative baseline is the right direction,
- but the current cheap implementation using only radial scaling `(r, 1/r, r)` and `(1/r, r, 1/r)` is still too crude to settle the production design,
- and the next refinement should use the exact torus coefficient baselines involving both `r` and `R(r, theta)` rather than only the simplified radial powers.