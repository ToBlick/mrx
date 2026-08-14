# Laplacian Preconditioner Notes

This note records the current production picture for the Laplacian
preconditioners, with emphasis on the scalar `k = 0` route that is presently
the mature tensor-based path.

## 1. General Rule

The harmonic space is handled differently in the singular and shifted problems.

For `\varepsilon = 0`, the correct operation is the pseudoinverse action on the
range of the operator, implemented by nullspace deflation.

For `\varepsilon > 0`, deflation is not the correct shifted inverse. The
harmonic space instead needs an explicit coarse correction with the required
`1 / \varepsilon` scaling.

So the settled policy is:

- unshifted solve: deflate the nullspace,
- shifted solve: use a complement preconditioner plus explicit harmonic coarse
  correction when the harmonic vector is available.

## 2. Settled Scalar `k = 0` Policy

The active tensor-Laplacian route is the scalar `k = 0` Hodge preconditioner.

- The operator is split into a small scalar surgery block and a scalar bulk
  block.
- The surgery part is handled by a dense Schur solve.
- The bulk is handled by the assembled tensor Hodge model.

So the active `kind="tensor"` scalar route is:

- scalar core Schur,
- scalar tensor bulk inverse,
- explicit harmonic handling outside that complement solve when needed.

This route is the production tensor building block for scalar Laplacian solves.

## 3. Shifted Scalar Behavior

For free `k = 0` shifted problems, the intended preconditioner is:

- tensor complement preconditioner on the orthogonal complement,
- exact harmonic coarse correction with scale `1 / \varepsilon`.

When the harmonic vector is not yet known, the code stays on a conservative
robust path until that vector has been constructed. In particular, inverse
iteration does not pretend that the shifted problem should be deflated.

## 4. Practical Policy For Other Degrees

For `k >= 1`, the Laplacian solve path is structurally saddle-point based
and depends on the corresponding mass preconditioners in its lower or inner mass
slots.

The practical production guidance is therefore:

- use the scalar tensor-Hodge route directly only for `k = 0`,
- use the mass preconditioners described in
  [docs/mass_preconditioners.md](mass_preconditioners.md) for the mass blocks
  that appear inside the higher-degree saddle solves,
- keep nullspace inverse iteration on Richardson-style robust paths rather than
  on Chebyshev-tuned paths.

The tensor mass routes are mature. The scalar tensor-Hodge route is mature.
Those are the active preconditioning building blocks on the Laplacian side.

## 5. Current Findings

The benchmark and validation picture is:

- scalar `k = 0` tensor Hodge is clearly better than whole-matrix Jacobi and
  Jacobi-Chebyshev on the tested rotating-ellipse family,
- but the scalar Laplacian tensor route is still a less exact model than the
  corresponding scalar mass tensor routes,
- and the recent forward-model diagnostics show that this is a bulk-model
  issue, not a Schur-routing issue: on the small mapped test case
  `ns = (4, 8, 4)`, `p = 3`, the rank-1 extracted scalar `k = 0` stiffness
  model has about `33%` Frobenius error, while the extracted bulk-only error is
  even worse at about `45%`,
- so the scalar tensor-Hodge path should be viewed as a strong practical
  complement preconditioner rather than as a near-direct inverse.

The important new comparison against the mass side is that higher rank does not
fix this scalar stiffness weakness in the same way it fixes the mass blocks.

- For the mass matrices, moving from rank `1` to rank `2` gives large forward
  and solve improvements across all degrees, which means the underlying issue
  is mostly that the mass-side geometric coefficient fields are not well
  represented by rank `1`.
- For scalar `k = 0` stiffness, the multirank path did not show the same
  behavior: improving the CP fit of the current shared surrogate field did not
  produce a better assembled bulk operator.

So the current scalar stiffness limitation is not primarily “rank too small”.
It is that the present multirank fit is aimed at the wrong object. The active
rank-`r` builder fits a shared proxy field and then reconstructs the three
directional operator terms from that proxy, but the actual scalar stiffness
operator is a sum of directional Kronecker terms,

$$
K_r \otimes M_t \otimes M_z
+
M_r \otimes K_t \otimes M_z
+
M_r \otimes M_t \otimes K_z,
$$

with different directional metric tensors. Better low-rank fit of the proxy
field does not by itself imply better approximation of that operator sum. So
the next stiffness-side improvement should change the fit target to an
operator-aware joint fit, rather than just increasing the rank inside the
current proxy-field construction.

The design note for that next step is in
[docs/operator_aware_scalar_stiffness_fit.md](/scratch/tblickhan/mrx/docs/operator_aware_scalar_stiffness_fit.md).

This is why the production default can now safely diverge between mass and
stiffness:

- mass defaults to per-degree tensor rank `2`,
- scalar stiffness/Hodge still keeps its current rank-`1` fallback,
- and raising the stiffness rank inside the present proxy-field fit is not the
  preferred next step.

The first operator-aware replacement check supports that diagnosis. On the same
small rotating-ellipse case where the old extracted scalar stiffness model was
about `33%` wrong in Frobenius norm, the new rank-`2` operator-aware bulk model
reduced the extracted Frobenius error to about `11.7%`, with sampled forward
error around `11.3%`. So changing the fit target helps materially, even though
the scalar stiffness model is still not yet as accurate as the mature mass-side
tensor models.

The follow-up bulk-only sweep showed that the remaining miss is still primarily
in the bulk tensor approximation rather than in the surgery coupling. On
`ns = (4, 8, 4)`, `p = 3`, the new operator-aware model gave full-extracted vs
bulk-only Frobenius errors of about `18.5%` vs `25.3%` at rank `1`,
`11.7%` vs `16.0%` at rank `2`, and `10.7%` vs `14.6%` at rank `4`. So the
exact core rows are helping rather than hurting, and the main remaining work is
still to improve the bulk scalar stiffness ansatz.

The first actual solve benchmark refined that conclusion further. On the
Dirichlet scalar Laplace solve at `ns = (6, 8, 4)`, `p = 3`, the new
operator-aware builder did not yet produce a better preconditioner as rank
increased: rank `1` took about `25` iterations and `6.68 ms`, while rank `2`
and rank `4` rose to about `30.5` / `7.92 ms` and `29.5` / `7.64 ms`.

The larger benchmark removes any doubt that this is a real inverse-side
failure rather than a small-case fluctuation. On the same Dirichlet
rotating-ellipse family at `ns = (16, 32, 8)`, `p = 3`, the scalar tensor-Hodge
route gave about `58.8` iterations / `287.8 ms` at rank `1`, degraded to about
`107.2` / `518.6 ms` at rank `2`, and then failed outright at rank `4` by
hitting the `1000`-iteration cap, even though the CP fit error kept improving
from about `3.65e-1` to `7.59e-2` to `1.08e-2`.

That large-case result is the clearest current stiffness-side diagnostic.
Better low-rank fit of the directional coefficient fields is no longer the
active blocker. The remaining blocker is the higher-rank scalar bulk inverse,
specifically the shared-basis/modal-denominator approximation used to invert
the operator-aware sum of directional Kronecker terms.

So the current scalar-stiffness picture is now:

- changing the fit target helped materially at the forward-model level,
- the remaining error is still mainly in the bulk model,
- but higher-rank operator-aware fitting does not yet improve the actual solve
  path and can become dramatically worse on larger cases,
- so the production default should remain the conservative scalar rank-`1`
  route until the inverse side is improved further.

## 6. Final Summary

The final Laplacian preconditioning picture is:

- unshifted problems use nullspace deflation,
- shifted problems use complement preconditioning plus explicit harmonic coarse
  correction,
- scalar `k = 0` uses the assembled tensor-Hodge Schur-plus-bulk route,
- higher-degree Laplacian solves rely on the settled mass preconditioners as
  their mass-side building blocks.
