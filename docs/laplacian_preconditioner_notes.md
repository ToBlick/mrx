# Laplacian Preconditioner Notes

This note records the current production picture for the Hodge-Laplacian
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

For `k >= 1`, the Hodge-Laplacian solve path is structurally saddle-point based
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
- so the scalar tensor-Hodge path should be viewed as a strong practical
  complement preconditioner rather than as a near-direct inverse.

## 6. Final Summary

The final Laplacian preconditioning picture is:

- unshifted problems use nullspace deflation,
- shifted problems use complement preconditioning plus explicit harmonic coarse
  correction,
- scalar `k = 0` uses the assembled tensor-Hodge Schur-plus-bulk route,
- higher-degree Laplacian solves rely on the settled mass preconditioners as
  their mass-side building blocks.
