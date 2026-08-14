# Higher-Form Hodge Tensor Plan

This note records the implementation plan for extending the tensor-Hodge ideas
beyond the scalar `k = 0` case.

## Goal

Build higher-form Hodge/Laplacian tensor preconditioners by following the same
design used for the scalar `k = 0` route:

- derive a regular-space tensor model first,
- keep a shared tensor-product mass basis,
- absorb directional geometry into the active stiffness terms,
- and only then wrap the extracted-space surgery or saddle structure around the
  resulting bulk model.

## Target Operators

The stored stiffness-side operators in the current code are:

- `k = 1`: `curl_curl`,
- `k = 2`: `div_div`,
- `k = 3`: no separate direct stiffness block; the upper action is carried by
  the saddle-point Schur structure through the lower-form mass inverse.

So the practical extension order is:

1. `k = 2` regular-space `div_div`,
2. `k = 1` regular-space `curl_curl`,
3. `k = 3` through the saddle/Hodge-dual structure built on the previous two
   ingredients rather than through an unrelated direct tensor block.

At this point the regular-space forward-model slice exists for both `k = 2`
and `k = 1`, but only `k = 1` has been numerically validated in this round.

## Modeling Principle

For each degree, the intended tensor model should have this shape:

- one shared component-wise mass basis,
- one Kronecker-sum or small block-Kronecker model for the regular-space bulk,
- directional geometry absorbed into the active one-dimensional stiffness
  factors,
- no forced collapse to a single shared three-dimensional geometric field.

The scalar `k = 0` route is now the template for this structure.

The important correction for the inverse / preconditioner side is that the FD
reference should be the mass triple, not the first separated stiffness term.

That is:

- mass-side tensor inverses fit the usual two-term Lynch template
  `M⊗M⊗M + A⊗A⊗A`,
- but stiffness-side FD should instead diagonalize with respect to the shared
  one-dimensional mass operators,
- and then assemble the modal denominator from the active stiffness channels in
  that mass-orthonormal basis.

Equivalently, the current generic rank-`2` helper of the form
`term0 + term1 -> 1 + lam_r lam_t lam_z` is too specific for stiffness.
It matches the mass-side builder, but not the scalar or higher-form stiffness
operators.

The intended modal structure is instead:

- scalar `k = 0` stiffness:
  `K_r⊗M_t⊗M_z + M_r⊗K_t⊗M_z + M_r⊗M_t⊗K_z`,
  so the modal denominator is additive in the three one-dimensional stiffness
  eigenvalues,
- `k = 1` diagonal curl-curl blocks:
  still referenced to mass in each axis, with the denominator assembled from
  the two active stiffness directions of that component block,
- `k = 2` div-div:
  again referenced to mass in each axis, with zero modal entries handled in the
  modal denominator rather than by switching to an unrelated dense block model.

So the current higher-form stiffness path should move toward a dedicated
mass-referenced FD builder rather than toward stronger special-casing of the
existing mass-style rank-`2` helper.

### Reference Table

For the diagonal block builders, the one-dimensional FD references should be
the mass matrices on the actual coefficient space of that block. The active
one-dimensional operators are the corresponding pulled-back stiffness factors
`K = G^T M^(d) G` on the axes that are differentiated in that block.

| Degree / block | Reference masses | Active 1D operators |
| --- | --- | --- |
| `k = 0` scalar | `M_r, M_t, M_z` | `K_r, K_t, K_z` |
| `k = 1` `r -> r` | `M_r^(d), M_t, M_z` | `K_t, K_z` |
| `k = 1` `theta -> theta` | `M_r, M_t^(d), M_z` | `K_r, K_z` |
| `k = 1` `zeta -> zeta` | `M_r, M_t, M_z^(d)` | `K_r, K_t` |
| `k = 2` `r -> r` | `M_r, M_t^(d), M_z^(d)` | `K_r` |
| `k = 2` `theta -> theta` | `M_r^(d), M_t, M_z^(d)` | `K_t` |
| `k = 2` `zeta -> zeta` | `M_r^(d), M_t^(d), M_z` | `K_z` |

Here `M^(d)` means the mass matrix on the derivative-spline space itself,
whereas `K` is the pulled-back stiffness matrix on the undifferentiated spline
space. So `K` already contains a derivative-space mass internally, while
`M^(d)` acts directly on coefficients that already live in the differentiated
space.

## `k = 2` First Slice

`k = 2` is the cleanest next target because

$$
K_2 = G_2^T M_3 G_2,
$$

and `M_3` carries the scalar mapped weight `1 / J`.

The first implementation step is therefore to expose the regular-space `k = 2`
component tensor shapes and the scalar quadrature weight tensor that drives the
`div_div` model. That foundation will support a clean tensor-block builder in a
later step.

## Validation Plan

Each higher-form tensor extension should be validated in this order:

1. regular-space modeled bulk forward apply versus the exact regular-space
  forward operator apply,
2. regular-space modeled bulk matrix versus exact regular-space bulk matrix,
3. production apply versus inverse of the assembled modeled bulk matrix,
4. extracted or saddle-wrapped preconditioner versus inverse of its own
   assembled model,
5. only then benchmark against Jacobi and the current production routes.

For `k = 2`, the first benchmark target should be the exact regular-space
`div_div` matvec

$$
v \mapsto G_2^T M_3 G_2 v,
$$

before any extracted-space or saddle wrapping is introduced.

That regular/extracted forward-model stage is now far enough along to guide
implementation priority:

- for rotating geometry and extracted `dbc` bulk, the current rank-`2`
  `k = 2` model gives CP relative error about `1.4e-3` and sampled forward
  relative error around `7e-3`,
- which is substantially tighter than the corresponding `k = 1` rank-`2`
  extracted bulk diagnostic,
- and is consistent with the expectation that `k = 2` is the cleaner place to
  invest in the next FD-style inverse/preconditioner step.

For `k = 1`, the corresponding regular-space check

$$
v \mapsto G_1^T M_2 G_1 v
$$

has now been validated for the identity map at rank `1`, with relative vector
and dense-matrix errors at about machine precision.

For rotating geometry, the extracted-space bulk forward model at rank `1`
shows nontrivial error. That is currently interpreted as rank-`1`
approximation error in the mapped diagonal `k = 2` mass channels rather than a
structural bug in the tensor curl-curl assembly.

## `k = 1` Block Diagnostic Update

The current extracted `k = 1` Dirichlet smoke case now has a more precise local
diagnostic.

For the restricted diagonal bulk blocks:

- the `r -> r` block is genuinely semidefinite in the tested case,
- the `theta -> theta` and `zeta -> zeta` blocks are numerically invertible,
- but all three rank-`2` surrogate blocks fail the current FD prerequisite
  check because the separated axis factors need not be SPD even when the
  assembled surrogate block itself is well behaved.

This matters for interpretation:

- the outer singular CG nullspace handling is not the source of the current
  rank-`2` failure,
- the main defect is that the present FD builder is diagonalizing the wrong
  template for stiffness,
- and the `r -> r` block additionally needs pseudoinverse-style handling in the
  correct mass-referenced modal basis because it carries a genuine local kernel.

The current dense inverse / pseudoinverse fallbacks should therefore be treated
only as temporary development scaffolding, not as the target higher-form
stiffness design.

## Current Recommendation

Do not widen to new polynomial wrappers or dense debug-only production paths
while building these higher-form models. The next work should stay focused on
the stiffness-side tensor models themselves.

Current recommendation:

1. keep `k = 1` and `k = 2` forward models rank-general for diagnostics,
2. replace the current mass-style stiffness FD assumptions by a dedicated
  mass-referenced stiffness FD builder,
3. build `k = 1` diagonal-block inverses in that shared mass basis, with modal
  pseudoinverse handling where the local block has genuine zero modes,
4. treat the present dense block inverse / pseudoinverse fallbacks as temporary
  debug scaffolding only,
5. once the mass-referenced FD builder exists, revisit `k = 2` as the cleaner
  full higher-form stiffness target and then return to the mixed Hodge path.