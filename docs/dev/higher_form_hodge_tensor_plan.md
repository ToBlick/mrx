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

## Modeling Principle

For each degree, the intended tensor model should have this shape:

- one shared component-wise mass basis,
- one Kronecker-sum or small block-Kronecker model for the regular-space bulk,
- directional geometry absorbed into the active one-dimensional stiffness
  factors,
- no forced collapse to a single shared three-dimensional geometric field.

The scalar `k = 0` route is now the template for this structure.

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

## Current Recommendation

Do not widen to new polynomial wrappers or dense debug-only production paths
while building these higher-form models. The next work should stay focused on
the regular-space tensor models themselves, starting with `k = 2`.