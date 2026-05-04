# Tensor Preconditioner Primer

This note explains the active tensor-preconditioning idea in `mrx`.

## 1. What The Tensor Route Approximates

The tensor route approximates mapped coefficient fields on tensor-product
quadrature grids, not the inverse of the full extracted operator.

In the ideal scalar tensor case one would have

$$
M = M_r \otimes M_\theta \otimes M_\zeta,
$$

so that the inverse also factors as a tensor product. In the real mapped polar
problems, two things break that exact structure:

- extraction introduces a small non-tensor surgery block,
- geometry turns the bulk operator into a weighted sum of tensor-product terms.

The production design handles those two effects separately.

## 2. Active Production Rule

The production tensor preconditioners now follow one rule across the active
mass and scalar-Laplacian paths:

- keep the small extracted-space non-tensor part exact through a dense Schur
  complement,
- approximate only the bulk tensor blocks,
- sample the relevant diagonal mapped coefficient field on the quadrature grid,
- fit that field by a short CP decomposition,
- and build tensor block inverses from the fitted one-dimensional factors.

So the tensor route is a coefficient-first compression wrapped around the true
extracted-space block structure.

## 3. Which Coefficients Are Used

Only the diagonal mapped coefficient fields are modeled in the production tensor
path.

- mass `k = 0`: `J`,
- mass `k = 1`: `J g^{rr}`, `J g^{theta theta}`, `J g^{zeta zeta}`,
- mass `k = 2`: `g_rr / J`, `g_theta theta / J`, `g_zeta zeta / J`,
- mass `k = 3`: `1 / J`,
- scalar Laplacian / Hodge `k = 0`: the scalar surgery-plus-bulk tensor Hodge
  model used by the active `kind="tensor"` route.

This is deliberate. The tensor route is not trying to represent the full mapped
operator entrywise. It is trying to keep the dominant diagonal tensor structure
cheap and robust.

## 4. Current Role Of Schur Structure

Schur complements are part of the final design, not a temporary workaround.

- `k = 0` mass uses a scalar core-plus-bulk Schur split,
- `k = 1` mass uses an outer surgery Schur and optional inner bulk coupling,
- `k = 2` mass uses an outer surgery Schur and optional inner bulk coupling,
- `k = 3` mass is direct and has no Schur split,
- scalar `k = 0` Laplacian/Hodge uses the assembled surgery-plus-bulk tensor
  Hodge model.

So the mature tensor route is not “one tensor inverse for the whole matrix”. It
is “exact small Schur structure plus tensor bulk blocks”.

## 5. What The Benchmarks Say

The final benchmark picture is now clear.

- In the scalar mass cases `k = 0` and `k = 3`, the tensor routes are both
  strong in iterations and strong in runtime.
- In the scalar Laplacian `k = 0` case, the tensor route is also clearly useful,
  although not as close to an exact inverse as scalar mass.
- In the vector mass cases `k = 1` and `k = 2`, the outer surgery Schur plus
  diagonal tensor bulk blocks already delivers most of the runtime benefit.
- The optional inner coupled bulk Schur for `k = 1` and `k = 2` lowers
  iterations somewhat on the tested family, but its cost is too high to win in
  wall-clock time.

So the active practical default is:

- keep the tensor route,
- keep the exact outer Schur,
- and prefer the cheaper diagonal tensor bulk model for `k = 1` and `k = 2`
  unless a harder geometry shows a clear need for the coupled bulk option.

## 6. Final Takeaway

The tensor route is now settled as a production building block for mass-like
operators because it respects the real extracted-space structure.

The key idea is not “invert a Kronecker product”. The key idea is:

- exact Schur treatment of the small irregular part,
- tensor compression of the regular bulk coefficient fields,
- and cheap one-dimensional factor work instead of dense three-dimensional
  inverse storage.
