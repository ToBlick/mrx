# Preconditioning Notes

This note records the current low-rank preconditioning direction for scalar
mass and scalar Hodge-like blocks on donut-like devices.

The main design choice is now fixed:

- use an outer `rt|z` split,
- infer the retained rank at preconditioner build time,
- cap the inferred rank by a small default maximum,
- and never materialize a full dense
  `(n_r n_t n_z) \times (n_r n_t n_z)` matrix during the compression.

## 1. Why `rt|z`

For toroidal or otherwise donut-like geometries, the toroidal direction `z`
is the most natural outer split.  The quadrature weights and the assembled
scalar operators show that the strongest near-separability is typically between
the `r-\theta` plane and the toroidal direction.

This does **not** mean the operator is exactly `A_{rt} \otimes A_z` in
general.  It means that `rt|z` is the best first compression layer, both
geometrically and numerically.

## 2. Compression hierarchy

For scalar weights such as

$$
W^0 = J, \qquad W^3 = 1/J,
$$

the preconditioner build now uses the quadrature-weight tensor itself, not the
fully assembled dense 3D operator.

The intended hierarchy is:

1. reshape the scalar quadrature weight into an `rt|z` matrix,
2. truncate its SVD to a small outer rank,
3. for each retained `rt` factor, compute an `r|t` SVD,
4. truncate those inner factors as well,
5. assemble only the retained 1D weighted mass factors.

This yields a representation of the form

$$
A \approx \sum_{j=1}^{r_{\mathrm{out}}} \sum_{\ell=1}^{r_j}
A_r^{(j,\ell)} \otimes A_\theta^{(j,\ell)} \otimes A_\zeta^{(j)}.
$$

So the final object is a short sum of fully separable 3-way terms.

## 3. Rank selection policy

The split is fixed, but the rank is not.

The current policy is:

- infer ranks at preconditioner build time,
- use cumulative singular-value energy of the chosen matricization,
- select the smallest rank whose discarded energy is below tolerance,
- and cap the automatically chosen rank by a default maximum.

The current default maximum is

$$
r_{\max} = 5.
$$

This applies both to the outer `rt|z` split and to the inner `r|t` splits,
unless explicitly overridden.

In implementation terms, if the singular values are
$\sigma_1 \ge \sigma_2 \ge \dots$, then the discarded energy after keeping
rank `r` is

$$
\frac{\sum_{j > r} \sigma_j^2}{\sum_j \sigma_j^2}.
$$

This is the quantity used to infer the rank in auto mode.

## 4. No dense 3D matrix

The hard constraint is:

- never build a dense full operator of size
  `(n_r n_t n_z) \times (n_r n_t n_z)`
  just to discover the low-rank structure.

The current scalar implementation respects that by compressing the quadrature
weight tensor first and only then assembling the retained 1D factors.

Dense extracted-space matrices are still acceptable in the small debug harness
for verification, but not as part of the build-time compression path.

## 5. Current validated results

The new hierarchical scalar-mass validation in
[scripts/debug_k0_polar_block_fd.py](../scripts/debug_k0_polar_block_fd.py)
compares the compressed apply against the exact extracted sparse mass operator.

On the rotating-ellipse test case with `n=(6,6,6)` and `p=3`:

### `k = 0`

- `dirichlet=False`: outer rank `2`, inner ranks `(2, 2)`, total terms `4`
- `dirichlet=True`: outer rank `2`, inner ranks `(2, 2)`, total terms `4`
- outer tail energy is zero in both cases
- extracted-space apply matches to roundoff, with relative errors about
  `2.3e-14`

So for this case the hierarchical factorization is effectively exact.

### `k = 3`

- `dirichlet=False`: outer rank `2`, inner ranks `(3, 5)`, total terms `8`
- `dirichlet=True`: outer rank `2`, inner ranks `(3, 5)`, total terms `8`
- outer tail energy is about `1.1e-9`
- extracted-space apply error is about `3e-4` in relative norm

So `k = 3` is still strongly compressible, but only approximately so at the
current truncation.

## 6. Interpretation

The main structural picture is now:

- `W^0 = J` on the rotating ellipse closes exactly at low hierarchical rank in
  the tested case,
- `W^3 = 1/J` does not close exactly at the same low rank, but is still well
  approximated by a short hierarchical expansion,
- the `rt|z` outer split remains the right production-facing default for
  toroidal devices,
- and the remaining question is how well the same strategy extends to the
  vector-valued `W^1 = J g^{-1}` and `W^2 = g/J` blocks.

## 7. Production direction

The likely production API should distinguish:

- the split policy, which is fixed,
- the rank policy, which can be automatic or user-forced,
- and the rank cap, which should have a conservative default.

One reasonable shape is:

```python
set_preconditioner_rank(rank=None, rank_max=5, tol=...)
```

with the meaning:

- `rank=None`: infer the smallest acceptable rank,
- `rank=k`: force rank `k`,
- `rank_max=5`: cap auto-selected rank at `5`.

For the hierarchical scalar case this should apply to both the outer and inner
SVD stages.

## 8. Next steps

- Benchmark the new hierarchical scalar preconditioner directly against the
  current production scalar mass preconditioners:
  `jacobi` and production `kronecker`.
- Extend the same build-time hierarchical compression to the production scalar
  mass preconditioner path if those comparisons are favorable.
- Decide whether the `k=3` default tolerance should target exactness on simple
  maps or just a stable preconditioned spectrum.
- Investigate analogous low-rank build paths for `W^1` and `W^2`.

The current debug harness now includes a direct comparison cell in
[scripts/debug_k0_polar_block_fd.py](../scripts/debug_k0_polar_block_fd.py)
that solves scalar mass systems with CG and swaps only the preconditioner:

- `jacobi`
- production `kronecker`
- hierarchical `rt|z` then `r|t`

That last option turned out to be only a compressed operator apply, not an
inverse-like preconditioner, so it is not the right comparison target.

The new debug prototype therefore tests a denser class that is closer to what a
production extension could actually store:

- choose a common dense `z` basis,
- transform the raw scalar mass matrix into that basis,
- drop the off-diagonal `z` couplings,
- and store one dense `rt` inverse for each retained `z` block.

This is exposed in the same debug script as the `rt-zblock` scalar mass
preconditioner benchmark.  It is still a debug-only construction, but it is a
genuine linear inverse-like preconditioner rather than a forward operator fit.

The intended acceptance metric is iteration count first, not standalone
operator-fit error.