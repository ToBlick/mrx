# Mass Preconditioners

This note records the current production picture for the mass preconditioners in
`mrx`. It is intentionally short and only describes the active design.

## 1. Shared Design

The production tensor route does not try to approximate the inverse of the full
extracted mass matrix directly. Instead it:

- keeps the extracted-space surgery rows exact through a small dense Schur
  complement,
- approximates only the bulk tensor blocks,
- fits the diagonal mapped coefficient fields on the quadrature grid,
- and builds tensor-diagonal block inverses from those fitted fields.

The active diagonal coefficient fields are:

- `k = 0`: `J`,
- `k = 1`: `J g^{rr}`, `J g^{theta theta}`, `J g^{zeta zeta}`,
- `k = 2`: `g_rr / J`, `g_theta theta / J`, `g_zeta zeta / J`,
- `k = 3`: `1 / J`.

Higher ranks are supported by the tensor block machinery. Recent solve and
forward-model checks now show that rank `2` is the practical default across
all four mass blocks on the tested rotating-ellipse family.

## 2. Degree-by-Degree Structure

### `k = 0`

`k = 0` is the scalar surgery case.

- The extracted matrix is split into a small core block and one scalar bulk
  tensor block.
- The core is handled by a dense Schur solve.
- The bulk is handled by a scalar tensor inverse built from a fit of `J`.

So the active route is:

- outer scalar core Schur,
- scalar tensor bulk inverse.

### `k = 1`

`k = 1` uses a surgery-first extracted ordering.

- The extracted `theta` and `zeta` surgery rows form the outer Schur block.
- The bulk is split into `r`, `theta_bulk`, and `zeta_bulk` tensor blocks.
- The tensor route can optionally treat the bulk by an additional coupled inner
  Schur, but that coupling is not required for the outer surgery model.

So the active route is:

- outer surgery Schur,
- tensor bulk blocks for `r`, `theta_bulk`, and `zeta_bulk`,
- optional inner bulk Schur coupling.

The assembly-time toggle is:

- `cp_kwargs["k1_inner_schur"] = True` for the coupled bulk model,
- `cp_kwargs["k1_inner_schur"] = False` for pure diagonal tensor bulk blocks.

### `k = 2`

`k = 2` has the same overall philosophy with a smaller surgery block.

- The extracted `r` surgery rows form the outer Schur block.
- The bulk is split into `r_bulk`, `theta`, and `zeta` tensor blocks.
- The tensor route can optionally treat the bulk by an additional coupled inner
  Schur, but the outer surgery split remains the dominant structure.

So the active route is:

- outer surgery Schur,
- tensor bulk blocks for `r_bulk`, `theta`, and `zeta`,
- optional inner bulk Schur coupling.

The assembly-time toggle is:

- `cp_kwargs["k2_inner_schur"] = True` for the coupled bulk model,
- `cp_kwargs["k2_inner_schur"] = False` for pure diagonal tensor bulk blocks.

### `k = 3`

`k = 3` is the second scalar case.

- There is no surgery split.
- The extracted matrix is treated as one scalar tensor block.
- The inverse apply uses the tensor model built from a fit of `1 / J`.

So the active route is:

- direct scalar tensor inverse,
- no surgery Schur.

## 3. Baselines And Practical Winners

The useful baselines remain:

- whole-matrix Jacobi,
- whole-matrix Chebyshev built on Jacobi.

Those are still useful for comparison, but they are not the preferred
production routes.

The current benchmark picture on the rotating-ellipse family is:

- `k = 0` mass: scalar Schur plus tensor bulk is decisively better than whole
  Jacobi and Jacobi-Chebyshev,
- `k = 3` mass: direct scalar tensor inversion is decisively better than whole
  Jacobi and Jacobi-Chebyshev,
- `k = 1` and `k = 2` mass: the outer surgery Schur plus diagonal tensor bulk
  blocks already delivers most of the gain,
- the optional inner bulk Schur for `k = 1` and `k = 2` reduces iteration
  counts only slightly on the tested family, but increases runtime
  substantially,
- wrapping Chebyshev around an already strong tensor route often lowers
  iteration counts but usually does not improve wall-clock time.

So the current practical recommendation is:

- `k = 0`: use the scalar Schur-plus-tensor route,
- `k = 1`: prefer `k1_inner_schur = False` unless a harder case shows a clear
  robustness benefit from the coupled bulk model,
- `k = 2`: prefer `k2_inner_schur = False` unless a harder case shows a clear
  robustness benefit from the coupled bulk model,
- `k = 3`: use the direct scalar tensor route.

## 4. Forward-Model Diagnostics

The recent small-case forward-model checks help separate model quality from
solve-path effects.

- `k = 0` mass is a good rank-1 tensor model on the tested mapped case:
  about `1.6%` full extracted Frobenius error and about `4.7%` bulk-only.
- `k = 1` mass is a weak rank-1 tensor model on the same case: about `24%`
  Frobenius error both on the full extracted operator and on the bulk-only
  restriction. So this is a bulk-model issue, not a surgery artifact.
- `k = 2` mass is moderate at rank `1`: about `5.3%` Frobenius error, again
  with bulk-only error at essentially the same level.
- `k = 3` mass is also moderate at rank `1`, with about `5.5%` Frobenius
  error.

So the current rank-1 model-quality ordering is:

- good: `k = 0`,
- moderate: `k = 2`, `k = 3`,
- bad: `k = 1`.

Those rank-1 diagnostics turned out to be directionally correct but too
conservative about useful production ranks. The later higher-rank checks gave a
cleaner picture:

- `k = 0` mass is effectively a rank-2 geometry on the tested family. Forward
  error drops to near machine precision at rank `2`, and the solve count drops
  from about `11` iterations to about `3`.
- `k = 1` mass improves strongly from rank `1` to rank `2`, with a smaller
  further gain at rank `3`. On `ns = (8, 16, 8)`, the solve count dropped from
  about `28` to `14` to `13`.
- `k = 2` mass shows the same pattern, with the main gain at rank `2` and a
  smaller extra gain at rank `3`. On the same case, the solve count dropped
  from about `26` to `14` to about `12.5`.
- `k = 3` mass also benefits strongly from rank `2`, but shows no practical
  gain from rank `3`. On the same case, the solve count dropped from about
  `11` to `6`, then stayed there.

So the practical higher-rank conclusion is:

- rank `2` is a good default for all mass blocks,
- rank `3` is only a plausible extra option for `k = 1` or `k = 2`,
- and rank `2` already captures essentially all of the useful gain for `k = 0`
  and `k = 3`.

The current production default follows that recommendation in the eager
operator-assembly path: the mass blocks are assembled with per-degree tensor
ranks `k0 = k1 = k2 = k3 = 2`, while the scalar stiffness/Hodge fallback rank
remains at `1`.

## 5. Final Summary

The final mass-preconditioner picture is simple.

- keep the extracted-space special rows exact through a small Schur solve,
- compress the diagonal mapped coefficient fields rather than the inverse,
- use tensor block inverses only on the regular bulk blocks,
- and keep the optional inner coupled bulk Schur for `k = 1` and `k = 2` as a
  benchmarked option rather than as the default practical choice,
- while treating rank `2` as the practical default tensor rank across
  `k = 0, 1, 2, 3` on the tested geometry family,
- with `k = 2` rank `3` left as an exposed tuning option rather than the
  default because the measured extra solve gain has not yet been weighed
  against additional setup cost.
