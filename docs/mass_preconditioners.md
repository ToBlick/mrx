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

Rank-1 fits are the practical default. Higher ranks are supported by the tensor
block machinery, but the current production guidance is still to treat rank 1
as the reference choice unless a benchmark shows otherwise.

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

## 4. Final Summary

The final mass-preconditioner picture is simple.

- keep the extracted-space special rows exact through a small Schur solve,
- compress the diagonal mapped coefficient fields rather than the inverse,
- use tensor block inverses only on the regular bulk blocks,
- and keep the optional inner coupled bulk Schur for `k = 1` and `k = 2` as a
  benchmarked option rather than as the default practical choice.
