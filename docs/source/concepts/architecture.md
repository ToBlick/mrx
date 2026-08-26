# Architecture

MRX discretises the de Rham complex

```
V0 --grad--> V1 --curl--> V2 --div--> V3
```

with tensor-product B-splines on the logical cube `[0,1]^3` in coordinates
`(r, theta, zeta)`, mapped to the physical domain by `F`. This page names the
objects and the order in which they are built. Assembly detail is in
[assembly.md](assembly.md), solvers and preconditioners in
[preconditioning.md](preconditioning.md), the polar axis in [polar.md](polar.md).

## 1. Spaces

### 1D bases

`SplineBasis(n, p, type)` in `mrx/spline_bases.py` is a 1D B-spline basis of
`n` functions and degree `p`. `type` is `"clamped"`, `"periodic"`, or
`"constant"`. `DerivativeSpline(s)` is the basis that contains the derivatives
of `s`: `n-1` functions of degree `p-1` on a clamped axis, `n` on a periodic
axis. `SplineBasis.evaluate_local(x)` returns the `p+1` nonzero values at `x`
and their indices; every evaluation of a field goes through it.

### k-forms

`DifferentialForm(k, ns, ps, types)` in `mrx/differential_forms.py` holds the
three 1D bases `Λ[a]` and their derivative bases `dΛ[a]`. The basis of `V^k`
is a product of one 1D function per axis, differentiated on the axes the
degree prescribes:

| k | components | axis bases per component |
|---|---|---|
| 0 | 1 | `(Λr, Λt, Λz)` |
| 1 | 3 | `(dΛr, Λt, Λz)`, `(Λr, dΛt, Λz)`, `(Λr, Λt, dΛz)` |
| 2 | 3 | `(Λr, dΛt, dΛz)`, `(dΛr, Λt, dΛz)`, `(dΛr, dΛt, Λz)` |
| 3 | 1 | `(dΛr, dΛt, dΛz)` |

A coefficient vector is the concatenation of the components; inside a
component the index is the C-order ravel over `shape[c]`. `grad`, `curl`, and
`div` of a basis function are exact combinations of the next space's basis
functions, so the discrete complex is exact.

`DiscreteFunction(dof, Λ, E)` pairs coefficients with a form and an extraction
operator (section 3). `DifferentialForm.raw_blocks` splits the raw coefficient
vector into per-component tensors and `DifferentialForm.contract` evaluates
the field at a point with one `(p+1)^3` window per component.

### Quadrature

`QuadratureRule(form, q)` in `mrx/quadrature.py` is the tensor product of
composite Gauss rules with `q` points per knot span (`composite_quad`). `q` is a
required argument of `DeRhamSequence`; every production entry point passes
`q = p + 1`. See [assembly.md](assembly.md) for why.

## 2. Operators from the tensor structure

### Mass matrices

The mass matrix of `V^k` is

```
(M_k)_IJ = ∫ Λ_I · W_k · Λ_J dx
```

with the metric weight built from the map Jacobian `DF` and `J = det DF`:

| k | `W_k` |
|---|---|
| 0 | `J` |
| 1 | `J g^{-1}` with `g^{-1} = DF^{-1} DF^{-T}` |
| 2 | `g / J` with `g = DF^T DF` |
| 3 | `1 / J` |

The mass matrices are the only operators that need quadrature and they are
never stored. `mass_core_apply(seq, operators, k)` in `mrx/operators.py`
returns the matrix-free apply on the raw tensor-product space; it is the
sum-factorised kernel of `mrx/local_assembly.py`.

### Incidence, derivative, and stiffness

The exterior derivative on coefficients is the topological incidence
`G_k` with entries in `{-1, 0, +1}`: coefficient differences along one axis.
`_MatrixFreeIncidence` in `mrx/operators.py` applies it as a difference
stencil; `assemble_incidence_operators(seq)` builds `g0`, `g1`, `g2` and their
transposes. No geometry enters.

Everything else is a composition of applies and is never materialised:

| operator | apply |
|---|---|
| weak derivative `D_k = M_{k+1} G_k` | `apply_derivative_matrix`: `E_out M_{k+1} G_k E_in^T v` |
| stiffness `K_k = G_k^T M_{k+1} G_k` | `apply_stiffness`: `E G_k^T M_{k+1} G_k E^T v` |
| strong derivative | `apply_incidence_matrix`: `G_k` with the polar corrections of [polar.md](polar.md) |
| projections `P_{k->l}` | `apply_projection_matrix` |

## 3. Extraction

Assembly runs on the unconstrained tensor-product basis. An extraction
operator `E_k` of shape `(n_k, n_k_raw)` maps it onto the conforming space:

```
Λ_I = Σ_J (E_k)_IJ Λ_raw_J,      A = E_k A_raw E_k^T
```

`MatrixFreeExtraction` in `mrx/extraction_operators.py` stores the nonzeros
as `(rows, cols, vals)` and applies `E` and `E^T` as one gather and one
segment sum. Two builders produce it:

- `BoundaryOperator(Λ, types)`: per-axis `'dirichlet'` (drop the boundary
  functions), `'left'`, `'right'`, or `'none'`. Used when `polar=False`.
- `PolarExtractionOperator(Λ, xi, zero_bc)`: fuses the ring-0 and ring-1
  radial functions of every `zeta` slice into three axis functions with the
  weights `xi` from `get_xi`. Used when `polar=True`. Dirichlet at `r = 1`
  is `zero_bc=True`.

A `DeRhamSequence` holds three extractions per degree, each with its
transpose: `e{k}` (periodic and polar only), `e{k}_dbc` (also drops the
`r = 1` functions), and `e{k}_bc` (the functions `e{k}_dbc` dropped, from
`bc_extraction_op`). Sizes are `n{k}`, `n{k}_dbc`, `n{k}_bc`. Every apply and
solve takes `dirichlet=True|False` and picks the pair.

## 4. k-form Laplacians

The Hodge Laplacian of degree `k` is

```
L_k = K_k + D_{k-1} M_{k-1}^{-1} D_{k-1}^T
```

with `K_3 = 0` and `D_{-1} = 0`. `apply_hodge_laplacian` applies it; the
inverse mass in the second term is a solve. The solves in `mrx/operators.py`
are:

| k | `apply_inverse_hodge_laplacian` | solver in `mrx/solvers.py` |
|---|---|---|
| 0 | CG on `K_0`, harmonic mode deflated | `solve_singular_cg` |
| 1, 2, 3 | MINRES on the saddle system `[[K_k, D_{k-1}], [D_{k-1}^T, -M_{k-1}]]` | `solve_saddle_point_minres` |

`apply_inverse_shifted_hodge_laplacian` solves `L_k + eps M_k` the same way
and `apply_inverse_mass_plus_eps_laplace_matrix` solves `M_k + eps L_k`. The
`DeRhamSequence` methods of the same names forward to these with the
sequence's own `operators`, `tol`, and `maxiter`. Every solve takes a
`preconditioner` argument; the default `'auto'` is described in
[preconditioning.md](preconditioning.md).

### Harmonic forms

`L_k` has a kernel of dimension given by the Betti numbers passed to
`DeRhamSequence(betti_numbers=...)`; `(1, 1, 0, 0)` is the solid torus. The
kernel vectors live on `SequenceOperators.null_{k}` and `null_{k}_dbc` as
arrays of fixed shape `(n_vectors, n_k)`, zero until computed, so a solve on a
fresh sequence deflates nothing. `mrx/nullspace.py` fills them:
`compute_nullspaces` by a direct Hodge decomposition (needs `b2 = 0`),
`compute_nullspaces_iterative` by shifted inverse iteration for any topology.
Both need mass, incidence, and Laplacian preconditioners assembled first.
Unshifted solves deflate the kernel; shifted solves do not.

## 5. Data model

The split follows JAX: what must not change between traces is a plain Python
object captured by closure; what may change is a pytree.

### Static: `DeRhamSequence`

`DeRhamSequence(ns, ps, q, types, *, polar, tol=None, maxiter=10_000, ...)`
in `mrx/derham_sequence.py` owns the topology: the four `DifferentialForm`
objects `basis_0..basis_3`, the `quad` rule, the extraction operators, the
1D basis tables at quadrature points (`basis_r_jk`, `d_basis_r_jk`, ...,
filled by `evaluate_1d()`), and the solve defaults `tol` (default
`mrx.sqrt_eps()`) and `maxiter`. `polar` is keyword-only; the map is not a
constructor argument. Extra arguments: `polar_order` (0, 1, or 2; see
[polar.md](polar.md)), `polar_ring1` (map-adapted axis weights),
`betti_numbers`, `knots`, `r_scale`.

### Dynamic: `SequenceGeometry`

`SequenceGeometry` in `mrx/geometry.py` is an `eqx.Module` with the map and
three arrays on the quadrature grid: the metric `metric_jkl = DF^T DF` of
shape `(N_q, 3, 3)`, its inverse `metric_inv_jkl` `(N_q, 3, 3)` and
`jacobian_j = det DF` `(N_q,)`. They are built once from `DF` by the
constructors and never recomputed; `DF` itself is not kept, because its only
consumers are the physical-frame pullbacks at load time (`load(frame='phys')`,
`io.load_grid_field(frame='phys')`), which recompute it with
`map_jacobian_at(seq.map, seq.quad.x)`. Everything on the hot path -- the mass
weights `J`, `J G^-1`, `G/J`, `1/J`, the force step's `cross_product_load`,
the lumped preconditioner builds -- reads the stored arrays. Build the
geometry with `SequenceGeometry.from_map(F, seq.quad.x)` (autodiff of `F`
under `jax.lax.map`) or `SequenceGeometry.from_spline_map(spline_map, seq)`
(sum factorisation of the spline coefficients). `seq.set_map(F)` and
`seq.set_spline_map(coefficients)` install it as `seq.geometry` and drop any
Laplacian preconditioner built for the previous geometry.

Maps enter by interpolation. An analytic map is a callable `F(x)`; a map from
data is fitted as three scalar 0-form splines on a map sequence and wrapped as
a `SplineMap` (`mrx/mappings.py`) or a `stellarator_map`. The fit is
`seq.interpolate(f, 0)`: 1D collocation solves on the tensor space followed by
the polar restriction. `mrx/geometry.py` has `greville_interpolate_map` and
`greville_interpolate_stellarator_map`; `mrx/gvec.py` has `build_gvec_map`
and `build_w7x_map` for GVEC and W7-X files (gridded `R, Z` go through
`fit_scalar_spline`). There is no reference mass matrix.

### Dynamic: `SequenceOperators`

`SequenceOperators` in `mrx/operators.py` is an `eqx.Module` holding the
assembled data, every field optional:

- extraction operators `e{k}`, `e{k}_dbc`, `e{k}_bc` and transposes,
- incidence `g0, g1, g2` and transposes; the polar stencils
  `g0_grad_*`, `g1_curl_*`,
- the mass Jacobi diagonals in `mass_preconds`,
- the Laplacian preconditioner diagonals `dd{k}_diaginv`, `dd{k}_diaginv_dbc`
  and the Schur diagonals `schur_diaginv_k{k}`,
- projections `p21, p12, p03, p30`,
- harmonic forms `null_{k}`, `null_{k}_dbc`.

`seq.set_operators(ops)` attaches a bundle; every `apply_*` on the sequence
uses it. The metric-lumping preconditioner payloads are pytrees keyed on the
geometry and stored on the sequence, so a rebuild does not recompile.

## 6. Assembly order

Each builder reads the previous one. `build_sequence(geometry, ns, p)` in
`mrx/geometries.py` is the production recipe:

```python
seq = DeRhamSequence(ns, (p,) * 3, p + 1, ("clamped", "periodic", "periodic"),
                     polar=True, betti_numbers=(1, 1, 0, 0))
seq.evaluate_1d()
seq.set_map(toroid_map(epsilon=1 / 3, R0=1.0))
ops = op.assemble_incidence_operators(seq)
ops = op.assemble_mass_jacobi_preconditioner(seq, ops, ks=(0, 1, 2, 3))
ops = op.assemble_metric_lumping_laplacian_preconditioner(
    seq, ops, ks=(0, 1, 2, 3), dirichlets=(False, True))
op.warm_mass_preconditioner_cache(seq, ops)
seq.set_operators(ops)
```

1. Topology: `DeRhamSequence`, then `evaluate_1d()`.
2. Geometry: `set_map` or `set_spline_map`.
3. Incidence: `assemble_incidence_operators`; builds the polar grad and curl
   stencils when the sequence is polar.
4. Mass preconditioners: `assemble_mass_jacobi_preconditioner`; the
   metric-lumping mass preconditioner builds on first use and
   `warm_mass_preconditioner_cache` forces it before any traced loop.
5. Laplacian preconditioners:
   `assemble_metric_lumping_laplacian_preconditioner`. Needs the mass
   preconditioners, because the weak term of `L_k` is applied through them.
6. Harmonic forms: `compute_nullspaces` or `compute_nullspaces_iterative`,
   after everything above. The projection masses `P_21, P_12, P_03, P_30`
   (helicity diagnostic) need no step: they are matrix-free applies built on
   first use and memoised on the geometry, like the masses.

`seq.build_preconditioners()` runs steps 3 to 5 and verifies every `(k, BC)`
built; `seq.set_map_and_preconditioners(F)` runs steps 2 to 5.
`assemble_all_operators(seq, geometry)` is the same chain plus projections,
and `operators_from_coeffs(seq, coeffs, ks, kinds)` rebuilds geometry and
operators from `SplineMap` coefficients in one differentiable call.
