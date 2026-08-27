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
never stored. `mass_core_apply(seq, k)` in `mrx/operators.py` returns the
matrix-free apply on the raw tensor-product space (`seq.mass_apply[k]`,
built by `set_geometry`); it is the sum-factorised kernel of
`mrx/local_assembly.py`.

### Incidence, derivative, and stiffness

The exterior derivative on coefficients is the topological incidence
`G_k` with entries in `{-1, 0, +1}`: coefficient differences along one axis.
`_MatrixFreeIncidence` in `mrx/operators.py` applies it as a difference
stencil; the sequence builds `g0`, `g1`, `g2` and their transposes in its
constructor. No geometry enters.

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
segment sum. Its builder:

- `PolarExtractionOperator(Λ, xi, zero_bc)`: fuses the ring-0 and ring-1
  radial functions of every `zeta` slice into three axis functions with the
  weights `xi` from `get_xi`. Used when `polar=True`. Dirichlet at `r = 1`
  is `zero_bc=True`.

A `DeRhamSequence` holds three extractions per degree: `E(k)` (periodic
and polar only), `E(k, True)` (also drops the `r = 1` functions), and
`E_bc(k)` (the functions `E(k, True)` dropped, from `bc_extraction_op`);
`.T` is the transpose. Sizes are `n(k)`, `n(k, True)`, `n_bc(k)`. Every
apply and solve takes `dirichlet=True|False` and picks the pair.

## 4. k-form Laplacians

The Hodge Laplacian of degree `k` is

```
L_k = K_k + D_{k-1} M_{k-1}^{-1} D_{k-1}^T
```

with `K_3 = 0` and `D_{-1} = 0`. `apply_laplacian` applies it; the
inverse mass in the second term is a solve. The solves in `mrx/operators.py`
are:

| k | `apply_inverse_laplacian` | solver in `mrx/solvers.py` |
|---|---|---|
| 0 | CG on `K_0`, harmonic mode deflated | `solve_singular_cg` |
| 1, 2, 3 | MINRES on the saddle system `[[K_k, D_{k-1}], [D_{k-1}^T, -M_{k-1}]]` | `solve_saddle_point_minres` |

`apply_inverse_shifted_laplacian` solves `L_k + eps M_k` the same way
and `apply_inverse_mass_plus_eps_laplace_matrix` solves `M_k + eps L_k`. The
`DeRhamSequence` methods of the same names forward to these with the
sequence's own `operators`, `tol`, and `maxiter`. Every solve takes a
`preconditioner` argument; the default `'auto'` is described in
[preconditioning.md](preconditioning.md).

### Harmonic forms

`L_k` has a kernel of dimension given by the Betti numbers passed to
`DeRhamSequence(betti_numbers=...)`; `(1, 1, 0, 0)` is the solid torus. The
kernel vectors live on `SequenceOperators.nullspaces[(k, dirichlet)]` as
arrays of fixed shape `(n_vectors, n_k)`, zero until computed, so a solve on a
fresh bundle deflates nothing. `mrx/nullspace.py` fills them:
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
objects `basis_0..basis_3`, the `quad` rule, the polar weights `xi`, the
extraction operators `e0..e3` (free, Dirichlet, boundary), the incidence
stencils `g0..g2` with the polar grad/curl corrections, the 1D basis tables
at the quadrature points (`basis_r_jk`, `d_basis_r_jk`, ...), the Greville
data, and the solve defaults `tol` (default `mrx.sqrt_eps()`) and `maxiter`.
All of it is built in the constructor. `polar` is keyword-only; the map is
not a constructor argument. Extra arguments: `betti_numbers`, `knots`,
`r_scale`.

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

`SequenceOperators` in `mrx/operators.py` is an `eqx.Module` holding
everything built from a geometry, three dicts keyed `(k, dirichlet)`:

- the metric-lumped mass atoms, `mass_lumping`,
- the metric-lumped Laplacian atoms, `laplacian_lumping`,
- the harmonic forms, `nullspaces`, arrays `(n_vectors, n_k)`.

`build_preconditioners` creates it and installs it as `seq.operators`; the
solves on the sequence read it. The atom payloads are pytrees with one jitted
apply per tree structure, so a rebuild for a new geometry does not
recompile.

## 6. Assembly order

Each builder reads the previous one. `build_sequence(geometry, ns, p)` in
`mrx/geometries.py` is the production recipe:

```python
seq = DeRhamSequence(ns, (p,) * 3, p + 1, ("clamped", "periodic", "periodic"),
                     polar=True, betti_numbers=(1, 1, 0, 0))
seq.set_map(toroid_map(epsilon=1 / 3, R0=1.0))
ops = seq.build_preconditioners()
ops = seq.set_operators(compute_nullspaces(seq, ops))
```

1. Topology: `DeRhamSequence`. Bases, extraction, incidence (with the polar
   grad and curl stencils), 1D tables, Greville data: everything static.
2. Geometry: `set_map` or `set_spline_map`. Installs the metric and builds
   the matrix-free mass and projection applies from it. Drops the operator
   bundle.
3. Preconditioners: `build_preconditioners`, one call:
   `assemble_mass_metric_lumping_preconditioner`, then
   `assemble_metric_lumping_laplacian_preconditioner` (the Laplacian atoms
   need the mass preconditioners, because the weak term of `L_k` is applied
   through them). These are what `kind='auto'` applies everywhere, the
   shift-and-invert nullspace route included.
4. Harmonic forms: `compute_nullspaces` or `compute_nullspaces_iterative`,
   after everything above; they live on the bundle.

Nothing on the bundle is built on first use, and nothing on it survives a
geometry change: after a new `set_map`, run steps 3 and 4 again. That is the
contract for an outer loop over geometries (relaxation inside, the map
outside). `seq.set_map_and_preconditioners(F)` is steps 2 and 3.
