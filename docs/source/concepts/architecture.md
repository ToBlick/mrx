# Architecture

MRX discretises the de Rham complex

```
V0 --grad--> V1 --curl--> V2 --div--> V3
```

with tensor-product B-splines on the logical cube `[0,1]^3` in coordinates
`(r, theta, zeta)`, mapped to the physical domain by `F`. This page names the
objects and the order in which they are built. Assembly detail is in
[mass.md](mass.md), solvers and preconditioners in
[preconditioning.md](preconditioning.md), the polar axis in [polar.md](polar.md).

## 1. Spaces

### 1D bases

`SplineBasis(n, p, type)` in `mrx/spline_bases.py` is a 1D B-spline basis of
`n` functions and degree `p`, `type` `"clamped"` or `"periodic"`.
`DerivativeSpline(s)` is the basis that contains the derivatives of `s`:
`n-1` functions of degree `p-1` on a clamped axis, `n` on a periodic axis.
`SplineBasis.evaluate_local(x)` returns the `p+1` nonzero values at `x` and
their indices; every evaluation of a field goes through it.

### k-forms

`DifferentialForm(k, ns, ps, types)` in `mrx/differential_forms.py` holds the
three 1D bases `Λ[a]` and their derivative bases `dΛ[a]`. The basis of `V^k`
is a product of one 1D function per axis, differentiated on the axes the
degree prescribes (`derivative_axes(c)`, the one place the pattern lives):

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
composite Gauss rules with `q` points per knot span. `q` is a required
argument of `DeRhamSequence`; every production entry point passes
`q = p + 1` ([mass.md](mass.md), section 5).

## 2. Operators from the tensor structure

### Mass matrices

The mass matrices `(M_k)_IJ = ∫ Λ_I · W_k · Λ_J dx`, with the metric weight
`W_k` of [mass.md](mass.md), are the only operators that need quadrature
and they are never stored. `mass_core_apply(seq, k)` in `mrx/operators.py`
returns the matrix-free apply on the raw tensor-product space: the
sum-factorised kernel of `mrx/mass.py` bound to the sequence's plan
(`seq.mass_plan[k]`) and the geometry's weights
(`seq.geometry.mass_weights[k]`, attached by `set_geometry`).

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

### Quadratic operators

The nonlinear terms of the physics are pointwise products of two discrete
forms, and MRX exposes each as an L2 load: the product is evaluated at the
quadrature points from the reference components of its factors and
integrated against the basis of the output space (`M_n^{-1}` of the load
is the L2 projection). `DeRhamSequence.cross_product_load(w, u, n, m, k)`
covers `w × u` for every `(n, m, k)` in `{1, 2}³`, `dot_product_load` the
inner product of two vector forms onto the 0- or 3-forms,
`scalar_product_load` the product of two scalar forms (0 or 3) onto either,
and `scalar_vector_load` a scalar form times a vector form onto the 1- or
2-forms; each has a `_values` twin that takes quadrature values, so a
factor evaluated once can feed several loads.

In reference components a 1-form is covariant (`a_i = DF^T a_phys`), a
2-form a contravariant density (`b^i = J DF^{-1} b_phys`), a 0-form a value
and a 3-form a density (`J` times the value). Wedge products and
contractions are metric-free in these components -- `a¹ ∧ b¹` is the
2-form with components `a × b`, `u × B` of a velocity 2-form and the flux
2-form `B` is the 1-form `(B × u)/J`, which is why the induction `dB/dt =
curl(u × B)` conserves flux exactly on the discrete level -- and the metric
enters only where a Hodge star does and in the pairing with the output
basis (a 1-form against the 1-form basis carries `J G^{-1}`, a 2-form
against the 2-form basis `G/J`, the mixed pairings nothing).
`test/test_products.py` checks every case against the mass and projection
matrices.

## 3. Extraction

Assembly runs on the unconstrained tensor-product basis. An extraction
operator `E_k` of shape `(n_k, n_k_raw)` maps it onto the conforming space:

```
Λ_I = Σ_J (E_k)_IJ Λ_raw_J,      A = E_k A_raw E_k^T
```

`MatrixFreeExtraction` in `mrx/extraction_operators.py` stores the nonzeros
as `(rows, cols, vals)` and applies `E` and `E^T` as one gather and one
segment sum. Its builder, `PolarExtractionOperator(Λ, xi, zero_bc)`, fuses
the ring-0 and ring-1 radial functions of every `zeta` slice into three
axis functions with the weights `xi` from `get_xi`; `zero_bc=True` is
Dirichlet at `r = 1`.

A `DeRhamSequence` holds two extractions per degree: `E(k)` (periodic and
polar only) and `E(k, True)` (also drops the `r = 1` functions); `.T` is
the transpose and `n(k, dirichlet)` the sizes. Every apply and solve takes
`dirichlet=True|False` and picks the pair.

## 4. k-form Laplacians

The Hodge Laplacian of degree `k` is

```
L_k = K_k + D_{k-1} M_{k-1}^{-1} D_{k-1}^T
```

with `K_3 = 0` and `D_{-1} = 0`. `apply_laplacian` applies it; the inverse
mass in the second term is a solve. `apply_inverse_laplacian` solves it: CG
on `K_0` with the harmonic mode deflated at `k = 0`, the Hodge split at
`k = 1, 2`, the saddle MINRES at `k = 3`
([preconditioning.md](preconditioning.md), section 1).
`apply_inverse_shifted_laplacian` solves `L_k + eps M_k`;
`apply_inverse_mass_plus_eps_laplace_matrix` solves `M_k + eps L_k` as two
SPD CG solves through the split identity. The `DeRhamSequence` methods of
the same names forward to these with the sequence's own `operators`,
`tol`, and `maxiter`. Every solve is preconditioned by the metric-lumped
atom of its `(k, BC)`.

### Harmonic forms

`L_k` has a kernel of dimension given by the Betti numbers passed to
`DeRhamSequence(betti_numbers=...)`; `(1, 1, 0, 0)` is the solid torus. The
kernel vectors live on `SequenceOperators.nullspaces[(k, dirichlet)]` as
arrays of fixed shape `(n_vectors, n_k)`, zero until computed, so a solve on a
fresh bundle deflates nothing. `compute_nullspaces(seq)` in
`mrx/nullspace.py` fills them by a direct Hodge decomposition (needs `b2 =
0`); `compute_nullspaces_iterative` is shifted inverse iteration for any
topology. Both need the preconditioners assembled first. Unshifted solves
deflate the kernel; shifted solves do not.

## 5. Data model

The split follows JAX: what must not change between traces is a plain Python
object captured by closure; what may change is a pytree.

### Static: `DeRhamSequence`

`DeRhamSequence(ns, ps, q, types, *, polar, tol=None, maxiter=10_000, knots=None, betti_numbers=(1, 1, 0, 0))`
in `mrx/derham_sequence.py` owns the topology: the four `DifferentialForm`
objects `basis_0..basis_3`, the `quad` rule, the polar weights `xi`, the
extraction operators (free and Dirichlet), the incidence stencils `g0..g2`
with the polar grad/curl corrections, the 1D basis tables at the
quadrature points (`basis_r_jk`, `d_basis_r_jk`, ...), the Greville data,
and the solve defaults `tol` (default `mrx.precision.SOLVE_TOL`) and
`maxiter`. All of it is built in the constructor. `polar` is keyword-only
and must be `True`; the map is not a constructor argument.

### Dynamic: `SequenceGeometry`

`SequenceGeometry` in `mrx/geometry.py` is an `eqx.Module` with the map and
three arrays on the quadrature grid: the metric `metric_jkl = DF^T DF` of
shape `(N_q, 3, 3)`, its inverse `metric_inv_jkl` `(N_q, 3, 3)` and
`jacobian_j = det DF` `(N_q,)`, plus the mass weights attached from them.
They are built once from `DF` by the constructors and never recomputed;
`DF` itself is not kept, because its only consumer is the physical-frame
pullback at load time (`load(frame='phys')`), which recomputes it with
`map_jacobian_at(seq.map, seq.quad.x)`. Build the geometry with
`SequenceGeometry.from_map(F, seq.quad.x)` (autodiff of `F`) or
`SequenceGeometry.from_spline_map(spline_map, seq)` (sum factorisation of
the spline coefficients). `seq.set_map(F)` and
`seq.set_spline_map(coefficients)` install it as `seq.geometry` and drop the
operator bundle built for the previous geometry.

Maps enter by interpolation. An analytic map is a callable `F(x)`, fitted
by `seq.interpolate(f, 0)`: 1D collocation solves on the tensor space
followed by the polar restriction (`greville_interpolate_map` in
`mrx/geometry.py`). A GVEC state or VMEC wout is not sampled at all:
`build_gvec_map` in `mrx/gvec.py` builds the polar coefficients of `R`, `Z`
from the series coefficients mode by mode (`series_spline_dofs`), wrapped
as a `SplineMap` (`mrx/mappings.py`). There is no reference mass matrix.

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
`mrx/geometry.py` is the production recipe for a geometry file:

```python
seq = DeRhamSequence(ns, (p,) * 3, p + 1, ("clamped", "periodic", "periodic"),
                     polar=True, betti_numbers=(1, 1, 0, 0))
seq.set_map(toroid_map(epsilon=1 / 3, R0=1.0))
ops = seq.build_preconditioners()
ops = compute_nullspaces(seq)
```

1. Topology: `DeRhamSequence`. Bases, extraction, incidence (with the polar
   grad and curl stencils), 1D tables, Greville data: everything static.
2. Geometry: `set_map` or `set_spline_map`. Installs the metric and the
   mass and projection weights. Drops the operator bundle.
3. Preconditioners: `build_preconditioners`, one call: the mass atoms, then
   the Laplacian atoms (which apply the weak term of `L_k` through the mass
   atoms). These are the preconditioners of every solve.
4. Harmonic forms: `compute_nullspaces`, after everything above; they live
   on the bundle.

Nothing on the bundle is built on first use, and nothing on it survives a
geometry change: after a new `set_map`, run steps 3 and 4 again. That is the
contract for an outer loop over geometries (relaxation inside, the map
outside).
