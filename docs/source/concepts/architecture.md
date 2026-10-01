# Architecture

MRX discretises the de Rham complex

```
V0 --grad--> V1 --curl--> V2 --div--> V3
```

with tensor-product B-splines on the logical cube `[0,1]^3` in coordinates
`(r, theta, zeta)`, mapped to the physical domain by the map `Phi` (its Jacobian is `DPhi`). This page names
the objects and the order in which they are built. The mass kernel is in
[mass.md](mass.md), the solvers and preconditioners in
[preconditioning.md](preconditioning.md), the polar axis in
[polar.md](polar.md).

## 1. Spaces

**1-D bases** (`mrx/spline_bases.py`). `SplineBasis(n, p, type)` is a 1-D
B-spline basis of `n` functions of degree `p`, `type` `"clamped"` or
`"periodic"`. `DerivativeSpline(s)` holds the derivatives of `s`: `n - 1`
functions of degree `p - 1` on a clamped axis, `n` on a periodic one, each
of unit integral. `evaluate_local(x)` returns the `p + 1` nonzero values at
`x` and their indices.

**k-forms** (`mrx/differential_forms.py`). `DifferentialForm(k, ns, ps, types)`
holds the three 1-D bases `Lambda[a]` and their derivative bases `dLambda[a]`. A
basis function of `V^k` is a product of one 1-D function per axis,
differentiated on the axes the degree prescribes (`derivative_axes(c)`):

| k | components | axis bases per component |
|---|---|---|
| 0 | 1 | `(Lr, Lt, Lz)` |
| 1 | 3 | `(dLr, Lt, Lz)`, `(Lr, dLt, Lz)`, `(Lr, Lt, dLz)` |
| 2 | 3 | `(Lr, dLt, dLz)`, `(dLr, Lt, dLz)`, `(dLr, dLt, Lz)` |
| 3 | 1 | `(dLr, dLt, dLz)` |

Here `Lr` is `Lambda[0]`, the radial basis, and `dLr` is `dLambda[0]`, its
derivative basis (likewise `t` for theta and `z` for zeta).

A coefficient vector is the concatenation of the components, each the
C-order ravel of its shape. `grad`, `curl` and `div` of a basis function
are exact combinations of the next space's basis functions, so the discrete
complex is exact. `DiscreteFunction(dof, Lambda, E)` evaluates a DoF vector at
logical points. `Pushforward(f, Phi, k)` gives its physical components.

**Quadrature** (`mrx/quadrature.py`). `QuadratureRule` is the tensor product
of composite Gauss rules with `q = p + 1` points per knot span. The flat
points are ordered r-major.

## 2. Operators

Only the mass matrices `(M_k)_IJ = int Lambda_I . W_k . Lambda_J dx` need quadrature,
and they are never stored ([mass.md](mass.md)). The exterior derivative on
coefficients is the topological incidence `G_k` with entries in
`{-1, 0, +1}` (`mrx/incidence.py`), independent of the geometry. Everything
else is a composition of applies. The sequence exposes each matrix as an
operator object indexed by the form degree and applied with `@`
(implemented in `mrx/operators.py`):

| operator | apply | inverse |
|---|---|---|
| mass `M_k` | `seq.M[k] @ v` | `seq.M[k].solve(b)` |
| strong derivative `G_k` (grad, curl, div) | `seq.G[k] @ v`, `seq.G[k].T @ w`, with the polar stencils of [polar.md](polar.md) | |
| weak derivative `D_k = M_{k+1} G_k` | `seq.D[k] @ v`, `seq.D[k].T @ w` | |
| stiffness `S_k = G_k^T M_{k+1} G_k` | `seq.S[k] @ v` | |
| Hodge Laplacian `L_k = S_k + M_k G_{k-1} M_{k-1}^{-1} G_{k-1}^T M_k` | `seq.L[k] @ v` | `seq.L[k].solve(b)` |
| shifted Laplacian `M_k + eps L_k` | | `seq.shifted(k, eps).solve(b)` |
| projection masses `P_{k->l}` | `seq.P[k, l] @ v` | |

`seq.M[k].precondition(r)` and `seq.L[k].precondition(r)` apply the
preconditioners of [preconditioning.md](preconditioning.md). An operator object
holds only the sequence and its degree, so creating one inside a jitted
function costs nothing and compiles nothing. For example, the weak curl of a
2-form and the helicity are

```python
J = seq.M[1].solve(seq.D[1].T @ B)
A = seq.L[1].solve(seq.D[1].T @ B)
H = A @ (seq.P[2, 1] @ (2 * B - seq.G[1] @ A))
```

**Products.** The nonlinear terms are pointwise products of two forms,
exposed as L2 loads: the factors are evaluated at the quadrature points
(`evaluate_at_quadrature`) and the product is integrated against the
output basis. `cross_product_load_values` covers `w x u` for every degree
combination, `dot_product_load_values`, `scalar_product_load_values` and
`scalar_vector_load_values` the others. In reference components a 1-form is
covariant (`DPhi^T a`), a 2-form a contravariant density (`det DPhi DPhi^{-1} b`), a
3-form a density. Wedge products and contractions are metric-free in these
components, so the induction `dB/dt = curl(u x B)` conserves flux exactly,
and the metric enters only where a Hodge star does.

## 3. Extraction

Assembly runs on the unconstrained tensor-product basis. An extraction
operator `E_k` of shape `(n_k, n_k_raw)` maps it onto the conforming space,
`A = E_k A_raw E_k^T`. `build_extraction` (`mrx/extraction_operators.py`)
returns it as a `MatrixFreeExtraction` (COO triplets, applied by a gather
and a segment sum): the polar functions at the axis ([polar.md](polar.md))
and, in the Dirichlet spaces, the `r = 1` functions dropped.

**Boundary conditions belong to the space.** The sequence `seq` has the
Dirichlet spaces (no flux through the wall, tangential `E` and `A` zero,
`p = 0`), the working spaces of `B`, `J`, `E` and the force. The view
`seq.free` has the free spaces. It shares every array with `seq`, and all
operators, `seq.E(k)`, `seq.n(k)`, `seq.load` and `seq.interpolate` act on
the spaces of the object they are called on. No operator takes a boundary
condition argument. `seq.restrict(v, k)` drops the wall DoFs of a free form
and `seq.extend(v, k)` pads a Dirichlet form with zeros. The initial field,
for example, is the curl of a free vector potential restricted to the
Dirichlet 2-forms:

```python
A = seq.free.interpolate(A_ref, 1, frame='logical')
B = seq.restrict(seq.free.G[1] @ A, 2)
```

The free view is a snapshot of what is installed on the sequence, so take it
anew after `set_map`, `build_preconditioners` or `compute_nullspaces`.

## 4. Harmonic forms

`L_k` has a kernel of the dimension of the Betti numbers of the domain,
always a solid torus (`(1, 1, 0, 0)`). `compute_nullspaces(seq)`
(`mrx/nullspace.py`) builds the harmonic forms by a direct Hodge
decomposition of closed seeds and stores them on the operator bundle.
`seq.nullspace(k)` and `seq.free.nullspace(k)` read them. Every unshifted solve deflates
them.

## 5. Data model

What must not change between traces is a plain Python object. What may
change is a pytree.

**`DeRhamSequence`** (`mrx/derham_sequence.py`),
`DeRhamSequence(ns, p, *, nfp, symmetry, knots, equilibrium, tol, maxiter)`:
the four `DifferentialForm`s `basis_0..basis_3` (clamped in `r`, periodic in
`theta` and `zeta`, degree `p`), the quadrature rule (`p + 1` Gauss points per
span), the extractions, the incidence stencils, the 1-D basis tables at the
quadrature points, the plans of the mass applies, the parity views and the
float64 copy `seq.residual` of the mixed configuration, all built in the
constructor. `seq.free` is taken on demand. It is registered as a pytree (`mrx/pytree.py`), so a function
of it under `eqx.filter_jit` takes its arrays as inputs.

**`SequenceGeometry`** (`mrx/geometry.py`), an `eqx.Module`: the map and,
at the quadrature points, the metric `metric_jkl = DPhi^T DPhi`, its inverse
`metric_inv_jkl` and `jacobian_j = det DPhi`, with the mass weights derived
from them. `seq.set_map(Phi)` builds it from `Phi` by autodiff and installs it
as `seq.geometry`. A map that folds (`det DPhi <= 0`) is refused.

**`SequenceOperators`** (`mrx/operators.py`), an `eqx.Module` keyed
`(k, dirichlet)`: the mass atoms `mass_lumping`, the Laplacian atoms
`laplacian_lumping` ([preconditioning.md](preconditioning.md)) and the
harmonic forms `nullspaces`. `seq.build_preconditioners()` builds it as
`seq.operators`, for the Dirichlet and the free spaces. The sequence and its
free view share it, and each picks the entries of its own spaces. A rebuild for a new geometry of the same discretisation
reuses the compiled programs.

## 6. Stellarator symmetry

A stellarator-symmetric map satisfies `Phi(r, -theta, -zeta) = S Phi(r, theta, zeta)`
with `S = diag(1, -1, -1)`, and every field of the relaxation has a parity
under the reflection (`mrx/symmetry.py`):

- odd: `B`, `A`, `J`, `E` and the harmonic 2-form.
- even: the velocity, the force, the pressures.

`build_sequence(..., symmetry="stellarator")`, the default, projects the
spline map onto the symmetry and builds a half-period sequence
(`seq.half_period`, whenever the angular knots are uniform): the quadrature
covers `zeta` in `[0, 1/2]` with doubled weights, and the reflection turns the half-period moments into the
full-period ones. `seq.odd` and `seq.even` are the sequence reduced to one
parity: the same geometry and kernels, half the DoFs, their own extraction,
stencils and reduced atoms. Every field lives on the view of its parity, and a
product load is formed on the view of the product. `"field-period"` and
`"none"` build a full-period sequence, on which both views are the sequence
itself. The parity views compose with the free view: `seq.odd.free` and
`seq.free.odd` are the same space. The half-period reduction needs uniform angular knots.

## 7. Assembly order

`build_sequence(geometry, ns, p)` in `mrx/geometry.py` is the recipe for a
geometry file:

1. Topology: `DeRhamSequence`, everything static.
2. Geometry: `set_map` with the map of the file (`mrx.equilibria.build_map`).
   Drops the operator bundle.
3. Preconditioners: `build_preconditioners`, the mass atoms, then the
   Laplacian atoms, on the base sequence and both parity views.
4. Harmonic forms: `compute_nullspaces(seq)`, a separate call.

```python
seq, ops = build_sequence("data/wout_li383_low_res_reference.nc", (8, 12, 12), 2)   # steps 1-3
compute_nullspaces(seq)                                                             # step 4
```

Nothing on the bundle is built on first use, and nothing on it survives a
geometry change. After a new `set_map`, run steps 3 and 4 again.
