# Mass operators

No mass matrix is stored. Every operator that needs quadrature is applied
element by element from the 1D basis tables and the metric weight at the
quadrature points. This page describes that kernel, the weights, the memory
it uses, and the quadrature rule.

## 1. What is applied

The mass matrix of `V^k` is `(M_k)_IJ = ∫ Λ_I · W_k · Λ_J dx`, with the
metric weight built from the map Jacobian `DF` and `J = det DF`:

| k | `W_k` at a quadrature point | formed from |
|---|---|---|
| 0 | `J` | `jac` |
| 1 | `J g^{-1}` with `g^{-1} = DF^{-1} DF^{-T}` | `metric_inv[..., i, j] * jac` |
| 2 | `g / J` with `g = DF^T DF` | `metric[..., i, j] / jac` |
| 3 | `1 / J` | `1 / jac` |

Each basis function is a product of three 1D functions and a degree-`p`
spline touches `p+1` neighbours per axis, so the operator is a sum over
elements of dense `(p+1)^3 × (p+1)^3` blocks. A matvec never forms the
block: it folds the input vector into the contraction.

`sumfact_apply(plan, weights, x)` in `mrx/mass.py` is `x -> M_k x` on the
raw tensor-product space, from the geometry-independent plan of the apply
(`mass_plan(seq, k)`: the 1-D basis tables, the gather and scatter index
plans, the pair structure of the weight; built once on the sequence) and
the weights that ride on the geometry (`attach_weights(seq, geometry)`,
applied by `set_geometry`: `geometry.mass_weights[k]`). `mass_core_apply`
in `mrx/operators.py` binds the two; the extraction `E (·) E^T` is applied
by the caller, so `apply_mass_matrix` is `E M_k E^T`. The weak derivatives,
stiffness blocks and Laplacians of [architecture.md](architecture.md) are
compositions of this apply with the incidence and extraction operators.

## 2. The kernel

`_sumfact_kernel` is the jitted body, one executable per operator shape.
For `n_comp` components (1 for k=0,3; 3 for k=1,2):

1. **Gather.** `x_local = x[gather_idx[c]]` for each column component `c`:
   the element-local coefficient cube of shape
   `(ne_x, ne_y, ne_z, p+1, p+1, p+1)`. `gather_idx` is a static integer
   array from `_flat_dof_plan`, built once on the host.
2. **Column transforms** (`_to_quadrature`): three einsums, one axis at a
   time, take the cube to values at the element's quadrature points,
   `(ne_x, ne_y, ne_z, qx, qy, qz)`. One transform per column component.
3. **Pointwise mix.** For each row component `cr`,
   `v = Σ_cc W[(cr, cc)] * u[cc]`: `n_comp^2` multiply-adds at the
   quadrature points, with the `(cr, cc)` entry of the metric weight
   reshaped to elements and the Gauss weights folded in.
4. **Row transforms** (`_from_quadrature`): the adjoint three einsums, one
   per row component.
5. **Scatter.** One `jax.ops.segment_sum` over the concatenated row cubes
   with `seg_idx`, into the concatenated output.

Steps 2 and 4 are sum factorisation: `O(q(p+1) + q^2(p+1) + q^3)` per element
instead of `O(q^3 (p+1)^3)`. Mixing at the quadrature points rather than per
`(cr, cc)` pair does a third of the transform work. Row and column bases are
the same tables, so the applied operator is symmetric by construction. The
plan is passed to the kernel as arguments, not captured as constants, so
XLA does not constant-fold the index tensors.

The 1D tables come from `evaluate_basis_local(basis, x_q_flat, q_per_elem)`:
per element and axis, the `p+1` active basis values at the element's Gauss
points and the global index of each. A periodic axis has `n_elem = n` with
wrapped indices; a clamped axis has `n_elem = n - p`; a derivative basis
reports `p` locals. The component's axis bases are the form's own
(`DifferentialForm.bases`, the derivative basis on `derivative_axes(c)`).

## 3. The metric weights

`SequenceGeometry` (`mrx/geometry.py`) stores `metric_jkl = DF^T DF`,
`metric_inv_jkl` and `jacobian_j`, built once from `DF` -- which comes from
`jax.jacfwd(map)` at every quadrature point (`SequenceGeometry.from_map`)
or from the spline coefficients by sum factorisation
(`SequenceGeometry.from_spline_map`) -- and then discarded. Every weight
entry is one elementwise product of stored arrays per unique entry (six for
k=1, 2; the symmetric pairs alias), formed once per geometry and moved to
the element layout with the Gauss weights folded in (memoising them was
measured against forming them in the kernel and won); a new map means the
same compiled kernel with a new weight. The projection masses between
degrees (`projection_plan`, `geometry.reference_weights`) use the same
kernel with the reference weight `W = I`. Nothing closes over the
geometry: the weights are fields of the `SequenceGeometry` pytree, cast
with it and, for an ensemble, batched with it.

The quadrature points are flattened r-major, `(r, theta, zeta)`: a flat
quadrature field is the `(nx, ny, nz)` array `field.reshape(seq.quad.shape)`.

`build_mass_diagonal(seq, k)` gives `diag(M_k)` on the raw space from the
same tables with no operator apply (only the `(c, c)` weight blocks
contribute; one `segment_sum` per component): the scaling of the mass
preconditioner.

## 4. Memory

Resident per quadrature point: the metric (9), its inverse (9) and the
Jacobian (1), plus the memoised mass weights, 14 unique entries over
k=0..3 (the projection masses add one field of ones per component).
Per apply, the largest transient is one element field at quadrature,
`O(n^3 (p+1)^2 q)`. A stored matrix would be `O(n^3 (p+1)^6)`: `M_1` at
`n = 32`, `p = 4` is about 83 GB, which is why it is not stored.

## 5. Quadrature: `q = p + 1`

`QuadratureRule(form, q)` uses `q` Gauss points per knot span on every axis
(`composite_quad`). `q` points integrate polynomials of degree `2q - 1`
exactly per span; with `q = p + 1` that is degree `2p + 1`, which covers
the product of two degree-`p` splines. The metric weight is not a
polynomial, so no rule is exact, and the quadrature error is then of the
same order as the approximation error. Every production entry point
(`build_sequence`, `test/conftest.py`) passes `p + 1`.

## 6. What remains assembled

Nothing. The only stored operators are index/value triplets: the extraction
operators and the analytic polar grad/curl stencils, both
`MatrixFreeExtraction` objects applied by gather and segment sum, with a
free transpose.
