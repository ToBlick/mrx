# Mass operators

No mass matrix is stored. Every operator that needs quadrature is applied
from the 1-D basis tables and the metric weight at the quadrature points
(`mrx/mass.py`).

## 1. The apply

The mass matrix of `V^k` is `(M_k)_IJ = int Lambda_I . W_k . Lambda_J dx`. A matvec
never forms it. It is a global sum factorisation on the whole coefficient grid:

1. the coefficients of each component are zero-padded to the largest grid
   of the form, so the three components of a 1- or 2-form form one batch.
2. one contraction per axis with the 1-D basis tables takes the batch to the
   quadrature grid.
3. the weight mixes the components at every quadrature point.
4. the same contractions with the row tables take it back.

On a GPU every launched kernel costs a few microseconds whatever its size, and
at the meshes MRX runs the apply is bound by the number of kernels, not by
arithmetic. The batched form needs a handful of large contractions where an
element-by-element apply (gather the local cube, two small contractions in,
two out, scatter, per component) needed about 30 kernels for `M_1`. It is
1.5x faster on li383 (12,16,16) and 1.4-2.2x on W7-X (16,32,32)
(`docs/research/performance.md`).

Row and column tables are the same, so the apply is symmetric by
construction. `mass_plan(seq, k)` is the geometry-independent part (the padded
tables and the component shapes), built once per sequence.
`attach_weights(seq, geometry)` puts the weights on the geometry.
`sumfact_apply(plan, weights, x)` applies them on the raw DoFs, and the
caller applies the extraction, `seq.M[k] @ v = E M_k E^T v`. The
inter-degree projection masses `P_21`, `P_12`, `P_03`, `P_30`
(`projection_plan`) are the same apply with the weight `I`.
`build_mass_diagonal(seq, k)` gives `diag(M_k)` by the same sum
factorisation with squared tables, with no apply.

## 2. The weights

| k | weight at a quadrature point |
|---|---|
| 0 | `det DPhi` |
| 1 | `det DPhi g^{-1}` |
| 2 | `g / det DPhi` |
| 3 | `1 / det DPhi` |

with `g = DPhi^T DPhi`, where `DPhi` is the Jacobian of the map `Phi`.
`SequenceGeometry` stores `g`, `g^{-1}` and `det DPhi` at the quadrature
points. The weight is one pointwise product or quotient of them, stored as
the full `n x n` matrix per quadrature point (`n` = 1 or 3 components),
formed once per geometry on the quadrature grid with the Gauss weights
folded in. The
weights are fields of the geometry pytree, so a new map runs the same compiled
kernel with new arguments.

## 3. Memory

Resident per quadrature point: the metric, its inverse and `det DPhi` (19
scalars) plus the weights (20 over k = 0..3, and 10 for the projection
masses). Per apply the largest transient is the batch of component fields on
the quadrature grid. A stored
`M_k` would be `O(n^3 (p+1)^6)`.

## 4. Quadrature: `q = p + 1`

`QuadratureRule` uses `q` Gauss points per knot span on every axis
(`composite_quad`). `q` points integrate polynomials of degree `2q - 1`
exactly, and with `q = p + 1` that covers the product of two degree-`p`
splines. The metric weight is not a polynomial, so no rule is exact, and
the quadrature error is of the order of the approximation error.
`build_sequence` passes `q = p + 1`.
