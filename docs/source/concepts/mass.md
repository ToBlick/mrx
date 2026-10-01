# Mass operators

No mass matrix is stored. Every operator that needs quadrature is applied
element by element from the 1-D basis tables and the metric weight at the
quadrature points (`mrx/mass.py`).

## 1. The apply

The mass matrix of `V^k` is `(M_k)_IJ = int Lambda_I . W_k . Lambda_J dx`. A degree-`p`
spline touches `p + 1` cells per axis, so `M_k` is a sum over elements of
dense `(p+1)^3 x (p+1)^3` blocks. A matvec never forms them. Per element it

1. reads the local coefficient cube (on a tensor-product basis the
   element-to-DoF map of every axis is a shift, so this is a stack of
   rolled slices, no index tensor).
2. contracts it to the element's quadrature points by 1-D contractions
   (sum factorisation, with the `theta` and `zeta` tables fused into one).
3. multiplies by the weight, mixing the components at the quadrature points.
4. contracts back and accumulates by the same shifts.

Row and column tables are the same, so the apply is symmetric by
construction. `mass_plan(seq, k)` is the geometry-independent part (tables,
shift plans, weight structure), built once per sequence.
`attach_weights(seq, geometry)` puts the weights on the geometry.
`sumfact_apply(plan, weights, x)` applies them on the raw DoFs, and the
caller applies the extraction, `seq.M[k] @ v = E M_k E^T v`. The
inter-degree projection masses `P_21`, `P_12`, `P_03`, `P_30`
(`projection_plan`) are the same apply with the weight `I`.
`build_mass_diagonal(seq, k)` gives `diag(M_k)` by the same sum
factorisation, with no apply.

## 2. The weights

| k | weight at a quadrature point |
|---|---|
| 0 | `det DPhi` |
| 1 | `det DPhi g^{-1}` |
| 2 | `g / det DPhi` |
| 3 | `1 / det DPhi` |

with `g = DPhi^T DPhi`, where `DPhi` is the Jacobian of the map `Phi`.
`SequenceGeometry` stores `g`, `g^{-1}` and `det DPhi` at the quadrature
points. Each weight entry is one elementwise
product or quotient of them (six unique entries for k = 1, 2), formed once
per geometry in the element layout with the Gauss weights folded in. The
weights are fields of the geometry pytree, so a new map runs the same compiled
kernel with new arguments.

## 3. Memory

Resident per quadrature point: the metric, its inverse and `det DPhi` (19
scalars) plus the unique weight entries (14 over k = 0..3). Per apply the
largest transient is one element field at the quadrature points. A stored
`M_k` would be `O(n^3 (p+1)^6)`.

## 4. Quadrature: `q = p + 1`

`QuadratureRule` uses `q` Gauss points per knot span on every axis
(`composite_quad`). `q` points integrate polynomials of degree `2q - 1`
exactly, and with `q = p + 1` that covers the product of two degree-`p`
splines. The metric weight is not a polynomial, so no rule is exact, and
the quadrature error is of the order of the approximation error.
`build_sequence` passes `q = p + 1`.
