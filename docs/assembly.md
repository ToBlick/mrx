# Matrix-free assembly

No mass matrix is stored. Every operator that needs quadrature is applied
element by element from the 1D basis tables and the metric weight at the
quadrature points. This page describes that kernel, the weights, the memory
it uses, and the quadrature rule.

## 1. What is applied

The mass matrix of `V^k` is `(M_k)_IJ = ∫ Λ_I · W_k · Λ_J dx`. Each basis
function is a product of three 1D functions and a degree-`p` spline touches
`p+1` neighbours per axis, so the operator is a sum over elements of dense
`(p+1)^3 × (p+1)^3` blocks. A matvec never forms the block: it folds the
input vector into the contraction.

`build_matrixfree_mass_apply(seq, k, geometry=None)` in
`mrx/local_assembly.py` returns a jitted `x -> M_k x` on the raw
tensor-product space. `mass_core_apply` in `mrx/operators.py` wraps it; the
extraction `E (·) E^T` is applied by the caller, so `apply_mass_matrix` is
`E M_k E^T`. The weak derivatives, stiffness blocks, and Laplacians of
[architecture.md](architecture.md) are compositions of this apply with the
incidence and extraction operators.

## 2. The kernel

`_impl(x, Bvals, W_split, gather_idx, seg_idx)` is the jitted body. For
`n_comp` components (1 for k=0,3; 3 for k=1,2):

1. **Gather.** `x_local = x[gather_idx[c]]` for each column component `c`:
   the element-local coefficient cube of shape
   `(ne_x, ne_y, ne_z, p+1, p+1, p+1)`. `gather_idx` is a static integer
   array from `_flat_dof_plan`, built once on the host.
2. **Column transforms** (`_to_quadrature`): three einsums, one axis at a
   time, take the cube to values at the element's quadrature points,
   `(ne_x, ne_y, ne_z, qx, qy, qz)`. One transform per column component.
3. **Pointwise mix.** For each row component `cr`,
   `v = Σ_cc W_split[(cr, cc)] * u[cc]`: `n_comp^2` multiply-adds at the
   quadrature points. `W_split[(cr, cc)]` is the `(cr, cc)` entry of the
   metric weight reshaped to elements with the Gauss weights folded in.
4. **Row transforms** (`_from_quadrature`): the adjoint three einsums, one
   per row component.
5. **Scatter.** One `jax.ops.segment_sum` over the concatenated row cubes
   with `seg_idx`, into the concatenated output.

Steps 2 and 4 are sum factorisation: `O(q(p+1) + q^2(p+1) + q^3)` per element
instead of `O(q^3 (p+1)^3)`. Mixing at the quadrature points rather than per
`(cr, cc)` pair does a third of the transform work. Row and column bases are
the same tables, so the applied operator is symmetric by construction.

The plan (basis tables, gather indices, segment ids, weights) is passed to
`_impl` as arguments, not captured as constants, so XLA does not constant-fold
the index tensors.

### The 1D tables

`evaluate_basis_local(basis, x_q_flat, q_per_elem)` returns, per element and
axis, the `p+1` active basis values at the element's Gauss points and the
global index of each: shape `(n_elem, q, p+1)` and `(n_elem, p+1)`. A
periodic axis has `n_elem = n` with wrapped indices; a clamped axis has
`n_elem = n - p`; a derivative basis reports `p` locals. The tables are
cached per basis and shared by every operator built on the sequence.

### Component bases

`_component_axis_bases_k1(form, c)` puts the derivative basis on axis `c`
and the primal basis elsewhere; `_component_axis_bases_k2` is the
complement. k=0 uses the three primal bases, k=3 the three derivative
bases.

## 3. The metric weights

`_mass_form_and_weights(seq, k, geometry)` picks the form, the component
bases and the weight field for every `(cr, cc)` pair:

| k | weight at a quadrature point | from `SequenceGeometry` |
|---|---|---|
| 0 | `J` | `jacobian_j` |
| 1 | `J g^{-1}` | `metric_inv_jkl * jacobian_j[:, None, None]` |
| 2 | `g / J` | `metric_jkl * (1 / jacobian_j)[:, None, None]` |
| 3 | `1 / J` | `1 / jacobian_j` |

`SequenceGeometry` (`mrx/geometry.py`) stores `DF_jkl`, `metric_inv_jkl`,
and `jacobian_j`; the metric `g = DF^T DF` is a property. `DF` comes from
`jax.jacfwd(map)` at every quadrature point (`SequenceGeometry.from_map`), or
from the spline coefficients by sum factorisation
(`SequenceGeometry.from_spline_map`). Geometry enters the kernel only through
`W_split`; a new map means new weights and the same compiled kernel.

The quadrature points are flattened theta-major, `(theta, r, zeta)`,
because `QuadratureRule` builds them with `jnp.meshgrid` in its default
`'xy'` indexing. `_split_field` undoes that when it reshapes a weight to
elements. Any code that reshapes a quadrature field must use the same
convention.

## 4. Diagonals without probing

The same tables give exact diagonals with no operator apply:

- `build_mass_diagonal(seq, k)`: `diag(M_k)` on the raw space; only the
  `(c, c)` weight blocks contribute; one `segment_sum` per component.
- `build_stiffness_diagonal(seq, k)`: `diag(G_k^T M_{k+1} G_k)` from the
  derivative tables lifted by `grad_1d`.
- `build_extracted_stiffness_diagonal_k0(seq, dirichlet)`: `diag(E K_0 E^T)`
  including the polar rows, which factor as a 2D `(r, theta)` shape times a
  1D `zeta` table because a k=0 polar row sits at one `zeta` index.
- `build_codifferential_diagonal(seq, k)`: the weak-term diagonal at k=3.

`assemble_mass_jacobi_preconditioner` probes the polar rows of
`diag(E M_k E^T)` through the same apply the solver uses, so the Jacobi
diagonal and the operator agree by construction.

## 5. Memory

Resident per quadrature point: `DF_jkl` (9), `metric_inv_jkl` (9),
`jacobian_j` (1), 19 scalars. Recomputing `DF` on every apply was rejected;
the geometry stays resident. A W7-X run at `(12, 24, 24)`, `p = 3` has
`N_q = (n_r - p) · n_t · n_z · q^3 = 9 · 24 · 24 · 64 ≈ 3.3e5` points.

Per apply, the largest transient is one element field at quadrature,
`O(n^3 (p+1)^2 q)` for the column transform and the `n_comp^2` weight
arrays. A stored matrix would be `O(n^3 (p+1)^6)`: `M_1` at `n = 32`,
`p = 4` is about 83 GB, which is why it is not stored.

Measured on a toroid, `p = 3`, H100: the `M_1` apply is 0.11 ms at
`(8, 16, 8)` and 0.47 ms at `(16, 32, 16)`.

## 6. Quadrature: `q = p + 1`

`QuadratureRule(form, q)` uses `q` Gauss points per knot span on the clamped
and periodic axes (`composite_quad`) and one point on a constant axis. `q`
points integrate polynomials of degree `2q - 1` exactly per span; with
`q = p + 1` that is degree `2p + 1`, which covers the product of two
degree-`p` splines. The metric weight is not a polynomial, so no rule is
exact, and the quadrature error is then of the same order as the
approximation error. `2p` points, the previous default, cost a factor
`(2p / (p + 1))^3` per apply for no gain in order. Every production entry
point passes `p + 1`:
`build_sequence` in `mrx/geometries.py`, `build_gvec_map` in `mrx/gvec.py`,
the Poisson scripts (`quad_order_offset: 0` in `conf/config_poisson_test.yaml`
adds to `p + 1`), and `test/conftest.py`.

## 7. What remains assembled

`mrx/assembly.py` keeps `assemble_vectorial` for the inter-degree projection
blocks `p21, p12, p03, p30` (`assemble_projection_operators`), which the
helicity diagnostic applies. It emits index/value triplets through
`_stencil_triplets`, batched over the angular offsets per radial offset, and
the triplets are applied by gather and segment sum. Nothing else is
assembled.
