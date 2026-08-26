# The polar axis

A polar map collapses the ring `r = 0` of the logical cube onto a curve. The
tensor-product basis has `n_t · n_z` functions on that ring that all sit at
the same physical points, and a field built from them is not smooth there.
This page describes the constraint that restores regularity, the extraction
that implements it, and the strong derivative on the constrained space.

## 1. Regularity at the pole

Write a scalar spline as `f = Σ_i c_i(θ) N_i(r)` with radial ring `i` and
angular coefficient functions `c_i(θ)`. The map is `x(r, θ) = Σ_i P_i(θ) N_i(r)`
with the whole ring-0 control ring at the pole. `f` is `C^k` at the pole when
its radial jets at `r = 0` match those of a polynomial `q` of degree `k`
composed with the map:

| order | ring condition | polar functions per `zeta` plane |
|---|---|---|
| C⁰ | `c_0(θ) = q_0` | 1 |
| C¹ | `c_1(θ) = q_0 + q_1 · ΔP_1(θ)` | 3 |
| C² | `c_2(θ) = q_0 + q_1 · ΔP_2(θ) + ρ ΔP_1(θ)^T Q ΔP_1(θ)` | 6 |

with `ΔP_i = P_i - pole` and `ρ = 2 N_1'(0)^2 / N_2''(0)`. Rings `0..k` are
replaced by `(k+1)(k+2)/2` functions. The C¹ condition does not depend on the
radial knots; the C² condition does, through `ρ`.

MRX uses C¹ by default: `DeRhamSequence(polar_order=1)`. The three polar
functions are `1`, `cos θ`, `sin θ` in barycentric form. `get_xi(nt, ring1=None)`
in `mrx/extraction_operators.py` returns their weights on rings 0 and 1 as an
array `xi` of shape `(3, 2, nt)`: the barycentric coordinates of the ring-1
control points with respect to the equilateral control triangle (Toshniwal et
al., CMAME 2017; Holderied, thesis eqs. 5.7-5.9). All weights lie in `[0, 1]`
and sum to one on every ring, so constants are exact. `ring1=None` uses the
unit circle, which is exact whenever `∂F/∂r` at the axis is a pure `m = ±1`
mode. `ring1_control_points(pol_map, basis_r, basis_t)` extracts the actual
ring-1 offsets of a poloidal map for `DeRhamSequence(polar_ring1=...)`.

`polar_order=2` uses `get_xi2(nt, basis_r)`, shape `(6, 3, nt)`. The exact C²
condition contains a product of splines of degree `2p`, which the degree-`p`
angular space cannot hold; MRX samples the quadratic term at the Greville
angles instead. Measured on the toroid Poisson problem with poloidal modes
`m = 0, 1, 2` at `p = 3`, the C² errors equal the C¹ errors to every printed
digit at 10-16% fewer degrees of freedom.
Only the k=0 pipeline supports order 2; the k >= 1 extractions and the
derivative stencils below encode C¹. `polar_order=0` builds the C⁰ space.

## 2. The extraction

`PolarExtractionOperator(Λ, xi, zero_bc)` reads `n_polar = xi.shape[0]` and
`ring_depth = xi.shape[1]` and builds a `MatrixFreeExtraction` of shape
`(n_k, n_k_raw)`; `zero_bc=True` also drops the outer ring for a Dirichlet
condition. Per degree, with `o = 1` under Dirichlet and `0` otherwise:

| k | extracted layout |
|---|---|
| 0 | `n_polar · n_z` polar rows, then the bulk rings `ring_depth .. n_r-1-o` |
| 1 | `2 n_z` θ-surgery rows, `3 d_z` ζ-surgery rows, then the `r`, `θ`, `ζ` bulk components |
| 2 | `2 d_z` surgery rows in the first component, then the bulk |
| 3 | a pure selection: no fused rows |

`_k1_row_slices` names the k=1 blocks. The bulk rows are copied by
`_append_bulk_selector`; the fused rows carry the `xi` weights (k=0) and
their radial and angular differences (k=1, 2), which is why `E E^T ≠ I` on a
polar sequence for k = 0, 1, 2 and `E_3 E_3^T = I`.

## 3. The strong derivative on the polar complex

On coefficients the exterior derivative is the topological incidence `G_k`
with entries in `{-1, 0, +1}`. On a non-polar sequence the extraction is a
0/1 selection, so `E_{k+1} G_k E_k^T` is the discrete derivative and
`d ∘ d = 0` holds. On a polar sequence the fused rows make the true
derivative

```
G_k = (E_{k+1} E_{k+1}^T)^{-1} E_{k+1} G_k^raw E_k^T
```

and dropping the Gram inverse breaks nilpotency: `curl ∘ grad ≈ 1`,
`div ∘ curl ≈ 0.02` instead of `1e-16`.

The Gram inverse cancels analytically. Away from the axis `G_k` is plain
`±1` differences; on the apex and first-ring rows it is coefficient
differences weighted by `xi` differences, `xi[l, 1, j] - xi[l, 0, j]` radially
and `xi[l, 1, j+1] - xi[l, 1, j]` angularly. `build_grad_stencil_g0(seq, xi,
dirichlet_in, dirichlet_out)` and `build_curl_stencil_g1` in
`mrx/operators.py` build these stencils from the incidence pattern and `xi`
alone, with no mass and no inverse, for all four `(dirichlet_in,
dirichlet_out)` pairs and their transposes. They are stored on
`SequenceOperators` as `g0_grad_{di}{do}`, `g1_curl_{di}{do}` and applied as
an indexed gather and segment sum. The divergence needs no stencil because
`E_3` is a selection. `assemble_incidence_operators` builds them when the
sequence is polar and `polar_order == 1`.

`apply_incidence_matrix(v, k, dirichlet_in, dirichlet_out, transpose)`
dispatches: the grad stencil at k=0, the curl stencil at k=1, the raw
incidence otherwise. Verified on a polar rotating ellipse against the
Gram-inverse oracle to `8e-16`; `curl ∘ grad` and `div ∘ curl` are `1e-16`
under both boundary conditions.

Use it in preference to the mass-projected `apply_strong_grad`,
`apply_strong_curl`, `apply_strong_div`, which compute `M_{k+1}^{-1} D_k` with
a Krylov solve per apply: on `quasr44970` at `(8, 16, 8)`, `p = 3`, the
incidence form gives `div ∘ curl = 8.6e-16` against `1.3e-10` for the
mass-projected form, and the two curls agree to `1e-12`. The relaxation loop
advances `B` with the incidence curl for that reason.

## 4. Consequences elsewhere

- The polar rows are not tensor-product functions. The preconditioners treat
  them as a dense core (`core_rows`, `probe_core_block` in
  `mrx/metric_lumping_laplacian.py`) and the tensor bulk starts at the first
  unfused ring; see [preconditioning.md](preconditioning.md).
- `MetricLumpingLaplacian` derives the bulk radial window from the extraction
  and raises if the bulk rows are not a full tensor product.
- Interpolation and histopolation on a polar space (`seq.interpolate`) solve
  the collocation problem on the tensor space and restrict it with the
  ring-0/ring-1 surgery; the map fit in `mrx/gvec.py` uses this, so the
  axis of a fitted map is one point per `zeta`.
- A `zeta`-dependent axis (a stellarator) would need `xi` per `zeta` plane.
  `PolarExtractionOperator` takes one `xi` for all planes; the unit-circle
  default is what runs on W7-X.
