# The polar axis

A polar map collapses the ring `r = 0` of the logical cube onto a curve.
The tensor-product basis has `n_t n_z` functions on that ring that all
sit at the same physical points, and a field built from them is not smooth
there. This page describes the constraint that restores regularity, the
extraction that implements it, and the strong derivative on the
constrained space.

## 1. Regularity at the pole

Write a scalar spline as `f = sum_i c_i(theta) N_i(r)` with radial ring `i` and
angular coefficient functions `c_i(theta)`, and the map as
`Phi(r, theta) = sum_i X_i(theta) N_i(r)` with the ring-0 control points at the pole.
`f` is `C^k` at the pole when its radial jets at `r = 0` match those of a
polynomial `q` of degree `k` composed with the map:

| order | ring condition | polar functions per `zeta` plane |
|---|---|---|
| C^0 | `c_0(theta) = q_0` | 1 |
| C^1 | `c_1(theta) = q_0 + q_1 . DeltaX_1(theta)` | 3 |

with `DeltaX_i = X_i - pole`. MRX uses C^1: rings 0 and 1 are replaced by three
polar functions in barycentric form. `get_xi(nt, p)`
(`mrx/extraction_operators.py`) returns their weights on the two rings, an
array `xi` of shape `(3, 2, nt)`: the barycentric coordinates of the ring-1
control points, on the unit circle at the centres of the periodic splines,
with respect to an equilateral control triangle (Toshniwal et al., CMAME
2017). The weights lie in `[0, 1]` and sum to one, so constants are exact,
and the three functions map onto each other under `theta -> -theta`, which the
stellarator symmetry needs.

## 2. The extraction

`build_extraction(form, xi, dirichlet)` returns the extraction of a k-form
space and its core rows, the rows that fuse raw DoFs. With `o = 1` under
Dirichlet and `0` otherwise:

| k | extracted layout |
|---|---|
| 0 | `3 n_z` polar rows, then the bulk rings `2 .. n_r-1-o` |
| 1 | `2 n_z` theta-surgery rows, `3 n_z` zeta-surgery rows, then the `r`, `theta`, `zeta` bulk components |
| 2 | `2 n_z` surgery rows in the first component, then the bulk |
| 3 | a pure selection, no fused rows |

The fused rows carry the `xi` weights (k = 0) and their radial and angular
differences (k = 1, 2), so `E E^T != I` for k = 0, 1, 2 and `E_3 E_3^T = I`.

## 3. The strong derivative on the polar complex

The raw incidence through a polar extraction, `E_{k+1} G_k E_k^T`, is not
nilpotent: the fused rows need the Gram inverse,
`(E_{k+1} E_{k+1}^T)^{-1} E_{k+1} G_k E_k^T`. That inverse cancels
analytically. Away from the axis the derivative is plain `+-1` differences.
On the apex and first-ring rows it is coefficient differences weighted by
`xi` differences. `build_grad_stencil_g0` and `build_curl_stencil_g1`
(`mrx/incidence.py`) build these stencils from the incidence pattern and
`xi` alone for the free and the Dirichlet spaces. The sequence
stores them as `seq.g0_grad` and `seq.g1_curl`, keyed by `dirichlet`. The divergence needs no
stencil because `E_3` is a selection.

`seq.G[k] @ v` (and `seq.G[k].T @ w`) dispatches: the grad stencil at k = 0, the curl stencil at k = 1, the raw
incidence at k = 2. `curl grad` and `div curl` are zero to round-off
under both boundary conditions. The relaxation advances `B` with this curl,
so `div B` is conserved to round-off.

## 4. Consequences elsewhere

- The polar rows are not tensor-product functions. The preconditioners
  treat them as a dense core (`seq.core_rows(k)`), probed through
  the operator (see [preconditioning.md](preconditioning.md)).
- Interpolation and histopolation on a polar space (`seq.interpolate`) and
  the map fit (`mrx.equilibria.build_map`) solve on the tensor space and
  restrict onto the extracted space, so the axis of a fitted map is one
  point per `zeta`.
