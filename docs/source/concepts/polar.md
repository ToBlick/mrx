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

`build_extraction(form, xi, dirichlet, counts)` returns the extraction of a
k-form space and its core rows, the rows that fuse raw DoFs. `counts` are the
theta counts of a graded space (section 3), `None` on a uniform one. With
`o = 1` under Dirichlet and `0` otherwise:

| k | extracted layout |
|---|---|
| 0 | `3 n_z` polar rows, then the bulk rings `2 .. n_r-1-o` |
| 1 | `2 n_z` theta-surgery rows, `3 n_z` zeta-surgery rows, then the `r`, `theta`, `zeta` bulk components |
| 2 | `2 n_z` surgery rows in the first component, then the bulk |
| 3 | no fused rows, the bulk rings `1 .. n_r-2` |

The fused rows carry the `xi` weights (k = 0) and their radial and angular
differences (k = 1, 2), so `E E^T != I` for k = 0, 1, 2. On a uniform space
every bulk row selects one raw DoF and `E_3 E_3^T = I`.
`ring_layout(form, dirichlet, counts)` numbers the bulk rows by component,
radial ring, theta and zeta index.

## 3. Graded theta resolution

On a uniform polar space every ring has `n_theta` cells, so the theta cells
near the axis are far narrower than the radial ones. The stiffness then
scales like `h^-4` instead of `h^-2`, and the CFL weights grow with it. A
graded space gives each radial ring of basis functions its own uniform
periodic theta space, with fewer cells near the axis.
`build_sequence(..., graded=True)` turns it on. The fields live on the graded
space. The map lives on the uniform polar space with `map_theta` theta cells on
every ring (default `n_theta`), written in the fields' basis by knot insertion,
because a map on the graded space would change its metric where the count
changes. The CFL weights of the relaxation loop
(`logical_cfl_weights`) take the theta cell of the finest ring active at
each point, not the wall's.

The counts come from `graded_theta_counts(T_r, p, n_theta)`, from the radial
knots alone. A ring at radius `r` has at least `n_theta r^(1/3)` theta cells
(`GRADED_EXPONENT`), and its theta cell is at most `GRADED_ASPECT` (pi / 2)
times its smallest radial cell. Each count is the coarsest count times the
smallest power of two that meets both. The coarsest is `n_theta / 2^l`, the
smallest such count not below `GRADED_THETA_MIN` (8). The counts never decrease
outward. The three innermost rings carry the polar axis functions and take the
count of the third, so the axis functions are not coarser than the ring next to
them.

The exponent 1/3 is what keeps the order. With `n_theta r` the rings near the
axis keep their opening angle under refinement, and a spline on a cell with a
fixed opening angle does not converge. On li383 p = 2 the L2 error of the
manufactured solutions then stalls from (36,60,36) to (48,80,48), at 2.7 times
the uniform error. With `n_theta r^(1/3)` it converges at order 2 with the
uniform error to 1 %, and the stiffness solves keep flat counts (89 and 218 at
(48,80,48), against 185 and 464 on the uniform space). A ring of derivative splines in `r` sits between two primal rings
and takes the count of the outer one. On li383 (24,64,12) the counts are
32 on the first four rings and 64 outside.

The raw tensor grid stays uniform. A graded basis function is the knot
insertion of one coarse theta function into the raw functions of its ring,
`B^m_J = sum_j R[j, J] B^n_j` (`prolongation(m, n, p, derivative)`). For
uniform dyadic knots `R` is a product of subdivision masks with exact dyadic
entries, for the primal and for the derivative splines. It commutes with the
coefficient difference, so the graded spaces form an exact subcomplex. The
uniform space is the graded one with every count equal to `n_theta`, and
`R = I`. Away from the axis `E E^T` is one small block `R^T R` per ring and
zeta layer (`gram_blocks`). `R` also commutes with the reflection
`theta -> -theta`, so the parity views of a half-period sequence carry over
unchanged.

## 4. The strong derivative on the polar complex

The derivative of an extracted k-form, written in the extracted (k+1)-form
basis, is `(E_{k+1} E_{k+1}^T)^{-1} E_{k+1} G_k E_k^T` with the raw incidence
`G_k`. `build_polar_derivative` (`mrx/incidence.py`) forms it with
`scipy.sparse` and drops the entries at round-off. `E E^T` is the identity
except on the polar core rows, where it is a small dense block, and on a
graded space on the rings with fewer theta cells than the wall, where it is
one block `R^T R` per ring and zeta layer (`gram_inverse` in
`mrx/extraction_operators.py`). The graded spaces form an exact subcomplex,
so the result is the exact derivative there too. The sequence stores grad,
curl and div as `seq.g0_grad`, `seq.g1_curl` and `seq.g2_div`, keyed by
`dirichlet`.

`seq.G[k] @ v` (and `seq.G[k].T @ w`) applies the derivative of degree k.
`curl grad` and `div curl` are zero to round-off
under both boundary conditions. The relaxation advances `B` with this curl,
so `div B` is conserved to round-off.

## 5. Consequences elsewhere

- The polar rows are not tensor-product functions. The preconditioners
  treat them as a dense core (`seq.core_rows(k)`), probed through
  the operator (see [preconditioning.md](preconditioning.md)).
- The map fit (`mrx.equilibria.build_map`) solves on the tensor space and
  restricts onto the extracted space by the coefficient fit `(E E^T)^-1 E`
  (`seq.conforming`), so the axis of a fitted map is one point per `zeta`.
  On a graded space this fit also inverts the blocks `R^T R`. Interpolation
  restricts differently, see section 6.
- On a graded space the atoms have one tensor-product block per component
  and theta count, with the Galerkin theta matrices `R^T A R` of the coarse
  functions. The blocks of neighbouring counts are coupled through a few
  rings, and the atoms keep that coupling (see
  [preconditioning.md](preconditioning.md)).

## 6. Interpolation

`seq.interpolate(f, k)` interpolates at the Greville points (k = 0) or
matches the integrals over the Greville edges, faces and cells (k = 1, 2, 3)
on the tensor space. These projections commute with the derivative there.
The tensor coefficients are then restricted onto the extracted space, and
the restriction decides whether the result still commutes,
`G_k Pi_k f = Pi_{k+1} df`.

On a ring with `n_theta` cells the restriction keeps the tensor
coefficients. On a ring with fewer cells it is the ring's own projection in
theta: interpolation at its Greville points, or histopolation over its
Greville cells for the derivative splines in theta. These two commute with
the theta difference. The coefficient fit of `seq.conforming` does not.

The radial difference couples rings of different counts. Where the count
grows, the ring of radial derivative splines between the two primal rings
also takes a term from the theta derivative on its inner ring. The term is
`K = (rho_out - R rho_in) D^+`, with `rho` the interpolation onto a ring's
count, `R` the knot insertion between the two counts and `D^+` the inverse
of the raw theta difference on mean-free vectors. It vanishes on graded
forms, so the restriction stays a projection. `ring_projections` in
`mrx/projectors.py` builds these matrices once per sequence.

At the axis the restriction is the conforming projection onto the C^1
polar splines of Güçlü, Patrizi and Campos Pinto (arXiv:2505.15996). It
acts on each zeta layer of the two innermost rings, after their theta
projection onto the coarsest count. The three axis functions of a 0-form,
and of the zeta part of a 1-form, take the mean of ring 0 as their value
on the axis and the part of ring 1 in the span of `cos theta_j` and
`sin theta_j`, with `theta_j` the angles of the polar weights. The two axis
functions of a 1-form, and of the `dtheta dzeta` and `dzeta dr` parts of a
2-form, take the part of the radial ring 0 in that span. Ring 1 of a
component with radial derivative splines takes ring 0 on top of its own,
except the part the axis functions carry. With derivative splines in theta
that is all of ring 0. So the restriction also changes these ring-1 rows of
the uniform space, and keeps every other row.

This commutes with the derivative whenever the tensor coefficients have
the regularity of a smooth form at the axis. Ring 0 of a 0-form, and of
the zeta part of a 1-form, does not depend on theta. Ring 0 of the theta
part of a 1-form, and of the `dtheta dzeta` part of a 2-form, vanishes. The
pullback of a smooth field in space has both properties. A field that
breaks them is singular at the axis, and the identity is not claimed for
it. On the axis functions the restriction is the identity, so
`interpolate` stays a projection.

`test_derham.py` checks the identity on the graded fixture to round-off,
axis rows included, with fields that have these properties and do not
vanish at the axis.
