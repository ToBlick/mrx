> **Status:** measured 2026-08-28; recommendation at the end
> **Read this for:** how the polar spline map of a GVEC state / VMEC wout is built from the series coefficients without an evaluation grid; interpolation versus L2 projection, mode by mode; the axis and wall behaviour; the measured table
> **Do not read for:** the state-file and wout readers themselves (`mrx/gvec.py`, `mrx/vmec.py` docstrings) or the Clebsch initial condition (`relaxation_ic_2026-08-25.md`)

# The map from the series coefficients, no grid

Tobias, 2026-08-28: "I want a spline map for sure. What I meant is that the
spline should be built directly from analytical data, not via the
intermediate step of evaluating the analytical map on a grid." This note
does that and measures it.

## 1. What the map is, and what the route was

A GVEC state (`.dat`) or a VMEC wout (`.nc`, refit by `mrx/vmec.py` into the
same blocks) gives `R` and `Z` as

    R(rho, theta, zeta) = sum_mn c_mn(rho) cos(2 pi (m theta - n zeta / nfp)),

`c_mn` a clamped B-spline in `rho` on the state's knots (GVEC: degree 5,
10 uniform elements, 15 basis functions; wout: degree 3 through the `ns`
surfaces at the nodes `rho_j = sqrt(s_j)`, data-placed knots, 201 for the
W7-X file), `Z` the same with sines. The map MRX runs on is

    F = (R_h cos(2 pi zeta / nfp), sign R_h sin(2 pi zeta / nfp), Z_h),

`R_h`, `Z_h` scalar splines on the sequence's own 0-form space: the
tensor product of `n_r` clamped uniform splines of degree `p` in `rho`
with `N_theta`, `N_zeta` periodic ones, restricted onto the C1 polar space
by the ring-0/ring-1 surgery (`_conforming_restriction`, the
Guclu-Campos Pinto composition `P_Z Pi_W`). `DF` is autodiff of that
spline; the sequence stores `DF^T DF`, its inverse and `det DF` at the
quadrature points once.

Until today `R_h` was `seq.interpolate(R_fn, 0)`: the series evaluated at
the `n_r N_theta N_zeta` Greville points, three 1-D collocation solves,
restriction. Since 2026-08-27 the samples came from the closed form
(`StateField`), so the only approximation was the map space's own -- but
it was still a sampled fit. `build_gvec_map` also built a SECOND
`DeRhamSequence` of the same `(ns, p)` just to own that fit.

## 2. Analysis

### (a) Angular direction: interpolation is a closed form, mode by mode

Interpolation is linear and the space is a tensor product, so the
interpolant of the series is the sum over modes of the interpolants of
`c_mn(rho) trig(2 pi (m theta - n zeta/nfp))`, and each of those is the
product of three 1-D interpolants (`cos(a - b) = cos a cos b + sin a sin
b`, every term a product). On the uniform periodic basis the Greville
points `x_j` are the centres of the basis functions, so the collocation
matrix `A_kj = B_j(x_k)` is circulant and symmetric, and the interpolant
of `exp(2 pi i m theta)` has coefficients

    c_j = exp(2 pi i m x_j) / sigma(m),    sigma(m) = sum_l B_0(x_l) exp(-2 pi i m l / N).

`sigma` is real (symmetric row) and never zero (Schoenberg's cardinal
interpolation is well posed at the knots for odd `p` and at the midpoints
for even `p`, which is exactly where the Greville points sit). Hence the
tensor coefficients of the whole series are

    C[i, j, k] = sum_mn c_mn[i] cos(2 pi (m x_j - n y_k)) / (sigma_t(m) sigma_z(n)),

with `c_mn[i]` the radial coefficients of section (b) -- the samples at
the Greville points, each mode divided by its symbol. No angular solve.
This reproduces the sampled route to round-off (measured: section 4,
`|dof sampled - dof interp|` at 1e-15 of the largest DoF): a cost and
clarity gain, not an accuracy gain, as anticipated.

The symbol is `N`-periodic in `m`: a mode beyond the Nyquist frequency
`N/2` is interpolated as its alias `m mod N`, with the alias's gain
`1/sigma`, which grows toward Nyquist (`1/sigma(N/2) = 3` at `p = 3`).
This is the aliasing every sampled fit has; W7-X at `(24, 12)` angular
resolution puts 138 of the 288 `R` modes beyond Nyquist (summed amplitude
1.4e-2 m against 0.53 m for `m = 1`), at `(32, 32)` none.

### (b) Radial direction: exact only when the map's space contains the state's

`c_mn` is a spline on the state's knots; the map's radial space is the
degree-`p` clamped uniform space on `n_r - p` elements. Interpolation (or
projection) of `c_mn` onto it is exact when and only when the map's space
contains it: `p >= deg_state` and the map's knots refine the state's.

* GVEC `.dat` (degree 5, 10 uniform elements): exact at `p = 5` with
  `n_r - 5` a multiple of 10 -- `n_r = 15` is GVEC's own basis, verbatim
  (measured: section 4, the `(15, 24, 24) p = 5` row).
* VMEC wout (degree 3 through 201 edge-refined nodes in `rho`): exact only
  with all 201 knots in the map, i.e. a 201-element radial mesh; at any
  `n_r` we run the radial direction is the map's own approximation.

`DeRhamSequence(knots=...)` takes a prescribed radial knot vector, so the
exact case can always be set up; it is not what production wants (`ns` IS
the map resolution by design). Either way the radial step is one small
`n_r x n_r` solve shared by all modes: the clamped collocation matrix
(interpolation) or the 1-D mass matrix (projection), the latter with the
moments `int B_i c_mn` by Gauss quadrature on the union of the two knot
sets, exact for the spline product.

### (c) L2 projection: the same closed form with a different symbol

The L2 (Galerkin, logical `L2[0,1]^3`) projection is linear and
tensor-product too, so the same per-mode construction applies with the
angular moments in closed form: the B-spline of degree `p` on spacing
`h = 1/N` centred at `x_j` has Fourier transform

    int B_j(theta) exp(2 pi i m theta) dtheta = h sinc(m h)^(p+1) exp(2 pi i m x_j),

(`sinc(x) = sin(pi x)/(pi x)`) and the 1-D mass matrix is circulant with
symbol `mu(m) = sum_l M_l0 exp(-2 pi i m l/N)`, so the projection of
`exp(2 pi i m theta)` has coefficients `exp(2 pi i m x_j) h sinc(m h)^(p+1) / mu(m)`.
Same coefficient formula as (a) with `gamma(m) = h sinc(mh)^(p+1) / mu(m)`
in place of `1/sigma(m)`. A mode beyond Nyquist is DAMPED by `sinc^(p+1)`
instead of aliased with gain up to 3; within Nyquist the projection is the
`L2`-optimal spline. The weight is the logical one: no `J`, no metric --
the map defines the metric, so a metric-weighted projection would be
circular, and this keeps every factor explicit.

Then the polar restriction, as for every 0-form: the restriction is
linear, so it commutes with the sum over modes, and it is the same
operator for all three routes. The one thing it does differently: the
interpolant of an `m > 0` mode vanishes on ring 0 exactly (`c_mn(0) = 0`
in the state, and the clamped Greville interpolant reproduces the end
value), so ring 0 is already theta-independent and the surgery only
touches ring 1; the L2 projection does not interpolate at `rho = 0`, ring
0 of an `m > 0` mode carries an `O(h^(p+1))` coefficient and the surgery
removes it. Neither route is the L2 projection onto the polar space itself
(that would need the polar mass matrix, `E M E^T`); the sampled route was
not either.

### The axis

GVEC pins, in the state, `coef[0] = 0` for `m > 0`, `coef[1] = 0` as well
for `m >= 2`, and `coef[0] = coef[1]` for `m = 0` (read off the W7-X
FMM002 file: 6e-18, 6e-19, 1e-17). With degree-5 clamped splines that is
`c_1 = a rho + O(rho^2)`, `c_m>=2 = O(rho^2)`, `c_0 = c_0(0) + O(rho^2)`:
in the poloidal plane `(R, Z) - axis = rho (linear in cos, sin theta) +
O(rho^2)`, `det DF = rho D(zeta) + O(rho^2)` with `D` independent of
theta. The series is C1 across the axis in Cartesian terms, and its axis
structure is exactly the C1 polar space's (ring 1 = the linear span). The
wout refit pins `c_m(0) = 0` for `m > 0` but not the slope of `m >= 2`,
so a wout series can carry a small cone (`det DF / rho` varying with
theta); the ring-1 surgery removes the `m >= 2` content of ring 1 and the
spline map is C1 regardless. Measured as `det DF / rho` over theta at
`rho = 1e-2, 1e-3, 1e-5`, section 4.

### The wall, `rho = 1` exactly

`DiscreteFunction` evaluates through `evaluate_local`, whose span clipping
evaluates the last polynomial piece at `rho = 1`, value and autodiff
derivative alike -- the `det DF = 0` wall artefact recorded in memory
(`spline-map-DF-singular-at-r1`) lived in `SplineBasis.evaluate`'s
`x == T[-1]` patch, which the map no longer goes through. `StateField`
clips `rho` to `[0, 1]`; JAX splits the gradient of a tie, so at `rho =
1.0` EXACTLY the series map's radial derivative is halved (`det` halved,
not zero). Nothing in the pipeline evaluates a map at `rho = 1` exactly
(Gauss points are interior; the Greville points, which include both ends,
are no longer where the map is evaluated); the number is printed in the
study for the record.

## 3. Implementation

`mrx.gvec.series_spline_dofs(block, sp, nfp, seq, l2)` builds the polar
DoFs from a block; `build_gvec_map(path, seq, ...)` takes the sequence the
map lives on (no second sequence). Study script:
`scripts/map_projection_study.py`. Tests: `test/test_gvec.py` (closed
form == sampled interpolant at even and odd `p`, including a mode beyond
Nyquist; radial exactness on the state's knots; the L2 symbol solves the
normal equations assembled by an independent quadrature).

## 4. Measurements

MEASUREMENTS_PLACEHOLDER

## 5. Recommendation

RECOMMENDATION_PLACEHOLDER
