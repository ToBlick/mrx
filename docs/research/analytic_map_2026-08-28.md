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

`mrx.gvec.series_tensor_coefficients(block, sp, nfp, seq)` builds the
tensor coefficients of the L2 projection from a block,
`series_spline_dofs` restricts them onto the polar space;
`build_gvec_map(path, seq, ...)` takes the sequence the map lives on (no
second sequence). The measurement below was made with both routes in the
library (commit a758617); the interpolant was then deleted. Study script:
`scripts/map_projection_study.py` (now: the sampled fit versus the
projection versus the series). Tests: `test/test_gvec.py` (the tensor
coefficients satisfy the 3-D normal equations assembled by an independent
Gauss rule, at even and odd `p`, cosine and sine, with modes beyond
Nyquist; radial exactness on the state's knots; the angular symbol solves
the 1-D normal equations; the wall derivative of the series is the left
limit).

## 4. Measurements

All runs `scripts/map_projection_study.py` (commit a758617, the version
with both routes), float64, one H100 each, logs under
`outputs/analytic_map/2026-08-28/13-1*/study_*.log`. W7-X = the beta = 5%
wout (200 modes, `m <= 9`, `|n/nfp| <= 10`, `ns = 201`), QA = the
Landreman-Paul 2021 QA wout (128 modes, `m <= 7`, `|n/nfp| <= 8`,
vacuum), GVEC = W7-X FMM002 state (288 modes, `m <= 11`, `|n/nfp| <= 12`).
"sampled" is the 2026-08-27 route; "interp" its per-mode closed form;
"L2" the per-mode projection.

**(a) confirmed.** `max |dof sampled - dof interp| / max |dof|`: 5e-15
(W7-X 12 p3), 6e-15 (QA), 1e-14 (W7-X 16 p3), 5e-14 (GVEC 15 p5): the
closed form IS the sampled fit. Coefficients: sampled 2.4-3.3 s, interp
0.03-0.12 s, L2 0.6-2.3 s (the 1-D angular mass assembly; irrelevant next
to the 16-27 s the second `DeRhamSequence` used to cost, now gone).

**Map against the series** on 4000 random points, `rho` in [0.05, 0.95]
(`|dX|` in metres; `|dDF|/|DF|` Frobenius, relative to the rms of the
series' `DF`):

| case | route | max dX | rms dX | max dDF | rms dDF | det/det_ref |
|---|---|---|---|---|---|---|
| W7-X (12,24,12) p3 | interp | 9.67e-4 | 1.75e-4 | 4.28e-3 | 1.02e-3 | [0.982, 1.024] |
| | L2 | 5.75e-4 | 1.27e-4 | 4.09e-3 | 9.52e-4 | [0.984, 1.023] |
| W7-X (12,24,12) p4 | interp | 4.88e-4 | 8.84e-5 | 2.62e-3 | 5.06e-4 | [0.978, 1.029] |
| | L2 | 3.78e-4 | 7.35e-5 | 2.12e-3 | 4.82e-4 | [0.983, 1.023] |
| W7-X (16,32,32) p3 | interp | 1.52e-4 | 1.34e-5 | 6.78e-4 | 1.02e-4 | [0.983, 1.020] |
| | L2 | 1.53e-4 | 1.16e-5 | 5.49e-4 | 9.67e-5 | [0.983, 1.020] |
| QA (12,24,12) p3 | interp | 3.92e-4 | 1.06e-4 | 3.89e-3 | 1.22e-3 | [0.986, 1.020] |
| | L2 | 2.32e-4 | 7.81e-5 | 3.65e-3 | 1.17e-3 | [0.985, 1.021] |
| QA (12,24,12) p4 | interp | 1.45e-4 | 3.68e-5 | 1.88e-3 | 4.37e-4 | [0.988, 1.016] |
| | L2 | 1.16e-4 | 3.23e-5 | 1.65e-3 | 4.25e-4 | [0.986, 1.019] |
| QA (16,32,32) p3 | interp | 2.12e-5 | 1.91e-6 | 1.88e-4 | 4.72e-5 | [0.993, 1.010] |
| | L2 | 2.18e-5 | 1.47e-6 | 1.91e-4 | 4.70e-5 | [0.993, 1.011] |
| GVEC (15,24,24) p5 | interp | 7.90e-4 | 9.40e-5 | 7.09e-3 | 9.54e-4 | [0.993, 1.008] |
| | L2 | 6.55e-4 | 8.44e-5 | 5.87e-3 | 8.92e-4 | [0.994, 1.007] |

L2 is better or equal on every map gauge: max `|dX|` -40% at
(12,24,12) p3, -20..-25% at p4, rms `|dX|` -15..-30%, `|dDF|` -5..-20%; at
(16,32,32) the two coincide within 3% (no mode beyond Nyquist there, and
the within-Nyquist interpolant is already close to the projection). The
`det DF` ranges at the quadrature points are identical to three digits
(e.g. W7-X 12 p3: [9.13e-2, 1.737e1] vs [9.12e-2, 1.738e1]); no route
comes near folding, `set_geometry` and `build_preconditioners` take both.

**Pipeline** (`set_map`, `build_preconditioners`, `compute_nullspaces`,
Clebsch potential IC, `||F||_M` at `||B||_M = 1`; QA also
`||B_hat -+ h_hat||_M` against the k=2 Dirichlet harmonic form):

| case | route | ||F||_M | QA harmonic dist. | nullspace solve |
|---|---|---|---|---|
| W7-X (12,24,12) p3 | interp | 2.579e-3 | | 50 s |
| | L2 | 3.004e-3 (+16%) | | 34 s |
| W7-X (12,24,12) p4 | interp | 3.988e-3 | | 79 s |
| | L2 | 3.174e-3 (-20%) | | 62 s |
| W7-X (16,32,32) p3 | interp | 4.737e-4 | | 127 s |
| | L2 | 4.994e-4 (+5%) | | 110 s |
| QA (12,24,12) p3 | interp | 1.057e-2 | 5.67e-4 | 55 s |
| | L2 | 1.059e-2 (0%) | 3.73e-4 (-34%) | 38 s |
| QA (12,24,12) p4 | interp | 9.574e-3 | 5.10e-4 | 82 s |
| | L2 | 8.238e-3 (-14%) | 3.43e-4 (-33%) | 64 s |
| QA (16,32,32) p3 | interp | 1.227e-3 | 2.26e-4 | 133 s |
| | L2 | 1.190e-3 (-3%) | 2.18e-4 (-4%) | 116 s |

The interp values reproduce the 2026-08-27 gate numbers exactly (W7-X
2.579e-3, QA 1.057e-2 and 5.67e-4), as they must. The IC force is NOT a
map gauge: it moves both ways by up to 20% (W7-X p3 +16%, p4 -20%), and
under interpolation it does not even improve from p3 to p4 at fixed `ns`
(2.58e-3 -> 3.99e-3) while under L2 it is flat (3.00e-3 -> 3.17e-3); at
the finer rung the two agree within 5%. The vacuum distance to the
harmonic form -- the one clean gauge of the map plus IC together --
favours L2 by a third at (12,24,12) and by 4% at (16,32,32). `div B` is
1e-15 on every row. The nullspace-solve times are dominated by JIT
caching (the second route of each job reuses the first's executables),
not by the route; the iteration counts in the reports are the same to
within one.

**Axis.** `det DF / rho` over 64 theta at `rho = 1e-5` (min, max):

| case | series | spline map (interp) | spline map (L2) |
|---|---|---|---|
| W7-X wout | (11.574, 13.801) | (12.7592, 12.7596) | (12.7607, 12.7611) |
| QA wout | (0.5077, 0.5342) | (0.5174, 0.5174) | (0.5174, 0.5174) |
| GVEC state | (13.0021, 13.0022) | (13.0045, 13.0046) | (13.0050, 13.0051) |

The GVEC series is C1 at the axis as analysed (theta-spread 1e-5); both
wout series carry a cone (+-9% for W7-X, +-2.5% for QA: the `m >= 2` slope
the refit does not pin, and VMEC's own near-axis `m >= 2` data). Both
spline routes are C1 (spread 3e-5) -- the ring-1 surgery does that
regardless of the route, and the axis value of `det DF / rho` sits at the
series' theta-mean. This is the polar-space behaviour the brief flagged as
the risk; it is the same for every route and it is what the spline map
is for.

**Wall.** `det DF` at `rho = 1.0` exactly versus `1 - 1e-9`: spline maps
16.32 / 16.32 (W7-X), 0.4279 / 0.4279 (QA) -- no wall artefact through
`evaluate_local`; the series 8.16 / 16.33 and 0.2141 / 0.4282 -- exactly
half, the `jnp.clip` tie gradient. The clip is removed (cfe082f): the
series then gives 0.4282 / 0.4282 (QA re-run, `study_final_QA_12_p3.log`);
`test_state_field_wall_derivative_is_the_left_limit`.

**Final code path.** With the interpolant deleted the QA (12,24,12) p3
re-run reproduces the L2 rows above exactly (`||F||_M` 1.0593e-2,
harmonic distance 3.7296e-4, `|dX|` max 2.316e-4), and
`test/test_vmec.py` gates pass under `MRX_WOUT_GATES=1`.
`test/test_synthetic_gvec.py::test_relaxation` fails its GPU-determinism
band (`|half.dt - check.dt| <= 1e4 eps dt`, measured 1.5e-12 .. 2.8e-11
at `dt = 0.021`) on this branch AND on the base commit d2bd6a5 in a
detached worktree (`base_relax_test.log`): pre-existing, unrelated to the
map; the band was calibrated at `dt = 1.17`.

**Exactness rung.** GVEC state at `(15, 24, 24) p = 5`: the map's radial
knots are GVEC's (`[0]*5 + linspace(0, 1, 11) + [1]*5`), so the radial
direction is exact (`test_radial_coefficients_are_exact_on_the_states_knots`
for the mechanism) and the 7e-4 m map error is the angular projection
alone (`m` up to 11 on 24 points, `|n/nfp|` up to 12 = Nyquist on 24).
`||F||_M` 2.97e-3 (interp), nullspace 167 s.

## 5. Recommendation

**Keep the L2 projection; the interpolant is deleted.** Both are closed
forms built from the coefficients with no grid; the projection is the
one that is optimal in a norm, damps what the mesh cannot carry instead
of aliasing it, and is better or equal on every direct gauge (map and
`DF` against the series, the vacuum harmonic distance). The only gauge
that moved against it is one IC-force number (+16% at W7-X (12,24,12)
p3) that moves -20% the other way at p4 and +5% at the finer rung: not a
map gauge, and within what a force at t = 0 is worth. Cost is a second
per map. The documented 2026-08-27 W7-X/QA gate numbers shift
accordingly (W7-X (12,24,12) p3 IC force 2.58e-3 -> 3.00e-3, QA 1.057e-2
-> 1.059e-2, QA harmonic distance 5.67e-4 -> 3.73e-4); the gates
(`< 0.1`) are untouched.

What stays as it was: the map is a C1 polar spline with autodiff `DF`;
the polar restriction is the one every 0-form interpolation applies; the
flat-schema `.h5` route still bridges its grid linearly to the Greville
points and fits (`seq.interpolate`) -- it has no coefficients to project,
and it remains the documented fallback with its 3.4% linear-bridge floor.
`build_gvec_map(path, seq, ...)` now takes the sequence the map serves;
the second `DeRhamSequence` is gone.

Open: the L2 projection is onto the tensor space, then restricted; the
projection onto the polar space itself (`E M E^T`, a 3-D solve touching
rings 0 and 1 only) would be the consistent object. Rings 0 and 1 differ
between the two by `O(h^(p+1))` (the ring-0 content of the `m > 0` modes
the surgery removes); not measured separately. And a wout's `m >= 2`
axis slope (the cone in the series, absent from the GVEC state) could be
pinned in the refit (`mrx/vmec.py`) the way GVEC pins it -- a separate,
small change to the reader, not to the map.
