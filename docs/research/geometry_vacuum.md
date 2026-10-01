# Geometry, readers, vacuum fields: measurements

Readers, map construction and the IC routes are described in the Sphinx docs. This file keeps the
convergence data and the measured traps.

## 1. Conventions verified against GVEC / VMEC

- Our reference 2-form components ARE GVEC/VMEC's sqrt(g) B^i: rebuilt from three scalars on hegna,
  sqrt(g) B^rho 3.8e-16, B^theta ratio 1.00000000 (std 2.9e-13), B^zeta ratio 1.0 (1.7e-16)
  (2026-08-25).
- GVEC unit conversions (measured by FD of LA: 6.274 vs 2pi, 2.0905 vs 2pi/nfp): Psi' = 2pi dPhi_dr,
  iota = (1/nfp) dchi_dr / dPhi_dr, lambda = LA / 2pi. Without 1/nfp iota is nfp times too large. On
  W7-X Clebsch exports the ratio reconstructed/file iota is +0.2000 at every radius. fmm002's
  dchi/dPsi is negative (-0.93..-1.05) and is tracked with its sign. dchi/dPsi is a flux function to
  6.7e-16.
- Store LA (the scalar), not its two derivatives: div B = 0 needs mixed partials from one
  interpolant. lambda is large: hegna max |LA| 31 deg, lam_chi up to 0.83, lam_zeta ~2x iota.
- Periodic endpoints: dropping the last point is right only for duplicated endpoints. Decide from
  data (|LA(0) - LA(1)| 6.9e-2 vs ~1e-16 for a duplicate).
- GVEC data defects (2026-08-25): axis_pert_* / interior_pert_*_dR5e-05_dZ3.75e-05.h5 declare nfp = 2
  but are quasr0044970 (nfp = 3) shifted by the filename amplitudes. w7x_ini_mrx.h5 declares
  axis_radial_index = 49, the axis is at rho[0]. w7x_ini_00000000 is GVEC's INITIAL GUESS (axis R =
  5.5 exactly), fmm002 a converged state (axis 5.5359528014861). The hegna export was not an
  equilibrium (|J x B|/|grad p| 0.042, 2 mu0 p / B^2 1.09 on axis) and is deleted.
- GVEC Clebsch vs our force operator: w7x_ini ||F||/||B|| 4.76e-2, pressure-shape residual 8.93e-2.
  fmm002 1.88e-2, 4.51e-2 (2026-08-25).

## 2. Spline map from series coefficients (2026-08-28, float64)

- The per-mode closed-form interpolant equals the sampled Greville fit to 5e-15..5e-14 (a cost gain
  only).
- L2 projection vs interpolant, max |dX| on 4000 points: W7-X (12,24,12) p3 5.75e-4 vs 9.67e-4 m, p4
  3.78e-4 vs 4.88e-4, (16,32,32) p3 1.53e-4 both. QA (12,24,12) p3 2.32e-4 vs 3.92e-4, (16,32,32) 2.2e-5
  both. GVEC (15,24,24) p5 6.55e-4 vs 7.90e-4 (radial direction exact on GVEC's own knots). L2 is
  better or equal on every map gauge (-40% max dX at coarse p3, equal where no mode exceeds Nyquist).
  QA vacuum harmonic distance -34% (5.67e-4 -> 3.73e-4) at (12,24,12). W7-X IC force 2.58e-3 ->
  3.00e-3 (the t = 0 force moves +-20% either way, not a map gauge).
- Aliasing: W7-X at (24,12) angular cells puts 138/288 R modes beyond Nyquist (1/sigma(N/2) = 3 at
  p = 3), none at (32,32).
- Non-uniform angular knots: the per-mode 1-D mass solve reproduces the circulant form to 3e-15
  (2026-09-10).
- Axis: det DPhi / rho over theta at rho = 1e-5: W7-X wout series 11.57..13.80 (+-9% cone), QA wout
  +-2.5%, GVEC state 1e-5 spread. Spline maps 3e-5 spread either route (C1 by the ring-1 surgery).
  GVEC pins coef[0] = 0 for m > 0 and coef[1] = 0 for m >= 2. The wout refit did not pin the m >= 2
  slope.
- Wall: a clip tie halves the autodiff derivative at rho = 1.0 exactly (det 8.16 vs 16.33). A clamped
  spline basis has gradient 0 at x = 1.0 exactly (half-open last piece). That was the "det DPhi = 0 at
  rho = 1" (and the Greville-layer zeros, 288/2880 on W7-X). Sample at 1 - 1e-7.
- Greville abscissae of a clamped basis include the endpoints where det DPhi = 0 (inf -> NaN -> OOM).
- Folds: rotating ellipse (kappa 1.5, nfp 3) at (8,16,24) has analytic det DPhi +0.26..+1.61 but the
  spline projection folds (projected profile in [-25.6, -0.159]) (2026-08-17). quasr0065575 at
  (12,24,12) det DPhi in [-0.236, +1.543]. quasr0065530 builds at (8,16,8) and (12,24,12) p=3, folds at
  (16,32,16) and at p = 4 (2026-08-25).
- Metric non-orthogonality (from R, Z only): |g_rt|/sqrt(g_rr g_tt) quasr9983 0.54-0.64, quasr44970
  0.87-0.91, W7-X 0.88-0.93 (rho 0.2 -> 0.86). g_rz, g_tz 0.18-0.51. Face weight J g^rr at r = 1,
  modes for 99% / 99.9% energy: toroid 3 / 3, rot-ellipse 9 / 25, W7-X 11 / 39 (26,11). W7-X mass
  coupling |g^ab|/sqrt(g^aa g^bb) = 0.39 / 0.51, ~5.5x the boundary tilt |g^rb|/g^rr (2026-08-22/25).
- The rotating-ellipse map pulsates (axis-aligned ellipse, Z -> -Z symmetric): its iota is exactly 0.

## 3. QA vacuum: VMEC wout vs the discrete harmonic 2-form (2026-08-28 .. 30)

Setting: Landreman-Paul 2021 QA lowres wout (mpol = ntor = 8, ns = 75), float64. D = distance of
the fitted wout field B_w to the mesh's harmonic 2-form on a common (192,288,144) evaluation grid.

- The scale fit c equals the flux match to 1e-11 (a theorem of the discrete Hodge decomposition).
- D floors at ~8e-5 global (~6e-5 bulk) for every p >= 2: the low-res reference's own truncation (the
  high-res reference floors at 4.3-4.6e-5, flat over n = 29, 37, 45 and p, 2026-09-09).
- Dense grid D by n_el:

  | p | 5 | 9 | 13 | 21 | 45 |
  |---|---|---|---|---|---|
  | 2 | 1.89e-3 | 3.95e-4 | 1.42e-4 | 8.09e-5 | 8.40e-5 |
  | 3 | 2.24e-3 | 3.82e-4 | 1.30e-4 | 8.36e-5 | 8.41e-5 |
  | 4 | 1.31e-3 | 3.60e-4 | 9.29e-5 | 7.97e-5 | 8.42e-5 |

  Pre-elbow slope p2 2.91, p3 3.13 (p4 floors too early to fit). Elbow at n_el 13 (p = 2, 3), 11
  (p = 4). A least-squares slope through the plateau inverts the p-ordering: read the pre-floor rate.
- Per-p self-convergence of B_w vs its own n_el = 45: slopes 0.89 / 1.67 / 2.83 / 3.46 for p = 1..4
  (O(h^p)). The map converges O(h^{p+1}) (slope 4.06 at p = 3).
- What floored D before the reader fixes ((24,48,24) p3): 3.92e-4 (sampled map) -> 3.98e-4 (lambda
  pinned: not it) -> 3.97e-4 (L2 map) -> 2.34e-4 (axis parity per mode) -> 8.4e-5 (every derivative
  of order < m or of wrong parity removed). ||J||(B_w) 0.374 -> 0.056, ||F||_M 1.98e-3 -> 6.7e-4. The
  remaining residual is axis-localised (max 7.9e-4).
- Representation independence: the k=1 free form h1 and the k=2 Dirichlet form h2 are one vacuum
  field (M-cosine 0.99999999). The scale-fitted lab-frame residual is O(h^p) (slope 2.82 at p3) down
  to 7.4e-5 at n_el 25, then rises to 1.7e-3 at n_el 45 because the k=1 free solve degrades
  (solvers_precision.md section 4). Iota of B_w = VMEC's iotaf to ~1e-5 (tracer limit). iota of h
  converges ~O(h^2-3).
- p = 1 passes the harmonic gate (2e-13) and converges O(h) (D 1.26e-1 -> 2.38e-2 at n_el 5 -> 29,
  bulk ~O(h^2)) but loses the axis field (|B| ~4e-4 T on axis vs ~1 T): the gate certifies
  harmonicity, not resolution.
- At fixed (24,12) angular cells the p-scan is angular-limited (D 3.95e-4 / 3.82e-4 / 3.60e-4 for
  p = 2/3/4): scale the angular cells with p in any p-study.
- The common evaluation grid must out-resolve the finest rung: (48,96,48) read the finest pair ~7%
  high. (192,288,144) is within 1.5% of 2x finer.

## 4. Analytic QA vacuum (paper, float64, n = 8..64, 2026-09-11)

- Fitted rates 0.956 / 1.975 / 3.017 / 3.889 and 0.942 / 1.973 / 2.931 / 3.925 for p = 1..4 (a caption
  value 4.07 came from rows that differ from the data).
- The three symmetry models give identical errors (2.379e-3 / 4.244e-3 at 26 splines, 1.364e-3 /
  2.426e-3 at 34) (2026-09-18).

## 5. W7-X vacuum from the harmonic form (2026-08-17/24)

- Against GVEC's 32^3-sampled field it converges ~O(h^4) (4.45e-3 at n = 8 -> 3.5e-4 at n = 16,
  (n,2n,2n) p3, k=1 form 3.85e-3 -> 2.53e-4) and floors at ~2e-4 = the sample's (a 25^3 subsample
  floors at ~2e-3). Traced iota 0.851-0.948, the standard-configuration vacuum range. Exactly one
  resonance in range (10/11).
- k=2 / k=1 harmonic-form angle converges on the MEDIAN over 512 points (0.0471 / 0.0153 / 0.0113 at
  ns 8 / 12 / 16, 0.0069 at 12 p4). The max is set by one sample near r -> 1.
- Vacuum-file validity: the two quasr GVEC h5 "equilibria" are not curl-free (harmonic error flat at
  1.3-2.4% under refinement while the projection keeps falling). Both simsopt B files are rotated
  exactly one field period off their own R, Z (88% -> 1.09%, 21.8% -> 1.08% after counter-rotation),
  then floor at 1.05-1.07% flat over 6x resolution. R, Z checks and L2 projections pass on the
  corrupted file. Only the harmonic comparison (energy fraction 0.226 vs > 0.999) and the rotation
  scan catch it. The cylindrical-fraction test is blind at nfp = 2.

## 6. Initial conditions

- L2 route and B^rho = 0: exact (1e-16) iff g_rho-chi = g_rho-zeta = 0. Shaped stellarators:
  per-surface B^rho/B^zeta grows 1.4e-4 (rho 0.05) -> 3.2e-3 (0.89), bulk max 6.8e-3. The toroid
  (diagonal g) still leaks 4.6e-9 / div B 9.4e-8 with iota != 0 (weights g_ii/J spread 17-67%,
  mechanism unverified).
- Clebsch IC L2-projected through M_2 reintroduces ||div B|| 2.7e-2 at (8,16,8) (70x the logical IC's
  3.7e-4). One Leray cleaning -> 6.5e-14, stays ~1e-12 (2026-08-25).
- Potential IC B = dA' (2026-08-26): div B 1e-16, B^rho = 0 on every rho-face exactly. Tracer h/2
  drift 2.8e-6 vs 3.4e-5 for the L2-projected IC. Better on every gauge at 50^3 / 33^3 / 20^3. The
  20^3 potential IC relaxes to the 50^3 core.
- Near-harmonic fields: quasr44970 logical IC 97.4% harmonic, fmm002 99.99%, li383 96.4%
  (||B - ch||/||B|| 0.036, c 0.9994). Harmonic remainder from the k=1 Hodge split and alignment with
  the nullspace vector agree to 5 digits (0.682018 on the dzeta IC).
- Helicity closed form for the logical ansatz: H = int (Psi X' - X Psi') drho, metric-free, zero at
  constant iota. For Psi' = rho^q, iota = iota0 + Diota rho^e: H = Diota e / [(q+1)(q+e+1)(2q+e+2)],
  verified to 1e-13 over six shapes. For constant-iota fields the computed helicity is entirely the
  harmonic gauge term (+7.81 vs analytic 0). A pure harmonic field returns exactly 0.
- The helicity rhs was once assembled in the wrong space (primal weak curl instead of dual): H wrong
  by ~1150x, reproducible to 8e-13. Caught by ||B_harm|| <= ||B|| (85.6 is not a fraction)
  (fixed 2026-08-25).
- Screw pinch manufactured equilibria (cylinder, polynomial profiles, lambda = 0): p vs closed form
  1.8e-3 / 9.0e-4 / 4.8e-3 (sheared / flat / q2), force ~1e-12, B^rho ~1e-16, div B ~3e-14. Control
  iota = 0: force 7.6e-15. The exact balance includes the tension term (33-49% of dp/drho).
- Toroid: lambda = 0 has no 1/R in B_phi. 1 + lam_chi = <1/R>^-1 / R cuts the force 8.2x (9.0e-2 ->
  1.1e-2, remainder = Shafranov shift). Vacuum needs also Psi' = rho <1/R>. A small force with iota =
  0 meant the residual was a pure gradient, not an equilibrium.
- Lambda equation at fixed geometry (dW/dlambda = 0 <=> (curl B).grad rho = 0, per surface): toroid vs
  closed form 1.3e-8. Hegna vs GVEC's own lambda corr +0.9984 (lam_chi) / +0.9992 (lam_zeta), median
  residual 0.061 / 0.045. Edge (rho 0.95) lam_zeta residual 0.378 unchanged at 2.8x coefficients: model
  error of frozen surfaces, not truncation.
- lambda changes compute_helicity by 3.5% on fmm002 (-1.7806e-4 vs -1.8434e-4): a different
  functional from the natural-gauge helicity.
- Relative cohomology of the Dirichlet complex: b_k^rel = b_{3-k}^abs, so b2^rel = 1.

## 7. Discretisation checks

- Poisson manufactured solutions (toroid, n = 6/8/10, p = 3, reproduced 2026-08-26):

  | case | n=6 | 8 | 10 | order |
  |---|---|---|---|---|
  | k0 | 2.072e-3 | 4.771e-4 | 1.737e-4 | ~4.5 |
  | nbc_k1 | 8.564e-3 | 3.244e-3 | 1.576e-3 | ~3.3 (open) |
  | dbc_k2 | 1.700623e-3 | 4.601128e-4 | 1.760652e-4 | 4.55 / 4.31 |

  dbc_k2 matches the projection error of the exact field to 4 digits. Load convention: k=1 source
  G^-1 f, k=2 (g/J) f. nbc k0 ||curl|| 1e-16. k=1/k=2 pullbacks max rel err 1.8e-15 over 4 (k,BC) pairs
  on 3 geometries. Stored harmonic forms Rayleigh 1e-26 on 6 geometries (2026-08-25).
- Manufactured-solution generators (toroid map R = 1 + eps r cos(2pi chi), g = diag(eps^2,
  4pi^2 eps^2 r^2, 4pi^2 R^2), J = 4pi^2 eps^2 r R, Hodge star 1 -> 2: *a = (4pi^2 r R a_r, R/r a_c,
  eps^2 r/R a_z), 2-form slots (cz, rz, rc)). All four paired by *, NBC <-> DBC. Verified by FD
  Laplace-Beltrami to ~1e-14:
  1. u = cos(2pi z), f0 = cos(2pi z)/R^2 [k0 NBC, k3 DBC with f3 = f0 J].
  2. u = cos(pi r^2/2), f0 = (2pi s + pi^2 r^2 c)/eps^2 + pi r s cos(2pi chi)/(eps R), s, c = sin, cos
     of pi r^2/2 [k0 DBC, k3 NBC].
  3. w1 = cos(2pi z) dz, sigma = sin(2pi z)/(2pi R^2), f1 = grad sigma. w2 = *w1 = (0, 0,
     eps^2 r cos(2pi z)/R), f2 = *f1 [k1 NBC, k2 DBC].
  4. w1 = c(r) cos(2pi z) dz, sigma = c sin(2pi z)/(2pi R^2), f1 = d sigma + delta d w1 with covariant
     f1_r = -eps C c S/(pi R^3), f1_c = 2 eps r Sc c S/R^3, f1_z = Z [c/R^2 + (2pi s + pi^2 r^2 c)/eps^2
     - pi r s C/(eps R)] (C, Sc = cos, sin(2pi chi), S, Z = sin, cos(2pi z)). w2 = *w1, f2 = *f1
     [k1 DBC, k2 NBC].
- C^2 polar k=0 (toroid dbc p=3, m = 0/1/2): C^1 and C^2 L2 errors identical to printed digits, rates
  4.44 / 4.32 / 4.39. C^2 has 10-16% fewer DoFs. Pole Taylor remainder order 3.00 vs < 2 for C^1
  (2026-07).
- Extraction is not biorthogonal (2026-08-25): ||E E^T - I||_max 1.556 at k=1 (30 of 606 rows),
  0.352 at k=2 (12 of 588), 0 at k=3: exactly the polar-ring rows, one dense block per zeta slice.
  Restriction (E E^T)^-1 E: k=0 round trip 5.29e-1 -> 2.8e-16. All k ~1e-16 at both parities after the
  seam fix.
- Histopolation failure modes (2026-08-25, all fixed): periodic Greville spans unsorted after mod 1
  (widths -0.83 / +1.17 at p >= 2). Even-p spans straddle knots (Gauss inexact at any order: 40 points
  leave 3.6e-7, split at knots). The even-p last periodic span crosses the seam and was evaluated
  unwrapped -> k >= 1 round trip 7e-2..1.3e-1 (analytic seam-row defect 0.125 at p = 2, n = 4). Odd-p
  exactness was a rounding accident at non-power-of-2 n (8.3e-2 at n = 6, p = 3). Physical pullbacks:
  k=1 omega = DPhi^T v, k=2 omega = adj(DPhi) v from cofactors (det * inv is 0 * inf on the axis).
- Polar histopolation: E . Pi_full is not a projector on the polar space (k=0 round-trip error 0.53
  free / 0.36 dbc) although smooth-function accuracy is fine (2.2e-2) (2026-08-25).
- DoF counts: polar (n,2n,2n) p=3, betti (1,1,0,0): V2 Dirichlet = 12n^3 - 24n^2 + 4n, V1 = V2 + 6n
  (2026-08-17). Odd-parity Dirichlet 2-form (m = n_r + 2): full = 12m^3 - 100m^2 + 260m - 216, the
  reflection fixes 4n_r - 12, odd = (full - (4n_r - 12))/2: 20998 / 74886 / 182278 / 631302 / 1515526
  at n_r = 16/24/32/48/64 (2026-09-22).

## 8. Stellarator symmetry: half-period quadrature and parity reduction

- Half-period quadrature vs a full-period twin (li383 (8,12,12) p=2 float64, 2026-09-17): M_k, P_kl,
  D_k, D_k^T, S_k, L_k agree to 1e-14 on vectors of definite parity. Mass solves to 1e-13 with equal
  counts (k = 0..3: 14, 57/62, 54/61, 7). k=0 Laplacian 44/73 vs 44/72. Harmonic forms 2e-15..7e-15. IC
  parity discarded 3e-8. Gate li383 (16,32,32): a reconnection identical to every printed digit,
  energy and helicity to 1e-12.
- Its failure modes: the metric-lumping atoms are not reflection-equivariant on the polar rows (M_1,
  M_2, L_1, L_2 solves ran to 10000 it until wrapped Pi P Pi^T, wrapped atoms differ by 1% mass, 2-6%
  Laplacian). Primal- and dual-purity differ on the polar rows (E E^T != I), so a primal-pure random
  rhs stalls CG. Round-off impurity of residuals built by cancellation took the k=1 Hodge inner CG to
  10152 it (357 full) with a 40% impure answer until every solve projected rhs and residuals (then 314
  vs 357 it, 6e-14).
- Parity as a DoF reduction (2026-09-20/21): n(2, dbc) 720 -> 354 (odd, B) / 366 (even, u) at (6,8,8).
  Odd-view mass, Laplacian and Leray solves agree with the base to 1e-7..2e-6 at equal counts (61/60,
  40/40, 32/32), harmonic 2-form 4e-7. The core parity basis is a choice (SVD of the eigenspace), so
  working view and float64 twin must share one build (independent builds made every polar-core solve
  garbage while k=3 was exact). A known base coefficient vector must be reduced through X^T (the
  constant has sqrt 2 on every orbit pair: 37% -> 2.8% error). X^T X = I holds, E_red E_red^T = I does
  not (polar core).
