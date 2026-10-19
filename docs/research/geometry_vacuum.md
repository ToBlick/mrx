# Geometry and vacuum fields: measurements

Checks against GVEC, VMEC and analytic fields, the map and initial-condition choices, the symmetry reductions and
the graded spaces.

## 1. Conventions verified against GVEC / VMEC (2026-08-25)

- Our 2-form components are GVEC's and VMEC's sqrt(g) B^i. Rebuilt from three scalars on hegna: sqrt(g) B^rho
  3.8e-16, B^theta ratio 1.00000000, B^zeta ratio 1.0.
- GVEC units, by finite differences of LA: Psi' = 2pi dPhi_dr, iota = (1/nfp) dchi_dr / dPhi_dr, lambda = LA / 2pi.
  Without the 1/nfp iota is nfp times too large.
- GVEC Clebsch states under our force operator: w7x_ini ||F||/||B|| 4.76e-2, fmm002 1.88e-2.
- The GVEC closed-form state against the gridded export, fmm002 (16,32,32), 300 steps (2026-08-26): ||F||_0 7.5e-4
  against 7.1e-3, energy released 4e-8 against 1.4e-5. The gridded route was deleted.

## 2. Spline map from series coefficients (2026-08-28)

- L2 projection against the interpolant, max |dX| on 4000 points: W7-X (12,24,12) p = 3 5.75e-4 against 9.67e-4 m,
  equal at (16,32,32). L2 is better or equal on every map gauge. The QA vacuum distance falls 34 % at (12,24,12).

## 3. QA vacuum: VMEC wout against the discrete harmonic 2-form (2026-08-28 .. 30)

Landreman-Paul 2021 QA lowres wout (mpol = ntor = 8, ns = 75), float64. D = distance of the fitted wout field to the
mesh's harmonic 2-form on a (192,288,144) evaluation grid.

| p | n_el 5 | 9 | 13 | 21 | 45 |
|---|---|---|---|---|---|
| 2 | 1.89e-3 | 3.95e-4 | 1.42e-4 | 8.09e-5 | 8.40e-5 |
| 3 | 2.24e-3 | 3.82e-4 | 1.30e-4 | 8.36e-5 | 8.41e-5 |
| 4 | 1.31e-3 | 3.60e-4 | 9.29e-5 | 7.97e-5 | 8.42e-5 |

- D floors at about 8e-5 for every p >= 2, the low-res reference's own truncation. The high-res reference floors at
  4.3-4.6e-5 (2026-09-09).
- Pre-floor slopes 2.91 at p = 2 and 3.13 at p = 3. The map converges as h^{p+1}, slope 4.06 at p = 3.
- Iota of the wout field equals VMEC's iotaf to about 1e-5, the tracer limit.
- The k = 1 free and the k = 2 Dirichlet harmonic forms are one vacuum field, M-cosine 0.99999999.

## 4. Analytic QA vacuum (paper, n = 8..64, 2026-09-11)

- Fitted rates 0.956 / 1.975 / 3.017 / 3.889 and 0.942 / 1.973 / 2.931 / 3.925 for p = 1..4.
- The three symmetry models give identical errors, 2.379e-3 / 4.244e-3 at 26 splines (2026-09-18).

## 5. W7-X vacuum (2026-08-17 .. 24)

- Against GVEC's 32^3-sampled field the harmonic form converges as about h^4, 4.45e-3 at n = 8 -> 3.5e-4 at n = 16,
  and floors at 2e-4, the sample's own floor. Traced iota 0.851-0.948, the standard-configuration vacuum range.
- Both simsopt B files were rotated exactly one field period off their own R, Z (88 % -> 1.09 % after
  counter-rotation). R, Z checks and L2 projections passed on them. Only the harmonic comparison caught it.
- The two quasr GVEC h5 equilibria are not curl-free. Their harmonic error stays flat at 1.3-2.4 % under refinement.

## 6. Initial conditions (2026-08-25/26)

- The potential IC B = dA' has div B 1e-16, B^rho = 0 on every rho face, and a tracer h/2 drift of 2.8e-6 against
  3.4e-5 for the L2-projected IC. The Clebsch IC L2-projected through M_2 had ||div B|| 2.7e-2 at (8,16,8).

## 7. Stellarator symmetry (2026-09-17 .. 21)

- Half-period quadrature against a full-period twin, li383 (8,12,12): M_k, D_k, S_k, L_k agree to 1e-14 on vectors of
  definite parity. Mass solves agree to 1e-13 with equal counts.
- The metric-lumping atoms are not reflection-equivariant on the polar rows. M_1, M_2, L_1, L_2 solves ran to 10000
  its until the atoms were wrapped in the parity projector. Wrapped atoms differ by 1 % for the mass and 2-6 % for the
  Laplacian.
- Parity as a DoF reduction: the Dirichlet 2-forms at (6,8,8) go 720 -> 354 odd and 366 even. Odd-view mass,
  Laplacian and Leray solves agree with the full space to 1e-7..2e-6 at equal counts.

## 8. Graded polar spaces (2026-10-17)

- li383 (18,32,12), odd view: Dirichlet 1-form DoFs 17724 -> 12828. Its uniform -> graded with the level-block atoms:
  S_0 77 -> 69, S_1 co-exact 111 -> 103, L_1 693 -> 602, S_2 99 -> 74, L_2 751 -> 762, M_1 51 -> 55.
- GRADED_ASPECT = pi / 2 and GRADED_THETA_MIN = 8 are not tuned. With such a rule the stiffness scale lam_max(M^-1
  S) h_min^2 stays at about 23 under every refinement of a 2-D disc at p = 2, against 137 -> 8580 for the uniform
  polar mesh from 16x32 to 128x256 (prototype, `outputs/multipatch/graded/README.md` 2, 2026-10-13).
