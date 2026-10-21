# Solvers and precision: measurements

Counts and timings of the solve routes in use and of the routes they replaced, and the precision limits behind the
tolerances. SOLVE_TOL = 1e-8. There are two precisions: float64, the default, and float32 with double-single
residuals for a device without float64 (section 8). Rows marked mixed (float64 residual, float32 iterations) or
plain float32 are the record of the path PR #33 removed (2026-10-14).

Setting of the 2026-10 records: li383 p = 2, float64, one H200, odd view. M = (48,64,24), L = (64,80,32), graded
from 2026-10-17 on.

## 1. Hodge split against saddle MINRES for L_1, L_2 (QA p = 3, tol 1.5e-8, 2026-09-02)

x_perp iterations [all solves of the split] and wall time:

| case | saddle MINRES | split |
|---|---|---|
| n=16 k1 dbc | 1022 / 2.7 s | 307 [450] / 3.1 s |
| n=16 k1 free | 2584 / 7.7 s | 1458 [1707] / 5.5 s |
| n=16 k2 dbc | 2555 / 8.8 s | 367 [987] / 4.7 s |
| n=16 k2 free | 7122 / 9.6 s | 1636 [4415] / 9.5 s |
| n=24 k1 dbc | 1553 / 7.1 s | 418 [602] / 4.1 s |
| n=24 k1 free | 3889 / 26.2 s | 2174 [2481] / 14.1 s |
| n=24 k2 dbc | 4556 / 35.5 s | 500 [1337] / 8.6 s |
| n=24 k2 free | cap 10000 / 35.2 s | 2390 [6485] / 29.9 s |

- The split gains 1.5-3.4x in iterations and 1-4x in wall time.
- PCG on the singular S_k failed then. Its right-hand side was consistent only to the tolerance of the g solve, and
  the residual reached 1.8e-5 at 80 its, 1.1 at 160 and 2e9 at 320. Section 3 makes it work with tighter inner
  tolerances.
- k = 3 by two L^_2 solves (QA n = 16) took 368 + 403 against 694 its, but left the exact residual at 3.8e-6 against
  8.7e-8. k = 3 stayed on saddle MINRES until 2026-10-18.
- CG on L_2 with a nested mass CG (2026-08-24): 993 its but 1007 s against 30 s.
- Free-BC solves end with errors 1e-5..1e-4 at tol 1.5e-8, err / res about 5000 from near-harmonic modes. Dirichlet
  solves have err / res about 1.

## 2. L_3 and Leray by the Hodge split (2026-10-18)

- The exact part by stiffness solves only, a gauged S_1 solve then the co-closed S_2 by multigrid, instead of the hat
  L^_2: L_3 on M 442 its / 299 ms -> 156 / 61.6 ms, on L 556 / 687 -> 165 / 113.7 ms. The S_1 gauge is 85 % of the
  iterations.
- Without the gauge the S_2 solve diverges. A gauge stopped relative to its own right-hand side at 1e-2 tol
  converged on (18,32,12) and diverged on M.
- L_3 as the Hodge split replaced saddle MINRES. With the multigrid, M 565 its / 289 ms -> 116 / 40.0 ms, L 607 /
  482 -> 125 / 67.6 ms. The solutions agree to 1.3e-9 and 5e-10.
- Leray of a field whose divergence-free part is 5 % of it, M: 247 -> 39 ms, |div F| / |F| 9.1e-5 -> 2.5e-9. Saddle
  MINRES did not drive the divergence residual to the tolerance.
- A Leray by co-closed solves without a pressure (2026-10-10) was not adopted. Its |div F| / |F| was 2.8e-8 against
  1e-12 for the saddle in float64.

## 3. Hodge split by stiffness solves in one pass (2026-10-18)

- L_1, L_2 by solves of S_{k-1}, S_k, S_{k-1} alone instead of the hat L^_k, with the S_2 multigrid. Time in ms of
  the hat split in two passes / in one pass -> the stiffness split: M L_1 678 / 345 -> 93.6, L_2 1786 / 926 ->
  132.4. L L_1 1435 / 736 -> 180, L_2 3828 / 2020 -> 250. Counts after: 279 / 363 on M, 300 / 386 on L.
- With the default atoms and no multigrid on M: L_1 93.8 ms, L_2 149.4 ms.
- The S_1 solves are 75-97 % of each split. The error on smooth right-hand sides is 1.3e-10 for L_1 and 4.2e-10 for
  L_2.
- Shifted solves on M, k = 1, 2, eps 7.4e-6 to 1, are unchanged within 0-5 %.
- Inner stops that failed: x_perp stopped on its own right-hand side diverged in the second pass of L_2 on M. z
  stopped at tol / 10 on its own right-hand side ran to maxiter on a co-exact b. Without the S_0 gauge at k = 2 the
  solve for a stalls.
- Weighting the exact-form term of L^_k by gamma < 1 was worse (2026-10-09). 24x42x42, li383 / W7-X: 467 / 384 its
  at gamma 1, 698 / 633 at 0.1, no convergence at 0.001. gamma = 0, the singular S_k on a consistent right-hand
  side, took 147 / 148.

## 4. Shifted split for M_k + eps L_k

2026-09-02, li383 p = 3, k = 2 Dirichlet, tol 1.5e-8. Iterations of MINRES with the mass atom / MINRES with 1/eps P_L
/ the split:

| mesh | eps | MINRES, mass atom | MINRES, P_L | split |
|---|---|---|---|---|
| (8,16,8) | 1e-6 | 497 | 3623 | 100 |
| (8,16,8) | 1e-3 (0.064/n_r^2) | 2134 | 1675 | 330 |
| (8,16,8) | 1e-2 | 5798 | 943 | 808 |
| (12,24,12) | 4.44e-4 (0.064/n_r^2) | 8478 | 3650 | 772 |
| (16,32,16) | 2.5e-4 (0.064/n_r^2) | 20362 | 6397 | 1326 |

- The resistive MINRES the split replaced (W7-X fmm002 (8,16,8) p = 3, 2026-08-26/27) took 383 its per step at eta
  1e-3 and hit maxiter 10000 at eta >= 1e-1.

2026-10-14, uniform M, against PR #33:
- Gradient correction: a single CG on M_1 + eps S_1 took eps 1e-2 4228 -> 220 and eps 1 > 5000 -> 274. An exact
  0-form solve in its place gains little: 211 and 247.
- Three solves instead of two: with two, the error of z reached the residual, and at eps = 1 every pass gained only
  a factor 8. With three, each solve stops at tol / 3 and one pass suffices.
- Tried and dropped: judging each solve on its contribution to the posed residual (+50 %, an M_k apply per
  iteration), and the same inside refine (0.49 s/step).

seq.shifted(k, eps).solve on M_k z, iterations and time, PR #33 -> both changes:

| eps | k = 1 (even view) | k = 2 (odd view) |
|---|---|---|
| 7.4e-6 | 901, 0.36 s -> 548, 0.22 s | 1254, 0.48 s -> 1015, 0.44 s |
| 1e-4 | 1190, 0.47 s -> 491, 0.18 s | 1629, 0.60 s -> 631, 0.28 s |
| 1e-3 | 3032, 1.23 s -> 568, 0.21 s | 4321, 1.53 s -> 695, 0.30 s |
| 1e-2 | 15533, 6.3 s -> 622, 0.23 s | 23315, 8.0 s -> 821, 0.35 s |
| 1e-1 | unconverged, 24.8 s -> 703, 0.27 s | unconverged, 35.5 s -> 1205, 0.50 s |
| 1 | unconverged, 24.9 s -> 984, 0.40 s | unconverged, 43.5 s -> 2538, 0.99 s |

- Velocity smoothing as one co-exact solve of M_1 + mu S_1: gradient relaxation 0.260 -> 0.228 s/step with the same
  energy and residual. The full three-solve smoothing took 0.307 s/step.
- W7-X (32,64,36) and li383 M at p = 3: shifted solve at eps 1e-2, k = 1, 1.17 -> 0.15 s and 15.8 -> 0.47 s.
  Gradient relaxation 0.124 -> 0.096 and 0.901 -> 0.652 s/step.
- li383 free k = 1 at eps 1: 29 s unconverged -> 0.39 s.
- Resistive step as one 1-form solve (2026-10-15): 223 its against 1606 for the full M_2 + eps L_2 solve, |div delta|
  / |delta| 2.1e-12 against 6.4e-8. 30 driven Newton steps: resistive its 1565 -> 229, 3.95 -> 3.41 s/step.

## 5. Stopping rules and part solves (2026-10-12/13, uniform M and L)

- Each Krylov solve stops on dual_norm in one pass, instead of a second judged pass that ended at 1e-13. Its /
  s before -> after on M: L_1 force potential 1346 / 0.96 -> 768 / 0.57, L_3 3086 / 1.64 -> 1491 / 0.87. On L the
  force potential went 1644 / 2.18 -> 963 / 1.34.
- A part solve costs far less than the full solve of the same right-hand side on M: L_1 co-exact 0.78 -> 0.094 s,
  L_2 co-exact 2.08 -> 0.054 s, L_1 exact 1.77 -> 0.105 s.
- The force potential warm-started from the step before takes 199 against 235 its per solve on M, 237 against 286 on
  L.
- HARMONIC_TOL = 0.1 (2026-10-11): Rayleigh quotient of the harmonic 2-form solved at tol / at 0.1 tol, float64 M
  1.5e-14 / 1.6e-16. Plain float32 1.1e-3 / 4.7e-5.

Newton step on li383 M before and after these changes, steady over the later chunks: 3.85 -> 3.33 s/step
(2026-10-12). The relax.py totals of that day, 5.90 -> 5.35, include the compile of the first chunk.

## 6. Harmonic forms

- Seeds from histopolated constants reach Rayleigh / lambda_1 1e-11 for every QA form, against up to 4.5e-4 with the
  old M^-1 load seeds (2026-09-02).
- Judge a harmonic form by its Rayleigh quotient. Across four solver arms with fields equal to 5 digits, ||L v||
  spanned 1e-8..4.6e-3 while the Rayleigh quotient stayed near 1e-13 (2026-08-28).
- Inverse iteration (W7-X, p = 3, eps 1e-4, 2026-08-17) fails above n ~ 18: Rayleigh 2.3e-10 at n = 18, 1.7e-3 at 24.
  A 6-step polish of a computed form walks away, W7-X 3.0e-4 -> 1.4e-2.

## 7. Precision limits

- float32 attainable true residual on the test mesh (2026-09-05): 2.4e-4 on the k = 2 Hodge split, 4.2e-4 on the
  k = 3 saddle. 5 of 8 Poisson solves could not reach 1e-6.
- Plain float32 at tol 1e-5 on li383 (12,20,12) (2026-10-13) failed 7 of 46 tests. Leray's second pass ran to
  maxiter and returned NaN, L_3 took 50278 its to 1.1e-5, and the harmonic 1-form reached Rayleigh 1.3e-8 against
  1.2e-10.
- float32 floor of the QA vacuum distance D: 3.4e-4, against 4.5e-5..8e-5 in float64 (2026-08-28).
- TPU v5e matmul precision: high and default are up to 1.55x faster on the mass kernel. At high DPhi carries 1.9e-4
  relative error and at default the map folds. highest matches the GPU counts (2026-09-03).
- Pollution law (li383 float64, 2026-09-04): a tolerance buys a force residual of about sqrt(tol / 10), 3e-4 at
  1e-6, 3e-5 at 1e-8, 3e-6 at 1e-10.
- At tol 1e-6 the descent fails (li383 (16,32,32) mixed, 2026-09-05). Energy sums turn positive before step 8410,
  then the descent collapses and the 1/2 chain opens. tol 1e-10 reproduces tol 1e-8 to 3 digits at 1.2x the cost.
- In plain float32 the residual follows the float64 power law while the stored-field energy rises by 5e-7 at
  8000-9000 steps. E ~ 0.5 stored in float32 has an ulp of 6e-8, so per-step energy changes are noise.
- Descent rates by precision, li383 (16,32,32) p = 2, H100 (2026-09-11): float64 0.669, mixed 0.411, plain float32
  0.125 s/step.
- Floor study, li383 (16,32,32), 20000 steps, L-BFGS, H100 (2026-09-05): float64 resid 8.9e-5 at 1.21 s/step, mixed
  8.8e-5 at 0.65, plain float32 1.02e-4 at 0.23 but 30 % less energy removed.
- Newton, li383 (16,32,32), before the penalty (2026-09-13): plain float32 floors at F2 1.4e-7 (1.6 s/step), mixed at
  4.7e-9 (4.05 s/step), float64 at 4.2e-9 (6.2 s/step). Inner tol 1e-6 / 1e-8 / 1e-10 gave the same floor at 3.1 /
  4.05 / 5.0 s/step: the floor is the discretisation.
- MINRES inner solve with the penalty, li383 and W7-X (16,32,32) (2026-09-18): true residual 0.20 / 0.14 / 0.081 /
  0.034 / 0.018 at 50 / 100 / 200 / 400 / 800 its, direction energy 78 % at 200 and 94 % at 400. Without the
  penalty the residual rises 0.22 -> 0.32.
- Constant MINRES tolerances and the Nocedal-Wright forcing all land on F2 1.53-1.55e-10. None is cheaper than a
  fixed 200 its.

## 8. float32 on a device without float64 (2026-10-13)

MRX_DTYPE=float32, JAX 64-bit mode off, solutions and residuals as double-single pairs. One H200, li383 (8,12,12)
p = 2, tol 1e-8, uniform mesh, before PR #35. Not run on a TPU or on Metal.

- Kernels against float64 on the same float32 data: sum-factorised mass apply 3-4e-15 for k = 0..3 (float32
  6e-8), extraction and its transpose 1e-15, dot product 3e-14 (float32 4.5e-8).
- Harmonic forms, Rayleigh quotient: 3e-15 (k = 2 dbc), 2e-14 (k = 1 free), 1e-14 (k = 3).
- True residual of the solves in double-single: mass 2e-10..6e-8, stiffness on G^T w 5e-9..7e-9, L_0 3e-9, L_1
  4e-9, L_2 2e-8, L_3 2e-8..3e-8. Rounded to float32 the solution has 2e-7..5e-7.
- Relaxation against float64: 20 Newton steps 4.29 against 3.12 s/step, residual 5.618e-8 against 5.626e-8. 400
  gradient steps 0.12 against 0.05 s/step, residual 5.786e-6 against 5.761e-6. div B 1.7e-6.
- li383 M (48,64,24) and L (64,80,32): every solve reaches the float64 true residual (1e-9..6e-8), the 1- and
  2-form Laplacians in 1.4-2x the iterations. 400 gradient steps 9.2e-6 / 1.5e-5 against 7.5e-6 / 1.8e-5 (M / L)
  at the same cost per step. Newton at L diverges.
- Stalled a pair refinement at 1e-7: a part of b outside the range of the operator projected in float32, polar
  weights computed on the device, G^T w formed in float32, a seed interpolated in float32, a form normalised in
  float32.
- Pass rule of refine: asking for tol relative to the current residual over-solves a warm-started singular
  stiffness solve into round-off (float64 Newton test: |F| 0.055 -> 0.42 in 10 steps). Asking for exactly the
  missing reduction stalls the shifted split just above tol. Kept: half the target, between the precision's floor
  and 1e-2.
- Ozaki scheme (exact float32 matmuls of 8-bit slices) on a prototype apply: 2.6e-15, also under TF32, but 8x the
  gather form and slow to compile. Not used.

## 9. Decided against / removed

- Velocity Leray, a second projection, li383 (16,32,32) p = 3 (2026-09-04): float64 2.47 against 0.95 s/step without
  it, same trajectory. Removed 2026-09-05.
- Warm starts of the Krylov solves gained 1 % on quasr44970 8^3 and 14 % on W7-X (12,24,12) (2026-08). Kept.
- Implicit midpoint: exact helicity, Picard 3.5-4.0 its per step in float64 at +1-5 % wall time. In mixed precision
  Picard stalls at the float32 rounding of B_mid and halves dt to 1/16 every step. Fully implicit midpoint diverges
  at the line-search step (2026-09-04 .. 11).
- CG-Steihaug on W7-X (2026-09-18) exits on negative curvature at j = 0 on every step from about step 40, with 197
  of 200 fallbacks, stuck at 1.7e-5. The penalised Hessian has lambda_min -1.8e-3 there. Trust-region Newton-CG works
  at 2x the cost and a 2x worse residual. MINRES passes.
- Hodge-solve alternatives on QA (2026-09-02): Chebyshev inner solves cost 4-5x the atom's wall time at n = 24
  Dirichlet. PCG inner solves took 319-374 its in 41-94 s against 1550 its in 7 s. CG on the true L_1 with inner mass
  PCG took 146-184 s at n = 16 Dirichlet and failed on free after 46 min.
