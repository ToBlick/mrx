# Relaxation: measurements

The measured basis of the relaxation defaults and the methods that lost. F2 = squared normalised force residual.
resid = the unsquared residual of notes before 2026-09-06. li383 = wout_li383_1.4m, W7-X = GVEC FMM002, p = 2
unless stated. Helicity drifts are absolute at ||B||_M = 1 unless written dH/H.

## 1. Gradient descent and its smoothing (2026-09)

- The descent residual falls as a power law t^-a with no plateau (li383 gamma = 1, 2026-09-03): a = 0.34 / 0.30 /
  0.20 / 0.22 / 0.23 at n = 8 / 12 / 16 / 24 / 32, 0.15 at p = 3. Newton reaches the mesh floor in 40-100 steps.
- Smoothing constant, m = 0 descent, li383 (16,32,32), 5000 steps (2026-09-17), in the unit mu = c / n_r^2. The
  code's SMOOTHING_C = 0.075 is in squared physical radial cells.

  | smoothing | F2 floor |
  |---|---|
  | c = 0.004 | 8.47e-8, slower to 1e-6 |
  | c = 0.02 | 9.99e-8, kept |
  | c = 0.1 | 1.36e-7 |
  | c = 0.5 | 1.75e-7 |
  | gamma = 0 | 1.78e-7 at its minimum, rising tail |

- Without smoothing the descent leaves the ideal manifold. The gamma = 0 arm degraded 2.2e-7 -> 1.9e-5 over 20000
  steps with dH/H +8.2e-4 (2026-09-13), 100x the drift of the smoothed arms.
- gamma = 1 on W7-X (12,24,12), 2000 steps (2026-08-27): floor 1.34e-4 by t = 40-77 with 8-16x larger stable steps.
  gamma = 0 was still at 1.75e-4 at t = 5.7.
- The potential route v = curl a equals the Leray route to 3 digits and is 26-29 % cheaper per step with smoothing,
  0.303 against 0.411 s/step (H100, 2026-09-11). The harmonic velocity it omits carries 1e-15 of the energy at every
  floor (2026-09-07).

## 2. Step size and surfaces (W7-X, CG with line search, 2026-08-25 .. 27)

- The maximal line-search step destroys surfaces far from equilibrium. On GVEC's initial guess over 3000 steps, |dH|
  per unit energy was 5.5e-2 against 9.6e-4 at a fixed dt 1e-3 (58x). A fixed dt 3e-3 gave the same force reduction
  with 73x less |dH|. This motivated the CFL cap.
- Absolute |dH| at ||B||_M = 1 predicts surface loss: survivors 1.5-2e-6, casualties 6e-5..3e-4. Energy, div B and
  the line-search identity all stayed satisfied while surfaces went.

## 3. Newton: the parallel-flow penalty (adopted 2026-09-18)

- Hessian on divergence-free velocities, li383 (16,32,32) float64: spectrum [0.12, 3.9e4], cond about 3.3e5. Its
  soft end is a continuum of field-aligned flows u = f B, the 32 lowest eigenvalues 2.6e-4..0.16 and 70-98 %
  field-aligned. An exact Newton step there would be 10-100x the physical step (2026-09-17).
- With the penalty H + alpha M_par the soft end lifts to 0.066-0.16. li383 over 200 steps: F2 2.19e-10 and still
  descending. Without it the run floors at 8e-9 and climbs to 1.34e-8 with dH/H +6.5e-5.
- Relative alpha sweep, li383, 200 steps: 0.025 3.2e-10, 0.05 2.0e-10, 0.075 1.9e-10, 0.1 1.6e-10, 0.2 2.1e-10.
  The cliff lies between 0.0075 and 0.025.
- NEWTON_PENALTY = 3.0 times the strain: li383 c = 1 / 2 / 3 / 5 gives F2 2.26 / 1.70 / 1.62 / 1.63e-10, W7-X 1.2e-10
  / 2.55e-11 / 2.62e-11 / 2.57e-11. c = 3 is within 6 % of the best on both. The Hessian scale is about 4 on li383
  and 0.115 on W7-X, 35x apart, so the penalty is in units of the strain.
- Step cap 1: lifted to 4 or infinity the search settles at dt 2.0 and F2 sits at 1.1e-7.
- Surfaces 20x past the old floor, li383 (16,32,32), 160 lines: 3-4 chaotic, against 51 for the unpenalised control
  200 steps past its floor.

F2 with the penalty against the previous configuration (2026-09-17/18):

| case | penalty | previous |
|---|---|---|
| li383 (16,32,32), 200 steps | 1.9e-10 | 4.7e-9 |
| li383 p = 3, 200 steps | 1.4e-10 | 3.1e-9 |
| li383 seeded (6,1) | 3.8e-10 | 5.2e-9 |
| li383 (32,64,64), 100 steps | 4.5e-10 | 3.1e-9 |
| W7-X (16,32,32) | 7.8e-11 at dt 1.00 | 7.4e-11 at dt 0.34 |
| W7-X (32,64,64) | 6.6e-11 | 9.7e-11 |

## 4. Newton tolerance and MINRES budget (2026-10-05 .. 13)

- NEWTON_TOL = 1e-3 in the M_1^-1 norm of the residual (2026-10-13). At 0.1 the warm start of a state that barely
  moved meets the test and the step is zero. li383 (48,64,24) at p = 3 then stops moving after 10 steps (E_0 - E
  1.3833e-6 against 1.5652e-6). At p = 2 no solve meets even 0.1 within 200 its on li383 (48,64,24), (64,80,32) and
  W7-X (32,64,36), so maxiter is the control.
- A cold start of every Newton solve removes 0.4-5 % less energy on the same cases.
- NEWTON_MAXITER = 200. Caps 50 and 100 cost more steps than they save, li383 cap 50 needs 127 steps to 1e-9. Cap
  400 takes W7-X to 1e-9 in 8 steps but at 2.4 s/step. On fixed li383 states from step 5 on, 25 its remove as much
  energy as 800. On late W7-X states 25-200 its give 0.78-0.90 of the energy of 800.

## 5. Seeds (2026-08-27 .. 09-05)

- A seeded island must be at least about 3 radial cells wide. On W7-X (12,24,24) p = 3 seeds of 2.9-11.5 % of rho,
  against a radial cell of 8 %, were squeezed back to the unseeded state. A seed of 29 % of rho kept its width
  through 2500 steps.
- Seeded widths under ideal descent match the pendulum formula on li383: (6,1) excursion 0.060 / 0.098 / 0.167 at eps
  1e-3 / 3e-3 / 1e-2 against 0.06 / 0.10 / 0.19.

## 6. Decided against / removed

Newton:
- A converged Newton direction (sandwich preconditioner, 1000 its, over 90 % of the decrement) gave the worst floor,
  F2 4.5e-8, 28x the Laplacian atom's (2026-09-08, before the penalty).
- A physical strain floor without the penalty goes past the floor at step 15-19. F2 climbs to 6e-8 with energy
  +25-40 % and dH 50-100x (2026-09-17).
- Smoothing the Newton potential or velocity: F2 1e-5..1e-6 with up to 46 of 100 fallbacks (2026-09-12, 09-17).
- A regularised line search on E + eps ||J||^2 / 2 and a dt floor are inert with the penalty. After a resistive solve
  the regularised search blocks every step (2026-09-11 .. 13).
- Newton ladder, 5 rungs x 60 steps with 2.5 % of H per resistive solve: every rung floors at 1.6-4.9e-9, 300 steps
  in 22 min against 40000 descent steps in 3.3 h. Removed with the regularised search (2026-09-12/13).
- Step dt = theta dt* up to the CFL limit instead of the cap 1 (2026-10-05): 1e-9 comes later (li383 never, W7-X
  after 116 against 13 steps) and islands open on li383 (12 against 3 chaotic lines). theta = 1/2 needs 68 against
  41 steps.
- Eisenstat-Walker forcing and a MINRES stagnation exit (2026-10-05): the forcing loosens to 0.5 near li383's floor
  and floors at 2e-9. The stall exit floors at 1e-8 with window 25 or needs 3x the steps on W7-X with window 50.
- CG-Steihaug and trust-region Newton-CG: solvers_precision.md 8.

Descent:
- L-BFGS (2026-08-26 .. 09-17): CG and L-BFGS m = 1 / 3 / 5 / 10 agree to 2e-8 in E on W7-X (8,16,8). On li383
  (16,32,32) m = 5 equals m = 1 (1.40 against 1.44e-4). m = 1 stalled for about 1000 steps until a Powell restart.
  Removed 2026-09-17 for being finicky.
- Smoothing lambda in the initial condition beyond h = 1e-4 adds current, J/B 0.43..1.22. lambda = 0 is the
  straight-field-line field with ||J||/||B|| about 1.2, not the vacuum (2026-08-26).

Helicity:
- A Dirichlet-H auxiliary field removes 1.7x more energy with a 5x higher force floor from the H_t = 0 wall layer.
  With a Newton direction it does not relax, dt* = 1e-3 every step (2026-09-07/11).
- Implicit midpoint: solvers_precision.md 8.
