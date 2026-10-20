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

## 4. Tolerance and MINRES budget of Newton-MR on the full Hessian (2026-10-05 .. 13, replaced 2026-10-18)

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

## 7. Gauss-Newton direction (li383 and W7-X graded, p = 2 unless stated, 2026-10-18)

Study driver on gp-review before #38, 40 steps from the file's field, H200, times without the compile. The harmonic
atom of #38 (one-sided zeta differences, Greville radial points) moves these numbers somewhat.

Force residual 1e-8 reached at step (time), standard CFL weights:

| case | Gauss-Newton, reorthogonalised MINRES | block Gauss-Newton, flexible GCR | Newton-MR, full Hessian |
|---|---|---|---|
| li383 (48,64,24), float32 slab factors | 25 (26.7 s) | 25 (29.4 s) | never, 1.8e-8 at 40 |
| li383 (64,80,32) | 19 (47.6 s) | 19 (53.6 s) | never, 1.5e-8 at 40 |
| W7-X FMM002 (32,64,36) | 28 (31.5 s) | 28 (34.8 s) | never, 4.7e-5 at 40 |
| li383 (48,64,24), p = 3 | 9 (23.3 s) | 9 (24.5 s) | 16 (72.0 s) |

- MINRES and GCR take the same steps with the same residual trace to two digits. Energy removed and helicity drift
  agree to 3 digits. On graded W7-X the helicity drifts 4.8e-4 relative over 40 steps under both, 20x the uniform
  record, so it belongs to the Gauss-Newton step, not the solver.
- With the theta width of each ring in the CFL weights (li383 (48,64,24)): MINRES 1e-8 at step 11 (19.5 s), GCR 11
  (21.5 s), Newton-MR 37 (99 s). The old weights capped dt at about 0.15 for 15 steps.
- Newton-MR on the full Hessian stalls on graded meshes: MINRES capped at 200 on every step on li383, nonpositive
  curvature exits on W7-X (residual 1.5e-4 for 30 steps).

Variants that lost (li383 (48,64,24)):
- MINRES without reorthogonalisation: 1.6-2.6x the iterations. At cap 300 1e-8 at step 39 (91.4 s), at cap 600
  step 25 (51.5 s).
- M_1 solves inside the operator: 1e-5 reaches 1e-8 at step 29 (31.8 s), 1e-4 stalls at 2e-7. 1e-8 equals 1e-6
  in steps and costs 14.2 against 10.9 ms per iteration.
- One MINRES on the 3 x 3 Gauss-Newton system with a fixed block-diagonal preconditioner (harmonic atom, shifted
  Laplacian atom, M^-1 (M + K) M^-1): the 3 x 3 residual reaches 0.1 in 30-140 its, 1.7e-2 in 1500 (slab blocks),
  2.8e-3 in 1500 with degree-8 Chebyshev mass blocks accurate to 1e-4. Even 3e-3 floors the relaxation at 1.6e-6,
  at 34.5 s/step. The error left in the constraint rows reaches a through K, lambda_max(M^-1 K) about 1e5.
- Block GCR with M_1 solves to 1e-2 in its triangular preconditioner stalls (cap 300 on 32 of 40 steps, 1.7e-6 at
  40). 1e-3, 1e-4 and 1e-6 give the same trajectory.
- 32 PCG steps on S_1 inside the a-block of GCR: the right-hand side is not in the range of S_1, the inner PCG drifts
  along its kernel and GCR stalls at 2.7e-4 (5.2 s/step).
- The full Newton block system gives the same trajectory as Gauss-Newton under GCR. Reorthogonalised Newton-MR on
  the full Hessian takes the nonpositive-curvature exit on steps 1-13 and barely moves (3.8 s/step).

Tolerance of the Gauss-Newton solve (NEWTON_TOL, relax.py library code on gp-newton, graded, new CFL weights, 30 steps
from the file's field, H200, 2026-10-19). 1e-8 at step (time without the compile), residual after 30 steps, s/step:

| tol | li383 (48,64,24) | li383 (64,80,32) | W7-X (32,64,36) |
|---|---|---|---|
| 0.03 | 8 (19.4 s), 6.8e-10, 2.94 | 7 (32.8 s), 8.0e-10, 5.27 | 11 (21.9 s), 2.1e-10, 2.26 |
| 0.05 | 8 (17.0 s), 7.7e-10, 2.49 | 8 (34.2 s), 8.2e-10, 5.00 | 12 (19.8 s), 2.8e-10, 1.92 |
| 0.1 | 11 (17.5 s), 1.6e-9, 1.78 | 8 (27.4 s), 1.9e-9, 3.55 | 15 (19.7 s), 1.1e-9, 1.60 |
| 0.2 | 21 (25.7 s), 1.3e-8, 1.29 | 18 (46.8 s), 7.3e-9, 2.49 | 21 (20.9 s), 2.7e-9, 1.14 |

- 0.1 is the fastest or level to 1e-8 on all three. Tighter solves end lower but do not get there sooner. 0.2 does
  not stall, as it did under GCR with the old CFL weights, but needs about twice the steps.
- Energy fell on every step of every run. The energy removed agrees to 0.5 %.

## 8. Preconditioning the Gauss-Newton solve (li383 p = 2, 2026-10-19)

Harness: `scripts/bench/newton_precond.py` (branch newton-precond, not merged) freezes the Newton systems of a
relaxation at given steps and runs the production MINRES (reorthogonalised) with each candidate preconditioner; its
count at tol 0.1 equals the relax.py `newton_it` on every state checked. CPU float64 unless stated.

Starting point (harmonic atom, tol 0.1, warm start): 16-34 MINRES iterations per step at the start and 25-65 later at
(12,20,12); more than 95 % of an iteration is the Hessian apply (two M_1 solves), the atom 0.3 ms. Ritz values of
P A: a clustered bulk, a few outliers at 1e-4..6e-4, and a top end growing with the mesh (95 at (8,12,12), 580 at
(12,20,12)); it is the polar-axis zeta functions. An exact Gauss-Newton model inverse (section 9 of
preconditioners.md, condition 12.7) would need about 5-8 iterations.

Krylov recycling (`--newton.recycle k`, `mrx.relaxation.recycle`, branch newton-precond, not merged):
- 16 consecutive steps, MINRES iterations at tol 0.1, harmonic -> k = 32 (8 bottom + 24 top) -> 64: (8,12,12) 380 ->
  172; (12,20,12) 935 -> 509 -> 399; (12,20,12) graded 662 -> 349 -> 291. Top Ritz value 95-580 -> 5-27.
- Variants that lost at (8,12,12): the exact coarse operator (k applies per step: 0.5-0.98x net), additive (1.13x, it
  cannot move the top), the start correction alone (1.2x), bottom vectors only (1.1-1.2x).
- relax.py (12,20,12) graded, 20 steps, CPU: 801 / 489 / 397 iterations for k = 0 / 32 / 64, 13.0 / 9.0 / 7.8 s per 10
  steps, final residual 4.4 / 4.3 / 4.1e-8.
- H200, graded, 30 steps from the file's field (1e-8 at step, s per step without the compile, residual after 30):

| setting | (48,64,24) | (64,80,32) |
|---|---|---|
| k = 0, tol 0.1 (production) | 11, 1.88, 1.4e-9 | 8, 3.90, 1.8e-9 |
| k = 64, tol 0.1 | 13, 1.53, 3.4e-9 | |
| k = 64, tol 0.05 | 8, 1.92, 1.9e-9 | 8, 3.87, 2.1e-9 |
| k = 64, tol 0.03 | 8, 2.64, 1.3e-9 | |
| k = 0, tol 0.03 | 8, 3.43, 6.3e-10 | |

  Recycling makes the iterations cheaper but, at the same tolerance, the steps no better: without it 23 of 30
  solves at (48,64,24) stop at the cap and overshoot 0.1. Paid into a tighter tolerance it reaches 1e-8 1.4x sooner at
  (48,64,24) and level at (64,80,32). Not merged.

Per-Fourier-mode radial block (branch np-block, `mrx/relaxation/radial_block.py`, not merged): angle-averaged model
operators made block diagonal per (m, n) and inverted exactly per block (polar axis modes |m| <= 1 as one joint block
per n), applied by rfft2 and batched matmuls.
- Inside the harmonic atom (the Hodge model in place of the Laplacian atom): 0.38-0.81x the iterations, but the top
  Ritz value rises (584 -> 928 at (12,20,12)).
- The full Gauss-Newton model (mixed spline symbols regularised at Nyquist, harmonic-symbol floor): top Ritz 584 ->
  173; relax.py (12,20,12) uniform, 20 steps: 36.4 against 56.5 iterations per step, 2.10 against 2.51 s per step,
  final residual 1.6e-8 against 1.2e-8. Uniform meshes only; its remaining top end is the first ring outside the axis
  at high m, the anisotropy the graded spaces address, and no model moves the low tail.

