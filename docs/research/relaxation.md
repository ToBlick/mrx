# Relaxation: measurements

The relaxation loop, its methods and options are described in the Sphinx docs. This file keeps the
measurements behind the choices. Conventions: F2 = squared normalised force residual (paper
convention). "resid" = the unsquared normalised residual used by notes before 2026-09-06. li383 =
wout_li383_1.4m (ns = 49) unless noted. W7-X reference = FMM002. p = 2, mixed precision unless
noted. Helicity drifts are absolute at ||B||_M = 1 unless written dH/H.

## 1. Descent (gradient on the smoothed force, line search + CFL cap)

### Identities and what they do not prove

- Exact minimiser along the frozen ray: <B,dB>_M = -(F,u)_M, measured 1.3e-11 relative / 5.6e-17
  absolute (2026-08-25). With an exact line search and t free in sign, energy decreases monotonically
  for ANY direction: monotone energy does not validate a method. Under a fixed small dt the measured
  drop is 2x the predicted one by construction (2.0000). With eta > 0 the identity breaks by exactly
  dt eta ||J||^2.
- ||F|| carries no monotonicity guarantee. Final-F scatter over the last 20% of an arm: 1.14-1.17x
  for fixed-dt arms, 2-4x for line-search arms without damping (2026-08-25): line-search arms cannot
  be ranked on final F below ~4x.
- The energy-force identity holds to tol |B||Q|, not to tol |dE| (2026-09-22).

### Ideal descent is a power law

- resid ~ t^-a with no plateau (li383 gamma = 1, 500-step blocks, 2026-09-03): a = 0.34 / 0.30 / 0.20 /
  0.22 / 0.23 at n = 8/12/16/24/32 p=2. 0.66 at p = 1. 0.15 at p = 3. Block noise ~0.05. A stall test
  "drop < tol over N steps" fires at t = aN/tol (a step count in disguise). gamma = 0 arms are not
  power laws at that block length.
- Chunk statistics at the floor: CV 2% within a 100-step chunk. Single sample vs chunk mean median
  1.6%, p90 8%.
- Realisation scatter (li383 (16,32,32), 2026-09-11): two runs of one method coincide through 1500
  steps, then diverge from round-off. The 5000-step residual scatters 3-5x between realisations.
  Quote 10000-step residuals or the run minimum.

### m = 0 gradient descent (the paper's descent), li383 (16,32,32), 5000 steps (2026-09-17)

| arm | F2 floor | notes |
|---|---|---|
| smoothing c = 0.02 | 9.99e-8 | 1e-6 at step 449, 1e-7 at 4995, never 1e-8 |
| c = 0.004 | 8.47e-8 | slower to 1e-6 |
| c = 0.1 | 1.36e-7 | |
| c = 0.5 | 1.75e-7 | |
| gamma = 0 | 1.78e-7 (min) | rising tail |

- Leray route = potential route to the digit. c = 0.02 kept.
- Steepest-descent 2-cycle (Akaike, 2026-09-18): gamma = 0 is an exact period-2 orbit (lag-1
  correlation -1.000), steps 3.00e-5 / 1.74e-4, 1/dt1 + 1/dt2 = 3.905e4 = the top Hessian eigenvalue
  (Lanczos 3.9e4), branches 5.80x apart. gamma = 1 same cycle, steps 9.4e-4 / 8.9e-4, sum 2.19e3 (the
  smoother lowers the effective top eigenvalue 18x). No breakdown at 7000 steps.

### Smoothing constant (mu = c / n_r^2)

- li383 (16,32,32), L-BFGS m=1 era, 5000 steps (2026-09-05): resid 2.8 / 2.3 / 2.7 / 3.2 / 3.4e-4 for
  c = 0.0064 / 0.02 / 0.064 / 0.2 / 0.64. Order 0 6.7e-4, stalling (flat 6.3-7.0e-4 from 2500). Helicity
  drift -3.7..-4.3e-8 for every smoothed arm (independent of c), -8.7e-8 unsmoothed. Energy removed
  1.88 -> 1.66e-6 with c. beta_vol 4.28-4.33e-2 for all. Decision c = 0.02.
- Under smooth-first (2026-09-11): c = 0.0064 reconnects, c = 0.64 lowest (3.6e-8), c = 0.02-0.2 at
  5.8-10e-8: a flat optimum is not resolvable at 5000 steps.
- Smooth-first ordering (smooth, then combine) is worth ~2x in the tail at m = 1 on either route
  (potential 1.08e-4 vs 2.37e-4 at 5000, Leray 2.08e-8 vs 5.72e-8 squared) (2026-09-06).
- gamma = 0 (no smoothing) is not ideal over long runs (2026-09-05): float64 and mixed identical chunk
  by chunk, descent cosine 0.003-0.004 (order 1: 0.2), helicity drift turns positive at ~10000 steps
  (+4.3e-6 at 20000, 100x the smoothed arms) with accelerated energy release and resid climbing to
  4e-3. Released-code gamma = 0 at 20000 steps degraded 2.2e-7 -> 1.9e-5 with +8.2e-4 dH/H
  (2026-09-13). Smoothing keeps the descent on the ideal manifold, not only faster.

### Velocity routes

- Potential route (v = curl a from a k=1 Hodge solve + harmonic coefficient) = Leray route to 3 digits
  per block without smoothing and at m = 0 (2026-09-06). m = 0 rows 11.37 / 11.37, gamma = 0 rows
  38.8 / 39.5 and 98 / 97 x1e-8 (2026-09-11).
- Harmonic velocity omitted by the potential / Newton route: share of |PF|^2 along h 3e-9..2.7e-6,
  h^T H h = 1.18e-3. Energy of the omitted h-step 1e-15 at every floor (li383, 16 and 32 radial cells,
  float64, 2026-09-07): negligible.
- B-only step (J x B, u x B on the 2-form, no H proxy) vs H-form (2026-09-03): at p >= 2 the B-only
  helicity term is below the explicit time error (float64 (12,24,24): -3.4e-8 vs -3.1e-8). Larger only
  at p = 1 (2.4x).

### Starting from the VMEC state

- li383 ns = 49 is within 1.4-1.6e-6 of the relaxed energy on fine meshes (E_0 - E 3.3 / 1.9 / 1.7 /
  1.6e-6 for n = 12/16/24/32). The ns = 16 file releases 1e-4. The ns = 49 reference starts 4x lower
  (resid 1.3e-2 vs 5.5e-2) and floors 3x sooner (2026-09-05).

## 2. Step size, topology and the helicity witness (W7-X, 2026-08-25 .. 27)

CG + line search, no CFL clip. w7x_ini = GVEC's initial guess (iteration 0, not an equilibrium).
fmm002 = GVEC's converged state.

- The maximal line-search step destroys surfaces far from equilibrium: w7x_ini, 3000 steps, line
  search (dt 5e-3..1.4e-2) vs fixed dt 1e-3: |dH| per unit energy 5.5e-2 vs 9.6e-4 (58x), pure chaos
  vs mostly nested. Fixed dt 3e-3: the same force reduction as the line search (5.75x vs 5.63x) with
  73x less |dH|, surfaces preserved. |dH|/H per dE flat 0.0216-0.0260 over dt 1e-4..3e-3, jumping 53x
  at the line search (a cliff). Near equilibrium (fmm002) the line search is 30x more productive and
  safe (189x vs 6.4x force reduction). Beta refuted as the discriminator. Motivated the CFL clip.
- No enforced invariant notices surface destruction: energy monotone 3000/3000, ||div B|| 6.7e-14,
  line-search identity 6.5e-17, harmonic amplitude 1e-16, all satisfied while the line-search arm went
  chaotic.
- ABSOLUTE |dH| at ||B||_M = 1 predicts surface loss (blind classification of Poincare pairs,
  monotone in |dH|: survivors 1.5-2e-6, casualties 6e-5..3e-4, ~30x). Relative drift misleads (H
  spans 3 decades). Confirmed on w7x-ini-conv 12^3: fixed dt 1e-3 vs line search 126x in |dH|.
- Force residual does NOT predict surfaces: the eta = 1e-3 arm had the campaign's lowest final F
  (4.5e-4), 21x the energy removed, 81% of H gone, islands everywhere, iota reversed 1.25 -> 0.
- H is not resolution-converged for near-harmonic fields (a near-cancellation): fmm002 IC helicity
  -1.78e-4 (8^3), -3.16e-5 (12^3), -5.76e-6 (16^3), sign change between p = 4 and 5 (+6.9e-6).
  Same IC on (8,16,8) / (12,24,12) / (16,32,16) / (12,24,24) / (16,32,32): -1.9e-4 / -2.9e-5 /
  -5.4e-6 / +3.5e-6 / +6.3e-6. Quote absolute dH within a resolution, never dH/H_0 across meshes. w7x_ini is stable
  (1 - B_harm_rel 1.416 / 1.423 / 1.424%).
- p-sweep, fmm002 8^3, |dH| per dE: p = 1 5.2e-2, p = 3 1.26e-2, p = 5 1.11e-3 (47x, monotone).
  Roughness ||J||/||B|| reduction flat 0.12-0.19 in p.
- Running longer is not free (fmm002): 13018 vs 3000 steps +0.0009% energy, final F 1.8x worse, |dH|
  2.2x, core restructured (iota 0.857-1.624). Energy converged by ~3000 steps.
- The pressure multiplier converges toward GVEC's p on coherent fields (2.1x closer) and away under
  greedy long runs (x3.53, x1.75): accumulated reconnection.
- lambda = 0 (fmm002) relaxes to the purely harmonic field: harmonic cosine 0.985353 -> 0.999980, 2.9%
  energy removed (120x the lambda-on case). dzeta IC: energy floor (1/2)cos^2||B||^2 = 0.2325746 set by
  the conserved harmonic amplitude (<h, curl E> = 0 exactly). 0.2438 at step 1560 (87%).
- gamma = 1 lever: far from equilibrium (quasr logical IC) force 5.1x better and |dH| 15x smaller. Near
  equilibrium (fmm002) 14% worse force and 3.1x time for 2.4x less |dH| and a 9x tighter axis.
- gamma = 1 on W7-X (12,24,12), potential IC, 2000 steps (2026-08-27): floor 1.34e-4 by t = 40-77 with
  8-16x larger stable steps vs gamma = 0 still descending at 1.75e-4 at t = 5.7. Same energy. Helicity
  jumps once in the first ~100 steps (-8.5e-8 / +1.2e-7) then is conserved, gamma = 0 drifts
  continuously (+8.8e-8). beta_vol +1.5% (+8% at (8,16,8)). Excess ||J||/||B|| at the floor +13%. Axis
  iota dip 0.910-0.912 vs 0.915 on every mesh (smoothing on the m <= 1 polar patch). At fixed
  mu/h^2 = 0.064 the helicity jump shrinks 4.1e-7 -> 0.9e-7 -> 0.9e-7 over three meshes.

## 3. W7-X initial conditions under relaxation (2026-08-26/27)

- GVEC closed-form state vs gridded export (fmm002, gamma = 0, 300 steps, (16,32,32)): ||J||/||B||_0
  0.055 vs 0.167, ||F||_0 7.5e-4 vs 7.1e-3, beta_vol(0) 1.09e-2 (GVEC's) vs 5.9e-3, ||F|| after 300
  steps 1.3e-4 vs 2.4e-3, E_0 - E 4e-8 vs 1.4e-5 (300x less energy released: the IC is the discrete
  equilibrium). The grid route's current ROSE with refinement (0.111 -> 0.167, resolving the
  linear-bridge noise). Every gridded-route edge structure (5/5, 25/24, 20/19 slivers, stochastic outer
  15%) was bridge noise. Basis for deleting the gridded route.
- 4000-step closed-form run: ||F|| 7.45e-4 -> 2.8e-5 (min 1.6e-5 near step 1700), nested to the wall,
  iota 0.915 -> 1.05. The floor is where the descent direction stops being resolved (line-search
  cosine 0.02).
- (8,16,8) is a smoke resolution: its iota = 1 ~4 cm core and wide islands are mesh artefacts.
  (12,24,12) 2000 steps gives GVEC's flat 0.916 core, a small 5/5 edge chain, ||F|| 1.9e-4 vs 5.8e-4.
- High-beta export w7x_ini_conv (16,32,32) gamma = 0, 2200 steps: ||J||/||B||_0 1.16 (20x fmm002),
  force falls only 2x (8.2e-3 -> 3.8e-3), dt 4e-4. The outer shear region (iota 10/11..25/26) breaks
  into a chaotic sea (physics vs unresolved current undecided).
- QA vacuum IC (12,24,12) p=3 (2026-08-28): 430 steps take D (distance to the mesh's harmonic form)
  5.7e-4 -> 9.5e-5, ||F|| 1.06e-2 -> 1.65e-3: descent removes the wout's axis current that the
  projection keeps.

## 4. Newton (second variation)

### Spectrum of the Hessian (li383 (16,32,32) p=2 float64)

- On div-free velocities, 150-200 Lanczos steps: [0.117, 3.93e4] at the IC, [0.123, 3.89e4] at step
  5000, none negative, cond ~3.3e5. The top converges (grid-scale (m,n) ~ (12,16) at the axis). The
  low end does NOT (smooth ~index^2 ladder of speckle mixtures). Reconnected and nested states agree
  to 2% on every Krylov number (2026-09-06, 2026-09-17).
- Block LOBPCG (block 32, 200 it, 2 s/it) (2026-09-17): the soft end is a CONTINUUM of field-aligned
  near-null flows u = f B: the 32 lowest eigenvalues 2.6e-4..0.16 and still sliding, 45-65% of the
  energy in the first radial cell, 70-98% field-aligned, dominant helicity the local axis resonance
  ((7,-1) where iota_axis = 3/7). With the first cell masked the soft modes sit on the rational
  surfaces: (6,-1) at r 0.60 (1/2 at 0.53), (5,-1) at 0.65-0.80 (3/5 at 0.79), (7,-1) at 0.25-0.30
  (3/7 at 0.27), 55-77% aligned, also in the nested state (rational surfaces, not islands).
- Force projection on the soft block: 2e-4..2e-2 of ||F||^2, but proj/lambda up to 2.1 (axis) .. 24
  (3/5 modes) against 1e-3..1e-2 for bulk modes: an exact Newton step would be parallel-flow garbage
  10-100x the physical step. The old kappa = 3 floor and 100-it truncation were Levenberg-Marquardt
  damping in disguise.

### Parallel-flow penalty (adopted 2026-09-18)

- H + alpha M_par with <v, M_par u> = int (v.B)(u.B)/|B|^2 J lifts the soft end to 0.066-0.16
  (alignment 0.01-0.03, max proj/lambda 0.03). The soft end becomes an m = 1 axis shift at 0.07.
  li383 200 steps: F2 2.19e-10, still descending, dt = 1 every step. Control without the penalty
  floors at 8e-9 then climbs to 1.34e-8 with +6.5e-5 dH/H (2026-09-17).
- Units: absolute alpha 0.3 on li383 = kappa 2.6 on W7-X. The Hessian scale (2pi)^2(h_t^2 + h_z^2) is
  ~4 on li383, 0.115 on W7-X (35x). Relative alpha sweep, li383 200 steps: 0.025 3.2e-10 (undamped
  signature starts), 0.05 2.0e-10, 0.075 1.9e-10, 0.1 1.6e-10, 0.2 2.1e-10. Cliff between 0.0075 and
  0.025.
- Penalty c x strain: li383 c = 1/2/3/5 F2 2.26 / 1.70 / 1.62 / 1.63e-10, W7-X 1.2e-10 / 2.55e-11 /
  2.62e-11 / 2.57e-11 (constant alpha 0.1: 1.53e-10 / 2.47e-11). c = 3 within 6% of the best on both.
- Step cap: lifted to 4 or infinity the search settles at dt 2.0 and F2 sits at 1.1e-7 (resolved
  modes reflected at dt* = 2). Cap 1 kept.
- MINRES 200 is where the true solve residual crosses 0.1 (solvers_precision.md section 6).

Results with the penalty (2026-09-17/18), vs the previous released configuration:

| case | penalty | previous |
|---|---|---|
| li383 (16,32,32) p=2, 200 steps | 1.9e-10 (merged default: 1.9e-10 at step 100) | 4.7e-9 |
| li383 p=3, 200 steps | 1.4e-10 | 3.1e-9 |
| li383 seeded (6,1) | 3.8e-10 | 5.2e-9 |
| li383 (32,64,64), 100 steps | 4.5e-10 (1e-8 at step ~40) | 3.1e-9 |
| W7-X FMM002 (16,32,32) | 7.8e-11 at dt 1.00 | 7.4e-11 at dt 0.34 |
| W7-X FMM002 (32,64,64) | 6.6e-11 | 9.7e-11 |

- Same energy removed and helicity drift as before in every case: the earlier "resolved floor" 4e-9
  was the method's, not the mesh's. Reconnection series (8 reconnections): 0 fallbacks, residual
  before reconnections 4-15e-10 vs 2-6e-9. Numerical 1/2 and 3/7 chains roughly halved (0.033 vs
  0.059, 0.036 vs 0.055).
- Surfaces: see poincare_islands.md section 3 (3-4/160 chaotic lines at 20x past the old floor, the
  unpenalised control 51/160).

### Newton before the penalty (Laplacian atom, 300 MINRES, 2026-09-06 .. 13)

- From a descended state the mesh floor in 40-100 steps (8-15 min) vs descent's t^-0.2. Floors F2
  1.95e-8 (12,24,24), 4.52e-9 (16,32,32), 2.50e-9 (24,48,48), 4.00e-9 (32,32,32), 9.36e-9
  (32,64,64), not monotone in the mesh. 120-min continuations all rose above their minima.
- From step 0 on (16,32,32): floor 4.92e-9 in 18 min (one step takes F 2.5e-4 -> 9.4e-7). Later starts
  give lower floors (3.6 / 3.3 / 2.6 / 2.0e-9 after 500 / 1000 / 2500 / 5000 descent steps). Descent
  after a Newton floor then Newton again finds nothing lower: corner selection, not valley walking.
- Past the floor: at 16 radial cells Newton reconnects (helicity drift 2-4e-5, residual climbs back).
  (32,32,32) drift +1.9e-7 (160x smaller). Refined radial windows around rationals do not fix it,
  uniform n_r = 32 does. Explicit and midpoint-on-B arms drift identically (-1.44e-5 at path 74): a
  projection error of the helicity pairing on grid-scale field-aligned modes, not time error. (The
  cause was the parallel null space, the penalty removes it.)
- Released kappa = 3 harmonic-atom config (2026-09-13) h/p sweep, li383: p = 1 floor 2.7e-4 (cannot
  resolve), p = 2 4.7e-9, p = 3 3.1e-9, p = 4 7.1e-9 at step 65 (3x the steps). n = 12 1.9e-8, 16 4.7e-9,
  24 2.1e-9, 32 4.0e-9, 48 7.8e-9.
- Harmonic atom from the VMEC field (2026-09-11): kappa = 1 at 300 MINRES 5.8e-9 at step 7, then climbs
  to 1e-6 by step 100 (dH/H +2.2e-4). kappa = 1e-2 never reaches 1e-8. kappa = 10 = the Laplacian atom.
  kappa = 3 at 100 MINRES below 1e-8 from step ~15, min 4.9e-9, holds, 8x faster to the floor, 30x
  less drift.
- A converged Newton direction (sandwich, 1000 it, > 90% of the decrement) gave the WORST floor
  (4.5e-8, 28x the Laplacian atom's): the second full step raised F2 30x (1.1e-7 -> 3.5e-6). The
  ordering of arms was the ordering of flat-mode content (2026-09-08).
- Newton on 10 radial cells ((10,16,16), 2026-09-10): good for ~5 steps, then leaks (lose 0.6% of H,
  a seeded field goes chaotic after 40 steps).

### W7-X Newton (released config, (16,32,32) p=2, 2026-09-12/13)

- wout reference 1.8e-4 -> 3.0e-10 at step 40 (5.4 s/step, 3.6 min). Seeded (5,1) 1.1e-9. GVEC fmm002
  7.0e-5 -> 3.6e-10 at step 39. Wout beta 0.05 reference 1.2e-5 -> 7.8e-9. Traced at 96 steps/period:
  no line lost.

### Decided against / removed (Newton)

- Physical strain floor without the penalty: goes past the floor (min at step 15-19, F2 climbs to
  6e-8, energy +25-40%, dH 50-100x) (2026-09-17).
- Smoothing the Newton potential or velocity: negative twice (F2 1e-5..1e-6, up to 46/100 fallbacks)
  (2026-09-12, 2026-09-17).
- CG-Steihaug and trust-region Newton-CG: solvers_precision.md section 6.
- Regularised line search (E + eps ||J||^2 / 2) and the dt floor: inert with the penalty. After a
  resistive solve the regularised search blocks every step (dt* < 0 on all directions). Gradient-only
  prototype: C >= 0.1 stops the descent, C = 0.01 slow (7e-7 after 100 steps) (2026-09-11 .. 13).
- Newton ladder (5 rungs x 60 steps, 2.5% H per resistive solve): every rung floors (4.9 / 3.2 / 1.6 /
  1.7 / 1.7e-9), 300 steps in 22 min vs 40000 descent steps (3.3 h). Removed with the reconnection
  series because of the regularised search (2026-09-12/13).

## 5. Helicity

- Budget, li383 (8,16,16) float64, 1000 steps (2026-09-04): explicit time error ~2e-7 (front-loaded).
  Natural-H wall leak up to 1e-6. B-only projection error up to 1e-6 (state-dependent, returns to
  ~0). Dirichlet-H midpoint 5e-12. (16,32,32): midpoint+H 1e-12, explicit+H 1e-7, explicit+B 4e-7,
  midpoint+B 1e-8 (B projection error 50x smaller than at 8^3: O(h^p)). In float32 all sit at the
  1e-7..4e-7 diagnostic floor.
- Helicity correction E <- E - lambda P_0 B (one k=1 Dirichlet mass solve), li383 (16,32,32), 1000
  steps (2026-09-11): |dH/H_0| explicit mixed 1.0e-5 -> one float32 ulp (9.3e-8). float64 1.0e-5 ->
  2.8e-15. Midpoint float64 corrected 8.7e-16. Energy removed unchanged (1.84-1.86e-6). lambda <=
  1.6e-6 on the first step, median 6e-12 after. With no smoothing: float64 -3.5e-15 corrected vs
  -1.5e-6 auxiliary H field vs -1.0e-5 plain B. Mixed -1.9e-7 / -1.0e-6 / -1.0e-5. m = 0 descent
  (2026-09-17): mixed 5.11e-6 -> 5.58e-7, float64 5.67e-6 -> 3.5e-16. Plain drift on the 1.4m file is
  7x smaller than on the low-res reference file (7-8e-5).

Decided against / removed:
- Dirichlet-H auxiliary field: changes the descent (1.7x more energy, 5x higher force floor from the
  H_t = 0 wall layer). Inconsistent for B x n != 0 (proxy differs from B by O(1) in a one-cell wall
  layer, J x H residual 3.4e-3 at the J x B state). With a Newton direction it does not relax at all
  (dt* = 1e-3 every step, resid 2.4e-4 for 100 steps) (2026-09-07/11).
- Implicit midpoint: exact for helicity (above) but its only remaining property after the correction
  is being variational in time. Picard 2.0 it/step float32 (max 3), 3.5-4.0 float64 (max 5-6), +1-5%
  wall. Mixed-precision stall and nonlinear divergence in solvers_precision.md (2026-09-04 .. 11).

## 6. Resolution sweeps on li383 (ideal descent, L-BFGS m=1 era, 2026-09-05)

- h at p = 2, 5000-10000 steps: resid at equal steps does not converge with h (2.6 / 2.2 / 2.4 /
  3.2e-4 at 5000 for n = 12/16/24/32). Relaxed iota profiles converge (rms vs n = 24 1.3e-3 / 7.3e-4 /
  6.0e-4 for n = 12/16/32). 3/5 chain 0.03 wide at n <= 16, 0.01 at n >= 24. 1/2 closed.
- p at (16,32,32): resid at equal steps RISES with p (2.2 / 2.9 / 7.2e-4 at 5000 for p = 2/3/4, p = 5
  2.8e-3 at 1000) while energy and helicity converge. Descent cosine negative on 46 (p = 4) / 65
  (p = 5) of the first 1000 steps, never at p <= 3. Axis iota hook grows with p (0.001 / 0.006 / 0.014
  above the IC at p = 2/3/4, confined to r < 0.06). p = 1 does not resolve the file (released 9.1e-4 of
  E_0, B^zeta -> 0 near the axis).

## 7. Islands, seeds and resistivity

- Seeded islands under ideal descent (li383, 2026-09-02/05): widths scale as sqrt(eps) (excursion
  0.060 / 0.098 / 0.167 for (6,1) at eps 1e-3 / 3e-3 / 1e-2, pendulum formula 0.06 / 0.10 / 0.19) and
  end within one tracer spacing (0.006) of the seed at both surfaces, gamma 0 and 1, both meshes:
  tearing-stable at eps <= 1e-2. 10000 steps: (6,1) 0.164 -> 0.167, (5,1) 0.145 -> 0.152.
- Released descent (2026-09-11, 10000 steps): (6,1) 2.62 -> 2.61 h_r, (5,1) 2.33 -> 2.23 h_r (+-5%).
  The (5,1) arm reconnected once (steps 1000-1500, helicity +4.4e-6). The unseeded anchor carries
  grid-scale 1/2 and 3/5 chains of 0.25-0.36 h_r, so seeded chains stand 6-7x above them.
- Seeded 5/5 in W7-X fmm002 (12,24,24) p=3, gamma = 0 (2026-08-27): seeds at eps 1e-4 / 4e-4 / 1.6e-3
  (pendulum widths 2.9 / 5.7 / 11.5% of rho vs radial cell 8%) are squeezed back to the unseeded state
  (E, F, beta, H equal to 3 digits). eps 1e-2 (29% of rho, ~3.5 cells) keeps its width through 2500
  steps with p flat across it. Ideal dynamics freezes topology, so the small seeds were removed by
  numerical reconnection (floor 1-2 radial cells): seed >= ~3 radial cells wide.
- Force balance in islands (seeded li383, 2026-09-10): residual relative to J x B smallest at the
  islands (5e-4), largest at the axis (1e-2) and wall (5e-3). p is NOT flat in islands (p x100
  3.6 -> 3.1 across the 1/2 chain, 2.8 -> 1.25 across 3/5). The implied parallel gradient ~2e-5 is below
  the 1e-4 residual floor: flattening needs a ~100x lower residual or resistivity.
- Resistive dose, li383 (16,32,32) gamma = 1 (2026-09-03): the dose int eta dt controls. Timing
  (pulse vs tanh) nearly irrelevant. 0.8 of the helicity lost per unit dose either way. The floor
  drops with eta only because current leaves (J/B 0.645 -> 0 at 1e-4). eta 1e-6: iota flattens to
  0.53-0.54. 1e-5: vacuum profile. Resistive (6,1) width ~0.075 independent of the seed. Helicity
  price per backward-Euler solve exact, dH = -2 eps int J.B, mesh-independent (-1.002 / -0.999 /
  -1.001%).
- Reconnection ladder (8 solves of 1.2% H over 18000 steps, 2026-09-04, 4 x 2.25-2.31% over 40000,
  2026-09-05): 3/5 width ~0.052 sqrt(% of H_0) (0.076 at 2.25%, 0.117 at 4.48%, 10-20% above at larger
  doses). J/B 0.66 -> 0.45, beta_vol 0.044 -> 0.027, axis iota 0.395 -> 0.423. Kick gone 2-11x within
  one 500-step chunk. Mesh-converged at (16,32,32) vs (32,32,32) uniform / refined (3/5 width 0.049 /
  0.050 / 0.050, J/B within 1%).
- Resistive demonstration (li383 32^3, 2026-09-22): ONE common reference (nested Newton state at step
  150 after a heat step c = 0.03), 200 resistive steps at C = 0.064 then 100 ideal. Nested, (6,1) at
  a*, (5,1) at a*/4 all land on the same state: 3/5 chain 0.061-0.063, 1/2 closed to the floor (0.023),
  beta_vol -1.4%, H x1e3 5.0122 -> 5.0077 in every arm. Residual 5.3e-10 after the resistive phase,
  9.0e-11 after the ideal one. With per-arm references it could not converge: a heat step removes
  sheets, not islands, so each J* memorises its topology.
- W7-X fmm002 has nothing to reconnect ((12,24,12), 2000 steps, 2026-08-27): eta_max 1e-6 is below the
  numerical floor (force, energy, section identical, extra drift +2e-8). eta_max 1e-4: force floor
  -25%, beta_vol -4.6%, dH 2.5e-6 (28x ideal), islands NOT wider. Island width is set by the resonant
  drive, not by allowed reconnection.
- eta sweep on fmm002 (in-loop resistivity, since deleted, 2026-08-25): eta = 1e-2 reaches F 2.2e-9
  with 99.98% of H gone (vacuum field). eta <= 1e-6 indistinguishable from ideal (1e-9 identical to 9
  digits for 5 steps, then round-off divergence). Transition between 1e-6 and 1e-3. A round-off
  perturbation opens a 4.3% spread in dE over ~1750 steps that re-converges.

## 8. Decided against / removed (descent methods)

- L-BFGS memory buys nothing (2026-08-26, W7-X fmm002 (8,16,8) p=3, 1000 steps): CG and L-BFGS
  m = 1/3/5/10 agree to 2e-8 in E. Memoryless BFGS with exact line search = Polak-Ribiere CG. li383
  (16,32,32): m = 5 = m = 1 (1.40 vs 1.44e-4). Line-search cosine falls with m (0.55 / 0.25 / 0.20 /
  0.11 for m = 0/1/3/5) without a better floor. m = 1 stalled (accepted step 2.2 -> 0.017 with cos 0
  for ~1000 steps) until a Powell restart (1e-8 at step 1472 vs 12079). m = 1 reached 1.9e-9
  (2026-09-11/12). CG removed 2026-08-28, L-BFGS removed 2026-09-17 for being finicky. All m = 1
  floors in section 6 and the floor study are L-BFGS numbers.
- Historical L-BFGS defect (quasr44970 8^3, 2026-08-25): storing the B increment instead of dt u plus
  a one-step lag of y. Fixing only the lag made it worse (0.10% vs 0.21% energy, sy < 0 on 135/250 vs
  31/250). Both fixed 0.557% (2.2x steepest descent). gamma > 0 with a quasi-Newton direction gives
  ascent directions (R H_k not SPD, 5/250).
- Velocity Leray and the float32 attainable tolerance: solvers_precision.md.
- A sharp smoothing optimum at mu = 1e-4 (fmm002 8^3) was a coarse-mesh artefact: flat at
  (16,32,32).
- lambda smoothing of the IC: beyond h = 1e-4 it ADDS current (0.43..1.22 J/B, stochastic at h =
  1e-2), because lambda = 0 is the straight-field-line field with ||J||/||B|| ~ 1.2, not vacuum
  (2026-08-26).
