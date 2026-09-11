# The harmonic atom from the VMEC field, and the helicity correction (2026-09-11)

Session "newton preconditioner", li383, worktree `.claude/worktrees/newton`,
branch `newton-second-variation` (04f5cfe helicity correction, 2653107
read_checkpoint). Runs under `outputs/chunk_speed/`, `outputs/harmonic_from0/`,
`outputs/midpoint_sweep_hcorr/` of the worktree. All meshes li383 p = 2, mixed
precision unless said, production defaults (Newton: Laplacian atom, 300 MINRES
iterations, tol 0.1, dt cap 1; L-BFGS m = 1, smoothing order 1, potential route).

## 1. The chunk boundary costs nothing; the paper's timings are stale, not polluted

Same Newton run in one compiled scan against chunks (job pairs 18364857/8,
18364922/4):

| mesh | chunk | steady s/step | job elapsed |
|---|---|---|---|
| (16,32,32) | 20 | 11.1 | 14:41 (floor at step 40) |
| (16,32,32) | 100 | 11.5 | 26:24 (ran to 100) |
| (32,64,64) | 10 | 42.3 | 48:52 |
| (32,64,64) | 50 | 42.1 | 48:27 |

The loop speed does not depend on the chunk (it is one `lax.scan` either way).
A chunk boundary costs 2-4 s at (32,64,64) against 420 s of stepping per chunk
of 10. The one-time compile of the diagnostics probe at the second boundary is
56 s at (16,32,32) and 195 s at (32,64,64). The single-scan arm cannot test the
floor and ran 60 steps past it. The paper's quoted s/step (stepping wall over
steps, samples excluded, first chunk's compile included) overstates the steady
cost by 1-4% (Newton) and < 1% (L-BFGS) from the compile. But the code got
faster after those runs (September 8, adb4e06): today's Newton arms reproduce
the paper's trajectories step for step to 1-5% in the residual and run 1.5x
faster at (16,32,32) (16.4 -> 11.1 s/step) and 1.8x at (32,64,64) (76.8 ->
42.3), same node type; the September 10 merges (CG carry kept in the requested
precision, refine's early stop) are the likely cause. The "runtimes" session
is re-measuring every paper row at b0f8d38 (`docs/research/runtimes_rescale_2026-09-11.md`).

## 2. A stall mode of the Newton step

At (32,64,64) the chunk-10 arm spent steps 10-22 with the direction's cosine
to the force at 1e-4 and dt = 0 (13 steps, 9 min of GPU at the full 300
MINRES iterations each); the chunk-50 arm 7 such steps. Mechanism: the
fallback test is the strict sign of `u . MF`, a barely-descending direction
passes, the line search returns dt = 0, the field does not move, and the
warm-started potential feeds the same MINRES solve the same start. The
paper's (32,64,64) arm (September 8) had none in 380 steps and matches
today's for nine steps before the stalls; older arms at 32^3 and (12,24,24)
had episodes too. Not fixed (Tobias's call: drop the warm start on a zero
step, or fall back below a cosine threshold).

## 3. The harmonic atom from step 0: better direction, worse relaxation

`outputs/harmonic_from0/{h16,h32}`, same field and settings as the Laplacian
arms of section 1 but `--newton-precond harmonic`:

| mesh | arm | F2 at step 10 / 30 / 50 | min | E removed | dH/H | fallbacks |
|---|---|---|---|---|---|---|
| (16,32,32) | Laplacian | 3.0e-8 / 7.8e-9 / 4.5e-9 | 4.5e-9 (50) | 2.0e-6 (100 steps) | -2.4e-5 | 0 |
| (16,32,32) | harmonic | 4.8e-8 / 6.3e-8 / 1.6e-7 | 1.5e-8 (14) | 3.7e-6 | +9.1e-5 | 0 |
| (32,64,64) | Laplacian | 2.0e-6 / 6.8e-7 / 2.5e-7 | 2.5e-7 (50) | 1.5e-6 (50 steps) | -3.9e-6 | 0 |
| (32,64,64) | harmonic | 2.8e-6 / 9.8e-8 / 1.6e-7 | 9.1e-8 (37) | 1.7e-6 | -6.0e-6 | 5 |

The harmonic atom wins for five steps at (16,32,32) (factor 2-3), turns at
step 10 while the residual is still 2e-8, then the residual climbs
monotonically with the energy still dropping and the line search's own
optimum dt* at 5-9 (the energy is nearly linear along the direction). At
(32,64,64) it reaches 1e-7 by step 30 where the Laplacian arm needed 73, with
dt 0.1-0.5, then the direction goes orthogonal to the force (cos 0) and the
residual jumps by 40 at a fallback.

**The floor of the atom is not the cause** (`h16_kappa1`, kappa 1e-2 -> 1
through a wrapper, `tmp/relax_floor.py`): 5.8e-9 at step 6 (the Laplacian's
50-step floor), then a monotone climb to 9.8e-7 by step 100, energy removed
5.6e-6, dH/H +2.2e-4, dt* 7-11. The ordering is the direction quality's: the
better the direction, the sooner the floor and the faster the climb after it.

**The Dirichlet-H route with a Newton direction is dead**
(`h16_kappa1_mid_f64_aux`: float64, midpoint, auxiliary field): helicity
exact to 1e-15, Picard 5 sweeps, and no relaxation, dt* = 1e-3 on every
step, residual 2.4e-4 for 100 steps. The direction is built from the
plain-B Hessian, the auxiliary route's force and induction carry the wall
layer of H_D, and the line search finds no descent. (The September 4 study
had measured the layer on the descent: 1.7x energy, 5x force floor.)

**The helicity leak is not the cause either** (`h16_kappa1_hcorr_explicit`,
`_f64`, `_midpoint_f64`, section 4): with the helicity exact to 1e-15 the
residual follows the uncorrected arm to three digits (5.8e-9 at step 6,
9.4e-7 at 100 against 9.8e-7), energy removed 5.5e-6 against 5.6e-6. Global
helicity is one number; reconnection at the grid scale happens under it. The
direction finds genuine descent of the discrete energy at fixed helicity, and
that descent raises the force residual: energy into thinner current sheets
the mesh cannot carry, where the L2 force grows while the energy falls. The
Laplacian atom does not find those directions quickly, which is why it looks
better; its floor is a property of the direction quality, not of an
equilibrium. Verdict for the paper: a direction closer to the exact Newton
step is closer to the ideal descent, and the ideal descent has no minimum on
the mesh at the floor. The Laplacian atom stays the default; no switch.

The mixed-precision midpoint with Newton steps hits the Picard defect floor
(float32 rounding of B_mid, 1e-5 relative to the increment): dt halved to
1/16 on every step (`n0_it300_dt1_midB`, `h16_kappa1_hcorr_midpoint`). In
float64 the Picard converges at dt = 1 in 5 sweeps (`n0_it300_dt1_midB_f64`).

## 4. The helicity correction (04f5cfe)

Identity, exact on the mesh for any E in V^1_0 (A and E Dirichlet: the
partial integration is exact; the harmonic part does not move):

    K_{n+1} - K_n = 2 dt <E, B_{n+1/2}> = 2 dt <E, B_n> + dt^2 <E, curl E>.

Plain step E = P_0(v x B_n): <E, B_n> = <E - v x B_n, B_n - P_0 B_n>, the
pairing of two projection residuals; B x n != 0 at the wall while V^1_0 has
zero tangential trace, so B - P_0 B is O(1) in the last cell layer and the
remainder is the wall pairing (the "leak" of the plain-B route). The
auxiliary H = P_0 B_{n+1/2} with the midpoint makes it vanish exactly, at the
price of the wall layer inside the dynamics. The correction keeps B and
removes E's component along B: E <- E - lambda P_0 B_n, lambda the root near
zero of the quadratic 2<E_l, B_n> + dt <E_l, curl E_l> = 0 (explicit), or
lambda = <E, P B_mid> / <P_0 B_mid, P B_mid> inside every midpoint sweep.
Pairings in the residual precision. `TimeStepper.helicity_correction`,
`--helicity-correction`, trace `hcorr`, one k=1 Dirichlet mass solve (free on
the auxiliary route) + two projections + one incidence per step. dt stays
the line search's (chosen on the uncorrected direction; the corrected
direction differs by a relative lambda ||P_0 B|| / ||E||); the step is not
variational (energy perturbed by lambda dt <J, B>).

Test mesh (8,12,12), 20 L-BFGS steps, dH/2E0: float64 1e-15 both schemes
(uncorrected 1e-7 explicit, 5e-8 midpoint); mixed the field's rounding.
`test_helicity_correction_conserves_helicity` added (explicit, plain B).

Paper table settings ((16,32,32) p=2, L-BFGS m=1, order 0, Leray, 1000
steps; `outputs/midpoint_sweep_hcorr/`):

| arm | s/step steady | |dH/H0| | dE<0 | E removed |
|---|---|---|---|---|
| explicit mixed, uncorrected | 0.249 | 1.0e-5 | 883/1000 | 1.864e-6 |
| explicit mixed, corrected | 0.255 | one float32 ulp (9.3e-8) | 888/1000 | 1.836e-6 |
| explicit f64, uncorrected | 0.409 | 1.0e-5 | 1000/1000 | 1.855e-6 |
| explicit f64, corrected | 0.427 | 2.8e-15 | 1000/1000 | 1.844e-6 |
| midpoint mixed, corrected | 0.270 | one ulp | 863/1000 | 1.829e-6 |
| midpoint f64, corrected | 0.503 | 8.7e-16 | 1000/1000 | 1.838e-6 |

Cost 3% (mixed) / 5% (float64) of a descent step, nothing measurable on a
Newton step (11.1 s/step either way); the midpoint's Picard costs more than
the correction. In mixed precision the helicity is itself a float32 number,
so the per-chunk values sit on 0 or +-1 ulp. lambda <= 1.6e-6 on the first
step from the VMEC field, median 6e-12 afterwards (Newton arms: max 2e-8).
The one-in-nine positive dE in mixed is the float32 rounding of the stored
field, present without the correction too. Production settings (potential
route, order 1): 0.300 -> 0.306 s/step (`outputs/chunk_speed/lbfgs500{,_corr}`).

Consequence for the paper: the midpoint scheme's only remaining property is
being variational in time, which nothing measures; the exact scheme is
explicit + correction on B. Draft of the changed blocks (algorithm, helicity
derivation in the order plain-B remainder -> auxiliary-H fix -> wall layer ->
correction, numerics paragraph, table with a third column):
`outputs/figures_2026-09/resubmit-4_hcorr.tex`, every block wrapped in
`% ==== CHANGE 2026-09-11 (helicity correction)` markers with the old text in
`% OLD:` comments. The "runtimes" session times explicit plain-B only; no
midpoint or auxiliary arms anywhere in its plan.
