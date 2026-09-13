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
route, order 1): 0.300 -> 0.303 s/step, 1% (`outputs/chunk_speed/lbfgs500{,_corr}`).

Consequence for the paper: the midpoint scheme's only remaining property is
being variational in time, which nothing measures; the exact scheme is
explicit + correction on B. Draft of the changed blocks (algorithm, helicity
derivation in the order plain-B remainder -> auxiliary-H fix -> wall layer ->
correction, numerics paragraph, table with a third column):
`outputs/figures_2026-09/resubmit-4_hcorr.tex`, every block wrapped in
`% ==== CHANGE 2026-09-11 (helicity correction)` markers with the old text in
`% OLD:` comments. The "runtimes" session times explicit plain-B only; no
midpoint or auxiliary arms anywhere in its plan.

## 5. The budget and kappa sweeps (evening): kappa = 3 at 100 MINRES iterations

All li383 (16,32,32) p=2 mixed, Newton from the VMEC field, 100 steps, `--floor-tol 0`,
per-step squared residual (`outputs/maxiter_sweep/`, `outputs/kappa_sweep/`, code e680ab4+).

| atom | kappa | MINRES it | s/step | min F2 (step) | F2 at 30 / 60 / 90 | E removed | dH/H |
|---|---|---|---|---|---|---|---|
| Laplacian | - | 300 | 12.5 | 4.5e-9 (50) | 7.8e-9 / 5.7e-9 / 6.9e-9 | 2.0e-6 | -2.4e-5 |
| harmonic | 1 | 300 | 11.8 | 5.8e-9 (7) | 3.8e-8 / 3.1e-7 / 8.2e-7 | 5.4e-6 | +2.2e-4 |
| harmonic | 1 | 600 | 22.4 | 1.5e-8 (5) | climbs faster | 8.3e-6 | +3.5e-4 |
| harmonic | 1 | 1000 | 37.2 | 7.8e-8 (3) | climbs faster still | 9.0e-6 (60 st.) | +3.5e-4 |
| harmonic | 1 | 1000, tol 0.01 | 36.7 | 6.2e-8 (3) | = 1000 (the tolerance is inert) | | |
| harmonic | 1 | 100 | 5.0 | 7.2e-9 (23) | 1.0e-8 / 1.0e-8 / 1.4e-8 | 2.6e-6 | +1.8e-5 |
| harmonic | 0.01 | 300 | 12.2 | 1.5e-8 (14) | 6.3e-8 / 1.8e-7 / 2.0e-7 | 3.7e-6 | +9e-5 |
| harmonic | 0.1 | 300 | 12.1 | 1.2e-8 (9) | 2.0e-7 / 4.4e-7 / 9.6e-7 | 4.9e-6 | +1.3e-4 |
| harmonic | 0.3 | 300 | 11.9 | 8.2e-9 (5) | 1.1e-7 / 5.0e-7 / 9.8e-7 | 5.0e-6 | +1.4e-4 |
| harmonic | 3 | 300 | 12.1 | 4.5e-9 (10) | 1.6e-8 / 4.7e-8 / 2.1e-7 | 4.6e-6 | +2.1e-4 |
| harmonic | 10 | 300 | 12.0 | 3.9e-9 (17) | 7.8e-9 / 1.5e-8 / 2.2e-8 | 2.7e-6 | +4.7e-5 |
| **harmonic** | **3** | **100** | **5.0** | **4.9e-9 (33)** | **5.7e-9 / 8.6e-9 / 7.6e-9** | **2.2e-6** | **+7e-7** |
| harmonic | 10 | 100 | 5.1 | 5.1e-9 (55) | 1.2e-8 / 5.4e-9 / 6.7e-9 | 2.0e-6 | -2.4e-5 |
| harmonic, dt cap 0.25 | 1 | 300 | 13.7 | 2.1e-8 (21) | 3.1e-8 at 30 (40 steps) | 2.2e-6 | +8e-7 |
| harmonic, dt cap 0.25 | 1 | 1000 | 39.1 | 9.1e-8 (26) | 1.4e-7 at 30 (40 steps) | 2.4e-6 | +7e-6 |

- The MINRES tolerance is inert (the 1000 / tol 0.01 arm tracks the 1000 arm): the true
  residual in the mass-atom norm follows N^-1/2 for both atoms and 0.1 is unreachable at 300.
- More iterations = a more exact Newton step = a faster route into the post-floor descent:
  the minimum comes earlier and higher, the climb steeper, energy removed and helicity
  drift grow with the budget. Fewer iterations (100) hold the floor.
- kappa interpolates between the harmonic atom (small kappa, the flat modes amplified) and
  the Laplacian atom (kappa = 10 is indistinguishable from it): larger kappa = lower
  minimum, slower climb.
- A dt cap of 0.25 does not find a lower floor (2.1e-8 vs 5.8e-9 at 300 it): the finer
  walk reaches a different, worse state; it only buys the helicity (the explicit step's
  error is O(dt^2)).
- **kappa = 3 at 100 iterations**: below 1e-8 from step ~15 (75 s against the Laplacian
  atom's 10 min to the same floor), holds it through step 100, the same energy removed
  as the Laplacian floor, helicity drift 7e-7 (30x below the Laplacian arm's). The
  candidate default (precond harmonic, HARMONIC_FLOOR 3, newton_maxiter 100), pending
  Tobias's word and the (32,64,64) confirmation (`outputs/kappa_sweep/k3_it100_h32`).
- The regularised energy E + eps ||J||^2 / 2 (branch `energy-regularisation-prototype`,
  gradient regularised, Hessian not): the Newton direction of E is nearly all penalty for
  E_eps (dt* 0.03 at C = 0.01, 0 at C = 1); C >= 0.1 stops the descent, C = 0.01 descends
  slowly (physical residual 7e-7 after 100 steps, no floor). Inconclusive without the
  Hessian of E_eps; not for the release.

**(32,64,64) confirmation** (`outputs/kappa_sweep/k3_it100_h32`, 100 steps, floor-tol 0):
kappa = 3 at 100 iterations reaches 1e-6 / 1e-7 / 1e-8 at steps 8 / 15 / 53 against
the paper's Laplacian arm's 13 / 73 / 256, minimum 3.1e-9 at step 82 (the paper's arm:
9.4e-9 at 325) and holding (last-chunk mean 3.7e-9), ~17 s/step steady (1e-8 in ~18
min against the paper's 5.7 h), dt ~ 1, no fallbacks, no stalls, dH/H -2.5e-6.

## 6. The iteration count, the regularised line search, and the L-BFGS stall (late evening)

**MINRES iterations at kappa = 3** (`outputs/kappa_sweep/k3_it{30,50,100,150,200,300}`): every
budget >= 100 reaches the same minimum (4.5-4.9e-9) in the same wall time (~2-2.5 min); what
differs is afterwards: 100 holds (last chunk 7.9e-9, dH/H +7e-7), 150 / 200 / 300 drift back to
1.4e-8 / 3.7e-8 / 2.9e-7 with dH/H 1.8e-5 / 6.8e-5 / 2.1e-4; 50 reaches 8.8e-9 at step 70, 30 no
floor in 100 steps. The optimum is 100.

**The regularised line search** (`TimeStepper.step_regularisation`, 68de2c3): the line search
minimises E + eps ||J||^2 / 2 along the increment, direction and force unchanged, dt* =
(<F,u> - eps <J, curl~ dB>) / (||dB||^2 + eps ||curl~ dB||^2), one weak curl of dB per step.
On the kappa = 3 / 100 arm (`outputs/stepreg_sweep/`): identical until the floor (dt = 1 there
is the cap, not dt*), then the accepted step collapses to 0.03 (C = 0.1) or 0.003 (C = 1) of
the Newton length and the run holds a floor 1.7x lower (4.5e-9 last-chunk mean at C = 0.1
against 7.9e-9); +2-4% per step. With `--dt-floor 0.1` the run stops itself at step 35 (244 s)
at the same floor (`k3_it100_c0.1_dtstop`). Defaults: C = 0.1 and dt-floor 0.1 for Newton, both
0 for L-BFGS, because on the descent the search is NOT inert (dt* ~ 2.2 is the binding step
there, not the CFL cap): the 500-step production arm's trajectory changed and cost +9%
(`outputs/chunk_speed/lbfgs500_stepreg`).

**The L-BFGS stall in the reruns** (`paper_rerun_2026-09-11`, smooth-first, m = 1): the anchor's
accepted step collapses from 2.2 to 0.017 over steps 1500-2000 with cos(u, F) = 0.00, no
progress for ~1000 steps, then recovery with a x4 bump of the residual at step 2000; tol1e-6
(same early trajectory) the same, seed61 / seed51 2 / 4 episodes, m5 11 episodes (its
"reconnection"), m0 / h12 / tol1e-10 / leray_m1g1 none. The paper's old arms (smooth-last)
have none in 30 000 steps of records. With m = 1 a tiny step collapses the single pair's scale
and reproduces itself (the descent's version of the Newton stall). m = 0 as the default would
cost 4x (1e-7 at 25 min against 6, never 1e-8 in 18 000 steps). Proposed and pending: Powell's
restart for m = 1 (= PR-CG): drop the pair when |<F_k, F_{k-1}>| > 0.2 ||F_k||^2.

**Figures**: every line figure of the paper regenerated from the run records in one style
(`outputs/figures_2026-09-11/paper_figures.py`: raw per-step traces, L-BFGS alpha 0.5 / lw 1,
Newton lw 1.5, one legend per factor, house fonts, PDF + PNG + PGF), plus the three Newton
sweeps as figures and `newton_sweeps_table.tex`, and `appendix_run_parameters.tex` (one
parameter table per figure and paper table, from the records).

## 7. Powell's restart, measured (2026-09-12, `outputs/powell/anchor18000`)

The anchor (li383 (16,32,32) p=2, potential route, smooth-first, m = 1, mixed, 18 000 steps,
`--floor-tol 0`) with `POWELL_RESTART = 0.2` (5f3be3b) against the rerun without it:

| | steps to 1e-6 / 1e-7 / 1e-8 / 5e-9 | resid at 18 000 (last 1000) | E removed | dH/H | s/step | restarts |
|---|---|---|---|---|---|---|
| rerun (no restart) | 320 / 1194 / 12 079 / 14 861 | 3.5e-9 | 1.963e-6 | -9.9e-6 | 0.302 | - |
| Powell restart | 285 / 1185 / 1472 / 1830 | 2.35e-9 | 2.015e-6 | -1.15e-5 | 0.314 | 1417 (all before step 8000) |

2000-step means, Powell: 1.5e-6, 2.5e-8, 2.4e-8, 2.7e-8, 4.3e-9, 3.6e-9, 3.1e-9, 2.7e-9, 2.4e-9;
rerun: 1.4e-6, 1.3e-7, 5.0e-8, 2.9e-8, 2.1e-8, 1.6e-8, 9.6e-9, 5.6e-9, 3.8e-9. The stall is gone
(dt 1.8-2.5, cos 0.4-0.55 throughout). The restart-heavy phase (steps 1500-8000, 20% of the
steps restarted) is a plateau at 2.5e-8 with a transient dip to 5e-9 at step 1830; once the
restarts stop the run descends to a floor 1.5x below the rerun's. Whether the paper's descent
arms are rerun with the restart is Tobias's call.

## 8. Smoothing the Newton direction (2026-09-12): negative

Prototype (not committed): the descent's smoother `(Id - eps L)^-1` applied to the Newton
direction before the line search, released configuration otherwise (harmonic kappa = 3, 100 it,
C = 0.1), li383 (16,32,32), 100 steps, stops off (`outputs/newton_smooth/k3_it100_c0.1_smooth`
against `outputs/stepreg_sweep/k3_it100_c0.1`): minimum 5.5e-6 at step 3, then the residual
rises to 3.6e-5 and the accepted step goes to 0 from step ~15 (the regularised search rejects
the smoothed direction: its curvature term dominates), energy removed 1.79e-6 against 1.97e-6.
The smoother removes exactly the content the Newton solve put in; the step-length
regularisation is the right place for the roughness, not the direction. Reverted.

## 9. The reconnection ladder and the island seeds as Newton runs (2026-09-12)

`outputs/newton_demos/ladder3` (li383 (16,32,32) p=2, harmonic kappa = 3, 100 MINRES it, plain
line search, five rungs of 60 Newton steps, a resistive solve spending 2.5% of the helicity
between them): every rung reaches its floor, 4.9e-9 / 3.2e-9 / 1.6e-9 / 1.7e-9 / 1.7e-9, no
fallbacks, dt = 1 throughout, helicity -2.26 / -2.28 / -2.30 / -2.31% per solve, beta_vol
4.42 -> 2.63%; 300 steps in 22 min against the descent ladder's 40 000 steps (3.3 h). With the
regularised search (C = 0.1) the rungs after the first did not move: after a resistive step
the regularised energy E + eps ||J||^2 / 2 RISES along every descent direction (the relaxation
regrows the current the resistive step diffused), dt* < 0 on all 60 steps of a rung; first the
plain descent test let the Newton direction through and the step went backwards, then with the
regularised-slope test every direction was refused. 8ebf010: the Newton descent test is the
regularised slope along the direction's own increment, dt* <= 0 is no step, and the driver sets
`--step-regularisation 0` with a reconnection series. The seeds (`seed61_2`, `seed51_2`, released
defaults): (6,1) floor 5.2e-9 at step 39 (265 s, stop on the floor test); (5,1) floor 1.3e-8 at
step 41, then a slow climb with the step shrinking, stop on the dt test at step 200 (the same
factor above the unseeded floor as in the descent table). The best-state file is
`checkpoints/best.h5` (renamed from `state_best.h5`, which the tracers' `state_*.h5` glob
mistook for a step).

## 10. The W7-X high-beta demonstration: Newton past the floor destroys the surfaces (2026-09-12)

`outputs/w7x_highbeta/h32` (GVEC `w7x_highbeta.dat`, (32,64,64) p=2, mixed, the released Newton
configuration, 200 steps, 37.7 s/step, 2.1 h) and `h16` ((16,32,32), same). h32: resid 1.28e-3
(IC) -> 8.3e-6 at step 10, then 4.4e-6..7e-6 for the remaining 190 steps, best 4.36e-6 at step
177, dt 0.4..1.0 (the cap), 100 MINRES iterations every step, no fallback, energy -5.6e-5 and
helicity -1.7e-5 relative, beta_vol 5.25 -> 3.27%. The floor of this system at this resolution is
~5e-6, far above `--floor-tol 1e-8`, so neither stop fired and the run spent its step budget
walking along the floor. The Poincare sections show what that walk does. IC: nested surfaces,
iota 0.856 (axis) .. 0.955, 7 of 160 lines lost, 3 chaotic (h/2 drift 0.17 -- W7-X wants more
than 24 trace steps per period). Snapshot traces (`trace_snapshots.npz`, steps 20..100 + 200):

| step | lost | chaotic | iota range |
|---|---|---|---|
| 0 | 7 | 3 | 0.856-0.955 |
| 20 | 16 | 11 | 0.856-0.938 |
| 40 | 9 | 28 | 0.856-0.938 |
| 60 | 24 | 25 | 0.855-0.909 |
| 80 | 27 | 47 | 0.855-0.865 |
| 100 | 50 | 33 | 0.853-0.872 |
| 177 (best) | 52 | 57 | 0.833-1.000 |
| 200 | 65 | 61 | 0.833-1.000 |

The edge rotational transform falls from 0.95 to the axis value by step 80, the outer half of the
plasma goes stochastic, and the axis pressure drops by a quarter (x100: 1.0 -> 0.75): past the
floor the Newton direction is discretisation noise and a dt ~ 1 step of it is a random walk of
numerical reconnection that drains the pressure toward the force-free state, at constant residual.
The best-state carry, which picks by residual, picks one of the worst states. h16 stopped on the
dt test at step 40 (best 2.4e-2 at step 7, unresolved at p=2). Consequences: a Newton run on a
system whose floor is above `--floor-tol` must stop at the floor (set `--floor-tol` to the expected
floor, or stop on no improvement over a window); the W7-X state to show is the first floored one
(step 10-20), re-traced with more steps per period; and the paper's W7-X subsection has to say
this rather than present step 177.

## 11. W7-X done right: fmm002 and the wout reference at (16,32,32) p=2 (2026-09-12, late)

Tobias: the high-beta file is a bad starting equilibrium; p = 2 only. `outputs/w7x_p2/fmm16`
(`data/GVEC_State_final.dat`, the fmm002 GVEC state) and `outputs/w7x_p2/wout16`
(`data/wout_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc`), released Newton
configuration, 200-step budget: both stop on the floor test at step 40. fmm16: resid 7.0e-5 ->
3.6e-10 (best, step 39), 5.4 s/step, 217 s, dH/H -1.7e-4, dE/E -3.5e-7, beta_vol 1.1%. wout16:
1.2e-5 -> 7.8e-9 (step 38), 6.2 s/step, 249 s, dH/H -5.7e-4, dE/E -1.1e-6, beta_vol 4.4%
(axis 11%). Traces at 96 steps/period (h/2 drift 1e-4, against 0.09-0.17 at 24 on the high-beta
file): no line of 160 lost, 2 / 6 chaotic, iota 0.915-1.056 / 0.856-0.978, nested surfaces
and a smooth p throughout; fmm16 shows the small 5/5 chain at r ~ 0.83. Pages in
`outputs/figures_2026-09-11/w7x/poincare_{fmm16,wout16}_{ic,best}_zeta0.*`; both drafts'
W7-X subsection rewritten around these two (which one to show is open).

## 12. The descent block on the released code (2026-09-13, `outputs/lbfgs_2026-09-13`)

Eight L-BFGS arms (li383 (16,32,32) p=2 mixed, potential route, smooth first, Powell restart,
`--floor-tol 0`, 5000 steps unless noted), ~5.4 GPU h: anchor (m=1, gamma=1, c=0.02) 0.33 s/step,
8.1e-9 at 5000, minimum 1.9e-9, first below 1e-8 at step 1481; m=0 1.0e-7 at 5000 (no restart to
help gradient descent); m=5 1.9e-8 at 5000, minimum 1.4e-9, first below 1e-8 at 1142 -- with the
restart it no longer reconnects (the 2026-09-11 m=5 arm gained 125% helicity); gamma=0 at 20 000
steps and 0.21 s/step reaches 2.2e-7 at step 2228 and then DEGRADES to 1.9e-5 with the helicity
drifting by +8.2e-4 (smoothing is what keeps the descent on the ideal manifold, not just a speedup);
c = 0.004 / 0.1 / 0.5: 7.4e-7 / 7.3e-8 / 1.4e-7 at 5000 (0.1 is faster over the first 1000 steps
and stalls), c = 0.02 stays the optimum; Leray 9.2e-9 at 5000 against the potential anchor's
8.1e-9, at 0.44 against 0.33 s/step. Figures m_sweep, gamma_sweep, c_sweep and lbfgs_tables.tex
(the c table and the two-row velocity table) come from paper_figures.py and are in the draft
(captions placeholders; the prose still quotes the 2026-09-11 numbers).

## 13. W7-X FMM002 (VMEC wout) unseeded and seeded at iota = 1 (2026-09-13, morning)

`data/wout_W7-X_FMM002_000_000000.nc` (iota 0.916 axis -> 1.070 edge, |iota| = 1 at rho = 0.83),
(16,32,32) p=2, released Newton configuration: `outputs/w7x_p2/fmm002w16` (unseeded) and
`fmm002w16_seed51` (seed (5,1) at rho0 = 0.83, width 0.1, eps 1e-2: for nfp = 5 the seed's n counts
field periods, so (5,1) is the iota = 1 chain; the (11,10) attempt on the wout reference the night
before was a non-resonant 50-period ripple, the corrected (11,2) seed is damped to a quarter cell at
m = 11 on 32 poloidal cells -- a null result, see `wout16_seed_wrong_n10`, `wout16_seed1011`).
Both stop on the floor test at step 40 (5.4 s/step, 3.6 min): unseeded 1.8e-4 -> 3.0e-10, seeded
1.8e-4 -> 1.1e-9; dH/H +2.3e-5 / -4.9e-5, dE/E -1.1e-6 / -1.6e-6, beta_vol 1.12%. Traces at 96
steps/period, no line lost. The 5/5 chain: unseeded 0 (IC) -> 0.65 h_r (relaxed, the natural
chain); seeded 3.03 h_r (IC) -> 2.74 h_r (relaxed): the seeded island survives the Newton descent
within 10% of its width, as the li383 seeds did, with the pressure flat across it (p x100 ~ 0.03
on all island lines, the iota shelf at 5/5 over r = 0.75-0.9). Pages:
`outputs/figures_2026-09-11/w7x/poincare_fmm002w16{,_seed51}_{ic,best}_zeta0.*`.

## 14. The Newton sweeps on the released code (2026-09-13, `outputs/newton_sweeps_2026-09-13`, `outputs/helicity_2026-09-13`)

Released Newton configuration (harmonic atom kappa = 3, 100 MINRES iterations, C = 0.1, dt-floor
0.1, floor-tol 1e-8, chunk 20, 200-step budget), li383, mixed precision, every arm stopping itself
on the floor test. s/step below is the steady rate over the chunks after the first (the first
chunk carries the compile: 104 s at (16,32,32), 30 min at (48,96,96)); the paper's wall axes put
that compile before the first step and the operators' setup (2-2.5 min at n = 16) is not on them.

- h sweep at p = 2 (traced ic + best, five planes): (12,24,24) 40 steps, 2.8 s/step, floor 1.9e-8
  (does not reach 1e-8); (16,32,32) 40 steps, 4.05 s/step, 4.7e-9, 1e-8 at step 19 (2.9 min);
  (24,48,48) 60, 8.8 s/step, 2.1e-9, 1e-8 at 30; (32,64,64) 80, 16.7 s/step, 4.0e-9, 1e-8 at 51;
  (48,96,96) 140, 53-61 s/step, 7.8e-9 at step 132, 1e-8 at 121 (126 min), MINRES at its 100
  budget on every step, no line lost in the trace. Steps to the floor grow like n (40, 40, 60,
  80, 140), the cost per step like the DoFs (x13 from 16 to 48), so the wall to 1e-8 goes 2.9 ->
  8.6 -> 26 -> 126 min. The L-BFGS reference at (16,32,32) takes 8.8 min to 1e-8 at 0.32 s/step.
- p at (16,32,32): p = 1 cannot resolve (floor 2.7e-4 after 80 steps, dH/H -7.5e-5); p = 2 4.7e-9
  (1e-8 at 19); p = 3 3.1e-9 (1e-8 at 15, 5.7 s/step); p = 4 7.1e-9 at step 65 (1e-8 at 53, 9.0
  s/step) -- the degrees converge to the same floor, p = 4 needing three times the steps.
- solver tolerance 1e-6 / 1e-8 / 1e-10: identical trajectories (4, 6, 19 steps to 1e-6/7/8, floor
  4.6e-9 in all three) at 3.1 / 4.05 / 5.0 s/step: the inexact Newton step does not see the inner
  tolerance, only the cost does.
- precision: float32 floors at 1.4e-7 (1e-6 at 17, never 1e-7; 1.6 s/step); mixed and float64 agree
  (4.7e-9 vs 4.2e-9, 1e-8 at 19 vs 17) at 4.05 vs 6.2 s/step.
- helicity table, six explicit L-BFGS arms (m = 1, gamma = 0, 1000 steps, midpoint rows dropped):
  |dH/H| auxiliary H 9.3e-7 (mixed) / 1.5e-6 (float64), B only 1.0e-5 / 1.0e-5, B corrected
  1.9e-7 / 1.7e-15. The H arms run the Leray route (the potential velocity is refused with the
  auxiliary field), the other four the potential route.
- the reconnection ladder with C = 0.1 (`outputs/newton_demos/ladder4`, one 5% reconnection after
  60 steps): the first 60 steps floor at 4.2e-9 like the reference run; the reconnection spends
  4.23% (the dose estimate is linear), residual 4e-9 -> 9.1e-6, J/B 0.64 -> 0.51; after it NOT ONE
  step is accepted in 60 (dt = 0, fallback on every step, dt* < 0): the regularised slope
  -eps <J, curl dB> outweighs <F, u> on every direction of the reconnected field, as the driver's
  comment says. C = 0 with a reconnection series stays (Tobias: keep the relaxation as is); the
  paper's ladder remains `ladder3` (four reconnections at 2.5%).
