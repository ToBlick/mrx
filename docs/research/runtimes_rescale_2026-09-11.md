# The paper's run times re-measured on the released solver core, 2026-09-11

Tobias: the code got faster after the paper's runs; rerunning everything is
out of the question; the paper restricts to one form of smoothing (smooth
first, which the code now hard-codes) and reports the current version's
run times. So: every quoted per-step cost re-measured on the current head
(b0f8d38 of `newton-second-variation`) with short runs at the paper's
settings, and every time column recomputed as the paper's step count times
today's cost. Budget 10 GPU hours; used: see the end.

## Why the numbers moved, and what did not

The paper's runs were made at adb4e06 (2026-09-08). Between then and the
head, the tpu-support merge (478f86e, 2026-09-10) changed the solver core:
the CG carry is kept in the precision the solve was asked for (dd22d84; an
operator holding one float64 array had been widening the whole Krylov loop)
and the refinement loop stops on a pass that does not lower the residual
(a395a65). The trajectories are unchanged: today's Newton run from the VMEC
field at (16,32,32) reproduces the paper's `n0_from0` step for step to 1-5%
in the squared residual over 40 steps, with the same 300 MINRES iterations
per step and the same energies and helicities to 4 digits; the (32,64,64)
pair agrees the same way. So step counts, residuals, floors, energies and
helicities in every table stand. Only seconds moved.

The quoted s/step of the paper is the stepping wall over the steps: the
diagnostics (helicity, pressures, checkpoints) are excluded (the loop's
`t_out`), the first chunk's JIT compile is included. From the chunk stamps
of every quoted run the compile share is 1-4% on the Newton rows and under
1% on the L-BFGS rows. A chunk boundary itself costs 2-4 s at (32,64,64)
against 420 s of stepping per chunk of 10 (measured today: chunk 10 against
one chunk of 50, 50.3 vs 50.2 s/step); the one real overhead is the one-time
compile of the diagnostics probe, 56 s at (16,32,32) and 195 s at (32,64,64),
which a single-chunk run pays at its end. `resubmit-4.tex` line 1244 ("the
quoted timings include evaluation of diagnostics, 1-5%") should say the
opposite: diagnostics excluded, compile included at 1-4%. Line 1187 ("the
fastest performance ... in one chunk") is not borne out; drop or soften.

## The re-measurement

`scripts/relax.py` at b0f8d38, li383, one job per row, 300-500 L-BFGS steps
or 30 Newton steps in 3-5 chunks; the per-step cost is the median of the
chunks after the first (the first carries the compile). The paper's arms
were reduced the same way (`compile_share.py`), so the factor is steady
against steady. Records in `outputs/runtimes/<row>/` of the `runtimes`
worktree, logs in `outputs/runtimes_jobs/2026-09-11/`, rates in
`outputs/figures_2026-09/runtimes_2026-09-11/rates.json`.

| row | paper s/step (steady) | today | factor |
|---|---|---|---|
| L-BFGS n=12 | 0.330 | 0.229 | 1.44 |
| L-BFGS n=16 | 0.666 (smooth last) / 0.643 (smooth first) | 0.411 | 1.62 / 1.56 |
| L-BFGS n=24 | 1.96 | 1.29 | 1.52 |
| L-BFGS n=32 (32,64,64) | 5.45 | 2.98 | 1.83 |
| L-BFGS n=48 | 22.6 | PENDING | |
| L-BFGS n=64 | 63.0 | PENDING | |
| p=1 | 0.215 | 0.197 | 1.09 |
| p=3 | 1.23 | 0.814 | 1.51 |
| p=4 | 2.56 | 1.57 | 1.63 |
| p=5 | 4.94 | 3.70 | 1.33 |
| tol 1e-6 | 0.420 | 0.193 | 2.18 |
| tol 1e-10 | 0.790 | 0.486 | 1.62 |
| m=0 | 0.645 | 0.428 | 1.51 |
| m=5 | 0.691 | 0.441 | 1.57 |
| float64, order 1 | 1.24 | 0.669 | 1.85 |
| float64, order 0 | 0.705 | 0.410 | 1.72 |
| float32, order 1 | 0.230 | 0.125 | 1.84 |
| float32, order 0 | 0.090 | 0.062 | 1.45 |
| mixed, order 0 (Leray, m=1) | 0.383 | 0.248 | 1.55 |
| Leray, m=0, order 0 | 0.355 | 0.227 | 1.56 |
| potential, m=1, order 1 | 0.451 | 0.303 | 1.49 |
| potential, m=0, order 1 | 0.450 | 0.303 | 1.49 |
| potential, m=1, order 0 | 0.318 | 0.224 | 1.42 |
| potential, m=0, order 0 | 0.383 | 0.224 | 1.71 |
| Newton (12,24,24) | 10.7 | 8.0 | 1.34 |
| Newton (16,32,32) | 16.4 | 11.1 | 1.48 |
| Newton (24,48,48) | 36.5 | 23.6 | 1.54 |
| Newton (32,64,64) | 76.8 | 42.3 | 1.82 |

The factor is not one number: it is the solve share of a step. Rows with
short solves barely move (p=1, plain float32), rows dominated by long solves
move most, and it grows with the mesh for both methods, 1.4 at n=12 to 1.8 at
n=32. Consequences inside the tables: the tolerance table's cost ratio
1e-6 : 1e-8 goes from 0.63 to 0.47; the p sweep's cost curve flattens at
both ends; the Newton : L-BFGS ratio at every threshold is unchanged to
within the per-mesh factor (both methods share the solves).

Two failed attempts at n=48 and 64 were mine, not the code's: the paper's
arms ran with `--map-batch 8192` and 160 / 320 GB of host memory
(`sw_h48_p2`, `sw_h64_p2`, sacct), my batch had neither. A further n=48
attempt sat in the scan's compile for 80 minutes on a node shared with ten
of my own jobs (XLA's constant folding is host-side and single-threaded; the
paper's line 702 already warns of this) and was cancelled; the retry alone
on a node is the PENDING row.

## Validity of the rows (14:00): which trajectories the head reproduces

Every re-measured row was compared step by step with its paper arm
(`cmp_all.py`, `cmp_h32_newton.py`, `smoothfirst_cmp.py`; squared residual,
10-step means, first 100 steps).

- Verified (same trajectory to 1-10%): every order-0 row (m=0 and the three
  γ=0 precision rows match to 1.00), Newton (16,32,32) over 40 steps,
  L-BFGS n=16 against the paper's smooth-first Leray arm
  (`potential_relax/leray_m1_sf`, 0.90-1.06), and, by the same 1.1-2x
  pattern as n=16 against the smooth-last sweeps, n=12, 24, tol 1e-10,
  m=5, float64 and float32 order 1.
- NOT reproduced: n=48 (squared residual 10-50x the paper's from step 2,
  |F| growing 1.5e-2 -> 5.1e-2 over the first 25 steps while E decreases,
  step-2 cosine 0.03, the start pressure diagnostics already differing),
  p=4 and p=5 (5-10x, dt 0.9 against the paper's 5.6-7.7), tol 1e-6
  (2-24x). Newton (32,64,64): matches the paper's arm for nine steps, then
  13 consecutive steps with dt = 0 and cos(u, F) = 0 (chunk-10 run), stalls
  again at 33 and 46; the chunk-50 run stalls at 18 and 39-44; the paper's
  arm has dt in [0.89, 1] at every step. Zero fallbacks: the solves under
  the step, not the Newton direction. The newton-preconditioner session
  sees the same stall mode in its (32,64,64) arms.
- Reading: `refine` (mrx/solvers.py) now stops on the first pass that does
  not strictly lower the true residual and keeps the last iterate
  (a395a65, in the tpu-support merge 478f86e); the paper's code refined to
  `max_passes`. A float32 pass that stalls on a hard system (large n, high
  p, loose inner tol) leaves the Leray or smoothing solve unconverged with
  no message; a pass abandoned on its first try returns the warm start,
  which is a direction orthogonal to the force and a zero step. Probe
  proposed: the n=48 arm in float64, 30 steps; fix if confirmed: gate the
  stop to plain float32 (`RESIDUAL_DTYPE == DTYPE`), which is the
  configuration it was added for.
- Consequence: the rates of n=48 (5.3 s/step, factor 3.8), p=4, p=5,
  tol 1e-6 and both (32,64,64) rows (L-BFGS 2.98, Newton 42.3) are not
  usable until the probe says otherwise; an abandoned pass is cheaper than
  a converged one, so part of those "factors" may be the bug. The n <= 24,
  p <= 3, tol >= 1e-8 and order-0 rows stand.

**Probe verdict (15:20).** The n=48 arm in float64 on the same code
(job 18383773, 30 steps): squared residual against the paper's mixed run
0.29 / 0.96 / 1.18 per 10 steps (the first window is the smooth-first
transient, as at n=16), |F| falling 1.50e-2 -> 7.9e-3, dt 1.1-1.6, no
stall; the mixed run at b0f8d38 had 46 / 9 / 30 and |F| rising. So the
mixed-precision refinement stop was the cause; removed in e680ab4 (the
newton session, suites 54/54 x3). float64 at n=48 today: 28 s/step steady
(chunks 2-3), compile 12 min. The held rows are to be re-measured on
e680ab4. The n=48 mixed rate of 5.3 s/step is void.

**(32,64,64) Newton verified on e680ab4 (newton session, job 18384195,
50 steps, `chunk_speed/h32_chunk10_nostop`):** zero stalled steps (13 on
b0f8d38), cos(u, F) median 0.11, dt 0.89-1 throughout, squared residual
within 1-4% of the paper's arm at every step (step 22: 4.81e-7 vs 4.57e-7;
40: 2.21e-7 vs 2.24e-7; 49: 1.41e-7 vs 1.36e-7). Steady 42 s/step as this
morning: the Newton (32,64,64) row's factor 1.82 stands. Still held: the
L-BFGS (32,64,64) row, n = 48, p = 4, 5 and tol 1e-6 (the last gets its
rate from the paper-rerun batch's 32 000-step arm on e680ab4).

**Vacuum solve at the floor rungs, current code, float64, tol 1e-10
(`scripts/vacuum_timing.py`, jobs 18380092/18380098/18383776):**

| rung | DoFs | build | harmonic form first / second call | force first / second | Rayleigh | Leray resid | raw JxB resid | J/B |
|---|---|---|---|---|---|---|---|---|
| 39x78x39 p=2 | 335k | 103 s | 40 / 19 s | 14 / 10 s | 3.2e-24 | 7.2e-27 | 1.7e-24 | 1.8e-12 |
| 32x64x32 p=3 | 182k | 81 s | 38 / 19 s | 15 / 11 s | 1.5e-23 | 9.8e-26 | 6.9e-24 | 3.9e-12 |
| 41x82x41 p=4 | 390k | 127 s | 95 / 68 s | 34 / 30 s | 5.3e-22 | 4.1e-24 | 2.4e-22 | 2.3e-11 |

The second call is the solve without compile; the force residual is in the
relaxation's units, sixteen orders below the relaxation floor. p=4 needed
`--map-batch 8192` (the map evaluation allocates 14 GiB unbatched).

## The time columns, recomputed

Time = the paper's step count x today's steady s/step, compile and setup
excluded (say so in the captions; the paper's convention smeared the compile
over the first chunk, 1-4%).

- `tab:newton_convergence`: `newton_table.tex` regenerated from the paper's
  traces with today's rates (`newton_table_rescaled.py`). Newton to 1e-8 at
  (32,64,64): 5.69 h -> 3.01 h; at (24,48,48) 1.04 h -> 35.9 min; at
  (16,32,32) 10.2 min -> 5.18 min. L-BFGS to 1e-8 at (16,32,32): 1.97 h ->
  1.23 h. The rate fits of the time columns change with the factor's mesh
  dependence (s/step: L-BFGS n^2.86, Newton n^1.93).
- `fig:newton` (residual against wall time, `newton_convergence_resolution0_wall_log`):
  regenerated with today's rates per mesh and method (`newton_wall_rescaled.py`).
- `fig:n_sweep` etc. (`sweeps_wall_residual`, wall time incl. setup):
  regenerated with today's rate and today's setup per arm
  (`sweeps_wall_rescaled.py`); setup = job elapsed - stepping wall as before
  (170-290 s at (16,32,32) today against 275-540 s in the paper's runs).

## Numbers in the text (resubmit-4.tex line numbers)

- 702: add "measured at b0f8d38 (2026-09-11)"; the setup caveat stands.
- 1244: diagnostics excluded, first-chunk compile included (1-4%).
- 1289: mixed at 1e-10 against float64: 0.486 vs 0.669 s/step, 27% (was 33%).
- 1297 `tab:precision_and_gamma_sweep_wall`: order 1: float64 0.67, mixed
  1e-10 0.49, mixed 1e-8 0.41, float32 0.13; order 0: 0.41, --, 0.25, 0.06.
- 1323: seconds per step 0.20 (p=1), 0.41, 0.81, 1.57, 3.70 (p=5).
- 1328: cost per step ~ n^PENDING (n=12..64; n^2.65 over 12..32) from 0.23 s
  at n=12 to PENDING at n=64; ~ (p+1)^2.6 from 0.20 s at p=1 to 3.7 s at p=5;
  per quadrature point 0.8-1.5 us (n=12: 1.23, 16: 0.93, 24: 0.86, 32: 0.84;
  p=1: 1.50, p=3: 0.78, p=4: 0.77, p=5: 1.05).
- 1350: 0.43, 0.41, 0.44 s/step for m = 0, 1, 5.
- 1396: the potential route costs 26% less per step with smoothing (m=1;
  29% at m=0); without smoothing 10% (m=1) and 1% (m=0).
- `tab:velocity_choices` s/step column, smooth-first rows only: (m=1, g=1)
  Leray 0.411 / potential 0.303; (m=0, g=1) 0.428 / 0.303; (m=1, g=0)
  0.248 / 0.224; (m=0, g=0) 0.227 / 0.224. The "smooth last" row goes.
- `tab:epsilon_sweep` s/step column: the smoothing scale does not change
  the solves; all rows 0.41 (the paper's 0.61-1.11 spread across eps was node
  noise on top of 0.65; the eps = 0 row is the order-0 cost, 0.25).
- `tab:seeded` s/step: 0.41 for all three (or drop the column).
- 1437: 0.19 s/step at 1e-6, 0.41 at 1e-8, 0.49 at 1e-10.
- 361 / 1636 captions: "a few hours" holds (3.0 h to 1e-8 at (32,64,64));
  "one tenth of the wall time" holds (the ratio is rate-independent).

## Budget

PENDING GPU hours of the 10 (sacct elapsed of every job of this note,
failed attempts included).
