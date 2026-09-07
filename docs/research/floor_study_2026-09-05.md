# Floor study: pollution or dynamics? (2026-09-05)

Plan: `floor_study_plan_2026-09-05.md`. li383 `wout_li383_1.4m.nc`, (16,32,32)
p=2, L-BFGS m=1, CFL 0.5, 20000 steps in chunks of 500, floor 0; six arms:
{plain float32 (tol sqrt(eps) = 3.5e-4), mixed (float32 work, float64
residual, tol 1e-8), float64 (tol 1e-10)} x {smoothing order 0, order 1 at
the default scale 0.02 / n_r^2}. `outputs/floor_study/<arm>/`; figure
`floor_study.png` (`floor_figures.py`: 100-step block means of the force
residual, the energy removed from the stored field), energies from the
checkpoints (`floor_energy.py`; since `532cca4` the sampler records the
energy of the stored field in float64 every chunk, `qoi["E"]`).

## The answer

| arm | tol | resid at 20000 (chunk mean) | E_0 - E | dH | s/step | GPU h |
|---|---|---|---|---|---|---|
| float64, order 1 | 1e-10 | 8.9e-5 | 1.92e-6 | -5.3e-8 | 1.21 | 6.7 |
| mixed, order 1 | 1e-8 | 8.8e-5 | 1.91e-6 | -5.3e-8 | 0.65 | 3.6 |
| plain float32, order 1 | 3.5e-4 | 1.02e-4 | 1.35e-6 (rose by 5e-7 after 8000) | -5.3e-8 | 0.23 | 1.3 |
| float64, order 0 | 1e-10 | 4.1e-3 | 4.85e-6 | +4.3e-6 | 0.71 | 3.9 |
| mixed, order 0 | 1e-8 | 4.0e-3 | 4.77e-6 | +4.3e-6 | 0.39 | 2.1 |
| plain float32, order 0 | 3.5e-4 | 3.2e-2 | -5.2e-5 (above the start since 8500) | -4.5e-7 | 0.09 | 0.5 |

Every arm ran its 20000 steps (`stopped on: steps`); the residual is the
last chunk's mean, `E_0 - E` from the stored fields (the trace's sum to
1e-9), `dH` the helicity change over the run. Second realisations of the
two plain float32 arms (`f32_g1_flush`, `f32_g0_flush`, cold optimiser
restart every chunk): 1.03e-4 with the energy 1.6e-6 ABOVE the start, and
6.5e-4 with 3.0e-6 removed and no blow-up.

- **Order 1: dynamics, not pollution, in every precision that has a float64
  residual.** Float64 and mixed lie on one curve in residual and in energy
  (within 2% at every chunk), a power law `resid ~ steps^-a` with `a` =
  0.5-0.6 and no floor to 20000 steps; the pollution term on the float64 arm
  is 1e-8 to 1e-6 of the descent, four decades below the 0.1 line.
- **Plain float32, order 1: pollution, and it shows in the energy first.** The
  residual follows the same power law as the other two to 1e-4 -- and the
  energy removed falls behind from 2000 steps on, then RISES by 5e-7 (a
  quarter of what had been removed) between 8000 and 9000 steps while the
  residual keeps falling. At that point `tol / resid` is 2: the gradient
  remnant is twice the force, and the direction reduces |F| while walking
  uphill in energy. The residual can lie in plain float32; the energy from
  the stored field cannot (the trace's sum of `dE` agrees with it to four
  digits, so the trace had not drifted).
- **Plain float32, order 0: the descent turns around.** To 1e-3 by 8000
  steps, then the energy rises above the start (by 5e-5 at 20000 steps, 25x
  the energy ever removed), the residual to 3e-2, twice the initial value.
  The precursor is the descent quality: dt shrinks from 2.24 to 1.67 and the
  descent cosine from 0.08 to 0.01 over 6000-8500 steps.
- **The plain-float32 events are stochastic, not accumulated.** A cold
  restart from the stored field (`floor_restart_fresh.py`) did not
  reproduce either event, and for a while that read as "the accumulated
  optimiser state"; it was not. A second realisation of each arm (run with a
  cold restart at every chunk, `f32_g0_flush`, `f32_g1_flush`) differs from
  the first in its FIRST chunk, before any restart: the order-1 arm's energy
  went UP by 1.85e-6 in its first 500 steps where the original's went down
  by 1.63e-6 (float64 energies of the stored fields; the float32 sampled
  energy, the host reduction and the trace's sum agree to 1e-8). Plain
  float32 trajectories on li383 diverge from round-off within tens of
  steps, so every run is its own realisation, and a single restart proves
  nothing. What the realisations share: the direction error is `tol /
  resid` of the force (27% at 1.3e-3, 200% at 1.7e-4), the energy is a
  random walk at the 1e-6 level from the first chunk, and the residual
  keeps its power law regardless. The order-0 blow-up is the fragile
  unsmoothed descent (below) meeting that error: the line search's pairing
  of force and direction loses its sign in one realisation at 8500 steps
  and not in the other over 20000 (that one sits on a plateau near 6e-4
  from 10000 steps, energy flat at 3.0e-6 removed, helicity drift steady at
  -1e-7, no reconnection). The second realisations ran with a cold restart
  of the optimiser state at every chunk, so with one run each they separate
  neither the noise from the restart nor a fate from a coincidence; only an
  ensemble would. A per-chunk flush of the warm starts was tried on the
  accumulation reading and removed again, unsupported. Mixed and float64
  order 0, by contrast, agree with each other chunk by chunk to 20000
  steps: their fate below is not realisation-dependent.
- **Order 0 in any precision: the unsmoothed descent stops being ideal.**
  Float64 and mixed order 0 are identical chunk by chunk: they flatten near
  6e-4 at 2500-4500 steps with the descent cosine at 0.003-0.004 (order 1:
  0.2), then from ~10000 steps the helicity drift turns positive and grows
  -- -8.7e-8 at 5000, +1.5e-7 at 10000, +4.4e-7 at 11000, +9.1e-7 at 12000,
  +4.3e-6 at 20000, the same numbers in float64 and mixed to two digits, a
  hundred times the order-1 arms' -5e-8 and 0.1% of H -- while the energy
  release accelerates (2.4e-6 removed at 10000, 4.8e-6 at 20000) and the
  residual climbs to 4.0e-3 (mixed) / 4.1e-3 (float64) at 20000 steps. That is numerical reconnection: the rough unsmoothed velocity's
  explicit step no longer conserves helicity, the topology changes, the
  energy it frees drops E faster and the field is far from any
  equilibrium. The smoothed descent holds the drift at 5e-8 over 20000
  steps in every precision. The order-0 arms remove more energy than the
  order-1 arms from the start (2.0 against 1.75e-6 at 2500) while sitting
  at three times the residual: the smoothed descent trades energy descent
  for residual descent, and it is the one that stays ideal.

## Consequences

- The pollution law's coefficient (`0.1 tol / resid^2`, measured on float64
  arms at tol 1.5e-8) does not transfer to plain float32: the order-0 turn
  came at resid 1e-3 where the law predicts the term at 35x the descent, and
  the order-1 arm ran to 1e-4 with the energy as the only witness. The
  mechanism is the law's; the number is not.
- Plain float32 (the TPU configuration) needs, beyond the residual, the
  energy of the stored field as the witness (the residual alone can lie
  there); nothing short of an accurate residual fixes the solves. See
  issue #21.
- A publication run is order 1 in mixed precision or float64; its floor
  within any budget is the descent's power law, not the solver. Order 0 is
  not an option for a long ideal run: it reconnects numerically after
  ~10000 steps at this mesh.
- The float32 storage of `E` resolves 3e-8 (one ulp of 0.5): late in a run
  the energy removed per 6000 steps is a few ulps, so E_0 - E is quantised
  there; the residual is the finer instrument for the rate, the energy for
  the sign.

Cost: 18 GPU hours for the six arms (the estimate was 16-18), plus 2 for
the two second realisations and 0.5 for the probes and restarts.
