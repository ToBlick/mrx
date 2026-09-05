# Floor study: pollution or dynamics? (plan, on hold 2026-09-05)

Question (Tobias): how much are the relaxation floors set by the
precision-related pollution effects, and how much by the dynamics
themselves? Held before launch: more code to fix first.

## The pollution effects

The force `F = J x B - grad p` from the force Leray carries the pressure
solve's residual, which is relative to `|J x B|` while `F` is `resid`
times that: `div F / |F| ~ tol / resid`. The smoothing solve passes it
through unchanged (the split adds no divergence of its own). Its energy
term is `0.1 tol / resid^2` of the descent (li383 float64,
`velocity_leray_ab_2026-09-04.md`), a tenth of the descent at `resid =
sqrt(tol)`, the whole descent at `sqrt(0.1 tol)`:

| configuration | tol | term = descent near | float32 storage cap |
|---|---|---|---|
| plain float32 (`MRX_RESIDUAL_DTYPE=float32`) | 1e-6 | 3e-4 | 1e-5 |
| mixed (float32 work, float64 residual) | 1e-8 | 3e-5 | ~1e-6 (rounding of the stored `F`: a discrete divergence of ~1e-6 relative to `|F|`) |
| float64 | 1e-10 | 3e-6 | none |

The velocity Leray projection (removed 2026-09-05) took the remnant out
relative to `|u|`; in mixed and float64 a force tolerance scaled by
`resid` is its single-solve equivalent (converges to ~1e-13 relative on
the float64 residual); in plain float32 the projection is the only
formulation whose floor scales with the force (its rhs is `M u`).

## The dynamics

Ideal descent is a power law with no stall, `resid ~ t^-a`, `a = 0.2` at
(16,32,32) p=2 with order-1 smoothing (2026-09-03). From 4.5e-4 at 2000
steps (the A/B at p=3) that puts 3e-4 near 15,000 steps and 3e-5 / 3e-6
some 1e5-1e6 times further than any budget. Expected answer: plain
float32 is pollution-limited, mixed and float64 dynamics-limited; the
study quantifies the margin. Order-0 smoothing descends 8-16x slower per
step and shows the dynamics floor at a higher residual.

## Arms and settings

li383 (`data/wout_li383_1.4m.nc`), `--ns 16,32,32 --p 2`, file initial
condition, `--history 1` (L-BFGS m=1), `--cfl 0.5`, `--chunk 500`,
`--floor-tol 0` (every arm runs to its step cap), `--steps 20000`.

| arm | precision | env | smoothing |
|---|---|---|---|
| f32_g0 | `--precision float32` | `MRX_RESIDUAL_DTYPE=float32` | `--velocity-smoothing-order 0` |
| f32_g1 | same | same | `--velocity-smoothing-order 1 --velocity-smoothing-scale 2.5e-4` (0.064 / 16^2) |
| mixed_g0 | `--precision float32` | (default) | order 0 |
| mixed_g1 | same | | order 1, 2.5e-4 |
| f64_g0 | `--precision float64` | | order 0 |
| f64_g1 | same | | order 1, 2.5e-4 |

Output `outputs/floor_study/<arm>/`; `slurm/run.sh` with
`SCRIPT=scripts/relax.py`, `EXTRA_ENV` for the residual dtype,
`PYTHONPATH` the worktree.

## Budget

Step times from the p=3 measurements at this mesh (plain float32 0.33 s,
mixed 1.02 s, float64 1.44 s; order 0 cheaper, no shifted solve) scaled
to p=2: about 0.2-0.3 / 0.6-1.0 / 0.9-1.4 s. 20,000 steps: about 18 GPU
hours expected, 30 worst case, six jobs in parallel, ~8 h wall for the
float64 arms. `TIMEOUT_MIN` at twice the estimate: 360 / 720 / 960.
Open choices: 10,000 steps (halves the budget, the plain float32 floor
near 15,000 steps may not be reached); 40,000 for the order-0 arms.

## What to measure

Per 100-step block (float32 storage makes the per-step `dE` noise, block
sums are the diagnostic): the residual; `|dE - dE_ls| / |dE|`, the
pollution term over the descent, which locates the crossing even where
the run does not reach it; `|div F| / |F|`; the power-law fit of the
residual against steps and its extrapolation to the budget (the
dynamics floor). Deliverable: one house-style figure (block means,
log-log; residual and the pollution ratio per arm) and a note
`docs/research/floor_study_2026-09-05.md`.
