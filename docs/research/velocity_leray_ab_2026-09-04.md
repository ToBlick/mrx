# The velocity Leray projection: A/B at li383 (16,32,32) p=3, 2026-09-04

Question (Tobias): the step projects the force (`compute_force`) and then
the smoothed descent velocity again (`_ideal_increment`); divergence
commutes with `(M + eps L)^-1 M` and the L-BFGS direction combines
projected forces, so in exact arithmetic the second projection is the
identity. Do we need both?

## Setup

`outputs/velocity_leray_ab/arm.py with|without`: the production step
(explicit, L-BFGS m=1, cfl 0.5, g=1 smoothing at `mu = 0.064 / n_r^2`,
`auxiliary_B_field` off), li383 (16,32,32) p=3, 2000 steps in chunks of
200, at b8701a1 (the sigma-warm-started force Leray). The `without` arm
wraps `seq.apply_leray_projection` after the initial field is built so
that only the calls from inside `compute_force` project; the velocity's
call returns its input. Figure `outputs/velocity_leray_ab/velocity_leray_ab.png`
(black solid with, teal dashed without; 100-step block means with sd
ribbons), numbers from `ab_numbers.py`.

## float32 (the package default)

| | with | without |
|---|---|---|
| s/step (steady) | 0.478 | 0.136 |
| resid at step 500 / 1000 / 1500 | 1.08e-3 / 7.0e-4 / 7.1e-4 | 1.52e-3 / 1.11e-3 / 1.05e-3 |
| 100-step mean resid at the end | 6.8e-4 | never reaches 6.8e-4 |
| sum of predicted dE (line search) | -1.6e-6 | -2.1e-6 |
| E_0 - E_end (measured) | +5.7e-7 | -8.9e-7 |
| dH/H_0 at the end | 1.2e-5 | 4.6e-6 (both at the float32 level, not a discriminator) |

The arm without the projection is 3.5x faster per step but is no longer
an energy descent: the predicted decrease does not happen and the energy
ends above where it started, while the residual at equal step is 40-57%
higher from step 500 on. The reason is the line-search identity. The
energy change of a step is `-dt <u, J x B>_M + dt^2 |dB|^2 / 2`, and `<u,
J x B> = <u, F> + <u, sigma>` with `sigma = grad p` the gradient part;
the second term vanishes only for a divergence-free `u`. In float32 the
force Leray leaves a divergence remnant of O(1) relative to `F` (tol
relative to `|J x B|`, and `|F| / |J x B| ~ 1e-3`; see
`simplify_findings_2026-09-04.md`, "div f = 0, measured"), so the
velocity carries a gradient part `g` and the missed term `dt <g, sigma>`
is of the order of `|g| |grad p|`, a hundred times `|F| |u|`: a random
walk in energy of the size of the descent itself. The velocity Leray is
what removes `g`; in float32 it is not optional.

## float64 (working precision float64, plain solves, tol 1.5e-8)

| | with | without |
|---|---|---|
| s/step (steady) | 2.468 | 0.946 |
| resid at step 500 / 1000 / 1500 / 2000 | 8.1e-4 / 6.9e-4 / 5.1e-4 / 4.5e-4 | 8.0e-4 / 5.5e-4 / 5.1e-4 / 5.0e-4 |
| 100-step mean resid at the end (with) | 4.67e-4 | reached at step 1939 |
| sum of dE (measured) | -1.476e-6 | -1.477e-6 |
| steps with dE > 0 | 0 | 0 |
| dH/H_0 at the end | 6.3e-6 | 6.2e-6 (the explicit scheme's own drift, equal in both) |

Identical descent, 2.6x faster per step without the projection: in
float64 the velocity Leray is redundant, as the algebra says. Both float64
arms end below the float32 arms (4.5e-4 to 5.0e-4 against 8.3e-4 to
8.4e-4 at step 2000): the float32 arms were tolerance-limited. Figure
`outputs/velocity_leray_ab/velocity_leray_ab_f64.png`.

## Mixed precision (e4220d5: float32 Krylov, float64 residual)

Measured first at (12,24,24) p=3 (`outputs/prune_smoke/probe_mp.py`): the
force's divergence is 3e-9 of the divergence of `J x B`, the smoothed
velocity is divergence-free to 1e-5, and the step costs 1.16 s against
0.25 before: a warm-started solve now closes six decades instead of one
and a half, at about a hundred MINRES iterations per decade on the k=3
saddle. Then three arms at (16,32,32), 2000 steps, `arm.py with|without
float32 _mixed [tol]`:

| | with, tol 1e-8 | without, tol 1e-8 | without, tol 1e-6 |
|---|---|---|---|
| s/step (steady) | 4.892 | 1.036 | 0.545 |
| resid mean, steps 501-1000 / 1001-1500 / 1501-2000 | 7.2e-4 / 5.6e-4 / 4.8e-4 | 7.2e-4 / 5.6e-4 / 4.8e-4 | 6.9e-4 / 5.5e-4 / 4.7e-4 |
| resid at step 2000 | 4.8e-4 | 4.7e-4 | 4.4e-4 |
| E removed, steps 1-500 / 501-1000 | 1.405e-6 / 4.5e-8 | 1.407e-6 / 3.7e-8 | 1.399e-6 / 3.1e-8 |

Identical descent with and without the projection, as in float64, and the
same at tol 1e-6 over these 2000 steps. Figure
`outputs/velocity_leray_ab/velocity_leray_ab_mixed.png`; per-block numbers
from `descent_check.py`.

**Two things the float64 arms add.** (1) The trace's `|dE - dE_ls| /
|dE|` is the velocity's gradient part against `grad p` relative to the
descent; in float64 it is 1.4e-4 at resid 1.5e-3 and 9.4e-3 at resid
4.8e-4 (without the projection, tol 1.5e-8), i.e. `0.1 tol / resid^2` on
li383: the term reaches the size of the descent near resid 3e-4 at tol
1e-6 and near 3e-5 at tol 1e-8, which is why the default stays 1e-8 and
a run aimed below 1e-4 wants 1e-10 or float64. (2) In float32 storage the
per-step `dE` is noise even in mixed precision: it disagrees with `dE_ls`
at the 100% level and is positive on 40% of the steps, while the block
sums agree (the rounding of the stored field per step is the size of the
descent increment's energy); in float64 the two agree to 1e-6 and no
step increases the energy. Float32 energy figures use block sums.

**Speed.** Mixed precision at tol 1e-8 without the projection, 1.04
s/step, is not faster than float64 without it, 0.95 s/step at tol
1.5e-8: at these meshes the step is launch-latency-bound, not
bandwidth-bound, so float32 buys memory (half the operators and state),
not time. Against the float32 run that had to keep the projection (0.478
s/step, tolerance-limited at 8e-4), mixed precision without it is 2x
slower per step and reaches the float64 floor.

## The production step on the current code (d933ffd and later, no velocity projection)

`arm.py step <precision>`, li383 (16,32,32) p=3, 2000 steps, the same
settings; `MRX_RESIDUAL_DTYPE=float32` for the plain float32 arm. Per
500-step block: the energy removed against the line search's prediction
(their difference is the velocity's gradient part against `grad p`), and
the mean residual.

| | float32 refined, tol 1e-8 | float64, tol 1e-10 | float32 plain, tol 1e-6 |
|---|---|---|---|
| s/step (steady) | 1.022 | 1.438 | 0.326 |
| resid mean, blocks 2 / 3 / 4 | 7.1e-4 / 5.6e-4 / 4.9e-4 | 7.1e-4 / 5.5e-4 / 4.9e-4 | 7.3e-4 / 5.7e-4 / 4.9e-4 |
| resid at step 2000 | 4.5e-4 | 4.0e-4 | 4.3e-4 |
| E removed / predicted, block 2 | 3.9e-8 / 3.9e-8 | 3.59e-8 / 3.59e-8 | 1.5e-8 / 3.7e-8 |
| E removed / predicted, block 4 | 2.0e-8 / 1.4e-8 | 1.25e-8 / 1.25e-8 | 0.2e-8 / 1.5e-8 |
| `\|dE - dE_ls\| / \|dE\|`, block 4 | 0.98 (storage noise) | 1.3e-4 | 1.01 |
| steps with dE > 0, block 4 | 212 | 0 | 254 |

All three descend the residual alike over these 2000 steps. In float64
the energy removed equals the prediction to 1e-4 in every block and no
step increases the energy. In float32 storage the per-step `dE` is noise
(the stored field's rounding), but the block sums tell the two float32
configurations apart: refined, the sums match the prediction (3.9e-8
against 3.9e-8, then 2.0e-8 against 1.4e-8); plain, they fall short by
2.5x, 4x and 7x in successive blocks, the gradient-part term of the
scaling law arriving at tol 1e-6 as the residual approaches 3e-4. Plain
float32 is the configuration for runs that stop near 5e-4; refined
float32 and float64 for anything deeper, at 3-4x the cost per step.
The float64 arm at the new default 1e-10 costs 1.44 s/step against 0.95
at 1.5e-8 for the same residual descent: the tolerance is the cost knob.

## Decision (2026-09-05)

The velocity Leray projection leaves the step (relaxation.py,
`_ideal_increment`): the float32-alone configuration that needed it no
longer exists, every solve being refined. `SOLVE_TOL` stays 1e-8 at
float32 and 1e-10 at float64. For the sweeps: float64 and mixed
precision cost the same per step at n <= 32 and reach the same floors;
float64 has a clean per-step energy trace.
