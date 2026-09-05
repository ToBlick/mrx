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

## Mixed precision (e4220d5: float32 Krylov, float64 residual, tol 1e-8)

Measured first at (12,24,24) p=3 (`outputs/prune_smoke/probe_mp.py`): the
force's divergence is 3e-9 of the divergence of `J x B` (5e-5 of the
force at the initial field, 1.6e-6 at step 300), the smoothed velocity is
divergence-free to 1e-5, and the step costs 1.16 s against 0.25 before:
a warm-started solve now closes six decades instead of one and a half,
at about a hundred MINRES iterations per decade on the k=3 saddle. The
velocity Leray is 1357 of the step's iterations and removes a 1e-5
remnant. Three arms at (16,32,32) decide the rest: with and without the
velocity Leray at tol 1e-8, and without it at tol 1e-6 (results below
when they land; `arm.py with|without float32 _mixed [tol]`).

## What follows

- float32 alone keeps the velocity Leray; the residual floor in float32
  alone is the solver tolerance in any case (the same remnant is inside
  `||F||`).
- In float64 and in mixed precision the remnant is 1e-5 or below
  relative to `F` and `<g, sigma>` is below the descent by the same
  factor: the second projection can go. Which of float64 and mixed
  precision is cheaper per step at equal floor is what the arms decide.
