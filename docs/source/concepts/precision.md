# Precision

MRX runs in one floating-point precision, chosen once per process.

## The switch

Set `MRX_DTYPE` to `float64` (the default) or `float32` before importing
`mrx`. `mrx/precision.py` reads it, sets `jax_enable_x64` accordingly, and
sets `jax_default_matmul_precision` to `"highest"` so that float32 dot
products run at full float32 precision rather than TF32. Nothing else in the
package touches `jax_enable_x64`.

The module exports, re-exported from `mrx`:

| name | value |
|---|---|
| `mrx.DTYPE` | the working dtype |
| `mrx.EPS` | its machine epsilon |
| `mrx.eps(c=1.0)` | `c * EPS` |
| `mrx.sqrt_eps(c=1.0)` | `c * sqrt(EPS)` |

Every array MRX creates uses `mrx.DTYPE`. Every tolerance that depends on
roundoff is written as `eps(c)` or `sqrt_eps(c)`: the rank cut-offs of the
preconditioners (see [preconditioning.md](preconditioning.md) section 6),
endpoint nudges `sqrt_eps() * h`, and the solver tolerance. Tolerances that
encode a physical or algorithmic choice (a force residual, an ODE
controller, a shift) stay ordinary parameters.

## Solver tolerance

`DeRhamSequence(tol=None)` and every solver in `mrx/solvers.py` default to
`sqrt_eps()`: `1.5e-8` in float64, `3.5e-4` in float32. That is the
relative residual a solve can reach when the matvec itself is rounded.
`scripts/poisson_study.py --tol` defaults to `1e-9` because the archived
convergence numbers were measured there. `scripts/relax.py` takes
`--precision` (default float32) and stops when the mean over `--floor-steps`
steps of the relative force residual drops below `--floor-tol` (default
`1e-3`).

The scripts set `MRX_DTYPE` from `--precision` before importing `mrx`.

## What float32 does

Measured on 2026-08-26, toroid and W7-X, `(8,16,8)`, `p=3`:

- Every preconditioner builds and every solve converges. Mass solves take
  the same iteration counts as float64 for k=0..3; the k=0 Laplacian too.
- The k=1 MINRES solves take 11% (Dirichlet) to 33% (free) more iterations.
- The Poisson cases at n=8 and n=16 all converge, with iteration counts up to
  1.5 times the float64 ones.
- A 200-step CG relaxation on W7-X runs monotone at 0.37 s per step against
  0.73 in float64, with `E = 0.486657` against `0.486666` and
  `||div B|| = 3e-7`.
- Without `jax_default_matmul_precision="highest"` the spline derivative
  contractions of the W7-X map lose digits in TF32 and `det DF` goes negative
  at the axis. The setting fixes it; the Jacobian then matches float64 to
  `1e-5`.

- Near an equilibrium the energy decrease per step is below the float32
  resolution: on the W7-X Clebsch initial condition the whole descent removes
  `2.4e-4` of `E`. The force residual in float32 floors at the
  solve-tolerance level, `~2e-3` at tol `1e-5` (the table below), so a
  `--floor-tol` below that never fires; the run ends on `--steps` or
  `--seconds`, and the float64 run continues to step 3000 with the same
  nested surfaces.
- Resistive increments `eps = dt * eta` of `1e-7` are a few ulps of `B`.
  Use `--eta-every K` (`K` of 10 to 100 at `eta ~ 1e-4`) so each solve applies
  a representable increment.

Measured on the same Clebsch relaxation after 111 steps, both states evaluated
in float64 (mass norms, force, helicity):

| quantity | float32 vs float64 |
|---|---|
| `||B32 - B64||_M / ||B64||_M` | 1.1e-5 at the initial condition, 5.4e-4 at step 111 |
| `|E32 - E64| / E` | 5.6e-7 |
| force norms | 2.06e-3 vs 2.46e-3; the force *vectors* differ by 126 % |
| helicity | absolute difference 3e-7 (relative 1.6e-3: H itself is 1.8e-4) |
| `||div B|| / ||B||` | 7e-5 vs 6e-12 (each at its own solve tolerance) |

The float32 run follows the same descent to the same energy, but its residual
force is not resolved below the solve-tolerance floor (~2e-3 here). Going
further in float32 needs a float64 state with float32 operators, not a
tolerance.

Every test tolerance is expressed through `eps()` or the solver tolerance; the
suite passes in both precisions.
