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
relative residual a solve can reach when the matvec itself is rounded. The
Hydra configs carry it as `NumericsConfig.solver_tol` (`mrx/config.py`),
passed as `DeRhamSequence(tol=cfg.solver_tol)`; `None` is the default above.
`conf/config_poisson_test.yaml` pins `solver_tol: 1.0e-9` because the archived
convergence numbers were measured there. `scripts/relax.py` takes
`--precision` and stops on a windowed energy decrease below `--floor-tol`,
whose default is `10 * eps()` of the working dtype.

The Hydra entry points set `MRX_DTYPE` from `precision=` before importing
`mrx` and raise if `cfg.precision` disagrees with `mrx.DTYPE`.

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

Tests that check an identity at `1e-12` are meaningless in float32; the test
suite is float64 until those tolerances are expressed through `eps`.
