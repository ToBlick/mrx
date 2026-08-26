# MRX documentation

MRX solves MHD equilibrium and relaxation problems with a B-spline finite
element exterior calculus in JAX. The paper is arXiv:2510.26986.

## This directory

Exposition of the code as it is. Every identifier named here exists in
`mrx/` or `scripts/`.

| page | what it covers |
|---|---|
| [architecture.md](architecture.md) | spaces, extraction, Laplacians, the static/dynamic data model, assembly order |
| [assembly.md](assembly.md) | the matrix-free mass apply, the metric weights, memory, the `p + 1` quadrature rule |
| [preconditioning.md](preconditioning.md) | which solver and preconditioner per operator; what `metric_lumping` is |
| [polar.md](polar.md) | pole regularity and the strong derivative on the polar complex |
| [precision.md](precision.md) | `MRX_DTYPE`, `mrx.eps`, the solver tolerance, float32 |
| [relaxation.md](relaxation.md) | the relaxation loop, initial conditions, geometries, `scripts/relax.py` |
| [gvec_mrx_interface.md](gvec_mrx_interface.md) | what MRX needs from a GVEC export |
| [manufactured_solutions.md](manufactured_solutions.md) | exact solutions of the toroidal Poisson cases |
| [PRODUCTION.md](PRODUCTION.md) | one page: what runs today |
| [dev/testing_strategy.md](dev/testing_strategy.md) | test conventions |

`docs/research/` is the campaign record: handoffs, plans, measurements,
refuted approaches. Its `README.md` indexes it by topic and `OPEN.md` lists
every open item once. `docs/source/` is the Sphinx tree with the tutorials.

## Running tests

Every run, including the test suite, is a GPU job:

```
SCRIPT="-m pytest -q" JOB_NAME=tests TIMEOUT_MIN=120 bash slurm/run.sh
```

`slurm/run.sh` reads `SLURM_ACCOUNT` and `SLURM_PARTITION` from the
environment or from the gitignored `slurm/site.env`, exports
`PYTHONPATH=$MRX_ROOT`, and prints `mrx from: <path>` as the first log line.
Read that line before the result. Test conventions: `test/conftest.py` builds
the shared sequences; `pyproject.toml` sets `testpaths = ["test"]` and
deselects the `gpu` tier by default (see `slurm/README.md`).

## Production settings

`PRODUCTION.md` lists them. In code: `PRODUCTION_BC_SCALE` in
`mrx/metric_lumping_laplacian.py`, the cut-off constants at the top of
`mrx/preconditioners.py`, `build_sequence` in `mrx/geometries.py` for the
canonical sequence and operator build, `NumericsConfig` in `mrx/config.py`
for the Hydra entry points, and the environment variables `MRX_DTYPE` and
`MRX_BJ_BC_SCALE` (`MRX_DATA` only locates the gpu-tier data test's file).
