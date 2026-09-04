# Running on a cluster

Every MRX run, including the test suite and smoke tests, is a GPU job
submitted through `slurm/run.sh`.

## Site settings

`slurm/run.sh` reads the account and partition from the environment,
never from the repository:

```bash
export SLURM_ACCOUNT=<account>
export SLURM_PARTITION=<gpu partition>
export SLURM_EXCLUDE=<node,node>      # optional
```

Put them in `slurm/site.env`, which is gitignored and sourced by
`run.sh`, or export them in the shell.

## One script: `slurm/run.sh`

```bash
SCRIPT=scripts/relax.py ARGS="--geometry data/torus.json --steps 50" JOB_NAME=smoke bash slurm/run.sh
SCRIPT="-m pytest -q test" JOB_NAME=tests TIMEOUT_MIN=30 bash slurm/run.sh
```

The job activates the virtualenv, exports `PYTHONPATH=$MRX_ROOT`, prints
`mrx from: <path>`, and runs `python -u $SCRIPT $ARGS`. The log is
`$MRX_ROOT/outputs/$OUTSUB/<date>/<time>/$JOB_NAME.log`.

| variable | meaning | default |
|---|---|---|
| `SCRIPT` | path relative to `MRX_ROOT`, or `-m module` | required |
| `ARGS` | arguments passed to the script | |
| `JOB_NAME` | job name and log file stem | `run` |
| `OUTSUB` | log directory under `outputs/` | `JOB_NAME` |
| `TIMEOUT_MIN` | wall time in minutes | 60 |
| `MEM_GB` | host memory | 64 |
| `CPUS` | CPUs per task | 32 |
| `EXTRA_ENV` | space-separated `VAR=VALUE` pairs exported in the job, for example `MRX_DTYPE=float32` | |
| `MRX_ROOT` | the checkout to run | the repository containing `run.sh` |
| `MRX_VENV` | the virtualenv | `$MRX_ROOT/.venv`, then the main checkout's |

`bash slurm/waitjob.sh <JOBID> <log>` blocks until the job leaves the
queue and prints the head and tail of the log.

The test suite is one such job, `SCRIPT="-m pytest -q test"`; everything
it reads is tracked. `EXTRA_ENV="JAX_PLATFORMS=cpu"
CPUS=4` measures the suite as the GitHub runner sees it (see
`slurm/README.md`).

## The Poisson convergence study

`scripts/poisson_study.py` sweeps the eight Hodge Laplacians on the toroid
over the resolutions in `--n`, one after another in one process, and writes
`<out>/result.json` after every resolution. One GPU job:

```bash
SCRIPT=scripts/poisson_study.py ARGS="--p 3 --n 8 16 32" \
  JOB_NAME=poisson MEM_GB=80 TIMEOUT_MIN=240 bash slurm/run.sh
```

A sweep over degrees is one job per `--p`. `--tol` (default `1e-9`, the
tolerance of the archived numbers) and `--precision` are the only numerics
knobs; `python scripts/poisson_study.py --help` lists the rest.

## Worktrees

A git worktree has no `data/` and no `.venv`. Geometries are passed by
path (`--geometry /path/to/mrx/data/GVEC_State_final.dat`), so no link is
needed for a run, and the suite reads only tracked files, so a worktree
runs it as it is.

`run.sh` defaults `MRX_ROOT` to the repository containing it, so a
worktree runs itself, and it exports `PYTHONPATH=$MRX_ROOT` so that the
editable install of the main checkout does not shadow it. A job you
submit any other way needs `export PYTHONPATH=$WT` yourself. Check the
`mrx from:` line of the log before reading a result.
