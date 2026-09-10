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
| `EXTRA_ENV` | space-separated `VAR=VALUE` pairs exported in the job, for example `MRX_DTYPE=float64` | |
| `MRX_ROOT` | the checkout to run | the repository containing `run.sh` |
| `MRX_VENV` | the virtualenv | `$MRX_ROOT/.venv`, then the main checkout's |

`bash slurm/waitjob.sh <JOBID> <log>` blocks until the job leaves the
queue and prints the head and tail of the log. `bash slurm/suite.sh`
submits the test suite in its three precision configurations
([Testing strategy](concepts/testing_strategy.md)); `EXTRA_ENV="JAX_PLATFORMS=cpu" CPUS=4`
measures the suite as the GitHub runner sees it.

## Worktrees

A git worktree has no `data/` and no `.venv`. Geometries are passed by
path, so no link is needed for a run, and the suite reads only tracked
files, so a worktree runs it as it is. `run.sh` defaults `MRX_ROOT` to the
repository containing it, so a worktree runs itself, and it exports
`PYTHONPATH=$MRX_ROOT` so that the editable install of the main checkout
does not shadow it. A job you submit any other way needs
`export PYTHONPATH=$WT` yourself. The first line of every log is
`mrx from: <path>`; read it before reading the result.
