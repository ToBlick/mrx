# Running on a cluster

On a slurm cluster, an MRX run, including the test suite, is a GPU job
submitted through `slurm/run.sh`. Small runs such as the tutorials also run
on a CPU without it.

## Site settings

`slurm/run.sh` reads the account and partition from the environment:

```bash
export SLURM_ACCOUNT=<account>
export SLURM_PARTITION=<gpu partition>
export SLURM_EXCLUDE=<node,node>      # optional
```

Put them in `slurm/site.env`, which is gitignored and sourced by
`run.sh`, or export them in the shell. `slurm/site.env.example` is a
template.

## The data root: `MRX_DATA`

Runs and large input files can live in one directory outside the
repository. Set `MRX_DATA` to it, in `slurm/site.env` or in the shell:

```bash
export MRX_DATA=/path/to/mrx-data
```

- `$MRX_DATA/data` holds input files that are not tracked in git. A
  geometry file that is not found at the path given is looked up there by
  its name.
- `$MRX_DATA/runs/mrx` takes the place of `outputs/`: the job logs of
  `run.sh`, the default run directory of `scripts/relax.py` and the run
  directories of the tutorials.
- `$MRX_DATA/cache/xla` holds the compilation cache of the test suite.

Without `MRX_DATA` nothing changes: runs go to `outputs/` in the
repository and files are read from the path given.

## One script: `slurm/run.sh`

```bash
SCRIPT=scripts/relax.py ARGS="--geometry.path data/wout_li383_low_res_reference.nc --geometry.resolution 8 12 12 --budget.steps 50" JOB_NAME=smoke bash slurm/run.sh
SCRIPT="-m pytest -q test" JOB_NAME=tests TIMEOUT_MIN=60 bash slurm/run.sh
```

The job activates the virtualenv, exports `PYTHONPATH=$MRX_ROOT`, prints
`mrx from: <path>`, and runs `python -u $SCRIPT $ARGS`. The log is
`$MRX_ROOT/outputs/$OUTSUB/<date>/<time>/$JOB_NAME.log`, or
`$MRX_DATA/runs/mrx/$OUTSUB/<date>/<time>/$JOB_NAME.log` with `MRX_DATA` set.

| variable | meaning | default |
|---|---|---|
| `SCRIPT` | path relative to `MRX_ROOT`, or `-m module` | required |
| `ARGS` | arguments passed to the script | |
| `JOB_NAME` | job name and log file stem | `run` |
| `OUTSUB` | log directory under `outputs/` (under `$MRX_DATA/runs/mrx` with `MRX_DATA` set) | `JOB_NAME` |
| `TIMEOUT_MIN` | wall time in minutes | 60 |
| `MEM_GB` | host memory | 64 |
| `CPUS` | CPUs per task | 32 |
| `EXTRA_ENV` | space-separated `VAR=VALUE` pairs exported in the job, for example `MRX_XLA_CACHE=/tmp/cache` | |
| `DEPENDENCY` | an `sbatch --dependency` spec | |
| `MRX_ROOT` | the checkout to run | the repository containing `run.sh` |
| `MRX_VENV` | the virtualenv | `$MRX_ROOT/.venv`, then the main checkout's |

`bash slurm/waitjob.sh <JOBID> <log>` blocks until the job leaves the
queue and prints the head and tail of the log. `bash slurm/suite.sh`
submits the test suite in its two precisions
([Testing strategy](concepts/testing_strategy.md)).

## Worktrees

A git worktree has no `.venv`, so `run.sh` falls back to the main checkout's.
`MRX_ROOT` defaults to the repository containing `run.sh`, so a worktree
runs itself, and `PYTHONPATH=$MRX_ROOT` keeps the editable install of the
main checkout from shadowing it. A job submitted any other way needs
`PYTHONPATH` set by hand. Read the first line of the log, `mrx from:
<path>`, before reading the result.
