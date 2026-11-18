# Running on a cluster

On a slurm cluster, an MRX run, including the test suite, is a GPU job
submitted through `slurm/run.sh`. Small runs such as the tutorials also run
on a CPU without it.

## Site settings

The account, the partition and the paths of a site do not live in the
repository. `slurm/run.sh` reads them from one file in your home,
`~/.config/mrx/site.env`, or from the file that `MRX_SITE_ENV` names, and
stops with a message if that file is missing. The same file serves every
MRX repository:

```bash
# ~/.config/mrx/site.env
export SLURM_ACCOUNT=<account>
export SLURM_PARTITION=<gpu partition>
# export SLURM_EXCLUDE=<node,node>
# export MRX_DATA=/path/to/mrx-data
```

## The data root: `MRX_DATA`

Runs, input files that are not tracked in git and caches live in one
directory outside the repository, best next to the checkout
(`<root>/mrx_data` beside `<root>/mrx`). Set `MRX_DATA` to it in
`~/.config/mrx/site.env`, which `run.sh` reads and passes on to the job,
and in the shell of runs started by hand:

```bash
export MRX_DATA=/path/to/mrx-data
```

MRX uses three directories in it. In Python only `mrx.paths` reads the
variable.

- `$MRX_DATA/runs/mrx` is the runs root. It holds the job logs of `run.sh`,
  the default run directory of `scripts/relax.py`, the run directories of
  the tutorials and the local copies of GKE jobs. `poincare_trace.py` and
  `poincare_plot.py` write into the run directory they read.
- `$MRX_DATA/data` holds input files that are not tracked in git. A
  geometry file that is not at the path given is looked up there by its
  name, so `--geometry.path data/<file>` works in every checkout and
  worktree.
- `$MRX_DATA/cache/xla` holds the compilation cache of the test suite.

Without `MRX_DATA` the runs root is `outputs/`, in the working directory for
the Python scripts and in the checkout for `run.sh`. Files are then read from
the path given, and the test suite caches in `outputs/xla_cache` of the
checkout. `outputs/` is gitignored.

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
