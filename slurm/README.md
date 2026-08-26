# Running MRX on the cluster

Every MRX run, including smoke tests, is a GPU job. `slurm/run.sh` submits one
script as a single-GPU job and prints the log path.

## slurm/run.sh

```
SCRIPT=scripts/relax.py ARGS="--geometry toroid --steps 50" JOB_NAME=smoke bash slurm/run.sh
SCRIPT="-m pytest -q test/test_solvers.py" JOB_NAME=tests TIMEOUT_MIN=120 bash slurm/run.sh
```

The job activates the venv, exports `PYTHONPATH=$MRX_ROOT`, prints
`mrx from: <path>` and then runs `python -u $SCRIPT $ARGS`. Logs go to
`$MRX_ROOT/outputs/$OUTSUB/<date>/<time>/$JOB_NAME.log`.

Variables: `SCRIPT` (required; a path relative to `MRX_ROOT` or `-m module`),
`ARGS`, `JOB_NAME` (default `run`), `OUTSUB` (default `JOB_NAME`),
`TIMEOUT_MIN` (60), `MEM_GB` (64), `CPUS` (32), `EXTRA_ENV` (space-separated
`VAR=VALUE` pairs exported in the job, for example `MRX_DTYPE=float32`).

## Site settings

`run.sh` reads the slurm account and partition from the environment, never
from the repository:

```
export SLURM_ACCOUNT=<account>
export SLURM_PARTITION=<gpu partition>
export SLURM_EXCLUDE=<node,node>      # optional
```

Put them in `slurm/site.env`, which is gitignored and sourced by `run.sh`, or
export them in the shell. The hydra configs under `conf/` read the same
variables through `${oc.env:...}` for their submitit launcher.

`MRX_ROOT` is the checkout to run; it defaults to the repository containing
`run.sh`, so a worktree runs itself. `MRX_VENV` is the virtualenv; it defaults
to `$MRX_ROOT/.venv`, then to the main checkout's `.venv`.

## Check the log

The first line of every log is `mrx from: <path>`. `mrx` is an editable
install pinned to the main checkout, so a job without `PYTHONPATH` imports the
main checkout's library while running a worktree's script, and reports the
result as a test of the worktree. Read that line before reading the result.

Hydra's submitit launcher does not export `PYTHONPATH` on its own. The configs
under `conf/` add `export PYTHONPATH=${oc.env:MRX_ROOT}` to the launcher
`setup` list, so `MRX_ROOT` must be set in the submitting shell for a hydra
multirun; a single run through `run.sh` needs nothing else.

## The test tiers

`pytest` (no arguments) is the CPU tier: it deselects the `gpu` marker
through `pyproject.toml`, needs no data files and runs in float64 and
float32 on the GitHub runners (`.github/workflows/ci.yml`). The `gpu` tier
holds the production-resolution fixture, the iteration-count bands measured
on it, the accuracy-at-resolution tests and everything that reads data
outside the repository. Run it nightly-style as a GPU job:

```
SCRIPT="-m pytest -q test -m gpu" JOB_NAME=tests_gpu TIMEOUT_MIN=60 MEM_GB=96 bash slurm/run.sh
```

and the whole suite (both tiers) with `ARGS='-m "gpu or not gpu"'`:

```
SCRIPT="-m pytest -q test" ARGS='-m "gpu or not gpu"' JOB_NAME=tests_all TIMEOUT_MIN=90 MEM_GB=96 bash slurm/run.sh
```

To measure the CPU tier as a runner would see it, run it inside a GPU job
with the GPU hidden and four cores:

```
SCRIPT="-m pytest -q test" JOB_NAME=tests_cpu CPUS=4 EXTRA_ENV="JAX_PLATFORMS=cpu" bash slurm/run.sh
```

`EXTRA_ENV="MRX_DTYPE=float32"` selects single precision for any of these.
The `gpu` tier is a float64 tier; it is not required to pass in float32.
Measured 2026-08-26 (H100, tree at the test-audit commit): the CPU tier is
249 items and passes in both precisions (11 min on the GPU in either
precision, 7:45 on four CPU cores -- it is compile-bound; the session torus
fixture is 50 s of that, the three relaxation-step tests 65 s and the
two-resolution Poisson order check 30 s); the `gpu` tier is 18 items and
passed in float64 in 7 min (the W7-X Clebsch fixture is 2.5 min of it).

Do not add `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false` to the CPU
recipe: XLA's CPU compiler aborted once and segfaulted once, both while
compiling the relaxation scan of `test_relaxation_loop` (2026-08-26). The
GitHub workflow does not set it, and four cores without it run the tier
as fast as 32 (the compile is single-threaded either way).

## Wait for a job

```
bash slurm/waitjob.sh <JOBID> <log>
```

blocks until the job leaves the queue and prints the head and tail of the log.
