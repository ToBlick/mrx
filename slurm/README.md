# Running MRX on the cluster

Every MRX run, including smoke tests, is a GPU job. `slurm/run.sh` submits one
script as a single-GPU job and prints the log path.

## slurm/run.sh

```
SCRIPT=scripts/relax.py ARGS="--geometry toroid --ic analytic --steps 50" JOB_NAME=smoke bash slurm/run.sh
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
export them in the shell.

`MRX_ROOT` is the checkout to run; it defaults to the repository containing
`run.sh`, so a worktree runs itself. `MRX_VENV` is the virtualenv; it defaults
to `$MRX_ROOT/.venv`, then to the main checkout's `.venv`.

## Check the log

The first line of every log is `mrx from: <path>`. `mrx` is an editable
install pinned to the main checkout, so a job without `PYTHONPATH` imports the
main checkout's library while running a worktree's script, and reports the
result as a test of the worktree. Read that line before reading the result.

## Run the suite on a GPU node

`pytest` (no arguments) runs the whole suite: one lean tier, 51 tests on two
session sequences at `(8, 12, 12)` p=2 (li383 and the analytic toroid,
`test/conftest.py`), in float64 and float32 on the GitHub runners
(`.github/workflows/ci.yml`). Everything it reads is tracked, so a worktree
runs it as it is. On the cluster it is one GPU job:

```
SCRIPT="-m pytest -q test" JOB_NAME=tests TIMEOUT_MIN=30 bash slurm/run.sh
```

To measure the suite as a runner sees it, run it inside a GPU job with the
GPU hidden and four cores:

```
SCRIPT="-m pytest -q test" JOB_NAME=tests_cpu CPUS=4 EXTRA_ENV="JAX_PLATFORMS=cpu" bash slurm/run.sh
```

`EXTRA_ENV="MRX_DTYPE=float32"` selects single precision for any of these.
Measured 2026-09-02 (H100): 51 pass in 3:54 (float64) and 4:35 (float32)
on the GPU, 3:55 on four CPU cores in float64. The suite is compile-bound:
the li383 fixture (`build_sequence` plus the harmonic forms) is ~75 s of it
on either backend, the toroid fixture ~25 s, the 50-step relaxation ~70 s,
the eight manufactured solves 1-6 s each.

The previous 263-test suite ran ~13 min on the GPU and crashed XLA's CPU
backend deterministically after ~230 tests: tens of thousands of separate
compilations in one process (every eager solve traces its own loop body),
neither memory nor stack; `jax.clear_caches()` between modules cured it.
The lean suite does not get near that count. Do not add
`XLA_FLAGS=--xla_cpu_multi_thread_eigen=false` to the CPU recipe either:
the compile is single-threaded, and four cores run the suite as fast as 32.

## Wait for a job

```
bash slurm/waitjob.sh <JOBID> <log>
```

blocks until the job leaves the queue and prints the head and tail of the log.
