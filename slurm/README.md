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
export them in the shell. `conf/config_poisson_test.yaml` reads the same
variables through `${oc.env:...}` for its submitit launcher.

`MRX_ROOT` is the checkout to run; it defaults to the repository containing
`run.sh`, so a worktree runs itself. `MRX_VENV` is the virtualenv; it defaults
to `$MRX_ROOT/.venv`, then to the main checkout's `.venv`.

## Check the log

The first line of every log is `mrx from: <path>`. `mrx` is an editable
install pinned to the main checkout, so a job without `PYTHONPATH` imports the
main checkout's library while running a worktree's script, and reports the
result as a test of the worktree. Read that line before reading the result.

Hydra's submitit launcher does not export `PYTHONPATH` on its own.
`conf/config_poisson_test.yaml` adds `export PYTHONPATH=${oc.env:MRX_ROOT}` to
the launcher `setup` list, so `MRX_ROOT` must be set in the submitting shell
for a multirun of `scripts/poisson_study.py`; a single run through `run.sh`
needs nothing else.

## Run the suite on a GPU node

`pytest` (no arguments) runs the whole suite: one tier, CPU-sized, in
float64 and float32 on the GitHub runners (`.github/workflows/ci.yml`).
On the cluster it is one GPU job:

```
SCRIPT="-m pytest -q test" JOB_NAME=tests TIMEOUT_MIN=45 bash slurm/run.sh
```

Tests that read files outside the repository (the W7-X Clebsch initial
condition from `MRX_W7X_FILE`, the archived relaxation traces from
`MRX_RELAX_ARCHIVE`) carry the `needs_data` marker and skip when the
variable is unset or the file is absent. To run them, name the files:

```
SCRIPT="-m pytest -q test -m needs_data" JOB_NAME=tests_data TIMEOUT_MIN=30 \
    EXTRA_ENV="MRX_W7X_FILE=/path/to/w7x_fmm002_clebsch_mrx.h5" bash slurm/run.sh
```

To measure the suite as a runner sees it, run it inside a GPU job with the
GPU hidden and four cores:

```
SCRIPT="-m pytest -q test" JOB_NAME=tests_cpu CPUS=4 EXTRA_ENV="JAX_PLATFORMS=cpu" bash slurm/run.sh
```

`EXTRA_ENV="MRX_DTYPE=float32"` selects single precision for any of these.
Measured 2026-08-26 (H100, tree at the single-tier commit): 248 items,
247 pass and the W7-X test skips without `MRX_W7X_FILE`; 6:09 on four
CPU cores in float64, 9:15 on the GPU in float64 and 9:01 in float32 (the
suite is compile-bound; the session torus fixture is 51 s of it, the
relaxation run 28 s, the projector tests 90 s). The `needs_data` tests
pass in 2:50 on the GPU (the W7-X fixture is 144 s of it).

Do not add `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false` to the CPU
recipe: XLA's CPU compiler aborted once and segfaulted once, both while
compiling the relaxation step (2026-08-26). The GitHub workflow does not
set it, and four cores without it run the suite as fast as 32 (the compile
is single-threaded either way).

## Wait for a job

```
bash slurm/waitjob.sh <JOBID> <log>
```

blocks until the job leaves the queue and prints the head and tail of the log.
