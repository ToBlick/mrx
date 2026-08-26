# Running on a cluster

Every MRX run, including the test suite and smoke tests, is a GPU job.
Two launchers exist: `slurm/run.sh` for one script, and Hydra's submitit
launcher for the config-driven studies.

## Site settings

`slurm/run.sh` reads the account and partition from the environment,
never from the repository:

```bash
export SLURM_ACCOUNT=<account>
export SLURM_PARTITION=<gpu partition>
export SLURM_EXCLUDE=<node,node>      # optional
```

Put them in `slurm/site.env`, which is gitignored and sourced by
`run.sh`, or export them in the shell. The Hydra configs read the same
variables through `${oc.env:...}`.

## One script: `slurm/run.sh`

```bash
SCRIPT=scripts/relax.py ARGS="--geometry toroid --steps 50" JOB_NAME=smoke bash slurm/run.sh
SCRIPT="-m pytest -q test/test_solvers.py" JOB_NAME=tests TIMEOUT_MIN=120 bash slurm/run.sh
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

## Hydra: single run and multirun

The studies under `scripts/config_scripts/` are Hydra entry points. Their
configs are in `conf/`, their schemas are dataclasses in `mrx.config`.
Override any key as `key=value`.

A single run executes in the current process and writes to
`outputs/<date>/<time>/`. Submit it as one GPU job through `run.sh`:

```bash
SCRIPT=scripts/config_scripts/test_torus_poisson_k0_sparse.py ARGS="p=3 n=16" \
  JOB_NAME=pois_k0 MEM_GB=80 TIMEOUT_MIN=120 bash slurm/run.sh
```

A multirun (`-m`) submits one job per combination through the submitit
launcher configured in the yaml file, and writes to
`multirun/<date>/<time>/<job>/`:

```bash
export MRX_ROOT=$PWD
python scripts/config_scripts/test_torus_poisson_k0_sparse.py -m p=2,3 n=8,16
```

The launcher does not export `PYTHONPATH` on its own. The configs add
`export PYTHONPATH=${oc.env:MRX_ROOT}` to its `setup` list, so `MRX_ROOT`
must be set in the submitting shell. `conf/config_poisson_test.yaml`
allots one GPU, 80 GB, and 120 minutes per job.

## Config schemas

Every top-level config inherits `NumericsConfig`:

| key | meaning | default |
|---|---|---|
| `precision` | `float64` or `float32`; the entry point exports it as `MRX_DTYPE` before importing `mrx` | `float64` |
| `solver_tol` | relative residual tolerance of every iterative solve in the sequence; `None` is `sqrt(eps)` of the working precision | `None` (the Poisson yaml pins `1e-9`) |

`PoissonTestConfig` adds `n` (a list or an int), `p`, `epsilon`,
`quad_order` (`None` selects `p + 1 + quad_order_offset`),
`quad_order_offset`, `cg_maxiter`, and the map batch sizes. The module
docstring of each script lists every key with its default. A
`quad_order` below `p + 1` raises.

## Worktrees

A git worktree has no `data/` and no `.venv`. Link the data directory or
set `MRX_DATA`:

```bash
ln -s /path/to/mrx/data data
```

`run.sh` defaults `MRX_ROOT` to the repository containing it, so a
worktree runs itself, and it exports `PYTHONPATH=$MRX_ROOT` so that the
editable install of the main checkout does not shadow it. A job you
submit any other way needs `export PYTHONPATH=$WT` yourself. Check the
`mrx from:` line of the log before reading a result.
