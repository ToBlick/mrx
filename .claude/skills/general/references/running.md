# Running MRX

How to run is in `docs/source/cluster.md` (site settings, `slurm/run.sh` and its variables, worktrees),
`docs/source/relaxation.md` (relax.py, stopping, output, Poincare sections), `docs/source/tutorials.md` and
`docs/source/concepts/testing_strategy.md`. This page adds the working habits.

## Where things run

- **Login node of a cluster:** only matplotlib/numpy (re-render a `trace.npz` with `scripts/poincare_plot.py`,
  figure scripts reading `relax.json`) and pure-numpy checks that import no `mrx` solver.
- **Everything else:** a GPU job through `slurm/run.sh`.

```bash
SCRIPT=scripts/relax.py JOB_NAME=relax_smoke TIMEOUT_MIN=30 CPUS=4 MEM_GB=16 \
  ARGS="--geometry.path data/wout_li383_low_res_reference.nc --geometry.resolution 8 12 12 --budget.steps 50" \
  bash slurm/run.sh
SCRIPT="-m pytest -q test/test_newton.py" JOB_NAME=tests TIMEOUT_MIN=60 CPUS=4 MEM_GB=16 bash slurm/run.sh
bash slurm/waitjob.sh <JOBID> <log>        # blocks, prints head and tail
```

**Sizing.** Jobs use about one core: `CPUS=4 MEM_GB=16` up to ~(16,32,32), the default `MEM_GB` on larger
meshes. `TIMEOUT_MIN` at least 2x the estimate (per-step cost varies 2x between files), partition maximum for
multi-hour runs. A kill loses the run, a long limit costs nothing.

A run at its floor is not continued: further steps only reconnect at the grid scale (`docs/source/sharp_bits.md`,
"The force residual need not decrease"). Cancel it rather than let it run out its time.

## Worktrees

- A new worktree may start from the default branch rather than the one you meant (Claude Code's
  `EnterWorktree` does): check `git log -1` and reset to the intended base.
- A new worktree needs no setup: site settings are in `~/.config/mrx/site.env`, and input files not in
  `data/` are found in `$MRX_DATA/data` by name (`mrx.paths.data_file`).
- Outside `run.sh` set `PYTHONPATH=$WT` by hand: `python scripts/x.py` otherwise imports the **main checkout's**
  mrx, and the run looks like a test of the branch while testing main. `python -m pytest` from the worktree
  directory is safe. Read `mrx from: <path>` in the log first.
- Write every plot or archive under `$MRX_DATA/runs/mrx/<line>/`, shared by all worktrees.

## Tests

`bash slurm/suite.sh [float64|float32]` (both by default): a change to solvers, precision or atoms must
pass in both. The suite is compile-bound, its XLA cache is `$MRX_DATA/cache/xla` (`MRX_XLA_CACHE` overrides).

## relax.py

`python scripts/relax.py --help` lists all flags. Defaults, from `mrx/relaxation/config.py`:

| group | flags (defaults) |
|---|---|
| geometry | `--geometry.path`, `.symmetry` (stellarator), `.resolution` (32 64 64), `.spline-degree` (2), `.knots-r/theta/zeta`, `.graded` (off), `.map-theta`, `.precision` (float64), `.solve-tol` (1e-8), `.solve-maxiter` (2000), `.max-batch` |
| seed | `--seed/--no-seed`, `--seed.iotas`, `--seed.amplitudes`, `--seed.scale` |
| descent | `--descent.method {newton,gradient}` (newton) |
| newton | `--newton.penalty` (3), `--newton.tol` (0.1), `--newton.maxiter` (300), `--newton.mass-solve-tol` (1e-6) |
| budget | `--budget.steps` (100 Newton, 2000 gradient), `--budget.chunk` (10 / 200), `--budget.floor-tol` (1e-10) |
| drive | `--drive.resistivity`, `--drive.reference`, `--drive.reference-smoothing` (0.1), `--drive.chain`, `--drive.eps` |
| output | `--output.out`, `--output.restart` |

Tuples take spaces (`--geometry.resolution 16 32 32`). Check the defaults in `config.py` before
quoting them. What `relax.json` and the checkpoints hold, and what a healthy ideal run looks like:
`docs/source/relaxation.md` "Output".

## Resolution and degree

- No fixed ratio such as `(n, 2n, 2n)` or cubes. Use the graded meshes of the research record (li383
  (32,44,20), (48,64,24), (64,80,32), W7-X (32,64,36)). Unit tests (16,32,12), tutorials (12,16,16).
- W7-X: the FMM002 wout (not distributed with the repository) at p = 2.

## Tutorials

They are API demos for a laptop (< 1 h total): a few Newton steps, never a finer mesh to make a method look
better. Run them in order, Tutorials 4 and 5 start from the previous run (`docs/source/tutorials.md`).

## Data files

Tracked: `wout_li383_low_res_reference.nc`, `wout_li383_1.4m.nc`, `wout_LandremanPaul2021_QA_lowres.nc`,
`desc_li383_low_res_reference.h5`, `desc_LandremanPaul2021_QA.h5`, `gvec_gframe/` (a G-frame GVEC state for
the reader tests). Larger inputs that appear in `docs/research/` (W7-X FMM002 wout, GVEC states,
high-resolution QA) are not distributed with the repository. Put any input you have in `$MRX_DATA/data`.

## Poincare sections

Commands and options: `docs/source/relaxation.md` "Poincare sections". Habits:

- Trace all states of one run in ONE call, so they share one iota and one pressure scale. `trace.npz` lives next
  to the checkpoints it came from.
- Never retrace for a cosmetic change: rerun only the plotter.
- Trace more `--lines` before reading small widths. The library has no residue diagnostic.

## Profiling (`docs/research/performance.md` 3)

- Capture inside the job with `jax.profiler.trace(dir)` after a warm-up, view it with XProf.
- Measure device time per apply with a compiled `fori_loop` of 10 against 30 applies, not with one call: one
  dispatch costs about 10 ms of host time.
