# Running MRX

Condensed from `docs/source/cluster.md`, `relaxation.md`, `tutorials.md`,
`concepts/testing_strategy.md`, plus lessons from runs.

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

`run.sh` variables: `SCRIPT` (path or `-m module`), `ARGS`, `JOB_NAME`, `OUTSUB`, `TIMEOUT_MIN`
(default 60), `MEM_GB` (default 64), `CPUS` (default 32), `EXTRA_ENV` (`"MRX_DTYPE=float32"`),
`DEPENDENCY`, `MRX_ROOT`, `MRX_VENV`. Log: `$MRX_DATA/runs/mrx/$OUTSUB/<date>/<time>/$JOB_NAME.log`.
Account, partition and `MRX_DATA` from `~/.config/mrx/site.env` (or `$MRX_SITE_ENV`).

**Sizing.** Jobs use about one core: `CPUS=4 MEM_GB=16` up to ~(16,32,32), the default `MEM_GB` on larger
meshes. `TIMEOUT_MIN` at least 2x the estimate (per-step cost varies 2x between files), partition maximum for
multi-hour runs. A kill loses the run, a long limit costs nothing.

A floored run is not continued (`history.md`). Cancel it rather than let it run out its time.

## Worktrees

- A new worktree may start from the default branch rather than the one you meant (Claude Code's
  `EnterWorktree` does): check `git log -1` and reset to the intended base.
- A new worktree needs no setup: site settings are in `~/.config/mrx/site.env`, and input files not in
  `data/` are found in `$MRX_DATA/data` by name (`mrx.paths.data_file`).
- `run.sh` sets `PYTHONPATH=$MRX_ROOT` (the worktree containing it). Any other launch needs
  `PYTHONPATH=$WT` by hand: `python scripts/x.py` otherwise imports the **main checkout's** mrx through
  the editable install, and the run looks like a test of the branch while testing main. `python -m pytest`
  from the worktree directory is safe. Read `mrx from: <path>` in the log first.
- Write every plot or archive under `$MRX_DATA/runs/mrx/<line>/`, shared by all worktrees.

## Tests (`concepts/testing_strategy.md`)

Mathematical statements plus the tracer, the shape derivative and the readers: `test_derham.py` (complex
exactness, SPD masses, commuting interpolation, cohomology), `test_solvers.py` (every solve inverts its
operator to a multiple of `seq.tol`), `test_poisson.py` (manufactured solutions k = 0..3 in measured bands),
`test_newton.py` (second variation, short run), `test_double_single.py` (the double-single kernels, run
first on a new device),
`test_poincare.py`, `test_shape_ad.py`, `test_readers.py` (GVEC/DESC/VMEC, right-handed angles). Fixtures
in `test/conftest.py`: li383 at (16,32,12) per period, p = 2, graded, in two representations (`half`,
`period`). The suite is compile-bound. XLA cache at `$MRX_DATA/cache/xla` (`MRX_XLA_CACHE` overrides,
without `MRX_DATA` `outputs/xla_cache`).

`bash slurm/suite.sh [float64|float32]` (both by default): a change to solvers, precision or atoms must
pass in both. Bare `pytest` is float64, `MRX_DTYPE=float32 pytest` is float32. CI runs both on a CPU.

## relax.py (`docs/source/relaxation.md`)

tyro CLI from `mrx/relaxation/config.py`, and `python scripts/relax.py --help` lists all flags.

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
quoting them, they move.

**Output:** `relax.json` (`params` flat, `ic`, `seed`, `drive`, per-step `trace`: `dE`, `dE_ls`, `F`,
`resid`, `dt`, `dt_star`, `cfl`, `div`, `cos`, `newton_it`, `res_it`, ..., per-chunk `qoi`: `E`, `F`,
`resid`, `helicity`, `JoverB`, `JB`, `beta_vol`, and `summary` with `best_step`),
`checkpoints/state_<step>.h5`, `checkpoints/best.h5`. Restart with `--output.restart`. Rebuild the
sequence from `checkpoint_attrs(path)` and read the field with `read_field(path, seq)`.

**A healthy ideal run:** `dE < 0` every step, `dE == dE_ls` to round-off, `div` at round-off, helicity
constant up to grid-scale reconnection, positive solver counts. `resid = ||F||^2_M / ||grad(B^2/2)||^2`
(squared, the paper convention). Stop at the floor, take `best.h5` if the run went past it.

## Resolution and degree

- No fixed ratio such as `(n, 2n, 2n)` or cubes. Use the graded meshes of the research record (li383
  (32,44,20), (48,64,24), (64,80,32), W7-X (32,64,36)). Unit tests (16,32,12), tutorials (12,16,16).
- A seeded island needs at least ~3 radial cells across its width (`docs/research/relaxation.md` 5).
- Check the file's own initial force residual first, before blaming the relaxation.
- No p = 5 relaxation, sweeps stop at p = 4.
- W7-X: the FMM002 wout (not distributed with the repository) at p = 2, higher resolution rather than
  higher degree.

## Tutorials (`docs/source/tutorials.md`)

`scripts/tutorials/1_qa_geometry.py` .. `6_qa_shape_optimization.py`. All at (12,16,16), p = 2, float64
(Tutorial 6: p = 3). About 30 min total on one H100, 42 min on 8 CPU cores. They are API demos for a
laptop (< 1 h total): a few Newton steps, never a finer mesh to make a method look better. Output in
`$MRX_DATA/runs/mrx/tutorials/<script>/`. Tutorials 4 and 5 reuse the previous tutorial's run.

## Data files

Tracked: `wout_li383_low_res_reference.nc`, `wout_li383_1.4m.nc`, `wout_LandremanPaul2021_QA_lowres.nc`,
`desc_li383_low_res_reference.h5`, `desc_LandremanPaul2021_QA.h5`, `gvec_gframe/` (a G-frame GVEC state for
the reader tests). Larger inputs that appear in `docs/research/` (W7-X FMM002 wout, GVEC states,
high-resolution QA) are not distributed with the repository. Put any input you have in `$MRX_DATA/data`.

## Poincare sections

```bash
python -u scripts/poincare_trace.py --geometry <file> <run>/checkpoints/state_000000.h5 <run>/checkpoints/best.h5   # GPU job
python scripts/poincare_plot.py <run>          # matplotlib, login node OK -> <run>/poincare/
```

- In code: `poincare(seq, B, lines=160, periods=400, planes=5)` (`mrx.diagnostics.poincare`),
  `poincare_many` for several fields in one batch (eight cost about as much as one). Steps per period >= 24.
- `trace.npz` lives next to the checkpoints it came from. Trace all states of one run in ONE call so they
  share one iota and one pressure scale. Never retrace for a cosmetic change: rerun only the plotter.
- Five planes over the distinct part of the period (half a field period with stellarator symmetry) by
  default, `--planes` to choose others. Sections are the map's `zeta = const` planes by default,
  `--sections phi` cuts at the physical toroidal angle.
- `locked_width(res, iota)` reads a quick island width, resolved only to the spacing of the lines. Trace
  more `--lines` before reading small widths. The library has no residue diagnostic.

## Profiling (`docs/research/performance.md` 3)

- Capture inside the job with `jax.profiler.trace(dir)` after a warm-up, view it with XProf.
- One dispatch of a filter_jit function of the sequence costs about 10 ms of host time. Measure device time
  per apply with a compiled `fori_loop` of 10 against 30 applies, not with one call.
