# Solve a relaxation problem

`scripts/relax.py` relaxes a magnetic field toward minimum energy at fixed
topology (up to grid-level effects). The fixed point is $J \times B = \nabla p$, 
a finite-beta equilibrium. This guide covers the input, the run and its output. The
algorithm is in [Relaxation](concepts/relaxation.md).

## Geometry and initial condition

The geometry file fixes the map and the initial field. A VMEC wout
(`.nc`), a GVEC state (`.dat`) or a DESC output (`.h5`) gives the polar spline map of the file's
series and the file's own field $B = dA'$ as the initial field
([Equilibrium input](concepts/equilibrium_input.md)). An analytic shape is
used by writing a mock equilibrium file from its formulas, for example a
VMEC wout or a GVEC state, as `test/synthetic_gvec.py` and
`test/synthetic_desc.py` do.

```python
from mrx.geometry import build_sequence
from mrx.relaxation.initial_conditions import initial_field
from mrx.nullspace import compute_nullspaces

seq, ops = build_sequence("data/wout_li383_1.4m.nc", (16, 32, 32), 2)
compute_nullspaces(seq)
B0, info = initial_field(seq)
```

## Run

Every run is a GPU job through `slurm/run.sh` ([Running on a cluster](cluster.md)):

```bash
SCRIPT=scripts/relax.py JOB_NAME=relax_li383 TIMEOUT_MIN=120 \
  ARGS="--geometry.path data/wout_li383_1.4m.nc --geometry.resolution 16 32 32" bash slurm/run.sh

SCRIPT=scripts/relax.py JOB_NAME=relax_smoke TIMEOUT_MIN=30 \
  ARGS="--geometry.path data/wout_li383_low_res_reference.nc --geometry.resolution 8 12 12 --budget.steps 50" bash slurm/run.sh
```

`python scripts/relax.py --help` lists every flag with its default. The
command line is built by [tyro](https://brentyi.github.io/tyro/) from the
dataclasses in {mod}`mrx.relaxation.config`, one group each. A field is the
flag `--<group>.<field>` with hyphens for underscores, a tuple takes its
values separated by spaces (`--geometry.resolution 16 32 32`) and a bool is
the pair `--seed` / `--no-seed`:

| group | flags |
|---|---|
| geometry | `--geometry.path`, `--geometry.symmetry`, `--geometry.resolution`, `--geometry.spline-degree`, `--geometry.knots-r/theta/zeta`, `--geometry.precision`, `--geometry.solve-tol`, `--geometry.solve-maxiter`, `--geometry.max-batch` |
| seed | `--seed`, `--seed.iotas`, `--seed.amplitudes`, `--seed.scale` |
| descent | `--descent.method {newton,gradient}` |
| newton | `--newton.penalty`, `--newton.tol`, `--newton.maxiter` |
| pressure | `--pressure.advected`, `--pressure.beta`, `--pressure.smoothing` |
| budget | `--budget.steps`, `--budget.chunk`, `--budget.floor-tol` |
| drive | `--drive.resistivity`, `--drive.reference`, `--drive.reference-smoothing`, `--drive.chain`, `--drive.eps` |
| output | `--output.out`, `--output.restart` |

The `params` of `relax.json` hold the same settings as one flat dict with
the keys of earlier runs (`geometry`, `resolution`, `spline_degree`,
`seed_iotas`, `method`, `newton_tol`, `steps`, `drive_resistivity`, `out`
and so on). `RelaxConfig.from_params` rebuilds the configuration from it.

The tutorials build the same configuration in Python
(`RelaxConfig(geometry=Geometry(...), ...)`).

## Stopping

The squared normalised force residual
$\mathrm{resid} = \|F\|_M^2 / \|\nabla(B^2/2)\|^2$ is recorded at every
step. The run stops when its mean over the last chunk falls below
`--budget.floor-tol`, or when `--budget.steps` is spent. The residual is not monotone.
Judge a run by the floor it settles at, and take `checkpoints/best.h5` when
it went past it. A job's time limit is no stop: the checkpoint of every
chunk restarts it (`--output.restart`).

## Output

| file | content |
|---|---|
| `relax.json` | `params`: the configuration, flat, plus `geometry_path`, `ic`, `resolution`, `nfp`, `h_r_sq`, `start_step`. `ic`: the initial field's numbers. `seed`, `drive`: when used. `trace`, per step: `dE`, `dE_ls`, `F` (the force residual), `resid`, `dt`, `dt_star`, `cfl`, `div`, `cos`, `newton_it`, `res_it`, `res_moved`. `qoi`, per chunk: `it`, `wall`, `E`, `F`, `resid`, `helicity`, `JoverB`, `JB`, `beta_vol`. `summary`: the stopping reason and `best_step`. |
| `checkpoints/state_<step>.h5` | the descent state at step 0 and after every chunk (`mrx.relaxation.loop.write_checkpoint`): every leaf of `State` as a dataset (`B_n`, `warm.p`, ...), the discretisation as attributes |
| `checkpoints/best.h5` | the field of lowest residual |

Both are rewritten at every chunk, so a run cut off by its time limit
leaves its trace and its last state. A healthy ideal run has `dE < 0` at
every step, `dE` equal to `dE_ls` to round-off, `helicity` constant up to
grid-scale reconnection and `div` at round-off.

To evaluate a checkpoint's field:

```python
import h5py
from mrx.differential_forms import DiscreteFunction, Pushforward
from mrx.relaxation.loop import checkpoint_attrs

path = "outputs/relax/<date>/<time>/checkpoints/best.h5"
a = checkpoint_attrs(path)
seq, ops = build_sequence("data/wout_li383_1.4m.nc", a["ns"], a["p"], knots=a["knots"], symmetry=a["symmetry"])
with h5py.File(path) as fh:
    B = fh["B_n"][...]
odd = seq.odd                     # B is odd under the stellarator symmetry
B_phys = Pushforward(DiscreteFunction(B, odd.basis_2, odd.E(2)), seq.map, 2)
```

## Pressure

The run carries two pressures: the strong `p` of the descent (state,
`warm.p` in every checkpoint) and the weak `p_w`, zero on the wall, from
which `beta_vol` is computed ([Relaxation](concepts/relaxation.md)).

With `--pressure.advected` the pressure is prescribed instead: the geometry
file's pressure profile as a function of `r`, scaled to the volume beta
`--pressure.beta`, moves with a compressible flow and the run lowers
`int |B|^2/2 - p dV`. Every checkpoint stores it as `p_n`, `qoi` adds
`p_int` (`int p dV`) and `beta_p`, and `E` is the energy with the pressure
term. The strong and weak pressures are still sampled, and at the fixed
point `p_w` should match `p_n`.

## Poincare sections

Two scripts split the work by cost:

```bash
python -u scripts/poincare_trace.py --geometry data/wout_li383_1.4m.nc \
    outputs/run/checkpoints/state_000000.h5 outputs/run/checkpoints/best.h5
python scripts/poincare_plot.py outputs/run            # -> outputs/run/poincare/
```

`poincare_trace.py` (a GPU job) traces every checkpoint it is given with
`mrx.diagnostics.poincare.poincare`, evaluates the weak pressure at every crossing and
writes one `trace.npz`. `poincare_plot.py` (plain matplotlib) renders it,
every field and plane on one iota and one pressure scale. It is the only
thing to rerun when the figure changes. A movie is the same two calls on
every checkpoint of a run, with `--window`, `--iota-lim` and `--p-lim`
holding the axes fixed (their values separated by spaces, for example
`--iota-lim 0.3 0.6`).

In code, `poincare(seq, B, lines=160, periods=400, planes=5)` seeds the
lines from the axis to the edge, measures iota per line, flags the chaotic
ones and cuts the trajectories at the planes (five over half a period for a
stellarator-symmetric map). The returned `drift` (step h against h/2)
justifies the step count. `mrx.diagnostics.islands.islands(seq, B)` finds
the island chains of a field by Newton on the Poincare map (`poincare_map`) and measures their
widths on a ray through each O-point.
