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

seq = build_sequence("data/wout_li383_1.4m.nc", (16, 32, 32), 2)
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
| geometry | `--geometry.path`, `--geometry.symmetry`, `--geometry.resolution`, `--geometry.spline-degree`, `--geometry.knots-r/theta/zeta`, `--geometry.graded`, `--geometry.solve-tol`, `--geometry.solve-maxiter`, `--geometry.max-batch` |
| seed | `--seed`, `--seed.iotas`, `--seed.amplitudes`, `--seed.scale` |
| descent | `--descent.method {newton,gradient}` |
| newton | `--newton.penalty`, `--newton.tol`, `--newton.maxiter`, `--newton.mass-solve-tol`, `--newton.mu-constant`, `--newton.elastic-model` |
| budget | `--budget.steps`, `--budget.chunk`, `--budget.floor-tol`, `--budget.floor-tol-el` |
| drive | `--drive.resistivity`, `--drive.reference`, `--drive.reference-smoothing`, `--drive.chain`, `--drive.eps` |
| elastic | `--elastic.lam`, `--elastic.reference`, `--elastic.lam-perp` |
| output | `--output.out`, `--output.restart` |

The `params` of `relax.json` hold the same settings as one flat dict with
the keys of earlier runs (`geometry`, `resolution`, `spline_degree`,
`seed_iotas`, `method`, `newton_tol`, `steps`, `drive_resistivity`, `out`
and so on).

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
| `checkpoints/state_<step>.h5` | the descent state at step 0 and after every chunk (`mrx.relaxation.loop.write_checkpoint`): every leaf of `State` as a dataset (`B_n`, `warm.J`, ...), the discretisation as attributes |
| `checkpoints/best.h5` | the field of lowest residual |

Both are rewritten at every chunk, so a run cut off by its time limit
leaves its trace and its last state. A healthy ideal run has `dE < 0` at
every step, `dE` equal to `dE_ls` to round-off, `helicity` constant up to
grid-scale reconnection and `div` at round-off.

To evaluate a checkpoint's field:

```python
from mrx.differential_forms import DiscreteFunction, Pushforward
from mrx.relaxation.loop import checkpoint_attrs, read_field

path = "outputs/relax/<date>/<time>/checkpoints/best.h5"
a = checkpoint_attrs(path)
seq = build_sequence("data/wout_li383_1.4m.nc", a["ns"], a["p"], knots=a["knots"], symmetry=a["symmetry"],
                     graded=a["graded"], map_theta=a["map_theta"])
B, step = read_field(path, seq)
odd = seq.odd                     # B is odd under the stellarator symmetry
B_phys = Pushforward(DiscreteFunction(B, odd.basis_2, odd.E(2)), seq.map, 2)
```

## Elastic regularisation

`--elastic.lam lam` adds the elastic energy of Burby, Cho & Ruth
(arXiv:2610.36241) and Ruth et al. (arXiv:2604.08878), the energy of
small-scale field fluctuations frozen into the flow. The elastic fields move
with the velocity that moves `B`, add `||S_k||^2 / 2` to the energy and their
force to the force ({mod}`mrx.relaxation.elastic`). The `E` of the output is
then the total energy and `E_el` its elastic part.

`--elastic.reference` picks the fields. Use `perpendicular`, here with
`lam` from 0.001 to 0.003. It carries the frame `e_R`, `e_phi`, `e_Z` of the
cylindrical coordinates projected across the start field,
`lam |B| (e_k - (e_k . b) b)` with `b = B / |B|`. Their variance is
`lam^2 (|B|^2 I - B B^T)` and needs no flux surfaces. With `B` it gives a matrix
of full rank in the energy, so the second variation controls the full gradient
of the flow. Its stress at zero strain is `-lam^2 B B^T`, so it puts no traction
on a boundary with `B . n = 0`. The fields are not tapered at the wall. They
cross it, so they live in the 2-forms without the wall condition. At zero strain
their force in `resid` is a discretisation error that falls with the mesh, 0.16,
0.11, 0.08 at `lam = 1` on li383 graded (32,44,20), (48,64,24), (64,80,32).

At `lam = 0.001` on those three meshes `resid` is 5.3e-9, 5.2e-9, 5.1e-9 and
`||J|| / ||B||` 0.6386, 0.6385, 0.6386 (2026-11). `resid` grows like
`3e-3 lam^2` above `lam = 0.001`. It flattens near 2e-9 below `lam = 0.0005`,
where `||J|| / ||B||` returns towards its value without elastic fields.

The other two references remain available. `labels`
carries the label 2-forms `d psi ^ d theta` and `d psi ^ d zeta` of the start
flux coordinates (`psi = r^2 / 2`), each scaled to `lam ||B||`. They follow the
flux surfaces of the start and push on `B` already before the flow has strained
them. On the seeded li383 start their force at `lam = 1` has `resid` 2.0.
`isotropic` carries the frame itself, each field scaled to `lam ||B||`, with the
variance `a^2 I`.

`--elastic.lam-perp lam_perp` adds to the label fields the normal field
`f d theta ^ d zeta` with `f = s (1 - s)` and `s = r^2`, scaled to
`lam_perp ||B||`. Its field lines cross the flux surfaces, so it couples
neighbouring surfaces. The weight vanishes at the axis, where the field is then
smooth, and on the wall, where `S . n = 0`. This field is not closed. It moves
as a flux density, `dS/dt = curl(u x S) - u div S`, so its divergence moves with
the flow as a density. The Newton preconditioner leaves it out. Keep `lam_perp`
well below `lam`. On li383 a tenth of it moves `B` by less than 1e-3, while at
`lam_perp = lam` the current gathers near the axis and grows with the resolution.

With elastic fields the penalty on flows along the field is one number,
`mu (u . B)^2` with `mu = c / L^2`. Here `L` is the length of the axis of one
field period over `2 pi`, and `c` is `--newton.mu-constant` (0.04). That makes
`mu` 0.16 on li383, 0.03 on W7-X high mirror and 0.31 on HSX. Without elastic
fields `--newton.penalty` applies, and nothing above changes that run.

`--newton.elastic-model` picks the terms of the elastic fields in the Newton
operator. The default `lumped` leaves out the drag of the density and replaces
their 1-form mass solves by one fixed application of the mass preconditioner,
which costs about a third per MINRES iteration. `exact` keeps both. On li383
(64,80,32) at `lam = 0.003` and `mu = 0.068` (2026-10-30, before the rule for
`mu`) `lumped` stopped at `resid_el` 2.1e-13 in 300 steps, where `exact` reached
1e-14. Neither the penalty nor this choice is in the equations of the
equilibrium. Both move the end state through the path only. On li383 at
`lam = 0.003` that is about 1e-6 in `B` for `mu` 0.16 against 0.32 and for
`lumped` against `exact`, about 300 times less than the distance to the end
state at `lam = 0`.

The residual `resid` that
stops the run and picks `best.h5` is that of `B` alone, `J x B - grad p`
without the elastic force, at one more solve per step. The residual of the
full force, which the descent drives to zero, is recorded as `resid_el`.
The run also stops when the chunk mean of `resid_el` falls below
`--budget.floor-tol-el` (1e-15). That is the elastic equilibrium, and the
`resid` left then is the elastic force that holds it.

## Pressure

MRX has two pressures: the strong `p` of the descent (`compute_force`, on
request) and the weak `p_w`, zero on the wall, from which `beta_vol` is
computed at every chunk ([Relaxation](concepts/relaxation.md)).

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
justifies the step count.
