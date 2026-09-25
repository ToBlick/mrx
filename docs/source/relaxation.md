# Solve a relaxation problem

`scripts/relax.py` relaxes a magnetic field toward minimum energy at fixed
helicity. The fixed point is $J \times B = \nabla p$: a finite-beta
equilibrium. This guide sets up the geometry and the initial condition,
runs the script, and reads its output. The algorithm is in
[Relaxation](concepts/relaxation.md).

## Geometry

Every geometry is a file, and the file also fixes the initial condition.
`mrx.geometry.build_sequence` turns it into a polar sequence with the map
installed and every solver operator built:

```python
from mrx.geometry import build_sequence

seq, ops = build_sequence("data/torus.json", ns=(8, 16, 8), p=3)
seq, ops = build_sequence("data/wout_li383_1.4m.nc", ns=(8, 16, 16), p=2)
```

| geometry file | map |
|---|---|
| a VMEC wout (`.nc`) or a GVEC state (`.dat`) | spline coefficients built from the file's series |
| an analytic geometry (`.json`) | a map of `mrx.mappings` with the parameters the file gives |

Anything else raises. A GVEC state or VMEC wout becomes two scalar
splines `R`, `Z` on the sequence's own spline space, built from the series
coefficients, so `ns` and `p` are also the map resolution. `build_gvec_map`
checks that $\det D\Phi > 0$ everywhere and raises otherwise. `nfp=`
overrides the file's value for a file that declares it wrong. What MRX
reads from the file is in [GVEC → MRX interface](concepts/gvec_mrx_interface.md).

An analytic geometry file names the map and its parameters, and the
profiles of the initial condition below (`mrx.geometry.read_analytic`):

```json
{"map": "torus",
 "map_params": {"epsilon": 0.3333, "kappa": 1.0, "R0": 1.0},
 "profile": {"iota": [0.4, 0.9], "iota_exp": 2.0, "flux_exp": 1.0, "lambda": []}}
```

`data/torus.json`, `data/cylinder.json` and `data/rot_ellipse.json` are the
three shipped ones (`toroid_map`, `cylinder_map`, `rotating_ellipse_map`);
copy one and edit the numbers. `mrx.geometry.geometry_kind(path)` returns
`vmec`, `gvec`, or the map's name.

## Initial condition

`mrx.initial_conditions` builds every initial field in the reference
2-form frame, where the components are $\sqrt{g} B^i$:

$$
\hat B^\rho = 0, \qquad
\hat B^\chi = \Phi'(\rho)\,(\iota(\rho) - \partial_\zeta \lambda), \qquad
\hat B^\zeta = \Phi'(\rho)\,(1 + \partial_\chi \lambda).
$$

This field is divergence-free and tangent to the boundary for any
$\lambda$ and any geometry. The geometry file decides where the profiles
come from:

| geometry | initial condition |
|---|---|
| VMEC wout, GVEC state | the equilibrium's own field, $B = dA'$ from the file's `dPhi_dr`, `dchi_dr` and `LA` through the histopolated potential (exactly divergence-free) |
| analytic `.json` | the `profile` block: $\iota = \iota_0 + (\iota_1 - \iota_0)\rho^e$, $\Phi' = \rho^q$, $\lambda = \sum a\, \rho^{|m|} \sin 2\pi(m\theta - n\zeta)$ on the logical grid, projected and Leray-cleaned |

In code:

```python
from mrx.initial_conditions import (analytic_profile_form, make_lambda, make_profiles,
                                    project_reference_two_form, leray_clean)

iota, dPhi = make_profiles(iota0=0.4, iota1=0.9, iota_exp=2.0, flux_exp=1.0)
omega_ref = analytic_profile_form(iota, dPhi, make_lambda([]))
B0, B_norm = project_reference_two_form(seq, omega_ref)   # DoFs of the Dirichlet k=2 space
B0, moved = leray_clean(seq, B0)                            # remove the projection's divergence
```

`project_reference_two_form` pushes the form forward and projects with
`load(frame='phys')`. Do not pass the primal components to
`load(frame='ref')`: that argument wants $g\omega/J$ and fails silently.
The script always Leray-projects the initial condition.

## Run

Every run is a GPU job through `slurm/run.sh`:

```bash
SCRIPT=scripts/relax.py JOB_NAME=relax_li383 TIMEOUT_MIN=60 \
  ARGS="--geometry data/wout_li383_1.4m.nc" bash slurm/run.sh

SCRIPT=scripts/relax.py JOB_NAME=relax_smoke TIMEOUT_MIN=30 \
  ARGS="--geometry data/torus.json --resolution 6,12,6 --steps 50" \
  bash slurm/run.sh
```

Flags, defaults in brackets:

| flag | meaning |
|---|---|
| `--geometry PATH` (required) | a VMEC wout (`.nc`), a GVEC state (`.dat`) or an analytic geometry (`.json`, which carries its `nfp`); the geometry and the initial condition |
| `--symmetry {stellarator,field-period,none} [stellarator]` | what the map satisfies (`mrx.geometry.SYMMETRIES`): `nfp` field periods and stellarator symmetry (the spline map is projected onto it, the quadrature covers half the period and the fields keep their parity), field periods only, or nothing (`zeta` in `[0, 1]` is the whole torus, `nfp = 1`) |
| `--resolution R,T,Z [32,64,64]`, `--spline-degree P [2]` | spline resolution (also the map's) and degree |
| `--knots-r LIST`, `--knots-theta LIST`, `--knots-zeta LIST` | the breakpoints of that axis, comma-separated from 0 to 1, instead of the uniform grid; the axis takes its `n` from them (`mrx.geometry.knot_vector`) |
| `--precision {mixed,float32,float64} [mixed]` | `mixed` is float32 fields and solves with a float64 residual (tolerance 1e-8), `float32` and `float64` are both (1e-5, 1e-10); exported as `MRX_DTYPE` and `MRX_RESIDUAL_DTYPE` before `mrx` is imported |
| `--solve-tol TOL [the precision's]`, `--solve-maxiter N [2000]` | residual tolerance and budget of every solve (`concepts/precision.md`) |
| `--max-batch N [0]` | cells per batch of the quadrature loops; 0 evaluates all points in one `vmap`; bound it at high resolution (8192 at (64,128,128)) |
| `--seed` / `--no-seed` [off] | equilibrium files only: the energy-criterion seed of `mrx.seeding` on the start field (the initial condition or the `--restart` checkpoint): every resonance `nfp n / m` in the field's iota range gets SIESTA's parallel seed at the amplitudes of least energy; no phase (the stellarator parity fixes it) |
| `--seed-iotas I,...`, `--seed-amplitudes A,...`, `--seed-scale Q [1]` | the chains to seed by their rotational transforms [all in range]; their amplitudes, the resonant normal field `|dB^r| / |B^zeta|` at `r_mn`, signed, instead of the criterion's (one per iota; amplitudes without iotas are ignored with a warning); a multiplier of the added perturbation |
| `--method {newton,gradient} [newton]` | the direction: Newton on the second variation (`mrx.hessian.newton_direction`, the rows below), or gradient descent on the smoothed force |
| `--scheme {explicit,midpoint} [explicit]` | the induction step: forward Euler, or midpoint-implicit at the predictor's velocity (Picard on the increment) |
| `--newton-penalty KAPPA [3]` | the parallel-flow penalty, `KAPPA` times the strain along the field: the one number of the Newton configuration |
| `--newton-tol TOL [0.1]`, `--newton-maxiter N [200]` | the forcing term that ends the Newton solve (its residual below `TOL` of the right-hand side) and the MINRES iterations at most |
| `--steps N [100 Newton, 2000 gradient]` | the step budget; a job's time limit is no stop, the checkpoint of every chunk restarts it (`--restart`) |
| `--chunk N [10 Newton, 200 gradient]` | steps per compiled chunk (one `lax.scan`, `mrx.relaxation.chunk_runner`): the per-step trace comes back, the qoi are sampled, a checkpoint and the outputs are written, and the floor test runs once per chunk; `--steps` is a multiple of it |
| `--floor-tol TOL [1e-10]` | stopping criterion: the last chunk's mean squared normalised force residual `‖F‖²_M / ‖grad(B²/2)‖²` below it |
| `--drive-resistivity C [0]`, `--drive-reference PATH`, `--drive-reference-smoothing c [0.1]` | the resistive steady state: a backward-Euler resistive step of dose $C h_r^2$ in every step, $E = \eta (J - J^*)$, $J^*$ the current of the reference checkpoint's field (required: a converged nested equilibrium) after one heat step of $c h_r^2$ |
| `--drive-chain I`, `--drive-eps A` | a resonant seed of the reference at the chain of rotational transform `I`, amplitude `A` as `--seed-amplitudes`: a drive that stays |
| `--out DIR [outputs/relax/<date>/<time>]` | output directory |
| `--restart PATH` | continue from a `checkpoints/state_<step>.h5` of the same geometry, mesh, degree and precision |

The CFL cap (0.5), the velocity smoothing (order 1, scale 0.075 $h_r^2$) and the Newton solve's passes (one) are constants of the library; the paper's variants of them (the auxiliary B field, the helicity correction, the potential velocity, other smoothings) are `scripts/paper_scripts/relax_paper.py`'s flags, the same run on an extended configuration.

`python scripts/relax.py --help` prints the same list. The script is the
command line of `mrx.relaxation.relax`, the chunked loop with the floor,
wall-budget and reconnection rules, which the tutorials and the tests call
directly; `mrx.initial_conditions.initial_field` builds the field and
`mrx.relaxation.write_checkpoint` / `read_checkpoint` the files below.

## Stopping criterion

The squared normalised force residual $\|F\|_M^2 / \|\nabla(B^2/2)\|^2$
is recorded at every step. The run stops when its mean over the last `W` steps,

$$
\frac{1}{W} \sum_{j=i-W+1}^{i} \mathrm{resid}[j] < \texttt{floor-tol},
$$

or when the step budget runs out. The relaxation guarantees
$dE/dt \le 0$ only, so the residual is not monotone; the window mean is
the quantity, never the last value. On the W7-X Clebsch run at `(8,16,8)`,
`p = 3`, float64, it reaches $2.9 \times 10^{-6}$ at step 500 and floors
around $10^{-6}$ by step 1000-3000. A float32 run's solves are
refined against a float64 residual ([Precision](concepts/precision.md)),
so its floor is no longer the solve tolerance; until 2026-09-04 it was
($\sim 4 \times 10^{-6}$ at tol $10^{-5}$), and a `--floor-tol` below it
never fired.

## Output

`--out` receives:

| file | content |
|---|---|
| `relax.json` | `params` (every flag, `geometry_path` resolved, `ic` the kind of initial condition); `ic`, the initial field's numbers; `trace` with per-step `dE` (the exact energy change of the step), `dE_ls` (the line search's prediction), `F`, `resid`, `dt`, `dt_star`, `cfl`, `div`, `cos`, `gain`; `qoi` with per-chunk `it`, `wall`, `F`, `resid`, `helicity`, `JoverB`, `JB` and the pressure diagnostics `gradp_cmp`, `p_cmp`, `weak_resid`, `dpdn_wall`, `JxBn_wall`, `beta_vol`, `beta_axis` (the first entry is the start of the run); `reconnect`, one record per reconnection; the `summary` with the stopping reason |
| `checkpoints/state_<step>.h5` | the descent state at that step, one file per chunk plus step 0 (the initial field): every leaf of `mrx.relaxation.State` as a dataset named by its key path (`B_n`, `warm.p` the strong pressure, the other warm starts, `last.F`, `best.B`, `dt`, ...; files written before 2026-09-20 have the flat names, of which the reader takes `B_n` and rebuilds the rest) and the step as an attribute. `--restart` continues from one; the plotters read the field and the strong pressure from them and compute the weak pressure on demand |
| `checkpoints/best.h5` | the field with the lowest per-step squared residual of the run, rebuilt into a state (one force evaluation) and tagged with its step: the run's answer when it went past its floor (the residual is not monotone, and past the resolved floor the ideal descent raises it). `summary.best_step`, `summary.best_resid` in `relax.json` say which step |

`relax.json` and the newest checkpoint are written at every chunk, so a
run that runs out of time still leaves its trace and its last state.

## Inspect

Read the trace with the standard library:

```python
import json
run = json.load(open("outputs/relax/<date>/<time>/relax.json"))
dE = run["trace"]["dE"]                   # the exact energy change of every step
resid = run["trace"]["resid"]             # ||F||² / ||grad(B²/2)||² after every step
H = run["qoi"]["helicity"]                # at the sampled steps
```

Three checks of a healthy run (no reconnection):

- `dE` is negative at every step and matches the line search's `dE_ls`
  to roundoff: their difference is `-dt <u, grad p>`, the velocity's
  gradient part, zero for a divergence-free velocity. `E_0 - E` is
  `-cumsum(dE)` (`E_0` is `summary["E0"]`); the energy itself is not in
  the trace, a step changes it by less than a float32 ulp of `E`.
- `helicity` is constant to the solver tolerance.
- `div` stays at roundoff.

`resid` is the squared force residual relative to the magnetic pressure gradient.
Judge a refinement by the floor it reaches, not by the rate.

## Two pressures

A run carries two pressures.

| | strong `p` | weak `p_w` |
|---|---|---|
| where | `compute_force`, the Leray multiplier of the descent | `weak_pressure`, from the same `J` and field |
| space | 3-form, Dirichlet complex | 0-form, zero on the wall |
| boundary | $\partial p / \partial n = 0$ by construction: the Lorentz force is projected onto the Dirichlet 2-form space first, which discards its normal component | $p_w = 0$; the force is projected onto the natural 1-form space, which keeps its normal component, and $\partial p_w / \partial n$ is the wall force once the remainder $F_w$ vanishes |
| gauge | a constant | none |
| read it for | the force residual of the constrained principle | the pressure profile, the wall force, beta |

The decomposition is $v = F_w + \nabla p_w$ with $(\nabla \phi, \nabla p_w) =
(\nabla \phi, v)$ for every $\phi$ vanishing on the wall
(`seq.apply_leray_projection(v, k=1, dirichlet_p=True)`, one Dirichlet
k=0 solve). Every qoi sample records, and `ic` / `summary` repeat:

| key | meaning |
|---|---|
| `gradp_cmp` | $\|\Pi_2 \nabla p_w - \nabla_w p\|_{M_2} / \|\Pi_2 \nabla p_w\|_{M_2}$, gauge-free: $\nabla_w p$ is the weak gradient of the 3-form in the Dirichlet 2-form space and $\Pi_2$ projects the exact $\nabla p_w$ onto the same space, so both lose the same normal trace |
| `p_cmp` | the $L^2$ distance of the two pressures as functions with their means removed, relative to $p_w$'s |
| `weak_resid` | $\|F_w\|_{M_1} / \|v\|_{M_1}$ |
| `dpdn_wall`, `JxBn_wall` | $\max \lvert \partial p_w / \partial n \rvert$ and $\max \lvert (J \times B) \cdot n \rvert$ on the wall, both relative to $\max \lvert \nabla p_w \rvert$ |
| `beta_vol` | $\int p_w \, dV / \int B^2/2 \, dV$; code units, the magnetic pressure is $B^2/2$ |
| `beta_axis` | the same ratio on the coordinate axis (logical $r = 0$: the innermost radial quadrature layer, averaged over $\theta$ and $\zeta$) |

`scripts/poincare_trace.py --pressure weak|strong` (default `weak`) evaluates
either pressure at the crossings, and `scripts/poincare_plot.py` draws it on
the sections. The details are in [Relaxation](concepts/relaxation.md), section 3.

To rebuild the field and evaluate it, load a checkpoint and the run's
geometry:

```python
import h5py, json
from mrx.differential_forms import DiscreteFunction, Pushforward

run = "outputs/relax/<date>/<time>"
prm = json.load(open(f"{run}/relax.json"))["params"]
with h5py.File(f"{run}/checkpoints/state_003000.h5") as fh:
    B = fh["B_n"][...]
seq, ops = build_sequence(prm["geometry_path"], ns=tuple(prm["ns"]), p=prm["p"])
B_phys = Pushforward(DiscreteFunction(B, seq.basis_2, seq.E(2, True)), seq.map, 2)
```

## Poincaré sections

`mrx.poincare` traces field lines of a discrete 2-form with the toroidal
angle as the independent variable, so every crossing of a section plane is
an integration time and nothing is interpolated. One call does it all:
`poincare(seq, B, lines=160, periods=400, planes=5, seed=0)`
seeds `lines` field lines from the magnetic axis to the edge, each at its own
radius and at a random poloidal angle (`seed`), follows them for `periods`
field periods (`seq.nfp` per toroidal turn), measures iota per line and
flags the chaotic ones, and cuts the trajectories at `planes` planes -- a
count spread over half a period for a stellarator-symmetric map, the whole
period otherwise (`seq.symmetry`), or the planes themselves as fractions of
a period; the step count follows from the planes
(every plane a step endpoint, at least 24 per period) and the returned
`drift` (h against h/2) justifies it. The building blocks underneath are
`logical_field`, `seed_from_axis`, `trace`, `rotational_transform` and
`to_RZ`; the module docstring explains the three design choices. Two
drivers split the work by cost. `scripts/poincare_trace.py` (a GPU job)
traces the checkpoints named on its command line on the sequence their
attributes describe over the run's geometry file, evaluates the weak
pressure at every crossing, and archives them in one `trace.npz` next to
the run; `scripts/poincare_plot.py` (plain matplotlib, the login node)
renders that archive, every field and plane on one iota and one pressure
colour scale (the pressure normalised to $p / \langle B^2/2 \rangle$), as
PDF + PNG pages in the publication layout, and is the only thing to rerun
when the figure changes:

```bash
python -u scripts/poincare_trace.py --geometry data/wout_li383_1.4m.nc outputs/run/checkpoints/state_000000.h5 outputs/run/checkpoints/best.h5
python scripts/poincare_plot.py outputs/run            # -> outputs/run/poincare/
```

`--help` lists the flags of both. A movie is the same two calls on every
checkpoint of a run (`checkpoints/state_*.h5`), the plotter's `--window`,
`--iota-lim` and `--p-lim` holding the axes fixed across the frames.

The island chains of a state are found, not read off the picture:
`mrx.poincare.islands(seq, B, res=None)` returns every chain and its width.
From a section `res` (`poincare`'s result; traced here when left out) it
takes the iota profile of the regular lines, lists the rationals
`nfp n / m` with `m <= m_max` inside its range (`resonances`), and at every
radius where the profile meets one it runs Newton on the `m`-period return
map from eight poloidal guesses across one chain period (`fixed_points`;
Cary and Hanson, Phys. Fluids 29, 2464 (1986)). The map over `m` periods is
the one-period map composed `m` times and its tangent map the product of
the one-period Jacobians, forward mode through the integrator, so one
compiled program serves every chain order and every field of a sequence.
Greene's residue `R = 1/2 - tr S / 4` classifies each fixed point (an
O-point for `0 < R < 1`, an X-point for `R < 0`). The width is then
MEASURED: lines seeded on the radial ray through the O-point, all chains
in one batched trace, and the largest `max(r) - min(r)` of the lines locked
to the chain -- the measure the figures and the paper quote, aimed through
the O-point instead of left to where the section's random seeds fall. A
chain is reported only when locked lines pass through its O-point: an
intact rational surface is a curve of fixed points with residue zero up to
integration error. Each entry carries `m`, `n`, `iota`, `r_chain`, the O
and X points with their residues, `width`, and the ray's extent.

The residue is a property of the O-point (the rotation rate about it), not
a width: the constant-shear single-harmonic pendulum relation between the
two overestimated the traced separatrix by 1.35 to 1.6 on the paper's
fields. What the fixed points give reliably is a chain's existence and its
phase (where the O-point sits). The tangent map needs 96 steps per period,
four times the trajectory's, for `det S = 1` to 1e-4. A chain inside a
chaotic band, with no regular line either side of its rational, is not
looked for (`docs/research/island_diagnostic_2026-09-18.md`).

## Figures

`mrx.plotting` draws a scalar on the geometry: `plot_torus` shows the
boundary surface as a wireframe with poloidal cuts coloured by the field,
`plot_crossections_separate` the same cuts side by side in the $(R, z)$
plane, and `plot_twin_axis` two traces against a shared abscissa with
separate y axes (a force residual next to an energy or a helicity); the
tutorials draw all three from their runs.

`scripts/compare_relaxations.py OUT label=run ...` overlays the traces of
several runs (force, energy, $-dE/dt$, helicity, $dt$, CFL, $\|J\|/\|B\|$,
$\beta$, line-search cosine) against relaxation time and step, and draws
the runtime (relaxation time reached per wall hour, seconds per step).

## float32

The default. `--precision float64` exports `MRX_DTYPE` before `mrx` is
imported. In float32 the fields and the Krylov iterations are float32
and every solve is refined against a float64 residual to `--solve-tol`,
so the force is accurate beyond float32's own tolerance and the residual
floor is set by the storage of `B`, not by the solver (until 2026-09-04
it floored at the solve tolerance, $\sim 2 \times 10^{-3}$). See
[Precision](concepts/precision.md).
