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
  ARGS="--geometry data/torus.json --ns 6,12,6 --steps 50 --chunk 25" \
  bash slurm/run.sh
```

Flags, defaults in brackets:

| flag | meaning |
|---|---|
| `--geometry PATH` (required) | a VMEC wout (`.nc`), a GVEC state (`.dat`) or an analytic geometry (`.json`); the geometry and the initial condition |
| `--nfp N [file attribute]` | field periods, for a file that declares them wrong |
| `--ns R,T,Z [8,16,16]`, `--p P [2]` | resolution (also the map's) and degree |
| `--r-refine a:b:m,... [""]` | radial refinement: `m` uniform cells in each window `[a, b]` of the logical radius, the remaining `n_r - p` cells spread over the gaps (`mrx.geometry.radial_knots`) |
| `--solve-maxiter N [2000]`, `--solve-tol TOL [sqrt(eps)]` | budget and tolerance of every inner solve |
| `--precision {float32,float64} [float32]` | exported as `MRX_DTYPE` before `mrx` is imported |
| `--seed m,n,rho0,width [""]`, `--seed-eps EPS [0]` | equilibrium files only: a resonant `cos(2π(mθ − s nζ))` term in `A'_ζ` at `rho0` (`EPS` = `|δB^ρ|/|B^ζ|` there) that opens an island of width ~`sqrt(EPS)` at the `|iota| = nfp n/m` surface -- a tearing-stability probe |
| `--auxiliary-B-field {false,true} [false]` | `false` reads the 2-form $B$ itself in both cross products, $J \times B$ and $u \times B$; `true` routes them through the auxiliary Dirichlet 1-form $H = M_1^{-1} P B$ ($H_t = 0$ on the wall), the variable that makes the midpoint scheme conserve the discrete helicity exactly |
| `--scheme {explicit,midpoint} [explicit]` | forward Euler on the descent velocity, or the midpoint-implicit induction with the explicit velocity (Picard on the increment, `dt` halved on a blow-up; `mrx.relaxation.PICARD_*`) |
| `--history M [1]` | L-BFGS secant pairs: 0 is steepest descent, 1 memoryless BFGS (= CG) |
| `--velocity-smoothing-order G [0]`, `--velocity-smoothing-scale MU [0.0]` | smoothed direction $v = (I - \mu L)^{-G} F$ |
| `--cfl C [0.5]` | cap on the line-search step, `C /` the velocity's largest logical CFL number; `inf` disables it |
| `--steps N [3000]`, `--seconds S [none]` | outer budgets |
| `--chunk N [500]` | steps per compiled chunk (one `lax.scan`, `mrx.relaxation.chunk_runner`): the per-step trace comes back, the qoi are sampled (helicity, the two pressures and beta, below), a snapshot, the checkpoint and the outputs are written, and the floor, reconnect and wall-time tests run once per chunk; `--steps` is a multiple of it |
| `--floor-tol TOL [1e-3]` | stopping criterion: the last chunk's mean relative force residual below it |
| `--reconnect-every K [0]`, `--reconnect-helicity X [0.01]` | the reconnection series: every `K` steps (rounded to whole chunks) the field (its checkpoint at that step is the one before the solve) is reconnected by one backward-Euler solve `(M_2 + eps L_2) delta = -eps L_2 B`, after which the descent restarts on the diffused field; the dose spends the fraction `X` of the helicity, `eps = X |H| / (2 |∫ J·B|)` from `dH = -2 eps ∫ J·B`; the ideal descent is a power law in the step, not a plateau, so the interval is a choice (`scripts/relax.py` docstring); `results["reconnect"]` records each solve with the helicity actually spent, `scripts/poincare_relax.py --fields ic,final,reconnect` traces the series on one colour scale |
| `--out DIR [outputs/relax/<date>/<time>]` | output directory |
| `--restart PATH` | continue from a `checkpoints/state_<step>.h5` of the same geometry, mesh, degree and precision |

`python scripts/relax.py --help` prints the same list. The script is the
command line of `mrx.relaxation.relax`, the chunked loop with the floor,
wall-budget and reconnection rules, which the tutorials and the tests call
directly; `mrx.initial_conditions.initial_field` builds the field and
`mrx.relaxation.write_checkpoint` / `read_checkpoint` the files below.

## Stopping criterion

The relative force residual $\|F\|_M / \|\nabla(B^2/2)\|$ is recorded at
every step. The run stops when its mean over the last `W` steps,

$$
\frac{1}{W} \sum_{j=i-W+1}^{i} \mathrm{resid}[j] < \texttt{floor-tol},
$$

or when the step or wall-clock budget runs out. The relaxation guarantees
$dE/dt \le 0$ only, so the residual is not monotone; the window mean is
the quantity, never the last value. On the W7-X Clebsch run at `(8,16,8)`,
`p = 3`, float64, the residual reaches $1.7 \times 10^{-3}$ at step 500
and floors around $10^{-3}$ by step 1000-3000. In float32 it floors at the
solve-tolerance level ($\sim 2 \times 10^{-3}$ at tol $10^{-5}$), so a
`--floor-tol` below that never fires.

## Output

`--out` receives:

| file | content |
|---|---|
| `relax.json` | `params` (every flag, `geometry_path` resolved, `ic` the kind of initial condition); `ic`, the initial field's numbers; `trace` with per-step `E`, `F`, `resid`, `dt`, `dt_star`, `cfl`, `div`, `cos`, `gain`, `picard_it`, `picard_resid`, `dE_meas`, `dE_pred`; `qoi` with per-chunk `it`, `wall`, `F`, `resid`, `helicity`, `JoverB`, `JB` and the pressure diagnostics `gradp_cmp`, `p_cmp`, `weak_resid`, `dpdn_wall`, `JxBn_wall`, `beta_vol`, `beta_axis` (the first entry is the start of the run); `reconnect`, one record per reconnection; the `summary` with the stopping reason |
| `checkpoints/state_<step>.h5` | the descent state at that step, one file per chunk plus step 0 (the initial field): every leaf of `mrx.relaxation.State` as a dataset named by its field (`B_n`, `p` the strong pressure, the warm starts, the L-BFGS pairs, `dt`, ...) and the step as an attribute. `--restart` continues from one; the plotters read the field and the strong pressure from them and compute the weak pressure on demand |

`relax.json` and the newest checkpoint are written at every chunk, so a
run that runs out of time still leaves its trace and its last state.

## Inspect

Read the trace with the standard library:

```python
import json
run = json.load(open("outputs/relax/<date>/<time>/relax.json"))
E = run["trace"]["E"]                     # energy after every step
resid = run["trace"]["resid"]             # ||F|| / ||grad(B²/2)|| after every step
H = run["qoi"]["helicity"]                # at the sampled steps
```

Three checks of a healthy run (`--scheme explicit`, no reconnection):

- `E` decreases at every step, and `dE_meas` matches `dE_pred` to
  roundoff. The prediction is an operator identity.
- `helicity` is constant to the solver tolerance.
- `div` stays at roundoff.

`resid` is the force residual relative to the magnetic pressure gradient.
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

`scripts/poincare_relax.py --pressure weak|strong` (default `weak`) draws
either pressure on the sections. The details are in
[Relaxation](concepts/relaxation.md), section 3.

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
an integration time and nothing is interpolated. The building blocks are
`logical_field(seq, dof, 2, dirichlet=True)` for the field,
`seed_from_axis` for the seeds, `trace` for the
integration, and `rotational_transform` and `to_RZ` for the section.
`step_convergence` justifies the fixed step count by refinement. The module
docstring explains the three design choices. `scripts/poincare_relax.py`
is the driver: it reads a run directory, traces the initial and the final
checkpoint (`--fields ic,final,reconnect` adds the field before every
reconnection, all on one colour scale), and renders one section per
requested plane:

```bash
python -u scripts/poincare_relax.py outputs/run --periods 400 --out outputs/run/poincare
```

Its module docstring lists the flags.

A relaxation run stores a checkpoint at every chunk boundary (`--chunk`); `scripts/poincare_relax.py --fields snapshots --planes 0.5` renders one
section per checkpoint with every axis held fixed, ready for `ffmpeg`.

## Figures

`mrx.plotting` draws a scalar on the geometry: `plot_torus` shows the
boundary surface as a wireframe with poloidal cuts coloured by the field,
`plot_crossections_separate` the same cuts side by side in the $(R, z)$
plane, and `plot_twin_axis` two traces against a shared abscissa with
separate y axes (a force residual next to an energy or a helicity).
`scripts/plot_relaxation.py` makes all three from a run -- the weak pressure
$p_w$ on the torus and in the cuts, and $\|F\|_M$ against $E$ from
`relax.json`:

```bash
python -u scripts/plot_relaxation.py outputs/run --cuts 6 --fields ic,final
```

`scripts/compare_relaxations.py OUT label=run ...` overlays the traces of
several runs (force, energy, $-dE/dt$, helicity, $dt$, CFL, $\|J\|/\|B\|$,
$\beta$, line-search cosine) against relaxation time and step, and draws
the runtime (relaxation time reached per wall hour, seconds per step).

## float32

The default. `--precision float64` exports `MRX_DTYPE` before `mrx` is
imported. A 200-step CG relaxation on W7-X runs at half the time per step
in float32, with the energy agreeing to five digits. The force residual in
float32 floors at the solve-tolerance level, $\sim 2 \times 10^{-3}$ at
tol $10^{-5}$, so a `--floor-tol` below that never fires and the run ends
on `--steps` or `--seconds`. See [Precision](concepts/precision.md).
