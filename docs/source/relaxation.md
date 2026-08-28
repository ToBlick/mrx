# Solve a relaxation problem

`scripts/relax.py` relaxes a magnetic field toward minimum energy at fixed
helicity. The fixed point is $J \times B = \nabla p$: a finite-beta
equilibrium. This guide sets up the geometry and the initial condition,
runs the script, and reads its output. The algorithm is in
[Relaxation](concepts/relaxation.md).

## Geometry

`mrx.geometry.build_sequence` turns a geometry into a polar sequence
with the map installed and every solver operator built:

```python
from mrx.geometry import build_sequence

seq, ops = build_sequence("toroid", ns=(8, 16, 8), p=3)
seq, ops = build_sequence("data/w7x_fmm002_clebsch_mrx.h5", ns=(8, 16, 16), p=2)
```

| geometry | map |
|---|---|
| `toroid`, `cylinder`, `rot-ellipse` | analytic, from `mrx.mappings` |
| the path of a GVEC export (`.h5`) | fitted from the file; `os.path.isfile` decides |

Any other string raises. A GVEC file is fitted as three scalar splines on
the sequence's own spline space, so `ns` and `p` are also the map
resolution. `build_gvec_map` checks that $\det D\Phi > 0$ everywhere and
raises otherwise. `nfp=` overrides the file's attribute for a file that
declares it wrong. What a GVEC export must contain is in
[GVEC → MRX interface](concepts/gvec_mrx_interface.md).

## Initial condition

`mrx.initial_conditions` builds every initial field in the reference
2-form frame, where the components are $\sqrt{g} B^i$:

$$
\hat B^\rho = 0, \qquad
\hat B^\chi = \Phi'(\rho)\,(\iota(\rho) - \partial_\zeta \lambda), \qquad
\hat B^\zeta = \Phi'(\rho)\,(1 + \partial_\chi \lambda).
$$

This field is divergence-free and tangent to the boundary for any
$\lambda$ and any geometry. Three sources of the profiles:

| `--ic` | profiles |
|---|---|
| `clebsch` (default) | GVEC's own `dPhi_dr`, `dchi_dr`, and `LA` from the geometry file; needs a file geometry |
| `analytic` | prescribed power laws on the logical grid: $\iota = \iota_0 + (\iota_1 - \iota_0)\rho^e$, $\Phi' = \rho^q$; no external data |
| `dzeta` | the constant 2-form $(0, 0, 1)$; relaxes to the harmonic field |

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
SCRIPT=scripts/relax.py JOB_NAME=relax_w7x TIMEOUT_MIN=60 \
  ARGS="--geometry data/w7x_fmm002_clebsch_mrx.h5" bash slurm/run.sh

SCRIPT=scripts/relax.py JOB_NAME=relax_smoke TIMEOUT_MIN=30 \
  ARGS="--geometry toroid --ic analytic --ns 6,12,6 --steps 50 --qoi-every 25" \
  bash slurm/run.sh
```

Flags, defaults in brackets:

| flag | meaning |
|---|---|
| `--geometry G` (required) | `toroid`, `cylinder`, `rot-ellipse`, or the path of a GVEC export |
| `--nfp N [file attribute]` | field periods, for a file that declares them wrong |
| `--ns R,T,Z [8,16,16]`, `--p P [2]` | resolution (also the map's) and degree |
| `--maxiter N [2000]`, `--tol TOL [sqrt(eps)]` | budget and tolerance of every inner solve |
| `--precision {float32,float64} [float32]` | exported as `MRX_DTYPE` before `mrx` is imported |
| `--ic {clebsch,analytic,dzeta} [clebsch]` | initial condition; `clebsch` with an analytic geometry stops with `use --ic analytic` |
| `--seed m,n,rho0,width [""]`, `--seed-eps EPS [0]` | clebsch IC only: a resonant `cos(2π(mθ − s nζ))` term in `A'_ζ` at `rho0` (`EPS` = `|δB^ρ|/|B^ζ|` there) that opens an island of width ~`sqrt(EPS)` at the `|iota| = nfp n/m` surface -- a tearing-stability probe |
| `--iota I0,I1 [0.4,0.9]`, `--iota-exp E [2.0]`, `--flux-exp Q [1.0]`, `--lam SPEC [""]` | analytic IC only: the profiles above and $\lambda$ modes `"m,n,amp;..."`; ignored for `--ic clebsch` |
| `--method {gradient,lbfgs} [lbfgs]`, `--history M [1]` | descent method and secant pairs (1 = CG) |
| `--velocity-smoothing-order G [0]`, `--velocity-smoothing-scale MU [0.0]` | smoothed direction $v = (I - \mu L)^{-G} F$ |
| `--dt-mode {linesearch,fixed} [linesearch]`, `--dt0 DT [1.0]`, `--cfl C [0.5]` | exact energy-minimising step or a fixed one, and the CFL cap |
| `--eta-max ETA [0.0]`, `--eta-schedule {tanh,constant,linear} [tanh]`, `--eta-every K [1]` | resistivity; `tanh` drops it to zero over the middle third of the run; the solve runs every `K` steps |
| `--steps N [3000]`, `--seconds S [none]` | outer budgets |
| `--floor-tol TOL [1e-3]`, `--floor-steps W [100]` | stopping criterion |
| `--qoi-every N [250]` | steps between the qoi samples: helicity, the two pressures and beta (below) |
| `--out DIR [outputs/relax/<date>/<time>]` | output directory |

`python scripts/relax.py --help` prints the same list.

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

`--out` receives two files:

| file | content |
|---|---|
| `relax.json` | `params`; `trace` with per-step `E`, `F`, `resid`, `dt`, `dt_star`, `cfl`, `div`, `cos`, `gain`, `eta`, `res_it`, `res_delta`, `dE_meas`, `dE_pred`; `qoi` with `it`, `helicity`, `JoverB`, `wall` and the pressure diagnostics `gradp_cmp`, `p_cmp`, `weak_resid`, `dpdn_wall`, `JxBn_wall`, `beta_vol`, `beta_axis`; the `ic` summary and the `summary` with the stopping reason, both with the same pressure diagnostics |
| `B.h5` | datasets `B_ic`, `B_final` (Dirichlet k=2 DoFs), `p_ic`, `p_final` (the strong pressures, 3-form DoFs), `pw_ic`, `pw_final` (the weak pressures, Dirichlet 0-form DoFs) with the run parameters as attributes; `geometry` as given and `geometry_path` resolved |

`relax.json` is rewritten at every qoi sample, so a run that runs out of
time still leaves its trace.

## Inspect

Read the trace with the standard library:

```python
import json
run = json.load(open("outputs/relax/<date>/<time>/relax.json"))
E = run["trace"]["E"]                     # energy after every step
resid = run["trace"]["resid"]             # ||F|| / ||grad(B²/2)|| after every step
H = run["qoi"]["helicity"]                # at the sampled steps
```

Three checks of a healthy ideal run (`--eta-max 0`, `--dt-mode linesearch`):

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
| where | `compute_force`, the Leray multiplier of the descent | `weak_pressure`, from the same `J` and `H` |
| space | 3-form, Dirichlet complex | 0-form, zero on the wall |
| boundary | $\partial p / \partial n = 0$ by construction: $J \times H$ is projected onto the Dirichlet 2-form space first, which discards $(J \times H) \cdot n$ | $p_w = 0$; $J \times H$ is projected onto the natural 1-form space, which keeps $(J \times H) \cdot n$, and $\partial p_w / \partial n$ is the wall force once the remainder $F_w$ vanishes |
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
| `dpdn_wall`, `JxBn_wall` | $\max \lvert \partial p_w / \partial n \rvert$ and $\max \lvert (J \times H) \cdot n \rvert$ on the wall, both relative to $\max \lvert \nabla p_w \rvert$ |
| `beta_vol` | $\int p_w \, dV / \int B^2/2 \, dV$; code units, the magnetic pressure is $B^2/2$ |
| `beta_axis` | the same ratio on the coordinate axis (logical $r = 0$: the innermost radial quadrature layer, averaged over $\theta$ and $\zeta$) |

`scripts/poincare_relax.py --pressure weak|strong` (default `weak`) draws
either pressure on the sections. The details are in
[Relaxation](concepts/relaxation.md), section 3.

To rebuild the field and evaluate it, load the DoFs and the same
geometry:

```python
import h5py
from mrx.differential_forms import DiscreteFunction, Pushforward

with h5py.File("outputs/relax/<date>/<time>/B.h5") as fh:
    B = fh["B_final"][...]
    geometry, ns, p = str(fh.attrs["geometry_path"]), tuple(fh.attrs["ns"]), int(fh.attrs["p"])
seq, ops = build_sequence(geometry, ns=ns, p=p)
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
is the driver: it reads the `B.h5` a relaxation wrote, traces `B_ic` and
`B_final`, and renders one section per requested plane:

```bash
python -u scripts/poincare_relax.py outputs/run/B.h5 --periods 400 --out outputs/run/poincare
```

Its module docstring lists the flags.

A relaxation run with `--save-every K` stores a field snapshot every `K`
steps; `scripts/poincare_relax.py --fields snapshots --planes 0.5` renders one
section per snapshot with every axis held fixed, ready for `ffmpeg`.

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
python -u scripts/plot_relaxation.py outputs/run/B.h5 --cuts 6 --fields ic,final
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
