# Solve a relaxation problem

`scripts/relax.py` relaxes a magnetic field toward minimum energy at fixed
helicity. The fixed point is $J \times B = \nabla p$: a finite-beta
equilibrium. This guide sets up the geometry and the initial condition,
runs the script, and reads its output. The algorithm is in
[Relaxation](concepts/relaxation.md).

## Geometry

`mrx.geometries.build_sequence` turns a geometry into a polar sequence
with the map installed and every solver operator built:

```python
from mrx.geometries import build_sequence

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
| `--iota I0,I1 [0.4,0.9]`, `--iota-exp E [2.0]`, `--flux-exp Q [1.0]`, `--lam SPEC [""]` | analytic IC only: the profiles above and $\lambda$ modes `"m,n,amp;..."`; ignored for `--ic clebsch` |
| `--method {gradient,cg,lbfgs} [cg]`, `--history M [3]` | descent method and history length |
| `--velocity-smoothing-order G [0]`, `--velocity-smoothing-scale MU [0.0]` | smoothed direction $v = (I - \mu L)^{-G} F$ |
| `--dt-mode {linesearch,fixed} [linesearch]`, `--dt0 DT [1.0]`, `--cfl C [0.5]` | exact energy-minimising step or a fixed one, and the CFL cap |
| `--eta-max ETA [0.0]`, `--eta-schedule {tanh,constant,linear} [tanh]`, `--eta-every K [1]` | resistivity; `tanh` drops it to zero over the middle third of the run; the solve runs every `K` steps |
| `--steps N [3000]`, `--seconds S [none]` | outer budgets |
| `--floor-tol TOL [1e-3]`, `--floor-steps W [100]` | stopping criterion |
| `--qoi-every N [250]` | steps between helicity samples; each is a k=1 Hodge solve |
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
| `relax.json` | `params`; `trace` with per-step `E`, `F`, `resid`, `dt`, `dt_star`, `cfl`, `div`, `cos`, `gain`, `eta`, `res_it`, `res_delta`, `dE_meas`, `dE_pred`; `qoi` with `it`, `helicity`, `JoverB`, `wall`; the `ic` summary and the `summary` with the stopping reason |
| `B.h5` | datasets `B_ic`, `B_final` (Dirichlet k=2 DoFs), `p_ic`, `p_final` (the Leray pressures, 3-form DoFs) with the run parameters as attributes; `geometry` as given and `geometry_path` resolved |

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

To rebuild the field and evaluate it, load the DoFs and the same
geometry:

```python
import h5py
from mrx.differential_forms import DiscreteFunction, Pushforward

with h5py.File("outputs/relax/<date>/<time>/B.h5") as fh:
    B = fh["B_final"][...]
    geometry, ns, p = str(fh.attrs["geometry_path"]), tuple(fh.attrs["ns"]), int(fh.attrs["p"])
seq, ops = build_sequence(geometry, ns=ns, p=p)
B_phys = Pushforward(DiscreteFunction(B, seq.basis_2, seq.e2_dbc), seq.map, 2)
```

## Poincaré sections

`mrx.poincare` traces field lines of a discrete 2-form with the toroidal
angle as the independent variable, so every crossing of a section plane is
an integration time and nothing is interpolated. The building blocks are
`logical_field(seq, dof, 2, dirichlet=True)` for the field,
`seed_line` or `seed_from_axis` for the seeds, `trace` for the
integration, and `rotational_transform` and `to_RZ` for the section.
`step_convergence` justifies the fixed step count by refinement. The module
docstring explains the three design choices. `scripts/poincare_relax.py`
is the driver: it reads the `B.h5` a relaxation wrote, traces `B_ic` and
`B_final`, and renders one section per requested plane:

```bash
python -u scripts/poincare_relax.py outputs/run/B.h5 --periods 400 --out outputs/run/poincare
```

Its module docstring lists the flags.

## float32

The default. `--precision float64` exports `MRX_DTYPE` before `mrx` is
imported. A 200-step CG relaxation on W7-X runs at half the time per step
in float32, with the energy agreeing to five digits. The force residual in
float32 floors at the solve-tolerance level, $\sim 2 \times 10^{-3}$ at
tol $10^{-5}$, so a `--floor-tol` below that never fires and the run ends
on `--steps` or `--seconds`. See [Precision](concepts/precision.md).
