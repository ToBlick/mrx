# Solve a relaxation problem

`scripts/relax.py` relaxes a magnetic field toward minimum energy at fixed
helicity. The fixed point is $J \times B = \nabla p$: a finite-beta
equilibrium. This guide sets up the geometry and the initial condition,
runs the script, and reads its output. The algorithm is in
[Relaxation](concepts/relaxation.md).

## Geometry

`mrx.geometries.build_sequence` turns a geometry name into a polar
sequence with the map installed and every solver operator built:

```python
from mrx.geometries import build_sequence

seq, ops = build_sequence("toroid", ns=(8, 16, 8), p=3)
```

Names:

| name | map |
|---|---|
| `toroid`, `cylinder`, `rot-ellipse` | analytic, from `mrx.mappings` |
| `w7x` | the W7-X map fitted from `W7-X.h5` |
| any key of `mrx.gvec.GVEC_GEOMETRIES` | a GVEC export, for example `quasr44970`, `w7x-fmm002` |

Files are read from `MRX_DATA` (default `data/`). A GVEC file is fitted
as three scalar splines on the sequence's own spline space, so `ns` and
`p` are also the map resolution. `build_gvec_map` checks that
$\det D\Phi > 0$ everywhere and raises otherwise. What a GVEC export must
contain is in [GVEC → MRX interface](concepts/gvec_mrx_interface.md).

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
| `logical` | power laws: $\iota = \iota_0 + (\iota_1 - \iota_0)\rho^e$, $\Phi' = \rho^q$; no external data |
| `clebsch` | GVEC's own `dPhi_dr`, `dchi_dr`, and `LA` from the geometry file |
| `dzeta` | the constant 2-form $(0, 0, 1)$; relaxes to the harmonic field |

In code:

```python
from mrx.initial_conditions import (logical_profile_form, make_lambda, make_profiles,
                                    project_reference_two_form, leray_clean)

iota, dPhi = make_profiles(iota0=0.4, iota1=0.9, iota_exp=2.0, flux_exp=1.0)
omega_ref = logical_profile_form(iota, dPhi, make_lambda([]))
B0, B_norm = project_reference_two_form(seq, omega_ref)   # DoFs of the Dirichlet k=2 space
B0, moved = leray_clean(seq, B0)                            # remove the projection's divergence
```

`project_reference_two_form` pushes the form forward and projects with
`load(frame='phys')`. Do not pass the primal components to
`load(frame='ref')`: that argument wants $g\omega/J$ and fails silently.

## Run

Every run is a GPU job through `slurm/run.sh`:

```bash
SCRIPT=scripts/relax.py JOB_NAME=relax_smoke TIMEOUT_MIN=30 \
  ARGS="--geometry toroid --ns 6,12,6 --p 2 --steps 50 --diag-every 25" \
  bash slurm/run.sh

SCRIPT=scripts/relax.py JOB_NAME=relax_w1 TIMEOUT_MIN=90 \
  ARGS="--geometry w7x-fmm002 --ic clebsch --ns 8,16,8 --p 3 --steps 3000" \
  bash slurm/run.sh
```

Flags, defaults in brackets:

| flag | meaning |
|---|---|
| `--geometry NAME [quasr44970]` | a name from the table above |
| `--ns R,T,Z [8,16,8]`, `--p P [3]` | resolution (also the map's) and degree |
| `--maxiter N [10000]`, `--tol TOL [sqrt(eps)]` | budget and tolerance of every inner solve |
| `--precision {float64,float32} [float64]` | exported as `MRX_DTYPE` before `mrx` is imported |
| `--ic {logical,clebsch,dzeta} [logical]` | initial condition |
| `--iota I0,I1 [0.4,0.9]`, `--iota-exp E [2.0]`, `--flux-exp Q [1.0]`, `--lam SPEC [""]` | the logical profiles; `SPEC` is `"m,n,amp;..."` |
| `--no-lambda`, `--no-leray-ic` | clebsch with $\lambda = 0$; skip the Leray clean-up |
| `--method {gradient,cg,lbfgs} [cg]`, `--history M [1]` | descent method and history length |
| `--gamma G [0]`, `--mu MU [0.0]` | hyperregularisation $v = (I - \mu L)^{-G} F$ |
| `--dt-mode {linesearch,fixed} [linesearch]`, `--dt0 DT [1.0]` | exact energy-minimising step, or a fixed one |
| `--eta-max ETA [0.0]`, `--eta-schedule {tanh,constant,linear} [tanh]` | resistivity; `tanh` drops it to zero over the middle third of the run |
| `--steps N [3000]`, `--seconds S [none]` | outer budgets |
| `--floor-tol TOL [10*eps]`, `--floor-window W [100]` | stopping criterion |
| `--diag-every N [250]` | steps between helicity and residual samples; each is a k=1 Hodge solve |
| `--out DIR [outputs/relax/<date>/<time>]` | output directory |

`python scripts/relax.py --help` prints the same list.

## Stopping criterion

The run stops when the energy decrease over the last `W` steps, relative
to the energy and per step,

$$
\frac{E[i - W] - E[i]}{W\,|E[i]|} < \texttt{floor-tol},
$$

or when the step or wall-clock budget runs out. The relaxation guarantees
$dE/dt \le 0$ only. The force residual need not fall monotonically and is
not used to stop.

## Output

`--out` receives two files:

| file | content |
|---|---|
| `relax.json` | `params`; `trace` with per-step `E`, `F`, `dt`, `div`, `cos`, `gain`, `eta`, `dE_meas`, `dE_pred`; `diagnostics` with `it`, `helicity`, `resid`, `gradp`, `JoverB`, `wall`; the `ic` summary and the `summary` with the stopping reason |
| `B.h5` | datasets `B_ic` and `B_final` (Dirichlet k=2 DoFs) with the run parameters as attributes |

`relax.json` is rewritten at every diagnostic sample, so a run that runs
out of time still leaves its trace.

## Inspect

Read the trace with the standard library:

```python
import json
run = json.load(open("outputs/relax/<date>/<time>/relax.json"))
E = run["trace"]["E"]                     # energy after every step
resid = run["diagnostics"]["resid"]       # ||F|| / ||grad(B²/2)|| at the sampled steps
H = run["diagnostics"]["helicity"]
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
seq, ops = build_sequence("toroid", ns=(8, 16, 8), p=3)
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

Pass `--precision float32`. The script exports `MRX_DTYPE` before `mrx`
is imported. A 200-step CG relaxation on W7-X runs at half the time per
step, with the energy agreeing to five digits. The stopping tolerance
scales with the precision: `--floor-tol` defaults to `10 * eps`, which is
$1.2 \times 10^{-6}$ in float32. See [Precision](concepts/precision.md).
