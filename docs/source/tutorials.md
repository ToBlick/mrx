# Tutorials

The scripts in `scripts/tutorials/` take a stellarator equilibrium from the
file all the way to a resistively relaxed field, one concept at a time. Five
numbered steps:

1. **load and visualise a geometry** (`1_qa_geometry.py`),
2. **solve a field** -- the vacuum (coil) field as a curl-curl problem
   (`2_qa_vacuum_field.py`),
3. **relax** an equilibrium field to a nested state (`3_li383_relaxation.py`),
4. **seed a magnetic island** in the initial field (`4_li383_island_seed.py`),
5. **reconnect** with a resistive step, warm-started from step 3
   (`5_li383_resistive.py`).

Steps 1-2 run on **QA** (`data/wout_LandremanPaul2021_QA_lowres.nc`, the
two-field-period quasi-axisymmetric *vacuum* equilibrium of Landreman & Paul
2021). Steps 3-5 run on **li383** (three-field-period NCSX, the project's
fruit-fly case): steps 3 and 5 on the coarse reference
(`data/wout_li383_low_res_reference.nc`, `ns = 16`), step 4 on the
high-resolution reference (`data/wout_li383_1.4m.nc`, `ns = 49`) -- the seeded
island must clear the coarse file's own reconstruction residual (see step 4). All are VMEC
`wout_*.nc` files read in closed form by `mrx.vmec`; the same `build_sequence`
call reads a GVEC `.dat` state instead (see the
[interface](concepts/gvec_mrx_interface.md)).

All five run in the package default, now **float32**, the production
precision. Tutorial 2's harmonic-form ratio reaches round-off only in double
precision -- run it with `MRX_DTYPE=float64` for that. On a cluster run them
through `slurm/run.sh` like
every other MRX script:

```bash
SCRIPT=scripts/tutorials/1_qa_geometry.py JOB_NAME=qa_geometry bash slurm/run.sh
```

Steps 1-2 default to `--ns 12,24,12 --p 3`; steps 3-5 to `--ns 10,16,16 --p 2`.
They write their figures to `outputs/tutorials/<name>/`.

## 1. Load the geometry (`1_qa_geometry.py`)

A VMEC `wout_*.nc` stores the flux surfaces as $R$ and $Z$ Fourier series in
the angles $(\theta, \zeta)$ -- $\zeta$ spans **one field period**, `nfp`
completes the torus -- with the stream function $\lambda$ and the profiles
$\Phi$, $\chi$, $\iota$, $p$ on the radial grid. `mrx.vmec` refits each Fourier
mode into a clamped B-spline in $\rho = \sqrt{s}$, so the wout lands in exactly
the same radial-splines $\times$ Fourier-series blocks a GVEC state
(`GVEC_State_*.dat`) is read into, and everything downstream is closed form:
MRX evaluates the series wherever it needs a value, there is no grid in between.

```python
from mrx.geometry import build_sequence
seq, ops = build_sequence("data/wout_LandremanPaul2021_QA_lowres.nc", (12, 24, 12), 3)
```

`build_sequence` is the import. It

- builds the spline coefficients of $R$ and $Z$ on the map's own spline space
  -- resolution `ns`, degree `p`, polar at the axis (the first radial ring of
  coefficients is $C^1$ across $\rho = 0$), periodic in both angles -- from
  the series coefficients, mode by mode (the L2 projection, no evaluation grid);
- measures the toroidal handedness so that $\det DF > 0$ and installs
  $F(\rho, \theta, \zeta) = (R \cos\varphi, \pm R \sin\varphi, Z)$ with
  $\varphi = 2\pi\zeta/n_{fp}$ (the `[seq]` line prints the range of $\det DF$);
- assembles the incidence operators, the mass and Laplacian preconditioners
  of all four form degrees on that metric.

QA is a **vacuum** equilibrium: its pressure is zero, so there is nothing to
colour a pressure plot with. The script draws the map's Jacobian $\det DF$
instead -- the volume element the whole complex is weighted by, larger on the
outboard side of the torus and squeezed on the inboard side -- with
`mrx.plotting.plot_torus` (the wall as a wireframe, poloidal cuts coloured by
the scalar) and `plot_crossections_separate`. It also prints what the file
holds (basis, modes, the $\iota$ profile) and the DoF counts.

## 2. Solve a field on the QA domain (`2_qa_vacuum_field.py`)

Inside a perfectly conducting wall the current-free field with $B \cdot n = 0$
and one unit of toroidal flux is the **harmonic 2-form** of the Dirichlet
complex: $\operatorname{curl} B = 0$, $\operatorname{div} B = 0$, tangential
to the wall. QA is a vacuum equilibrium -- zero pressure, zero current -- so
this harmonic 2-form *is* its equilibrium field, reconstructed here from the
bounded geometry alone. MRX constructs it directly -- Leray-project a seed
field, subtract the $\operatorname{curl}$ part with one more Hodge solve --
and keeps it on the operators, where the relaxation's Leray projection and
helicity are deflated against it:

```python
from mrx.nullspace import compute_nullspaces, get_nullspace
seq.set_operators(compute_nullspaces(seq, ops))
B = get_nullspace(seq.get_operators(), 2, True)[0]
```

The script verifies $\|\operatorname{div} B\|$, $\|\operatorname{curl} B\|$
and the Rayleigh quotient of the Hodge Laplacian (both floor near
single-precision epsilon ~$10^{-3}$ in the default float32, and reach
round-off ~$10^{-10}$ in float64; the divergence sits at the Leray solve's
tolerance), draws $|B|$ on the torus (`Pushforward(..., k=2)` is the Piola map
$B = DF\,\hat B/\det DF$), then traces the field lines once and takes Poincaré
sections at five toroidal planes $\zeta = 0, 0.125, 0.25, 0.375, 0.5$. It runs
in the default float32; for the harmonic-form ratio at round-off use
`MRX_DTYPE=float64`. This is the vacuum field *of the bounded domain* -- the wall
is the equilibrium's last closed flux surface -- so it differs from the coil
field outside it.

## 3. A relaxation on li383 (`3_li383_relaxation.py`)

li383 is the project's fruit-fly stellarator: a three-field-period
(`nfp = 3`) NCSX configuration with $\iota \approx 0.40 \to 0.66$ and a genuine
pressure. The initial condition is the state's own equilibrium field as
$B = dA'$ from the histopolated Clebsch potential (exactly divergence-free,
tangential to the wall, nested surfaces -- see [Relaxation](relaxation.md),
section 4):

```python
cb = load_clebsch(seq.equilibrium)   # the file build_sequence parsed
B0, norm, wall = potential_two_form(seq, clebsch_potential_form(cb))
```

The descent is `mrx.relaxation` with `scripts/relax.py`'s defaults -- L-BFGS
with history 1 (equivalent to conjugate gradient), analytic line search under
a CFL cap of 0.5, no resistivity -- plus **velocity smoothing of order 1**
(gamma = 1), the descent direction $(I - \text{scale}\,L)^{-1} F$ with
$\text{scale} \approx 0.064 / n_r^2$, run through `relax`:

```python
ts = TimeStepper(seq=seq, history_size=1, cfl=0.5,
                 velocity_smoothing_order=1, velocity_smoothing_scale=0.064 / ns[0] ** 2)
res = relax(initial_state(B0, ts), ts, steps=500, chunk=50, floor_tol=1e-3)
```

On li383 gamma = 1 reaches a clean nested floor in ~1000 steps where the
unsmoothed descent grinds for ~6000; the force residual need not fall
monotonically, what is judged is the floor it settles at. The relaxation
conserves helicity and lowers the magnetic energy until $J \times B = \nabla p$
in the weak sense; $p$ is not prescribed, it is the multiplier the descent
finds (`weak_pressure`). It runs in float32, the production precision.

The script prints the traces, draws $\|F\|_M$ against $E$ on twin axes
(`plot_twin_axis`) and the weak pressure on the torus, and writes the run in
`scripts/relax.py`'s layout (`relax.json` and two checkpoints). `scripts/poincare_relax.py` then draws the
Poincaré sections of the initial and relaxed fields at the standing three
planes $\zeta = 0, 0.125, 0.25, 0.375, 0.5$ (half a field period; the other half follows by stellarator symmetry):

```bash
python -u scripts/poincare_relax.py outputs/tutorials/li383_relaxation \
    --planes 0,0.125,0.25,0.375,0.5 --out outputs/tutorials/li383_relaxation
```

## 4. Seed a magnetic island (`4_li383_island_seed.py`)

The ideal (eta = 0) descent is a frozen-in flow: it cannot change the field's
topology, so a seeded island can only breathe, never reconnect (that is
Tutorial 5's job). Here we skip the descent and just look at the **initial
field** the seed produces, next to the unseeded control, so the effect of the
seed is unmistakable. The seed rides on the Clebsch potential, so $B = dA'$
stays exactly divergence-free and wall-tangent:

```python
seed = (6, 1, 0.544, 0.1, 1e-2)                           # (m, n, rho0, width, eps)
B_seeded, norm, wall = potential_two_form(seq, clebsch_potential_form(cb, seed))
```

The seed adds $\varepsilon\,|\Phi'(\rho_0)|/m\;g(\rho)\cos(2\pi(m\theta - s n\zeta))$
to $A'_\zeta$; $\varepsilon$ is the resonant normal field
$|dB^\rho|/|B^\zeta|$ at $\rho_0$, the chain sits where
$|\iota| = n_{fp}\,n/m$ (`resonant_rho`), and the island has full width about
$1.6\sqrt{\varepsilon\,n_{fp}/(m|\iota'|)}$ in $\rho$. The default
$(m, n) = (6, 1)$ lands on li383's $\iota = 1/2$ surface ($\rho \approx 0.54$).

The script traces both initial fields once and draws Poincaré sections at five
toroidal planes: the island at the resonant chain shows in the seeded section,
not the unseeded one. It uses the **high-resolution reference**
`data/wout_li383_1.4m.nc` -- on the coarse reference the field's reconstruction
residual sits on top of the seeded signal, so the seed cannot be told from the
noise. Sweep `--seed-eps` over `1e-3, 3e-3, 1e-2` to watch the width track
$\sqrt{\varepsilon}$, and `--seed 5,1,0.794,0.1` to move to the $3/5$ surface.
There is no relaxation and no run directory; this is the cheapest tutorial.

## 5. Reconnect with finite resistivity (`5_li383_resistive.py`)

Turn on a small resistivity and the frozen-in constraint breaks. A step is now
the ideal move followed by a backward-Euler diffusion of $B$ (an implicit
resistive solve): field lines can **reconnect**, nested surfaces merge, a
seeded island heals or grows, and helicity is no longer conserved -- it decays
at the resistive rate.

This tutorial is arranged to be cheap. It **warm-starts from Tutorial 3's
relaxed field** if the run `outputs/tutorials/li383_relaxation` is present (same
$(10, 16, 16)\ p = 2$ mesh), so the initial descent is not repeated; otherwise
it builds the equilibrium initial condition itself. It then takes a **single
resistive step** at `--eta-max` and relaxes ideally for another 500 steps:

```python
B_reconnected, _, rel = resistive_step(B0, seq, eps)                    # one reconnection step
res = relax(initial_state(B_reconnected, ts), ts, steps=500, chunk=50,
            floor_tol=1e-4)                                             # 500 ideal steps
```

The helicity drop across the resistive step is the reconnection; the ideal tail
conserves it. The script draws $\|F\|_M$ against $E$ over the tail and the weak
pressure on the torus, and writes the run for `poincare_relax.py`. Pass
`--seed 6,1,0.544,0.1 --seed-eps 3e-3` (the Tutorial 4 syntax) when it falls
back to building the IC, to watch a seeded island reconnect.

---

The production driver `scripts/relax.py` is the command line of the same
`mrx.relaxation.relax`: the checkpoints at every chunk (movies, restarts),
the reconnection series and the island seeds as flags;
`scripts/poincare_relax.py` and `scripts/plot_relaxation.py` draw from its
run directory.
