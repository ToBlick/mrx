# Tutorials

The scripts in `scripts/tutorials/` take a stellarator equilibrium from the
file all the way to a resistively relaxed field, one concept at a time. Five
numbered steps:

1. **load and visualise a geometry** (`qa_geometry.py`),
2. **solve a field** -- the vacuum (coil) field as a curl-curl problem
   (`qa_vacuum_field.py`, with `qa_poisson.py` as a scalar warm-up and
   `qa_vacuum_convergence.py` as the convergence study),
3. **relax** an equilibrium field to a nested state (`li383_relaxation.py`),
4. **seed a magnetic island** in an ideal relaxation (`li383_island_seed.py`),
5. **relax with finite resistivity**, so the field can reconnect
   (`li383_resistive.py`).

Steps 1-2 run on **QA** (`data/wout_LandremanPaul2021_QA_lowres.nc`, the
two-field-period quasi-axisymmetric *vacuum* equilibrium of Landreman & Paul
2021). Steps 3-5 run on **li383** (three-field-period NCSX, the project's
fruit-fly case): step 3 on the coarse reference
(`data/wout_li383_low_res_reference.nc`, `ns = 16`), steps 4-5 on the
high-resolution reference (`data/wout_li383_1.4m.nc`, `ns = 49`) -- the seeded
island and the resistive evolution need an initial condition cleaner than the
coarse file's own reconstruction residual (see step 4). All are VMEC
`wout_*.nc` files read in closed form by `mrx.vmec`; the same `build_sequence`
call reads a GVEC `.dat` state instead (see the
[interface](concepts/gvec_mrx_interface.md)).

Steps 1-2 run in the package's default **float64** -- the vacuum field's
harmonic-form ratio needs it. The li383 relaxations (3-5) run in **float32**,
the production precision. On a cluster run them through `slurm/run.sh` like
every other MRX script:

```bash
SCRIPT=scripts/tutorials/qa_geometry.py JOB_NAME=qa_geometry bash slurm/run.sh
```

They default to `--ns 12,24,12 --p 3` (steps 4-5 use `--ns 12,24,24` and
`--ns 16,32,32`) and write their figures to `outputs/tutorials/<name>/`.

## 1. Load the geometry (`qa_geometry.py`)

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

## 2. Solve a field on the QA domain (`qa_vacuum_field.py`)

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
and the Rayleigh quotient of the Hodge Laplacian (curl and Rayleigh quotient
at round-off ~$10^{-10}$, the divergence at the Leray solve's tolerance),
draws $|B|$ on the torus (`Pushforward(..., k=2)` is the Piola map
$B = DF\,\hat B/\det DF$), and traces a Poincaré section with the rotational
transform profile through `mrx.poincare.section_figure`. **Run it in float64**
(the package default): the harmonic-form ratio only reaches round-off in
double precision. This is the vacuum field *of the bounded domain* -- the wall
is the equilibrium's last closed flux surface -- so it differs from the coil
field outside it.

**A scalar warm-up (`qa_poisson.py`).** Before the curl-curl solve, the same
machinery solves the simplest case, $-\Delta u = 1 - \rho^2$ with $u = 0$ on
the wall, in 0-forms:

```python
rhs = seq.load(f, 0, dirichlet=True)                      # int f v dV
u_hat, info = seq.apply_inverse_laplacian(rhs, 0, dirichlet=True, return_info=True)
```

`load` integrates the source against the basis with the map's volume element,
the Laplacian is the stiffness of the mapped metric, and the solve is the
preconditioned CG behind `apply_inverse_laplacian` (`info` is its iteration
count). The script checks the relative residual and the energy identity
$\int |\nabla u|^2 = \int f u$; see [Solve a Poisson problem](poisson.md) for
the same solve with manufactured solutions on analytic maps.

**Convergence (`qa_vacuum_convergence.py`).** With a coil field known in closed
form there is no discretisation floor, so the error falls at $O(h^p)$. The
script fits an analytic vacuum field
$B^\ast = e_\varphi / R + \lambda\,\nabla(R^2\cos 2\varphi)$ two ways over a
mesh sweep -- as a **scalar potential** $H = \nabla f + \alpha h_1$ (a $k=0$
solve) and as a **vector potential** $B = \operatorname{curl} A$ (a $k=1$
curl-curl solve) -- and plots the relative $M$-norm error against $h$ with an
$O(h^p)$ guide. Both reuse the preconditioned Hodge-Laplacian solvers; nothing
new is built.

## 3. A relaxation on li383 (`li383_relaxation.py`)

li383 is the project's fruit-fly stellarator: a three-field-period
(`nfp = 3`) NCSX configuration with $\iota \approx 0.40 \to 0.66$ and a genuine
pressure. The initial condition is the state's own equilibrium field as
$B = dA'$ from the histopolated Clebsch potential (exactly divergence-free,
tangential to the wall, nested surfaces -- see [Relaxation](relaxation.md),
section 4):

```python
cb = load_clebsch("data/wout_li383_low_res_reference.nc")
B0, norm, wall = potential_two_form(seq, clebsch_potential_form(cb))
```

The descent is `mrx.relaxation` with `scripts/relax.py`'s defaults -- L-BFGS
with history 1 (equivalent to conjugate gradient), analytic line search under
a CFL cap of 0.5, no resistivity -- plus **velocity smoothing of order 1**
(gamma = 1), the descent direction $(I - \text{scale}\,L)^{-1} F$ with
$\text{scale} \approx 0.064 / n_r^2$, run through `relaxation_loop`:

```python
ts = TimeStepper(seq=seq, descent_method=DescentMethod.LBFGS, history_size=1,
                 dt_mode=TimeStepChoice.ANALYTIC_LINESEARCH, cfl=0.5,
                 velocity_smoothing_order=1, velocity_smoothing_scale=0.064 / ns[0] ** 2)
state, traces = relaxation_loop(B0, ts, num_iters_outer=40, num_iters_inner=50,
                                force_tolerance=1e-3)
```

On li383 gamma = 1 reaches a clean nested floor in ~1000 steps where the
unsmoothed descent grinds for ~6000; the force residual need not fall
monotonically, what is judged is the floor it settles at. The relaxation
conserves helicity and lowers the magnetic energy until $J \times B = \nabla p$
in the weak sense; $p$ is not prescribed, it is the multiplier the descent
finds (`weak_pressure`). It runs in float32, the production precision.

The script prints the traces, draws $\|F\|_M$ against $E$ on twin axes
(`plot_twin_axis`) and the weak pressure on the torus, and writes a `B.h5` in
`scripts/relax.py`'s format. `scripts/poincare_relax.py` then draws the
Poincaré sections of the initial and relaxed fields at the standing three
planes $\zeta = 0, 0.25, 0.5$ (per half period):

```bash
python -u scripts/poincare_relax.py outputs/tutorials/li383_relaxation/B.h5 \
    --planes 0,0.25,0.5 --out outputs/tutorials/li383_relaxation
```

## 4. Seed a magnetic island (`li383_island_seed.py`)

The ideal (eta = 0) descent is a frozen-in flow: it moves the field along its
own streamlines, so it cannot change the field's topology. Add a small
**resonant perturbation** to the initial condition and the descent can only let
it breathe -- shrink back if the resonant surface is tearing-stable, or settle
at an $\varepsilon$-independent width if it is tearing-unstable -- it cannot
reconnect the island away. The seed rides on the Clebsch potential, so
$B = dA'$ stays exactly divergence-free and wall-tangent:

```python
seed = (6, 1, 0.544, 0.1, 1e-2)                           # (m, n, rho0, width, eps)
B0, norm, wall = potential_two_form(seq, clebsch_potential_form(cb, seed))
```

The seed adds $\varepsilon\,|\Phi'(\rho_0)|/m\;g(\rho)\cos(2\pi(m\theta - s n\zeta))$
to $A'_\zeta$; $\varepsilon$ is the resonant normal field
$|dB^\rho|/|B^\zeta|$ at $\rho_0$, the chain sits where
$|\iota| = n_{fp}\,n/m$ (`resonant_rho`), and the island has full width about
$1.6\sqrt{\varepsilon\,n_{fp}/(m|\iota'|)}$ in $\rho$. The default
$(m, n) = (6, 1)$ lands on li383's $\iota = 1/2$ surface ($\rho \approx 0.54$).

**This step uses the high-resolution reference** `data/wout_li383_1.4m.nc`
(`ns = 49`). The coarse reference reconstructs the field to a relative residual
of $0.054$, which sits *on top of* the seeded-island signal -- the seed cannot
be told from the reconstruction noise, and the unseeded control already carries
island-scale junk. The high-res file starts at $0.013$ and the seeded chain
stands clear of it. On $(12, 24, 24)\ p = 3$ the $\varepsilon = 1\text{e-}2$
chain reaches a full width near $0.15$ in $\rho$, unmistakable in the Poincaré
section; sweep `--seed-eps` over `1e-3, 3e-3, 1e-2` to watch the width track
$\sqrt{\varepsilon}$, and `--seed 5,1,0.794,0.1` to move to the $3/5$ surface.
The run is otherwise Tutorial 3's descent (gamma = 1, float32) and writes the
same `B.h5`; the island shows in `poincare_relax.py`'s section at the seeded
chain.

## 5. Relax with finite resistivity (`li383_resistive.py`)

Turn on a small resistivity and the frozen-in constraint breaks. Each step is
now the ideal move followed by a backward-Euler diffusion of $B$ (an implicit
resistive solve), first order in $dt$. Field lines can **reconnect**: nested
surfaces merge, a seeded island heals or grows past its frozen-in width, and --
unlike the ideal descent -- helicity is no longer exactly conserved, it decays
at the resistive rate. The resistivity follows a **tanh schedule** -- $\eta$
rises to `--eta-max` over the first third of the run, holds, then drops back to
~0 over the last third, so the tail relaxes ideally to a clean floor once
reconnection has done its work:

```python
def eta_schedule(i):                                      # per outer block
    frac = (i - 0.5) / num_iters_outer
    return eta_max * 0.5 * (1.0 - np.tanh(4.0 * np.pi * (frac - 0.5)))

ts = TimeStepper(seq=seq, ..., eta_every=1, resistive=True)
state, traces = relaxation_loop(B0, ts, num_iters_outer=100, num_iters_inner=50,
                                resistivity_schedule=eta_schedule, force_tolerance=1e-4)
```

The script draws $\|F\|_M$ against $E$ and, on a second twin-axis panel,
helicity $H$ against $\eta$ -- the helicity drop lines up with where the
resistivity is on. It runs on the high-resolution reference at
$(16, 32, 32)\ p = 2$ with `--eta-max 1e-4`, the settings of the `li383_eta`
resistive sweep. Combine with `--seed 6,1,0.544,0.1 --seed-eps 3e-3` (the
Tutorial 4 syntax) to watch a seeded island reconnect instead of merely
breathing: the ideal run freezes the chain, the resistive one lets the resonant
surface tear or heal. It writes the same `B.h5` for `poincare_relax.py`.

---

The production driver `scripts/relax.py` adds the archive, the QoIs, snapshots
for movies, the resistive pre-smoothing, and the island seeds as first-class
options; `scripts/poincare_relax.py` and `scripts/plot_relaxation.py` draw from
its `B.h5`.
