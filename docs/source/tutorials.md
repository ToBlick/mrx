# Tutorials

Four scripts in `scripts/tutorials/` take a stellarator equilibrium from the
file all the way to a relaxed field, one concept at a time. The first three
run on **QA** (`data/wout_LandremanPaul2021_QA_lowres.nc`, the two-field-period
quasi-axisymmetric vacuum equilibrium of Landreman & Paul 2021); the fourth
runs a relaxation on **li383** (`data/wout_li383_low_res_reference.nc`, the
three-field-period NCSX configuration, the project's fruit-fly case). Both are
VMEC `wout_*.nc` files read in closed form by `mrx.vmec`; the same
`build_sequence` call reads a GVEC `.dat` state instead (see the
[interface](concepts/gvec_mrx_interface.md)). They default to
`--ns 12,24,12 --p 3` and write their figures to `outputs/tutorials/<name>/`.

The three QA scripts run in the package's default **float64** -- the vacuum
field's harmonic-form ratio needs it. The li383 relaxation runs in **float32**,
the production precision. On a cluster run them through `slurm/run.sh` like
every other MRX script:

```bash
SCRIPT=scripts/tutorials/qa_geometry.py JOB_NAME=qa_geometry bash slurm/run.sh
```

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

## 2. A scalar Poisson problem (`qa_poisson.py`)

$-\Delta u = 1 - \rho^2$ with $u = 0$ on the wall, in 0-forms:

```python
rhs = seq.load(f, 0, dirichlet=True)                      # int f v dV
u_hat, info = seq.apply_inverse_laplacian(rhs, 0, dirichlet=True, return_info=True)
u_h = DiscreteFunction(u_hat, seq.basis_0, seq.E(0, True))
```

`load` integrates the source against the basis with the map's volume element,
the Laplacian is the stiffness of the mapped metric, and the solve is the
preconditioned CG behind `apply_inverse_laplacian` (`info` is its iteration
count). The script checks the relative residual and the energy identity
$\int |\nabla u|^2 = \int f u$ from the discrete operators, and draws $u$ on
the QA torus. This is the same discrete Laplacian the vacuum field and the
relaxation lean on; see [Solve a Poisson problem](poisson.md) for the same
solve with manufactured solutions on analytic maps.

## 3. The vacuum field (`qa_vacuum_field.py`)

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

## 4. A relaxation on li383 (`li383_relaxation.py`)

The initial condition is li383's equilibrium field as $B = dA'$ from the
histopolated Clebsch potential (exactly divergence-free, tangential to the
wall, nested surfaces -- see [Relaxation](relaxation.md), section 4):

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

The production driver `scripts/relax.py` adds the archive, the QoIs, snapshots
for movies, resistivity and island seeds; `scripts/poincare_relax.py` and
`scripts/plot_relaxation.py` draw from its `B.h5`.
