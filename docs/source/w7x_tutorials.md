# W7-X tutorials

Four scripts in `scripts/tutorials/` take a GVEC equilibrium of W7-X from
the file to a relaxed equilibrium. They all default to GVEC's own state
file `data/GVEC_State_final.dat` (330 KB; a VMEC `wout_*.nc` works the
same way, see the [interface](concepts/gvec_mrx_interface.md)),
`--ns 12,24,24 --p 3` and write their figures to `outputs/tutorials/<name>/`;
each takes a few minutes on a GPU and correspondingly longer on a CPU. On
a cluster run them through `slurm/run.sh` like every other MRX script:

```bash
SCRIPT=scripts/tutorials/w7x_geometry.py JOB_NAME=w7x_geometry bash slurm/run.sh
```

## 1. Load the geometry (`w7x_geometry.py`)

The state file `GVEC_State_*.dat` is GVEC's own representation of the
equilibrium: $R$, $Z$ and $\lambda$ as radial B-splines times Fourier
series in $(\theta, \zeta)$ -- $\zeta$ spans **one field period**, `nfp`
completes the torus -- plus the profiles $\Phi$, $\chi$, $\iota$, $p$ at
the radial interpolation points. It is closed form: MRX evaluates the
series wherever it needs a value (`mrx.gvec.StateField`), there is no
grid in between, and the map error is the map space's own. A VMEC
`wout_*.nc` is refit into the same blocks by `mrx.vmec` (each Fourier
mode a clamped spline in $\rho = \sqrt{s}$).

```python
from mrx.geometry import build_sequence
seq, ops = build_sequence("data/GVEC_State_final.dat", (12, 24, 24), 3)
```

`build_sequence` is the import. It

- builds the spline coefficients of $R$ and $Z$ on the map's own spline
  space -- resolution `ns`, degree `p`, polar at the axis (the first radial
  ring of coefficients is $C^1$ across $\rho = 0$), periodic in both
  angles -- from the series coefficients, mode by mode (the L2 projection,
  no evaluation grid);
- measures the toroidal handedness so that $\det DF > 0$ and installs
  $F(\rho, \theta, \zeta) = (R \cos\varphi, \pm R \sin\varphi, Z)$ with
  $\varphi = 2\pi\zeta/n_{fp}$ (the `[geom]` line prints the sign and the
  range of $\det DF$);
- assembles the incidence operators, the mass and Laplacian preconditioners
  of all four form degrees on that metric.

The script prints what the file holds (basis, modes and profiles), the
DoF counts, and draws the pressure with `mrx.plotting.plot_torus` (the
wall as a wireframe, poloidal cuts coloured by the scalar) and
`plot_crossections_separate`. The pressure comes from `load_clebsch`,
which tabulates the state's profile splines on 401 uniform radii -- the
same dict that feeds the relaxation's initial condition in tutorial 4.

## 2. A Poisson problem (`w7x_poisson.py`)

$-\Delta u = 1 - \rho^2$ with $u = 0$ on the wall, in 0-forms:

```python
rhs = seq.load(f, 0, dirichlet=True)                      # int f v dV
u_hat, info = seq.apply_inverse_laplacian(rhs, 0, dirichlet=True, return_info=True)
u_h = DiscreteFunction(u_hat, seq.basis_0, seq.E(0, True))
```

`load` integrates the source against the basis with the map's volume
element, the Laplacian is the stiffness of the mapped metric, and the
solve is the preconditioned CG behind `apply_inverse_laplacian`
(`info` is its iteration count). The script checks the relative residual
and the energy identity $\int |\nabla u|^2 = \int f u$ from the discrete
operators, and draws $u$ on the torus. See [Solve a Poisson problem](poisson.md)
for the same solve with manufactured solutions on analytic maps.

## 3. The vacuum field (`w7x_vacuum_field.py`)

Inside a perfectly conducting wall the current-free field with
$B \cdot n = 0$ and one unit of toroidal flux is the **harmonic 2-form** of
the Dirichlet complex: $\operatorname{curl} B = 0$, $\operatorname{div} B = 0$,
tangential to the wall. MRX constructs it directly -- Leray-project a seed
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
at round-off, the divergence at the Leray solve's tolerance), draws
$|B|$ on the torus (`Pushforward(..., k=2)` is the Piola map
$B = DF\,\hat B/\det DF$), and traces a Poincaré section with the
rotational transform profile through `mrx.poincare.section_figure`. This
is the vacuum field *of the bounded domain*: the wall is the equilibrium's
last closed flux surface, so it differs from the coil field outside it.
`scripts/relax.py --ic dzeta` relaxes onto this field.

## 4. A short relaxation (`w7x_relaxation.py`)

The initial condition is the state's equilibrium field as $B = dA'$ from
the histopolated Clebsch potential (exactly divergence-free, tangential to
the wall, nested surfaces -- see [Relaxation](relaxation.md), section 4):

```python
cb = load_clebsch(path)
B0, norm, wall = potential_two_form(seq, clebsch_potential_form(cb))
```

The descent is `mrx.relaxation` with the defaults of `scripts/relax.py`
-- conjugate-gradient direction, analytic line search under a CFL cap of
0.5, no resistivity, no velocity smoothing -- run through
`relaxation_loop` (a Python loop of compiled `lax.scan` chunks):

```python
ts = TimeStepper(seq=seq, descent_method=DescentMethod.LBFGS,
                 dt_mode=TimeStepChoice.ANALYTIC_LINESEARCH, cfl=0.5, ...)
state, traces = relaxation_loop(B0, ts, num_iters_outer=10, num_iters_inner=30)
```

It conserves helicity and lowers the magnetic energy until
$J \times B = \nabla p$ in the weak sense; $p$ is not prescribed, it is the
multiplier the descent finds (`weak_pressure`). The script prints the
traces, draws $\|F\|_M$ against $E$ on twin axes (`plot_twin_axis`), the
weak pressure on the torus, and the sections before and after. Three
hundred steps at this resolution take $\|F\|$ from $2.7 \cdot 10^{-3}$ to
$1.5 \cdot 10^{-4}$ with the helicity conserved to $10^{-8}$; the
production driver `scripts/relax.py` adds the archive, the QoIs, snapshots
for movies, resistivity, velocity smoothing and island seeds, and
`scripts/poincare_relax.py` and `scripts/plot_relaxation.py` draw from its
`B.h5`.
