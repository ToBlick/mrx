# Tutorials

The scripts in `scripts/tutorials/` take a stellarator equilibrium from the
file all the way to a resistively relaxed field, one concept at a time, and
end with a shape optimization. Seven numbered steps:

1. **load and visualise a geometry** (`1_qa_geometry.py`),
2. **solve a field** -- the vacuum (coil) field as a curl-curl problem
   (`2_qa_vacuum_field.py`),
3. **relax** an equilibrium field to a nested state (`3_li383_relaxation.py`),
4. **Newton** on the second variation, from step 3's state to the floor
   (`4_li383_newton.py`),
5. **seed magnetic islands** in step 4's floor by the energy criterion and relax
   them ideally (`5_li383_island_seed.py`),
6. **drive** the seeded field towards the resistive steady state of the
   unseeded equilibrium's current (`6_li383_drive.py`),
7. **optimize the shape** of the QA boundary for quasi-axisymmetry by
   reverse-mode differentiation of the vacuum field
   (`7_qa_shape_optimization.py`).

Steps 1, 2 and 7 run on **QA** (`data/wout_LandremanPaul2021_QA_lowres.nc`, the
two-field-period quasi-axisymmetric *vacuum* equilibrium of Landreman & Paul
2021). Steps 3-6 run on **li383** (three-field-period NCSX, the project's
fruit-fly case), all four on the coarse reference
(`data/wout_li383_low_res_reference.nc`, `ns = 16`). All are VMEC
`wout_*.nc` files read in closed form by `mrx.vmec`; the same `build_sequence`
call reads a GVEC `.dat` state instead (see the
[interface](concepts/gvec_mrx_interface.md)).

All but step 7 run in the package default, now **float32**, the production
precision; step 7 sets float64 itself (see step 7). Tutorial 2's harmonic-form ratio reaches round-off only in double
precision -- run it with `MRX_DTYPE=float64` for that. On a cluster run them
through `slurm/run.sh` like
every other MRX script:

```bash
SCRIPT=scripts/tutorials/1_qa_geometry.py JOB_NAME=qa_geometry bash slurm/run.sh
```

All of them default to the mesh `--ns 12,16,16`, steps 1, 2 and 7 with
`--p 3`, steps 3-6 with `--p 2`.
They write their figures to `outputs/tutorials/<name>/`.

The li383 tutorials can be run in any order: the end state of each ships in
`data/tutorials/<name>/` (a `relax.json` with the parameters and the last
checkpoint, from the runs at the shipped defaults). Steps 4, 5 and 6 warm-start
from the user's own run in `outputs/tutorials/` when it exists and from the
shipped state otherwise, and `scripts/poincare_trace.py --run
data/tutorials/li383_newton` sections a shipped state without running
anything.

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
seq, ops = build_sequence("data/wout_LandremanPaul2021_QA_lowres.nc", (12, 16, 16), 3)
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
compute_nullspaces(seq)   # installs the forms on the sequence
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

The descent is `mrx.relaxation` with `scripts/relax.py --method gradient`'s
defaults -- gradient descent on the smoothed projected force by the potential
route, analytic line search under a CFL cap
of 0.5, no resistivity -- plus **velocity smoothing of order 1**
(gamma = 1), the descent direction $(I - \text{scale}\,L)^{-1} F$ with
$\text{scale} = 0.075\, h_r^2$ ($h_r$ the physical radial cell) (`mrx.relaxation.SMOOTHING_C`, the stepper's
default), run through `relax`:

```python
ts = TimeStepper(seq=seq, cfl=0.5, velocity_smoothing_order=1)
res = relax(initial_state(B0, ts), ts, steps=500, chunk=50, floor_tol=1e-6)
```

On li383 gamma = 1 reaches a clean nested floor in ~1000 steps where the
unsmoothed descent grinds for ~6000; the force residual need not fall
monotonically, what is judged is the floor it settles at. The relaxation
conserves helicity and lowers the magnetic energy until $J \times B = \nabla p$
in the weak sense; $p$ is not prescribed, it is the multiplier the descent
finds (`weak_pressure`). It runs in float32, the production precision.

The script prints the traces, draws $\|F\|_M$ against $E$ on twin axes
(`plot_twin_axis`) and the weak pressure on the torus, and writes the run in
`scripts/relax.py`'s layout (`relax.json` and two checkpoints). `scripts/poincare_trace.py` then traces the
initial and relaxed fields at the standing five planes $\zeta = 0, 0.125, 0.25, 0.375, 0.5$
(half a field period; the other half follows by stellarator symmetry) into the run's
`trace.npz`, and `scripts/poincare_plot.py` draws the sections from it (no GPU):

```bash
python -u scripts/poincare_trace.py --geometry data/wout_li383_low_res_reference.nc outputs/tutorials/li383_relaxation/checkpoints/state_*.h5
python scripts/poincare_plot.py outputs/tutorials/li383_relaxation
```

## 4. Newton to the floor (`4_li383_newton.py`)

The descent's tail is a power law: the residual falls as a power of the step,
never a plateau, and the directions the energy is flat along -- surfaces
sliding past each other, current sheets thinning -- crawl, because a gradient
method scales every mode by the same time step and the stiffest modes set it.
Newton scales each mode by the inverse of its own curvature. The energy along
the flow of a divergence-free $u$ expands to second order: the gradient is
minus the Lorentz force, the Hessian $H$ is the second variation ($2\,\delta W$
of ideal MHD at $p = 0$; minus the linearised force operator at an
equilibrium). Newton's equation $H u = J \times B$ is solved in the potential
form, $u = \operatorname{curl} a$ with
$\operatorname{curl}^T H \operatorname{curl} a = \operatorname{curl}^T (J \times B)$
-- divergence-free by construction, no Leray solve -- by MINRES with the
harmonic atom (the Laplacian atom with the parallel symbol of the harmonic
field divided in), **truncated**: 100 iterations, warm-started from the
previous step's potential. The budget is the parameter: a more exact
direction is closer to the ideal descent, which past the resolved floor
thins current sheets the mesh cannot carry, so 100 iterations hold the floor
where 300 leave it. The
rest of the step is Tutorial 3's -- the analytic line search along the
direction, capped at the Newton step $\Delta t = 1$, the CFL cap, the update a
curl so $\operatorname{div} B$ and the helicity stay exact -- and a direction
that is not a descent direction is replaced by the smoothed force for that
step:

```python
ts = TimeStepper(seq=seq, cfl=0.5, newton=True)
res = relax(initial_state(B_start, ts), ts, steps=10, chunk=5)
```

Newton is the floor finder. Warm-started from a state the descent has taken
through its fast phase it reaches the mesh's residual floor in tens of steps
where the descent needs thousands -- on li383 at $(16, 32, 32)\ p = 2$, 49
Newton steps (16 min) reach a floor the descent does not reach in 18 000
steps (3.2 h) -- at 15-30 descent steps per Newton step. The floor it finds
depends on the route, which corner of the orbit the descent left it in, and
neither method leaves a corner for a lower one; so the rule is descent
through its fast phase, then Newton
(`docs/research/newton_second_variation_2026-09-06.md`). The paper's floors
are at sixteen radial cells and more; on the tutorial's twelve, ten Newton
steps take $\|F\|_M$ down 27-fold (14-fold in the first five) with the
helicity held to $10^{-4}$, where 200 descent steps take it down threefold --
ten steps, a minute or two on a GPU.

The script warm-starts from Tutorial 3's run (or runs that descent itself),
continues the smoothed descent and Newton from the same state, prints the
step costs and draws $\|F\|_M$ against the step and the wall time for both,
and writes the Newton run in `scripts/relax.py`'s layout for Tutorial 6 and
`poincare_trace.py`. `scripts/relax.py` runs the same from the command line,
Newton being its default method (the `--newton-*` flags in
[Relaxation](relaxation.md)).

## 5. Seed magnetic islands by the energy criterion (`5_li383_island_seed.py`)

The ideal (eta = 0) flow is frozen-in: it cannot change the field's topology,
so a seeded island can only move and reshape, never reconnect (that is
Tutorial 6's job). Here we open islands in Tutorial 4's Newton floor the way
the paper does (Sec. 6.2): every resonance $\iota = n_{fp} n / m$ inside the
field's iota range gets SIESTA's parallel seed $dB = \mathrm{curl}(A B / |B|)$,
$A = a(r) \cos(2\pi(m\theta - n\zeta))$, its radial profile free in the mesh's
own spline basis near the resonant radius, and all chains are solved together
for the amplitudes of least energy:

```python
from mrx.seeding import energy_seed
B_seeded, rows = energy_seed(seq, B_floor)                      # every chain in range, the criterion's amplitudes
B_hand, rows = energy_seed(seq, B_floor, iotas=[0.5], amplitudes=[3e-3])   # one chain, by hand
```

There is no phase to choose: a phase shift only scales the seed (the
stellarator parity projector removes the odd part exactly), and the sign of
the profile is the criterion's. The script prints, for every chain, its
rotational transform and radius, the amplitude chosen (the resonant normal
field $|dB^r| / |B^\zeta|$ at the chain), the pendulum width it implies and
how much energy the seed removes; then it sections the floor and both seeded
fields at five toroidal planes and measures each chain's island width from
the lines locked to its rotational transform, relaxes the automatic seed
with 5 Newton steps (the chains survive), and writes the run in
`scripts/relax.py`'s layout plus `reference.h5`, the unseeded floor. On the
command line the same is `scripts/relax.py --seed [--seed-iotas ... --seed-amplitudes ...]`.

## 6. Drive the field towards the unseeded equilibrium's current (`6_li383_drive.py`)

The paper's last experiment (Sec. 7) breaks the frozen-in constraint with a
**drive**: after every ideal step a backward-Euler resistive step
$\partial_t B = -\eta\,\mathrm{curl}(J - J^*)$ with the dose
$\varepsilon = C h_r^2$ pulls the current towards $J^*$, the current of a
reference field $B^*$ -- here Tutorial 5's unseeded floor after one heat step
that removes its rational-surface sheets. Field lines can now reconnect and
helicity is no longer conserved: the field goes to the resistive steady state
of the reference current, the same state from any start, seeded or not (the
paper's three arms converge to it), with islands where the reference's sheets
were -- the chains settle at that state's widths, they do not simply close.

```python
cfg = RelaxConfig(geometry=geometry, budget=Budget(steps=20, chunk=5, floor_tol=0.0),
                  drive=Drive(resistivity=0.064, reference="reference.h5", reference_smoothing=0.1))
ts = cfg.stepper(seq, h_r_sq)                                       # the stepper with the resistivity
B_star = resistive_step(B_floor, seq, 0.1 * h_r_sq)[0]              # the smoothed reference
ts = eqx.tree_at(lambda t: t.resistive_reference, ts, B_star, is_leaf=lambda x: x is None)
res = relax(initial_state(B_seeded, ts), ts, on_chunk=progress, **cfg.relax_kwargs())
```

The script starts from Tutorial 5's seeded, relaxed state (or makes it), and
prints the dose, the reference's smoothing and, after every chunk, the force
residual, the helicity and the distance to the reference; before and after
it sections the field and measures every seeded chain's island width. On the
command line the same is `scripts/relax.py --drive-resistivity 0.064
--drive-reference reference.h5 --restart <seeded checkpoint>`.

## 7. Optimize the QA boundary for quasi-axisymmetry (`7_qa_shape_optimization.py`)

The paper's shape optimization (Sec. 3.4) in small. The vacuum field of
Tutorial 2 is one linear solve on the map, $\mathfrak h_2 = \mathfrak s -
\operatorname{curl} A$ with $\mathfrak s$ a geometry-free flux seed, wrapped in
`jax.lax.custom_linear_solve`, so the reverse-mode gradient of any function of
it with respect to every spline coefficient of the map costs one more (adjoint)
solve:

```python
from mrx.shape_ad import (BoundaryShape, cylindrical_geometry, flux_seed, mean_iota, normal_field_fraction,
                          quasisymmetry_residual, vacuum_two_form, with_geometry)
shape = BoundaryShape.from_coefficients(seq, raw_R, raw_Z, nfp, sign, aspect=6.0, extension="harmonic", free="all")

def terms(x, seq, shape, seed):
    R, Z, _, _ = shape.map_coefficients(seq, 0.01 * x.reshape((2,) + shape.raw_R.shape))   # V and A held
    sq = with_geometry(seq, cylindrical_geometry(seq, R, Z, nfp, sign))
    h, _ = vacuum_two_form(sq, seed)
    return quasisymmetry_residual(sq, h, R, Z, nfp, sign, h_r)[0], mean_iota(sq, h), normal_field_fraction(sq, h)
```

The problem is the paper's: minimize $\langle Q_{\mathrm{QA}}^2
\rangle_{r \geq h_r}$ subject to $\langle \bar\iota \rangle_s$ at LP's own
value and $P \leq 10^{-8}$ (the field normal to the logical surfaces, which
keeps them close to flux surfaces), at the volume and aspect ratio of the
device, by an augmented Lagrangian over `scipy.optimize`'s L-BFGS-B on
`jax.value_and_grad`. The start is LP plus a smooth random normal displacement
of its boundary, 10 mm RMS in modes $m, |n| \leq 4$ per field period, and the
run stops when the criterion is back at $F_{\mathrm{LP}}$, LP's own value on
this mesh, with both constraints met. At the defaults that takes 86 L-BFGS-B
iterations, from $2.2 \times 10^3\,F_{\mathrm{LP}}$ to $0.67\,F_{\mathrm{LP}}$,
about 5 min on a GPU and 11 min on four CPU cores, setup included; the boundary ends $3.0 \times 10^{-2}$ of
the minor radius from LP's (from $5.5 \times 10^{-2}$ at the start): as in the
paper, quasi-axisymmetry and the mean iota come back, the device does not.
The script draws the criterion and the mean iota against the iteration and
the three boundaries in three toroidal planes.

It is the one tutorial in **float64** (the script sets `MRX_DTYPE=float64`):
in float32 the criterion, a residual near $10^{-5}$, and its adjoint gradient
carry the solves' round-off, and the run stalls near $50\,F_{\mathrm{LP}}$.
The paper's driver is `scripts/paper_scripts/ad_recovery.py`.

---

The production driver `scripts/relax.py` is the command line of the same
`mrx.relaxation.relax`: the checkpoints at every chunk (movies, restarts),
the drive and the island seeds as flags;
`scripts/poincare_trace.py` + `scripts/poincare_plot.py` draw its checkpoints.
