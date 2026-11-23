# Solve a relaxation problem

`scripts/relax.py` relaxes a magnetic field toward minimum energy at fixed
topology (up to grid-level effects). The fixed point is $J \times B = \nabla p$, 
a finite-beta equilibrium. This guide covers the input, the run and its output. The
algorithm is in [Relaxation](concepts/relaxation.md).

## Geometry and initial condition

The geometry file fixes the map and the initial field. A VMEC wout
(`.nc`), a GVEC state (`.dat`) or a DESC output (`.h5`) gives the polar spline map of the file's
series and the file's own field $B = dA'$ as the initial field
([Equilibrium input](concepts/equilibrium_input.md)). An analytic shape is
used by writing a mock equilibrium file from its formulas, for example a
VMEC wout or a GVEC state, as `test/synthetic_gvec.py` and
`test/synthetic_desc.py` do.

A VMEC input namelist (`--geometry.path input.<name>`) gives the map from
its boundary alone. The interior is filled between a guessed magnetic axis
and the boundary as VMEC fills its first surfaces
([Equilibrium input](concepts/equilibrium_input.md)), and the initial field
is the vacuum field of the domain with the namelist's toroidal flux
`PHIEDGE`. The vacuum field carries no current, so it is already an
equilibrium without pressure. A run from it needs a target, for example the
namelist's own pressure (`--pressure.target input.<name>`, see "Prescribed
pressure" below) or its current. The logical surfaces of the fill are not the
flux surfaces of the field, so the run takes more steps than on the mesh of
an equilibrium file.

    python -u scripts/relax.py --geometry.path input.li383 --geometry.graded --geometry.resolution 48 64 24 \
        --budget.steps 400 --pressure.target input.li383

With li383's boundary and pressure (`PHIEDGE` and the wout's pressure
tabulated in the namelist, beta 4.3 %), the squared force residual stays
below 1e-8 from step 257 on, and the pressure misses its target by 9.7e-3
after 50 steps and 4.7e-3 after 400. The label's solve takes 4267
iterations on the vacuum and 1700 to 3300 later (one H200, float64, 4.7
minutes for the operators and 5.7 for the 257 steps, 2026-11-19).

The toroidal coordinate of the first mesh is set by `--geometry.zeta`. With
`axis-normal`, the default, the surfaces $\zeta = $ const are the planes
normal to the axis of the file's map, as in GVEC's G-frame. With `phi` they
are the file's own sections, the planes $\phi = $ const of the cylindrical
angle for a VMEC or DESC file. The planes $\phi = $ const are tilted
against the axis of a stellarator, by up to 11 degrees on li383, 18 on
W7-X and 35 on HSX. On a tilted plane the metric couples the field along the
axis to the components across it in the first radial cell. A field made by
a metric projection, a vacuum field above all, then has a rotational
transform near the axis that is off by an amount that does not fall with the
mesh (HSX vacuum, (24,48,48): 1.000 within 5 mm of the axis against 1.042
outside). On planes normal to the axis the coupling vanishes there, and the
same vacuum has 1.042 down to 5 mm.

The cut (`mrx.relaxation.mesh_motion.axis_normal_map`) takes the planes
normal to the axis of the file's map, in the frame of that axis: the axis,
the radial direction made normal to the axis, and the third direction. Every
point of the mesh slides along its line of constant $r$ and $\theta$ into
the plane of its $\zeta$. So every surface $r = $ const of the file, the
wall among them, only slides along itself, and the poloidal angle stays. The
new coordinates in the plane are interpolated by the splines of the map.
The file's field is carried exactly: the new point $(r, \theta, \zeta)$ is
the file's point $(r, \theta, \zeta + f)$, and with
$\lambda'(r, \theta, \zeta) = \lambda(r, \theta, \zeta + f) - \iota(r)
(2\pi / N_{\rm fp}) f$ the field of the potential is the file's field, not
the field dragged along with the points. On li383 (48,64,24), W7-X
(24,48,28) and HSX (32,64,36) the cut takes 5 to 8 s on 8 CPU cores, the
sections at the axis are normal to it to 3e-4 degrees, the map is at most
0.08, 0.13 and 0.02 mm from the exact cut, and the field on the cut map is
the file's to 1.1e-3, 1.6e-3 and 6.7e-4 relative rms. Dragged with the
points it would be off by 2.0e-2, 2.9e-2 and 3.4e-2 (2026-11-20).

With `harmonic` the surfaces $\zeta = $ const are the level sets of the
potential of the vacuum field of the domain,
so that $\nabla \zeta$ is the harmonic 1-form $h$, the direction of the
vacuum field, fixed by the boundary alone. The planes are not: on li383, W7-X
and HSX $\|\nabla \zeta - h\| / \|h\|$ is 0.29, 0.35 and 0.65 on the
planes. The map gets there by moves of its points along their lines of
constant $r$ and $\theta$.

The misfit is a scalar $q$ on the mesh. One Neumann solve for the Laplacian
on the map gives the harmonic angle $\zeta + q$
(`mrx.nullspace.harmonic_angle`), so $q$ says how far the logical $\zeta$
is from the harmonic angle at each point. The mesh is aligned in $\zeta$
when $q = 0$. A move slides every point along its line of constant $r$ and
$\theta$, from $\zeta$ to the point the map has at
$\zeta - q / (1 + \partial_\zeta q)$
(`mrx.relaxation.mesh_motion.harmonic_displacement`). The surfaces
$r = $ const and the poloidal angle stay, and the wall slides along
itself. The change of the map is projected onto its spline space and halved
if a cell would fold, as in a move of the pole. On the moved map $q$ is solved
again. The shift is the Newton step of $q = 0$ along each line, so each
move about squares the error until the tolerance of the solves. From the planes the distance
$\|\nabla \zeta - h\| / \|h\|$ (`mrx.nullspace.zeta_harmonic_distance`) falls
0.29, 1.6e-2, 1.6e-5, 2.5e-7 on li383, 0.35, 3.1e-2, 9.5e-5, 2.3e-7 on
W7-X and 0.65, 9.3e-2, 2.4e-3, 5.3e-6 on HSX. The moves stop below 1e-5,
after three moves on all three. The first move shifts the points by up to
1.4, 1.8 and 2.8 toroidal cells, the later ones by a tenth of a cell and
less. The wall moves by 0.3, 0.2 and 0.1 mm, the error of its spline. The
map is the one a single exact section along the lines gives, to 0.2 mm.

The moves are made before the field exists, so they only change the map and
nothing is carried. Each costs one Neumann solve, a fraction of a second
after the first. The solves need only the preconditioners of the 0-forms,
so the others are built once, on the final map, and the vacuum field is
computed there. On one H200 on li383 (48,64,24), W7-X (24,48,28) and HSX (32,64,36) the
setup takes 35 to 55 s longer than with `phi`, which takes 240 to 285 s,
and the relaxation is the same (2026-11-19). The harmonic choice
needs a VMEC input namelist, whose start is the vacuum field: the harmonic
moves start from the file's sections and do not carry a file's field.

```python
from mrx.geometry import build_sequence
from mrx.relaxation.initial_conditions import initial_field
from mrx.nullspace import compute_nullspaces

seq = build_sequence("data/wout_li383_1.4m.nc", (16, 32, 32), 2)
compute_nullspaces(seq)
B0, info = initial_field(seq)
```

## Run

Every run is a GPU job through `slurm/run.sh` ([Running on a cluster](cluster.md)):

```bash
SCRIPT=scripts/relax.py JOB_NAME=relax_li383 TIMEOUT_MIN=120 \
  ARGS="--geometry.path data/wout_li383_1.4m.nc --geometry.resolution 16 32 32" bash slurm/run.sh

SCRIPT=scripts/relax.py JOB_NAME=relax_smoke TIMEOUT_MIN=30 \
  ARGS="--geometry.path data/wout_li383_low_res_reference.nc --geometry.resolution 8 12 12 --budget.steps 50" bash slurm/run.sh
```

`python scripts/relax.py --help` lists every flag with its default. The
command line is built by [tyro](https://brentyi.github.io/tyro/) from the
dataclasses in {mod}`mrx.relaxation.config`, one group each. A field is the
flag `--<group>.<field>` with hyphens for underscores, a tuple takes its
values separated by spaces (`--geometry.resolution 16 32 32`) and a bool is
the pair `--seed` / `--no-seed`:

| group | flags |
|---|---|
| geometry | `--geometry.path`, `--geometry.symmetry`, `--geometry.resolution`, `--geometry.spline-degree`, `--geometry.knots-r/theta/zeta`, `--geometry.graded`, `--geometry.zeta`, `--geometry.solve-tol`, `--geometry.solve-maxiter`, `--geometry.max-batch` |
| iota | `--iota.target` |
| seed | `--seed`, `--seed.iotas`, `--seed.amplitudes`, `--seed.scale` |
| descent | `--descent.method {newton,gradient}` |
| newton | `--newton.penalty`, `--newton.tol`, `--newton.maxiter`, `--newton.mass-solve-tol` |
| budget | `--budget.steps`, `--budget.chunk`, `--budget.floor-tol` |
| drive | `--drive.resistivity`, `--drive.reference`, `--drive.reference-smoothing`, `--drive.chain`, `--drive.eps` |
| pressure | `--pressure.target`, `--pressure.scale`, `--pressure.beta`, `--pressure.anisotropy`, `--pressure.label-tol`, `--pressure.cells` |
| current | `--current.target`, `--current.scale` |
| mesh | `--mesh.align`, `--mesh.rebuild-factor`, `--mesh.bar`, `--mesh.move-tol` |
| output | `--output.out`, `--output.restart` |

The `params` of `relax.json` hold the same settings as one flat dict with
the keys of earlier runs (`geometry`, `resolution`, `spline_degree`,
`seed_iotas`, `method`, `newton_tol`, `steps`, `drive_resistivity`, `out`
and so on).

The tutorials build the same configuration in Python
(`RelaxConfig(geometry=Geometry(...), ...)`).

## Island seeds

`--seed` adds small resonant perturbations to the start field, or to the
field of a restart, so that the relaxation can open island chains
({func}`mrx.relaxation.seeding.energy_seed`). The seeds sit at the rational
surfaces of the field's own transform. It is measured on the field's flux
label as a function of $s$, the same label the iota and pressure targets use,
and never taken from the file. The seeds' radial profiles are functions of
$s$ too, so a seed sits on its resonant surface wherever the mesh puts it. The amplitudes lower the energy the most, or are given per
chain with `--seed.iotas` and `--seed.amplitudes`. With `--iota.target` the
seed waits until the transform is at its target (see below).

## Prescribed pressure

The incompressible relaxation keeps the volume of every flux shell, so it ends
with the pressure the start's volumes give. With `--pressure.target` the
volumes become unknowns of every Newton step and the run ends at the target
pressure (`mrx.relaxation.shells`, `mrx.relaxation.newton.target_direction`).
The shells are the splines of the profile space, $s$ the flux label of the
field. The profile space is its own, shared by the pressure and the current:
cubic splines in $s$ on `--pressure.cells` cells whose breakpoints are
uniform in $\sqrt{s}$, by default half the mesh's radial cells
(`mrx.relaxation.shells.profile_cells`): every second radial knot $r_k$
carried into the label by $s = r^2$, so every profile cell spans two
radial cells and the space refines with the mesh. A spline in $s$ is smooth in space next to the
magnetic axis, where $s$ goes as the squared distance from it, so every
profile is regular there. Each shell's volume change is carried by a
gradient $\xi_k = \nabla \chi_k$ with $\nabla \cdot \xi_k = f_k$ and
$\xi_k \cdot n = 0$ on the wall, $f_k$ the projection of the shell function
less its mean onto the 3-forms. It is the velocity part of a mixed Poisson
problem of the 3-forms, the same solve as the Leray projection's. A gradient is
orthogonal to every divergence-free velocity, so the shells add to the step
only what changes the volumes. The rebuild solves one such problem per shell
and keeps one velocity per shell, 200 to 420 conjugate gradients each on li383
and W7-X. The step minimises $W = \int B^2/2 \,dV - \int
p^*(s) \,dV$ over the divergence-free velocities and the shells' volume
changes. Its shell rows vanish when the shell moments of the pressure and the
target agree. The shells and the label are rebuilt before every chunk.

The step lowers $W$, not the magnetic energy, which may rise on a step that
changes the shells' volumes. The trace has `dW` next to `dE`: the change of
$W$ over the step, with the target pressure's work to first order in the step.
The summary's `W_removed` is its sum.

The target is the geometry file's pressure (`--pressure.target file`) or the
pressure profile of a VMEC input namelist (`--pressure.target input.<name>`).
Measured pressure is fitted by `scripts/pressure_fit.py`, an outer loop (see
Outer loops below). The target is $\mu_0 p$ in T^2, the units of the
field. A profile is multiplied by `--pressure.scale`, or
scaled to `--pressure.beta`: at every rebuild its factor is set so that
$\int p^* \,dV / \int B^2/2 \,dV$ is that beta on the field as it is then.
The shells change their volumes along the run, and with them the volume mean
of a function of the label, so a factor set once at the start gives another
beta at the end (1.2 % less on li383 with the file's profile times 0.25, from
1.065 % at the start).

Two things a target cannot ask for. A uniform pressure does no work on a
fixed wall, so no move changes the pressure's constant. It is fixed by the
gauge $\langle p \rangle = \langle p^* \rangle$ of the volume means: the
run's pressure is $p_w + C$, with $p_w$ the weak pressure, which vanishes on
the wall, and $C = \langle p^* - p_w \rangle$. A target is set to zero on the
wall. The target is evaluated at the label of every quadrature point. An
island chain wider than the label's flattening width is one level set of the
label, so the target has one value on it, as the two branches of the
separatrix are one surface.

The `pressure` record has the error of $p_w + C$ against the target, $C$ and
`beta`, the volume beta of the target and so of $p_w + C$. `beta_vol` is that
of $p_w$ alone and differs from it by $C$ (0.9984 % against 1 % in the
example below).

    python -u scripts/relax.py --geometry.path data/wout_li383_1.4m.nc --geometry.resolution 32 44 20 \
        --geometry.graded --budget.steps 300 --pressure.target file --pressure.beta 0.01

On this mesh, from the file's field at 4.28 %, the pressure's error against
the target over 48 bins of $\sqrt{s}$ is 1.4e-2 after 20 Newton steps, 7.5e-3
after 30, 1.3e-3 after 100 and 2.7e-3 after 300, with $C$ at 1.2e-3 of the
target's maximum. `dW` is negative on every step (one H200, 10 minutes with
the build of the operators, 2026-11-18). The 32 radial cells of this mesh
resolve the 16 cells of the profile space only just: on (48,64,24) with both
of the file's targets the error is 4.9e-4 after 200 steps, and 6.7e-4 and
3.4e-4 with `--pressure.cells` 8 and 24.

A VMEC input namelist gives its pressure profile, `PRES_SCALE` times the
`PMASS_TYPE` profile of `AM` (or of `AM_AUX_S` and `AM_AUX_F`) in pascal, as
$\mu_0 p$ (`mrx.equilibria.indata`). As a target only the profiles of the
namelist are read, the geometry is the geometry file's. A namelist with `GAMMA` other than
zero is refused. With a namelist `input.li383_two_power` that holds
`PMASS_TYPE = 'two_power', AM = 7.34e4 1.0 2.0`, the pressure $(1 - s)^2$,

    python -u scripts/relax.py --geometry.path data/wout_li383_1.4m.nc --geometry.resolution 32 44 20 \
        --geometry.graded --budget.steps 300 --pressure.target input.li383_two_power --pressure.beta 0.01

ends at an error of 2.7e-3, with $C$ at 8.5e-5 of the target's maximum,
from 1.3 at the start (7.4e-3 after 20 steps, 9.5e-4 after 100, one H200,
10 minutes). A namelist that tabulates the file's own pressure ends where
`--pressure.target file` ends, to 1.3e-6 in the field's coefficients, as close as two runs
of the file's target on different nodes (3.8e-6).

## Prescribed current

The ideal relaxation keeps the toroidal and the poloidal flux of every flux
surface, so it ends with the net toroidal current that these fluxes give.
With `--current.target` the poloidal flux between the flux surfaces becomes
an unknown of every Newton step and the run ends at the target current inside
every surface (`mrx.relaxation.twist.Slips`,
`mrx.relaxation.newton.target_direction`). The move is a twist of the
flux label's surfaces, $dB = \nabla \times (F(s) \nabla \zeta)$ with $\zeta$ the
logical toroidal angle. It keeps every
surface and its toroidal flux and has no flux through the wall. $F$ is a
combination of steps of the splines of the profile space, each of which slips
poloidal flux across one spline. The steps vanish on the axis, since a
constant $F$ is no move. They are splines in $s$, so the first goes as $s$ at
the magnetic axis and the potential is smooth there. The last step changes
the poloidal flux inside the wall and with it the total current.

The derivative of the magnetic energy along a step is the mean of the
enclosed current $\mu_0 I$ over its spline. So the step minimises the energy
less the work of the target current on the poloidal flux it slips,
$W = \int B^2/2 \,dV - \int \mu_0 I^*(s) \,d\psi(s)$, over the ideal moves and
the slips. Its slip rows vanish when the spline means of the current and of
the target agree, the spline next to the axis included. There is no constant
to fix, since the current vanishes on the axis. $W$ is quadratic in the slips, so the slip
block of the Newton system is the energy's own second derivative and it is
positive definite. An iteration of the step's MINRES costs a tenth more than one
without slips.
The slips change the helicity, as a current drive does. With a pressure
target too, the step has both the shells and the slips on one flux label, and
both are rebuilt before every chunk.

The target is the current of the geometry file's field on its flux label
(`--current.target file`) or the current profile of a VMEC input namelist
with `NCURR = 1` (`CURTOR` times the `PCURR_TYPE` profile of `AC`, or of
`AC_AUX_S` and `AC_AUX_F`), multiplied by `--current.scale`. It is $\mu_0 I$
in T m, the units of the field. The namelist's current is along $\phi$, it is
turned along the field by the sign of `PHIEDGE`. One namelist gives both
targets, `--pressure.target input.x --current.target input.x`. A current
target sets the transform, so it goes without `--iota.target`, which reads a
namelist with `NCURR = 0` only and sets the current target itself (below). The label of the current target is the
pressure's (`--pressure.anisotropy`, `--pressure.label-tol`).

The `current` record has, before every chunk and at the end, the spline means
of the field's current and their error against the target's, the L2 error in
$s$ relative to the target. A zero target has no size to compare with, its
error is the absolute one in T m.

A pressure fixes an equilibrium only together with the current or the
transform, the other is an outcome. Without `--current.target` and
`--iota.target` the slips are no unknowns: every surface keeps its poloidal
flux, so the run keeps the start's transform and the current follows from it.
A boundary and a pressure alone are closed by no net current on any surface,
`--current.target file --current.scale 0` from the vacuum field, which has
none: the slips are unknowns and the transform is free.

    python -u scripts/relax.py --geometry.path data/wout_li383_1.4m.nc --geometry.resolution 48 64 24 \
        --geometry.graded --budget.steps 200 --current.target file --current.scale 0.9

On this mesh the current's error against 0.9 times the file's current is
1.1e-1 at the start, 4.4e-4 after 10 Newton steps and 2.5e-6 after 200. `dW`
is negative on every step, the force residual reaches 4.2e-11 as without the
target, and the helicity changes by 10 %. With a namelist `input.x` that
holds the file's pressure and current, the pressure reshaped by
$1 + 0.05 (1 - 2s)$ and the current times 0.95, `--pressure.target input.x
--current.target input.x` ends at a pressure error of 4.9e-4 and a current
error of 2.3e-6 after 200 steps (2.7e-3 and 1.2e-4 after 20, one H200,
2026-11-18).

## Rotational transform

The relaxation keeps the fluxes of every flux surface, so it keeps the
rotational transform of the start field. With `--iota.target <file>` it ends
at the transform of a wout or of a VMEC input namelist with `NCURR = 0`. The
transform is no control of the energy, the current is. So the transform sets
the target of the prescribed current in an outer loop, before every chunk
({func}`mrx.relaxation.twist.iota_newton`):

1. the miss is the weak residual on the profile space,
   $r_j = \int_0^1 (\iota - \iota^*) M_j \, ds$, with $M_j$ the derivative
   splines of the profile space, the local mean of the miss over each spline
   ({func}`mrx.relaxation.twist.iota_residual`). By the coarea formula it is
   the volume integral of $M_j(s)$ against the poloidal flux density
   $n_{fp} B \cdot \nabla\theta_a$ less $\iota^*$ times the toroidal one, with
   $\theta_a$ the angle about the magnetic axis
   (`mrx.relaxation.mesh_motion.magnetic_axis`). It needs no sorting by
   $s$, no derivative of a sampled flux and no division by $s$,
2. the target enters only the slip rows, so the derivative of the state along
   the target's moments is the Gauss-Newton step's solution for the slip rows'
   derivative, one MINRES solve per moment,
3. ideal moves keep the poloidal flux of every flux surface as a function of
   its toroidal flux, so only the slips change $r$. A twist $F$ changes the
   transform by $-n_{fp} F'(s)/\Phi$, and the change is taken from the
   discrete twists, which resolve the innermost steps poorly
   ({func}`mrx.relaxation.twist.slip_transform`),
4. the target takes the joint Newton step on the field and the target, a
   square system on the profile space that includes the slips of the Newton
   direction at the old target.

The loop starts from the current the field carries, and every checkpoint
keeps the target, so a restart goes on from it. With `--seed` the seed is applied at the first chunk boundary at which
every local mean of the weak residual is below 1 % of the target's largest
value, since the seed resonates with the field's own transform. The `iota` record has,
before every chunk, the weak residual, the step and the current target.

On li383 (48,64,24), graded, on the file's sections, with 0.95 times the
file's transform from the file's field after 80 Newton steps, the largest weak residual falls from 0.18 to
1.5e-2 after the first update, 1.3e-3 after the third and 2.7e-5 at step 300,
the target near the wall settles within three updates and the force residual
reaches 1.1e-9. The 19 MINRES solves cost 256 s over the 22 updates (one H200,
2026-11-20, with the induction through $B$ of that date). The transform keeps an oscillation at the scale of the profile
cells, 1.5e-3 at most, which the current-prescribed relaxation has too
(4.4e-3): it is orthogonal to the $M_j$, so no target reaches it.
On W7-X standard (24,48,28), graded, p = 2, with the polynomial transform and
pressure of a namelist, from the namelist alone, the start puts the pole on
the vacuum's axis, 2.9 first cells off the fill's, in 8 rounds and 96 s. The
pole stays at step 0. The pressure carries the field's axis 2.9 first cells
back, the field first reaches the bar at step 80, and the pole moves there.
The transfer changes the physical field at fixed points by 0.15 and the
residual from 9e-9 to 3e-5. The field stays below the bar from step 90, the
axis ends 0.013 first cells off the pole, and the preconditioners are never
built again. The traced transform within rho 0.3 of the axis is within 5.5e-4
of the target, the largest weak residual 6.2e-6 (one H200, 2026-11-23).

The `mesh` record of `relax.json` holds the rounds
of the start, every move (the step, the axis offset in first radial cells, the
axis finder's record, the largest move and its backtracking factor, the range
of the change of $\det D\Phi$, the transfer's substeps, the relative change of
the coefficients and of the energy, the force residual right before and right
after the move and the wall time), the checks without a move and why, the
chunks' mean MINRES counts against the reference and the builds of the
preconditioners with their reason and wall time. Every checkpoint of such a
run holds its map, which a restart and `scripts/poincare_trace.py` install,
and `moved_<step>.h5` holds the field right after a move. `best.h5` is the
best field since the last move.

## Stopping

The squared normalised force residual
$\mathrm{resid} = \|F\|_M^2 / \|\nabla(B^2/2)\|^2$ is recorded at every
step. The run stops when its mean over the last chunk falls below
`--budget.floor-tol`, or when `--budget.steps` is spent. With
`--pressure.target` or `--current.target` the run takes all of
`--budget.steps`: the residual does not see the shells or the slips, so its
floor does not end a run that still approaches its target. The residual holds the
harmonic part of the force, which a field without stellarator symmetry may
have. The Newton step then has the amplitude of the harmonic flow as one more
unknown and removes that part too. The residual is not monotone.
Judge a run by the floor it settles at, and take `checkpoints/best.h5` when
it went past it. A job's time limit is no stop: the checkpoint of every
chunk restarts it (`--output.restart`).

## Output

A run writes to `--output.out`, by default `relax/<date>/<time>/` in the
runs root, which is `$MRX_DATA/runs/mrx`, or `outputs/` in the working
directory without `MRX_DATA` ([Running on a cluster](cluster.md)).

| file | content |
|---|---|
| `relax.json` | `params`: the configuration, flat, plus `geometry_path`, `ic`, `resolution`, `nfp`, `h_r_sq`, `start_step`. `ic`: the initial field's numbers. `iota`, `seed`, `drive`: when used. `pressure`: the target and `rebuilds`, before every chunk and at the end of the run: `step`, `label_it`, `error` and `constant` (the pressure's error against the target and its constant $C$ over the target's maximum), `beta` (the target's), `scale` (its factor, with `--pressure.beta`) and, before every chunk, `shell_it` (the signed counts of the shells' Poisson solves), with `scripts/pressure_fit.py` also `fit`, the posterior. `current`: the target, the centres `x` (in $\sqrt{s}$) of the splines of the profile space and the target's moments `target_moments` ($\mu_0 I$ in T m), and `rebuilds`, before every chunk and at the end: `step`, `label_it`, `error` (the current's error against the target) and the field's `moments`. `trace`, per step: `dE`, `dE_ls`, `F` (the force residual), `resid`, `dt`, `dt_star`, `cfl` (`dt cfl` cells crossed by the step), `substeps`, `div`, `cos`, `newton_it`, `res_it`, `res_moved`, with a pressure or current target also `dW`, without stellarator symmetry also `F_h` (the norm of the harmonic part of the force, a part of `F`) and `c_h` (the norm of the step's harmonic displacement). `qoi`, per chunk: `it`, `wall`, `E`, `F`, `resid`, `helicity`, `JoverB`, `JB`, `beta_vol`. `summary`: the stopping reason and `best_step`, with a target also `W_removed`, with `--pressure.target` `beta`, `pressure_error` and `pressure_constant` at the end, with `--current.target` `current_error`, with `--iota.target` `iota_r_max`. `iota`: the target, the centres `x` of the moments, `rebuilds` before every chunk at which the loop steps (`r`, the weak residual, `r_max`, `r_rms`, `mu` and `lam`, the field's moments and the new target, `delta`, `r_eq`, `minres`), `held` (the boundaries at which the target was held after a move), `end` (the weak residual at the end) and `seed_step`. `mesh`, with `--mesh.align`: `start` (the rounds of the first move), `moves` and `checks` (the chunk boundaries with and without a move), `chunks` and `builds` (the preconditioners). |
| `checkpoints/state_<step>.h5` | the descent state at step 0 and after every chunk (`mrx.relaxation.loop.write_checkpoint`): every leaf of `State` as a dataset (`B_n`, `warm.J`, ...), the discretisation as attributes |
| `checkpoints/best.h5` | the field of lowest residual |
| `checkpoints/moved_<step>.h5` | with `--mesh.align`, the state right after a move of the pole, on the moved map |

Both are rewritten at every chunk, so a run cut off by its time limit
leaves its trace and its last state. A healthy ideal run has `dE < 0` at
every step, `dE` close to `dE_ls` (equal on a step of one substep), `helicity` constant up to
grid-scale reconnection and `div` at round-off.

The field is in the units of the equilibrium file, tesla for every reader,
and the norms integrate over one field period. So `E` is in T^2 m^3,
`F` in T^2 m^1/2, `helicity` in T^2 m^4 and the pressure in T^2. The
gradient-descent `dt` scales as `|B|^-2`. `resid`, `beta_vol`, `cos`,
`dt/dt*`, the CFL number, the cells crossed and the printed `(E_0-E)/E_0` and `dH/|H_0|` are
ratios. A checkpoint written before the field was kept in the file's units
holds it normalised to `||B||_M = 1`. A restart from it continues the same
run in those units, but it must not be paired with a field in tesla, for
example as `--drive.reference` of a run from the file.

To evaluate a checkpoint's field:

```python
from mrx.differential_forms import DiscreteFunction, Pushforward
from mrx.relaxation.loop import checkpoint_attrs, read_field
from mrx.relaxation.mesh_motion import install_checkpoint_map

path = "<run>/checkpoints/best.h5"
a = checkpoint_attrs(path)
seq = build_sequence("data/wout_li383_1.4m.nc", a["ns"], a["p"], knots=a["knots"], symmetry=a["symmetry"],
                     graded=a["graded"], map_theta=a["map_theta"])
install_checkpoint_map(seq, path)     # the map of a run with a moving mesh, if the checkpoint holds one
B, step = read_field(path, seq)
odd = seq.odd                     # B is odd under the stellarator symmetry
B_phys = Pushforward(DiscreteFunction(B, odd.basis_2, odd.E(2)), seq.map, 2)
```

## Pressure

MRX has two pressures: the strong `p` of the descent (`compute_force`, on
request) and the weak `p_w`, zero on the wall, from which `beta_vol` is
computed at every chunk ([Relaxation](concepts/relaxation.md)).

## Poincare sections

Two scripts split the work by cost:

```bash
python -u scripts/poincare_trace.py --geometry data/wout_li383_1.4m.nc \
    <run>/checkpoints/state_000000.h5 <run>/checkpoints/best.h5   # -> <run>/trace.npz
python scripts/poincare_plot.py <run>                            # -> <run>/poincare/
```

`poincare_trace.py` (a GPU job) traces every checkpoint it is given in one
batch, evaluates the weak pressure at every crossing and writes one
`trace.npz`. `poincare_plot.py` (plain matplotlib) renders it,
every field and plane on one iota and one pressure scale. It is the only
thing to rerun when the figure changes. A movie is the same two calls on
every checkpoint of a run, with `--window`, `--iota-lim` and `--p-lim`
holding the axes fixed (their values separated by spaces, for example
`--iota-lim 0.3 0.6`).

In code, `poincare(seq, B, lines=160, periods=400, planes=5)` seeds the
lines from the axis to the edge, measures iota per line, flags the chaotic
ones and cuts the trajectories at the planes (five over half a period for a
stellarator-symmetric map). The returned `drift` (step h against h/2)
justifies the step count. `poincare_many` traces several fields at once.
A step costs about the same for one line as for ten thousand on a GPU, so
eight fields take about as long as one.

The sections are the planes $\phi = $ const of the physical toroidal angle
by default, in cylindrical `(R, Z)`, as people expect them. On the default
mesh, whose sections are normal to the axis, and on a G-frame state every
crossing is found by Newton's method on a partial step, and each line must
cross each plane once per period. `sections="zeta"` (`--sections zeta`)
cuts at the map's own planes `zeta = const` instead. They cost nothing
extra, and their coordinates are `(X1, X2)` in the plane of the frame. On a
file's map with `--geometry.zeta phi` the two are the same planes.

`poincare` also takes a free 2-form (`seq.odd.free`), a field with a flux
through the wall such as that of a free-boundary run. Its lines can leave
the domain. A line that leaves is cut there: its later crossings are NaN.
`reseed` traces further rounds of lines on the radii of the lost ones, and
`padding=(n, periods, r_min)` adds `n` short lines over the edge, traced
both ways, so that a region whose lines leave within a few periods is still
covered.
