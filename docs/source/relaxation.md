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
`phi`, the default, the surfaces $\zeta = $ const are the planes
$\phi = $ const of the cylindrical angle, as in the files. With `harmonic`
they are the level sets of the potential of the vacuum field of the domain,
so that $\nabla \zeta$ is the harmonic 1-form $h$, the direction of the
vacuum field, fixed by the boundary alone. The planes are not: on li383, W7-X
and HSX $\|\nabla \zeta - h\| / \|h\|$ is 0.29, 0.35 and 0.65 on the
planes. The map gets there by moves of the same kind as the alignment with
the flux surfaces (see Mesh motion below), with a different misfit.

The misfit is a scalar $q$ on the mesh. One Neumann solve for the Laplacian
on the map gives the harmonic angle $\zeta + q$
(`mrx.nullspace.harmonic_angle`), so $q$ says how far the logical $\zeta$
is from the harmonic angle at each point. The mesh is aligned in $\zeta$
when $q = 0$, as it is aligned radially when the flux label is a function
of $r$. A move slides every point along its line of constant $r$ and
$\theta$, from $\zeta$ to the point the map has at
$\zeta - q / (1 + \partial_\zeta q)$
(`mrx.relaxation.mesh_motion.harmonic_displacement`). The surfaces
$r = $ const and the poloidal angle stay, and the wall slides along
itself. The change of the map is projected onto its spline space and halved
if a cell would fold, as in a flux move. On the moved map $q$ is solved
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
computed there. On one H200 at the resolutions of the mesh motion below the
setup takes 35 to 55 s longer than with `phi`, which takes 240 to 285 s,
and the relaxation is the same (2026-11-19). The harmonic choice
needs a VMEC input namelist, whose start is the vacuum field. A file's field
lives on the file's sections.

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
| pressure | `--pressure.target`, `--pressure.scale`, `--pressure.beta`, `--pressure.plateau-margin`, `--pressure.anisotropy`, `--pressure.label-tol`, `--pressure.smoothing`, `--pressure.cells` |
| current | `--current.target`, `--current.scale` |
| mesh | `--mesh.align`, `--mesh.axis-only`, `--mesh.tol`, `--mesh.moves`, `--mesh.start`, `--mesh.every` |
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
cubic splines in $s$ on `--pressure.cells` cells (16 by default) whose
breakpoints are uniform in $\sqrt{s}$. It does not refer to the mesh, which
has only to resolve it. A spline in $s$ is smooth in space next to the
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

The target is the geometry file's pressure (`--pressure.target file`), the
pressure profile of a VMEC input namelist (`--pressure.target input.<name>`) or
measured pressure in pascal from a JSON file, on flux surfaces (`{"rho": [...], "p_pa": [...]}`) or at points in space
(`{"points": [[R, Z, phi_deg], ...], "p_pa": [...]}`). Data at points are
refitted to a profile of their current labels at every rebuild, so they follow
the surfaces as they move. The target is $\mu_0 p$ in T^2, the units of the
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
wall. And a pressure that falls across an island chain cannot be met, since
the two branches of its separatrix are one surface. With `--seed` the target
is made constant across each seeded chain, over $\sqrt{s} \in x \pm m\, w/2$
from the seeding's estimate of its resonant surface $x = \sqrt{s}$ and width $w$
(`--pressure.plateau-margin` $m$).

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
({func}`mrx.relaxation.twist.iota_slips`):

1. the poloidal flux $\psi(s)$ inside the flux label's level sets is measured
   and compared with the target's, $\psi^*(s) = \Phi/n_{fp} \int_0^s \iota^*$.
   It is the volume integral of $B \cdot \nabla\theta_a$ over the region inside
   each level set, with $\theta_a$ an angle about the label's own axis, so it
   depends on $s$ alone, wherever the magnetic axis sits on the mesh,
2. the slips that close the miss with the rest of the field frozen are a
   least-squares fit to it, in the mean transform inside each surface,
3. one Newton solve with the slips held there lets the ideal part and the
   shells follow, and the current the field then carries is the current
   target of the chunk.

The current's response to a slip with the ideal part relaxed is 0.6 to 0.9
times the slip block's, so the slip block alone would overshoot and the
update would oscillate. Nothing is carried from one chunk to the next, so a
restart recomputes the target from the field. With `--seed` the seed is applied at the first chunk boundary at which
the transform misses its target by less than 1 % of its largest value, since
the seed resonates with the field's own transform. With `--mesh.align` the
loop starts at the first chunk after the mesh is aligned or out of moves:
its first slips sit next to the magnetic axis, which a mesh resolves only
once its axis is there. The `iota` record has,
before every chunk, the transform's miss, the slips asked for and the current
target.

On li383 (48,64,24), graded, with 0.95 times the file's transform from the
file's field, the label's transform misses by 3.3e-2 at the start, 7.9e-4
after 10 Newton steps and 4.1e-4 after 200, the floor of the label's fit. `dW`
is negative on every step and the force residual reaches 1.4e-10 after 200
steps. A step costs what a step without targets costs, the update 1.6 s per
chunk, about one Newton step (one H200, 2026-11-18).

## Mesh motion

A mesh filled from the boundary alone (`--geometry.path input.<name>`) is
not aligned with the flux surfaces of the field, and its axis is where the
fill's guess puts it, not on the magnetic axis. The relaxation
converges more slowly on it. With `--mesh.align` the mesh moves towards the
flux surfaces between chunks (`mrx.relaxation.mesh_motion`). A move is a
property of the mesh only. The physical field stays, and the pressure and
current targets, functions of the flux label, do not see it.

Every point moves within its surface $\zeta = $ const and the wall stays, so
a harmonic $\zeta$ stays harmonic: $\|\nabla \zeta - h\| / \|h\|$ is
recorded after every move. A
move first puts the mesh axis on the magnetic axis, the maximum of the flux
label's temperature in each plane, by the shift with which the fill places
its axis, $d(\zeta)(1 - r^2)$. Then it moves the points so that the label
$s$ comes closer to its volume mean $S(r)$ on each logical surface, by a
velocity from one weighted Helmholtz solve per component of the plane,
smoothed over two radial cells. Both parts are halved until no cell loses
more than half its volume. With `--mesh.axis-only` a move is the axis step
alone, and the mesh counts as aligned once its axis is on the magnetic axis.

The field keeps its coefficients, which moves the physical field with the
mesh and keeps every flux through every face of the mesh. The drag is then
undone by the relaxation's ideal update with the mesh's velocity in place of
the flow's, on the maps between the old and the new one, in SSP-RK3 substeps
that cross at most half a cell, the relaxation's own substep, the cheapest
explicit method that is stable for this advection, which nothing corrects
afterwards
(`mrx.relaxation.mesh_motion.undrag`). $\nabla \cdot B = 0$, $B \cdot n = 0$ and
the toroidal flux hold exactly. The fluxes inside the other surfaces and the
topology are kept to the accuracy of the transport. One move changes the
physical field by the transport's spatial error, 2.6e-3 for the 9.9 cm first
move on W7-X graded (24,48,28), in 27 substeps and 2 s on one H200. Twelve
moves there and back return the field to 2.5e-5, and a seeded island chain
on li383 keeps its width (2026-11-18). The new map is data of the sequence:
the compiled relaxation is reused and the preconditioners of the start map
are kept. Once the mesh is aligned or out of moves the preconditioners of
the final map are built, in about a minute of host assembly at the sizes
above. The kept ones precondition the moved operators with the same Newton
and label counts, but the mass solves inside every step take more
iterations, which made the steps on the moved meshes of li383, W7-X and HSX
12 to 26 % slower (2026-11-19).

A move happens before the shells and the slips are rebuilt, from step
`--mesh.start` on and every `--mesh.every` steps, at most `--mesh.moves`
times. The motion stops for good once the volume rms of $s - S(r)$ is below
`--mesh.tol` and the label's maximum is within a quarter of the first radial
cell of the mesh axis. Every move kicks the force residual up by the error
of the carried field, so a few moves while the field is far from its
equilibrium cost little and moves late in a run delay it.

    python -u scripts/relax.py --geometry.path input.li383 --geometry.graded --geometry.resolution 48 64 24 \
        --budget.steps 400 --pressure.target input.li383 --mesh.align

From the vacuum on li383's fill with the namelist's pressure, as above, four
moves at steps 20 to 80 align the mesh. The rms of $s - S(r)$ falls from
5.0e-2 to 8.5e-3, and the mesh axis, 5.5 radial cells off at first, ends on
the magnetic axis. The squared force residual stays below 1e-8 from step 90
on, against step 257 without the motion. A move costs 5 to 8 s after the
first, which compiles (106 s). The steps on the moved mesh cost more, 10 s per
chunk of ten against 6 s, so both runs reach a residual of 1e-8 in about ten
minutes with the build of the operators (one H200, float64, 2026-11-19).

    python -u scripts/relax.py --geometry.path input.w7x --geometry.graded --geometry.resolution 24 48 28 \
        --budget.steps 200 --pressure.target input.w7x --current.target input.w7x --mesh.align

From the fill and the vacuum with both targets of one namelist, the squared
force residual stays below 1e-8 from step 63 on W7-X graded (24,48,28) (14
on the nested solver's mesh), from step 64 on li383 graded (48,64,24) against
239 without the motion (40), and from step 24 on HSX graded (32,64,36) (8),
after three, three and one moves (one H200, 2026-11-19). A tighter `--mesh.tol`
costs steps, since every move after the residual is low delays it.

The `mesh` record of `relax.json` holds every move: the step, the alignment
error before and after it, the axis offset in first radial cells, the
largest move, the range of the change of $\det D\Phi$, the change of the
physical field at fixed physical points and of the energy, the substeps of
the transport and the label's
CG counts, and `preconditioners`, the step and the wall time of the build on
the final map. Every checkpoint of such a run holds its map, which a restart and
`scripts/poincare_trace.py` install. It also holds the state of the schedule
(the moves made, the step of the last move and the step the mesh was aligned
at), so a restart moves on the steps the run would have moved on. `best.h5` is
the best field since the last move.

## Stopping

The squared normalised force residual
$\mathrm{resid} = \|F\|_M^2 / \|\nabla(B^2/2)\|^2$ is recorded at every
step. The run stops when its mean over the last chunk falls below
`--budget.floor-tol`, or when `--budget.steps` is spent. With
`--pressure.target` or `--current.target` the run takes all of
`--budget.steps`: the residual does not see the shells or the slips, so its
floor does not end a run that still approaches its target. The residual is not monotone.
Judge a run by the floor it settles at, and take `checkpoints/best.h5` when
it went past it. A job's time limit is no stop: the checkpoint of every
chunk restarts it (`--output.restart`).

## Output

A run writes to `--output.out`, by default `relax/<date>/<time>/` in the
runs root, which is `$MRX_DATA/runs/mrx`, or `outputs/` in the working
directory without `MRX_DATA` ([Running on a cluster](cluster.md)).

| file | content |
|---|---|
| `relax.json` | `params`: the configuration, flat, plus `geometry_path`, `ic`, `resolution`, `nfp`, `h_r_sq`, `start_step`. `ic`: the initial field's numbers. `iota`, `seed`, `drive`: when used. `pressure`: the target, its plateaus and `rebuilds`, before every chunk and at the end of the run: `step`, `label_it`, `error` and `constant` (the pressure's error against the target and its constant $C$ over the target's maximum), `beta` (the target's), `scale` (its factor, with `--pressure.beta`) and, before every chunk, `shell_it` (the signed counts of the shells' Poisson solves). `current`: the target, the centres `x` (in $\sqrt{s}$) of the splines of the profile space and the target's moments `target_moments` ($\mu_0 I$ in T m), and `rebuilds`, before every chunk and at the end: `step`, `label_it`, `error` (the current's error against the target) and the field's `moments`. `trace`, per step: `dE`, `dE_ls`, `F` (the force residual), `resid`, `dt`, `dt_star`, `cfl` (`dt cfl` cells crossed by the step), `substeps`, `div`, `cos`, `newton_it`, `res_it`, `res_moved`, with a pressure or current target also `dW`. `qoi`, per chunk: `it`, `wall`, `E`, `F`, `resid`, `helicity`, `JoverB`, `JB`, `beta_vol`. `summary`: the stopping reason and `best_step`, with a target also `W_removed`, with `--pressure.target` `beta`, `pressure_error` and `pressure_constant` at the end, with `--current.target` `current_error`. |
| `checkpoints/state_<step>.h5` | the descent state at step 0 and after every chunk (`mrx.relaxation.loop.write_checkpoint`): every leaf of `State` as a dataset (`B_n`, `warm.J`, ...), the discretisation as attributes |
| `checkpoints/best.h5` | the field of lowest residual |

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

path = "<run>/checkpoints/best.h5"
a = checkpoint_attrs(path)
seq = build_sequence("data/wout_li383_1.4m.nc", a["ns"], a["p"], knots=a["knots"], symmetry=a["symmetry"],
                     graded=a["graded"], map_theta=a["map_theta"])
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

The sections are the map's planes `zeta = const` by default. They cost
nothing extra, and on a G-frame state their coordinates are `(X1, X2)` in
the plane of the frame. `sections="phi"` (`--sections phi`) cuts at the
planes of the physical toroidal angle instead, in cylindrical `(R, Z)`. On
VMEC and DESC maps the two are the same planes. On a G-frame map every
crossing is then found by Newton's method on a partial step, and each line
must cross each plane once per period.

`poincare` also takes a free 2-form (`seq.odd.free`), a field with a flux
through the wall such as that of a free-boundary run. Its lines can leave
the domain. A line that leaves is cut there: its later crossings are NaN.
`reseed` traces further rounds of lines on the radii of the lost ones, and
`padding=(n, periods, r_min)` adds `n` short lines over the edge, traced
both ways, so that a region whose lines leave within a few periods is still
covered.
