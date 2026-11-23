"""The settings of a relaxation run, used by ``scripts/relax.py`` and the tutorials.

:class:`RelaxConfig` holds every setting of a run, grouped into small frozen dataclasses (geometry, the target transform, seeding,
descent method, Newton solve, step budget, resistive drive, prescribed pressure, prescribed current, mesh motion, output). The defaults on the classes are the
production values, and invalid combinations raise a ``ValueError`` when the config is created. ``relax.py`` turns
the same classes into its command line with ``tyro.cli``. The docstring under each field is its help text, and a
field ``x`` of the group ``g`` is the option ``--g.x`` (underscores written as hyphens). :attr:`RelaxConfig.params`
is the flat record stored as ``params`` in ``relax.json``. :meth:`RelaxConfig.stepper` and :meth:`RelaxConfig.relax_kwargs` give what
:func:`~mrx.relaxation.loop.relax` needs.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import Enum, StrEnum
from typing import Annotated, Literal, Optional

import tyro

from mrx.flux_label import ANISOTROPY
from mrx.paths import data_file
from mrx.relaxation.newton import NEWTON_MASS_SOLVE_TOL, NEWTON_MAXITER, NEWTON_PENALTY, NEWTON_TOL
from mrx.relaxation.shells import LABEL_TOL


class Symmetry(StrEnum):
    """The symmetry the geometry has (:data:`mrx.geometry.SYMMETRIES`)."""
    STELLARATOR = "stellarator"     # nfp field periods and stellarator symmetry. Integrates half a period
    FIELD_PERIOD = "field-period"   # nfp field periods only
    NONE = "none"                   # no symmetry: zeta in [0, 1] covers the whole torus and nfp = 1


class Method(StrEnum):
    """How the relaxation step is computed."""
    NEWTON = "newton"               # Newton steps with the energy Hessian (mrx.relaxation.newton)
    GRADIENT = "gradient"           # gradient descent on the smoothed force


class CurrentStart(StrEnum):
    """The label of the start that adds the target current along the start field (--current.start)."""
    LABEL = "label"                 # the flux label of the start field (mrx.flux_label)
    RADIUS = "radius"               # the toroidal flux inside the logical surfaces r = const (mrx.flux_label.radius_flux)


@dataclass(frozen=True)
class Geometry:
    """Geometry, initial condition and discretisation."""
    path: str = field(metadata=dict(record="geometry"))
    """The file that sets both the geometry and the initial field. A VMEC wout (.nc), GVEC state (.dat) or DESC output (.h5) gives the map and the equilibrium's own magnetic field. A file that is not at this path is looked up by name in $MRX_DATA/data."""
    symmetry: Symmetry = Symmetry.STELLARATOR
    """The symmetry of the geometry: nfp field periods with stellarator symmetry, field periods only, or none."""
    resolution: tuple[int, int, int] = (32, 64, 64)
    """The number of splines in (r, theta, zeta), used for the fields and the map. An axis given knots takes its number from them."""
    spline_degree: int = 2
    """The spline degree p."""
    knots_r: Optional[tuple[float, ...]] = None
    """The cell boundaries in r, increasing from 0 to 1, instead of a uniform grid."""
    knots_theta: Optional[tuple[float, ...]] = None
    """The cell boundaries in theta, increasing from 0 to 1, instead of a uniform grid."""
    knots_zeta: Optional[tuple[float, ...]] = None
    """The cell boundaries in zeta, increasing from 0 to 1, instead of a uniform grid."""
    graded: bool = False
    """The graded polar space: fewer theta cells on the rings near the axis (see mrx.geometry.build_sequence)."""
    map_theta: Optional[int] = None
    """The theta cells of the map on every ring, n_theta / 2^l. Unset, n_theta (see mrx.geometry.build_sequence)."""
    precision: Literal["float64", "float32"] = "float64"
    """float64, or float32 for a device without float64. In float32 every linear solve is refined in double-single arithmetic."""
    solve_tol: Optional[float] = None
    """The relative residual tolerance of every linear solve. Unset, it is 1e-8."""
    solve_maxiter: int = 2000
    """The maximum number of iterations of every linear solve."""
    zeta: Literal["axis-normal", "phi", "harmonic"] = "axis-normal"
    """The toroidal coordinate of the first mesh. axis-normal: the surfaces zeta = const are the planes normal to the axis of the file's map, each point slid along its flux surface and the file's field carried (mrx.relaxation.mesh_motion.axis_normal_map). phi: the file's own sections (the planes phi = const of a VMEC or DESC file). harmonic: the angle of the vacuum field (the level sets of its potential, the map moved there from the file's sections by mrx.relaxation.mesh_motion.harmonic_build). harmonic needs a VMEC input namelist, whose start is the vacuum field."""
    max_batch: int = 262144
    """The number of points evaluated at once in the quadrature loops. 0 evaluates all of them at once. Lower it when one of these loops runs out of device memory."""

    def __post_init__(self):
        if self.max_batch < 0:
            raise ValueError("--geometry.max-batch must be non-negative (0 is one vmap over all points)")
        if self.zeta == "harmonic" and not os.path.basename(self.path).startswith("input."):
            raise ValueError("--geometry.zeta harmonic re-sections a VMEC input namelist's fill: a file's field lives "
                             "on the file's sections")
        if not os.path.isfile(data_file(self.path)):
            raise ValueError(f"--geometry.path {self.path!r} is not a file (a .nc, .dat or .h5), neither there nor by "
                             "name in $MRX_DATA/data")

    @property
    def knots(self):
        return [self.knots_r, self.knots_theta, self.knots_zeta]

    def build(self):
        """Build the de Rham sequence of this geometry and return it, see :func:`mrx.geometry.build_sequence`, with its
        map re-sectioned as ``zeta`` says.
        ``max_batch`` becomes the batch of the loops over quadrature points (:data:`mrx.MAP_BATCH_SIZE_INNER`)."""
        import mrx
        from mrx.geometry import build_sequence
        mrx.MAP_BATCH_SIZE_INNER = self.max_batch
        return build_sequence(self.path, self.resolution, self.spline_degree, self.solve_maxiter,
                              tol=self.solve_tol, knots=self.knots, symmetry=str(self.symmetry), graded=self.graded,
                              map_theta=self.map_theta, zeta=self.zeta)


@dataclass(frozen=True)
class Iota:
    """Target rotational transform: an outer loop updates the target of the prescribed current by a Newton step on the transform's weak residual (mrx.relaxation.twist.iota_newton) before every chunk, from the transform the field has on its flux label. Its flux label is the one of --pressure.anisotropy and --pressure.label-tol."""
    target: Optional[str] = None
    """A wout or a VMEC input namelist with NCURR = 0 whose transform the relaxation ends at. A namelist with NCURR = 1 prescribes the current, see --current.target. Unset, the run keeps the start's poloidal fluxes."""

    def __bool__(self):
        return self.target is not None


@dataclass(frozen=True)
class Seed:
    """Island seeding of the start field, at the resonances of its own transform (see mrx.relaxation.seeding)."""
    seed: Annotated[bool, tyro.conf.arg(name="seed", prefix_name=False)] = field(
        default=False, metadata=dict(record="seed"))
    """Add island seeds to the start field (the initial field or the --output.restart checkpoint)."""
    iotas: Optional[tuple[float, ...]] = None
    """The rotational transforms nfp n / m of the island chains to seed. Unset, every resonance in the iota range of the field, measured on its flux label, is seeded."""
    amplitudes: Optional[tuple[float, ...]] = None
    """One signed amplitude per --seed.iotas value: the resonant normal field |dB^r| / |B^zeta| at the resonant surface. Unset, the amplitudes minimise the energy."""
    scale: float = 1.0
    """The factor applied to the whole seed perturbation."""

    def __post_init__(self):
        if self.amplitudes is not None and self.iotas is None:
            warnings.warn("--seed.amplitudes without --seed.iotas is ignored: the energy criterion sets them",
                          stacklevel=2)
            object.__setattr__(self, "amplitudes", None)
        if self.amplitudes is not None and len(self.amplitudes) != len(self.iotas):
            raise ValueError("--seed.amplitudes needs one value per --seed.iotas")

    def __bool__(self):
        return self.seed


@dataclass(frozen=True)
class Descent:
    """Descent method."""
    method: Method = Method.NEWTON
    """newton takes Newton steps with the energy Hessian, gradient takes gradient steps along the smoothed force."""

    @property
    def newton(self):
        return self.method == Method.NEWTON


@dataclass(frozen=True)
class Newton:
    """Gauss-Newton solve (with --descent.method newton, see mrx.relaxation.newton)."""
    penalty: float = NEWTON_PENALTY
    """The weight of the penalty on flows along the field, in units of the field's strain along the field."""
    tol: float = NEWTON_TOL
    """The relative tolerance of each Newton solve, in the norm of its preconditioner."""
    maxiter: int = NEWTON_MAXITER
    """The maximum number of MINRES iterations of each Newton solve."""
    mass_solve_tol: float = NEWTON_MASS_SOLVE_TOL
    """The relative tolerance of the two 1-form mass solves inside every apply of the Gauss-Newton operator. Looser than 1e-5 the relaxation stalls."""


@dataclass(frozen=True)
class Budget:
    """Step budget and stopping."""
    steps: Optional[int] = None
    """The maximum number of steps. Unset, it is 100 for Newton and 2000 for gradient descent."""
    chunk: Optional[int] = None
    """The number of steps per chunk. Diagnostics, checkpoints and the stopping test run once per chunk, and --budget.steps must be a multiple of it. Unset, it is 10 for Newton and 200 for gradient descent."""
    floor_tol: float = 1e-10
    """Stop when the mean squared normalised force residual over the last chunk is below this."""
    stages: int = 1
    """Bring the targets (the pressure of --pressure.target, the current of --current.target, the transform of --iota.target) in over this many stages from the start field's own values, one stage per chunk boundary: the targets at stage k are (1 - t) times the start's plus t times the prescribed ones, t = k / stages. 1 puts them on at once."""


@dataclass(frozen=True)
class Drive:
    """Resistive drive: every step also diffuses the current towards that of a reference field."""
    resistivity: float = 0.0
    """The resistivity eta in units of h_r^2 (the squared radial cell size). Each step adds the electric field E = eta (J - J*), with J* the current of the reference field B*. 0 turns the drive off."""
    reference: Optional[str] = None
    """The checkpoint file of the reference field B*, typically a converged equilibrium with nested surfaces. Required with --drive.resistivity."""
    reference_smoothing: float = 0.1
    """Smooth B* by one resistive diffusion step of this size in units of h_r^2, which removes current sheets on its rational surfaces. 0 skips it."""
    chain: Optional[float] = None
    """Add to B* the island seed of the chain at this rotational transform nfp n / m."""
    eps: float = 0.0
    """The signed amplitude of that seed: the resonant normal field |dB^r| / |B^zeta| at the resonant surface."""

    def __post_init__(self):
        if self.resistivity and self.reference is None:
            raise ValueError("--drive.resistivity needs --drive.reference, the checkpoint of B*")
        if self.chain is not None and not self.resistivity:
            raise ValueError("--drive.chain needs --drive.resistivity (the drive is the source's, not the field's)")

    def __bool__(self):
        return bool(self.resistivity)


@dataclass(frozen=True)
class Pressure:
    """Prescribed pressure: the volumes of the flux shells are unknowns of every Newton step (mrx.relaxation.shells), so the relaxation ends at the target pressure instead of keeping the start's."""
    target: Optional[str] = None
    """The pressure to reach: file (the geometry file's pressure) or a VMEC input namelist (its pressure profile). Measured pressure, a JSON file, is fitted by scripts/pressure_fit.py, which takes the flags of this script. Unset, the run keeps the start's pressure."""
    scale: float = 1.0
    """Multiply the target by this factor."""
    beta: Optional[float] = None
    """Instead of --pressure.scale, scale the target's profile to this volume beta, int p* dV / int B^2/2 dV, on the field of every rebuild. It is the beta of the run's pressure in the gauge <p> = <p*>. beta_vol, of the weak pressure that vanishes on the wall, differs from it by the pressure's constant."""
    anisotropy: float = ANISOTROPY
    """The perpendicular conductivity kperp of the flux label the shells and the target are functions of (mrx.flux_label)."""
    label_tol: float = LABEL_TOL
    """The relative tolerance of that label's solve."""
    label_check: bool = True
    """Stop a fresh run that needs the flux label (targets or a seed) when its start field's transform is too small for it: (iota a / R)^2 below 100 kperp (mrx.flux_label.label_condition), the vacuum of an axisymmetric domain for one. --pressure.no-label-check makes it a warning."""
    cells: Optional[int] = None
    """The number of cells of the profile space, cubic splines in s with breakpoints uniform in sqrt(s), shared by the pressure (the shells), the current (the slips), the transform's loop and the fit of scripts/pressure_fit.py (mrx.relaxation.shells.profile_basis). Unset, half the mesh's radial cells (mrx.relaxation.shells.profile_cells)."""

    def __post_init__(self):
        if self.beta is not None and self.scale != 1.0:
            raise ValueError("--pressure.beta and --pressure.scale both set the target's size: give one")

    def __bool__(self):
        return self.target is not None


@dataclass(frozen=True)
class Current:
    """Prescribed current: poloidal flux slips between the flux surfaces are unknowns of every Newton step (mrx.relaxation.twist.Slips), so the relaxation ends at the target net toroidal current inside every surface instead of keeping the start's poloidal fluxes. Its flux label is the one of --pressure.anisotropy and --pressure.label-tol."""
    target: Optional[str] = None
    """The current to reach: file (the current of the geometry file's field on its flux label) or a VMEC input namelist with NCURR = 1 (CURTOR times its PCURR_TYPE profile of AC, or of AC_AUX_S and AC_AUX_F). Unset, the run keeps the start's poloidal fluxes."""
    scale: float = 1.0
    """Multiply the target by this factor."""
    start: Optional[CurrentStart] = None
    """Add the target current to the start field along the field, J = sigma(s) B with sigma = (mu0 I*)'(s) / Phi, by one 1-form Hodge solve (mrx.relaxation.twist.current_start): the start carries the target current and keeps the surfaces of the file's field to first order. The sum is the start field, whose magnetic axis the first move of --mesh.align puts the pole on. The current part of the stages of --budget.stages is then at its target from the first. label takes s from the start field's flux label, radius from the toroidal flux inside the mesh's logical surfaces, for an axisymmetric vacuum, whose flux label is degenerate. With --iota.target the current target is the start's only: the iota loop starts from it and holds it until the field is first at an equilibrium. Unset, the file's field is the start field."""

    def __post_init__(self):
        if self.start is not None and self.target is None:
            raise ValueError("--current.start needs --current.target")

    def __bool__(self):
        return self.target is not None


@dataclass(frozen=True)
class Mesh:
    """Motion of the mesh's axis onto the magnetic axis (mrx.relaxation.mesh_motion). The physical field and the targets stay, only the map moves."""
    align: bool = False
    """Keep the pole of the mesh on the magnetic axis: a run from the vacuum field of a namelist starts on a mesh moved onto the vacuum's axis (mrx.relaxation.mesh_motion.vacuum_start), and at a chunk boundary where the field is at an equilibrium (its squared force residual below --mesh.bar) and its magnetic axis more than --mesh.move-tol first radial cells off the pole, the pole moves onto that axis with the field carried onto the moved map (mrx.relaxation.mesh_motion.axis_move). The moves keep the preconditioners, which are built again on the installed map when the chunk's mean MINRES count of the Newton solve exceeds --mesh.rebuild-factor times that of the first chunk after their last build, or a Newton solve stopped at --newton.maxiter while the field is above the bar."""
    rebuild_factor: float = 1.5
    """The growth of the chunk's mean MINRES count over that of the first chunk after the last build of the preconditioners at which they are built again."""
    bar: float = 1e-8
    """The squared force residual below which the field is at an equilibrium: only there the pole moves. During a ramp the field's axis is not that of the equilibrium it relaxes to."""
    move_tol: float = 0.1
    """The distance of the magnetic axis from the pole, in first radial cells, above which the pole moves at an equilibrium."""

    def __bool__(self):
        return self.align


@dataclass(frozen=True)
class Output:
    """Output directory and restart."""
    out: Optional[str] = None
    """The directory the run writes to. Unset, it is relax/<date>/<time> under $MRX_DATA/runs/mrx, or under outputs without MRX_DATA."""
    restart: Optional[str] = None
    """Continue from a checkpoint file checkpoints/state_<step>.h5 written with the same geometry, resolution and spline degree."""


@tyro.conf.configure(tyro.conf.EnumChoicesFromValues)
@dataclass(frozen=True)
class RelaxConfig:
    """All settings of a relaxation run, one group per field in the order of the command line.

    Only ``geometry`` has no default. Unset ``steps`` and ``chunk`` are filled in from the method when the config
    is created. The ``record_prefix`` of a group is put before its field names in the flat record ``params``.
    """
    geometry: Geometry
    iota: Iota = field(default=Iota(), metadata=dict(record_prefix="iota_"))
    seed: Seed = field(default=Seed(), metadata=dict(record_prefix="seed_"))
    descent: Descent = Descent()
    newton: Newton = field(default=Newton(), metadata=dict(record_prefix="newton_"))
    budget: Budget = Budget()
    drive: Drive = field(default=Drive(), metadata=dict(record_prefix="drive_"))
    pressure: Pressure = field(default=Pressure(), metadata=dict(record_prefix="pressure_"))
    current: Current = field(default=Current(), metadata=dict(record_prefix="current_"))
    mesh: Mesh = field(default=Mesh(), metadata=dict(record_prefix="mesh_"))
    output: Output = Output()

    def __post_init__(self):
        d, b = self.descent, self.budget
        # the budget's defaults depend on the method, so they are filled in here
        steps = b.steps if b.steps is not None else (100 if d.newton else 2000)
        chunk = b.chunk if b.chunk is not None else (10 if d.newton else 200)
        object.__setattr__(self, "budget", replace(b, steps=steps, chunk=chunk))
        if chunk < 1 or steps % chunk:
            raise ValueError("--budget.steps must be a positive multiple of --budget.chunk")
        if self.pressure and (not d.newton or self.drive):
            raise ValueError("--pressure.target needs Newton steps (--descent.method newton) and goes without "
                             "--drive.resistivity")
        if self.current and (not d.newton or self.drive):
            raise ValueError("--current.target needs Newton steps (--descent.method newton) and goes without "
                             "--drive.resistivity")
        if self.iota and (not d.newton or self.drive):
            raise ValueError("--iota.target needs Newton steps (--descent.method newton) and goes without "
                             "--drive.resistivity")
        if self.current and self.iota and not self.current.start:
            raise ValueError("--iota.target and --current.target both set the current: give one, or the current "
                             "target as the start's with --current.start")
        m = self.mesh
        if m and self.drive:
            raise ValueError("--mesh.align goes without --drive.resistivity: the reference field lives on the mesh it "
                             "was written on")

    def stepper(self, seq):
        """Return the :class:`~mrx.relaxation.loop.TimeStepper` for this configuration on the sequence ``seq``."""
        from mrx.relaxation.loop import TimeStepper, radial_cell_sq
        n = self.newton
        return TimeStepper(seq=seq, newton=self.descent.newton, newton_penalty=n.penalty, newton_tol=n.tol,
                           newton_maxiter=n.maxiter, newton_mass_solve_tol=n.mass_solve_tol,
                           resistivity=self.drive.resistivity * radial_cell_sq(seq))

    def relax_kwargs(self):
        """Return the keyword arguments ``steps``, ``chunk`` and ``floor_tol`` for
        :func:`~mrx.relaxation.loop.relax`."""
        b = self.budget
        return dict(steps=b.steps, chunk=b.chunk, floor_tol=b.floor_tol)

    @property
    def params(self) -> dict:
        """This configuration as the flat record stored as ``params`` in ``relax.json``, for example
        ``resolution``, ``spline_degree``, ``newton_tol`` and ``drive_resistivity``. The driver script adds facts
        about the run to it."""
        return flatten(self)



def flatten(cfg, prefix="") -> dict:
    """Return a config of nested dataclasses as a flat dict of JSON values (tuples as lists, enums as their
    values). A field is keyed by its name after the ``record_prefix`` of its group, or by its ``record``
    metadata."""
    out = {}
    for f in fields(cfg):
        v = getattr(cfg, f.name)
        if is_dataclass(v):
            out.update(flatten(v, f.metadata.get("record_prefix", "")))
        else:
            key = f.metadata.get("record", prefix + f.name)
            out[key] = list(v) if isinstance(v, tuple) else (v.value if isinstance(v, Enum) else v)
    return out

