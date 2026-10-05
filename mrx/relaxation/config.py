"""The settings of a relaxation run, used by ``scripts/relax.py`` and the tutorials.

:class:`RelaxConfig` holds every setting of a run, grouped into small frozen dataclasses (geometry, seeding,
descent method, Newton solve, step budget, resistive drive, output). The defaults on the classes are the
production values, and invalid combinations raise a ``ValueError`` when the config is created. ``relax.py`` turns
the same classes into its command line with ``tyro.cli``. The docstring under each field is its help text, and a
field ``x`` of the group ``g`` is the option ``--g.x`` (underscores written as hyphens). :attr:`RelaxConfig.params`
is the flat record stored as ``params`` in ``relax.json``, and :meth:`RelaxConfig.from_params` rebuilds the config
from it. :meth:`RelaxConfig.stepper` and :meth:`RelaxConfig.relax_kwargs` give what
:func:`~mrx.relaxation.loop.relax` needs.
"""
from __future__ import annotations

import os
import typing
import warnings
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import Enum, StrEnum
from typing import Annotated, Optional

import tyro

from mrx.relaxation.newton import NEWTON_MASS_TOL, NEWTON_MAXITER, NEWTON_PENALTY, NEWTON_TOL


class Symmetry(StrEnum):
    """The symmetry the geometry has (:data:`mrx.geometry.SYMMETRIES`)."""
    STELLARATOR = "stellarator"     # nfp field periods and stellarator symmetry. Integrates half a period
    FIELD_PERIOD = "field-period"   # nfp field periods only
    NONE = "none"                   # no symmetry: zeta in [0, 1] covers the whole torus and nfp = 1


class Precision(StrEnum):
    """The floating-point precision of a run."""
    FLOAT32 = "float32"             # float32 throughout (solve tolerance 1e-5). Fastest, can stall on shaped maps
    MIXED = "mixed"                 # float32 fields and solves, refined against a float64 residual (tol 1e-8). The default
    FLOAT64 = "float64"             # float64 throughout (tol 1e-10)


class Method(StrEnum):
    """How the relaxation step is computed."""
    NEWTON = "newton"               # Newton steps with the energy Hessian (mrx.relaxation.newton)
    GRADIENT = "gradient"           # gradient descent on the smoothed force


@dataclass(frozen=True)
class Geometry:
    """Geometry, initial condition and discretisation."""
    path: str = field(metadata=dict(record="geometry"))
    """The file that sets both the geometry and the initial field. A VMEC wout (.nc), GVEC state (.dat) or DESC output (.h5) gives the map and the equilibrium's own magnetic field."""
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
    precision: Precision = Precision.MIXED
    """float32 or float64 use that precision everywhere. mixed keeps fields and solves in float32 and refines them against a float64 residual."""
    solve_tol: Optional[float] = None
    """The relative residual tolerance of every linear solve. Unset, it follows the precision: float32 1e-5, mixed 1e-8, float64 1e-10."""
    solve_maxiter: int = 2000
    """The maximum number of iterations of every linear solve."""
    max_batch: int = 0
    """The number of cells evaluated at once in the quadrature loops. 0 evaluates all of them at once. Set a bound at high resolution to limit memory."""

    def __post_init__(self):
        if self.max_batch < 0:
            raise ValueError("--geometry.max-batch must be non-negative (0 is one vmap over all points)")
        if not os.path.isfile(self.path):
            raise ValueError(f"--geometry.path {self.path!r} is not a file (a .nc, .dat or .h5)")

    @property
    def knots(self):
        return [self.knots_r, self.knots_theta, self.knots_zeta]

    def build(self):
        """Build the de Rham sequence of this geometry and return ``(seq, ops)``, see
        :func:`mrx.geometry.build_sequence`."""
        from mrx.geometry import build_sequence
        return build_sequence(self.path, self.resolution, self.spline_degree, self.solve_maxiter,
                              tol=self.solve_tol, knots=self.knots, symmetry=str(self.symmetry))


@dataclass(frozen=True)
class Seed:
    """Island seeding of the start field (see mrx.relaxation.seeding)."""
    seed: Annotated[bool, tyro.conf.arg(name="seed", prefix_name=False)] = field(
        default=False, metadata=dict(record="seed"))
    """Add island seeds to the start field (the initial field or the --output.restart checkpoint)."""
    iotas: Optional[tuple[float, ...]] = None
    """The rotational transforms nfp n / m of the island chains to seed. Unset, every resonance in the iota range is seeded."""
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
    """Newton solve (with --descent.method newton, see mrx.relaxation.newton)."""
    penalty: float = NEWTON_PENALTY
    """The weight of the penalty on flows along the field, in units of the field's strain along the field."""
    tol: float = NEWTON_TOL
    """The relative tolerance of each Newton solve."""
    maxiter: int = NEWTON_MAXITER
    """The maximum number of MINRES iterations of each Newton solve."""
    mass_tol: Optional[float] = NEWTON_MASS_TOL
    """The accuracy of the Chebyshev polynomial that replaces the 1-form mass solves inside the Hessian (mrx.relaxation.newton.MassChebyshev). Its degree follows from this and the spectral interval of the mass atom. None keeps the mass solves, which cost about three times as much."""


@dataclass(frozen=True)
class Budget:
    """Step budget and stopping."""
    steps: Optional[int] = None
    """The maximum number of steps. Unset, it is 100 for Newton and 2000 for gradient descent."""
    chunk: Optional[int] = None
    """The number of steps per chunk. Diagnostics, checkpoints and the stopping test run once per chunk, and --budget.steps must be a multiple of it. Unset, it is 10 for Newton and 200 for gradient descent."""
    floor_tol: float = 1e-10
    """Stop when the mean squared normalised force residual over the last chunk is below this."""


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
class Output:
    """Output directory and restart."""
    out: Optional[str] = None
    """The directory the run writes to. Unset, it is outputs/relax/<date>/<time>."""
    restart: Optional[str] = None
    """Continue from a checkpoint file checkpoints/state_<step>.h5 written with the same geometry, resolution, spline degree and precision."""


@tyro.conf.configure(tyro.conf.EnumChoicesFromValues)
@dataclass(frozen=True)
class RelaxConfig:
    """All settings of a relaxation run, one group per field in the order of the command line.

    Only ``geometry`` has no default. Unset ``steps`` and ``chunk`` are filled in from the method when the config
    is created. The ``record_prefix`` of a group is put before its field names in the flat record ``params``.
    """
    geometry: Geometry
    seed: Seed = field(default=Seed(), metadata=dict(record_prefix="seed_"))
    descent: Descent = Descent()
    newton: Newton = field(default=Newton(), metadata=dict(record_prefix="newton_"))
    budget: Budget = Budget()
    drive: Drive = field(default=Drive(), metadata=dict(record_prefix="drive_"))
    output: Output = Output()

    def __post_init__(self):
        d, b = self.descent, self.budget
        # the budget's defaults depend on the method, so they are filled in here
        steps = b.steps if b.steps is not None else (100 if d.newton else 2000)
        chunk = b.chunk if b.chunk is not None else (10 if d.newton else 200)
        object.__setattr__(self, "budget", replace(b, steps=steps, chunk=chunk))
        if chunk < 1 or steps % chunk:
            raise ValueError("--budget.steps must be a positive multiple of --budget.chunk")

    def stepper(self, seq):
        """Return the :class:`~mrx.relaxation.loop.TimeStepper` for this configuration on the sequence ``seq``."""
        from mrx.relaxation.loop import TimeStepper, radial_cell_sq
        n = self.newton
        return TimeStepper(seq=seq, newton=self.descent.newton, newton_penalty=n.penalty, newton_tol=n.tol,
                           newton_maxiter=n.maxiter, newton_mass_tol=n.mass_tol,
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

    @classmethod
    def from_params(cls, params: dict) -> "RelaxConfig":
        """Rebuild a configuration from the ``params`` of a ``relax.json``. Unknown keys are ignored."""
        return unflatten(cls, params)


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


def unflatten(cls, params: dict, prefix=""):
    """Rebuild a ``cls`` from a record written by :func:`flatten`. Missing keys keep their defaults, and keys that
    are not fields are ignored."""
    hints = typing.get_type_hints(cls)
    kw = {}
    for f in fields(cls):
        t = hints[f.name]
        if is_dataclass(t):
            kw[f.name] = unflatten(t, params, f.metadata.get("record_prefix", ""))
            continue
        key = f.metadata.get("record", prefix + f.name)
        if key in params:
            v = params[key]
            kw[f.name] = tuple(v) if isinstance(v, list) else (t(v) if isinstance(t, type) and issubclass(t, Enum) else v)
    return cls(**kw)
