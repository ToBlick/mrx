"""The configuration of a relaxation run, ``scripts/relax.py``'s and the tutorials'.

One frozen dataclass per group, the defaults on the classes, the validation
in ``__post_init__``; :mod:`mrx.cli` makes the command line of it and the
flat ``params`` of ``relax.json``. :meth:`RelaxConfig.stepper` builds the
:class:`~mrx.relaxation.TimeStepper` and :meth:`RelaxConfig.relax_kwargs`
the arguments of :func:`~mrx.relaxation.relax`, so a tutorial and the
command line configure the same objects the same way. The paper's driver
(``scripts/paper_scripts/relax_paper.py``) extends :class:`Descent` with the
options the paper compared and the default script no longer exposes.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Optional

from mrx.hessian import NEWTON_MAXITER, NEWTON_PENALTY, NEWTON_TOL

#: --precision -> (MRX_DTYPE, MRX_RESIDUAL_DTYPE). scripts/relax.py carries the same table: it must set the
#: environment BEFORE mrx is imported (mrx.precision fixes the dtypes at import), so it cannot read this one.
PRECISIONS = {"mixed": ("float32", "float64"), "float32": ("float32", "float32"),
              "float64": ("float64", "float64")}


class Symmetry(StrEnum):
    """What the map satisfies (:data:`mrx.geometry.SYMMETRIES`)."""
    STELLARATOR = "stellarator"     # nfp field periods and stellarator symmetry: half-period quadrature, parity views
    FIELD_PERIOD = "field-period"   # nfp field periods only
    NONE = "none"                   # zeta in [0, 1] is the whole torus, nfp = 1


class Precision(StrEnum):
    FLOAT32 = "float32"             # plain float32 (tol 1e-5): the default, runs without float64
    MIXED = "mixed"                 # float32 fields and solves, float64 residual (tol 1e-8): the paper's runs
    FLOAT64 = "float64"             # plain float64 (tol 1e-10)


class Method(StrEnum):
    NEWTON = "newton"               # Newton-MR on the second variation
    GRADIENT = "gradient"           # gradient descent on the smoothed force


class Scheme(StrEnum):
    EXPLICIT = "explicit"           # forward Euler induction
    MIDPOINT = "midpoint"           # midpoint-implicit induction at the predictor's velocity (Picard)


def current_precision() -> Precision:
    """The precision mrx runs in (a tutorial's configuration records it)."""
    import mrx
    from mrx.precision import RESIDUAL_DTYPE
    return Precision({v: k for k, v in PRECISIONS.items()}[(str(mrx.DTYPE), str(RESIDUAL_DTYPE))])


def _ints(s):
    return tuple(int(v) for v in s.split(","))


def _floats(s):
    return tuple(float(v) for v in s.split(","))


def _knots(s):
    from mrx.geometry import parse_knots
    return parse_knots(s)


@dataclass(frozen=True)
class Geometry:
    """Geometry, initial condition and discretisation."""
    path: str = field(metadata=dict(
        flag="--geometry",
        help="the geometry AND the initial condition: a VMEC wout (.nc) or GVEC state (.dat) gives the map and the "
             "equilibrium's own field B = dA'; an analytic geometry (.json: a map of mrx.mappings with its parameters, "
             "nfp, and the profiles of the logical-grid field) gives the map and that field"))
    symmetry: Symmetry = field(default=Symmetry.STELLARATOR, metadata=dict(
        help="what the map satisfies: nfp field periods and stellarator symmetry, field periods only, or nothing"))
    resolution: tuple[int, int, int] = field(default=(32, 64, 64), metadata=dict(
        parse=_ints, help="spline resolution (r, theta, zeta), also the map's"))
    spline_degree: int = field(default=2, metadata=dict(help="spline degree; p+1 Gauss points per span"))
    knots_r: Optional[tuple] = field(default=None, metadata=dict(
        parse=_knots, help="breakpoints of the r axis, comma-separated from 0 to 1, instead of the uniform grid"))
    knots_theta: Optional[tuple] = field(default=None, metadata=dict(parse=_knots, help="the theta axis's"))
    knots_zeta: Optional[tuple] = field(default=None, metadata=dict(parse=_knots, help="the zeta axis's"))
    precision: Precision = field(default=Precision.FLOAT32, metadata=dict(
        help="float32 / float64: fields, solves and residuals; mixed: float32 fields and solves with a float64 "
             "residual (the paper's runs)"))
    solve_tol: Optional[float] = field(default=None, metadata=dict(
        help="residual tolerance of every solve [the precision's: mixed 1e-8, float32 1e-5, float64 1e-10]"))
    solve_maxiter: int = field(default=2000, metadata=dict(help="iteration budget of every inner solve"))
    max_batch: int = field(default=0, metadata=dict(
        help="cells per batch of the quadrature loops; 0 = all points in one vmap; bound it at high resolution"))

    def __post_init__(self):
        if self.max_batch < 0:
            raise ValueError("--max-batch must be non-negative (0 is one vmap over all points)")
        if not os.path.isfile(self.path):
            raise ValueError(f"--geometry {self.path!r} is not a file (a .nc, .dat or .json)")

    @property
    def knots(self):
        return [self.knots_r, self.knots_theta, self.knots_zeta]

    @property
    def analytic(self):
        return self.path.endswith(".json")

    def build(self):
        """The sequence and its operators, :func:`mrx.geometry.build_sequence` of this group."""
        from mrx.geometry import build_sequence
        return build_sequence(self.path, self.resolution, self.spline_degree, self.solve_maxiter,
                              tol=self.solve_tol, knots=self.knots, symmetry=str(self.symmetry))


@dataclass(frozen=True)
class Seed:
    """Island seeds in the start field by the energy criterion (mrx.seeding; equilibrium files only)."""
    seed: bool = field(default=False, metadata=dict(
        flag="--seed", help="seed the start field (the initial condition or the --restart checkpoint)"))
    iotas: Optional[tuple[float, ...]] = field(default=None, metadata=dict(
        parse=_floats, help="the chains to seed, by their rotational transforms nfp n / m [every resonance in range]"))
    amplitudes: Optional[tuple[float, ...]] = field(default=None, metadata=dict(
        parse=_floats, help="one per iota: the resonant normal field |dB^r| / |B^zeta| at r_mn, signed, instead "
                            "of the energy criterion's"))
    scale: float = field(default=1.0, metadata=dict(help="multiply the added perturbation"))

    def __post_init__(self):
        if self.amplitudes is not None and self.iotas is None:
            warnings.warn("--seed-amplitudes without --seed-iotas is ignored: the energy criterion sets them",
                          stacklevel=2)
            object.__setattr__(self, "amplitudes", None)
        if self.amplitudes is not None and len(self.amplitudes) != len(self.iotas):
            raise ValueError("--seed-amplitudes needs one value per --seed-iotas")

    def __bool__(self):
        return self.seed


@dataclass(frozen=True)
class Descent:
    """The descent direction and the induction step."""
    method: Method = field(default=Method.NEWTON, metadata=dict(
        help="the direction: Newton on the second variation, or gradient descent on the smoothed force"))
    scheme: Scheme = field(default=Scheme.EXPLICIT, metadata=dict(
        help="the induction step: forward Euler, or midpoint-implicit at the predictor's velocity"))

    @property
    def newton(self):
        return self.method == Method.NEWTON

    def stepper_kwargs(self) -> dict:
        """The :class:`~mrx.relaxation.TimeStepper` options of this group (the paper driver extends them)."""
        return dict(newton=self.newton, midpoint=self.scheme == Scheme.MIDPOINT)


@dataclass(frozen=True)
class Newton:
    """Newton (--method newton): u = curl a from curl^T H curl a = curl^T M F by Newton-MR (mrx.hessian)."""
    penalty: float = field(default=NEWTON_PENALTY, metadata=dict(
        help="kappa of the parallel-flow penalty, kappa times the strain along the field"))
    tol: float = field(default=NEWTON_TOL, metadata=dict(
        help="the forcing term: the residual of the Newton system below tol of the right-hand side ends the solve"))
    maxiter: int = field(default=NEWTON_MAXITER, metadata=dict(help="MINRES iterations of the Newton solve at most"))


@dataclass(frozen=True)
class Budget:
    """Step budget and stopping."""
    steps: Optional[int] = field(default=None, metadata=dict(help="maximum steps [100 Newton, 2000 gradient]"))
    chunk: Optional[int] = field(default=None, metadata=dict(
        help="steps per compiled chunk: trace, qoi sample, checkpoint, outputs and the floor test once per chunk; "
             "steps is a multiple of it [10 Newton, 200 gradient]"))
    floor_tol: float = field(default=1e-10, metadata=dict(
        help="stop when the last chunk's mean squared normalised force residual is below this"))


@dataclass(frozen=True)
class Drive:
    """Resistive steady state: a resistive dose in every step towards a reference current, optionally driven."""
    resistivity: float = field(default=0.0, metadata=dict(
        help="a resistive dose C h_r^2 in every step, E = eta (J - J*), J* the current of the reference field B*"))
    reference: Optional[str] = field(default=None, metadata=dict(
        help="the checkpoint of B* (required with --drive-resistivity): a converged nested equilibrium, one common "
             "reference for differently seeded arms"))
    reference_smoothing: float = field(default=0.1, metadata=dict(
        help="B* after one heat step of c h_r^2, which removes its rational-surface sheets"))
    chain: Optional[float] = field(default=None, metadata=dict(
        help="drive the reference with the resonant seed of the chain at this rotational transform nfp n / m"))
    eps: float = field(default=0.0, metadata=dict(
        help="the drive's amplitude, the resonant normal field |dB^r| / |B^zeta| at r_mn, signed"))

    def __post_init__(self):
        if self.resistivity and self.reference is None:
            raise ValueError("--drive-resistivity needs --drive-reference, the checkpoint of B*")
        if self.chain is not None and not self.resistivity:
            raise ValueError("--drive-chain needs --drive-resistivity (the drive is the source's, not the field's)")

    def __bool__(self):
        return bool(self.resistivity)


@dataclass(frozen=True)
class Output:
    """Where the run goes and where it continues from."""
    out: Optional[str] = field(default=None, metadata=dict(help="run directory [outputs/relax/<date>/<time>]"))
    restart: Optional[str] = field(default=None, metadata=dict(
        help="continue from a checkpoints/state_<step>.h5 of the same geometry, mesh, degree and precision"))


@dataclass(frozen=True)
class RelaxConfig:
    """A relaxation run. Groups in the order of the command line; every field's default is the production one."""
    geometry: Geometry
    seed: Seed = field(default=Seed(), metadata=dict(prefix="seed"))
    descent: Descent = Descent()
    newton: Newton = field(default=Newton(), metadata=dict(prefix="newton"))
    budget: Budget = Budget()
    drive: Drive = field(default=Drive(), metadata=dict(prefix="drive"))
    output: Output = Output()

    def __post_init__(self):
        d, b = self.descent, self.budget
        # the budget's defaults depend on the method: resolved here, once, into the frozen config
        steps = b.steps if b.steps is not None else (100 if d.newton else 2000)
        chunk = b.chunk if b.chunk is not None else (10 if d.newton else 200)
        object.__setattr__(self, "budget", replace(b, steps=steps, chunk=chunk))
        if chunk < 1 or steps % chunk:
            raise ValueError("--steps must be a positive multiple of --chunk")
        if self.seed and self.geometry.analytic:
            raise ValueError("--seed needs an equilibrium file (.nc or .dat)")

    def stepper(self, seq, h_r_sq):
        """The :class:`~mrx.relaxation.TimeStepper` of this configuration on ``seq``."""
        from mrx.relaxation import TimeStepper
        n = self.newton
        return TimeStepper(seq=seq, newton_penalty=n.penalty, newton_tol=n.tol, newton_maxiter=n.maxiter,
                           resistivity=self.drive.resistivity * h_r_sq, **self.descent.stepper_kwargs())

    def relax_kwargs(self):
        """The keyword arguments of :func:`~mrx.relaxation.relax` beyond the state and the stepper."""
        b = self.budget
        return dict(steps=b.steps, chunk=b.chunk, floor_tol=b.floor_tol)

    @property
    def params(self) -> dict:
        """The flat record of this configuration, ``relax.json``'s ``params`` (the run's facts are added by the driver)."""
        from mrx.cli import flatten
        return flatten(self)

    @classmethod
    def from_params(cls, params: dict) -> "RelaxConfig":
        """The configuration of a record's ``params``."""
        from mrx.cli import unflatten
        return unflatten(cls, params)
