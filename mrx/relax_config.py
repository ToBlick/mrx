"""The configuration of a relaxation run, ``scripts/relax.py``'s and the tutorials'.

One frozen dataclass per group, the defaults on the classes, the validation
in ``__post_init__``; :mod:`mrx.cli` makes the command line of it and the
flat ``params`` of ``relax.json``. :meth:`RelaxConfig.stepper` builds the
:class:`~mrx.relaxation.TimeStepper` and :meth:`RelaxConfig.relax_kwargs`
the arguments of :func:`~mrx.relaxation.relax`, so a tutorial and the
command line configure the same objects the same way.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Literal, Optional

from mrx.hessian import NEWTON_MAXITER, NEWTON_PASSES, NEWTON_PENALTY, NEWTON_TOL


def _ints(s):
    return tuple(int(v) for v in s.split(","))


def _knots(s):
    from mrx.geometry import parse_knots
    return parse_knots(s)


def _window(s):
    a, b = s.split(":")
    return int(a), int(b)


@dataclass(frozen=True)
class Geometry:
    """Geometry, initial condition and discretisation."""
    path: str = field(metadata=dict(
        flag="--geometry",
        help="the geometry AND the initial condition: a VMEC wout (.nc) or GVEC state (.dat) gives the map and the "
             "equilibrium's own field B = dA'; an analytic geometry (.json: a map of mrx.mappings with its parameters "
             "and the profiles of the logical-grid field) gives the map and that field"))
    nfp: Optional[int] = field(default=None, metadata=dict(help="field periods; overrides the file's nfp attribute"))
    symmetry: Literal["stellarator", "field-period", "none"] = field(
        default="stellarator", metadata=dict(help="what the map satisfies (mrx.geometry.SYMMETRIES)"))
    ns: tuple[int, int, int] = field(default=(32, 64, 64), metadata=dict(
        parse=_ints, help="spline resolution (r, theta, zeta), also the map's"))
    knots_r: Optional[tuple] = field(default=None, metadata=dict(
        parse=_knots, help="breakpoints of the r axis, comma-separated from 0 to 1, instead of the uniform grid"))
    knots_theta: Optional[tuple] = field(default=None, metadata=dict(parse=_knots, help="the theta axis's"))
    knots_zeta: Optional[tuple] = field(default=None, metadata=dict(parse=_knots, help="the zeta axis's"))
    p: int = field(default=2, metadata=dict(help="spline degree; p+1 Gauss points per span"))
    solve_maxiter: int = field(default=2000, metadata=dict(help="iteration budget of every inner solve"))
    solve_tol: Optional[float] = field(default=None, metadata=dict(
        help="residual tolerance of every solve [the precision's: 1e-8 refined float32, 1e-10 float64]"))
    precision: Literal["mixed", "float32", "float64"] = field(default="mixed", metadata=dict(
        help="mixed: float32 fields and solves with a float64 residual; float32, float64: both"))
    map_batch: int = field(default=0, metadata=dict(
        help="cells per batch of the quadrature loops (mrx.MAP_BATCH_SIZE_INNER); 0 = all points in one vmap; "
             "bound it at high resolution"))

    def __post_init__(self):
        if self.map_batch < 0:
            raise ValueError("--map-batch must be non-negative (0 is one vmap over all points)")
        if not os.path.isfile(self.path):
            raise ValueError(f"--geometry {self.path!r} is not a file (a .nc, .dat or .json)")

    @property
    def knots(self):
        return [self.knots_r, self.knots_theta, self.knots_zeta]

    @property
    def analytic(self):
        return self.path.endswith(".json")


@dataclass(frozen=True)
class Seed:
    """Island seed in the initial field (equilibrium files only)."""
    spec: str = field(default="", metadata=dict(
        flag="--seed", help='resonant seed "m,n,rho0,width" added to the potential'))
    eps: float = field(default=0.0, metadata=dict(
        help="its amplitude |dB^rho| / |B^zeta| at rho0 (island width ~ sqrt of it)"))
    phase: float = field(default=0.0, metadata=dict(
        help="the seed's phase in turns of the resonant angle (m theta - s n zeta)"))

    def __bool__(self):
        return bool(self.spec)

    def parsed(self):
        from mrx.initial_conditions import parse_seed
        return parse_seed(self.spec, self.eps, self.phase)


@dataclass(frozen=True)
class Drive:
    """Resonant drive of the resistive source (with --resistivity)."""
    spec: str = field(default="", metadata=dict(
        flag="--drive", help='a resonant drive "m,n,rho0,width" (the --seed perturbation of the potential) added to '
                             'the source B* only, not to the field'))
    eps: float = field(default=0.0, metadata=dict(help="the drive's amplitude, as --seed-eps"))
    phase: float = field(default=0.0, metadata=dict(help="the drive's phase, as --seed-phase"))

    def __bool__(self):
        return bool(self.spec)

    def parsed(self):
        from mrx.initial_conditions import parse_seed
        return parse_seed(self.spec, self.eps, self.phase)


@dataclass(frozen=True)
class Descent:
    """The descent direction and the induction step."""
    method: Literal["newton", "gradient"] = field(default="newton", metadata=dict(
        help="the direction: Newton on the second variation, or gradient descent on the smoothed force"))
    auxiliary_B_field: bool = field(default=False, metadata=dict(
        flag="--auxiliary-B-field",
        help="route the cross products through the Dirichlet 1-form H = M_1^-1 P B"))
    midpoint: bool = field(default=False, metadata=dict(
        help="midpoint-implicit induction at the predictor's velocity (Picard on the increment)"))
    helicity_correction: bool = field(default=False, metadata=dict(
        help="zero the step's discrete helicity change by one scalar correction of E"))
    velocity_smoothing_order: int = field(default=1, metadata=dict(
        help="descent direction v = (I - scale L)^-order F; 0 is off and fragile (the unsmoothed descent stops "
             "conserving helicity after ~1e4 steps)"))
    velocity_smoothing_scale: Optional[float] = field(default=None, metadata=dict(
        help="length scale of the velocity smoothing [mrx.relaxation.SMOOTHING_C h_r^2, h_r the physical radial cell]"))
    cfl: float = field(default=0.5, metadata=dict(
        help="cap the line-search step at cfl / (largest logical CFL number of the velocity); inf disables it"))
    potential_velocity: Optional[bool] = field(default=None, metadata=dict(
        help="the projected force as curl a + c h (k=1 Hodge solve) instead of the Leray solve [on for the gradient "
             "descent; Newton and the auxiliary field have their own routes]"))

    @property
    def newton(self):
        return self.method == "newton"


@dataclass(frozen=True)
class Newton:
    """Newton (--method newton): u = curl a from curl^T H curl a = curl^T M F by Newton-MR (mrx.hessian)."""
    penalty: float = field(default=NEWTON_PENALTY, metadata=dict(
        help="kappa of the parallel-flow penalty, kappa times the strain along the field"))
    tol: float = field(default=NEWTON_TOL, metadata=dict(
        help="the forcing term: the residual of the Newton system below tol of the right-hand side ends the solve"))
    maxiter: int = field(default=NEWTON_MAXITER, metadata=dict(help="MINRES iterations per pass"))
    passes: int = field(default=NEWTON_PASSES, metadata=dict(help="passes of the Newton solve at most"))


@dataclass(frozen=True)
class Budget:
    """Step budget and stopping."""
    steps: Optional[int] = field(default=None, metadata=dict(help="maximum steps [150 Newton, 3000 gradient]"))
    chunk: Optional[int] = field(default=None, metadata=dict(
        help="steps per compiled chunk: trace, qoi sample, checkpoint, outputs and the floor / reconnect tests once "
             "per chunk; steps is a multiple of it [25 Newton, 500 gradient]"))
    floor_tol: float = field(default=1e-8, metadata=dict(
        help="stop when the last chunk's mean squared normalised force residual is below this"))


@dataclass(frozen=True)
class Reconnection:
    """Reconnection series: one resistive solve every K steps of the ideal descent."""
    every: int = field(default=0, metadata=dict(
        help="reconnect the field with one resistive solve every K steps, rounded to whole chunks; 0 = off"))
    helicity: float = field(default=0.01, metadata=dict(help="the helicity each reconnection spends, |dH| / |H|"))
    eps: Optional[float] = field(default=None, metadata=dict(
        help="a constant dose eps = C h_r^2 per resistive solve (h_r the physical radial cell) instead of the "
             "helicity target"))
    window: Optional[tuple[int, int]] = field(default=None, metadata=dict(
        parse=_window, help="A:B, the resistive solves only at steps A..B"))


@dataclass(frozen=True)
class Resistive:
    """Resistive steady state: a resistive dose in every step towards a reference current."""
    resistivity: float = field(default=0.0, metadata=dict(
        help="a resistive dose C h_r^2 in every step, E = eta (J - J*)"))
    reference_smoothing: float = field(default=0.1, metadata=dict(
        help="B* = the start field after one heat step of c h_r^2"))
    reference: Optional[str] = field(default=None, metadata=dict(
        help="B* from this checkpoint's field instead of the start field (one common reference for differently "
             "seeded arms, e.g. the nested equilibrium); smoothed as above"))


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
    drive: Drive = field(default=Drive(), metadata=dict(prefix="drive"))
    descent: Descent = Descent()
    newton: Newton = field(default=Newton(), metadata=dict(prefix="newton"))
    budget: Budget = Budget()
    reconnect: Reconnection = field(default=Reconnection(), metadata=dict(prefix="reconnect"))
    resistive: Resistive = Resistive()
    output: Output = Output()

    def __post_init__(self):
        d, b = self.descent, self.budget
        # the budget's defaults depend on the method: resolved here, once, into the frozen config
        steps = b.steps if b.steps is not None else (150 if d.newton else 3000)
        chunk = b.chunk if b.chunk is not None else (25 if d.newton else 500)
        object.__setattr__(self, "budget", replace(b, steps=steps, chunk=chunk))
        if d.potential_velocity is None:
            object.__setattr__(self, "descent", replace(d, potential_velocity=not d.newton))
        if chunk < 1 or steps % chunk:
            raise ValueError("--steps must be a positive multiple of --chunk")
        if self.seed and self.geometry.analytic:
            raise ValueError("--seed needs an equilibrium file (.nc or .dat)")
        if self.drive and (self.seed or not self.resistive.resistivity):
            raise ValueError("--drive needs --resistivity and no --seed (the drive is the source's, not the field's)")

    def stepper(self, seq, h_r_sq):
        """The :class:`~mrx.relaxation.TimeStepper` of this configuration on ``seq``."""
        from mrx.relaxation import TimeStepper
        d, n = self.descent, self.newton
        return TimeStepper(
            seq=seq, auxiliary_B_field=d.auxiliary_B_field, cfl=d.cfl, midpoint=d.midpoint,
            helicity_correction=d.helicity_correction, velocity_smoothing_order=d.velocity_smoothing_order,
            velocity_smoothing_scale=d.velocity_smoothing_scale, potential_velocity=d.potential_velocity,
            newton=d.newton, newton_penalty=n.penalty, newton_tol=n.tol, newton_maxiter=n.maxiter,
            newton_passes=n.passes, resistivity=self.resistive.resistivity * h_r_sq)

    def relax_kwargs(self, h_r_sq):
        """The keyword arguments of :func:`~mrx.relaxation.relax` beyond the state and the stepper."""
        b, r = self.budget, self.reconnect
        return dict(steps=b.steps, chunk=b.chunk, floor_tol=b.floor_tol, reconnect_every=r.every,
                    reconnect_helicity=r.helicity, reconnect_eps=None if r.eps is None else r.eps * h_r_sq,
                    reconnect_window=r.window)

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
