r"""Poincare sections of a magnetic field given as a discrete 2-form.

The entry point is :func:`poincare`. It starts field lines on a ray from the magnetic axis to the edge,
follows them for many field periods, measures the rotational transform iota of each line, marks the lines
that are chaotic, and returns their crossings with a few toroidal planes as ``(R, Z)`` points.
:func:`locked_width` reads a quick island width off that result. :func:`trace_archive` traces several fields
into one archive with the weak pressure at every crossing, the input of
:func:`mrx.diagnostics.plotting.plot_archive` and of ``scripts/poincare_plot.py``.

Field lines are integrated in logical coordinates with the toroidal angle zeta as the time variable,
:math:`dr/d\zeta = \hat B^r/\hat B^\zeta` and :math:`d\theta/d\zeta = \hat B^\theta/\hat B^\zeta`, where
:math:`\hat B` are the contravariant components of the field in logical coordinates. This is only valid while
:math:`\hat B^\zeta` keeps one sign, which :func:`require_zeta_parameterisation` checks before any tracing.

The tracer is compiled once per sequence and number of steps per period. Tracing another field on the same
sequence, or the same sequence after a new geometry, reuses the compiled tracer.
"""
from __future__ import annotations

import time
from fractions import Fraction
from functools import lru_cache, partial, reduce
from math import gcd

import diffrax as dfx
import jax
import jax.numpy as jnp
import numpy as np

from mrx.differential_forms import DiscreteFunction
from mrx.relaxation.physics import compute_force, weak_pressure

TWO_PI = 2.0 * jnp.pi

#: Lines are frozen once they reach this logical radius: the spline maps are singular at ``r = 1``.
R_MAX = 1.0 - 1e-6

#: Fewest integration steps per field period. The planes can raise it (see :func:`steps_for`).
MIN_STEPS_PER_PERIOD = 24

#: ``|B^zeta|/|B|`` at or below which the toroidal-angle parameterisation is refused.
BZETA_MIN_FRACTION = 0.05

#: A line is called chaotic when the iota measured on the two halves of its trace differ by more than this
#: number divided by the number of periods traced. On a line lying on a surface the difference falls like ``1/N``.
CHAOS_TOL_PER_PERIOD = 0.4

#: Distance from the magnetic axis of the extra line (seed 0) used to locate the axis.
R_AXIS = 0.01
#: Logical radius of the outermost seed.
R_EDGE = 0.97
#: Innermost seed, as a fraction of the distance from the axis to the edge.
T_MIN = 0.02
#: Field periods traced to locate the magnetic axis before seeding.
PROBE_PERIODS = 64
#: Field periods and number of lines of the step-size check (the ``drift`` of :func:`poincare`).
DRIFT_PERIODS, DRIFT_LINES = 64, 8


# ---------------------------------------------------------------------------
# The field
# ---------------------------------------------------------------------------

def logical_field(seq):
    r"""The function ``(x, dof) ->`` :math:`\hat B(x)`, the logical contravariant components of the Dirichlet
    2-form with coefficients ``dof`` at the logical point ``x``. This is the field-line direction in logical
    space. It does not depend on the geometry, and the same sequence always returns the same function object,
    so the compiled tracer is reused."""
    return _logical_field(seq.odd)      # cached per view. set_geometry keeps the views


@lru_cache(maxsize=None)
def _logical_field(odd):
    basis, extraction = odd.basis_2, odd.E(2)

    def field(x, dof):
        return DiscreteFunction(dof, basis, extraction)(x)
    return field


@partial(jax.jit, static_argnames=("field",))
def _field_values(field, dof, x):
    return jax.vmap(field, (0, None))(x, dof)


def require_zeta_parameterisation(field, dof, name="field"):
    """Check that zeta can serve as the time variable of the field lines and return the range ``(lo, hi)`` of
    ``B^zeta/|B|`` over 4096 random interior points. Raises ``ValueError`` if this ratio changes sign or comes
    within :data:`BZETA_MIN_FRACTION` of zero. Traced anyway, such a field would give a section that looks
    chaotic."""
    x = jax.random.uniform(jax.random.PRNGKey(23), (4096, 3))
    x = x.at[:, 0].multiply(0.96).at[:, 0].add(0.02)       # r in [0.02, 0.98]: off the polar axis and off r = 1
    b = _field_values(field, jnp.asarray(dof), x)
    frac = b[:, 2] / jnp.linalg.norm(b, axis=1)
    lo, hi = float(jnp.min(frac)), float(jnp.max(frac))
    absmin = float(jnp.min(jnp.abs(frac)))
    if lo < 0.0 < hi or absmin <= BZETA_MIN_FRACTION:
        worst = tuple(round(float(v), 4) for v in x[int(jnp.argmin(jnp.abs(frac)))])
        raise ValueError(f"{name}: B^zeta/|B| in [{lo:+.3e}, {hi:+.3e}], min |.| {absmin:.3e} at logical "
                         f"(r, theta, zeta) = {worst}: the toroidal angle is not a valid independent variable "
                         f"for this field (tol {BZETA_MIN_FRACTION:g})")
    return lo, hi


def to_uv(r, theta):
    """The Cartesian chart ``(u, v) = r (cos 2 pi theta, sin 2 pi theta)`` of the logical cross-section, stacked
    on a new last axis. Lines are traced in this chart because it has no singularity at the polar axis."""
    return jnp.stack([r * jnp.cos(TWO_PI * theta), r * jnp.sin(TWO_PI * theta)], axis=-1)


def to_polar(uv):
    """The inverse of :func:`to_uv`: ``(r, theta)`` of ``(u, v)`` points given on the last axis, with ``theta``
    in ``[0, 1)``."""
    return jnp.sqrt(uv[..., 0] ** 2 + uv[..., 1] ** 2), jnp.arctan2(uv[..., 1], uv[..., 0]) / TWO_PI % 1.0


def cross_section_rhs(field):
    """The field-line equation ``(zeta, y, dof) -> dy/dzeta`` in the ``(u, v)`` chart. A line stops moving once
    it reaches :data:`R_MAX`, which means it has left the domain and is reported as lost."""
    def rhs(zeta, y, dof):
        r, theta = to_polar(y)
        b = field(jnp.array([r, theta, zeta % 1.0]), dof)
        dr, dtheta = b[0] / b[2], b[1] / b[2]
        c, s = jnp.cos(TWO_PI * theta), jnp.sin(TWO_PI * theta)
        du = c * dr - TWO_PI * r * s * dtheta
        dv = s * dr + TWO_PI * r * c * dtheta
        return jnp.where(r < R_MAX, jnp.array([du, dv]), jnp.zeros(2))
    return rhs


# ---------------------------------------------------------------------------
# The trace
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("field", "n_periods", "steps_per_period"))
def trace(field, dof, seeds, n_periods, steps_per_period):
    """Trace field lines from the logical points ``seeds`` (rows ``(r, theta)`` at ``zeta = 0``) over
    ``n_periods`` field periods with ``steps_per_period`` fixed steps per period.

    Returns ``(ys, ok)``. ``ys`` holds the ``(u, v)`` position after every step, shape ``(n_seeds, n_periods *
    steps_per_period + 1, 2)``, and ``ok`` says per seed whether the integrator succeeded. Changing
    ``n_periods`` or ``steps_per_period`` compiles a new tracer."""
    n_steps = n_periods * steps_per_period
    step_ts = jnp.arange(n_steps + 1) / steps_per_period
    term = dfx.ODETerm(cross_section_rhs(field))

    def one(y0):
        sol = dfx.diffeqsolve(
            terms=term, solver=dfx.Tsit5(),
            t0=0.0, t1=float(n_periods), dt0=None, y0=y0, args=dof,
            saveat=dfx.SaveAt(ts=step_ts),
            stepsize_controller=dfx.StepTo(ts=step_ts),
            max_steps=n_steps + 1, throw=False,
        )
        return sol.ys, sol.result == dfx.RESULTS.successful

    return jax.vmap(one)(to_uv(seeds[:, 0], seeds[:, 1]))


def _escaped_mask(ys):
    """``True`` for seeds whose line reached the domain boundary."""
    r = jnp.sqrt(ys[..., 0] ** 2 + ys[..., 1] ** 2)
    return jnp.any(r >= R_MAX, axis=-1) | jnp.any(~jnp.isfinite(r), axis=-1)


def _step_convergence(field, dof, seeds, lo, n_periods, steps_per_period):
    """The largest ``(u, v)`` distance between the trace ``lo`` of ``seeds`` and a retrace at twice the steps,
    over the lines that stay inside the domain."""
    hi, _ = trace(field, dof, seeds, n_periods, 2 * steps_per_period)
    hi = hi[:, ::2]
    good = ~(_escaped_mask(lo) | _escaped_mask(hi))
    d = jnp.linalg.norm(lo - hi, axis=-1).max(axis=-1)
    return float(jnp.max(jnp.where(good, d, -jnp.inf)))


# ---------------------------------------------------------------------------
# Rotational transform
# ---------------------------------------------------------------------------

def axis_track(ys, steps_per_period):
    """The ``(u, v)`` position of the magnetic axis at every saved step. It is the average, over all traced
    periods, of the innermost line ``ys[0]`` at the same zeta within the period."""
    inner = ys[0, :-1].reshape(-1, steps_per_period, 2)
    center = jnp.mean(inner, axis=0)                    # (steps_per_period, 2)
    n_saves = ys.shape[1]
    reps = -(-n_saves // steps_per_period)
    return jnp.tile(center, (reps, 1))[:n_saves]


def _winding(ys, steps_per_period, center):
    """The unwrapped poloidal angle (turns) of every line about ``center`` and the logical zeta of every save."""
    d = ys - center
    angle = jnp.unwrap(jnp.arctan2(d[..., 1], d[..., 0]), axis=-1) / TWO_PI
    return angle, jnp.arange(ys.shape[1]) / steps_per_period


def _slope(angle, zeta):
    """Least-squares slope of the rows of ``angle`` against ``zeta``."""
    zc = zeta - jnp.mean(zeta)
    ac = angle - jnp.mean(angle, axis=-1, keepdims=True)
    return (ac @ zc) / (zc @ zc)


def rotational_transform(ys, steps_per_period, nfp, center=None):
    """The rotational transform iota of every line, in poloidal turns per full toroidal turn. It is the
    least-squares rate at which the line winds about ``center`` (by default :func:`axis_track`), per field
    period, times ``nfp``. Always non-negative."""
    if center is None:
        center = axis_track(ys, steps_per_period)
    return jnp.abs(_slope(*_winding(ys, steps_per_period, center))) * nfp


def _iota_convergence(ys, steps_per_period, nfp, center):
    """``|iota(first half) - iota(second half)|`` per line. It is small only on lines that have an iota."""
    angle, zeta = _winding(ys, steps_per_period, center)
    half = ys.shape[1] // 2
    i1 = jnp.abs(_slope(angle[:, :half], zeta[:half])) * nfp
    i2 = jnp.abs(_slope(angle[:, half:], zeta[half:])) * nfp
    return jnp.abs(i1 - i2)


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def seed_from_axis(field, dof, n_lines, steps_per_period, *, seed=0):
    """``n_lines + 1`` logical ``(r, theta)`` seeds at ``zeta = 0``. Seed 0 lies :data:`R_AXIS` from the magnetic
    axis and is used to track the axis. The others are spread from the magnetic axis to radius :data:`R_EDGE`,
    each towards its own random poloidal angle (random key ``seed``), so that the seeds do not all line up
    with the X-points of an island chain. Locating the axis traces two short probes first."""
    # Two passes: the first probe orbits the coordinate axis, the second the first estimate of the magnetic axis.
    probe = jnp.array([[R_AXIS, 0.0], [R_EDGE, 0.0]])
    ys, _ = trace(field, dof, probe, PROBE_PERIODS, steps_per_period)
    centre = jnp.mean(ys[0, ::steps_per_period], axis=0)
    probe2 = jnp.array([to_polar(centre + jnp.array([R_AXIS, 0.0])), [R_EDGE, 0.0]])
    ys, _ = trace(field, dof, probe2, PROBE_PERIODS, steps_per_period)
    centre = jnp.mean(ys[0, ::steps_per_period], axis=0)

    thetas = jax.random.uniform(jax.random.PRNGKey(seed), (n_lines,))
    edge = to_uv(R_EDGE, thetas)
    t = jnp.linspace(T_MIN, 1.0, n_lines)[:, None]
    uv = centre[None, :] + t * (edge - centre[None, :])
    return jnp.concatenate([probe2[:1], jnp.stack(to_polar(uv), axis=1)], axis=0)


# ---------------------------------------------------------------------------
# The section
# ---------------------------------------------------------------------------

def planes_for(seq, planes):
    """The section planes, as values of zeta in units of one field period. A sequence ``planes`` is returned as
    is. An integer gives that many equispaced planes: over half a period, both ends included, for a
    stellarator-symmetric map, and over the whole period otherwise."""
    if not isinstance(planes, int):
        return tuple(float(p) for p in planes)
    if seq.symmetry == "stellarator":
        return tuple(np.linspace(0.0, 0.5, planes).tolist())
    return tuple((np.arange(planes) / planes).tolist())


def steps_for(planes):
    """The number of steps per field period: the smallest multiple of the planes' common denominator that is at
    least :data:`MIN_STEPS_PER_PERIOD`. Every plane then falls exactly on the end of a step."""
    den = reduce(lambda a, b: a * b // gcd(a, b),
                 (Fraction(float(p)).limit_denominator(4096).denominator for p in planes), 1)
    return den * -(-MIN_STEPS_PER_PERIOD // den)


@partial(jax.jit, static_argnames=("seq",))
def _map_points(seq, x):
    return jax.vmap(seq.map)(x.reshape(-1, 3)).reshape(x.shape)


def to_RZ(seq, ys, zeta):
    """The cylindrical ``(R, Z)`` of ``(u, v)`` cross-section points at the logical angle ``zeta``."""
    r, theta = to_polar(ys)
    xyz = _map_points(seq, jnp.stack([r, theta, jnp.broadcast_to(jnp.asarray(zeta) % 1.0, r.shape)], axis=-1))
    return jnp.sqrt(xyz[..., 0] ** 2 + xyz[..., 1] ** 2), xyz[..., 2]


def _section_RZ(seq, ys, axis_uv, steps_per_period, plane):
    """The section at ``plane`` as a dict: ``R``, ``Z`` and the logical ``logr``, ``logth`` of the crossings,
    and ``axisR``, ``axisZ`` of the magnetic axis (the crossings of the line ``axis_uv``)."""
    off = int(round(plane * steps_per_period))
    if abs(off - plane * steps_per_period) > 1e-9:
        raise ValueError(f"plane {plane} is not a step endpoint at {steps_per_period} steps per period")
    uv = np.asarray(ys)[:, off::steps_per_period, :]
    R, Z = to_RZ(seq, jnp.asarray(uv), plane)
    aR, aZ = to_RZ(seq, jnp.asarray(np.asarray(axis_uv)[off::steps_per_period, :]), plane)
    return {"R": np.asarray(R), "Z": np.asarray(Z), "axisR": np.asarray(aR), "axisZ": np.asarray(aZ),
            "logr": np.hypot(uv[..., 0], uv[..., 1]),
            "logth": np.arctan2(uv[..., 1], uv[..., 0]) / (2.0 * np.pi) % 1.0}


def poincare(seq, dof, *, lines=160, periods=400, planes=5, seed=0, name="field"):
    """The Poincare sections of the field given by the Dirichlet 2-form ``dof`` on ``seq``.

    ``lines`` field lines (:func:`seed_from_axis`) are traced for ``periods`` field periods and cut at the
    ``planes`` (a count or the planes themselves, see :func:`planes_for`). ``name`` only labels error messages.
    Returns a dict with

    * per line: ``iota``, ``seed_r`` (the logical radius of the seed), ``keep`` (the line stayed inside the
      domain to the end), ``chaotic`` (the iota of the two halves of the trace differ by more than
      :data:`CHAOS_TOL_PER_PERIOD` ``/ periods``) and ``shown`` (``keep & ~chaotic``).
    * ``sections[plane]``: a dict with ``R``, ``Z``, the logical ``logr`` and ``logth`` of the crossings, arrays
      of shape ``(line, crossing)``, and ``axisR``, ``axisZ`` of the magnetic axis.
    * ``drift``: how far a few regular lines move, in logical radius, when the step is halved. It is the check
      that the step count is sufficient, and it is NaN when no line is regular. It is only measured on regular
      lines because on a chaotic line it would measure the divergence of neighbouring lines instead.
    * ``steps`` (per period), ``nfp``, ``bz_over_b`` (the range of ``B^zeta/|B|``) and ``walltime`` (seconds of
      the main trace).
    """
    field, dof, nfp = logical_field(seq), jnp.asarray(dof), seq.nfp
    planes = planes_for(seq, planes)
    steps = steps_for(planes)
    bz_over_b = require_zeta_parameterisation(field, dof, name)
    seeds = seed_from_axis(field, dof, lines, steps, seed=seed)

    t0 = time.perf_counter()
    ys, ok = trace(field, dof, seeds, periods, steps)
    ys = ys.block_until_ready()
    walltime = time.perf_counter() - t0

    escaped = _escaped_mask(ys)
    centre = axis_track(ys, steps)
    iota = rotational_transform(ys, steps, nfp, center=centre)
    chaotic = _iota_convergence(ys, steps, nfp, centre) > CHAOS_TOL_PER_PERIOD / periods

    # Up to DRIFT_LINES regular lines, spread over all regular ones and excluding the axis probe. A fixed count
    # keeps the compiled drift trace the same from one field to the next.
    regular = np.flatnonzero(~np.asarray(chaotic | escaped))
    regular = regular[regular > 0]
    idx = regular[np.unique(np.linspace(0, len(regular) - 1,
                                        min(DRIFT_LINES, len(regular))).round().astype(int))]
    n_drift = min(periods, DRIFT_PERIODS)
    drift = (_step_convergence(field, dof, seeds[idx], ys[idx, :n_drift * steps + 1], n_drift, steps)
             if idx.size else float("nan"))

    keep = np.asarray(~(escaped | ~ok))[1:]
    chaotic = np.asarray(chaotic)[1:]
    return {"iota": np.asarray(iota)[1:], "seed_r": np.asarray(seeds[1:, 0]), "keep": keep, "chaotic": chaotic,
            "shown": keep & ~chaotic,
            "sections": {plane: _section_RZ(seq, ys[1:], ys[0], steps, plane) for plane in planes},
            "drift": drift, "steps": steps, "nfp": nfp, "bz_over_b": bz_over_b, "walltime": walltime}


def trace_archive(seq, fields, *, lines=160, periods=400, planes=5, seed=0, source=""):
    """Trace several fields on ``seq`` into one archive and return ``(archive, results)``.

    ``fields`` maps a name to ``(B, step)``, with ``B`` the odd Dirichlet 2-form of the field and ``step`` the
    relaxation step it belongs to. Every field is traced by :func:`poincare` with the same ``lines``,
    ``periods``, ``planes`` and ``seed``, and ``results`` maps each name to its result. The weak pressure of
    each field (:func:`mrx.relaxation.physics.weak_pressure`, two solves) is evaluated at every crossing.
    ``archive`` is a flat dict of arrays, the content of the ``trace.npz`` of ``scripts/poincare_trace.py``
    (see there for the keys), and ``source`` describes the run in it.
    """
    planes = planes_for(seq, planes)
    volume = float(jnp.sum(seq.quad.w * seq.jacobian_j))

    @jax.jit
    def weak_at(pd, x):
        return jax.vmap(DiscreteFunction(pd, seq.basis_0, seq.even.E(0)))(x)[:, 0]

    archive = {"fields": np.array(list(fields)), "planes": np.array(planes), "resolution": np.array(seq.ns),
               "p": seq.p, "nfp": seq.nfp, "symmetry": np.array(seq.symmetry), "steps": steps_for(planes),
               "source": np.array(source), "trace_precision": np.array(jnp.dtype(seq.dtype).name)}
    results = {}
    for name, (B, step) in fields.items():
        B = jnp.asarray(B)
        _, _, J, _ = compute_force(B, seq)
        pw = jnp.asarray(weak_pressure(J, B, seq))
        res = poincare(seq, B, lines=lines, periods=periods, planes=planes, seed=seed, name=name)
        results[name] = res
        archive[f"{name}_label"] = np.array(f"{name} (step {step})")
        archive[f"{name}_step"] = step
        archive[f"{name}_bsq"] = float(seq.odd.l2_norm_sq(B, 2)) / volume
        for key in ("iota", "seed_r", "keep", "chaotic", "shown", "drift"):
            archive[f"{name}_{key}"] = np.asarray(res[key])
        for plane, sec in res["sections"].items():
            tag = f"{name}_zeta{plane:g}"
            for key, arr in sec.items():
                archive[f"{tag}_{key}"] = arr
            x = jnp.stack([jnp.asarray(sec["logr"]).ravel(), jnp.asarray(sec["logth"]).ravel(),
                           jnp.full(sec["logr"].size, plane)], axis=1)
            archive[f"{tag}_pressure"] = np.asarray(weak_at(pw, x)).reshape(sec["logr"].shape)
        shown = res["shown"]
        span = (f"iota {float(res['iota'][shown].min()):.4f}..{float(res['iota'][shown].max()):.4f}"
                if shown.any() else "no line converged")
        print(f"[{name}] B^zeta/|B| in [{res['bz_over_b'][0]:+.3e}, {res['bz_over_b'][1]:+.3e}], "
              f"trace {res['walltime']:.1f}s, {int((~res['keep']).sum())}/{res['keep'].size} lost, "
              f"{int((res['keep'] & res['chaotic']).sum())} chaotic, drift {res['drift']:.2e}, {span}", flush=True)
    return archive, results


def locked_width(res, iota, tol=2e-3):
    """Return ``(width, n_locked)`` for the rotational transform ``iota`` in ``res``, the result of
    :func:`poincare`. The locked lines are the kept lines whose ``|iota|`` lies within ``tol`` of it, and
    ``width`` is ``max(r) - min(r)`` of their seed radii (0 with fewer than two). This is a quick island width
    with the resolution of the seed spacing. :func:`mrx.diagnostics.islands.islands` measures a chain properly
    but traces many more lines."""
    locked = np.asarray(res["keep"]) & (np.abs(np.abs(np.asarray(res["iota"])) - iota) < tol)
    r = np.asarray(res["seed_r"])[locked]
    return (float(r.max() - r.min()) if r.size > 1 else 0.0), int(locked.sum())
