r"""Poincare sections of a magnetic field given as a discrete 2-form.

The entry point is :func:`poincare`. It starts field lines on a ray from the magnetic axis to the edge,
follows them for many field periods, measures the rotational transform iota of each line, marks the lines
that are chaotic, and returns their crossings with a few planes ``phi = const`` of the physical toroidal angle as
cylindrical ``(R, Z)`` points, the sections people expect. On request the sections are the map's own ``zeta =
const`` instead, which cost nothing extra but differ from the planes ``phi = const`` on a map whose sections are
normal to its axis (the default of :func:`mrx.geometry.build_sequence`) or on a G-frame state, and whose
coordinates are then ``(X1, X2)`` in the plane of the frame, under the same keys.
:func:`locked_width` reads a quick island width off that result. :func:`trace_archive` traces several fields
into one archive with the weak pressure at every crossing, the input of
:func:`mrx.diagnostics.plotting.plot_archive` and of ``scripts/poincare_plot.py``.

Field lines are integrated in logical coordinates with the toroidal angle zeta as the time variable,
:math:`dr/d\zeta = \hat B^r/\hat B^\zeta` and :math:`d\theta/d\zeta = \hat B^\theta/\hat B^\zeta`, where
:math:`\hat B` are the contravariant components of the field in logical coordinates. This is only valid while
:math:`\hat B^\zeta` keeps one sign, which :func:`require_zeta_parameterisation` checks before any tracing.

The tracer is compiled once per sequence and number of steps per period. Tracing another field on the same
sequence, or the same sequence after a new geometry, reuses the compiled tracer. :func:`poincare_many` traces
several fields and all their lines in one batch (see "The batched trace" below), which :func:`poincare` and
:func:`trace_archive` use.
"""
from __future__ import annotations

import time
from fractions import Fraction
from functools import lru_cache, partial, reduce
from math import gcd

import jax
import jax.numpy as jnp
import numpy as np

from mrx.differential_forms import DiscreteFunction
from mrx.relaxation.physics import current, weak_pressure

TWO_PI = 2.0 * jnp.pi

#: Lines are frozen once they reach this logical radius: the spline maps are singular at ``r = 1``.
R_MAX = 1.0 - 1e-6

#: Fewest integration steps per field period. The planes can raise it (see :func:`steps_for`).
MIN_STEPS_PER_PERIOD = 300

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

def logical_field(seq, dof):
    r"""The function ``(x, dof) ->`` :math:`\hat B(x)`, the logical contravariant components of the 2-form with
    coefficients ``dof`` at the logical point ``x``. This is the field-line direction in logical space. ``dof`` is
    a Dirichlet 2-form or, with a flux through the wall, a free one (``seq.odd.free``), told apart by its length.
    The function does not depend on the geometry, and the same sequence and space always return the same function
    object, so the compiled tracer is reused.

    Each call extracts the spline coefficients from ``dof`` (a sparse product over the whole field). The tracer
    instead takes them once per field, ``field.coefficients(dof)``, and evaluates ``field.at(x, coefficients)``,
    the local spline sum alone, which is the same for both spaces."""
    odd = seq.odd
    return _logical_field(odd, jnp.shape(dof)[0] == odd.free.n(2))     # cached per view and space


@lru_cache(maxsize=None)
def _logical_field(odd, free):
    basis, extraction = odd.basis_2, (odd.free if free else odd).E(2)

    def field(x, dof):
        return DiscreteFunction(dof, basis, extraction)(x)

    def coefficients(dof):
        return basis.raw_blocks(extraction.T @ dof)

    def at(x, coef):
        return basis.contract(coef, x)
    field.coefficients, field.at = coefficients, at
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
    """The field-line equation ``(zeta, y, coef) -> dy/dzeta`` in the ``(u, v)`` chart, with ``coef`` the field's
    spline coefficients (``field.coefficients(dof)``, taken once per field). A line stops moving once it reaches
    :data:`R_MAX`, which means it has left the domain and is reported as lost."""
    at = getattr(field, "at", field)                # a plain function (x, dof) -> B works too, coef = dof

    def rhs(zeta, y, coef):
        r, theta = to_polar(y)
        b = at(jnp.array([r, theta, zeta % 1.0]), coef)
        dr, dtheta = b[0] / b[2], b[1] / b[2]
        c, s = jnp.cos(TWO_PI * theta), jnp.sin(TWO_PI * theta)
        du = c * dr - TWO_PI * r * s * dtheta
        dv = s * dr + TWO_PI * r * c * dtheta
        return jnp.where(r < R_MAX, jnp.array([du, dv]), jnp.zeros(2))
    return rhs


# ---------------------------------------------------------------------------
# The trace
# ---------------------------------------------------------------------------

#: The Runge-Kutta pair of Tsitouras (2011), of which the fifth-order solution is taken with fixed steps: the
#: nodes, the rows of the stage matrix and the weights.
_TSIT5_C = (0.0, 0.161, 0.327, 0.9, 0.98002554090450968572981028628702459549, 1.0)
_TSIT5_A = ((),
            (0.161,),
            (-0.0084806554923569885444268742502307746751, 0.33548065549235698854442687425023077468),
            (2.8971530571054934321304325941929387649, -6.3594484899750748431481599123838256260,
             4.3622954328695814110177273181908868610),
            (5.3258648284392566044288779208405113178, -11.748883564062827877747170339785772962,
             7.4955393428898362083046047845643581557, -0.092495066361755249256502079332071916113),
            (5.8614554429464200286592514869826478904, -12.920969317847109291706118681783359395,
             8.1593678985761586431804007945392534852, -0.071584973281400997224530542525829738691,
             -0.028269050394068382909003057212712241467))
_TSIT5_B = (0.096460766818065229518167313165128763337, 0.01, 0.47988965041449957477524953229059651991,
            1.3790085741037418931922748218568727708, -3.2900695154360806799010475857113638501,
            2.3247105240997739824153559183987657961)


def _tsit5_step(rhs, t, y, h, args):
    """One step of length ``h`` of ``dy/dt = rhs(t, y, args)`` from ``(t, y)``."""
    k = []
    for c, a in zip(_TSIT5_C, _TSIT5_A):
        k.append(rhs(t + c * h, y + h * sum(aj * kj for aj, kj in zip(a, k)) if a else y, args))
    return y + h * sum(b * kj for b, kj in zip(_TSIT5_B, k))


@partial(jax.jit, static_argnames=("field", "n_periods", "steps_per_period"))
def trace(field, dof, seeds, n_periods, steps_per_period, direction=1):
    """Trace field lines from the logical points ``seeds`` (rows ``(r, theta)`` at ``zeta = 0``) over
    ``n_periods`` field periods with ``steps_per_period`` fixed steps per period, towards increasing ``zeta``
    (``direction = 1``) or decreasing (``-1``).

    Returns the ``(u, v)`` position after every step, shape ``(n_seeds, n_periods * steps_per_period + 1, 2)``.
    Changing ``n_periods`` or ``steps_per_period`` compiles a new tracer."""
    return _trace_saved(field, _coefficients(field, dof), to_uv(seeds[:, 0], seeds[:, 1]), n_periods,
                        steps_per_period, direction)


def _coefficients(field, dof):
    """The spline coefficients the tracer evaluates ``field`` with: ``field.coefficients(dof)`` for a field of
    :func:`logical_field`, ``dof`` itself for a plain function ``(x, dof) -> B``."""
    return field.coefficients(dof) if hasattr(field, "coefficients") else dof


def _trace_saved(field, coef, uv0, n_periods, steps_per_period, direction):
    """:func:`trace` from the field's spline coefficients ``coef`` and the ``(u, v)`` starts ``uv0``."""
    rhs = cross_section_rhs(field)
    h = direction / steps_per_period

    def step(y, i):
        y = _tsit5_step(rhs, direction * i / steps_per_period, y, h, coef)
        return y, y

    def one(y0):
        _, ys = jax.lax.scan(step, y0, jnp.arange(n_periods * steps_per_period))
        return jnp.concatenate([y0[None], ys])

    return jax.vmap(one)(uv0)


@partial(jax.jit, static_argnames=("field", "n_periods", "steps_per_period"))
def _trace_saved_many(field, coefs, uv0, n_periods, steps_per_period):
    """:func:`_trace_saved` forwards for several fields at once: ``coefs`` stacked on a leading axis, ``uv0``
    ``(fields, lines, 2)``."""
    return jax.vmap(lambda c, u: _trace_saved(field, c, u, n_periods, steps_per_period, 1))(coefs, uv0)


# ---------------------------------------------------------------------------
# The batched trace
# ---------------------------------------------------------------------------
#
# A step of the tracer costs nearly the same whatever the number of lines up to about 10^4 on a GPU (an H200,
# W7-X at (60, 80, 80): 0.17 ms for 120 lines and 0.25 ms for 12,000, the per-step cost is kernel launches, not
# arithmetic). :func:`_reduced_trace` therefore traces many fields and all their lines (the main seeds, the padding
# forwards and backwards) in one batch, and keeps only what the sections and the fits need instead of every step.
# These are the states at the steps that end on a plane (:func:`_on_step_ends`) or else the steps that cross a plane
# ``phi = const`` (refined by :func:`_refine_crossings`), the running fits of the winding, and the step at which a
# line left the domain.

#: The most lines (fields times lines per field) traced in one batch. More fields are traced in several batches.
MAX_BATCH_LINES = 200_000


def _record_offsets(planes, steps_per_period):
    """The steps within a period (``1 .. steps_per_period``) at whose end the batched trace keeps the state: the
    forward and the backward crossings of every plane, and the period's end."""
    offsets = {steps_per_period}
    for p in planes:
        k = int(round(p * steps_per_period)) % steps_per_period
        offsets.update({k or steps_per_period, (steps_per_period - k) or steps_per_period})
    return tuple(sorted(offsets))


@partial(jax.jit, static_argnames=("field", "n_periods", "steps_per_period", "offsets", "planes", "n_cross"))
def _reduced_trace(field, coefs, uv0, direction, track, n_saves, n_periods, steps_per_period, offsets, Phi=None,
                   planes=(), n_cross=0):
    """Trace the lines of several fields at once and keep only what the section needs.

    ``coefs`` are the fields' spline coefficients (``field.coefficients``) stacked on a leading axis, ``uv0``
    ``(fields, lines, 2)`` the starts in the ``(u, v)`` chart at ``zeta = 0``, ``direction`` ``(lines,)`` +1 or -1
    per line, ``track`` ``(fields, steps_per_period, 2)`` the magnetic axis at every step of a period, about which
    the winding is counted, and ``n_saves`` ``(lines,)`` the number of steps, the start included, over which each
    line's winding rate is fitted. Returns

    * ``rec`` ``(fields, lines, n_periods, len(offsets), 2)``: the state at the end of the steps ``offsets`` of
      every period (:func:`_record_offsets`).
    * ``rates`` ``(fields, lines, 3)``: the least-squares rate of the unwrapped poloidal angle about the axis, in
      turns per period, over the first ``n_saves`` steps, over their first half and over their second half, fitted
      on every step and accumulated as the lines go (:func:`_rates_iota` turns them into iota and the chaos test).
    * ``escape`` ``(fields, lines)``: the first step at whose end a line reached :data:`R_MAX` or stopped being
      finite (0 for a start outside), ``n_periods * steps_per_period + 1`` for a line that never did. A line is
      frozen from there on (:func:`cross_section_rhs`).

    With the map ``Phi`` (for sections ``phi = const`` that are not the map's) it also follows every line's
    physical toroidal angle (:func:`_toroidal_periods`) and keeps, for the crossings ``n = 1 .. n_cross`` of each
    of the ``planes`` (``phi = 2 pi (plane + direction n) / nfp``), the step that crosses. These are ``brackets``
    ``(fields, lines, len(planes), n_cross, 5)``, the ``(u, v)`` and the logical ``zeta`` at its start and the
    angle at its start and end, which :func:`_refine_crossings` turns into the crossings, and ``reached`` ``(fields,
    lines, len(planes))``, the number of crossings found."""
    rhs = cross_section_rhs(field)
    S = steps_per_period
    h = direction / S
    segments = np.diff((0,) + tuple(offsets))
    never = n_periods * S + 1
    # the least-squares fits over the saves 0 .. n - 1 at zeta = i / S, centred: rate = sum (zeta - mean) a / sum
    # (zeta - mean)^2, over the full range and over its two halves split at n // 2
    n = n_saves * 1.0
    half = n_saves // 2
    lo = jnp.stack([jnp.zeros_like(n), jnp.zeros_like(n), half.astype(n.dtype)])           # (3, lines)
    hi = jnp.stack([n, half.astype(n.dtype), n])
    count = hi - lo
    mean = (lo + hi - 1.0) / (2.0 * S)
    denom = count * (count ** 2 - 1.0) / (12.0 * S ** 2)

    def per_field(coef, uv, ctr):
        def angle(y, g):
            d = y - ctr[jnp.mod(jnp.where(direction > 0, g, -g), S)]
            return jnp.arctan2(d[:, 1], d[:, 0])

        def out(y):
            r = jnp.sqrt(y[:, 0] ** 2 + y[:, 1] ** 2)
            return (r >= R_MAX) | ~jnp.isfinite(r)

        def toroidal(y, g):                                      # the physical angle, signed by the direction
            return direction * jax.vmap(lambda yy, z: _toroidal_periods(Phi, yy, z))(y, direction * g / S)

        rows = jnp.arange(uv.shape[0])

        def capture(y0, g0, F0, F1, nxt, brackets):
            # a line crosses the level direction * plane + nxt in this step when its angle reaches it
            z0 = direction * g0 / S
            for j, p in enumerate(planes):
                hit = (F1 >= direction * p + nxt[:, j]) & (nxt[:, j] <= n_cross)
                slot = jnp.clip(nxt[:, j] - 1, 0, n_cross - 1)
                new = jnp.concatenate([y0, z0[:, None], F0[:, None], F1[:, None]], axis=1)
                brackets = brackets.at[rows, j, slot].set(jnp.where(hit[:, None], new, brackets[rows, j, slot]))
                nxt = nxt.at[:, j].add(hit.astype(nxt.dtype))
            return nxt, brackets

        def step(carry, _):
            # the winding is the angle plus whole turns, as jnp.unwrap counts them, so that it does not gather
            # round-off over the steps. The fits are summed per period (part) and then over the periods (acc)
            y, g, a, turns, esc, acc, part, cross = carry
            y0 = y
            y = jax.vmap(lambda yy, t, hh: _tsit5_step(rhs, t, yy, hh, coef))(y, direction * g / S, h)
            if Phi is not None:
                F0, nxt, brackets = cross
                F1 = toroidal(y, g + 1)
                nxt, brackets = capture(y0, g, F0, F1, nxt, brackets)
                cross = (F1, nxt, brackets)
            g = g + 1
            a_new = angle(y, g)
            turns = turns - jnp.round((a_new - a) / TWO_PI).astype(jnp.int32)
            esc = jnp.where(out(y) & (esc == never), g, esc)
            wind = a_new - a0 + TWO_PI * turns
            part = part + jnp.where((g >= lo) & (g < hi), (g / S - mean) * wind, 0.0)
            return (y, g, a_new, turns, esc, acc, part, cross), None

        def period(carry, _):
            kept = []
            for k in segments:
                carry, _ = jax.lax.scan(step, carry, None, length=int(k))
                kept.append(carry[0])
            y, g, a, turns, esc, acc, part, cross = carry
            return (y, g, a, turns, esc, acc + part, jnp.zeros_like(part), cross), jnp.stack(kept, axis=1)

        a0 = angle(uv, 0)                         # the winding counted from the start (save 0 adds nothing)
        acc = jnp.zeros_like(lo)
        g0 = jnp.asarray(0, dtype=int)
        cross = ()
        if Phi is not None:
            cross = (toroidal(uv, g0), jnp.ones((uv.shape[0], len(planes)), dtype=int),
                     jnp.full((uv.shape[0], len(planes), n_cross, 5), jnp.nan, uv.dtype))
        # the step count in the default integer (int64 in float64 mode, as jnp.arange in trace): an int32 count
        # divided by S is float32 even in float64 mode
        carry = (uv, g0, a0, jnp.zeros(a0.shape, jnp.int32), jnp.where(out(uv), 0, never), acc, acc, cross)
        carry, rec = jax.lax.scan(period, carry, None, length=n_periods)
        rates = (carry[5] / denom / TWO_PI).T
        out_ = (jnp.moveaxis(rec, 0, 1), rates, carry[4])
        if Phi is not None:
            out_ = out_ + (carry[7][2], carry[7][1] - 1)
        return out_

    return jax.vmap(per_field)(coefs, uv0, track)


@partial(jax.jit, static_argnames=("field", "steps_per_period", "planes"))
def _refine_crossings(field, coef, Phi, brackets, reached, direction, steps_per_period, planes):
    """The crossings of the lines of one field with the ``planes`` from the steps :func:`_reduced_trace` kept
    (``brackets`` ``(lines, planes, n_cross, 5)``, ``reached``): Newton's method on the length of a partial step of
    the tracer from the start of the crossing step, so the crossing has the tracer's accuracy. Returns the ``(u,
    v)`` ``(lines, planes, n_cross, 2)`` and the logical ``zeta`` ``(lines, planes, n_cross)``, NaN for the
    crossings not reached."""
    rhs = cross_section_rhs(field)
    S = steps_per_period
    n_cross = brackets.shape[2]
    level = (direction[:, None, None] * jnp.asarray(planes)[None, :, None]
             + jnp.arange(1, n_cross + 1)[None, None, :])                               # (lines, planes, n_cross)
    sign = jnp.broadcast_to(direction[:, None, None], level.shape)

    def one(b, L, d):
        y0, z0, F0, F1 = b[:2], b[2], b[3], b[4]
        h = d / S

        def g(t):
            return d * _toroidal_periods(Phi, _tsit5_step(rhs, z0, y0, t, coef), z0 + t) - L
        t = h * (L - F0) / (F1 - F0)
        for _ in range(CROSSING_NEWTON_STEPS):
            val, slope = jax.jvp(g, (t,), (jnp.ones_like(t),))
            t = t - val / slope
        return _tsit5_step(rhs, z0, y0, t, coef), z0 + t

    flat = (brackets.reshape(-1, 5), level.reshape(-1), sign.reshape(-1))
    uv, z = jax.lax.map(lambda a: one(*a), flat, batch_size=65536)
    ok = (jnp.arange(1, n_cross + 1)[None, None, :] <= reached[:, :, None]).reshape(-1)
    uv = jnp.where(ok[:, None], uv, jnp.nan).reshape(level.shape + (2,))
    z = jnp.where(ok, z, jnp.nan).reshape(level.shape)
    return uv, z


@partial(jax.jit, static_argnames=("field", "n_periods", "steps_per_period"))
def _drift_many(field, coefs, uv0, n_periods, steps_per_period):
    """The step-size check of several fields at once: per field, the largest ``(u, v)`` distance between the
    lines from ``uv0`` ``(fields, lines, 2)`` traced at ``steps_per_period`` and at twice that, at every step of
    the coarse trace, over the lines that neither trace lost, and ``-inf`` when every line was lost. Both traces run
    in one batch: the coarse lines take their step on every second half step and a step of length zero between."""
    rhs = cross_section_rhs(field)
    S = steps_per_period

    def per_field(coef, uv):
        m = uv.shape[0]

        def step(carry, j):
            y, dmax, bad = carry
            coarse = j % 2 == 0
            t = jnp.concatenate([jnp.full(m, (j // 2) / S), jnp.full(m, j / (2 * S))])
            hh = jnp.concatenate([jnp.full(m, jnp.where(coarse, 1.0 / S, 0.0)), jnp.full(m, 0.5 / S)])
            y = jax.vmap(lambda yy, tt, hs: _tsit5_step(rhs, tt, yy, hs, coef))(y, t, hh)
            r = jnp.sqrt(y[:, 0] ** 2 + y[:, 1] ** 2)
            bad = bad | (r >= R_MAX) | ~jnp.isfinite(r)
            d = jnp.linalg.norm(y[:m] - y[m:], axis=-1)
            return (y, jnp.where(coarse, dmax, jnp.maximum(dmax, d)), bad), None

        y0 = jnp.concatenate([uv, uv])
        (_, dmax, bad), _ = jax.lax.scan(step, (y0, jnp.zeros(m), jnp.zeros(2 * m, bool)),
                                         jnp.arange(2 * n_periods * S))
        good = ~(bad[:m] | bad[m:])
        return jnp.max(jnp.where(good, dmax, -jnp.inf))

    return jax.vmap(per_field)(coefs, uv0)


# ---------------------------------------------------------------------------
# The section
# ---------------------------------------------------------------------------

def padding_seeds(n, r_min, r_max=R_EDGE, *, seed=0):
    """``n`` logical ``(r, theta)`` seeds filling the annulus ``r_min < r < r_max`` (the padding between the plasma
    and the boundary, and the plasma edge): a low-discrepancy set, uniform in ``r`` and in ``theta``, so that a
    region whose lines leave quickly is still covered by many short traces."""
    k = np.arange(n) + 0.5
    r = r_min + (r_max - r_min) * (k / n)
    theta = (k * (np.sqrt(5.0) - 1.0) / 2.0 + np.random.default_rng(seed).uniform()) % 1.0
    return jnp.stack([jnp.asarray(r), jnp.asarray(theta)], axis=1)


def planes_for(seq, planes):
    """The section planes in units of one field period: logical ``zeta = plane`` for sections ``"zeta"``, physical
    ``phi = 2 pi plane / nfp`` for sections ``"phi"``. A sequence ``planes`` is returned as is. An integer gives
    that many equispaced planes: over half a period, both ends included, for a stellarator-symmetric map, and over
    the whole period otherwise."""
    if not isinstance(planes, int):
        return tuple(float(p) for p in planes)
    if seq.symmetry == "stellarator":
        return tuple(np.linspace(0.0, 0.5, planes).tolist())
    return tuple((np.arange(planes) / planes).tolist())


def steps_for(planes, min_steps=MIN_STEPS_PER_PERIOD):
    """The number of steps per field period: the smallest multiple of the planes' common denominator that is at
    least ``min_steps``. Every plane ``zeta = const`` then falls exactly on the end of a step."""
    den = reduce(lambda a, b: a * b // gcd(a, b),
                 (Fraction(float(p)).limit_denominator(4096).denominator for p in planes), 1)
    return den * -(-min_steps // den)


def _on_step_ends(seq, sections):
    """Whether the crossings of the ``sections`` (``"zeta"`` or ``"phi"``) are states at the ends of steps: always
    for ``"zeta"``, and for ``"phi"`` on a map whose sections ``zeta = const`` are the planes ``phi = const``. That
    is a map in the cylindrical frame with ``X3 = 0``, the map of every file but a GVEC state with its own frame."""
    from mrx.equilibria import _frame_arrays, cylindrical_frame  # noqa: PLC0415  (imports the diagnostics' users)
    Phi = seq.map
    frame = _frame_arrays(cylindrical_frame(), Phi.nfp)
    planes_of_phi = (all(np.array_equal(np.asarray(a), np.asarray(b))
                         for a, b in zip((Phi.frame_cos, Phi.frame_sin, Phi.frame_n), frame))
                     and not np.asarray(Phi.raw_X3).any())
    return {"zeta": True, "phi": planes_of_phi}[sections]


#: Newton steps that put a crossing on its plane ``phi = const``, each a partial step of the tracer from the start
#: of the step that crosses.
CROSSING_NEWTON_STEPS = 4


def _toroidal_periods(Phi, uv, zeta):
    """The physical toroidal angle of the point ``(u, v)`` at the logical ``zeta`` of a line, in field periods and
    continuous along the line: ``zeta`` plus the difference to it, wrapped to half a period. The sections ``zeta =
    const`` of the map must stay within half a period of the planes ``phi = const``."""
    r, theta = to_polar(uv)
    P = Phi(jnp.stack([r, theta, zeta % 1.0]))
    d = jnp.arctan2(P[1], P[0]) * Phi.nfp / TWO_PI - zeta
    return zeta + (d + 0.5) % 1.0 - 0.5


@jax.jit
def _cylindrical(Phi, uv, zeta):
    """The cylindrical ``(R, Z)`` of ``(u, v)`` points at the logical ``zeta``."""
    r, theta = to_polar(uv)
    P = jax.vmap(Phi)(jnp.stack([r, theta, zeta % 1.0], axis=-1).reshape(-1, 3)).reshape(uv.shape[:-1] + (3,))
    return jnp.hypot(P[..., 0], P[..., 1]), P[..., 2]


@jax.jit
def _in_section(Phi, uv, zeta):
    """The coordinates of ``(u, v)`` points in their section ``zeta = const`` (``Phi.section``: cylindrical ``(R,
    Z)``, or ``(X1, X2)`` in the plane of a G-frame)."""
    r, theta = to_polar(uv)
    x = jnp.stack([r, theta, jnp.broadcast_to(zeta % 1.0, r.shape)], axis=-1)
    xy = jax.vmap(Phi.section)(x.reshape(-1, 3)).reshape(x.shape[:-1] + (2,))
    return xy[..., 0], xy[..., 1]


def _seeds_many(field, coefs, n_lines, steps_per_period, seed):
    """The seeds of several fields at once: ``(seeds, track)``, the seeds ``(fields, n_lines + 1, 2)`` and the
    magnetic axis at every step of a period ``(fields, steps_per_period, 2)``. Seed 0 lies :data:`R_AXIS` from the
    magnetic axis. The others are spread from the magnetic axis to radius :data:`R_EDGE`, each towards its own
    random poloidal angle (random key ``seed``), so that the seeds do not all line up with the X-points of an
    island chain. The axis is located by two short probes, the second about the first's estimate, and ``track``
    is the mean over the second probe's periods of its line next to the axis. The winding of
    :func:`_reduced_trace` is counted about it."""
    n_fields = jax.tree.leaves(coefs)[0].shape[0]
    S = steps_per_period
    probe = jnp.array([[R_AXIS, 0.0], [R_EDGE, 0.0]])
    ys = _trace_saved_many(field, coefs, jnp.broadcast_to(to_uv(probe[:, 0], probe[:, 1]), (n_fields, 2, 2)),
                           PROBE_PERIODS, S)
    centre = jnp.mean(ys[:, 0, ::S], axis=1)                                          # (fields, 2)
    first = jnp.stack(to_polar(centre + jnp.array([R_AXIS, 0.0])), axis=-1)            # (fields, 2)
    probe2 = jnp.stack([first, jnp.broadcast_to(jnp.array([R_EDGE, 0.0]), first.shape)], axis=1)
    ys = _trace_saved_many(field, coefs, to_uv(probe2[..., 0], probe2[..., 1]), PROBE_PERIODS, S)
    centre = jnp.mean(ys[:, 0, ::S], axis=1)
    track = jnp.mean(ys[:, 0, :-1].reshape(n_fields, -1, S, 2), axis=1)
    thetas = jax.random.uniform(jax.random.PRNGKey(seed), (n_lines,))
    edge = to_uv(R_EDGE, thetas)
    t = jnp.linspace(T_MIN, 1.0, n_lines)[None, :, None]
    uv = centre[:, None, :] + t * (edge[None] - centre[:, None, :])
    return jnp.concatenate([probe2[:, :1], jnp.stack(to_polar(uv), axis=-1)], axis=1), track


def _rates_iota(rates, nfp):
    """``(iota, convergence)`` per line from the winding rates of :func:`_reduced_trace` ``(lines, 3)``: the
    rotational transform in poloidal turns per full toroidal turn (non-negative) and ``|iota(first half) -
    iota(second half)|``, which is small only on lines that have an iota."""
    r = np.abs(np.asarray(rates)) * nfp
    return r[:, 0], np.abs(r[:, 1] - r[:, 2])


def _step_end_crossings(rec, steps_per_period, planes, offsets, n_cross, direction):
    """The crossings ``n = 1 .. n_cross`` of every plane that falls on step ends, the states kept by
    :func:`_reduced_trace` (``rec`` ``(lines, periods, len(offsets), 2)``), as ``{plane: (uv, z)}`` with ``uv``
    ``(lines, n_cross, 2)`` and the logical ``z`` ``(lines, n_cross)``."""
    S = steps_per_period
    where = {o: i for i, o in enumerate(offsets)}
    out = {}
    for plane in planes:
        k = int(round(plane * S)) % S
        n = np.arange(1, n_cross + 1)
        g = n * S + k if direction > 0 else n * S - k                  # the step whose end is the crossing
        period = (g - 1) // S
        slot = np.array([where[o] for o in g - period * S], dtype=int)
        uv = rec[:, period, slot]
        out[plane] = (uv, jnp.broadcast_to(jnp.asarray(direction * g / S), uv.shape[:2]))
    return out


def _batch_sections(seq, sections, crossings, escape, steps_per_period, direction, axis=None, seeds_uv=None):
    """The sections from the crossings ``{plane: (uv, z)}`` of the lines (``uv`` ``(lines, n_cross, 2)``, the
    logical ``z``), as dicts with the section coordinates ``R``, ``Z`` (:func:`_in_section` for ``sections =
    "zeta"``, cylindrical for ``"phi"``) and the logical ``logr``, ``logth``, ``logz``. They are NaN from the step
    ``escape`` on and where a crossing was not reached. With ``axis`` ``{plane: (uv, z)}``, the crossings of the
    axis line, they hold ``axisR``, ``axisZ``, and with ``seeds_uv`` a first column holding the seed where it
    lies on the plane (within half a step, NaN elsewhere)."""
    S = steps_per_period
    coords = {"zeta": _in_section, "phi": _cylindrical}[sections]
    out = {}
    for plane, (uv, z) in crossings.items():
        R, Z = coords(seq.map, uv, z)
        r, theta = to_polar(uv)
        z = np.asarray(z)
        sec = {"R": np.asarray(R), "Z": np.asarray(Z), "logr": np.asarray(r), "logth": np.asarray(theta),
               "logz": z % 1.0}
        gone = ~(direction * z < np.asarray(escape, float)[:, None] / S)          # NaN z (not reached) too
        for key in sec:
            sec[key] = np.where(gone, np.nan, sec[key])
        if seeds_uv is not None:
            zero = jnp.zeros(seeds_uv.shape[0])
            F0 = (np.asarray(jax.vmap(lambda y: _toroidal_periods(seq.map, y, 0.0))(seeds_uv))
                  if sections == "phi" else np.zeros(seeds_uv.shape[0]))
            on = np.abs((F0 - plane + 0.5) % 1.0 - 0.5) < 0.5 / S
            R0, Z0 = (np.asarray(v) for v in coords(seq.map, seeds_uv, zero))
            r0, th0 = (np.asarray(v) for v in to_polar(seeds_uv))
            for key, val in (("R", R0), ("Z", Z0), ("logr", r0), ("logth", th0), ("logz", np.zeros_like(r0))):
                sec[key] = np.concatenate([np.where(on, val, np.nan)[:, None], sec[key]], axis=1)
        if axis is not None:
            aR, aZ = coords(seq.map, *axis[plane])
            sec.update(axisR=np.asarray(aR), axisZ=np.asarray(aZ))
        out[plane] = sec
    return out


def poincare(seq, dof, *, lines=160, periods=400, planes=5, seed=0, name="field", reseed=0, padding=None,
             sections="phi", min_steps=MIN_STEPS_PER_PERIOD):
    """The Poincare sections of the field given by the 2-form ``dof`` on ``seq``, a Dirichlet 2-form or, with a flux
    through the wall, a free one (:func:`logical_field`). The lines of a free one can leave through the wall.

    ``lines`` field lines (:func:`_seeds_many`) are traced for ``periods`` field periods with at least ``min_steps``
    steps per period and cut at the ``planes`` (a count or the planes themselves, see :func:`planes_for`).
    ``sections`` chooses the planes. With ``"phi"``, the default, they are the planes ``phi = const`` of the physical
    toroidal angle and the coordinates are cylindrical ``(R, Z)``. With ``"zeta"`` they are the map's sections ``zeta
    = const``, the crossings are states at the ends of steps and their coordinates are ``Phi.section``'s
    (cylindrical ``(R, Z)``, or ``(X1, X2)`` in the plane of a G-frame). On a map in the cylindrical frame with ``X3
    = 0`` (a file's map with ``zeta="phi"``) the two are the same planes (:func:`_on_step_ends`). On other maps, the
    default sections normal to the axis among them, a ``"phi"`` crossing is found by Newton's method on a partial
    step of the tracer, and every line must cross every plane once per period.

    A line that reaches the domain boundary is not ``keep``. Its crossings up to the boundary are kept and the
    later ones are NaN. With ``reseed`` > 0, up to that many further rounds of ``lines`` lines are traced on the
    radii of the lines the previous round lost, with fresh poloidal angles, so that the chaotic region up to the
    boundary is populated. All rounds are returned together, the first ``lines`` entries being the first round.
    With ``padding = (n, periods, r_min)``, ``n`` further lines seeded uniformly over ``r_min < r < R_EDGE``
    (:func:`padding_seeds`) are traced forwards and backwards for that many periods and returned under
    ``"padding"`` with the same per-line and per-plane entries, so that the region between the plasma and the
    boundary, whose lines leave within a few periods, is covered densely by short traces. ``name`` only labels
    error messages. Returns a dict with

    * per line: ``iota``, ``seed_r`` (the logical radius of the seed), ``keep`` (the line stayed inside the
      domain to the end), ``chaotic`` (the iota of the two halves of the trace differ by more than
      :data:`CHAOS_TOL_PER_PERIOD` ``/ periods``), ``shown`` (``keep & ~chaotic``), ``round`` (0 for the
      first seeding) and ``escape_period`` (the period at which the line reached the boundary, ``inf`` if never).
    * ``sections[plane]``: a dict with the section coordinates ``R``, ``Z`` and the logical ``logr``, ``logth``,
      ``logz`` of the crossings, arrays of shape ``(line, periods - 1)``, and ``axisR``, ``axisZ`` of the magnetic
      axis.
    * ``drift``: how far a few regular lines move, in the logical ``(u, v)`` chart, when the step is halved. It is
      the check that the step count is sufficient, and it is NaN when no line is regular. It is only measured on
      regular lines because on a chaotic line it would measure the divergence of neighbouring lines instead.
    * ``steps`` (per period), ``nfp``, ``bz_over_b`` (the range of ``B^zeta/|B|``) and ``walltime`` (seconds of
      the main trace).
    """
    return poincare_many(seq, [dof], lines=lines, periods=periods, planes=planes, seed=seed, names=[name],
                         reseed=reseed, padding=padding, sections=sections, min_steps=min_steps)[0]


def poincare_many(seq, dofs, *, lines=160, periods=400, planes=5, seed=0, names=None, reseed=0, padding=None,
                  sections="phi", min_steps=MIN_STEPS_PER_PERIOD):
    """:func:`poincare` of several fields on ``seq`` at once (a list of 2-forms ``dofs``): a list of results. All
    fields and all their lines go through one batched trace (:func:`_reduced_trace`), in batches of at most
    :data:`MAX_BATCH_LINES` lines, so a movie's frames cost about one frame's trace. Where the crossings fall on
    step ends (:func:`_on_step_ends`) the trace keeps the states there. Otherwise it keeps the steps that cross and
    :func:`_refine_crossings` finds the crossings in them."""
    dofs = [jnp.asarray(d) for d in dofs]
    names = list(names) if names is not None else [f"field {i}" for i in range(len(dofs))]
    n_pad, periods_pad, r_min = padding if padding is not None else (0, 0, 0.0)
    per_field = lines + 1 + 2 * n_pad
    group = max(1, MAX_BATCH_LINES // per_field)
    if len(dofs) > group:
        return [res for a in range(0, len(dofs), group)
                for res in poincare_many(seq, dofs[a:a + group], lines=lines, periods=periods, planes=planes,
                                         seed=seed, names=names[a:a + group], reseed=reseed, padding=padding,
                                         sections=sections, min_steps=min_steps)]
    fields = [logical_field(seq, d) for d in dofs]
    field, nfp = fields[0], seq.nfp       # the trace evaluates field.at, the same for both spaces
    planes = planes_for(seq, planes)
    S = steps_for(planes, min_steps)
    offsets = _record_offsets(planes, S)
    planar = _on_step_ends(seq, sections)
    bz = [require_zeta_parameterisation(fd, d, n) for fd, d, n in zip(fields, dofs, names)]
    coefs = jax.tree.map(lambda *a: jnp.stack(a), *[fd.coefficients(d) for fd, d in zip(fields, dofs)])
    n_fields = len(dofs)
    seeds, track = _seeds_many(field, coefs, lines, S, seed)

    def trace_batch(uv, direction, n_saves, n_periods):
        """The batched trace, and off step ends the crossing steps for crossings 1 .. n_periods - 1."""
        extra = {} if planar else dict(Phi=seq.map, planes=planes, n_cross=n_periods - 1)
        out = _reduced_trace(field, coefs, uv, direction, track, n_saves, n_periods, S, offsets, **extra)
        return out if not planar else out + (None, None)

    def crossings(f, rec_rows, brackets_rows, reached_rows, n_cross, direction):
        """``{plane: (uv, z)}`` of the crossings 1 .. n_cross of some lines of field ``f``."""
        if planar:
            return _step_end_crossings(rec_rows, S, planes, offsets, n_cross, direction)
        dirs = jnp.full(brackets_rows.shape[0], float(direction))
        uv, z = _refine_crossings(field, jax.tree.map(lambda a: a[f], coefs), seq.map,
                                  brackets_rows[:, :, :n_cross], reached_rows, dirs, S, planes)
        return {plane: (uv[:, j], z[:, j]) for j, plane in enumerate(planes)}

    def cut(*args, **kw):
        return _batch_sections(seq, sections, *args, **kw)

    # one batch: the axis line and the seeds forwards, the padding seeds forwards and backwards
    pad = padding_seeds(n_pad, r_min, seed=seed) if n_pad else jnp.zeros((0, 2))
    pad_f = jnp.broadcast_to(pad, (n_fields, n_pad, 2))
    starts = jnp.concatenate([seeds, pad_f, pad_f], axis=1)
    direction = jnp.concatenate([jnp.ones(lines + 1 + n_pad), -jnp.ones(n_pad)])
    n_saves = jnp.asarray([periods * S + 1] * (lines + 1) + [periods_pad * S + 1] * (2 * n_pad))
    n_main = max(periods, periods_pad)
    t0 = time.perf_counter()
    rec, rates, esc, brackets, reached = jax.block_until_ready(
        trace_batch(to_uv(starts[..., 0], starts[..., 1]), direction, n_saves, n_main))
    walltime = time.perf_counter() - t0
    esc = np.asarray(esc)

    # further rounds on the radii of the lost lines, cycled to fill a round (so the compiled trace is reused), with
    # fresh poloidal angles, each field until a round loses none
    lim = periods * S
    lost = [np.asarray(seeds[f, 1:, 0])[esc[f, 1:lines + 1] <= lim] for f in range(n_fields)]
    rounds = []
    for k in range(reseed):
        active = np.array([v.size > 0 for v in lost])
        if not active.any():
            break
        theta = jax.random.uniform(jax.random.PRNGKey(seed + 1 + k), (lines,))
        r = np.stack([np.resize(v, lines) if v.size else np.full(lines, 0.5) for v in lost])
        new = jnp.stack([jnp.asarray(r), jnp.broadcast_to(theta, (n_fields, lines))], axis=-1)
        rr, ww, ee, bb, nn = trace_batch(to_uv(new[..., 0], new[..., 1]), jnp.ones(lines),
                                         jnp.full(lines, periods * S + 1), periods)
        ee = np.asarray(ee)
        rounds.append((new, rr, ww, ee, active, bb, nn))
        lost = [np.asarray(new[f, :, 0])[ee[f] <= lim] if active[f] else np.zeros(0) for f in range(n_fields)]

    def rows(a, f, sl, n_per=None):                     # a field's rows of a batch output, None passed through
        if a is None:
            return None
        return a[f, sl] if n_per is None else a[f, sl, :n_per]

    results = []
    for f in range(n_fields):
        mine = [ro for ro in rounds if ro[4][f]]
        s_f = jnp.concatenate([seeds[f]] + [ro[0][f] for ro in mine])
        rates_f = jnp.concatenate([rates[f, :lines + 1]] + [ro[2][f] for ro in mine])
        esc_f = np.minimum(np.concatenate([esc[f, :lines + 1]] + [ro[3][f] for ro in mine]), lim + 1)
        escaped = esc_f <= lim
        iota, conv = _rates_iota(rates_f, nfp)
        chaotic = conv > CHAOS_TOL_PER_PERIOD / periods
        keep = ~escaped[1:]
        main = slice(0, lines + 1)
        parts = [crossings(f, rows(rec, f, main, periods), rows(brackets, f, main), rows(reached, f, main),
                           periods - 1, 1)]
        for ro in mine:
            parts.append(crossings(f, ro[1][f], None if planar else ro[5][f], None if planar else ro[6][f],
                                   periods - 1, 1))
        cr = {plane: tuple(jnp.concatenate([p[plane][c] for p in parts])[1:] for c in (0, 1)) for plane in planes}
        axis = {plane: (parts[0][plane][0][0], parts[0][plane][1][0]) for plane in planes}
        secs = cut(cr, esc_f[1:], S, 1, axis=axis)
        out = {"iota": iota[1:], "seed_r": np.asarray(s_f[1:, 0]), "keep": keep, "chaotic": chaotic[1:],
               "shown": keep & ~chaotic[1:], "escape_period": np.where(keep, np.inf, esc_f[1:] / S),
               "sections": secs,
               "round": np.concatenate([np.zeros(lines, int)] + [np.full(lines, k + 1) for k in range(len(mine))]),
               "steps": S, "nfp": nfp, "bz_over_b": bz[f], "walltime": walltime}
        if n_pad:
            a, b = lines + 1, lines + 1 + n_pad
            lim_p = periods_pad * S
            esc_pf = np.minimum(esc[f, a:b], lim_p + 1)
            esc_pb = np.minimum(esc[f, b:], lim_p + 1)
            axis_p = {plane: (cr_ax[0][:periods_pad - 1], cr_ax[1][:periods_pad - 1])
                      for plane, cr_ax in axis.items()} if periods_pad <= periods else None
            if axis_p is None:                    # the axis line's crossings beyond the main trace's
                full = crossings(f, rows(rec, f, slice(0, 1), periods_pad), rows(brackets, f, slice(0, 1)),
                                 rows(reached, f, slice(0, 1)), periods_pad - 1, 1)
                axis_p = {plane: (full[plane][0][0], full[plane][1][0]) for plane in planes}
            fwd = cut(crossings(f, rows(rec, f, slice(a, b), periods_pad),
                                                 rows(brackets, f, slice(a, b)), rows(reached, f, slice(a, b)),
                                                 periods_pad - 1, 1),
                                  esc_pf, S, 1, axis=axis_p, seeds_uv=to_uv(pad[:, 0], pad[:, 1]))
            back = cut(crossings(f, rows(rec, f, slice(b, None), periods_pad),
                                                  rows(brackets, f, slice(b, None)), rows(reached, f, slice(b, None)),
                                                  periods_pad - 1, -1),
                                   esc_pb, S, -1)
            for plane, sec in fwd.items():
                for key in ("R", "Z", "logr", "logth", "logz"):
                    sec[key] = np.concatenate([sec[key], back[plane][key]], axis=1)
            iota_p, conv_p = _rates_iota(rates[f, a:b], nfp)
            chaotic_p = conv_p > CHAOS_TOL_PER_PERIOD / periods_pad
            keep_f, keep_b = esc_pf > lim_p, esc_pb > lim_p
            escape_p = np.where(keep_f, np.inf, esc_pf / S)
            escape_p = np.where(keep_f & keep_b, np.inf,
                                np.minimum(escape_p, np.where(keep_b, np.inf, esc_pb / S)))
            keep_p = keep_f & keep_b
            out["padding"] = {"iota": iota_p, "seed_r": np.asarray(pad[:, 0]), "keep": keep_p, "chaotic": chaotic_p,
                              "shown": keep_p & ~chaotic_p, "escape_period": escape_p, "sections": fwd}
        results.append(out)

    # the step-size check: up to DRIFT_LINES regular lines of the first round per field, retraced at half the step
    n_drift = min(periods, DRIFT_PERIODS)
    picks = []
    for f, out in enumerate(results):
        regular = np.flatnonzero(out["shown"][:lines]) + 1                  # rows of the main batch
        picks.append(regular[np.unique(np.linspace(0, len(regular) - 1,
                                                   min(DRIFT_LINES, len(regular))).round().astype(int))]
                     if regular.size else np.zeros(0, int))
    if any(p.size for p in picks):
        sel = np.stack([np.resize(p, DRIFT_LINES) if p.size else np.zeros(DRIFT_LINES, int) for p in picks])
        picked = seeds[jnp.arange(n_fields)[:, None], sel]                          # (fields, DRIFT_LINES, 2)
        drift = np.asarray(_drift_many(field, coefs, to_uv(picked[..., 0], picked[..., 1]), n_drift, S))
    for f, out in enumerate(results):
        out["drift"] = float(drift[f]) if picks[f].size else float("nan")
    return results


def trace_archive(seq, fields, *, lines=160, periods=400, planes=5, seed=0, source="", reseed=0, padding=None,
                  sections="phi", min_steps=MIN_STEPS_PER_PERIOD):
    """Trace several fields on ``seq`` into one archive and return ``(archive, results)``.

    ``fields`` maps a name to ``(B, step)``, with ``B`` the odd Dirichlet 2-form of the field and ``step`` the
    relaxation step it belongs to. All fields are traced together by :func:`poincare_many` with the same
    ``lines``, ``periods``, ``planes``, ``seed``, ``reseed`` rounds, ``padding`` lines (archived under
    ``<name>_pad_...``), ``sections`` and ``min_steps``, and ``results`` maps each name to its result. The weak
    pressure of each field (:func:`mrx.relaxation.physics.weak_pressure`, two solves) is evaluated at every
    crossing. ``archive`` is a flat dict of arrays, the content of the ``trace.npz`` of
    ``scripts/poincare_trace.py`` (see there for the keys), and ``source`` describes the run in it.
    """
    planes = planes_for(seq, planes)
    volume = float(jnp.sum(seq.quad.w * seq.jacobian_j))

    @jax.jit
    def weak_at(pd, x):
        return jax.vmap(DiscreteFunction(pd, seq.basis_0, seq.even.E(0)))(x)[:, 0]

    labels = seq.map.section_labels if sections == "zeta" else ("R", "Z")
    archive = {"fields": np.array(list(fields)), "planes": np.array(planes), "resolution": np.array(seq.ns),
               "p": seq.p, "nfp": seq.nfp, "symmetry": np.array(seq.symmetry), "steps": steps_for(planes, min_steps),
               "sections": np.array(sections), "section_labels": np.array(labels), "source": np.array(source)}
    batched = poincare_many(seq, [B for B, _ in fields.values()], lines=lines, periods=periods, planes=planes,
                            seed=seed, names=list(fields), reseed=reseed, padding=padding, sections=sections,
                            min_steps=min_steps)
    results = dict(zip(fields, batched))
    for name, (B, step) in fields.items():
        B = jnp.asarray(B)
        pw = jnp.asarray(weak_pressure(current(B, seq), B, seq))
        res = results[name]
        archive[f"{name}_label"] = np.array(f"{name} (step {step})")
        archive[f"{name}_step"] = step
        archive[f"{name}_bsq"] = float(seq.odd.l2_norm_sq(B, 2)) / volume
        for prefix, r in ((name, res), (name + "_pad", res.get("padding"))):
            if r is None:
                continue
            for key in ("iota", "seed_r", "keep", "chaotic", "shown", "drift", "round", "escape_period"):
                if key in r:
                    archive[f"{prefix}_{key}"] = np.asarray(r[key])
            for plane, sec in r["sections"].items():
                tag = f"{prefix}_zeta{plane:g}"
                for key, arr in sec.items():
                    archive[f"{tag}_{key}"] = arr
                x = jnp.stack([jnp.asarray(sec["logr"]).ravel(), jnp.asarray(sec["logth"]).ravel(),
                               jnp.asarray(sec["logz"]).ravel()], axis=1)
                archive[f"{tag}_pressure"] = np.asarray(weak_at(pw, x)).reshape(sec["logr"].shape)
        shown = res["shown"]
        span = (f"iota {float(res['iota'][shown].min()):.4f}..{float(res['iota'][shown].max()):.4f}"
                if shown.any() else "no line converged")
        rounds = np.asarray(res["round"])
        per_round = ", ".join(f"round {k}: {int((~res['keep'][rounds == k]).sum())}/{int((rounds == k).sum())} lost"
                              for k in range(int(rounds.max()) + 1))
        print(f"[{name}] B^zeta/|B| in [{res['bz_over_b'][0]:+.3e}, {res['bz_over_b'][1]:+.3e}], "
              f"trace {res['walltime']:.1f}s, {int((~res['keep']).sum())}/{res['keep'].size} lost ({per_round}), "
              f"{int((res['keep'] & res['chaotic']).sum())} chaotic, drift {res['drift']:.2e}, {span}"
              + (f", padding: {int((~res['padding']['keep']).sum())}/{res['padding']['keep'].size} lost, "
                 f"{int((res['padding']['keep'] & res['padding']['chaotic']).sum())} chaotic"
                 if "padding" in res else ""), flush=True)
    return archive, results


def locked_width(res, iota, tol=2e-3):
    """Return ``(width, n_locked)`` for the rotational transform ``iota`` in ``res``, the result of
    :func:`poincare`. The locked lines are the kept lines whose ``|iota|`` lies within ``tol`` of it, and
    ``width`` is ``max(r) - min(r)`` of their seed radii (0 with fewer than two). This is a quick island width
    with the resolution of the seed spacing."""
    locked = np.asarray(res["keep"]) & (np.abs(np.abs(np.asarray(res["iota"])) - iota) < tol)
    r = np.asarray(res["seed_r"])[locked]
    return (float(r.max() - r.min()) if r.size > 1 else 0.0), int(locked.sum())
