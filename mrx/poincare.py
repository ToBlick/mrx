r"""Poincaré sections of a discrete field, parameterised by the toroidal angle.

Three things differ from the arclength field-line tracer MRX used before, and
each one removes a specific error source rather than trading one for another.

**1. The independent variable is the toroidal angle, not arclength.**
The field line satisfies :math:`d\hat x/ds \propto \hat B` for any
parameterisation, so dividing through by the third contravariant component
gives

.. math:: dr/d\zeta = \hat B^r/\hat B^\zeta, \qquad
          d\theta/d\zeta = \hat B^\theta/\hat B^\zeta ,

a *non-autonomous two-dimensional* system whose independent variable is the
section coordinate itself.  Crossings of the plane :math:`\zeta = \zeta_0` then
occur at :math:`\zeta = \zeta_0 + m` exactly -- they are integration times, not
roots to be hunted.  Nothing is detected and nothing is interpolated, so the
phase error an arclength integrator accumulates over thousands of turns (each
crossing located to the tolerance of a bracketed root, each error fed into the
next) does not exist.  The reparameterisation is exact wherever
:math:`\hat B^\zeta \ne 0`, which for a toroidal field is everywhere.

**2. The step schedule is prescribed, so lanes do not couple.**
``diffrax`` adaptive controllers run a whole ``vmap``ed batch on the *smallest*
step any lane asks for: one seed in a chaotic edge region drags the entire
batch down, which is why the old code chunked into groups of eight and still
paid for the worst seed in each group.  With :math:`\zeta` as the independent
variable the natural step is a fixed fraction of a field period -- geometry-
uniform, unlike arclength -- so ``StepTo`` can prescribe the whole schedule up
front.  Every lane then executes the same number of identical-cost steps, the
batch is one ``vmap``, and a pathological seed costs what a healthy one costs.
Every step endpoint is kept, so a section plane is a column of the result and
the planes decide the step count (:func:`steps_for`).  The ``drift`` of
:func:`poincare` is the price: fixed steps have no error control, so the step
count has to be *justified* by refinement (h against h/2) instead of assumed.

**3. The state is a Cartesian chart on the cross-section.**
:math:`\hat B^\theta \sim 1/r` near the polar axis (the coordinate vector
:math:`\partial_\theta` has length :math:`O(r)`), so :math:`d\theta/d\zeta`
diverges at the origin and the innermost seeds -- the ones that resolve the
axis and the low-shear core -- are exactly the ones an integrator handles
worst.  In :math:`(u, v) = (r\cos 2\pi\theta, r\sin 2\pi\theta)` the
:math:`1/r` cancels against the :math:`O(r)` length of the same coordinate
vector and the right-hand side is bounded through the origin.
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

TWO_PI = 2.0 * jnp.pi

#: Field lines are frozen once they reach this logical radius.  The spline maps
#: are genuinely singular at ``r = 1`` (``det DF = 0`` at the outer knot), so
#: the domain has to stop just short of it.
R_MAX = 1.0 - 1e-6

#: Fewest integration steps per field period. Tsit5 at 24 steps per period
#: puts the h/2 drift of a regular li383 line at 1e-3 of the minor radius
#: over 64 periods (the ``drift`` every trace reports); the section planes
#: raise the count to a multiple of their denominator (:func:`steps_for`).
MIN_STEPS_PER_PERIOD = 24

#: Steps per period for the tangent map of the return map (:func:`fixed_points`):
#: the variational equation needs more than the trajectory. On the seeded li383
#: (6,1) chain at (10,16,16) the O-point's residue moved 5% from 24 to 48 steps
#: and 1% from 48 to 96, where ``det S`` is 1 to 5e-5 (2026-09-18).
TANGENT_STEPS_PER_PERIOD = 96


# ---------------------------------------------------------------------------
# The field
# ---------------------------------------------------------------------------

def logical_field(seq, k, dirichlet):
    r"""``(x, dof) ->`` contravariant logical components of the vector field
    behind the k-form with coefficients ``dof``.

    A 2-form pushes forward by Piola, :math:`B = DF\,\hat B/J`, so its
    coefficients *are* the contravariant components and the field-line
    direction in logical space is :math:`\hat B` itself.  A 1-form pushes
    forward as :math:`v = DF^{-T}\hat A`, so the logical direction is
    :math:`DF^{-1} v = g^{-1}\hat A` with :math:`g = DF^T DF`.

    Only the direction matters below -- the third component divides out -- so
    no Jacobian factor is applied.

    The coefficients are an ARGUMENT of the returned function, not a closure
    constant, and every tracer below takes them as one: the compiled
    integrator is keyed on the function object (one per sequence and degree)
    and the shapes, so a movie of a relaxation run compiles the trace ONCE and
    every further field is pure execution. Closing over the coefficients
    instead baked a few-hundred-thousand-float constant into every trace and
    recompiled the whole integrator per field (five compiles per field:
    the two seed probes, the trace, the two drift traces). Cached per
    ``(seq, k, dirichlet)`` so that the function object IS one per sequence.
    """
    return _logical_field(seq, int(k), bool(dirichlet))


@lru_cache(maxsize=None)
def _logical_field(seq, k, dirichlet):
    if k not in (1, 2):
        raise ValueError(f"logical_field: k must be 1 or 2, got {k}")
    basis = seq.basis_2 if k == 2 else seq.basis_1
    extraction = seq.E(k, dirichlet)

    if k == 2:
        def field(x, dof):
            return DiscreteFunction(dof, basis, extraction)(x)
    else:
        def field(x, dof):
            df = jax.jacfwd(seq.map)(x)
            return jnp.linalg.solve(df.T @ df, DiscreteFunction(dof, basis, extraction)(x))
    return field


@partial(jax.jit, static_argnames=("field",))
def _field_values(field, dof, x):
    return jax.vmap(field, (0, None))(x, dof)


class BzetaParameterisationError(RuntimeError):
    """``B^zeta`` is not bounded away from zero, so ``zeta`` is not a valid
    independent variable for the field-line ODE on this field.

    Raised by :func:`require_zeta_parameterisation`. It carries the measured
    range so the caller can report it rather than guess at it.
    """

    def __init__(self, message, *, lo, hi, tol, worst_x=None):
        super().__init__(message)
        self.lo, self.hi, self.tol, self.worst_x = lo, hi, tol, worst_x


#: ``|B^zeta|/|B|`` below which the toroidal-angle parameterisation is refused.
#: Not a tuned number -- it is far below anything a usable field produces. The
#: quasr family, including the genuinely chaotic k=1 cases, measured >= 0.774
#: (handoff_2026-08-24_poincare.md section 4.2), so a field that trips this is
#: qualitatively different from anything seen, not marginally worse.
BZETA_MIN_FRACTION = 0.05


def require_zeta_parameterisation(field, dof, name="field", *, n=4096,
                                  tol=BZETA_MIN_FRACTION, seed=23):
    r"""Refuse to trace unless ``B^zeta`` keeps one sign and stays off zero.

    The tracer integrates :math:`dr/d\zeta = \hat B^r/\hat B^\zeta`, which is a
    valid change of variables only where :math:`\hat B^\zeta \neq 0`. Where it
    is not, the section still *renders* -- as something that looks like a
    chaotic sea and is really a broken parameterisation. Distinguishing those
    two after the fact cost a full investigation once already.

    This FAILS rather than repairing. Clamping the denominator was considered
    and is wrong in both directions: clamped to ``+eps`` the right-hand side
    becomes ``~1/eps * B^r`` and the line flies off, and if ``B^zeta`` genuinely
    crossed zero, clamping on the negative side flips the sign of the whole RHS
    and the line silently traces BACKWARDS -- a rendered plot with no NaN and no
    warning. A masked invariant here would resurface as "the tracer is noisy".

    Returns the measured diagnostics on success so callers can record them.
    """
    x = jax.random.uniform(jax.random.PRNGKey(seed), (n, 3))
    # Sample the interior only: r -> 1 is where the spline map is singular, and
    # r -> 0 is the polar axis. Neither is where a parameterisation failure
    # would be a property of the FIELD.
    x = x.at[:, 0].multiply(0.96).at[:, 0].add(0.02)
    b = _field_values(field, jnp.asarray(dof), x)
    bz = b[:, 2]
    frac = bz / jnp.linalg.norm(b, axis=1)
    lo, hi = float(jnp.min(frac)), float(jnp.max(frac))
    worst = int(jnp.argmin(jnp.abs(frac)))
    worst_x = tuple(float(v) for v in x[worst])
    info = {"bz_over_b_min": lo, "bz_over_b_max": hi,
            "bz_over_b_absmin": float(jnp.min(jnp.abs(frac))),
            "sign_change": bool(lo < 0.0 < hi), "worst_x": worst_x, "tol": tol}

    if lo < 0.0 < hi:
        raise BzetaParameterisationError(
            f"{name}: B^zeta CHANGES SIGN over the interior "
            f"(B^zeta/|B| in [{lo:+.3e}, {hi:+.3e}], {n} samples). The "
            "toroidal angle is not a valid independent variable for this "
            "field: where B^zeta = 0 the field line is locally tangent to the "
            "section plane and dr/dzeta is undefined. Trace this field by "
            "arclength instead, or fix the field -- do NOT clamp the "
            "denominator, which makes the line trace backwards silently.",
            lo=lo, hi=hi, tol=tol, worst_x=worst_x)
    if info["bz_over_b_absmin"] <= tol:
        raise BzetaParameterisationError(
            f"{name}: B^zeta comes within {info['bz_over_b_absmin']:.3e} of "
            f"zero relative to |B| (tol {tol:g}; range [{lo:+.3e}, {hi:+.3e}], "
            f"{n} samples, worst at logical (r, theta, zeta) = "
            f"({worst_x[0]:.4f}, {worst_x[1]:.4f}, {worst_x[2]:.4f})). The "
            "toroidal-angle parameterisation is ill conditioned here: "
            "dr/dzeta ~ B^r/B^zeta is stiff and the step schedule is "
            "prescribed, so this would surface as drift that does not fall "
            "under refinement -- indistinguishable from chaos. Trace by "
            "arclength instead of raising the tolerance.",
            lo=lo, hi=hi, tol=tol, worst_x=worst_x)
    return info


def _uv_to_logical(y, zeta):
    r = jnp.sqrt(y[0] ** 2 + y[1] ** 2)
    theta = jnp.arctan2(y[1], y[0]) / TWO_PI
    return jnp.array([r, theta % 1.0, zeta % 1.0]), r, theta


def cross_section_rhs(field):
    """``dy/dzeta`` for ``y = (u, v)``, the Cartesian cross-section chart.

    Freezing a line at ``r >= R_MAX`` is an *event*, not a guard: the physical
    field of a harmonic form is tangent to the boundary, so a line that gets
    there has left the domain the discrete field is defined on, and the only
    honest thing to do is stop it and count it (see :func:`_escaped_mask`).
    Freezing rather than erroring also keeps a lost lane from costing anything.

    ``args`` is the coefficient vector of the field (``diffeqsolve(args=dof)``).
    """
    def rhs(zeta, y, dof):
        x, r, theta = _uv_to_logical(y, zeta)
        b = field(x, dof)
        dr, dtheta = b[0] / b[2], b[1] / b[2]
        c, s = jnp.cos(TWO_PI * theta), jnp.sin(TWO_PI * theta)
        du = c * dr - TWO_PI * r * s * dtheta
        dv = s * dr + TWO_PI * r * c * dtheta
        return jnp.where(r < R_MAX, jnp.array([du, dv]), jnp.zeros(2))
    return rhs


# ---------------------------------------------------------------------------
# The trace
# ---------------------------------------------------------------------------

def trace(field, dof, seeds, n_periods, steps_per_period=MIN_STEPS_PER_PERIOD):
    """Integrate ``seeds`` for ``n_periods`` units of logical zeta, every step
    endpoint kept.

    Args:
        field: ``(x, dof) -> (B^r, B^theta, B^zeta)``, from :func:`logical_field`.
        dof: the field's coefficient vector.
        seeds: ``(n_seeds, 2)`` array of logical ``(r, theta)`` start points at
            ``zeta = 0``.
        n_periods: number of field periods to follow.  One unit of logical zeta
            is one field period for the stellarator maps and one full toroidal
            turn for the axisymmetric ones.
        steps_per_period: prescribed steps per period. Every step endpoint is
            a saved point, so no dense interpolation enters the saved values
            and the section at ``zeta = k / steps_per_period`` is a column of
            the result.

    Returns:
        ``(ys, ok)`` with ``ys`` of shape
        ``(n_seeds, n_periods * steps_per_period + 1, 2)`` in the ``(u, v)``
        chart, and ``ok`` a per-seed boolean from the solver.
    """
    return _trace(field, jnp.asarray(dof), jnp.asarray(seeds), int(n_periods),
                  int(steps_per_period))


@partial(jax.jit, static_argnames=("field", "n_periods", "steps_per_period"))
def _trace(field, dof, seeds, n_periods, steps_per_period):
    # One compile per (field function, shapes, schedule); ``dof`` is an input.
    n_steps = n_periods * steps_per_period
    step_ts = jnp.arange(n_steps + 1) / steps_per_period

    r, theta = seeds[:, 0], seeds[:, 1]
    y0s = jnp.stack([r * jnp.cos(TWO_PI * theta),
                     r * jnp.sin(TWO_PI * theta)], axis=1)

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

    # With a prescribed schedule every lane executes the same steps: one vmap.
    return jax.vmap(one)(y0s)


def _escaped_mask(ys):
    """``True`` for seeds whose line reached the domain boundary."""
    r = jnp.sqrt(ys[..., 0] ** 2 + ys[..., 1] ** 2)
    return jnp.any(r >= R_MAX, axis=-1) | jnp.any(~jnp.isfinite(r), axis=-1)


def _step_convergence(field, dof, seeds, lo, n_periods, steps_per_period):
    """Max cross-section displacement between ``steps_per_period`` and twice it.

    Fixed steps carry no error estimate, so the step count has to be earned.
    ``lo`` is the trace of ``seeds`` at ``steps_per_period`` over ``n_periods``
    -- the caller already has it (the first ``n_periods`` of the main trace
    ARE that trace, the schedule being prescribed), so only the ``h/2`` trace
    is integrated here. Returned in units of the logical minor radius, over
    healthy seeds only.
    """
    hi, _ = trace(field, dof, seeds, n_periods, 2 * steps_per_period)
    hi = hi[:, ::2]
    good = ~(_escaped_mask(lo) | _escaped_mask(hi))
    d = jnp.linalg.norm(lo - hi, axis=-1).max(axis=-1)
    return float(jnp.max(jnp.where(good, d, -jnp.inf)))


# ---------------------------------------------------------------------------
# Rotational transform
# ---------------------------------------------------------------------------

def axis_track(ys, steps_per_period):
    """The magnetic axis as a function of zeta, from the innermost seed.

    ``ys[0]`` must be the innermost seed.  Its orbit is a small invariant curve
    encircling the axis, so the mean over turns at a *fixed* phase within the
    period is the axis position at that phase -- exact for a circle, and
    second order in the orbit radius otherwise.  Doing it per phase rather than
    once matters because the axis moves within a period, and the poloidal angle
    has to be measured about the axis *at the same zeta* or the winding picks
    up the axis excursion.
    """
    inner = ys[0, :-1].reshape(-1, steps_per_period, 2)
    center = jnp.mean(inner, axis=0)                    # (steps_per_period, 2)
    n_saves = ys.shape[1]
    reps = -(-n_saves // steps_per_period)
    return jnp.tile(center, (reps, 1))[:n_saves]


def _winding(ys, steps_per_period, center):
    """The unwrapped poloidal angle (in turns) of every line about ``center``
    and the logical zeta of every save: ``(angle (n_seeds, n_saves), zeta (n_saves,))``."""
    d = ys - center
    angle = jnp.unwrap(jnp.arctan2(d[..., 1], d[..., 0]), axis=-1) / TWO_PI
    return angle, jnp.arange(ys.shape[1]) / steps_per_period


def _slope(angle, zeta):
    """Least-squares slope of ``angle`` (rows) against ``zeta``, and the RMS
    deviation of the rows from their fitted lines."""
    zc = zeta - jnp.mean(zeta)
    ac = angle - jnp.mean(angle, axis=-1, keepdims=True)
    slope = (ac @ zc) / (zc @ zc)
    resid = jnp.sqrt(jnp.mean((ac - slope[:, None] * zc) ** 2, axis=-1))
    return slope, resid


def rotational_transform(ys, steps_per_period, nfp, center=None):
    """Iota (poloidal turns per *toroidal* turn) by least squares on the angle.

    One unit of logical zeta is one field period, i.e. ``1/nfp`` of a toroidal
    turn, hence the ``nfp`` factor.  A least-squares slope over every sample is
    used rather than the endpoint difference: on an island chain or a noisy
    orbit the endpoints are two arbitrary points on a bounded oscillation,
    while the slope is the winding rate that oscillation is riding on.

    Returns ``(iota, residual)``, the residual being the RMS deviation of the
    unwrapped angle from the fitted line in poloidal turns -- small on an
    invariant surface, ``O(island width)`` on an island, large in a chaotic
    region.
    """
    if center is None:
        center = axis_track(ys, steps_per_period)
    slope, resid = _slope(*_winding(ys, steps_per_period, center))
    return jnp.abs(slope) * nfp, resid


# ---------------------------------------------------------------------------
# Physical coordinates
# ---------------------------------------------------------------------------

#: Half-split ``|d iota|`` above which a line is treated as chaotic and given
#: NO iota, per traced period: the threshold is ``CHAOS_TOL_PER_PERIOD / N``.
#: A quasi-periodic line's half-split difference falls like ``1/N`` (it is the
#: bounded angle oscillation divided by the window), so a fixed threshold
#: flags island lines on short traces and stops flagging them on long ones. A
#: chaotic line's difference does not fall with ``N``. Measured 2026-08-26 on
#: W7-X fmm002 at 400 and 800 periods: converged Clebsch relaxations score
#: <= 9e-04 / 1.2e-04 on their regular lines (islands included) while the
#: chaotic analytic-profile field scores >= 1.4e-03 / 3.7e-03 on 29 of 40
#: lines at both lengths. ``0.4 / N`` (1e-03 at 400 periods) separates them
#: with a decade to spare at both lengths.
CHAOS_TOL_PER_PERIOD = 0.4


def _iota_convergence(ys, steps_per_period, nfp, center=None):
    """``|iota(first half) - iota(second half)|`` -- has the winding converged?

    A quasi-periodic line has a rotational transform and its estimate converges
    like ``1/N``, so the two halves of a long trace agree. A chaotic line has no
    rotational transform at all: the estimate does not converge and the halves
    disagree at the scale of the shear. That is the honest test of whether iota
    EXISTS for a line, which is a different question from whether the trace was
    accurate.

    Preferred over the angle-fit residual, which was measured and does not
    separate: hegna's clean lines score 2.4e-02 against 2.0e-02 for the chaotic
    quasr65530 k=1 sea, while this splits them 1e-06 against 5.6e-04.
    """
    if center is None:
        center = axis_track(ys, steps_per_period)
    angle, zeta = _winding(ys, steps_per_period, center)
    half = ys.shape[1] // 2
    i1 = jnp.abs(_slope(angle[:, :half], zeta[:half])[0]) * nfp
    i2 = jnp.abs(_slope(angle[:, half:], zeta[half:])[0]) * nfp
    return jnp.abs(i1 - i2)


#: Windows for the iota scatter band (:func:`_iota_window_scatter`): the trace
#: is split into this many equal ζ-windows, iota is fitted in each, and their
#: spread is the profile ribbon. 16 keeps each window a few dozen poloidal turns
#: at the default trace length -- enough for a clean per-window slope -- while
#: resolving the along-line variation an island or chaotic line carries.
N_IOTA_WINDOWS = 16


def _iota_window_scatter(ys, steps_per_period, nfp, n_windows=N_IOTA_WINDOWS,
                         center=None):
    r"""Std of the per-window rotational transform over ``n_windows`` equal
    ζ-windows: the along-line scatter of iota.

    The direct analog of the pressure band, which is the std of ``p`` over the
    line's crossings. iota is a *slope*, not a per-crossing value, so its
    scatter needs one window per estimate -- a single window would be one
    sample of the slope, the very thing that made the half-split noisy as a
    ribbon (:func:`_iota_convergence` is kept for the chaos test, where that
    property is wanted). On a flux surface the windows agree and the band is
    the per-window fit noise; on an island or a chaotic line the local winding
    varies from window to window and the band opens, exactly as ``p``'s does.
    """
    if center is None:
        center = axis_track(ys, steps_per_period)
    angle, zeta = _winding(ys, steps_per_period, center)
    edges = jnp.linspace(0, ys.shape[1], n_windows + 1).astype(int)
    iotas = jnp.stack([jnp.abs(_slope(angle[:, lo:hi], zeta[lo:hi])[0]) * nfp
                       for lo, hi in zip(edges[:-1].tolist(), edges[1:].tolist())], axis=-1)
    return jnp.std(iotas, axis=-1)


@partial(jax.jit, static_argnames=("seq",))
def _map_points(seq, x):
    return jax.vmap(seq.map)(x.reshape(-1, 3)).reshape(x.shape)


def to_xyz(seq, ys, zeta):
    """Map ``(u, v)`` points at logical ``zeta`` (a scalar, or one per point) to
    Cartesian ``(x, y, z)``."""
    r = jnp.sqrt(ys[..., 0] ** 2 + ys[..., 1] ** 2)
    theta = jnp.arctan2(ys[..., 1], ys[..., 0]) / TWO_PI % 1.0
    x = jnp.stack([r, theta, jnp.broadcast_to(jnp.asarray(zeta) % 1.0, r.shape)], axis=-1)
    return _map_points(seq, x)


def to_RZ(seq, ys, zeta):
    """Map ``(u, v)`` cross-section points at fixed logical zeta to ``(R, Z)``."""
    xyz = to_xyz(seq, ys, zeta)
    R = jnp.sqrt(xyz[..., 0] ** 2 + xyz[..., 1] ** 2)
    return R, xyz[..., 2]


def midplane_crossings(R, Z, centre_R, centre_Z, max_gap=0.5):
    """``R`` where each line crosses the midplane through the magnetic axis,
    outboard and inboard: shape ``(n_lines, 2)``.

    The profile panels are a SLICE of the section along ``Z = centre_Z``, so
    their abscissa is the physical ``R`` of the crossing and every line
    appears twice, once on each side of the axis. Sorted by ``R`` the profile
    reads as one curve inboard -> axis -> outboard, and an island chain is hit
    on whichever side has a lobe on the midplane.

    Each crossing is interpolated between the two orbit points that bracket
    the ray in poloidal angle about the axis (``alpha = 0`` outboard,
    ``alpha = +-pi`` inboard; ``arctan2``'s branch cut lies on the inboard ray,
    so that side is handled by reflecting ``dR``). The interpolation assumes
    the two points are NEIGHBOURS on one curve. On an island chain they can
    sit on two different lobes with the ray between them, and the chord
    between the lobes crosses the ray anywhere (measured 0.3-0.5 m for a 5/5
    chain on a 0.25 m plasma), so a bracketing gap wider than ``max_gap``
    radians is NaN: this line has no crossing on that side, and the profiles
    leave it out there. Measured on w7x, w7x-ini and hegna, the relative
    residual of a linear ``r(alpha)`` fit either side of the ray is ~3e-4 on
    both sides, far below the marker size.
    """
    dR, dZ = R - centre_R, Z - centre_Z
    rad = jnp.sqrt(dR ** 2 + dZ ** 2)

    def crossing(ang):
        big = jnp.asarray(jnp.inf)
        above = jnp.where(ang >= 0.0, ang, big)          # smallest angle above
        below = jnp.where(ang < 0.0, -ang, big)          # smallest |angle| below
        i = jnp.argmin(above, axis=-1)
        j = jnp.argmin(below, axis=-1)
        take = jnp.take_along_axis
        a_hi = take(ang, i[..., None], -1)[..., 0]
        a_lo = take(ang, j[..., None], -1)[..., 0]
        r_hi = take(rad, i[..., None], -1)[..., 0]
        r_lo = take(rad, j[..., None], -1)[..., 0]
        ok = (jnp.min(above, axis=-1) < jnp.inf) & (jnp.min(below, axis=-1) < jnp.inf)
        ok &= (a_hi - a_lo) <= max_gap
        t = (0.0 - a_lo) / (a_hi - a_lo)
        return jnp.where(ok, r_lo + t * (r_hi - r_lo), jnp.nan)

    r_out = crossing(jnp.arctan2(dZ, dR))
    r_in = crossing(jnp.arctan2(dZ, -dR))
    return jnp.stack([centre_R + r_out, centre_R - r_in], axis=-1)




# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

#: The axis probe (entry 0 of the seeds) orbits the magnetic axis at this
#: logical radius: small enough to be a circle, large enough that its own
#: angle is not rounding noise.
R_AXIS = 0.01
#: Outermost seed radius. The spline maps are singular at ``r = 1``.
R_EDGE = 0.97
#: Innermost seed as a fraction of the axis-to-edge distance.
T_MIN = 0.02
#: Periods of the two axis probes.
PROBE_PERIODS = 64
#: Periods and line count of the drift check (:func:`_step_convergence`).
DRIFT_PERIODS, DRIFT_LINES = 64, 8


def seed_from_axis(field, dof, n_lines, steps_per_period=MIN_STEPS_PER_PERIOD, *, seed=0):
    """``n_lines + 1`` seeds from the MAGNETIC axis to the edge, entry 0 the
    axis probe; every line at its own radius and at a random poloidal angle.

    Seeding along a ray of constant *logical* angle from ``r = 0`` starts at
    the coordinate axis. That is fine only while the two axes coincide. They
    do not have to: the maps come from equilibria, and a finite-beta one puts ``r = 0``
    at its own Shafranov-shifted axis, which is not where the vacuum field's
    axis is. Measured on ``w7x-ini`` (beta 4.2%): 4.9 cm apart, against 0.6 mm
    on vacuum W7-X. When they differ, every inner seed lands on a surface of
    size comparable to the OFFSET rather than a small one, and the section has
    a hole in the middle with no lines sampling the core at all.

    So find the axis first (one short probe trace, mean of its crossings) and
    lay the seeds between there and the edge in the ``(u, v)`` chart. Entry 0
    is the probe re-seeded at ``R_AXIS`` from the first estimate of the axis
    (two passes, see below): it is the centre reference for
    :func:`axis_track`, and it has to keep a small ORBIT around the axis
    rather than sit on it, or its own angle is rounding noise.

    The poloidal angles are uniform random (``seed``; a rerun is a rerun).
    ONE ray misses island chains: a stellarator-symmetric field has X-points
    on the symmetry line ``theta = 0``, a seed on the separatrix traces the
    separatrix, and every chain the ray crosses shows as a kink in the iota
    profile with no lines inside the islands. Random angles line up with no
    chain. Every line has its OWN radius -- one ladder of ``n_lines`` radial
    fractions -- because lines that share a radius trace the same nested
    surface twice and put two coincident points on the iota profile for no
    gain; only inside an island does the angle matter, and a line at a
    nearby radius samples that chain just as well.
    """
    # Two passes. The first probe sits at logical R_AXIS, i.e. near the
    # COORDINATE axis; if the magnetic axis has moved (w7x-ini: 4.9 cm), its
    # orbit is large, the mean of its crossings is a poor centre, and every
    # inner seed then lies INSIDE the probe's orbit with its angle measured
    # about a point off by a fraction of its own radius. So re-seed the probe
    # at the first estimate plus R_AXIS and trace again: around a true axis
    # its orbit is now small, and that probe is entry 0, the centre reference.
    # (A probe whose orbit stays large after this pass is on a wide structure
    # -- an island at the core -- not on a shifted axis; measured 2026-08-26.)
    probe = jnp.array([[R_AXIS, 0.0], [R_EDGE, 0.0]])
    ys, _ = trace(field, dof, probe, PROBE_PERIODS, steps_per_period)
    centre = jnp.mean(ys[0, ::steps_per_period], axis=0)
    probe2_uv = centre + jnp.array([R_AXIS, 0.0])
    probe2 = jnp.array([[jnp.sqrt(probe2_uv[0] ** 2 + probe2_uv[1] ** 2),
                         jnp.arctan2(probe2_uv[1], probe2_uv[0]) / TWO_PI % 1.0],
                        [R_EDGE, 0.0]])
    ys, _ = trace(field, dof, probe2, PROBE_PERIODS, steps_per_period)
    centre = jnp.mean(ys[0, ::steps_per_period], axis=0)

    thetas = jax.random.uniform(jax.random.PRNGKey(seed), (n_lines,))
    edge = R_EDGE * jnp.stack([jnp.cos(TWO_PI * thetas), jnp.sin(TWO_PI * thetas)], axis=1)
    t = jnp.linspace(T_MIN, 1.0, n_lines)[:, None]        # one radial ladder, one line each
    uv = centre[None, :] + t * (edge - centre[None, :])
    r = jnp.sqrt(uv[:, 0] ** 2 + uv[:, 1] ** 2)
    th = jnp.arctan2(uv[:, 1], uv[:, 0]) / TWO_PI % 1.0
    return jnp.concatenate([probe2[:1], jnp.stack([r, th], axis=1)], axis=0)


# ---------------------------------------------------------------------------
# The section
# ---------------------------------------------------------------------------

def planes_for(seq, planes):
    """The section planes as fractions of a field period: ``planes`` itself
    if it is a sequence, or ``planes`` equispaced planes over what the map's
    symmetry leaves distinct -- half a period for a stellarator-symmetric
    map (five: 0, 1/8, 1/4, 3/8, 1/2; the other half is the mirror image),
    the whole period otherwise (``k / planes``)."""
    if not isinstance(planes, int):
        return tuple(float(p) for p in planes)
    if seq.symmetry == "stellarator":
        return tuple(np.linspace(0.0, 0.5, planes).tolist())
    return tuple((np.arange(planes) / planes).tolist())


def steps_for(planes):
    """Steps per period for these section planes: the smallest multiple of
    the planes' common denominator that is at least
    :data:`MIN_STEPS_PER_PERIOD`, so that every plane is a step endpoint.
    The five standing planes give 24, one plane 24, a 64-plane fly 64."""
    den = reduce(lambda a, b: a * b // gcd(a, b),
                 (Fraction(float(p)).limit_denominator(4096).denominator for p in planes), 1)
    return den * -(-MIN_STEPS_PER_PERIOD // den)


def _section_RZ(seq, ys, axis_uv, steps_per_period, plane):
    """``(R, Z)`` of the crossings and of the magnetic axis at ``plane``, plus
    the logical ``(r, theta)`` of the crossings: the section as archived.

    The magnetic axis has no reason to sit on the coordinate axis ``F(0, .,
    zeta)``: the maps come from equilibria, and a finite-beta one puts ``r =
    0`` at its own Shafranov-shifted axis. The poloidal angle is measured
    about the tracked magnetic axis, which is what makes the offset measurable
    rather than fatal.
    """
    off = int(round(plane * steps_per_period))
    if abs(off - plane * steps_per_period) > 1e-9:
        raise ValueError(f"plane {plane} is not a step endpoint at {steps_per_period} steps per period")
    uv = np.asarray(ys)[:, off::steps_per_period, :]
    R, Z = to_RZ(seq, jnp.asarray(uv), plane)
    aR, aZ = to_RZ(seq, jnp.asarray(np.asarray(axis_uv)[off::steps_per_period, :]), plane)
    return {"R": np.asarray(R), "Z": np.asarray(Z), "axisR": np.asarray(aR), "axisZ": np.asarray(aZ),
            "logr": np.hypot(uv[..., 0], uv[..., 1]),
            "logth": np.arctan2(uv[..., 1], uv[..., 0]) / (2.0 * np.pi) % 1.0}


def surface_label(R, Z, axis_R, axis_Z):
    """The abscissa of the profile panels and its axis label: ``R`` on the
    midplane through the magnetic axis, both crossings per line
    (:func:`midplane_crossings`). A property of the physical curve, so two
    runs on different maps are comparable -- a logical seed radius names a
    different surface as soon as the map changes.
    """
    aR, aZ = float(np.mean(axis_R)), float(np.mean(axis_Z))
    return (np.asarray(midplane_crossings(jnp.asarray(R), jnp.asarray(Z), aR, aZ)),
            r"$R$ on the midplane through the axis  [m]")


def poincare(seq, dof, *, lines=160, periods=400, planes=5, seed=0, name="field"):
    """The Poincare sections of the Dirichlet 2-form ``dof`` on ``seq``.

    Seed ``lines`` field lines from the magnetic axis to the edge
    (:func:`seed_from_axis`), follow each for ``periods`` field periods
    (``seq.nfp`` of them per toroidal turn) with :func:`steps_for` steps per
    period, measure iota, say which lines have one, and cut the trajectories
    at every plane of :func:`planes_for` -- a count, spread over what the
    map's symmetry leaves distinct, or the planes themselves as fractions of
    a period. One integration, every plane a column of it.

    Returns a dict, the archive layout of ``scripts/poincare_trace.py``:

    * per line -- ``iota``, ``iota_err`` (the fit's RMS deviation over the
      window, in poloidal turns per toroidal turn), ``iota_scatter`` (the
      std over :data:`N_IOTA_WINDOWS` windows, the profile ribbon),
      ``seed_r`` (logical seed radius), ``keep`` (traced to the end, inside
      the domain), ``chaotic`` (no iota: the two halves of the trace
      disagree by more than ``CHAOS_TOL_PER_PERIOD / periods``), ``shown``
      (``keep & ~chaotic``);
    * ``sections[plane]`` -- ``R, Z`` of every crossing (line, crossing),
      ``axisR, axisZ`` of the magnetic axis at that plane, ``logr, logth``
      of the crossings;
    * ``drift`` -- the h vs h/2 displacement over :data:`DRIFT_PERIODS`
      periods of :data:`DRIFT_LINES` regular lines, in units of the logical
      minor radius (NaN when no line is regular: then the step cannot be
      checked this way, and a number would be a lie); ``steps``,
      ``bz_over_b`` (the range of ``B^zeta/|B|``), ``walltime`` of the trace.

    The chaos test is the half-split difference and not the angle-fit
    residual because the residual does not separate (hegna's clean lines
    2.4e-02 against 2.0e-02 for the chaotic quasr65530 sea; the half-split
    1e-06 against 5.6e-04). The drift is over the REGULAR lines only: two
    nearby chaotic trajectories separate exponentially, so on a stochastic
    line the h vs h/2 displacement measures the Lyapunov exponent, not the
    integration error, and does not fall under refinement -- which is exactly
    the signature of a broken zeta parameterisation on a regular line.
    """
    field, dof, nfp = logical_field(seq, 2, True), jnp.asarray(dof), seq.nfp
    planes = planes_for(seq, planes)
    steps = steps_for(planes)
    info = require_zeta_parameterisation(field, dof, name)
    seeds = seed_from_axis(field, dof, lines, steps, seed=seed)

    t0 = time.perf_counter()
    ys, ok = trace(field, dof, seeds, periods, steps)
    ys = ys.block_until_ready()
    walltime = time.perf_counter() - t0

    escaped = _escaped_mask(ys)
    centre = axis_track(ys, steps)
    iota, resid = rotational_transform(ys, steps, nfp, center=centre)
    chaotic = _iota_convergence(ys, steps, nfp, center=centre) > CHAOS_TOL_PER_PERIOD / periods

    # The drift subsample: DRIFT_LINES regular lines spread over the regular
    # ones, the probe excluded (its orbit at R_AXIS says nothing about the
    # step the edge needs); a fixed count keeps the shapes, and so the
    # compiled programs, the same from one field to the next.
    regular = np.flatnonzero(~np.asarray(chaotic | escaped))
    regular = regular[regular > 0]
    idx = regular[np.unique(np.linspace(0, len(regular) - 1,
                                        min(DRIFT_LINES, len(regular))).round().astype(int))]
    n_drift = min(periods, DRIFT_PERIODS)
    drift = (_step_convergence(field, dof, seeds[idx], ys[idx, :n_drift * steps + 1], n_drift, steps)
             if idx.size else float("nan"))

    keep = np.asarray(~(escaped | ~ok))[1:]
    chaotic = np.asarray(chaotic)[1:]
    return {"iota": np.asarray(iota)[1:], "iota_err": np.asarray(nfp * resid / periods)[1:],
            "iota_scatter": np.asarray(_iota_window_scatter(ys, steps, nfp, center=centre))[1:],
            "seed_r": np.asarray(seeds[1:, 0]), "keep": keep, "chaotic": chaotic,
            "shown": keep & ~chaotic,
            "sections": {plane: _section_RZ(seq, ys[1:], ys[0], steps, plane) for plane in planes},
            "drift": drift, "steps": steps, "nfp": nfp,
            "bz_over_b": (info["bz_over_b_min"], info["bz_over_b_max"]), "walltime": walltime}


# ---------------------------------------------------------------------------
# Island chains: fixed points of the return map, widths by tracing through them
# ---------------------------------------------------------------------------

def _period_map(field, steps_per_period):
    """``(y, dof) ->`` the ``(u, v)`` cross-section point one field period on
    from ``y`` at ``zeta = 0``: the fixed-step integration of :func:`trace`,
    forward-mode differentiable. The field is periodic in ``zeta``, so the
    return map over ``m`` periods is this map composed ``m`` times and its
    tangent map the product of the one-period Jacobians -- one compiled
    program for every chain order."""
    term = dfx.ODETerm(cross_section_rhs(field))
    ts = jnp.arange(steps_per_period + 1) / steps_per_period

    def one(y, dof):
        sol = dfx.diffeqsolve(
            terms=term, solver=dfx.Tsit5(), t0=0.0, t1=1.0, dt0=None, y0=y, args=dof,
            saveat=dfx.SaveAt(t1=True), stepsize_controller=dfx.StepTo(ts=ts),
            max_steps=steps_per_period + 1, throw=False, adjoint=dfx.ForwardMode())
        return sol.ys[0]
    return one


@partial(jax.jit, static_argnames=("field", "steps_per_period", "iters", "step_cap"))
def _fixed_points(field, dof, y0s, periods, steps_per_period, iters, step_cap):
    """Newton on ``Phi^m(y) - y = 0`` in the ``(u, v)`` chart from every row
    of ``y0s`` at once, ``iters`` steps each capped at ``step_cap``:
    ``(y, |Phi^m(y) - y|, D Phi^m(y))``. ``periods`` (``m``) is a traced loop
    bound and the coefficients an argument, so one compile serves every
    chain and every field of a sequence."""
    one = _period_map(field, steps_per_period)
    eye = jnp.eye(2, dtype=jnp.float64)
    step = jax.jacfwd(lambda y: (one(y, dof),) * 2, has_aux=True)      # (Jacobian, value) in one pass

    def phi(y):
        def body(_, c):
            J, y_next = step(c[0])
            return y_next, J @ c[1]
        return jax.lax.fori_loop(0, periods, body, (y, eye))

    def solve(y0):
        def body(_, y):
            y_m, S = phi(y)
            d = jnp.linalg.solve(S - eye, y_m - y)
            size = jnp.linalg.norm(d)
            return y - jnp.where(size > step_cap, d * (step_cap / size), d)

        y = jax.lax.fori_loop(0, iters, body, y0)
        y_m, S = phi(y)
        return y, jnp.linalg.norm(y_m - y), S

    return jax.vmap(solve)(y0s)


def fixed_points(seq, dof, periods, guesses, *, steps_per_period=TANGENT_STEPS_PER_PERIOD,
                 iters=12, step_cap=0.05):
    """The fixed points of the ``periods``-period return map near the
    logical ``(r, theta)`` ``guesses`` -- the O and X points of an island
    chain that closes after ``periods`` field periods -- and Greene's
    residue of each (Cary & Hanson, Phys. Fluids 29, 2464 (1986)).

    Newton on ``Phi^m(y) - y`` with the tangent map ``S`` the product of the
    one-period Jacobians (forward mode through the integrator), a capped
    step, ``iters`` iterations from every guess at once. The residue
    ``R = 1/2 - tr(S) / 4`` of an area-preserving map classifies the point:
    ``0 < R < 1`` elliptic (an O-point, the map rotates about it by
    ``2 pi nu`` per return with ``R = sin^2(pi nu)``), ``R < 0`` hyperbolic
    (an X-point), ``R > 1`` hyperbolic with reflection. ``det S`` is 1 for a
    divergence-free field; its departure is the tangent map's integration
    error, and ``defect`` is ``|Phi^m(y) - y|`` at the end.

    The residue is a property of the fixed point, NOT a width: the
    constant-shear single-harmonic pendulum relation between the two
    overestimated the traced separatrix by 1.35 to 1.6 on the paper's
    fields (``docs/research/island_diagnostic_2026-09-18.md``). Use the fixed
    points for a chain's existence and phase and to aim a width measurement
    (:func:`islands`).

    Returns a dict of arrays over the guesses: ``r``, ``theta`` (logical, at
    ``zeta = 0``), ``uv``, ``residue``, ``det``, ``defect``, ``kind``
    (``"O"``, ``"X"`` or ``"reflecting"``).
    """
    field, dof = logical_field(seq, 2, True), jnp.asarray(dof)
    guesses = jnp.asarray(guesses, dtype=jnp.float64).reshape(-1, 2)
    y0 = jnp.stack([guesses[:, 0] * jnp.cos(TWO_PI * guesses[:, 1]),
                    guesses[:, 0] * jnp.sin(TWO_PI * guesses[:, 1])], axis=1)
    ys, defect, S = _fixed_points(field, dof, y0, jnp.asarray(int(periods)), int(steps_per_period), int(iters),
                                  float(step_cap))
    residue = 0.5 - jnp.trace(S, axis1=1, axis2=2) / 4.0
    kind = np.where(residue < 0.0, "X", np.where(residue < 1.0, "O", "reflecting"))
    return {"r": np.asarray(jnp.sqrt(ys[:, 0] ** 2 + ys[:, 1] ** 2)),
            "theta": np.asarray(jnp.arctan2(ys[:, 1], ys[:, 0]) / TWO_PI % 1.0),
            "uv": np.asarray(ys), "residue": np.asarray(residue),
            "det": np.asarray(jnp.linalg.det(S)), "defect": np.asarray(defect), "kind": kind}


def resonances(iota_lo, iota_hi, nfp, m_max):
    """The chains that can sit between two rotational transforms: ``(m, n)``
    coprime with ``m <= m_max`` and ``iota_lo < nfp n / m < iota_hi``, by
    increasing ``m`` -- a chain at ``iota = nfp n / m`` closes after ``m``
    field periods (the seeds' convention, ``mrx.initial_conditions``)."""
    out = []
    for m in range(1, int(m_max) + 1):
        for n in range(1, m + 1):
            if gcd(m, n) == 1 and iota_lo < nfp * n / m < iota_hi:
                out.append((m, n))
    return out


def _chain_radii(r, iota, target, tol):
    """The radii at which a profile meets a rational: the mean radius of
    every contiguous run of locked lines (iota within ``tol`` of it), and
    every crossing between two lines on either side that no locked run
    already covers (a reversed-shear profile meets a rational twice)."""
    d = iota - target
    d = np.where(np.abs(d) < tol, 0.0, d)
    radii, i = [], 0
    while i < d.size:
        if d[i] == 0.0:
            j = i
            while j + 1 < d.size and d[j + 1] == 0.0:
                j += 1
            radii.append(float(r[i:j + 1].mean()))
            i = j + 1
        else:
            if i + 1 < d.size and d[i + 1] != 0.0 and d[i] * d[i + 1] < 0.0:
                radii.append(float(r[i] + (0.0 - d[i]) * (r[i + 1] - r[i]) / (d[i + 1] - d[i])))
            i += 1
    return radii


def islands(seq, dof, res=None, *, m_max=12, n_theta=8, residue_min=1e-3, window=0.08, ray_seeds=81,
            ray_halfwidth=0.2, periods=300, tol=2e-3):
    """Every island chain of a field and its width.

    ``res`` is a section of the same field (:func:`poincare`'s result: its
    regular lines' ``seed_r`` and ``iota`` are the profile searched); left
    ``None`` the section is traced here with :func:`poincare`'s defaults.
    A chain inside a chaotic band, with no regular line either side of
    its rational, is not looked for. For every
    rational ``nfp n / m`` with ``m <= m_max`` inside the iota range of its
    regular lines (:func:`resonances`) and every radius at which the profile
    meets it, Newton looks for the chain's fixed points from ``n_theta``
    poloidal guesses across one chain period ``1/m`` (a chain dominated by
    its second harmonic has them every ``1/(4m)``, and nothing fixes the
    phase on a field without stellarator symmetry). The fixed points of all the guesses at one
    rational are pooled and split by radius: a flattened profile meets the
    rational several times across one island, and all of those guesses find
    the same chain, while two groups mean the rational really does resonate
    at two radii. A chain is reported when
    an O-point is found (residue above ``residue_min``, within ``window`` of
    the radius) AND at least two lines on the ray are locked to the chain:
    the O-point's own line is locked by definition, so one is no evidence of
    an island, and a closed rational surface (residue zero up to the tangent
    map's integration error) is not reported. A chain narrower than the ray
    spacing ``2 ray_halfwidth / (ray_seeds - 1)`` is therefore not resolved
    rather than measured small.

    The width is MEASURED, not inferred from the residue: ``ray_seeds``
    lines on the radial ray through the O-point, ``+- ray_halfwidth`` about
    it, traced for ``periods`` field periods in one batch for all chains;
    the lines locked to the chain (fitted iota within ``tol`` of the
    rational) contiguous with the O-point are the island. ``width`` is the
    largest ``max(r) - min(r)`` of such a line over eight planes per period
    (the figures' and the paper's measure, here aimed through the O-point
    instead of left to where uniform seeds fall), ``ray`` the radial extent
    of the locked set on the ray.

    Returns a list of dicts by increasing radius: ``m``, ``n``, ``iota``,
    ``r_chain`` (the mean radius of its O-points), ``O`` and ``X`` (lists of ``(r, theta, residue)``),
    ``residue`` (the largest O-point residue), ``width``, ``ray``,
    ``ray_lo``, ``ray_hi``, ``n_locked``.
    """
    nfp = seq.nfp
    if res is None:
        res = poincare(seq, dof)
    shown = np.asarray(res["shown"])
    r, iota = np.asarray(res["seed_r"])[shown], np.abs(np.asarray(res["iota"])[shown])
    order = np.argsort(r)
    r, iota = r[order], iota[order]

    chains = []
    for m, n in resonances(float(iota.min()), float(iota.max()), nfp, m_max):
        target = nfp * n / m
        # Every radius at which the profile meets the rational is a STARTING GUESS: a
        # flattened profile meets it several times across one island, and Newton then
        # converges from all of them to the same chain. So pool the fixed points of all
        # the guesses, drop the repeats, and split what is left by radius -- one group
        # per chain, two only where the same rational really does resonate twice.
        found = []
        for r_guess in _chain_radii(r, iota, target, tol):
            fp = fixed_points(seq, dof, m, [(r_guess, j / (n_theta * m)) for j in range(n_theta)])
            ok = (fp["defect"] < 1e-8) & (np.abs(fp["r"] - r_guess) < window)
            for k in np.flatnonzero(ok):
                pt = (float(fp["r"][k]), float(fp["theta"][k]), float(fp["residue"][k]))
                if all(np.hypot(*(fp["uv"][k] - uv)) > 1e-4 for uv, _ in found):
                    found.append((np.asarray(fp["uv"][k]), pt))
        groups, pts = [], sorted((pt for _, pt in found), key=lambda q: q[0])
        for pt in pts:
            if groups and pt[0] - groups[-1][-1][0] < window:
                groups[-1].append(pt)
            else:
                groups.append([pt])
        for g in groups:
            o_pts = sorted((q for q in g if residue_min < q[2] < 1.0), key=lambda q: -q[2])
            x_pts = [q for q in g if q[2] < -residue_min]
            if o_pts:
                chains.append(dict(m=m, n=n, iota=target, r_chain=float(np.mean([q[0] for q in o_pts])),
                                   O=o_pts, X=x_pts, residue=o_pts[0][2]))
    if not chains:
        return []

    # the widths: one batched trace of the rays through the O-points
    field, dof = logical_field(seq, 2, True), jnp.asarray(dof)
    steps = MIN_STEPS_PER_PERIOD
    rays = [np.linspace(max(c["O"][0][0] - ray_halfwidth, R_AXIS), min(c["O"][0][0] + ray_halfwidth, R_EDGE), ray_seeds)
            for c in chains]
    seeds = np.concatenate([np.stack([ray, np.full_like(ray, c["O"][0][1])], axis=1) for ray, c in zip(rays, chains)])
    ys, _ = trace(field, dof, seeds, periods, steps)
    line_iota, _ = rotational_transform(ys, steps, nfp)
    line_iota = np.abs(np.asarray(line_iota)).reshape(len(chains), ray_seeds)
    rr = np.sqrt(np.sum(np.asarray(ys)[:, ::steps // 8, :] ** 2, axis=-1)).reshape(len(chains), ray_seeds, -1)
    for c, ray, li, rc in zip(chains, rays, line_iota, rr):
        locked = np.abs(li - c["iota"]) < tol
        lo = hi = int(np.argmin(np.abs(ray - c["O"][0][0])))
        if locked[lo]:
            while lo > 0 and locked[lo - 1]:
                lo -= 1
            while hi < ray_seeds - 1 and locked[hi + 1]:
                hi += 1
            inside = rc[lo:hi + 1]
            c.update(width=float(np.max(inside.max(axis=1) - inside.min(axis=1))), ray=float(ray[hi] - ray[lo]),
                     ray_lo=float(ray[lo]), ray_hi=float(ray[hi]), n_locked=hi - lo + 1)
        else:
            c.update(width=0.0, ray=0.0, ray_lo=float(ray[lo]), ray_hi=float(ray[lo]), n_locked=0)
    # An intact rational surface is a curve of fixed points whose residue is zero up to the tangent map's
    # integration error, which can pass ``residue_min`` at high ``m``; an island has lines locked to it.
    # TWO of them, not one: the O-point's own line is locked by definition (its iota IS the rational), so a
    # phantom passes a bare ``width > 0`` test on that single line, with a width below printing precision.
    return sorted((c for c in chains if c["n_locked"] > 1), key=lambda c: c["r_chain"])
