r"""Poincaré sections of a discrete field, parameterised by the toroidal angle.

Three things differ from :func:`mrx.plotting.integrate_fieldlines`, and each
one removes a specific error source rather than trading one for another.

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
front.  Every lane then executes the same number of identical-cost steps,
batching becomes a pure memory knob, and a pathological seed costs what a
healthy one costs.  :func:`step_convergence` is the price: fixed steps have no
error control, so the step count has to be *justified* by refinement instead of
assumed.

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


# ---------------------------------------------------------------------------
# The field
# ---------------------------------------------------------------------------

def logical_field(seq, dof, k, dirichlet):
    r"""Contravariant logical components of the vector field behind a k-form.

    A 2-form pushes forward by Piola, :math:`B = DF\,\hat B/J`, so its
    coefficients *are* the contravariant components and the field-line
    direction in logical space is :math:`\hat B` itself.  A 1-form pushes
    forward as :math:`v = DF^{-T}\hat A`, so the logical direction is
    :math:`DF^{-1} v = g^{-1}\hat A` with :math:`g = DF^T DF`.

    Only the direction matters below -- the third component divides out -- so
    no Jacobian factor is applied.
    """
    if k not in (1, 2):
        raise ValueError(f"logical_field: k must be 1 or 2, got {k}")
    basis = seq.basis_2 if k == 2 else seq.basis_1
    extraction = getattr(seq, f"e{k}_dbc" if dirichlet else f"e{k}")
    # DiscreteFunction folds the extraction into the coefficients once and
    # evaluates only the basis functions that are nonzero at x.
    discrete = DiscreteFunction(jnp.asarray(dof), basis, extraction)

    if k == 2:
        def field(x):
            return discrete(x)
    else:
        def field(x):
            df = jax.jacfwd(seq.map)(x)
            return jnp.linalg.solve(df.T @ df, discrete(x))
    return field


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


def require_zeta_parameterisation(field, *, n=4096, tol=BZETA_MIN_FRACTION,
                                  name="field", seed=23):
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
    b = jax.vmap(field)(x)
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
    honest thing to do is stop it and count it (see :func:`escaped_mask`).
    Freezing rather than erroring also keeps a lost lane from costing anything.
    """
    def rhs(zeta, y, args):
        x, r, theta = _uv_to_logical(y, zeta)
        b = field(x)
        dr, dtheta = b[0] / b[2], b[1] / b[2]
        c, s = jnp.cos(TWO_PI * theta), jnp.sin(TWO_PI * theta)
        du = c * dr - TWO_PI * r * s * dtheta
        dv = s * dr + TWO_PI * r * c * dtheta
        return jnp.where(r < R_MAX, jnp.array([du, dv]), jnp.zeros(2))
    return rhs


# ---------------------------------------------------------------------------
# The trace
# ---------------------------------------------------------------------------

def trace(field, seeds, n_periods, steps_per_period=32, saves_per_period=8,
          batch_size=None, adaptive=False, rtol=1e-8, atol=1e-10):
    """Integrate ``seeds`` for ``n_periods`` units of logical zeta.

    Args:
        field: ``x -> (B^r, B^theta, B^zeta)``, from :func:`logical_field`.
        seeds: ``(n_seeds, 2)`` array of logical ``(r, theta)`` start points at
            ``zeta = 0``.
        n_periods: number of field periods to follow.  One unit of logical zeta
            is one field period for the stellarator maps and one full toroidal
            turn for the axisymmetric ones.
        steps_per_period: prescribed steps per period.  Must be a multiple of
            ``saves_per_period`` so that every save time is a step endpoint and
            no dense interpolation enters the saved values.
        saves_per_period: samples kept per period.  One would suffice for the
            section itself; more are needed to unwrap the poloidal angle
            without aliasing (``saves_per_period`` must exceed twice the
            poloidal turns per period).
        adaptive: use a PID controller instead of the prescribed schedule.
            Only for measuring what the prescribed schedule buys.

    Returns:
        ``(ys, ok)`` with ``ys`` of shape
        ``(n_seeds, n_periods * saves_per_period + 1, 2)`` in the ``(u, v)``
        chart, and ``ok`` a per-seed boolean from the solver.
    """
    if steps_per_period % saves_per_period:
        raise ValueError(
            f"steps_per_period={steps_per_period} must be a multiple of "
            f"saves_per_period={saves_per_period}; otherwise the saved values "
            "come from dense interpolation rather than from steps")

    n_steps = n_periods * steps_per_period
    step_ts = jnp.arange(n_steps + 1) / steps_per_period
    save_ts = jnp.arange(n_periods * saves_per_period + 1) / saves_per_period

    seeds = jnp.asarray(seeds)
    r, theta = seeds[:, 0], seeds[:, 1]
    y0s = jnp.stack([r * jnp.cos(TWO_PI * theta),
                     r * jnp.sin(TWO_PI * theta)], axis=1)

    term = dfx.ODETerm(cross_section_rhs(field))
    if adaptive:
        controller = dfx.PIDController(rtol=rtol, atol=atol)
        dt0, max_steps = 1.0 / steps_per_period, 100 * n_steps + 1000
    else:
        controller = dfx.StepTo(ts=step_ts)
        dt0, max_steps = None, n_steps + 1

    def one(y0):
        sol = dfx.diffeqsolve(
            terms=term, solver=dfx.Tsit5(),
            t0=0.0, t1=float(n_periods), dt0=dt0, y0=y0,
            saveat=dfx.SaveAt(ts=save_ts),
            stepsize_controller=controller,
            max_steps=max_steps, throw=False,
        )
        return sol.ys, sol.result == dfx.RESULTS.successful

    # Full vmap by default.  With a prescribed schedule every lane executes the
    # same steps, so there is nothing to gain from chunking and the batch size
    # is purely a memory knob -- the opposite of the adaptive case, where the
    # chunk exists to stop one bad seed from setting the step for the rest.
    if batch_size is None:
        return jax.vmap(one)(y0s)
    return jax.lax.map(one, y0s, batch_size=batch_size)


def escaped_mask(ys):
    """``True`` for seeds whose line reached the domain boundary."""
    r = jnp.sqrt(ys[..., 0] ** 2 + ys[..., 1] ** 2)
    return jnp.any(r >= R_MAX, axis=-1) | jnp.any(~jnp.isfinite(r), axis=-1)


def step_convergence(field, seeds, n_periods, steps_per_period,
                     saves_per_period=8, batch_size=None):
    """Max cross-section displacement between ``steps_per_period`` and twice it.

    Fixed steps carry no error estimate, so the step count has to be earned.
    Returned in units of the logical minor radius, over healthy seeds only.
    """
    lo, _ = trace(field, seeds, n_periods, steps_per_period, saves_per_period,
                  batch_size=batch_size)
    hi, _ = trace(field, seeds, n_periods, 2 * steps_per_period,
                  saves_per_period, batch_size=batch_size)
    good = ~(escaped_mask(lo) | escaped_mask(hi))
    d = jnp.linalg.norm(lo - hi, axis=-1).max(axis=-1)
    return float(jnp.max(jnp.where(good, d, -jnp.inf)))


# ---------------------------------------------------------------------------
# Rotational transform
# ---------------------------------------------------------------------------

def axis_track(ys, saves_per_period):
    """The magnetic axis as a function of zeta, from the innermost seed.

    ``ys[0]`` must be the innermost seed.  Its orbit is a small invariant curve
    encircling the axis, so the mean over turns at a *fixed* phase within the
    period is the axis position at that phase -- exact for a circle, and
    second order in the orbit radius otherwise.  Doing it per phase rather than
    once matters because the axis moves within a period, and the poloidal angle
    has to be measured about the axis *at the same zeta* or the winding picks
    up the axis excursion.
    """
    inner = ys[0, :-1].reshape(-1, saves_per_period, 2)
    center = jnp.mean(inner, axis=0)                    # (saves_per_period, 2)
    n_saves = ys.shape[1]
    reps = -(-n_saves // saves_per_period)
    return jnp.tile(center, (reps, 1))[:n_saves]


def rotational_transform(ys, saves_per_period, nfp, center=None):
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
        center = axis_track(ys, saves_per_period)
    d = ys - center
    angle = jnp.unwrap(jnp.arctan2(d[..., 1], d[..., 0]), axis=-1) / TWO_PI
    zeta = jnp.arange(ys.shape[1]) / saves_per_period

    zc = zeta - jnp.mean(zeta)
    ac = angle - jnp.mean(angle, axis=-1, keepdims=True)
    slope = (ac @ zc) / (zc @ zc)
    resid = jnp.sqrt(jnp.mean((ac - slope[:, None] * zc) ** 2, axis=-1))
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


def iota_convergence(ys, saves_per_period, nfp, center=None):
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
        center = axis_track(ys, saves_per_period)
    d = ys - center
    angle = jnp.unwrap(jnp.arctan2(d[..., 1], d[..., 0]), axis=-1) / TWO_PI
    zeta = jnp.arange(ys.shape[1]) / saves_per_period
    half = ys.shape[1] // 2

    def slope(a, z):
        zc = z - jnp.mean(z)
        ac = a - jnp.mean(a, axis=-1, keepdims=True)
        return (ac @ zc) / (zc @ zc)

    i1 = jnp.abs(slope(angle[:, :half], zeta[:half])) * nfp
    i2 = jnp.abs(slope(angle[:, half:], zeta[half:])) * nfp
    return jnp.abs(i1 - i2)


def to_RZ(seq, ys, zeta):
    """Map ``(u, v)`` cross-section points at fixed logical zeta to ``(R, Z)``."""
    r = jnp.sqrt(ys[..., 0] ** 2 + ys[..., 1] ** 2)
    theta = jnp.arctan2(ys[..., 1], ys[..., 0]) / TWO_PI % 1.0
    x = jnp.stack([r, theta, jnp.full_like(r, zeta % 1.0)], axis=-1)
    xyz = jax.vmap(seq.map)(x.reshape(-1, 3)).reshape(x.shape)
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


def resonant_rationals(iota_min, iota_max, nfp, denom_max=30, min_sep=0.06):
    """Rationals in ``[iota_min, iota_max]`` where an island chain can form.

    An island chain needs a resonant perturbation: ``iota = n/m`` with ``n`` the
    toroidal and ``m`` the poloidal mode number. A field with ``nfp`` field
    periods carries only toroidal harmonics ``n = 0 (mod nfp)``, so the only
    rationals that can open an island are those whose NUMERATOR is a multiple
    of ``nfp``. Every other rational surface is resonance-free and closes on
    itself harmlessly.

    Deduplicated by VALUE, so ``5/6`` is kept and ``10/12`` -- the same
    surface, driven by a weaker harmonic -- is not repeated.

    Which of them to label is a spacing problem: ``denom_max = 30`` on W7-X
    puts ~40 resonances in an iota range of 0.2, and labelling all of them
    overprints, while the two or three lowest orders leave the scale unreadable.
    So the candidates are ranked by poloidal mode number (an ``n/m`` island is
    wider the smaller ``m`` is, and that is also how they are read), and each
    is accepted only if it is at least ``min_sep`` of the range away from every
    label already placed. Low orders always win their slot; higher orders fill
    the gaps until the spacing rule stops them.
    """
    span = max(iota_max - iota_min, 1e-12)
    candidates, seen = [], set()
    for j in range(1, max(denom_max // nfp, 1) + 1):
        n_tor = j * nfp
        for m_pol in range(1, denom_max + 1):
            value = n_tor / m_pol
            if iota_min <= value <= iota_max and value not in seen:
                candidates.append((m_pol, n_tor, value))
                seen.add(value)
    ticks, labels = [], []
    for m_pol, n_tor, value in sorted(candidates):
        if all(abs(value - t) >= min_sep * span for t in ticks):
            ticks.append(value)
            labels.append(f"{n_tor}/{m_pol}")
    order = sorted(range(len(ticks)), key=lambda i: ticks[i])
    return [ticks[i] for i in order], [labels[i] for i in order]


#: Colormap for the iota scale. ``gist_rainbow`` rather than ``turbo`` to match
#: the reference figure (data/poincare_plot_pretty_w7x.pdf): with one colour per
#: nested surface, a rainbow's hue cycle separates ADJACENT surfaces, which is
#: what the eye follows here. turbo's luminance ramp is better for a continuous
#: field and worse for a stack of discrete curves.
SECTION_CMAP = "gist_rainbow"
#: Colormap for pressure. Sequential, because p is a magnitude with a zero,
#: unlike iota which is read against its rationals.
PRESSURE_CMAP = "plasma"


def render_section(R, Z, iota, iota_err, seed_r, keep, *, title, subtitle,
                   axis_RZ=None, path=None, profile_x=None,
                   profile_xlabel="seed radius $r$", nfp=None, denom_max=30,
                   logical=None, pressure=None,
                   pressure_label=r"$p$", split_iota_p=None, pressure_scale=100.0,
                   cmap=SECTION_CMAP, iota_lim=None):
    """The section coloured by iota, with the iota profile and optionally p.

    Pure arrays in, so a run can be re-rendered from its archive without
    rebuilding the map -- which is the expensive half of producing it.

    ``iota_lim`` pins the colour scale instead of taking it from this figure's
    own lines.  Two sections drawn on limits fitted separately are not
    comparable by colour at all -- the same hue means a different transform in
    each -- so any caller producing a set that is meant to be read side by side
    (two relaxation states, a plane scan) must pass one shared pair.

    ``pressure`` is per-crossing, the same shape as ``R``. It is OPTIONAL
    because the fields this traces are harmonic (vacuum-like) and carry no
    pressure at all; a vacuum run leaves it ``None`` and gets exactly the
    previous figure. When it is given, the third panel is the pressure PROFILE:
    per line, the mean of p over its crossings with a one-standard-deviation
    band, against the same surface label the iota profile uses (labelled
    ``pressure_label``). On a flux surface of an equilibrium p is constant and
    the band collapses; on an island chain or a chaotic line it is not, and the
    band width measures how far that line is from ``B . grad p = 0``.

    ``pressure_scale`` multiplies p wherever it is drawn (colour and profile),
    and the labels say so.

    Every kept line is drawn and fitted, chaotic ones included: the iota
    profile carries ``iota_err`` as a ribbon (see the comment
    at the plot), so a line without a rotational transform shows as a point
    with a wide ribbon rather than as a separate category.

    ``split_iota_p`` colours the section by iota ABOVE the magnetic axis and by
    p BELOW it, in one panel; the default is on whenever ``pressure`` is given.
    It needs ``axis_RZ`` and raises without it rather than quietly drawing a
    half-empty panel: 'above' and 'below' are defined against the MAGNETIC
    axis, not ``Z = 0``, which would cut a Shafranov-shifted plasma off-centre.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415  (keep the module headless)

    if split_iota_p is None:
        split_iota_p = pressure is not None
    if split_iota_p and pressure is None:
        raise ValueError("split_iota_p=True needs a pressure array: the "
                         "below-axis half of the section is coloured by p.")
    if split_iota_p and axis_RZ is None:
        raise ValueError("split_iota_p=True needs axis_RZ: the split is at the "
                         "magnetic axis, not at Z = 0.")

    has_p = pressure is not None
    # The pressure is drawn at ``pressure_scale`` times its value (the natural
    # scale of p in code units is ~1e-2, and 100 p reads in units).
    p_label = f"{pressure_label} $\\times$ {pressure_scale:g}"
    if logical is None and not has_p:
        fig = plt.figure(figsize=(11.5, 5.0), constrained_layout=True)
        ax, bx = fig.subplots(1, 2, width_ratios=[1.3, 1.0])
        lx = px = None
    elif has_p:
        # The three panels Tobias asked for: the section, the iota profile
        # unchanged, and p. The logical chart is dropped here rather than made
        # a fourth -- it is a diagnostic, and this is the presentation figure.
        fig = plt.figure(figsize=(15.5, 4.8), constrained_layout=True)
        ax, bx, px = fig.subplots(1, 3, width_ratios=[1.3, 1.0, 1.0])
        lx = None
    else:
        fig = plt.figure(figsize=(15.5, 4.8), constrained_layout=True)
        ax, lx, bx = fig.subplots(1, 3, width_ratios=[1.15, 1.0, 1.15])
        px = None

    shown = keep
    good = iota[shown][jnp.isfinite(iota[shown])] if shown.any() else iota[:0]
    if iota_lim is not None:
        lo, hi = float(iota_lim[0]), float(iota_lim[1])
    else:
        lo, hi = ((float(jnp.min(good)), float(jnp.max(good)))
                  if good.size else (0.0, 1.0))
    if hi - lo < 1e-9:
        lo, hi = lo - 5e-3, hi + 5e-3

    # One marker per crossing: ~10^4 points want a hairline to show the surface
    # texture, ~10^2 want something you can actually see.
    npts = max(int(keep.sum()) * R.shape[1], 1)
    size = float(jnp.clip(3000.0 / npts, 0.35, 15.0))
    colour = jnp.broadcast_to(iota[:, None], R.shape)

    # The split is per CROSSING, not per line: a surface straddles the axis, so
    # the same line is iota-coloured where it is above and p-coloured below.
    # axis_RZ carries the axis crossing at each save; the dividing line is
    # their mean (the axis wanders by ~1e-3 of the minor radius over a period).
    if split_iota_p:
        z_axis = float(jnp.mean(jnp.asarray(axis_RZ[1])))
        upper = Z >= z_axis
    else:
        z_axis = None
        upper = jnp.ones_like(R, dtype=bool)
    # `shown` selects LINES, `upper` selects CROSSINGS.
    shown2 = jnp.broadcast_to(shown[:, None], R.shape)
    sel_iota = shown2 & upper
    sc = ax.scatter(R[sel_iota], Z[sel_iota], c=colour[sel_iota],
                    s=size, vmin=lo, vmax=hi, cmap=cmap, linewidths=0,
                    rasterized=True)
    psc = None
    if split_iota_p:
        # Chaotic lines keep their grey in BOTH halves: colouring one half of
        # a line and greying the other reads as two different objects.
        sel_p = shown2 & ~upper
        if sel_p.any():
            psc = ax.scatter(R[sel_p], Z[sel_p], c=pressure_scale * pressure[sel_p], s=size,
                             cmap=PRESSURE_CMAP, linewidths=0, rasterized=True)
    res_ticks, res_labels = (resonant_rationals(lo, hi, int(nfp), denom_max)
                             if nfp else ([], []))
    if (~keep).any():
        ax.scatter(R[~keep], Z[~keep], c="0.55", s=size, linewidths=0,
                   rasterized=True, label=f"lost ({int((~keep).sum())})")
        ax.legend(loc="upper right", fontsize=7, markerscale=4)
    if axis_RZ is not None:
        # ONE marker at the mean, plus a hairline through the wander. Drawing a
        # "k+" at every save stacked 401 opaque markers into a black blob ~10%
        # of the minor radius across, which reads as a failed line at the axis
        # -- it is neither a line nor a failure, it is the axis doing what a
        # stellarator axis does.
        aR, aZ = jnp.asarray(axis_RZ[0]), jnp.asarray(axis_RZ[1])
        if aR.ndim and aR.size > 1:
            ax.plot(aR, aZ, "-", color="0.35", lw=0.4, alpha=0.6, zorder=4)
        ax.plot(jnp.mean(aR), jnp.mean(aZ), "k+", ms=7, mew=1.2, zorder=5)
    cbar = fig.colorbar(sc, ax=ax, label=r"$\iota$", fraction=0.046, pad=0.02)
    if res_ticks:
        # Only the rationals an nfp-periodic field can actually resonate with:
        # everything else on the colorbar is a surface no island can open on.
        cbar.set_ticks(res_ticks)
        cbar.set_ticklabels(res_labels)
    if psc is not None:
        pbar = fig.colorbar(psc, ax=ax, label=p_label, fraction=0.046, pad=0.02)
        pbar.ax.tick_params(labelsize=7)
        ax.axhline(z_axis, color="0.35", lw=0.6, ls=":", zorder=1)

    # An axisymmetric vacuum field has iota = 0, so every line is a fixed point
    # of the return map and the section collapses onto the midplane.  That is
    # the right answer, but equal aspect renders it as a hairline, so the aspect
    # is only held when the two spans are within a factor of 20.
    xlim = _padded(R[keep])
    # The floor has to be RELATIVE to the other axis: an absolute one is
    # meaningless against whatever units R happens to be in, and leaves a 1e-16
    # Z-range labelled in units of 1e-16.
    ylim = _padded(Z[keep], floor=0.04 * (xlim[1] - xlim[0]))
    spans = (xlim[1] - xlim[0], ylim[1] - ylim[0])
    to_scale = max(spans) / max(min(spans), 1e-30) < 20.0
    ax.set_aspect("equal" if to_scale else "auto")
    if np.ptp(Z[keep]) < 1e-6 * (xlim[1] - xlim[0]):
        ax.text(0.5, 0.86, "iota = 0: every line is a fixed point of the\n"
                           "return map, so each surface is a single dot",
                transform=ax.transAxes, ha="center", fontsize=8, color="0.35")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("R")
    ax.set_ylabel("Z")
    ax.set_title(title + ("" if to_scale else "\nAXES NOT TO SCALE"),
                 fontsize=10)

    if lx is not None:
        # The SAME crossings in the logical chart: r against theta, both in
        # [0,1]. Nested surfaces are horizontal bands here, and anything that
        # is not -- a chain of islands, or surfaces sitting off-centre in r
        # because the magnetic axis is not at r=0 -- shows up immediately,
        # where the physical panel hides it behind the shaping.
        lr, lth = logical
        lx.scatter(lr[shown], lth[shown], c=colour[shown], s=size, vmin=lo,
                   vmax=hi, cmap=cmap, linewidths=0, rasterized=True)
        if (~keep).any():
            lx.scatter(lr[~keep], lth[~keep], c="0.55", s=size, linewidths=0,
                       rasterized=True)
        lx.set_xlim(0.0, 1.0)
        lx.set_ylim(0.0, 1.0)
        lx.set_xlabel(r"logical $r$")
        lx.set_ylabel(r"logical $\theta$")
        lx.set_title("logical chart", fontsize=10)

    x = seed_r if profile_x is None else profile_x
    # The abscissa may carry several crossings per line (both midplane
    # crossings, see midplane_crossings): per-line quantities are tiled to
    # match, and a NaN crossing drops that entry -- an island chain whose
    # lobes the ray passes between is in the section but has no honest place
    # in the profile on that side. Sorted by abscissa, the curve runs
    # inboard -> axis -> outboard as a slice of the section panel.
    X = jnp.asarray(x)
    X = X[:, None] if X.ndim == 1 else X
    xs = X.ravel()

    def per_line(a):
        return jnp.broadcast_to(jnp.asarray(a)[:, None], X.shape).ravel()

    prof = per_line(shown) & jnp.isfinite(xs)
    order = jnp.argsort(xs[prof])
    xo, io, eo = xs[prof][order], per_line(iota)[prof][order], per_line(iota_err)[prof][order]
    # The ribbon is iota +- iota_err, the difference between the fits on the
    # two halves of the trace (iota_convergence): in iota units, ~1/N on a
    # flux surface, the island width on a chain, the shear scale on a chaotic
    # line. NOT the angle-fit residual: that is in poloidal turns and grows
    # with the length of the fit, so it says nothing about iota by itself.
    bx.fill_between(xo, io - eo, io + eo, color="tab:blue", alpha=0.2, lw=0,
                    label=r"$\iota \pm |\iota_{1st\,half} - \iota_{2nd\,half}|$")
    bx.plot(xo, io, "o-", ms=3, lw=0.8, color="tab:blue")
    bx.legend(loc="upper left", fontsize=7)
    for value, lab in zip(res_ticks, res_labels):
        bx.axhline(value, color="0.55", lw=0.6, ls="--", zorder=0)
        bx.annotate(lab, (0.995, value), xycoords=("axes fraction", "data"),
                    ha="right", va="bottom", fontsize=6.5, color="0.4")
    bx.set_xlabel(profile_xlabel)
    bx.set_ylabel(r"$\iota$")
    if iota_lim is not None:
        # Same reason as the colour scale: two profiles drawn on separately
        # fitted y-axes look alike however far the transform actually moved.
        bx.set_ylim(lo, hi)
    bx.grid(alpha=0.3)
    bx.set_title(subtitle, fontsize=10)

    if px is not None:
        # p against the same surface label the iota profile uses, so the two
        # panels read against a common abscissa. One number per line, the mean
        # over its crossings, with the spread as a band: a flux surface has a
        # constant p and no band, an island chain or a chaotic line does not,
        # and the band width is how far that line is from B . grad p = 0.
        p_mean = per_line(jnp.mean(pressure_scale * pressure, axis=1))
        p_std = per_line(jnp.std(pressure_scale * pressure, axis=1))
        if prof.any():
            order = jnp.argsort(xs[prof])
            xo, mo, so = xs[prof][order], p_mean[prof][order], p_std[prof][order]
            px.fill_between(xo, mo - so, mo + so, color="tab:blue", alpha=0.2,
                            lw=0, label=r"$\pm 1$ std over the line")
            px.plot(xo, mo, "o-", ms=3, lw=0.8, color="tab:blue", label="mean")
        px.set_xlabel(profile_xlabel)
        px.set_ylabel(p_label)
        px.grid(alpha=0.3)
        px.legend(loc="best", fontsize=7)
        px.set_title("pressure profile", fontsize=10)

    if path is not None:
        fig.savefig(path, dpi=200)
        plt.close(fig)
    return fig


def _padded(v, pad=0.06, floor=0.0):
    lo, hi = float(jnp.nanmin(v)), float(jnp.nanmax(v))
    span = max(hi - lo, floor)
    mid = 0.5 * (lo + hi)
    return mid - 0.5 * span - pad * span, mid + 0.5 * span + pad * span


def seed_from_axis(field, n_seeds, saves_per_period, *, r_axis=0.01,
                   r_edge=0.97, theta=0.0, n_rays=4, probe_periods=64,
                   steps_per_period=24, t_min=0.02):
    """Seeds spaced from the MAGNETIC axis to the edge, not from ``r = 0``.

    :func:`seed_line` walks out along constant *logical* radius, i.e. from the
    coordinate axis. That is fine only while the two axes coincide. They do not
    have to: the maps come from equilibria, and a finite-beta one puts ``r = 0``
    at its own Shafranov-shifted axis, which is not where the vacuum field's
    axis is. Measured on ``w7x-ini`` (beta 4.2%): 4.9 cm apart, against 0.6 mm
    on vacuum W7-X.

    When they differ, every inner seed lands on a surface of size comparable to
    the OFFSET rather than a small one -- w7x-ini's innermost surface came out
    at ``a_eff = 0.10 m`` -- and the section has a hole in the middle with no
    lines sampling the core at all.

    So find the axis first (one short probe trace, mean of its crossings) and
    lay the seeds along the ray from there to the edge point in the ``(u, v)``
    chart. Entry 0 is still the ``r_axis`` probe, unmoved: it is the centre
    reference for :func:`axis_track`, and it has to keep a small ORBIT around
    the axis rather than sit on it, or its own angle is rounding noise.

    ``n_rays`` rays are seeded, ``n_seeds`` each. ONE ray misses island
    chains: a stellarator-symmetric field has X-points on the symmetry line
    ``theta = 0``, a seed on the separatrix traces the separatrix, and every
    chain the ray crosses shows as a kink in the iota profile with no lines
    inside the islands. The extra rays are offset by multiples of the golden
    angle, ``theta_j = j * 0.618...``, which no low-order chain's X-points can
    all line up with -- equally spaced rays would, for every chain whose
    poloidal mode number divides ``n_rays``.
    """
    probe = jnp.array([[r_axis, theta], [r_edge, theta]])
    ys, _ = trace(field, probe, probe_periods, steps_per_period,
                  saves_per_period)
    centre = jnp.mean(ys[0, ::saves_per_period], axis=0)
    golden = 0.5 * (jnp.sqrt(5.0) - 1.0)
    thetas = (theta + golden * jnp.arange(n_rays)) % 1.0
    edge = r_edge * jnp.stack([jnp.cos(TWO_PI * thetas),
                               jnp.sin(TWO_PI * thetas)], axis=1)

    t = jnp.linspace(t_min, 1.0, n_seeds)[None, :, None]
    uv = (centre[None, None, :]
          + t * (edge - centre[None, :])[:, None, :]).reshape(-1, 2)
    r = jnp.sqrt(uv[:, 0] ** 2 + uv[:, 1] ** 2)
    th = jnp.arctan2(uv[:, 1], uv[:, 0]) / TWO_PI % 1.0
    seeds = jnp.stack([r, th], axis=1)
    return jnp.concatenate([jnp.array([[r_axis, theta]]), seeds], axis=0)


def seed_line(n_seeds, r_min=0.03, r_max=0.97, theta=0.0, r_axis=0.01):
    """Seeds along a logical radial ray at ``zeta = 0``, axis probe first.

    Entry 0 is a dedicated probe at ``r_axis`` whose only job is to supply the
    centre for :func:`axis_track`.  It has to be a separate seed: the angle of
    the reference orbit *about itself* is the difference of two identical
    floats, so its own winding is pure rounding noise and any iota reported for
    it is meaningless.  Callers should drop entry 0 from anything they plot.
    """
    r = jnp.concatenate([jnp.array([r_axis]), jnp.linspace(r_min, r_max, n_seeds)])
    return jnp.stack([r, jnp.full_like(r, theta)], axis=1)


# ---------------------------------------------------------------------------
# Driver glue
# ---------------------------------------------------------------------------
#
# Everything below is shared by the scripts that produce a figure.  They differ
# only in where the field comes from -- a nullspace solve, a relaxation state,
# a file -- and nothing past that point should be written twice.

def trace_and_classify(field, seeds, nfp, *, n_periods, steps_per_period,
                       saves_per_period, batch_size=None, drift_periods=64,
                       drift_seeds=8):
    """Trace ``seeds``, measure iota, and say which lines have one.

    Seed 0 is the axis probe: it defines the centre, so its own winding is the
    difference of two identical numbers.  Its orbit is kept -- as ``axis`` --
    and it is dropped from everything that is reported, exactly as
    :func:`seed_line` and :func:`seed_from_axis` document.

    ``chaotic`` comes from :func:`iota_convergence` rather than from the
    angle-fit residual, and is computed here, on the full trace and about the
    same centre that iota used.  A caller that recomputes it from an archive
    with the probe already stripped is winding about a different point.

    Returns a dict of numpy arrays plus ``walltime``, ``drift`` and
    ``saves_per_period``.
    """
    t0 = time.perf_counter()
    ys, ok = trace(field, seeds, n_periods, steps_per_period, saves_per_period,
                   batch_size=batch_size)
    ys = jnp.asarray(ys).block_until_ready()
    walltime = time.perf_counter() - t0

    escaped = escaped_mask(ys)
    centre = axis_track(ys, saves_per_period)
    iota, _ = rotational_transform(ys, saves_per_period, nfp, center=centre)
    iota_err = iota_convergence(ys, saves_per_period, nfp, center=centre)
    chaotic = iota_err > CHAOS_TOL_PER_PERIOD / n_periods

    # The drift check re-traces at h and h/2, so it is priced per seed: a
    # subsample says the same thing.
    #
    # It is measured over the REGULAR lines only, and that is not a cosmetic
    # choice. Two nearby chaotic trajectories separate exponentially, so on a
    # stochastic line the h vs h/2 displacement measures the Lyapunov exponent
    # rather than the integration error -- it saturates at the size of the
    # stochastic region and does NOT fall under refinement. Which is exactly
    # the signature that means "the zeta parameterisation is broken" on a
    # regular line, so a mixed sample reports a healthy trace as a broken one.
    # The probe is excluded too: a seed at r = 0.01 is the cheapest orbit in
    # the batch and the least informative about the step the edge needs.
    #
    # ``drift`` is NaN when no line is regular. That is not a failed step
    # check; it is the statement that on this trace the step cannot be checked
    # this way at all, and a number would be a lie.
    regular = np.flatnonzero(~np.asarray(chaotic | escaped))
    regular = regular[regular > 0]
    idx = regular[:: max(1, len(regular) // drift_seeds)]
    drift = (step_convergence(field, seeds[idx],
                              min(n_periods, drift_periods),
                              steps_per_period, saves_per_period,
                              batch_size=batch_size)
             if idx.size else float("nan"))

    return {"ys": np.asarray(ys[1:]), "ok": np.asarray(ok[1:]),
            "escaped": np.asarray(escaped[1:]), "iota": np.asarray(iota[1:]),
            "iota_err": np.asarray(iota_err[1:]), "chaotic": np.asarray(chaotic[1:]),
            "seeds": np.asarray(seeds[1:]), "axis": np.asarray(ys[0]),
            "walltime": walltime, "drift": drift, "drift_lines": int(idx.size),
            "saves_per_period": saves_per_period}


def section_RZ(seq, ys, axis_uv, saves_per_period, plane):
    """``(R, Z)`` of the crossings, of the magnetic axis, and of ``r = 0``.

    The magnetic axis has no reason to sit on the coordinate axis
    ``F(0, ., zeta)``: the maps come from equilibria, and a finite-beta one puts
    ``r = 0`` at its own Shafranov-shifted axis.  Both are returned, so the
    distance between them is a number the caller can print.  Nothing downstream
    depends on the two coinciding -- the poloidal angle is measured about the
    tracked magnetic axis, which is what makes the offset measurable rather
    than fatal.

    Returns ``(R, Z, axis_R, axis_Z, coord_R, coord_Z, logical_r,
    logical_theta)``.
    """
    off = int(round(plane * saves_per_period))
    uv = np.asarray(ys)[:, off::saves_per_period, :]
    R, Z = to_RZ(seq, jnp.asarray(uv), plane)
    aR, aZ = to_RZ(
        seq, jnp.asarray(np.asarray(axis_uv)[off::saves_per_period, :]), plane)
    cR, cZ = to_RZ(seq, jnp.zeros((1, 2)), plane)
    lr = np.hypot(uv[..., 0], uv[..., 1])
    lth = np.arctan2(uv[..., 1], uv[..., 0]) / (2.0 * np.pi) % 1.0
    return (np.asarray(R), np.asarray(Z), np.asarray(aR), np.asarray(aZ),
            float(cR[0]), float(cZ[0]), lr, lth)


#: Choices for :func:`surface_label`, best first.
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
