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

import diffrax as dfx
import jax
import jax.numpy as jnp
import numpy as np

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
    # dof @ (E @ V) == (E^T dof) @ V: fold the extraction into the coefficients
    # once instead of applying it at every one of the millions of evaluations.
    weights = extraction.T @ jnp.asarray(dof)
    ns = basis.ns

    if k == 2:
        def field(x):
            return weights @ jax.vmap(basis, (None, 0))(x, ns)
    else:
        def field(x):
            a = weights @ jax.vmap(basis, (None, 0))(x, ns)
            df = jax.jacfwd(seq.map)(x)
            return jnp.linalg.solve(df.T @ df, a)
    return field


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

#: Half-split |d iota| above which a line is treated as chaotic and given NO
#: iota. Measured over the archived traces: quasi-periodic lines score ~1e-6
#: (median) with a p90 of ~1e-5, while the chaotic quasr65530 k=1 sea scores
#: 5.6e-04 median. Three orders of magnitude of separation, so the threshold is
#: not delicate.
CHAOS_TOL = 1e-4


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


def enclosed_area(R, Z, centre_R, centre_Z):
    """Section area enclosed by each surface, by shoelace on the crossings.

    The surface label has to be map-independent for two runs to be comparable:
    the seed radius ``r`` names a different surface as soon as the map changes,
    which is exactly what a resolution sweep or an interior perturbation does.
    Area is a property of the physical curve, so it is the same label in every
    run.

    Sorting the crossings by poloidal angle about the axis and running the
    shoelace formula converges to the true area from below as the crossings
    fill the curve.  It assumes the surface is star-shaped about the axis --
    true for a flux surface, false for an island chain, where the crossings are
    disjoint lobes and the number is meaningless.  The angle-fit residual from
    :func:`rotational_transform` is what flags those.
    """
    ang = jnp.arctan2(Z - centre_Z, R - centre_R)
    order = jnp.argsort(ang, axis=-1)
    Rs = jnp.take_along_axis(R, order, axis=-1)
    Zs = jnp.take_along_axis(Z, order, axis=-1)
    cross = Rs * jnp.roll(Zs, -1, axis=-1) - jnp.roll(Rs, -1, axis=-1) * Zs
    return 0.5 * jnp.abs(jnp.sum(cross, axis=-1))


def midplane_radius(R, Z, centre_R, centre_Z):
    """Distance from the magnetic axis to each surface on the OUTBOARD midplane.

    The surface label to prefer. Nested curves cross any fixed ray from the axis
    at strictly increasing distance, so this is monotone *by nesting* -- which
    neither :func:`mean_axis_distance` nor :func:`effective_radius` is.

    That is not a technicality. The mean averages over the CROSSING POINTS, and
    their distribution in poloidal angle is set by the field-line dynamics, not
    by the surface, so two properly nested surfaces can come out non-monotone
    from sampling weight alone. Fixing the ray removes the weighting entirely.

    It is a property of the physical curve, so it stays comparable across maps,
    unlike the seed radius.

    The ray is the outboard midplane THROUGH THE AXIS (``Z = centre_Z``), not
    ``Z = 0``; they coincide when the axis is on the midplane, which is the
    usual case here, and the axis-centred one keeps meaning when it is not.

    A closed surface meets the midplane TWICE, outboard at ``alpha = 0`` and
    inboard at ``alpha = +-pi``. Both argmins below minimise ``|alpha|``, so
    they bracket ``alpha = 0`` and the inboard crossing, sitting at the far end
    of the angle range, can never win -- ``arctan2``'s branch cut falling on the
    inboard midplane is what makes the two unambiguous.

    Outboard is a CONVENTION, not a robustness argument. The interpolation
    wants ``r(alpha)`` single-valued near the ray, and it was tempting to argue
    that the inboard side is the risky one because that is where a bean section
    carries its indentation. Measured on w7x, w7x-ini and hegna, that is wrong:
    the relative residual of a linear ``r(alpha)`` fit in a window either side
    is ~3e-4 on BOTH, and the inboard side is slightly the better behaved.
    Concave curvature is not the same as a ray crossing twice, and about the
    magnetic axis these sections are star-shaped either way.

    Returns NaN for a surface whose crossings do not straddle the ray -- that
    needs the orbit to miss an entire half-plane, so it is a real defect and is
    left visible rather than patched.
    """
    dR, dZ = R - centre_R, Z - centre_Z
    ang = jnp.arctan2(dZ, dR)
    rad = jnp.sqrt(dR ** 2 + dZ ** 2)

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

    straddles = (jnp.min(above, axis=-1) < jnp.inf) & (jnp.min(below, axis=-1) < jnp.inf)
    t = (0.0 - a_lo) / (a_hi - a_lo)
    return jnp.where(straddles, r_lo + t * (r_hi - r_lo), jnp.nan)


def mean_axis_distance(R, Z, centre_R, centre_Z):
    """Mean distance from the magnetic axis over a surface's crossings.

    The surface label of choice. Like :func:`effective_radius` it is a property
    of the physical curve, so two runs on different maps are comparable -- the
    seed radius is not, it names a different surface as soon as the map changes.
    Unlike the area it needs NO ordering of the crossings and makes no
    star-shape assumption, so it degrades gracefully: on an island chain or a
    broken trace it is still the mean radius of whatever was traced, where the
    shoelace area silently stops being monotone and the profile curve doubles
    back on itself.

    Equals the radius exactly on a circle, and lies between the semi-axes on an
    ellipse.
    """
    return jnp.mean(jnp.sqrt((R - centre_R) ** 2 + (Z - centre_Z) ** 2),
                    axis=-1)


def effective_radius(R, Z, centre_R, centre_Z):
    """``sqrt(area/pi)`` -- the enclosed area as a length."""
    return jnp.sqrt(enclosed_area(R, Z, centre_R, centre_Z) / jnp.pi)


def resonant_rationals(iota_min, iota_max, nfp, denom_max=15):
    """Rationals in ``[iota_min, iota_max]`` where an island chain can form.

    An island chain needs a resonant perturbation: ``iota = n/m`` with ``n`` the
    toroidal and ``m`` the poloidal mode number. A field with ``nfp`` field
    periods carries only toroidal harmonics ``n = 0 (mod nfp)``, so the only
    rationals that can open an island are those whose NUMERATOR is a multiple
    of ``nfp``. Every other rational surface is resonance-free and closes on
    itself harmlessly.

    Returned lowest-numerator-first and deduplicated by VALUE, so ``5/6`` is
    kept and ``10/12`` -- the same surface, driven by a weaker harmonic -- is
    not repeated.

    Ported from the ``denom_max`` ticks in :func:`mrx.plotting.poincare_plot`.
    """
    ticks, labels, seen = [], [], set()
    for j in range(1, max(denom_max // nfp, 1) + 1):
        n_tor = j * nfp
        for m_pol in range(1, denom_max + 1):
            value = n_tor / m_pol
            if iota_min <= value <= iota_max and value not in seen:
                ticks.append(value)
                labels.append(f"{n_tor}/{m_pol}")
                seen.add(value)
    order = sorted(range(len(ticks)), key=lambda i: ticks[i])
    return [ticks[i] for i in order], [labels[i] for i in order]


def render_section(R, Z, iota, resid, seed_r, keep, *, title, subtitle,
                   axis_RZ=None, path=None, profile_x=None,
                   profile_xlabel="seed radius $r$", nfp=None, denom_max=15,
                   logical=None, chaotic=None):
    """The two-panel figure: the section coloured by iota, and the profile.

    Pure arrays in, so a run can be re-rendered from its archive without
    rebuilding the map -- which is the expensive half of producing it.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415  (keep the module headless)

    if logical is None:
        fig = plt.figure(figsize=(11.5, 5.0), constrained_layout=True)
        ax, bx = fig.subplots(1, 2, width_ratios=[1.3, 1.0])
        lx = None
    else:
        fig = plt.figure(figsize=(15.5, 4.8), constrained_layout=True)
        ax, lx, bx = fig.subplots(1, 3, width_ratios=[1.15, 1.0, 1.15])

    # Chaotic lines are REAL and get plotted -- they are the physics of an
    # overlapped island region -- but they have no rotational transform, so
    # they must not be painted on the iota scale or fitted into the profile.
    # Dark grey, the convention for "iota could not be inferred".
    if chaotic is None:
        chaotic = jnp.zeros_like(keep)
    shown = keep & ~chaotic
    good = iota[shown][jnp.isfinite(iota[shown])] if shown.any() else iota[:0]
    lo, hi = (float(jnp.min(good)), float(jnp.max(good))) if good.size else (0.0, 1.0)
    if hi - lo < 1e-9:
        lo, hi = lo - 5e-3, hi + 5e-3

    # One marker per crossing: ~10^4 points want a hairline to show the surface
    # texture, ~10^2 want something you can actually see.
    npts = max(int(keep.sum()) * R.shape[1], 1)
    size = float(jnp.clip(3000.0 / npts, 0.35, 15.0))
    colour = jnp.broadcast_to(iota[:, None], R.shape)
    sc = ax.scatter(R[shown], Z[shown], c=colour[shown], s=size, vmin=lo,
                    vmax=hi, cmap="turbo", linewidths=0, rasterized=True)
    if (keep & chaotic).any():
        m = keep & chaotic
        ax.scatter(R[m], Z[m], c="0.25", s=size, linewidths=0, rasterized=True,
                   label=f"chaotic ({int(m.sum())})")
    res_ticks, res_labels = (resonant_rationals(lo, hi, int(nfp), denom_max)
                             if nfp else ([], []))
    if (~keep).any():
        ax.scatter(R[~keep], Z[~keep], c="0.55", s=size, linewidths=0,
                   rasterized=True, label=f"lost ({int((~keep).sum())})")
    if (~keep).any() or (keep & chaotic).any():
        ax.legend(loc="upper right", fontsize=7, markerscale=4)
    if axis_RZ is not None:
        ax.plot(axis_RZ[0], axis_RZ[1], "k+", ms=5, mew=0.8)
    cbar = fig.colorbar(sc, ax=ax, label=r"$\iota$", fraction=0.046, pad=0.02)
    if res_ticks:
        # Only the rationals an nfp-periodic field can actually resonate with:
        # everything else on the colorbar is a surface no island can open on.
        cbar.set_ticks(res_ticks)
        cbar.set_ticklabels(res_labels)

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
                   vmax=hi, cmap="turbo", linewidths=0, rasterized=True)
        if (keep & chaotic).any():
            m = keep & chaotic
            lx.scatter(lr[m], lth[m], c="0.25", s=size, linewidths=0,
                       rasterized=True)
        if (~keep).any():
            lx.scatter(lr[~keep], lth[~keep], c="0.55", s=size, linewidths=0,
                       rasterized=True)
        lx.set_xlim(0.0, 1.0)
        lx.set_ylim(0.0, 1.0)
        lx.set_xlabel(r"logical $r$")
        lx.set_ylabel(r"logical $\theta$")
        lx.set_title("logical chart", fontsize=10)

    x = seed_r if profile_x is None else profile_x
    bx.plot(x[shown], iota[shown], "o-", ms=3, lw=0.8)
    for value, lab in zip(res_ticks, res_labels):
        bx.axhline(value, color="0.55", lw=0.6, ls="--", zorder=0)
        bx.annotate(lab, (0.995, value), xycoords=("axes fraction", "data"),
                    ha="right", va="bottom", fontsize=6.5, color="0.4")
    bx.set_xlabel(profile_xlabel)
    bx.set_ylabel(r"$\iota$")
    bx.grid(alpha=0.3)
    bx2 = bx.twinx()
    bx2.semilogy(x[shown], jnp.maximum(resid[shown], 1e-16), ".", ms=4,
                 color="tab:red", alpha=0.7)
    bx2.set_ylabel("angle-fit residual [turns]", color="tab:red")
    bx.set_title(subtitle, fontsize=10)

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
                   r_edge=0.97, theta=0.0, probe_periods=64,
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
    """
    probe = jnp.array([[r_axis, theta], [r_edge, theta]])
    ys, _ = trace(field, probe, probe_periods, steps_per_period,
                  saves_per_period)
    centre = jnp.mean(ys[0, ::saves_per_period], axis=0)
    edge = jnp.array([r_edge * jnp.cos(TWO_PI * theta),
                      r_edge * jnp.sin(TWO_PI * theta)])

    t = jnp.linspace(t_min, 1.0, n_seeds)[:, None]
    uv = centre[None, :] + t * (edge - centre)[None, :]
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
