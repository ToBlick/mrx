"""Experimental island diagnostics: where a chain is, how wide, and the straight-field-line angle.

None of this is on the production path (relax.py, the tutorials, the paper's figures); it is the toolbox behind the
island studies, kept here so :mod:`mrx.poincare` stays the section-tracing module (Tobias 2026-09-19).

* :func:`fixed_points` -- Newton on the return map, with the tangent map from the variational equation, and Greene's
  residue of each fixed point. :func:`resonances` lists the chains that can sit in an iota range.
* :func:`islands` -- every chain of a field and its width, MEASURED on a ray through the O-point. The residue gives
  existence, phase and rotation, never a width: on the paper's fields the width inferred from it overestimates the
  traced one by 1.35 to 1.6.
* :func:`straight_field_line_lambda` -- theta* = theta + lambda per logical surface, by least squares on the
  magnetic differential equation. CAVEAT: the logical surfaces are not flux surfaces. Near a rational the true
  surfaces are displaced from them, so resonant quantities taken on a logical surface (a resonant B^r, say) carry
  that displacement and are not island amplitudes; the field-line slope in theta* still becomes constant to 0.1-0.4%
  against 1% for the equilibrium file's lambda (2026-09-19).
"""
import numpy as np
from functools import partial
from math import gcd

import diffrax as dfx
import jax
import jax.numpy as jnp

from mrx.poincare import (MIN_STEPS_PER_PERIOD, R_AXIS, R_EDGE, cross_section_rhs, logical_field, poincare,
                          rotational_transform, to_polar, to_uv, trace)

#: Steps per period for the tangent map of the return map (:func:`fixed_points`):
#: the variational equation needs more than the trajectory. On the seeded li383
#: (6,1) chain at (10,16,16) the O-point's residue moved 5% from 24 to 48 steps
#: and 1% from 48 to 96, where ``det S`` is 1 to 5e-5 (2026-09-18).
TANGENT_STEPS_PER_PERIOD = 96



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
    y0 = to_uv(guesses[:, 0], guesses[:, 1])
    ys, defect, S = _fixed_points(field, dof, y0, jnp.asarray(int(periods)), int(steps_per_period), int(iters),
                                  float(step_cap))
    residue = 0.5 - jnp.trace(S, axis1=1, axis2=2) / 4.0
    kind = np.where(residue < 0.0, "X", np.where(residue < 1.0, "O", "reflecting"))
    r, theta = to_polar(ys)
    return {"r": np.asarray(r), "theta": np.asarray(theta),
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
    the radius) AND lines locked to the chain pass through it; a closed
    rational surface has residue zero up to integration error and no
    locked line, and is not.

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
    return sorted((c for c in chains if c["width"] > 0.0), key=lambda c: c["r_chain"])


def straight_field_line_lambda(bh, m_max=24, k_max=8):
    r"""lambda of a field on every LOGICAL surface (not a flux surface: see the module docstring): the straight-field-line condition
    B_hat^theta (1 + d_theta lambda) + B_hat^zeta d_zeta lambda = iota_p B_hat^zeta, i.e.
    (u d_theta + d_zeta) lambda = iota_p - u with u = B_hat^theta / B_hat^zeta, iota_p the flux ratio, solved in the
    least-squares sense on the (theta, zeta) grid per surface for lambda = sum c_mk sin 2 pi (m theta + k zeta)
    (stellarator symmetry: u even, lambda odd), m <= m_max, |k| <= k_max. Least squares, not the fixed point: next to
    a rational the (m, -n) divisor m iota_p - n is small and the fixed point diverges; there the least-squares
    residual is the resonant forcing it cannot absorb. Returns lambda, d_theta lambda, d_zeta lambda and the rms
    residual per surface (relative to iota_p)."""
    nr, nt, nz, _ = bh.shape
    th = np.arange(nt) / nt
    ze = np.arange(nz) / nz
    TH, ZE = np.meshgrid(th, ze, indexing="ij")
    modes = [(0, k) for k in range(1, k_max + 1)] + [(m, k) for m in range(1, m_max + 1)
                                                        for k in range(-k_max, k_max + 1)]
    mm = np.array([m for m, _ in modes], float)
    kk = np.array([k for _, k in modes], float)
    arg = 2 * np.pi * (TH.ravel()[:, None] * mm + ZE.ravel()[:, None] * kk)       # (pts, modes)
    phi, dphi = jnp.asarray(np.sin(arg)), jnp.asarray(np.cos(arg) * 2 * np.pi)
    u = jnp.asarray((bh[..., 1] / bh[..., 2]).reshape(nr, -1))
    iota_p = jnp.asarray(bh[..., 1].mean(axis=(1, 2)) / bh[..., 2].mean(axis=(1, 2)))
    mmj, kkj = jnp.asarray(mm), jnp.asarray(kk)

    @jax.jit
    def solve(u_s, i_s):
        a = dphi * (u_s[:, None] * mmj + kkj)                   # (pts, modes)
        b = i_s - u_s
        ata = a.T @ a
        c = jnp.linalg.solve(ata + 1e-12 * jnp.trace(ata) / ata.shape[0] * jnp.eye(ata.shape[0]), a.T @ b)
        return c, jnp.sqrt(jnp.mean((a @ c - b) ** 2)) / jnp.abs(i_s)

    cs, res = [], []
    for j in range(nr):
        c, rr = solve(u[j], iota_p[j])
        cs.append(c), res.append(rr)
    c = jnp.stack(cs)                                            # (nr, modes)
    lam = np.asarray(c @ phi.T).reshape(nr, nt, nz)
    dth = np.asarray((c * mmj) @ dphi.T).reshape(nr, nt, nz)
    dze = np.asarray((c * kkj) @ dphi.T).reshape(nr, nt, nz)
    return lam, dth, dze, np.asarray(jnp.stack(res))
