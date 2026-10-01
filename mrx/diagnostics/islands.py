"""Magnetic island diagnostics: where the island chains of a field are and how wide they are.

* :func:`islands` is the main entry point. It finds every island chain of a field from a Poincare section
  (:func:`mrx.diagnostics.poincare.poincare`) and measures the width of each chain by tracing field lines
  across it.
* :func:`fixed_points` finds the O- and X-points of a chain, the points where a field line closes on itself
  after a given number of field periods, together with Greene's residue of each point. The residue tells
  whether a chain exists and where its O-points are, but not how wide it is.
* :func:`straight_field_line_lambda` computes the straight-field-line angle ``theta* = theta + lambda`` on
  every logical surface. The logical surfaces are not flux surfaces, so near a rational surface the resonant
  part of lambda is not an island amplitude.
"""
import numpy as np
from functools import partial

import diffrax as dfx
import jax
import jax.numpy as jnp

from mrx.relaxation.seeding import resonances
from mrx.diagnostics.poincare import (MIN_STEPS_PER_PERIOD, R_AXIS, R_EDGE, cross_section_rhs, logical_field, poincare,
                          rotational_transform, to_polar, to_uv, trace)

#: Steps per field period in :func:`fixed_points`. The derivative of the return map needs finer steps than the
#: field line itself.
TANGENT_STEPS_PER_PERIOD = 96


def _period_map(field, steps_per_period):
    """The one-period map ``(y, dof) ->`` the ``(u, v)`` point one field period after ``y`` at ``zeta = 0``,
    differentiable in forward mode. The Poincare map ``P`` over ``m`` periods is this map applied ``m``
    times."""
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
    """Newton's method for ``poincare_map(y) = y``, the ``m``-period Poincare map, from every row of ``y0s``. Returns
    ``(y, |poincare_map(y) - y|, D poincare_map(y))``. ``m = periods`` and the field coefficients ``dof`` are traced arguments, so one
    compile serves every chain and every field on a sequence."""
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
    """The O- and X-points of an island chain that closes after ``periods`` field periods, found by Newton's
    method from the logical ``(r, theta)`` ``guesses`` at ``zeta = 0``, and Greene's residue of each point
    (Cary & Hanson, Phys. Fluids 29, 2464 (1986)).

    Write ``poincare_map`` for the map that takes a point of the plane ``zeta = 0`` to the point where its field
    line returns after ``periods`` periods, and ``S`` for its Jacobian at a fixed point. The residue
    ``R = 1/2 - tr(S) / 4`` classifies the point: ``0 < R < 1`` is an O-point (nearby lines rotate about it by
    ``2 pi nu`` per return, with ``R = sin^2(pi nu)``), ``R < 0`` an X-point and ``R > 1`` an X-point with
    reflection. Newton runs ``iters`` steps from every guess, each step at most ``step_cap`` long in the
    ``(u, v)`` chart.

    Returns a dict of arrays over the guesses: ``r``, ``theta`` (logical, at ``zeta = 0``), ``uv``,
    ``residue``, ``det``, ``defect`` and ``kind`` (``"O"``, ``"X"`` or ``"reflecting"``). ``defect`` is
    ``|poincare_map(y) - y|`` at the end, so a large value means Newton did not converge. ``det S`` is exactly 1 for a
    divergence-free field, and its distance from 1 measures the integration error.
    """
    field, dof = logical_field(seq), jnp.asarray(dof)
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


def _chain_radii(r, iota, target, tol):
    """The radii at which the iota profile meets the rational ``target``. A run of neighbouring lines with iota
    within ``tol`` of it gives its mean radius, and a sign change of ``iota - target`` between two neighbours
    gives the interpolated crossing. A profile with reversed shear can meet a rational twice."""
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
    """Every island chain of the field ``dof`` on ``seq``, with its measured width.

    ``res`` is a Poincare section of the same field (the result of :func:`poincare`). The iota profile of its
    regular lines decides where to look. With ``res=None`` the section is traced here with the defaults of
    :func:`poincare`. A chain inside a chaotic band, with no regular line on either side of its rational, is
    not found.

    For every rational ``iota = nfp n / m`` with ``m <= m_max`` inside the iota range of the section
    (:func:`~mrx.relaxation.seeding.resonances`), and every radius where the profile meets it,
    :func:`fixed_points` starts from ``n_theta`` poloidal guesses spread over one poloidal period ``1/m`` of
    the chain. Fixed points further than ``window`` from that radius are dropped, and those left are grouped by
    radius, one group per chain. A group is reported as a chain if it has an O-point with residue above
    ``residue_min`` and if field lines are locked to the chain, that is their iota lies within ``tol`` of the
    rational. An intact rational surface also consists of fixed points, with residue near zero, but has no
    locked lines, and is not reported.

    The width is measured, not inferred from the residue. ``ray_seeds`` lines are started on the radial ray
    through the O-point, within ``ray_halfwidth`` of it, and traced for ``periods`` field periods. The locked
    lines next to the O-point form the island. ``width`` is the largest radial excursion ``max(r) - min(r)``
    of one of these lines, sampled at eight planes per period, and ``ray`` is the radial extent of the locked
    lines on the ray itself, from ``ray_lo`` to ``ray_hi``.

    Returns a list of dicts sorted by radius, with keys ``m``, ``n``, ``iota``, ``r_chain`` (the mean radius of
    the O-points), ``O`` and ``X`` (lists of ``(r, theta, residue)``), ``residue`` (the largest O-point
    residue), ``width``, ``ray``, ``ray_lo``, ``ray_hi`` and ``n_locked`` (the number of locked lines).
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
        # Every radius at which the profile meets the rational is only a starting guess. A flattened profile
        # meets it several times across one island, and Newton then finds the same chain from all of them.
        # So pool the fixed points of all guesses, drop repeats and split the rest by radius. There are two
        # groups only where the same rational really resonates at two radii.
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
    field, dof = logical_field(seq), jnp.asarray(dof)
    steps = MIN_STEPS_PER_PERIOD
    rays = [np.linspace(max(c["O"][0][0] - ray_halfwidth, R_AXIS), min(c["O"][0][0] + ray_halfwidth, R_EDGE), ray_seeds)
            for c in chains]
    seeds = np.concatenate([np.stack([ray, np.full_like(ray, c["O"][0][1])], axis=1) for ray, c in zip(rays, chains)])
    ys, _ = trace(field, dof, seeds, periods, steps)
    line_iota = rotational_transform(ys, steps, nfp)
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
    # An intact rational surface is a curve of fixed points with residue zero up to integration error, which
    # can exceed residue_min at high m. Only a real island has lines locked to it.
    return sorted((c for c in chains if c["width"] > 0.0), key=lambda c: c["r_chain"])


def straight_field_line_lambda(bh, m_max=24, k_max=8):
    r"""The straight-field-line angle correction lambda on every logical surface of a field.

    ``bh`` holds the logical contravariant components ``B_hat`` of the field on a grid of shape
    ``(n_r, n_theta, n_zeta, 3)``, equispaced in theta and zeta. On each logical surface lambda solves
    ``B_hat^theta (1 + d_theta lambda) + B_hat^zeta d_zeta lambda = iota_p B_hat^zeta``, where ``iota_p`` is the
    ratio of the surface means of ``B_hat^theta`` and ``B_hat^zeta``. lambda is expanded as ``sum c_mk sin 2 pi
    (m theta + k zeta)`` with ``m <= m_max`` and ``|k| <= k_max``, which assumes stellarator symmetry.

    The equation is solved in the least-squares sense. Near a rational surface the exact solution blows up
    (the divisor ``m iota_p - n`` is small), and the least-squares residual then measures the resonant
    forcing that lambda cannot absorb. The logical surfaces are not flux surfaces (see the module docstring).

    Returns ``(lambda, d_theta lambda, d_zeta lambda, residual)``, the first three on the grid of ``bh`` and
    ``residual`` the rms residual per surface relative to ``iota_p``.
    """
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
