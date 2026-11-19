"""A map from a boundary alone: VMEC's initial surfaces between a guessed magnetic axis and the boundary.

The boundary is a stellarator-symmetric Fourier series, rows ``(m, n, R_mn, Z_mn)`` with

    R_b = sum R_mn cos(m u - n v),   Z_b = sum Z_mn sin(m u - n v),

``n`` a multiple of ``nfp``, ``v`` the geometric toroidal angle and ``u`` the boundary's own poloidal angle. The axis
is a curve, rows ``(n, R_n, Z_n)`` with ``R_a = sum R_n cos(n v)``, ``Z_a = -sum Z_n sin(n v)`` (VMEC's ``RAXIS_CC``,
``ZAXIS_CS``). :func:`fill_rows` fills the interior as VMEC initialises its surfaces,
``R_mn(rho) = rho^m R_mn`` for ``m >= 1`` and ``R_0n(rho) = R_n + rho^2 (R_0n - R_n)``, the same for ``Z``. In the
plane ``v = const`` this is the harmonic extension of the boundary into the disk ``rho <= 1`` plus a shift that moves
its centre onto the axis, so every mode has the behaviour of a smooth function at the axis. ``rho`` is the radial
coordinate of the map, not a flux label.

The fill folds when the axis sits in the wrong place (the area centroid of li383's bean does, by 7 % of the cells).
:func:`guess_axis` places it in each plane at the point where the mean of ``log J`` of the fill over the plane is
largest, a barrier that keeps ``J`` positive and away from zero everywhere, then fits the axis as a Fourier series in
``v`` with the boundary's toroidal modes. VMEC's own ``guess_axis`` maximises the smallest ``J`` instead. Its point
jumps from plane to plane, and on W7-X its axis carries 2 cm in the fourth toroidal harmonic, against 1e-4 m with the
mean of ``log J`` at the same smallest ``J``.

:func:`fill_state` is the state (:mod:`mrx.equilibria`) of the fill: ``X1``, ``X2`` as radial splines per mode
(:func:`mrx.equilibria.fit.fit_modes`), lambda zero and no field.
"""
from __future__ import annotations

import numpy as np

from mrx.equilibria.fit import fit_modes

#: The degree of the radial splines of the fill.
DEG = 5
#: The radii at which the fill is sampled for the radial fit.
N_RHO = 41


def fill_rows(boundary, axis, rho):
    """Return ``(m, n, R, Z)``: the modes of the fill between the ``axis`` rows and the ``boundary`` rows and its
    coefficients at the radii ``rho``, shape ``(len(rho), n_modes)``."""
    table = {(int(m), int(n)): (R, Z) for m, n, R, Z in np.asarray(boundary, dtype=np.float64)}
    # the axis in the boundary's form: cos(n v) = cos(0 u - (-n) v) and -sin(n v) = sin(0 u - n v)
    ax = {int(n): (R, Z) for n, R, Z in np.asarray(axis, dtype=np.float64)}
    for n in ax:
        table.setdefault((0, n), (0.0, 0.0))
    keys = sorted(table)
    m = np.array([k[0] for k in keys])
    n = np.array([k[1] for k in keys])
    Rb, Zb = (np.array([table[k][i] for k in keys]) for i in (0, 1))
    Ra, Za = (np.array([ax.get(k[1], (0.0, 0.0))[i] if k[0] == 0 else 0.0 for k in keys]) for i in (0, 1))
    r = np.asarray(rho, dtype=np.float64)[:, None]
    shape = np.where(m == 0, r ** 2, r ** m)
    return m, n, np.where(m == 0, Ra + shape * (Rb - Ra), shape * Rb), np.where(m == 0, Za + shape * (Zb - Za), shape * Zb)


def guess_axis(boundary, nfp, n_u=128, n_rho=32, grid=61):
    """The axis rows ``(n, R_n, Z_n)`` for :func:`fill_rows`: in each of ``4 (N + 1)`` planes per field period
    (``N`` the boundary's largest toroidal mode number per period) the point that maximises the mean of
    ``log(J / rho)`` of the fill, ``J = R_rho Z_u - R_u Z_rho``, over ``n_rho`` radii and ``n_u`` angles (minus
    infinity where ``J`` is not positive), found on a ``grid`` x ``grid`` grid over the section's box and once more on
    a finer grid around the best point. The axis is the least-squares Fourier series with ``n <= N`` through these
    points."""
    b = np.asarray(boundary, dtype=np.float64)
    m, n, Rb, Zb = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    N = int(np.abs(n).max() / nfp)
    u = np.linspace(0.0, 2.0 * np.pi, n_u, endpoint=False)
    rho = np.linspace(0.0, 1.0, n_rho + 1)[1:, None]
    shape = np.where(m == 0, rho ** 2, rho ** m)
    dshape = np.where(m == 0, 2.0 * rho, m * rho ** np.maximum(m - 1, 0))
    v = 2.0 * np.pi / nfp * np.arange(4 * (N + 1)) / (4 * (N + 1))
    points = []
    for vk in v:
        arg = m * u[:, None] - n * vk
        cos, sin = np.cos(arg), np.sin(arg)
        # with the axis a the fill is a (1 - rho^2) + sum shape_m b_m, so only R_rho, Z_rho see the axis
        Rr, Zr = dshape @ (cos * Rb).T, dshape @ (sin * Zb).T
        Ru, Zu = shape @ (-m * sin * Rb).T, shape @ (m * cos * Zb).T
        R, Z = cos @ Rb, sin @ Zb
        sign = np.sign(np.sum(R * np.gradient(Z)))         # the orientation of u in this plane

        def barrier(aR, aZ):
            J = sign * ((Rr - 2.0 * rho * aR[:, None, None]) * Zu - Ru * (Zr - 2.0 * rho * aZ[:, None, None])) / rho
            folded = (J <= 0.0).any(axis=(1, 2))
            return np.where(folded, -np.inf, np.log(np.where(J > 0.0, J, 1.0)).mean(axis=(1, 2)))
        cR, cZ, hR, hZ = R.mean(), Z.mean(), np.ptp(R) / 2, np.ptp(Z) / 2
        for refine in (1.0, 2.0 / grid):
            gR, gZ = np.meshgrid(cR + hR * refine * np.linspace(-1, 1, grid), cZ + hZ * refine * np.linspace(-1, 1, grid),
                                 indexing="ij")
            best = np.argmax(np.concatenate([barrier(a, z) for a, z in zip(np.array_split(gR.ravel(), grid),
                                                                              np.array_split(gZ.ravel(), grid))]))
            cR, cZ = gR.ravel()[best], gZ.ravel()[best]
        points.append((cR, cZ))
    points = np.array(points)
    k = np.arange(N + 1) * nfp
    R_n = np.linalg.lstsq(np.cos(np.outer(v, k)), points[:, 0], rcond=None)[0]
    Z_n = np.concatenate([[0.0], np.linalg.lstsq(-np.sin(np.outer(v, k[1:])), points[:, 1], rcond=None)[0]])
    return np.stack([k, R_n, Z_n], axis=1)


def fill_state(boundary, axis, nfp):
    """The state of the fill (:mod:`mrx.equilibria`): ``X1 = R`` and ``X2 = Z`` per mode as radial splines of degree
    :data:`DEG` through the fill at :data:`N_RHO` radii, lambda zero, no profiles. ``boundary`` and ``axis`` are
    kept in the state."""
    rho = np.linspace(0.0, 1.0, N_RHO)
    m, n, R, Z = fill_rows(boundary, axis, rho)
    T, cos_R = fit_modes(rho, R, m, DEG)
    sin_Z = fit_modes(rho, Z, m, DEG)[1]
    zero = np.zeros_like(cos_R)

    def block(cos, sin):
        return dict(m=m, n=n, cos=cos, sin=sin, deg=DEG, T=T)

    return dict(nfp=int(nfp), X1=block(cos_R, zero), X2=block(zero, sin_Z), LA=block(zero, zero), profiles={},
                boundary=np.asarray(boundary), axis=np.asarray(axis))
