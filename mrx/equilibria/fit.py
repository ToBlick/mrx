"""Radial B-spline interpolation of data given on a set of flux surfaces, used by the VMEC reader.

:func:`fit_modes` turns the values of Fourier modes on the surfaces into one clamped B-spline in ``r``
per mode. Each spline has the behaviour at the axis that a smooth field requires of its mode ``m``.
:func:`fit_profile` does the same for a radial profile, which is a smooth function of ``s = r^2``.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline

from mrx.geometry import knot_vector


def axis_orders(m, deg):
    """The derivative orders among ``1..deg`` that vanish at the axis for a mode of poloidal number ``m``.
    The mode of a smooth field is ``r^m`` times an even function of ``r``. Hence every derivative of order
    below ``m``, and every order of the other parity than ``m``, is zero at ``r = 0``."""
    return tuple(j for j in range(1, deg + 1) if j < m or (j - m) % 2 == 1)


def fit_modes(r, samples, m, deg):
    """Interpolate the values of Fourier modes on flux surfaces by radial splines of degree ``deg``.

    ``samples`` has shape ``(len(r), n_modes)``, where the surfaces ``r`` increase from ``r[0] = 0`` to
    ``r[-1] = 1`` and ``m`` holds the poloidal mode numbers. Each mode's spline passes through its samples
    and has the derivatives of :func:`axis_orders` zero at ``r = 0``. All modes share one knot vector
    ``T``. Returns ``(T, coef)`` with ``coef`` of shape ``(n_modes, n_base)``."""
    # A mode with fewer axis conditions is fitted on T with some innermost interior knots removed. That
    # space is a subspace of the shared one, so re-expressing the fit on T is exact.
    m = np.asarray(m)
    groups = {}
    for i, mi in enumerate(m):
        groups.setdefault(axis_orders(int(mi), deg), []).append(i)
    k_max = max(len(o) for o in groups)
    # k_max phantom nodes in the first interval carry the axis conditions
    nodes = np.sort(np.concatenate([r, r[0] + (r[1] - r[0]) * (np.arange(1, k_max + 1) / (k_max + 1))]))
    # de Boor's knot averaging: interpolation through any increasing sample is well posed on it
    T = knot_vector(np.concatenate([[0.0], [nodes[j:j + deg].mean() for j in range(1, len(nodes) - deg)], [1.0]]),
                    deg, False)
    n_base = len(nodes)
    greville = np.array([T[j + 1:j + deg + 1].mean() for j in range(n_base)])
    A_union = BSpline.design_matrix(greville, T, deg).toarray()
    coef = np.empty((n_base, samples.shape[1]))
    for orders, cols in groups.items():
        drop = k_max - len(orders)
        T_m = np.delete(T, np.arange(deg + 1, deg + 1 + drop))
        n_m = n_base - drop
        eye = np.eye(n_m)
        rows = [BSpline.design_matrix(r, T_m, deg).toarray()]
        rows += [np.array([BSpline(T_m, eye[j], deg).derivative(o)(0.0) for j in range(n_m)])[None, :]
                 for o in orders]
        rhs = np.vstack([samples[:, cols], np.zeros((len(orders), len(cols)))])
        c_m = np.linalg.solve(np.vstack(rows), rhs)                 # (n_m, len(cols))
        coef[:, cols] = np.linalg.solve(A_union, BSpline.design_matrix(greville, T_m, deg).toarray() @ c_m)
    return T, coef.T


def fit_profile(r, values, deg):
    """The scipy ``BSpline`` of degree ``deg`` through a profile's ``values`` on the surfaces ``r``. Its odd
    derivatives vanish at the axis, as for an ``m = 0`` mode in :func:`fit_modes`."""
    T, coef = fit_modes(r, np.asarray(values, dtype=np.float64)[:, None], np.zeros(1, dtype=int), deg)
    return BSpline(T, coef[0], deg)
