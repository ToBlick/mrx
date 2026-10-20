"""Turning a given function into a discrete ``k``-form by commuting projections. Users call this through
``seq.interpolate``. (``seq.load`` gives the right-hand side of an L2 projection instead.)

:func:`interpolate` returns the coefficients of a form directly, by interpolation at the Greville points
(k = 0) or by matching integrals over Greville cells (histopolation, k = 1, 2, 3). On the tensor-product
spline spaces these projections commute with the exterior derivative. The result is then restricted to the
polar (axis-regular) subspace the sequence works in. The function is evaluated at logical points and returns
the logical components of the form.

On a graded space the restriction onto a ring with fewer theta cells is the coarse ring's own projection in
theta: interpolation at its Greville points of the tensor-product splines, or histopolation over its Greville
cells of the derivative splines. Where the theta count grows between two primal rings, the ring of radial
derivative splines between them also takes a term from the theta derivative of the inner ring
(:func:`ring_projections`). With these, the restriction commutes with the derivative on every ring off the axis.
The sequence builds the matrices once as ``seq.ring_projections``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array


import mrx
from mrx.extraction_operators import conforming_restriction, prolongation, ring_layout
from mrx.spline_bases import DerivativeSpline, SplineBasis, basis_table

if TYPE_CHECKING:
    from mrx.derham_sequence import DeRhamSequence


def _as_single_component(values: Array) -> Array:
    """A scalar or length-1 array reshaped to ``(1,)``."""
    return jnp.reshape(jnp.asarray(values), (1,))


def _solve_tensor_collocation_axis(matrix: Array, values: Array, axis: int) -> Array:
    """Solve with the 1D matrix ``matrix`` along axis ``axis`` of the tensor ``values``."""
    moved = jnp.moveaxis(values, axis, 0)
    solved = jnp.linalg.solve(matrix, moved.reshape(matrix.shape[0], -1))
    return jnp.moveaxis(solved.reshape(moved.shape), 0, axis)


def _span_quadrature(basis, spans: Array, knots=None) -> tuple[Array, Array]:
    """Gauss points and weights ``(xs, ws)``, ``(n_spans, n_pts)``, that integrate the splines of ``basis`` exactly
    over every span. The pieces of a span end at the ``knots`` inside it, by default those of ``basis``."""
    # split at the knots inside a span (at even p a span straddles a knot), padded to the widest span with
    # zero-width pieces
    xi_ref, w_ref = np.polynomial.legendre.leggauss(basis.p + 2)
    spans = np.asarray(spans)
    knots = np.unique(np.asarray(basis.T)) if knots is None else np.asarray(knots)
    cuts = [np.concatenate([[a], knots[(knots > a) & (knots < b)], [b]])
            for a, b in spans]
    width = max(len(c) for c in cuts)
    cuts = np.stack([np.concatenate([c, np.full(width - len(c), c[-1])])
                     for c in cuts])
    lo, hi = cuts[:, :-1], cuts[:, 1:]
    centers = 0.5 * (lo + hi)
    halfwidths = 0.5 * (hi - lo)
    xs = (centers[:, :, None] + halfwidths[:, :, None] * xi_ref).reshape(len(spans), -1)
    ws = (halfwidths[:, :, None] * w_ref).reshape(len(spans), -1)
    return jnp.asarray(xs), jnp.asarray(ws)


def _histopolation(basis, span_rule) -> Array:
    """The ``(n_spans, n)`` integrals of every function of ``basis`` over every span of ``span_rule``."""
    xs, ws = span_rule
    values = basis_table(basis, xs.reshape(-1)).reshape((basis.n,) + xs.shape)
    return jnp.einsum('sq,isq->si', ws, values)


class _GrevilleAxis(NamedTuple):
    """What Greville interpolation needs about one logical axis."""
    coll: Array          # (n, n)   parent-basis collocation matrix at the Greville points
    hist: Array          # (nd, nd) derivative-basis histopolation matrix on the Greville spans
    point_rule: tuple    # (pts[:, None], ones): one point per cell, unit weight
    span_rule: tuple     # (xs, ws) of _span_quadrature: one Gauss rule per cell


def greville_axes(seq) -> tuple[_GrevilleAxis, _GrevilleAxis, _GrevilleAxis]:
    """The per-axis data of :func:`interpolate` for ``seq``. The sequence builds it once and keeps it as
    ``seq.greville``."""
    axes = []
    for lam, d in zip(seq.basis_0.Lambda, seq.basis_0.dLambda):
        pts = lam.greville_points()
        span_rule = _span_quadrature(d, d.greville_spans())
        axes.append(_GrevilleAxis(
            coll=lam.collocation_matrix(pts),
            hist=_histopolation(d, span_rule),
            point_rule=(pts[:, None], jnp.ones((pts.shape[0], 1), dtype=pts.dtype)),
            span_rule=span_rule))
    return tuple(axes)


def _wrap_periodic_point(seq, xi):
    """The logical point ``xi`` with its periodic coordinates wrapped into ``[0, 1)``."""
    # periodic Greville spans run past x = 1 (at even p the last one is [1 - h/2, 1 + h/2])
    wrapped = []
    for axis, basis in enumerate(seq.basis_0.Lambda):
        coord = xi[axis]
        if basis.type == 'periodic':
            coord = jnp.mod(coord, 1.0)
        wrapped.append(coord)
    return jnp.asarray(wrapped)


def _wrapped(seq, v, k):
    """``v`` with its periodic coordinates wrapped into ``[0, 1)`` first, as a vector (of length one at k = 3)."""
    if k == 3:
        return lambda x: _as_single_component(v(_wrap_periodic_point(seq, x)))
    return lambda x: v(_wrap_periodic_point(seq, x))


def _greville_moments(fn, rules) -> Array:
    """The integrals of the scalar ``fn`` over every cell of the tensor-product rule ``rules``,
    ``(n_r, n_t, n_z)``."""
    sizes = [xs.shape[0] for xs, _ in rules]
    idx = [i.ravel() for i in jnp.meshgrid(
        *[jnp.arange(n) for n in sizes], indexing='ij')]
    cells = tuple((xs[i], ws[i]) for (xs, ws), i in zip(rules, idx))

    def integrate(cell):
        (xr, wr), (xt, wt), (xz, wz) = cell
        rr, tt, zz = jnp.meshgrid(xr, xt, xz, indexing='ij')
        x = jnp.stack([rr.ravel(), tt.ravel(), zz.ravel()], axis=-1)
        w = (wr[:, None, None] * wt[None, :, None] * wz[None, None, :]).ravel()
        return jnp.sum(jax.vmap(fn)(x) * w)

    return jax.lax.map(
        integrate, cells, batch_size=mrx.MAP_BATCH_SIZE_INNER).reshape(sizes)


#: Per form degree, the component with radial derivative splines and primal theta splines, its partner with primal
#: radial splines and theta derivative splines, and the sign of the partner's theta difference in the derivative
#: of the next lower degree: the r-component of a 1-form is ``c_{a+1} - c_a`` beside the theta-component
#: ``+D_theta c_a`` of a gradient, and the ``dzeta dr``-component of a 2-form is ``d_zeta a - d_r c`` beside the
#: ``dtheta dzeta``-component ``d_theta c - d_zeta b`` of a curl.
_JUMP_PARTNER = {1: (0, 1, 1.0), 2: (1, 0, -1.0)}


def _theta_projection(theta, m, derivative):
    """The ``(m, n)`` matrix that projects the periodic theta splines ``theta`` (``n`` cells) onto those with ``m``
    cells, ``n = m 2^l``: interpolation at the coarse Greville points, or with ``derivative`` histopolation of the
    derivative splines over the coarse Greville cells. The two commute with the coefficient difference."""
    n, p = theta.n, theta.p
    if m == n:
        return np.eye(n)
    coarse = SplineBasis(m, p, "periodic")
    if not derivative:
        points = coarse.greville_points()
        return np.linalg.solve(np.asarray(coarse.collocation_matrix(points)),
                               np.asarray(theta.collocation_matrix(points)))
    fine_d, coarse_d = DerivativeSpline(theta), DerivativeSpline(coarse)
    # the last coarse span ends up to half a coarse cell past 1, beyond the knot vector of the raw splines
    rule = _span_quadrature(fine_d, coarse_d.greville_spans(), knots=np.arange(-n, 2 * n + 1) / n)
    return np.linalg.solve(np.asarray(_histopolation(coarse_d, rule)), np.asarray(_histopolation(fine_d, rule)))


def _jump_correction(theta, m_in, m_out):
    """The ``(m_out, n)`` matrix ``K = (rho^{m_out} - R rho^{m_in}) D^+`` for primal rings of ``m_in`` and ``m_out``
    theta cells: ``rho`` are the interpolations of :func:`_theta_projection`, ``R`` the knot insertion from ``m_in``
    to ``m_out`` cells and ``D^+`` the inverse of the periodic coefficient difference on the ``n`` raw cells. ``D^+``
    takes the mean out of the differences and sums them up, so ``D^+ D c`` is ``c`` up to a constant and ``D^+``
    vanishes on constants. Both projections keep constants, so ``K D = rho^{m_out} - R rho^{m_in}``. ``K`` vanishes
    on the knot insertion of the derivative splines of ``m_in`` cells, so the projection onto the graded space keeps
    its forms."""
    n = theta.n
    # an explicit D^+: the SVD of D leaves its zero singular value at 1e-15 and np.linalg.pinv inverts it
    inverse_difference = np.tri(n, k=-1) @ (np.eye(n) - 1.0 / n)
    gap = (_theta_projection(theta, m_out, False)
           - prolongation(m_in, m_out, theta.p, False) @ _theta_projection(theta, m_in, False))
    return gap @ inverse_difference


def ring_projections(seq):
    """The restriction of :func:`interpolate` onto the rings of a graded space with fewer theta cells than the wall,
    per ``(k, dirichlet)``, as groups ``(rows, cols, matrix)``. The DoFs ``rows`` (shape ``(g, m, n_z)``) of ``g``
    rings with ``m`` theta cells take ``matrix @ c[cols]`` of the raw coefficients ``c``, ``cols`` of shape
    ``(g, n_cols, n_z)``.

    A ring takes the coarse theta projection of its raw coefficients (:func:`_theta_projection`). A ring of radial
    derivative splines and primal theta splines between primal rings of ``m_in < m_out`` cells (k = 1, 2) also takes
    the correction :func:`_jump_correction` of the theta derivative on its inner primal ring, so that the
    restriction commutes with the radial difference. Rings with ``n_theta`` cells and no such correction keep their
    raw coefficients and are left out, all of them on a uniform space. The restriction commutes with the reflection
    ``theta -> -theta``, so a parity view reduces its result like the coefficient fit of ``seq.conforming``."""
    counts = seq.theta_counts
    theta = seq.basis_0.Lambda[1]
    built = {}
    for k, form in enumerate((seq.basis_0, seq.basis_1, seq.basis_2, seq.basis_3)):
        offsets = np.cumsum([0] + [math.prod(s) for s in form.shape])
        c_jump, c_partner, sign = _JUMP_PARTNER.get(k, (None, None, 0.0))

        def raw(c, a):
            nt, nz = form.shape[c][1], form.shape[c][2]
            return offsets[c] + np.ravel_multi_index((a, np.arange(nt)[:, None], np.arange(nz)[None, :]), form.shape[c])

        for dirichlet in (False, True):
            groups = {}
            records = () if counts is None else ring_layout(form, dirichlet, counts)
            for c, a, m, derivative, first, nz in records:
                m_in = counts[a] if c == c_jump else m
                if m == theta.n and m_in == m:
                    continue
                rows = first + np.arange(m)[:, None] * nz + np.arange(nz)[None, :]
                cols = raw(c, a) if m_in == m else np.concatenate([raw(c, a), raw(c_partner, a)])
                groups.setdefault((c, m, m_in, derivative), []).append((rows, cols))
            built[(k, dirichlet)] = tuple(
                (jnp.asarray(np.stack([r for r, _ in rings])), jnp.asarray(np.stack([q for _, q in rings])),
                 jnp.asarray(_theta_projection(theta, m, derivative) if m_in == m else
                             np.hstack([_theta_projection(theta, m, False), sign * _jump_correction(theta, m_in, m)])))
                for (c, m, m_in, derivative), rings in groups.items())
    return built


def interpolate(seq: "DeRhamSequence", f, k: int):
    """The coefficients of the ``k``-form that interpolates ``f`` (k = 0) or matches its integrals over the
    Greville cells (k = 1, 2, 3), in the ``k``-form space of ``seq`` (the Dirichlet space, or the free one
    on ``seq.free``). ``f`` returns the logical components of the form at a logical point (for k = 3 the
    density times ``det DPhi``).

    The tensor-product coefficients are restricted to the polar space by the coefficient fit
    ``(E E^T)^-1 E`` on the axis rows and by :func:`ring_projections` on the graded rings. Off the axis
    ``G_k`` of the result is the result for ``df``, on uniform and graded spaces alike. The axis rows commute
    only when the tensor-product coefficients of the two innermost rings lie in the span of the axis functions."""
    axes = seq.greville
    logical = _wrapped(seq, f, k) if k else None
    coeffs = []
    for c in range(3 if k in (1, 2) else 1):
        # per axis: histopolated (span rule, histopolation matrix) or collocated (point, collocation matrix)
        hist = [(j == c) if k == 1 else (j != c) if k == 2 else k == 3 for j in range(3)]
        if k == 0:
            x_r, x_t, x_z = (ax.point_rule[0][:, 0] for ax in axes)
            r, t, z = jnp.meshgrid(x_r, x_t, x_z, indexing='ij')
            pts = jnp.stack([r.ravel(), t.ravel(), z.ravel()], axis=-1)
            m = jax.lax.map(lambda xi: _as_single_component(f(xi)), pts,
                            batch_size=mrx.MAP_BATCH_SIZE_INNER).reshape(len(x_r), len(x_t), len(x_z))
        else:
            rules = tuple(ax.span_rule if h else ax.point_rule for ax, h in zip(axes, hist))
            m = _greville_moments(lambda x, c=c: logical(x)[c], rules)
        for j, ax in enumerate(axes):
            m = _solve_tensor_collocation_axis(ax.hist if hist[j] else ax.coll, m, axis=j)
        coeffs.append(m.reshape(-1))
    return _restrict(seq, jnp.concatenate(coeffs), k)


def _restrict(seq, raw, k):
    """The ``k``-form DoFs of ``seq`` from the tensor-product coefficients ``raw``: the coefficient fit
    ``(E E^T)^-1 E`` on the axis rows, the raw coefficients on the other rings with ``n_theta`` cells and
    :func:`ring_projections` on the graded rings. On a parity view it is ``X^T`` of the full space's."""
    key = (int(k), seq.dirichlet)
    base = seq if seq.parity is None else seq._base
    v = conforming_restriction(base.extraction[key], raw, base.core[key])
    for rows, cols, matrix in base.ring_projections[key]:
        v = v.at[rows].set(jnp.einsum('ij,gjl->gil', matrix, raw[cols]))
    return v if seq.parity is None else seq.reduction[key].T @ v
