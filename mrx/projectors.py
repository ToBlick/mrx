"""Turning a given function into a discrete ``k``-form by commuting projections. Users call this through
``seq.interpolate``. (``seq.load`` gives the right-hand side of an L2 projection instead.)

:func:`interpolate` returns the coefficients of a form directly, by interpolation at the Greville points
(k = 0) or by matching integrals over Greville cells (histopolation, k = 1, 2, 3). On the tensor-product
spline spaces these projections commute with the exterior derivative. The result is then restricted to the
polar (axis-regular) subspace the sequence works in. The function is evaluated at logical points and returns
the logical components of the form.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array


import mrx
from mrx.extraction_operators import conforming_restriction
from mrx.spline_bases import basis_table

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


def _span_quadrature(basis, spans: Array) -> tuple[Array, Array]:
    """Gauss points and weights ``(xs, ws)``, ``(n_spans, n_pts)``, that integrate the splines exactly
    over every Greville span."""
    # split at the knots inside a span (at even p a span straddles a knot), padded to the widest span with
    # zero-width pieces
    xi_ref, w_ref = np.polynomial.legendre.leggauss(basis.p + 2)
    spans = np.asarray(spans)
    knots = np.unique(np.asarray(basis.T))
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


def interpolate(seq: "DeRhamSequence", f, k: int):
    """The coefficients of the ``k``-form that interpolates ``f`` (k = 0) or matches its integrals over the
    Greville cells (k = 1, 2, 3), in the ``k``-form space of ``seq`` (the Dirichlet space, or the free one
    on ``seq.free``). ``f`` returns the logical components of the form at a logical point (for k = 3 the
    density times ``det DPhi``)."""
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
    return conforming_restriction(seq.E(k), jnp.concatenate(coeffs), seq.core_rows(k))
