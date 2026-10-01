"""Turning a given function into a discrete ``k``-form. Users call these through ``seq.load`` and
``seq.interpolate``.

- :func:`load` returns the dual vector ``v_i = int Lambda^k_i . f dx`` of the basis forms
  ``Lambda^k_i`` against ``f``. Applying the inverse mass matrix to it gives the L2 projection of ``f``.
- :func:`interpolate` returns the coefficients of a form directly, by interpolation at the Greville
  points (k = 0) or by matching integrals over Greville cells (histopolation, k = 1, 2, 3). On the
  tensor-product spline spaces these projections commute with the exterior derivative. The result is
  then restricted to the polar (axis-regular) subspace the sequence works in.

The function ``f`` is evaluated at logical points. It returns physical components, or with
:func:`interpolate` also logical ones (``frame='logical'``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from mrx.precision import DTYPE

import mrx
from mrx.differential_forms import adj33
from mrx.extraction_operators import conforming_restriction
from mrx.geometry import map_jacobian_at
from mrx.quadrature import integrate_against

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
    # split at the knots inside a span (at even p a span straddles a knot), the same rule as
    # SplineBasis.histopolation_matrix, padded to the widest span with zero-width pieces
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
    return jnp.asarray(xs, dtype=DTYPE), jnp.asarray(ws, dtype=DTYPE)


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
        axes.append(_GrevilleAxis(
            coll=lam.collocation_matrix(pts),
            hist=d.histopolation_matrix(),
            point_rule=(pts[:, None], jnp.ones((pts.shape[0], 1), dtype=pts.dtype)),
            span_rule=_span_quadrature(d, d.greville_spans())))
    return tuple(axes)


def load(seq: "DeRhamSequence", f, k: int, parity=None):
    """The dual vector ``v_i = int Lambda^k_i . f dx`` of the ``k``-form space of ``seq`` (the
    Dirichlet space, or the free one on ``seq.free``). ``f`` takes a logical point and returns a scalar (k = 0, 3) or three
    physical components (k = 1, 2), which are pulled back here. ``parity`` is as in
    :meth:`~mrx.derham_sequence.DeRhamSequence.symmetrize`."""
    fn = f if k in (1, 2) else (lambda x: _as_single_component(f(x)))
    f_jk = jax.lax.map(fn, seq.quad.x, batch_size=mrx.MAP_BATCH_SIZE_INNER)
    if k in (1, 2):
        # DPhi is not stored on the geometry, so it is recomputed here once per call
        f_jk = jnp.einsum('qji,qj->qi', map_jacobian_at(seq.map, seq.quad.x), f_jk)    # DPhi^T v
        if k == 1:
            f_jk = jnp.einsum('qij,qj->qi', seq.metric_inv_jkl, f_jk)                    # DPhi^-1 v = G^-1 DPhi^T v
    if k in (0, 1):
        weight = seq.quad.w * seq.jacobian_j
    else:
        weight = seq.quad.w
    w_jk = f_jk * weight[:, None]
    return seq.E(k) @ seq.symmetrize(integrate_against(w_jk, seq._form_comp_info(k)), k, parity)


def _wrap_periodic_point(seq, xi):
    """The logical point ``xi`` with its periodic coordinates wrapped into ``[0, 1)``."""
    # periodic Greville spans run past x = 1 (at even p the last one is [1 - h/2, 1 + h/2])
    wrapped = []
    for axis, basis in enumerate(seq.basis_0.Lambda):
        coord = xi[axis]
        if basis.type == 'periodic':
            coord = jnp.mod(coord, 1.0)
        wrapped.append(coord)
    return jnp.asarray(wrapped, dtype=DTYPE)


def _pullback(seq, v, k, frame):
    """The function giving the logical components of the ``k``-form ``v``: ``DPhi^T v`` (k=1) or
    ``adj(DPhi) v = J DPhi^-1 v`` (k=2) or ``J v`` (k=3) when ``v`` returns physical components, ``v`` itself
    otherwise."""
    # adj(DPhi) rather than J DPhi^-1 stays finite where det DPhi -> 0 (the axis)
    if frame == 'logical':
        if k == 3:
            return lambda x: _as_single_component(v(_wrap_periodic_point(seq, x)))
        return lambda x: v(_wrap_periodic_point(seq, x))
    DPhi = jax.jacfwd(seq.map)
    if k == 3:
        def density(x):
            x_eval = _wrap_periodic_point(seq, x)
            return jnp.linalg.det(DPhi(x_eval)) * _as_single_component(v(x_eval))
        return density
    transform = (lambda D: D.T) if k == 1 else adj33

    def pullback(x):
        x_eval = _wrap_periodic_point(seq, x)
        return transform(DPhi(x_eval)) @ v(x_eval)

    return pullback


def _greville_moments(seq, fn, rules) -> Array:
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


def interpolate(seq: "DeRhamSequence", f, k: int, frame: str = 'physical'):
    """The coefficients of the ``k``-form that interpolates ``f`` (k = 0) or matches its integrals over the
    Greville cells (k = 1, 2, 3), in the ``k``-form space of ``seq`` (the Dirichlet space, or the free one
    on ``seq.free``).

    With ``frame='physical'`` (the default), ``f`` returns physical components (for k = 3 the physical
    density). With ``frame='logical'``, ``f`` returns the logical components of the form (for k = 3 the
    density times ``det DPhi``)."""
    if frame not in ('physical', 'logical'):
        raise ValueError(f"frame must be 'physical' or 'logical', got {frame!r}")
    axes = seq.greville
    pullback = _pullback(seq, f, k, frame) if k else None
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
            m = _greville_moments(seq, lambda x, c=c: pullback(x)[c], rules)
        for j, ax in enumerate(axes):
            m = _solve_tensor_collocation_axis(ax.hist if hist[j] else ax.coll, m, axis=j)
        coeffs.append(m.reshape(-1))
    return conforming_restriction(seq.E(k), jnp.concatenate(coeffs), seq.core_rows(k)).astype(DTYPE)
