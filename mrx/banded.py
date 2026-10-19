"""Banded 1-D basis tables: the sum-factorised contractions without the zeros of the dense tables.

A B-spline of degree ``p`` is nonzero on ``p + 1`` knot spans only, so the dense ``(N, Q)`` table of a basis at the
quadrature points is zero outside a band: on knot span ``e`` only the functions ``e, ..., e + w - 1`` (modulo ``N``
on a periodic axis) can be nonzero, with ``w = p + 1`` (``p`` for the derivative basis). :class:`Band` stores
those values, shape ``(n_comp, n_spans, w, q)`` for ``q`` quadrature points per span, and :func:`forward` and
:func:`backward` contract an axis of a coefficient or quadrature grid with it. Each is a sum of ``w`` products
with statically shifted slices of the grid, so no index array is gathered or scattered, and the work is
``w / N`` of the dense contraction.

A band is built from the dense table and kept only if it reproduces that table exactly
(:meth:`Band.from_dense`), so the banded and the dense contraction add the same nonzero products, in a different
order. :func:`enabled` decides where a band exists whether the contractions use it: ``MRX_BANDED=1`` always,
``0`` never, and ``auto`` (the default) on a TPU only. On a CPU the dense tables are as fast or faster: the
contractions go to the matrix library and the weights dominate the apply. It is read when a function is traced,
so set it before the first call. Inside :func:`dense` the contractions are traced with the dense tables whatever the
mode.
"""

import contextlib
import os
from typing import Optional

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np

#: ``"1"``, ``"0"`` or ``"auto"``: whether the sum-factorised contractions use the banded tables (:func:`enabled`).
MODE = os.environ.get("MRX_BANDED", "auto")
if MODE not in ("0", "1", "auto"):
    raise ValueError(f"MRX_BANDED={MODE!r}, expected '0', '1' or 'auto'")


#: The depth of :func:`dense` blocks being traced: inside one the contractions use the dense tables.
_DENSE_DEPTH = 0


@contextlib.contextmanager
def dense():
    """Trace the contractions inside the block with the dense tables, whatever :data:`MODE` says.

    The batched probes of the preconditioner setup (:func:`mrx.metric_lumping._probe_columns`) use it. On a TPU v6e
    (JAX 0.11.2, libtpu 0.0.48.1) XLA aborts compiling the banded contraction batched by their ``lax.map`` at li383
    (32,44,20) ("Check failed ... Cannot find a valid window" in fusion_emitter.cc), while it compiles the unbatched
    contraction of the relaxation steps. The probes run once per sequence, so the dense tables cost little there."""
    global _DENSE_DEPTH
    _DENSE_DEPTH += 1
    try:
        yield
    finally:
        _DENSE_DEPTH -= 1


def enabled() -> bool:
    """Whether the sum-factorised contractions use the banded tables where they exist (:data:`MODE`), outside
    :func:`dense`."""
    return _DENSE_DEPTH == 0 and (MODE == "1" or (MODE == "auto" and jax.default_backend() == "tpu"))


class Band(eqx.Module):
    """The nonzero band of the 1-D tables of the components of a form along one axis.

    ``values[c, e, i, j]`` is function ``(e + i) mod n`` of component ``c`` at quadrature point ``j`` of knot
    span ``e``, so the dense table is ``T[c, (e + i) mod n, e q + j] = values[c, e, i, j]``."""

    values: jnp.ndarray                     # (n_comp, n_spans, w, q)
    n: int = eqx.field(static=True)         # the number of coefficients along the axis

    @property
    def n_q(self):
        """The number of quadrature points along the axis."""
        return self.values.shape[1] * self.values.shape[3]

    @classmethod
    def from_dense(cls, table, n_spans) -> Optional["Band"]:
        """The band of the dense ``(n_comp, N, Q)`` table on ``n_spans`` knot spans of ``Q / n_spans`` points,
        or ``None`` if the band does not reproduce the table exactly (repeated knots), does not save work, or the
        table is traced."""
        try:
            t = np.asarray(table)
        except Exception:       # a traced table: keep the dense contraction
            return None
        c, n, n_q = t.shape
        if n_spans <= 0 or n_q % n_spans:
            return None
        blocks = t.reshape(c, n, n_spans, n_q // n_spans)
        _, rows, spans = np.nonzero(np.any(blocks != 0, axis=3))
        w = 1 + int(np.max((rows - spans) % n)) if rows.size else 1
        if w >= n:
            return None
        span = np.arange(n_spans)[:, None]
        row = (span + np.arange(w)[None, :]) % n                        # (n_spans, w)
        values = blocks[:, row, span, :]                                # (c, n_spans, w, q)
        rebuilt = np.zeros_like(blocks)
        rebuilt[:, row, span, :] = values
        if not np.array_equal(rebuilt, blocks):
            return None
        return cls(jnp.asarray(values), n)


def _band_term(band, i, axis, ndim):
    """``values[:, :, i, :]`` shaped to broadcast against a grid split as ``(..., n_spans, q, ...)`` at ``axis``."""
    c, n_spans, _, q = band.values.shape
    return band.values[:, :, i, :].reshape((c,) + (1,) * (axis - 1) + (n_spans, q) + (1,) * (ndim - 1 - axis))


def forward(x, band, axis):
    """``y[..., e q + j, ...] = sum_i values[c, e, i, j] x[c, ..., (e + i) mod n, ...]``: the coefficients along
    ``axis`` (>= 1) of the component-batched grid ``x`` to the quadrature points, the dense
    ``einsum('c..i..,cia->c..a..')``."""
    _, n_spans, w, q = band.values.shape
    n = x.shape[axis]
    length = n_spans + w - 1
    if length > n:          # periodic wrap: append the first coefficients
        x = jnp.concatenate([x, lax.slice_in_dim(x, 0, length - n, axis=axis)], axis=axis)
    y = 0
    for i in range(w):
        shifted = jnp.expand_dims(lax.slice_in_dim(x, i, i + n_spans, axis=axis), axis + 1)
        y = y + shifted * _band_term(band, i, axis, x.ndim)
    return y.reshape(y.shape[:axis] + (n_spans * q,) + y.shape[axis + 2:])


def backward(v, band, axis):
    """The transpose of :func:`forward`: the quadrature points along ``axis`` of ``v`` back to the ``band.n``
    coefficients, ``y[c, ..., m, ...] = sum_{e, i: (e + i) mod n = m} sum_j values[c, e, i, j] v[c, ..., e q + j,
    ...]``, the dense ``einsum('c..a..,cia->c..i..')``."""
    _, n_spans, w, q = band.values.shape
    n = band.n
    v = v.reshape(v.shape[:axis] + (n_spans, q) + v.shape[axis + 1:])
    y = 0
    for i in range(w):
        g = jnp.sum(v * _band_term(band, i, axis, v.ndim - 1), axis=axis + 1)       # span e -> function e + i
        pads = [(0, 0)] * g.ndim
        pads[axis] = (i, w - 1 - i)
        y = y + jnp.pad(g, pads)
    length = n_spans + w - 1
    if length > n:          # periodic wrap: fold the tail onto the first coefficients
        tail = lax.slice_in_dim(y, n, length, axis=axis)
        pads = [(0, 0)] * y.ndim
        pads[axis] = (0, 2 * n - length)
        return lax.slice_in_dim(y, 0, n, axis=axis) + jnp.pad(tail, pads)
    if length < n:
        pads = [(0, 0)] * y.ndim
        pads[axis] = (0, n - length)
        y = jnp.pad(y, pads)
    return y
