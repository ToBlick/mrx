"""One-dimensional B-spline bases, the building blocks of every discrete form in MRX.

- :class:`SplineBasis` is a clamped or periodic B-spline basis of degree ``p`` on ``[0, 1]``.
- :class:`DerivativeSpline` is the matching basis of degree ``p - 1`` that spans the derivatives
  of a :class:`SplineBasis`, with each function scaled to unit integral.
- :class:`TensorBasis` combines three 1-D bases into a basis on the logical cube.
- :func:`basis_table` and :func:`basis_derivative_table` return the values or the derivatives of
  all functions of a basis at a set of points.
"""
import functools
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np


def _nonzero_bsplines(T, p, x):
    """Return the values at ``x`` of the ``p + 1`` B-splines that can be nonzero there, and the
    index of the first of them."""
    # At the right end of a clamped knot vector the value and its derivatives are the left limits.
    s = jnp.clip(jnp.searchsorted(T, x, side='right') - 1, p, T.shape[0] - p - 2)
    N = [jnp.ones_like(x)]
    if p == 0:
        return jnp.stack(N), s
    # knots T[s-p+1 .. s+p]: t[m] = T[s - p + 1 + m]
    t = jax.lax.dynamic_slice(T, (s - p + 1,), (2 * p,))
    for j in range(1, p + 1):
        left = [x - t[p - 1 - m] for m in range(j)]    # x - T[s - m]
        right = [t[p + m] - x for m in range(j)]       # T[s + m + 1] - x
        saved = 0.0
        new = []
        for r in range(j):
            temp = N[r] / (right[r] + left[j - 1 - r])
            new.append(saved + right[r] * temp)
            saved = left[j - 1 - r] * temp
        new.append(saved)
        N = new
    return jnp.stack(N), s - p


def contract_local(coefficients, local):
    """Return ``sum_ijk c[..., i, j, k] B_i B_j B_k`` from the per-axis ``(values, indices)`` of
    ``evaluate_local``. Leading axes of ``coefficients`` are kept."""
    (vr, ir), (vt, it), (vz, iz) = local
    window = coefficients[..., ir[:, None, None], it[None, :, None], iz[None, None, :]]
    return jnp.einsum('i,j,k,...ijk->...', vr, vt, vz, window)


# The jitted tables compile once per (kind, n, p, type). New knot values reuse the executable.

def basis_key(basis):
    """Return the hashable description ``(kind, n, p, type)`` of a :class:`SplineBasis` or
    :class:`DerivativeSpline`, together with its knots."""
    if isinstance(basis, DerivativeSpline):
        par = basis.parent
        return ("d", par.n, par.p, par.type), par.T
    return ("s", basis.n, basis.p, basis.type), basis.T


def rebuild_basis(key, T):
    """Rebuild the basis described by :func:`basis_key` from its description and its knots,
    which may be traced."""
    kind, n, p, typ = key
    base = SplineBasis(n, p, typ, T=T)
    return DerivativeSpline(base) if kind == "d" else base


@functools.partial(jax.jit, static_argnames=("key", "periodic"))
def _histopolation(T, spans, xi_ref, w_ref, knots, *, key, periodic):
    """Return the integral of every basis function over every span, shape ``(n_spans, n)``."""
    basis = rebuild_basis(key, T)

    def integrate_span(span):
        a, b = span
        cuts = jnp.clip(knots, a, b)
        cuts = jnp.sort(jnp.concatenate([jnp.array([a]), cuts, jnp.array([b])]))
        lo, hi = cuts[:-1], cuts[1:]
        centers = 0.5 * (lo + hi)
        halfwidths = 0.5 * (hi - lo)
        xs = centers[:, None] + halfwidths[:, None] * xi_ref[None, :]
        if periodic:
            xs = jnp.mod(xs, 1.0)
        values = _table(basis, xs.reshape(-1)).reshape((basis.n,) + xs.shape)
        return jnp.einsum('s,q,isq->i', halfwidths, w_ref, values)

    return jax.vmap(integrate_span)(spans)


class SplineBasis:
    """A basis of ``n`` B-splines of degree ``p`` on ``[0, 1]``, clamped or periodic.

    ``basis(x, i)`` is the value of the ``i``-th spline at ``x``. A clamped basis interpolates at
    both ends, a periodic one has period 1.

    Attributes:
        n, p, type: the number of splines, the degree, and ``'clamped'`` or ``'periodic'``.
        ns: ``arange(n)``.
        T: the knot vector, uniform unless given.
    """

    def __init__(self, n: int, p: int, type: str, T: Optional[jnp.ndarray] = None) -> None:
        if p >= n:
            raise ValueError(
                f"Degree {p} is greater than or equal to the number of splines {n}")
        if type not in ['clamped', 'periodic']:
            raise ValueError(f"Invalid spline type: {type}")
        self.n = n
        self.ns = jnp.arange(self.n)
        self.p = p
        self.type = type
        self.T = self._init_knots() if T is None else T

    def _init_knots(self) -> jnp.ndarray:
        n, p = self.n, self.p
        if self.type == 'periodic':
            _T = jnp.linspace(0, 1, n+1)
            return jnp.concatenate([_T[-(p+1):-1] - 1, _T, _T[1:(p+1)] + 1])
        return jnp.concatenate([jnp.zeros(p), jnp.linspace(0, 1, n-p+1), jnp.ones(p)])

    def __call__(self, x: float, i: int) -> jnp.ndarray:
        """Return the value of the ``i``-th spline at ``x``."""
        return _single(self, x, i)

    def evaluate_local(self, x: float) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return the values and indices of the ``p + 1`` splines that can be nonzero at ``x``.
        On a periodic basis ``x`` is taken modulo 1 and the indices modulo ``n``."""
        if self.type == 'periodic':
            x = jnp.mod(x, 1.0)
        values, first = _nonzero_bsplines(self.T, self.p, x)
        indices = first + jnp.arange(self.p + 1)
        if self.type == 'periodic':
            indices = indices % self.n
        return values, indices

    def greville_points(self) -> jnp.ndarray:
        """Return the Greville points. For spline ``i`` this is the average of the knots
        ``T[i + 1], ..., T[i + p]``, or the midpoint of its support at ``p = 0``. On a periodic
        basis the points are wrapped into ``[0, 1)``."""
        if self.p == 0:
            points = 0.5 * (self.T[:self.n] + self.T[1:self.n + 1])
        else:
            offsets = jnp.arange(1, self.p + 1)
            knot_ids = self.ns[:, None] + offsets[None, :]
            points = jnp.mean(self.T[knot_ids], axis=1)
        if self.type == 'periodic':
            points = jnp.mod(points, 1.0)
        return points

    def collocation_matrix(self, points: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Return the ``(len(points), n)`` matrix whose entry ``[k, i]`` is spline ``i`` at
        ``points[k]``. The points default to the Greville points."""
        if points is None:
            points = self.greville_points()
        return basis_table(self, jnp.asarray(points)).T


class TensorBasis:
    """The tensor-product basis ``B_i(x) B_j(y) B_k(z)`` of three 1-D ``bases``."""

    def __init__(self, bases: list) -> None:
        if len(bases) != 3:
            raise ValueError(
                f"TensorBasis requires exactly 3 bases, got {len(bases)}")
        self.bases = bases

    def evaluate_local(self, x: jnp.ndarray) -> tuple:
        """Return, per axis, the values and indices of the 1-D functions that can be nonzero at
        ``x``."""
        return tuple(b.evaluate_local(xi) for b, xi in zip(self.bases, x))

    def contract(self, coefficients: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Return ``sum_ijk c[..., i, j, k] B_i(x_0) B_j(x_1) B_k(x_2)`` at the point ``x``."""
        return contract_local(coefficients, self.evaluate_local(x))


class DerivativeSpline:
    """The basis that spans the derivatives of the spline basis ``s``.

    It has ``n - 1`` functions (clamped) or ``n`` (periodic) of degree ``p - 1`` on the knots
    ``T[1:-1]`` of ``s``. Each function is scaled to unit integral, so the coefficients of the
    derivative of a spline in ``s`` are the differences of its coefficients.

    Attributes:
        n, p, type, T, ns: as for :class:`SplineBasis`.
        parent: the basis ``s``.
        s: the same degree ``p - 1`` splines without the scaling.
    """

    def __init__(self, s: SplineBasis) -> None:
        self.n = s.n - 1 if s.type == 'clamped' else s.n
        self.p = s.p - 1
        self.type = s.type
        self.T = s.T[1:-1]
        self.parent = s
        self.s = SplineBasis(self.n, self.p, self.type, self.T)
        self.ns = jnp.arange(self.n)

    def _scale(self, i):
        """Return the unit-integral scale ``(p + 1) / (T[i + p + 1] - T[i])`` of function ``i``."""
        p = self.p
        return (p + 1) / (self.T[i + p + 1] - self.T[i])

    def __call__(self, x: float, i: int) -> jnp.ndarray:
        """Return the value of the ``i``-th function at ``x``."""
        return _single(self, x, i)

    def evaluate_local(self, x: float) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return the values and indices of the ``p + 1`` functions that can be nonzero at ``x``,
        as in :meth:`SplineBasis.evaluate_local`."""
        values, indices = self.s.evaluate_local(x)
        return values * self._scale(indices), indices

    def greville_spans(self) -> jnp.ndarray:
        """Return the ``(n, 2)`` intervals ``[a, b]`` between consecutive Greville points of the
        parent basis. On a periodic basis the last interval ends past 1."""
        points = self.parent.greville_points()
        if self.type == 'periodic':
            # The last span crosses x = 1, so users of it evaluate the periodic extension there.
            points = jnp.sort(points)
            next_points = jnp.roll(points, -1)
            next_points = next_points.at[-1].set(next_points[-1] + 1.0)
            return jnp.stack([points, next_points], axis=1)
        return jnp.stack([points[:-1], points[1:]], axis=1)

    def histopolation_matrix(self) -> jnp.ndarray:
        """Return the ``(n, n)`` matrix whose entry ``[k, i]`` is the integral of function ``i``
        over the span ``k`` of :meth:`greville_spans`."""
        spans = self.greville_spans()
        xi_ref, w_ref = np.polynomial.legendre.leggauss(max(2, self.p + 2))
        # A span can contain a knot, so it is integrated piece by piece between the knots.
        key, T = basis_key(self)
        return _histopolation(T, jnp.asarray(spans), jnp.asarray(xi_ref),
                              jnp.asarray(w_ref), jnp.unique(self.T),
                              key=key, periodic=self.type == 'periodic')


def _single(basis, x, i):
    """Return function ``i`` of ``basis`` at ``x``."""
    values, indices = basis.evaluate_local(x)
    return jnp.sum(jnp.where(indices == i, values, 0.0))


def _table(basis, x, derivative=False):
    """Return the ``(n, n_q)`` table of every function (or its derivative) at the points ``x``."""
    def local(xi):
        values, indices = basis.evaluate_local(xi)
        if derivative:
            values = jax.jacfwd(lambda y: basis.evaluate_local(y)[0])(xi)
        return values, indices
    values, indices = jax.vmap(local)(x)
    cols = jnp.broadcast_to(jnp.arange(x.shape[0])[:, None], indices.shape)
    return jnp.zeros((basis.n, x.shape[0]), values.dtype).at[indices, cols].add(values)


def basis_table(basis, x):
    """Return the ``(n, n_q)`` table of every function of ``basis`` at every point of ``x``."""
    key, T = basis_key(basis)
    return _basis_table(T, x, key=key)


@functools.partial(jax.jit, static_argnames=("key",))
def _basis_table(T, x, *, key):
    return _table(rebuild_basis(key, T), x)


def basis_derivative_table(basis, x):
    """Return the ``(n, n_q)`` table of the derivative of every function of ``basis`` at every
    point of ``x``."""
    key, T = basis_key(basis)
    return _basis_derivative_table(T, x, key=key)


@functools.partial(jax.jit, static_argnames=("key",))
def _basis_derivative_table(T, x, *, key):
    return _table(rebuild_basis(key, T), x, derivative=True)


def evaluate_basis_local(basis, x_q_flat, q_per_elem):
    """Return, for each knot span, the values of the ``p + 1`` functions of ``basis`` that are
    nonzero on it at its quadrature points, and the indices of those functions.

    ``x_q_flat`` holds ``q_per_elem`` points per span, span by span, starting at the first span
    of the axis. The values have shape ``(n_elem, q_per_elem, p + 1)`` and the indices
    ``(n_elem, p + 1)``.
    """
    n_elem = x_q_flat.shape[0] // q_per_elem
    elems = jnp.arange(n_elem)
    gdof = elems[:, None] + jnp.arange(basis.p + 1)[None, :]
    if basis.type == "periodic":
        gdof = gdof % basis.n
    points = elems[:, None] * q_per_elem + jnp.arange(q_per_elem)[None, :]
    return basis_table(basis, x_q_flat)[gdof[:, None, :], points[:, :, None]], gdof
