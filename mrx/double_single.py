"""Double-single arithmetic: a number as an unevaluated sum ``hi + lo`` of two float32, about 48 bits.

A device without float64 computes the residuals of the solves with it (:mod:`mrx.precision`). Every operation
here is float32. The building blocks are the error-free transformations :func:`two_sum` and :func:`two_prod`
(Dekker's, since XLA has no fused multiply-add), which return a float32 result together with its exact rounding
error.

:class:`DS` is the array type. It supports the arithmetic, indexing and reshaping that the operator kernels
need, and the functions :func:`concatenate`, :func:`pad`, :func:`roll` and :func:`stack` take float arrays and
:class:`DS` alike, so a kernel written with them runs in either. :class:`Rows` is a sparse matrix in a form that
can be applied without a scatter, and :func:`contract` and :func:`dot` are the two reductions.

Host data (NumPy float64) enters through :func:`from_host`, which splits each number into its two parts exactly.
"""

from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

_F32 = jnp.float32
#: Veltkamp's constant ``2^12 + 1`` for the 24-bit mantissa of float32.
_SPLITTER = 4097.0


def two_sum(a, b):
    """``(s, e)`` with ``s = fl(a + b)`` and ``a + b = s + e`` exactly."""
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def fast_two_sum(a, b):
    """:func:`two_sum` for ``|a| >= |b|``, with three operations."""
    s = a + b
    return s, b - (s - a)


def _split(a):
    c = _F32(_SPLITTER) * a
    hi = c - (c - a)
    return hi, a - hi


def two_prod(a, b):
    """``(p, e)`` with ``p = fl(a b)`` and ``a b = p + e`` exactly."""
    p = a * b
    ah, al = _split(a)
    bh, bl = _split(b)
    return p, ((ah * bh - p) + ah * bl + al * bh) + al * bl


def _parts(x):
    """``(hi, lo)`` of a :class:`DS`, a float32 array or a Python number. ``lo`` is ``None`` when it is zero."""
    if isinstance(x, DS):
        return x.hi, x.lo
    return jnp.asarray(x, dtype=_F32), None


class DS(eqx.Module):
    """A double-single array, the unevaluated sum ``hi + lo`` with ``|lo|`` at most half a unit in the last
    place of ``hi``."""

    hi: jnp.ndarray
    lo: jnp.ndarray

    @property
    def shape(self):
        return self.hi.shape

    @property
    def ndim(self):
        return self.hi.ndim

    @property
    def size(self):
        return self.hi.size

    def __getitem__(self, index):
        return DS(self.hi[index], self.lo[index])

    def reshape(self, *shape):
        return DS(self.hi.reshape(*shape), self.lo.reshape(*shape))

    def ravel(self):
        return DS(self.hi.ravel(), self.lo.ravel())

    def __neg__(self):
        return DS(-self.hi, -self.lo)

    def __add__(self, other):
        bh, bl = _parts(other)
        s, e = two_sum(self.hi, bh)
        return DS(*fast_two_sum(s, e + (self.lo if bl is None else self.lo + bl)))

    __radd__ = __add__

    def __sub__(self, other):
        bh, bl = _parts(other)
        s, e = two_sum(self.hi, -bh)
        return DS(*fast_two_sum(s, e + (self.lo if bl is None else self.lo - bl)))

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        bh, bl = _parts(other)
        p, e = two_prod(self.hi, bh)
        e = e + self.lo * bh
        return DS(*fast_two_sum(p, e if bl is None else e + self.hi * bl))

    __rmul__ = __mul__

    def round(self):
        """The float32 array nearest to the value."""
        return self.hi + self.lo


def from_host(x):
    """The :class:`DS` of a NumPy float64 array, each number split exactly into its float32 parts."""
    x = np.asarray(x, dtype=np.float64)
    hi = x.astype(np.float32)
    return DS(jnp.asarray(hi), jnp.asarray((x - hi).astype(np.float32)))


def to_host(x):
    """The NumPy float64 array ``hi + lo`` of a :class:`DS`."""
    return np.asarray(x.hi, dtype=np.float64) + np.asarray(x.lo, dtype=np.float64)


def lift(x):
    """``x`` as a :class:`DS`. A float32 array gets a zero second part."""
    if isinstance(x, DS):
        return x
    x = jnp.asarray(x, dtype=_F32)
    return DS(x, jnp.zeros_like(x))


def zeros(shape):
    return DS(jnp.zeros(shape, dtype=_F32), jnp.zeros(shape, dtype=_F32))


def zeros_like(x):
    """Zeros of the shape and kind of ``x``, a float array or a :class:`DS`."""
    return zeros(x.shape) if isinstance(x, DS) else jnp.zeros_like(x)


def _both(fn, xs):
    """``fn`` applied to the first and to the second parts of a list of arrays, as a :class:`DS`."""
    xs = [lift(x) for x in xs]
    return DS(fn([x.hi for x in xs]), fn([x.lo for x in xs]))


def concatenate(xs, axis=0):
    """``jnp.concatenate`` for float arrays and :class:`DS`."""
    if not any(isinstance(x, DS) for x in xs):
        return jnp.concatenate(xs, axis=axis)
    return _both(lambda parts: jnp.concatenate(parts, axis=axis), xs)


def stack(xs, axis=0):
    """``jnp.stack`` for float arrays and :class:`DS`."""
    if not any(isinstance(x, DS) for x in xs):
        return jnp.stack(xs, axis=axis)
    return _both(lambda parts: jnp.stack(parts, axis=axis), xs)


def pad(x, pad_width):
    """``jnp.pad`` with zeros for a float array or a :class:`DS`."""
    if not isinstance(x, DS):
        return jnp.pad(x, pad_width)
    return DS(jnp.pad(x.hi, pad_width), jnp.pad(x.lo, pad_width))


def roll(x, shift, axis):
    """``jnp.roll`` for a float array or a :class:`DS`."""
    if not isinstance(x, DS):
        return jnp.roll(x, shift, axis=axis)
    return DS(jnp.roll(x.hi, shift, axis=axis), jnp.roll(x.lo, shift, axis=axis))


def moveaxis(x, source, destination):
    """``jnp.moveaxis`` for a float array or a :class:`DS`."""
    if not isinstance(x, DS):
        return jnp.moveaxis(x, source, destination)
    return DS(jnp.moveaxis(x.hi, source, destination), jnp.moveaxis(x.lo, source, destination))


def _sum_last(p, e):
    """The :class:`DS` sum over the last axis of the float32 terms ``p``, with ``e`` their second parts.

    The terms are added in pairs, each addition by :func:`two_sum`, so the first parts are summed without error
    and only the small second parts are added in float32."""
    while p.shape[-1] > 1:
        if p.shape[-1] % 2:
            p = jnp.concatenate([p, jnp.zeros_like(p[..., :1])], axis=-1)
            e = jnp.concatenate([e, jnp.zeros_like(e[..., :1])], axis=-1)
        s, t = two_sum(p[..., 0::2], p[..., 1::2])
        p, e = s, e[..., 0::2] + e[..., 1::2] + t
    return DS(*fast_two_sum(p[..., 0], e[..., 0]))


def _products(a, b):
    """The float32 products ``p`` of ``a`` and ``b`` (each a :class:`DS` or float32) with their second parts."""
    ah, al = _parts(a)
    bh, bl = _parts(b)
    p, e = two_prod(ah, bh)
    if al is not None:
        e = e + al * bh
    if bl is not None:
        e = e + ah * bl
    return p, e


def contract(a, b):
    """``sum_j a[..., j] b[..., j]`` over the last axis, as a :class:`DS`. ``a`` and ``b`` broadcast."""
    return _sum_last(*_products(a, b))


def dot(a, b):
    """The dot product of two vectors as a :class:`DS`."""
    a, b = (x.ravel() for x in (lift(a), lift(b)))
    return contract(a, b)


def rsqrt(x):
    """``1 / sqrt(x)`` of a :class:`DS` scalar, by one Newton step on the float32 value."""
    y = 1.0 / jnp.sqrt(x.hi)
    # y (1 + (1 - x y^2) / 2), where 1 - x y^2 is of the size of the float32 round-off
    return lift(y) + y * (0.5 * (1.0 - x * y * y).round())


def matvec(A, x):
    """``A x`` for a dense matrix and a vector. With a :class:`DS` among them the result is a :class:`DS`."""
    if not isinstance(A, DS) and not isinstance(x, DS):
        return A @ x
    return contract(A, lift(x)[None, :])


def set_at(x, index, values):
    """``x.at[index].set(values)`` for a float array or a :class:`DS`."""
    if not isinstance(x, DS):
        return x.at[index].set(values)
    values = lift(values)
    return DS(x.hi.at[index].set(values.hi), x.lo.at[index].set(values.lo))


class Rows(eqx.Module):
    """A sparse matrix stored by rows for a :class:`DS` apply.

    A scatter adds its terms in an order the device chooses, which no error-free sum allows. So each row
    gathers its own entries instead. The rows are sorted into groups of equal padded width, every group is one
    gather of shape ``(rows, width)`` and one :func:`contract`, and ``order`` puts the results back.
    """

    idx: tuple            # per group: the column indices, shape (rows, width)
    val: tuple            # per group: the entries as a DS of that shape, zero on the padding
    order: jnp.ndarray    # the position of every row in the concatenated groups

    @classmethod
    def from_coo(cls, rows, cols, vals, n_rows):
        """The matrix with the given entries (NumPy, float64 values, duplicates summed)."""
        rows, cols = np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)
        vals = np.asarray(vals, dtype=np.float64)
        count = np.bincount(rows, minlength=n_rows)
        by_row = np.argsort(rows, kind="stable")
        start = np.concatenate([[0], np.cumsum(count)])
        rank = np.arange(rows.size) - start[rows[by_row]]
        # widths 1, 2, 4, ...: the padding at most doubles the entries
        width = np.maximum(1, 2 ** np.ceil(np.log2(np.maximum(count, 1))).astype(np.int64))
        idx, val, members = [], [], []
        for w in np.unique(width):
            group = np.flatnonzero(width == w)
            position = np.full(n_rows, -1)
            position[group] = np.arange(group.size)
            keep = position[rows[by_row]] >= 0
            i, v = np.zeros((group.size, w), dtype=np.int32), np.zeros((group.size, w))
            i[position[rows[by_row][keep]], rank[keep]] = cols[by_row][keep]
            v[position[rows[by_row][keep]], rank[keep]] = vals[by_row][keep]
            idx.append(jnp.asarray(i))
            val.append(from_host(v))
            members.append(group)
        order = np.empty(n_rows, dtype=np.int32)
        order[np.concatenate(members)] = np.arange(n_rows)
        return cls(tuple(idx), tuple(val), jnp.asarray(order))

    def __matmul__(self, x):
        x = lift(x)
        return concatenate([contract(v, x[i]) for i, v in zip(self.idx, self.val)])[self.order]


class Tables(eqx.Module):
    """The 1-D basis tables of one axis of a sum-factorised apply, per component, in the two gather forms:
    from the coefficients to the quadrature points and back."""

    to_idx: jnp.ndarray     # (n_comp, Q, width): the coefficients seen from a quadrature point
    to_val: jnp.ndarray     # their float32 basis values
    from_idx: jnp.ndarray   # (n_comp, N, width): the quadrature points seen from a coefficient
    from_val: jnp.ndarray

    @classmethod
    def from_dense(cls, table):
        """From the dense ``(n_comp, N, Q)`` table of the float32 apply."""
        table = np.asarray(table)

        def gather(t):       # (n_comp, rows, cols) -> the nonzero columns of every row
            width = max(1, int((t != 0).sum(axis=2).max()))
            idx = np.zeros(t.shape[:2] + (width,), dtype=np.int32)
            val = np.zeros(t.shape[:2] + (width,), dtype=np.float32)
            for c, r in np.ndindex(*t.shape[:2]):
                (cols,) = np.nonzero(t[c, r])
                idx[c, r, :cols.size], val[c, r, :cols.size] = cols, t[c, r, cols]
            return jnp.asarray(idx), jnp.asarray(val)
        return cls(*gather(table.transpose(0, 2, 1)), *gather(table))


def along(x, idx, val, axis):
    """Contract axis ``axis`` (1, 2 or 3) of the component-batched :class:`DS` ``x`` with a gather-form table:
    ``y[c, ..., q, ...] = sum_j val[c, q, j] x[c, ..., idx[c, q, j], ...]``."""
    x = moveaxis(x, axis, 1)

    def one(xc_hi, xc_lo, i, v):
        g = DS(jnp.moveaxis(xc_hi[i], 1, -1), jnp.moveaxis(xc_lo[i], 1, -1))      # (rows, ..., width)
        y = contract(v.reshape(v.shape[:1] + (1,) * (g.ndim - 2) + v.shape[1:]), g)
        return y.hi, y.lo
    hi, lo = jax.vmap(one)(x.hi, x.lo, idx, val)
    return moveaxis(DS(hi, lo), 1, axis)


def sumfact(tables_r, tables_c, shapes_r, shapes_c, weights, x):
    """The sum-factorised apply of :func:`mrx.mass.sumfact_apply` for a :class:`DS` vector ``x``, with the same
    float32 tables and weights. ``tables_r`` and ``tables_c`` are the three :class:`Tables` of the row and the
    column form."""
    n_max = tuple(int(t.from_idx.shape[1]) for t in tables_c)
    parts, offset = [], 0
    for s in shapes_c:
        n = s[0] * s[1] * s[2]
        parts.append(pad(x[offset:offset + n].reshape(s), [(0, n_max[a] - s[a]) for a in range(3)]))
        offset += n
    u = stack(parts)                                             # (c, i, j, k)
    for a, t in enumerate(tables_c):
        u = along(u, t.to_idx, t.to_val, a + 1)                  # (c, Q_r, Q_t, Q_z)
    # v[r] = sum_c W[..., r, c] u[c] at every quadrature point
    v = contract(jnp.moveaxis(weights, 3, 0), moveaxis(u, 0, -1)[None])
    for a, t in enumerate(tables_r):
        v = along(v, t.from_idx, t.from_val, a + 1)
    return concatenate([v[c, :s[0], :s[1], :s[2]].reshape(-1) for c, s in enumerate(shapes_r)])


def host_matrix(A) -> Optional[DS]:
    """A dense host matrix as a :class:`DS`, or ``None`` for an empty one."""
    A = np.asarray(A, dtype=np.float64)
    return None if A.size == 0 else from_host(A)
