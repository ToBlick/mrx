"""Incidence matrices: the exterior derivative (grad, curl, div) acting on spline coefficients.

In a spline de Rham sequence the derivative of a k-form spline lies in the next spline space, and
its coefficients are differences of neighbouring coefficients (entries -1, 0, +1). These matrices
depend only on the mesh, not on the geometry, and the sequence builds them once when it is created.

- :func:`build_matrixfree_incidence` returns ``G_k`` (``k = 0, 1, 2``: grad, curl, div) on the raw
  tensor-product coefficients. Its ``.T`` is the transpose.
- :func:`build_polar_derivative` returns grad, curl and div on the extracted DoFs of
  :mod:`mrx.extraction_operators`, axis functions included, on the uniform and on the graded spaces. On these
  spaces ``curl grad = 0`` and ``div curl = 0`` hold exactly.
"""
import equinox as eqx
import numpy as np
import scipy.sparse as sp

import mrx.double_single as ds
from mrx.extraction_operators import gram_inverse, summed_coo


def _diff_fwd(V, axis: int, typ: str):
    """1-D incidence along ``axis``: ``c_{j+1} - c_j``, wrapped on a periodic axis."""
    if typ == 'clamped':
        upper, lower = [slice(None)] * V.ndim, [slice(None)] * V.ndim
        upper[axis], lower[axis] = slice(1, None), slice(None, -1)
        return V[tuple(upper)] - V[tuple(lower)]
    return ds.roll(V, -1, axis=axis) - V


def _diff_adj(Y, axis: int, typ: str):
    """Transpose of :func:`_diff_fwd` along ``axis``."""
    if typ == 'clamped':
        pad_end = [(0, 0)] * Y.ndim
        pad_end[axis] = (0, 1)
        pad_start = [(0, 0)] * Y.ndim
        pad_start[axis] = (1, 0)
        return ds.pad(-Y, pad_end) + ds.pad(Y, pad_start)
    return ds.roll(Y, 1, axis=axis) - Y


def _prod3(shape) -> int:
    return int(shape[0] * shape[1] * shape[2])


def _split3(x, shapes):
    """Split a flat vector into three 3-D component arrays of ``shapes``."""
    n0 = _prod3(shapes[0])
    n1 = _prod3(shapes[1])
    a = x[:n0].reshape(shapes[0])
    b = x[n0:n0 + n1].reshape(shapes[1])
    c = x[n0 + n1:].reshape(shapes[2])
    return a, b, c


def _apply_incidence_mf(op, x):
    """Apply a :class:`_MatrixFreeIncidence` operator to the flat vector ``x``."""
    tr, tt, tz = op.types
    if op.k == 0 and not op.transpose:
        # grad: 0-form -> (d_r, d_t, d_z)
        V = x.reshape(op.s0)
        return ds.concatenate([
            _diff_fwd(V, 0, tr).ravel(),
            _diff_fwd(V, 1, tt).ravel(),
            _diff_fwd(V, 2, tz).ravel(),
        ])
    if op.k == 0:
        a, b, c = _split3(x, op.s1)
        return (_diff_adj(a, 0, tr) + _diff_adj(b, 1, tt) + _diff_adj(c, 2, tz)).ravel()
    if op.k == 1 and not op.transpose:
        # curl: (a, b, c) -> (P, Q, R)
        a, b, c = _split3(x, op.s1)
        P = -_diff_fwd(b, 2, tz) + _diff_fwd(c, 1, tt)
        Q = _diff_fwd(a, 2, tz) - _diff_fwd(c, 0, tr)
        R = -_diff_fwd(a, 1, tt) + _diff_fwd(b, 0, tr)
        return ds.concatenate([P.ravel(), Q.ravel(), R.ravel()])
    if op.k == 1:
        P, Q, R = _split3(x, op.s2)
        a = _diff_adj(Q, 2, tz) - _diff_adj(R, 1, tt)
        b = -_diff_adj(P, 2, tz) + _diff_adj(R, 0, tr)
        c = _diff_adj(P, 1, tt) - _diff_adj(Q, 0, tr)
        return ds.concatenate([a.ravel(), b.ravel(), c.ravel()])
    if not op.transpose:
        # div: (a, b, c) -> d_r a + d_t b + d_z c
        a, b, c = _split3(x, op.s2)
        return (_diff_fwd(a, 0, tr) + _diff_fwd(b, 1, tt) + _diff_fwd(c, 2, tz)).ravel()
    Y = x.reshape(op.s3)
    return ds.concatenate([
        _diff_adj(Y, 0, tr).ravel(),
        _diff_adj(Y, 1, tt).ravel(),
        _diff_adj(Y, 2, tz).ravel(),
    ])


class _MatrixFreeIncidence(eqx.Module):
    """The incidence matrix ``G_k`` or its transpose, applied to a flat raw coefficient vector
    with ``@``. It holds only shapes, no arrays."""
    k: int = eqx.field(static=True)
    transpose: bool = eqx.field(static=True)
    types: tuple = eqx.field(static=True)
    s0: tuple = eqx.field(static=True)
    s1: tuple = eqx.field(static=True)
    s2: tuple = eqx.field(static=True)
    s3: tuple = eqx.field(static=True)

    def __matmul__(self, x):
        return _apply_incidence_mf(self, x)

    @property
    def T(self):
        return _MatrixFreeIncidence(self.k, not self.transpose, self.types, self.s0, self.s1, self.s2, self.s3)


def build_matrixfree_incidence(seq, k: int):
    """``G_k`` on the raw tensor-product coefficients of ``seq``, for ``k = 0, 1, 2``."""
    s0 = tuple(int(v) for v in seq.basis_0.shape[0])
    s3 = tuple(int(v) for v in seq.basis_3.shape[0])
    s1 = tuple(tuple(int(v) for v in comp) for comp in seq.basis_1.shape)
    s2 = tuple(tuple(int(v) for v in comp) for comp in seq.basis_2.shape)
    return _MatrixFreeIncidence(k, False, tuple(seq.basis_0.types), s0, s1, s2, s3)


def diff_1d(n, typ):
    """The 1-D incidence ``c_{j+1} - c_j`` on an axis of ``n`` coefficients as a sparse matrix, wrapped if
    periodic."""
    rows = np.arange(n - 1 if typ == 'clamped' else n)
    return sp.csr_matrix((np.concatenate([-np.ones(rows.size), np.ones(rows.size)]),
                          (np.concatenate([rows, rows]), np.concatenate([rows, (rows + 1) % n]))),
                         shape=(rows.size, n))


def raw_incidence_matrix(seq, k):
    """Grad (k = 0), curl (k = 1) or div (k = 2) on the raw tensor-product coefficients of ``seq`` as a sparse
    matrix, the map that :func:`build_matrixfree_incidence` applies."""
    D = [diff_1d(int(n), t) for n, t in zip(seq.basis_0.shape[0], seq.basis_0.types)]
    I_p = [sp.identity(int(n)) for n in seq.basis_0.shape[0]]       # the splines of each axis
    I_d = [sp.identity(d.shape[0]) for d in D]                       # the derivative splines

    def kron3(a, b, c):
        return sp.kron(sp.kron(a, b), c)
    if k == 0:
        return sp.vstack([kron3(D[0], I_p[1], I_p[2]), kron3(I_p[0], D[1], I_p[2]),
                          kron3(I_p[0], I_p[1], D[2])]).tocsr()
    if k == 1:
        # the curl of (a, b, c) is (-d_z b + d_t c, d_z a - d_r c, -d_t a + d_r b)
        return sp.bmat([[None, -kron3(I_p[0], I_d[1], D[2]), kron3(I_p[0], D[1], I_d[2])],
                        [kron3(I_d[0], I_p[1], D[2]), None, -kron3(D[0], I_p[1], I_d[2])],
                        [-kron3(I_d[0], D[1], I_p[2]), kron3(D[0], I_d[1], I_p[2]), None]]).tocsr()
    # the div of (a, b, c) is d_r a + d_t b + d_z c
    return sp.hstack([kron3(D[0], I_d[1], I_d[2]), kron3(I_d[0], D[1], I_d[2]),
                      kron3(I_d[0], I_d[1], D[2])]).tocsr()


def build_polar_derivative(seq, k, dirichlet: bool):
    """Grad (k = 0), curl (k = 1) or div (k = 2) on the extracted DoFs of the free or the Dirichlet spaces, as a
    sparse matrix.

    It is ``(E_{k+1} E_{k+1}^T)^{-1} E_{k+1} G_k E_k^T`` with the raw incidence ``G_k`` and the extractions
    ``E_k``: the exact derivative of an extracted k-form, written in the extracted (k+1)-form basis.
    ``E E^T`` is the identity except on the polar core rows and, on a graded space, on the rings with fewer
    theta cells than the wall (:func:`~mrx.extraction_operators.gram_inverse`). Entries at round-off are
    dropped.
    """
    def csr(e):
        r, c, v = e.entries()
        return sp.csr_matrix((v, (r, c)), shape=e.forward_shape)

    key = (k + 1, dirichlet)
    e_in, e_out = seq.extraction[(k, dirichlet)], seq.extraction[key]
    P = csr(e_out) @ raw_incidence_matrix(seq, k) @ csr(e_in).T
    P = (gram_inverse(e_out, seq.core[key], seq._gram[key]) @ P).tocoo()
    keep = np.abs(P.data) > 1e3 * np.finfo(np.float64).eps * np.abs(P.data).max()
    return summed_coo(P.row[keep], P.col[keep], P.data[keep], P.shape)
