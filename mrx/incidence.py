"""Incidence matrices: the exterior derivative (grad, curl, div) acting on spline coefficients.

In a spline de Rham sequence the derivative of a k-form spline lies in the next spline space, and
its coefficients are differences of neighbouring coefficients (entries -1, 0, +1). These matrices
depend only on the mesh, not on the geometry, and the sequence builds them once when it is created.

- :func:`build_matrixfree_incidence` returns ``G_k`` (``k = 0, 1, 2``: grad, curl, div) and its
  transpose on the raw tensor-product coefficients.
- :func:`build_grad_stencil_g0` and :func:`build_curl_stencil_g1` return grad and curl on the
  extracted DoFs of :mod:`mrx.extraction_operators`, axis functions included. On these spaces
  ``curl grad = 0`` holds exactly.
"""
import equinox as eqx
import jax.numpy as jnp
import numpy as np

from mrx.extraction_operators import summed_coo


def _diff_fwd(V, axis: int, typ: str):
    """1-D incidence along ``axis``: ``c_{j+1} - c_j``, wrapped on a periodic axis."""
    if typ == 'clamped':
        return jnp.diff(V, axis=axis)
    return jnp.roll(V, -1, axis=axis) - V


def _diff_adj(Y, axis: int, typ: str):
    """Transpose of :func:`_diff_fwd` along ``axis``."""
    if typ == 'clamped':
        pad_end = [(0, 0)] * Y.ndim
        pad_end[axis] = (0, 1)
        pad_start = [(0, 0)] * Y.ndim
        pad_start[axis] = (1, 0)
        return jnp.pad(-Y, pad_end) + jnp.pad(Y, pad_start)
    return jnp.roll(Y, 1, axis=axis) - Y


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
        return jnp.concatenate([
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
        return jnp.concatenate([P.ravel(), Q.ravel(), R.ravel()])
    if op.k == 1:
        P, Q, R = _split3(x, op.s2)
        a = _diff_adj(Q, 2, tz) - _diff_adj(R, 1, tt)
        b = -_diff_adj(P, 2, tz) + _diff_adj(R, 0, tr)
        c = _diff_adj(P, 1, tt) - _diff_adj(Q, 0, tr)
        return jnp.concatenate([a.ravel(), b.ravel(), c.ravel()])
    if not op.transpose:
        # div: (a, b, c) -> d_r a + d_t b + d_z c
        a, b, c = _split3(x, op.s2)
        return (_diff_fwd(a, 0, tr) + _diff_fwd(b, 1, tt) + _diff_fwd(c, 2, tz)).ravel()
    Y = x.reshape(op.s3)
    return jnp.concatenate([
        _diff_adj(Y, 0, tr).ravel(),
        _diff_adj(Y, 1, tt).ravel(),
        _diff_adj(Y, 2, tz).ravel(),
    ])


class _MatrixFreeIncidence(eqx.Module):
    """The incidence matrix ``G_k`` or its transpose, applied to a flat raw coefficient vector
    with ``@`` or a call. It holds only shapes, no arrays."""
    k: int = eqx.field(static=True)
    transpose: bool = eqx.field(static=True)
    types: tuple = eqx.field(static=True)
    s0: tuple = eqx.field(static=True)
    s1: tuple = eqx.field(static=True)
    s2: tuple = eqx.field(static=True)
    s3: tuple = eqx.field(static=True)
    shape: tuple = eqx.field(static=True)

    def __matmul__(self, x):
        return _apply_incidence_mf(self, x)

    def __call__(self, x):
        return _apply_incidence_mf(self, x)

    @property
    def T(self):
        return _MatrixFreeIncidence(
            k=self.k,
            transpose=not self.transpose,
            types=self.types,
            s0=self.s0, s1=self.s1, s2=self.s2, s3=self.s3,
            shape=(self.shape[1], self.shape[0]),
        )


def build_matrixfree_incidence(seq, k: int):
    """``(G_k, G_k^T)`` on the raw tensor-product coefficients of ``seq``, for ``k = 0, 1, 2``."""
    s0 = tuple(int(v) for v in seq.basis_0.shape[0])
    s3 = tuple(int(v) for v in seq.basis_3.shape[0])
    s1 = tuple(tuple(int(v) for v in comp) for comp in seq.basis_1.shape)
    s2 = tuple(tuple(int(v) for v in comp) for comp in seq.basis_2.shape)
    sizes = (_prod3(s0), sum(_prod3(c) for c in s1), sum(_prod3(c) for c in s2), _prod3(s3))
    n_in, n_out = sizes[k], sizes[k + 1]
    common = dict(k=k, types=tuple(seq.basis_0.types), s0=s0, s1=s1, s2=s2, s3=s3)
    g = _MatrixFreeIncidence(transpose=False, shape=(n_out, n_in), **common)
    g_T = _MatrixFreeIncidence(transpose=True, shape=(n_in, n_out), **common)
    return g, g_T


def _stencil_grid(*dims):
    """The flattened index grids over ``dims``, in C order."""
    return [g.reshape(-1) for g in np.meshgrid(*(np.arange(d) for d in dims),
                                               indexing='ij')]


class _StencilTriplets:
    """Collects the entries of a sparse matrix. ``emit`` drops zero entries."""

    def __init__(self):
        self.rows, self.cols, self.data = [], [], []

    def emit(self, rows, cols, data):
        rows, cols = np.broadcast_arrays(rows, cols)
        data = np.broadcast_to(np.asarray(data, dtype=np.float64), rows.shape)
        keep = data != 0.0
        self.rows.append(rows[keep])
        self.cols.append(cols[keep])
        self.data.append(data[keep])

    def operator(self, shape, dtype=None):
        """The collected entries, duplicates summed, as a :class:`~mrx.extraction_operators.MatrixFreeExtraction`
        with values in ``dtype`` (the working dtype by default)."""
        return summed_coo(np.concatenate(self.rows), np.concatenate(self.cols), np.concatenate(self.data),
                          shape, dtype=dtype)


def build_grad_stencil_g0(seq, xi, dirichlet: bool, dtype=None):
    """The gradient ``V0 -> V1`` on the extracted DoFs, as a sparse matrix.

    ``xi`` are the polar weights of :func:`~mrx.extraction_operators.get_xi`, and ``dirichlet``
    chooses the free or the Dirichlet version of both spaces. The result equals ``(E_1 E_1^T)^{-1} E_1 G_0 E_0^T`` with the raw gradient
    ``G_0`` and the extractions ``E_k``: the exact gradient of an extracted 0-form, written in the
    extracted 1-form basis. It is assembled directly from coefficient differences.
    """
    xi = np.asarray(xi)
    nr, nt, nz = (int(v) for v in seq.basis_0.shape[0])
    dr = nr - 1            # clamped r derivative count
    dt, dz = nt, nz        # periodic theta, z: derivative count == primal
    o0 = o1 = 1 if dirichlet else 0
    radial0 = nr - 2 - o0  # V0 bulk radial rings (raw rings >= 2)
    radial1 = nr - 2 - o1  # V1 comp1/comp2 bulk radial rings

    base_bulk0 = 3 * nz
    out = _StencilTriplets()

    def expand(r, a, j, k, s):
        """Add ``s`` times the raw V0 DoF ``(a, j, k)``, written in extracted DoFs, to rows ``r``.
        Rings 0 and 1 are ``sum_p xi[p, ring, j] axis(p, k)`` with the axis DoFs ``axis(p, k)``."""
        for ring in (0, 1):
            m = a == ring
            for p in range(3):
                out.emit(r[m], p * nz + k[m], s * xi[p, ring, j[m]])
        m = (a >= 2) & (a - 2 < radial0)
        out.emit(r[m], base_bulk0 + ((a[m] - 2) * nt + j[m]) * nz + k[m], s)

    # Extracted V0: the 3 nz axis DoFs, then the rings from 2 outwards. Extracted V1 rows, in the
    # order of build_extraction: 2 nz theta axis rows, 3 dz zeta axis rows, then the r, theta and
    # zeta components of the bulk.
    r_theta_s = 0
    r_zeta_s = 2 * nz
    r_r = 2 * nz + 3 * dz
    r_theta_b = r_r + (dr - 1) * nt * nz
    r_zeta_b = r_theta_b + radial1 * dt * nz

    # theta axis rows: axis(p + 1, m) - axis(0, m)
    pl, m = _stencil_grid(2, nz)
    out.emit(r_theta_s + pl * nz + m, (pl + 1) * nz + m, 1.0)
    out.emit(r_theta_s + pl * nz + m, m, -1.0)

    # zeta axis rows: periodic zeta difference of the axis DoFs
    p, m = _stencil_grid(3, dz)
    out.emit(r_zeta_s + p * dz + m, p * nz + (m + 1) % nz, 1.0)
    out.emit(r_zeta_s + p * dz + m, p * nz + m, -1.0)

    # r component: raw(i+2,j,k) - raw(i+1,j,k)
    i, j, k = _stencil_grid(dr - 1, nt, nz)
    r = r_r + np.arange(i.size)
    expand(r, i + 2, j, k, 1.0)
    expand(r, i + 1, j, k, -1.0)

    # theta component (periodic): raw(i+2,j+1) - raw(i+2,j)
    i, j, k = _stencil_grid(radial1, dt, nz)
    r = r_theta_b + np.arange(i.size)
    expand(r, i + 2, (j + 1) % nt, k, 1.0)
    expand(r, i + 2, j, k, -1.0)

    # zeta component (periodic): raw(i+2,k+1) - raw(i+2,k)
    i, j, k = _stencil_grid(radial1, nt, dz)
    r = r_zeta_b + np.arange(i.size)
    expand(r, i + 2, j, (k + 1) % nz, 1.0)
    expand(r, i + 2, j, k, -1.0)

    n0, n1 = (seq.extraction[(k, bool(dirichlet))].forward_shape[0] for k in (0, 1))
    return out.operator((n1, n0), dtype=dtype)


def build_curl_stencil_g1(seq, xi, dirichlet: bool, dtype=None):
    """The curl ``V1 -> V2`` on the extracted DoFs, as a sparse matrix.

    The counterpart of :func:`build_grad_stencil_g0` one degree up. It equals
    ``(E_2 E_2^T)^{-1} E_2 G_1 E_1^T``. On the raw components ``(a, b, c)`` (r, theta, zeta)
    of a 1-form the curl has the components ``P = -d_z b + d_t c``, ``Q = d_z a - d_r c`` and
    ``R = -d_t a + d_r b``, where ``d`` is the coefficient difference along an axis.
    """
    xi = np.asarray(xi)
    nr, nt, nz = (int(v) for v in seq.basis_0.shape[0])
    dr, dt, dz = nr - 1, nt, nz
    o_in = o_out = 1 if dirichlet else 0
    radial_in = nr - 2 - o_in
    radial_out = nr - 2 - o_out

    # V1 extracted (input) columns
    base_r1 = 2 * nz + 3 * dz
    base_tb1 = base_r1 + (dr - 1) * nt * nz
    base_zb1 = base_tb1 + radial_in * dt * nz
    out = _StencilTriplets()

    def c_ths(pl, m):                                  # V1 theta axis column
        return pl * nz + m

    def c_zes(p, m):                                   # V1 zeta axis column
        return 2 * nz + p * dz + m

    def expand_v1(r, comp, a, j, k, s):
        """Add ``s`` times the raw V1 DoF ``(comp, a, j, k)``, written in extracted DoFs, to rows ``r``."""
        if comp == 0:                                  # r, raw radial a in [0,dr)
            m = a == 0
            for pl in range(2):
                out.emit(r[m], c_ths(pl, k[m]),
                         s * (xi[pl + 1, 1, j[m]] - xi[pl + 1, 0, j[m]]))
            m = (a >= 1) & (a - 1 < dr - 1)
            out.emit(r[m], base_r1 + ((a[m] - 1) * nt + j[m]) * nz + k[m], s)
        elif comp == 1:                                # theta, raw radial a in [0,nr)
            m = a == 1
            for pl in range(2):
                out.emit(r[m], c_ths(pl, k[m]),
                         s * (xi[pl + 1, 1, (j[m] + 1) % dt] - xi[pl + 1, 1, j[m]]))
            m = (a >= 2) & (a - 2 < radial_in)
            out.emit(r[m], base_tb1 + ((a[m] - 2) * dt + j[m]) * nz + k[m], s)
        else:                                          # zeta, raw radial a in [0,nr)
            for ring in (0, 1):
                m = a == ring
                for p in range(3):
                    out.emit(r[m], c_zes(p, k[m]), s * xi[p, ring, j[m]])
            m = (a >= 2) & (a - 2 < radial_in)
            out.emit(r[m], base_zb1 + ((a[m] - 2) * nt + j[m]) * dz + k[m], s)

    # V2 extracted (output) row offsets (match the k=2 layout of build_extraction)
    n1_v2 = (radial_out * dt + 2) * dz   # comp0 extracted size (2 dz axis rows + bulk)
    n2_v2 = (dr - 1) * nt * dz           # comp1 extracted size
    r_c0b = 2 * dz                       # comp0 bulk start
    r_c1 = n1_v2                         # comp1 bulk start
    r_c2 = n1_v2 + n2_v2                 # comp2 bulk start

    # comp0 axis rows [0, 2 dz), the only V2 rows that combine raw DoFs:
    # P = -d_z(theta axis DoF) + (zeta axis DoF difference)
    pl, m = _stencil_grid(2, dz)
    r = pl * dz + m
    out.emit(r, c_ths(pl, m), 1.0)
    out.emit(r, c_ths(pl, (m + 1) % dz), -1.0)
    out.emit(r, c_zes(pl + 1, m), 1.0)
    out.emit(r, c_zes(0, m), -1.0)

    # comp0 bulk: P[i+2,j,k] = -d_z(theta) + d_t(zeta)
    i, j, k = _stencil_grid(radial_out, dt, dz)
    r = r_c0b + np.arange(i.size)
    expand_v1(r, 1, i + 2, j, (k + 1) % nz, -1.0)
    expand_v1(r, 1, i + 2, j, k, 1.0)
    expand_v1(r, 2, i + 2, (j + 1) % nt, k, 1.0)
    expand_v1(r, 2, i + 2, j, k, -1.0)

    # comp1 bulk: Q[i+1,j,k] = d_z(r) - d_r(zeta)
    i, j, k = _stencil_grid(dr - 1, nt, dz)
    r = r_c1 + np.arange(i.size)
    expand_v1(r, 0, i + 1, j, (k + 1) % nz, 1.0)
    expand_v1(r, 0, i + 1, j, k, -1.0)
    expand_v1(r, 2, i + 2, j, k, -1.0)
    expand_v1(r, 2, i + 1, j, k, 1.0)

    # comp2 bulk: R[i+1,j,k] = -d_t(r) + d_r(theta)
    i, j, k = _stencil_grid(dr - 1, dt, nz)
    r = r_c2 + np.arange(i.size)
    expand_v1(r, 0, i + 1, (j + 1) % nt, k, -1.0)
    expand_v1(r, 0, i + 1, j, k, 1.0)
    expand_v1(r, 1, i + 2, j, k, 1.0)
    expand_v1(r, 1, i + 1, j, k, -1.0)

    n1, n2 = (seq.extraction[(k, bool(dirichlet))].forward_shape[0] for k in (1, 2))
    return out.operator((n2, n1), dtype=dtype)
