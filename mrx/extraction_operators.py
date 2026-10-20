"""Polar extraction: the spline spaces of the de Rham sequence near the magnetic axis.

Tensor-product splines in ``(r, theta, zeta)`` are not smooth at the axis ``r = 0``. The
extraction matrix ``E`` of a k-form space combines the two innermost radial rings of basis
functions into a few smooth axis functions (Toshniwal et al., CMAME 2017). The rows of ``E`` are the
extracted basis functions and its columns the raw tensor-product ones, so an extracted DoF vector
``v`` has the raw coefficients ``E^T v``.

On a uniform space every other basis function is kept unchanged. On a graded space each radial ring
of basis functions has its own uniform periodic theta space, with fewer cells near the axis
(:func:`graded_theta_counts`). Its basis functions are combinations of the raw ones on the same ring,
by knot insertion (:func:`prolongation`). The polar axis functions are built on the coarsest of these
theta spaces. The uniform space is the graded one with every count equal to ``n_theta``.

- :func:`build_extraction` builds ``E`` of a k-form space, free or with homogeneous Dirichlet
  conditions at ``r = 1``, together with its polar core rows (the axis functions).
- :func:`ring_layout` numbers the other rows by component, radial ring, theta and zeta index.
- :class:`SparseMatrix` is the sparse matrix type of ``E`` and of the other topological operators (the polar
  grad and curl, the parity reductions). It acts by ``E @ x`` and ``E.T @ x``.
- :func:`conforming_restriction` maps raw coefficients to the closest extracted DoF vector.
- :func:`dirichlet_dofs` locates the DoFs of a Dirichlet space among those of the free space.
- :func:`get_xi` returns the polar weights that define the axis functions.

All of these depend only on the mesh and the spline degrees, not on the geometry. The sequence
builds them once when it is created.
"""

import functools
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp


@functools.partial(jax.jit, static_argnames=("num_segments",))
def _apply_coo(vals, gather_idx, segment_idx, x, num_segments):
    """``A x`` for the sparse matrix ``A`` given by its entries."""
    # module-level jit: every operator of one shape shares a single compiled program
    weights = vals if x.ndim == 1 else vals[:, None]
    return jax.ops.segment_sum(weights * x[gather_idx], segment_idx,
                               num_segments=num_segments)


class SparseMatrix(eqx.Module):
    """A sparse matrix stored as its nonzero entries ``(rows, cols, vals)``.

    ``A @ x`` applies it to a vector, or to each column of a matrix ``x``, and ``A.T`` is its
    transpose (no copy). Duplicate ``(row, col)`` entries are summed. It is an equinox module, so
    it can be passed into jitted functions."""

    rows: jnp.ndarray
    cols: jnp.ndarray
    vals: jnp.ndarray
    forward_shape: tuple = eqx.field(static=True)
    transposed: bool = eqx.field(static=True)

    @classmethod
    def from_coo(cls, rows, cols, vals, shape):
        """The matrix of shape ``(n_row, n_col)`` with the given entries."""
        return cls(
            rows=jnp.asarray(np.asarray(rows, dtype=np.int32)),
            cols=jnp.asarray(np.asarray(cols, dtype=np.int32)),
            vals=jnp.asarray(np.asarray(vals), dtype=jnp.float64),
            forward_shape=(int(shape[0]), int(shape[1])),
            transposed=False,
        )

    @property
    def shape(self):
        if self.transposed:
            return (self.forward_shape[1], self.forward_shape[0])
        return self.forward_shape

    @property
    def dtype(self):
        return self.vals.dtype

    @property
    def T(self):
        return SparseMatrix(rows=self.rows, cols=self.cols, vals=self.vals,
                                    forward_shape=self.forward_shape,
                                    transposed=not self.transposed)

    def __matmul__(self, x):
        x = jnp.asarray(x)
        if self.transposed:     # gather from the rows, scatter into the columns
            return _apply_coo(self.vals, self.rows, self.cols, x,
                              num_segments=self.forward_shape[1])
        return _apply_coo(self.vals, self.cols, self.rows, x,
                          num_segments=self.forward_shape[0])

    def entries(self, rows=None):
        """The entries ``(rows, cols, values)`` of the untransposed matrix as NumPy arrays. With ``rows``, only the entries of those rows, renumbered
        ``0 .. len(rows) - 1``."""
        r, c = np.asarray(self.rows), np.asarray(self.cols)
        v = np.asarray(self.vals)
        if rows is None:
            return r, c, v
        pos = np.full(self.forward_shape[0], -1)
        pos[np.asarray(rows)] = np.arange(len(rows))
        keep = pos[r] >= 0
        return pos[r[keep]], c[keep], v[keep]


def summed_coo(rows, cols, vals, shape):
    """A :class:`SparseMatrix` of ``shape`` from the given entries, with duplicate
    entries summed into one and the entries sorted by row."""
    key = np.asarray(rows, dtype=np.int64) * shape[1] + np.asarray(cols, dtype=np.int64)
    uniq, inv = np.unique(key, return_inverse=True)
    return SparseMatrix.from_coo(uniq // shape[1], uniq % shape[1],
                                         np.bincount(inv, weights=vals, minlength=uniq.size), shape)


def row_products(a, b, shape):
    """The dense product ``A B^T`` of shape ``shape`` for two sparse matrices given by their entries
    ``(rows, cols, values)`` over the same columns. It is fast only when each column holds few
    entries, as on the polar core rows."""
    ra, ca, va = a
    order = np.argsort(b[1], kind="stable")
    rb, cb, vb = (x[order] for x in b)
    lo, hi = np.searchsorted(cb, ca, "left"), np.searchsorted(cb, ca, "right")
    out = np.zeros(shape)
    for j in range(int((hi - lo).max(initial=0))):
        hit = lo + j < hi
        np.add.at(out, (ra[hit], rb[lo[hit] + j]), va[hit] * vb[lo[hit] + j])
    return out


def core_gram_inverse(e, core):
    """The inverse of ``E E^T`` on the polar core rows of the extraction ``e``, as a dense float64
    matrix. Away from the core rows ``E E^T`` is the identity on a uniform space and one small block per
    ring and zeta layer on a graded one (:func:`gram_blocks`). Neither couples to the core, so this block
    is all it takes on the core rows. The same holds for the parity-reduced extractions of
    :mod:`mrx.symmetry`."""
    rows = e.entries(core)
    return np.linalg.inv(row_products(rows, rows, (len(core), len(core))))


def conforming_restriction(e, c, core, blocks=()):
    """The extracted DoF vector ``v = (E E^T)^-1 E c`` whose raw coefficients ``E^T v`` are closest
    (in the least-squares sense) to the raw tensor-product coefficients ``c``. ``core`` are the
    polar core rows of ``e``. Away from them ``E E^T`` is the identity on a uniform space and one
    small block per ring and zeta layer on a graded one, given as ``blocks`` (:func:`gram_blocks`)."""
    a = e @ c
    for rows, inverse in blocks:
        a = a.at[rows].set(a[rows] @ inverse)
    if len(core) == 0:
        return a
    return a.at[core].set(jnp.asarray(core_gram_inverse(e, core)) @ a[core])


def gram_inverse(e, core, blocks=()):
    """``(E E^T)^-1`` of the extraction ``e`` as a scipy CSR matrix: the dense inverse on the polar core rows
    ``core``, ``(R^T R)^-1`` on each ring and zeta layer of a graded space (``blocks``, :func:`gram_blocks`) and
    the identity on all other rows."""
    n = e.forward_shape[0]
    covered = np.zeros(n, bool)
    rows, cols, vals = [], [], []
    for group, inverse in blocks:
        group, inverse = np.asarray(group), np.asarray(inverse)
        m = group.shape[1]
        rows.append(np.repeat(group, m, axis=1).ravel())
        cols.append(np.tile(group, (1, m)).ravel())
        vals.append(np.broadcast_to(inverse.ravel(), (group.shape[0], m * m)).ravel())
        covered[group] = True
    core = np.asarray(core)
    if core.size:
        rows.append(np.repeat(core, core.size))
        cols.append(np.tile(core, core.size))
        vals.append(core_gram_inverse(e, core).ravel())
        covered[core] = True
    rest = np.flatnonzero(~covered)
    rows, cols, vals = (np.concatenate(a + [b]) for a, b in zip((rows, cols, vals), (rest, rest, np.ones(rest.size))))
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


#: The fewest theta cells of a ring on a graded space. The coarsest count is ``n_theta / 2^l``, the
#: smallest such count not below this.
GRADED_THETA_MIN = 8

#: Grading rule: the theta width of a ring is at least this times its smallest radial cell.
GRADED_ASPECT = np.pi / 2

#: Grading rule: a ring at radius ``r`` has at least ``n_theta r^GRADED_EXPONENT`` theta cells. With the exponent 1 the
#: rings near the axis keep their opening angle under refinement and the L2 error stalls (li383 p = 2, true L2,
#: 36 -> 48: rates 0.8-1.0, 2.7x the uniform error). With 1/3 it converges at order 2 with the uniform error to 1 %,
#: and the stiffness solves keep flat counts (2026-10-20).
GRADED_EXPONENT = 1.0 / 3.0


def prolongation(m, n, p, derivative):
    """The knot insertion ``R`` (shape ``(n, m)``) from the uniform periodic splines of degree ``p`` with ``m``
    cells to those with ``n = m 2^l`` cells: ``B^m_J = sum_j R[j, J] B^n_j``. With ``derivative`` it acts on the
    derivative splines (degree ``p - 1``, unit integral) of :class:`mrx.spline_bases.DerivativeSpline`. Each
    doubling is the subdivision mask ``2^-q C(q + 1, k)`` of degree ``q``, times ``1 / 2`` for the unit integral of
    the derivative splines, so the entries are exact dyadic numbers and ``R`` commutes with the coefficient
    difference: ``G_n R_primal = R_derivative G_m``."""
    q = p - 1 if derivative else p
    R = np.eye(m)
    while m < n:
        D = np.zeros((2 * m, m))
        for k in range(q + 2):
            D[(2 * np.arange(m) - q + k) % (2 * m), np.arange(m)] += math.comb(q + 1, k) / 2.0 ** p
        R, m = D @ R, 2 * m
    if m != n:
        raise ValueError(f"the theta counts {R.shape[1]} and {n} are not related by doubling")
    return R


def graded_theta_counts(T_r, p, nt):
    """The theta count of every primal radial basis function of a graded space with ``nt`` theta cells at the wall.

    Ring ``i`` (``r_i`` its Greville point) has at least ``nt r_i^GRADED_EXPONENT`` theta cells
    (:data:`GRADED_EXPONENT`), or a theta width ``2 pi r_i / m`` of at most :data:`GRADED_ASPECT` times its smallest
    radial cell if that asks for more. ``m`` is the coarsest count (``nt / 2^l``, at least :data:`GRADED_THETA_MIN`)
    times the smallest power of two that reaches this, at most ``nt``, and does not decrease outward. The three
    innermost rings carry the polar axis functions and share the count of the third, so the axis functions are not
    coarser than the ring next to them."""
    T_r = np.asarray(T_r)
    m_min = nt
    while m_min % 2 == 0 and m_min // 2 >= GRADED_THETA_MIN:
        m_min //= 2
    counts = []
    for i in range(T_r.size - p - 1):
        knots = T_r[i:i + p + 2]
        r_i = knots[1:-1].mean()
        h_i = np.diff(np.unique(knots)).min()
        allowed = max(2 * np.pi * r_i / (GRADED_ASPECT * h_i), nt * r_i ** GRADED_EXPONENT)
        counts.append(min(nt, m_min * 2 ** int(np.ceil(np.log2(max(allowed / m_min, 1.0))))))
    counts = np.maximum.accumulate(np.asarray(counts))
    counts[:3] = counts[2]
    return tuple(int(m) for m in counts)


def dirichlet_dofs(form, counts, e_free, e_dirichlet):
    """For each DoF of the Dirichlet space, the index of the DoF of the free space with the same basis
    function, as an integer NumPy array. The Dirichlet space is the free one without the wall
    functions, and both list the axis functions first in the same order. ``counts`` are the theta
    counts of :func:`build_extraction`."""
    rf, cf, vf = e_free.entries()
    rd, cd, vd = e_dirichlet.entries()
    n_d = e_dirichlet.forward_shape[0]
    index = np.arange(n_d)
    free = {(c, a): first for c, a, _, _, first, _ in ring_layout(form, False, counts)}
    for c, a, m, _, first, nz in ring_layout(form, True, counts):
        index[first:first + m * nz] = free[(c, a)] + np.arange(m * nz)
    pos = np.full(e_free.forward_shape[0], -1)
    pos[index] = np.arange(n_d)
    keep = pos[rf] >= 0

    def triplets(r, c, v):
        order = np.lexsort((c, r))
        return r[order], c[order], v[order]
    same = all(np.array_equal(a, b) for a, b in zip(triplets(rd, cd, vd),
                                                     triplets(pos[rf[keep]], cf[keep], vf[keep])))
    if (index < 0).any() or not same:
        raise RuntimeError("the Dirichlet space is not the free space without its wall functions")
    return index


#: For each form degree, the blocks of axis functions at the top of ``E``, in row order.
#: ``("rings", c)`` combines radial rings 0 and 1 of component ``c`` into three axis functions
#: per zeta layer. ``("pair", c_theta, c_r, sign)`` builds two axis functions per zeta layer
#: from ring 1 of component ``c_theta`` and ring 0 of component ``c_r``.
_SURGERY = {0: (("rings", 0),), 1: (("pair", 1, 0, 1.0), ("rings", 2)),
            2: (("pair", 0, 1, -1.0),), 3: ()}


def _core_size(form):
    """The number of polar axis rows of the k-form space ``form``."""
    return sum((3 if block[0] == "rings" else 2) * form.shape[block[1]][2] for block in _SURGERY[form.k])


def _counts(form, counts):
    return np.full(form.nr, form.nt) if counts is None else np.asarray(counts)


def ring_layout(form, dirichlet, counts=None):
    """The rows of :func:`build_extraction` away from the axis, as one record ``(c, a, m, derivative, first, n_z)``
    per component ``c`` and raw radial index ``a``. Ring ``a`` of the component carries a uniform periodic theta
    space with ``m`` cells, of derivative splines if ``derivative``, and its basis function with theta index ``g``
    and zeta index ``l`` is row ``first + g n_z + l``. ``counts`` are the theta counts of the primal radial basis
    functions (:func:`graded_theta_counts`), ``None`` for ``n_theta`` everywhere. A ring of derivative splines in
    ``r`` sits between two primal rings and takes the count of the outer one."""
    counts = _counts(form, counts)
    o = 1 if dirichlet else 0
    first = _core_size(form)
    records = []
    for c, shape in enumerate(form.shape):
        derivative_r = 0 in form.derivative_axes(c)
        derivative_t = 1 in form.derivative_axes(c)
        rings = range(1, form.dr) if derivative_r else range(2, form.nr - o)
        for a in rings:
            m = int(counts[a + 1] if derivative_r else counts[a])
            records.append((c, a, m, derivative_t, first, int(shape[2])))
            first += m * int(shape[2])
    return records


def gram_blocks(form, dirichlet, counts):
    """The blocks of ``(E E^T)^-1`` away from the axis rows of a graded space, as ``(rows, inverse)``: ``rows``
    (shape ``(n, m)``) holds the rows of one ring and zeta layer, ``inverse`` is ``(R^T R)^-1`` of its knot
    insertion ``R``. Rings with the full theta count are left out (``R = I``)."""
    groups = {}
    p = form.Lambda[1].p
    for c, a, m, derivative, first, nz in ring_layout(form, dirichlet, counts):
        n = form.shape[c][1]
        if m == n:
            continue
        rows = first + np.arange(m)[None, :] * nz + np.arange(nz)[:, None]
        groups.setdefault((m, n, derivative), []).append(rows)
    blocks = []
    for (m, n, derivative), rows in groups.items():
        R = prolongation(m, n, p, derivative)
        blocks.append((jnp.asarray(np.concatenate(rows)), jnp.asarray(np.linalg.inv(R.T @ R))))
    return tuple(blocks)


def build_extraction(form, xi, dirichlet, counts=None):
    """The extraction ``E`` of the k-form space ``form`` and the indices of its polar core rows.

    The first rows of ``E`` are the axis functions, built with the polar weights ``xi`` of
    :func:`get_xi` on the coarsest theta count. These are the polar core rows. The other rows
    follow :func:`ring_layout`: per component, all radial rings from ring 2 outwards (from ring 1
    for a component with the derivative basis in ``r``), each with its own theta count. A row is
    the knot insertion of one coarse theta function into the raw functions of its ring, and on a
    uniform space it selects one raw function. With ``dirichlet`` the outermost ring of the primal
    radial basis is dropped, which imposes homogeneous Dirichlet conditions at ``r = 1``.

    Returns ``(E, core_rows)``, ``E`` a :class:`SparseMatrix` of shape
    ``(n_extracted, form.n)``.
    """
    xi = np.asarray(xi)
    p = form.Lambda[1].p
    m0 = xi.shape[2]
    R0, R1 = prolongation(m0, form.nt, p, False), prolongation(m0, form.dt, p, True)
    offsets = np.cumsum([0] + [math.prod(s) for s in form.shape])
    rows, cols, data, core = [], [], [], []

    def col(c, i, j, m):
        return offsets[c] + np.ravel_multi_index((i, j, m), form.shape[c])

    def append(row, col_idx, values):
        col_idx = np.asarray(col_idx, dtype=np.int32).reshape(-1)
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        valid = values != 0.0
        if not np.any(valid):
            return
        core.append(int(row))
        rows.append(np.full(int(np.count_nonzero(valid)), row, dtype=np.int32))
        cols.append(col_idx[valid])
        data.append(values[valid])

    row0 = 0
    for block in _SURGERY[form.k]:
        if block[0] == "rings":
            c = block[1]
            n_layers, js = form.shape[c][2], np.arange(form.nt)
            for q in range(3):
                for m in range(n_layers):
                    for i in range(2):
                        append(row0 + q * n_layers + m, col(c, i, js, m), R0 @ xi[q, i, :])
            row0 += 3 * n_layers
        else:
            _, c_t, c_r, sign = block
            n_layers = form.shape[c_t][2]
            js_t, js_r = np.arange(form.dt), np.arange(form.nt)
            g = np.arange(m0)
            for q_local in range(2):
                q = q_local + 1
                for m in range(n_layers):
                    row = row0 + q_local * n_layers + m
                    append(row, col(c_t, 1, js_t, m), R1 @ (xi[q, 1, (g + 1) % m0] - xi[q, 1, g]))
                    append(row, col(c_r, 0, js_r, m), sign * (R0 @ (xi[q, 1, g] - xi[q, 0, g])))
            row0 += 2 * n_layers

    for c, a, m, derivative, first, nz in ring_layout(form, dirichlet, counts):
        R = prolongation(m, form.shape[c][1], p, derivative)
        j, g = np.nonzero(R)
        iz = np.arange(nz)
        rows.append((first + g[:, None] * nz + iz[None, :]).astype(np.int32).ravel())
        cols.append(np.asarray(col(c, a, j[:, None], iz[None, :]), dtype=np.int32).ravel())
        data.append(np.repeat(R[j, g], nz))
        row0 = first + m * nz

    e = summed_coo(np.concatenate(rows), np.concatenate(cols), np.concatenate(data), (row0, form.n))
    return e, np.unique(np.asarray(core, dtype=np.int32))


def get_xi(nt, p):
    """The polar weights ``xi[l, i, j]``, shape ``(3, 2, nt)``, of the three axis functions
    (Toshniwal et al., CMAME 2017).

    Axis function ``l`` is the sum of the raw basis functions of radial rings ``i = 0, 1`` and
    angular index ``j`` with weights ``xi[l, i, j]``. The weights are the barycentric coordinates
    of the ring control points in an equilateral triangle that encloses them. Ring 0 is the axis
    itself (all weights 1/3). Ring 1 lies on the unit circle at ``theta_j = 2 pi (j - (p - 1) / 2) / nt``,
    the centre of the periodic degree-``p`` spline ``B_j``. With this choice the three axis
    functions map onto each other under ``theta -> -theta``, which the stellarator symmetry of
    :mod:`mrx.symmetry` relies on.
    """
    theta_js = ((jnp.arange(nt) - (p - 1) / 2.0) / nt) * 2 * jnp.pi
    dR, dY = jnp.cos(theta_js), jnp.sin(theta_js)

    s3 = jnp.sqrt(3.0)
    tau = jnp.max(jnp.array([jnp.max(-2.0 * dR),
                             jnp.max(dR - s3 * dY),
                             jnp.max(dR + s3 * dY)]))
    xi1 = jnp.stack([1/3 + 2.0 * dR / (3.0 * tau),
                    1/3 - dR / (3.0 * tau) + s3 * dY / (3.0 * tau),
                    1/3 - dR / (3.0 * tau) - s3 * dY / (3.0 * tau)])  # (3, n_theta)
    xi0 = jnp.full((3, nt), 1.0 / 3.0)
    return jnp.stack([xi0, xi1], axis=1)
