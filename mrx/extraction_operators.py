"""Polar extraction: the spline spaces of the de Rham sequence near the magnetic axis.

Tensor-product splines in ``(r, theta, zeta)`` are not smooth at the axis ``r = 0``. The
extraction matrix ``E`` of a k-form space combines the two innermost radial rings of basis
functions into a few smooth axis functions (Toshniwal et al., CMAME 2017) and keeps every other
basis function unchanged. The rows of ``E`` are the extracted basis functions and its columns the
raw tensor-product ones, so an extracted DoF vector ``v`` has the raw coefficients ``E^T v``.

- :func:`build_extraction` builds ``E`` of a k-form space, free or with homogeneous Dirichlet
  conditions at ``r = 1``, together with its polar core rows (the axis functions).
- :class:`MatrixFreeExtraction` is the sparse matrix type of ``E`` and of the other topological
  operators. It acts by ``E @ x`` and ``E.T @ x``.
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

import mrx


@functools.partial(jax.jit, static_argnames=("num_segments",))
def _apply_coo(vals, gather_idx, segment_idx, x, num_segments):
    """``A x`` for the sparse matrix ``A`` given by its entries."""
    # module-level jit: every operator of one shape shares a single compiled program
    weights = vals if x.ndim == 1 else vals[:, None]
    return jax.ops.segment_sum(weights * x[gather_idx], segment_idx,
                               num_segments=num_segments)


class MatrixFreeExtraction(eqx.Module):
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
    def from_coo(cls, rows, cols, vals, shape, dtype=None):
        """The matrix of shape ``(n_row, n_col)`` with the given entries. The values are stored
        in ``dtype``, the working dtype by default."""
        return cls(
            rows=jnp.asarray(np.asarray(rows, dtype=np.int32)),
            cols=jnp.asarray(np.asarray(cols, dtype=np.int32)),
            vals=jnp.asarray(np.asarray(vals), dtype=mrx.DTYPE if dtype is None else dtype),
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
        return MatrixFreeExtraction(rows=self.rows, cols=self.cols, vals=self.vals,
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
        """The entries ``(rows, cols, values)`` of the untransposed matrix as NumPy arrays, the
        values in float64. With ``rows``, only the entries of those rows, renumbered
        ``0 .. len(rows) - 1``."""
        r, c = np.asarray(self.rows), np.asarray(self.cols)
        v = np.asarray(self.vals, dtype=np.float64)
        if rows is None:
            return r, c, v
        pos = np.full(self.forward_shape[0], -1)
        pos[np.asarray(rows)] = np.arange(len(rows))
        keep = pos[r] >= 0
        return pos[r[keep]], c[keep], v[keep]


def summed_coo(rows, cols, vals, shape, dtype=None):
    """A :class:`MatrixFreeExtraction` of ``shape`` from the given entries, with duplicate
    entries summed into one and the entries sorted by row."""
    key = np.asarray(rows, dtype=np.int64) * shape[1] + np.asarray(cols, dtype=np.int64)
    uniq, inv = np.unique(key, return_inverse=True)
    return MatrixFreeExtraction.from_coo(uniq // shape[1], uniq % shape[1],
                                         np.bincount(inv, weights=vals, minlength=uniq.size), shape, dtype=dtype)


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
    matrix. Away from the core rows ``E E^T`` is the identity and does not couple to the core, so
    this block is all it takes to invert ``E E^T``. The same holds for the parity-reduced
    extractions of :mod:`mrx.symmetry`."""
    rows = e.entries(core)
    return np.linalg.inv(row_products(rows, rows, (len(core), len(core))))


def conforming_restriction(e, c, core):
    """The extracted DoF vector ``v = (E E^T)^-1 E c`` whose raw coefficients ``E^T v`` are closest
    (in the least-squares sense) to the raw tensor-product coefficients ``c``. ``core`` are the
    polar core rows of ``e``, the only rows where ``E E^T`` differs from the identity."""
    a = e @ c
    if len(core) == 0:
        return a
    return a.at[core].set(jnp.asarray(core_gram_inverse(e, core), dtype=a.dtype) @ a[core])


def dirichlet_dofs(e_free, e_dirichlet):
    """For each DoF of the Dirichlet space, the index of the DoF of the free space with the same basis
    function, as an integer NumPy array. The Dirichlet space is the free one without the wall
    functions, and both list the axis functions first in the same order."""
    rf, cf, vf = e_free.entries()
    rd, cd, vd = e_dirichlet.entries()
    n_d = e_dirichlet.forward_shape[0]
    single_f = np.bincount(rf, minlength=e_free.forward_shape[0])[rf] == 1
    single_d = np.bincount(rd, minlength=n_d)[rd] == 1
    # a row with one entry selects one raw function, and its free row selects the same one
    owner = np.full(e_free.forward_shape[1], -1)
    owner[cf[single_f]] = rf[single_f]
    index = np.arange(n_d)
    index[rd[single_d]] = owner[cd[single_d]]
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


def build_extraction(form, xi, dirichlet, dtype=None):
    """The extraction ``E`` of the k-form space ``form`` and the indices of its polar core rows.

    The first rows of ``E`` are the axis functions, built with the polar weights ``xi`` of
    :func:`get_xi`. These are the polar core rows. Every other row selects one raw basis
    function: per component, all radial rings from ring 2 outwards (from ring 1 for a component
    with the derivative basis in ``r``). With ``dirichlet`` the outermost ring of the primal
    radial basis is dropped, which imposes homogeneous Dirichlet conditions at ``r = 1``.

    Returns ``(E, core_rows)``, ``E`` a :class:`MatrixFreeExtraction` of shape
    ``(n_extracted, form.n)`` with values in ``dtype`` (the working dtype by default).
    """
    dtype = mrx.DTYPE if dtype is None else dtype
    xi = np.asarray(xi)
    o = 1 if dirichlet else 0
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
            for p in range(3):
                for m in range(n_layers):
                    for i in range(2):
                        append(row0 + p * n_layers + m, col(c, i, js, m), xi[p, i, :])
            row0 += 3 * n_layers
        else:
            _, c_t, c_r, sign = block
            n_layers = form.shape[c_t][2]
            js_t, js_r = np.arange(form.dt), np.arange(form.nt)
            for p_local in range(2):
                p = p_local + 1
                for m in range(n_layers):
                    row = row0 + p_local * n_layers + m
                    append(row, col(c_t, 1, js_t, m), xi[p, 1, np.mod(js_t + 1, form.dt)] - xi[p, 1, js_t])
                    append(row, col(c_r, 0, js_r, m), sign * (xi[p, 1, js_r] - xi[p, 0, js_r]))
            row0 += 2 * n_layers

    for c, shape in enumerate(form.shape):
        derivative_r = 0 in form.derivative_axes(c)
        i_offset = 1 if derivative_r else 2
        radial = form.dr - 1 if derivative_r else form.nr - 2 - o
        i, j, m = (ax.ravel() for ax in np.indices((radial,) + tuple(shape[1:])))
        rows.append((row0 + np.arange(i.shape[0])).astype(np.int32))
        cols.append(np.asarray(col(c, i + i_offset, j, m), dtype=np.int32))
        data.append(np.ones(i.shape[0], dtype=np.float64))
        row0 += i.shape[0]

    e = MatrixFreeExtraction(
        rows=jnp.asarray(np.concatenate(rows), dtype=jnp.int32),
        cols=jnp.asarray(np.concatenate(cols), dtype=jnp.int32),
        vals=jnp.asarray(np.concatenate(data), dtype=dtype),
        forward_shape=(row0, form.n), transposed=False)
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
