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
- :class:`StructuredExtraction` applies an extraction without index arrays: static slices for the
  rows that select raw functions, a small dense block for the axis functions.
- :func:`conforming_restriction` maps raw coefficients to the closest extracted DoF vector.
- :func:`dirichlet_dofs` locates the DoFs of a Dirichlet space among those of the free space.
- :func:`get_xi` returns the polar weights that define the axis functions.

All of these depend only on the mesh and the spline degrees, not on the geometry. The sequence
builds them once when it is created.
"""

import dataclasses
import functools
import math
import os
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import mrx.double_single as ds
from mrx.precision import PAIRS

#: Whether an extraction applies through its :class:`StructuredExtraction` when it has one
#: (``MRX_EXTRACTION=structured``, the default) or through its entries (``MRX_EXTRACTION=coo``).
STRUCTURED = os.environ.get("MRX_EXTRACTION", "structured") != "coo"


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
    it can be passed into jitted functions.

    In float32 (:mod:`mrx.precision`) the entries are kept to double-single accuracy: ``lo`` holds their second
    parts and ``by_rows``, ``by_cols`` the matrix and its transpose in the row-wise form of
    :class:`mrx.double_single.Rows`. Applied to a :class:`~mrx.double_single.DS` the matrix returns one."""

    rows: jnp.ndarray
    cols: jnp.ndarray
    vals: jnp.ndarray
    forward_shape: tuple = eqx.field(static=True)
    transposed: bool = eqx.field(static=True)
    lo: Optional[jnp.ndarray] = None
    by_rows: Optional[ds.Rows] = None
    by_cols: Optional[ds.Rows] = None
    structured: Optional["StructuredExtraction"] = None

    @classmethod
    def from_coo(cls, rows, cols, vals, shape):
        """The matrix of shape ``(n_row, n_col)`` with the given entries (NumPy, float64 values)."""
        shape = (int(shape[0]), int(shape[1]))
        rows, cols = np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32)
        vals = np.asarray(vals, dtype=np.float64)
        if not PAIRS:
            return cls(rows=jnp.asarray(rows), cols=jnp.asarray(cols), vals=jnp.asarray(vals),
                       forward_shape=shape, transposed=False)
        parts = ds.from_host(vals)
        return cls(rows=jnp.asarray(rows), cols=jnp.asarray(cols), vals=parts.hi, forward_shape=shape,
                   transposed=False, lo=parts.lo, by_rows=ds.Rows.from_coo(rows, cols, vals, shape[0]),
                   by_cols=ds.Rows.from_coo(cols, rows, vals, shape[1]))

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
                                    transposed=not self.transposed, lo=self.lo, by_rows=self.by_rows,
                                    by_cols=self.by_cols, structured=self.structured)

    def __matmul__(self, x):
        if STRUCTURED and self.structured is not None:
            return _apply_structured(self.structured, x if isinstance(x, ds.DS) else jnp.asarray(x),
                                     self.transposed)
        return self.coo_matmul(x)

    def coo_matmul(self, x):
        """``A @ x`` from the entries (a gather and a segment sum, or :class:`~mrx.double_single.Rows`), also
        when the matrix has a :class:`StructuredExtraction`. It is the reference the structured apply is
        tested against."""
        if isinstance(x, ds.DS):
            return (self.by_cols if self.transposed else self.by_rows) @ x
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
        v = np.asarray(self.vals, dtype=np.float64)
        if self.lo is not None:
            v = v + np.asarray(self.lo, dtype=np.float64)
        if rows is None:
            return r, c, v
        pos = np.full(self.forward_shape[0], -1)
        pos[np.asarray(rows)] = np.arange(len(rows))
        keep = pos[r] >= 0
        return pos[r[keep]], c[keep], v[keep]


def _zeros(x, shape):
    """Zeros of ``shape`` of the kind of ``x``, a float array or a :class:`~mrx.double_single.DS`."""
    return ds.zeros(shape) if isinstance(x, ds.DS) else jnp.zeros(shape, dtype=x.dtype)


def _part(w, x):
    """The weights ``w`` as ``x`` takes them: a float array is multiplied by their first part only."""
    return w.hi if isinstance(w, ds.DS) and not isinstance(x, ds.DS) else w


def _dense(A, X):
    """``A @ X`` for a dense matrix ``A`` and a matrix ``X``, in double-single if either is a :class:`DS`."""
    if not isinstance(A, ds.DS) and not isinstance(X, ds.DS):
        return A @ X
    return ds.contract(A[:, None, :], ds.moveaxis(X, 0, 1)[None])


def _host(x):
    """A float64 NumPy array as the device keeps the operator entries (a :class:`DS` in float32)."""
    return ds.from_host(x) if PAIRS else jnp.asarray(x)


class StructuredExtraction(eqx.Module):
    """An extraction ``E``, or its parity reduction ``X^T E`` (:mod:`mrx.symmetry`), applied without index
    arrays. A device such as a TPU runs a gather or a scatter far slower than slices, reshapes and small
    dense products, and the extractions are made of the latter:

    - **Bulk rows.** Each selects one raw function, or a reflected pair of them with weights ``w`` and
      ``+-w``. With ``R`` the (unsigned) reflection of the raw coefficients, a reversal and a roll of the
      angular axes, the bulk rows are ``w * (S x + sigma * S R x)``, where ``S`` keeps the lead functions
      in raw order and ``sigma`` is ``+-1``, or 0 for a function its own reflection. Only the
      multiplication by ``w`` rounds. The lead functions form a few runs that repeat with a fixed stride
      (once per radial ring), so ``S`` is a list of segments ``(start, stride, count, pieces)``: entries
      ``start + i stride + [o, o + n)`` for ``i < count`` and each piece ``(o, n)``, one strided slice of
      the vector each.
    - **Core rows** (the axis functions). The axis functions of ``E`` combine rings 0 and 1 within one zeta
      layer: ``C = W X``, with ``X`` the first two rings of each component involved as a matrix
      (ring and theta index, zeta layer) and ``W`` of shape (axis functions per layer, ring and theta
      index). A parity reduction combines these rows with the small matrix ``K``.

    :meth:`build` checks the structure against the entries of the extraction and returns ``None`` where it
    does not hold, which keeps the extraction on its entries. The weights are kept as the operator entries
    are, in double-single in float32, and the apply takes float arrays and :class:`DS` alike.
    """

    w: Optional[object]             # the weight of each bulk row, None when it is 1
    sigma: Optional[jnp.ndarray]    # the sign of the reflected function in each bulk row, None without a reflection
    W: Optional[object]             # (Q, window rows): the axis functions of one zeta layer
    WT: Optional[object]
    K: Optional[object]             # (core rows, Q * layers): the parity reduction of the axis functions
    KT: Optional[object]
    shape: tuple = eqx.field(static=True)
    components: tuple = eqx.field(static=True)      # (offset, shape) of each component in the raw vector
    shifts: Optional[tuple] = eqx.field(static=True)    # per component, the theta and zeta rolls of R
    segments: tuple = eqx.field(static=True)
    pad: int = eqx.field(static=True)               # zeros past the raw vector that the last segment reads
    windows: tuple = eqx.field(static=True)         # (offset, size) of the first two rings of each core component
    layers: int = eqx.field(static=True)
    n_core: int = eqx.field(static=True)
    core_first: bool = eqx.field(static=True)

    @classmethod
    def build(cls, e, components, core, full=None, reduce=None, reflection=None):
        """The structured form of the extraction ``e`` (a :class:`MatrixFreeExtraction`) of a space whose
        raw coefficients have the component shapes ``components``, with polar core rows ``core``, or
        ``None`` if its entries do not have that structure.

        For a parity reduction ``e = X^T E`` pass ``full = (E, core rows of E)``, ``reduce``, the block of
        ``X^T`` from the core rows of ``E`` to the core rows of ``e`` (dense, float64), and ``reflection``,
        the theta and zeta index permutations of each component (:func:`mrx.symmetry.reflection_plan`)."""
        rows, cols, vals = e.entries()
        n_rows, n_cols = e.forward_shape
        offsets = np.cumsum([0] + [math.prod(s) for s in components])
        core = np.asarray(core, dtype=np.int64)
        n_core = core.size
        core_first = n_core == 0 or core[0] == 0
        if n_core and not np.array_equal(core, np.arange(n_core) if core_first else np.arange(n_rows - n_core, n_rows)):
            return None
        in_core = np.zeros(n_rows, dtype=bool)
        in_core[core] = True

        # the reflection as a raw index map, and as rolls of the reversed angular axes
        r_map, shifts = None, None
        if reflection is not None:
            r_map, shifts = [], []
            for off, shape, (perm_t, perm_z) in zip(offsets, components, reflection):
                perm_t, perm_z = np.asarray(perm_t), np.asarray(perm_z)
                index = np.arange(math.prod(shape)).reshape(shape)[:, perm_t][:, :, perm_z]
                r_map.append(off + index.ravel())
                shifts.append(((int(perm_t[0]) + 1) % shape[1], (int(perm_z[0]) + 1) % shape[2]))
                rolled = np.roll(np.flip(np.arange(math.prod(shape)).reshape(shape), (1, 2)), shifts[-1], (1, 2))
                if not np.array_equal(rolled, index):
                    return None
            r_map, shifts = np.concatenate(r_map), tuple(shifts)

        # bulk rows: one or two entries, the lead (smallest) column increasing with the row
        bulk = ~in_core[rows]
        order = np.lexsort((cols[bulk], rows[bulk]))
        br, bc, bv = rows[bulk][order], cols[bulk][order], vals[bulk][order]
        bulk_rows = np.flatnonzero(~in_core)
        count = np.bincount(br, minlength=n_rows)[bulk_rows]
        if bulk_rows.size == 0 or count.min() < 1 or count.max() > 2:
            return None
        first = np.concatenate([[0], np.cumsum(count)[:-1]])
        lead, two = bc[first], count == 2
        if np.any(np.diff(lead) <= 0):
            return None
        w, sigma = bv[first], None
        if two.any():
            if r_map is None or not np.array_equal(bc[first[two] + 1], r_map[lead[two]]):
                return None
            sigma = np.zeros(lead.size)
            sigma[two] = bv[first[two] + 1] / w[two]
            if not np.all(np.abs(sigma[two]) == 1.0):
                return None
        segments = _segments(lead)
        implied = [(bulk_rows, lead, w)]
        if sigma is not None:
            implied.append((bulk_rows[two], r_map[lead[two]], sigma[two] * w[two]))

        # core rows: the axis functions of the full extraction, layer by layer
        windows, layers, W, K = (), 0, None, None
        if n_core:
            e_full, core_full = full if full is not None else (e, core)
            fr, fc, fv = e_full.entries(core_full)
            comp = np.searchsorted(offsets, fc, "right") - 1
            used = np.unique(comp)
            layers = int(components[used[0]][2])
            local = fc - offsets[comp]
            if any(components[c][2] != layers for c in used) or len(core_full) % layers \
                    or not np.array_equal(np.asarray(core_full), np.arange(len(core_full))) \
                    or np.any(local % layers != fr % layers):
                return None
            depth = {c: int(local[comp == c].max()) // layers + 1 for c in used}
            base = dict(zip(used, np.cumsum([0] + [depth[c] for c in used])))
            windows = tuple((int(offsets[c]), depth[c] * layers) for c in used)
            W = np.zeros((len(core_full) // layers, sum(depth.values())))
            # one weight per layer and window row: the check below finds layers that differ
            W[fr // layers, np.array([base[c] for c in comp]) + local // layers] = fv
            K = None if reduce is None else np.asarray(reduce, dtype=np.float64)
            if (K is None and len(core_full) != n_core) or (K is not None and K.shape != (n_core, len(core_full))):
                return None
            # the core entries implied by W and K
            q, j = np.nonzero(W)
            which = np.searchsorted(np.cumsum([depth[c] for c in used]), j, "right")
            m = np.arange(layers)
            full_rows = (q[:, None] * layers + m).ravel()
            full_cols = ((offsets[used][which] + (j - np.array([base[c] for c in used])[which]) * layers)[:, None]
                         + m).ravel()
            full_vals = np.repeat(W[q, j], layers)
            if K is None:
                implied.append((core[full_rows], full_cols, full_vals))
            else:
                for i, f in zip(*np.nonzero(K)):
                    hit = full_rows == f
                    implied.append((np.full(int(hit.sum()), core[i]), full_cols[hit], K[i, f] * full_vals[hit]))

        # the structure must reproduce every entry
        def key(r, c):
            return np.asarray(r, dtype=np.int64) * n_cols + np.asarray(c, dtype=np.int64)
        keys = np.concatenate([key(rows, cols)] + [key(r, c) for r, c, _ in implied])
        diff = np.concatenate([vals] + [-v for _, _, v in implied])
        _, inv = np.unique(keys, return_inverse=True)
        if np.abs(np.bincount(inv, weights=diff)).max() > 1e-13 * max(1.0, np.abs(vals).max()):
            return None

        return cls(w=None if np.all(w == 1.0) else _host(w),
                   sigma=None if sigma is None else jnp.asarray(sigma, dtype=jnp.float32 if PAIRS else float),
                   W=None if W is None else _host(W), WT=None if W is None else _host(W.T),
                   K=None if K is None else _host(K), KT=None if K is None else _host(K.T),
                   shape=(int(n_rows), int(n_cols)),
                   components=tuple((int(o), tuple(int(v) for v in s)) for o, s in zip(offsets, components)),
                   shifts=shifts, segments=segments,
                   pad=max(0, max(st + n * stride for st, stride, n, _ in segments) - int(n_cols)),
                   windows=windows, layers=layers, n_core=int(n_core), core_first=bool(core_first))

    def _reflect(self, u):
        """``R u``, the unsigned reflection of raw coefficients: reversed and rolled angular axes."""
        parts = []
        for (off, shape), shift in zip(self.components, self.shifts):
            n = math.prod(shape)
            parts.append(_map(u[off:off + n], lambda v: jnp.roll(jnp.flip(v.reshape(shape), (1, 2)), shift,
                                                                 (1, 2)).ravel()))
        return ds.concatenate(parts)

    def _select(self, u):
        """``S u``: the lead functions, segment by segment."""
        if self.pad:
            u = ds.pad(u, ((0, self.pad),))
        out = []
        for start, stride, count, pieces in self.segments:
            block = u[start:start + count * stride].reshape(count, stride)
            out.append(ds.concatenate([block[:, o:o + n] for o, n in pieces], axis=1).ravel())
        return ds.concatenate(out)

    def _place(self, v):
        """``S^T v``: the bulk values put back at their lead functions, zeros elsewhere."""
        out, pos, at = [], 0, 0
        for start, stride, count, pieces in self.segments:
            width = sum(n for _, n in pieces)
            block = v[pos:pos + count * width].reshape(count, width)
            pos += count * width
            cols, c0, end = [], 0, 0
            for o, n in pieces:
                if o > end:
                    cols.append(_zeros(v, (count, o - end)))
                cols.append(block[:, c0:c0 + n])
                c0, end = c0 + n, o + n
            if stride > end:
                cols.append(_zeros(v, (count, stride - end)))
            if start > at:
                out.append(_zeros(v, (start - at,)))
            out.append(ds.concatenate(cols, axis=1).ravel()[:(count - 1) * stride + end])
            at = start + (count - 1) * stride + end
        if self.shape[1] > at:
            out.append(_zeros(v, (self.shape[1] - at,)))
        return ds.concatenate(out)

    def forward(self, x):
        """``E x`` for a vector ``x`` of raw coefficients."""
        v = self._select(x)
        if self.sigma is not None:
            v = v + _map(self._select(self._reflect(x)), lambda u: self.sigma * u)
        if self.w is not None:
            v = v * _part(self.w, v)
        parts = [v]
        if self.n_core:
            X = ds.concatenate([x[o:o + n].reshape(n // self.layers, self.layers) for o, n in self.windows])
            c = _dense(_part(self.W, x), X).ravel()
            if self.K is not None:
                c = ds.matvec(_part(self.K, x), c)
            parts.insert(0 if self.core_first else 1, c)
        return ds.concatenate(parts)

    def transpose(self, y):
        """``E^T y`` for a vector ``y`` of extracted DoFs."""
        n_bulk = self.shape[0] - self.n_core
        if self.core_first:
            y_core, y_bulk = y[:self.n_core], y[self.n_core:]
        else:
            y_bulk, y_core = y[:n_bulk], y[n_bulk:]
        if self.w is not None:
            y_bulk = y_bulk * _part(self.w, y_bulk)
        x = self._place(y_bulk)
        if self.sigma is not None:
            x = x + self._reflect(self._place(_map(y_bulk, lambda u: self.sigma * u)))
        if not self.n_core:
            return x
        if self.K is not None:
            y_core = ds.matvec(_part(self.KT, y_core), y_core)
        Xw = _dense(_part(self.WT, y_core), y_core.reshape(-1, self.layers))
        parts, at, row = [], 0, 0
        for o, n in self.windows:
            if o > at:
                parts.append(x[at:o])
            parts.append(x[o:o + n] + Xw[row:row + n // self.layers].ravel())
            at, row = o + n, row + n // self.layers
        parts.append(x[at:])
        return ds.concatenate(parts)


def _map(x, fn):
    """``fn``, a linear map made of reshapes, slices and reversals, applied to a float array or to both
    parts of a :class:`~mrx.double_single.DS`."""
    return ds.DS(fn(x.hi), fn(x.lo)) if isinstance(x, ds.DS) else fn(x)


def _segments(lead, max_period=64):
    """The increasing indices ``lead`` as segments ``(start, stride, count, pieces)`` of
    :class:`StructuredExtraction`. The runs of consecutive indices are grouped greedily into the period
    of runs that repeats the most often with a fixed stride."""
    brk = np.flatnonzero(np.diff(lead) != 1) + 1
    starts = lead[np.concatenate([[0], brk])].astype(np.int64)
    lens = np.diff(np.concatenate([[0], brk, [lead.size]]))
    segments, k = [], 0
    while k < starts.size:
        best_n, best_p = 1, 1
        for p in range(1, min(max_period, (starts.size - k) // 2) + 1):
            stride = starts[k + p] - starts[k]
            n = 1
            while k + (n + 1) * p <= starts.size \
                    and np.array_equal(lens[k + n * p:k + (n + 1) * p], lens[k:k + p]) \
                    and np.all(starts[k + n * p:k + (n + 1) * p] - starts[k:k + p] == n * stride):
                n += 1
            if n >= 2 and n * p > best_n * best_p:
                best_n, best_p = n, p
        p, n = best_p, best_n
        end = starts[k + p - 1] + lens[k + p - 1]
        stride = int(starts[k + p] - starts[k]) if n > 1 else int(end - starts[k])
        pieces = tuple((int(s - starts[k]), int(m)) for s, m in zip(starts[k:k + p], lens[k:k + p]))
        segments.append((int(starts[k]), stride, int(n), pieces))
        k += n * p
    return tuple(segments)


@eqx.filter_jit
def _apply_structured(s, x, transposed):
    """``E x`` or ``E^T x`` with the :class:`StructuredExtraction` ``s``, for a vector or each column of a matrix."""
    fn = s.transpose if transposed else s.forward
    if x.ndim == 1:
        return fn(x)
    return jax.vmap(fn, in_axes=1, out_axes=1)(x)


def summed_coo(rows, cols, vals, shape):
    """A :class:`MatrixFreeExtraction` of ``shape`` from the given entries, with duplicate
    entries summed into one and the entries sorted by row."""
    key = np.asarray(rows, dtype=np.int64) * shape[1] + np.asarray(cols, dtype=np.int64)
    uniq, inv = np.unique(key, return_inverse=True)
    return MatrixFreeExtraction.from_coo(uniq // shape[1], uniq % shape[1],
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
    gram_inverse = core_gram_inverse(e, core)
    gram_inverse = ds.from_host(gram_inverse) if isinstance(a, ds.DS) else jnp.asarray(gram_inverse)
    return ds.set_at(a, core, ds.matvec(gram_inverse, a[core]))


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


def build_extraction(form, xi, dirichlet):
    """The extraction ``E`` of the k-form space ``form`` and the indices of its polar core rows.

    The first rows of ``E`` are the axis functions, built with the polar weights ``xi`` of
    :func:`get_xi`. These are the polar core rows. Every other row selects one raw basis
    function: per component, all radial rings from ring 2 outwards (from ring 1 for a component
    with the derivative basis in ``r``). With ``dirichlet`` the outermost ring of the primal
    radial basis is dropped, which imposes homogeneous Dirichlet conditions at ``r = 1``.

    Returns ``(E, core_rows)``, ``E`` a :class:`MatrixFreeExtraction` of shape
    ``(n_extracted, form.n)``.
    """
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

    e = MatrixFreeExtraction.from_coo(np.concatenate(rows), np.concatenate(cols), np.concatenate(data),
                                      (row0, form.n))
    core = np.unique(np.asarray(core, dtype=np.int32))
    return dataclasses.replace(e, structured=StructuredExtraction.build(e, form.shape, core)), core


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
    # on the host in float64: the identities of the polar stencils hold to the round-off of these weights
    theta_js = ((np.arange(nt) - (p - 1) / 2.0) / nt) * 2 * np.pi
    dR, dY = np.cos(theta_js), np.sin(theta_js)

    s3 = np.sqrt(3.0)
    tau = max(np.max(-2.0 * dR), np.max(dR - s3 * dY), np.max(dR + s3 * dY))
    xi1 = np.stack([1/3 + 2.0 * dR / (3.0 * tau),
                    1/3 - dR / (3.0 * tau) + s3 * dY / (3.0 * tau),
                    1/3 - dR / (3.0 * tau) - s3 * dY / (3.0 * tau)])  # (3, n_theta)
    xi0 = np.full((3, nt), 1.0 / 3.0)
    return np.stack([xi0, xi1], axis=1)
