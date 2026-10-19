"""The zeta-slab preconditioner of the 1- and 2-form masses ``M_1`` and ``M_2``.

The metric-lumping mass atom (:class:`mrx.metric_lumping.MetricLumpingMass`) treats each vector
component separately with separable metric weights. On shaped cross-sections it loses the coupling
between the radial and the poloidal components (the ``g^{r theta}`` terms of the metric), and its
condition number on ``M_1`` is about 50. :class:`SlabMass` keeps that coupling.

For every toroidal spline index (a "zeta slab") it takes the exact block of ``M_k`` that couples
the degrees of freedom of that slab, and factors it. Neighbouring slabs are glued by the 1-D toroidal
mass. With ``L_j L_j^T`` the exact block of slab ``j`` and ``T`` the 1-D toroidal mass scaled to unit
diagonal, the model is ``M[j, l] ~ L_j T[j, l] L_l^T``, and its inverse is
``L^-T (T^-1 (x) I) L^-1``. The diagonal blocks of the model are exact. Only the coupling between
slabs is approximated.

Inside a slab the components fall into two groups that the toroidal mass glues with one 1-D matrix
each: the radial and poloidal components (same spline kind in zeta), and the toroidal component.
The coupling between the two groups is dropped, as the atom drops all coupling between components.
Ordered by radial position, the slab blocks are block tridiagonal in blocks of ``p`` rings, so the
factor is a block Cholesky factor, and an apply is one forward and one backward sweep over the
radial blocks, all slabs at once. The steps of a sweep are sequential, and on a GPU each step costs
a few microseconds whatever it computes. So each step is a single product: the factor is stored as
rows that join the inverse diagonal block with its coupling to the previous block.

The model is built by probing the assembled mass with colored unit vectors, once per geometry, like
the axis block of the atoms. The dense factor stores about ``7 n_theta`` numbers per degree of freedom
(``n^4`` in all, 262M numbers on li383 (64,80,32)). Its entries decay fast in theta, so the factor is
cut to a theta band of half-width :data:`THETA_HALF_WIDTH` and stored row by row with the kept
columns: a fixed number per degree of freedom (``n^3``, about 270 on li383 (48,64,24)), with the condition
number of the full factor.
It preconditions the mass solves of ``k = 1, 2`` by default (``mass_atom="slab"`` in
:meth:`mrx.derham_sequence.DeRhamSequence.build_preconditioners`). The atom stays the norm of the solves, every
``M^{-1}`` inside an iteration and ``seq.M[k].precondition``, whose users need a symmetric apply. The scalar
masses keep the atom, which is already close to exact there.
"""

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

from mrx.extraction_operators import ring_layout
from mrx.metric_lumping import PROBE_BATCH, _kron_mass_model_1d, _parity_split
from mrx.pytree import register_arrays

#: Storage precision of the slab factors. The vectors and the solves
#: stay in float64, the sweeps run in this precision. With float32 the condition number is unchanged (li383
#: (48,64,24): 3.48 both), the apply reads half the data and the M_1 solve is 4-16 % faster than with float64.
FACTOR_DTYPE = jnp.float32

#: Half-width of the theta band the factor is cut to: entries of a factor block whose two degrees of freedom are more
#: than this many theta indices apart (periodic) are dropped, and the blocks are stored row by row with their kept
#: columns. Rows and columns of the axis keep all their entries. The factor of the mass decays fast in theta: at 4 the
#: condition number is that of the full factor (li383 L: 3.43 against 3.39). The sweep tables of the banded factor hold
#: about 270 numbers per degree of freedom, the dense factor about 7 n_theta. ``None`` keeps the dense factor.
THETA_HALF_WIDTH = 4

#: Component groups of the slab model, glued by one 1-D toroidal mass each: the radial and the
#: poloidal component share their spline kind in zeta, the toroidal component has the other.
GROUPS = ((0, 1), (2,))


def _layout(E, shapes, rows, records, groups=GROUPS):
    """The degrees of freedom of each group as index tables ``(n_zeta, n_blocks, b)`` into the
    extracted vector (padding = its length ``n``), and the block sizes. Block ``s`` holds the rings
    ``rows s .. rows s + rows - 1``. Every degree of freedom lies on one zeta index, and every slab
    has the same structure (the extraction acts within a zeta layer), which is checked. ``records``
    is the :func:`~mrx.extraction_operators.ring_layout` of ``E``. The second return value holds per
    degree of freedom its theta position on the raw theta grid, its ring's theta count and whether it
    is a polar axis row, and the raw theta count."""
    r, c, _ = E.entries()
    n = E.forward_shape[0]
    offsets = np.cumsum([0] + [int(np.prod(s)) for s in shapes])
    comp = np.searchsorted(offsets, c, side="right") - 1
    local = c - offsets[comp]
    ir, it, iz = (np.zeros_like(c) for _ in range(3))
    for cc in range(len(shapes)):
        m = comp == cc
        ir[m], it[m], iz[m] = np.unravel_index(local[m], shapes[cc])
    # per degree of freedom: theta position on the raw grid and theta count of its ring, the axis rows
    nt = int(shapes[0][1])
    theta, ntheta = np.zeros(n + 1, int), np.full(n + 1, nt)
    theta[n] = -1
    axis = np.concatenate([np.ones(n, bool), [False]])
    ring_key = np.full(n, -1)
    for cc, a, m, _, first, nz in records:
        span = first + np.arange(m * nz)
        theta[span] = np.repeat(np.arange(m) * (nt // m), nz)
        ntheta[span], axis[span] = m, False
        ring_key[span] = cc * 10 ** 6 + a * 10 ** 3 + theta[span]
    tables = []
    for group in groups:
        sel = np.isin(comp, group)
        dofs = np.unique(r[sel])
        if np.intersect1d(dofs, np.unique(r[~sel])).size:
            raise ValueError("a degree of freedom combines components of both slab groups")
        zmin, zmax, rmax = np.full(n, 10 ** 9), np.full(n, -1), np.full(n, -1)
        np.minimum.at(zmin, r[sel], iz[sel])
        np.maximum.at(zmax, r[sel], iz[sel])
        np.maximum.at(rmax, r[sel], ir[sel])
        if np.any(zmin[dofs] != zmax[dofs]):
            raise ValueError("a degree of freedom spans several zeta indices")
        key = np.full(n, 10 ** 15)
        np.minimum.at(key, r[sel], comp[sel] * 10 ** 6 + ir[sel] * 10 ** 3 + it[sel])
        key = np.where(ring_key >= 0, ring_key, key)       # a graded row: its ring and theta position
        block = rmax // rows
        nz, ns = int(zmax[dofs].max()) + 1, int(block[dofs].max()) + 1
        per_slab = []
        for j in range(nz):
            dj = dofs[zmin[dofs] == j]
            per_slab.append(dj[np.lexsort((key[dj], block[dj]))])
        sizes = np.bincount(block[per_slab[0]], minlength=ns)
        for dj in per_slab[1:]:
            if not (np.array_equal(np.bincount(block[dj], minlength=ns), sizes)
                    and np.array_equal(key[dj], key[per_slab[0]])):
                raise ValueError("the zeta slabs differ in structure")
        b = int(sizes.max())
        start = np.concatenate([[0], np.cumsum(sizes)])
        idx = np.full((nz, ns, b), n)
        for j, dj in enumerate(per_slab):
            for s in range(ns):
                idx[j, s, :sizes[s]] = dj[start[s]:start[s + 1]]
        tables.append((idx, sizes))
    return tables, (theta, ntheta, axis, nt)


def _probe(apply, idx, sizes, n):
    """The diagonal blocks ``D[j, s]`` and the sub-diagonal blocks ``Lo[j, s]`` (block ``s`` against
    ``s - 1``) of every slab, from colored unit vectors: slabs ``ncz >= 5`` apart and blocks 5 apart
    in one probe, one column position at a time. Raises if the slab blocks are not block tridiagonal."""
    nz, ns, b = idx.shape
    ncz = next(c for c in range(5, nz + 1) if nz % c == 0)
    idx_p = jnp.asarray(np.concatenate([idx, np.full((nz, 1, b), n)], axis=1))     # block ns: padding only
    nsel = -(-ns // 5)

    @jax.jit
    def one(D, Lo, js, ss):
        cols = idx_p[js][:, ss]

        def column(t):
            x = jnp.zeros(n + 1).at[cols[:, :, t]].set(1.0).at[n].set(0.0)
            return jnp.concatenate([apply(x[:n]), jnp.zeros(1)])[idx_p[js]]
        Y = jnp.transpose(jax.lax.map(column, jnp.arange(b), batch_size=PROBE_BATCH), (1, 2, 3, 0))
        jj = js[:, None]
        D = D.at[jj, ss[None, :]].set(Y[:, ss], mode="drop")
        below = jnp.where(ss + 1 < ns, ss + 1, ns + 5)
        Lo = Lo.at[jj, below[None, :]].set(Y[:, jnp.minimum(ss + 1, ns)], mode="drop")
        two = jnp.where((ss + 2 < ns)[None, :, None, None], Y[:, jnp.minimum(ss + 2, ns)], 0.0)
        return D, Lo, jnp.max(jnp.abs(two))

    D = jnp.zeros((nz, ns, b, b))
    Lo = jnp.zeros((nz, ns, b, b))
    far = 0.0
    for cz in range(ncz):
        js = jnp.asarray(np.arange(cz, nz, ncz))
        for cs in range(5):
            ss = np.arange(cs, ns, 5)
            ss = jnp.asarray(np.concatenate([ss, np.full(nsel - ss.size, ns)]))      # pad with the padding block
            D, Lo, f = one(D, Lo, js, ss)
            far = max(far, float(f))
    scale = float(jnp.max(jnp.abs(D)))
    if far > 1e-12 * scale:
        raise ValueError(f"the slab blocks are not block tridiagonal ({far / scale:.1e}): too few rings per block")
    pad = jnp.asarray(np.arange(b)[None, :] >= sizes[:, None])                       # (ns, b)
    D = 0.5 * (D + jnp.swapaxes(D, -1, -2))
    D = D + jnp.where(pad[None, :, :, None] & pad[None, :, None, :], jnp.eye(b), 0.0) * scale
    return D, Lo


@jax.jit
def _factor(D, Lo):
    """Block Cholesky factor of the block-tridiagonal slab matrices, batched over the slabs:
    ``Linv[:, s] = L_ss^-1`` and ``B[:, s] = L_{s,s-1}``."""
    nz, ns, b, _ = D.shape

    def body(previous, s):
        Lss = jnp.linalg.cholesky(D[:, s] - previous @ jnp.swapaxes(previous, -1, -2))
        Linv = jax.scipy.linalg.solve_triangular(Lss, jnp.broadcast_to(jnp.eye(b), Lss.shape), lower=True)
        following = Lo[:, jnp.minimum(s + 1, ns - 1)] @ jnp.swapaxes(Linv, -1, -2)
        return following, (Linv, previous)
    _, (Linv, B) = jax.lax.scan(body, jnp.zeros((nz, b, b)), jnp.arange(ns))
    return jnp.swapaxes(Linv, 0, 1), jnp.swapaxes(B, 0, 1)


def _rows(Linv, B):
    """The forward rows ``W[:, s] = [L_ss^-1 | -L_ss^-1 L_{s,s-1}]`` of the factor ``(Linv, B)`` of :func:`_factor`, as
    ``(n_slabs, n_blocks, b, 2 b)``. Block ``s`` of ``L^-1 x`` is ``W[:, s] (x_s, y_{s-1})``, one product per step."""
    return jnp.concatenate([Linv, -(Linv @ B)], -1)


def _forward(W, x):
    """``L^-1 x`` with the rows ``W`` of :func:`_rows` stored block-major, ``(n_blocks, n_slabs, b, 2 b)``, and ``x`` of
    shape ``(n_blocks, n_slabs, b)``."""
    def step(y, args):
        Ws, xs = args
        y = jnp.einsum("jab,jb->ja", Ws, jnp.concatenate([xs, y], 1))
        return y, y
    return jax.lax.scan(step, jnp.zeros_like(x[0]), (W, x), unroll=True)[1]


def _backward(W, y):
    """``L^-T y``, as :func:`_forward`. With ``q_s = y_s - C_{s+1}^T q_{s+1}`` (``C_s = L_ss^-1 L_{s,s-1}``) the result
    is ``z_s = L_ss^-T q_s``, and ``W[s]^T q_s = (z_s, q_{s-1} - y_{s-1})`` gives both in one product."""
    b = y.shape[2]
    before = jnp.concatenate([jnp.zeros_like(y[:1]), y[:-1]])

    def step(q, args):
        Ws, ys = args
        out = jnp.einsum("jab,ja->jb", Ws, q)
        return ys + out[:, b:], out[:, :b]
    return jax.lax.scan(step, y[-1], (W, before), reverse=True, unroll=True)[1]


def _sweep(W, Tinv, x):
    """``L^-T (T^-1 (x) I) L^-1 x``, as :func:`_forward`."""
    return _backward(W, jnp.einsum("jk,skb->sjb", Tinv, _forward(W, x)))


def _band_mask(rows_idx, cols_idx, info, w, lower):
    """Per block row ``s`` the entries of a factor block of one group kept by the theta band of half-width ``w``,
    ``(n_blocks, b, b)``, for the degrees of freedom ``rows_idx[s]`` against ``cols_idx[s]``."""
    theta, ntheta, axis, nt = info
    masks = []
    for ri, ci in zip(rows_idx, cols_idx):
        # periodic distance on the raw theta grid, at most w cells of the coarser of the two rings
        d = np.abs(theta[ri][:, None] - theta[ci][None, :])
        d = np.minimum(d, nt - d)
        coarse = np.minimum(ntheta[ri][:, None], ntheta[ci][None, :])
        keep = (d * coarse <= w * nt) | axis[ri][:, None] | axis[ci][None, :]
        keep &= (theta[ri] >= 0)[:, None] & (theta[ci] >= 0)[None, :]
        masks.append(np.tril(keep) if lower else keep)
    return np.stack(masks)


def _ell(A, mask):
    """ELL rows of the blocks ``A`` (n_slabs, n_blocks, R, m) on the mask (n_groups, n_blocks, R, m) of each group of
    consecutive slabs, one pair per block: the column table ``(n_groups, R, K)`` shared by the slabs of a group, with
    the zero slot ``m``, and the values ``(n_slabs, R, K)``. ``K`` is the widest row of the block, so blocks with
    narrow rows (near the axis, or on coarse rings of a graded space) store and read less."""
    S, G = A.shape[0], mask.shape[0]
    blocks = []
    for s in range(A.shape[1]):
        m = mask[:, s]
        order = np.argsort(~m, axis=-1, kind="stable")[..., :max(1, int(m.sum(-1).max()))]
        valid = np.take_along_axis(m, order, -1)
        vals = jnp.take_along_axis(A[:, s], jnp.asarray(np.repeat(order.astype(np.int32), S // G, 0)), -1)
        blocks.append((jnp.asarray(np.where(valid, order, A.shape[-1]).astype(np.int32)),
                       vals * np.repeat(valid, S // G, 0)))
    return tuple(blocks)


def _gather(c, V, v):
    """Per slab ``j``: ``block_j @ v_j`` with ELL blocks ``c`` (n_groups, R, K), ``V`` (S, R, K) and ``v`` (S, m)."""
    S = v.shape[0]
    vp = jnp.concatenate([v, jnp.zeros((S, 1), v.dtype)], 1)
    c = jnp.repeat(c, S // c.shape[0], axis=0)
    return jnp.sum(V * jnp.take_along_axis(vp, c.reshape(S, -1), axis=1).reshape(V.shape), -1)


def _scatter(c, V, v, m):
    """Per slab ``j``: ``block_j^T @ v_j`` for ELL blocks with ``m`` columns."""
    S = v.shape[0]
    c = jnp.repeat(c, S // c.shape[0], axis=0)
    return jnp.zeros((S, m + 1), v.dtype).at[jnp.arange(S)[:, None, None], c].add(V * v[:, :, None])[:, :m]


def _banded(Linv, B, rows, n, info, w, dtype):
    """The sweep tables of the factor ``(Linv, B)`` of :func:`_factor` cut to the theta band of half-width ``w``.

    ``Linv`` and ``B`` hold the slabs of all groups, ``(n_slabs, n_blocks, b, b)``, ``rows`` the index tables of the
    groups ``(n_groups, n_blocks, b)`` with padding ``n``, and ``info`` the theta positions of :func:`_layout`. The
    tables hold the forward rows ``[Linv_s | -C_s]`` with ``C_s = Linv_s B_s``, the backward rows
    ``[Linv_s^T | -E_s^T]`` of the blocks ``s >= 1`` with ``E_s = B_{s+1} Linv_s``, so that
    ``z_s = Linv_s^T y_s - E_s^T z_{s+1}``, and the rows of ``Linv_0`` and ``-E_0`` stacked for block 0, applied by a
    scatter because its axis columns are dense. All products are formed from the banded factors, so the sweeps
    apply the banded factor exactly."""
    S, G = Linv.shape[0], rows.shape[0]
    prev = np.concatenate([np.full_like(rows[:, :1], n), rows[:, :-1]], 1)
    mL = np.stack([_band_mask(r, r, info, w, True) for r in rows])
    mB = np.stack([_band_mask(r, p, info, w, False) for r, p in zip(rows, prev)])
    Linv = Linv * np.repeat(mL, S // G, 0)
    B = B * np.repeat(mB, S // G, 0)
    C, mC = Linv @ B, mL.astype(np.float32) @ mB.astype(np.float32) > 0
    E = jnp.concatenate([B[:, 1:] @ Linv[:, :-1], jnp.zeros_like(Linv[:, :1])], 1)
    mE = np.concatenate([mB[:, 1:].astype(np.float32) @ mL[:, :-1].astype(np.float32) > 0, np.zeros_like(mL[:, :1])], 1)
    forward = _ell(jnp.concatenate([Linv, -C], -1), np.concatenate([mL, mC], -1))
    backward = _ell(jnp.concatenate([jnp.swapaxes(Linv, -1, -2), -jnp.swapaxes(E, -1, -2)], -1)[:, 1:],
                    np.concatenate([np.swapaxes(mL, -1, -2), np.swapaxes(mE, -1, -2)], -1)[:, 1:])
    axis = _ell(jnp.concatenate([Linv[:, :1], -E[:, :1]], -2), np.concatenate([mL[:, :1], mE[:, :1]], -2))
    return tuple(tuple((c, V.astype(dtype)) for c, V in t) for t in (forward, backward, axis))


def _band_forward(tables, x):
    """:func:`_forward` with the tables of :func:`_banded`, one block per step (unrolled)."""
    y, out = jnp.zeros_like(x[0]), []
    for (c, V), xs in zip(tables[0], x):
        y = _gather(c, V, jnp.concatenate([xs, y], 1))
        out.append(y)
    return jnp.stack(out)


def _band_backward(tables, y):
    """:func:`_backward` with the tables of :func:`_banded`."""
    z, out = jnp.zeros_like(y[0]), []
    for (c, V), ys in zip(tables[1][::-1], y[:0:-1]):
        z = _gather(c, V, jnp.concatenate([ys, z], 1))
        out.append(z)
    (c, V), = tables[2]
    out.append(_scatter(c, V, jnp.concatenate([y[0], z], 1), y.shape[2]))
    return jnp.stack(out[::-1])


def _band_sweep(tables, Tinv, x):
    """:func:`_sweep` with the tables of :func:`_banded`."""
    return _band_backward(tables, jnp.einsum("jk,skb->sjb", Tinv, _band_forward(tables, x)))


@register_arrays
class SlabMass:
    """Approximate inverse of the mass matrix ``M_k``, ``k = 1, 2``, on the space of ``seq``, by the
    zeta-slab model (see the module docstring).

    Built by ``DeRhamSequence.build_preconditioners()`` (``mass_atom="slab"``, the default) for the installed
    geometry.
    Apply it with :meth:`apply`. Needs at least ``p`` rings per radial block (it uses exactly ``p``).
    """

    def __init__(self, seq, operators, k):
        if k not in (1, 2):
            raise ValueError(f"the slab mass model is for k = 1, 2, not {k}")
        self.parity_projector = seq.free_projector(k)
        E = seq.extraction[(k, seq.dirichlet)]
        n = E.forward_shape[0]
        self._n = n
        shapes = [tuple(int(v) for v in s) for s in getattr(seq, f"basis_{k}").shape]
        split = _parity_split(seq, k)
        mass = seq.M[k]

        def apply_mass(x):
            # exact only per parity on a half-period sequence, so the probe sums the two parts
            return mass @ x if split is None else sum(mass @ part for part in split(x))
        mass_1d = _kron_mass_model_1d(seq, k)
        tables, info = _layout(E, shapes, seq.p, ring_layout(getattr(seq, f"basis_{k}"), seq.dirichlet,
                                                              seq.theta_counts))
        factors = []
        for (idx, sizes), group in zip(tables, GROUPS):
            D, Lo = _probe(apply_mass, idx, sizes, n)
            Linv, B = _factor(D, Lo)
            mz = np.asarray(mass_1d[group[0]][2])
            d = np.sqrt(np.diag(mz))
            factors.append((idx, Linv, B, np.linalg.inv(mz / np.outer(d, d))))
        # one sweep for both groups: the second group padded to the block count and size of the first. On the free
        # 2-forms the first group reaches one ring further out (the outer ring of the radial component), a block of
        # its own when (n_r - 1) % p == 0.
        nz, ns, b = factors[0][0].shape
        tinv = np.zeros((2 * nz, 2 * nz))
        tinv[:nz, :nz], tinv[nz:, nz:] = factors[0][3], factors[1][3]
        self._tinv = jnp.asarray(tinv, FACTOR_DTYPE)
        idx_all = [np.pad(idx, ((0, 0), (0, ns - idx.shape[1]), (0, b - idx.shape[2])), constant_values=n)
                   for idx, *_ in factors]
        # block-major storage, so that each sweep step reads one contiguous block
        self._idx = jnp.asarray(np.swapaxes(np.concatenate(idx_all), 0, 1))
        Linv = jnp.concatenate([jnp.broadcast_to(jnp.eye(b), (nz, ns, b, b))
                                .at[:, :idx.shape[1], :idx.shape[2], :idx.shape[2]].set(L) for idx, L, *_ in factors])
        B = jnp.concatenate([jnp.zeros((nz, ns, b, b)).at[:, :idx.shape[1], :idx.shape[2], :idx.shape[2]].set(Bg)
                             for idx, _, Bg, _ in factors])
        if THETA_HALF_WIDTH is None:
            self._w = jnp.swapaxes(_rows(Linv, B), 0, 1).astype(FACTOR_DTYPE)
            self._band = None
        else:
            self._w = None
            self._band = _banded(Linv, B, np.stack([idx[0] for idx in idx_all]), n, info, THETA_HALF_WIDTH,
                                 FACTOR_DTYPE)

    def _apply_full(self, x):
        n = x.shape[0]
        xs = jnp.concatenate([x, jnp.zeros(1)])[self._idx].astype(self._tinv.dtype)
        if self._band is None:
            y = _sweep(self._w, self._tinv, xs)
        else:
            y = _band_sweep(self._band, self._tinv, xs)
        return jnp.zeros(n + 1).at[self._idx].add(y.astype(x.dtype))[:n]

    def apply(self, x):
        """Apply the preconditioner to a coefficient vector of the free degrees of freedom."""
        if self.parity_projector is None:
            return self._apply_full(jnp.asarray(x))
        return self.parity_projector(self._apply_full, jnp.asarray(x))

    def reduced(self, X):
        """The payload of ``X^T P X`` for a reduced view with the expansion ``X``: ``X`` itself."""
        return X

    def apply_reduced(self, X, x):
        """``X^T P X x`` for the payload ``X`` of :meth:`reduced`."""
        return X.T @ self._apply_full(X @ jnp.asarray(x))

    def stored(self):
        """The number of stored floating-point numbers (the column tables of the band not counted)."""
        if self._band is None:
            return int(self._w.size + self._tinv.size)
        values = [a for a in jax.tree_util.tree_leaves(self._band) if jnp.issubdtype(a.dtype, jnp.floating)]
        return int(sum(a.size for a in values) + self._tinv.size)
