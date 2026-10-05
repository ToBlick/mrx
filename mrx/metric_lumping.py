"""Approximate inverses of the mass matrices ``M_k`` and the Hodge Laplacians ``L_k``.

These are the preconditioners of every mass and Laplacian solve in MRX. There is one of each
per form degree ``k`` and space (the Dirichlet space of a sequence and its free view):

- :class:`MetricLumpingMass` approximates ``M_k^{-1}``. It also stands in for ``M_{k-1}^{-1}``
  inside the approximate Laplacian the solvers use, so it is part of that operator as well.
- :class:`MetricLumpingLaplacian` approximates ``L_k^{-1}``. Its
  :meth:`~MetricLumpingLaplacian.shifted_stiffness_apply` approximates ``(M_k + eps S_k)^{-1}``,
  where ``S_k = d_k^T M_{k+1} d_k`` is the stiffness, the half of ``L_k`` built from the
  exterior derivative ``d_k`` of ``k``-forms.
- :class:`ReducedAtom` is either of them restricted to the even or odd part of a
  stellarator-symmetric (half-period) sequence.

You do not build them yourself: ``DeRhamSequence.build_preconditioners`` builds all of them for
the installed geometry. They depend on the metric, so call it again after every new map. The
build probes the operator once per degree of freedom at the magnetic axis and is done once per
geometry, never inside a solve.

The name describes the approximation. Away from the axis the metric coefficients are replaced by
their averages along each coordinate ("lumped"), which turns each vector component of ``M_k`` into a
Kronecker product and of ``L_k`` into a Kronecker sum of 1-D matrices, both cheap to invert
exactly. The few degrees of freedom at the axis form a small block that is inverted densely.
Couplings between the two parts, between vector components and through off-diagonal metric
entries are neglected.
"""

from __future__ import annotations

import numpy as np

import equinox as eqx
import jax
import jax.numpy as jnp

from mrx.differential_forms import DERIV_AXES
from mrx.mass import build_mass_diagonal
from mrx.precision import DTYPE, RESIDUAL_DTYPE, sqrt_eps
from mrx.pytree import register_arrays
from mrx.symmetry import mirror_component, mirror_zeta_1d

#: Relative eigenvalue cut-off of the dense inverses of the axis blocks (in the residual precision).
CORE_TOL = 4096.0 * float(jnp.finfo(RESIDUAL_DTYPE).eps)

#: Axis rows probed per batch when the dense axis block is built (bounds the memory of the batched applies).
PROBE_BATCH = 64

#: Empirical factor on the boundary term of the Laplacian preconditioner under natural boundary
#: conditions. Three times the exact surface integral gave the smallest condition number.
PRODUCTION_BC_SCALE = 3.0


# --------------------------------------------------------------------------- #
# 1-D building blocks                                                          #
# --------------------------------------------------------------------------- #

def _symmetrize(matrix):
    return 0.5 * (matrix + matrix.T)


def _assemble_weighted_1d_mass(B, weights):
    return (B * weights[None, :]) @ B.T


def _simultaneous_diagonalize_pair(M, A):
    """``(V, lam)`` with ``V^T M V = I`` and ``V^T A V = diag(lam)`` for SPD ``M``, symmetric ``A``."""
    M_sym = _symmetrize(jnp.asarray(M, dtype=DTYPE))
    A_sym = _symmetrize(jnp.asarray(A, dtype=DTYPE))
    L = jnp.linalg.cholesky(M_sym)
    Linv_A = jax.scipy.linalg.solve_triangular(L, A_sym, lower=True)
    B = jax.scipy.linalg.solve_triangular(L, Linv_A.T, lower=True).T
    lam, U = jnp.linalg.eigh(_symmetrize(B))
    V = jax.scipy.linalg.solve_triangular(L.T, U, lower=False)
    return V, lam


def _dense_incidence_1d(n0, typ):
    """The 1-D incidence ``(G c)_j = c_{j+1} - c_j`` of an axis of type ``typ`` (wrapped if
    periodic, zero if constant)."""
    if typ == 'clamped':
        j = jnp.arange(n0 - 1)
        return jnp.zeros((n0 - 1, n0), dtype=DTYPE).at[j, j].set(-1.0).at[j, j + 1].set(1.0)
    if typ == 'periodic':
        j = jnp.arange(n0)
        return jnp.zeros((n0, n0), dtype=DTYPE).at[j, j].set(-1.0).at[j, (j + 1) % n0].set(1.0)
    return jnp.zeros((n0, n0), dtype=DTYPE)


# --------------------------------------------------------------------------- #
# Metric weights and 1-D factors                                               #
# --------------------------------------------------------------------------- #

def bundled_axis_profiles(seq, field):
    """The averages ``(p_r, p_t, p_z)`` of a field on the quadrature grid, each a function of one
    coordinate averaged over the other two.

    The radial profile leaves out the element next to the axis, which the dense axis block covers.
    """
    xi1 = jnp.asarray(seq.basis_0.Lambda[0].T)[seq.p + 1]
    wx = seq.quad.w_x * (jnp.asarray(seq.quad.x_x) >= xi1)
    wy, wz = seq.quad.w_y, seq.quad.w_z
    sx, sy, sz = jnp.sum(wx), jnp.sum(wy), jnp.sum(wz)
    pr = jnp.einsum('qrs,r,s->q', field, wy, wz) / (sy * sz)
    pt = jnp.einsum('qrs,q,s->r', field, wx, wz) / (sx * sz)
    pz = jnp.einsum('qrs,q,r->s', field, wx, wy) / (sx * sy)
    return pr, pt, pz


def weight_fields(seq):
    """The Jacobian ``J`` and the diagonal entries of the metric and its inverse at the quadrature
    points, as ``{"jac": J, "ginv_aa": (g^{rr}, g^{tt}, g^{zz}), "met_aa": (g_{rr}, g_{tt}, g_{zz})}``."""
    shape = seq.quad.shape
    jac = jnp.asarray(seq.geometry.jacobian_j).reshape(shape)
    ginv = jnp.asarray(seq.geometry.metric_inv_jkl).reshape(*shape, 3, 3)
    met = jnp.asarray(seq.geometry.metric_jkl).reshape(*shape, 3, 3)
    return {
        "jac": jac,
        "ginv_aa": tuple(ginv[..., a, a] for a in range(3)),
        "met_aa": tuple(met[..., a, a] for a in range(3)),
    }


def _axis_bases(seq):
    """The per-axis spline and derivative-spline tables and the 1-D quadrature weights."""
    primal = (seq.basis_r_jk, seq.basis_t_jk, seq.basis_z_jk)
    deriv = (seq.d_basis_r_jk, seq.d_basis_t_jk, seq.d_basis_z_jk)
    quad_w = (seq.quad.w_x, seq.quad.w_y, seq.quad.w_z)
    return primal, deriv, quad_w


def _fd_stiffness_degree0(seq, axis, profile):
    """The 1-D stiffness on a derivative axis at ``p = 1``, where the derivative splines are
    piecewise constant and have no derivative of their own. It is the finite-volume jump form
    ``sum_f t_f (u_{i+1} - u_i)^2`` with ``t_f`` the harmonic mean of ``profile`` over the two cells
    divided by the distance of their centres."""
    lam = seq.basis_0.Lambda[axis]
    nodes = np.asarray((seq.quad.x_x, seq.quad.x_y, seq.quad.x_z)[axis])
    edges = np.asarray(lam.greville_points())
    periodic = lam.type == "periodic"
    if periodic:
        edges = np.concatenate([edges, [edges[0] + 1.0]])
    h = np.diff(edges)
    n_cell = h.size

    prof = np.asarray(profile)
    idx = np.clip(np.searchsorted(edges, nodes, side="right") - 1, 0, n_cell - 1)
    w = np.array([prof[idx == i].mean() if np.any(idx == i) else prof.mean()
                  for i in range(n_cell)])

    centre = 0.5 * (edges[:-1] + edges[1:])
    if periodic:
        pairs = [(i, (i + 1) % n_cell) for i in range(n_cell)]
        dist = np.array([abs(((centre[(i + 1) % n_cell] - centre[i]) + 0.5)
                             % 1.0 - 0.5) for i in range(n_cell)])
    else:
        pairs = [(i, i + 1) for i in range(n_cell - 1)]
        dist = np.diff(centre)

    d = np.zeros((len(pairs), n_cell))
    trans = np.zeros(len(pairs))
    for f, (i, j) in enumerate(pairs):
        d[f, i], d[f, j] = -1.0, 1.0
        trans[f] = 2.0 / (1.0 / w[i] + 1.0 / w[j]) / dist[f]
    k = d.T @ (trans[:, None] * d)
    # on the D-spline coefficients: they have unit integral, value_i = u_i / h_i
    k = k / np.outer(h, h)
    return jnp.asarray(0.5 * (k + k.T), dtype=DTYPE)


def _h_last(seq):
    """Width of the last radial element."""
    uniq = np.unique(np.asarray(seq.basis_0.Lambda[0].T))
    return float(uniq[-1] - uniq[-2])


def _face_term(seq, k, c, window):
    """The boundary term of component ``c`` of ``L_k`` under natural boundary conditions, a
    rank-one addition ``alpha e e^T / h`` to the radial stiffness.

    ``e`` holds the derivative splines at ``r = 1`` and ``h`` is the width of the last element.
    ``alpha`` is the surface average of the component's mass weight times ``sqrt(g^{rr})``,
    relative to the part of that weight the diagonal scaling already carries, times
    :data:`PRODUCTION_BC_SCALE`.
    """
    fields = weight_fields(seq)
    ginv, met, jac = fields["ginv_aa"], fields["met_aa"], fields["jac"]
    wy, wz = seq.quad.w_y, seq.quad.w_z
    norm = jnp.sum(wy) * jnp.sum(wz)

    def fm(field):
        return jnp.einsum('rs,r,s->', field[-1], wy, wz) / norm

    m_k = {0: jac, 1: ginv[c] * jac, 2: met[c] / jac, 3: 1.0 / jac}[k]
    alpha = fm(m_k * jnp.sqrt(ginv[0])) / fm(m_k / jac) * PRODUCTION_BC_SCALE
    dlam = seq.basis_0.dLambda[0]
    # a clamped spline at exactly x = 1 takes the wrong branch: sample just inside the last element
    end = 1.0 - sqrt_eps() * _h_last(seq) if dlam.type != "periodic" else 0.0
    e = jax.vmap(lambda i: jnp.sum(dlam(end, i)))(dlam.ns)[window[0]:window[0] + window[1]]
    return alpha * jnp.outer(e, e) * (1.0 / _h_last(seq))


def component_factors(seq, k, c, window):
    """The 1-D mass and stiffness matrices per axis for component ``c`` of ``L_k`` on the space of
    ``seq``, restricted to the radial rows ``window = (lo, n)``.

    The masses carry no metric weight. The stiffness along axis ``a`` is weighted by the axis
    average of ``g^{aa} J``. Along an axis where the component is a derivative spline, the
    stiffness is that of the derivative splines themselves, plus the boundary term under natural
    boundary conditions.
    """
    primal, deriv, quad_w = _axis_bases(seq)
    fields = weight_fields(seq)
    ginv, jac = fields["ginv_aa"], fields["jac"]
    deriv_axes = DERIV_AXES[k][c]
    degree0 = int(seq.basis_0.Lambda[0].p) < 2
    lo, n = window

    def cut(mat, axis):
        return mat[lo:lo + n, lo:lo + n] if axis == 0 else mat

    masses, stiffs = [], []
    for a in range(3):
        prof = bundled_axis_profiles(seq, ginv[a] * jac)[a]
        m_full = _assemble_weighted_1d_mass(deriv[a] if a in deriv_axes else primal[a], quad_w[a])
        if a == 2:
            m_full = mirror_zeta_1d(seq, m_full, a in deriv_axes)
        masses.append(cut(m_full, a))
        if a in deriv_axes:
            if degree0:
                kt = cut(_fd_stiffness_degree0(seq, a, prof), a)
            else:
                kt = _assemble_weighted_1d_mass(seq.dd_basis_jk[a], quad_w[a] * prof)
                if a == 2:
                    kt = mirror_zeta_1d(seq, kt, True)
                kt = cut(kt, a)
            if a == 0 and not seq.dirichlet:
                kt = kt + _face_term(seq, k, c, window)
            stiffs.append(kt)
        else:
            G = _dense_incidence_1d(int(m_full.shape[0]), seq.basis_0.types[a])
            k_full = _symmetrize(G.T @ (_assemble_weighted_1d_mass(deriv[a], quad_w[a] * prof) @ G))
            if a == 2:
                k_full = mirror_zeta_1d(seq, k_full, False)
            stiffs.append(cut(k_full, a))
    return tuple(masses), tuple(stiffs)


def component_diagonal(seq, k, c):
    """The diagonal scaling ``D_i = int phi_i^2 m_k / int phi_i^2 J`` of component ``c``, i.e. the
    component's metric factor averaged over the support of each basis function ``phi_i``
    (``m_k`` is the metric weight of the ``k``-form mass matrix)."""
    fields = weight_fields(seq)
    jac = fields["jac"]
    w_comp = {0: jnp.ones_like(jac), 1: fields["ginv_aa"][c],
              2: fields["met_aa"][c] / jac ** 2, 3: 1.0 / jac ** 2}[k]
    primal, deriv, quad_w = _axis_bases(seq)
    deriv_axes = DERIV_AXES[k][c]
    tabs = [(deriv[a] if a in deriv_axes else primal[a]) ** 2 for a in range(3)]
    wq = seq.quad.w.reshape(seq.quad.shape)

    def contract(field):
        f = wq * field
        t1 = jnp.einsum('ax,xyz->ayz', tabs[0], f)
        t2 = jnp.einsum('by,ayz->abz', tabs[1], t1)
        return jnp.einsum('cz,abz->abc', tabs[2], t2)

    num = mirror_component(seq, contract(w_comp * jac), deriv_axes)
    den = mirror_component(seq, contract(jac), deriv_axes)
    return num / den


def _kron_mass_model_1d(seq, k):
    """The model ``M_k ~ Lam_c (A_r (x) A_t (x) A_z) Lam_c`` per component ``c``. Returns the
    unweighted 1-D masses ``A`` and the diagonal scaling ``Lam_c``, chosen so that the model has
    exactly the diagonal of ``M_k``."""
    form = getattr(seq, f"basis_{k}")
    d_raw = build_mass_diagonal(seq, k)
    primal, deriv, quad_w = _axis_bases(seq)
    mass_1d, lam, start = [], [], 0
    for c, shape in enumerate(form.shape):
        deriv_axes = form.derivative_axes(c)
        m1 = [_assemble_weighted_1d_mass(deriv[a] if a in deriv_axes else primal[a], quad_w[a])
              for a in range(3)]
        m1[2] = mirror_zeta_1d(seq, m1[2], 2 in deriv_axes)
        kron_diag = jnp.einsum('i,j,l->ijl', jnp.diag(m1[0]), jnp.diag(m1[1]), jnp.diag(m1[2]))
        size = int(np.prod(shape))
        mass_1d.append(m1)
        lam.append(jnp.sqrt(d_raw[start:start + size].reshape(shape) / kron_diag))
        start += size
    return mass_1d, lam


# --------------------------------------------------------------------------- #
# Core block and the bulk/core split                                           #
# --------------------------------------------------------------------------- #

def _parity_split(seq, k):
    """On a half-period sequence, the split ``x -> (x_even, x_odd)``, else ``None``. Operators on
    such a sequence are exact only on vectors of one parity, so a probe applies them to each part."""
    if not seq.half_period:
        return None
    projector = seq.free_projector(k)
    return lambda x: (projector.post(x, 1.0), projector.post(x, -1.0))


def _probe_rows(apply, size, rows, dtype, split):
    """The dense, symmetrised block of the operator ``apply`` on ``rows``, one apply per row."""
    if rows.size == 0:
        return jnp.zeros((0, 0), dtype=dtype)
    rows_j = jnp.asarray(rows)

    def column(e):
        return (apply(e) if split is None else sum(apply(part) for part in split(e)))[rows_j]

    # the columns in batches of PROBE_BATCH inside one call: one host dispatch instead of one per axis row
    units = jnp.zeros((rows.size, size), dtype=dtype).at[jnp.arange(rows.size), rows_j].set(1.0)
    block = jax.lax.map(column, units, batch_size=PROBE_BATCH).T
    return 0.5 * (block + block.T)


def _dense_symmetric_inverse(block, tol):
    """Pseudoinverse of a symmetric ``block`` dropping ``|w| <= tol max|w|``."""
    if block.size == 0:
        return block
    w, v = jnp.linalg.eigh(block)
    keep = jnp.abs(w) > tol * jnp.max(jnp.abs(w))
    inv_w = jnp.where(keep, 1.0 / jnp.where(keep, w, 1.0), 0.0)
    return (v * inv_w) @ v.T


def _tensor_blocks(seq, k):
    """Split the degrees of freedom into the axis rows (``core``) and, per vector component, the
    tensor-product rows away from the axis, as ``(c, rows, vals, (r0, nr), shape, offset)``.

    ``rows`` and ``vals`` map the component's radial slab ``[r0, r0 + nr)`` of shape
    ``(nr, n_t, n_z)`` to the free degrees of freedom. ``offset >= 0`` marks a slab that is just a
    contiguous range starting at ``offset``.
    """
    shapes = [tuple(int(s) for s in sh) for sh in getattr(seq, f"basis_{k}").shape]
    starts = np.cumsum([0] + [int(np.prod(s)) for s in shapes])
    e = seq.E(k)
    core = np.asarray(seq.core_rows(k))
    bulk = np.setdiff1d(np.arange(int(e.forward_shape[0])), core)
    rows, cols, vals = (np.asarray(e.rows), np.asarray(e.cols), np.asarray(e.vals))
    keep = np.isin(rows, bulk)
    rows_b, cols_b, vals_b = rows[keep], cols[keep], vals[keep]
    comp = np.searchsorted(starts[1:], cols_b, side="right")
    loc = cols_b - starts[comp]

    blocks = []
    for c, shape in enumerate(shapes):
        sel = comp == c
        if not sel.any():
            continue
        lidx = loc[sel]
        r0 = int((lidx // (shape[1] * shape[2])).min())
        nr = int((lidx // (shape[1] * shape[2])).max()) + 1 - r0
        order = np.argsort(lidx - r0 * shape[1] * shape[2])
        rows_t, vals_t = rows_b[sel][order], vals_b[sel][order]
        selector = (np.array_equal(rows_t, rows_t[0] + np.arange(rows_t.size))
                    and np.all(vals_t == 1.0))
        blocks.append((c, rows_t, vals_t, (r0, nr), (nr, shape[1], shape[2]),
                       int(rows_t[0]) if selector else -1))
    return core, blocks

# --------------------------------------------------------------------------- #
# The applied payload                                                          #
# --------------------------------------------------------------------------- #
#
# Away from the axis every vector component is a tensor-product block, post * R D^-1 L (pre * x), with the
# per-axis matrices L (the 1-D inverses for the mass, V^T for the Laplacian), D the Laplacian's Kronecker-sum
# eigenvalues (none for the mass) and R = V (none for the mass). The components are zero-padded to one grid and
# applied as one batch, with identity on the padding of the 1-D matrices. Everything linear and fixed around them
# (the component gathers, the parity expansion X of a reduced view, the output permutation and X^T) is composed
# on the host into one input gather and one output gather, because on a GPU every launched kernel costs a few
# microseconds whatever its size.

class _Batched(eqx.Module):
    """The arrays of one batched preconditioner apply (see the section comment)."""

    in_idx: jnp.ndarray          # (nb, N0 N1 N2, T_in) input entries feeding each padded block entry
    in_w: jnp.ndarray            # (nb, N0 N1 N2, T_in)
    pre: jnp.ndarray             # (nb, N0, N1, N2)
    post: jnp.ndarray            # (nb, N0, N1, N2)
    left: tuple                  # 3 x (nb, Na, Na)
    right: object                # 3 x (nb, Na, Na), or None for the mass
    lam: object                  # 3 x (nb, Na) Kronecker-sum eigenvalues, or None for the mass
    core_idx: jnp.ndarray        # (n_core, T_c) input entries feeding each axis row
    core_w: jnp.ndarray
    core_inv: jnp.ndarray        # (n_core, n_core)
    out_idx: jnp.ndarray         # (n_out, T_out) positions in concat(block results, axis results)
    out_w: jnp.ndarray
    N: tuple = eqx.field(static=True)


def _rows_table(rows, cols, vals, n_rows):
    """A sparse matrix given by its entries as padded per-row tables ``(cols, vals)`` of shape ``(n_rows, T)``."""
    order = np.argsort(rows, kind="stable")
    rows, cols, vals = rows[order], cols[order], vals[order]
    counts = np.bincount(rows, minlength=n_rows)
    T = max(int(counts.max()) if counts.size else 1, 1)
    first = np.concatenate([[0], np.cumsum(counts)[:-1]])
    slot = np.arange(rows.size) - first[rows]
    c_tab, v_tab = np.zeros((n_rows, T), dtype=np.int64), np.zeros((n_rows, T))
    c_tab[rows, slot], v_tab[rows, slot] = cols, vals
    return c_tab, v_tab


def _batch(specs, core, core_inv, n_full, X=None):
    """The :class:`_Batched` of the blocks ``specs`` (dicts with ``rows``, ``vals``, ``shape``, ``pre``, ``post``,
    ``left`` and, for the Laplacian, ``right`` and ``lam``) and the axis rows ``core`` with their dense inverse,
    on a space of ``n_full`` entries. With ``X`` (a reduced view's expansion, full <- reduced) the apply acts on
    the reduced vectors as ``X^T P X``."""
    core = np.asarray(core)
    if X is None:
        x_cols, x_vals = np.arange(n_full)[:, None], np.ones((n_full, 1))
        out_rows = (np.arange(n_full), np.arange(n_full), np.ones(n_full))
        n_out = n_full
    else:
        r, c, v = X.entries()
        x_cols, x_vals = _rows_table(r, c, v, n_full)              # x_full[r] = sum_t x_vals x_red[x_cols]
        out_rows = (c, r, v)                                       # y_red[c] = sum v y_full[r]
        n_out = int(X.forward_shape[1])
    nb = len(specs)
    N = tuple(max(s["shape"][a] for s in specs) for a in range(3))
    n_tot = N[0] * N[1] * N[2]
    T = x_cols.shape[1]
    in_idx, in_w = np.zeros((nb, n_tot, T), dtype=np.int64), np.zeros((nb, n_tot, T))
    pre, post = np.ones((nb,) + N), np.ones((nb,) + N)
    laplacian = "right" in specs[0]
    left = [np.zeros((nb, N[a], N[a])) for a in range(3)]
    right = [np.zeros((nb, N[a], N[a])) for a in range(3)]
    lam = [np.zeros((nb, N[a])) for a in range(3)]
    source = np.zeros((n_full, 2))                                 # y_full[j] = source[j, 1] z[source[j, 0]]
    for b, s in enumerate(specs):
        shape = s["shape"]
        n = shape[0] * shape[1] * shape[2]
        padded = np.ravel_multi_index(np.unravel_index(np.arange(n), shape), N)
        rows, vals = np.asarray(s["rows"]), np.asarray(s["vals"], dtype=np.float64)
        in_idx[b, padded] = x_cols[rows]
        in_w[b, padded] = x_vals[rows] * vals[:, None]
        sl = (b, slice(0, shape[0]), slice(0, shape[1]), slice(0, shape[2]))
        pre[sl] = np.broadcast_to(np.asarray(s["pre"], dtype=np.float64), shape)
        post[sl] = np.broadcast_to(np.asarray(s["post"], dtype=np.float64), shape)
        for a in range(3):
            left[a][b] = np.eye(N[a])
            left[a][b, :shape[a], :shape[a]] = np.asarray(s["left"][a], dtype=np.float64)
            if laplacian:
                right[a][b] = np.eye(N[a])
                right[a][b, :shape[a], :shape[a]] = np.asarray(s["right"][a], dtype=np.float64)
                lam[a][b, :shape[a]] = np.asarray(s["lam"][a], dtype=np.float64)
        source[rows] = np.stack([b * n_tot + padded, vals], axis=1)
    source[core] = np.stack([nb * n_tot + np.arange(core.size), np.ones(core.size)], axis=1)
    o_cols, o_vals = _rows_table(*out_rows, n_out)

    def dev(arrays):
        return tuple(jnp.asarray(m, DTYPE) for m in arrays)
    return _Batched(jnp.asarray(in_idx), jnp.asarray(in_w, DTYPE), jnp.asarray(pre, DTYPE),
                    jnp.asarray(post, DTYPE), dev(left), dev(right) if laplacian else None,
                    dev(lam) if laplacian else None, jnp.asarray(x_cols[core]), jnp.asarray(x_vals[core], DTYPE),
                    jnp.asarray(core_inv, DTYPE), jnp.asarray(source[o_cols, 0].astype(np.int64)),
                    jnp.asarray(o_vals * source[o_cols, 1], DTYPE), N)


@eqx.filter_jit
def _apply_batched(p, x, core_inv, alpha, shift):
    """Apply the :class:`_Batched` ``p`` to ``x``. ``core_inv`` replaces ``p.core_inv`` when given. ``alpha``
    (per block and axis) weighs the Laplacian's Kronecker terms, ``None`` for all ones. ``shift`` is ``1/eps`` of
    the shifted-stiffness preconditioner and ``None`` otherwise."""
    nb = p.pre.shape[0]
    xb = jnp.sum(p.in_w * x[p.in_idx], axis=-1).reshape((nb,) + p.N) * p.pre
    y = jnp.einsum('bia,bjc,bkd,bacd->bijk', *p.left, xb)
    if p.lam is not None:
        lam = p.lam if alpha is None else tuple(alpha[:, a, None] * p.lam[a] for a in range(3))
        denom = lam[0][:, :, None, None] + lam[1][:, None, :, None] + lam[2][:, None, None, :]
        if shift is None:
            # the kernel (the constants) is dropped
            null = jnp.abs(denom) < sqrt_eps(6.7e-3) * jnp.max(jnp.abs(denom), axis=(1, 2, 3), keepdims=True)
            y = jnp.where(null, 0.0, y / jnp.where(null, 1.0, denom))
        else:
            y = shift * y / (denom + shift)
        y = jnp.einsum('bia,bjc,bkd,bacd->bijk', *p.right, y)
    y = y * p.post
    core = (p.core_inv if core_inv is None else core_inv) @ jnp.sum(p.core_w * x[p.core_idx], axis=-1)
    z = jnp.concatenate([y.reshape(-1), core])
    return jnp.sum(p.out_w * z[p.out_idx], axis=-1)


def _cast(p, dtype):
    """``p`` with its floating-point arrays in ``dtype``."""
    return jax.tree_util.tree_map(lambda a: a.astype(dtype) if jnp.issubdtype(a.dtype, jnp.floating) else a, p)


class _Atom:
    """What the mass and the Laplacian preconditioner share: the split into axis and off-axis
    rows, the dense axis block, and the apply."""

    def _prologue(self, seq, k):
        """Return the axis rows, the off-axis blocks, the sequence to probe on and ``probe``.
        ``probe(apply)`` is the dense block of ``apply`` on the axis rows, computed in the
        residual precision."""
        self.parity_projector = seq.free_projector(k)   # None on a full-period sequence
        core, raw_blocks = _tensor_blocks(seq, k)
        on = seq if seq.residual is None else seq.residual
        size, split = seq.n(k), _parity_split(on, k)
        self._n = size
        return core, raw_blocks, on, lambda apply: _probe_rows(apply, size, core, on.dtype, split)

    def _install(self, core, specs, core_inv):
        # device arrays, not NumPy: NumPy attributes are static and a new geometry would recompile
        self._core = jnp.asarray(core)
        self._specs = [dict(s, rows=jnp.asarray(s["rows"]), vals=jnp.asarray(s["vals"], dtype=DTYPE)) for s in specs]
        self._payload = _batch(self._specs, core, core_inv, self._n)

    def reduced(self, X):
        """The batched payload of ``X^T P X`` for a reduced view with the expansion ``X``."""
        return _batch(self._specs, self._core, self._payload.core_inv, self._n, X)

    def apply(self, x):
        """Apply the preconditioner to a coefficient vector of the free degrees of freedom."""
        p = self._payload
        if self.parity_projector is None:
            return _apply_batched(p, jnp.asarray(x), None, None, None)
        return self.parity_projector(lambda v: _apply_batched(p, v, None, None, None), jnp.asarray(x))

    def apply_in(self, dtype):
        """Return :meth:`apply` computed in ``dtype``, for use inside an operator of that
        precision."""
        dtype = jnp.dtype(dtype)
        p = _cast(self._payload, dtype)
        project = self.parity_projector

        def apply(x):
            if project is None:
                return _apply_batched(p, jnp.asarray(x, dtype), None, None, None)
            return project(lambda v: _apply_batched(p, jnp.asarray(v, dtype), None, None, None),
                           jnp.asarray(x, dtype))
        return apply


@register_arrays
class MetricLumpingLaplacian(_Atom):
    """Approximate inverse of the Hodge Laplacian ``L_k`` on the space of ``seq``.

    Built by ``DeRhamSequence.build_preconditioners`` for the installed geometry. Apply it with
    :meth:`apply`. Needs at least two radial elements (``n >= p + 2``).
    """

    def __init__(self, seq, operators, k):
        from mrx.operators import apply_laplacian_approx  # noqa: PLC0415
        core, raw_blocks, on, probe = self._prologue(seq, k)
        specs, strong = [], []
        for c, rows, vals, (r0, nr), shape, offset in raw_blocks:
            masses, stiffs = component_factors(seq, k, c, (r0, nr))
            v, lam = zip(*map(_simultaneous_diagonalize_pair, masses, stiffs))
            # D_i is a ratio of two positive integrals
            dscale = 1.0 / jnp.sqrt(component_diagonal(seq, k, c)[r0:r0 + nr])
            specs.append(dict(rows=rows, vals=vals, shape=shape, pre=dscale, post=dscale,
                              left=tuple(m.T for m in v), right=v, lam=lam))
            # the strong half S_k: the Kronecker terms of the primal axes
            strong.append([0.0 if a in DERIV_AXES[k][c] else 1.0 for a in range(3)])
        self._strong = jnp.asarray(strong, dtype=DTYPE)
        core_inv = _dense_symmetric_inverse(
            probe(lambda x: apply_laplacian_approx(on, operators, x, k)), CORE_TOL)
        # (M_k, S_k) on the core, diagonalised once for the shifted apply's (M + eps S)^-1
        mass_core = probe(lambda x: on.M[k] @ x)
        stiffness_core = probe(lambda x: on.S[k] @ x)
        self._core_pair = (_simultaneous_diagonalize_pair(mass_core, stiffness_core) if core.size
                           else (mass_core, jnp.zeros(0, dtype=DTYPE)))
        self._install(core, specs, core_inv)

    def shifted_stiffness_apply(self, eps, payload=None):
        """Return ``x -> (M_k + eps S_k)^{-1} x`` in the same approximation, the preconditioner of
        the shifted Laplacian solves. ``eps`` may be a traced value, so changing it does not
        rebuild or recompile anything. ``payload`` is a reduced view's :meth:`reduced` payload."""
        V, mu = self._core_pair
        core_inv = (V / (1.0 + eps * mu)) @ V.T
        p, strong = (self._payload if payload is None else payload), self._strong

        def apply(x):
            return _apply_batched(p, jnp.asarray(x), core_inv, strong, 1.0 / eps)
        return apply


@register_arrays
class MetricLumpingMass(_Atom):
    """Approximate inverse of the mass matrix ``M_k`` on the space of ``seq``.

    Built by ``DeRhamSequence.build_preconditioners`` for the installed geometry. Apply it with
    :meth:`apply`.
    """

    def __init__(self, seq, operators, k):
        core, raw_blocks, on, probe = self._prologue(seq, k)
        mass_1d, lam = _kron_mass_model_1d(seq, k)
        specs = [dict(rows=rows, vals=vals, shape=shape, pre=1.0 / lam[c][r0:r0 + nr],
                      post=1.0 / lam[c][r0:r0 + nr],
                      left=tuple(jnp.linalg.inv(m[r0:r0 + nr, r0:r0 + nr] if a == 0 else m)
                                 for a, m in enumerate(mass_1d[c])))
                 for c, rows, vals, (r0, nr), shape, offset in raw_blocks]
        core_inv = _dense_symmetric_inverse(
            probe(lambda x: on.M[k] @ x), CORE_TOL)
        self._install(core, specs, core_inv)


class ReducedAtom(eqx.Module):
    """A preconditioner ``P`` of the half-period sequence restricted to its even or odd part, as
    ``X^T P X`` where ``X`` expands a vector of that part to the full space. The parity views
    of the sequence hold these, built from the base sequence's preconditioners, with ``X`` composed into the
    gathers of the batched apply."""
    atom: object
    payload: _Batched

    def __init__(self, atom, X):
        self.atom = atom
        self.payload = atom.reduced(X)

    def apply(self, x):
        return _apply_batched(self.payload, jnp.asarray(x), None, None, None)

    def apply_in(self, dtype):
        """Return :meth:`apply` computed in ``dtype``."""
        dtype = jnp.dtype(dtype)
        p = _cast(self.payload, dtype)

        def apply(x):
            return _apply_batched(p, jnp.asarray(x, dtype), None, None, None)
        return apply

    def shifted_stiffness_apply(self, eps):
        """The restricted :meth:`MetricLumpingLaplacian.shifted_stiffness_apply`, ``x -> X^T (M + eps S)^{-1} X x``."""
        return self.atom.shifted_stiffness_apply(eps, self.payload)
