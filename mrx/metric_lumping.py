"""Approximate inverses of the mass matrices ``M_k`` and the Hodge Laplacians ``L_k``.

These are the preconditioners of every mass and Laplacian solve in MRX. There is one of each
per form degree ``k`` and space (the Dirichlet space of a sequence and its free view):

- :class:`MetricLumpingMass` approximates ``M_k^{-1}``. It also stands in for ``M_{k-1}^{-1}``
  inside the approximate Laplacian the solvers use, so it is part of that operator as well.
- :class:`MetricLumpingLaplacian` approximates ``L_k^{-1}``. Its
  :meth:`~MetricLumpingLaplacian.shifted_stiffness_apply` approximates ``(M_k + eps S_k)^{-1}``,
  where ``S_k = d_k^T M_{k+1} d_k`` is the stiffness, the half of ``L_k`` built from the
  exterior derivative ``d_k`` of ``k``-forms.
- :class:`ReducedAtom` is either of them, or the slab mass model of :mod:`mrx.slab_mass`,
  restricted to the even or odd part of a stellarator-symmetric (half-period) sequence.

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

On a graded space a component has one Kronecker block per theta level. The neighbouring levels are
coupled through a few rings, the strips, and the model with these couplings is still inverted
exactly, by block elimination with a small dense Schur complement on the strips. The angular
metric averages of the Laplacian are then taken per level.
"""

from __future__ import annotations

import os

import numpy as np

import equinox as eqx
import jax
import jax.numpy as jnp

from mrx.differential_forms import DERIV_AXES
from mrx.extraction_operators import prolongation, ring_layout
from mrx.incidence import diff_1d
from mrx.mass import build_mass_diagonal
from mrx.precision import PRECOND_DTYPE, eps, lift, sqrt_eps, to_precond, to_working
from mrx.pytree import register_arrays
from mrx.symmetry import mirror_component, mirror_zeta_1d

#: Axis rows probed per batch when the dense axis block is built. The probe holds one field at every
#: quadrature point per row, so the batch sets the peak device memory of the setup: on li383 (64,80,32),
#: p = 2, float64 it is 15.1 GiB at 64, 4.4 GiB at 8 and 3.0 GiB at 1, with the same build time, and 2.2 GiB
#: stay in use afterwards (2026-10-12). Set it with the environment variable ``MRX_PROBE_BATCH``.
PROBE_BATCH = int(os.environ.get("MRX_PROBE_BATCH", "8"))

#: Empirical factor on the boundary term of the Laplacian preconditioner under natural boundary
#: conditions. Three times the exact surface integral gave the smallest condition number.
PRODUCTION_BC_SCALE = 3.0

#: Exponent of the angular profile of the stiffness diagonal in the diagonal scaling of the Laplacian atom
#: (:func:`_angular_profile`). On graded li383 (18,32,12) the co-exact S_1 takes 50 / 42 / 42 / 49 iterations at
#: 0.25 / 0.5 / 0.6 / 1, S_0 39 at 0.5 and 48 at 1 (2026-10-18).
STIFFNESS_PROFILE_EXPONENT = 0.5


# --------------------------------------------------------------------------- #
# 1-D building blocks                                                          #
# --------------------------------------------------------------------------- #

def _symmetrize(matrix):
    return 0.5 * (matrix + matrix.T)


def _assemble_weighted_1d_mass(B, weights):
    return (B * weights[None, :]) @ B.T


def _simultaneous_diagonalize_pair(M, A):
    """``(V, lam)`` with ``V^T M V = I`` and ``V^T A V = diag(lam)`` for SPD ``M``, symmetric ``A``."""
    M_sym = _symmetrize(jnp.asarray(M))
    A_sym = _symmetrize(jnp.asarray(A))
    L = jnp.linalg.cholesky(M_sym)
    Linv_A = jax.scipy.linalg.solve_triangular(L, A_sym, lower=True)
    B = jax.scipy.linalg.solve_triangular(L, Linv_A.T, lower=True).T
    lam, U = jnp.linalg.eigh(_symmetrize(B))
    V = jax.scipy.linalg.solve_triangular(L.T, U, lower=False)
    return V, lam


# --------------------------------------------------------------------------- #
# Metric weights and 1-D factors                                               #
# --------------------------------------------------------------------------- #

def bundled_axis_profiles(seq, field, r_mask=None):
    """The averages ``(p_r, p_t, p_z)`` of a field on the quadrature grid, each a function of one
    coordinate averaged over the other two.

    The radial profile leaves out the element next to the axis, which the dense axis block covers.
    ``r_mask`` (0 or 1 per radial quadrature point) restricts the angular profiles to part of the radius.
    """
    xi1 = jnp.asarray(seq.basis_0.Lambda[0].T)[seq.p + 1]
    wx = seq.quad.w_x * (jnp.asarray(seq.quad.x_x) >= xi1)
    if r_mask is not None:
        wx = wx * r_mask
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
    return seq.basis_jk, seq.d_basis_jk, (seq.quad.w_x, seq.quad.w_y, seq.quad.w_z)


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
    return jnp.asarray(0.5 * (k + k.T))


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


def _axis_stiffness(seq, k, c, a, profile):
    """The 1-D stiffness of component ``c`` of ``L_k`` along axis ``a`` with the weight ``profile`` at the 1-D
    quadrature points, on all functions of the axis."""
    primal, deriv, quad_w = _axis_bases(seq)
    if a in DERIV_AXES[k][c]:
        if int(seq.basis_0.Lambda[0].p) < 2:
            return _fd_stiffness_degree0(seq, a, profile)
        kt = _assemble_weighted_1d_mass(seq.dd_basis_jk[a], quad_w[a] * profile)
        return mirror_zeta_1d(seq, kt, True) if a == 2 else kt
    G = jnp.asarray(diff_1d(int(primal[a].shape[0]), seq.basis_0.types[a]).toarray())
    kt = _symmetrize(G.T @ (_assemble_weighted_1d_mass(deriv[a], quad_w[a] * profile) @ G))
    return mirror_zeta_1d(seq, kt, False) if a == 2 else kt


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
    lo, n = window

    def cut(mat, axis):
        return mat[lo:lo + n, lo:lo + n] if axis == 0 else mat

    masses, stiffs = [], []
    for a in range(3):
        m_full = _assemble_weighted_1d_mass(deriv[a] if a in deriv_axes else primal[a], quad_w[a])
        if a == 2:
            m_full = mirror_zeta_1d(seq, m_full, a in deriv_axes)
        masses.append(cut(m_full, a))
        kt = cut(_axis_stiffness(seq, k, c, a, bundled_axis_profiles(seq, ginv[a] * jac)[a]), a)
        if a == 0 and a in deriv_axes and not seq.dirichlet:
            kt = kt + _face_term(seq, k, c, window)
        stiffs.append(kt)
    return tuple(masses), tuple(stiffs)


def component_diagonal(seq, k, c, theta=None):
    """The diagonal scaling ``D_i = int phi_i^2 m_k / int phi_i^2 J`` of component ``c``, i.e. the
    component's metric factor averaged over the support of each basis function ``phi_i``
    (``m_k`` is the metric weight of the ``k``-form mass matrix). With ``theta`` (a knot insertion
    ``R``) the theta functions are the coarse ones of a graded ring."""
    fields = weight_fields(seq)
    jac = fields["jac"]
    w_comp = {0: jnp.ones_like(jac), 1: fields["ginv_aa"][c],
              2: fields["met_aa"][c] / jac ** 2, 3: 1.0 / jac ** 2}[k]
    primal, deriv, quad_w = _axis_bases(seq)
    deriv_axes = DERIV_AXES[k][c]
    tabs = [deriv[a] if a in deriv_axes else primal[a] for a in range(3)]
    if theta is not None:
        tabs[1] = jnp.asarray(theta).T @ tabs[1]
    tabs = [t ** 2 for t in tabs]
    wq = seq.quad.w.reshape(seq.quad.shape)

    def contract(field):
        f = wq * field
        t1 = jnp.einsum('ax,xyz->ayz', tabs[0], f)
        t2 = jnp.einsum('by,ayz->abz', tabs[1], t1)
        return jnp.einsum('cz,abz->abc', tabs[2], t2)

    num = mirror_component(seq, contract(w_comp * jac), deriv_axes)
    den = mirror_component(seq, contract(jac), deriv_axes)
    return num / den


def stiffness_diagonal(seq, k, c, theta=None):
    """The diagonal of component ``c`` of the stiffness ``S_k = G_k^T M_{k+1} G_k`` on the raw ``(n_r, n_theta,
    n_zeta)`` grid of its tensor-product coefficients, for ``k = 0, 1, 2``. ``theta`` is a knot insertion as in
    :func:`component_diagonal`. It is computed directly, without applying ``S_k``.

    The exterior derivative of a basis function has one term per axis on which the function is a spline, not a
    derivative spline: its derivative along that axis, a component of the ``(k + 1)``-form. The diagonal is the
    integral of these terms against the weight of ``M_{k+1}``. Under the vector-proxy curl the two terms of a
    1-form enter with opposite signs, so their cross term at ``k = 1`` is negative."""
    pk, pk1 = seq.mass_plan[k], seq.mass_plan[k + 1]
    W = seq.geometry.mass_weights[k + 1]
    s = pk.shapes_c[c]
    tables = [pk.tables_c[a][c][:s[a]] for a in range(3)]
    deriv_axes = DERIV_AXES[k][c]
    terms = []
    for a in range(3):
        if a in deriv_axes:
            continue
        cp = DERIV_AXES[k + 1].index(tuple(sorted(deriv_axes + (a,))))
        t = list(tables)
        # the derivative of a spline in the derivative splines: B_i' = D_{i-1} - D_i
        t[a] = jnp.asarray(diff_1d(s[a], seq.basis_0.types[a]).toarray()).T @ pk1.tables_c[a][cp][:pk1.shapes_c[cp][a]]
        if theta is not None:
            t[1] = jnp.asarray(theta).T @ t[1]
        terms.append((a, cp, t))
    d = 0.0
    for a, ca, ta in terms:
        for b, cb, tb in terms:
            sign = -1.0 if a != b and k == 1 else 1.0
            d = d + sign * jnp.einsum('ia,jb,kd,abd->ijk', ta[0] * tb[0], ta[1] * tb[1], ta[2] * tb[2],
                                      W[..., ca, cb])
    # as in build_mass_diagonal: an entry and its mirror image each hold one half of the full integral
    return mirror_component(seq, d, deriv_axes)


def _angular_profile(seq, k, c, R, terms, zeta, D, r0, a0):
    """The factor on the diagonal scaling ``D`` of a block of the Laplacian atom (radial rows from ``r0``, the
    component's model starting at ring ``a0``, theta functions ``R``) that gives the model the angular profile of
    ``diag(S_k)``.

    The lumped model is too stiff where the stiffness weights ``g^{aa} J`` are small on a ring and too soft where
    they are large. On li383 they vary by up to 20x along the last rings near ``r = 1``, and the lowest modes of the
    preconditioned stiffness sat there. The factor is ``q^STIFFNESS_PROFILE_EXPONENT``, ``q`` the ratio of
    ``diag(S_k)`` to the model's diagonal divided by its mean over each ring. So the radial profile of the model
    stays, and only the distribution in theta and zeta changes. ``terms`` are the model's Kronecker terms as in
    :func:`_strip_tables`, ``zeta`` the zeta mass and stiffness."""
    rings = slice(r0 - a0, r0 - a0 + D.shape[0])
    model = 0.0
    for radial, theta, part in terms:
        model = model + jnp.einsum('i,j,l->ijl', jnp.diag(jnp.asarray(radial))[rings],
                                   jnp.diag(R.T @ jnp.asarray(theta) @ R), jnp.diag(zeta[part]))
    q = stiffness_diagonal(seq, k, c, R)[r0:r0 + D.shape[0]] / (D * model)
    return (q / jnp.mean(q, axis=(1, 2), keepdims=True)) ** STIFFNESS_PROFILE_EXPONENT


def _kron_mass_model_1d(seq, k):
    """The unweighted 1-D masses ``(A_r, A_t, A_z)`` per component of the model
    ``M_k ~ Lam (A_r (x) A_t (x) A_z) Lam`` of the mass atom."""
    form = getattr(seq, f"basis_{k}")
    primal, deriv, quad_w = _axis_bases(seq)
    mass_1d = []
    for c in range(len(form.shape)):
        deriv_axes = form.derivative_axes(c)
        m1 = [_assemble_weighted_1d_mass(deriv[a] if a in deriv_axes else primal[a], quad_w[a])
              for a in range(3)]
        m1[2] = mirror_zeta_1d(seq, m1[2], 2 in deriv_axes)
        mass_1d.append(m1)
    return mass_1d


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


def _probe_rows(apply, size, rows, split):
    """The dense, symmetrised block of the operator ``apply`` on ``rows``, one apply per row. In float32 the
    operator is applied in double-single and the block is its rounded value."""
    if rows.size == 0:
        return jnp.zeros((0, 0))
    rows_j = jnp.asarray(rows)

    def column(e):
        parts = (e,) if split is None else split(e)
        return to_working(sum(apply(lift(part)) for part in parts))[rows_j]

    # the columns in batches of PROBE_BATCH inside one call: one host dispatch instead of one per axis row
    units = jnp.zeros((rows.size, size)).at[jnp.arange(rows.size), rows_j].set(1.0)
    block = jax.lax.map(column, units, batch_size=PROBE_BATCH).T
    return 0.5 * (block + block.T)


def _dense_symmetric_inverse(block, name):
    """The inverse of the symmetric positive definite axis block ``block``, by its eigenvalues.

    The axis blocks are an operator restricted to the axis rows and have no small eigenvalues: the
    smallest is 9e-6 to 4e-4 of the largest on li383 (32,44,20) to (80,104,40) and W7-X (32,64,36),
    (40,80,48) at p = 2 and 3 (2026-10-11). So nothing is cut off. A block whose smallest eigenvalue
    is not above the round-off of its largest raises, with ``name`` in the message."""
    if block.size == 0:
        return block
    w, v = jnp.linalg.eigh(block)
    if not isinstance(w, jax.core.Tracer):
        floor = eps(block.shape[0]) * float(w[-1])
        if not float(w[0]) > floor:
            raise ValueError(f"the axis block of {name} is not positive definite: its eigenvalues span "
                             f"[{float(w[0]):.3e}, {float(w[-1]):.3e}]")
    return (v / w) @ v.T


def _tensor_blocks(seq, k):
    """Split the degrees of freedom into the axis rows (``core``) and the tensor-product blocks away
    from the axis, one per vector component and theta count, as ``(c, rows, vals, (r0, nr), shape, R)``.

    ``rows`` and ``vals`` map the block's radial slab ``[r0, r0 + nr)`` of shape ``(nr, m, n_z)`` to the
    degrees of freedom, and ``R`` is the knot insertion of its ``m`` theta functions into the raw ones
    (the identity on a uniform space, which has one block per component).
    """
    form = getattr(seq, f"basis_{k}")
    p = form.Lambda[1].p
    core = np.asarray(seq.core_rows(k))
    blocks = []
    for c, a, m, derivative, first, nz in ring_layout(form, seq.dirichlet, seq.theta_counts):
        if blocks and blocks[-1][0] == c and blocks[-1][4][1] == m:
            _, _, _, (r0, nr), _, R = blocks.pop()
        else:
            r0, nr, R = a, 0, prolongation(m, form.shape[c][1], p, derivative)
            first0 = first
        nr += 1
        rows = first0 + np.arange(nr * m * nz)
        blocks.append((c, rows, np.ones(rows.size), (r0, nr), (nr, m, nz), R))
    return core, blocks


def _by_component(raw_blocks):
    """The blocks of :func:`_tensor_blocks` grouped by vector component, as ``[(c, [b, ...]), ...]`` in block
    order. The blocks of one component are consecutive and ordered outward."""
    groups = []
    for b, block in enumerate(raw_blocks):
        if groups and groups[-1][0] == block[0]:
            groups[-1][1].append(b)
        else:
            groups.append((block[0], [b]))
    return groups


def _radial_bandwidth(seq, k, c):
    """The largest distance in rings at which two radial functions of component ``c`` overlap."""
    return max(int(seq.basis_0.Lambda[0].p) - (0 in DERIV_AXES[k][c]), 1)


def _strip_rings(seq, k, raw_blocks):
    """The number of strip rings at the outer end of each block of :func:`_tensor_blocks`: the rings that overlap a
    ring of the component's next theta level. The outermost block of a component has none, so a uniform space has
    none at all."""
    out = []
    for b, (c, _, _, (_, nr), _, _) in enumerate(raw_blocks):
        last = b + 1 == len(raw_blocks) or raw_blocks[b + 1][0] != c
        out.append(0 if last else min(_radial_bandwidth(seq, k, c), nr))
    return out


def _band_factors(seq, k, c, members, raw_blocks, a0, n_rings):
    """The band-wise angular factors of component ``c`` of ``L_k`` on a graded space.

    Each theta level of the component (the blocks ``members``) has a band of radial elements: from the first element
    in the support of the level's first ring up to the next level's band. The interior rings of a level lie inside
    its band. The theta and zeta stiffness of a band carry the profiles of ``g^{aa} J`` averaged over the band
    instead of the whole radius. The zeta profiles of the bands share one shape, the mean of the profiles divided
    by their means, and keep their means as scales, so that all levels of a component have one zeta eigenbasis.
    The element at the axis is left out of the averages as in :func:`bundled_axis_profiles`. A band without other
    elements takes the profiles of the next band.

    Returns per level the theta stiffness and the zeta scale, the common zeta stiffness, and per level the radial
    mass of the band on the component's rings (window ``(a0, n_rings)``). Their sum is the radial mass.
    """
    fields = weight_fields(seq)
    ginv, jac = fields["ginv_aa"], fields["jac"]
    primal, deriv, quad_w = _axis_bases(seq)
    table = deriv[0] if 0 in DERIV_AXES[k][c] else primal[0]
    knots = np.unique(np.asarray(seq.basis_0.Lambda[0].T))
    n_cell = knots.size - 1
    cell = np.clip(np.searchsorted(knots, np.asarray(seq.quad.x_x), side="right") - 1, 0, n_cell - 1)
    support = np.asarray(table) != 0
    lo = [0] + [int(cell[support[raw_blocks[b][3][0]]].min()) for b in members[1:]] + [n_cell]
    bands = [(cell >= lo[i]) & (cell < lo[i + 1]) for i in range(len(members))]
    averaged = [band & (cell >= 1) for band in bands]
    for i in reversed(range(len(members) - 1)):
        if not averaged[i].any():
            averaged[i] = averaged[i + 1]
    kt, scale, shape, mr = [], [], [], []
    for band, region in zip(bands, averaged):
        mask = jnp.asarray(region, dtype=float)
        pt = bundled_axis_profiles(seq, ginv[1] * jac, mask)[1]
        pz = bundled_axis_profiles(seq, ginv[2] * jac, mask)[2]
        kt.append(_axis_stiffness(seq, k, c, 1, pt))
        scale.append(jnp.sum(quad_w[2] * pz) / jnp.sum(quad_w[2]))
        shape.append(pz / scale[-1])
        m = _assemble_weighted_1d_mass(table, quad_w[0] * jnp.asarray(band, dtype=float))
        mr.append(m[a0:a0 + n_rings, a0:a0 + n_rings])
    return kt, scale, _axis_stiffness(seq, k, c, 2, sum(shape) / len(shape)), mr


def _strip_tables(comps, N, nb):
    """The host tables of the strip elimination (see :class:`_Strips`). ``comps`` lists, per component with strips,
    its blocks as ``(b, r, nr, ns, R)`` (batch index, first ring in the component's radial matrices, rings, strip
    rings, knot insertion of the theta functions) and the terms ``(radial, theta, part)`` of its model. On one zeta
    mode the model is the sum of ``radial (x) R^T theta R'`` over the terms of part 0, plus the zeta eigenvalue
    times the sum over those of part 1."""
    N0, N1 = N[0], N[1]
    per = []
    for blocks, terms in comps:
        pos, strip, start = [], [], [0]
        for b, r, nr, ns, R in blocks:
            ii, jj = np.divmod(np.arange(nr * R.shape[1]), R.shape[1])
            pos.append((b * N0 + ii) * N1 + jj)
            strip.append(ii >= nr - ns)
            start.append(start[-1] + nr * R.shape[1])
        pos, strip = np.concatenate(pos), np.concatenate(strip)
        row_of = np.cumsum(strip) - 1
        A = np.zeros((2, int(strip.sum()), pos.size))
        for i, (b, r, nr, ns, R) in enumerate(blocks):
            rows = row_of[start[i] + (nr - ns) * R.shape[1]:start[i + 1]]
            for j, (_, r2, nr2, _, R2) in enumerate(blocks):
                for rad, theta, part in terms:
                    sub = rad[r + nr - ns:r + nr, r2:r2 + nr2]
                    if ns and sub.any():
                        A[part][np.ix_(rows, np.arange(start[j], start[j + 1]))] += np.kron(sub, R.T @ theta @ R2)
        per.append((pos, strip, A))
    nc, G = len(per), max(A.shape[1] for _, _, A in per)
    slots = np.full((nc, G), nb * N0 * N1)                  # out of range on the padding
    ass = np.zeros((2, nc, G, G))
    c_ent = []
    for s, (pos, strip, A) in enumerate(per):
        g_n = A.shape[1]
        slots[s, :g_n] = pos[strip]
        ass[:, s, :g_n, :g_n] = A[:, :, strip]
        ass[0, s, g_n:, g_n:] = np.eye(G - g_n)
        inter = np.flatnonzero(~strip)
        g, e = np.nonzero((A[0][:, inter] != 0) | (A[1][:, inter] != 0))
        c_ent.append((s * G + g, pos[inter[e]], A[0][g, inter[e]], A[1][g, inter[e]], np.full(g.size, s)))
    rows, cols, v0, v1, slot = (np.concatenate(a) for a in zip(*c_ent))
    c_idx, c0 = _rows_table(rows, cols, v0, nc * G)
    _, c1 = _rows_table(rows, cols, v1, nc * G)
    e_idx, local = np.unique(cols, return_inverse=True)
    d_idx, d0 = _rows_table(local, rows, v0, e_idx.size)
    _, d1 = _rows_table(local, rows, v1, e_idx.size)
    d_slot = np.zeros(e_idx.size, dtype=np.int64)
    d_slot[local] = slot
    shape = (nc, G, -1)
    return dict(slots=slots, c_idx=c_idx.reshape(shape), c0=c0.reshape(shape), c1=c1.reshape(shape), e_idx=e_idx,
                d_idx=d_idx, d0=d0, d1=d1, d_slot=d_slot), ass


def _build_strips(p, specs, comps):
    """The :class:`_Strips` of the payload ``p`` (built without them) from the blocks' ``interior`` radial factors and
    ``comps``, per component with strips ``(blocks, terms, lam_z)`` (:func:`_strip_tables`, ``lam_z`` the zeta
    eigenvalues of the Laplacian, ``None`` for the mass)."""
    nb, (N0, _, N2) = p.pre.shape[0], p.N
    laplacian = p.lam is not None
    left, right, lam = np.tile(np.eye(N0), (nb, 1, 1)), np.tile(np.eye(N0), (nb, 1, 1)), np.zeros((nb, N0))
    for b, spec in enumerate(specs):
        L, R, lr = spec["interior"]
        n = L.shape[0]
        left[b, :n, :n] = np.asarray(L)
        if laplacian:
            right[b, :n, :n], lam[b, :n] = np.asarray(R), np.asarray(lr)
    tables, ass = _strip_tables([(blocks, terms) for blocks, terms, _ in comps], p.N, nb)
    lam_z = np.zeros((len(comps), N2))
    if laplacian:
        for s, (_, _, lz) in enumerate(comps):
            lam_z[s, :len(lz)] = np.asarray(lz)
    dev = {key: jnp.asarray(val) for key, val in tables.items()}
    if not laplacian:
        dev["c1"] = dev["d1"] = None
    strips = _Strips(jnp.asarray(left), jnp.asarray(right) if laplacian else None,
                     jnp.asarray(lam) if laplacian else None, jnp.asarray(lam_z) if laplacian else None,
                     inv=None, **dev)
    return eqx.tree_at(lambda s: s.inv, strips, _schur_inverse(p, strips, jnp.asarray(ass)),
                       is_leaf=lambda x: x is None)

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
#
# On a graded space a component has one block per theta level, and the model couples neighbouring levels through
# their strips, the last rings of a level that overlap rings of the next one. Without the strips the levels are
# uncoupled interiors, so the model is inverted exactly by block elimination: the interior solves, a dense Schur
# complement on the strips, the interior solves again. The strip rings stay inside their level's padded block. An
# interior solve uses a radial factor of the interior rings alone, the identity on the strip rings. The Laplacian
# does all of it in the eigenbasis of its zeta pair, which every block of a component shares, so the Schur
# complement is one dense block per component and zeta mode. The mass applies its zeta inverse first and has one
# Schur block per component.

class _Strips(eqx.Module):
    """The arrays of the strip elimination of a batched apply (see the section comment). ``nc`` components have
    strips, ``G`` is the largest number of strip rows (ring and theta index) of one of them."""

    left_r: jnp.ndarray          # (nb, N0, N0) radial factor of the interior rings, the identity on the strip rings
    right_r: object              # (nb, N0, N0) for the Laplacian, None for the mass
    lam_r: object                # (nb, N0) radial eigenvalues of the interior rings, zero on the strip rings
    lam_z: object                # (nc, N2) zeta eigenvalues of each component's model, None for the mass
    inv: object                  # (nc, N2, G, G) inverse Schur complement per zeta mode, (nc, G, G) for the mass
    slots: jnp.ndarray           # (nc, G) the strip rows in the flattened batch (nb N0 N1), out of range on padding
    c_idx: jnp.ndarray           # (nc, G, T) the interior rows coupled to each strip row, and the coupling A_SI
    c0: jnp.ndarray              # its part that does not depend on the zeta mode
    c1: object                   # its part proportional to the zeta eigenvalue, None for the mass
    e_idx: jnp.ndarray           # (n_e,) the interior rows coupled to a strip
    d_idx: jnp.ndarray           # (n_e, T') their strip rows, as component * G + strip row, and A_IS
    d0: jnp.ndarray
    d1: object
    d_slot: jnp.ndarray          # (n_e,) the component of each


def _interior_solve(p, s, v):
    """The interior solves of all blocks on ``v`` (shape ``(nb, N0, N1, N2)``, zeta already transformed). The
    entries on the strip rings come out meaningless."""
    w = jnp.einsum('bia,bjc,bacd->bijd', s.left_r, p.left[1], v)
    if s.lam_r is None:
        return w
    denom = s.lam_r[:, :, None, None] + p.lam[1][:, None, :, None] + p.lam[2][:, None, None, :]
    null = jnp.abs(denom) < sqrt_eps(6.7e-3) * jnp.max(jnp.abs(denom), axis=(1, 2, 3), keepdims=True)
    w = jnp.where(null, 0.0, w / jnp.where(null, 1.0, denom))
    return jnp.einsum('bia,bjc,bacd->bijd', s.right_r, p.right[1], w)


def _couple(s, y):
    """``A_SI y`` per zeta mode, ``y`` of shape ``(nb N0 N1, N2)``, as ``(nc, G, N2)``."""
    yg = y[s.c_idx]
    out = jnp.einsum('sgt,sgtk->sgk', s.c0, yg)
    if s.c1 is not None:
        out = out + s.lam_z[:, None, :] * jnp.einsum('sgt,sgtk->sgk', s.c1, yg)
    return out


def _lift(s, z):
    """``A_IS z`` on the interior rows coupled to a strip, ``z`` of shape ``(nc, G, N2)``."""
    zg = z.reshape(-1, z.shape[-1])[s.d_idx]
    out = jnp.einsum('et,etk->ek', s.d0, zg)
    if s.d1 is not None:
        out = out + s.lam_z[s.d_slot] * jnp.einsum('et,etk->ek', s.d1, zg)
    return out


def _strip_solve(p, xb):
    """The exact inverse of the blocks' model with the strips, on the padded batch ``xb``."""
    s = p.strips
    v = jnp.einsum('bkd,bijd->bijk', p.left[2], xb)
    flat = v.reshape(-1, v.shape[-1])
    r = flat.at[s.slots].get(mode='fill', fill_value=0.0) - _couple(s, _interior_solve(p, s, v).reshape(flat.shape))
    z = jnp.einsum('skgh,shk->sgk' if s.inv.ndim == 4 else 'sgh,shk->sgk', s.inv, r)
    flat = flat.at[s.e_idx].add(-_lift(s, z))
    y = _interior_solve(p, s, flat.reshape(v.shape)).reshape(flat.shape).at[s.slots].set(z, mode='drop')
    y = y.reshape(v.shape)
    return y if p.right is None else jnp.einsum('bkd,bijd->bijk', p.right[2], y)


def _schur_inverse(p, s, ass):
    """The inverse Schur complements ``(A_SS - A_SI A_II^-1 A_IS)^-1``, built with the interior solves of the
    apply, one strip row per column for all components and zeta modes at once. ``ass`` holds ``A_SS`` as
    ``(2, nc, G, G)``, its two parts as in :func:`_strip_tables`."""
    nc, G = s.slots.shape
    shape = (p.pre.shape[0],) + p.N

    def column(g):
        z = jnp.zeros((nc, G, p.N[2])).at[:, g, :].set(1.0)
        v = jnp.zeros((shape[0] * shape[1] * shape[2], p.N[2])).at[s.e_idx].set(_lift(s, z))
        return _couple(s, _interior_solve(p, s, v.reshape(shape)).reshape(v.shape))
    C = jnp.transpose(jax.lax.map(column, jnp.arange(G), batch_size=PROBE_BATCH), (1, 3, 2, 0))
    if s.c1 is None:
        S = ass[0] - C[:, 0]
    else:
        S = ass[0][:, None] + s.lam_z[:, :, None, None] * ass[1][:, None] - C
    # a Schur complement of a positive definite model
    w, V = jnp.linalg.eigh(0.5 * (S + jnp.swapaxes(S, -1, -2)))
    return (V / w[..., None, :]) @ jnp.swapaxes(V, -1, -2)


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
    strips: object               # the strip elimination (:class:`_Strips`), None on a uniform space
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


def _batch(specs, core, core_inv, n_full, X=None, strips=None):
    """The :class:`_Batched` of the blocks ``specs`` (dicts with ``rows``, ``vals``, ``shape``, ``pre``, ``post``,
    ``left`` and, for the Laplacian, ``right`` and ``lam``) and the axis rows ``core`` with their dense inverse,
    on a space of ``n_full`` entries, with the strip elimination ``strips``. With ``X`` (a reduced view's
    expansion, full <- reduced) the apply acts on the reduced vectors as ``X^T P X``."""
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
        return tuple(jnp.asarray(m) for m in arrays)
    return _Batched(jnp.asarray(in_idx), jnp.asarray(in_w), jnp.asarray(pre),
                    jnp.asarray(post), dev(left), dev(right) if laplacian else None,
                    dev(lam) if laplacian else None, jnp.asarray(x_cols[core]), jnp.asarray(x_vals[core]),
                    jnp.asarray(core_inv), jnp.asarray(source[o_cols, 0].astype(np.int64)),
                    jnp.asarray(o_vals * source[o_cols, 1]), strips, N)


@eqx.filter_jit
def _apply_batched(p, x, core_inv, alpha, shift):
    """Apply the :class:`_Batched` ``p`` to ``x``. ``core_inv`` replaces ``p.core_inv`` when given. ``alpha``
    (per block and axis) weighs the Laplacian's Kronecker terms, ``None`` for all ones. ``shift`` is ``1/eps`` of
    the shifted-stiffness preconditioner and ``None`` otherwise. It runs in :data:`~mrx.precision.PRECOND_DTYPE`
    and returns the precision of ``x``."""
    dtype, x = x.dtype, x.astype(PRECOND_DTYPE)
    core_inv, alpha, shift = (None if v is None else jnp.asarray(v, PRECOND_DTYPE) for v in (core_inv, alpha, shift))
    nb = p.pre.shape[0]
    xb = jnp.sum(p.in_w * x[p.in_idx], axis=-1).reshape((nb,) + p.N) * p.pre
    if p.strips is not None and shift is None:
        y = _strip_solve(p, xb)
    else:
        y = _block_solve(p, xb, alpha, shift)
    y = y * p.post
    core = (p.core_inv if core_inv is None else core_inv) @ jnp.sum(p.core_w * x[p.core_idx], axis=-1)
    z = jnp.concatenate([y.reshape(-1), core])
    return jnp.sum(p.out_w * z[p.out_idx], axis=-1).astype(dtype)


def _block_solve(p, xb, alpha, shift):
    """Every block of the batch ``xb`` inverted on its own, by fast diagonalisation."""
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
    return y


class _Atom:
    """What the mass and the Laplacian preconditioner share: the split into axis and off-axis
    rows, the dense axis block, and the apply."""

    def _prologue(self, seq, k):
        """Return the axis rows, the off-axis blocks and ``probe``. ``probe(apply)`` is the dense
        block of ``apply`` on the axis rows."""
        self.parity_projector = seq.free_projector(k)   # None on a full-period sequence
        core, raw_blocks = _tensor_blocks(seq, k)
        size, split = seq.n(k), _parity_split(seq, k)
        self._n = size
        return core, raw_blocks, lambda apply: _probe_rows(apply, size, core, split)

    def _install(self, core, specs, core_inv, comps):
        # device arrays, not NumPy: NumPy attributes are static and a new geometry would recompile
        self._core = jnp.asarray(core)
        self._specs = [dict(s, rows=jnp.asarray(s["rows"]), vals=jnp.asarray(s["vals"])) for s in specs]
        payload = _batch(self._specs, core, core_inv, self._n)
        if comps:
            payload = eqx.tree_at(lambda q: q.strips, payload, _build_strips(payload, specs, comps),
                                  is_leaf=lambda x: x is None)
        self._payload = to_precond(payload)

    def reduced(self, X):
        """The batched payload of ``X^T P X`` for a reduced view with the expansion ``X``."""
        return to_precond(_batch(self._specs, self._core, self._payload.core_inv, self._n, X, self._payload.strips))

    def apply_reduced(self, payload, x):
        """``X^T P X x`` for the payload of :meth:`reduced`."""
        return _apply_batched(payload, jnp.asarray(x), None, None, None)

    def apply(self, x):
        """Apply the preconditioner to a coefficient vector of the free degrees of freedom."""
        p = self._payload
        if self.parity_projector is None:
            return _apply_batched(p, jnp.asarray(x), None, None, None)
        return self.parity_projector(lambda v: _apply_batched(p, v, None, None, None), jnp.asarray(x))


@register_arrays
class MetricLumpingLaplacian(_Atom):
    """Approximate inverse of the Hodge Laplacian ``L_k`` on the space of ``seq``.

    Built by ``DeRhamSequence.build_preconditioners`` for the installed geometry. Apply it with
    :meth:`apply`. Needs at least two radial elements (``n >= p + 2``).
    """

    def __init__(self, seq, operators, k):
        from mrx.operators import apply_laplacian_approx   # imports this module
        core, raw_blocks, probe = self._prologue(seq, k)
        n_strip = _strip_rings(seq, k, raw_blocks)
        specs, strong, comps = [], [], []
        for c, members in _by_component(raw_blocks):
            a0 = raw_blocks[members[0]][3][0]
            n_rings = sum(raw_blocks[b][3][1] for b in members)
            masses, stiffs = component_factors(seq, k, c, (a0, n_rings))
            mr_full, kr_full = masses[0], stiffs[0]
            if len(members) > 1:
                kts, czs, kz, mrs = _band_factors(seq, k, c, members, raw_blocks, a0, n_rings)
            else:
                kts, czs, kz, mrs = [stiffs[1]], [1.0], stiffs[2], [mr_full]
            # one zeta pair for the component: the strips need a common zeta eigenbasis
            vz, lz = _simultaneous_diagonalize_pair(masses[2], kz)
            # the model: K_r (x) M_t + sum over bands of M_r (x) K_t, plus the zeta eigenvalue times M_r (x) M_t
            mt = np.asarray(masses[1])
            terms = ([(np.asarray(kr_full), mt, 0)] + [(np.asarray(m), np.asarray(kt), 0) for m, kt in zip(mrs, kts)]
                     + [(np.asarray(cz * m), mt, 1) for m, cz in zip(mrs, czs)])
            blocks = []
            for b, kt, cz in zip(members, kts, czs):
                _, rows, vals, (r0, nr), shape, R = raw_blocks[b]
                cut = slice(r0 - a0, r0 - a0 + nr)
                mr, kr = mr_full[cut, cut], kr_full[cut, cut]
                # the theta factors of a graded ring: the Galerkin matrices of its coarse functions
                (vr, lr), (vt, lt) = (_simultaneous_diagonalize_pair(mr, kr),
                                      _simultaneous_diagonalize_pair(R.T @ masses[1] @ R, R.T @ kt @ R))
                # D_i is a ratio of two positive integrals, and so is the angular profile
                D = component_diagonal(seq, k, c, R)[r0:r0 + nr]
                if k < 3:
                    D = D * _angular_profile(seq, k, c, R, terms, (masses[2], kz), D, r0, a0)
                dscale = 1.0 / jnp.sqrt(D)
                ni = nr - n_strip[b]
                vi, li = (vr, lr) if ni == nr else (_simultaneous_diagonalize_pair(mr[:ni, :ni], kr[:ni, :ni])
                                                    if ni else (jnp.zeros((0, 0)), jnp.zeros(0)))
                specs.append(dict(rows=rows, vals=vals, shape=shape, pre=dscale, post=dscale,
                                  left=(vr.T, vt.T, vz.T), right=(vr, vt, vz), lam=(lr, lt, cz * lz),
                                  interior=(vi.T, vi, li)))
                # the strong half S_k: the Kronecker terms of the primal axes
                strong.append([0.0 if a in DERIV_AXES[k][c] else 1.0 for a in range(3)])
                blocks.append((b, r0 - a0, nr, n_strip[b], np.asarray(R)))
            if any(n_strip[b] for b in members):
                comps.append((blocks, terms, lz))
        self._strong = jnp.asarray(strong)
        core_inv = _dense_symmetric_inverse(
            probe(lambda x: apply_laplacian_approx(seq, operators, x, k)), f"the Laplacian preconditioner, k = {k}")
        # (M_k, S_k) on the core, diagonalised once for the shifted apply's (M + eps S)^-1
        mass_core = operators.mass_lumping[(k, seq.dirichlet)].core_block    # probed by the mass preconditioner
        stiffness_core = probe(lambda x: seq.S[k] @ x)
        self._core_pair = (_simultaneous_diagonalize_pair(mass_core, stiffness_core) if core.size
                           else (mass_core, jnp.zeros(0)))
        self._install(core, specs, core_inv, comps)

    def shifted_stiffness_apply(self, eps, payload=None):
        """Return ``x -> (M_k + eps S_k)^{-1} x`` in the same approximation, the preconditioner of
        the shifted Laplacian solves. ``eps`` may be a traced value, so changing it does not
        rebuild or recompile anything. ``payload`` is a reduced view's :meth:`reduced` payload. On a graded
        space the theta levels stay uncoupled here, because their Schur complement would depend on ``eps``."""
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

    def __init__(self, seq, k):
        core, raw_blocks, probe = self._prologue(seq, k)
        mass_1d = _kron_mass_model_1d(seq, k)
        n_strip = _strip_rings(seq, k, raw_blocks)
        specs, comps = [], []
        for c, members in _by_component(raw_blocks):
            blocks = []
            for b in members:
                _, rows, vals, (r0, nr), shape, R = raw_blocks[b]
                m = [mass_1d[c][0][r0:r0 + nr, r0:r0 + nr], R.T @ mass_1d[c][1] @ R, mass_1d[c][2]]
                # Lam: the model has exactly the diagonal of M_k
                kron_diag = jnp.einsum('i,j,l->ijl', *(jnp.diag(f) for f in m))
                lam = jnp.sqrt(build_mass_diagonal(seq, k, c, R)[r0:r0 + nr] / kron_diag)
                left = tuple(jnp.linalg.inv(f) for f in m)
                ni = nr - n_strip[b]
                interior = left[0] if ni == nr else jnp.linalg.inv(m[0][:ni, :ni]) if ni else jnp.zeros((0, 0))
                specs.append(dict(rows=rows, vals=vals, shape=shape, pre=1.0 / lam, post=1.0 / lam, left=left,
                                  interior=(interior, None, None)))
                blocks.append((b, r0, nr, n_strip[b], np.asarray(R)))
            if any(n_strip[b] for b in members):
                comps.append((blocks, [(np.asarray(mass_1d[c][0]), np.asarray(mass_1d[c][1]), 0)], None))
        #: the mass matrix on the axis rows, which the Laplacian preconditioner of the same space reuses
        self.core_block = probe(lambda x: seq.M[k] @ x)
        core_inv = _dense_symmetric_inverse(self.core_block, f"the mass preconditioner, k = {k}")
        self._install(core, specs, core_inv, comps)


class ReducedAtom(eqx.Module):
    """A preconditioner ``P`` of the half-period sequence restricted to its even or odd part, as
    ``X^T P X`` where ``X`` expands a vector of that part to the full space. The parity views
    of the sequence hold these, built from the base sequence's preconditioners. Each preconditioner
    prepares its own payload for ``X``: the atoms compose ``X`` into the gathers of the batched apply,
    the slab mass model (:class:`mrx.slab_mass.SlabMass`) applies ``X`` and ``X^T`` around its sweep."""
    atom: object
    payload: object

    def __init__(self, atom, X):
        self.atom = atom
        self.payload = atom.reduced(X)

    def apply(self, x):
        return self.atom.apply_reduced(self.payload, x)

    def shifted_stiffness_apply(self, eps):
        """The restricted :meth:`MetricLumpingLaplacian.shifted_stiffness_apply`, ``x -> X^T (M + eps S)^{-1} X x``."""
        return self.atom.shifted_stiffness_apply(eps, self.payload)
