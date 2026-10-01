"""Matrix-free mass matrices of the k-forms and their diagonals.

The mass matrix of the k-forms is ``M_k = int Lambda^k . W_k . Lambda^k`` over the logical domain,
with the geometric weight ``W_0 = J``, ``W_1 = J G^{-1}``, ``W_2 = G / J`` and ``W_3 = 1 / J``
(``G = DPhi^T DPhi`` the metric tensor and ``J = det DPhi`` the Jacobian determinant of the
logical-to-physical map ``Phi``). No matrix is stored.
Every product ``M_k x`` is computed from the basis values at the quadrature points. The projection
masses ``int Lambda^{k_row} . Lambda^{k_col}`` between different forms work the same way with the
weight 1. All of these act on the full tensor-product spline coefficients, before the boundary and
polar constraints are applied. The callers in :mod:`mrx.operators` add those constraints.

The work is split into a static part and a geometry part. :func:`mass_plan` and
:func:`projection_plan` hold the basis values and are built once with the de Rham sequence.
:func:`attach_weights` computes the weights from a geometry and stores them on it, so a new map
``Phi`` only recomputes the weights and reuses the compiled code. :func:`sumfact_apply` applies a plan with its
weights, and :func:`build_mass_diagonal` returns ``diag(M_k)``.
"""

import functools
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from mrx.spline_bases import evaluate_basis_local


def _elem_counts(seq):
    """The number of elements per axis covered by the quadrature and the Gauss points per element."""
    q = seq.quad
    return q.ne_x, q.ne_y, q.ne_z, q.nx // q.ne_x, q.ny // q.ne_y, q.nz // q.ne_z


def _split_field(field_flat, ne_x, ne_y, ne_z, qx, qy, qz):
    """Reshape a flat (r-major) quadrature field to shape ``(ne_x, ne_y, ne_z, qx, qy, qz, ...)``.

    Trailing axes, such as the ``(3, 3)`` of the metric, are kept.
    """
    trailing = tuple(range(6, 6 + field_flat.ndim - 1))
    f = field_flat.reshape(ne_x, qx, ne_y, qy, ne_z, qz, *field_flat.shape[1:])
    return f.transpose(0, 2, 4, 1, 3, 5, *trailing)


def _element_layout(seq):
    """Return ``(split, gauss)``: the per-element reshape of this sequence and the 3-D Gauss weights."""
    ne_x, ne_y, ne_z, qx, qy, qz = _elem_counts(seq)
    wx = seq.quad.w_x.reshape(ne_x, qx)
    wy = seq.quad.w_y.reshape(ne_y, qy)
    wz = seq.quad.w_z.reshape(ne_z, qz)
    gauss = (wx[:, None, None, :, None, None]
             * wy[None, :, None, None, :, None]
             * wz[None, None, :, None, None, :])

    def split(field_flat):
        return _split_field(field_flat, ne_x, ne_y, ne_z, qx, qy, qz)

    return split, gauss


def _shift_plan_axis(g, S, axis):
    """Return ``(ne, nloc, S)`` for the element-to-DoF map ``g[e, l] == (e + l) % S`` of one axis.

    Raises ``ValueError`` if ``g`` is not of that form.
    """
    g = np.asarray(g)
    ne, nloc = g.shape
    e = np.arange(ne)[:, None]
    lo = np.arange(nloc)[None, :]
    if not np.array_equal(g, (e + lo) % int(S)):
        raise ValueError(
            f"axis {axis}: element-to-DoF map is not the shift (e + l) % {S} "
            f"for ne={ne}, nloc={nloc}")
    return (int(ne), int(nloc), int(S))


def _shift_plan(gx, gy, gz, shape):
    """The shift plans of the three axes of one component."""
    return tuple(_shift_plan_axis(g, s, axis)
                 for g, s, axis in zip((gx, gy, gz), shape, "xyz"))


def _structured_accumulate(y, plan):
    """Sum element contributions ``(ne_x, ne_y, ne_z, nloc_x, nloc_y, nloc_z)`` into the DoF grid.

    The result has shape ``(S_x, S_y, S_z)`` with
    ``out[i, j, k] = sum_{lx, ly, lz} y[i-lx, j-ly, k-lz, lx, ly, lz]``, indices taken modulo
    the axis size.
    """
    (ne_x, nl_x, S_x), (ne_y, nl_y, S_y), (ne_z, nl_z, S_z) = plan

    def accumulate(a, axis, ne, nloc, S):
        """Sum the ``nloc`` local slices of ``a`` (axis 3), each padded to ``S`` and shifted along ``axis``."""
        total = None
        for il in range(nloc):
            slab = jnp.take(a, il, axis=3)
            if S != ne:
                pad = [(0, 0)] * slab.ndim
                pad[axis] = (0, S - ne)
                slab = jnp.pad(slab, pad)
            slab = jnp.roll(slab, il, axis=axis)
            total = slab if total is None else total + slab
        return total

    # (nex,ney,nez,nlx,nly,nlz) -> (Sx,ney,nez,nly,nlz)
    a = accumulate(y, 0, ne_x, nl_x, S_x)
    # (Sx,ney,nez,nly,nlz) -> (Sx,Sy,nez,nlz)
    a = accumulate(a, 1, ne_y, nl_y, S_y)
    # (Sx,Sy,nez,nlz) -> (Sx,Sy,Sz)
    a = accumulate(a, 2, ne_z, nl_z, S_z)
    return a


def _structured_gather(x_flat, plan):
    """Read the element-local coefficients ``x_local[e, l] = x[(e + l) mod S]`` on every axis.

    This is the transpose of :func:`_structured_accumulate`.
    """
    (ne_x, nl_x, S_x), (ne_y, nl_y, S_y), (ne_z, nl_z, S_z) = plan
    a = x_flat.reshape(S_x, S_y, S_z)
    a = jnp.stack([jnp.roll(a, -lx, axis=0)[:ne_x] for lx in range(nl_x)],
                  axis=3)
    a = jnp.stack([jnp.roll(a, -ly, axis=1)[:, :ne_y] for ly in range(nl_y)],
                  axis=4)
    a = jnp.stack([jnp.roll(a, -lz, axis=2)[:, :, :ne_z] for lz in range(nl_z)],
                  axis=5)
    return a


def _fuse_yz(By, Bz):
    """The product table ``Byz[y, z, (r, s), (d, f)] = By[y, r, d] * Bz[z, s, f]`` of the y and z bases."""
    ne_y, qy, nly = By.shape
    ne_z, qz, nlz = Bz.shape
    return jnp.einsum('yrd,zsf->yzrsdf', By, Bz).reshape(
        ne_y, ne_z, qy * qz, nly * nlz)


def _to_quadrature(Bvals, x_local):
    """Evaluate one component at the quadrature points of every element from its element-local coefficients."""
    Bx, By, Bz, Byz = Bvals
    ne_x, qx, _ = Bx.shape
    ne_y, qy, nly = By.shape
    ne_z, qz, nlz = Bz.shape
    t1 = jnp.einsum('xqb,xyzbdf->xyzqdf', Bx, x_local)
    t1 = t1.reshape(ne_x, ne_y, ne_z, qx, nly * nlz)
    u = jnp.einsum('yzQD,xyzqD->xyzqQ', Byz, t1)
    return u.reshape(ne_x, ne_y, ne_z, qx, qy, qz)


def _from_quadrature(Bvals, u):
    """Test a quadrature-point field against the local basis functions of each element.

    This is the transpose of :func:`_to_quadrature`. The Gauss weights must already be in ``u``.
    """
    Bx, By, Bz, Byz = Bvals
    ne_x, qx, _ = Bx.shape
    ne_y, qy, nly = By.shape
    ne_z, qz, nlz = Bz.shape
    v = u.reshape(ne_x, ne_y, ne_z, qx, qy * qz)
    s1 = jnp.einsum('yzQD,xyzqQ->xyzqD', Byz, v)
    s1 = s1.reshape(ne_x, ne_y, ne_z, qx, nly, nlz)
    return jnp.einsum('xqa,xyzqdf->xyzadf', Bx, s1)


def _form_bases(seq, k):
    """Return the k-form basis and, per component, the 1-D basis values and DoF indices ``(Bx, gx, By, gy, Bz, gz)``."""
    form = getattr(seq, f"basis_{k}")
    ne_x, ne_y, ne_z, qx, qy, qz = _elem_counts(seq)
    cache: dict[int, tuple] = {}

    def local_eval(basis, x_q, q):
        key = id(basis)
        if key not in cache:
            cache[key] = evaluate_basis_local(basis, x_q, q)
        return cache[key]

    comp = []
    for c in range(len(form.shape)):
        d = form.derivative_axes(c)
        b = [(form.dLambda if a in d else form.Lambda)[a] for a in range(3)]
        Bx, gx = local_eval(b[0], seq.quad.x_x, qx)
        By, gy = local_eval(b[1], seq.quad.x_y, qy)
        Bz, gz = local_eval(b[2], seq.quad.x_z, qz)
        comp.append((Bx, gx, By, gy, Bz, gz))
    return form, comp


def _mass_structure(k):
    """Return ``(pairs, cols)``: the ``(row, col)`` component pairs of ``M_k`` and the weight each one uses.

    For k = 0 and 3 there is one pair. For k = 1 and 2 all nine pairs appear, and ``(i, j)`` and
    ``(j, i)`` share one of the six entries of the symmetric weight returned by :func:`_mass_weight`.
    """
    if k in (0, 3):
        return ((0, 0),), (0,)
    unique = [(i, j) for i in range(3) for j in range(i, 3)]
    pairs = tuple((i, j) for i in range(3) for j in range(3))
    return pairs, tuple(unique.index((min(i, j), max(i, j))) for i, j in pairs)


def _mass_weight(k, metric, metric_inv, jac):
    """The distinct pointwise weights of ``M_k``: ``J`` for k=0, the entries ``i <= j`` of
    ``J G^{-1}`` for k=1 and of ``G / J`` for k=2, and ``1 / J`` for k=3."""
    if k == 0:
        return (jac,)
    if k == 3:
        return (1.0 / jac,)
    return tuple(metric_inv[..., i, j] * jac if k == 1 else metric[..., i, j] / jac
                 for i in range(3) for j in range(i, 3))


def _reference_structure(n_comp):
    """Return ``(pairs, cols)`` of a projection mass: only the matching components, all with weight 1."""
    return tuple((c, c) for c in range(n_comp)), (0,) * n_comp


def build_mass_diagonal(seq, k):
    """Return ``diag(M_k)`` on the installed geometry, for the full tensor-product coefficients.

    The components are concatenated. The diagonal is computed directly, without applying ``M_k``.
    """
    g = seq.geometry
    split, gauss = _element_layout(seq)
    form, comp = _form_bases(seq, k)
    unique = _mass_weight(k, g.metric_jkl, g.metric_inv_jkl, g.jacobian_j)
    pairs, cols = _mass_structure(k)
    shapes = form.shape

    parts = []
    for c in range(len(comp)):
        Wf = split(unique[cols[pairs.index((c, c))]]) * gauss
        Bx, By, Bz = comp[c][0], comp[c][2], comp[c][4]
        # Squared basis tables: the row and column bases coincide on the diagonal.
        t1 = jnp.einsum('xqa,xyzqrs->xyzars', Bx * Bx, Wf)
        t2 = jnp.einsum('yrb,xyzars->xyzabs', By * By, t1)
        d_local = jnp.einsum('zse,xyzabs->xyzabe', Bz * Bz, t2)
        plan = _shift_plan(comp[c][1], comp[c][3], comp[c][5], shapes[c])
        parts.append(_structured_accumulate(d_local, plan).reshape(-1))
    d = jnp.concatenate(parts)
    if seq.half_period:
        # d holds twice the half-period integrals. An entry and its mirror image each hold
        # one half of the full integral, so their unsigned mean is the full diagonal.
        from mrx.symmetry import symmetrize  # noqa: PLC0415
        d = symmetrize(d, seq.reflection_plan[k], 1, signed=False)
    return d


class SumfactPlan(NamedTuple):
    """The geometry-independent part of a mass or projection apply.

    It holds the basis values at the quadrature points and the index structure of the row and
    column forms. The sequence builds one per mass matrix (:func:`mass_plan`) and one per
    projection (:func:`projection_plan`). The geometric weights are stored on the geometry
    instead (:func:`attach_weights`). The index fields are static arguments of the compiled
    apply, so a new geometry reuses the compiled code, while a new plan compiles it again.
    """

    Bvals_r: tuple
    Bvals_c: tuple
    gather_plans: tuple
    shift_plans: tuple
    pairs: tuple
    cols: tuple
    starts_c: tuple


def build_sumfact_plan(seq, k_row, k_col, pairs, cols):
    """The :class:`SumfactPlan` of ``int Lambda^{k_row} . W . Lambda^{k_col}``.

    ``pairs`` lists the ``(row, col)`` component pairs where ``W`` is nonzero, and ``cols`` gives
    the index of the weight array each pair uses.
    """
    form_r, comp_r = _form_bases(seq, k_row)
    form_c, comp_c = _form_bases(seq, k_col)
    n_r, n_c = len(comp_r), len(comp_c)

    def starts(form, n_comp):
        out = [0]
        for c in range(n_comp):
            out.append(out[-1] + int(np.prod(form.shape[c])))
        return tuple(out)

    # The y-z product table is built here, not in the kernel, so solver loops do not rebuild it.
    Bvals_r = tuple((c[0], c[2], c[4], _fuse_yz(c[2], c[4])) for c in comp_r)
    Bvals_c = tuple((c[0], c[2], c[4], _fuse_yz(c[2], c[4])) for c in comp_c)
    gather_plans = tuple(_shift_plan(comp_c[c][1], comp_c[c][3], comp_c[c][5],
                                     form_c.shape[c]) for c in range(n_c))
    shift_plans = tuple(_shift_plan(comp_r[c][1], comp_r[c][3], comp_r[c][5],
                                    form_r.shape[c]) for c in range(n_r))
    return SumfactPlan(Bvals_r, Bvals_c, gather_plans, shift_plans,
                       tuple(pairs), tuple(cols), starts(form_c, n_c))


def mass_plan(seq, k):
    """The :class:`SumfactPlan` of ``M_k``."""
    return build_sumfact_plan(seq, k, k, *_mass_structure(k))


def projection_plan(seq, k_row, k_col):
    """The :class:`SumfactPlan` of the projection mass ``int Lambda^{k_row} . Lambda^{k_col}`` (weight 1).

    It maps ``k_col``-form coefficients to ``k_row``-form moments.
    """
    return build_sumfact_plan(seq, k_row, k_col, *_reference_structure(1 if k_row in (0, 3) else 3))


def element_weights(seq, unique):
    """The weight arrays reshaped per element and multiplied by the Gauss weights."""
    split, gauss = _element_layout(seq)
    return tuple(split(w) * gauss for w in unique)


def attach_weights(seq, geometry):
    """Return ``geometry`` with the weights of all mass and projection matrices attached.

    ``mass_weights[k]`` holds the weights of ``M_k`` and ``reference_weights`` those of the
    projection masses. The sequence calls this when a map is installed. Code that installs a
    geometry by hand, such as the shape-derivative code, must call it too.
    """
    metric, metric_inv, jac = geometry.metric_jkl, geometry.metric_inv_jkl, geometry.jacobian_j
    mass_weights = {k: element_weights(seq, _mass_weight(k, metric, metric_inv, jac))
                    for k in range(4)}
    reference = element_weights(seq, (jnp.ones_like(jac),))
    return eqx.tree_at(lambda g: (g.mass_weights, g.reference_weights), geometry,
                       (mass_weights, reference), is_leaf=lambda x: x is None)


def sumfact_apply(plan, weights, x):
    """Apply ``x -> int Lambda_row . W . Lambda_col x`` given a :class:`SumfactPlan` and its weights.

    ``x`` holds the full tensor-product coefficients. Boundary and polar constraints are left to
    the caller. On a half-period sequence the result is twice the integral over the half period.
    """
    return _sumfact_kernel(x, plan.Bvals_r, plan.Bvals_c, weights,
                           pairs=plan.pairs, cols=plan.cols,
                           starts_c=plan.starts_c,
                           shift_plans=plan.shift_plans,
                           gather_plans=plan.gather_plans)


@functools.partial(jax.jit, static_argnames=("pairs", "cols", "starts_c",
                                             "shift_plans", "gather_plans"))
def _sumfact_kernel(x, Bvals_r, Bvals_c, Ws, *,
                    pairs, cols, starts_c, shift_plans, gather_plans):
    """The compiled apply. It compiles once per plan, and a new geometry of the same shapes reuses it."""
    W = {pair: Ws[c] for pair, c in zip(pairs, cols)}
    n_c, n_r = len(Bvals_c), len(Bvals_r)
    u = [_to_quadrature(
            Bvals_c[c],
            _structured_gather(x[starts_c[c]:starts_c[c + 1]], gather_plans[c]))
         for c in range(n_c)]
    y_parts = []
    for cr in range(n_r):
        v = sum(W[(cr, cc)] * u[cc] for cc in range(n_c) if (cr, cc) in pairs)
        y_local = _from_quadrature(Bvals_r[cr], v)
        y_parts.append(
            _structured_accumulate(y_local, shift_plans[cr]).reshape(-1))
    return jnp.concatenate(y_parts)
