"""The mass and projection operators as sum-factorised applies, and their closed-form diagonals.

No mass matrix is stored. Every mass-like operator ``M = int Lambda_row . W
. Lambda_col`` is applied per element by three 1-D contractions to the
quadrature points, a pointwise multiply by the weight, and three 1-D
contractions back (:func:`_sumfact_kernel`, one compiled executable per
operator shape). The applies act in the raw (unextracted, periodic) DoF
space; the polar extraction ``E (.) E^T`` is applied by the caller
(:mod:`mrx.operators`), and ``DeRhamSequence.set_geometry`` builds them.

The weights are elementwise products of the stored geometry (``G = DF^T DF``,
``G^{-1}`` and ``J = det DF`` per quadrature point), formed once per geometry
and memoised in the element layout:

* k=0: ``W = J``          (scalar)
* k=1: ``W = G^{-1} J``   (3x3, derivative basis on axis c)
* k=2: ``W = G / J``      (3x3, primal basis on axis c)
* k=3: ``W = 1/J``        (scalar, derivative basis on all axes)

The projection masses between different form degrees (``P_21`` etc.) use the
same kernel with the reference-domain weight ``W = I``. The quadrature
weights are folded in per axis via the 1-D Gauss weights.

:func:`build_mass_diagonal`, the closed-form ``diag(M_k)`` by the same sum
factorisation, is the diagonal the metric-lumping mass atom is scaled by.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np

from mrx.spline_bases import evaluate_basis_local

__all__ = [
    "build_mass_diagonal",
    "build_matrixfree_mass_apply",
    "build_matrixfree_projection_apply",
]


def _elem_counts(seq):
    """(ne_x, ne_y, ne_z, qx, qy, qz) derived from the primal (k=0) basis."""
    nx, ny, nz = seq.quad.nx, seq.quad.ny, seq.quad.nz
    b0 = seq.basis_0.Λ
    ne_x = b0[0].n if b0[0].type == "periodic" else b0[0].n - b0[0].p
    ne_y = b0[1].n if b0[1].type == "periodic" else b0[1].n - b0[1].p
    ne_z = b0[2].n if b0[2].type == "periodic" else b0[2].n - b0[2].p
    return ne_x, ne_y, ne_z, nx // ne_x, ny // ne_y, nz // ne_z


def _split_field(field_flat, nx, ny, nz, ne_x, ne_y, ne_z, qx, qy, qz):
    """Reshape a flat quad field (meshgrid 'xy' layout) to per-element blocks.

    Returns shape ``(ne_x, ne_y, ne_z, qx, qy, qz, *trailing)``; trailing axes
    (e.g. the ``(3, 3)`` of the metric) ride along. One reshape and one transpose,
    so it fuses into the consumer inside a jit.
    """
    del nx, ny, nz
    trailing = tuple(range(6, 6 + field_flat.ndim - 1))
    f = field_flat.reshape(ne_y, qy, ne_x, qx, ne_z, qz, *field_flat.shape[1:])
    return f.transpose(2, 0, 4, 3, 1, 5, *trailing)


def _element_layout(seq):
    """Return ``(split, gauss)``: the flat->element reshape and the Gauss weights.

    ``split(field_flat)`` is :func:`_split_field` with this sequence's counts
    bound; ``gauss`` is the ``(ne_x, ne_y, ne_z, qx, qy, qz)`` outer product of
    the per-axis Gauss weights.
    """
    nx, ny, nz = seq.quad.nx, seq.quad.ny, seq.quad.nz
    ne_x, ne_y, ne_z, qx, qy, qz = _elem_counts(seq)
    wx = seq.quad.w_x.reshape(ne_x, qx)
    wy = seq.quad.w_y.reshape(ne_y, qy)
    wz = seq.quad.w_z.reshape(ne_z, qz)
    gauss = (wx[:, None, None, :, None, None]
             * wy[None, :, None, None, :, None]
             * wz[None, None, :, None, None, :])

    def split(field_flat):
        return _split_field(field_flat, nx, ny, nz, ne_x, ne_y, ne_z, qx, qy, qz)

    return split, gauss


# --------------------------------------------------------------------------- #
# Component basis selectors for vectorial forms
# --------------------------------------------------------------------------- #
def _component_axis_bases_k1(form, c):
    """k=1 component ``c``: derivative basis on axis ``c``, primal elsewhere."""
    bases = [form.Λ[0], form.Λ[1], form.Λ[2]]
    bases[c] = form.dΛ[c]
    return bases


def _component_axis_bases_k2(form, c):
    """k=2 component ``c``: primal basis on axis ``c``, derivative elsewhere."""
    bases = [form.dΛ[0], form.dΛ[1], form.dΛ[2]]
    bases[c] = form.Λ[c]
    return bases


# --------------------------------------------------------------------------- #
# Matrix-free (sum-factorized) mass apply
# --------------------------------------------------------------------------- #
# The functions below apply ``M @ x`` in the raw tensor-product DOF space
# *without ever materializing* ``M``: the element-local sum factorization is
# folded against the input vector instead of forming element blocks.
# Transient memory is O(n^3 (p+1)^2 q) instead of the O(n^3 (p+1)^6) of a
# stored matrix.
def _bases_for_form(seq, form, comp_bases_fn, n_comp):
    """Evaluate the 1D bases (values + global DOF ids) for each component."""
    ne_x, ne_y, ne_z, qx, qy, qz = _elem_counts(seq)
    cache: dict[int, tuple] = {}

    def local_eval(basis, x_q, q):
        key = id(basis)
        if key not in cache:
            cache[key] = evaluate_basis_local(basis, x_q, q)
        return cache[key]

    comp = []
    for c in range(n_comp):
        b = comp_bases_fn(form, c)
        Bx, gx = local_eval(b[0], seq.quad.x_x, qx)
        By, gy = local_eval(b[1], seq.quad.x_y, qy)
        Bz, gz = local_eval(b[2], seq.quad.x_z, qz)
        comp.append((Bx, gx, By, gy, Bz, gz))
    return comp


def _flat_dof_plan(gx, gy, gz, shape):
    """Static flat index plan into a component's flattened DOF grid.

    ``gx (ne_x, nloc_x)``, ``gy``, ``gz`` are the per-axis global DOF ids of
    each element's local DOFs. Returns a single ``int32`` array of shape
    ``(ne_x, ne_y, ne_z, nloc_x, nloc_y, nloc_z)`` whose entries are the flat
    indices into a ``shape``-grid reshaped to 1D. Built once on the host so the
    matvec needs no index arithmetic -- just one gather / one ``segment_sum``.
    """
    Sx, Sy, Sz = (int(s) for s in shape)
    gx = np.asarray(gx)
    gy = np.asarray(gy)
    gz = np.asarray(gz)
    idx = (gx[:, None, None, :, None, None] * (Sy * Sz)
           + gy[None, :, None, None, :, None] * Sz
           + gz[None, None, :, None, None, :])
    return jnp.asarray(idx.astype(np.int32))


def _to_quadrature(Bvals, x_flat, gather_idx):
    """Column half of :func:`_elem_block_mixed` folded against a vector.

    Gathers the element-local input with the precomputed flat index plan (no
    index arithmetic in the matvec) and evaluates the component's field at the
    element quadrature points, ``(ne_x, ne_y, ne_z, qx, qy, qz)``.
    """
    Bx, By, Bz = Bvals
    x_local = x_flat[gather_idx]  # (ne_x,ne_y,ne_z,nxc,nyc,nzc)
    t1 = jnp.einsum('xqb,xyzbdf->xyzqdf', Bx, x_local)
    t2 = jnp.einsum('yrd,xyzqdf->xyzqrf', By, t1)
    return jnp.einsum('zsf,xyzqrf->xyzqrs', Bz, t2)


def _from_quadrature(Bvals, u):
    """Row half of :func:`_elem_block_mixed`: test a quadrature-point field
    (Gauss weights already folded in) against the element-local row basis."""
    Bx, By, Bz = Bvals
    s1 = jnp.einsum('xqa,xyzqrs->xyzars', Bx, u)
    s2 = jnp.einsum('yrc,xyzars->xyzacs', By, s1)
    return jnp.einsum('zse,xyzacs->xyzace', Bz, s2)


def _form_bases(seq, k):
    """Return ``(form, comp, n_comp)``: the k-form and its per-component 1D tables.

    Both the matvecs and :func:`build_mass_diagonal` go through here so the
    diagonal is derived from *the same tables the solver applies*. The removed
    ``diag_EAET_direct`` route recomputed its own tables and drifted from the
    matvec; that failure mode is structural, so the plan is shared rather than
    mirrored.
    """
    if k == 0:
        form = seq.basis_0
        return form, _bases_for_form(
            seq, form, lambda f, c: [f.Λ[0], f.Λ[1], f.Λ[2]], 1), 1
    if k == 3:
        form = seq.basis_3
        return form, _bases_for_form(
            seq, form, lambda f, c: [f.dΛ[0], f.dΛ[1], f.dΛ[2]], 1), 1
    if k == 1:
        form = seq.basis_1
        return form, _bases_for_form(seq, form, _component_axis_bases_k1, 3), 3
    if k == 2:
        form = seq.basis_2
        return form, _bases_for_form(seq, form, _component_axis_bases_k2, 3), 3
    raise ValueError("k must be 0, 1, 2 or 3")


def _mass_weight(k, metric, metric_inv, jac):
    """Pointwise ``M_k`` weight per component pair, from the stored geometry.

    ``metric`` / ``metric_inv`` are ``(..., 3, 3)`` and ``jac`` ``(...)`` in
    any layout; returns ``{(cr, cc): (...) array}``.  Every entry is ONE
    elementwise product (or quotient) of stored arrays -- no adjugate, no
    3x3 algebra:

    * k=0: ``J``
    * k=1: ``J g^{-1}``
    * k=2: ``g / J``
    * k=3: ``1 / J``

    The symmetric pairs of k=1, 2 share one array, which the apply relies on
    to form six weights instead of nine.
    """
    if k == 0:
        return {(0, 0): jac}
    if k == 3:
        return {(0, 0): 1.0 / jac}
    if k not in (1, 2):
        raise ValueError("k must be 0, 1, 2 or 3")
    w = {}
    for i in range(3):
        for j in range(i, 3):
            w[(i, j)] = (metric_inv[..., i, j] * jac if k == 1
                         else metric[..., i, j] / jac)
            w[(j, i)] = w[(i, j)]                         # symmetric: shared array
    return w


def _reference_weight(n_comp):
    """Pointwise weight of the reference-domain projection masses (``W = I``)."""
    def weight(metric, metric_inv, jac):
        del metric, metric_inv
        one = jnp.ones_like(jac)
        return {(c, c): one for c in range(n_comp)}
    return weight


def build_mass_diagonal(seq, k, geometry=None):
    """Return ``diag(M_k)`` in raw DOF space, exactly and probe-free.

    A diagonal entry only ever sees its own component, so only the ``(c, c)``
    metric blocks contribute and the ``_to_quadrature`` / ``_from_quadrature``
    pair of the matvec collapses to the *same* sum factorization against **squared** basis tables
    with no input vector:

        d[a,b,e] = sum_elem sum_{q,r,s} Bx[q,a]^2 By[r,b]^2 Bz[s,e]^2 W(q,r,s)

    Cost is one contraction -- O(1) applies -- against the O(n) full applies a
    probed diagonal needs. The result is exact to floating point, not an
    estimate, and is the ``D`` of the pow2 sandwich as well as the diagonal of
    the Jacobi mass preconditioner.

    The returned vector is the components concatenated in form order, matching
    the layout of :func:`build_matrixfree_mass_apply`.
    """
    geometry = seq.geometry if geometry is None else geometry
    split, gauss = _element_layout(seq)
    form, comp, n_comp = _form_bases(seq, k)
    weight_of = _mass_weight(k, geometry.metric_jkl, geometry.metric_inv_jkl,
                             geometry.jacobian_j)
    shapes = form.shape

    parts = []
    for c in range(n_comp):
        Wf = split(weight_of[(c, c)]) * gauss
        Bx, By, Bz = comp[c][0], comp[c][2], comp[c][4]
        # Squared basis tables: the row and column bases coincide on the diagonal.
        t1 = jnp.einsum('xqa,xyzqrs->xyzars', Bx * Bx, Wf)
        t2 = jnp.einsum('yrb,xyzars->xyzabs', By * By, t1)
        d_local = jnp.einsum('zse,xyzabs->xyzabe', Bz * Bz, t2)
        seg = _flat_dof_plan(comp[c][1], comp[c][3], comp[c][5],
                             shapes[c]).reshape(-1)
        parts.append(jax.ops.segment_sum(
            d_local.reshape(-1), seg, num_segments=int(np.prod(shapes[c]))))
    return jnp.concatenate(parts)


def _build_sumfact_apply(seq, k_row, k_col, weight_fn, geometry):
    """Jitted raw-DOF apply of ``int Lambda^{k_row} . W . Lambda^{k_col}``.

    ``weight_fn(metric, metric_inv, jac)`` returns the pointwise weight per
    ``(row_comp, col_comp)`` pair as ``{(cr, cc): (N_q,)}``; pairs it omits
    are skipped and pairs that share one array (a symmetric weight) are formed
    once.

    The element plan (basis values, gather indices, scatter segment ids) and
    the weight -- the unique entries, moved to the element layout with the
    Gauss weights folded in -- are built once here and passed to the jitted
    kernel as runtime arguments (not captured as constants) to avoid XLA
    constant-folding of the large integer index tensors.  Memoising the
    weight is a measured choice: forming it inside the kernel from the stored
    metric, even as bare elementwise products with one stacked transpose,
    costs +18% on the k=1/2 apply at (16,32,16) (0.353 vs 0.297 ms) -- the
    cost is the per-apply pass over the geometry, not the algebra.  The price
    is six element-layout fields per vector degree resident on the sequence
    (14 scalars per quadrature point over k=0..3), rebuilt with the plan when
    the geometry changes; the compiled kernel is shared (see
    :func:`_sumfact_kernel`).

    The matvec is the sum factorization split at the quadrature points: each
    column component is gathered and pushed to the quadrature points ONCE, the
    weight mixes the components pointwise, and each row component is tested
    ONCE, followed by a single ``segment_sum`` into the concatenated output.
    """
    split, gauss = _element_layout(seq)
    form_r, comp_r, n_r = _form_bases(seq, k_row)
    form_c, comp_c, n_c = _form_bases(seq, k_col)

    def starts(form, n_comp):
        out = [0]
        for c in range(n_comp):
            out.append(out[-1] + int(np.prod(form.shape[c])))
        return tuple(out)

    starts_r = starts(form_r, n_r)
    starts_c = starts(form_c, n_c)
    # The weight: which pairs exist, which share an array, and the unique
    # arrays themselves in the element layout with the Gauss weights folded in.
    weight_of = weight_fn(geometry.metric_jkl, geometry.metric_inv_jkl,
                          geometry.jacobian_j)
    pairs = tuple(weight_of)
    unique_ids, column = [], {}
    for pair, w in weight_of.items():
        if id(w) not in unique_ids:
            unique_ids.append(id(w))
        column[pair] = unique_ids.index(id(w))
    Ws = [None] * len(unique_ids)
    for pair, w in weight_of.items():
        Ws[column[pair]] = split(w) * gauss
    Ws = tuple(Ws)

    # Basis VALUES (for the einsums) are separated from the gather/scatter
    # index plans, which depend only on the mesh topology: flat gather indices
    # per column component and one flat scatter (segment-id) array for the
    # whole output, offset by the component starts.
    Bvals_r = tuple((c[0], c[2], c[4]) for c in comp_r)
    Bvals_c = tuple((c[0], c[2], c[4]) for c in comp_c)
    gather_idx = tuple(
        _flat_dof_plan(comp_c[c][1], comp_c[c][3], comp_c[c][5], form_c.shape[c])
        for c in range(n_c))
    seg_idx = jnp.concatenate([
        _flat_dof_plan(comp_r[c][1], comp_r[c][3], comp_r[c][5],
                       form_r.shape[c]).reshape(-1) + starts_r[c]
        for c in range(n_r)])
    n_out = starts_r[-1]
    cols = tuple(column[pair] for pair in pairs)

    def apply(x):
        return _sumfact_kernel(x, Bvals_r, Bvals_c, Ws, gather_idx, seg_idx,
                               pairs=pairs, cols=cols, starts_c=starts_c,
                               n_out=n_out)

    return apply


@functools.partial(jax.jit, static_argnames=("pairs", "cols", "starts_c", "n_out"))
def _sumfact_kernel(x, Bvals_r, Bvals_c, Ws, gather_idx, seg_idx, *,
                    pairs, cols, starts_c, n_out):
    """The sum-factorised matvec, ONE executable per operator shape.

    Module-level and keyed on the static plan (which component pairs exist,
    which weight array each uses, the component offsets, the output size),
    so a new geometry -- new ``Ws`` of the same shapes -- reuses the compiled
    kernel; a kernel defined inside the builder was a new function object per
    build and recompiled on every ``set_geometry``.
    """
    W = {pair: Ws[c] for pair, c in zip(pairs, cols)}
    n_c, n_r = len(Bvals_c), len(Bvals_r)
    u = [_to_quadrature(Bvals_c[c], x[starts_c[c]:starts_c[c + 1]],
                        gather_idx[c]) for c in range(n_c)]
    y_parts = []
    for cr in range(n_r):
        v = sum(W[(cr, cc)] * u[cc] for cc in range(n_c) if (cr, cc) in pairs)
        y_parts.append(_from_quadrature(Bvals_r[cr], v).reshape(-1))
    return jax.ops.segment_sum(jnp.concatenate(y_parts), seg_idx,
                               num_segments=n_out)


def build_matrixfree_mass_apply(seq, k, geometry=None):
    """Return a jitted raw-DOF-space ``x -> M_k x`` that never stores ``M_k``.

    The returned callable acts on a vector in the *raw tensor-product* DOF
    space (the unextracted, periodic DOF layout). Boundary / polar extraction
    ``E (.) E^T`` is applied by the caller. The metric weight is formed once
    from the geometry's stored metric, inverse metric and ``det DF`` (see
    :func:`_mass_weight`) and memoised in the element layout.
    """
    geometry = seq.geometry if geometry is None else geometry
    return _build_sumfact_apply(
        seq, k, k,
        lambda metric, metric_inv, jac: _mass_weight(k, metric, metric_inv, jac),
        geometry)


def build_matrixfree_projection_apply(seq, k_row, k_col):
    """Return a jitted raw-DOF apply of the projection mass ``int Lambda^{k_row} . Lambda^{k_col}``.

    The weight is the reference-domain identity (no metric), so only the
    matching component pairs contribute; the input is a raw ``k_col``-form
    vector and the output a raw ``k_row``-form vector.
    """
    n_comp = 1 if k_row in (0, 3) else 3
    if n_comp != (1 if k_col in (0, 3) else 3):
        raise ValueError(
            f"projection mass needs matching component counts, got k_row={k_row}, k_col={k_col}")
    return _build_sumfact_apply(
        seq, k_row, k_col, _reference_weight(n_comp), seq.geometry)
