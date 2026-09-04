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


def _split_field(field_flat, ne_x, ne_y, ne_z, qx, qy, qz):
    """Reshape a flat (r-major) quad field to per-element blocks.

    Returns shape ``(ne_x, ne_y, ne_z, qx, qy, qz, *trailing)``; trailing axes
    (e.g. the ``(3, 3)`` of the metric) ride along. One reshape and one transpose,
    so it fuses into the consumer inside a jit.
    """
    trailing = tuple(range(6, 6 + field_flat.ndim - 1))
    f = field_flat.reshape(ne_x, qx, ne_y, qy, ne_z, qz, *field_flat.shape[1:])
    return f.transpose(0, 2, 4, 1, 3, 5, *trailing)


def _element_layout(seq):
    """Return ``(split, gauss)``: the flat->element reshape and the Gauss weights.

    ``split(field_flat)`` is :func:`_split_field` with this sequence's counts
    bound; ``gauss`` is the ``(ne_x, ne_y, ne_z, qx, qy, qz)`` outer product of
    the per-axis Gauss weights.
    """
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


def _shift_plan_axis(g, S, axis="?"):
    """Return ``(ne, nloc, S)`` for the shift map ``g[e, l] == (e + l) % S``.

    On a tensor-product B-spline axis element ``e``'s local DoF ``l`` is global
    DoF ``e + l`` wrapped, so the element-to-DoF map is a pure shift. That is
    what lets the element gather and the element assembly run without indices
    at all (see :func:`_structured_gather` and :func:`_structured_accumulate`).

    Raises:
        ValueError: if ``g`` is not that map. The kernel has no other way to
            read or assemble, so this is fatal rather than a fallback: a basis
            numbered differently would otherwise be silently wrong.
    """
    g = np.asarray(g)
    if g.ndim != 2:
        raise ValueError(
            f"axis {axis}: element-to-DoF map must be a 2-D (ne, nloc) array, "
            f"got shape {g.shape}")
    ne, nloc = g.shape
    e = np.arange(ne)[:, None]
    lo = np.arange(nloc)[None, :]
    if not np.array_equal(g, (e + lo) % int(S)):
        raise ValueError(
            f"axis {axis}: element-to-DoF map is not the shift (e + l) % {S} "
            f"for ne={ne}, nloc={nloc}. The sum-factorised mass kernel "
            f"assembles by shifted dense adds and cannot express this basis.")
    return (int(ne), int(nloc), int(S))


def _shift_plan(gx, gy, gz, shape):
    """The three-axis shift plan of a component.

    Raises:
        ValueError: if any axis is not a pure shift, via
            :func:`_shift_plan_axis`.
    """
    return tuple(_shift_plan_axis(g, s, axis)
                 for g, s, axis in zip((gx, gy, gz), shape, "xyz"))


def _structured_accumulate(y, plan):
    """Element-to-DoF assembly as shifted dense adds instead of a scatter.

    ``y`` is ``(ne_x, ne_y, ne_z, nloc_x, nloc_y, nloc_z)`` and the result is
    the ``(S_x, S_y, S_z)`` DoF grid with

        out[i, j, k] = sum over (lx, ly, lz) of y[i-lx, j-ly, k-lz, lx, ly, lz]

    the wrap being modulo the axis size. This is the same sum an indexed
    scatter performs, but every destination is known at compile time, so it
    lowers to dense shifts and adds with no indexed writes.

    That matters because indexed writes are the one thing a TPU has no fast
    path for. Measured on a v5e at (12,24,12) p=3, 221k contributions into
    3456 DoFs: 2.011 ms as a ``segment_sum``, 0.061 ms this way, agreeing to
    3e-7 in float32. The mass apply is the innermost kernel of every Krylov
    iteration, so that factor propagates to the whole solve.

    The sum is taken one axis at a time. Doing all ``prod(nloc)`` terms at full
    ``(S_x, S_y, S_z)`` size would also work, but accumulating x first shrinks
    the array the y pass has to touch, and again for z -- the same sum
    factorisation the matvec itself uses, applied to the assembly.
    """
    (ne_x, nl_x, S_x), (ne_y, nl_y, S_y), (ne_z, nl_z, S_z) = plan

    def accumulate(a, axis, ne, nloc, S):
        """Sum ``nloc`` copies of ``a`` shifted along ``axis``, padded to ``S``.

        The element axis being consumed is ``axis``; its matching local axis is
        always at index 3, because the element axes are consumed left to right
        and dropping index 3 does not move indices 0..2.

        ``roll`` is circular, which is exactly the ``mod S`` in the index map.
        Where ``ne < S`` the padding supplies the zeros for destinations no
        element contributes to.
        """
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
    """Element-local read as rolled slices instead of an indexed gather.

    The mirror of :func:`_structured_accumulate`. Where that one writes
    ``out[e + l] += y[e, l]``, this one reads

        x_local[e, l] = x[(e + l) mod S]

    over the same shift plan, so again every source is known at compile time
    and no index tensor reaches the device.

    Rolling by ``-l`` puts source ``e + l`` at position ``e``; taking the first
    ``ne`` entries drops the wrapped tail, which on a clamped axis is the part
    that never had an element. The three axes are done one at a time, which is
    ``nl_x + nl_y + nl_z`` rolls rather than the ``prod(nloc)`` a direct
    expansion would need, and it keeps each roll on the smallest array that
    still carries the axis.

    Measured on a v5e at (12,24,12) p=3, 3456 DoFs read 221k times: 1.624 ms
    as ``x[gather_idx]``, 0.049 ms this way, agreeing exactly. The gather was
    about 80% of the whole mass apply, so this is the larger half of the
    kernel rather than a tidy-up.
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
    """Fuse the last two 1-D bases into one two-axis table.

    ``Byz[y, z, (r,s), (d,f)] = By[y, r, d] * Bz[z, s, f]``, which lets the y
    and z stages of the sum factorization run as a single contraction of width
    ``nly * nlz`` instead of two of width ``nly`` and ``nlz``.

    That is a deliberate trade of arithmetic for shape. Three sequential
    contractions of width 3-4 is the FLOP-minimal factorization and the right
    one on a machine that charges per FLOP, but both a v5e and an H200 charge
    per contraction far more than per FLOP at these widths: measured over both
    halves of a k=2 component, folding costs 1.5x the FLOPs and returns
    1.48-1.70x on a v5e, 1.23-1.49x on an H200 and 1.62x on a CPU
    (``docs/research/tpu_v5e_benchmark.md``). Folding all three axes was also measured and
    loses -- 4.8x the FLOPs is too much to buy back, and it needs a per-element
    basis tensor two orders of magnitude larger.

    Args:
        By: y basis values, ``(ne_y, qy, nly)``.
        Bz: z basis values, ``(ne_z, qz, nlz)``.

    Returns:
        ``(ne_y, ne_z, qy * qz, nly * nlz)``. Small: 166 KB at ``(12,24,12)``
        p=3 in float32, growing only with the y-z element count.
    """
    ne_y, qy, nly = By.shape
    ne_z, qz, nlz = Bz.shape
    return jnp.einsum('yrd,zsf->yzrsdf', By, Bz).reshape(
        ne_y, ne_z, qy * qz, nly * nlz)


def _to_quadrature(Bvals, x_local):
    """Column half of :func:`_elem_block_mixed` folded against a vector.

    Evaluates the component's field at the element quadrature points,
    ``(ne_x, ne_y, ne_z, qx, qy, qz)``, from an already element-local input.
    The read that produces ``x_local`` is :func:`_structured_gather`, done by
    the caller, so this is a contraction and nothing else.

    Two stages, not three: see :func:`_fuse_yz` for why the y and z
    contractions are fused.
    """
    Bx, By, Bz, Byz = Bvals
    ne_x, qx, _ = Bx.shape
    ne_y, qy, nly = By.shape
    ne_z, qz, nlz = Bz.shape
    t1 = jnp.einsum('xqb,xyzbdf->xyzqdf', Bx, x_local)
    t1 = t1.reshape(ne_x, ne_y, ne_z, qx, nly * nlz)
    u = jnp.einsum('yzQD,xyzqD->xyzqQ', Byz, t1)
    return u.reshape(ne_x, ne_y, ne_z, qx, qy, qz)


def _from_quadrature(Bvals, u):
    """Row half of :func:`_elem_block_mixed`: test a quadrature-point field
    (Gauss weights already folded in) against the element-local row basis.

    The transpose of :func:`_to_quadrature`, and it reuses the same fused
    table: contracting ``(r,s)`` against ``(c,e)`` needs exactly the tensor
    that the column half contracts the other way round.
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
        plan = _shift_plan(comp[c][1], comp[c][3], comp[c][5], shapes[c])
        parts.append(_structured_accumulate(d_local, plan).reshape(-1))
    return jnp.concatenate(parts)


def _build_sumfact_apply(seq, k_row, k_col, weight_fn, geometry):
    """Jitted raw-DOF apply of ``int Lambda^{k_row} . W . Lambda^{k_col}``.

    ``weight_fn(metric, metric_inv, jac)`` returns the pointwise weight per
    ``(row_comp, col_comp)`` pair as ``{(cr, cc): (N_q,)}``; pairs it omits
    are skipped and pairs that share one array (a symmetric weight) are formed
    once.

    The element plan (basis values and the per-axis shift plans) and the
    weight -- the unique entries, moved to the element layout with the Gauss
    weights folded in -- are built once here and passed to the jitted kernel,
    the weight as a runtime argument and the plans as static ones.  Memoising
    the weight is a measured choice: forming it inside the kernel from the stored
    metric, even as bare elementwise products with one stacked transpose,
    costs +18% on the k=1/2 apply at (16,32,16) (0.353 vs 0.297 ms) -- the
    cost is the per-apply pass over the geometry, not the algebra.  The price
    is six element-layout fields per vector degree resident on the sequence
    (14 scalars per quadrature point over k=0..3), rebuilt with the plan when
    the geometry changes; the compiled kernel is shared (see
    :func:`_sumfact_kernel`).

    The matvec is the sum factorization split at the quadrature points: each
    column component is read and pushed to the quadrature points ONCE, the
    weight mixes the components pointwise, and each row component is tested
    ONCE and assembled by shifted adds into the concatenated output.
    """
    split, gauss = _element_layout(seq)
    form_r, comp_r, n_r = _form_bases(seq, k_row)
    form_c, comp_c, n_c = _form_bases(seq, k_col)

    starts_c = [0]
    for c in range(n_c):
        starts_c.append(starts_c[-1] + int(np.prod(form_c.shape[c])))
    starts_c = tuple(starts_c)
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

    # Basis VALUES (for the einsums) are separated from the shift plans, which
    # depend only on the mesh topology and are static: one per column component
    # for the read, one per row component for the assembly.
    # The fused y-z table is built once here, not inside the kernel: it is a
    # function of the kernel's arguments, so building it there would leave it
    # to XLA to hoist out of a solver's scan body.
    Bvals_r = tuple((c[0], c[2], c[4], _fuse_yz(c[2], c[4])) for c in comp_r)
    Bvals_c = tuple((c[0], c[2], c[4], _fuse_yz(c[2], c[4])) for c in comp_c)
    gather_plans = tuple(_shift_plan(comp_c[c][1], comp_c[c][3], comp_c[c][5],
                                     form_c.shape[c]) for c in range(n_c))
    shift_plans = tuple(_shift_plan(comp_r[c][1], comp_r[c][3], comp_r[c][5],
                                    form_r.shape[c]) for c in range(n_r))
    cols = tuple(column[pair] for pair in pairs)

    def apply(x):
        return _sumfact_kernel(x, Bvals_r, Bvals_c, Ws,
                               pairs=pairs, cols=cols, starts_c=starts_c,
                               shift_plans=shift_plans,
                               gather_plans=gather_plans)

    return apply


@functools.partial(jax.jit, static_argnames=("pairs", "cols", "starts_c",
                                             "shift_plans", "gather_plans"))
def _sumfact_kernel(x, Bvals_r, Bvals_c, Ws, *,
                    pairs, cols, starts_c, shift_plans, gather_plans):
    """The sum-factorised matvec, ONE executable per operator shape.

    Module-level and keyed on the static plan (which component pairs exist,
    which weight array each uses, the component offsets, the shift plans),
    so a new geometry -- new ``Ws`` of the same shapes -- reuses the compiled
    kernel; a kernel defined inside the builder was a new function object per
    build and recompiled on every ``set_geometry``.
    """
    W = {pair: Ws[c] for pair, c in zip(pairs, cols)}
    n_c, n_r = len(Bvals_c), len(Bvals_r)
    # The read is here rather than inside _to_quadrature, so that the two
    # halves of the element block stay pure contractions and mirror each other:
    # gather, contract, weight, contract, accumulate.
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
    # Already assembled per component and in component order, so the
    # concatenation is the whole output.
    return jnp.concatenate(y_parts)


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
