"""Element-local, sum-factorized mass-matrix assembly.

These assemblers exploit the separable tensor-product structure of the spline
bases together with the *element-local* support of each basis function. Instead
of the global ``O(n^6)`` einsum path in :mod:`mrx.assembly`, each element block
is formed once by sum factorization and scattered into a sparse triplet list,
giving ``O(n^3 p^6)`` work that is flat in ``n`` on the GPU.

The matrices produced here are the raw tensor-product mass matrices ``M`` in the
periodic/unextracted DOF space, identical (to machine precision) to the global
``assemble_scalar``/``assemble_vectorial`` output. Polar / boundary extraction
``E M E.T`` is applied afterward exactly as before.

Form weights (matching :func:`mrx.operators._assemble_mass_block`):

* k=0: ``W = J``                       (scalar)
* k=1: ``W = G^{-1} J``                (3x3, derivative basis on axis c)
* k=2: ``W = G (1/J)``                 (3x3, primal basis on axis c)
* k=3: ``W = 1/J``                     (scalar, derivative basis on all axes)

The quadrature weights ``w`` are folded in per axis via the 1D Gauss weights.
"""

import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import numpy as np

from mrx.geometry import grad_1d

__all__ = [
    "assemble_m0_local",
    "build_mass_diagonal",
    "build_stiffness_diagonal",
    "build_extracted_stiffness_diagonal_k0",
    "assemble_m1_local",
    "assemble_m2_local",
    "assemble_m3_local",
    "assemble_mass_local",
    "build_matrixfree_mass_apply",
]


# --------------------------------------------------------------------------- #
# Element-local 1D basis evaluation
# --------------------------------------------------------------------------- #
def evaluate_basis_local(basis, x_q_flat, q_per_elem):
    """Evaluate a 1D spline basis on each element at its local quad points.

    Works for both primal (``SplineBasis``) and derivative
    (``DerivativeSpline``) bases: the derivative basis simply reports a smaller
    degree, hence ``p`` locals per element instead of ``p+1``.

    Parameters
    ----------
    basis : SplineBasis or DerivativeSpline
        1D basis with ``.p``, ``.n`` and ``.type`` attributes and a callable
        ``basis(x, i)`` interface.
    x_q_flat : (n_elem * q,) array
        Composite Gauss quadrature points, ordered element-by-element.
    q_per_elem : int
        Number of Gauss points per knot interval.

    Returns
    -------
    B_loc : (n_elem, q_per_elem, p+1) array
        Values of the locally-active bases at the local quad points.
    gdof : (n_elem, p+1) int array
        Global DOF index of each local basis on each element.
    """
    p = basis.p
    n = basis.n
    n_local = p + 1
    if basis.type == "periodic":
        n_elem = n
        elems = jnp.arange(n_elem)
        ks = jnp.arange(n_local)
        gdof = (elems[:, None] + ks[None, :]) % n
    elif basis.type == "clamped":
        n_elem = n - p
        elems = jnp.arange(n_elem)
        ks = jnp.arange(n_local)
        gdof = elems[:, None] + ks[None, :]
    elif basis.type == "constant":
        # Single element, single DOF (p=0, n=1).
        n_elem = 1
        gdof = jnp.zeros((1, 1), dtype=jnp.int32)
    else:
        raise NotImplementedError(basis.type)

    x_local = x_q_flat.reshape(n_elem, q_per_elem)

    def eval_e(x_e, dof_e):
        return jax.vmap(
            lambda x: jax.vmap(lambda i: basis(x, i))(dof_e)
        )(x_e)

    B_loc = jax.vmap(eval_e, in_axes=(0, 0))(x_local, gdof)
    return B_loc, gdof


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

    Returns shape ``(ne_x, ne_y, ne_z, qx, qy, qz)``.
    """
    f = field_flat.reshape(ny, nx, nz).transpose(1, 0, 2)
    f = f.reshape(ne_x, qx, ne_y, qy, ne_z, qz).transpose(0, 2, 4, 1, 3, 5)
    return f


# --------------------------------------------------------------------------- #
# Element block via sum factorization (mixed row/col bases)
# --------------------------------------------------------------------------- #
def _elem_block_mixed(Bxr, Bxc, Byr, Byc, Bzr, Bzc, Wf, wx_e, wy_e, wz_e):
    """One element block for (possibly distinct) row/col bases.

    Returns ``block[a, b, c, d, e, f]`` with ``a/b`` = x row/col local,
    ``c/d`` = y row/col local, ``e/f`` = z row/col local.
    """
    W = Wf * wx_e[:, None, None] * wy_e[None, :, None] * wz_e[None, None, :]
    A = jnp.einsum('qa,qb,qrs->abrs', Bxr, Bxc, W)
    Bm = jnp.einsum('rc,rd,abrs->abcds', Byr, Byc, A)
    C = jnp.einsum('se,sf,abcds->abcdef', Bzr, Bzc, Bm)
    return C


_elem_block_mixed_vmapped = jax.vmap(jax.vmap(jax.vmap(
    _elem_block_mixed,
    in_axes=(None, None, None, None, 0, 0, 0, None, None, 0)),   # ez
    in_axes=(None, None, 0, 0, None, None, 0, None, 0, None)),   # ey
    in_axes=(0, 0, None, None, None, None, 0, 0, None, None))    # ex


@jax.jit
def _block_compute(Bxr, Bxc, Byr, Byc, Bzr, Bzc, Wf, wx, wy, wz):
    """JIT the expensive element-block contraction for one component pair."""
    return _elem_block_mixed_vmapped(
        Bxr, Bxc, Byr, Byc, Bzr, Bzc, Wf, wx, wy, wz)


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
# Scalar (k=0, k=3) assembler
# --------------------------------------------------------------------------- #
def _assemble_scalar_local(seq, bases3, weight_flat, shape3):
    """Element-local sum-factorized scalar mass assembler (BCOO).

    Parameters
    ----------
    bases3 : [bx, by, bz]
        The three 1D bases (same on rows and cols).
    weight_flat : (nquad,) array
        Geometry weight (WITHOUT quadrature weights ``w``).
    shape3 : (Sx, Sy, Sz)
        DOF-grid shape for C-order flat indexing.
    """
    nx, ny, nz = seq.quad.nx, seq.quad.ny, seq.quad.nz
    ne_x, ne_y, ne_z, qx, qy, qz = _elem_counts(seq)

    wx = seq.quad.w_x.reshape(ne_x, qx)
    wy = seq.quad.w_y.reshape(ne_y, qy)
    wz = seq.quad.w_z.reshape(ne_z, qz)

    Bx, gx = evaluate_basis_local(bases3[0], seq.quad.x_x, qx)
    By, gy = evaluate_basis_local(bases3[1], seq.quad.x_y, qy)
    Bz, gz = evaluate_basis_local(bases3[2], seq.quad.x_z, qz)

    Wf = _split_field(weight_flat, nx, ny, nz, ne_x, ne_y, ne_z, qx, qy, qz)
    blocks = _block_compute(Bx, Bx, By, By, Bz, Bz, Wf, wx, wy, wz)

    Sx, Sy, Sz = shape3
    nlx, nly, nlz = Bx.shape[-1], By.shape[-1], Bz.shape[-1]

    gx_r = gx.reshape(ne_x, 1, 1, nlx, 1, 1, 1, 1, 1)
    gy_r = gy.reshape(1, ne_y, 1, 1, 1, nly, 1, 1, 1)
    gz_r = gz.reshape(1, 1, ne_z, 1, 1, 1, 1, nlz, 1)
    row = ((gx_r * Sy + gy_r) * Sz + gz_r).astype(jnp.int32)

    gx_c = gx.reshape(ne_x, 1, 1, 1, nlx, 1, 1, 1, 1)
    gy_c = gy.reshape(1, ne_y, 1, 1, 1, 1, nly, 1, 1)
    gz_c = gz.reshape(1, 1, ne_z, 1, 1, 1, 1, 1, nlz)
    col = ((gx_c * Sy + gy_c) * Sz + gz_c).astype(jnp.int32)

    vals = blocks.reshape(-1)
    rows = jnp.broadcast_to(row, blocks.shape).reshape(-1)
    cols = jnp.broadcast_to(col, blocks.shape).reshape(-1)
    n_total = Sx * Sy * Sz
    indices = jnp.stack([rows, cols], axis=-1)
    return jsparse.BCOO((vals, indices), shape=(n_total, n_total))


# --------------------------------------------------------------------------- #
# Vectorial (k=1, k=2) assembler
# --------------------------------------------------------------------------- #
def _assemble_vectorial_local(seq, form, comp_bases_fn, weight_3x3_flat):
    """Element-local sum-factorized vectorial mass assembler (BCOO).

    Parameters
    ----------
    form : DifferentialForm
        The k=1 or k=2 form whose basis layout defines the DOF blocks.
    comp_bases_fn : callable(form, c) -> [bx, by, bz]
        Returns the three 1D bases used by component ``c``.
    weight_3x3_flat : (nquad, 3, 3) array
        Metric weight (WITHOUT the quadrature weights ``w``; folded in via the
        per-axis Gauss weights).
    """
    nx, ny, nz = seq.quad.nx, seq.quad.ny, seq.quad.nz
    ne_x, ne_y, ne_z, qx, qy, qz = _elem_counts(seq)

    wx = seq.quad.w_x.reshape(ne_x, qx)
    wy = seq.quad.w_y.reshape(ne_y, qy)
    wz = seq.quad.w_z.reshape(ne_z, qz)

    shapes = form.shape                       # component DOF-grid shapes
    starts = [0, form.n1, form.n1 + form.n2]  # flat offsets per component
    n_total = form.n

    # Cache local basis evaluations (each axis basis evaluated once).
    eval_cache = {}

    def local_eval(basis, x_q, q):
        key = id(basis)
        if key not in eval_cache:
            eval_cache[key] = evaluate_basis_local(basis, x_q, q)
        return eval_cache[key]

    all_vals, all_rows, all_cols = [], [], []

    for cr in range(3):
        br = comp_bases_fn(form, cr)
        Bxr, gxr = local_eval(br[0], seq.quad.x_x, qx)
        Byr, gyr = local_eval(br[1], seq.quad.x_y, qy)
        Bzr, gzr = local_eval(br[2], seq.quad.x_z, qz)
        _, Syr, Szr = shapes[cr]

        for cc in range(3):
            bc = comp_bases_fn(form, cc)
            Bxc, gxc = local_eval(bc[0], seq.quad.x_x, qx)
            Byc, gyc = local_eval(bc[1], seq.quad.x_y, qy)
            Bzc, gzc = local_eval(bc[2], seq.quad.x_z, qz)
            _, Syc, Szc = shapes[cc]

            Wf = _split_field(weight_3x3_flat[:, cr, cc], nx, ny, nz,
                              ne_x, ne_y, ne_z, qx, qy, qz)

            blocks = _block_compute(
                Bxr, Bxc, Byr, Byc, Bzr, Bzc, Wf, wx, wy, wz)
            # blocks: (ne_x, ne_y, ne_z, nxr, nxc, nyr, nyc, nzr, nzc)

            nxr, nxc = Bxr.shape[-1], Bxc.shape[-1]
            nyr, nyc = Byr.shape[-1], Byc.shape[-1]
            nzr, nzc = Bzr.shape[-1], Bzc.shape[-1]

            # Row flat index (component cr grid + offset)
            gxr_b = gxr.reshape(ne_x, 1, 1, nxr, 1, 1, 1, 1, 1)
            gyr_b = gyr.reshape(1, ne_y, 1, 1, 1, nyr, 1, 1, 1)
            gzr_b = gzr.reshape(1, 1, ne_z, 1, 1, 1, 1, nzr, 1)
            row = (starts[cr] + (gxr_b * Syr + gyr_b) * Szr + gzr_b
                   ).astype(jnp.int32)

            # Col flat index (component cc grid + offset)
            gxc_b = gxc.reshape(ne_x, 1, 1, 1, nxc, 1, 1, 1, 1)
            gyc_b = gyc.reshape(1, ne_y, 1, 1, 1, 1, nyc, 1, 1)
            gzc_b = gzc.reshape(1, 1, ne_z, 1, 1, 1, 1, 1, nzc)
            col = (starts[cc] + (gxc_b * Syc + gyc_b) * Szc + gzc_b
                   ).astype(jnp.int32)

            all_vals.append(blocks.reshape(-1))
            all_rows.append(jnp.broadcast_to(row, blocks.shape).reshape(-1))
            all_cols.append(jnp.broadcast_to(col, blocks.shape).reshape(-1))

    vals = jnp.concatenate(all_vals)
    rows = jnp.concatenate(all_rows)
    cols = jnp.concatenate(all_cols)
    indices = jnp.stack([rows, cols], axis=-1)
    return jsparse.BCOO((vals, indices), shape=(n_total, n_total))


# --------------------------------------------------------------------------- #
# Public per-degree entry points
# --------------------------------------------------------------------------- #
def assemble_m0_local(seq, geometry=None):
    """k=0 mass matrix (BCOO): ``M0_ij = integral Λ0_i Λ0_j det DF``."""
    geometry = seq.geometry if geometry is None else geometry
    form = seq.basis_0
    weight = geometry.jacobian_j
    return _assemble_scalar_local(
        seq, [form.Λ[0], form.Λ[1], form.Λ[2]], weight, form.shape[0])


def assemble_m1_local(seq, geometry=None):
    """k=1 mass matrix (BCOO): ``M1_ij = integral Λ1_i · G^{-1} Λ1_j det DF``."""
    geometry = seq.geometry if geometry is None else geometry
    weight = geometry.metric_inv_jkl * geometry.jacobian_j[:, None, None]
    return _assemble_vectorial_local(
        seq, seq.basis_1, _component_axis_bases_k1, weight)


def assemble_m2_local(seq, geometry=None):
    """k=2 mass matrix (BCOO): ``M2_ij = integral Λ2_i · G Λ2_j (det DF)^{-1}``."""
    geometry = seq.geometry if geometry is None else geometry
    weight = geometry.metric_jkl * (1.0 / geometry.jacobian_j)[:, None, None]
    return _assemble_vectorial_local(
        seq, seq.basis_2, _component_axis_bases_k2, weight)


def assemble_m3_local(seq, geometry=None):
    """k=3 mass matrix (BCOO): ``M3_ij = integral Λ3_i Λ3_j (det DF)^{-1}``."""
    geometry = seq.geometry if geometry is None else geometry
    form = seq.basis_3
    weight = 1.0 / geometry.jacobian_j
    return _assemble_scalar_local(
        seq, [form.dΛ[0], form.dΛ[1], form.dΛ[2]], weight, form.shape[0])


_LOCAL_ASSEMBLERS = {
    0: assemble_m0_local,
    1: assemble_m1_local,
    2: assemble_m2_local,
    3: assemble_m3_local,
}


def assemble_mass_local(seq, k, geometry=None):
    """Dispatch element-local mass assembly for form degree ``k`` (BCOO)."""
    try:
        fn = _LOCAL_ASSEMBLERS[k]
    except KeyError:
        raise ValueError("k must be 0, 1, 2 or 3") from None
    return fn(seq, geometry)


# --------------------------------------------------------------------------- #
# Matrix-free (sum-factorized) mass apply
# --------------------------------------------------------------------------- #
# The functions below apply ``M_k @ x`` in the raw tensor-product DOF space
# *without ever materializing* ``M_k``. They reuse the same element-local sum
# factorization and metric weights as the assemblers above, but fold the
# contraction against the input vector instead of forming the dense element
# block. Transient memory is O(n^3 (p+1)^2 q) instead of the O(n^3 (p+1)^6) of
# the stored matrix, which removes the high-(n, p) storage bottleneck for M1.
def _quad_gauss_weight(seq):
    """``(ne_x,ne_y,ne_z,qx,qy,qz)`` outer product of the per-axis Gauss weights."""
    ne_x, ne_y, ne_z, qx, qy, qz = _elem_counts(seq)
    wx = seq.quad.w_x.reshape(ne_x, qx)
    wy = seq.quad.w_y.reshape(ne_y, qy)
    wz = seq.quad.w_z.reshape(ne_z, qz)
    return (wx[:, None, None, :, None, None]
            * wy[None, :, None, None, :, None]
            * wz[None, None, :, None, None, :])


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


def _element_apply(Bvals_r, Bvals_c, W, x_flat_c, gather_idx_c):
    """One (row-comp, col-comp) element contraction folded against a vector.

    Mirrors :func:`_elem_block_mixed` but contracts against the gathered input
    instead of forming the dense element block. The gather uses a precomputed
    flat index plan (``gather_idx_c``); no index arithmetic runs in the matvec.
    """
    Bxr, Byr, Bzr = Bvals_r
    Bxc, Byc, Bzc = Bvals_c
    # Gather element-local input for the column component (single gather).
    x_local = x_flat_c[gather_idx_c]  # (ne_x,ne_y,ne_z,nxc,nyc,nzc)

    # Column bases -> quadrature points.
    t1 = jnp.einsum('xqb,xyzbdf->xyzqdf', Bxc, x_local)
    t2 = jnp.einsum('yrd,xyzqdf->xyzqrf', Byc, t1)
    u = jnp.einsum('zsf,xyzqrf->xyzqrs', Bzc, t2)

    # Metric weight at the quadrature points (already includes Gauss weights).
    u = u * W

    # Row bases <- quadrature points.
    s1 = jnp.einsum('xqa,xyzqrs->xyzars', Bxr, u)
    s2 = jnp.einsum('yrc,xyzars->xyzacs', Byr, s1)
    y_local = jnp.einsum('zse,xyzacs->xyzace', Bzr, s2)
    return y_local


def _mass_form_and_weights(seq, k, geometry):
    """Shared element plan for the ``M_k`` matvec and its exact diagonal.

    Returns ``(form, comp, pairs, weight_of, n_comp)``: the form object, the
    per-component 1D basis tables from :func:`_bases_for_form`, the list of
    ``(row_comp, col_comp)`` pairs, the quadrature-point metric weight for each
    pair, and the component count.

    Both :func:`build_matrixfree_mass_apply` and :func:`build_mass_diagonal`
    go through here so the diagonal is derived from *the same tables the solver
    applies*. The removed ``diag_EAET_direct`` route recomputed its own tables
    and drifted from the matvec; that failure mode is structural, not a bug to
    be fixed once, so the plan is shared rather than mirrored.
    """
    if k == 0:
        form = seq.basis_0
        comp = _bases_for_form(seq, form, lambda f, c: [f.Λ[0], f.Λ[1], f.Λ[2]], 1)
        weight = geometry.jacobian_j  # scalar (nquad,)
        pairs = [(0, 0)]
        weight_of = {(0, 0): weight}
        n_comp = 1
    elif k == 3:
        form = seq.basis_3
        comp = _bases_for_form(seq, form, lambda f, c: [f.dΛ[0], f.dΛ[1], f.dΛ[2]], 1)
        weight = 1.0 / geometry.jacobian_j
        pairs = [(0, 0)]
        weight_of = {(0, 0): weight}
        n_comp = 1
    elif k == 1:
        form = seq.basis_1
        comp = _bases_for_form(seq, form, _component_axis_bases_k1, 3)
        metric = geometry.metric_inv_jkl * geometry.jacobian_j[:, None, None]
        pairs = [(cr, cc) for cr in range(3) for cc in range(3)]
        weight_of = {(cr, cc): metric[:, cr, cc] for cr, cc in pairs}
        n_comp = 3
    elif k == 2:
        form = seq.basis_2
        comp = _bases_for_form(seq, form, _component_axis_bases_k2, 3)
        metric = geometry.metric_jkl * (1.0 / geometry.jacobian_j)[:, None, None]
        pairs = [(cr, cc) for cr in range(3) for cc in range(3)]
        weight_of = {(cr, cc): metric[:, cr, cc] for cr, cc in pairs}
        n_comp = 3
    else:
        raise ValueError("k must be 0, 1, 2 or 3")
    return form, comp, pairs, weight_of, n_comp


def build_mass_diagonal(seq, k, geometry=None):
    """Return ``diag(M_k)`` in raw DOF space, exactly and probe-free.

    A diagonal entry only ever sees its own component, so only the ``(c, c)``
    metric blocks contribute and the contraction of :func:`_element_apply`
    collapses to the *same* sum factorization against **squared** basis tables
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
    nx, ny, nz = seq.quad.nx, seq.quad.ny, seq.quad.nz
    ne_x, ne_y, ne_z, qx, qy, qz = _elem_counts(seq)
    gw = _quad_gauss_weight(seq)

    form, comp, _pairs, weight_of, n_comp = _mass_form_and_weights(seq, k, geometry)
    shapes = form.shape

    parts = []
    for c in range(n_comp):
        Wf = _split_field(weight_of[(c, c)], nx, ny, nz,
                          ne_x, ne_y, ne_z, qx, qy, qz) * gw
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


# For k=1 (curl) and k=2 (div): which (k+1)-form component each k-form
# component feeds, with what sign, by differentiating along which axis.
# Read off _apply_incidence_mf:
#     curl: P = -d_z b + d_t c ,  Q = +d_z a - d_r c ,  R = -d_t a + d_r b
#     div : d_r P + d_t Q + d_z R
# Note the differentiated axis is never the component's own derivative axis, so
# the axis being differentiated always carries a PRIMAL table and its derivative
# is grad_1d of the degree-(p-1) table.
_CURL_CONTRIB = {0: ((1, +1.0, 2), (2, -1.0, 1)),
                 1: ((0, -1.0, 2), (2, +1.0, 0)),
                 2: ((0, +1.0, 1), (1, -1.0, 0))}
_DIV_CONTRIB = {0: ((0, +1.0, 0),), 1: ((0, +1.0, 1),), 2: ((0, +1.0, 2),)}


def build_stiffness_diagonal(seq, k, geometry=None):
    """Return ``diag(S_k)`` in raw DOF space, exactly and probe-free.

    ``S_k = G_k^T M_{k+1} G_k``, so the diagonal is the ``M_{k+1}``-energy of
    the DERIVATIVE of each basis function::

        diag(S_k)_a = || d phi_a ||^2_{M_{k+1}}
                    = sum_{i,j} sum_q (d phi_a)_i W_ij (d phi_a)_j

    Every component of ``d phi_a`` is still a tensor product of 1D tables --
    the incidence differentiates one axis at a time -- so this is the same sum
    factorization as the mass diagonal, with one term per pair of
    ``(k+1)``-form components that ``phi_a`` feeds.

    * k=0: 3 components (grad), ``W = g^ij J`` (the 1-form weight)
    * k=1: 2 components per 1-form component (curl), ``W = g_ij / J``
    * k=2: 1 component (div), ``W = 1/J`` (scalar 3-form weight)
    * k=3: ``S_3 = 0`` -- there is nothing above V3.
    """
    geometry = seq.geometry if geometry is None else geometry
    nx, ny, nz = seq.quad.nx, seq.quad.ny, seq.quad.nz
    types = seq.basis_0.types

    primal = (seq.basis_r_jk, seq.basis_t_jk, seq.basis_z_jk)
    deriv = (seq.d_basis_r_jk, seq.d_basis_t_jk, seq.d_basis_z_jk)
    grad = tuple(grad_1d(deriv[a], types[a]) for a in range(3))

    def rs(field):
        return jnp.asarray(field).reshape(ny, nx, nz).transpose(1, 0, 2)

    if k == 3:
        form = seq.basis_3
        return jnp.zeros(int(np.prod(form.shape[0])))

    if k == 0:
        form = seq.basis_0
        comps = [((primal[0], primal[1], primal[2]), (0, 0, 0))]
        contrib = {0: ((0, +1.0, 0), (1, +1.0, 1), (2, +1.0, 2))}
        Wq = rs(geometry.jacobian_j)[..., None, None] * jnp.transpose(
            geometry.metric_inv_jkl.reshape(ny, nx, nz, 3, 3), (1, 0, 2, 3, 4))
        n_tgt = 3
    elif k == 1:
        form = seq.basis_1
        comps = [tuple(deriv[a] if a == c else primal[a] for a in range(3))
                 for c in range(3)]
        contrib = _CURL_CONTRIB
        Wq = jnp.transpose(geometry.metric_jkl.reshape(ny, nx, nz, 3, 3),
                           (1, 0, 2, 3, 4)) / rs(geometry.jacobian_j)[..., None, None]
        n_tgt = 3
    elif k == 2:
        form = seq.basis_2
        comps = [tuple(primal[a] if a == c else deriv[a] for a in range(3))
                 for c in range(3)]
        contrib = _DIV_CONTRIB
        Wq = (1.0 / rs(geometry.jacobian_j))[..., None, None]
        n_tgt = 1
    else:
        raise ValueError("k must be 0, 1, 2 or 3")

    wq = rs(seq.quad.w)
    parts = []
    for c, base in enumerate(comps if k != 0 else [comps[0][0]]):
        base = base if k != 0 else comps[0][0]
        terms = contrib[c if k != 0 else 0]
        total = None
        for (tgt_p, sgn_p, ax_p) in terms:
            for (tgt_q, sgn_q, ax_q) in terms:
                Wf = wq * Wq[..., tgt_p if n_tgt > 1 else 0,
                             tgt_q if n_tgt > 1 else 0]
                tp = [grad[a] if a == ax_p else base[a] for a in range(3)]
                tq = [grad[a] if a == ax_q else base[a] for a in range(3)]
                t1 = jnp.einsum('ax,xyz->ayz', tp[0] * tq[0], Wf)
                t2 = jnp.einsum('by,ayz->abz', tp[1] * tq[1], t1)
                blk = sgn_p * sgn_q * jnp.einsum('cz,abz->abc', tp[2] * tq[2], t2)
                total = blk if total is None else total + blk
        parts.append(total.reshape(-1))
    return jnp.concatenate(parts)


def build_extracted_stiffness_diagonal_k0(seq, dirichlet: bool):
    """``diag(E S_0 E^T)`` with no operator applies at all.

    At k=0 there is no lower term (``L_0 = S_0``), so this is the whole
    Laplacian Jacobi diagonal.

    The extracted diagonal is an ENERGY, not a sum of raw matrix entries::

        (E S E^T)_ii = int <grad psi_i, W grad psi_i>,   psi_i = sum_a E_ia phi_a

    so it never needs off-diagonal entries of ``S`` and never needs a probe.

    * **bulk rows** -- ``E`` is a pure selector there, so ``psi_i = phi_a`` and
      the raw closed-form diagonal supplies them directly.
    * **polar rows** -- ``psi`` mixes a ring, but a k=0 polar row sits at a
      SINGLE zeta index, so it factors as a 2D ``(r,theta)`` shape times a 1D
      zeta table. Only ``n_polar`` distinct 2D shapes exist and they do not
      depend on the zeta index, so the cost is
      ``O(n_polar * n_q^{r,theta} * n_q^z)``.

    Verified against the probe to 3.3e-16 on the polar rows (see
    ``scripts/debug/polar_row_energy.py``).
    """
    from mrx.geometry import grad_1d  # noqa: PLC0415

    e = getattr(seq, "e0_dbc" if dirichlet else "e0")
    n_ext = int(e.shape[0])
    rows = np.asarray(e.rows)
    cols = np.asarray(e.cols)
    vals = np.asarray(e.vals)
    counts = np.bincount(rows, minlength=n_ext)

    diag = np.zeros(n_ext, dtype=np.float64)

    # --- bulk rows: pure selectors, straight from the raw closed form --------
    d_raw = np.asarray(build_stiffness_diagonal(seq, 0))
    single = counts[rows] == 1
    diag[rows[single]] = (vals[single] ** 2) * d_raw[cols[single]]

    polar = np.flatnonzero(counts > 1)
    if polar.size == 0:
        return jnp.asarray(diag)

    # --- polar rows: energy of the extracted basis function ------------------
    types = seq.basis_0.types
    Rt, Tt, Zt = (np.asarray(t) for t in
                  (seq.basis_r_jk, seq.basis_t_jk, seq.basis_z_jk))
    Rd = np.asarray(grad_1d(seq.d_basis_r_jk, types[0]))
    Td = np.asarray(grad_1d(seq.d_basis_t_jk, types[1]))
    Zd = np.asarray(grad_1d(seq.d_basis_z_jk, types[2]))

    minv = np.asarray(jnp.transpose(
        seq.geometry.metric_inv_jkl.reshape(
            seq.quad.ny, seq.quad.nx, seq.quad.nz, 3, 3), (1, 0, 2, 3, 4)))
    jq = np.asarray(jnp.transpose(
        seq.geometry.jacobian_j.reshape(
            seq.quad.ny, seq.quad.nx, seq.quad.nz), (1, 0, 2)))
    W = minv * jq[..., None, None]
    wr = np.asarray(seq.quad.w_x)
    wt = np.asarray(seq.quad.w_y)
    wz = np.asarray(seq.quad.w_z)

    nt, nzb = seq.basis_0.nt, seq.basis_0.nz
    for i in polar:
        sel = rows == i
        c, v = cols[sel], vals[sel]
        ir, it, iz = c // (nt * nzb), (c // nzb) % nt, c % nzb
        m = int(iz[0])
        # 2D (r,theta) shape and its partials; zeta stays a 1D factor.
        X = (np.einsum('a,aq,ar->qr', v, Rd[ir], Tt[it]),
             np.einsum('a,aq,ar->qr', v, Rt[ir], Td[it]),
             np.einsum('a,aq,ar->qr', v, Rt[ir], Tt[it]))
        Y = (Zt[m], Zt[m], Zd[m])
        total = 0.0
        for a in range(3):
            for b in range(3):
                # contract zeta first, then the (r,theta) plane
                F = np.einsum('qrs,s,s,s->qr', W[..., a, b], wz, Y[a], Y[b])
                total += float(np.einsum('qr,q,r,qr->', X[a] * X[b], wr, wt, F))
        diag[i] = total
    return jnp.asarray(diag)


def build_matrixfree_mass_apply(seq, k, geometry=None):
    """Return a jitted raw-DOF-space ``x -> M_k x`` that never stores ``M_k``.

    The returned callable acts on a vector in the *raw tensor-product* DOF
    space (the unextracted, periodic DOF layout that the stored ``M_k`` matrix
    also acts on). Boundary / polar extraction ``E (.) E^T`` is applied by the
    caller exactly as for the stored path.

    The element plan (basis values, gather indices, scatter segment ids and
    metric weights) is built once on the host; the jitted matvec performs a
    single gather and a single ``segment_sum`` per component pair with no index
    arithmetic. The plan arrays are passed as runtime arguments to the jitted
    kernel (not captured as constants) to avoid XLA constant-folding of the
    large integer index tensors.
    """
    geometry = seq.geometry if geometry is None else geometry
    nx, ny, nz = seq.quad.nx, seq.quad.ny, seq.quad.nz
    ne_x, ne_y, ne_z, qx, qy, qz = _elem_counts(seq)
    gw = _quad_gauss_weight(seq)

    form, comp, pairs, weight_of, n_comp = _mass_form_and_weights(seq, k, geometry)

    shapes = form.shape
    starts = [0]
    for c in range(n_comp):
        Sx, Sy, Sz = shapes[c]
        starts.append(starts[-1] + Sx * Sy * Sz)

    # Pre-split + fold Gauss weights into each (cr, cc) metric field.
    W_split = {}
    for (cr, cc) in pairs:
        Wf = _split_field(weight_of[(cr, cc)], nx, ny, nz,
                          ne_x, ne_y, ne_z, qx, qy, qz)
        W_split[(cr, cc)] = Wf * gw

    # --- Static element plan (built ONCE, reused for every matvec) -----------
    # Basis VALUES (for the einsums) are separated from the gather/scatter
    # index plans. The index plans -- flat gather indices per column component
    # and flat scatter (segment-id) arrays per row component -- depend only on
    # the mesh topology, so they are precomputed here and passed in as device
    # int32 arrays. The matvec then performs a single gather and a single
    # segment_sum per (cr, cc) pair with NO index arithmetic.
    Bvals = tuple((c[0], c[2], c[4]) for c in comp)          # (Bx, By, Bz)
    gather_idx = tuple(
        _flat_dof_plan(comp[cc][1], comp[cc][3], comp[cc][5], shapes[cc])
        for cc in range(n_comp))
    seg_idx = tuple(
        _flat_dof_plan(comp[cr][1], comp[cr][3], comp[cr][5],
                       shapes[cr]).reshape(-1)
        for cr in range(n_comp))
    nseg = tuple(int(np.prod(shapes[c])) for c in range(n_comp))

    starts_t = tuple(int(s) for s in starts)

    @jax.jit
    def _impl(x, Bvals, W_split, gather_idx, seg_idx):
        # Split the input into flattened component DOF vectors.
        Xc = [x[starts_t[c]:starts_t[c + 1]] for c in range(n_comp)]

        out_parts = []
        for cr in range(n_comp):
            acc = jnp.zeros((nseg[cr],), dtype=x.dtype)
            for cc in range(n_comp):
                if (cr, cc) not in W_split:
                    continue
                y_local = _element_apply(
                    Bvals[cr], Bvals[cc], W_split[(cr, cc)],
                    Xc[cc], gather_idx[cc])
                acc = acc + jax.ops.segment_sum(
                    y_local.reshape(-1), seg_idx[cr], num_segments=nseg[cr])
            out_parts.append(acc)
        return jnp.concatenate(out_parts)

    def apply(x):
        return _impl(x, Bvals, W_split, gather_idx, seg_idx)

    return apply
