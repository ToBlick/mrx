# %%
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import numpy as np



def _offsets(hw, s):
    """Unique periodic column offsets of a stencil of half-width ``hw`` on ``s`` DOFs."""
    if 2 * hw + 1 <= s:
        return np.arange(-hw, hw + 1)
    return np.arange(-(s // 2), s - s // 2)


def _stencil_triplets(pairs, row_shape, col_shape, hw, row_start=0, col_start=0):
    """COO triplets of one tensor-product block ``Σ_pairs sign ∫ (R T Z)_row W (R T Z)_col``.

    ``pairs`` is a list of ``(W_3d, R_row, T_row, Z_row, R_col, T_col, Z_col,
    sign)`` with ``W_3d`` of shape ``(n_qt, n_qr, n_qz)`` and the 1D factors
    of shape ``(s, n_q)``; every pair contributes to the same ``(row, col)``
    positions, so they are stacked and contracted together. The stencil is
    batched over the θ and ζ offsets for each radial offset, i.e. one
    contraction per radial offset with ``(2 hw_t + 1)(2 hw_z + 1) N`` values,
    and the radial factor is contracted first (smallest quadrature axis).

    Returns ``(vals, rows, cols)``: the values on device, the index arrays as
    host ``int32`` arrays, in ``(dr, dt, dz, row)`` order.
    """
    hw_r, hw_t, hw_z = hw
    c1, c2, c3 = col_shape
    offsets_r = _offsets(hw_r, c1)
    offsets_t = _offsets(hw_t, c2)
    offsets_z = _offsets(hw_z, c3)

    signs = jnp.asarray([pair[7] for pair in pairs], dtype=pairs[0][0].dtype)
    W = jnp.stack([pair[0] for pair in pairs]) * signs[:, None, None, None]
    Rr, Tr, Zr, Rc, Tc, Zc = (jnp.stack([pair[m] for pair in pairs])
                              for m in range(1, 7))

    s1, s2, s3 = row_shape
    ct = (np.arange(s2)[None, :] + offsets_t[:, None]) % c2      # (T, s2)
    cz = (np.arange(s3)[None, :] + offsets_z[:, None]) % c3      # (Z, s3)
    Pt = Tr[:, None] * Tc[:, ct]                                 # (P, T, s2, n_qt)
    Pz = Zr[:, None] * Zc[:, cz]                                 # (P, Z, s3, n_qz)

    row_flat = row_start + np.arange(s1 * s2 * s3)
    col_tz = ct[:, None, None, :, None] * c3 + cz[None, :, None, None, :]
    vals, cols = [], []
    for dr in offsets_r:
        cr = (np.arange(s1) + dr) % c1                           # (s1,)
        Pr = Rr * Rc[:, cr]                                      # (P, s1, n_qr)
        A = jnp.einsum('pbac,pia->pbic', W, Pr)                  # (P, n_qt, s1, n_qz)
        B = jnp.einsum('pbic,pzkc->pbizk', A, Pz)                # (P, n_qt, s1, Z, s3)
        V = jnp.einsum('pbizk,ptjb->tzijk', B, Pt)               # (T, Z, s1, s2, s3)
        vals.append(V.ravel())
        col = col_start + cr[None, None, :, None, None] * (c2 * c3) + col_tz
        cols.append(np.broadcast_to(col, V.shape).ravel())
    n_blocks = len(offsets_r) * len(offsets_t) * len(offsets_z)
    return (jnp.concatenate(vals),
            np.tile(row_flat, n_blocks).astype(np.int32),
            np.concatenate(cols).astype(np.int32))


def _bcoo(vals, rows, cols, shape):
    indices = jnp.asarray(np.stack([rows, cols], axis=-1), dtype=jnp.int32)
    return jsparse.BCOO((vals, indices), shape=shape)


def assemble_scalar(R_row, T_row, Z_row, R_col, T_col, Z_col,
                       W_flat, quad_shape, dof_shape, hw_r, hw_t, hw_z):
    """Tensor-product assembly for scalar-valued form mass-like matrices.

    Exploits the separable structure Λ(x) = R(r)·T(θ)·Z(ζ) to assemble via
    1D basis overlap products contracted against a 3D weight tensor.  All
    directions are treated as periodic; boundary conditions are enforced later
    by the extraction operators.

    Parameters
    ----------
    R_row, T_row, Z_row : arrays of shape (s1, n_qr), (s2, n_qt), (s3, n_qz)
        1D basis evaluations at quadrature points for the row form.
    R_col, T_col, Z_col : arrays of same shapes
        1D basis evaluations at quadrature points for the column form.
    W_flat : array of shape (n_q,)
        Scalar quadrature weights at each quadrature point (e.g. J·w).
    quad_shape : tuple (n_qt, n_qr, n_qz)
        Shape of the 3D quadrature grid (matches meshgrid ordering: θ, r, ζ).
    dof_shape : tuple (s1, s2, s3)
        Shape of the DOF grid (radial, poloidal, toroidal).
    hw_r, hw_t, hw_z : int
        Stencil half-widths in each direction (typically the polynomial degree).

    Returns
    -------
    M : jax.experimental.sparse.BCOO, shape (n_dof, n_dof)
    """
    n_dof = int(np.prod(dof_shape))
    pairs = [(W_flat.reshape(quad_shape), R_row, T_row, Z_row,
              R_col, T_col, Z_col, 1.0)]
    vals, rows, cols = _stencil_triplets(
        pairs, dof_shape, dof_shape, (hw_r, hw_t, hw_z))
    return _bcoo(vals, rows, cols, (n_dof, n_dof))


def assemble_vectorial(row_terms, col_terms, W_flat_3x3,
                          quad_shape, comp_shapes, hw,
                          col_comp_shapes=None):
    """Tensor-product assembly for vectorial DOFs with block structure.

    Computes M[i,j] = Σ_{k,l} ∫ (OpΛ_i)_k · W_{kl} · (OpΛ_j)_l dx

    where the operator maps each source component c to one or more output
    components k, each factoring as a product of 1D functions.

    For mass matrices each component has a single identity term
    ``[(c, R, T, Z, +1)]``.  For stiffness matrices (e.g. curl-curl)
    each component may have multiple signed terms.

    Supports rectangular matrices when ``col_comp_shapes`` is provided
    (e.g. derivative matrices mapping between different form degrees).

    Parameters
    ----------
    row_terms : list of lists
        row_terms[c] is a list of (output_idx, R, T, Z, sign) tuples.
    col_terms : list of lists
        Same structure for the column operator.
    W_flat_3x3 : array, shape (n_q, 3, 3)
        Weight tensor indexed by output component pair (k, l).
    quad_shape : tuple (n_qt, n_qr, n_qz)
    comp_shapes : list of tuples (s1, s2, s3)
        DOF grid shape per row source component.
    hw : int
        Stencil half-width (polynomial degree p).
    col_comp_shapes : list of tuples (s1, s2, s3), optional
        DOF grid shape per column source component.  When ``None``,
        defaults to ``comp_shapes`` (square matrix).

    Returns
    -------
    M : jax.experimental.sparse.BCOO
    """
    row_comp_shapes = comp_shapes
    if col_comp_shapes is None:
        col_comp_shapes = row_comp_shapes

    # Normalise hw to a per-pair per-axis table. The bandwidth on a given
    # axis depends on which 1D basis (primal / derivative) each side uses
    # there; passing a 3-D table avoids over-allocating zero positions when
    # both sides use the derivative basis (degree p-1 instead of p).
    n_row_comp = len(row_terms)
    n_col_comp = len(col_terms)
    if isinstance(hw, (int, np.integer)):
        hw_table = [[[int(hw)] * 3 for _ in range(n_col_comp)]
                    for _ in range(n_row_comp)]
    else:
        hw_arr = np.asarray(hw, dtype=int)
        if hw_arr.shape != (n_row_comp, n_col_comp, 3):
            raise ValueError(
                f"hw table shape {hw_arr.shape} does not match "
                f"({n_row_comp}, {n_col_comp}, 3)")
        hw_table = hw_arr.tolist()

    row_starts = np.concatenate(
        [[0], np.cumsum([int(np.prod(s)) for s in row_comp_shapes])])
    col_starts = np.concatenate(
        [[0], np.cumsum([int(np.prod(s)) for s in col_comp_shapes])])

    W_3d = {(k, l): W_flat_3x3[:, k, l].reshape(quad_shape)
            for k in range(3) for l in range(3)}

    vals, rows, cols = [], [], []
    for c_row in range(n_row_comp):
        for c_col in range(n_col_comp):
            pairs = [
                (W_3d[(k, l)], Rk, Tk, Zk, Rl, Tl, Zl, sk * sl)
                for (k, Rk, Tk, Zk, sk) in row_terms[c_row]
                for (l, Rl, Tl, Zl, sl) in col_terms[c_col]
            ]
            v, r, c = _stencil_triplets(
                pairs, row_comp_shapes[c_row], col_comp_shapes[c_col],
                hw_table[c_row][c_col],
                row_start=int(row_starts[c_row]),
                col_start=int(col_starts[c_col]))
            vals.append(v)
            rows.append(r)
            cols.append(c)

    return _bcoo(jnp.concatenate(vals), np.concatenate(rows),
                 np.concatenate(cols),
                 (int(row_starts[-1]), int(col_starts[-1])))


def assemble_dense_mass_matrix(seq, k, dirichlet=True, operators=None):
    """Compatibility wrapper for dense mass matrices from an operator bundle."""
    if operators is None:
        operators = seq.get_operators() if hasattr(seq, 'get_operators') else None
    if operators is None:
        raise ValueError(
            'Assemble operators first, for example with seq.assemble_all_sparse().')
    return operators.todense(seq, 'mass', k, dirichlet=dirichlet)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Element-wise basis evaluation at quadrature points
# ---------------------------------------------------------------------------

def eval_basis_0_ijk(seq, i, j, k):
    """Get the kth component of the ith 0-form evaluated at quadrature point j."""
    j2, j1, j3 = jnp.unravel_index(
        j, (seq.quad.ny, seq.quad.nx, seq.quad.nz))
    _, i1, i2, i3 = seq.basis_0._unravel_index(i)
    return seq.basis_r_jk[i1, j1] * seq.basis_t_jk[i2, j2] * seq.basis_z_jk[i3, j3]


def eval_d_basis_0_ijk(seq, i, j, k):
    """Get the kth component of the gradient of the ith 0-form evaluated at quadrature point j."""
    j2, j1, j3 = jnp.unravel_index(
        j, (seq.quad.ny, seq.quad.nx, seq.quad.nz))
    _, i1, i2, i3 = seq.basis_0._unravel_index(i)
    dr = jnp.where(i1 == seq.basis_0.nt-1, 0.0,
                   seq.d_basis_r_jk[i1, j1])
    dr_m1 = jnp.where(i1 > 0, seq.d_basis_r_jk[i1-1, j1], 0.0)
    dtheta_m1 = jnp.where(
        i2 > 0, seq.d_basis_t_jk[i2-1, j2], seq.d_basis_t_jk[seq.basis_0.nt-1, j2])
    dtheta = seq.d_basis_t_jk[i2, j2]
    dz_m1 = jnp.where(
        i3 > 0, seq.d_basis_z_jk[i3-1, j3], seq.d_basis_z_jk[seq.basis_0.nt-1, j3])
    dz = seq.d_basis_z_jk[i3, j3]
    return jnp.array([
        (dr_m1 - dr) * seq.basis_t_jk[i2, j2] * seq.basis_z_jk[i3, j3],
        seq.basis_r_jk[i1, j1] *
        (dtheta_m1 - dtheta) * seq.basis_z_jk[i3, j3],
        seq.basis_r_jk[i1, j1] * seq.basis_t_jk[i2, j2] * (dz_m1 - dz)
    ])[k]


def eval_basis_1_ijk(seq, i, j, k):
    """Get the kth component of the ith 1-form evaluated at quadrature point j."""
    j2, j1, j3 = jnp.unravel_index(
        j, (seq.quad.ny, seq.quad.nx, seq.quad.nz))
    c, i1, i2, i3 = seq.basis_1._unravel_index(i)
    components = jnp.array([
        seq.d_basis_r_jk[i1, j1] *
        seq.basis_t_jk[i2, j2] * seq.basis_z_jk[i3, j3],
        seq.basis_r_jk[i1, j1] * seq.d_basis_t_jk[i2,
                                                  j2] * seq.basis_z_jk[i3, j3],
        seq.basis_r_jk[i1, j1] *
        seq.basis_t_jk[i2, j2] * seq.d_basis_z_jk[i3, j3]
    ])
    return jnp.where(k == c, components[c], 0.0)


def eval_d_basis_1_ijk(seq, i, j, k):
    """Get the kth component of the curl of the ith 1-form evaluated at quadrature point j."""
    j2, j1, j3 = jnp.unravel_index(
        j, (seq.quad.ny, seq.quad.nx, seq.quad.nz))
    c, i1, i2, i3 = seq.basis_1._unravel_index(i)
    dr = jnp.where(i1 == seq.basis_1.nt-1, 0.0,
                   seq.d_basis_r_jk[i1, j1])
    dr_m1 = jnp.where(i1 > 0, seq.d_basis_r_jk[i1-1, j1], 0.0)
    dtheta_m1 = jnp.where(
        i2 > 0, seq.d_basis_t_jk[i2-1, j2], seq.d_basis_t_jk[seq.basis_1.nt-1, j2])
    dtheta = seq.d_basis_t_jk[i2, j2]
    dz_m1 = jnp.where(
        i3 > 0, seq.d_basis_z_jk[i3-1, j3], seq.d_basis_z_jk[seq.basis_1.nt-1, j3])
    dz = seq.d_basis_z_jk[i3, j3]
    d3dy = seq.basis_r_jk[i1, j1] * \
        (dtheta_m1 - dtheta) * seq.d_basis_z_jk[i3, j3]
    d2dz = seq.basis_r_jk[i1, j1] * \
        seq.d_basis_t_jk[i2, j2] * (dz_m1 - dz)
    d1dz = seq.d_basis_r_jk[i1, j1] * \
        seq.basis_t_jk[i2, j2] * (dz_m1 - dz)
    d3dx = (dr_m1 - dr) * \
        seq.basis_t_jk[i2, j2] * seq.d_basis_z_jk[i3, j3]
    d2dx = (dr_m1 - dr) * \
        seq.d_basis_t_jk[i2, j2] * seq.basis_z_jk[i3, j3]
    d1dy = seq.d_basis_r_jk[i1, j1] * \
        (dtheta_m1 - dtheta) * seq.basis_z_jk[i3, j3]

    curl_matrix = jnp.array([
        [0.0,    d1dz,  -d1dy],
        [-d2dz,  0.0,    d2dx],
        [d3dy,  -d3dx,   0.0]
    ])
    return curl_matrix[c, k]


def eval_basis_2_ijk(seq, i, j, k):
    """Get the kth component of the ith 2-form evaluated at quadrature point j."""
    j2, j1, j3 = jnp.unravel_index(
        j, (seq.quad.ny, seq.quad.nx, seq.quad.nz))
    c, i1, i2, i3 = seq.basis_2._unravel_index(i)
    components = jnp.array([
        seq.basis_r_jk[i1, j1] * seq.d_basis_t_jk[i2,
                                                  j2] * seq.d_basis_z_jk[i3, j3],
        seq.d_basis_r_jk[i1, j1] * seq.basis_t_jk[i2,
                                                  j2] * seq.d_basis_z_jk[i3, j3],
        seq.d_basis_r_jk[i1, j1] *
        seq.d_basis_t_jk[i2, j2] * seq.basis_z_jk[i3, j3]
    ])
    return jnp.where(k == c, components[c], 0.0)


def eval_d_basis_2_ijk(seq, i, j, k):
    """Get the kth component of the divergence of the ith 2-form evaluated at quadrature point j."""
    j2, j1, j3 = jnp.unravel_index(
        j, (seq.quad.ny, seq.quad.nx, seq.quad.nz))
    c, i1, i2, i3 = seq.basis_2._unravel_index(i)
    dr = jnp.where(i1 == seq.basis_2.nt-1, 0.0,
                   seq.d_basis_r_jk[i1, j1])
    dr_m1 = jnp.where(i1 > 0, seq.d_basis_r_jk[i1-1, j1], 0.0)
    dtheta_m1 = jnp.where(
        i2 > 0, seq.d_basis_t_jk[i2-1, j2], seq.d_basis_t_jk[seq.basis_2.nt-1, j2])
    dtheta = seq.d_basis_t_jk[i2, j2]
    dz_m1 = jnp.where(
        i3 > 0, seq.d_basis_z_jk[i3-1, j3], seq.d_basis_z_jk[seq.basis_2.nt-1, j3])
    dz = seq.d_basis_z_jk[i3, j3]

    return jnp.array([
        (dr_m1 - dr) * seq.d_basis_t_jk[i2,
                                        j2] * seq.d_basis_z_jk[i3, j3],
        seq.d_basis_r_jk[i1, j1] *
        (dtheta_m1 - dtheta) * seq.d_basis_z_jk[i3, j3],
        seq.d_basis_r_jk[i1, j1] *
        seq.d_basis_t_jk[i2, j2] * (dz_m1 - dz)
    ])[c]


def eval_basis_3_ijk(seq, i, j, k):
    """Get the kth component of the ith 3-form evaluated at quadrature point j."""
    j2, j1, j3 = jnp.unravel_index(
        j, (seq.quad.ny, seq.quad.nx, seq.quad.nz))
    _, i1, i2, i3 = seq.basis_3._unravel_index(i)
    return seq.d_basis_r_jk[i1, j1] * seq.d_basis_t_jk[i2, j2] * seq.d_basis_z_jk[i3, j3]


def assemble_leray_projection(seq):
    """Assemble the Leray projection matrix."""
    seq.P_Leray = jnp.eye(seq.m2.shape[0]) + \
        seq.weak_grad @ jnp.linalg.pinv(seq.dd3) @ seq.strong_div
