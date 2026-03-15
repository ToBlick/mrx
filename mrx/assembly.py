# %%
import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp

import mrx


def assemble(getter_1, getter_2, W, n1, n2):
    """
    Assemble a matrix M[a, b] = Σ_{a,j,k} Λ1[a,j,i] * W[j,i,k] * Λ2[b,j,k]

    Parameters
    ----------
    getter_1 : callable
        Function (a, j, k) -> scalar. kth component of form a evaluated at quadrature point j.
    getter_2 : callable
        Function (b, j, k) -> scalar. kth component of form b evaluated at quadrature point j.
    W : jnp.ndarray, shape (n_q, 3, 3)
        Weight tensor combining metric, Jacobian, and quadrature weights.
        (For example: G_inv[q, ...] * J[q] * w[q])
    n1 : int
        Number of row basis functions.
    n2 : int
        Number of column basis functions.

    Returns
    -------
    M : jnp.ndarray, shape (n1, n2)
        The assembled matrix.
    """

    n_q, d, _ = W.shape

    get_A_k = jax.vmap(getter_1, in_axes=(None, None, 0))  # over k
    get_B_k = jax.vmap(getter_2, in_axes=(None, None, 0))  # over k

    def get_A_jk(i, js, ks):
        return jax.lax.map(lambda j: get_A_k(i, j, ks), js,
                           batch_size=mrx.MAP_BATCH_SIZE_INNER)

    def get_B_jk(m, js, ks):
        return jax.lax.map(lambda j: get_B_k(m, j, ks), js,
                           batch_size=mrx.MAP_BATCH_SIZE_INNER)  # over j (quadrature points)

    def body_fun(i):
        ΛA_i = get_A_jk(i, jnp.arange(n_q), jnp.arange(d))

        def compute_row(m):
            ΛB_m = get_B_jk(m, jnp.arange(n_q), jnp.arange(d))
            return jnp.einsum("jk,jkm,jm->", ΛA_i, W, ΛB_m)

        return jax.lax.map(compute_row, jnp.arange(n2),
                           batch_size=mrx.MAP_BATCH_SIZE_OUTER)  # over m (basis functions for columns) -- batched

    # over i (rows) sequentially - not batched
    M = jax.lax.map(body_fun, jnp.arange(n1), batch_size=None)
    return M


def assemble_scalar_tp(R_row, T_row, Z_row, R_col, T_col, Z_col,
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
    s1, s2, s3 = dof_shape
    n_dof = s1 * s2 * s3

    W_3d = W_flat.reshape(quad_shape)  # (n_qt, n_qr, n_qz)

    # Unique periodic offsets per direction, avoiding mod-s duplicates
    def _offsets(hw, s):
        if 2 * hw + 1 <= s:
            return range(-hw, hw + 1)
        return range(-(s // 2), s - s // 2)

    offsets_r = _offsets(hw_r, s1)
    offsets_t = _offsets(hw_t, s2)
    offsets_z = _offsets(hw_z, s3)

    # Precompute 1D overlap products per offset
    Pr = {dr: R_row * jnp.roll(R_col, -dr, axis=0) for dr in offsets_r}
    Pt = {dt: T_row * jnp.roll(T_col, -dt, axis=0) for dt in offsets_t}
    Pz = {dz: Z_row * jnp.roll(Z_col, -dz, axis=0) for dz in offsets_z}

    # Row DOF grid indices (flat)
    I1, I2, I3 = jnp.meshgrid(
        jnp.arange(s1), jnp.arange(s2), jnp.arange(s3), indexing='ij')
    row_flat = jnp.ravel_multi_index(
        (I1, I2, I3), dof_shape, mode='wrap').ravel()

    all_data = []
    all_rows = []
    all_cols = []

    for dr in offsets_r:
        M1 = (I1 + dr) % s1
        for dt in offsets_t:
            M2 = (I2 + dt) % s2
            for dz in offsets_z:
                M3 = (I3 + dz) % s3

                # einsum: W[b,a,c] * Pr[i,a] * Pt[j,b] * Pz[k,c] -> M[i,j,k]
                vals = jnp.einsum('bac,ia,jb,kc->ijk',
                                  W_3d, Pr[dr], Pt[dt], Pz[dz])

                col_flat = jnp.ravel_multi_index(
                    (M1, M2, M3), dof_shape, mode='wrap').ravel()

                all_data.append(vals.ravel())
                all_rows.append(row_flat)
                all_cols.append(col_flat)

    data = jnp.concatenate(all_data)
    rows = jnp.concatenate(all_rows)
    cols = jnp.concatenate(all_cols)
    indices = jnp.stack([rows, cols], axis=-1)

    return jsparse.BCOO((data, indices), shape=(n_dof, n_dof))


def assemble_vectorial_tp(row_terms, col_terms, W_flat_3x3,
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

    # Row sizes and offsets
    row_sizes = [s[0] * s[1] * s[2] for s in row_comp_shapes]
    n_row_total = sum(row_sizes)
    row_starts = []
    acc = 0
    for sz in row_sizes:
        row_starts.append(acc)
        acc += sz

    # Col sizes and offsets
    col_sizes = [s[0] * s[1] * s[2] for s in col_comp_shapes]
    n_col_total = sum(col_sizes)
    col_starts = []
    acc = 0
    for sz in col_sizes:
        col_starts.append(acc)
        acc += sz

    def _offsets(hw, s):
        if 2 * hw + 1 <= s:
            return range(-hw, hw + 1)
        return range(-(s // 2), s - s // 2)

    # Precompute W_{kl} reshaped to 3D
    W_3d = {}
    for k in range(3):
        for l in range(3):
            W_3d[(k, l)] = W_flat_3x3[:, k, l].reshape(quad_shape)

    all_data = []
    all_rows = []
    all_cols = []

    for c_row in range(len(row_terms)):
        s_row = row_comp_shapes[c_row]
        I1, I2, I3 = jnp.meshgrid(
            jnp.arange(s_row[0]), jnp.arange(s_row[1]),
            jnp.arange(s_row[2]), indexing='ij')
        row_flat = row_starts[c_row] + jnp.ravel_multi_index(
            (I1, I2, I3), s_row, mode='wrap').ravel()

        for c_col in range(len(col_terms)):
            s_col = col_comp_shapes[c_col]
            offsets_r = _offsets(hw, s_col[0])
            offsets_t = _offsets(hw, s_col[1])
            offsets_z = _offsets(hw, s_col[2])

            # Precompute 1D overlap products for all term pairs and offsets
            Pr = {}
            Pt = {}
            Pz = {}
            for (k, Rk, Tk, Zk, _sk) in row_terms[c_row]:
                for (l, Rl, Tl, Zl, _sl) in col_terms[c_col]:
                    for dr in offsets_r:
                        cidx = (jnp.arange(s_row[0]) + dr) % s_col[0]
                        Pr[(k, l, dr)] = Rk * Rl[cidx, :]
                    for dt in offsets_t:
                        cidx = (jnp.arange(s_row[1]) + dt) % s_col[1]
                        Pt[(k, l, dt)] = Tk * Tl[cidx, :]
                    for dz in offsets_z:
                        cidx = (jnp.arange(s_row[2]) + dz) % s_col[2]
                        Pz[(k, l, dz)] = Zk * Zl[cidx, :]

            for dr in offsets_r:
                J1 = (I1 + dr) % s_col[0]
                for dt in offsets_t:
                    J2 = (I2 + dt) % s_col[1]
                    for dz in offsets_z:
                        J3 = (I3 + dz) % s_col[2]

                        vals = jnp.zeros(s_row)
                        for (k, _, _, _, sk) in row_terms[c_row]:
                            for (l, _, _, _, sl) in col_terms[c_col]:
                                vals = vals + sk * sl * jnp.einsum(
                                    'bac,ia,jb,kc->ijk',
                                    W_3d[(k, l)],
                                    Pr[(k, l, dr)],
                                    Pt[(k, l, dt)],
                                    Pz[(k, l, dz)])

                        col_flat = col_starts[c_col] + jnp.ravel_multi_index(
                            (J1, J2, J3), s_col, mode='wrap').ravel()

                        all_data.append(vals.ravel())
                        all_rows.append(row_flat)
                        all_cols.append(col_flat)

    data = jnp.concatenate(all_data)
    rows = jnp.concatenate(all_rows)
    cols = jnp.concatenate(all_cols)
    indices = jnp.stack([rows, cols], axis=-1)

    return jsparse.BCOO((data, indices), shape=(n_row_total, n_col_total))


def assemble_stiffness_scalar_tp(row_basis_1d, col_basis_1d, W_flat_3x3,
                                 quad_shape, dof_shape, hw_r, hw_t, hw_z):
    """Tensor-product assembly for stiffness-like matrix with scalar DOFs.

    Computes M_ij = Σ_{a,b} ∫ (dΛ_i)_a · W_{ab} · (dΛ_j)_b dx

    where dΛ is a vector-valued operator (e.g. gradient) applied to scalar
    basis functions, producing 3 components, each factoring as a product
    of 1D functions.  All 9 (a,b) blocks contribute to a single scalar
    DOF matrix.

    Parameters
    ----------
    row_basis_1d : list of 3 tuples (R, T, Z)
        Per-component 1D basis evaluations for the row operator.
        All arrays share the same DOF dimension per direction.
    col_basis_1d : list of 3 tuples (R, T, Z)
        Same, for the column operator.
    W_flat_3x3 : array, shape (n_q, 3, 3)
        Weight tensor at each quadrature point.
    quad_shape : tuple (n_qt, n_qr, n_qz)
    dof_shape : tuple (s1, s2, s3)
    hw_r, hw_t, hw_z : int
        Stencil half-widths per direction.

    Returns
    -------
    M : jax.experimental.sparse.BCOO, shape (n_dof, n_dof)
    """
    s1, s2, s3 = dof_shape
    n_dof = s1 * s2 * s3

    def _offsets(hw, s):
        if 2 * hw + 1 <= s:
            return range(-hw, hw + 1)
        return range(-(s // 2), s - s // 2)

    offsets_r = _offsets(hw_r, s1)
    offsets_t = _offsets(hw_t, s2)
    offsets_z = _offsets(hw_z, s3)

    # Precompute 1D overlap products for all (a, b) blocks and offsets
    Pr = {}
    Pt = {}
    Pz = {}
    for a in range(3):
        R_row, T_row, Z_row = row_basis_1d[a]
        for b in range(3):
            R_col, T_col, Z_col = col_basis_1d[b]
            for dr in offsets_r:
                Pr[(a, b, dr)] = R_row * jnp.roll(R_col, -dr, axis=0)
            for dt in offsets_t:
                Pt[(a, b, dt)] = T_row * jnp.roll(T_col, -dt, axis=0)
            for dz in offsets_z:
                Pz[(a, b, dz)] = Z_row * jnp.roll(Z_col, -dz, axis=0)

    # Precompute W_{ab} reshaped to 3D
    W_ab_3d = {}
    for a in range(3):
        for b in range(3):
            W_ab_3d[(a, b)] = W_flat_3x3[:, a, b].reshape(quad_shape)

    # Row DOF grid
    I1, I2, I3 = jnp.meshgrid(
        jnp.arange(s1), jnp.arange(s2), jnp.arange(s3), indexing='ij')
    row_flat = jnp.ravel_multi_index(
        (I1, I2, I3), dof_shape, mode='wrap').ravel()

    all_data = []
    all_rows = []
    all_cols = []

    for dr in offsets_r:
        M1 = (I1 + dr) % s1
        for dt in offsets_t:
            M2 = (I2 + dt) % s2
            for dz in offsets_z:
                M3 = (I3 + dz) % s3

                # Sum contributions from all 9 (a, b) blocks
                vals = jnp.zeros((s1, s2, s3))
                for a in range(3):
                    for b in range(3):
                        vals = vals + jnp.einsum(
                            'bac,ia,jb,kc->ijk',
                            W_ab_3d[(a, b)],
                            Pr[(a, b, dr)],
                            Pt[(a, b, dt)],
                            Pz[(a, b, dz)])

                col_flat = jnp.ravel_multi_index(
                    (M1, M2, M3), dof_shape, mode='wrap').ravel()

                all_data.append(vals.ravel())
                all_rows.append(row_flat)
                all_cols.append(col_flat)

    data = jnp.concatenate(all_data)
    rows = jnp.concatenate(all_rows)
    cols = jnp.concatenate(all_cols)
    indices = jnp.stack([rows, cols], axis=-1)

    return jsparse.BCOO((data, indices), shape=(n_dof, n_dof))


def assemble_sparse(getter_1, getter_2, W, n1, n2, max_nnz, neighbors):
    """
    Assemble a sparse BCOO matrix M[a, b] = Σ_{j,k} Λ1[a,j,k] * W[j,k,m] * Λ2[b,j,m]

    Parameters
    ----------
    getter_1 : callable
        Function (a, j, k) -> scalar.
    getter_2 : callable
        Function (b, j, k) -> scalar.
    W : jnp.ndarray, shape (n_q, d, d)
        Weight tensor combining metric, Jacobian, and quadrature weights.
    n1 : int
        Number of row basis functions.
    n2 : int
        Number of column basis functions.
    max_nnz : int
        Upper bound on the number of non-zero entries per row.
    neighbors : callable, optional
        Function (i) -> jnp.ndarray of shape (max_nnz,) returning the column
        indices whose supports overlap with row i. When provided, only those
        entries are computed — no sorting/filtering is needed. When None, all
        n2 columns are evaluated and the top max_nnz non-zeros are kept.

    Returns
    -------
    M : jax.experimental.sparse.BCOO, shape (n1, n2)
        The assembled sparse matrix.
    """

    n_q, d, _ = W.shape
    max_nnz = min(max_nnz, n2)

    get_A_k = jax.vmap(getter_1, in_axes=(
        None, None, 0))  # over k - dimensions
    get_B_k = jax.vmap(getter_2, in_axes=(None, None, 0))  # over k

    def get_A_jk(i, js, ks):
        return jax.lax.map(lambda j: get_A_k(i, j, ks), js,
                           batch_size=mrx.MAP_BATCH_SIZE_INNER)

    def get_B_jk(m, js, ks):
        return jax.lax.map(lambda j: get_B_k(m, j, ks), js,
                           batch_size=mrx.MAP_BATCH_SIZE_INNER)

    # Fast path: only compute the max_nnz entries we know can be nonzero
    def process_row(i):
        ΛA_i = get_A_jk(i, jnp.arange(n_q), jnp.arange(d))
        cols = neighbors(i)  # shape (max_nnz,)

        def compute_entry(m):
            ΛB_m = get_B_jk(m, jnp.arange(n_q), jnp.arange(d))
            return jnp.einsum("jk,jkm,jm->", ΛA_i, W, ΛB_m)

        # over m (columns)
        vals = jax.lax.map(compute_entry, cols,
                           batch_size=mrx.MAP_BATCH_SIZE_OUTER)
        return vals, cols

    all_vals, all_cols = jax.lax.map(
        process_row, jnp.arange(n1), batch_size=None)  # (n1, max_nnz)
    # over i (rows) sequentially - not batched

    row_indices = jnp.broadcast_to(
        jnp.arange(n1)[:, None], all_cols.shape
    )
    indices = jnp.stack([row_indices.ravel(), all_cols.ravel()], axis=-1)
    data = all_vals.ravel()

    return jsparse.BCOO((data, indices), shape=(n1, n2))


def build_neighbors(row_form, col_form=None):
    """Build a neighbors function for sparse assembly based on B-spline support overlap.

    For each row index, returns the fixed-size array of column indices whose
    supports can overlap. Uses a conservative half-width of ``p+1`` per
    direction, which covers all assembly types (mass, stiffness, derivative).

    Parameters
    ----------
    row_form : DifferentialForm
        Form defining the row basis (used to unravel the row index).
    col_form : DifferentialForm, optional
        Form defining the column basis (used to ravel neighbor column indices).
        Defaults to ``row_form`` when both row and column forms are the same.

    Returns
    -------
    neighbors : callable
        Function ``(i: int) -> jnp.ndarray`` of shape ``(max_nnz,)``.
    max_nnz : int
        Number of neighbor column indices per row.
    """
    if col_form is None:
        col_form = row_form

    hw_r = max(row_form.pr, col_form.pr)
    hw_t = max(row_form.pt, col_form.pt)
    hw_z = max(row_form.pz, col_form.pz)

    offsets_r = jnp.arange(-hw_r, hw_r + 1)
    offsets_t = jnp.arange(-hw_t, hw_t + 1)
    offsets_z = jnp.arange(-hw_z, hw_z + 1)
    dr, dt, dz = jnp.meshgrid(offsets_r, offsets_t, offsets_z, indexing='ij')
    dr, dt, dz = dr.ravel(), dt.ravel(), dz.ravel()
    per_block = len(dr)

    if col_form.k == 0 or col_form.k == 3:
        s1, s2, s3 = col_form.shape[0]
        # If stencil wider than any dimension, just use all columns
        if (2 * hw_r + 1 >= s1) or (2 * hw_t + 1 >= s2) or (2 * hw_z + 1 >= s3):
            max_nnz = s1 * s2 * s3

            def neighbors(i):
                return jnp.arange(max_nnz)
        else:
            max_nnz = per_block

            def neighbors(i):
                _, i1, i2, i3 = row_form._unravel_index(i)
                j1 = (i1 + dr) % s1
                j2 = (i2 + dt) % s2
                j3 = (i3 + dz) % s3
                return jnp.ravel_multi_index((j1, j2, j3), (s1, s2, s3), mode='wrap')

    else:  # k == 1, 2, or -1
        shapes = col_form.shape
        n1, n2 = col_form.n1, col_form.n2
        n_total = col_form.n
        comp_starts = jnp.array([0, n1, n1 + n2], dtype=jnp.int32)
        # Check if stencil overflows any component dimension
        overflow = False
        for c in range(3):
            sc1, sc2, sc3 = shapes[c]
            if (2 * hw_r + 1 >= sc1) or (2 * hw_t + 1 >= sc2) or (2 * hw_z + 1 >= sc3):
                overflow = True
                break
        if overflow:
            max_nnz = n_total

            def neighbors(i):
                return jnp.arange(max_nnz)
        else:
            max_nnz = 3 * per_block

            def neighbors(i):
                _, i1, i2, i3 = row_form._unravel_index(i)
                cols = []
                for c in range(3):
                    s1, s2, s3 = shapes[c]
                    j1 = (i1 + dr) % s1
                    j2 = (i2 + dt) % s2
                    j3 = (i3 + dz) % s3
                    local = jnp.ravel_multi_index(
                        (j1, j2, j3), (s1, s2, s3), mode='wrap')
                    cols.append(local + comp_starts[c])
                return jnp.concatenate(cols)

    return neighbors, max_nnz
