# %%
import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp

import mrx


_INDEX_DTYPE = jnp.int32


def _index_arange(stop):
    return jnp.arange(stop, dtype=_INDEX_DTYPE)


def _as_index_array(values):
    return jnp.asarray(values, dtype=_INDEX_DTYPE)


def _init_triplet_buffers(total_nnz, data_dtype):
    """Allocate fixed-size COO triplet buffers.

    Using one final buffer per field avoids keeping every block alive until a
    terminal ``concatenate``. The buffer sizes are pure shape functions of the
    tensor-product stencil, so this remains compatible with JIT tracing.
    """
    data = jnp.zeros((total_nnz,), dtype=data_dtype)
    rows = jnp.zeros((total_nnz,), dtype=_INDEX_DTYPE)
    cols = jnp.zeros((total_nnz,), dtype=_INDEX_DTYPE)
    return data, rows, cols


def _write_triplet_block(data, rows, cols, offset, vals, row_flat, col_flat):
    """Write one dense tensor-product block into preallocated COO buffers."""
    block_nnz = row_flat.shape[0]
    sl = slice(offset, offset + block_nnz)
    data = data.at[sl].set(vals.ravel())
    rows = rows.at[sl].set(_as_index_array(row_flat))
    cols = cols.at[sl].set(_as_index_array(col_flat))
    return data, rows, cols


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
        _index_arange(s1), _index_arange(s2), _index_arange(s3), indexing='ij')
    row_flat = _as_index_array(jnp.ravel_multi_index(
        (I1, I2, I3), dof_shape, mode='wrap').ravel())

    n_blocks = len(offsets_r) * len(offsets_t) * len(offsets_z)
    block_nnz = row_flat.shape[0]
    data, rows, cols = _init_triplet_buffers(
        n_blocks * block_nnz, W_flat.dtype)
    offset = 0

    for dr in offsets_r:
        M1 = (I1 + dr) % s1
        for dt in offsets_t:
            M2 = (I2 + dt) % s2
            for dz in offsets_z:
                M3 = (I3 + dz) % s3

                # einsum: W[b,a,c] * Pr[i,a] * Pt[j,b] * Pz[k,c] -> M[i,j,k]
                vals = jnp.einsum('bac,ia,jb,kc->ijk',
                                  W_3d, Pr[dr], Pt[dt], Pz[dz])

                col_flat = _as_index_array(jnp.ravel_multi_index(
                    (M1, M2, M3), dof_shape, mode='wrap').ravel())

                data, rows, cols = _write_triplet_block(
                    data, rows, cols, offset, vals, row_flat, col_flat)
                offset += block_nnz

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

    n_blocks = 0
    block_sizes = []
    for c_row in range(len(row_terms)):
        s_row = row_comp_shapes[c_row]
        row_block_nnz = s_row[0] * s_row[1] * s_row[2]
        for c_col in range(len(col_terms)):
            s_col = col_comp_shapes[c_col]
            offsets_r = _offsets(hw, s_col[0])
            offsets_t = _offsets(hw, s_col[1])
            offsets_z = _offsets(hw, s_col[2])
            count = len(offsets_r) * len(offsets_t) * len(offsets_z)
            n_blocks += count
            block_sizes.append(count * row_block_nnz)

    total_nnz = sum(block_sizes)
    data, rows, cols = _init_triplet_buffers(total_nnz, W_flat_3x3.dtype)
    offset = 0

    for c_row in range(len(row_terms)):
        s_row = row_comp_shapes[c_row]
        I1, I2, I3 = jnp.meshgrid(
            _index_arange(s_row[0]), _index_arange(s_row[1]),
            _index_arange(s_row[2]), indexing='ij')
        row_r = _index_arange(s_row[0])
        row_t = _index_arange(s_row[1])
        row_z = _index_arange(s_row[2])
        row_flat = _as_index_array(row_starts[c_row] + jnp.ravel_multi_index(
            (I1, I2, I3), s_row, mode='wrap').ravel())

        for c_col in range(len(col_terms)):
            s_col = col_comp_shapes[c_col]
            offsets_r = _offsets(hw, s_col[0])
            offsets_t = _offsets(hw, s_col[1])
            offsets_z = _offsets(hw, s_col[2])

            for dr in offsets_r:
                cidx_r = (row_r + dr) % s_col[0]
                J1 = (I1 + dr) % s_col[0]
                for dt in offsets_t:
                    cidx_t = (row_t + dt) % s_col[1]
                    J2 = (I2 + dt) % s_col[1]
                    for dz in offsets_z:
                        cidx_z = (row_z + dz) % s_col[2]
                        J3 = (I3 + dz) % s_col[2]

                        vals = jnp.zeros(s_row)
                        for (k, Rk, Tk, Zk, sk) in row_terms[c_row]:
                            for (l, Rl, Tl, Zl, sl) in col_terms[c_col]:
                                vals = vals + sk * sl * jnp.einsum(
                                    'bac,ia,jb,kc->ijk',
                                    W_3d[(k, l)],
                                    Rk * Rl[cidx_r, :],
                                    Tk * Tl[cidx_t, :],
                                    Zk * Zl[cidx_z, :])

                        col_flat = _as_index_array(col_starts[c_col] + jnp.ravel_multi_index(
                            (J1, J2, J3), s_col, mode='wrap').ravel())

                        data, rows, cols = _write_triplet_block(
                            data, rows, cols, offset, vals, row_flat, col_flat)
                        offset += row_flat.shape[0]

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
        _index_arange(s1), _index_arange(s2), _index_arange(s3), indexing='ij')
    row_flat = _as_index_array(jnp.ravel_multi_index(
        (I1, I2, I3), dof_shape, mode='wrap').ravel())

    n_blocks = len(offsets_r) * len(offsets_t) * len(offsets_z)
    block_nnz = row_flat.shape[0]
    data, rows, cols = _init_triplet_buffers(
        n_blocks * block_nnz, W_flat_3x3.dtype)
    offset = 0

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

                col_flat = _as_index_array(jnp.ravel_multi_index(
                    (M1, M2, M3), dof_shape, mode='wrap').ravel())

                data, rows, cols = _write_triplet_block(
                    data, rows, cols, offset, vals, row_flat, col_flat)
                offset += block_nnz

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

    row_indices = jnp.broadcast_to(_index_arange(n1)[:, None], all_cols.shape)
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

    offsets_r = jnp.arange(-hw_r, hw_r + 1, dtype=_INDEX_DTYPE)
    offsets_t = jnp.arange(-hw_t, hw_t + 1, dtype=_INDEX_DTYPE)
    offsets_z = jnp.arange(-hw_z, hw_z + 1, dtype=_INDEX_DTYPE)
    dr, dt, dz = jnp.meshgrid(offsets_r, offsets_t, offsets_z, indexing='ij')
    dr, dt, dz = dr.ravel(), dt.ravel(), dz.ravel()
    per_block = len(dr)

    if col_form.k == 0 or col_form.k == 3:
        s1, s2, s3 = col_form.shape[0]
        # If stencil wider than any dimension, just use all columns
        if (2 * hw_r + 1 >= s1) or (2 * hw_t + 1 >= s2) or (2 * hw_z + 1 >= s3):
            max_nnz = s1 * s2 * s3

            def neighbors(i):
                return _index_arange(max_nnz)
        else:
            max_nnz = per_block

            def neighbors(i):
                _, i1, i2, i3 = row_form._unravel_index(i)
                j1 = (i1 + dr) % s1
                j2 = (i2 + dt) % s2
                j3 = (i3 + dz) % s3
                return _as_index_array(jnp.ravel_multi_index((j1, j2, j3), (s1, s2, s3), mode='wrap'))

    else:  # k == 1, 2, or -1
        shapes = col_form.shape
        n1, n2 = col_form.n1, col_form.n2
        n_total = col_form.n
        comp_starts = jnp.array([0, n1, n1 + n2], dtype=_INDEX_DTYPE)
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
                return _index_arange(max_nnz)
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
                    local = _as_index_array(jnp.ravel_multi_index(
                        (j1, j2, j3), (s1, s2, s3), mode='wrap'))
                    cols.append(local + comp_starts[c])
                return jnp.concatenate(cols)

    return neighbors, max_nnz


def assemble_dense_mass_matrix(seq, k, dirichlet=True, operators=None):
    """Compatibility wrapper for dense mass matrices from an operator bundle."""
    if operators is None:
        operators = seq.get_operators() if hasattr(seq, 'get_operators') else None
    if operators is None:
        raise ValueError(
            'Assemble operators first, for example with seq.assemble_all_sparse().')
    return operators.todense(seq, 'mass', k, dirichlet=dirichlet)


def assemble_dense_hodge_laplacian(seq, k, dirichlet=True, operators=None):
    """Compatibility wrapper for dense Hodge Laplacians from an operator bundle."""
    if operators is None:
        operators = seq.get_operators() if hasattr(seq, 'get_operators') else None
    if operators is None:
        raise ValueError(
            'Assemble operators first, for example with seq.assemble_all_sparse().')
    return operators.todense(seq, 'hodge_laplacian', k, dirichlet=dirichlet)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def grad_1d(d_basis, boundary_type):
    """Compute gradient basis from derivative spline: dΛ(i-1) - dΛ(i)."""
    if boundary_type == 'clamped':
        padded = jnp.pad(d_basis, ((1, 1), (0, 0)))
        return padded[:-1] - padded[1:]
    else:  # periodic
        return jnp.roll(d_basis, 1, axis=0) - d_basis


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


# ---------------------------------------------------------------------------
# Deprecated assembly (element-wise quadrature)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Tensor-product assembly (current)
# ---------------------------------------------------------------------------

def assemble_mass_matrix(seq, k):
    """Assemble the mass matrix using tensor-product contraction."""
    from mrx.utils import diag_EAET
    quad_shape = (seq.quad.ny, seq.quad.nx, seq.quad.nz)
    match k:
        case 0:
            W_flat = seq.jacobian_j * seq.quad.w
            sp = assemble_scalar_tp(
                seq.basis_r_jk, seq.basis_t_jk, seq.basis_z_jk,
                seq.basis_r_jk, seq.basis_t_jk, seq.basis_z_jk,
                W_flat, quad_shape, seq.basis_0.shape[0],
                seq.basis_0.pr, seq.basis_0.pt, seq.basis_0.pz)
            seq.m0_sp = jsparse.BCSR.from_bcoo(sp)
            seq.m0_sp_diaginv = 1.0 / \
                diag_EAET(seq.e0, seq.m0_sp, seq.e0_T)
            seq.m0_sp_diaginv_dbc = 1.0 / \
                diag_EAET(seq.e0_dbc, seq.m0_sp, seq.e0_dbc_T)
        case 1:
            W_3x3 = seq.metric_inv_jkl * \
                (seq.jacobian_j * seq.quad.w)[:, None, None]
            terms = [
                [(0, seq.d_basis_r_jk, seq.basis_t_jk, seq.basis_z_jk, +1)],
                [(1, seq.basis_r_jk, seq.d_basis_t_jk, seq.basis_z_jk, +1)],
                [(2, seq.basis_r_jk, seq.basis_t_jk, seq.d_basis_z_jk, +1)],
            ]
            sp = assemble_vectorial_tp(
                terms, terms, W_3x3, quad_shape,
                list(seq.basis_1.shape),
                seq.basis_1.pr)
            seq.m1_sp = jsparse.BCSR.from_bcoo(sp)
            seq.m1_sp_diaginv = 1.0 / \
                diag_EAET(seq.e1, seq.m1_sp, seq.e1_T)
            seq.m1_sp_diaginv_dbc = 1.0 / \
                diag_EAET(seq.e1_dbc, seq.m1_sp, seq.e1_dbc_T)
        case 2:
            W_3x3 = seq.metric_jkl * \
                (1 / seq.jacobian_j * seq.quad.w)[:, None, None]
            terms = [
                [(0, seq.basis_r_jk, seq.d_basis_t_jk, seq.d_basis_z_jk, +1)],
                [(1, seq.d_basis_r_jk, seq.basis_t_jk, seq.d_basis_z_jk, +1)],
                [(2, seq.d_basis_r_jk, seq.d_basis_t_jk, seq.basis_z_jk, +1)],
            ]
            sp = assemble_vectorial_tp(
                terms, terms, W_3x3, quad_shape,
                list(seq.basis_2.shape),
                seq.basis_2.pr)
            seq.m2_sp = jsparse.BCSR.from_bcoo(sp)
            seq.m2_sp_diaginv = 1.0 / \
                diag_EAET(seq.e2, seq.m2_sp, seq.e2_T)
            seq.m2_sp_diaginv_dbc = 1.0 / \
                diag_EAET(seq.e2_dbc, seq.m2_sp, seq.e2_dbc_T)
        case 3:
            W_flat = (1 / seq.jacobian_j) * seq.quad.w
            sp = assemble_scalar_tp(
                seq.d_basis_r_jk, seq.d_basis_t_jk, seq.d_basis_z_jk,
                seq.d_basis_r_jk, seq.d_basis_t_jk, seq.d_basis_z_jk,
                W_flat, quad_shape, seq.basis_3.shape[0],
                seq.basis_3.pr, seq.basis_3.pt, seq.basis_3.pz)
            seq.m3_sp = jsparse.BCSR.from_bcoo(sp)
            seq.m3_sp_diaginv = 1.0 / \
                diag_EAET(seq.e3, seq.m3_sp, seq.e3_T)
            seq.m3_sp_diaginv_dbc = 1.0 / \
                diag_EAET(seq.e3_dbc, seq.m3_sp, seq.e3_dbc_T)
        case _:
            raise ValueError(
                "Tensor-product assembly supports k=0, 1, 2, 3")


def assemble_derivative_matrix(seq, k):
    """Assemble the exterior derivative matrix using tensor-product contraction."""
    quad_shape = (seq.quad.ny, seq.quad.nx, seq.quad.nz)
    types = seq.basis_0.types
    gr = grad_1d(seq.d_basis_r_jk, types[0])
    gt = grad_1d(seq.d_basis_t_jk, types[1])
    gz = grad_1d(seq.d_basis_z_jk, types[2])
    match k:
        case 0:
            W_3x3 = seq.metric_inv_jkl * \
                (seq.jacobian_j * seq.quad.w)[:, None, None]
            row_terms = [
                [(0, seq.d_basis_r_jk, seq.basis_t_jk, seq.basis_z_jk, +1)],
                [(1, seq.basis_r_jk, seq.d_basis_t_jk, seq.basis_z_jk, +1)],
                [(2, seq.basis_r_jk, seq.basis_t_jk, seq.d_basis_z_jk, +1)],
            ]
            col_terms = [
                [(0, gr, seq.basis_t_jk, seq.basis_z_jk, +1),
                 (1, seq.basis_r_jk, gt, seq.basis_z_jk, +1),
                 (2, seq.basis_r_jk, seq.basis_t_jk, gz, +1)],
            ]
            sp = assemble_vectorial_tp(
                row_terms, col_terms, W_3x3, quad_shape,
                list(seq.basis_1.shape), seq.basis_1.pr,
                col_comp_shapes=list(seq.basis_0.shape))
            seq.d0_sp = jsparse.BCSR.from_bcoo(sp)
            seq.d0_sp_T = jsparse.BCSR.from_bcoo(sp.T)
        case 1:
            W_3x3 = seq.metric_jkl * \
                (1 / seq.jacobian_j * seq.quad.w)[:, None, None]
            dR = seq.d_basis_r_jk
            dT = seq.d_basis_t_jk
            dZ = seq.d_basis_z_jk
            R = seq.basis_r_jk
            T = seq.basis_t_jk
            Z = seq.basis_z_jk
            row_terms = [
                [(0, R, dT, dZ, +1)],
                [(1, dR, T, dZ, +1)],
                [(2, dR, dT, Z, +1)],
            ]
            col_terms = [
                [(1, dR, T, gz, +1),
                 (2, dR, gt, Z, -1)],
                [(0, R, dT, gz, -1),
                 (2, gr, dT, Z, +1)],
                [(0, R, gt, dZ, +1),
                 (1, gr, T, dZ, -1)],
            ]
            sp = assemble_vectorial_tp(
                row_terms, col_terms, W_3x3, quad_shape,
                list(seq.basis_2.shape), seq.basis_2.pr,
                col_comp_shapes=list(seq.basis_1.shape))
            seq.d1_sp = jsparse.BCSR.from_bcoo(sp)
            seq.d1_sp_T = jsparse.BCSR.from_bcoo(sp.T)
        case 2:
            W_scalar = (1 / seq.jacobian_j) * seq.quad.w
            W_1x1 = W_scalar.reshape(-1, 1, 1)
            dR = seq.d_basis_r_jk
            dT = seq.d_basis_t_jk
            dZ = seq.d_basis_z_jk
            row_terms = [
                [(0, dR, dT, dZ, +1)],
            ]
            col_terms = [
                [(0, gr, dT, dZ, +1)],
                [(0, dR, gt, dZ, +1)],
                [(0, dR, dT, gz, +1)],
            ]
            sp = assemble_vectorial_tp(
                row_terms, col_terms, W_1x1, quad_shape,
                list(seq.basis_3.shape), seq.basis_3.pr,
                col_comp_shapes=list(seq.basis_2.shape))
            seq.d2_sp = jsparse.BCSR.from_bcoo(sp)
            seq.d2_sp_T = jsparse.BCSR.from_bcoo(sp.T)
        case _:
            raise ValueError(
                "Tensor-product derivative assembly supports k=0, 1, 2")


def assemble_hodge_laplacian(seq, k):
    """Assemble the stiffness matrix (δd) using tensor-product contraction."""
    from mrx.utils import diag_EAET, diag_schur_complement
    quad_shape = (seq.quad.ny, seq.quad.nx, seq.quad.nz)
    types = seq.basis_0.types
    gr = grad_1d(seq.d_basis_r_jk, types[0])
    gt = grad_1d(seq.d_basis_t_jk, types[1])
    gz = grad_1d(seq.d_basis_z_jk, types[2])
    match k:
        case 0:
            W_3x3 = seq.metric_inv_jkl * \
                (seq.jacobian_j * seq.quad.w)[:, None, None]
            grad_basis_1d = [
                (gr, seq.basis_t_jk, seq.basis_z_jk),
                (seq.basis_r_jk, gt, seq.basis_z_jk),
                (seq.basis_r_jk, seq.basis_t_jk, gz),
            ]
            sp = assemble_stiffness_scalar_tp(
                grad_basis_1d, grad_basis_1d, W_3x3, quad_shape,
                seq.basis_0.shape[0],
                seq.basis_0.pr, seq.basis_0.pt, seq.basis_0.pz)
            seq.grad_grad_sp = jsparse.BCSR.from_bcoo(sp)
            seq.dd0_sp_diaginv = 1.0 / \
                diag_EAET(seq.e0, seq.grad_grad_sp, seq.e0_T)
            seq.dd0_sp_diaginv_dbc = 1.0 / \
                diag_EAET(seq.e0_dbc, seq.grad_grad_sp, seq.e0_dbc_T)
        case 1:
            W_3x3 = seq.metric_jkl * \
                (1 / seq.jacobian_j * seq.quad.w)[:, None, None]
            dR = seq.d_basis_r_jk
            dT = seq.d_basis_t_jk
            dZ = seq.d_basis_z_jk
            R = seq.basis_r_jk
            T = seq.basis_t_jk
            Z = seq.basis_z_jk
            curl_terms = [
                [(1, dR, T, gz, +1),
                 (2, dR, gt, Z, -1)],
                [(0, R, dT, gz, -1),
                 (2, gr, dT, Z, +1)],
                [(0, R, gt, dZ, +1),
                 (1, gr, T, dZ, -1)],
            ]
            sp = assemble_vectorial_tp(
                curl_terms, curl_terms, W_3x3, quad_shape,
                list(seq.basis_1.shape), seq.basis_1.pr)
            seq.curl_curl_sp = jsparse.BCSR.from_bcoo(sp)
            d_stiff = diag_EAET(seq.e1, seq.curl_curl_sp, seq.e1_T)
            d_schur = diag_schur_complement(
                lambda v: seq.e0 @ (seq.d0_sp_T @ (seq.e1_T @ v)),
                seq.m0_sp_diaginv, seq.n1)
            seq.dd1_sp_diaginv = 1.0 / (d_stiff + d_schur)
            d_stiff_dbc = diag_EAET(
                seq.e1_dbc, seq.curl_curl_sp, seq.e1_dbc_T)
            d_schur_dbc = diag_schur_complement(
                lambda v: seq.e0_dbc @ (seq.d0_sp_T @
                                        (seq.e1_dbc_T @ v)),
                seq.m0_sp_diaginv_dbc, seq.n1_dbc)
            seq.dd1_sp_diaginv_dbc = 1.0 / (d_stiff_dbc + d_schur_dbc)
        case 2:
            W_scalar = (1 / seq.jacobian_j) * seq.quad.w
            W_3x3 = W_scalar[:, None, None] * jnp.ones((1, 3, 3))
            div_terms = [
                [(0, gr, seq.d_basis_t_jk, seq.d_basis_z_jk, +1)],
                [(1, seq.d_basis_r_jk, gt, seq.d_basis_z_jk, +1)],
                [(2, seq.d_basis_r_jk, seq.d_basis_t_jk, gz, +1)],
            ]
            sp = assemble_vectorial_tp(
                div_terms, div_terms, W_3x3, quad_shape,
                list(seq.basis_2.shape), seq.basis_2.pr)
            seq.div_div_sp = jsparse.BCSR.from_bcoo(sp)
            d_stiff = diag_EAET(seq.e2, seq.div_div_sp, seq.e2_T)
            d_schur = diag_schur_complement(
                lambda v: seq.e1 @ (seq.d1_sp_T @ (seq.e2_T @ v)),
                seq.m1_sp_diaginv, seq.n2)
            seq.dd2_sp_diaginv = 1.0 / (d_stiff + d_schur)
            d_stiff_dbc = diag_EAET(
                seq.e2_dbc, seq.div_div_sp, seq.e2_dbc_T)
            d_schur_dbc = diag_schur_complement(
                lambda v: seq.e1_dbc @ (seq.d1_sp_T @
                                        (seq.e2_dbc_T @ v)),
                seq.m1_sp_diaginv_dbc, seq.n2_dbc)
            seq.dd2_sp_diaginv_dbc = 1.0 / (d_stiff_dbc + d_schur_dbc)
        case 3:
            d_schur = diag_schur_complement(
                lambda v: seq.e2 @ (seq.d2_sp_T @ (seq.e3_T @ v)),
                seq.m2_sp_diaginv, seq.n3)
            seq.dd3_sp_diaginv = 1.0 / d_schur
            d_schur_dbc = diag_schur_complement(
                lambda v: seq.e2_dbc @ (seq.d2_sp_T @
                                        (seq.e3_dbc_T @ v)),
                seq.m2_sp_diaginv_dbc, seq.n3_dbc)
            seq.dd3_sp_diaginv_dbc = 1.0 / d_schur_dbc
        case _:
            raise ValueError("k must be 0, 1, 2, or 3")


def assemble_leray_projection(seq):
    """Assemble the Leray projection matrix."""
    seq.P_Leray = jnp.eye(seq.m2.shape[0]) + \
        seq.weak_grad @ jnp.linalg.pinv(seq.dd3) @ seq.strong_div
