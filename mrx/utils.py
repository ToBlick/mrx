# %%
import os
from typing import Any, Callable, Optional

import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg

import mrx


def is_running_in_github_actions():
    """
    Checks if the current Python script is running within a GitHub Actions environment.
    """
    return os.getenv("GITHUB_ACTIONS") == "true"


def double_map(f, xs, ys):
    """Apply f(x, y) over all combinations of xs and ys using nested jax.lax.map.

    Batch sizes are controlled by mrx.MAP_BATCH_SIZE_OUTER (outer loop)
    and mrx.MAP_BATCH_SIZE_INNER (inner loop).

    Args:
        f: Function (x, y) -> array.
        xs: Outer loop values.
        ys: Inner loop values.

    Returns:
        Array of shape (len(xs), len(ys), ...).
    """
    def outer(x):
        return jax.lax.map(lambda y: f(x, y), ys, batch_size=mrx.MAP_BATCH_SIZE_INNER)
    return jax.lax.map(outer, xs, batch_size=mrx.MAP_BATCH_SIZE_OUTER)


def norm_2(u: jnp.ndarray, Seq) -> float:
    """Compute the L2 norm of a vector field.

    Args:
        u: Vector field.
        Seq: DeRham sequence object.

    Returns:
        L2 norm of the vector field.
    """
    return (u @ Seq.M2 @ u)**0.5


def jacobian_determinant(f: Callable[[jnp.ndarray], jnp.ndarray]) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Compute the determinant of the Jacobian matrix for a given function.

    Args:
        f: Function mapping from R^n to R^n for which to compute the Jacobian determinant

    Returns:
        Function that computes the Jacobian determinant at a given point
    """
    return lambda x: jnp.linalg.det(jax.jacfwd(f)(x))


def det33(mat: jnp.ndarray) -> jnp.ndarray:
    """Compute the determinant of a 3x3 matrix using explicit formula.

    This function computes the determinant using the rule of Sarrus, which is
    more efficient than general determinant computation for 3x3 matrices.

    Args:
        mat: 3x3 matrix for which to compute the determinant
    Returns:
        The determinant of the input matrix
    """
    m1, m2, m3 = mat[0]
    m4, m5, m6 = mat[1]
    m7, m8, m9 = mat[2]
    return m1 * (m5 * m9 - m6 * m8) - m2 * (m4 * m9 - m6 * m7) + m3 * (m4 * m8 - m5 * m7)


def inv33(mat: jnp.ndarray) -> jnp.ndarray:
    """Compute the inverse of a 3x3 matrix using explicit formula.

    This function computes the inverse using the adjugate matrix formula,
    which is more efficient than general matrix inversion for 3x3 matrices.

    Args:
        mat: 3x3 matrix to invert

    Returns:
        The inverse of the input matrix
    """
    m1, m2, m3 = mat[0]
    m4, m5, m6 = mat[1]
    m7, m8, m9 = mat[2]
    det = m1 * (m5 * m9 - m6 * m8) + m4 * \
        (m8 * m3 - m2 * m9) + m7 * (m2 * m6 - m3 * m5)
    # Return zero matrix if determinant is zero
    return jnp.where(
        jnp.abs(det) < 1e-10,  # Careful with this tolerance
        jnp.zeros((3, 3)),
        jnp.array([
            [m5 * m9 - m6 * m8, m3 * m8 - m2 * m9, m2 * m6 - m3 * m5],
            [m6 * m7 - m4 * m9, m1 * m9 - m3 * m7, m3 * m4 - m1 * m6],
            [m4 * m8 - m5 * m7, m2 * m7 - m1 * m8, m1 * m5 - m2 * m4],
        ]) / det
    )


def div(F: Callable[[jnp.ndarray], jnp.ndarray]) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Compute the divergence of a vector field.

    Args:
        F: Vector field function for which to compute the divergence

    Returns:
        Function that computes the divergence at a given point
    """
    def div_F(x: jnp.ndarray) -> jnp.ndarray:
        DF = jax.jacfwd(F)(x)
        return jnp.trace(DF) * jnp.ones(1)
    return div_F


def curl(F: Callable[[jnp.ndarray], jnp.ndarray]) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Compute the curl of a vector field in 3D.

    Args:
        F: Vector field function for which to compute the curl

    Returns:
        Function that computes the curl at a given point
    """
    def curl_F(x: jnp.ndarray) -> jnp.ndarray:
        DF = jax.jacfwd(F)(x)
        return jnp.array([DF[2, 1] - DF[1, 2],
                         DF[0, 2] - DF[2, 0],
                         DF[1, 0] - DF[0, 1]])
    return curl_F


def grad(F: Callable[[jnp.ndarray], jnp.ndarray]) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Compute the gradient of a scalar field.

    Args:
        F: Scalar field function for which to compute the gradient

    Returns:
        Function that computes the gradient at a given point
    """
    def grad_F(x: jnp.ndarray) -> jnp.ndarray:
        DF = jax.jacfwd(F)(x)
        return jnp.ravel(DF)
    return grad_F


def l2_product(f: Callable[[jnp.ndarray], jnp.ndarray],
               g: Callable[[jnp.ndarray], jnp.ndarray],
               Q: Any,
               F: Callable[[jnp.ndarray], jnp.ndarray] = lambda x: x) -> jnp.ndarray:
    """Compute the L2 inner product of two functions over a domain.

    Computes the integral of f·g over the domain defined by the quadrature rule Q,
    with optional coordinate transformation F.

    Args:
        f: First function in the inner product
        g: Second function in the inner product
        Q: Quadrature rule object with attributes x (points) and w (weights)
        F: Optional coordinate transformation function (default is identity)

    Returns:
        The L2 inner product value
    """
    J_i = jax.lax.map(jacobian_determinant(F), Q.x,
                      batch_size=mrx.MAP_BATCH_SIZE_INNER)
    f_ij = jax.lax.map(f, Q.x, batch_size=mrx.MAP_BATCH_SIZE_INNER)
    g_ij = jax.lax.map(g, Q.x, batch_size=mrx.MAP_BATCH_SIZE_INNER)
    return jnp.einsum("ij,ij,i,i->", f_ij, g_ij, J_i, Q.w)


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


def evaluate_at_xq_deprecated(getter, dofs, n_q, d):
    """
    Evaluate a finite element function at quadrature points.

    Parameters
    ----------
    getter : callable
        Function (i, j, k) -> scalar. kth component of form i evaluated at quadrature point j.
    dofs : jnp.ndarray, shape (m,)
        Degrees of freedom of the finite element function, already contracted with extraction matrices
    n_q : int
        Number of quadrature points.
    d : int
        Number of dimensions.

    Returns
    -------
    f_h_jk : jnp.ndarray, shape (n_q, d)
        Function values at quadrature points.
    """
    # Evaluate the finite element function at quadrature points
    get_f_k = jax.vmap(getter, in_axes=(None, None, 0))  # over k (dimensions)

    def get_f_jk(i, js, ks):
        return jax.lax.map(lambda j: get_f_k(i, j, ks), js,
                           batch_size=mrx.MAP_BATCH_SIZE_INNER)

    v_i = dofs  # shape (n_i,)

    def body_fun(carry, i):
        L_i = get_f_jk(i, jnp.arange(n_q), jnp.arange(d))  # shape (n_q, d)
        # broadcast scalar over last axis (dimesions)
        carry += L_i * v_i[i]
        return carry, None

    R_init = jnp.zeros_like(
        get_f_jk(0, jnp.arange(n_q), jnp.arange(d)))  # shape (n_q, d)
    R, _ = jax.lax.scan(body_fun, R_init, jnp.arange(v_i.shape[0]))
    return R


def integrate_against_deprecated(getter, w_jk, n):
    """
    Integrate a function represented at quadrature points against a set of basis functions.

    Args:
        getter (callable): Function (i, j, k) -> scalar. kth component of form i evaluated at quadrature point j.
        w_jk (jnp.ndarray): Function values at quadrature points, shape (n_q, d).
        n (int): Number of basis functions.

    Returns:
        jnp.ndarray: Integrated values, shape (n,). Entries are given by
        ∑_{j,k} Λ[i,j,k] * w[j,k]
    """
    n_q, d = w_jk.shape
    get_f_k = jax.vmap(getter, in_axes=(None, None, 0))  # over k (dimensions)

    def get_f_jk(i, js, ks):
        return jax.lax.map(lambda j: get_f_k(i, j, ks), js,
                           batch_size=mrx.MAP_BATCH_SIZE_INNER)

    def body_fun(carry, i):
        L_i = get_f_jk(i, jnp.arange(n_q), jnp.arange(d))  # shape (n_q, d)
        return None, jnp.sum(L_i * w_jk)

    _, R = jax.lax.scan(body_fun, None, jnp.arange(n))
    return R


def evaluate_at_xq(dofs, comp_info, comp_shapes, quad_shape, d):
    """Evaluate a k-form at quadrature points using tensor-product structure.

    Parameters
    ----------
    dofs : array, shape (n_total,)
        DOF vector (internal, already contracted with extraction matrices).
    comp_info : list of (output_dim, R, T, Z)
        For each component c, the output dimension and 1D basis arrays.
        R shape (s1_c, nq_r), T shape (s2_c, nq_t), Z shape (s3_c, nq_z).
    comp_shapes : list of tuples (s1_c, s2_c, s3_c)
        DOF grid shape per component.
    quad_shape : tuple (nq_t, nq_r, nq_z)
    d : int
        Number of output dimensions.

    Returns
    -------
    f_jk : array, shape (n_q, d)
    """
    f = jnp.zeros((d,) + quad_shape)

    offset = 0
    for c, (out_dim, R, T, Z) in enumerate(comp_info):
        s = comp_shapes[c]
        n_c = s[0] * s[1] * s[2]
        V = dofs[offset:offset + n_c].reshape(s)
        # V[i,j,k], R[i,a], T[j,b], Z[k,c] -> f[b,a,c]
        # quad_shape = (nq_t, nq_r, nq_z) -> output indices (b, a, c)
        val = jnp.einsum('ijk,ia,jb,kc->bac', V, R, T, Z)
        f = f.at[out_dim].add(val)
        offset += n_c

    return f.transpose(1, 2, 3, 0).reshape(-1, d)


def integrate_against(f_jk, comp_info, comp_shapes, quad_shape):
    """Integrate quad-point values against k-form basis using TP structure.

    Parameters
    ----------
    f_jk : array, shape (n_q, d)
        Values at quadrature points (already weighted).
    comp_info : list of (input_dim, R, T, Z)
        For each component c, the input dimension and 1D basis arrays.
    comp_shapes : list of tuples (s1_c, s2_c, s3_c)
    quad_shape : tuple (nq_t, nq_r, nq_z)

    Returns
    -------
    result : array, shape (n_total,)
    """
    d = f_jk.shape[1]
    # Reshape to (d, nq_t, nq_r, nq_z)
    f = f_jk.reshape(quad_shape + (d,)).transpose(3, 0, 1, 2)

    parts = []
    for c, (in_dim, R, T, Z) in enumerate(comp_info):
        # f[in_dim] has shape (nq_t, nq_r, nq_z)
        # R[i,a], T[j,b], Z[k,c], f[b,a,c] -> result[i,j,k]
        val = jnp.einsum('ia,jb,kc,bac->ijk', R, T, Z, f[in_dim])
        parts.append(val.ravel())

    return jnp.concatenate(parts)


# Device-specific parameter presets for the relaxation
DEVICE_PRESETS = {
    "ITER":  {"eps": 0.32, "kappa": 1.7, "delta": 0.33, "q_star": 1.57, "type": "tokamak", "n_zeta": 1, "p_zeta": 0},
    "NSTX":  {"eps": 0.78, "kappa": 2.0, "delta": 0.35, "q_star": 2.0, "type": "tokamak", "n_zeta": 1, "p_zeta": 0},
    "SPHERO": {"eps": 0.95, "kappa": 1.0, "delta": 0.2,  "q_star": 0.0, "type": "tokamak", "n_zeta": 1, "p_zeta": 0},
    "ROT_ELL": {"a": 0.1, "b": 0.025, "m_rot": 5, "q_star": 1.6, "type": "rotating_ellipse"},
    "HELIX": {"eps": 0.33, "h_helix": 0.20, "kappa": 1.7, "delta": 0.33, "m_helix": 3, "q_star": 2.0, "type": "helix"},
}

# Default configuration parameters for the relaxation
DEFAULT_CONFIG = {
    # Run parameters
    "run_name": "",
    # Type of boundary: "tokamak" or "helix" or "rotating_ellipse"
    "boundary_type": "rotating_ellipse",

    # Parameters describing the domain. Some of these parameters are ignored for certain domain shapes.
    "eps":      0.2,  # aspect ratio
    "kappa":    1.7,   # Elongation parameter
    "q_star":   1.57,   # toroidal field strength
    "delta": 0.0,   # triangularity
    "nfp":  3,     # poloidal mode number of helix (number of field periods)
    "h_helix":  0,   # radius of helix turns
    "R_0": 1.0,   # major radius of the domain

    # Discretization parameters for the finite element space
    "n_r": 8,       # Number of radial splines
    "n_theta": 8,   # Number of poloidal splines
    "n_zeta": 6,    # Number of toroidal splines
    "p_r": 3,       # Degree of radial splines
    "p_theta": 3,     # Degree of poloidal splines
    "p_zeta": 3,    # Degree of toroidal splines

    # Hyperparameters for the outer loop of the magnetic relaxation solver
    "maxit":                 5_000,   # max. Number of time steps
    "precond":               False,     # Use preconditioner
    "precond_compute_every": 1000,       # Recompute preconditioner every n iterations
    "gamma":                 0,  # Regularization, u = (-Δ)⁻ᵞ (J x B - grad p)
    "dt":                    1e-6,      # initial time step
    # time-steps are increased by this factor and decreased by its square
    "dt_factor":             1.01,
    # Convergence tolerance for |JxB - grad p| (or |JxB| if force_free)
    "force_tol":             1e-15,
    "eta":                   0.0,       # Resistivity
    # If True, solve for JxB = 0. If False, JxB = grad p
    "force_free":            False,

    # Solver hyperparameters for the inner loop of the magnetic relaxation solver
    "solver_maxit": 20,    # Maximum number of iterations before Picard solver gives up
    # If Picard solver converges in less than this number of iterations, increase time step
    "solver_critit": 4,
    "solver_tol": 1e-12,   # Tolerance for convergence
    "verbose": False,      # If False, prints only force every 'print_every'
    "print_every": 1000,    # Print every n iterations
    "save_every": 100,     # Save intermediate results every n iterations
    "save_B": False,       # Save intermediate B fields to file
    "save_B_every": 500,   # Save full B every n iterations

    # Hyperparameters pertaining to island seeding
    "pert_strength":       0.0,  # strength of perturbation
    "pert_pol_mode":          2,  # poloidal mode number of perturbation
    "pert_tor_mode":          1,  # toroidal mode number of perturbation
    "pert_radial_loc":      1/2,  # radial location of perturbation
    "pert_radial_width":    0.07,  # radial width of perturbation
    # apply perturbation after n steps (0 = to initial condition)
    "apply_pert_after":     2000,
}


# Default trace dictionary for the relaxation loop
default_trace_dict = {
    "iterations": [],
    "force_trace": [],
    "energy_trace": [],
    "helicity_trace": [],
    "divergence_trace": [],
    "picard_iterations": [],
    "picard_errors": [],
    "timesteps": [],
    "velocity_trace": [],
    "wall_time_trace": [],
    "B_fields": [],
    "p_fields": [],
    "start_time": None,
    "end_time": None,
}


def append_to_trace_dict(
        trace_dict: dict, i: int, f: float, E: float,
        H: float, dvg: float, v: float, p_i: int,
        e: float, dt: float, end_time: float,
        B: Optional[jnp.ndarray] = None) -> dict:
    """
    Append values to the trace dictionary.

    Args:
        trace_dict: Dictionary to append values to.
        i: Iteration number.
        f: Force norm.
        E: Energy.
        H: Helicity.
        dvg: Divergence norm.
        v: Velocity norm.
        p_i: Picard iterations.
        e: Picard error.
        dt: Time step.
        end_time: End time.
        B: Magnetic field.

    Returns:
        trace_dict: Dictionary with appended values.
    """
    trace_dict["iterations"].append(i)
    trace_dict["force_trace"].append(f)
    trace_dict["energy_trace"].append(E)
    trace_dict["helicity_trace"].append(H)
    trace_dict["divergence_trace"].append(dvg)
    trace_dict["velocity_trace"].append(v)
    trace_dict["picard_iterations"].append(p_i)
    trace_dict["picard_errors"].append(e)
    trace_dict["timesteps"].append(dt)
    trace_dict["wall_time_trace"].append(end_time - trace_dict["start_time"])
    if B is not None:
        trace_dict["B_fields"].append(B)
    return trace_dict


def save_trace_dict_to_hdf5(trace_dict: dict, diagnostics, filename: str, CONFIG: dict):
    """
    Save the trace dictionary to an HDF5 file.

    Args:
        trace_dict: Trace dictionary.
        diagnostics: MRXDiagnostics object.
        filename: Name of the file to save the trace dictionary to.
        CONFIG: Configuration dictionary.

    Returns:
        None
    """
    import h5py
    import numpy as np
    Seq = diagnostics.Seq
    print(filename)
    with h5py.File(filename + ".h5", "w") as f:
        f.create_dataset("iterations", data=jnp.array(
            trace_dict["iterations"]))
        f.create_dataset("force_trace", data=jnp.array(
            trace_dict["force_trace"]))
        f.create_dataset("B_final", data=trace_dict["B_final"])
        f.create_dataset("p_final", data=trace_dict["p_final"])
        f.create_dataset("energy_trace", data=jnp.array(
            trace_dict["energy_trace"]))
        f.create_dataset("helicity_trace", data=jnp.array(
            trace_dict["helicity_trace"]))
        f.create_dataset("divergence_trace", data=jnp.array(
            trace_dict["divergence_trace"]))
        f.create_dataset("velocity_trace", data=jnp.array(
            trace_dict["velocity_trace"]))
        f.create_dataset("picard_iterations", data=jnp.array(
            trace_dict["picard_iterations"]))
        f.create_dataset("picard_errors", data=jnp.array(
            trace_dict["picard_errors"]))
        f.create_dataset("timesteps", data=jnp.array(trace_dict["timesteps"]))
        f.create_dataset("harmonic_norm", data=jnp.array(
            [norm_2(diagnostics.harmonic_component(trace_dict["B_final"]), Seq)]))
        f.create_dataset("total_time", data=jnp.array(
            [trace_dict["end_time"] - trace_dict["start_time"]]))
        f.create_dataset("time_setup", data=jnp.array(
            [trace_dict["setup_done_time"] - trace_dict["start_time"]]))
        f.create_dataset("time_solve", data=jnp.array(
            [trace_dict["end_time"] - trace_dict["setup_done_time"]]))
        f.create_dataset("wall_time_trace", data=jnp.array(
            trace_dict["wall_time_trace"]))
        if CONFIG["save_B"]:
            f.create_dataset("B_fields", data=jnp.array(
                trace_dict["B_fields"]))
            f.create_dataset("p_fields", data=jnp.array(
                trace_dict["p_fields"]))
        # Store config variables in a group
        cfg_group = f.create_group("config")
        for key, val in CONFIG.items():
            # Skip callable objects (functions) and other non-serializable types
            if callable(val):
                continue
            # Skip numpy arrays with object dtype
            if isinstance(val, np.ndarray) and val.dtype == object:
                continue
            # Try to convert to numpy array and check for object dtype
            try:
                val_array = np.asarray(val)
                if val_array.dtype == object:
                    continue
            except (ValueError, TypeError):
                pass

            if isinstance(val, str):
                # Strings need special handling
                cfg_group.attrs[key] = np.bytes_(val)
            else:
                try:
                    cfg_group.attrs[key] = val
                except (TypeError, ValueError):
                    # Skip values that can't be serialized to HDF5
                    continue


def run_relaxation_loop(CONFIG, trace_dict, state, diagnostics):
    """
    Run the relaxation loop.

    Args:
        CONFIG: Configuration dictionary.
        trace_dict: Trace dictionary.
        state: State object.
        diagnostics: MRXDiagnostics object.
    """
    import time

    from mrx.relaxation import DescentMethod, MRXHessian, TimeStepper

    # Construct the time stepper and the Hessian
    Seq = diagnostics.Seq
    timestepper = TimeStepper(Seq,
                              gamma=CONFIG["gamma"],
                              descent_method=DescentMethod.NEWTON if CONFIG[
                                  "precond"] else DescentMethod.GRADIENT,
                              force_free=CONFIG["force_free"],
                              picard_tol=CONFIG["solver_tol"],
                              picard_k_restart=CONFIG["solver_maxit"])

    compute_hessian = jax.jit(MRXHessian(Seq).assemble)  # defaults to identity
    step = jax.jit(lambda state, key: timestepper.relaxation_step(state, key))
    B_hat = state.B_n
    get_energy = jax.jit(diagnostics.energy)
    get_helicity = jax.jit(diagnostics.helicity)
    get_divergence_B = jax.jit(diagnostics.divergence_norm)

    # Compile and record initial values
    dry_run = step(state, state.key)
    trace_dict = append_to_trace_dict(trace_dict, 0,
                                      dry_run.F_norm,
                                      get_energy(state.B_n),
                                      get_helicity(state.B_n),
                                      get_divergence_B(state.B_n),
                                      dry_run.v_norm,
                                      0,
                                      0,
                                      dry_run.dt,
                                      time.time(),
                                      B_hat if CONFIG["save_B"] else None)

    print(f"Initial force error: {trace_dict['force_trace'][-1]:.2e}")
    print(f"Initial energy: {trace_dict['energy_trace'][-1]:.2e}")
    print(f"Initial helicity: {trace_dict['helicity_trace'][-1]:.2e}")
    print(f"Initial ||div B||: {trace_dict['divergence_trace'][-1]:.2e}")

    setup_done_time = time.time()
    trace_dict["setup_done_time"] = setup_done_time
    print(
        f"Setup took {setup_done_time - trace_dict['start_time']:.2e} seconds.")

    print("Starting relaxation loop...")
    for i in range(1, CONFIG["maxit"] + 1):

        state = step(state, state.key)
        if (state.picard_residuum > CONFIG["solver_tol"]
                or ~jnp.isfinite(state.picard_residuum)):
            # half time step and try again
            state = timestepper.update_field(state, "dt", state.dt / 2)
            state = timestepper.update_field(state, "B_nplus1", state.B_n)
            continue
        # otherwise, we converged - proceed
        state = timestepper.update_field(state, "B_n", state.B_nplus1)

        if i == CONFIG["apply_pert_after"] and CONFIG["pert_strength"] > 0:
            print(f"Applying perturbation after {i} steps...")
            dB_hat = jnp.linalg.solve(Seq.M2, Seq.P2(CONFIG["dB_xyz"]))
            dB_hat = Seq.P_Leray @ dB_hat
            dB_hat /= norm_2(dB_hat, Seq)
            B_new = state.B_n + CONFIG["pert_strength"] * dB_hat
            state = timestepper.update_field(state, "B_n", B_new)

        if CONFIG["precond"] and (i % CONFIG["precond_compute_every"] == 0):
            state = timestepper.update_field(
                state, "hessian", compute_hessian(state.B_n))

        if state.picard_iterations < CONFIG["solver_critit"]:
            dt_new = state.dt * CONFIG["dt_factor"]
        else:
            dt_new = state.dt / (CONFIG["dt_factor"])**2
        state = timestepper.update_field(state, "dt", dt_new)

        if i % CONFIG["save_every"] == 0 or i == CONFIG["maxit"]:
            trace_dict = append_to_trace_dict(trace_dict, i,
                                              state.F_norm,
                                              get_energy(state.B_n),
                                              get_helicity(state.B_n),
                                              get_divergence_B(state.B_n),
                                              state.v_norm,
                                              state.picard_iterations,
                                              state.picard_residuum,
                                              state.dt,
                                              time.time(),
                                              state.B_n if CONFIG["save_B"] else None)

        if i % CONFIG["print_every"] == 0:
            print(
                f"Iteration {i}, u norm: {state.v_norm:.2e}, force norm: {state.F_norm:.2e}")
            if CONFIG["verbose"]:
                print(
                    f"   dt: {dt_new:.2e}, picard iters: {state.picard_iterations:.2e}, picard err: {state.picard_residuum:.2e}")
        if trace_dict["force_trace"][-1] < CONFIG["force_tol"]:
            print(
                f"Converged to force tolerance {CONFIG['force_tol']} after {i} steps.")
            break


def update_config(params: dict, CONFIG: dict):
    """
    Get the configuration from parameters specified on the command line.

    Args:
        params: Parameters dictionary.
        CONFIG: Configuration dictionary.

    Returns:
        CONFIG: Updated configuration dictionary.
    """
    # Step 1: If device specified, apply defaults
    device_name = params.get("device")
    if device_name:
        preset = DEVICE_PRESETS.get(device_name.upper())
        if preset:
            for k, v in preset.items():
                CONFIG[k] = v
        else:
            print(f"Unknown device '{device_name}' - ignoring.")

    # Step 2: Override with user-supplied parameters
    for k, v in params.items():
        if k in CONFIG:
            CONFIG[k] = v
        elif k != "device":
            print(f"Unknown parameter '{k}' - ignoring.")

    print("Configuration:")
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")

    return CONFIG


def interpolate_B(B_vals, eval_points, Seq, exclude_axis_tol=1e-3):
    """
    Interpolate B-field onto Seq.Lambda_2 basis.

    Parameters
    ----------
    B_vals : jnp.ndarray
        B-field values at evaluation points, shape (mρ mθ mζ, 3).
    eval_points : jnp.ndarray
        Evaluation points in logical coordinates, shape (mρ mθ mζ, 3).
    Seq : DeRhamSequence
        DeRham sequence to interpolate the B-field onto.
    exclude_axis_tol : float
        Tolerance for excluding points near the axis and exact boundary.

    Returns
    -------
    B_dof : jnp.ndarray
        B-field coefficients.
    residuals : jnp.ndarray
        Residuals of the interpolation.
    rank : int
        Rank of the interpolation.
    s : jnp.ndarray
        Singular values of the interpolation.
    """
    # valid interpolation points (avoid axis and exact boundary)
    valid_pts = (eval_points[:, 0] > exclude_axis_tol) & (
        eval_points[:, 0] < 1 - exclude_axis_tol
    )

    def Λ2_phys(i, x):
        """
        Evaluate the physical 2-form basis function Phi*Λ2[i] at x.

        Parameters
        ----------
        i : int
            Index of the basis function.
        x : jnp.ndarray
            Point to evaluate the basis function at.

        Returns
        -------
        jnp.ndarray
            Value of the basis function at x.
        """
        # Pullback of basis function
        DPhix = jax.jacfwd(Seq.F)(x)  # Jacobian of Phi at x
        J = jnp.linalg.det(DPhix)
        return DPhix @ Seq.Lambda_2[i](x) / J

    def body_fun(_, i):
        # Evaluate Λ2_phys(i, x) for all points (vectorized over x)
        return None, jax.lax.map(lambda x: Λ2_phys(i, x), eval_points[valid_pts], batch_size=mrx.MAP_BATCH_SIZE_INNER)

    _, M = jax.lax.scan(body_fun, None, Seq.Lambda_2.ns)
    M = jnp.einsum("il,ljk->ijk", Seq.E2, M)  # Λ2[i](x_hat_j)_k
    y = B_vals[valid_pts]  # B(x'_j)_k
    A = M.reshape(M.shape[0], -1).T
    b = y.ravel()
    # Solve least squares
    B_dof, residuals, _, _ = jnp.linalg.lstsq(A, b, rcond=None)
    return B_dof, residuals


def _bcsr_to_coo_indices(mat: jsparse.BCSR):
    """Expand BCSR indptr to COO-style (row, col) index array."""
    nse = mat.data.shape[0]
    lengths = mat.indptr[1:] - mat.indptr[:-1]
    rows = jnp.repeat(jnp.arange(mat.shape[0]), lengths,
                      total_repeat_length=nse)
    return jnp.stack([rows, mat.indices], axis=1)


def extract_diag_vector(mat) -> jnp.ndarray:
    """Extracts the main diagonal of a sparse matrix as a 1D array."""
    n = mat.shape[0]
    if isinstance(mat, jsparse.BCSR):
        indices = _bcsr_to_coo_indices(mat)
        rows, cols = indices[:, 0], indices[:, 1]
    else:
        rows = mat.indices[:, 0]
        cols = mat.indices[:, 1]
    is_diag = rows == cols
    diag_data = jnp.where(is_diag, mat.data, 0.0)
    return jnp.zeros(n, dtype=mat.dtype).at[rows].add(diag_data)


def square_sparse(mat) -> jsparse.BCOO:
    """Squares the non-zero elements of a sparse matrix. Always returns BCOO."""
    if isinstance(mat, jsparse.BCSR):
        indices = _bcsr_to_coo_indices(mat)
        return jsparse.BCOO((mat.data**2, indices), shape=mat.shape)
    return jsparse.BCOO((mat.data**2, mat.indices), shape=mat.shape)


# backward compat alias
square_bcoo = square_sparse

# %%

# def solve_singular_cg(A_matvec, b, mass_matvec=None, precond_matvec=lambda x: x, x0=None, vs=[], maxiter=1000, tol=1e-6):

#    if mass_matvec is None:
#        mass_matvec = lambda x: x

#     # --- 1. Your Exact Projection Logic ---
#     def inner_product(x, y):
#         return jnp.dot(x, mass_matvec(y))

#     def project_primal(x):
#         for v in vs:
#             x = x - inner_product(v, x) * v
#         return x

#     def project_dual(f):
#         for v in vs:
#             f = f - jnp.dot(v, f) * mass_matvec(v)
#         return f

#     # --- 2. Initial Setup ---
#     b_proj = project_dual(b)
#     if x0 is None:
#         x0 = jnp.zeros_like(b_proj)
#     else:
#         x0 = project_primal(x0)

#     # Initial residual (Dual)
#     Ax0 = project_dual(A_matvec(x0))
#     r0 = b_proj - Ax0
#     r0 = project_dual(r0)  # Clean the initial residual

#     # Initial preconditioned residual (Primal)
#     z0 = precond_matvec(r0)
#     z0 = project_primal(z0) # Clean the preconditioner output

#     p0 = z0

#     # State: (iteration, x, r, p, z, r_dot_z, residual_norm)
#     r_dot_z_0 = jnp.vdot(r0, z0)
#     init_state = (0, x0, r0, p0, z0, r_dot_z_0, jnp.linalg.norm(r0))

#     # --- 3. The CG Loop ---
#     def cond_fun(state):
#         i, _, _, _, _, _, r_norm = state
#         return (i < maxiter) & (r_norm > tol)

#     def body_fun(state):
#         i, x, r, p, z, r_dot_z, _ = state

#         # Matrix-vector multiply (output is Dual)
#         Ap = A_matvec(p)
#         Ap = project_dual(Ap) # Safe evaluation

#         # Step size
#         p_dot_Ap = jnp.vdot(p, Ap)
#         alpha = r_dot_z / p_dot_Ap

#         # Update solution and residual
#         x_next = x + alpha * p
#         r_next = r - alpha * Ap

#         r_next = project_dual(r_next)

#         # Apply and project preconditioner
#         z_next = precond_matvec(r_next)
#         z_next = project_primal(z_next) # PREVENTS PRECONDITIONER DRIFT

#         # Update search direction
#         r_dot_z_next = jnp.vdot(r_next, z_next)
#         beta = r_dot_z_next / r_dot_z
#         p_next = z_next + beta * p

#         # Keep p strictly in the Primal subspace
#         p_next = project_primal(p_next)

#         return (i + 1, x_next, r_next, p_next, z_next, r_dot_z_next, jnp.linalg.norm(r_next))

#     # --- 4. Execute ---
#     final_state = jax.lax.while_loop(cond_fun, body_fun, init_state)

#     iters, x_final, _, _, _, _, info_norm = final_state

#     # Final safety projection
#     return project_primal(x_final), {"iterations": iters, "residual_norm": info_norm}


def solve_singular_cg(A_matvec, b, mass_matvec=None, precond_matvec=lambda x: x, x0=None, vs=[], maxiter=None, tol=1e-6):
    """
    Solve the singular SPSD system for the minimum norm solution using CG.

    Args:
        A_matvec: Callable representing bilinear form (outputs Dual vectors).
        mass_matvec: Callable representing mass matrix.
        b: The right-hand side vector (Dual vector).
        x0: Optional initial guess (Primal vector).
        vs: List of mass-normalized zero eigenvectors (Primal vectors).
        maxiter: Maximum number of CG iterations.
        tol: CG tolerance.
    """
    if mass_matvec is None:
        def mass_matvec(x): return x

    def inner_product(x, y):
        return jnp.dot(x, mass_matvec(y))

    def project_primal(x):
        for v in vs:
            x = x - inner_product(v, x) * v
        return x

    def project_dual(f):
        for v in vs:
            f = f - jnp.dot(v, f) * mass_matvec(v)
        return f

    b_proj = project_dual(b)

    def A_matvec_safe(x):
        x = project_primal(x)
        # Apply the bilinear form (output is Dual)
        Ax = A_matvec(x)
        return project_dual(Ax)

    if x0 is None:
        x0 = jnp.zeros_like(b_proj)
    else:
        x0 = project_primal(x0)

    x, info = cg(A_matvec_safe, b_proj, x0=x0,
                 M=precond_matvec, tol=tol, maxiter=maxiter)
    return project_primal(x), info


def get_smallest_ev_pair(A_matvec, mass_matvec, x0, precond_matvec=lambda x: x, vs=[], shift=1e-9, maxiter=20, tol=1e-6):
    """
    Finds the generalized eigenvector using shifted inverse iteration. 
    """
    def inner_product(x, y):
        return jnp.dot(x, mass_matvec(y))

    def normalize(x):
        return x / jnp.sqrt(inner_product(x, x))

    def project_primal(x):
        # These are DoF vectors
        for v in vs:
            x = x - inner_product(v, x) * v
        return x

    def project_dual(f):
        # These are bilinear form outputs
        for v in vs:
            f = f - jnp.dot(v, f) * mass_matvec(v)
        return f

    def A_shifted(x):
        x = project_primal(x)
        # (outputs are Dual vectors)
        Ax = A_matvec(x) + shift * mass_matvec(x)
        return project_dual(Ax)

    x0 = normalize(project_primal(x0))

    def cond_fun(val):
        i, x, x_prev = val
        # Check both signs
        diff = jnp.minimum(jnp.linalg.norm(x - x_prev),
                           jnp.linalg.norm(x + x_prev))
        return jnp.logical_and(i < maxiter, diff > tol)

    def body_fun(val):
        i, x, _ = val
        rhs = mass_matvec(x)
        rhs = project_dual(rhs)
        y, _ = cg(A_shifted, rhs, x0=jnp.zeros_like(
            x), M=precond_matvec, tol=tol, maxiter=maxiter)
        y = project_primal(y)
        x_next = normalize(y)
        return (i + 1, x_next, x)

    init_val = (0, x0, jnp.zeros_like(x0))
    _, v, _ = jax.lax.while_loop(cond_fun, body_fun, init_val)

    # Rayleigh quotient
    lmbda = jnp.dot(v, A_matvec(v))

    return v, lmbda
