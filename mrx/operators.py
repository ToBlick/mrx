from __future__ import annotations

from typing import Optional, Sequence

import equinox as eqx
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import jax.scipy as jsp

from mrx.assembly import assemble_scalar_tp, assemble_vectorial_tp
from mrx.solvers import solve_saddle_point_minres, solve_singular_cg
from mrx.utils import diag_EAET, diag_EAET_matvec, diag_schur_complement


def _nullspace_vectors(operators, k: int, dirichlet: bool):
    """Return the stacked nullspace array for ``(k, dirichlet)``."""
    from mrx.nullspace import get_nullspace
    return get_nullspace(operators, k, dirichlet)


def _saddle_nullspaces(seq, operators, k: int, dirichlet: bool):
    """Return upper/lower nullspace arrays for the saddle-point system."""
    from mrx.nullspace import get_saddle_point_nullspaces
    return get_saddle_point_nullspaces(seq, operators, k, dirichlet)


def _shifted_harmonic_coarse_vector(
        seq, operators: SequenceOperators, k: int, dirichlet: bool):
    """Return the stored M_k-normalised coarse vector for shifted solves."""
    n_dof = getattr(seq, f"n{k}_dbc" if dirichlet else f"n{k}")
    try:
        vs = _nullspace_vectors(operators, k, dirichlet)
    except ValueError:
        return jnp.zeros(n_dof)
    if vs.shape[0] == 0:
        return jnp.zeros(n_dof)
    stored = vs[0]
    stored_norm = seq.l2_norm(stored, k, dirichlet=dirichlet)
    return stored / jnp.where(stored_norm > 0, stored_norm, 1.0)


def _wrap_shifted_harmonic_coarse_correction(
        seq, operators: SequenceOperators, base_precond, eps: float,
        k: int, dirichlet: bool):
    """Add an exact ``1/eps`` coarse correction on the stored harmonic mode."""
    z = _shifted_harmonic_coarse_vector(seq, operators, k, dirichlet)
    mz = apply_mass_matrix(seq, operators, z, k, dirichlet=dirichlet)

    def precond(x):
        alpha = z @ x
        x_perp = x - alpha * mz
        y_perp = base_precond(x_perp)
        beta = z @ apply_mass_matrix(
            seq, operators, y_perp, k, dirichlet=dirichlet)
        return y_perp - beta * z + (alpha / eps) * z

    return precond


class SequenceOperators(eqx.Module):
    """Dynamic operator bundle for a de Rham sequence.

    Stores geometry-dependent operator data explicitly so it can be carried
    through JAX transforms while the sequence object remains a static topology
    shell.
    """

    m0_sp: Optional[jsparse.BCSR] = None
    m1_sp: Optional[jsparse.BCSR] = None
    m2_sp: Optional[jsparse.BCSR] = None
    m3_sp: Optional[jsparse.BCSR] = None
    m0_sp_diaginv: Optional[object] = None
    m1_sp_diaginv: Optional[object] = None
    m2_sp_diaginv: Optional[object] = None
    m3_sp_diaginv: Optional[object] = None
    m0_sp_diaginv_dbc: Optional[object] = None
    m1_sp_diaginv_dbc: Optional[object] = None
    m2_sp_diaginv_dbc: Optional[object] = None
    m3_sp_diaginv_dbc: Optional[object] = None
    # Reference-domain 1-D mass-matrix inverses for the regular ('p') and
    # derivative ('d') splines along each axis. Used by the Kronecker
    # ("fast diagonalisation") mass preconditioner; geometry-independent.
    # ``None`` means not yet assembled (call
    # :func:`assemble_kron_mass_preconditioner`).
    m1d_inv_p_r: Optional[jnp.ndarray] = None
    m1d_inv_p_t: Optional[jnp.ndarray] = None
    m1d_inv_p_z: Optional[jnp.ndarray] = None
    m1d_inv_d_r: Optional[jnp.ndarray] = None
    m1d_inv_d_t: Optional[jnp.ndarray] = None
    m1d_inv_d_z: Optional[jnp.ndarray] = None
    # Per-component geometric scale factors for the Kronecker mass
    # preconditioner.  Each array has one entry per vector component (length
    # 1 for k=0, 3, length 3 for k=1, 2) holding the quadrature-average of
    # the diagonal mass coefficient for that component, so that
    # ``~M_k ≈ diag(α_i) · (M_r ⊗ M_θ ⊗ M_ζ)``. Geometry-dependent;
    # reassembled whenever the mass operator is reassembled.
    m0_kron_scale: Optional[jnp.ndarray] = None
    m1_kron_scale: Optional[jnp.ndarray] = None
    m2_kron_scale: Optional[jnp.ndarray] = None
    m3_kron_scale: Optional[jnp.ndarray] = None
    # Reference-domain 1-D fast-diagonalisation eigendecompositions of the
    # generalised eigenproblem ``K_a v = λ M_a v`` for the regular ('p')
    # and derivative ('d') spline spaces along each axis.  ``fd_V_*_a``
    # columns are the ``M_a``-orthonormal eigenvectors and ``fd_lam_*_a``
    # the corresponding eigenvalues.  The apply step uses ``V`` directly:
    # ``K_a^{-1} = V_a Λ^{-1} V_a^T`` (since ``V_a^T M_a V_a = I`` implies
    # ``V_a^{-1} = V_a^T M_a`` and the ``M_a^{-1}`` factor in ``K_a^{-1}``
    # cancels).  The ``'p'`` variants drive the ``k = 0`` Hodge
    # preconditioner; the ``'d'`` variants drive ``k = 3`` via the Hodge
    # duality ``L_3 ≅ L_0`` on the derivative spline space.  Geometry-
    # independent, populated by :func:`assemble_fd_hodge_preconditioner`.
    fd_V_p_r: Optional[jnp.ndarray] = None
    fd_V_p_t: Optional[jnp.ndarray] = None
    fd_V_p_z: Optional[jnp.ndarray] = None
    fd_lam_p_r: Optional[jnp.ndarray] = None
    fd_lam_p_t: Optional[jnp.ndarray] = None
    fd_lam_p_z: Optional[jnp.ndarray] = None
    fd_V_d_r: Optional[jnp.ndarray] = None
    fd_V_d_t: Optional[jnp.ndarray] = None
    fd_V_d_z: Optional[jnp.ndarray] = None
    fd_lam_d_r: Optional[jnp.ndarray] = None
    fd_lam_d_t: Optional[jnp.ndarray] = None
    fd_lam_d_z: Optional[jnp.ndarray] = None
    # Per-direction scaling for the FD Hodge preconditioner. The entry along
    # axis ``i`` is the quadrature-average of ``J·g^{ii}``, which is the
    # diagonal of the stiffness integrand on the mapped domain and captures
    # the leading anisotropy for both ``L_0`` (regular splines) and ``L_3``
    # (derivative splines, by Hodge duality).  Geometry-dependent;
    # reassembled with :func:`update_hodge_operator`.
    dd0_fd_scale_K: Optional[jnp.ndarray] = None
    dd3_fd_scale_K: Optional[jnp.ndarray] = None
    d0_sp: Optional[jsparse.BCSR] = None
    d0_sp_T: Optional[jsparse.BCSR] = None
    d1_sp: Optional[jsparse.BCSR] = None
    d1_sp_T: Optional[jsparse.BCSR] = None
    d2_sp: Optional[jsparse.BCSR] = None
    d2_sp_T: Optional[jsparse.BCSR] = None
    # Topological exterior-derivative incidence matrices on the full
    # pre-extraction DoF grid. Entries are in {-1, 0, +1}; they encode the
    # discrete de Rham complex structure and are geometry-independent. The
    # strong derivatives ``apply_strong_{grad,curl,div}`` multiply by these
    # directly (no mass solve).
    g0_sp: Optional[jsparse.BCSR] = None
    g0_sp_T: Optional[jsparse.BCSR] = None
    g1_sp: Optional[jsparse.BCSR] = None
    g1_sp_T: Optional[jsparse.BCSR] = None
    g2_sp: Optional[jsparse.BCSR] = None
    g2_sp_T: Optional[jsparse.BCSR] = None
    grad_grad_sp: Optional[jsparse.BCSR] = None
    curl_curl_sp: Optional[jsparse.BCSR] = None
    div_div_sp: Optional[jsparse.BCSR] = None
    dd0_sp_diaginv: Optional[object] = None
    dd1_sp_diaginv: Optional[object] = None
    dd2_sp_diaginv: Optional[object] = None
    dd3_sp_diaginv: Optional[object] = None
    dd0_sp_diaginv_dbc: Optional[object] = None
    dd1_sp_diaginv_dbc: Optional[object] = None
    dd2_sp_diaginv_dbc: Optional[object] = None
    dd3_sp_diaginv_dbc: Optional[object] = None
    p21_sp: Optional[jsparse.BCSR] = None
    p12_sp: Optional[jsparse.BCSR] = None
    p03_sp: Optional[jsparse.BCSR] = None
    p30_sp: Optional[jsparse.BCSR] = None

    # Harmonic nullspaces of the Hodge Laplacians. Each field, when set, holds
    # a stacked array of shape ``(n_vectors, n_k)`` with one nullspace basis
    # vector per row. Shapes are topology-determined (from the Betti numbers);
    # the DoFs are dynamic and may be overwritten when the geometry changes.
    null_0: Optional[jnp.ndarray] = None
    null_1: Optional[jnp.ndarray] = None
    null_2: Optional[jnp.ndarray] = None
    null_3: Optional[jnp.ndarray] = None
    null_0_dbc: Optional[jnp.ndarray] = None
    null_1_dbc: Optional[jnp.ndarray] = None
    null_2_dbc: Optional[jnp.ndarray] = None
    null_3_dbc: Optional[jnp.ndarray] = None

    def todense(self, seq, operator: str, k, dirichlet: bool = True,
                transpose: bool = False):
        """Return a dense matrix for one assembled operator block."""
        match operator:
            case "mass":
                return dense_mass_matrix(seq, self, k, dirichlet=dirichlet)
            case "derivative":
                return dense_derivative_matrix(
                    seq, self, k,
                    dirichlet_in=dirichlet,
                    dirichlet_out=dirichlet,
                    transpose=transpose,
                )
            case "stiffness":
                return dense_stiffness_matrix(seq, self, k, dirichlet=dirichlet)
            case "hodge_laplacian":
                return dense_hodge_laplacian(seq, self, k, dirichlet=dirichlet)
            case "projection":
                if not isinstance(k, tuple) or len(k) != 2:
                    raise ValueError(
                        "Projection dense conversion expects k=(k_in, k_out)")
                return dense_projection_matrix(
                    seq, self, k[0], k[1],
                    dirichlet_in=dirichlet,
                    dirichlet_out=dirichlet,
                )
        raise ValueError(
            "operator must be one of 'mass', 'derivative', 'stiffness', 'hodge_laplacian', or 'projection'"
        )


def _kron_geometric_scales(seq, geometry, k):
    """Quadrature-average of the diagonal mass coefficient per component.

    Used by the Kronecker mass preconditioner so that the approximation
    ``~M_k^{(i)} ≈ α_i · (M_r ⊗ M_θ ⊗ M_ζ)`` captures the leading
    geometric anisotropy that is invisible to the reference Kronecker
    factors.
    """
    w = seq.quad.w
    w_sum = jnp.sum(w)
    J = geometry.jacobian_j
    match k:
        case 0:
            return jnp.array([jnp.sum(J * w) / w_sum])
        case 1:
            g_inv = geometry.metric_inv_jkl  # (n_quad, 3, 3)
            return jnp.array([
                jnp.sum(J * g_inv[:, i, i] * w) / w_sum for i in range(3)
            ])
        case 2:
            g = geometry.metric_jkl  # (n_quad, 3, 3)
            return jnp.array([
                jnp.sum(g[:, i, i] / J * w) / w_sum for i in range(3)
            ])
        case 3:
            return jnp.array([jnp.sum(w / J) / w_sum])
    raise ValueError("k must be 0, 1, 2 or 3")


def _assemble_mass_block(seq, geometry, k):
    quad_shape = (seq.quad.ny, seq.quad.nx, seq.quad.nz)

    match k:
        case 0:
            W_flat = geometry.jacobian_j * seq.quad.w
            sp = assemble_scalar_tp(
                seq.basis_r_jk, seq.basis_t_jk, seq.basis_z_jk,
                seq.basis_r_jk, seq.basis_t_jk, seq.basis_z_jk,
                W_flat, quad_shape, seq.basis_0.shape[0],
                seq.basis_0.pr, seq.basis_0.pt, seq.basis_0.pz)
            sp = jsparse.BCSR.from_bcoo(sp)
            diaginv = 1.0 / diag_EAET(seq.e0, sp, seq.e0_T)
            diaginv_dbc = 1.0 / diag_EAET(seq.e0_dbc, sp, seq.e0_dbc_T)
        case 1:
            W_3x3 = geometry.metric_inv_jkl * \
                (geometry.jacobian_j * seq.quad.w)[:, None, None]
            terms = [
                [(0, seq.d_basis_r_jk, seq.basis_t_jk, seq.basis_z_jk, +1)],
                [(1, seq.basis_r_jk, seq.d_basis_t_jk, seq.basis_z_jk, +1)],
                [(2, seq.basis_r_jk, seq.basis_t_jk, seq.d_basis_z_jk, +1)],
            ]
            sp = assemble_vectorial_tp(
                terms, terms, W_3x3, quad_shape,
                list(seq.basis_1.shape), seq.basis_1.pr)
            sp = jsparse.BCSR.from_bcoo(sp)
            diaginv = 1.0 / diag_EAET(seq.e1, sp, seq.e1_T)
            diaginv_dbc = 1.0 / diag_EAET(seq.e1_dbc, sp, seq.e1_dbc_T)
        case 2:
            W_3x3 = geometry.metric_jkl * \
                (1 / geometry.jacobian_j * seq.quad.w)[:, None, None]
            terms = [
                [(0, seq.basis_r_jk, seq.d_basis_t_jk, seq.d_basis_z_jk, +1)],
                [(1, seq.d_basis_r_jk, seq.basis_t_jk, seq.d_basis_z_jk, +1)],
                [(2, seq.d_basis_r_jk, seq.d_basis_t_jk, seq.basis_z_jk, +1)],
            ]
            sp = assemble_vectorial_tp(
                terms, terms, W_3x3, quad_shape,
                list(seq.basis_2.shape), seq.basis_2.pr)
            sp = jsparse.BCSR.from_bcoo(sp)
            diaginv = 1.0 / diag_EAET(seq.e2, sp, seq.e2_T)
            diaginv_dbc = 1.0 / diag_EAET(seq.e2_dbc, sp, seq.e2_dbc_T)
        case 3:
            W_flat = (1 / geometry.jacobian_j) * seq.quad.w
            sp = assemble_scalar_tp(
                seq.d_basis_r_jk, seq.d_basis_t_jk, seq.d_basis_z_jk,
                seq.d_basis_r_jk, seq.d_basis_t_jk, seq.d_basis_z_jk,
                W_flat, quad_shape, seq.basis_3.shape[0],
                seq.basis_3.pr, seq.basis_3.pt, seq.basis_3.pz)
            sp = jsparse.BCSR.from_bcoo(sp)
            diaginv = 1.0 / diag_EAET(seq.e3, sp, seq.e3_T)
            diaginv_dbc = 1.0 / diag_EAET(seq.e3_dbc, sp, seq.e3_dbc_T)
        case _:
            raise ValueError("k must be 0, 1, 2 or 3")

    return sp, diaginv, diaginv_dbc


def update_mass_operator(seq, geometry, operators: Optional[SequenceOperators], k: int):
    """Return an operator bundle with the k-th mass operator updated."""
    sp, diaginv, diaginv_dbc = _assemble_mass_block(seq, geometry, k)
    kron_scale = _kron_geometric_scales(seq, geometry, k)
    if operators is None:
        operators = SequenceOperators()

    match k:
        case 0:
            return eqx.tree_at(
                lambda ops: (ops.m0_sp, ops.m0_sp_diaginv,
                             ops.m0_sp_diaginv_dbc, ops.m0_kron_scale),
                operators,
                (sp, diaginv, diaginv_dbc, kron_scale),
                is_leaf=lambda x: x is None,
            )
        case 1:
            return eqx.tree_at(
                lambda ops: (ops.m1_sp, ops.m1_sp_diaginv,
                             ops.m1_sp_diaginv_dbc, ops.m1_kron_scale),
                operators,
                (sp, diaginv, diaginv_dbc, kron_scale),
                is_leaf=lambda x: x is None,
            )
        case 2:
            return eqx.tree_at(
                lambda ops: (ops.m2_sp, ops.m2_sp_diaginv,
                             ops.m2_sp_diaginv_dbc, ops.m2_kron_scale),
                operators,
                (sp, diaginv, diaginv_dbc, kron_scale),
                is_leaf=lambda x: x is None,
            )
        case 3:
            return eqx.tree_at(
                lambda ops: (ops.m3_sp, ops.m3_sp_diaginv,
                             ops.m3_sp_diaginv_dbc, ops.m3_kron_scale),
                operators,
                (sp, diaginv, diaginv_dbc, kron_scale),
                is_leaf=lambda x: x is None,
            )
    raise ValueError("k must be 0, 1, 2 or 3")


def assemble_mass_operators(seq, geometry, operators: Optional[SequenceOperators] = None,
                            ks: Sequence[int] = (0, 1, 2, 3)):
    """Assemble mass operators for the requested form degrees."""
    for k in ks:
        operators = update_mass_operator(seq, geometry, operators, k)
    return operators


# ---------------------------------------------------------------------------
# Kronecker ("fast diagonalisation") mass preconditioner.
#
# On the reference cube the ``k``-form mass matrix is a sum of geometry-
# weighted Kronecker products of 1-D mass matrices.  Dropping the geometric
# weighting yields a single Kronecker product per component
#
#     ~M_k^{comp} = M_r^{(α)} ⊗ M_θ^{(β)} ⊗ M_ζ^{(γ)},   α,β,γ ∈ {p, d}
#
# where ``p`` is the regular spline mass and ``d`` is the derivative-spline
# mass along the relevant axis.  Its inverse can be applied as three small
# dense matvecs per axis (cost O(N^4) flops via einsum), and is spectrally
# equivalent to ``M_k^{-1}`` on quasi-uniform meshes — typically dropping
# CG iteration counts by an order of magnitude versus the plain Jacobi
# preconditioner.  All factors are geometry-independent and assembled once.
# ---------------------------------------------------------------------------


def _assemble_1d_mass_inverse(B, w):
    """Build ``(B diag(w) B^T)^{-1}`` for a 1-D basis evaluated on quad points."""
    M = (B * w[None, :]) @ B.T
    M_inv = jnp.linalg.inv(M)
    # Symmetrise to wash out floating-point asymmetry.
    return 0.5 * (M_inv + M_inv.T)


def assemble_kron_mass_preconditioner(
        seq, operators: Optional[SequenceOperators] = None):
    """Assemble the 1-D reference mass-matrix inverses on ``operators``.

    The resulting preconditioner is geometry-independent; call this once
    after ``seq.evaluate_1d()``.  Subsequent calls to
    :func:`apply_mass_matrix_preconditioner` will then default to the
    Kronecker variant.
    """
    if operators is None:
        operators = SequenceOperators()
    inv_p_r = _assemble_1d_mass_inverse(seq.basis_r_jk, seq.quad.w_x)
    inv_p_t = _assemble_1d_mass_inverse(seq.basis_t_jk, seq.quad.w_y)
    inv_p_z = _assemble_1d_mass_inverse(seq.basis_z_jk, seq.quad.w_z)
    inv_d_r = _assemble_1d_mass_inverse(seq.d_basis_r_jk, seq.quad.w_x)
    inv_d_t = _assemble_1d_mass_inverse(seq.d_basis_t_jk, seq.quad.w_y)
    inv_d_z = _assemble_1d_mass_inverse(seq.d_basis_z_jk, seq.quad.w_z)
    return eqx.tree_at(
        lambda ops: (ops.m1d_inv_p_r, ops.m1d_inv_p_t, ops.m1d_inv_p_z,
                     ops.m1d_inv_d_r, ops.m1d_inv_d_t, ops.m1d_inv_d_z),
        operators,
        (inv_p_r, inv_p_t, inv_p_z, inv_d_r, inv_d_t, inv_d_z),
        is_leaf=lambda x: x is None,
    )


def _kron_component_specs(seq, k: int):
    """Return ``[(component_shape, (kind_r, kind_t, kind_z)), ...]`` for form ``k``."""
    nr_p = seq.basis_r_jk.shape[0]
    nt_p = seq.basis_t_jk.shape[0]
    nz_p = seq.basis_z_jk.shape[0]
    nr_d = seq.d_basis_r_jk.shape[0]
    nt_d = seq.d_basis_t_jk.shape[0]
    nz_d = seq.d_basis_z_jk.shape[0]
    if k == 0:
        return [((nr_p, nt_p, nz_p), ('p', 'p', 'p'))]
    if k == 1:
        return [
            ((nr_d, nt_p, nz_p), ('d', 'p', 'p')),
            ((nr_p, nt_d, nz_p), ('p', 'd', 'p')),
            ((nr_p, nt_p, nz_d), ('p', 'p', 'd')),
        ]
    if k == 2:
        return [
            ((nr_p, nt_d, nz_d), ('p', 'd', 'd')),
            ((nr_d, nt_p, nz_d), ('d', 'p', 'd')),
            ((nr_d, nt_d, nz_p), ('d', 'd', 'p')),
        ]
    if k == 3:
        return [((nr_d, nt_d, nz_d), ('d', 'd', 'd'))]
    raise ValueError("k must be 0, 1, 2 or 3")


def _kron_inv_table(operators: SequenceOperators):
    return {
        ('p', 'r'): operators.m1d_inv_p_r,
        ('p', 't'): operators.m1d_inv_p_t,
        ('p', 'z'): operators.m1d_inv_p_z,
        ('d', 'r'): operators.m1d_inv_d_r,
        ('d', 't'): operators.m1d_inv_d_t,
        ('d', 'z'): operators.m1d_inv_d_z,
    }


def _kron_available(seq, operators: SequenceOperators, k: int) -> bool:
    table = _kron_inv_table(operators)
    needed = set()
    for _, kinds in _kron_component_specs(seq, k):
        for axis, kind in zip('rtz', kinds):
            needed.add((kind, axis))
    return all(table[key] is not None for key in needed)


def _kron_apply_3d(Mr_inv, Mt_inv, Mz_inv, x):
    """Apply ``(Mr_inv ⊗ Mt_inv ⊗ Mz_inv)`` to a 3-tensor ``x``."""
    x = jnp.einsum('ij,jkl->ikl', Mr_inv, x)
    x = jnp.einsum('ij,kjl->kil', Mt_inv, x)
    x = jnp.einsum('ij,klj->kli', Mz_inv, x)
    return x


def _kron_scale_for_k(operators: SequenceOperators, k: int):
    match k:
        case 0:
            return operators.m0_kron_scale
        case 1:
            return operators.m1_kron_scale
        case 2:
            return operators.m2_kron_scale
        case 3:
            return operators.m3_kron_scale
    raise ValueError("k must be 0, 1, 2 or 3")


def _kron_apply_full(seq, operators: SequenceOperators, v_full, k: int):
    """Apply the block-diagonal Kronecker mass-inverse on a full-grid vector."""
    table = _kron_inv_table(operators)
    specs = _kron_component_specs(seq, k)
    scales = _kron_scale_for_k(operators, k)
    parts = []
    offset = 0
    for i, (shape, kinds) in enumerate(specs):
        size = shape[0] * shape[1] * shape[2]
        x = v_full[offset:offset + size].reshape(shape)
        Mr_inv = table[(kinds[0], 'r')]
        Mt_inv = table[(kinds[1], 't')]
        Mz_inv = table[(kinds[2], 'z')]
        y = _kron_apply_3d(Mr_inv, Mt_inv, Mz_inv, x)
        if scales is not None:
            y = y / scales[i]
        parts.append(y.reshape(-1))
        offset += size
    return jnp.concatenate(parts) if len(parts) > 1 else parts[0]


def apply_mass_kron_preconditioner(
        seq, operators: SequenceOperators, v, k: int, dirichlet: bool = True):
    """``E (M_r^{-1} ⊗ M_θ^{-1} ⊗ M_ζ^{-1}) E^T v`` (block-diagonal over components).

    SPD by construction; sandwiches the reference Kronecker inverse with the
    extraction operators so the same code path handles polar fusion and
    Dirichlet boundary clamping.  If any 1-D inverse is unavailable,
    :func:`apply_mass_matrix_preconditioner` falls back to Jacobi instead
    of calling this function.
    """
    if dirichlet:
        e = getattr(seq, f'e{k}_dbc')
        e_T = getattr(seq, f'e{k}_dbc_T')
    else:
        e = getattr(seq, f'e{k}')
        e_T = getattr(seq, f'e{k}_T')
    v_full = e_T @ v
    y_full = _kron_apply_full(seq, operators, v_full, k)
    return e @ y_full


# ---------------------------------------------------------------------------
# Fast-diagonalisation Hodge-Laplacian preconditioner.
#
# For a 0-form on the reference cube the discrete Hodge Laplacian
# ``L_0 = K_0`` is a Kronecker SUM
#
#     L_0 ≈  K_r ⊗ M_t ⊗ M_z + M_r ⊗ K_t ⊗ M_z + M_r ⊗ M_t ⊗ K_z ,
#
# with 1-D mass ``M_a = ∫ B^p_a (B^p_a)^T`` and 1-D stiffness
# ``K_a = ∫ (∂B^p_a)(∂B^p_a)^T = G_a^T M^d_a G_a`` (incidence relation).
# Reducing the per-axis generalised eigenproblem ``K_a v = λ M_a v`` to a
# standard one via Cholesky gives an ``M``-orthonormal eigenbasis and the
# inverse can be applied as three small dense matmuls per axis combined with
# a divide by ``Σ_i α_i λ_i`` on the 3-tensor.  ``α_i = ⟨J·g^{ii}⟩_quad``
# captures the leading metric anisotropy on the mapped domain.
# ---------------------------------------------------------------------------


def _dense_incidence_1d(n0: int, typ: str) -> jnp.ndarray:
    """Return the dense 1-D incidence matrix ``G_a`` for axis basis type."""
    rows, cols, vals, n_out, n_in = _incidence_1d_coo(n0, typ)
    G = jnp.zeros((n_out, n_in))
    G = G.at[rows, cols].add(vals)
    return G


def _assemble_1d_mass(B: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    """Symmetrised 1-D mass ``B diag(w) B^T``."""
    M = (B * w[None, :]) @ B.T
    return 0.5 * (M + M.T)


def _assemble_1d_stiffness_d(
        M_d: jnp.ndarray, M_p: jnp.ndarray, G: jnp.ndarray) -> jnp.ndarray:
    """1-D stiffness on the derivative spline space.

    ``K^{(d)} = M_d · G · M_p^{-1} · G^T · M_d``, the Hodge-dual of the
    regular-space stiffness ``G^T M_d G`` obtained by exchanging the roles
    of primal and derivative spaces via the weighted adjoint of ``G``.  The
    resulting generalised eigenproblem ``K^{(d)} v = λ M_d v`` has the same
    nonzero spectrum as ``K^{(p)} v = λ M_p v``.
    """
    # Solve M_p X = G^T for X, then K = M_d G X M_d, using Cholesky.
    L = jnp.linalg.cholesky(M_p)
    GT = G.T
    Y = jsp.linalg.solve_triangular(L, GT, lower=True)
    X = jsp.linalg.solve_triangular(L.T, Y, lower=False)
    # X = M_p^{-1} G^T,  shape (n_p, n_d); G X has shape (n_d, n_d).
    K = M_d @ (G @ X) @ M_d
    return 0.5 * (K + K.T)


def _assemble_1d_fd_eigendecomp(M: jnp.ndarray, K: jnp.ndarray):
    """Reduce ``K v = λ M v`` to a standard eigenproblem via Cholesky.

    Returns ``(V, lam)`` where ``V`` columns are ``M``-orthonormal
    eigenvectors and ``lam`` the eigenvalues.
    """
    L = jnp.linalg.cholesky(M)
    # K_tilde = L^{-1} K L^{-T}
    Y = jsp.linalg.solve_triangular(L, K, lower=True)
    K_tilde = jsp.linalg.solve_triangular(L, Y.T, lower=True).T
    K_tilde = 0.5 * (K_tilde + K_tilde.T)
    lam, W = jnp.linalg.eigh(K_tilde)
    # V = L^{-T} W satisfies V^T M V = I and K V = M V diag(lam).
    V = jsp.linalg.solve_triangular(L.T, W, lower=False)
    return V, lam


def assemble_fd_hodge_preconditioner(
        seq, operators: Optional[SequenceOperators] = None):
    """Precompute the 1-D FD eigendecompositions used by the Hodge
    preconditioner for ``k = 0`` and ``k = 3``.

    Geometry-independent; call once after ``seq.evaluate_1d()``.  After this,
    :func:`apply_hodge_laplacian_preconditioner` with ``kind='hx'``
    becomes available for ``k = 0`` (and is used internally by the
    ``k = 3`` auxiliary-space path).
    """
    if operators is None:
        operators = SequenceOperators()
    types = seq.basis_0.types
    G_r = _dense_incidence_1d(seq.basis_0.nr, types[0])
    G_t = _dense_incidence_1d(seq.basis_0.nt, types[1])
    G_z = _dense_incidence_1d(seq.basis_0.nz, types[2])
    M_p_r = _assemble_1d_mass(seq.basis_r_jk, seq.quad.w_x)
    M_p_t = _assemble_1d_mass(seq.basis_t_jk, seq.quad.w_y)
    M_p_z = _assemble_1d_mass(seq.basis_z_jk, seq.quad.w_z)
    M_d_r = _assemble_1d_mass(seq.d_basis_r_jk, seq.quad.w_x)
    M_d_t = _assemble_1d_mass(seq.d_basis_t_jk, seq.quad.w_y)
    M_d_z = _assemble_1d_mass(seq.d_basis_z_jk, seq.quad.w_z)
    # Regular-space stiffness: K^{(p)} = G^T M_d G.
    K_p_r = G_r.T @ (M_d_r @ G_r)
    K_p_r = 0.5 * (K_p_r + K_p_r.T)
    K_p_t = G_t.T @ (M_d_t @ G_t)
    K_p_t = 0.5 * (K_p_t + K_p_t.T)
    K_p_z = G_z.T @ (M_d_z @ G_z)
    K_p_z = 0.5 * (K_p_z + K_p_z.T)
    V_p_r, lam_p_r = _assemble_1d_fd_eigendecomp(M_p_r, K_p_r)
    V_p_t, lam_p_t = _assemble_1d_fd_eigendecomp(M_p_t, K_p_t)
    V_p_z, lam_p_z = _assemble_1d_fd_eigendecomp(M_p_z, K_p_z)
    # Derivative-space stiffness: K^{(d)} = M_d G M_p^{-1} G^T M_d (Hodge
    # dual of K^{(p)}).
    K_d_r = _assemble_1d_stiffness_d(M_d_r, M_p_r, G_r)
    K_d_t = _assemble_1d_stiffness_d(M_d_t, M_p_t, G_t)
    K_d_z = _assemble_1d_stiffness_d(M_d_z, M_p_z, G_z)
    V_d_r, lam_d_r = _assemble_1d_fd_eigendecomp(M_d_r, K_d_r)
    V_d_t, lam_d_t = _assemble_1d_fd_eigendecomp(M_d_t, K_d_t)
    V_d_z, lam_d_z = _assemble_1d_fd_eigendecomp(M_d_z, K_d_z)
    return eqx.tree_at(
        lambda ops: (
            ops.fd_V_p_r, ops.fd_V_p_t, ops.fd_V_p_z,
            ops.fd_lam_p_r, ops.fd_lam_p_t, ops.fd_lam_p_z,
            ops.fd_V_d_r, ops.fd_V_d_t, ops.fd_V_d_z,
            ops.fd_lam_d_r, ops.fd_lam_d_t, ops.fd_lam_d_z,
        ),
        operators,
        (V_p_r, V_p_t, V_p_z, lam_p_r, lam_p_t, lam_p_z,
         V_d_r, V_d_t, V_d_z, lam_d_r, lam_d_t, lam_d_z),
        is_leaf=lambda x: x is None,
    )


def _fd_hodge_scales_K(seq, geometry, k: int) -> jnp.ndarray:
    """Per-direction quadrature-averaged stiffness coefficients for form ``k``.

    The diagonal of the stiffness integrand on the mapped domain is
    ``J · g^{ii}`` along axis ``i`` for both ``L_0`` (regular splines) and
    ``L_3`` (derivative splines, by Hodge duality ``L_3 ≅ L_0`` on the
    physical manifold).  Only ``k ∈ {0, 3}`` are implemented.
    """
    if k not in (0, 3):
        raise ValueError("FD Hodge scale currently implemented for k ∈ {0, 3}")
    w = seq.quad.w
    w_sum = jnp.sum(w)
    J = geometry.jacobian_j
    g_inv = geometry.metric_inv_jkl  # (n_quad, 3, 3)
    return jnp.array([
        jnp.sum(J * g_inv[:, i, i] * w) / w_sum for i in range(3)
    ])


def _fd_hodge_available(operators: SequenceOperators, k: int) -> bool:
    """True iff the FD Hodge preconditioner can be applied for form ``k``."""
    if k == 0:
        needed = (
            operators.fd_V_p_r, operators.fd_V_p_t, operators.fd_V_p_z,
            operators.fd_lam_p_r, operators.fd_lam_p_t, operators.fd_lam_p_z,
            operators.dd0_fd_scale_K,
        )
        return all(x is not None for x in needed)
    if k == 3:
        needed = (
            operators.fd_V_d_r, operators.fd_V_d_t, operators.fd_V_d_z,
            operators.fd_lam_d_r, operators.fd_lam_d_t, operators.fd_lam_d_z,
            operators.dd3_fd_scale_K,
        )
        return all(x is not None for x in needed)
    return False


def _fd_apply_3d(V_r, V_t, V_z, lam_r, lam_t, lam_z, alpha, x, eps: float = 0.0):
    """Apply ``(L + eps * M)^{-1}`` to a 3-tensor ``x`` via fast diagonalisation.

    For ``L = Σ_i α_i (… ⊗ K_i ⊗ …)`` with reference 1-D masses ``M_a``
    and eigendecomposition ``K_a v = λ M_a v``, ``V_a^T M_a V_a = I``,
    the inverse of ``L + eps M`` is
    ``(V_r⊗V_t⊗V_z) (D + eps I)^{-1} (V_r⊗V_t⊗V_z)^T``
    with ``D = Σ_i α_i (… ⊗ Λ_i ⊗ …)``.  For ``eps == 0`` this is the
    Moore--Penrose pseudo-inverse (null directions are projected out); for
    ``eps > 0`` the shift lifts the kernel and no masking is needed.
    """
    # Forward transform: y = V^T x (in all three axes).
    y = jnp.einsum('ji,jkl->ikl', V_r, x)
    y = jnp.einsum('ji,kjl->kil', V_t, y)
    y = jnp.einsum('ji,klj->kli', V_z, y)
    # Diagonal solve in the eigenbasis.
    denom = (alpha[0] * lam_r[:, None, None]
             + alpha[1] * lam_t[None, :, None]
             + alpha[2] * lam_z[None, None, :]) + eps
    if eps == 0:
        # The pure-constant 0-form is in the null space; threshold relative
        # to the largest entry so we don't amplify it into a huge spurious
        # negative direction.
        denom_max = jnp.max(jnp.abs(denom))
        null_mask = jnp.abs(denom) < 1e-10 * denom_max
        safe = jnp.where(null_mask, 1.0, denom)
        y = jnp.where(null_mask, 0.0, y / safe)
    else:
        y = y / denom
    # Back transform: x_out = V y (in all three axes).
    y = jnp.einsum('ij,jkl->ikl', V_r, y)
    y = jnp.einsum('ij,kjl->kil', V_t, y)
    y = jnp.einsum('ij,klj->kli', V_z, y)
    return y


def _fd_apply_full(seq, operators: SequenceOperators, v_full, k: int,
                   eps: float = 0.0):
    if k == 0:
        nr = seq.basis_r_jk.shape[0]
        nt = seq.basis_t_jk.shape[0]
        nz = seq.basis_z_jk.shape[0]
        x = v_full.reshape((nr, nt, nz))
        y = _fd_apply_3d(
            operators.fd_V_p_r, operators.fd_V_p_t, operators.fd_V_p_z,
            operators.fd_lam_p_r, operators.fd_lam_p_t, operators.fd_lam_p_z,
            operators.dd0_fd_scale_K, x, eps=eps,
        )
        return y.reshape(-1)
    if k == 3:
        nr = seq.d_basis_r_jk.shape[0]
        nt = seq.d_basis_t_jk.shape[0]
        nz = seq.d_basis_z_jk.shape[0]
        x = v_full.reshape((nr, nt, nz))
        y = _fd_apply_3d(
            operators.fd_V_d_r, operators.fd_V_d_t, operators.fd_V_d_z,
            operators.fd_lam_d_r, operators.fd_lam_d_t, operators.fd_lam_d_z,
            operators.dd3_fd_scale_K, x, eps=eps,
        )
        return y.reshape(-1)
    raise ValueError("FD Hodge apply implemented for k ∈ {0, 3} only")


def apply_hodge_kron_preconditioner(
        seq, operators: SequenceOperators, v, k: int, dirichlet: bool = True,
        eps: float = 0.0):
    """``E · (L_ref + eps * M_ref)^{-1} · E^T v`` via fast diagonalisation (k ∈ {0, 3}).

    ``eps = 0`` gives the usual pseudo-inverse Hodge Laplacian preconditioner.
    ``eps > 0`` gives a linear, SPD preconditioner for the shifted system
    ``L_k + eps M_k`` with no kernel handling required.
    """
    if not _fd_hodge_available(operators, k):
        raise ValueError(
            "FD Hodge preconditioner not available for k="
            f"{k}; assemble it via assemble_fd_hodge_preconditioner")
    if dirichlet:
        e = getattr(seq, f'e{k}_dbc')
        e_T = getattr(seq, f'e{k}_dbc_T')
    else:
        e = getattr(seq, f'e{k}')
        e_T = getattr(seq, f'e{k}_T')
    v_full = e_T @ v
    y_full = _fd_apply_full(seq, operators, v_full, k, eps=eps)
    return e @ y_full


def _mass_components(operators: SequenceOperators, k: int):
    match k:
        case 0:
            return operators.m0_sp, operators.m0_sp_diaginv, operators.m0_sp_diaginv_dbc
        case 1:
            return operators.m1_sp, operators.m1_sp_diaginv, operators.m1_sp_diaginv_dbc
        case 2:
            return operators.m2_sp, operators.m2_sp_diaginv, operators.m2_sp_diaginv_dbc
        case 3:
            return operators.m3_sp, operators.m3_sp_diaginv, operators.m3_sp_diaginv_dbc
    raise ValueError("k must be 0, 1, 2 or 3")


def _mass_diaginv(operators: SequenceOperators, k: int, dirichlet: bool):
    _, diaginv, diaginv_dbc = _mass_components(operators, k)
    selected = diaginv_dbc if dirichlet else diaginv
    if selected is None:
        raise ValueError(f"Mass preconditioner k={k} is not assembled")
    return selected


def _hodge_components(operators: SequenceOperators, k: int):
    match k:
        case 0:
            return operators.grad_grad_sp, operators.dd0_sp_diaginv, operators.dd0_sp_diaginv_dbc
        case 1:
            return operators.curl_curl_sp, operators.dd1_sp_diaginv, operators.dd1_sp_diaginv_dbc
        case 2:
            return operators.div_div_sp, operators.dd2_sp_diaginv, operators.dd2_sp_diaginv_dbc
        case 3:
            return None, operators.dd3_sp_diaginv, operators.dd3_sp_diaginv_dbc
    raise ValueError("k must be 0, 1, 2 or 3")


def _hodge_diaginv(operators: SequenceOperators, k: int, dirichlet: bool):
    _, diaginv, diaginv_dbc = _hodge_components(operators, k)
    selected = diaginv_dbc if dirichlet else diaginv
    if selected is None:
        raise ValueError(f"Hodge preconditioner k={k} is not assembled")
    return selected


def _derivative_components(operators: SequenceOperators, k: int):
    match k:
        case 0:
            return operators.d0_sp, operators.d0_sp_T
        case 1:
            return operators.d1_sp, operators.d1_sp_T
        case 2:
            return operators.d2_sp, operators.d2_sp_T
    raise ValueError("k must be 0, 1 or 2")


def _projection_components(operators: SequenceOperators, k_in: int, k_out: int):
    match (k_in, k_out):
        case (2, 1):
            return operators.p21_sp
        case (1, 2):
            return operators.p12_sp
        case (0, 3):
            return operators.p03_sp
        case (3, 0):
            return operators.p30_sp
    raise ValueError(
        "Only (k_in, k_out) = (1, 2), (2, 1), (0, 3), or (3, 0) supported"
    )


def _assemble_derivative_block(seq, operators: SequenceOperators, k: int):
    """No-op: the weak derivative ``D_k = M_{k+1} G_k`` is applied lazily.

    Historically we materialised ``D_k`` as a BCSR product of the mass
    ``M_{k+1}`` and the topological incidence ``G_k``. The ``BCOO @ BCOO``
    intermediates allocated by JAX are quadratic in row/column occupancy
    and were the main contributor to peak memory during
    :func:`assemble_all_operators`. Since every downstream use is either a
    matvec or a sandwiched diagonal extraction, we now apply ``D_k`` as the
    composition of two cheap BCSR matvecs ``M_{k+1} @ (G_k @ v)``.

    This function just validates that ``G_k`` and ``M_{k+1}`` are available
    and returns ``(None, None)``; the stored ``d{k}_sp`` / ``d{k}_sp_T``
    fields remain ``None``.
    """
    g_sp, _ = _incidence_components(operators, k)
    m_sp, _, _ = _mass_components(operators, k + 1)
    if g_sp is None:
        raise ValueError(
            f"Incidence operator G{k} is required to apply D{k}")
    if m_sp is None:
        raise ValueError(
            f"Mass operator M{k + 1} is required to apply D{k}")
    return None, None


def update_derivative_operator(seq, geometry, operators: Optional[SequenceOperators], k: int):
    """Return an operator bundle with the k-th weak derivative available.

    ``geometry`` is accepted for backward compatibility but no longer used:
    the weak derivative ``D_k = M_{k+1} G_k`` is not materialised. We only
    ensure that the topological incidence ``G_k`` is assembled so that
    :func:`apply_derivative_matrix` and :func:`apply_stiffness` can apply
    ``D_k`` as a composition of BCSR matvecs.

    Callers must still assemble ``M_{k+1}`` before any downstream use.
    """
    del geometry  # unused
    if operators is None:
        operators = SequenceOperators()
    if _incidence_components(operators, k)[0] is None:
        operators = update_incidence_operator(seq, operators, k)
    # Validates ``G_k`` and ``M_{k+1}`` are present; returns ``(None, None)``.
    _assemble_derivative_block(seq, operators, k)
    if k in (0, 1, 2):
        return operators
    raise ValueError("k must be 0, 1 or 2")


def assemble_derivative_operators(seq, geometry, operators: Optional[SequenceOperators] = None,
                                  ks: Sequence[int] = (0, 1, 2)):
    """Assemble weak derivative operators for the requested form degrees."""
    for k in ks:
        operators = update_derivative_operator(seq, geometry, operators, k)
    return operators


# ---------------------------------------------------------------------------
# Topological incidence matrices (geometry-independent strong derivatives)
# ---------------------------------------------------------------------------
#
# On a FEEC B-spline de Rham complex the exterior derivative at the DoF level
# is a topological incidence matrix with entries in {-1, 0, +1}. The 1-D
# building block maps 0-form DoFs (nodes) to 1-form DoFs (edges) via
#
#     (G c)_j = c_{j+1} - c_j           (periodic: indices mod n)
#
# so the 3-D operators are Kronecker sums/products of these with identities.
# Because the incidence is geometry-independent, it does not need to be
# re-assembled when the spline map changes.

def _incidence_1d_coo(n0: int, typ: str):
    """Return (rows, cols, vals, n_out, n_in) for the 1-D incidence matrix."""
    if typ == 'clamped':
        n_out = n0 - 1
        j = jnp.arange(n_out)
        rows = jnp.concatenate([j, j])
        cols = jnp.concatenate([j, j + 1])
        vals = jnp.concatenate([-jnp.ones(n_out), jnp.ones(n_out)])
        return rows, cols, vals, n_out, n0
    if typ == 'periodic':
        n_out = n0
        j = jnp.arange(n_out)
        rows = jnp.concatenate([j, j])
        cols = jnp.concatenate([j, (j + 1) % n0])
        vals = jnp.concatenate([-jnp.ones(n_out), jnp.ones(n_out)])
        return rows, cols, vals, n_out, n0
    if typ == 'constant':
        # Derivative of a constant field is zero; DerivativeSpline has n_d = n0.
        rows = jnp.zeros((0,), dtype=jnp.int32)
        cols = jnp.zeros((0,), dtype=jnp.int32)
        vals = jnp.zeros((0,))
        return rows, cols, vals, n0, n0
    raise ValueError(f"Unknown basis type {typ!r}")


def _identity_coo(n: int):
    """Return (rows, cols, vals) for a 1-D identity of size n."""
    j = jnp.arange(n)
    return j, j, jnp.ones(n)


def _kron3_block(
    f_r, f_t, f_z,
    row_shape, col_shape,
):
    """Build one rank-1 Kronecker block ``f_r ⊗ f_t ⊗ f_z`` in BCOO.

    ``f_*`` is a triple ``(rows, cols, vals)`` of COO data for each 1-D
    factor. ``row_shape`` and ``col_shape`` give the 3-D DoF shapes of the
    factor output/input, used to ravel the multi-indices into flat indices.
    Returns a ``jsparse.BCOO`` of shape ``(prod(row_shape), prod(col_shape))``.
    """
    rr, rc, rv = f_r
    tr, tc, tv = f_t
    zr, zc, zv = f_z

    # Cartesian product of the three 1-D nonzero sets.
    Rr, Tr, Zr = jnp.meshgrid(rr, tr, zr, indexing='ij')
    Rc, Tc, Zc = jnp.meshgrid(rc, tc, zc, indexing='ij')
    Rv, Tv, Zv = jnp.meshgrid(rv, tv, zv, indexing='ij')

    row_flat = jnp.ravel_multi_index(
        (Rr.ravel(), Tr.ravel(), Zr.ravel()),
        row_shape, mode='clip')
    col_flat = jnp.ravel_multi_index(
        (Rc.ravel(), Tc.ravel(), Zc.ravel()),
        col_shape, mode='clip')
    vals = (Rv * Tv * Zv).ravel()

    n_rows = int(row_shape[0] * row_shape[1] * row_shape[2])
    n_cols = int(col_shape[0] * col_shape[1] * col_shape[2])
    indices = jnp.stack([row_flat, col_flat], axis=-1)
    return jsparse.BCOO((vals, indices), shape=(n_rows, n_cols))


def _bcoo_hstack(blocks, n_rows: int):
    """Horizontally concatenate BCOO blocks that share the same row count."""
    datas = []
    indices = []
    offset = 0
    for b in blocks:
        datas.append(b.data)
        idx = b.indices.at[:, 1].add(offset)
        indices.append(idx)
        offset += b.shape[1]
    data = jnp.concatenate(datas)
    idx = jnp.concatenate(indices, axis=0)
    return jsparse.BCOO((data, idx), shape=(n_rows, offset))


def _bcoo_vstack(blocks, n_cols: int):
    """Vertically concatenate BCOO blocks that share the same column count."""
    datas = []
    indices = []
    offset = 0
    for b in blocks:
        datas.append(b.data)
        idx = b.indices.at[:, 0].add(offset)
        indices.append(idx)
        offset += b.shape[0]
    data = jnp.concatenate(datas)
    idx = jnp.concatenate(indices, axis=0)
    return jsparse.BCOO((data, idx), shape=(offset, n_cols))


def _assemble_incidence_block(seq, k: int):
    """Build the topological derivative Dk on the full pre-extraction DoF grid.

    The 3-D incidence operators decompose into rank-1 Kronecker blocks:
    for a derivative in axis ``d``, the block is ``I ⊗ ... ⊗ G_d ⊗ ... ⊗ I``
    where the non-``d`` identity factors have sizes equal to the *input*
    component's shape in those axes (which must match the output).
    """
    types = seq.basis_0.types
    G_1d = {
        0: _incidence_1d_coo(seq.basis_0.nr, types[0])[:3],
        1: _incidence_1d_coo(seq.basis_0.nt, types[1])[:3],
        2: _incidence_1d_coo(seq.basis_0.nz, types[2])[:3],
    }

    def dblock(axis, in_shape, out_shape):
        """One-directional derivative block ``∂_axis`` from ``in_shape`` → ``out_shape``.

        The two non-differentiated axes must have matching sizes in/out; the
        differentiated axis must go from the 0-form size to the derivative
        size (or stay the same for periodic/constant).
        """
        factors = [None, None, None]
        for a in range(3):
            if a == axis:
                factors[a] = G_1d[a]
            else:
                assert in_shape[a] == out_shape[a], (
                    f"axis {a} size mismatch: {in_shape[a]} vs {out_shape[a]}")
                factors[a] = _identity_coo(in_shape[a])
        return _kron3_block(factors[0], factors[1], factors[2], out_shape, in_shape)

    def neg(block):
        return jsparse.BCOO((-block.data, block.indices), shape=block.shape)

    s0 = seq.basis_0.shape[0]            # (nr, nt, nz)
    s3 = seq.basis_3.shape[0]            # (dr, dt, dz)
    s1_r, s1_t, s1_z = seq.basis_1.shape
    s2_r, s2_t, s2_z = seq.basis_2.shape

    match k:
        case 0:
            # D0 v = (∂_r v, ∂_t v, ∂_z v).
            b_r = dblock(0, s0, s1_r)
            b_t = dblock(1, s0, s1_t)
            b_z = dblock(2, s0, s1_z)
            n_cols = int(s0[0] * s0[1] * s0[2])
            sp = _bcoo_vstack([b_r, b_t, b_z], n_cols)
        case 1:
            # Curl: (v0, v1, v2) ↦
            #   (∂_t v2 - ∂_z v1,  ∂_z v0 - ∂_r v2,  ∂_r v1 - ∂_t v0).
            # Row 0 (→ s2_r): [0, -∂_z v1, +∂_t v2]
            zero_00 = _empty_bcoo(s2_r, s1_r)
            b_01 = neg(dblock(2, s1_t, s2_r))
            b_02 = dblock(1, s1_z, s2_r)
            row0 = _bcoo_hstack(
                [zero_00, b_01, b_02], int(s2_r[0] * s2_r[1] * s2_r[2]))

            # Row 1 (→ s2_t): [+∂_z v0, 0, -∂_r v2]
            b_10 = dblock(2, s1_r, s2_t)
            zero_11 = _empty_bcoo(s2_t, s1_t)
            b_12 = neg(dblock(0, s1_z, s2_t))
            row1 = _bcoo_hstack(
                [b_10, zero_11, b_12], int(s2_t[0] * s2_t[1] * s2_t[2]))

            # Row 2 (→ s2_z): [-∂_t v0, +∂_r v1, 0]
            b_20 = neg(dblock(1, s1_r, s2_z))
            b_21 = dblock(0, s1_t, s2_z)
            zero_22 = _empty_bcoo(s2_z, s1_z)
            row2 = _bcoo_hstack(
                [b_20, b_21, zero_22], int(s2_z[0] * s2_z[1] * s2_z[2]))

            n_cols_1form = int(
                s1_r[0] * s1_r[1] * s1_r[2]
                + s1_t[0] * s1_t[1] * s1_t[2]
                + s1_z[0] * s1_z[1] * s1_z[2])
            sp = _bcoo_vstack([row0, row1, row2], n_cols_1form)
        case 2:
            # D2 (v0, v1, v2) = ∂_r v0 + ∂_t v1 + ∂_z v2.
            b_r = dblock(0, s2_r, s3)
            b_t = dblock(1, s2_t, s3)
            b_z = dblock(2, s2_z, s3)
            n_rows = int(s3[0] * s3[1] * s3[2])
            sp = _bcoo_hstack([b_r, b_t, b_z], n_rows)
        case _:
            raise ValueError("k must be 0, 1 or 2")

    sp = sp.sum_duplicates()
    sp_T = jsparse.BCSR.from_bcoo(sp.T)
    sp = jsparse.BCSR.from_bcoo(sp)
    return sp, sp_T


def _empty_bcoo(row_shape, col_shape):
    """Return a structurally-zero BCOO block of the given 3-D shapes."""
    n_rows = int(row_shape[0] * row_shape[1] * row_shape[2])
    n_cols = int(col_shape[0] * col_shape[1] * col_shape[2])
    data = jnp.zeros((0,))
    indices = jnp.zeros((0, 2), dtype=jnp.int32)
    return jsparse.BCOO((data, indices), shape=(n_rows, n_cols))


def update_incidence_operator(seq, operators: Optional[SequenceOperators], k: int):
    """Return an operator bundle with the k-th topological incidence updated."""
    sp, sp_T = _assemble_incidence_block(seq, k)
    if operators is None:
        operators = SequenceOperators()

    match k:
        case 0:
            return eqx.tree_at(
                lambda ops: (ops.g0_sp, ops.g0_sp_T),
                operators,
                (sp, sp_T),
                is_leaf=lambda x: x is None,
            )
        case 1:
            return eqx.tree_at(
                lambda ops: (ops.g1_sp, ops.g1_sp_T),
                operators,
                (sp, sp_T),
                is_leaf=lambda x: x is None,
            )
        case 2:
            return eqx.tree_at(
                lambda ops: (ops.g2_sp, ops.g2_sp_T),
                operators,
                (sp, sp_T),
                is_leaf=lambda x: x is None,
            )
    raise ValueError("k must be 0, 1 or 2")


def assemble_incidence_operators(seq, operators: Optional[SequenceOperators] = None,
                                 ks: Sequence[int] = (0, 1, 2)):
    """Assemble topological incidence operators (geometry-independent)."""
    for k in ks:
        if k not in (0, 1, 2):
            continue
        operators = update_incidence_operator(seq, operators, k)
    return operators


def _incidence_components(operators: SequenceOperators, k: int):
    match k:
        case 0:
            return operators.g0_sp, operators.g0_sp_T
        case 1:
            return operators.g1_sp, operators.g1_sp_T
        case 2:
            return operators.g2_sp, operators.g2_sp_T
    raise ValueError("k must be 0, 1 or 2")


def apply_incidence_matrix(seq, operators: SequenceOperators, v, k: int,
                           dirichlet_in: bool = True,
                           dirichlet_out: bool = True,
                           transpose: bool = False):
    """Apply the topological exterior-derivative incidence matrix Gk.

    ``Gk`` maps full k-form DoFs to full (k+1)-form DoFs with entries in
    ``{-1, 0, +1}``; this routine sandwiches it with the extraction operators
    so it acts on extracted DoF spaces (periodic and/or Dirichlet-reduced).
    """
    sp, sp_T = _incidence_components(operators, k)
    if sp is None or sp_T is None:
        raise ValueError(f"Incidence operator k={k} is not assembled")

    match k:
        case 0:
            e_in = seq.e0_dbc if dirichlet_in else seq.e0
            e_in_T = seq.e0_dbc_T if dirichlet_in else seq.e0_T
            e_out = seq.e1_dbc if dirichlet_out else seq.e1
            e_out_T = seq.e1_dbc_T if dirichlet_out else seq.e1_T
        case 1:
            e_in = seq.e1_dbc if dirichlet_in else seq.e1
            e_in_T = seq.e1_dbc_T if dirichlet_in else seq.e1_T
            e_out = seq.e2_dbc if dirichlet_out else seq.e2
            e_out_T = seq.e2_dbc_T if dirichlet_out else seq.e2_T
        case 2:
            e_in = seq.e2_dbc if dirichlet_in else seq.e2
            e_in_T = seq.e2_dbc_T if dirichlet_in else seq.e2_T
            e_out = seq.e3_dbc if dirichlet_out else seq.e3
            e_out_T = seq.e3_dbc_T if dirichlet_out else seq.e3_T
        case _:
            raise ValueError("k must be 0, 1 or 2")

    if transpose:
        return e_in @ (sp_T @ (e_out_T @ v))
    return e_out @ (sp @ (e_in_T @ v))


def _assemble_projection_block(seq, k_in: int, k_out: int):
    quad_shape = (seq.quad.ny, seq.quad.nx, seq.quad.nz)
    dR = seq.d_basis_r_jk
    dT = seq.d_basis_t_jk
    dZ = seq.d_basis_z_jk
    R = seq.basis_r_jk
    T = seq.basis_t_jk
    Z = seq.basis_z_jk

    match (k_in, k_out):
        case (2, 1):
            W_3x3 = seq.quad.w[:, None, None] * jnp.eye(3)
            row_terms = [
                [(0, dR, T, Z, +1)],
                [(1, R, dT, Z, +1)],
                [(2, R, T, dZ, +1)],
            ]
            col_terms = [
                [(0, R, dT, dZ, +1)],
                [(1, dR, T, dZ, +1)],
                [(2, dR, dT, Z, +1)],
            ]
            sp = assemble_vectorial_tp(
                row_terms, col_terms, W_3x3, quad_shape,
                list(seq.basis_1.shape), seq.basis_1.pr,
                col_comp_shapes=list(seq.basis_2.shape))
            return jsparse.BCSR.from_bcoo(sp)
        case (1, 2):
            W_3x3 = seq.quad.w[:, None, None] * jnp.eye(3)
            row_terms = [
                [(0, dR, T, Z, +1)],
                [(1, R, dT, Z, +1)],
                [(2, R, T, dZ, +1)],
            ]
            col_terms = [
                [(0, R, dT, dZ, +1)],
                [(1, dR, T, dZ, +1)],
                [(2, dR, dT, Z, +1)],
            ]
            sp = assemble_vectorial_tp(
                row_terms, col_terms, W_3x3, quad_shape,
                list(seq.basis_1.shape), seq.basis_1.pr,
                col_comp_shapes=list(seq.basis_2.shape))
            return jsparse.BCSR.from_bcoo(sp.T)
        case (0, 3):
            W_1x1 = seq.quad.w.reshape(-1, 1, 1)
            row_terms = [
                [(0, R, T, Z, +1)],
            ]
            col_terms = [
                [(0, dR, dT, dZ, +1)],
            ]
            sp = assemble_vectorial_tp(
                row_terms, col_terms, W_1x1, quad_shape,
                list(seq.basis_0.shape), seq.basis_0.pr,
                col_comp_shapes=list(seq.basis_3.shape))
            return jsparse.BCSR.from_bcoo(sp)
        case (3, 0):
            W_1x1 = seq.quad.w.reshape(-1, 1, 1)
            row_terms = [
                [(0, R, T, Z, +1)],
            ]
            col_terms = [
                [(0, dR, dT, dZ, +1)],
            ]
            sp = assemble_vectorial_tp(
                row_terms, col_terms, W_1x1, quad_shape,
                list(seq.basis_0.shape), seq.basis_0.pr,
                col_comp_shapes=list(seq.basis_3.shape))
            return jsparse.BCSR.from_bcoo(sp.T)
    raise ValueError(
        "Only (k_in, k_out) = (1, 2), (2, 1), (0, 3), or (3, 0) supported"
    )


def update_projection_operator(seq, operators: Optional[SequenceOperators],
                               k_in: int, k_out: int):
    """Return an operator bundle with the requested projection updated."""
    sp = _assemble_projection_block(seq, k_in, k_out)
    if operators is None:
        operators = SequenceOperators()

    match (k_in, k_out):
        case (2, 1):
            return eqx.tree_at(
                lambda ops: ops.p21_sp,
                operators,
                sp,
                is_leaf=lambda x: x is None,
            )
        case (1, 2):
            return eqx.tree_at(
                lambda ops: ops.p12_sp,
                operators,
                sp,
                is_leaf=lambda x: x is None,
            )
        case (0, 3):
            return eqx.tree_at(
                lambda ops: ops.p03_sp,
                operators,
                sp,
                is_leaf=lambda x: x is None,
            )
        case (3, 0):
            return eqx.tree_at(
                lambda ops: ops.p30_sp,
                operators,
                sp,
                is_leaf=lambda x: x is None,
            )
    raise ValueError(
        "Only (k_in, k_out) = (1, 2), (2, 1), (0, 3), or (3, 0) supported"
    )


def assemble_projection_operators(seq, operators: Optional[SequenceOperators] = None,
                                  pairs: Sequence[tuple[int, int]] = ((2, 1), (1, 2), (0, 3), (3, 0))):
    """Assemble projection operators for the requested degree pairs."""
    for k_in, k_out in pairs:
        operators = update_projection_operator(seq, operators, k_in, k_out)
    return operators


def _assemble_hodge_block(seq, geometry, operators: SequenceOperators, k):
    # Stiffness matrices satisfy
    #
    #     K_k = G_k^T M_{k+1} G_k,
    #
    # which follows directly from ``D_k = M_{k+1} G_k``.  Materialising the
    # sparse product blows up peak memory through large ``BCOO @ BCOO``
    # intermediates, so we apply ``K_k`` as a composition of BCSR matvecs
    # instead and extract the Jacobi diagonal ``diag(E K_k E^T)`` via
    # :func:`diag_EAET_matvec`.  No full stiffness matrix is stored; the
    # ``sp`` field in the returned tuple is always ``None``.
    del geometry  # unused

    def _stiffness_matvec(kk: int):
        g_sp, g_sp_T = _incidence_components(operators, kk)
        m_sp, _, _ = _mass_components(operators, kk + 1)
        if g_sp is None or g_sp_T is None:
            raise ValueError(
                f"Incidence operator G{kk} is required to apply K{kk}")
        if m_sp is None:
            raise ValueError(
                f"Mass operator M{kk + 1} is required to apply K{kk}")
        return lambda v: g_sp_T @ (m_sp @ (g_sp @ v))

    # ``D_{kk-1}^T v = G_{kk-1}^T (M_{kk} v)``; the Schur-complement helper
    # needs this as a callable, so compose it from the stored incidence and
    # mass matrices without materialising ``D``.
    def _derivative_T_matvec(kk: int):
        _, g_prev_T = _incidence_components(operators, kk - 1)
        m_curr, _, _ = _mass_components(operators, kk)
        if g_prev_T is None:
            raise ValueError(
                f"Incidence operator G{kk - 1} is required to apply D{kk - 1}^T")
        if m_curr is None:
            raise ValueError(
                f"Mass operator M{kk} is required to apply D{kk - 1}^T")
        return lambda v: g_prev_T @ (m_curr @ v)

    sp = None
    match k:
        case 0:
            K_apply = _stiffness_matvec(0)
            diaginv = 1.0 / diag_EAET_matvec(
                seq.e0, lambda v: K_apply(v), seq.n0, seq.e0_T)
            diaginv_dbc = 1.0 / diag_EAET_matvec(
                seq.e0_dbc, lambda v: K_apply(v), seq.n0_dbc, seq.e0_dbc_T)
        case 1:
            if operators.m0_sp_diaginv is None or operators.m0_sp_diaginv_dbc is None:
                raise ValueError(
                    "Assemble mass operator k=0 before Hodge operator k=1")
            K_apply = _stiffness_matvec(1)
            DT_apply = _derivative_T_matvec(1)
            d_stiff = diag_EAET_matvec(
                seq.e1, lambda v: K_apply(v), seq.n1, seq.e1_T)
            d_schur = diag_schur_complement(
                lambda v: seq.e0 @ DT_apply(seq.e1_T @ v),
                operators.m0_sp_diaginv, seq.n1)
            diaginv = 1.0 / (d_stiff + d_schur)
            d_stiff_dbc = diag_EAET_matvec(
                seq.e1_dbc, lambda v: K_apply(v), seq.n1_dbc, seq.e1_dbc_T)
            d_schur_dbc = diag_schur_complement(
                lambda v: seq.e0_dbc @ DT_apply(seq.e1_dbc_T @ v),
                operators.m0_sp_diaginv_dbc, seq.n1_dbc)
            diaginv_dbc = 1.0 / (d_stiff_dbc + d_schur_dbc)
        case 2:
            if operators.m1_sp_diaginv is None or operators.m1_sp_diaginv_dbc is None:
                raise ValueError(
                    "Assemble mass operator k=1 before Hodge operator k=2")
            K_apply = _stiffness_matvec(2)
            DT_apply = _derivative_T_matvec(2)
            d_stiff = diag_EAET_matvec(
                seq.e2, lambda v: K_apply(v), seq.n2, seq.e2_T)
            d_schur = diag_schur_complement(
                lambda v: seq.e1 @ DT_apply(seq.e2_T @ v),
                operators.m1_sp_diaginv, seq.n2)
            diaginv = 1.0 / (d_stiff + d_schur)
            d_stiff_dbc = diag_EAET_matvec(
                seq.e2_dbc, lambda v: K_apply(v), seq.n2_dbc, seq.e2_dbc_T)
            d_schur_dbc = diag_schur_complement(
                lambda v: seq.e1_dbc @ DT_apply(seq.e2_dbc_T @ v),
                operators.m1_sp_diaginv_dbc, seq.n2_dbc)
            diaginv_dbc = 1.0 / (d_stiff_dbc + d_schur_dbc)
        case 3:
            if operators.m2_sp_diaginv is None or operators.m2_sp_diaginv_dbc is None:
                raise ValueError(
                    "Assemble mass operator k=2 before Hodge operator k=3")
            DT_apply = _derivative_T_matvec(3)
            diaginv = 1.0 / diag_schur_complement(
                lambda v: seq.e2 @ DT_apply(seq.e3_T @ v),
                operators.m2_sp_diaginv, seq.n3)
            diaginv_dbc = 1.0 / diag_schur_complement(
                lambda v: seq.e2_dbc @ DT_apply(seq.e3_dbc_T @ v),
                operators.m2_sp_diaginv_dbc, seq.n3_dbc)
        case _:
            raise ValueError("k must be 0, 1, 2, or 3")

    return sp, diaginv, diaginv_dbc


def update_hodge_operator(seq, geometry, operators: Optional[SequenceOperators], k: int):
    """Return an operator bundle with the k-th Hodge/stiffness data updated."""
    if operators is None:
        operators = SequenceOperators()
    # Stiffness blocks for k=0,1,2 are built from the topological incidence
    # ``G_k`` and mass ``M_{k+1}``; make sure ``G_k`` is available.
    if k in (0, 1, 2) and _incidence_components(operators, k)[0] is None:
        operators = update_incidence_operator(seq, operators, k)
    sp, diaginv, diaginv_dbc = _assemble_hodge_block(
        seq, geometry, operators, k)

    match k:
        case 0:
            fd_scale_K = _fd_hodge_scales_K(seq, geometry, 0)
            return eqx.tree_at(
                lambda ops: (ops.grad_grad_sp, ops.dd0_sp_diaginv,
                             ops.dd0_sp_diaginv_dbc, ops.dd0_fd_scale_K),
                operators,
                (sp, diaginv, diaginv_dbc, fd_scale_K),
                is_leaf=lambda x: x is None,
            )
        case 1:
            return eqx.tree_at(
                lambda ops: (ops.curl_curl_sp, ops.dd1_sp_diaginv,
                             ops.dd1_sp_diaginv_dbc),
                operators,
                (sp, diaginv, diaginv_dbc),
                is_leaf=lambda x: x is None,
            )
        case 2:
            return eqx.tree_at(
                lambda ops: (ops.div_div_sp, ops.dd2_sp_diaginv,
                             ops.dd2_sp_diaginv_dbc),
                operators,
                (sp, diaginv, diaginv_dbc),
                is_leaf=lambda x: x is None,
            )
        case 3:
            fd_scale_K = _fd_hodge_scales_K(seq, geometry, 3)
            return eqx.tree_at(
                lambda ops: (ops.dd3_sp_diaginv, ops.dd3_sp_diaginv_dbc,
                             ops.dd3_fd_scale_K),
                operators,
                (diaginv, diaginv_dbc, fd_scale_K),
                is_leaf=lambda x: x is None,
            )
    raise ValueError("k must be 0, 1, 2, or 3")


def assemble_hodge_operators(seq, geometry, operators: Optional[SequenceOperators] = None,
                             ks: Sequence[int] = (0, 1, 2, 3)):
    """Assemble Hodge/stiffness operators for the requested form degrees."""
    for k in ks:
        operators = update_hodge_operator(seq, geometry, operators, k)
    return operators


def assemble_all_operators(seq, geometry,
                           operators: Optional[SequenceOperators] = None):
    """Assemble all geometry-dependent mass, derivative, and Hodge operators."""
    operators = assemble_mass_operators(seq, geometry, operators=operators)
    operators = assemble_kron_mass_preconditioner(seq, operators=operators)
    operators = assemble_fd_hodge_preconditioner(seq, operators=operators)
    operators = assemble_incidence_operators(seq, operators=operators)
    operators = assemble_derivative_operators(
        seq, geometry, operators=operators)
    operators = assemble_hodge_operators(seq, geometry, operators=operators)
    operators = assemble_projection_operators(seq, operators=operators)
    return operators


def operators_from_coeffs(seq, coeffs,
                          ks: Sequence[int] = (0,),
                          kinds: Sequence[str] = ("mass", "derivative", "hodge")):
    """Build operators from spline-map coefficients.

    Routes ``coeffs`` through :meth:`DeRhamSequence.geometry_from_spline_map`
    and assembles only the requested operator ``kinds`` for the requested
    form degrees ``ks``. Useful as a pure function of ``coeffs`` for
    adjoint / shape-derivative workflows, where assembling the full
    operator bundle on every gradient call is wasteful.

    Parameters
    ----------
    seq : DeRhamSequence
    coeffs : (3, n_dof) array
        Cartesian spline coefficients defining the physical map.
    ks : sequence of int
        Form degrees to assemble (subset of ``(0, 1, 2, 3)``).
    kinds : sequence of str
        Any subset of ``("mass", "derivative", "hodge")``.

    Returns
    -------
    (operators, geometry) : (SequenceOperators, SequenceGeometry)
    """
    geometry = seq.geometry_from_spline_map(coeffs)
    ops: Optional[SequenceOperators] = None
    if "mass" in kinds:
        ops = assemble_mass_operators(seq, geometry, operators=ops, ks=ks)
    if "derivative" in kinds:
        ops = assemble_incidence_operators(seq, operators=ops, ks=ks)
        ops = assemble_derivative_operators(
            seq, geometry, operators=ops, ks=ks)
    if "hodge" in kinds:
        ops = assemble_hodge_operators(seq, geometry, operators=ops, ks=ks)
    return ops, geometry


def _mass_extraction(seq, k: int, dirichlet: bool):
    match k:
        case 0:
            return (seq.e0_dbc, seq.e0_dbc_T) if dirichlet else (seq.e0, seq.e0_T)
        case 1:
            return (seq.e1_dbc, seq.e1_dbc_T) if dirichlet else (seq.e1, seq.e1_T)
        case 2:
            return (seq.e2_dbc, seq.e2_dbc_T) if dirichlet else (seq.e2, seq.e2_T)
        case 3:
            return (seq.e3_dbc, seq.e3_dbc_T) if dirichlet else (seq.e3, seq.e3_T)
    raise ValueError("k must be 0, 1, 2 or 3")


def _derivative_extraction(seq, k: int, dirichlet_in: bool, dirichlet_out: bool):
    match k:
        case 0:
            e_in = seq.e0_dbc if dirichlet_in else seq.e0
            e_in_T = seq.e0_dbc_T if dirichlet_in else seq.e0_T
            e_out = seq.e1_dbc if dirichlet_out else seq.e1
            e_out_T = seq.e1_dbc_T if dirichlet_out else seq.e1_T
        case 1:
            e_in = seq.e1_dbc if dirichlet_in else seq.e1
            e_in_T = seq.e1_dbc_T if dirichlet_in else seq.e1_T
            e_out = seq.e2_dbc if dirichlet_out else seq.e2
            e_out_T = seq.e2_dbc_T if dirichlet_out else seq.e2_T
        case 2:
            e_in = seq.e2_dbc if dirichlet_in else seq.e2
            e_in_T = seq.e2_dbc_T if dirichlet_in else seq.e2_T
            e_out = seq.e3_dbc if dirichlet_out else seq.e3
            e_out_T = seq.e3_dbc_T if dirichlet_out else seq.e3_T
        case _:
            raise ValueError("k must be 0, 1 or 2")
    return e_in, e_in_T, e_out, e_out_T


def dense_mass_matrix(seq, operators: SequenceOperators, k: int,
                      dirichlet: bool = True):
    """Return the dense extracted mass matrix for degree k."""
    sp, _, _ = _mass_components(operators, k)
    if sp is None:
        raise ValueError(f"Mass operator k={k} is not assembled")
    e, e_T = _mass_extraction(seq, k, dirichlet)
    return e.todense() @ sp.todense() @ e_T.todense()


def dense_derivative_matrix(seq, operators: SequenceOperators, k: int,
                            dirichlet_in: bool = True,
                            dirichlet_out: bool = True,
                            transpose: bool = False):
    """Return the dense extracted weak derivative matrix for degree k.

    ``D_k`` is materialised lazily from ``M_{k+1}`` and ``G_k`` via dense
    matmul; only used for debugging/reporting paths.
    """
    g_sp, _ = _incidence_components(operators, k)
    m_sp, _, _ = _mass_components(operators, k + 1)
    if g_sp is None:
        raise ValueError(f"Incidence operator G{k} is required for dense D{k}")
    if m_sp is None:
        raise ValueError(f"Mass operator M{k + 1} is required for dense D{k}")
    d_dense = m_sp.todense() @ g_sp.todense()
    e_in, e_in_T, e_out, e_out_T = _derivative_extraction(
        seq, k, dirichlet_in, dirichlet_out)
    if transpose:
        return e_in.todense() @ d_dense.T @ e_out_T.todense()
    return e_out.todense() @ d_dense @ e_in_T.todense()


def dense_stiffness_matrix(seq, operators: SequenceOperators, k: int,
                           dirichlet: bool = True):
    """Return the dense extracted stiffness matrix for degree k.

    ``K_k = G_k^T M_{k+1} G_k`` is materialised lazily via dense matmul;
    only used for debugging/reporting paths.
    """
    if k == 3:
        n = seq.n3_dbc if dirichlet else seq.n3
        return jnp.zeros((n, n))
    g_sp, _ = _incidence_components(operators, k)
    m_sp, _, _ = _mass_components(operators, k + 1)
    if g_sp is None:
        raise ValueError(f"Incidence operator G{k} is required for dense K{k}")
    if m_sp is None:
        raise ValueError(f"Mass operator M{k + 1} is required for dense K{k}")
    g_dense = g_sp.todense()
    k_dense = g_dense.T @ m_sp.todense() @ g_dense
    e, e_T = _mass_extraction(seq, k, dirichlet)
    return e.todense() @ k_dense @ e_T.todense()


def dense_hodge_laplacian(seq, operators: SequenceOperators, k: int,
                          dirichlet: bool = True):
    """Return the dense extracted Hodge Laplacian for degree k."""
    match k:
        case 0:
            return dense_stiffness_matrix(seq, operators, 0, dirichlet=dirichlet)
        case 1:
            stiffness = dense_stiffness_matrix(
                seq, operators, 1, dirichlet=dirichlet)
            derivative = dense_derivative_matrix(
                seq, operators, 0,
                dirichlet_in=dirichlet,
                dirichlet_out=dirichlet,
            )
            mass = dense_mass_matrix(seq, operators, 0, dirichlet=dirichlet)
            return stiffness + derivative @ jnp.linalg.solve(mass, derivative.T)
        case 2:
            stiffness = dense_stiffness_matrix(
                seq, operators, 2, dirichlet=dirichlet)
            derivative = dense_derivative_matrix(
                seq, operators, 1,
                dirichlet_in=dirichlet,
                dirichlet_out=dirichlet,
            )
            mass = dense_mass_matrix(seq, operators, 1, dirichlet=dirichlet)
            return stiffness + derivative @ jnp.linalg.solve(mass, derivative.T)
        case 3:
            derivative = dense_derivative_matrix(
                seq, operators, 2,
                dirichlet_in=dirichlet,
                dirichlet_out=dirichlet,
            )
            mass = dense_mass_matrix(seq, operators, 2, dirichlet=dirichlet)
            return derivative @ jnp.linalg.solve(mass, derivative.T)
    raise ValueError("k must be 0, 1, 2 or 3")


def dense_projection_matrix(seq, operators: SequenceOperators, k_in: int, k_out: int,
                            dirichlet_in: bool = True,
                            dirichlet_out: bool = True):
    """Return the dense extracted projection matrix for the requested degrees."""
    sp = _projection_components(operators, k_in, k_out)
    if sp is None:
        raise ValueError(
            f"Projection operator ({k_in}, {k_out}) is not assembled"
        )

    match (k_in, k_out):
        case (2, 1):
            e_in = seq.e2_dbc if dirichlet_in else seq.e2
            e_in_T = seq.e2_dbc_T if dirichlet_in else seq.e2_T
            e_out = seq.e1_dbc if dirichlet_out else seq.e1
        case (1, 2):
            e_in = seq.e1_dbc if dirichlet_in else seq.e1
            e_in_T = seq.e1_dbc_T if dirichlet_in else seq.e1_T
            e_out = seq.e2_dbc if dirichlet_out else seq.e2
        case (0, 3):
            e_in = seq.e3_dbc if dirichlet_in else seq.e3
            e_in_T = seq.e3_dbc_T if dirichlet_in else seq.e3_T
            e_out = seq.e0_dbc if dirichlet_out else seq.e0
        case (3, 0):
            e_in = seq.e0_dbc if dirichlet_in else seq.e0
            e_in_T = seq.e0_dbc_T if dirichlet_in else seq.e0_T
            e_out = seq.e3_dbc if dirichlet_out else seq.e3
        case _:
            raise ValueError(
                "Only (k_in, k_out) = (1, 2), (2, 1), (0, 3), or (3, 0) supported"
            )
    return e_out.todense() @ sp.todense() @ e_in_T.todense()


def apply_mass_matrix(seq, operators: SequenceOperators, v, k: int, dirichlet: bool = True):
    """Apply a mass matrix from an explicit operator bundle."""
    sp, _, _ = _mass_components(operators, k)
    if sp is None:
        raise ValueError(f"Mass operator k={k} is not assembled")

    match k:
        case 0:
            e = seq.e0_dbc if dirichlet else seq.e0
            e_T = seq.e0_dbc_T if dirichlet else seq.e0_T
        case 1:
            e = seq.e1_dbc if dirichlet else seq.e1
            e_T = seq.e1_dbc_T if dirichlet else seq.e1_T
        case 2:
            e = seq.e2_dbc if dirichlet else seq.e2
            e_T = seq.e2_dbc_T if dirichlet else seq.e2_T
        case 3:
            e = seq.e3_dbc if dirichlet else seq.e3
            e_T = seq.e3_dbc_T if dirichlet else seq.e3_T
        case _:
            raise ValueError("k must be 0, 1, 2 or 3")
    return e @ (sp @ (e_T @ v))


def apply_projection_matrix(seq, operators: SequenceOperators, v,
                            k_in: int, k_out: int,
                            dirichlet_in: bool = True,
                            dirichlet_out: bool = True):
    """Apply a projection matrix from an explicit operator bundle."""
    sp = _projection_components(operators, k_in, k_out)
    if sp is None:
        raise ValueError(
            f"Projection operator ({k_in}, {k_out}) is not assembled"
        )

    match (k_in, k_out):
        case (2, 1):
            e_in = seq.e2_dbc if dirichlet_in else seq.e2
            e_in_T = seq.e2_dbc_T if dirichlet_in else seq.e2_T
            e_out = seq.e1_dbc if dirichlet_out else seq.e1
        case (1, 2):
            e_in = seq.e1_dbc if dirichlet_in else seq.e1
            e_in_T = seq.e1_dbc_T if dirichlet_in else seq.e1_T
            e_out = seq.e2_dbc if dirichlet_out else seq.e2
        case (0, 3):
            e_in = seq.e3_dbc if dirichlet_in else seq.e3
            e_in_T = seq.e3_dbc_T if dirichlet_in else seq.e3_T
            e_out = seq.e0_dbc if dirichlet_out else seq.e0
        case (3, 0):
            e_in = seq.e0_dbc if dirichlet_in else seq.e0
            e_in_T = seq.e0_dbc_T if dirichlet_in else seq.e0_T
            e_out = seq.e3_dbc if dirichlet_out else seq.e3
        case _:
            raise ValueError(
                "Only (k_in, k_out) = (1, 2), (2, 1), (0, 3), or (3, 0) supported"
            )

    return e_out @ (sp @ (e_in_T @ v))


def apply_derivative_matrix(seq, operators: SequenceOperators, v, k: int,
                            dirichlet_in: bool = True,
                            dirichlet_out: bool = True,
                            transpose: bool = False):
    """Apply a weak derivative matrix from an explicit operator bundle.

    ``D_k = M_{k+1} G_k`` is applied as a composition of BCSR matvecs; the
    full ``D_k`` is never materialised.
    """
    g_sp, g_sp_T = _incidence_components(operators, k)
    m_sp, _, _ = _mass_components(operators, k + 1)
    if g_sp is None or g_sp_T is None:
        raise ValueError(f"Incidence operator G{k} is required to apply D{k}")
    if m_sp is None:
        raise ValueError(f"Mass operator M{k + 1} is required to apply D{k}")

    match k:
        case 0:
            e_in = seq.e0_dbc if dirichlet_in else seq.e0
            e_in_T = seq.e0_dbc_T if dirichlet_in else seq.e0_T
            e_out = seq.e1_dbc if dirichlet_out else seq.e1
            e_out_T = seq.e1_dbc_T if dirichlet_out else seq.e1_T
        case 1:
            e_in = seq.e1_dbc if dirichlet_in else seq.e1
            e_in_T = seq.e1_dbc_T if dirichlet_in else seq.e1_T
            e_out = seq.e2_dbc if dirichlet_out else seq.e2
            e_out_T = seq.e2_dbc_T if dirichlet_out else seq.e2_T
        case 2:
            e_in = seq.e2_dbc if dirichlet_in else seq.e2
            e_in_T = seq.e2_dbc_T if dirichlet_in else seq.e2_T
            e_out = seq.e3_dbc if dirichlet_out else seq.e3
            e_out_T = seq.e3_dbc_T if dirichlet_out else seq.e3_T
        case _:
            raise ValueError("k must be 0, 1 or 2")

    if transpose:
        # D^T v = G^T M^T v = G^T (M v) (M is symmetric)
        return e_in @ (g_sp_T @ (m_sp @ (e_out_T @ v)))
    return e_out @ (m_sp @ (g_sp @ (e_in_T @ v)))


def apply_mass_matrix_preconditioner(seq, operators: SequenceOperators, v, k: int,
                                     dirichlet: bool = True,
                                     kind: str = 'auto'):
    """Apply a mass-matrix preconditioner from an explicit operator bundle.

    Parameters
    ----------
    kind : {'auto', 'jacobi', 'kronecker'}
        Which preconditioner to use.  ``'auto'`` picks ``'kronecker'`` when
        the 1-D mass-matrix inverses have been assembled and are available
        for this ``k`` (see :func:`assemble_kron_mass_preconditioner`),
        otherwise ``'jacobi'``.
    """
    if kind == 'auto':
        kind = 'kronecker' if _kron_available(seq, operators, k) else 'jacobi'
    if kind == 'kronecker':
        if not _kron_available(seq, operators, k):
            raise ValueError(
                f"Kronecker mass preconditioner not assembled for k={k}; "
                "call assemble_kron_mass_preconditioner(seq, operators) first")
        return apply_mass_kron_preconditioner(
            seq, operators, v, k, dirichlet=dirichlet)
    if kind == 'jacobi':
        _, diaginv, diaginv_dbc = _mass_components(operators, k)
        if diaginv is None or diaginv_dbc is None:
            raise ValueError(f"Mass preconditioner k={k} is not assembled")
        return (diaginv_dbc if dirichlet else diaginv) * v
    raise ValueError(
        f"kind must be 'auto', 'jacobi' or 'kronecker' (got {kind!r})")


def apply_inverse_mass_matrix(seq, operators: SequenceOperators, v, k: int,
                              dirichlet: bool = True, guess=None,
                              tol: Optional[float] = None,
                              maxiter: Optional[int] = None,
                              precond: str = 'auto'):
    """Solve with the inverse mass matrix from an explicit operator bundle.

    ``precond`` is forwarded as ``kind`` to
    :func:`apply_mass_matrix_preconditioner`.
    """
    tol = seq.tol if tol is None else tol
    maxiter = seq.maxiter if maxiter is None else maxiter
    return solve_singular_cg(
        lambda x: apply_mass_matrix(seq, operators, x, k, dirichlet=dirichlet),
        v,
        mass_matvec=lambda x: apply_mass_matrix(
            seq, operators, x, k, dirichlet=dirichlet),
        precond_matvec=lambda x: apply_mass_matrix_preconditioner(
            seq, operators, x, k, dirichlet=dirichlet, kind=precond),
        x0=guess,
        tol=tol,
        maxiter=maxiter,
    )[0]


def apply_stiffness(seq, operators: SequenceOperators, v, k: int, dirichlet: bool = True):
    """Apply a stiffness matrix from an explicit operator bundle.

    ``K_k = G_k^T M_{k+1} G_k`` is applied as a composition of BCSR matvecs;
    the full ``K_k`` is never materialised.
    """
    if k == 3:
        return jnp.zeros_like(v)
    g_sp, g_sp_T = _incidence_components(operators, k)
    m_sp, _, _ = _mass_components(operators, k + 1)
    if g_sp is None or g_sp_T is None:
        raise ValueError(f"Incidence operator G{k} is required to apply K{k}")
    if m_sp is None:
        raise ValueError(f"Mass operator M{k + 1} is required to apply K{k}")

    match k:
        case 0:
            e = seq.e0_dbc if dirichlet else seq.e0
            e_T = seq.e0_dbc_T if dirichlet else seq.e0_T
        case 1:
            e = seq.e1_dbc if dirichlet else seq.e1
            e_T = seq.e1_dbc_T if dirichlet else seq.e1_T
        case 2:
            e = seq.e2_dbc if dirichlet else seq.e2
            e_T = seq.e2_dbc_T if dirichlet else seq.e2_T
        case _:
            raise ValueError("k must be 0, 1, 2 or 3")
    return e @ (g_sp_T @ (m_sp @ (g_sp @ (e_T @ v))))


def apply_hodge_hx_preconditioner(seq, operators: SequenceOperators, v,
                                  k: int, dirichlet: bool = True,
                                  eps: float = 0.0):
    """Hiptmair-Xu-style auxiliary-space preconditioner for ``L_k + eps M_k``.

    Implemented for ``k = 3`` with both Dirichlet and natural BCs:

        P^{-1} v = (diag(L_3) + eps diag(M_3))^{-1} v
                 + tilde(M_3)^{-1} M_{03} (tilde(L_0) + eps tilde(M_0))^{-1}
                                             M_{30} tilde(M_3)^{-1} v

    The first term is a local shifted-Jacobi smoother on the shifted
    Hodge Laplacian itself, killing high-frequency error components.
    The second term is the auxiliary-space correction exploiting the
    Hodge duality ``star : V_3 -> V_0``: it maps into the 0-form space,
    applies the (well-preconditioned) shifted scalar Laplacian inverse,
    and maps back. The tildes denote Kronecker fast-diagonalisation
    preconditioners (one direct apply each). The sandwich is manifestly
    SPD and so is the whole operator.

    The auxiliary 0-form space uses the **dual** BC: ``dirichlet=True``
    (DBC on 3-forms) maps to NBC 0-forms, and vice versa.

    For ``eps == 0`` the 0-form inverse is a Moore-Penrose pseudo-inverse
    (the constant mode is projected out); for ``eps > 0`` the shift lifts
    every eigenvalue off zero and no null handling is needed.
    """
    if k != 3:
        raise ValueError(
            "HX preconditioner currently only implemented for k=3")
    # Auxiliary 0-form space uses the Hodge-dual BC.
    aux_dirichlet = not dirichlet
    # Local smoother: shifted Jacobi on L_3 + eps M_3.
    if eps == 0:
        smooth_diaginv = _hodge_diaginv(operators, 3, dirichlet=dirichlet)
    else:
        stiffness_diaginv = _hodge_diaginv(operators, 3, dirichlet=dirichlet)
        mass_diaginv_3 = _mass_diaginv(operators, 3, dirichlet=dirichlet)
        smooth_diaginv = 1.0 / \
            (1.0 / stiffness_diaginv + eps / mass_diaginv_3)
    smooth = smooth_diaginv * v
    # Auxiliary-space correction: tilde(M_3)^{-1} M_{03}
    #   (tilde(L_0) + eps tilde(M_0))^{-1} M_{30} tilde(M_3)^{-1}.
    # Read right-to-left; each step is a single direct apply.
    w = apply_mass_kron_preconditioner(
        seq, operators, v, 3, dirichlet=dirichlet)
    # M_{03}: 3-form primal -> 0-form dual (dual BC on 0-forms).
    w = apply_projection_matrix(
        seq, operators, w, k_in=0, k_out=3,
        dirichlet_in=dirichlet, dirichlet_out=aux_dirichlet)
    # (tilde(L_0) + eps tilde(M_0))^{-1}: 0-form dual -> 0-form primal.
    w = apply_hodge_kron_preconditioner(
        seq, operators, w, 0, dirichlet=aux_dirichlet, eps=eps)
    # M_{30} = M_{03}^T: 0-form primal -> 3-form dual.
    w = apply_projection_matrix(
        seq, operators, w, k_in=3, k_out=0,
        dirichlet_in=aux_dirichlet, dirichlet_out=dirichlet)
    # tilde(M_3)^{-1}: 3-form dual -> 3-form primal.
    w = apply_mass_kron_preconditioner(
        seq, operators, w, 3, dirichlet=dirichlet)
    return smooth + w


def apply_hodge_laplacian_preconditioner(seq, operators: SequenceOperators, v, k: int,
                                         dirichlet: bool = True,
                                         kind: str = 'auto'):
    """Apply the Hodge-Laplacian preconditioner from an operator bundle.

    ``kind`` options:

    * ``'none'`` — identity (no preconditioning).
    * ``'jacobi'`` — per-DoF diagonal of ``L_k``; always available.
    * ``'hx'`` — Hiptmair-Xu auxiliary-space preconditioner. For ``k = 0``
      this reduces to the fast-diagonalisation Laplacian inverse (no
      auxiliary space needed); for ``k = 3`` it uses the ``V_3 ↔ V_0``
      Hodge duality and :func:`apply_hodge_hx_preconditioner` (with the
      dual BC on the auxiliary 0-form space).
      Not available for ``k = 1, 2``.
    * ``'auto'`` — picks ``'hx'`` when available (``k = 0`` or ``k = 3``
      with FD assembled) and falls back to ``'jacobi'``.
    """
    if kind not in ('auto', 'none', 'jacobi', 'hx'):
        raise ValueError(
            f"kind must be 'auto', 'none', 'jacobi' or 'hx' (got {kind!r})")
    if kind == 'auto':
        kind = 'jacobi'
    if kind == 'none':
        return v
    if kind == 'jacobi':
        return _hodge_diaginv(operators, k, dirichlet) * v
    if kind == 'hx':
        if k == 0:
            if not _fd_hodge_available(operators, 0):
                raise ValueError(
                    "HX preconditioner for k=0 requires the FD Hodge "
                    "preconditioner; call assemble_fd_hodge_preconditioner first")
            return apply_hodge_kron_preconditioner(
                seq, operators, v, 0, dirichlet=dirichlet)
        if k == 3:
            if not _fd_hodge_available(operators, 0):
                raise ValueError(
                    "HX preconditioner for k=3 requires the FD Hodge "
                    "preconditioner; call assemble_fd_hodge_preconditioner first")
            return apply_hodge_hx_preconditioner(
                seq, operators, v, 3, dirichlet=dirichlet)
        raise ValueError(
            f"HX preconditioner not available for k={k}; use 'jacobi' instead")
    raise AssertionError("unreachable")


def apply_inverse_shifted_hodge_laplacian(seq, operators: SequenceOperators, v, k: int,
                                          eps: float, dirichlet: bool = True, guess=None,
                                          tol: Optional[float] = None,
                                          maxiter: Optional[int] = None,
                                          precond_kind: str = 'auto',
                                          lower_precond_kind: str = 'auto',
                                          use_harmonic_coarse: Optional[bool] = None,
                                          return_info: bool = False):
    """Solve with the inverse of the shifted Hodge Laplacian ``L_k + eps M_k``.

    For ``k = 0`` this is the scalar system ``(S_0 + eps M_0) u = v`` (CG).
    For ``k >= 1`` it is the symmetric saddle-point form of ``L_k + eps M_k``
    with Schur block ``-M_{k-1}`` (MINRES).  The upper-block matrix is
    ``S_k + eps M_k`` -- i.e. only the stiffness (grad-grad / curl-curl /
    div-div) block is shifted; the Schur complement embedded by the saddle
    system completes it to the full Hodge Laplacian.

    ``precond_kind`` selects the upper-block (k-form) preconditioner.
    Accepts ``'auto'``, ``'none'``, ``'jacobi'``, or ``'hx'`` — see
    :func:`apply_hodge_laplacian_preconditioner` for the ``eps = 0``
    meaning.  For ``eps > 0`` the ``'hx'`` path uses the shifted
    fast-diagonalisation inverse of ``L_k + eps M_k`` (available for
    ``k = 0`` and ``k = 3`` when FD is assembled); ``'auto'`` picks
    ``'hx'`` when available and falls back to the shifted-Jacobi diagonal
    otherwise.  ``'none'`` is rejected for ``eps > 0``.

    For ``k >= 1`` MINRES additionally needs a preconditioner for the
    lower (k-1)-form mass block.  When the Kronecker mass preconditioner
    is assembled for ``k-1`` and ``precond_kind != 'jacobi'`` it is used;
    otherwise the Jacobi mass diagonal is used.
    """
    tol = seq.tol if tol is None else tol
    maxiter = seq.maxiter if maxiter is None else maxiter
    if precond_kind not in ('auto', 'none', 'jacobi', 'hx'):
        raise ValueError(
            f"precond_kind must be 'auto', 'none', 'jacobi' or 'hx' "
            f"(got {precond_kind!r})")
    if lower_precond_kind not in ('auto', 'jacobi', 'kronecker'):
        raise ValueError(
            "lower_precond_kind must be 'auto', 'jacobi' or 'kronecker' "
            f"(got {lower_precond_kind!r})")

    def _hodge_precond(x):
        return apply_hodge_laplacian_preconditioner(
            seq, operators, x, k, dirichlet=dirichlet, kind=precond_kind)

    if eps == 0:
        precond_upper = _hodge_precond
    else:
        if precond_kind not in ('auto', 'jacobi', 'hx'):
            raise ValueError(
                "precond_kind other than 'jacobi' or 'hx' is not supported "
                "for eps > 0")
        # Availability of the shifted HX/Kron FD paths.
        shifted_hx_ok = (
            k == 3 and _fd_hodge_available(operators, 0)
        )
        shifted_kron_ok = (
            k == 0 and _fd_hodge_available(operators, 0)
        )
        if precond_kind == 'hx' and not (shifted_hx_ok or shifted_kron_ok):
            raise ValueError(
                f"Shifted HX preconditioner not available for k={k} "
                f"(dirichlet={dirichlet}); use 'jacobi' instead")
        use_shifted_hx = (
            precond_kind == 'hx' and shifted_hx_ok
        )
        use_shifted_kron = (
            precond_kind == 'hx' and shifted_kron_ok
        )
        if use_shifted_hx:
            def precond_upper(x):
                return apply_hodge_hx_preconditioner(
                    seq, operators, x, 3, dirichlet=dirichlet, eps=eps)
        elif use_shifted_kron:
            def precond_upper(x):
                return apply_hodge_kron_preconditioner(
                    seq, operators, x, 0, dirichlet=dirichlet, eps=eps)
        else:
            stiffness_diaginv = _hodge_diaginv(operators, k, dirichlet)
            mass_diaginv_k = _mass_diaginv(operators, k, dirichlet)
            shifted_diaginv = 1.0 / \
                (1.0 / stiffness_diaginv + eps / mass_diaginv_k)

            def precond_upper(x):
                return shifted_diaginv * x

        if use_harmonic_coarse is None:
            wrap_harmonic_coarse = (
                (use_shifted_hx and k == 3 and dirichlet)
                or (use_shifted_kron and k == 0 and not dirichlet)
            )
        else:
            wrap_harmonic_coarse = use_harmonic_coarse
        if wrap_harmonic_coarse:
            precond_upper = _wrap_shifted_harmonic_coarse_correction(
                seq, operators, precond_upper, eps, k, dirichlet)

    if k == 0:
        vs = _nullspace_vectors(
            operators, 0, dirichlet) if eps == 0 else jnp.zeros((0, v.shape[0]))
        u, info = solve_singular_cg(
            lambda x: apply_stiffness(
                seq, operators, x, 0, dirichlet=dirichlet)
            + eps * apply_mass_matrix(seq, operators,
                                      x, 0, dirichlet=dirichlet),
            v,
            mass_matvec=(
                lambda x: apply_mass_matrix(
                    seq, operators, x, 0, dirichlet=dirichlet)
            ) if eps == 0 else None,
            precond_matvec=precond_upper,
            x0=guess,
            vs=vs,
            tol=tol,
            maxiter=maxiter,
        )
        return (u, info) if return_info else u

    vs_upper, vs_lower = _saddle_nullspaces(
        seq, operators, k, dirichlet) if eps == 0 else (
            jnp.zeros((0, v.shape[0])), jnp.zeros((0, 0)))
    mass_lower_diaginv = _mass_diaginv(operators, k - 1, dirichlet)
    if lower_precond_kind == 'auto':
        use_kron_lower = (precond_kind != 'jacobi'
                          and _kron_available(seq, operators, k - 1))
    elif lower_precond_kind == 'kronecker':
        if not _kron_available(seq, operators, k - 1):
            raise ValueError(
                f"Kronecker lower preconditioner not available for k={k - 1}")
        use_kron_lower = True
    else:
        use_kron_lower = False
    suffix = "_dbc" if dirichlet else ""
    n_upper = getattr(seq, f"n{k}{suffix}")
    n_lower = getattr(seq, f"n{k-1}{suffix}")

    if use_kron_lower:
        def precond_lower(x):
            return apply_mass_kron_preconditioner(
                seq, operators, x, k - 1, dirichlet=dirichlet)
    else:
        def precond_lower(x):
            return mass_lower_diaginv * x

    u, sigma, info = solve_saddle_point_minres(
        stiffness_matvec=lambda x: apply_stiffness(
            seq, operators, x, k, dirichlet=dirichlet)
        + eps * apply_mass_matrix(seq, operators, x, k, dirichlet=dirichlet),
        derivative_matvec=lambda s: apply_derivative_matrix(
            seq, operators, s, k - 1, dirichlet_in=dirichlet, dirichlet_out=dirichlet),
        derivative_T_matvec=lambda u: apply_derivative_matrix(
            seq, operators, u, k - 1, dirichlet_in=dirichlet,
            dirichlet_out=dirichlet, transpose=True),
        mass_lower_matvec=lambda s: apply_mass_matrix(
            seq, operators, s, k - 1, dirichlet=dirichlet),
        b_upper=v,
        n_upper=n_upper,
        n_lower=n_lower,
        precond_upper=precond_upper,
        precond_lower=precond_lower,
        mass_upper_matvec=lambda x: apply_mass_matrix(
            seq, operators, x, k, dirichlet=dirichlet),
        vs_upper=vs_upper,
        vs_lower=vs_lower,
        x0_upper=guess,
        tol=tol,
        maxiter=maxiter,
    )
    return (u, info) if return_info else u


def apply_inverse_diffusion(seq, operators: SequenceOperators, v, k: int,
                            alpha: float, dirichlet: bool = True, guess=None,
                            tol: Optional[float] = None,
                            maxiter: Optional[int] = None):
    """Solve with the inverse of M_k + alpha L_k using an explicit operator bundle."""
    tol = seq.tol if tol is None else tol
    maxiter = seq.maxiter if maxiter is None else maxiter

    if k == 0:
        mass_diaginv = _mass_diaginv(operators, 0, dirichlet)
        stiffness_diaginv = _hodge_diaginv(operators, 0, dirichlet)
        diaginv = 1.0 / (1.0 / mass_diaginv + alpha / stiffness_diaginv)
        return solve_singular_cg(
            lambda x: apply_mass_matrix(
                seq, operators, x, 0, dirichlet=dirichlet)
            + alpha * apply_stiffness(seq, operators,
                                      x, 0, dirichlet=dirichlet),
            v,
            precond_matvec=lambda x: diaginv * x,
            x0=guess,
            tol=tol,
            maxiter=maxiter,
        )[0]

    mass_diaginv = _mass_diaginv(operators, k, dirichlet)
    mass_lower_diaginv = _mass_diaginv(operators, k - 1, dirichlet)
    suffix = "_dbc" if dirichlet else ""
    n_upper = getattr(seq, f"n{k}{suffix}")
    n_lower = getattr(seq, f"n{k-1}{suffix}")

    def precond_lower(x):
        return (1.0 / alpha) * mass_lower_diaginv * x

    def precond_upper(x):
        return mass_diaginv * x

    u, sigma, info = solve_saddle_point_minres(
        stiffness_matvec=lambda x: apply_mass_matrix(
            seq, operators, x, k, dirichlet=dirichlet)
        + alpha * apply_stiffness(seq, operators, x, k, dirichlet=dirichlet),
        derivative_matvec=lambda s: alpha * apply_derivative_matrix(
            seq, operators, s, k - 1, dirichlet_in=dirichlet, dirichlet_out=dirichlet),
        derivative_T_matvec=lambda u: alpha * apply_derivative_matrix(
            seq, operators, u, k - 1, dirichlet_in=dirichlet,
            dirichlet_out=dirichlet, transpose=True),
        mass_lower_matvec=lambda s: alpha * apply_mass_matrix(
            seq, operators, s, k - 1, dirichlet=dirichlet),
        b_upper=v,
        n_upper=n_upper,
        n_lower=n_lower,
        precond_upper=precond_upper,
        precond_lower=precond_lower,
        x0_upper=guess,
        tol=tol,
        maxiter=maxiter,
    )
    return u


def apply_hodge_laplacian(seq, operators: SequenceOperators, v, k: int,
                          dirichlet: bool = True, guess=None,
                          tol: Optional[float] = None,
                          maxiter: Optional[int] = None):
    """Apply the Hodge Laplacian using explicit operator data.

    This uses bundled mass, weak derivative, and stiffness operators.
    """
    match k:
        case 0:
            return apply_stiffness(seq, operators, v, 0, dirichlet=dirichlet)
        case 1:
            Dt_v = apply_derivative_matrix(
                seq, operators,
                v, 0, dirichlet_in=dirichlet, dirichlet_out=dirichlet, transpose=True)
            Minv_Dt_v = apply_inverse_mass_matrix(
                seq, operators, Dt_v, 0, dirichlet=dirichlet,
                guess=guess, tol=tol, maxiter=maxiter)
            return apply_stiffness(seq, operators, v, 1, dirichlet=dirichlet) + \
                apply_derivative_matrix(
                    seq, operators,
                    Minv_Dt_v, 0, dirichlet_in=dirichlet, dirichlet_out=dirichlet)
        case 2:
            Dt_v = apply_derivative_matrix(
                seq, operators,
                v, 1, dirichlet_in=dirichlet, dirichlet_out=dirichlet, transpose=True)
            Minv_Dt_v = apply_inverse_mass_matrix(
                seq, operators, Dt_v, 1, dirichlet=dirichlet,
                guess=guess, tol=tol, maxiter=maxiter)
            return apply_stiffness(seq, operators, v, 2, dirichlet=dirichlet) + \
                apply_derivative_matrix(
                    seq, operators,
                    Minv_Dt_v, 1, dirichlet_in=dirichlet, dirichlet_out=dirichlet)
        case 3:
            Dt_v = apply_derivative_matrix(
                seq, operators,
                v, 2, dirichlet_in=dirichlet, dirichlet_out=dirichlet, transpose=True)
            Minv_Dt_v = apply_inverse_mass_matrix(
                seq, operators, Dt_v, 2, dirichlet=dirichlet,
                guess=guess, tol=tol, maxiter=maxiter)
            return apply_derivative_matrix(
                seq, operators,
                Minv_Dt_v, 2, dirichlet_in=dirichlet, dirichlet_out=dirichlet)
        case _:
            raise ValueError("k must be 0, 1, 2 or 3")


def apply_hodge_laplacian_approx(seq, operators: SequenceOperators, v, k: int,
                                 dirichlet: bool = True):
    """Linear approximation of the Hodge Laplacian apply.

    Replaces the exact ``M_{k-1}^{-1}`` in the Schur term of ``L_k`` with one
    apply of the Kronecker mass preconditioner (a single direct solve of a
    fast-diagonalisation surrogate).  The result is a fully linear SPD
    matvec: safe to nest inside Krylov iterations and to use as a
    preconditioner or a diagnostic ``L_k``-apply.  It is not exactly
    ``L_k`` unless the metric is tensor-separable on the reference domain.
    """
    match k:
        case 0:
            return apply_stiffness(seq, operators, v, 0, dirichlet=dirichlet)
        case 1:
            Dt_v = apply_derivative_matrix(
                seq, operators, v, 0,
                dirichlet_in=dirichlet, dirichlet_out=dirichlet, transpose=True)
            Minv_Dt_v = apply_mass_kron_preconditioner(
                seq, operators, Dt_v, 0, dirichlet=dirichlet)
            return apply_stiffness(seq, operators, v, 1, dirichlet=dirichlet) + \
                apply_derivative_matrix(
                    seq, operators, Minv_Dt_v, 0,
                    dirichlet_in=dirichlet, dirichlet_out=dirichlet)
        case 2:
            Dt_v = apply_derivative_matrix(
                seq, operators, v, 1,
                dirichlet_in=dirichlet, dirichlet_out=dirichlet, transpose=True)
            Minv_Dt_v = apply_mass_kron_preconditioner(
                seq, operators, Dt_v, 1, dirichlet=dirichlet)
            return apply_stiffness(seq, operators, v, 2, dirichlet=dirichlet) + \
                apply_derivative_matrix(
                    seq, operators, Minv_Dt_v, 1,
                    dirichlet_in=dirichlet, dirichlet_out=dirichlet)
        case 3:
            Dt_v = apply_derivative_matrix(
                seq, operators, v, 2,
                dirichlet_in=dirichlet, dirichlet_out=dirichlet, transpose=True)
            Minv_Dt_v = apply_mass_kron_preconditioner(
                seq, operators, Dt_v, 2, dirichlet=dirichlet)
            return apply_derivative_matrix(
                seq, operators, Minv_Dt_v, 2,
                dirichlet_in=dirichlet, dirichlet_out=dirichlet)
        case _:
            raise ValueError("k must be 0, 1, 2 or 3")
