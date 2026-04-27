from __future__ import annotations

from typing import Optional, Sequence

import equinox as eqx
import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import jax.scipy as jsp

from mrx.assembly import assemble_scalar_tp, assemble_vectorial_tp
from mrx.preconditioners import (
    BoundaryConditionPair,
    MassPreconditioners,
    MassPreconditionerSpec,
    SchurPreconditionerSpec,
    SaddlePointPreconditionerSpec,
    _bulk_tensor_shape,
    _core_size,
    _cp_als_3tensor,
    _split_blocks,
    _symmetrize,
    apply_mass_tensor_preconditioner,
    apply_mass_kronecker_preconditioner,
    build_mass_jacobi_pair,
    build_mass_tensor_preconditioner,
    build_mass_kronecker_preconditioner,
    default_mass_preconditioner,
    default_saddle_preconditioner,
    get_mass_jacobi_diaginv,
    mass_tensor_available,
    mass_kronecker_available,
    select_boundary_data,
    set_mass_jacobi_pair,
    set_mass_tensor,
    set_mass_kronecker,
)
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
    vs = _nullspace_vectors(operators, k, dirichlet)
    if vs.shape[0] == 0:
        return jnp.zeros(n_dof)
    stored = vs[0]
    stored_norm = seq.l2_norm(stored, k, dirichlet=dirichlet)
    return stored / jnp.where(stored_norm > 0, stored_norm, 1.0)


def _shifted_harmonic_coarse_ready(
        seq, operators: SequenceOperators, k: int, dirichlet: bool) -> bool:
    """True iff a nonzero stored harmonic coarse vector is available."""
    vs = _nullspace_vectors(operators, k, dirichlet)
    if vs.shape[0] == 0:
        return jnp.asarray(False)
    stored = vs[0]
    stored_norm = seq.l2_norm(stored, k, dirichlet=dirichlet)
    return stored_norm > 0


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
    k0_tensor_hodge_precond: Optional[BoundaryConditionPair] = None
    e0: Optional[jsparse.BCSR] = None
    e0_T: Optional[jsparse.BCSR] = None
    e0_dbc: Optional[jsparse.BCSR] = None
    e0_dbc_T: Optional[jsparse.BCSR] = None
    e0_bc: Optional[jsparse.BCSR] = None
    e0_bc_T: Optional[jsparse.BCSR] = None
    e1: Optional[jsparse.BCSR] = None
    e1_T: Optional[jsparse.BCSR] = None
    e1_dbc: Optional[jsparse.BCSR] = None
    e1_dbc_T: Optional[jsparse.BCSR] = None
    e1_bc: Optional[jsparse.BCSR] = None
    e1_bc_T: Optional[jsparse.BCSR] = None
    e2: Optional[jsparse.BCSR] = None
    e2_T: Optional[jsparse.BCSR] = None
    e2_dbc: Optional[jsparse.BCSR] = None
    e2_dbc_T: Optional[jsparse.BCSR] = None
    e2_bc: Optional[jsparse.BCSR] = None
    e2_bc_T: Optional[jsparse.BCSR] = None
    e3: Optional[jsparse.BCSR] = None
    e3_T: Optional[jsparse.BCSR] = None
    e3_dbc: Optional[jsparse.BCSR] = None
    e3_dbc_T: Optional[jsparse.BCSR] = None
    e3_bc: Optional[jsparse.BCSR] = None
    e3_bc_T: Optional[jsparse.BCSR] = None
    mass_preconds: Optional[MassPreconditioners] = None
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


class K0TensorHodgePreconditionerFactors(eqx.Module):
    core_size: int = eqx.field(static=True)
    acb: jnp.ndarray
    abc: jnp.ndarray
    bulk_inv: jnp.ndarray
    schur_inv: jnp.ndarray
    schur_projector: Optional[jnp.ndarray] = None


_EXTRACTION_OPERATOR_NAMES = (
    'e0', 'e0_T', 'e0_dbc', 'e0_dbc_T', 'e0_bc', 'e0_bc_T',
    'e1', 'e1_T', 'e1_dbc', 'e1_dbc_T', 'e1_bc', 'e1_bc_T',
    'e2', 'e2_T', 'e2_dbc', 'e2_dbc_T', 'e2_bc', 'e2_bc_T',
    'e3', 'e3_T', 'e3_dbc', 'e3_dbc_T', 'e3_bc', 'e3_bc_T',
)


def _ensure_extraction_operators(seq, operators: Optional[SequenceOperators]):
    if operators is None:
        operators = SequenceOperators()
    current = seq.get_operators() if hasattr(seq, 'get_operators') else None
    if current is not None:
        replacements = {
            name: getattr(current, name)
            for name in _EXTRACTION_OPERATOR_NAMES
            if getattr(operators, name, None) is None and getattr(current, name, None) is not None
        }
        if replacements:
            operators = eqx.tree_at(
                lambda ops: tuple(getattr(ops, name) for name in replacements),
                operators,
                tuple(replacements.values()),
                is_leaf=lambda x: x is None,
            )

    if getattr(operators, 'null_0', None) is None:
        from mrx.nullspace import init_nullspaces
        operators = init_nullspaces(
            seq,
            operators,
            betti_numbers=getattr(seq, 'betti_numbers', None),
        )

    return operators


def _reshape_quadrature_scalar_field(seq, values: jnp.ndarray) -> jnp.ndarray:
    return jnp.asarray(values).reshape(seq.quad.ny, seq.quad.nx, seq.quad.nz)


def _reshape_quadrature_matrix_field(seq, values: jnp.ndarray) -> jnp.ndarray:
    field = jnp.asarray(values)
    return field.reshape(seq.quad.ny, seq.quad.nx, seq.quad.nz, *field.shape[1:])


def _assemble_weighted_1d_mass(B: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    return (B * weights[None, :]) @ B.T


def _assemble_weighted_1d_stiffness(
        primal_basis: jnp.ndarray,
        derivative_basis: jnp.ndarray,
        weights: jnp.ndarray,
        incidence: jnp.ndarray) -> jnp.ndarray:
    mass_d = _assemble_weighted_1d_mass(derivative_basis, weights)
    stiffness = incidence.T @ (mass_d @ incidence)
    return _symmetrize(stiffness)


def _restrict_radial_window(raw_matrix: jnp.ndarray, radial_start: int,
                            nr: int) -> jnp.ndarray:
    radial_stop = radial_start + nr
    return raw_matrix[radial_start:radial_stop, radial_start:radial_stop]


def _k0_stiffness_diagonal_metric_tensors(seq) -> dict[str, jnp.ndarray]:
    jacobian = _reshape_quadrature_scalar_field(seq, seq.geometry.jacobian_j)
    metric_inv = _reshape_quadrature_matrix_field(seq, seq.geometry.metric_inv_jkl)
    return {
        'alpha_rr': jacobian * metric_inv[..., 0, 0],
        'alpha_thetatheta': jacobian * metric_inv[..., 1, 1],
        'alpha_zetazeta': jacobian * metric_inv[..., 2, 2],
    }


def _k0_tensor_hodge_config(operators: SequenceOperators):
    tensor = None if operators.mass_preconds is None else operators.mass_preconds.tensor
    if tensor is None:
        return 3, 100, 1e-9, 1e-12
    return tensor.rank, tensor.cp_maxiter, tensor.cp_tol, tensor.cp_ridge


def _build_k0_tensor_hodge_bulk_model(
        seq, *, dirichlet: bool, rank: int,
        cp_maxiter: int, cp_tol: float, cp_ridge: float) -> jnp.ndarray:
    bulk_shape = _bulk_tensor_shape(seq, dirichlet)
    nr_bulk, _, _ = bulk_shape
    metric_tensors = _k0_stiffness_diagonal_metric_tensors(seq)
    model_size = int(jnp.prod(jnp.asarray(bulk_shape)))
    model = jnp.zeros((model_size, model_size), dtype=jnp.float64)
    types = seq.basis_0.types
    g_r = _dense_incidence_1d(seq.basis_0.nr, types[0])
    g_t = _dense_incidence_1d(seq.basis_0.nt, types[1])
    g_z = _dense_incidence_1d(seq.basis_0.nz, types[2])

    field_specs = (
        (
            'alpha_rr',
            lambda scaled_weights: _restrict_radial_window(
                _assemble_weighted_1d_stiffness(
                    seq.basis_r_jk,
                    seq.d_basis_r_jk,
                    scaled_weights,
                    g_r,
                ),
                radial_start=2,
                nr=nr_bulk,
            ),
            lambda scaled_weights: _assemble_weighted_1d_mass(seq.basis_t_jk, scaled_weights),
            lambda scaled_weights: _assemble_weighted_1d_mass(seq.basis_z_jk, scaled_weights),
        ),
        (
            'alpha_thetatheta',
            lambda scaled_weights: _restrict_radial_window(
                _assemble_weighted_1d_mass(seq.basis_r_jk, scaled_weights),
                radial_start=2,
                nr=nr_bulk,
            ),
            lambda scaled_weights: _assemble_weighted_1d_stiffness(
                seq.basis_t_jk,
                seq.d_basis_t_jk,
                scaled_weights,
                g_t,
            ),
            lambda scaled_weights: _assemble_weighted_1d_mass(seq.basis_z_jk, scaled_weights),
        ),
        (
            'alpha_zetazeta',
            lambda scaled_weights: _restrict_radial_window(
                _assemble_weighted_1d_mass(seq.basis_r_jk, scaled_weights),
                radial_start=2,
                nr=nr_bulk,
            ),
            lambda scaled_weights: _assemble_weighted_1d_mass(seq.basis_t_jk, scaled_weights),
            lambda scaled_weights: _assemble_weighted_1d_stiffness(
                seq.basis_z_jk,
                seq.d_basis_z_jk,
                scaled_weights,
                g_z,
            ),
        ),
    )

    for label, radial_builder, theta_builder, zeta_builder in field_specs:
        weights, factors = _cp_als_3tensor(
            metric_tensors[label],
            rank,
            maxiter=cp_maxiter,
            tol=cp_tol,
            ridge=cp_ridge,
        )
        factor_theta, factor_r, factor_z = factors
        for idx in range(rank):
            term_r = radial_builder(seq.quad.w_x * (weights[idx] * factor_r[:, idx]))
            term_t = theta_builder(seq.quad.w_y * factor_theta[:, idx])
            term_z = zeta_builder(seq.quad.w_z * factor_z[:, idx])
            model = model + jnp.kron(jnp.kron(term_r, term_t), term_z)
    return _symmetrize(model)


def _assemble_k0_tensor_hodge_preconditioner(
        seq, operators: SequenceOperators, *,
        rank: int, cp_maxiter: int, cp_tol: float, cp_ridge: float,
        dirichlet_flags: tuple[bool, ...] = (False, True)) -> BoundaryConditionPair:
    pair = BoundaryConditionPair()
    core_size = _core_size(seq)

    for dirichlet in dirichlet_flags:
        matrix = jnp.asarray(dense_hodge_laplacian(seq, operators, 0, dirichlet=dirichlet))
        acc, acb, abc, _ = _split_blocks(matrix, core_size)
        bulk_model = _build_k0_tensor_hodge_bulk_model(
            seq,
            dirichlet=dirichlet,
            rank=rank,
            cp_maxiter=cp_maxiter,
            cp_tol=cp_tol,
            cp_ridge=cp_ridge,
        )
        bulk_inv = _symmetrize(jnp.linalg.inv(bulk_model))
        schur = _symmetrize(acc - acb @ (bulk_inv @ abc))

        schur_projector = None
        if dirichlet:
            schur_inv = _symmetrize(jnp.linalg.inv(schur))
        else:
            mass_matrix = jnp.asarray(dense_mass_matrix(seq, operators, 0, dirichlet=False))
            null_vector = jnp.ones((mass_matrix.shape[0],), dtype=jnp.float64)
            null_norm = jnp.sqrt(jnp.abs(null_vector @ (mass_matrix @ null_vector)))
            null_vector = null_vector / jnp.where(null_norm > 0, null_norm, 1.0)
            schur_null = null_vector[:core_size]
            schur_null_norm = jnp.linalg.norm(schur_null)
            schur_null = schur_null / jnp.where(schur_null_norm > 0, schur_null_norm, 1.0)
            schur_projector = jnp.eye(core_size, dtype=jnp.float64) - jnp.outer(schur_null, schur_null)
            schur_reg = _symmetrize(schur + jnp.outer(schur_null, schur_null))
            schur_inv = _symmetrize(jnp.linalg.inv(schur_reg))

        factors = K0TensorHodgePreconditionerFactors(
            core_size=core_size,
            acb=acb,
            abc=abc,
            bulk_inv=bulk_inv,
            schur_inv=schur_inv,
            schur_projector=schur_projector,
        )
        pair = eqx.tree_at(
            lambda boundary_pair: boundary_pair.dbc if dirichlet else boundary_pair.free,
            pair,
            factors,
            is_leaf=lambda x: x is None,
        )
    return pair


def _k0_tensor_hodge_available(operators: SequenceOperators) -> bool:
    pair = operators.k0_tensor_hodge_precond
    return pair is not None and pair.free is not None and pair.dbc is not None


def _apply_k0_tensor_hodge_preconditioner(
        operators: SequenceOperators, rhs: jnp.ndarray, *, dirichlet: bool) -> jnp.ndarray:
    pair = operators.k0_tensor_hodge_precond
    if pair is None:
        raise ValueError('Tensor Hodge preconditioner k=0 is not assembled')
    factors = select_boundary_data(pair, dirichlet, 'Tensor Hodge k=0')
    rhs_c = rhs[:factors.core_size]
    rhs_b = rhs[factors.core_size:]
    y = factors.bulk_inv @ rhs_b
    schur_rhs = rhs_c - factors.acb @ y
    if factors.schur_projector is not None:
        schur_rhs = factors.schur_projector @ schur_rhs
    z = factors.schur_inv @ schur_rhs
    if factors.schur_projector is not None:
        z = factors.schur_projector @ z
    x_b = y - factors.bulk_inv @ (factors.abc @ z)
    return jnp.concatenate([z, x_b])


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
        case 3:
            W_flat = (1 / geometry.jacobian_j) * seq.quad.w
            sp = assemble_scalar_tp(
                seq.d_basis_r_jk, seq.d_basis_t_jk, seq.d_basis_z_jk,
                seq.d_basis_r_jk, seq.d_basis_t_jk, seq.d_basis_z_jk,
                W_flat, quad_shape, seq.basis_3.shape[0],
                seq.basis_3.pr, seq.basis_3.pt, seq.basis_3.pz)
            sp = jsparse.BCSR.from_bcoo(sp)
        case _:
            raise ValueError("k must be 0, 1, 2 or 3")

    return sp


def update_mass_operator(seq, geometry, operators: Optional[SequenceOperators], k: int):
    """Return an operator bundle with the k-th mass operator updated."""
    del geometry  # geometry is already attached to seq and used inside assembly
    sp = _assemble_mass_block(seq, seq.geometry, k)
    jacobi_pair = build_mass_jacobi_pair(seq, sp, k)
    operators = _ensure_extraction_operators(seq, operators)
    mass_preconds = set_mass_jacobi_pair(operators.mass_preconds, k, jacobi_pair)
    if k in (0, 1) and mass_preconds is not None and mass_preconds.tensor is not None:
        mass_preconds = set_mass_tensor(
            mass_preconds,
            build_mass_tensor_preconditioner(
                seq,
                jnp.asarray(sp.todense()),
                k=k,
                rank=mass_preconds.tensor.rank,
                cp_kwargs={
                    "maxiter": mass_preconds.tensor.cp_maxiter,
                    "tol": mass_preconds.tensor.cp_tol,
                    "ridge": mass_preconds.tensor.cp_ridge,
                },
                existing=mass_preconds.tensor,
            ),
        )
    match k:
        case 0:
            return eqx.tree_at(
                lambda ops: (ops.m0_sp, ops.mass_preconds),
                operators,
                (sp, mass_preconds),
                is_leaf=lambda x: x is None,
            )
        case 1:
            return eqx.tree_at(
                lambda ops: (ops.m1_sp, ops.mass_preconds),
                operators,
                (sp, mass_preconds),
                is_leaf=lambda x: x is None,
            )
        case 2:
            return eqx.tree_at(
                lambda ops: (ops.m2_sp, ops.mass_preconds),
                operators,
                (sp, mass_preconds),
                is_leaf=lambda x: x is None,
            )
        case 3:
            return eqx.tree_at(
                lambda ops: (ops.m3_sp, ops.mass_preconds),
                operators,
                (sp, mass_preconds),
                is_leaf=lambda x: x is None,
            )
    raise ValueError("k must be 0, 1, 2 or 3")


def assemble_mass_operators(seq, geometry, operators: Optional[SequenceOperators] = None,
                            ks: Sequence[int] = (0, 1, 2, 3)):
    """Assemble mass operators for the requested form degrees."""
    for k in ks:
        operators = update_mass_operator(seq, geometry, operators, k)
    return operators


def assemble_kron_mass_preconditioner(
        seq, operators: Optional[SequenceOperators] = None):
    """Assemble the 1-D reference mass-matrix inverses on ``operators``.

    This is now a legacy helper for FD/Hodge paths and debug experiments.
    Production mass-preconditioner dispatch no longer routes through the
    Kronecker variant.
    """
    operators = _ensure_extraction_operators(seq, operators)
    mass_preconds = set_mass_kronecker(
        operators.mass_preconds,
        build_mass_kronecker_preconditioner(seq),
    )
    return eqx.tree_at(
        lambda ops: ops.mass_preconds,
        operators,
        mass_preconds,
        is_leaf=lambda x: x is None,
    )


def _kron_available(seq, operators: SequenceOperators, k: int) -> bool:
    return mass_kronecker_available(seq, operators.mass_preconds, k)


def assemble_tensor_mass_preconditioner(
        seq, operators: Optional[SequenceOperators] = None,
        *, ks: Sequence[int] = (1,),
        rank: int = 3,
        cp_kwargs: Optional[dict] = None):
    """Assemble the k=0/k=1/k=2/k=3 tensor mass preconditioner on ``operators``.

    The current production tensor path implements the extracted scalar
    core-plus-bulk Schur model for polar ``k=0`` and the surgery-plus-Schur
    model for polar ``k=1``. For polar ``k=2`` it uses an outer Schur split on
    the extracted ``r`` surgery block together with tensor-diagonal bulk block
    inverses. For polar ``k=3`` it implements a direct extracted scalar tensor
    inverse. All use low-rank CP fits of the diagonal metric factors to build
    tensor block inverse applies.
    """
    operators = _ensure_extraction_operators(seq, operators)
    missing_ks = []
    for k in ks:
        if k not in (0, 1, 2, 3):
            raise ValueError("Tensor mass preconditioner assembly only supports k=0, k=1, k=2 and k=3")
        if getattr(operators, f"m{k}_sp") is None:
            missing_ks.append(k)
    if missing_ks:
        operators = assemble_mass_operators(seq, seq.geometry, operators, ks=tuple(missing_ks))

    tensor_precond = operators.mass_preconds.tensor if operators.mass_preconds is not None else None
    for k in ks:
        full_matrix = jnp.asarray(getattr(operators, f"m{k}_sp").todense())
        tensor_precond = build_mass_tensor_preconditioner(
            seq,
            full_matrix,
            k=k,
            rank=rank,
            cp_kwargs=cp_kwargs,
            existing=tensor_precond,
        )
    mass_preconds = set_mass_tensor(operators.mass_preconds, tensor_precond)
    return eqx.tree_at(
        lambda ops: ops.mass_preconds,
        operators,
        mass_preconds,
        is_leaf=lambda x: x is None,
    )


def _tensor_available(seq, operators: SequenceOperators, k: int) -> bool:
    return mass_tensor_available(seq, operators.mass_preconds, k)


def apply_mass_tensor_preconditioner_ops(
        seq, operators: SequenceOperators, v, k: int, dirichlet: bool = True):
    return apply_mass_tensor_preconditioner(
        seq, operators.mass_preconds, v, k, dirichlet=dirichlet)


def apply_mass_kron_preconditioner(
        seq, operators: SequenceOperators, v, k: int, dirichlet: bool = True):
    """``E (M_r^{-1} ⊗ M_θ^{-1} ⊗ M_ζ^{-1}) E^T v`` (block-diagonal over components).

    SPD by construction; sandwiches the reference Kronecker inverse with the
    extraction operators so the same code path handles polar fusion and
    Dirichlet boundary clamping.  If any 1-D inverse is unavailable,
    :func:`apply_mass_matrix_preconditioner` falls back to Jacobi instead
    of calling this function.
    """
    return apply_mass_kronecker_preconditioner(
        seq, operators.mass_preconds, v, k, dirichlet=dirichlet)


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


def assemble_tensor_hodge_preconditioner(
        seq, operators: Optional[SequenceOperators] = None):
    """Precompute the tensor-Hodge auxiliary data for ``k = 0`` and ``k = 3``.

    This stores the 1-D eigendecompositions used by the tensorized reference
    Hodge inverse. Geometry-independent; call once after ``seq.evaluate_1d()``.
    After this, :func:`apply_hodge_laplacian_preconditioner` with
    ``kind='tensor'`` becomes available for ``k = 0`` and is also used by the
    ``k = 3`` tensor auxiliary-space path.
    """
    operators = _ensure_extraction_operators(seq, operators)
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


def assemble_fd_hodge_preconditioner(
        seq, operators: Optional[SequenceOperators] = None):
    """Legacy alias for :func:`assemble_tensor_hodge_preconditioner`."""
    return assemble_tensor_hodge_preconditioner(seq, operators=operators)


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
            "Tensor Hodge preconditioner not available for k="
            f"{k}; assemble it via assemble_tensor_hodge_preconditioner")
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
    try:
        diaginv = get_mass_jacobi_diaginv(operators.mass_preconds, k, False)
        diaginv_dbc = get_mass_jacobi_diaginv(operators.mass_preconds, k, True)
    except ValueError:
        diaginv = None
        diaginv_dbc = None
    match k:
        case 0:
            return operators.m0_sp, diaginv, diaginv_dbc
        case 1:
            return operators.m1_sp, diaginv, diaginv_dbc
        case 2:
            return operators.m2_sp, diaginv, diaginv_dbc
        case 3:
            return operators.m3_sp, diaginv, diaginv_dbc
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
    operators = _ensure_extraction_operators(seq, operators)
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
    operators = _ensure_extraction_operators(seq, operators)

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
    """Assemble topological incidence operators for the requested degrees."""
    for k in ks:
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
    e_in, e_in_T, e_out, e_out_T = _derivative_extraction(
        operators, k, dirichlet_in, dirichlet_out)

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
    operators = _ensure_extraction_operators(seq, operators)

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
            e, e_T = _mass_extraction(operators, 0, False)
            e_dbc, e_dbc_T = _mass_extraction(operators, 0, True)
            diaginv = 1.0 / diag_EAET_matvec(
                e, lambda v: K_apply(v), seq.n0, e_T)
            diaginv_dbc = 1.0 / diag_EAET_matvec(
                e_dbc, lambda v: K_apply(v), seq.n0_dbc, e_dbc_T)
        case 1:
            try:
                mass_diaginv = _mass_diaginv(operators, 0, dirichlet=False)
                mass_diaginv_dbc = _mass_diaginv(operators, 0, dirichlet=True)
            except ValueError as exc:
                raise ValueError(
                    "Assemble mass operator k=0 before Hodge operator k=1") from exc
            K_apply = _stiffness_matvec(1)
            DT_apply = _derivative_T_matvec(1)
            e_prev, _ = _mass_extraction(operators, 0, False)
            e, e_T = _mass_extraction(operators, 1, False)
            e_prev_dbc, _ = _mass_extraction(operators, 0, True)
            e_dbc, e_dbc_T = _mass_extraction(operators, 1, True)
            d_stiff = diag_EAET_matvec(
                e, lambda v: K_apply(v), seq.n1, e_T)
            d_schur = diag_schur_complement(
                lambda v: e_prev @ DT_apply(e_T @ v),
                mass_diaginv, seq.n1)
            diaginv = 1.0 / (d_stiff + d_schur)
            d_stiff_dbc = diag_EAET_matvec(
                e_dbc, lambda v: K_apply(v), seq.n1_dbc, e_dbc_T)
            d_schur_dbc = diag_schur_complement(
                lambda v: e_prev_dbc @ DT_apply(e_dbc_T @ v),
                mass_diaginv_dbc, seq.n1_dbc)
            diaginv_dbc = 1.0 / (d_stiff_dbc + d_schur_dbc)
        case 2:
            try:
                mass_diaginv = _mass_diaginv(operators, 1, dirichlet=False)
                mass_diaginv_dbc = _mass_diaginv(operators, 1, dirichlet=True)
            except ValueError as exc:
                raise ValueError(
                    "Assemble mass operator k=1 before Hodge operator k=2") from exc
            K_apply = _stiffness_matvec(2)
            DT_apply = _derivative_T_matvec(2)
            e_prev, _ = _mass_extraction(operators, 1, False)
            e, e_T = _mass_extraction(operators, 2, False)
            e_prev_dbc, _ = _mass_extraction(operators, 1, True)
            e_dbc, e_dbc_T = _mass_extraction(operators, 2, True)
            d_stiff = diag_EAET_matvec(
                e, lambda v: K_apply(v), seq.n2, e_T)
            d_schur = diag_schur_complement(
                lambda v: e_prev @ DT_apply(e_T @ v),
                mass_diaginv, seq.n2)
            diaginv = 1.0 / (d_stiff + d_schur)
            d_stiff_dbc = diag_EAET_matvec(
                e_dbc, lambda v: K_apply(v), seq.n2_dbc, e_dbc_T)
            d_schur_dbc = diag_schur_complement(
                lambda v: e_prev_dbc @ DT_apply(e_dbc_T @ v),
                mass_diaginv_dbc, seq.n2_dbc)
            diaginv_dbc = 1.0 / (d_stiff_dbc + d_schur_dbc)
        case 3:
            try:
                mass_diaginv = _mass_diaginv(operators, 2, dirichlet=False)
                mass_diaginv_dbc = _mass_diaginv(operators, 2, dirichlet=True)
            except ValueError as exc:
                raise ValueError(
                    "Assemble mass operator k=2 before Hodge operator k=3") from exc
            DT_apply = _derivative_T_matvec(3)
            e_prev, _ = _mass_extraction(operators, 2, False)
            _, e_T = _mass_extraction(operators, 3, False)
            e_prev_dbc, _ = _mass_extraction(operators, 2, True)
            _, e_dbc_T = _mass_extraction(operators, 3, True)
            diaginv = 1.0 / diag_schur_complement(
                lambda v: e_prev @ DT_apply(e_T @ v),
                mass_diaginv, seq.n3)
            diaginv_dbc = 1.0 / diag_schur_complement(
                lambda v: e_prev_dbc @ DT_apply(e_dbc_T @ v),
                mass_diaginv_dbc, seq.n3_dbc)
        case _:
            raise ValueError("k must be 0, 1, 2, or 3")

    return sp, diaginv, diaginv_dbc


def update_hodge_operator(seq, geometry, operators: Optional[SequenceOperators], k: int):
    """Return an operator bundle with the k-th Hodge/stiffness data updated."""
    operators = _ensure_extraction_operators(seq, operators)
    # Stiffness blocks for k=0,1,2 are built from the topological incidence
    # ``G_k`` and mass ``M_{k+1}``; make sure ``G_k`` is available.
    if k in (0, 1, 2) and _incidence_components(operators, k)[0] is None:
        operators = update_incidence_operator(seq, operators, k)
    sp, diaginv, diaginv_dbc = _assemble_hodge_block(
        seq, geometry, operators, k)

    match k:
        case 0:
            fd_scale_K = _fd_hodge_scales_K(seq, geometry, 0)
            rank, cp_maxiter, cp_tol, cp_ridge = _k0_tensor_hodge_config(operators)
            tensor_precond = _assemble_k0_tensor_hodge_preconditioner(
                seq,
                operators,
                rank=rank,
                cp_maxiter=cp_maxiter,
                cp_tol=cp_tol,
                cp_ridge=cp_ridge,
            )
            return eqx.tree_at(
                lambda ops: (ops.grad_grad_sp, ops.dd0_sp_diaginv,
                             ops.dd0_sp_diaginv_dbc, ops.dd0_fd_scale_K,
                             ops.k0_tensor_hodge_precond),
                operators,
                (sp, diaginv, diaginv_dbc, fd_scale_K, tensor_precond),
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
    operators = assemble_tensor_mass_preconditioner(seq, operators=operators, ks=(0, 1, 2, 3))
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

def _mass_extraction(operators: SequenceOperators, k: int, dirichlet: bool):
    match k:
        case 0:
            return (operators.e0_dbc, operators.e0_dbc_T) if dirichlet else (operators.e0, operators.e0_T)
        case 1:
            return (operators.e1_dbc, operators.e1_dbc_T) if dirichlet else (operators.e1, operators.e1_T)
        case 2:
            return (operators.e2_dbc, operators.e2_dbc_T) if dirichlet else (operators.e2, operators.e2_T)
        case 3:
            return (operators.e3_dbc, operators.e3_dbc_T) if dirichlet else (operators.e3, operators.e3_T)
    raise ValueError("k must be 0, 1, 2 or 3")


def _derivative_extraction(operators: SequenceOperators, k: int,
                           dirichlet_in: bool, dirichlet_out: bool):
    match k:
        case 0:
            e_in = operators.e0_dbc if dirichlet_in else operators.e0
            e_in_T = operators.e0_dbc_T if dirichlet_in else operators.e0_T
            e_out = operators.e1_dbc if dirichlet_out else operators.e1
            e_out_T = operators.e1_dbc_T if dirichlet_out else operators.e1_T
        case 1:
            e_in = operators.e1_dbc if dirichlet_in else operators.e1
            e_in_T = operators.e1_dbc_T if dirichlet_in else operators.e1_T
            e_out = operators.e2_dbc if dirichlet_out else operators.e2
            e_out_T = operators.e2_dbc_T if dirichlet_out else operators.e2_T
        case 2:
            e_in = operators.e2_dbc if dirichlet_in else operators.e2
            e_in_T = operators.e2_dbc_T if dirichlet_in else operators.e2_T
            e_out = operators.e3_dbc if dirichlet_out else operators.e3
            e_out_T = operators.e3_dbc_T if dirichlet_out else operators.e3_T
        case _:
            raise ValueError("k must be 0, 1 or 2")
    return e_in, e_in_T, e_out, e_out_T


def _projection_extraction(operators: SequenceOperators,
                           k_in: int, k_out: int,
                           dirichlet_in: bool, dirichlet_out: bool):
    match (k_in, k_out):
        case (2, 1):
            e_in = operators.e2_dbc if dirichlet_in else operators.e2
            e_in_T = operators.e2_dbc_T if dirichlet_in else operators.e2_T
            e_out = operators.e1_dbc if dirichlet_out else operators.e1
        case (1, 2):
            e_in = operators.e1_dbc if dirichlet_in else operators.e1
            e_in_T = operators.e1_dbc_T if dirichlet_in else operators.e1_T
            e_out = operators.e2_dbc if dirichlet_out else operators.e2
        case (0, 3):
            e_in = operators.e3_dbc if dirichlet_in else operators.e3
            e_in_T = operators.e3_dbc_T if dirichlet_in else operators.e3_T
            e_out = operators.e0_dbc if dirichlet_out else operators.e0
        case (3, 0):
            e_in = operators.e0_dbc if dirichlet_in else operators.e0
            e_in_T = operators.e0_dbc_T if dirichlet_in else operators.e0_T
            e_out = operators.e3_dbc if dirichlet_out else operators.e3
        case _:
            raise ValueError(
                "Only (k_in, k_out) = (1, 2), (2, 1), (0, 3), or (3, 0) supported"
            )
    return e_in, e_in_T, e_out


def dense_mass_matrix(seq, operators: SequenceOperators, k: int,
                      dirichlet: bool = True):
    """Return the dense extracted mass matrix for degree k."""
    sp, _, _ = _mass_components(operators, k)
    if sp is None:
        raise ValueError(f"Mass operator k={k} is not assembled")
    e, e_T = _mass_extraction(operators, k, dirichlet)
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
        operators, k, dirichlet_in, dirichlet_out)
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
    e, e_T = _mass_extraction(operators, k, dirichlet)
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

    e_in, e_in_T, e_out = _projection_extraction(
        operators, k_in, k_out, dirichlet_in, dirichlet_out)
    return e_out.todense() @ sp.todense() @ e_in_T.todense()


def apply_mass_matrix(seq, operators: SequenceOperators, v, k: int, dirichlet: bool = True):
    """Apply a mass matrix from an explicit operator bundle."""
    sp, _, _ = _mass_components(operators, k)
    if sp is None:
        raise ValueError(f"Mass operator k={k} is not assembled")

    e, e_T = _mass_extraction(operators, k, dirichlet)
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

    e_in, e_in_T, e_out = _projection_extraction(
        operators, k_in, k_out, dirichlet_in, dirichlet_out)

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

    e_in, e_in_T, e_out, e_out_T = _derivative_extraction(
        operators, k, dirichlet_in, dirichlet_out)

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
    kind : {'auto', 'jacobi', 'tensor'}
        Which preconditioner to use. ``'auto'`` picks ``'tensor'`` when the
        tensor mass preconditioner is assembled and available for this ``k``;
        otherwise it falls back to ``'jacobi'``.
    """
    if kind == 'auto':
        if _tensor_available(seq, operators, k):
            kind = 'tensor'
        else:
            kind = 'jacobi'
    if kind == 'tensor':
        if not _tensor_available(seq, operators, k):
            raise ValueError(
                f"Tensor mass preconditioner not assembled for k={k}; "
                "call assemble_tensor_mass_preconditioner(seq, operators, ...) first")
        return apply_mass_tensor_preconditioner_ops(
            seq, operators, v, k, dirichlet=dirichlet)
    if kind == 'jacobi':
        return _mass_diaginv(operators, k, dirichlet) * v
    raise ValueError(
        f"kind must be 'auto', 'jacobi' or 'tensor' (got {kind!r})")


def apply_inverse_mass_matrix(seq, operators: SequenceOperators, rhs, k: int,
                              dirichlet: bool = True, guess=None,
                              tol: Optional[float] = None,
                              maxiter: Optional[int] = None,
                              preconditioner='auto',
                              return_info: bool = False):
    """Solve with the inverse mass matrix from an explicit operator bundle.

    ``preconditioner`` accepts a kind string or a
    :class:`MassPreconditionerSpec`. When omitted, the default is tensor when
    assembled and Jacobi otherwise.
    """
    tol = seq.tol if tol is None else tol
    maxiter = seq.maxiter if maxiter is None else maxiter
    precond_apply = _build_mass_preconditioner_apply(
        seq,
        operators,
        k=k,
        dirichlet=dirichlet,
        preconditioner=preconditioner,
        allow_none=True,
    )
    x, info = solve_singular_cg(
        lambda x: apply_mass_matrix(seq, operators, x, k, dirichlet=dirichlet),
        rhs,
        mass_matvec=lambda x: apply_mass_matrix(
            seq, operators, x, k, dirichlet=dirichlet),
        precond_matvec=precond_apply,
        x0=guess,
        tol=tol,
        maxiter=maxiter,
    )
    return (x, info) if return_info else x


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

    e, e_T = _mass_extraction(operators, k, dirichlet)
    return e @ (g_sp_T @ (m_sp @ (g_sp @ (e_T @ v))))


def apply_hodge_k3_tensor_preconditioner(seq, operators: SequenceOperators, rhs,
                                         k: int, dirichlet: bool = True,
                                         eps: float = 0.0):
    """Tensorized auxiliary-space preconditioner for ``L_k + eps M_k``.

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
            "tensor auxiliary-space preconditioner currently only implemented for k=3")
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
    smooth = smooth_diaginv * rhs
    # Auxiliary-space correction: tilde(M_3)^{-1} M_{03}
    #   (tilde(L_0) + eps tilde(M_0))^{-1} M_{30} tilde(M_3)^{-1}.
    # Read right-to-left; each step is a single direct apply.
    w = apply_mass_kron_preconditioner(
        seq, operators, rhs, 3, dirichlet=dirichlet)
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


def _estimate_preconditioned_max_eigenvalue_apply(
        operator_apply, smoother_apply, size: int, *,
    n_iter: int = 10, seed: int = 0):
    vector = jax.random.normal(
        jax.random.PRNGKey(seed), (size,), dtype=jnp.float64)

    def operator_norm(x):
        ax = operator_apply(x)
        return jnp.sqrt(jnp.abs(jnp.vdot(x, ax).real))

    init_norm = operator_norm(vector)
    vector = vector / jnp.where(init_norm > 0, init_norm, 1.0)

    def body(_, state):
        current, _ = state
        image = smoother_apply(operator_apply(current))
        image_norm = operator_norm(image)
        safe_norm = jnp.where(image_norm > 0, image_norm, 1.0)
        updated = image / safe_norm
        rayleigh = jnp.real(
            jnp.vdot(updated, operator_apply(smoother_apply(operator_apply(updated)))))
        return updated, rayleigh

    _, rayleigh = jax.lax.fori_loop(
        0, n_iter, body, (vector, jnp.asarray(0.0, dtype=jnp.float64)))
    return jnp.maximum(rayleigh, jnp.asarray(0.0, dtype=jnp.float64))


def _build_chebyshev_apply_preconditioner(
        operator_apply, smoother_apply, *,
        steps: int, min_eig: float, max_eig: float):
    if steps < 1:
        raise ValueError("Chebyshev step count must be positive")
    tiny = jnp.asarray(jnp.finfo(jnp.float64).tiny, dtype=jnp.float64)
    max_eig = jnp.maximum(jnp.asarray(max_eig, dtype=jnp.float64), tiny)
    min_eig = jnp.clip(jnp.asarray(min_eig, dtype=jnp.float64), tiny, max_eig)

    d = 0.5 * (max_eig + min_eig)
    c = 0.5 * (max_eig - min_eig)

    def apply(rhs):
        alpha0 = jnp.asarray(1.0, dtype=rhs.dtype) / d.astype(rhs.dtype)

        def body(iteration, state):
            x, residual, direction, alpha = state
            correction = smoother_apply(residual)
            beta = (0.5 * c.astype(rhs.dtype) * alpha) ** 2
            new_alpha = jnp.where(
                iteration == 0,
                alpha,
                jnp.asarray(1.0, dtype=rhs.dtype)
                / (d.astype(rhs.dtype) - beta),
            )
            new_direction = jnp.where(
                iteration == 0,
                correction,
                correction + beta * direction,
            )
            x = x + new_alpha * new_direction
            residual = residual - new_alpha * operator_apply(new_direction)
            return x, residual, new_direction, new_alpha

        x, _, _, _ = jax.lax.fori_loop(
            0,
            steps,
            body,
            (jnp.zeros_like(rhs), rhs, jnp.zeros_like(rhs), alpha0),
        )
        return x

    return apply


def _build_richardson_apply_preconditioner(
        operator_apply, smoother_apply, *,
        steps: int, omega: float):
    if steps < 1:
        raise ValueError("Richardson step count must be positive")

    def apply(rhs):
        omega_array = jnp.asarray(omega, dtype=rhs.dtype)

        def body(_, state):
            x, residual = state
            correction = smoother_apply(residual)
            x = x + omega_array * correction
            residual = residual - omega_array * operator_apply(correction)
            return x, residual

        x, _ = jax.lax.fori_loop(
            0,
            steps,
            body,
            (jnp.zeros_like(rhs), rhs),
        )
        return x

    return apply


def _coerce_mass_preconditioner_spec(preconditioner):
    if preconditioner is None:
        return default_mass_preconditioner()
    if isinstance(preconditioner, MassPreconditionerSpec):
        return preconditioner
    if isinstance(preconditioner, str):
        return MassPreconditionerSpec(kind=preconditioner)
    raise TypeError(
        "mass preconditioner must be a kind string or MassPreconditionerSpec")


def _resolve_legacy_mass_preconditioner(seq, operators, k: int, preconditioner):
    if isinstance(preconditioner, str) and preconditioner == 'auto':
        if _tensor_available(seq, operators, k):
            return default_mass_preconditioner()
        return MassPreconditionerSpec(kind='jacobi')
    return _coerce_mass_preconditioner_spec(preconditioner)


def _build_operator_preconditioner_apply(
        seq, operators: SequenceOperators, *, k: int, dirichlet: bool,
        operator_apply, preconditioner, allow_none: bool = True):
    spec = _resolve_legacy_mass_preconditioner(seq, operators, k, preconditioner)
    valid_kinds = ('none', 'jacobi', 'richardson', 'chebyshev', 'tensor')
    if spec.kind not in valid_kinds:
        raise ValueError(
            "preconditioner kind must be one of "
            f"{valid_kinds} (got {spec.kind!r})")
    if spec.kind == 'none':
        if not allow_none:
            raise ValueError("this preconditioner slot does not allow kind='none'")
        return lambda x: x
    if spec.kind == 'jacobi':
        diaginv = _mass_diaginv(operators, k, dirichlet)
        return lambda x, diaginv=diaginv: diaginv * x
    if spec.kind == 'tensor':
        if not _tensor_available(seq, operators, k):
            raise ValueError(
                f"Tensor mass preconditioner not assembled for k={k}")
        return lambda x: apply_mass_tensor_preconditioner_ops(
            seq, operators, x, k, dirichlet=dirichlet)

    smoother_spec = spec.smoother
    if smoother_spec is None:
        smoother_spec = MassPreconditionerSpec(kind='jacobi')
    smoother_apply = _build_operator_preconditioner_apply(
        seq,
        operators,
        k=k,
        dirichlet=dirichlet,
        operator_apply=operator_apply,
        preconditioner=smoother_spec,
        allow_none=False,
    )
    suffix = '_dbc' if dirichlet else ''
    size = getattr(seq, f'n{k}{suffix}')
    max_eig = _estimate_preconditioned_max_eigenvalue_apply(
        operator_apply,
        smoother_apply,
        size,
        n_iter=spec.power_iterations,
        seed=1000 * k + int(dirichlet),
    )
    if spec.kind == 'richardson':
        omega = jnp.where(
            max_eig > 0.0,
            jnp.asarray(spec.damping_safety, dtype=jnp.float64) / max_eig,
            jnp.asarray(1.0, dtype=jnp.float64),
        )
        return _build_richardson_apply_preconditioner(
            operator_apply,
            smoother_apply,
            steps=spec.steps,
            omega=omega,
        )
    min_eig = jnp.where(
        max_eig > 0.0,
        jnp.asarray(spec.min_eig_fraction, dtype=jnp.float64) * max_eig,
        jnp.asarray(spec.min_eig_fraction, dtype=jnp.float64),
    )
    return _build_chebyshev_apply_preconditioner(
        operator_apply,
        smoother_apply,
        steps=spec.steps,
        min_eig=min_eig,
        max_eig=max_eig,
    )


def _build_mass_preconditioner_apply(
        seq, operators: SequenceOperators, *, k: int, dirichlet: bool,
        preconditioner, allow_none: bool = True):
    def operator_apply(x):
        return apply_mass_matrix(seq, operators, x, k, dirichlet=dirichlet)

    return _build_operator_preconditioner_apply(
        seq,
        operators,
        k=k,
        dirichlet=dirichlet,
        operator_apply=operator_apply,
        preconditioner=preconditioner,
        allow_none=allow_none,
    )


def _build_schur_operator_apply(
        seq, operators: SequenceOperators, *, k: int, dirichlet: bool,
        eps: float, inner_preconditioner_apply):
    def apply(x):
        d_t_x = apply_derivative_matrix(
            seq,
            operators,
            x,
            k - 1,
            dirichlet_in=dirichlet,
            dirichlet_out=dirichlet,
            transpose=True,
        )
        inner_d_t_x = inner_preconditioner_apply(d_t_x)
        schur = apply_derivative_matrix(
            seq,
            operators,
            inner_d_t_x,
            k - 1,
            dirichlet_in=dirichlet,
            dirichlet_out=dirichlet,
        )
        return apply_stiffness(seq, operators, x, k, dirichlet=dirichlet) \
            + eps * apply_mass_matrix(seq, operators, x, k, dirichlet=dirichlet) \
            + schur

    return apply


def _coerce_scalar_hodge_preconditioner(
        seq, operators: SequenceOperators, *, k: int, preconditioner):
    if preconditioner is None or preconditioner == 'auto':
        return _materialize_default_scalar_hodge_preconditioner(
            seq, operators, k=k)
    if isinstance(preconditioner, MassPreconditionerSpec):
        return preconditioner
    if isinstance(preconditioner, str):
        return preconditioner
    raise TypeError(
        'scalar Hodge preconditioner must be a kind string or '
        'MassPreconditionerSpec')


def _coerce_saddle_preconditioner_spec(
        seq, operators: SequenceOperators, *, k: int,
        preconditioner) -> SaddlePointPreconditionerSpec:
    if preconditioner is None or preconditioner == 'auto':
        return _materialize_default_saddle_preconditioner(
            seq, operators, k=k)
    if isinstance(preconditioner, SaddlePointPreconditionerSpec):
        if preconditioner.schur.outer.kind == 'tensor':
            raise ValueError(
                "schur.outer kind='tensor' is not supported; "
                "tensor saddle preconditioning is only valid for the lower "
                "mass block and schur.inner")
        return preconditioner
    if isinstance(preconditioner, str):
        if preconditioner == 'tensor' and k == 3:
            raise ValueError(
                "preconditioner='tensor' is not supported for saddle solves; "
                "tensor saddle preconditioning is only valid for the lower "
                "mass block and schur.inner")
        lower_kind = 'tensor' if (
            preconditioner != 'jacobi'
            and _tensor_available(seq, operators, k - 1)
        ) else 'jacobi'
        if preconditioner == 'tensor':
            raise ValueError(
                "schur.outer kind='tensor' is not supported; "
                "tensor saddle preconditioning is only valid for the lower "
                "mass block and schur.inner")
        lower = MassPreconditionerSpec(kind=lower_kind)
        return SaddlePointPreconditionerSpec(
            mass=lower,
            schur=SchurPreconditionerSpec(
                inner=lower,
                outer=MassPreconditionerSpec(kind=preconditioner),
            ),
        )
    raise TypeError(
        'saddle preconditioner must be a kind string or '
        'SaddlePointPreconditionerSpec')


def _materialize_default_mass_preconditioner(
        seq, operators: SequenceOperators, *, k: int):
    if _tensor_available(seq, operators, k):
        return default_mass_preconditioner()
    return MassPreconditionerSpec(kind='jacobi')


def _materialize_default_saddle_preconditioner(
        seq, operators: SequenceOperators, *, k: int,
        coupled_preconditioner: bool = False):
    lower = _materialize_default_mass_preconditioner(
        seq, operators, k=k - 1)
    return SaddlePointPreconditionerSpec(
        mass=lower,
        schur=SchurPreconditionerSpec(
            inner=lower,
            outer=MassPreconditionerSpec(kind='jacobi'),
        ),
        coupled=coupled_preconditioner,
    )


def _materialize_default_scalar_hodge_preconditioner(
        seq, operators: SequenceOperators, *, k: int):
    if k == 0 and _k0_tensor_hodge_available(operators):
        return MassPreconditionerSpec(kind='tensor')
    return MassPreconditionerSpec(kind='jacobi')


def _build_scalar_hodge_preconditioner_apply(
        seq, operators: SequenceOperators, *, k: int, dirichlet: bool,
        eps: float, preconditioner, allow_none: bool = True):
    spec = _coerce_mass_preconditioner_spec(preconditioner)
    valid_kinds = ('none', 'jacobi', 'richardson', 'chebyshev', 'tensor')
    if spec.kind not in valid_kinds:
        raise ValueError(
            "preconditioner kind must be one of "
            f"{valid_kinds} (got {spec.kind!r})")
    if spec.kind == 'none':
        if not allow_none:
            raise ValueError("this preconditioner slot does not allow kind='none'")
        return lambda x: x
    if spec.kind == 'jacobi':
        stiffness_diaginv = _hodge_diaginv(operators, k, dirichlet)
        if eps == 0.0:
            shifted_diaginv = stiffness_diaginv
        else:
            mass_diaginv_k = _mass_diaginv(operators, k, dirichlet)
            shifted_diaginv = 1.0 / (1.0 / stiffness_diaginv + eps / mass_diaginv_k)
        return lambda x, diaginv=shifted_diaginv: diaginv * x
    if spec.kind == 'tensor':
        if k != 0:
            raise ValueError(
                f"Tensor Hodge preconditioner is only implemented for k=0 (got k={k})")
        if not _k0_tensor_hodge_available(operators):
            raise ValueError(
                f"Tensor Hodge preconditioner not assembled for k={k}")
        tensor_apply = lambda x: _apply_k0_tensor_hodge_preconditioner(
            operators, x, dirichlet=dirichlet)
        if eps > 0.0 and not dirichlet:
            # Without a valid coarse vector, the pure tensor Hodge inverse is
            # still singular on the harmonic mode. For shifted solves, fall
            # back to the regular shifted Jacobi apply until a real coarse
            # vector is available.
            jacobi_apply = _build_scalar_hodge_preconditioner_apply(
                seq,
                operators,
                k=k,
                dirichlet=dirichlet,
                eps=eps,
                preconditioner=MassPreconditionerSpec(kind='jacobi'),
                allow_none=False,
            )
            coarse_ready = _shifted_harmonic_coarse_ready(
                seq, operators, k, dirichlet)
            return lambda x: jax.lax.cond(
                coarse_ready,
                tensor_apply,
                jacobi_apply,
                x,
            )
        return tensor_apply
    def operator_apply(x):
        return apply_stiffness(seq, operators, x, k, dirichlet=dirichlet) \
            + eps * apply_mass_matrix(seq, operators, x, k, dirichlet=dirichlet)

    smoother_spec = spec.smoother
    if smoother_spec is None:
        smoother_spec = MassPreconditionerSpec(kind='jacobi')
    smoother_apply = _build_scalar_hodge_preconditioner_apply(
        seq,
        operators,
        k=k,
        dirichlet=dirichlet,
        eps=eps,
        preconditioner=smoother_spec,
        allow_none=False,
    )
    suffix = '_dbc' if dirichlet else ''
    size = getattr(seq, f'n{k}{suffix}')
    max_eig = _estimate_preconditioned_max_eigenvalue_apply(
        operator_apply,
        smoother_apply,
        size,
        n_iter=spec.power_iterations,
        seed=100 * k + int(dirichlet),
    )
    if spec.kind == 'richardson':
        omega = jnp.where(
            max_eig > 0.0,
            jnp.asarray(spec.damping_safety, dtype=jnp.float64) / max_eig,
            jnp.asarray(1.0, dtype=jnp.float64),
        )
        return _build_richardson_apply_preconditioner(
            operator_apply,
            smoother_apply,
            steps=spec.steps,
            omega=omega,
        )
    min_eig = jnp.where(
        max_eig > 0.0,
        jnp.asarray(spec.min_eig_fraction, dtype=jnp.float64) * max_eig,
        jnp.asarray(spec.min_eig_fraction, dtype=jnp.float64),
    )
    return _build_chebyshev_apply_preconditioner(
        operator_apply,
        smoother_apply,
        steps=spec.steps,
        min_eig=min_eig,
        max_eig=max_eig,
    )


def _build_shifted_chebyshev_hodge_preconditioner(
        seq, operators: SequenceOperators, *, k: int, dirichlet: bool,
        eps: float, steps: int, power_iterations: int,
        min_eig_fraction: float):
    suffix = "_dbc" if dirichlet else ""
    size = getattr(seq, f"n{k}{suffix}")
    stiffness_diaginv = _hodge_diaginv(operators, k, dirichlet)
    if eps == 0.0:
        shifted_diaginv = stiffness_diaginv
    else:
        mass_diaginv_k = _mass_diaginv(operators, k, dirichlet)
        shifted_diaginv = 1.0 / (1.0 / stiffness_diaginv + eps / mass_diaginv_k)

    def operator_apply(x):
        return apply_hodge_laplacian_approx(
            seq, operators, x, k, dirichlet=dirichlet) + eps * apply_mass_matrix(
                seq, operators, x, k, dirichlet=dirichlet)

    def smoother_apply(x):
        return shifted_diaginv * x

    max_eig = _estimate_preconditioned_max_eigenvalue_apply(
        operator_apply,
        smoother_apply,
        size,
        n_iter=power_iterations,
        seed=100 * k + int(dirichlet),
    )
    min_eig = jnp.where(
        max_eig > 0.0,
        jnp.asarray(min_eig_fraction, dtype=jnp.float64) * max_eig,
        jnp.asarray(min_eig_fraction, dtype=jnp.float64),
    )
    return _build_chebyshev_apply_preconditioner(
        operator_apply,
        smoother_apply,
        steps=steps,
        min_eig=min_eig,
        max_eig=max_eig,
    )


def _build_coupled_saddle_preconditioner(
        seq, operators: SequenceOperators, *, k: int, dirichlet: bool,
        upper_preconditioner, lower_preconditioner):
    suffix = "_dbc" if dirichlet else ""
    n_upper = getattr(seq, f"n{k}{suffix}")

    def apply(x):
        u = x[:n_upper]
        s = x[n_upper:]
        m_inv_s = lower_preconditioner(s)
        w_u = u + apply_derivative_matrix(
            seq, operators, m_inv_s, k - 1,
            dirichlet_in=dirichlet, dirichlet_out=dirichlet)
        y_u = upper_preconditioner(w_u)
        d_t_y_u = apply_derivative_matrix(
            seq, operators, y_u, k - 1,
            dirichlet_in=dirichlet, dirichlet_out=dirichlet, transpose=True)
        z_s = m_inv_s + lower_preconditioner(d_t_y_u)
        return jnp.concatenate([y_u, z_s])

    return apply


def apply_hodge_laplacian_preconditioner(seq, operators: SequenceOperators, v, k: int,
                                         dirichlet: bool = True,
                                         kind: str = 'auto'):
    """Apply the Hodge-Laplacian preconditioner from an operator bundle.

    ``kind`` options:

    * ``'none'`` — identity (no preconditioning).
    * ``'jacobi'`` — per-DoF diagonal of ``L_k``; always available.
    * ``'tensor'`` — tensorized auxiliary-space preconditioner. For ``k = 0``
      this reduces to the tensorized reference Laplacian inverse (no
      auxiliary space needed); for ``k = 3`` it uses the ``V_3 ↔ V_0``
      duality round trip and :func:`apply_hodge_k3_tensor_preconditioner`.
      Not available for ``k = 1, 2``.
    * ``'auto'`` — picks ``'tensor'`` when available (``k = 0`` or ``k = 3``
      with tensor Hodge data assembled) and falls back to ``'jacobi'``.
    """
    if kind not in ('auto', 'none', 'jacobi', 'tensor'):
        raise ValueError(
            f"kind must be 'auto', 'none', 'jacobi' or 'tensor' (got {kind!r})")
    if kind == 'auto':
        kind = 'tensor' if (
            k in (0, 3) and _fd_hodge_available(operators, k if k == 0 else 0)
        ) else 'jacobi'
    if kind == 'none':
        return v
    if kind == 'jacobi':
        return _hodge_diaginv(operators, k, dirichlet) * v
    if kind == 'tensor':
        if k == 0:
            if not _fd_hodge_available(operators, 0):
                raise ValueError(
                    "Tensor Hodge preconditioner for k=0 requires the tensor Hodge "
                    "assembly; call assemble_tensor_hodge_preconditioner first")
            return apply_hodge_kron_preconditioner(
                seq, operators, v, 0, dirichlet=dirichlet)
        if k == 3:
            if not _fd_hodge_available(operators, 0):
                raise ValueError(
                    "Tensor Hodge preconditioner for k=3 requires the tensor Hodge "
                    "assembly; call assemble_tensor_hodge_preconditioner first")
            return apply_hodge_k3_tensor_preconditioner(
                seq, operators, v, 3, dirichlet=dirichlet)
        raise ValueError(
            f"Tensor Hodge preconditioner not available for k={k}; use 'jacobi' instead")
    raise AssertionError("unreachable")


def apply_inverse_hodge_laplacian(seq, operators: SequenceOperators, rhs, k: int,
                                  dirichlet: bool = True, guess=None,
                                  tol: Optional[float] = None,
                                  maxiter: Optional[int] = None,
                                  preconditioner='auto',
                                  return_info: bool = False):
    """Solve with the inverse of the unshifted Hodge Laplacian ``L_k``.

    For ``k = 0`` this uses the dedicated singular scalar-Laplacian solve
    directly rather than routing through the shifted ``eps = 0`` path.
    For ``k >= 1`` the saddle-point implementation remains shared with the
    shifted solve because the only difference is the absent mass shift.
    """
    operators = _ensure_extraction_operators(seq, operators)
    tol = seq.tol if tol is None else tol
    maxiter = seq.maxiter if maxiter is None else maxiter

    if k == 0:
        selected_preconditioner = _coerce_scalar_hodge_preconditioner(
            seq, operators, k=k, preconditioner=preconditioner)

        precond_upper = _build_scalar_hodge_preconditioner_apply(
            seq,
            operators,
            k=k,
            dirichlet=dirichlet,
            eps=0.0,
            preconditioner=selected_preconditioner,
            allow_none=True,
        )

        vs = _nullspace_vectors(operators, 0, dirichlet)
        u, info = solve_singular_cg(
            lambda x: apply_stiffness(
                seq, operators, x, 0, dirichlet=dirichlet),
            rhs,
            mass_matvec=lambda x: apply_mass_matrix(
                seq, operators, x, 0, dirichlet=dirichlet),
            precond_matvec=precond_upper,
            x0=guess,
            vs=vs,
            tol=tol,
            maxiter=maxiter,
        )
        return (u, info) if return_info else u

    return apply_inverse_shifted_hodge_laplacian(
        seq,
        operators,
        rhs,
        k,
        0.0,
        dirichlet=dirichlet,
        guess=guess,
        tol=tol,
        maxiter=maxiter,
        preconditioner=preconditioner,
        use_harmonic_coarse=None,
        return_info=return_info,
    )


def apply_inverse_shifted_hodge_laplacian(seq, operators: SequenceOperators, rhs, k: int,
                                          eps: float, dirichlet: bool = True, guess=None,
                                          tol: Optional[float] = None,
                                          maxiter: Optional[int] = None,
                                          preconditioner='auto',
                                          use_harmonic_coarse: Optional[bool] = None,
                                          return_info: bool = False):
    """Solve with the inverse of the shifted Hodge Laplacian ``L_k + eps M_k``.

    For ``k >= 1`` the interface is ``preconditioner``, a structured
    saddle-point preconditioner spec with a lower mass block, a Schur-inner
    mass inverse, a Schur-outer preconditioner, and an optional coupled
    completion. Kind strings are accepted as convenience shorthands.
    """
    operators = _ensure_extraction_operators(seq, operators)
    tol = seq.tol if tol is None else tol
    maxiter = seq.maxiter if maxiter is None else maxiter

    if k == 0:
        if use_harmonic_coarse is None:
            use_harmonic_coarse = eps > 0 and not dirichlet

        selected_preconditioner = _coerce_scalar_hodge_preconditioner(
            seq, operators, k=k, preconditioner=preconditioner)

        precond_upper = _build_scalar_hodge_preconditioner_apply(
            seq,
            operators,
            k=k,
            dirichlet=dirichlet,
            eps=eps,
            preconditioner=selected_preconditioner,
            allow_none=True,
        )

        if use_harmonic_coarse:
            precond_upper = _wrap_shifted_harmonic_coarse_correction(
                seq, operators, precond_upper, eps, k, dirichlet)

        vs = _nullspace_vectors(
            operators, 0, dirichlet) if eps == 0 else jnp.zeros((0, rhs.shape[0]))
        u, info = solve_singular_cg(
            lambda x: apply_stiffness(
                seq, operators, x, 0, dirichlet=dirichlet)
            + eps * apply_mass_matrix(seq, operators,
                                      x, 0, dirichlet=dirichlet),
            rhs,
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
            jnp.zeros((0, rhs.shape[0])), jnp.zeros((0, 0)))
    suffix = "_dbc" if dirichlet else ""
    n_upper = getattr(seq, f"n{k}{suffix}")
    n_lower = getattr(seq, f"n{k-1}{suffix}")
    saddle_preconditioner = _coerce_saddle_preconditioner_spec(
        seq, operators, k=k, preconditioner=preconditioner)

    if saddle_preconditioner.schur.inner.kind == 'none':
        raise ValueError("schur.inner cannot use kind='none'")
    if saddle_preconditioner.schur.outer.kind == 'tensor':
        raise ValueError(
            "schur.outer kind='tensor' is not supported; "
            "tensor saddle preconditioning is only valid for the lower "
            "mass block and schur.inner")

    precond_lower = _build_mass_preconditioner_apply(
        seq,
        operators,
        k=k - 1,
        dirichlet=dirichlet,
        preconditioner=saddle_preconditioner.mass,
        allow_none=True,
    )
    schur_inner = _build_mass_preconditioner_apply(
        seq,
        operators,
        k=k - 1,
        dirichlet=dirichlet,
        preconditioner=saddle_preconditioner.schur.inner,
        allow_none=False,
    )
    schur_apply = _build_schur_operator_apply(
        seq,
        operators,
        k=k,
        dirichlet=dirichlet,
        eps=eps,
        inner_preconditioner_apply=schur_inner,
    )
    precond_upper = _build_operator_preconditioner_apply(
        seq,
        operators,
        k=k,
        dirichlet=dirichlet,
        operator_apply=schur_apply,
        preconditioner=saddle_preconditioner.schur.outer,
        allow_none=True,
    )
    precond_matvec = (
        _build_coupled_saddle_preconditioner(
            seq,
            operators,
            k=k,
            dirichlet=dirichlet,
            upper_preconditioner=precond_upper,
            lower_preconditioner=precond_lower,
        )
        if saddle_preconditioner.coupled
        else None
    )

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
        b_upper=rhs,
        n_upper=n_upper,
        n_lower=n_lower,
        precond_matvec=precond_matvec,
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


def apply_inverse_mass_plus_eps_laplace_matrix(seq, operators: SequenceOperators, rhs, k: int,
                                               eps: float, dirichlet: bool = True, guess=None,
                                               tol: Optional[float] = None,
                                               maxiter: Optional[int] = None,
                                               preconditioner='auto',
                                               return_info: bool = False):
    """Solve with the inverse of M_k + eps L_k using an explicit operator bundle.

    Out-of-the-box diffusion preconditioners currently use the same basic
    mass-side building blocks as the other solver paths: Jacobi, tensor, and
    polynomial variants such as Chebyshev.

    TODO: add the second-order small-eps correction
    ``M^{-1} - eps M^{-1} L M^{-1}`` as an explicit diffusion
    preconditioner option once we want a stronger asymptotic path.
    """
    tol = seq.tol if tol is None else tol
    maxiter = seq.maxiter if maxiter is None else maxiter

    if eps < 0.0:
        raise ValueError("eps must be nonnegative")

    if eps == 0.0:
        return apply_inverse_mass_matrix(
            seq,
            operators,
            rhs,
            k,
            dirichlet=dirichlet,
            guess=guess,
            tol=tol,
            maxiter=maxiter,
            preconditioner=preconditioner,
            return_info=return_info,
        )

    if k == 0:
        def operator_apply(x):
            return apply_mass_matrix(
                seq, operators, x, 0, dirichlet=dirichlet) + eps * apply_stiffness(
                    seq, operators, x, 0, dirichlet=dirichlet)

        precond_apply = _build_operator_preconditioner_apply(
            seq,
            operators,
            k=0,
            dirichlet=dirichlet,
            operator_apply=operator_apply,
            preconditioner=preconditioner,
            allow_none=True,
        )
        x, info = solve_singular_cg(
            operator_apply,
            rhs,
            precond_matvec=precond_apply,
            x0=guess,
            tol=tol,
            maxiter=maxiter,
        )
        return (x, info) if return_info else x

    suffix = "_dbc" if dirichlet else ""
    n_upper = getattr(seq, f"n{k}{suffix}")
    n_lower = getattr(seq, f"n{k-1}{suffix}")

    def upper_operator_apply(x):
        return apply_mass_matrix(
            seq, operators, x, k, dirichlet=dirichlet) + eps * apply_stiffness(
                seq, operators, x, k, dirichlet=dirichlet)

    upper_preconditioner = _build_operator_preconditioner_apply(
        seq,
        operators,
        k=k,
        dirichlet=dirichlet,
        operator_apply=upper_operator_apply,
        preconditioner=preconditioner,
        allow_none=True,
    )
    lower_preconditioner_apply = _build_mass_preconditioner_apply(
        seq,
        operators,
        k=k - 1,
        dirichlet=dirichlet,
        preconditioner=preconditioner,
        allow_none=True,
    )

    def precond_lower(x):
        return (1.0 / eps) * lower_preconditioner_apply(x)

    def precond_upper(x):
        return upper_preconditioner(x)

    u, sigma, info = solve_saddle_point_minres(
        stiffness_matvec=upper_operator_apply,
        derivative_matvec=lambda s: eps * apply_derivative_matrix(
            seq, operators, s, k - 1, dirichlet_in=dirichlet, dirichlet_out=dirichlet),
        derivative_T_matvec=lambda u: eps * apply_derivative_matrix(
            seq, operators, u, k - 1, dirichlet_in=dirichlet,
            dirichlet_out=dirichlet, transpose=True),
        mass_lower_matvec=lambda s: eps * apply_mass_matrix(
            seq, operators, s, k - 1, dirichlet=dirichlet),
        b_upper=rhs,
        n_upper=n_upper,
        n_lower=n_lower,
        precond_upper=precond_upper,
        precond_lower=precond_lower,
        x0_upper=guess,
        tol=tol,
        maxiter=maxiter,
    )
    return (u, info) if return_info else u




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
    apply of the configured mass preconditioner. The result is a fully linear SPD
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
            Minv_Dt_v = apply_mass_matrix_preconditioner(
                seq, operators, Dt_v, 0, dirichlet=dirichlet, kind='auto')
            return apply_stiffness(seq, operators, v, 1, dirichlet=dirichlet) + \
                apply_derivative_matrix(
                    seq, operators, Minv_Dt_v, 0,
                    dirichlet_in=dirichlet, dirichlet_out=dirichlet)
        case 2:
            Dt_v = apply_derivative_matrix(
                seq, operators, v, 1,
                dirichlet_in=dirichlet, dirichlet_out=dirichlet, transpose=True)
            Minv_Dt_v = apply_mass_matrix_preconditioner(
                seq, operators, Dt_v, 1, dirichlet=dirichlet, kind='auto')
            return apply_stiffness(seq, operators, v, 2, dirichlet=dirichlet) + \
                apply_derivative_matrix(
                    seq, operators, Minv_Dt_v, 1,
                    dirichlet_in=dirichlet, dirichlet_out=dirichlet)
        case 3:
            Dt_v = apply_derivative_matrix(
                seq, operators, v, 2,
                dirichlet_in=dirichlet, dirichlet_out=dirichlet, transpose=True)
            Minv_Dt_v = apply_mass_matrix_preconditioner(
                seq, operators, Dt_v, 2, dirichlet=dirichlet, kind='auto')
            return apply_derivative_matrix(
                seq, operators, Minv_Dt_v, 2,
                dirichlet_in=dirichlet, dirichlet_out=dirichlet)
        case _:
            raise ValueError("k must be 0, 1, 2 or 3")
