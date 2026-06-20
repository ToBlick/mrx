# %% [markdown]
# # k=0 Tensor Hodge Preconditioner Debug
#
# Dense diagnostic script for the production `k=0` scalar Hodge/Laplacian
# tensor preconditioner.
#
# The goal is to inspect the exact object used by the scalar Hodge solver:
#
# 1. assemble the dense extracted stiffness matrix `K`,
# 2. assemble the production tensor Hodge preconditioner,
# 3. contrast it against the production tensor mass preconditioners for `M0`
#    and `M1`,
# 4. compare the dense apply matrices against the exact inverse /
#    pseudoinverse,
# 5. inspect the exact vs. modeled bulk and Schur blocks,
# 6. and visualize the main discrepancies with `imshow` plots.

# %%
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np

from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import rotating_ellipse_map, toroid_map
from mrx.operators import (
    _apply_k0_tensor_hodge_bulk_inverse,
    _apply_k0_tensor_hodge_bulk_to_surgery_coupling,
    _apply_k0_tensor_hodge_surgery_to_bulk_coupling,
    _build_k0_tensor_hodge_bulk_model,
    _k0_tensor_hodge_config,
    apply_mass_tensor_preconditioner_ops,
    apply_laplacian_preconditioner,
    assemble_laplacian_operators,
    assemble_incidence_operators,
    assemble_mass_operators,
    assemble_tensor_hodge_preconditioner,
    assemble_tensor_mass_preconditioner,
    dense_mass_matrix,
    dense_laplacian,
)
from mrx.preconditioners import (
    _apply_bulk_to_surgery_coupling,
    _apply_surgery_to_bulk_coupling,
    _apply_k1_rt_art_coupling,
    _apply_k1_rt_atr_coupling,
    _apply_tensor_diagonal_block,
    _arr_shape_k1,
    _core_size,
    _k1_diagonal_metric_tensors,
    _select_mass_surgery_factors,
    _select_mass_tensor_factors,
    _theta_bulk_shape_k1,
    select_boundary_data,
    _split_blocks,
    _zeta_bulk_shape_k1,
)

jax.config.update("jax_enable_x64", True)


# %% Configuration
@dataclass(frozen=True)
class Config:
    ns: tuple[int, int, int] = (6, 8, 6)
    p: int = 3
    tol: float = 1e-9
    maxiter: int = 1000
    betti: tuple[int, int, int, int] = (1, 1, 0, 0)
    map_kind: str = "rotating_ellipse"
    torus_epsilon: float = 1.0 / 3.0
    torus_r0: float = 1.0
    rotating_eps: float = 0.33
    rotating_kappa: float = 1.2
    rotating_r0: float = 1.0
    rotating_nfp: int = 3
    dirichlet: bool = True
    tensor_rank: int = 3
    cp_maxiter: int = 100
    cp_tol: float = 1e-9
    cp_ridge: float = 1e-12
    richardson_steps: int = 3
    richardson_omega: float = 1.0
    spectral_rtol: float = 1e-10
    n_random_rhs: int = 8
    seed: int = 0


CONFIG = Config()
SEQ = None
OPERATORS = None
K_EXACT = None
K_PINV = None
RANGE_PROJECTOR = None
P_TENSOR = None
BLOCKS = None
MASS_DIAGNOSTICS = None
HODGE_DIAGNOSTICS = None
RANDOM_RHS = None
RANDOM_EXACT = None
RANDOM_TENSOR = None
RANDOM_REL_ERRORS = None
RANDOM_REL_RESIDUALS = None


# %% Helpers
def _build_map(config: Config = CONFIG):
    if config.map_kind == "toroidal":
        return toroid_map(epsilon=config.torus_epsilon, R0=config.torus_r0)
    if config.map_kind == "rotating_ellipse":
        return rotating_ellipse_map(
            eps=config.rotating_eps,
            kappa=config.rotating_kappa,
            R0=config.rotating_r0,
            nfp=config.rotating_nfp,
        )
    raise ValueError(f"Unsupported map kind: {config.map_kind}")


def _build_case(config: Config = CONFIG):
    seq = DeRhamSequence(
        config.ns,
        (config.p, config.p, config.p),
        2 * config.p,
        ("clamped", "periodic", "periodic"),
        polar=True,
        tol=config.tol,
        maxiter=config.maxiter,
        betti_numbers=config.betti,
    )
    seq.evaluate_1d()
    seq.assemble_reference_mass_matrix()
    seq.set_map(_build_map(config))

    operators = assemble_mass_operators(seq, seq.geometry, ks=(0, 1))
    operators = assemble_tensor_mass_preconditioner(
        seq,
        operators=operators,
        ks=(0, 1),
        rank=config.tensor_rank,
        cp_kwargs={
            "tol": config.cp_tol,
            "maxiter": config.cp_maxiter,
            "ridge": config.cp_ridge,
            "richardson_steps": config.richardson_steps,
            "richardson_omega": config.richardson_omega,
        },
    )
    operators = assemble_tensor_hodge_preconditioner(seq, operators=operators)
    operators = assemble_incidence_operators(seq, operators=operators, ks=(0,))
    operators = assemble_laplacian_operators(seq, seq.geometry, operators=operators, ks=(0,))
    operators = seq.set_operators(operators, sync_legacy=False)
    if not config.dirichlet:
        seq._compute_nullspaces(config.betti, eps=config.tol**0.5)
        operators = seq._require_operators()
    return seq, operators


def _matrix_from_apply(apply, size: int) -> jnp.ndarray:
    basis = jnp.eye(size, dtype=jnp.float64)
    return jax.vmap(apply, in_axes=1, out_axes=1)(basis)


def _spectral_pseudoinverse(matrix: jnp.ndarray, *, rtol: float):
    eigvals, eigvecs = jnp.linalg.eigh(matrix)
    scale = jnp.max(jnp.abs(eigvals))
    threshold = rtol * jnp.where(scale > 0, scale, 1.0)
    mask = jnp.abs(eigvals) > threshold
    inv_eigvals = jnp.where(mask, 1.0 / eigvals, 0.0)
    projector = (eigvecs * mask.astype(jnp.float64)) @ eigvecs.T
    pinv = (eigvecs * inv_eigvals) @ eigvecs.T
    return pinv, projector, eigvals, float(threshold)


def _relative_fro_error(approx: jnp.ndarray, exact: jnp.ndarray) -> float:
    denom = jnp.linalg.norm(exact)
    denom = jnp.where(denom > 0, denom, 1.0)
    return float(jnp.linalg.norm(approx - exact) / denom)


def _random_rhs_batch(config: Config, size: int, projector: jnp.ndarray) -> jnp.ndarray:
    rhs = jax.random.normal(
        jax.random.PRNGKey(config.seed),
        (config.n_random_rhs, size),
        dtype=jnp.float64,
    )
    return jax.vmap(lambda x: projector @ x)(rhs)


def _preconditioner_matrix(seq, operators, config: Config, size: int) -> jnp.ndarray:
    return _matrix_from_apply(
        lambda x: apply_laplacian_preconditioner(
            seq,
            operators,
            x,
            0,
            dirichlet=config.dirichlet,
            kind="tensor",
        ),
        size,
    )


def _mass_preconditioner_matrix(seq, operators, config: Config, k: int, size: int) -> jnp.ndarray:
    return _matrix_from_apply(
        lambda x: apply_mass_tensor_preconditioner_ops(
            seq,
            operators,
            x,
            k,
            dirichlet=config.dirichlet,
        ),
        size,
    )


def _k0_bulk_inverse_matrix(operators, config: Config, size: int) -> jnp.ndarray:
    bulk_factors = _select_mass_tensor_factors(
        operators.mass_preconds,
        0,
        config.dirichlet,
    ).bulk
    return _matrix_from_apply(
        lambda x: _apply_tensor_diagonal_block(bulk_factors, x),
        size,
    )


def _k1_bulk_inverse_matrix(operators, config: Config) -> jnp.ndarray:
    surgery = _select_mass_surgery_factors(
        operators.mass_preconds,
        1,
        config.dirichlet,
    )
    factors = _select_mass_tensor_factors(
        operators.mass_preconds,
        1,
        config.dirichlet,
    )

    def rt_apply(rhs_rt):
        rhs_r = rhs_rt[:factors.rt_r_size]
        rhs_theta = rhs_rt[factors.rt_r_size:factors.rt_r_size + factors.rt_theta_size]
        y = _apply_tensor_diagonal_block(factors.arr, rhs_r)
        z = _apply_tensor_diagonal_block(factors.theta, rhs_theta - _apply_k1_rt_atr_coupling(surgery, y))
        x_r = y - _apply_tensor_diagonal_block(factors.arr, _apply_k1_rt_art_coupling(surgery, z))
        return jnp.concatenate([x_r, z])

    def bulk_apply(rhs_bulk):
        rhs_rt = rhs_bulk[:surgery.bulk_rt_size]
        rhs_zeta = rhs_bulk[surgery.bulk_rt_size:surgery.bulk_rt_size + surgery.bulk_zeta_size]
        return jnp.concatenate([
            rt_apply(rhs_rt),
            _apply_tensor_diagonal_block(factors.zeta, rhs_zeta),
        ])

    return _matrix_from_apply(bulk_apply, surgery.bulk_rt_size + surgery.bulk_zeta_size)


def _build_k1_bulk_model(operators, config: Config):
    surgery = _select_mass_surgery_factors(
        operators.mass_preconds,
        1,
        config.dirichlet,
    )
    factors = _select_mass_tensor_factors(
        operators.mass_preconds,
        1,
        config.dirichlet,
    )

    arr_inv = _matrix_from_apply(
        lambda x: _apply_tensor_diagonal_block(factors.arr, x),
        surgery.rt_r_size,
    )
    theta_inv = _matrix_from_apply(
        lambda x: _apply_tensor_diagonal_block(factors.theta, x),
        surgery.rt_theta_size,
    )
    zeta_inv = _matrix_from_apply(
        lambda x: _apply_tensor_diagonal_block(factors.zeta, x),
        surgery.bulk_zeta_size,
    )
    rt_art = _matrix_from_apply(
        lambda x: _apply_k1_rt_art_coupling(surgery, x),
        surgery.rt_theta_size,
    )
    rt_atr = _matrix_from_apply(
        lambda x: _apply_k1_rt_atr_coupling(surgery, x),
        surgery.rt_r_size,
    )

    arr_model = jnp.linalg.inv(arr_inv)
    theta_model = jnp.linalg.inv(theta_inv)
    zeta_model = jnp.linalg.inv(zeta_inv)
    rt_model = jnp.block(
        [
            [arr_model, rt_art],
            [
                rt_atr,
                theta_model + rt_atr @ (arr_inv @ rt_art),
            ],
        ]
    )
    bulk_model = jnp.block(
        [
            [rt_model, jnp.zeros((surgery.bulk_rt_size, surgery.bulk_zeta_size), dtype=jnp.float64)],
            [jnp.zeros((surgery.bulk_zeta_size, surgery.bulk_rt_size), dtype=jnp.float64), zeta_model],
        ]
    )
    return {
        "arr_inv": arr_inv,
        "theta_inv": theta_inv,
        "zeta_inv": zeta_inv,
        "arr_model": arr_model,
        "theta_model": theta_model,
        "zeta_model": zeta_model,
        "rt_atr": rt_atr,
        "rt_art": rt_art,
        "rt_model": rt_model,
        "bulk_model": bulk_model,
    }


def _build_k1_metric_field_diagnostics(seq):
    jacobian = jnp.asarray(seq.geometry.jacobian_j).reshape(seq.quad.ny, seq.quad.nx, seq.quad.nz)
    metric_inv = jnp.asarray(seq.geometry.metric_inv_jkl).reshape(seq.quad.ny, seq.quad.nx, seq.quad.nz, 3, 3)

    alpha_rr = jacobian * metric_inv[..., 0, 0]
    alpha_thetatheta = jacobian * metric_inv[..., 1, 1]
    alpha_zetazeta = jacobian * metric_inv[..., 2, 2]
    alpha_rtheta = jacobian * metric_inv[..., 0, 1]
    alpha_rzeta = jacobian * metric_inv[..., 0, 2]
    alpha_thetazeta = jacobian * metric_inv[..., 1, 2]

    diag_norms = {
        "alpha_rr": float(jnp.linalg.norm(alpha_rr)),
        "alpha_thetatheta": float(jnp.linalg.norm(alpha_thetatheta)),
        "alpha_zetazeta": float(jnp.linalg.norm(alpha_zetazeta)),
    }
    diag_scale = max(diag_norms.values()) if diag_norms else 1.0
    diag_scale = 1.0 if diag_scale == 0.0 else diag_scale

    return {
        "alpha_rr": alpha_rr,
        "alpha_thetatheta": alpha_thetatheta,
        "alpha_zetazeta": alpha_zetazeta,
        "alpha_rtheta": alpha_rtheta,
        "alpha_rzeta": alpha_rzeta,
        "alpha_thetazeta": alpha_thetazeta,
        "rel_alpha_rtheta": float(jnp.linalg.norm(alpha_rtheta) / diag_scale),
        "rel_alpha_rzeta": float(jnp.linalg.norm(alpha_rzeta) / diag_scale),
        "rel_alpha_thetazeta": float(jnp.linalg.norm(alpha_thetazeta) / diag_scale),
    }


def _assemble_exact_tensor_diagonal_block(
    tensor: jnp.ndarray,
    full_shape: tuple[int, int, int],
    *,
    radial_basis: jnp.ndarray,
    theta_basis: jnp.ndarray,
    zeta_basis: jnp.ndarray,
    radial_weights: jnp.ndarray,
    theta_weights: jnp.ndarray,
    zeta_weights: jnp.ndarray,
    radial_start: int,
) -> jnp.ndarray:
    nr, nt, nz = full_shape
    br = jnp.asarray(radial_basis)[radial_start:radial_start + nr, :]
    bt = jnp.asarray(theta_basis)[:nt, :]
    bz = jnp.asarray(zeta_basis)[:nz, :]
    if br.shape[0] != nr or bt.shape[0] != nt or bz.shape[0] != nz:
        raise ValueError(
            "Exact tensor diagonal block basis restriction does not match the requested extracted shape"
        )

    weighted_tensor = (
        jnp.asarray(tensor)
        * jnp.asarray(theta_weights)[:, None, None]
        * jnp.asarray(radial_weights)[None, :, None]
        * jnp.asarray(zeta_weights)[None, None, :]
    )
    pointwise_basis = jnp.einsum("by,ax,cz->abcyxz", bt, br, bz).reshape(nr * nt * nz, -1)
    matrix = (pointwise_basis * weighted_tensor.reshape(1, -1)) @ pointwise_basis.T
    return 0.5 * (matrix + matrix.T)


def _build_k1_exact_diagonal_model(seq, config: Config, rt_atr: jnp.ndarray, rt_art: jnp.ndarray):
    metric_tensors = _k1_diagonal_metric_tensors(seq)
    arr_exact_diag = _assemble_exact_tensor_diagonal_block(
        metric_tensors["alpha_rr"],
        _arr_shape_k1(seq, config.dirichlet),
        radial_basis=seq.d_basis_r_jk,
        theta_basis=seq.basis_t_jk,
        zeta_basis=seq.basis_z_jk,
        radial_weights=seq.quad.w_x,
        theta_weights=seq.quad.w_y,
        zeta_weights=seq.quad.w_z,
        radial_start=1,
    )
    theta_schur_exact_diag = _assemble_exact_tensor_diagonal_block(
        metric_tensors["alpha_thetatheta"],
        _theta_bulk_shape_k1(seq, config.dirichlet),
        radial_basis=seq.basis_r_jk,
        theta_basis=seq.d_basis_t_jk,
        zeta_basis=seq.basis_z_jk,
        radial_weights=seq.quad.w_x,
        theta_weights=seq.quad.w_y,
        zeta_weights=seq.quad.w_z,
        radial_start=2,
    )
    zeta_exact_diag = _assemble_exact_tensor_diagonal_block(
        metric_tensors["alpha_zetazeta"],
        _zeta_bulk_shape_k1(seq, config.dirichlet),
        radial_basis=seq.basis_r_jk,
        theta_basis=seq.basis_t_jk,
        zeta_basis=seq.d_basis_z_jk,
        radial_weights=seq.quad.w_x,
        theta_weights=seq.quad.w_y,
        zeta_weights=seq.quad.w_z,
        radial_start=2,
    )
    arr_exact_diag_inv = jnp.linalg.inv(arr_exact_diag)
    rt_exact_diag = jnp.block(
        [
            [arr_exact_diag, rt_art],
            [rt_atr, theta_schur_exact_diag + rt_atr @ (arr_exact_diag_inv @ rt_art)],
        ]
    )
    return {
        "rr_exact_diag": arr_exact_diag,
        "theta_schur_exact_diag": theta_schur_exact_diag,
        "zeta_exact_diag": zeta_exact_diag,
        "rt_exact_diag": rt_exact_diag,
    }


def _build_hodge_model_diagnostics(operators, config: Config, exact_matrix: jnp.ndarray, approx_inverse: jnp.ndarray):
    factors = select_boundary_data(
        operators.k0_tensor_hodge_precond,
        config.dirichlet,
        "Tensor Hodge k=0",
    )
    core_size = factors.core_size
    bulk_size = exact_matrix.shape[0] - core_size
    bulk_inv = _matrix_from_apply(
        lambda x: _apply_k0_tensor_hodge_bulk_inverse(factors, x),
        bulk_size,
    )
    acb = _matrix_from_apply(
        lambda x: _apply_k0_tensor_hodge_bulk_to_surgery_coupling(
            SEQ,
            OPERATORS,
            core_size,
            x,
            dirichlet=config.dirichlet,
        ),
        bulk_size,
    )
    abc = _matrix_from_apply(
        lambda x: _apply_k0_tensor_hodge_surgery_to_bulk_coupling(
            SEQ,
            OPERATORS,
            core_size,
            x,
            dirichlet=config.dirichlet,
        ),
        core_size,
    )
    bulk_model = jnp.linalg.inv(0.5 * (bulk_inv + bulk_inv.T))
    schur_model = jnp.linalg.inv(0.5 * (factors.schur_inv + factors.schur_inv.T))
    acc_model = schur_model + acb @ (bulk_inv @ abc)
    coupled_model = jnp.block(
        [
            [acc_model, acb],
            [abc, bulk_model],
        ]
    )
    coupled_model_inverse = jnp.linalg.inv(coupled_model)
    _, _, _, bulk_exact = _split_blocks(exact_matrix, core_size)
    return {
        "core_size": core_size,
        "bulk_size": bulk_size,
        "bulk_inv": bulk_inv,
        "bulk_model": bulk_model,
        "schur_model": schur_model,
        "acc_model": acc_model,
        "coupled_model": coupled_model,
        "coupled_model_inverse": coupled_model_inverse,
        "rel_bulk_error": _relative_fro_error(bulk_model, bulk_exact),
        "rel_coupled_model_inverse_error": _relative_fro_error(approx_inverse, coupled_model_inverse),
        "rel_coupled_model_action_error": _relative_fro_error(coupled_model @ approx_inverse, jnp.eye(exact_matrix.shape[0], dtype=jnp.float64)),
    }


def _block_order_to_global_matrix(matrix: jnp.ndarray, order: jnp.ndarray) -> jnp.ndarray:
    inverse_permutation = jnp.argsort(order)
    return matrix[inverse_permutation][:, inverse_permutation]


def _build_inverse_diagnostics(
    config: Config,
    *,
    name: str,
    operator: jnp.ndarray,
    exact_inverse: jnp.ndarray,
    approx_inverse: jnp.ndarray,
    rhs_target: jnp.ndarray,
):
    rhs_batch = _random_rhs_batch(config, operator.shape[0], rhs_target)
    exact_solutions = jax.vmap(lambda rhs: exact_inverse @ rhs)(rhs_batch)
    approx_solutions = jax.vmap(lambda rhs: approx_inverse @ rhs)(rhs_batch)
    target_rhs = jax.vmap(lambda rhs: rhs_target @ rhs)(rhs_batch)
    rel_errors = jax.vmap(
        lambda x_exact, x_approx: jnp.linalg.norm(x_exact - x_approx)
        / jnp.where(jnp.linalg.norm(x_exact) > 0, jnp.linalg.norm(x_exact), 1.0)
    )(exact_solutions, approx_solutions)
    rel_residuals = jax.vmap(
        lambda rhs_tgt, x_approx: jnp.linalg.norm(operator @ x_approx - rhs_tgt)
        / jnp.where(jnp.linalg.norm(rhs_tgt) > 0, jnp.linalg.norm(rhs_tgt), 1.0)
    )(target_rhs, approx_solutions)
    return {
        "name": name,
        "operator": operator,
        "exact_inverse": exact_inverse,
        "approx_inverse": approx_inverse,
        "rhs_target": rhs_target,
        "rhs_batch": rhs_batch,
        "exact_solutions": exact_solutions,
        "approx_solutions": approx_solutions,
        "rel_inverse_error": _relative_fro_error(approx_inverse, exact_inverse),
        "rel_operator_error": _relative_fro_error(operator @ approx_inverse, rhs_target),
        "rel_errors": rel_errors,
        "rel_residuals": rel_residuals,
    }


def _build_mass_diagnostics(seq, operators, config: Config):
    diagnostics = {}
    for k in (0, 1):
        matrix = jnp.asarray(dense_mass_matrix(seq, operators, k, dirichlet=config.dirichlet))
        exact_inverse = jnp.linalg.inv(matrix)
        approx_inverse = _mass_preconditioner_matrix(seq, operators, config, k, matrix.shape[0])
        diag = _build_inverse_diagnostics(
            config,
            name=f"M{k}",
            operator=matrix,
            exact_inverse=exact_inverse,
            approx_inverse=approx_inverse,
            rhs_target=jnp.eye(matrix.shape[0], dtype=jnp.float64),
        )
        if k == 0:
            core_size = _core_size(seq)
            ass, asb, abs_, bulk_exact = _split_blocks(matrix, core_size)
            surgery_inv = jnp.diag(1.0 / jnp.diag(ass))
            surgery = _select_mass_surgery_factors(
                operators.mass_preconds,
                0,
                config.dirichlet,
            )
            tensor_factors = _select_mass_tensor_factors(
                operators.mass_preconds,
                0,
                config.dirichlet,
            )
            bulk_inv = _k0_bulk_inverse_matrix(operators, config, bulk_exact.shape[0])
            bulk_model = jnp.linalg.inv(0.5 * (bulk_inv + bulk_inv.T))
            block_diagonal_surrogate = jnp.block(
                [
                    [surgery_inv, jnp.zeros((core_size, bulk_exact.shape[0]), dtype=jnp.float64)],
                    [jnp.zeros((bulk_exact.shape[0], core_size), dtype=jnp.float64), bulk_inv],
                ]
            )
            coupled_model = jnp.block(
                [
                    [ass, asb],
                    [abs_, bulk_model],
                ]
            )
            coupled_model_inverse = jnp.linalg.inv(coupled_model)
            diag.update(
                {
                    "core_size": core_size,
                    "ass": ass,
                    "asb": asb,
                    "abs": abs_,
                    "bulk_exact": bulk_exact,
                    "bulk_inv": bulk_inv,
                    "bulk_model": bulk_model,
                    "rel_bulk_error": _relative_fro_error(bulk_model, bulk_exact),
                    "block_diagonal_surrogate": block_diagonal_surrogate,
                    "coupled_model": coupled_model,
                    "coupled_model_inverse": coupled_model_inverse,
                    "stored_schur_inv": tensor_factors.schur_inv,
                    "recomputed_schur_inv": jnp.linalg.inv(ass - asb @ (bulk_inv @ abs_)),
                    "rel_stored_vs_recomputed_schur": _relative_fro_error(
                        tensor_factors.schur_inv,
                        jnp.linalg.inv(ass - asb @ (bulk_inv @ abs_)),
                    ),
                    "rel_surgery_asb": _relative_fro_error(
                        _matrix_from_apply(
                            lambda x: _apply_bulk_to_surgery_coupling(surgery, x),
                            bulk_exact.shape[0],
                        ),
                        asb,
                    ),
                    "rel_surgery_abs": _relative_fro_error(
                        _matrix_from_apply(
                            lambda x: _apply_surgery_to_bulk_coupling(surgery, x),
                            ass.shape[0],
                        ),
                        abs_,
                    ),
                    "rel_block_diag_match": _relative_fro_error(approx_inverse, block_diagonal_surrogate),
                    "rel_coupled_model_inverse_error": _relative_fro_error(approx_inverse, coupled_model_inverse),
                    "rel_coupled_model_action_error": _relative_fro_error(coupled_model @ approx_inverse, jnp.eye(matrix.shape[0], dtype=jnp.float64)),
                }
            )
        if k == 1:
            surgery = _select_mass_surgery_factors(
                operators.mass_preconds,
                1,
                config.dirichlet,
            )
            tensor_factors = _select_mass_tensor_factors(
                operators.mass_preconds,
                1,
                config.dirichlet,
            )
            ass = matrix[surgery.surgery_indices][:, surgery.surgery_indices]
            asb = matrix[surgery.surgery_indices][:, surgery.bulk_indices]
            abs_ = matrix[surgery.bulk_indices][:, surgery.surgery_indices]
            bulk_exact = matrix[surgery.bulk_indices][:, surgery.bulk_indices]
            rt_exact = bulk_exact[:surgery.bulk_rt_size, :surgery.bulk_rt_size]
            rr_exact = rt_exact[:surgery.rt_r_size, :surgery.rt_r_size]
            rtheta_exact = rt_exact[:surgery.rt_r_size, surgery.rt_r_size:]
            thetar_exact = rt_exact[surgery.rt_r_size:, :surgery.rt_r_size]
            thetatheta_exact = rt_exact[surgery.rt_r_size:, surgery.rt_r_size:]
            theta_schur_exact = thetatheta_exact - thetar_exact @ (jnp.linalg.inv(rr_exact) @ rtheta_exact)
            bulk_inv = _k1_bulk_inverse_matrix(operators, config)
            bulk_model_data = _build_k1_bulk_model(operators, config)
            metric_diag = _build_k1_metric_field_diagnostics(seq)
            exact_diag_model = _build_k1_exact_diagonal_model(
                seq,
                config,
                thetar_exact,
                rtheta_exact,
            )
            bulk_model = bulk_model_data["bulk_model"]
            coupled_model = jnp.block(
                [
                    [ass, asb],
                    [abs_, bulk_model],
                ]
            )
            coupled_model_inverse = jnp.linalg.inv(coupled_model)
            bulk_model_denseinv = jnp.linalg.inv(bulk_inv)
            ass_from_schur = jnp.linalg.inv(tensor_factors.schur_inv) + asb @ (bulk_inv @ abs_)
            coupled_model_schur = jnp.block(
                [
                    [ass_from_schur, asb],
                    [abs_, bulk_model_denseinv],
                ]
            )
            coupled_model_schur_inverse = jnp.linalg.inv(coupled_model_schur)
            block_order = jnp.concatenate([surgery.surgery_indices, surgery.bulk_indices])
            coupled_model_global = _block_order_to_global_matrix(coupled_model, block_order)
            coupled_model_inverse_global = _block_order_to_global_matrix(coupled_model_inverse, block_order)
            coupled_model_schur_global = _block_order_to_global_matrix(coupled_model_schur, block_order)
            coupled_model_schur_inverse_global = _block_order_to_global_matrix(coupled_model_schur_inverse, block_order)
            rt_diag = jnp.diag(jnp.diag(rt_exact))
            diag.update(
                {
                    "ass": ass,
                    "asb": asb,
                    "abs": abs_,
                    "bulk_exact": bulk_exact,
                    "bulk_inv": bulk_inv,
                    "bulk_model": bulk_model,
                    "bulk_model_denseinv": bulk_model_denseinv,
                    "metric_diag": metric_diag,
                    "exact_diag_model": exact_diag_model,
                    "arr_cp_relative_error": tensor_factors.arr.cp_relative_error,
                    "theta_cp_relative_error": tensor_factors.theta.cp_relative_error,
                    "zeta_cp_relative_error": tensor_factors.zeta.cp_relative_error,
                    "arr_cp_final_delta": tensor_factors.arr.cp_final_delta,
                    "theta_cp_final_delta": tensor_factors.theta.cp_final_delta,
                    "zeta_cp_final_delta": tensor_factors.zeta.cp_final_delta,
                    "rt_model": bulk_model_data["rt_model"],
                    "rr_model": bulk_model_data["arr_model"],
                    "theta_schur_model": bulk_model_data["theta_model"],
                    "zeta_model": bulk_model_data["zeta_model"],
                    "rt_exact": rt_exact,
                    "rr_exact": rr_exact,
                    "rr_exact_diag": exact_diag_model["rr_exact_diag"],
                    "theta_schur_exact": theta_schur_exact,
                    "theta_schur_exact_diag": exact_diag_model["theta_schur_exact_diag"],
                    "rt_exact_diag": exact_diag_model["rt_exact_diag"],
                    "zeta_exact": bulk_exact[
                        surgery.bulk_rt_size:surgery.bulk_rt_size + surgery.bulk_zeta_size,
                        surgery.bulk_rt_size:surgery.bulk_rt_size + surgery.bulk_zeta_size,
                    ],
                    "rel_rr_model_vs_exact_diag": _relative_fro_error(
                        bulk_model_data["arr_model"],
                        exact_diag_model["rr_exact_diag"],
                    ),
                    "rel_theta_schur_model_vs_exact_diag": _relative_fro_error(
                        bulk_model_data["theta_model"],
                        exact_diag_model["theta_schur_exact_diag"],
                    ),
                    "rel_rt_model_vs_exact_diag": _relative_fro_error(
                        bulk_model_data["rt_model"],
                        exact_diag_model["rt_exact_diag"],
                    ),
                    "rel_rr_exact_diag_vs_exact": _relative_fro_error(
                        exact_diag_model["rr_exact_diag"],
                        rr_exact,
                    ),
                    "rel_theta_schur_exact_diag_vs_exact": _relative_fro_error(
                        exact_diag_model["theta_schur_exact_diag"],
                        theta_schur_exact,
                    ),
                    "rel_rt_exact_diag_vs_exact": _relative_fro_error(
                        exact_diag_model["rt_exact_diag"],
                        rt_exact,
                    ),
                    "rel_rr_error": _relative_fro_error(
                        bulk_model_data["arr_model"],
                        rr_exact,
                    ),
                    "rel_theta_schur_error": _relative_fro_error(
                        bulk_model_data["theta_model"],
                        theta_schur_exact,
                    ),
                    "rel_rt_error": _relative_fro_error(
                        bulk_model_data["rt_model"],
                        rt_exact,
                    ),
                    "rel_zeta_error": _relative_fro_error(
                        bulk_model_data["zeta_model"],
                        bulk_exact[
                            surgery.bulk_rt_size:surgery.bulk_rt_size + surgery.bulk_zeta_size,
                            surgery.bulk_rt_size:surgery.bulk_rt_size + surgery.bulk_zeta_size,
                        ],
                    ),
                    "rel_exact_vs_parts_model": _relative_fro_error(coupled_model_global, matrix),
                    "rel_bulk_error": _relative_fro_error(bulk_model, bulk_exact),
                    "coupled_model": coupled_model_global,
                    "coupled_model_inverse": coupled_model_inverse_global,
                    "coupled_model_schur": coupled_model_schur_global,
                    "coupled_model_schur_inverse": coupled_model_schur_inverse_global,
                    "rel_coupled_model_inverse_error": _relative_fro_error(approx_inverse, coupled_model_inverse_global),
                    "rel_coupled_model_action_error": _relative_fro_error(coupled_model_global @ approx_inverse, jnp.eye(matrix.shape[0], dtype=jnp.float64)),
                    "rel_schur_vs_dense_inverse": _relative_fro_error(approx_inverse, coupled_model_schur_inverse_global),
                    "rel_parts_vs_schur_model": _relative_fro_error(coupled_model_global, coupled_model_schur_global),
                    "rt_offdiag_fraction": float(
                        jnp.linalg.norm(rt_exact - rt_diag)
                        / jnp.maximum(jnp.linalg.norm(rt_exact), 1.0)
                    ),
                }
            )
        diagnostics[k] = diag
    return diagnostics


def _collect_block_diagnostics(seq, operators, config: Config, matrix: jnp.ndarray):
    core_size = _core_size(seq)
    acc, acb, abc, abb = _split_blocks(matrix, core_size)
    rank, cp_maxiter, cp_tol, cp_ridge = _k0_tensor_hodge_config(operators)
    bulk_model = _build_k0_tensor_hodge_bulk_model(
        seq,
        dirichlet=config.dirichlet,
        rank=rank,
        cp_maxiter=cp_maxiter,
        cp_tol=cp_tol,
        cp_ridge=cp_ridge,
    )
    bulk_inv = jnp.linalg.inv(bulk_model)
    schur_exact = acc - acb @ (jnp.linalg.inv(abb) @ abc)
    schur_model = acc - acb @ (bulk_inv @ abc)
    return {
        "core_size": core_size,
        "acc": acc,
        "acb": acb,
        "abc": abc,
        "abb": abb,
        "bulk_model": bulk_model,
        "schur_exact": schur_exact,
        "schur_model": schur_model,
    }


def _imshow_signed(ax, matrix: jnp.ndarray, title: str):
    array = np.asarray(matrix)
    vmax = np.percentile(np.abs(array), 99)
    vmax = 1.0 if vmax == 0.0 else vmax
    nonzero = np.abs(array[np.nonzero(array)])
    if nonzero.size == 0:
        linthresh = 1.0
    else:
        linthresh = max(float(np.percentile(nonzero, 10)), vmax * 1e-6)
    image = ax.imshow(
        array,
        origin="lower",
        cmap="coolwarm",
        norm=mcolors.SymLogNorm(
            linthresh=linthresh,
            linscale=1.0,
            vmin=-vmax,
            vmax=vmax,
            base=10,
        ),
    )
    ax.set_title(title)
    plt.colorbar(image, ax=ax, shrink=0.8)


def _build_dense_diagnostics(config: Config = CONFIG):
    seq, operators = _build_case(config)
    matrix = jnp.asarray(dense_laplacian(seq, operators, 0, dirichlet=config.dirichlet))
    pinv, projector, eigvals, threshold = _spectral_pseudoinverse(
        matrix,
        rtol=config.spectral_rtol,
    )
    tensor_matrix = _preconditioner_matrix(seq, operators, config, matrix.shape[0])
    mass_diagnostics = _build_mass_diagnostics(seq, operators, config)
    block_data = _collect_block_diagnostics(seq, operators, config, matrix)
    hodge_model_diag = _build_hodge_model_diagnostics(operators, config, matrix, tensor_matrix)
    hodge_diag = _build_inverse_diagnostics(
        config,
        name="K0",
        operator=matrix,
        exact_inverse=pinv,
        approx_inverse=tensor_matrix,
        rhs_target=projector,
    )

    print("k=0 tensor Hodge debug build:")
    print(f"  ns={config.ns}, p={config.p}, dirichlet={config.dirichlet}")
    print(f"  matrix size={matrix.shape[0]}, core_size={block_data['core_size']}, bulk_size={block_data['abb'].shape[0]}")
    print(f"  tensor rank={config.tensor_rank}, cp_tol={config.cp_tol}, cp_maxiter={config.cp_maxiter}")
    print(f"  block Richardson steps={config.richardson_steps}, omega={config.richardson_omega}")
    print(f"  spectral threshold={threshold:.3e}")
    print(f"  eig min/max=({float(jnp.min(eigvals)):.3e}, {float(jnp.max(eigvals)):.3e})")
    print("mass tensor contrasts:")
    for k in (0, 1):
        mass_diag = mass_diagnostics[k]
        print(f"  M{k}: ||P_tensor - M{k}^(-1)|| / ||M{k}^(-1)|| = {mass_diag['rel_inverse_error']:.3e}")
        print(f"      ||M{k} P_tensor - I|| / ||I|| = {mass_diag['rel_operator_error']:.3e}")
        print(f"      median relative solution error = {float(jnp.median(mass_diag['rel_errors'])):.3e}")
        print(f"      max relative solution error    = {float(jnp.max(mass_diag['rel_errors'])):.3e}")
        print(f"      median relative residual       = {float(jnp.median(mass_diag['rel_residuals'])):.3e}")
        print(f"      max relative residual          = {float(jnp.max(mass_diag['rel_residuals'])):.3e}")
        if k == 0:
            print(f"      ||M0_bb(model) - M0_bb(exact)|| / ||M0_bb(exact)|| = {mass_diag['rel_bulk_error']:.3e}")
            print(f"      ||P_M0 - P_bd|| / ||P_bd|| = {mass_diag['rel_block_diag_match']:.3e}")
            print(f"      ||P_M0 - A_model^(-1)|| / ||A_model^(-1)|| = {mass_diag['rel_coupled_model_inverse_error']:.3e}")
            print(f"      ||A_model P_M0 - I|| / ||I|| = {mass_diag['rel_coupled_model_action_error']:.3e}")
            print(f"      ||S_stored^(-1) - S_recomputed^(-1)|| / ||S_recomputed^(-1)|| = {mass_diag['rel_stored_vs_recomputed_schur']:.3e}")
        if k == 1:
            print(f"      cp(alpha_rr) relative error = {mass_diag['arr_cp_relative_error']:.3e} (delta {mass_diag['arr_cp_final_delta']:.3e})")
            print(f"      cp(alpha_thetatheta) relative error = {mass_diag['theta_cp_relative_error']:.3e} (delta {mass_diag['theta_cp_final_delta']:.3e})")
            print(f"      cp(alpha_zetazeta) relative error = {mass_diag['zeta_cp_relative_error']:.3e} (delta {mass_diag['zeta_cp_final_delta']:.3e})")
            print(f"      ||M1_rr(model) - M1_rr(diag-exact)|| / ||M1_rr(diag-exact)|| = {mass_diag['rel_rr_model_vs_exact_diag']:.3e}")
            print(f"      ||M1_rr(diag-exact) - M1_rr(exact)|| / ||M1_rr(exact)|| = {mass_diag['rel_rr_exact_diag_vs_exact']:.3e}")
            print(f"      ||M1_rr(model) - M1_rr(exact)|| / ||M1_rr(exact)|| = {mass_diag['rel_rr_error']:.3e}")
            print(f"      ||S_theta(model) - S_theta(diag-exact)|| / ||S_theta(diag-exact)|| = {mass_diag['rel_theta_schur_model_vs_exact_diag']:.3e}")
            print(f"      ||S_theta(diag-exact) - S_theta(exact)|| / ||S_theta(exact)|| = {mass_diag['rel_theta_schur_exact_diag_vs_exact']:.3e}")
            print(f"      ||S_theta(model) - S_theta(exact)|| / ||S_theta(exact)|| = {mass_diag['rel_theta_schur_error']:.3e}")
            print(f"      ||M1_rt(model) - M1_rt(diag-exact)|| / ||M1_rt(diag-exact)|| = {mass_diag['rel_rt_model_vs_exact_diag']:.3e}")
            print(f"      ||M1_rt(diag-exact) - M1_rt(exact)|| / ||M1_rt(exact)|| = {mass_diag['rel_rt_exact_diag_vs_exact']:.3e}")
            print(f"      ||M1_model(parts) - M1|| / ||M1|| = {mass_diag['rel_exact_vs_parts_model']:.3e}")
            print(f"      ||M1_rt(model) - M1_rt(exact)|| / ||M1_rt(exact)|| = {mass_diag['rel_rt_error']:.3e}")
            print(f"      ||M1_zeta(model) - M1_zeta(exact)|| / ||M1_zeta(exact)|| = {mass_diag['rel_zeta_error']:.3e}")
            print(f"      ||M1_bb(model) - M1_bb(exact)|| / ||M1_bb(exact)|| = {mass_diag['rel_bulk_error']:.3e}")
            print(f"      ||P_M1 - A_model^(-1)|| / ||A_model^(-1)|| = {mass_diag['rel_coupled_model_inverse_error']:.3e}")
            print(f"      ||A_model P_M1 - I|| / ||I|| = {mass_diag['rel_coupled_model_action_error']:.3e}")
            print(f"      ||P_M1 - A_schur^(-1)|| / ||A_schur^(-1)|| = {mass_diag['rel_schur_vs_dense_inverse']:.3e}")
            print(f"      ||A_model(parts) - A_schur|| / ||A_schur|| = {mass_diag['rel_parts_vs_schur_model']:.3e}")
            print(f"      ||offdiag(M1_rt)|| / ||M1_rt|| = {mass_diag['rt_offdiag_fraction']:.3e}")
            print(f"      ||J g^01|| / max(||J g^ii||) = {mass_diag['metric_diag']['rel_alpha_rtheta']:.3e}")
            print(f"      ||J g^02|| / max(||J g^ii||) = {mass_diag['metric_diag']['rel_alpha_rzeta']:.3e}")
            print(f"      ||J g^12|| / max(||J g^ii||) = {mass_diag['metric_diag']['rel_alpha_thetazeta']:.3e}")
    print("relative Frobenius errors:")
    print(f"  ||P_tensor - K^+|| / ||K^+|| = {hodge_diag['rel_inverse_error']:.3e}")
    print(f"  ||K P_tensor - P_range|| / ||P_range|| = {hodge_diag['rel_operator_error']:.3e}")
    print(f"  ||A_bb(model) - A_bb(exact)|| / ||A_bb(exact)|| = {_relative_fro_error(block_data['bulk_model'], block_data['abb']):.3e}")
    print(f"  ||S(model) - S(exact)|| / ||S(exact)|| = {_relative_fro_error(block_data['schur_model'], block_data['schur_exact']):.3e}")
    print(f"  ||P_tensor - H_model^(-1)|| / ||H_model^(-1)|| = {hodge_model_diag['rel_coupled_model_inverse_error']:.3e}")
    print(f"  ||H_model P_tensor - I|| / ||I|| = {hodge_model_diag['rel_coupled_model_action_error']:.3e}")
    print("random range-RHS diagnostics:")
    print(f"  median relative solution error = {float(jnp.median(hodge_diag['rel_errors'])):.3e}")
    print(f"  max relative solution error    = {float(jnp.max(hodge_diag['rel_errors'])):.3e}")
    print(f"  median relative residual       = {float(jnp.median(hodge_diag['rel_residuals'])):.3e}")
    print(f"  max relative residual          = {float(jnp.max(hodge_diag['rel_residuals'])):.3e}")

    return (
        seq,
        operators,
        matrix,
        pinv,
        projector,
        tensor_matrix,
        block_data,
        mass_diagnostics,
        hodge_model_diag,
        hodge_diag['rhs_batch'],
        hodge_diag['exact_solutions'],
        hodge_diag['approx_solutions'],
        hodge_diag['rel_errors'],
        hodge_diag['rel_residuals'],
    )


# %% Build the dense diagnostic objects
(
    SEQ,
    OPERATORS,
    K_EXACT,
    K_PINV,
    RANGE_PROJECTOR,
    P_TENSOR,
    BLOCKS,
    MASS_DIAGNOSTICS,
    HODGE_DIAGNOSTICS,
    RANDOM_RHS,
    RANDOM_EXACT,
    RANDOM_TENSOR,
    RANDOM_REL_ERRORS,
    RANDOM_REL_RESIDUALS,
) = _build_dense_diagnostics()


# %% Plot the exact operator, exact inverse, tensor preconditioner, and inverse error
fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
_imshow_signed(axes[0, 0], K_EXACT, "Exact k=0 stiffness K")
_imshow_signed(axes[0, 1], K_PINV, "Exact inverse / pseudoinverse K^+")
_imshow_signed(axes[1, 0], P_TENSOR, "Tensor preconditioner matrix P_tensor")
_imshow_signed(axes[1, 1], P_TENSOR - K_PINV, "P_tensor - K^+")


# %% Contrast M0 and M1 tensor mass inverses against the exact inverses
fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
for row, k in enumerate((0, 1)):
    diag = MASS_DIAGNOSTICS[k]
    _imshow_signed(axes[row, 0], diag["operator"], f"Exact M{k}")
    _imshow_signed(axes[row, 1], diag["exact_inverse"], f"Exact M{k}^(-1)")
    _imshow_signed(axes[row, 2], diag["approx_inverse"], f"Tensor mass preconditioner P_M{k}")
    _imshow_signed(axes[row, 3], diag["approx_inverse"] - diag["exact_inverse"], f"P_M{k} - M{k}^(-1)")


# %% Contrast the preconditioned mass operators against the identity
fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
for row, k in enumerate((0, 1)):
    diag = MASS_DIAGNOSTICS[k]
    target = diag["rhs_target"]
    preconditioned = diag["operator"] @ diag["approx_inverse"]
    _imshow_signed(axes[row, 0], preconditioned, f"M{k} P_M{k}")
    _imshow_signed(axes[row, 1], target, f"Identity target for M{k}")
    _imshow_signed(axes[row, 2], preconditioned - target, f"M{k} P_M{k} - I")


# %% Plot the scalar M0 bulk model extracted from the tensor mass preconditioner
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
_imshow_signed(axes[0], MASS_DIAGNOSTICS[0]["bulk_exact"], "Exact M0 bulk block")
_imshow_signed(axes[1], MASS_DIAGNOSTICS[0]["bulk_model"], "Modeled M0 bulk block")
_imshow_signed(axes[2], MASS_DIAGNOSTICS[0]["bulk_model"] - MASS_DIAGNOSTICS[0]["bulk_exact"], "M0 bulk model error")


# %% Contrast M0 against the coupled model inverse and block-diagonal surrogate
fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
_imshow_signed(axes[0, 0], MASS_DIAGNOSTICS[0]["approx_inverse"], "Production P_M0")
_imshow_signed(axes[0, 1], MASS_DIAGNOSTICS[0]["block_diagonal_surrogate"], "Block-diagonal surrogate P_bd")
_imshow_signed(axes[0, 2], MASS_DIAGNOSTICS[0]["approx_inverse"] - MASS_DIAGNOSTICS[0]["block_diagonal_surrogate"], "P_M0 - P_bd")
_imshow_signed(axes[1, 0], MASS_DIAGNOSTICS[0]["coupled_model_inverse"], "Coupled model inverse A_model^(-1)")
_imshow_signed(axes[1, 1], MASS_DIAGNOSTICS[0]["coupled_model"] @ MASS_DIAGNOSTICS[0]["approx_inverse"], "A_model P_M0")
_imshow_signed(axes[1, 2], MASS_DIAGNOSTICS[0]["coupled_model"] @ MASS_DIAGNOSTICS[0]["approx_inverse"] - jnp.eye(MASS_DIAGNOSTICS[0]["operator"].shape[0], dtype=jnp.float64), "A_model P_M0 - I")


# %% Contrast M1 against the coupled model inverse
fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
_imshow_signed(axes[0, 0], MASS_DIAGNOSTICS[1]["rt_exact"], "Exact M1 rt block")
_imshow_signed(axes[0, 1], MASS_DIAGNOSTICS[1]["rt_model"], "Modeled M1 rt block")
_imshow_signed(axes[0, 2], MASS_DIAGNOSTICS[1]["rt_model"] - MASS_DIAGNOSTICS[1]["rt_exact"], "M1 rt model error")
_imshow_signed(axes[1, 0], MASS_DIAGNOSTICS[1]["coupled_model_inverse"], "Coupled model inverse M1")
_imshow_signed(axes[1, 1], MASS_DIAGNOSTICS[1]["coupled_model"] @ MASS_DIAGNOSTICS[1]["approx_inverse"], "A_model P_M1")
_imshow_signed(axes[1, 2], MASS_DIAGNOSTICS[1]["coupled_model"] @ MASS_DIAGNOSTICS[1]["approx_inverse"] - jnp.eye(MASS_DIAGNOSTICS[1]["operator"].shape[0], dtype=jnp.float64), "A_model P_M1 - I")


# %% Plot the exact and modeled M1 bulk block
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
_imshow_signed(axes[0], MASS_DIAGNOSTICS[1]["bulk_exact"], "Exact M1 bulk block")
_imshow_signed(axes[1], MASS_DIAGNOSTICS[1]["bulk_model"], "Modeled M1 bulk block")
_imshow_signed(axes[2], MASS_DIAGNOSTICS[1]["bulk_model"] - MASS_DIAGNOSTICS[1]["bulk_exact"], "M1 bulk model error")


# %% Plot the exact and modeled bulk / Schur blocks
fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
_imshow_signed(axes[0, 0], BLOCKS["abb"], "Exact bulk block A_bb")
_imshow_signed(axes[0, 1], BLOCKS["bulk_model"], "Modeled bulk block")
_imshow_signed(axes[0, 2], BLOCKS["bulk_model"] - BLOCKS["abb"], "Bulk model error")
_imshow_signed(axes[1, 0], BLOCKS["schur_exact"], "Exact Schur block")
_imshow_signed(axes[1, 1], BLOCKS["schur_model"], "Modeled Schur block")
_imshow_signed(axes[1, 2], BLOCKS["schur_model"] - BLOCKS["schur_exact"], "Schur model error")


# %% Contrast the k=0 Hodge tensor apply against the coupled model inverse
fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
_imshow_signed(axes[0, 0], HODGE_DIAGNOSTICS["bulk_model"], "Modeled Hodge bulk block")
_imshow_signed(axes[0, 1], HODGE_DIAGNOSTICS["coupled_model_inverse"], "Coupled model inverse H_model^(-1)")
_imshow_signed(axes[0, 2], P_TENSOR - HODGE_DIAGNOSTICS["coupled_model_inverse"], "P_tensor - H_model^(-1)")
_imshow_signed(axes[1, 0], HODGE_DIAGNOSTICS["coupled_model"] @ P_TENSOR, "H_model P_tensor")
_imshow_signed(axes[1, 1], jnp.eye(K_EXACT.shape[0], dtype=jnp.float64), "Identity target")
_imshow_signed(axes[1, 2], HODGE_DIAGNOSTICS["coupled_model"] @ P_TENSOR - jnp.eye(K_EXACT.shape[0], dtype=jnp.float64), "H_model P_tensor - I")


# %% Plot the preconditioned operator against the exact range projector
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
_imshow_signed(axes[0], K_EXACT @ P_TENSOR, "K P_tensor")
_imshow_signed(axes[1], RANGE_PROJECTOR, "Exact range projector")
_imshow_signed(axes[2], K_EXACT @ P_TENSOR - RANGE_PROJECTOR, "K P_tensor - P_range")


# %% Plot random-RHS diagnostics
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
axes[0].plot(np.asarray(RANDOM_REL_ERRORS), marker="o")
axes[0].set_title("Random RHS relative solution error")
axes[0].set_xlabel("sample")
axes[0].set_ylabel(r"$\|x_{\mathrm{tensor}} - x_{\mathrm{exact}}\| / \|x_{\mathrm{exact}}\|$")
axes[0].set_yscale("log")

axes[1].plot(np.asarray(RANDOM_REL_RESIDUALS), marker="o")
axes[1].set_title("Random RHS relative residual")
axes[1].set_xlabel("sample")
axes[1].set_ylabel(r"$\|K x_{\mathrm{tensor}} - P_{\mathrm{range}} b\| / \|P_{\mathrm{range}} b\|$")
axes[1].set_yscale("log")


# %% Plot random-RHS mass diagnostics for M0 and M1 on the same scale
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
for k, marker in ((0, "o"), (1, "s")):
    axes[0].plot(np.asarray(MASS_DIAGNOSTICS[k]["rel_errors"]), marker=marker, label=f"M{k}")
    axes[1].plot(np.asarray(MASS_DIAGNOSTICS[k]["rel_residuals"]), marker=marker, label=f"M{k}")

axes[0].set_title("Random RHS mass relative solution error")
axes[0].set_xlabel("sample")
axes[0].set_ylabel(r"$\|x_{\mathrm{tensor}} - x_{\mathrm{exact}}\| / \|x_{\mathrm{exact}}\|$")
axes[0].set_yscale("log")
axes[0].legend()

axes[1].set_title("Random RHS mass relative residual")
axes[1].set_xlabel("sample")
axes[1].set_ylabel(r"$\|M x_{\mathrm{tensor}} - b\| / \|b\|$")
axes[1].set_yscale("log")
axes[1].legend()
