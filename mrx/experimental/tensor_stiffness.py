"""k>=1 tensor stiffness (block_fd P_A) research machinery.

NOT production: the production k>=1 Laplacian preconditioner is the
Schur-outer Jacobi with the tensor-probed diagonal (see
``docs/PRODUCTION.md``). This module holds the tensor stiffness forward
models, the block_fd stiffness preconditioner with its atom variants
(``MRX_K1_ATOM`` = bundled | cp | profile | rank1), and the k=1/k=2
stiffness surgery factors used by the research benchmarks
(``scripts/benchmark/benchmark_graddiv_k1_preconditioner.py`` and the
greville verify scripts). Campaign results and reopen conditions:
``docs/research/handoff_2026-08-13_eod.md``.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence
import os

import equinox as eqx
import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import numpy as np

from mrx.operators import (
    K1TensorCurlCurlForwardModel,
    K2TensorDivDivForwardModel,
    SequenceOperators,
    _assemble_dense_from_apply,
    _assemble_unweighted_1d_mass,
    _assemble_weighted_1d_mass,
    _assemble_weighted_1d_stiffness,
    _dense_incidence_1d,
    _ensure_extraction_operators,
    _incidence_components,
    _mass_components,
    _mass_extraction,
    _prod3,
    _reshape_quadrature_scalar_field,
    _tensor_mass_rank,
    assemble_incidence_operators,
    assemble_mass_operators,
)
from mrx.preconditioners import (
    BoundaryConditionPair,
    K1MassSurgeryPreconditionerFactors,
    K1TensorMassPreconditionerFactors,
    K2MassSurgeryPreconditionerFactors,
    K2TensorMassPreconditionerFactors,
    TensorDiagonalBlockInverseFactors,
    _apply_bulk_to_surgery_coupling,
    _apply_extracted_submatrix,
    _apply_k1_bulk_diagonal_preconditioner,
    _apply_k1_bulk_preconditioner,
    _apply_k2_bulk_diagonal_preconditioner,
    _apply_k2_bulk_preconditioner,
    _apply_surgery_to_bulk_coupling,
    _arr_shape_k1,
    _assemble_surgery_schur_inverse_from_applies,
    _build_extracted_mass_apply_data,
    _build_mass_referenced_tensor_block_factors,
    _cp_als_3tensor,
    _k2_diagonal_metric_tensors,
    _mass_orthonormal_basis,
    _modal_regularized_inverse_denom,
    _r_bulk_shape_k2,
    _restrict_radial_mass,
    _simultaneous_diagonalize_pair,
    _symmetrize,
    _tensor_block_indices_k1,
    _tensor_block_indices_k2,
    _theta_bulk_shape_k1,
    _theta_shape_k2,
    _zeta_bulk_shape_k1,
    _zeta_shape_k2,
    select_boundary_data,
)

class K1TensorStiffnessPreconditioner(eqx.Module):
    surgery: K1MassSurgeryPreconditionerFactors
    factors: K1TensorMassPreconditionerFactors


class K2TensorStiffnessPreconditioner(eqx.Module):
    surgery: K2MassSurgeryPreconditionerFactors
    factors: K2TensorMassPreconditionerFactors


class _ComposedStiffnessMatvec(eqx.Module):
    g: object
    g_t: object
    m_next: object

    def __matmul__(self, x):
        return self.g_t @ (self.m_next @ (self.g @ x))

    def __call__(self, x):
        return self.g_t @ (self.m_next @ (self.g @ x))


def _k1_regular_component_shapes(seq) -> dict[str, tuple[int, int, int]]:
    return {
        'r': (seq.basis_1.dr, seq.basis_1.nt, seq.basis_1.nz),
        'theta': (seq.basis_1.nr, seq.basis_1.dt, seq.basis_1.nz),
        'zeta': (seq.basis_1.nr, seq.basis_1.nt, seq.basis_1.dz),
    }


def _k2_regular_component_shapes(seq) -> dict[str, tuple[int, int, int]]:
    return {
        'r': (seq.basis_2.nr, seq.basis_2.dt, seq.basis_2.dz),
        'theta': (seq.basis_2.dr, seq.basis_2.nt, seq.basis_2.dz),
        'zeta': (seq.basis_2.dr, seq.basis_2.dt, seq.basis_2.nz),
    }


def _k2_divdiv_weight_tensor(seq) -> jnp.ndarray:
    return _reshape_quadrature_scalar_field(seq, 1.0 / seq.geometry.jacobian_j)


def _apply_kron3_operators(
        operator_r: jnp.ndarray,
        operator_t: jnp.ndarray,
        operator_z: jnp.ndarray,
        tensor: jnp.ndarray) -> jnp.ndarray:
    return jnp.einsum(
        'ai,bj,ck,ijk->abc',
        operator_r,
        operator_t,
        operator_z,
        tensor,
        optimize=True,
    )


def _bundled_rank1_mass_factors(seq, tensor):
    """Deterministic mean-field rank-1 factors of a bundled quadrature weight
    (layout (theta, r, zeta)): cross-axis quad-weighted mean profiles, with
    the radial averaging cut at xi_1 (the polar-surgery element; the 1/r-type
    channels otherwise let the innermost Gauss points dominate the mean), and
    a 1/mean^2 normalization so an exactly rank-1 tensor is reproduced. The
    fdbund recipe from the 2026-08-13 k=0 campaign, replacing CP-ALS with a
    deterministic fit (no ALS iteration, W7-X-robust). Returns
    (scale, f_theta, f_r, f_zeta, relative_error)."""
    wx = jnp.asarray(seq.quad.w_x, dtype=jnp.float64)
    wy = jnp.asarray(seq.quad.w_y, dtype=jnp.float64)
    wz = jnp.asarray(seq.quad.w_z, dtype=jnp.float64)
    xi1 = jnp.asarray(seq.basis_0.Λ[0].T)[seq.ps[0] + 1]
    wx_cut = wx * (jnp.asarray(seq.quad.x_x) >= xi1)
    sy, sz = jnp.sum(wy), jnp.sum(wz)
    sxc = jnp.sum(wx_cut)
    f_t = jnp.einsum('trz,r,z->t', tensor, wx_cut, wz) / (sxc * sz)
    f_r = jnp.einsum('trz,t,z->r', tensor, wy, wz) / (sy * sz)
    f_z = jnp.einsum('trz,t,r->z', tensor, wy, wx_cut) / (sy * sxc)

    def _floor(v):
        return jnp.maximum(v, 1e-8 * jnp.abs(jnp.median(v)))

    f_t, f_r, f_z = _floor(f_t), _floor(f_r), _floor(f_z)
    mean_cut = jnp.einsum('trz,t,r,z->', tensor, wy, wx_cut, wz) / (sy * sxc * sz)
    scale = 1.0 / jnp.maximum(mean_cut, 1e-30) ** 2
    model = scale * f_t[:, None, None] * f_r[None, :, None] * f_z[None, None, :]
    rel_err = float(jnp.linalg.norm(tensor - model) / jnp.maximum(jnp.linalg.norm(tensor), 1e-30))
    return scale, f_t, f_r, f_z, rel_err


def _assemble_weighted_cp_mass_terms(
        *,
        seq,
        rank: int,
        tensor: jnp.ndarray,
        basis_r: jnp.ndarray,
        basis_t: jnp.ndarray,
        basis_z: jnp.ndarray,
        cp_maxiter: int,
        cp_tol: float,
        cp_ridge: float,
        bundled: bool = False) -> tuple[tuple[jnp.ndarray, ...], tuple[jnp.ndarray, ...], tuple[jnp.ndarray, ...], float, float]:
    if bundled:
        scale, f_t, f_r, f_z, rel_err = _bundled_rank1_mass_factors(seq, tensor)
        weights = jnp.asarray([scale])
        factors = (f_t[:, None], f_r[:, None], f_z[:, None])
        cp_relative_error, cp_final_delta = rel_err, 0.0
        rank = 1
    else:
        weights, factors, cp_relative_error, cp_final_delta, _ = _cp_als_3tensor(
            tensor,
            rank,
            maxiter=cp_maxiter,
            tol=cp_tol,
            ridge=cp_ridge,
        )
    mass_r_terms = []
    mass_t_terms = []
    mass_z_terms = []
    component_mass_r_terms = []
    component_mass_t_terms = []
    component_mass_z_terms = []
    for idx in range(rank):
        factor_theta = jnp.ravel(factors[0][:, idx])
        factor_r = jnp.ravel(factors[1][:, idx])
        factor_z = jnp.ravel(factors[2][:, idx])
        scale = weights[idx]
        mass_r_terms.append(_symmetrize(_assemble_weighted_1d_mass(
            basis_r,
            seq.quad.w_x * (scale * factor_r),
        )))
        mass_t_terms.append(_symmetrize(_assemble_weighted_1d_mass(
            basis_t,
            seq.quad.w_y * factor_theta,
        )))
        mass_z_terms.append(_symmetrize(_assemble_weighted_1d_mass(
            basis_z,
            seq.quad.w_z * factor_z,
        )))
    return (
        tuple(mass_r_terms),
        tuple(mass_t_terms),
        tuple(mass_z_terms),
        cp_relative_error,
        cp_final_delta,
    )


def _assemble_k1_curlcurl_regular_tensor_model(
        seq, *, rank: int, cp_maxiter: int, cp_tol: float,
        cp_ridge: float) -> K1TensorCurlCurlForwardModel:
    if rank < 1:
        raise ValueError(f"k=1 curl-curl tensor model requires rank >= 1 (got rank={rank})")

    # Bundled deterministic rank-1 channel fits are the default (adopted with
    # the 2026-08-13 fdbund k=0 campaign); MRX_K1_ATOM=cp reverts to CP-ALS.
    bundled = os.environ.get("MRX_K1_ATOM", "bundled") != "cp"
    metric_tensors = _k2_diagonal_metric_tensors(seq)
    component_shapes = _k1_regular_component_shapes(seq)
    curl_shapes = _k2_regular_component_shapes(seq)
    types = seq.basis_0.types
    g_r = _dense_incidence_1d(seq.basis_0.nr, types[0])
    g_t = _dense_incidence_1d(seq.basis_0.nt, types[1])
    g_z = _dense_incidence_1d(seq.basis_0.nz, types[2])

    rr_mass_r_terms, rr_mass_t_terms, rr_mass_z_terms, rr_rel_err, rr_final_delta = _assemble_weighted_cp_mass_terms(
        seq=seq,
        rank=rank,
        tensor=metric_tensors['beta_rr'],
        bundled=bundled,
        basis_r=seq.basis_r_jk,
        basis_t=seq.d_basis_t_jk,
        basis_z=seq.d_basis_z_jk,
        cp_maxiter=cp_maxiter,
        cp_tol=cp_tol,
        cp_ridge=cp_ridge,
    )
    tt_mass_r_terms, tt_mass_t_terms, tt_mass_z_terms, tt_rel_err, tt_final_delta = _assemble_weighted_cp_mass_terms(
        seq=seq,
        rank=rank,
        tensor=metric_tensors['beta_thetatheta'],
        bundled=bundled,
        basis_r=seq.d_basis_r_jk,
        basis_t=seq.basis_t_jk,
        basis_z=seq.d_basis_z_jk,
        cp_maxiter=cp_maxiter,
        cp_tol=cp_tol,
        cp_ridge=cp_ridge,
    )
    zz_mass_r_terms, zz_mass_t_terms, zz_mass_z_terms, zz_rel_err, zz_final_delta = _assemble_weighted_cp_mass_terms(
        seq=seq,
        rank=rank,
        tensor=metric_tensors['beta_zetazeta'],
        bundled=bundled,
        basis_r=seq.d_basis_r_jk,
        basis_t=seq.d_basis_t_jk,
        basis_z=seq.basis_z_jk,
        cp_maxiter=cp_maxiter,
        cp_tol=cp_tol,
        cp_ridge=cp_ridge,
    )

    return K1TensorCurlCurlForwardModel(
        r_shape=component_shapes['r'],
        theta_shape=component_shapes['theta'],
        zeta_shape=component_shapes['zeta'],
        curl_r_shape=curl_shapes['r'],
        curl_theta_shape=curl_shapes['theta'],
        curl_zeta_shape=curl_shapes['zeta'],
        rank=rank,
        g_r=g_r,
        g_t=g_t,
        g_z=g_z,
        rr_mass_r_terms=rr_mass_r_terms,
        rr_mass_t_terms=rr_mass_t_terms,
        rr_mass_z_terms=rr_mass_z_terms,
        tt_mass_r_terms=tt_mass_r_terms,
        tt_mass_t_terms=tt_mass_t_terms,
        tt_mass_z_terms=tt_mass_z_terms,
        zz_mass_r_terms=zz_mass_r_terms,
        zz_mass_t_terms=zz_mass_t_terms,
        zz_mass_z_terms=zz_mass_z_terms,
        cp_relative_error=max(rr_rel_err, tt_rel_err, zz_rel_err),
        cp_final_delta=max(rr_final_delta, tt_final_delta, zz_final_delta),
    )


def _apply_k1_curlcurl_regular_tensor_model(
        model: K1TensorCurlCurlForwardModel,
        rhs: jnp.ndarray) -> jnp.ndarray:
    r_size = _prod3(model.r_shape)
    theta_size = _prod3(model.theta_shape)
    zeta_size = _prod3(model.zeta_shape)
    rhs_r = rhs[:r_size].reshape(model.r_shape)
    rhs_theta = rhs[r_size:r_size + theta_size].reshape(model.theta_shape)
    rhs_zeta = rhs[r_size + theta_size:r_size + theta_size + zeta_size].reshape(model.zeta_shape)

    identity_dr = jnp.eye(model.r_shape[0], dtype=rhs.dtype)
    identity_nr = jnp.eye(model.theta_shape[0], dtype=rhs.dtype)
    identity_nt = jnp.eye(model.r_shape[1], dtype=rhs.dtype)
    identity_dt = jnp.eye(model.theta_shape[1], dtype=rhs.dtype)
    identity_nz = jnp.eye(model.r_shape[2], dtype=rhs.dtype)
    identity_dz = jnp.eye(model.zeta_shape[2], dtype=rhs.dtype)

    curl_r = _apply_kron3_operators(identity_nr, model.g_t, identity_dz, rhs_zeta)
    curl_r = curl_r - _apply_kron3_operators(identity_nr, identity_dt, model.g_z, rhs_theta)
    curl_theta = _apply_kron3_operators(identity_dr, identity_nt, model.g_z, rhs_r)
    curl_theta = curl_theta - _apply_kron3_operators(model.g_r, identity_nt, identity_dz, rhs_zeta)
    curl_zeta = _apply_kron3_operators(model.g_r, identity_dt, identity_nz, rhs_theta)
    curl_zeta = curl_zeta - _apply_kron3_operators(identity_dr, model.g_t, identity_nz, rhs_r)

    weighted_r = jnp.zeros(model.curl_r_shape, dtype=rhs.dtype)
    weighted_theta = jnp.zeros(model.curl_theta_shape, dtype=rhs.dtype)
    weighted_zeta = jnp.zeros(model.curl_zeta_shape, dtype=rhs.dtype)

    for mass_r, mass_t, mass_z in zip(
            model.rr_mass_r_terms,
            model.rr_mass_t_terms,
            model.rr_mass_z_terms):
        weighted_r = weighted_r + _apply_kron3_operators(mass_r, mass_t, mass_z, curl_r)
    for mass_r, mass_t, mass_z in zip(
            model.tt_mass_r_terms,
            model.tt_mass_t_terms,
            model.tt_mass_z_terms):
        weighted_theta = weighted_theta + _apply_kron3_operators(mass_r, mass_t, mass_z, curl_theta)
    for mass_r, mass_t, mass_z in zip(
            model.zz_mass_r_terms,
            model.zz_mass_t_terms,
            model.zz_mass_z_terms):
        weighted_zeta = weighted_zeta + _apply_kron3_operators(mass_r, mass_t, mass_z, curl_zeta)

    out_r = _apply_kron3_operators(identity_dr, identity_nt, model.g_z.T, weighted_theta)
    out_r = out_r - _apply_kron3_operators(identity_dr, model.g_t.T, identity_nz, weighted_zeta)
    out_theta = -_apply_kron3_operators(identity_nr, identity_dt, model.g_z.T, weighted_r)
    out_theta = out_theta + _apply_kron3_operators(model.g_r.T, identity_dt, identity_nz, weighted_zeta)
    out_zeta = _apply_kron3_operators(identity_nr, model.g_t.T, identity_dz, weighted_r)
    out_zeta = out_zeta - _apply_kron3_operators(model.g_r.T, identity_nt, identity_dz, weighted_theta)

    return jnp.concatenate([
        out_r.reshape(-1),
        out_theta.reshape(-1),
        out_zeta.reshape(-1),
    ])


def _apply_k1_curlcurl_regular_forward(
        operators: SequenceOperators,
        rhs: jnp.ndarray) -> jnp.ndarray:
    g1, g1_T = _incidence_components(operators, 1)
    m2, _, _ = _mass_components(operators, 2)
    if g1 is None or g1_T is None:
        raise ValueError("Incidence operator G1 is required for regular-space curl-curl apply")
    if m2 is None:
        raise ValueError("Mass operator M2 is required for regular-space curl-curl apply")
    return g1_T @ (m2 @ (g1 @ rhs))


def _apply_k1_curlcurl_extracted_tensor_model(
        operators: SequenceOperators,
        model: K1TensorCurlCurlForwardModel,
        rhs: jnp.ndarray,
        *,
        dirichlet: bool = True) -> jnp.ndarray:
    e1, e1_T = _mass_extraction(operators, 1, dirichlet)
    if e1 is None or e1_T is None:
        side = "dbc" if dirichlet else "free"
        raise ValueError(f"Extraction operator E1 is required for extracted {side} k=1 tensor apply")
    return e1 @ _apply_k1_curlcurl_regular_tensor_model(model, e1_T @ rhs)


def _assemble_k1_curlcurl_regular_tensor_dense_matrix(
        model: K1TensorCurlCurlForwardModel) -> jnp.ndarray:
    size = _prod3(model.r_shape) + _prod3(model.theta_shape) + _prod3(model.zeta_shape)
    return _assemble_dense_from_apply(
        lambda x, tensor_model=model: _apply_k1_curlcurl_regular_tensor_model(tensor_model, x),
        size,
    )


def _assemble_k2_divdiv_regular_tensor_model(
        seq, *, rank: int, cp_maxiter: int, cp_tol: float,
        cp_ridge: float) -> K2TensorDivDivForwardModel:
    if rank < 1:
        raise ValueError(f"k=2 div-div tensor model requires rank >= 1 (got rank={rank})")

    weight_tensor = _k2_divdiv_weight_tensor(seq)
    weights, factors, cp_relative_error, cp_final_delta, _ = _cp_als_3tensor(
        weight_tensor,
        rank,
        maxiter=cp_maxiter,
        tol=cp_tol,
        ridge=cp_ridge,
    )
    component_shapes = _k2_regular_component_shapes(seq)
    scalar_shape = seq.basis_3.shape[0]
    types = seq.basis_0.types
    g_r = _dense_incidence_1d(seq.basis_0.nr, types[0])
    g_t = _dense_incidence_1d(seq.basis_0.nt, types[1])
    g_z = _dense_incidence_1d(seq.basis_0.nz, types[2])

    mass_r_terms = []
    mass_t_terms = []
    mass_z_terms = []
    component_mass_r_terms = []
    component_mass_t_terms = []
    component_mass_z_terms = []

    for idx in range(rank):
        factor_theta = jnp.ravel(factors[0][:, idx])
        factor_r = jnp.ravel(factors[1][:, idx])
        factor_z = jnp.ravel(factors[2][:, idx])
        scale = weights[idx]

        mass_r = _symmetrize(_assemble_weighted_1d_mass(
            seq.d_basis_r_jk,
            seq.quad.w_x * (scale * factor_r),
        ))
        component_mass_r = _symmetrize(_assemble_weighted_1d_mass(
            seq.basis_r_jk,
            seq.quad.w_x * (scale * factor_r),
        ))
        mass_t = _symmetrize(_assemble_weighted_1d_mass(
            seq.d_basis_t_jk,
            seq.quad.w_y * factor_theta,
        ))
        component_mass_t = _symmetrize(_assemble_weighted_1d_mass(
            seq.basis_t_jk,
            seq.quad.w_y * factor_theta,
        ))
        mass_z = _symmetrize(_assemble_weighted_1d_mass(
            seq.d_basis_z_jk,
            seq.quad.w_z * factor_z,
        ))
        component_mass_z = _symmetrize(_assemble_weighted_1d_mass(
            seq.basis_z_jk,
            seq.quad.w_z * factor_z,
        ))

        mass_r_terms.append(mass_r)
        mass_t_terms.append(mass_t)
        mass_z_terms.append(mass_z)
        component_mass_r_terms.append(component_mass_r)
        component_mass_t_terms.append(component_mass_t)
        component_mass_z_terms.append(component_mass_z)

    return K2TensorDivDivForwardModel(
        r_shape=component_shapes['r'],
        theta_shape=component_shapes['theta'],
        zeta_shape=component_shapes['zeta'],
        scalar_shape=scalar_shape,
        rank=rank,
        g_r=g_r,
        g_t=g_t,
        g_z=g_z,
        mass_r_terms=tuple(mass_r_terms),
        mass_t_terms=tuple(mass_t_terms),
        mass_z_terms=tuple(mass_z_terms),
        component_mass_r_terms=tuple(component_mass_r_terms),
        component_mass_t_terms=tuple(component_mass_t_terms),
        component_mass_z_terms=tuple(component_mass_z_terms),
        cp_relative_error=cp_relative_error,
        cp_final_delta=cp_final_delta,
    )


def _apply_k2_divdiv_regular_tensor_model(
        model: K2TensorDivDivForwardModel,
        rhs: jnp.ndarray) -> jnp.ndarray:
    r_size = _prod3(model.r_shape)
    theta_size = _prod3(model.theta_shape)
    zeta_size = _prod3(model.zeta_shape)
    rhs_r = rhs[:r_size].reshape(model.r_shape)
    rhs_theta = rhs[r_size:r_size + theta_size].reshape(model.theta_shape)
    rhs_zeta = rhs[r_size + theta_size:r_size + theta_size + zeta_size].reshape(model.zeta_shape)

    identity_r = jnp.eye(model.scalar_shape[0], dtype=rhs.dtype)
    identity_t = jnp.eye(model.scalar_shape[1], dtype=rhs.dtype)
    identity_z = jnp.eye(model.scalar_shape[2], dtype=rhs.dtype)

    divergence = _apply_kron3_operators(model.g_r, identity_t, identity_z, rhs_r)
    divergence = divergence + _apply_kron3_operators(identity_r, model.g_t, identity_z, rhs_theta)
    divergence = divergence + _apply_kron3_operators(identity_r, identity_t, model.g_z, rhs_zeta)

    out_r = jnp.zeros(model.r_shape, dtype=rhs.dtype)
    out_theta = jnp.zeros(model.theta_shape, dtype=rhs.dtype)
    out_zeta = jnp.zeros(model.zeta_shape, dtype=rhs.dtype)

    for mass_r, mass_t, mass_z in zip(
            model.mass_r_terms,
            model.mass_t_terms,
            model.mass_z_terms):
        weighted_divergence = _apply_kron3_operators(mass_r, mass_t, mass_z, divergence)
        out_r = out_r + _apply_kron3_operators(model.g_r.T, identity_t, identity_z, weighted_divergence)
        out_theta = out_theta + _apply_kron3_operators(identity_r, model.g_t.T, identity_z, weighted_divergence)
        out_zeta = out_zeta + _apply_kron3_operators(identity_r, identity_t, model.g_z.T, weighted_divergence)

    return jnp.concatenate([
        out_r.reshape(-1),
        out_theta.reshape(-1),
        out_zeta.reshape(-1),
    ])


def _assemble_k2_divdiv_regular_tensor_dense_matrix(
        model: K2TensorDivDivForwardModel) -> jnp.ndarray:
    size = _prod3(model.r_shape) + _prod3(model.theta_shape) + _prod3(model.zeta_shape)
    return _assemble_dense_from_apply(
        lambda x, tensor_model=model: _apply_k2_divdiv_regular_tensor_model(tensor_model, x),
        size,
    )
def assemble_tensor_stiffness_models(
        seq, operators: Optional[SequenceOperators] = None,
        *, ks: Sequence[int] = (2,), rank: int = 1,
        cp_kwargs: Optional[dict] = None):
    """Assemble the stored higher-form tensor stiffness forward models.

    This stores regular-space tensor models for `k = 1` curl-curl and
    `k = 2` div-div on the operator bundle so they can be applied through a
    stable API rather than only via internal debug helpers.
    """
    operators = _ensure_extraction_operators(seq, operators)
    cp_kwargs = {} if cp_kwargs is None else cp_kwargs

    missing_mass_ks = []
    missing_incidence_ks = []
    for k in ks:
        if k not in (1, 2):
            raise ValueError("Tensor stiffness model assembly only supports k=1 and k=2")
        if getattr(operators, f"m{k + 1}") is None:
            missing_mass_ks.append(k + 1)
        if getattr(operators, f"g{k}") is None:
            missing_incidence_ks.append(k)
    if missing_mass_ks:
        operators = assemble_mass_operators(
            seq,
            seq.geometry,
            operators,
            ks=tuple(sorted(set(missing_mass_ks))),
        )
    if missing_incidence_ks:
        operators = assemble_incidence_operators(
            seq,
            operators=operators,
            ks=tuple(sorted(set(missing_incidence_ks))),
        )

    k1_model = operators.k1_tensor_stiff_model
    k2_model = operators.k2_tensor_stiff_model
    for k in ks:
        tensor_rank = _tensor_mass_rank(rank, cp_kwargs, k)
        if k == 1:
            k1_model = _assemble_k1_curlcurl_regular_tensor_model(
                seq,
                rank=tensor_rank,
                cp_maxiter=int(cp_kwargs.get("maxiter", 100)),
                cp_tol=float(cp_kwargs.get("tol", 1e-9)),
                cp_ridge=float(cp_kwargs.get("ridge", 1e-12)),
            )
        else:
            k2_model = _assemble_k2_divdiv_regular_tensor_model(
                seq,
                rank=tensor_rank,
                cp_maxiter=int(cp_kwargs.get("maxiter", 100)),
                cp_tol=float(cp_kwargs.get("tol", 1e-9)),
                cp_ridge=float(cp_kwargs.get("ridge", 1e-12)),
            )
    return eqx.tree_at(
        lambda ops: (ops.k1_tensor_stiff_model, ops.k2_tensor_stiff_model),
        operators,
        (k1_model, k2_model),
        is_leaf=lambda x: x is None,
    )


def tensor_stiffness_model_available(operators: SequenceOperators, k: int) -> bool:
    if k == 1:
        return operators.k1_tensor_stiff_model is not None
    if k == 2:
        return operators.k2_tensor_stiff_model is not None
    return False


def apply_stiffness_tensor_forward_model(
        seq, operators: SequenceOperators, v, k: int,
        dirichlet: bool = True, *, regular_space: bool = False):
    """Apply the stored tensor stiffness forward model for `k = 1` or `k = 2`.

    By default this mirrors :func:`apply_stiffness` on the extracted space.
    Set `regular_space=True` to apply the regular-space tensor model directly.
    """
    if k == 1:
        model = operators.k1_tensor_stiff_model
        if model is None:
            raise ValueError(
                "Tensor stiffness model for k=1 is not assembled; "
                "call assemble_tensor_stiffness_models(seq, operators, ks=(1,)) first"
            )
        if regular_space:
            return _apply_k1_curlcurl_regular_tensor_model(model, v)
        return _apply_k1_curlcurl_extracted_tensor_model(
            operators,
            model,
            v,
            dirichlet=dirichlet,
        )
    if k == 2:
        model = operators.k2_tensor_stiff_model
        if model is None:
            raise ValueError(
                "Tensor stiffness model for k=2 is not assembled; "
                "call assemble_tensor_stiffness_models(seq, operators, ks=(2,)) first"
            )
        if regular_space:
            return _apply_k2_divdiv_regular_tensor_model(model, v)
        return _apply_k2_divdiv_extracted_tensor_model(
            operators,
            model,
            v,
            dirichlet=dirichlet,
        )
    raise ValueError("Tensor stiffness forward model only supports k=1 and k=2")


def _stiffness_axis_from_mass_term(
        mass_term: jnp.ndarray,
        incidence: jnp.ndarray) -> jnp.ndarray:
    return _symmetrize(incidence.T @ (mass_term @ incidence))


def _safe_diaginv(diagonal: jnp.ndarray) -> jnp.ndarray:
    diagonal = jnp.asarray(diagonal, dtype=jnp.float64)
    return jnp.where(jnp.abs(diagonal) > 0.0, 1.0 / diagonal, 0.0)


def _build_extracted_stiffness_apply_data(
        seq,
        operators: SequenceOperators,
        *,
        k: int,
        dirichlet: bool):
    if k not in (1, 2):
        raise ValueError("Extracted stiffness apply data is only implemented for k=1 and k=2")
    g_sp, g_sp_t = _incidence_components(operators, k)
    m_sp, _, _ = _mass_components(operators, k + 1)
    if g_sp is None or g_sp_t is None:
        raise ValueError(f"Incidence operator G{k} is required for stiffness k={k}")
    if m_sp is None:
        raise ValueError(f"Mass operator M{k + 1} is required for stiffness k={k}")
    return _build_extracted_mass_apply_data(
        seq,
        _ComposedStiffnessMatvec(g=g_sp, g_t=g_sp_t, m_next=m_sp),
        k,
        dirichlet,
    )


def _build_k1_stiffness_surgery_factors(
        seq,
        operators: SequenceOperators,
        *,
        dirichlet: bool,
        precompute_coupling: bool = True) -> K1MassSurgeryPreconditionerFactors:
    block_indices = _tensor_block_indices_k1(seq, dirichlet)
    apply_data = _build_extracted_stiffness_apply_data(
        seq,
        operators,
        k=1,
        dirichlet=dirichlet,
    )
    surgery_indices = block_indices["surgery"]
    bulk_indices = block_indices["bulk"]
    surgery_size = int(surgery_indices.shape[0])
    ass = _symmetrize(_assemble_dense_from_apply(
        lambda x, apply_data=apply_data, idx=surgery_indices: _apply_extracted_submatrix(
            apply_data,
            idx,
            idx,
            x,
        ),
        surgery_size,
    ))
    # Precompute the dense surgery->bulk coupling block C (bulk x surgery) once,
    # so the per-apply surgery couplings become dense matvecs (C @ / C.T @,
    # extracted curl-curl is symmetric) instead of a full matrix-free apply of
    # the extracted operator (O(n^3 p^6) from the M_2 mass apply). The surgery
    # space is the polar axis (small), so the block is cheap to store/probe.
    coupling_sb = None
    if precompute_coupling:
        coupling_sb = _assemble_dense_from_apply(
            lambda x, apply_data=apply_data, rows=bulk_indices, cols=surgery_indices:
            _apply_extracted_submatrix(apply_data, rows, cols, x),
            surgery_size,
            sequential=True,
        )
    return K1MassSurgeryPreconditionerFactors(
        surgery_indices=surgery_indices,
        bulk_indices=bulk_indices,
        r_indices=block_indices["r"],
        theta_bulk_indices=block_indices["theta_bulk"],
        zeta_bulk_indices=block_indices["zeta_bulk"],
        rt_indices=block_indices["rt"],
        surgery_size=surgery_size,
        rt_r_size=int(block_indices["rt_r_size"]),
        rt_theta_size=int(block_indices["rt_theta_size"]),
        bulk_rt_size=int(block_indices["bulk_rt_size"]),
        bulk_zeta_size=int(block_indices["bulk_zeta_size"]),
        apply_data=apply_data,
        surgery_diaginv=_safe_diaginv(jnp.diag(ass)),
        ass=ass,
        coupling_sb=coupling_sb,
    )


def _build_k2_stiffness_surgery_factors(
        seq,
        operators: SequenceOperators,
        *,
        dirichlet: bool) -> K2MassSurgeryPreconditionerFactors:
    block_indices = _tensor_block_indices_k2(seq, dirichlet)
    apply_data = _build_extracted_stiffness_apply_data(
        seq,
        operators,
        k=2,
        dirichlet=dirichlet,
    )
    surgery_indices = block_indices["surgery"]
    surgery_size = int(surgery_indices.shape[0])
    ass = _symmetrize(_assemble_dense_from_apply(
        lambda x, apply_data=apply_data, idx=surgery_indices: _apply_extracted_submatrix(
            apply_data,
            idx,
            idx,
            x,
        ),
        surgery_size,
    ))
    return K2MassSurgeryPreconditionerFactors(
        surgery_indices=surgery_indices,
        bulk_indices=block_indices["bulk"],
        r_bulk_indices=block_indices["r_bulk"],
        theta_indices=block_indices["theta"],
        zeta_indices=block_indices["zeta"],
        surgery_size=surgery_size,
        r_bulk_size=int(block_indices["r_bulk_size"]),
        theta_size=int(block_indices["theta_size"]),
        zeta_size=int(block_indices["zeta_size"]),
        apply_data=apply_data,
        surgery_diaginv=_safe_diaginv(jnp.diag(ass)),
        ass=ass,
    )


def assemble_tensor_stiffness_preconditioner(
        seq,
        operators: Optional[SequenceOperators] = None,
        *,
        ks: Sequence[int] = (1, 2),
        rank: int = 1,
        cp_kwargs: Optional[dict] = None):
    """Assemble standalone tensor stiffness preconditioners for `k = 1, 2`.

    These are preconditioners for the semidefinite stiffness blocks
    `curl-curl` and `div-div` themselves. They are intentionally kept
    separate from the mixed saddle-point Hodge-Laplacian path.
    """
    operators = _ensure_extraction_operators(seq, operators)
    cp_kwargs = {} if cp_kwargs is None else dict(cp_kwargs)
    cp_maxiter = int(cp_kwargs.get("maxiter", 100))
    cp_tol = float(cp_kwargs.get("tol", 1e-9))
    cp_ridge = float(cp_kwargs.get("ridge", 1e-12))
    precompute_coupling = bool(cp_kwargs.get("precompute_coupling", True))
    surgery_schur_pinv_tol = float(
        cp_kwargs.get("surgery_schur_pinv_tol", cp_kwargs.get("schur_pinv_tol", 1e-8))
    )
    bulk_block_pinv_tol = float(cp_kwargs.get("bulk_block_pinv_tol", 1e-8))
    bulk_schur = bool(cp_kwargs.get("bulk_schur", False))
    greville = bool(cp_kwargs.get("greville", False))  # greville P_A stiffness atom (opt-in)

    operators = assemble_tensor_stiffness_models(
        seq,
        operators=operators,
        ks=ks,
        rank=rank,
        cp_kwargs=cp_kwargs,
    )

    missing_mass = []
    missing_incidence = []
    for k in ks:
        if k not in (1, 2):
            raise ValueError("Tensor stiffness preconditioner assembly only supports k=1 and k=2")
        if getattr(operators, f"m{k + 1}") is None:
            missing_mass.append(k + 1)
        if _incidence_components(operators, k)[0] is None:
            missing_incidence.append(k)
    if missing_mass:
        operators = assemble_mass_operators(
            seq,
            seq.geometry,
            operators,
            ks=tuple(sorted(set(missing_mass))),
        )
    if missing_incidence:
        operators = assemble_incidence_operators(
            seq,
            operators=operators,
            ks=tuple(sorted(set(missing_incidence))),
        )

    k1_pair = operators.k1_tensor_stiff_precond or BoundaryConditionPair()
    k2_pair = operators.k2_tensor_stiff_precond or BoundaryConditionPair()

    for k in ks:
        tensor_rank = _tensor_mass_rank(rank, cp_kwargs, k)
        if k == 1:
            model = operators.k1_tensor_stiff_model
            if model is None:
                raise ValueError("Tensor stiffness model k=1 is not assembled")
            pair = k1_pair
            for dirichlet in (False, True):
                surgery = _build_k1_stiffness_surgery_factors(
                    seq,
                    operators,
                    dirichlet=dirichlet,
                    precompute_coupling=precompute_coupling,
                )
                arr_shape = _arr_shape_k1(seq, dirichlet)
                theta_shape = _theta_bulk_shape_k1(seq, dirichlet)
                zeta_shape = _zeta_bulk_shape_k1(seq, dirichlet)

                # MRX_K1_ATOM=profile: the exact k=0-fdbund analog. Own-axis
                # mean profiles of each beta channel go into the K of the term
                # they weight; ALL masses stay unweighted. Per axis the block
                # then has one clean pencil (M, K[p]) -> the modal diagonals
                # below are EXACT (no chopping), the block inverse is exact
                # Lynch, and the analytic block null (own-axis modes) is
                # zeroed by the modal denom guard instead of floor-amplified.
                _k1_atom_env = os.environ.get("MRX_K1_ATOM", "bundled")
                k1_profile = _k1_atom_env == "profile"
                # MRX_K1_ATOM=rank1: full rank-1 weights on EVERY axis,
                # inverted EXACTLY. A two-term block carries exactly two 1D
                # matrices per axis, and any SPD pair is simultaneously
                # diagonalizable by its generalized pencil (mass-mass pairs
                # included) -> pass each axis's two term matrices as the
                # (reference, operator) pencil and the generic modal
                # diagonals become exact. Only valid at rank 1 (2 terms).
                k1_rank1 = _k1_atom_env == "rank1"
                if k1_profile:
                    _mt = _k2_diagonal_metric_tensors(seq)
                    _prof = {ch: _bundled_rank1_mass_factors(seq, _mt[ch])
                             for ch in ("beta_rr", "beta_thetatheta", "beta_zetazeta")}
                    # _prof[ch] = (scale, f_theta, f_r, f_zeta, rel_err); raw means.

                    def _wk_t(f):
                        return _assemble_weighted_1d_stiffness(
                            seq.basis_t_jk, seq.d_basis_t_jk, seq.quad.w_y * f, model.g_t)

                    def _wk_z(f):
                        return _assemble_weighted_1d_stiffness(
                            seq.basis_z_jk, seq.d_basis_z_jk, seq.quad.w_z * f, model.g_z)

                    def _wk_r(f):
                        return _assemble_weighted_1d_stiffness(
                            seq.basis_r_jk, seq.d_basis_r_jk, seq.quad.w_x * f, model.g_r)

                    Kt_zz = _wk_t(_prof["beta_zetazeta"][1])
                    Kz_tt = _wk_z(_prof["beta_thetatheta"][3])
                    Kz_rr = _wk_z(_prof["beta_rr"][3])
                    Kr_zz = _wk_r(_prof["beta_zetazeta"][2])
                    Kt_rr = _wk_t(_prof["beta_rr"][1])
                    Kr_tt = _wk_r(_prof["beta_thetatheta"][2])

                arr_op_r_pencil = None
                arr_true_apply = lambda x, surgery=surgery: _apply_extracted_submatrix(
                    surgery.apply_data, surgery.r_indices, surgery.r_indices, x)
                theta_op_t_pencil = None
                theta_true_apply = lambda x, surgery=surgery: _apply_extracted_submatrix(
                    surgery.apply_data, surgery.theta_bulk_indices, surgery.theta_bulk_indices, x)
                zeta_op_z_pencil = None
                zeta_true_apply = lambda x, surgery=surgery: _apply_extracted_submatrix(
                    surgery.apply_data, surgery.zeta_bulk_indices, surgery.zeta_bulk_indices, x)

                full_stiff_r = _assemble_weighted_1d_stiffness(
                    seq.basis_r_jk,
                    seq.d_basis_r_jk,
                    seq.quad.w_x,
                    model.g_r,
                )
                stiff_t = _assemble_weighted_1d_stiffness(
                    seq.basis_t_jk,
                    seq.d_basis_t_jk,
                    seq.quad.w_y,
                    model.g_t,
                )
                stiff_z = _assemble_weighted_1d_stiffness(
                    seq.basis_z_jk,
                    seq.d_basis_z_jk,
                    seq.quad.w_z,
                    model.g_z,
                )

                arr_terms = []
                for mass_r, mass_t, mass_z in zip(model.tt_mass_r_terms, model.tt_mass_t_terms, model.tt_mass_z_terms):
                    arr_terms.append((
                        _restrict_radial_mass(mass_r, 1, arr_shape[0]),
                        mass_t,
                        _stiffness_axis_from_mass_term(mass_z, model.g_z),
                    ))
                for mass_r, mass_t, mass_z in zip(model.zz_mass_r_terms, model.zz_mass_t_terms, model.zz_mass_z_terms):
                    arr_terms.append((
                        _restrict_radial_mass(mass_r, 1, arr_shape[0]),
                        _stiffness_axis_from_mass_term(mass_t, model.g_t),
                        mass_z,
                    ))
                arr_ref_r = _restrict_radial_mass(
                    _assemble_unweighted_1d_mass(seq.d_basis_r_jk, seq.quad.w_x),
                    1,
                    arr_shape[0],
                )
                arr_ref_t = _assemble_unweighted_1d_mass(seq.basis_t_jk, seq.quad.w_y)
                arr_ref_z = _assemble_unweighted_1d_mass(seq.basis_z_jk, seq.quad.w_z)
                arr_op_t = stiff_t
                arr_op_z = stiff_z
                if k1_profile:
                    arr_terms = [
                        (arr_ref_r, Kt_zz, arr_ref_z),
                        (arr_ref_r, arr_ref_t, Kz_tt),
                    ]
                    arr_op_t, arr_op_z = Kt_zz, Kz_tt
                if k1_rank1:
                    if len(arr_terms) != 2:
                        raise ValueError("MRX_K1_ATOM=rank1 requires rank-1 channel fits (2 terms/block)")
                    # terms: [0] = tt-channel (M_r, M_t, K_z), [1] = zz-channel (M_r, K_t, M_z)
                    arr_ref_r, arr_op_r_pencil = arr_terms[0][0], arr_terms[1][0]
                    arr_ref_t, arr_op_t = arr_terms[0][1], arr_terms[1][1]
                    arr_ref_z, arr_op_z = arr_terms[1][2], arr_terms[0][2]
                arr_factors = _build_greville_stiffness_block_factors(
                    seq, k=1, shape=arr_shape, diff=(True, False, False), comp=0,
                ) if greville else _build_mass_referenced_tensor_block_factors(
                    full_shape=arr_shape,
                    reference_r=arr_ref_r,
                    reference_t=arr_ref_t,
                    reference_z=arr_ref_z,
                    axis_operator_r=arr_op_r_pencil if k1_rank1 else None,
                    axis_operator_t=arr_op_t,
                    axis_operator_z=arr_op_z,
                    term_matrices=tuple(arr_terms),
                    cp_relative_error=model.cp_relative_error,
                    cp_final_delta=model.cp_final_delta,
                    modal_pinv_tol=bulk_block_pinv_tol,
                    true_block_apply=arr_true_apply,
                )

                theta_terms = []
                for mass_r, mass_t, mass_z in zip(model.rr_mass_r_terms, model.rr_mass_t_terms, model.rr_mass_z_terms):
                    theta_terms.append((
                        _restrict_radial_mass(mass_r, 2, theta_shape[0]),
                        mass_t,
                        _stiffness_axis_from_mass_term(mass_z, model.g_z),
                    ))
                for mass_r, mass_t, mass_z in zip(model.zz_mass_r_terms, model.zz_mass_t_terms, model.zz_mass_z_terms):
                    theta_terms.append((
                        _restrict_radial_mass(_stiffness_axis_from_mass_term(mass_r, model.g_r), 2, theta_shape[0]),
                        mass_t,
                        mass_z,
                    ))
                theta_ref_r = _restrict_radial_mass(
                    _assemble_unweighted_1d_mass(seq.basis_r_jk, seq.quad.w_x),
                    2,
                    theta_shape[0],
                )
                theta_ref_t = _assemble_unweighted_1d_mass(seq.d_basis_t_jk, seq.quad.w_y)
                theta_ref_z = _assemble_unweighted_1d_mass(seq.basis_z_jk, seq.quad.w_z)
                theta_op_r = _restrict_radial_mass(full_stiff_r, 2, theta_shape[0])
                theta_op_z = stiff_z
                if k1_profile:
                    _Kr_zz_w = _restrict_radial_mass(Kr_zz, 2, theta_shape[0])
                    theta_terms = [
                        (theta_ref_r, theta_ref_t, Kz_rr),
                        (_Kr_zz_w, theta_ref_t, theta_ref_z),
                    ]
                    theta_op_r, theta_op_z = _Kr_zz_w, Kz_rr
                if k1_rank1:
                    # terms: [0] = rr-channel (M_r, M_t, K_z), [1] = zz-channel (K_r, M_t, M_z)
                    theta_ref_r, theta_op_r = theta_terms[0][0], theta_terms[1][0]
                    theta_ref_t, theta_op_t_pencil = theta_terms[0][1], theta_terms[1][1]
                    theta_ref_z, theta_op_z = theta_terms[1][2], theta_terms[0][2]
                theta_factors = _build_greville_stiffness_block_factors(
                    seq, k=1, shape=theta_shape, diff=(False, True, False), comp=1,
                ) if greville else _build_mass_referenced_tensor_block_factors(
                    full_shape=theta_shape,
                    reference_r=theta_ref_r,
                    reference_t=theta_ref_t,
                    reference_z=theta_ref_z,
                    axis_operator_r=theta_op_r,
                    axis_operator_t=theta_op_t_pencil if k1_rank1 else None,
                    axis_operator_z=theta_op_z,
                    term_matrices=tuple(theta_terms),
                    cp_relative_error=model.cp_relative_error,
                    cp_final_delta=model.cp_final_delta,
                    modal_pinv_tol=bulk_block_pinv_tol,
                    true_block_apply=theta_true_apply,
                )

                zeta_terms = []
                for mass_r, mass_t, mass_z in zip(model.rr_mass_r_terms, model.rr_mass_t_terms, model.rr_mass_z_terms):
                    zeta_terms.append((
                        _restrict_radial_mass(mass_r, 2, zeta_shape[0]),
                        _stiffness_axis_from_mass_term(mass_t, model.g_t),
                        mass_z,
                    ))
                for mass_r, mass_t, mass_z in zip(model.tt_mass_r_terms, model.tt_mass_t_terms, model.tt_mass_z_terms):
                    zeta_terms.append((
                        _restrict_radial_mass(_stiffness_axis_from_mass_term(mass_r, model.g_r), 2, zeta_shape[0]),
                        mass_t,
                        mass_z,
                    ))
                zeta_ref_r = _restrict_radial_mass(
                    _assemble_unweighted_1d_mass(seq.basis_r_jk, seq.quad.w_x),
                    2,
                    zeta_shape[0],
                )
                zeta_ref_t = _assemble_unweighted_1d_mass(seq.basis_t_jk, seq.quad.w_y)
                zeta_ref_z = _assemble_unweighted_1d_mass(seq.d_basis_z_jk, seq.quad.w_z)
                zeta_op_r = _restrict_radial_mass(full_stiff_r, 2, zeta_shape[0])
                zeta_op_t = stiff_t
                if k1_profile:
                    _Kr_tt_w = _restrict_radial_mass(Kr_tt, 2, zeta_shape[0])
                    zeta_terms = [
                        (zeta_ref_r, Kt_rr, zeta_ref_z),
                        (_Kr_tt_w, zeta_ref_t, zeta_ref_z),
                    ]
                    zeta_op_r, zeta_op_t = _Kr_tt_w, Kt_rr
                if k1_rank1:
                    # terms: [0] = rr-channel (M_r, K_t, M_z), [1] = tt-channel (K_r, M_t, M_z)
                    zeta_ref_r, zeta_op_r = zeta_terms[0][0], zeta_terms[1][0]
                    zeta_ref_t, zeta_op_t = zeta_terms[1][1], zeta_terms[0][1]
                    zeta_ref_z, zeta_op_z_pencil = zeta_terms[0][2], zeta_terms[1][2]
                zeta_factors = _build_greville_stiffness_block_factors(
                    seq, k=1, shape=zeta_shape, diff=(False, False, True), comp=2,
                ) if greville else _build_mass_referenced_tensor_block_factors(
                    full_shape=zeta_shape,
                    reference_r=zeta_ref_r,
                    reference_t=zeta_ref_t,
                    reference_z=zeta_ref_z,
                    axis_operator_r=zeta_op_r,
                    axis_operator_t=zeta_op_t,
                    axis_operator_z=zeta_op_z_pencil if k1_rank1 else None,
                    term_matrices=tuple(zeta_terms),
                    cp_relative_error=model.cp_relative_error,
                    cp_final_delta=model.cp_final_delta,
                    modal_pinv_tol=bulk_block_pinv_tol,
                    true_block_apply=zeta_true_apply,
                )

                bulk_apply = (
                    lambda rhs_bulk, surgery=surgery, arr_factors=arr_factors, theta_factors=theta_factors, zeta_factors=zeta_factors:
                    _apply_k1_bulk_preconditioner(surgery, arr_factors, theta_factors, zeta_factors, rhs_bulk)
                ) if bulk_schur else (
                    lambda rhs_bulk, surgery=surgery, arr_factors=arr_factors, theta_factors=theta_factors, zeta_factors=zeta_factors:
                    _apply_k1_bulk_diagonal_preconditioner(surgery, arr_factors, theta_factors, zeta_factors, rhs_bulk)
                )
                schur_inv = _assemble_surgery_schur_inverse_from_applies(
                    surgery.ass,
                    lambda rhs_s, surgery=surgery: _apply_surgery_to_bulk_coupling(surgery, rhs_s),
                    bulk_apply,
                    lambda rhs_b, surgery=surgery: _apply_bulk_to_surgery_coupling(surgery, rhs_b),
                    relative_tol=surgery_schur_pinv_tol,
                )

                payload = K1TensorStiffnessPreconditioner(
                    surgery=surgery,
                    factors=K1TensorMassPreconditionerFactors(
                        r_indices=surgery.r_indices,
                        theta_bulk_indices=surgery.theta_bulk_indices,
                        zeta_bulk_indices=surgery.zeta_bulk_indices,
                        rt_r_size=surgery.rt_r_size,
                        rt_theta_size=surgery.rt_theta_size,
                        bulk_schur=bulk_schur,
                        arr=arr_factors,
                        theta=theta_factors,
                        zeta=zeta_factors,
                        schur_inv=schur_inv,
                    ),
                )
                pair = eqx.tree_at(
                    lambda boundary_pair: boundary_pair.dbc if dirichlet else boundary_pair.free,
                    pair,
                    payload,
                    is_leaf=lambda x: x is None,
                )
            k1_pair = pair
            continue

        model = operators.k2_tensor_stiff_model
        if model is None:
            raise ValueError("Tensor stiffness model k=2 is not assembled")
        pair = k2_pair
        for dirichlet in (False, True):
            surgery = _build_k2_stiffness_surgery_factors(
                seq,
                operators,
                dirichlet=dirichlet,
            )
            r_bulk_shape = _r_bulk_shape_k2(seq, dirichlet)
            theta_shape = _theta_shape_k2(seq, dirichlet)
            zeta_shape = _zeta_shape_k2(seq, dirichlet)

            r_bulk_true_apply = lambda x, surgery=surgery: _apply_extracted_submatrix(
                surgery.apply_data, surgery.r_bulk_indices, surgery.r_bulk_indices, x)
            theta_true_apply = lambda x, surgery=surgery: _apply_extracted_submatrix(
                surgery.apply_data, surgery.theta_indices, surgery.theta_indices, x)
            zeta_true_apply = lambda x, surgery=surgery: _apply_extracted_submatrix(
                surgery.apply_data, surgery.zeta_indices, surgery.zeta_indices, x)

            full_stiff_r = _assemble_weighted_1d_stiffness(
                seq.basis_r_jk,
                seq.d_basis_r_jk,
                seq.quad.w_x,
                model.g_r,
            )
            stiff_t = _assemble_weighted_1d_stiffness(
                seq.basis_t_jk,
                seq.d_basis_t_jk,
                seq.quad.w_y,
                model.g_t,
            )
            stiff_z = _assemble_weighted_1d_stiffness(
                seq.basis_z_jk,
                seq.d_basis_z_jk,
                seq.quad.w_z,
                model.g_z,
            )

            r_bulk_terms = tuple(
                (
                    _restrict_radial_mass(_stiffness_axis_from_mass_term(mass_r, model.g_r), 2, r_bulk_shape[0]),
                    mass_t,
                    mass_z,
                )
                for mass_r, mass_t, mass_z in zip(model.mass_r_terms, model.mass_t_terms, model.mass_z_terms)
            )
            r_bulk_ref_r = _restrict_radial_mass(
                _assemble_unweighted_1d_mass(seq.basis_r_jk, seq.quad.w_x),
                2,
                r_bulk_shape[0],
            )
            r_bulk_ref_t = _assemble_unweighted_1d_mass(seq.d_basis_t_jk, seq.quad.w_y)
            r_bulk_ref_z = _assemble_unweighted_1d_mass(seq.d_basis_z_jk, seq.quad.w_z)
            r_bulk_op_r = _restrict_radial_mass(full_stiff_r, 2, r_bulk_shape[0])
            r_bulk_factors = _build_greville_stiffness_block_factors(
                seq, k=2, shape=r_bulk_shape, diff=(False, True, True), comp=0,
            ) if greville else _build_mass_referenced_tensor_block_factors(
                full_shape=r_bulk_shape,
                reference_r=r_bulk_ref_r,
                reference_t=r_bulk_ref_t,
                reference_z=r_bulk_ref_z,
                axis_operator_r=r_bulk_op_r,
                axis_operator_t=None,
                axis_operator_z=None,
                term_matrices=r_bulk_terms,
                cp_relative_error=model.cp_relative_error,
                cp_final_delta=model.cp_final_delta,
                modal_pinv_tol=bulk_block_pinv_tol,
                true_block_apply=r_bulk_true_apply,
            )

            theta_terms = tuple(
                (
                    _restrict_radial_mass(mass_r, 1, theta_shape[0]),
                    _stiffness_axis_from_mass_term(mass_t, model.g_t),
                    mass_z,
                )
                for mass_r, mass_t, mass_z in zip(model.mass_r_terms, model.mass_t_terms, model.mass_z_terms)
            )
            theta_ref_r = _restrict_radial_mass(
                _assemble_unweighted_1d_mass(seq.d_basis_r_jk, seq.quad.w_x),
                1,
                theta_shape[0],
            )
            theta_ref_t = _assemble_unweighted_1d_mass(seq.basis_t_jk, seq.quad.w_y)
            theta_ref_z = _assemble_unweighted_1d_mass(seq.d_basis_z_jk, seq.quad.w_z)
            theta_op_t = stiff_t
            theta_factors = _build_greville_stiffness_block_factors(
                seq, k=2, shape=theta_shape, diff=(True, False, True), comp=1,
            ) if greville else _build_mass_referenced_tensor_block_factors(
                full_shape=theta_shape,
                reference_r=theta_ref_r,
                reference_t=theta_ref_t,
                reference_z=theta_ref_z,
                axis_operator_r=None,
                axis_operator_t=theta_op_t,
                axis_operator_z=None,
                term_matrices=theta_terms,
                cp_relative_error=model.cp_relative_error,
                cp_final_delta=model.cp_final_delta,
                modal_pinv_tol=bulk_block_pinv_tol,
                true_block_apply=theta_true_apply,
            )

            zeta_terms = tuple(
                (
                    _restrict_radial_mass(mass_r, 1, zeta_shape[0]),
                    mass_t,
                    _stiffness_axis_from_mass_term(mass_z, model.g_z),
                )
                for mass_r, mass_t, mass_z in zip(model.mass_r_terms, model.mass_t_terms, model.mass_z_terms)
            )
            zeta_ref_r = _restrict_radial_mass(
                _assemble_unweighted_1d_mass(seq.d_basis_r_jk, seq.quad.w_x),
                1,
                zeta_shape[0],
            )
            zeta_ref_t = _assemble_unweighted_1d_mass(seq.d_basis_t_jk, seq.quad.w_y)
            zeta_ref_z = _assemble_unweighted_1d_mass(seq.basis_z_jk, seq.quad.w_z)
            zeta_op_z = stiff_z
            zeta_factors = _build_greville_stiffness_block_factors(
                seq, k=2, shape=zeta_shape, diff=(True, True, False), comp=2,
            ) if greville else _build_mass_referenced_tensor_block_factors(
                full_shape=zeta_shape,
                reference_r=zeta_ref_r,
                reference_t=zeta_ref_t,
                reference_z=zeta_ref_z,
                axis_operator_r=None,
                axis_operator_t=None,
                axis_operator_z=zeta_op_z,
                term_matrices=zeta_terms,
                cp_relative_error=model.cp_relative_error,
                cp_final_delta=model.cp_final_delta,
                modal_pinv_tol=bulk_block_pinv_tol,
                true_block_apply=zeta_true_apply,
            )

            bulk_apply = (
                lambda rhs_bulk, surgery=surgery, r_bulk_factors=r_bulk_factors, theta_factors=theta_factors, zeta_factors=zeta_factors:
                _apply_k2_bulk_preconditioner(surgery, r_bulk_factors, theta_factors, zeta_factors, rhs_bulk)
            ) if bulk_schur else (
                lambda rhs_bulk, surgery=surgery, r_bulk_factors=r_bulk_factors, theta_factors=theta_factors, zeta_factors=zeta_factors:
                _apply_k2_bulk_diagonal_preconditioner(surgery, r_bulk_factors, theta_factors, zeta_factors, rhs_bulk)
            )
            schur_inv = _assemble_surgery_schur_inverse_from_applies(
                surgery.ass,
                lambda rhs_s, surgery=surgery: _apply_surgery_to_bulk_coupling(surgery, rhs_s),
                bulk_apply,
                lambda rhs_b, surgery=surgery: _apply_bulk_to_surgery_coupling(surgery, rhs_b),
                relative_tol=surgery_schur_pinv_tol,
            )

            payload = K2TensorStiffnessPreconditioner(
                surgery=surgery,
                factors=K2TensorMassPreconditionerFactors(
                    r_bulk_indices=surgery.r_bulk_indices,
                    theta_indices=surgery.theta_indices,
                    zeta_indices=surgery.zeta_indices,
                    r_bulk_size=surgery.r_bulk_size,
                    theta_size=surgery.theta_size,
                    zeta_size=surgery.zeta_size,
                    bulk_schur=bulk_schur,
                    r_bulk=r_bulk_factors,
                    theta=theta_factors,
                    zeta=zeta_factors,
                    schur_inv=schur_inv,
                ),
            )
            pair = eqx.tree_at(
                lambda boundary_pair: boundary_pair.dbc if dirichlet else boundary_pair.free,
                pair,
                payload,
                is_leaf=lambda x: x is None,
            )
        k2_pair = pair

    return eqx.tree_at(
        lambda ops: (ops.k1_tensor_stiff_precond, ops.k2_tensor_stiff_precond),
        operators,
        (k1_pair, k2_pair),
        is_leaf=lambda x: x is None,
    )


def stiffness_tensor_preconditioner_available(
        operators: SequenceOperators,
        k: int) -> bool:
    if k == 1:
        pair = operators.k1_tensor_stiff_precond
    elif k == 2:
        pair = operators.k2_tensor_stiff_precond
    else:
        return False
    return pair is not None and pair.free is not None and pair.dbc is not None


def apply_stiffness_tensor_preconditioner(
        seq,
        operators: SequenceOperators,
        v,
        k: int,
        dirichlet: bool = True):
    del seq
    if k == 2:
        pair = operators.k2_tensor_stiff_precond
        if pair is None:
            raise ValueError(
                "Tensor stiffness preconditioner for k=2 is not assembled; "
                "call assemble_tensor_stiffness_preconditioner(seq, operators, ks=(2,)) first"
            )
        payload = select_boundary_data(pair, dirichlet, "Tensor stiffness k=2")
        surgery = payload.surgery
        factors = payload.factors
        rhs_s = v[surgery.surgery_indices]
        rhs_b = v[surgery.bulk_indices]
        bulk_apply = _apply_k2_bulk_preconditioner if factors.bulk_schur else _apply_k2_bulk_diagonal_preconditioner
        y = bulk_apply(surgery, factors.r_bulk, factors.theta, factors.zeta, rhs_b)
        z = factors.schur_inv @ (rhs_s - _apply_bulk_to_surgery_coupling(surgery, y))
        x_b = y - bulk_apply(
            surgery,
            factors.r_bulk,
            factors.theta,
            factors.zeta,
            _apply_surgery_to_bulk_coupling(surgery, z),
        )
        x = jnp.zeros_like(v)
        x = x.at[surgery.surgery_indices].set(z)
        x = x.at[surgery.bulk_indices].set(x_b)
        return x
    if k == 1:
        pair = operators.k1_tensor_stiff_precond
        if pair is None:
            raise ValueError(
                "Tensor stiffness preconditioner for k=1 is not assembled; "
                "call assemble_tensor_stiffness_preconditioner(seq, operators, ks=(1,)) first"
            )
        payload = select_boundary_data(pair, dirichlet, "Tensor stiffness k=1")
        surgery = payload.surgery
        factors = payload.factors
        rhs_s = v[surgery.surgery_indices]
        rhs_b = v[surgery.bulk_indices]
        bulk_apply = _apply_k1_bulk_preconditioner if factors.bulk_schur else _apply_k1_bulk_diagonal_preconditioner
        y = bulk_apply(surgery, factors.arr, factors.theta, factors.zeta, rhs_b)
        z = factors.schur_inv @ (rhs_s - _apply_bulk_to_surgery_coupling(surgery, y))
        x_b = y - bulk_apply(
            surgery,
            factors.arr,
            factors.theta,
            factors.zeta,
            _apply_surgery_to_bulk_coupling(surgery, z),
        )
        x = jnp.zeros_like(v)
        x = x.at[surgery.surgery_indices].set(z)
        x = x.at[surgery.bulk_indices].set(x_b)
        return x
    raise ValueError("Tensor stiffness preconditioner only supports k=1 and k=2")


def _apply_k2_divdiv_regular_forward(
        operators: SequenceOperators,
        rhs: jnp.ndarray) -> jnp.ndarray:
    g2, g2_T = _incidence_components(operators, 2)
    m3, _, _ = _mass_components(operators, 3)
    if g2 is None or g2_T is None:
        raise ValueError("Incidence operator G2 is required for regular-space div-div apply")
    if m3 is None:
        raise ValueError("Mass operator M3 is required for regular-space div-div apply")
    return g2_T @ (m3 @ (g2 @ rhs))


def _apply_k2_divdiv_extracted_tensor_model(
        operators: SequenceOperators,
        model: K2TensorDivDivForwardModel,
        rhs: jnp.ndarray,
        *,
        dirichlet: bool = True) -> jnp.ndarray:
    e2, e2_T = _mass_extraction(operators, 2, dirichlet)
    if e2 is None or e2_T is None:
        side = "dbc" if dirichlet else "free"
        raise ValueError(f"Extraction operator E2 is required for extracted {side} k=2 tensor apply")
    return e2 @ _apply_k2_divdiv_regular_tensor_model(model, e2_T @ rhs)

def _build_greville_stiffness_block_factors(
    seq, *, k: int, shape, diff, comp: int,
) -> TensorDiagonalBlockInverseFactors:
    """Greville P_A: greville-collocation k=1 curl-curl / k=2 div-div bulk block.

    Mirrors the greville mass atom but for a stiffness block, which is an ADDITIVE-FD
    form (D^{-1/2} V diag(1/denom) V^T D^{-1/2}) rather than a pure product. UNWEIGHTED
    1D atoms; the metric weight is collocated as the pointwise D sandwich.

    The PRIMAL (non-differentiated) axes carry the stiffness; the differentiated axis
    is the de Rham "form" direction (mass only). So:
      - k=2 div-div: ONE stiff axis (= comp), weight D = 1/J, alpha = 1.
      - k=1 curl-curl: TWO stiff axes b,c; CROSS-weighted (curl structure) K_b <- channel
        c, K_c <- channel b; common D = sqrt(beta_cc * beta_bb), arithmetic alpha_b =
        mean(beta_cc/D), alpha_c = mean(beta_bb/D), with beta_aa = g_aa/J.
    The 1D stiffness atoms are singular (constant null); the modal denom deflation
    (`_modal_regularized_inverse_denom`) zeros those modes (surgery-corrected).
    """
    from mrx.geometry import compute_geometry_terms  # noqa: PLC0415
    from mrx.spline_bases import SplineBasis  # noqa: PLC0415
    from mrx.operators import (  # noqa: PLC0415
        _assemble_weighted_1d_mass as _m1d,
        _assemble_weighted_1d_stiffness as _k1d,
        _dense_incidence_1d,
    )

    nr, ntc, nzc = (int(s) for s in shape)
    dims = (nr, ntc, nzc)
    radial_start = 1 if diff[0] else 2
    primal = (seq.basis_r_jk, seq.basis_t_jk, seq.basis_z_jk)
    deriv = (seq.d_basis_r_jk, seq.d_basis_t_jk, seq.d_basis_z_jk)
    quad_w = (seq.quad.w_x, seq.quad.w_y, seq.quad.w_z)
    types = seq.basis_0.types
    bsizes = (seq.basis_0.nr, seq.basis_0.nt, seq.basis_0.nz)
    bases = tuple(deriv[a] if diff[a] else primal[a] for a in range(3))

    def _restrict(mat, axis):
        return _restrict_radial_mass(mat, radial_start, nr) if axis == 0 else mat

    M0 = [_restrict(_m1d(bases[a], quad_w[a]), a) for a in range(3)]
    stiff_axes = tuple(a for a in range(3) if not diff[a])

    fd_V, fd_lam = [None, None, None], [None, None, None]
    for a in range(3):
        if a in stiff_axes:
            K0 = _restrict(_k1d(primal[a], deriv[a], quad_w[a],
                                _dense_incidence_1d(bsizes[a], types[a])), a)
            fd_V[a], fd_lam[a] = _simultaneous_diagonalize_pair(M0[a], K0)
        else:
            fd_V[a] = _mass_orthonormal_basis(M0[a])
            fd_lam[a] = jnp.ones((M0[a].shape[0],), dtype=jnp.float64)

    # Greville abscissae + metric at the collocation points (as in the mass builder).
    eps = 1e-7
    grev = []
    for axis in range(3):
        if diff[axis]:
            d = seq.basis_0.dΛ[axis]
            g = SplineBasis(int(d.n), int(d.p), d.type).greville_points()
        else:
            g = seq.basis_0.Λ[axis].greville_points()
        if types[axis] == "clamped":
            g = jnp.clip(g, eps, 1.0 - eps)
        grev.append(g)
    grev_r = grev[0][radial_start:radial_start + nr]
    rr, tt, zz = jnp.meshgrid(grev_r, grev[1], grev[2], indexing="ij")
    pts = jnp.stack([rr.ravel(), tt.ravel(), zz.ravel()], axis=-1)
    metric, _, jac = compute_geometry_terms(seq.map, pts)

    def _bcast(vec, axis):
        sh = [1, 1, 1]; sh[axis] = dims[axis]
        return jnp.asarray(vec).reshape(sh)

    def _chan(a):  # beta_aa = g_aa / J at the Greville points
        return jnp.asarray(metric[:, a, a] / jac).reshape(dims)

    denom = jnp.zeros(dims, dtype=jnp.float64)
    if k == 2:
        a = stiff_axes[0]
        D = jnp.asarray(1.0 / jac).reshape(dims)
        denom = denom + _bcast(fd_lam[a], a)            # alpha = 1 (single channel = D)
    else:  # k == 1
        b, c = stiff_axes
        wb, wc = _chan(c), _chan(b)                     # CROSS: K_b<-chan c, K_c<-chan b
        prod = jnp.where(jnp.isfinite(wb) & jnp.isfinite(wc) & (wb > 0) & (wc > 0),
                         wb * wc, jnp.nan)
        D = jnp.sqrt(prod)
        good = jnp.isfinite(D) & (D > 0)
        scale = jnp.median(D[good]) if int(good.sum()) > 0 else jnp.asarray(1.0)
        Dm = jnp.where(good, D, scale)
        alpha_b = jnp.mean((wb / Dm)[good]) if int(good.sum()) > 0 else jnp.asarray(1.0)
        alpha_c = jnp.mean((wc / Dm)[good]) if int(good.sum()) > 0 else jnp.asarray(1.0)
        denom = denom + alpha_b * _bcast(fd_lam[b], b) + alpha_c * _bcast(fd_lam[c], c)
        D = Dm

    valid = jnp.isfinite(D) & (D > 0)
    fin = D[valid]
    scale = jnp.median(fin) if fin.size > 0 else jnp.asarray(1.0, dtype=jnp.float64)
    D = jnp.where(valid, D, scale)
    inv_sqrt_D = 1.0 / jnp.sqrt(D)

    return TensorDiagonalBlockInverseFactors(
        shape=dims,
        fd_V_r=fd_V[0], fd_V_t=fd_V[1], fd_V_z=fd_V[2],
        fd_lam_r=fd_lam[0], fd_lam_t=fd_lam[1], fd_lam_z=fd_lam[2],
        fd_inv_denom=_modal_regularized_inverse_denom(denom, relative_tol=1e-8),
        greville_inv_sqrt_D=inv_sqrt_D,
    )


