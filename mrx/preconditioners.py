from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Mapping, Optional

import equinox as eqx
import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp

from mrx.utils import diag_EAET


class BoundaryConditionPair(eqx.Module):
    free: Optional[object] = None
    dbc: Optional[object] = None


class JacobiMassPreconditioner(eqx.Module):
    k0: BoundaryConditionPair = eqx.field(default_factory=BoundaryConditionPair)
    k1: BoundaryConditionPair = eqx.field(default_factory=BoundaryConditionPair)
    k2: BoundaryConditionPair = eqx.field(default_factory=BoundaryConditionPair)
    k3: BoundaryConditionPair = eqx.field(default_factory=BoundaryConditionPair)


class ExtractedMassApplyData(eqx.Module):
    mass_sp: object
    extraction: object
    extraction_t: object
    size: int = eqx.field(static=True)


class RestrictedExtractedMassApplyData(eqx.Module):
    mass_sp: object
    row_extraction: object
    col_extraction_t: object
    output_size: int = eqx.field(static=True)
    input_size: int = eqx.field(static=True)


class TensorDiagonalBlockInverseFactors(eqx.Module):
    shape: tuple[int, int, int] = eqx.field(static=True)
    cp_relative_error: Optional[float] = None
    cp_final_delta: Optional[float] = None
    chebyshev_steps: int = eqx.field(static=True, default=0)
    chebyshev_lambda_min: Optional[float] = None
    chebyshev_lambda_max: Optional[float] = None
    richardson_steps: int = eqx.field(static=True, default=0)
    richardson_omega: float = eqx.field(static=True, default=1.0)
    direct_inv_r: Optional[jnp.ndarray] = None
    direct_inv_t: Optional[jnp.ndarray] = None
    direct_inv_z: Optional[jnp.ndarray] = None
    term_r: tuple[jnp.ndarray, ...] = ()
    term_t: tuple[jnp.ndarray, ...] = ()
    term_z: tuple[jnp.ndarray, ...] = ()
    modal_basis_r: Optional[jnp.ndarray] = None
    modal_basis_t: Optional[jnp.ndarray] = None
    modal_basis_z: Optional[jnp.ndarray] = None
    modal_r: tuple[jnp.ndarray, ...] = ()
    modal_t: tuple[jnp.ndarray, ...] = ()
    modal_z: tuple[jnp.ndarray, ...] = ()


class K0TensorMassPreconditionerFactors(eqx.Module):
    bulk: TensorDiagonalBlockInverseFactors
    schur_inv: Optional[jnp.ndarray] = None


class K1TensorMassPreconditionerFactors(eqx.Module):
    r_indices: jnp.ndarray
    theta_bulk_indices: jnp.ndarray
    zeta_bulk_indices: jnp.ndarray
    rt_r_size: int = eqx.field(static=True)
    rt_theta_size: int = eqx.field(static=True)
    arr: TensorDiagonalBlockInverseFactors
    theta: TensorDiagonalBlockInverseFactors
    zeta: TensorDiagonalBlockInverseFactors
    use_inner_schur: bool = eqx.field(static=True, default=True)
    schur_inv: Optional[jnp.ndarray] = None


class K2TensorMassPreconditionerFactors(eqx.Module):
    r_bulk_indices: jnp.ndarray
    theta_indices: jnp.ndarray
    zeta_indices: jnp.ndarray
    r_bulk_size: int = eqx.field(static=True)
    theta_size: int = eqx.field(static=True)
    zeta_size: int = eqx.field(static=True)
    r_bulk: TensorDiagonalBlockInverseFactors
    theta: TensorDiagonalBlockInverseFactors
    zeta: TensorDiagonalBlockInverseFactors
    use_inner_schur: bool = eqx.field(static=True, default=True)
    schur_inv: Optional[jnp.ndarray] = None


class K0MassSurgeryPreconditionerFactors(eqx.Module):
    surgery_size: int = eqx.field(static=True)
    apply_data: ExtractedMassApplyData
    surgery_diaginv: jnp.ndarray
    ass: jnp.ndarray
    surgery_to_bulk_data: Optional[RestrictedExtractedMassApplyData] = None
    bulk_to_surgery_data: Optional[RestrictedExtractedMassApplyData] = None


class K1MassSurgeryPreconditionerFactors(eqx.Module):
    surgery_indices: jnp.ndarray
    bulk_indices: jnp.ndarray
    r_indices: jnp.ndarray
    theta_bulk_indices: jnp.ndarray
    zeta_bulk_indices: jnp.ndarray
    rt_indices: jnp.ndarray
    surgery_size: int = eqx.field(static=True)
    rt_r_size: int = eqx.field(static=True)
    rt_theta_size: int = eqx.field(static=True)
    bulk_rt_size: int = eqx.field(static=True)
    bulk_zeta_size: int = eqx.field(static=True)
    apply_data: ExtractedMassApplyData
    surgery_diaginv: jnp.ndarray
    ass: jnp.ndarray
    surgery_to_bulk_data: Optional[RestrictedExtractedMassApplyData] = None
    bulk_to_surgery_data: Optional[RestrictedExtractedMassApplyData] = None
    rt_atr_data: Optional[RestrictedExtractedMassApplyData] = None
    rt_art_data: Optional[RestrictedExtractedMassApplyData] = None
    rt_to_zeta_data: Optional[RestrictedExtractedMassApplyData] = None
    zeta_to_rt_data: Optional[RestrictedExtractedMassApplyData] = None


class K2MassSurgeryPreconditionerFactors(eqx.Module):
    surgery_indices: jnp.ndarray
    bulk_indices: jnp.ndarray
    r_bulk_indices: jnp.ndarray
    theta_indices: jnp.ndarray
    zeta_indices: jnp.ndarray
    surgery_size: int = eqx.field(static=True)
    r_bulk_size: int = eqx.field(static=True)
    theta_size: int = eqx.field(static=True)
    zeta_size: int = eqx.field(static=True)
    apply_data: ExtractedMassApplyData
    surgery_diaginv: jnp.ndarray
    ass: jnp.ndarray
    surgery_to_bulk_data: Optional[RestrictedExtractedMassApplyData] = None
    bulk_to_surgery_data: Optional[RestrictedExtractedMassApplyData] = None
    r_to_theta_data: Optional[RestrictedExtractedMassApplyData] = None
    theta_to_r_data: Optional[RestrictedExtractedMassApplyData] = None
    rt_to_zeta_data: Optional[RestrictedExtractedMassApplyData] = None
    zeta_to_rt_data: Optional[RestrictedExtractedMassApplyData] = None


class MassSurgeryPreconditioner(eqx.Module):
    k0: BoundaryConditionPair = eqx.field(default_factory=BoundaryConditionPair)
    k1: BoundaryConditionPair = eqx.field(default_factory=BoundaryConditionPair)
    k2: BoundaryConditionPair = eqx.field(default_factory=BoundaryConditionPair)
    k3: BoundaryConditionPair = eqx.field(default_factory=BoundaryConditionPair)


class TensorMassPreconditioner(eqx.Module):
    rank: int = eqx.field(static=True)
    cp_maxiter: int = eqx.field(static=True)
    cp_tol: float = eqx.field(static=True)
    cp_ridge: float = eqx.field(static=True)
    k0_rank: int = eqx.field(static=True, default=1)
    k1_rank: int = eqx.field(static=True, default=1)
    k2_rank: int = eqx.field(static=True, default=1)
    k3_rank: int = eqx.field(static=True, default=1)
    block_chebyshev_steps: int = eqx.field(static=True, default=3)
    block_lanczos_iterations: int = eqx.field(static=True, default=16)
    block_lanczos_max_eig_inflation: float = eqx.field(static=True, default=1.1)
    block_lanczos_min_eig_deflation: float = eqx.field(static=True, default=0.85)
    block_lanczos_min_eig_floor_fraction: float = eqx.field(static=True, default=1e-3)
    richardson_steps: int = eqx.field(static=True, default=0)
    richardson_omega: float = eqx.field(static=True, default=1.0)
    k0: BoundaryConditionPair = eqx.field(default_factory=BoundaryConditionPair)
    k1: BoundaryConditionPair = eqx.field(default_factory=BoundaryConditionPair)
    k2: BoundaryConditionPair = eqx.field(default_factory=BoundaryConditionPair)
    k3: BoundaryConditionPair = eqx.field(default_factory=BoundaryConditionPair)


class MassPreconditioners(eqx.Module):
    jacobi: Optional[JacobiMassPreconditioner] = None
    surgery: Optional[MassSurgeryPreconditioner] = None
    tensor: Optional[TensorMassPreconditioner] = None


def tensor_mass_rank_for_degree(tensor: TensorMassPreconditioner, k: int) -> int:
    match k:
        case 0:
            return int(getattr(tensor, 'k0_rank', tensor.rank))
        case 1:
            return int(getattr(tensor, 'k1_rank', tensor.rank))
        case 2:
            return int(getattr(tensor, 'k2_rank', tensor.rank))
        case 3:
            return int(getattr(tensor, 'k3_rank', tensor.rank))
        case _:
            raise ValueError("k must be 0, 1, 2 or 3")


@dataclass(frozen=True)
class MassPreconditionerSpec:
    kind: str = 'tensor'
    surgery_schur: bool = False
    steps: int = 4
    power_iterations: int = 30
    damping_safety: float = 0.8
    min_eig_fraction: float = 1e-3
    lanczos_iterations: int = 16
    lanczos_max_eig_inflation: float = 1.1
    lanczos_min_eig_deflation: float = 0.85
    lanczos_min_eig_floor_fraction: float = 1e-3
    smoother: Optional[MassPreconditionerSpec] = None


@dataclass(frozen=True)
class SchurPreconditionerSpec:
    inner: MassPreconditionerSpec = dataclass_field(
        default_factory=MassPreconditionerSpec)
    outer: MassPreconditionerSpec = dataclass_field(
        default_factory=lambda: MassPreconditionerSpec(kind='jacobi'))


@dataclass(frozen=True)
class SaddlePointPreconditionerSpec:
    mass: MassPreconditionerSpec = dataclass_field(
        default_factory=MassPreconditionerSpec)
    schur: SchurPreconditionerSpec = dataclass_field(
        default_factory=SchurPreconditionerSpec)
    coupled: bool = False


def default_mass_preconditioner() -> MassPreconditionerSpec:
    return MassPreconditionerSpec(kind='tensor', surgery_schur=True)


def default_saddle_preconditioner() -> SaddlePointPreconditionerSpec:
    return SaddlePointPreconditionerSpec()


def select_boundary_data(pair: BoundaryConditionPair, dirichlet: bool, label: str):
    data = pair.dbc if dirichlet else pair.free
    if data is None:
        side = "dbc" if dirichlet else "free"
        raise ValueError(f"{label} preconditioner is not assembled for {side} BCs")
    return data


def _mass_jacobi_pair(preconds: Optional[MassPreconditioners], k: int) -> Optional[BoundaryConditionPair]:
    if preconds is None or preconds.jacobi is None:
        return None
    match k:
        case 0:
            return preconds.jacobi.k0
        case 1:
            return preconds.jacobi.k1
        case 2:
            return preconds.jacobi.k2
        case 3:
            return preconds.jacobi.k3
    raise ValueError("k must be 0, 1, 2 or 3")


def get_mass_jacobi_diaginv(preconds: Optional[MassPreconditioners], k: int, dirichlet: bool):
    pair = _mass_jacobi_pair(preconds, k)
    if pair is None:
        raise ValueError(f"Jacobi mass preconditioner k={k} is not assembled")
    return select_boundary_data(pair, dirichlet, f"Jacobi mass k={k}")


def set_mass_jacobi_pair(preconds: Optional[MassPreconditioners], k: int, pair: BoundaryConditionPair):
    if preconds is None:
        preconds = MassPreconditioners()
    jacobi = preconds.jacobi if preconds.jacobi is not None else JacobiMassPreconditioner()
    match k:
        case 0:
            jacobi = eqx.tree_at(lambda data: data.k0, jacobi, pair)
        case 1:
            jacobi = eqx.tree_at(lambda data: data.k1, jacobi, pair)
        case 2:
            jacobi = eqx.tree_at(lambda data: data.k2, jacobi, pair)
        case 3:
            jacobi = eqx.tree_at(lambda data: data.k3, jacobi, pair)
        case _:
            raise ValueError("k must be 0, 1, 2 or 3")
    return eqx.tree_at(
        lambda data: data.jacobi,
        preconds,
        jacobi,
        is_leaf=lambda x: x is None,
    )


def _mass_surgery_pair(preconds: Optional[MassPreconditioners], k: int) -> Optional[BoundaryConditionPair]:
    if preconds is None or preconds.surgery is None:
        return None
    match k:
        case 0:
            return preconds.surgery.k0
        case 1:
            return preconds.surgery.k1
        case 2:
            return preconds.surgery.k2
        case 3:
            return preconds.surgery.k3
    raise ValueError("k must be 0, 1, 2 or 3")


def set_mass_surgery_pair(preconds: Optional[MassPreconditioners], k: int, pair: BoundaryConditionPair):
    if preconds is None:
        preconds = MassPreconditioners()
    surgery = preconds.surgery if preconds.surgery is not None else MassSurgeryPreconditioner()
    match k:
        case 0:
            surgery = eqx.tree_at(lambda data: data.k0, surgery, pair)
        case 1:
            surgery = eqx.tree_at(lambda data: data.k1, surgery, pair)
        case 2:
            surgery = eqx.tree_at(lambda data: data.k2, surgery, pair)
        case 3:
            surgery = eqx.tree_at(lambda data: data.k3, surgery, pair)
        case _:
            raise ValueError("k must be 0, 1, 2 or 3")
    return eqx.tree_at(
        lambda data: data.surgery,
        preconds,
        surgery,
        is_leaf=lambda x: x is None,
    )


def set_mass_surgery(preconds: Optional[MassPreconditioners], data: MassSurgeryPreconditioner):
    if preconds is None:
        preconds = MassPreconditioners()
    return eqx.tree_at(
        lambda payload: payload.surgery,
        preconds,
        data,
        is_leaf=lambda x: x is None,
    )


def set_mass_tensor(preconds: Optional[MassPreconditioners], data: TensorMassPreconditioner):
    if preconds is None:
        preconds = MassPreconditioners()
    return eqx.tree_at(
        lambda payload: payload.tensor,
        preconds,
        data,
        is_leaf=lambda x: x is None,
    )


def mass_surgery_available(seq, preconds: Optional[MassPreconditioners], k: int) -> bool:
    del seq
    if k not in (0, 1, 2) or preconds is None or preconds.surgery is None:
        return False
    pair = _mass_surgery_pair(preconds, k)
    return pair is not None and pair.free is not None and pair.dbc is not None


def _select_mass_surgery_factors(preconds: Optional[MassPreconditioners], k: int, dirichlet: bool):
    pair = _mass_surgery_pair(preconds, k)
    if pair is None:
        raise ValueError(f"Mass surgery preconditioner k={k} is not assembled")
    return select_boundary_data(pair, dirichlet, f"Mass surgery k={k}")


def set_mass_rtzblock_factor(preconds: Optional[MassPreconditioners], k: int, dirichlet: bool, factor_data):
    raise ValueError("rt-zblock mass preconditioner has been retired from production")


def invalidate_mass_rtzblock(preconds: Optional[MassPreconditioners], k: int):
    return preconds


def build_mass_jacobi_pair(seq, mass_sp, k: int) -> BoundaryConditionPair:
    e = getattr(seq, f"e{k}")
    e_t = getattr(seq, f"e{k}_T")
    e_dbc = getattr(seq, f"e{k}_dbc")
    e_t_dbc = getattr(seq, f"e{k}_dbc_T")
    return BoundaryConditionPair(
        free=1.0 / diag_EAET(e, mass_sp, e_t),
        dbc=1.0 / diag_EAET(e_dbc, mass_sp, e_t_dbc),
    )


def _quadrature_tensor_shape(seq) -> tuple[int, int, int]:
    return seq.quad.ny, seq.quad.nx, seq.quad.nz


def _reshape_quadrature_scalar_field(seq, values: jnp.ndarray) -> jnp.ndarray:
    return jnp.asarray(values).reshape(_quadrature_tensor_shape(seq))


def _reshape_quadrature_matrix_field(seq, values: jnp.ndarray) -> jnp.ndarray:
    field = jnp.asarray(values)
    return field.reshape(*_quadrature_tensor_shape(seq), *field.shape[1:])


def _k1_diagonal_metric_tensors(seq) -> dict[str, jnp.ndarray]:
    jacobian = _reshape_quadrature_scalar_field(seq, seq.geometry.jacobian_j)
    metric_inv = _reshape_quadrature_matrix_field(seq, seq.geometry.metric_inv_jkl)
    return {
        "alpha_rr": jacobian * metric_inv[..., 0, 0],
        "alpha_thetatheta": jacobian * metric_inv[..., 1, 1],
        "alpha_zetazeta": jacobian * metric_inv[..., 2, 2],
    }


def _k2_diagonal_metric_tensors(seq) -> dict[str, jnp.ndarray]:
    jacobian = _reshape_quadrature_scalar_field(seq, seq.geometry.jacobian_j)
    metric = _reshape_quadrature_matrix_field(seq, seq.geometry.metric_jkl)
    return {
        "beta_rr": metric[..., 0, 0] / jacobian,
        "beta_thetatheta": metric[..., 1, 1] / jacobian,
        "beta_zetazeta": metric[..., 2, 2] / jacobian,
    }


def _mean_one(values: jnp.ndarray) -> jnp.ndarray:
    mean_value = jnp.mean(values)
    safe_mean = jnp.where(jnp.abs(mean_value) > 0, mean_value, 1.0)
    return values / safe_mean


def _safe_radial_quadrature(seq) -> jnp.ndarray:
    return jnp.maximum(jnp.asarray(seq.quad.x_x, dtype=jnp.float64), 1e-8)


def _k0_radial_reference_baseline(seq) -> jnp.ndarray:
    return _mean_one(_safe_radial_quadrature(seq))


def _k1_radial_reference_baselines(seq) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    safe_r = _safe_radial_quadrature(seq)
    return (
        _mean_one(safe_r),
        _mean_one(1.0 / safe_r),
        _mean_one(safe_r),
    )


def _k2_radial_reference_baselines(seq) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    safe_r = _safe_radial_quadrature(seq)
    return (
        _mean_one(1.0 / safe_r),
        _mean_one(safe_r),
        _mean_one(1.0 / safe_r),
    )


def _k3_radial_reference_baseline(seq) -> jnp.ndarray:
    return _mean_one(1.0 / _safe_radial_quadrature(seq))


def _mode_unfold_3tensor(tensor: jnp.ndarray, mode: int) -> jnp.ndarray:
    return jnp.moveaxis(tensor, mode, 0).reshape(tensor.shape[mode], -1)


def _khatri_rao(left: jnp.ndarray, right: jnp.ndarray) -> jnp.ndarray:
    if left.shape[1] != right.shape[1]:
        raise ValueError(
            f"Khatri-Rao factors must have matching column counts, got {left.shape[1]} and {right.shape[1]}"
        )
    return (left[:, None, :] * right[None, :, :]).reshape(left.shape[0] * right.shape[0], left.shape[1])


def _normalize_cp_columns(factor: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    norms = jnp.linalg.norm(factor, axis=0)
    safe_norms = jnp.where(norms > 0, norms, 1.0)
    return factor / safe_norms, norms


def _reconstruct_cp_3tensor(
    weights: jnp.ndarray,
    factors: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> jnp.ndarray:
    factor_theta, factor_r, factor_z = factors
    return jnp.einsum("r,ir,jr,kr->ijk", weights, factor_theta, factor_r, factor_z)


def _cp_als_3tensor(
    tensor: jnp.ndarray,
    rank: int,
    *,
    maxiter: int,
    tol: float,
    ridge: float,
) -> tuple[
    jnp.ndarray,
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    float,
    float,
    int,
]:
    if tensor.ndim != 3:
        raise ValueError(f"CP-ALS expects a 3-tensor, got shape {tensor.shape}")
    if rank < 1 or rank > min(tensor.shape):
        raise ValueError(f"Requested CP rank {rank} outside valid range [1, {min(tensor.shape)}]")

    unfolded_0 = _mode_unfold_3tensor(tensor, 0)
    unfolded_1 = _mode_unfold_3tensor(tensor, 1)
    unfolded_2 = _mode_unfold_3tensor(tensor, 2)

    factor_theta = jnp.linalg.svd(unfolded_0, full_matrices=False)[0][:, :rank]
    factor_r = jnp.linalg.svd(unfolded_1, full_matrices=False)[0][:, :rank]
    factor_z = jnp.linalg.svd(unfolded_2, full_matrices=False)[0][:, :rank]
    factor_theta, _ = _normalize_cp_columns(factor_theta)
    factor_r, _ = _normalize_cp_columns(factor_r)
    factor_z, _ = _normalize_cp_columns(factor_z)
    weights = jnp.ones((rank,), dtype=tensor.dtype)

    eye = jnp.eye(rank, dtype=tensor.dtype)
    previous_error = jnp.inf
    relative_error = jnp.inf
    final_delta = jnp.inf
    n_iterations = 0

    for iteration in range(maxiter):
        factor_z_eff = factor_z * weights[None, :]

        khatri_rao_tz = _khatri_rao(factor_r, factor_z_eff)
        gram_tz = (factor_r.T @ factor_r) * (factor_z_eff.T @ factor_z_eff)
        factor_theta_raw = jnp.linalg.solve(gram_tz + ridge * eye, (unfolded_0 @ khatri_rao_tz).T).T

        khatri_rao_rz = _khatri_rao(factor_theta_raw, factor_z_eff)
        gram_rz = (factor_theta_raw.T @ factor_theta_raw) * (factor_z_eff.T @ factor_z_eff)
        factor_r_raw = jnp.linalg.solve(gram_rz + ridge * eye, (unfolded_1 @ khatri_rao_rz).T).T

        khatri_rao_rt = _khatri_rao(factor_theta_raw, factor_r_raw)
        gram_rt = (factor_theta_raw.T @ factor_theta_raw) * (factor_r_raw.T @ factor_r_raw)
        factor_z_eff_raw = jnp.linalg.solve(gram_rt + ridge * eye, (unfolded_2 @ khatri_rao_rt).T).T

        factor_theta, theta_norms = _normalize_cp_columns(factor_theta_raw)
        factor_r, r_norms = _normalize_cp_columns(factor_r_raw)
        factor_z_temp = factor_z_eff_raw * (theta_norms * r_norms)[None, :]
        factor_z, weights = _normalize_cp_columns(factor_z_temp)

        reconstruction = _reconstruct_cp_3tensor(weights, (factor_theta, factor_r, factor_z))
        relative_error = float(
            jnp.linalg.norm(reconstruction - tensor) / jnp.maximum(jnp.linalg.norm(tensor), 1.0)
        )
        final_delta = abs(relative_error - previous_error) if previous_error < jnp.inf else jnp.inf
        previous_error = relative_error
        n_iterations = iteration + 1
        if final_delta < tol:
            break

    return weights, (factor_theta, factor_r, factor_z), relative_error, final_delta, n_iterations


def _apply_tensor_diagonal_block_forward(
    factors: TensorDiagonalBlockInverseFactors,
    x: jnp.ndarray,
) -> jnp.ndarray:
    nr, nt, nz = factors.shape
    field = jnp.asarray(x).reshape(nr, nt, nz)
    result = jnp.zeros_like(field)
    for mass_r, mass_t, mass_z in zip(factors.term_r, factors.term_t, factors.term_z):
        term = jnp.einsum("ij,jkl->ikl", mass_r, field)
        term = jnp.einsum("ij,kjl->kil", mass_t, term)
        term = jnp.einsum("ij,klj->kli", mass_z, term)
        result = result + term
    return result.reshape(-1)


def _apply_tensor_diagonal_block_preconditioner(
    factors: TensorDiagonalBlockInverseFactors,
    rhs: jnp.ndarray,
) -> jnp.ndarray:
    nr, nt, nz = factors.shape
    modes = jnp.asarray(rhs).reshape(nr, nt, nz)

    if factors.direct_inv_r is not None:
        modes = jnp.einsum("ij,jkl->ikl", factors.direct_inv_r, modes)
        modes = jnp.einsum("ij,kjl->kil", factors.direct_inv_t, modes)
        modes = jnp.einsum("ij,klj->kli", factors.direct_inv_z, modes)
        return modes.reshape(-1)

    if factors.modal_basis_r is None or factors.modal_basis_t is None or factors.modal_basis_z is None:
        raise ValueError("Missing modal bases for tensor diagonal block preconditioner")

    modes = jnp.einsum("ji,jkl->ikl", factors.modal_basis_r, modes)
    modes = jnp.einsum("ji,kjl->kil", factors.modal_basis_t, modes)
    modes = jnp.einsum("ji,klj->kli", factors.modal_basis_z, modes)

    denom = jnp.zeros((nr, nt, nz), dtype=rhs.dtype)
    for lam_r, lam_t, lam_z in zip(factors.modal_r, factors.modal_t, factors.modal_z):
        denom = denom + lam_r[:, None, None] * lam_t[None, :, None] * lam_z[None, None, :]

    denom_floor = 1e-12 * jnp.max(jnp.abs(denom))
    safe_floor = jnp.where(denom_floor > 0, denom_floor, 1.0)
    safe_denom = jnp.where(denom > safe_floor, denom, safe_floor)
    modes = modes / safe_denom

    modes = jnp.einsum("ij,jkl->ikl", factors.modal_basis_r, modes)
    modes = jnp.einsum("ij,kjl->kil", factors.modal_basis_t, modes)
    modes = jnp.einsum("ij,klj->kli", factors.modal_basis_z, modes)
    return modes.reshape(-1)


def _estimate_preconditioned_max_eigenvalue_apply(
    operator_apply,
    smoother_apply,
    size: int,
    *,
    n_iter: int = 10,
    seed: int = 0,
):
    vector = jax.random.normal(
        jax.random.PRNGKey(seed),
        (size,),
        dtype=jnp.float64,
    )

    def operator_norm(x):
        ax = operator_apply(x)
        return jnp.sqrt(jnp.abs(jnp.vdot(x, ax).real))

    init_norm = operator_norm(vector)
    vector = vector / jnp.where(init_norm > 0, init_norm, 1.0)

    rayleigh = jnp.asarray(0.0, dtype=jnp.float64)
    current = vector
    for _ in range(n_iter):
        image = smoother_apply(operator_apply(current))
        image_norm = operator_norm(image)
        safe_norm = jnp.where(image_norm > 0, image_norm, 1.0)
        current = image / safe_norm
        rayleigh = jnp.real(
            jnp.vdot(
                current,
                operator_apply(smoother_apply(operator_apply(current))),
            )
        )
    return jnp.maximum(rayleigh, jnp.asarray(0.0, dtype=jnp.float64))


def _estimate_chebyshev_lanczos_bounds_apply(
    operator_apply,
    smoother_apply,
    size: int,
    *,
    lanczos_iterations: int,
    lanczos_max_eig_inflation: float,
    lanczos_min_eig_deflation: float,
    lanczos_min_eig_floor_fraction: float,
    seed: int = 0,
):
    if lanczos_iterations < 1:
        raise ValueError("Lanczos iteration count must be positive")

    tiny = jnp.asarray(jnp.finfo(jnp.float64).tiny, dtype=jnp.float64)

    def operator_norm(x):
        ax = operator_apply(x)
        return jnp.sqrt(jnp.maximum(jnp.abs(jnp.vdot(x, ax).real), tiny))

    vector = jax.random.normal(
        jax.random.PRNGKey(seed),
        (size,),
        dtype=jnp.float64,
    )
    init_norm = operator_norm(vector)
    current = vector / jnp.where(init_norm > 0, init_norm, 1.0)
    previous = jnp.zeros_like(current)
    beta_prev = jnp.asarray(0.0, dtype=jnp.float64)
    alphas = []
    betas = []

    for iteration in range(lanczos_iterations):
        image = smoother_apply(operator_apply(current))
        alpha = jnp.real(jnp.vdot(current, operator_apply(image)))
        residual = image - alpha * current
        if iteration > 0:
            residual = residual - beta_prev * previous
        beta = operator_norm(residual)
        alphas.append(alpha)
        continue_iteration = iteration + 1 < lanczos_iterations and float(beta) > float(tiny)
        betas.append(beta if continue_iteration else jnp.asarray(0.0, dtype=jnp.float64))
        if not continue_iteration:
            break
        previous, current = current, residual / beta
        beta_prev = beta

    alpha_array = jnp.asarray(alphas, dtype=jnp.float64)
    beta_array = jnp.asarray(betas[:-1], dtype=jnp.float64) if len(betas) > 1 else jnp.asarray([], dtype=jnp.float64)
    tridiagonal = jnp.diag(alpha_array)
    if beta_array.size > 0:
        tridiagonal = tridiagonal + jnp.diag(beta_array, k=1) + jnp.diag(beta_array, k=-1)
    ritz_values = jnp.linalg.eigvalsh(tridiagonal)
    max_ritz = jnp.maximum(ritz_values[-1], tiny)
    max_eig = jnp.maximum(
        jnp.asarray(lanczos_max_eig_inflation, dtype=jnp.float64) * max_ritz,
        tiny,
    )
    floor = jnp.asarray(lanczos_min_eig_floor_fraction, dtype=jnp.float64) * max_eig
    min_positive_ritz = jnp.min(jnp.where(ritz_values > tiny, ritz_values, jnp.inf))
    guarded_min = jnp.asarray(lanczos_min_eig_deflation, dtype=jnp.float64) * min_positive_ritz
    min_eig = jnp.where(
        jnp.isfinite(min_positive_ritz),
        jnp.maximum(floor, guarded_min),
        floor,
    )
    return min_eig, max_eig


def _assemble_shared_modal_basis(
    reference_mass: jnp.ndarray,
    matrices: tuple[jnp.ndarray, ...],
    term_weights: jnp.ndarray,
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, ...]]:
    if not matrices:
        raise ValueError("shared modal basis requires at least one matrix")

    L = jnp.linalg.cholesky(reference_mass)
    whitened_matrices = []
    for matrix in matrices:
        Y = jnp.linalg.solve(L, matrix)
        matrix_tilde = jnp.linalg.solve(L, Y.T).T
        whitened_matrices.append(0.5 * (matrix_tilde + matrix_tilde.T))

    reference_tilde = jnp.zeros_like(whitened_matrices[0])
    for weight, matrix_tilde in zip(term_weights, whitened_matrices):
        reference_tilde = reference_tilde + weight * matrix_tilde
    weight_sum = jnp.sum(term_weights)
    safe_weight_sum = jnp.where(weight_sum > 0, weight_sum, 1.0)
    reference_tilde = 0.5 * (reference_tilde + reference_tilde.T) / safe_weight_sum

    _, Q = jnp.linalg.eigh(reference_tilde)
    V = jnp.linalg.solve(L.T, Q)
    modal_diagonals = tuple(jnp.diag(Q.T @ matrix_tilde @ Q) for matrix_tilde in whitened_matrices)
    return V, modal_diagonals


def _apply_tensor_diagonal_block(factors: TensorDiagonalBlockInverseFactors, rhs: jnp.ndarray) -> jnp.ndarray:
    x = _apply_tensor_diagonal_block_preconditioner(factors, rhs)
    if factors.richardson_steps <= 0:
        return x

    for _ in range(factors.richardson_steps):
        residual = rhs - _apply_tensor_diagonal_block_forward(factors, x)
        x = x + factors.richardson_omega * _apply_tensor_diagonal_block_preconditioner(factors, residual)
    return x


def _k1_layout_sizes(seq, dirichlet: bool):
    boundary_offset = 1 if dirichlet else 0
    return {
        "theta_surgery": 2 * seq.basis_1.nz,
        "zeta_surgery": 3 * seq.basis_1.dz,
        "r": (seq.basis_1.dr - 1) * seq.basis_1.nt * seq.basis_1.nz,
        "theta_bulk": (seq.basis_1.nr - 2 - boundary_offset) * seq.basis_1.dt * seq.basis_1.nz,
        "zeta_bulk": (seq.basis_1.nr - 2 - boundary_offset) * seq.basis_1.nt * seq.basis_1.dz,
    }


def _component_sizes_k2(seq, dirichlet: bool):
    if dirichlet:
        return seq.n2_1_dbc, seq.n2_2_dbc, seq.n2_3_dbc
    return seq.n2_1, seq.n2_2, seq.n2_3


def _surgery_slices_k1(seq, dirichlet: bool):
    sizes = _k1_layout_sizes(seq, dirichlet)
    theta_surgery = slice(0, sizes["theta_surgery"])
    zeta_surgery = slice(theta_surgery.stop, theta_surgery.stop + sizes["zeta_surgery"])
    r_slice = slice(zeta_surgery.stop, zeta_surgery.stop + sizes["r"])
    theta_bulk = slice(r_slice.stop, r_slice.stop + sizes["theta_bulk"])
    zeta_bulk = slice(theta_bulk.stop, theta_bulk.stop + sizes["zeta_bulk"])
    return {
        "r": r_slice,
        "theta_surgery": theta_surgery,
        "theta_bulk": theta_bulk,
        "zeta_surgery": zeta_surgery,
        "zeta_bulk": zeta_bulk,
    }


def _surgery_slices_k2(seq, dirichlet: bool):
    n_r, n_theta, n_zeta = _component_sizes_k2(seq, dirichlet)
    r_slice = slice(0, n_r)
    theta_slice = slice(r_slice.stop, r_slice.stop + n_theta)
    zeta_slice = slice(theta_slice.stop, theta_slice.stop + n_zeta)
    r_surgery = slice(r_slice.start, r_slice.start + 2 * seq.basis_2.dz)
    r_bulk = slice(r_surgery.stop, r_slice.stop)
    return {
        "r_surgery": r_surgery,
        "r_bulk": r_bulk,
        "theta": theta_slice,
        "zeta": zeta_slice,
    }


def _schur_blocks(matrix: jnp.ndarray, surgery_size: int):
    ass = matrix[:surgery_size, :surgery_size]
    asb = matrix[:surgery_size, surgery_size:]
    abs_ = matrix[surgery_size:, :surgery_size]
    abb = matrix[surgery_size:, surgery_size:]
    return ass, asb, abs_, abb


def _tensor_block_indices_k1(seq, dirichlet: bool):
    slices = _surgery_slices_k1(seq, dirichlet)
    surgery_indices = jnp.concatenate(
        [
            jnp.arange(slices["theta_surgery"].start, slices["theta_surgery"].stop),
            jnp.arange(slices["zeta_surgery"].start, slices["zeta_surgery"].stop),
        ]
    )
    r_indices = jnp.arange(slices["r"].start, slices["r"].stop)
    theta_bulk_indices = jnp.arange(slices["theta_bulk"].start, slices["theta_bulk"].stop)
    zeta_bulk_indices = jnp.arange(slices["zeta_bulk"].start, slices["zeta_bulk"].stop)
    bulk_indices = jnp.concatenate([r_indices, theta_bulk_indices, zeta_bulk_indices])
    rt_indices = jnp.concatenate([r_indices, theta_bulk_indices])
    return {
        "surgery": surgery_indices,
        "bulk": bulk_indices,
        "r": r_indices,
        "theta_bulk": theta_bulk_indices,
        "rt": rt_indices,
        "zeta_bulk": zeta_bulk_indices,
        "rt_r_size": r_indices.shape[0],
        "rt_theta_size": theta_bulk_indices.shape[0],
        "bulk_rt_size": rt_indices.shape[0],
        "bulk_zeta_size": zeta_bulk_indices.shape[0],
    }


def _tensor_block_indices_k2(seq, dirichlet: bool):
    slices = _surgery_slices_k2(seq, dirichlet)
    surgery_indices = jnp.arange(slices["r_surgery"].start, slices["r_surgery"].stop)
    r_bulk_indices = jnp.arange(slices["r_bulk"].start, slices["r_bulk"].stop)
    theta_indices = jnp.arange(slices["theta"].start, slices["theta"].stop)
    zeta_indices = jnp.arange(slices["zeta"].start, slices["zeta"].stop)
    bulk_indices = jnp.concatenate([r_bulk_indices, theta_indices, zeta_indices])
    return {
        "surgery": surgery_indices,
        "bulk": bulk_indices,
        "r_bulk": r_bulk_indices,
        "theta": theta_indices,
        "zeta": zeta_indices,
        "r_bulk_size": r_bulk_indices.shape[0],
        "theta_size": theta_indices.shape[0],
        "zeta_size": zeta_indices.shape[0],
    }


def _arr_shape_k1(seq, dirichlet: bool) -> tuple[int, int, int]:
    nt = seq.basis_1.nt
    nz = seq.basis_1.nz
    n_r = _k1_layout_sizes(seq, dirichlet)["r"]
    nr = n_r // (nt * nz)
    if nr * nt * nz != n_r:
        raise ValueError(f"Extracted r size {n_r} is not divisible by nt*nz = {nt * nz}")
    return nr, nt, nz


def _theta_bulk_shape_k1(seq, dirichlet: bool) -> tuple[int, int, int]:
    dt = seq.basis_1.dt
    nz = seq.basis_1.nz
    n_theta = _k1_layout_sizes(seq, dirichlet)["theta_bulk"]
    nr = n_theta // (dt * nz)
    if nr * dt * nz != n_theta:
        raise ValueError(f"theta_bulk size {n_theta} is not divisible by dt*nz = {dt * nz}")
    return nr, dt, nz


def _zeta_bulk_shape_k1(seq, dirichlet: bool) -> tuple[int, int, int]:
    nt = seq.basis_1.nt
    dz = seq.basis_1.dz
    n_zeta = _k1_layout_sizes(seq, dirichlet)["zeta_bulk"]
    nr = n_zeta // (nt * dz)
    if nr * nt * dz != n_zeta:
        raise ValueError(f"zeta_bulk size {n_zeta} is not divisible by nt*dz = {nt * dz}")
    return nr, nt, dz


def _r_bulk_shape_k2(seq, dirichlet: bool) -> tuple[int, int, int]:
    dt = seq.basis_2.dt
    dz = seq.basis_2.dz
    n_r = _component_sizes_k2(seq, dirichlet)[0] - 2 * seq.basis_2.dz
    nr = n_r // (dt * dz)
    if nr * dt * dz != n_r:
        raise ValueError(f"r_bulk size {n_r} is not divisible by dt*dz = {dt * dz}")
    return nr, dt, dz


def _theta_shape_k2(seq, dirichlet: bool) -> tuple[int, int, int]:
    nt = seq.basis_2.nt
    dz = seq.basis_2.dz
    n_theta = _component_sizes_k2(seq, dirichlet)[1]
    nr = n_theta // (nt * dz)
    if nr * nt * dz != n_theta:
        raise ValueError(f"theta size {n_theta} is not divisible by nt*dz = {nt * dz}")
    return nr, nt, dz


def _zeta_shape_k2(seq, dirichlet: bool) -> tuple[int, int, int]:
    dt = seq.basis_2.dt
    nz = seq.basis_2.nz
    n_zeta = _component_sizes_k2(seq, dirichlet)[2]
    nr = n_zeta // (dt * nz)
    if nr * dt * nz != n_zeta:
        raise ValueError(f"zeta size {n_zeta} is not divisible by dt*nz = {dt * nz}")
    return nr, dt, nz


def _build_diagonal_tensor_block_factors(
    seq,
    tensor: jnp.ndarray,
    full_shape: tuple[int, int, int],
    rank: int,
    *,
    radial_basis: jnp.ndarray,
    theta_basis: jnp.ndarray,
    zeta_basis: jnp.ndarray,
    radial_weights: jnp.ndarray,
    theta_weights: jnp.ndarray,
    zeta_weights: jnp.ndarray,
    radial_start: int,
    cp_maxiter: int,
    cp_tol: float,
    cp_ridge: float,
    radial_baseline: Optional[jnp.ndarray] = None,
    chebyshev_steps: int = 0,
    chebyshev_lanczos_iterations: int = 16,
    chebyshev_lanczos_max_eig_inflation: float = 1.1,
    chebyshev_lanczos_min_eig_deflation: float = 0.85,
    chebyshev_lanczos_min_eig_floor_fraction: float = 1e-3,
    chebyshev_seed: int = 0,
    richardson_steps: int = 0,
    richardson_omega: float = 1.0,
) -> TensorDiagonalBlockInverseFactors:
    if radial_baseline is None:
        corrected_tensor = tensor
        scaled_radial_baseline = None
    else:
        scaled_radial_baseline = jnp.asarray(radial_baseline, dtype=tensor.dtype)
        corrected_tensor = tensor / scaled_radial_baseline[None, :, None]

    weights, factors, cp_relative_error, cp_final_delta, cp_iterations = _cp_als_3tensor(
        corrected_tensor,
        rank,
        maxiter=cp_maxiter,
        tol=cp_tol,
        ridge=cp_ridge,
    )
    nr, nt, nz = full_shape
    term_data = []
    for idx in range(rank):
        factor_theta = jnp.ravel(factors[0][:, idx])
        factor_r = jnp.ravel(factors[1][:, idx])
        factor_z = jnp.ravel(factors[2][:, idx])
        scale = weights[idx]
        radial_weight = scale * factor_r
        if scaled_radial_baseline is not None:
            radial_weight = scaled_radial_baseline * radial_weight
        raw_mass_r = _assemble_weighted_1d_mass(radial_basis, radial_weights * (scale * factor_r))
        if scaled_radial_baseline is not None:
            raw_mass_r = _assemble_weighted_1d_mass(radial_basis, radial_weights * radial_weight)
        mass_r = _restrict_radial_mass(raw_mass_r, radial_start, nr)
        mass_t = _assemble_weighted_1d_mass(theta_basis, theta_weights * factor_theta)
        mass_z = _assemble_weighted_1d_mass(zeta_basis, zeta_weights * factor_z)
        term_data.append(
            {
                "matrices": (mass_r, mass_t, mass_z),
                "size": float(jnp.linalg.norm(mass_r) * jnp.linalg.norm(mass_t) * jnp.linalg.norm(mass_z)),
            }
        )

    term_data.sort(key=lambda item: item["size"], reverse=True)
    term_matrices = tuple(item["matrices"] for item in term_data)
    if len(term_matrices) == 1:
        mass_r, mass_t, mass_z = term_matrices[0]
        return TensorDiagonalBlockInverseFactors(
            shape=full_shape,
            cp_relative_error=cp_relative_error,
            cp_final_delta=cp_final_delta,
            chebyshev_steps=chebyshev_steps,
            chebyshev_lambda_min=None,
            chebyshev_lambda_max=None,
            richardson_steps=0,
            richardson_omega=1.0,
            direct_inv_r=_symmetrize(jnp.linalg.inv(mass_r)),
            direct_inv_t=_symmetrize(jnp.linalg.inv(mass_t)),
            direct_inv_z=_symmetrize(jnp.linalg.inv(mass_z)),
            term_r=(mass_r,),
            term_t=(mass_t,),
            term_z=(mass_z,),
        )

    term_weights = jnp.asarray([item["size"] for item in term_data], dtype=tensor.dtype)
    reference_mass_r = _restrict_radial_mass(_assemble_weighted_1d_mass(radial_basis, radial_weights), radial_start, nr)
    reference_mass_t = _assemble_weighted_1d_mass(theta_basis, theta_weights)
    reference_mass_z = _assemble_weighted_1d_mass(zeta_basis, zeta_weights)
    V_r, modal_r = _assemble_shared_modal_basis(reference_mass_r, tuple(mass_r for mass_r, _, _ in term_matrices), term_weights)
    V_t, modal_t = _assemble_shared_modal_basis(reference_mass_t, tuple(mass_t for _, mass_t, _ in term_matrices), term_weights)
    V_z, modal_z = _assemble_shared_modal_basis(reference_mass_z, tuple(mass_z for _, _, mass_z in term_matrices), term_weights)
    return TensorDiagonalBlockInverseFactors(
        shape=full_shape,
        cp_relative_error=cp_relative_error,
        cp_final_delta=cp_final_delta,
        chebyshev_steps=chebyshev_steps,
        chebyshev_lambda_min=None,
        chebyshev_lambda_max=None,
        richardson_steps=richardson_steps,
        richardson_omega=richardson_omega,
        term_r=tuple(mass_r for mass_r, _, _ in term_matrices),
        term_t=tuple(mass_t for _, mass_t, _ in term_matrices),
        term_z=tuple(mass_z for _, _, mass_z in term_matrices),
        modal_basis_r=V_r,
        modal_basis_t=V_t,
        modal_basis_z=V_z,
        modal_r=modal_r,
        modal_t=modal_t,
        modal_z=modal_z,
    )


def _annotate_tensor_block_chebyshev_bounds(
    factors: TensorDiagonalBlockInverseFactors,
    *,
    lanczos_iterations: int,
    lanczos_max_eig_inflation: float,
    lanczos_min_eig_deflation: float,
    lanczos_min_eig_floor_fraction: float,
    seed: int,
) -> TensorDiagonalBlockInverseFactors:
    if factors.chebyshev_steps <= 0 or factors.direct_inv_r is not None:
        return factors

    operator_apply = lambda x, block_factors=factors: _apply_tensor_diagonal_block_forward(block_factors, x)
    smoother_apply = lambda x, block_factors=factors: _apply_tensor_diagonal_block_preconditioner(block_factors, x)
    lambda_min, lambda_max = _estimate_chebyshev_lanczos_bounds_apply(
        operator_apply,
        smoother_apply,
        int(jnp.prod(jnp.asarray(factors.shape))),
        lanczos_iterations=lanczos_iterations,
        lanczos_max_eig_inflation=lanczos_max_eig_inflation,
        lanczos_min_eig_deflation=lanczos_min_eig_deflation,
        lanczos_min_eig_floor_fraction=lanczos_min_eig_floor_fraction,
        seed=seed,
    )
    factors = eqx.tree_at(
        lambda block: block.chebyshev_lambda_min,
        factors,
        float(lambda_min),
        is_leaf=lambda x: x is None,
    )
    return eqx.tree_at(
        lambda block: block.chebyshev_lambda_max,
        factors,
        float(lambda_max),
        is_leaf=lambda x: x is None,
    )


def _build_chebyshev_apply_preconditioner(
    operator_apply,
    smoother_apply,
    *,
    steps: int,
    min_eig: float,
    max_eig: float,
):
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
                jnp.asarray(1.0, dtype=rhs.dtype) / (d.astype(rhs.dtype) - beta),
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


def _apply_tensor_exact_block(
    block_matrix: jnp.ndarray,
    factors: TensorDiagonalBlockInverseFactors,
    rhs: jnp.ndarray,
) -> jnp.ndarray:
    del block_matrix
    if (
        factors.direct_inv_r is None
        and
        factors.chebyshev_steps > 0
        and factors.chebyshev_lambda_min is not None
        and factors.chebyshev_lambda_max is not None
    ):
        apply = _build_chebyshev_apply_preconditioner(
            lambda x, block_factors=factors: _apply_tensor_diagonal_block_forward(block_factors, x),
            lambda x, block_factors=factors: _apply_tensor_diagonal_block_preconditioner(block_factors, x),
            steps=factors.chebyshev_steps,
            min_eig=factors.chebyshev_lambda_min,
            max_eig=factors.chebyshev_lambda_max,
        )
        return apply(rhs)
    return _apply_tensor_diagonal_block(factors, rhs)


def _extraction_operator(seq, k: int, dirichlet: bool):
    return getattr(seq, f"e{k}_dbc" if dirichlet else f"e{k}")


def _extraction_operator_transpose(seq, k: int, dirichlet: bool):
    return getattr(seq, f"e{k}_dbc_T" if dirichlet else f"e{k}_T")


def _extracted_size(seq, k: int, dirichlet: bool) -> int:
    return int(getattr(seq, f"n{k}_dbc" if dirichlet else f"n{k}"))


def _build_extracted_mass_apply_data(seq, mass_sp, k: int, dirichlet: bool) -> ExtractedMassApplyData:
    return ExtractedMassApplyData(
        mass_sp=mass_sp,
        extraction=_extraction_operator(seq, k, dirichlet),
        extraction_t=_extraction_operator_transpose(seq, k, dirichlet),
        size=_extracted_size(seq, k, dirichlet),
    )


def _as_bcoo(matrix):
    return matrix.to_bcoo() if hasattr(matrix, "to_bcoo") else matrix


def _restrict_sparse_rows(matrix, row_indices: jnp.ndarray):
    row_indices = jnp.asarray(row_indices, dtype=jnp.int32)
    matrix_bcoo = _as_bcoo(matrix)
    row_map = jnp.full((matrix_bcoo.shape[0],), -1, dtype=jnp.int32)
    row_map = row_map.at[row_indices].set(jnp.arange(row_indices.shape[0], dtype=jnp.int32))
    new_rows = row_map[matrix_bcoo.indices[:, 0]]
    mask = new_rows >= 0
    restricted = jsparse.BCOO(
        (matrix_bcoo.data[mask], jnp.stack([new_rows[mask], matrix_bcoo.indices[mask, 1]], axis=1)),
        shape=(row_indices.shape[0], matrix_bcoo.shape[1]),
    )
    return jsparse.BCSR.from_bcoo(restricted)


def _restrict_sparse_cols(matrix, col_indices: jnp.ndarray):
    col_indices = jnp.asarray(col_indices, dtype=jnp.int32)
    matrix_bcoo = _as_bcoo(matrix)
    col_map = jnp.full((matrix_bcoo.shape[1],), -1, dtype=jnp.int32)
    col_map = col_map.at[col_indices].set(jnp.arange(col_indices.shape[0], dtype=jnp.int32))
    new_cols = col_map[matrix_bcoo.indices[:, 1]]
    mask = new_cols >= 0
    restricted = jsparse.BCOO(
        (matrix_bcoo.data[mask], jnp.stack([matrix_bcoo.indices[mask, 0], new_cols[mask]], axis=1)),
        shape=(matrix_bcoo.shape[0], col_indices.shape[0]),
    )
    return jsparse.BCSR.from_bcoo(restricted)


def _build_restricted_extracted_mass_apply_data(
    data: ExtractedMassApplyData,
    row_indices: jnp.ndarray,
    col_indices: jnp.ndarray,
) -> RestrictedExtractedMassApplyData:
    row_indices = jnp.asarray(row_indices, dtype=jnp.int32)
    col_indices = jnp.asarray(col_indices, dtype=jnp.int32)
    return RestrictedExtractedMassApplyData(
        mass_sp=data.mass_sp,
        row_extraction=_restrict_sparse_rows(data.extraction, row_indices),
        col_extraction_t=_restrict_sparse_cols(data.extraction_t, col_indices),
        output_size=int(row_indices.shape[0]),
        input_size=int(col_indices.shape[0]),
    )


def _apply_extracted_mass_operator(extraction, extraction_t, mass_sp, x: jnp.ndarray) -> jnp.ndarray:
    raw = extraction_t @ x
    return jnp.asarray(extraction @ (mass_sp @ raw))


def _apply_extracted_mass_operator_data(data: ExtractedMassApplyData, x: jnp.ndarray) -> jnp.ndarray:
    return _apply_extracted_mass_operator(data.extraction, data.extraction_t, data.mass_sp, x)


def _apply_restricted_extracted_mass_operator_data(data: RestrictedExtractedMassApplyData, x: jnp.ndarray) -> jnp.ndarray:
    raw = data.col_extraction_t @ x
    return jnp.asarray(data.row_extraction @ (data.mass_sp @ raw))


def _apply_extracted_submatrix(data: ExtractedMassApplyData, row_indices: jnp.ndarray, col_indices: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    full = jnp.zeros((data.size,), dtype=x.dtype)
    full = full.at[col_indices].set(x)
    return _apply_extracted_mass_operator_data(data, full)[row_indices]


def _assemble_schur_inverse_from_applies(
    ass: jnp.ndarray,
    surgery_to_bulk_apply,
    bulk_apply,
    bulk_to_surgery_apply,
) -> jnp.ndarray:
    basis = jnp.eye(ass.shape[0], dtype=ass.dtype)

    def schur_apply(rhs_s: jnp.ndarray) -> jnp.ndarray:
        bulk_rhs = surgery_to_bulk_apply(rhs_s)
        bulk_response = bulk_apply(bulk_rhs)
        return ass @ rhs_s - bulk_to_surgery_apply(bulk_response)

    schur = jax.vmap(schur_apply, in_axes=1, out_axes=1)(basis)
    return _symmetrize(jnp.linalg.inv(schur))


def _apply_k0_surgery_to_bulk_coupling(surgery: K0MassSurgeryPreconditionerFactors, rhs_s: jnp.ndarray) -> jnp.ndarray:
    if surgery.surgery_to_bulk_data is not None:
        return _apply_restricted_extracted_mass_operator_data(surgery.surgery_to_bulk_data, rhs_s)
    full = jnp.zeros((surgery.apply_data.size,), dtype=rhs_s.dtype)
    full = full.at[:surgery.surgery_size].set(rhs_s)
    return _apply_extracted_mass_operator_data(surgery.apply_data, full)[surgery.surgery_size:]


def _apply_k0_bulk_to_surgery_coupling(surgery: K0MassSurgeryPreconditionerFactors, rhs_b: jnp.ndarray) -> jnp.ndarray:
    if surgery.bulk_to_surgery_data is not None:
        return _apply_restricted_extracted_mass_operator_data(surgery.bulk_to_surgery_data, rhs_b)
    full = jnp.zeros((surgery.apply_data.size,), dtype=rhs_b.dtype)
    full = full.at[surgery.surgery_size:].set(rhs_b)
    return _apply_extracted_mass_operator_data(surgery.apply_data, full)[:surgery.surgery_size]


def _apply_k1_surgery_to_bulk_coupling(surgery: K1MassSurgeryPreconditionerFactors, rhs_s: jnp.ndarray) -> jnp.ndarray:
    if surgery.surgery_to_bulk_data is not None:
        return _apply_restricted_extracted_mass_operator_data(surgery.surgery_to_bulk_data, rhs_s)
    return _apply_extracted_submatrix(surgery.apply_data, surgery.bulk_indices, surgery.surgery_indices, rhs_s)


def _apply_k1_bulk_to_surgery_coupling(surgery: K1MassSurgeryPreconditionerFactors, rhs_b: jnp.ndarray) -> jnp.ndarray:
    if surgery.bulk_to_surgery_data is not None:
        return _apply_restricted_extracted_mass_operator_data(surgery.bulk_to_surgery_data, rhs_b)
    return _apply_extracted_submatrix(surgery.apply_data, surgery.surgery_indices, surgery.bulk_indices, rhs_b)


def _apply_k1_rt_atr_coupling(surgery: K1MassSurgeryPreconditionerFactors, rhs_r: jnp.ndarray) -> jnp.ndarray:
    if surgery.rt_atr_data is not None:
        return _apply_restricted_extracted_mass_operator_data(surgery.rt_atr_data, rhs_r)
    return _apply_extracted_submatrix(surgery.apply_data, surgery.theta_bulk_indices, surgery.r_indices, rhs_r)


def _apply_k1_rt_art_coupling(surgery: K1MassSurgeryPreconditionerFactors, rhs_theta: jnp.ndarray) -> jnp.ndarray:
    if surgery.rt_art_data is not None:
        return _apply_restricted_extracted_mass_operator_data(surgery.rt_art_data, rhs_theta)
    return _apply_extracted_submatrix(surgery.apply_data, surgery.r_indices, surgery.theta_bulk_indices, rhs_theta)


def _apply_k1_rt_to_zeta_coupling(surgery: K1MassSurgeryPreconditionerFactors, rhs_rt: jnp.ndarray) -> jnp.ndarray:
    if surgery.rt_to_zeta_data is not None:
        return _apply_restricted_extracted_mass_operator_data(surgery.rt_to_zeta_data, rhs_rt)
    return _apply_extracted_submatrix(surgery.apply_data, surgery.zeta_bulk_indices, surgery.rt_indices, rhs_rt)


def _apply_k1_zeta_to_rt_coupling(surgery: K1MassSurgeryPreconditionerFactors, rhs_zeta: jnp.ndarray) -> jnp.ndarray:
    if surgery.zeta_to_rt_data is not None:
        return _apply_restricted_extracted_mass_operator_data(surgery.zeta_to_rt_data, rhs_zeta)
    return _apply_extracted_submatrix(surgery.apply_data, surgery.rt_indices, surgery.zeta_bulk_indices, rhs_zeta)


def _apply_k1_rt_preconditioner(
    surgery: K1MassSurgeryPreconditionerFactors,
    arr_factors: TensorDiagonalBlockInverseFactors,
    theta_factors: TensorDiagonalBlockInverseFactors,
    rhs_rt: jnp.ndarray,
) -> jnp.ndarray:
    rhs_r = rhs_rt[:surgery.rt_r_size]
    rhs_theta = rhs_rt[surgery.rt_r_size:surgery.rt_r_size + surgery.rt_theta_size]
    y = _apply_tensor_exact_block(None, arr_factors, rhs_r)
    z = _apply_tensor_exact_block(None, theta_factors, rhs_theta - _apply_k1_rt_atr_coupling(surgery, y))
    x_r = y - _apply_tensor_exact_block(None, arr_factors, _apply_k1_rt_art_coupling(surgery, z))
    return jnp.concatenate([x_r, z])


def _apply_k1_rt_forward_model(
    surgery: K1MassSurgeryPreconditionerFactors,
    arr_factors: TensorDiagonalBlockInverseFactors,
    theta_factors: TensorDiagonalBlockInverseFactors,
    rhs_rt: jnp.ndarray,
) -> jnp.ndarray:
    rhs_r = rhs_rt[:surgery.rt_r_size]
    rhs_theta = rhs_rt[surgery.rt_r_size:surgery.rt_r_size + surgery.rt_theta_size]
    out_r = _apply_tensor_diagonal_block_forward(arr_factors, rhs_r) + _apply_k1_rt_art_coupling(surgery, rhs_theta)
    out_theta = _apply_k1_rt_atr_coupling(surgery, rhs_r) + _apply_tensor_diagonal_block_forward(theta_factors, rhs_theta)
    return jnp.concatenate([out_r, out_theta])


def _apply_k1_bulk_preconditioner(
    surgery: K1MassSurgeryPreconditionerFactors,
    arr_factors: TensorDiagonalBlockInverseFactors,
    theta_factors: TensorDiagonalBlockInverseFactors,
    zeta_factors: TensorDiagonalBlockInverseFactors,
    rhs_bulk: jnp.ndarray,
) -> jnp.ndarray:
    rhs_rt = rhs_bulk[:surgery.bulk_rt_size]
    rhs_zeta = rhs_bulk[surgery.bulk_rt_size:surgery.bulk_rt_size + surgery.bulk_zeta_size]
    y_rt = _apply_k1_rt_preconditioner(surgery, arr_factors, theta_factors, rhs_rt)
    z = _apply_tensor_exact_block(
        None,
        zeta_factors,
        rhs_zeta - _apply_k1_rt_to_zeta_coupling(surgery, y_rt),
    )
    x_rt = y_rt - _apply_k1_rt_preconditioner(
        surgery,
        arr_factors,
        theta_factors,
        _apply_k1_zeta_to_rt_coupling(surgery, z),
    )
    return jnp.concatenate([
        x_rt,
        z,
    ])


def _apply_k1_bulk_forward_model(
    surgery: K1MassSurgeryPreconditionerFactors,
    arr_factors: TensorDiagonalBlockInverseFactors,
    theta_factors: TensorDiagonalBlockInverseFactors,
    zeta_factors: TensorDiagonalBlockInverseFactors,
    rhs_bulk: jnp.ndarray,
) -> jnp.ndarray:
    rhs_rt = rhs_bulk[:surgery.bulk_rt_size]
    rhs_zeta = rhs_bulk[surgery.bulk_rt_size:surgery.bulk_rt_size + surgery.bulk_zeta_size]
    out_rt = _apply_k1_rt_forward_model(surgery, arr_factors, theta_factors, rhs_rt) + _apply_k1_zeta_to_rt_coupling(surgery, rhs_zeta)
    out_zeta = _apply_k1_rt_to_zeta_coupling(surgery, rhs_rt) + _apply_tensor_diagonal_block_forward(zeta_factors, rhs_zeta)
    return jnp.concatenate([out_rt, out_zeta])


def _apply_k1_bulk_diagonal_preconditioner(
    surgery: K1MassSurgeryPreconditionerFactors,
    arr_factors: TensorDiagonalBlockInverseFactors,
    theta_factors: TensorDiagonalBlockInverseFactors,
    zeta_factors: TensorDiagonalBlockInverseFactors,
    rhs_bulk: jnp.ndarray,
) -> jnp.ndarray:
    rhs_r = rhs_bulk[:surgery.rt_r_size]
    rhs_theta = rhs_bulk[surgery.rt_r_size:surgery.bulk_rt_size]
    rhs_zeta = rhs_bulk[surgery.bulk_rt_size:surgery.bulk_rt_size + surgery.bulk_zeta_size]
    return jnp.concatenate([
        _apply_tensor_exact_block(None, arr_factors, rhs_r),
        _apply_tensor_exact_block(None, theta_factors, rhs_theta),
        _apply_tensor_exact_block(None, zeta_factors, rhs_zeta),
    ])


def _apply_k2_surgery_to_bulk_coupling(surgery: K2MassSurgeryPreconditionerFactors, rhs_s: jnp.ndarray) -> jnp.ndarray:
    if surgery.surgery_to_bulk_data is not None:
        return _apply_restricted_extracted_mass_operator_data(surgery.surgery_to_bulk_data, rhs_s)
    return _apply_extracted_submatrix(surgery.apply_data, surgery.bulk_indices, surgery.surgery_indices, rhs_s)


def _apply_k2_bulk_to_surgery_coupling(surgery: K2MassSurgeryPreconditionerFactors, rhs_b: jnp.ndarray) -> jnp.ndarray:
    if surgery.bulk_to_surgery_data is not None:
        return _apply_restricted_extracted_mass_operator_data(surgery.bulk_to_surgery_data, rhs_b)
    return _apply_extracted_submatrix(surgery.apply_data, surgery.surgery_indices, surgery.bulk_indices, rhs_b)


def _apply_k2_r_to_theta_coupling(surgery: K2MassSurgeryPreconditionerFactors, rhs_r: jnp.ndarray) -> jnp.ndarray:
    if surgery.r_to_theta_data is not None:
        return _apply_restricted_extracted_mass_operator_data(surgery.r_to_theta_data, rhs_r)
    return _apply_extracted_submatrix(surgery.apply_data, surgery.theta_indices, surgery.r_bulk_indices, rhs_r)


def _apply_k2_theta_to_r_coupling(surgery: K2MassSurgeryPreconditionerFactors, rhs_theta: jnp.ndarray) -> jnp.ndarray:
    if surgery.theta_to_r_data is not None:
        return _apply_restricted_extracted_mass_operator_data(surgery.theta_to_r_data, rhs_theta)
    return _apply_extracted_submatrix(surgery.apply_data, surgery.r_bulk_indices, surgery.theta_indices, rhs_theta)


def _k2_rt_indices(surgery: K2MassSurgeryPreconditionerFactors) -> jnp.ndarray:
    return jnp.concatenate([surgery.r_bulk_indices, surgery.theta_indices])


def _apply_k2_rt_to_zeta_coupling(surgery: K2MassSurgeryPreconditionerFactors, rhs_rt: jnp.ndarray) -> jnp.ndarray:
    if surgery.rt_to_zeta_data is not None:
        return _apply_restricted_extracted_mass_operator_data(surgery.rt_to_zeta_data, rhs_rt)
    return _apply_extracted_submatrix(surgery.apply_data, surgery.zeta_indices, _k2_rt_indices(surgery), rhs_rt)


def _apply_k2_zeta_to_rt_coupling(surgery: K2MassSurgeryPreconditionerFactors, rhs_zeta: jnp.ndarray) -> jnp.ndarray:
    if surgery.zeta_to_rt_data is not None:
        return _apply_restricted_extracted_mass_operator_data(surgery.zeta_to_rt_data, rhs_zeta)
    return _apply_extracted_submatrix(surgery.apply_data, _k2_rt_indices(surgery), surgery.zeta_indices, rhs_zeta)


def _apply_k2_rt_preconditioner(
    surgery: K2MassSurgeryPreconditionerFactors,
    r_bulk_factors: TensorDiagonalBlockInverseFactors,
    theta_factors: TensorDiagonalBlockInverseFactors,
    rhs_rt: jnp.ndarray,
) -> jnp.ndarray:
    rhs_r = rhs_rt[:surgery.r_bulk_size]
    rhs_theta = rhs_rt[surgery.r_bulk_size:surgery.r_bulk_size + surgery.theta_size]
    y = _apply_tensor_exact_block(None, r_bulk_factors, rhs_r)
    z = _apply_tensor_exact_block(None, theta_factors, rhs_theta - _apply_k2_r_to_theta_coupling(surgery, y))
    x_r = y - _apply_tensor_exact_block(None, r_bulk_factors, _apply_k2_theta_to_r_coupling(surgery, z))
    return jnp.concatenate([x_r, z])


def _apply_k2_rt_forward_model(
    surgery: K2MassSurgeryPreconditionerFactors,
    r_bulk_factors: TensorDiagonalBlockInverseFactors,
    theta_factors: TensorDiagonalBlockInverseFactors,
    rhs_rt: jnp.ndarray,
) -> jnp.ndarray:
    rhs_r = rhs_rt[:surgery.r_bulk_size]
    rhs_theta = rhs_rt[surgery.r_bulk_size:surgery.r_bulk_size + surgery.theta_size]
    out_r = _apply_tensor_diagonal_block_forward(r_bulk_factors, rhs_r) + _apply_k2_theta_to_r_coupling(surgery, rhs_theta)
    out_theta = _apply_k2_r_to_theta_coupling(surgery, rhs_r) + _apply_tensor_diagonal_block_forward(theta_factors, rhs_theta)
    return jnp.concatenate([out_r, out_theta])


def _apply_k2_bulk_preconditioner(
    surgery: K2MassSurgeryPreconditionerFactors,
    r_bulk_factors: TensorDiagonalBlockInverseFactors,
    theta_factors: TensorDiagonalBlockInverseFactors,
    zeta_factors: TensorDiagonalBlockInverseFactors,
    rhs_bulk: jnp.ndarray,
) -> jnp.ndarray:
    rhs_rt = rhs_bulk[:surgery.r_bulk_size + surgery.theta_size]
    rhs_zeta = rhs_bulk[surgery.r_bulk_size + surgery.theta_size:surgery.r_bulk_size + surgery.theta_size + surgery.zeta_size]
    y_rt = _apply_k2_rt_preconditioner(surgery, r_bulk_factors, theta_factors, rhs_rt)
    z = _apply_tensor_exact_block(
        None,
        zeta_factors,
        rhs_zeta - _apply_k2_rt_to_zeta_coupling(surgery, y_rt),
    )
    x_rt = y_rt - _apply_k2_rt_preconditioner(
        surgery,
        r_bulk_factors,
        theta_factors,
        _apply_k2_zeta_to_rt_coupling(surgery, z),
    )
    return jnp.concatenate([
        x_rt,
        z,
    ])


def _apply_k2_bulk_forward_model(
    surgery: K2MassSurgeryPreconditionerFactors,
    r_bulk_factors: TensorDiagonalBlockInverseFactors,
    theta_factors: TensorDiagonalBlockInverseFactors,
    zeta_factors: TensorDiagonalBlockInverseFactors,
    rhs_bulk: jnp.ndarray,
) -> jnp.ndarray:
    bulk_rt_size = surgery.r_bulk_size + surgery.theta_size
    rhs_rt = rhs_bulk[:bulk_rt_size]
    rhs_zeta = rhs_bulk[bulk_rt_size:bulk_rt_size + surgery.zeta_size]
    out_rt = _apply_k2_rt_forward_model(surgery, r_bulk_factors, theta_factors, rhs_rt) + _apply_k2_zeta_to_rt_coupling(surgery, rhs_zeta)
    out_zeta = _apply_k2_rt_to_zeta_coupling(surgery, rhs_rt) + _apply_tensor_diagonal_block_forward(zeta_factors, rhs_zeta)
    return jnp.concatenate([out_rt, out_zeta])


def _apply_k2_bulk_diagonal_preconditioner(
    surgery: K2MassSurgeryPreconditionerFactors,
    r_bulk_factors: TensorDiagonalBlockInverseFactors,
    theta_factors: TensorDiagonalBlockInverseFactors,
    zeta_factors: TensorDiagonalBlockInverseFactors,
    rhs_bulk: jnp.ndarray,
) -> jnp.ndarray:
    rhs_r = rhs_bulk[:surgery.r_bulk_size]
    rhs_theta = rhs_bulk[surgery.r_bulk_size:surgery.r_bulk_size + surgery.theta_size]
    rhs_zeta = rhs_bulk[surgery.r_bulk_size + surgery.theta_size:surgery.r_bulk_size + surgery.theta_size + surgery.zeta_size]
    return jnp.concatenate([
        _apply_tensor_exact_block(None, r_bulk_factors, rhs_r),
        _apply_tensor_exact_block(None, theta_factors, rhs_theta),
        _apply_tensor_exact_block(None, zeta_factors, rhs_zeta),
    ])


def _extract_selected_columns(seq, mass_sp, k: int, dirichlet: bool, column_indices: jnp.ndarray) -> jnp.ndarray:
    extraction = _extraction_operator(seq, k, dirichlet)
    extraction_t = _extraction_operator_transpose(seq, k, dirichlet)
    size = _extracted_size(seq, k, dirichlet)
    basis = jax.nn.one_hot(jnp.asarray(column_indices), size, dtype=jnp.float64).T
    return jax.vmap(
        lambda col: _apply_extracted_mass_operator(extraction, extraction_t, mass_sp, col),
        in_axes=1,
        out_axes=1,
    )(basis)


def build_mass_surgery_preconditioner(
    seq,
    mass_sp,
    *,
    k: int,
    existing: Optional[MassSurgeryPreconditioner] = None,
    dirichlet_flags: tuple[bool, ...] = (False, True),
) -> MassSurgeryPreconditioner:
    surgery_precond = existing if existing is not None else MassSurgeryPreconditioner()

    if k == 3:
        return surgery_precond

    pair = BoundaryConditionPair()
    if k == 0:
        surgery_size = _core_size(seq)
        for dirichlet in dirichlet_flags:
            surgery_indices = jnp.arange(surgery_size)
            surgery_cols = _extract_selected_columns(seq, mass_sp, 0, dirichlet, surgery_indices)
            ass = _symmetrize(surgery_cols[surgery_indices, :])
            apply_data = _build_extracted_mass_apply_data(seq, mass_sp, 0, dirichlet)
            bulk_indices = jnp.arange(surgery_size, apply_data.size)
            factors = K0MassSurgeryPreconditionerFactors(
                surgery_size=surgery_size,
                apply_data=apply_data,
                surgery_to_bulk_data=_build_restricted_extracted_mass_apply_data(apply_data, bulk_indices, surgery_indices),
                bulk_to_surgery_data=_build_restricted_extracted_mass_apply_data(apply_data, surgery_indices, bulk_indices),
                surgery_diaginv=1.0 / jnp.diag(ass),
                ass=ass,
            )
            pair = eqx.tree_at(
                lambda boundary_pair: boundary_pair.dbc if dirichlet else boundary_pair.free,
                pair,
                factors,
                is_leaf=lambda x: x is None,
            )
        return eqx.tree_at(lambda data: data.k0, surgery_precond, pair)

    if k == 1:
        for dirichlet in dirichlet_flags:
            block_indices = _tensor_block_indices_k1(seq, dirichlet)
            surgery_indices = block_indices["surgery"]
            bulk_indices = block_indices["bulk"]
            r_indices = block_indices["r"]
            theta_bulk_indices = block_indices["theta_bulk"]
            rt_indices = block_indices["rt"]
            zeta_bulk_indices = block_indices["zeta_bulk"]
            surgery_size = int(surgery_indices.shape[0])
            surgery_cols = _extract_selected_columns(seq, mass_sp, 1, dirichlet, surgery_indices)
            ass = _symmetrize(surgery_cols[surgery_indices, :])
            rt_r_size = int(block_indices["rt_r_size"])
            rt_theta_size = int(block_indices["rt_theta_size"])
            apply_data = _build_extracted_mass_apply_data(seq, mass_sp, 1, dirichlet)
            factors = K1MassSurgeryPreconditionerFactors(
                surgery_indices=surgery_indices,
                bulk_indices=bulk_indices,
                r_indices=r_indices,
                theta_bulk_indices=theta_bulk_indices,
                zeta_bulk_indices=zeta_bulk_indices,
                rt_indices=rt_indices,
                surgery_size=surgery_size,
                rt_r_size=rt_r_size,
                rt_theta_size=rt_theta_size,
                bulk_rt_size=int(block_indices["bulk_rt_size"]),
                bulk_zeta_size=int(block_indices["bulk_zeta_size"]),
                apply_data=apply_data,
                surgery_to_bulk_data=_build_restricted_extracted_mass_apply_data(apply_data, bulk_indices, surgery_indices),
                bulk_to_surgery_data=_build_restricted_extracted_mass_apply_data(apply_data, surgery_indices, bulk_indices),
                rt_atr_data=_build_restricted_extracted_mass_apply_data(apply_data, theta_bulk_indices, r_indices),
                rt_art_data=_build_restricted_extracted_mass_apply_data(apply_data, r_indices, theta_bulk_indices),
                rt_to_zeta_data=_build_restricted_extracted_mass_apply_data(apply_data, zeta_bulk_indices, rt_indices),
                zeta_to_rt_data=_build_restricted_extracted_mass_apply_data(apply_data, rt_indices, zeta_bulk_indices),
                surgery_diaginv=1.0 / jnp.diag(ass),
                ass=ass,
            )
            pair = eqx.tree_at(
                lambda boundary_pair: boundary_pair.dbc if dirichlet else boundary_pair.free,
                pair,
                factors,
                is_leaf=lambda x: x is None,
            )
        return eqx.tree_at(lambda data: data.k1, surgery_precond, pair)

    if k == 2:
        for dirichlet in dirichlet_flags:
            block_indices = _tensor_block_indices_k2(seq, dirichlet)
            surgery_indices = block_indices["surgery"]
            apply_data = _build_extracted_mass_apply_data(seq, mass_sp, 2, dirichlet)
            ass = _symmetrize(
                _extract_selected_columns(seq, mass_sp, 2, dirichlet, surgery_indices)[surgery_indices, :]
            )
            factors = K2MassSurgeryPreconditionerFactors(
                surgery_indices=surgery_indices,
                bulk_indices=block_indices["bulk"],
                r_bulk_indices=block_indices["r_bulk"],
                theta_indices=block_indices["theta"],
                zeta_indices=block_indices["zeta"],
                surgery_size=int(surgery_indices.shape[0]),
                r_bulk_size=int(block_indices["r_bulk_size"]),
                theta_size=int(block_indices["theta_size"]),
                zeta_size=int(block_indices["zeta_size"]),
                apply_data=apply_data,
                surgery_to_bulk_data=_build_restricted_extracted_mass_apply_data(
                    apply_data,
                    block_indices["bulk"],
                    surgery_indices,
                ),
                bulk_to_surgery_data=_build_restricted_extracted_mass_apply_data(
                    apply_data,
                    surgery_indices,
                    block_indices["bulk"],
                ),
                r_to_theta_data=_build_restricted_extracted_mass_apply_data(
                    apply_data,
                    block_indices["theta"],
                    block_indices["r_bulk"],
                ),
                theta_to_r_data=_build_restricted_extracted_mass_apply_data(
                    apply_data,
                    block_indices["r_bulk"],
                    block_indices["theta"],
                ),
                rt_to_zeta_data=_build_restricted_extracted_mass_apply_data(
                    apply_data,
                    block_indices["zeta"],
                    jnp.concatenate([block_indices["r_bulk"], block_indices["theta"]]),
                ),
                zeta_to_rt_data=_build_restricted_extracted_mass_apply_data(
                    apply_data,
                    jnp.concatenate([block_indices["r_bulk"], block_indices["theta"]]),
                    block_indices["zeta"],
                ),
                ass=ass,
                surgery_diaginv=1.0 / jnp.diag(ass),
            )
            pair = eqx.tree_at(
                lambda boundary_pair: boundary_pair.dbc if dirichlet else boundary_pair.free,
                pair,
                factors,
                is_leaf=lambda x: x is None,
            )
        return eqx.tree_at(lambda data: data.k2, surgery_precond, pair)

    raise ValueError("Mass surgery preconditioner currently only supports k=0, k=1, k=2 and k=3")


def build_mass_tensor_preconditioner(
    seq,
    full_matrix,
    *,
    k: int,
    rank: int = 1,
    fallback_rank: Optional[int] = None,
    cp_kwargs: Optional[Mapping[str, object]] = None,
    existing: Optional[TensorMassPreconditioner] = None,
    surgery_precond: Optional[MassSurgeryPreconditioner] = None,
    dirichlet_flags: tuple[bool, ...] = (False, True),
) -> TensorMassPreconditioner:
    del full_matrix
    fallback_rank = rank if fallback_rank is None else int(fallback_rank)
    cp_kwargs = {} if cp_kwargs is None else dict(cp_kwargs)
    cp_maxiter = int(cp_kwargs.get("maxiter", 100))
    cp_tol = float(cp_kwargs.get("tol", 1e-9))
    cp_ridge = float(cp_kwargs.get("ridge", 1e-12))
    block_chebyshev_steps = int(cp_kwargs.get("block_chebyshev_steps", 3))
    block_lanczos_iterations = int(cp_kwargs.get("block_lanczos_iterations", 16))
    block_lanczos_max_eig_inflation = float(cp_kwargs.get("block_lanczos_max_eig_inflation", 1.1))
    block_lanczos_min_eig_deflation = float(cp_kwargs.get("block_lanczos_min_eig_deflation", 0.85))
    block_lanczos_min_eig_floor_fraction = float(cp_kwargs.get("block_lanczos_min_eig_floor_fraction", 1e-3))
    richardson_steps = int(cp_kwargs.get("richardson_steps", 0))
    richardson_omega = float(cp_kwargs.get("richardson_omega", 1.0))
    k1_inner_schur = bool(cp_kwargs.get("k1_inner_schur", True))
    k2_inner_schur = bool(cp_kwargs.get("k2_inner_schur", True))

    reuse_existing = (
        existing is not None
        and existing.cp_maxiter == cp_maxiter
        and existing.cp_tol == cp_tol
        and existing.cp_ridge == cp_ridge
        and existing.block_chebyshev_steps == block_chebyshev_steps
        and existing.block_lanczos_iterations == block_lanczos_iterations
        and existing.block_lanczos_max_eig_inflation == block_lanczos_max_eig_inflation
        and existing.block_lanczos_min_eig_deflation == block_lanczos_min_eig_deflation
        and existing.block_lanczos_min_eig_floor_fraction == block_lanczos_min_eig_floor_fraction
        and existing.richardson_steps == richardson_steps
        and existing.richardson_omega == richardson_omega
    )
    tensor_precond = TensorMassPreconditioner(
        rank=fallback_rank,
        k0_rank=(rank if k == 0 else (existing.k0_rank if reuse_existing else fallback_rank)),
        k1_rank=(rank if k == 1 else (existing.k1_rank if reuse_existing else fallback_rank)),
        k2_rank=(rank if k == 2 else (existing.k2_rank if reuse_existing else fallback_rank)),
        k3_rank=(rank if k == 3 else (existing.k3_rank if reuse_existing else fallback_rank)),
        cp_maxiter=cp_maxiter,
        cp_tol=cp_tol,
        cp_ridge=cp_ridge,
        block_chebyshev_steps=block_chebyshev_steps,
        block_lanczos_iterations=block_lanczos_iterations,
        block_lanczos_max_eig_inflation=block_lanczos_max_eig_inflation,
        block_lanczos_min_eig_deflation=block_lanczos_min_eig_deflation,
        block_lanczos_min_eig_floor_fraction=block_lanczos_min_eig_floor_fraction,
        richardson_steps=richardson_steps,
        richardson_omega=richardson_omega,
        k0=existing.k0 if reuse_existing else BoundaryConditionPair(),
        k1=existing.k1 if reuse_existing else BoundaryConditionPair(),
        k2=existing.k2 if reuse_existing else BoundaryConditionPair(),
        k3=existing.k3 if reuse_existing else BoundaryConditionPair(),
    )

    pair = BoundaryConditionPair()
    if k == 0:
        weight_tensor = _k0_bulk_weight_tensor(seq)
        radial_baseline = _k0_radial_reference_baseline(seq)
        if surgery_precond is None:
            raise ValueError("Tensor mass k=0 requires surgery factors to be assembled first")
        for dirichlet in dirichlet_flags:
            surgery = select_boundary_data(surgery_precond.k0, dirichlet, "Mass surgery k=0")
            bulk_shape = _bulk_tensor_shape(seq, dirichlet)
            bulk_factors = _build_diagonal_tensor_block_factors(
                seq,
                weight_tensor,
                bulk_shape,
                rank,
                radial_basis=seq.basis_r_jk,
                theta_basis=seq.basis_t_jk,
                zeta_basis=seq.basis_z_jk,
                radial_weights=seq.quad.w_x,
                theta_weights=seq.quad.w_y,
                zeta_weights=seq.quad.w_z,
                radial_start=2,
                cp_maxiter=cp_maxiter,
                cp_tol=cp_tol,
                cp_ridge=cp_ridge,
                radial_baseline=radial_baseline,
                chebyshev_steps=block_chebyshev_steps,
                chebyshev_lanczos_iterations=block_lanczos_iterations,
                chebyshev_lanczos_max_eig_inflation=block_lanczos_max_eig_inflation,
                chebyshev_lanczos_min_eig_deflation=block_lanczos_min_eig_deflation,
                chebyshev_lanczos_min_eig_floor_fraction=block_lanczos_min_eig_floor_fraction,
                chebyshev_seed=100 + int(dirichlet),
                richardson_steps=richardson_steps,
                richardson_omega=richardson_omega,
            )
            bulk_factors = _annotate_tensor_block_chebyshev_bounds(
                bulk_factors,
                lanczos_iterations=block_lanczos_iterations,
                lanczos_max_eig_inflation=block_lanczos_max_eig_inflation,
                lanczos_min_eig_deflation=block_lanczos_min_eig_deflation,
                lanczos_min_eig_floor_fraction=block_lanczos_min_eig_floor_fraction,
                seed=100 + int(dirichlet),
            )
            schur_inv = _assemble_schur_inverse_from_applies(
                surgery.ass,
                lambda rhs_s, surgery=surgery: _apply_k0_surgery_to_bulk_coupling(surgery, rhs_s),
                lambda rhs_b, bulk_factors=bulk_factors: _apply_tensor_exact_block(None, bulk_factors, rhs_b),
                lambda rhs_b, surgery=surgery: _apply_k0_bulk_to_surgery_coupling(surgery, rhs_b),
            )
            factors = K0TensorMassPreconditionerFactors(
                bulk=bulk_factors,
                schur_inv=schur_inv,
            )
            pair = eqx.tree_at(
                lambda boundary_pair: boundary_pair.dbc if dirichlet else boundary_pair.free,
                pair,
                factors,
                is_leaf=lambda x: x is None,
            )
        return eqx.tree_at(lambda data: data.k0, tensor_precond, pair)

    if k == 1:
        metric_tensors = _k1_diagonal_metric_tensors(seq)
        radial_baselines = _k1_radial_reference_baselines(seq)
        if surgery_precond is None:
            raise ValueError("Tensor mass k=1 requires surgery factors to be assembled first")
        for dirichlet in dirichlet_flags:
            block_indices = _tensor_block_indices_k1(seq, dirichlet)
            surgery = select_boundary_data(surgery_precond.k1, dirichlet, "Mass surgery k=1")
            r_indices = block_indices["r"]
            theta_bulk_indices = block_indices["theta_bulk"]
            zeta_bulk_indices = block_indices["zeta_bulk"]
            rt_r_size = surgery.rt_r_size
            rt_theta_size = surgery.rt_theta_size

            arr_shape = _arr_shape_k1(seq, dirichlet)
            theta_shape = _theta_bulk_shape_k1(seq, dirichlet)
            zeta_shape = _zeta_bulk_shape_k1(seq, dirichlet)

            arr_factors = _build_diagonal_tensor_block_factors(
                seq,
                metric_tensors["alpha_rr"],
                arr_shape,
                rank,
                radial_basis=seq.d_basis_r_jk,
                theta_basis=seq.basis_t_jk,
                zeta_basis=seq.basis_z_jk,
                radial_weights=seq.quad.w_x,
                theta_weights=seq.quad.w_y,
                zeta_weights=seq.quad.w_z,
                radial_start=1,
                cp_maxiter=cp_maxiter,
                cp_tol=cp_tol,
                cp_ridge=cp_ridge,
                radial_baseline=radial_baselines[0],
                chebyshev_steps=block_chebyshev_steps,
                chebyshev_lanczos_iterations=block_lanczos_iterations,
                chebyshev_lanczos_max_eig_inflation=block_lanczos_max_eig_inflation,
                chebyshev_lanczos_min_eig_deflation=block_lanczos_min_eig_deflation,
                chebyshev_lanczos_min_eig_floor_fraction=block_lanczos_min_eig_floor_fraction,
                chebyshev_seed=200 + 10 * int(dirichlet),
                richardson_steps=richardson_steps,
                richardson_omega=richardson_omega,
            )
            arr_factors = _annotate_tensor_block_chebyshev_bounds(
                arr_factors,
                lanczos_iterations=block_lanczos_iterations,
                lanczos_max_eig_inflation=block_lanczos_max_eig_inflation,
                lanczos_min_eig_deflation=block_lanczos_min_eig_deflation,
                lanczos_min_eig_floor_fraction=block_lanczos_min_eig_floor_fraction,
                seed=200 + 10 * int(dirichlet),
            )
            theta_factors = _build_diagonal_tensor_block_factors(
                seq,
                metric_tensors["alpha_thetatheta"],
                theta_shape,
                rank,
                radial_basis=seq.basis_r_jk,
                theta_basis=seq.d_basis_t_jk,
                zeta_basis=seq.basis_z_jk,
                radial_weights=seq.quad.w_x,
                theta_weights=seq.quad.w_y,
                zeta_weights=seq.quad.w_z,
                radial_start=2,
                cp_maxiter=cp_maxiter,
                cp_tol=cp_tol,
                cp_ridge=cp_ridge,
                radial_baseline=radial_baselines[1],
                chebyshev_steps=block_chebyshev_steps,
                chebyshev_lanczos_iterations=block_lanczos_iterations,
                chebyshev_lanczos_max_eig_inflation=block_lanczos_max_eig_inflation,
                chebyshev_lanczos_min_eig_deflation=block_lanczos_min_eig_deflation,
                chebyshev_lanczos_min_eig_floor_fraction=block_lanczos_min_eig_floor_fraction,
                chebyshev_seed=201 + 10 * int(dirichlet),
                richardson_steps=richardson_steps,
                richardson_omega=richardson_omega,
            )
            theta_factors = _annotate_tensor_block_chebyshev_bounds(
                theta_factors,
                lanczos_iterations=block_lanczos_iterations,
                lanczos_max_eig_inflation=block_lanczos_max_eig_inflation,
                lanczos_min_eig_deflation=block_lanczos_min_eig_deflation,
                lanczos_min_eig_floor_fraction=block_lanczos_min_eig_floor_fraction,
                seed=201 + 10 * int(dirichlet),
            )
            zeta_factors = _build_diagonal_tensor_block_factors(
                seq,
                metric_tensors["alpha_zetazeta"],
                zeta_shape,
                rank,
                radial_basis=seq.basis_r_jk,
                theta_basis=seq.basis_t_jk,
                zeta_basis=seq.d_basis_z_jk,
                radial_weights=seq.quad.w_x,
                theta_weights=seq.quad.w_y,
                zeta_weights=seq.quad.w_z,
                radial_start=2,
                cp_maxiter=cp_maxiter,
                cp_tol=cp_tol,
                cp_ridge=cp_ridge,
                radial_baseline=radial_baselines[2],
                chebyshev_steps=block_chebyshev_steps,
                chebyshev_lanczos_iterations=block_lanczos_iterations,
                chebyshev_lanczos_max_eig_inflation=block_lanczos_max_eig_inflation,
                chebyshev_lanczos_min_eig_deflation=block_lanczos_min_eig_deflation,
                chebyshev_lanczos_min_eig_floor_fraction=block_lanczos_min_eig_floor_fraction,
                chebyshev_seed=202 + 10 * int(dirichlet),
                richardson_steps=richardson_steps,
                richardson_omega=richardson_omega,
            )
            zeta_factors = _annotate_tensor_block_chebyshev_bounds(
                zeta_factors,
                lanczos_iterations=block_lanczos_iterations,
                lanczos_max_eig_inflation=block_lanczos_max_eig_inflation,
                lanczos_min_eig_deflation=block_lanczos_min_eig_deflation,
                lanczos_min_eig_floor_fraction=block_lanczos_min_eig_floor_fraction,
                seed=202 + 10 * int(dirichlet),
            )
            schur_inv = _assemble_schur_inverse_from_applies(
                surgery.ass,
                lambda rhs_s, surgery=surgery: _apply_k1_surgery_to_bulk_coupling(surgery, rhs_s),
                lambda rhs_bulk, surgery=surgery, arr_factors=arr_factors, theta_factors=theta_factors, zeta_factors=zeta_factors, use_inner_schur=k1_inner_schur: (
                    _apply_k1_bulk_preconditioner(
                        surgery,
                        arr_factors,
                        theta_factors,
                        zeta_factors,
                        rhs_bulk,
                    ) if use_inner_schur else _apply_k1_bulk_diagonal_preconditioner(
                        surgery,
                        arr_factors,
                        theta_factors,
                        zeta_factors,
                        rhs_bulk,
                    )
                ),
                lambda rhs_bulk, surgery=surgery: _apply_k1_bulk_to_surgery_coupling(surgery, rhs_bulk),
            )

            factors = K1TensorMassPreconditionerFactors(
                r_indices=r_indices,
                theta_bulk_indices=theta_bulk_indices,
                zeta_bulk_indices=zeta_bulk_indices,
                rt_r_size=rt_r_size,
                rt_theta_size=rt_theta_size,
                use_inner_schur=k1_inner_schur,
                arr=arr_factors,
                theta=theta_factors,
                zeta=zeta_factors,
                schur_inv=schur_inv,
            )
            pair = eqx.tree_at(
                lambda boundary_pair: boundary_pair.dbc if dirichlet else boundary_pair.free,
                pair,
                factors,
                is_leaf=lambda x: x is None,
            )
        return eqx.tree_at(lambda data: data.k1, tensor_precond, pair)

    if k == 2:
        metric_tensors = _k2_diagonal_metric_tensors(seq)
        radial_baselines = _k2_radial_reference_baselines(seq)
        if surgery_precond is None:
            raise ValueError("Tensor mass k=2 requires surgery factors to be assembled first")
        for dirichlet in dirichlet_flags:
            block_indices = _tensor_block_indices_k2(seq, dirichlet)
            surgery = select_boundary_data(surgery_precond.k2, dirichlet, "Mass surgery k=2")
            r_bulk_indices = block_indices["r_bulk"]
            theta_indices = block_indices["theta"]
            zeta_indices = block_indices["zeta"]
            r_bulk_size = int(block_indices["r_bulk_size"])
            theta_size = int(block_indices["theta_size"])
            zeta_size = int(block_indices["zeta_size"])

            r_bulk_factors = _build_diagonal_tensor_block_factors(
                seq,
                metric_tensors["beta_rr"],
                _r_bulk_shape_k2(seq, dirichlet),
                rank,
                radial_basis=seq.basis_r_jk,
                theta_basis=seq.d_basis_t_jk,
                zeta_basis=seq.d_basis_z_jk,
                radial_weights=seq.quad.w_x,
                theta_weights=seq.quad.w_y,
                zeta_weights=seq.quad.w_z,
                radial_start=2,
                cp_maxiter=cp_maxiter,
                cp_tol=cp_tol,
                cp_ridge=cp_ridge,
                radial_baseline=radial_baselines[0],
                chebyshev_steps=block_chebyshev_steps,
                chebyshev_lanczos_iterations=block_lanczos_iterations,
                chebyshev_lanczos_max_eig_inflation=block_lanczos_max_eig_inflation,
                chebyshev_lanczos_min_eig_deflation=block_lanczos_min_eig_deflation,
                chebyshev_lanczos_min_eig_floor_fraction=block_lanczos_min_eig_floor_fraction,
                chebyshev_seed=300 + 10 * int(dirichlet),
                richardson_steps=richardson_steps,
                richardson_omega=richardson_omega,
            )
            r_bulk_factors = _annotate_tensor_block_chebyshev_bounds(
                r_bulk_factors,
                lanczos_iterations=block_lanczos_iterations,
                lanczos_max_eig_inflation=block_lanczos_max_eig_inflation,
                lanczos_min_eig_deflation=block_lanczos_min_eig_deflation,
                lanczos_min_eig_floor_fraction=block_lanczos_min_eig_floor_fraction,
                seed=300 + 10 * int(dirichlet),
            )
            theta_factors = _build_diagonal_tensor_block_factors(
                seq,
                metric_tensors["beta_thetatheta"],
                _theta_shape_k2(seq, dirichlet),
                rank,
                radial_basis=seq.d_basis_r_jk,
                theta_basis=seq.basis_t_jk,
                zeta_basis=seq.d_basis_z_jk,
                radial_weights=seq.quad.w_x,
                theta_weights=seq.quad.w_y,
                zeta_weights=seq.quad.w_z,
                radial_start=1,
                cp_maxiter=cp_maxiter,
                cp_tol=cp_tol,
                cp_ridge=cp_ridge,
                radial_baseline=radial_baselines[1],
                chebyshev_steps=block_chebyshev_steps,
                chebyshev_lanczos_iterations=block_lanczos_iterations,
                chebyshev_lanczos_max_eig_inflation=block_lanczos_max_eig_inflation,
                chebyshev_lanczos_min_eig_deflation=block_lanczos_min_eig_deflation,
                chebyshev_lanczos_min_eig_floor_fraction=block_lanczos_min_eig_floor_fraction,
                chebyshev_seed=301 + 10 * int(dirichlet),
                richardson_steps=richardson_steps,
                richardson_omega=richardson_omega,
            )
            theta_factors = _annotate_tensor_block_chebyshev_bounds(
                theta_factors,
                lanczos_iterations=block_lanczos_iterations,
                lanczos_max_eig_inflation=block_lanczos_max_eig_inflation,
                lanczos_min_eig_deflation=block_lanczos_min_eig_deflation,
                lanczos_min_eig_floor_fraction=block_lanczos_min_eig_floor_fraction,
                seed=301 + 10 * int(dirichlet),
            )
            zeta_factors = _build_diagonal_tensor_block_factors(
                seq,
                metric_tensors["beta_zetazeta"],
                _zeta_shape_k2(seq, dirichlet),
                rank,
                radial_basis=seq.d_basis_r_jk,
                theta_basis=seq.d_basis_t_jk,
                zeta_basis=seq.basis_z_jk,
                radial_weights=seq.quad.w_x,
                theta_weights=seq.quad.w_y,
                zeta_weights=seq.quad.w_z,
                radial_start=1,
                cp_maxiter=cp_maxiter,
                cp_tol=cp_tol,
                cp_ridge=cp_ridge,
                radial_baseline=radial_baselines[2],
                chebyshev_steps=block_chebyshev_steps,
                chebyshev_lanczos_iterations=block_lanczos_iterations,
                chebyshev_lanczos_max_eig_inflation=block_lanczos_max_eig_inflation,
                chebyshev_lanczos_min_eig_deflation=block_lanczos_min_eig_deflation,
                chebyshev_lanczos_min_eig_floor_fraction=block_lanczos_min_eig_floor_fraction,
                chebyshev_seed=302 + 10 * int(dirichlet),
                richardson_steps=richardson_steps,
                richardson_omega=richardson_omega,
            )
            zeta_factors = _annotate_tensor_block_chebyshev_bounds(
                zeta_factors,
                lanczos_iterations=block_lanczos_iterations,
                lanczos_max_eig_inflation=block_lanczos_max_eig_inflation,
                lanczos_min_eig_deflation=block_lanczos_min_eig_deflation,
                lanczos_min_eig_floor_fraction=block_lanczos_min_eig_floor_fraction,
                seed=302 + 10 * int(dirichlet),
            )
            schur_inv = _assemble_schur_inverse_from_applies(
                surgery.ass,
                lambda rhs_s, surgery=surgery: _apply_k2_surgery_to_bulk_coupling(surgery, rhs_s),
                lambda rhs_bulk, surgery=surgery, r_bulk_factors=r_bulk_factors, theta_factors=theta_factors, zeta_factors=zeta_factors, use_inner_schur=k2_inner_schur: (
                    _apply_k2_bulk_preconditioner(
                        surgery,
                        r_bulk_factors,
                        theta_factors,
                        zeta_factors,
                        rhs_bulk,
                    ) if use_inner_schur else _apply_k2_bulk_diagonal_preconditioner(
                        surgery,
                        r_bulk_factors,
                        theta_factors,
                        zeta_factors,
                        rhs_bulk,
                    )
                ),
                lambda rhs_bulk, surgery=surgery: _apply_k2_bulk_to_surgery_coupling(surgery, rhs_bulk),
            )
            factors = K2TensorMassPreconditionerFactors(
                r_bulk_indices=r_bulk_indices,
                theta_indices=theta_indices,
                zeta_indices=zeta_indices,
                r_bulk_size=r_bulk_size,
                theta_size=theta_size,
                zeta_size=zeta_size,
                use_inner_schur=k2_inner_schur,
                r_bulk=r_bulk_factors,
                theta=theta_factors,
                zeta=zeta_factors,
                schur_inv=schur_inv,
            )
            pair = eqx.tree_at(
                lambda boundary_pair: boundary_pair.dbc if dirichlet else boundary_pair.free,
                pair,
                factors,
                is_leaf=lambda x: x is None,
            )
        return eqx.tree_at(lambda data: data.k2, tensor_precond, pair)

    if k == 3:
        weight_tensor = _k3_weight_tensor(seq)
        extracted_shape = _k3_extracted_shape(seq)
        radial_baseline = _k3_radial_reference_baseline(seq)
        for dirichlet in dirichlet_flags:
            factors = _build_diagonal_tensor_block_factors(
                seq,
                weight_tensor,
                extracted_shape,
                rank,
                radial_basis=seq.d_basis_r_jk,
                theta_basis=seq.d_basis_t_jk,
                zeta_basis=seq.d_basis_z_jk,
                radial_weights=seq.quad.w_x,
                theta_weights=seq.quad.w_y,
                zeta_weights=seq.quad.w_z,
                radial_start=1,
                cp_maxiter=cp_maxiter,
                cp_tol=cp_tol,
                cp_ridge=cp_ridge,
                radial_baseline=radial_baseline,
                chebyshev_steps=block_chebyshev_steps,
                chebyshev_lanczos_iterations=block_lanczos_iterations,
                chebyshev_lanczos_max_eig_inflation=block_lanczos_max_eig_inflation,
                chebyshev_lanczos_min_eig_deflation=block_lanczos_min_eig_deflation,
                chebyshev_lanczos_min_eig_floor_fraction=block_lanczos_min_eig_floor_fraction,
                chebyshev_seed=400 + int(dirichlet),
                richardson_steps=richardson_steps,
                richardson_omega=richardson_omega,
            )
            pair = eqx.tree_at(
                lambda boundary_pair: boundary_pair.dbc if dirichlet else boundary_pair.free,
                pair,
                factors,
                is_leaf=lambda x: x is None,
            )
        return eqx.tree_at(lambda data: data.k3, tensor_precond, pair)

    raise ValueError("Tensor mass preconditioner currently only supports k=0, k=1, k=2 and k=3")


def mass_tensor_available(seq, preconds: Optional[MassPreconditioners], k: int) -> bool:
    if k not in (0, 1, 2, 3) or preconds is None or preconds.tensor is None:
        return False
    if k == 0:
        pair = preconds.tensor.k0
    elif k == 1:
        pair = preconds.tensor.k1
    elif k == 2:
        pair = preconds.tensor.k2
    else:
        pair = preconds.tensor.k3
    ready = pair.free is not None and pair.dbc is not None
    if not ready:
        return False
    if k in (0, 1, 2):
        return mass_surgery_available(seq, preconds, k)
    return True


def _select_mass_tensor_factors(preconds: Optional[MassPreconditioners], k: int, dirichlet: bool):
    if preconds is None or preconds.tensor is None:
        raise ValueError(f"Tensor mass preconditioner k={k} is not assembled")
    if k == 0:
        return select_boundary_data(preconds.tensor.k0, dirichlet, "Tensor mass k=0")
    if k == 1:
        return select_boundary_data(preconds.tensor.k1, dirichlet, "Tensor mass k=1")
    if k == 2:
        return select_boundary_data(preconds.tensor.k2, dirichlet, "Tensor mass k=2")
    if k == 3:
        return select_boundary_data(preconds.tensor.k3, dirichlet, "Tensor mass k=3")
    raise ValueError(f"Tensor mass preconditioner currently only supports k=0, k=1, k=2 and k=3 (got k={k})")


def apply_mass_tensor_preconditioner(seq, preconds: Optional[MassPreconditioners], v, k: int, dirichlet: bool = True):
    factors = _select_mass_tensor_factors(preconds, k, dirichlet)
    if k == 0:
        surgery = _select_mass_surgery_factors(preconds, k, dirichlet)
        rhs_s = v[:surgery.surgery_size]
        rhs_b = v[surgery.surgery_size:]
        y = _apply_tensor_exact_block(None, factors.bulk, rhs_b)
        z = factors.schur_inv @ (rhs_s - _apply_k0_bulk_to_surgery_coupling(surgery, y))
        x_b = y - _apply_tensor_exact_block(None, factors.bulk, _apply_k0_surgery_to_bulk_coupling(surgery, z))
        return jnp.concatenate([z, x_b])

    if k == 3:
        return _apply_tensor_diagonal_block(factors, v)

    if k == 2:
        surgery = _select_mass_surgery_factors(preconds, k, dirichlet)
        rhs_s = v[surgery.surgery_indices]
        rhs_b = v[surgery.bulk_indices]

        bulk_apply = _apply_k2_bulk_preconditioner if factors.use_inner_schur else _apply_k2_bulk_diagonal_preconditioner
        y = bulk_apply(surgery, factors.r_bulk, factors.theta, factors.zeta, rhs_b)
        z = factors.schur_inv @ (rhs_s - _apply_k2_bulk_to_surgery_coupling(surgery, y))
        x_b = y - bulk_apply(
            surgery,
            factors.r_bulk,
            factors.theta,
            factors.zeta,
            _apply_k2_surgery_to_bulk_coupling(surgery, z),
        )

        x = jnp.zeros_like(v)
        x = x.at[surgery.surgery_indices].set(z)
        x = x.at[surgery.bulk_indices].set(x_b)
        return x

    if k != 1:
        raise ValueError(f"Tensor mass preconditioner currently only supports k=0, k=1, k=2 and k=3 (got k={k})")

    surgery = _select_mass_surgery_factors(preconds, k, dirichlet)
    rhs_s = v[surgery.surgery_indices]
    rhs_b = v[surgery.bulk_indices]

    bulk_apply = _apply_k1_bulk_preconditioner if factors.use_inner_schur else _apply_k1_bulk_diagonal_preconditioner
    y = bulk_apply(surgery, factors.arr, factors.theta, factors.zeta, rhs_b)
    z = factors.schur_inv @ (rhs_s - _apply_k1_bulk_to_surgery_coupling(surgery, y))
    x_b = y - bulk_apply(
        surgery,
        factors.arr,
        factors.theta,
        factors.zeta,
        _apply_k1_surgery_to_bulk_coupling(surgery, z),
    )

    x = jnp.zeros_like(v)
    x = x.at[surgery.surgery_indices].set(z)
    x = x.at[surgery.bulk_indices].set(x_b)
    return x


def apply_mass_tensor_forward_model(seq, preconds: Optional[MassPreconditioners], v, k: int, dirichlet: bool = True):
    del seq
    factors = _select_mass_tensor_factors(preconds, k, dirichlet)

    if k == 0:
        surgery = _select_mass_surgery_factors(preconds, k, dirichlet)
        rhs_s = v[:surgery.surgery_size]
        rhs_b = v[surgery.surgery_size:]
        out_s = surgery.ass @ rhs_s + _apply_k0_bulk_to_surgery_coupling(surgery, rhs_b)
        out_b = _apply_k0_surgery_to_bulk_coupling(surgery, rhs_s) + _apply_tensor_diagonal_block_forward(factors.bulk, rhs_b)
        return jnp.concatenate([out_s, out_b])

    if k == 3:
        return _apply_tensor_diagonal_block_forward(factors, v)

    if k == 2:
        surgery = _select_mass_surgery_factors(preconds, k, dirichlet)
        rhs_s = v[surgery.surgery_indices]
        rhs_b = v[surgery.bulk_indices]
        out_s = surgery.ass @ rhs_s + _apply_k2_bulk_to_surgery_coupling(surgery, rhs_b)
        out_b = _apply_k2_surgery_to_bulk_coupling(surgery, rhs_s) + _apply_k2_bulk_forward_model(
            surgery,
            factors.r_bulk,
            factors.theta,
            factors.zeta,
            rhs_b,
        )
        out = jnp.zeros_like(v)
        out = out.at[surgery.surgery_indices].set(out_s)
        out = out.at[surgery.bulk_indices].set(out_b)
        return out

    if k != 1:
        raise ValueError(f"Tensor mass forward model currently only supports k=0, k=1, k=2 and k=3 (got k={k})")

    surgery = _select_mass_surgery_factors(preconds, k, dirichlet)
    rhs_s = v[surgery.surgery_indices]
    rhs_b = v[surgery.bulk_indices]
    out_s = surgery.ass @ rhs_s + _apply_k1_bulk_to_surgery_coupling(surgery, rhs_b)
    out_b = _apply_k1_surgery_to_bulk_coupling(surgery, rhs_s) + _apply_k1_bulk_forward_model(
        surgery,
        factors.arr,
        factors.theta,
        factors.zeta,
        rhs_b,
    )
    out = jnp.zeros_like(v)
    out = out.at[surgery.surgery_indices].set(out_s)
    out = out.at[surgery.bulk_indices].set(out_b)
    return out
def _symmetrize(matrix: jnp.ndarray) -> jnp.ndarray:
    return 0.5 * (matrix + matrix.T)


def _assemble_weighted_1d_mass(B: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    return (B * weights[None, :]) @ B.T


def _restrict_radial_mass(matrix: jnp.ndarray, radial_start: int, nr: int) -> jnp.ndarray:
    radial_stop = radial_start + nr
    if radial_start < 0 or nr < 0 or radial_stop > matrix.shape[0] or radial_stop > matrix.shape[1]:
        raise ValueError(
            f"Invalid radial restriction start={radial_start}, nr={nr} for matrix shape {matrix.shape}"
        )
    return matrix[radial_start:radial_stop, radial_start:radial_stop]


def _core_size(seq) -> int:
    return 3 * seq.basis_0.nz


def _bulk_tensor_shape(seq, dirichlet: bool) -> tuple[int, int, int]:
    nr_bulk = seq.basis_0.nr - 2 - int(dirichlet)
    nt = seq.basis_0.nt
    nz = seq.basis_0.nz
    return nr_bulk, nt, nz


def _split_blocks(matrix: jnp.ndarray, core_size: int):
    acc = matrix[:core_size, :core_size]
    acb = matrix[:core_size, core_size:]
    abc = matrix[core_size:, :core_size]
    abb = matrix[core_size:, core_size:]
    return acc, acb, abc, abb


def _k0_bulk_weight_tensor(seq) -> jnp.ndarray:
    return _reshape_quadrature_scalar_field(seq, seq.geometry.jacobian_j)


def _k3_weight_tensor(seq) -> jnp.ndarray:
    return _reshape_quadrature_scalar_field(seq, 1.0 / seq.geometry.jacobian_j)


def _k3_extracted_shape(seq) -> tuple[int, int, int]:
    return seq.basis_3.dr - 1, seq.basis_3.dt, seq.basis_3.dz


def apply_mass_rtzblock_preconditioner(seq, preconds: Optional[MassPreconditioners], rhs: jnp.ndarray, k: int, dirichlet: bool = True) -> jnp.ndarray:
    raise ValueError("rt-zblock mass preconditioner has been retired from production")