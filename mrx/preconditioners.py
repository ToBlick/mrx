from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Mapping, Optional

import equinox as eqx
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


class KroneckerMassPreconditioner(eqx.Module):
    m1d_inv_p_r: Optional[jnp.ndarray] = None
    m1d_inv_p_t: Optional[jnp.ndarray] = None
    m1d_inv_p_z: Optional[jnp.ndarray] = None
    m1d_inv_d_r: Optional[jnp.ndarray] = None
    m1d_inv_d_t: Optional[jnp.ndarray] = None
    m1d_inv_d_z: Optional[jnp.ndarray] = None
    k0_scale: Optional[jnp.ndarray] = None
    k1_scale: Optional[jnp.ndarray] = None
    k2_scale: Optional[jnp.ndarray] = None
    k3_scale: Optional[jnp.ndarray] = None


class TensorDiagonalBlockInverseFactors(eqx.Module):
    shape: tuple[int, int, int] = eqx.field(static=True)
    direct_inv_r: Optional[jnp.ndarray] = None
    direct_inv_t: Optional[jnp.ndarray] = None
    direct_inv_z: Optional[jnp.ndarray] = None
    modal_basis_r: Optional[jnp.ndarray] = None
    modal_basis_t: Optional[jnp.ndarray] = None
    modal_basis_z: Optional[jnp.ndarray] = None
    modal_r: tuple[jnp.ndarray, ...] = ()
    modal_t: tuple[jnp.ndarray, ...] = ()
    modal_z: tuple[jnp.ndarray, ...] = ()


class K0TensorMassPreconditionerFactors(eqx.Module):
    core_size: int = eqx.field(static=True)
    acb: jnp.ndarray
    abc: jnp.ndarray
    schur_inv: jnp.ndarray
    bulk: TensorDiagonalBlockInverseFactors


class K1TensorMassPreconditionerFactors(eqx.Module):
    surgery_indices: jnp.ndarray
    rt_indices: jnp.ndarray
    zeta_bulk_indices: jnp.ndarray
    surgery_size: int = eqx.field(static=True)
    rt_r_size: int = eqx.field(static=True)
    rt_theta_size: int = eqx.field(static=True)
    bulk_rt_size: int = eqx.field(static=True)
    bulk_zeta_size: int = eqx.field(static=True)
    outer_asb: jnp.ndarray
    outer_abs: jnp.ndarray
    outer_schur_inv: jnp.ndarray
    rt_atr: jnp.ndarray
    rt_art: jnp.ndarray
    arr: TensorDiagonalBlockInverseFactors
    theta: TensorDiagonalBlockInverseFactors
    zeta: TensorDiagonalBlockInverseFactors


class K2TensorMassPreconditionerFactors(eqx.Module):
    surgery_indices: jnp.ndarray
    r_bulk_indices: jnp.ndarray
    theta_indices: jnp.ndarray
    zeta_indices: jnp.ndarray
    surgery_size: int = eqx.field(static=True)
    r_bulk_size: int = eqx.field(static=True)
    theta_size: int = eqx.field(static=True)
    zeta_size: int = eqx.field(static=True)
    outer_asb: jnp.ndarray
    outer_abs: jnp.ndarray
    outer_schur_inv: jnp.ndarray
    r_bulk: TensorDiagonalBlockInverseFactors
    theta: TensorDiagonalBlockInverseFactors
    zeta: TensorDiagonalBlockInverseFactors


class TensorMassPreconditioner(eqx.Module):
    rank: int = eqx.field(static=True)
    cp_maxiter: int = eqx.field(static=True)
    cp_tol: float = eqx.field(static=True)
    cp_ridge: float = eqx.field(static=True)
    k0: BoundaryConditionPair = eqx.field(default_factory=BoundaryConditionPair)
    k1: BoundaryConditionPair = eqx.field(default_factory=BoundaryConditionPair)
    k2: BoundaryConditionPair = eqx.field(default_factory=BoundaryConditionPair)
    k3: BoundaryConditionPair = eqx.field(default_factory=BoundaryConditionPair)


class MassPreconditioners(eqx.Module):
    jacobi: Optional[JacobiMassPreconditioner] = None
    tensor: Optional[TensorMassPreconditioner] = None
    kronecker: Optional[KroneckerMassPreconditioner] = None


@dataclass(frozen=True)
class MassPreconditionerSpec:
    kind: str = 'tensor'
    steps: int = 4
    power_iterations: int = 30
    damping_safety: float = 0.8
    min_eig_fraction: float = 1e-3
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
    return MassPreconditionerSpec()


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


def set_mass_kronecker(preconds: Optional[MassPreconditioners], data: KroneckerMassPreconditioner):
    if preconds is None:
        preconds = MassPreconditioners()
    return eqx.tree_at(
        lambda payload: payload.kronecker,
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


def _assemble_1d_mass_inverse(B, w):
    M = (B * w[None, :]) @ B.T
    M_inv = jnp.linalg.inv(M)
    return 0.5 * (M_inv + M_inv.T)


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
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
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

    for _ in range(maxiter):
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
        relative_error = float(jnp.linalg.norm(reconstruction - tensor) / jnp.maximum(jnp.linalg.norm(tensor), 1.0))
        final_delta = abs(relative_error - previous_error) if previous_error < jnp.inf else jnp.inf
        previous_error = relative_error
        if final_delta < tol:
            break

    return weights, (factor_theta, factor_r, factor_z)


def _restrict_radial_mass(raw_mass_r: jnp.ndarray, radial_start: int, nr: int) -> jnp.ndarray:
    radial_stop = radial_start + nr
    if radial_start < 0 or radial_stop > raw_mass_r.shape[0]:
        raise ValueError(
            f"requested extracted radial window [{radial_start}, {radial_stop}) outside raw radial size {raw_mass_r.shape[0]}"
        )
    return raw_mass_r[radial_start:radial_stop, radial_start:radial_stop]


def _kron_apply_3(Mr_inv, Mt_inv, Mz_inv, x):
    x = jnp.einsum("ij,jkl->ikl", Mr_inv, x)
    x = jnp.einsum("ij,kjl->kil", Mt_inv, x)
    x = jnp.einsum("ij,klj->kli", Mz_inv, x)
    return x


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
    nr, nt, nz = factors.shape
    tensor = rhs.reshape(factors.shape)
    if factors.direct_inv_r is not None:
        out = _kron_apply_3(factors.direct_inv_r, factors.direct_inv_t, factors.direct_inv_z, tensor)
        return out.reshape(-1)

    modes = jnp.einsum("ji,jkl->ikl", factors.modal_basis_r, tensor)
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


def _component_sizes_k1(seq, dirichlet: bool):
    if dirichlet:
        return seq.n1_1_dbc, seq.n1_2_dbc, seq.n1_3_dbc
    return seq.n1_1, seq.n1_2, seq.n1_3


def _component_sizes_k2(seq, dirichlet: bool):
    if dirichlet:
        return seq.n2_1_dbc, seq.n2_2_dbc, seq.n2_3_dbc
    return seq.n2_1, seq.n2_2, seq.n2_3


def _surgery_slices_k1(seq, dirichlet: bool):
    n_r, n_theta, n_zeta = _component_sizes_k1(seq, dirichlet)
    r_slice = slice(0, n_r)
    theta_slice = slice(r_slice.stop, r_slice.stop + n_theta)
    zeta_slice = slice(theta_slice.stop, theta_slice.stop + n_zeta)
    theta_surgery = slice(theta_slice.start, theta_slice.start + 2 * seq.basis_1.nz)
    theta_bulk = slice(theta_surgery.stop, theta_slice.stop)
    zeta_surgery = slice(zeta_slice.start, zeta_slice.start + 3 * seq.basis_1.dz)
    zeta_bulk = slice(zeta_surgery.stop, zeta_slice.stop)
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
    n_r = _component_sizes_k1(seq, dirichlet)[0]
    nr = n_r // (nt * nz)
    if nr * nt * nz != n_r:
        raise ValueError(f"Extracted r size {n_r} is not divisible by nt*nz = {nt * nz}")
    return nr, nt, nz


def _theta_bulk_shape_k1(seq, dirichlet: bool) -> tuple[int, int, int]:
    dt = seq.basis_1.dt
    nz = seq.basis_1.nz
    n_theta = _component_sizes_k1(seq, dirichlet)[1] - 2 * seq.basis_1.nz
    nr = n_theta // (dt * nz)
    if nr * dt * nz != n_theta:
        raise ValueError(f"theta_bulk size {n_theta} is not divisible by dt*nz = {dt * nz}")
    return nr, dt, nz


def _zeta_bulk_shape_k1(seq, dirichlet: bool) -> tuple[int, int, int]:
    nt = seq.basis_1.nt
    dz = seq.basis_1.dz
    n_zeta = _component_sizes_k1(seq, dirichlet)[2] - 3 * seq.basis_1.dz
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
) -> TensorDiagonalBlockInverseFactors:
    weights, factors = _cp_als_3tensor(
        tensor,
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
        raw_mass_r = _assemble_weighted_1d_mass(radial_basis, radial_weights * (scale * factor_r))
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
            direct_inv_r=_symmetrize(jnp.linalg.inv(mass_r)),
            direct_inv_t=_symmetrize(jnp.linalg.inv(mass_t)),
            direct_inv_z=_symmetrize(jnp.linalg.inv(mass_z)),
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
        modal_basis_r=V_r,
        modal_basis_t=V_t,
        modal_basis_z=V_z,
        modal_r=modal_r,
        modal_t=modal_t,
        modal_z=modal_z,
    )


def build_mass_tensor_preconditioner(
    seq,
    full_matrix: jnp.ndarray,
    *,
    k: int,
    rank: int = 3,
    cp_kwargs: Optional[Mapping[str, object]] = None,
    existing: Optional[TensorMassPreconditioner] = None,
    dirichlet_flags: tuple[bool, ...] = (False, True),
) -> TensorMassPreconditioner:
    cp_kwargs = {} if cp_kwargs is None else dict(cp_kwargs)
    cp_maxiter = int(cp_kwargs.get("maxiter", 100))
    cp_tol = float(cp_kwargs.get("tol", 1e-9))
    cp_ridge = float(cp_kwargs.get("ridge", 1e-12))

    reuse_existing = (
        existing is not None
        and existing.rank == rank
        and existing.cp_maxiter == cp_maxiter
        and existing.cp_tol == cp_tol
        and existing.cp_ridge == cp_ridge
    )
    tensor_precond = TensorMassPreconditioner(
        rank=rank,
        cp_maxiter=cp_maxiter,
        cp_tol=cp_tol,
        cp_ridge=cp_ridge,
        k0=existing.k0 if reuse_existing else BoundaryConditionPair(),
        k1=existing.k1 if reuse_existing else BoundaryConditionPair(),
        k2=existing.k2 if reuse_existing else BoundaryConditionPair(),
        k3=existing.k3 if reuse_existing else BoundaryConditionPair(),
    )

    pair = BoundaryConditionPair()
    if k == 0:
        weight_tensor = _k0_bulk_weight_tensor(seq)
        for dirichlet in dirichlet_flags:
            e = jnp.asarray((seq.e0_dbc if dirichlet else seq.e0).todense())
            matrix = e @ full_matrix @ e.T
            core_size = _core_size(seq)
            acc, acb, abc, _ = _split_blocks(matrix, core_size)
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
            )

            def bulk_apply(rhs_bulk: jnp.ndarray) -> jnp.ndarray:
                return _apply_tensor_diagonal_block(bulk_factors, rhs_bulk)

            u_cols = [bulk_apply(abc[:, idx]) for idx in range(abc.shape[1])]
            u = jnp.stack(u_cols, axis=1)
            schur = acc - acb @ u
            factors = K0TensorMassPreconditionerFactors(
                core_size=core_size,
                acb=acb,
                abc=abc,
                schur_inv=_symmetrize(jnp.linalg.inv(schur)),
                bulk=bulk_factors,
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
        for dirichlet in dirichlet_flags:
            e = jnp.asarray((seq.e1_dbc if dirichlet else seq.e1).todense())
            matrix = e @ full_matrix @ e.T
            block_indices = _tensor_block_indices_k1(seq, dirichlet)
            surgery_indices = block_indices["surgery"]
            bulk_indices = block_indices["bulk"]
            rt_indices = block_indices["rt"]
            zeta_bulk_indices = block_indices["zeta_bulk"]
            surgery_size = int(surgery_indices.shape[0])
            ass = matrix[surgery_indices][:, surgery_indices]
            asb = matrix[surgery_indices][:, bulk_indices]
            abs_ = matrix[bulk_indices][:, surgery_indices]

            rt_block = matrix[rt_indices][:, rt_indices]
            rt_r_size = int(block_indices["rt_r_size"])
            rt_theta_size = int(block_indices["rt_theta_size"])
            rt_atr = rt_block[rt_r_size:, :rt_r_size]
            rt_art = rt_block[:rt_r_size, rt_r_size:]

            bulk_rt_size = int(block_indices["bulk_rt_size"])
            bulk_zeta_size = int(block_indices["bulk_zeta_size"])

            arr_shape = _arr_shape_k1(seq, dirichlet)
            theta_shape = _theta_bulk_shape_k1(seq, dirichlet)
            zeta = matrix[zeta_bulk_indices][:, zeta_bulk_indices]
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
            )

            def rt_apply(rhs_rt: jnp.ndarray) -> jnp.ndarray:
                rhs_r = rhs_rt[:rt_r_size]
                rhs_theta = rhs_rt[rt_r_size:rt_r_size + rt_theta_size]
                y = _apply_tensor_diagonal_block(arr_factors, rhs_r)
                z = _apply_tensor_diagonal_block(theta_factors, rhs_theta - rt_atr @ y)
                x_r = y - _apply_tensor_diagonal_block(arr_factors, rt_art @ z)
                return jnp.concatenate([x_r, z])

            def bulk_apply(rhs_bulk: jnp.ndarray) -> jnp.ndarray:
                rhs_rt = rhs_bulk[:bulk_rt_size]
                rhs_zeta = rhs_bulk[bulk_rt_size:bulk_rt_size + bulk_zeta_size]
                x_rt = rt_apply(rhs_rt)
                x_zeta = _apply_tensor_diagonal_block(zeta_factors, rhs_zeta)
                return jnp.concatenate([x_rt, x_zeta])

            u_cols = [bulk_apply(abs_[:, idx]) for idx in range(abs_.shape[1])]
            u = jnp.stack(u_cols, axis=1)
            outer_schur = ass - asb @ u
            factors = K1TensorMassPreconditionerFactors(
                surgery_indices=surgery_indices,
                rt_indices=rt_indices,
                zeta_bulk_indices=zeta_bulk_indices,
                surgery_size=surgery_size,
                rt_r_size=rt_r_size,
                rt_theta_size=rt_theta_size,
                bulk_rt_size=bulk_rt_size,
                bulk_zeta_size=bulk_zeta_size,
                outer_asb=asb,
                outer_abs=abs_,
                outer_schur_inv=_symmetrize(jnp.linalg.inv(outer_schur)),
                rt_atr=rt_atr,
                rt_art=rt_art,
                arr=arr_factors,
                theta=theta_factors,
                zeta=zeta_factors,
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
        for dirichlet in dirichlet_flags:
            e = jnp.asarray((seq.e2_dbc if dirichlet else seq.e2).todense())
            matrix = e @ full_matrix @ e.T
            block_indices = _tensor_block_indices_k2(seq, dirichlet)
            surgery_indices = block_indices["surgery"]
            bulk_indices = block_indices["bulk"]
            r_bulk_indices = block_indices["r_bulk"]
            theta_indices = block_indices["theta"]
            zeta_indices = block_indices["zeta"]

            surgery_size = int(surgery_indices.shape[0])
            r_bulk_size = int(block_indices["r_bulk_size"])
            theta_size = int(block_indices["theta_size"])
            zeta_size = int(block_indices["zeta_size"])

            ass = matrix[surgery_indices][:, surgery_indices]
            asb = matrix[surgery_indices][:, bulk_indices]
            abs_ = matrix[bulk_indices][:, surgery_indices]

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
            )

            def bulk_apply(rhs_bulk: jnp.ndarray) -> jnp.ndarray:
                rhs_r = rhs_bulk[:r_bulk_size]
                rhs_theta = rhs_bulk[r_bulk_size:r_bulk_size + theta_size]
                rhs_zeta = rhs_bulk[r_bulk_size + theta_size:r_bulk_size + theta_size + zeta_size]
                x_r = _apply_tensor_diagonal_block(r_bulk_factors, rhs_r)
                x_theta = _apply_tensor_diagonal_block(theta_factors, rhs_theta)
                x_zeta = _apply_tensor_diagonal_block(zeta_factors, rhs_zeta)
                return jnp.concatenate([x_r, x_theta, x_zeta])

            u_cols = [bulk_apply(abs_[:, idx]) for idx in range(abs_.shape[1])]
            u = jnp.stack(u_cols, axis=1)
            outer_schur = ass - asb @ u
            factors = K2TensorMassPreconditionerFactors(
                surgery_indices=surgery_indices,
                r_bulk_indices=r_bulk_indices,
                theta_indices=theta_indices,
                zeta_indices=zeta_indices,
                surgery_size=surgery_size,
                r_bulk_size=r_bulk_size,
                theta_size=theta_size,
                zeta_size=zeta_size,
                outer_asb=asb,
                outer_abs=abs_,
                outer_schur_inv=_symmetrize(jnp.linalg.inv(outer_schur)),
                r_bulk=r_bulk_factors,
                theta=theta_factors,
                zeta=zeta_factors,
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
        for dirichlet in dirichlet_flags:
            e = jnp.asarray((seq.e3_dbc if dirichlet else seq.e3).todense())
            matrix = e @ full_matrix @ e.T
            expected_size = extracted_shape[0] * extracted_shape[1] * extracted_shape[2]
            if matrix.shape != (expected_size, expected_size):
                raise ValueError(
                    f"Extracted k=3 matrix shape {matrix.shape} does not match extracted tensor shape {extracted_shape}"
                )
            factors = _build_diagonal_tensor_block_factors(
                seq,
                weight_tensor,
                extracted_shape,
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
    return pair.free is not None and pair.dbc is not None


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
        rhs_c = v[:factors.core_size]
        rhs_b = v[factors.core_size:]

        def bulk_apply(rhs_bulk: jnp.ndarray) -> jnp.ndarray:
            return _apply_tensor_diagonal_block(factors.bulk, rhs_bulk)

        y = bulk_apply(rhs_b)
        z = factors.schur_inv @ (rhs_c - factors.acb @ y)
        x_b = y - bulk_apply(factors.abc @ z)
        return jnp.concatenate([z, x_b])

    if k == 3:
        return _apply_tensor_diagonal_block(factors, v)

    if k == 2:
        rhs_s = v[factors.surgery_indices]
        rhs_r = v[factors.r_bulk_indices]
        rhs_theta = v[factors.theta_indices]
        rhs_zeta = v[factors.zeta_indices]
        rhs_b = jnp.concatenate([rhs_r, rhs_theta, rhs_zeta])

        def bulk_apply(rhs_bulk: jnp.ndarray) -> jnp.ndarray:
            rhs_r = rhs_bulk[:factors.r_bulk_size]
            rhs_theta = rhs_bulk[factors.r_bulk_size:factors.r_bulk_size + factors.theta_size]
            rhs_zeta = rhs_bulk[
                factors.r_bulk_size + factors.theta_size:
                factors.r_bulk_size + factors.theta_size + factors.zeta_size
            ]
            x_r = _apply_tensor_diagonal_block(factors.r_bulk, rhs_r)
            x_theta = _apply_tensor_diagonal_block(factors.theta, rhs_theta)
            x_zeta = _apply_tensor_diagonal_block(factors.zeta, rhs_zeta)
            return jnp.concatenate([x_r, x_theta, x_zeta])

        y = bulk_apply(rhs_b)
        z = factors.outer_schur_inv @ (rhs_s - factors.outer_asb @ y)
        x_b = y - bulk_apply(factors.outer_abs @ z)
        x = jnp.zeros_like(v)
        x = x.at[factors.surgery_indices].set(z)
        x = x.at[factors.r_bulk_indices].set(x_b[:factors.r_bulk_size])
        x = x.at[factors.theta_indices].set(
            x_b[factors.r_bulk_size:factors.r_bulk_size + factors.theta_size]
        )
        x = x.at[factors.zeta_indices].set(
            x_b[factors.r_bulk_size + factors.theta_size:]
        )
        return x

    if k != 1:
        raise ValueError(f"Tensor mass preconditioner currently only supports k=0, k=1, k=2 and k=3 (got k={k})")
    rhs_s = v[factors.surgery_indices]
    rhs_rt = v[factors.rt_indices]
    rhs_zeta = v[factors.zeta_bulk_indices]
    rhs_b = jnp.concatenate([rhs_rt, rhs_zeta])

    def rt_apply(rhs_rt: jnp.ndarray) -> jnp.ndarray:
        rhs_r = rhs_rt[:factors.rt_r_size]
        rhs_theta = rhs_rt[factors.rt_r_size:factors.rt_r_size + factors.rt_theta_size]
        y = _apply_tensor_diagonal_block(factors.arr, rhs_r)
        z = _apply_tensor_diagonal_block(factors.theta, rhs_theta - factors.rt_atr @ y)
        x_r = y - _apply_tensor_diagonal_block(factors.arr, factors.rt_art @ z)
        return jnp.concatenate([x_r, z])

    def bulk_apply(rhs_bulk: jnp.ndarray) -> jnp.ndarray:
        rhs_rt = rhs_bulk[:factors.bulk_rt_size]
        rhs_zeta = rhs_bulk[factors.bulk_rt_size:factors.bulk_rt_size + factors.bulk_zeta_size]
        x_rt = rt_apply(rhs_rt)
        x_zeta = _apply_tensor_diagonal_block(factors.zeta, rhs_zeta)
        return jnp.concatenate([x_rt, x_zeta])

    y = bulk_apply(rhs_b)
    z = factors.outer_schur_inv @ (rhs_s - factors.outer_asb @ y)
    x_b = y - bulk_apply(factors.outer_abs @ z)
    x_rt = x_b[:factors.bulk_rt_size]
    x_zeta = x_b[factors.bulk_rt_size:]
    x = jnp.zeros_like(v)
    x = x.at[factors.surgery_indices].set(z)
    x = x.at[factors.rt_indices].set(x_rt)
    x = x.at[factors.zeta_bulk_indices].set(x_zeta)
    return x


def _kron_geometric_scales(seq, k: int):
    geometry = seq.geometry
    w = seq.quad.w
    w_sum = jnp.sum(w)
    J = geometry.jacobian_j
    match k:
        case 0:
            return jnp.array([jnp.sum(J * w) / w_sum])
        case 1:
            g_inv = geometry.metric_inv_jkl
            return jnp.array([
                jnp.sum(J * g_inv[:, i, i] * w) / w_sum for i in range(3)
            ])
        case 2:
            g = geometry.metric_jkl
            return jnp.array([
                jnp.sum(g[:, i, i] / J * w) / w_sum for i in range(3)
            ])
        case 3:
            return jnp.array([jnp.sum(w / J) / w_sum])
    raise ValueError("k must be 0, 1, 2 or 3")


def build_mass_kronecker_preconditioner(seq) -> KroneckerMassPreconditioner:
    return KroneckerMassPreconditioner(
        m1d_inv_p_r=_assemble_1d_mass_inverse(seq.basis_r_jk, seq.quad.w_x),
        m1d_inv_p_t=_assemble_1d_mass_inverse(seq.basis_t_jk, seq.quad.w_y),
        m1d_inv_p_z=_assemble_1d_mass_inverse(seq.basis_z_jk, seq.quad.w_z),
        m1d_inv_d_r=_assemble_1d_mass_inverse(seq.d_basis_r_jk, seq.quad.w_x),
        m1d_inv_d_t=_assemble_1d_mass_inverse(seq.d_basis_t_jk, seq.quad.w_y),
        m1d_inv_d_z=_assemble_1d_mass_inverse(seq.d_basis_z_jk, seq.quad.w_z),
        k0_scale=_kron_geometric_scales(seq, 0),
        k1_scale=_kron_geometric_scales(seq, 1),
        k2_scale=_kron_geometric_scales(seq, 2),
        k3_scale=_kron_geometric_scales(seq, 3),
    )


def _kron_component_specs(seq, k: int):
    nr_p = seq.basis_r_jk.shape[0]
    nt_p = seq.basis_t_jk.shape[0]
    nz_p = seq.basis_z_jk.shape[0]
    nr_d = seq.d_basis_r_jk.shape[0]
    nt_d = seq.d_basis_t_jk.shape[0]
    nz_d = seq.d_basis_z_jk.shape[0]
    if k == 0:
        return [((nr_p, nt_p, nz_p), ("p", "p", "p"))]
    if k == 1:
        return [
            ((nr_d, nt_p, nz_p), ("d", "p", "p")),
            ((nr_p, nt_d, nz_p), ("p", "d", "p")),
            ((nr_p, nt_p, nz_d), ("p", "p", "d")),
        ]
    if k == 2:
        return [
            ((nr_p, nt_d, nz_d), ("p", "d", "d")),
            ((nr_d, nt_p, nz_d), ("d", "p", "d")),
            ((nr_d, nt_d, nz_p), ("d", "d", "p")),
        ]
    if k == 3:
        return [((nr_d, nt_d, nz_d), ("d", "d", "d"))]
    raise ValueError("k must be 0, 1, 2 or 3")


def _kron_inv_table(data: KroneckerMassPreconditioner):
    return {
        ("p", "r"): data.m1d_inv_p_r,
        ("p", "t"): data.m1d_inv_p_t,
        ("p", "z"): data.m1d_inv_p_z,
        ("d", "r"): data.m1d_inv_d_r,
        ("d", "t"): data.m1d_inv_d_t,
        ("d", "z"): data.m1d_inv_d_z,
    }


def mass_kronecker_available(seq, preconds: Optional[MassPreconditioners], k: int) -> bool:
    if k in (0, 3):
        return False
    if preconds is None or preconds.kronecker is None:
        return False
    table = _kron_inv_table(preconds.kronecker)
    needed = set()
    for _, kinds in _kron_component_specs(seq, k):
        for axis, kind in zip("rtz", kinds):
            needed.add((kind, axis))
    return all(table[key] is not None for key in needed)


def _kron_scale_for_k(data: KroneckerMassPreconditioner, k: int):
    match k:
        case 0:
            return data.k0_scale
        case 1:
            return data.k1_scale
        case 2:
            return data.k2_scale
        case 3:
            return data.k3_scale
    raise ValueError("k must be 0, 1, 2 or 3")


def _kron_apply_3d(Mr_inv, Mt_inv, Mz_inv, x):
    x = jnp.einsum("ij,jkl->ikl", Mr_inv, x)
    x = jnp.einsum("ij,kjl->kil", Mt_inv, x)
    x = jnp.einsum("ij,klj->kli", Mz_inv, x)
    return x


def _kron_apply_full(seq, data: KroneckerMassPreconditioner, v_full, k: int):
    table = _kron_inv_table(data)
    specs = _kron_component_specs(seq, k)
    scales = _kron_scale_for_k(data, k)
    parts = []
    offset = 0
    for i, (shape, kinds) in enumerate(specs):
        size = shape[0] * shape[1] * shape[2]
        x = v_full[offset:offset + size].reshape(shape)
        Mr_inv = table[(kinds[0], "r")]
        Mt_inv = table[(kinds[1], "t")]
        Mz_inv = table[(kinds[2], "z")]
        y = _kron_apply_3d(Mr_inv, Mt_inv, Mz_inv, x)
        if scales is not None:
            y = y / scales[i]
        parts.append(y.reshape(-1))
        offset += size
    return jnp.concatenate(parts) if len(parts) > 1 else parts[0]


def apply_mass_kronecker_preconditioner(seq, preconds: Optional[MassPreconditioners], v, k: int, dirichlet: bool = True):
    if k in (0, 3):
        raise ValueError(f"Kronecker mass preconditioner no longer supports k={k}")
    if not mass_kronecker_available(seq, preconds, k):
        raise ValueError(f"Kronecker mass preconditioner not assembled for k={k}")
    if dirichlet:
        e = getattr(seq, f"e{k}_dbc")
        e_t = getattr(seq, f"e{k}_dbc_T")
    else:
        e = getattr(seq, f"e{k}")
        e_t = getattr(seq, f"e{k}_T")
    v_full = e_t @ v
    y_full = _kron_apply_full(seq, preconds.kronecker, v_full, k)
    return e @ y_full


def _symmetrize(matrix: jnp.ndarray) -> jnp.ndarray:
    return 0.5 * (matrix + matrix.T)


def _assemble_weighted_1d_mass(B: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    return (B * weights[None, :]) @ B.T


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
    return seq.basis_3.dr - 1, seq.basis_3.dt, seq.basis_3.nz


def apply_mass_rtzblock_preconditioner(seq, preconds: Optional[MassPreconditioners], rhs: jnp.ndarray, k: int, dirichlet: bool = True) -> jnp.ndarray:
    raise ValueError("rt-zblock mass preconditioner has been retired from production")