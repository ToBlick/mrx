from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Mapping, Optional

import os

import equinox as eqx
import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import numpy as np

import mrx
from mrx.extraction_operators import MatrixFreeExtraction


class BoundaryConditionPair(eqx.Module):
    free: Optional[object] = None
    dbc: Optional[object] = None


class JacobiMassPreconditioner(eqx.Module):
    k0: BoundaryConditionPair = eqx.field(default_factory=BoundaryConditionPair)
    k1: BoundaryConditionPair = eqx.field(default_factory=BoundaryConditionPair)
    k2: BoundaryConditionPair = eqx.field(default_factory=BoundaryConditionPair)
    k3: BoundaryConditionPair = eqx.field(default_factory=BoundaryConditionPair)


class ExtractedMassApplyData(eqx.Module):
    # ``mass_apply`` is a raw-DOF-space matvec callable ``v -> M_k v`` (matrix
    # free for k=0, a BCSR-wrapped lambda for k=1/k=2). It replaces the former
    # stored BCSR ``mass_sp`` so no mass matrix needs to be materialised.
    mass_apply: object
    extraction: object
    extraction_t: object
    size: int = eqx.field(static=True)


class RestrictedExtractedMassApplyData(eqx.Module):
    mass_apply: object
    row_extraction: object
    col_extraction_t: object
    output_size: int = eqx.field(static=True)
    input_size: int = eqx.field(static=True)


class TensorDiagonalBlockInverseFactors(eqx.Module):
    shape: tuple[int, int, int] = eqx.field(static=True)
    cp_relative_error: Optional[float] = None
    cp_final_delta: Optional[float] = None
    split_backbone_relative_norm: Optional[float] = None
    split_correction_relative_norm: Optional[float] = None
    split_correction_over_backbone: Optional[float] = None
    split_backbone_residual_relative: Optional[float] = None
    chebyshev_steps: int = eqx.field(static=True, default=0)
    chebyshev_lambda_min: Optional[float] = None
    chebyshev_lambda_max: Optional[float] = None
    richardson_steps: int = eqx.field(static=True, default=0)
    # ``richardson_omega`` is computed from a Lanczos spectral estimate
    # (see ``_maybe_autotune_richardson_omega``), so it is a dynamic leaf.
    # The user-facing safety/scale knob lives on ``TensorMassPreconditioner``.
    richardson_omega: jnp.ndarray = eqx.field(
        default_factory=lambda: jnp.asarray(1.0, dtype=jnp.float64)
    )
    direct_inv_r: Optional[jnp.ndarray] = None
    direct_inv_t: Optional[jnp.ndarray] = None
    direct_inv_z: Optional[jnp.ndarray] = None
    dense_inverse: Optional[jnp.ndarray] = None
    split_backbone_inv_r: Optional[jnp.ndarray] = None
    split_backbone_inv_t: Optional[jnp.ndarray] = None
    split_backbone_inv_z: Optional[jnp.ndarray] = None
    # FD-style modal inverse data. When ``fd_V_r`` is non-None the block apply
    # projects to a per-axis mass-orthonormal basis, multiplies by the stored
    # modal pseudoinverse denominator ``fd_inv_denom``, then maps back.
    # The mass-side rank-2 path uses this for the exact ``1 + lam_r lam_t
    # lam_z`` denominator, while the stiffness-side path reuses the same
    # storage for mass-referenced modal denominators assembled from additive
    # directional terms.
    fd_V_r: Optional[jnp.ndarray] = None
    fd_V_t: Optional[jnp.ndarray] = None
    fd_V_z: Optional[jnp.ndarray] = None
    fd_lam_r: Optional[jnp.ndarray] = None
    fd_lam_t: Optional[jnp.ndarray] = None
    fd_lam_z: Optional[jnp.ndarray] = None
    fd_inv_denom: Optional[jnp.ndarray] = None
    term_r: tuple[jnp.ndarray, ...] = ()
    term_t: tuple[jnp.ndarray, ...] = ()
    term_z: tuple[jnp.ndarray, ...] = ()
    # Greville-collocation sandwich. When ``greville_inv_sqrt_D`` is non-None the
    # block inverse is D^{-1/2} (M0_r^{-1} x M0_t^{-1} x M0_z^{-1}) D^{-1/2}, with
    # UNWEIGHTED 1D mass inverses and D the metric weight collocated at the
    # component's Greville abscissae (the CP fields above are then all None).
    greville_inv_r: Optional[jnp.ndarray] = None
    greville_inv_t: Optional[jnp.ndarray] = None
    greville_inv_z: Optional[jnp.ndarray] = None
    greville_inv_sqrt_D: Optional[jnp.ndarray] = None


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
    bulk_schur: bool = eqx.field(static=True, default=False)
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
    bulk_schur: bool = eqx.field(static=True, default=False)
    schur_inv: Optional[jnp.ndarray] = None


class K0MassSurgeryPreconditionerFactors(eqx.Module):
    surgery_size: int = eqx.field(static=True)
    apply_data: ExtractedMassApplyData
    surgery_diaginv: jnp.ndarray
    ass: jnp.ndarray
    # Explicit index layout (contiguous for k=0: surgery first, then bulk) so the
    # generic surgery-Schur layer can gather/scatter and fall back to the
    # extracted-submatrix coupling uniformly with k=1/k=2.
    surgery_indices: jnp.ndarray
    bulk_indices: jnp.ndarray
    surgery_to_bulk_data: Optional[RestrictedExtractedMassApplyData] = None
    bulk_to_surgery_data: Optional[RestrictedExtractedMassApplyData] = None
    # Optional precomputed dense surgery->bulk coupling block (bulk x surgery).
    # When present, the coupling applies use dense matvecs (``coupling_sb @`` /
    # ``coupling_sb.T @``; M_0 is symmetric) instead of a full matrix-free M_0
    # apply (the restricted path still runs a whole-grid mass apply, O(n^3 p^6),
    # on a mostly-zero input). The surgery space is the polar axis (small), so
    # the block is cheap to store/probe. Mirrors the k=0 Hodge ``core_coupling``.
    coupling_sb: Optional[jnp.ndarray] = None


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
    # Optional precomputed dense surgery->bulk coupling block (bulk x surgery).
    # When present, ``_apply_surgery_to_bulk_coupling`` /
    # ``_apply_bulk_to_surgery_coupling`` use dense matvecs (``C @`` /
    # ``C.T @``; the extracted operator is symmetric) instead of a full
    # matrix-free apply of the extracted operator. Built by the stiffness
    # surgery factory (curl-curl K_1) and, when ``precompute_coupling`` is on,
    # by the mass surgery factory (M_1) too -- in both cases the O(n^3 p^6)
    # per-call apply of the restricted-sparse path is avoided. Only the
    # *surgery<->bulk* block is densified; the inner r/theta/zeta bulk<->bulk
    # couplings stay matrix-free (they are bulk-scale, not storable densely).
    coupling_sb: Optional[jnp.ndarray] = None


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
    # Optional precomputed dense surgery->bulk coupling block (bulk x surgery);
    # see ``K1MassSurgeryPreconditionerFactors.coupling_sb``. Densifies only the
    # surgery<->bulk coupling (M_2 symmetric => bulk->surgery is its transpose);
    # the inner r/theta/zeta bulk<->bulk couplings stay matrix-free.
    coupling_sb: Optional[jnp.ndarray] = None


class MassSurgeryPreconditioner(eqx.Module):
    k0: BoundaryConditionPair = eqx.field(default_factory=BoundaryConditionPair)
    k1: BoundaryConditionPair = eqx.field(default_factory=BoundaryConditionPair)
    k2: BoundaryConditionPair = eqx.field(default_factory=BoundaryConditionPair)
    k3: BoundaryConditionPair = eqx.field(default_factory=BoundaryConditionPair)


class TensorMassPreconditioner(eqx.Module):
    ranks: tuple = eqx.field(static=True, default=(3, 3, 3, 3))
    cp_maxiter: int = eqx.field(static=True, default=100)
    cp_tol: float = eqx.field(static=True, default=1e-9)
    cp_ridge: float = eqx.field(static=True, default=1e-12)
    block_chebyshev_steps: int = eqx.field(static=True, default=0)
    block_lanczos_iterations: int = eqx.field(static=True, default=16)
    block_lanczos_max_eig_inflation: float = eqx.field(static=True, default=1.1)
    block_lanczos_min_eig_deflation: float = eqx.field(static=True, default=0.85)
    block_lanczos_min_eig_floor_fraction: float = eqx.field(static=True, default=1e-3)
    richardson_steps: int = eqx.field(static=True, default=0)
    richardson_omega: float = eqx.field(static=True, default=1.0)
    surgery_schur_pinv_tol: float = eqx.field(static=True, default=1e-8)
    k0: BoundaryConditionPair = eqx.field(default_factory=BoundaryConditionPair)
    k1: BoundaryConditionPair = eqx.field(default_factory=BoundaryConditionPair)
    k2: BoundaryConditionPair = eqx.field(default_factory=BoundaryConditionPair)
    k3: BoundaryConditionPair = eqx.field(default_factory=BoundaryConditionPair)


class MassPreconditioners(eqx.Module):
    jacobi: Optional[JacobiMassPreconditioner] = None
    surgery: Optional[MassSurgeryPreconditioner] = None
    tensor: Optional[TensorMassPreconditioner] = None


def tensor_mass_rank_for_degree(tensor: TensorMassPreconditioner, k: int) -> int:
    if k not in (0, 1, 2, 3):
        raise ValueError("k must be 0, 1, 2 or 3")
    return int(tensor.ranks[k])


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
    schur_diag_mode: str = 'tensor_probe'
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


def build_mass_jacobi_pair(seq, mass_apply, k: int) -> BoundaryConditionPair:
    """Build a Jacobi (diagonal-inverse) pair for the k-form mass matrix.

    ``mass_apply`` is the raw-DOF-space matvec ``v -> M_k v`` returned by
    :func:`mrx.operators.build_matrixfree_mass_apply`.  The diagonal
    ``diag(E M_k E^T)`` is extracted by probing with canonical basis vectors
    via :func:`diag_matvec`; no assembled sparse matrix is needed.
    """
    e = getattr(seq, f"e{k}")
    e_dbc = getattr(seq, f"e{k}_dbc")
    return BoundaryConditionPair(
        free=1.0 / diag_matvec(lambda x: e @ mass_apply(e.T @ x), e.shape[0]),
        dbc=1.0 / diag_matvec(lambda x: e_dbc @ mass_apply(e_dbc.T @ x), e_dbc.shape[0]),
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


def _k1_radial_reference_baselines(seq) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # Kept for `mrx.operators` (Hodge stiffness preconditioner) which still
    # consumes radial-baseline priors. The mass tensor preconditioner no
    # longer uses any prior channels.
    safe_r = _safe_radial_quadrature(seq)
    return (
        _mean_one(safe_r),
        _mean_one(1.0 / safe_r),
        _mean_one(safe_r),
    )


def _normalize_cp_term_signs(
    scale: jnp.ndarray,
    factor_theta: jnp.ndarray,
    factor_r: jnp.ndarray,
    factor_z: jnp.ndarray,
):
    if jnp.mean(factor_theta) < 0:
        factor_theta = -factor_theta
        scale = -scale
    if jnp.mean(factor_r) < 0:
        factor_r = -factor_r
        scale = -scale
    if jnp.mean(factor_z) < 0:
        factor_z = -factor_z
        scale = -scale
    if scale < 0:
        factor_r = -factor_r
        scale = -scale
    return scale, factor_theta, factor_r, factor_z


def _make_separated_term(
    theta_factor: jnp.ndarray,
    radial_factor: jnp.ndarray,
    zeta_factor: jnp.ndarray,
    *,
    scale: float | jnp.ndarray = 1.0,
) -> dict[str, jnp.ndarray]:
    dtype = jnp.result_type(theta_factor, radial_factor, zeta_factor, scale)
    return {
        "scale": jnp.asarray(scale, dtype=dtype),
        "theta_factor": jnp.asarray(theta_factor, dtype=dtype),
        "radial_factor": jnp.asarray(radial_factor, dtype=dtype),
        "zeta_factor": jnp.asarray(zeta_factor, dtype=dtype),
    }


def _combine_separated_term_sets(
    left_terms: tuple[Mapping[str, jnp.ndarray], ...],
    right_terms: tuple[Mapping[str, jnp.ndarray], ...],
) -> tuple[dict[str, jnp.ndarray], ...]:
    combined = []
    for left in left_terms:
        for right in right_terms:
            combined.append(_make_separated_term(
                left["theta_factor"] * right["theta_factor"],
                left["radial_factor"] * right["radial_factor"],
                left["zeta_factor"] * right["zeta_factor"],
                scale=left["scale"] * right["scale"],
            ))
    return tuple(combined)


def _tensor_from_separated_terms(
    terms: tuple[Mapping[str, jnp.ndarray], ...],
    shape: tuple[int, int, int],
    dtype,
) -> jnp.ndarray:
    tensor = jnp.zeros(shape, dtype=dtype)
    for term in terms:
        tensor = tensor + (
            jnp.asarray(term["scale"], dtype=dtype)
            * jnp.asarray(term["theta_factor"], dtype=dtype)[:, None, None]
            * jnp.asarray(term["radial_factor"], dtype=dtype)[None, :, None]
            * jnp.asarray(term["zeta_factor"], dtype=dtype)[None, None, :]
        )
    return tensor


def _build_effective_prior_terms(
    shape: tuple[int, int, int],
    *,
    radial_baseline: Optional[jnp.ndarray] = None,
    prior_terms: Optional[tuple[Mapping[str, jnp.ndarray], ...]] = None,
    dtype=jnp.float64,
) -> Optional[tuple[dict[str, jnp.ndarray], ...]]:
    radial_terms = None
    if radial_baseline is not None:
        radial_terms = (
            _make_separated_term(
                jnp.ones((shape[0],), dtype=dtype),
                jnp.asarray(radial_baseline, dtype=dtype),
                jnp.ones((shape[2],), dtype=dtype),
            ),
        )

    if prior_terms is None:
        return radial_terms

    cast_prior_terms = tuple(
        _make_separated_term(
            term["theta_factor"],
            term["radial_factor"],
            term["zeta_factor"],
            scale=term["scale"],
        )
        for term in prior_terms
    )
    if radial_terms is None:
        return cast_prior_terms
    return _combine_separated_term_sets(radial_terms, cast_prior_terms)


def _expand_residual_terms_with_prior(
    residual_terms: tuple[dict[str, jnp.ndarray], ...],
    prior_terms: Optional[tuple[Mapping[str, jnp.ndarray], ...]],
) -> tuple[dict[str, jnp.ndarray], ...]:
    if prior_terms is None:
        return residual_terms
    return _combine_separated_term_sets(prior_terms, residual_terms)


def _fit_known_prior_terms(
    tensor_field: jnp.ndarray,
    *,
    rank: int,
    cp_maxiter: int,
    cp_tol: float,
    cp_ridge: float,
) -> tuple[dict[str, jnp.ndarray], ...]:
    weights, factors, _, _, _ = _cp_als_3tensor(
        tensor_field,
        rank,
        maxiter=cp_maxiter,
        tol=cp_tol,
        ridge=cp_ridge,
    )
    terms = []
    for idx in range(rank):
        factor_theta = jnp.ravel(factors[0][:, idx])
        factor_r = jnp.ravel(factors[1][:, idx])
        factor_z = jnp.ravel(factors[2][:, idx])
        scale, factor_theta, factor_r, factor_z = _normalize_cp_term_signs(
            weights[idx],
            factor_theta,
            factor_r,
            factor_z,
        )
        terms.append(_make_separated_term(
            factor_theta,
            factor_r,
            factor_z,
            scale=scale,
        ))
    return tuple(terms)


def _major_radius_tensor(seq) -> jnp.ndarray:
    mapped = jax.vmap(seq.geometry.map)(seq.quad.x)
    major_radius = jnp.sqrt(mapped[:, 0] * mapped[:, 0] + mapped[:, 1] * mapped[:, 1])
    return _mean_one(_reshape_quadrature_scalar_field(seq, major_radius))


def _major_radius_prior_terms(
    seq,
    *,
    inverse: bool,
    rank: int,
    cp_maxiter: int,
    cp_tol: float,
    cp_ridge: float,
) -> tuple[dict[str, jnp.ndarray], ...]:
    major_radius = _major_radius_tensor(seq)
    prior_tensor = 1.0 / jnp.maximum(major_radius, 1e-12) if inverse else major_radius
    prior_tensor = _mean_one(prior_tensor)
    return _fit_known_prior_terms(
        prior_tensor,
        rank=rank,
        cp_maxiter=cp_maxiter,
        cp_tol=cp_tol,
        cp_ridge=cp_ridge,
    )


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
    if factors.greville_inv_sqrt_D is not None:
        raise NotImplementedError(
            "Greville mass block has no forward-model apply (D^{1/2} M0 D^{1/2}); "
            "only the inverse sandwich is implemented. The forward model is off the "
            "solve path; wire it before enabling Chebyshev-on-greville."
        )
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
    if factors.greville_inv_sqrt_D is not None:
        # Greville sandwich: D^{-1/2} (M0_r^{-1} x M0_t^{-1} x M0_z^{-1}) D^{-1/2}.
        f = jnp.asarray(rhs).reshape(nr, nt, nz) * factors.greville_inv_sqrt_D
        f = jnp.einsum("ij,jkl->ikl", factors.greville_inv_r, f)
        f = jnp.einsum("ij,kjl->kil", factors.greville_inv_t, f)
        f = jnp.einsum("ij,klj->kli", factors.greville_inv_z, f)
        f = f * factors.greville_inv_sqrt_D
        return f.reshape(-1)
    if factors.dense_inverse is not None:
        return factors.dense_inverse @ jnp.asarray(rhs).reshape(-1)
    if factors.fd_V_r is not None:
        # Rank-2 fast-diagonalization: exact inverse of the sum of two
        # Kronecker terms. ``fd_V_*`` are the simultaneous M-orthonormal /
        # A-diagonalizing eigenvectors per axis.
        modes = jnp.asarray(rhs).reshape(nr, nt, nz)
        modes = jnp.einsum("ji,jkl->ikl", factors.fd_V_r, modes)
        modes = jnp.einsum("ji,kjl->kil", factors.fd_V_t, modes)
        modes = jnp.einsum("ji,klj->kli", factors.fd_V_z, modes)
        modes = modes * factors.fd_inv_denom
        modes = jnp.einsum("ij,jkl->ikl", factors.fd_V_r, modes)
        modes = jnp.einsum("ij,kjl->kil", factors.fd_V_t, modes)
        modes = jnp.einsum("ij,klj->kli", factors.fd_V_z, modes)
        return modes.reshape(-1)
    if factors.direct_inv_r is None:
        raise ValueError(
            "TensorDiagonalBlockInverseFactors is missing both direct_inv_* and fd_V_* "
            "(rank-1 and rank-2 fast paths). The modal/multirank smoother has been retired."
        )
    modes = jnp.asarray(rhs).reshape(nr, nt, nz)
    modes = jnp.einsum("ij,jkl->ikl", factors.direct_inv_r, modes)
    modes = jnp.einsum("ij,kjl->kil", factors.direct_inv_t, modes)
    modes = jnp.einsum("ij,klj->kli", factors.direct_inv_z, modes)
    return modes.reshape(-1)


def _apply_tensor_diagonal_block_backbone_preconditioner(
    factors: TensorDiagonalBlockInverseFactors,
    rhs: jnp.ndarray,
) -> jnp.ndarray:
    if factors.split_backbone_inv_r is None:
        return _apply_tensor_diagonal_block_preconditioner(factors, rhs)

    nr, nt, nz = factors.shape
    modes = jnp.asarray(rhs).reshape(nr, nt, nz)
    modes = jnp.einsum("ij,jkl->ikl", factors.split_backbone_inv_r, modes)
    modes = jnp.einsum("ij,kjl->kil", factors.split_backbone_inv_t, modes)
    modes = jnp.einsum("ij,klj->kli", factors.split_backbone_inv_z, modes)
    return modes.reshape(-1)


def _estimate_preconditioned_max_eigenvalue_apply(
        operator_apply, smoother_apply, size: int, *,
        n_iter: int = 10, seed: int = 0):
    """Estimate the largest Rayleigh quotient of ``S A`` via power iteration."""
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


def _project_out_vectors(vector, orthogonal_vectors=None):
    if orthogonal_vectors is None or orthogonal_vectors.shape[0] == 0:
        return vector

    def body(index, projected):
        basis_vector = orthogonal_vectors[index]
        denom = jnp.vdot(basis_vector, basis_vector).real
        coeff = jnp.where(
            denom > 0.0,
            jnp.vdot(basis_vector, projected).real / denom,
            jnp.asarray(0.0, dtype=projected.dtype),
        )
        return projected - coeff * basis_vector

    return jax.lax.fori_loop(0, orthogonal_vectors.shape[0], body, vector)


def _estimate_chebyshev_lanczos_bounds_apply(
        operator_apply, smoother_apply, size: int, *,
        spec: 'Optional[MassPreconditionerSpec]' = None,
        lanczos_iterations: Optional[int] = None,
        lanczos_max_eig_inflation: float = 1.1,
        lanczos_min_eig_deflation: float = 0.85,
        lanczos_min_eig_floor_fraction: float = 1e-3,
        seed: int = 0,
        orthogonal_vectors=None):
    """Estimate spectral bounds of the preconditioned operator ``S A`` via Lanczos.

    Either pass a :class:`MassPreconditionerSpec` via ``spec`` (the lanczos_*
    fields on the spec are used) or pass ``lanczos_iterations`` and the
    associated knobs explicitly.
    """
    if spec is not None:
        lanczos_iterations = spec.lanczos_iterations
        lanczos_max_eig_inflation = spec.lanczos_max_eig_inflation
        lanczos_min_eig_deflation = spec.lanczos_min_eig_deflation
        lanczos_min_eig_floor_fraction = spec.lanczos_min_eig_floor_fraction
    if lanczos_iterations is None or lanczos_iterations < 1:
        raise ValueError("Lanczos iteration count must be positive")

    tiny = jnp.asarray(jnp.finfo(jnp.float64).tiny, dtype=jnp.float64)

    def operator_norm(x):
        ax = operator_apply(x)
        return jnp.sqrt(jnp.maximum(jnp.abs(jnp.vdot(x, ax).real), tiny))

    vector = jax.random.normal(
        jax.random.PRNGKey(seed), (size,), dtype=jnp.float64)
    vector = _project_out_vectors(vector, orthogonal_vectors)
    init_norm = operator_norm(vector)
    vector = vector / jnp.where(init_norm > 0, init_norm, 1.0)

    def do_iteration(iteration, state):
        previous, current, beta_prev, alphas, betas, active = state

        def step(active_state):
            previous, current, beta_prev, alphas, betas, _ = active_state
            image = smoother_apply(operator_apply(current))
            alpha = jnp.real(jnp.vdot(current, operator_apply(image)))
            residual = image - alpha * current
            residual = residual - jnp.where(iteration > 0, beta_prev, 0.0) * previous
            residual = _project_out_vectors(residual, orthogonal_vectors)
            beta = operator_norm(residual)

            alphas = alphas.at[iteration].set(alpha)
            continue_iteration = (iteration + 1 < lanczos_iterations) & (beta > tiny)
            betas = betas.at[iteration].set(jnp.where(continue_iteration, beta, 0.0))

            safe_beta = jnp.where(beta > 0.0, beta, 1.0)
            next_current = residual / safe_beta
            previous = jnp.where(continue_iteration, current, previous)
            current = jnp.where(continue_iteration, next_current, current)
            beta_prev = jnp.where(continue_iteration, beta, beta_prev)
            return previous, current, beta_prev, alphas, betas, continue_iteration

        return jax.lax.cond(active, step, lambda inactive_state: inactive_state, state)

    initial_state = (
        jnp.zeros_like(vector),
        vector,
        jnp.asarray(0.0, dtype=jnp.float64),
        jnp.zeros((lanczos_iterations,), dtype=jnp.float64),
        jnp.zeros((lanczos_iterations,), dtype=jnp.float64),
        jnp.asarray(True),
    )
    _, _, _, alphas, betas, _ = jax.lax.fori_loop(
        0,
        lanczos_iterations,
        do_iteration,
        initial_state,
    )

    tridiagonal = jnp.diag(alphas)
    offdiag = betas[:-1]
    tridiagonal = tridiagonal + jnp.diag(offdiag, k=1) + jnp.diag(offdiag, k=-1)
    ritz_values = jnp.linalg.eigvalsh(tridiagonal)
    max_ritz = jnp.maximum(ritz_values[-1], tiny)
    max_eig = jnp.maximum(
        jnp.asarray(lanczos_max_eig_inflation, dtype=jnp.float64) * max_ritz,
        tiny,
    )
    floor = jnp.asarray(
        lanczos_min_eig_floor_fraction, dtype=jnp.float64
    ) * max_eig
    min_positive_ritz = jnp.min(jnp.where(ritz_values > tiny, ritz_values, jnp.inf))
    guarded_min = jnp.asarray(
        lanczos_min_eig_deflation, dtype=jnp.float64
    ) * min_positive_ritz
    min_eig = jnp.where(
        jnp.isfinite(min_positive_ritz),
        jnp.maximum(floor, guarded_min),
        floor,
    )
    return min_eig, max_eig


def _estimate_richardson_omega_apply(
    operator_apply,
    smoother_apply,
    size: int,
    *,
    lanczos_iterations: int,
    lanczos_max_eig_inflation: float,
    lanczos_min_eig_deflation: float,
    lanczos_min_eig_floor_fraction: float,
    seed: int = 0,
) -> float:
    """Auto-tune the Richardson relaxation parameter via Lanczos.

    Estimates the spectral bounds of the preconditioned operator
    ``S A`` (where ``A = operator_apply`` and ``S = smoother_apply``) and
    returns the optimal Richardson weight ``omega = 2 / (lambda_min + lambda_max)``
    for SPD systems. ``S`` and ``A`` are both required to be SPD so that
    ``S A`` has positive real eigenvalues.
    """
    lambda_min, lambda_max = _estimate_chebyshev_lanczos_bounds_apply(
        operator_apply,
        smoother_apply,
        size,
        lanczos_iterations=lanczos_iterations,
        lanczos_max_eig_inflation=lanczos_max_eig_inflation,
        lanczos_min_eig_deflation=lanczos_min_eig_deflation,
        lanczos_min_eig_floor_fraction=lanczos_min_eig_floor_fraction,
        seed=seed,
    )
    denom = jnp.maximum(lambda_min + lambda_max,
                        jnp.asarray(jnp.finfo(jnp.float64).tiny, dtype=jnp.float64))
    return float(2.0 / denom)


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


def _apply_tensor_diagonal_block(
    factors: TensorDiagonalBlockInverseFactors,
    rhs: jnp.ndarray,
    *,
    true_block_apply=None,
) -> jnp.ndarray:
    """Apply the tensor diagonal block inverse, optionally with true-block Richardson.

    When ``true_block_apply`` is provided AND ``richardson_steps > 0``, the
    Richardson residual is computed against the true (extracted-mass restricted
    to this block's indices) operator, with the rank-1 backbone tensor inverse
    serving as the smoother. This is the only configuration in which extra
    Richardson sweeps actually attack the geometric coupling that the CP fit
    cannot represent. When ``true_block_apply`` is None, the residual falls
    back to the rank-r CP forward model (legacy behavior; mostly a no-op for
    rank 1, hence the parameter).
    """
    smoother_apply = _apply_tensor_diagonal_block_preconditioner
    if factors.split_backbone_inv_r is not None and factors.richardson_steps > 0:
        smoother_apply = _apply_tensor_diagonal_block_backbone_preconditioner

    x = smoother_apply(factors, rhs)
    if factors.richardson_steps <= 0:
        return x

    if true_block_apply is None:
        forward_apply = lambda y, block_factors=factors: _apply_tensor_diagonal_block_forward(block_factors, y)
    else:
        forward_apply = true_block_apply

    for _ in range(factors.richardson_steps):
        residual = rhs - forward_apply(x)
        x = x + factors.richardson_omega * smoother_apply(factors, residual)
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


def _greedy_cp_terms(
    tensor: jnp.ndarray,
    *,
    rank: int,
    cp_maxiter: int,
    cp_tol: float,
    cp_ridge: float,
) -> tuple[tuple[dict[str, jnp.ndarray], ...], float, float]:
    """Greedy rank-r CP fit: r sequential rank-1 ALS fits against the residual.

    Returns ``(terms, relative_error, last_step_residual_drop)`` where
    ``terms`` is a tuple of ``_make_separated_term`` dicts of length ``rank``,
    ``relative_error = ||tensor - sum(terms)|| / max(||tensor||, 1)``, and
    ``last_step_residual_drop`` is the drop in residual norm at the final
    rank-1 step (useful as a convergence diagnostic).

    Greedy rank-r is monotone (rank-(r+1) strictly extends rank-r) and
    deterministic, which is what we want for a preconditioner: rank-1
    output is a strict subset of the rank-2 result, etc. Joint rank-r CP
    can give a slightly tighter fit at the cost of non-uniqueness and
    randomized restarts; we trade that for monotonicity here.
    """
    if rank < 1:
        raise ValueError(f"_greedy_cp_terms requires rank >= 1; got {rank}.")
    terms: list[dict[str, jnp.ndarray]] = []
    residual = tensor
    last_drop = 0.0
    tensor_norm = jnp.maximum(jnp.linalg.norm(tensor), 1.0)
    for _ in range(rank):
        weights, factors, _, _, _ = _cp_als_3tensor(
            residual,
            1,
            maxiter=cp_maxiter,
            tol=cp_tol,
            ridge=cp_ridge,
        )
        factor_theta = jnp.ravel(factors[0][:, 0])
        factor_r = jnp.ravel(factors[1][:, 0])
        factor_z = jnp.ravel(factors[2][:, 0])
        scale, factor_theta, factor_r, factor_z = _normalize_cp_term_signs(
            weights[0], factor_theta, factor_r, factor_z,
        )
        new_term = _make_separated_term(
            factor_theta, factor_r, factor_z, scale=scale,
        )
        terms.append(new_term)
        prev_norm = jnp.linalg.norm(residual)
        residual = residual - _tensor_from_separated_terms(
            (new_term,), tensor.shape, tensor.dtype,
        )
        new_norm = jnp.linalg.norm(residual)
        last_drop = float(prev_norm - new_norm)
    relative_error = float(jnp.linalg.norm(residual) / tensor_norm)
    return tuple(terms), relative_error, last_drop


def _cp_ntf_3tensor(
    tensor: jnp.ndarray,
    rank: int,
    *,
    maxiter: int,
    tol: float,
    eps: float = 1e-12,
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], float, float, int]:
    """Joint rank-r NON-NEGATIVE CP (NTF) of a non-negative 3-tensor via
    Lee-Seung multiplicative updates.

    Contrast with :func:`_greedy_cp_terms`, which fits ``rank`` sequential
    rank-1 terms against a *residual*: the first term subtracts a rank-1 piece
    from the positive weight tensor, so every later term fits a sign-indefinite
    target and necessarily has sign-changing factors. Those indefinite factors
    are what make the assembled rank>=2 Kronecker surrogate indefinite (failed
    Cholesky anchor / non-positive FD denominator) on a non-separable (W7-X)
    metric.

    Fitting all ``rank`` terms jointly with the multiplicative update keeps
    every factor >= 0. A non-negative factor ``w`` gives a per-axis weighted
    mass ``B diag(quad_w * w) B^T`` that is SPSD, so the Kronecker sum is SPD by
    construction -- the rank>=2 fast-diagonalization anchor is SPD (Cholesky
    valid) and its generalized eigenvalues are >= 0, so the FD denominator
    ``1 + lam_r lam_t lam_z >= 1 > 0`` with no clamp and no dense fallback.
    """
    if tensor.ndim != 3:
        raise ValueError(f"NTF expects a 3-tensor, got shape {tensor.shape}")
    if rank < 1 or rank > min(tensor.shape):
        raise ValueError(f"Requested NTF rank {rank} outside valid range [1, {min(tensor.shape)}]")

    # Metric/Jacobian weight tensors are positive; clip tiny negative
    # interpolation noise so the multiplicative updates stay well-defined.
    tensor = jnp.maximum(tensor, 0.0)
    unfolded = [_mode_unfold_3tensor(tensor, mode) for mode in range(3)]

    # Deterministic non-negative init from |leading singular vectors|.
    factors = [
        jnp.abs(jnp.linalg.svd(unfolded[mode], full_matrices=False)[0][:, :rank]) + eps
        for mode in range(3)
    ]

    tensor_norm = jnp.maximum(jnp.linalg.norm(tensor), 1.0)
    previous_error = jnp.inf
    relative_error = jnp.inf
    final_delta = jnp.inf
    n_iterations = 0
    for iteration in range(maxiter):
        for mode in range(3):
            others = [factors[axis] for axis in range(3) if axis != mode]
            khatri_rao = _khatri_rao(others[0], others[1])
            numerator = unfolded[mode] @ khatri_rao
            gram = (others[0].T @ others[0]) * (others[1].T @ others[1])
            denominator = factors[mode] @ gram
            factors[mode] = factors[mode] * numerator / (denominator + eps)

        reconstruction = _reconstruct_cp_3tensor(
            jnp.ones((rank,), dtype=tensor.dtype), tuple(factors),
        )
        relative_error = float(jnp.linalg.norm(reconstruction - tensor) / tensor_norm)
        final_delta = abs(relative_error - previous_error) if previous_error < jnp.inf else jnp.inf
        previous_error = relative_error
        n_iterations = iteration + 1
        if final_delta < tol:
            break

    # Pull per-column norms into weights; factors stay unit-norm and >= 0.
    weights = jnp.ones((rank,), dtype=tensor.dtype)
    for mode in range(3):
        factors[mode], norms = _normalize_cp_columns(factors[mode])
        weights = weights * norms
    return weights, (factors[0], factors[1], factors[2]), relative_error, final_delta, n_iterations


def _ntf_terms(
    tensor: jnp.ndarray,
    *,
    rank: int,
    cp_maxiter: int,
    cp_tol: float,
) -> tuple[tuple[dict[str, jnp.ndarray], ...], float, float]:
    """Joint non-negative CP terms -- drop-in replacement for the output of
    :func:`_greedy_cp_terms` but with every factor (and scale) >= 0, yielding an
    SPD-by-construction Kronecker surrogate at any rank. See
    :func:`_cp_ntf_3tensor`."""
    weights, (factor_0, factor_1, factor_2), relative_error, final_delta, _ = _cp_ntf_3tensor(
        tensor, rank, maxiter=cp_maxiter, tol=cp_tol,
    )
    terms = tuple(
        _make_separated_term(factor_0[:, k], factor_1[:, k], factor_2[:, k], scale=weights[k])
        for k in range(rank)
    )
    return terms, relative_error, float(final_delta)


def _build_tensor_block_factors_from_terms(
    *,
    full_shape: tuple[int, int, int],
    term_matrices: tuple[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], ...],
    cp_relative_error: Optional[float],
    cp_final_delta: Optional[float],
    chebyshev_steps: int = 0,
    chebyshev_lanczos_iterations: int = 16,
    chebyshev_lanczos_max_eig_inflation: float = 1.1,
    chebyshev_lanczos_min_eig_deflation: float = 0.85,
    chebyshev_lanczos_min_eig_floor_fraction: float = 1e-3,
    chebyshev_seed: int = 0,
    richardson_steps: int = 0,
    richardson_omega: float = 1.0,
    true_block_apply=None,
) -> TensorDiagonalBlockInverseFactors:
    if len(term_matrices) < 1:
        raise ValueError("Tensor block factor builder requires at least one Kronecker term")

    def _direct_axis_inverses(
        matrices: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        *,
        pseudo: bool = False,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        inverse = jnp.linalg.pinv if pseudo else jnp.linalg.inv
        return tuple(_symmetrize(inverse(matrix)) for matrix in matrices)

    def _assemble_dense_surrogate(
        matrices: tuple[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], ...],
    ) -> jnp.ndarray:
        shape = matrices[0][0].shape[0] * matrices[0][1].shape[0] * matrices[0][2].shape[0]
        dense = jnp.zeros((shape, shape), dtype=jnp.float64)
        for matrix_r, matrix_t, matrix_z in matrices:
            dense = dense + jnp.kron(matrix_z, jnp.kron(matrix_t, matrix_r))
        return _symmetrize(dense)

    def _dense_surrogate_inverse(
        matrices: tuple[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], ...],
    ) -> jnp.ndarray:
        dense = _assemble_dense_surrogate(matrices)
        evals = jnp.linalg.eigvalsh(dense)
        max_abs_eval = jnp.max(jnp.abs(evals))
        tol = jnp.maximum(jnp.asarray(1e-10, dtype=jnp.float64), 1e-12 * max_abs_eval)
        if bool(jnp.min(jnp.abs(evals)) <= tol):
            return _symmetrize(jnp.linalg.pinv(dense))
        return _symmetrize(jnp.linalg.inv(dense))

    # Vestigial split-backbone metadata (always None now that priors are gone).
    split_backbone_relative_norm = None
    split_correction_relative_norm = None
    split_correction_over_backbone = None
    split_backbone_residual_relative = None
    split_backbone_inv_r = None
    split_backbone_inv_t = None
    split_backbone_inv_z = None

    if len(term_matrices) == 1:
        mass_r, mass_t, mass_z = term_matrices[0]
        fd_V_r = fd_V_t = fd_V_z = None
        fd_lam_r = fd_lam_t = fd_lam_z = None
        fd_inv_denom = None
        dense_inverse = None
        # SPD-project each per-axis inverse: a no-op for well-conditioned SPD
        # blocks (cylinder exactness, W7-X <= p3) but lifts the indefinite
        # factors that arise when a greedy-CP weight changes sign on a
        # non-separable metric, keeping the rank-1 preconditioner SPD.
        direct_inv_r = _spd_clamped_inverse(mass_r)
        direct_inv_t = _spd_clamped_inverse(mass_t)
        direct_inv_z = _spd_clamped_inverse(mass_z)
    else:
        mass_r_0, mass_t_0, mass_z_0 = term_matrices[0]
        mass_r_1, mass_t_1, mass_z_1 = term_matrices[1]
        dense_inverse = None
        try:
            fd = _build_kron_sum_fd_factors(
                mass_r_0, mass_t_0, mass_z_0, mass_r_1, mass_t_1, mass_z_1,
            )
            fd_V_r, fd_V_t, fd_V_z = fd["V_r"], fd["V_t"], fd["V_z"]
            fd_lam_r, fd_lam_t, fd_lam_z = fd["lam_r"], fd["lam_t"], fd["lam_z"]
            denom = (
                1.0
                + fd_lam_r[:, None, None] * fd_lam_t[None, :, None] * fd_lam_z[None, None, :]
            )
            for idx in range(2, len(term_matrices)):
                mass_r_i, mass_t_i, mass_z_i = term_matrices[idx]
                d_r = jnp.einsum("ji,jk,ki->i", fd_V_r, mass_r_i, fd_V_r)
                d_t = jnp.einsum("ji,jk,ki->i", fd_V_t, mass_t_i, fd_V_t)
                d_z = jnp.einsum("ji,jk,ki->i", fd_V_z, mass_z_i, fd_V_z)
                denom = denom + d_r[:, None, None] * d_t[None, :, None] * d_z[None, None, :]
            min_denom = float(jnp.min(denom))
            if not jnp.isfinite(min_denom) or min_denom <= 0.0:
                raise ValueError(
                    "Diagonal-truncated rank-r Kronecker sum is not SPD: "
                    f"min(denom) = {min_denom:.3e}. Reduce rank or check the "
                    "diagonal-metric tensor."
                )
            fd_inv_denom = 1.0 / denom
            direct_inv_r = direct_inv_t = direct_inv_z = None
        except ValueError:
            # Greedy CP terms need not preserve SPD per-axis factors even when
            # the assembled surrogate block is invertible. Fall back to a dense
            # inverse/pseudoinverse of the assembled surrogate block instead of
            # aborting or degrading to a single Kronecker term.
            fd_V_r = fd_V_t = fd_V_z = None
            fd_lam_r = fd_lam_t = fd_lam_z = None
            fd_inv_denom = None
            direct_inv_r = direct_inv_t = direct_inv_z = None
            dense_inverse = _dense_surrogate_inverse(term_matrices)

    return _maybe_autotune_richardson_omega(
        TensorDiagonalBlockInverseFactors(
            shape=full_shape,
            cp_relative_error=cp_relative_error,
            cp_final_delta=cp_final_delta,
            split_backbone_relative_norm=split_backbone_relative_norm,
            split_correction_relative_norm=split_correction_relative_norm,
            split_correction_over_backbone=split_correction_over_backbone,
            split_backbone_residual_relative=split_backbone_residual_relative,
            chebyshev_steps=chebyshev_steps,
            chebyshev_lambda_min=None,
            chebyshev_lambda_max=None,
            richardson_steps=richardson_steps,
            richardson_omega=jnp.asarray(richardson_omega, dtype=jnp.float64),
            direct_inv_r=direct_inv_r,
            direct_inv_t=direct_inv_t,
            direct_inv_z=direct_inv_z,
            dense_inverse=dense_inverse,
            split_backbone_inv_r=split_backbone_inv_r,
            split_backbone_inv_t=split_backbone_inv_t,
            split_backbone_inv_z=split_backbone_inv_z,
            fd_V_r=fd_V_r,
            fd_V_t=fd_V_t,
            fd_V_z=fd_V_z,
            fd_lam_r=fd_lam_r,
            fd_lam_t=fd_lam_t,
            fd_lam_z=fd_lam_z,
            fd_inv_denom=fd_inv_denom,
            term_r=tuple(t[0] for t in term_matrices),
            term_t=tuple(t[1] for t in term_matrices),
            term_z=tuple(t[2] for t in term_matrices),
        ),
        lanczos_iterations=chebyshev_lanczos_iterations,
        lanczos_max_eig_inflation=chebyshev_lanczos_max_eig_inflation,
        lanczos_min_eig_deflation=chebyshev_lanczos_min_eig_deflation,
        lanczos_min_eig_floor_fraction=chebyshev_lanczos_min_eig_floor_fraction,
        seed=chebyshev_seed,
        true_block_apply=true_block_apply,
    )


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
    prior_terms: Optional[tuple[Mapping[str, jnp.ndarray], ...]] = None,
    chebyshev_steps: int = 0,
    chebyshev_lanczos_iterations: int = 16,
    chebyshev_lanczos_max_eig_inflation: float = 1.1,
    chebyshev_lanczos_min_eig_deflation: float = 0.85,
    chebyshev_lanczos_min_eig_floor_fraction: float = 1e-3,
    chebyshev_seed: int = 0,
    richardson_steps: int = 0,
    richardson_omega: float = 1.0,
    true_block_apply=None,
) -> TensorDiagonalBlockInverseFactors:
    # Tensor preconditioner: greedy rank-r CP fit (sequential rank-1 ALS
    # against the residual) of the diagonal-metric tensor on the quadrature
    # grid. The preconditioner is then assembled as:
    #   rank=1   single Kronecker block (direct per-axis inverse);
    #   rank=2   sum of two Kronecker terms, EXACT via Lynch fast-
    #            diagonalization (simultaneous (M, A) generalized eigh);
    #   rank>=3  Lynch FD on the leading two terms (defines V_r/V_t/V_z);
    #            the additional terms are projected into that basis and
    #            their *diagonals* are added to the FD denominator. This
    #            is no longer exact for the assembled CP fit (off-diagonals
    #            in V are dropped), but every rank>=3 apply costs the same
    #            6 einsums as rank=2.
    # Geometry/prior channels are intentionally NOT used: the preconditioner
    # treats the diagonal metric tensor as a black box.
    del radial_baseline, prior_terms  # accepted for API compat; unused
    if rank < 1:
        raise ValueError(
            f"Tensor diagonal block builder requires rank >= 1; got {rank}."
        )
    nr, nt, nz = full_shape

    # Default: joint non-negative factorization (NTF) of the diagonal-metric
    # tensor. NTF keeps every factor >= 0, so each per-axis weighted mass
    # B diag(quad_w * factor) B^T is SPSD and the assembled Kronecker surrogate
    # is SPD by construction at ANY rank -- one PSD-by-construction path for
    # rank 1 and rank 2 alike (no sign-flipped factors -> no indefinite rank-2
    # FD anchor/denominator, and no reliance on the rank-1 SPD-clamp fallback).
    # MRX_CP_GREEDY=1 restores the legacy unconstrained greedy rank-1 ALS fit
    # for A/B comparison. See _cp_ntf_3tensor.
    if os.environ.get("MRX_CP_GREEDY", "0") == "1":
        expanded_terms, cp_relative_error, cp_final_delta = _greedy_cp_terms(
            tensor,
            rank=rank,
            cp_maxiter=cp_maxiter,
            cp_tol=cp_tol,
            cp_ridge=cp_ridge,
        )
    else:
        expanded_terms, cp_relative_error, cp_final_delta = _ntf_terms(
            tensor,
            rank=rank,
            cp_maxiter=cp_maxiter,
            cp_tol=cp_tol,
        )

    term_data = []
    for term in expanded_terms:
        radial_weight = term["scale"] * term["radial_factor"]
        raw_mass_r = _assemble_weighted_1d_mass(radial_basis, radial_weights * radial_weight)
        mass_r = _restrict_radial_mass(raw_mass_r, radial_start, nr)
        mass_t = _assemble_weighted_1d_mass(theta_basis, theta_weights * term["theta_factor"])
        mass_z = _assemble_weighted_1d_mass(zeta_basis, zeta_weights * term["zeta_factor"])
        term_data.append((mass_r, mass_t, mass_z))

    return _build_tensor_block_factors_from_terms(
        full_shape=full_shape,
        term_matrices=tuple(term_data),
        cp_relative_error=cp_relative_error,
        cp_final_delta=cp_final_delta,
        chebyshev_steps=chebyshev_steps,
        chebyshev_lanczos_iterations=chebyshev_lanczos_iterations,
        chebyshev_lanczos_max_eig_inflation=chebyshev_lanczos_max_eig_inflation,
        chebyshev_lanczos_min_eig_deflation=chebyshev_lanczos_min_eig_deflation,
        chebyshev_lanczos_min_eig_floor_fraction=chebyshev_lanczos_min_eig_floor_fraction,
        chebyshev_seed=chebyshev_seed,
        richardson_steps=richardson_steps,
        richardson_omega=richardson_omega,
        true_block_apply=true_block_apply,
    )


def _maybe_autotune_richardson_omega(
    factors: TensorDiagonalBlockInverseFactors,
    *,
    lanczos_iterations: int,
    lanczos_max_eig_inflation: float,
    lanczos_min_eig_deflation: float,
    lanczos_min_eig_floor_fraction: float,
    seed: int,
    true_block_apply=None,
) -> TensorDiagonalBlockInverseFactors:
    """Fill ``richardson_omega`` from a Lanczos estimate when requested.

    The convention is ``richardson_omega <= 0`` opts in to auto-tuning. The
    backbone smoother (``split_backbone_inv_*``) must be present, otherwise the
    Richardson loop is inactive and the field is left at 1.0.

    When ``true_block_apply`` is provided, the Lanczos bounds are estimated
    against the true (extracted-mass restricted to this block) operator rather
    than the rank-r CP forward; this matches the operator used by the
    Richardson residual at apply time.
    """
    if factors.richardson_steps <= 0 or float(factors.richardson_omega) > 0.0:
        return factors
    if factors.split_backbone_inv_r is None:
        return eqx.tree_at(
            lambda f: f.richardson_omega,
            factors,
            jnp.asarray(1.0, dtype=jnp.float64),
        )
    if true_block_apply is None:
        forward_apply = lambda x, block_factors=factors: _apply_tensor_diagonal_block_forward(block_factors, x)
    else:
        forward_apply = true_block_apply
    omega = _estimate_richardson_omega_apply(
        forward_apply,
        lambda x, block_factors=factors: _apply_tensor_diagonal_block_backbone_preconditioner(block_factors, x),
        int(jnp.prod(jnp.asarray(factors.shape))),
        lanczos_iterations=lanczos_iterations,
        lanczos_max_eig_inflation=lanczos_max_eig_inflation,
        lanczos_min_eig_deflation=lanczos_min_eig_deflation,
        lanczos_min_eig_floor_fraction=lanczos_min_eig_floor_fraction,
        seed=seed,
    )
    return eqx.tree_at(
        lambda f: f.richardson_omega,
        factors,
        jnp.asarray(omega, dtype=jnp.float64),
    )


def _annotate_tensor_block_chebyshev_bounds(
    factors: TensorDiagonalBlockInverseFactors,
    *,
    lanczos_iterations: int,
    lanczos_max_eig_inflation: float,
    lanczos_min_eig_deflation: float,
    lanczos_min_eig_floor_fraction: float,
    seed: int,
    true_block_apply=None,
) -> TensorDiagonalBlockInverseFactors:
    if factors.chebyshev_steps <= 0:
        return factors

    # Bounds are estimated against the *operator* the Chebyshev iteration
    # will see at apply time. When ``true_block_apply`` is provided we
    # iterate against the assembled bulk mass; otherwise we fall back to
    # the rank-r CP surrogate forward.
    if true_block_apply is not None:
        operator_apply = true_block_apply
    else:
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
    *,
    true_block_apply=None,
) -> jnp.ndarray:
    del block_matrix
    if (
        factors.chebyshev_steps > 0
        and factors.chebyshev_lambda_min is not None
        and factors.chebyshev_lambda_max is not None
    ):
        # Chebyshev polish: iterate against the true bulk mass when it is
        # available, otherwise against the rank-r CP surrogate forward.
        # The smoother is always the tensor surrogate inverse, so this is
        # the natural way to absorb modelling error from a crude CP fit.
        forward = (
            true_block_apply
            if true_block_apply is not None
            else (lambda x, block_factors=factors: _apply_tensor_diagonal_block_forward(block_factors, x))
        )
        apply = _build_chebyshev_apply_preconditioner(
            forward,
            lambda x, block_factors=factors: _apply_tensor_diagonal_block_preconditioner(block_factors, x),
            steps=factors.chebyshev_steps,
            min_eig=factors.chebyshev_lambda_min,
            max_eig=factors.chebyshev_lambda_max,
        )
        return apply(rhs)
    return _apply_tensor_diagonal_block(factors, rhs, true_block_apply=true_block_apply)


def _extraction_operator(seq, k: int, dirichlet: bool):
    return getattr(seq, f"e{k}_dbc" if dirichlet else f"e{k}")


def _extraction_operator_transpose(seq, k: int, dirichlet: bool):
    return getattr(seq, f"e{k}_dbc_T" if dirichlet else f"e{k}_T")


def _extracted_size(seq, k: int, dirichlet: bool) -> int:
    return int(getattr(seq, f"n{k}_dbc" if dirichlet else f"n{k}"))


def _build_extracted_mass_apply_data(seq, mass_apply, k: int, dirichlet: bool) -> ExtractedMassApplyData:
    return ExtractedMassApplyData(
        mass_apply=mass_apply,
        extraction=_extraction_operator(seq, k, dirichlet),
        extraction_t=_extraction_operator_transpose(seq, k, dirichlet),
        size=_extracted_size(seq, k, dirichlet),
    )


def _restrict_sparse_rows(matrix, row_indices: jnp.ndarray):
    return matrix.restrict_rows(row_indices)


def _restrict_sparse_cols(matrix, col_indices: jnp.ndarray):
    return matrix.restrict_cols(col_indices)


def _build_restricted_extracted_mass_apply_data(
    data: ExtractedMassApplyData,
    row_indices: jnp.ndarray,
    col_indices: jnp.ndarray,
) -> RestrictedExtractedMassApplyData:
    row_indices = jnp.asarray(row_indices, dtype=jnp.int32)
    col_indices = jnp.asarray(col_indices, dtype=jnp.int32)
    return RestrictedExtractedMassApplyData(
        mass_apply=data.mass_apply,
        row_extraction=_restrict_sparse_rows(data.extraction, row_indices),
        col_extraction_t=_restrict_sparse_cols(data.extraction_t, col_indices),
        output_size=int(row_indices.shape[0]),
        input_size=int(col_indices.shape[0]),
    )


def _apply_extracted_mass_operator(extraction, extraction_t, mass_apply, x: jnp.ndarray) -> jnp.ndarray:
    raw = extraction_t @ x
    return jnp.asarray(extraction @ mass_apply(raw))


def _apply_extracted_mass_operator_data(data: ExtractedMassApplyData, x: jnp.ndarray) -> jnp.ndarray:
    return _apply_extracted_mass_operator(data.extraction, data.extraction_t, data.mass_apply, x)


def _apply_restricted_extracted_mass_operator_data(data: RestrictedExtractedMassApplyData, x: jnp.ndarray) -> jnp.ndarray:
    raw = data.col_extraction_t @ x
    return jnp.asarray(data.row_extraction @ data.mass_apply(raw))


def _apply_extracted_submatrix(data: ExtractedMassApplyData, row_indices: jnp.ndarray, col_indices: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    full = jnp.zeros((data.size,), dtype=x.dtype)
    full = full.at[col_indices].set(x)
    return _apply_extracted_mass_operator_data(data, full)[row_indices]


def _symmetric_pseudoinverse(matrix: jnp.ndarray, *, relative_tol: float = 1e-8) -> jnp.ndarray:
    matrix = _symmetrize(matrix)
    eigvals, eigvecs = jnp.linalg.eigh(matrix)
    scale = jnp.max(jnp.abs(eigvals))
    safe_scale = jnp.where(scale > 0, scale, 1.0)
    cutoff = relative_tol * safe_scale
    inv_eigvals = jnp.where(jnp.abs(eigvals) > cutoff, 1.0 / eigvals, 0.0)
    return _symmetrize((eigvecs * inv_eigvals[jnp.newaxis, :]) @ eigvecs.T)


def _assemble_surgery_schur_inverse_from_applies(
    ass: jnp.ndarray,
    surgery_to_bulk_apply,
    bulk_apply,
    bulk_to_surgery_apply,
    *,
    relative_tol: float = 1e-8,
    sequential: bool = False,
) -> jnp.ndarray:
    basis = jnp.eye(ass.shape[0], dtype=ass.dtype)

    def schur_apply(rhs_s: jnp.ndarray) -> jnp.ndarray:
        bulk_rhs = surgery_to_bulk_apply(rhs_s)
        bulk_response = bulk_apply(bulk_rhs)
        return ass @ rhs_s - bulk_to_surgery_apply(bulk_response)

    if sequential:
        # The coupling applies may be matrix free; probe columns one at a time
        # via ``jax.lax.map`` so the dense element transient is not batched.
        surgery_schur = jax.lax.map(schur_apply, basis).T
    else:
        surgery_schur = jax.vmap(schur_apply, in_axes=1, out_axes=1)(basis)
    return _symmetric_pseudoinverse(surgery_schur, relative_tol=relative_tol)


def _apply_surgery_to_bulk_coupling(surgery, rhs_s: jnp.ndarray) -> jnp.ndarray:
    """Apply the surgery->bulk coupling block M[bulk, surgery] @ rhs_s.

    Generic across k=0/1/2 mass and k=1/2 stiffness surgery factors (all expose
    ``coupling_sb`` / ``surgery_to_bulk_data`` / ``apply_data`` /
    ``surgery_indices`` / ``bulk_indices``). Prefers the precomputed dense block,
    then the restricted-sparse apply, then a full extracted-submatrix probe.
    """
    if surgery.coupling_sb is not None:
        return surgery.coupling_sb @ rhs_s
    if surgery.surgery_to_bulk_data is not None:
        return _apply_restricted_extracted_mass_operator_data(surgery.surgery_to_bulk_data, rhs_s)
    return _apply_extracted_submatrix(surgery.apply_data, surgery.bulk_indices, surgery.surgery_indices, rhs_s)


def _apply_bulk_to_surgery_coupling(surgery, rhs_b: jnp.ndarray) -> jnp.ndarray:
    """Apply the bulk->surgery coupling block M[surgery, bulk] @ rhs_b.

    The extracted operator is symmetric, so this is exactly ``coupling_sb.T``
    when the dense block is present. Generic across the same factor types as
    :func:`_apply_surgery_to_bulk_coupling`.
    """
    if surgery.coupling_sb is not None:
        return surgery.coupling_sb.T @ rhs_b
    if surgery.bulk_to_surgery_data is not None:
        return _apply_restricted_extracted_mass_operator_data(surgery.bulk_to_surgery_data, rhs_b)
    return _apply_extracted_submatrix(surgery.apply_data, surgery.surgery_indices, surgery.bulk_indices, rhs_b)


def _apply_surgery_schur(surgery, schur_inv: jnp.ndarray, bulk_inv, rhs: jnp.ndarray) -> jnp.ndarray:
    """Generic surgery/bulk block-factorization apply, shared by k=0/1/2 mass.

    ``bulk_inv`` is the (k-specific) bulk inverse callable. The surgery space is
    small; the bulk space is tensor-product. Computes the exact block inverse
    ``y = bulk_inv(rhs_b); z = Sigma^{-1}(rhs_s - M_sb y); x_b = y - bulk_inv(M_bs z)``.
    """
    rhs_s = rhs[surgery.surgery_indices]
    rhs_b = rhs[surgery.bulk_indices]
    y = bulk_inv(rhs_b)
    z = schur_inv @ (rhs_s - _apply_bulk_to_surgery_coupling(surgery, y))
    x_b = y - bulk_inv(_apply_surgery_to_bulk_coupling(surgery, z))
    x = jnp.zeros_like(rhs)
    x = x.at[surgery.surgery_indices].set(z)
    x = x.at[surgery.bulk_indices].set(x_b)
    return x


def _apply_surgery_schur_forward(surgery, bulk_fwd, rhs: jnp.ndarray) -> jnp.ndarray:
    """Generic surgery/bulk forward-model apply (the operator, not its inverse)."""
    rhs_s = rhs[surgery.surgery_indices]
    rhs_b = rhs[surgery.bulk_indices]
    out_s = surgery.ass @ rhs_s + _apply_bulk_to_surgery_coupling(surgery, rhs_b)
    out_b = _apply_surgery_to_bulk_coupling(surgery, rhs_s) + bulk_fwd(rhs_b)
    out = jnp.zeros_like(rhs)
    out = out.at[surgery.surgery_indices].set(out_s)
    out = out.at[surgery.bulk_indices].set(out_b)
    return out


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
    arr_true = lambda x: _apply_extracted_submatrix(surgery.apply_data, surgery.r_indices, surgery.r_indices, x)
    theta_true = lambda x: _apply_extracted_submatrix(surgery.apply_data, surgery.theta_bulk_indices, surgery.theta_bulk_indices, x)
    y = _apply_tensor_exact_block(None, arr_factors, rhs_r, true_block_apply=arr_true)
    z = _apply_tensor_exact_block(None, theta_factors, rhs_theta - _apply_k1_rt_atr_coupling(surgery, y), true_block_apply=theta_true)
    x_r = y - _apply_tensor_exact_block(None, arr_factors, _apply_k1_rt_art_coupling(surgery, z), true_block_apply=arr_true)
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
    zeta_true = lambda x: _apply_extracted_submatrix(surgery.apply_data, surgery.zeta_bulk_indices, surgery.zeta_bulk_indices, x)
    y_rt = _apply_k1_rt_preconditioner(surgery, arr_factors, theta_factors, rhs_rt)
    z = _apply_tensor_exact_block(
        None,
        zeta_factors,
        rhs_zeta - _apply_k1_rt_to_zeta_coupling(surgery, y_rt),
        true_block_apply=zeta_true,
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
    arr_true = lambda x: _apply_extracted_submatrix(surgery.apply_data, surgery.r_indices, surgery.r_indices, x)
    theta_true = lambda x: _apply_extracted_submatrix(surgery.apply_data, surgery.theta_bulk_indices, surgery.theta_bulk_indices, x)
    zeta_true = lambda x: _apply_extracted_submatrix(surgery.apply_data, surgery.zeta_bulk_indices, surgery.zeta_bulk_indices, x)
    return jnp.concatenate([
        _apply_tensor_exact_block(None, arr_factors, rhs_r, true_block_apply=arr_true),
        _apply_tensor_exact_block(None, theta_factors, rhs_theta, true_block_apply=theta_true),
        _apply_tensor_exact_block(None, zeta_factors, rhs_zeta, true_block_apply=zeta_true),
    ])


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
    r_true = lambda x: _apply_extracted_submatrix(surgery.apply_data, surgery.r_bulk_indices, surgery.r_bulk_indices, x)
    theta_true = lambda x: _apply_extracted_submatrix(surgery.apply_data, surgery.theta_indices, surgery.theta_indices, x)
    y = _apply_tensor_exact_block(None, r_bulk_factors, rhs_r, true_block_apply=r_true)
    z = _apply_tensor_exact_block(None, theta_factors, rhs_theta - _apply_k2_r_to_theta_coupling(surgery, y), true_block_apply=theta_true)
    x_r = y - _apply_tensor_exact_block(None, r_bulk_factors, _apply_k2_theta_to_r_coupling(surgery, z), true_block_apply=r_true)
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
    zeta_true = lambda x: _apply_extracted_submatrix(surgery.apply_data, surgery.zeta_indices, surgery.zeta_indices, x)
    y_rt = _apply_k2_rt_preconditioner(surgery, r_bulk_factors, theta_factors, rhs_rt)
    z = _apply_tensor_exact_block(
        None,
        zeta_factors,
        rhs_zeta - _apply_k2_rt_to_zeta_coupling(surgery, y_rt),
        true_block_apply=zeta_true,
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
    r_true = lambda x: _apply_extracted_submatrix(surgery.apply_data, surgery.r_bulk_indices, surgery.r_bulk_indices, x)
    theta_true = lambda x: _apply_extracted_submatrix(surgery.apply_data, surgery.theta_indices, surgery.theta_indices, x)
    zeta_true = lambda x: _apply_extracted_submatrix(surgery.apply_data, surgery.zeta_indices, surgery.zeta_indices, x)
    return jnp.concatenate([
        _apply_tensor_exact_block(None, r_bulk_factors, rhs_r, true_block_apply=r_true),
        _apply_tensor_exact_block(None, theta_factors, rhs_theta, true_block_apply=theta_true),
        _apply_tensor_exact_block(None, zeta_factors, rhs_zeta, true_block_apply=zeta_true),
    ])


def _extract_selected_columns(
    seq, mass_apply, k: int, dirichlet: bool, column_indices: jnp.ndarray,
    *, sequential: bool = False,
) -> jnp.ndarray:
    extraction = _extraction_operator(seq, k, dirichlet)
    extraction_t = _extraction_operator_transpose(seq, k, dirichlet)
    size = _extracted_size(seq, k, dirichlet)
    basis = jax.nn.one_hot(jnp.asarray(column_indices), size, dtype=jnp.float64).T
    apply_col = lambda col: _apply_extracted_mass_operator(extraction, extraction_t, mass_apply, col)
    if sequential:
        # ``mass_apply`` may be a matrix-free element operator whose per-call
        # transient is a dense O(ne*q^3) tensor. ``jax.vmap`` would batch that
        # transient by the number of probed columns and blow up memory, so we
        # probe one column at a time with ``jax.lax.map`` instead.
        cols = jax.lax.map(apply_col, basis.T)
        return cols.T
    return jax.vmap(apply_col, in_axes=1, out_axes=1)(basis)


def build_mass_surgery_preconditioner(
    seq,
    mass_apply,
    *,
    k: int,
    existing: Optional[MassSurgeryPreconditioner] = None,
    dirichlet_flags: tuple[bool, ...] = (False, True),
    precompute_coupling: bool = True,
) -> MassSurgeryPreconditioner:
    surgery_precond = existing if existing is not None else MassSurgeryPreconditioner()

    if k == 3:
        return surgery_precond

    pair = BoundaryConditionPair()
    if k == 0:
        surgery_size = _core_size(seq)
        for dirichlet in dirichlet_flags:
            surgery_indices = jnp.arange(surgery_size)
            surgery_cols = _extract_selected_columns(seq, mass_apply, 0, dirichlet, surgery_indices, sequential=True)
            ass = _symmetrize(surgery_cols[surgery_indices, :])
            apply_data = _build_extracted_mass_apply_data(seq, mass_apply, 0, dirichlet)
            bulk_indices = jnp.arange(surgery_size, apply_data.size)
            # The dense surgery->bulk coupling block (bulk x surgery) is already
            # contained in ``surgery_cols`` (the extracted-mass columns probed
            # for ``ass``), so the precompute is free here: it is exactly the
            # bulk rows of those columns. M_0 is symmetric => bulk->surgery is
            # its transpose. See ``coupling_sb`` on the factors class.
            coupling_sb = surgery_cols[bulk_indices, :] if precompute_coupling else None
            factors = K0MassSurgeryPreconditionerFactors(
                surgery_size=surgery_size,
                apply_data=apply_data,
                surgery_indices=surgery_indices,
                bulk_indices=bulk_indices,
                surgery_to_bulk_data=_build_restricted_extracted_mass_apply_data(apply_data, bulk_indices, surgery_indices),
                bulk_to_surgery_data=_build_restricted_extracted_mass_apply_data(apply_data, surgery_indices, bulk_indices),
                surgery_diaginv=1.0 / jnp.diag(ass),
                ass=ass,
                coupling_sb=coupling_sb,
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
            surgery_cols = _extract_selected_columns(seq, mass_apply, 1, dirichlet, surgery_indices, sequential=True)
            ass = _symmetrize(surgery_cols[surgery_indices, :])
            rt_r_size = int(block_indices["rt_r_size"])
            rt_theta_size = int(block_indices["rt_theta_size"])
            apply_data = _build_extracted_mass_apply_data(seq, mass_apply, 1, dirichlet)
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
                # Dense surgery->bulk block, free from the ``surgery_cols`` probe
                # done for ``ass``. Only the surgery<->bulk coupling; the inner
                # rt/zeta couplings stay matrix-free (bulk-scale).
                coupling_sb=(surgery_cols[bulk_indices, :] if precompute_coupling else None),
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
            apply_data = _build_extracted_mass_apply_data(seq, mass_apply, 2, dirichlet)
            surgery_cols = _extract_selected_columns(seq, mass_apply, 2, dirichlet, surgery_indices, sequential=True)
            ass = _symmetrize(surgery_cols[surgery_indices, :])
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
                # Dense surgery->bulk block, free from the ``surgery_cols`` probe
                # done for ``ass`` (surgery<->bulk only; inner couplings stay
                # matrix-free).
                coupling_sb=(surgery_cols[block_indices["bulk"], :] if precompute_coupling else None),
            )
            pair = eqx.tree_at(
                lambda boundary_pair: boundary_pair.dbc if dirichlet else boundary_pair.free,
                pair,
                factors,
                is_leaf=lambda x: x is None,
            )
        return eqx.tree_at(lambda data: data.k2, surgery_precond, pair)

    raise ValueError("Mass surgery preconditioner currently only supports k=0, k=1, k=2 and k=3")


def _build_greville_mass_block_factors(
    seq, *, shape, diff, wkind: str, comp: int,
) -> TensorDiagonalBlockInverseFactors:
    """Greville-collocation mass bulk block factors.

    P^{-1} = D^{-1/2} (M0_r^{-1} x M0_t^{-1} x M0_z^{-1}) D^{-1/2}, with UNWEIGHTED
    1D masses (degree p on primal axes, p-1 on the differentiated axis) and D the
    metric weight collocated at the component's Greville abscissae. Ports
    scripts/debug/greville_bulk_precond.py:build_greville_component.

    ``diff`` = (r,t,z) booleans (True => differentiated degree-(p-1) axis);
    ``wkind`` in {'J','invJ','Jginv','ginvJ'}; ``comp`` = metric diagonal index.
    """
    from mrx.geometry import compute_geometry_terms  # noqa: PLC0415
    from mrx.spline_bases import SplineBasis  # noqa: PLC0415

    nr, ntc, nzc = (int(s) for s in shape)
    radial_start = 1 if diff[0] else 2

    primal = (seq.basis_r_jk, seq.basis_t_jk, seq.basis_z_jk)
    deriv = (seq.d_basis_r_jk, seq.d_basis_t_jk, seq.d_basis_z_jk)
    bases = tuple(deriv[a] if diff[a] else primal[a] for a in range(3))
    quad_w = (seq.quad.w_x, seq.quad.w_y, seq.quad.w_z)
    M0_r = _restrict_radial_mass(_assemble_weighted_1d_mass(bases[0], quad_w[0]), radial_start, nr)
    M0_t = _assemble_weighted_1d_mass(bases[1], quad_w[1])
    M0_z = _assemble_weighted_1d_mass(bases[2], quad_w[2])
    inv_r = jnp.linalg.inv(M0_r)
    inv_t = jnp.linalg.inv(M0_t)
    inv_z = jnp.linalg.inv(M0_z)

    # Greville abscissae per axis: primal degree-p, or fresh degree-(p-1) SplineBasis
    # on the differentiated axis (dΛ[axis].s inherits parent knots -> spurious double
    # boundary point). Clamped endpoints nudged inward (a spline map's clamped
    # evaluate() has a constant branch -> jacfwd det=0 at the exact endpoint).
    types = seq.basis_0.types
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
    metric, minv, jac = compute_geometry_terms(seq.map, pts)
    if wkind == "J":
        weight = jac
    elif wkind == "invJ":
        weight = 1.0 / jac
    elif wkind == "Jginv":          # k=1: J g^{ii}
        weight = jac * minv[:, comp, comp]
    elif wkind == "ginvJ":          # k=2: g_{ii} / J
        weight = metric[:, comp, comp] / jac
    else:
        raise ValueError(f"unknown greville mass wkind {wkind!r}")
    D = jnp.asarray(weight).reshape(nr, ntc, nzc)
    # D MUST be positive (SPD); degenerate collocation points (clamped Greville at a
    # geometry fold) -> positive median, NOT a tiny floor (which would spike
    # 1/sqrt(D) into a spurious near-null mode); that region is surgery-corrected.
    valid = jnp.isfinite(D) & (D > 0)
    fin = D[valid]
    scale = jnp.median(fin) if fin.size > 0 else jnp.asarray(1.0, dtype=jnp.float64)
    D = jnp.where(valid, D, scale)
    inv_sqrt_D = 1.0 / jnp.sqrt(D)

    return TensorDiagonalBlockInverseFactors(
        shape=(nr, ntc, nzc),
        greville_inv_r=inv_r,
        greville_inv_t=inv_t,
        greville_inv_z=inv_z,
        greville_inv_sqrt_D=inv_sqrt_D,
    )


def build_mass_tensor_preconditioner(
    seq,
    *,
    k: int,
    rank: int = 1,
    fallback_rank: Optional[int] = None,
    cp_kwargs: Optional[Mapping[str, object]] = None,
    existing: Optional[TensorMassPreconditioner] = None,
    surgery_precond: Optional[MassSurgeryPreconditioner] = None,
    dirichlet_flags: tuple[bool, ...] = (False, True),
    k3_true_block_apply: Optional[Mapping[bool, object]] = None,
) -> TensorMassPreconditioner:
    fallback_rank = rank if fallback_rank is None else int(fallback_rank)
    cp_kwargs = {} if cp_kwargs is None else dict(cp_kwargs)
    cp_maxiter = int(cp_kwargs.get("maxiter", 100))
    cp_tol = float(cp_kwargs.get("tol", 1e-9))
    cp_ridge = float(cp_kwargs.get("ridge", 1e-12))
    # Default 0 (no block-Chebyshev polish): the polish cuts mass-precond
    # iterations but costs ~8-11x more wall (each step is a full matrix-free mass
    # matvec x3 components x2 bulk_inv calls), so it is a large net wall LOSS on
    # both toroid and W7-X (see outputs/mass_bcheb/sweep/). Matches the validated
    # 2026-05-09 production config (bcheb=0). Opt in via cp_kwargs if ever wanted.
    block_chebyshev_steps = int(cp_kwargs.get("block_chebyshev_steps", 0))
    block_lanczos_iterations = int(cp_kwargs.get("block_lanczos_iterations", 16))
    block_lanczos_max_eig_inflation = float(cp_kwargs.get("block_lanczos_max_eig_inflation", 1.1))
    block_lanczos_min_eig_deflation = float(cp_kwargs.get("block_lanczos_min_eig_deflation", 0.85))
    block_lanczos_min_eig_floor_fraction = float(cp_kwargs.get("block_lanczos_min_eig_floor_fraction", 1e-3))
    richardson_steps = int(cp_kwargs.get("richardson_steps", 0))
    richardson_omega = float(cp_kwargs.get("richardson_omega", 1.0))
    surgery_schur_pinv_tol = float(
        cp_kwargs.get("surgery_schur_pinv_tol", cp_kwargs.get("schur_pinv_tol", 1e-8))
    )
    bulk_schur = bool(cp_kwargs.get("bulk_schur", False))
    # Greville collocation: replace the per-component CP-fit bulk factors with the
    # unweighted-atom + pointwise-D sandwich (built by _build_greville_mass_block_factors).
    # The surgery/Schur envelope and the apply path are unchanged. Greville is now the
    # ONLY mass bulk path (the CP `else` branches below are unreachable dead code,
    # retained pending a cosmetic cleanup; the shared CP core stays for the stiffness).
    greville = True

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
        and existing.surgery_schur_pinv_tol == surgery_schur_pinv_tol
    )
    new_ranks = tuple(
        rank if k == kk
        else (existing.ranks[kk] if reuse_existing else fallback_rank)
        for kk in range(4)
    )
    tensor_precond = TensorMassPreconditioner(
        ranks=new_ranks,
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
        surgery_schur_pinv_tol=surgery_schur_pinv_tol,
        k0=existing.k0 if reuse_existing else BoundaryConditionPair(),
        k1=existing.k1 if reuse_existing else BoundaryConditionPair(),
        k2=existing.k2 if reuse_existing else BoundaryConditionPair(),
        k3=existing.k3 if reuse_existing else BoundaryConditionPair(),
    )

    pair = BoundaryConditionPair()
    if k == 0:
        weight_tensor = _k0_bulk_weight_tensor(seq)
        if surgery_precond is None:
            raise ValueError("Tensor mass k=0 requires surgery factors to be assembled first")
        for dirichlet in dirichlet_flags:
            surgery = select_boundary_data(surgery_precond.k0, dirichlet, "Mass surgery k=0")
            bulk_shape = _bulk_tensor_shape(seq, dirichlet)
            bulk_indices_k0 = jnp.arange(surgery.surgery_size, surgery.apply_data.size, dtype=jnp.int32)
            bulk_true_apply = lambda x, surgery=surgery, bulk_indices_k0=bulk_indices_k0: _apply_extracted_submatrix(surgery.apply_data, bulk_indices_k0, bulk_indices_k0, x)
            if greville:
                bulk_factors = _build_greville_mass_block_factors(
                    seq, shape=bulk_shape, diff=(False, False, False), wkind="J", comp=0)
            else:
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
                    radial_baseline=None,
                    prior_terms=None,
                    chebyshev_steps=block_chebyshev_steps,
                    chebyshev_lanczos_iterations=block_lanczos_iterations,
                    chebyshev_lanczos_max_eig_inflation=block_lanczos_max_eig_inflation,
                    chebyshev_lanczos_min_eig_deflation=block_lanczos_min_eig_deflation,
                    chebyshev_lanczos_min_eig_floor_fraction=block_lanczos_min_eig_floor_fraction,
                    chebyshev_seed=100 + int(dirichlet),
                    richardson_steps=richardson_steps,
                    richardson_omega=richardson_omega,
                    true_block_apply=bulk_true_apply,
                )
                bulk_factors = _annotate_tensor_block_chebyshev_bounds(
                    bulk_factors,
                    lanczos_iterations=block_lanczos_iterations,
                    lanczos_max_eig_inflation=block_lanczos_max_eig_inflation,
                    lanczos_min_eig_deflation=block_lanczos_min_eig_deflation,
                    lanczos_min_eig_floor_fraction=block_lanczos_min_eig_floor_fraction,
                    seed=100 + int(dirichlet),
                    true_block_apply=bulk_true_apply,
                )
            bulk_indices_k0 = jnp.arange(surgery.surgery_size, surgery.apply_data.size, dtype=jnp.int32)
            bulk_true_k0 = lambda x: _apply_extracted_submatrix(surgery.apply_data, bulk_indices_k0, bulk_indices_k0, x)
            schur_inv = _assemble_surgery_schur_inverse_from_applies(
                surgery.ass,
                lambda rhs_s, surgery=surgery: _apply_surgery_to_bulk_coupling(surgery, rhs_s),
                lambda rhs_b, bulk_factors=bulk_factors, bulk_true_k0=bulk_true_k0: _apply_tensor_exact_block(None, bulk_factors, rhs_b, true_block_apply=bulk_true_k0),
                lambda rhs_b, surgery=surgery: _apply_bulk_to_surgery_coupling(surgery, rhs_b),
                relative_tol=tensor_precond.surgery_schur_pinv_tol,
                sequential=True,
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

            arr_true_apply = lambda x, surgery=surgery, idx=r_indices: _apply_extracted_submatrix(surgery.apply_data, idx, idx, x)
            theta_true_apply = lambda x, surgery=surgery, idx=theta_bulk_indices: _apply_extracted_submatrix(surgery.apply_data, idx, idx, x)
            zeta_true_apply = lambda x, surgery=surgery, idx=zeta_bulk_indices: _apply_extracted_submatrix(surgery.apply_data, idx, idx, x)

            if greville:
                arr_factors = _build_greville_mass_block_factors(
                    seq, shape=arr_shape, diff=(True, False, False), wkind="Jginv", comp=0)
            else:
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
                    radial_baseline=None,
                    prior_terms=None,
                    chebyshev_steps=block_chebyshev_steps,
                    chebyshev_lanczos_iterations=block_lanczos_iterations,
                    chebyshev_lanczos_max_eig_inflation=block_lanczos_max_eig_inflation,
                    chebyshev_lanczos_min_eig_deflation=block_lanczos_min_eig_deflation,
                    chebyshev_lanczos_min_eig_floor_fraction=block_lanczos_min_eig_floor_fraction,
                    chebyshev_seed=200 + 10 * int(dirichlet),
                    richardson_steps=richardson_steps,
                    richardson_omega=richardson_omega,
                    true_block_apply=arr_true_apply,
                )
                arr_factors = _annotate_tensor_block_chebyshev_bounds(
                    arr_factors,
                    lanczos_iterations=block_lanczos_iterations,
                    lanczos_max_eig_inflation=block_lanczos_max_eig_inflation,
                    lanczos_min_eig_deflation=block_lanczos_min_eig_deflation,
                    lanczos_min_eig_floor_fraction=block_lanczos_min_eig_floor_fraction,
                    seed=200 + 10 * int(dirichlet),
                    true_block_apply=arr_true_apply,
                )
            if greville:
                theta_factors = _build_greville_mass_block_factors(
                    seq, shape=theta_shape, diff=(False, True, False), wkind="Jginv", comp=1)
            else:
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
                    radial_baseline=None,
                    prior_terms=None,
                    chebyshev_steps=block_chebyshev_steps,
                    chebyshev_lanczos_iterations=block_lanczos_iterations,
                    chebyshev_lanczos_max_eig_inflation=block_lanczos_max_eig_inflation,
                    chebyshev_lanczos_min_eig_deflation=block_lanczos_min_eig_deflation,
                    chebyshev_lanczos_min_eig_floor_fraction=block_lanczos_min_eig_floor_fraction,
                    chebyshev_seed=201 + 10 * int(dirichlet),
                    richardson_steps=richardson_steps,
                    richardson_omega=richardson_omega,
                    true_block_apply=theta_true_apply,
                )
                theta_factors = _annotate_tensor_block_chebyshev_bounds(
                    theta_factors,
                    lanczos_iterations=block_lanczos_iterations,
                    lanczos_max_eig_inflation=block_lanczos_max_eig_inflation,
                    lanczos_min_eig_deflation=block_lanczos_min_eig_deflation,
                    lanczos_min_eig_floor_fraction=block_lanczos_min_eig_floor_fraction,
                    seed=201 + 10 * int(dirichlet),
                    true_block_apply=theta_true_apply,
                )
            if greville:
                zeta_factors = _build_greville_mass_block_factors(
                    seq, shape=zeta_shape, diff=(False, False, True), wkind="Jginv", comp=2)
            else:
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
                    radial_baseline=None,
                    prior_terms=None,
                    chebyshev_steps=block_chebyshev_steps,
                    chebyshev_lanczos_iterations=block_lanczos_iterations,
                    chebyshev_lanczos_max_eig_inflation=block_lanczos_max_eig_inflation,
                    chebyshev_lanczos_min_eig_deflation=block_lanczos_min_eig_deflation,
                    chebyshev_lanczos_min_eig_floor_fraction=block_lanczos_min_eig_floor_fraction,
                    chebyshev_seed=202 + 10 * int(dirichlet),
                    richardson_steps=richardson_steps,
                    richardson_omega=richardson_omega,
                    true_block_apply=zeta_true_apply,
                )
                zeta_factors = _annotate_tensor_block_chebyshev_bounds(
                    zeta_factors,
                    lanczos_iterations=block_lanczos_iterations,
                    lanczos_max_eig_inflation=block_lanczos_max_eig_inflation,
                    lanczos_min_eig_deflation=block_lanczos_min_eig_deflation,
                    lanczos_min_eig_floor_fraction=block_lanczos_min_eig_floor_fraction,
                    seed=202 + 10 * int(dirichlet),
                    true_block_apply=zeta_true_apply,
                )
            schur_inv = _assemble_surgery_schur_inverse_from_applies(
                surgery.ass,
                lambda rhs_s, surgery=surgery: _apply_surgery_to_bulk_coupling(surgery, rhs_s),
                lambda rhs_bulk, surgery=surgery, arr_factors=arr_factors, theta_factors=theta_factors, zeta_factors=zeta_factors, bulk_schur=bulk_schur: (
                    _apply_k1_bulk_preconditioner(
                        surgery,
                        arr_factors,
                        theta_factors,
                        zeta_factors,
                        rhs_bulk,
                    ) if bulk_schur else _apply_k1_bulk_diagonal_preconditioner(
                        surgery,
                        arr_factors,
                        theta_factors,
                        zeta_factors,
                        rhs_bulk,
                    )
                ),
                lambda rhs_bulk, surgery=surgery: _apply_bulk_to_surgery_coupling(surgery, rhs_bulk),
                relative_tol=tensor_precond.surgery_schur_pinv_tol,
                sequential=True,
            )

            factors = K1TensorMassPreconditionerFactors(
                r_indices=r_indices,
                theta_bulk_indices=theta_bulk_indices,
                zeta_bulk_indices=zeta_bulk_indices,
                rt_r_size=rt_r_size,
                rt_theta_size=rt_theta_size,
                bulk_schur=bulk_schur,
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

            r_bulk_true_apply = lambda x, surgery=surgery, idx=r_bulk_indices: _apply_extracted_submatrix(surgery.apply_data, idx, idx, x)
            theta_true_apply = lambda x, surgery=surgery, idx=theta_indices: _apply_extracted_submatrix(surgery.apply_data, idx, idx, x)
            zeta_true_apply = lambda x, surgery=surgery, idx=zeta_indices: _apply_extracted_submatrix(surgery.apply_data, idx, idx, x)

            if greville:
                r_bulk_factors = _build_greville_mass_block_factors(
                    seq, shape=_r_bulk_shape_k2(seq, dirichlet), diff=(False, True, True), wkind="ginvJ", comp=0)
            else:
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
                    radial_baseline=None,
                    prior_terms=None,
                    chebyshev_steps=block_chebyshev_steps,
                    chebyshev_lanczos_iterations=block_lanczos_iterations,
                    chebyshev_lanczos_max_eig_inflation=block_lanczos_max_eig_inflation,
                    chebyshev_lanczos_min_eig_deflation=block_lanczos_min_eig_deflation,
                    chebyshev_lanczos_min_eig_floor_fraction=block_lanczos_min_eig_floor_fraction,
                    chebyshev_seed=300 + 10 * int(dirichlet),
                    richardson_steps=richardson_steps,
                    richardson_omega=richardson_omega,
                    true_block_apply=r_bulk_true_apply,
                )
                r_bulk_factors = _annotate_tensor_block_chebyshev_bounds(
                    r_bulk_factors,
                    lanczos_iterations=block_lanczos_iterations,
                    lanczos_max_eig_inflation=block_lanczos_max_eig_inflation,
                    lanczos_min_eig_deflation=block_lanczos_min_eig_deflation,
                    lanczos_min_eig_floor_fraction=block_lanczos_min_eig_floor_fraction,
                    seed=300 + 10 * int(dirichlet),
                    true_block_apply=r_bulk_true_apply,
                )
            if greville:
                theta_factors = _build_greville_mass_block_factors(
                    seq, shape=_theta_shape_k2(seq, dirichlet), diff=(True, False, True), wkind="ginvJ", comp=1)
            else:
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
                    radial_baseline=None,
                    prior_terms=None,
                    chebyshev_steps=block_chebyshev_steps,
                    chebyshev_lanczos_iterations=block_lanczos_iterations,
                    chebyshev_lanczos_max_eig_inflation=block_lanczos_max_eig_inflation,
                    chebyshev_lanczos_min_eig_deflation=block_lanczos_min_eig_deflation,
                    chebyshev_lanczos_min_eig_floor_fraction=block_lanczos_min_eig_floor_fraction,
                    chebyshev_seed=301 + 10 * int(dirichlet),
                    richardson_steps=richardson_steps,
                    richardson_omega=richardson_omega,
                    true_block_apply=theta_true_apply,
                )
                theta_factors = _annotate_tensor_block_chebyshev_bounds(
                    theta_factors,
                    lanczos_iterations=block_lanczos_iterations,
                    lanczos_max_eig_inflation=block_lanczos_max_eig_inflation,
                    lanczos_min_eig_deflation=block_lanczos_min_eig_deflation,
                    lanczos_min_eig_floor_fraction=block_lanczos_min_eig_floor_fraction,
                    seed=301 + 10 * int(dirichlet),
                    true_block_apply=theta_true_apply,
                )
            if greville:
                zeta_factors = _build_greville_mass_block_factors(
                    seq, shape=_zeta_shape_k2(seq, dirichlet), diff=(True, True, False), wkind="ginvJ", comp=2)
            else:
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
                    radial_baseline=None,
                    prior_terms=None,
                    chebyshev_steps=block_chebyshev_steps,
                    chebyshev_lanczos_iterations=block_lanczos_iterations,
                    chebyshev_lanczos_max_eig_inflation=block_lanczos_max_eig_inflation,
                    chebyshev_lanczos_min_eig_deflation=block_lanczos_min_eig_deflation,
                    chebyshev_lanczos_min_eig_floor_fraction=block_lanczos_min_eig_floor_fraction,
                    chebyshev_seed=302 + 10 * int(dirichlet),
                    richardson_steps=richardson_steps,
                    richardson_omega=richardson_omega,
                    true_block_apply=zeta_true_apply,
                )
                zeta_factors = _annotate_tensor_block_chebyshev_bounds(
                    zeta_factors,
                    lanczos_iterations=block_lanczos_iterations,
                    lanczos_max_eig_inflation=block_lanczos_max_eig_inflation,
                    lanczos_min_eig_deflation=block_lanczos_min_eig_deflation,
                    lanczos_min_eig_floor_fraction=block_lanczos_min_eig_floor_fraction,
                    seed=302 + 10 * int(dirichlet),
                    true_block_apply=zeta_true_apply,
                )
            schur_inv = _assemble_surgery_schur_inverse_from_applies(
                surgery.ass,
                lambda rhs_s, surgery=surgery: _apply_surgery_to_bulk_coupling(surgery, rhs_s),
                lambda rhs_bulk, surgery=surgery, r_bulk_factors=r_bulk_factors, theta_factors=theta_factors, zeta_factors=zeta_factors, bulk_schur=bulk_schur: (
                    _apply_k2_bulk_preconditioner(
                        surgery,
                        r_bulk_factors,
                        theta_factors,
                        zeta_factors,
                        rhs_bulk,
                    ) if bulk_schur else _apply_k2_bulk_diagonal_preconditioner(
                        surgery,
                        r_bulk_factors,
                        theta_factors,
                        zeta_factors,
                        rhs_bulk,
                    )
                ),
                lambda rhs_bulk, surgery=surgery: _apply_bulk_to_surgery_coupling(surgery, rhs_bulk),
                relative_tol=tensor_precond.surgery_schur_pinv_tol,
                sequential=True,
            )
            factors = K2TensorMassPreconditionerFactors(
                r_bulk_indices=r_bulk_indices,
                theta_indices=theta_indices,
                zeta_indices=zeta_indices,
                r_bulk_size=r_bulk_size,
                theta_size=theta_size,
                zeta_size=zeta_size,
                bulk_schur=bulk_schur,
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
        for dirichlet in dirichlet_flags:
            true_apply = (
                k3_true_block_apply.get(dirichlet)
                if k3_true_block_apply is not None
                else None
            )
            if greville:
                factors = _build_greville_mass_block_factors(
                    seq, shape=extracted_shape, diff=(True, True, True), wkind="invJ", comp=0)
            else:
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
                    radial_baseline=None,
                    prior_terms=None,
                    chebyshev_steps=block_chebyshev_steps,
                    chebyshev_lanczos_iterations=block_lanczos_iterations,
                    chebyshev_lanczos_max_eig_inflation=block_lanczos_max_eig_inflation,
                    chebyshev_lanczos_min_eig_deflation=block_lanczos_min_eig_deflation,
                    chebyshev_lanczos_min_eig_floor_fraction=block_lanczos_min_eig_floor_fraction,
                    chebyshev_seed=400 + int(dirichlet),
                    richardson_steps=richardson_steps,
                    richardson_omega=richardson_omega,
                    true_block_apply=true_apply,
                )
                factors = _annotate_tensor_block_chebyshev_bounds(
                    factors,
                    lanczos_iterations=block_lanczos_iterations,
                    lanczos_max_eig_inflation=block_lanczos_max_eig_inflation,
                    lanczos_min_eig_deflation=block_lanczos_min_eig_deflation,
                    lanczos_min_eig_floor_fraction=block_lanczos_min_eig_floor_fraction,
                    seed=400 + int(dirichlet),
                    true_block_apply=true_apply,
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


def _make_mass_bulk_inverse(k: int, surgery, factors):
    """Per-k bulk inverse closure for the generic surgery-Schur layer.

    k=0 is a single scalar fast-diagonalization block; k=1/k=2 are the
    3-component vector bulk inverses (optionally with the inner r/theta/zeta
    Schur). These are the genuinely k-specific plug-ins.
    """
    if k == 0:
        bulk_true = lambda x: _apply_extracted_submatrix(
            surgery.apply_data, surgery.bulk_indices, surgery.bulk_indices, x)
        return lambda rhs_b: _apply_tensor_exact_block(
            None, factors.bulk, rhs_b, true_block_apply=bulk_true)
    if k == 1:
        bulk_apply = _apply_k1_bulk_preconditioner if factors.bulk_schur else _apply_k1_bulk_diagonal_preconditioner
        return lambda rhs_b: bulk_apply(surgery, factors.arr, factors.theta, factors.zeta, rhs_b)
    if k == 2:
        bulk_apply = _apply_k2_bulk_preconditioner if factors.bulk_schur else _apply_k2_bulk_diagonal_preconditioner
        return lambda rhs_b: bulk_apply(surgery, factors.r_bulk, factors.theta, factors.zeta, rhs_b)
    raise ValueError(f"surgery-Schur mass bulk inverse only supports k=0, k=1, k=2 (got k={k})")


def _make_mass_bulk_forward(k: int, surgery, factors):
    """Per-k bulk forward-model closure for the generic surgery-Schur layer."""
    if k == 0:
        return lambda rhs_b: _apply_tensor_diagonal_block_forward(factors.bulk, rhs_b)
    if k == 1:
        return lambda rhs_b: _apply_k1_bulk_forward_model(surgery, factors.arr, factors.theta, factors.zeta, rhs_b)
    if k == 2:
        return lambda rhs_b: _apply_k2_bulk_forward_model(surgery, factors.r_bulk, factors.theta, factors.zeta, rhs_b)
    raise ValueError(f"surgery-Schur mass bulk forward only supports k=0, k=1, k=2 (got k={k})")


def apply_mass_tensor_preconditioner(seq, preconds: Optional[MassPreconditioners], v, k: int, dirichlet: bool = True, *, true_block_apply_k3=None):
    factors = _select_mass_tensor_factors(preconds, k, dirichlet)
    if k == 3:
        # k=3 has no surgery split: a single scalar tensor block, no coupling.
        return _apply_tensor_exact_block(None, factors, v, true_block_apply=true_block_apply_k3)
    if k not in (0, 1, 2):
        raise ValueError(f"Tensor mass preconditioner currently only supports k=0, k=1, k=2 and k=3 (got k={k})")
    surgery = _select_mass_surgery_factors(preconds, k, dirichlet)
    bulk_inv = _make_mass_bulk_inverse(k, surgery, factors)
    return _apply_surgery_schur(surgery, factors.schur_inv, bulk_inv, v)


def apply_mass_tensor_forward_model(seq, preconds: Optional[MassPreconditioners], v, k: int, dirichlet: bool = True):
    del seq
    factors = _select_mass_tensor_factors(preconds, k, dirichlet)
    if k == 3:
        return _apply_tensor_diagonal_block_forward(factors, v)
    if k not in (0, 1, 2):
        raise ValueError(f"Tensor mass forward model currently only supports k=0, k=1, k=2 and k=3 (got k={k})")
    surgery = _select_mass_surgery_factors(preconds, k, dirichlet)
    bulk_fwd = _make_mass_bulk_forward(k, surgery, factors)
    return _apply_surgery_schur_forward(surgery, bulk_fwd, v)
def _symmetrize(matrix: jnp.ndarray) -> jnp.ndarray:
    return 0.5 * (matrix + matrix.T)


def _spd_clamped_inverse(
    matrix: jnp.ndarray, *, rel_floor: float = 1e-8
) -> jnp.ndarray:
    """SPD-projected inverse of a symmetric ``matrix``.

    Eigendecompose, lift any eigenvalue below ``rel_floor * max_eigenvalue``
    up to that floor, then invert from the clamped spectrum. For a genuinely
    SPD, well-conditioned block this is a no-op (every eigenvalue already sits
    above the floor) and reduces to the plain inverse. For an indefinite block
    -- which the rank-1 Kronecker path can produce when a greedy-CP weight
    factor changes sign on a non-separable (e.g. W7-X) metric -- it projects the
    factor back onto the SPD cone, guaranteeing the assembled tensor
    preconditioner stays SPD so PCG/Chebyshev cannot break down.
    """
    evals, vecs = jnp.linalg.eigh(_symmetrize(matrix))
    floor = rel_floor * jnp.maximum(jnp.max(evals), jnp.asarray(1e-300, jnp.float64))
    clamped = jnp.maximum(evals, floor)
    return _symmetrize((vecs * (1.0 / clamped)) @ vecs.T)


def _simultaneous_diagonalize_pair(M: jnp.ndarray, A: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Simultaneously diagonalize an SPD ``M`` and a symmetric ``A``.

    Returns ``(V, lam)`` such that ``V.T @ M @ V = I`` and
    ``V.T @ A @ V = diag(lam)``, via Cholesky ``M = L L.T`` and a
    symmetric eigendecomposition of ``L^{-1} A L^{-T}``. This is the
    per-axis primitive of the Lynch fast-diagonalization (FD) method:
    given a 3D Kronecker sum ``M_r (x) M_t (x) M_z + A_r (x) A_t (x) A_z``
    with all ``M_axis`` SPD, applying the per-axis ``V`` reduces it to
    the diagonal ``1 + lam_r (x) lam_t (x) lam_z`` in the M-orthonormal
    basis. Reusable by the stiffness preconditioner.
    """
    M_sym = _symmetrize(jnp.asarray(M, dtype=jnp.float64))
    A_sym = _symmetrize(jnp.asarray(A, dtype=jnp.float64))
    L = jnp.linalg.cholesky(M_sym)
    Linv_A = jax.scipy.linalg.solve_triangular(L, A_sym, lower=True)
    B = jax.scipy.linalg.solve_triangular(L, Linv_A.T, lower=True).T
    B = _symmetrize(B)
    lam, U = jnp.linalg.eigh(B)
    V = jax.scipy.linalg.solve_triangular(L.T, U, lower=False)
    return V, lam


def _build_kron_sum_fd_factors(
    mass_r: jnp.ndarray, mass_t: jnp.ndarray, mass_z: jnp.ndarray,
    aux_r: jnp.ndarray, aux_t: jnp.ndarray, aux_z: jnp.ndarray,
) -> dict:
    """Assemble per-axis FD factors for ``mass + aux`` (sum of two Kron terms).

    Both Kronecker triples must be SPD on their axis; the resulting
    diagonal ``1 + lam_r (x) lam_t (x) lam_z`` is checked to be positive.
    Returns a dict with keys ``V_r/V_t/V_z``, ``lam_r/lam_t/lam_z``,
    and ``inv_denom`` (precomputed reciprocal of ``1 + lam_r lam_t lam_z``).
    """
    V_r, lam_r = _simultaneous_diagonalize_pair(mass_r, aux_r)
    V_t, lam_t = _simultaneous_diagonalize_pair(mass_t, aux_t)
    V_z, lam_z = _simultaneous_diagonalize_pair(mass_z, aux_z)
    denom = (
        1.0
        + lam_r[:, None, None] * lam_t[None, :, None] * lam_z[None, None, :]
    )
    min_denom = float(jnp.min(denom))
    if not jnp.isfinite(min_denom) or min_denom <= 0.0:
        raise ValueError(
            "Rank-2 Kronecker sum is not SPD: min(1 + lam_r*lam_t*lam_z) = "
            f"{min_denom:.3e}. Reduce to rank-1 or check assembly."
        )
    return {
        "V_r": V_r, "V_t": V_t, "V_z": V_z,
        "lam_r": lam_r, "lam_t": lam_t, "lam_z": lam_z,
        "inv_denom": 1.0 / denom,
    }


def _mass_orthonormal_basis(mass: jnp.ndarray) -> jnp.ndarray:
    mass_sym = _symmetrize(jnp.asarray(mass, dtype=jnp.float64))
    L = jnp.linalg.cholesky(mass_sym)
    eye = jnp.eye(mass_sym.shape[0], dtype=mass_sym.dtype)
    return jax.scipy.linalg.solve_triangular(L.T, eye, lower=False)


def _modal_diagonal_from_basis(basis: jnp.ndarray, matrix: jnp.ndarray) -> jnp.ndarray:
    matrix_sym = _symmetrize(jnp.asarray(matrix, dtype=jnp.float64))
    return jnp.einsum("ji,jk,ki->i", basis, matrix_sym, basis)


def _modal_regularized_inverse_denom(
    denom: jnp.ndarray,
    *,
    relative_tol: float = 1e-8,
) -> jnp.ndarray:
    denom = jnp.asarray(denom, dtype=jnp.float64)
    scale = jnp.max(jnp.abs(denom))
    cutoff = jnp.maximum(
        jnp.asarray(relative_tol, dtype=denom.dtype) * scale,
        jnp.asarray(1e-14, dtype=denom.dtype),
    )
    return jnp.where(denom > cutoff, 1.0 / denom, 0.0)


def _build_mass_referenced_tensor_block_factors(
    *,
    full_shape: tuple[int, int, int],
    reference_r: jnp.ndarray,
    reference_t: jnp.ndarray,
    reference_z: jnp.ndarray,
    axis_operator_r: Optional[jnp.ndarray],
    axis_operator_t: Optional[jnp.ndarray],
    axis_operator_z: Optional[jnp.ndarray],
    term_matrices: tuple[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], ...],
    cp_relative_error: Optional[float],
    cp_final_delta: Optional[float],
    chebyshev_steps: int = 0,
    chebyshev_lanczos_iterations: int = 16,
    chebyshev_lanczos_max_eig_inflation: float = 1.1,
    chebyshev_lanczos_min_eig_deflation: float = 0.85,
    chebyshev_lanczos_min_eig_floor_fraction: float = 1e-3,
    chebyshev_seed: int = 0,
    richardson_steps: int = 0,
    richardson_omega: float = 1.0,
    modal_pinv_tol: float = 1e-8,
    true_block_apply=None,
) -> TensorDiagonalBlockInverseFactors:
    if len(term_matrices) < 1:
        raise ValueError("Mass-referenced tensor block builder requires at least one Kronecker term")

    def _axis_basis(reference_mass: jnp.ndarray, operator: Optional[jnp.ndarray]):
        if operator is None:
            basis = _mass_orthonormal_basis(reference_mass)
            lam = jnp.ones((reference_mass.shape[0],), dtype=jnp.float64)
            return basis, lam
        return _simultaneous_diagonalize_pair(reference_mass, operator)

    fd_V_r, fd_lam_r = _axis_basis(reference_r, axis_operator_r)
    fd_V_t, fd_lam_t = _axis_basis(reference_t, axis_operator_t)
    fd_V_z, fd_lam_z = _axis_basis(reference_z, axis_operator_z)

    denom = jnp.zeros(full_shape, dtype=jnp.float64)
    for term_r, term_t, term_z in term_matrices:
        d_r = _modal_diagonal_from_basis(fd_V_r, term_r)
        d_t = _modal_diagonal_from_basis(fd_V_t, term_t)
        d_z = _modal_diagonal_from_basis(fd_V_z, term_z)
        denom = denom + d_r[:, None, None] * d_t[None, :, None] * d_z[None, None, :]

    return _maybe_autotune_richardson_omega(
        TensorDiagonalBlockInverseFactors(
            shape=full_shape,
            cp_relative_error=cp_relative_error,
            cp_final_delta=cp_final_delta,
            split_backbone_relative_norm=None,
            split_correction_relative_norm=None,
            split_correction_over_backbone=None,
            split_backbone_residual_relative=None,
            chebyshev_steps=chebyshev_steps,
            chebyshev_lambda_min=None,
            chebyshev_lambda_max=None,
            richardson_steps=richardson_steps,
            richardson_omega=jnp.asarray(richardson_omega, dtype=jnp.float64),
            direct_inv_r=None,
            direct_inv_t=None,
            direct_inv_z=None,
            dense_inverse=None,
            split_backbone_inv_r=None,
            split_backbone_inv_t=None,
            split_backbone_inv_z=None,
            fd_V_r=fd_V_r,
            fd_V_t=fd_V_t,
            fd_V_z=fd_V_z,
            fd_lam_r=fd_lam_r,
            fd_lam_t=fd_lam_t,
            fd_lam_z=fd_lam_z,
            fd_inv_denom=_modal_regularized_inverse_denom(
                denom,
                relative_tol=modal_pinv_tol,
            ),
            term_r=tuple(t[0] for t in term_matrices),
            term_t=tuple(t[1] for t in term_matrices),
            term_z=tuple(t[2] for t in term_matrices),
        ),
        lanczos_iterations=chebyshev_lanczos_iterations,
        lanczos_max_eig_inflation=chebyshev_lanczos_max_eig_inflation,
        lanczos_min_eig_deflation=chebyshev_lanczos_min_eig_deflation,
        lanczos_min_eig_floor_fraction=chebyshev_lanczos_min_eig_floor_fraction,
        seed=chebyshev_seed,
        true_block_apply=true_block_apply,
    )


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


# ---------------------------------------------------------------------------
# Diagonal probing utilities (matrix-free, probing-based)
# ---------------------------------------------------------------------------

def diag_matvec(A_matvec, n, *, dtype=jnp.float64, batch_size=None):
    """Probe ``diag(A)`` from a forward operator on the extracted space.

    The operator is queried on small batches of canonical basis vectors.
    This is the matrix-free-compatible way to extract a diagonal.
    """
    if batch_size is None:
        configured_batch_size = mrx.MAP_BATCH_SIZE_OUTER
        if configured_batch_size is None:
            batch_size = 16
        else:
            batch_size = max(1, min(int(configured_batch_size), 16))
    if n == 0:
        return jnp.zeros((0,), dtype=dtype)
    diag_chunks = []
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        idx = jnp.arange(start, stop)
        basis = jax.nn.one_hot(idx, n, dtype=dtype)
        images = jax.vmap(A_matvec)(basis)
        diag_chunks.append(images[jnp.arange(stop - start), idx])
    return jnp.concatenate(diag_chunks)


def diag_EAET(E, A, E_T=None):
    """Compute ``diag(E @ A @ E^T)`` via probed matvecs (matrix-free)."""
    n = E.shape[0]
    if E_T is None:
        if isinstance(E, jsparse.BCSR):
            coo_idx = _bcsr_to_coo_indices(E)
            E_T = jsparse.BCOO((E.data, coo_idx), shape=E.shape).T
        else:
            E_T = E.T
    dtype = getattr(A, "dtype", getattr(E, "dtype", jnp.float64))
    return diag_matvec(lambda x: E @ (A @ (E_T @ x)), n, dtype=dtype)


def diag_EAET_matvec(E, A_matvec, n, E_T=None):
    """Compute ``diag(E @ A @ E^T)`` with ``A`` given as a matvec (matrix-free)."""
    if E_T is None:
        if isinstance(E, jsparse.BCSR):
            coo_idx = _bcsr_to_coo_indices(E)
            E_T = jsparse.BCOO((E.data, coo_idx), shape=E.shape).T
        else:
            E_T = E.T
    dtype = getattr(E, "dtype", jnp.float64)
    return diag_matvec(lambda x: E @ A_matvec(E_T @ x), n, dtype=dtype)


def diag_schur_complement(apply_DT, diag_inv, n):
    """Compute ``diag(D @ diag(diag_inv) @ D^T)`` via probed matvecs (matrix-free).

    For each row ``i``: ``e_i^T D diag(diag_inv) D^T e_i =
    ||diag_inv^{1/2} D^T e_i||^2``.
    """
    def entry(i):
        e_i = jnp.zeros(n).at[i].set(1.0)
        Dt_ei = apply_DT(e_i)
        return jnp.dot(Dt_ei, diag_inv * Dt_ei)
    return jax.lax.map(entry, jnp.arange(n), batch_size=mrx.MAP_BATCH_SIZE_OUTER)


# ---------------------------------------------------------------------------
# TODO: remove — the functions below access sparse .data arrays directly and
# are incompatible with the matrix-free paradigm. Jacobi preconditioners
# should use diag_matvec / diag_EAET probing instead.
# ---------------------------------------------------------------------------

def _bcsr_to_coo_indices(mat: jsparse.BCSR):
    """Expand BCSR indptr to COO-style (row, col) index array."""
    nse = mat.data.shape[0]
    lengths = mat.indptr[1:] - mat.indptr[:-1]
    rows = jnp.repeat(jnp.arange(mat.shape[0]), lengths,
                      total_repeat_length=nse)
    return jnp.stack([rows, mat.indices], axis=1)


def extract_diag_vector(mat) -> jnp.ndarray:
    """Extract the main diagonal of a sparse matrix as a 1-D array.

    .. deprecated::
        Reads ``.data`` directly — incompatible with matrix-free paradigm.
        Use :func:`diag_matvec` instead.
    """
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


def _coo_indices_host(mat):
    """Return ``(rows, cols)`` as host int64 numpy arrays.

    .. deprecated:: incompatible with matrix-free paradigm.
    """
    if isinstance(mat, jsparse.BCSR):
        idx = _bcsr_to_coo_indices(mat)
        rows = np.asarray(idx[:, 0], dtype=np.int64)
        cols = np.asarray(idx[:, 1], dtype=np.int64)
    else:
        rows = np.asarray(mat.indices[:, 0], dtype=np.int64)
        cols = np.asarray(mat.indices[:, 1], dtype=np.int64)
    return rows, cols


def _coo_host(mat):
    """Return ``(rows, cols, vals)`` as host numpy arrays.

    .. deprecated:: incompatible with matrix-free paradigm.
    """
    rows, cols = _coo_indices_host(mat)
    vals = np.asarray(mat.data, dtype=np.float64)
    return rows, cols, vals


def _build_diag_EAET_plan(rows_E, cols_E, vals_E, n_in, a_arr, b_arr,
                           chunk=1_000_000):
    """Build a static scatter plan for ``diag(E A E^T)``.

    .. deprecated:: incompatible with matrix-free paradigm. Use
        :func:`diag_EAET` (probing) instead.
    """
    counts = np.bincount(cols_E, minlength=n_in)
    R = int(counts.max()) if counts.size else 0
    if R == 0:
        empty_i = np.zeros((0,), dtype=np.int64)
        return empty_i, empty_i.copy(), np.zeros((0,), dtype=np.float64)
    order = np.argsort(cols_E, kind="stable")
    cs = cols_E[order]; rs = rows_E[order]; ws = vals_E[order]
    start = np.zeros(n_in, dtype=np.int64)
    if n_in > 0:
        start[1:] = np.cumsum(counts)[:-1]
    pos = np.arange(cs.shape[0], dtype=np.int64) - start[cs]
    row_pad = np.full((n_in, R), -1, dtype=np.int64)
    w_pad = np.zeros((n_in, R), dtype=np.float64)
    row_pad[cs, pos] = rs; w_pad[cs, pos] = ws
    nnz = a_arr.shape[0]
    seg_i_list, seg_m_list, seg_coef_list = [], [], []
    for s in range(0, nnz, chunk):
        e = min(s + chunk, nnz)
        a = a_arr[s:e]; b = b_arr[s:e]
        ra = row_pad[a]; wa = w_pad[a]
        rb = row_pad[b]; wb = w_pad[b]
        RA = ra[:, :, None]; RB = rb[:, None, :]
        match = (RA == RB) & (RA >= 0)
        coef = wa[:, :, None] * wb[:, None, :]
        mp = np.broadcast_to(
            np.arange(s, e, dtype=np.int64)[:, None, None], match.shape)
        iidx = np.broadcast_to(RA, match.shape)
        seg_i_list.append(iidx[match])
        seg_m_list.append(mp[match])
        seg_coef_list.append(coef[match])
    seg_i = np.concatenate(seg_i_list) if seg_i_list else np.zeros((0,), np.int64)
    seg_m = np.concatenate(seg_m_list) if seg_m_list else np.zeros((0,), np.int64)
    seg_coef = (np.concatenate(seg_coef_list)
                if seg_coef_list else np.zeros((0,), np.float64))
    return seg_i, seg_m, seg_coef


def diag_EAET_direct(E, A):
    """Compute ``diag(E @ A @ E^T)`` via a static scatter plan.

    .. deprecated:: incompatible with matrix-free paradigm. Use
        :func:`diag_EAET` (probing) instead.
    """
    n_out, n_in = E.shape
    rows_E, cols_E, vals_E = _coo_host(E)
    a_arr, b_arr = _coo_indices_host(A)
    seg_i, seg_m, seg_coef = _build_diag_EAET_plan(
        rows_E, cols_E, vals_E, n_in, a_arr, b_arr)
    if seg_i.shape[0] == 0:
        return jnp.zeros((n_out,), dtype=jnp.float64)
    contrib = jnp.asarray(seg_coef) * A.data[jnp.asarray(seg_m)]
    return jax.ops.segment_sum(contrib, jnp.asarray(seg_i), num_segments=n_out)


def diag_EGtMGEt_direct(E, G, M):
    """Compute ``diag(E @ G^T @ M @ G @ E^T)`` via a scatter plan.

    .. deprecated:: incompatible with matrix-free paradigm. Uses
        ``scipy.sparse`` and reads ``.data`` directly.
    """
    import scipy.sparse as sps
    n_out = E.shape[0]
    re, ce, ve = _coo_host(E)
    rg, cg_arr, vg = _coo_host(G)
    E_sp = sps.csr_matrix((ve, (re, ce)), shape=E.shape)
    G_sp = sps.csr_matrix((vg, (rg, cg_arr)), shape=G.shape)
    Eeff = (E_sp @ G_sp.transpose()).tocoo()
    n_in = M.shape[0]
    a_arr, b_arr = _coo_indices_host(M)
    seg_i, seg_m, seg_coef = _build_diag_EAET_plan(
        np.asarray(Eeff.row, dtype=np.int64),
        np.asarray(Eeff.col, dtype=np.int64),
        np.asarray(Eeff.data, dtype=np.float64),
        n_in, a_arr, b_arr)
    if seg_i.shape[0] == 0:
        return jnp.zeros((n_out,), dtype=jnp.float64)
    contrib = jnp.asarray(seg_coef) * M.data[jnp.asarray(seg_m)]
    return jax.ops.segment_sum(contrib, jnp.asarray(seg_i), num_segments=n_out)
