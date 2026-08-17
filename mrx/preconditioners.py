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
















class TensorMassPreconditioner(eqx.Module):
    ranks: tuple = eqx.field(static=True, default=(3, 3, 3, 3))
    cp_maxiter: int = eqx.field(static=True, default=100)
    cp_tol: float = eqx.field(static=True, default=1e-9)
    cp_ridge: float = eqx.field(static=True, default=1e-12)
    surgery_schur_pinv_tol: float = eqx.field(static=True, default=1e-8)
    k0: BoundaryConditionPair = eqx.field(default_factory=BoundaryConditionPair)
    k1: BoundaryConditionPair = eqx.field(default_factory=BoundaryConditionPair)
    k2: BoundaryConditionPair = eqx.field(default_factory=BoundaryConditionPair)
    k3: BoundaryConditionPair = eqx.field(default_factory=BoundaryConditionPair)


class MassPreconditioners(eqx.Module):
    # ``surgery`` is annotated loosely because MassSurgeryPreconditioner now
    # lives in mrx/experimental/mass_surgery.py, and importing it here at class
    # definition time would defeat the lazy re-export below (and reintroduce the
    # import cycle). The slot still holds exactly that type when populated.
    jacobi: Optional[JacobiMassPreconditioner] = None
    surgery: Optional[eqx.Module] = None
    tensor: Optional[TensorMassPreconditioner] = None
    # raw_kron preconditioner: {(k, dirichlet): RawKronMassFactors}. The default
    # mass path as of 2026-08-17; see docs/research/mass_preconditioner_pivot.md.
    raw_kron: Optional[dict] = None


def tensor_mass_rank_for_degree(tensor: TensorMassPreconditioner, k: int) -> int:
    if k not in (0, 1, 2, 3):
        raise ValueError("k must be 0, 1, 2 or 3")
    return int(tensor.ranks[k])


@dataclass(frozen=True)
class MassPreconditionerSpec:
    # kinds: none | jacobi | tensor | raw_kron
    #   raw_kron = Kronecker model on the raw grid with pseudoinverse
    #     extraction transfer; the production default since
    #     2026-08-17 (docs/research/mass_preconditioner_pivot.md)
    #   tensor   = surgery/Schur split, retained as a fallback; its machinery
    #     lives in mrx/experimental/mass_surgery.py
    #   (richardson/chebyshev removed 2026-08-14, see mrx/experimental/chebyshev.py)
    kind: str = 'raw_kron'
    surgery_schur: bool = False
    schur_diag_mode: str = 'raw_kron_probe'
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
    """The production mass preconditioner: the raw_kron preconditioner.

    Changed 2026-08-17 from ``kind='tensor', surgery_schur=True``. raw_kron needs no
    surgery split, no dense Schur complement and no CP fit, and its
    ``(CC^T)^-1`` is metric independent -- it depends only on the sparsity of
    ``E``, so it never rebuilds when the geometry changes.
    """
    return MassPreconditionerSpec(kind='raw_kron', surgery_schur=False)


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


def _extracted_mass_diagonal(e, d_raw, mass_apply, *, batch_size: int = 16):
    """``diag(E M E^T)`` from the raw diagonal, probing only the coupled rows.

    ``E`` has exactly two kinds of row (verified for k=0,1,2 x both BCs at
    every resolution):

    * **bulk** -- a single nonzero, so ``(E M E^T)_ii = v^2 M_aa`` and the
      closed-form raw diagonal supplies it outright, with no operator apply;
    * **coupled** -- the polar rows, which mix several raw DOFs and therefore
      pick up *off-diagonal* entries of ``M`` that no diagonal can supply.

    Only the coupled rows are probed. There are ``3 n_z / 5 n_z / 2 n_z / 0``
    of them for k=0/1/2/3, so the apply count drops from ``O(N)`` to
    ``O(n_z)`` while the result stays exact -- this is not an approximation of
    the probed diagonal, it agrees with it to floating point.
    """
    rows = np.asarray(e.rows)
    cols = np.asarray(e.cols)
    vals = np.asarray(e.vals)
    n_rows, n_raw = (int(s) for s in e.forward_shape)
    d_raw_np = np.asarray(d_raw)

    counts = np.bincount(rows, minlength=n_rows)
    diag = np.zeros(n_rows, dtype=np.float64)

    # Bulk rows: one nonzero each, so only the raw diagonal is involved.
    single = counts[rows] == 1
    diag[rows[single]] = (vals[single] ** 2) * d_raw_np[cols[single]]

    # Coupled rows: e_i^T M e_i with e_i the (short) raw row of E.
    coupled = np.flatnonzero(counts > 1)
    for start in range(0, coupled.size, batch_size):
        blk = coupled[start:start + batch_size]
        probe = np.zeros((blk.size, n_raw), dtype=np.float64)
        for t, r in enumerate(blk):
            sel = rows == r
            probe[t, cols[sel]] = vals[sel]
        probe_j = jnp.asarray(probe)
        images = jax.vmap(mass_apply)(probe_j)
        diag[blk] = np.asarray(jnp.sum(images * probe_j, axis=1))

    return jnp.asarray(diag)


def build_mass_jacobi_pair(seq, mass_apply, k: int) -> BoundaryConditionPair:
    """Build a Jacobi (diagonal-inverse) pair for the k-form mass matrix.

    ``mass_apply`` is the raw-DOF-space matvec ``v -> M_k v`` returned by
    :func:`mrx.operators.build_matrixfree_mass_apply`.

    ``diag(E M_k E^T)`` comes from the **closed-form** raw mass diagonal
    (:func:`mrx.local_assembly.build_mass_diagonal` -- one sum-factorized
    contraction against squared basis tables) rather than from ``O(N)``
    canonical-basis probes. Only the ``O(n_z)`` coupled polar rows still need
    an apply. The result is exact, not an estimate.
    """
    from mrx.local_assembly import build_mass_diagonal  # noqa: PLC0415

    d_raw = build_mass_diagonal(seq, k)
    e = getattr(seq, f"e{k}")
    e_dbc = getattr(seq, f"e{k}_dbc")
    return BoundaryConditionPair(
        free=1.0 / _extracted_mass_diagonal(e, d_raw, mass_apply),
        dbc=1.0 / _extracted_mass_diagonal(e_dbc, d_raw, mass_apply),
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
        s = factors.greville_inv_sqrt_D
        f = jnp.asarray(rhs).reshape(nr, nt, nz) * s
        if factors.greville_inv_r is not None:
            # Product sandwich (mass / single-term): D^{-1/2}(M0_r^{-1}xM0_t^{-1}xM0_z^{-1})D^{-1/2}.
            f = jnp.einsum("ij,jkl->ikl", factors.greville_inv_r, f)
            f = jnp.einsum("ij,kjl->kil", factors.greville_inv_t, f)
            f = jnp.einsum("ij,klj->kli", factors.greville_inv_z, f)
        else:
            # Additive-FD sandwich (greville P_A stiffness): D^{-1/2} V diag(1/denom) V^T D^{-1/2}.
            f = jnp.einsum("ji,jkl->ikl", factors.fd_V_r, f)
            f = jnp.einsum("ji,kjl->kil", factors.fd_V_t, f)
            f = jnp.einsum("ji,klj->kli", factors.fd_V_z, f)
            f = f * factors.fd_inv_denom
            f = jnp.einsum("ij,jkl->ikl", factors.fd_V_r, f)
            f = jnp.einsum("ij,kjl->kil", factors.fd_V_t, f)
            f = jnp.einsum("ij,klj->kli", factors.fd_V_z, f)
        f = f * s
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
    """Apply the tensor diagonal block inverse.

    (The optional true-block Richardson polish was removed 2026-08-14 with
    the rest of the relaxation machinery -- see mrx/experimental/chebyshev.py.
    ``true_block_apply`` is retained in the signature for call-site
    compatibility but is unused.)
    """
    del true_block_apply
    return _apply_tensor_diagonal_block_preconditioner(factors, rhs)










def _schur_blocks(matrix: jnp.ndarray, surgery_size: int):
    ass = matrix[:surgery_size, :surgery_size]
    asb = matrix[:surgery_size, surgery_size:]
    abs_ = matrix[surgery_size:, :surgery_size]
    abb = matrix[surgery_size:, surgery_size:]
    return ass, asb, abs_, abb


















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

    return TensorDiagonalBlockInverseFactors(
        shape=full_shape,
        cp_relative_error=cp_relative_error,
        cp_final_delta=cp_final_delta,
        split_backbone_relative_norm=split_backbone_relative_norm,
        split_correction_relative_norm=split_correction_relative_norm,
        split_correction_over_backbone=split_correction_over_backbone,
        split_backbone_residual_relative=split_backbone_residual_relative,
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
    )


def _apply_tensor_exact_block(
    block_matrix: jnp.ndarray,
    factors: TensorDiagonalBlockInverseFactors,
    rhs: jnp.ndarray,
    *,
    true_block_apply=None,
) -> jnp.ndarray:
    del block_matrix
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
    """PSD (positive-part) pseudoinverse. Both call sites invert Schur
    complements of SPD operators, which are PSD analytically but can dip
    slightly negative when rebuilt through an approximate bulk inverse;
    inverting a near-null eigenvalue by magnitude WITH its roundoff sign
    injects a huge negative Rayleigh direction and stalls CG (observed as
    ~1e-2 residual floors in the 2026-08-13 single-level campaign). Negative
    and sub-cutoff eigenvalues are dropped instead."""
    matrix = _symmetrize(matrix)
    eigvals, eigvecs = jnp.linalg.eigh(matrix)
    scale = jnp.max(jnp.abs(eigvals))
    safe_scale = jnp.where(scale > 0, scale, 1.0)
    cutoff = relative_tol * safe_scale
    inv_eigvals = jnp.where(eigvals > cutoff, 1.0 / eigvals, 0.0)
    return _symmetrize((eigvecs * inv_eigvals[jnp.newaxis, :]) @ eigvecs.T)


















































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
    modal_pinv_tol: float = 1e-8,
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

    return TensorDiagonalBlockInverseFactors(
        shape=full_shape,
        cp_relative_error=cp_relative_error,
        cp_final_delta=cp_final_delta,
        split_backbone_relative_norm=None,
        split_correction_relative_norm=None,
        split_correction_over_backbone=None,
        split_backbone_residual_relative=None,
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
    )


def _assemble_weighted_1d_mass(B: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    return (B * weights[None, :]) @ B.T


def _raw_kron_diff_flags(k: int, c: int) -> tuple:
    """Differentiated-axis flags for component ``c`` of a k-form.

    Mirrors ``_component_axis_bases_k0/k1/k2/k3`` in :mod:`mrx.local_assembly`:
    k=0 differentiates nothing, k=3 everything, k=1 only axis ``c``, and k=2
    every axis *except* ``c``.
    """
    match k:
        case 0:
            return (False, False, False)
        case 3:
            return (True, True, True)
        case 1:
            return tuple(a == c for a in range(3))
        case 2:
            return tuple(a != c for a in range(3))
    raise ValueError("k must be 0, 1, 2 or 3")


def _extraction_gram_inverse(e):
    """``(E E^T)^{-1}`` for raw_kron, as ``(coupled_rows, inverse)``.

    ``E E^T = diag(C C^T, I)``: bulk rows of ``E`` are orthonormal selectors, and
    the coupled/bulk cross block is exactly zero, so the Gram restricted to the
    coupled rows *is* ``C C^T``. It is block diagonal with blocks of size <= 3,
    so a dense inverse over the ``O(n_z)`` coupled rows reproduces the blocked
    inverse exactly while keeping the construction trivial.

    Returns ``(None, None)`` when there are no coupled rows (k=3), where the
    pseudoinverse degenerates to ``E^T`` and raw_kron is a plain tensor block.

    The returned ``cross`` is the largest coupled-bulk overlap found; it must be
    zero for the block structure to hold, and the caller asserts that rather
    than trusting the documented invariant.
    """
    rows = np.asarray(e.rows)
    cols = np.asarray(e.cols)
    vals = np.asarray(e.vals)
    n_rows = int(e.forward_shape[0])

    counts = np.bincount(rows, minlength=n_rows)
    coupled = np.flatnonzero(counts > 1)
    if coupled.size == 0:
        return None, None, 0.0

    pos = -np.ones(n_rows, dtype=np.int64)
    pos[coupled] = np.arange(coupled.size)

    by_col: dict = {}
    for r, c, v in zip(rows, cols, vals):
        by_col.setdefault(int(c), []).append((int(r), float(v)))

    m = int(coupled.size)
    gram = np.zeros((m, m), dtype=np.float64)
    cross = 0.0
    for entries in by_col.values():
        cp = [(pos[r], v) for r, v in entries if pos[r] >= 0]
        bulk = [v for r, v in entries if pos[r] < 0]
        if cp and bulk:
            cross = max(cross, max(abs(v1 * v2) for _, v1 in cp for v2 in bulk))
        for i, vi in cp:
            for j, vj in cp:
                gram[i, j] += vi * vj

    return jnp.asarray(coupled), jnp.asarray(np.linalg.inv(gram)), cross


def _raw_kron_block_apply(inv3, X):
    """Apply ``(M_r^{-1} x M_t^{-1} x M_z^{-1})`` to a ``(Sx,Sy,Sz)`` block."""
    X = jnp.tensordot(inv3[0], X, axes=([1], [0]))
    X = jnp.tensordot(inv3[1], X, axes=([1], [1])).transpose(1, 0, 2)
    X = jnp.tensordot(inv3[2], X, axes=([1], [2])).transpose(1, 2, 0)
    return X


def build_mass_raw_kron_factors(seq, k: int, *, dirichlet: bool, d_raw=None):
    """Build the raw_kron mass preconditioner factors for ``M_k``.

    The space is never split. A per-component diagonally-scaled Kronecker
    inverse acts on the full raw grid, and the pseudoinverse
    ``E+ = E^T (E E^T)^{-1}`` moves between raw and extracted coordinates::

        M^-1 ~ (E+)^T [ (+)_c D_c^-1/2 (M_r^-1 x M_t^-1 x M_z^-1)_c D_c^-1/2 ] E+

    with unweighted 1D masses (degree ``p`` on primal axes, ``p-1`` on each
    differentiated axis) and ``D`` the phi^2-weighted support average of the
    metric weight, taken straight from the exact mass diagonal.

    This is the same model class as the Greville-collocation sandwich --
    both are ``M_ab ~ sqrt(v_a v_b) (M_unw)_ab`` -- differing only in whether
    the weight is sampled at a point or averaged over the support. The averaged
    form is what makes it well defined on the innermost rings, where a Greville
    point sits at ``r ~ 0`` and ``J -> 0``.

    **Both sides must carry the full** ``(E E^T)^{-1}``, and this is the single
    easiest thing to get wrong: substituting ``E^T`` for ``E+`` still runs, and
    still looks acceptable on the easiest test case. In the pivot doc's
    pow0/pow1/pow2 ablation sweep -- pow-n = the correction applied n times --
    dropping it entirely (pow0) costs 2.3x the iterations at k=1 and drifts
    upward under refinement, because the mis-scaled subspace has dimension
    ``O(n_z)`` and grows; applying it once (pow1) recovers about half of that.
    ``raw_kron`` is the pow2 arm, i.e. the correction on both sides.

    Returns :class:`RawKronMassFactors`; use
    :func:`apply_mass_raw_kron_preconditioner` to apply them, or
    :func:`build_mass_raw_kron_preconditioner` for a ready-made jitted callable.
    """
    from mrx.local_assembly import build_mass_diagonal  # noqa: PLC0415

    e = getattr(seq, f"e{k}_dbc" if dirichlet else f"e{k}")
    form = getattr(seq, f"basis_{k}")
    shapes = [tuple(int(s) for s in sh) for sh in form.shape]
    n_comp = len(shapes)

    if d_raw is None:
        d_raw = build_mass_diagonal(seq, k)
    d_raw = jnp.asarray(d_raw)

    primal = (seq.basis_r_jk, seq.basis_t_jk, seq.basis_z_jk)
    deriv = (seq.d_basis_r_jk, seq.d_basis_t_jk, seq.d_basis_z_jk)
    quad_w = (seq.quad.w_x, seq.quad.w_y, seq.quad.w_z)

    inv_1d, inv_sqrt_D, starts = [], [], [0]
    for c in range(n_comp):
        diff = _raw_kron_diff_flags(k, c)
        bases = tuple(deriv[a] if diff[a] else primal[a] for a in range(3))
        m1 = [_assemble_weighted_1d_mass(bases[a], quad_w[a]) for a in range(3)]
        for a in range(3):
            if int(m1[a].shape[0]) != shapes[c][a]:
                raise ValueError(
                    f"raw_kron k={k} component {c} axis {a}: 1D mass is "
                    f"{m1[a].shape[0]} but the raw block axis is {shapes[c][a]}"
                )
        inv_1d.append(tuple(jnp.linalg.inv(m) for m in m1))

        # D = exact mass diagonal / unweighted Kronecker diagonal.
        kron_diag = jnp.einsum('i,j,l->ijl', jnp.diag(m1[0]),
                               jnp.diag(m1[1]), jnp.diag(m1[2]))
        size = int(np.prod(shapes[c]))
        d_c = d_raw[starts[-1]:starts[-1] + size].reshape(shapes[c])
        inv_sqrt_D.append(1.0 / jnp.sqrt(d_c / kron_diag))
        starts.append(starts[-1] + size)

    coupled, gram_inv, cross = _extraction_gram_inverse(e)
    if cross > 1e-12:
        raise ValueError(
            f"raw_kron k={k} dirichlet={dirichlet}: E E^T is not block diagonal "
            f"(max coupled-bulk overlap {cross:.3e}); the (CC^T, I) split that "
            "the pseudoinverse relies on does not hold here"
        )

    return RawKronMassFactors(
        inv_1d=tuple(inv_1d),
        inv_sqrt_D=tuple(inv_sqrt_D),
        coupled=coupled,
        gram_inv=gram_inv,
        shapes=tuple(shapes),
        starts=tuple(starts),
    )


class RawKronMassFactors(eqx.Module):
    """Precomputed raw_kron factors for one ``(k, dirichlet)`` pair.

    Storage is ``O(n_z)``: the 1D inverses are tiny dense blocks and
    ``gram_inv`` covers only the coupled rows. Nothing here depends on the
    element count, and ``gram_inv`` depends only on the *sparsity* of ``E``, so
    it never rebuilds when the geometry changes -- unlike ``coupling_sb``,
    which is ``O(N n_z)`` and metric dependent.
    """
    inv_1d: tuple
    inv_sqrt_D: tuple
    coupled: Optional[jnp.ndarray]
    gram_inv: Optional[jnp.ndarray]
    shapes: tuple = eqx.field(static=True)
    starts: tuple = eqx.field(static=True)


def apply_mass_raw_kron_preconditioner(factors: RawKronMassFactors, e, x):
    """Apply the raw_kron preconditioner to an extracted-space vector."""
    def gram_apply(v):
        if factors.coupled is None:
            return v
        return v.at[factors.coupled].set(factors.gram_inv @ v[factors.coupled])

    raw = e.T @ gram_apply(x)
    parts = []
    for c, shape in enumerate(factors.shapes):
        Xc = raw[factors.starts[c]:factors.starts[c + 1]].reshape(shape)
        Xc = _raw_kron_block_apply(
            factors.inv_1d[c], Xc * factors.inv_sqrt_D[c]) * factors.inv_sqrt_D[c]
        parts.append(Xc.reshape(-1))
    return gram_apply(e @ jnp.concatenate(parts))


def build_raw_kron_pinv_columns(factors: RawKronMassFactors, e):
    """Columns of ``E+ = E^T (E E^T)^{-1}`` in padded form.

    Returns ``(idx, coef)`` of shape ``(n_ext, w)``: for extracted index ``a``,
    ``E+[:, a]`` has raw entries ``coef[a, m]`` at raw index ``idx[a, m]``,
    zero-padded to a common width ``w``. Bulk columns carry a single entry
    (``(E E^T)^{-1}`` is the identity there); coupled polar columns carry the
    small ``(CC^T)^{-1}`` combination, so ``w`` stays tiny.

    This is what makes entrywise access to the raw_kron operator O(1): together
    with :func:`raw_kron_entry` it gives ``P_ab`` without any operator apply.
    """
    rows = np.asarray(e.rows)
    cols = np.asarray(e.cols)
    vals = np.asarray(e.vals)
    n_ext = int(e.forward_shape[0])

    # E^T column a  <-  row a of E.
    by_row: dict = {}
    for r, c, v in zip(rows, cols, vals):
        by_row.setdefault(int(r), []).append((int(c), float(v)))

    coupled = (np.asarray(factors.coupled) if factors.coupled is not None
               else np.zeros(0, dtype=np.int64))
    gram_inv = (np.asarray(factors.gram_inv) if factors.gram_inv is not None
                else np.zeros((0, 0)))
    pos = {int(r): t for t, r in enumerate(coupled)}

    entries = []
    for a in range(n_ext):
        if a in pos:
            # column a mixes the coupled rows through (CC^T)^{-1}
            acc: dict = {}
            ta = pos[a]
            for tb, b in enumerate(coupled):
                g = float(gram_inv[tb, ta])
                if g == 0.0:
                    continue
                for c, v in by_row.get(int(b), ()):
                    acc[c] = acc.get(c, 0.0) + v * g
            entries.append(sorted(acc.items()))
        else:
            entries.append(by_row.get(a, []))

    w = max((len(x) for x in entries), default=1)
    idx = np.zeros((n_ext, w), dtype=np.int64)
    coef = np.zeros((n_ext, w), dtype=np.float64)
    for a, ent in enumerate(entries):
        for m, (c, v) in enumerate(ent):
            idx[a, m] = c
            coef[a, m] = v
    return jnp.asarray(idx), jnp.asarray(coef)


def raw_kron_entry(factors: RawKronMassFactors, alpha, beta):
    """Raw-space entries ``K[alpha, beta]`` of the raw_kron kernel, vectorized.

    ``K = (+)_c D_c^{-1/2} (M_r^{-1} x M_t^{-1} x M_z^{-1})_c D_c^{-1/2}`` is
    block diagonal over components and a Kronecker product within each, so a
    single entry is three 1D-inverse lookups times two diagonal scalars -- O(1),
    no solve and no apply. Entries across different components are zero.

    ``alpha``/``beta`` are flat raw indices (any broadcastable shape).
    """
    alpha = jnp.asarray(alpha)
    beta = jnp.asarray(beta)
    starts = factors.starts
    total = jnp.zeros(jnp.broadcast_shapes(alpha.shape, beta.shape))
    for c, shape in enumerate(factors.shapes):
        lo, hi = starts[c], starts[c + 1]
        in_a = (alpha >= lo) & (alpha < hi)
        in_b = (beta >= lo) & (beta < hi)
        both = in_a & in_b
        la = jnp.clip(alpha - lo, 0, hi - lo - 1)
        lb = jnp.clip(beta - lo, 0, hi - lo - 1)
        sy, sz = shape[1], shape[2]
        ia, ja, ka = la // (sy * sz), (la // sz) % sy, la % sz
        ib, jb, kb = lb // (sy * sz), (lb // sz) % sy, lb % sz
        inv_r, inv_t, inv_z = factors.inv_1d[c]
        sD = factors.inv_sqrt_D[c]
        val = (sD[ia, ja, ka] * sD[ib, jb, kb]
               * inv_r[ia, ib] * inv_t[ja, jb] * inv_z[ka, kb])
        total = total + jnp.where(both, val, 0.0)
    return total


def raw_kron_extracted_entry(factors: RawKronMassFactors, pinv_idx, pinv_coef,
                             a, b):
    """Extracted-space entries ``P[a, b] = (E+)^T K E+`` at index pairs ``(a, b)``.

    ``a``/``b`` are extracted indices. Uses the padded ``E+`` columns from
    :func:`build_raw_kron_pinv_columns`, so the cost is ``w^2`` calls to
    :func:`raw_kron_entry` with ``w <= 3``.
    """
    ia, ca = pinv_idx[a], pinv_coef[a]        # (..., w)
    ib, cb = pinv_idx[b], pinv_coef[b]
    w = ia.shape[-1]
    acc = 0.0
    for m in range(w):
        for n_ in range(w):
            acc = acc + (ca[..., m] * cb[..., n_]
                         * raw_kron_entry(factors, ia[..., m], ib[..., n_]))
    return acc


def build_mass_raw_kron_preconditioner(seq, k: int, *, dirichlet: bool, d_raw=None):
    """Convenience wrapper: build the raw_kron factors and return a jitted apply."""
    e = getattr(seq, f"e{k}_dbc" if dirichlet else f"e{k}")
    factors = build_mass_raw_kron_factors(seq, k, dirichlet=dirichlet, d_raw=d_raw)
    return jax.jit(lambda x: apply_mass_raw_kron_preconditioner(factors, e, x))


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


# --------------------------------------------------------------------------- #
# Retired surgery / Schur / tensor machinery
# --------------------------------------------------------------------------- #
# Moved to mrx/experimental/mass_surgery.py on 2026-08-17, when raw_kron became
# the default mass preconditioner. Re-exported lazily rather than imported at
# module load, so that the dependency stays one-way: mass_surgery imports
# primitives from here, and this module never imports it at load time. A module
# __getattr__ only fires once THIS module has finished executing, so the
# back-import always sees a fully initialised module and there is no cycle.
#
# NOTE: this hook serves ATTRIBUTE access (``preconditioners.foo`` and
# ``from mrx.preconditioners import foo``). It does NOT serve bare global
# lookups from functions defined in this file -- Python resolves those against
# the module dict and builtins only. That is why the whole tensor path moved as
# a dependency closure rather than just the surgery leaves: anything left behind
# that called into the moved code would raise NameError at runtime, not resolve
# through here.
_SURGERY_EXPORTS = frozenset({
    "K0MassSurgeryPreconditionerFactors",
    "K0TensorMassPreconditionerFactors",
    "K1MassSurgeryPreconditionerFactors",
    "K1TensorMassPreconditionerFactors",
    "K2MassSurgeryPreconditionerFactors",
    "K2TensorMassPreconditionerFactors",
    "MassSurgeryPreconditioner",
    "_apply_bulk_to_surgery_coupling",
    "_apply_k1_bulk_diagonal_preconditioner",
    "_apply_k1_bulk_forward_model",
    "_apply_k1_bulk_preconditioner",
    "_apply_k1_rt_art_coupling",
    "_apply_k1_rt_atr_coupling",
    "_apply_k1_rt_forward_model",
    "_apply_k1_rt_preconditioner",
    "_apply_k1_rt_to_zeta_coupling",
    "_apply_k1_zeta_to_rt_coupling",
    "_apply_k2_bulk_diagonal_preconditioner",
    "_apply_k2_bulk_forward_model",
    "_apply_k2_bulk_preconditioner",
    "_apply_k2_r_to_theta_coupling",
    "_apply_k2_rt_forward_model",
    "_apply_k2_rt_preconditioner",
    "_apply_k2_rt_to_zeta_coupling",
    "_apply_k2_theta_to_r_coupling",
    "_apply_k2_zeta_to_rt_coupling",
    "_apply_surgery_schur",
    "_apply_surgery_schur_forward",
    "_apply_surgery_to_bulk_coupling",
    "_arr_shape_k1",
    "_assemble_surgery_schur_inverse_from_applies",
    "_component_sizes_k2",
    "_k1_layout_sizes",
    "_k2_rt_indices",
    "_make_mass_bulk_forward",
    "_make_mass_bulk_inverse",
    "_mass_surgery_pair",
    "_r_bulk_shape_k2",
    "_select_mass_surgery_factors",
    "_surgery_slices_k1",
    "_surgery_slices_k2",
    "_tensor_block_indices_k1",
    "_tensor_block_indices_k2",
    "_theta_bulk_shape_k1",
    "_theta_shape_k2",
    "_zeta_bulk_shape_k1",
    "_zeta_shape_k2",
    "apply_mass_tensor_forward_model",
    "apply_mass_tensor_preconditioner",
    "build_mass_surgery_preconditioner",
    "build_mass_tensor_preconditioner",
    "mass_surgery_available",
    "mass_tensor_available",
    "set_mass_surgery",
    "set_mass_surgery_pair",
})


def __getattr__(name):
    if name in _SURGERY_EXPORTS:
        from mrx.experimental import mass_surgery  # noqa: PLC0415
        return getattr(mass_surgery, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _SURGERY_EXPORTS)
