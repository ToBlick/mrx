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


@dataclass(frozen=True)
class MassPreconditionerSpec:
    # kinds: none | jacobi | tensor | raw_kron | block_jacobi
    #   block_jacobi = separable bulk with the polar core probed and inverted
    #     DENSELY. THE PRODUCTION DEFAULT since 2026-08-22 -- but that default
    #     lives in default_mass_preconditioner(), NOT in the field default
    #     below, which is still 'raw_kron'. Anything constructing a bare
    #     MassPreconditionerSpec() gets raw_kron, including
    #     SchurPreconditionerSpec.inner and default_saddle_preconditioner().
    #   raw_kron = the same separable-bulk shape, but the polar core is reached
    #     through the E+ pseudoinverse instead. Production 2026-08-17..08-22,
    #     still the schur.inner default (docs/research/mass_preconditioner_pivot.md).
    #   tensor   = surgery/Schur split. RETIRED from production 2026-08-17; kept
    #     reachable by explicit spec only, and its machinery lives in
    #     mrx/experimental/mass_surgery.py. Nothing in mrx/ or test/ selects it.
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
    """The production mass preconditioner: block_jacobi.

    Changed 2026-08-22 from ``kind='raw_kron'`` (which had replaced
    ``kind='tensor', surgery_schur=True`` on 2026-08-17). Same separable-bulk
    shape as raw_kron, but the polar core rows are PROBED AND INVERTED DENSELY
    instead of reached through the ``E+`` pseudoinverse, whose "both sides must
    carry the full ``(CC^T)^-1``" requirement raw_kron's own docstring calls
    the single easiest thing to get wrong.

    MEASURED (docs/research/production_simplification_plan.md
    §10), 224 cells over four geometries, n = 8..20, p = 2..5:

    * the mass solve itself: median **0.83x** raw_kron's iterations, and
      **0.70-0.77x** at k=1,2 where the cost is. The advantage HOLDS OR GROWS
      with h and is flat in p. Build time is equal (2.0 vs 2.2 s median), which
      matters because needing no eager assemble is why raw_kron was the
      default.
    * the effect on ``L_k`` -- the mass preconditioner is the weak term's inner
      inverse, so this changes the OPERATOR at k >= 1, not just the solve:
      **median 0.91x, better in 12 of 16 cells**, up to 0.79x on the Dirichlet
      rows. Only regression is cylinder k=1 (1.07x).
    * the natural-BC scale SURVIVES: worst-case penalty against each cell's
      own optimum moves 1.14 -> 1.22, and only on the toroid, where the basin
      is flat. The shaped geometries are unchanged (1.01-1.04).

    Only regression anywhere is ~5% at k=0, on mass solves that take 7-17
    iterations either way.

    **THIS IS THE SWAP POINT.** Every mass-preconditioner decision routes
    through this function (``mrx/operators.py`` x3 and ``mrx/nullspace.py``),
    so moving to the block-Jacobi mass is exactly::

        return MassPreconditionerSpec(kind='raw_kron', surgery_schur=False)

    to go back. Everything is in place for either: ``kind='block_jacobi'`` is accepted by
    the spec validator, dispatched in
    ``_build_operator_preconditioner_apply``, available as ``schur.inner``, and
    built on demand by ``_mass_block_jacobi_for`` (memoised on the sequence and
    invalidated by ``set_map``), with
    ``assemble_mass_block_jacobi_preconditioner`` for an eager build.

    THE BUILD IS NOT JIT-SAFE, AND DOES NOT NEED TO BE. It is host-side numpy
    (1-D inverses, a dense core probe). What that costs is that a COLD cache
    inside a traced loop dies -- and once the main kind and ``schur.inner``
    differ, schur.inner's raw_kron factors are cold exactly there. The apply
    was made jit-safe in 3bd62aa; the build is instead warmed OUTSIDE the loop
    by ``operators.warm_mass_preconditioner_cache``. Any new traced entry point
    that solves must warm first.

    CAVEAT ON THE EVIDENCE: the mass A/B covers h = 8..20 and p = 2..5, but the
    effect on ``L_k`` was measured at n=12, p=3 only. The overnight sweep in
    ``outputs/diag_newstack/`` extends that to n = 8..32 and p = 2..5.
    """
    # DIAGNOSTIC override, so the swap can be MEASURED without editing code:
    # MRX_MASS_KIND=block_jacobi flips every mass decision at once, which is
    # what makes an honest A/B possible -- including its effect on L_k, since
    # the mass preconditioner is the weak term's inner inverse.
    kind = os.environ.get("MRX_MASS_KIND", "block_jacobi")
    if kind not in ('raw_kron', 'block_jacobi'):
        raise ValueError(
            f"MRX_MASS_KIND must be 'raw_kron' or 'block_jacobi' (got {kind!r})")
    return MassPreconditionerSpec(kind=kind, surgery_schur=False)


def default_saddle_preconditioner() -> SaddlePointPreconditionerSpec:
    """The k>=1 saddle default, as far as a no-argument function can state it.

    It used to return a bare ``SaddlePointPreconditionerSpec()``, i.e. mass
    ``raw_kron`` and outer ``jacobi`` -- neither of which has been the default
    since 2026-08-22 and 2026-08-24 respectively. The field default
    ``MassPreconditionerSpec.kind = 'raw_kron'`` never moved when
    :func:`default_mass_preconditioner` did, so every bare spec still carries
    it. That made this function a plausible-looking answer to "what is the
    default saddle preconditioner" that disagreed with the real one; audit item
    3.4.

    The authoritative resolver is
    ``operators._materialize_default_saddle_preconditioner``, because the outer
    block depends on whether the atom has been assembled for a given
    ``(k, BC)`` and that needs a sequence. ``outer`` is stated as ``jacobi``
    here for exactly that reason: it is the value the real default falls back
    to, and it upgrades to ``'block'`` whenever the atom is present.

    ``schur.inner`` stays ``raw_kron``: that IS the real default's inner, and
    :func:`~mrx.operators.warm_mass_preconditioner_cache` -- this function's
    only caller -- needs it warmed, because the Schur operator is built before
    the outer branch is taken even when the atom ends up serving the apply.
    """
    return SaddlePointPreconditionerSpec(
        mass=default_mass_preconditioner(),
        schur=SchurPreconditionerSpec(
            inner=MassPreconditionerSpec(kind='raw_kron'),
            outer=MassPreconditionerSpec(kind='jacobi'),
        ),
    )


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
    e = getattr(seq, f"e{k}_dbc" if dirichlet else f"e{k}")
    shapes, mass_1d, lam = _kron_mass_model_1d(seq, k, d_raw=d_raw)

    inv_1d = [tuple(jnp.linalg.inv(m) for m in mass_1d[c])
              for c in range(len(shapes))]
    inv_sqrt_D = [1.0 / lam_c for lam_c in lam]
    starts = [0]
    for sh in shapes:
        starts.append(starts[-1] + int(np.prod(sh)))

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


def _kron_mass_model_1d(seq, k: int, d_raw=None):
    """1-D factors of the Kronecker model of ``M_k``::

        M_k  ~  (+)_c  Lam_c (A^c_r x A^c_t x A^c_z) Lam_c

    with *unweighted* 1-D masses (degree ``p`` on primal axes, ``p-1`` on each
    differentiated axis) and the diagonal scaling ``Lam_c`` chosen so that the
    model reproduces ``diag(M_k)`` **exactly**: it is the support-averaged
    metric weight, ``sqrt(diag(M_k)_c / diag(A^c_r x A^c_t x A^c_z))``.

    This is the forward half of :func:`build_mass_raw_kron_factors` -- that one
    inverts the 1-D masses and stores ``1/Lam`` -- and it is also the mass model
    the weak-term diagonal builds on. Measured ``||M~ - M||_F / ||M||_F`` on a
    spline toroid: ``3.1e-2`` at k=1, ``7e-3`` at k=2 and k=3.

    Returns ``(shapes, mass_1d, lam)``: the raw block shapes, the three 1-D
    masses per component, and the 3-D scaling per component.
    """
    from mrx.local_assembly import build_mass_diagonal  # noqa: PLC0415

    form = getattr(seq, f"basis_{k}")
    shapes = [tuple(int(s) for s in sh) for sh in form.shape]

    if d_raw is None:
        d_raw = build_mass_diagonal(seq, k)
    d_raw = jnp.asarray(d_raw)

    primal = (seq.basis_r_jk, seq.basis_t_jk, seq.basis_z_jk)
    deriv = (seq.d_basis_r_jk, seq.d_basis_t_jk, seq.d_basis_z_jk)
    quad_w = (seq.quad.w_x, seq.quad.w_y, seq.quad.w_z)

    mass_1d, lam, start = [], [], 0
    for c, shape in enumerate(shapes):
        diff = _raw_kron_diff_flags(k, c)
        bases = tuple(deriv[a] if diff[a] else primal[a] for a in range(3))
        m1 = tuple(_assemble_weighted_1d_mass(bases[a], quad_w[a]) for a in range(3))
        for a in range(3):
            if int(m1[a].shape[0]) != shape[a]:
                raise ValueError(
                    f"kron mass model k={k} component {c} axis {a}: 1D mass is "
                    f"{m1[a].shape[0]} but the raw block axis is {shape[a]}"
                )
        kron_diag = jnp.einsum('i,j,l->ijl', jnp.diag(m1[0]),
                               jnp.diag(m1[1]), jnp.diag(m1[2]))
        size = int(np.prod(shape))
        mass_1d.append(m1)
        lam.append(jnp.sqrt(d_raw[start:start + size].reshape(shape) / kron_diag))
        start += size
    return shapes, tuple(mass_1d), tuple(lam)


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


# --------------------------------------------------------------------------- #
# Closed-form diagonal of the WEAK term of the Hodge Laplacian                 #
# --------------------------------------------------------------------------- #
#
#   L_k = S_k + W_k ,   W_k = D_{k-1} B_{k-1} D_{k-1}^T ,   D_l = E_k M_k G_l E_l^T
#
# with ``B`` the raw_kron mass preconditioner standing in for ``M_{k-1}^{-1}``.
# ``diag(S_k)`` is already closed form (:func:`mrx.local_assembly.
# build_stiffness_diagonal`); this is the other half, and it is what forced
# k>=1 Jacobi to probe the Laplacian at O(N) applies.
#
# Substituting ``B = (E+)^T K E+`` and folding the two pseudoinverses into the
# extraction projector ``Pi = E_l^T (E_l E_l^T)^{-1} E_l`` gives
#
#     W_k = E_k [ M_k G_l Pi ] K [ Pi G_l^T M_k ] E_k^T
#
# in which EVERY factor is a sum of Kronecker products.  A Kronecker product's
# diagonal is the outer product of its 1-D diagonals, so the whole thing
# collapses to a handful of small 1-D matrix chains plus one outer product per
# term pair -- O(N), against the O(N) full applies a probe needs.
#
#   M_k, M_{k-1}  ->  the raw_kron model ``Lam (x)_a A_a Lam`` of
#            :func:`_kron_mass_model_1d`, in all three places -- so
#            ``M_{k-1}^{-1}`` here is literally the production mass
#            preconditioner, not a second model built alongside it.
#   G_l  ->  exact, one Kronecker term per (out component, in component) pair.
#   Pi   ->  exact, see :func:`_extraction_projector_kron_terms`.
#
# Note ``K = Sig (x)_a Cinv_a Sig`` with ``Sig = Lam_l^-1`` is EXACTLY the
# inverse of the raw_kron model of ``M_{k-1}``, so "raw_kron preconditioner" and
# "Kronecker model of the mass, inverted" are the same object here.
#
# The one wrinkle is that ``Lam (x)_a A_a Lam`` is a Kronecker product SANDWICHED
# by a non-separable diagonal, and a diagonal does not push through a Kronecker
# factorization.  Of the six diagonals in ``M_k G M_{k-1}^-1 G^T M_k`` -- two per
# mass -- the two OUTERMOST are free: they multiply the finished diagonal
# pointwise, so they are kept EXACT and cost nothing.  The remaining four are
# interior (they land between a 1-D mass and the incidence) and are split rank-1
# in closed form by :func:`_rank1_diagonal_split`: no iteration, no fit, and no
# extra term pairs.  Measured against the same expansion with every ``Lam`` kept
# exact, the split costs a 2.4e-2 / 5.3e-3 / 2.5e-3 median at k=1/2/3.
#
# Do NOT replace M by its diagonal instead: that discards the mass coupling
# entirely and was rejected on 2026-08-18.
#
# Pi is NOT optional and NOT diagonal.  Both cheap surrogates were measured and
# both fail: masking (Pi ~ the bulk indicator) and the exact leverage diagonal
# (Pi ~ diag(Pi)) each leave ~90% error on the near-axis rows -- p99 8.9e-1 and
# 9.1e-1 against 3.1e-2 for the exact expansion, on a 10x8x6 toroid at k=1.

# Out component -> ((in component, differentiated axis, sign), ...) for G_l.
# Read off ``_apply_incidence_mf``: grad = (d_r, d_t, d_z);
# curl P = -d_z b + d_t c, Q = d_z a - d_r c, R = -d_t a + d_r b;
# div = d_r a + d_t b + d_z c.
_INCIDENCE_KRON_TERMS = {
    0: {0: ((0, 0, 1.0),),
        1: ((0, 1, 1.0),),
        2: ((0, 2, 1.0),)},
    1: {0: ((1, 2, -1.0), (2, 1, 1.0)),
        1: ((0, 2, 1.0), (2, 0, -1.0)),
        2: ((0, 1, -1.0), (1, 0, 1.0))},
    2: {0: ((0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0))},
}


def _raw_block_starts(shapes):
    starts = [0]
    for shape in shapes:
        starts.append(starts[-1] + int(np.prod(shape)))
    return np.asarray(starts)


def _decode_raw_indices(flat, shapes, starts):
    """Flat raw DOF indices -> (component, i_r, i_t, i_z)."""
    flat = np.asarray(flat)
    comp = np.searchsorted(starts, flat, side='right') - 1
    loc = flat - starts[comp]
    shape = np.asarray(shapes)[comp]
    nt, nz = shape[:, 1], shape[:, 2]
    return comp, loc // (nt * nz), (loc // nz) % nt, loc % nz


def _extraction_projector_kron_terms(e, shapes, *, tol=1e-10):
    """Exact Kronecker expansion of ``Pi = E^T (E E^T)^{-1} E``.

    ``Pi`` is the identity on the bulk raw DOFs, zero on the raw DOFs that
    ``E`` drops, and a rank-deficient projector on the POLAR RING -- and the
    ring is only ``ring_depth`` radial indices thick, sits at a single zeta
    index per coupled row, and is the same block in every zeta plane.  So

        Pi = (+)_c diag(chi^c_r) x I_t x I_z  +  sum_{c,c'} sum_j F_r x F_t x I_z

    with ``chi^c_r`` the bulk radial indicator.  The ring blocks are split by an
    SVD in the ``(i_r, j_r)`` vs ``(i_t, j_t)`` grouping; that split is EXACT
    and its rank is at most ``ring_depth^2 <= 4`` per component pair, because
    the ring is radially thin.  Hence a handful of terms, not O(n_t).

    Every structural assumption is checked against the actual ``E`` and raises
    rather than silently degrading -- an unnoticed failure here is a ~90% error
    on the near-axis rows, which is exactly where a Jacobi diagonal matters.

    Returns a list of ``(src_component, dst_component, (F_r, F_t, F_z))``.
    """
    rows = np.asarray(e.rows)
    cols = np.asarray(e.cols)
    vals = np.asarray(e.vals)
    n_ext = int(e.forward_shape[0])
    n_raw = int(e.forward_shape[1])
    starts = _raw_block_starts(shapes)

    counts = np.bincount(rows, minlength=n_ext)
    coupled_rows = np.flatnonzero(counts > 1)
    ring_cols = np.unique(cols[np.isin(rows, coupled_rows)])
    touched = np.unique(cols)
    bulk_cols = np.setdiff1d(touched, ring_cols)

    terms = []

    # --- identity part: the bulk radial indicator ---------------------------
    # Pi is exactly the identity on a bulk column and exactly zero on a dropped
    # one, so this part is a 0/1 diagonal; it is separable only if it is
    # constant over (theta, zeta) at fixed (component, i_r).
    if np.bincount(cols[np.isin(cols, bulk_cols)]).max(initial=0) > 1:
        raise ValueError(
            "extraction projector: a bulk raw DOF is shared by more than one "
            "extracted row, so Pi is not the identity there")
    mask = np.zeros(n_raw)
    mask[bulk_cols] = 1.0
    for c, shape in enumerate(shapes):
        block = mask[starts[c]:starts[c + 1]].reshape(shape)
        chi = block[:, 0, 0]
        if not np.array_equal(block, np.broadcast_to(chi[:, None, None], shape)):
            raise ValueError(
                f"extraction projector component {c}: the bulk indicator is not "
                "a radial slab, so the identity part of Pi is not separable")
        terms.append((c, c, (np.diag(chi), np.eye(shape[1]), np.eye(shape[2]))))

    if coupled_rows.size == 0:
        return terms

    # --- ring part ----------------------------------------------------------
    coupled, gram_inv, _ = _extraction_gram_inverse(e)
    coupled = np.asarray(coupled)
    gram_inv = np.asarray(gram_inv)
    pos = {int(r): t for t, r in enumerate(coupled)}

    _, _, _, ring_iz = _decode_raw_indices(ring_cols, shapes, starts)
    row_zeta = {}
    for i in coupled_rows:
        _, _, _, iz = _decode_raw_indices(cols[rows == i], shapes, starts)
        if len(set(iz.tolist())) != 1:
            raise ValueError(
                f"extraction projector: coupled row {int(i)} spans more than one "
                "zeta index, so Pi is not block diagonal in zeta")
        row_zeta[int(i)] = int(iz[0])

    # Ring index space at one zeta plane: (component, ring radial index, theta).
    ring_comp, ring_ir, _, _ = _decode_raw_indices(ring_cols, shapes, starts)
    ring_radial = {c: np.unique(ring_ir[ring_comp == c])
                   for c in np.unique(ring_comp)}

    def plane_block(z0):
        """Dense ``Pi`` restricted to the ring at zeta index ``z0``."""
        plane_rows = [i for i in coupled_rows if row_zeta[int(i)] == z0]
        offsets, size = {}, 0
        for c, radial in ring_radial.items():
            offsets[c] = size
            size += len(radial) * shapes[c][1]
        cmat = np.zeros((len(plane_rows), size))
        for n, i in enumerate(plane_rows):
            sel = rows == i
            comp, ir, it, _ = _decode_raw_indices(cols[sel], shapes, starts)
            for c, r, t, v in zip(comp, ir, it, vals[sel]):
                radial = ring_radial[int(c)]
                loc = int(np.searchsorted(radial, r)) * shapes[int(c)][1] + int(t)
                cmat[n, offsets[int(c)] + loc] += float(v)
        idx = np.array([pos[int(i)] for i in plane_rows])
        return cmat.T @ gram_inv[np.ix_(idx, idx)] @ cmat, offsets

    zetas = sorted(set(row_zeta.values()))
    block, offsets = plane_block(zetas[0])
    for z0 in zetas[1:]:
        other, _ = plane_block(z0)
        if not np.allclose(other, block, atol=1e-11, rtol=1e-9):
            raise ValueError(
                "extraction projector: the polar ring block differs between "
                "zeta planes, so Pi does not factor as (ring block) x I_z")

    for c1, radial1 in ring_radial.items():
        for c2, radial2 in ring_radial.items():
            c1, c2 = int(c1), int(c2)
            nt1, nt2 = shapes[c1][1], shapes[c2][1]
            if shapes[c1][2] != shapes[c2][2]:
                raise ValueError(
                    f"extraction projector: components {c1} and {c2} are coupled "
                    "by the polar ring but have different zeta dimensions")
            sub = block[offsets[c1]:offsets[c1] + len(radial1) * nt1,
                        offsets[c2]:offsets[c2] + len(radial2) * nt2]
            sub = sub.reshape(len(radial1), nt1, len(radial2), nt2)
            flat = sub.transpose(0, 2, 1, 3).reshape(
                len(radial1) * len(radial2), nt1 * nt2)
            u, s, vt = np.linalg.svd(flat, full_matrices=False)
            keep = s > max(tol * s[0], 0.0) if s.size else s
            for j in np.flatnonzero(keep):
                root = np.sqrt(s[j])
                f_r = np.zeros((shapes[c1][0], shapes[c2][0]))
                f_r[np.ix_(radial1, radial2)] = (
                    root * u[:, j]).reshape(len(radial1), len(radial2))
                f_t = (root * vt[j]).reshape(nt1, nt2)
                terms.append((c1, c2, (f_r, f_t, np.eye(shapes[c1][2]))))
    return terms


def _rank1_diagonal_split(tensor, *, mode: str = "geometric"):
    """Closed-form rank-1 separable split of a positive 3-D diagonal.

    Returns ``(f_r, f_t, f_z)`` with ``T ~ f_r x f_t x f_z``.  Each factor is
    the average of ``T`` over the other two axes, rescaled by the global mean so
    the product reproduces ``T`` EXACTLY whenever ``T`` really is separable.
    Three contractions, deterministic, symmetric in the axes -- no iteration and
    no fit to babysit.

    ``mode='geometric'`` averages ``log T`` -- the multiplicative version, and
    the default because ``T`` is a multiplicative scaling: against the unsplit
    raw_kron model of the weak term on a spline toroid it is 2-20x closer than
    the arithmetic split at every k (median 2.4e-2 / 5.3e-3 / 2.5e-3 at
    k=1/2/3, against 4.9e-2 / 4.1e-2 / 6.0e-2).  ``mode='arithmetic'`` averages
    ``T`` itself.  Both are exact on a separable tensor and differ only in how
    they spread a non-separable residual.
    """
    tensor = np.asarray(tensor)
    if mode == "geometric":
        work = np.log(tensor)
    elif mode == "arithmetic":
        work = tensor
    else:
        raise ValueError(f"unknown rank-1 split mode {mode!r}")
    mean = work.mean()
    factors = []
    for a in range(3):
        others = tuple(b for b in range(3) if b != a)
        factors.append(work.mean(axis=others) - (2.0 / 3.0) * mean
                       if mode == "geometric" else
                       work.mean(axis=others) / mean ** (2.0 / 3.0))
    return [np.exp(f) for f in factors] if mode == "geometric" else factors


def _rank1_split_residual(tensor, factors):
    """``T / (f_r x f_t x f_z)`` -- what :func:`_rank1_diagonal_split` missed.

    Positive, and identically 1 wherever ``T`` really is separable, so every
    correction built from it is a no-op on a separable weight.
    """
    return np.asarray(tensor) / np.einsum('i,j,l->ijl', *[np.asarray(f)
                                                          for f in factors])


def _axis_resample(m: int, n: int) -> Optional[np.ndarray]:
    """Linear interpolation from an ``m``-point axis onto an ``n``-point one.

    ``None`` for ``m == n`` (identity).  Only ever called with ``|m - n| = 1``:
    differentiating a clamped axis drops one DOF and a periodic axis none, so
    this is the two-point average between a degree-``p`` and a degree-``p-1``
    grid, written generically because which axis is differentiated depends on
    the component pair.
    """
    if m == n:
        return None
    src = np.linspace(0.0, 1.0, m)
    dst = np.linspace(0.0, 1.0, n)
    idx = np.clip(np.searchsorted(src, dst) - 1, 0, m - 2)
    w = (dst - src[idx]) / (src[idx + 1] - src[idx])
    out = np.zeros((n, m))
    out[np.arange(n), idx] = 1.0 - w
    out[np.arange(n), idx + 1] = w
    return out


def _transfer_tensor(tensor, shape_out) -> np.ndarray:
    """Resample a positive 3-D tensor onto ``shape_out``, axis by axis.

    Multiplicative (works on ``log T``), so the result stays positive and a
    constant tensor transfers to itself.
    """
    work = np.log(np.asarray(tensor))
    for a in range(3):
        T = _axis_resample(work.shape[a], int(shape_out[a]))
        if T is not None:
            work = np.moveaxis(np.tensordot(T, work, axes=([1], [a])), 0, a)
    return np.exp(work)


def _axis_index_delta(n: int, typ: str) -> np.ndarray:
    """``j - i`` on an ``n``-point axis, wrapped to the short way round if the
    axis is periodic.  Only ever multiplied against a banded 1-D mass, so the
    entries that survive are ``|j - i| <= p``.
    """
    d = np.arange(n)[None, :] - np.arange(n)[:, None]
    if typ == "periodic":
        d = (d + n // 2) % n - n // 2
    return d.astype(float)


def _log_index_gradient(tensor, types) -> list:
    """``d(log T) / d(index)`` per axis: central differences on the DOF grid,
    wrapping on periodic axes and one-sided at a clamped end.
    """
    work = np.log(np.asarray(tensor))
    out = []
    for a in range(3):
        if types[a] == "periodic":
            out.append(0.5 * (np.roll(work, -1, axis=a) - np.roll(work, 1, axis=a)))
        else:
            out.append(np.gradient(work, axis=a))
    return out


def _weak_term_raw_terms(seq, k: int, *, dirichlet: bool):
    """Unscaled Kronecker terms of ``X = A^u [Lam] G Pi [Sig]``.

    Each term is ``(c_u, dst, sign, gf)`` with ``gf`` the three 1-D factors of
    ``G Pi`` -- everything BETWEEN the two inner scalings, which is the part
    that is genuinely separable.  What to do about the scalings in brackets is
    left to the caller: split them rank-1, expand them locally, or keep them
    exact.  ``ctx`` carries the 1-D masses, both scalings and the lower inverse.
    """
    from mrx.operators import _dense_incidence_1d  # noqa: PLC0415

    lower = k - 1
    shapes_u, mass_u, lam_u = _kron_mass_model_1d(seq, k)
    factors = build_mass_raw_kron_factors(seq, lower, dirichlet=dirichlet)
    shapes_l = [tuple(int(s) for s in sh) for sh in factors.shapes]
    inv_l = [tuple(np.asarray(m) for m in factors.inv_1d[c])
             for c in range(len(shapes_l))]
    e_lower = getattr(seq, f"e{lower}_dbc" if dirichlet else f"e{lower}")
    pi_terms = _extraction_projector_kron_terms(e_lower, shapes_l)
    types = seq.basis_0.types

    terms = []
    for c_u, contributions in _INCIDENCE_KRON_TERMS[lower].items():
        for (c_g, axis, sign) in contributions:
            g = np.asarray(_dense_incidence_1d(shapes_l[c_g][axis], types[axis]))
            for a in range(3):
                if a != axis and shapes_u[c_u][a] != shapes_l[c_g][a]:
                    raise ValueError(
                        f"weak-term k={k}: undifferentiated axis {a} has size "
                        f"{shapes_l[c_g][a]} below and {shapes_u[c_u][a]} above")
            if g.shape != (shapes_u[c_u][axis], shapes_l[c_g][axis]):
                raise ValueError(
                    f"weak-term k={k}: incidence on axis {axis} has shape "
                    f"{g.shape}, expected "
                    f"{(shapes_u[c_u][axis], shapes_l[c_g][axis])}")
            for (src, dst, f_axes) in pi_terms:
                if src != c_g:
                    continue
                gf = [np.asarray(g @ f_axes[a]) if a == axis
                      else np.asarray(f_axes[a]) for a in range(3)]
                for a in range(3):
                    if gf[a].shape != (shapes_u[c_u][a], shapes_l[dst][a]):
                        raise ValueError(
                            f"weak-term k={k}: axis {a} factor has shape "
                            f"{gf[a].shape}, expected "
                            f"{(shapes_u[c_u][a], shapes_l[dst][a])}")
                terms.append((c_u, dst, sign, gf))

    ctx = {"shapes_u": shapes_u, "mass_u": mass_u, "lam_u": lam_u,
           "shapes_l": shapes_l, "inv_l": inv_l,
           "sigma": factors.inv_sqrt_D, "types": types}
    return terms, ctx


def _group_terms(terms) -> dict:
    """Terms bucketed by ``(upper component, Pi destination)``."""
    groups: dict = {}
    for (c_u, dst, sign, gf) in terms:
        groups.setdefault((c_u, dst), []).append((sign, gf))
    return groups


def _weak_term_kron_terms(seq, k: int, *, dirichlet: bool, split: str = "geometric",
                          rescale: str = "none"):
    """Kronecker terms of ``X = A^u Lam~^u G_{k-1} Pi Sig~``, grouped by the
    pair ``(upper component, lower component)`` that ``M_{k-1}^{-1}`` couples.

    Each entry is ``(sign, L, L_inv)`` with ``L`` the three 1-D factors and
    ``L_inv = L (A^l)^-1`` pre-multiplied, so a term PAIR costs one row-wise dot
    per axis instead of a matrix chain.

    Also returns a per-group multiplicative CORRECTION, see ``rescale`` in
    :func:`build_weak_term_raw_diagonal`; ``'none'`` gives all-ones.
    """
    terms, ctx = _weak_term_raw_terms(seq, k, dirichlet=dirichlet)
    shapes_u, mass_u, lam_u = ctx["shapes_u"], ctx["mass_u"], ctx["lam_u"]
    shapes_l, inv_l, sigma = ctx["shapes_l"], ctx["inv_l"], ctx["sigma"]

    # The two INNER diagonal scalings, split rank-1 so they fold into the 1-D
    # factors. The upper Lam appears once more on the OUTSIDE, where it is kept
    # exact (see build_weak_term_raw_diagonal).
    lam_split = [_rank1_diagonal_split(lam_u[c], mode=split)
                 for c in range(len(shapes_u))]
    sigma_split = [_rank1_diagonal_split(sigma[c], mode=split)
                   for c in range(len(shapes_l))]

    # Leading-order repair of the two inner splits, off by default. Only the
    # residuals are needed; see build_weak_term_raw_diagonal for the counting.
    if rescale not in ("none", "upper", "both"):
        raise ValueError(f"unknown weak-term rescale mode {rescale!r}")
    resid_u = [_rank1_split_residual(lam_u[c], lam_split[c]) ** 2
               for c in range(len(shapes_u))] if rescale != "none" else None
    resid_l = [_rank1_split_residual(sigma[c], sigma_split[c]) ** 2
               for c in range(len(shapes_l))] if rescale == "both" else None

    groups: dict = {}
    for (c_u, dst, sign, gf) in terms:
        left = [((np.asarray(mass_u[c_u][a]) * lam_split[c_u][a][None, :])
                 @ gf[a]) * sigma_split[dst][a][None, :] for a in range(3)]
        groups.setdefault((c_u, dst), []).append(
            (sign, left, [left[a] @ inv_l[dst][a] for a in range(3)]))

    corr = {}
    for (c_u, dst) in groups:
        if rescale == "none":
            corr[(c_u, dst)] = None
            continue
        factor = resid_u[c_u]
        if resid_l is not None:
            factor = factor * _transfer_tensor(resid_l[dst], shapes_u[c_u])
        corr[(c_u, dst)] = factor
    return groups, shapes_u, lam_u, corr


def _weak_term_taylor_parts(seq, k: int, *, dirichlet: bool,
                            split: str = "geometric"):
    """``split='taylor1'``: expand the inner ``Lam`` LOCALLY instead of fitting
    it globally.

    The rank-1 split is a global fit to a quantity that is only ever sampled
    locally: ``A^u`` has bandwidth ``p``, so ``Lam_j`` enters only for ``j``
    within a few knots of the row ``i``.  Expand about the row instead::

        Lam_j ~ Lam_i (1 + g_i . (j - i)) ,   g = grad log Lam at i

    ``A_ij (j_a - i_a)`` is a first-moment 1-D mass -- still one small dense
    matrix per axis, still separable -- and ``g_a(i)`` is evaluated at the ROW,
    so like the outer ``Lam^2`` it multiplies pointwise and costs nothing.  Each
    term therefore splits into four (no moment, plus one per axis) and the pair
    sum is bucketed by which moments the two sides carry.

    The point of it: the error is now ``O(h^2 |grad^2 log Lam|)``, a local
    SMOOTHNESS assumption that refines away, instead of a global SEPARABILITY
    assumption that does not -- and the measured max error of the rank-1 split
    plateaus under refinement, which is what a global fit residual looks like.
    Positivity survives because the correction stays inside ``X``: the result is
    still ``diag(X~ B X~^T)`` for a genuine ``X~``.

    Only the upper ``Lam`` is expanded.  ``Sig`` sits against the DENSE (though
    exponentially decaying) ``(A^l)^-1``, where the locality argument is much
    weaker, so it keeps the rank-1 split given by ``split``.
    """
    terms, ctx = _weak_term_raw_terms(seq, k, dirichlet=dirichlet)
    shapes_u, mass_u, lam_u = ctx["shapes_u"], ctx["mass_u"], ctx["lam_u"]
    shapes_l, inv_l, sigma = ctx["shapes_l"], ctx["inv_l"], ctx["sigma"]
    types = ctx["types"]

    sigma_split = [_rank1_diagonal_split(sigma[c], mode=split)
                   for c in range(len(shapes_l))]
    grads = [_log_index_gradient(lam_u[c], types) for c in range(len(shapes_u))]
    deltas = [[_axis_index_delta(shapes_u[c][a], types[a]) for a in range(3)]
              for c in range(len(shapes_u))]

    entries: dict = {}
    for (c_u, dst, sign, gf) in terms:
        for v in range(4):  # 0 = plain mass, v = 1..3 = first moment on axis v-1
            fac = []
            for a in range(3):
                m = np.asarray(mass_u[c_u][a])
                if v == a + 1:
                    m = m * deltas[c_u][a]
                fac.append((m @ gf[a]) * sigma_split[dst][a][None, :])
            entries.setdefault((c_u, dst), []).append(
                (sign, v, fac, [fac[a] @ inv_l[dst][a] for a in range(3)]))

    parts = [np.zeros(shape) for shape in shapes_u]
    n_pairs = 0
    for (c_u, dst), group in entries.items():
        blocks: dict = {}
        for i, (sign_i, v_i, _, linv_i) in enumerate(group):
            for j in range(i, len(group)):
                sign_j, v_j, l_j, _ = group[j]
                z = [np.einsum('mn,mn->m', linv_i[a], l_j[a]) for a in range(3)]
                weight = (sign_i * sign_j) * (1.0 if i == j else 2.0)
                key = (min(v_i, v_j), max(v_i, v_j))
                block = weight * np.einsum('i,j,l->ijl', *z)
                blocks[key] = blocks.get(key, 0.0) + block
                n_pairs += 1
        total = np.zeros(shapes_u[c_u])
        for (v_i, v_j), block in blocks.items():
            coef = 1.0
            if v_i:
                coef = coef * grads[c_u][v_i - 1]
            if v_j:
                coef = coef * grads[c_u][v_j - 1]
            total += coef * block
        # The Taylor prefactor: Lam_i factors out of the row on both sides. The
        # OUTER Lam^2 is applied by the caller, so the model carries Lam^4 in
        # all -- the same power the split form carries when Lam is constant.
        parts[c_u] += np.asarray(lam_u[c_u]) ** 2 * total
    return parts, lam_u, n_pairs


def _weak_term_exact_parts(seq, k: int, *, dirichlet: bool):
    """``split='exact'``: the same expansion with BOTH inner scalings kept
    exact -- the oracle that separates the two error sources.

    The closed form carries two independent approximations: the mass model
    (``M~`` vs ``M``, Kronecker 1-D masses) and the rank-1 split of the inner
    scalings.  Against the operator probe they are measured together.  This
    path removes the second one, so ``probe - exact`` is the mass-model error
    and ``exact - closed`` is the split error.

    There is no cheap way to do it -- an exact non-separable diagonal between
    two Kronecker factors is precisely what does not factorize, which is why
    the split exists at all.  So this forms ``X`` DENSELY per group, one
    ``(N_u x N_l)`` matrix, and is a diagnostic at A/B resolution only, never
    production.  ``MRX_WEAK_EXACT_MAXDIM`` (default 2e7 entries, ~160 MB) caps
    it rather than letting it OOM a node.
    """
    terms, ctx = _weak_term_raw_terms(seq, k, dirichlet=dirichlet)
    shapes_u, mass_u, lam_u = ctx["shapes_u"], ctx["mass_u"], ctx["lam_u"]
    shapes_l, inv_l, sigma = ctx["shapes_l"], ctx["inv_l"], ctx["sigma"]

    max_dense = float(os.environ.get("MRX_WEAK_EXACT_MAXDIM", 2e7))
    parts = [np.zeros(shape) for shape in shapes_u]
    a_kron: dict = {}
    for (c_u, dst), group in _group_terms(terms).items():
        n_u = int(np.prod(shapes_u[c_u]))
        n_l = int(np.prod(shapes_l[dst]))
        if n_u * n_l > max_dense or n_l * n_l > max_dense:
            raise MemoryError(
                f"weak-term split='exact' k={k} needs a dense {n_u}x{n_l} "
                f"block ({n_u * n_l:.3g} entries) over the "
                f"{max_dense:.3g} cap. It is a diagnostic oracle; run it at "
                "A/B resolution or raise MRX_WEAK_EXACT_MAXDIM.")
        # X = (x)A^u . D_Lam . [sum_t sign_t (x)gf_t] . D_Sig -- the bracket is
        # the only part that differs between terms, so it is summed first and
        # the two dense products are paid once per group.
        y = np.zeros((n_u, n_l))
        for (sign, gf) in group:
            y += sign * np.kron(np.kron(gf[0], gf[1]), gf[2])
        if c_u not in a_kron:
            a_kron[c_u] = np.kron(np.kron(np.asarray(mass_u[c_u][0]),
                                          np.asarray(mass_u[c_u][1])),
                                  np.asarray(mass_u[c_u][2]))
        x = a_kron[c_u] @ (np.asarray(lam_u[c_u]).reshape(-1)[:, None] * y)
        x *= np.asarray(sigma[dst]).reshape(-1)[None, :]
        v = np.kron(np.kron(inv_l[dst][0], inv_l[dst][1]), inv_l[dst][2])
        parts[c_u] += ((x @ v) * x).sum(axis=1).reshape(shapes_u[c_u])
    return parts, lam_u


def _greville_transfer_v3_to_v0(seq, geometry, *, bandwidth=None,
                                metric_free=False):
    """Sparse-ish transfer ``Pi: V_3 -> V_0`` representing ``star phi_i``.

    ``star`` of the k=3 basis function is the SCALAR ``phi_i / J`` (the k=3 mass
    weight is ``1/J``, which pins the convention).  Rather than differentiate
    that -- which needs a derivative of an already-differentiated spline and a
    ``dJ`` pass over the map -- represent it in ``V_0`` first, where the basis
    carries no derivative at all, and differentiate THERE.  The gradient is then
    an ordinary k=0 stiffness energy.

    The representation is Greville collocation: match ``phi_i / J`` at the k=0
    Greville abscissae, i.e. ``Pi = (x)A_a^-1 . diag(1/J_g) . (x)B_a``, with
    ``A_a`` the 1-D collocation matrix of the degree-``p`` basis and ``B_a`` the
    degree-``p-1`` (k=3) basis sampled at the same points.  ``A_a`` is
    metric-free, so its inverse decays at a rate that depends only on ``p``:
    ``bandwidth`` truncates it to make ``Pi`` genuinely local, and ``None``
    keeps it exact, which measures the ceiling before locality costs anything.

    Note ``J`` is needed only at the ``n_0`` Greville points, not on the
    quadrature grid, and only ``det DF`` -- no second derivatives anywhere.
    """
    lam, dlam = seq.basis_0.Λ, seq.basis_0.dΛ
    pts = [np.asarray(lam[a].greville_points()) for a in range(3)]

    # Greville abscissae of a CLAMPED basis include the endpoints, and
    # det(DF) = 0 at the outer knot of a spline map -- so 1/J is infinite there.
    # Quadrature points never land on the boundary, which is exactly why this
    # trap does not show up in any of the assembly paths. Pull the sample points
    # in by eps; the collocation matrix is evaluated at the SAME points, so the
    # interpolation stays consistent (and Schoenberg-Whitney still holds).
    eps = 1e-8
    for a in range(3):
        if lam[a].type != "periodic":
            pts[a] = np.clip(pts[a], eps, 1.0 - eps)

    b_g, a_inv = [], []
    for a in range(3):
        tbl = jax.vmap(lambda x, a=a: jax.vmap(
            lambda i, x=x, a=a: jnp.sum(dlam[a](x, i)))(dlam[a].ns))(
                jnp.asarray(pts[a]))
        b_g.append(np.asarray(tbl))                       # (n_g_a, n_3_a)
        coll = np.asarray(lam[a].collocation_matrix(jnp.asarray(pts[a])))
        inv = np.linalg.inv(coll)
        if bandwidth is not None:
            n = inv.shape[0]
            off = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
            if lam[a].type == "periodic":
                off = np.minimum(off, n - off)
            inv = np.where(off <= bandwidth, inv, 0.0)
        a_inv.append(inv)

    if metric_free:
        # The L2 pairing <u, s> between a 0-form and a 3-form's physical proxy
        # carries no metric at all: the 1/J in the proxy cancels the J in the
        # 0-form measure. So interpolating the k=3 COEFFICIENT function is the
        # metric-free transfer, and dividing by J here -- as the original
        # version did -- inserts a factor the correct pairing cancels. That
        # spurious 1/J is what produced the Jacobian-shaped error peaking on
        # the ring next to the axis, where the cells collapse.
        #
        # Without it the whole transfer is a pure Kronecker product of three
        # 1-D matrices, needs no map evaluation whatsoever, and is banded as
        # soon as the collocation inverse is truncated.
        return np.kron(np.kron(a_inv[0] @ b_g[0], a_inv[1] @ b_g[1]),
                       a_inv[2] @ b_g[2])

    # J at the tensor Greville grid: n_0 map evaluations, against n_q for the
    # quadrature grid -- and jacfwd only, no second derivative.
    grid = np.stack(np.meshgrid(*pts, indexing="ij"), axis=-1).reshape(-1, 3)
    def jdet(x):
        return jnp.linalg.det(jax.jacfwd(geometry.map)(x))
    jac_g = np.asarray(jax.lax.map(jdet, jnp.asarray(grid),
                                   batch_size=mrx.MAP_BATCH_SIZE_INNER or 256))
    if not np.isfinite(jac_g).all() or np.abs(jac_g).min() == 0.0:
        raise ValueError(
            f"det(DF) at the Greville grid is degenerate: finite="
            f"{np.isfinite(jac_g).all()} min|J|={np.abs(jac_g).min():.3e}. "
            "The star divides by it, so this has to be caught here rather "
            "than surface as a NaN diagonal.")

    kron_b = np.kron(np.kron(b_g[0], b_g[1]), b_g[2])     # (n_g, n_3)
    work = kron_b / jac_g[:, None]
    shape_g = tuple(len(p_) for p_ in pts)
    work = work.reshape(*shape_g, -1)
    for a in range(3):
        work = np.moveaxis(np.tensordot(a_inv[a], work, axes=([1], [a])), 0, a)
    return work.reshape(int(np.prod(shape_g)), -1)         # (n_0, n_3)


def _metric_free_star_1d(seq, axis):
    """1-D metric-free, local discrete Hodge star ``V_3 axis -> V_0 axis``.

    The IGA/FEEC construction, on DEGREES OF FREEDOM rather than on function
    values.  The degree-``p-1`` (k=3) basis is dual to HISTOPOLATION over the
    Greville spans -- its DOF is the integral over a cell -- and the degree-``p``
    (k=0) basis is dual to INTERPOLATION at the Greville points -- its DOF is a
    point value.  So the star between them is "cell integral -> point value",
    i.e. divide by the cell measure and average the cells meeting at a point::

        d_j = (c_{j-1} + c_j) / (h_{j-1} + h_j)

    with ``h`` the Greville-span widths, one-sided at a clamped end.  Everything
    here comes from the KNOT VECTOR: the operator is bandwidth-1, exactly local,
    and carries no geometry at all -- the metric stays in the mass matrices,
    where FEEC puts it.

    This is the thing the previous two attempts were not.  Both of those tried
    to represent the k=3 basis FUNCTION in ``V_0`` (pointwise, by collocation),
    which is an O(1) request: a basis function varies on the scale of one
    element, so approximating its shape in a different space on the same mesh
    does not converge in ``h``.  Mapping DOF vectors asks nothing of the shapes
    and reproduces constants exactly.
    """
    lam, dlam = seq.basis_0.Λ[axis], seq.basis_0.dΛ[axis]
    typ = lam.type
    grev = np.asarray(lam.greville_points())
    n0, n3 = int(lam.n), int(dlam.n)

    if typ == "periodic":
        h = np.diff(np.concatenate([grev, [grev[0] + 1.0]]))
    else:
        h = np.diff(grev)
    h = np.abs(h)
    if h.shape[0] != n3:
        raise ValueError(
            f"axis {axis}: {h.shape[0]} Greville spans but {n3} k=3 DOFs; the "
            "histopolation duality this star relies on does not hold here")

    star = np.zeros((n0, n3))
    for j in range(n0):
        left = (j - 1) % n3 if typ == "periodic" else j - 1
        right = j % n3 if typ == "periodic" else j
        cells = [c for c in (left, right) if 0 <= c < n3]
        denom = sum(h[c] for c in cells)
        for c in cells:
            star[j, c] = 1.0 / denom
    return star


def _raw_grad_incidence(seq):
    """Raw ``G_0: V_0 -> V_1`` as one dense matrix, component-blocked to match
    the flat layout ``assemble_m1_local`` uses."""
    from mrx.operators import _dense_incidence_1d  # noqa: PLC0415

    types = seq.basis_0.types
    shape0 = tuple(int(s) for s in seq.basis_0.shape[0])
    blocks = []
    for c in range(3):
        factors = []
        for a in range(3):
            if a == c:
                factors.append(np.asarray(
                    _dense_incidence_1d(shape0[a], types[a])))
            else:
                factors.append(np.eye(shape0[a]))
        blocks.append(np.kron(np.kron(factors[0], factors[1]), factors[2]))
    return np.concatenate(blocks, axis=0)


def build_transfer_weak_diagonal(seq, k: int, *, dirichlet: bool,
                                 bandwidth=None, geometry=None):
    """``diag(W_3)`` as the k=0 stiffness energy of the transferred basis.

    ``diag(W_k)_i = ||delta_h phi_i||^2``, and at k=3 ``delta = star d star``
    with ``star phi_i`` a scalar.  Transfer that scalar into ``V_0``
    (:func:`_greville_transfer_v3_to_v0`) and the remaining ``d`` is the
    ordinary gradient of a degree-``p`` spline::

        diag(W_3)_i ~ || grad (Pi e_i) ||^2 = (Pi^T G_0^T M_1 G_0 Pi)_ii

    which is exactly what :func:`diag_EGtMGEt_direct` computes.  No derivative
    of the k=3 basis, no ``dJ``, and every factor is a standard object.

    **BC pairing.** Hodge duality flips essential and natural conditions, so the
    partner of k=3 DIRICHLET is k=0 FREE and vice versa.  This builds the raw
    (free) k=0 energy, so it is the k=3 ``dirichlet=True`` case that is clean;
    the k=3 free case additionally needs the boundary trace that the k=2
    Dirichlet condition otherwise kills.  Measured directly: on a toroid the
    outer-ring error of the un-traced form is 0.072 under k=3 dbc against 5.1
    free -- a factor of 70, entirely from that term.

    Dense at present: ``Pi`` is built as one ``(n_0 x n_3)`` array, so this is
    an A/B-resolution diagnostic.  Making it production means truncating with
    ``bandwidth`` (the collocation inverse is metric-free, so its decay depends
    only on ``p``) and assembling ``Pi`` as sparse.
    """
    if k != 3:
        raise NotImplementedError("transfer diagonal is k=3 only so far")
    from mrx.local_assembly import assemble_m1_local  # noqa: PLC0415

    geometry = seq.geometry if geometry is None else geometry
    if bandwidth == "star":
        t = [_metric_free_star_1d(seq, a) for a in range(3)]
        pi = np.kron(np.kron(t[0], t[1]), t[2])
    elif isinstance(bandwidth, tuple):        # ('free', band)
        pi = _greville_transfer_v3_to_v0(seq, geometry, bandwidth=bandwidth[1],
                                         metric_free=True)
    else:
        pi = _greville_transfer_v3_to_v0(seq, geometry, bandwidth=bandwidth)

    # BC FLIP. Hodge duality swaps essential and natural conditions, so the
    # partner of k=3 DIRICHLET is the FREE k=0 operator and the partner of k=3
    # FREE is the k=0 DIRICHLET one. Evaluating the k=0 energy in its extracted
    # space is what imposes that: embed back through E^T so the quadratic form
    # is the extracted operator's, not the raw one's.
    e0 = seq.e0 if dirichlet else seq.e0_dbc
    e0_mat = np.zeros((int(e0.forward_shape[0]), pi.shape[0]))
    e0_mat[np.asarray(e0.rows), np.asarray(e0.cols)] = np.asarray(e0.vals)
    pi = e0_mat.T @ (e0_mat @ pi)

    grad = _raw_grad_incidence(seq)
    m1 = assemble_m1_local(seq, geometry)

    # diag_EGtMGEt_direct scatters an O(nnz_per_row^2) plan, which suits a
    # sparse EXTRACTION and blows up (748 GiB) on a transfer whose columns are
    # dense. The columns are what we want here, so contract them directly:
    # diag(Pi^T S Pi)_i = <(S Pi)_i, Pi_i>, three matrix products, no plan.
    pi_j = jnp.asarray(pi)
    gp = jnp.asarray(grad) @ pi_j
    energy_s = jnp.einsum('ai,ai->i', jnp.asarray(grad).T @ (m1 @ gp), pi_j)

    # MASS NORMALIZATION. The spectral statement is between the mass-normalized
    # operators, M_3^-1 W_3 ~ M_0^-1 S_0, so what transfers is the RAYLEIGH
    # QUOTIENT, not the energy:
    #
    #   diag(W_3)_i ~ diag(M_3)_i * (T^T S_0 T)_ii / (T^T M_0 T)_ii
    #
    # Both diagonals are exact closed forms already in the code. Note this is
    # invariant under T -> cT, so however the metric-free star is normalized --
    # cell measures, averaging weights -- it cannot affect the answer. That
    # invariance is the reason to trust the form.
    from mrx.local_assembly import (assemble_m0_local,  # noqa: PLC0415
                                    build_mass_diagonal)
    m0 = assemble_m0_local(seq, geometry)
    energy_m = jnp.einsum('ai,ai->i', m0 @ pi_j, pi_j)
    floor = 1e-300 + jnp.zeros_like(energy_m)
    return jnp.asarray(build_mass_diagonal(seq, 3)) * energy_s / jnp.maximum(
        energy_m, floor)


def build_weak_term_raw_diagonal(seq, k: int, *, dirichlet: bool,
                                 split: Optional[str] = None,
                                 rescale: Optional[str] = None,
                                 return_info: bool = False):
    """Raw-DOF-space diagonal of the weak term, closed form and O(N).

    ``diag(W)`` for ``W = M_k G Pi M_{k-1}^{-1} Pi G^T M_k`` with every mass
    replaced by the raw_kron Kronecker model -- so ``M_{k-1}^{-1}`` here is
    literally the production mass preconditioner.  Each term of the expansion is
    then a pure Kronecker product, whose diagonal is the OUTER PRODUCT of its
    three 1-D diagonals::

        diag(term_{t,t'})_i = Lam_i^2  prod_a [ L^t_a (A^l_a)^-1 (L^t'_a)^T ]_{i_a i_a}

    with ``L^t_a = A^u_a lam_a g_a F_a sig_a`` -- upper 1-D mass, the rank-1
    split of the inner scaling, the 1-D incidence, one Kronecker factor of
    ``Pi``, and the rank-1 split of the lower scaling.  The pair sum is
    symmetric, so only ``t <= t'`` is formed.

    The OUTER ``Lam^2`` is the exact, unsplit scaling: it multiplies the
    finished diagonal pointwise, so it costs nothing and needs no separability.
    Only the two inner copies are split.

    **split** -- how the two inner scalings are handled.  Global default from
    ``MRX_LAPLACIAN_DIAG_SPLIT``.

    * ``'geometric'`` (default) / ``'arithmetic'`` -- rank-1 split, see
      :func:`_rank1_diagonal_split`.
    * ``'taylor1'`` -- local first-order expansion about the row instead of a
      global fit, see :func:`_weak_term_taylor_parts`.
    * ``'exact'`` -- no separation at all; dense, diagnostic only, see
      :func:`_weak_term_exact_parts`.
    * ``'codiff'`` -- abandons the expansion entirely for the codifferential
      energy ``||delta phi_i||^2``, by quadrature; k=3 only so far.  See
      :func:`mrx.local_assembly.build_codifferential_diagonal`.

    **rescale** -- leading-order repair of the inner splits, off by default and
    settable globally with ``MRX_LAPLACIAN_DIAG_RESCALE``.  Write the model as

        W = Lam A [Lam] G Pi Sig [A^l]^-1 Sig Pi G^T [Lam] A Lam

    Of the four ``Lam`` the two outer are exact; the two bracketed inner ones
    are split, as are the two ``Sig = Lam_l^-1`` inside ``B``.  ``A`` is a 1-D
    mass, so its bandwidth is ``p``: the inner scaling is sampled at rows ``j``
    a few knots from ``i``, and to leading order in ``h`` it is just
    ``Lam_i / lam_i``.  Two copies, so:

    * ``'upper'`` multiplies each group by ``(Lam / lam)^2`` on the upper grid.
      That ratio IS ``diag(M_k) / diag(M^_k)``: the split breaks the raw_kron
      model's defining property that it reproduces the exact mass diagonal, and
      this restores it.  Note the exponent -- the handoff note said
      ``(diag(M)/diag(M^))^2``, which double counts; there are two inner
      copies, not four.
    * ``'both'`` additionally multiplies by ``(Sig / sig)^2`` for the lower
      component of the group, resampled onto the upper grid (linear per axis,
      in log space).  The transfer is the approximation here: ``[A^l]^-1`` is
      dense, though exponentially decaying, so the locality argument is weaker
      than for the upper half.

    Both are free -- the residuals are byproducts of the split already computed
    -- exact on a separable weight, and positive, so each group's contribution
    stays ``diag(X_g [A^l]^-1 X_g^T) >= 0`` and the total stays positive.
    ``'upper'`` is the zeroth-order term of the ``'taylor1'`` series; measured
    against the probe it helps only where the weak term IS the operator (k=3)
    and costs iterations at k=1/2, so both stay off by default.
    """
    if k not in (1, 2, 3):
        raise ValueError("the weak term exists only for k = 1, 2, 3")
    if split is None:
        split = os.environ.get("MRX_LAPLACIAN_DIAG_SPLIT", "geometric")
    if rescale is None:
        rescale = os.environ.get("MRX_LAPLACIAN_DIAG_RESCALE", "none")

    info = {"split": split, "rescale": rescale}
    if split.startswith("transfer"):
        # Represent star(phi_i) in V_0 and take the gradient THERE: no
        # derivative of the k=3 basis and no dJ. See
        # build_transfer_weak_diagonal.
        if split == "transfer":
            band = None
        elif split == "transfer_star":
            band = "star"          # metric-free, bandwidth-1 DOF star
        elif split == "transfer_free":
            band = ("free", None)  # metric-free interpolation, exact inverse
        elif split.startswith("transfer_free_"):
            band = ("free", int(split.rsplit("_", 1)[1]))
        else:
            band = int(split.split("_")[1])
        raw = jnp.asarray(build_transfer_weak_diagonal(
            seq, k, dirichlet=dirichlet, bandwidth=band))
        return (raw, info) if return_info else raw
    if split == "codiff":
        # Not an expansion at all: diag(W)_i = ||delta phi_i||^2 by quadrature.
        # No mass model, no Sig, no separability assumption -- see
        # mrx.local_assembly.build_codifferential_diagonal.
        from mrx.local_assembly import (  # noqa: PLC0415
            build_codifferential_diagonal)
        raw = jnp.asarray(build_codifferential_diagonal(seq, k))
        return (raw, info) if return_info else raw
    if split == "exact":
        parts, lam_u = _weak_term_exact_parts(seq, k, dirichlet=dirichlet)
    elif split == "taylor1":
        parts, lam_u, info["term_pairs"] = _weak_term_taylor_parts(
            seq, k, dirichlet=dirichlet)
    else:
        groups, shapes_u, lam_u, corr = _weak_term_kron_terms(
            seq, k, dirichlet=dirichlet, split=split, rescale=rescale)
        parts = [np.zeros(shape) for shape in shapes_u]
        n_pairs = 0
        for key, entries in groups.items():
            c_u = key[0]
            block = np.zeros(shapes_u[c_u])
            for i, (sign_i, _, linv_i) in enumerate(entries):
                for j in range(i, len(entries)):
                    sign_j, l_j, _ = entries[j]
                    z = [np.einsum('mn,mn->m', linv_i[a], l_j[a]) for a in range(3)]
                    weight = (sign_i * sign_j) * (1.0 if i == j else 2.0)
                    block += weight * np.einsum('i,j,l->ijl', *z)
                    n_pairs += 1
            # Per GROUP, not per term: the group is one diag(X B_g X^T) with
            # B_g SPD, so it is nonnegative and a positive rescale cannot flip
            # it.
            parts[c_u] += block if corr[key] is None else block * corr[key]
        info["term_pairs"] = n_pairs
        info["terms"] = {key: len(v) for key, v in groups.items()}

    raw = np.concatenate([(p * np.asarray(lam_u[c]) ** 2).reshape(-1)
                          for c, p in enumerate(parts)])
    if return_info:
        return jnp.asarray(raw), info
    return jnp.asarray(raw)


def _weak_term_rows_by_apply(seq, operators, k: int, *, dirichlet: bool, indices):
    """Exact ``diag(W)`` at a handful of extracted rows, by operator applies."""
    from mrx.operators import (apply_derivative_matrix,  # noqa: PLC0415
                               apply_mass_matrix_preconditioner)

    lower = k - 1
    suffix = "_dbc" if dirichlet else ""
    size = int(getattr(seq, f"n{k}{suffix}"))

    def weak_apply(x):
        d_t_x = apply_derivative_matrix(
            seq, operators, x, lower, dirichlet_in=dirichlet,
            dirichlet_out=dirichlet, transpose=True)
        inner = apply_mass_matrix_preconditioner(
            seq, operators, d_t_x, lower, dirichlet=dirichlet, kind='auto')
        return apply_derivative_matrix(
            seq, operators, inner, lower, dirichlet_in=dirichlet,
            dirichlet_out=dirichlet)

    def row(i):
        return weak_apply(jnp.zeros(size).at[i].set(1.0))[i]

    # Warm the apply on a concrete vector first: the matrix-free mass plan is
    # HOST-built, so building it inside the trace raises
    # TracerArrayConversionError.
    weak_apply(jnp.zeros(size))
    # lax.map, never vmap: a batched probe fuses into a transpose kernel that
    # spills registers and crashes ptxas. See _diagonal_from_matvec.
    return np.asarray(jax.lax.map(row, jnp.asarray(indices)))


def build_weak_term_diagonal(seq, operators, k: int, *, dirichlet: bool, **kwargs):
    """``diag(E_k W_k E_k^T)``, the weak half of the k>=1 Jacobi Laplacian.

    Bulk extracted rows are pure selectors, so the closed-form raw diagonal
    supplies them directly.  The ``n_polar * n_zeta`` coupled rows would need
    off-diagonal raw entries of ``W``; they are taken EXACTLY instead, by one
    operator apply each.  That is a handful of applies against the probe's one
    per extracted row -- and it puts the exact value on the near-axis rows,
    which is where the Kronecker mass model is least accurate.
    """
    raw = np.asarray(build_weak_term_raw_diagonal(
        seq, k, dirichlet=dirichlet, **kwargs))

    e = getattr(seq, f"e{k}_dbc" if dirichlet else f"e{k}")
    rows = np.asarray(e.rows)
    cols = np.asarray(e.cols)
    vals = np.asarray(e.vals)
    n_ext = int(e.forward_shape[0])
    counts = np.bincount(rows, minlength=n_ext)

    diag = np.zeros(n_ext)
    single = counts[rows] == 1
    diag[rows[single]] = (vals[single] ** 2) * raw[cols[single]]

    coupled = np.flatnonzero(counts > 1)
    if coupled.size:
        diag[coupled] = _weak_term_rows_by_apply(
            seq, operators, k, dirichlet=dirichlet, indices=coupled)
    return jnp.asarray(diag)


def build_extracted_laplacian_diagonal(seq, operators, k: int, *, dirichlet: bool,
                                       **kwargs):
    """``diag(E_k L_k E_k^T)`` for ``k >= 1``, with no O(N) probe.

    ``L_k = S_k + W_k``.  Both halves are closed form in the raw DOF space --
    :func:`mrx.local_assembly.build_stiffness_diagonal` exactly, and
    :func:`build_weak_term_raw_diagonal` under the Kronecker mass model -- and
    the bulk rows of ``E`` are pure selectors, so the raw diagonal transfers
    straight through.  The ``n_polar * n_zeta`` coupled rows are taken exactly,
    by one apply of ``L_k`` each.

    ``L_0 = S_0`` has no weak term at all and is handled by
    :func:`mrx.local_assembly.build_extracted_stiffness_diagonal_k0`.
    """
    from mrx.local_assembly import build_stiffness_diagonal  # noqa: PLC0415
    from mrx.operators import apply_hodge_laplacian_approx  # noqa: PLC0415

    if k not in (1, 2, 3):
        raise ValueError("use build_extracted_stiffness_diagonal_k0 for k=0")

    raw = (np.asarray(build_stiffness_diagonal(seq, k))
           + np.asarray(build_weak_term_raw_diagonal(
               seq, k, dirichlet=dirichlet, **kwargs)))

    e = getattr(seq, f"e{k}_dbc" if dirichlet else f"e{k}")
    rows = np.asarray(e.rows)
    cols = np.asarray(e.cols)
    vals = np.asarray(e.vals)
    n_ext = int(e.forward_shape[0])
    counts = np.bincount(rows, minlength=n_ext)

    diag = np.zeros(n_ext)
    single = counts[rows] == 1
    diag[rows[single]] = (vals[single] ** 2) * raw[cols[single]]

    coupled = np.flatnonzero(counts > 1)

    # MRX_LAPLACIAN_DIAG_EXACT_RINGS=n also takes the n innermost radial rings
    # exactly. Every closed-form and transfer model measured so far degrades
    # sharply on the ring next to the polar block -- the transfer routes by
    # 20-50x, because V_3's extraction is unitary while V_0's folds that ring
    # into polar rows -- and it is a thin set, O(n_theta n_zeta) applies, the
    # same mechanism the coupled rows already use.
    n_rings = int(os.environ.get("MRX_LAPLACIAN_DIAG_EXACT_RINGS", "0"))
    if n_rings > 0:
        shapes_k = [tuple(int(v) for v in sh)
                    for sh in getattr(seq, f"basis_{k}").shape]
        starts_k = np.cumsum([0] + [int(np.prod(sh)) for sh in shapes_k])
        single_rows, single_cols = rows[single], cols[single]
        comp = np.searchsorted(starts_k[1:], single_cols, side="right")
        loc = single_cols - starts_k[comp]
        nt = np.array([sh[1] for sh in shapes_k])[comp]
        nz = np.array([sh[2] for sh in shapes_k])[comp]
        i_r = loc // (nt * nz)
        coupled = np.union1d(coupled, single_rows[i_r < n_rings])

    if coupled.size:
        size = int(getattr(seq, f"n{k}_dbc" if dirichlet else f"n{k}"))

        def row(i):
            x = jnp.zeros(size).at[i].set(1.0)
            return apply_hodge_laplacian_approx(
                seq, operators, x, k, dirichlet=dirichlet)[i]

        # Warm the apply outside the trace: its matrix-free mass plan is
        # host-built and cannot be constructed on tracers.
        apply_hodge_laplacian_approx(
            seq, operators, jnp.zeros(size), k, dirichlet=dirichlet)
        # lax.map, never vmap -- see _diagonal_from_matvec.
        diag[coupled] = np.asarray(jax.lax.map(row, jnp.asarray(coupled)))
    return jnp.asarray(diag)
