"""Mass surgery / Schur / tensor preconditioner -- retired from production.

Moved out of ``mrx/preconditioners.py`` on 2026-08-17 when the production mass
preconditioner became ``raw_kron`` -- the Kronecker model applied on the raw
(unextracted) grid, transferred with the extraction pseudoinverse
``E+ = E^T (E E^T)^-1``. See ``docs/research/mass_preconditioner_pivot.md``.

This code is **kept, not deleted**. It holds the surgery/bulk split, the dense
surgery Schur complement, the ``A_ss`` probe, the ``coupling_sb`` block, the
per-degree K0/K1/K2 surgery and tensor factor classes, and the tensor-mass
builder/apply built on top of them. All of it is still reachable via
``kind='tensor'`` and ``schur_diag_mode='tensor_probe'`` -- now research options
rather than defaults.

Why it was retired, one line each:

* ``coupling_sb`` is ``O(N n_z)``, asymptotically ``O(n^4)`` -- larger than the
  solution vector by a factor of ``n``, in a code whose premise is matrix-free;
  an estimated 24 GB at 64x128x64. ``raw_kron``'s ``(CC^T)^-1`` is ``O(n_z)``
  (27 KB at that resolution) and depends only on the sparsity of ``E``, so it
  never rebuilds when the geometry changes.
* The surgery Schur complement is dense with no closed entry form, which blocks
  the closed-form Schur diagonal that ``raw_kron`` makes possible.
* Setup cost: the CP/NTF fits and surgery probes dominate assembly.

Measured trade (CG to 1e-10, GPU, p=3, free BC): raw_kron costs a modest
iteration premium on the toroid and degrades *less* than the tensor path under
cross-section shaping, while removing all of the above.

``mrx.preconditioners`` re-exports every name here lazily through a module
``__getattr__``, so existing call sites keep working unchanged. The dependency
is one-way: this module imports shared primitives from ``mrx.preconditioners``,
which never imports this module at load time.
"""
from typing import Mapping, Optional  # noqa: F401

import equinox as eqx
import jax
import jax.numpy as jnp

from mrx.preconditioners import (
    BoundaryConditionPair,
    ExtractedMassApplyData,
    MassPreconditioners,
    RestrictedExtractedMassApplyData,
    TensorDiagonalBlockInverseFactors,
    TensorMassPreconditioner,
    _apply_extracted_submatrix,
    _apply_restricted_extracted_mass_operator_data,
    _apply_tensor_diagonal_block_forward,
    _apply_tensor_exact_block,
    _build_diagonal_tensor_block_factors,
    _build_extracted_mass_apply_data,
    _build_greville_mass_block_factors,
    _build_restricted_extracted_mass_apply_data,
    _bulk_tensor_shape,
    _core_size,
    _extract_selected_columns,
    _k0_bulk_weight_tensor,
    _k1_diagonal_metric_tensors,
    _k2_diagonal_metric_tensors,
    _k3_extracted_shape,
    _k3_weight_tensor,
    _select_mass_tensor_factors,
    _symmetric_pseudoinverse,
    _symmetrize,
    select_boundary_data,
)

__all__ = [
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
]


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
