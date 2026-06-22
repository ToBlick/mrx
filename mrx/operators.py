from __future__ import annotations

from typing import Mapping, Optional, Sequence
import warnings

import equinox as eqx
import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import jax.scipy as jsp

from mrx.assembly import assemble_vectorial
from mrx.extraction_operators import MatrixFreeExtraction, get_xi
import numpy as np

from mrx.local_assembly import assemble_mass_local
from mrx.preconditioners import (
    BoundaryConditionPair,
    K1MassSurgeryPreconditionerFactors,
    K1TensorMassPreconditionerFactors,
    K2MassSurgeryPreconditionerFactors,
    K2TensorMassPreconditionerFactors,
    MassPreconditioners,
    MassPreconditionerSpec,
    SchurPreconditionerSpec,
    SaddlePointPreconditionerSpec,
    _apply_bulk_to_surgery_coupling,
    _apply_extracted_submatrix,
    _apply_k1_bulk_diagonal_preconditioner,
    _apply_k1_bulk_preconditioner,
    _apply_k1_rt_art_coupling,
    _apply_k1_rt_atr_coupling,
    _apply_k1_rt_to_zeta_coupling,
    _apply_k1_zeta_to_rt_coupling,
    _apply_k2_bulk_diagonal_preconditioner,
    _apply_k2_bulk_preconditioner,
    _apply_k2_r_to_theta_coupling,
    _apply_k2_theta_to_r_coupling,
    _apply_k2_rt_to_zeta_coupling,
    _apply_k2_zeta_to_rt_coupling,
    _apply_surgery_to_bulk_coupling,
    _assemble_surgery_schur_inverse_from_applies,
    _build_effective_prior_terms,
    _build_extracted_mass_apply_data,
    _build_mass_referenced_tensor_block_factors,
    _build_tensor_block_factors_from_terms,
    _bulk_tensor_shape,
    _core_size,
    _cp_als_3tensor,
    _expand_residual_terms_with_prior,
    _apply_tensor_diagonal_block,
    _apply_tensor_exact_block,
    _apply_tensor_diagonal_block_forward,
    _arr_shape_k1,
    _build_chebyshev_apply_preconditioner,
    _estimate_chebyshev_lanczos_bounds_apply,
    _estimate_preconditioned_max_eigenvalue_apply,
    _greedy_cp_terms,
    _k2_diagonal_metric_tensors,
    _r_bulk_shape_k2,
    _restrict_radial_mass,
    _simultaneous_diagonalize_pair,
    _split_blocks,
    _symmetric_pseudoinverse,
    _select_mass_surgery_factors,
    _select_mass_tensor_factors,
    _symmetrize,
    _tensor_block_indices_k1,
    _tensor_block_indices_k2,
    _theta_bulk_shape_k1,
    _theta_shape_k2,
    _tensor_from_separated_terms,
    _zeta_bulk_shape_k1,
    _zeta_shape_k2,
    apply_mass_tensor_forward_model,
    apply_mass_tensor_preconditioner,
    diag_EAET_direct,
    build_mass_surgery_preconditioner,
    build_mass_tensor_preconditioner,
    default_mass_preconditioner,
    default_saddle_preconditioner,
    get_mass_jacobi_diaginv,
    mass_surgery_available,
    mass_tensor_available,
    select_boundary_data,
    set_mass_jacobi_pair,
    set_mass_surgery,
    set_mass_tensor,
)
from mrx.solvers import solve_saddle_point_minres, solve_singular_cg
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


class DenseSequenceOperators(eqx.Module):
    """Optional dense cache for extracted operator matrices."""

    m0: BoundaryConditionPair = eqx.field(default_factory=BoundaryConditionPair)
    m1: BoundaryConditionPair = eqx.field(default_factory=BoundaryConditionPair)
    m2: BoundaryConditionPair = eqx.field(default_factory=BoundaryConditionPair)
    m3: BoundaryConditionPair = eqx.field(default_factory=BoundaryConditionPair)
    d0: BoundaryConditionPair = eqx.field(default_factory=BoundaryConditionPair)
    d1: BoundaryConditionPair = eqx.field(default_factory=BoundaryConditionPair)
    d2: BoundaryConditionPair = eqx.field(default_factory=BoundaryConditionPair)
    s0: BoundaryConditionPair = eqx.field(default_factory=BoundaryConditionPair)
    s1: BoundaryConditionPair = eqx.field(default_factory=BoundaryConditionPair)
    s2: BoundaryConditionPair = eqx.field(default_factory=BoundaryConditionPair)
    s3: BoundaryConditionPair = eqx.field(default_factory=BoundaryConditionPair)
    l0: BoundaryConditionPair = eqx.field(default_factory=BoundaryConditionPair)
    l1: BoundaryConditionPair = eqx.field(default_factory=BoundaryConditionPair)
    l2: BoundaryConditionPair = eqx.field(default_factory=BoundaryConditionPair)
    l3: BoundaryConditionPair = eqx.field(default_factory=BoundaryConditionPair)
    p21: BoundaryConditionPair = eqx.field(default_factory=BoundaryConditionPair)
    p12: BoundaryConditionPair = eqx.field(default_factory=BoundaryConditionPair)
    p03: BoundaryConditionPair = eqx.field(default_factory=BoundaryConditionPair)
    p30: BoundaryConditionPair = eqx.field(default_factory=BoundaryConditionPair)


class IterativeRuntimeTuning(eqx.Module):
    """Dynamic spectral data for an iterative polynomial preconditioner."""

    lambda_max: Optional[jnp.ndarray] = None
    lambda_min: Optional[jnp.ndarray] = None


class BoundaryIterativeRuntimeTuning(eqx.Module):
    free: IterativeRuntimeTuning = eqx.field(default_factory=IterativeRuntimeTuning)
    dbc: IterativeRuntimeTuning = eqx.field(default_factory=IterativeRuntimeTuning)


class ShiftedIterativeRuntimeTuning(eqx.Module):
    unshifted: IterativeRuntimeTuning = eqx.field(default_factory=IterativeRuntimeTuning)
    shifted: IterativeRuntimeTuning = eqx.field(default_factory=IterativeRuntimeTuning)


class BoundaryShiftedIterativeRuntimeTuning(eqx.Module):
    free: ShiftedIterativeRuntimeTuning = eqx.field(default_factory=ShiftedIterativeRuntimeTuning)
    dbc: ShiftedIterativeRuntimeTuning = eqx.field(default_factory=ShiftedIterativeRuntimeTuning)


class DegreeIterativeRuntimeTuning(eqx.Module):
    k0: BoundaryIterativeRuntimeTuning = eqx.field(default_factory=BoundaryIterativeRuntimeTuning)
    k1: BoundaryIterativeRuntimeTuning = eqx.field(default_factory=BoundaryIterativeRuntimeTuning)
    k2: BoundaryIterativeRuntimeTuning = eqx.field(default_factory=BoundaryIterativeRuntimeTuning)
    k3: BoundaryIterativeRuntimeTuning = eqx.field(default_factory=BoundaryIterativeRuntimeTuning)


class DegreeShiftedIterativeRuntimeTuning(eqx.Module):
    k0: BoundaryShiftedIterativeRuntimeTuning = eqx.field(default_factory=BoundaryShiftedIterativeRuntimeTuning)
    k1: BoundaryShiftedIterativeRuntimeTuning = eqx.field(default_factory=BoundaryShiftedIterativeRuntimeTuning)
    k2: BoundaryShiftedIterativeRuntimeTuning = eqx.field(default_factory=BoundaryShiftedIterativeRuntimeTuning)
    k3: BoundaryShiftedIterativeRuntimeTuning = eqx.field(default_factory=BoundaryShiftedIterativeRuntimeTuning)


class SequenceRuntimeTuning(eqx.Module):
    """Dynamic runtime-tuning payload carried by ``SequenceOperators``."""

    mass: DegreeIterativeRuntimeTuning = eqx.field(default_factory=DegreeIterativeRuntimeTuning)
    scalar_hodge: DegreeShiftedIterativeRuntimeTuning = eqx.field(default_factory=DegreeShiftedIterativeRuntimeTuning)
    schur: DegreeShiftedIterativeRuntimeTuning = eqx.field(default_factory=DegreeShiftedIterativeRuntimeTuning)
    diffusion: DegreeShiftedIterativeRuntimeTuning = eqx.field(default_factory=DegreeShiftedIterativeRuntimeTuning)


class SequenceOperators(eqx.Module):
    """Dynamic operator bundle for a de Rham sequence.

    Stores geometry-dependent operator data explicitly so it can be carried
    through JAX transforms while the sequence object remains a static topology
    shell.
    """

    m0: Optional[jsparse.BCSR] = None
    m1: Optional[jsparse.BCSR] = None
    m2: Optional[jsparse.BCSR] = None
    m3: Optional[jsparse.BCSR] = None
    k0_tensor_hodge_precond: Optional[BoundaryConditionPair] = None
    k1_tensor_stiff_model: Optional[K1TensorCurlCurlForwardModel] = None
    k2_tensor_stiff_model: Optional[K2TensorDivDivForwardModel] = None
    k1_tensor_stiff_precond: Optional[BoundaryConditionPair] = None
    k2_tensor_stiff_precond: Optional[BoundaryConditionPair] = None
    e0: Optional[MatrixFreeExtraction] = None
    e0_T: Optional[MatrixFreeExtraction] = None
    e0_dbc: Optional[MatrixFreeExtraction] = None
    e0_dbc_T: Optional[MatrixFreeExtraction] = None
    e0_bc: Optional[MatrixFreeExtraction] = None
    e0_bc_T: Optional[MatrixFreeExtraction] = None
    e1: Optional[MatrixFreeExtraction] = None
    e1_T: Optional[MatrixFreeExtraction] = None
    e1_dbc: Optional[MatrixFreeExtraction] = None
    e1_dbc_T: Optional[MatrixFreeExtraction] = None
    e1_bc: Optional[MatrixFreeExtraction] = None
    e1_bc_T: Optional[MatrixFreeExtraction] = None
    e2: Optional[MatrixFreeExtraction] = None
    e2_T: Optional[MatrixFreeExtraction] = None
    e2_dbc: Optional[MatrixFreeExtraction] = None
    e2_dbc_T: Optional[MatrixFreeExtraction] = None
    e2_bc: Optional[MatrixFreeExtraction] = None
    e2_bc_T: Optional[MatrixFreeExtraction] = None
    e3: Optional[MatrixFreeExtraction] = None
    e3_T: Optional[MatrixFreeExtraction] = None
    e3_dbc: Optional[MatrixFreeExtraction] = None
    e3_dbc_T: Optional[MatrixFreeExtraction] = None
    e3_bc: Optional[MatrixFreeExtraction] = None
    e3_bc_T: Optional[MatrixFreeExtraction] = None
    mass_preconds: Optional[MassPreconditioners] = None
    runtime_tuning: SequenceRuntimeTuning = eqx.field(default_factory=SequenceRuntimeTuning)
    d0: Optional[jsparse.BCSR] = None
    d0_T: Optional[jsparse.BCSR] = None
    d1: Optional[jsparse.BCSR] = None
    d1_T: Optional[jsparse.BCSR] = None
    d2: Optional[jsparse.BCSR] = None
    d2_T: Optional[jsparse.BCSR] = None
    # Topological exterior-derivative incidence matrices on the full
    # pre-extraction DoF grid. Entries are in {-1, 0, +1}; they encode the
    # discrete de Rham complex structure and are geometry-independent. The
    # strong derivatives ``apply_strong_{grad,curl,div}`` multiply by these
    # directly (no mass solve). Stored as :class:`_MatrixFreeIncidence`
    # (difference stencils); no BCSR is ever materialised.
    g0: Optional[_MatrixFreeIncidence] = None
    g0_T: Optional[_MatrixFreeIncidence] = None
    g1: Optional[_MatrixFreeIncidence] = None
    g1_T: Optional[_MatrixFreeIncidence] = None
    g2: Optional[_MatrixFreeIncidence] = None
    g2_T: Optional[_MatrixFreeIncidence] = None
    # TRUE polar derivative correction. The directly-built incidence
    # ``E^T sp E`` is the topological d only when the extraction is unitary
    # (``E^T E = I``); on the polar axis the gluing is non-unitary, so the true
    # strong derivative is ``G = M^{-1} D = Gram_{k+1}^{-1} (E^T sp E)`` with the
    # (mass-free) coefficient Gram ``Gram = E^T E``. ``Gram`` is identity in the
    # bulk plus a small dense axis block, so its inverse is sparse (identity +
    # axis block) and is stored here per OUTPUT space and BC. ``None`` means the
    # extraction is unitary there (e.g. non-polar, or V3) so no correction is
    # needed and the apply is bit-identical to the raw incidence.
    inc_gram_inv_1: Optional[jsparse.BCSR] = None
    inc_gram_inv_1_dbc: Optional[jsparse.BCSR] = None
    inc_gram_inv_2: Optional[jsparse.BCSR] = None
    inc_gram_inv_2_dbc: Optional[jsparse.BCSR] = None
    inc_gram_inv_3: Optional[jsparse.BCSR] = None
    inc_gram_inv_3_dbc: Optional[jsparse.BCSR] = None
    # Analytic inverse-free polar grad G_0 (V0->V1), built from the incidence
    # pattern + polar coefficients xi alone (NO assembly inverse). Stored per
    # (dirichlet_in, dirichlet_out) BC pair, forward + transpose. ``None`` on
    # non-polar sequences -> apply falls back to the raw/Gram incidence path.
    g0_grad_00: Optional[jsparse.BCSR] = None
    g0_grad_00_T: Optional[jsparse.BCSR] = None
    g0_grad_01: Optional[jsparse.BCSR] = None
    g0_grad_01_T: Optional[jsparse.BCSR] = None
    g0_grad_10: Optional[jsparse.BCSR] = None
    g0_grad_10_T: Optional[jsparse.BCSR] = None
    g0_grad_11: Optional[jsparse.BCSR] = None
    g0_grad_11_T: Optional[jsparse.BCSR] = None
    # Analytic inverse-free polar curl G_1 (V1->V2), same construction one degree
    # up. ``None`` on non-polar -> raw incidence fallback.
    g1_curl_00: Optional[jsparse.BCSR] = None
    g1_curl_00_T: Optional[jsparse.BCSR] = None
    g1_curl_01: Optional[jsparse.BCSR] = None
    g1_curl_01_T: Optional[jsparse.BCSR] = None
    g1_curl_10: Optional[jsparse.BCSR] = None
    g1_curl_10_T: Optional[jsparse.BCSR] = None
    g1_curl_11: Optional[jsparse.BCSR] = None
    g1_curl_11_T: Optional[jsparse.BCSR] = None
    grad_grad: Optional[jsparse.BCSR] = None
    curl_curl: Optional[jsparse.BCSR] = None
    div_div: Optional[jsparse.BCSR] = None
    dd0_diaginv: Optional[object] = None
    dd1_diaginv: Optional[object] = None
    dd2_diaginv: Optional[object] = None
    dd3_diaginv: Optional[object] = None
    dd0_diaginv_dbc: Optional[object] = None
    dd1_diaginv_dbc: Optional[object] = None
    dd2_diaginv_dbc: Optional[object] = None
    dd3_diaginv_dbc: Optional[object] = None
    p21: Optional[jsparse.BCSR] = None
    p12: Optional[jsparse.BCSR] = None
    p03: Optional[jsparse.BCSR] = None
    p30: Optional[jsparse.BCSR] = None

    # Pre-probed diagonal inverses of the approximate Schur operator
    # S_k + D_{k-1} M_tensor^{-1}_{k-1} D_{k-1}^T.  Built at assembly time
    # by assemble_schur_jacobi_preconditioner; used as a cheap multiply in
    # the saddle-point Schur-outer Jacobi preconditioner instead of probing
    # at solve time.
    schur_diaginv_k1: Optional[jnp.ndarray] = None
    schur_diaginv_k1_dbc: Optional[jnp.ndarray] = None
    schur_diaginv_k2: Optional[jnp.ndarray] = None
    schur_diaginv_k2_dbc: Optional[jnp.ndarray] = None
    schur_diaginv_k3: Optional[jnp.ndarray] = None
    schur_diaginv_k3_dbc: Optional[jnp.ndarray] = None
    schur_diaginv_mode_k1: Optional[str] = None
    schur_diaginv_mode_k1_dbc: Optional[str] = None
    schur_diaginv_mode_k2: Optional[str] = None
    schur_diaginv_mode_k2_dbc: Optional[str] = None
    schur_diaginv_mode_k3: Optional[str] = None
    schur_diaginv_mode_k3_dbc: Optional[str] = None

    # Harmonic nullspaces of the k-form Laplacians. Each field, when set, holds
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
    dense: Optional[DenseSequenceOperators] = None

    def _laplacian_diaginv_field_name(self, k: int, dirichlet: bool) -> str:
        if k not in (0, 1, 2, 3):
            raise ValueError("k must be 0, 1, 2, or 3")
        suffix = "_dbc" if dirichlet else ""
        return f"dd{k}_diaginv{suffix}"

    def get_laplacian_diaginv(self, k: int, dirichlet: bool = True):
        """Return the stored Jacobi inverse diagonal for ``L_k`` if available."""
        return getattr(self, self._laplacian_diaginv_field_name(k, dirichlet))

    def with_laplacian_diaginv(self, k: int, value, dirichlet: bool = True):
        """Return a copy with updated stored Jacobi inverse diagonal for ``L_k``."""
        field_name = self._laplacian_diaginv_field_name(k, dirichlet)
        return eqx.tree_at(
            lambda ops: getattr(ops, field_name),
            self,
            value,
            is_leaf=lambda x: x is None,
        )

    @property
    def laplacian0_diaginv(self):
        return self.dd0_diaginv

    @property
    def laplacian1_diaginv(self):
        return self.dd1_diaginv

    @property
    def laplacian2_diaginv(self):
        return self.dd2_diaginv

    @property
    def laplacian3_diaginv(self):
        return self.dd3_diaginv

    @property
    def laplacian0_diaginv_dbc(self):
        return self.dd0_diaginv_dbc

    @property
    def laplacian1_diaginv_dbc(self):
        return self.dd1_diaginv_dbc

    @property
    def laplacian2_diaginv_dbc(self):
        return self.dd2_diaginv_dbc

    @property
    def laplacian3_diaginv_dbc(self):
        return self.dd3_diaginv_dbc

    def todense(self, seq, operator: str, k, dirichlet: bool = True,
                transpose: bool = False):
        """Return a dense matrix for one assembled operator block."""
        if self.dense is not None:
            match operator:
                case "mass":
                    pair = getattr(self.dense, f"m{k}")
                    return select_boundary_data(pair, dirichlet, f"Dense mass k={k}")
                case "derivative":
                    pair = getattr(self.dense, f"d{k}")
                    matrix = select_boundary_data(pair, dirichlet, f"Dense derivative k={k}")
                    return matrix.T if transpose else matrix
                case "stiffness":
                    pair = getattr(self.dense, f"s{k}")
                    return select_boundary_data(pair, dirichlet, f"Dense stiffness k={k}")
                case "hodge_laplacian" | "laplacian":
                    pair = getattr(self.dense, f"l{k}")
                    return select_boundary_data(pair, dirichlet, f"Dense Laplacian k={k}")
                case "projection":
                    if not isinstance(k, tuple) or len(k) != 2:
                        raise ValueError(
                            "Projection dense conversion expects k=(k_in, k_out)")
                    pair = getattr(self.dense, f"p{k[0]}{k[1]}")
                    return select_boundary_data(pair, dirichlet, f"Dense projection ({k[0]}, {k[1]})")
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
            case "hodge_laplacian" | "laplacian":
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
            "operator must be one of 'mass', 'derivative', 'stiffness', 'laplacian' (or legacy 'hodge_laplacian'), or 'projection'"
        )


def _select_degree_runtime_tuning(runtime_tuning, k: int, dirichlet: bool):
    pair = getattr(runtime_tuning, f"k{k}")
    return pair.dbc if dirichlet else pair.free


def _select_mass_runtime_tuning(
        operators: SequenceOperators, k: int, dirichlet: bool):
    return _select_degree_runtime_tuning(operators.runtime_tuning.mass, k, dirichlet)


def _select_scalar_hodge_runtime_tuning(
        operators: SequenceOperators, k: int, dirichlet: bool, eps: float):
    shifted_tuning = _select_degree_runtime_tuning(
        operators.runtime_tuning.scalar_hodge,
        k,
        dirichlet,
    )
    return shifted_tuning.shifted if eps != 0.0 else shifted_tuning.unshifted


def _select_schur_runtime_tuning(
        operators: SequenceOperators, k: int, dirichlet: bool, eps: float):
    shifted_tuning = _select_degree_runtime_tuning(
        operators.runtime_tuning.schur,
        k,
        dirichlet,
    )
    return shifted_tuning.shifted if eps != 0.0 else shifted_tuning.unshifted


def _select_diffusion_runtime_tuning(
        operators: SequenceOperators, k: int, dirichlet: bool, eps: float):
    shifted_tuning = _select_degree_runtime_tuning(
        operators.runtime_tuning.diffusion,
        k,
        dirichlet,
    )
    return shifted_tuning.shifted if eps != 0.0 else shifted_tuning.unshifted


def _set_mass_runtime_tuning(
        operators: SequenceOperators, *, k: int, dirichlet: bool,
        tuning: IterativeRuntimeTuning):
    bc_attr = 'dbc' if dirichlet else 'free'
    return eqx.tree_at(
        lambda ops: getattr(getattr(ops.runtime_tuning.mass, f"k{k}"), bc_attr),
        operators,
        tuning,
    )


def _set_scalar_hodge_runtime_tuning(
        operators: SequenceOperators, *, k: int, dirichlet: bool,
        eps: float, tuning: IterativeRuntimeTuning):
    bc_attr = 'dbc' if dirichlet else 'free'
    shift_attr = 'shifted' if eps != 0.0 else 'unshifted'
    return eqx.tree_at(
        lambda ops: getattr(
            getattr(getattr(ops.runtime_tuning.scalar_hodge, f"k{k}"), bc_attr),
            shift_attr,
        ),
        operators,
        tuning,
    )


def _set_schur_runtime_tuning(
        operators: SequenceOperators, *, k: int, dirichlet: bool,
        eps: float, tuning: IterativeRuntimeTuning):
    bc_attr = 'dbc' if dirichlet else 'free'
    shift_attr = 'shifted' if eps != 0.0 else 'unshifted'
    return eqx.tree_at(
        lambda ops: getattr(
            getattr(getattr(ops.runtime_tuning.schur, f"k{k}"), bc_attr),
            shift_attr,
        ),
        operators,
        tuning,
    )


def _set_diffusion_runtime_tuning(
        operators: SequenceOperators, *, k: int, dirichlet: bool,
        eps: float, tuning: IterativeRuntimeTuning):
    bc_attr = 'dbc' if dirichlet else 'free'
    shift_attr = 'shifted' if eps != 0.0 else 'unshifted'
    return eqx.tree_at(
        lambda ops: getattr(
            getattr(getattr(ops.runtime_tuning.diffusion, f"k{k}"), bc_attr),
            shift_attr,
        ),
        operators,
        tuning,
    )


class K0TensorHodgePreconditionerFactors(eqx.Module):
    core_size: int = eqx.field(static=True)
    bulk_shape: tuple[int, int, int] = eqx.field(static=True)
    schur_inv: jnp.ndarray
    schur_projector: Optional[jnp.ndarray] = None
    bulk_alpha: Optional[jnp.ndarray] = None
    bulk_V_r: Optional[jnp.ndarray] = None
    bulk_V_t: Optional[jnp.ndarray] = None
    bulk_V_z: Optional[jnp.ndarray] = None
    bulk_lam_r: Optional[jnp.ndarray] = None
    bulk_lam_t: Optional[jnp.ndarray] = None
    bulk_lam_z: Optional[jnp.ndarray] = None
    bulk_mass_r: Optional[jnp.ndarray] = None
    bulk_mass_t: Optional[jnp.ndarray] = None
    bulk_mass_z: Optional[jnp.ndarray] = None
    bulk_stiff_r: Optional[jnp.ndarray] = None
    bulk_stiff_t: Optional[jnp.ndarray] = None
    bulk_stiff_z: Optional[jnp.ndarray] = None
    bulk_term_mass_r: tuple[jnp.ndarray, ...] = ()
    bulk_term_mass_t: tuple[jnp.ndarray, ...] = ()
    bulk_term_mass_z: tuple[jnp.ndarray, ...] = ()
    bulk_term_stiff_r: tuple[jnp.ndarray, ...] = ()
    bulk_term_stiff_t: tuple[jnp.ndarray, ...] = ()
    bulk_term_stiff_z: tuple[jnp.ndarray, ...] = ()
    bulk_term_op_r: tuple[jnp.ndarray, ...] = ()
    bulk_term_op_t: tuple[jnp.ndarray, ...] = ()
    bulk_term_op_z: tuple[jnp.ndarray, ...] = ()
    bulk_modal_basis_r: Optional[jnp.ndarray] = None
    bulk_modal_basis_t: Optional[jnp.ndarray] = None
    bulk_modal_basis_z: Optional[jnp.ndarray] = None
    bulk_modal_mass_r: tuple[jnp.ndarray, ...] = ()
    bulk_modal_mass_t: tuple[jnp.ndarray, ...] = ()
    bulk_modal_mass_z: tuple[jnp.ndarray, ...] = ()
    bulk_modal_stiff_r: tuple[jnp.ndarray, ...] = ()
    bulk_modal_stiff_t: tuple[jnp.ndarray, ...] = ()
    bulk_modal_stiff_z: tuple[jnp.ndarray, ...] = ()
    bulk_modal_op_r: tuple[jnp.ndarray, ...] = ()
    bulk_modal_op_t: tuple[jnp.ndarray, ...] = ()
    bulk_modal_op_z: tuple[jnp.ndarray, ...] = ()
    bulk_modal_denom: Optional[jnp.ndarray] = None
    cp_relative_error: Optional[float] = None
    cp_final_delta: Optional[float] = None
    # Whether the dense core<->bulk coupling block was precomputed at build
    # time (default yes). When ``core_coupling`` is present the apply replaces
    # the two matrix-free k=0 stiffness couplings (K_0 = G_0^T M_1 G_0, an M_1
    # mass apply between two incidences, O(n^3 p^6)) with dense matvecs
    # ``core_coupling @`` / ``core_coupling.T @`` of cost O(bulk * core). The
    # core (polar-axis) size is small, so the block is cheap to store and probe.
    precompute_coupling: bool = eqx.field(static=True, default=True)
    core_coupling: Optional[jnp.ndarray] = None


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


class K2TensorDivDivForwardModel(eqx.Module):
    r_shape: tuple[int, int, int] = eqx.field(static=True)
    theta_shape: tuple[int, int, int] = eqx.field(static=True)
    zeta_shape: tuple[int, int, int] = eqx.field(static=True)
    scalar_shape: tuple[int, int, int] = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    g_r: jnp.ndarray
    g_t: jnp.ndarray
    g_z: jnp.ndarray
    mass_r_terms: tuple[jnp.ndarray, ...] = ()
    mass_t_terms: tuple[jnp.ndarray, ...] = ()
    mass_z_terms: tuple[jnp.ndarray, ...] = ()
    component_mass_r_terms: tuple[jnp.ndarray, ...] = ()
    component_mass_t_terms: tuple[jnp.ndarray, ...] = ()
    component_mass_z_terms: tuple[jnp.ndarray, ...] = ()
    cp_relative_error: Optional[float] = None
    cp_final_delta: Optional[float] = None


class K1TensorCurlCurlForwardModel(eqx.Module):
    r_shape: tuple[int, int, int] = eqx.field(static=True)
    theta_shape: tuple[int, int, int] = eqx.field(static=True)
    zeta_shape: tuple[int, int, int] = eqx.field(static=True)
    curl_r_shape: tuple[int, int, int] = eqx.field(static=True)
    curl_theta_shape: tuple[int, int, int] = eqx.field(static=True)
    curl_zeta_shape: tuple[int, int, int] = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    g_r: jnp.ndarray
    g_t: jnp.ndarray
    g_z: jnp.ndarray
    rr_mass_r_terms: tuple[jnp.ndarray, ...] = ()
    rr_mass_t_terms: tuple[jnp.ndarray, ...] = ()
    rr_mass_z_terms: tuple[jnp.ndarray, ...] = ()
    tt_mass_r_terms: tuple[jnp.ndarray, ...] = ()
    tt_mass_t_terms: tuple[jnp.ndarray, ...] = ()
    tt_mass_z_terms: tuple[jnp.ndarray, ...] = ()
    zz_mass_r_terms: tuple[jnp.ndarray, ...] = ()
    zz_mass_t_terms: tuple[jnp.ndarray, ...] = ()
    zz_mass_z_terms: tuple[jnp.ndarray, ...] = ()
    cp_relative_error: Optional[float] = None
    cp_final_delta: Optional[float] = None


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
        cp_ridge: float) -> tuple[tuple[jnp.ndarray, ...], tuple[jnp.ndarray, ...], tuple[jnp.ndarray, ...], float, float]:
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


def _assemble_weighted_1d_mass(B: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    return (B * weights[None, :]) @ B.T


def _assemble_unweighted_1d_mass(B: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    return _symmetrize(_assemble_weighted_1d_mass(B, weights))


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


def _assemble_dense_from_apply(apply, size: int, *, sequential: bool = False) -> jnp.ndarray:
    basis = jnp.eye(size, dtype=jnp.float64)
    if sequential:
        # Probe one column at a time. ``vmap`` batches every transient of the
        # probed apply by ``size``; for a matrix-free (sum-factorized) operator
        # the per-apply transient is a dense ``(ne, q, q, q)`` tensor, so the
        # batched peak is ``size``x larger and overflows device memory at high
        # resolution. ``lax.map`` evaluates the columns sequentially, keeping
        # the peak to a single apply at the cost of a serial build.
        #
        # One eager warmup call first: ``apply`` may build host-side static
        # state lazily (e.g. the matrix-free mass index plan, which converts
        # basis arrays via ``np.asarray``). Under ``lax.map`` the body is traced
        # as a ``scan``, so any such build would see tracers and raise
        # ``TracerArrayConversionError``. The eager call builds and caches that
        # state with concrete arrays, after which ``lax.map`` only re-invokes
        # the already-jitted apply.
        apply(basis[0])
        cols = jax.lax.map(apply, basis)  # cols[j] = apply(e_j)
        return cols.T
    return jax.vmap(apply, in_axes=1, out_axes=1)(basis)


def _k0_stiffness_diagonal_metric_tensors(seq) -> dict[str, jnp.ndarray]:
    jacobian = _reshape_quadrature_scalar_field(seq, seq.geometry.jacobian_j)
    metric_inv = _reshape_quadrature_matrix_field(seq, seq.geometry.metric_inv_jkl)
    return {
        'alpha_rr': jacobian * metric_inv[..., 0, 0],
        'alpha_thetatheta': jacobian * metric_inv[..., 1, 1],
        'alpha_zetazeta': jacobian * metric_inv[..., 2, 2],
    }


def _k0_tensor_hodge_config(
        operators: SequenceOperators,
        *,
        rank: Optional[int] = None,
        cp_maxiter: Optional[int] = None,
        cp_tol: Optional[float] = None,
        cp_ridge: Optional[float] = None):
    if None not in (rank, cp_maxiter, cp_tol, cp_ridge):
        return int(rank), int(cp_maxiter), float(cp_tol), float(cp_ridge)
    tensor = None if operators.mass_preconds is None else operators.mass_preconds.tensor
    if tensor is None:
        return 3, 100, 1e-9, 1e-12
    return tensor.ranks[0], tensor.cp_maxiter, tensor.cp_tol, tensor.cp_ridge


def _tensor_mass_rank(rank: int, cp_kwargs: Optional[Mapping[str, object]], k: int) -> int:
    if cp_kwargs is None:
        return int(rank)
    override = cp_kwargs.get(f"k{k}_rank")
    return int(rank if override is None else override)


def _normalize_cp_term_signs(
        scale: jnp.ndarray,
        factor_theta: jnp.ndarray,
        factor_r: jnp.ndarray,
        factor_z: jnp.ndarray):
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


def _fit_positive_rank_tensor_field(
        tensor_field: jnp.ndarray, *, rank: int, cp_maxiter: int, cp_tol: float,
        cp_ridge: float, radial_baseline: Optional[jnp.ndarray] = None,
        prior_terms: Optional[tuple[Mapping[str, jnp.ndarray], ...]] = None):
    effective_prior_terms = _build_effective_prior_terms(
        tensor_field.shape,
        radial_baseline=radial_baseline,
        prior_terms=prior_terms,
        dtype=tensor_field.dtype,
    )
    if effective_prior_terms is None:
        corrected_field = tensor_field
    else:
        prior_tensor = _tensor_from_separated_terms(effective_prior_terms, tensor_field.shape, tensor_field.dtype)
        prior_floor = 1e-12 * jnp.max(jnp.abs(prior_tensor))
        safe_floor = jnp.where(prior_floor > 0, prior_floor, 1.0)
        safe_prior = jnp.where(jnp.abs(prior_tensor) > safe_floor, prior_tensor, safe_floor)
        corrected_field = tensor_field / safe_prior

    weights, factors, cp_relative_error, cp_final_delta, _ = _cp_als_3tensor(
        corrected_field,
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
        scale = weights[idx]
        scale, factor_theta, factor_r, factor_z = _normalize_cp_term_signs(
            scale,
            factor_theta,
            factor_r,
            factor_z,
        )
        terms.append({
            'scale': scale,
            'theta_factor': factor_theta,
            'radial_factor': factor_r,
            'zeta_factor': factor_z,
        })

    terms = _expand_residual_terms_with_prior(tuple(terms), effective_prior_terms)

    return {
        'terms': tuple(terms),
        'cp_relative_error': cp_relative_error,
        'cp_final_delta': cp_final_delta,
    }


def _fit_positive_rank1_tensor_field(
        tensor_field: jnp.ndarray, *, cp_maxiter: int, cp_tol: float,
        cp_ridge: float, radial_baseline: Optional[jnp.ndarray] = None,
        prior_terms: Optional[tuple[Mapping[str, jnp.ndarray], ...]] = None):
    fit = _fit_positive_rank_tensor_field(
        tensor_field,
        rank=1,
        cp_maxiter=cp_maxiter,
        cp_tol=cp_tol,
        cp_ridge=cp_ridge,
        radial_baseline=radial_baseline,
        prior_terms=prior_terms,
    )
    return {
        **fit['terms'][0],
        'cp_relative_error': fit['cp_relative_error'],
        'cp_final_delta': fit['cp_final_delta'],
    }


def _project_tensor_to_radial_active_factor(
        tensor_field: jnp.ndarray,
        theta_factor: jnp.ndarray,
        zeta_factor: jnp.ndarray,
        *,
        radial_baseline: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    if radial_baseline is None:
        corrected_field = tensor_field
        scaled_radial_baseline = None
    else:
        scaled_radial_baseline = jnp.asarray(radial_baseline, dtype=tensor_field.dtype)
        corrected_field = tensor_field / scaled_radial_baseline[None, :, None]
    denom = jnp.maximum(jnp.sum(theta_factor * theta_factor) * jnp.sum(zeta_factor * zeta_factor), 1e-30)
    active = jnp.einsum('ijk,i,k->j', corrected_field, theta_factor, zeta_factor) / denom
    if scaled_radial_baseline is not None:
        active = scaled_radial_baseline * active
    return active


def _project_tensor_to_theta_active_factor(
        tensor_field: jnp.ndarray,
        radial_factor: jnp.ndarray,
        zeta_factor: jnp.ndarray) -> jnp.ndarray:
    denom = jnp.maximum(jnp.sum(radial_factor * radial_factor) * jnp.sum(zeta_factor * zeta_factor), 1e-30)
    return jnp.einsum('ijk,j,k->i', tensor_field, radial_factor, zeta_factor) / denom


def _project_tensor_to_zeta_active_factor(
        tensor_field: jnp.ndarray,
        radial_factor: jnp.ndarray,
        theta_factor: jnp.ndarray) -> jnp.ndarray:
    denom = jnp.maximum(jnp.sum(radial_factor * radial_factor) * jnp.sum(theta_factor * theta_factor), 1e-30)
    return jnp.einsum('ijk,j,i->k', tensor_field, radial_factor, theta_factor) / denom


def _solve_rank_coupled_projection(
        gram: jnp.ndarray,
        rhs: jnp.ndarray) -> jnp.ndarray:
    ridge_scale = jnp.maximum(jnp.max(jnp.abs(jnp.diag(gram))), 1.0)
    regularized = gram + (1e-12 * ridge_scale) * jnp.eye(gram.shape[0], dtype=gram.dtype)
    return jnp.linalg.solve(regularized, rhs)


def _project_tensor_to_radial_active_factors(
        tensor_field: jnp.ndarray,
        theta_factors: tuple[jnp.ndarray, ...],
        zeta_factors: tuple[jnp.ndarray, ...],
        *,
        radial_baseline: Optional[jnp.ndarray] = None) -> tuple[jnp.ndarray, ...]:
    if radial_baseline is None:
        corrected_field = tensor_field
        scaled_radial_baseline = None
    else:
        scaled_radial_baseline = jnp.asarray(radial_baseline, dtype=tensor_field.dtype)
        corrected_field = tensor_field / scaled_radial_baseline[None, :, None]

    theta_stack = jnp.stack(theta_factors, axis=0)
    zeta_stack = jnp.stack(zeta_factors, axis=0)
    gram = (theta_stack @ theta_stack.T) * (zeta_stack @ zeta_stack.T)
    rhs = jnp.einsum('ijk,si,sk->sj', corrected_field, theta_stack, zeta_stack)
    coeffs = _solve_rank_coupled_projection(gram, rhs)
    if scaled_radial_baseline is not None:
        coeffs = coeffs * scaled_radial_baseline[None, :]
    return tuple(coeffs[idx] for idx in range(coeffs.shape[0]))


def _project_tensor_to_theta_active_factors(
        tensor_field: jnp.ndarray,
        radial_factors: tuple[jnp.ndarray, ...],
        zeta_factors: tuple[jnp.ndarray, ...]) -> tuple[jnp.ndarray, ...]:
    radial_stack = jnp.stack(radial_factors, axis=0)
    zeta_stack = jnp.stack(zeta_factors, axis=0)
    gram = (radial_stack @ radial_stack.T) * (zeta_stack @ zeta_stack.T)
    rhs = jnp.einsum('ijk,sj,sk->si', tensor_field, radial_stack, zeta_stack)
    coeffs = _solve_rank_coupled_projection(gram, rhs)
    return tuple(coeffs[idx] for idx in range(coeffs.shape[0]))


def _project_tensor_to_zeta_active_factors(
        tensor_field: jnp.ndarray,
        radial_factors: tuple[jnp.ndarray, ...],
        theta_factors: tuple[jnp.ndarray, ...]) -> tuple[jnp.ndarray, ...]:
    radial_stack = jnp.stack(radial_factors, axis=0)
    theta_stack = jnp.stack(theta_factors, axis=0)
    gram = (radial_stack @ radial_stack.T) * (theta_stack @ theta_stack.T)
    rhs = jnp.einsum('ijk,sj,si->sk', tensor_field, radial_stack, theta_stack)
    coeffs = _solve_rank_coupled_projection(gram, rhs)
    return tuple(coeffs[idx] for idx in range(coeffs.shape[0]))




def _weighted_average_dense_matrix(
        matrices: tuple[jnp.ndarray, ...],
        weights: jnp.ndarray) -> jnp.ndarray:
    if len(matrices) == 0:
        raise ValueError("weighted average requires at least one matrix")
    total = jnp.zeros_like(matrices[0])
    for weight, matrix in zip(weights, matrices):
        total = total + weight * matrix
    weight_sum = jnp.sum(weights)
    safe_weight_sum = jnp.where(weight_sum > 0, weight_sum, 1.0)
    return _symmetrize(total / safe_weight_sum)


def _average_dense_matrices(matrices: tuple[jnp.ndarray, ...]) -> jnp.ndarray:
    if len(matrices) == 0:
        raise ValueError("average requires at least one matrix")
    weights = jnp.ones((len(matrices),), dtype=jnp.float64)
    return _weighted_average_dense_matrix(matrices, weights)


def _sum_dense_matrices(matrices: tuple[jnp.ndarray, ...]) -> jnp.ndarray:
    if len(matrices) == 0:
        raise ValueError("sum requires at least one matrix")
    total = jnp.zeros_like(matrices[0])
    for matrix in matrices:
        total = total + matrix
    return _symmetrize(total)


def _assemble_k0_stiffness_fd_bulk_factors(
        seq, *, dirichlet: bool, rank: int,
        cp_maxiter: int, cp_tol: float, cp_ridge: float):
    """Build k=0 stiffness bulk factors via Lynch fast diagonalisation.

    For k=0 the Hodge Laplacian equals the stiffness ``K_0 = G_0^T M_1 G_0``.
    With a diagonal-metric assumption the bulk operator separates as
    ``K_r⊗M_t⊗M_z + M_r⊗K_t⊗M_z + M_r⊗M_t⊗K_z`` with *different* metric
    channels on the three summands:

    - ``alpha_rr = J g^{rr}`` for ``K_r⊗M_t⊗M_z``
    - ``alpha_thetatheta = J g^{θθ}`` for ``M_r⊗K_t⊗M_z``
    - ``alpha_zetazeta = J g^{ζζ}`` for ``M_r⊗M_t⊗K_z``

    Each channel is fit independently by greedy rank-r CP. The rank-1
    leading terms define a per-axis Lynch FD basis ``V_a`` using a
    reference mass assembled from the rank-1 masses carried by the *other*
    two summands on that axis. All additive terms, including the leading
    rank-1 ones, are then projected into this basis and only their
    diagonal contributions are kept in the denominator. This matches the
    requested per-channel assembly and keeps the apply at the same six
    einsums regardless of rank.
    """
    bulk_shape = _bulk_tensor_shape(seq, dirichlet)
    nr_bulk, _, _ = bulk_shape
    types = seq.basis_0.types
    g_r = _dense_incidence_1d(seq.basis_0.nr, types[0])
    g_t = _dense_incidence_1d(seq.basis_0.nt, types[1])
    g_z = _dense_incidence_1d(seq.basis_0.nz, types[2])

    metric = _k0_stiffness_diagonal_metric_tensors(seq)
    rr_terms, rr_rel_err, rr_final_delta = _greedy_cp_terms(
        metric['alpha_rr'],
        rank=rank,
        cp_maxiter=cp_maxiter,
        cp_tol=cp_tol,
        cp_ridge=cp_ridge,
    )
    tt_terms, tt_rel_err, tt_final_delta = _greedy_cp_terms(
        metric['alpha_thetatheta'],
        rank=rank,
        cp_maxiter=cp_maxiter,
        cp_tol=cp_tol,
        cp_ridge=cp_ridge,
    )
    zz_terms, zz_rel_err, zz_final_delta = _greedy_cp_terms(
        metric['alpha_zetazeta'],
        rank=rank,
        cp_maxiter=cp_maxiter,
        cp_tol=cp_tol,
        cp_ridge=cp_ridge,
    )

    def _assemble_directional_term(term, direction: str):
        radial_w = seq.quad.w_x * (term['scale'] * term['radial_factor'])
        theta_w = seq.quad.w_y * term['theta_factor']
        zeta_w = seq.quad.w_z * term['zeta_factor']

        full_mass_r = _assemble_weighted_1d_mass(seq.basis_r_jk, radial_w)
        mass_r = _restrict_radial_window(full_mass_r, radial_start=2, nr=nr_bulk)
        mass_t = _assemble_weighted_1d_mass(seq.basis_t_jk, theta_w)
        mass_z = _assemble_weighted_1d_mass(seq.basis_z_jk, zeta_w)

        if direction == 'rr':
            full_stiff_r = _assemble_weighted_1d_stiffness(
                seq.basis_r_jk, seq.d_basis_r_jk, radial_w, g_r)
            return {
                'mass_r': mass_r,
                'mass_t': mass_t,
                'mass_z': mass_z,
                'stiff_r': _restrict_radial_window(full_stiff_r, radial_start=2, nr=nr_bulk),
            }
        if direction == 'tt':
            return {
                'mass_r': mass_r,
                'mass_t': mass_t,
                'mass_z': mass_z,
                'stiff_t': _assemble_weighted_1d_stiffness(
                    seq.basis_t_jk, seq.d_basis_t_jk, theta_w, g_t),
            }
        if direction == 'zz':
            return {
                'mass_r': mass_r,
                'mass_t': mass_t,
                'mass_z': mass_z,
                'stiff_z': _assemble_weighted_1d_stiffness(
                    seq.basis_z_jk, seq.d_basis_z_jk, zeta_w, g_z),
            }
        raise ValueError(f"Unknown k=0 stiffness direction {direction!r}")

    rr_data = tuple(_assemble_directional_term(term, 'rr') for term in rr_terms)
    tt_data = tuple(_assemble_directional_term(term, 'tt') for term in tt_terms)
    zz_data = tuple(_assemble_directional_term(term, 'zz') for term in zz_terms)

    rr0 = rr_data[0]
    tt0 = tt_data[0]
    zz0 = zz_data[0]

    ref_mass_r = _weighted_average_dense_matrix(
        (tt0['mass_r'], zz0['mass_r']),
        jnp.asarray([
            jnp.linalg.norm(tt0['stiff_t']) * jnp.linalg.norm(tt0['mass_z']),
            jnp.linalg.norm(zz0['mass_t']) * jnp.linalg.norm(zz0['stiff_z']),
        ], dtype=jnp.float64),
    )
    ref_mass_t = _weighted_average_dense_matrix(
        (rr0['mass_t'], zz0['mass_t']),
        jnp.asarray([
            jnp.linalg.norm(rr0['stiff_r']) * jnp.linalg.norm(rr0['mass_z']),
            jnp.linalg.norm(zz0['mass_r']) * jnp.linalg.norm(zz0['stiff_z']),
        ], dtype=jnp.float64),
    )
    ref_mass_z = _weighted_average_dense_matrix(
        (rr0['mass_z'], tt0['mass_z']),
        jnp.asarray([
            jnp.linalg.norm(rr0['stiff_r']) * jnp.linalg.norm(rr0['mass_t']),
            jnp.linalg.norm(tt0['mass_r']) * jnp.linalg.norm(tt0['stiff_t']),
        ], dtype=jnp.float64),
    )

    # The rank-1 leading terms define the per-axis Lynch FD basis.
    V_r, _ = _simultaneous_diagonalize_pair(ref_mass_r, rr0['stiff_r'])
    V_t, _ = _simultaneous_diagonalize_pair(ref_mass_t, tt0['stiff_t'])
    V_z, _ = _simultaneous_diagonalize_pair(ref_mass_z, zz0['stiff_z'])

    denom = jnp.zeros(bulk_shape, dtype=jnp.float64)
    term_op_r = []
    term_op_t = []
    term_op_z = []

    for term in rr_data:
        d_r = jnp.einsum("ji,jk,ki->i", V_r, term['stiff_r'], V_r)
        d_t = jnp.einsum("ji,jk,ki->i", V_t, term['mass_t'], V_t)
        d_z = jnp.einsum("ji,jk,ki->i", V_z, term['mass_z'], V_z)
        denom = denom + d_r[:, None, None] * d_t[None, :, None] * d_z[None, None, :]
        term_op_r.append(term['stiff_r'])
        term_op_t.append(term['mass_t'])
        term_op_z.append(term['mass_z'])

    for term in tt_data:
        d_r = jnp.einsum("ji,jk,ki->i", V_r, term['mass_r'], V_r)
        d_t = jnp.einsum("ji,jk,ki->i", V_t, term['stiff_t'], V_t)
        d_z = jnp.einsum("ji,jk,ki->i", V_z, term['mass_z'], V_z)
        denom = denom + d_r[:, None, None] * d_t[None, :, None] * d_z[None, None, :]
        term_op_r.append(term['mass_r'])
        term_op_t.append(term['stiff_t'])
        term_op_z.append(term['mass_z'])

    for term in zz_data:
        d_r = jnp.einsum("ji,jk,ki->i", V_r, term['mass_r'], V_r)
        d_t = jnp.einsum("ji,jk,ki->i", V_t, term['mass_t'], V_t)
        d_z = jnp.einsum("ji,jk,ki->i", V_z, term['stiff_z'], V_z)
        denom = denom + d_r[:, None, None] * d_t[None, :, None] * d_z[None, None, :]
        term_op_r.append(term['mass_r'])
        term_op_t.append(term['mass_t'])
        term_op_z.append(term['stiff_z'])

    # Floor near-kernel modes: under free BCs, K_0 has a constant-mode null
    # space; the Schur surgery handles it, but the bulk denom may still touch
    # zero. Under Dirichlet BCs the radial axis is clamped so denom is
    # strictly positive, but we keep the floor for robustness.
    denom_floor = 1e-12 * jnp.max(jnp.abs(denom))
    safe_floor = jnp.where(denom_floor > 0, denom_floor, 1.0)
    denom = jnp.where(denom > safe_floor, denom, safe_floor)

    return {
        'bulk_shape': bulk_shape,
        'bulk_modal_basis_r': V_r,
        'bulk_modal_basis_t': V_t,
        'bulk_modal_basis_z': V_z,
        'bulk_modal_denom': denom,
        'bulk_term_op_r': tuple(term_op_r),
        'bulk_term_op_t': tuple(term_op_t),
        'bulk_term_op_z': tuple(term_op_z),
        'cp_relative_error': max(rr_rel_err, tt_rel_err, zz_rel_err),
        'cp_final_delta': max(rr_final_delta, tt_final_delta, zz_final_delta),
    }


def _build_k0_tensor_hodge_preconditioner_factors(
        *, core_size: int, schur_inv: jnp.ndarray, bulk_data: dict,
        schur_projector: Optional[jnp.ndarray] = None,
        precompute_coupling: bool = True,
        core_coupling: Optional[jnp.ndarray] = None) -> K0TensorHodgePreconditionerFactors:
    return K0TensorHodgePreconditionerFactors(
        core_size=core_size,
        bulk_shape=bulk_data['bulk_shape'],
        schur_inv=schur_inv,
        schur_projector=schur_projector,
        bulk_alpha=bulk_data.get('bulk_alpha'),
        bulk_V_r=bulk_data.get('bulk_V_r'),
        bulk_V_t=bulk_data.get('bulk_V_t'),
        bulk_V_z=bulk_data.get('bulk_V_z'),
        bulk_lam_r=bulk_data.get('bulk_lam_r'),
        bulk_lam_t=bulk_data.get('bulk_lam_t'),
        bulk_lam_z=bulk_data.get('bulk_lam_z'),
        bulk_mass_r=bulk_data.get('bulk_mass_r'),
        bulk_mass_t=bulk_data.get('bulk_mass_t'),
        bulk_mass_z=bulk_data.get('bulk_mass_z'),
        bulk_stiff_r=bulk_data.get('bulk_stiff_r'),
        bulk_stiff_t=bulk_data.get('bulk_stiff_t'),
        bulk_stiff_z=bulk_data.get('bulk_stiff_z'),
        bulk_term_mass_r=bulk_data.get('bulk_term_mass_r', ()),
        bulk_term_mass_t=bulk_data.get('bulk_term_mass_t', ()),
        bulk_term_mass_z=bulk_data.get('bulk_term_mass_z', ()),
        bulk_term_stiff_r=bulk_data.get('bulk_term_stiff_r', ()),
        bulk_term_stiff_t=bulk_data.get('bulk_term_stiff_t', ()),
        bulk_term_stiff_z=bulk_data.get('bulk_term_stiff_z', ()),
        bulk_term_op_r=bulk_data.get('bulk_term_op_r', ()),
        bulk_term_op_t=bulk_data.get('bulk_term_op_t', ()),
        bulk_term_op_z=bulk_data.get('bulk_term_op_z', ()),
        bulk_modal_basis_r=bulk_data.get('bulk_modal_basis_r'),
        bulk_modal_basis_t=bulk_data.get('bulk_modal_basis_t'),
        bulk_modal_basis_z=bulk_data.get('bulk_modal_basis_z'),
        bulk_modal_mass_r=bulk_data.get('bulk_modal_mass_r', ()),
        bulk_modal_mass_t=bulk_data.get('bulk_modal_mass_t', ()),
        bulk_modal_mass_z=bulk_data.get('bulk_modal_mass_z', ()),
        bulk_modal_stiff_r=bulk_data.get('bulk_modal_stiff_r', ()),
        bulk_modal_stiff_t=bulk_data.get('bulk_modal_stiff_t', ()),
        bulk_modal_stiff_z=bulk_data.get('bulk_modal_stiff_z', ()),
        bulk_modal_op_r=bulk_data.get('bulk_modal_op_r', ()),
        bulk_modal_op_t=bulk_data.get('bulk_modal_op_t', ()),
        bulk_modal_op_z=bulk_data.get('bulk_modal_op_z', ()),
        bulk_modal_denom=bulk_data.get('bulk_modal_denom'),
        cp_relative_error=bulk_data.get('cp_relative_error'),
        cp_final_delta=bulk_data.get('cp_final_delta'),
        precompute_coupling=precompute_coupling,
        core_coupling=core_coupling,
    )


def _apply_k0_tensor_hodge_bulk_shared_inverse(
        factors: K0TensorHodgePreconditionerFactors,
        rhs_b: jnp.ndarray) -> jnp.ndarray:
    if factors.bulk_modal_basis_r is None or factors.bulk_modal_basis_t is None or factors.bulk_modal_basis_z is None:
        raise ValueError("Missing shared modal basis for multi-rank k=0 tensor Hodge bulk inverse")

    nr, nt, nz = factors.bulk_shape
    modes = rhs_b.reshape(nr, nt, nz)
    modes = jnp.einsum('ji,jkl->ikl', factors.bulk_modal_basis_r, modes)
    modes = jnp.einsum('ji,kjl->kil', factors.bulk_modal_basis_t, modes)
    modes = jnp.einsum('ji,klj->kli', factors.bulk_modal_basis_z, modes)

    if factors.bulk_modal_denom is not None:
        denom = factors.bulk_modal_denom.astype(rhs_b.dtype)
    elif len(factors.bulk_modal_op_r) > 0:
        denom = jnp.zeros((nr, nt, nz), dtype=rhs_b.dtype)
        for op_r, op_t, op_z in zip(
                factors.bulk_modal_op_r,
                factors.bulk_modal_op_t,
                factors.bulk_modal_op_z):
            denom = denom + op_r[:, None, None] * op_t[None, :, None] * op_z[None, None, :]
    else:
        denom = jnp.zeros((nr, nt, nz), dtype=rhs_b.dtype)
        for stiff_r, mass_r, stiff_t, mass_t, stiff_z, mass_z in zip(
                factors.bulk_modal_stiff_r,
                factors.bulk_modal_mass_r,
                factors.bulk_modal_stiff_t,
                factors.bulk_modal_mass_t,
                factors.bulk_modal_stiff_z,
                factors.bulk_modal_mass_z):
            denom = denom + (
                stiff_r[:, None, None] * mass_t[None, :, None] * mass_z[None, None, :]
                + mass_r[:, None, None] * stiff_t[None, :, None] * mass_z[None, None, :]
                + mass_r[:, None, None] * mass_t[None, :, None] * stiff_z[None, None, :]
            )

    denom_floor = 1e-12 * jnp.max(jnp.abs(denom))
    safe_floor = jnp.where(denom_floor > 0, denom_floor, 1.0)
    safe_denom = jnp.where(denom > safe_floor, denom, safe_floor)
    modes = modes / safe_denom

    modes = jnp.einsum('ij,jkl->ikl', factors.bulk_modal_basis_r, modes)
    modes = jnp.einsum('ij,kjl->kil', factors.bulk_modal_basis_t, modes)
    modes = jnp.einsum('ij,klj->kli', factors.bulk_modal_basis_z, modes)
    return modes.reshape(-1)


def _apply_k0_tensor_hodge_bulk_inverse(
        factors: K0TensorHodgePreconditionerFactors,
        rhs_b: jnp.ndarray) -> jnp.ndarray:
    if factors.bulk_modal_basis_r is not None:
        return _apply_k0_tensor_hodge_bulk_shared_inverse(factors, rhs_b)

    tensor = rhs_b.reshape(factors.bulk_shape)
    out = _fd_apply_3d(
        factors.bulk_V_r,
        factors.bulk_V_t,
        factors.bulk_V_z,
        factors.bulk_lam_r,
        factors.bulk_lam_t,
        factors.bulk_lam_z,
        factors.bulk_alpha,
        tensor,
        eps=0.0,
    )
    return out.reshape(-1)


def _apply_k0_tensor_hodge_bulk_forward(
        factors: K0TensorHodgePreconditionerFactors,
        rhs_b: jnp.ndarray) -> jnp.ndarray:
    if len(factors.bulk_term_op_r) > 0:
        tensor = rhs_b.reshape(factors.bulk_shape)
        out = jnp.zeros_like(tensor)
        for op_r, op_t, op_z in zip(
                factors.bulk_term_op_r,
                factors.bulk_term_op_t,
                factors.bulk_term_op_z):
            term = jnp.einsum('ij,jkl->ikl', op_r, tensor)
            term = jnp.einsum('ij,kjl->kil', op_t, term)
            term = jnp.einsum('ij,klj->kli', op_z, term)
            out = out + term
        return out.reshape(-1)

    term_mass_r = factors.bulk_term_mass_r
    term_mass_t = factors.bulk_term_mass_t
    term_mass_z = factors.bulk_term_mass_z
    term_stiff_r = factors.bulk_term_stiff_r
    term_stiff_t = factors.bulk_term_stiff_t
    term_stiff_z = factors.bulk_term_stiff_z

    if len(term_mass_r) == 0:
        if any(matrix is None for matrix in (
                factors.bulk_mass_r,
                factors.bulk_mass_t,
                factors.bulk_mass_z,
                factors.bulk_stiff_r,
                factors.bulk_stiff_t,
                factors.bulk_stiff_z)):
            raise ValueError("Missing stored bulk matrices for k=0 tensor Hodge forward model")
        term_mass_r = (factors.bulk_mass_r,)
        term_mass_t = (factors.bulk_mass_t,)
        term_mass_z = (factors.bulk_mass_z,)
        term_stiff_r = (factors.bulk_stiff_r,)
        term_stiff_t = (factors.bulk_stiff_t,)
        term_stiff_z = (factors.bulk_stiff_z,)

    if not (len(term_mass_r) == len(term_mass_t) == len(term_mass_z) == len(term_stiff_r) == len(term_stiff_t) == len(term_stiff_z)):
        raise ValueError("Missing stored bulk matrices for k=0 tensor Hodge forward model")

    tensor = rhs_b.reshape(factors.bulk_shape)
    out = jnp.zeros_like(tensor)
    for mass_r, mass_t, mass_z, stiff_r, stiff_t, stiff_z in zip(
            term_mass_r,
            term_mass_t,
            term_mass_z,
            term_stiff_r,
            term_stiff_t,
            term_stiff_z):
        term = jnp.einsum('ij,jkl->ikl', stiff_r, tensor)
        term = jnp.einsum('ij,kjl->kil', mass_t, term)
        term = jnp.einsum('ij,klj->kli', mass_z, term)
        out = out + term

        term = jnp.einsum('ij,jkl->ikl', mass_r, tensor)
        term = jnp.einsum('ij,kjl->kil', stiff_t, term)
        term = jnp.einsum('ij,klj->kli', mass_z, term)
        out = out + term

        term = jnp.einsum('ij,jkl->ikl', mass_r, tensor)
        term = jnp.einsum('ij,kjl->kil', mass_t, term)
        term = jnp.einsum('ij,klj->kli', stiff_z, term)
        out = out + term
    return out.reshape(-1)


def _apply_k0_tensor_hodge_core_block(
        seq,
        operators: SequenceOperators,
        core_size: int,
        rhs_c: jnp.ndarray,
        *,
        dirichlet: bool) -> jnp.ndarray:
    size = seq.n0_dbc if dirichlet else seq.n0
    full = jnp.zeros((size,), dtype=rhs_c.dtype)
    full = full.at[:core_size].set(rhs_c)
    return apply_stiffness(seq, operators, full, 0, dirichlet=dirichlet)[:core_size]


def _apply_k0_tensor_hodge_surgery_to_bulk_coupling(
        seq,
        operators: SequenceOperators,
        core_size: int,
        rhs_c: jnp.ndarray,
        *,
        dirichlet: bool) -> jnp.ndarray:
    size = seq.n0_dbc if dirichlet else seq.n0
    full = jnp.zeros((size,), dtype=rhs_c.dtype)
    full = full.at[:core_size].set(rhs_c)
    return apply_stiffness(seq, operators, full, 0, dirichlet=dirichlet)[core_size:]


def _apply_k0_tensor_hodge_bulk_to_surgery_coupling(
        seq,
        operators: SequenceOperators,
        core_size: int,
        rhs_b: jnp.ndarray,
        *,
        dirichlet: bool) -> jnp.ndarray:
    size = seq.n0_dbc if dirichlet else seq.n0
    full = jnp.zeros((size,), dtype=rhs_b.dtype)
    full = full.at[core_size:].set(rhs_b)
    return apply_stiffness(seq, operators, full, 0, dirichlet=dirichlet)[:core_size]


def _assemble_k0_tensor_hodge_preconditioner(
        seq, operators: SequenceOperators, *,
        rank: int, cp_maxiter: int, cp_tol: float, cp_ridge: float,
        precompute_coupling: bool = True,
        dirichlet_flags: tuple[bool, ...] = (False, True)) -> BoundaryConditionPair:
    pair = BoundaryConditionPair()
    core_size = _core_size(seq)

    for dirichlet in dirichlet_flags:
        bulk_data = _assemble_k0_stiffness_fd_bulk_factors(
            seq,
            dirichlet=dirichlet,
            rank=rank,
            cp_maxiter=cp_maxiter,
            cp_tol=cp_tol,
            cp_ridge=cp_ridge,
        )

        bulk_factors = _build_k0_tensor_hodge_preconditioner_factors(
            core_size=core_size,
            schur_inv=jnp.eye(core_size, dtype=jnp.float64),
            bulk_data=bulk_data,
        )
        ass = _symmetrize(_assemble_dense_from_apply(
            lambda rhs_c, seq=seq, operators=operators, core_size=core_size, dirichlet=dirichlet:
            _apply_k0_tensor_hodge_core_block(seq, operators, core_size, rhs_c, dirichlet=dirichlet),
            core_size,
            sequential=True,
        ))
        bulk_apply = lambda rhs_b, bulk_factors=bulk_factors: _apply_k0_tensor_hodge_bulk_inverse(bulk_factors, rhs_b)
        surgery_to_bulk_apply = lambda rhs_c, seq=seq, operators=operators, core_size=core_size, dirichlet=dirichlet: _apply_k0_tensor_hodge_surgery_to_bulk_coupling(seq, operators, core_size, rhs_c, dirichlet=dirichlet)
        bulk_to_surgery_apply = lambda rhs_b, seq=seq, operators=operators, core_size=core_size, dirichlet=dirichlet: _apply_k0_tensor_hodge_bulk_to_surgery_coupling(seq, operators, core_size, rhs_b, dirichlet=dirichlet)
        schur = _symmetrize(_assemble_dense_from_apply(
            lambda rhs_c, ass=ass, bulk_apply=bulk_apply, surgery_to_bulk_apply=surgery_to_bulk_apply, bulk_to_surgery_apply=bulk_to_surgery_apply:
            ass @ rhs_c - bulk_to_surgery_apply(bulk_apply(surgery_to_bulk_apply(rhs_c))),
            core_size,
            sequential=True,
        ))

        schur_inv = _symmetric_pseudoinverse(schur)

        # Precompute the dense core->bulk coupling block C0 (bulk x core) once,
        # so the per-apply core<->bulk couplings become dense matvecs instead of
        # full matrix-free k=0 stiffness applies. K_0 is symmetric, so the
        # bulk->core block is exactly C0^T. Probed sequentially (one matrix-free
        # stiffness apply per core DOF, same as the Schur build above).
        core_coupling = None
        if precompute_coupling:
            core_coupling = _assemble_dense_from_apply(
                surgery_to_bulk_apply,
                core_size,
                sequential=True,
            )

        factors = _build_k0_tensor_hodge_preconditioner_factors(
            core_size=core_size,
            schur_inv=schur_inv,
            bulk_data=bulk_data,
            precompute_coupling=precompute_coupling,
            core_coupling=core_coupling,
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
        seq, operators: SequenceOperators, rhs: jnp.ndarray, *, dirichlet: bool) -> jnp.ndarray:
    pair = operators.k0_tensor_hodge_precond
    if pair is None:
        raise ValueError('Tensor Hodge preconditioner k=0 is not assembled')
    factors = select_boundary_data(pair, dirichlet, 'Tensor Hodge k=0')
    core_size = factors.core_size
    rhs_c = rhs[:core_size]
    rhs_b = rhs[core_size:]
    y = _apply_k0_tensor_hodge_bulk_inverse(factors, rhs_b)
    if factors.core_coupling is not None:
        # Dense precomputed coupling: bulk->core = C0^T (K_0 symmetric).
        schur_rhs = rhs_c - factors.core_coupling.T @ y
    else:
        schur_rhs = rhs_c - _apply_k0_tensor_hodge_bulk_to_surgery_coupling(
            seq,
            operators,
            core_size,
            y,
            dirichlet=dirichlet,
        )
    z = factors.schur_inv @ schur_rhs
    if factors.core_coupling is not None:
        # Dense precomputed coupling: core->bulk = C0.
        x_b = y - _apply_k0_tensor_hodge_bulk_inverse(factors, factors.core_coupling @ z)
    else:
        x_b = y - _apply_k0_tensor_hodge_bulk_inverse(
            factors,
            _apply_k0_tensor_hodge_surgery_to_bulk_coupling(
                seq,
                operators,
                core_size,
                z,
                dirichlet=dirichlet,
            ),
        )
    return jnp.concatenate([z, x_b])


def _apply_k0_tensor_hodge_forward_model(
        seq,
        operators: SequenceOperators,
        rhs: jnp.ndarray,
        *,
        dirichlet: bool) -> jnp.ndarray:
    pair = operators.k0_tensor_hodge_precond
    if pair is None:
        raise ValueError('Tensor Hodge forward model k=0 is not assembled')
    factors = select_boundary_data(pair, dirichlet, 'Tensor Hodge k=0')
    core_size = factors.core_size
    rhs_c = rhs[:core_size]
    rhs_b = rhs[core_size:]
    out_c = _apply_k0_tensor_hodge_core_block(
        seq,
        operators,
        core_size,
        rhs_c,
        dirichlet=dirichlet,
    ) + _apply_k0_tensor_hodge_bulk_to_surgery_coupling(
        seq,
        operators,
        core_size,
        rhs_b,
        dirichlet=dirichlet,
    )
    out_b = _apply_k0_tensor_hodge_surgery_to_bulk_coupling(
        seq,
        operators,
        core_size,
        rhs_c,
        dirichlet=dirichlet,
    ) + _apply_k0_tensor_hodge_bulk_forward(factors, rhs_b)
    return jnp.concatenate([out_c, out_b])
# TODO: remove when assembly files are gone
def _assemble_mass_block(seq, geometry, k):
    if k not in (0, 1, 2, 3):
        raise ValueError("k must be 0, 1, 2 or 3")
    sp = assemble_mass_local(seq, k, geometry)
    return jsparse.BCSR.from_bcoo(sp)



# TODO: remove when assembly files are gone (mass apply is matrix-free via build_matrixfree_mass_apply)
def update_mass_operator(seq, geometry, operators: Optional[SequenceOperators], k: int):
    """Return an operator bundle with the k-th mass operator updated.

    Only the sparse mass matrix ``m{k}`` is built and stored here. Mass
    preconditioners (Jacobi, surgery, tensor) are intentionally *not*
    assembled as a side effect: the Jacobi diagonal is built on demand by
    :func:`_mass_diaginv` (direct selection), and the surgery/tensor
    preconditioners have dedicated explicit builders
    (:func:`assemble_tensor_mass_preconditioner` etc.). This avoids computing
    Jacobi data that the tensor preconditioner never uses.
    """
    del geometry  # geometry is already attached to seq and used inside assembly
    sp = _assemble_mass_block(seq, seq.geometry, k)
    operators = _ensure_extraction_operators(seq, operators)
    match k:
        case 0:
            return eqx.tree_at(
                lambda ops: ops.m0,
                operators,
                sp,
                is_leaf=lambda x: x is None,
            )
        case 1:
            return eqx.tree_at(
                lambda ops: ops.m1,
                operators,
                sp,
                is_leaf=lambda x: x is None,
            )
        case 2:
            return eqx.tree_at(
                lambda ops: ops.m2,
                operators,
                sp,
                is_leaf=lambda x: x is None,
            )
        case 3:
            return eqx.tree_at(
                lambda ops: ops.m3,
                operators,
                sp,
                is_leaf=lambda x: x is None,
            )
    raise ValueError("k must be 0, 1, 2 or 3")


# TODO: remove when assembly files are gone
def assemble_mass_operators(seq, geometry, operators: Optional[SequenceOperators] = None,
                            ks: Sequence[int] = (0, 1, 2, 3)):
    """Assemble mass operators for the requested form degrees."""
    for k in ks:
        operators = update_mass_operator(seq, geometry, operators, k)
    return operators


def assemble_mass_jacobi_preconditioner(
        seq, operators: Optional[SequenceOperators] = None,
        *, ks: Sequence[int] = (0, 1, 2, 3)):
    """Assemble/store Jacobi mass diagonals eagerly for requested degrees.

    Reuses the existing direct diagonal extraction helper and stores only the
    inverse-diagonal vectors on ``operators.mass_preconds.jacobi`` (free + DBC)
    for reuse by Jacobi preconditioners.
    """
    operators = _ensure_extraction_operators(seq, operators)
    preconds = operators.mass_preconds

    for k in ks:
        if k not in (0, 1, 2, 3):
            raise ValueError("Mass Jacobi assembly only supports k=0,1,2,3")

        sp = _assemble_mass_block(seq, seq.geometry, k)
        e_free, _ = _mass_extraction(operators, k, False)
        e_dbc, _ = _mass_extraction(operators, k, True)
        diag_free = diag_EAET_direct(e_free, sp)
        diag_dbc = diag_EAET_direct(e_dbc, sp)
        pair = BoundaryConditionPair(
            free=_invert_diagonal(diag_free),
            dbc=_invert_diagonal(diag_dbc),
        )
        preconds = set_mass_jacobi_pair(preconds, k, pair)

    return eqx.tree_at(
        lambda ops: ops.mass_preconds,
        operators,
        preconds,
        is_leaf=lambda x: x is None,
    )


def assemble_mass_surgery_preconditioner(
        seq, operators: Optional[SequenceOperators] = None,
        *, ks: Sequence[int] = (0, 1, 2), precompute_coupling: bool = True):
    operators = _ensure_extraction_operators(seq, operators)
    for k in ks:
        if k not in (0, 1, 2):
            raise ValueError("Mass surgery preconditioner assembly only supports k=0, k=1 and k=2")

    surgery_precond = operators.mass_preconds.surgery if operators.mass_preconds is not None else None
    for k in ks:
        # The surgery preconditioner is built entirely matrix-free: it only
        # probes/applies M_k through the sum-factorized kernel, so no BCSR mass
        # matrix is ever assembled.
        mass_apply = mass_core_apply(seq, operators, k)
        surgery_precond = build_mass_surgery_preconditioner(
            seq,
            mass_apply,
            k=k,
            existing=surgery_precond,
            precompute_coupling=precompute_coupling,
        )
    mass_preconds = set_mass_surgery(operators.mass_preconds, surgery_precond)
    return eqx.tree_at(
        lambda ops: ops.mass_preconds,
        operators,
        mass_preconds,
        is_leaf=lambda x: x is None,
    )


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
    precompute_coupling = bool((cp_kwargs or {}).get("precompute_coupling", True))
    surgery_ks = tuple(k for k in ks if k in (0, 1, 2))
    if surgery_ks:
        operators = assemble_mass_surgery_preconditioner(
            seq,
            operators=operators,
            ks=surgery_ks,
            precompute_coupling=precompute_coupling,
        )
    for k in ks:
        if k not in (0, 1, 2, 3):
            raise ValueError("Tensor mass preconditioner assembly only supports k=0, k=1, k=2 and k=3")
    # The tensor mass preconditioner is built entirely matrix-free (surgery
    # probes/applies and the k=3 true block all go through ``mass_core_apply``),
    # so no BCSR mass matrix is assembled here.

    tensor_precond = operators.mass_preconds.tensor if operators.mass_preconds is not None else None
    for k in ks:
        tensor_rank = _tensor_mass_rank(rank, cp_kwargs, k)
        k3_true_block_apply = None
        if k == 3:
            ops_for_k3 = operators
            k3_true_block_apply = {
                False: lambda x, _ops=ops_for_k3: apply_mass_matrix(
                    seq, _ops, x, 3, dirichlet=False),
                True: lambda x, _ops=ops_for_k3: apply_mass_matrix(
                    seq, _ops, x, 3, dirichlet=True),
            }
        tensor_precond = build_mass_tensor_preconditioner(
            seq,
            k=k,
            rank=tensor_rank,
            fallback_rank=rank,
            cp_kwargs=cp_kwargs,
            existing=tensor_precond,
            surgery_precond=operators.mass_preconds.surgery,
            k3_true_block_apply=k3_true_block_apply,
        )
    mass_preconds = set_mass_tensor(operators.mass_preconds, tensor_precond)
    return eqx.tree_at(
        lambda ops: ops.mass_preconds,
        operators,
        mass_preconds,
        is_leaf=lambda x: x is None,
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
    block_chebyshev_steps = int(cp_kwargs.get("block_chebyshev_steps", 3))
    block_lanczos_iterations = int(cp_kwargs.get("block_lanczos_iterations", 16))
    block_lanczos_max_eig_inflation = float(cp_kwargs.get("block_lanczos_max_eig_inflation", 1.1))
    block_lanczos_min_eig_deflation = float(cp_kwargs.get("block_lanczos_min_eig_deflation", 0.85))
    block_lanczos_min_eig_floor_fraction = float(cp_kwargs.get("block_lanczos_min_eig_floor_fraction", 1e-3))
    richardson_steps = int(cp_kwargs.get("richardson_steps", 0))
    richardson_omega = float(cp_kwargs.get("richardson_omega", 1.0))
    precompute_coupling = bool(cp_kwargs.get("precompute_coupling", True))
    surgery_schur_pinv_tol = float(
        cp_kwargs.get("surgery_schur_pinv_tol", cp_kwargs.get("schur_pinv_tol", 1e-8))
    )
    bulk_block_pinv_tol = float(cp_kwargs.get("bulk_block_pinv_tol", 1e-8))
    k1_inner_schur = bool(cp_kwargs.get("k1_inner_schur", False))
    k2_inner_schur = bool(cp_kwargs.get("k2_inner_schur", False))

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

                arr_true_apply = lambda x, surgery=surgery: _apply_extracted_submatrix(
                    surgery.apply_data, surgery.r_indices, surgery.r_indices, x)
                theta_true_apply = lambda x, surgery=surgery: _apply_extracted_submatrix(
                    surgery.apply_data, surgery.theta_bulk_indices, surgery.theta_bulk_indices, x)
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
                arr_factors = _build_mass_referenced_tensor_block_factors(
                    full_shape=arr_shape,
                    reference_r=arr_ref_r,
                    reference_t=arr_ref_t,
                    reference_z=arr_ref_z,
                    axis_operator_r=None,
                    axis_operator_t=arr_op_t,
                    axis_operator_z=arr_op_z,
                    term_matrices=tuple(arr_terms),
                    cp_relative_error=model.cp_relative_error,
                    cp_final_delta=model.cp_final_delta,
                    chebyshev_steps=block_chebyshev_steps,
                    chebyshev_lanczos_iterations=block_lanczos_iterations,
                    chebyshev_lanczos_max_eig_inflation=block_lanczos_max_eig_inflation,
                    chebyshev_lanczos_min_eig_deflation=block_lanczos_min_eig_deflation,
                    chebyshev_lanczos_min_eig_floor_fraction=block_lanczos_min_eig_floor_fraction,
                    chebyshev_seed=1200 + 10 * int(dirichlet),
                    richardson_steps=richardson_steps,
                    richardson_omega=richardson_omega,
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
                theta_factors = _build_mass_referenced_tensor_block_factors(
                    full_shape=theta_shape,
                    reference_r=theta_ref_r,
                    reference_t=theta_ref_t,
                    reference_z=theta_ref_z,
                    axis_operator_r=theta_op_r,
                    axis_operator_t=None,
                    axis_operator_z=theta_op_z,
                    term_matrices=tuple(theta_terms),
                    cp_relative_error=model.cp_relative_error,
                    cp_final_delta=model.cp_final_delta,
                    chebyshev_steps=block_chebyshev_steps,
                    chebyshev_lanczos_iterations=block_lanczos_iterations,
                    chebyshev_lanczos_max_eig_inflation=block_lanczos_max_eig_inflation,
                    chebyshev_lanczos_min_eig_deflation=block_lanczos_min_eig_deflation,
                    chebyshev_lanczos_min_eig_floor_fraction=block_lanczos_min_eig_floor_fraction,
                    chebyshev_seed=1201 + 10 * int(dirichlet),
                    richardson_steps=richardson_steps,
                    richardson_omega=richardson_omega,
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
                zeta_factors = _build_mass_referenced_tensor_block_factors(
                    full_shape=zeta_shape,
                    reference_r=zeta_ref_r,
                    reference_t=zeta_ref_t,
                    reference_z=zeta_ref_z,
                    axis_operator_r=zeta_op_r,
                    axis_operator_t=zeta_op_t,
                    axis_operator_z=None,
                    term_matrices=tuple(zeta_terms),
                    cp_relative_error=model.cp_relative_error,
                    cp_final_delta=model.cp_final_delta,
                    chebyshev_steps=block_chebyshev_steps,
                    chebyshev_lanczos_iterations=block_lanczos_iterations,
                    chebyshev_lanczos_max_eig_inflation=block_lanczos_max_eig_inflation,
                    chebyshev_lanczos_min_eig_deflation=block_lanczos_min_eig_deflation,
                    chebyshev_lanczos_min_eig_floor_fraction=block_lanczos_min_eig_floor_fraction,
                    chebyshev_seed=1202 + 10 * int(dirichlet),
                    richardson_steps=richardson_steps,
                    richardson_omega=richardson_omega,
                    modal_pinv_tol=bulk_block_pinv_tol,
                    true_block_apply=zeta_true_apply,
                )

                bulk_apply = (
                    lambda rhs_bulk, surgery=surgery, arr_factors=arr_factors, theta_factors=theta_factors, zeta_factors=zeta_factors:
                    _apply_k1_bulk_preconditioner(surgery, arr_factors, theta_factors, zeta_factors, rhs_bulk)
                ) if k1_inner_schur else (
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
                        use_inner_schur=k1_inner_schur,
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
            r_bulk_factors = _build_mass_referenced_tensor_block_factors(
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
                chebyshev_steps=block_chebyshev_steps,
                chebyshev_lanczos_iterations=block_lanczos_iterations,
                chebyshev_lanczos_max_eig_inflation=block_lanczos_max_eig_inflation,
                chebyshev_lanczos_min_eig_deflation=block_lanczos_min_eig_deflation,
                chebyshev_lanczos_min_eig_floor_fraction=block_lanczos_min_eig_floor_fraction,
                chebyshev_seed=1300 + 10 * int(dirichlet),
                richardson_steps=richardson_steps,
                richardson_omega=richardson_omega,
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
            theta_factors = _build_mass_referenced_tensor_block_factors(
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
                chebyshev_steps=block_chebyshev_steps,
                chebyshev_lanczos_iterations=block_lanczos_iterations,
                chebyshev_lanczos_max_eig_inflation=block_lanczos_max_eig_inflation,
                chebyshev_lanczos_min_eig_deflation=block_lanczos_min_eig_deflation,
                chebyshev_lanczos_min_eig_floor_fraction=block_lanczos_min_eig_floor_fraction,
                chebyshev_seed=1301 + 10 * int(dirichlet),
                richardson_steps=richardson_steps,
                richardson_omega=richardson_omega,
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
            zeta_factors = _build_mass_referenced_tensor_block_factors(
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
                chebyshev_steps=block_chebyshev_steps,
                chebyshev_lanczos_iterations=block_lanczos_iterations,
                chebyshev_lanczos_max_eig_inflation=block_lanczos_max_eig_inflation,
                chebyshev_lanczos_min_eig_deflation=block_lanczos_min_eig_deflation,
                chebyshev_lanczos_min_eig_floor_fraction=block_lanczos_min_eig_floor_fraction,
                chebyshev_seed=1302 + 10 * int(dirichlet),
                richardson_steps=richardson_steps,
                richardson_omega=richardson_omega,
                modal_pinv_tol=bulk_block_pinv_tol,
                true_block_apply=zeta_true_apply,
            )

            bulk_apply = (
                lambda rhs_bulk, surgery=surgery, r_bulk_factors=r_bulk_factors, theta_factors=theta_factors, zeta_factors=zeta_factors:
                _apply_k2_bulk_preconditioner(surgery, r_bulk_factors, theta_factors, zeta_factors, rhs_bulk)
            ) if k2_inner_schur else (
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
                    use_inner_schur=k2_inner_schur,
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
        bulk_apply = _apply_k2_bulk_preconditioner if factors.use_inner_schur else _apply_k2_bulk_diagonal_preconditioner
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
        bulk_apply = _apply_k1_bulk_preconditioner if factors.use_inner_schur else _apply_k1_bulk_diagonal_preconditioner
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


def _tensor_available(seq, operators: SequenceOperators, k: int) -> bool:
    return mass_tensor_available(seq, operators.mass_preconds, k)


def _surgery_available(seq, operators: SequenceOperators, k: int) -> bool:
    return mass_surgery_available(seq, operators.mass_preconds, k)


def apply_mass_tensor_preconditioner_ops(
        seq, operators: SequenceOperators, v, k: int, dirichlet: bool = True):
    # For k=3 there is no surgery split, so the inner dispatch never had a
    # handle on the assembled mass and bulk-Chebyshev would polish only
    # surrogate-against-surrogate (a no-op).  Provide the true mass apply
    # so Chebyshev can absorb CP modelling error.
    true_block_apply_k3 = None
    if k == 3:
        true_block_apply_k3 = lambda x: apply_mass_matrix(
            seq, operators, x, 3, dirichlet=dirichlet,
        )
    return apply_mass_tensor_preconditioner(
        seq, operators.mass_preconds, v, k, dirichlet=dirichlet,
        true_block_apply_k3=true_block_apply_k3,
    )


def apply_mass_tensor_forward_model_ops(
        seq, operators: SequenceOperators, v, k: int, dirichlet: bool = True):
    return apply_mass_tensor_forward_model(
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


# TODO: remove (deprecated no-op shim)
def assemble_tensor_hodge_preconditioner(
    seq, operators: Optional[SequenceOperators] = None, *,
    rank: Optional[int] = None,
    cp_maxiter: Optional[int] = None,
    cp_tol: Optional[float] = None,
    cp_ridge: Optional[float] = None):
    """Deprecated no-op; use :func:`assemble_tensor_laplacian_preconditioner`."""
    return _ensure_extraction_operators(seq, operators)


def assemble_tensor_laplacian_preconditioner(
        seq, operators: Optional[SequenceOperators] = None, *,
        ks: Sequence[int] = (0,),
        rank: Optional[int] = None,
        cp_kwargs: Optional[dict] = None):
    """Assemble the scalar k=0 tensor Hodge-Laplacian preconditioner.

    Only k=0 is supported. ``rank`` and ``cp_kwargs`` override the CP fit
    parameters; when ``None`` the values stored on the tensor mass
    preconditioner are used.
    """
    operators = _ensure_extraction_operators(seq, operators)
    cp_kwargs = {} if cp_kwargs is None else dict(cp_kwargs)
    cp_maxiter = cp_kwargs.get("maxiter")
    cp_tol = cp_kwargs.get("tol")
    cp_ridge = cp_kwargs.get("ridge")
    precompute_coupling = bool(cp_kwargs.get("precompute_coupling", True))

    for k in ks:
        if k != 0:
            raise ValueError(
                "Tensor Laplacian preconditioner assembly only supports k=0")

    # The core-block apply uses ``apply_stiffness(seq, operators, ., 0)`` which
    # composes ``G0`` and a matrix-free ``M1`` apply, so ``M1`` need not be
    # stored; only ``G0`` is required here.
    if _incidence_components(operators, 0)[0] is None:
        operators = update_incidence_operator(seq, operators, 0)

    cfg_rank, cfg_maxiter, cfg_tol, cfg_ridge = _k0_tensor_hodge_config(
        operators,
        rank=rank,
        cp_maxiter=cp_maxiter,
        cp_tol=cp_tol,
        cp_ridge=cp_ridge,
    )
    tensor_precond = _assemble_k0_tensor_hodge_preconditioner(
        seq,
        operators,
        rank=cfg_rank,
        cp_maxiter=cfg_maxiter,
        cp_tol=cfg_tol,
        cp_ridge=cfg_ridge,
        precompute_coupling=precompute_coupling,
    )
    return eqx.tree_at(
        lambda ops: ops.k0_tensor_hodge_precond,
        operators,
        tensor_precond,
        is_leaf=lambda x: x is None,
    )



def _fd_apply_3d(V_r, V_t, V_z, lam_r, lam_t, lam_z, alpha, x, eps: float = 0.0):
    """Apply ``(L + eps M)^{-1}`` via fast diagonalisation on a 3-tensor ``x``."""
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


# ---------------------------------------------------------------------------
# Matrix-free mass apply
# ---------------------------------------------------------------------------

def _flat_dof_plan(gx, gy, gz, shape):
    """Static flat index plan into a component's flattened DOF grid."""
    Sx, Sy, Sz = (int(s) for s in shape)
    gx = np.asarray(gx)
    gy = np.asarray(gy)
    gz = np.asarray(gz)
    idx = (gx[:, None, None, :, None, None] * (Sy * Sz)
           + gy[None, :, None, None, :, None] * Sz
           + gz[None, None, :, None, None, :])
    return jnp.asarray(idx.astype(np.int32))


def _element_apply(Bvals_r, Bvals_c, W, x_flat_c, gather_idx_c):
    """One (row-comp, col-comp) element contraction folded against a vector."""
    Bxr, Byr, Bzr = Bvals_r
    Bxc, Byc, Bzc = Bvals_c
    x_local = x_flat_c[gather_idx_c]
    t1 = jnp.einsum('xqb,xyzbdf->xyzqdf', Bxc, x_local)
    t2 = jnp.einsum('yrd,xyzqdf->xyzqrf', Byc, t1)
    u = jnp.einsum('zsf,xyzqrf->xyzqrs', Bzc, t2)
    u = u * W
    s1 = jnp.einsum('xqa,xyzqrs->xyzars', Bxr, u)
    s2 = jnp.einsum('yrc,xyzars->xyzacs', Byr, s1)
    return jnp.einsum('zse,xyzacs->xyzace', Bzr, s2)


def build_matrixfree_mass_apply(seq, k, geometry=None):
    """Return a jitted raw-DOF-space ``x -> M_k x`` that never stores ``M_k``."""
    from mrx.local_assembly import (
        _elem_counts, _split_field, _component_axis_bases_k1,
        _component_axis_bases_k2, _quad_gauss_weight, _bases_for_form,
    )
    geometry = seq.geometry if geometry is None else geometry
    nx, ny, nz = seq.quad.nx, seq.quad.ny, seq.quad.nz
    ne_x, ne_y, ne_z, qx, qy, qz = _elem_counts(seq)
    gw = _quad_gauss_weight(seq)

    if k == 0:
        form = seq.basis_0
        comp = _bases_for_form(seq, form, lambda f, c: [f.Λ[0], f.Λ[1], f.Λ[2]], 1)
        weight = geometry.jacobian_j
        pairs = [(0, 0)]
        weight_of = {(0, 0): weight}
        n_comp = 1
    elif k == 3:
        form = seq.basis_3
        comp = _bases_for_form(seq, form, lambda f, c: [f.dΛ[0], f.dΛ[1], f.dΛ[2]], 1)
        weight = 1.0 / geometry.jacobian_j
        pairs = [(0, 0)]
        weight_of = {(0, 0): weight}
        n_comp = 1
    elif k == 1:
        form = seq.basis_1
        comp = _bases_for_form(seq, form, _component_axis_bases_k1, 3)
        metric = geometry.metric_inv_jkl * geometry.jacobian_j[:, None, None]
        pairs = [(cr, cc) for cr in range(3) for cc in range(3)]
        weight_of = {(cr, cc): metric[:, cr, cc] for cr, cc in pairs}
        n_comp = 3
    elif k == 2:
        form = seq.basis_2
        comp = _bases_for_form(seq, form, _component_axis_bases_k2, 3)
        metric = geometry.metric_jkl * (1.0 / geometry.jacobian_j)[:, None, None]
        pairs = [(cr, cc) for cr in range(3) for cc in range(3)]
        weight_of = {(cr, cc): metric[:, cr, cc] for cr, cc in pairs}
        n_comp = 3
    else:
        raise ValueError("k must be 0, 1, 2 or 3")

    shapes = form.shape
    starts = [0]
    for c in range(n_comp):
        Sx, Sy, Sz = shapes[c]
        starts.append(starts[-1] + Sx * Sy * Sz)

    W_split = {}
    for (cr, cc) in pairs:
        Wf = _split_field(weight_of[(cr, cc)], nx, ny, nz,
                          ne_x, ne_y, ne_z, qx, qy, qz)
        W_split[(cr, cc)] = Wf * gw

    Bvals = tuple((c[0], c[2], c[4]) for c in comp)
    gather_idx = tuple(
        _flat_dof_plan(comp[cc][1], comp[cc][3], comp[cc][5], shapes[cc])
        for cc in range(n_comp))
    seg_idx = tuple(
        _flat_dof_plan(comp[cr][1], comp[cr][3], comp[cr][5],
                       shapes[cr]).reshape(-1)
        for cr in range(n_comp))
    nseg = tuple(int(np.prod(shapes[c])) for c in range(n_comp))
    starts_t = tuple(int(s) for s in starts)

    @jax.jit
    def _impl(x, Bvals, W_split, gather_idx, seg_idx):
        Xc = [x[starts_t[c]:starts_t[c + 1]] for c in range(n_comp)]
        out_parts = []
        for cr in range(n_comp):
            acc = jnp.zeros((nseg[cr],), dtype=x.dtype)
            for cc in range(n_comp):
                if (cr, cc) not in W_split:
                    continue
                y_local = _element_apply(
                    Bvals[cr], Bvals[cc], W_split[(cr, cc)],
                    Xc[cc], gather_idx[cc])
                acc = acc + jax.ops.segment_sum(
                    y_local.reshape(-1), seg_idx[cr], num_segments=nseg[cr])
            out_parts.append(acc)
        return jnp.concatenate(out_parts)

    def apply(x):
        return _impl(x, Bvals, W_split, gather_idx, seg_idx)

    return apply


def _mass_components(operators: SequenceOperators, k: int):
    try:
        diaginv = get_mass_jacobi_diaginv(operators.mass_preconds, k, False)
        diaginv_dbc = get_mass_jacobi_diaginv(operators.mass_preconds, k, True)
    except ValueError:
        diaginv = None
        diaginv_dbc = None
    match k:
        case 0:
            return operators.m0, diaginv, diaginv_dbc
        case 1:
            return operators.m1, diaginv, diaginv_dbc
        case 2:
            return operators.m2, diaginv, diaginv_dbc
        case 3:
            return operators.m3, diaginv, diaginv_dbc
    raise ValueError("k must be 0, 1, 2 or 3")


def mass_core_apply(seq, operators: SequenceOperators, k: int):
    """Return a raw-DOF-space callable ``x -> M_k @ x``.

    The returned callable acts in the unextracted tensor-product DOF space and
    is evaluated matrix-free: the sum-factorized kernel never materializes
    ``M_k``, removing the high-(n, p) storage bottleneck (notably for M1). The
    element plan is built once per geometry and cached on ``seq``.
    """
    del operators  # mass apply no longer reads the stored BCSR matrix
    return _matrixfree_mass_apply_cached(seq, k)


def _matrixfree_mass_apply_cached(seq, k: int):
    """Build (and cache on ``seq``) the matrix-free ``M_k`` apply.

    The element plan inside :func:`build_matrixfree_mass_apply` is host-built
    and reused across matvecs, so it must be constructed once rather than per
    apply. The cache is keyed by the current geometry object so that re-mapping
    the sequence (``set_map``) transparently rebuilds the plan.
    """
    geometry = seq.geometry
    cache = getattr(seq, "_matrixfree_mass_apply_cache", None)
    if cache is None:
        cache = {}
        seq._matrixfree_mass_apply_cache = cache
    entry = cache.get(k)
    if entry is not None and entry[0] is geometry:
        return entry[1]
    apply = build_matrixfree_mass_apply(seq, k, geometry)
    cache[k] = (geometry, apply)
    return apply


def _mass_diaginv(seq, operators: SequenceOperators, k: int, dirichlet: bool):
    del seq
    _, diaginv, diaginv_dbc = _mass_components(operators, k)
    selected = diaginv_dbc if dirichlet else diaginv
    if selected is None:
        side = "dbc" if dirichlet else "free"
        raise ValueError(
            f"Jacobi mass diagonal for k={k} ({side}) is not assembled. "
            "Call assemble_mass_jacobi_preconditioner(...) during operator assembly."
        )
    return selected


def _laplacian_components(operators: SequenceOperators, k: int):
    match k:
        case 0:
            return operators.grad_grad, operators.laplacian0_diaginv, operators.laplacian0_diaginv_dbc
        case 1:
            return operators.curl_curl, operators.laplacian1_diaginv, operators.laplacian1_diaginv_dbc
        case 2:
            return operators.div_div, operators.laplacian2_diaginv, operators.laplacian2_diaginv_dbc
        case 3:
            return None, operators.laplacian3_diaginv, operators.laplacian3_diaginv_dbc
    raise ValueError("k must be 0, 1, 2 or 3")


def _laplacian_diaginv(seq, operators: SequenceOperators, k: int, dirichlet: bool):
    _, diaginv, diaginv_dbc = _laplacian_components(operators, k)
    selected = diaginv_dbc if dirichlet else diaginv
    if selected is None:
        if k in (0, 1, 2, 3):
            # The Hodge Jacobi diagonal (a comparison preconditioner; the
            # tensor model is the production path) is no longer assembled
            # eagerly. With the incidence ``G_k`` matrix-free and the mass
            # ``M_{k+1}`` stored nowhere, the direct entry-scatter form is
            # unavailable, so probe the matrix-free Hodge-Laplacian apply
            # ``L_k`` sequentially with unit vectors to recover its diagonal.
            # Returns the inverse diagonal.
            suffix = "_dbc" if dirichlet else ""
            size = int(getattr(seq, f"n{k}{suffix}"))
            diag = _diagonal_from_matvec(
                lambda x: apply_hodge_laplacian_approx(
                    seq, operators, x, k, dirichlet=dirichlet),
                size,
            )
            selected = _invert_diagonal(diag)
        else:
            raise ValueError(f"Laplacian preconditioner k={k} is not assembled")
    return selected


# Backward-compatible aliases during naming transition.
def _hodge_components(operators: SequenceOperators, k: int):
    return _laplacian_components(operators, k)


def _hodge_diaginv(seq, operators: SequenceOperators, k: int, dirichlet: bool):
    return _laplacian_diaginv(seq, operators, k, dirichlet)


def _derivative_components(operators: SequenceOperators, k: int):
    match k:
        case 0:
            return operators.d0, operators.d0_T
        case 1:
            return operators.d1, operators.d1_T
        case 2:
            return operators.d2, operators.d2_T
    raise ValueError("k must be 0, 1 or 2")


def _projection_components(operators: SequenceOperators, k_in: int, k_out: int):
    match (k_in, k_out):
        case (2, 1):
            return operators.p21
        case (1, 2):
            return operators.p12
        case (0, 3):
            return operators.p03
        case (3, 0):
            return operators.p30
    raise ValueError(
        "Only (k_in, k_out) = (1, 2), (2, 1), (0, 3), or (3, 0) supported"
    )


# TODO: remove (no-op validator; will be unneeded once assembly files are gone)
def _assemble_derivative_block(seq, operators: SequenceOperators, k: int):
    """Validate that G_k is available; returns (None, None).

    M_{k+1} is applied matrix-free via :func:`mass_core_apply`, which does not
    read ``operators.m{k+1}``, so no BCSR mass matrix needs to be assembled.
    """
    g_sp, _ = _incidence_components(operators, k)
    if g_sp is None:
        raise ValueError(
            f"Incidence operator G{k} is required to apply D{k}")
    return None, None


# TODO: remove (no-op validator path; G_k assembly is the only real work)
def update_derivative_operator(seq, geometry, operators: Optional[SequenceOperators], k: int):
    """Ensure the k-th incidence G_k is assembled (D_k is applied lazily)."""
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


def _incidence_forward_bcoo(k: int, types, s0, s1, s2, s3):
    """Build the forward topological derivative ``Dk`` as a BCOO matrix.

    The 3-D incidence operators decompose into rank-1 Kronecker blocks:
    for a derivative in axis ``d``, the block is ``I ⊗ ... ⊗ G_d ⊗ ... ⊗ I``
    where the non-``d`` identity factors have sizes equal to the *input*
    component's shape in those axes (which must match the output).

    This is the slow, materialised reference used only for lazy ``to_bcoo`` /
    ``todense`` on :class:`_MatrixFreeIncidence` (debug + retired diagonal
    paths). The production solve uses the matrix-free apply instead.
    """
    G_1d = {
        0: _incidence_1d_coo(s0[0], types[0])[:3],
        1: _incidence_1d_coo(s0[1], types[1])[:3],
        2: _incidence_1d_coo(s0[2], types[2])[:3],
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

    s1_r, s1_t, s1_z = s1
    s2_r, s2_t, s2_z = s2

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


# ---------------------------------------------------------------------------
# Matrix-free topological incidence (G0/G1/G2 and transposes)
#
# The incidence is a {-1, 0, +1} difference stencil, so it never needs to be
# stored. In non-flattened (tensor) form the apply is just per-axis forward
# differences (grad/curl/div) or their adjoints, which makes the zero structure
# explicit. ``_MatrixFreeIncidence`` carries only static shape metadata and
# applies via reshape + difference; ``to_bcoo``/``todense`` rebuild the sparse
# matrix lazily for the debug/reporting paths.
# ---------------------------------------------------------------------------

def _diff_fwd(V, axis: int, typ: str):
    """Forward 1-D incidence (discrete derivative) along ``axis``.

    ``clamped``: ``(G c)_j = c_{j+1} - c_j`` (size shrinks by one);
    ``periodic``: ``c_{(j+1) mod n} - c_j`` (size preserved);
    ``constant``: derivative of a constant is zero (size preserved).
    """
    if typ == 'clamped':
        return jnp.diff(V, axis=axis)
    if typ == 'periodic':
        return jnp.roll(V, -1, axis=axis) - V
    if typ == 'constant':
        return jnp.zeros_like(V)
    raise ValueError(f"Unknown basis type {typ!r}")


def _diff_adj(Y, axis: int, typ: str):
    """Adjoint of :func:`_diff_fwd` along ``axis`` (transpose incidence)."""
    if typ == 'clamped':
        pad_end = [(0, 0)] * Y.ndim
        pad_end[axis] = (0, 1)
        pad_start = [(0, 0)] * Y.ndim
        pad_start[axis] = (1, 0)
        return jnp.pad(-Y, pad_end) + jnp.pad(Y, pad_start)
    if typ == 'periodic':
        return jnp.roll(Y, 1, axis=axis) - Y
    if typ == 'constant':
        return jnp.zeros_like(Y)
    raise ValueError(f"Unknown basis type {typ!r}")


def _prod3(shape) -> int:
    return int(shape[0] * shape[1] * shape[2])


def _split3(x, shapes):
    """Split a flat vector into three 3-D component arrays of ``shapes``."""
    n0 = _prod3(shapes[0])
    n1 = _prod3(shapes[1])
    a = x[:n0].reshape(shapes[0])
    b = x[n0:n0 + n1].reshape(shapes[1])
    c = x[n0 + n1:].reshape(shapes[2])
    return a, b, c


def _apply_incidence_mf(op, x):
    """Apply a :class:`_MatrixFreeIncidence` operator to flat vector ``x``."""
    types = op.types
    tr, tt, tz = types
    s0, s1, s2, s3 = op.s0, op.s1, op.s2, op.s3
    s1_r, s1_t, s1_z = s1
    s2_r, s2_t, s2_z = s2

    if op.k == 0 and not op.transpose:
        # G0 grad: 0-form -> (d_r, d_t, d_z).
        V = x.reshape(s0)
        return jnp.concatenate([
            _diff_fwd(V, 0, tr).ravel(),
            _diff_fwd(V, 1, tt).ravel(),
            _diff_fwd(V, 2, tz).ravel(),
        ])
    if op.k == 0 and op.transpose:
        a, b, c = _split3(x, s1)
        out = (_diff_adj(a, 0, tr)
               + _diff_adj(b, 1, tt)
               + _diff_adj(c, 2, tz))
        return out.ravel()

    if op.k == 1 and not op.transpose:
        # G1 curl: (a, b, c) -> (P, Q, R).
        a, b, c = _split3(x, s1)
        P = -_diff_fwd(b, 2, tz) + _diff_fwd(c, 1, tt)
        Q = _diff_fwd(a, 2, tz) - _diff_fwd(c, 0, tr)
        R = -_diff_fwd(a, 1, tt) + _diff_fwd(b, 0, tr)
        return jnp.concatenate([P.ravel(), Q.ravel(), R.ravel()])
    if op.k == 1 and op.transpose:
        P, Q, R = _split3(x, s2)
        a = _diff_adj(Q, 2, tz) - _diff_adj(R, 1, tt)
        b = -_diff_adj(P, 2, tz) + _diff_adj(R, 0, tr)
        c = _diff_adj(P, 1, tt) - _diff_adj(Q, 0, tr)
        return jnp.concatenate([a.ravel(), b.ravel(), c.ravel()])

    if op.k == 2 and not op.transpose:
        # G2 div: (a, b, c) -> d_r a + d_t b + d_z c.
        a, b, c = _split3(x, s2)
        out = (_diff_fwd(a, 0, tr)
               + _diff_fwd(b, 1, tt)
               + _diff_fwd(c, 2, tz))
        return out.ravel()
    if op.k == 2 and op.transpose:
        Y = x.reshape(s3)
        return jnp.concatenate([
            _diff_adj(Y, 0, tr).ravel(),
            _diff_adj(Y, 1, tt).ravel(),
            _diff_adj(Y, 2, tz).ravel(),
        ])
    raise ValueError(f"Unsupported incidence apply (k={op.k}, transpose={op.transpose})")


class _MatrixFreeIncidence(eqx.Module):
    """Lazy {-1,0,+1} incidence operator applied as a difference stencil.

    Carries only static shape metadata (no stored matrix). Supports the matvec
    protocol (``@`` / ``__call__``) used throughout the solve path, plus lazy
    ``to_bcoo`` / ``todense`` for the debug/reporting helpers that still want a
    materialised matrix.
    """
    k: int = eqx.field(static=True)
    transpose: bool = eqx.field(static=True)
    types: tuple = eqx.field(static=True)
    s0: tuple = eqx.field(static=True)
    s1: tuple = eqx.field(static=True)
    s2: tuple = eqx.field(static=True)
    s3: tuple = eqx.field(static=True)
    shape: tuple = eqx.field(static=True)

    def __matmul__(self, x):
        return _apply_incidence_mf(self, x)

    def __call__(self, x):
        return _apply_incidence_mf(self, x)

    @property
    def T(self):
        return _MatrixFreeIncidence(
            k=self.k,
            transpose=not self.transpose,
            types=self.types,
            s0=self.s0, s1=self.s1, s2=self.s2, s3=self.s3,
            shape=(self.shape[1], self.shape[0]),
        )

    def to_bcoo(self):
        sp, sp_T = _incidence_forward_bcoo(
            self.k, self.types, self.s0, self.s1, self.s2, self.s3)
        chosen = sp_T if self.transpose else sp
        return chosen.to_bcoo()

    def todense(self):
        return self.to_bcoo().todense()


def _incidence_shapes(seq):
    """Return the four DoF shape groups ``(s0, s1, s2, s3)`` for ``seq``."""
    s0 = tuple(int(v) for v in seq.basis_0.shape[0])
    s3 = tuple(int(v) for v in seq.basis_3.shape[0])
    s1 = tuple(tuple(int(v) for v in comp) for comp in seq.basis_1.shape)
    s2 = tuple(tuple(int(v) for v in comp) for comp in seq.basis_2.shape)
    return s0, s1, s2, s3


def _build_matrixfree_incidence(seq, k: int):
    """Return ``(Gk, Gk_T)`` as matrix-free incidence operators."""
    types = tuple(seq.basis_0.types)
    s0, s1, s2, s3 = _incidence_shapes(seq)
    if k == 0:
        n_in = _prod3(s0)
        n_out = sum(_prod3(c) for c in s1)
    elif k == 1:
        n_in = sum(_prod3(c) for c in s1)
        n_out = sum(_prod3(c) for c in s2)
    elif k == 2:
        n_in = sum(_prod3(c) for c in s2)
        n_out = _prod3(s3)
    else:
        raise ValueError("k must be 0, 1 or 2")
    common = dict(k=k, types=types, s0=s0, s1=s1, s2=s2, s3=s3)
    g = _MatrixFreeIncidence(transpose=False, shape=(n_out, n_in), **common)
    g_T = _MatrixFreeIncidence(transpose=True, shape=(n_in, n_out), **common)
    return g, g_T


def update_incidence_operator(seq, operators: Optional[SequenceOperators], k: int):
    """Return an operator bundle with the k-th topological incidence updated."""
    sp, sp_T = _build_matrixfree_incidence(seq, k)
    operators = _ensure_extraction_operators(seq, operators)

    match k:
        case 0:
            return eqx.tree_at(
                lambda ops: (ops.g0, ops.g0_T),
                operators,
                (sp, sp_T),
                is_leaf=lambda x: x is None,
            )
        case 1:
            return eqx.tree_at(
                lambda ops: (ops.g1, ops.g1_T),
                operators,
                (sp, sp_T),
                is_leaf=lambda x: x is None,
            )
        case 2:
            return eqx.tree_at(
                lambda ops: (ops.g2, ops.g2_T),
                operators,
                (sp, sp_T),
                is_leaf=lambda x: x is None,
            )
    raise ValueError("k must be 0, 1 or 2")


def build_grad_stencil_g0(seq, xi, dirichlet_in: bool, dirichlet_out: bool):
    """Analytic, INVERSE-FREE polar discrete gradient ``G_0`` (V0 -> V1).

    Builds the true strong gradient on extracted DoFs as an explicit sparse
    matrix straight from the incidence pattern and the polar mapping coefficients
    ``xi`` (shape ``(3, 2, nt)``) -- coefficient differences and ``xi`` weights
    only, NO mass and NO matrix inverse. This is the closed form of
    ``Gram_1^{-1} (E_1 sp_0 E_0^T)``; the axis-fusion inverse cancels to clean
    ``+/-1`` / ``-xi[l,1,j]`` stencils (verified bit-exact vs that oracle).

    Layout (see ``extraction_operators.build_extraction`` k=0/k=1 branches):
    V0 extracted = apex ``(p,m) -> p*nz+m`` (p in 0..2) then bulk
    ``(i,j,k) -> 3 nz + ravel((i,j,k),(radial0,nt,nz))`` with full radial ``i+2``.
    V1 extracted = theta_surgery ``[0,2 nz)`` | zeta_surgery ``[2 nz, 2 nz+3 dz)``
    | r-slice (comp0) | theta_bulk (comp1) | zeta_bulk (comp2). The full-space
    grad is ``d_r f``, ``d_theta f`` (periodic), ``d_z f`` (periodic), with the
    near-axis full radial rows 0/1 expanded as ``f(0,j,k)=sum_p xi[p,0,j] apex``,
    ``f(1,j,k)=sum_p xi[p,1,j] apex``.
    """
    import scipy.sparse as _sps
    xi = np.asarray(xi)
    nr, nt, nz = (int(v) for v in seq.basis_0.shape[0])
    dr = nr - 1            # clamped r derivative count
    dt, dz = nt, nz        # periodic theta, z -> derivative count == primal
    o0 = 1 if dirichlet_in else 0
    o1 = 1 if dirichlet_out else 0
    radial0 = nr - 2 - o0  # V0 bulk radial rings (full radial >= 2)
    radial1 = nr - 2 - o1  # V1 comp1/comp2 bulk radial rings

    base_bulk0 = 3 * nz

    def c_bulk0(i, j, k):           # V0 bulk col for full radial i+2, or None
        if i < 0 or i >= radial0:
            return None
        return base_bulk0 + (i * nt + j) * nz + k

    def expand(a, j, k):           # full V0 (a,j,k) -> list of (v0_col, weight)
        if a == 0:
            return [(p * nz + k, float(xi[p, 0, j])) for p in range(3)]
        if a == 1:
            return [(p * nz + k, float(xi[p, 1, j])) for p in range(3)]
        col = c_bulk0(a - 2, j, k)
        return [(col, 1.0)] if col is not None else []

    # V1 extracted row offsets (must match _k1_row_slices with o == o1).
    r_theta_s = 0
    r_zeta_s = 2 * nz
    r_r = 2 * nz + 3 * dz
    r_theta_b = r_r + (dr - 1) * nt * nz
    r_zeta_b = r_theta_b + radial1 * dt * nz

    rows, cols, data = [], [], []

    def add(r, terms):
        for c, w in terms:
            if c is None or w == 0.0:
                continue
            rows.append(r)
            cols.append(c)
            data.append(w)

    # theta_surgery: apex difference  apex(p_local+1, m) - apex(0, m)
    for pl in range(2):
        p = pl + 1
        for m in range(nz):
            add(r_theta_s + pl * nz + m,
                [(p * nz + m, 1.0), (0 * nz + m, -1.0)])

    # zeta_surgery: periodic z-difference of the apex DoFs
    for p in range(3):
        for m in range(dz):
            add(r_zeta_s + p * dz + m,
                [(p * nz + (m + 1) % nz, 1.0), (p * nz + m, -1.0)])

    # r-slice (comp0, radial grad):  full(i+2,j,k) - full(i+1,j,k)
    for i in range(dr - 1):
        for j in range(nt):
            for k in range(nz):
                r = r_r + (i * nt + j) * nz + k
                add(r, expand(i + 2, j, k)
                    + [(c, -w) for c, w in expand(i + 1, j, k)])

    # theta_bulk (comp1, angular grad, periodic):  full(i+2,j+1) - full(i+2,j)
    for i in range(radial1):
        for j in range(dt):
            for k in range(nz):
                r = r_theta_b + (i * dt + j) * nz + k
                add(r, expand(i + 2, (j + 1) % nt, k)
                    + [(c, -w) for c, w in expand(i + 2, j, k)])

    # zeta_bulk (comp2, z grad, periodic):  full(i+2,k+1) - full(i+2,k)
    for i in range(radial1):
        for j in range(nt):
            for k in range(dz):
                r = r_zeta_b + (i * nt + j) * dz + k
                add(r, expand(i + 2, j, (k + 1) % nz)
                    + [(c, -w) for c, w in expand(i + 2, j, k)])

    n0 = int(seq.n0_dbc if dirichlet_in else seq.n0)
    n1 = int(seq.n1_dbc if dirichlet_out else seq.n1)
    coo = _sps.coo_matrix(
        (np.asarray(data, dtype=np.float64),
         (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32))),
        shape=(n1, n0)).tocsr()
    coo.sum_duplicates()
    bcoo = jsparse.BCOO(
        (jnp.asarray(coo.data),
         jnp.asarray(np.stack([coo.tocoo().row, coo.tocoo().col], axis=1))),
        shape=(n1, n0))
    return jsparse.BCSR.from_bcoo(bcoo)


def build_curl_stencil_g1(seq, xi, dirichlet_in: bool, dirichlet_out: bool):
    """Analytic, INVERSE-FREE polar discrete curl ``G_1`` (V1 -> V2).

    The degree-1 analog of :func:`build_grad_stencil_g0`: the true strong curl on
    extracted DoFs as an explicit sparse matrix from the incidence pattern and the
    polar coefficients ``xi`` (shape ``(3, 2, nt)``) -- coefficient differences and
    ``xi`` weights only, NO mass and NO matrix inverse. The closed form of
    ``Gram_2^{-1} (E_2 sp_1 E_1^T)``; the V2 axis-fusion inverse cancels to clean
    ``+/-1`` / ``xi``-difference stencils (verified bit-exact vs that oracle).

    Full-space curl (a=s, b=chi, c=zeta -> V2 comps P,Q,R; see ``_apply_incidence_mf``):
    ``P=-d_z b + d_t c``, ``Q=d_z a - d_r c``, ``R=-d_t a + d_r b``. V1 input fusion
    is inverted by ``expand_v1`` (the V1 analog of grad's ``expand``); the only fused
    V2 *output* DoFs are the comp0 surgery rows, whose stencil is the axis form of
    ``P = -d_z(chi apex) + d_t(zeta apex)``.
    """
    import scipy.sparse as _sps
    xi = np.asarray(xi)
    nr, nt, nz = (int(v) for v in seq.basis_0.shape[0])
    dr, dt, dz = nr - 1, nt, nz
    o_in = 1 if dirichlet_in else 0
    o_out = 1 if dirichlet_out else 0
    radial_in = nr - 2 - o_in
    radial_out = nr - 2 - o_out

    # --- V1 extracted (input) columns + fusion-inverting expand ---
    base_r1 = 2 * nz + 3 * dz
    base_tb1 = base_r1 + (dr - 1) * nt * nz
    base_zb1 = base_tb1 + radial_in * dt * nz

    def c_ths(pl, m):                                  # V1 theta_surgery col
        return pl * nz + m
    def c_zes(p, m):                                   # V1 zeta_surgery col
        return 2 * nz + p * dz + m
    def c_r1(i, j, k):                                 # V1 comp0 r-slice
        return None if (i < 0 or i >= dr - 1) else base_r1 + (i * nt + j) * nz + k
    def c_tb1(i, j, k):                                # V1 comp1 theta_bulk
        return None if (i < 0 or i >= radial_in) else base_tb1 + (i * dt + j) * nz + k
    def c_zb1(i, j, k):                                # V1 comp2 zeta_bulk
        return None if (i < 0 or i >= radial_in) else base_zb1 + (i * nt + j) * dz + k

    def expand_v1(comp, a, j, k):  # full V1 (comp,a,j,k) -> [(v1_col, weight)]
        if comp == 0:                                  # s, full radial a in [0,dr)
            if a == 0:
                return [(c_ths(pl, k), float(xi[pl + 1, 1, j] - xi[pl + 1, 0, j]))
                        for pl in range(2)]
            c = c_r1(a - 1, j, k)
            return [(c, 1.0)] if c is not None else []
        if comp == 1:                                  # chi, full radial a in [0,nr)
            if a == 1:
                return [(c_ths(pl, k), float(xi[pl + 1, 1, (j + 1) % dt] - xi[pl + 1, 1, j]))
                        for pl in range(2)]
            c = c_tb1(a - 2, j, k)
            return [(c, 1.0)] if c is not None else []
        # comp == 2: zeta, full radial a in [0,nr)
        if a == 0:
            return [(c_zes(p, k), float(xi[p, 0, j])) for p in range(3)]
        if a == 1:
            return [(c_zes(p, k), float(xi[p, 1, j])) for p in range(3)]
        c = c_zb1(a - 2, j, k)
        return [(c, 1.0)] if c is not None else []

    # --- V2 extracted (output) row offsets (match build_extraction k==2) ---
    n1_v2 = (radial_out * dt + 2) * dz   # comp0 extracted size (2dz surgery + bulk)
    n2_v2 = (dr - 1) * nt * dz           # comp1 extracted size
    r_c0b = 2 * dz                       # comp0 bulk start
    r_c1 = n1_v2                         # comp1 bulk start
    r_c2 = n1_v2 + n2_v2                 # comp2 bulk start

    rows, cols, data = [], [], []

    def add(r, terms):
        for c, w in terms:
            if c is None or w == 0.0:
                continue
            rows.append(r)
            cols.append(c)
            data.append(w)

    def scaled(terms, s):
        return [(c, s * w) for c, w in terms]

    # comp0 surgery [0,2dz): P axis = -d_z(chi apex) + (zeta apex difference)
    for pl in range(2):
        p = pl + 1
        for m in range(dz):
            add(pl * dz + m,
                [(c_ths(pl, m), 1.0), (c_ths(pl, (m + 1) % dz), -1.0),
                 (c_zes(p, m), 1.0), (c_zes(0, m), -1.0)])

    # comp0 bulk: P[i+2,j,k] = -d_z(chi) + d_t(zeta)
    for i in range(radial_out):
        for j in range(dt):
            for k in range(dz):
                add(r_c0b + (i * dt + j) * dz + k,
                    scaled(expand_v1(1, i + 2, j, (k + 1) % nz), -1) + expand_v1(1, i + 2, j, k)
                    + expand_v1(2, i + 2, (j + 1) % nt, k) + scaled(expand_v1(2, i + 2, j, k), -1))

    # comp1 bulk: Q[i+1,j,k] = d_z(s) - d_r(zeta)
    for i in range(dr - 1):
        for j in range(nt):
            for k in range(dz):
                add(r_c1 + (i * nt + j) * dz + k,
                    expand_v1(0, i + 1, j, (k + 1) % nz) + scaled(expand_v1(0, i + 1, j, k), -1)
                    + scaled(expand_v1(2, i + 2, j, k), -1) + expand_v1(2, i + 1, j, k))

    # comp2 bulk: R[i+1,j,k] = -d_t(s) + d_r(chi)
    for i in range(dr - 1):
        for j in range(dt):
            for k in range(nz):
                add(r_c2 + (i * dt + j) * nz + k,
                    scaled(expand_v1(0, i + 1, (j + 1) % nt, k), -1) + expand_v1(0, i + 1, j, k)
                    + expand_v1(1, i + 2, j, k) + scaled(expand_v1(1, i + 1, j, k), -1))

    n1 = int(seq.n1_dbc if dirichlet_in else seq.n1)
    n2 = int(seq.n2_dbc if dirichlet_out else seq.n2)
    coo = _sps.coo_matrix(
        (np.asarray(data, dtype=np.float64),
         (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32))),
        shape=(n2, n1)).tocsr()
    coo.sum_duplicates()
    coo = coo.tocoo()
    bcoo = jsparse.BCOO(
        (jnp.asarray(coo.data),
         jnp.asarray(np.stack([coo.row, coo.col], axis=1))),
        shape=(n2, n1))
    return jsparse.BCSR.from_bcoo(bcoo)


def _build_inc_gram_inv(seq, operators, space: int, dirichlet: bool):
    """Sparse ``(E_space^T E_space)^{-1}`` for the TRUE polar-derivative fix.

    The true strong derivative on extracted DoFs is
    ``G = M^{-1} D = Gram_{k+1}^{-1} (E^T sp E)`` with the (mass-free) coefficient
    Gram ``Gram = E^T E``. ``Gram`` is the identity in the bulk plus a small dense
    block on the polar-axis fusion DoFs, so its inverse is sparse (identity +
    that block). Returns ``None`` when the extraction is unitary (``Gram = I``,
    e.g. non-polar or the top space V3) -- then no correction is needed.
    """
    import scipy.sparse as _sps
    e, e_T = _mass_extraction(operators, space, dirichlet)
    if e is None or e_T is None:
        return None
    n_ext = int(getattr(seq, f"n{space}_dbc" if dirichlet else f"n{space}"))
    # Build the (ext x full) restriction E as a SPARSE matrix straight from the
    # extraction's own triplets -- never densify the (ext x full) operator (that
    # is the "densify an incidence operator" madness). The coefficient Gram
    # E E^T is (ext x ext) and stays sparse; only the small polar-axis fusion
    # block is touched densely (to invert), exactly the "small dense ops near the
    # axis" worst case.
    bcoo = e.to_bcoo()
    idx = np.asarray(bcoo.indices)
    dat = np.asarray(bcoo.data)
    shp = tuple(int(s) for s in bcoo.shape)
    E = _sps.coo_matrix((dat, (idx[:, 0], idx[:, 1])), shape=shp).tocsr()
    if E.shape[0] != n_ext and E.shape[1] == n_ext:
        E = E.T.tocsr()
    n = E.shape[0]
    gram = (E @ E.T).tocsr()                          # (ext, ext) coeff Gram, sparse
    diff = (gram - _sps.identity(n, format="csr", dtype=gram.dtype)).tocoo()
    mask = np.abs(diff.data) > 1e-10
    S = np.unique(diff.row[mask]) if mask.any() else np.array([], dtype=int)
    if S.size == 0:
        return None                                  # unitary extraction
    gram_ss_inv = np.linalg.inv(np.asarray(gram[np.ix_(S, S)].todense()))
    in_S = np.zeros(n, dtype=bool)
    in_S[S] = True
    bulk = np.where(~in_S)[0]
    II, JJ = np.meshgrid(S, S, indexing="ij")
    rows = np.concatenate([bulk, II.ravel()]).astype(np.int32)
    cols = np.concatenate([bulk, JJ.ravel()]).astype(np.int32)
    data = np.concatenate([np.ones(bulk.size), gram_ss_inv.ravel()])
    bcoo = jsparse.BCOO((jnp.asarray(data), jnp.asarray(np.stack([rows, cols], axis=1))),
                        shape=(n, n))
    return jsparse.BCSR.from_bcoo(bcoo)


def _grad_stencil(operators: SequenceOperators, dirichlet_in: bool,
                  dirichlet_out: bool, transpose: bool):
    """Look up the analytic inverse-free polar grad ``G_0`` (or None on non-polar)."""
    name = f"g0_grad_{int(dirichlet_in)}{int(dirichlet_out)}"
    if transpose:
        name += "_T"
    return getattr(operators, name, None)


def _curl_stencil(operators: SequenceOperators, dirichlet_in: bool,
                  dirichlet_out: bool, transpose: bool):
    """Look up the analytic inverse-free polar curl ``G_1`` (or None on non-polar)."""
    name = f"g1_curl_{int(dirichlet_in)}{int(dirichlet_out)}"
    if transpose:
        name += "_T"
    return getattr(operators, name, None)


def _inc_gram_inv(operators: SequenceOperators, space: int, dirichlet: bool):
    """Look up the stored true-derivative Gram^{-1} correction (or None).

    Retained for the k=2 (div) apply path -- V3 is unitary so this is always
    ``None`` (-> raw incidence = true div). grad/curl use the analytic stencils;
    the stored Gram inverses are no longer precomputed (see _extraction_is_polar).
    """
    if not (1 <= space <= 3):
        return None
    name = f"inc_gram_inv_{space}" + ("_dbc" if dirichlet else "")
    return getattr(operators, name, None)


def _extraction_is_polar(operators: SequenceOperators, space: int) -> bool:
    """True iff the extraction of ``space`` is non-unitary (polar axis fusion).

    Gram-free polar signal: tests ``E E^T x != x`` on one probe (E E^T = I on the
    0/1 non-polar/unitary extractions). Replaces the old "inc_gram_inv non-None"
    check so the analytic grad/curl stencils can be built without precomputing the
    Gram inverses at all.
    """
    e, e_T = _mass_extraction(operators, space, False)
    if e is None or e_T is None:
        return False
    n_ext = int(e.shape[0])
    x = jax.random.normal(jax.random.PRNGKey(0), (n_ext,), dtype=jnp.float64)
    return bool(jnp.max(jnp.abs(e @ (e_T @ x) - x)) > 1e-10)


def assemble_incidence_operators(seq, operators: Optional[SequenceOperators] = None,
                                 ks: Sequence[int] = (0, 1, 2)):
    """Assemble topological incidence operators for the requested degrees.

    Also caches the TRUE polar-derivative Gram^{-1} corrections for the output
    spaces ``{k+1}`` so :func:`apply_incidence_matrix` returns the strong
    derivative ``M^{-1} D`` (exact ``d.d = 0`` on extracted DoFs) on polar
    sequences, while staying bit-identical to the raw incidence elsewhere.
    """
    for k in ks:
        operators = update_incidence_operator(seq, operators, k)
    operators = _ensure_extraction_operators(seq, operators)

    # The true polar derivative is realized by the analytic inverse-free stencils
    # below (grad G_0 for k=0, curl G_1 for k=1); div G_2 (output V3) needs no
    # correction (V3 extraction is unitary). The old per-operator Gram^{-1}
    # precompute (`_build_inc_gram_inv` -> inc_gram_inv_*) is therefore NOT built;
    # `_build_inc_gram_inv` is kept only as an independent oracle for the
    # validation harness, and `_inc_gram_inv` stays None (-> raw div apply).

    # Analytic inverse-free polar grad G_0 (replaces the Gram^{-1} path for k=0),
    # built when grad is requested (0 in ks) on a polar sequence (V1 extraction
    # non-unitary). Stored per BC pair, forward + transpose; bit-exact with Gram.
    polar = _extraction_is_polar(operators, 1)
    if 0 in ks and polar and operators.g0_grad_00 is None:
        xi = get_xi(seq.ns[1])
        gfields, gvals = [], []
        for din in (False, True):
            for dout in (False, True):
                g0 = build_grad_stencil_g0(seq, xi, din, dout)
                base = f"g0_grad_{int(din)}{int(dout)}"
                gfields += [base, base + "_T"]
                gvals += [g0, jsparse.BCSR.from_bcoo(g0.to_bcoo().T)]
        operators = eqx.tree_at(
            lambda o: tuple(getattr(o, f) for f in gfields),
            operators, tuple(gvals),
            is_leaf=lambda x: x is None,
        )

    # Analytic inverse-free polar curl G_1 (replaces the Gram^{-1} path for k=1),
    # built when curl is requested (1 in ks) on a polar sequence. Div (k=2, output
    # V3) needs no stencil: the V3 extraction is a 0/1 selection (Gram_3 = I), so
    # apply_incidence(.,2) is already the true div.
    polar2 = _extraction_is_polar(operators, 2)
    if 1 in ks and polar2 and operators.g1_curl_00 is None:
        xi = get_xi(seq.ns[1])
        cfields, cvals = [], []
        for din in (False, True):
            for dout in (False, True):
                g1 = build_curl_stencil_g1(seq, xi, din, dout)
                base = f"g1_curl_{int(din)}{int(dout)}"
                cfields += [base, base + "_T"]
                cvals += [g1, jsparse.BCSR.from_bcoo(g1.to_bcoo().T)]
        operators = eqx.tree_at(
            lambda o: tuple(getattr(o, f) for f in cfields),
            operators, tuple(cvals),
            is_leaf=lambda x: x is None,
        )
    return operators


def _incidence_components(operators: SequenceOperators, k: int):
    match k:
        case 0:
            return operators.g0, operators.g0_T
        case 1:
            return operators.g1, operators.g1_T
        case 2:
            return operators.g2, operators.g2_T
    raise ValueError("k must be 0, 1 or 2")


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


def apply_incidence_matrix(seq, operators: SequenceOperators, v, k: int,
                           dirichlet_in: bool = True,
                           dirichlet_out: bool = True,
                           transpose: bool = False):
    """Apply the strong exterior-derivative ``G_k`` on extracted DoF spaces.

    The raw extracted incidence is ``E_out^T sp E_in`` (``sp`` has entries in
    ``{-1, 0, +1}``). On polar sequences the extraction is non-unitary at the
    axis, so the raw form is NOT the topological derivative and ``d.d != 0``.
    The true strong derivative is ``G = Gram_{k+1}^{-1} (E_out^T sp E_in)`` with
    the cached coefficient-Gram inverse (``None`` / identity where the extraction
    is unitary, e.g. non-polar). The correction is a sparse matvec localised to
    the polar-axis DoFs, so off-axis the result is bit-identical to the raw
    incidence and ``d.d = 0`` holds exactly on extracted DoFs everywhere.
    """
    # k=0 grad: use the analytic inverse-free polar stencil when available
    # (built on polar sequences). Bit-exact with the Gram form; on non-polar the
    # fields are None and we fall through to the raw incidence path below.
    if k == 0:
        g0 = _grad_stencil(operators, dirichlet_in, dirichlet_out, transpose)
        if g0 is not None:
            return g0 @ v
    # k=1 curl: analytic inverse-free polar stencil when available (polar only).
    if k == 1:
        g1 = _curl_stencil(operators, dirichlet_in, dirichlet_out, transpose)
        if g1 is not None:
            return g1 @ v

    sp, sp_T = _incidence_components(operators, k)
    if sp is None or sp_T is None:
        raise ValueError(f"Incidence operator k={k} is not assembled")
    e_in, e_in_T, e_out, e_out_T = _derivative_extraction(
        operators, k, dirichlet_in, dirichlet_out)
    gram_inv = _inc_gram_inv(operators, k + 1, dirichlet_out)

    if transpose:
        # G^T = (E^T sp E)^T Gram_{k+1}^{-1}  (Gram symmetric -> proper adjoint)
        w = gram_inv @ v if gram_inv is not None else v
        return e_in @ (sp_T @ (e_out_T @ w))
    y = e_out @ (sp @ (e_in_T @ v))
    return gram_inv @ y if gram_inv is not None else y


def _assemble_projection_block(seq, k_in: int, k_out: int):
    quad_shape = (seq.quad.ny, seq.quad.nx, seq.quad.nz)
    dR = seq.d_basis_r_jk
    dT = seq.d_basis_t_jk
    dZ = seq.d_basis_z_jk
    R = seq.basis_r_jk
    T = seq.basis_t_jk
    Z = seq.basis_z_jk

    match (k_in, k_out):
        case (2, 1) | (1, 2):
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
            sp = assemble_vectorial(
                row_terms, col_terms, W_3x3, quad_shape,
                list(seq.basis_1.shape), seq.basis_1.pr,
                col_comp_shapes=list(seq.basis_2.shape))
            return jsparse.BCSR.from_bcoo(sp if k_in == 2 else sp.T)
        case (0, 3):
            W_1x1 = seq.quad.w.reshape(-1, 1, 1)
            row_terms = [
                [(0, R, T, Z, +1)],
            ]
            col_terms = [
                [(0, dR, dT, dZ, +1)],
            ]
            sp = assemble_vectorial(
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
            sp = assemble_vectorial(
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
                lambda ops: ops.p21,
                operators,
                sp,
                is_leaf=lambda x: x is None,
            )
        case (1, 2):
            return eqx.tree_at(
                lambda ops: ops.p12,
                operators,
                sp,
                is_leaf=lambda x: x is None,
            )
        case (0, 3):
            return eqx.tree_at(
                lambda ops: ops.p03,
                operators,
                sp,
                is_leaf=lambda x: x is None,
            )
        case (3, 0):
            return eqx.tree_at(
                lambda ops: ops.p30,
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
    # which follows directly from ``D_k = M_{k+1} G_k``.  We never materialise
    # ``K_k``: it is applied as a composition of matrix-free incidence and mass
    # matvecs (see :func:`apply_stiffness` / :func:`apply_hodge_laplacian`).
    #
    # The Jacobi diagonal is likewise not assembled here. The incidence ``G_k``
    # is matrix-free and the mass ``M_{k+1}`` is stored nowhere, so the old
    # direct entry-scatter selection is unavailable. The Jacobi preconditioner
    # is only a comparison method (the tensor Hodge model is the production
    # path), so its diagonal is built lazily on demand by ``_hodge_diaginv``
    # via a sequential unit-vector probe of the matrix-free Hodge-Laplacian
    # apply. Both ``sp`` and the Jacobi diagonals are therefore ``None``.
    del geometry  # unused
    if k not in (0, 1, 2, 3):
        raise ValueError("k must be 0, 1, 2, or 3")
    return None, None, None


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
            # The k=0 Hodge-Laplacian carries no materialized stiffness and no
            # Jacobi diagonal (built lazily on demand). The tensor Laplacian
            # preconditioner is built explicitly by
            # ``assemble_tensor_laplacian_preconditioner``; only build it here
            # as a fallback when it has not already been assembled, so this
            # operator assembly stays effectively a no-op once it is present.
            operators = eqx.tree_at(
                lambda ops: (ops.grad_grad, ops.dd0_diaginv,
                             ops.dd0_diaginv_dbc),
                operators,
                (sp, diaginv, diaginv_dbc),
                is_leaf=lambda x: x is None,
            )
            if operators.k0_tensor_hodge_precond is None:
                operators = assemble_tensor_laplacian_preconditioner(
                    seq, operators, ks=(0,))
            return operators
        case 1:
            return eqx.tree_at(
                lambda ops: (ops.curl_curl, ops.dd1_diaginv,
                             ops.dd1_diaginv_dbc),
                operators,
                (sp, diaginv, diaginv_dbc),
                is_leaf=lambda x: x is None,
            )
        case 2:
            return eqx.tree_at(
                lambda ops: (ops.div_div, ops.dd2_diaginv,
                             ops.dd2_diaginv_dbc),
                operators,
                (sp, diaginv, diaginv_dbc),
                is_leaf=lambda x: x is None,
            )
        case 3:
            return eqx.tree_at(
                lambda ops: (ops.dd3_diaginv, ops.dd3_diaginv_dbc),
                operators,
                (diaginv, diaginv_dbc),
                is_leaf=lambda x: x is None,
            )
    raise ValueError("k must be 0, 1, 2, or 3")


def assemble_hodge_operators(seq, geometry, operators: Optional[SequenceOperators] = None,
                             ks: Sequence[int] = (0, 1, 2, 3)):
    """Assemble Hodge/stiffness operators for the requested form degrees."""
    for k in ks:
        operators = update_hodge_operator(seq, geometry, operators, k)
    return operators


def assemble_laplacian_operators(seq, geometry, operators: Optional[SequenceOperators] = None,
                                 ks: Sequence[int] = (0, 1, 2, 3)):
    """Alias of assemble_hodge_operators using Laplacian naming."""
    return assemble_hodge_operators(seq, geometry, operators=operators, ks=ks)


def assemble_all_operators(seq, geometry,
                           operators: Optional[SequenceOperators] = None,
                           include_preconditioners: bool = True):
    """Assemble all geometry-dependent operators.

    When ``include_preconditioners`` is true, also assemble the eager
    preconditioner payloads that back the solver-facing convenience paths.
    Set it to false when only the sparse operators are needed, for example
    for densification or direct solves.
    """
    operators = assemble_mass_operators(seq, geometry, operators=operators)
    if include_preconditioners:
        operators = assemble_tensor_mass_preconditioner(
            seq,
            operators=operators,
            ks=(0, 1, 2, 3),
            rank=3,
            cp_kwargs={
                'k0_rank': 3,
                'k1_rank': 3,
                'k2_rank': 3,
                'k3_rank': 3,
            },
        )
    operators = assemble_incidence_operators(seq, operators=operators)
    operators = assemble_derivative_operators(
        seq, geometry, operators=operators)
    operators = assemble_laplacian_operators(seq, geometry, operators=operators)
    operators = assemble_projection_operators(seq, operators=operators)
    return operators


# TODO: remove — debug/dense path only
def assemble_all_dense_operators(
        seq,
        operators: Optional[SequenceOperators] = None):
    """Materialize dense extracted operators into ``operators.dense``.

    Requires the corresponding sparse operators to already be assembled.
    This is intended as a courtesy/debugging path for dense inspection and
    direct solves, not as the default operator assembly route.
    """
    operators = _ensure_extraction_operators(seq, operators)
    dense = DenseSequenceOperators(
        m0=BoundaryConditionPair(
            free=dense_mass_matrix(seq, operators, 0, dirichlet=False),
            dbc=dense_mass_matrix(seq, operators, 0, dirichlet=True),
        ),
        m1=BoundaryConditionPair(
            free=dense_mass_matrix(seq, operators, 1, dirichlet=False),
            dbc=dense_mass_matrix(seq, operators, 1, dirichlet=True),
        ),
        m2=BoundaryConditionPair(
            free=dense_mass_matrix(seq, operators, 2, dirichlet=False),
            dbc=dense_mass_matrix(seq, operators, 2, dirichlet=True),
        ),
        m3=BoundaryConditionPair(
            free=dense_mass_matrix(seq, operators, 3, dirichlet=False),
            dbc=dense_mass_matrix(seq, operators, 3, dirichlet=True),
        ),
        d0=BoundaryConditionPair(
            free=dense_derivative_matrix(seq, operators, 0, dirichlet_in=False, dirichlet_out=False),
            dbc=dense_derivative_matrix(seq, operators, 0, dirichlet_in=True, dirichlet_out=True),
        ),
        d1=BoundaryConditionPair(
            free=dense_derivative_matrix(seq, operators, 1, dirichlet_in=False, dirichlet_out=False),
            dbc=dense_derivative_matrix(seq, operators, 1, dirichlet_in=True, dirichlet_out=True),
        ),
        d2=BoundaryConditionPair(
            free=dense_derivative_matrix(seq, operators, 2, dirichlet_in=False, dirichlet_out=False),
            dbc=dense_derivative_matrix(seq, operators, 2, dirichlet_in=True, dirichlet_out=True),
        ),
        s0=BoundaryConditionPair(
            free=dense_stiffness_matrix(seq, operators, 0, dirichlet=False),
            dbc=dense_stiffness_matrix(seq, operators, 0, dirichlet=True),
        ),
        s1=BoundaryConditionPair(
            free=dense_stiffness_matrix(seq, operators, 1, dirichlet=False),
            dbc=dense_stiffness_matrix(seq, operators, 1, dirichlet=True),
        ),
        s2=BoundaryConditionPair(
            free=dense_stiffness_matrix(seq, operators, 2, dirichlet=False),
            dbc=dense_stiffness_matrix(seq, operators, 2, dirichlet=True),
        ),
        s3=BoundaryConditionPair(
            free=dense_stiffness_matrix(seq, operators, 3, dirichlet=False),
            dbc=dense_stiffness_matrix(seq, operators, 3, dirichlet=True),
        ),
        l0=BoundaryConditionPair(
            free=dense_hodge_laplacian(seq, operators, 0, dirichlet=False),
            dbc=dense_hodge_laplacian(seq, operators, 0, dirichlet=True),
        ),
        l1=BoundaryConditionPair(
            free=dense_hodge_laplacian(seq, operators, 1, dirichlet=False),
            dbc=dense_hodge_laplacian(seq, operators, 1, dirichlet=True),
        ),
        l2=BoundaryConditionPair(
            free=dense_hodge_laplacian(seq, operators, 2, dirichlet=False),
            dbc=dense_hodge_laplacian(seq, operators, 2, dirichlet=True),
        ),
        l3=BoundaryConditionPair(
            free=dense_hodge_laplacian(seq, operators, 3, dirichlet=False),
            dbc=dense_hodge_laplacian(seq, operators, 3, dirichlet=True),
        ),
        p21=BoundaryConditionPair(
            free=dense_projection_matrix(seq, operators, 2, 1, dirichlet_in=False, dirichlet_out=False),
            dbc=dense_projection_matrix(seq, operators, 2, 1, dirichlet_in=True, dirichlet_out=True),
        ),
        p12=BoundaryConditionPair(
            free=dense_projection_matrix(seq, operators, 1, 2, dirichlet_in=False, dirichlet_out=False),
            dbc=dense_projection_matrix(seq, operators, 1, 2, dirichlet_in=True, dirichlet_out=True),
        ),
        p03=BoundaryConditionPair(
            free=dense_projection_matrix(seq, operators, 0, 3, dirichlet_in=False, dirichlet_out=False),
            dbc=dense_projection_matrix(seq, operators, 0, 3, dirichlet_in=True, dirichlet_out=True),
        ),
        p30=BoundaryConditionPair(
            free=dense_projection_matrix(seq, operators, 3, 0, dirichlet_in=False, dirichlet_out=False),
            dbc=dense_projection_matrix(seq, operators, 3, 0, dirichlet_in=True, dirichlet_out=True),
        ),
    )
    return eqx.tree_at(
        lambda ops: ops.dense,
        operators,
        dense,
        is_leaf=lambda x: x is None,
    )


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
        Any subset of ``("mass", "derivative", "laplacian")``;
        legacy ``"hodge"`` is also accepted.

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
    if "laplacian" in kinds or "hodge" in kinds:
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


def dense_laplacian(seq, operators: SequenceOperators, k: int,
                    dirichlet: bool = True):
    """Alias of dense_hodge_laplacian using Laplacian naming."""
    return dense_hodge_laplacian(seq, operators, k, dirichlet=dirichlet)


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
    core = mass_core_apply(seq, operators, k)
    e, e_T = _mass_extraction(operators, k, dirichlet)
    return e @ core(e_T @ v)


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

    ``D_k = M_{k+1} G_k`` is applied as a composition of matrix-free applies;
    the full ``D_k`` is never materialised.
    """
    g_sp, g_sp_T = _incidence_components(operators, k)
    if g_sp is None or g_sp_T is None:
        raise ValueError(f"Incidence operator G{k} is required to apply D{k}")
    m_apply = mass_core_apply(seq, operators, k + 1)

    e_in, e_in_T, e_out, e_out_T = _derivative_extraction(
        operators, k, dirichlet_in, dirichlet_out)

    if transpose:
        # D^T v = G^T M^T v = G^T (M v) (M is symmetric)
        return e_in @ (g_sp_T @ m_apply(e_out_T @ v))
    return e_out @ m_apply(g_sp @ (e_in_T @ v))


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
    apply = _build_mass_preconditioner_apply(
        seq,
        operators,
        k=k,
        dirichlet=dirichlet,
        preconditioner=kind,
        allow_none=False,
    )
    return apply(v)


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

    ``K_k = G_k^T M_{k+1} G_k`` is applied as a composition of matrix-free
    applies; the full ``K_k`` is never materialised.
    """
    if k == 3:
        return jnp.zeros_like(v)
    g_sp, g_sp_T = _incidence_components(operators, k)
    if g_sp is None or g_sp_T is None:
        raise ValueError(f"Incidence operator G{k} is required to apply K{k}")
    m_apply = mass_core_apply(seq, operators, k + 1)

    e, e_T = _mass_extraction(operators, k, dirichlet)
    return e @ (g_sp_T @ m_apply(g_sp @ (e_T @ v)))


def _diagonal_from_matvec(operator_apply, size: int):
    # Eager warmup: operator_apply may lazily build host-side static state
    # (e.g. matrix-free mass index plans that call np.asarray internally).
    # Under lax.map the body is traced as a scan, so those calls would see
    # tracers and raise TracerArrayConversionError.  One concrete call first
    # forces that state to be built and cached before the traced loop runs.
    operator_apply(jnp.zeros(size, dtype=jnp.float64))

    def entry(i):
        basis = jnp.zeros(size, dtype=jnp.float64).at[i].set(1.0)
        return operator_apply(basis)[i]

    return jax.lax.map(entry, jnp.arange(size))


def _invert_diagonal(diagonal):
    diagonal = jnp.asarray(diagonal, dtype=jnp.float64)
    return jnp.where(diagonal != 0.0, 1.0 / diagonal, 0.0)


def _get_schur_diaginv(operators: SequenceOperators, k: int, dirichlet: bool, mode: str):
    """Return stored Schur diaginv for ``(k, dirichlet, mode)``, or ``None``."""
    suffix = '_dbc' if dirichlet else ''
    diaginv = getattr(operators, f'schur_diaginv_k{k}{suffix}', None)
    mode_stored = getattr(operators, f'schur_diaginv_mode_k{k}{suffix}', None)
    if diaginv is None:
        return None
    # Backward-compatible fallback for older caches that predate mode tagging.
    if mode_stored is None and mode == 'tensor_probe':
        return diaginv
    if mode_stored == mode:
        return diaginv
    return None


def _set_schur_diaginv(operators: SequenceOperators, k: int, dirichlet: bool, diaginv, mode: str):
    """Return operators with Schur diaginv + mode tag for ``(k, dirichlet)`` updated."""
    suffix = '_dbc' if dirichlet else ''
    field = f'schur_diaginv_k{k}{suffix}'
    mode_field = f'schur_diaginv_mode_k{k}{suffix}'
    return eqx.tree_at(
        lambda ops: (getattr(ops, field), getattr(ops, mode_field)),
        operators,
        (diaginv, mode),
        is_leaf=lambda x: x is None,
    )


_SCHUR_DIAG_MODES = ('tensor_probe', 'exact_probe', 'diag')


def _coerce_schur_diag_mode(spec: MassPreconditionerSpec, *, context: str) -> str:
    mode = spec.schur_diag_mode
    if mode not in _SCHUR_DIAG_MODES:
        raise ValueError(
            f"{context} schur_diag_mode must be one of {_SCHUR_DIAG_MODES} "
            f"(got {mode!r})"
        )
    return mode


def _build_schur_probe_apply(
        seq, operators: SequenceOperators, *,
        k: int, dirichlet: bool, eps: float,
        mode: str,
        saddle_preconditioner: SaddlePointPreconditionerSpec):
    if mode == 'tensor_probe':
        if not _tensor_available(seq, operators, k - 1):
            raise ValueError(
                f"schur_diag_mode='tensor_probe' requires an assembled tensor "
                f"schur.inner at k={k - 1}"
            )
        return _build_schur_apply_from_saddle_preconditioner(
            seq,
            operators,
            k=k,
            dirichlet=dirichlet,
            eps=eps,
            saddle_preconditioner=saddle_preconditioner,
        )

    if mode == 'exact_probe':
        exact_lower = lambda rhs: apply_inverse_mass_matrix(
            seq,
            operators,
            rhs,
            k - 1,
            dirichlet=dirichlet,
            preconditioner='jacobi',
        )
        return _build_schur_operator_apply(
            seq,
            operators,
            k=k,
            dirichlet=dirichlet,
            eps=eps,
            inner_preconditioner_apply=exact_lower,
        )

    if mode == 'diag':
        lower_diaginv = _mass_diaginv(seq, operators, k - 1, dirichlet)
        lower_diag_apply = lambda rhs, d=lower_diaginv: d * rhs
        return _build_schur_operator_apply(
            seq,
            operators,
            k=k,
            dirichlet=dirichlet,
            eps=eps,
            inner_preconditioner_apply=lower_diag_apply,
        )

    raise ValueError(
        f"Unsupported Schur diagonal probe mode {mode!r}; "
        f"expected one of {_SCHUR_DIAG_MODES}"
    )


def _build_schur_outer_jacobi_diaginv(
        seq, operators: SequenceOperators, *,
        k: int, dirichlet: bool, eps: float,
        outer_spec: MassPreconditionerSpec,
        saddle_preconditioner: SaddlePointPreconditionerSpec,
        allow_stored_tensor_diaginv: bool):
    mode = _coerce_schur_diag_mode(
        outer_spec,
        context=f"schur.outer kind={outer_spec.kind!r}",
    )
    if allow_stored_tensor_diaginv:
        stored_diaginv = _get_schur_diaginv(operators, k, dirichlet, mode)
        if stored_diaginv is not None:
            return stored_diaginv

    if mode == 'exact_probe':
        warnings.warn(
            "schur_diag_mode='exact_probe' estimates the Schur Jacobi diagonal "
            "by repeated exact lower mass solves; this is setup-heavy and intended "
            "as a reference path",
            stacklevel=2,
        )

    probe_apply = _build_schur_probe_apply(
        seq,
        operators,
        k=k,
        dirichlet=dirichlet,
        eps=eps,
        mode=mode,
        saddle_preconditioner=saddle_preconditioner,
    )
    suffix = '_dbc' if dirichlet else ''
    size = getattr(seq, f'n{k}{suffix}')
    diagonal = _diagonal_from_matvec(probe_apply, size)
    return _invert_diagonal(diagonal)


def assemble_schur_jacobi_preconditioner(
        seq, operators: Optional[SequenceOperators] = None,
        *, ks: Sequence[int] = (1, 2, 3),
        dirichlet_variants: Optional[Sequence[bool]] = None,
    eps: float = 0.0,
    schur_diag_mode: str = 'tensor_probe') -> SequenceOperators:
    """Probe and store the approximate Schur diagonal at assembly time.

    For each (k, dirichlet) pair, builds the approximate Schur operator

        A_k(x) = S_k x + D_{k-1} B_{k-1} D_{k-1}^T x

    and probes its diagonal by O(n_k) matrix-vector products.  The
    resulting ``1/diag(A_k)`` is stored on the operator bundle so that
    the saddle-point Schur-outer Jacobi preconditioner is a cheap
    multiply at solve time rather than an O(n_k) probing scan.

    Parameters
    ----------
    seq : DeRhamSequence
    operators : SequenceOperators, optional
    ks : sequence of int
        Form degrees to assemble (must be in 1, 2, 3).
    dirichlet_variants : sequence of bool, optional
        Boundary condition variants to assemble.  Defaults to (True, False).
    eps : float
        Shift for the stiffness term; 0 gives the unshifted Schur.

    where ``B_{k-1}`` is selected by ``schur_diag_mode``:
    - ``'tensor_probe'``: tensor schur.inner inverse (default)
    - ``'exact_probe'``: exact lower inverse via CG solve
    - ``'diag'``: diagonal lower inverse ``diag(M_{k-1})^{-1}``

    ``'tensor_probe'`` requires the tensor mass preconditioner for ``k-1``
    to be assembled first.
    """
    if dirichlet_variants is None:
        dirichlet_variants = (True, False)
    operators = _ensure_extraction_operators(seq, operators)
    if schur_diag_mode not in _SCHUR_DIAG_MODES:
        raise ValueError(
            "assemble_schur_jacobi_preconditioner schur_diag_mode must be one "
            f"of {_SCHUR_DIAG_MODES} (got {schur_diag_mode!r})"
        )
    dummy_spec = SaddlePointPreconditionerSpec(
        mass=MassPreconditionerSpec(kind='tensor'),
        schur=SchurPreconditionerSpec(
            inner=MassPreconditionerSpec(kind='tensor'),
            outer=MassPreconditionerSpec(kind='none'),
        ),
    )
    for k in ks:
        if k not in (1, 2, 3):
            raise ValueError(
                f"assemble_schur_jacobi_preconditioner: k must be 1, 2, or 3 (got {k})")
        for dirichlet in dirichlet_variants:
            if schur_diag_mode == 'tensor_probe' and not _tensor_available(seq, operators, k - 1):
                raise ValueError(
                    f"tensor mass preconditioner for k={k - 1} must be assembled before "
                    f"probing the Schur diagonal for k={k}"
                )
            schur_apply = _build_schur_probe_apply(
                seq,
                operators,
                k=k,
                dirichlet=dirichlet,
                eps=eps,
                mode=schur_diag_mode,
                saddle_preconditioner=dummy_spec,
            )
            suffix = '_dbc' if dirichlet else ''
            n = getattr(seq, f'n{k}{suffix}')
            diagonal = _diagonal_from_matvec(schur_apply, n)
            diaginv = _invert_diagonal(diagonal)
            operators = _set_schur_diaginv(
                operators,
                k,
                dirichlet,
                diaginv,
                mode=schur_diag_mode,
            )
    return operators


def _build_exact_jacobi_preconditioner_apply(
        operator_apply, size: int, *, warning_context: str):
    warnings.warn(
        f"{warning_context} probes the Schur operator diagonal by repeated applies; "
        "this is intended as a setup-heavy reference path rather than a scalable default",
        stacklevel=2,
    )
    diagonal = _diagonal_from_matvec(operator_apply, size)
    diaginv = _invert_diagonal(diagonal)
    return lambda rhs, inv=diaginv: inv * rhs


def _normalize_recursive_scalar_leaf_spec(spec: MassPreconditionerSpec):
    if spec.kind == 'tensor':
        return MassPreconditionerSpec(kind='tensor')
    if spec.kind == 'none':
        if spec.smoother is None:
            return MassPreconditionerSpec(kind='none')
        return _normalize_recursive_scalar_leaf_spec(spec.smoother)

    smoother_spec = spec.smoother
    if smoother_spec is not None:
        smoother_spec = _normalize_recursive_scalar_leaf_spec(smoother_spec)
    return MassPreconditionerSpec(
        kind=spec.kind,
        steps=spec.steps,
        power_iterations=spec.power_iterations,
        damping_safety=spec.damping_safety,
        min_eig_fraction=spec.min_eig_fraction,
        lanczos_iterations=spec.lanczos_iterations,
        lanczos_max_eig_inflation=spec.lanczos_max_eig_inflation,
        lanczos_min_eig_deflation=spec.lanczos_min_eig_deflation,
        lanczos_min_eig_floor_fraction=spec.lanczos_min_eig_floor_fraction,
        smoother=smoother_spec,
    )


def _build_restricted_mass_block_operator_apply(
        seq, operators: SequenceOperators, *, k: int, dirichlet: bool,
        indices):
    suffix = '_dbc' if dirichlet else ''
    size = getattr(seq, f'n{k}{suffix}')

    def apply(block_x):
        full_x = jnp.zeros(size, dtype=block_x.dtype)
        full_x = full_x.at[indices].set(block_x)
        full_y = apply_mass_matrix(seq, operators, full_x, k, dirichlet=dirichlet)
        return full_y[indices]

    return apply


def _k0_bulk_indices_from_surgery(surgery):
    return jnp.arange(surgery.surgery_size, surgery.apply_data.size)


def _mass_surgery_coupling_applies(k: int, surgery):
    if k == 0:
        return (
            lambda rhs_s, surgery=surgery: _apply_surgery_to_bulk_coupling(surgery, rhs_s),
            lambda rhs_b, surgery=surgery: _apply_bulk_to_surgery_coupling(surgery, rhs_b),
        )
    if k == 1:
        return (
            lambda rhs_s, surgery=surgery: _apply_surgery_to_bulk_coupling(surgery, rhs_s),
            lambda rhs_b, surgery=surgery: _apply_bulk_to_surgery_coupling(surgery, rhs_b),
        )
    if k == 2:
        return (
            lambda rhs_s, surgery=surgery: _apply_surgery_to_bulk_coupling(surgery, rhs_s),
            lambda rhs_b, surgery=surgery: _apply_bulk_to_surgery_coupling(surgery, rhs_b),
        )
    raise ValueError(f"Mass surgery coupling is only implemented for k=0,1,2 (got k={k})")


def _build_scalar_leaf_preconditioner_apply(
        operator_apply, size: int, spec: MassPreconditionerSpec, *,
        jacobi_diaginv, tensor_factors=None, seed_base: int = 0):
    jacobi_apply = lambda rhs, inv=jacobi_diaginv: inv * rhs
    tensor_apply = None
    if tensor_factors is not None:
        tensor_apply = lambda rhs: _apply_tensor_diagonal_block(tensor_factors, rhs)
    return _build_k0_operator_preconditioner_apply(
        operator_apply,
        size,
        spec,
        jacobi_apply=jacobi_apply,
        tensor_apply=tensor_apply,
        seed_base=seed_base,
    )


def _build_nested_iterative_preconditioner_apply(
        operator_apply, smoother_apply, size: int, *,
    spec: MassPreconditionerSpec, seed: int,
    orthogonal_vectors=None):
    if spec.kind == 'jacobi':
        diagonal = _diagonal_from_matvec(
            lambda x: smoother_apply(operator_apply(x)),
            size,
        )
        diaginv = _invert_diagonal(diagonal)
        return lambda rhs, inv=diaginv: inv * smoother_apply(rhs)

    max_eig = _estimate_preconditioned_max_eigenvalue_apply(
        operator_apply,
        smoother_apply,
        size,
        n_iter=spec.power_iterations,
        seed=seed,
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

    min_eig, max_eig = _estimate_chebyshev_lanczos_bounds_apply(
        operator_apply,
        smoother_apply,
        size,
        spec=spec,
        seed=seed,
        orthogonal_vectors=orthogonal_vectors,
    )
    return _build_chebyshev_apply_preconditioner(
        operator_apply,
        smoother_apply,
        steps=spec.steps,
        min_eig=min_eig,
        max_eig=max_eig,
    )


def _build_k0_operator_preconditioner_apply(
        operator_apply, size: int, spec: MassPreconditionerSpec, *,
        jacobi_apply, tensor_apply=None, seed_base: int = 0):
    valid_kinds = ('none', 'jacobi', 'richardson', 'chebyshev', 'tensor')
    if spec.kind not in valid_kinds:
        raise ValueError(
            "preconditioner kind must be one of "
            f"{valid_kinds} (got {spec.kind!r})"
        )
    if spec.surgery_schur:
        raise ValueError(
            "nested k=0 smoothers do not support surgery_schur"
        )
    if spec.kind == 'none':
        if spec.smoother is not None:
            raise ValueError("kind='none' does not support an additional smoother")
        return lambda x: x
    if spec.kind == 'tensor':
        if spec.smoother is not None:
            raise ValueError("kind='tensor' does not support an additional smoother")
        if tensor_apply is None:
            raise ValueError("Tensor mass preconditioner not assembled for k=0")
        return tensor_apply
    if spec.kind == 'jacobi' and spec.smoother is None:
        return jacobi_apply

    smoother_spec = _validate_inner_tensor_only_spec(
        spec.smoother,
        require_explicit=False,
        context=f"{spec.kind} iterative mass preconditioner",
    )
    if tensor_apply is None:
        raise ValueError(
            f"{spec.kind} iterative mass preconditioners currently require an assembled tensor smoother"
        )
    smoother_apply = _build_k0_operator_preconditioner_apply(
        operator_apply,
        size,
        smoother_spec,
        jacobi_apply=jacobi_apply,
        tensor_apply=tensor_apply,
        seed_base=seed_base + 1000,
    )
    return _build_nested_iterative_preconditioner_apply(
        operator_apply,
        smoother_apply,
        size,
        spec=spec,
        seed=seed_base + 17,
    )


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


def _build_mass_surgery_bulk_apply(
        seq, operators: SequenceOperators, *, k: int, dirichlet: bool,
        spec: MassPreconditionerSpec):
    surgery = _select_mass_surgery_factors(operators.mass_preconds, k, dirichlet)
    kind = spec.kind

    if k == 1 and spec.surgery_schur:
        diaginv = _mass_diaginv(seq, operators, k, dirichlet)
        tensor = None
        if _tensor_available(seq, operators, k):
            tensor = _select_mass_tensor_factors(operators.mass_preconds, k, dirichlet)
        use_inner_schur = True if tensor is None else tensor.use_inner_schur

        arr_spec = _normalize_recursive_scalar_leaf_spec(spec)
        scalar_spec = _normalize_mass_preconditioner_spec_for_degree(spec, k=3)

        r_operator_apply = _build_restricted_mass_block_operator_apply(
            seq,
            operators,
            k=k,
            dirichlet=dirichlet,
            indices=surgery.r_indices,
        )
        theta_operator_apply = _build_restricted_mass_block_operator_apply(
            seq,
            operators,
            k=k,
            dirichlet=dirichlet,
            indices=surgery.theta_bulk_indices,
        )
        zeta_operator_apply = _build_restricted_mass_block_operator_apply(
            seq,
            operators,
            k=k,
            dirichlet=dirichlet,
            indices=surgery.zeta_bulk_indices,
        )

        r_apply = _build_scalar_leaf_preconditioner_apply(
            r_operator_apply,
            surgery.r_indices.shape[0],
            arr_spec,
            jacobi_diaginv=diaginv[surgery.r_indices],
            tensor_factors=None if tensor is None else tensor.arr,
            seed_base=2100 + 100 * k + int(dirichlet),
        )
        theta_apply = _build_scalar_leaf_preconditioner_apply(
            theta_operator_apply,
            surgery.theta_bulk_indices.shape[0],
            scalar_spec,
            jacobi_diaginv=diaginv[surgery.theta_bulk_indices],
            tensor_factors=None if tensor is None else tensor.theta,
            seed_base=2200 + 100 * k + int(dirichlet),
        )
        zeta_apply = _build_scalar_leaf_preconditioner_apply(
            zeta_operator_apply,
            surgery.zeta_bulk_indices.shape[0],
            scalar_spec,
            jacobi_diaginv=diaginv[surgery.zeta_bulk_indices],
            tensor_factors=None if tensor is None else tensor.zeta,
            seed_base=2300 + 100 * k + int(dirichlet),
        )

        def rt_apply(rhs_rt):
            rhs_r = rhs_rt[:surgery.rt_r_size]
            rhs_theta = rhs_rt[surgery.rt_r_size:surgery.rt_r_size + surgery.rt_theta_size]
            y = r_apply(rhs_r)
            z = theta_apply(rhs_theta - _apply_k1_rt_atr_coupling(surgery, y))
            x_r = y - r_apply(_apply_k1_rt_art_coupling(surgery, z))
            return jnp.concatenate([x_r, z])

        def bulk_apply(rhs_bulk):
            rhs_r = rhs_bulk[:surgery.rt_r_size]
            rhs_theta = rhs_bulk[surgery.rt_r_size:surgery.bulk_rt_size]
            rhs_zeta = rhs_bulk[
                surgery.bulk_rt_size:surgery.bulk_rt_size + surgery.bulk_zeta_size
            ]
            if not use_inner_schur:
                return jnp.concatenate([
                    r_apply(rhs_r),
                    theta_apply(rhs_theta),
                    zeta_apply(rhs_zeta),
                ])
            rhs_rt = rhs_bulk[:surgery.bulk_rt_size]
            y_rt = rt_apply(rhs_rt)
            z = zeta_apply(rhs_zeta - _apply_k1_rt_to_zeta_coupling(surgery, y_rt))
            x_rt = y_rt - rt_apply(_apply_k1_zeta_to_rt_coupling(surgery, z))
            return jnp.concatenate([x_rt, z])

        return surgery, bulk_apply

    if k == 2 and spec.surgery_schur:
        diaginv = _mass_diaginv(seq, operators, k, dirichlet)
        tensor = None
        if _tensor_available(seq, operators, k):
            tensor = _select_mass_tensor_factors(operators.mass_preconds, k, dirichlet)
        use_inner_schur = True if tensor is None else tensor.use_inner_schur

        scalar_spec = _normalize_mass_preconditioner_spec_for_degree(spec, k=3)

        r_operator_apply = _build_restricted_mass_block_operator_apply(
            seq,
            operators,
            k=k,
            dirichlet=dirichlet,
            indices=surgery.r_bulk_indices,
        )
        theta_operator_apply = _build_restricted_mass_block_operator_apply(
            seq,
            operators,
            k=k,
            dirichlet=dirichlet,
            indices=surgery.theta_indices,
        )
        zeta_operator_apply = _build_restricted_mass_block_operator_apply(
            seq,
            operators,
            k=k,
            dirichlet=dirichlet,
            indices=surgery.zeta_indices,
        )

        r_apply = _build_scalar_leaf_preconditioner_apply(
            r_operator_apply,
            surgery.r_bulk_indices.shape[0],
            scalar_spec,
            jacobi_diaginv=diaginv[surgery.r_bulk_indices],
            tensor_factors=None if tensor is None else tensor.r_bulk,
            seed_base=2400 + 100 * k + int(dirichlet),
        )
        theta_apply = _build_scalar_leaf_preconditioner_apply(
            theta_operator_apply,
            surgery.theta_indices.shape[0],
            scalar_spec,
            jacobi_diaginv=diaginv[surgery.theta_indices],
            tensor_factors=None if tensor is None else tensor.theta,
            seed_base=2500 + 100 * k + int(dirichlet),
        )
        zeta_apply = _build_scalar_leaf_preconditioner_apply(
            zeta_operator_apply,
            surgery.zeta_indices.shape[0],
            scalar_spec,
            jacobi_diaginv=diaginv[surgery.zeta_indices],
            tensor_factors=None if tensor is None else tensor.zeta,
            seed_base=2600 + 100 * k + int(dirichlet),
        )

        def bulk_apply(rhs_bulk):
            rhs_r = rhs_bulk[:surgery.r_bulk_size]
            rhs_theta = rhs_bulk[
                surgery.r_bulk_size:surgery.r_bulk_size + surgery.theta_size
            ]
            rhs_zeta = rhs_bulk[
                surgery.r_bulk_size + surgery.theta_size:
                surgery.r_bulk_size + surgery.theta_size + surgery.zeta_size
            ]
            if not use_inner_schur:
                return jnp.concatenate([
                    r_apply(rhs_r),
                    theta_apply(rhs_theta),
                    zeta_apply(rhs_zeta),
                ])
            rhs_rt = rhs_bulk[:surgery.r_bulk_size + surgery.theta_size]
            rhs_r = rhs_rt[:surgery.r_bulk_size]
            rhs_theta = rhs_rt[surgery.r_bulk_size:surgery.r_bulk_size + surgery.theta_size]
            y = r_apply(rhs_r)
            z_theta = theta_apply(rhs_theta - _apply_k2_r_to_theta_coupling(surgery, y))
            x_r = y - r_apply(_apply_k2_theta_to_r_coupling(surgery, z_theta))
            y_rt = jnp.concatenate([x_r, z_theta])
            z = zeta_apply(rhs_zeta - _apply_k2_rt_to_zeta_coupling(surgery, y_rt))
            x_rt = y_rt - jnp.concatenate([
                r_apply(_apply_k2_zeta_to_rt_coupling(surgery, z)[:surgery.r_bulk_size]),
                theta_apply(_apply_k2_zeta_to_rt_coupling(surgery, z)[surgery.r_bulk_size:surgery.r_bulk_size + surgery.theta_size]),
            ])
            return jnp.concatenate([x_rt, z])

        return surgery, bulk_apply

    if kind == 'none':
        if k != 0:
            raise ValueError(
                "surgery_schur for kind='none' is currently only implemented for k=0"
            )
        smoother_spec = spec.smoother
        if smoother_spec is None:
            raise ValueError(
                "kind='none' with surgery_schur requires an explicit smoother"
            )
        bulk_indices = _k0_bulk_indices_from_surgery(surgery)
        bulk_operator_apply = _build_restricted_mass_block_operator_apply(
            seq,
            operators,
            k=k,
            dirichlet=dirichlet,
            indices=bulk_indices,
        )
        bulk_diaginv = _mass_diaginv(seq, operators, k, dirichlet)[surgery.surgery_size:]
        jacobi_apply = lambda rhs, inv=bulk_diaginv: inv * rhs
        tensor_apply = None
        if _tensor_available(seq, operators, k):
            tensor = _select_mass_tensor_factors(operators.mass_preconds, k, dirichlet)
            tensor_apply = lambda rhs: _apply_tensor_diagonal_block(tensor.bulk, rhs)
        return surgery, _build_k0_operator_preconditioner_apply(
            bulk_operator_apply,
            bulk_indices.shape[0],
            smoother_spec,
            jacobi_apply=jacobi_apply,
            tensor_apply=tensor_apply,
            seed_base=1700 + 100 * k + int(dirichlet),
        )

    if kind == 'tensor':
        tensor = _select_mass_tensor_factors(operators.mass_preconds, k, dirichlet)
        if k == 0:
            return surgery, lambda rhs: _apply_tensor_diagonal_block(tensor.bulk, rhs)
        if k == 1:
            use_inner_schur = tensor.use_inner_schur
            r_apply = lambda rhs: _apply_tensor_diagonal_block(tensor.arr, rhs)
            theta_apply = lambda rhs: _apply_tensor_diagonal_block(tensor.theta, rhs)
            zeta_apply = lambda rhs: _apply_tensor_diagonal_block(tensor.zeta, rhs)
        elif k == 2:
            use_inner_schur = tensor.use_inner_schur
            r_apply = lambda rhs: _apply_tensor_diagonal_block(tensor.r_bulk, rhs)
            theta_apply = lambda rhs: _apply_tensor_diagonal_block(tensor.theta, rhs)
            zeta_apply = lambda rhs: _apply_tensor_diagonal_block(tensor.zeta, rhs)
        else:
            raise ValueError(f"Mass surgery wrapper is not used for k={k}")
    elif kind == 'jacobi':
        diaginv = _mass_diaginv(seq, operators, k, dirichlet)
        if k == 0:
            return surgery, lambda rhs, inv=diaginv[surgery.surgery_size:]: inv * rhs
        if k == 1:
            use_inner_schur = True
            r_inv = diaginv[surgery.r_indices]
            theta_inv = diaginv[surgery.theta_bulk_indices]
            zeta_inv = diaginv[surgery.zeta_bulk_indices]
            r_apply = lambda rhs, inv=r_inv: inv * rhs
            theta_apply = lambda rhs, inv=theta_inv: inv * rhs
            zeta_apply = lambda rhs, inv=zeta_inv: inv * rhs
        elif k == 2:
            use_inner_schur = True
            r_inv = diaginv[surgery.r_bulk_indices]
            theta_inv = diaginv[surgery.theta_indices]
            zeta_inv = diaginv[surgery.zeta_indices]
            r_apply = lambda rhs, inv=r_inv: inv * rhs
            theta_apply = lambda rhs, inv=theta_inv: inv * rhs
            zeta_apply = lambda rhs, inv=zeta_inv: inv * rhs
        else:
            raise ValueError(f"Mass surgery wrapper is not used for k={k}")
    elif kind == 'richardson':
        if k != 0:
            raise ValueError(
                "surgery_schur for richardson is currently only implemented for k=0"
            )
        smoother_spec = spec.smoother
        if smoother_spec is None:
            smoother_spec = MassPreconditionerSpec(kind='jacobi')
        if smoother_spec.surgery_schur:
            raise ValueError(
                "richardson/schur does not support a surgery_schur smoother"
            )
        bulk_indices = _k0_bulk_indices_from_surgery(surgery)
        bulk_operator_apply = _build_restricted_mass_block_operator_apply(
            seq,
            operators,
            k=k,
            dirichlet=dirichlet,
            indices=bulk_indices,
        )
        bulk_diaginv = _mass_diaginv(seq, operators, k, dirichlet)[surgery.surgery_size:]
        jacobi_apply = lambda rhs, inv=bulk_diaginv: inv * rhs
        tensor_apply = None
        if _tensor_available(seq, operators, k):
            tensor = _select_mass_tensor_factors(operators.mass_preconds, k, dirichlet)
            tensor_apply = lambda rhs: _apply_tensor_diagonal_block(tensor.bulk, rhs)
        return surgery, _build_k0_operator_preconditioner_apply(
            bulk_operator_apply,
            bulk_indices.shape[0],
            spec,
            jacobi_apply=jacobi_apply,
            tensor_apply=tensor_apply,
            seed_base=1700 + 100 * k + int(dirichlet),
        )
    elif kind == 'chebyshev':
        if k != 0:
            raise ValueError(
                "surgery_schur for chebyshev is currently only implemented for k=0"
            )
        smoother_spec = spec.smoother
        if smoother_spec is None:
            smoother_spec = MassPreconditionerSpec(kind='jacobi')
        if smoother_spec.surgery_schur:
            raise ValueError(
                "chebyshev/schur does not support a surgery_schur smoother"
            )
        bulk_indices = _k0_bulk_indices_from_surgery(surgery)
        bulk_operator_apply = _build_restricted_mass_block_operator_apply(
            seq,
            operators,
            k=k,
            dirichlet=dirichlet,
            indices=bulk_indices,
        )
        bulk_diaginv = _mass_diaginv(seq, operators, k, dirichlet)[surgery.surgery_size:]
        jacobi_apply = lambda rhs, inv=bulk_diaginv: inv * rhs
        tensor_apply = None
        if _tensor_available(seq, operators, k):
            tensor = _select_mass_tensor_factors(operators.mass_preconds, k, dirichlet)
            tensor_apply = lambda rhs: _apply_tensor_diagonal_block(tensor.bulk, rhs)
        return surgery, _build_k0_operator_preconditioner_apply(
            bulk_operator_apply,
            bulk_indices.shape[0],
            spec,
            jacobi_apply=jacobi_apply,
            tensor_apply=tensor_apply,
            seed_base=1800 + 100 * k + int(dirichlet),
        )
    else:
        raise ValueError(
            "surgery_schur is currently only implemented for jacobi, richardson, chebyshev, and tensor "
            f"mass preconditioners (got {kind!r})"
        )

    if k == 2:
        def bulk_apply(rhs_bulk):
            rhs_r = rhs_bulk[:surgery.r_bulk_size]
            rhs_theta = rhs_bulk[surgery.r_bulk_size:surgery.r_bulk_size + surgery.theta_size]
            rhs_zeta = rhs_bulk[
                surgery.r_bulk_size + surgery.theta_size:
                surgery.r_bulk_size + surgery.theta_size + surgery.zeta_size
            ]
            if not use_inner_schur:
                return jnp.concatenate([
                    r_apply(rhs_r),
                    theta_apply(rhs_theta),
                    zeta_apply(rhs_zeta),
                ])
            rhs_rt = rhs_bulk[:surgery.r_bulk_size + surgery.theta_size]
            rhs_r = rhs_rt[:surgery.r_bulk_size]
            rhs_theta = rhs_rt[surgery.r_bulk_size:surgery.r_bulk_size + surgery.theta_size]
            y = r_apply(rhs_r)
            z_theta = theta_apply(rhs_theta - _apply_k2_r_to_theta_coupling(surgery, y))
            x_r = y - r_apply(_apply_k2_theta_to_r_coupling(surgery, z_theta))
            y_rt = jnp.concatenate([x_r, z_theta])
            z = zeta_apply(rhs_zeta - _apply_k2_rt_to_zeta_coupling(surgery, y_rt))
            correction_rt = _apply_k2_zeta_to_rt_coupling(surgery, z)
            x_rt = y_rt - jnp.concatenate([
                r_apply(correction_rt[:surgery.r_bulk_size]),
                theta_apply(correction_rt[surgery.r_bulk_size:surgery.r_bulk_size + surgery.theta_size]),
            ])
            return jnp.concatenate([x_rt, z])

        return surgery, bulk_apply

    def rt_apply(rhs_rt):
        rhs_r = rhs_rt[:surgery.rt_r_size]
        rhs_theta = rhs_rt[surgery.rt_r_size:surgery.rt_r_size + surgery.rt_theta_size]
        y = r_apply(rhs_r)
        z = theta_apply(rhs_theta - _apply_k1_rt_atr_coupling(surgery, y))
        x_r = y - r_apply(_apply_k1_rt_art_coupling(surgery, z))
        return jnp.concatenate([x_r, z])

    def bulk_apply(rhs_bulk):
        rhs_r = rhs_bulk[:surgery.rt_r_size]
        rhs_theta = rhs_bulk[surgery.rt_r_size:surgery.bulk_rt_size]
        rhs_zeta = rhs_bulk[surgery.bulk_rt_size:surgery.bulk_rt_size + surgery.bulk_zeta_size]
        if not use_inner_schur:
            return jnp.concatenate([
                r_apply(rhs_r),
                theta_apply(rhs_theta),
                zeta_apply(rhs_zeta),
            ])
        rhs_rt = rhs_bulk[:surgery.bulk_rt_size]
        y_rt = rt_apply(rhs_rt)
        z = zeta_apply(rhs_zeta - _apply_k1_rt_to_zeta_coupling(surgery, y_rt))
        x_rt = y_rt - rt_apply(_apply_k1_zeta_to_rt_coupling(surgery, z))
        return jnp.concatenate([x_rt, z])

    return surgery, bulk_apply


def _build_mass_surgery_wrapped_preconditioner_apply(
        seq, operators: SequenceOperators, *, k: int, dirichlet: bool,
        spec: MassPreconditionerSpec):
    # Pure tensor surgery_schur is exactly the production tensor apply
    # (preconditioners.apply_mass_tensor_preconditioner). Delegate so the
    # two paths share a single implementation; this is what keeps the
    # "routed == direct" test invariant when the tensor apply gains
    # bulk-block Chebyshev polish driven by `true_block_apply`.
    if (
        spec.kind == 'tensor'
        and spec.surgery_schur
        and spec.smoother is None
        and k in (0, 1, 2)
    ):
        return lambda x: apply_mass_tensor_preconditioner_ops(
            seq, operators, x, k, dirichlet=dirichlet,
        )
    if k in (0, 2) and spec.kind in ('none', 'jacobi', 'richardson', 'chebyshev', 'tensor'):
        inner_spec = spec.smoother
        outer_spec = None
        if spec.kind == 'none':
            if inner_spec is None:
                raise ValueError(
                    "kind='none' with surgery_schur requires an explicit smoother"
                )
        elif spec.kind == 'tensor':
            if inner_spec is not None:
                raise ValueError(
                    "kind='tensor' with surgery_schur is a legacy alias and does not accept a smoother"
                )
            inner_spec = MassPreconditionerSpec(kind='tensor')
        else:
            if inner_spec is None:
                inner_spec = MassPreconditionerSpec(kind='jacobi')
            outer_spec = MassPreconditionerSpec(
                kind=spec.kind,
                steps=spec.steps,
                power_iterations=spec.power_iterations,
                damping_safety=spec.damping_safety,
                min_eig_fraction=spec.min_eig_fraction,
                lanczos_iterations=spec.lanczos_iterations,
                lanczos_max_eig_inflation=spec.lanczos_max_eig_inflation,
                lanczos_min_eig_deflation=spec.lanczos_min_eig_deflation,
                lanczos_min_eig_floor_fraction=spec.lanczos_min_eig_floor_fraction,
            )

        surgery, bulk_apply = _build_mass_surgery_bulk_apply(
            seq,
            operators,
            k=k,
            dirichlet=dirichlet,
            spec=MassPreconditionerSpec(
                kind='none',
                surgery_schur=True,
                smoother=inner_spec,
            ),
        )
        surgery_to_bulk_apply, bulk_to_surgery_apply = _mass_surgery_coupling_applies(k, surgery)
        schur_inv = _assemble_surgery_schur_inverse_from_applies(
            surgery.ass,
            surgery_to_bulk_apply,
            bulk_apply,
            bulk_to_surgery_apply,
        )

        def base_apply(rhs):
            if k == 0:
                rhs_s = rhs[:surgery.surgery_size]
                rhs_b = rhs[surgery.surgery_size:]
                y = bulk_apply(rhs_b)
                z = schur_inv @ (rhs_s - bulk_to_surgery_apply(y))
                x_b = y - bulk_apply(surgery_to_bulk_apply(z))
                return jnp.concatenate([z, x_b])

            rhs_s = rhs[surgery.surgery_indices]
            rhs_b = rhs[surgery.bulk_indices]
            y = bulk_apply(rhs_b)
            z = schur_inv @ (rhs_s - bulk_to_surgery_apply(y))
            x_b = y - bulk_apply(surgery_to_bulk_apply(z))
            x = jnp.zeros_like(rhs)
            x = x.at[surgery.surgery_indices].set(z)
            x = x.at[surgery.bulk_indices].set(x_b)
            return x

        if outer_spec is None:
            return base_apply

        def operator_apply(x):
            return apply_mass_matrix(seq, operators, x, k, dirichlet=dirichlet)

        size = surgery.apply_data.size
        return _build_nested_iterative_preconditioner_apply(
            operator_apply,
            base_apply,
            size,
            spec=outer_spec,
            seed=2700 + 100 * k + int(dirichlet),
        )

    surgery, bulk_apply = _build_mass_surgery_bulk_apply(
        seq,
        operators,
        k=k,
        dirichlet=dirichlet,
        spec=spec,
    )
    surgery_to_bulk_apply, bulk_to_surgery_apply = _mass_surgery_coupling_applies(k, surgery)
    schur_inv = _assemble_surgery_schur_inverse_from_applies(
        surgery.ass,
        surgery_to_bulk_apply,
        bulk_apply,
        bulk_to_surgery_apply,
    )

    if k == 0:
        def apply(rhs):
            rhs_s = rhs[:surgery.surgery_size]
            rhs_b = rhs[surgery.surgery_size:]
            y = bulk_apply(rhs_b)
            z = schur_inv @ (rhs_s - bulk_to_surgery_apply(y))
            x_b = y - bulk_apply(surgery_to_bulk_apply(z))
            return jnp.concatenate([z, x_b])

        return apply

    def apply(rhs):
        rhs_s = rhs[surgery.surgery_indices]
        rhs_b = rhs[surgery.bulk_indices]
        y = bulk_apply(rhs_b)
        z = schur_inv @ (rhs_s - bulk_to_surgery_apply(y))
        x_b = y - bulk_apply(surgery_to_bulk_apply(z))
        x = jnp.zeros_like(rhs)
        x = x.at[surgery.surgery_indices].set(z)
        x = x.at[surgery.bulk_indices].set(x_b)
        return x

    return apply


def _coerce_mass_preconditioner_spec(preconditioner):
    if preconditioner is None:
        return default_mass_preconditioner()
    if isinstance(preconditioner, MassPreconditionerSpec):
        return preconditioner
    if isinstance(preconditioner, str):
        if preconditioner == 'tensor':
            return MassPreconditionerSpec(kind='tensor', surgery_schur=True)
        return MassPreconditionerSpec(kind=preconditioner)
    raise TypeError(
        "mass preconditioner must be a kind string or MassPreconditionerSpec")


def _validate_inner_tensor_only_spec(
        inner_spec: Optional[MassPreconditionerSpec], *,
        require_explicit: bool, context: str):
    if inner_spec is None:
        if require_explicit:
            raise ValueError(
                f"{context} requires an explicit inner smoother with kind='tensor'"
            )
        return MassPreconditionerSpec(kind='tensor')
    if inner_spec.kind != 'tensor':
        raise ValueError(
            f"{context} only supports kind='tensor' as the inner smoother"
        )
    if inner_spec.surgery_schur:
        raise ValueError("inner Schur smoothers cannot themselves use surgery_schur")
    if inner_spec.smoother is not None:
        raise ValueError(
            f"{context} only supports a terminal tensor inner smoother"
        )
    return inner_spec


def _validate_public_k0_mass_preconditioner_spec(spec: MassPreconditionerSpec):
    if not spec.surgery_schur:
        if spec.kind in ('jacobi', 'none'):
            if spec.smoother is not None:
                raise ValueError(
                    f"k=0 kind='{spec.kind}' with surgery_schur=False does not accept an inner smoother"
                )
            return
        if spec.kind in ('richardson', 'chebyshev'):
            _validate_inner_tensor_only_spec(
                spec.smoother,
                require_explicit=False,
                context=f"k=0 {spec.kind} iterative mass preconditioner",
            )
            return
        if spec.kind == 'tensor':
            if spec.smoother is not None:
                raise ValueError(
                    "k=0 kind='tensor' with surgery_schur=False does not accept an inner smoother"
                )
            return
        return

    if spec.kind == 'tensor':
        if spec.smoother is not None:
            raise ValueError(
                "kind='tensor' with surgery_schur=True is a legacy alias and does not accept an inner smoother"
            )
        return

    if spec.kind == 'none':
        _validate_inner_tensor_only_spec(
            spec.smoother,
            require_explicit=True,
            context="kind='none' with surgery_schur=True",
        )
        return

    if spec.kind == 'richardson':
        _validate_inner_tensor_only_spec(
            spec.smoother,
            require_explicit=False,
            context="richardson/surgery_schur",
        )
        return

    if spec.kind == 'jacobi':
        raise ValueError(
            "jacobi/surgery_schur is disabled: the outer Jacobi sees a non-local Schur-preconditioned operator"
        )

    if spec.kind == 'chebyshev':
        raise ValueError(
            "chebyshev/surgery_schur is disabled at the public operator level"
        )


def _validate_public_k1_mass_preconditioner_spec(spec: MassPreconditionerSpec):
    if not spec.surgery_schur:
        if spec.kind in ('jacobi', 'none'):
            if spec.smoother is not None:
                raise ValueError(
                    f"k=1 kind='{spec.kind}' with surgery_schur=False does not accept an inner smoother"
                )
            return
        if spec.kind in ('richardson', 'chebyshev'):
            _validate_inner_tensor_only_spec(
                spec.smoother,
                require_explicit=False,
                context=f"k=1 {spec.kind} iterative mass preconditioner",
            )
            return
        if spec.kind == 'tensor':
            if spec.smoother is not None:
                raise ValueError(
                    "k=1 kind='tensor' with surgery_schur=False does not accept an inner smoother"
                )
            return
        return

    if spec.kind == 'tensor':
        if spec.smoother is not None:
            raise ValueError(
                "kind='tensor' with surgery_schur=True is a legacy alias and does not accept an inner smoother"
            )
        return

    if spec.kind == 'none':
        _validate_inner_tensor_only_spec(
            spec.smoother,
            require_explicit=True,
            context="kind='none' with surgery_schur=True",
        )
        return

    if spec.kind == 'richardson':
        _validate_inner_tensor_only_spec(
            spec.smoother,
            require_explicit=False,
            context="richardson/surgery_schur",
        )
        return

    if spec.kind == 'jacobi':
        raise ValueError(
            "jacobi/surgery_schur is disabled: the outer Jacobi sees a non-local Schur-preconditioned operator"
        )

    if spec.kind == 'chebyshev':
        raise ValueError(
            "chebyshev/surgery_schur is disabled at the public operator level"
        )


def _validate_public_k2_mass_preconditioner_spec(spec: MassPreconditionerSpec):
    _validate_public_k1_mass_preconditioner_spec(spec)


def _resolve_legacy_mass_preconditioner(seq, operators, k: int, preconditioner):
    if isinstance(preconditioner, str) and preconditioner == 'auto':
        if _tensor_available(seq, operators, k):
            return default_mass_preconditioner()
        return MassPreconditionerSpec(kind='jacobi')
    return _coerce_mass_preconditioner_spec(preconditioner)


def _normalize_mass_preconditioner_spec_for_degree(
        spec: MassPreconditionerSpec, *, k: int):
    if k != 3:
        return spec

    inner_spec = spec.smoother
    if inner_spec is not None:
        return _normalize_mass_preconditioner_spec_for_degree(inner_spec, k=k)

    if not spec.surgery_schur:
        return spec

    return MassPreconditionerSpec(
        kind=spec.kind,
        steps=spec.steps,
        power_iterations=spec.power_iterations,
        damping_safety=spec.damping_safety,
        min_eig_fraction=spec.min_eig_fraction,
        lanczos_iterations=spec.lanczos_iterations,
        lanczos_max_eig_inflation=spec.lanczos_max_eig_inflation,
        lanczos_min_eig_deflation=spec.lanczos_min_eig_deflation,
        lanczos_min_eig_floor_fraction=spec.lanczos_min_eig_floor_fraction,
    )


def _estimate_iterative_runtime_tuning_apply(
        operator_apply, smoother_apply, size: int, *,
        spec: MassPreconditionerSpec, seed: int,
        orthogonal_vectors=None) -> IterativeRuntimeTuning:
    if spec.kind == 'richardson':
        max_eig = _estimate_preconditioned_max_eigenvalue_apply(
            operator_apply,
            smoother_apply,
            size,
            n_iter=spec.power_iterations,
            seed=seed,
        )
        return IterativeRuntimeTuning(lambda_max=max_eig)
    if spec.kind == 'chebyshev':
        min_eig, max_eig = _estimate_chebyshev_lanczos_bounds_apply(
            operator_apply,
            smoother_apply,
            size,
            spec=spec,
            seed=seed,
            orthogonal_vectors=orthogonal_vectors,
        )
        return IterativeRuntimeTuning(
            lambda_max=max_eig,
            lambda_min=min_eig,
        )
    raise ValueError(
        "iterative runtime tuning is only defined for richardson and chebyshev"
    )


def _resolve_iterative_runtime_tuning_apply(
        operator_apply, smoother_apply, size: int, *,
        spec: MassPreconditionerSpec, seed: int,
        orthogonal_vectors=None,
        runtime_tuning: Optional[IterativeRuntimeTuning] = None,
) -> IterativeRuntimeTuning:
    if spec.kind == 'richardson':
        if runtime_tuning is not None and runtime_tuning.lambda_max is not None:
            return runtime_tuning
    elif spec.kind == 'chebyshev':
        if runtime_tuning is not None and \
                runtime_tuning.lambda_min is not None and \
                runtime_tuning.lambda_max is not None:
            return runtime_tuning
    else:
        raise ValueError(
            "iterative runtime tuning is only defined for richardson and chebyshev"
        )

    return _estimate_iterative_runtime_tuning_apply(
        operator_apply,
        smoother_apply,
        size,
        spec=spec,
        seed=seed,
        orthogonal_vectors=orthogonal_vectors,
    )


def _build_operator_preconditioner_apply(
        seq, operators: SequenceOperators, *, k: int, dirichlet: bool,
    operator_apply, preconditioner, allow_none: bool = True,
    orthogonal_vectors=None,
    runtime_tuning: Optional[IterativeRuntimeTuning] = None):
    spec = _resolve_legacy_mass_preconditioner(seq, operators, k, preconditioner)
    spec = _normalize_mass_preconditioner_spec_for_degree(spec, k=k)
    if k == 0:
        _validate_public_k0_mass_preconditioner_spec(spec)
    if k == 1:
        _validate_public_k1_mass_preconditioner_spec(spec)
    if k == 2:
        _validate_public_k2_mass_preconditioner_spec(spec)
    valid_kinds = ('none', 'jacobi', 'richardson', 'chebyshev', 'tensor')
    if spec.kind not in valid_kinds:
        raise ValueError(
            "preconditioner kind must be one of "
            f"{valid_kinds} (got {spec.kind!r})")
    if spec.kind == 'none':
        if spec.surgery_schur:
            if k not in (0, 1, 2):
                raise ValueError(
                    f"surgery_schur is not used for k={k} with kind='none'"
                )
            if not _surgery_available(seq, operators, k):
                raise ValueError(
                    f"Mass surgery preconditioner not assembled for k={k}; "
                    "call assemble_mass_surgery_preconditioner(seq, operators, ...) first"
                )
            return _build_mass_surgery_wrapped_preconditioner_apply(
                seq,
                operators,
                k=k,
                dirichlet=dirichlet,
                spec=spec,
            )
        if not allow_none:
            raise ValueError("this preconditioner slot does not allow kind='none'")
        return lambda x: x
    if spec.surgery_schur and spec.kind in ('jacobi', 'richardson', 'chebyshev', 'tensor') and k in (0, 1, 2):
        if not _surgery_available(seq, operators, k):
            raise ValueError(
                f"Mass surgery preconditioner not assembled for k={k}; "
                "call assemble_mass_surgery_preconditioner(seq, operators, ...) first"
            )
        if spec.kind == 'tensor' and not _tensor_available(seq, operators, k):
            raise ValueError(
                f"Tensor mass preconditioner not assembled for k={k}; "
                "call assemble_tensor_mass_preconditioner(seq, operators, ...) first"
            )
        return _build_mass_surgery_wrapped_preconditioner_apply(
            seq,
            operators,
            k=k,
            dirichlet=dirichlet,
            spec=spec,
        )
    if spec.kind == 'jacobi':
        diaginv = _mass_diaginv(seq, operators, k, dirichlet)
        return lambda x, diaginv=diaginv: diaginv * x
    if spec.kind == 'tensor':
        if not _tensor_available(seq, operators, k):
            raise ValueError(
                f"Tensor mass preconditioner not assembled for k={k}")
        return lambda x: apply_mass_tensor_preconditioner_ops(
            seq, operators, x, k, dirichlet=dirichlet)
    if spec.surgery_schur:
        raise ValueError(
            "surgery_schur is currently only implemented for jacobi, richardson, chebyshev, and tensor "
            f"mass preconditioners (got {spec.kind!r})"
        )

    smoother_spec = _validate_inner_tensor_only_spec(
        spec.smoother,
        require_explicit=False,
        context=f"{spec.kind} iterative mass preconditioner",
    )
    if not _tensor_available(seq, operators, k):
        raise ValueError(
            f"{spec.kind} iterative mass preconditioners require an assembled tensor inner smoother for k={k}"
        )
    smoother_apply = lambda x: apply_mass_tensor_preconditioner_ops(
        seq, operators, x, k, dirichlet=dirichlet)
    suffix = '_dbc' if dirichlet else ''
    size = getattr(seq, f'n{k}{suffix}')
    tuning = _resolve_iterative_runtime_tuning_apply(
        operator_apply,
        smoother_apply,
        size,
        spec=spec,
        seed=1000 * k + int(dirichlet),
        orthogonal_vectors=orthogonal_vectors,
        runtime_tuning=runtime_tuning,
    )
    if spec.kind == 'richardson':
        omega = jnp.where(
            tuning.lambda_max > 0.0,
            jnp.asarray(spec.damping_safety, dtype=jnp.float64) / tuning.lambda_max,
            jnp.asarray(1.0, dtype=jnp.float64),
        )
        return _build_richardson_apply_preconditioner(
            operator_apply,
            smoother_apply,
            steps=spec.steps,
            omega=omega,
        )
    return _build_chebyshev_apply_preconditioner(
        operator_apply,
        smoother_apply,
        steps=spec.steps,
        min_eig=tuning.lambda_min,
        max_eig=tuning.lambda_max,
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
        runtime_tuning=_select_mass_runtime_tuning(operators, k, dirichlet),
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


def _build_schur_apply_from_saddle_preconditioner(
        seq, operators: SequenceOperators, *, k: int, dirichlet: bool,
        eps: float, saddle_preconditioner: SaddlePointPreconditionerSpec):
    schur_inner_spec = saddle_preconditioner.schur.inner
    if schur_inner_spec.kind != 'tensor':
        raise ValueError(
            "schur.inner currently only supports kind='tensor'"
        )
    if schur_inner_spec.surgery_schur or schur_inner_spec.smoother is not None:
        raise ValueError(
            "schur.inner must be a terminal tensor preconditioner"
        )
    if not _tensor_available(seq, operators, k - 1):
        raise ValueError(
            "saddle preconditioners currently require an assembled tensor schur.inner"
        )

    schur_inner = lambda x: apply_mass_tensor_preconditioner_ops(
        seq, operators, x, k - 1, dirichlet=dirichlet
    )
    return _build_schur_operator_apply(
        seq,
        operators,
        k=k,
        dirichlet=dirichlet,
        eps=eps,
        inner_preconditioner_apply=schur_inner,
    )


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


def update_mass_runtime_tuning(
        seq, operators: Optional[SequenceOperators], *, k: int,
        dirichlet: bool = True, preconditioner='auto'):
    """Estimate and store dynamic tuning for a polynomial mass preconditioner."""
    operators = _ensure_extraction_operators(seq, operators)

    def operator_apply(x):
        return apply_mass_matrix(seq, operators, x, k, dirichlet=dirichlet)

    spec = _resolve_legacy_mass_preconditioner(seq, operators, k, preconditioner)
    spec = _normalize_mass_preconditioner_spec_for_degree(spec, k=k)
    if k == 0:
        _validate_public_k0_mass_preconditioner_spec(spec)
    if k == 1:
        _validate_public_k1_mass_preconditioner_spec(spec)
    if k == 2:
        _validate_public_k2_mass_preconditioner_spec(spec)
    if spec.kind not in ('richardson', 'chebyshev'):
        return operators
    if spec.surgery_schur:
        raise NotImplementedError(
            "runtime tuning storage for surgery_schur mass preconditioners is not implemented yet"
        )

    _validate_inner_tensor_only_spec(
        spec.smoother,
        require_explicit=False,
        context=f"{spec.kind} iterative mass preconditioner",
    )
    if not _tensor_available(seq, operators, k):
        raise ValueError(
            f"{spec.kind} iterative mass preconditioners require an assembled tensor inner smoother for k={k}"
        )

    smoother_apply = lambda x: apply_mass_tensor_preconditioner_ops(
        seq, operators, x, k, dirichlet=dirichlet)
    suffix = '_dbc' if dirichlet else ''
    size = getattr(seq, f'n{k}{suffix}')
    tuning = _estimate_iterative_runtime_tuning_apply(
        operator_apply,
        smoother_apply,
        size,
        spec=spec,
        seed=1000 * k + int(dirichlet),
    )
    return _set_mass_runtime_tuning(
        operators,
        k=k,
        dirichlet=dirichlet,
        tuning=tuning,
    )


def update_scalar_hodge_runtime_tuning(
        seq, operators: Optional[SequenceOperators], *, k: int,
        dirichlet: bool = True, eps: float = 0.0,
        preconditioner='auto'):
    """Estimate and store dynamic tuning for a scalar Hodge preconditioner."""
    operators = _ensure_extraction_operators(seq, operators)
    selected_preconditioner = _coerce_scalar_hodge_preconditioner(
        seq,
        operators,
        k=k,
        preconditioner=preconditioner,
    )
    spec = _coerce_mass_preconditioner_spec(selected_preconditioner)
    if spec.kind not in ('richardson', 'chebyshev'):
        return operators

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
    orthogonal_vectors = _nullspace_vectors(operators, k, dirichlet) if eps == 0.0 else None
    tuning = _estimate_iterative_runtime_tuning_apply(
        operator_apply,
        smoother_apply,
        size,
        spec=spec,
        seed=100 * k + int(dirichlet),
        orthogonal_vectors=orthogonal_vectors,
    )
    return _set_scalar_hodge_runtime_tuning(
        operators,
        k=k,
        dirichlet=dirichlet,
        eps=eps,
        tuning=tuning,
    )


def update_scalar_laplacian_runtime_tuning(
        seq, operators: Optional[SequenceOperators], *, k: int,
        dirichlet: bool = True, eps: float = 0.0,
        preconditioner='auto'):
    """Alias of update_scalar_hodge_runtime_tuning using Laplacian naming."""
    return update_scalar_hodge_runtime_tuning(
        seq,
        operators,
        k=k,
        dirichlet=dirichlet,
        eps=eps,
        preconditioner=preconditioner,
    )


def _coerce_saddle_preconditioner_spec(
        seq, operators: SequenceOperators, *, k: int,
        preconditioner) -> SaddlePointPreconditionerSpec:
    if preconditioner is None or preconditioner == 'auto':
        return _materialize_default_saddle_preconditioner(
            seq, operators, k=k)
    if isinstance(preconditioner, SaddlePointPreconditionerSpec):
        valid_outer_kinds = ('none', 'jacobi', 'richardson', 'chebyshev', 'exact_jacobi')
        if preconditioner.schur.outer.kind not in valid_outer_kinds:
            raise ValueError(
                "schur.outer kind must be one of "
                f"{valid_outer_kinds} (got {preconditioner.schur.outer.kind!r})"
            )
        if preconditioner.schur.outer.kind == 'tensor':
            raise ValueError(
                "schur.outer kind='tensor' is not supported; "
                "tensor saddle preconditioning is only valid for the lower "
                "mass block and schur.inner")
        _coerce_schur_diag_mode(
            preconditioner.schur.outer,
            context=f"schur.outer kind={preconditioner.schur.outer.kind!r}",
        )
        if preconditioner.schur.outer.kind != 'exact_jacobi':
            _validate_inner_tensor_only_spec(
                preconditioner.schur.inner,
                require_explicit=True,
                context="schur.inner",
            )
        return preconditioner
    if isinstance(preconditioner, str):
        if preconditioner == 'tensor' and k == 3:
            raise ValueError(
                "preconditioner='tensor' is not supported for saddle solves; "
                "tensor saddle preconditioning is only valid for the lower "
                "mass block and schur.inner")
        if not _tensor_available(seq, operators, k - 1):
            raise ValueError(
                "saddle preconditioners currently require an assembled tensor schur.inner"
            )
        lower_kind = 'tensor' if preconditioner != 'jacobi' else 'jacobi'
        if preconditioner == 'tensor':
            raise ValueError(
                "schur.outer kind='tensor' is not supported; "
                "tensor saddle preconditioning is only valid for the lower "
                "mass block and schur.inner")
        valid_outer_kinds = ('none', 'jacobi', 'richardson', 'chebyshev', 'exact_jacobi')
        if preconditioner not in valid_outer_kinds:
            raise ValueError(
                "saddle outer kind must be one of "
                f"{valid_outer_kinds} (got {preconditioner!r})"
            )
        lower = MassPreconditionerSpec(kind=lower_kind)
        return SaddlePointPreconditionerSpec(
            mass=lower,
            schur=SchurPreconditionerSpec(
                inner=MassPreconditionerSpec(kind='tensor'),
                outer=MassPreconditionerSpec(kind=preconditioner),
            ),
        )
    raise TypeError(
        'saddle preconditioner must be a kind string or '
        'SaddlePointPreconditionerSpec')


def update_schur_runtime_tuning(
        seq, operators: Optional[SequenceOperators], *, k: int,
        dirichlet: bool = True, eps: float = 0.0,
        preconditioner='auto'):
    """Estimate and store dynamic tuning for a polynomial Schur-outer preconditioner."""
    if k <= 0:
        raise ValueError("Schur runtime tuning is only defined for k >= 1")

    operators = _ensure_extraction_operators(seq, operators)
    saddle_preconditioner = _coerce_saddle_preconditioner_spec(
        seq,
        operators,
        k=k,
        preconditioner=preconditioner,
    )
    outer_spec = saddle_preconditioner.schur.outer
    if outer_spec.kind not in ('richardson', 'chebyshev'):
        return operators

    schur_apply = _build_schur_apply_from_saddle_preconditioner(
        seq,
        operators,
        k=k,
        dirichlet=dirichlet,
        eps=eps,
        saddle_preconditioner=saddle_preconditioner,
    )
    if outer_spec.smoother is None:
        diaginv = _build_schur_outer_jacobi_diaginv(
            seq,
            operators,
            k=k,
            dirichlet=dirichlet,
            eps=eps,
            outer_spec=outer_spec,
            saddle_preconditioner=saddle_preconditioner,
            # If a mode-matched Schur Jacobi diagonal was preassembled, reuse it
            # for all outer kinds to avoid repeated probe builds.
            allow_stored_tensor_diaginv=True,
        )
        smoother_apply = lambda x, d=diaginv: d * x
    else:
        _validate_inner_tensor_only_spec(
            outer_spec.smoother,
            require_explicit=False,
            context=f"schur.outer {outer_spec.kind} iterative preconditioner",
        )
        if not _tensor_available(seq, operators, k):
            raise ValueError(
                f"schur.outer {outer_spec.kind} iterative preconditioners require an assembled tensor smoother for k={k}"
            )
        smoother_apply = lambda x: apply_mass_tensor_preconditioner_ops(
            seq, operators, x, k, dirichlet=dirichlet
        )
    suffix = '_dbc' if dirichlet else ''
    size = getattr(seq, f'n{k}{suffix}')
    orthogonal_vectors = _saddle_nullspaces(seq, operators, k, dirichlet)[0] if eps == 0.0 else None
    tuning = _estimate_iterative_runtime_tuning_apply(
        schur_apply,
        smoother_apply,
        size,
        spec=outer_spec,
        seed=10_000 * k + int(dirichlet),
        orthogonal_vectors=orthogonal_vectors,
    )
    return _set_schur_runtime_tuning(
        operators,
        k=k,
        dirichlet=dirichlet,
        eps=eps,
        tuning=tuning,
    )


def _coerce_diffusion_preconditioner_spec(
        seq, operators: SequenceOperators, *, k: int, preconditioner):
    if preconditioner is None or preconditioner == 'auto':
        if _tensor_available(seq, operators, k):
            return MassPreconditionerSpec(kind='tensor')
        return MassPreconditionerSpec(kind='jacobi')
    if isinstance(preconditioner, MassPreconditionerSpec):
        return preconditioner
    if isinstance(preconditioner, str):
        if preconditioner == 'tensor':
            return MassPreconditionerSpec(kind='tensor')
        return MassPreconditionerSpec(kind=preconditioner)
    raise TypeError(
        'diffusion preconditioner must be a kind string or MassPreconditionerSpec')


def _build_diffusion_preconditioner_apply(
        seq, operators: SequenceOperators, *, k: int, dirichlet: bool,
        eps: float, preconditioner, allow_none: bool = True,
        runtime_tuning: Optional[IterativeRuntimeTuning] = None):
    spec = _coerce_diffusion_preconditioner_spec(
        seq,
        operators,
        k=k,
        preconditioner=preconditioner,
    )
    spec = _normalize_mass_preconditioner_spec_for_degree(spec, k=k)
    valid_kinds = ('none', 'jacobi', 'richardson', 'chebyshev', 'tensor')
    if spec.kind not in valid_kinds:
        raise ValueError(
            "preconditioner kind must be one of "
            f"{valid_kinds} (got {spec.kind!r})")
    if spec.surgery_schur:
        raise ValueError(
            "diffusion upper-block preconditioners do not support surgery_schur"
        )
    if spec.kind == 'none':
        if not allow_none:
            raise ValueError("this preconditioner slot does not allow kind='none'")
        return lambda x: x
    if spec.kind == 'jacobi':
        diaginv = _mass_diaginv(seq, operators, k, dirichlet)
        return lambda x, diaginv=diaginv: diaginv * x
    if spec.kind == 'tensor':
        if not _tensor_available(seq, operators, k):
            raise ValueError(
                f"Tensor diffusion preconditioner not assembled for k={k}"
            )
        return lambda x: apply_mass_tensor_preconditioner_ops(
            seq, operators, x, k, dirichlet=dirichlet
        )

    def operator_apply(x):
        return apply_mass_matrix(
            seq, operators, x, k, dirichlet=dirichlet
        ) + eps * apply_stiffness(
            seq, operators, x, k, dirichlet=dirichlet
        )

    _validate_inner_tensor_only_spec(
        spec.smoother,
        require_explicit=False,
        context=f"{spec.kind} iterative diffusion preconditioner",
    )
    if not _tensor_available(seq, operators, k):
        raise ValueError(
            f"{spec.kind} iterative diffusion preconditioners require an assembled tensor inner smoother for k={k}"
        )
    smoother_apply = lambda x: apply_mass_tensor_preconditioner_ops(
        seq, operators, x, k, dirichlet=dirichlet
    )
    suffix = '_dbc' if dirichlet else ''
    size = getattr(seq, f'n{k}{suffix}')
    tuning = _resolve_iterative_runtime_tuning_apply(
        operator_apply,
        smoother_apply,
        size,
        spec=spec,
        seed=20_000 * k + int(dirichlet),
        runtime_tuning=runtime_tuning,
    )
    if spec.kind == 'richardson':
        omega = jnp.where(
            tuning.lambda_max > 0.0,
            jnp.asarray(spec.damping_safety, dtype=jnp.float64) / tuning.lambda_max,
            jnp.asarray(1.0, dtype=jnp.float64),
        )
        return _build_richardson_apply_preconditioner(
            operator_apply,
            smoother_apply,
            steps=spec.steps,
            omega=omega,
        )
    return _build_chebyshev_apply_preconditioner(
        operator_apply,
        smoother_apply,
        steps=spec.steps,
        min_eig=tuning.lambda_min,
        max_eig=tuning.lambda_max,
    )


def update_diffusion_runtime_tuning(
        seq, operators: Optional[SequenceOperators], *, k: int,
        dirichlet: bool = True, eps: float = 0.0,
        preconditioner='auto'):
    """Estimate and store dynamic tuning for a polynomial diffusion preconditioner."""
    if eps < 0.0:
        raise ValueError("eps must be nonnegative")

    operators = _ensure_extraction_operators(seq, operators)
    if eps == 0.0:
        return update_mass_runtime_tuning(
            seq,
            operators,
            k=k,
            dirichlet=dirichlet,
            preconditioner=preconditioner,
        )

    def operator_apply(x):
        return apply_mass_matrix(
            seq, operators, x, k, dirichlet=dirichlet
        ) + eps * apply_stiffness(
            seq, operators, x, k, dirichlet=dirichlet
        )

    spec = _coerce_diffusion_preconditioner_spec(
        seq,
        operators,
        k=k,
        preconditioner=preconditioner,
    )
    spec = _normalize_mass_preconditioner_spec_for_degree(spec, k=k)
    if spec.kind not in ('richardson', 'chebyshev'):
        return operators
    if spec.surgery_schur:
        raise NotImplementedError(
            "runtime tuning storage for surgery_schur diffusion preconditioners is not implemented yet"
        )

    _validate_inner_tensor_only_spec(
        spec.smoother,
        require_explicit=False,
        context=f"{spec.kind} iterative diffusion preconditioner",
    )
    if not _tensor_available(seq, operators, k):
        raise ValueError(
            f"{spec.kind} iterative diffusion preconditioners require an assembled tensor inner smoother for k={k}"
        )

    smoother_apply = lambda x: apply_mass_tensor_preconditioner_ops(
        seq, operators, x, k, dirichlet=dirichlet
    )
    suffix = '_dbc' if dirichlet else ''
    size = getattr(seq, f'n{k}{suffix}')
    tuning = _estimate_iterative_runtime_tuning_apply(
        operator_apply,
        smoother_apply,
        size,
        spec=spec,
        seed=20_000 * k + int(dirichlet),
    )
    return _set_diffusion_runtime_tuning(
        operators,
        k=k,
        dirichlet=dirichlet,
        eps=eps,
        tuning=tuning,
    )


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
    if not _tensor_available(seq, operators, k - 1):
        raise ValueError(
            "default saddle preconditioners currently require an assembled tensor schur.inner"
        )
    return SaddlePointPreconditionerSpec(
        mass=lower,
        schur=SchurPreconditionerSpec(
            inner=MassPreconditionerSpec(kind='tensor'),
            outer=MassPreconditionerSpec(kind='jacobi'),
        ),
        coupled=coupled_preconditioner,
    )


def _materialize_default_scalar_hodge_preconditioner(
        seq, operators: SequenceOperators, *, k: int):
    if k == 0 and _k0_tensor_hodge_available(operators):
        return MassPreconditionerSpec(kind='tensor', surgery_schur=True)
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
        stiffness_diaginv = _hodge_diaginv(seq, operators, k, dirichlet)
        if eps == 0.0:
            shifted_diaginv = stiffness_diaginv
        else:
            mass_diaginv_k = _mass_diaginv(seq, operators, k, dirichlet)
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
            seq, operators, x, dirichlet=dirichlet)
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
    orthogonal_vectors = _nullspace_vectors(operators, k, dirichlet) if eps == 0.0 else None
    tuning = _resolve_iterative_runtime_tuning_apply(
        operator_apply,
        smoother_apply,
        size,
        spec=spec,
        seed=100 * k + int(dirichlet),
        orthogonal_vectors=orthogonal_vectors,
        runtime_tuning=_select_scalar_hodge_runtime_tuning(
            operators,
            k,
            dirichlet,
            eps,
        ),
    )
    if spec.kind == 'richardson':
        omega = jnp.where(
            tuning.lambda_max > 0.0,
            jnp.asarray(spec.damping_safety, dtype=jnp.float64) / tuning.lambda_max,
            jnp.asarray(1.0, dtype=jnp.float64),
        )
        return _build_richardson_apply_preconditioner(
            operator_apply,
            smoother_apply,
            steps=spec.steps,
            omega=omega,
        )
    return _build_chebyshev_apply_preconditioner(
        operator_apply,
        smoother_apply,
        steps=spec.steps,
        min_eig=tuning.lambda_min,
        max_eig=tuning.lambda_max,
    )


def _build_shifted_chebyshev_hodge_preconditioner(
        seq, operators: SequenceOperators, *, k: int, dirichlet: bool,
        eps: float, steps: int, power_iterations: int,
        min_eig_fraction: float):
    suffix = "_dbc" if dirichlet else ""
    size = getattr(seq, f"n{k}{suffix}")
    stiffness_diaginv = _hodge_diaginv(seq, operators, k, dirichlet)
    if eps == 0.0:
        shifted_diaginv = stiffness_diaginv
    else:
        mass_diaginv_k = _mass_diaginv(seq, operators, k, dirichlet)
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
    lanczos_spec = MassPreconditionerSpec(
        kind='chebyshev',
        steps=steps,
        power_iterations=power_iterations,
        min_eig_fraction=min_eig_fraction,
    )
    orthogonal_vectors = _nullspace_vectors(operators, k, dirichlet) if eps == 0.0 else None
    min_eig, max_eig = _estimate_chebyshev_lanczos_bounds_apply(
        operator_apply,
        smoother_apply,
        size,
        spec=lanczos_spec,
        seed=100 * k + int(dirichlet),
        orthogonal_vectors=orthogonal_vectors,
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
        * ``'tensor'`` — assembled surgery-plus-Schur tensor Hodge model for
            ``k = 0`` only.
        * ``'auto'`` — picks ``'tensor'`` when available for ``k = 0`` and falls
            back to ``'jacobi'`` otherwise.
    """
    if kind not in ('auto', 'none', 'jacobi', 'tensor'):
        raise ValueError(
            f"kind must be 'auto', 'none', 'jacobi' or 'tensor' (got {kind!r})")
    if kind == 'auto':
        if k == 0 and _k0_tensor_hodge_available(operators):
            kind = 'tensor'
        else:
            kind = 'jacobi'
    if kind == 'none':
        return v
    if kind == 'jacobi':
        return _hodge_diaginv(seq, operators, k, dirichlet) * v
    if kind == 'tensor':
        if k == 0:
            if not _k0_tensor_hodge_available(operators):
                raise ValueError(
                    "Tensor Hodge preconditioner for k=0 requires the tensor Hodge "
                    "assembly; call assemble_tensor_hodge_preconditioner first")
            return _apply_k0_tensor_hodge_preconditioner(
                seq,
                operators,
                v,
                dirichlet=dirichlet,
            )
        raise ValueError(
            f"Tensor Hodge preconditioner not available for k={k}; use 'jacobi' instead")
    raise AssertionError("unreachable")


def apply_laplacian_preconditioner(seq, operators: SequenceOperators, v, k: int,
                                   dirichlet: bool = True,
                                   kind: str = 'auto'):
    """Alias of apply_hodge_laplacian_preconditioner using Laplacian naming."""
    return apply_hodge_laplacian_preconditioner(
        seq, operators, v, k, dirichlet=dirichlet, kind=kind)


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


def apply_inverse_laplacian(seq, operators: SequenceOperators, rhs, k: int,
                            dirichlet: bool = True, guess=None,
                            tol: Optional[float] = None,
                            maxiter: Optional[int] = None,
                            preconditioner='auto',
                            return_info: bool = False):
    """Alias of apply_inverse_hodge_laplacian using Laplacian naming."""
    return apply_inverse_hodge_laplacian(
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
    if saddle_preconditioner.schur.outer.kind == 'exact_jacobi':
        exact_lower = lambda rhs: apply_inverse_mass_matrix(
            seq,
            operators,
            rhs,
            k - 1,
            dirichlet=dirichlet,
            preconditioner='jacobi',
        )
        schur_probe_apply = _build_schur_operator_apply(
            seq,
            operators,
            k=k,
            dirichlet=dirichlet,
            eps=eps,
            inner_preconditioner_apply=exact_lower,
        )
        precond_upper = _build_exact_jacobi_preconditioner_apply(
            schur_probe_apply,
            n_upper,
            warning_context=(
                f"schur.outer kind='exact_jacobi' for k={k}"
            ),
        )
    else:
        schur_apply = _build_schur_apply_from_saddle_preconditioner(
            seq,
            operators,
            k=k,
            dirichlet=dirichlet,
            eps=eps,
            saddle_preconditioner=saddle_preconditioner,
        )
        outer_spec = saddle_preconditioner.schur.outer
        if outer_spec.kind in ('jacobi', 'richardson', 'chebyshev') and outer_spec.smoother is None:
            schur_diaginv = _build_schur_outer_jacobi_diaginv(
                seq,
                operators,
                k=k,
                dirichlet=dirichlet,
                eps=eps,
                outer_spec=outer_spec,
                saddle_preconditioner=saddle_preconditioner,
                # If a mode-matched Schur Jacobi diagonal was preassembled, reuse it
                # for all outer kinds to avoid repeated probe builds.
                allow_stored_tensor_diaginv=True,
            )
            if outer_spec.kind == 'jacobi':
                precond_upper = lambda x, d=schur_diaginv: d * x
            else:
                smoother_apply = lambda x, d=schur_diaginv: d * x
                tuning = _resolve_iterative_runtime_tuning_apply(
                    schur_apply,
                    smoother_apply,
                    n_upper,
                    spec=outer_spec,
                    seed=10_000 * k + int(dirichlet),
                    orthogonal_vectors=vs_upper if eps == 0.0 else None,
                    runtime_tuning=_select_schur_runtime_tuning(
                        operators,
                        k,
                        dirichlet,
                        eps,
                    ),
                )
                if outer_spec.kind == 'richardson':
                    omega = jnp.where(
                        tuning.lambda_max > 0.0,
                        jnp.asarray(outer_spec.damping_safety, dtype=jnp.float64) / tuning.lambda_max,
                        jnp.asarray(1.0, dtype=jnp.float64),
                    )
                    precond_upper = _build_richardson_apply_preconditioner(
                        schur_apply,
                        smoother_apply,
                        steps=outer_spec.steps,
                        omega=omega,
                    )
                else:
                    precond_upper = _build_chebyshev_apply_preconditioner(
                        schur_apply,
                        smoother_apply,
                        steps=outer_spec.steps,
                        min_eig=tuning.lambda_min,
                        max_eig=tuning.lambda_max,
                    )
        else:
            precond_upper = _build_operator_preconditioner_apply(
                seq,
                operators,
                k=k,
                dirichlet=dirichlet,
                operator_apply=schur_apply,
                preconditioner=outer_spec,
                allow_none=True,
                orthogonal_vectors=vs_upper if eps == 0.0 else None,
                runtime_tuning=_select_schur_runtime_tuning(
                    operators,
                    k,
                    dirichlet,
                    eps,
                ),
            )
    # Apply 1/eps coarse correction on the harmonic upper-block mode, mirroring
    # the k=0 treatment.  For k>=1 the DBC nullspace is always empty on this
    # topology, so only the NBC case is relevant.
    if use_harmonic_coarse is None:
        use_harmonic_coarse = eps > 0 and not dirichlet
    if use_harmonic_coarse and eps > 0:
        # _shifted_harmonic_coarse_ready may return a traced bool when this
        # function is called inside a jax.lax.while_loop body.  Use
        # jax.lax.cond so the selection is JAX-traceable, mirroring the
        # tensor/jacobi fallback in the k=0 path.
        coarse_ready = _shifted_harmonic_coarse_ready(seq, operators, k, dirichlet)
        precond_with_coarse = _wrap_shifted_harmonic_coarse_correction(
            seq, operators, precond_upper, eps, k, dirichlet)
        precond_no_coarse = precond_upper
        precond_upper = lambda x, r=coarse_ready, a=precond_with_coarse, b=precond_no_coarse: (
            jax.lax.cond(r, a, b, x))

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


def apply_inverse_shifted_laplacian(seq, operators: SequenceOperators, rhs, k: int,
                                    eps: float, dirichlet: bool = True, guess=None,
                                    tol: Optional[float] = None,
                                    maxiter: Optional[int] = None,
                                    preconditioner='auto',
                                    use_harmonic_coarse: Optional[bool] = None,
                                    return_info: bool = False):
    """Alias of apply_inverse_shifted_hodge_laplacian using Laplacian naming."""
    return apply_inverse_shifted_hodge_laplacian(
        seq,
        operators,
        rhs,
        k,
        eps,
        dirichlet=dirichlet,
        guess=guess,
        tol=tol,
        maxiter=maxiter,
        preconditioner=preconditioner,
        use_harmonic_coarse=use_harmonic_coarse,
        return_info=return_info,
    )


def apply_inverse_mass_plus_eps_laplace_matrix(seq, operators: SequenceOperators, rhs, k: int,
                                               eps: float, dirichlet: bool = True, guess=None,
                                               tol: Optional[float] = None,
                                               maxiter: Optional[int] = None,
                                               preconditioner='auto',
                                               return_info: bool = False):
    """Solve with the inverse of M_k + eps L_k using an explicit operator bundle."""
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

        precond_apply = _build_diffusion_preconditioner_apply(
            seq,
            operators,
            k=0,
            dirichlet=dirichlet,
            eps=eps,
            preconditioner=preconditioner,
            allow_none=True,
            runtime_tuning=_select_diffusion_runtime_tuning(
                operators,
                0,
                dirichlet,
                eps,
            ),
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

    upper_preconditioner = _build_diffusion_preconditioner_apply(
        seq,
        operators,
        k=k,
        dirichlet=dirichlet,
        eps=eps,
        preconditioner=preconditioner,
        allow_none=True,
        runtime_tuning=_select_diffusion_runtime_tuning(
            operators,
            k,
            dirichlet,
            eps,
        ),
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


def apply_laplacian(seq, operators: SequenceOperators, v, k: int,
                    dirichlet: bool = True, guess=None,
                    tol: Optional[float] = None,
                    maxiter: Optional[int] = None):
    """Alias of apply_hodge_laplacian using Laplacian naming."""
    return apply_hodge_laplacian(
        seq,
        operators,
        v,
        k,
        dirichlet=dirichlet,
        guess=guess,
        tol=tol,
        maxiter=maxiter,
    )


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


def apply_laplacian_approx(seq, operators: SequenceOperators, v, k: int,
                           dirichlet: bool = True):
    """Alias of apply_hodge_laplacian_approx using Laplacian naming."""
    return apply_hodge_laplacian_approx(
        seq,
        operators,
        v,
        k,
        dirichlet=dirichlet,
    )


# ---------------------------------------------------------------------------
