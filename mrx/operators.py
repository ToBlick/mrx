from __future__ import annotations

from typing import Optional, Sequence
import warnings

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
    _apply_k0_bulk_to_surgery_coupling,
    _apply_k0_surgery_to_bulk_coupling,
    _apply_k1_bulk_to_surgery_coupling,
    _apply_k1_rt_art_coupling,
    _apply_k1_rt_atr_coupling,
    _apply_k1_rt_to_zeta_coupling,
    _apply_k1_surgery_to_bulk_coupling,
    _apply_k1_zeta_to_rt_coupling,
    _apply_k2_bulk_to_surgery_coupling,
    _apply_k2_r_to_theta_coupling,
    _apply_k2_surgery_to_bulk_coupling,
    _apply_k2_theta_to_r_coupling,
    _apply_k2_rt_to_zeta_coupling,
    _apply_k2_zeta_to_rt_coupling,
    _assemble_schur_inverse_from_applies,
    _assemble_shared_modal_basis,
    _bulk_tensor_shape,
    _core_size,
    _cp_als_3tensor,
    _apply_tensor_diagonal_block,
    _apply_tensor_diagonal_block_forward,
    _split_blocks,
    _k1_radial_reference_baselines,
    _select_mass_surgery_factors,
    _select_mass_tensor_factors,
    _symmetrize,
    apply_mass_tensor_forward_model,
    apply_mass_tensor_preconditioner,
    build_mass_surgery_preconditioner,
    build_mass_jacobi_pair,
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
    tensor_mass_rank_for_degree,
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
    runtime_tuning: SequenceRuntimeTuning = eqx.field(default_factory=SequenceRuntimeTuning)
    # Reference-domain 1-D fast-diagonalisation eigendecompositions of the
    # generalised eigenproblem ``K_a v = λ M_a v`` for the regular spline
    # space along each axis. ``fd_V_p_*`` columns are the ``M_a``-orthonormal
    # eigenvectors and ``fd_lam_p_*`` the corresponding eigenvalues.
    # Geometry-independent, populated by
    # :func:`assemble_tensor_hodge_preconditioner`.
    fd_V_p_r: Optional[jnp.ndarray] = None
    fd_V_p_t: Optional[jnp.ndarray] = None
    fd_V_p_z: Optional[jnp.ndarray] = None
    fd_lam_p_r: Optional[jnp.ndarray] = None
    fd_lam_p_t: Optional[jnp.ndarray] = None
    fd_lam_p_z: Optional[jnp.ndarray] = None
    # Per-direction scaling for the FD Hodge preconditioner. The entry along
    # axis ``i`` is the quadrature-average of ``J·g^{ii}``, which is the
    # diagonal of the stiffness integrand on the mapped domain and captures
    # the leading anisotropy for ``L_0``. Geometry-dependent;
    # reassembled with :func:`update_hodge_operator`.
    dd0_fd_scale_K: Optional[jnp.ndarray] = None
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
    # directly (no mass solve).
    g0: Optional[jsparse.BCSR] = None
    g0_T: Optional[jsparse.BCSR] = None
    g1: Optional[jsparse.BCSR] = None
    g1_T: Optional[jsparse.BCSR] = None
    g2: Optional[jsparse.BCSR] = None
    g2_T: Optional[jsparse.BCSR] = None
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
    dense: Optional[DenseSequenceOperators] = None

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
                case "hodge_laplacian":
                    pair = getattr(self.dense, f"l{k}")
                    return select_boundary_data(pair, dirichlet, f"Dense Hodge Laplacian k={k}")
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
    bulk_modal_basis_r: Optional[jnp.ndarray] = None
    bulk_modal_basis_t: Optional[jnp.ndarray] = None
    bulk_modal_basis_z: Optional[jnp.ndarray] = None
    bulk_modal_mass_r: tuple[jnp.ndarray, ...] = ()
    bulk_modal_mass_t: tuple[jnp.ndarray, ...] = ()
    bulk_modal_mass_z: tuple[jnp.ndarray, ...] = ()
    bulk_modal_stiff_r: tuple[jnp.ndarray, ...] = ()
    bulk_modal_stiff_t: tuple[jnp.ndarray, ...] = ()
    bulk_modal_stiff_z: tuple[jnp.ndarray, ...] = ()
    cp_relative_error: Optional[float] = None
    cp_final_delta: Optional[float] = None


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
    cp_relative_error: Optional[float] = None
    cp_final_delta: Optional[float] = None


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

    for idx in range(rank):
        factor_theta = jnp.ravel(factors[0][:, idx])
        factor_r = jnp.ravel(factors[1][:, idx])
        factor_z = jnp.ravel(factors[2][:, idx])
        scale = weights[idx]

        mass_r = _symmetrize(_assemble_weighted_1d_mass(
            seq.d_basis_r_jk,
            seq.quad.w_x * (scale * factor_r),
        ))
        mass_t = _symmetrize(_assemble_weighted_1d_mass(
            seq.d_basis_t_jk,
            seq.quad.w_y * factor_theta,
        ))
        mass_z = _symmetrize(_assemble_weighted_1d_mass(
            seq.d_basis_z_jk,
            seq.quad.w_z * factor_z,
        ))

        mass_r_terms.append(mass_r)
        mass_t_terms.append(mass_t)
        mass_z_terms.append(mass_z)

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
        cp_relative_error=cp_relative_error,
        cp_final_delta=cp_final_delta,
    )


def _apply_k2_divdiv_regular_tensor_model(
        model: K2TensorDivDivForwardModel,
        rhs: jnp.ndarray) -> jnp.ndarray:
    r_size = int(jnp.prod(jnp.asarray(model.r_shape)))
    theta_size = int(jnp.prod(jnp.asarray(model.theta_shape)))
    zeta_size = int(jnp.prod(jnp.asarray(model.zeta_shape)))
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
    size = (
        int(jnp.prod(jnp.asarray(model.r_shape)))
        + int(jnp.prod(jnp.asarray(model.theta_shape)))
        + int(jnp.prod(jnp.asarray(model.zeta_shape)))
    )
    return _assemble_dense_from_apply(
        lambda x, tensor_model=model: _apply_k2_divdiv_regular_tensor_model(tensor_model, x),
        size,
    )


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


def _assemble_dense_from_apply(apply, size: int) -> jnp.ndarray:
    basis = jnp.eye(size, dtype=jnp.float64)
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
    return tensor.rank, tensor.cp_maxiter, tensor.cp_tol, tensor.cp_ridge


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
        cp_ridge: float, radial_baseline: Optional[jnp.ndarray] = None):
    if radial_baseline is None:
        corrected_field = tensor_field
        scaled_radial_baseline = None
    else:
        scaled_radial_baseline = jnp.asarray(radial_baseline, dtype=tensor_field.dtype)
        corrected_field = tensor_field / scaled_radial_baseline[None, :, None]

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
        if scaled_radial_baseline is not None:
            factor_r = scaled_radial_baseline * factor_r
        terms.append({
            'scale': scale,
            'theta_factor': factor_theta,
            'radial_factor': factor_r,
            'zeta_factor': factor_z,
        })

    return {
        'terms': tuple(terms),
        'cp_relative_error': cp_relative_error,
        'cp_final_delta': cp_final_delta,
    }


def _fit_positive_rank1_tensor_field(
        tensor_field: jnp.ndarray, *, cp_maxiter: int, cp_tol: float,
        cp_ridge: float, radial_baseline: Optional[jnp.ndarray] = None):
    fit = _fit_positive_rank_tensor_field(
        tensor_field,
        rank=1,
        cp_maxiter=cp_maxiter,
        cp_tol=cp_tol,
        cp_ridge=cp_ridge,
        radial_baseline=radial_baseline,
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


def _k0_tensor_hodge_directional_rank1_metrics(
        seq, *, cp_maxiter: int, cp_tol: float, cp_ridge: float):
    return _k0_tensor_hodge_directional_rank_metrics(
        seq,
        rank=1,
        cp_maxiter=cp_maxiter,
        cp_tol=cp_tol,
        cp_ridge=cp_ridge,
    )


def _k0_tensor_hodge_directional_rank_metrics(
        seq, *, rank: int, cp_maxiter: int, cp_tol: float, cp_ridge: float):
    metric_tensors = _k0_stiffness_diagonal_metric_tensors(seq)
    direction_scales = _fd_hodge_scales_K(seq, seq.geometry, 0)
    safe_scales = jnp.where(jnp.abs(direction_scales) > 0, direction_scales, 1.0)
    normalized_rr = metric_tensors['alpha_rr'] / safe_scales[0]
    normalized_tt = metric_tensors['alpha_thetatheta'] / safe_scales[1]
    normalized_zz = metric_tensors['alpha_zetazeta'] / safe_scales[2]
    shared_target = jnp.cbrt(normalized_rr * normalized_tt * normalized_zz)

    shared = _fit_positive_rank_tensor_field(
        shared_target,
        rank=rank,
        cp_maxiter=cp_maxiter,
        cp_tol=cp_tol,
        cp_ridge=cp_ridge,
    )

    return {
        'shared_terms': shared['terms'],
        'normalized_rr': normalized_rr,
        'normalized_tt': normalized_tt,
        'normalized_zz': normalized_zz,
        'direction_scales': direction_scales,
        'cp_relative_error': shared['cp_relative_error'],
        'cp_final_delta': shared['cp_final_delta'],
    }


def _assemble_k0_tensor_hodge_rank1_bulk_factors(
        seq, *, dirichlet: bool, cp_maxiter: int, cp_tol: float, cp_ridge: float):
    bulk_shape = _bulk_tensor_shape(seq, dirichlet)
    nr_bulk, _, _ = bulk_shape
    types = seq.basis_0.types
    g_r = _dense_incidence_1d(seq.basis_0.nr, types[0])
    g_t = _dense_incidence_1d(seq.basis_0.nt, types[1])
    g_z = _dense_incidence_1d(seq.basis_0.nz, types[2])
    fits = _k0_tensor_hodge_directional_rank1_metrics(
        seq,
        cp_maxiter=cp_maxiter,
        cp_tol=cp_tol,
        cp_ridge=cp_ridge,
    )
    shared = fits['shared_terms'][0]
    radial_baselines = _k1_radial_reference_baselines(seq)
    shared_radial_mass_factor = shared['scale'] * shared['radial_factor']
    shared_theta_mass_factor = shared['theta_factor']
    shared_zeta_mass_factor = shared['zeta_factor']

    rr_active_factor = _project_tensor_to_radial_active_factor(
        fits['normalized_rr'],
        shared_theta_mass_factor,
        shared_zeta_mass_factor,
        radial_baseline=radial_baselines[0],
    )
    tt_active_factor = _project_tensor_to_theta_active_factor(
        fits['normalized_tt'],
        shared_radial_mass_factor,
        shared_zeta_mass_factor,
    )
    zz_active_factor = _project_tensor_to_zeta_active_factor(
        fits['normalized_zz'],
        shared_radial_mass_factor,
        shared_theta_mass_factor,
    )

    mass_r = _restrict_radial_window(
        _assemble_weighted_1d_mass(seq.basis_r_jk, seq.quad.w_x * shared_radial_mass_factor),
        radial_start=2,
        nr=nr_bulk,
    )
    mass_t = _assemble_weighted_1d_mass(seq.basis_t_jk, seq.quad.w_y * shared_theta_mass_factor)
    mass_z = _assemble_weighted_1d_mass(seq.basis_z_jk, seq.quad.w_z * shared_zeta_mass_factor)

    stiff_r = _restrict_radial_window(
        _assemble_weighted_1d_stiffness(
            seq.basis_r_jk,
            seq.d_basis_r_jk,
            seq.quad.w_x * (fits['direction_scales'][0] * rr_active_factor),
            g_r,
        ),
        radial_start=2,
        nr=nr_bulk,
    )
    stiff_t = _assemble_weighted_1d_stiffness(
        seq.basis_t_jk,
        seq.d_basis_t_jk,
        seq.quad.w_y * (fits['direction_scales'][1] * tt_active_factor),
        g_t,
    )
    stiff_z = _assemble_weighted_1d_stiffness(
        seq.basis_z_jk,
        seq.d_basis_z_jk,
        seq.quad.w_z * (fits['direction_scales'][2] * zz_active_factor),
        g_z,
    )

    V_r, lam_r = _assemble_1d_fd_eigendecomp(mass_r, stiff_r)
    V_t, lam_t = _assemble_1d_fd_eigendecomp(mass_t, stiff_t)
    V_z, lam_z = _assemble_1d_fd_eigendecomp(mass_z, stiff_z)
    return {
        'bulk_shape': bulk_shape,
        'bulk_alpha': jnp.ones((3,), dtype=jnp.float64),
        'bulk_V_r': V_r,
        'bulk_V_t': V_t,
        'bulk_V_z': V_z,
        'bulk_lam_r': lam_r,
        'bulk_lam_t': lam_t,
        'bulk_lam_z': lam_z,
        'bulk_mass_r': mass_r,
        'bulk_mass_t': mass_t,
        'bulk_mass_z': mass_z,
        'bulk_stiff_r': stiff_r,
        'bulk_stiff_t': stiff_t,
        'bulk_stiff_z': stiff_z,
        'bulk_term_mass_r': (mass_r,),
        'bulk_term_mass_t': (mass_t,),
        'bulk_term_mass_z': (mass_z,),
        'bulk_term_stiff_r': (stiff_r,),
        'bulk_term_stiff_t': (stiff_t,),
        'bulk_term_stiff_z': (stiff_z,),
        'cp_relative_error': fits['cp_relative_error'],
        'cp_final_delta': fits['cp_final_delta'],
    }


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


def _assemble_k0_tensor_hodge_multirank_bulk_factors(
        seq, *, dirichlet: bool, rank: int,
        cp_maxiter: int, cp_tol: float, cp_ridge: float):
    if rank < 2:
        raise ValueError(f"multi-rank k=0 tensor Hodge bulk requires rank >= 2 (got rank={rank})")

    bulk_shape = _bulk_tensor_shape(seq, dirichlet)
    nr_bulk, _, _ = bulk_shape
    types = seq.basis_0.types
    g_r = _dense_incidence_1d(seq.basis_0.nr, types[0])
    g_t = _dense_incidence_1d(seq.basis_0.nt, types[1])
    g_z = _dense_incidence_1d(seq.basis_0.nz, types[2])
    fits = _k0_tensor_hodge_directional_rank_metrics(
        seq,
        rank=rank,
        cp_maxiter=cp_maxiter,
        cp_tol=cp_tol,
        cp_ridge=cp_ridge,
    )
    radial_baselines = _k1_radial_reference_baselines(seq)
    shared_radial_mass_factors = tuple(shared['scale'] * shared['radial_factor'] for shared in fits['shared_terms'])
    shared_theta_mass_factors = tuple(shared['theta_factor'] for shared in fits['shared_terms'])
    shared_zeta_mass_factors = tuple(shared['zeta_factor'] for shared in fits['shared_terms'])

    rr_active_factors = _project_tensor_to_radial_active_factors(
        fits['normalized_rr'],
        shared_theta_mass_factors,
        shared_zeta_mass_factors,
        radial_baseline=radial_baselines[0],
    )
    tt_active_factors = _project_tensor_to_theta_active_factors(
        fits['normalized_tt'],
        shared_radial_mass_factors,
        shared_zeta_mass_factors,
    )
    zz_active_factors = _project_tensor_to_zeta_active_factors(
        fits['normalized_zz'],
        shared_radial_mass_factors,
        shared_theta_mass_factors,
    )

    mass_r_terms = []
    mass_t_terms = []
    mass_z_terms = []
    stiff_r_terms = []
    stiff_t_terms = []
    stiff_z_terms = []
    size_r_terms = []
    size_t_terms = []
    size_z_terms = []

    for shared_radial_mass_factor, shared_theta_mass_factor, shared_zeta_mass_factor, rr_active_factor, tt_active_factor, zz_active_factor in zip(
            shared_radial_mass_factors,
            shared_theta_mass_factors,
            shared_zeta_mass_factors,
            rr_active_factors,
            tt_active_factors,
            zz_active_factors):

        mass_r = _restrict_radial_window(
            _assemble_weighted_1d_mass(seq.basis_r_jk, seq.quad.w_x * shared_radial_mass_factor),
            radial_start=2,
            nr=nr_bulk,
        )
        mass_t = _assemble_weighted_1d_mass(seq.basis_t_jk, seq.quad.w_y * shared_theta_mass_factor)
        mass_z = _assemble_weighted_1d_mass(seq.basis_z_jk, seq.quad.w_z * shared_zeta_mass_factor)

        stiff_r = _restrict_radial_window(
            _assemble_weighted_1d_stiffness(
                seq.basis_r_jk,
                seq.d_basis_r_jk,
                seq.quad.w_x * (fits['direction_scales'][0] * rr_active_factor),
                g_r,
            ),
            radial_start=2,
            nr=nr_bulk,
        )
        stiff_t = _assemble_weighted_1d_stiffness(
            seq.basis_t_jk,
            seq.d_basis_t_jk,
            seq.quad.w_y * (fits['direction_scales'][1] * tt_active_factor),
            g_t,
        )
        stiff_z = _assemble_weighted_1d_stiffness(
            seq.basis_z_jk,
            seq.d_basis_z_jk,
            seq.quad.w_z * (fits['direction_scales'][2] * zz_active_factor),
            g_z,
        )

        mass_r_terms.append(mass_r)
        mass_t_terms.append(mass_t)
        mass_z_terms.append(mass_z)
        stiff_r_terms.append(stiff_r)
        stiff_t_terms.append(stiff_t)
        stiff_z_terms.append(stiff_z)
        size_r_terms.append(jnp.linalg.norm(stiff_r) * jnp.linalg.norm(mass_t) * jnp.linalg.norm(mass_z))
        size_t_terms.append(jnp.linalg.norm(mass_r) * jnp.linalg.norm(stiff_t) * jnp.linalg.norm(mass_z))
        size_z_terms.append(jnp.linalg.norm(mass_r) * jnp.linalg.norm(mass_t) * jnp.linalg.norm(stiff_z))

    mass_r_terms = tuple(mass_r_terms)
    mass_t_terms = tuple(mass_t_terms)
    mass_z_terms = tuple(mass_z_terms)
    stiff_r_terms = tuple(stiff_r_terms)
    stiff_t_terms = tuple(stiff_t_terms)
    stiff_z_terms = tuple(stiff_z_terms)
    size_r_terms = jnp.asarray(size_r_terms, dtype=jnp.float64)
    size_t_terms = jnp.asarray(size_t_terms, dtype=jnp.float64)
    size_z_terms = jnp.asarray(size_z_terms, dtype=jnp.float64)
    total_sizes = size_r_terms + size_t_terms + size_z_terms

    reference_mass_r = _weighted_average_dense_matrix(mass_r_terms, total_sizes)
    reference_mass_t = _weighted_average_dense_matrix(mass_t_terms, total_sizes)
    reference_mass_z = _weighted_average_dense_matrix(mass_z_terms, total_sizes)

    basis_r, modal_r = _assemble_shared_modal_basis(
        reference_mass_r,
        stiff_r_terms + mass_r_terms,
        jnp.concatenate([size_r_terms, size_t_terms + size_z_terms]),
    )
    basis_t, modal_t = _assemble_shared_modal_basis(
        reference_mass_t,
        stiff_t_terms + mass_t_terms,
        jnp.concatenate([size_t_terms, size_r_terms + size_z_terms]),
    )
    basis_z, modal_z = _assemble_shared_modal_basis(
        reference_mass_z,
        stiff_z_terms + mass_z_terms,
        jnp.concatenate([size_z_terms, size_r_terms + size_t_terms]),
    )

    return {
        'bulk_shape': bulk_shape,
        'bulk_term_mass_r': mass_r_terms,
        'bulk_term_mass_t': mass_t_terms,
        'bulk_term_mass_z': mass_z_terms,
        'bulk_term_stiff_r': stiff_r_terms,
        'bulk_term_stiff_t': stiff_t_terms,
        'bulk_term_stiff_z': stiff_z_terms,
        'bulk_modal_basis_r': basis_r,
        'bulk_modal_basis_t': basis_t,
        'bulk_modal_basis_z': basis_z,
        'bulk_modal_stiff_r': modal_r[:rank],
        'bulk_modal_stiff_t': modal_t[:rank],
        'bulk_modal_stiff_z': modal_z[:rank],
        'bulk_modal_mass_r': modal_r[rank:],
        'bulk_modal_mass_t': modal_t[rank:],
        'bulk_modal_mass_z': modal_z[rank:],
        'cp_relative_error': fits['cp_relative_error'],
        'cp_final_delta': fits['cp_final_delta'],
    }


def _build_k0_tensor_hodge_preconditioner_factors(
        *, core_size: int, schur_inv: jnp.ndarray, bulk_data: dict,
        schur_projector: Optional[jnp.ndarray] = None) -> K0TensorHodgePreconditionerFactors:
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
        bulk_modal_basis_r=bulk_data.get('bulk_modal_basis_r'),
        bulk_modal_basis_t=bulk_data.get('bulk_modal_basis_t'),
        bulk_modal_basis_z=bulk_data.get('bulk_modal_basis_z'),
        bulk_modal_mass_r=bulk_data.get('bulk_modal_mass_r', ()),
        bulk_modal_mass_t=bulk_data.get('bulk_modal_mass_t', ()),
        bulk_modal_mass_z=bulk_data.get('bulk_modal_mass_z', ()),
        bulk_modal_stiff_r=bulk_data.get('bulk_modal_stiff_r', ()),
        bulk_modal_stiff_t=bulk_data.get('bulk_modal_stiff_t', ()),
        bulk_modal_stiff_z=bulk_data.get('bulk_modal_stiff_z', ()),
        cp_relative_error=bulk_data.get('cp_relative_error'),
        cp_final_delta=bulk_data.get('cp_final_delta'),
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
        dirichlet_flags: tuple[bool, ...] = (False, True)) -> BoundaryConditionPair:
    pair = BoundaryConditionPair()
    core_size = _core_size(seq)

    for dirichlet in dirichlet_flags:
        if rank == 1:
            bulk_data = _assemble_k0_tensor_hodge_rank1_bulk_factors(
                seq,
                dirichlet=dirichlet,
                cp_maxiter=cp_maxiter,
                cp_tol=cp_tol,
                cp_ridge=cp_ridge,
            )
        else:
            bulk_data = _assemble_k0_tensor_hodge_multirank_bulk_factors(
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
        ))
        bulk_apply = lambda rhs_b, bulk_factors=bulk_factors: _apply_k0_tensor_hodge_bulk_inverse(bulk_factors, rhs_b)
        surgery_to_bulk_apply = lambda rhs_c, seq=seq, operators=operators, core_size=core_size, dirichlet=dirichlet: _apply_k0_tensor_hodge_surgery_to_bulk_coupling(seq, operators, core_size, rhs_c, dirichlet=dirichlet)
        bulk_to_surgery_apply = lambda rhs_b, seq=seq, operators=operators, core_size=core_size, dirichlet=dirichlet: _apply_k0_tensor_hodge_bulk_to_surgery_coupling(seq, operators, core_size, rhs_b, dirichlet=dirichlet)
        schur = _symmetrize(_assemble_dense_from_apply(
            lambda rhs_c, ass=ass, bulk_apply=bulk_apply, surgery_to_bulk_apply=surgery_to_bulk_apply, bulk_to_surgery_apply=bulk_to_surgery_apply:
            ass @ rhs_c - bulk_to_surgery_apply(bulk_apply(surgery_to_bulk_apply(rhs_c))),
            core_size,
        ))

        schur_projector = None
        if dirichlet:
            schur_inv = _symmetrize(jnp.linalg.inv(schur))
        else:
            schur_null = jnp.ones((core_size,), dtype=jnp.float64)
            schur_null_norm = jnp.linalg.norm(schur_null)
            schur_null = schur_null / jnp.where(schur_null_norm > 0, schur_null_norm, 1.0)
            schur_projector = jnp.eye(core_size, dtype=jnp.float64) - jnp.outer(schur_null, schur_null)
            schur_reg = _symmetrize(schur + jnp.outer(schur_null, schur_null))
            schur_inv = _symmetrize(jnp.linalg.inv(schur_reg))

        factors = _build_k0_tensor_hodge_preconditioner_factors(
            core_size=core_size,
            schur_inv=schur_inv,
            schur_projector=schur_projector,
            bulk_data=bulk_data,
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
    schur_rhs = rhs_c - _apply_k0_tensor_hodge_bulk_to_surgery_coupling(
        seq,
        operators,
        core_size,
        y,
        dirichlet=dirichlet,
    )
    if factors.schur_projector is not None:
        schur_rhs = factors.schur_projector @ schur_rhs
    z = factors.schur_inv @ schur_rhs
    if factors.schur_projector is not None:
        z = factors.schur_projector @ z
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
    if k in (0, 1, 2) and mass_preconds is not None and (
            mass_preconds.surgery is not None or mass_preconds.tensor is not None):
        mass_preconds = set_mass_surgery(
            mass_preconds,
            build_mass_surgery_preconditioner(
                seq,
                sp,
                k=k,
                existing=mass_preconds.surgery,
            ),
        )
    if mass_preconds is not None and mass_preconds.tensor is not None:
        tensor_rank = tensor_mass_rank_for_degree(mass_preconds.tensor, k)
        mass_preconds = set_mass_tensor(
            mass_preconds,
            build_mass_tensor_preconditioner(
                seq,
                sp,
                k=k,
                rank=tensor_rank,
                fallback_rank=mass_preconds.tensor.rank,
                cp_kwargs={
                    "maxiter": mass_preconds.tensor.cp_maxiter,
                    "tol": mass_preconds.tensor.cp_tol,
                    "ridge": mass_preconds.tensor.cp_ridge,
                    "block_chebyshev_steps": mass_preconds.tensor.block_chebyshev_steps,
                    "block_lanczos_iterations": mass_preconds.tensor.block_lanczos_iterations,
                    "block_lanczos_max_eig_inflation": mass_preconds.tensor.block_lanczos_max_eig_inflation,
                    "block_lanczos_min_eig_deflation": mass_preconds.tensor.block_lanczos_min_eig_deflation,
                    "block_lanczos_min_eig_floor_fraction": mass_preconds.tensor.block_lanczos_min_eig_floor_fraction,
                    "richardson_steps": mass_preconds.tensor.richardson_steps,
                    "richardson_omega": mass_preconds.tensor.richardson_omega,
                },
                existing=mass_preconds.tensor,
                surgery_precond=mass_preconds.surgery,
            ),
        )
    match k:
        case 0:
            return eqx.tree_at(
                lambda ops: (ops.m0, ops.mass_preconds),
                operators,
                (sp, mass_preconds),
                is_leaf=lambda x: x is None,
            )
        case 1:
            return eqx.tree_at(
                lambda ops: (ops.m1, ops.mass_preconds),
                operators,
                (sp, mass_preconds),
                is_leaf=lambda x: x is None,
            )
        case 2:
            return eqx.tree_at(
                lambda ops: (ops.m2, ops.mass_preconds),
                operators,
                (sp, mass_preconds),
                is_leaf=lambda x: x is None,
            )
        case 3:
            return eqx.tree_at(
                lambda ops: (ops.m3, ops.mass_preconds),
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


def assemble_mass_surgery_preconditioner(
        seq, operators: Optional[SequenceOperators] = None,
        *, ks: Sequence[int] = (0, 1, 2)):
    operators = _ensure_extraction_operators(seq, operators)
    missing_ks = []
    for k in ks:
        if k not in (0, 1, 2):
            raise ValueError("Mass surgery preconditioner assembly only supports k=0, k=1 and k=2")
        if getattr(operators, f"m{k}") is None:
            missing_ks.append(k)
    if missing_ks:
        operators = assemble_mass_operators(seq, seq.geometry, operators, ks=tuple(missing_ks))

    surgery_precond = operators.mass_preconds.surgery if operators.mass_preconds is not None else None
    for k in ks:
        surgery_precond = build_mass_surgery_preconditioner(
            seq,
            getattr(operators, f"m{k}"),
            k=k,
            existing=surgery_precond,
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
    rank: int = 1,
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
    surgery_ks = tuple(k for k in ks if k in (0, 1, 2))
    if surgery_ks:
        operators = assemble_mass_surgery_preconditioner(
            seq,
            operators=operators,
            ks=surgery_ks,
        )
    missing_ks = []
    for k in ks:
        if k not in (0, 1, 2, 3):
            raise ValueError("Tensor mass preconditioner assembly only supports k=0, k=1, k=2 and k=3")
        if getattr(operators, f"m{k}") is None:
            missing_ks.append(k)
    if missing_ks:
        operators = assemble_mass_operators(seq, seq.geometry, operators, ks=tuple(missing_ks))

    tensor_precond = operators.mass_preconds.tensor if operators.mass_preconds is not None else None
    for k in ks:
        tensor_rank = _tensor_mass_rank(rank, cp_kwargs, k)
        tensor_precond = build_mass_tensor_preconditioner(
            seq,
            getattr(operators, f"m{k}"),
            k=k,
            rank=tensor_rank,
            fallback_rank=rank,
            cp_kwargs=cp_kwargs,
            existing=tensor_precond,
            surgery_precond=operators.mass_preconds.surgery,
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


def _surgery_available(seq, operators: SequenceOperators, k: int) -> bool:
    return mass_surgery_available(seq, operators.mass_preconds, k)


def apply_mass_tensor_preconditioner_ops(
        seq, operators: SequenceOperators, v, k: int, dirichlet: bool = True):
    return apply_mass_tensor_preconditioner(
        seq, operators.mass_preconds, v, k, dirichlet=dirichlet)


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


def assemble_tensor_hodge_preconditioner(
    seq, operators: Optional[SequenceOperators] = None, *,
    rank: Optional[int] = None,
    cp_maxiter: Optional[int] = None,
    cp_tol: Optional[float] = None,
    cp_ridge: Optional[float] = None):
    """Precompute the tensor-Hodge auxiliary data for ``k = 0``.

    This stores the 1-D eigendecompositions used by the tensorized reference
    Hodge inverse. Geometry-independent; call once after ``seq.evaluate_1d()``.
    After this, :func:`apply_hodge_laplacian_preconditioner` with
    ``kind='tensor'`` becomes available for ``k = 0``.
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
    return eqx.tree_at(
        lambda ops: (
            ops.fd_V_p_r, ops.fd_V_p_t, ops.fd_V_p_z,
            ops.fd_lam_p_r, ops.fd_lam_p_t, ops.fd_lam_p_z,
        ),
        operators,
        (V_p_r, V_p_t, V_p_z, lam_p_r, lam_p_t, lam_p_z),
        is_leaf=lambda x: x is None,
    )


def _fd_hodge_scales_K(seq, geometry, k: int) -> jnp.ndarray:
    """Per-direction quadrature-averaged stiffness coefficients for ``k = 0``."""
    if k != 0:
        raise ValueError("FD Hodge scale currently implemented for k = 0 only")
    w = seq.quad.w
    w_sum = jnp.sum(w)
    J = geometry.jacobian_j
    g_inv = geometry.metric_inv_jkl  # (n_quad, 3, 3)
    return jnp.array([
        jnp.sum(J * g_inv[:, i, i] * w) / w_sum for i in range(3)
    ])


def _fd_hodge_available(operators: SequenceOperators, k: int) -> bool:
    """True iff the FD Hodge preconditioner can be applied for form ``k``."""
    if k != 0:
        return False
    needed = (
        operators.fd_V_p_r, operators.fd_V_p_t, operators.fd_V_p_z,
        operators.fd_lam_p_r, operators.fd_lam_p_t, operators.fd_lam_p_z,
        operators.dd0_fd_scale_K,
    )
    return all(x is not None for x in needed)


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
    if k != 0:
        raise ValueError("FD Hodge apply implemented for k = 0 only")
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


def apply_hodge_kron_preconditioner(
        seq, operators: SequenceOperators, v, k: int, dirichlet: bool = True,
        eps: float = 0.0):
    """Apply the legacy tensor-Hodge helper for ``k = 0`` only."""
    if k != 0:
        raise ValueError(
            f"Tensor Hodge preconditioner not available for k={k}; use 'jacobi' instead")
    if eps != 0.0:
        raise ValueError("Shifted legacy hodge-kron apply is not implemented for k=0")
    if not _k0_tensor_hodge_available(operators):
        raise ValueError(
            "Tensor Hodge preconditioner not available for k=0; "
            "assemble it via assemble_tensor_hodge_preconditioner"
        )
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
            return operators.m0, diaginv, diaginv_dbc
        case 1:
            return operators.m1, diaginv, diaginv_dbc
        case 2:
            return operators.m2, diaginv, diaginv_dbc
        case 3:
            return operators.m3, diaginv, diaginv_dbc
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
            return operators.grad_grad, operators.dd0_diaginv, operators.dd0_diaginv_dbc
        case 1:
            return operators.curl_curl, operators.dd1_diaginv, operators.dd1_diaginv_dbc
        case 2:
            return operators.div_div, operators.dd2_diaginv, operators.dd2_diaginv_dbc
        case 3:
            return None, operators.dd3_diaginv, operators.dd3_diaginv_dbc
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


def assemble_incidence_operators(seq, operators: Optional[SequenceOperators] = None,
                                 ks: Sequence[int] = (0, 1, 2)):
    """Assemble topological incidence operators for the requested degrees."""
    for k in ks:
        operators = update_incidence_operator(seq, operators, k)
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
            rank, cp_maxiter, cp_tol, cp_ridge = _k0_tensor_hodge_config(
                operators,
            )
            tensor_precond = _assemble_k0_tensor_hodge_preconditioner(
                seq,
                operators,
                rank=rank,
                cp_maxiter=cp_maxiter,
                cp_tol=cp_tol,
                cp_ridge=cp_ridge,
            )
            return eqx.tree_at(
                lambda ops: (ops.grad_grad, ops.dd0_diaginv,
                             ops.dd0_diaginv_dbc, ops.dd0_fd_scale_K,
                             ops.k0_tensor_hodge_precond),
                operators,
                (sp, diaginv, diaginv_dbc, fd_scale_K, tensor_precond),
                is_leaf=lambda x: x is None,
            )
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
            rank=1,
            cp_kwargs={
                'k0_rank': 2,
                'k1_rank': 2,
                'k2_rank': 2,
                'k3_rank': 2,
            },
        )
        operators = assemble_tensor_hodge_preconditioner(seq, operators=operators)
    operators = assemble_incidence_operators(seq, operators=operators)
    operators = assemble_derivative_operators(
        seq, geometry, operators=operators)
    operators = assemble_hodge_operators(seq, geometry, operators=operators)
    operators = assemble_projection_operators(seq, operators=operators)
    return operators


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
        spec: MassPreconditionerSpec, seed: int = 0,
        orthogonal_vectors=None):
    if spec.lanczos_iterations < 1:
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
            continue_iteration = (iteration + 1 < spec.lanczos_iterations) & (beta > tiny)
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
        jnp.zeros((spec.lanczos_iterations,), dtype=jnp.float64),
        jnp.zeros((spec.lanczos_iterations,), dtype=jnp.float64),
        jnp.asarray(True),
    )
    _, _, _, alphas, betas, _ = jax.lax.fori_loop(
        0,
        spec.lanczos_iterations,
        do_iteration,
        initial_state,
    )

    tridiagonal = jnp.diag(alphas)
    offdiag = betas[:-1]
    tridiagonal = tridiagonal + jnp.diag(offdiag, k=1) + jnp.diag(offdiag, k=-1)
    ritz_values = jnp.linalg.eigvalsh(tridiagonal)
    max_ritz = jnp.maximum(ritz_values[-1], tiny)
    max_eig = jnp.maximum(
        jnp.asarray(spec.lanczos_max_eig_inflation, dtype=jnp.float64) * max_ritz,
        tiny,
    )
    floor = jnp.asarray(
        spec.lanczos_min_eig_floor_fraction, dtype=jnp.float64
    ) * max_eig
    min_positive_ritz = jnp.min(jnp.where(ritz_values > tiny, ritz_values, jnp.inf))
    guarded_min = jnp.asarray(
        spec.lanczos_min_eig_deflation, dtype=jnp.float64
    ) * min_positive_ritz
    min_eig = jnp.where(
        jnp.isfinite(min_positive_ritz),
        jnp.maximum(floor, guarded_min),
        floor,
    )
    return min_eig, max_eig


def _diagonal_from_matvec(operator_apply, size: int):
    def entry(i):
        basis = jnp.zeros(size, dtype=jnp.float64).at[i].set(1.0)
        return operator_apply(basis)[i]

    return jax.lax.map(entry, jnp.arange(size))


def _invert_diagonal(diagonal):
    diagonal = jnp.asarray(diagonal, dtype=jnp.float64)
    return jnp.where(diagonal != 0.0, 1.0 / diagonal, 0.0)


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
            lambda rhs_s, surgery=surgery: _apply_k0_surgery_to_bulk_coupling(surgery, rhs_s),
            lambda rhs_b, surgery=surgery: _apply_k0_bulk_to_surgery_coupling(surgery, rhs_b),
        )
    if k == 1:
        return (
            lambda rhs_s, surgery=surgery: _apply_k1_surgery_to_bulk_coupling(surgery, rhs_s),
            lambda rhs_b, surgery=surgery: _apply_k1_bulk_to_surgery_coupling(surgery, rhs_b),
        )
    if k == 2:
        return (
            lambda rhs_s, surgery=surgery: _apply_k2_surgery_to_bulk_coupling(surgery, rhs_s),
            lambda rhs_b, surgery=surgery: _apply_k2_bulk_to_surgery_coupling(surgery, rhs_b),
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


def _build_mass_surgery_bulk_apply(
        seq, operators: SequenceOperators, *, k: int, dirichlet: bool,
        spec: MassPreconditionerSpec):
    surgery = _select_mass_surgery_factors(operators.mass_preconds, k, dirichlet)
    kind = spec.kind

    if k == 1 and spec.surgery_schur:
        diaginv = _mass_diaginv(operators, k, dirichlet)
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
        diaginv = _mass_diaginv(operators, k, dirichlet)
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
        bulk_diaginv = _mass_diaginv(operators, k, dirichlet)[surgery.surgery_size:]
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
        diaginv = _mass_diaginv(operators, k, dirichlet)
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
        bulk_diaginv = _mass_diaginv(operators, k, dirichlet)[surgery.surgery_size:]
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
        bulk_diaginv = _mass_diaginv(operators, k, dirichlet)[surgery.surgery_size:]
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
        schur_inv = _assemble_schur_inverse_from_applies(
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
    schur_inv = _assemble_schur_inverse_from_applies(
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
        diaginv = _mass_diaginv(operators, k, dirichlet)
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
        diaginv = _mass_diaginv(operators, k, dirichlet)
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
        return _hodge_diaginv(operators, k, dirichlet) * v
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
    if saddle_preconditioner.schur.outer.kind == 'exact_jacobi':
        schur_probe_apply = _build_schur_operator_apply(
            seq,
            operators,
            k=k,
            dirichlet=dirichlet,
            eps=eps,
            inner_preconditioner_apply=precond_lower,
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
        precond_upper = _build_operator_preconditioner_apply(
            seq,
            operators,
            k=k,
            dirichlet=dirichlet,
            operator_apply=schur_apply,
            preconditioner=saddle_preconditioner.schur.outer,
            allow_none=True,
            orthogonal_vectors=vs_upper if eps == 0.0 else None,
            runtime_tuning=_select_schur_runtime_tuning(
                operators,
                k,
                dirichlet,
                eps,
            ),
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

    Out-of-the-box diffusion preconditioners use the same basic mass-side
    building blocks as the other solver paths, but ``preconditioner='auto'``
    now prefers the plain mass-tensor apply on the upper diffusion block when
    that tensor data is available. Polynomial variants such as Chebyshev
    remain available through explicit specs.

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
