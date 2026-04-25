# %% [markdown]
# # k=1 Polar Mass Surgery Debug
#
# This interactive script isolates the extracted-space surgery rows for the
# polar `k=1` mass matrix.
#
# The immediate questions are:
#
# 1. which extracted rows are dense,
# 2. which raw components they touch,
# 3. and what the corresponding block structure of
#
#        A = E_1 M_{1,raw} E_1^T
#
#    looks like.
#
# The expected row families are:
#
# - `r`: ordinary selector/permutation rows,
# - `theta_surgery`: first `2 * n_z` rows of the extracted `theta` block,
# - `theta_bulk`: remaining `theta` rows,
# - `zeta_surgery`: first `3 * d_z` rows of the extracted `zeta` block,
# - `zeta_bulk`: remaining `zeta` rows.

# %%
from __future__ import annotations

import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import rotating_ellipse_map, toroid_map
from mrx.operators import (
    apply_mass_matrix,
    apply_mass_matrix_preconditioner,
    assemble_kron_mass_preconditioner,
    assemble_mass_operators,
    dense_mass_matrix,
)
from mrx.solvers import solve_singular_cg

jax.config.update("jax_enable_x64", True)


# %% Configuration
@dataclass(frozen=True)
class ExperimentConfig:
    ns: tuple[int, int, int] = (8, 8, 8)
    p: int = 2
    tol: float = 1e-10
    maxiter: int = 2000
    betti: tuple[int, int, int, int] = (1, 1, 0, 0)
    map_kind: str = "rotating_ellipse"
    torus_epsilon: float = 1.0 / 3.0
    torus_r0: float = 1.0
    rotating_eps: float = 0.2
    rotating_kappa: float = 1.1
    rotating_r0: float = 1.0
    rotating_nfp: int = 2


@dataclass
class ExtractionSupportSummary:
    label: str
    n_rows: int
    total_min: int
    total_max: int
    total_mean: float
    raw_r_mean: float
    raw_theta_mean: float
    raw_zeta_mean: float


@dataclass
class RtZInverseDiagnostics:
    relative_operator_error: float
    relative_solve_error: float
    relative_residual: float
    min_eigenvalue: float
    max_eigenvalue: float


@dataclass
class CoupledBlockDiagnostics:
    offdiag_relative_frobenius: float
    exact_blockdiag_operator_error: float
    mixed_blockdiag_operator_error: float
    exact_blockdiag_solve_error: float
    mixed_blockdiag_solve_error: float
    exact_blockdiag_residual: float
    mixed_blockdiag_residual: float


@dataclass
class CoupledSchurDiagnostics:
    exact_schur_operator_error: float
    mixed_schur_operator_error: float
    exact_schur_solve_error: float
    mixed_schur_solve_error: float
    exact_schur_residual: float
    mixed_schur_residual: float


@dataclass
class ThetaSchurCorrectionDiagnostics:
    exact_correction_relative_to_atheta: float
    mixed_correction_relative_to_atheta: float
    mixed_vs_exact_correction_error: float
    exact_schur_relative_to_atheta: float
    mixed_schur_vs_exact_error: float


@dataclass
class BulkRtZetaDiagnostics:
    offdiag_relative_frobenius: float
    exact_blockdiag_operator_error: float
    mixed_blockdiag_operator_error: float
    exact_blockdiag_solve_error: float
    mixed_blockdiag_solve_error: float
    exact_blockdiag_residual: float
    mixed_blockdiag_residual: float


@dataclass
class FullSurgerySchurDiagnostics:
    exact_schur_operator_error: float
    mixed_schur_operator_error: float
    exact_schur_solve_error: float
    mixed_schur_solve_error: float
    exact_schur_residual: float
    mixed_schur_residual: float


@dataclass
class K1BenchmarkReport:
    label: str
    dirichlet: bool
    n_rhs: int
    n_converged: int
    avg_iters: float
    std_iters: float
    max_iters: int
    avg_time_ms: float
    avg_relative_residual: float


@dataclass
class MetricFactorFitDiagnostics:
    label: str
    rank: int
    relative_error: float
    max_abs_error: float
    n_iters: int
    final_delta: float


@dataclass
class KroneckerBlockDiagnostics:
    label: str
    relative_operator_error: float
    relative_inverse_operator_error: float
    relative_solve_error: float
    relative_residual: float


CONFIG = ExperimentConfig()
SEQ = None
OPERATORS = None
BUILT_CONFIG = None


# %% Helpers
def _build_map(config: ExperimentConfig):
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


def build_case(config: ExperimentConfig = CONFIG):
    seq = DeRhamSequence(
        config.ns,
        (config.p, config.p, config.p),
        2 * config.p,
        ("clamped", "periodic", "periodic"),
        lambda x: x,
        polar=True,
        tol=config.tol,
        maxiter=config.maxiter,
        betti_numbers=config.betti,
    )
    seq.evaluate_1d()
    seq.assemble_reference_mass_matrix()
    seq.set_map(_build_map(config))
    operators = assemble_mass_operators(seq, seq.geometry, ks=(1,))
    return seq, operators


def ensure_built(config: ExperimentConfig = CONFIG, rebuild: bool = False):
    global SEQ, OPERATORS, BUILT_CONFIG
    if rebuild or SEQ is None or OPERATORS is None or BUILT_CONFIG != config:
        SEQ, OPERATORS = build_case(config)
        BUILT_CONFIG = config
    return SEQ, OPERATORS


def _component_sizes(seq, dirichlet: bool):
    if dirichlet:
        return seq.n1_1_dbc, seq.n1_2_dbc, seq.n1_3_dbc
    return seq.n1_1, seq.n1_2, seq.n1_3


def _component_slices(seq, dirichlet: bool):
    n_r, n_theta, n_zeta = _component_sizes(seq, dirichlet)
    r_slice = slice(0, n_r)
    theta_slice = slice(r_slice.stop, r_slice.stop + n_theta)
    zeta_slice = slice(theta_slice.stop, theta_slice.stop + n_zeta)
    return {
        "r": r_slice,
        "theta": theta_slice,
        "zeta": zeta_slice,
    }


def _surgery_slices(seq, dirichlet: bool):
    component_slices = _component_slices(seq, dirichlet)
    theta = component_slices["theta"]
    zeta = component_slices["zeta"]
    theta_surgery = slice(theta.start, theta.start + 2 * seq.basis_1.nz)
    theta_bulk = slice(theta_surgery.stop, theta.stop)
    zeta_surgery = slice(zeta.start, zeta.start + 3 * seq.basis_1.dz)
    zeta_bulk = slice(zeta_surgery.stop, zeta.stop)
    return {
        "r": component_slices["r"],
        "theta_surgery": theta_surgery,
        "theta_bulk": theta_bulk,
        "zeta_surgery": zeta_surgery,
        "zeta_bulk": zeta_bulk,
    }


def _raw_component_slices(seq):
    n_r = seq.basis_1.n1
    n_theta = seq.basis_1.n2
    n_zeta = seq.basis_1.n3
    r_slice = slice(0, n_r)
    theta_slice = slice(r_slice.stop, r_slice.stop + n_theta)
    zeta_slice = slice(theta_slice.stop, theta_slice.stop + n_zeta)
    return {
        "raw_r": r_slice,
        "raw_theta": theta_slice,
        "raw_zeta": zeta_slice,
    }


def dense_extraction_matrix(seq, dirichlet: bool) -> jnp.ndarray:
    e = seq.e1_dbc if dirichlet else seq.e1
    return jnp.asarray(e.todense())


def dense_raw_mass_matrix(operators) -> jnp.ndarray:
    return jnp.asarray(operators.m1_sp.todense())


def dense_extracted_mass_matrix(seq, operators, dirichlet: bool) -> jnp.ndarray:
    return jnp.asarray(dense_mass_matrix(seq, operators, 1, dirichlet=dirichlet))


def verify_extracted_mass_identity(seq, operators, dirichlet: bool):
    e = dense_extraction_matrix(seq, dirichlet)
    e_t = e.T
    m_raw = dense_raw_mass_matrix(operators)
    a_ref = dense_extracted_mass_matrix(seq, operators, dirichlet)
    a_sandwich = e @ m_raw @ e_t
    diff = jnp.max(jnp.abs(a_ref - a_sandwich))
    return a_ref, a_sandwich, float(diff)


def summarize_extraction_support(seq, dirichlet: bool):
    e = dense_extraction_matrix(seq, dirichlet)
    raw_slices = _raw_component_slices(seq)
    surgery = _surgery_slices(seq, dirichlet)

    reports = []
    for label in ("theta_surgery", "zeta_surgery", "theta_bulk", "zeta_bulk", "r"):
        row_slice = surgery[label]
        rows = e[row_slice, :]
        total = jnp.count_nonzero(rows, axis=1)
        raw_r = jnp.count_nonzero(rows[:, raw_slices["raw_r"]], axis=1)
        raw_theta = jnp.count_nonzero(rows[:, raw_slices["raw_theta"]], axis=1)
        raw_zeta = jnp.count_nonzero(rows[:, raw_slices["raw_zeta"]], axis=1)
        reports.append(
            ExtractionSupportSummary(
                label=label,
                n_rows=rows.shape[0],
                total_min=int(jnp.min(total)) if rows.shape[0] else 0,
                total_max=int(jnp.max(total)) if rows.shape[0] else 0,
                total_mean=float(jnp.mean(total)) if rows.shape[0] else 0.0,
                raw_r_mean=float(jnp.mean(raw_r)) if rows.shape[0] else 0.0,
                raw_theta_mean=float(jnp.mean(raw_theta)) if rows.shape[0] else 0.0,
                raw_zeta_mean=float(jnp.mean(raw_zeta)) if rows.shape[0] else 0.0,
            )
        )
    return reports


def print_support_summary(reports):
    print("-" * 112)
    print(
        f"{'label':<16} {'n_rows':>8} {'nnz[min,max,mean]':>24} "
        f"{'raw_r':>10} {'raw_theta':>12} {'raw_zeta':>11}"
    )
    for report in reports:
        print(
            f"{report.label:<16} {report.n_rows:>8d} "
            f"[{report.total_min:>3d}, {report.total_max:>3d}, {report.total_mean:>6.1f}] "
            f"{report.raw_r_mean:>10.1f} {report.raw_theta_mean:>12.1f} {report.raw_zeta_mean:>11.1f}"
        )


def block_norm_table(matrix: jnp.ndarray, slices: dict[str, slice]):
    labels = list(slices.keys())
    total = float(jnp.linalg.norm(matrix))
    table = []
    for row_label in labels:
        row = []
        for col_label in labels:
            block = matrix[slices[row_label], slices[col_label]]
            fro = float(jnp.linalg.norm(block))
            row.append((fro, fro / total if total > 0 else 0.0))
        table.append(row)
    return labels, table, total


def print_block_norm_table(labels, table, total_norm: float):
    print("-" * 112)
    print(f"total Frobenius norm: {total_norm:.6e}")
    header = " " * 18 + " ".join(f"{label:>18}" for label in labels)
    print(header)
    for row_label, row in zip(labels, table):
        entries = " ".join(f"{fro:8.2e}/{rel:7.3f}" for fro, rel in row)
        print(f"{row_label:<18} {entries}")


def reorder_indices(seq, dirichlet: bool):
    slices = _surgery_slices(seq, dirichlet)
    labels = ("theta_surgery", "zeta_surgery", "r", "theta_bulk", "zeta_bulk")
    blocks = [jnp.arange(slices[label].start, slices[label].stop) for label in labels]
    permutation = jnp.concatenate(blocks)
    reordered_slices = {}
    offset = 0
    for label, block in zip(labels, blocks):
        reordered_slices[label] = slice(offset, offset + block.shape[0])
        offset += block.shape[0]
    return labels, permutation, reordered_slices


def reorder_matrix(matrix: jnp.ndarray, permutation: jnp.ndarray) -> jnp.ndarray:
    return matrix[permutation][:, permutation]


def schur_blocks(matrix: jnp.ndarray, surgery_size: int):
    ass = matrix[:surgery_size, :surgery_size]
    asb = matrix[:surgery_size, surgery_size:]
    abs_ = matrix[surgery_size:, :surgery_size]
    abb = matrix[surgery_size:, surgery_size:]
    return ass, asb, abs_, abb


def dense_schur_inverse_apply(matrix: jnp.ndarray, surgery_size: int, rhs: jnp.ndarray):
    ass, asb, abs_, abb = schur_blocks(matrix, surgery_size)
    abb_inv = jnp.linalg.inv(abb)
    schur = ass - asb @ abb_inv @ abs_
    schur_inv = jnp.linalg.inv(schur)

    rhs_s = rhs[:surgery_size]
    rhs_b = rhs[surgery_size:]
    y = abb_inv @ rhs_b
    z = schur_inv @ (rhs_s - asb @ y)
    x_b = y - abb_inv @ (abs_ @ z)
    return jnp.concatenate([z, x_b]), schur, schur_inv


def bulk_component_slices(reordered_slices: dict[str, slice]):
    surgery_size = reordered_slices["zeta_surgery"].stop
    theta_bulk = reordered_slices["theta_bulk"]
    zeta_bulk = reordered_slices["zeta_bulk"]
    return {
        "r": slice(
            reordered_slices["r"].start - surgery_size,
            reordered_slices["r"].stop - surgery_size,
        ),
        "theta_bulk": slice(
            theta_bulk.start - surgery_size,
            theta_bulk.stop - surgery_size,
        ),
        "zeta_bulk": slice(
            zeta_bulk.start - surgery_size,
            zeta_bulk.stop - surgery_size,
        ),
    }


def _symmetrize(matrix: jnp.ndarray) -> jnp.ndarray:
    return 0.5 * (matrix + matrix.T)


def quadrature_tensor_shape(seq) -> tuple[int, int, int]:
    if seq is None:
        seq, _ = ensure_built(CONFIG)
    return seq.quad.ny, seq.quad.nx, seq.quad.nz


def reshape_quadrature_scalar_field(seq, values: jnp.ndarray) -> jnp.ndarray:
    return jnp.asarray(values).reshape(quadrature_tensor_shape(seq))


def reshape_quadrature_matrix_field(seq, values: jnp.ndarray) -> jnp.ndarray:
    field = jnp.asarray(values)
    return field.reshape(*quadrature_tensor_shape(seq), *field.shape[1:])


def k1_diagonal_metric_tensors(seq) -> dict[str, jnp.ndarray]:
    jacobian = reshape_quadrature_scalar_field(seq, seq.jacobian_j)
    metric_inv = reshape_quadrature_matrix_field(seq, seq.metric_inv_jkl)
    return {
        "alpha_rr": jacobian * metric_inv[..., 0, 0],
        "alpha_thetatheta": jacobian * metric_inv[..., 1, 1],
        "alpha_zetazeta": jacobian * metric_inv[..., 2, 2],
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


def reconstruct_cp_3tensor(
    weights: jnp.ndarray,
    factors: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> jnp.ndarray:
    factor_r, factor_t, factor_z = factors
    return jnp.einsum("r,ir,jr,kr->ijk", weights, factor_r, factor_t, factor_z)


def cp_als_3tensor(
    tensor: jnp.ndarray,
    rank: int,
    *,
    maxiter: int = 100,
    tol: float = 1e-10,
    ridge: float = 1e-12,
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], MetricFactorFitDiagnostics]:
    if tensor.ndim != 3:
        raise ValueError(f"CP-ALS expects a 3-tensor, got shape {tensor.shape}")
    if rank < 1 or rank > min(tensor.shape):
        raise ValueError(f"Requested CP rank {rank} outside valid range [1, {min(tensor.shape)}]")

    unfolded_0 = _mode_unfold_3tensor(tensor, 0)
    unfolded_1 = _mode_unfold_3tensor(tensor, 1)
    unfolded_2 = _mode_unfold_3tensor(tensor, 2)

    factor_r = jnp.linalg.svd(unfolded_0, full_matrices=False)[0][:, :rank]
    factor_t = jnp.linalg.svd(unfolded_1, full_matrices=False)[0][:, :rank]
    factor_z = jnp.linalg.svd(unfolded_2, full_matrices=False)[0][:, :rank]
    factor_r, _ = _normalize_cp_columns(factor_r)
    factor_t, _ = _normalize_cp_columns(factor_t)
    factor_z, _ = _normalize_cp_columns(factor_z)
    weights = jnp.ones((rank,), dtype=tensor.dtype)

    tensor_norm = jnp.linalg.norm(tensor)
    tensor_norm_safe = jnp.where(tensor_norm > 0, tensor_norm, 1.0)
    eye = jnp.eye(rank, dtype=tensor.dtype)
    previous_error = jnp.inf
    final_delta = jnp.inf
    n_iters = 0

    for iteration in range(1, maxiter + 1):
        factor_z_eff = factor_z * weights[None, :]

        khatri_rao_tz = _khatri_rao(factor_t, factor_z_eff)
        gram_tz = (factor_t.T @ factor_t) * (factor_z_eff.T @ factor_z_eff)
        factor_r_raw = jnp.linalg.solve(gram_tz + ridge * eye, (unfolded_0 @ khatri_rao_tz).T).T

        khatri_rao_rz = _khatri_rao(factor_r_raw, factor_z_eff)
        gram_rz = (factor_r_raw.T @ factor_r_raw) * (factor_z_eff.T @ factor_z_eff)
        factor_t_raw = jnp.linalg.solve(gram_rz + ridge * eye, (unfolded_1 @ khatri_rao_rz).T).T

        khatri_rao_rt = _khatri_rao(factor_r_raw, factor_t_raw)
        gram_rt = (factor_r_raw.T @ factor_r_raw) * (factor_t_raw.T @ factor_t_raw)
        factor_z_eff_raw = jnp.linalg.solve(gram_rt + ridge * eye, (unfolded_2 @ khatri_rao_rt).T).T

        factor_r, r_norms = _normalize_cp_columns(factor_r_raw)
        factor_t, t_norms = _normalize_cp_columns(factor_t_raw)
        factor_z_temp = factor_z_eff_raw * (r_norms * t_norms)[None, :]
        factor_z, weights = _normalize_cp_columns(factor_z_temp)

        reconstruction = reconstruct_cp_3tensor(weights, (factor_r, factor_t, factor_z))
        relative_error = float(jnp.linalg.norm(reconstruction - tensor) / tensor_norm_safe)
        final_delta = abs(relative_error - previous_error) if previous_error < jnp.inf else jnp.inf
        previous_error = relative_error
        n_iters = iteration
        if final_delta < tol:
            break

    reconstruction = reconstruct_cp_3tensor(weights, (factor_r, factor_t, factor_z))
    diagnostics = MetricFactorFitDiagnostics(
        label="",
        rank=rank,
        relative_error=float(jnp.linalg.norm(reconstruction - tensor) / tensor_norm_safe),
        max_abs_error=float(jnp.max(jnp.abs(reconstruction - tensor))),
        n_iters=n_iters,
        final_delta=float(final_delta),
    )
    return weights, (factor_r, factor_t, factor_z), diagnostics


def print_metric_factor_fit_reports(reports: list[MetricFactorFitDiagnostics]):
    print("-" * 112)
    print(
        f"{'field':<20} {'rank':>6} {'rel err':>12} {'max abs':>12} {'iters':>8} {'delta':>12}"
    )
    for report in reports:
        print(
            f"{report.label:<20} {report.rank:>6d} {report.relative_error:>12.3e} {report.max_abs_error:>12.3e} "
            f"{report.n_iters:>8d} {report.final_delta:>12.3e}"
        )


def _assemble_weighted_1d_mass(basis: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    basis = jnp.asarray(basis)
    weights = jnp.asarray(weights)
    if basis.ndim != 2 or weights.ndim != 1:
        raise ValueError(
            f"weighted 1d mass expects a rank-2 basis and rank-1 weights, got {basis.shape} and {weights.shape}"
        )
    if basis.shape[1] == weights.shape[0]:
        return (basis * weights[None, :]) @ basis.T
    if basis.shape[0] == weights.shape[0]:
        return (basis * weights[:, None]).T @ basis
    raise ValueError(
        f"cannot align quadrature weights of length {weights.shape[0]} with basis shape {basis.shape}"
    )


def kron_apply_3(
    matrix_r: jnp.ndarray,
    matrix_t: jnp.ndarray,
    matrix_z: jnp.ndarray,
    rhs: jnp.ndarray,
    shape: tuple[int, int, int],
) -> jnp.ndarray:
    nr, nt, nz = shape
    tensor = rhs.reshape(nr, nt, nz)
    out = jnp.einsum("ai,ijk->ajk", matrix_r, tensor)
    out = jnp.einsum("bj,ajk->abk", matrix_t, out)
    out = jnp.einsum("ck,abk->abc", matrix_z, out)
    return out.reshape(-1)


def _restrict_radial_mass(raw_mass_r: jnp.ndarray, radial_start: int, nr: int) -> jnp.ndarray:
    radial_stop = radial_start + nr
    if radial_start < 0 or radial_stop > raw_mass_r.shape[0]:
        raise ValueError(
            f"requested extracted radial window [{radial_start}, {radial_stop}) outside raw radial size {raw_mass_r.shape[0]}"
        )
    return raw_mass_r[radial_start:radial_stop, radial_start:radial_stop]


def _sum_kron_apply_3(
    terms: tuple[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], ...],
    rhs: jnp.ndarray,
    shape: tuple[int, int, int],
) -> jnp.ndarray:
    out = jnp.zeros_like(rhs)
    for matrix_r, matrix_t, matrix_z in terms:
        out = out + kron_apply_3(matrix_r, matrix_t, matrix_z, rhs, shape)
    return out


def _neumann_inverse_apply(
    dominant_term: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    residual_terms: tuple[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], ...],
    rhs: jnp.ndarray,
    shape: tuple[int, int, int],
    order: int,
) -> jnp.ndarray:
    dominant_r, dominant_t, dominant_z = dominant_term
    dominant_r_inv = jnp.linalg.inv(dominant_r)
    dominant_t_inv = jnp.linalg.inv(dominant_t)
    dominant_z_inv = jnp.linalg.inv(dominant_z)

    def dominant_inv_apply(vec: jnp.ndarray) -> jnp.ndarray:
        return kron_apply_3(dominant_r_inv, dominant_t_inv, dominant_z_inv, vec, shape)

    out = dominant_inv_apply(rhs)
    correction = out
    sign = -1.0
    for _ in range(order):
        correction = dominant_inv_apply(_sum_kron_apply_3(residual_terms, correction, shape))
        out = out + sign * correction
        sign *= -1.0
    return out


def build_diagonal_metric_cp_kron_approx(
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
    neumann_order: int = 2,
) -> tuple[jnp.ndarray, callable, callable]:
    weights, factors, _ = cp_als_3tensor(tensor, rank=rank, maxiter=100, tol=1e-10, ridge=1e-12)
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
    kron_matrix = jnp.zeros((nr * nt * nz, nr * nt * nz), dtype=tensor.dtype)

    for mass_r, mass_t, mass_z in term_matrices:
        kron_matrix = kron_matrix + jnp.kron(jnp.kron(mass_r, mass_t), mass_z)

    def apply(rhs: jnp.ndarray) -> jnp.ndarray:
        return _sum_kron_apply_3(term_matrices, rhs, (nr, nt, nz))

    def apply_inv(rhs: jnp.ndarray) -> jnp.ndarray:
        if len(term_matrices) == 1:
            mass_r, mass_t, mass_z = term_matrices[0]
            return kron_apply_3(
                jnp.linalg.inv(mass_r),
                jnp.linalg.inv(mass_t),
                jnp.linalg.inv(mass_z),
                rhs,
                (nr, nt, nz),
            )
        return _neumann_inverse_apply(
            term_matrices[0],
            term_matrices[1:],
            rhs,
            (nr, nt, nz),
            neumann_order,
        )

    return kron_matrix, apply, apply_inv


def build_arr_metric_cp_kron_approx(
    seq,
    tensor: jnp.ndarray,
    full_shape: tuple[int, int, int],
    rank: int,
) -> tuple[jnp.ndarray, callable, callable]:
    return build_diagonal_metric_cp_kron_approx(
        seq,
        tensor,
        full_shape,
        rank,
        radial_basis=seq.d_basis_r_jk,
        theta_basis=seq.basis_t_jk,
        zeta_basis=seq.basis_z_jk,
        radial_weights=seq.quad.w_x,
        theta_weights=seq.quad.w_y,
        zeta_weights=seq.quad.w_z,
        radial_start=1,
    )


def build_theta_metric_cp_kron_approx(
    seq,
    tensor: jnp.ndarray,
    full_shape: tuple[int, int, int],
    rank: int,
) -> tuple[jnp.ndarray, callable, callable]:
    return build_diagonal_metric_cp_kron_approx(
        seq,
        tensor,
        full_shape,
        rank,
        radial_basis=seq.basis_r_jk,
        theta_basis=seq.d_basis_t_jk,
        zeta_basis=seq.basis_z_jk,
        radial_weights=seq.quad.w_x,
        theta_weights=seq.quad.w_y,
        zeta_weights=seq.quad.w_z,
        radial_start=2,
    )


def build_zeta_metric_cp_kron_approx(
    seq,
    tensor: jnp.ndarray,
    full_shape: tuple[int, int, int],
    rank: int,
) -> tuple[jnp.ndarray, callable, callable]:
    return build_diagonal_metric_cp_kron_approx(
        seq,
        tensor,
        full_shape,
        rank,
        radial_basis=seq.basis_r_jk,
        theta_basis=seq.basis_t_jk,
        zeta_basis=seq.d_basis_z_jk,
        radial_weights=seq.quad.w_x,
        theta_weights=seq.quad.w_y,
        zeta_weights=seq.quad.w_z,
        radial_start=2,
    )


def arr_metric_cp_kron_diagnostics(
    seq,
    matrix: jnp.ndarray,
    tensor: jnp.ndarray,
    full_shape: tuple[int, int, int],
    rank: int,
) -> KroneckerBlockDiagnostics:
    kron_matrix, _, apply_inv = build_arr_metric_cp_kron_approx(seq, tensor, full_shape, rank)
    exact_inv = jnp.linalg.inv(matrix)
    approx_inv = _build_inverse_from_apply(apply_inv, matrix.shape[0], matrix.dtype)

    key = jax.random.PRNGKey(37)
    rhs = jax.random.normal(key, (matrix.shape[0],), dtype=matrix.dtype)
    exact_sol = exact_inv @ rhs
    approx_sol = apply_inv(rhs)
    residual = matrix @ approx_sol - rhs

    return KroneckerBlockDiagnostics(
        label=f"A_rr-rank{rank}-metric-kron",
        relative_operator_error=float(
            jnp.linalg.norm(kron_matrix - matrix) / jnp.linalg.norm(matrix)
        ),
        relative_inverse_operator_error=float(
            jnp.linalg.norm(approx_inv - exact_inv) / jnp.linalg.norm(exact_inv)
        ),
        relative_solve_error=float(
            jnp.linalg.norm(approx_sol - exact_sol) / jnp.linalg.norm(exact_sol)
        ),
        relative_residual=float(jnp.linalg.norm(residual) / jnp.linalg.norm(rhs)),
    )


def diagonal_metric_cp_kron_diagnostics(
    matrix: jnp.ndarray,
    kron_matrix: jnp.ndarray,
    apply_inv,
    *,
    label: str,
) -> KroneckerBlockDiagnostics:
    exact_inv = jnp.linalg.inv(matrix)
    approx_inv = _build_inverse_from_apply(apply_inv, matrix.shape[0], matrix.dtype)

    key = jax.random.PRNGKey(37)
    rhs = jax.random.normal(key, (matrix.shape[0],), dtype=matrix.dtype)
    exact_sol = exact_inv @ rhs
    approx_sol = apply_inv(rhs)
    residual = matrix @ approx_sol - rhs

    return KroneckerBlockDiagnostics(
        label=label,
        relative_operator_error=float(
            jnp.linalg.norm(kron_matrix - matrix) / jnp.linalg.norm(matrix)
        ),
        relative_inverse_operator_error=float(
            jnp.linalg.norm(approx_inv - exact_inv) / jnp.linalg.norm(exact_inv)
        ),
        relative_solve_error=float(
            jnp.linalg.norm(approx_sol - exact_sol) / jnp.linalg.norm(exact_sol)
        ),
        relative_residual=float(jnp.linalg.norm(residual) / jnp.linalg.norm(rhs)),
    )


def _rt_z_tensor_from_matrix(matrix: jnp.ndarray, full_shape: tuple[int, int, int]) -> jnp.ndarray:
    nr, nt, nz = full_shape
    rt_size = nr * nt
    return matrix.reshape(nr, nt, nz, nr, nt, nz).transpose(0, 1, 3, 4, 2, 5).reshape(
        rt_size, rt_size, nz, nz
    )


def _apply_rt_z_block_inverse(
    rt_block_inverses: tuple[jnp.ndarray, ...],
    z_basis: jnp.ndarray,
    rhs: jnp.ndarray,
    rt_size: int,
) -> jnp.ndarray:
    x = rhs.reshape(rt_size, z_basis.shape[0])
    x_hat = x @ z_basis
    y_hat = jnp.stack(
        [block_inv @ x_hat[:, idx] for idx, block_inv in enumerate(rt_block_inverses)],
        axis=1,
    )
    y = y_hat @ z_basis.T
    return y.reshape(-1)


def arr_shape(seq, dirichlet: bool) -> tuple[int, int, int]:
    if seq is None:
        seq, _ = ensure_built(CONFIG)
    nt = seq.basis_1.nt
    nz = seq.basis_1.nz
    n_r = _component_sizes(seq, dirichlet)[0]
    if n_r % (nt * nz) != 0:
        raise ValueError(
            f"Extracted r size {n_r} is not divisible by nt*nz = {nt * nz}"
        )
    nr = n_r // (nt * nz)
    return nr, nt, nz


def theta_bulk_shape(seq, dirichlet: bool) -> tuple[int, int, int]:
    if seq is None:
        seq, _ = ensure_built(CONFIG)
    dt = seq.basis_1.dt
    nz = seq.basis_1.nz
    n_theta = _component_sizes(seq, dirichlet)[1] - 2 * seq.basis_1.nz
    if n_theta % (dt * nz) != 0:
        raise ValueError(
            f"theta_bulk size {n_theta} is not divisible by dt*nz = {dt * nz}"
        )
    nr = n_theta // (dt * nz)
    return nr, dt, nz


def zeta_bulk_shape(seq, dirichlet: bool) -> tuple[int, int, int]:
    if seq is None:
        seq, _ = ensure_built(CONFIG)
    nt = seq.basis_1.nt
    dz = seq.basis_1.dz
    n_zeta = _component_sizes(seq, dirichlet)[2] - 3 * seq.basis_1.dz
    if n_zeta % (nt * dz) != 0:
        raise ValueError(
            f"zeta_bulk size {n_zeta} is not divisible by nt*dz = {nt * dz}"
        )
    nr = n_zeta // (nt * dz)
    return nr, nt, dz


def arr_block_from_reordered(matrix: jnp.ndarray, reordered_slices: dict[str, slice]) -> jnp.ndarray:
    r_slice = reordered_slices["r"]
    return matrix[r_slice, r_slice]


def diagonal_block(matrix: jnp.ndarray, block_slice: slice) -> jnp.ndarray:
    return matrix[block_slice, block_slice]


def _truncate_rt_z_rank(
    z_basis: jnp.ndarray,
    z_eigenvalues: jnp.ndarray,
    z_rank: int | None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    max_rank = z_basis.shape[1]
    if z_rank is None:
        return z_basis, z_eigenvalues
    if z_rank < 1 or z_rank > max_rank:
        raise ValueError(f"Requested rt-z rank {z_rank} outside valid range [1, {max_rank}]")
    return z_basis[:, :z_rank], z_eigenvalues[:z_rank]


def build_diagonal_rt_z_apply(
    matrix: jnp.ndarray,
    full_shape: tuple[int, int, int],
    z_rank: int | None = None,
):
    z_basis, _, _, rt_block_inverses = build_arr_rt_z_inverse(matrix, full_shape, z_rank=z_rank)
    return lambda rhs, rb=rt_block_inverses, zb=z_basis, fs=full_shape: apply_arr_rt_z_inverse(
        rb, zb, rhs, fs
    )


def build_arr_rt_z_inverse(
    matrix: jnp.ndarray,
    full_shape: tuple[int, int, int],
    z_rank: int | None = None,
):
    tensor = _rt_z_tensor_from_matrix(matrix, full_shape)
    rt_size = full_shape[0] * full_shape[1]

    z_average = _symmetrize(jnp.einsum("ijab->ab", tensor) / rt_size)
    z_eigenvalues, z_basis = jnp.linalg.eigh(z_average)
    order = jnp.argsort(z_eigenvalues)[::-1]
    z_eigenvalues = z_eigenvalues[order]
    z_basis = z_basis[:, order]
    z_basis, z_eigenvalues = _truncate_rt_z_rank(z_basis, z_eigenvalues, z_rank)

    rt_blocks = []
    rt_block_inverses = []
    for idx in range(z_basis.shape[1]):
        q = z_basis[:, idx]
        block = _symmetrize(jnp.einsum("ijab,a,b->ij", tensor, q, q))
        rt_blocks.append(block)
        rt_block_inverses.append(jnp.linalg.inv(block))

    return z_basis, z_eigenvalues, tuple(rt_blocks), tuple(rt_block_inverses)


def apply_arr_rt_z_inverse(
    rt_block_inverses: tuple[jnp.ndarray, ...],
    z_basis: jnp.ndarray,
    rhs: jnp.ndarray,
    full_shape: tuple[int, int, int],
) -> jnp.ndarray:
    rt_size = full_shape[0] * full_shape[1]
    return _apply_rt_z_block_inverse(rt_block_inverses, z_basis, rhs, rt_size)


def arr_rt_z_diagnostics(matrix: jnp.ndarray, full_shape: tuple[int, int, int]) -> RtZInverseDiagnostics:
    z_basis, _, _, rt_block_inverses = build_arr_rt_z_inverse(matrix, full_shape)
    exact_inv = jnp.linalg.inv(matrix)

    eye = jnp.eye(matrix.shape[0], dtype=matrix.dtype)
    approx_inv_columns = [
        apply_arr_rt_z_inverse(rt_block_inverses, z_basis, eye[:, idx], full_shape)
        for idx in range(matrix.shape[0])
    ]
    approx_inv = jnp.stack(approx_inv_columns, axis=1)

    key = jax.random.PRNGKey(0)
    rhs = jax.random.normal(key, (matrix.shape[0],), dtype=matrix.dtype)
    exact_sol = exact_inv @ rhs
    approx_sol = apply_arr_rt_z_inverse(rt_block_inverses, z_basis, rhs, full_shape)
    residual = matrix @ approx_sol - rhs

    eigvals = jnp.linalg.eigvalsh(_symmetrize(matrix))
    return RtZInverseDiagnostics(
        relative_operator_error=float(
            jnp.linalg.norm(approx_inv - exact_inv) / jnp.linalg.norm(exact_inv)
        ),
        relative_solve_error=float(
            jnp.linalg.norm(approx_sol - exact_sol) / jnp.linalg.norm(exact_sol)
        ),
        relative_residual=float(jnp.linalg.norm(residual) / jnp.linalg.norm(rhs)),
        min_eigenvalue=float(jnp.min(eigvals)),
        max_eigenvalue=float(jnp.max(eigvals)),
    )


def coupled_r_theta_block(matrix: jnp.ndarray, bulk_slices: dict[str, slice]) -> jnp.ndarray:
    r_slice = bulk_slices["r"]
    theta_slice = bulk_slices["theta_bulk"]
    indices = jnp.concatenate(
        [
            jnp.arange(r_slice.start, r_slice.stop),
            jnp.arange(theta_slice.start, theta_slice.stop),
        ]
    )
    return matrix[indices][:, indices]


def coupled_r_theta_block_slices(bulk_slices: dict[str, slice]) -> dict[str, slice]:
    n_r = bulk_slices["r"].stop - bulk_slices["r"].start
    n_theta = bulk_slices["theta_bulk"].stop - bulk_slices["theta_bulk"].start
    return {
        "r": slice(0, n_r),
        "theta_bulk": slice(n_r, n_r + n_theta),
    }


def _block_diag_apply(
    arr_apply,
    theta_inv: jnp.ndarray,
    rhs: jnp.ndarray,
    block_slices: dict[str, slice],
) -> jnp.ndarray:
    rhs_r = rhs[block_slices["r"]]
    rhs_theta = rhs[block_slices["theta_bulk"]]
    x_r = arr_apply(rhs_r)
    x_theta = theta_inv @ rhs_theta
    return jnp.concatenate([x_r, x_theta])


def coupled_block_diagnostics(
    block: jnp.ndarray,
    block_slices: dict[str, slice],
    arr_exact_inv: jnp.ndarray,
    arr_rt_z_apply,
) -> CoupledBlockDiagnostics:
    arr = block[block_slices["r"], block_slices["r"]]
    art = block[block_slices["r"], block_slices["theta_bulk"]]
    atr = block[block_slices["theta_bulk"], block_slices["r"]]
    atheta = block[block_slices["theta_bulk"], block_slices["theta_bulk"]]
    theta_inv = jnp.linalg.inv(atheta)
    exact_inv = jnp.linalg.inv(block)

    offdiag = jnp.sqrt(jnp.linalg.norm(art) ** 2 + jnp.linalg.norm(atr) ** 2)
    offdiag_relative = float(offdiag / jnp.linalg.norm(block))

    eye = jnp.eye(block.shape[0], dtype=block.dtype)
    exact_blockdiag_cols = [
        _block_diag_apply(lambda rhs: arr_exact_inv @ rhs, theta_inv, eye[:, idx], block_slices)
        for idx in range(block.shape[0])
    ]
    mixed_blockdiag_cols = [
        _block_diag_apply(arr_rt_z_apply, theta_inv, eye[:, idx], block_slices)
        for idx in range(block.shape[0])
    ]
    exact_blockdiag_inv = jnp.stack(exact_blockdiag_cols, axis=1)
    mixed_blockdiag_inv = jnp.stack(mixed_blockdiag_cols, axis=1)

    key = jax.random.PRNGKey(11)
    rhs = jax.random.normal(key, (block.shape[0],), dtype=block.dtype)
    exact_sol = exact_inv @ rhs
    exact_blockdiag_sol = _block_diag_apply(
        lambda vec: arr_exact_inv @ vec,
        theta_inv,
        rhs,
        block_slices,
    )
    mixed_blockdiag_sol = _block_diag_apply(arr_rt_z_apply, theta_inv, rhs, block_slices)

    exact_blockdiag_residual = block @ exact_blockdiag_sol - rhs
    mixed_blockdiag_residual = block @ mixed_blockdiag_sol - rhs

    return CoupledBlockDiagnostics(
        offdiag_relative_frobenius=offdiag_relative,
        exact_blockdiag_operator_error=float(
            jnp.linalg.norm(exact_blockdiag_inv - exact_inv) / jnp.linalg.norm(exact_inv)
        ),
        mixed_blockdiag_operator_error=float(
            jnp.linalg.norm(mixed_blockdiag_inv - exact_inv) / jnp.linalg.norm(exact_inv)
        ),
        exact_blockdiag_solve_error=float(
            jnp.linalg.norm(exact_blockdiag_sol - exact_sol) / jnp.linalg.norm(exact_sol)
        ),
        mixed_blockdiag_solve_error=float(
            jnp.linalg.norm(mixed_blockdiag_sol - exact_sol) / jnp.linalg.norm(exact_sol)
        ),
        exact_blockdiag_residual=float(
            jnp.linalg.norm(exact_blockdiag_residual) / jnp.linalg.norm(rhs)
        ),
        mixed_blockdiag_residual=float(
            jnp.linalg.norm(mixed_blockdiag_residual) / jnp.linalg.norm(rhs)
        ),
    )


def _build_inverse_from_apply(apply_fn, size: int, dtype) -> jnp.ndarray:
    eye = jnp.eye(size, dtype=dtype)
    return jnp.stack([apply_fn(eye[:, idx]) for idx in range(size)], axis=1)


def schur_r_theta_apply(
    block: jnp.ndarray,
    block_slices: dict[str, slice],
    arr_apply,
) -> tuple[callable, jnp.ndarray]:
    art = block[block_slices["r"], block_slices["theta_bulk"]]
    atr = block[block_slices["theta_bulk"], block_slices["r"]]
    atheta = block[block_slices["theta_bulk"], block_slices["theta_bulk"]]

    u_cols = [arr_apply(art[:, idx]) for idx in range(art.shape[1])]
    u = jnp.stack(u_cols, axis=1)
    schur = atheta - atr @ u
    schur_inv = jnp.linalg.inv(schur)

    def apply(rhs: jnp.ndarray) -> jnp.ndarray:
        rhs_r = rhs[block_slices["r"]]
        rhs_theta = rhs[block_slices["theta_bulk"]]
        y = arr_apply(rhs_r)
        z = schur_inv @ (rhs_theta - atr @ y)
        x_r = y - arr_apply(art @ z)
        return jnp.concatenate([x_r, z])

    return apply, schur


def schur_r_theta_apply_lowrank(
    block: jnp.ndarray,
    block_slices: dict[str, slice],
    arr_apply,
    theta_apply,
) -> callable:
    art = block[block_slices["r"], block_slices["theta_bulk"]]
    atr = block[block_slices["theta_bulk"], block_slices["r"]]

    def apply(rhs: jnp.ndarray) -> jnp.ndarray:
        rhs_r = rhs[block_slices["r"]]
        rhs_theta = rhs[block_slices["theta_bulk"]]
        y = arr_apply(rhs_r)
        z = theta_apply(rhs_theta - atr @ y)
        x_r = y - arr_apply(art @ z)
        return jnp.concatenate([x_r, z])

    return apply


def coupled_schur_diagnostics(
    block: jnp.ndarray,
    block_slices: dict[str, slice],
    arr_exact_inv: jnp.ndarray,
    arr_rt_z_apply,
) -> CoupledSchurDiagnostics:
    exact_inv = jnp.linalg.inv(block)
    exact_apply, _ = schur_r_theta_apply(block, block_slices, lambda rhs: arr_exact_inv @ rhs)
    mixed_apply, _ = schur_r_theta_apply(block, block_slices, arr_rt_z_apply)

    exact_schur_inv = _build_inverse_from_apply(exact_apply, block.shape[0], block.dtype)
    mixed_schur_inv = _build_inverse_from_apply(mixed_apply, block.shape[0], block.dtype)

    key = jax.random.PRNGKey(17)
    rhs = jax.random.normal(key, (block.shape[0],), dtype=block.dtype)
    exact_sol = exact_inv @ rhs
    exact_schur_sol = exact_apply(rhs)
    mixed_schur_sol = mixed_apply(rhs)
    exact_schur_residual = block @ exact_schur_sol - rhs
    mixed_schur_residual = block @ mixed_schur_sol - rhs

    return CoupledSchurDiagnostics(
        exact_schur_operator_error=float(
            jnp.linalg.norm(exact_schur_inv - exact_inv) / jnp.linalg.norm(exact_inv)
        ),
        mixed_schur_operator_error=float(
            jnp.linalg.norm(mixed_schur_inv - exact_inv) / jnp.linalg.norm(exact_inv)
        ),
        exact_schur_solve_error=float(
            jnp.linalg.norm(exact_schur_sol - exact_sol) / jnp.linalg.norm(exact_sol)
        ),
        mixed_schur_solve_error=float(
            jnp.linalg.norm(mixed_schur_sol - exact_sol) / jnp.linalg.norm(exact_sol)
        ),
        exact_schur_residual=float(
            jnp.linalg.norm(exact_schur_residual) / jnp.linalg.norm(rhs)
        ),
        mixed_schur_residual=float(
            jnp.linalg.norm(mixed_schur_residual) / jnp.linalg.norm(rhs)
        ),
    )


def theta_schur_correction_diagnostics(
    block: jnp.ndarray,
    block_slices: dict[str, slice],
    arr_exact_inv: jnp.ndarray,
    arr_rt_z_apply,
) -> ThetaSchurCorrectionDiagnostics:
    art = block[block_slices["r"], block_slices["theta_bulk"]]
    atr = block[block_slices["theta_bulk"], block_slices["r"]]
    atheta = block[block_slices["theta_bulk"], block_slices["theta_bulk"]]

    exact_correction = atr @ arr_exact_inv @ art
    mixed_u = jnp.stack([arr_rt_z_apply(art[:, idx]) for idx in range(art.shape[1])], axis=1)
    mixed_correction = atr @ mixed_u

    exact_schur = atheta - exact_correction
    mixed_schur = atheta - mixed_correction

    atheta_norm = jnp.linalg.norm(atheta)
    exact_correction_norm = jnp.linalg.norm(exact_correction)
    mixed_correction_norm = jnp.linalg.norm(mixed_correction)
    exact_schur_norm = jnp.linalg.norm(exact_schur)

    return ThetaSchurCorrectionDiagnostics(
        exact_correction_relative_to_atheta=float(exact_correction_norm / atheta_norm),
        mixed_correction_relative_to_atheta=float(mixed_correction_norm / atheta_norm),
        mixed_vs_exact_correction_error=float(
            jnp.linalg.norm(mixed_correction - exact_correction) / exact_correction_norm
        ),
        exact_schur_relative_to_atheta=float(exact_schur_norm / atheta_norm),
        mixed_schur_vs_exact_error=float(
            jnp.linalg.norm(mixed_schur - exact_schur) / jnp.linalg.norm(exact_schur)
        ),
    )


def rt_zeta_block_slices(bulk_slices: dict[str, slice]) -> dict[str, slice]:
    n_r = bulk_slices["r"].stop - bulk_slices["r"].start
    n_theta = bulk_slices["theta_bulk"].stop - bulk_slices["theta_bulk"].start
    n_zeta = bulk_slices["zeta_bulk"].stop - bulk_slices["zeta_bulk"].start
    return {
        "rt": slice(0, n_r + n_theta),
        "zeta_bulk": slice(n_r + n_theta, n_r + n_theta + n_zeta),
    }


def _two_block_diag_apply(
    rt_apply,
    zeta_apply,
    rhs: jnp.ndarray,
    block_slices: dict[str, slice],
) -> jnp.ndarray:
    rhs_rt = rhs[block_slices["rt"]]
    rhs_zeta = rhs[block_slices["zeta_bulk"]]
    x_rt = rt_apply(rhs_rt)
    x_zeta = zeta_apply(rhs_zeta)
    return jnp.concatenate([x_rt, x_zeta])


def bulk_rt_zeta_diagnostics(
    bulk: jnp.ndarray,
    block_slices: dict[str, slice],
    exact_rt_apply,
    mixed_rt_apply,
) -> BulkRtZetaDiagnostics:
    zeta = bulk[block_slices["zeta_bulk"], block_slices["zeta_bulk"]]
    brz = bulk[block_slices["rt"], block_slices["zeta_bulk"]]
    bzr = bulk[block_slices["zeta_bulk"], block_slices["rt"]]
    zeta_inv = jnp.linalg.inv(zeta)
    zeta_apply = lambda rhs, zi=zeta_inv: zi @ rhs
    exact_inv = jnp.linalg.inv(bulk)

    offdiag = jnp.sqrt(jnp.linalg.norm(brz) ** 2 + jnp.linalg.norm(bzr) ** 2)
    offdiag_relative = float(offdiag / jnp.linalg.norm(bulk))

    exact_blockdiag_inv = _build_inverse_from_apply(
        lambda rhs: _two_block_diag_apply(exact_rt_apply, zeta_apply, rhs, block_slices),
        bulk.shape[0],
        bulk.dtype,
    )
    mixed_blockdiag_inv = _build_inverse_from_apply(
        lambda rhs: _two_block_diag_apply(mixed_rt_apply, zeta_apply, rhs, block_slices),
        bulk.shape[0],
        bulk.dtype,
    )

    key = jax.random.PRNGKey(23)
    rhs = jax.random.normal(key, (bulk.shape[0],), dtype=bulk.dtype)
    exact_sol = exact_inv @ rhs
    exact_blockdiag_sol = _two_block_diag_apply(exact_rt_apply, zeta_apply, rhs, block_slices)
    mixed_blockdiag_sol = _two_block_diag_apply(mixed_rt_apply, zeta_apply, rhs, block_slices)
    exact_blockdiag_residual = bulk @ exact_blockdiag_sol - rhs
    mixed_blockdiag_residual = bulk @ mixed_blockdiag_sol - rhs

    return BulkRtZetaDiagnostics(
        offdiag_relative_frobenius=offdiag_relative,
        exact_blockdiag_operator_error=float(
            jnp.linalg.norm(exact_blockdiag_inv - exact_inv) / jnp.linalg.norm(exact_inv)
        ),
        mixed_blockdiag_operator_error=float(
            jnp.linalg.norm(mixed_blockdiag_inv - exact_inv) / jnp.linalg.norm(exact_inv)
        ),
        exact_blockdiag_solve_error=float(
            jnp.linalg.norm(exact_blockdiag_sol - exact_sol) / jnp.linalg.norm(exact_sol)
        ),
        mixed_blockdiag_solve_error=float(
            jnp.linalg.norm(mixed_blockdiag_sol - exact_sol) / jnp.linalg.norm(exact_sol)
        ),
        exact_blockdiag_residual=float(
            jnp.linalg.norm(exact_blockdiag_residual) / jnp.linalg.norm(rhs)
        ),
        mixed_blockdiag_residual=float(
            jnp.linalg.norm(mixed_blockdiag_residual) / jnp.linalg.norm(rhs)
        ),
    )


def outer_surgery_schur_apply(
    matrix: jnp.ndarray,
    surgery_size: int,
    bulk_apply,
) -> tuple[callable, jnp.ndarray]:
    ass, asb, abs_, _ = schur_blocks(matrix, surgery_size)
    u_cols = [bulk_apply(abs_[:, idx]) for idx in range(abs_.shape[1])]
    u = jnp.stack(u_cols, axis=1)
    schur = ass - asb @ u
    schur_inv = jnp.linalg.inv(schur)

    def apply(rhs: jnp.ndarray) -> jnp.ndarray:
        rhs_s = rhs[:surgery_size]
        rhs_b = rhs[surgery_size:]
        y = bulk_apply(rhs_b)
        z = schur_inv @ (rhs_s - asb @ y)
        x_b = y - bulk_apply(abs_ @ z)
        return jnp.concatenate([z, x_b])

    return apply, schur


def full_surgery_schur_diagnostics(
    matrix: jnp.ndarray,
    surgery_size: int,
    exact_bulk_apply,
    mixed_bulk_apply,
) -> FullSurgerySchurDiagnostics:
    exact_inv = jnp.linalg.inv(matrix)
    exact_apply, _ = outer_surgery_schur_apply(matrix, surgery_size, exact_bulk_apply)
    mixed_apply, _ = outer_surgery_schur_apply(matrix, surgery_size, mixed_bulk_apply)

    exact_schur_inv = _build_inverse_from_apply(exact_apply, matrix.shape[0], matrix.dtype)
    mixed_schur_inv = _build_inverse_from_apply(mixed_apply, matrix.shape[0], matrix.dtype)

    key = jax.random.PRNGKey(29)
    rhs = jax.random.normal(key, (matrix.shape[0],), dtype=matrix.dtype)
    exact_sol = exact_inv @ rhs
    exact_schur_sol = exact_apply(rhs)
    mixed_schur_sol = mixed_apply(rhs)
    exact_schur_residual = matrix @ exact_schur_sol - rhs
    mixed_schur_residual = matrix @ mixed_schur_sol - rhs

    return FullSurgerySchurDiagnostics(
        exact_schur_operator_error=float(
            jnp.linalg.norm(exact_schur_inv - exact_inv) / jnp.linalg.norm(exact_inv)
        ),
        mixed_schur_operator_error=float(
            jnp.linalg.norm(mixed_schur_inv - exact_inv) / jnp.linalg.norm(exact_inv)
        ),
        exact_schur_solve_error=float(
            jnp.linalg.norm(exact_schur_sol - exact_sol) / jnp.linalg.norm(exact_sol)
        ),
        mixed_schur_solve_error=float(
            jnp.linalg.norm(mixed_schur_sol - exact_sol) / jnp.linalg.norm(exact_sol)
        ),
        exact_schur_residual=float(
            jnp.linalg.norm(exact_schur_residual) / jnp.linalg.norm(rhs)
        ),
        mixed_schur_residual=float(
            jnp.linalg.norm(mixed_schur_residual) / jnp.linalg.norm(rhs)
        ),
    )


def full_apply_diagnostics(
    matrix: jnp.ndarray,
    exact_apply,
    mixed_apply,
) -> FullSurgerySchurDiagnostics:
    exact_inv = jnp.linalg.inv(matrix)
    exact_inv_from_apply = _build_inverse_from_apply(exact_apply, matrix.shape[0], matrix.dtype)
    mixed_inv_from_apply = _build_inverse_from_apply(mixed_apply, matrix.shape[0], matrix.dtype)

    key = jax.random.PRNGKey(31)
    rhs = jax.random.normal(key, (matrix.shape[0],), dtype=matrix.dtype)
    exact_sol = exact_inv @ rhs
    exact_apply_sol = exact_apply(rhs)
    mixed_apply_sol = mixed_apply(rhs)
    exact_apply_residual = matrix @ exact_apply_sol - rhs
    mixed_apply_residual = matrix @ mixed_apply_sol - rhs

    return FullSurgerySchurDiagnostics(
        exact_schur_operator_error=float(
            jnp.linalg.norm(exact_inv_from_apply - exact_inv) / jnp.linalg.norm(exact_inv)
        ),
        mixed_schur_operator_error=float(
            jnp.linalg.norm(mixed_inv_from_apply - exact_inv) / jnp.linalg.norm(exact_inv)
        ),
        exact_schur_solve_error=float(
            jnp.linalg.norm(exact_apply_sol - exact_sol) / jnp.linalg.norm(exact_sol)
        ),
        mixed_schur_solve_error=float(
            jnp.linalg.norm(mixed_apply_sol - exact_sol) / jnp.linalg.norm(exact_sol)
        ),
        exact_schur_residual=float(
            jnp.linalg.norm(exact_apply_residual) / jnp.linalg.norm(rhs)
        ),
        mixed_schur_residual=float(
            jnp.linalg.norm(mixed_apply_residual) / jnp.linalg.norm(rhs)
        ),
    )


def bulk_rt_zeta_schur_apply(
    bulk: jnp.ndarray,
    block_slices: dict[str, slice],
    rt_apply,
) -> tuple[callable, jnp.ndarray]:
    brz = bulk[block_slices["rt"], block_slices["zeta_bulk"]]
    bzr = bulk[block_slices["zeta_bulk"], block_slices["rt"]]
    zeta = bulk[block_slices["zeta_bulk"], block_slices["zeta_bulk"]]
    rt_size = brz.shape[0]
    zeta_size = zeta.shape[0]

    u_cols = [rt_apply(brz[:, idx]) for idx in range(brz.shape[1])]
    u = jnp.stack(u_cols, axis=1)
    schur = zeta - bzr @ u
    schur_inv = jnp.linalg.inv(schur)

    def apply(rhs: jnp.ndarray) -> jnp.ndarray:
        if rhs.shape[0] != rt_size + zeta_size:
            raise ValueError(
                f"bulk rt-zeta apply expected rhs of length {rt_size + zeta_size}, got {rhs.shape[0]}"
            )
        rhs_rt = rhs[:rt_size]
        rhs_zeta = rhs[rt_size:rt_size + zeta_size]
        y = rt_apply(rhs_rt)
        z = schur_inv @ (rhs_zeta - bzr @ y)
        x_rt = y - rt_apply(brz @ z)
        return jnp.concatenate([x_rt, z])

    return apply, schur


def build_k1_full_model_applies(
    dirichlet: bool,
    *,
    couple_zeta: bool,
    arr_mode: str = "rt-z",
    theta_mode: str = "dense",
    zeta_mode: str = "rt-z",
    rt_z_rank: int | None = None,
    metric_rank: int = 3,
) -> tuple[callable, callable]:
    data = K1_REORDERED[dirichlet]
    matrix = data["matrix"]
    surgery_size = data["slices"]["zeta_surgery"].stop

    bulk = K1_BULK_BLOCK_NORMS[dirichlet]["matrix"]
    bulk_block_slices = rt_zeta_block_slices(K1_BULK_BLOCK_NORMS[dirichlet]["slices"])
    bulk_slices = K1_BULK_BLOCK_NORMS[dirichlet]["slices"]
    rt_block = coupled_r_theta_block(bulk, bulk_slices)
    rt_block_slices = coupled_r_theta_block_slices(bulk_slices)
    arr = ARR_BLOCKS[dirichlet]["matrix"]
    full_shape = ARR_BLOCKS[dirichlet]["shape"]

    arr_exact_inv = jnp.linalg.inv(arr)
    if arr_mode == "rt-z":
        z_basis, _, _, rt_block_inverses = build_arr_rt_z_inverse(arr, full_shape, z_rank=rt_z_rank)
        arr_apply = lambda rhs, rb=rt_block_inverses, zb=z_basis, fs=full_shape: apply_arr_rt_z_inverse(
            rb, zb, rhs, fs
        )
    elif arr_mode == "metric-kron":
        _, _, arr_apply = build_arr_metric_cp_kron_approx(
            SEQ,
            K1_METRIC_CP["alpha_rr"]["tensor"],
            full_shape,
            metric_rank,
        )
    else:
        raise ValueError(f"Unsupported arr_mode: {arr_mode}")

    atheta = diagonal_block(rt_block, rt_block_slices["theta_bulk"])
    atheta_shape = theta_bulk_shape(SEQ, dirichlet)
    theta_rt_z_apply = build_diagonal_rt_z_apply(atheta, atheta_shape, z_rank=rt_z_rank)
    _, _, theta_metric_apply = build_theta_metric_cp_kron_approx(
        SEQ,
        K1_METRIC_CP["alpha_thetatheta"]["tensor"],
        atheta_shape,
        metric_rank,
    )
    if theta_mode == "dense":
        exact_rt_apply, _ = schur_r_theta_apply(
            rt_block,
            rt_block_slices,
            lambda rhs: arr_exact_inv @ rhs,
        )
        mixed_rt_apply, _ = schur_r_theta_apply(rt_block, rt_block_slices, arr_apply)
    elif theta_mode == "rt-z-lowrank":
        exact_rt_apply = schur_r_theta_apply_lowrank(
            rt_block,
            rt_block_slices,
            lambda rhs: arr_exact_inv @ rhs,
            theta_rt_z_apply,
        )
        mixed_rt_apply = schur_r_theta_apply_lowrank(
            rt_block,
            rt_block_slices,
            arr_apply,
            theta_rt_z_apply,
        )
    elif theta_mode == "metric-kron":
        exact_rt_apply = schur_r_theta_apply_lowrank(
            rt_block,
            rt_block_slices,
            lambda rhs: arr_exact_inv @ rhs,
            theta_metric_apply,
        )
        mixed_rt_apply = schur_r_theta_apply_lowrank(
            rt_block,
            rt_block_slices,
            arr_apply,
            theta_metric_apply,
        )
    else:
        raise ValueError(f"Unsupported theta_mode: {theta_mode}")

    zeta = bulk[bulk_block_slices["zeta_bulk"], bulk_block_slices["zeta_bulk"]]
    zeta_shape = zeta_bulk_shape(SEQ, dirichlet)
    zeta_exact_apply = lambda rhs, zi=jnp.linalg.inv(zeta): zi @ rhs
    zeta_rt_z_apply = build_diagonal_rt_z_apply(zeta, zeta_shape, z_rank=rt_z_rank)
    _, _, zeta_metric_apply = build_zeta_metric_cp_kron_approx(
        SEQ,
        K1_METRIC_CP["alpha_zetazeta"]["tensor"],
        zeta_shape,
        metric_rank,
    )
    if zeta_mode == "exact":
        zeta_apply = zeta_exact_apply
    elif zeta_mode == "rt-z":
        zeta_apply = zeta_rt_z_apply
    elif zeta_mode == "metric-kron":
        zeta_apply = zeta_metric_apply
    else:
        raise ValueError(f"Unsupported zeta_mode: {zeta_mode}")
    if couple_zeta:
        exact_bulk_apply, _ = bulk_rt_zeta_schur_apply(bulk, bulk_block_slices, exact_rt_apply)
        mixed_bulk_apply, _ = bulk_rt_zeta_schur_apply(bulk, bulk_block_slices, mixed_rt_apply)
    else:
        exact_bulk_apply = lambda rhs, rta=exact_rt_apply, za=zeta_apply, bs=bulk_block_slices: _two_block_diag_apply(
            rta, za, rhs, bs
        )
        mixed_bulk_apply = lambda rhs, rta=mixed_rt_apply, za=zeta_apply, bs=bulk_block_slices: _two_block_diag_apply(
            rta, za, rhs, bs
        )

    exact_apply, _ = outer_surgery_schur_apply(matrix, surgery_size, exact_bulk_apply)
    mixed_apply, _ = outer_surgery_schur_apply(matrix, surgery_size, mixed_bulk_apply)
    return exact_apply, mixed_apply


def wrap_reordered_apply(permutation: jnp.ndarray, reordered_apply) -> callable:
    inverse_permutation = jnp.argsort(permutation)

    def apply(rhs: jnp.ndarray) -> jnp.ndarray:
        rhs_perm = rhs[permutation]
        sol_perm = reordered_apply(rhs_perm)
        return sol_perm[inverse_permutation]

    return apply


def benchmark_k1_preconditioners(
    seq,
    operators,
    *,
    dirichlet: bool,
    n_rhs: int = 8,
    seed: int = 0,
) -> list[K1BenchmarkReport]:
    rhs_size = seq.n1_dbc if dirichlet else seq.n1
    rhs_batch = jax.random.normal(
        jax.random.PRNGKey(seed + 100 * int(dirichlet)),
        (n_rhs, rhs_size),
        dtype=jnp.float64,
    )
    permutation = K1_REORDERED[dirichlet]["permutation"]
    _, mixed_full_apply = build_k1_full_model_applies(
        dirichlet,
        couple_zeta=False,
        arr_mode="rt-z",
        theta_mode="dense",
        zeta_mode="rt-z",
    )
    _, mixed_full_coupled_apply = build_k1_full_model_applies(
        dirichlet,
        couple_zeta=True,
        arr_mode="rt-z",
        theta_mode="dense",
    )
    _, mixed_full_lowrank_apply = build_k1_full_model_applies(
        dirichlet,
        couple_zeta=False,
        arr_mode="rt-z",
        theta_mode="rt-z-lowrank",
        zeta_mode="rt-z",
    )
    _, mixed_full_lowrank_coupled_apply = build_k1_full_model_applies(
        dirichlet,
        couple_zeta=True,
        arr_mode="rt-z",
        theta_mode="rt-z-lowrank",
        zeta_mode="rt-z",
    )
    _, mixed_full_metric_apply = build_k1_full_model_applies(
        dirichlet,
        couple_zeta=False,
        arr_mode="metric-kron",
        theta_mode="metric-kron",
        zeta_mode="metric-kron",
        metric_rank=1,
    )
    surgery_apply = wrap_reordered_apply(permutation, mixed_full_apply)
    surgery_coupled_apply = wrap_reordered_apply(permutation, mixed_full_coupled_apply)
    surgery_lowrank_apply = wrap_reordered_apply(permutation, mixed_full_lowrank_apply)
    surgery_lowrank_coupled_apply = wrap_reordered_apply(permutation, mixed_full_lowrank_coupled_apply)
    surgery_metric_apply = wrap_reordered_apply(permutation, mixed_full_metric_apply)

    labels = {
        "jacobi": lambda x: apply_mass_matrix_preconditioner(
            seq, operators, x, 1, dirichlet=dirichlet, kind="jacobi"
        ),
        "kronecker": lambda x: apply_mass_matrix_preconditioner(
            seq, operators, x, 1, dirichlet=dirichlet, kind="kronecker"
        ),
        "k1-surgery": surgery_apply,
        "k1-surgery-metric-r1": surgery_metric_apply,
        "k1-surgery-lowrank": surgery_lowrank_apply,
        "k1-surgery-lowrank-coupled-zeta": surgery_lowrank_coupled_apply,
        "k1-surgery-coupled-zeta": surgery_coupled_apply,
    }

    return benchmark_preconditioner_applies(
        seq,
        operators,
        labels,
        dirichlet=dirichlet,
        rhs_batch=rhs_batch,
    )


def benchmark_preconditioner_applies(
    seq,
    operators,
    labels: dict[str, callable],
    *,
    dirichlet: bool,
    rhs_batch: jnp.ndarray,
) -> list[K1BenchmarkReport]:
    n_rhs = rhs_batch.shape[0]

    def A_mv(x):
        return apply_mass_matrix(seq, operators, x, 1, dirichlet=dirichlet)

    def mass_norm(x):
        return jnp.sqrt(jnp.abs(jnp.dot(x, A_mv(x))))

    reports = []
    for label, M_mv in labels.items():
        @jax.jit
        def solve(rhs):
            x, info = solve_singular_cg(
                A_mv,
                rhs,
                mass_matvec=A_mv,
                precond_matvec=M_mv,
                tol=seq.tol,
                maxiter=seq.maxiter,
            )
            residual = A_mv(x) - rhs
            relative_residual = mass_norm(residual) / jnp.where(mass_norm(rhs) > 0, mass_norm(rhs), 1.0)
            return x, info, relative_residual

        x0, _, _ = solve(rhs_batch[0])
        jax.block_until_ready(x0)

        iterations = []
        converged = []
        times_ms = []
        residuals = []
        for rhs in rhs_batch:
            t0 = time.perf_counter()
            x, info, residual = solve(rhs)
            jax.block_until_ready(x)
            times_ms.append((time.perf_counter() - t0) * 1e3)
            iterations.append(int(jnp.abs(info)))
            converged.append(bool(residual < seq.tol))
            residuals.append(float(residual))

        reports.append(
            K1BenchmarkReport(
                label=label,
                dirichlet=dirichlet,
                n_rhs=n_rhs,
                n_converged=int(sum(converged)),
                avg_iters=float(jnp.mean(jnp.asarray(iterations))),
                std_iters=float(jnp.std(jnp.asarray(iterations))),
                max_iters=int(jnp.max(jnp.asarray(iterations))),
                avg_time_ms=float(jnp.mean(jnp.asarray(times_ms))),
                avg_relative_residual=float(jnp.mean(jnp.asarray(residuals))),
            )
        )
    return reports


def k1_rt_z_max_rank(seq, dirichlet: bool) -> int:
    return min(
        arr_shape(seq, dirichlet)[2],
        theta_bulk_shape(seq, dirichlet)[2],
        zeta_bulk_shape(seq, dirichlet)[2],
    )


def benchmark_k1_lowrank_rank_ablation(
    seq,
    operators,
    *,
    dirichlet: bool,
    ranks: tuple[int, ...] | None = None,
    n_rhs: int = 8,
    seed: int = 0,
    couple_zeta: bool = False,
) -> list[K1BenchmarkReport]:
    rhs_size = seq.n1_dbc if dirichlet else seq.n1
    rhs_batch = jax.random.normal(
        jax.random.PRNGKey(seed + 100 * int(dirichlet)),
        (n_rhs, rhs_size),
        dtype=jnp.float64,
    )
    max_rank = k1_rt_z_max_rank(seq, dirichlet)
    if ranks is None:
        ranks = tuple(range(1, max_rank + 1))
    permutation = K1_REORDERED[dirichlet]["permutation"]

    labels = {}
    for rank in ranks:
        _, mixed_apply = build_k1_full_model_applies(
            dirichlet,
            couple_zeta=couple_zeta,
            theta_mode="rt-z-lowrank",
            zeta_mode="rt-z",
            rt_z_rank=rank,
        )
        labels[f"rank-{rank}"] = wrap_reordered_apply(permutation, mixed_apply)

    return benchmark_preconditioner_applies(
        seq,
        operators,
        labels,
        dirichlet=dirichlet,
        rhs_batch=rhs_batch,
    )


def print_k1_benchmark_reports(reports: list[K1BenchmarkReport]):
    print("-" * 112)
    print(
        f"{'label':<24} {'conv':>8} {'avg iters':>10} {'std':>8} {'max':>6} {'avg ms':>10} {'avg relres':>14}"
    )
    for report in reports:
        print(
            f"{report.label:<24} {report.n_converged:>4d}/{report.n_rhs:<3d} {report.avg_iters:>10.2f} {report.std_iters:>8.2f} {report.max_iters:>6d} "
            f"{report.avg_time_ms:>10.2f} {report.avg_relative_residual:>14.3e}"
        )


# %% Build once
SEQ, OPERATORS = ensure_built(CONFIG, rebuild=True)
print(
    f"built k=1 surgery case: ns={CONFIG.ns}, p={CONFIG.p}, map_kind={CONFIG.map_kind}, "
    f"eps={CONFIG.rotating_eps}, kappa={CONFIG.rotating_kappa}, nfp={CONFIG.rotating_nfp}"
)


# %% Row-family bookkeeping
for dirichlet in (False, True):
    print("=" * 112)
    print(f"dirichlet={dirichlet}")
    print(_component_slices(SEQ, dirichlet))
    print(_surgery_slices(SEQ, dirichlet))


# %% Dense extraction support summary
for dirichlet in (False, True):
    print("=" * 112)
    print(f"dense extraction support summary: dirichlet={dirichlet}")
    print_support_summary(summarize_extraction_support(SEQ, dirichlet))


# %% Verify A = E M_raw E^T
K1_MASS_IDENTITY = {}
for dirichlet in (False, True):
    a_ref, a_sandwich, diff = verify_extracted_mass_identity(SEQ, OPERATORS, dirichlet)
    K1_MASS_IDENTITY[dirichlet] = (a_ref, a_sandwich, diff)
    print(f"dirichlet={dirichlet}: max |A - E M_raw E^T| = {diff:.3e}")


# %% Report block norms of the extracted mass matrix
K1_BLOCK_NORMS = {}
for dirichlet in (False, True):
    a_ref, _, _ = K1_MASS_IDENTITY[dirichlet]
    slices = _surgery_slices(SEQ, dirichlet)
    labels, table, total_norm = block_norm_table(a_ref, slices)
    K1_BLOCK_NORMS[dirichlet] = (labels, table, total_norm)
    print("=" * 112)
    print(f"extracted k=1 mass block norms: dirichlet={dirichlet}")
    print_block_norm_table(labels, table, total_norm)


# %% Reorder to move surgery rows and columns to the front
K1_REORDERED = {}
for dirichlet in (False, True):
    a_ref, _, _ = K1_MASS_IDENTITY[dirichlet]
    labels, permutation, reordered_slices = reorder_indices(SEQ, dirichlet)
    a_perm = reorder_matrix(a_ref, permutation)
    K1_REORDERED[dirichlet] = {
        "labels": labels,
        "permutation": permutation,
        "slices": reordered_slices,
        "matrix": a_perm,
    }
    print("=" * 112)
    print(f"reordered k=1 mass structure: dirichlet={dirichlet}")
    print(labels)
    print(reordered_slices)


# %% Report block norms of the reordered extracted mass matrix
K1_REORDERED_BLOCK_NORMS = {}
for dirichlet in (False, True):
    data = K1_REORDERED[dirichlet]
    labels = list(data["labels"])
    slices = data["slices"]
    labels_out, table, total_norm = block_norm_table(
        data["matrix"],
        {label: slices[label] for label in labels},
    )
    K1_REORDERED_BLOCK_NORMS[dirichlet] = (labels_out, table, total_norm)
    print("=" * 112)
    print(f"reordered extracted k=1 mass block norms: dirichlet={dirichlet}")
    print_block_norm_table(labels_out, table, total_norm)


# %% Dense Schur prototype on reordered surgery-first matrix
K1_SCHUR_REPORTS = {}
for dirichlet in (False, True):
    data = K1_REORDERED[dirichlet]
    a_perm = data["matrix"]
    surgery_size = data["slices"]["zeta_surgery"].stop
    rhs = jax.random.normal(
        jax.random.PRNGKey(100 + int(dirichlet)),
        (a_perm.shape[0],),
        dtype=jnp.float64,
    )
    x_block, schur, schur_inv = dense_schur_inverse_apply(a_perm, surgery_size, rhs)
    x_dense = jnp.linalg.solve(a_perm, rhs)
    err = float(jnp.linalg.norm(x_block - x_dense))
    residual = float(jnp.linalg.norm(a_perm @ x_block - rhs))
    K1_SCHUR_REPORTS[dirichlet] = {
        "surgery_size": surgery_size,
        "matrix_size": a_perm.shape[0],
        "schur": schur,
        "schur_inv": schur_inv,
        "dense_error_l2": err,
        "residual_l2": residual,
    }
    print("=" * 112)
    print(f"dense Schur prototype: dirichlet={dirichlet}")
    print(f"surgery size  : {surgery_size}")
    print(f"matrix size   : {a_perm.shape[0]}")
    print(f"dense error   : {err:.3e}")
    print(f"residual l2   : {residual:.3e}")


# %% Inspect the 3x3 bulk block structure inside A_bb
K1_BULK_BLOCK_NORMS = {}
for dirichlet in (False, True):
    data = K1_REORDERED[dirichlet]
    a_perm = data["matrix"]
    surgery_size = data["slices"]["zeta_surgery"].stop
    _, _, _, abb = schur_blocks(a_perm, surgery_size)
    bulk_slices = bulk_component_slices(data["slices"])
    labels, table, total_norm = block_norm_table(abb, bulk_slices)
    K1_BULK_BLOCK_NORMS[dirichlet] = {
        "labels": labels,
        "table": table,
        "total_norm": total_norm,
        "slices": bulk_slices,
        "matrix": abb,
    }
    print("=" * 112)
    print(f"bulk 3x3 block norms inside A_bb: dirichlet={dirichlet}")
    print_block_norm_table(labels, table, total_norm)


# %% Cache A_rr and compare it against the rt-z surrogate
ARR_BLOCKS = {}
for dirichlet in (False, True):
    data = K1_REORDERED[dirichlet]
    arr = arr_block_from_reordered(data["matrix"], data["slices"])
    full_shape = arr_shape(SEQ, dirichlet)
    expected_size = full_shape[0] * full_shape[1] * full_shape[2]
    if arr.shape != (expected_size, expected_size):
        raise ValueError(
            f"A_rr shape mismatch for dirichlet={dirichlet}: got {arr.shape}, expected {(expected_size, expected_size)}"
        )
    diagnostics = arr_rt_z_diagnostics(arr, full_shape)
    ARR_BLOCKS[dirichlet] = {
        "matrix": arr,
        "shape": full_shape,
        "diagnostics": diagnostics,
    }
    print("=" * 112)
    print(f"A_rr rt-z diagnostics: dirichlet={dirichlet}, shape={full_shape}, size={arr.shape[0]}")
    print(f"relative operator error : {diagnostics.relative_operator_error:.3e}")
    print(f"relative solve error    : {diagnostics.relative_solve_error:.3e}")
    print(f"relative residual       : {diagnostics.relative_residual:.3e}")
    print(
        f"eigenvalue range        : [{diagnostics.min_eigenvalue:.3e}, {diagnostics.max_eigenvalue:.3e}]"
    )


# %% Try the same rt-z strategy directly on A_thetatheta and A_zetazeta
K1_DIAGONAL_RT_Z = {}
for dirichlet in (False, True):
    bulk = K1_BULK_BLOCK_NORMS[dirichlet]["matrix"]
    bulk_slices = K1_BULK_BLOCK_NORMS[dirichlet]["slices"]

    att = diagonal_block(bulk, bulk_slices["theta_bulk"])
    att_shape = theta_bulk_shape(SEQ, dirichlet)
    if att.shape != (att_shape[0] * att_shape[1] * att_shape[2],) * 2:
        raise ValueError(
            f"A_thetatheta shape mismatch for dirichlet={dirichlet}: got {att.shape}, expected {(att_shape[0] * att_shape[1] * att_shape[2],) * 2}"
        )
    att_diag = arr_rt_z_diagnostics(att, att_shape)

    azz = diagonal_block(bulk, bulk_slices["zeta_bulk"])
    azz_shape = zeta_bulk_shape(SEQ, dirichlet)
    if azz.shape != (azz_shape[0] * azz_shape[1] * azz_shape[2],) * 2:
        raise ValueError(
            f"A_zetazeta shape mismatch for dirichlet={dirichlet}: got {azz.shape}, expected {(azz_shape[0] * azz_shape[1] * azz_shape[2],) * 2}"
        )
    azz_diag = arr_rt_z_diagnostics(azz, azz_shape)

    K1_DIAGONAL_RT_Z[dirichlet] = {
        "A_thetatheta": {"shape": att_shape, "diagnostics": att_diag},
        "A_zetazeta": {"shape": azz_shape, "diagnostics": azz_diag},
    }

    print("=" * 112)
    print(f"A_thetatheta rt-z diagnostics: dirichlet={dirichlet}, shape={att_shape}, size={att.shape[0]}")
    print(f"relative operator error : {att_diag.relative_operator_error:.3e}")
    print(f"relative solve error    : {att_diag.relative_solve_error:.3e}")
    print(f"relative residual       : {att_diag.relative_residual:.3e}")
    print(
        f"eigenvalue range        : [{att_diag.min_eigenvalue:.3e}, {att_diag.max_eigenvalue:.3e}]"
    )

    print("-" * 112)
    print(f"A_zetazeta rt-z diagnostics: dirichlet={dirichlet}, shape={azz_shape}, size={azz.shape[0]}")
    print(f"relative operator error : {azz_diag.relative_operator_error:.3e}")
    print(f"relative solve error    : {azz_diag.relative_solve_error:.3e}")
    print(f"relative residual       : {azz_diag.relative_residual:.3e}")
    print(
        f"eigenvalue range        : [{azz_diag.min_eigenvalue:.3e}, {azz_diag.max_eigenvalue:.3e}]"
    )


# %% Check whether the next issue is r-theta coupling rather than Arr itself
R_THETA_BULK = {}
for dirichlet in (False, True):
    bulk = K1_BULK_BLOCK_NORMS[dirichlet]["matrix"]
    bulk_slices = K1_BULK_BLOCK_NORMS[dirichlet]["slices"]
    block = coupled_r_theta_block(bulk, bulk_slices)
    block_slices = coupled_r_theta_block_slices(bulk_slices)

    arr = ARR_BLOCKS[dirichlet]["matrix"]
    full_shape = ARR_BLOCKS[dirichlet]["shape"]
    arr_exact_inv = jnp.linalg.inv(arr)
    z_basis, _, _, rt_block_inverses = build_arr_rt_z_inverse(arr, full_shape)
    arr_rt_z_apply = lambda rhs, rb=rt_block_inverses, zb=z_basis, fs=full_shape: apply_arr_rt_z_inverse(
        rb, zb, rhs, fs
    )

    diagnostics = coupled_block_diagnostics(
        block,
        block_slices,
        arr_exact_inv,
        arr_rt_z_apply,
    )
    R_THETA_BULK[dirichlet] = {
        "matrix": block,
        "slices": block_slices,
        "diagnostics": diagnostics,
    }
    print("=" * 112)
    print(f"(r, theta_bulk) coupled-block diagnostics: dirichlet={dirichlet}, size={block.shape[0]}")
    print(f"offdiag relative Frobenius     : {diagnostics.offdiag_relative_frobenius:.3e}")
    print(f"exact blockdiag operator error: {diagnostics.exact_blockdiag_operator_error:.3e}")
    print(f"mixed blockdiag operator error: {diagnostics.mixed_blockdiag_operator_error:.3e}")
    print(f"exact blockdiag solve error   : {diagnostics.exact_blockdiag_solve_error:.3e}")
    print(f"mixed blockdiag solve error   : {diagnostics.mixed_blockdiag_solve_error:.3e}")
    print(f"exact blockdiag residual      : {diagnostics.exact_blockdiag_residual:.3e}")
    print(f"mixed blockdiag residual      : {diagnostics.mixed_blockdiag_residual:.3e}")


# %% Promote the (r, theta_bulk) block to a Schur model
R_THETA_SCHUR = {}
for dirichlet in (False, True):
    block = R_THETA_BULK[dirichlet]["matrix"]
    block_slices = R_THETA_BULK[dirichlet]["slices"]

    arr = ARR_BLOCKS[dirichlet]["matrix"]
    full_shape = ARR_BLOCKS[dirichlet]["shape"]
    arr_exact_inv = jnp.linalg.inv(arr)
    z_basis, _, _, rt_block_inverses = build_arr_rt_z_inverse(arr, full_shape)
    arr_rt_z_apply = lambda rhs, rb=rt_block_inverses, zb=z_basis, fs=full_shape: apply_arr_rt_z_inverse(
        rb, zb, rhs, fs
    )

    diagnostics = coupled_schur_diagnostics(
        block,
        block_slices,
        arr_exact_inv,
        arr_rt_z_apply,
    )
    R_THETA_SCHUR[dirichlet] = {
        "matrix": block,
        "slices": block_slices,
        "diagnostics": diagnostics,
    }
    print("=" * 112)
    print(f"(r, theta_bulk) Schur diagnostics: dirichlet={dirichlet}, size={block.shape[0]}")
    print(f"exact Schur operator error: {diagnostics.exact_schur_operator_error:.3e}")
    print(f"mixed Schur operator error: {diagnostics.mixed_schur_operator_error:.3e}")
    print(f"exact Schur solve error   : {diagnostics.exact_schur_solve_error:.3e}")
    print(f"mixed Schur solve error   : {diagnostics.mixed_schur_solve_error:.3e}")
    print(f"exact Schur residual      : {diagnostics.exact_schur_residual:.3e}")
    print(f"mixed Schur residual      : {diagnostics.mixed_schur_residual:.3e}")


# %% Measure how much the theta Schur correction changes A_thetatheta
THETA_SCHUR_CORRECTION = {}
for dirichlet in (False, True):
    block = R_THETA_BULK[dirichlet]["matrix"]
    block_slices = R_THETA_BULK[dirichlet]["slices"]

    arr = ARR_BLOCKS[dirichlet]["matrix"]
    full_shape = ARR_BLOCKS[dirichlet]["shape"]
    arr_exact_inv = jnp.linalg.inv(arr)
    z_basis, _, _, rt_block_inverses = build_arr_rt_z_inverse(arr, full_shape)
    arr_rt_z_apply = lambda rhs, rb=rt_block_inverses, zb=z_basis, fs=full_shape: apply_arr_rt_z_inverse(
        rb, zb, rhs, fs
    )

    diagnostics = theta_schur_correction_diagnostics(
        block,
        block_slices,
        arr_exact_inv,
        arr_rt_z_apply,
    )
    THETA_SCHUR_CORRECTION[dirichlet] = diagnostics
    print("=" * 112)
    print(f"theta Schur correction diagnostics: dirichlet={dirichlet}")
    print(f"||Atr Arr^-1 Art|| / ||Athetatheta||           : {diagnostics.exact_correction_relative_to_atheta:.3e}")
    print(f"||Atr Atilde_rr^-1 Art|| / ||Athetatheta||     : {diagnostics.mixed_correction_relative_to_atheta:.3e}")
    print(f"relative correction error (mixed vs exact)     : {diagnostics.mixed_vs_exact_correction_error:.3e}")
    print(f"||Schur_theta|| / ||Athetatheta||              : {diagnostics.exact_schur_relative_to_atheta:.3e}")
    print(f"relative Schur error from A_rr approximation   : {diagnostics.mixed_schur_vs_exact_error:.3e}")


# %% Lift the coupled rt Schur model back to the full bulk block A_bb
ABB_RT_SCHUR = {}
for dirichlet in (False, True):
    bulk = K1_BULK_BLOCK_NORMS[dirichlet]["matrix"]
    bulk_slices = K1_BULK_BLOCK_NORMS[dirichlet]["slices"]
    block_slices = rt_zeta_block_slices(bulk_slices)

    rt_block = R_THETA_BULK[dirichlet]["matrix"]
    rt_block_slices = R_THETA_BULK[dirichlet]["slices"]

    arr = ARR_BLOCKS[dirichlet]["matrix"]
    full_shape = ARR_BLOCKS[dirichlet]["shape"]
    arr_exact_inv = jnp.linalg.inv(arr)
    z_basis, _, _, rt_block_inverses = build_arr_rt_z_inverse(arr, full_shape)
    arr_rt_z_apply = lambda rhs, rb=rt_block_inverses, zb=z_basis, fs=full_shape: apply_arr_rt_z_inverse(
        rb, zb, rhs, fs
    )
    exact_rt_apply, _ = schur_r_theta_apply(rt_block, rt_block_slices, lambda rhs: arr_exact_inv @ rhs)
    mixed_rt_apply, _ = schur_r_theta_apply(rt_block, rt_block_slices, arr_rt_z_apply)

    diagnostics = bulk_rt_zeta_diagnostics(
        bulk,
        block_slices,
        exact_rt_apply,
        mixed_rt_apply,
    )
    ABB_RT_SCHUR[dirichlet] = {
        "slices": block_slices,
        "diagnostics": diagnostics,
    }
    print("=" * 112)
    print(f"A_bb with coupled rt Schur diagnostics: dirichlet={dirichlet}, size={bulk.shape[0]}")
    print(f"offdiag relative Frobenius    : {diagnostics.offdiag_relative_frobenius:.3e}")
    print(f"exact blockdiag operator error: {diagnostics.exact_blockdiag_operator_error:.3e}")
    print(f"mixed blockdiag operator error: {diagnostics.mixed_blockdiag_operator_error:.3e}")
    print(f"exact blockdiag solve error   : {diagnostics.exact_blockdiag_solve_error:.3e}")
    print(f"mixed blockdiag solve error   : {diagnostics.mixed_blockdiag_solve_error:.3e}")
    print(f"exact blockdiag residual      : {diagnostics.exact_blockdiag_residual:.3e}")
    print(f"mixed blockdiag residual      : {diagnostics.mixed_blockdiag_residual:.3e}")


# %% Use the new A_bb model inside the outer surgery Schur split
K1_FULL_APPROX = {}
for dirichlet in (False, True):
    data = K1_REORDERED[dirichlet]
    matrix = data["matrix"]
    surgery_size = data["slices"]["zeta_surgery"].stop
    exact_apply, mixed_apply = build_k1_full_model_applies(
        dirichlet,
        couple_zeta=False,
        zeta_mode="rt-z",
    )
    diagnostics = full_apply_diagnostics(matrix, exact_apply, mixed_apply)
    K1_FULL_APPROX[dirichlet] = {
        "diagnostics": diagnostics,
    }
    print("=" * 112)
    print(f"full k=1 surgery Schur diagnostics: dirichlet={dirichlet}, size={matrix.shape[0]}")
    print(f"exact Schur operator error: {diagnostics.exact_schur_operator_error:.3e}")
    print(f"mixed Schur operator error: {diagnostics.mixed_schur_operator_error:.3e}")
    print(f"exact Schur solve error   : {diagnostics.exact_schur_solve_error:.3e}")
    print(f"mixed Schur solve error   : {diagnostics.mixed_schur_solve_error:.3e}")
    print(f"exact Schur residual      : {diagnostics.exact_schur_residual:.3e}")
    print(f"mixed Schur residual      : {diagnostics.mixed_schur_residual:.3e}")


# %% Fully lowrank bulk prototype: rt-z on Arr, Att, and Azz
K1_FULL_LOWRANK = {}
for dirichlet in (False, True):
    data = K1_REORDERED[dirichlet]
    matrix = data["matrix"]
    exact_apply, mixed_apply = build_k1_full_model_applies(
        dirichlet,
        couple_zeta=False,
        theta_mode="rt-z-lowrank",
        zeta_mode="rt-z",
    )
    diagnostics = full_apply_diagnostics(matrix, exact_apply, mixed_apply)
    K1_FULL_LOWRANK[dirichlet] = {"diagnostics": diagnostics}
    print("=" * 112)
    print(f"full k=1 lowrank surgery diagnostics: dirichlet={dirichlet}, size={matrix.shape[0]}")
    print(f"exact Schur operator error: {diagnostics.exact_schur_operator_error:.3e}")
    print(f"mixed Schur operator error: {diagnostics.mixed_schur_operator_error:.3e}")
    print(f"exact Schur solve error   : {diagnostics.exact_schur_solve_error:.3e}")
    print(f"mixed Schur solve error   : {diagnostics.mixed_schur_solve_error:.3e}")
    print(f"exact Schur residual      : {diagnostics.exact_schur_residual:.3e}")
    print(f"mixed Schur residual      : {diagnostics.mixed_schur_residual:.3e}")


# %% Ceiling check: include the rt-zeta bulk coupling as one more Schur layer
K1_FULL_COUPLED = {}
for dirichlet in (False, True):
    data = K1_REORDERED[dirichlet]
    matrix = data["matrix"]
    exact_apply, mixed_apply = build_k1_full_model_applies(dirichlet, couple_zeta=True)
    diagnostics = full_apply_diagnostics(
        matrix,
        exact_apply,
        mixed_apply,
    )
    K1_FULL_COUPLED[dirichlet] = {"diagnostics": diagnostics}
    print("=" * 112)
    print(f"full k=1 surgery Schur diagnostics with rt-zeta coupling: dirichlet={dirichlet}, size={matrix.shape[0]}")
    print(f"exact Schur operator error: {diagnostics.exact_schur_operator_error:.3e}")
    print(f"mixed Schur operator error: {diagnostics.mixed_schur_operator_error:.3e}")
    print(f"exact Schur solve error   : {diagnostics.exact_schur_solve_error:.3e}")
    print(f"mixed Schur solve error   : {diagnostics.mixed_schur_solve_error:.3e}")
    print(f"exact Schur residual      : {diagnostics.exact_schur_residual:.3e}")
    print(f"mixed Schur residual      : {diagnostics.mixed_schur_residual:.3e}")


# %% Benchmark k=1 mass preconditioners against jacobi and kronecker
OPERATORS = assemble_kron_mass_preconditioner(SEQ, operators=OPERATORS)
K1_BENCHMARKS = {}
for dirichlet in (False, True):
    reports = benchmark_k1_preconditioners(
        SEQ,
        OPERATORS,
        dirichlet=dirichlet,
        n_rhs=8,
        seed=0,
    )
    K1_BENCHMARKS[dirichlet] = reports
    print("=" * 112)
    print(f"k=1 mass preconditioner benchmark: dirichlet={dirichlet}")
    print_k1_benchmark_reports(reports)


# %% Ablation: vary the rt-z rank in the lowrank k=1 surgery preconditioner
K1_LOWRANK_RANK_ABLATION = {}
for dirichlet in (False, True):
    max_rank = k1_rt_z_max_rank(SEQ, dirichlet)
    ranks = tuple(range(1, max_rank + 1))
    reports = benchmark_k1_lowrank_rank_ablation(
        SEQ,
        OPERATORS,
        dirichlet=dirichlet,
        ranks=ranks,
        n_rhs=8,
        seed=0,
        couple_zeta=False,
    )
    K1_LOWRANK_RANK_ABLATION[dirichlet] = {
        "max_rank": max_rank,
        "reports": reports,
    }
    print("=" * 112)
    print(f"k=1 lowrank rt-z rank ablation: dirichlet={dirichlet}, max_rank={max_rank}")
    print_k1_benchmark_reports(reports)


# %% Metric-factor CP-ALS on the quadrature-grid diagonal coefficient tensors
METRIC_CP_RANKS = (1, 2, 3, 4)
K1_METRIC_CP = {}
for label, tensor in k1_diagonal_metric_tensors(SEQ).items():
    reports = []
    fits = {}
    print("=" * 112)
    print(f"k=1 metric CP-ALS diagnostics: field={label}, shape={tensor.shape}")
    for rank in METRIC_CP_RANKS:
        weights, factors, diagnostics = cp_als_3tensor(
            tensor,
            rank,
            maxiter=200,
            tol=1e-10,
            ridge=1e-12,
        )
        diagnostics = MetricFactorFitDiagnostics(
            label=label,
            rank=diagnostics.rank,
            relative_error=diagnostics.relative_error,
            max_abs_error=diagnostics.max_abs_error,
            n_iters=diagnostics.n_iters,
            final_delta=diagnostics.final_delta,
        )
        fits[rank] = {
            "weights": weights,
            "factors": factors,
            "diagnostics": diagnostics,
        }
        reports.append(diagnostics)
    K1_METRIC_CP[label] = {
        "tensor": tensor,
        "fits": fits,
    }
    print_metric_factor_fit_reports(reports)


# %% A_rr metric-Kronecker rank sweep
ARR_METRIC_KRON_SWEEP_RANKS = (1, 2, 3, 4)
ARR_METRIC_KRON_SWEEP = {}
for dirichlet in (False, True):
    arr = ARR_BLOCKS[dirichlet]["matrix"]
    full_shape = ARR_BLOCKS[dirichlet]["shape"]
    reports = {}
    print("=" * 112)
    print(f"A_rr metric-Kronecker rank sweep: dirichlet={dirichlet}, size={arr.shape[0]}")
    for rank in ARR_METRIC_KRON_SWEEP_RANKS:
        diagnostics = arr_metric_cp_kron_diagnostics(
            SEQ,
            arr,
            K1_METRIC_CP["alpha_rr"]["tensor"],
            full_shape,
            rank,
        )
        reports[rank] = {"diagnostics": diagnostics}
        print("-" * 112)
        print(f"rank={rank}")
        print(f"relative operator error        : {diagnostics.relative_operator_error:.3e}")
        print(f"relative inverse operator error: {diagnostics.relative_inverse_operator_error:.3e}")
        print(f"relative solve error           : {diagnostics.relative_solve_error:.3e}")
        print(f"relative residual              : {diagnostics.relative_residual:.3e}")
    ARR_METRIC_KRON_SWEEP[dirichlet] = reports


# %% Rank-1 exact metric-Kronecker surrogates for A_thetatheta and A_zetazeta
K1_DIAGONAL_METRIC_KRON_R1 = {}
for dirichlet in (False, True):
    bulk = K1_BULK_BLOCK_NORMS[dirichlet]["matrix"]
    bulk_slices = K1_BULK_BLOCK_NORMS[dirichlet]["slices"]

    att = diagonal_block(bulk, bulk_slices["theta_bulk"])
    att_shape = theta_bulk_shape(SEQ, dirichlet)
    att_kron, _, att_apply_inv = build_theta_metric_cp_kron_approx(
        SEQ,
        K1_METRIC_CP["alpha_thetatheta"]["tensor"],
        att_shape,
        1,
    )
    att_diag = diagonal_metric_cp_kron_diagnostics(
        att,
        att_kron,
        att_apply_inv,
        label="A_thetatheta-rank1-metric-kron",
    )

    azz = diagonal_block(bulk, bulk_slices["zeta_bulk"])
    azz_shape = zeta_bulk_shape(SEQ, dirichlet)
    azz_kron, _, azz_apply_inv = build_zeta_metric_cp_kron_approx(
        SEQ,
        K1_METRIC_CP["alpha_zetazeta"]["tensor"],
        azz_shape,
        1,
    )
    azz_diag = diagonal_metric_cp_kron_diagnostics(
        azz,
        azz_kron,
        azz_apply_inv,
        label="A_zetazeta-rank1-metric-kron",
    )

    K1_DIAGONAL_METRIC_KRON_R1[dirichlet] = {
        "A_thetatheta": {"shape": att_shape, "diagnostics": att_diag},
        "A_zetazeta": {"shape": azz_shape, "diagnostics": azz_diag},
    }

    print("=" * 112)
    print(f"A_thetatheta rank-1 exact metric-Kronecker diagnostics: dirichlet={dirichlet}, shape={att_shape}, size={att.shape[0]}")
    print(f"relative operator error        : {att_diag.relative_operator_error:.3e}")
    print(f"relative inverse operator error: {att_diag.relative_inverse_operator_error:.3e}")
    print(f"relative solve error           : {att_diag.relative_solve_error:.3e}")
    print(f"relative residual              : {att_diag.relative_residual:.3e}")

    print("-" * 112)
    print(f"A_zetazeta rank-1 exact metric-Kronecker diagnostics: dirichlet={dirichlet}, shape={azz_shape}, size={azz.shape[0]}")
    print(f"relative operator error        : {azz_diag.relative_operator_error:.3e}")
    print(f"relative inverse operator error: {azz_diag.relative_inverse_operator_error:.3e}")
    print(f"relative solve error           : {azz_diag.relative_solve_error:.3e}")
    print(f"relative residual              : {azz_diag.relative_residual:.3e}")


# %% Optional: visualize the reordered bulk block A_bb
DIRICHLET_TO_PLOT = False
ABB_TO_PLOT = K1_BULK_BLOCK_NORMS[DIRICHLET_TO_PLOT]["matrix"]
BULK_SLICES_TO_PLOT = K1_BULK_BLOCK_NORMS[DIRICHLET_TO_PLOT]["slices"]

fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(jnp.log10(jnp.abs(ABB_TO_PLOT) + 1e-16), cmap="viridis", origin="lower")
for boundary in (
    BULK_SLICES_TO_PLOT["r"].stop,
    BULK_SLICES_TO_PLOT["theta_bulk"].stop,
):
    ax.axhline(boundary - 0.5, color="white", linewidth=1.0)
    ax.axvline(boundary - 0.5, color="white", linewidth=1.0)
ax.set_title(f"log10 |A_bb|, dirichlet={DIRICHLET_TO_PLOT}")
fig.colorbar(im, ax=ax, shrink=0.8)
plt.show()


# %% Optional: visualize the extracted matrix with surgery separators
DIRICHLET_TO_PLOT = False
A_TO_PLOT, _, _ = K1_MASS_IDENTITY[DIRICHLET_TO_PLOT]
SLICES_TO_PLOT = _surgery_slices(SEQ, DIRICHLET_TO_PLOT)

fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(jnp.log10(jnp.abs(A_TO_PLOT) + 1e-16), cmap="viridis", origin="lower")
for boundary in (
    SLICES_TO_PLOT["r"].stop,
    SLICES_TO_PLOT["theta_surgery"].stop,
    SLICES_TO_PLOT["theta_bulk"].stop,
    SLICES_TO_PLOT["zeta_surgery"].stop,
):
    ax.axhline(boundary - 0.5, color="white", linewidth=1.0)
    ax.axvline(boundary - 0.5, color="white", linewidth=1.0)
ax.set_title(f"log10 |E1 M1 E1^T|, dirichlet={DIRICHLET_TO_PLOT}")
fig.colorbar(im, ax=ax, shrink=0.8)
plt.show()


# %% Optional: visualize the reordered extracted matrix with surgery-first ordering
DIRICHLET_TO_PLOT = False
REORDERED_TO_PLOT = K1_REORDERED[DIRICHLET_TO_PLOT]
A_PERM_TO_PLOT = REORDERED_TO_PLOT["matrix"]
SLICES_TO_PLOT = REORDERED_TO_PLOT["slices"]

fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(jnp.log10(jnp.abs(A_PERM_TO_PLOT) + 1e-16), cmap="viridis", origin="lower")
for boundary in (
    SLICES_TO_PLOT["theta_surgery"].stop,
    SLICES_TO_PLOT["zeta_surgery"].stop,
    SLICES_TO_PLOT["r"].stop,
    SLICES_TO_PLOT["theta_bulk"].stop,
):
    ax.axhline(boundary - 0.5, color="white", linewidth=1.0)
    ax.axvline(boundary - 0.5, color="white", linewidth=1.0)
ax.set_title(f"log10 |P E1 M1 E1^T P^T|, dirichlet={DIRICHLET_TO_PLOT}")
fig.colorbar(im, ax=ax, shrink=0.8)
plt.show()


# %% Optional: visualize metric coefficient tensors against rank-1 and rank-3 CP fits
METRIC_FIELDS_TO_PLOT = ("alpha_rr", "alpha_thetatheta", "alpha_zetazeta")
METRIC_COMPARE_RANKS = (1, 3)
NZ_SLICES = K1_METRIC_CP["alpha_rr"]["tensor"].shape[2]
SLICE_Z_INDICES = (0, NZ_SLICES // 3, (2 * NZ_SLICES) // 3)
RADIAL_QUAD_GRID = SEQ.quad.x_x[None, :, None]

for label in METRIC_FIELDS_TO_PLOT:
    tensor = K1_METRIC_CP[label]["tensor"]
    display_tensor = tensor
    display_label = label
    if label == "alpha_thetatheta":
        display_tensor = RADIAL_QUAD_GRID * tensor
        display_label = "r * alpha_thetatheta"

    reconstructions = {}
    for rank in METRIC_COMPARE_RANKS:
        fit = K1_METRIC_CP[label]["fits"][rank]
        reconstruction = reconstruct_cp_3tensor(fit["weights"], fit["factors"])
        if label == "alpha_thetatheta":
            reconstruction = RADIAL_QUAD_GRID * reconstruction
        reconstructions[rank] = reconstruction

    fig, axes = plt.subplots(
        nrows=len(SLICE_Z_INDICES),
        ncols=4,
        figsize=(16, 12),
        constrained_layout=True,
    )

    for row, slice_z_index in enumerate(SLICE_Z_INDICES):
        exact_slice = display_tensor[:, :, slice_z_index]
        rank1_slice = reconstructions[1][:, :, slice_z_index]
        rank3_slice = reconstructions[3][:, :, slice_z_index]
        error_slice = jnp.abs(rank3_slice - exact_slice)

        value_min = float(jnp.min(jnp.stack([exact_slice, rank1_slice, rank3_slice], axis=0)))
        value_max = float(jnp.max(jnp.stack([exact_slice, rank1_slice, rank3_slice], axis=0)))

        images = (
            (exact_slice, "exact"),
            (rank1_slice, "rank-1"),
            (rank3_slice, "rank-3"),
            (error_slice, "|rank-3 - exact|"),
        )

        zeta_value = float(SEQ.quad.x_z[slice_z_index])
        for col, (image, title) in enumerate(images):
            ax = axes[row, col]
            if col < 3:
                im = ax.imshow(image, origin="lower", cmap="viridis", vmin=value_min, vmax=value_max, aspect="auto")
            else:
                im = ax.imshow(image, origin="lower", cmap="magma", aspect="auto")
            ax.set_title(f"{title}, z idx={slice_z_index}, z={zeta_value:.3f}")
            ax.set_xlabel("r quadrature index")
            ax.set_ylabel("theta quadrature index")
            fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(f"{display_label}: slice comparison against rank-1 and rank-3 CP fits")
    plt.show()
# %%
