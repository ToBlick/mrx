# %% [markdown]
# # k=1 Tensor Mass Benchmark
#
# This script keeps only the production-facing k=1 workflow:
#
# 1. assemble the k=1 mass operator,
# 2. assemble the production tensor mass preconditioner,
# 3. compare it against jacobi,
# 4. and keep the matrix / metric plots at the end.
#
# The explicit reordered matrices below are reconstructed only for
# visualization. The production tensor preconditioner now assembles directly
# from the extracted ordering.

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
    assemble_mass_operators,
    assemble_tensor_mass_preconditioner,
    dense_mass_matrix,
)
from mrx.solvers import solve_singular_cg

jax.config.update("jax_enable_x64", True)


# %% Configuration
@dataclass(frozen=True)
class ExperimentConfig:
    ns: tuple[int, int, int] = (10, 10, 10)
    p: int = 2
    tol: float = 1e-9
    maxiter: int = 1000
    betti: tuple[int, int, int, int] = (1, 1, 0, 0)
    map_kind: str = "rotating_ellipse"
    torus_epsilon: float = 1.0 / 3.0
    torus_r0: float = 1.0
    rotating_eps: float = 0.2
    rotating_kappa: float = 1.1
    rotating_r0: float = 1.0
    rotating_nfp: int = 2


@dataclass
class K1BenchmarkReport:
    label: str
    dirichlet: bool
    n_rhs: int
    avg_iters: float
    std_iters: float
    max_iters: int
    avg_time_ms: float
    std_time_ms: float
    max_time_ms: float
    avg_relative_residual: float
    std_relative_residual: float
    max_relative_residual: float


@dataclass
class MetricFactorFitDiagnostics:
    label: str
    rank: int
    relative_error: float
    max_abs_error: float
    n_iters: int
    final_delta: float


CONFIG = ExperimentConfig()
SEQ = None
OPERATORS = None
BUILT_CONFIG = None
TENSOR_CP_KWARGS = {"tol": 1e-9, "maxiter": 100}


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


def dense_extracted_mass_matrix(seq, operators, dirichlet: bool) -> jnp.ndarray:
    return jnp.asarray(dense_mass_matrix(seq, operators, 1, dirichlet=dirichlet))


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
    return permutation, reordered_slices


def reorder_matrix(matrix: jnp.ndarray, permutation: jnp.ndarray) -> jnp.ndarray:
    return matrix[permutation][:, permutation]


def schur_blocks(matrix: jnp.ndarray, surgery_size: int):
    ass = matrix[:surgery_size, :surgery_size]
    asb = matrix[:surgery_size, surgery_size:]
    abs_ = matrix[surgery_size:, :surgery_size]
    abb = matrix[surgery_size:, surgery_size:]
    return ass, asb, abs_, abb


def bulk_component_slices(reordered_slices: dict[str, slice]):
    surgery_size = reordered_slices["zeta_surgery"].stop
    theta_bulk = reordered_slices["theta_bulk"]
    zeta_bulk = reordered_slices["zeta_bulk"]
    return {
        "r": slice(reordered_slices["r"].start - surgery_size, reordered_slices["r"].stop - surgery_size),
        "theta_bulk": slice(theta_bulk.start - surgery_size, theta_bulk.stop - surgery_size),
        "zeta_bulk": slice(zeta_bulk.start - surgery_size, zeta_bulk.stop - surgery_size),
    }


def build_k1_plot_data(seq, operators):
    plot_data = {}
    for dirichlet in (False, True):
        extracted = dense_extracted_mass_matrix(seq, operators, dirichlet)
        surgery_slices = _surgery_slices(seq, dirichlet)
        permutation, reordered_slices = reorder_indices(seq, dirichlet)
        reordered = reorder_matrix(extracted, permutation)
        surgery_size = reordered_slices["zeta_surgery"].stop
        _, _, _, bulk = schur_blocks(reordered, surgery_size)
        plot_data[dirichlet] = {
            "extracted": extracted,
            "surgery_slices": surgery_slices,
            "reordered": reordered,
            "reordered_slices": reordered_slices,
            "bulk": bulk,
            "bulk_slices": bulk_component_slices(reordered_slices),
        }
    return plot_data


def quadrature_tensor_shape(seq) -> tuple[int, int, int]:
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
    factor_theta, factor_r, factor_z = factors
    return jnp.einsum("r,ir,jr,kr->ijk", weights, factor_theta, factor_r, factor_z)


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

    factor_theta = jnp.linalg.svd(unfolded_0, full_matrices=False)[0][:, :rank]
    factor_r = jnp.linalg.svd(unfolded_1, full_matrices=False)[0][:, :rank]
    factor_z = jnp.linalg.svd(unfolded_2, full_matrices=False)[0][:, :rank]
    factor_theta, _ = _normalize_cp_columns(factor_theta)
    factor_r, _ = _normalize_cp_columns(factor_r)
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

        reconstruction = reconstruct_cp_3tensor(weights, (factor_theta, factor_r, factor_z))
        relative_error = float(jnp.linalg.norm(reconstruction - tensor) / tensor_norm_safe)
        final_delta = abs(relative_error - previous_error) if previous_error < jnp.inf else jnp.inf
        previous_error = relative_error
        n_iters = iteration
        if final_delta < tol:
            break

    reconstruction = reconstruct_cp_3tensor(weights, (factor_theta, factor_r, factor_z))
    diagnostics = MetricFactorFitDiagnostics(
        label="",
        rank=rank,
        relative_error=float(jnp.linalg.norm(reconstruction - tensor) / tensor_norm_safe),
        max_abs_error=float(jnp.max(jnp.abs(reconstruction - tensor))),
        n_iters=n_iters,
        final_delta=float(final_delta),
    )
    return weights, (factor_theta, factor_r, factor_z), diagnostics


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
        times_ms = []
        residuals = []
        for rhs in rhs_batch:
            t0 = time.perf_counter()
            x, info, residual = solve(rhs)
            jax.block_until_ready(x)
            times_ms.append((time.perf_counter() - t0) * 1e3)
            iterations.append(int(jnp.abs(info)))
            residuals.append(float(residual))

        times_ms_array = jnp.asarray(times_ms)
        residuals_array = jnp.asarray(residuals)

        reports.append(
            K1BenchmarkReport(
                label=label,
                dirichlet=dirichlet,
                n_rhs=n_rhs,
                avg_iters=float(jnp.mean(jnp.asarray(iterations))),
                std_iters=float(jnp.std(jnp.asarray(iterations))),
                max_iters=int(jnp.max(jnp.asarray(iterations))),
                avg_time_ms=float(jnp.mean(times_ms_array)),
                std_time_ms=float(jnp.std(times_ms_array)),
                max_time_ms=float(jnp.max(times_ms_array)),
                avg_relative_residual=float(jnp.mean(residuals_array)),
                std_relative_residual=float(jnp.std(residuals_array)),
                max_relative_residual=float(jnp.max(residuals_array)),
            )
        )
    return reports


def benchmark_k1_preconditioners(
    seq,
    operators,
    *,
    dirichlet: bool,
    n_rhs: int = 8,
    seed: int = 0,
    tensor_ranks: tuple[int, ...] = (1, 3, 5),
    tensor_operator_cache: dict[int, object] | None = None,
) -> list[K1BenchmarkReport]:
    rhs_size = seq.n1_dbc if dirichlet else seq.n1
    rhs_batch = jax.random.normal(
        jax.random.PRNGKey(seed + 100 * int(dirichlet)),
        (n_rhs, rhs_size),
        dtype=jnp.float64,
    )

    labels = {
        "jacobi": lambda x: apply_mass_matrix_preconditioner(
            seq, operators, x, 1, dirichlet=dirichlet, kind="jacobi"
        ),
    }

    for tensor_rank in tensor_ranks:
        tensor_operators = tensor_operator_cache[tensor_rank] if tensor_operator_cache is not None else assemble_tensor_mass_preconditioner(
            seq,
            operators=operators,
            rank=tensor_rank,
            cp_kwargs=TENSOR_CP_KWARGS,
        )
        labels[f"tensor-r{tensor_rank}"] = (
            lambda x, tensor_ops=tensor_operators: apply_mass_matrix_preconditioner(
                seq,
                tensor_ops,
                x,
                1,
                dirichlet=dirichlet,
                kind="tensor",
            )
        )

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
        f"{'label':<16} {'avg iters':>10} {'std':>8} {'max':>6} {'avg ms':>10} {'std ms':>10} {'max ms':>10} {'avg relres':>14} {'std relres':>14} {'max relres':>14}"
    )
    for report in reports:
        print(
            f"{report.label:<16} {report.avg_iters:>10.2f} {report.std_iters:>8.2f} {report.max_iters:>6d} "
            f"{report.avg_time_ms:>10.2f} {report.std_time_ms:>10.2f} {report.max_time_ms:>10.2f} "
            f"{report.avg_relative_residual:>14.3e} {report.std_relative_residual:>14.3e} {report.max_relative_residual:>14.3e}"
        )


# %% Build once
SEQ, OPERATORS = ensure_built(CONFIG, rebuild=True)
print(
    f"built k=1 tensor case: ns={CONFIG.ns}, p={CONFIG.p}, map_kind={CONFIG.map_kind}, "
    f"eps={CONFIG.rotating_eps}, kappa={CONFIG.rotating_kappa}, nfp={CONFIG.rotating_nfp}"
)
print("production tensor preconditioner now assembles without reordering; the reordered matrices below are only for plotting")
K1_PLOT_DATA = build_k1_plot_data(SEQ, OPERATORS)


# %% Metric-factor CP-ALS on the quadrature-grid diagonal coefficient tensors
METRIC_CP_RANKS = (1, 2, 3, 4, 5)
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
            maxiter=100,
            tol=1e-9,
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


# %% Assemble production tensor preconditioners once per rank
TENSOR_BENCHMARK_RANKS = (1, 3, 5)
TENSOR_OPERATOR_CACHE = {
    rank: assemble_tensor_mass_preconditioner(
        SEQ,
        operators=OPERATORS,
        rank=rank,
        cp_kwargs=TENSOR_CP_KWARGS,
    )
    for rank in TENSOR_BENCHMARK_RANKS
}


# %% Benchmark k=1 mass preconditioners against jacobi and tensor
K1_BENCHMARKS = {}
for dirichlet in (False, True):
    reports = benchmark_k1_preconditioners(
        SEQ,
        OPERATORS,
        dirichlet=dirichlet,
        n_rhs=8,
        seed=0,
        tensor_ranks=TENSOR_BENCHMARK_RANKS,
        tensor_operator_cache=TENSOR_OPERATOR_CACHE,
    )
    K1_BENCHMARKS[dirichlet] = reports
    print("=" * 112)
    print(f"k=1 mass preconditioner benchmark: dirichlet={dirichlet}")
    print_k1_benchmark_reports(reports)


# %% Optional: visualize the reordered bulk block A_bb
DIRICHLET_TO_PLOT = False
ABB_TO_PLOT = K1_PLOT_DATA[DIRICHLET_TO_PLOT]["bulk"]
BULK_SLICES_TO_PLOT = K1_PLOT_DATA[DIRICHLET_TO_PLOT]["bulk_slices"]

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
A_TO_PLOT = K1_PLOT_DATA[DIRICHLET_TO_PLOT]["extracted"]
SLICES_TO_PLOT = K1_PLOT_DATA[DIRICHLET_TO_PLOT]["surgery_slices"]

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
A_PERM_TO_PLOT = K1_PLOT_DATA[DIRICHLET_TO_PLOT]["reordered"]
SLICES_TO_PLOT = K1_PLOT_DATA[DIRICHLET_TO_PLOT]["reordered_slices"]

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
    if label == "alpha_rr":
        display_tensor = 1/RADIAL_QUAD_GRID * tensor
        display_label = "alpha_rr / r"
    elif label == "alpha_thetatheta":
        display_tensor = RADIAL_QUAD_GRID * tensor
        display_label = "r * alpha_thetatheta"
    elif label == "alpha_zetazeta":
        display_tensor = 1/RADIAL_QUAD_GRID * tensor
        display_label = "alpha_zetazeta / r"

    raw_reconstructions = {}
    reconstructions = {}
    for rank in METRIC_COMPARE_RANKS:
        fit = K1_METRIC_CP[label]["fits"][rank]
        reconstruction = reconstruct_cp_3tensor(fit["weights"], fit["factors"])
        raw_reconstructions[rank] = reconstruction
        if label == "alpha_thetatheta":
            reconstruction = RADIAL_QUAD_GRID * reconstruction
        else:
            reconstruction = 1/RADIAL_QUAD_GRID * reconstruction
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
        error_slice = jnp.abs(raw_reconstructions[3][:, :, slice_z_index] - tensor[:, :, slice_z_index])

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