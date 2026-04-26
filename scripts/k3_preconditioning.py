# %% [markdown]
# # k=3 Tensor Mass Benchmark
# This script keeps only the production-facing k=3 workflow:
#
# 1. assemble the k=3 mass operator,
# 2. assemble the production tensor mass preconditioner,
# 3. compare it against jacobi,
# 4. and keep the matrix and metric plots at the end.
#
# For polar k=3 there is no Schur structure at all: the extracted mass matrix
# is one scalar tensor block, so the production apply is just the tensor block
# inverse built from CP-ALS fits of the scalar factor 1 / J.

# %%
from __future__ import annotations

import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from mrx.assembly import grad_1d
from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import rotating_ellipse_map, toroid_map
from mrx.operators import (
    apply_derivative_matrix,
    apply_hodge_laplacian_preconditioner,
    apply_mass_matrix,
    apply_mass_matrix_preconditioner,
    apply_stiffness,
    assemble_incidence_operators,
    assemble_mass_operators,
    assemble_projection_operators,
    assemble_tensor_mass_preconditioner,
    dense_derivative_matrix,
    dense_hodge_laplacian,
    dense_mass_matrix,
    dense_stiffness_matrix,
)
from mrx.preconditioners import (
    _r_bulk_shape_k2,
    _tensor_block_indices_k2,
    _theta_shape_k2,
    _zeta_shape_k2,
)
from mrx.solvers import solve_saddle_point_minres, solve_singular_cg

jax.config.update("jax_enable_x64", True)

# %%
# Careful!
SEQ = None
OPERATORS = None
BUILT_CONFIG = None
TENSOR_CP_KWARGS = {"tol": 1e-8, "maxiter": 500}

# %% Configuration
@dataclass(frozen=True)
class ExperimentConfig:
    ns: tuple[int, int, int] = (8, 10, 6)
    p: int = 3
    tol: float = 1e-8
    maxiter: int = 500
    betti: tuple[int, int, int, int] = (1, 1, 0, 0)
    map_kind: str = "rotating_ellipse"
    torus_epsilon: float = 1.0 / 3.0
    torus_r0: float = 1.0
    rotating_eps: float = 0.2
    rotating_kappa: float = 1.1
    rotating_r0: float = 1.0
    rotating_nfp: int = 3


@dataclass
class K3BenchmarkReport:
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


@dataclass
class K3SaddleBenchmarkReport:
    label: str
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
class K3SchurApproximationReport:
    label: str
    inverse_relative_error: float
    solve_relative_error: float
    min_precond_eig: float
    max_precond_eig: float


@dataclass
class K3SaddleSpectrumReport:
    label: str
    min_eig: float
    max_eig: float
    min_abs_eig: float
    max_abs_eig: float
    pos_count: int
    neg_count: int


@dataclass
class K2StiffnessKroneckerApproximationReport:
    label: str
    matrix_relative_error: float
    inverse_relative_error: float
    range_projector_relative_error: float
    avg_range_action_error: float
    max_range_action_error: float


@dataclass
class K2BenchmarkReport:
    label: str
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
class PreconditionerPropertyReport:
    label: str
    metric: str
    n_probes: int
    avg_symmetry_defect: float
    max_symmetry_defect: float
    min_quadratic_form: float
    max_quadratic_form: float
    negative_quadratic_forms: int


CONFIG = ExperimentConfig()
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
    operators = assemble_mass_operators(seq, seq.geometry, ks=(0, 1, 2, 3))
    operators = assemble_incidence_operators(seq, operators=operators, ks=(0, 1, 2))
    operators = assemble_projection_operators(
        seq,
        operators=operators,
        pairs=((0, 3), (3, 0)),
    )
    return seq, operators


def ensure_built(config: ExperimentConfig = CONFIG, rebuild: bool = False):
    global SEQ, OPERATORS, BUILT_CONFIG
    if rebuild or SEQ is None or OPERATORS is None or BUILT_CONFIG != config:
        SEQ, OPERATORS = build_case(config)
        BUILT_CONFIG = config
    return SEQ, OPERATORS


def dense_extracted_mass_matrix(seq, operators, dirichlet: bool) -> jnp.ndarray:
    return jnp.asarray(dense_mass_matrix(seq, operators, 3, dirichlet=dirichlet))


def build_k3_plot_data(seq, operators):
    plot_data = {}
    for dirichlet in (False, True):
        extracted = dense_extracted_mass_matrix(seq, operators, dirichlet)
        plot_data[dirichlet] = {
            "extracted": extracted,
        }
    return plot_data


def quadrature_tensor_shape(seq) -> tuple[int, int, int]:
    return seq.quad.ny, seq.quad.nx, seq.quad.nz


def reshape_quadrature_scalar_field(seq, values: jnp.ndarray) -> jnp.ndarray:
    return jnp.asarray(values).reshape(quadrature_tensor_shape(seq))


def k3_metric_tensor(seq) -> jnp.ndarray:
    return reshape_quadrature_scalar_field(seq, 1.0 / seq.geometry.jacobian_j)


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
) -> list[K3BenchmarkReport]:
    n_rhs = rhs_batch.shape[0]

    def A_mv(x):
        return apply_mass_matrix(seq, operators, x, 3, dirichlet=dirichlet)

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
            K3BenchmarkReport(
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


def benchmark_k3_preconditioners(
    seq,
    operators,
    *,
    dirichlet: bool,
    n_rhs: int = 8,
    seed: int = 0,
    tensor_ranks: tuple[int, ...] = (1, 3, 5),
    richardson_steps: tuple[int, ...] = (4, 8, 16, 32),
    richardson_power_iterations: int = 20,
    richardson_damping_safety: float = 0.8,
    tensor_operator_cache: dict[int, object] | None = None,
) -> list[K3BenchmarkReport]:
    rhs_size = seq.n3_dbc if dirichlet else seq.n3
    rhs_batch = jax.random.normal(
        jax.random.PRNGKey(seed + 100 * int(dirichlet)),
        (n_rhs, rhs_size),
        dtype=jnp.float64,
    )

    def mass_apply(x: jnp.ndarray) -> jnp.ndarray:
        return apply_mass_matrix(seq, operators, x, 3, dirichlet=dirichlet)

    def jacobi_apply(x: jnp.ndarray) -> jnp.ndarray:
        return apply_mass_matrix_preconditioner(
            seq, operators, x, 3, dirichlet=dirichlet, kind="jacobi"
        )

    richardson_max_eig = estimate_preconditioned_max_eigenvalue_apply(
        mass_apply,
        jacobi_apply,
        rhs_size,
        n_iter=richardson_power_iterations,
        seed=seed + 310,
    )
    richardson_omega = richardson_damping_safety / richardson_max_eig if richardson_max_eig > 0.0 else 1.0

    labels = {
        "none": lambda x: x,
        "jacobi": jacobi_apply,
    }

    for steps in richardson_steps:
        labels[f"richardson-{steps}"] = build_richardson_apply_preconditioner(
            mass_apply,
            jacobi_apply,
            steps=steps,
            omega=richardson_omega,
        )

    for tensor_rank in tensor_ranks:
        tensor_operators = tensor_operator_cache[tensor_rank] if tensor_operator_cache is not None else assemble_tensor_mass_preconditioner(
            seq,
            operators=operators,
            ks=(3,),
            rank=tensor_rank,
            cp_kwargs=TENSOR_CP_KWARGS,
        )
        labels[f"tensor-r{tensor_rank}"] = (
            lambda x, tensor_ops=tensor_operators: apply_mass_matrix_preconditioner(
                seq,
                tensor_ops,
                x,
                3,
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


def print_k3_benchmark_reports(reports: list[K3BenchmarkReport]):
    print("-" * 112)
    print(
        f"{'label':<20} {'avg iters':>10} {'std':>8} {'max':>6} {'avg ms':>10} {'std ms':>10} {'max ms':>10} {'avg relres':>14} {'std relres':>14} {'max relres':>14}"
    )
    for report in reports:
        print(
            f"{report.label:<20} {report.avg_iters:>10.2f} {report.std_iters:>8.2f} {report.max_iters:>6d} "
            f"{report.avg_time_ms:>10.2f} {report.std_time_ms:>10.2f} {report.max_time_ms:>10.2f} "
            f"{report.avg_relative_residual:>14.3e} {report.std_relative_residual:>14.3e} {report.max_relative_residual:>14.3e}"
        )


def _symmetrize(matrix: jnp.ndarray) -> jnp.ndarray:
    return 0.5 * (matrix + matrix.T)


def estimate_max_eigenvalue_power(
    matrix: jnp.ndarray,
    *,
    n_iter: int = 10,
    seed: int = 0,
) -> float:
    matrix = _symmetrize(matrix)
    vector = jax.random.normal(jax.random.PRNGKey(seed), (matrix.shape[0],), dtype=jnp.float64)
    norm = jnp.linalg.norm(vector)
    vector = vector / jnp.where(norm > 0, norm, 1.0)
    rayleigh = 0.0
    for _ in range(n_iter):
        image = matrix @ vector
        image_norm = jnp.linalg.norm(image)
        safe_norm = jnp.where(image_norm > 0, image_norm, 1.0)
        vector = image / safe_norm
        rayleigh = float(jnp.vdot(vector, matrix @ vector).real)
    return max(rayleigh, 0.0)


def estimate_preconditioned_max_eigenvalue_apply(
    operator_apply,
    smoother_apply,
    size: int,
    *,
    n_iter: int = 10,
    seed: int = 0,
) -> float:
    vector = jax.random.normal(jax.random.PRNGKey(seed), (size,), dtype=jnp.float64)

    def operator_norm(x: jnp.ndarray) -> jnp.ndarray:
        ax = operator_apply(x)
        return jnp.sqrt(jnp.abs(jnp.vdot(x, ax).real))

    vector = vector / jnp.where(operator_norm(vector) > 0, operator_norm(vector), 1.0)
    rayleigh = 0.0
    for _ in range(n_iter):
        image = smoother_apply(operator_apply(vector))
        image_norm = operator_norm(image)
        safe_norm = jnp.where(image_norm > 0, image_norm, 1.0)
        vector = image / safe_norm
        rayleigh = float(jnp.vdot(vector, operator_apply(smoother_apply(operator_apply(vector)))).real)
    return max(rayleigh, 0.0)


def build_richardson_apply_preconditioner(
    operator_apply,
    smoother_apply,
    *,
    steps: int,
    omega: float,
):
    if steps < 1:
        raise ValueError("Richardson step count must be positive")

    def apply(rhs: jnp.ndarray) -> jnp.ndarray:
        x = jnp.zeros_like(rhs)
        residual = rhs
        for _ in range(steps):
            correction = smoother_apply(residual)
            x = x + omega * correction
            residual = residual - omega * operator_apply(correction)
        return x

    return apply


def benchmark_k3_saddle_preconditioners(
    seq,
    operators,
    *,
    dirichlet: bool,
    extra_upper_preconditioners: dict[str, jnp.ndarray] | None = None,
    extra_block_preconditioners: dict[str, jnp.ndarray] | None = None,
    n_rhs: int = 8,
    seed: int = 0,
) -> list[K3SaddleBenchmarkReport]:
    if dirichlet:
        raise ValueError("Start with the nullspace-free k=3 case: dirichlet=False")

    n_upper = seq.n3_dbc if dirichlet else seq.n3
    n_lower = seq.n2_dbc if dirichlet else seq.n2
    rhs_batch = jax.random.normal(
        jax.random.PRNGKey(seed + 700),
        (n_rhs, n_upper),
        dtype=jnp.float64,
    )

    def stiffness_matvec(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros_like(x)

    def derivative_matvec(s: jnp.ndarray) -> jnp.ndarray:
        return apply_derivative_matrix(
            seq,
            operators,
            s,
            2,
            dirichlet_in=dirichlet,
            dirichlet_out=dirichlet,
        )

    def derivative_T_matvec(u: jnp.ndarray) -> jnp.ndarray:
        return apply_derivative_matrix(
            seq,
            operators,
            u,
            2,
            dirichlet_in=dirichlet,
            dirichlet_out=dirichlet,
            transpose=True,
        )

    def mass_lower_matvec(s: jnp.ndarray) -> jnp.ndarray:
        return apply_mass_matrix(seq, operators, s, 2, dirichlet=dirichlet)

    entries: list[tuple[str, str, callable]] = [
        (
            "upper-none",
            "split",
            lambda x: x,
        ),
        (
            "upper-jacobi",
            "split",
            lambda x: apply_mass_matrix_preconditioner(
                seq,
                operators,
                x,
                3,
                dirichlet=dirichlet,
                kind="jacobi",
            ),
        ),
    ]

    if extra_upper_preconditioners is not None:
        for label, upper_preconditioner in extra_upper_preconditioners.items():
            if callable(upper_preconditioner):
                entries.append((label, "split", upper_preconditioner))
            else:
                entries.append(
                    (label, "split", lambda x, dense_upper=upper_preconditioner: dense_upper @ x)
                )

    if extra_block_preconditioners is not None:
        for label, block_preconditioner in extra_block_preconditioners.items():
            if callable(block_preconditioner):
                entries.append((label, "block", block_preconditioner))
            else:
                entries.append(
                    (label, "block", lambda x, dense_block=block_preconditioner: dense_block @ x)
                )

    def lower_preconditioner(x: jnp.ndarray) -> jnp.ndarray:
        return apply_mass_matrix_preconditioner(
            seq,
            operators,
            x,
            2,
            dirichlet=dirichlet,
            kind="tensor",
        )

    def saddle_matvec(u: jnp.ndarray, s: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return derivative_matvec(s), derivative_T_matvec(u) - mass_lower_matvec(s)

    reports = []
    for label, preconditioner_kind, preconditioner in entries:
        @jax.jit
        def solve(rhs):
            if preconditioner_kind == "block":
                u, sigma, info = solve_saddle_point_minres(
                    stiffness_matvec=stiffness_matvec,
                    derivative_matvec=derivative_matvec,
                    derivative_T_matvec=derivative_T_matvec,
                    mass_lower_matvec=mass_lower_matvec,
                    b_upper=rhs,
                    n_upper=n_upper,
                    n_lower=n_lower,
                    precond_matvec=preconditioner,
                    tol=seq.tol,
                    maxiter=seq.maxiter,
                )
            else:
                u, sigma, info = solve_saddle_point_minres(
                    stiffness_matvec=stiffness_matvec,
                    derivative_matvec=derivative_matvec,
                    derivative_T_matvec=derivative_T_matvec,
                    mass_lower_matvec=mass_lower_matvec,
                    b_upper=rhs,
                    n_upper=n_upper,
                    n_lower=n_lower,
                    precond_upper=preconditioner,
                    precond_lower=lower_preconditioner,
                    tol=seq.tol,
                    maxiter=seq.maxiter,
                )
            res_upper, res_lower = saddle_matvec(u, sigma)
            res_upper = res_upper - rhs
            residual = jnp.sqrt(
                jnp.linalg.norm(res_upper) ** 2 + jnp.linalg.norm(res_lower) ** 2
            )
            rhs_norm = jnp.where(jnp.linalg.norm(rhs) > 0, jnp.linalg.norm(rhs), 1.0)
            return u, sigma, info, residual / rhs_norm

        u0, sigma0, _, _ = solve(rhs_batch[0])
        jax.block_until_ready((u0, sigma0))

        iterations = []
        times_ms = []
        residuals = []
        for rhs in rhs_batch:
            t0 = time.perf_counter()
            u, sigma, info, residual = solve(rhs)
            jax.block_until_ready((u, sigma))
            times_ms.append((time.perf_counter() - t0) * 1e3)
            iterations.append(int(jnp.abs(info)))
            residuals.append(float(residual))

        iterations_array = jnp.asarray(iterations)
        times_ms_array = jnp.asarray(times_ms)
        residuals_array = jnp.asarray(residuals)
        reports.append(
            K3SaddleBenchmarkReport(
                label=label,
                n_rhs=n_rhs,
                avg_iters=float(jnp.mean(iterations_array)),
                std_iters=float(jnp.std(iterations_array)),
                max_iters=int(jnp.max(iterations_array)),
                avg_time_ms=float(jnp.mean(times_ms_array)),
                std_time_ms=float(jnp.std(times_ms_array)),
                max_time_ms=float(jnp.max(times_ms_array)),
                avg_relative_residual=float(jnp.mean(residuals_array)),
                std_relative_residual=float(jnp.std(residuals_array)),
                max_relative_residual=float(jnp.max(residuals_array)),
            )
        )
    return reports


def print_k3_saddle_benchmark_reports(reports: list[K3SaddleBenchmarkReport]):
    print("-" * 112)
    print(
        f"{'label':<44} {'avg iters':>10} {'std':>8} {'max':>6} {'avg ms':>10} {'std ms':>10} {'max ms':>10} {'avg relres':>14} {'std relres':>14} {'max relres':>14}"
    )
    for report in reports:
        print(
            f"{report.label:<44} {report.avg_iters:>10.2f} {report.std_iters:>8.2f} {report.max_iters:>6d} "
            f"{report.avg_time_ms:>10.2f} {report.std_time_ms:>10.2f} {report.max_time_ms:>10.2f} "
            f"{report.avg_relative_residual:>14.3e} {report.std_relative_residual:>14.3e} {report.max_relative_residual:>14.3e}"
        )


def _dense_operator_from_apply(apply, size: int) -> jnp.ndarray:
    eye = jnp.eye(size, dtype=jnp.float64)
    columns = [apply(eye[:, idx]) for idx in range(size)]
    return jnp.stack(columns, axis=1)


def assemble_weighted_1d_overlap(
    row_basis: jnp.ndarray,
    col_basis: jnp.ndarray,
    weights: jnp.ndarray,
) -> jnp.ndarray:
    return (row_basis * weights[None, :]) @ col_basis.T


def restrict_2d_window(
    raw_matrix: jnp.ndarray,
    row_start: int,
    n_rows: int,
    col_start: int,
    n_cols: int,
) -> jnp.ndarray:
    return raw_matrix[row_start:row_start + n_rows, col_start:col_start + n_cols]


def kron3(left: jnp.ndarray, middle: jnp.ndarray, right: jnp.ndarray) -> jnp.ndarray:
    return jnp.kron(jnp.kron(left, middle), right)


def build_k2_stiffness_kronecker_terms(
    seq,
    *,
    dirichlet: bool,
    cp_weights: jnp.ndarray,
    cp_factors: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> tuple[
    tuple[str, ...],
    dict[str, dict[str, tuple[int, int, int] | int | jnp.ndarray]],
    list[dict[tuple[str, str], tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]],
]:
    block_order = ("r_bulk", "theta", "zeta")
    types = seq.basis_0.types
    component_specs = {
        "r_bulk": {
            "shape": _r_bulk_shape_k2(seq, dirichlet),
            "radial_basis": grad_1d(seq.d_basis_r_jk, types[0]),
            "theta_basis": seq.d_basis_t_jk,
            "zeta_basis": seq.d_basis_z_jk,
            "radial_start": 2,
        },
        "theta": {
            "shape": _theta_shape_k2(seq, dirichlet),
            "radial_basis": seq.d_basis_r_jk,
            "theta_basis": grad_1d(seq.d_basis_t_jk, types[1]),
            "zeta_basis": seq.d_basis_z_jk,
            "radial_start": 1,
        },
        "zeta": {
            "shape": _zeta_shape_k2(seq, dirichlet),
            "radial_basis": seq.d_basis_r_jk,
            "theta_basis": seq.d_basis_t_jk,
            "zeta_basis": grad_1d(seq.d_basis_z_jk, types[2]),
            "radial_start": 1,
        },
    }

    cp_weights = jnp.asarray(cp_weights, dtype=jnp.float64)
    factor_theta, factor_r, factor_z = tuple(jnp.asarray(factor, dtype=jnp.float64) for factor in cp_factors)
    if cp_weights.shape[0] == 0:
        raise ValueError("empty CP decomposition")

    terms = []
    for idx in range(cp_weights.shape[0]):
        radial_weights = seq.quad.w_x * (cp_weights[idx] * factor_r[:, idx])
        theta_weights = seq.quad.w_y * factor_theta[:, idx]
        zeta_weights = seq.quad.w_z * factor_z[:, idx]
        term = {}
        for row_name in block_order:
            row_spec = component_specs[row_name]
            row_nr = row_spec["shape"][0]
            for col_name in block_order:
                col_spec = component_specs[col_name]
                col_nr = col_spec["shape"][0]
                radial_block = restrict_2d_window(
                    assemble_weighted_1d_overlap(
                        row_spec["radial_basis"],
                        col_spec["radial_basis"],
                        radial_weights,
                    ),
                    row_spec["radial_start"],
                    row_nr,
                    col_spec["radial_start"],
                    col_nr,
                )
                theta_block = assemble_weighted_1d_overlap(
                    row_spec["theta_basis"],
                    col_spec["theta_basis"],
                    theta_weights,
                )
                zeta_block = assemble_weighted_1d_overlap(
                    row_spec["zeta_basis"],
                    col_spec["zeta_basis"],
                    zeta_weights,
                )
                term[(row_name, col_name)] = (
                    radial_block,
                    theta_block,
                    zeta_block,
                )
        terms.append(term)

    return block_order, component_specs, terms


def assemble_k2_stiffness_kronecker_bulk_matrix(
    block_order: tuple[str, ...],
    component_specs: dict[str, dict[str, tuple[int, int, int] | int | jnp.ndarray]],
    terms: list[dict[tuple[str, str], tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]],
) -> jnp.ndarray:
    blocks = {
        (row_name, col_name): jnp.zeros(
            (
                int(jnp.prod(jnp.asarray(component_specs[row_name]["shape"]))),
                int(jnp.prod(jnp.asarray(component_specs[col_name]["shape"]))),
            ),
            dtype=jnp.float64,
        )
        for row_name in block_order
        for col_name in block_order
    }
    for term in terms:
        for key, (radial_block, theta_block, zeta_block) in term.items():
            blocks[key] = blocks[key] + kron3(radial_block, theta_block, zeta_block)
    assembled_rows = []
    for row_name in block_order:
        assembled_rows.append([blocks[(row_name, col_name)] for col_name in block_order])
    return _symmetrize(jnp.block(assembled_rows))


def build_k2_stiffness_kronecker_preconditioner(
    seq,
    operators,
    *,
    dirichlet: bool,
    cp_weights: jnp.ndarray,
    cp_factors: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> dict[str, object]:
    exact_matrix = jnp.asarray(dense_stiffness_matrix(seq, operators, 2, dirichlet=dirichlet))
    block_indices = _tensor_block_indices_k2(seq, dirichlet)
    surgery_indices = block_indices["surgery"]
    bulk_indices = block_indices["bulk"]
    ass = exact_matrix[surgery_indices][:, surgery_indices]
    asb = exact_matrix[surgery_indices][:, bulk_indices]
    abs_ = exact_matrix[bulk_indices][:, surgery_indices]

    block_order, component_specs, terms = build_k2_stiffness_kronecker_terms(
        seq,
        dirichlet=dirichlet,
        cp_weights=cp_weights,
        cp_factors=cp_factors,
    )
    bulk_model = assemble_k2_stiffness_kronecker_bulk_matrix(block_order, component_specs, terms)
    approx_matrix = exact_matrix.at[bulk_indices[:, None], bulk_indices].set(bulk_model)
    approx_matrix = _symmetrize(approx_matrix)
    approx_pinv = _symmetrize(jnp.linalg.pinv(approx_matrix))
    bulk_inverse = _symmetrize(jnp.linalg.inv(bulk_model))
    schur_inverse = _symmetrize(jnp.linalg.inv(ass - asb @ (bulk_inverse @ abs_)))

    def apply(rhs: jnp.ndarray) -> jnp.ndarray:
        rhs_s = rhs[surgery_indices]
        rhs_b = rhs[bulk_indices]
        y = bulk_inverse @ rhs_b
        z = schur_inverse @ (rhs_s - asb @ y)
        x_b = y - bulk_inverse @ (abs_ @ z)
        out = jnp.zeros_like(rhs)
        out = out.at[surgery_indices].set(z)
        out = out.at[bulk_indices].set(x_b)
        return out

    return {
        "matrix": exact_matrix,
        "approx_matrix": approx_matrix,
        "approx_pinv": approx_pinv,
        "bulk_exact": exact_matrix[bulk_indices][:, bulk_indices],
        "block_order": block_order,
        "component_specs": component_specs,
        "terms": terms,
        "bulk_model": bulk_model,
        "bulk_inverse": bulk_inverse,
        "apply": apply,
    }


def compare_k2_stiffness_kronecker_approximations(
    exact_bulk_matrix: jnp.ndarray,
    approx_matrices: dict[str, jnp.ndarray],
    *,
    n_rhs: int = 8,
    seed: int = 0,
) -> list[K2StiffnessKroneckerApproximationReport]:
    exact_matrix = _symmetrize(exact_bulk_matrix)
    exact_pinv = _symmetrize(jnp.linalg.pinv(exact_matrix))
    exact_range_projector = _symmetrize(exact_pinv @ exact_matrix)
    matrix_norm = jnp.where(jnp.linalg.norm(exact_matrix) > 0, jnp.linalg.norm(exact_matrix), 1.0)
    inverse_norm = jnp.where(jnp.linalg.norm(exact_pinv) > 0, jnp.linalg.norm(exact_pinv), 1.0)
    projector_norm = jnp.where(jnp.linalg.norm(exact_range_projector) > 0, jnp.linalg.norm(exact_range_projector), 1.0)
    probe_batch = jax.random.normal(
        jax.random.PRNGKey(seed + 1700),
        (n_rhs, exact_matrix.shape[0]),
        dtype=jnp.float64,
    )

    reports = []
    for label, approx_matrix in approx_matrices.items():
        approx_matrix = _symmetrize(approx_matrix)
        approx_pinv = _symmetrize(jnp.linalg.pinv(approx_matrix))
        approx_range_projector = _symmetrize(approx_pinv @ exact_matrix)

        action_errors = []
        for probe in probe_batch:
            rhs = exact_matrix @ probe
            exact_output = exact_pinv @ rhs
            approx_output = approx_pinv @ rhs
            rel_error = jnp.linalg.norm(approx_output - exact_output) / jnp.where(
                jnp.linalg.norm(exact_output) > 0,
                jnp.linalg.norm(exact_output),
                1.0,
            )
            action_errors.append(float(rel_error))

        action_errors_array = jnp.asarray(action_errors)
        reports.append(
            K2StiffnessKroneckerApproximationReport(
                label=label,
                matrix_relative_error=float(jnp.linalg.norm(approx_matrix - exact_matrix) / matrix_norm),
                inverse_relative_error=float(jnp.linalg.norm(approx_pinv - exact_pinv) / inverse_norm),
                range_projector_relative_error=float(
                    jnp.linalg.norm(approx_range_projector - exact_range_projector) / projector_norm
                ),
                avg_range_action_error=float(jnp.mean(action_errors_array)),
                max_range_action_error=float(jnp.max(action_errors_array)),
            )
        )
    return reports


def print_k2_stiffness_kronecker_approximation_reports(
    reports: list[K2StiffnessKroneckerApproximationReport],
):
    print("-" * 112)
    print(
        f"{'label':<20} {'K rel err':>16} {'K^+ rel err':>16} {'|B K-P| rel':>16} {'avg range err':>16} {'max range err':>16}"
    )
    for report in reports:
        print(
            f"{report.label:<20} {report.matrix_relative_error:>16.3e} {report.inverse_relative_error:>16.3e} "
            f"{report.range_projector_relative_error:>16.3e} {report.avg_range_action_error:>16.3e} {report.max_range_action_error:>16.3e}"
        )


def extract_k2_surgery_split_data_from_apply(
    seq,
    *,
    dirichlet: bool,
    operator_apply,
) -> dict[str, jnp.ndarray]:
    block_indices = _tensor_block_indices_k2(seq, dirichlet)
    surgery_indices = jnp.asarray(block_indices["surgery"])
    bulk_indices = jnp.asarray(block_indices["bulk"])
    n_total = seq.n2_dbc if dirichlet else seq.n2
    n_surgery = surgery_indices.shape[0]

    ass_columns = []
    abs_columns = []
    for local_idx in range(n_surgery):
        basis = jnp.zeros((n_total,), dtype=jnp.float64)
        basis = basis.at[surgery_indices[local_idx]].set(1.0)
        image = operator_apply(basis)
        ass_columns.append(image[surgery_indices])
        abs_columns.append(image[bulk_indices])

    ass = _symmetrize(jnp.stack(ass_columns, axis=1))
    abs_ = jnp.stack(abs_columns, axis=1)
    asb = abs_.T
    return {
        "surgery_indices": surgery_indices,
        "bulk_indices": bulk_indices,
        "ass": ass,
        "asb": asb,
        "abs": abs_,
    }


def extract_k2_surgery_split_data(
    seq,
    operators,
    *,
    dirichlet: bool,
) -> dict[str, jnp.ndarray]:
    return extract_k2_surgery_split_data_from_apply(
        seq,
        dirichlet=dirichlet,
        operator_apply=lambda x: apply_stiffness(seq, operators, x, 2, dirichlet=dirichlet),
    )


def build_k2_bulk_restricted_apply_from_apply(
    seq,
    *,
    dirichlet: bool,
    bulk_indices: jnp.ndarray,
    operator_apply,
    smoother_apply,
):
    n_total = seq.n2_dbc if dirichlet else seq.n2

    def bulk_apply(x_bulk: jnp.ndarray) -> jnp.ndarray:
        full = jnp.zeros((n_total,), dtype=x_bulk.dtype)
        full = full.at[bulk_indices].set(x_bulk)
        return operator_apply(full)[bulk_indices]

    def bulk_smoother_apply(x_bulk: jnp.ndarray) -> jnp.ndarray:
        full = jnp.zeros((n_total,), dtype=x_bulk.dtype)
        full = full.at[bulk_indices].set(x_bulk)
        return smoother_apply(full)[bulk_indices]

    return bulk_apply, bulk_smoother_apply


def build_k2_bulk_restricted_apply(
    seq,
    operators,
    *,
    dirichlet: bool,
    bulk_indices: jnp.ndarray,
):
    return build_k2_bulk_restricted_apply_from_apply(
        seq,
        dirichlet=dirichlet,
        bulk_indices=bulk_indices,
        operator_apply=lambda x: apply_stiffness(seq, operators, x, 2, dirichlet=dirichlet),
        smoother_apply=lambda x: apply_hodge_laplacian_preconditioner(
            seq,
            operators,
            x,
            2,
            dirichlet=dirichlet,
            kind="jacobi",
        ),
    )


def build_k2_surgery_schur_richardson_preconditioner_from_apply(
    seq,
    *,
    dirichlet: bool,
    split_data: dict[str, jnp.ndarray],
    steps: int,
    operator_apply,
    smoother_apply,
    power_iterations: int = 20,
    damping_safety: float = 0.8,
):
    bulk_indices = split_data["bulk_indices"]
    ass = split_data["ass"]
    asb = split_data["asb"]
    abs_ = split_data["abs"]
    bulk_apply, bulk_smoother_apply = build_k2_bulk_restricted_apply_from_apply(
        seq,
        dirichlet=dirichlet,
        bulk_indices=bulk_indices,
        operator_apply=operator_apply,
        smoother_apply=smoother_apply,
    )
    bulk_size = bulk_indices.shape[0]
    max_eig = estimate_preconditioned_max_eigenvalue_apply(
        bulk_apply,
        bulk_smoother_apply,
        bulk_size,
        n_iter=power_iterations,
    )
    omega = damping_safety / max_eig if max_eig > 0.0 else 1.0
    bulk_richardson_apply = build_richardson_apply_preconditioner(
        bulk_apply,
        bulk_smoother_apply,
        steps=steps,
        omega=omega,
    )
    bulk_inv_abs = jnp.stack(
        [bulk_richardson_apply(abs_[:, idx]) for idx in range(abs_.shape[1])],
        axis=1,
    )
    schur = _symmetrize(ass - asb @ bulk_inv_abs)
    schur_inverse = _symmetrize(jnp.linalg.inv(schur))

    def apply(rhs: jnp.ndarray) -> jnp.ndarray:
        rhs_s = rhs[split_data["surgery_indices"]]
        rhs_b = rhs[split_data["bulk_indices"]]
        y = bulk_richardson_apply(rhs_b)
        z = schur_inverse @ (rhs_s - asb @ y)
        x_b = y - bulk_richardson_apply(abs_ @ z)
        out = jnp.zeros_like(rhs)
        out = out.at[split_data["surgery_indices"]].set(z)
        out = out.at[split_data["bulk_indices"]].set(x_b)
        return out

    return apply, {"steps": int(steps), "omega": float(omega), "max_eig": float(max_eig)}


def build_k2_surgery_schur_richardson_preconditioner(
    seq,
    operators,
    *,
    dirichlet: bool,
    split_data: dict[str, jnp.ndarray],
    steps: int,
    power_iterations: int = 20,
    damping_safety: float = 0.8,
):
    return build_k2_surgery_schur_richardson_preconditioner_from_apply(
        seq,
        dirichlet=dirichlet,
        split_data=split_data,
        steps=steps,
        operator_apply=lambda x: apply_stiffness(seq, operators, x, 2, dirichlet=dirichlet),
        smoother_apply=lambda x: apply_hodge_laplacian_preconditioner(
            seq,
            operators,
            x,
            2,
            dirichlet=dirichlet,
            kind="jacobi",
        ),
        power_iterations=power_iterations,
        damping_safety=damping_safety,
    )


def benchmark_k2_range_preconditioners(
    seq,
    operators,
    *,
    dirichlet: bool,
    richardson_steps: tuple[int, ...] = (4, 8, 16, 32),
    power_iterations: int = 20,
    damping_safety: float = 0.8,
    n_rhs: int = 8,
    seed: int = 0,
) -> tuple[list[K2BenchmarkReport], dict[str, dict[str, float | int]]]:
    if dirichlet:
        raise ValueError("Start with the nullspace-aware free k=2 case: dirichlet=False")

    n2 = seq.n2_dbc if dirichlet else seq.n2
    n3 = seq.n3_dbc if dirichlet else seq.n3

    def k2_apply(x: jnp.ndarray) -> jnp.ndarray:
        return apply_stiffness(seq, operators, x, 2, dirichlet=dirichlet)

    def k2_jacobi_apply(x: jnp.ndarray) -> jnp.ndarray:
        return apply_hodge_laplacian_preconditioner(
            seq,
            operators,
            x,
            2,
            dirichlet=dirichlet,
            kind="jacobi",
        )

    raw_batch = jax.random.normal(
        jax.random.PRNGKey(seed + 2600),
        (n_rhs, n3),
        dtype=jnp.float64,
    )
    rhs_batch = jnp.stack(
        [
            apply_derivative_matrix(
                seq,
                operators,
                raw_batch[idx],
                2,
                dirichlet_in=dirichlet,
                dirichlet_out=dirichlet,
                transpose=True,
            )
            for idx in range(n_rhs)
        ],
        axis=0,
    )

    split_data = extract_k2_surgery_split_data(
        seq,
        operators,
        dirichlet=dirichlet,
    )
    pure_max_eig = estimate_preconditioned_max_eigenvalue_apply(
        k2_apply,
        k2_jacobi_apply,
        n2,
        n_iter=power_iterations,
        seed=seed + 2700,
    )
    pure_omega = damping_safety / pure_max_eig if pure_max_eig > 0.0 else 1.0

    labels = {}
    parameters = {}
    for steps in richardson_steps:
        pure_label = f"pure-richardson-{steps}"
        labels[pure_label] = build_richardson_apply_preconditioner(
            k2_apply,
            k2_jacobi_apply,
            steps=steps,
            omega=pure_omega,
        )
        parameters[pure_label] = {
            "steps": int(steps),
            "omega": float(pure_omega),
            "max_eig": float(pure_max_eig),
        }

        split_label = f"surgery-schur-richardson-{steps}"
        labels[split_label], parameters[split_label] = build_k2_surgery_schur_richardson_preconditioner(
            seq,
            operators,
            dirichlet=dirichlet,
            split_data=split_data,
            steps=steps,
            power_iterations=power_iterations,
            damping_safety=damping_safety,
        )

    reports = []
    for label, preconditioner in labels.items():
        @jax.jit
        def solve(rhs):
            x, info = solve_singular_cg(
                k2_apply,
                rhs,
                precond_matvec=preconditioner,
                tol=seq.tol,
                maxiter=seq.maxiter,
            )
            residual = k2_apply(x) - rhs
            rhs_norm = jnp.where(jnp.linalg.norm(rhs) > 0, jnp.linalg.norm(rhs), 1.0)
            return x, info, jnp.linalg.norm(residual) / rhs_norm

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

        iterations_array = jnp.asarray(iterations)
        times_ms_array = jnp.asarray(times_ms)
        residuals_array = jnp.asarray(residuals)
        reports.append(
            K2BenchmarkReport(
                label=label,
                n_rhs=n_rhs,
                avg_iters=float(jnp.mean(iterations_array)),
                std_iters=float(jnp.std(iterations_array)),
                max_iters=int(jnp.max(iterations_array)),
                avg_time_ms=float(jnp.mean(times_ms_array)),
                std_time_ms=float(jnp.std(times_ms_array)),
                max_time_ms=float(jnp.max(times_ms_array)),
                avg_relative_residual=float(jnp.mean(residuals_array)),
                std_relative_residual=float(jnp.std(residuals_array)),
                max_relative_residual=float(jnp.max(residuals_array)),
            )
        )

    return reports, parameters


def benchmark_k2_mass_preconditioners(
    seq,
    operators,
    *,
    dirichlet: bool,
    richardson_steps: tuple[int, ...] = (4, 8, 16, 32),
    power_iterations: int = 20,
    damping_safety: float = 0.8,
    n_rhs: int = 8,
    seed: int = 0,
) -> tuple[list[K2BenchmarkReport], dict[str, dict[str, float | int]]]:
    n2 = seq.n2_dbc if dirichlet else seq.n2

    def m2_apply(x: jnp.ndarray) -> jnp.ndarray:
        return apply_mass_matrix(seq, operators, x, 2, dirichlet=dirichlet)

    def m2_jacobi_apply(x: jnp.ndarray) -> jnp.ndarray:
        return apply_mass_matrix_preconditioner(
            seq,
            operators,
            x,
            2,
            dirichlet=dirichlet,
            kind="jacobi",
        )

    rhs_batch = jax.random.normal(
        jax.random.PRNGKey(seed + 2800),
        (n_rhs, n2),
        dtype=jnp.float64,
    )

    split_data = extract_k2_surgery_split_data_from_apply(
        seq,
        dirichlet=dirichlet,
        operator_apply=m2_apply,
    )
    pure_max_eig = estimate_preconditioned_max_eigenvalue_apply(
        m2_apply,
        m2_jacobi_apply,
        n2,
        n_iter=power_iterations,
        seed=seed + 2900,
    )
    pure_omega = damping_safety / pure_max_eig if pure_max_eig > 0.0 else 1.0

    labels = {}
    parameters = {}
    for steps in richardson_steps:
        pure_label = f"pure-richardson-{steps}"
        labels[pure_label] = build_richardson_apply_preconditioner(
            m2_apply,
            m2_jacobi_apply,
            steps=steps,
            omega=pure_omega,
        )
        parameters[pure_label] = {
            "steps": int(steps),
            "omega": float(pure_omega),
            "max_eig": float(pure_max_eig),
        }

        split_label = f"surgery-schur-richardson-{steps}"
        labels[split_label], parameters[split_label] = build_k2_surgery_schur_richardson_preconditioner_from_apply(
            seq,
            dirichlet=dirichlet,
            split_data=split_data,
            steps=steps,
            operator_apply=m2_apply,
            smoother_apply=m2_jacobi_apply,
            power_iterations=power_iterations,
            damping_safety=damping_safety,
        )

    reports = []
    for label, preconditioner in labels.items():
        @jax.jit
        def solve(rhs):
            x, info = solve_singular_cg(
                m2_apply,
                rhs,
                mass_matvec=m2_apply,
                precond_matvec=preconditioner,
                tol=seq.tol,
                maxiter=seq.maxiter,
            )
            residual = m2_apply(x) - rhs
            rhs_norm = jnp.where(jnp.linalg.norm(rhs) > 0, jnp.linalg.norm(rhs), 1.0)
            return x, info, jnp.linalg.norm(residual) / rhs_norm

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

        iterations_array = jnp.asarray(iterations)
        times_ms_array = jnp.asarray(times_ms)
        residuals_array = jnp.asarray(residuals)
        reports.append(
            K2BenchmarkReport(
                label=label,
                n_rhs=n_rhs,
                avg_iters=float(jnp.mean(iterations_array)),
                std_iters=float(jnp.std(iterations_array)),
                max_iters=int(jnp.max(iterations_array)),
                avg_time_ms=float(jnp.mean(times_ms_array)),
                std_time_ms=float(jnp.std(times_ms_array)),
                max_time_ms=float(jnp.max(times_ms_array)),
                avg_relative_residual=float(jnp.mean(residuals_array)),
                std_relative_residual=float(jnp.std(residuals_array)),
                max_relative_residual=float(jnp.max(residuals_array)),
            )
        )

    return reports, parameters


def print_k2_benchmark_reports(reports: list[K2BenchmarkReport]):
    print("-" * 112)
    print(
        f"{'label':<36} {'avg iters':>10} {'std':>8} {'max':>6} {'avg ms':>10} {'std ms':>10} {'max ms':>10} {'avg relres':>14} {'std relres':>14} {'max relres':>14}"
    )
    for report in reports:
        print(
            f"{report.label:<36} {report.avg_iters:>10.2f} {report.std_iters:>8.2f} {report.max_iters:>6d} "
            f"{report.avg_time_ms:>10.2f} {report.std_time_ms:>10.2f} {report.max_time_ms:>10.2f} "
            f"{report.avg_relative_residual:>14.3e} {report.std_relative_residual:>14.3e} {report.max_relative_residual:>14.3e}"
        )


def diagnose_preconditioner_properties(
    preconditioners: dict[str, callable],
    size: int,
    *,
    metric: str,
    metric_apply=None,
    n_probes: int = 16,
    seed: int = 0,
) -> list[PreconditionerPropertyReport]:
    x_batch = jax.random.normal(
        jax.random.PRNGKey(seed),
        (n_probes, size),
        dtype=jnp.float64,
    )
    y_batch = jax.random.normal(
        jax.random.PRNGKey(seed + 1),
        (n_probes, size),
        dtype=jnp.float64,
    )

    def inner(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        if metric_apply is None:
            return jnp.dot(x, y)
        return jnp.dot(x, metric_apply(y))

    reports = []
    for label, apply_preconditioner in preconditioners.items():
        symmetry_defects = []
        quadratic_forms = []
        for idx in range(n_probes):
            x = x_batch[idx]
            y = y_batch[idx]
            px = apply_preconditioner(x)
            py = apply_preconditioner(y)
            lhs = float(inner(x, py))
            rhs = float(inner(px, y))
            symmetry_scale = max(abs(lhs), abs(rhs), 1.0)
            symmetry_defects.append(abs(lhs - rhs) / symmetry_scale)
            quadratic_forms.append(float(inner(x, px)))

        quadratic_array = jnp.asarray(quadratic_forms)
        symmetry_array = jnp.asarray(symmetry_defects)
        reports.append(
            PreconditionerPropertyReport(
                label=label,
                metric=metric,
                n_probes=n_probes,
                avg_symmetry_defect=float(jnp.mean(symmetry_array)),
                max_symmetry_defect=float(jnp.max(symmetry_array)),
                min_quadratic_form=float(jnp.min(quadratic_array)),
                max_quadratic_form=float(jnp.max(quadratic_array)),
                negative_quadratic_forms=int(jnp.sum(quadratic_array <= 0.0)),
            )
        )
    return reports


def print_preconditioner_property_reports(reports: list[PreconditionerPropertyReport]):
    print("-" * 112)
    print(
        f"{'label':<36} {'metric':<12} {'avg sym defect':>16} {'max sym defect':>16} {'min <x,Px>':>16} {'max <x,Px>':>16} {'# nonpos':>10}"
    )
    for report in reports:
        print(
            f"{report.label:<36} {report.metric:<12} {report.avg_symmetry_defect:>16.3e} {report.max_symmetry_defect:>16.3e} "
            f"{report.min_quadratic_form:>16.3e} {report.max_quadratic_form:>16.3e} {report.negative_quadratic_forms:>10d}"
        )


def diagnose_k3_schur_upper_preconditioners(
    seq,
    operators,
    *,
    dirichlet: bool,
    upper_preconditioners: dict[str, callable] | None = None,
) -> tuple[jnp.ndarray, list[K3SchurApproximationReport]]:
    schur = jnp.asarray(dense_hodge_laplacian(seq, operators, 3, dirichlet=dirichlet))
    schur_inv = _symmetrize(jnp.linalg.inv(schur))
    n_upper = schur.shape[0]
    identity = jnp.eye(n_upper, dtype=jnp.float64)

    labels = {
        "upper-none": lambda x: x,
        "upper-jacobi": lambda x: apply_mass_matrix_preconditioner(
            seq,
            operators,
            x,
            3,
            dirichlet=dirichlet,
            kind="jacobi",
        ),
    }

    reports = []
    for label, apply_upper in labels.items():
        approx_inv = _dense_operator_from_apply(apply_upper, n_upper)
        preconditioned = _symmetrize(approx_inv @ schur)
        eigvals = jnp.linalg.eigvalsh(preconditioned)
        reports.append(
            K3SchurApproximationReport(
                label=label,
                inverse_relative_error=float(
                    jnp.linalg.norm(approx_inv - schur_inv)
                    / jnp.where(jnp.linalg.norm(schur_inv) > 0, jnp.linalg.norm(schur_inv), 1.0)
                ),
                solve_relative_error=float(
                    jnp.linalg.norm(approx_inv @ schur - identity)
                    / jnp.linalg.norm(identity)
                ),
                min_precond_eig=float(jnp.min(eigvals)),
                max_precond_eig=float(jnp.max(eigvals)),
            )
        )

    if upper_preconditioners is not None:
        for label, apply_upper in upper_preconditioners.items():
            approx_inv = _symmetrize(_dense_operator_from_apply(apply_upper, n_upper))
            preconditioned = _symmetrize(approx_inv @ schur)
            eigvals = jnp.linalg.eigvalsh(preconditioned)
            reports.append(
                K3SchurApproximationReport(
                    label=label,
                    inverse_relative_error=float(
                        jnp.linalg.norm(approx_inv - schur_inv)
                        / jnp.where(jnp.linalg.norm(schur_inv) > 0, jnp.linalg.norm(schur_inv), 1.0)
                    ),
                    solve_relative_error=float(
                        jnp.linalg.norm(approx_inv @ schur - identity)
                        / jnp.linalg.norm(identity)
                    ),
                    min_precond_eig=float(jnp.min(eigvals)),
                    max_precond_eig=float(jnp.max(eigvals)),
                )
            )
    return schur, reports


def print_k3_schur_approximation_reports(reports: list[K3SchurApproximationReport]):
    print("-" * 112)
    print(
        f"{'label':<48} {'inv rel err':>16} {'|BS-I| rel':>16} {'min eig(BS)':>16} {'max eig(BS)':>16}"
    )
    for report in reports:
        print(
            f"{report.label:<48} {report.inverse_relative_error:>16.3e} {report.solve_relative_error:>16.3e} "
            f"{report.min_precond_eig:>16.3e} {report.max_precond_eig:>16.3e}"
        )


def build_diagonal_inverse_from_matrix(matrix: jnp.ndarray, *, rtol: float = 1e-12) -> jnp.ndarray:
    diagonal = jnp.diag(matrix)
    scale = jnp.max(jnp.abs(diagonal))
    safe_scale = jnp.where(scale > 0, scale, 1.0)
    floor = rtol * safe_scale
    diagonal_inverse = jnp.where(jnp.abs(diagonal) > floor, 1.0 / diagonal, 0.0)
    return jnp.diag(diagonal_inverse)


def build_symmetric_richardson_inverse(
    operator: jnp.ndarray,
    smoother_inverse: jnp.ndarray,
    *,
    steps: int,
    power_iterations: int = 10,
    max_eig_override: float | None = None,
    damping_safety: float = 0.8,
) -> dict[str, float | int | jnp.ndarray]:
    if steps < 1:
        raise ValueError("Richardson step count must be positive")
    operator = _symmetrize(operator)
    smoother_inverse = _symmetrize(smoother_inverse)
    sqrt_diag = jnp.sqrt(jnp.clip(jnp.diag(smoother_inverse), a_min=0.0))
    smoother_half = jnp.diag(sqrt_diag)
    preconditioned = _symmetrize(smoother_half @ operator @ smoother_half)
    max_eig = max_eig_override if max_eig_override is not None else estimate_max_eigenvalue_power(
        preconditioned,
        n_iter=power_iterations,
    )
    omega = damping_safety / max_eig if max_eig > 0.0 else 1.0
    identity = jnp.eye(operator.shape[0], dtype=jnp.float64)
    residual_iteration = identity - omega * preconditioned
    power = identity
    polynomial = jnp.zeros_like(operator)
    for _ in range(steps):
        polynomial = polynomial + power
        power = power @ residual_iteration
    matrix = _symmetrize(omega * smoother_half @ polynomial @ smoother_half)
    return {
        "matrix": matrix,
        "omega": float(omega),
        "max_eig": max_eig,
        "steps": steps,
    }


def build_k3_tensor_mass_induced_schur_apply(
    seq,
    operators,
    *,
    dirichlet: bool,
):
    def apply(x: jnp.ndarray) -> jnp.ndarray:
        d2_t_x = apply_derivative_matrix(
            seq,
            operators,
            x,
            2,
            dirichlet_in=dirichlet,
            dirichlet_out=dirichlet,
            transpose=True,
        )
        m2_inv_d2_t_x = apply_mass_matrix_preconditioner(
            seq,
            operators,
            d2_t_x,
            2,
            dirichlet=dirichlet,
            kind="tensor",
        )
        return apply_derivative_matrix(
            seq,
            operators,
            m2_inv_d2_t_x,
            2,
            dirichlet_in=dirichlet,
            dirichlet_out=dirichlet,
        )

    return apply


def build_k3_richardson_upper_preconditioners(
    seq,
    operators,
    *,
    dirichlet: bool,
    richardson_steps: tuple[int, ...],
    power_iterations: int = 10,
    damping_safety: float = 0.8,
) -> tuple[dict[str, callable], dict[str, dict[str, float | int]]]:
    n_upper = seq.n3_dbc if dirichlet else seq.n3
    jacobi_apply = lambda x: apply_mass_matrix_preconditioner(
        seq,
        operators,
        x,
        3,
        dirichlet=dirichlet,
        kind="jacobi",
    )
    schur_apply = build_k3_tensor_mass_induced_schur_apply(
        seq,
        operators,
        dirichlet=dirichlet,
    )
    max_eig = estimate_preconditioned_max_eigenvalue_apply(
        schur_apply,
        jacobi_apply,
        n_upper,
        n_iter=power_iterations,
    )
    omega = damping_safety / max_eig if max_eig > 0.0 else 1.0
    upper_preconditioners = {}
    parameters = {}

    for steps in richardson_steps:
        schur_richardson = build_richardson_apply_preconditioner(
            schur_apply,
            jacobi_apply,
            steps=steps,
            omega=omega,
        )
        schur_label = f"upper-schur-richardson-{steps}"
        upper_preconditioners[schur_label] = schur_richardson
        parameters[schur_label] = {
            "steps": int(steps),
            "omega": float(omega),
            "max_eig": float(max_eig),
        }

    return upper_preconditioners, parameters


def build_k3_coupled_preconditioners(
    seq,
    operators,
    *,
    dirichlet: bool,
    upper_preconditioners: dict[str, callable],
) -> dict[str, callable]:
    def lower_preconditioner(x: jnp.ndarray) -> jnp.ndarray:
        return apply_mass_matrix_preconditioner(
            seq,
            operators,
            x,
            2,
            dirichlet=dirichlet,
            kind="tensor",
        )

    coupled_preconditioners = {}

    for label, upper_preconditioner in upper_preconditioners.items():
        coupled_label = label.replace("upper-", "coupled-", 1)

        def apply(x: jnp.ndarray, upper_apply=upper_preconditioner) -> jnp.ndarray:
            n_upper = seq.n3_dbc if dirichlet else seq.n3
            u = x[:n_upper]
            s = x[n_upper:]
            d2_t_u = apply_derivative_matrix(
                seq,
                operators,
                u,
                2,
                dirichlet_in=dirichlet,
                dirichlet_out=dirichlet,
                transpose=True,
            )
            y_u = u
            y_s = s - lower_preconditioner(d2_t_u)
            z_u = upper_apply(y_u)
            z_s = lower_preconditioner(y_s)
            lift = apply_derivative_matrix(
                seq,
                operators,
                z_s,
                2,
                dirichlet_in=dirichlet,
                dirichlet_out=dirichlet,
            )
            return jnp.concatenate([z_u - lift, z_s])

        coupled_preconditioners[coupled_label] = apply

    return coupled_preconditioners


def print_k3_richardson_parameter_reports(parameters: dict[str, dict[str, float | int]]):
    print("-" * 112)
    print(f"{'label':<40} {'steps':>8} {'omega':>16} {'max eig':>16}")
    for label, data in parameters.items():
        print(
            f"{label:<40} {int(data['steps']):>8d} {float(data['omega']):>16.3e} {float(data['max_eig']):>16.3e}"
        )


def _dense_spd_matrix_sqrt(matrix: jnp.ndarray, *, tol: float = 1e-12) -> jnp.ndarray:
    eigvals, eigvecs = jnp.linalg.eigh(_symmetrize(matrix))
    clipped = jnp.clip(eigvals, a_min=tol, a_max=None)
    return (eigvecs * jnp.sqrt(clipped)) @ eigvecs.T


def _dense_k3_saddle_matrix_and_lower_inverse(
    seq,
    operators,
    *,
    dirichlet: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    d2 = jnp.asarray(
        dense_derivative_matrix(
            seq,
            operators,
            2,
            dirichlet_in=dirichlet,
            dirichlet_out=dirichlet,
        )
    )
    m2 = jnp.asarray(dense_mass_matrix(seq, operators, 2, dirichlet=dirichlet))
    zero = jnp.zeros((d2.shape[0], d2.shape[0]), dtype=jnp.float64)
    saddle = jnp.block([[zero, d2], [d2.T, -m2]])
    lower_inverse = _symmetrize(
        _dense_operator_from_apply(
            lambda x: apply_mass_matrix_preconditioner(
                seq,
                operators,
                x,
                2,
                dirichlet=dirichlet,
                kind="tensor",
            ),
            m2.shape[0],
        )
    )
    return saddle, lower_inverse


def _dense_preconditioned_saddle_eigvals(
    saddle: jnp.ndarray,
    upper_inverse: jnp.ndarray,
    lower_inverse: jnp.ndarray,
) -> jnp.ndarray:
    upper_sqrt = _dense_spd_matrix_sqrt(upper_inverse)
    lower_sqrt = _dense_spd_matrix_sqrt(lower_inverse)
    block_sqrt = jnp.block(
        [
            [upper_sqrt, jnp.zeros((upper_sqrt.shape[0], lower_sqrt.shape[1]), dtype=jnp.float64)],
            [jnp.zeros((lower_sqrt.shape[0], upper_sqrt.shape[1]), dtype=jnp.float64), lower_sqrt],
        ]
    )
    preconditioned = _symmetrize(block_sqrt @ saddle @ block_sqrt)
    return jnp.linalg.eigvalsh(preconditioned)


def _dense_preconditioned_saddle_eigvals_block(
    saddle: jnp.ndarray,
    block_inverse: jnp.ndarray,
) -> jnp.ndarray:
    block_sqrt = _dense_spd_matrix_sqrt(block_inverse)
    preconditioned = _symmetrize(block_sqrt @ saddle @ block_sqrt)
    return jnp.linalg.eigvalsh(preconditioned)


def diagnose_k3_saddle_preconditioned_spectra(
    seq,
    operators,
    *,
    dirichlet: bool,
    upper_preconditioners: dict[str, callable] | None = None,
    block_preconditioners: dict[str, callable] | None = None,
) -> list[K3SaddleSpectrumReport]:
    saddle, lower_inverse = _dense_k3_saddle_matrix_and_lower_inverse(
        seq,
        operators,
        dirichlet=dirichlet,
    )
    n_upper = saddle.shape[0] - lower_inverse.shape[0]

    upper_preconditioners = {
        "upper-none": jnp.eye(n_upper, dtype=jnp.float64),
        "upper-jacobi": _symmetrize(
            _dense_operator_from_apply(
                lambda x: apply_mass_matrix_preconditioner(
                    seq,
                    operators,
                    x,
                    3,
                    dirichlet=dirichlet,
                    kind="jacobi",
                ),
                n_upper,
            )
        ),
    }
    if upper_preconditioners is not None:
        for label, apply_upper in upper_preconditioners.items():
            upper_preconditioners[label] = _symmetrize(_dense_operator_from_apply(apply_upper, n_upper))

    reports = []
    for label, upper_inverse in upper_preconditioners.items():
        eigvals = _dense_preconditioned_saddle_eigvals(saddle, upper_inverse, lower_inverse)
        abs_eigvals = jnp.abs(eigvals)
        reports.append(
            K3SaddleSpectrumReport(
                label=label,
                min_eig=float(jnp.min(eigvals)),
                max_eig=float(jnp.max(eigvals)),
                min_abs_eig=float(jnp.min(abs_eigvals)),
                max_abs_eig=float(jnp.max(abs_eigvals)),
                pos_count=int(jnp.sum(eigvals > 0.0)),
                neg_count=int(jnp.sum(eigvals < 0.0)),
            )
        )

    if block_preconditioners is not None:
        n_total = saddle.shape[0]
        for label, apply_block in block_preconditioners.items():
            block_inverse = _symmetrize(_dense_operator_from_apply(apply_block, n_total))
            eigvals = _dense_preconditioned_saddle_eigvals_block(
                saddle,
                block_inverse,
            )
            abs_eigvals = jnp.abs(eigvals)
            reports.append(
                K3SaddleSpectrumReport(
                    label=label,
                    min_eig=float(jnp.min(eigvals)),
                    max_eig=float(jnp.max(eigvals)),
                    min_abs_eig=float(jnp.min(abs_eigvals)),
                    max_abs_eig=float(jnp.max(abs_eigvals)),
                    pos_count=int(jnp.sum(eigvals > 0.0)),
                    neg_count=int(jnp.sum(eigvals < 0.0)),
                )
            )
    return reports


def print_k3_saddle_spectrum_reports(reports: list[K3SaddleSpectrumReport]):
    print("-" * 112)
    print(
        f"{'label':<48} {'min eig':>16} {'max eig':>16} {'min |eig|':>16} {'max |eig|':>16} {'#pos':>8} {'#neg':>8}"
    )
    for report in reports:
        print(
            f"{report.label:<48} {report.min_eig:>16.3e} {report.max_eig:>16.3e} "
            f"{report.min_abs_eig:>16.3e} {report.max_abs_eig:>16.3e} {report.pos_count:>8d} {report.neg_count:>8d}"
        )


# %% Build once
SEQ, OPERATORS = ensure_built(CONFIG, rebuild=False)
print(
    f"built k=3 tensor case: ns={CONFIG.ns}, p={CONFIG.p}, map_kind={CONFIG.map_kind}, "
    f"eps={CONFIG.rotating_eps}, kappa={CONFIG.rotating_kappa}, nfp={CONFIG.rotating_nfp}"
)
K3_PLOT_DATA = build_k3_plot_data(SEQ, OPERATORS)


# %% Metric-factor CP-ALS on the quadrature-grid inverse-Jacobian tensor
METRIC_CP_RANKS = (1, 2, 3, 4, 5)
K3_METRIC_CP = {}
reports = []
metric_tensor = k3_metric_tensor(SEQ)
print("=" * 112)
print(f"k=3 metric CP-ALS diagnostics: field=1/jacobian, shape={metric_tensor.shape}")
for rank in METRIC_CP_RANKS:
    weights, factors, diagnostics = cp_als_3tensor(
        metric_tensor,
        rank,
        maxiter=100,
        tol=1e-9,
        ridge=1e-12,
    )
    diagnostics = MetricFactorFitDiagnostics(
        label="1/jacobian",
        rank=diagnostics.rank,
        relative_error=diagnostics.relative_error,
        max_abs_error=diagnostics.max_abs_error,
        n_iters=diagnostics.n_iters,
        final_delta=diagnostics.final_delta,
    )
    K3_METRIC_CP[rank] = {
        "weights": weights,
        "factors": factors,
        "diagnostics": diagnostics,
    }
    reports.append(diagnostics)
print_metric_factor_fit_reports(reports)


# %% Assemble production tensor preconditioners once per rank
TENSOR_BENCHMARK_RANKS = (1, 2, 4, 8)
TENSOR_OPERATOR_CACHE = {
    rank: assemble_tensor_mass_preconditioner(
        SEQ,
        operators=OPERATORS,
        ks=(3,),
        rank=rank,
        cp_kwargs=TENSOR_CP_KWARGS,
    )
    for rank in TENSOR_BENCHMARK_RANKS
}


# %% Assemble the extra tensor blocks needed for the k=3 saddle/HX experiment
K3_SADDLE_OPERATORS = assemble_tensor_mass_preconditioner(
    SEQ,
    operators=OPERATORS,
    ks=(0, 2, 3),
    rank=3,
    cp_kwargs=TENSOR_CP_KWARGS,
)


# %% Benchmark k=3 mass preconditioners against jacobi and tensor
K3_MASS_BENCHMARKS = {}
for dirichlet in (False,):
    reports = benchmark_k3_preconditioners(
        SEQ,
        K3_SADDLE_OPERATORS,
        dirichlet=dirichlet,
        n_rhs=8,
        seed=0,
        tensor_ranks=TENSOR_BENCHMARK_RANKS,
        tensor_operator_cache=TENSOR_OPERATOR_CACHE,
    )
    K3_MASS_BENCHMARKS[dirichlet] = reports
    print("=" * 112)
    print(f"k=3 mass preconditioner benchmark: dirichlet={dirichlet}")
    print_k3_benchmark_reports(reports)


# # %% Matrix-level trust checks for compact Kronecker models of the k=2 stiffness bulk block
# K2_STIFFNESS_COMPARE_RANKS = (1, 3)
# K2_STIFFNESS_KRONECKER_MODELS = {
#     f"kron-r{rank}": build_k2_stiffness_kronecker_preconditioner(
#         SEQ,
#         K3_SADDLE_OPERATORS,
#         dirichlet=False,
#         cp_weights=K3_METRIC_CP[rank]["weights"],
#         cp_factors=K3_METRIC_CP[rank]["factors"],
#     )
#     for rank in K2_STIFFNESS_COMPARE_RANKS
# }
# K2_STIFFNESS_BULK_EXACT = next(iter(K2_STIFFNESS_KRONECKER_MODELS.values()))["bulk_exact"]
# K2_STIFFNESS_KRONECKER_APPROXIMATIONS = {"exact-self": K2_STIFFNESS_BULK_EXACT}
# for label, model in K2_STIFFNESS_KRONECKER_MODELS.items():
#     K2_STIFFNESS_KRONECKER_APPROXIMATIONS[label] = model["bulk_model"]
# K2_STIFFNESS_KRONECKER_REPORTS = compare_k2_stiffness_kronecker_approximations(
#     K2_STIFFNESS_BULK_EXACT,
#     K2_STIFFNESS_KRONECKER_APPROXIMATIONS,
#     n_rhs=8,
#     seed=0,
# )
# print("=" * 112)
# print("k=2 stiffness Kronecker-bulk diagnostics: matrix and pseudoinverse")
# print_k2_stiffness_kronecker_approximation_reports(K2_STIFFNESS_KRONECKER_REPORTS)

# %%
K3_RICHARDSON_STEPS = (4, 8, 16, 32)
K3_RICHARDSON_POWER_ITERATIONS = 20
K3_RICHARDSON_DAMPING_SAFETY = 0.8
K3_RICHARDSON_UPPERS, K3_RICHARDSON_PARAMETERS = build_k3_richardson_upper_preconditioners(
    SEQ,
    K3_SADDLE_OPERATORS,
    dirichlet=False,
    richardson_steps=K3_RICHARDSON_STEPS,
    power_iterations=K3_RICHARDSON_POWER_ITERATIONS,
    damping_safety=K3_RICHARDSON_DAMPING_SAFETY,
)
K3_COUPLED_PRECONDITIONERS = build_k3_coupled_preconditioners(
    SEQ,
    K3_SADDLE_OPERATORS,
    dirichlet=False,
    upper_preconditioners=K3_RICHARDSON_UPPERS,
)
print("=" * 112)
print("k=3 Richardson upper-block parameters, dirichlet=False")
print_k3_richardson_parameter_reports(K3_RICHARDSON_PARAMETERS)


# %% k=3 saddle solve with Jacobi, Schur-Richardson, and coupled SPD blocks
K3_LAPLACE_BENCHMARKS_RICHARDSON = benchmark_k3_saddle_preconditioners(
    SEQ,
    K3_SADDLE_OPERATORS,
    dirichlet=False,
    extra_upper_preconditioners=K3_RICHARDSON_UPPERS,
    extra_block_preconditioners=K3_COUPLED_PRECONDITIONERS,
    n_rhs=8,
    seed=0,
)
print("=" * 112)
print("k=3 saddle benchmark: jacobi, Schur-Richardson, and coupled SPD blocks, dirichlet=False")
print_k3_saddle_benchmark_reports(K3_LAPLACE_BENCHMARKS_RICHARDSON)


# %% Matrix-free symmetry / PSD checks for the coupled saddle preconditioners
def k3_block_mass_metric_apply(x: jnp.ndarray) -> jnp.ndarray:
    n_upper = SEQ.n3
    upper = apply_mass_matrix(SEQ, K3_SADDLE_OPERATORS, x[:n_upper], 3, dirichlet=False)
    lower = apply_mass_matrix(SEQ, K3_SADDLE_OPERATORS, x[n_upper:], 2, dirichlet=False)
    return jnp.concatenate([upper, lower])


K3_COUPLED_PRECONDITIONER_EUCLIDEAN_REPORTS = diagnose_preconditioner_properties(
    K3_COUPLED_PRECONDITIONERS,
    SEQ.n3 + SEQ.n2,
    metric="euclidean",
    metric_apply=None,
    n_probes=16,
    seed=0,
)
print("=" * 112)
print("k=3 coupled preconditioner checks: Euclidean symmetry / positivity probes")
print_preconditioner_property_reports(K3_COUPLED_PRECONDITIONER_EUCLIDEAN_REPORTS)

K3_COUPLED_PRECONDITIONER_FEM_REPORTS = diagnose_preconditioner_properties(
    K3_COUPLED_PRECONDITIONERS,
    SEQ.n3 + SEQ.n2,
    metric="block-mass",
    metric_apply=k3_block_mass_metric_apply,
    n_probes=16,
    seed=0,
)
print("=" * 112)
print("k=3 coupled preconditioner checks: block FEM-mass symmetry / positivity probes")
print_preconditioner_property_reports(K3_COUPLED_PRECONDITIONER_FEM_REPORTS)


# %% Dense Schur-complement diagnostics for Jacobi and Richardson upper blocks
K3_SCHUR_MATRIX, K3_SCHUR_REPORTS = diagnose_k3_schur_upper_preconditioners(
    SEQ,
    K3_SADDLE_OPERATORS,
    dirichlet=False,
    upper_preconditioners=K3_RICHARDSON_UPPERS,
)
print("=" * 112)
print("k=3 dense Schur-complement diagnostics: jacobi vs Richardson, dirichlet=False")
print(f"schur size: {K3_SCHUR_MATRIX.shape[0]}")
print_k3_schur_approximation_reports(K3_SCHUR_REPORTS)


# %% k=2 stiffness benchmark on rhs in range(div^T): pure Richardson vs surgery-Schur + bulk Richardson
# K2_RANGE_BENCHMARKS, K2_RANGE_PARAMETERS = benchmark_k2_range_preconditioners(
#     SEQ,
#     K3_SADDLE_OPERATORS,
#     dirichlet=False,
#     richardson_steps=K3_RICHARDSON_STEPS,
#     power_iterations=K3_RICHARDSON_POWER_ITERATIONS,
#     damping_safety=K3_RICHARDSON_DAMPING_SAFETY,
#     n_rhs=8,
#     seed=0,
# )
# print("=" * 112)
# print("k=2 stiffness benchmark on rhs in range(div^T): pure Richardson vs surgery-Schur + bulk Richardson")
# print_k2_benchmark_reports(K2_RANGE_BENCHMARKS)


# %% k=2 mass benchmark: pure Richardson vs surgery-Schur + bulk Richardson
K2_MASS_BENCHMARKS, K2_MASS_PARAMETERS = benchmark_k2_mass_preconditioners(
    SEQ,
    K3_SADDLE_OPERATORS,
    dirichlet=False,
    richardson_steps=K3_RICHARDSON_STEPS,
    power_iterations=K3_RICHARDSON_POWER_ITERATIONS,
    damping_safety=K3_RICHARDSON_DAMPING_SAFETY,
    n_rhs=8,
    seed=0,
)
print("=" * 112)
print("k=2 mass benchmark: pure Richardson vs surgery-Schur + bulk Richardson")
print_k2_benchmark_reports(K2_MASS_BENCHMARKS)


# # %% Dense preconditioned saddle-spectrum diagnostics for Jacobi and Richardson upper blocks
# K3_SADDLE_SPECTRUM_REPORTS = diagnose_k3_saddle_preconditioned_spectra(
#     SEQ,
#     K3_SADDLE_OPERATORS,
#     dirichlet=False,
#     upper_preconditioners=K3_RICHARDSON_UPPERS,
#     block_preconditioners=K3_COUPLED_PRECONDITIONERS,
# )
# print("=" * 112)
# print("k=3 preconditioned saddle-spectrum diagnostics: jacobi, Richardson, and coupled, dirichlet=False")
# print_k3_saddle_spectrum_reports(K3_SADDLE_SPECTRUM_REPORTS)


# %% Optional: visualize the extracted k=3 matrix
DIRICHLET_TO_PLOT = False
A_TO_PLOT = K3_PLOT_DATA[DIRICHLET_TO_PLOT]["extracted"]

fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(jnp.log10(jnp.abs(A_TO_PLOT) + 1e-16), cmap="viridis", origin="lower")
ax.set_title(f"log10 |E3 M3 E3^T|, dirichlet={DIRICHLET_TO_PLOT}")
fig.colorbar(im, ax=ax, shrink=0.8)
plt.show()


# %% Optional: visualize the metric tensor against rank-1 and rank-3 CP fits
METRIC_COMPARE_RANKS = (1, 3)
NZ_SLICES = metric_tensor.shape[2]
SLICE_Z_INDICES = (0, NZ_SLICES // 3, (2 * NZ_SLICES) // 3)
RADIAL_QUAD_GRID = SEQ.quad.x_x[None, :, None]

display_tensor = RADIAL_QUAD_GRID * metric_tensor
display_label = "r / jacobian"
raw_reconstructions = {
    rank: reconstruct_cp_3tensor(K3_METRIC_CP[rank]["weights"], K3_METRIC_CP[rank]["factors"])
    for rank in METRIC_COMPARE_RANKS
}
reconstructions = {
    rank: RADIAL_QUAD_GRID * raw_reconstructions[rank]
    for rank in METRIC_COMPARE_RANKS
}

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
    error_slice = jnp.abs(raw_reconstructions[3][:, :, slice_z_index] - metric_tensor[:, :, slice_z_index])

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