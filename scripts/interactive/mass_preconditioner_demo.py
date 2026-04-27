# %% [markdown]
# # Mass Preconditioner Choices
#
# This interactive script assembles the production mass operators and
# compares five preconditioner choices on the same CG solve for
# `k = 0, 1, 2, 3` and both boundary-condition choices:
#
# 1. `jacobi`
# 2. `richardson-4`
# 3. `chebyshev-4`
# 4. `tensor-r3`

# %%
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import rotating_ellipse_map, toroid_map
from mrx.operators import (
    assemble_mass_operators,
    assemble_tensor_mass_preconditioner,
    dense_mass_matrix,
)
from mrx.preconditioners import MassPreconditionerSpec
from mrx.solvers import solve_singular_cg

jax.config.update("jax_enable_x64", True)


# %% Configuration
@dataclass(frozen=True)
class ExperimentConfig:
    ns: tuple[int, int, int] = (8, 16, 6)
    p: int = 3
    tol: float = 1e-9
    maxiter: int = 1000
    betti: tuple[int, int, int, int] = (1, 1, 0, 0)
    map_kind: str = "rotating_ellipse"
    torus_epsilon: float = 1.0 / 3.0
    torus_r0: float = 1.0
    rotating_eps: float = 0.33
    rotating_kappa: float = 1.2
    rotating_r0: float = 1.0
    rotating_nfp: int = 3


@dataclass
class MassBenchmarkReport:
    k: int
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
class MassSpectrumReport:
    k: int
    dirichlet: bool
    size: int
    diag_min: float
    diag_max: float
    eig_min: float
    eig_max: float
    condition_number: float


CONFIG = ExperimentConfig()
SEQ = None
OPERATORS = None
PRODUCTION_OPERATORS = None
BUILT_CONFIG = None
DENSE = False
EXACT_HYPERPARAMS = True
TENSOR_CP_KWARGS = {"tol": 1e-8, "maxiter": 200}
TENSOR_RANK = 4
POLY_STEPS = 4
POWER_ITERATIONS = 30
RICHARDSON_DAMPING_SAFETY = 0.8
CHEBYSHEV_MIN_EIG_FRACTION = 1e-4


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
    tensor_operators = assemble_tensor_mass_preconditioner(
        seq,
        operators=operators,
        ks=(0, 1, 2, 3),
        rank=TENSOR_RANK,
        cp_kwargs=TENSOR_CP_KWARGS,
    )
    return seq, operators, tensor_operators


def ensure_built(config: ExperimentConfig = CONFIG, rebuild: bool = False):
    global SEQ, OPERATORS, PRODUCTION_OPERATORS, BUILT_CONFIG
    if rebuild or SEQ is None or OPERATORS is None or BUILT_CONFIG != config:
        SEQ, OPERATORS, PRODUCTION_OPERATORS = build_case(config)
        BUILT_CONFIG = config
    return SEQ, OPERATORS, PRODUCTION_OPERATORS


def _mass_rhs_size(seq, k: int, dirichlet: bool) -> int:
    suffix = "_dbc" if dirichlet else ""
    return getattr(seq, f"n{k}{suffix}")


def benchmark_labels() -> list[str]:
    labels = [
        "jacobi",
        f"richardson-{POLY_STEPS}",
        f"chebyshev-{POLY_STEPS}",
        f"tensor-r{TENSOR_RANK}",
    ]
    if DENSE:
        labels.append("dense-direct")
    return labels


def _log10_safe(values):
    array = jnp.asarray(values, dtype=jnp.float64)
    tiny = jnp.finfo(jnp.float64).tiny
    return jnp.log10(jnp.maximum(array, tiny))


def _build_tuned_chebyshev_apply_preconditioner(
    operator_apply,
    smoother_apply,
    *,
    steps: int,
    lambda_min: float,
    lambda_max: float,
):
    if steps < 1:
        raise ValueError("Chebyshev step count must be positive")
    tiny = jnp.asarray(jnp.finfo(jnp.float64).tiny, dtype=jnp.float64)
    max_eig = jnp.maximum(jnp.asarray(lambda_max, dtype=jnp.float64), tiny)
    min_eig = jnp.clip(jnp.asarray(lambda_min, dtype=jnp.float64), tiny, max_eig)

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


def _build_tuned_richardson_apply_preconditioner(
    operator_apply,
    smoother_apply,
    *,
    steps: int,
    omega: float,
):
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


def _dense_optimal_jacobi_poly_tuning(mass_matrix):
    diagonal = jnp.diag(mass_matrix)
    diaginv = 1.0 / diagonal
    diag_sqrt_inv = jnp.sqrt(diaginv)
    preconditioned = diag_sqrt_inv[:, None] * mass_matrix * diag_sqrt_inv[None, :]
    eigvals = jnp.linalg.eigvalsh(preconditioned)
    tiny = jnp.asarray(jnp.finfo(jnp.float64).tiny, dtype=jnp.float64)
    lambda_min = jnp.maximum(jnp.min(eigvals), tiny)
    lambda_max = jnp.maximum(jnp.max(eigvals), lambda_min)
    omega = 2.0 / (lambda_min + lambda_max)
    return diaginv, omega, lambda_min, lambda_max


def _solve_mass_with_preconditioner_apply(
    seq,
    operators,
    rhs,
    *,
    k: int,
    dirichlet: bool,
    precond_apply,
):
    def mass_apply(x):
        return seq.apply_mass_matrix(x, k, dirichlet=dirichlet, operators=operators)

    x, info = solve_singular_cg(
        mass_apply,
        rhs,
        mass_matvec=mass_apply,
        precond_matvec=precond_apply,
        tol=seq.tol,
        maxiter=seq.maxiter,
    )
    residual = mass_apply(x) - rhs
    rhs_norm = jnp.where(jnp.linalg.norm(rhs) > 0, jnp.linalg.norm(rhs), 1.0)
    return x, info, jnp.linalg.norm(residual) / rhs_norm


def benchmark_mass_preconditioners(
    seq,
    operators,
    *,
    k: int,
    dirichlet: bool,
    n_rhs: int = 8,
    seed: int = 0,
) -> list[MassBenchmarkReport]:
    rhs_size = _mass_rhs_size(seq, k, dirichlet)
    rhs_batch = jax.random.normal(
        jax.random.PRNGKey(seed + 100 * int(dirichlet) + 1000 * k),
        (n_rhs, rhs_size),
        dtype=jnp.float64,
    )

    specs = {
        "jacobi": MassPreconditionerSpec(kind="jacobi"),
        f"richardson-{POLY_STEPS}": MassPreconditionerSpec(
            kind="richardson",
            steps=POLY_STEPS,
            power_iterations=POWER_ITERATIONS,
            damping_safety=RICHARDSON_DAMPING_SAFETY,
            smoother=MassPreconditionerSpec(kind="jacobi"),
        ),
        f"chebyshev-{POLY_STEPS}": MassPreconditionerSpec(
            kind="chebyshev",
            steps=POLY_STEPS,
            power_iterations=POWER_ITERATIONS,
            min_eig_fraction=CHEBYSHEV_MIN_EIG_FRACTION,
            smoother=MassPreconditionerSpec(kind="jacobi"),
        ),
        f"tensor-r{TENSOR_RANK}": MassPreconditionerSpec(kind="tensor"),
    }
    dense_matrix = None
    tuned_diaginv = None
    tuned_omega = None
    tuned_lambda_min = None
    tuned_lambda_max = None
    if EXACT_HYPERPARAMS or DENSE:
        dense_matrix = jnp.asarray(dense_mass_matrix(seq, operators, k, dirichlet=dirichlet))
    if EXACT_HYPERPARAMS:
        tuned_diaginv, tuned_omega, tuned_lambda_min, tuned_lambda_max = (
            _dense_optimal_jacobi_poly_tuning(dense_matrix)
        )

    reports = []
    for label, preconditioner in specs.items():
        if EXACT_HYPERPARAMS and preconditioner.kind == "richardson":
            def operator_apply(x):
                return seq.apply_mass_matrix(
                    x, k, dirichlet=dirichlet, operators=operators
                )

            def smoother_apply(x, diaginv=tuned_diaginv):
                return diaginv * x

            tuned_precond = _build_tuned_richardson_apply_preconditioner(
                operator_apply,
                smoother_apply,
                steps=preconditioner.steps,
                omega=tuned_omega,
            )

            @jax.jit
            def solve(rhs):
                return _solve_mass_with_preconditioner_apply(
                    seq,
                    operators,
                    rhs,
                    k=k,
                    dirichlet=dirichlet,
                    precond_apply=tuned_precond,
                )
        elif EXACT_HYPERPARAMS and preconditioner.kind == "chebyshev":
            def operator_apply(x):
                return seq.apply_mass_matrix(
                    x, k, dirichlet=dirichlet, operators=operators
                )

            def smoother_apply(x, diaginv=tuned_diaginv):
                return diaginv * x

            tuned_precond = _build_tuned_chebyshev_apply_preconditioner(
                operator_apply,
                smoother_apply,
                steps=preconditioner.steps,
                lambda_min=tuned_lambda_min,
                lambda_max=tuned_lambda_max,
            )

            @jax.jit
            def solve(rhs):
                return _solve_mass_with_preconditioner_apply(
                    seq,
                    operators,
                    rhs,
                    k=k,
                    dirichlet=dirichlet,
                    precond_apply=tuned_precond,
                )
        else:
            @jax.jit
            def solve(rhs):
                x, info = seq.apply_inverse_mass_matrix(
                    rhs,
                    k,
                    dirichlet=dirichlet,
                    operators=operators,
                    preconditioner=preconditioner,
                    return_info=True,
                )
                residual = seq.apply_mass_matrix(
                    x, k, dirichlet=dirichlet, operators=operators) - rhs
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
            MassBenchmarkReport(
                k=k,
                label=label,
                dirichlet=dirichlet,
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

    if DENSE:
        @jax.jit
        def solve_dense(rhs):
            x = jnp.linalg.solve(dense_matrix, rhs)
            residual = dense_matrix @ x - rhs
            rhs_norm = jnp.where(jnp.linalg.norm(rhs) > 0, jnp.linalg.norm(rhs), 1.0)
            return x, jnp.asarray(0, dtype=jnp.int32), jnp.linalg.norm(residual) / rhs_norm

        x0, _, _ = solve_dense(rhs_batch[0])
        jax.block_until_ready(x0)

        iterations = []
        times_ms = []
        residuals = []
        for rhs in rhs_batch:
            t0 = time.perf_counter()
            x, info, residual = solve_dense(rhs)
            jax.block_until_ready(x)
            times_ms.append((time.perf_counter() - t0) * 1e3)
            iterations.append(int(jnp.abs(info)))
            residuals.append(float(residual))

        iterations_array = jnp.asarray(iterations)
        times_ms_array = jnp.asarray(times_ms)
        residuals_array = jnp.asarray(residuals)
        reports.append(
            MassBenchmarkReport(
                k=k,
                label="dense-direct",
                dirichlet=dirichlet,
                n_rhs=n_rhs,
                avg_iters=float(1),
                std_iters=float(0),
                max_iters=int(1),
                avg_time_ms=float(jnp.mean(times_ms_array)),
                std_time_ms=float(jnp.std(times_ms_array)),
                max_time_ms=float(jnp.max(times_ms_array)),
                avg_relative_residual=float(jnp.mean(residuals_array)),
                std_relative_residual=float(jnp.std(residuals_array)),
                max_relative_residual=float(jnp.max(residuals_array)),
            )
        )
    return reports


def print_mass_benchmark_reports(reports: list[MassBenchmarkReport]):
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


def summarize_mass_matrix(seq, operators, *, k: int, dirichlet: bool) -> MassSpectrumReport:
    mass_matrix = jnp.asarray(dense_mass_matrix(seq, operators, k, dirichlet=dirichlet))
    diagonal = jnp.diag(mass_matrix)
    eigvals = jnp.linalg.eigvalsh(mass_matrix)
    eig_min = float(jnp.min(eigvals))
    eig_max = float(jnp.max(eigvals))
    return MassSpectrumReport(
        k=k,
        dirichlet=dirichlet,
        size=int(mass_matrix.shape[0]),
        diag_min=float(jnp.min(diagonal)),
        diag_max=float(jnp.max(diagonal)),
        eig_min=eig_min,
        eig_max=eig_max,
        condition_number=eig_max / eig_min,
    )


def print_mass_spectrum_report(report: MassSpectrumReport):
    print("=" * 112)
    print(
        f"k={report.k} mass matrix summary: dirichlet={report.dirichlet}, shape=({report.size}, {report.size}) \n"
        f"diag min={report.diag_min:.3e}, diag max={report.diag_max:.3e} \n"
        f"eigval min={report.eig_min:.3e}, eigval max={report.eig_max:.3e} \n"
        f"max/min eigval={report.condition_number:.3e}"
    )


def plot_mass_benchmark_reports(
    benchmarks: dict[tuple[int, bool], list[MassBenchmarkReport]],
    *,
    show: bool = True,
):
    labels = benchmark_labels()
    x = jnp.arange(len(labels))
    color_map = {
        "jacobi": "#1f77b4",
        f"richardson-{POLY_STEPS}": "#ff7f0e",
        f"chebyshev-{POLY_STEPS}": "#2ca02c",
        f"tensor-r{TENSOR_RANK}": "#d62728",
        "dense-direct": "#9467bd",
    }
    colors = [color_map[label] for label in labels]
    metrics = [
        ("avg_iters", "std_iters", "Average Iterations", True),
        ("avg_time_ms", "std_time_ms", "Average Time [ms]", True),
        ("avg_relative_residual", "std_relative_residual", "Average Relative Residual", False),
    ]

    fig, axes = plt.subplots(
        len(metrics),
        4,
        figsize=(18, 10),
        sharex=True,
        sharey="row",
        constrained_layout=True,
    )

    for col, k in enumerate((0, 1, 2, 3)):
        for row, (metric_name, std_name, metric_label, log_scale) in enumerate(metrics):
            ax = axes[row, col]
            free_reports = {report.label: report for report in benchmarks[(k, False)]}
            dbc_reports = {report.label: report for report in benchmarks[(k, True)]}
            free_values = [getattr(free_reports[label], metric_name) for label in labels]
            dbc_values = [getattr(dbc_reports[label], metric_name) for label in labels]
            free_errors = None if std_name is None else [
                getattr(free_reports[label], std_name) for label in labels
            ]
            dbc_errors = None if std_name is None else [
                getattr(dbc_reports[label], std_name) for label in labels
            ]
            width = 0.38
            ax.bar(
                x - width / 2,
                free_values,
                width=width,
                color=colors,
                alpha=0.9,
                yerr=free_errors,
                error_kw={"elinewidth": 1.0, "capsize": 3, "capthick": 1.0},
            )
            ax.bar(
                x + width / 2,
                dbc_values,
                width=width,
                color=colors,
                alpha=0.45,
                yerr=dbc_errors,
                error_kw={"elinewidth": 1.0, "capsize": 3, "capthick": 1.0},
            )
            if log_scale:
                ax.set_yscale("log", base=10)
            ax.set_title(f"k={k} {metric_label}")
            ax.grid(axis="y", alpha=0.25)
            if col == 0:
                ax.set_ylabel(metric_label)
            if row == len(metrics) - 1:
                ax.set_xticks(x, labels, rotation=25, ha="right")
            else:
                ax.set_xticks(x, [])

    legend_handles = [
        Patch(facecolor="#595959", alpha=0.9, label="dirichlet=False"),
        Patch(facecolor="#595959", alpha=0.45, label="dirichlet=True"),
    ]
    axes[0, 0].legend(handles=legend_handles, loc="upper right", frameon=False)
    if show:
        plt.show()
    return fig


def plot_mass_benchmark_heatmaps(
    benchmarks: dict[tuple[int, bool], list[MassBenchmarkReport]],
    *,
    show: bool = True,
):
    labels = benchmark_labels()
    case_labels = [f"k={k}, dbc={dirichlet}" for k in (0, 1, 2, 3) for dirichlet in (False, True)]
    iteration_matrix = jnp.asarray([
        [
            next(report.avg_iters for report in benchmarks[(k, dirichlet)] if report.label == label)
            for k in (0, 1, 2, 3)
            for dirichlet in (False, True)
        ]
        for label in labels
    ])
    time_matrix = jnp.asarray([
        [
            next(report.avg_time_ms for report in benchmarks[(k, dirichlet)] if report.label == label)
            for k in (0, 1, 2, 3)
            for dirichlet in (False, True)
        ]
        for label in labels
    ])

    fig, axes = plt.subplots(1, 2, figsize=(18, 4.8), constrained_layout=True)
    for ax, matrix, title, cmap in (
        (axes[0], iteration_matrix, "Average Iterations", "viridis"),
        (axes[1], time_matrix, "Average Time [ms]", "magma"),
    ):
        log_matrix = _log10_safe(matrix)
        image = ax.imshow(log_matrix, aspect="auto", cmap=cmap)
        ax.set_title(f"{title} ($\\log_{{10}}$ scale)")
        ax.set_xticks(range(len(case_labels)), case_labels, rotation=35, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        colorbar = fig.colorbar(image, ax=ax, shrink=0.9)
        colorbar.set_label(r"$\log_{10}(\mathrm{value})$")

    if show:
        plt.show()
    return fig


def plot_mass_spectrum_reports(
    spectra: dict[tuple[int, bool], MassSpectrumReport],
    *,
    show: bool = True,
):
    metrics = [
        ("eig_min", r"$\lambda_{\min}(M_k)$", True),
        ("eig_max", r"$\lambda_{\max}(M_k)$", True),
        ("condition_number", r"$\kappa(M_k)$", True),
    ]
    x = jnp.arange(4)
    width = 0.36

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for ax, (metric_name, title, log_scale) in zip(axes, metrics, strict=True):
        free_values = [getattr(spectra[(k, False)], metric_name) for k in (0, 1, 2, 3)]
        dbc_values = [getattr(spectra[(k, True)], metric_name) for k in (0, 1, 2, 3)]
        ax.bar(x - width / 2, free_values, width=width, color="#1f77b4", alpha=0.9, label="dirichlet=False")
        ax.bar(x + width / 2, dbc_values, width=width, color="#ff7f0e", alpha=0.75, label="dirichlet=True")
        if log_scale:
            ax.set_yscale("log", base=10)
        ax.set_title(title)
        ax.set_xticks(x, ["k=0", "k=1", "k=2", "k=3"])
        ax.grid(axis="y", alpha=0.25)

    axes[0].legend(frameon=False)
    if show:
        plt.show()
    return fig


def plot_mass_spectrum_heatmap(
    spectra: dict[tuple[int, bool], MassSpectrumReport],
    *,
    show: bool = True,
):
    case_labels = [f"k={k}, dbc={dirichlet}" for k in (0, 1, 2, 3) for dirichlet in (False, True)]
    metric_names = ["eig_min", "eig_max", "condition_number"]
    metric_labels = [r"$\lambda_{\min}$", r"$\lambda_{\max}$", r"$\kappa$"]
    matrix = jnp.asarray([
        [getattr(spectra[(k, dirichlet)], metric_name) for k in (0, 1, 2, 3) for dirichlet in (False, True)]
        for metric_name in metric_names
    ])

    fig, ax = plt.subplots(1, 1, figsize=(12, 3.8), constrained_layout=True)
    image = ax.imshow(_log10_safe(matrix), aspect="auto", cmap="cividis")
    ax.set_title("Mass Matrix Spectrum Metrics ($\\log_{10}$ scale)")
    ax.set_xticks(range(len(case_labels)), case_labels, rotation=35, ha="right")
    ax.set_yticks(range(len(metric_labels)), metric_labels)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.9)
    colorbar.set_label(r"$\log_{10}(\mathrm{value})$")
    if show:
        plt.show()
    return fig


def _flatten_benchmarks(
    benchmarks: dict[tuple[int, bool], list[MassBenchmarkReport]],
) -> list[dict]:
    rows = []
    for reports in benchmarks.values():
        rows.extend(asdict(report) for report in reports)
    return rows


def _flatten_spectra(
    spectra: dict[tuple[int, bool], MassSpectrumReport],
) -> list[dict]:
    return [asdict(report) for report in spectra.values()]


def save_mass_benchmark_artifacts(
    benchmarks: dict[tuple[int, bool], list[MassBenchmarkReport]],
    spectra: dict[tuple[int, bool], MassSpectrumReport],
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs") / "interactive" / f"k0_mass_preconditioner_choices_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "config": asdict(CONFIG),
        "dense": DENSE,
        "exact_hyperparams": EXACT_HYPERPARAMS,
        "polynomial_tuning": (
            "exact_dense_jacobi_preconditioned_spectrum"
            if EXACT_HYPERPARAMS
            else "heuristic_power_iteration"
        ),
        "tensor_rank": TENSOR_RANK,
        "poly_steps": POLY_STEPS,
        "power_iterations": POWER_ITERATIONS,
        "richardson_damping_safety": RICHARDSON_DAMPING_SAFETY,
        "chebyshev_min_eig_fraction": CHEBYSHEV_MIN_EIG_FRACTION,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (output_dir / "benchmarks.json").write_text(
        json.dumps(_flatten_benchmarks(benchmarks), indent=2)
    )
    if DENSE:
        (output_dir / "spectra.json").write_text(
            json.dumps(_flatten_spectra(spectra), indent=2)
        )

    figures = [
        ("benchmark_bars.png", plot_mass_benchmark_reports(benchmarks, show=False)),
        ("benchmark_heatmaps.png", plot_mass_benchmark_heatmaps(benchmarks, show=False)),
    ]
    if DENSE:
        figures.extend([
            ("spectrum_bars.png", plot_mass_spectrum_reports(spectra, show=False)),
            ("spectrum_heatmap.png", plot_mass_spectrum_heatmap(spectra, show=False)),
        ])

    for filename, fig in figures:
        fig.savefig(output_dir / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)

    return output_dir


# %% Build once
SEQ, OPERATORS, PRODUCTION_OPERATORS = ensure_built(CONFIG, rebuild=False)
print(
    f"built mass benchmark case: ns={CONFIG.ns}, p={CONFIG.p}, map_kind={CONFIG.map_kind}, "
    f"eps={CONFIG.rotating_eps}, kappa={CONFIG.rotating_kappa}, nfp={CONFIG.rotating_nfp}"
)
print(f"dense diagnostics/direct baseline enabled: {DENSE}")
print(f"exact polynomial hyperparameters enabled: {EXACT_HYPERPARAMS}")


# %% Inspect the assembled extracted mass matrices
ALL_MASS_SPECTRA = {}
if DENSE:
    for k in (0, 1, 2, 3):
        for dirichlet in (False, True):
            spectrum = summarize_mass_matrix(SEQ, OPERATORS, k=k, dirichlet=dirichlet)
            ALL_MASS_SPECTRA[(k, dirichlet)] = spectrum
            print_mass_spectrum_report(spectrum)


# %% Compare the five production-facing preconditioner choices for all k and BC combinations
ALL_BENCHMARKS = {}
for k in (0, 1, 2, 3):
    for dirichlet in (False, True):
        reports = benchmark_mass_preconditioners(
            SEQ,
            PRODUCTION_OPERATORS,
            k=k,
            dirichlet=dirichlet,
            n_rhs=8,
            seed=0,
        )
        ALL_BENCHMARKS[(k, dirichlet)] = reports
        print("=" * 112)
        print(
            f"k={k} mass preconditioner comparison: dirichlet={dirichlet}, "
            f"tensor rank={TENSOR_RANK}, richardson/chebyshev steps={POLY_STEPS}"
        )
        print_mass_benchmark_reports(reports)


# %% Visualize the benchmark summaries
plot_mass_benchmark_reports(ALL_BENCHMARKS)
plot_mass_benchmark_heatmaps(ALL_BENCHMARKS)
if DENSE:
    plot_mass_spectrum_reports(ALL_MASS_SPECTRA)
    plot_mass_spectrum_heatmap(ALL_MASS_SPECTRA)


# %% Store results and plots
OUTPUT_DIR = save_mass_benchmark_artifacts(ALL_BENCHMARKS, ALL_MASS_SPECTRA)
print(f"saved benchmark artifacts to: {OUTPUT_DIR}")
# %%
