# %% [markdown]
# # k=0 / k=1 / k=2 / k=3 Hodge-Laplacian Preconditioner Choices
#
# This interactive script assembles the production k=0 scalar solve plus the
# structured saddle-point solves for k=1, k=2, and k=3. It compares the four
# scalar preconditioner choices for k=0 and the structured saddle-point choices
# for k=1 / k=2 / k=3, for both the singular Laplacian and a lightly shifted
# Laplacian:
#
# 1. `jacobi`
# 2. `richardson-4`
# 3. `chebyshev-4`
# 4. `tensor`

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

from mrx.nullspace import _set_null, get_nullspace
from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import rotating_ellipse_map, toroid_map
from mrx.operators import (
    apply_derivative_matrix,
    apply_mass_matrix,
    apply_stiffness,
    assemble_tensor_hodge_preconditioner,
    assemble_tensor_mass_preconditioner,
    assemble_hodge_operators,
    assemble_incidence_operators,
    assemble_mass_operators,
    dense_hodge_laplacian,
)
from mrx.preconditioners import (
    MassPreconditionerSpec,
    SaddlePointPreconditionerSpec,
    SchurPreconditionerSpec,
)

jax.config.update("jax_enable_x64", True)


# %% Configuration
@dataclass(frozen=True)
class ExperimentConfig:
    ns: tuple[int, int, int] = (8, 12, 6)
    p: int = 2
    tol: float = 1e-9
    maxiter: int = 2000
    betti: tuple[int, int, int, int] = (1, 1, 0, 0)
    map_kind: str = "rotating_ellipse"
    torus_epsilon: float = 1.0 / 3.0
    torus_r0: float = 1.0
    rotating_eps: float = 0.2
    rotating_kappa: float = 1.2
    rotating_r0: float = 1.0
    rotating_nfp: int = 3


@dataclass
class K0LaplacianBenchmarkReport:
    label: str
    eps: float
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
class K0LaplacianSpectrumReport:
    dirichlet: bool
    size: int
    zero_count: int
    min_nonzero_eig: float
    max_eig: float
    nonzero_condition_number: float


@dataclass
class SaddleBenchmarkReport:
    k: int
    label: str
    eps: float
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
class DiffusionBenchmarkReport:
    k: int
    label: str
    eps: float
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

# %%
CONFIG = ExperimentConfig()
SEQ = None
OPERATORS = None
BUILT_CONFIG = None
DENSE = False
eps = 0.0
eps_list = (eps,)
NULLSPACE_ITERATION_EPS = eps if eps > 0.0 else 1e-3
K3_TENSOR_RANK = 3
TENSOR_CP_KWARGS = {"tol": 1e-8, "maxiter": 500}
POLY_STEPS = 4
POWER_ITERATIONS = 30
RICHARDSON_DAMPING_SAFETY = 0.8
CHEBYSHEV_MIN_EIG_FRACTION = 1e-4
DIFFUSION_EPS = 1e-4


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
    operators = assemble_tensor_hodge_preconditioner(seq, operators=operators)
    operators = assemble_tensor_mass_preconditioner(
        seq,
        operators=operators,
        ks=(0, 1, 2, 3),
        rank=K3_TENSOR_RANK,
        cp_kwargs=TENSOR_CP_KWARGS,
    )
    operators = assemble_incidence_operators(seq, operators=operators, ks=(0, 1, 2))
    operators = assemble_hodge_operators(seq, seq.geometry, operators=operators, ks=(0,))
    operators = seq.set_operators(operators, sync_legacy=False)
    k0_nullspace_vectors, k0_nullspace_iters = seq._find_nullspace_vectors(
        0,
        1,
        NULLSPACE_ITERATION_EPS,
        dirichlet=False,
    )
    operators = _set_null(operators, 0, False, k0_nullspace_vectors)
    operators = seq.set_operators(operators, sync_legacy=False)
    k1_nullspace_vectors, k1_nullspace_iters = seq._find_nullspace_vectors(
        1,
        1,
        NULLSPACE_ITERATION_EPS,
        dirichlet=False,
    )
    operators = _set_null(operators, 1, False, k1_nullspace_vectors)
    operators = seq.set_operators(operators, sync_legacy=False)
    k2_nullspace_vectors, k2_nullspace_iters = seq._find_nullspace_vectors(
        2,
        1,
        NULLSPACE_ITERATION_EPS,
        dirichlet=True,
    )
    operators = _set_null(operators, 2, True, k2_nullspace_vectors)
    operators = seq.set_operators(operators, sync_legacy=False)
    k3_nullspace_vectors, k3_nullspace_iters = seq._find_nullspace_vectors(
        3,
        1,
        NULLSPACE_ITERATION_EPS,
        dirichlet=True,
    )
    operators = _set_null(operators, 3, True, k3_nullspace_vectors)
    operators = seq.set_operators(operators, sync_legacy=False)
    nullspace_info = {
        (0, False): k0_nullspace_iters,
        (1, False): k1_nullspace_iters,
        (2, True): k2_nullspace_iters,
        (3, True): k3_nullspace_iters,
    }
    _print_nullspace_iteration_info(nullspace_info, eps=NULLSPACE_ITERATION_EPS)
    operators = seq._require_operators()
    return seq, operators


def ensure_built(config: ExperimentConfig = CONFIG, rebuild: bool = False):
    global SEQ, OPERATORS, BUILT_CONFIG
    if rebuild or SEQ is None or OPERATORS is None or BUILT_CONFIG != config:
        SEQ, OPERATORS = build_case(config)
        BUILT_CONFIG = config
    return SEQ, OPERATORS


def benchmark_labels() -> list[str]:
    labels = list(_benchmark_mass_preconditioner_specs())
    if DENSE:
        labels.append("dense-pinv")
    return labels


def _benchmark_mass_preconditioner_specs() -> dict[str, MassPreconditionerSpec]:
    labels = [
        ("jacobi", MassPreconditionerSpec(kind="jacobi")),
        (
            f"richardson-{POLY_STEPS}",
            MassPreconditionerSpec(
                kind="richardson",
                steps=POLY_STEPS,
                power_iterations=POWER_ITERATIONS,
                damping_safety=RICHARDSON_DAMPING_SAFETY,
                smoother=MassPreconditionerSpec(kind="jacobi"),
            ),
        ),
        (
            f"chebyshev-{POLY_STEPS}",
            MassPreconditionerSpec(
                kind="chebyshev",
                steps=POLY_STEPS,
                power_iterations=POWER_ITERATIONS,
                min_eig_fraction=CHEBYSHEV_MIN_EIG_FRACTION,
                smoother=MassPreconditionerSpec(kind="jacobi"),
            ),
        ),
        ("tensor", MassPreconditionerSpec(kind="tensor")),
    ]
    return dict(labels)


def _diffusion_preconditioner_specs() -> dict[str, MassPreconditionerSpec]:
    return {
        "jacobi": MassPreconditionerSpec(kind="jacobi"),
        f"chebyshev-{POLY_STEPS}": MassPreconditionerSpec(
            kind="chebyshev",
            steps=POLY_STEPS,
            power_iterations=POWER_ITERATIONS,
            min_eig_fraction=CHEBYSHEV_MIN_EIG_FRACTION,
            smoother=MassPreconditionerSpec(kind="jacobi"),
        ),
        "tensor": MassPreconditionerSpec(kind="tensor"),
    }


def _saddle_preconditioner_specs() -> dict[str, SaddlePointPreconditionerSpec]:
    mass_spec = MassPreconditionerSpec(kind="tensor")
    outer_specs = {
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
    }
    return {
        f"mass=tensor / outer={outer_label}": SaddlePointPreconditionerSpec(
            mass=mass_spec,
            schur=SchurPreconditionerSpec(
                inner=mass_spec,
                outer=outer_spec,
            ),
            coupled=False,
        )
        for outer_label, outer_spec in outer_specs.items()
    }


def _log10_safe(values):
    array = jnp.asarray(values, dtype=jnp.float64)
    tiny = jnp.finfo(jnp.float64).tiny
    return jnp.log10(jnp.maximum(array, tiny))


def _print_nullspace_iteration_info(info: dict[tuple[int, bool], list[tuple[int, float]]], *, eps: float):
    print(f"nullspace inverse-iteration info (eps={eps:.1e}):")
    for (k, dirichlet), entries in sorted(info.items()):
        if not entries:
            print(f"  k={k}, dirichlet={dirichlet}: []")
            continue
        summary = ", ".join(
            f"(iters={n_iters}, residual={residual:.3e})"
            for n_iters, residual in entries
        )
        print(f"  k={k}, dirichlet={dirichlet}: [{summary}]")


def _k0_nullspace_vector(seq, operators, dirichlet: bool):
    if dirichlet:
        return None
    vector = jnp.ones((seq.n0,), dtype=jnp.float64)
    mass_vector = apply_mass_matrix(seq, operators, vector, 0, dirichlet=dirichlet)
    norm = jnp.sqrt(jnp.abs(jnp.dot(vector, mass_vector)))
    return vector / jnp.where(norm > 0, norm, 1.0)


def _project_rhs_to_range(seq, operators, rhs, dirichlet: bool):
    null_vector = _k0_nullspace_vector(seq, operators, dirichlet)
    if null_vector is None:
        return rhs
    return rhs - jnp.dot(null_vector, rhs) * apply_mass_matrix(
        seq, operators, null_vector, 0, dirichlet=dirichlet)


def _project_saddle_rhs_to_range(seq, operators, rhs, *, k: int, dirichlet: bool):
    nullspace = get_nullspace(operators, k, dirichlet)
    if nullspace.shape[0] == 0:
        return rhs
    projected = rhs
    for vector in nullspace:
        projected = projected - jnp.dot(vector, projected) * apply_mass_matrix(
            seq, operators, vector, k, dirichlet=dirichlet
        )
    return projected


def _apply_saddle_hodge_operator(seq, operators, x, *, k: int, eps: float, dirichlet: bool):
    stiffness = apply_stiffness(seq, operators, x, k, dirichlet=dirichlet)
    d_t_x = apply_derivative_matrix(
        seq,
        operators,
        x,
        k - 1,
        dirichlet_in=dirichlet,
        dirichlet_out=dirichlet,
        transpose=True,
    )
    sigma = seq.apply_inverse_mass_matrix(d_t_x, k - 1, dirichlet=dirichlet)
    schur = apply_derivative_matrix(
        seq,
        operators,
        sigma,
        k - 1,
        dirichlet_in=dirichlet,
        dirichlet_out=dirichlet,
    )
    operator = stiffness + schur
    if eps > 0.0:
        operator = operator + eps * apply_mass_matrix(
            seq, operators, x, k, dirichlet=dirichlet
        )
    return operator


def _dense_k0_hodge_inverse(seq, operators, dirichlet: bool, *, eps: float = 0.0, rtol: float = 1e-12):
    matrix = jnp.asarray(dense_hodge_laplacian(seq, operators, 0, dirichlet=dirichlet))
    eigvals, eigvecs = jnp.linalg.eigh(matrix)
    scale = jnp.max(jnp.abs(eigvals))
    threshold = rtol * jnp.where(scale > 0, scale, 1.0)
    shifted_eigvals = eigvals + eps
    if eps == 0.0:
        inv_eigvals = jnp.where(jnp.abs(eigvals) > threshold, 1.0 / eigvals, 0.0)
    else:
        inv_eigvals = 1.0 / shifted_eigvals
    inverse = (eigvecs * inv_eigvals) @ eigvecs.T
    return matrix, inverse, eigvals


def benchmark_k0_laplacian_preconditioners(
    seq,
    operators,
    *,
    eps: float,
    dirichlet: bool,
    n_rhs: int = 8,
    seed: int = 0,
) -> list[K0LaplacianBenchmarkReport]:
    rhs_size = seq.n0_dbc if dirichlet else seq.n0
    rhs_batch = jax.random.normal(
        jax.random.PRNGKey(seed + 100 * int(dirichlet)),
        (n_rhs, rhs_size),
        dtype=jnp.float64,
    )
    if eps == 0.0:
        rhs_batch = jax.vmap(
            lambda rhs: _project_rhs_to_range(seq, operators, rhs, dirichlet)
        )(rhs_batch)

    specs = _benchmark_mass_preconditioner_specs()

    dense_matrix = None
    dense_pinv = None
    if DENSE:
        dense_matrix, dense_pinv, _ = _dense_k0_hodge_inverse(
            seq, operators, dirichlet, eps=eps)

    reports = []
    for label, preconditioner in specs.items():
        @jax.jit
        def solve(rhs):
            rhs_use = rhs if eps > 0.0 else _project_rhs_to_range(
                seq, operators, rhs, dirichlet)
            if eps == 0.0:
                x, info = seq.apply_inverse_hodge_laplacian(
                    rhs_use,
                    0,
                    dirichlet=dirichlet,
                    operators=operators,
                    preconditioner=preconditioner,
                    return_info=True,
                )
            else:
                x, info = seq.apply_inverse_shifted_hodge_laplacian(
                    rhs_use,
                    0,
                    eps,
                    dirichlet=dirichlet,
                    operators=operators,
                    preconditioner=preconditioner,
                    return_info=True,
                )
            residual = apply_stiffness(seq, operators, x, 0, dirichlet=dirichlet)
            if eps > 0.0:
                residual = residual + eps * apply_mass_matrix(
                    seq, operators, x, 0, dirichlet=dirichlet)
            residual = residual - rhs_use
            rhs_norm = jnp.where(jnp.linalg.norm(rhs_use) > 0, jnp.linalg.norm(rhs_use), 1.0)
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
            K0LaplacianBenchmarkReport(
                label=label,
                eps=eps,
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
            rhs_use = rhs if eps > 0.0 else _project_rhs_to_range(
                seq, operators, rhs, dirichlet)
            x = dense_pinv @ rhs_use
            residual = dense_matrix @ x + eps * apply_mass_matrix(
                seq, operators, x, 0, dirichlet=dirichlet) - rhs_use
            rhs_norm = jnp.where(jnp.linalg.norm(rhs_use) > 0, jnp.linalg.norm(rhs_use), 1.0)
            return x, jnp.asarray(1, dtype=jnp.int32), jnp.linalg.norm(residual) / rhs_norm

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
            K0LaplacianBenchmarkReport(
                label="dense-pinv",
                eps=eps,
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

    return reports


def benchmark_saddle_laplacian_preconditioners(
    seq,
    operators,
    *,
    k: int,
    eps: float,
    dirichlet: bool,
    n_rhs: int = 8,
    seed: int = 0,
) -> list[SaddleBenchmarkReport]:
    suffix = "_dbc" if dirichlet else ""
    rhs_size = getattr(seq, f"n{k}{suffix}")
    rhs_batch = jax.random.normal(
        jax.random.PRNGKey(seed + 700 + 1000 * k + 100 * int(dirichlet)),
        (n_rhs, rhs_size),
        dtype=jnp.float64,
    )
    if eps == 0.0:
        rhs_batch = jax.vmap(
            lambda rhs: _project_saddle_rhs_to_range(
                seq, operators, rhs, k=k, dirichlet=dirichlet
            )
        )(rhs_batch)
    specs = _saddle_preconditioner_specs()

    reports = []
    for label, preconditioner in specs.items():
        @jax.jit
        def solve(rhs):
            rhs_use = rhs if eps > 0.0 else _project_saddle_rhs_to_range(
                seq, operators, rhs, k=k, dirichlet=dirichlet
            )
            if eps == 0.0:
                x, info = seq.apply_inverse_hodge_laplacian(
                    rhs_use,
                    k,
                    dirichlet=dirichlet,
                    operators=operators,
                    preconditioner=preconditioner,
                    return_info=True,
                )
            else:
                x, info = seq.apply_inverse_shifted_hodge_laplacian(
                    rhs_use,
                    k,
                    eps,
                    dirichlet=dirichlet,
                    operators=operators,
                    preconditioner=preconditioner,
                    return_info=True,
                )
            return x, info

        def relative_residual(x, rhs):
            rhs_use = rhs if eps > 0.0 else _project_saddle_rhs_to_range(
                seq, operators, rhs, k=k, dirichlet=dirichlet
            )
            residual = _apply_saddle_hodge_operator(
                seq, operators, x, k=k, eps=eps, dirichlet=dirichlet
            ) - rhs_use
            rhs_norm = jnp.where(
                jnp.linalg.norm(rhs_use) > 0,
                jnp.linalg.norm(rhs_use),
                1.0,
            )
            return jnp.linalg.norm(residual) / rhs_norm

        x0, _ = solve(rhs_batch[0])
        jax.block_until_ready(x0)
        residual0 = relative_residual(x0, rhs_batch[0])
        jax.block_until_ready(residual0)

        iterations = []
        times_ms = []
        residuals = []
        for rhs in rhs_batch:
            t0 = time.perf_counter()
            x, info = solve(rhs)
            jax.block_until_ready(x)
            times_ms.append((time.perf_counter() - t0) * 1e3)
            iterations.append(int(jnp.abs(info)))
            residuals.append(float(relative_residual(x, rhs)))

        iterations_array = jnp.asarray(iterations)
        times_ms_array = jnp.asarray(times_ms)
        residuals_array = jnp.asarray(residuals)
        reports.append(
            SaddleBenchmarkReport(
                k=k,
                label=label,
                eps=eps,
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

    return reports


def benchmark_diffusion_preconditioners(
    seq,
    operators,
    *,
    k: int,
    eps: float,
    dirichlet: bool,
    n_rhs: int = 8,
    seed: int = 0,
) -> list[DiffusionBenchmarkReport]:
    suffix = "_dbc" if dirichlet else ""
    rhs_size = getattr(seq, f"n{k}{suffix}")
    rhs_batch = jax.random.normal(
        jax.random.PRNGKey(seed + 1300 + 1000 * k + 100 * int(dirichlet)),
        (n_rhs, rhs_size),
        dtype=jnp.float64,
    )
    specs = _diffusion_preconditioner_specs()

    reports = []
    for label, preconditioner in specs.items():
        @jax.jit
        def solve(rhs):
            x, info = seq.apply_inverse_mass_plus_eps_laplace_matrix(
                rhs,
                k,
                eps,
                dirichlet=dirichlet,
                operators=operators,
                preconditioner=preconditioner,
                return_info=True,
            )
            residual = seq.apply_mass_plus_eps_laplace_matrix(
                x,
                k,
                eps,
                dirichlet=dirichlet,
                operators=operators,
            ) - rhs
            rhs_norm = jnp.where(jnp.linalg.norm(rhs) > 0, jnp.linalg.norm(rhs), 1.0)
            return x, info, jnp.linalg.norm(residual) / rhs_norm

        x0, _, residual0 = solve(rhs_batch[0])
        jax.block_until_ready(x0)
        jax.block_until_ready(residual0)

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
            DiffusionBenchmarkReport(
                k=k,
                label=label,
                eps=eps,
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

    return reports


def print_k0_laplacian_benchmark_reports(reports: list[K0LaplacianBenchmarkReport]):
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


def print_saddle_benchmark_reports(reports: list[SaddleBenchmarkReport]):
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


def print_diffusion_benchmark_reports(reports: list[DiffusionBenchmarkReport]):
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


def summarize_k0_laplacian(seq, operators, *, dirichlet: bool) -> K0LaplacianSpectrumReport:
    _, _, eigvals = _dense_k0_hodge_inverse(seq, operators, dirichlet)
    abs_eigvals = jnp.abs(eigvals)
    scale = jnp.max(abs_eigvals)
    threshold = 1e-12 * jnp.where(scale > 0, scale, 1.0)
    nonzero_mask = abs_eigvals > threshold
    zero_count = int(jnp.sum(~nonzero_mask))
    min_nonzero = float(jnp.min(eigvals[nonzero_mask])) if zero_count < eigvals.shape[0] else 0.0
    max_eig = float(jnp.max(eigvals))
    condition = max_eig / min_nonzero if min_nonzero > 0.0 else jnp.inf
    return K0LaplacianSpectrumReport(
        dirichlet=dirichlet,
        size=int(eigvals.shape[0]),
        zero_count=zero_count,
        min_nonzero_eig=min_nonzero,
        max_eig=max_eig,
        nonzero_condition_number=float(condition),
    )


def print_k0_laplacian_spectrum_report(report: K0LaplacianSpectrumReport):
    print("=" * 112)
    print(
        f"k=0 Laplacian summary: dirichlet={report.dirichlet}, shape=({report.size}, {report.size}) \n"
        f"#zero eigs={report.zero_count}, min nonzero eig={report.min_nonzero_eig:.3e} \n"
        f"max eig={report.max_eig:.3e}, max/min nonzero eig={report.nonzero_condition_number:.3e}"
    )


def plot_k0_laplacian_benchmark_reports(
    benchmarks: dict[bool, list[K0LaplacianBenchmarkReport]],
    *,
    eps: float,
    show: bool = True,
):
    labels = benchmark_labels()
    x = jnp.arange(len(labels))
    color_map = {
        "jacobi": "#1f77b4",
        f"richardson-{POLY_STEPS}": "#ff7f0e",
        f"chebyshev-{POLY_STEPS}": "#2ca02c",
        "tensor": "#d62728",
        "dense-pinv": "#9467bd",
    }
    colors = [color_map[label] for label in labels]
    metrics = [
        ("avg_iters", "std_iters", "Average Iterations", True),
        ("avg_time_ms", "std_time_ms", "Average Time [ms]", True),
        ("avg_relative_residual", "std_relative_residual", "Average Relative Residual", True),
    ]

    fig, axes = plt.subplots(
        len(metrics),
        1,
        figsize=(6.8, 10),
        sharex=True,
        constrained_layout=True,
    )
    free_reports = {report.label: report for report in benchmarks[False]}
    dbc_reports = {report.label: report for report in benchmarks[True]}

    for row, (metric_name, std_name, metric_label, log_scale) in enumerate(metrics):
        ax = axes[row]
        free_values = [getattr(free_reports[label], metric_name) for label in labels]
        dbc_values = [getattr(dbc_reports[label], metric_name) for label in labels]
        free_errors = [getattr(free_reports[label], std_name) for label in labels]
        dbc_errors = [getattr(dbc_reports[label], std_name) for label in labels]
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
        ax.set_title(f"k=0 {metric_label}")
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylabel(metric_label)
        if row == len(metrics) - 1:
            ax.set_xticks(x, labels, rotation=25, ha="right")
        else:
            ax.set_xticks(x, [])

    axes[0].legend(
        handles=[
            Patch(facecolor="#595959", alpha=0.9, label="dirichlet=False"),
            Patch(facecolor="#595959", alpha=0.45, label="dirichlet=True"),
        ],
        loc="upper right",
        frameon=False,
    )
    if show:
        plt.show()
    return fig


def plot_k0_laplacian_benchmark_heatmaps(
    benchmarks: dict[bool, list[K0LaplacianBenchmarkReport]],
    *,
    eps: float,
    show: bool = True,
):
    labels = benchmark_labels()
    case_labels = ["k=0, dbc=False", "k=0, dbc=True"]
    iteration_matrix = jnp.asarray([
        [next(report.avg_iters for report in benchmarks[dirichlet] if report.label == label) for dirichlet in (False, True)]
        for label in labels
    ])
    time_matrix = jnp.asarray([
        [next(report.avg_time_ms for report in benchmarks[dirichlet] if report.label == label) for dirichlet in (False, True)]
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


def plot_k0_laplacian_spectrum_reports(
    spectra: dict[bool, K0LaplacianSpectrumReport],
    *,
    show: bool = True,
):
    metrics = [
        ("min_nonzero_eig", r"$\lambda_{\min}^{+}(L_0)$", True),
        ("max_eig", r"$\lambda_{\max}(L_0)$", True),
        ("nonzero_condition_number", r"$\kappa^{+}(L_0)$", True),
    ]
    x = jnp.arange(1)
    width = 0.36

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for ax, (metric_name, title, log_scale) in zip(axes, metrics, strict=True):
        free_values = [getattr(spectra[False], metric_name)]
        dbc_values = [getattr(spectra[True], metric_name)]
        ax.bar(x - width / 2, free_values, width=width, color="#1f77b4", alpha=0.9, label="dirichlet=False")
        ax.bar(x + width / 2, dbc_values, width=width, color="#ff7f0e", alpha=0.75, label="dirichlet=True")
        if log_scale:
            ax.set_yscale("log", base=10)
        ax.set_title(title)
        ax.set_xticks(x, ["k=0"])
        ax.grid(axis="y", alpha=0.25)

    axes[0].legend(frameon=False)
    if show:
        plt.show()
    return fig


def plot_k0_laplacian_spectrum_heatmap(
    spectra: dict[bool, K0LaplacianSpectrumReport],
    *,
    show: bool = True,
):
    metric_names = ["min_nonzero_eig", "max_eig", "nonzero_condition_number"]
    metric_labels = [r"$\lambda_{\min}^{+}$", r"$\lambda_{\max}$", r"$\kappa^{+}$"]
    matrix = jnp.asarray([
        [getattr(spectra[dirichlet], metric_name) for dirichlet in (False, True)]
        for metric_name in metric_names
    ])

    fig, ax = plt.subplots(1, 1, figsize=(12, 3.8), constrained_layout=True)
    image = ax.imshow(_log10_safe(matrix), aspect="auto", cmap="cividis")
    ax.set_title("k=0 Laplacian Spectrum Metrics ($\\log_{10}$ scale)")
    ax.set_xticks(range(2), ["k=0, dbc=False", "k=0, dbc=True"], rotation=35, ha="right")
    ax.set_yticks(range(len(metric_labels)), metric_labels)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.9)
    colorbar.set_label(r"$\log_{10}(\mathrm{value})$")
    if show:
        plt.show()
    return fig


def plot_saddle_benchmark_reports(
    benchmarks: dict[bool, list[SaddleBenchmarkReport]],
    *,
    k: int,
    eps: float,
    show: bool = True,
):
    labels = [report.label for report in benchmarks[False]]
    x = jnp.arange(len(labels))
    color_map = {
        "mass=tensor / outer=jacobi": "#1f77b4",
        f"mass=tensor / outer=richardson-{POLY_STEPS}": "#ff7f0e",
        f"mass=tensor / outer=chebyshev-{POLY_STEPS}": "#2ca02c",
    }
    colors = [color_map[label] for label in labels]
    metrics = [
        ("avg_iters", "std_iters", "Average Iterations", True),
        ("avg_time_ms", "std_time_ms", "Average Time [ms]", True),
        ("avg_relative_residual", "std_relative_residual", "Average Relative Residual", True),
    ]

    fig, axes = plt.subplots(
        len(metrics),
        1,
        figsize=(7.4, 10),
        sharex=True,
        constrained_layout=True,
    )
    free_reports = {report.label: report for report in benchmarks[False]}
    dbc_reports = {report.label: report for report in benchmarks[True]}

    for row, (metric_name, std_name, metric_label, log_scale) in enumerate(metrics):
        ax = axes[row]
        free_values = [getattr(free_reports[label], metric_name) for label in labels]
        dbc_values = [getattr(dbc_reports[label], metric_name) for label in labels]
        free_errors = [getattr(free_reports[label], std_name) for label in labels]
        dbc_errors = [getattr(dbc_reports[label], std_name) for label in labels]
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
        ax.set_ylabel(metric_label)
        if row == len(metrics) - 1:
            ax.set_xticks(x, labels, rotation=25, ha="right")
        else:
            ax.set_xticks(x, [])

    axes[0].legend(
        handles=[
            Patch(facecolor="#595959", alpha=0.9, label="dirichlet=False"),
            Patch(facecolor="#595959", alpha=0.45, label="dirichlet=True"),
        ],
        loc="upper right",
        frameon=False,
    )
    if show:
        plt.show()
    return fig


def plot_saddle_benchmark_heatmaps(
    benchmarks: dict[bool, list[SaddleBenchmarkReport]],
    *,
    k: int,
    eps: float,
    show: bool = True,
):
    labels = [report.label for report in benchmarks[False]]
    case_labels = [f"k={k}, dbc=False", f"k={k}, dbc=True"]
    iteration_matrix = jnp.asarray([
        [next(report.avg_iters for report in benchmarks[dirichlet] if report.label == label) for dirichlet in (False, True)]
        for label in labels
    ])
    time_matrix = jnp.asarray([
        [next(report.avg_time_ms for report in benchmarks[dirichlet] if report.label == label) for dirichlet in (False, True)]
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


def _saddle_benchmarks_for_eps(
    saddle_benchmarks_by_k: dict[int, dict[float, dict[bool, list[SaddleBenchmarkReport]]]] | None,
    eps: float,
) -> dict[int, dict[bool, list[SaddleBenchmarkReport]]]:
    if saddle_benchmarks_by_k is None:
        return {}
    return {
        k: benchmarks_by_eps[eps]
        for k, benchmarks_by_eps in saddle_benchmarks_by_k.items()
        if eps in benchmarks_by_eps
    }


def plot_laplacian_benchmark_reports(
    k0_benchmarks: dict[bool, list[K0LaplacianBenchmarkReport]],
    saddle_benchmarks: dict[int, dict[bool, list[SaddleBenchmarkReport]]],
    *,
    eps: float,
    show: bool = True,
):
    del eps
    metrics = [
        ("avg_iters", "std_iters", "Average Iterations", True),
        ("avg_time_ms", "std_time_ms", "Average Time [ms]", True),
        ("avg_relative_residual", "std_relative_residual", "Average Relative Residual", True),
    ]
    k0_color_map = {
        "jacobi": "#1f77b4",
        f"richardson-{POLY_STEPS}": "#ff7f0e",
        f"chebyshev-{POLY_STEPS}": "#2ca02c",
        "tensor": "#d62728",
        "dense-pinv": "#9467bd",
    }
    saddle_color_map = {
        "mass=tensor / outer=jacobi": "#1f77b4",
        f"mass=tensor / outer=richardson-{POLY_STEPS}": "#ff7f0e",
        f"mass=tensor / outer=chebyshev-{POLY_STEPS}": "#2ca02c",
    }
    column_specs = [
        (0, benchmark_labels(), k0_benchmarks, k0_color_map),
        *[
            (
                k,
                [report.label for report in saddle_benchmarks[k][False]],
                saddle_benchmarks[k],
                saddle_color_map,
            )
            for k in sorted(saddle_benchmarks)
        ],
    ]

    fig, axes = plt.subplots(
        len(metrics),
        len(column_specs),
        figsize=(6.0 * len(column_specs), 10),
        sharey="row",
        constrained_layout=True,
        squeeze=False,
    )

    for col, (k, labels, benchmarks, color_map) in enumerate(column_specs):
        x = jnp.arange(len(labels))
        colors = [color_map[label] for label in labels]
        free_reports = {report.label: report for report in benchmarks[False]}
        dbc_reports = {report.label: report for report in benchmarks[True]}

        for row, (metric_name, std_name, metric_label, log_scale) in enumerate(metrics):
            ax = axes[row, col]
            free_values = [getattr(free_reports[label], metric_name) for label in labels]
            dbc_values = [getattr(dbc_reports[label], metric_name) for label in labels]
            free_errors = [getattr(free_reports[label], std_name) for label in labels]
            dbc_errors = [getattr(dbc_reports[label], std_name) for label in labels]
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

    axes[0, 0].legend(
        handles=[
            Patch(facecolor="#595959", alpha=0.9, label="dirichlet=False"),
            Patch(facecolor="#595959", alpha=0.45, label="dirichlet=True"),
        ],
        loc="upper right",
        frameon=False,
    )
    if show:
        plt.show()
    return fig


def plot_laplacian_benchmark_heatmaps(
    k0_benchmarks: dict[bool, list[K0LaplacianBenchmarkReport]],
    saddle_benchmarks: dict[int, dict[bool, list[SaddleBenchmarkReport]]],
    *,
    eps: float,
    show: bool = True,
):
    del eps
    labels = benchmark_labels()
    saddle_label_map = {
        "jacobi": "mass=tensor / outer=jacobi",
        f"richardson-{POLY_STEPS}": f"mass=tensor / outer=richardson-{POLY_STEPS}",
        f"chebyshev-{POLY_STEPS}": f"mass=tensor / outer=chebyshev-{POLY_STEPS}",
    }
    case_labels = ["k=0, dbc=False", "k=0, dbc=True"] + [
        f"k={k}, dbc={dirichlet}"
        for k in sorted(saddle_benchmarks)
        for dirichlet in (False, True)
    ]

    k0_free_reports = {report.label: report for report in k0_benchmarks[False]}
    k0_dbc_reports = {report.label: report for report in k0_benchmarks[True]}
    saddle_free_reports = {
        k: {report.label: report for report in benchmarks[False]}
        for k, benchmarks in saddle_benchmarks.items()
    }
    saddle_dbc_reports = {
        k: {report.label: report for report in benchmarks[True]}
        for k, benchmarks in saddle_benchmarks.items()
    }

    def _metric_row(metric_name: str, label: str):
        values = [
            getattr(k0_free_reports[label], metric_name),
            getattr(k0_dbc_reports[label], metric_name),
        ]
        saddle_label = saddle_label_map.get(label)
        for k in sorted(saddle_benchmarks):
            if saddle_label is None:
                values.extend([jnp.nan, jnp.nan])
                continue
            values.extend([
                getattr(saddle_free_reports[k][saddle_label], metric_name),
                getattr(saddle_dbc_reports[k][saddle_label], metric_name),
            ])
        return values

    iteration_matrix = jnp.asarray([
        _metric_row("avg_iters", label)
        for label in labels
    ], dtype=jnp.float64)
    time_matrix = jnp.asarray([
        _metric_row("avg_time_ms", label)
        for label in labels
    ], dtype=jnp.float64)

    fig, axes = plt.subplots(1, 2, figsize=(18, 4.8), constrained_layout=True)
    for ax, matrix, title, cmap_name in (
        (axes[0], iteration_matrix, "Average Iterations", "viridis"),
        (axes[1], time_matrix, "Average Time [ms]", "magma"),
    ):
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad(color="white")
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


def _flatten_benchmarks(
    benchmarks: dict[bool, list[K0LaplacianBenchmarkReport]],
) -> list[dict]:
    rows = []
    for reports in benchmarks.values():
        rows.extend(asdict(report) for report in reports)
    return rows


def _flatten_spectra(
    spectra: dict[bool, K0LaplacianSpectrumReport],
) -> list[dict]:
    return [asdict(report) for report in spectra.values()]


def _flatten_saddle_benchmarks(
    benchmarks_by_eps: dict[float, dict[bool, list[SaddleBenchmarkReport]]],
) -> dict[str, list[dict]]:
    return {
        f"{eps:.1e}": [asdict(report) for reports in benchmarks.values() for report in reports]
        for eps, benchmarks in benchmarks_by_eps.items()
    }


def save_k0_laplacian_benchmark_artifacts(
    benchmarks_by_eps: dict[float, dict[bool, list[K0LaplacianBenchmarkReport]]],
    spectra: dict[bool, K0LaplacianSpectrumReport],
    saddle_benchmarks_by_k: dict[int, dict[float, dict[bool, list[SaddleBenchmarkReport]]]] | None = None,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs") / "interactive" / f"k0_laplacian_preconditioner_choices_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "config": asdict(CONFIG),
        "dense": DENSE,
        "eps_values": list(eps_list),
        "poly_steps": POLY_STEPS,
        "power_iterations": POWER_ITERATIONS,
        "richardson_damping_safety": RICHARDSON_DAMPING_SAFETY,
        "chebyshev_min_eig_fraction": CHEBYSHEV_MIN_EIG_FRACTION,
        "k3_tensor_rank": K3_TENSOR_RANK,
        "tensor_cp_kwargs": TENSOR_CP_KWARGS,
        "saddle_ks": sorted(saddle_benchmarks_by_k) if saddle_benchmarks_by_k is not None else [],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (output_dir / "benchmarks.json").write_text(
        json.dumps(
            {
                f"{eps:.1e}": _flatten_benchmarks(benchmarks)
                for eps, benchmarks in benchmarks_by_eps.items()
            },
            indent=2,
        )
    )
    if DENSE:
        (output_dir / "spectra.json").write_text(
            json.dumps(_flatten_spectra(spectra), indent=2)
        )
    if saddle_benchmarks_by_k is not None:
        (output_dir / "saddle_benchmarks.json").write_text(
            json.dumps(
                {
                    f"k{k}": _flatten_saddle_benchmarks(benchmarks_by_eps)
                    for k, benchmarks_by_eps in saddle_benchmarks_by_k.items()
                },
                indent=2,
            )
        )

    figures = []
    for eps, benchmarks in benchmarks_by_eps.items():
        eps_tag = f"eps_{eps:.1e}".replace("+", "").replace("-", "m")
        saddle_benchmarks = _saddle_benchmarks_for_eps(saddle_benchmarks_by_k, eps)
        figures.extend([
            (
                f"benchmark_bars_{eps_tag}.png",
                plot_laplacian_benchmark_reports(
                    benchmarks,
                    saddle_benchmarks,
                    eps=eps,
                    show=False,
                ),
            ),
            (
                f"benchmark_heatmaps_{eps_tag}.png",
                plot_laplacian_benchmark_heatmaps(
                    benchmarks,
                    saddle_benchmarks,
                    eps=eps,
                    show=False,
                ),
            ),
        ])
    if DENSE:
        figures.extend([
            (
                "spectrum_bars.png",
                plot_k0_laplacian_spectrum_reports(spectra, show=False),
            ),
            (
                "spectrum_heatmap.png",
                plot_k0_laplacian_spectrum_heatmap(spectra, show=False),
            ),
        ])

    for filename, fig in figures:
        fig.savefig(output_dir / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)

    return output_dir


# %% Build once
SEQ, OPERATORS = ensure_built(CONFIG, rebuild=False)
print(
    f"built Laplacian benchmark case: ns={CONFIG.ns}, p={CONFIG.p}, map_kind={CONFIG.map_kind}, "
    f"eps={CONFIG.rotating_eps}, kappa={CONFIG.rotating_kappa}, nfp={CONFIG.rotating_nfp}"
)
print(f"dense diagnostics/pseudoinverse baseline enabled: {DENSE}")


# %% Dense spectral diagnostics
ALL_SPECTRA = {}
if DENSE:
    for dirichlet in (False, True):
        spectrum = summarize_k0_laplacian(SEQ, OPERATORS, dirichlet=dirichlet)
        ALL_SPECTRA[dirichlet] = spectrum
        print_k0_laplacian_spectrum_report(spectrum)


# %% Compare the production-facing preconditioner choices
ALL_BENCHMARKS = {}
for eps in eps_list:
    benchmark_by_bc = {}
    for dirichlet in (False, True):
        reports = benchmark_k0_laplacian_preconditioners(
            SEQ,
            OPERATORS,
            eps=eps,
            dirichlet=dirichlet,
            n_rhs=8,
            seed=0,
        )
        benchmark_by_bc[dirichlet] = reports
        print("=" * 112)
        print(
            f"k=0 Laplacian preconditioner comparison: eps={eps:.1e}, dirichlet={dirichlet}, "
            f"richardson/chebyshev steps={POLY_STEPS}"
        )
        print_k0_laplacian_benchmark_reports(reports)
    ALL_BENCHMARKS[eps] = benchmark_by_bc


# %% Compare the production-facing k=1 / k=2 / k=3 Laplacian preconditioner choices
SADDLE_BENCHMARKS_BY_K = {}
for k in (1, 2, 3):
    benchmarks_by_eps = {}
    for eps in eps_list:
        benchmark_by_bc = {}
        for dirichlet in (False, True):
            reports = benchmark_saddle_laplacian_preconditioners(
                SEQ,
                OPERATORS,
                k=k,
                eps=eps,
                dirichlet=dirichlet,
                n_rhs=8,
                seed=0,
            )
            benchmark_by_bc[dirichlet] = reports
            print("=" * 112)
            print(
                f"k={k} Laplacian preconditioner comparison: eps={eps:.1e}, dirichlet={dirichlet}, "
                f"mass=tensor, schur.inner=tensor, coupled=False"
            )
            print_saddle_benchmark_reports(reports)
        benchmarks_by_eps[eps] = benchmark_by_bc
    SADDLE_BENCHMARKS_BY_K[k] = benchmarks_by_eps


# %% Diffusion smoke benchmark using the new built-in preconditioner options
DIFFUSION_BENCHMARKS_BY_K = {}
for k in (0, 1, 2, 3):
    benchmark_by_bc = {}
    for dirichlet in (False, True):
        reports = benchmark_diffusion_preconditioners(
            SEQ,
            OPERATORS,
            k=k,
            eps=DIFFUSION_EPS,
            dirichlet=dirichlet,
            n_rhs=4,
            seed=11,
        )
        benchmark_by_bc[dirichlet] = reports
        print("=" * 112)
        print(
            f"diffusion preconditioner comparison: k={k}, eps={DIFFUSION_EPS:.1e}, "
            f"dirichlet={dirichlet}, chebyshev steps={POLY_STEPS}"
        )
        print_diffusion_benchmark_reports(reports)
    DIFFUSION_BENCHMARKS_BY_K[k] = benchmark_by_bc


# %% Visualize the benchmark summaries
for eps, benchmarks in ALL_BENCHMARKS.items():
    saddle_benchmarks = _saddle_benchmarks_for_eps(SADDLE_BENCHMARKS_BY_K, eps)
    plot_laplacian_benchmark_reports(benchmarks, saddle_benchmarks, eps=eps)
    plot_laplacian_benchmark_heatmaps(benchmarks, saddle_benchmarks, eps=eps)
if DENSE:
    plot_k0_laplacian_spectrum_reports(ALL_SPECTRA)
    plot_k0_laplacian_spectrum_heatmap(ALL_SPECTRA)


# %% Store results and plots
OUTPUT_DIR = save_k0_laplacian_benchmark_artifacts(
    ALL_BENCHMARKS,
    ALL_SPECTRA,
    saddle_benchmarks_by_k=SADDLE_BENCHMARKS_BY_K,
)
print(f"saved benchmark artifacts to: {OUTPUT_DIR}")
# %%