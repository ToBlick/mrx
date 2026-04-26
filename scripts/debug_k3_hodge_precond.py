# %% [markdown]
# # k=3 Hodge Schur-Block Debug
#
# This script is the interactive `k = 3` Hodge debug harness.
#
# It stays on the `k = 3`, `dirichlet = False` case only, where there is no
# harmonic nullspace to deflate. The goal is not to benchmark a scalable `L_3`
# preconditioner yet, because we do not have one. Instead, the script compares
# dense debug preconditioners for the full Schur block
#
# $$
# L_3 = D_2 M_2^{-1} D_2^T.
# $$
#
# The two candidates compared here are:
#
# 1. `jacobi`: the current diagonal `k = 3` Schur approximation.
# 2. `mass`: the dense Schur model
#
#    $$
#    \widetilde L_3^{(mass)} = D_2 \widetilde M_2^{-1} D_2^T
#    $$
#
#    where `\widetilde M_2^{-1}` is the production tensor mass preconditioner.
#
# Because this is a debug script, the mass candidate is inverted densely on a
# small grid. That lets us test whether the idea is good before worrying about
# a scalable inverse for the full Schur block.

# %%
from __future__ import annotations

import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import rotating_ellipse_map, toroid_map
from mrx.nullspace import get_nullspace
from mrx.operators import (
    _hodge_diaginv,
    apply_mass_matrix_preconditioner,
    assemble_derivative_operators,
    assemble_hodge_operators,
    assemble_incidence_operators,
    assemble_mass_operators,
    assemble_tensor_mass_preconditioner,
    dense_derivative_matrix,
    dense_hodge_laplacian,
)
from mrx.solvers import solve_singular_cg

jax.config.update("jax_enable_x64", True)


# %% Configuration
@dataclass(frozen=True)
class ExperimentConfig:
    ns: tuple[int, int, int] = (6, 6, 6)
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
    tensor_rank: int = 3
    tensor_cp_maxiter: int = 100
    tensor_cp_tol: float = 1e-9
    n_rhs: int = 10


@dataclass
class SchurModelDiagnostics:
    label: str
    relative_operator_error: float
    min_precond_eig: float
    max_precond_eig: float
    condition_estimate: float


@dataclass
class BenchmarkReport:
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

    operators = assemble_mass_operators(seq, seq.geometry, ks=(2, 3))
    operators = assemble_tensor_mass_preconditioner(
        seq,
        operators=operators,
        ks=(2, 3),
        rank=config.tensor_rank,
        cp_kwargs={
            "maxiter": config.tensor_cp_maxiter,
            "tol": config.tensor_cp_tol,
        },
    )
    operators = assemble_incidence_operators(seq, operators=operators, ks=(2,))
    operators = assemble_derivative_operators(seq, seq.geometry, operators=operators, ks=(2,))
    operators = assemble_hodge_operators(seq, seq.geometry, operators=operators, ks=(3,))

    seq.operators = operators
    seq._compute_nullspaces(config.betti, eps=seq.tol**0.5)
    return seq, seq.operators


def ensure_built(config: ExperimentConfig = CONFIG, rebuild: bool = False):
    global SEQ, OPERATORS, BUILT_CONFIG
    if rebuild or SEQ is None or OPERATORS is None or BUILT_CONFIG != config:
        SEQ, OPERATORS = build_case(config)
        BUILT_CONFIG = config
    return SEQ, OPERATORS


def _symmetrize(matrix: jnp.ndarray) -> jnp.ndarray:
    return 0.5 * (matrix + matrix.T)


def dense_operator_from_apply(apply_fn, size: int, dtype=jnp.float64) -> jnp.ndarray:
    eye = jnp.eye(size, dtype=dtype)
    columns = [apply_fn(eye[:, j]) for j in range(size)]
    return jnp.stack(columns, axis=1)


def build_dense_k3_data(seq, operators):
    dirichlet = False
    n3 = seq.n3
    n2 = seq.n2
    d2 = jnp.asarray(
        dense_derivative_matrix(
            seq,
            operators,
            2,
            dirichlet_in=dirichlet,
            dirichlet_out=dirichlet,
        )
    )
    exact_l3 = _symmetrize(
        jnp.asarray(dense_hodge_laplacian(seq, operators, 3, dirichlet=dirichlet))
    )
    m2_jacobi_inv = dense_operator_from_apply(
        lambda x: apply_mass_matrix_preconditioner(seq, operators, x, 2, dirichlet=dirichlet, kind="jacobi"),
        n2,
    )
    m2_tensor_inv = dense_operator_from_apply(
        lambda x: apply_mass_matrix_preconditioner(seq, operators, x, 2, dirichlet=dirichlet, kind="tensor"),
        n2,
    )
    jacobi_inv_diag = _hodge_diaginv(operators, 3, dirichlet=dirichlet)
    jacobi_schur = jnp.diag(1.0 / jacobi_inv_diag)
    mass_schur = _symmetrize(d2 @ m2_tensor_inv @ d2.T)
    lower_jacobi_schur = _symmetrize(d2 @ m2_jacobi_inv @ d2.T)
    return {
        "d2": d2,
        "exact_l3": exact_l3,
        "jacobi_schur": jacobi_schur,
        "mass_schur": mass_schur,
        "lower_jacobi_schur": lower_jacobi_schur,
        "m2_jacobi_inv": m2_jacobi_inv,
        "m2_tensor_inv": m2_tensor_inv,
        "n3": n3,
        "n2": n2,
    }


def schur_model_diagnostics(exact_l3: jnp.ndarray, candidate: jnp.ndarray, label: str) -> SchurModelDiagnostics:
    candidate = _symmetrize(candidate)
    candidate_inv = jnp.linalg.inv(candidate)
    precond_matrix = _symmetrize(candidate_inv @ exact_l3)
    eigs = jnp.linalg.eigvalsh(precond_matrix)
    min_eig = float(jnp.min(eigs))
    max_eig = float(jnp.max(eigs))
    safe_min = min_eig if min_eig > 0 else jnp.nan
    return SchurModelDiagnostics(
        label=label,
        relative_operator_error=float(jnp.linalg.norm(candidate - exact_l3) / jnp.linalg.norm(exact_l3)),
        min_precond_eig=min_eig,
        max_precond_eig=max_eig,
        condition_estimate=float(max_eig / safe_min),
    )


def print_schur_model_diagnostics(reports: list[SchurModelDiagnostics]):
    print("-" * 112)
    print(
        f"{'label':<18} {'rel op err':>12} {'min eig':>12} {'max eig':>12} {'cond est':>12}"
    )
    for report in reports:
        print(
            f"{report.label:<18} {report.relative_operator_error:>12.3e} "
            f"{report.min_precond_eig:>12.3e} {report.max_precond_eig:>12.3e} {report.condition_estimate:>12.3e}"
        )


def benchmark_dense_preconditioners(exact_l3: jnp.ndarray, preconditioners: dict[str, jnp.ndarray], rhs_batch: jnp.ndarray, tol: float, maxiter: int):
    reports = []

    def operator_matvec(x):
        return exact_l3 @ x

    for label, preconditioner_inv in preconditioners.items():
        def precond_matvec(x, pinv=preconditioner_inv):
            return pinv @ x

        @jax.jit
        def solve(rhs):
            solution, info = solve_singular_cg(
                operator_matvec,
                rhs,
                precond_matvec=precond_matvec,
                tol=tol,
                maxiter=maxiter,
            )
            residual = operator_matvec(solution) - rhs
            relative_residual = jnp.linalg.norm(residual) / jnp.maximum(jnp.linalg.norm(rhs), 1.0)
            return solution, info, relative_residual

        solution0, _, _ = solve(rhs_batch[0])
        jax.block_until_ready(solution0)

        iterations = []
        times_ms = []
        residuals = []
        for rhs in rhs_batch:
            t0 = time.perf_counter()
            solution, info, relative_residual = solve(rhs)
            jax.block_until_ready(solution)
            times_ms.append((time.perf_counter() - t0) * 1e3)
            iterations.append(int(jnp.abs(info)))
            residuals.append(float(relative_residual))

        iterations_arr = jnp.asarray(iterations)
        times_arr = jnp.asarray(times_ms)
        residuals_arr = jnp.asarray(residuals)
        reports.append(
            BenchmarkReport(
                label=label,
                n_rhs=rhs_batch.shape[0],
                avg_iters=float(jnp.mean(iterations_arr)),
                std_iters=float(jnp.std(iterations_arr)),
                max_iters=int(jnp.max(iterations_arr)),
                avg_time_ms=float(jnp.mean(times_arr)),
                std_time_ms=float(jnp.std(times_arr)),
                max_time_ms=float(jnp.max(times_arr)),
                avg_relative_residual=float(jnp.mean(residuals_arr)),
                std_relative_residual=float(jnp.std(residuals_arr)),
                max_relative_residual=float(jnp.max(residuals_arr)),
            )
        )

    return reports


def print_benchmark_reports(reports: list[BenchmarkReport]):
    print("-" * 132)
    print(
        f"{'label':<18} {'avg iters':>10} {'std':>8} {'max':>6} {'avg ms':>10} {'std ms':>10} {'max ms':>10} "
        f"{'avg relres':>14} {'std relres':>14} {'max relres':>14}"
    )
    for report in reports:
        print(
            f"{report.label:<18} {report.avg_iters:>10.2f} {report.std_iters:>8.2f} {report.max_iters:>6d} "
            f"{report.avg_time_ms:>10.2f} {report.std_time_ms:>10.2f} {report.max_time_ms:>10.2f} "
            f"{report.avg_relative_residual:>14.3e} {report.std_relative_residual:>14.3e} {report.max_relative_residual:>14.3e}"
        )


# %% Build once
SEQ, OPERATORS = ensure_built(CONFIG, rebuild=True)
print(
    f"built k=3 Schur debug case: ns={CONFIG.ns}, p={CONFIG.p}, map_kind={CONFIG.map_kind}, "
    f"tensor_rank={CONFIG.tensor_rank}, tol={CONFIG.tol}, maxiter={CONFIG.maxiter}"
)


# %% Verify there is no harmonic nullspace in the chosen case
K3_NULLSPACE = get_nullspace(OPERATORS, 3, False)
print(f"k=3 free nullspace dimension = {K3_NULLSPACE.shape[0]}")
if K3_NULLSPACE.shape[0] != 0:
    raise ValueError("This script expects the k=3 free case to have no harmonic nullspace")


# %% Build dense exact and approximate Schur-block models
K3_DENSE = build_dense_k3_data(SEQ, OPERATORS)
print(
    f"dense sizes: n3={K3_DENSE['n3']}, n2={K3_DENSE['n2']}, "
    f"L3 shape={tuple(K3_DENSE['exact_l3'].shape)}"
)


# %% Compare full Schur-block candidates against the exact L3 block
K3_SCHUR_DIAGNOSTICS = [
    schur_model_diagnostics(K3_DENSE["exact_l3"], K3_DENSE["jacobi_schur"], "jacobi"),
    schur_model_diagnostics(K3_DENSE["exact_l3"], K3_DENSE["mass_schur"], "mass"),
    schur_model_diagnostics(K3_DENSE["exact_l3"], K3_DENSE["lower_jacobi_schur"], "D M2_jac D^T"),
]
print_schur_model_diagnostics(K3_SCHUR_DIAGNOSTICS)


# %% Build random Gaussian DoF right-hand sides
RHS_KEY = jax.random.PRNGKey(0)
RHS_BATCH = jax.random.normal(RHS_KEY, (CONFIG.n_rhs, K3_DENSE["n3"]), dtype=jnp.float64)
print(
    f"rhs batch shape={tuple(RHS_BATCH.shape)}, avg ||rhs||_2={float(jnp.mean(jnp.linalg.norm(RHS_BATCH, axis=1))):.3e}"
)


# %% Benchmark dense debug preconditioners for the exact L3 system
PRECONDITIONER_INVERSES = {
    "jacobi": jnp.diag(_hodge_diaginv(OPERATORS, 3, dirichlet=False)),
    "mass": jnp.linalg.inv(K3_DENSE["mass_schur"]),
}
K3_BENCHMARKS = benchmark_dense_preconditioners(
    K3_DENSE["exact_l3"],
    PRECONDITIONER_INVERSES,
    RHS_BATCH,
    tol=CONFIG.tol,
    maxiter=CONFIG.maxiter,
)
print_benchmark_reports(K3_BENCHMARKS)


# %% Optional: visualize the exact and candidate Schur blocks
MATRICES_TO_PLOT = {
    "exact L3": K3_DENSE["exact_l3"],
    "jacobi Schur": K3_DENSE["jacobi_schur"],
    "mass Schur": K3_DENSE["mass_schur"],
}

fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
for ax, (label, matrix) in zip(axes, MATRICES_TO_PLOT.items()):
    image = ax.imshow(jnp.log10(jnp.abs(matrix) + 1e-16), origin="lower", cmap="viridis")
    ax.set_title(label)
    fig.colorbar(image, ax=ax, shrink=0.8)
plt.show()


# %% Optional: compare spectra of the preconditioned exact operator
PRECONDITIONED_SPECTRA = {}
for label, preconditioner_inv in PRECONDITIONER_INVERSES.items():
    preconditioned = _symmetrize(preconditioner_inv @ K3_DENSE["exact_l3"])
    PRECONDITIONED_SPECTRA[label] = jnp.linalg.eigvalsh(preconditioned)

fig, ax = plt.subplots(figsize=(8, 5))
for label, eigs in PRECONDITIONED_SPECTRA.items():
    ax.plot(jnp.asarray(eigs), marker="o", linestyle="none", label=label)
ax.set_title("Preconditioned L3 spectra")
ax.set_xlabel("eigenvalue index")
ax.set_ylabel("eigenvalue")
ax.legend()
plt.show()
