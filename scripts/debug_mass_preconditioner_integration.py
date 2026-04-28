# %% [markdown]
# # Scalar Mass Preconditioner Benchmark
#
# This interactive script benchmarks the production scalar mass
# preconditioners across
#
# - resolutions `n = 4, 8, 16`,
# - spline orders `p = 1, 2, 3`,
# - form degrees `k = 0, 3`,
# - both `dirichlet = False/True`,
# - and multiple Gaussian right-hand sides per case.
#
# It assembles
#
# 1. the scalar mass operators,
# 2. production `jacobi`,
# 3. and production `tensor`,
#
# then records average iteration counts and solve times from a jitted CG solve.
#
# The final cells generate summary plots for
#
# - average iterations,
# - average solve time,
#
# across all `(p, k, dirichlet)` slices.

# %%
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import rotating_ellipse_map
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
    ns: tuple[int, int, int] = (4, 4, 4)
    p: int = 1
    tol: float = 1e-10
    maxiter: int = 2000
    betti: tuple[int, int, int, int] = (1, 1, 0, 0)
    rotating_eps: float = 0.2
    rotating_kappa: float = 1.1
    rotating_r0: float = 1.0
    rotating_nfp: int = 2


@dataclass
class MassBenchmarkReport:
    resolution: int
    p: int
    k: int
    dirichlet: bool
    kind: str
    n_rhs: int
    n_dof: int
    avg_iters: float
    std_iters: float
    max_iters: int
    avg_ms: float
    std_ms: float
    max_ms: float


CONFIG = ExperimentConfig()
RESOLUTIONS = (4, 8, 16)
PS = (1, 2, 3)
KS = (0, 3)
DIRICHLET_FLAGS = (False, True)
LABELS = ("jacobi", "tensor")
N_RHS = 10
BUILD_CACHE = {}
BENCHMARK_REPORTS = []
OUTPUT_DIR = Path("outputs/mass_preconditioner_benchmarks")


# %% Helpers
def build_sequence_and_operators(config: ExperimentConfig = CONFIG):
    seq = DeRhamSequence(
        config.ns,
        (config.p, config.p, config.p),
        2 * config.p,
        ("clamped", "periodic", "periodic"),
        polar=True,
        tol=config.tol,
        maxiter=config.maxiter,
        betti_numbers=config.betti,
    )
    seq.evaluate_1d()
    seq.assemble_reference_mass_matrix()
    seq.set_map(
        rotating_ellipse_map(
            eps=config.rotating_eps,
            kappa=config.rotating_kappa,
            R0=config.rotating_r0,
            nfp=config.rotating_nfp,
        )
    )

    operators = assemble_mass_operators(seq, seq.geometry, ks=(0, 3))
    operators = assemble_tensor_mass_preconditioner(seq, operators=operators, ks=(0, 3))
    return seq, operators


def get_built_case(config: ExperimentConfig):
    key = (config.ns, config.p, config.tol, config.maxiter,
           config.rotating_eps, config.rotating_kappa,
           config.rotating_r0, config.rotating_nfp)
    if key not in BUILD_CACHE:
        BUILD_CACHE[key] = build_sequence_and_operators(config)
    return BUILD_CACHE[key]


def gaussian_rhs(n_dof: int, seed: int) -> jnp.ndarray:
    return jax.random.normal(
        jax.random.PRNGKey(seed),
        (n_dof,),
        dtype=jnp.float64,
    )


def rhs_batch(n_dof: int, seed: int, n_rhs: int = N_RHS) -> jnp.ndarray:
    return jax.random.normal(
        jax.random.PRNGKey(seed),
        (n_rhs, n_dof),
        dtype=jnp.float64,
    )


def benchmark_mass_system(seq, operators, resolution: int, p: int,
                          k: int, dirichlet: bool, kind: str,
                          n_rhs: int = N_RHS, seed: int = 0):
    n_dof = getattr(seq, f"n{k}_dbc" if dirichlet else f"n{k}")
    rhs_values = rhs_batch(
        n_dof,
        seed + 1000 * resolution + 100 * p + 10 * k + int(dirichlet),
        n_rhs=n_rhs,
    )

    def A_mv(x):
        return apply_mass_matrix(seq, operators, x, k, dirichlet=dirichlet)

    M_mv = lambda x: apply_mass_matrix_preconditioner(
        seq,
        operators,
        x,
        k,
        dirichlet=dirichlet,
        kind=kind,
    )

    @jax.jit
    def solve(rhs):
        x, info = solve_singular_cg(
            A_mv,
            rhs,
            mass_matvec=A_mv,
            precond_matvec=M_mv,
            tol=CONFIG.tol,
            maxiter=CONFIG.maxiter,
        )
        return x, jnp.abs(info)

    x0, _ = solve(rhs_values[0])
    jax.block_until_ready(x0)

    iterations = []
    times_ms = []
    for rhs in rhs_values:
        t0 = time.perf_counter()
        x, info = solve(rhs)
        jax.block_until_ready(x)
        times_ms.append((time.perf_counter() - t0) * 1e3)
        iterations.append(int(info))

    iterations = jnp.asarray(iterations)
    times_ms = jnp.asarray(times_ms)
    return MassBenchmarkReport(
        resolution=resolution,
        p=p,
        k=k,
        dirichlet=dirichlet,
        kind=kind,
        n_rhs=n_rhs,
        n_dof=n_dof,
        avg_iters=float(jnp.mean(iterations)),
        std_iters=float(jnp.std(iterations)),
        max_iters=int(jnp.max(iterations)),
        avg_ms=float(jnp.mean(times_ms)),
        std_ms=float(jnp.std(times_ms)),
        max_ms=float(jnp.max(times_ms)),
    )


def print_report(report: MassBenchmarkReport):
    print("-" * 96)
    print(
        f"scalar mass preconditioner benchmark: n={report.resolution} p={report.p} "
        f"k={report.k} dirichlet={report.dirichlet}"
    )
    print(f"label={report.kind} n_rhs={report.n_rhs} n_dof={report.n_dof}")
    print(f"avg_iters : {report.avg_iters:.2f}")
    print(f"std_iters : {report.std_iters:.2f}")
    print(f"max_iters : {report.max_iters:d}")
    print(f"avg_ms    : {report.avg_ms:.2f}")
    print(f"std_ms    : {report.std_ms:.2f}")
    print(f"max_ms    : {report.max_ms:.2f}")


def _scenario_title(k: int, dirichlet: bool) -> str:
    bc = "dbc" if dirichlet else "no dbc"
    return f"k={k}, {bc}"


def _collect_metric(reports, p: int, k: int, dirichlet: bool, kind: str, metric: str):
    selected = [
        report for report in reports
        if report.p == p and report.k == k and report.dirichlet == dirichlet and report.kind == kind
    ]
    selected = sorted(selected, key=lambda report: report.resolution)
    xs = [report.resolution for report in selected]
    ys = [getattr(report, metric) for report in selected]
    return xs, ys


def _subplot_scaling_reference(reports, k: int, dirichlet: bool, metric: str):
    points = []
    for p in PS:
        for kind in LABELS:
            xs, ys = _collect_metric(reports, p, k, dirichlet, kind, metric)
            points.extend(zip(xs, ys))
    if not points:
        return [], []
    xs_ref = list(RESOLUTIONS)
    scale = 1.2 * max(y / (x ** 3) for x, y in points)
    ys_ref = [scale * (x ** 3) for x in xs_ref]
    return xs_ref, ys_ref


def make_benchmark_figure(reports, metric: str, ylabel: str, filename: str):
    from matplotlib.lines import Line2D

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        len(DIRICHLET_FLAGS),
        len(KS),
        figsize=(12, 8),
        sharex=True,
    )
    colors = {
        "jacobi": "#264653",
        "tensor": "#2a9d8f",
    }
    markers = {
        "jacobi": "o",
        "tensor": "s",
    }
    linestyles = {
        1: "-",
        2: "--",
        3: "-.",
    }
    y_values = [getattr(report, metric) for report in reports if getattr(report, metric) > 0]
    if metric == "avg_ms":
        for dirichlet in DIRICHLET_FLAGS:
            for k in KS:
                _, ys_ref = _subplot_scaling_reference(reports, k, dirichlet, metric)
                y_values.extend(y for y in ys_ref if y > 0)
    global_ymin = min(y_values) / 1.15
    global_ymax = max(y_values) * 1.15

    for row, dirichlet in enumerate(DIRICHLET_FLAGS):
        for col, k in enumerate(KS):
            ax = axes[row, col]
            for p in PS:
                for kind in LABELS:
                    xs, ys = _collect_metric(reports, p, k, dirichlet, kind, metric)
                    if not xs:
                        continue
                    ax.plot(
                        xs,
                        ys,
                        marker=markers[kind],
                        color=colors[kind],
                        linestyle=linestyles[p],
                        linewidth=2,
                    )
            if metric == "avg_ms":
                xs_ref, ys_ref = _subplot_scaling_reference(reports, k, dirichlet, metric)
                if xs_ref:
                    ax.plot(
                        xs_ref,
                        ys_ref,
                        color="grey",
                        linestyle=":",
                        linewidth=1.5,
                        alpha=0.8,
                    )
            if row == 0:
                ax.set_title(f"k={k}")
            if col == 0:
                bc_label = "dbc" if dirichlet else "no dbc"
                ax.set_ylabel(f"{bc_label}\n{ylabel}")
            if row == len(DIRICHLET_FLAGS) - 1:
                ax.set_xlabel("resolution n")
            ax.set_xticks(list(RESOLUTIONS))
            ax.grid(alpha=0.25)
            ax.set_yscale("log")
            ax.set_xscale("log")
            ax.set_ylim(global_ymin, global_ymax)

    kind_handles = [
        Line2D(
            [0],
            [0],
            color=colors[kind],
            marker=markers[kind],
            linestyle="-",
            linewidth=2,
            label=kind,
        )
        for kind in LABELS
    ]
    p_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=linestyles[p],
            linewidth=2,
            label=f"p={p}",
        )
        for p in PS
    ]
    if metric == "avg_ms":
        p_handles.append(
            Line2D(
                [0],
                [0],
                color="grey",
                linestyle=":",
                linewidth=1.5,
                label=r"$n^3$",
            )
        )
    fig.legend(kind_handles, LABELS, loc="upper center", bbox_to_anchor=(0.35, 1.02), ncol=len(LABELS), frameon=False)
    p_labels = [f"p={p}" for p in PS]
    if metric == "avg_ms":
        p_labels.append(r"$n^3$")
    fig.legend(p_handles, p_labels, loc="upper center", bbox_to_anchor=(0.82, 1.02), ncol=len(p_labels), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path = OUTPUT_DIR / filename
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"saved {out_path}")
    return fig

# %% Run the full benchmark scan
BENCHMARK_REPORTS = []
for p in PS:
    for resolution in RESOLUTIONS:
        config = ExperimentConfig(ns=(resolution, resolution, resolution), p=p)
        seq, operators = get_built_case(config)
        print(f"running benchmark for n={resolution}, p={p}")
        for k in KS:
            for dirichlet in DIRICHLET_FLAGS:
                for kind in LABELS:
                    report = benchmark_mass_system(
                        seq,
                        operators,
                        resolution,
                        p,
                        k,
                        dirichlet,
                        kind,
                        n_rhs=N_RHS,
                    )
                    print_report(report)
                    BENCHMARK_REPORTS.append(report)


# %% Plot average iterations
FIG_ITERS = make_benchmark_figure(
    BENCHMARK_REPORTS,
    metric="avg_iters",
    ylabel="avg iterations",
    filename="scalar_mass_preconditioner_avg_iters.png",
)
plt.show()


# %% Plot average solve time
FIG_MS = make_benchmark_figure(
    BENCHMARK_REPORTS,
    metric="avg_ms",
    ylabel="avg solve time [ms]",
    filename="scalar_mass_preconditioner_avg_ms.png",
)
plt.show()

# %%
