# %% [markdown]
# # k=0 Tensor Mass Benchmark
#
# This script keeps only the production-facing k=0 workflow:
#
# 1. assemble the k=0 mass operator,
# 2. assemble the production tensor mass preconditioner,
# 3. compare it against jacobi,
# 4. and keep the matrix and metric plots at the end.
#
# It now also includes `k = 0` stiffness diagnostics for the extracted matrix
#
#     E G^T M_1 G E^T,
#
# so we can check directly whether the same core-plus-bulk Schur split used by
# the scalar mass matrix survives on the scalar Hodge block.
#
# For polar k=0 the extracted matrix already splits as a small fused core plus
# one bulk tensor block, so the production apply is just a core Schur solve plus
# one tensor bulk inverse built from CP-ALS fits of the Jacobian factor J.

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
    _dense_incidence_1d,
    apply_mass_matrix,
    apply_mass_matrix_preconditioner,
    assemble_incidence_operators,
    assemble_mass_operators,
    assemble_projection_operators,
    assemble_tensor_mass_preconditioner,
    dense_hodge_laplacian,
    dense_mass_matrix,
    dense_projection_matrix,
)
from mrx.solvers import solve_singular_cg
from test.random_fields import build_random_besov_rhs_batch

jax.config.update("jax_enable_x64", True)


# %% Configuration
@dataclass(frozen=True)
class ExperimentConfig:
    ns: tuple[int, int, int] = (6, 12, 4)
    p: int = 3
    tol: float = 1e-9
    maxiter: int = 1000
    betti: tuple[int, int, int, int] = (1, 1, 0, 0)
    map_kind: str = "rotating_ellipse"
    torus_epsilon: float = 1.0 / 3.0
    torus_r0: float = 1.0
    rotating_eps: float = 0.3
    rotating_kappa: float = 1.2
    rotating_r0: float = 1.0
    rotating_nfp: int = 3


@dataclass
class K0BenchmarkReport:
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
class K0DiagonalStiffnessModelReport:
    dirichlet: bool
    rank: int
    bulk_relative_error: float
    full_relative_error: float
    schur_relative_error: float


@dataclass
class K0StiffnessPreconditionerSpectrumReport:
    label: str
    dirichlet: bool
    n_near_zero: int
    min_abs_nonzero_eig: float
    min_real_eig: float
    max_real_eig: float
    min_sym_eig: float
    max_sym_eig: float


@dataclass
class K0TransferComparisonReport:
    label: str
    scalar_dirichlet: bool
    k3_dirichlet: bool
    dense_relative_error: float
    mass_relative_error: float
    avg_action_relative_error: float
    max_action_relative_error: float


CONFIG = ExperimentConfig()
SEQ = None
OPERATORS = None
BUILT_CONFIG = None
TENSOR_CP_KWARGS = {"tol": 1e-9, "maxiter": 100}
BESOV_RHS_KWARGS = {
    "s": 1.0,
    "upper_limit": 24,
    "num_modes": 64,
    "scale": 1.0,
    "smoothness_margin": 0.25,
    "normalization_samples": 256,
}


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
    operators = assemble_mass_operators(seq, seq.geometry, ks=(0, 1))
    operators = assemble_incidence_operators(seq, operators=operators, ks=(0,))
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


def k0_surgery_block_indices(seq, dirichlet: bool) -> tuple[jnp.ndarray, jnp.ndarray]:
    e = jnp.asarray((seq.e0_dbc if dirichlet else seq.e0).todense())
    row_nnz = jnp.count_nonzero(e != 0, axis=1)
    surgery = jnp.where(row_nnz > 1)[0]
    bulk = jnp.where(row_nnz == 1)[0]
    return surgery, bulk


def _core_size(seq, dirichlet: bool) -> int:
    surgery_indices, _ = k0_surgery_block_indices(seq, dirichlet)
    return int(surgery_indices.shape[0])


def _bulk_tensor_shape(seq, dirichlet: bool) -> tuple[int, int, int]:
    nr_bulk = seq.basis_0.nr - 2 - int(dirichlet)
    nt = seq.basis_0.nt
    nz = seq.basis_0.nz
    return nr_bulk, nt, nz


def dense_extracted_mass_matrix(seq, operators, dirichlet: bool) -> jnp.ndarray:
    return jnp.asarray(dense_mass_matrix(seq, operators, 0, dirichlet=dirichlet))


def dense_extracted_stiffness_matrix(seq, operators, dirichlet: bool) -> jnp.ndarray:
    return jnp.asarray(dense_hodge_laplacian(seq, operators, 0, dirichlet=dirichlet))


def reorder_k0_blocks(seq, matrix: jnp.ndarray, dirichlet: bool):
    surgery_indices, bulk_indices = k0_surgery_block_indices(seq, dirichlet)
    permutation = jnp.concatenate([surgery_indices, bulk_indices])
    reordered = matrix[permutation][:, permutation]
    return reordered, surgery_indices, bulk_indices


def split_blocks(matrix: jnp.ndarray, core_size: int):
    acc = matrix[:core_size, :core_size]
    acb = matrix[:core_size, core_size:]
    abc = matrix[core_size:, :core_size]
    abb = matrix[core_size:, core_size:]
    return acc, acb, abc, abb


def build_k0_plot_data(seq, operators):
    plot_data = {}
    for dirichlet in (False, True):
        extracted = dense_extracted_mass_matrix(seq, operators, dirichlet)
        reordered, surgery_indices, bulk_indices = reorder_k0_blocks(seq, extracted, dirichlet)
        core_size = int(surgery_indices.shape[0])
        _, _, _, bulk = split_blocks(reordered, core_size)
        plot_data[dirichlet] = {
            "extracted": extracted,
            "reordered": reordered,
            "core_size": core_size,
            "surgery_indices": surgery_indices,
            "bulk_indices": bulk_indices,
            "is_prefix_layout": bool(jnp.array_equal(surgery_indices, jnp.arange(core_size))),
            "bulk": bulk,
            "bulk_shape": _bulk_tensor_shape(seq, dirichlet),
        }
    return plot_data


def build_k0_stiffness_plot_data(seq, operators):
    plot_data = {}
    for dirichlet in (False, True):
        extracted = dense_extracted_stiffness_matrix(seq, operators, dirichlet)
        reordered, surgery_indices, bulk_indices = reorder_k0_blocks(seq, extracted, dirichlet)
        core_size = int(surgery_indices.shape[0])
        _, _, _, bulk = split_blocks(reordered, core_size)
        plot_data[dirichlet] = {
            "extracted": extracted,
            "reordered": reordered,
            "core_size": core_size,
            "surgery_indices": surgery_indices,
            "bulk_indices": bulk_indices,
            "is_prefix_layout": bool(jnp.array_equal(surgery_indices, jnp.arange(core_size))),
            "bulk": bulk,
            "bulk_shape": _bulk_tensor_shape(seq, dirichlet),
        }
    return plot_data


def k0_split_diagnostics(matrix: jnp.ndarray, core_size: int) -> dict[str, float | int]:
    acc, acb, abc, abb = split_blocks(matrix, core_size)
    total_norm = float(jnp.linalg.norm(matrix))
    safe_total = total_norm if total_norm > 0 else 1.0
    return {
        "matrix_size": int(matrix.shape[0]),
        "core_size": int(core_size),
        "bulk_size": int(matrix.shape[0] - core_size),
        "total_norm": total_norm,
        "acc_norm": float(jnp.linalg.norm(acc)),
        "acb_norm": float(jnp.linalg.norm(acb)),
        "abc_norm": float(jnp.linalg.norm(abc)),
        "abb_norm": float(jnp.linalg.norm(abb)),
        "acc_rel": float(jnp.linalg.norm(acc) / safe_total),
        "acb_rel": float(jnp.linalg.norm(acb) / safe_total),
        "abc_rel": float(jnp.linalg.norm(abc) / safe_total),
        "abb_rel": float(jnp.linalg.norm(abb) / safe_total),
    }


def print_k0_split_diagnostics(label: str, diagnostics: dict[str, float | int]):
    print("-" * 112)
    print(
        f"{label}: size={diagnostics['matrix_size']}, core={diagnostics['core_size']}, bulk={diagnostics['bulk_size']}, "
        f"||A||_F={diagnostics['total_norm']:.3e}"
    )
    print(
        f"  ||A_cc||/||A||={diagnostics['acc_rel']:.3e}  "
        f"||A_cb||/||A||={diagnostics['acb_rel']:.3e}  "
        f"||A_bc||/||A||={diagnostics['abc_rel']:.3e}  "
        f"||A_bb||/||A||={diagnostics['abb_rel']:.3e}"
    )


def print_k0_extraction_split_info(label: str, plot_data: dict[str, object]):
    print(
        f"  {label}: extracted surgery rows={plot_data['core_size']}, "
        f"prefix_layout={plot_data['is_prefix_layout']}"
    )


def quadrature_tensor_shape(seq) -> tuple[int, int, int]:
    return seq.quad.ny, seq.quad.nx, seq.quad.nz


def reshape_quadrature_scalar_field(seq, values: jnp.ndarray) -> jnp.ndarray:
    return jnp.asarray(values).reshape(quadrature_tensor_shape(seq))


def k0_metric_tensor(seq) -> jnp.ndarray:
    return reshape_quadrature_scalar_field(seq, seq.geometry.jacobian_j)


def reshape_quadrature_matrix_field(seq, values: jnp.ndarray) -> jnp.ndarray:
    field = jnp.asarray(values)
    return field.reshape(*quadrature_tensor_shape(seq), *field.shape[1:])


def k0_stiffness_diagonal_metric_tensors(seq) -> dict[str, jnp.ndarray]:
    jacobian = reshape_quadrature_scalar_field(seq, seq.geometry.jacobian_j)
    metric_inv = reshape_quadrature_matrix_field(seq, seq.geometry.metric_inv_jkl)
    return {
        "alpha_rr": jacobian * metric_inv[..., 0, 0],
        "alpha_thetatheta": jacobian * metric_inv[..., 1, 1],
        "alpha_zetazeta": jacobian * metric_inv[..., 2, 2],
    }


def assemble_weighted_1d_mass(B: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    return (B * weights[None, :]) @ B.T


def assemble_weighted_1d_stiffness(
    primal_basis: jnp.ndarray,
    derivative_basis: jnp.ndarray,
    weights: jnp.ndarray,
    incidence: jnp.ndarray,
) -> jnp.ndarray:
    mass_d = assemble_weighted_1d_mass(derivative_basis, weights)
    stiffness = incidence.T @ (mass_d @ incidence)
    return 0.5 * (stiffness + stiffness.T)


def restrict_radial_window(raw_matrix: jnp.ndarray, radial_start: int, nr: int) -> jnp.ndarray:
    radial_stop = radial_start + nr
    return raw_matrix[radial_start:radial_stop, radial_start:radial_stop]


def kron3(left: jnp.ndarray, middle: jnp.ndarray, right: jnp.ndarray) -> jnp.ndarray:
    return jnp.kron(jnp.kron(left, middle), right)


def build_k0_diagonal_stiffness_bulk_model(seq, dirichlet: bool, rank: int):
    bulk_shape = _bulk_tensor_shape(seq, dirichlet)
    nr_bulk, _, _ = bulk_shape
    metric_tensors = k0_stiffness_diagonal_metric_tensors(seq)
    model = jnp.zeros((int(jnp.prod(jnp.asarray(bulk_shape))),) * 2, dtype=jnp.float64)
    fit_reports = []
    types = seq.basis_0.types
    g_r = _dense_incidence_1d(seq.basis_0.nr, types[0])
    g_t = _dense_incidence_1d(seq.basis_0.nt, types[1])
    g_z = _dense_incidence_1d(seq.basis_0.nz, types[2])

    field_specs = (
        (
            "alpha_rr",
            lambda scaled_weights: restrict_radial_window(
                assemble_weighted_1d_stiffness(
                    seq.basis_r_jk,
                    seq.d_basis_r_jk,
                    scaled_weights,
                    g_r,
                ),
                radial_start=2,
                nr=nr_bulk,
            ),
            lambda scaled_weights: assemble_weighted_1d_mass(seq.basis_t_jk, scaled_weights),
            lambda scaled_weights: assemble_weighted_1d_mass(seq.basis_z_jk, scaled_weights),
        ),
        (
            "alpha_thetatheta",
            lambda scaled_weights: restrict_radial_window(
                assemble_weighted_1d_mass(seq.basis_r_jk, scaled_weights),
                radial_start=2,
                nr=nr_bulk,
            ),
            lambda scaled_weights: assemble_weighted_1d_stiffness(
                seq.basis_t_jk,
                seq.d_basis_t_jk,
                scaled_weights,
                g_t,
            ),
            lambda scaled_weights: assemble_weighted_1d_mass(seq.basis_z_jk, scaled_weights),
        ),
        (
            "alpha_zetazeta",
            lambda scaled_weights: restrict_radial_window(
                assemble_weighted_1d_mass(seq.basis_r_jk, scaled_weights),
                radial_start=2,
                nr=nr_bulk,
            ),
            lambda scaled_weights: assemble_weighted_1d_mass(seq.basis_t_jk, scaled_weights),
            lambda scaled_weights: assemble_weighted_1d_stiffness(
                seq.basis_z_jk,
                seq.d_basis_z_jk,
                scaled_weights,
                g_z,
            ),
        ),
    )
    for label, radial_builder, theta_builder, zeta_builder in field_specs:
        weights, factors, diagnostics = cp_als_3tensor(
            metric_tensors[label],
            rank,
            maxiter=TENSOR_CP_KWARGS["maxiter"],
            tol=TENSOR_CP_KWARGS["tol"],
            ridge=1e-12,
        )
        fit_reports.append(
            MetricFactorFitDiagnostics(
                label=label,
                rank=diagnostics.rank,
                relative_error=diagnostics.relative_error,
                max_abs_error=diagnostics.max_abs_error,
                n_iters=diagnostics.n_iters,
                final_delta=diagnostics.final_delta,
            )
        )
        factor_theta, factor_r, factor_z = factors
        for idx in range(rank):
            term_r = radial_builder(seq.quad.w_x * (weights[idx] * factor_r[:, idx]))
            term_t = theta_builder(seq.quad.w_y * factor_theta[:, idx])
            term_z = zeta_builder(seq.quad.w_z * factor_z[:, idx])
            model = model + kron3(term_r, term_t, term_z)
    return model, fit_reports


def build_k0_diagonal_stiffness_surrogate(seq, operators, dirichlet: bool, rank: int):
    reordered = K0_STIFFNESS_PLOT_DATA[dirichlet]["reordered"]
    core_size = K0_STIFFNESS_PLOT_DATA[dirichlet]["core_size"]
    acc, acb, abc, _ = split_blocks(reordered, core_size)
    bulk_model, fit_reports = build_k0_diagonal_stiffness_bulk_model(seq, dirichlet, rank)
    top = jnp.concatenate([acc, acb], axis=1)
    bottom = jnp.concatenate([abc, bulk_model], axis=1)
    surrogate = jnp.concatenate([top, bottom], axis=0)

    schur_exact = acc - acb @ jnp.linalg.solve(K0_STIFFNESS_PLOT_DATA[dirichlet]["bulk"], abc)
    schur_model = acc - acb @ jnp.linalg.solve(bulk_model, abc)
    bulk_relative_error = float(jnp.linalg.norm(bulk_model - K0_STIFFNESS_PLOT_DATA[dirichlet]["bulk"]) / jnp.linalg.norm(K0_STIFFNESS_PLOT_DATA[dirichlet]["bulk"]))
    full_relative_error = float(jnp.linalg.norm(surrogate - reordered) / jnp.linalg.norm(reordered))
    schur_relative_error = float(jnp.linalg.norm(schur_model - schur_exact) / jnp.linalg.norm(schur_exact))
    report = K0DiagonalStiffnessModelReport(
        dirichlet=dirichlet,
        rank=rank,
        bulk_relative_error=bulk_relative_error,
        full_relative_error=full_relative_error,
        schur_relative_error=schur_relative_error,
    )
    return {
        "bulk_model": bulk_model,
        "surrogate": surrogate,
        "fit_reports": fit_reports,
        "report": report,
    }


def print_k0_diagonal_stiffness_model_reports(reports: list[K0DiagonalStiffnessModelReport]):
    print("-" * 112)
    print(
        f"{'rank':<8} {'bulk rel err':>16} {'full rel err':>16} {'schur rel err':>16}"
    )
    for report in reports:
        print(
            f"{report.rank:<8d} {report.bulk_relative_error:>16.3e} {report.full_relative_error:>16.3e} {report.schur_relative_error:>16.3e}"
        )


def _symmetrize(matrix: jnp.ndarray) -> jnp.ndarray:
    return 0.5 * (matrix + matrix.T)


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


def build_dense_k0_block_preconditioner(
    reordered_matrix: jnp.ndarray,
    core_size: int,
    bulk_inverse: jnp.ndarray,
    schur_null_vector: jnp.ndarray | None = None,
):
    acc, acb, abc, _ = split_blocks(reordered_matrix, core_size)
    schur = acc - acb @ (bulk_inverse @ abc)
    if schur_null_vector is None:
        schur_inv = _symmetrize(jnp.linalg.inv(schur))
        schur_project = lambda x: x
    else:
        schur_null_vector = jnp.asarray(schur_null_vector, dtype=jnp.float64)
        schur_null_vector = schur_null_vector / jnp.where(
            jnp.linalg.norm(schur_null_vector) > 0,
            jnp.linalg.norm(schur_null_vector),
            1.0,
        )
        schur_projector = jnp.eye(core_size, dtype=jnp.float64) - jnp.outer(
            schur_null_vector,
            schur_null_vector,
        )
        schur_reg = _symmetrize(schur + jnp.outer(schur_null_vector, schur_null_vector))
        schur_inv = _symmetrize(jnp.linalg.inv(schur_reg))
        schur_project = lambda x, projector=schur_projector: projector @ x

    def apply(rhs: jnp.ndarray) -> jnp.ndarray:
        rhs_c = rhs[:core_size]
        rhs_b = rhs[core_size:]
        y = bulk_inverse @ rhs_b
        schur_rhs = schur_project(rhs_c - acb @ y)
        z = schur_project(schur_inv @ schur_rhs)
        x_b = y - bulk_inverse @ (abc @ z)
        return jnp.concatenate([z, x_b])

    return apply


def k0_stiffness_nullspace_vectors(mass_matrix: jnp.ndarray, dirichlet: bool):
    if dirichlet:
        return []
    vector = jnp.ones((mass_matrix.shape[0],), dtype=jnp.float64)
    norm = jnp.sqrt(jnp.abs(jnp.dot(vector, mass_matrix @ vector)))
    vector = vector / jnp.where(norm > 0, norm, 1.0)
    return [vector]


def benchmark_k0_hodge_preconditioners(
    seq,
    mass_plot_data,
    stiffness_plot_data,
    stiffness_model_cache,
    *,
    dirichlet: bool,
    n_rhs: int = 8,
    seed: int = 0,
    tensor_ranks: tuple[int, ...] = (1, 3, 5),
    richardson_steps: tuple[int, ...] = (4, 8, 16),
    richardson_power_iterations: int = 20,
    richardson_damping_safety: float = 0.8,
    include_exact_block: bool = False,
) -> list[K0BenchmarkReport]:
    mass_reordered = mass_plot_data[dirichlet]["reordered"]
    reordered = stiffness_plot_data[dirichlet]["reordered"]
    core_size = stiffness_plot_data[dirichlet]["core_size"]
    _, _, _, bulk_exact = split_blocks(reordered, core_size)
    bulk_exact_inv = _symmetrize(jnp.linalg.inv(bulk_exact))
    diag_inv = 1.0 / jnp.diag(reordered)
    nullspace_vectors = k0_stiffness_nullspace_vectors(mass_reordered, dirichlet)
    schur_null_vector = nullspace_vectors[0][:core_size] if nullspace_vectors else None

    def M_mv(x):
        return mass_reordered @ x

    def project_primal(x: jnp.ndarray) -> jnp.ndarray:
        out = x
        for vector in nullspace_vectors:
            out = out - jnp.dot(vector, M_mv(out)) * vector
        return out

    def project_dual(f: jnp.ndarray) -> jnp.ndarray:
        out = f
        for vector in nullspace_vectors:
            out = out - jnp.dot(vector, out) * M_mv(vector)
        return out

    def make_projected_preconditioner(raw_preconditioner):
        def apply(x: jnp.ndarray) -> jnp.ndarray:
            return project_primal(raw_preconditioner(project_dual(x)))

        return apply

    def stiffness_apply(x: jnp.ndarray) -> jnp.ndarray:
        return reordered @ x

    def projected_stiffness_apply(x: jnp.ndarray) -> jnp.ndarray:
        return project_dual(stiffness_apply(project_primal(x)))

    jacobi_apply = make_projected_preconditioner(lambda x: diag_inv * x)
    labels = {
        "none": lambda x: x,
        "jacobi": jacobi_apply,
    }

    richardson_max_eig = estimate_preconditioned_max_eigenvalue_apply(
        projected_stiffness_apply,
        jacobi_apply,
        reordered.shape[0],
        n_iter=richardson_power_iterations,
        seed=seed + 610 + 100 * int(dirichlet),
    )
    richardson_omega = (
        richardson_damping_safety / richardson_max_eig if richardson_max_eig > 0.0 else 1.0
    )

    for steps in richardson_steps:
        labels[f"richardson-{steps}"] = build_richardson_apply_preconditioner(
            projected_stiffness_apply,
            jacobi_apply,
            steps=steps,
            omega=richardson_omega,
        )

    if include_exact_block:
        labels["exact-block"] = make_projected_preconditioner(
            build_dense_k0_block_preconditioner(
                reordered,
                core_size,
                bulk_exact_inv,
                schur_null_vector=schur_null_vector,
            )
        )

    for rank in tensor_ranks:
        model_data = stiffness_model_cache[dirichlet][rank]
        bulk_model = model_data["bulk_model"]
        bulk_model_inv = _symmetrize(jnp.linalg.inv(bulk_model))
        block_apply = build_dense_k0_block_preconditioner(
            reordered,
            core_size,
            bulk_model_inv,
            schur_null_vector=schur_null_vector,
        )
        labels[f"tensor-r{rank}"] = make_projected_preconditioner(block_apply)

    rhs_batch = jax.random.normal(
        jax.random.PRNGKey(seed + 300 * int(dirichlet)),
        (n_rhs, reordered.shape[0]),
        dtype=jnp.float64,
    )
    rhs_batch = jax.vmap(project_dual)(rhs_batch)

    reports = []
    for label, preconditioner in labels.items():
        @jax.jit
        def solve(rhs):
            x, info = solve_singular_cg(
                stiffness_apply,
                rhs,
                mass_matvec=M_mv,
                precond_matvec=preconditioner,
                vs=nullspace_vectors,
                tol=seq.tol,
                maxiter=seq.maxiter,
            )
            rhs_proj = project_dual(rhs)
            residual = project_dual(stiffness_apply(x) - rhs)
            relative_residual = jnp.linalg.norm(residual) / jnp.where(
                jnp.linalg.norm(rhs_proj) > 0,
                jnp.linalg.norm(rhs_proj),
                1.0,
            )
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
        iteration_array = jnp.asarray(iterations)
        reports.append(
            K0BenchmarkReport(
                label=label,
                dirichlet=dirichlet,
                n_rhs=n_rhs,
                avg_iters=float(jnp.mean(iteration_array)),
                std_iters=float(jnp.std(iteration_array)),
                max_iters=int(jnp.max(iteration_array)),
                avg_time_ms=float(jnp.mean(times_ms_array)),
                std_time_ms=float(jnp.std(times_ms_array)),
                max_time_ms=float(jnp.max(times_ms_array)),
                avg_relative_residual=float(jnp.mean(residuals_array)),
                std_relative_residual=float(jnp.std(residuals_array)),
                max_relative_residual=float(jnp.max(residuals_array)),
            )
        )
    return reports


def _dense_operator_from_apply(apply, size: int) -> jnp.ndarray:
    eye = jnp.eye(size, dtype=jnp.float64)
    columns = [apply(eye[:, idx]) for idx in range(size)]
    return jnp.stack(columns, axis=1)


def _mass_weighted_frobenius(matrix: jnp.ndarray, mass_matrix: jnp.ndarray) -> float:
    value = jnp.sum(matrix * (mass_matrix @ matrix))
    return float(jnp.sqrt(jnp.maximum(value, 0.0)))


def compare_k0_transfer_operators(
    seq,
    operators,
    *,
    scalar_dirichlet: bool,
    k3_dirichlet: bool,
    tensor_operator_cache: dict[int, object],
    n_rhs: int = 8,
    seed: int = 0,
) -> list[K0TransferComparisonReport]:
    mass_matrix = dense_extracted_mass_matrix(seq, operators, scalar_dirichlet)
    projection_matrix = jnp.asarray(
        dense_projection_matrix(
            seq,
            operators,
            0,
            3,
            dirichlet_in=k3_dirichlet,
            dirichlet_out=scalar_dirichlet,
        )
    )
    exact_transfer = jnp.linalg.solve(mass_matrix, projection_matrix)

    def output_norm(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.sqrt(jnp.maximum(jnp.dot(x, mass_matrix @ x), 0.0))

    labels = {
        "jacobi": lambda x: apply_mass_matrix_preconditioner(
            seq,
            operators,
            x,
            0,
            dirichlet=scalar_dirichlet,
            kind="jacobi",
        ),
    }
    for rank, tensor_operators in tensor_operator_cache.items():
        labels[f"tensor-r{rank}"] = (
            lambda x, tensor_ops=tensor_operators: apply_mass_matrix_preconditioner(
                seq,
                tensor_ops,
                x,
                0,
                dirichlet=scalar_dirichlet,
                kind="tensor",
            )
        )

    rhs_batch = jax.random.normal(
        jax.random.PRNGKey(seed + 500 * int(scalar_dirichlet) + 1000 * int(k3_dirichlet)),
        (n_rhs, projection_matrix.shape[1]),
        dtype=jnp.float64,
    )

    reports = []
    for label, apply_inv in labels.items():
        approx_transfer = jax.vmap(apply_inv, in_axes=1, out_axes=1)(projection_matrix)
        dense_relative_error = float(
            jnp.linalg.norm(approx_transfer - exact_transfer)
            / jnp.where(jnp.linalg.norm(exact_transfer) > 0, jnp.linalg.norm(exact_transfer), 1.0)
        )
        mass_relative_error = float(
            _mass_weighted_frobenius(approx_transfer - exact_transfer, mass_matrix)
            / max(_mass_weighted_frobenius(exact_transfer, mass_matrix), 1.0)
        )

        action_errors = []
        for rhs in rhs_batch:
            projected_rhs = projection_matrix @ rhs
            exact_output = jnp.linalg.solve(mass_matrix, projected_rhs)
            approx_output = apply_inv(projected_rhs)
            rel_error = output_norm(approx_output - exact_output) / jnp.where(
                output_norm(exact_output) > 0,
                output_norm(exact_output),
                1.0,
            )
            action_errors.append(float(rel_error))

        action_errors_array = jnp.asarray(action_errors)
        reports.append(
            K0TransferComparisonReport(
                label=label,
                scalar_dirichlet=scalar_dirichlet,
                k3_dirichlet=k3_dirichlet,
                dense_relative_error=dense_relative_error,
                mass_relative_error=mass_relative_error,
                avg_action_relative_error=float(jnp.mean(action_errors_array)),
                max_action_relative_error=float(jnp.max(action_errors_array)),
            )
        )
    return reports


def print_k0_transfer_comparison_reports(reports: list[K0TransferComparisonReport]):
    print("-" * 112)
    print(
        f"{'label':<16} {'dense rel err':>16} {'M0 rel err':>16} {'avg action err':>16} {'max action err':>16}"
    )
    for report in reports:
        print(
            f"{report.label:<16} {report.dense_relative_error:>16.3e} {report.mass_relative_error:>16.3e} "
            f"{report.avg_action_relative_error:>16.3e} {report.max_action_relative_error:>16.3e}"
        )


def k0_stiffness_preconditioner_spectrum_diagnostics(
    mass_plot_data,
    stiffness_plot_data,
    stiffness_model_cache,
    *,
    dirichlet: bool,
    zero_tol: float = 1e-10,
) -> list[K0StiffnessPreconditionerSpectrumReport]:
    mass_reordered = mass_plot_data[dirichlet]["reordered"]
    reordered = stiffness_plot_data[dirichlet]["reordered"]
    core_size = stiffness_plot_data[dirichlet]["core_size"]
    _, _, _, bulk_exact = split_blocks(reordered, core_size)
    bulk_exact_inv = _symmetrize(jnp.linalg.inv(bulk_exact))
    diag_inv = 1.0 / jnp.diag(reordered)
    nullspace_vectors = k0_stiffness_nullspace_vectors(mass_reordered, dirichlet)
    schur_null_vector = nullspace_vectors[0][:core_size] if nullspace_vectors else None

    def M_mv(x):
        return mass_reordered @ x

    def project_primal(x: jnp.ndarray) -> jnp.ndarray:
        out = x
        for vector in nullspace_vectors:
            out = out - jnp.dot(vector, M_mv(out)) * vector
        return out

    def project_dual(f: jnp.ndarray) -> jnp.ndarray:
        out = f
        for vector in nullspace_vectors:
            out = out - jnp.dot(vector, out) * M_mv(vector)
        return out

    def make_projected_preconditioner(raw_preconditioner):
        def apply(x: jnp.ndarray) -> jnp.ndarray:
            return project_primal(raw_preconditioner(project_dual(x)))

        return apply

    labels = {
        "jacobi": make_projected_preconditioner(lambda x: diag_inv * x),
        "exact-block": make_projected_preconditioner(
            build_dense_k0_block_preconditioner(
                reordered,
                core_size,
                bulk_exact_inv,
                schur_null_vector=schur_null_vector,
            )
        ),
    }
    for rank, model_data in stiffness_model_cache[dirichlet].items():
        bulk_model_inv = _symmetrize(jnp.linalg.inv(model_data["bulk_model"]))
        labels[f"diag-r{rank}"] = make_projected_preconditioner(
            build_dense_k0_block_preconditioner(
                reordered,
                core_size,
                bulk_model_inv,
                schur_null_vector=schur_null_vector,
            )
        )

    reports = []
    for label, preconditioner in labels.items():
        matrix = _dense_operator_from_apply(preconditioner, reordered.shape[0])
        eigvals = jnp.linalg.eigvals(matrix)
        sym_eigvals = jnp.linalg.eigvalsh(_symmetrize(matrix))
        abs_eigvals = jnp.abs(eigvals)
        n_near_zero = int(jnp.sum(abs_eigvals < zero_tol))
        nonzero_mask = abs_eigvals >= zero_tol
        min_abs_nonzero = float(
            jnp.min(jnp.where(nonzero_mask, abs_eigvals, jnp.inf))
        ) if bool(jnp.any(nonzero_mask)) else 0.0
        reports.append(
            K0StiffnessPreconditionerSpectrumReport(
                label=label,
                dirichlet=dirichlet,
                n_near_zero=n_near_zero,
                min_abs_nonzero_eig=min_abs_nonzero,
                min_real_eig=float(jnp.min(jnp.real(eigvals))),
                max_real_eig=float(jnp.max(jnp.real(eigvals))),
                min_sym_eig=float(jnp.min(sym_eigvals)),
                max_sym_eig=float(jnp.max(sym_eigvals)),
            )
        )
    return reports


def print_k0_stiffness_preconditioner_spectrum_reports(
    reports: list[K0StiffnessPreconditionerSpectrumReport],
):
    print("-" * 112)
    print(
        f"{'label':<16} {'n0':>6} {'min |lam| nz':>14} {'min Re lam':>14} {'max Re lam':>14} {'min sym lam':>14} {'max sym lam':>14}"
    )
    for report in reports:
        print(
            f"{report.label:<16} {report.n_near_zero:>6d} {report.min_abs_nonzero_eig:>14.3e} "
            f"{report.min_real_eig:>14.3e} {report.max_real_eig:>14.3e} "
            f"{report.min_sym_eig:>14.3e} {report.max_sym_eig:>14.3e}"
        )


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
) -> list[K0BenchmarkReport]:
    n_rhs = rhs_batch.shape[0]

    def A_mv(x):
        return apply_mass_matrix(seq, operators, x, 0, dirichlet=dirichlet)

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
            K0BenchmarkReport(
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


def benchmark_k0_preconditioners(
    seq,
    operators,
    *,
    dirichlet: bool,
    n_rhs: int = 8,
    seed: int = 0,
    tensor_ranks: tuple[int, ...] = (1, 3, 5),
    richardson_steps: tuple[int, ...] = (4, 8, 16),
    richardson_power_iterations: int = 20,
    richardson_damping_safety: float = 0.8,
    tensor_operator_cache: dict[int, object] | None = None,
) -> list[K0BenchmarkReport]:
    rhs_batch = build_random_besov_rhs_batch(
        seq,
        0,
        dirichlet=dirichlet,
        n_rhs=n_rhs,
        seed=seed + 100 * int(dirichlet),
        **BESOV_RHS_KWARGS,
    )

    def mass_apply(x: jnp.ndarray) -> jnp.ndarray:
        return apply_mass_matrix(seq, operators, x, 0, dirichlet=dirichlet)

    def jacobi_apply(x: jnp.ndarray) -> jnp.ndarray:
        return apply_mass_matrix_preconditioner(
            seq, operators, x, 0, dirichlet=dirichlet, kind="jacobi"
        )

    richardson_max_eig = estimate_preconditioned_max_eigenvalue_apply(
        mass_apply,
        jacobi_apply,
        rhs_size,
        n_iter=richardson_power_iterations,
        seed=seed + 310,
    )
    richardson_omega = (
        richardson_damping_safety / richardson_max_eig if richardson_max_eig > 0.0 else 1.0
    )

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
            ks=(0,),
            rank=tensor_rank,
            cp_kwargs=TENSOR_CP_KWARGS,
        )
        labels[f"tensor-r{tensor_rank}"] = (
            lambda x, tensor_ops=tensor_operators: apply_mass_matrix_preconditioner(
                seq,
                tensor_ops,
                x,
                0,
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


def print_k0_benchmark_reports(reports: list[K0BenchmarkReport]):
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


# %% Build once
SEQ, OPERATORS = ensure_built(CONFIG, rebuild=True)
print(
    f"built k=0 tensor case: ns={CONFIG.ns}, p={CONFIG.p}, map_kind={CONFIG.map_kind}, "
    f"eps={CONFIG.rotating_eps}, kappa={CONFIG.rotating_kappa}, nfp={CONFIG.rotating_nfp}"
)
K0_PLOT_DATA = build_k0_plot_data(SEQ, OPERATORS)
K0_STIFFNESS_PLOT_DATA = build_k0_stiffness_plot_data(SEQ, OPERATORS)


# %% Dense k=0 stiffness diagnostics for E G^T M1 G E^T
print("=" * 112)
print("k=0 extracted stiffness diagnostics: E G^T M1 G E^T")
for dirichlet in (False, True):
    print(f"dirichlet={dirichlet}")
    print_k0_extraction_split_info("mass", K0_PLOT_DATA[dirichlet])
    print_k0_extraction_split_info("stiffness", K0_STIFFNESS_PLOT_DATA[dirichlet])
    mass_diag = k0_split_diagnostics(
        K0_PLOT_DATA[dirichlet]["reordered"],
        K0_PLOT_DATA[dirichlet]["core_size"],
    )
    stiff_diag = k0_split_diagnostics(
        K0_STIFFNESS_PLOT_DATA[dirichlet]["reordered"],
        K0_STIFFNESS_PLOT_DATA[dirichlet]["core_size"],
    )
    print_k0_split_diagnostics("mass", mass_diag)
    print_k0_split_diagnostics("stiffness", stiff_diag)


# %% Pure-diagonal k=0 stiffness bulk model from mass-style tensor factors
K0_STIFFNESS_DIAGONAL_RANKS = (1, 3, 5)
K0_STIFFNESS_DIAGONAL_MODELS = {}
for dirichlet in (False, True):
    models = {}
    reports = []
    print("=" * 112)
    print(f"k=0 pure-diagonal stiffness surrogate: dirichlet={dirichlet}")
    for rank in K0_STIFFNESS_DIAGONAL_RANKS:
        model_data = build_k0_diagonal_stiffness_surrogate(SEQ, OPERATORS, dirichlet, rank)
        models[rank] = model_data
        reports.append(model_data["report"])
        print(f"rank={rank}: field-fit diagnostics")
        print_metric_factor_fit_reports(model_data["fit_reports"])
    K0_STIFFNESS_DIAGONAL_MODELS[dirichlet] = models
    print(f"rank summary: dirichlet={dirichlet}")
    print_k0_diagonal_stiffness_model_reports(reports)


# %% Benchmark k=0 stiffness preconditioners against Jacobi and diagonal bulk surrogates
K0_RICHARDSON_STEPS = (4, 8, 16)
K0_RICHARDSON_POWER_ITERATIONS = 20
K0_RICHARDSON_DAMPING_SAFETY = 0.8
K0_HODGE_BENCHMARKS = {}
for dirichlet in (False, True):
    reports = benchmark_k0_hodge_preconditioners(
        SEQ,
        K0_PLOT_DATA,
        K0_STIFFNESS_PLOT_DATA,
        K0_STIFFNESS_DIAGONAL_MODELS,
        dirichlet=dirichlet,
        n_rhs=8,
        seed=0,
        tensor_ranks=K0_STIFFNESS_DIAGONAL_RANKS,
        richardson_steps=K0_RICHARDSON_STEPS,
        richardson_power_iterations=K0_RICHARDSON_POWER_ITERATIONS,
        richardson_damping_safety=K0_RICHARDSON_DAMPING_SAFETY,
    )
    K0_HODGE_BENCHMARKS[dirichlet] = reports
    print("=" * 112)
    print(f"k=0 hodge-laplacian benchmark: dirichlet={dirichlet}")
    print_k0_benchmark_reports(reports)


# %% Spectrum diagnostics for the assembled k=0 stiffness preconditioners
for dirichlet in (False, True):
    reports = k0_stiffness_preconditioner_spectrum_diagnostics(
        K0_PLOT_DATA,
        K0_STIFFNESS_PLOT_DATA,
        K0_STIFFNESS_DIAGONAL_MODELS,
        dirichlet=dirichlet,
    )
    print("=" * 112)
    print(f"k=0 stiffness preconditioner spectrum: dirichlet={dirichlet}")
    print_k0_stiffness_preconditioner_spectrum_reports(reports)


# %% Metric-factor CP-ALS on the quadrature-grid Jacobian tensor
METRIC_CP_RANKS = (1, 2, 3, 4, 5)
K0_JACOBIAN_CP = {}
reports = []
jacobian_tensor = k0_metric_tensor(SEQ)
print("=" * 112)
print(f"k=0 metric CP-ALS diagnostics: field=jacobian, shape={jacobian_tensor.shape}")
for rank in METRIC_CP_RANKS:
    weights, factors, diagnostics = cp_als_3tensor(
        jacobian_tensor,
        rank,
        maxiter=100,
        tol=1e-9,
        ridge=1e-12,
    )
    diagnostics = MetricFactorFitDiagnostics(
        label="jacobian",
        rank=diagnostics.rank,
        relative_error=diagnostics.relative_error,
        max_abs_error=diagnostics.max_abs_error,
        n_iters=diagnostics.n_iters,
        final_delta=diagnostics.final_delta,
    )
    K0_JACOBIAN_CP[rank] = {
        "weights": weights,
        "factors": factors,
        "diagnostics": diagnostics,
    }
    reports.append(diagnostics)
print_metric_factor_fit_reports(reports)


# %% Assemble production tensor preconditioners once per rank
TENSOR_BENCHMARK_RANKS = (1, 3, 5)
TENSOR_OPERATOR_CACHE = {
    rank: assemble_tensor_mass_preconditioner(
        SEQ,
        operators=OPERATORS,
        ks=(0,),
        rank=rank,
        cp_kwargs=TENSOR_CP_KWARGS,
    )
    for rank in TENSOR_BENCHMARK_RANKS
}


# %% Compare exact M0^{-1} M03 against tensor-left-inverted M03
K0_TRANSFER_COMPARISONS = {}
for scalar_dirichlet in (False, True):
    k3_dirichlet = not scalar_dirichlet
    reports = compare_k0_transfer_operators(
        SEQ,
        OPERATORS,
        scalar_dirichlet=scalar_dirichlet,
        k3_dirichlet=k3_dirichlet,
        tensor_operator_cache=TENSOR_OPERATOR_CACHE,
        n_rhs=8,
        seed=0,
    )
    K0_TRANSFER_COMPARISONS[(scalar_dirichlet, k3_dirichlet)] = reports
    print("=" * 112)
    print(
        "k=0 transfer comparison: "
        f"scalar_dirichlet={scalar_dirichlet}, k3_dirichlet={k3_dirichlet}"
    )
    print_k0_transfer_comparison_reports(reports)


# %% Benchmark k=0 mass preconditioners against jacobi and tensor
K0_BENCHMARKS = {}
for dirichlet in (False, True):
    reports = benchmark_k0_preconditioners(
        SEQ,
        OPERATORS,
        dirichlet=dirichlet,
        n_rhs=8,
        seed=0,
        tensor_ranks=TENSOR_BENCHMARK_RANKS,
        richardson_steps=K0_RICHARDSON_STEPS,
        richardson_power_iterations=K0_RICHARDSON_POWER_ITERATIONS,
        richardson_damping_safety=K0_RICHARDSON_DAMPING_SAFETY,
        tensor_operator_cache=TENSOR_OPERATOR_CACHE,
    )
    K0_BENCHMARKS[dirichlet] = reports
    print("=" * 112)
    print(f"k=0 mass preconditioner benchmark: dirichlet={dirichlet}")
    print_k0_benchmark_reports(reports)


# %% Optional: visualize the bulk block A_bb
DIRICHLET_TO_PLOT = False
ABB_TO_PLOT = K0_PLOT_DATA[DIRICHLET_TO_PLOT]["bulk"]

fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(jnp.log10(jnp.abs(ABB_TO_PLOT) + 1e-16), cmap="viridis", origin="lower")
ax.set_title(f"log10 |A_bb|, dirichlet={DIRICHLET_TO_PLOT}")
fig.colorbar(im, ax=ax, shrink=0.8)
plt.show()


# %% Optional: visualize the extracted k=0 matrix with core separator
DIRICHLET_TO_PLOT = False
A_TO_PLOT = K0_PLOT_DATA[DIRICHLET_TO_PLOT]["reordered"]
CORE_SIZE = K0_PLOT_DATA[DIRICHLET_TO_PLOT]["core_size"]

fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(jnp.log10(jnp.abs(A_TO_PLOT) + 1e-16), cmap="viridis", origin="lower")
ax.axhline(CORE_SIZE - 0.5, color="white", linewidth=1.0)
ax.axvline(CORE_SIZE - 0.5, color="white", linewidth=1.0)
ax.set_title(f"log10 |P E0 M0 E0^T P^T|, dirichlet={DIRICHLET_TO_PLOT}")
fig.colorbar(im, ax=ax, shrink=0.8)
plt.show()


# %% Optional: visualize the k=0 stiffness bulk block K_bb
DIRICHLET_TO_PLOT = False
KBB_TO_PLOT = K0_STIFFNESS_PLOT_DATA[DIRICHLET_TO_PLOT]["bulk"]

fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(jnp.log10(jnp.abs(KBB_TO_PLOT) + 1e-16), cmap="viridis", origin="lower")
ax.set_title(f"log10 |K_bb|, dirichlet={DIRICHLET_TO_PLOT}")
fig.colorbar(im, ax=ax, shrink=0.8)
plt.show()


# %% Optional: visualize the pure-diagonal stiffness bulk surrogate
DIRICHLET_TO_PLOT = False
RANK_TO_PLOT = 3
KBB_MODEL_TO_PLOT = K0_STIFFNESS_DIAGONAL_MODELS[DIRICHLET_TO_PLOT][RANK_TO_PLOT]["bulk_model"]

fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(jnp.log10(jnp.abs(KBB_MODEL_TO_PLOT) + 1e-16), cmap="viridis", origin="lower")
ax.set_title(
    f"log10 |K_bb diagonal surrogate|, dirichlet={DIRICHLET_TO_PLOT}, rank={RANK_TO_PLOT}"
)
fig.colorbar(im, ax=ax, shrink=0.8)
plt.show()


# %% Optional: visualize the extracted k=0 stiffness matrix with core separator
DIRICHLET_TO_PLOT = False
K_TO_PLOT = K0_STIFFNESS_PLOT_DATA[DIRICHLET_TO_PLOT]["reordered"]
CORE_SIZE = K0_STIFFNESS_PLOT_DATA[DIRICHLET_TO_PLOT]["core_size"]

fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(jnp.log10(jnp.abs(K_TO_PLOT) + 1e-16), cmap="viridis", origin="lower")
ax.axhline(CORE_SIZE - 0.5, color="white", linewidth=1.0)
ax.axvline(CORE_SIZE - 0.5, color="white", linewidth=1.0)
ax.set_title(f"log10 |P E0 G^T M1 G E0^T P^T|, dirichlet={DIRICHLET_TO_PLOT}")
fig.colorbar(im, ax=ax, shrink=0.8)
plt.show()


# %% Optional: visualize the Jacobian tensor against rank-1 and rank-3 CP fits
METRIC_COMPARE_RANKS = (1, 3)
NZ_SLICES = jacobian_tensor.shape[2]
SLICE_Z_INDICES = (0, NZ_SLICES // 3, (2 * NZ_SLICES) // 3)
RADIAL_QUAD_GRID = SEQ.quad.x_x[None, :, None]

display_tensor = jacobian_tensor / RADIAL_QUAD_GRID
display_label = "jacobian / r"

raw_reconstructions = {
    rank: reconstruct_cp_3tensor(K0_JACOBIAN_CP[rank]["weights"], K0_JACOBIAN_CP[rank]["factors"])
    for rank in METRIC_COMPARE_RANKS
}
reconstructions = {
    rank: raw_reconstructions[rank] / RADIAL_QUAD_GRID
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
    error_slice = jnp.abs(raw_reconstructions[3][:, :, slice_z_index] - jacobian_tensor[:, :, slice_z_index])

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