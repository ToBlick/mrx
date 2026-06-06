# %% [markdown]
# # Mass Preconditioner Choices
#
# This interactive script assembles the production mass operators and
# compares the currently admitted mass-preconditioner options on the same CG
# solve for `k = 0, 1, 2, 3`. It also hosts the scalar `k = 0`
# Hodge/Laplacian benchmark, since that path shares the same scalar
# `MassPreconditionerSpec` interface.
#
# For `k = 0, 1, 2` the options follow the reduced `outer/schur/inner`
# interface; the plots display those choices as `outer/inner`, using `-` when
# the corresponding outer or inner layer is absent. Both `dirichlet=False`
# and `dirichlet=True` are benchmarked and plotted together.

# %%
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, fields as dataclass_fields, is_dataclass, replace as dataclass_replace
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import rotating_ellipse_map, toroid_map
from mrx.operators import (
    _estimate_chebyshev_lanczos_bounds_apply,
    _estimate_preconditioned_max_eigenvalue_apply,
    _hodge_diaginv,
    _mass_diaginv,
    apply_mass_matrix,
    apply_mass_tensor_preconditioner_ops,
    apply_stiffness,
    assemble_hodge_operators,
    assemble_incidence_operators,
    assemble_mass_operators,
    assemble_tensor_hodge_preconditioner,
    assemble_tensor_mass_preconditioner,
    dense_hodge_laplacian,
    dense_mass_matrix,
)
from mrx.nullspace import get_nullspace
from mrx.preconditioners import (
    MassPreconditionerSpec,
    _apply_k0_bulk_to_surgery_coupling,
    _apply_k0_surgery_to_bulk_coupling,
    _apply_k1_bulk_to_surgery_coupling,
    _apply_k1_rt_art_coupling,
    _apply_k1_rt_atr_coupling,
    _apply_k1_surgery_to_bulk_coupling,
    _apply_k2_bulk_to_surgery_coupling,
    _apply_k2_surgery_to_bulk_coupling,
    _apply_tensor_diagonal_block,
    _apply_tensor_exact_block,
    _assemble_schur_inverse_from_applies,
    _select_mass_surgery_factors,
    _select_mass_tensor_factors,
    get_mass_jacobi_diaginv,
)
from mrx.solvers import solve_singular_cg
from test.random_fields import build_random_besov_rhs_batch

jax.config.update("jax_enable_x64", True)


# %% Configuration
@dataclass(frozen=True)
class ExperimentConfig:
    ns: tuple[int, int, int] = (6, 8, 4)
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


def _parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value from {value!r}")


def _coerce_like(value, template):
    if isinstance(template, bool):
        return _parse_bool(value)
    if isinstance(template, tuple):
        parsed = value
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("["):
                parsed = json.loads(text)
            else:
                parsed = [part.strip() for part in text.split(",") if part.strip()]
        if not isinstance(parsed, (list, tuple)):
            raise ValueError(f"Expected a list/tuple override for {template!r}, got {value!r}")
        if len(template) == 0:
            return tuple(parsed)
        elem_template = template[0]
        return tuple(_coerce_like(item, elem_template) for item in parsed)
    if isinstance(template, int) and not isinstance(template, bool):
        return int(value)
    if isinstance(template, float):
        return float(value)
    return value


def _apply_override_mapping(current_values: dict[str, object], overrides: dict[str, object], *, label: str):
    for key, value in overrides.items():
        if key not in current_values:
            raise ValueError(f"Unknown {label} override {key!r}")
        current_values[key] = _coerce_like(value, current_values[key])


def _resolve_experiment_config(prefix: str, default: ExperimentConfig) -> ExperimentConfig:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config-json")
    args, _ = parser.parse_known_args()

    values = {
        field.name: getattr(default, field.name)
        for field in dataclass_fields(default)
    }

    env_json = os.getenv(f"{prefix}_CONFIG_JSON")
    if env_json:
        _apply_override_mapping(values, json.loads(env_json), label=f"{prefix} config")
    if args.config_json:
        _apply_override_mapping(values, json.loads(args.config_json), label="CLI config")

    for field in dataclass_fields(default):
        env_name = f"{prefix}_{field.name.upper()}"
        if env_name in os.environ:
            values[field.name] = _coerce_like(os.environ[env_name], values[field.name])

    return dataclass_replace(default, **values)


def _resolve_global_override(prefix: str, name: str, default):
    env_name = f"{prefix}_{name}"
    if env_name not in os.environ:
        return default
    return _coerce_like(os.environ[env_name], default)


@dataclass
class MassBenchmarkReport:
    k: int
    label: str
    outer: str
    inner: str
    reason: str
    dirichlet: bool
    n_rhs: int
    n_converged: int
    n_not_converged: int
    avg_iters: float
    std_iters: float
    max_iters: int
    avg_time_ms: float
    std_time_ms: float
    max_time_ms: float
    diagnostics: dict[str, object] | None = None


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


@dataclass
class K0LaplacianBenchmarkReport:
    label: str
    eps: float
    dirichlet: bool
    n_rhs: int
    n_converged: int
    n_not_converged: int
    avg_iters: float
    std_iters: float
    max_iters: int
    avg_time_ms: float
    std_time_ms: float
    max_time_ms: float
    avg_relative_residual: float
    std_relative_residual: float
    max_relative_residual: float
    diagnostics: dict[str, object] | None = None


@dataclass
class K0LaplacianSpectrumReport:
    dirichlet: bool
    size: int
    zero_count: int
    min_nonzero_eig: float
    max_eig: float
    nonzero_condition_number: float


CONFIG = _resolve_experiment_config("MRX_MASS", ExperimentConfig())
SEQ = None
OPERATORS = None
PRODUCTION_OPERATORS = None
BUILT_CONFIG = None
DENSE = _resolve_global_override("MRX_MASS", "DENSE", False)
EXACT_HYPERPARAMS = _resolve_global_override("MRX_MASS", "EXACT_HYPERPARAMS", False)
RUN_TEXT_SANITY = _resolve_global_override("MRX_MASS", "RUN_TEXT_SANITY", True)
TEXT_SANITY_PROBES = _resolve_global_override("MRX_MASS", "TEXT_SANITY_PROBES", 1)
DENSE_TEXT_SANITY = _resolve_global_override("MRX_MASS", "DENSE_TEXT_SANITY", False)
TENSOR_CP_KWARGS = {"tol": 1e-8, "maxiter": 200}
TENSOR_RANKS = _resolve_global_override("MRX_MASS", "TENSOR_RANKS", (1, 2, 4))
POLY_STEP_OPTIONS = _resolve_global_override("MRX_MASS", "POLY_STEP_OPTIONS", (1, 2, 4))
RHS_KIND = _resolve_global_override("MRX_MASS", "RHS_KIND", "besov")
RHS_S = _resolve_global_override("MRX_MASS", "RHS_S", 1.0)
RHS_UPPER_LIMIT = _resolve_global_override("MRX_MASS", "RHS_UPPER_LIMIT", 24)
RHS_NUM_MODES = _resolve_global_override("MRX_MASS", "RHS_NUM_MODES", 64)
RHS_SCALE = _resolve_global_override("MRX_MASS", "RHS_SCALE", 1.0)
RHS_SMOOTHNESS_MARGIN = _resolve_global_override("MRX_MASS", "RHS_SMOOTHNESS_MARGIN", 0.25)
RHS_NORMALIZATION_SAMPLES = _resolve_global_override("MRX_MASS", "RHS_NORMALIZATION_SAMPLES", 256)
INCLUDE_RICHARDSON = _resolve_global_override("MRX_MASS", "INCLUDE_RICHARDSON", False)
POWER_ITERATIONS = 30
RICHARDSON_DAMPING_SAFETY = 0.8
CHEBYSHEV_MIN_EIG_FRACTION = 1e-4
BENCHMARK_DIRICHLET_CASES = (False, )
DEFAULT_DIRICHLET = False
RUN_K0_LAPLACIAN = True
RUN_BLOCK_CHEB_BENCHMARK = True
K0_LAPLACE_EPS_LIST = (0.0,)
K0_LAPLACE_DIRICHLET_CASES = (True,)
K0_LAPLACE_NULLSPACE_EPS = 1e-6
MASS_METADATA_SETTING_NAMES = (
    "CONFIG",
    "DENSE",
    "EXACT_HYPERPARAMS",
    "RUN_TEXT_SANITY",
    "TEXT_SANITY_PROBES",
    "DENSE_TEXT_SANITY",
    "TENSOR_CP_KWARGS",
    "TENSOR_RANKS",
    "POLY_STEP_OPTIONS",
    "RHS_KIND",
    "RHS_S",
    "RHS_UPPER_LIMIT",
    "RHS_NUM_MODES",
    "RHS_SCALE",
    "RHS_SMOOTHNESS_MARGIN",
    "RHS_NORMALIZATION_SAMPLES",
    "INCLUDE_RICHARDSON",
    "POWER_ITERATIONS",
    "RICHARDSON_DAMPING_SAFETY",
    "CHEBYSHEV_MIN_EIG_FRACTION",
    "BENCHMARK_DIRICHLET_CASES",
    "DEFAULT_DIRICHLET",
    "RUN_K0_LAPLACIAN",
    "RUN_BLOCK_CHEB_BENCHMARK",
    "K0_LAPLACE_EPS_LIST",
    "K0_LAPLACE_DIRICHLET_CASES",
    "K0_LAPLACE_NULLSPACE_EPS",
)


# %% Helpers
def _serialize_metadata_value(value):
    if is_dataclass(value):
        return {key: _serialize_metadata_value(val) for key, val in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _serialize_metadata_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_metadata_value(val) for val in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _mass_run_metadata(
    benchmarks: dict[tuple[int, bool], list[MassBenchmarkReport]],
    k0_laplacian_benchmarks: dict[tuple[float, bool], list[K0LaplacianBenchmarkReport]] | None,
):
    metadata = {
        "script": "mass_preconditioner_demo",
        "experiment_config": _serialize_metadata_value(CONFIG),
        "script_hyperparameters": {
            name: _serialize_metadata_value(globals()[name])
            for name in MASS_METADATA_SETTING_NAMES
        },
        "benchmark_cases": [
            {"family": "mass", "k": k, "dirichlet": dirichlet}
            for k, dirichlet in _available_benchmark_cases(benchmarks)
        ],
    }
    if k0_laplacian_benchmarks:
        metadata["benchmark_cases"].extend([
            {"family": "k0_laplacian", "eps": eps, "dirichlet": dirichlet}
            for eps, dirichlet in sorted(k0_laplacian_benchmarks)
        ])
    return metadata


def _build_rhs_batch(seq, *, k: int, dirichlet: bool, n_rhs: int, seed: int) -> jnp.ndarray:
    rhs_seed = seed + 100 * int(dirichlet) + 1000 * k
    rhs_size = _mass_rhs_size(seq, k, dirichlet)
    if RHS_KIND == "gaussian":
        return jax.random.normal(
            jax.random.PRNGKey(rhs_seed),
            (n_rhs, rhs_size),
            dtype=jnp.float64,
        )
    if RHS_KIND == "besov":
        return build_random_besov_rhs_batch(
            seq,
            k,
            dirichlet=dirichlet,
            n_rhs=n_rhs,
            seed=rhs_seed,
            s=RHS_S,
            upper_limit=RHS_UPPER_LIMIT,
            num_modes=RHS_NUM_MODES,
            scale=RHS_SCALE,
            smoothness_margin=RHS_SMOOTHNESS_MARGIN,
            normalization_samples=RHS_NORMALIZATION_SAMPLES,
        )
    raise ValueError(f"Unsupported RHS kind: {RHS_KIND!r}")


def _float_or_none(value):
    if value is None:
        return None
    return float(jnp.asarray(value, dtype=jnp.float64))


def _drop_empty_values(mapping: dict[str, object]) -> dict[str, object]:
    cleaned = {}
    for key, value in mapping.items():
        if value is None:
            continue
        if isinstance(value, dict) and not value:
            continue
        cleaned[key] = value
    return cleaned


def _matrix_from_apply(apply, size: int) -> jnp.ndarray:
    basis = jnp.eye(size, dtype=jnp.float64)
    return jax.vmap(apply, in_axes=1, out_axes=1)(basis)


def _relative_fro_error(approx: jnp.ndarray, exact: jnp.ndarray) -> float:
    denom = jnp.linalg.norm(exact)
    denom = jnp.where(denom > 0, denom, 1.0)
    return float(jnp.linalg.norm(approx - exact) / denom)


def _sampled_identity_sanity(
    seq,
    operators,
    *,
    k: int,
    dirichlet: bool,
    probe_batch: jnp.ndarray,
    precond_apply,
) -> dict[str, float]:
    def operator_apply(x):
        return apply_mass_matrix(seq, operators, x, k, dirichlet=dirichlet)

    n_probes = max(1, min(int(TEXT_SANITY_PROBES), int(probe_batch.shape[0])))
    pm_errors = []
    mp_errors = []
    for rhs in probe_batch[:n_probes]:
        denom = jnp.where(jnp.linalg.norm(rhs) > 0, jnp.linalg.norm(rhs), 1.0)
        pm_errors.append(float(jnp.linalg.norm(precond_apply(operator_apply(rhs)) - rhs) / denom))
        mp_errors.append(float(jnp.linalg.norm(operator_apply(precond_apply(rhs)) - rhs) / denom))
    pm_errors = jnp.asarray(pm_errors, dtype=jnp.float64)
    mp_errors = jnp.asarray(mp_errors, dtype=jnp.float64)
    return {
        "rel_pm_identity_median": float(jnp.median(pm_errors)),
        "rel_pm_identity_max": float(jnp.max(pm_errors)),
        "rel_mp_identity_median": float(jnp.median(mp_errors)),
        "rel_mp_identity_max": float(jnp.max(mp_errors)),
    }


def _should_collect_text_sanity(label: str) -> bool:
    if label == "jacobi":
        return True
    if label.startswith("tensor-r"):
        return True
    if label.startswith("schur-block-cheb"):
        return True
    return False


def _dense_identity_sanity(matrix: jnp.ndarray, precond_apply) -> dict[str, float]:
    precond_matrix = _matrix_from_apply(precond_apply, matrix.shape[0])
    identity = jnp.eye(matrix.shape[0], dtype=jnp.float64)
    pm = precond_matrix @ matrix
    mp = matrix @ precond_matrix
    pm_sym = 0.5 * (pm + pm.T)
    eigvals = jnp.linalg.eigvalsh(pm_sym)
    tiny = jnp.asarray(jnp.finfo(jnp.float64).tiny, dtype=jnp.float64)
    eig_abs_min = jnp.maximum(jnp.min(jnp.abs(eigvals)), tiny)
    eig_abs_max = jnp.maximum(jnp.max(jnp.abs(eigvals)), eig_abs_min)
    return {
        "rel_pm_identity": _relative_fro_error(pm, identity),
        "rel_mp_identity": _relative_fro_error(mp, identity),
        "pm_symmetry_defect": _relative_fro_error(pm, pm.T),
        "pm_sym_eig_min": float(jnp.min(eigvals)),
        "pm_sym_eig_max": float(jnp.max(eigvals)),
        "pm_sym_condition_number": float(eig_abs_max / eig_abs_min),
    }


def _mass_preconditioner_apply_from_spec(
    seq,
    operators,
    *,
    k: int,
    dirichlet: bool,
    preconditioner,
):
    if not isinstance(preconditioner, MassPreconditionerSpec):
        return None
    if preconditioner.kind == "jacobi" and not preconditioner.surgery_schur:
        diaginv = _mass_diaginv(operators, k, dirichlet)
        return lambda x, inv=diaginv: inv * x
    if preconditioner.kind == "tensor":
        return lambda x: apply_mass_tensor_preconditioner_ops(
            seq,
            operators,
            x,
            k,
            dirichlet=dirichlet,
        )
    smoother = preconditioner.smoother
    if (
        preconditioner.surgery_schur
        and preconditioner.kind == "none"
        and smoother is not None
        and smoother.kind == "tensor"
    ):
        return lambda x: apply_mass_tensor_preconditioner_ops(
            seq,
            operators,
            x,
            k,
            dirichlet=dirichlet,
        )
    return None


def _format_metric(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3e}"


def _print_mass_preconditioner_sanity(reports: list[MassBenchmarkReport]):
    rows = []
    for report in reports:
        diagnostics = report.diagnostics or {}
        sampled = diagnostics.get("sampled_identity")
        dense = diagnostics.get("dense_identity")
        if sampled is None and dense is None:
            continue
        rows.append((report.label, sampled, dense))
    if not rows:
        return

    print("preconditioner-apply sanity:")
    print(
        f"{'label':<24} {'med ||PM-I||':>12} {'max ||PM-I||':>12} {'med ||MP-I||':>12} {'max ||MP-I||':>12} {'dense PM':>12} {'dense MP':>12} {'asym(PM)':>12}"
    )
    for label, sampled, dense in rows:
        print(
            f"{label:<24} "
            f"{_format_metric(None if sampled is None else sampled.get('rel_pm_identity_median')):>12} "
            f"{_format_metric(None if sampled is None else sampled.get('rel_pm_identity_max')):>12} "
            f"{_format_metric(None if sampled is None else sampled.get('rel_mp_identity_median')):>12} "
            f"{_format_metric(None if sampled is None else sampled.get('rel_mp_identity_max')):>12} "
            f"{_format_metric(None if dense is None else dense.get('rel_pm_identity')):>12} "
            f"{_format_metric(None if dense is None else dense.get('rel_mp_identity')):>12} "
            f"{_format_metric(None if dense is None else dense.get('pm_symmetry_defect')):>12}"
        )
    if any(dense is not None for _, _, dense in rows):
        print("  dense PM/MP columns are full-operator Frobenius defects; asym(PM) is ||PM - (PM)^T|| / ||(PM)^T||.")
    print(f"  sampled columns use the first {int(TEXT_SANITY_PROBES)} probe RHS vectors.")


def _tensor_block_fit_diagnostics(block) -> dict[str, float] | None:
    diagnostics = _drop_empty_values(
        {
            "relative_error": _float_or_none(getattr(block, "cp_relative_error", None)),
            "final_delta": _float_or_none(getattr(block, "cp_final_delta", None)),
            "block_chebyshev_steps": _float_or_none(getattr(block, "chebyshev_steps", None)),
            "block_lambda_min": _float_or_none(getattr(block, "chebyshev_lambda_min", None)),
            "block_lambda_max": _float_or_none(getattr(block, "chebyshev_lambda_max", None)),
        }
    )
    return diagnostics or None


def _mass_tensor_fit_diagnostics(operators, *, k: int, dirichlet: bool) -> dict[str, object] | None:
    preconds = operators.mass_preconds
    if preconds is None or preconds.tensor is None:
        return None
    try:
        factors = _select_mass_tensor_factors(preconds, k, dirichlet)
    except ValueError:
        return None
    if k == 0:
        return _tensor_block_fit_diagnostics(factors.bulk)
    if k == 1:
        diagnostics = _drop_empty_values(
            {
                "arr": _tensor_block_fit_diagnostics(factors.arr),
                "theta": _tensor_block_fit_diagnostics(factors.theta),
                "zeta": _tensor_block_fit_diagnostics(factors.zeta),
            }
        )
        return diagnostics or None
    if k == 2:
        diagnostics = _drop_empty_values(
            {
                "r_bulk": _tensor_block_fit_diagnostics(factors.r_bulk),
                "theta": _tensor_block_fit_diagnostics(factors.theta),
                "zeta": _tensor_block_fit_diagnostics(factors.zeta),
            }
        )
        return diagnostics or None
    if k == 3:
        return _tensor_block_fit_diagnostics(factors)
    return None


def _nullspace_rows(operators, *, k: int, dirichlet: bool):
    try:
        nullspace = get_nullspace(operators, k, dirichlet)
    except ValueError:
        return None
    return nullspace if nullspace.shape[0] > 0 else None


def _iterative_tuning_diagnostics(
    operator_apply,
    smoother_apply,
    *,
    size: int,
    spec: MassPreconditionerSpec,
    seed: int,
    smoother_kind: str,
    source: str,
    orthogonal_vectors=None,
) -> dict[str, object]:
    max_eig = _estimate_preconditioned_max_eigenvalue_apply(
        operator_apply,
        smoother_apply,
        size,
        n_iter=spec.power_iterations,
        seed=seed,
    )
    if spec.kind == "richardson":
        omega = jnp.where(
            max_eig > 0.0,
            jnp.asarray(spec.damping_safety, dtype=jnp.float64) / max_eig,
            jnp.asarray(1.0, dtype=jnp.float64),
        )
        return {
            "method": "richardson",
            "source": source,
            "smoother": smoother_kind,
            "max_eig": float(max_eig),
            "omega": float(omega),
        }
    min_eig, max_eig = _estimate_chebyshev_lanczos_bounds_apply(
        operator_apply,
        smoother_apply,
        size,
        spec=spec,
        seed=seed,
        orthogonal_vectors=orthogonal_vectors,
    )
    return {
        "method": "chebyshev",
        "source": source,
        "smoother": smoother_kind,
        "min_eig": float(min_eig),
        "max_eig": float(max_eig),
    }


def _mass_iterative_smoother_apply(seq, operators, *, k: int, dirichlet: bool, spec: MassPreconditionerSpec):
    smoother_spec = spec.smoother
    if smoother_spec is None:
        smoother_spec = MassPreconditionerSpec(kind="tensor")
    if smoother_spec.kind == "tensor":
        return (
            lambda x: apply_mass_tensor_preconditioner_ops(
                seq,
                operators,
                x,
                k,
                dirichlet=dirichlet,
            ),
            "tensor",
        )
    if smoother_spec.kind == "jacobi":
        diaginv = _mass_diaginv(operators, k, dirichlet)
        return (lambda x, inv=diaginv: inv * x, "jacobi")
    if smoother_spec.kind == "none":
        return (lambda x: x, "none")
    raise ValueError(f"Unsupported iterative smoother kind {smoother_spec.kind!r}")


def _mass_preconditioner_diagnostics(
    seq,
    operators,
    *,
    k: int,
    dirichlet: bool,
    preconditioner,
    exact_tuning: dict[str, float] | None = None,
) -> dict[str, object] | None:
    if not isinstance(preconditioner, MassPreconditionerSpec):
        return None

    diagnostics: dict[str, object] = {}
    uses_tensor = preconditioner.kind == "tensor" or preconditioner.surgery_schur
    if preconditioner.kind in ("richardson", "chebyshev"):
        if exact_tuning is not None and _uses_plain_jacobi_smoother(preconditioner):
            tuning = {
                "method": preconditioner.kind,
                "source": "dense_exact_jacobi",
                "smoother": "jacobi",
                "max_eig": exact_tuning["lambda_max"],
            }
            if preconditioner.kind == "richardson":
                tuning["omega"] = exact_tuning["omega"]
            else:
                tuning["min_eig"] = exact_tuning["lambda_min"]
            diagnostics["tuning"] = tuning
        else:
            operator_apply = lambda x: apply_mass_matrix(seq, operators, x, k, dirichlet=dirichlet)
            smoother_apply, smoother_kind = _mass_iterative_smoother_apply(
                seq,
                operators,
                k=k,
                dirichlet=dirichlet,
                spec=preconditioner,
            )
            diagnostics["tuning"] = _iterative_tuning_diagnostics(
                operator_apply,
                smoother_apply,
                size=_mass_rhs_size(seq, k, dirichlet),
                spec=preconditioner,
                seed=1000 * k + int(dirichlet),
                smoother_kind=smoother_kind,
                source="estimated_runtime",
            )
            uses_tensor = uses_tensor or smoother_kind == "tensor"

    if uses_tensor:
        tensor_fit = _mass_tensor_fit_diagnostics(operators, k=k, dirichlet=dirichlet)
        if tensor_fit is not None:
            diagnostics["tensor_fit"] = tensor_fit

    return diagnostics or None


def _scalar_hodge_preconditioner_diagnostics(
    seq,
    operators,
    *,
    eps: float,
    dirichlet: bool,
    preconditioner,
) -> dict[str, object] | None:
    if not isinstance(preconditioner, MassPreconditionerSpec):
        return None
    if preconditioner.kind not in ("richardson", "chebyshev"):
        return None

    stiffness_diaginv = _hodge_diaginv(operators, 0, dirichlet)
    if eps == 0.0:
        shifted_diaginv = stiffness_diaginv
    else:
        mass_diaginv = _mass_diaginv(operators, 0, dirichlet)
        shifted_diaginv = 1.0 / (1.0 / stiffness_diaginv + eps / mass_diaginv)

    def operator_apply(x):
        result = apply_stiffness(seq, operators, x, 0, dirichlet=dirichlet)
        if eps > 0.0:
            result = result + eps * apply_mass_matrix(seq, operators, x, 0, dirichlet=dirichlet)
        return result

    smoother_apply = lambda x, inv=shifted_diaginv: inv * x
    return {
        "tuning": _iterative_tuning_diagnostics(
            operator_apply,
            smoother_apply,
            size=seq.n0_dbc if dirichlet else seq.n0,
            spec=preconditioner,
            seed=int(dirichlet),
            smoother_kind="shifted_jacobi" if eps > 0.0 else "jacobi",
            source="estimated_runtime",
            orthogonal_vectors=_nullspace_rows(operators, k=0, dirichlet=dirichlet) if eps == 0.0 else None,
        )
    }


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
    operators = assemble_hodge_operators(seq, seq.geometry, operators=operators, ks=(0, 1, 2, 3))
    operators = seq.set_operators(operators, sync_legacy=False)
    if (
        RUN_K0_LAPLACIAN
        and any(not dirichlet for dirichlet in K0_LAPLACE_DIRICHLET_CASES)
        and any(eps == 0.0 for eps in K0_LAPLACE_EPS_LIST)
    ):
        seq._compute_nullspaces(config.betti, eps=K0_LAPLACE_NULLSPACE_EPS)
        operators = seq._require_operators()
    tensor_operators = {
        rank: None
        for rank in TENSOR_RANKS
    }
    for rank in TENSOR_RANKS:
        rank_operators = assemble_tensor_mass_preconditioner(
            seq,
            operators=operators,
            ks=(0, 1, 2, 3),
            rank=rank,
            cp_kwargs=TENSOR_CP_KWARGS,
        )
        rank_operators = assemble_tensor_hodge_preconditioner(
            seq,
            operators=rank_operators,
            rank=rank,
            cp_maxiter=TENSOR_CP_KWARGS["maxiter"],
            cp_tol=TENSOR_CP_KWARGS["tol"],
            cp_ridge=TENSOR_CP_KWARGS.get("ridge", 1e-12),
        )
        rank_operators = assemble_hodge_operators(
            seq,
            seq.geometry,
            operators=rank_operators,
            ks=(0,),
        )
        tensor_operators[rank] = rank_operators
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


def _scalar_polynomial_mass_preconditioners():
    configs = []
    if INCLUDE_RICHARDSON:
        for steps in POLY_STEP_OPTIONS:
            configs.append(
                (
                    f"richardson-{steps}",
                    "richardson",
                    dict(
                        steps=steps,
                        power_iterations=POWER_ITERATIONS,
                        damping_safety=RICHARDSON_DAMPING_SAFETY,
                    ),
                )
            )
    for steps in POLY_STEP_OPTIONS:
        configs.append(
            (
                f"chebyshev-{steps}",
                "chebyshev",
                dict(
                    steps=steps,
                    power_iterations=POWER_ITERATIONS,
                    min_eig_fraction=CHEBYSHEV_MIN_EIG_FRACTION,
                ),
            )
        )
    return configs


def _display_outer(outer: str) -> str:
    return "-" if outer in ("-", "none") else outer


def _display_inner(inner: str) -> str:
    return "-" if inner in ("-", "none") else inner


def _format_option_label(*, outer: str, inner: str) -> str:
    display_outer = _display_outer(outer)
    display_inner = _display_inner(inner)
    if display_outer != "-":
        return display_outer
    if display_inner != "-":
        return display_inner
    return "-"


def _tensor_rank_label(rank: int) -> str:
    return f"tensor-r{rank}"


def _scalar_schur_mass_preconditioner_catalog():
    def _entry(*, outer, schur, inner, reason, preconditioner, label=None, tensor_rank=None):
        return {
            "outer": outer,
            "inner": inner,
            "schur": schur,
            "label": label if label is not None else _format_option_label(outer=outer, inner=inner),
            "preconditioner": preconditioner,
            "reason": reason,
            "tensor_rank": tensor_rank,
        }

    def _inner_configs():
        return [("tensor", MassPreconditionerSpec(kind="tensor"))]

    def _schur_spec(outer, inner_spec, **kwargs):
        if outer == "none":
            return MassPreconditionerSpec(
                kind="none",
                surgery_schur=True,
                smoother=inner_spec,
            )
        return MassPreconditionerSpec(
            kind=outer,
            surgery_schur=True,
            smoother=inner_spec,
            **kwargs,
        )

    catalog = [
        _entry(
            outer="jacobi",
            schur=False,
            inner="-",
            reason="plain diagonal inverse",
            preconditioner=MassPreconditionerSpec(kind="jacobi", surgery_schur=False),
            label="jacobi",
        ),
    ]

    for label, kind, kwargs in _scalar_polynomial_mass_preconditioners():
        catalog.append(
            _entry(
                outer=kind,
                schur=False,
                inner="-",
                reason="full-operator polynomial acceleration with tensor smoother (no surgery)",
                preconditioner=MassPreconditionerSpec(
                    kind=kind,
                    surgery_schur=False,
                    **kwargs,
                ),
                label=label,
            )
        )

    for rank in TENSOR_RANKS:
        for inner_label, inner_spec in _inner_configs():
            catalog.append(
                _entry(
                    outer="none",
                    schur=True,
                    inner=inner_label,
                    reason="legacy tensor Schur route",
                    preconditioner=_schur_spec("none", inner_spec),
                    label=_tensor_rank_label(rank),
                    tensor_rank=rank,
                )
            )

    return catalog


def _k3_mass_preconditioner_catalog():
    catalog = [
        {
            "outer": "-",
            "inner": "jacobi",
            "schur": False,
            "label": "jacobi",
            "preconditioner": MassPreconditionerSpec(kind="jacobi"),
            "reason": "plain diagonal inverse",
            "tensor_rank": None,
        },
    ]
    for label, kind, kwargs in _scalar_polynomial_mass_preconditioners():
        catalog.append(
            {
                "outer": "-",
                "inner": kind,
                "schur": False,
                "label": label,
                "preconditioner": MassPreconditionerSpec(kind=kind, **kwargs),
                "reason": "polynomial iteration on the direct scalar mass operator",
                "tensor_rank": None,
            }
        )
    for rank in TENSOR_RANKS:
        catalog.append(
            {
                "outer": "-",
                "inner": "tensor",
                "schur": False,
                "label": _tensor_rank_label(rank),
                "preconditioner": MassPreconditionerSpec(kind="tensor"),
                "reason": "direct scalar tensor inverse",
                "tensor_rank": rank,
            }
        )
    return catalog


def _mass_preconditioner_catalog(k: int):
    if k in (0, 1, 2):
        return _scalar_schur_mass_preconditioner_catalog()
    if k == 3:
        return _k3_mass_preconditioner_catalog()
    raise ValueError(f"Unsupported degree k={k}")


def _uses_plain_jacobi_smoother(spec: MassPreconditionerSpec) -> bool:
    smoother = spec.smoother
    return (
        smoother is not None
        and smoother.kind == "jacobi"
        and not smoother.surgery_schur
        and smoother.smoother is None
    )


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


def _build_dense_block_chebyshev_apply(
    block_matrix,
    smoother_apply,
    *,
    steps: int,
    seed: int,
    smoother_kind: str,
):
    spec = MassPreconditionerSpec(
        kind="chebyshev",
        steps=steps,
        power_iterations=POWER_ITERATIONS,
        min_eig_fraction=CHEBYSHEV_MIN_EIG_FRACTION,
    )
    operator_apply = lambda x, matrix=block_matrix: matrix @ x
    lambda_min, lambda_max = _estimate_chebyshev_lanczos_bounds_apply(
        operator_apply,
        smoother_apply,
        block_matrix.shape[0],
        spec=spec,
        seed=seed,
    )
    return (
        _build_tuned_chebyshev_apply_preconditioner(
            operator_apply,
            smoother_apply,
            steps=steps,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
        ),
        {
            "method": "chebyshev",
            "source": "estimated_runtime",
            "smoother": smoother_kind,
            "steps": steps,
            "lambda_min": float(lambda_min),
            "lambda_max": float(lambda_max),
        },
    )


def _stored_tensor_block_tuning(block):
    return _drop_empty_values(
        {
            "method": "chebyshev",
            "source": "assembled_tensor_block",
            "smoother": "tensor",
            "steps": int(getattr(block, "chebyshev_steps", 0)),
            "lambda_min": _float_or_none(getattr(block, "chebyshev_lambda_min", None)),
            "lambda_max": _float_or_none(getattr(block, "chebyshev_lambda_max", None)),
        }
    )


def _build_exact_block_chebyshev_mass_preconditioner_apply(
    seq,
    operators,
    *,
    k: int,
    dirichlet: bool,
    smoother_kind: str,
):
    surgery = _select_mass_surgery_factors(operators.mass_preconds, k, dirichlet)
    tensor = _select_mass_tensor_factors(operators.mass_preconds, k, dirichlet)
    block_steps = int(operators.mass_preconds.tensor.block_chebyshev_steps)
    if block_steps <= 0:
        raise ValueError("Tensor block Chebyshev benchmark requires block_chebyshev_steps > 0")
    diaginv = get_mass_jacobi_diaginv(operators.mass_preconds, k, dirichlet)
    diagnostics: dict[str, object] = {"inner_block_tuning": {}}
    exact_matrix = jnp.asarray(dense_mass_matrix(seq, operators, k, dirichlet=dirichlet))

    if k == 0:
        bulk_matrix = exact_matrix[surgery.surgery_size:, surgery.surgery_size:]
        if smoother_kind == "jacobi":
            bulk_apply, bulk_tuning = _build_dense_block_chebyshev_apply(
                bulk_matrix,
                lambda x, inv=diaginv[surgery.surgery_size:]: inv * x,
                steps=block_steps,
                seed=8100 + int(dirichlet),
                smoother_kind="jacobi",
            )
        else:
            bulk_apply = lambda rhs: _apply_tensor_exact_block(None, tensor.bulk, rhs)
            bulk_tuning = _stored_tensor_block_tuning(tensor.bulk)

        diagnostics["inner_block_tuning"]["bulk"] = bulk_tuning
        diagnostics["tensor_fit"] = _mass_tensor_fit_diagnostics(operators, k=k, dirichlet=dirichlet)
        surgery_to_bulk_apply = lambda rhs_s: _apply_k0_surgery_to_bulk_coupling(surgery, rhs_s)
        bulk_to_surgery_apply = lambda rhs_b: _apply_k0_bulk_to_surgery_coupling(surgery, rhs_b)
        schur_inv = _assemble_schur_inverse_from_applies(
            surgery.ass,
            surgery_to_bulk_apply,
            bulk_apply,
            bulk_to_surgery_apply,
        )

        def apply(rhs):
            rhs_s = rhs[:surgery.surgery_size]
            rhs_b = rhs[surgery.surgery_size:]
            y = bulk_apply(rhs_b)
            z = schur_inv @ (rhs_s - bulk_to_surgery_apply(y))
            x_b = y - bulk_apply(surgery_to_bulk_apply(z))
            return jnp.concatenate([z, x_b])

        return apply, diagnostics

    if k == 1:
        arr_matrix = exact_matrix[surgery.r_indices][:, surgery.r_indices]
        theta_matrix = exact_matrix[surgery.theta_bulk_indices][:, surgery.theta_bulk_indices]
        zeta_matrix = exact_matrix[surgery.zeta_bulk_indices][:, surgery.zeta_bulk_indices]
        if smoother_kind == "jacobi":
            r_apply, r_tuning = _build_dense_block_chebyshev_apply(
                arr_matrix,
                lambda x, inv=diaginv[tensor.r_indices]: inv * x,
                steps=block_steps,
                seed=8200 + int(dirichlet),
                smoother_kind="jacobi",
            )
            theta_apply, theta_tuning = _build_dense_block_chebyshev_apply(
                theta_matrix,
                lambda x, inv=diaginv[tensor.theta_bulk_indices]: inv * x,
                steps=block_steps,
                seed=8201 + int(dirichlet),
                smoother_kind="jacobi",
            )
            zeta_apply, zeta_tuning = _build_dense_block_chebyshev_apply(
                zeta_matrix,
                lambda x, inv=diaginv[tensor.zeta_bulk_indices]: inv * x,
                steps=block_steps,
                seed=8202 + int(dirichlet),
                smoother_kind="jacobi",
            )
        else:
            r_apply = lambda rhs: _apply_tensor_exact_block(None, tensor.arr, rhs)
            theta_apply = lambda rhs: _apply_tensor_exact_block(None, tensor.theta, rhs)
            zeta_apply = lambda rhs: _apply_tensor_exact_block(None, tensor.zeta, rhs)
            r_tuning = _stored_tensor_block_tuning(tensor.arr)
            theta_tuning = _stored_tensor_block_tuning(tensor.theta)
            zeta_tuning = _stored_tensor_block_tuning(tensor.zeta)

        diagnostics["inner_block_tuning"] = {
            "arr": r_tuning,
            "theta": theta_tuning,
            "zeta": zeta_tuning,
        }
        diagnostics["tensor_fit"] = _mass_tensor_fit_diagnostics(operators, k=k, dirichlet=dirichlet)

        def rt_apply(rhs_rt):
            rhs_r = rhs_rt[:surgery.rt_r_size]
            rhs_theta = rhs_rt[surgery.rt_r_size:surgery.rt_r_size + surgery.rt_theta_size]
            y = r_apply(rhs_r)
            z = theta_apply(rhs_theta - _apply_k1_rt_atr_coupling(surgery, y))
            x_r = y - r_apply(_apply_k1_rt_art_coupling(surgery, z))
            return jnp.concatenate([x_r, z])

        def bulk_apply(rhs_bulk):
            rhs_rt = rhs_bulk[:surgery.bulk_rt_size]
            rhs_zeta = rhs_bulk[surgery.bulk_rt_size:surgery.bulk_rt_size + surgery.bulk_zeta_size]
            return jnp.concatenate([rt_apply(rhs_rt), zeta_apply(rhs_zeta)])
        surgery_to_bulk_apply = lambda rhs_s: _apply_k1_surgery_to_bulk_coupling(surgery, rhs_s)
        bulk_to_surgery_apply = lambda rhs_b: _apply_k1_bulk_to_surgery_coupling(surgery, rhs_b)
        schur_inv = _assemble_schur_inverse_from_applies(
            surgery.ass,
            surgery_to_bulk_apply,
            bulk_apply,
            bulk_to_surgery_apply,
        )

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

        return apply, diagnostics

    if k == 2:
        r_bulk_matrix = exact_matrix[surgery.r_bulk_indices][:, surgery.r_bulk_indices]
        theta_matrix = exact_matrix[surgery.theta_indices][:, surgery.theta_indices]
        zeta_matrix = exact_matrix[surgery.zeta_indices][:, surgery.zeta_indices]
        if smoother_kind == "jacobi":
            r_apply, r_tuning = _build_dense_block_chebyshev_apply(
                r_bulk_matrix,
                lambda x, inv=diaginv[tensor.r_bulk_indices]: inv * x,
                steps=block_steps,
                seed=8300 + int(dirichlet),
                smoother_kind="jacobi",
            )
            theta_apply, theta_tuning = _build_dense_block_chebyshev_apply(
                theta_matrix,
                lambda x, inv=diaginv[tensor.theta_indices]: inv * x,
                steps=block_steps,
                seed=8301 + int(dirichlet),
                smoother_kind="jacobi",
            )
            zeta_apply, zeta_tuning = _build_dense_block_chebyshev_apply(
                zeta_matrix,
                lambda x, inv=diaginv[tensor.zeta_indices]: inv * x,
                steps=block_steps,
                seed=8302 + int(dirichlet),
                smoother_kind="jacobi",
            )
        else:
            r_apply = lambda rhs: _apply_tensor_exact_block(None, tensor.r_bulk, rhs)
            theta_apply = lambda rhs: _apply_tensor_exact_block(None, tensor.theta, rhs)
            zeta_apply = lambda rhs: _apply_tensor_exact_block(None, tensor.zeta, rhs)
            r_tuning = _stored_tensor_block_tuning(tensor.r_bulk)
            theta_tuning = _stored_tensor_block_tuning(tensor.theta)
            zeta_tuning = _stored_tensor_block_tuning(tensor.zeta)

        diagnostics["inner_block_tuning"] = {
            "r_bulk": r_tuning,
            "theta": theta_tuning,
            "zeta": zeta_tuning,
        }
        diagnostics["tensor_fit"] = _mass_tensor_fit_diagnostics(operators, k=k, dirichlet=dirichlet)

        def bulk_apply(rhs_bulk):
            rhs_r = rhs_bulk[:surgery.r_bulk_size]
            rhs_theta = rhs_bulk[surgery.r_bulk_size:surgery.r_bulk_size + surgery.theta_size]
            rhs_zeta = rhs_bulk[surgery.r_bulk_size + surgery.theta_size:surgery.r_bulk_size + surgery.theta_size + surgery.zeta_size]
            return jnp.concatenate([r_apply(rhs_r), theta_apply(rhs_theta), zeta_apply(rhs_zeta)])
        surgery_to_bulk_apply = lambda rhs_s: _apply_k2_surgery_to_bulk_coupling(surgery, rhs_s)
        bulk_to_surgery_apply = lambda rhs_b: _apply_k2_bulk_to_surgery_coupling(surgery, rhs_b)
        schur_inv = _assemble_schur_inverse_from_applies(
            surgery.ass,
            surgery_to_bulk_apply,
            bulk_apply,
            bulk_to_surgery_apply,
        )

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

        return apply, diagnostics

    raise ValueError(f"Exact block Chebyshev benchmark only supports k=0,1,2 (got k={k})")


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


def _k0_laplacian_preconditioner_catalog():
    return _scalar_schur_mass_preconditioner_catalog()


def _k0_laplacian_labels() -> list[str]:
    labels = [entry["label"] for entry in _k0_laplacian_preconditioner_catalog()]
    if DENSE:
        labels.append("dense-pinv")
    return labels


def _k0_nullspace_vector(operators, dirichlet: bool):
    if dirichlet:
        return None
    try:
        nullspace = get_nullspace(operators, 0, dirichlet)
    except ValueError:
        return None
    if nullspace.shape[0] == 0:
        return None
    return nullspace[0]


def _project_k0_laplacian_rhs_to_range(seq, operators, rhs, dirichlet: bool):
    null_vector = _k0_nullspace_vector(operators, dirichlet)
    if null_vector is None:
        return rhs
    return rhs - jnp.dot(null_vector, rhs) * apply_mass_matrix(
        seq, operators, null_vector, 0, dirichlet=dirichlet
    )


def _dense_k0_hodge_inverse(
    seq,
    operators,
    dirichlet: bool,
    *,
    eps: float = 0.0,
    rtol: float = 1e-12,
):
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
    operators_by_rank,
    *,
    eps: float,
    dirichlet: bool,
    n_rhs: int = 8,
    seed: int = 0,
) -> list[K0LaplacianBenchmarkReport]:
    default_operators = operators_by_rank[TENSOR_RANKS[-1]]
    rhs_batch = _build_rhs_batch(seq, k=0, dirichlet=dirichlet, n_rhs=n_rhs, seed=seed)
    if eps == 0.0:
        rhs_batch = jax.vmap(
            lambda rhs: _project_k0_laplacian_rhs_to_range(seq, default_operators, rhs, dirichlet)
        )(rhs_batch)

    catalog = _k0_laplacian_preconditioner_catalog()

    dense_matrix = None
    dense_pinv = None
    if DENSE:
        dense_matrix, dense_pinv, _ = _dense_k0_hodge_inverse(
            seq, default_operators, dirichlet, eps=eps
        )

    reports = []
    for entry in catalog:
        label = entry["label"]
        preconditioner = entry["preconditioner"]
        tensor_rank = entry.get("tensor_rank")
        operators = (
            operators_by_rank[tensor_rank]
            if tensor_rank is not None
            else default_operators
        )

        @jax.jit
        def solve(rhs):
            rhs_use = rhs if eps > 0.0 else _project_k0_laplacian_rhs_to_range(
                seq, operators, rhs, dirichlet
            )
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
                    seq, operators, x, 0, dirichlet=dirichlet
                )
            residual = residual - rhs_use
            rhs_norm = jnp.where(jnp.linalg.norm(rhs_use) > 0, jnp.linalg.norm(rhs_use), 1.0)
            return x, info, jnp.linalg.norm(residual) / rhs_norm

        x0, _, _ = solve(rhs_batch[0])
        jax.block_until_ready(x0)

        iterations = []
        times_ms = []
        residuals = []
        n_converged = 0
        for rhs in rhs_batch:
            t0 = time.perf_counter()
            x, info, residual = solve(rhs)
            jax.block_until_ready(x)
            info_int = int(info)
            times_ms.append((time.perf_counter() - t0) * 1e3)
            iterations.append(abs(info_int))
            residuals.append(float(residual))
            n_converged += int(info_int <= 0)

        iterations_array = jnp.asarray(iterations)
        times_ms_array = jnp.asarray(times_ms)
        residuals_array = jnp.asarray(residuals)
        reports.append(
            K0LaplacianBenchmarkReport(
                label=label,
                eps=eps,
                dirichlet=dirichlet,
                n_rhs=n_rhs,
                n_converged=n_converged,
                n_not_converged=n_rhs - n_converged,
                avg_iters=float(jnp.mean(iterations_array)),
                std_iters=float(jnp.std(iterations_array)),
                max_iters=int(jnp.max(iterations_array)),
                avg_time_ms=float(jnp.mean(times_ms_array)),
                std_time_ms=float(jnp.std(times_ms_array)),
                max_time_ms=float(jnp.max(times_ms_array)),
                avg_relative_residual=float(jnp.mean(residuals_array)),
                std_relative_residual=float(jnp.std(residuals_array)),
                max_relative_residual=float(jnp.max(residuals_array)),
                diagnostics=_scalar_hodge_preconditioner_diagnostics(
                    seq,
                    operators,
                    eps=eps,
                    dirichlet=dirichlet,
                    preconditioner=preconditioner,
                ),
            )
        )

    if DENSE:
        @jax.jit
        def solve_dense(rhs):
            rhs_use = rhs if eps > 0.0 else _project_k0_laplacian_rhs_to_range(
                seq, operators, rhs, dirichlet
            )
            x = dense_pinv @ rhs_use
            residual = dense_matrix @ x + eps * apply_mass_matrix(
                seq, operators, x, 0, dirichlet=dirichlet
            ) - rhs_use
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
            info_int = int(info)
            times_ms.append((time.perf_counter() - t0) * 1e3)
            iterations.append(abs(info_int))
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
                n_converged=n_rhs,
                n_not_converged=0,
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
    print("-" * 128)
    print(
        f"{'label':<18} {'conv':>8} {'avg iters':>10} {'std':>8} {'max':>6} {'avg ms':>10} {'std ms':>10} {'max ms':>10} {'avg relres':>12} {'max relres':>12}"
    )
    for report in reports:
        print(
            f"{report.label:<18} {f'{report.n_converged}/{report.n_rhs}':>8} {report.avg_iters:>10.2f} {report.std_iters:>8.2f} {report.max_iters:>6d} "
            f"{report.avg_time_ms:>10.2f} {report.std_time_ms:>10.2f} {report.max_time_ms:>10.2f} "
            f"{report.avg_relative_residual:>12.3e} {report.max_relative_residual:>12.3e}"
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
    return x, info


def benchmark_mass_preconditioners(
    seq,
    operators_by_rank,
    *,
    k: int,
    dirichlet: bool,
    n_rhs: int = 8,
    seed: int = 0,
) -> list[MassBenchmarkReport]:
    rhs_size = _mass_rhs_size(seq, k, dirichlet)
    rhs_batch = _build_rhs_batch(seq, k=k, dirichlet=dirichlet, n_rhs=n_rhs, seed=seed)

    catalog = _mass_preconditioner_catalog(k)
    dense_matrix = None
    tuned_diaginv = None
    tuned_omega = None
    tuned_lambda_min = None
    tuned_lambda_max = None
    exact_tuning = None
    default_operators = operators_by_rank[TENSOR_RANKS[-1]]
    if EXACT_HYPERPARAMS or DENSE:
        dense_matrix = jnp.asarray(
            dense_mass_matrix(seq, default_operators, k, dirichlet=dirichlet)
        )
    if EXACT_HYPERPARAMS:
        tuned_diaginv, tuned_omega, tuned_lambda_min, tuned_lambda_max = (
            _dense_optimal_jacobi_poly_tuning(dense_matrix)
        )
        exact_tuning = {
            "omega": float(tuned_omega),
            "lambda_min": float(tuned_lambda_min),
            "lambda_max": float(tuned_lambda_max),
        }

    reports = []
    for entry in catalog:
        label = entry["label"]
        outer = entry["outer"]
        inner = entry["inner"]
        reason = entry["reason"]
        preconditioner = entry["preconditioner"]
        tensor_rank = entry.get("tensor_rank")
        operators = (
            operators_by_rank[tensor_rank]
            if tensor_rank is not None
            else default_operators
        )

        if (
            EXACT_HYPERPARAMS
            and isinstance(preconditioner, MassPreconditionerSpec)
            and preconditioner.kind == "richardson"
            and not preconditioner.surgery_schur
            and _uses_plain_jacobi_smoother(preconditioner)
        ):
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
        elif (
            EXACT_HYPERPARAMS
            and isinstance(preconditioner, MassPreconditionerSpec)
            and preconditioner.kind == "chebyshev"
            and not preconditioner.surgery_schur
            and _uses_plain_jacobi_smoother(preconditioner)
        ):
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
                return x, info

        x0, _ = solve(rhs_batch[0])
        jax.block_until_ready(x0)

        iterations = []
        times_ms = []
        n_converged = 0
        for rhs in rhs_batch:
            t0 = time.perf_counter()
            x, info = solve(rhs)
            jax.block_until_ready(x)
            info_int = int(info)
            times_ms.append((time.perf_counter() - t0) * 1e3)
            iterations.append(abs(info_int))
            n_converged += int(info_int <= 0)

        iterations_array = jnp.asarray(iterations)
        times_ms_array = jnp.asarray(times_ms)
        diagnostics = _mass_preconditioner_diagnostics(
            seq,
            operators,
            k=k,
            dirichlet=dirichlet,
            preconditioner=preconditioner,
            exact_tuning=exact_tuning,
        )
        precond_apply = _mass_preconditioner_apply_from_spec(
            seq,
            operators,
            k=k,
            dirichlet=dirichlet,
            preconditioner=preconditioner,
        )
        if RUN_TEXT_SANITY and precond_apply is not None and _should_collect_text_sanity(label):
            diagnostics = dict(diagnostics or {})
            diagnostics["sampled_identity"] = _sampled_identity_sanity(
                seq,
                operators,
                k=k,
                dirichlet=dirichlet,
                probe_batch=rhs_batch,
                precond_apply=precond_apply,
            )
            if DENSE and DENSE_TEXT_SANITY:
                diagnostics["dense_identity"] = _dense_identity_sanity(
                    dense_matrix,
                    precond_apply,
                )

        reports.append(
            MassBenchmarkReport(
                k=k,
                label=label,
                outer=outer,
                inner=inner,
                reason=reason,
                dirichlet=dirichlet,
                n_rhs=n_rhs,
                n_converged=n_converged,
                n_not_converged=n_rhs - n_converged,
                avg_iters=float(jnp.mean(iterations_array)),
                std_iters=float(jnp.std(iterations_array)),
                max_iters=int(jnp.max(iterations_array)),
                avg_time_ms=float(jnp.mean(times_ms_array)),
                std_time_ms=float(jnp.std(times_ms_array)),
                max_time_ms=float(jnp.max(times_ms_array)),
                diagnostics=diagnostics,
            )
        )

    if DENSE:
        @jax.jit
        def solve_dense(rhs):
            x = jnp.linalg.solve(dense_matrix, rhs)
            return x, jnp.asarray(0, dtype=jnp.int32)

        x0, _ = solve_dense(rhs_batch[0])
        jax.block_until_ready(x0)

        iterations = []
        times_ms = []
        n_converged = 0
        for rhs in rhs_batch:
            t0 = time.perf_counter()
            x, info = solve_dense(rhs)
            jax.block_until_ready(x)
            info_int = int(info)
            times_ms.append((time.perf_counter() - t0) * 1e3)
            iterations.append(abs(info_int))
            n_converged += int(info_int <= 0)

        iterations_array = jnp.asarray(iterations)
        times_ms_array = jnp.asarray(times_ms)
        reports.append(
            MassBenchmarkReport(
                k=k,
                label="direct/-",
                outer="direct",
                inner="-",
                reason="dense direct solve baseline",
                dirichlet=dirichlet,
                n_rhs=n_rhs,
                n_converged=n_converged,
                n_not_converged=n_rhs - n_converged,
                avg_iters=float(1),
                std_iters=float(0),
                max_iters=int(1),
                avg_time_ms=float(jnp.mean(times_ms_array)),
                std_time_ms=float(jnp.std(times_ms_array)),
                max_time_ms=float(jnp.max(times_ms_array)),
            )
        )
    return reports


def benchmark_mass_block_chebyshev_preconditioners(
    seq,
    operators_by_rank,
    *,
    k: int,
    dirichlet: bool,
    n_rhs: int = 8,
    seed: int = 0,
) -> list[MassBenchmarkReport]:
    if k not in (0, 1, 2):
        raise ValueError("Block Chebyshev benchmark only applies to k=0,1,2")

    rhs_batch = _build_rhs_batch(seq, k=k, dirichlet=dirichlet, n_rhs=n_rhs, seed=seed)
    tensor_rank = TENSOR_RANKS[-1]
    operators = operators_by_rank[tensor_rank]
    reports = []
    dense_matrix = None
    if DENSE:
        dense_matrix = jnp.asarray(dense_mass_matrix(seq, operators, k, dirichlet=dirichlet))

    for smoother_kind, label, reason in (
        ("jacobi", "schur-block-cheb-jacobi", "exact block Chebyshev with Jacobi smoothing"),
        ("tensor", f"schur-block-cheb-tensor-r{tensor_rank}", "exact block Chebyshev with tensor smoothing"),
    ):
        precond_apply, diagnostics = _build_exact_block_chebyshev_mass_preconditioner_apply(
            seq,
            operators,
            k=k,
            dirichlet=dirichlet,
            smoother_kind=smoother_kind,
        )

        @jax.jit
        def solve(rhs):
            return _solve_mass_with_preconditioner_apply(
                seq,
                operators,
                rhs,
                k=k,
                dirichlet=dirichlet,
                precond_apply=precond_apply,
            )

        x0, _ = solve(rhs_batch[0])
        jax.block_until_ready(x0)

        iterations = []
        times_ms = []
        n_converged = 0
        for rhs in rhs_batch:
            t0 = time.perf_counter()
            x, info = solve(rhs)
            jax.block_until_ready(x)
            info_int = int(info)
            times_ms.append((time.perf_counter() - t0) * 1e3)
            iterations.append(abs(info_int))
            n_converged += int(info_int <= 0)

        iterations_array = jnp.asarray(iterations)
        times_ms_array = jnp.asarray(times_ms)
        if RUN_TEXT_SANITY and _should_collect_text_sanity(label):
            diagnostics = dict(diagnostics)
            diagnostics["sampled_identity"] = _sampled_identity_sanity(
                seq,
                operators,
                k=k,
                dirichlet=dirichlet,
                probe_batch=rhs_batch,
                precond_apply=precond_apply,
            )
            if dense_matrix is not None and DENSE_TEXT_SANITY:
                diagnostics["dense_identity"] = _dense_identity_sanity(dense_matrix, precond_apply)

        reports.append(
            MassBenchmarkReport(
                k=k,
                label=label,
                outer="schur",
                inner=f"block-cheb-{smoother_kind}",
                reason=reason,
                dirichlet=dirichlet,
                n_rhs=n_rhs,
                n_converged=n_converged,
                n_not_converged=n_rhs - n_converged,
                avg_iters=float(jnp.mean(iterations_array)),
                std_iters=float(jnp.std(iterations_array)),
                max_iters=int(jnp.max(iterations_array)),
                avg_time_ms=float(jnp.mean(times_ms_array)),
                std_time_ms=float(jnp.std(times_ms_array)),
                max_time_ms=float(jnp.max(times_ms_array)),
                diagnostics=diagnostics,
            )
        )

    return reports


def print_mass_benchmark_reports(reports: list[MassBenchmarkReport]):
    print("-" * 104)
    print(
        f"{'label':<18} {'conv':>8} {'avg iters':>10} {'std':>8} {'max':>6} {'avg ms':>10} {'std ms':>10} {'max ms':>10} {'reason':<0}"
    )
    for report in reports:
        print(
            f"{report.label:<18} {f'{report.n_converged}/{report.n_rhs}':>8} {report.avg_iters:>10.2f} {report.std_iters:>8.2f} {report.max_iters:>6d} "
            f"{report.avg_time_ms:>10.2f} {report.std_time_ms:>10.2f} {report.max_time_ms:>10.2f} "
            f"{report.reason}"
        )
    _print_mass_preconditioner_sanity(reports)


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


def _report_display_labels(reports: list[MassBenchmarkReport]) -> list[str]:
    labels = []
    for report in reports:
        suffix = "*" if report.n_not_converged > 0 else ""
        labels.append(f"{report.label}{suffix}")
    return labels


def _report_labels(reports: list[MassBenchmarkReport]) -> list[str]:
    return [report.label for report in reports]


def _labels_for_k(
    benchmarks: dict[tuple[int, bool], list[MassBenchmarkReport]],
    *,
    k: int,
) -> list[str]:
    cases = _available_dirichlet_cases_for_k(benchmarks, k=k)
    labels = []
    for dirichlet in cases:
        for report in benchmarks[(k, dirichlet)]:
            if report.label not in labels:
                labels.append(report.label)
    return labels


def _available_dirichlet_cases_for_k(
    benchmarks: dict[tuple[int, bool], list[MassBenchmarkReport]],
    *,
    k: int,
) -> list[bool]:
    cases = [dirichlet for dirichlet in BENCHMARK_DIRICHLET_CASES if (k, dirichlet) in benchmarks]
    if cases:
        return cases
    discovered = []
    for key_k, dirichlet in benchmarks:
        if key_k == k and dirichlet not in discovered:
            discovered.append(dirichlet)
    return discovered


def _available_benchmark_cases(
    benchmarks: dict[tuple[int, bool], list[MassBenchmarkReport]],
) -> list[tuple[int, bool]]:
    cases = []
    for k in (0, 1, 2, 3):
        for dirichlet in _available_dirichlet_cases_for_k(benchmarks, k=k):
            cases.append((k, dirichlet))
    return cases


def _display_labels_for_k(
    benchmarks: dict[tuple[int, bool], list[MassBenchmarkReport]],
    *,
    k: int,
) -> list[str]:
    labels = _labels_for_k(benchmarks, k=k)
    cases = _available_dirichlet_cases_for_k(benchmarks, k=k)
    displays = []
    for label in labels:
        failed = False
        for dirichlet in cases:
            for report in benchmarks[(k, dirichlet)]:
                if report.label == label and report.n_not_converged > 0:
                    failed = True
                    break
            if failed:
                break
        displays.append(f"{label}*" if failed else label)
    return displays


def _all_benchmark_labels(
    benchmarks: dict[tuple[int, bool], list[MassBenchmarkReport]],
) -> list[str]:
    labels = []
    for k in (0, 1, 2, 3):
        for label in _labels_for_k(benchmarks, k=k):
            if label not in labels:
                labels.append(label)
    return labels


def _display_labels_from_case_reports(reports_by_case: dict[bool, list]) -> list[str]:
    labels = []
    for label in _k0_laplacian_labels():
        if any(label in {report.label for report in reports} for reports in reports_by_case.values()):
            labels.append(label)
    displays = []
    for label in labels:
        failed = any(
            next(report for report in reports if report.label == label).n_not_converged > 0
            for reports in reports_by_case.values()
            if any(report.label == label for report in reports)
        )
        displays.append(f"{label}*" if failed else label)
    return displays


def _label_family(label: str) -> str:
    if label.startswith("richardson"):
        return "richardson"
    if label.startswith("chebyshev"):
        return "chebyshev"
    if label.startswith("tensor"):
        return "tensor"
    return label


def _grouped_bar_positions(
    labels: list[str],
    *,
    intra_group_step: float = 1.0,
    inter_group_gap: float = 0.8,
) -> jnp.ndarray:
    positions = []
    current = 0.0
    previous_family = None
    for label in labels:
        family = _label_family(label)
        if previous_family is None:
            positions.append(current)
        else:
            current += intra_group_step
            if family != previous_family:
                current += inter_group_gap
            positions.append(current)
        previous_family = family
    return jnp.asarray(positions, dtype=jnp.float64)


def _blend_toward_white(color, amount: float):
    red, green, blue = plt.matplotlib.colors.to_rgb(color)
    return (
        red + (1.0 - red) * amount,
        green + (1.0 - green) * amount,
        blue + (1.0 - blue) * amount,
    )


def _grouped_bar_colors(labels: list[str]):
    base_colors = {
        "jacobi": "#1f77b4",
        "richardson": "#ff7f0e",
        "chebyshev": "#2ca02c",
        "tensor": "#d62728",
    }
    family_counts = {}
    family_offsets = {}
    for label in labels:
        family = _label_family(label)
        family_counts[family] = family_counts.get(family, 0) + 1

    colors = []
    for label in labels:
        family = _label_family(label)
        offset = family_offsets.get(family, 0)
        family_offsets[family] = offset + 1
        total = family_counts[family]
        if total <= 1:
            lighten = 0.0
        else:
            lighten = 0.18 * offset / (total - 1)
        colors.append(_blend_toward_white(base_colors.get(family, "#7f7f7f"), lighten))
    return colors


def _benchmark_figure_note(reports: list[MassBenchmarkReport]) -> str:
    failed = [
        f"{report.label} ({report.n_not_converged}/{report.n_rhs} failed)"
        for report in reports
        if report.n_not_converged > 0
    ]
    if not failed:
        return f"All {reports[0].n_rhs}/{reports[0].n_rhs} RHS converged."
    return "* indicates a non-converged option. " + "; ".join(failed)


def _benchmark_figure_note_grouped(
    benchmarks: dict[tuple[int, bool], list[MassBenchmarkReport]],
) -> str:
    failed = []
    for k in (0, 1, 2, 3):
        for dirichlet in _available_dirichlet_cases_for_k(benchmarks, k=k):
            for report in benchmarks[(k, dirichlet)]:
                if report.n_not_converged > 0:
                    failed.append(
                        f"k={k}, dirichlet={dirichlet} {report.label} ({report.n_not_converged}/{report.n_rhs} failed)"
                    )
    if not failed:
        sample_key = next(iter(benchmarks))
        sample = benchmarks[sample_key][0]
        return f"All {sample.n_rhs}/{sample.n_rhs} RHS converged for all shown options."
    return "* indicates a non-converged option. " + "; ".join(failed)


def plot_mass_benchmark_reports(
    benchmarks: dict[tuple[int, bool], list[MassBenchmarkReport]],
    *,
    k: int,
    dirichlet: bool = DEFAULT_DIRICHLET,
    show: bool = True,
    logscale: bool = True,
):
    reports = benchmarks[(k, dirichlet)]
    labels = _report_display_labels(reports)
    x = _grouped_bar_positions(labels)
    colors = _grouped_bar_colors(labels)
    metrics = [
        ("avg_iters", "std_iters", "Average Iterations"),
        ("avg_time_ms", "std_time_ms", "Average Time [ms]"),
    ]

    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(max(12.0, 1.25 * len(labels)), 4.8),
        constrained_layout=True,
    )
    if len(metrics) == 1:
        axes = [axes]

    for ax, (metric_name, std_name, metric_label) in zip(axes, metrics, strict=True):
        values = [getattr(report, metric_name) for report in reports]
        errors = [getattr(report, std_name) for report in reports]
        ax.bar(
            x,
            values,
            color=colors,
            yerr=errors,
            error_kw={"elinewidth": 1.0, "capsize": 3, "capthick": 1.0},
        )
        if logscale:
            ax.set_yscale("log", base=10)
        ax.set_title(metric_label)
        ax.set_ylabel(metric_label)
        ax.set_xlim(float(x[0]) - 0.75, float(x[-1]) + 0.75)
        ax.set_xticks(x, labels, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle(f"k={k} mass preconditioner benchmark (dirichlet={dirichlet})")
    if show:
        plt.show()
    return fig


def plot_all_mass_benchmark_reports(
    benchmarks: dict[tuple[int, bool], list[MassBenchmarkReport]],
    *,
    k0_laplacian_benchmarks: dict[bool, list[K0LaplacianBenchmarkReport]] | None = None,
    k0_laplacian_eps: float | None = None,
    show: bool = True,
    logscale: bool = True,
):
    case_specs = [("mass", k) for k in (0, 1, 2, 3)]
    if k0_laplacian_benchmarks:
        case_specs.append(("laplace", 0))
    width = 0.38
    metrics = [
        ("avg_iters", "std_iters", "Average Iterations"),
        ("avg_time_ms", "std_time_ms", "Average Time [ms]"),
    ]

    fig, axes = plt.subplots(
        len(metrics),
        len(case_specs),
        figsize=(4.6 * len(case_specs), 8.0),
        sharey="row",
        constrained_layout=True,
        squeeze=False,
    )

    for col, (case_kind, k) in enumerate(case_specs):
        if case_kind == "mass":
            cases = _available_dirichlet_cases_for_k(benchmarks, k=k)
            labels = _labels_for_k(benchmarks, k=k)
            display_labels = _display_labels_for_k(benchmarks, k=k)
            report_maps = {
                dirichlet: {report.label: report for report in benchmarks[(k, dirichlet)]}
                for dirichlet in cases
            }
            title_prefix = f"k={k} mass"
        else:
            cases = list(k0_laplacian_benchmarks)
            labels = [label.rstrip("*") for label in _display_labels_from_case_reports(k0_laplacian_benchmarks)]
            display_labels = _display_labels_from_case_reports(k0_laplacian_benchmarks)
            report_maps = {
                dirichlet: {report.label: report for report in k0_laplacian_benchmarks[dirichlet]}
                for dirichlet in cases
            }
            title_prefix = f"k=0 laplace ($\\epsilon$={k0_laplacian_eps:g})"
        if not labels:
            continue
        x = _grouped_bar_positions(labels)
        colors = _grouped_bar_colors(labels)
        if len(cases) == 1:
            offsets = {cases[0]: 0.0}
            alphas = {cases[0]: 0.9}
            bar_width = 0.6
        else:
            bar_width = width
            offsets = {
                dirichlet: (-bar_width / 2 if index == 0 else bar_width / 2)
                for index, dirichlet in enumerate(cases)
            }
            alphas = {
                dirichlet: (0.9 if not dirichlet else 0.45)
                for dirichlet in cases
            }
        for row, (metric_name, std_name, metric_label) in enumerate(metrics):
            ax = axes[row, col]
            for dirichlet in cases:
                case_reports = report_maps[dirichlet]
                values = [
                    getattr(case_reports[label], metric_name) if label in case_reports else jnp.nan
                    for label in labels
                ]
                errors = [
                    getattr(case_reports[label], std_name) if label in case_reports else 0.0
                    for label in labels
                ]
                ax.bar(
                    x + offsets[dirichlet],
                    values,
                    width=bar_width,
                    color=colors,
                    alpha=alphas[dirichlet],
                    yerr=errors,
                    error_kw={"elinewidth": 1.0, "capsize": 3, "capthick": 1.0},
                )
            if logscale:
                ax.set_yscale("log", base=10)
            ax.set_title(f"{title_prefix} {metric_label}")
            ax.set_xlim(float(x[0]) - 0.75, float(x[-1]) + 0.75)
            ax.grid(axis="y", alpha=0.25)
            if col == 0:
                ax.set_ylabel(metric_label)
            if row == len(metrics) - 1:
                ax.set_xticks(x, display_labels, rotation=25, ha="right")
            else:
                ax.set_xticks(x, [])

    present_dirichlet_cases = []
    for _, dirichlet in _available_benchmark_cases(benchmarks):
        if dirichlet not in present_dirichlet_cases:
            present_dirichlet_cases.append(dirichlet)
    legend_handles = [
        plt.matplotlib.patches.Patch(
            facecolor="#595959",
            alpha=(0.9 if not dirichlet else 0.45),
            label=f"dirichlet={dirichlet}",
        )
        for dirichlet in present_dirichlet_cases
    ]
    axes[0, 0].legend(handles=legend_handles, loc="upper right", frameon=False)
    fig.suptitle("Mass and k=0 Hodge/Laplacian preconditioner benchmark")
    if show:
        plt.show()
    return fig


def plot_mass_benchmark_heatmaps(
    benchmarks: dict[tuple[int, bool], list[MassBenchmarkReport]],
    *,
    k: int,
    dirichlet: bool = DEFAULT_DIRICHLET,
    show: bool = True,
    logscale: bool = True,
):
    reports = benchmarks[(k, dirichlet)]
    labels = _report_display_labels(reports)
    matrix = jnp.asarray([
        [report.avg_iters for report in reports],
        [report.avg_time_ms for report in reports],
    ])
    display_matrix = _log10_safe(matrix) if logscale else matrix
    metric_labels = ["Average Iterations", "Average Time [ms]"]

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(max(12.0, 1.25 * len(labels)), 3.8),
        constrained_layout=True,
    )
    image = ax.imshow(display_matrix, aspect="auto", cmap="viridis")
    ax.set_title(
        f"k={k} benchmark summary ({'log10' if logscale else 'linear'} scale, dirichlet={dirichlet})"
    )
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(metric_labels)), metric_labels)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.9)
    colorbar.set_label(r"$\log_{10}(\mathrm{value})$" if logscale else "value")

    if show:
        plt.show()
    return fig


def plot_all_mass_benchmark_heatmaps(
    benchmarks: dict[tuple[int, bool], list[MassBenchmarkReport]],
    *,
    k0_laplacian_benchmarks: dict[bool, list[K0LaplacianBenchmarkReport]] | None = None,
    k0_laplacian_eps: float | None = None,
    show: bool = True,
    logscale: bool = True,
):
    case_specs = [("mass", k) for k in (0, 1, 2, 3)]
    if k0_laplacian_benchmarks:
        case_specs.append(("laplace", 0))

    fig, axes = plt.subplots(
        1,
        len(case_specs),
        figsize=(4.2 * len(case_specs), 4.8),
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes[0]
    for col, (case_kind, k) in enumerate(case_specs):
        ax = axes[col]
        if case_kind == "mass":
            dirichlet = _available_dirichlet_cases_for_k(benchmarks, k=k)[0]
            reports = benchmarks[(k, dirichlet)]
            labels = _report_display_labels(reports)
            title = f"k={k} mass"
        else:
            dirichlet = next(iter(k0_laplacian_benchmarks))
            reports = k0_laplacian_benchmarks[dirichlet]
            labels = _display_labels_from_case_reports(k0_laplacian_benchmarks)
            title = f"k=0 laplace ($\\epsilon$={k0_laplacian_eps:g})"

        report_map = {report.label: report for report in reports}
        base_labels = [label.rstrip("*") for label in labels]
        matrix = jnp.asarray([
            [report_map[label].avg_iters for label in base_labels],
            [report_map[label].avg_time_ms for label in base_labels],
        ])
        display_matrix = _log10_safe(matrix) if logscale else matrix
        image = ax.imshow(display_matrix, aspect="auto", cmap="viridis")
        ax.set_title(title)
        ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
        ax.set_yticks(range(2), ["Average Iterations", "Average Time [ms]"] if col == 0 else [])
        fig.colorbar(image, ax=ax, shrink=0.85)

    fig.suptitle("Mass and k=0 Hodge/Laplacian benchmark summary")
    if show:
        plt.show()
    return fig


def plot_mass_spectrum_reports(
    spectra: dict[tuple[int, bool], MassSpectrumReport],
    *,
    dirichlet: bool = DEFAULT_DIRICHLET,
    show: bool = True,
    logscale: bool = True,
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
        values = [getattr(spectra[(k, dirichlet)], metric_name) for k in (0, 1, 2, 3)]
        ax.bar(x, values, width=0.6, color="#1f77b4", alpha=0.9)
        if logscale and log_scale:
            ax.set_yscale("log", base=10)
        ax.set_title(title)
        ax.set_xticks(x, ["k=0", "k=1", "k=2", "k=3"])
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle(f"Mass matrix spectrum summary (dirichlet={dirichlet})")
    if show:
        plt.show()
    return fig


def plot_mass_spectrum_heatmap(
    spectra: dict[tuple[int, bool], MassSpectrumReport],
    *,
    dirichlet: bool = DEFAULT_DIRICHLET,
    show: bool = True,
):
    metric_names = ["eig_min", "eig_max", "condition_number"]
    metric_labels = [r"$\lambda_{\min}$", r"$\lambda_{\max}$", r"$\kappa$"]
    matrix = jnp.asarray([
        [getattr(spectra[(k, dirichlet)], metric_name) for k in (0, 1, 2, 3)]
        for metric_name in metric_names
    ])

    fig, ax = plt.subplots(1, 1, figsize=(12, 3.8), constrained_layout=True)
    image = ax.imshow(_log10_safe(matrix), aspect="auto", cmap="cividis")
    ax.set_title(f"Mass Matrix Spectrum Metrics ($\\log_{{10}}$ scale, dirichlet={dirichlet})")
    ax.set_xticks(range(4), [f"k={k}" for k in (0, 1, 2, 3)])
    ax.set_yticks(range(len(metric_labels)), metric_labels)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.9)
    colorbar.set_label(r"$\log_{10}(\mathrm{value})$")
    if show:
        plt.show()
    return fig


def plot_k0_laplacian_benchmark_reports(
    benchmarks: dict[bool, list[K0LaplacianBenchmarkReport]],
    *,
    eps: float,
    show: bool = True,
):
    dirichlet_cases = list(benchmarks)
    labels = _k0_laplacian_labels()
    display_labels = []
    for label in labels:
        failed = any(
            next(report for report in benchmarks[dirichlet] if report.label == label).n_not_converged > 0
            for dirichlet in dirichlet_cases
        )
        display_labels.append(f"{label}*" if failed else label)
    x = _grouped_bar_positions(labels)
    colors = _grouped_bar_colors(labels)
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
    if len(dirichlet_cases) == 1:
        offsets = {dirichlet_cases[0]: 0.0}
        alphas = {dirichlet_cases[0]: 0.9}
        width = 0.6
    else:
        width = 0.38
        offsets = {
            dirichlet: (-width / 2 if index == 0 else width / 2)
            for index, dirichlet in enumerate(dirichlet_cases)
        }
        alphas = {
            dirichlet: (0.9 if not dirichlet else 0.45)
            for dirichlet in dirichlet_cases
        }

    report_maps = {
        dirichlet: {report.label: report for report in benchmarks[dirichlet]}
        for dirichlet in dirichlet_cases
    }
    for row, (metric_name, std_name, metric_label, log_scale) in enumerate(metrics):
        ax = axes[row]
        for dirichlet in dirichlet_cases:
            case_reports = report_maps[dirichlet]
            values = [getattr(case_reports[label], metric_name) for label in labels]
            errors = [getattr(case_reports[label], std_name) for label in labels]
            ax.bar(
                x + offsets[dirichlet],
                values,
                width=width,
                color=colors,
                alpha=alphas[dirichlet],
                yerr=errors,
                error_kw={"elinewidth": 1.0, "capsize": 3, "capthick": 1.0},
            )
        if log_scale:
            ax.set_yscale("log", base=10)
        ax.set_title(f"k=0 {metric_label}")
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylabel(metric_label)
        if row == len(metrics) - 1:
            ax.set_xticks(x, display_labels, rotation=25, ha="right")
        else:
            ax.set_xticks(x, [])

    axes[0].legend(
        handles=[
            plt.matplotlib.patches.Patch(
                facecolor="#595959",
                alpha=alphas[dirichlet],
                label=f"dirichlet={dirichlet}",
            )
            for dirichlet in dirichlet_cases
        ],
        loc="upper right",
        frameon=False,
    )
    fig.suptitle(f"k=0 Hodge/Laplacian preconditioner benchmark ($\\epsilon$={eps:g})")
    if show:
        plt.show()
    return fig


def plot_k0_laplacian_benchmark_heatmaps(
    benchmarks: dict[bool, list[K0LaplacianBenchmarkReport]],
    *,
    eps: float,
    show: bool = True,
):
    dirichlet_cases = list(benchmarks)
    labels = _k0_laplacian_labels()
    case_labels = [f"k=0, dbc={dirichlet}" for dirichlet in dirichlet_cases]
    benchmark_maps = {
        dirichlet: {report.label: report for report in benchmarks[dirichlet]}
        for dirichlet in dirichlet_cases
    }
    iteration_matrix = jnp.asarray([
        [benchmark_maps[dirichlet][label].avg_iters for dirichlet in dirichlet_cases]
        for label in labels
    ])
    time_matrix = jnp.asarray([
        [benchmark_maps[dirichlet][label].avg_time_ms for dirichlet in dirichlet_cases]
        for label in labels
    ])

    fig, axes = plt.subplots(1, 2, figsize=(18, 4.8), constrained_layout=True)
    for ax, matrix, title, cmap in (
        (axes[0], iteration_matrix, "Average Iterations", "viridis"),
        (axes[1], time_matrix, "Average Time [ms]", "magma"),
    ):
        image = ax.imshow(_log10_safe(matrix), aspect="auto", cmap=cmap)
        ax.set_title(f"{title} ($\\log_{{10}}$ scale)")
        ax.set_xticks(range(len(case_labels)), case_labels, rotation=35, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        colorbar = fig.colorbar(image, ax=ax, shrink=0.9)
        colorbar.set_label(r"$\log_{10}(\mathrm{value})$")

    fig.suptitle(f"k=0 Hodge/Laplacian benchmark summary ($\\epsilon$={eps:g})")
    if show:
        plt.show()
    return fig


def plot_k0_laplacian_spectrum_reports(
    spectra: dict[bool, K0LaplacianSpectrumReport],
    *,
    show: bool = True,
):
    dirichlet_cases = list(spectra)
    metrics = [
        ("min_nonzero_eig", r"$\lambda_{\min}^{+}(L_0)$", True),
        ("max_eig", r"$\lambda_{\max}(L_0)$", True),
        ("nonzero_condition_number", r"$\kappa^{+}(L_0)$", True),
    ]
    x = jnp.arange(1)
    width = 0.36 if len(dirichlet_cases) > 1 else 0.6

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for ax, (metric_name, title, log_scale) in zip(axes, metrics, strict=True):
        for index, dirichlet in enumerate(dirichlet_cases):
            offset = 0.0 if len(dirichlet_cases) == 1 else (-width / 2 if index == 0 else width / 2)
            alpha = 0.9 if not dirichlet else 0.45
            ax.bar(
                x + offset,
                [getattr(spectra[dirichlet], metric_name)],
                width=width,
                color="#1f77b4",
                alpha=alpha,
                label=f"dirichlet={dirichlet}",
            )
        if log_scale:
            ax.set_yscale("log", base=10)
        ax.set_title(title)
        ax.set_xticks(x, ["k=0"])
        ax.grid(axis="y", alpha=0.25)

    axes[0].legend(frameon=False)
    fig.suptitle("k=0 Hodge/Laplacian spectrum summary")
    if show:
        plt.show()
    return fig


def plot_k0_laplacian_spectrum_heatmap(
    spectra: dict[bool, K0LaplacianSpectrumReport],
    *,
    show: bool = True,
):
    dirichlet_cases = list(spectra)
    metric_names = ["min_nonzero_eig", "max_eig", "nonzero_condition_number"]
    metric_labels = [r"$\lambda_{\min}^{+}$", r"$\lambda_{\max}$", r"$\kappa^{+}$"]
    matrix = jnp.asarray([
        [getattr(spectra[dirichlet], metric_name) for dirichlet in dirichlet_cases]
        for metric_name in metric_names
    ])

    fig, ax = plt.subplots(1, 1, figsize=(12, 3.8), constrained_layout=True)
    image = ax.imshow(_log10_safe(matrix), aspect="auto", cmap="cividis")
    ax.set_title("k=0 Hodge/Laplacian spectrum metrics ($\\log_{10}$ scale)")
    ax.set_xticks(
        range(len(dirichlet_cases)),
        [f"k=0, dbc={dirichlet}" for dirichlet in dirichlet_cases],
        rotation=35,
        ha="right",
    )
    ax.set_yticks(range(len(metric_labels)), metric_labels)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.9)
    colorbar.set_label(r"$\log_{10}(\mathrm{value})$")
    if show:
        plt.show()
    return fig


def _flatten_k0_laplacian_benchmarks(
    benchmarks: dict[tuple[float, bool], list[K0LaplacianBenchmarkReport]],
) -> list[dict]:
    rows = []
    for reports in benchmarks.values():
        rows.extend(asdict(report) for report in reports)
    return rows


def _flatten_k0_laplacian_spectra(
    spectra: dict[bool, K0LaplacianSpectrumReport],
) -> list[dict]:
    return [asdict(report) for report in spectra.values()]


def _eps_tag(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


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
    *,
    k0_laplacian_benchmarks: dict[tuple[float, bool], list[K0LaplacianBenchmarkReport]] | None = None,
    k0_laplacian_spectra: dict[bool, K0LaplacianSpectrumReport] | None = None,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs") / "interactive" / f"mass_preconditioner_choices_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = _mass_run_metadata(benchmarks, k0_laplacian_benchmarks)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (output_dir / "benchmarks.json").write_text(
        json.dumps(_flatten_benchmarks(benchmarks), indent=2)
    )
    if DENSE:
        (output_dir / "spectra.json").write_text(
            json.dumps(_flatten_spectra(spectra), indent=2)
        )
    if k0_laplacian_benchmarks:
        (output_dir / "k0_laplacian_benchmarks.json").write_text(
            json.dumps(_flatten_k0_laplacian_benchmarks(k0_laplacian_benchmarks), indent=2)
        )
    if DENSE and k0_laplacian_spectra:
        (output_dir / "k0_laplacian_spectra.json").write_text(
            json.dumps(_flatten_k0_laplacian_spectra(k0_laplacian_spectra), indent=2)
        )

    figures = []
    if k0_laplacian_benchmarks:
        eps_values = sorted({eps for eps, _ in k0_laplacian_benchmarks})
        for eps in eps_values:
            case_benchmarks = {
                dirichlet: reports
                for (eps_value, dirichlet), reports in k0_laplacian_benchmarks.items()
                if eps_value == eps
            }
            if not case_benchmarks:
                continue
            suffix = "" if len(eps_values) == 1 else f"_eps_{_eps_tag(eps)}"
            figures.extend([
                (
                    f"benchmark_bars{suffix}.png",
                    plot_all_mass_benchmark_reports(
                        benchmarks,
                        k0_laplacian_benchmarks=case_benchmarks,
                        k0_laplacian_eps=eps,
                        show=False,
                        logscale=True,
                    ),
                ),
                (
                    f"benchmark_heatmap{suffix}.png",
                    plot_all_mass_benchmark_heatmaps(
                        benchmarks,
                        k0_laplacian_benchmarks=case_benchmarks,
                        k0_laplacian_eps=eps,
                        show=False,
                        logscale=True,
                    ),
                ),
            ])
    else:
        figures.extend([
            (
                "benchmark_bars.png",
                plot_all_mass_benchmark_reports(
                    benchmarks,
                    show=False,
                    logscale=True,
                ),
            ),
            (
                "benchmark_heatmap.png",
                plot_all_mass_benchmark_heatmaps(
                    benchmarks,
                    show=False,
                    logscale=True,
                ),
            ),
        ])
    if DENSE:
        for dirichlet in BENCHMARK_DIRICHLET_CASES:
            suffix = "dbc" if dirichlet else "free"
            figures.extend([
                (
                    f"spectrum_bars_{suffix}.png",
                    plot_mass_spectrum_reports(
                        spectra,
                        dirichlet=dirichlet,
                        show=False,
                        logscale=True,
                    ),
                ),
                (
                    f"spectrum_heatmap_{suffix}.png",
                    plot_mass_spectrum_heatmap(
                        spectra,
                        dirichlet=dirichlet,
                        show=False,
                    ),
                ),
            ])
    if DENSE and k0_laplacian_spectra:
        figures.extend([
            (
                "k0_laplacian_spectrum_bars.png",
                plot_k0_laplacian_spectrum_reports(
                    k0_laplacian_spectra,
                    show=False,
                ),
            ),
            (
                "k0_laplacian_spectrum_heatmap.png",
                plot_k0_laplacian_spectrum_heatmap(
                    k0_laplacian_spectra,
                    show=False,
                ),
            ),
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
    for dirichlet in BENCHMARK_DIRICHLET_CASES:
        for k in (0, 1, 2, 3):
            spectrum = summarize_mass_matrix(
                SEQ,
                OPERATORS,
                k=k,
                dirichlet=dirichlet,
            )
            ALL_MASS_SPECTRA[(k, dirichlet)] = spectrum
            print_mass_spectrum_report(spectrum)


# %% Inspect the scalar k=0 Hodge/Laplacian spectrum that shares the scalar mass interface
ALL_K0_LAPLACIAN_SPECTRA = {}
if RUN_K0_LAPLACIAN and DENSE:
    for dirichlet in K0_LAPLACE_DIRICHLET_CASES:
        spectrum = summarize_k0_laplacian(
            SEQ,
            OPERATORS,
            dirichlet=dirichlet,
        )
        ALL_K0_LAPLACIAN_SPECTRA[dirichlet] = spectrum
        print_k0_laplacian_spectrum_report(spectrum)


# %% Compare the production-facing preconditioner choices for all k and both boundary cases
ALL_BENCHMARKS = {}
for dirichlet in BENCHMARK_DIRICHLET_CASES:
    for k in (0, 1, 2, 3):
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
            f"tensor ranks={TENSOR_RANKS}, richardson/chebyshev steps={POLY_STEP_OPTIONS}"
        )
        print_mass_benchmark_reports(reports)


# %% Compare only the exact-block Chebyshev Schur variants requested for the inner solves
ALL_BLOCK_CHEB_BENCHMARKS = {}
if RUN_BLOCK_CHEB_BENCHMARK:
    for dirichlet in BENCHMARK_DIRICHLET_CASES:
        for k in (0, 1, 2):
            reports = benchmark_mass_block_chebyshev_preconditioners(
                SEQ,
                PRODUCTION_OPERATORS,
                k=k,
                dirichlet=dirichlet,
                n_rhs=8,
                seed=0,
            )
            ALL_BLOCK_CHEB_BENCHMARKS[(k, dirichlet)] = reports
            print("=" * 112)
            print(
                f"k={k} exact-block Chebyshev comparison: dirichlet={dirichlet}, "
                f"tensor rank={TENSOR_RANKS[-1]}"
            )
            print_mass_benchmark_reports(reports)


# %% Compare the scalar k=0 Hodge/Laplacian choices on the shared MassPreconditionerSpec interface
ALL_K0_LAPLACIAN_BENCHMARKS = {}
if RUN_K0_LAPLACIAN:
    for eps in K0_LAPLACE_EPS_LIST:
        for dirichlet in K0_LAPLACE_DIRICHLET_CASES:
            reports = benchmark_k0_laplacian_preconditioners(
                SEQ,
                PRODUCTION_OPERATORS,
                eps=eps,
                dirichlet=dirichlet,
                n_rhs=8,
                seed=0,
            )
            ALL_K0_LAPLACIAN_BENCHMARKS[(eps, dirichlet)] = reports
            print("=" * 112)
            print(
                f"k=0 Hodge/Laplacian comparison: eps={eps}, dirichlet={dirichlet}, "
                f"richardson/chebyshev steps={POLY_STEP_OPTIONS}"
            )
            print_k0_laplacian_benchmark_reports(reports)


# %% Visualize the benchmark summaries
if RUN_K0_LAPLACIAN:
    for eps in K0_LAPLACE_EPS_LIST:
        case_benchmarks = {
            dirichlet: ALL_K0_LAPLACIAN_BENCHMARKS[(eps, dirichlet)]
            for dirichlet in K0_LAPLACE_DIRICHLET_CASES
            if (eps, dirichlet) in ALL_K0_LAPLACIAN_BENCHMARKS
        }
        plot_all_mass_benchmark_reports(
            ALL_BENCHMARKS,
            k0_laplacian_benchmarks=case_benchmarks,
            k0_laplacian_eps=eps,
            logscale=True,
        )
        plot_all_mass_benchmark_heatmaps(
            ALL_BENCHMARKS,
            k0_laplacian_benchmarks=case_benchmarks,
            k0_laplacian_eps=eps,
            logscale=True,
        )
else:
    plot_all_mass_benchmark_reports(
        ALL_BENCHMARKS,
        logscale=True,
    )
    plot_all_mass_benchmark_heatmaps(
        ALL_BENCHMARKS,
        logscale=True,
    )
if DENSE:
    for dirichlet in BENCHMARK_DIRICHLET_CASES:
        plot_mass_spectrum_reports(
            ALL_MASS_SPECTRA,
            dirichlet=dirichlet,
            logscale=True,
        )
        plot_mass_spectrum_heatmap(
            ALL_MASS_SPECTRA,
            dirichlet=dirichlet,
        )
if DENSE and ALL_K0_LAPLACIAN_SPECTRA:
    plot_k0_laplacian_spectrum_reports(ALL_K0_LAPLACIAN_SPECTRA)
    plot_k0_laplacian_spectrum_heatmap(ALL_K0_LAPLACIAN_SPECTRA)


# %% Store results and plots
OUTPUT_DIR = save_mass_benchmark_artifacts(
    ALL_BENCHMARKS,
    ALL_MASS_SPECTRA,
    k0_laplacian_benchmarks=ALL_K0_LAPLACIAN_BENCHMARKS,
    k0_laplacian_spectra=ALL_K0_LAPLACIAN_SPECTRA,
)
print(f"saved benchmark artifacts to: {OUTPUT_DIR}")
# %%
