# %% [markdown]
# # Runtime Tuning Debug
#
# Interactive harness for the small-scope JIT-table runtime-tuning refactor.
#
# The intended workflow is to run this file cell-by-cell, just like the other
# interactive scripts in this folder. State is kept in module globals so you
# can rebuild, retune, inspect, and solve without restarting from scratch.
#
# It can also be driven from the command line with repeated `--stage` flags,
# for example:
#
# `python scripts/interactive/runtime_tuning_debug.py --stage build --stage tune --stage inspect`

# %%
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, fields as dataclass_fields, replace as dataclass_replace

import jax
import jax.numpy as jnp

from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import rotating_ellipse_map, toroid_map
from mrx.operators import (
    assemble_laplacian_operators,
    assemble_incidence_operators,
    assemble_mass_operators,
    assemble_tensor_mass_preconditioner,
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
    ns: tuple[int, int, int] = (4, 4, 4)
    p: int = 2
    tol: float = 1e-8
    maxiter: int = 200
    betti: tuple[int, int, int, int] = (1, 1, 0, 0)
    map_kind: str = "rotating_ellipse"
    torus_epsilon: float = 1.0 / 3.0
    torus_r0: float = 1.0
    rotating_eps: float = 0.2
    rotating_kappa: float = 1.2
    rotating_r0: float = 1.0
    rotating_nfp: int = 3
    hodge_eps: float = 1e-6
    tensor_rank: int = 2
    tensor_cp_tol: float = 1e-6
    tensor_cp_maxiter: int = 30


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


CONFIG = _resolve_experiment_config("MRX_RUNTIME_TUNING", ExperimentConfig())
SEQ = None
OPERATORS = None
MASS_SPEC = MassPreconditionerSpec(
    kind="richardson",
    steps=2,
)
HODGE_SPEC = MassPreconditionerSpec(
    kind="chebyshev",
    steps=2,
    smoother=MassPreconditionerSpec(kind="jacobi"),
)
# Keep one Schur combination fixed for the lifetime of the process so the
# stored Schur tuning is never reused across mismatched operators.
SCHUR_SPEC = SaddlePointPreconditionerSpec(
    mass=MassPreconditionerSpec(kind="tensor", surgery_schur=True),
    schur=SchurPreconditionerSpec(
        inner=MassPreconditionerSpec(kind="tensor"),
        outer=MassPreconditionerSpec(kind="chebyshev", steps=2),
    ),
    coupled=False,
)
DIFFUSION_SPEC = MassPreconditionerSpec(
    kind="chebyshev",
    steps=2,
)


# %% Helpers
def _build_map(config: ExperimentConfig):
    if config.map_kind == "rotating_ellipse":
        return rotating_ellipse_map(
            eps=config.rotating_eps,
            kappa=config.rotating_kappa,
            R0=config.rotating_r0,
            nfp=config.rotating_nfp,
        )
    if config.map_kind == "toroid":
        return toroid_map(
            epsilon=config.torus_epsilon,
            R0=config.torus_r0,
        )
    raise ValueError(f"Unsupported map_kind {config.map_kind!r}")


def _solver_status(info) -> tuple[str, int]:
    info_int = int(info)
    return ("converged" if info_int <= 0 else "not-converged", abs(info_int))


def _require_sequence() -> DeRhamSequence:
    if SEQ is None:
        raise RuntimeError("SEQ is not built yet; run build_case() first")
    return SEQ


def _require_operators():
    if OPERATORS is None:
        raise RuntimeError("OPERATORS are not assembled yet; run assemble_case() first")
    return OPERATORS


def _runtime_tuning_snapshot(operators) -> dict[str, object]:
    return {
        "mass": {
            "k3_free": {
                "lambda_max": operators.runtime_tuning.mass.k3.free.lambda_max,
                "lambda_min": operators.runtime_tuning.mass.k3.free.lambda_min,
            },
        },
        "scalar_hodge": {
            "k0_dbc_shifted": {
                "lambda_max": operators.runtime_tuning.scalar_hodge.k0.dbc.shifted.lambda_max,
                "lambda_min": operators.runtime_tuning.scalar_hodge.k0.dbc.shifted.lambda_min,
            },
        },
        "schur": {
            "k1_dbc_shifted": {
                "lambda_max": operators.runtime_tuning.schur.k1.dbc.shifted.lambda_max,
                "lambda_min": operators.runtime_tuning.schur.k1.dbc.shifted.lambda_min,
            },
        },
        "diffusion": {
            "k1_dbc_shifted": {
                "lambda_max": operators.runtime_tuning.diffusion.k1.dbc.shifted.lambda_max,
                "lambda_min": operators.runtime_tuning.diffusion.k1.dbc.shifted.lambda_min,
            },
        },
    }


# %% Stage 1: build the sequence
def build_case(config: ExperimentConfig = CONFIG):
    global SEQ, OPERATORS
    SEQ = DeRhamSequence(
        config.ns,
        (config.p, config.p, config.p),
        2 * config.p,
        ("clamped", "periodic", "periodic"),
        polar=True,
        tol=config.tol,
        maxiter=config.maxiter,
        betti_numbers=config.betti,
    )
    SEQ.evaluate_1d()
    SEQ.assemble_reference_mass_matrix()
    SEQ.set_map(_build_map(config))
    OPERATORS = None
    print(
        "built sequence:",
        f"ns={config.ns}, p={config.p}, map_kind={config.map_kind}, hodge_eps={config.hodge_eps}"
    )
    return SEQ


# %% Stage 2: assemble the minimal operator bundle
def assemble_case(config: ExperimentConfig = CONFIG):
    global OPERATORS
    seq = _require_sequence()
    operators = assemble_mass_operators(seq, seq.geometry, ks=(0, 1, 2, 3))
    operators = assemble_tensor_mass_preconditioner(
        seq,
        operators=operators,
        ks=(0, 1, 3),
        rank=config.tensor_rank,
        cp_kwargs={
            "tol": config.tensor_cp_tol,
            "maxiter": config.tensor_cp_maxiter,
        },
    )
    operators = assemble_incidence_operators(seq, operators=operators, ks=(0,))
    operators = assemble_incidence_operators(seq, operators=operators, ks=(1,))
    operators = assemble_laplacian_operators(seq, seq.geometry, operators=operators, ks=(0, 1))
    OPERATORS = operators
    print(
        "assembled operators:",
        f"mass ks=(0,1,2,3), incidence ks=(0,1), hodge ks=(0,1), tensor mass ks=(0,1,3)"
    )
    return OPERATORS


# %% Stage 3: estimate and store runtime tuning
def tune_case(config: ExperimentConfig = CONFIG):
    global OPERATORS
    seq = _require_sequence()
    operators = _require_operators()
    operators = seq.update_mass_runtime_tuning(
        3,
        dirichlet=False,
        operators=operators,
        preconditioner=MASS_SPEC,
    )
    operators = seq.update_scalar_hodge_runtime_tuning(
        0,
        eps=config.hodge_eps,
        dirichlet=True,
        operators=operators,
        preconditioner=HODGE_SPEC,
    )
    operators = seq.update_schur_runtime_tuning(
        1,
        eps=config.hodge_eps,
        dirichlet=True,
        operators=operators,
        preconditioner=SCHUR_SPEC,
    )
    operators = seq.update_diffusion_runtime_tuning(
        1,
        eps=config.hodge_eps,
        dirichlet=True,
        operators=operators,
        preconditioner=DIFFUSION_SPEC,
    )
    OPERATORS = operators
    print("stored runtime tuning")
    print_tuning_state()
    return OPERATORS


# %% Stage 4: inspect the stored tuning payload
def print_tuning_state():
    operators = _require_operators()
    snapshot = _runtime_tuning_snapshot(operators)
    print("runtime tuning snapshot:")
    for family, family_values in snapshot.items():
        print(f"  {family}:")
        for label, values in family_values.items():
            print(f"    {label}:")
            for key, value in values.items():
                print(f"      {key}: {value}")
    return snapshot


# %% Stage 5: run the mass solve using stored tuning
def run_mass_solve(rhs=None):
    seq = _require_sequence()
    operators = _require_operators()
    if rhs is None:
        rhs = jnp.ones(seq.n3)
    t0 = time.perf_counter()
    x, info = seq.apply_inverse_mass_matrix(
        rhs,
        3,
        dirichlet=False,
        operators=operators,
        preconditioner=MASS_SPEC,
        return_info=True,
    )
    jax.block_until_ready(x)
    elapsed_ms = 1e3 * (time.perf_counter() - t0)
    status, iters = _solver_status(info)
    print(
        "mass solve:",
        f"status={status}, iters={iters}, elapsed_ms={elapsed_ms:.2f}, finite={bool(jnp.all(jnp.isfinite(x)))}"
    )
    print(f"mass solution norm: {float(jnp.linalg.norm(x)):.6e}")
    return x, info


# %% Stage 6: run the scalar Hodge solve using stored tuning
def run_scalar_hodge_solve(config: ExperimentConfig = CONFIG, rhs=None):
    seq = _require_sequence()
    operators = _require_operators()
    if rhs is None:
        rhs = jnp.ones(seq.n0_dbc)
    t0 = time.perf_counter()
    x, info = seq.apply_inverse_shifted_laplacian(
        rhs,
        0,
        config.hodge_eps,
        dirichlet=True,
        operators=operators,
        preconditioner=HODGE_SPEC,
        return_info=True,
    )
    jax.block_until_ready(x)
    elapsed_ms = 1e3 * (time.perf_counter() - t0)
    status, iters = _solver_status(info)
    print(
        "scalar hodge solve:",
        f"status={status}, iters={iters}, elapsed_ms={elapsed_ms:.2f}, finite={bool(jnp.all(jnp.isfinite(x)))}"
    )
    print(f"scalar hodge solution norm: {float(jnp.linalg.norm(x)):.6e}")
    return x, info


# %% Stage 7: combined validation without raising immediately
def run_schur_solve(config: ExperimentConfig = CONFIG, rhs=None):
    seq = _require_sequence()
    operators = _require_operators()
    if rhs is None:
        rhs = jnp.ones(seq.n1_dbc)
    t0 = time.perf_counter()
    x, info = seq.apply_inverse_shifted_laplacian(
        rhs,
        1,
        config.hodge_eps,
        dirichlet=True,
        operators=operators,
        preconditioner=SCHUR_SPEC,
        return_info=True,
    )
    jax.block_until_ready(x)
    elapsed_ms = 1e3 * (time.perf_counter() - t0)
    status, iters = _solver_status(info)
    print(
        "schur solve:",
        f"status={status}, iters={iters}, elapsed_ms={elapsed_ms:.2f}, finite={bool(jnp.all(jnp.isfinite(x)))}"
    )
    print(f"schur solution norm: {float(jnp.linalg.norm(x)):.6e}")
    return x, info


# %% Stage 8: combined validation without raising immediately
def run_diffusion_solve(config: ExperimentConfig = CONFIG, rhs=None):
    seq = _require_sequence()
    operators = _require_operators()
    if rhs is None:
        rhs = jnp.ones(seq.n1_dbc)
    t0 = time.perf_counter()
    x, info = seq.apply_inverse_mass_plus_eps_laplace_matrix(
        rhs,
        1,
        config.hodge_eps,
        dirichlet=True,
        operators=operators,
        preconditioner=DIFFUSION_SPEC,
        return_info=True,
    )
    jax.block_until_ready(x)
    elapsed_ms = 1e3 * (time.perf_counter() - t0)
    status, iters = _solver_status(info)
    print(
        "diffusion solve:",
        f"status={status}, iters={iters}, elapsed_ms={elapsed_ms:.2f}, finite={bool(jnp.all(jnp.isfinite(x)))}"
    )
    print(f"diffusion solution norm: {float(jnp.linalg.norm(x)):.6e}")
    return x, info


# %% Stage 9: combined validation without raising immediately
def validate_case(config: ExperimentConfig = CONFIG, *, raise_on_failure: bool = False):
    operators = _require_operators()
    snapshot = _runtime_tuning_snapshot(operators)

    checks = {
        "mass_lambda_max_present": snapshot["mass"]["k3_free"]["lambda_max"] is not None,
        "hodge_lambda_min_present": snapshot["scalar_hodge"]["k0_dbc_shifted"]["lambda_min"] is not None,
        "hodge_lambda_max_present": snapshot["scalar_hodge"]["k0_dbc_shifted"]["lambda_max"] is not None,
        "schur_lambda_min_present": snapshot["schur"]["k1_dbc_shifted"]["lambda_min"] is not None,
        "schur_lambda_max_present": snapshot["schur"]["k1_dbc_shifted"]["lambda_max"] is not None,
        "diffusion_lambda_min_present": snapshot["diffusion"]["k1_dbc_shifted"]["lambda_min"] is not None,
        "diffusion_lambda_max_present": snapshot["diffusion"]["k1_dbc_shifted"]["lambda_max"] is not None,
    }

    mass_x, _ = run_mass_solve()
    hodge_x, _ = run_scalar_hodge_solve(config=config)
    schur_x, _ = run_schur_solve(config=config)
    diffusion_x, _ = run_diffusion_solve(config=config)
    checks["mass_solution_finite"] = bool(jnp.all(jnp.isfinite(mass_x)))
    checks["hodge_solution_finite"] = bool(jnp.all(jnp.isfinite(hodge_x)))
    checks["schur_solution_finite"] = bool(jnp.all(jnp.isfinite(schur_x)))
    checks["diffusion_solution_finite"] = bool(jnp.all(jnp.isfinite(diffusion_x)))

    print("validation summary:")
    failed = []
    for name, passed in checks.items():
        print(f"  {name}: {passed}")
        if not passed:
            failed.append(name)
    if failed and raise_on_failure:
        raise AssertionError(f"failed checks: {failed}")
    return checks


# %%
build_case()

# %%
assemble_case()

# %%
tune_case()

# %%
print_tuning_state()

# %%
run_mass_solve()

# %%
run_scalar_hodge_solve()

# %%
run_schur_solve()

# %%
run_diffusion_solve()

# %%
validate_case()

# %%

# %% CLI
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        action="append",
        choices=("build", "assemble", "tune", "inspect", "solve-mass", "solve-hodge", "solve-schur", "solve-diffusion", "validate", "all"),
    )
    args = parser.parse_args()

    stages = args.stage or ["all"]
    if "all" in stages:
        stages = ["build", "assemble", "tune", "inspect", "solve-mass", "solve-hodge", "solve-schur", "solve-diffusion", "validate"]

    for stage in stages:
        if stage == "build":
            build_case()
        elif stage == "assemble":
            if SEQ is None:
                build_case()
            assemble_case()
        elif stage == "tune":
            if SEQ is None:
                build_case()
            if OPERATORS is None:
                assemble_case()
            tune_case()
        elif stage == "inspect":
            print_tuning_state()
        elif stage == "solve-mass":
            run_mass_solve()
        elif stage == "solve-hodge":
            run_scalar_hodge_solve()
        elif stage == "solve-schur":
            run_schur_solve()
        elif stage == "solve-diffusion":
            run_diffusion_solve()
        elif stage == "validate":
            validate_case()


if __name__ == "__main__":
    main()
# %%
