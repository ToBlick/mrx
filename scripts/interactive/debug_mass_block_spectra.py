from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from mrx.assembly import assemble_scalar_tp
from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import rotating_ellipse_map, toroid_map
from mrx.operators import assemble_mass_operators, assemble_tensor_mass_preconditioner, dense_mass_matrix
from mrx.preconditioners import (
    _assemble_weighted_1d_mass,
    _apply_tensor_diagonal_block,
    _apply_tensor_exact_block,
    _cp_als_3tensor,
    _core_size,
    _estimate_chebyshev_lanczos_bounds_apply,
    _k1_diagonal_metric_tensors,
    _k2_diagonal_metric_tensors,
    _restrict_radial_mass,
    _select_mass_surgery_factors,
    _select_mass_tensor_factors,
    get_mass_jacobi_diaginv,
)

jax.config.update("jax_enable_x64", True)


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
INTERACTIVE_OUTPUT_DIR = WORKSPACE_ROOT / "outputs" / "interactive"
DEFAULT_SAVE_PREFIX = INTERACTIVE_OUTPUT_DIR / "mass_block_spectra"


@dataclass(frozen=True)
class Config:
    ns: tuple[int, int, int] = (4, 4, 4)
    p: int = 3
    tol: float = 1e-9
    maxiter: int = 1000
    betti: tuple[int, int, int, int] = (1, 1, 0, 0)
    dirichlet: bool = True
    map_kind: str = "rotating_ellipse"
    torus_epsilon: float = 1.0 / 3.0
    torus_r0: float = 1.0
    rotating_eps: float = 0.33
    rotating_kappa: float = 1.4
    rotating_r0: float = 1.0
    rotating_nfp: int = 3
    ranks: tuple[int, ...] = (1,)
    cp_maxiter: int = 150
    cp_tol: float = 1e-9
    cp_ridge: float = 1e-12
    richardson_steps: int = 0
    richardson_omega: float = 1.0
    rank1_als_maxiter: int = 50
    rank1_als_tol: float = 1e-10
    lanczos_iterations: int = 16
    lanczos_max_eig_inflation: float = 1.1
    lanczos_min_eig_deflation: float = 0.85
    lanczos_min_eig_floor_fraction: float = 1e-3
    show: bool = False
    save_prefix: str | None = str(DEFAULT_SAVE_PREFIX)


def _parse_bool(value: str) -> bool:
    text = value.strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value from {value!r}")


def _parse_int_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def _resolve_save_prefix(raw_value: str | None) -> str:
    if raw_value is None:
        return str(DEFAULT_SAVE_PREFIX)
    candidate = Path(raw_value)
    base_name = candidate.name if candidate.name else DEFAULT_SAVE_PREFIX.name
    return str(INTERACTIVE_OUTPUT_DIR / base_name)


def _parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description=(
            "Dense mass-matrix block spectrum diagnostic for tensor smoothers and one-term inverse models."
        )
    )
    parser.add_argument("--ns", type=_parse_int_tuple, default=Config.ns)
    parser.add_argument("--p", type=int, default=Config.p)
    parser.add_argument("--dirichlet", type=_parse_bool, default=Config.dirichlet)
    parser.add_argument("--map-kind", choices=("rotating_ellipse", "toroidal"), default=Config.map_kind)
    parser.add_argument("--torus-epsilon", type=float, default=Config.torus_epsilon)
    parser.add_argument("--torus-r0", type=float, default=Config.torus_r0)
    parser.add_argument("--rotating-eps", type=float, default=Config.rotating_eps)
    parser.add_argument("--rotating-kappa", type=float, default=Config.rotating_kappa)
    parser.add_argument("--rotating-r0", type=float, default=Config.rotating_r0)
    parser.add_argument("--rotating-nfp", type=int, default=Config.rotating_nfp)
    parser.add_argument("--ranks", type=_parse_int_tuple, default=Config.ranks)
    parser.add_argument("--cp-maxiter", type=int, default=Config.cp_maxiter)
    parser.add_argument("--cp-tol", type=float, default=Config.cp_tol)
    parser.add_argument("--cp-ridge", type=float, default=Config.cp_ridge)
    parser.add_argument("--richardson-steps", type=int, default=Config.richardson_steps)
    parser.add_argument("--richardson-omega", type=float, default=Config.richardson_omega)
    parser.add_argument("--rank1-als-maxiter", type=int, default=Config.rank1_als_maxiter)
    parser.add_argument("--rank1-als-tol", type=float, default=Config.rank1_als_tol)
    parser.add_argument("--lanczos-iterations", type=int, default=Config.lanczos_iterations)
    parser.add_argument("--lanczos-max-eig-inflation", type=float, default=Config.lanczos_max_eig_inflation)
    parser.add_argument("--lanczos-min-eig-deflation", type=float, default=Config.lanczos_min_eig_deflation)
    parser.add_argument("--lanczos-min-eig-floor-fraction", type=float, default=Config.lanczos_min_eig_floor_fraction)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--save-prefix")
    args = parser.parse_args()
    return Config(
        ns=tuple(args.ns),
        p=args.p,
        dirichlet=args.dirichlet,
        map_kind=args.map_kind,
        torus_epsilon=args.torus_epsilon,
        torus_r0=args.torus_r0,
        rotating_eps=args.rotating_eps,
        rotating_kappa=args.rotating_kappa,
        rotating_r0=args.rotating_r0,
        rotating_nfp=args.rotating_nfp,
        ranks=tuple(args.ranks),
        cp_maxiter=args.cp_maxiter,
        cp_tol=args.cp_tol,
        cp_ridge=args.cp_ridge,
        richardson_steps=args.richardson_steps,
        richardson_omega=args.richardson_omega,
        rank1_als_maxiter=args.rank1_als_maxiter,
        rank1_als_tol=args.rank1_als_tol,
        lanczos_iterations=args.lanczos_iterations,
        lanczos_max_eig_inflation=args.lanczos_max_eig_inflation,
        lanczos_min_eig_deflation=args.lanczos_min_eig_deflation,
        lanczos_min_eig_floor_fraction=args.lanczos_min_eig_floor_fraction,
        show=not args.no_show,
        save_prefix=_resolve_save_prefix(args.save_prefix),
    )


def _build_map(config: Config):
    if config.map_kind == "toroidal":
        return toroid_map(epsilon=config.torus_epsilon, R0=config.torus_r0)
    return rotating_ellipse_map(
        eps=config.rotating_eps,
        kappa=config.rotating_kappa,
        R0=config.rotating_r0,
        nfp=config.rotating_nfp,
    )


def _build_seq(config: Config):
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
    # seq.assemble_reference_mass_matrix()
    seq.set_map(_build_map(config))
    return seq


def _assemble_base_operators(seq):
    operators = assemble_mass_operators(seq, seq.geometry, ks=(0, 1, 2, 3))
    return seq.set_operators(operators, sync_legacy=False)


def _assemble_ranked_operators(seq, base_operators, config: Config, rank: int):
    operators = assemble_tensor_mass_preconditioner(
        seq,
        operators=base_operators,
        ks=(0, 1, 2, 3),
        rank=rank,
        cp_kwargs={
            "maxiter": config.cp_maxiter,
            "tol": config.cp_tol,
            "ridge": config.cp_ridge,
            "richardson_steps": config.richardson_steps,
            "richardson_omega": config.richardson_omega,
            "block_chebyshev_steps": 0,
        },
    )
    return seq.set_operators(operators, sync_legacy=False)


def _assemble_production_operators(seq, base_operators):
    operators = assemble_tensor_mass_preconditioner(
        seq,
        operators=base_operators,
        ks=(0, 1, 2, 3),
    )
    return seq.set_operators(operators, sync_legacy=False)


def _matrix_from_apply(apply, size: int) -> jnp.ndarray:
    basis = jnp.eye(size, dtype=jnp.float64)
    return jax.vmap(apply, in_axes=1, out_axes=1)(basis)


def _matrix_from_terms(term_r, term_t, term_z) -> jnp.ndarray:
    matrix = jnp.zeros(
        (term_r[0].shape[0] * term_t[0].shape[0] * term_z[0].shape[0],) * 2,
        dtype=term_r[0].dtype,
    )
    for mass_r, mass_t, mass_z in zip(term_r, term_t, term_z):
        matrix = matrix + jnp.kron(jnp.kron(mass_r, mass_t), mass_z)
    return 0.5 * (matrix + matrix.T)


def _symmetrize(matrix: jnp.ndarray) -> jnp.ndarray:
    return 0.5 * (matrix + matrix.T)


def _normalize_factor(matrix: jnp.ndarray) -> jnp.ndarray:
    matrix = _symmetrize(matrix)
    norm = float(jnp.linalg.norm(matrix))
    if norm <= np.finfo(np.float64).tiny:
        return matrix
    return matrix / norm


def _normalize_columns(factor: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    norms = jnp.linalg.norm(factor, axis=0)
    safe_norms = jnp.where(norms > 0, norms, 1.0)
    return factor / safe_norms, norms


def _mode_unfold_nd(tensor: jnp.ndarray, mode: int) -> jnp.ndarray:
    return jnp.moveaxis(tensor, mode, 0).reshape(tensor.shape[mode], -1)


def _khatri_rao(left: jnp.ndarray, right: jnp.ndarray) -> jnp.ndarray:
    return (left[:, None, :] * right[None, :, :]).reshape(left.shape[0] * right.shape[0], left.shape[1])


def _init_cp_factor(unfolded: jnp.ndarray, rank: int, *, seed: int) -> jnp.ndarray:
    leading = jnp.linalg.svd(unfolded, full_matrices=False)[0]
    n_cols = min(rank, leading.shape[1])
    factor = leading[:, :n_cols]
    if rank > n_cols:
        extra = jax.random.normal(
            jax.random.PRNGKey(seed),
            (unfolded.shape[0], rank - n_cols),
            dtype=unfolded.dtype,
        )
        factor = jnp.concatenate([factor, extra], axis=1)
    return _normalize_columns(factor)[0]


def _reconstruct_cp_4tensor(
    weights: jnp.ndarray,
    factors: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> jnp.ndarray:
    factor_component, factor_theta, factor_r, factor_z = factors
    return jnp.einsum("r,ar,br,cr,dr->abcd", weights, factor_component, factor_theta, factor_r, factor_z)


def _cp_als_4tensor_shared(
    tensor: jnp.ndarray,
    rank: int,
    *,
    maxiter: int,
    tol: float,
    ridge: float,
) -> tuple[
    jnp.ndarray,
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    float,
    tuple[float, ...],
    float,
    int,
]:
    if tensor.ndim != 4:
        raise ValueError(f"Shared CP-ALS expects a 4-tensor, got shape {tensor.shape}")
    if rank < 1:
        raise ValueError(f"Requested CP rank {rank} must be positive")

    unfolded_component = _mode_unfold_nd(tensor, 0)
    unfolded_theta = _mode_unfold_nd(tensor, 1)
    unfolded_r = _mode_unfold_nd(tensor, 2)
    unfolded_z = _mode_unfold_nd(tensor, 3)

    factor_component = _init_cp_factor(unfolded_component, rank, seed=0)
    factor_theta = _init_cp_factor(unfolded_theta, rank, seed=1)
    factor_r = _init_cp_factor(unfolded_r, rank, seed=2)
    factor_z = _init_cp_factor(unfolded_z, rank, seed=3)
    weights = jnp.ones((rank,), dtype=tensor.dtype)

    eye = jnp.eye(rank, dtype=tensor.dtype)
    previous_error = jnp.inf
    relative_error = jnp.inf
    final_delta = jnp.inf
    n_iterations = 0
    field_relative_errors = tuple(float("inf") for _ in range(tensor.shape[0]))

    for iteration in range(maxiter):
        factor_z_eff = factor_z * weights[None, :]

        khatri_rao_trz = _khatri_rao(factor_theta, _khatri_rao(factor_r, factor_z_eff))
        gram_trz = (factor_theta.T @ factor_theta) * (factor_r.T @ factor_r) * (factor_z_eff.T @ factor_z_eff)
        factor_component_raw = jnp.linalg.solve(
            gram_trz + ridge * eye,
            (unfolded_component @ khatri_rao_trz).T,
        ).T

        khatri_rao_crz = _khatri_rao(factor_component_raw, _khatri_rao(factor_r, factor_z_eff))
        gram_crz = (factor_component_raw.T @ factor_component_raw) * (factor_r.T @ factor_r) * (factor_z_eff.T @ factor_z_eff)
        factor_theta_raw = jnp.linalg.solve(
            gram_crz + ridge * eye,
            (unfolded_theta @ khatri_rao_crz).T,
        ).T

        khatri_rao_ctz = _khatri_rao(factor_component_raw, _khatri_rao(factor_theta_raw, factor_z_eff))
        gram_ctz = (factor_component_raw.T @ factor_component_raw) * (factor_theta_raw.T @ factor_theta_raw) * (factor_z_eff.T @ factor_z_eff)
        factor_r_raw = jnp.linalg.solve(
            gram_ctz + ridge * eye,
            (unfolded_r @ khatri_rao_ctz).T,
        ).T

        khatri_rao_ctr = _khatri_rao(factor_component_raw, _khatri_rao(factor_theta_raw, factor_r_raw))
        gram_ctr = (factor_component_raw.T @ factor_component_raw) * (factor_theta_raw.T @ factor_theta_raw) * (factor_r_raw.T @ factor_r_raw)
        factor_z_eff_raw = jnp.linalg.solve(
            gram_ctr + ridge * eye,
            (unfolded_z @ khatri_rao_ctr).T,
        ).T

        factor_component, component_norms = _normalize_columns(factor_component_raw)
        factor_theta, theta_norms = _normalize_columns(factor_theta_raw)
        factor_r, r_norms = _normalize_columns(factor_r_raw)
        factor_z_temp = factor_z_eff_raw * (component_norms * theta_norms * r_norms)[None, :]
        factor_z, weights = _normalize_columns(factor_z_temp)

        reconstruction = _reconstruct_cp_4tensor(weights, (factor_component, factor_theta, factor_r, factor_z))
        relative_error = float(
            jnp.linalg.norm(reconstruction - tensor) / jnp.maximum(jnp.linalg.norm(tensor), 1.0)
        )
        field_relative_errors = tuple(
            float(
                jnp.linalg.norm(reconstruction[idx] - tensor[idx]) / jnp.maximum(jnp.linalg.norm(tensor[idx]), 1.0)
            )
            for idx in range(tensor.shape[0])
        )
        final_delta = abs(relative_error - previous_error) if previous_error < jnp.inf else jnp.inf
        previous_error = relative_error
        n_iterations = iteration + 1
        if final_delta < tol:
            break

    return (
        weights,
        (factor_component, factor_theta, factor_r, factor_z),
        relative_error,
        field_relative_errors,
        final_delta,
        n_iterations,
    )


def _build_aligned_vector_mass_models(
    tensor_fields: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    block_specs: tuple[dict[str, object], dict[str, object], dict[str, object]],
    *,
    rank: int,
    config: Config,
    radial_baselines: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None = None,
    field_baselines: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None = None,
) -> dict[str, object]:
    if field_baselines is not None and radial_baselines is not None:
        raise ValueError("Use either radial_baselines or field_baselines, not both")

    if radial_baselines is None:
        radial_baselines = tuple(jnp.ones_like(seq.quad.x_x, dtype=jnp.float64) for _ in range(3))
    if field_baselines is None:
        field_baselines = tuple(baseline[None, :, None] for baseline in radial_baselines)

    corrected_fields = tuple(
        field / baseline
        for field, baseline in zip(tensor_fields, field_baselines, strict=True)
    )
    stacked = jnp.stack(corrected_fields, axis=0)
    weights, factors, relative_error, field_relative_errors, _, _ = _cp_als_4tensor_shared(
        stacked,
        rank,
        maxiter=config.cp_maxiter,
        tol=config.cp_tol,
        ridge=config.cp_ridge,
    )
    factor_component, factor_theta, factor_r, factor_z = factors

    blocks = {}
    for block_idx, spec in enumerate(block_specs):
        nr, _, _ = spec["shape"]
        term_r = []
        term_t = []
        term_z = []
        for term_idx in range(rank):
            scale = weights[term_idx] * factor_component[block_idx, term_idx]
            if spec.get("assemble_full_field", False):
                term_field = (
                    scale
                    * field_baselines[block_idx]
                    * factor_theta[:, term_idx][:, None, None]
                    * factor_r[:, term_idx][None, :, None]
                    * factor_z[:, term_idx][None, None, :]
                )
                term_r.append(term_field)
            else:
                raw_mass_r = _assemble_weighted_1d_mass(
                    spec["radial_basis"],
                    seq.quad.w_x * (scale * radial_baselines[block_idx] * factor_r[:, term_idx]),
                )
                mass_r = _restrict_radial_mass(raw_mass_r, spec["radial_start"], nr)
                mass_t = _assemble_weighted_1d_mass(spec["theta_basis"], seq.quad.w_y * factor_theta[:, term_idx])
                mass_z = _assemble_weighted_1d_mass(spec["zeta_basis"], seq.quad.w_z * factor_z[:, term_idx])
                term_r.append(mass_r)
                term_t.append(mass_t)
                term_z.append(mass_z)
        if spec.get("assemble_full_field", False):
            model_matrix = _assemble_scalar_block_from_fields(spec, tuple(term_r))
        else:
            model_matrix = _matrix_from_terms(tuple(term_r), tuple(term_t), tuple(term_z))
        blocks[spec["name"]] = {
            "model_matrix": model_matrix,
            "field_relative_error": field_relative_errors[block_idx],
        }

    return {
        "relative_error": relative_error,
        "blocks": blocks,
    }


def _build_independent_vector_mass_models(
    tensor_fields: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    block_specs: tuple[dict[str, object], dict[str, object], dict[str, object]],
    *,
    rank: int,
    config: Config,
    radial_baselines: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None = None,
    field_baselines: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None = None,
) -> dict[str, object]:
    if field_baselines is not None and radial_baselines is not None:
        raise ValueError("Use either radial_baselines or field_baselines, not both")

    if radial_baselines is None:
        radial_baselines = tuple(jnp.ones_like(seq.quad.x_x, dtype=jnp.float64) for _ in range(3))
    if field_baselines is None:
        field_baselines = tuple(None for _ in range(3))

    blocks = {}
    for field, spec, radial_baseline, field_baseline in zip(
        tensor_fields,
        block_specs,
        radial_baselines,
        field_baselines,
        strict=True,
    ):
        model_matrix = _build_scalar_reference_model(
            field,
            rank=rank,
            radial_basis=spec["radial_basis"],
            theta_basis=spec["theta_basis"],
            zeta_basis=spec["zeta_basis"],
            radial_start=spec["radial_start"],
            shape=spec["shape"],
            full_shape=spec["full_shape"],
            hw_r=spec["hw_r"],
            hw_t=spec["hw_t"],
            hw_z=spec["hw_z"],
            config=config,
            radial_baseline=radial_baseline,
            field_baseline=field_baseline,
        )
        blocks[spec["name"]] = {"model_matrix": model_matrix}

    return {"blocks": blocks}


def _mean_one(values: jnp.ndarray) -> jnp.ndarray:
    mean_value = jnp.mean(values)
    safe_mean = jnp.where(jnp.abs(mean_value) > 0, mean_value, 1.0)
    return values / safe_mean


def _torus_radial_baselines(kind: str) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    safe_r = jnp.maximum(jnp.asarray(seq.quad.x_x, dtype=jnp.float64), 1e-8)
    if kind == "m1":
        return (
            _mean_one(safe_r),
            _mean_one(1.0 / safe_r),
            _mean_one(safe_r),
        )
    if kind == "m2":
        return (
            _mean_one(1.0 / safe_r),
            _mean_one(safe_r),
            _mean_one(1.0 / safe_r),
        )
    raise ValueError(f"Unknown torus radial baseline kind {kind!r}")


def _closest_circular_torus_parameters(config: Config) -> tuple[float, float]:
    if config.map_kind == "toroidal":
        return float(config.torus_epsilon), float(config.torus_r0)
    return float(config.rotating_eps), float(config.rotating_r0)


def _torus_exact_field_baselines(kind: str, config: Config) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    epsilon, r0 = _closest_circular_torus_parameters(config)
    theta = jnp.asarray(seq.quad.x_y, dtype=jnp.float64)
    radius = jnp.maximum(jnp.asarray(seq.quad.x_x, dtype=jnp.float64), 1e-8)
    zeta = jnp.asarray(seq.quad.x_z, dtype=jnp.float64)
    del zeta
    theta_grid = theta[:, None, None]
    radius_grid = radius[None, :, None]
    big_r = r0 + epsilon * radius_grid * jnp.cos(2.0 * jnp.pi * theta_grid)

    if kind == "m1":
        return (
            _mean_one(4.0 * (jnp.pi ** 2) * radius_grid * big_r),
            _mean_one(big_r / radius_grid),
            _mean_one((epsilon ** 2) * radius_grid / big_r),
        )
    if kind == "m2":
        return (
            _mean_one(1.0 / (4.0 * (jnp.pi ** 2) * radius_grid * big_r)),
            _mean_one(radius_grid / big_r),
            _mean_one(big_r / ((epsilon ** 2) * radius_grid)),
        )
    raise ValueError(f"Unknown torus exact baseline kind {kind!r}")


def _assemble_scalar_block_from_fields(spec: dict[str, object], term_fields: tuple[jnp.ndarray, ...]) -> jnp.ndarray:
    field = sum(term_fields)
    full_shape = spec["full_shape"]
    weighted_field = field * seq.quad.w.reshape(seq.quad.ny, seq.quad.nx, seq.quad.nz)
    matrix = jnp.asarray(
        assemble_scalar_tp(
            spec["radial_basis"],
            spec["theta_basis"],
            spec["zeta_basis"],
            spec["radial_basis"],
            spec["theta_basis"],
            spec["zeta_basis"],
            weighted_field.reshape(-1),
            (seq.quad.ny, seq.quad.nx, seq.quad.nz),
            full_shape,
            spec["hw_r"],
            spec["hw_t"],
            spec["hw_z"],
        ).todense()
    )
    radial_start = spec["radial_start"]
    nr, nt, nz = spec["shape"]
    radial_indices = jnp.arange(radial_start, radial_start + nr)
    theta_indices = jnp.arange(nt)
    zeta_indices = jnp.arange(nz)
    grid = jnp.meshgrid(radial_indices, theta_indices, zeta_indices, indexing="ij")
    keep = jnp.ravel_multi_index(grid, full_shape)
    keep = jnp.asarray(keep).reshape(-1)
    restricted = matrix[keep][:, keep]
    return _symmetrize(restricted)


def _build_scalar_reference_model(
    tensor_field: jnp.ndarray,
    *,
    rank: int,
    radial_basis: jnp.ndarray,
    theta_basis: jnp.ndarray,
    zeta_basis: jnp.ndarray,
    radial_start: int,
    shape: tuple[int, int, int],
    full_shape: tuple[int, int, int],
    hw_r: int,
    hw_t: int,
    hw_z: int,
    config: Config,
    radial_baseline: jnp.ndarray | None = None,
    field_baseline: jnp.ndarray | None = None,
) -> jnp.ndarray:
    if radial_baseline is not None and field_baseline is not None:
        raise ValueError("Use either radial_baseline or field_baseline, not both")

    if field_baseline is None:
        if radial_baseline is None:
            field_baseline = jnp.ones((seq.quad.ny, seq.quad.nx, seq.quad.nz), dtype=jnp.float64)
        else:
            field_baseline = radial_baseline[None, :, None]

    corrected_field = tensor_field / field_baseline
    weights, factors, _, _, _ = _cp_als_3tensor(
        corrected_field,
        rank,
        maxiter=config.cp_maxiter,
        tol=config.cp_tol,
        ridge=config.cp_ridge,
    )
    factor_theta, factor_r, factor_z = factors

    if radial_baseline is not None:
        term_r = []
        term_t = []
        term_z = []
        nr, _, _ = shape
        for idx in range(rank):
            scale = weights[idx]
            raw_mass_r = _assemble_weighted_1d_mass(
                radial_basis,
                seq.quad.w_x * (scale * radial_baseline * factor_r[:, idx]),
            )
            mass_r = _restrict_radial_mass(raw_mass_r, radial_start, nr)
            mass_t = _assemble_weighted_1d_mass(theta_basis, seq.quad.w_y * factor_theta[:, idx])
            mass_z = _assemble_weighted_1d_mass(zeta_basis, seq.quad.w_z * factor_z[:, idx])
            term_r.append(mass_r)
            term_t.append(mass_t)
            term_z.append(mass_z)
        return _matrix_from_terms(tuple(term_r), tuple(term_t), tuple(term_z))

    term_fields = []
    for idx in range(rank):
        term_fields.append(
            weights[idx]
            * field_baseline
            * factor_theta[:, idx][:, None, None]
            * factor_r[:, idx][None, :, None]
            * factor_z[:, idx][None, None, :]
        )
    return _assemble_scalar_block_from_fields(
        {
            "shape": shape,
            "full_shape": full_shape,
            "radial_basis": radial_basis,
            "theta_basis": theta_basis,
            "zeta_basis": zeta_basis,
            "radial_start": radial_start,
            "hw_r": hw_r,
            "hw_t": hw_t,
            "hw_z": hw_z,
        },
        tuple(term_fields),
    )


def _build_aligned_m1_models(rank_blocks: dict[str, object], config: Config, rank: int) -> dict[str, object]:
    tensors = _k1_diagonal_metric_tensors(seq)
    block_specs = (
        {
            "name": "arr",
            "shape": rank_blocks["arr"].shape,
            "full_shape": seq.basis_1.shape[0],
            "radial_basis": seq.d_basis_r_jk,
            "theta_basis": seq.basis_t_jk,
            "zeta_basis": seq.basis_z_jk,
            "radial_start": 1,
            "hw_r": seq.basis_1.pr,
            "hw_t": seq.basis_1.pt,
            "hw_z": seq.basis_1.pz,
        },
        {
            "name": "theta",
            "shape": rank_blocks["theta"].shape,
            "full_shape": seq.basis_1.shape[1],
            "radial_basis": seq.basis_r_jk,
            "theta_basis": seq.d_basis_t_jk,
            "zeta_basis": seq.basis_z_jk,
            "radial_start": 2,
            "hw_r": seq.basis_1.pr,
            "hw_t": seq.basis_1.pt,
            "hw_z": seq.basis_1.pz,
        },
        {
            "name": "zeta",
            "shape": rank_blocks["zeta"].shape,
            "full_shape": seq.basis_1.shape[2],
            "radial_basis": seq.basis_r_jk,
            "theta_basis": seq.basis_t_jk,
            "zeta_basis": seq.d_basis_z_jk,
            "radial_start": 2,
            "hw_r": seq.basis_1.pr,
            "hw_t": seq.basis_1.pt,
            "hw_z": seq.basis_1.pz,
        },
    )
    return _build_aligned_vector_mass_models(
        (tensors["alpha_rr"], tensors["alpha_thetatheta"], tensors["alpha_zetazeta"]),
        block_specs,
        rank=rank,
        config=config,
    )


def _build_torus_scaled_m1_models(rank_blocks: dict[str, object], config: Config, rank: int) -> dict[str, object]:
    tensors = _k1_diagonal_metric_tensors(seq)
    block_specs = (
        {
            "name": "arr",
            "shape": rank_blocks["arr"].shape,
            "full_shape": seq.basis_1.shape[0],
            "radial_basis": seq.d_basis_r_jk,
            "theta_basis": seq.basis_t_jk,
            "zeta_basis": seq.basis_z_jk,
            "radial_start": 1,
            "hw_r": seq.basis_1.pr,
            "hw_t": seq.basis_1.pt,
            "hw_z": seq.basis_1.pz,
        },
        {
            "name": "theta",
            "shape": rank_blocks["theta"].shape,
            "full_shape": seq.basis_1.shape[1],
            "radial_basis": seq.basis_r_jk,
            "theta_basis": seq.d_basis_t_jk,
            "zeta_basis": seq.basis_z_jk,
            "radial_start": 2,
            "hw_r": seq.basis_1.pr,
            "hw_t": seq.basis_1.pt,
            "hw_z": seq.basis_1.pz,
        },
        {
            "name": "zeta",
            "shape": rank_blocks["zeta"].shape,
            "full_shape": seq.basis_1.shape[2],
            "radial_basis": seq.basis_r_jk,
            "theta_basis": seq.basis_t_jk,
            "zeta_basis": seq.d_basis_z_jk,
            "radial_start": 2,
            "hw_r": seq.basis_1.pr,
            "hw_t": seq.basis_1.pt,
            "hw_z": seq.basis_1.pz,
        },
    )
    return _build_aligned_vector_mass_models(
        (tensors["alpha_rr"], tensors["alpha_thetatheta"], tensors["alpha_zetazeta"]),
        block_specs,
        rank=rank,
        config=config,
        radial_baselines=_torus_radial_baselines("m1"),
    )


def _build_independent_torus_scaled_m1_models(
    rank_blocks: dict[str, object], config: Config, rank: int
) -> dict[str, object]:
    tensors = _k1_diagonal_metric_tensors(seq)
    block_specs = (
        {
            "name": "arr",
            "shape": rank_blocks["arr"].shape,
            "full_shape": seq.basis_1.shape[0],
            "radial_basis": seq.d_basis_r_jk,
            "theta_basis": seq.basis_t_jk,
            "zeta_basis": seq.basis_z_jk,
            "radial_start": 1,
            "hw_r": seq.basis_1.pr,
            "hw_t": seq.basis_1.pt,
            "hw_z": seq.basis_1.pz,
        },
        {
            "name": "theta",
            "shape": rank_blocks["theta"].shape,
            "full_shape": seq.basis_1.shape[1],
            "radial_basis": seq.basis_r_jk,
            "theta_basis": seq.d_basis_t_jk,
            "zeta_basis": seq.basis_z_jk,
            "radial_start": 2,
            "hw_r": seq.basis_1.pr,
            "hw_t": seq.basis_1.pt,
            "hw_z": seq.basis_1.pz,
        },
        {
            "name": "zeta",
            "shape": rank_blocks["zeta"].shape,
            "full_shape": seq.basis_1.shape[2],
            "radial_basis": seq.basis_r_jk,
            "theta_basis": seq.basis_t_jk,
            "zeta_basis": seq.d_basis_z_jk,
            "radial_start": 2,
            "hw_r": seq.basis_1.pr,
            "hw_t": seq.basis_1.pt,
            "hw_z": seq.basis_1.pz,
        },
    )
    return _build_independent_vector_mass_models(
        (tensors["alpha_rr"], tensors["alpha_thetatheta"], tensors["alpha_zetazeta"]),
        block_specs,
        rank=rank,
        config=config,
        radial_baselines=_torus_radial_baselines("m1"),
    )


def _build_exact_torus_m1_models(rank_blocks: dict[str, object], config: Config, rank: int) -> dict[str, object]:
    tensors = _k1_diagonal_metric_tensors(seq)
    block_specs = (
        {
            "name": "arr",
            "shape": rank_blocks["arr"].shape,
            "full_shape": seq.basis_1.shape[0],
            "radial_basis": seq.d_basis_r_jk,
            "theta_basis": seq.basis_t_jk,
            "zeta_basis": seq.basis_z_jk,
            "radial_start": 1,
            "hw_r": seq.basis_1.pr,
            "hw_t": seq.basis_1.pt,
            "hw_z": seq.basis_1.pz,
            "assemble_full_field": True,
        },
        {
            "name": "theta",
            "shape": rank_blocks["theta"].shape,
            "full_shape": seq.basis_1.shape[1],
            "radial_basis": seq.basis_r_jk,
            "theta_basis": seq.d_basis_t_jk,
            "zeta_basis": seq.basis_z_jk,
            "radial_start": 2,
            "hw_r": seq.basis_1.pr,
            "hw_t": seq.basis_1.pt,
            "hw_z": seq.basis_1.pz,
            "assemble_full_field": True,
        },
        {
            "name": "zeta",
            "shape": rank_blocks["zeta"].shape,
            "full_shape": seq.basis_1.shape[2],
            "radial_basis": seq.basis_r_jk,
            "theta_basis": seq.basis_t_jk,
            "zeta_basis": seq.d_basis_z_jk,
            "radial_start": 2,
            "hw_r": seq.basis_1.pr,
            "hw_t": seq.basis_1.pt,
            "hw_z": seq.basis_1.pz,
            "assemble_full_field": True,
        },
    )
    return _build_aligned_vector_mass_models(
        (tensors["alpha_rr"], tensors["alpha_thetatheta"], tensors["alpha_zetazeta"]),
        block_specs,
        rank=rank,
        config=config,
        field_baselines=_torus_exact_field_baselines("m1", config),
    )


def _build_aligned_m2_models(rank_blocks: dict[str, object], config: Config, rank: int) -> dict[str, object]:
    tensors = _k2_diagonal_metric_tensors(seq)
    block_specs = (
        {
            "name": "r_bulk",
            "shape": rank_blocks["r_bulk"].shape,
            "full_shape": seq.basis_2.shape[0],
            "radial_basis": seq.basis_r_jk,
            "theta_basis": seq.d_basis_t_jk,
            "zeta_basis": seq.d_basis_z_jk,
            "radial_start": 2,
            "hw_r": seq.basis_2.pr,
            "hw_t": seq.basis_2.pt,
            "hw_z": seq.basis_2.pz,
        },
        {
            "name": "theta",
            "shape": rank_blocks["theta"].shape,
            "full_shape": seq.basis_2.shape[1],
            "radial_basis": seq.d_basis_r_jk,
            "theta_basis": seq.basis_t_jk,
            "zeta_basis": seq.d_basis_z_jk,
            "radial_start": 1,
            "hw_r": seq.basis_2.pr,
            "hw_t": seq.basis_2.pt,
            "hw_z": seq.basis_2.pz,
        },
        {
            "name": "zeta",
            "shape": rank_blocks["zeta"].shape,
            "full_shape": seq.basis_2.shape[2],
            "radial_basis": seq.d_basis_r_jk,
            "theta_basis": seq.d_basis_t_jk,
            "zeta_basis": seq.basis_z_jk,
            "radial_start": 1,
            "hw_r": seq.basis_2.pr,
            "hw_t": seq.basis_2.pt,
            "hw_z": seq.basis_2.pz,
        },
    )
    return _build_aligned_vector_mass_models(
        (tensors["beta_rr"], tensors["beta_thetatheta"], tensors["beta_zetazeta"]),
        block_specs,
        rank=rank,
        config=config,
    )


def _build_torus_scaled_m2_models(rank_blocks: dict[str, object], config: Config, rank: int) -> dict[str, object]:
    tensors = _k2_diagonal_metric_tensors(seq)
    block_specs = (
        {
            "name": "r_bulk",
            "shape": rank_blocks["r_bulk"].shape,
            "full_shape": seq.basis_2.shape[0],
            "radial_basis": seq.basis_r_jk,
            "theta_basis": seq.d_basis_t_jk,
            "zeta_basis": seq.d_basis_z_jk,
            "radial_start": 2,
            "hw_r": seq.basis_2.pr,
            "hw_t": seq.basis_2.pt,
            "hw_z": seq.basis_2.pz,
        },
        {
            "name": "theta",
            "shape": rank_blocks["theta"].shape,
            "full_shape": seq.basis_2.shape[1],
            "radial_basis": seq.d_basis_r_jk,
            "theta_basis": seq.basis_t_jk,
            "zeta_basis": seq.d_basis_z_jk,
            "radial_start": 1,
            "hw_r": seq.basis_2.pr,
            "hw_t": seq.basis_2.pt,
            "hw_z": seq.basis_2.pz,
        },
        {
            "name": "zeta",
            "shape": rank_blocks["zeta"].shape,
            "full_shape": seq.basis_2.shape[2],
            "radial_basis": seq.d_basis_r_jk,
            "theta_basis": seq.d_basis_t_jk,
            "zeta_basis": seq.basis_z_jk,
            "radial_start": 1,
            "hw_r": seq.basis_2.pr,
            "hw_t": seq.basis_2.pt,
            "hw_z": seq.basis_2.pz,
        },
    )
    return _build_aligned_vector_mass_models(
        (tensors["beta_rr"], tensors["beta_thetatheta"], tensors["beta_zetazeta"]),
        block_specs,
        rank=rank,
        config=config,
        radial_baselines=_torus_radial_baselines("m2"),
    )


def _build_independent_torus_scaled_m2_models(
    rank_blocks: dict[str, object], config: Config, rank: int
) -> dict[str, object]:
    tensors = _k2_diagonal_metric_tensors(seq)
    block_specs = (
        {
            "name": "r_bulk",
            "shape": rank_blocks["r_bulk"].shape,
            "full_shape": seq.basis_2.shape[0],
            "radial_basis": seq.basis_r_jk,
            "theta_basis": seq.d_basis_t_jk,
            "zeta_basis": seq.d_basis_z_jk,
            "radial_start": 2,
            "hw_r": seq.basis_2.pr,
            "hw_t": seq.basis_2.pt,
            "hw_z": seq.basis_2.pz,
        },
        {
            "name": "theta",
            "shape": rank_blocks["theta"].shape,
            "full_shape": seq.basis_2.shape[1],
            "radial_basis": seq.d_basis_r_jk,
            "theta_basis": seq.basis_t_jk,
            "zeta_basis": seq.d_basis_z_jk,
            "radial_start": 1,
            "hw_r": seq.basis_2.pr,
            "hw_t": seq.basis_2.pt,
            "hw_z": seq.basis_2.pz,
        },
        {
            "name": "zeta",
            "shape": rank_blocks["zeta"].shape,
            "full_shape": seq.basis_2.shape[2],
            "radial_basis": seq.d_basis_r_jk,
            "theta_basis": seq.d_basis_t_jk,
            "zeta_basis": seq.basis_z_jk,
            "radial_start": 1,
            "hw_r": seq.basis_2.pr,
            "hw_t": seq.basis_2.pt,
            "hw_z": seq.basis_2.pz,
        },
    )
    return _build_independent_vector_mass_models(
        (tensors["beta_rr"], tensors["beta_thetatheta"], tensors["beta_zetazeta"]),
        block_specs,
        rank=rank,
        config=config,
        radial_baselines=_torus_radial_baselines("m2"),
    )


def _build_exact_torus_m2_models(rank_blocks: dict[str, object], config: Config, rank: int) -> dict[str, object]:
    tensors = _k2_diagonal_metric_tensors(seq)
    block_specs = (
        {
            "name": "r_bulk",
            "shape": rank_blocks["r_bulk"].shape,
            "full_shape": seq.basis_2.shape[0],
            "radial_basis": seq.basis_r_jk,
            "theta_basis": seq.d_basis_t_jk,
            "zeta_basis": seq.d_basis_z_jk,
            "radial_start": 2,
            "hw_r": seq.basis_2.pr,
            "hw_t": seq.basis_2.pt,
            "hw_z": seq.basis_2.pz,
            "assemble_full_field": True,
        },
        {
            "name": "theta",
            "shape": rank_blocks["theta"].shape,
            "full_shape": seq.basis_2.shape[1],
            "radial_basis": seq.d_basis_r_jk,
            "theta_basis": seq.basis_t_jk,
            "zeta_basis": seq.d_basis_z_jk,
            "radial_start": 1,
            "hw_r": seq.basis_2.pr,
            "hw_t": seq.basis_2.pt,
            "hw_z": seq.basis_2.pz,
            "assemble_full_field": True,
        },
        {
            "name": "zeta",
            "shape": rank_blocks["zeta"].shape,
            "full_shape": seq.basis_2.shape[2],
            "radial_basis": seq.d_basis_r_jk,
            "theta_basis": seq.d_basis_t_jk,
            "zeta_basis": seq.basis_z_jk,
            "radial_start": 1,
            "hw_r": seq.basis_2.pr,
            "hw_t": seq.basis_2.pt,
            "hw_z": seq.basis_2.pz,
            "assemble_full_field": True,
        },
    )
    return _build_aligned_vector_mass_models(
        (tensors["beta_rr"], tensors["beta_thetatheta"], tensors["beta_zetazeta"]),
        block_specs,
        rank=rank,
        config=config,
        field_baselines=_torus_exact_field_baselines("m2", config),
    )


def _rank1_als_anchor_from_terms(
    term_r: tuple[jnp.ndarray, ...],
    term_t: tuple[jnp.ndarray, ...],
    term_z: tuple[jnp.ndarray, ...],
    *,
    maxiter: int,
    tol: float,
) -> jnp.ndarray:
    if not term_r:
        raise ValueError("Expected at least one Kronecker term")

    factor_r = _normalize_factor(sum(
        float(jnp.linalg.norm(mass_t)) * float(jnp.linalg.norm(mass_z)) * mass_r
        for mass_r, mass_t, mass_z in zip(term_r, term_t, term_z)
    ))
    factor_t = _normalize_factor(sum(
        float(jnp.linalg.norm(mass_r)) * float(jnp.linalg.norm(mass_z)) * mass_t
        for mass_r, mass_t, mass_z in zip(term_r, term_t, term_z)
    ))
    factor_z = _normalize_factor(sum(
        float(jnp.linalg.norm(mass_r)) * float(jnp.linalg.norm(mass_t)) * mass_z
        for mass_r, mass_t, mass_z in zip(term_r, term_t, term_z)
    ))

    for _ in range(maxiter):
        next_r = _normalize_factor(sum(
            float(jnp.vdot(mass_t, factor_t)) * float(jnp.vdot(mass_z, factor_z)) * mass_r
            for mass_r, mass_t, mass_z in zip(term_r, term_t, term_z)
        ))
        next_t = _normalize_factor(sum(
            float(jnp.vdot(mass_r, next_r)) * float(jnp.vdot(mass_z, factor_z)) * mass_t
            for mass_r, mass_t, mass_z in zip(term_r, term_t, term_z)
        ))
        next_z = _normalize_factor(sum(
            float(jnp.vdot(mass_r, next_r)) * float(jnp.vdot(mass_t, next_t)) * mass_z
            for mass_r, mass_t, mass_z in zip(term_r, term_t, term_z)
        ))
        delta = max(
            float(jnp.linalg.norm(next_r - factor_r)),
            float(jnp.linalg.norm(next_t - factor_t)),
            float(jnp.linalg.norm(next_z - factor_z)),
        )
        factor_r, factor_t, factor_z = next_r, next_t, next_z
        if delta < tol:
            break

    alpha = sum(
        float(jnp.vdot(mass_r, factor_r))
        * float(jnp.vdot(mass_t, factor_t))
        * float(jnp.vdot(mass_z, factor_z))
        for mass_r, mass_t, mass_z in zip(term_r, term_t, term_z)
    )
    if alpha == 0.0:
        return jnp.zeros(
            (term_r[0].shape[0] * term_t[0].shape[0] * term_z[0].shape[0],) * 2,
            dtype=term_r[0].dtype,
        )

    scale = abs(alpha) ** (1.0 / 3.0)
    sign = -1.0 if alpha < 0.0 else 1.0
    factor_r = scale * factor_r
    factor_t = scale * factor_t
    factor_z = sign * scale * factor_z
    return _symmetrize(jnp.kron(jnp.kron(factor_r, factor_t), factor_z))


def _reshuffle_2factor(matrix: jnp.ndarray, left_dim: int, right_dim: int) -> jnp.ndarray:
    return matrix.reshape(left_dim, right_dim, left_dim, right_dim).transpose(0, 2, 1, 3).reshape(
        left_dim * left_dim,
        right_dim * right_dim,
    )


def _vlp_rank1_2factor(matrix: jnp.ndarray, left_dim: int, right_dim: int):
    reshuffled = _reshuffle_2factor(matrix, left_dim, right_dim)
    u, s, vh = jnp.linalg.svd(reshuffled, full_matrices=False)
    sigma = s[0]
    scale = jnp.sqrt(jnp.maximum(sigma, 0.0))
    left = (scale * u[:, 0]).reshape(left_dim, left_dim)
    right = (scale * vh[0, :]).reshape(right_dim, right_dim)
    return left, right


def _permute_three_factor_matrix(
    matrix: jnp.ndarray,
    dims: tuple[int, int, int],
    order: tuple[str, str, str],
) -> tuple[jnp.ndarray, tuple[int, int, int]]:
    axis_index = {"r": 0, "t": 1, "z": 2}
    perm = tuple(axis_index[label] for label in order)
    tensor = matrix.reshape(*dims, *dims)
    permuted = jnp.transpose(tensor, axes=perm + tuple(index + 3 for index in perm))
    permuted_dims = tuple(dims[index] for index in perm)
    return permuted.reshape(np.prod(permuted_dims), np.prod(permuted_dims)), permuted_dims


def _hierarchical_vlp_rank1_3factor(
    matrix: jnp.ndarray,
    dims: tuple[int, int, int],
    order: tuple[str, str, str],
) -> jnp.ndarray:
    permuted_matrix, permuted_dims = _permute_three_factor_matrix(matrix, dims, order)
    left, right = _vlp_rank1_2factor(permuted_matrix, permuted_dims[0], permuted_dims[1] * permuted_dims[2])
    mid, last = _vlp_rank1_2factor(right, permuted_dims[1], permuted_dims[2])
    factors = {
        order[0]: left,
        order[1]: mid,
        order[2]: last,
    }
    anchor = jnp.kron(jnp.kron(factors["r"], factors["t"]), factors["z"])
    return 0.5 * (anchor + anchor.T)


def _vlp_rank1_anchors(matrix: jnp.ndarray, dims: tuple[int, int, int]) -> list[tuple[str, jnp.ndarray]]:
    splits = [
        ("r|tz", ("r", "t", "z")),
        ("t|rz", ("t", "r", "z")),
        ("z|rt", ("z", "r", "t")),
    ]
    return [
        (label, _hierarchical_vlp_rank1_3factor(matrix, dims, order))
        for label, order in splits
    ]


def _invert_symmetric_matrix(matrix: jnp.ndarray) -> jnp.ndarray:
    sym_matrix = 0.5 * (matrix + matrix.T)
    eigvals = jnp.linalg.eigvalsh(sym_matrix)
    svals = jnp.linalg.svd(sym_matrix, compute_uv=False)
    smax = float(svals[0]) if svals.size else 0.0
    smin = float(svals[-1]) if svals.size else 0.0
    if smin > 1e-12 * max(smax, 1.0) and float(jnp.min(eigvals)) > 0.0:
        return jnp.linalg.inv(sym_matrix)
    return jnp.linalg.pinv(sym_matrix)


def _sorted_eigvals_spd(matrix: jnp.ndarray) -> np.ndarray:
    values = np.asarray(jnp.linalg.eigvalsh(0.5 * (matrix + matrix.T)))
    return np.sort(values)


def _sorted_preconditioned_eigvals(precond_inv: jnp.ndarray, matrix: jnp.ndarray) -> tuple[np.ndarray, float]:
    eigvals = np.asarray(jnp.linalg.eigvals(precond_inv @ matrix))
    max_imag = float(np.max(np.abs(np.imag(eigvals)))) if eigvals.size else 0.0
    values = np.sort(np.real(eigvals))
    return values, max_imag


def _restrict_diag_inverse(diag_inverse: jnp.ndarray, indices: jnp.ndarray | slice) -> jnp.ndarray:
    if isinstance(indices, slice):
        values = diag_inverse[indices]
    else:
        values = diag_inverse[jnp.asarray(indices)]
    return jnp.diag(values)


def _plot_raw_spectra(ax, spectra: list[tuple[str, np.ndarray]], title: str):
    for label, values in spectra:
        ax.semilogy(np.arange(1, values.size + 1), values, marker="o", linewidth=1.0, markersize=3.0, label=label)
    ax.set_title(title)
    ax.set_xlabel("Sorted eigenvalue index")
    ax.set_ylabel("Eigenvalue")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)


def _plot_preconditioned_spectra(ax, spectra: list[tuple[str, np.ndarray, float]], title: str):
    for label, values, max_imag in spectra:
        label_text = f"{label} (max|Im|={max_imag:.1e})"
        ax.plot(np.arange(1, values.size + 1), values, marker="o", linewidth=1.0, markersize=3.0, label=label_text)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("Sorted eigenvalue index")
    ax.set_ylabel("Eigenvalue of P A")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)


def _print_spectrum_summary(name: str, values: np.ndarray):
    condition = values[-1] / max(values[0], np.finfo(np.float64).tiny)
    print(f"{name}: min={values[0]:.3e}, max={values[-1]:.3e}, cond={condition:.3e}")


def _print_preconditioned_summary(name: str, spectra: list[tuple[str, np.ndarray, float]]):
    print(name)
    for label, values, max_imag in spectra:
        min_eval = float(values[0])
        max_eval = float(values[-1])
        if min_eval > 0.0:
            condition = max_eval / min_eval
            cond_text = f"{condition:.3e}"
        else:
            cond_text = "undefined"
        print(
            f"  {label}: min={min_eval:.3e}, max={max_eval:.3e}, "
            f"cond={cond_text}, max|Im|={max_imag:.3e}"
        )


def _extreme_preconditioned_eigs(precond_inv: jnp.ndarray, matrix: jnp.ndarray) -> tuple[float, float, float]:
    eigvals = np.asarray(jnp.linalg.eigvals(precond_inv @ matrix))
    max_imag = float(np.max(np.abs(np.imag(eigvals)))) if eigvals.size else 0.0
    values = np.sort(np.real(eigvals))
    return float(values[0]), float(values[-1]), max_imag


def _estimate_smoother_bounds(
    matrix: jnp.ndarray,
    smoother_apply,
    *,
    config: Config,
    seed: int,
) -> tuple[float, float]:
    lambda_min, lambda_max = _estimate_chebyshev_lanczos_bounds_apply(
        lambda x, block_matrix=matrix: block_matrix @ x,
        smoother_apply,
        matrix.shape[0],
        lanczos_iterations=config.lanczos_iterations,
        lanczos_max_eig_inflation=config.lanczos_max_eig_inflation,
        lanczos_min_eig_deflation=config.lanczos_min_eig_deflation,
        lanczos_min_eig_floor_fraction=config.lanczos_min_eig_floor_fraction,
        seed=seed,
    )
    return float(lambda_min), float(lambda_max)


def _build_smoother_bound_summary(
    label: str,
    matrix: jnp.ndarray,
    precond_inv: jnp.ndarray,
    smoother_apply,
    *,
    config: Config,
    seed: int,
) -> dict[str, float | str]:
    true_min, true_max, max_imag = _extreme_preconditioned_eigs(precond_inv, matrix)
    est_min, est_max = _estimate_smoother_bounds(matrix, smoother_apply, config=config, seed=seed)
    return {
        "label": label,
        "true_min": true_min,
        "true_max": true_max,
        "estimated_min": est_min,
        "estimated_max": est_max,
        "max_imag": max_imag,
    }


def _print_smoother_bound_summary(name: str, bounds: list[dict[str, float | str]]):
    print(name)
    for summary in bounds:
        print(
            f"  {summary['label']}: "
            f"est[min,max]=({float(summary['estimated_min']):.3e}, {float(summary['estimated_max']):.3e}), "
            f"true[min,max]=({float(summary['true_min']):.3e}, {float(summary['true_max']):.3e}), "
            f"max|Im|={float(summary['max_imag']):.3e}"
        )


def _build_m0_payload(base_operators, ranked_operators: dict[int, object], production_operators, config: Config):
    matrix = jnp.asarray(dense_mass_matrix(seq, base_operators, 0, dirichlet=config.dirichlet))
    core_size = _core_size(seq)
    bulk_exact = matrix[core_size:, core_size:]
    jacobi_diaginv = get_mass_jacobi_diaginv(base_operators.mass_preconds, 0, config.dirichlet)
    jacobi_bulk = _restrict_diag_inverse(jacobi_diaginv, slice(core_size, None))
    weight_tensor = seq.geometry.jacobian_j.reshape(seq.quad.ny, seq.quad.nx, seq.quad.nz)
    radial_baseline = _mean_one(jnp.maximum(jnp.asarray(seq.quad.x_x, dtype=jnp.float64), 1e-8))

    preconditioned = [("jacobi",) + _sorted_preconditioned_eigvals(jacobi_bulk, bulk_exact)]
    smoother_bounds = [
        _build_smoother_bound_summary(
            "jacobi",
            bulk_exact,
            jacobi_bulk,
            lambda x, block=jacobi_bulk: block @ x,
            config=config,
            seed=0,
        )
    ]
    production_factors = _select_mass_tensor_factors(production_operators.mass_preconds, 0, config.dirichlet).bulk
    production_inv = _matrix_from_apply(
        lambda x, block=bulk_exact, factors=production_factors: _apply_tensor_exact_block(block, factors, x),
        bulk_exact.shape[0],
    )
    preconditioned.append(("production",) + _sorted_preconditioned_eigvals(production_inv, bulk_exact))
    for rank, operators in ranked_operators.items():
        factors = _select_mass_tensor_factors(operators.mass_preconds, 0, config.dirichlet).bulk
        tensor_inv = _matrix_from_apply(lambda x, block=factors: _apply_tensor_diagonal_block(block, x), bulk_exact.shape[0])
        model_matrix = _matrix_from_terms(factors.term_r, factors.term_t, factors.term_z)
        model_inv = jnp.linalg.inv(model_matrix)
        torus_scaled_model = _build_scalar_reference_model(
            weight_tensor,
            rank=rank,
            radial_basis=seq.basis_r_jk,
            theta_basis=seq.basis_t_jk,
            zeta_basis=seq.basis_z_jk,
            radial_start=2,
            shape=factors.shape,
            full_shape=seq.basis_0.shape[0],
            hw_r=seq.basis_0.pr,
            hw_t=seq.basis_0.pt,
            hw_z=seq.basis_0.pz,
            config=config,
            radial_baseline=radial_baseline,
        )
        torus_scaled_inv = _invert_symmetric_matrix(torus_scaled_model)
        preconditioned.append((f"tensor-r{rank}",) + _sorted_preconditioned_eigvals(tensor_inv, bulk_exact))
        preconditioned.append((f"model-inv-r{rank}",) + _sorted_preconditioned_eigvals(model_inv, bulk_exact))
        preconditioned.append(
            (f"radial-ref-model-inv-r{rank}",) + _sorted_preconditioned_eigvals(torus_scaled_inv, bulk_exact)
        )
        smoother_bounds.append(
            _build_smoother_bound_summary(
                f"tensor-r{rank}",
                bulk_exact,
                tensor_inv,
                lambda x, block=factors: _apply_tensor_diagonal_block(block, x),
                config=config,
                seed=100 + rank,
            )
        )

    return {
        "full": _sorted_eigvals_spd(matrix),
        "bulk": _sorted_eigvals_spd(bulk_exact),
        "preconditioned": preconditioned,
        "smoother_bounds": smoother_bounds,
    }


def _build_m1_payload(base_operators, ranked_operators: dict[int, object], production_operators, config: Config):
    matrix = jnp.asarray(dense_mass_matrix(seq, base_operators, 1, dirichlet=config.dirichlet))
    if not ranked_operators:
        raise ValueError("Need at least one tensor rank to extract the M1 bulk block structure")
    reference_operators = next(iter(ranked_operators.values()))
    surgery = _select_mass_surgery_factors(reference_operators.mass_preconds, 1, config.dirichlet)
    jacobi_diaginv = get_mass_jacobi_diaginv(base_operators.mass_preconds, 1, config.dirichlet)
    blocks = {
        "arr": {
            "indices": surgery.r_indices,
            "matrix": matrix[surgery.r_indices][:, surgery.r_indices],
        },
        "theta": {
            "indices": surgery.theta_bulk_indices,
            "matrix": matrix[surgery.theta_bulk_indices][:, surgery.theta_bulk_indices],
        },
        "zeta": {
            "indices": surgery.zeta_bulk_indices,
            "matrix": matrix[surgery.zeta_bulk_indices][:, surgery.zeta_bulk_indices],
        },
    }

    for name, payload in blocks.items():
        exact_block = payload["matrix"]
        jacobi_block = _restrict_diag_inverse(jacobi_diaginv, payload["indices"])
        payload["raw_spectrum"] = _sorted_eigvals_spd(exact_block)
        payload["preconditioned"] = [("jacobi",) + _sorted_preconditioned_eigvals(jacobi_block, exact_block)]
        payload["smoother_bounds"] = [
            _build_smoother_bound_summary(
                "jacobi",
                exact_block,
                jacobi_block,
                lambda x, block=jacobi_block: block @ x,
                config=config,
                seed=10,
            )
        ]

    production_factors = _select_mass_tensor_factors(production_operators.mass_preconds, 1, config.dirichlet)
    production_blocks = {
        "arr": production_factors.arr,
        "theta": production_factors.theta,
        "zeta": production_factors.zeta,
    }
    for name, block_factors in production_blocks.items():
        exact_block = blocks[name]["matrix"]
        production_inv = _matrix_from_apply(
            lambda x, block=exact_block, factors=block_factors: _apply_tensor_exact_block(block, factors, x),
            exact_block.shape[0],
        )
        blocks[name]["preconditioned"].append(
            ("production",) + _sorted_preconditioned_eigvals(production_inv, exact_block)
        )

    for rank, operators in ranked_operators.items():
        factors = _select_mass_tensor_factors(operators.mass_preconds, 1, config.dirichlet)
        rank_blocks = {
            "arr": factors.arr,
            "theta": factors.theta,
            "zeta": factors.zeta,
        }
        independent_torus_scaled_models = _build_independent_torus_scaled_m1_models(rank_blocks, config, rank)
        for name, block_factors in rank_blocks.items():
            exact_block = blocks[name]["matrix"]
            tensor_inv = _matrix_from_apply(
                lambda x, block=block_factors: _apply_tensor_diagonal_block(block, x),
                exact_block.shape[0],
            )
            model_matrix = _matrix_from_terms(block_factors.term_r, block_factors.term_t, block_factors.term_z)
            model_inv = jnp.linalg.inv(model_matrix)
            independent_torus_scaled_model_inv = _invert_symmetric_matrix(
                independent_torus_scaled_models["blocks"][name]["model_matrix"]
            )
            blocks[name]["preconditioned"].append(
                (f"tensor-r{rank}",) + _sorted_preconditioned_eigvals(tensor_inv, exact_block)
            )
            blocks[name]["preconditioned"].append(
                (f"model-inv-r{rank}",) + _sorted_preconditioned_eigvals(model_inv, exact_block)
            )
            blocks[name]["preconditioned"].append(
                (f"radial-ref-model-inv-r{rank}",)
                + _sorted_preconditioned_eigvals(independent_torus_scaled_model_inv, exact_block)
            )
            blocks[name]["smoother_bounds"].append(
                _build_smoother_bound_summary(
                    f"tensor-r{rank}",
                    exact_block,
                    tensor_inv,
                    lambda x, block=block_factors: _apply_tensor_diagonal_block(block, x),
                    config=config,
                    seed=110 + rank,
                )
            )

    return {
        "full": _sorted_eigvals_spd(matrix),
        "blocks": blocks,
    }


def _build_m2_payload(base_operators, ranked_operators: dict[int, object], production_operators, config: Config):
    matrix = jnp.asarray(dense_mass_matrix(seq, base_operators, 2, dirichlet=config.dirichlet))
    if not ranked_operators:
        raise ValueError("Need at least one tensor rank to extract the M2 bulk block structure")
    reference_operators = next(iter(ranked_operators.values()))
    surgery = _select_mass_surgery_factors(reference_operators.mass_preconds, 2, config.dirichlet)
    jacobi_diaginv = get_mass_jacobi_diaginv(base_operators.mass_preconds, 2, config.dirichlet)
    blocks = {
        "r_bulk": {
            "indices": surgery.r_bulk_indices,
            "matrix": matrix[surgery.r_bulk_indices][:, surgery.r_bulk_indices],
        },
        "theta": {
            "indices": surgery.theta_indices,
            "matrix": matrix[surgery.theta_indices][:, surgery.theta_indices],
        },
        "zeta": {
            "indices": surgery.zeta_indices,
            "matrix": matrix[surgery.zeta_indices][:, surgery.zeta_indices],
        },
    }

    for name, payload in blocks.items():
        exact_block = payload["matrix"]
        jacobi_block = _restrict_diag_inverse(jacobi_diaginv, payload["indices"])
        payload["raw_spectrum"] = _sorted_eigvals_spd(exact_block)
        payload["preconditioned"] = [("jacobi",) + _sorted_preconditioned_eigvals(jacobi_block, exact_block)]
        payload["smoother_bounds"] = [
            _build_smoother_bound_summary(
                "jacobi",
                exact_block,
                jacobi_block,
                lambda x, block=jacobi_block: block @ x,
                config=config,
                seed=20,
            )
        ]

    production_factors = _select_mass_tensor_factors(production_operators.mass_preconds, 2, config.dirichlet)
    production_blocks = {
        "r_bulk": production_factors.r_bulk,
        "theta": production_factors.theta,
        "zeta": production_factors.zeta,
    }
    for name, block_factors in production_blocks.items():
        exact_block = blocks[name]["matrix"]
        production_inv = _matrix_from_apply(
            lambda x, block=exact_block, factors=block_factors: _apply_tensor_exact_block(block, factors, x),
            exact_block.shape[0],
        )
        blocks[name]["preconditioned"].append(
            ("production",) + _sorted_preconditioned_eigvals(production_inv, exact_block)
        )

    for rank, operators in ranked_operators.items():
        factors = _select_mass_tensor_factors(operators.mass_preconds, 2, config.dirichlet)
        rank_blocks = {
            "r_bulk": factors.r_bulk,
            "theta": factors.theta,
            "zeta": factors.zeta,
        }
        independent_torus_scaled_models = _build_independent_torus_scaled_m2_models(rank_blocks, config, rank)
        for name, block_factors in rank_blocks.items():
            exact_block = blocks[name]["matrix"]
            tensor_inv = _matrix_from_apply(
                lambda x, block=block_factors: _apply_tensor_diagonal_block(block, x),
                exact_block.shape[0],
            )
            model_matrix = _matrix_from_terms(block_factors.term_r, block_factors.term_t, block_factors.term_z)
            model_inv = jnp.linalg.inv(model_matrix)
            independent_torus_scaled_model_inv = _invert_symmetric_matrix(
                independent_torus_scaled_models["blocks"][name]["model_matrix"]
            )
            blocks[name]["preconditioned"].append(
                (f"tensor-r{rank}",) + _sorted_preconditioned_eigvals(tensor_inv, exact_block)
            )
            blocks[name]["preconditioned"].append(
                (f"model-inv-r{rank}",) + _sorted_preconditioned_eigvals(model_inv, exact_block)
            )
            blocks[name]["preconditioned"].append(
                (f"radial-ref-model-inv-r{rank}",)
                + _sorted_preconditioned_eigvals(independent_torus_scaled_model_inv, exact_block)
            )
            blocks[name]["smoother_bounds"].append(
                _build_smoother_bound_summary(
                    f"tensor-r{rank}",
                    exact_block,
                    tensor_inv,
                    lambda x, block=block_factors: _apply_tensor_diagonal_block(block, x),
                    config=config,
                    seed=120 + rank,
                )
            )

    return {
        "full": _sorted_eigvals_spd(matrix),
        "blocks": blocks,
    }


def _build_m3_payload(base_operators, ranked_operators: dict[int, object], production_operators, config: Config):
    matrix = jnp.asarray(dense_mass_matrix(seq, base_operators, 3, dirichlet=config.dirichlet))
    jacobi_diaginv = get_mass_jacobi_diaginv(base_operators.mass_preconds, 3, config.dirichlet)
    jacobi_full = jnp.diag(jacobi_diaginv)
    weight_tensor = (1.0 / seq.geometry.jacobian_j).reshape(seq.quad.ny, seq.quad.nx, seq.quad.nz)
    radial_baseline = _mean_one(1.0 / jnp.maximum(jnp.asarray(seq.quad.x_x, dtype=jnp.float64), 1e-8))
    preconditioned = [("jacobi",) + _sorted_preconditioned_eigvals(jacobi_full, matrix)]
    smoother_bounds = [
        _build_smoother_bound_summary(
            "jacobi",
            matrix,
            jacobi_full,
            lambda x, block=jacobi_full: block @ x,
            config=config,
            seed=30,
        )
    ]
    production_factors = _select_mass_tensor_factors(production_operators.mass_preconds, 3, config.dirichlet)
    production_inv = _matrix_from_apply(
        lambda x, block=matrix, factors=production_factors: _apply_tensor_exact_block(block, factors, x),
        matrix.shape[0],
    )
    preconditioned.append(("production",) + _sorted_preconditioned_eigvals(production_inv, matrix))

    for rank, operators in ranked_operators.items():
        factors = _select_mass_tensor_factors(operators.mass_preconds, 3, config.dirichlet)
        tensor_inv = _matrix_from_apply(
            lambda x, block=factors: _apply_tensor_diagonal_block(block, x),
            matrix.shape[0],
        )
        model_matrix = _matrix_from_terms(factors.term_r, factors.term_t, factors.term_z)
        model_inv = jnp.linalg.inv(model_matrix)
        torus_scaled_model = _build_scalar_reference_model(
            weight_tensor,
            rank=rank,
            radial_basis=seq.d_basis_r_jk,
            theta_basis=seq.d_basis_t_jk,
            zeta_basis=seq.d_basis_z_jk,
            radial_start=1,
            shape=factors.shape,
            full_shape=seq.basis_3.shape[0],
            hw_r=seq.basis_3.pr,
            hw_t=seq.basis_3.pt,
            hw_z=seq.basis_3.pz,
            config=config,
            radial_baseline=radial_baseline,
        )
        torus_scaled_inv = _invert_symmetric_matrix(torus_scaled_model)
        preconditioned.append((f"tensor-r{rank}",) + _sorted_preconditioned_eigvals(tensor_inv, matrix))
        preconditioned.append((f"model-inv-r{rank}",) + _sorted_preconditioned_eigvals(model_inv, matrix))
        preconditioned.append(
            (f"radial-ref-model-inv-r{rank}",) + _sorted_preconditioned_eigvals(torus_scaled_inv, matrix)
        )
        smoother_bounds.append(
            _build_smoother_bound_summary(
                f"tensor-r{rank}",
                matrix,
                tensor_inv,
                lambda x, block=factors: _apply_tensor_diagonal_block(block, x),
                config=config,
                seed=130 + rank,
            )
        )

    return {
        "full": _sorted_eigvals_spd(matrix),
        "preconditioned": preconditioned,
        "smoother_bounds": smoother_bounds,
    }


def _save_or_show(figures: list[tuple[str, plt.Figure]], config: Config):
    saved_paths = []
    if config.save_prefix:
        prefix = Path(config.save_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        for suffix, figure in figures:
            output_path = prefix.with_name(f"{prefix.name}_{suffix}.png")
            figure.savefig(output_path, dpi=180, bbox_inches="tight")
            saved_paths.append(output_path)
        print("Saved figures:")
        for output_path in saved_paths:
            print(f"  {output_path}")
    if config.show:
        plt.show()
    else:
        plt.close("all")


config = _parse_args()
seq = _build_seq(config)
base_operators = _assemble_base_operators(seq)
ranked_operators = {
    rank: _assemble_ranked_operators(seq, base_operators, config, rank)
    for rank in config.ranks
}
production_operators = _assemble_production_operators(seq, base_operators)

m0 = _build_m0_payload(base_operators, ranked_operators, production_operators, config)
m1 = _build_m1_payload(base_operators, ranked_operators, production_operators, config)
m2 = _build_m2_payload(base_operators, ranked_operators, production_operators, config)
m3 = _build_m3_payload(base_operators, ranked_operators, production_operators, config)

_print_spectrum_summary("M0 full", m0["full"])
_print_spectrum_summary("M0 bulk", m0["bulk"])
_print_spectrum_summary("M1 full", m1["full"])
for name, payload in m1["blocks"].items():
    _print_spectrum_summary(f"M1 {name}", payload["raw_spectrum"])
_print_spectrum_summary("M2 full", m2["full"])
for name, payload in m2["blocks"].items():
    _print_spectrum_summary(f"M2 {name}", payload["raw_spectrum"])
_print_spectrum_summary("M3 full", m3["full"])

_print_preconditioned_summary("M0 bulk preconditioned", m0["preconditioned"])
for name, payload in m1["blocks"].items():
    _print_preconditioned_summary(f"M1 {name} preconditioned", payload["preconditioned"])
for name, payload in m2["blocks"].items():
    _print_preconditioned_summary(f"M2 {name} preconditioned", payload["preconditioned"])
_print_preconditioned_summary("M3 full preconditioned", m3["preconditioned"])

_print_smoother_bound_summary("M0 bulk smoother bounds", m0["smoother_bounds"])
for name, payload in m1["blocks"].items():
    _print_smoother_bound_summary(f"M1 {name} smoother bounds", payload["smoother_bounds"])
for name, payload in m2["blocks"].items():
    _print_smoother_bound_summary(f"M2 {name} smoother bounds", payload["smoother_bounds"])
_print_smoother_bound_summary("M3 full smoother bounds", m3["smoother_bounds"])

fig1, axes1 = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
_plot_raw_spectra(axes1[0, 0], [("full M0", m0["full"])], "M0 full spectrum")
_plot_raw_spectra(axes1[0, 1], [("full M1", m1["full"])], "M1 full spectrum")
_plot_raw_spectra(axes1[1, 0], [("full M2", m2["full"])], "M2 full spectrum")
_plot_raw_spectra(axes1[1, 1], [("full M3", m3["full"])], "M3 full spectrum")

fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
for ax, name in zip(axes2, ("arr", "theta", "zeta"), strict=True):
    _plot_raw_spectra(ax, [(f"M1 {name}", m1["blocks"][name]["raw_spectrum"])], f"M1 {name} raw spectrum")

fig3, axes3 = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
for ax, name in zip(axes3, ("r_bulk", "theta", "zeta"), strict=True):
    _plot_raw_spectra(ax, [(f"M2 {name}", m2["blocks"][name]["raw_spectrum"])], f"M2 {name} raw spectrum")

fig4, axes4 = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
_plot_preconditioned_spectra(axes4[0, 0], m0["preconditioned"], "M0 bulk preconditioned spectra")
_plot_preconditioned_spectra(axes4[0, 1], m1["blocks"]["arr"]["preconditioned"], "M1 arr preconditioned spectra")
_plot_preconditioned_spectra(axes4[1, 0], m1["blocks"]["theta"]["preconditioned"], "M1 theta preconditioned spectra")
_plot_preconditioned_spectra(axes4[1, 1], m1["blocks"]["zeta"]["preconditioned"], "M1 zeta preconditioned spectra")

fig5, axes5 = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
_plot_preconditioned_spectra(axes5[0, 0], m2["blocks"]["r_bulk"]["preconditioned"], "M2 r_bulk preconditioned spectra")
_plot_preconditioned_spectra(axes5[0, 1], m2["blocks"]["theta"]["preconditioned"], "M2 theta preconditioned spectra")
_plot_preconditioned_spectra(axes5[1, 0], m2["blocks"]["zeta"]["preconditioned"], "M2 zeta preconditioned spectra")
_plot_preconditioned_spectra(axes5[1, 1], m3["preconditioned"], "M3 full preconditioned spectra")

figures = [
    ("raw_full", fig1),
    ("raw_blocks_m1", fig2),
    ("raw_blocks_m2", fig3),
    ("preconditioned_m0_m1", fig4),
    ("preconditioned_m2_m3", fig5),
]
_save_or_show(figures, config)