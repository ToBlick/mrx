from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import numpy as np

from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import rotating_ellipse_map, toroid_map
from mrx.operators import (
    assemble_mass_operators,
    assemble_tensor_mass_preconditioner,
    dense_mass_matrix,
)
from mrx.preconditioners import _core_size, _select_mass_surgery_factors, _select_mass_tensor_factors

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class Config:
    ns: tuple[int, int, int] = (6, 8, 6)
    p: int = 3
    tol: float = 1e-9
    maxiter: int = 1000
    betti: tuple[int, int, int, int] = (1, 1, 0, 0)
    dirichlet: bool = True
    map_kind: str = "rotating_ellipse"
    torus_epsilon: float = 1.0 / 3.0
    torus_r0: float = 1.0
    rotating_eps: float = 0.33
    rotating_kappa: float = 1.2
    rotating_r0: float = 1.0
    rotating_nfp: int = 3
    tensor_rank: int = 4
    t0_terms: int = 3
    cp_maxiter: int = 150
    cp_tol: float = 1e-9
    cp_ridge: float = 1e-12
    richardson_steps: int = 0
    richardson_omega: float = 1.0
    max_svals: int = 8
    json: bool = False


def _parse_bool(value: str) -> bool:
    text = value.strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value from {value!r}")


def _parse_int_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def _parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description=(
            "Dense VLP/Neumann screening for the scalar M0 bulk block and the "
            "tensor-inverted M1 blocks used by the production mass preconditioner."
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
    parser.add_argument("--tensor-rank", type=int, default=Config.tensor_rank)
    parser.add_argument("--t0-terms", type=int, default=Config.t0_terms)
    parser.add_argument("--cp-maxiter", type=int, default=Config.cp_maxiter)
    parser.add_argument("--cp-tol", type=float, default=Config.cp_tol)
    parser.add_argument("--cp-ridge", type=float, default=Config.cp_ridge)
    parser.add_argument("--richardson-steps", type=int, default=Config.richardson_steps)
    parser.add_argument("--richardson-omega", type=float, default=Config.richardson_omega)
    parser.add_argument("--max-svals", type=int, default=Config.max_svals)
    parser.add_argument("--json", action="store_true")
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
        tensor_rank=args.tensor_rank,
        t0_terms=args.t0_terms,
        cp_maxiter=args.cp_maxiter,
        cp_tol=args.cp_tol,
        cp_ridge=args.cp_ridge,
        richardson_steps=args.richardson_steps,
        richardson_omega=args.richardson_omega,
        max_svals=args.max_svals,
        json=args.json,
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


def _build_case(config: Config):
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

    operators = assemble_mass_operators(seq, seq.geometry, ks=(0, 1))
    operators = assemble_tensor_mass_preconditioner(
        seq,
        operators=operators,
        ks=(0, 1),
        rank=config.tensor_rank,
        cp_kwargs={
            "maxiter": config.cp_maxiter,
            "tol": config.cp_tol,
            "ridge": config.cp_ridge,
            "richardson_steps": config.richardson_steps,
            "richardson_omega": config.richardson_omega,
            "block_chebyshev_steps": 0,
        },
    )
    operators = seq.set_operators(operators, sync_legacy=False)
    return seq, operators


def _kron3(a: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
    return jnp.kron(jnp.kron(a, b), c)


def _symmetrize(matrix: jnp.ndarray) -> jnp.ndarray:
    return 0.5 * (matrix + matrix.T)


def _matrix_from_terms(
    term_r: tuple[jnp.ndarray, ...],
    term_t: tuple[jnp.ndarray, ...],
    term_z: tuple[jnp.ndarray, ...],
    *,
    n_terms: int | None = None,
) -> jnp.ndarray:
    if not term_r:
        raise ValueError("Expected at least one Kronecker term")
    limit = len(term_r) if n_terms is None else min(n_terms, len(term_r))
    matrix = jnp.zeros(
        (term_r[0].shape[0] * term_t[0].shape[0] * term_z[0].shape[0],) * 2,
        dtype=term_r[0].dtype,
    )
    for mass_r, mass_t, mass_z in zip(term_r[:limit], term_t[:limit], term_z[:limit]):
        matrix = matrix + _kron3(mass_r, mass_t, mass_z)
    return matrix


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
    return left, right, s


def _hierarchical_vlp_rank1_3factor(matrix: jnp.ndarray, dims: tuple[int, int, int]):
    nr, nt, nz = dims
    left, right, stage1_svals = _vlp_rank1_2factor(matrix, nr, nt * nz)
    mid, last, stage2_svals = _vlp_rank1_2factor(right, nt, nz)
    anchor = _kron3(left, mid, last)
    return left, mid, last, anchor, stage1_svals, stage2_svals


def _spectral_norm(matrix: jnp.ndarray) -> float:
    return float(jnp.linalg.svd(matrix, compute_uv=False)[0])


def _relative_fro_error(approx: jnp.ndarray, exact: jnp.ndarray) -> float:
    return float(jnp.linalg.norm(approx - exact) / jnp.maximum(jnp.linalg.norm(exact), 1.0))


def _eta(anchor: jnp.ndarray, residual: jnp.ndarray) -> dict[str, float | str]:
    svals = jnp.linalg.svd(anchor, compute_uv=False)
    smax = float(svals[0])
    smin = float(svals[-1])
    tiny = np.finfo(np.float64).tiny
    cond = smax / max(smin, tiny)
    if smin > 1e-12 * max(smax, 1.0):
        scaled = jnp.linalg.solve(anchor, residual)
        method = "solve"
    else:
        scaled = jnp.linalg.pinv(anchor) @ residual
        method = "pinv"
    return {
        "eta": _spectral_norm(scaled),
        "anchor_sigma_min": smin,
        "anchor_sigma_max": smax,
        "anchor_condition_number": cond,
        "inverse_method": method,
    }


def _direct_inverse(matrix: jnp.ndarray) -> tuple[jnp.ndarray, str, float, float]:
    sym_matrix = _symmetrize(matrix)
    eigvals = jnp.linalg.eigvalsh(sym_matrix)
    svals = jnp.linalg.svd(sym_matrix, compute_uv=False)
    smax = float(svals[0])
    smin = float(svals[-1])
    tiny = np.finfo(np.float64).tiny
    if smin > 1e-12 * max(smax, 1.0) and float(jnp.min(eigvals)) > 0.0:
        return jnp.linalg.inv(sym_matrix), "inv", float(jnp.min(eigvals)), float(jnp.max(eigvals))
    return jnp.linalg.pinv(sym_matrix), "pinv", float(jnp.min(eigvals)), float(jnp.max(eigvals))


def _top_svals(values: jnp.ndarray, max_count: int) -> list[float]:
    return [float(v) for v in np.asarray(values[:max_count])]


def _dominance_ratio(values: jnp.ndarray) -> float | None:
    if values.shape[0] < 2:
        return None
    sigma2 = float(values[1])
    if sigma2 == 0.0:
        return float("inf")
    return float(values[0] / sigma2)


def _block_summary(
    *,
    name: str,
    exact_matrix: jnp.ndarray,
    modeled_matrix: jnp.ndarray,
    base_matrix: jnp.ndarray,
    dims: tuple[int, int, int],
    max_svals: int,
) -> dict[str, object]:
    left, mid, last, anchor, stage1_svals, stage2_svals = _hierarchical_vlp_rank1_3factor(base_matrix, dims)
    exact_eta = _eta(anchor, exact_matrix - anchor)
    modeled_eta = _eta(anchor, modeled_matrix - anchor)
    compression_eta = _eta(anchor, base_matrix - anchor)
    return {
        "name": name,
        "shape": list(exact_matrix.shape),
        "tensor_shape": list(dims),
        "relative_model_error": _relative_fro_error(modeled_matrix, exact_matrix),
        "relative_anchor_error_exact": _relative_fro_error(anchor, exact_matrix),
        "relative_anchor_error_modeled": _relative_fro_error(anchor, modeled_matrix),
        "stage1_top_singular_values": _top_svals(stage1_svals, max_svals),
        "stage2_top_singular_values": _top_svals(stage2_svals, max_svals),
        "stage1_sigma1_over_sigma2": _dominance_ratio(stage1_svals),
        "stage2_sigma1_over_sigma2": _dominance_ratio(stage2_svals),
        "compression_eta": compression_eta,
        "modeled_eta": modeled_eta,
        "exact_eta": exact_eta,
        "anchor_factor_min_eigs": [
            float(jnp.min(jnp.linalg.eigvalsh(0.5 * (left + left.T)))),
            float(jnp.min(jnp.linalg.eigvalsh(0.5 * (mid + mid.T)))),
            float(jnp.min(jnp.linalg.eigvalsh(0.5 * (last + last.T)))),
        ],
    }


def _rt_z_vlp_summary(
    *,
    name: str,
    exact_matrix: jnp.ndarray,
    modeled_matrix: jnp.ndarray,
    base_matrix: jnp.ndarray,
    dims: tuple[int, int, int],
    max_svals: int,
) -> dict[str, object]:
    nr, nt, nz = dims
    rt_factor_raw, z_factor_raw, svals = _vlp_rank1_2factor(base_matrix, nr * nt, nz)
    rt_factor = _symmetrize(rt_factor_raw)
    z_factor = _symmetrize(z_factor_raw)
    anchor = jnp.kron(rt_factor, z_factor)
    exact_eta = _eta(anchor, exact_matrix - anchor)
    modeled_eta = _eta(anchor, modeled_matrix - anchor)
    compression_eta = _eta(anchor, base_matrix - anchor)
    rt_inv, rt_inv_method, rt_min_eig, rt_max_eig = _direct_inverse(rt_factor)
    z_inv, z_inv_method, z_min_eig, z_max_eig = _direct_inverse(z_factor)
    direct_inverse = jnp.kron(rt_inv, z_inv)
    exact_inverse = jnp.linalg.inv(exact_matrix)
    modeled_inverse = jnp.linalg.inv(modeled_matrix)
    return {
        "name": name,
        "shape": list(exact_matrix.shape),
        "tensor_shape": list(dims),
        "grouping": "rt|z",
        "relative_model_error": _relative_fro_error(modeled_matrix, exact_matrix),
        "relative_anchor_error_exact": _relative_fro_error(anchor, exact_matrix),
        "relative_anchor_error_modeled": _relative_fro_error(anchor, modeled_matrix),
        "top_singular_values": _top_svals(svals, max_svals),
        "sigma1_over_sigma2": _dominance_ratio(svals),
        "compression_eta": compression_eta,
        "modeled_eta": modeled_eta,
        "exact_eta": exact_eta,
        "direct_inverse_error_exact": _relative_fro_error(direct_inverse, exact_inverse),
        "direct_inverse_error_modeled": _relative_fro_error(direct_inverse, modeled_inverse),
        "direct_inverse_action_error_exact": _relative_fro_error(exact_matrix @ direct_inverse, jnp.eye(exact_matrix.shape[0], dtype=exact_matrix.dtype)),
        "direct_inverse_action_error_modeled": _relative_fro_error(modeled_matrix @ direct_inverse, jnp.eye(modeled_matrix.shape[0], dtype=modeled_matrix.dtype)),
        "anchor_factor_min_eigs": [rt_min_eig, z_min_eig],
        "anchor_factor_max_eigs": [rt_max_eig, z_max_eig],
        "inverse_methods": [rt_inv_method, z_inv_method],
    }


def _build_m0_summary(seq, operators, config: Config) -> dict[str, object]:
    matrix = jnp.asarray(dense_mass_matrix(seq, operators, 0, dirichlet=config.dirichlet))
    core_size = _core_size(seq)
    bulk_exact = matrix[core_size:, core_size:]
    tensor_factors = _select_mass_tensor_factors(operators.mass_preconds, 0, config.dirichlet)
    modeled_matrix = _matrix_from_terms(
        tensor_factors.bulk.term_r,
        tensor_factors.bulk.term_t,
        tensor_factors.bulk.term_z,
    )
    return {
        "full_shape": list(matrix.shape),
        "bulk_shape": list(bulk_exact.shape),
        "analysis": _block_summary(
            name="M0 bulk",
            exact_matrix=bulk_exact,
            modeled_matrix=modeled_matrix,
            base_matrix=modeled_matrix,
            dims=tensor_factors.bulk.shape,
            max_svals=config.max_svals,
        ),
    }


def _build_m1_summary(seq, operators, config: Config) -> dict[str, object]:
    matrix = jnp.asarray(dense_mass_matrix(seq, operators, 1, dirichlet=config.dirichlet))
    surgery = _select_mass_surgery_factors(operators.mass_preconds, 1, config.dirichlet)
    factors = _select_mass_tensor_factors(operators.mass_preconds, 1, config.dirichlet)
    blocks = [
        (
            "arr",
            matrix[surgery.r_indices][:, surgery.r_indices],
            factors.arr,
        ),
        (
            "theta",
            matrix[surgery.theta_bulk_indices][:, surgery.theta_bulk_indices],
            factors.theta,
        ),
        (
            "zeta",
            matrix[surgery.zeta_bulk_indices][:, surgery.zeta_bulk_indices],
            factors.zeta,
        ),
    ]
    summaries = []
    worst_exact_eta = -np.inf
    worst_name = None
    for name, exact_block, block_factors in blocks:
        modeled_matrix = _matrix_from_terms(block_factors.term_r, block_factors.term_t, block_factors.term_z)
        t0_matrix = _matrix_from_terms(
            block_factors.term_r,
            block_factors.term_t,
            block_factors.term_z,
            n_terms=config.t0_terms,
        )
        summary = _block_summary(
            name=f"M1 {name}",
            exact_matrix=exact_block,
            modeled_matrix=modeled_matrix,
            base_matrix=t0_matrix,
            dims=block_factors.shape,
            max_svals=config.max_svals,
        )
        rt_z_summary = _rt_z_vlp_summary(
            name=f"M1 {name}",
            exact_matrix=exact_block,
            modeled_matrix=modeled_matrix,
            base_matrix=t0_matrix,
            dims=block_factors.shape,
            max_svals=config.max_svals,
        )
        summary["t0_terms"] = min(config.t0_terms, len(block_factors.term_r))
        summary["n_modeled_terms"] = len(block_factors.term_r)
        summary["hierarchical_r_t_z"] = summary.copy()
        summary["rt_z"] = rt_z_summary
        exact_eta = float(rt_z_summary["exact_eta"]["eta"])
        if exact_eta > worst_exact_eta:
            worst_exact_eta = exact_eta
            worst_name = name
        summaries.append(summary)
    return {
        "full_shape": list(matrix.shape),
        "bulk_shape": [int(surgery.bulk_indices.shape[0]), int(surgery.bulk_indices.shape[0])],
        "rt_shape": [int(surgery.bulk_rt_size), int(surgery.bulk_rt_size)],
        "blocks": summaries,
        "max_exact_eta_block": worst_name,
        "max_exact_eta": worst_exact_eta,
    }


def _format_scalar(value) -> str:
    if value is None:
        return "None"
    if isinstance(value, str):
        return value
    if np.isinf(value):
        return "inf"
    return f"{value:.3e}"


def _print_block(label: str, summary: dict[str, object]):
    print(label)
    print(f"  tensor shape: {tuple(summary['tensor_shape'])}")
    print(f"  rel(model, exact): {_format_scalar(summary['relative_model_error'])}")
    print(f"  rel(anchor, modeled): {_format_scalar(summary['relative_anchor_error_modeled'])}")
    print(f"  rel(anchor, exact): {_format_scalar(summary['relative_anchor_error_exact'])}")
    print(
        "  stage-1 svals: "
        + ", ".join(_format_scalar(v) for v in summary["stage1_top_singular_values"])
    )
    print(
        "  stage-2 svals: "
        + ", ".join(_format_scalar(v) for v in summary["stage2_top_singular_values"])
    )
    print(f"  stage-1 sigma1/sigma2: {_format_scalar(summary['stage1_sigma1_over_sigma2'])}")
    print(f"  stage-2 sigma1/sigma2: {_format_scalar(summary['stage2_sigma1_over_sigma2'])}")
    print(f"  compression eta: {_format_scalar(summary['compression_eta']['eta'])}")
    print(f"  modeled eta: {_format_scalar(summary['modeled_eta']['eta'])}")
    print(f"  exact eta: {_format_scalar(summary['exact_eta']['eta'])}")
    print(
        "  anchor cond: "
        f"{_format_scalar(summary['exact_eta']['anchor_condition_number'])} "
        f"({summary['exact_eta']['inverse_method']})"
    )


def _print_rt_z_block(label: str, summary: dict[str, object]):
    print(label)
    print(f"  tensor shape: {tuple(summary['tensor_shape'])}")
    print(f"  grouping: {summary['grouping']}")
    print(f"  rel(model, exact): {_format_scalar(summary['relative_model_error'])}")
    print(f"  rel(anchor, modeled): {_format_scalar(summary['relative_anchor_error_modeled'])}")
    print(f"  rel(anchor, exact): {_format_scalar(summary['relative_anchor_error_exact'])}")
    print(
        "  grouped svals: "
        + ", ".join(_format_scalar(v) for v in summary["top_singular_values"])
    )
    print(f"  sigma1/sigma2: {_format_scalar(summary['sigma1_over_sigma2'])}")
    print(f"  compression eta: {_format_scalar(summary['compression_eta']['eta'])}")
    print(f"  modeled eta: {_format_scalar(summary['modeled_eta']['eta'])}")
    print(f"  exact eta: {_format_scalar(summary['exact_eta']['eta'])}")
    print(
        "  anchor cond: "
        f"{_format_scalar(summary['exact_eta']['anchor_condition_number'])} "
        f"({summary['exact_eta']['inverse_method']})"
    )
    print(
        "  direct inverse rel err (modeled/exact): "
        f"{_format_scalar(summary['direct_inverse_error_modeled'])} / "
        f"{_format_scalar(summary['direct_inverse_error_exact'])}"
    )
    print(
        "  direct inverse action err (modeled/exact): "
        f"{_format_scalar(summary['direct_inverse_action_error_modeled'])} / "
        f"{_format_scalar(summary['direct_inverse_action_error_exact'])}"
    )
    print(
        "  factor inverse methods: "
        f"{summary['inverse_methods'][0]} / {summary['inverse_methods'][1]}"
    )


def main():
    config = _parse_args()
    seq, operators = _build_case(config)
    report = {
        "config": asdict(config),
        "m0": _build_m0_summary(seq, operators, config),
        "m1": _build_m1_summary(seq, operators, config),
    }

    if config.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    print("VLP + Neumann screening")
    print(f"  ns={config.ns}, p={config.p}, dirichlet={config.dirichlet}, map={config.map_kind}")
    print(f"  tensor rank={config.tensor_rank}, T0 terms={config.t0_terms}")
    print("")
    print("M0")
    print(f"  full extracted shape: {tuple(report['m0']['full_shape'])}")
    print(f"  bulk shape: {tuple(report['m0']['bulk_shape'])}")
    _print_block("  scalar bulk", report["m0"]["analysis"])
    print("")
    print("M1")
    print(f"  full extracted shape: {tuple(report['m1']['full_shape'])}")
    print(f"  bulk shape: {tuple(report['m1']['bulk_shape'])}")
    print(f"  rt shape: {tuple(report['m1']['rt_shape'])}")
    for block in report["m1"]["blocks"]:
        print(f"  {block['name']}")
        print(f"    T0 terms used: {block['t0_terms']} / {block['n_modeled_terms']}")
        _print_block("    hierarchical r|t|z", block["hierarchical_r_t_z"])
        _print_rt_z_block("    grouped rt|z", block["rt_z"])
    print("")
    print(
        "M1 worst exact eta: "
        f"{report['m1']['max_exact_eta_block']} -> {_format_scalar(report['m1']['max_exact_eta'])}"
    )


if __name__ == "__main__":
    main()