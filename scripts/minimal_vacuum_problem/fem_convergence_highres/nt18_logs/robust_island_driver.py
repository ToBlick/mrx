#!/usr/bin/env python3
"""Consistent island diagnostics for a saved dense-verified FEM field.

This driver intentionally separates post-processing from the expensive Hodge
solve.  Every grid is traced with the same seed set, turn count, tolerances,
and Fourier grid so that island widths and resonance locations are directly
comparable.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from scripts.minimal_vacuum_problem import compare_push_u_to_simsopt as cmp  # noqa: E402

MVP = ROOT / "scripts" / "minimal_vacuum_problem"
META_JSON = MVP / "hodge_k2_nullspace_meta.json"
COIL_JSON = MVP / "serial0044970.json"


def _log(message: str, path: Path) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with path.open("a") as stream:
        stream.write(line + "\n")


def _parse_ns(text: str) -> tuple[int, int, int]:
    values = tuple(int(value) for value in text.lower().split("x"))
    if len(values) != 3:
        raise ValueError(f"invalid grid {text!r}; expected nrxnthetaxnzeta")
    return values  # type: ignore[return-value]


def _aligned_metrics(
    seq: Any,
    dof: jnp.ndarray,
    map_fn: Any,
    simsopt_field: Any,
) -> dict[str, float]:
    mrx: list[np.ndarray] = []
    ref: list[np.ndarray] = []
    for zeta in (0.0, 0.25, 0.5, 0.75):
        logical, _, _ = cmp._poloidal_slice_points(zeta, 24, 48)
        logical_jax = jnp.asarray(logical, dtype=jnp.float64)
        physical = np.asarray(
            jax.lax.map(map_fn, logical_jax, batch_size=512),
            dtype=np.float64,
        )
        mrx.append(
            np.asarray(
                cmp._evaluate_pushforward(seq, dof, map_fn, logical_jax, k=2),
                dtype=np.float64,
            )
        )
        ref.append(cmp._evaluate_simsopt_field(simsopt_field, physical))
    return cmp._pointwise_vector_metrics(
        np.concatenate(mrx),
        np.concatenate(ref),
        align_pointwise=True,
    )


def _bootstrap_trapped_width(
    lines: list[np.ndarray],
    classifications: list[dict[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    """Cluster-bootstrap trapped lines, preserving within-line correlation."""
    trapped = [
        np.asarray(line, dtype=np.float64)[:, 0]
        for line, item in zip(lines, classifications)
        if bool(item["trapped"])
    ]
    if not trapped:
        return {
            "samples": int(samples),
            "cluster": "trapped_field_line",
            "confidence": 0.90,
            "median": None,
            "q05": None,
            "q95": None,
            "standard_error": None,
        }
    rng = np.random.default_rng(int(seed))
    widths = np.empty(int(samples), dtype=np.float64)
    for index in range(int(samples)):
        chosen = rng.integers(0, len(trapped), size=len(trapped))
        radial = np.concatenate([trapped[item] for item in chosen])
        q05, q95 = np.quantile(radial, [0.05, 0.95])
        widths[index] = q95 - q05
    return {
        "samples": int(samples),
        "cluster": "trapped_field_line",
        "confidence": 0.90,
        "median": float(np.median(widths)),
        "q05": float(np.quantile(widths, 0.05)),
        "q95": float(np.quantile(widths, 0.95)),
        "standard_error": float(np.std(widths, ddof=1)),
    }


def _target_resonance(
    resonances: list[dict[str, Any]],
) -> dict[str, Any]:
    matches = [
        item
        for item in resonances
        if int(item["poloidal_mode"]) == 6
        and abs(int(item["toroidal_mode"])) == 3
    ]
    if not matches:
        raise RuntimeError("(m,n)=(6,3) resonance not found")
    return min(matches, key=lambda item: float(item["sample_mismatch"]))


def _field_based_shear(
    map_fn: Any,
    simsopt_field: Any,
    resonance_rho: float,
    *,
    rho_min: float,
    rho_max: float,
    nfp: int = 3,
) -> tuple[float, list[dict[str, float]]]:
    """Shear from flux-surface-averaged SIMSOPT pitch on a fine ρ grid.

    This is independent of Poincaré seed density and is the common reference
    used for width-law comparisons across grids.
    """
    rho_grid = np.linspace(float(rho_min), float(rho_max), 33)
    profile = cmp._simsopt_iota_profile_from_pitch(
        map_fn,
        simsopt_field,
        rho_grid,
        nfp=int(nfp),
        ntheta=32,
        nzeta=32,
    )
    shear = cmp._local_shear_from_iota_profile(profile, float(resonance_rho))
    return shear, profile


def _save_trace_cache(
    path: Path,
    seeds: np.ndarray,
    lines: list[np.ndarray],
) -> None:
    payload: dict[str, np.ndarray] = {"seeds": np.asarray(seeds)}
    for index, line in enumerate(lines):
        payload[f"line_{index:04d}"] = np.asarray(line)
    np.savez_compressed(path, **payload)


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    ns = _parse_ns(args.grid)
    label = "x".join(str(value) for value in ns)
    dof_path = args.dof.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log.expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    _log(f"load {label} from {dof_path}", log_path)
    hcurl_mod = cmp._load_hcurl_module()
    meta = cmp._load_meta_for_dof(dof_path, k=2, meta_json=META_JSON)
    meta = dict(meta)
    meta["ns"] = list(ns)
    seq, map_fn, _, _, _ = cmp._rebuild_sequence_from_meta(
        meta,
        hcurl_mod=hcurl_mod,
        pushforward_only=True,
    )
    dof = jnp.asarray(np.load(dof_path), dtype=jnp.float64).reshape(-1)
    if int(dof.size) != int(seq.n2_dbc):
        raise ValueError(f"DOF size {dof.size} != n2_dbc {seq.n2_dbc}")

    _log("load SIMSOPT and compute aligned scale", log_path)
    simsopt_field, _ = cmp._load_simsopt_field(COIL_JSON)
    metrics = _aligned_metrics(seq, dof, map_fn, simsopt_field)
    scale = float(metrics["pointwise_u_scale_to_optimal"])

    rho_values = np.linspace(args.rho_min, args.rho_max, args.zoom_nrho)
    theta_values = (
        args.theta0
        + np.arange(args.zoom_phases)
        / (6.0 * max(args.zoom_phases, 1))
    ) % 1.0
    logical_sections: dict[float, list[np.ndarray]] = {}
    audit: dict[str, Any] = {}
    method = str(args.method)
    _log(
        f"trace {args.zoom_nrho}x{args.zoom_phases} seeds, "
        f"{args.turns} turns, method={method}",
        log_path,
    )
    started = time.perf_counter()
    _, seeds, _, transits = cmp._trace_mrx_poincare(
        seq,
        dof,
        map_fn,
        [0.0],
        nlines=int(args.zoom_nrho),
        turns=int(args.turns),
        theta0=float(args.theta0),
        tol=float(args.tol),
        logical_sections_out=logical_sections,
        seed_rho_values=rho_values,
        theta0_values=theta_values,
        method=method,
        audit_out=audit,
    )
    lines = logical_sections[0.0]
    _log(f"trace complete in {time.perf_counter() - started:.1f}s", log_path)

    trapped = cmp._trapped_separatrix_summary(lines, seeds, poloidal_mode=6)
    bootstrap = _bootstrap_trapped_width(
        lines,
        trapped["classifications"],
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )

    # Use the first phase at every radius for a uniformly sampled iota profile.
    seeds_array = np.asarray(seeds, dtype=np.float64)
    iota_lines: list[np.ndarray] = []
    iota_rho: list[float] = []
    for rho in np.unique(seeds_array[:, 0]):
        indices = np.flatnonzero(np.isclose(seeds_array[:, 0], rho))
        iota_lines.append(lines[int(indices[0])])
        iota_rho.append(float(rho))
    iota_profile = cmp._rotational_transform_profile(
        iota_lines,
        np.asarray(iota_rho),
        nfp=3,
    )
    resonances = cmp._identify_iota_resonances(
        iota_profile,
        nfp=3,
        rho_min=float(args.rho_min),
        rho_max=float(args.rho_max),
        max_poloidal_mode=16,
    )
    target = _target_resonance(resonances)
    shear, pitch_iota_profile = _field_based_shear(
        map_fn,
        simsopt_field,
        float(target["rho"]),
        rho_min=float(args.rho_min),
        rho_max=float(args.rho_max),
        nfp=3,
    )
    traced_shear = cmp._local_shear_from_iota_profile(
        iota_profile, float(target["rho"]), half_window=2
    )

    amplitudes = cmp._resonant_normal_error_amplitudes(
        seq,
        dof,
        map_fn,
        simsopt_field,
        resonances,
        mrx_scale=scale,
        ntheta=int(args.fourier_n),
        nzeta=int(args.fourier_n),
    )
    target_amplitude = _target_resonance(amplitudes)
    a63_relative = float(target_amplitude["normal_error_fourier_relative"])
    # Legacy unscaled predictor kept for aggregation compatibility; the
    # physically derived pendulum width uses 2π in the denominator (see
    # cmp.pendulum_island_width). Absolute scale is fit empirically.
    predictor = float(
        np.sqrt(a63_relative / (2.0 * np.pi * max(abs(shear), 1.0e-14)))
    )

    width_profile = cmp._island_width_profile(lines, seeds_array[:, 0])
    trace_cache = output_dir / f"robust_trace_{label}.npz"
    _save_trace_cache(trace_cache, seeds_array, lines)
    zoom_path = cmp._plot_island_zoom(
        lines,
        seeds_array,
        output_dir,
        label=f"robust_{label}",
        rho_min=float(args.rho_min),
        rho_max=float(args.rho_max),
    )

    return {
        "schema_version": 1,
        "ns": list(ns),
        "label": label,
        "p": int(meta["p"]),
        "n2_dbc": int(seq.n2_dbc),
        "dof_npy": str(dof_path),
        "aligned_metrics": metrics,
        "tracer": {
            "rho_min": float(args.rho_min),
            "rho_max": float(args.rho_max),
            "zoom_nrho": int(args.zoom_nrho),
            "zoom_phases": int(args.zoom_phases),
            "turns": int(args.turns),
            "theta0": float(args.theta0),
            "tol": float(args.tol),
            "fourier_n": int(args.fourier_n),
            "completed_transits": [int(value) for value in transits],
            "audit": audit,
        },
        "trapped_separatrix": trapped,
        "trapped_width_bootstrap": bootstrap,
        "iota_profile": iota_profile,
        "resonances": resonances,
        "target_resonance": target,
        "local_shear_diota_drho": shear,
        "local_shear_source": "simsopt_pitch_average",
        "traced_local_shear_diota_drho": traced_shear,
        "pitch_iota_profile": pitch_iota_profile,
        "resonant_normal_error": amplitudes,
        "a63_relative": a63_relative,
        "width_law_predictor": predictor,
        "width_law_model": "sqrt(a63 / (2*pi*|diota/drho|))",
        "island_width_profile": width_profile,
        "trace_cache_npz": str(trace_cache),
        "island_zoom_png": str(zoom_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", required=True)
    parser.add_argument("--dof", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--rho-min", type=float, default=0.58)
    parser.add_argument("--rho-max", type=float, default=0.82)
    parser.add_argument("--zoom-nrho", type=int, default=16)
    parser.add_argument("--zoom-phases", type=int, default=6)
    parser.add_argument("--turns", type=int, default=2000)
    parser.add_argument("--theta0", type=float, default=0.5)
    parser.add_argument("--tol", type=float, default=1.0e-9)
    parser.add_argument(
        "--method",
        default="DIFFRAX_TSIT5",
        help=(
            "Poincare integrator: DIFFRAX_TSIT5 (batched, default), "
            "RK45/DOP853 (per-seed scipy), or BATCHED_RK45"
        ),
    )
    parser.add_argument("--fourier-n", type=int, default=48)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260722)
    args = parser.parse_args()

    try:
        result = diagnose(args)
        args.output_json.expanduser().resolve().write_text(
            json.dumps(result, indent=2) + "\n"
        )
        _log(f"wrote {args.output_json}", args.log.expanduser().resolve())
        return 0
    except Exception:
        log_path = args.log.expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        _log("FAILED\n" + traceback.format_exc(), log_path)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
