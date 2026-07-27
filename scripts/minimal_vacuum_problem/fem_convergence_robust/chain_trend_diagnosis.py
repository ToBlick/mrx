#!/usr/bin/env python3
"""Diagnose qualitative SIMSOPT/MRX island-chain differences.

The cached SIMSOPT trace is stored after converting physical ``(R, Z)`` section
hits to logical ``(rho, theta)``.  MRX is traced directly in logical
coordinates.  This script quantifies the coordinate-pipeline error by applying
the same map/inverse-map round trip to known logical points and cached MRX
hits.  It also checks whether logical ``zeta=0`` is the physical ``phi=0``
section used by SIMSOPT.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.minimal_vacuum_problem import compare_push_u_to_simsopt as cmp  # noqa: E402
MVP = ROOT / "scripts" / "minimal_vacuum_problem"
OUT = MVP / "fem_convergence_robust"
DOF = MVP / "fem_convergence_highres" / "fem_sweep_11x22x11_dof.npy"
CACHE = OUT / "robust_simsopt_mrx_trace_11x22x11.npz"
CORRECTED_CACHE = OUT / "robust_simsopt_mrx_trace_11x22x11_phi0.npz"
_MAP_EVALUATORS: dict[int, Any] = {}


def _cached_lines(path: Path, prefix: str) -> list[np.ndarray]:
    data = np.load(path)
    keys = sorted(key for key in data.files if key.startswith(prefix))
    return [np.asarray(data[key], dtype=np.float64) for key in keys]


def _batched_map(map_fn: Any, logical: np.ndarray, batch_size: int = 512) -> np.ndarray:
    """Evaluate the geometry map with one bounded-size JAX compilation."""
    points = np.asarray(logical, dtype=np.float64).reshape(-1, 3)
    evaluate = _MAP_EVALUATORS.setdefault(id(map_fn), jax.jit(jax.vmap(map_fn)))
    mapped: list[np.ndarray] = []
    for start in range(0, points.shape[0], int(batch_size)):
        chunk = points[start : start + int(batch_size)]
        count = chunk.shape[0]
        if count < int(batch_size):
            chunk = np.pad(chunk, ((0, int(batch_size) - count), (0, 0)), mode="edge")
        mapped.append(
            np.asarray(evaluate(jnp.asarray(chunk, dtype=jnp.float64)))[:count]
        )
    return np.concatenate(mapped)


def _build_mapping() -> Any:
    hcurl_mod = cmp._load_hcurl_module()
    meta = cmp._load_meta_for_dof(
        DOF,
        k=2,
        meta_json=MVP / "hodge_k2_nullspace_meta.json",
    )
    meta = dict(meta)
    meta["ns"] = [11, 22, 11]
    _, map_fn, _, _, _ = cmp._rebuild_sequence_from_meta(
        meta,
        hcurl_mod=hcurl_mod,
        pushforward_only=True,
    )
    return map_fn


def _logical_slice(map_fn: Any, nrho: int = 64, ntheta: int = 128) -> dict[str, Any]:
    logical, rho_grid, theta_grid = cmp._poloidal_slice_points(
        0.0, int(nrho), int(ntheta)
    )
    mapped = _batched_map(map_fn, logical).reshape(int(nrho), int(ntheta), 3)
    return {
        "zeta": 0.0,
        "rho": rho_grid,
        "theta": theta_grid,
        "R": np.linalg.norm(mapped[..., :2], axis=-1),
        "Z": mapped[..., 2],
        "phi": np.arctan2(mapped[..., 1], mapped[..., 0]),
    }


# Re-export primary-module helpers so callers and this script share one
# implementation of the phi=0 section correction.
_periodic_difference = cmp._periodic_difference
_physical_phi_zero_slice = cmp._physical_phi_zero_slice
_logical_to_phi_zero_rz = cmp._logical_to_phi_zero_rz


def _map_logical_lines_to_rz(lines: list[np.ndarray], map_fn: Any) -> list[np.ndarray]:
    points = [
        np.asarray(line, dtype=np.float64).reshape(-1, 2) for line in lines
    ]
    counts = np.asarray([item.shape[0] for item in points], dtype=int)
    flat = np.concatenate(points)
    logical = jnp.asarray(
        np.column_stack(
            [flat[:, 0], flat[:, 1] % 1.0, np.zeros(flat.shape[0])]
        ),
        dtype=jnp.float64,
    )
    xyz = _batched_map(map_fn, np.asarray(logical))
    rz = np.column_stack([np.linalg.norm(xyz[:, :2], axis=1), xyz[:, 2]])
    split_at = np.cumsum(counts)[:-1]
    return list(np.split(rz, split_at))


def _errors(
    original: list[np.ndarray], recovered: list[np.ndarray]
) -> dict[str, float]:
    delta_rho = np.concatenate(
        [new[:, 0] - old[:, 0] for old, new in zip(original, recovered)]
    )
    delta_theta = np.concatenate(
        [
            _periodic_difference(new[:, 1], old[:, 1])
            for old, new in zip(original, recovered)
        ]
    )
    return {
        "rho_rms": float(np.sqrt(np.mean(delta_rho**2))),
        "rho_max_abs": float(np.max(np.abs(delta_rho))),
        "theta_rms_cycles": float(np.sqrt(np.mean(delta_theta**2))),
        "theta_max_abs_cycles": float(np.max(np.abs(delta_theta))),
    }


def _subsample_lines(
    lines: list[np.ndarray], maximum_points: int = 256
) -> list[np.ndarray]:
    """Keep the audit fast without changing any plotted orbit's extent."""
    sampled: list[np.ndarray] = []
    for line in lines:
        points = np.asarray(line, dtype=np.float64).reshape(-1, 2)
        stride = max(1, int(np.ceil(points.shape[0] / int(maximum_points))))
        sampled.append(points[::stride])
    return sampled


def _plot_overlay(
    simsopt: list[np.ndarray],
    mrx_direct: list[np.ndarray],
    mrx_roundtrip: list[np.ndarray],
) -> Path:
    all_points = np.concatenate(simsopt + mrx_direct + mrx_roundtrip)
    rho_limits = (
        float(np.nanpercentile(all_points[:, 0], 0.05)),
        float(np.nanpercentile(all_points[:, 0], 99.95)),
    )
    fig, axes = plt.subplots(
        1, 3, figsize=(15.5, 4.8), sharex=True, sharey=True
    )
    panels = (
        (simsopt, "SIMSOPT: physical trace then inversion"),
        (mrx_direct, "MRX: direct logical trace"),
        (mrx_roundtrip, "MRX: map then same inversion"),
    )
    for axis, (lines, title) in zip(axes, panels):
        for line in lines:
            axis.scatter(
                line[:, 1],
                line[:, 0],
                s=0.18,
                alpha=0.35,
                rasterized=True,
            )
        axis.set_xlim(0.0, 1.0 / 6.0)
        axis.set_ylim(*rho_limits)
        axis.set_xlabel(r"$\theta$ [cycles]")
        axis.set_title(title)
        axis.grid(True, alpha=0.2)
    axes[0].set_ylabel(r"$\rho$")
    fig.suptitle("Island-chain coordinate-pipeline audit")
    fig.tight_layout()
    path = OUT / "chain_trend_pipeline_audit.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_corrected_cache(
    simsopt: list[np.ndarray], mrx: list[np.ndarray]
) -> None:
    source = np.load(CACHE)
    payload: dict[str, np.ndarray] = {
        "logical_seeds": np.asarray(source["logical_seeds"])
    }
    for index, line in enumerate(simsopt):
        payload[f"simsopt_line_{index:04d}"] = np.asarray(line)
    for index, line in enumerate(mrx):
        payload[f"mrx_line_{index:04d}"] = np.asarray(line)
    np.savez_compressed(CORRECTED_CACHE, **payload)


def _plot_physical_phi_zero(
    simsopt: list[np.ndarray],
    mrx: list[np.ndarray],
    physical_slice: dict[str, Any],
) -> Path:
    simsopt_trapped = [
        line
        for line in simsopt
        if cmp._classify_resonant_orbit(line, poloidal_mode=6)["trapped"]
    ]
    mrx_trapped = [
        line
        for line in mrx
        if cmp._classify_resonant_orbit(line, poloidal_mode=6)["trapped"]
    ]
    simsopt_rz = _logical_to_phi_zero_rz(
        _subsample_lines(simsopt_trapped), physical_slice
    )
    mrx_rz = _logical_to_phi_zero_rz(
        _subsample_lines(mrx_trapped), physical_slice
    )
    all_points = np.concatenate(simsopt_rz + mrx_rz)
    padding = 0.02 * np.ptp(all_points, axis=0)
    lower = np.min(all_points, axis=0) - padding
    upper = np.max(all_points, axis=0) + padding
    figure, axes = plt.subplots(
        1, 2, figsize=(12.5, 5.2), sharex=True, sharey=True
    )
    for axis, lines, title in (
        (axes[0], simsopt_rz, "SIMSOPT Biot–Savart"),
        (axes[1], mrx_rz, r"MRX $11\times22\times11$"),
    ):
        for line in lines:
            axis.scatter(
                line[:, 0], line[:, 1], s=0.35, alpha=0.4, rasterized=True
            )
        axis.set_xlim(lower[0], upper[0])
        axis.set_ylim(lower[1], upper[1])
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel(r"$R$ [m]")
        axis.set_ylabel(r"$Z$ [m]")
        axis.set_title(title)
        axis.grid(True, alpha=0.2)
    figure.suptitle(r"Trapped island orbits in a common physical $\phi=0$ embedding")
    figure.tight_layout()
    path = OUT / "poincare_physical_island_zoom_simsopt_vs_mrx_11x22x11.png"
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)
    return path


def main() -> None:
    if not CACHE.exists():
        raise FileNotFoundError(f"missing cached trace: {CACHE}")
    map_fn = _build_mapping()
    logical_slice = _logical_slice(map_fn)
    physical_slice = _physical_phi_zero_slice(map_fn)

    phi = np.unwrap(np.asarray(logical_slice["phi"]), axis=1)
    section_plane = {
        "phi_rms_radians": float(np.sqrt(np.mean(phi**2))),
        "phi_max_abs_radians": float(np.max(np.abs(phi))),
        "phi_peak_to_peak_radians": float(np.ptp(phi)),
    }

    rho = np.linspace(0.58, 0.82, 49)
    theta = np.arange(192, dtype=np.float64) / 192.0
    rho_grid, theta_grid = np.meshgrid(rho, theta, indexing="ij")
    roundtrip_original = [
        np.column_stack([rho_grid.ravel(), theta_grid.ravel()])
    ]
    roundtrip_physical = _map_logical_lines_to_rz(roundtrip_original, map_fn)
    roundtrip_recovered = cmp._physical_sections_to_logical(
        roundtrip_physical, logical_slice
    )
    grid_roundtrip = _errors(roundtrip_original, roundtrip_recovered)

    simsopt_full = _cached_lines(CACHE, "simsopt_line_")
    mrx_full = _cached_lines(CACHE, "mrx_line_")
    simsopt = _subsample_lines(simsopt_full)
    mrx_direct = _subsample_lines(mrx_full)
    mrx_physical = _map_logical_lines_to_rz(mrx_direct, map_fn)
    mrx_roundtrip = cmp._physical_sections_to_logical(
        mrx_physical, logical_slice
    )
    cached_mrx_roundtrip = _errors(mrx_direct, mrx_roundtrip)
    figure = _plot_overlay(simsopt, mrx_direct, mrx_roundtrip)

    reconstructed_simsopt_rz = _map_logical_lines_to_rz(simsopt, map_fn)
    simsopt_phi_zero = cmp._physical_sections_to_logical(
        reconstructed_simsopt_rz, physical_slice
    )
    _save_corrected_cache(simsopt_phi_zero, mrx_direct)
    seeds = np.asarray(np.load(CACHE)["logical_seeds"])
    reference_path, comparison_path = cmp._plot_reference_island_zooms(
        simsopt_phi_zero,
        mrx_direct,
        seeds,
        OUT,
        rho_min=0.58,
        rho_max=0.82,
        dpi=190,
    )
    physical_path = _plot_physical_phi_zero(
        simsopt_phi_zero, mrx_direct, physical_slice
    )

    result = {
        "section_plane": section_plane,
        "physical_phi_zero_slice": {
            "phi_rms_radians": float(
                np.sqrt(np.mean(np.asarray(physical_slice["phi"]) ** 2))
            ),
            "phi_max_abs_radians": float(
                np.max(np.abs(np.asarray(physical_slice["phi"])))
            ),
            "zeta_min": float(np.min(np.asarray(physical_slice["zeta"]))),
            "zeta_max": float(np.max(np.asarray(physical_slice["zeta"]))),
        },
        "grid_roundtrip": grid_roundtrip,
        "cached_mrx_roundtrip": cached_mrx_roundtrip,
        "interpretation": "section_plane_mismatch_dominates_visual_radial_trend",
        "pipeline_audit_figure": str(figure),
        "corrected_cache": str(CORRECTED_CACHE),
        "corrected_reference_figure": str(reference_path),
        "corrected_comparison_figure": str(comparison_path),
        "common_phi_zero_embedding_figure": str(physical_path),
    }
    path = OUT / "chain_trend_diagnosis.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
