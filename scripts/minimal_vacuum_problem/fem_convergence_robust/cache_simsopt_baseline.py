#!/usr/bin/env python3
"""Trace and cache the matched SIMSOPT/MRX island baseline once."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.minimal_vacuum_problem import compare_push_u_to_simsopt as cmp  # noqa: E402
from scripts.minimal_vacuum_problem.fem_convergence_robust import (  # noqa: E402
    chain_trend_diagnosis as trend,
)

MVP = ROOT / "scripts" / "minimal_vacuum_problem"
OUT = MVP / "fem_convergence_robust"
DOF = MVP / "fem_convergence_highres" / "fem_sweep_11x22x11_dof.npy"


def _cached_lines(path: Path, prefix: str) -> list[np.ndarray]:
    data = np.load(path)
    keys = sorted(key for key in data.files if key.startswith(prefix))
    return [np.asarray(data[key], dtype=np.float64) for key in keys]


def _logical_lines_to_rz(
    lines: list[np.ndarray],
    map_fn: object,
    *,
    max_points_per_line: int = 400,
) -> list[np.ndarray]:
    """Map a plotting-sized subset of logical section hits to physical R-Z."""
    physical: list[np.ndarray] = []
    for line in lines:
        points = np.asarray(line, dtype=np.float64).reshape(-1, 2)
        stride = max(1, int(np.ceil(points.shape[0] / max_points_per_line)))
        sampled = points[::stride]
        logical = jnp.asarray(
            np.column_stack(
                [sampled[:, 0], sampled[:, 1] % 1.0, np.zeros(sampled.shape[0])]
            ),
            dtype=jnp.float64,
        )
        xyz = np.asarray(jax.lax.map(map_fn, logical, batch_size=256))
        physical.append(
            np.column_stack(
                [np.linalg.norm(xyz[:, :2], axis=1), xyz[:, 2]]
            )
        )
    return physical


def _plot_physical_comparison(map_fn: object, cache_path: Path) -> Path:
    simsopt = _logical_lines_to_rz(
        _cached_lines(cache_path, "simsopt_line_"), map_fn
    )
    mrx = _logical_lines_to_rz(_cached_lines(cache_path, "mrx_line_"), map_fn)
    all_points = np.concatenate(simsopt + mrx)
    r_pad = 0.02 * np.ptp(all_points[:, 0])
    z_pad = 0.02 * np.ptp(all_points[:, 1])
    limits = (
        float(np.min(all_points[:, 0]) - r_pad),
        float(np.max(all_points[:, 0]) + r_pad),
        float(np.min(all_points[:, 1]) - z_pad),
        float(np.max(all_points[:, 1]) + z_pad),
    )
    fig, axes = plt.subplots(
        1, 2, figsize=(12.5, 5.2), sharex=True, sharey=True
    )
    for axis, lines, title in (
        (axes[0], simsopt, "SIMSOPT Biot--Savart"),
        (axes[1], mrx, r"MRX $11\times22\times11$"),
    ):
        for line in lines:
            axis.scatter(
                line[:, 0],
                line[:, 1],
                s=0.45,
                alpha=0.45,
                rasterized=True,
            )
        axis.set_xlim(limits[0], limits[1])
        axis.set_ylim(limits[2], limits[3])
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel(r"$R$ [m]")
        axis.set_ylabel(r"$Z$ [m]")
        axis.set_title(title)
        axis.grid(True, alpha=0.2)
    fig.suptitle(r"Matched physical $\zeta=0$ island zoom")
    fig.tight_layout()
    path = OUT / "poincare_physical_island_zoom_simsopt_vs_mrx_11x22x11.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    hcurl_mod = cmp._load_hcurl_module()
    meta = cmp._load_meta_for_dof(
        DOF,
        k=2,
        meta_json=MVP / "hodge_k2_nullspace_meta.json",
    )
    meta = dict(meta)
    meta["ns"] = [11, 22, 11]
    seq, map_fn, _, _, _ = cmp._rebuild_sequence_from_meta(
        meta,
        hcurl_mod=hcurl_mod,
        pushforward_only=True,
    )
    dof = jnp.asarray(np.load(DOF), dtype=jnp.float64).reshape(-1)
    simsopt_field, surfaces = cmp._load_simsopt_field(
        MVP / "serial0044970.json"
    )
    if not surfaces:
        raise RuntimeError("QUASR serialization contains no boundary surfaces")
    boundary = cmp._full_torus_surface(surfaces[-1])

    # SIMSOPT records crossings of the cylindrical phi=0 plane.  Logical
    # zeta=0 is a wavy physical section for this map, so inverting against it
    # creates a spurious theta-dependent radial displacement.
    logical_slice = trend._physical_phi_zero_slice(map_fn)
    cache_path = OUT / "robust_simsopt_mrx_trace_11x22x11.npz"
    result = cmp._simsopt_island_baseline(
        seq,
        dof,
        map_fn,
        simsopt_field,
        boundary,
        logical_slice,
        OUT,
        nfp=3,
        rho_min=0.58,
        rho_max=0.82,
        nrho=32,
        phases=6,
        theta0=0.5,
        mrx_turns=2000,
        mrx_tol=1.0e-9,
        simsopt_tmax=80000.0,
        simsopt_tol=1.0e-12,
        interpolation_degree=4,
        interpolation_points=24,
        minimum_intersections=100,
        cache_path=cache_path,
        mrx_method="DIFFRAX_TSIT5",
    )
    result["physical_comparison_file"] = str(
        _plot_physical_comparison(map_fn, cache_path)
    )
    (OUT / "robust_simsopt_baseline.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print("SIMSOPT_BASELINE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
