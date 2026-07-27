#!/usr/bin/env python3
"""Establish convergence of the SIMSOPT reference-island width."""
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
OUT = MVP / "fem_convergence_robust" / "simsopt_width_convergence"
DOF = MVP / "fem_convergence_highres" / "fem_sweep_11x22x11_dof.npy"

# One-factor-at-a-time around the canonical baseline. This tests every knob in
# the report plan without an unnecessary 48-case Cartesian product.
CASES = [
    {"name": "t10k", "tmax": 10000, "nrho": 16, "phases": 3, "tol": 1e-12},
    {"name": "baseline", "tmax": 20000, "nrho": 16, "phases": 3, "tol": 1e-12},
    {"name": "t40k", "tmax": 40000, "nrho": 16, "phases": 3, "tol": 1e-12},
    {"name": "t80k", "tmax": 80000, "nrho": 16, "phases": 3, "tol": 1e-12},
    {"name": "rho24", "tmax": 20000, "nrho": 24, "phases": 3, "tol": 1e-12},
    {"name": "rho32", "tmax": 20000, "nrho": 32, "phases": 3, "tol": 1e-12},
    {"name": "phase6", "tmax": 20000, "nrho": 16, "phases": 6, "tol": 1e-12},
    {"name": "tol1e-13", "tmax": 20000, "nrho": 16, "phases": 3, "tol": 1e-13},
]


def _logical_slice(map_fn: Any) -> dict[str, Any]:
    logical, rho, theta = cmp._poloidal_slice_points(0.0, 96, 256)
    mapped = np.asarray(
        jax.lax.map(
            map_fn,
            jnp.asarray(logical, dtype=jnp.float64),
            batch_size=512,
        )
    ).reshape(96, 256, 3)
    return {
        "zeta": 0.0,
        "rho": rho,
        "theta": theta,
        "R": np.sqrt(mapped[..., 0] ** 2 + mapped[..., 1] ** 2),
        "Z": mapped[..., 2],
    }


def _run_case(
    case: dict[str, Any],
    *,
    map_fn: Any,
    simsopt_field: Any,
    boundary: Any,
    logical_slice: dict[str, Any],
) -> dict[str, Any]:
    path = OUT / f"{case['name']}.json"
    if path.is_file():
        return json.loads(path.read_text())
    seed_rho = np.linspace(0.58, 0.82, int(case["nrho"]))
    seed_theta = (
        0.5 + np.arange(int(case["phases"])) / (6.0 * int(case["phases"]))
    ) % 1.0
    logical_seeds = np.asarray(
        [[rho, theta, 0.0] for theta in seed_theta for rho in seed_rho]
    )
    mapped = np.asarray(
        jax.lax.map(
            map_fn,
            jnp.asarray(logical_seeds, dtype=jnp.float64),
            batch_size=min(64, logical_seeds.shape[0]),
        )
    )
    physical_seeds = np.column_stack(
        [np.linalg.norm(mapped[:, :2], axis=1), mapped[:, 2]]
    )
    sections, _, _, lost, seconds = cmp._trace_simsopt_poincare(
        simsopt_field,
        boundary,
        physical_seeds,
        [0.0],
        nfp=3,
        tol=float(case["tol"]),
        tmax=float(case["tmax"]),
        interpolation_degree=4,
        interpolation_points=24,
    )
    lines = cmp._physical_sections_to_logical(sections[0.0], logical_slice)
    counts = np.asarray([line.shape[0] for line in lines])
    valid = (counts >= 100) & ~np.asarray(lost)
    valid_lines = [line for line, keep in zip(lines, valid) if bool(keep)]
    valid_seeds = logical_seeds[valid]
    width_profile = cmp._island_width_profile(valid_lines, valid_seeds[:, 0])
    trapped = cmp._trapped_separatrix_summary(
        valid_lines, valid_seeds, poloidal_mode=6
    )
    result = {
        **case,
        "rho_min": 0.58,
        "rho_max": 0.82,
        "interpolation_degree": 4,
        "interpolation_points": 24,
        "total_lines": int(logical_seeds.shape[0]),
        "valid_lines": int(np.sum(valid)),
        "lost_lines": int(np.sum(lost)),
        "median_intersections": float(np.median(counts)),
        "trace_seconds": float(seconds),
        "max_detrended_width": float(
            max(item["detrended_width_rho_q05_q95"] for item in width_profile)
        ),
        "trapped_width": float(trapped["trapped_separatrix_width_rho"]),
        "trapped_line_count": int(trapped["trapped_line_count"]),
    }
    path.write_text(json.dumps(result, indent=2) + "\n")
    payload: dict[str, np.ndarray] = {"logical_seeds": logical_seeds}
    for index, line in enumerate(lines):
        payload[f"line_{index:04d}"] = np.asarray(line)
    np.savez_compressed(OUT / f"{case['name']}.npz", **payload)
    print(
        f"{case['name']}: trapped={result['trapped_width']:.6f} "
        f"detrended={result['max_detrended_width']:.6f} "
        f"hits={result['median_intersections']:.0f}",
        flush=True,
    )
    return result


def _plot(results: list[dict[str, Any]]) -> None:
    by_name = {item["name"]: item for item in results}
    baseline = by_name["baseline"]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), sharey=True)
    groups = [
        ("SIMSOPT integration length", ["t10k", "baseline", "t40k", "t80k"], "tmax"),
        ("Radial seed count", ["baseline", "rho24", "rho32"], "nrho"),
        ("Phase/tolerance checks", ["baseline", "phase6", "tol1e-13"], "name"),
    ]
    for axis, (title, names, xkey) in zip(axes, groups):
        chosen = [by_name[name] for name in names]
        if xkey == "name":
            x = np.arange(len(chosen))
            axis.set_xticks(x, ["3 phases", "6 phases", r"$10^{-13}$ tol"])
        else:
            x = [item[xkey] for item in chosen]
            axis.set_xlabel(xkey)
        axis.plot(x, [item["trapped_width"] for item in chosen], "o-", label="trapped")
        axis.plot(
            x,
            [item["max_detrended_width"] for item in chosen],
            "s--",
            label="max detrended",
        )
        axis.axhline(
            baseline["trapped_width"], color="tab:green", linestyle=":", linewidth=1
        )
        axis.set_title(title)
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("Logical island width")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT.parent / "simsopt_width_convergence.png", dpi=200)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    hcurl = cmp._load_hcurl_module()
    meta = dict(
        cmp._load_meta_for_dof(
            DOF, k=2, meta_json=MVP / "hodge_k2_nullspace_meta.json"
        )
    )
    meta["ns"] = [11, 22, 11]
    _, map_fn, _, _, _ = cmp._rebuild_sequence_from_meta(
        meta, hcurl_mod=hcurl, pushforward_only=True
    )
    simsopt_field, surfaces = cmp._load_simsopt_field(MVP / "serial0044970.json")
    boundary = cmp._full_torus_surface(surfaces[-1])
    logical_slice = _logical_slice(map_fn)
    results = [
        _run_case(
            case,
            map_fn=map_fn,
            simsopt_field=simsopt_field,
            boundary=boundary,
            logical_slice=logical_slice,
        )
        for case in CASES
    ]
    _plot(results)
    summary = {"cases": results, "baseline": next(r for r in results if r["name"] == "baseline")}
    (OUT.parent / "simsopt_width_convergence.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print("SIMSOPT_WIDTH_CONVERGENCE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
