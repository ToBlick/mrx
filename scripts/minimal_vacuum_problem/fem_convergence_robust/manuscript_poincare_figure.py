#!/usr/bin/env python3
"""Manuscript-quality SIMSOPT vs MRX Poincaré comparison figure.

Builds a 2×3 panel figure from existing frozen-trace caches:

* row 1: logical ``(rho, theta)`` island zoom
* row 2: physical ``(R, Z)`` embedding on the common ``phi=0`` slice

Columns are SIMSOPT Biot–Savart, MRX ``8×32×8`` (5.33 elements/lobe), and
MRX ``8×36×8`` (6.00 elements/lobe).  No new field-line traces are computed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.minimal_vacuum_problem import compare_push_u_to_simsopt as cmp  # noqa: E402

MVP = ROOT / "scripts" / "minimal_vacuum_problem"
OUT = MVP / "fem_convergence_robust"
SIMSOPT_CACHE = OUT / "robust_simsopt_mrx_trace_11x22x11_phi0.npz"
MRX_8X32 = OUT / "robust_trace_8x32x8.npz"
MRX_8X36 = OUT / "robust_trace_8x36x8.npz"
SLICE_CACHE = OUT / "phi_zero_slice.npz"
META = MVP / "hodge_k2_nullspace_meta.json"

RHO_ZOOM = (0.60, 0.80)
THETA_ZOOM = (0.0, 1.0)


def _cached_lines(path: Path, prefix: str) -> list[np.ndarray]:
    data = np.load(path)
    keys = sorted(key for key in data.files if key.startswith(prefix))
    return [np.asarray(data[key], dtype=np.float64) for key in keys]


def _load_mrx_trace(path: Path) -> list[np.ndarray]:
    return _cached_lines(path, "line_")


def _load_simsopt_trace(path: Path) -> list[np.ndarray]:
    return _cached_lines(path, "simsopt_line_")


def _build_mapping() -> Any:
    import json

    hcurl_mod = cmp._load_hcurl_module()
    meta = json.loads(META.read_text())
    _, map_fn, _, _, _ = cmp._rebuild_sequence_from_meta(
        meta,
        hcurl_mod=hcurl_mod,
        pushforward_only=True,
    )
    return map_fn


def build_or_load_phi_zero_slice(
    map_fn: Any | None = None,
    *,
    cache_path: Path = SLICE_CACHE,
    nrho: int = 96,
    ntheta: int = 192,
    refresh: bool = False,
) -> dict[str, Any]:
    """Return (and optionally cache) the fine ``phi=0`` embedding grid."""
    if cache_path.exists() and not refresh:
        data = np.load(cache_path)
        return {key: np.asarray(data[key]) for key in data.files}
    if map_fn is None:
        map_fn = _build_mapping()
    physical_slice = cmp._physical_phi_zero_slice(
        map_fn, nrho=int(nrho), ntheta=int(ntheta)
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **physical_slice)
    return physical_slice


def plot_manuscript_poincare_figure(
    panels: list[tuple[str, list[np.ndarray]]],
    physical_slice: dict[str, Any],
    output_dir: Path,
    *,
    stem: str = "manuscript_poincare_simsopt_vs_mrx_8x32_8x36",
    rho_limits: tuple[float, float] = RHO_ZOOM,
    dpi: int = 600,
    marker_size: float = 0.05,
) -> tuple[Path, Path]:
    """Write the 2×3 manuscript Poincaré figure as PDF and PNG.

    Parameters
    ----------
    panels:
        Length-3 list of ``(title, logical_lines)`` for SIMSOPT and the two
        MRX grids.  Each line is an ``(n_hits, 2)`` array of ``(rho, theta)``.
    physical_slice:
        Output of :func:`cmp._physical_phi_zero_slice` (or a cached equivalent).
    """
    if len(panels) != 3:
        raise ValueError("manuscript figure expects exactly three panels")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "mathtext.fontset": "cm",
        }
    )

    physical_panels = [
        (title, cmp._logical_to_phi_zero_rz(lines, physical_slice))
        for title, lines in panels
    ]
    all_rz = np.concatenate(
        [point for _, lines in physical_panels for point in lines if len(point)]
    )
    if all_rz.size == 0:
        raise RuntimeError("no points available for physical embedding")
    padding = 0.03 * np.ptp(all_rz, axis=0)
    rz_lower = np.min(all_rz, axis=0) - padding
    rz_upper = np.max(all_rz, axis=0) + padding

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(12.5, 7.8),
        sharex="row",
        sharey="row",
        constrained_layout=True,
    )

    for column, (title, lines) in enumerate(panels):
        axis = axes[0, column]
        for line in lines:
            points = np.asarray(line, dtype=np.float64).reshape(-1, 2)
            axis.scatter(
                points[:, 1] % 1.0,
                points[:, 0],
                s=marker_size,
                alpha=0.55,
                c="0.15",
                linewidths=0,
                rasterized=True,
            )
        axis.set_xlim(*THETA_ZOOM)
        axis.set_ylim(*rho_limits)
        axis.set_title(title)
        axis.grid(True, alpha=0.22, linewidth=0.4)
        if column == 0:
            axis.set_ylabel(r"Logical radius $\rho$")
        if column == 1:
            axis.set_xlabel(r"Logical poloidal angle $\theta$ [cycles]")

    for column, (title, lines) in enumerate(physical_panels):
        axis = axes[1, column]
        for line in lines:
            points = np.asarray(line, dtype=np.float64).reshape(-1, 2)
            finite = np.isfinite(points).all(axis=1)
            points = points[finite]
            if points.size == 0:
                continue
            axis.scatter(
                points[:, 0],
                points[:, 1],
                s=marker_size,
                alpha=0.55,
                c="0.15",
                linewidths=0,
                rasterized=True,
            )
        axis.set_xlim(float(rz_lower[0]), float(rz_upper[0]))
        axis.set_ylim(float(rz_lower[1]), float(rz_upper[1]))
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, alpha=0.22, linewidth=0.4)
        if column == 0:
            axis.set_ylabel(r"$Z$ [m]")
        if column == 1:
            axis.set_xlabel(r"$R$ [m]")
        axis.set_title(title)

    axes[0, 0].annotate(
        r"logical $(\rho,\theta)$",
        xy=(0.0, 1.08),
        xycoords="axes fraction",
        fontsize=9,
        fontstyle="italic",
    )
    axes[1, 0].annotate(
        r"physical $\phi=0$ embedding",
        xy=(0.0, 1.08),
        xycoords="axes fraction",
        fontsize=9,
        fontstyle="italic",
    )

    pdf_path = (output_dir / f"{stem}.pdf").resolve()
    png_path = (output_dir / f"{stem}.png").resolve()
    figure.savefig(pdf_path, dpi=int(dpi), bbox_inches="tight")
    figure.savefig(png_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)
    return pdf_path, png_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-slice",
        action="store_true",
        help="recompute and overwrite the cached phi=0 slice",
    )
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT,
        help="directory for the PDF/PNG outputs",
    )
    args = parser.parse_args(argv)

    for path in (SIMSOPT_CACHE, MRX_8X32, MRX_8X36):
        if not path.exists():
            raise FileNotFoundError(f"missing trace cache: {path}")

    simsopt = _load_simsopt_trace(SIMSOPT_CACHE)
    mrx_832 = _load_mrx_trace(MRX_8X32)
    mrx_836 = _load_mrx_trace(MRX_8X36)
    physical_slice = build_or_load_phi_zero_slice(refresh=bool(args.refresh_slice))

    panels = [
        (
            r"SIMSOPT Biot–Savart"
            "\n"
            r"($32\rho\times6\theta$)",
            simsopt,
        ),
        (
            r"MRX $8\times32\times8$"
            "\n"
            r"(5.33 el/lobe, $16\rho\times6\theta$)",
            mrx_832,
        ),
        (
            r"MRX $8\times36\times8$"
            "\n"
            r"(6.00 el/lobe, $16\rho\times6\theta$)",
            mrx_836,
        ),
    ]
    pdf_path, png_path = plot_manuscript_poincare_figure(
        panels,
        physical_slice,
        args.output_dir,
        dpi=int(args.dpi),
    )
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
