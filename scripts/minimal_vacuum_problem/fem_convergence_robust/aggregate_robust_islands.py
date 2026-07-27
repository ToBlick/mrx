#!/usr/bin/env python3
"""Aggregate robust island diagnostics and generate convergence figures."""
from __future__ import annotations

import json
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

OUT = ROOT / "scripts" / "minimal_vacuum_problem" / "fem_convergence_robust"
HIGHRES = (
    ROOT / "scripts" / "minimal_vacuum_problem" / "fem_convergence_highres"
)
SIMSOPT_DETRENDED_BAND = (0.0486, 0.0503)
QUARANTINED_LABELS = {"13x18x13"}
TRUNCATED_TRACE_LABELS = {"12x24x12"}
# Dense verification replaces the saved DOF with the dense eigenvector, so a
# low iterative/dense M2 overlap is a solver-health warning about the
# iterative warm-start, not a disqualification of the saved field.  Only
# explicitly listed labels are quarantined from fits.
MIN_M2_OVERLAP_CAVEAT = 0.90


def _m2_overlap_lookup() -> dict[str, float]:
    """Collect dense-verification overlaps from solve / synthesis JSON files."""
    overlaps: dict[str, float] = {}
    candidates = [
        HIGHRES / "fem_convergence_highres_synthesis.json",
        *sorted(OUT.glob("robust_solve_*.json")),
        *sorted(
            (ROOT / "scripts" / "minimal_vacuum_problem").glob(
                "fem_convergence_*.json"
            )
        ),
    ]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        stack: list[Any] = [data]
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                if "ns" in item and "dense_verification" in item:
                    label = "x".join(str(int(value)) for value in item["ns"])
                    overlap = (item.get("dense_verification") or {}).get(
                        "m2_overlap_with_iterative"
                    )
                    if overlap is not None:
                        overlaps[label] = float(overlap)
                stack.extend(item.values())
            elif isinstance(item, list):
                stack.extend(item)
    return overlaps


def _annotate_reliability(
    record: dict[str, Any],
    overlaps: dict[str, float],
) -> dict[str, Any]:
    label = str(record["label"])
    turns = int(record.get("tracer", {}).get("turns", 0) or 0)
    overlap = overlaps.get(label)
    quarantined = label in QUARANTINED_LABELS
    truncated = label in TRUNCATED_TRACE_LABELS or turns < 2000
    iterative_caveat = (
        overlap is not None and float(overlap) < MIN_M2_OVERLAP_CAVEAT
    )
    return {
        "label": label,
        "m2_overlap_with_iterative": overlap,
        "quarantined_nullspace": quarantined,
        "iterative_solver_caveat": iterative_caveat,
        "truncated_trace": truncated,
        "trusted_for_fit": not quarantined and not truncated,
        "turns": turns,
        "elements_per_lobe": float(int(record["ns"][1]) / 6.0),
    }


def _load_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    overlaps = _m2_overlap_lookup()
    for path in OUT.glob("robust_*x*x*.json"):
        # Keep only the canonical grid records: robust_<nr>x<nt>x<nz>.json
        if path.stem.count("x") != 2 or "_" in path.stem.split("x", 1)[0].removeprefix(
            "robust_"
        ):
            continue
        if any(token in path.stem for token in ("smoke", "probe", "solve")):
            continue
        try:
            record = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        bootstrap = record.get("trapped_width_bootstrap") or {}
        if (
            "width_law_predictor" in record
            and bootstrap.get("q05") is not None
            and bootstrap.get("q95") is not None
        ):
            record = dict(record)
            record["reliability"] = _annotate_reliability(record, overlaps)
            records.append(record)
    return sorted(records, key=lambda item: int(item["n2_dbc"]))


def _width(record: dict[str, Any]) -> float:
    return float(record["trapped_separatrix"]["trapped_separatrix_width_rho"])


def _detrended_width(record: dict[str, Any]) -> float:
    return float(
        max(
            item["detrended_width_rho_q05_q95"]
            for item in record["island_width_profile"]
        )
    )


def _ci(record: dict[str, Any]) -> tuple[float, float]:
    bootstrap = record["trapped_width_bootstrap"]
    return float(bootstrap["q05"]), float(bootstrap["q95"])


def _fit_width_law(records: list[dict[str, Any]]) -> dict[str, Any]:
    # Prefer trusted, poloidally resolved grids; fall back only if too few remain.
    fit_records = [
        item
        for item in records
        if item.get("reliability", {}).get("trusted_for_fit", True)
        and float(item["aligned_metrics"]["rel_l2_aligned"]) < 2.5e-3
        and int(item["ns"][1]) >= 20
    ]
    if len(fit_records) < 4:
        fit_records = [
            item
            for item in records
            if item.get("reliability", {}).get("trusted_for_fit", True)
        ]
    if len(fit_records) < 4:
        fit_records = records
    x = np.asarray(
        [item["width_law_predictor"] for item in fit_records],
        dtype=np.float64,
    )
    y = np.asarray([_width(item) for item in fit_records], dtype=np.float64)
    design = np.column_stack([np.ones_like(x), x])
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = intercept + slope * x
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    rng = np.random.default_rng(20260722)
    intercepts = np.empty(5000)
    for index in range(intercepts.size):
        selected = rng.integers(0, len(x), size=len(x))
        sample_design = design[selected]
        sample_y = y[selected]
        if np.linalg.matrix_rank(sample_design) < 2:
            intercepts[index] = np.nan
        else:
            intercepts[index] = np.linalg.lstsq(
                sample_design, sample_y, rcond=None
            )[0][0]
    intercepts = intercepts[np.isfinite(intercepts)]
    intercept_ci = np.quantile(intercepts, [0.05, 0.95])
    return {
        "fit_labels": [item["label"] for item in fit_records],
        "aligned_l2_threshold": 2.5e-3,
        "zero_error_intercept": float(intercept),
        "slope": float(slope),
        "r_squared": r_squared,
        "intercept_ci90": [float(value) for value in intercept_ci],
    }


def _a63(record: dict[str, Any]) -> float:
    value = record.get("a63_relative")
    if value is not None:
        return float(value)
    resonances = record.get("resonant_normal_error") or []
    for item in resonances:
        if int(item.get("m", -1)) == 6 and int(item.get("n", -1)) == 3:
            return float(item["normal_error_fourier_relative"])
    raise KeyError(f"missing a63 for {record.get('label')}")


def _is_isotropic(record: dict[str, Any]) -> bool:
    nr, nt, nz = (int(value) for value in record["ns"])
    return nr == nz and nt == 2 * nr


def _isotropic_l2_fit(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Fit ``L2 ~ C * h^p`` on the isotropic ladder ``nr = nz``, ``nt = 2 nr``.

    The mesh scale is taken as ``h = 1/nr``.  Quarantined nullspaces are
    excluded so the fit reflects the dense-verified field ladder.
    """
    isotropic = [
        item
        for item in records
        if _is_isotropic(item)
        and not (item.get("reliability") or {}).get("quarantined_nullspace")
    ]
    if len(isotropic) < 3:
        return {
            "labels": [item["label"] for item in isotropic],
            "order_p": float("nan"),
            "log_prefactor": float("nan"),
            "r_squared": float("nan"),
        }
    h = np.asarray([1.0 / float(item["ns"][0]) for item in isotropic], dtype=np.float64)
    l2 = np.asarray(
        [float(item["aligned_metrics"]["rel_l2_aligned"]) for item in isotropic],
        dtype=np.float64,
    )
    log_h = np.log(h)
    log_l2 = np.log(l2)
    design = np.column_stack([np.ones_like(log_h), log_h])
    log_c, order_p = np.linalg.lstsq(design, log_l2, rcond=None)[0]
    fitted = log_c + order_p * log_h
    ss_res = float(np.sum((log_l2 - fitted) ** 2))
    ss_tot = float(np.sum((log_l2 - np.mean(log_l2)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "labels": [item["label"] for item in isotropic],
        "order_p": float(order_p),
        "log_prefactor": float(log_c),
        "r_squared": float(r_squared),
        "h": h.tolist(),
        "rel_l2_aligned": l2.tolist(),
    }


def _record_marker_color(record: dict[str, Any]) -> str:
    reliability = record.get("reliability") or {}
    if reliability.get("quarantined_nullspace"):
        return "0.45"
    if reliability.get("truncated_trace"):
        return "tab:orange"
    if int(record["ns"][0]) == 8:
        return "tab:red"
    if int(record["ns"][1]) >= 20:
        return "tab:blue"
    return "0.6"


def _plot_field_l2(records: list[dict[str, Any]]) -> None:
    """Log-log aligned relative L2 versus k=2 Dirichlet DOFs."""
    fig, axis = plt.subplots(figsize=(8.4, 5.2))
    dofs = np.asarray([int(item["n2_dbc"]) for item in records], dtype=np.float64)
    l2 = np.asarray(
        [float(item["aligned_metrics"]["rel_l2_aligned"]) for item in records],
        dtype=np.float64,
    )
    colors = [_record_marker_color(item) for item in records]
    axis.scatter(dofs, l2, c=colors, s=48, zorder=3)
    for record, x, y in zip(records, dofs, l2):
        axis.annotate(
            record["label"],
            (x, y),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=6.5,
        )
    # Connect the fixed-aperture 8x poloidal ladder so the new points stand out.
    poloidal = sorted(
        (item for item in records if int(item["ns"][0]) == 8),
        key=lambda item: int(item["ns"][1]),
    )
    if len(poloidal) >= 2:
        axis.plot(
            [int(item["n2_dbc"]) for item in poloidal],
            [
                float(item["aligned_metrics"]["rel_l2_aligned"])
                for item in poloidal
            ],
            color="tab:red",
            linewidth=1.0,
            alpha=0.65,
            zorder=2,
            label=r"fixed-aperture $8\times n_\theta\times 8$",
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(r"$k=2$ Dirichlet DOFs ($n_{2,\mathrm{dbc}}$)")
    axis.set_ylabel(r"Aligned relative $L^2$")
    axis.scatter([], [], color="tab:red", label=r"$n_r=n_\zeta=8$ ladder")
    axis.scatter([], [], color="tab:blue", label=r"trusted, $n_\theta\geq20$")
    axis.scatter([], [], color="0.6", label=r"under-resolved ($n_\theta<20$)")
    axis.scatter([], [], color="tab:orange", label="truncated trace")
    axis.scatter([], [], color="0.45", label="quarantined nullspace")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "robust_l2_vs_dofs.png", dpi=180)
    plt.close(fig)


def _plot_resonant_amplitude(records: list[dict[str, Any]]) -> None:
    """Resonant normal-error amplitude versus grid label, ordered by DOFs."""
    ordered = sorted(records, key=lambda item: int(item["n2_dbc"]))
    labels = [item["label"] for item in ordered]
    a63 = np.asarray([_a63(item) for item in ordered], dtype=np.float64)
    x = np.arange(len(ordered))
    colors = [_record_marker_color(item) for item in ordered]
    fig, axis = plt.subplots(figsize=(10.0, 5.2))
    axis.scatter(x, a63, c=colors, s=48, zorder=3)
    axis.plot(x, a63, color="0.75", linewidth=0.8, zorder=1)
    # Overlay poloidal-ladder trend separately for visual grouping by n_theta.
    by_theta: dict[int, list[tuple[int, float]]] = {}
    for index, record in enumerate(ordered):
        by_theta.setdefault(int(record["ns"][1]), []).append((index, a63[index]))
    for ntheta, points in sorted(by_theta.items()):
        if len(points) < 2:
            continue
        xs, ys = zip(*points)
        axis.plot(
            xs,
            ys,
            color="tab:purple",
            linewidth=0.7,
            alpha=0.35,
            zorder=1,
        )
    axis.set_yscale("log")
    axis.set_xticks(x, labels, rotation=38, ha="right")
    axis.set_ylabel(r"Resonant normal-error amplitude $a_{6,3}$")
    axis.scatter([], [], color="tab:red", label=r"$n_r=n_\zeta=8$ ladder")
    axis.scatter([], [], color="tab:blue", label=r"trusted, $n_\theta\geq20$")
    axis.scatter([], [], color="0.6", label=r"under-resolved ($n_\theta<20$)")
    axis.scatter([], [], color="tab:orange", label="truncated trace")
    axis.scatter([], [], color="0.45", label="quarantined nullspace")
    axis.grid(True, which="both", axis="y", alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "robust_a63_vs_grid.png", dpi=180)
    plt.close(fig)


def _plot_width_law(
    records: list[dict[str, Any]],
    fit: dict[str, Any],
) -> None:
    fig, axis = plt.subplots(figsize=(8.2, 5.4))
    fit_labels = set(fit["fit_labels"])
    for record in records:
        x = float(record["width_law_predictor"])
        y = _width(record)
        q05, q95 = _ci(record)
        color = "tab:blue" if record["label"] in fit_labels else "0.55"
        axis.errorbar(
            x,
            y,
            yerr=[[y - q05], [q95 - y]],
            fmt="o",
            color=color,
            capsize=2,
        )
        axis.annotate(
            record["label"],
            (x, y),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=7,
        )
    xmax = max(float(item["width_law_predictor"]) for item in records)
    grid = np.linspace(0.0, 1.05 * xmax, 200)
    axis.plot(
        grid,
        fit["zero_error_intercept"] + fit["slope"] * grid,
        color="black",
        label=rf"high-res fit, $R^2={fit['r_squared']:.2f}$",
    )
    axis.errorbar(
        0.0,
        fit["zero_error_intercept"],
        yerr=[
            [
                fit["zero_error_intercept"]
                - fit["intercept_ci90"][0]
            ],
            [
                fit["intercept_ci90"][1]
                - fit["zero_error_intercept"]
            ],
        ],
        fmt="s",
        color="black",
        capsize=4,
        label="zero-error intercept (90% bootstrap CI)",
    )
    axis.set_xlabel(
        r"Numerical island predictor "
        r"$\sqrt{a_{6,3}/(2\pi\,|\mathrm{d}\iota/\mathrm{d}\rho|)}$"
    )
    axis.set_ylabel(r"Trapped-separatrix width $w_{\mathrm{trapped}}$")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "robust_width_law.png", dpi=180)
    plt.close(fig)


def _plot_detrended_width(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare MRX detrended widths with the converged SIMSOPT band."""
    labels = [record["label"] for record in records]
    widths = np.asarray([_detrended_width(record) for record in records])
    ntheta = np.asarray([int(record["ns"][1]) for record in records])
    x = np.arange(len(records))
    fig, axis = plt.subplots(figsize=(10.0, 5.2))
    colors = []
    for record, nth in zip(records, ntheta):
        reliability = record.get("reliability") or {}
        if reliability.get("quarantined_nullspace"):
            colors.append("0.45")
        elif reliability.get("truncated_trace"):
            colors.append("tab:orange")
        elif nth >= 20:
            colors.append("tab:blue")
        else:
            colors.append("0.6")
    axis.scatter(x, widths, c=colors, s=42, zorder=3)
    axis.plot(x, widths, color="0.75", linewidth=0.8, zorder=1)
    axis.axhspan(
        SIMSOPT_DETRENDED_BAND[0],
        SIMSOPT_DETRENDED_BAND[1],
        color="tab:green",
        alpha=0.2,
        label="converged SIMSOPT detrended band",
    )
    axis.scatter([], [], color="tab:blue", label=r"trusted, $n_\theta\geq20$")
    axis.scatter([], [], color="0.6", label=r"under-resolved ($n_\theta<20$)")
    axis.scatter([], [], color="tab:orange", label="truncated trace")
    axis.scatter([], [], color="0.45", label="quarantined nullspace")
    axis.set_xticks(x, labels, rotation=38, ha="right")
    axis.set_ylabel(r"Maximum Fourier-detrended width $\Delta\rho$")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "robust_detrended_width_vs_grid.png", dpi=180)
    plt.close(fig)
    return {
        "simsopt_band": list(SIMSOPT_DETRENDED_BAND),
        "labels": labels,
        "mrx_detrended_widths": widths.tolist(),
        "reliability": [item.get("reliability") for item in records],
    }


def _plot_location(records: list[dict[str, Any]]) -> dict[str, Any]:
    fig, axis = plt.subplots(figsize=(9.0, 5.2))
    cmap = plt.cm.viridis
    for index, record in enumerate(records):
        profile = record["iota_profile"]
        rho = [float(item["rho"]) for item in profile]
        iota = [float(item["iota"]) for item in profile]
        axis.plot(
            rho,
            iota,
            marker=".",
            linewidth=0.9,
            alpha=0.75,
            color=cmap(index / max(len(records) - 1, 1)),
            label=record["label"],
        )
    axis.axhline(0.5, color="black", linestyle="--", linewidth=1)
    axis.set_xlabel(r"Logical radius $\rho$")
    axis.set_ylabel(r"Rotational transform $\iota$")
    axis.grid(True, alpha=0.25)
    axis.legend(ncol=4, fontsize=6)
    fig.tight_layout()
    fig.savefig(OUT / "robust_iota_profiles.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9.2, 4.8))
    labels = [item["label"] for item in records]
    rhos = [float(item["target_resonance"]["rho"]) for item in records]
    axis.plot(np.arange(len(records)), rhos, "o-", color="tab:purple")
    axis.set_xticks(np.arange(len(records)), labels, rotation=55, ha="right")
    axis.set_ylabel(r"Resonance location $\rho^\star$ at $\iota=1/2$")
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "robust_rho_star_vs_grid.png", dpi=180)
    plt.close(fig)
    return {
        "rho_star_min": float(np.min(rhos)),
        "rho_star_max": float(np.max(rhos)),
        "finest_rho_star": float(rhos[-1]),
    }


def _plot_balanced_width(records: list[dict[str, Any]]) -> None:
    # Show every grid ordered by DOF with its bootstrap interval, and mark the
    # poloidally resolved grids so the reader can see the fixed-protocol trend.
    selected = sorted(records, key=lambda item: int(item["n2_dbc"]))
    if len(selected) < 2:
        return
    x = np.arange(len(selected))
    y = np.asarray([_width(item) for item in selected])
    q = np.asarray([_ci(item) for item in selected])
    # The elements-per-lobe law: six lobes for (m,n)=(6,3) need n_theta >= ~20
    # (>3 elements/lobe) before the numerical over-island collapses.
    colors = [
        "tab:blue" if int(item["ns"][1]) >= 20 else "0.55"
        for item in selected
    ]
    fig, axis = plt.subplots(figsize=(9.2, 5.0))
    for index in range(len(selected)):
        axis.errorbar(
            x[index],
            y[index],
            yerr=[[y[index] - q[index, 0]], [q[index, 1] - y[index]]],
            fmt="o",
            color=colors[index],
            capsize=4,
        )
    axis.scatter([], [], color="tab:blue", label=r"poloidally resolved ($n_\theta\geq20$)")
    axis.scatter([], [], color="0.55", label=r"under-resolved ($n_\theta<20$)")
    axis.set_xticks(x, [item["label"] for item in selected], rotation=55, ha="right")
    axis.set_ylabel(r"$w_{\mathrm{trapped}}$ (90% cluster-bootstrap CI)")
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUT / "robust_trapped_width_ci_vs_grid.png", dpi=180)
    plt.close(fig)


def _load_trace(path: Path, prefix: str = "line_") -> tuple[np.ndarray, list[np.ndarray]]:
    data = np.load(path)
    seed_key = "seeds" if "seeds" in data else "logical_seeds"
    keys = sorted(key for key in data.files if key.startswith(prefix))
    return np.asarray(data[seed_key]), [np.asarray(data[key]) for key in keys]


def _plot_self_convergence(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    target = next((item for item in records if item["label"] == "11x22x11"), None)
    if target is None:
        return None
    seeds, lines = _load_trace(Path(target["trace_cache_npz"]))
    rows: list[dict[str, Any]] = []
    all_rho = np.unique(seeds[:, 0])
    all_theta = np.unique(seeds[:, 1])
    for turns in (200, 500, 1000, 2000):
        for nrho, phases in ((8, 3), (12, 3), (16, 3), (16, 6)):
            rho_indices = np.linspace(0, len(all_rho) - 1, nrho).round().astype(int)
            theta_indices = np.linspace(
                0, len(all_theta) - 1, phases
            ).round().astype(int)
            rho_keep = all_rho[rho_indices]
            theta_keep = all_theta[theta_indices]
            keep = [
                index
                for index, seed in enumerate(seeds)
                if np.any(np.isclose(seed[0], rho_keep))
                and np.any(np.isclose(seed[1], theta_keep))
            ]
            sub_lines = [lines[index][:turns] for index in keep]
            sub_seeds = seeds[keep]
            summary = cmp._trapped_separatrix_summary(
                sub_lines, sub_seeds, poloidal_mode=6
            )
            rows.append(
                {
                    "turns": turns,
                    "nrho": nrho,
                    "phases": phases,
                    "width": summary["trapped_separatrix_width_rho"],
                }
            )
    fig, axis = plt.subplots(figsize=(8.0, 5.0))
    for nrho, phases in ((8, 3), (12, 3), (16, 3), (16, 6)):
        chosen = [
            item
            for item in rows
            if item["nrho"] == nrho and item["phases"] == phases
        ]
        axis.plot(
            [item["turns"] for item in chosen],
            [item["width"] for item in chosen],
            "o-",
            label=rf"${nrho}\rho\times{phases}\theta$",
        )
    axis.set_xlabel("Poincaré turns per field line")
    axis.set_ylabel(r"$w_{\mathrm{trapped}}$")
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUT / "robust_self_convergence.png", dpi=180)
    plt.close(fig)
    return {"grid": "11x22x11", "records": rows}


def _estimated_o_points(lines: list[np.ndarray]) -> np.ndarray:
    trapped = [
        np.asarray(line)
        for line in lines
        if cmp._classify_resonant_orbit(line, poloidal_mode=6)["trapped"]
    ]
    if not trapped:
        return np.empty((0, 2))
    points = np.concatenate(trapped)
    theta = points[:, 1] % 1.0
    lobe = np.floor(6.0 * theta).astype(int) % 6
    centers: list[list[float]] = []
    for index in range(6):
        selected = points[lobe == index]
        if selected.size:
            centers.append(
                [float(np.median(selected[:, 1])), float(np.median(selected[:, 0]))]
            )
    return np.asarray(centers)


def _plot_overlay() -> dict[str, Any] | None:
    corrected = OUT / "robust_simsopt_mrx_trace_11x22x11_phi0.npz"
    path = (
        corrected
        if corrected.is_file()
        else OUT / "robust_simsopt_mrx_trace_11x22x11.npz"
    )
    if not path.is_file():
        return None
    data = np.load(path)
    seeds = np.asarray(data["logical_seeds"])
    simsopt_keys = sorted(
        key for key in data.files if key.startswith("simsopt_line_")
    )
    mrx_keys = sorted(key for key in data.files if key.startswith("mrx_line_"))
    simsopt = [np.asarray(data[key]) for key in simsopt_keys]
    mrx = [np.asarray(data[key]) for key in mrx_keys]
    sim_centers = _estimated_o_points(simsopt)
    mrx_centers = _estimated_o_points(mrx)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharex=True, sharey=True)
    for axis, lines, centers, title in (
        (axes[0], simsopt, sim_centers, "SIMSOPT Biot–Savart"),
        (axes[1], mrx, mrx_centers, "MRX 11x22x11"),
    ):
        for line, seed in zip(lines, seeds):
            axis.scatter(
                line[:, 1] % 1.0,
                line[:, 0],
                s=0.25,
                alpha=0.25,
                color=plt.cm.viridis((seed[1] % 1.0)),
                rasterized=True,
            )
        if centers.size:
            axis.plot(
                centers[:, 0] % 1.0,
                centers[:, 1],
                "x",
                color="crimson",
                markersize=8,
                markeredgewidth=2,
                label="estimated O-points",
            )
        axis.set_title(title)
        axis.set_xlabel(r"$\theta$ [cycles]")
        axis.grid(True, alpha=0.2)
        axis.legend(fontsize=8)
    axes[0].set_ylabel(r"Logical radius $\rho$")
    axes[0].set_xlim(0.0, 1.0)
    axes[0].set_ylim(0.58, 0.82)
    fig.tight_layout()
    fig.savefig(OUT / "robust_oppoint_overlay.png", dpi=180)
    plt.close(fig)

    mismatch = None
    if sim_centers.shape == mrx_centers.shape and sim_centers.size:
        mismatch = float(
            np.sqrt(np.mean((sim_centers[:, 1] - mrx_centers[:, 1]) ** 2))
        )
    return {
        "simsopt_o_points": sim_centers.tolist(),
        "mrx_o_points": mrx_centers.tolist(),
        "radial_rms_mismatch": mismatch,
    }


def main() -> None:
    records = _load_records()
    if not records:
        raise RuntimeError("no robust grid JSON files found")
    fit = _fit_width_law(records)
    isotropic_l2 = _isotropic_l2_fit(records)
    _plot_field_l2(records)
    _plot_resonant_amplitude(records)
    _plot_width_law(records, fit)
    location = _plot_location(records)
    _plot_balanced_width(records)
    self_convergence = _plot_self_convergence(records)
    detrended = _plot_detrended_width(records)
    overlay = _plot_overlay()
    reliability_table = [item.get("reliability") for item in records]
    summary = {
        "record_count": len(records),
        "labels": [item["label"] for item in records],
        "reliability": reliability_table,
        "quarantined_labels": sorted(
            {
                item["label"]
                for item in records
                if (item.get("reliability") or {}).get("quarantined_nullspace")
            }
        ),
        "truncated_trace_labels": sorted(
            {
                item["label"]
                for item in records
                if (item.get("reliability") or {}).get("truncated_trace")
            }
        ),
        "width_law": fit,
        "isotropic_l2_fit": isotropic_l2,
        "location": location,
        "self_convergence": self_convergence,
        "detrended_width": detrended,
        "overlay": overlay,
    }
    (OUT / "robust_island_convergence_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(
        "ROBUST_AGGREGATE_COMPLETE "
        f"records={len(records)} "
        f"isotropic_p={isotropic_l2['order_p']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
