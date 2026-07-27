#!/usr/bin/env python3
"""Attribute traced island widths to the resonant radial-pitch harmonic."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.minimal_vacuum_problem import compare_push_u_to_simsopt as cmp  # noqa: E402
from scripts.minimal_vacuum_problem.fem_convergence_robust import (  # noqa: E402
    normal_pitch_spectra as pitch,
)

MVP = ROOT / "scripts" / "minimal_vacuum_problem"
OUT = MVP / "fem_convergence_robust"
OUTPUT_JSON = OUT / "width_attribution.json"
OUTPUT_FIGURE = OUT / "width_attribution.png"
HEADLINE_FIGURE = OUT / "elements_per_lobe_convergence.png"
SIMSOPT_WIDTH_REF = 0.0501
SIMSOPT_WIDTH_BAND = (0.0486, 0.0503)
POLOIDAL_MODE = 6
# Grids whose traces or nullspaces are not trusted for the headline fit.
QUARANTINED_LABELS = {"13x18x13"}
TRUNCATED_TRACE_LABELS = {"12x24x12"}


def _load_robust_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in OUT.glob("robust_*x*x*.json"):
        if any(token in path.stem for token in ("smoke", "probe", "solve")):
            continue
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        dof_path = Path(str(record.get("dof_npy", "")))
        if (
            "width_law_predictor" not in record
            or "target_resonance" not in record
            or not dof_path.is_file()
        ):
            continue
        records.append(record)
    return sorted(records, key=lambda item: int(item["n2_dbc"]))


def _detrended_width(record: dict[str, Any]) -> float:
    return float(
        max(
            item["detrended_width_rho_q05_q95"]
            for item in record["island_width_profile"]
        )
    )


def _elements_per_lobe(record: dict[str, Any]) -> float:
    return float(int(record["ns"][1]) / POLOIDAL_MODE)


def _evaluate_record(
    record: dict[str, Any],
    simsopt_field: Any,
    hcurl_mod: Any,
    simsopt_shear: float,
) -> dict[str, Any]:
    ns = tuple(int(value) for value in record["ns"])
    dof_path = Path(record["dof_npy"])
    meta = cmp._load_meta_for_dof(
        dof_path,
        k=2,
        meta_json=MVP / "hodge_k2_nullspace_meta.json",
    )
    meta = dict(meta)
    meta["ns"] = list(ns)
    seq, map_fn, _, _, _ = cmp._rebuild_sequence_from_meta(
        meta,
        hcurl_mod=hcurl_mod,
        pushforward_only=True,
    )
    dof = jnp.asarray(np.load(dof_path), dtype=jnp.float64).reshape(-1)
    rho = float(record["target_resonance"]["rho"])
    simsopt_coefficients, mrx_coefficients = pitch._evaluate_grid(
        seq, map_fn, dof, simsopt_field, rho
    )
    simsopt_complex = simsopt_coefficients[(6, 1)]
    mrx_complex = mrx_coefficients[(6, 1)]
    traced = _detrended_width(record)
    amplitude_ratio = float(
        abs(mrx_complex) / max(abs(simsopt_complex), 1.0e-30)
    )
    return {
        "label": record["label"],
        "ns": list(ns),
        "n2_dbc": int(record["n2_dbc"]),
        "dof_npy": str(dof_path),
        "rho_resonance": rho,
        "elements_per_lobe": _elements_per_lobe(record),
        "local_shear_diota_drho": float(simsopt_shear),
        "local_shear_source": "simsopt_pitch_average",
        "record_local_shear_diota_drho": float(
            record["local_shear_diota_drho"]
        ),
        "aligned_rel_l2": float(record["aligned_metrics"]["rel_l2_aligned"]),
        "traced_detrended_width": traced,
        "width_ratio_mrx_over_simsopt": float(traced / SIMSOPT_WIDTH_REF),
        "sqrt_amplitude_ratio": float(np.sqrt(amplitude_ratio)),
        "reliability": _reliability(record),
        "simsopt_pitch_complex": [
            float(simsopt_complex.real),
            float(simsopt_complex.imag),
        ],
        "simsopt_pitch_amplitude": float(abs(simsopt_complex)),
        "simsopt_pitch_phase_radians": float(np.angle(simsopt_complex)),
        "mrx_pitch_complex": [float(mrx_complex.real), float(mrx_complex.imag)],
        "mrx_pitch_amplitude": float(abs(mrx_complex)),
        "mrx_pitch_phase_radians": float(np.angle(mrx_complex)),
        "complex_pitch_error_amplitude": float(
            abs(mrx_complex - simsopt_complex)
        ),
        "amplitude_ratio_mrx_over_simsopt": amplitude_ratio,
        "predicted_width_mrx": cmp.pendulum_island_width(
            abs(mrx_complex), simsopt_shear
        ),
        "predicted_width_simsopt": cmp.pendulum_island_width(
            abs(simsopt_complex), simsopt_shear
        ),
    }


def _reliability(record: dict[str, Any]) -> dict[str, Any]:
    label = str(record["label"])
    turns = int(record.get("tracer", {}).get("turns", 0) or 0)
    return {
        "quarantined": label in QUARANTINED_LABELS,
        "truncated_trace": label in TRUNCATED_TRACE_LABELS or turns < 2000,
        "turns": turns,
    }


def _summary(records: list[dict[str, Any]]) -> dict[str, float]:
    usable = [
        item
        for item in records
        if not item["reliability"]["quarantined"]
        and not item["reliability"]["truncated_trace"]
        and float(item["elements_per_lobe"]) >= 3.0
    ]
    if len(usable) < 3:
        usable = [
            item
            for item in records
            if not item["reliability"]["quarantined"]
        ]
    predicted = np.asarray(
        [item["predicted_width_mrx"] for item in usable], dtype=np.float64
    )
    traced = np.asarray(
        [item["traced_detrended_width"] for item in usable], dtype=np.float64
    )
    width_ratio = np.asarray(
        [item["width_ratio_mrx_over_simsopt"] for item in usable],
        dtype=np.float64,
    )
    sqrt_amp = np.asarray(
        [item["sqrt_amplitude_ratio"] for item in usable], dtype=np.float64
    )
    correlation = float(np.corrcoef(predicted, traced)[0, 1])
    relative_law_rms = float(np.sqrt(np.mean((width_ratio - sqrt_amp) ** 2)))
    scale = float(np.dot(predicted, traced) / np.dot(predicted, predicted))
    residual = traced - scale * predicted
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((traced - np.mean(traced)) ** 2))
    return {
        "usable_labels": [item["label"] for item in usable],
        "pearson_correlation_predicted_vs_traced": correlation,
        "best_fit_scale_through_origin": scale,
        "scaled_prediction_r_squared": float(1.0 - ss_res / ss_tot),
        "scaled_prediction_rms_error": float(np.sqrt(np.mean(residual**2))),
        "relative_law_rms_error": relative_law_rms,
        "simsopt_width_reference": SIMSOPT_WIDTH_REF,
    }


def _plot(records: list[dict[str, Any]], summary: dict[str, float]) -> None:
    records = sorted(records, key=lambda item: int(item["n2_dbc"]))
    labels = [str(item["label"]) for item in records]
    x = np.arange(len(records))
    ntheta = np.asarray([int(item["ns"][1]) for item in records])
    colors = np.where(ntheta >= 20, "tab:blue", "0.55")
    amplitude = np.asarray([item["mrx_pitch_amplitude"] for item in records])
    reference = np.asarray(
        [item["simsopt_pitch_amplitude"] for item in records]
    )
    predicted = np.asarray([item["predicted_width_mrx"] for item in records])
    traced = np.asarray([item["traced_detrended_width"] for item in records])
    phases = np.asarray([item["mrx_pitch_phase_radians"] for item in records])
    reference_phase = np.asarray(
        [item["simsopt_pitch_phase_radians"] for item in records]
    )
    phase_error = np.angle(np.exp(1.0j * (phases - reference_phase)))

    figure, axes = plt.subplots(1, 3, figsize=(16.0, 5.2))
    axes[0].scatter(x, amplitude, c=colors, s=45, zorder=3)
    axes[0].plot(x, amplitude, color="0.75", linewidth=0.8)
    axes[0].plot(
        x,
        reference,
        "--",
        color="tab:green",
        label="SIMSOPT at each resonant radius",
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$|\widehat{B^\rho/B^\zeta}_{6,1}|$")
    axes[0].set_title("Resonant radial-pitch amplitude")

    upper = 1.08 * max(float(np.max(predicted)), float(np.max(traced)))
    axes[1].plot([0.0, upper], [0.0, upper], "k--", label="prediction = trace")
    axes[1].scatter(predicted, traced, c=colors, s=48)
    for item, xp, yp in zip(records, predicted, traced):
        axes[1].annotate(
            item["label"],
            (xp, yp),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
        )
    axes[1].axhspan(
        *SIMSOPT_WIDTH_BAND,
        color="tab:green",
        alpha=0.16,
        label="SIMSOPT traced-width band",
    )
    axes[1].set_xlim(0.0, upper)
    axes[1].set_ylim(0.0, upper)
    axes[1].set_xlabel(
        r"Pendulum-law predicted width $4\sqrt{|c|/(2\pi|\iota'|)}$"
    )
    axes[1].set_ylabel("Traced detrended width")
    axes[1].set_title(
        "Width attribution "
        rf"($r={summary['pearson_correlation_predicted_vs_traced']:.2f}$)"
    )
    axes[1].legend(fontsize=8)

    axes[2].scatter(x, phase_error, c=colors, s=45, zorder=3)
    axes[2].axhline(0.0, color="tab:green", linestyle="--")
    axes[2].set_ylabel("MRX − SIMSOPT phase [rad]")
    axes[2].set_title("Resonant-harmonic phase error")

    for axis in (axes[0], axes[2]):
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=65, ha="right", fontsize=7)
        axis.set_xlabel("FEM grid")
        axis.grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(OUTPUT_FIGURE, dpi=190, bbox_inches="tight")
    plt.close(figure)


def _plot_elements_per_lobe(records: list[dict[str, Any]]) -> None:
    """Headline figure: island fidelity versus poloidal elements per lobe."""
    ordered = sorted(records, key=lambda item: _elements_per_lobe(item))
    x = np.asarray([_elements_per_lobe(item) for item in ordered])
    amp_ratio = np.asarray(
        [item["amplitude_ratio_mrx_over_simsopt"] for item in ordered]
    )
    width_ratio = np.asarray(
        [item["width_ratio_mrx_over_simsopt"] for item in ordered]
    )
    sqrt_amp = np.sqrt(np.maximum(amp_ratio, 0.0))

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharex=True)
    for item, xp, ar, wr in zip(ordered, x, amp_ratio, width_ratio):
        reliability = item["reliability"]
        if reliability["quarantined"]:
            marker, color, zorder = "x", "0.45", 2
        elif reliability["truncated_trace"]:
            marker, color, zorder = "D", "tab:orange", 3
        else:
            marker, color, zorder = "o", "tab:blue", 4
        axes[0].plot(
            xp, ar, marker=marker, color=color, markersize=8, zorder=zorder
        )
        axes[1].plot(
            xp, wr, marker=marker, color=color, markersize=8, zorder=zorder
        )
        for axis, yp in ((axes[0], ar), (axes[1], wr)):
            axis.annotate(
                item["label"],
                (xp, yp),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
            )

    axes[0].axhline(1.0, color="tab:green", linestyle="--", label="unity")
    axes[0].set_ylabel(r"$A_{\mathrm{MRX}}/A_{\mathrm{SIMSOPT}}$")
    axes[0].set_title(r"Resonant pitch amplitude $(6,1)$")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8)

    axes[1].plot(
        x,
        sqrt_amp,
        "s--",
        color="0.35",
        linewidth=1.0,
        markersize=5,
        label=r"$\sqrt{A_{\mathrm{MRX}}/A_{\mathrm{SIMSOPT}}}$",
        zorder=1,
    )
    axes[1].axhline(1.0, color="tab:green", linestyle="--", label="unity")
    axes[1].set_ylabel(r"$w_{\mathrm{MRX}}/w_{\mathrm{SIMSOPT}}$")
    axes[1].set_title("Traced detrended width vs relative pendulum law")
    axes[1].legend(fontsize=8)

    for axis in axes:
        axis.set_xlabel(r"Poloidal elements per island lobe $n_\theta/6$")
        axis.grid(True, alpha=0.25)
        axis.set_xlim(max(1.0, float(np.min(x)) - 0.3), float(np.max(x)) + 0.4)

    # Legend proxies for reliability classes.
    axes[0].plot([], [], "o", color="tab:blue", label="trusted")
    axes[0].plot([], [], "D", color="tab:orange", label="truncated trace")
    axes[0].plot([], [], "x", color="0.45", label="quarantined nullspace")
    axes[0].legend(fontsize=7, loc="best")

    figure.tight_layout()
    figure.savefig(HEADLINE_FIGURE, dpi=190, bbox_inches="tight")
    plt.close(figure)


def _simsopt_reference_shear(
    map_fn: Any,
    simsopt_field: Any,
    resonance_rho: float,
) -> float:
    profile = cmp._simsopt_iota_profile_from_pitch(
        map_fn,
        simsopt_field,
        np.linspace(0.58, 0.82, 33),
        nfp=3,
        ntheta=32,
        nzeta=32,
    )
    return cmp._local_shear_from_iota_profile(profile, float(resonance_rho))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels",
        nargs="*",
        help="Optional subset; completed records in the output cache are reused.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Recompute all requested labels even if cached.",
    )
    args = parser.parse_args()
    robust = _load_robust_records()
    requested = set(args.labels or [item["label"] for item in robust])
    cached: dict[str, dict[str, Any]] = {}
    if OUTPUT_JSON.is_file():
        previous = json.loads(OUTPUT_JSON.read_text())
        cached = {
            item["label"]: item
            for item in previous.get("records", [])
            if "elements_per_lobe" in item and "reliability" in item
        }
        if args.refresh:
            for label in requested:
                cached.pop(label, None)
    simsopt_field, _ = cmp._load_simsopt_field(MVP / "serial0044970.json")
    hcurl_mod = cmp._load_hcurl_module()

    # One shared SIMSOPT shear reference near the island (uses any available map).
    reference = next(iter(robust), None)
    if reference is None:
        raise RuntimeError("no robust records available")
    meta = cmp._load_meta_for_dof(
        Path(reference["dof_npy"]),
        k=2,
        meta_json=MVP / "hodge_k2_nullspace_meta.json",
    )
    meta = dict(meta)
    meta["ns"] = list(reference["ns"])
    _, map_fn, _, _, _ = cmp._rebuild_sequence_from_meta(
        meta,
        hcurl_mod=hcurl_mod,
        pushforward_only=True,
    )
    simsopt_shear = _simsopt_reference_shear(
        map_fn,
        simsopt_field,
        float(reference["target_resonance"]["rho"]),
    )
    print(f"SIMSOPT_REFERENCE_SHEAR {simsopt_shear:.6e}", flush=True)

    for record in robust:
        label = str(record["label"])
        if label not in requested:
            continue
        if label in cached and not args.refresh:
            # Refresh only the reliability flags / ratios that do not need FEM.
            cached[label]["reliability"] = _reliability(record)
            cached[label]["elements_per_lobe"] = _elements_per_lobe(record)
            cached[label]["width_ratio_mrx_over_simsopt"] = float(
                cached[label]["traced_detrended_width"] / SIMSOPT_WIDTH_REF
            )
            cached[label]["sqrt_amplitude_ratio"] = float(
                np.sqrt(cached[label]["amplitude_ratio_mrx_over_simsopt"])
            )
            continue
        print(f"WIDTH_ATTRIBUTION {label}", flush=True)
        cached[label] = _evaluate_record(
            record, simsopt_field, hcurl_mod, simsopt_shear
        )
        completed = sorted(cached.values(), key=lambda item: int(item["n2_dbc"]))
        OUTPUT_JSON.write_text(
            json.dumps({"records": completed}, indent=2) + "\n"
        )
    completed = [
        cached[item["label"]]
        for item in robust
        if item["label"] in cached
    ]
    if not completed:
        raise RuntimeError("no width-attribution records were evaluated")
    summary = _summary(completed)
    output = {
        "model": (
            "W = 4 sqrt(|pitch_6,1| / (2*pi*|diota/drho|)); "
            "relative form w_MRX/w_SIMSOPT = sqrt(A_MRX/A_SIMSOPT)"
        ),
        "model_note": (
            "Using m=6 instead of 2*pi is accidentally correct to ~2.3% "
            "because the cycle/radian, A=2|c|, and field-period index factors "
            "nearly cancel."
        ),
        "simsopt_traced_width_band": list(SIMSOPT_WIDTH_BAND),
        "simsopt_width_reference": SIMSOPT_WIDTH_REF,
        "simsopt_reference_shear": float(simsopt_shear),
        "summary": summary,
        "records": completed,
        "figure": str(OUTPUT_FIGURE),
        "headline_figure": str(HEADLINE_FIGURE),
    }
    OUTPUT_JSON.write_text(json.dumps(output, indent=2) + "\n")
    _plot(completed, summary)
    _plot_elements_per_lobe(completed)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
