from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class RunArtifacts:
    path: Path
    config: dict
    benchmarks: list[dict]
    k0_laplacian_benchmarks: list[dict]


_AXIS_INFO = {
    "n": ("n", lambda config: int(config["ns"][0])),
    "p": ("p", lambda config: int(config["p"])),
    "kappa": ("kappa", lambda config: float(config["rotating_kappa"])),
    "eps": ("eps", lambda config: float(config["rotating_eps"])),
}

_FIXED_FIELDS = ("n", "p", "kappa", "eps")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot summaries for mass-preconditioner sweep artifact folders.")
    parser.add_argument(
        "artifact_dirs",
        nargs="*",
        help="Explicit mass_preconditioner_choices_* directories to aggregate.",
    )
    parser.add_argument(
        "--root",
        default="outputs/interactive",
        help="Root directory used when artifact directories are not given.",
    )
    parser.add_argument(
        "--latest",
        type=int,
        default=4,
        help="Number of most recent artifact directories to use when none are provided.",
    )
    parser.add_argument(
        "--axis",
        choices=tuple(_AXIS_INFO),
        help="Sweep axis to plot. When omitted, infer it from the selected runs.",
    )
    parser.add_argument(
        "--labels",
        help="Optional comma-separated list of benchmark labels to keep.",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for the generated figures. Defaults to <root>/sweep_plots/<axis>_<count>cases.",
    )
    return parser.parse_args()


def _load_json(path: Path):
    return json.loads(path.read_text())


def _discover_artifact_dirs(root: Path, latest: int) -> list[Path]:
    candidates = sorted(
        [path for path in root.glob("mass_preconditioner_choices_*") if path.is_dir()],
        key=lambda path: path.name,
    )
    if latest > 0:
        candidates = candidates[-latest:]
    return candidates


def _load_run(path: Path) -> RunArtifacts:
    metadata = _load_json(path / "metadata.json")
    benchmarks = _load_json(path / "benchmarks.json")
    laplacian_path = path / "k0_laplacian_benchmarks.json"
    k0_laplacian_benchmarks = _load_json(laplacian_path) if laplacian_path.exists() else []
    return RunArtifacts(
        path=path,
        config=metadata["experiment_config"],
        benchmarks=benchmarks,
        k0_laplacian_benchmarks=k0_laplacian_benchmarks,
    )


def _value_for_axis(run: RunArtifacts, axis: str) -> float:
    return _AXIS_INFO[axis][1](run.config)


def _infer_axis(runs: list[RunArtifacts]) -> str:
    varying_axes = []
    for axis in _AXIS_INFO:
        values = {_value_for_axis(run, axis) for run in runs}
        if len(values) > 1:
            varying_axes.append(axis)
    if len(varying_axes) != 1:
        raise ValueError(
            "Could not infer a unique sweep axis from the selected runs; "
            f"varying axes are {varying_axes or '<none>'}. Pass --axis explicitly."
        )
    return varying_axes[0]


def _select_labels(runs: list[RunArtifacts], requested: str | None) -> list[str]:
    labels = sorted({entry["label"] for run in runs for entry in run.benchmarks})
    if requested is None:
        return [label for label in labels if not label.startswith("richardson-")]
    requested_labels = [label.strip() for label in requested.split(",") if label.strip()]
    missing = [label for label in requested_labels if label not in labels]
    if missing:
        raise ValueError(f"Unknown labels requested: {missing}. Available labels: {labels}")
    return requested_labels


def _fixed_config_summary(runs: list[RunArtifacts], axis: str) -> str:
    sample = runs[0].config
    parts = []
    for field in _FIXED_FIELDS:
        if field == axis:
            continue
        if field == "n":
            parts.append(f"n={sample['ns'][0]}")
        elif field == "p":
            parts.append(f"p={sample['p']}")
        elif field == "kappa":
            parts.append(f"kappa={sample['rotating_kappa']}")
        elif field == "eps":
            parts.append(f"eps={sample['rotating_eps']}")
    return ", ".join(parts)


def _sort_runs(runs: list[RunArtifacts], axis: str) -> list[RunArtifacts]:
    return sorted(runs, key=lambda run: _value_for_axis(run, axis))


def _collect_mass_series(
        runs: list[RunArtifacts],
        axis: str,
        labels: list[str],
        metric: str) -> dict[int, dict[str, tuple[list[float], list[float]]]]:
    result: dict[int, dict[str, tuple[list[float], list[float]]]] = {
        k: {label: ([], []) for label in labels} for k in range(4)
    }
    for run in _sort_runs(runs, axis):
        x_value = _value_for_axis(run, axis)
        for entry in run.benchmarks:
            k = int(entry["k"])
            label = entry["label"]
            if label not in labels:
                continue
            xs, ys = result[k][label]
            xs.append(x_value)
            ys.append(float(entry[metric]))
    return result


def _collect_speedup_series(
        runs: list[RunArtifacts],
        axis: str,
        labels: list[str]) -> dict[int, dict[str, tuple[list[float], list[float]]]]:
    result: dict[int, dict[str, tuple[list[float], list[float]]]] = {
        k: {label: ([], []) for label in labels if label != "jacobi"} for k in range(4)
    }
    for run in _sort_runs(runs, axis):
        x_value = _value_for_axis(run, axis)
        entries_by_key = {(int(entry["k"]), entry["label"]): entry for entry in run.benchmarks}
        for k in range(4):
            jacobi = entries_by_key.get((k, "jacobi"))
            if jacobi is None:
                continue
            base_time = float(jacobi["avg_time_ms"])
            if base_time <= 0.0:
                continue
            for label in result[k]:
                entry = entries_by_key.get((k, label))
                if entry is None:
                    continue
                xs, ys = result[k][label]
                xs.append(x_value)
                ys.append(base_time / float(entry["avg_time_ms"]))
    return result


def _collect_k0_laplacian_series(
        runs: list[RunArtifacts],
        axis: str,
        metric: str) -> dict[str, tuple[list[float], list[float]]]:
    labels = sorted({entry["label"] for run in runs for entry in run.k0_laplacian_benchmarks})
    result = {label: ([], []) for label in labels}
    for run in _sort_runs(runs, axis):
        x_value = _value_for_axis(run, axis)
        for entry in run.k0_laplacian_benchmarks:
            xs, ys = result[entry["label"]]
            xs.append(x_value)
            ys.append(float(entry[metric]))
    return result


def _default_xscale(axis: str) -> str:
    return "log" if axis == "n" else "linear"


def _style_for_label(label: str) -> dict[str, object]:
    if label == "jacobi":
        return {
            "color": "0.15",
            "marker": "o",
            "s": 44,
            "alpha": 0.85,
        }
    if label.startswith("richardson-"):
        step = int(label.split("-")[-1])
        return {
            "color": "tab:blue",
            "marker": {1: "o", 2: "s", 4: "D"}.get(step, "o"),
            "s": 42,
            "alpha": 0.72,
        }
    if label.startswith("chebyshev-"):
        step = int(label.split("-")[-1])
        return {
            "color": "tab:orange",
            "marker": {1: "o", 2: "s", 4: "D"}.get(step, "o"),
            "s": 42,
            "alpha": 0.72,
        }
    if label.startswith("tensor-r"):
        rank = int(label.split("r")[-1])
        return {
            "color": "tab:green",
            "marker": {1: "o", 2: "s", 4: "D"}.get(rank, "o"),
            "s": 42,
            "alpha": 0.72,
        }
    return {"marker": "o", "s": 42, "alpha": 0.72}


def _x_offset_unit(x_values: list[float], xscale: str) -> float:
    unique = sorted({float(value) for value in x_values})
    if len(unique) <= 1:
        return 0.04
    if xscale == "log":
        logs = np.log10(np.asarray(unique, dtype=float))
        diffs = np.diff(logs)
        diffs = diffs[diffs > 0]
        if diffs.size == 0:
            return 0.04
        return 0.18 * float(np.min(diffs))
    diffs = np.diff(np.asarray(unique, dtype=float))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 0.04
    return 0.18 * float(np.min(diffs))


def _apply_horizontal_offset(x: float, offset: float, xscale: str) -> float:
    if offset == 0.0:
        return x
    if xscale == "log":
        return float(x * (10.0 ** offset))
    return float(x + offset)


def _set_axis_limits_from_series(
        ax: plt.Axes,
        series: dict[str, tuple[list[float], list[float]]],
        *,
        xscale: str,
        yscale: str) -> None:
    x_values = np.asarray([float(x) for xs, _ in series.values() for x in xs], dtype=float)
    y_values = np.asarray([float(y) for _, ys in series.values() for y in ys], dtype=float)
    if x_values.size == 0 or y_values.size == 0:
        return

    xmin = float(np.min(x_values))
    xmax = float(np.max(x_values))
    if xscale == "log":
        xmin = max(xmin, np.finfo(float).tiny)
        xmax = max(xmax, xmin * 1.01)
        log_min = np.log10(xmin)
        log_max = np.log10(xmax)
        if log_max <= log_min:
            log_max = log_min + 1.0
        pad = 0.088 * (log_max - log_min)
        ax.set_xlim(10.0 ** (log_min - pad), 10.0 ** (log_max + pad))
    else:
        if xmax <= xmin:
            xmax = xmin + 1.0
        pad = 0.088 * (xmax - xmin)
        ax.set_xlim(xmin - pad, xmax + pad)

    if yscale == "log":
        y_values = np.maximum(y_values, np.finfo(float).tiny)
        ymin = float(np.min(y_values))
        ymax = float(np.max(y_values))
        ymax = max(ymax, ymin * 1.01)
        log_min = np.log10(ymin)
        log_max = np.log10(ymax)
        if log_max <= log_min:
            log_max = log_min + 1.0
        pad = 0.055 * (log_max - log_min)
        ax.set_ylim(10.0 ** (log_min - pad), 10.0 ** (log_max + pad))
    else:
        ymin = float(np.min(y_values))
        ymax = float(np.max(y_values))
        if ymax <= ymin:
            ymax = ymin + 1.0
        pad = 0.055 * (ymax - ymin)
        ax.set_ylim(ymin - pad, ymax + pad)


def _cluster_same_y(
        points: list[tuple[str, int, float]],
        *,
        ax: plt.Axes,
        x_value: float,
        display_tol_px: float = 12.0) -> list[list[tuple[str, int, float]]]:
    ordered = sorted(points, key=lambda item: ax.transData.transform((x_value, item[2]))[1])
    clusters: list[list[tuple[str, int, float]]] = []
    for point in ordered:
        if not clusters:
            clusters.append([point])
            continue
        reference = ax.transData.transform((x_value, clusters[-1][-1][2]))[1]
        current = ax.transData.transform((x_value, point[2]))[1]
        if abs(current - reference) <= display_tol_px:
            clusters[-1].append(point)
        else:
            clusters.append([point])
    return clusters


def _offset_overlapping_series(
        series: dict[str, tuple[list[float], list[float]]],
        *,
    ax: plt.Axes,
    xscale: str,
    yscale: str) -> dict[str, tuple[list[float], list[float]]]:
    adjusted = {
        label: ([float(x) for x in xs], [float(y) for y in ys])
        for label, (xs, ys) in series.items()
    }
    x_values = [float(x) for xs, _ in series.values() for x in xs]
    if not x_values:
        return adjusted

    points_by_x: dict[float, list[tuple[str, int, float]]] = defaultdict(list)
    for label, (xs, ys) in adjusted.items():
        for index, (x, y) in enumerate(zip(xs, ys)):
            points_by_x[float(x)].append((label, index, float(y)))

    for x_value, points in points_by_x.items():
        for cluster in _cluster_same_y(points, ax=ax, x_value=x_value):
            if len(cluster) <= 1:
                continue
            center = 0.5 * (len(cluster) - 1)
            base_display_x = ax.transData.transform((x_value, cluster[0][2]))[0]
            for offset_index, (label, index, _) in enumerate(cluster):
                display_y = ax.transData.transform((x_value, adjusted[label][1][index]))[1]
                display_x = base_display_x + 10.0 * (offset_index - center)
                adjusted[label][0][index] = float(
                    ax.transData.inverted().transform((display_x, display_y))[0]
                )
    return adjusted


def _plot_grid(
        series: dict[int, dict[str, tuple[list[float], list[float]]]],
        *,
        axis: str,
        ylabel: str,
        title: str,
        output_path: Path,
        fixed_summary: str,
    xscale: str,
    yscale: str = "log") -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True, sharex=True)
    axes = axes.ravel()
    handles = []
    labels_seen = []
    for k, ax in enumerate(axes):
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        _set_axis_limits_from_series(ax, series[k], xscale=xscale, yscale=yscale)
        fig.canvas.draw()
        adjusted_series = _offset_overlapping_series(series[k], ax=ax, xscale=xscale, yscale=yscale)
        for label, (xs, ys) in adjusted_series.items():
            if not xs:
                continue
            scatter = ax.scatter(xs, ys, label=label, **_style_for_label(label))
            if label not in labels_seen:
                labels_seen.append(label)
                handles.append(scatter)
        ax.set_title(f"k={k}")
        ax.set_xlabel(axis)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"{title}\n{fixed_summary}")
    if handles:
        fig.legend(handles, labels_seen, loc="outside right center")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_k0_laplacian(
        runs: list[RunArtifacts],
        axis: str,
        output_path: Path,
        fixed_summary: str) -> None:
    iter_series = _collect_k0_laplacian_series(runs, axis, "avg_iters")
    time_series = _collect_k0_laplacian_series(runs, axis, "avg_time_ms")
    if not iter_series:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True, sharex=True)
    xscale = _default_xscale(axis)
    axes[0].set_xscale(xscale)
    axes[0].set_yscale("log")
    axes[1].set_xscale(xscale)
    axes[1].set_yscale("log")
    _set_axis_limits_from_series(axes[0], iter_series, xscale=xscale, yscale="log")
    _set_axis_limits_from_series(axes[1], time_series, xscale=xscale, yscale="log")
    fig.canvas.draw()
    legend_handles = []
    legend_labels = []
    iter_series = _offset_overlapping_series(iter_series, ax=axes[0], xscale=xscale, yscale="log")
    time_series = _offset_overlapping_series(time_series, ax=axes[1], xscale=xscale, yscale="log")
    for label, (xs, ys) in iter_series.items():
        scatter = axes[0].scatter(xs, ys, label=label, **_style_for_label(label))
        legend_handles.append(scatter)
        legend_labels.append(label)
    for label, (xs, ys) in time_series.items():
        axes[1].scatter(xs, ys, label=label, **_style_for_label(label))
    axes[0].set_title("k=0 Laplacian avg iterations")
    axes[1].set_title("k=0 Laplacian avg time [ms]")
    for ax in axes:
        ax.set_xlabel(axis)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("avg_iters")
    axes[1].set_ylabel("avg_time_ms")
    fig.suptitle(f"k=0 Laplacian Sweep Summary\n{fixed_summary}")
    fig.legend(
        legend_handles,
        legend_labels,
        loc="outside right center",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_summary(runs: list[RunArtifacts], axis: str, output_dir: Path) -> None:
    summary = {
        "axis": axis,
        "artifact_dirs": [str(run.path) for run in _sort_runs(runs, axis)],
        "axis_values": [_value_for_axis(run, axis) for run in _sort_runs(runs, axis)],
        "fixed_config": {
            "n": int(runs[0].config["ns"][0]),
            "p": int(runs[0].config["p"]),
            "kappa": float(runs[0].config["rotating_kappa"]),
            "eps": float(runs[0].config["rotating_eps"]),
        },
    }
    (output_dir / "sweep_summary.json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    args = _parse_args()
    if args.artifact_dirs:
        artifact_dirs = [Path(path) for path in args.artifact_dirs]
    else:
        artifact_dirs = _discover_artifact_dirs(Path(args.root), args.latest)
    if not artifact_dirs:
        raise ValueError("No artifact directories found to plot.")

    runs = [_load_run(path) for path in artifact_dirs]
    axis = args.axis or _infer_axis(runs)
    labels = _select_labels(runs, args.labels)
    fixed_summary = _fixed_config_summary(runs, axis)

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(args.root) / "sweep_plots" / f"{axis}_{len(runs)}cases"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    avg_iter_series = _collect_mass_series(runs, axis, labels, "avg_iters")
    avg_time_series = _collect_mass_series(runs, axis, labels, "avg_time_ms")
    speedup_series = _collect_speedup_series(runs, axis, labels)

    _plot_grid(
        avg_iter_series,
        axis=axis,
        ylabel="avg_iters",
        title="Mass Sweep: Average Iterations",
        output_path=output_dir / "mass_sweep_avg_iters.png",
        fixed_summary=fixed_summary,
        xscale=_default_xscale(axis),
    )
    _plot_grid(
        avg_time_series,
        axis=axis,
        ylabel="avg_time_ms",
        title="Mass Sweep: Average Solve Time",
        output_path=output_dir / "mass_sweep_avg_time_ms.png",
        fixed_summary=fixed_summary,
        xscale=_default_xscale(axis),
    )
    _plot_grid(
        speedup_series,
        axis=axis,
        ylabel="jacobi time / method time",
        title="Mass Sweep: Time Speedup Over Jacobi",
        output_path=output_dir / "mass_sweep_speedup.png",
        fixed_summary=fixed_summary,
        xscale=_default_xscale(axis),
    )
    _plot_k0_laplacian(
        runs,
        axis,
        output_path=output_dir / "k0_laplacian_sweep.png",
        fixed_summary=fixed_summary,
    )
    _write_summary(runs, axis, output_dir)

    print(f"loaded {len(runs)} artifact directories")
    print(f"axis: {axis}")
    print(f"labels: {labels}")
    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()