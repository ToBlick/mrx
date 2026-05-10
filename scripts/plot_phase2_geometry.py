"""Plot Phase 2 geometry-sweep results.

Accepts either:

* a Phase-2 run directory containing ``cell_*.csv`` files, or
* an aggregated CSV such as ``phase2_sweep.csv``.

The output is a single PDF with one column per benchmark block and one row
per selected swept axis. Each panel shows the locked Phase-2 competitors
against the axis slice obtained by holding all other parameters at their
baseline value.

Examples
--------
    python scripts/plot_phase2_geometry.py outputs/phase2/2026-05-09/12-02-10
    python scripts/plot_phase2_geometry.py outputs/phase2/.../phase2_sweep.csv

By default the script emits both the iteration-count and solve-time figures
in one run.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


STRATEGY_COLOR = {
    "jacobi": "black",
    "chebyshev": "tab:blue",
    "tensor": "tab:red",
    "tensor_inner_schur": "purple",
}

CHEB_LINESTYLE = {0: "-", 2: "--", 3: "-.", 5: ":"}
CASE_ORDER = {"M0": 0, "M1": 1, "M2": 2, "M3": 3, "K0_dbc": 4}
MARKER_CYCLE = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "*")

AXES = (
    ("ns", r"$n_s$"),
    ("p", r"$p$"),
    ("kappa", r"$\kappa$"),
)


def load_rows(path: Path) -> list[dict]:
    if path.is_dir():
        files = sorted(path.glob("cell_*.csv"))
        if not files:
            raise SystemExit(f"No cell_*.csv files found under {path}")
    elif path.is_file():
        files = [path]
    else:
        raise SystemExit(f"Input not found: {path}")

    rows: list[dict] = []
    for csv_path in files:
        with csv_path.open() as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def _f(row: dict, key: str, default: float = float("nan")) -> float:
    val = row.get(key, "")
    if val in ("", None):
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _i(row: dict, key: str, default: int = -1) -> int:
    val = row.get(key, "")
    if val in ("", None):
        return default
    try:
        return int(float(val))
    except ValueError:
        return default


def _ns(row: dict) -> tuple[int, int, int]:
    return (_i(row, "n_r"), _i(row, "n_t"), _i(row, "n_z"))


def _method_label(row: dict) -> str:
    return row.get("method", row.get("strategy", "unknown"))


def _case(row: dict) -> str:
    case = row.get("case", "")
    if case:
        return str(case)
    k = _i(row, "k")
    return f"M{k}" if k >= 0 else "unknown"


def _case_key(case: str) -> tuple[int, str]:
    return (CASE_ORDER.get(case, 99), case)


def _case_title(case: str) -> str:
    if case.startswith("M") and case[1:].isdigit():
        return f"M{case[1:]}"
    if case == "K0_dbc":
        return "K0 (dbc)"
    return case


def _linestyle(row: dict) -> str:
    strategy = row.get("strategy", "")
    if strategy == "jacobi":
        return "-"
    if strategy == "chebyshev":
        return CHEB_LINESTYLE.get(_i(row, "cheb_steps"), "--")
    if strategy in ("tensor", "tensor_inner_schur"):
        return CHEB_LINESTYLE.get(_i(row, "bulk_cheb_steps"), "-")
    return "-"


def _marker_map(rows: list[dict]) -> dict[str, str]:
    labels = sorted({_method_label(row) for row in rows})
    return {
        label: MARKER_CYCLE[idx % len(MARKER_CYCLE)]
        for idx, label in enumerate(labels)
    }


def _err(row: dict, metric: str) -> tuple[float, float] | None:
    if metric == "avg_iters":
        avg = _f(row, "avg_iters")
        lo = _f(row, "min_iters", default=avg)
        hi = _f(row, "max_iters", default=avg)
    else:
        avg = _f(row, "avg_solve_ms")
        lo = _f(row, "min_solve_ms", default=avg)
        hi = _f(row, "max_solve_ms", default=avg)
    if not (np.isfinite(avg) and np.isfinite(lo) and np.isfinite(hi)):
        return None
    return max(avg - lo, 0.0), max(hi - avg, 0.0)


def infer_baselines(rows: list[dict]) -> dict:
    def mode(values):
        return Counter(values).most_common(1)[0][0]

    return {
        "kappa": mode([_f(r, "kappa") for r in rows]),
        "eps": mode([_f(r, "eps") for r in rows]),
        "p": mode([_i(r, "p") for r in rows]),
        "ns": mode([_ns(r) for r in rows]),
        "nfp": mode([_i(r, "nfp") for r in rows]),
        "besov_s": mode([_f(r, "besov_s") for r in rows]),
    }


def _row_matches_axis_slice(row: dict, axis: str, baselines: dict) -> bool:
    values = {
        "kappa": _f(row, "kappa"),
        "eps": _f(row, "eps"),
        "p": _i(row, "p"),
        "ns": _ns(row),
        "nfp": _i(row, "nfp"),
        "besov_s": _f(row, "besov_s"),
    }
    for key, baseline in baselines.items():
        if key == axis:
            continue
        if values[key] != baseline:
            return False
    return True


def _x_value(row: dict, axis: str):
    if axis == "ns":
        return _ns(row)
    if axis == "p":
        return _i(row, "p")
    return _f(row, axis)


def _x_label(axis: str, value) -> str:
    if axis == "ns":
        return f"{value[0]},{value[1]},{value[2]}"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _panel_limits(rows: list[dict], metric: str, yscale: str) -> tuple[float, float]:
    ys: list[float] = []
    for row in rows:
        y = _f(row, metric)
        if np.isfinite(y):
            ys.append(y)
        err = _err(row, metric)
        if err is not None and np.isfinite(y):
            ys.extend([y - err[0], y + err[1]])
    ys = [y for y in ys if np.isfinite(y) and (y > 0 if yscale == "log" else True)]
    if not ys:
        return (1.0, 10.0) if yscale == "log" else (0.0, 1.0)
    ymin = min(ys)
    ymax = max(ys)
    if yscale == "log":
        return ymin / 1.3, ymax * 1.3
    pad = 0.05 * max(ymax - ymin, 1e-12)
    return ymin - pad, ymax + pad


def plot(rows: list[dict], out_pdf: Path, *, metric: str, yscale: str) -> None:
    baselines = infer_baselines(rows)
    cases = sorted({_case(r) for r in rows}, key=_case_key)
    if not cases:
        raise SystemExit("No benchmark cases found in input rows.")
    marker_by_label = _marker_map(rows)

    metric_label = "avg CG iterations" if metric == "avg_iters" else "avg solve time (ms)"
    fig, axes = plt.subplots(len(AXES), len(cases), figsize=(3.7 * len(cases), 2.8 * len(AXES)), squeeze=False)

    legend_handles: OrderedDict[str, plt.Line2D] = OrderedDict()

    for row_idx, (axis, axis_title) in enumerate(AXES):
        axis_rows = [r for r in rows if _row_matches_axis_slice(r, axis, baselines)]
        row_ymin, row_ymax = _panel_limits(axis_rows, metric, yscale)

        for col_idx, case in enumerate(cases):
            ax = axes[row_idx, col_idx]
            panel_rows = [
                r for r in axis_rows
                if _case(r) == case
            ]
            if not panel_rows:
                ax.set_visible(False)
                continue

            grouped: dict[str, list[dict]] = defaultdict(list)
            x_order = sorted({_x_value(r, axis) for r in panel_rows})
            x_pos = {x: i for i, x in enumerate(x_order)}

            for row in panel_rows:
                grouped[_method_label(row)].append(row)

            for label, grp in grouped.items():
                grp = sorted(grp, key=lambda r: _x_value(r, axis))
                strategy = grp[0].get("strategy", "")
                color = STRATEGY_COLOR.get(strategy, "0.4")
                linestyle = _linestyle(grp[0])
                marker = marker_by_label[label]

                if axis == "ns":
                    xs = np.array([x_pos[_x_value(r, axis)] for r in grp], dtype=float)
                else:
                    xs = np.array([_x_value(r, axis) for r in grp], dtype=float)
                ys = np.array([_f(r, metric) for r in grp], dtype=float)
                errs = [_err(r, metric) for r in grp]
                yerr = None
                if any(e is not None for e in errs):
                    lo = np.array([0.0 if e is None else e[0] for e in errs])
                    hi = np.array([0.0 if e is None else e[1] for e in errs])
                    yerr = np.vstack([lo, hi])

                handle = ax.errorbar(
                    xs, ys, yerr=yerr, color=color, linestyle=linestyle,
                    marker=marker, markersize=5.5,
                    capsize=2.5, elinewidth=0.8, linewidth=1.2,
                    label=label,
                )
                legend_handles.setdefault(label, handle.lines[0])

            if axis == "ns":
                ax.set_xticks(np.arange(len(x_order)))
                ax.set_xticklabels([_x_label(axis, x) for x in x_order], rotation=45, ha="right")

            if row_idx == 0:
                ax.set_title(_case_title(case))
            if col_idx == 0:
                ax.set_ylabel(f"{axis_title}\n{metric_label}")
            else:
                ax.tick_params(labelleft=False)

            ax.set_yscale(yscale)
            ax.set_ylim(row_ymin, row_ymax)
            ax.grid(True, which="both", linestyle=":", alpha=0.35)

    fig.legend(
        list(legend_handles.values()),
        list(legend_handles.keys()),
        loc="upper center",
        ncol=min(4, max(1, len(legend_handles))),
        frameon=False,
        fontsize=10,
        handlelength=2.8,
        columnspacing=1.4,
        handletextpad=0.6,
        borderaxespad=0.8,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=300)
    plt.close(fig)


def _metric_output(base_pdf: Path, metric_key: str) -> Path:
    if metric_key == "avg_iters":
        return base_pdf
    return base_pdf.with_name(base_pdf.stem + "_timings.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path", type=Path,
        help="Phase-2 run directory with cell_*.csv files or an aggregated CSV.",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Base output PDF path for the iteration figure (default: <dir>/phase2_sweep.pdf or <csv>.pdf).",
    )
    parser.add_argument(
        "--metric", choices=("both", "iters", "solve_ms"), default="both",
        help="Which metric(s) to plot (default: both).",
    )
    parser.add_argument(
        "--yscale", choices=("log", "linear"), default="log",
        help="Y-axis scale (default: log).",
    )
    args = parser.parse_args()

    path = args.path.resolve()
    rows = load_rows(path)
    if not rows:
        raise SystemExit(f"No rows found in {path}")

    if args.out is not None:
        base_pdf = args.out.resolve()
    elif path.is_dir():
        base_pdf = path / "phase2_sweep.pdf"
    else:
        base_pdf = path.with_suffix(".pdf")

    metric_keys = {
        "both": ("avg_iters", "avg_solve_ms"),
        "iters": ("avg_iters",),
        "solve_ms": ("avg_solve_ms",),
    }[args.metric]
    for metric in metric_keys:
        out_pdf = _metric_output(base_pdf, metric)
        plot(rows, out_pdf, metric=metric, yscale=args.yscale)
        print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()