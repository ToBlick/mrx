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
from matplotlib.ticker import NullLocator


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

BENCHMARK_KEY_FIELDS = (
    "kappa",
    "eps",
    "p",
    "n_r",
    "n_t",
    "n_z",
    "nfp",
    "besov_s",
    "case",
    "k",
    "strategy",
    "cheb_steps",
    "rank",
    "bulk_cheb_steps",
    "inner_schur",
    "dirichlet",
)


def _benchmark_key(row: dict) -> tuple[str, ...]:
    return tuple(str(row.get(field, "")) for field in BENCHMARK_KEY_FIELDS)


def _deduplicate_rows(rows: list[dict]) -> tuple[list[dict], int]:
    deduped: dict[tuple[str, ...], dict] = {}
    for row in rows:
        deduped[_benchmark_key(row)] = row
    return list(deduped.values()), len(rows) - len(deduped)


def _drop_nonconverged_rows(rows: list[dict]) -> tuple[list[dict], int]:
    kept = [row for row in rows if _i(row, "max_iters") < 1000]
    return kept, len(rows) - len(kept)


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
    rows, dropped = _deduplicate_rows(rows)
    if dropped:
        print(f"Dropped {dropped} duplicate benchmark rows from {path}")
    rows, omitted = _drop_nonconverged_rows(rows)
    if omitted:
        print(f"Omitted {omitted} non-converged benchmark rows from {path}")
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


def _reference_axis_value(axis: str, value) -> float:
    if axis == "ns":
        return float(value[0])
    return float(value)


def _triangle_anchor(axis: str, grouped: dict[str, list[dict]], metric: str):
    candidates: list[tuple[float, float]] = []
    fallback: list[tuple[float, float]] = []
    for grp in grouped.values():
        if not grp:
            continue
        grp_sorted = sorted(grp, key=lambda r: _x_value(r, axis))
        last = grp_sorted[-1]
        point = (_reference_axis_value(
            axis, _x_value(last, axis)), _f(last, metric))
        fallback.append(point)
        if last.get("strategy", "") in ("tensor", "tensor_inner_schur"):
            candidates.append(point)
    points = candidates or fallback
    if not points:
        return None
    points = [point for point in points if np.isfinite(
        point[0]) and np.isfinite(point[1]) and point[1] > 0.0]
    if not points:
        return None
    return max(points, key=lambda point: point[0])


def _draw_ns_growth_triangle(ax, anchor: tuple[float, float], x_order: list, y_limits: tuple[float, float], yscale: str) -> None:
    if len(x_order) < 2 or yscale != "log":
        return

    x_scale = np.array([_reference_axis_value("ns", value)
                       for value in x_order], dtype=float)
    ymin, ymax = y_limits
    if np.any(~np.isfinite(x_scale)) or np.any(x_scale <= 0.0):
        return
    if not (np.isfinite(ymin) and np.isfinite(ymax) and ymin > 0.0 and ymax > ymin):
        return

    x_anchor, y_anchor = anchor
    if not (np.isfinite(x_anchor) and np.isfinite(y_anchor) and x_anchor > 0.0 and y_anchor > 0.0):
        return

    log_xmin = np.log(x_scale.min())
    log_ymin = np.log(ymin)
    log_xanchor = np.log(x_anchor)

    dx = 0.18 * (np.log(x_scale.max()) - log_xmin)
    x1 = x_anchor
    x0 = np.exp(log_xanchor - dx)
    y1 = y_anchor / 1.80
    y0 = y1 / (x1 / x0) ** 3.0
    if x0 <= x_scale.min() or y0 <= ymin or y1 >= ymax:
        dx = 0.12 * (np.log(x_scale.max()) - log_xmin)
        x0 = np.exp(log_xanchor - dx)
        y1 = y_anchor / 1.60
        y0 = y1 / (x1 / x0) ** 3.0
        if x0 <= x_scale.min() or y0 <= ymin or y1 >= ymax:
            return

    style = {
        "linestyle": "-",
        "linewidth": 1.0,
        "color": "0.25",
        "alpha": 0.9,
        "zorder": 0,
        "clip_on": False,
    }
    ax.plot([x0, x1], [y0, y0], **style)
    ax.plot([x1, x1], [y0, y1], **style)
    ax.plot([x0, x1], [y0, y1], **style)
    x_text = np.sqrt(x0 * x1)
    y_text = y0 / 1.60
    ax.text(
        x_text,
        y_text,
        r"$n^3$",
        ha="center",
        va="top",
        fontsize=9,
        color="0.15",
        clip_on=False,
    )


def _draw_p_growth_triangle(ax, anchor: tuple[float, float], x_order: list, y_limits: tuple[float, float], yscale: str) -> None:
    if len(x_order) < 2 or yscale != "log":
        return

    x_scale = np.array([_reference_axis_value("p", value)
                       for value in x_order], dtype=float)
    ymin, ymax = y_limits
    if np.any(~np.isfinite(x_scale)):
        return
    if not (np.isfinite(ymin) and np.isfinite(ymax) and ymin > 0.0 and ymax > ymin):
        return

    xmin = float(np.min(x_scale))
    if float(np.max(x_scale)) <= xmin:
        return
    x_anchor, y_anchor = anchor
    if not (np.isfinite(x_anchor) and np.isfinite(y_anchor) and y_anchor > 0.0):
        return

    x1 = x_anchor
    x0 = x_anchor - 0.22 * (float(np.max(x_scale)) - xmin)
    if x0 <= xmin:
        x0 = x_anchor - 0.14 * (float(np.max(x_scale)) - xmin)
    if x0 <= xmin:
        return
    x_ref = 0.5 * (x0 + x1)
    y1 = y_anchor / 1.80
    y0 = y1 / np.exp((3.0 / x_ref) * (x1 - x0))
    if y0 <= ymin or y1 >= ymax:
        y1 = y_anchor / 1.60
        y0 = y1 / np.exp((3.0 / x_ref) * (x1 - x0))
        if y0 <= ymin or y1 >= ymax:
            return

    style = {
        "linestyle": "-",
        "linewidth": 1.0,
        "color": "0.25",
        "alpha": 0.9,
        "zorder": 0,
        "clip_on": False,
    }
    ax.plot([x0, x1], [y0, y0], **style)
    ax.plot([x1, x1], [y0, y1], **style)
    ax.plot([x0, x1], [y0, y1], **style)
    x_text = 0.5 * (x0 + x1)
    y_text = y0 / 1.60
    ax.text(
        x_text,
        y_text,
        r"$p^3$",
        ha="center",
        va="top",
        fontsize=9,
        color="0.15",
        clip_on=False,
    )


def _panel_limits(rows: list[dict], metric: str, yscale: str) -> tuple[float, float]:
    ys: list[float] = []
    for row in rows:
        y = _f(row, metric)
        if np.isfinite(y):
            ys.append(y)
        err = _err(row, metric)
        if err is not None and np.isfinite(y):
            ys.extend([y - err[0], y + err[1]])
    ys = [y for y in ys if np.isfinite(y) and (
        y > 0 if yscale == "log" else True)]
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
    fig, axes = plt.subplots(len(AXES), len(cases), figsize=(
        3.7 * len(cases), 2.8 * len(AXES)), squeeze=False)

    legend_handles: OrderedDict[str, plt.Line2D] = OrderedDict()

    for row_idx, (axis, axis_title) in enumerate(AXES):
        axis_rows = [r for r in rows if _row_matches_axis_slice(
            r, axis, baselines)]
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
            for row in panel_rows:
                grouped[_method_label(row)].append(row)

            triangle_anchor = _triangle_anchor(
                axis, grouped, metric) if metric == "avg_solve_ms" else None

            if metric == "avg_solve_ms" and axis == "ns" and triangle_anchor is not None:
                _draw_ns_growth_triangle(
                    ax, triangle_anchor, x_order, (row_ymin, row_ymax), yscale)
            if metric == "avg_solve_ms" and axis == "p" and triangle_anchor is not None:
                _draw_p_growth_triangle(
                    ax, triangle_anchor, x_order, (row_ymin, row_ymax), yscale)

            for label, grp in grouped.items():
                grp = sorted(grp, key=lambda r: _x_value(r, axis))
                strategy = grp[0].get("strategy", "")
                color = STRATEGY_COLOR.get(strategy, "0.4")
                linestyle = _linestyle(grp[0])
                marker = marker_by_label[label]

                if axis == "ns":
                    xs = np.array([_reference_axis_value(
                        axis, _x_value(r, axis)) for r in grp], dtype=float)
                else:
                    xs = np.array([_x_value(r, axis)
                                  for r in grp], dtype=float)
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
                ax.set_xscale("log")
                ax.set_xticks([_reference_axis_value(axis, x)
                              for x in x_order])
                ax.xaxis.set_minor_locator(NullLocator())
                ax.set_xticklabels([_x_label(axis, x)
                                   for x in x_order], rotation=45, ha="right")

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
