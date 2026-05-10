"""Plot Phase 1 mass-block benchmark CSV.

Renders two artifacts that the caller can compose in LaTeX:

* ``<out>.pdf``       — minimal "lines-only" figure: data lines + error
  bars, but no axis labels, no titles, no tick labels and no legend.
  Use as ``\\includegraphics`` foreground in the paper.
* ``<out>_annot.eps`` — same axes layout with no data, only chrome
  (axis labels, titles, ticks, legend). Edit and overlay in LaTeX.

Layout: 1 x len(ks) panels (one per k). All panels share the y-axis.

Style conventions:

* color    = strategy: jacobi=black, chebyshev=blue, tensor=red.
* linestyle (number of inner Chebyshev applications):
    - tensor   ``bulk_cheb_steps``: 0 -> solid, 2 -> dashed,
      3 -> dash-dot, 5 -> dotted.
    - chebyshev ``cheb_steps``: 2 -> dashed, 3 -> dash-dot, 5 -> dotted.
    - jacobi: solid horizontal.
* tensor curves: x = ``rank``; baselines: horizontal lines.

Error bars: from ``min_iters/max_iters`` (or ``min_solve_ms/max_solve_ms``)
when available; falls back to no bars on legacy CSVs.

By default the script emits both the iteration-count and solve-time figure
pairs in one run.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


STRATEGY_COLOR = {
    "jacobi": "black",
    "chebyshev": "tab:blue",
    "tensor": "tab:red",
    "tensor_inner_schur": "purple",
}

# Number of Chebyshev applications -> linestyle (per request).
CHEB_LINESTYLE = {0: "-", 2: "--", 3: "-.", 5: ":"}
CASE_ORDER = {"M0": 0, "M1": 1, "M2": 2, "M3": 3, "K0_dbc": 4}


def load_rows(csv_path: Path) -> list[dict]:
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        return list(reader)


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


def _b(row: dict, key: str, default: bool = False) -> bool:
    val = row.get(key, "")
    if val in ("", None):
        return default
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("1", "true", "yes", "on")


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


def _err(row: dict, metric: str) -> tuple[float, float] | None:
    """Return (lower, upper) error magnitudes around the mean for *metric*."""
    if metric == "avg_iters":
        avg = _f(row, "avg_iters")
        lo = _f(row, "min_iters", default=avg)
        hi = _f(row, "max_iters", default=avg)
    else:  # avg_solve_ms
        avg = _f(row, "avg_solve_ms")
        lo = _f(row, "min_solve_ms", default=avg)
        hi = _f(row, "max_solve_ms", default=avg)
    if not (np.isfinite(avg) and np.isfinite(lo) and np.isfinite(hi)):
        return None
    if hi <= avg and lo >= avg:
        return None
    return max(avg - lo, 0.0), max(hi - avg, 0.0)


def _draw_data_panel(ax, k_rows: list[dict], metric: str) -> None:
    jac = [r for r in k_rows if r["strategy"] == "jacobi"]
    cheb = [r for r in k_rows if r["strategy"] == "chebyshev"]
    ten = [r for r in k_rows if r["strategy"] in ("tensor", "tensor_inner_schur")]

    # Tensor curves: group by (variant, bulk_cheb_steps).
    ten_by_variant_bcheb: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in ten:
        ten_by_variant_bcheb[(r["strategy"], _i(r, "bulk_cheb_steps"))].append(r)

    for strategy, b in sorted(ten_by_variant_bcheb):
        grp = sorted(ten_by_variant_bcheb[(strategy, b)], key=lambda r: _i(r, "rank"))
        xs = np.array([_i(r, "rank") for r in grp], dtype=float)
        ys = np.array([_f(r, metric) for r in grp])
        errs = [_err(r, metric) for r in grp]
        ls = CHEB_LINESTYLE.get(b, "-")
        color = STRATEGY_COLOR.get(strategy, "tab:red")
        if any(e is not None for e in errs):
            lo = np.array([0.0 if e is None else e[0] for e in errs])
            hi = np.array([0.0 if e is None else e[1] for e in errs])
            ax.errorbar(xs, ys, yerr=np.vstack([lo, hi]),
                        color=color, linestyle=ls, marker="o",
                        capsize=2.5, elinewidth=0.8)
        else:
            ax.plot(xs, ys, color=color, linestyle=ls, marker="o")

    # Chebyshev baseline: horizontal lines per cheb_steps.
    color = STRATEGY_COLOR["chebyshev"]
    for r in sorted(cheb, key=lambda r: _i(r, "cheb_steps")):
        s = _i(r, "cheb_steps")
        y = _f(r, metric)
        ls = CHEB_LINESTYLE.get(s, "--")
        ax.axhline(y, color=color, linestyle=ls, linewidth=1.2)
        e = _err(r, metric)
        if e is not None:
            ax.axhspan(y - e[0], y + e[1], color=color, alpha=0.08, linewidth=0)

    # Jacobi baseline: solid horizontal.
    color = STRATEGY_COLOR["jacobi"]
    for r in jac:
        y = _f(r, metric)
        ax.axhline(y, color=color, linestyle="-", linewidth=1.2)
        e = _err(r, metric)
        if e is not None:
            ax.axhspan(y - e[0], y + e[1], color=color, alpha=0.08, linewidth=0)


def _bare_panel(ax) -> None:
    """Strip annotations from a panel for the data-only PDF."""
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(labelbottom=False, labelleft=False, length=0)
    ax.grid(False)


def _annotate_panel(ax, k: int, col: int, metric_label: str,
                    rank_ticks: list[int]) -> None:
    ax.set_title(f"k = {k}")
    ax.set_xlabel("tensor rank")
    if col == 0:
        ax.set_ylabel(metric_label)
    if rank_ticks:
        ax.set_xticks(rank_ticks)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)


def _build_legend_handles(tensor_keys: list[tuple[str, int]],
                          cheb_steps_present: list[int]):
    handles = [plt.Line2D([], [], color=STRATEGY_COLOR["jacobi"],
                          linestyle="-", label="Jacobi")]
    for s in cheb_steps_present:
        ls = CHEB_LINESTYLE.get(s, "--")
        handles.append(plt.Line2D(
            [], [], color=STRATEGY_COLOR["chebyshev"],
            linestyle=ls, label=fr"Chebyshev$_J(s={s})$",
        ))
    for strategy, b in tensor_keys:
        ls = CHEB_LINESTYLE.get(b, "-")
        label = (
            fr"tensor + inner schur (bulk-cheb $={b}$)"
            if strategy == "tensor_inner_schur" else
            fr"tensor (bulk-cheb $={b}$)"
        )
        handles.append(plt.Line2D(
            [], [], color=STRATEGY_COLOR[strategy],
            linestyle=ls, marker="o",
            label=label,
        ))
    return handles


def _axis_limits(rows: list[dict], metric: str,
                 rank_ticks: list[int]) -> tuple[float, float, float, float]:
    ys: list[float] = []
    for r in rows:
        y = _f(r, metric)
        ys.append(y)
        e = _err(r, metric)
        if e is not None:
            ys.extend([y - e[0], y + e[1]])
    ys_arr = np.array([y for y in ys if np.isfinite(y) and y > 0])
    if ys_arr.size:
        ymin = float(ys_arr.min()) / 1.3
        ymax = float(ys_arr.max()) * 1.3
    else:
        ymin, ymax = 1.0, 10.0
    xmin = (min(rank_ticks) - 0.5) if rank_ticks else 0.0
    xmax = (max(rank_ticks) + 0.5) if rank_ticks else 1.0
    return xmin, xmax, ymin, ymax


def plot(rows: list[dict], out_pdf: Path, out_eps: Path, *,
         metric: str = "avg_iters",
         metric_label: str | None = None) -> None:
    cases = sorted({_case(r) for r in rows}, key=_case_key)
    if metric_label is None:
        metric_label = (
            "avg CG iterations" if metric == "avg_iters"
            else "avg solve time (ms)"
        )

    tensor_keys = sorted({
        (r["strategy"], _i(r, "bulk_cheb_steps")) for r in rows
        if r["strategy"] in ("tensor", "tensor_inner_schur")
        and _i(r, "bulk_cheb_steps") >= 0
    })
    cheb_steps_present = sorted({
        _i(r, "cheb_steps") for r in rows
        if r["strategy"] == "chebyshev" and _i(r, "cheb_steps") >= 0
    })
    rank_ticks = sorted({
        _i(r, "rank") for r in rows
        if r["strategy"] in ("tensor", "tensor_inner_schur") and _i(r, "rank") >= 0
    })
    xmin, xmax, ymin, ymax = _axis_limits(rows, metric, rank_ticks)

    figsize = (3.4 * len(cases), 3.4)

    # ----- Data-only PDF (no chrome) -----------------------------------
    fig_d, axes_d = plt.subplots(
        1, len(cases), figsize=figsize, squeeze=False, sharey=True,
    )
    axes_d = axes_d[0]
    for ax, case in zip(axes_d, cases):
        case_rows = [r for r in rows if _case(r) == case]
        _draw_data_panel(ax, case_rows, metric)
        ax.set_yscale("log")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        if rank_ticks:
            ax.set_xticks(rank_ticks)
        _bare_panel(ax)
    fig_d.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig_d.savefig(out_pdf, dpi=300, transparent=True)
    plt.close(fig_d)

    # ----- Annotation-only EPS (no data) -------------------------------
    fig_a, axes_a = plt.subplots(
        1, len(cases), figsize=figsize, squeeze=False, sharey=True,
    )
    axes_a = axes_a[0]
    for col, (ax, case) in enumerate(zip(axes_a, cases)):
        ax.set_yscale("log")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        _annotate_panel(ax, _case_title(case), col, metric_label, rank_ticks)
    handles = _build_legend_handles(tensor_keys, cheb_steps_present)
    axes_a[-1].legend(handles=handles, fontsize=7, loc="best")
    fig_a.tight_layout()
    fig_a.savefig(out_eps, format="eps")
    plt.close(fig_a)


def _metric_outputs(base_pdf: Path, metric_key: str) -> tuple[Path, Path]:
    if metric_key == "avg_iters":
        out_pdf = base_pdf
    else:
        out_pdf = base_pdf.with_name(base_pdf.stem + "_timings.pdf")
    out_eps = out_pdf.with_name(out_pdf.stem + "_annot.eps")
    return out_pdf, out_eps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path,
                        help="CSV emitted by benchmark_phase1_mass_baseline_vs_tensor.py")
    parser.add_argument("--out", type=Path, default=None,
                        help="Base output PDF path for the iteration figure (default: alongside CSV).")
    parser.add_argument("--metric", choices=("both", "iters", "solve_ms"),
                        default="both",
                        help="Which metric(s) to plot (default: both).")
    args = parser.parse_args()

    csv_path: Path = args.csv_path.resolve()
    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")

    rows = load_rows(csv_path)
    if not rows:
        raise SystemExit(f"No rows in {csv_path}")

    base_pdf = args.out or csv_path.with_suffix(".pdf")
    metric_keys = {
        "both": ("avg_iters", "avg_solve_ms"),
        "iters": ("avg_iters",),
        "solve_ms": ("avg_solve_ms",),
    }[args.metric]
    for metric in metric_keys:
        out_pdf, out_eps = _metric_outputs(base_pdf, metric)
        plot(rows, out_pdf, out_eps, metric=metric)
        print(f"Wrote {out_pdf}")
        print(f"Wrote {out_eps}")


if __name__ == "__main__":
    main()
