"""Aggregate and plot the donut-Poisson convergence sweep.

Reads the per-job ``result.json`` files emitted by
``scripts/config_scripts/test_torus_poisson_sparse.py`` under a Hydra
multirun directory, collects ``(n, p, q, error)`` tuples when available,
and produces a log-log convergence plot of the relative L2 error against
``n``. If multiple quadrature orders are present, it keeps them on
separate curves instead of collapsing them into one ``p``-only series.

Usage
-----
    python scripts/plot_poisson_convergence.py multirun/2026-05-09/14-30-00
    python scripts/plot_poisson_convergence.py multirun/<run> --out figs/poisson_conv.pdf

The script also writes a tidy CSV next to the figure and prints the
empirical convergence rate per ``p`` (least squares slope of
``log(error)`` against ``log(1/n)``).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def collect_results(root: Path) -> list[dict]:
    """Walk *root* for any ``result.json`` and return their concatenated rows."""
    rows: list[dict] = []
    for path in sorted(root.rglob("result.json")):
        with path.open() as fh:
            data = json.load(fh)
        for entry in data:
            timings = entry.get("timings", {})
            rows.append(
                {
                    "n": int(entry["n"]),
                    "p": int(entry["p"]),
                    "q": None if entry.get("q") is None else int(entry["q"]),
                    "error": float(entry["error"]),
                    "cg_time_s": float(timings.get("inverse_hodge_laplacian", 0)),
                    "m0_assembly_time_s": timings.get("assemble_mass_matrix_0"),
                    "m1_assembly_time_s": timings.get("assemble_mass_matrix_1"),
                    "source": str(path.relative_to(root)),
                }
            )
    return rows


def write_csv(rows: list[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["n", "p", "q", "error", "total_time_s", "source"]
        )
        writer.writeheader()
        writer.writerows(rows)


def _series_label(p: int, q: int | None) -> str:
    return f"p = {p}" if q is None else f"p = {p}, q = {q}"


def _dim_label(n: int) -> str:
    """Resolution tuple label for the torus Poisson test (ns = (n, 2n, n))."""
    return f"$({n},{2*n},{n})$"


def _tail_loglog_slope(xs: np.ndarray, ys: np.ndarray) -> float | None:
    """Log-log slope estimated from the last two data points."""
    if len(xs) < 2:
        return None
    return float(np.log(ys[-1] / ys[-2]) / np.log(xs[-1] / xs[-2]))


def _draw_slope_triangle(
    ax, x_anchor: float, y_anchor: float, slope: float, log_x_span: float,
    *, C: float = 1.3, fontsize: int = 10,
) -> None:
    """Draw a right-triangle slope indicator above (x_anchor, y_anchor).

    Width is 0.05 * log_x_span in log space.  All offsets are multiplicative so
    the visual gap is uniform across series on a log-log axes.

    The *nearest corner to the anchor* (at x = x_anchor) is always placed at
    y_anchor * C, giving a constant log-space gap of log(C) regardless of slope.
    The far corner is derived so that the hypotenuse has the correct log-log slope.

    Positive slope (panels 3, 4): shape  _
                                         |/
      Right angle at top-left.  Horizontal top, vertical left, hyp /.

    Negative slope (panel 1):   shape  ‾\
                                         |
      Right angle at top-right.  Horizontal top, vertical right, hyp \\.
    """
    dx = 0.05 * log_x_span
    x1 = x_anchor
    x0 = np.exp(np.log(x1) - dx)

    style = dict(linestyle="-", linewidth=0.8, color="0.25", alpha=0.85,
                 zorder=5, clip_on=False)
    label_x = np.exp(0.5 * (np.log(x0) + np.log(x1)))

    if slope >= 0:
        # Nearest corner: top-right (x1, y_h) — gap from anchor = log(C).
        # Far corner:     bottom-left (x0, y_v),  y_v = y_h * exp(-slope*dx) < y_h.
        y_h = y_anchor * C
        y_v = y_h * np.exp(-slope * dx)
        ax.plot([x0, x1], [y_h, y_h], **style)   # horizontal top
        ax.plot([x0, x0], [y_h, y_v], **style)   # vertical left
        ax.plot([x0, x1], [y_v, y_h], **style)   # hyp: bottom-left → top-right
    else:
        # Nearest corner: bottom-right (x1, y_v) — gap from anchor = log(C).
        # Far corner:     top-left (x0, y_h),  y_h = y_v * exp(-slope*dx) > y_v.
        y_v = y_anchor * C
        y_h = y_v * np.exp(-slope * dx)
        ax.plot([x0, x1], [y_h, y_h], **style)   # horizontal top
        ax.plot([x1, x1], [y_h, y_v], **style)   # vertical right
        ax.plot([x0, x1], [y_h, y_v], **style)   # hyp: top-left → bottom-right

    ax.text(label_x, y_h * C, f"{abs(slope):.3g}",
            ha="center", va="bottom", fontsize=fontsize, color="0.15", clip_on=False)


def plot_convergence(rows: list[dict], out_fig: Path) -> dict[tuple[int, int | None], float]:
    """Produce a 2x2 figure: L2 error, CG iterations (bar), CG solve time, M0/M1 assembly time."""
    by_series: dict[tuple[int, int | None], list] = {}
    for row in rows:
        key = (row["p"], row["q"])
        by_series.setdefault(key, []).append(row)

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 10.0))
    ax_err, ax_cg = axes[0]
    ax_time, ax_asm = axes[1]
    slopes: dict[tuple[int, int | None], float] = {}

    all_ns: set[int] = set()

    def sort_key(item: tuple[int, int | None]) -> tuple[int, int]:
        p, q = item
        return (p, -1 if q is None else q)

    series_list = sorted(by_series, key=sort_key)
    ref_offsets = [1.4, 1.6, 1.8, 2.0]
    MARKERS = ["o", "s", "^", "D", "P", "v"]
    LINESTYLES = ["-", "--", "-.", ":", (0, (4, 1.5)), (0, (1, 1))]
    n_series = len(series_list)

    lines_data = []
    bars_data = []  # deferred: (color, label, ns_cg, vals_cg, idx)
    triangles = []  # deferred: (ax, x_anchor, y_anchor, slope, C)

    for idx, (p, q) in enumerate(series_list):
        pts = sorted(by_series[(p, q)], key=lambda r: r["n"])
        ns = np.array([r["n"] for r in pts], dtype=float)
        errs = np.array([r["error"] for r in pts], dtype=float)
        all_ns.update(int(n) for n in ns)

        marker = MARKERS[idx % len(MARKERS)]
        ls = LINESTYLES[idx % len(LINESTYLES)]
        (line,) = ax_err.loglog(ns, errs, marker=marker, linestyle=ls, label=_series_label(p, q))
        color = line.get_color()
        lines_data.append((p, q, ns, errs, color, idx, marker, ls))
        s_err = _tail_loglog_slope(ns, errs)
        if s_err is not None:
            triangles.append((ax_err, ns[-1], errs[-1], s_err, 1.3))

        if len(pts) >= 2:
            slope, _ = np.polyfit(np.log(1.0 / ns), np.log(errs), 1)
            slopes[(p, q)] = float(slope)

        # CG iterations — collect for deferred bar chart (needs final tick_ns)
        cg = [r.get("cg_iters") for r in pts]
        if any(v is not None for v in cg):
            ns_cg = [ns[i] for i, v in enumerate(cg) if v is not None]
            vals_cg = [v for v in cg if v is not None]
            bars_data.append((color, _series_label(p, q), ns_cg, vals_cg, idx))

        # Solve time (laplacian inversion only)
        times = [r.get("cg_time_s") for r in pts]
        if any(v is not None and v > 0 for v in times):
            ns_t = np.array([ns[i] for i, v in enumerate(times) if v is not None and v > 0])
            vals_t = np.array([v for v in times if v is not None and v > 0], dtype=float)
            ax_time.loglog(ns_t, vals_t, marker=marker, linestyle=ls,
                           color=color, label=_series_label(p, q))
            s_t = _tail_loglog_slope(ns_t, vals_t)
            if s_t is not None:
                triangles.append((ax_time, ns_t[-1], vals_t[-1], s_t, 1.03))

        # M1 assembly time
        m1_times = [r.get("m1_assembly_time_s") for r in pts]
        if any(v is not None and v > 0 for v in m1_times):
            ns_a = np.array([ns[i] for i, v in enumerate(m1_times) if v is not None and v > 0])
            vals_m1 = np.array([v for v in m1_times if v is not None and v > 0], dtype=float)
            ax_asm.loglog(ns_a, vals_m1, marker=marker, linestyle=ls, color=color,
                          label=_series_label(p, q))
            s_m1 = _tail_loglog_slope(ns_a, vals_m1)
            if s_m1 is not None:
                triangles.append((ax_asm, ns_a[-1], vals_m1[-1], s_m1, 1.3))

    # Draw CG bar chart now that tick_ns is finalised.
    tick_ns = sorted(all_ns)
    tick_index = {n: i for i, n in enumerate(tick_ns)}
    bar_width = 0.7 / len(bars_data) if bars_data else 0.1
    for color, label, ns_cg, vals_cg, idx in bars_data:
        x_positions = np.array([tick_index[int(n)] for n in ns_cg], dtype=float)
        x_offset = bar_width * (idx - (n_series - 1) / 2)
        ax_cg.bar(x_positions + x_offset, vals_cg, width=bar_width,
                  color=color, label=label)

    # Log-scale panels share the same n-axis ticks
    tick_labels = [_dim_label(n) for n in tick_ns]
    for ax in (ax_err, ax_time, ax_asm):
        ax.set_xticks(tick_ns)
        ax.set_xticklabels(tick_labels, fontsize=7, rotation=15, ha="right")
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.set_xlabel(r"$(n_r,\,n_\theta,\,n_\zeta)$", fontsize=12)
        ax.grid(True, which="both", linestyle=":", alpha=0.5)

    # Bar chart x-axis: integer positions mapped to resolution labels
    ax_cg.set_xticks(np.arange(len(tick_ns)))
    ax_cg.set_xticklabels(tick_labels, fontsize=7, rotation=15, ha="right")
    ax_cg.set_xlabel(r"$(n_r,\,n_\theta,\,n_\zeta)$", fontsize=12)
    ax_cg.grid(True, axis="y", linestyle=":", alpha=0.5)

    # Error panel
    ax_err.legend(fontsize=8, loc="lower left")
    ax_err.set_ylabel(r"relative $L^2$ error", fontsize=13)
    ax_err.set_title("L² convergence", fontsize=11)

    # CG panel
    ax_cg.legend(fontsize=8)
    ax_cg.set_ylabel("CG iterations", fontsize=13)
    ax_cg.set_title("CG iterations", fontsize=11)
    ax_cg.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Time panel
    ax_time.legend(fontsize=8, loc="upper left")
    ax_time.set_ylabel("Laplacian inversion time (s)", fontsize=13)
    ax_time.set_title("Laplacian inversion time", fontsize=11)

    # Assembly panel
    ax_asm.legend(fontsize=8, loc="upper left")
    ax_asm.set_ylabel("M1 assembly time (s)", fontsize=13)
    ax_asm.set_title("1-form mass matrix assembly time", fontsize=11)

    # Slope triangles on panels 1, 3, 4
    log_x_span = np.log(tick_ns[-1]) - np.log(tick_ns[0])
    for ax, x_anchor, y_anchor, slope, c in triangles:
        _draw_slope_triangle(ax, x_anchor, y_anchor, slope, log_x_span, C=c, fontsize=10)

    # Give panels 3 and 4 headroom so triangle labels stay inside the axes.
    ylo, yhi = ax_time.get_ylim()
    ax_time.set_ylim(ylo, yhi * 1.15)
    ylo, yhi = ax_asm.get_ylim()
    ax_asm.set_ylim(ylo, yhi * 3)

    fig.tight_layout()

    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return slopes


def _rows_from_csv(csv_path: Path) -> list[dict]:
    """Read a pre-built tidy CSV (columns: n,p,relative_l2_error,...) into rows."""
    rows: list[dict] = []
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        for rec in reader:
            cg = rec.get("cg_iters")
            # Support old format (solve_time_s) and new format (cg_time_s)
            cg_raw = rec.get("cg_time_s") or rec.get("solve_time_s") or ""
            m0_raw = rec.get("m0_assembly_time_s", "")
            m1_raw = rec.get("m1_assembly_time_s", "")
            rows.append({
                "n": int(rec["n"]),
                "p": int(rec["p"]),
                "q": None,
                "error": float(rec["relative_l2_error"]),
                "cg_time_s": float(cg_raw) if cg_raw != "" else None,
                "cg_iters": int(cg) if cg is not None and cg != "" else None,
                "m0_assembly_time_s": float(m0_raw) if m0_raw != "" else None,
                "m1_assembly_time_s": float(m1_raw) if m1_raw != "" else None,
                "source": str(csv_path),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        type=Path,
        nargs="?",
        default=None,
        help="Hydra multirun directory to aggregate (e.g. multirun/<DATE>/<TIME>).",
    )
    parser.add_argument(
        "--from-csv",
        type=Path,
        default=None,
        metavar="CSV",
        help="Read a pre-built CSV instead of walking a multirun directory.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output figure path.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Output CSV path (only used when aggregating from a run_dir).",
    )
    args = parser.parse_args()

    if args.from_csv is not None:
        rows = _rows_from_csv(args.from_csv.resolve())
        out_fig = args.out or args.from_csv.with_suffix(".png")
        out_csv = None
    elif args.run_dir is not None:
        run_dir: Path = args.run_dir.resolve()
        if not run_dir.is_dir():
            raise SystemExit(f"run_dir not found: {run_dir}")
        rows = collect_results(run_dir)
        if not rows:
            raise SystemExit(f"No result.json files found under {run_dir}.")
        out_fig = args.out or (run_dir / "poisson_convergence.pdf")
        out_csv = args.csv or (run_dir / "poisson_convergence.csv")
        write_csv(rows, out_csv)
        print(f"Aggregated {len(rows)} rows from {run_dir}.")
        print(f"  CSV : {out_csv}")
    else:
        raise SystemExit("Provide either run_dir or --from-csv.")

    slopes = plot_convergence(rows, out_fig)
    print(f"  Plot: {out_fig}")
    if slopes:
        print("Empirical convergence slopes (log error vs log h):")
        for p, q in sorted(slopes, key=lambda item: (item[0], -1 if item[1] is None else item[1])):
            label = _series_label(p, q)
            print(f"  {label}: slope = {slopes[(p, q)]:.2f}")


if __name__ == "__main__":
    main()
