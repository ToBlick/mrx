"""Aggregate and plot the donut-Poisson convergence sweep.

Reads the per-job ``result.json`` files emitted by
``scripts/config_scripts/test_torus_poisson_sparse.py`` under a Hydra
multirun directory, collects ``(n, p, error)`` triples, and produces a
log-log convergence plot of the relative L2 error against ``n``, with
one curve per ``p``.

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
            rows.append(
                {
                    "n": int(entry["n"]),
                    "p": int(entry["p"]),
                    "error": float(entry["error"]),
                    "total_time_s": float(entry["timings"]["TOTAL"]),
                    "source": str(path.relative_to(root)),
                }
            )
    return rows


def write_csv(rows: list[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["n", "p", "error", "total_time_s", "source"]
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_convergence(rows: list[dict], out_fig: Path) -> dict[int, float]:
    """Produce the convergence plot. Return empirical slopes per ``p``."""
    by_p: dict[int, list[tuple[int, float]]] = {}
    for row in rows:
        by_p.setdefault(row["p"], []).append((row["n"], row["error"]))

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    slopes: dict[int, float] = {}

    for p in sorted(by_p):
        pts = sorted(by_p[p])
        ns = np.array([n for n, _ in pts], dtype=float)
        errs = np.array([e for _, e in pts], dtype=float)
        ax.loglog(ns, errs, marker="o", label=f"p = {p}")

        if len(pts) >= 2:
            log_h = np.log(1.0 / ns)
            log_e = np.log(errs)
            slope, _ = np.polyfit(log_h, log_e, 1)
            slopes[p] = float(slope)

    ax.set_xlabel(r"$n$ (per dimension)")
    ax.set_ylabel(r"relative $L^2$ error")
    ax.set_title("Donut Poisson convergence")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()

    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=200)
    plt.close(fig)
    return slopes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Hydra multirun directory to aggregate (e.g. multirun/<DATE>/<TIME>).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output figure path (default: <run_dir>/poisson_convergence.pdf).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Output CSV path (default: <run_dir>/poisson_convergence.csv).",
    )
    args = parser.parse_args()

    run_dir: Path = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"run_dir not found: {run_dir}")

    rows = collect_results(run_dir)
    if not rows:
        raise SystemExit(f"No result.json files found under {run_dir}.")

    out_fig = args.out or (run_dir / "poisson_convergence.pdf")
    out_csv = args.csv or (run_dir / "poisson_convergence.csv")

    write_csv(rows, out_csv)
    slopes = plot_convergence(rows, out_fig)

    print(f"Aggregated {len(rows)} rows from {run_dir}.")
    print(f"  CSV : {out_csv}")
    print(f"  Plot: {out_fig}")
    if slopes:
        print("Empirical convergence slopes (log error vs log h):")
        for p in sorted(slopes):
            print(f"  p = {p}: slope = {slopes[p]:.2f}")


if __name__ == "__main__":
    main()
