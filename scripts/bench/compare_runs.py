"""Compare ``scripts/relax.py`` runs across hardware: setup, compile and steady-state step time, and the wall time and
cost to reach given force residuals.

    python scripts/bench/compare_runs.py outputs/run-a outputs/run-b ... [--targets 1e-7 3e-8] [--json table.json]

Each argument is a directory holding a ``relax.json`` (searched recursively, so a GKE job directory with ``run-1/``
to ``run-4/`` works too). The device comes from ``params.device_kind`` (written by ``relax.py`` since it records
the hardware; older files fall back to ``--device``).

Times:

``setup``
    ``timing.setup_s + timing.ic_s``: the de Rham sequence, nullspaces and initial field, before the first step.
``compile``
    the wall time of the first chunk of steps less those steps at the steady rate: compiling the stepper (or
    loading it from the persistent cache).
``s/step``
    the steady state: the wall time of the later chunks divided by their steps.
``to <target>``
    the steps until ``trace.resid`` first falls below the target, times ``s/step``: the solver time to the target,
    without setup and compile (listed separately, they are paid once per run). ``-`` if never reached.

Cost is that time times the price per hour of the hardware (``--price``, default the Spot price of one chip or the
cost of one GPU; a v6e or v5p chip is a quarter of the 4-chip node GKE bills, the share of a run with ``RUNS``).
"""
from __future__ import annotations

import argparse
import glob
import json
import os

# USD per hour of one device: Spot per chip, us-central1 / us-south1, 2026-10-18 (Cloud Billing Catalog API, see
# gke/spot_prices.py), and the H200 cost Tillicum quotes per GPU-hour
PRICES = {"TPU v6 lite": 0.584, "TPU v5p": 0.664, "TPU v5": 0.664, "NVIDIA H200": 0.90}


def load(path, device=None):
    rec = json.load(open(path))
    p, q, tr = rec["params"], rec["qoi"], rec["trace"]
    its, wall = list(q["it"]), list(q["wall"])
    timing = rec.get("timing", {})
    row = dict(path=os.path.dirname(path), device=p.get("device_kind") or device or "?",
               mesh="x".join(str(n) for n in p["resolution"]), precision=p.get("precision"),
               method=p.get("method"), maxiter=p.get("newton_maxiter"),
               setup_s=timing.get("setup_s", float("nan")) + timing.get("ic_s", 0.0),
               steps=rec["summary"]["steps"], E_removed=rec["summary"]["E_removed"],
               best_resid=rec["summary"]["best_resid"])
    # qoi rows are at the chunk ends: it[0] = start (wall 0), then one per chunk
    if len(its) >= 3:
        row["s_per_step"] = (wall[-1] - wall[1]) / (its[-1] - its[1])
        row["compile_s"] = (wall[1] - wall[0]) - (its[1] - its[0]) * row["s_per_step"]
    row["_resid"] = list(tr["resid"])
    return row


def time_to(row, target):
    for n, r in enumerate(row["_resid"], 1):
        if r <= target:
            return n * row["s_per_step"] if "s_per_step" in row else None
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dirs", nargs="+", help="run directories (searched for relax.json)")
    ap.add_argument("--targets", type=float, nargs="+", default=[1e-7, 3e-8], help="force residuals to time")
    ap.add_argument("--price", action="append", default=[], metavar="DEVICE=USD_PER_H",
                    help="price per device-hour, e.g. 'NVIDIA H200=0.9' (repeatable)")
    ap.add_argument("--device", help="device of relax.json files that do not record it")
    ap.add_argument("--json", help="also write the table as JSON")
    a = ap.parse_args()
    prices = dict(PRICES, **{k: float(v) for k, v in (x.rsplit("=", 1) for x in a.price)})
    rows = []
    for d in a.dirs:
        for path in sorted(glob.glob(os.path.join(d, "**", "relax.json"), recursive=True)):
            rows.append(load(path, a.device))
    head = ["device", "mesh", "prec", "method", "maxiter", "setup s", "compile s", "s/step", "best resid",
            "E removed"] + [f"to {t:.0e} s (USD)" for t in a.targets]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    out = []
    for r in sorted(rows, key=lambda r: (r["mesh"], r["device"], r["precision"] or "", r["maxiter"] or 0)):
        price = prices.get(r["device"])
        cells = [r["device"], r["mesh"], r["precision"], r["method"], r["maxiter"], f"{r['setup_s']:.0f}",
                 f"{r.get('compile_s', float('nan')):.0f}", f"{r.get('s_per_step', float('nan')):.2f}",
                 f"{r['best_resid']:.2e}", f"{r['E_removed']:.3e}"]
        for t in a.targets:
            w = time_to(r, t)
            r[f"to_{t:g}_s"] = w
            cells.append("-" if w is None else
                         f"{w:.0f} ({w / 3600 * price:.4f})" if price else f"{w:.0f}")
        print("| " + " | ".join(str(c) for c in cells) + " |")
        out.append({k: v for k, v in r.items() if not k.startswith("_")})
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
