"""Measured island widths of traced runs against the seed table (login node, numpy).

For every resonance of the seed table (landreman_seed.py's .json) and every trace archive: the lines whose traced
iota is within 2e-3 of ``nfp n / m`` are the locked ones, and the width is max(r) - min(r) over all their
crossings on all planes (the paper's Tab. 4 measurement, outputs/seed_scan_2026-09-19/measure_widths.py). At low
shear the tolerance itself selects a band of regular surfaces 2 tol / |iota'| wide (0.085 for (4,3) of shearedB), so
the width is also reported at tol = 2e-4 together with both bands ``w_tol``.

    python landreman_widths.py SEED.json NAME=trace.npz [NAME=trace.npz ...] [--out table.json]
"""
import json
import sys

import numpy as np


def widths(path, rationals, tol=2e-3):
    t = np.load(path, allow_pickle=True)
    f = str(t["fields"][-1])
    iota = np.abs(np.asarray(t[f + "_iota"], float))
    R = np.concatenate([np.asarray(t[k], float) for k in t.files if k.startswith(f + "_zeta") and k.endswith("_logr")],
                       axis=1)
    out = {}
    for key, target in rationals.items():
        sel = np.abs(iota - target) < tol
        out[key] = dict(n_locked=int(sel.sum()),
                        w=float(np.nanmax(R[sel]) - np.nanmin(R[sel])) if sel.any() else 0.0)
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    out = None
    if "--out" in args:
        out = args[args.index("--out") + 1]
        args = args[:args.index("--out")]
    seed = json.load(open(args[0]))
    nfp = seed["nfp"]
    rationals = {f"({r['m']},{r['n']})": nfp * r["n"] / r["m"] for r in seed["resonances"]}
    TOLS = (2e-3, 2e-4)
    res = {(name, tol): widths(path, rationals, tol) for name, path in (a.split("=", 1) for a in args[1:])
           for tol in TOLS}
    rows = []
    for r in seed["resonances"]:
        key = f"({r['m']},{r['n']})"
        row = dict(r, **{f"w_tol_{tol:g}": 2 * tol / r["diota"] for tol in TOLS},
                   **{f"w_meas_{name}_{tol:g}": v[key]["w"] for (name, tol), v in res.items()},
                   **{f"n_locked_{name}_{tol:g}": v[key]["n_locked"] for (name, tol), v in res.items()})
        rows.append(row)
        print(key, " ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()
                            if k not in ("m", "n", "a_star")))
    if out:
        json.dump(dict(h_r=seed["h_r"], nfp=nfp, rows=rows), open(out, "w"), indent=1)
