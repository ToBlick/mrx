"""Re-render Poincaré figures from a saved trace, with no solve and no map.

``poincare_vacuum.py`` archives ``(R, Z)`` per plane alongside the raw ``(u, v)``
orbits precisely so that presentation can be changed without paying for the
nullspace solves and the map again -- which is minutes per geometry, against
under a second here.  Runs on a login node: matplotlib and numpy only.

    python scripts/debug/poincare_replot.py outputs/poincare_v2/*/trace_*.npz
"""
from __future__ import annotations

import argparse
import os
import re

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402

from mrx.poincare import render_section  # noqa: E402

NFP = {"toroid": 1, "cylinder": 1, "rot-ellipse": 3, "w7x": 5,
       "quasr9983": 2, "quasr44970": 3, "w7x-gvec": 5, "hegna": 3}
LABEL = {"k2": "k=2, essential BC", "k1": "k=1, natural BC"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("traces", nargs="+", help="trace_*.npz files")
    ap.add_argument("--outdir", default=None,
                    help="defaults to the directory each trace sits in")
    cli = ap.parse_args()

    for path in cli.traces:
        m = re.match(r"trace_(.+)_(k[12])\.npz$", os.path.basename(path))
        if not m:
            raise ValueError(f"cannot parse geometry/field out of {path!r}")
        geometry, field = m.groups()
        d = np.load(path)
        planes = sorted(float(k[len("R_zeta"):]) for k in d.files
                        if k.startswith("R_zeta"))
        if not planes:
            raise KeyError(
                f"{path} has no R_zeta* arrays -- it predates the (R, Z) "
                "archive, so re-rendering it needs the map and a rerun")

        keep = ~(d["escaped"] | ~d["ok"])
        outdir = cli.outdir or os.path.dirname(path) or "."
        os.makedirs(outdir, exist_ok=True)
        for plane in planes:
            R, Z = d[f"R_zeta{plane:g}"], d[f"Z_zeta{plane:g}"]
            axis = ((d[f"axisR_zeta{plane:g}"], d[f"axisZ_zeta{plane:g}"])
                    if f"axisR_zeta{plane:g}" in d.files else None)
            out = os.path.join(
                outdir, f"poincare_{geometry}_{field}_zeta{plane:g}.png")
            render_section(
                R, Z, d["iota"], d["resid"], d["seeds"][:, 0], keep,
                title=f"{geometry}  |  {LABEL[field]}  |  $\\zeta = {plane:g}$\n"
                      f"{R.shape[1]} crossings/line",
                subtitle=f"nfp = {NFP[geometry]}   |   "
                         f"{int(keep.sum())}/{len(keep)} lines kept",
                axis_RZ=axis, path=out)
            print(out, flush=True)


if __name__ == "__main__":
    main()
