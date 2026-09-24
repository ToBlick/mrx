#!/usr/bin/env python
"""Every Poincare page of the paper from the trace archives (login node, matplotlib only).

    python paper/poincare_pages.py [--records DIR] [--out DIR] [--only PAGE,...]

Renders each archive with scripts/poincare_plot.py --paper into <out>/poincare/<page>/ and copies the page the
paper shows, its .pgf and the -img*.png rasters it references, into <out>/figs/pgf/<page>/, the paper's layout.
The pressure is drawn normalised, p_norm = p / <|B|^2 / 2> = 2 |Omega| p for the initial field's ||B|| = 1, with
|Omega| the volume of one field period (the run's wout: volume_p / nfp). The pages of Fig. C.18 share one iota and
one pressure scale. The PGF needs pdflatex on PATH.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import scipy.io

PLOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "scripts", "poincare_plot.py")

C18 = ["--planes", "0.5", "--dot-scale", "0.4"]
#: page -> (archive under the records root, the plane the paper shows, poincare_plot.py flags)
PAGES = {
    "vacuum_qa_poincare": ("vacuum_vmec/highres/rung_32x64x32_p3/run/trace.npz", 0.25,
                           ["--no-pressure", "--planes", "0.25", "--dot-scale", "0.4"]),
    "islands_equilibrium48": ("newton_convergence/newton_48/trace.npz", 0.0, []),
    "islands_seeded32": ("seeding/relax32/trace.npz", 0.0, []),
    "reconnection_unseeded_before": ("newton_convergence/newton_32/trace.npz", 0.375, []),
    "reconnection_s51q_before": ("reconnection/s51q/ideal/trace.npz", 0.375, []),
    "reconnection_s61_before": ("reconnection/s61/ideal/trace.npz", 0.375, []),
    "reconnection_unseeded_after": ("reconnection/unseeded/ideal_after/trace.npz", 0.375, []),
    "reconnection_s51q_after": ("reconnection/s51q/ideal_after/trace.npz", 0.375, []),
    "reconnection_s61_after": ("reconnection/s61/ideal_after/trace.npz", 0.375, []),
    "newton_poincare_n16": ("newton_convergence/newton_16/cont/trace.npz", 0.5, C18),
    "newton_poincare_n24": ("newton_convergence/newton_24/cont/trace.npz", 0.5, C18),
    "newton_poincare_n32": ("newton_convergence/newton_32/cont/trace.npz", 0.5, C18),
}
SHARED = ("newton_poincare_n16", "newton_poincare_n24", "newton_poincare_n32")


def pressure_factor(archive):
    """2 |Omega| from the wout of the run the archive sits next to."""
    params = json.load(open(os.path.join(os.path.dirname(archive), "relax.json")))["params"]
    with scipy.io.netcdf_file(params["geometry_path"], "r", mmap=False) as d:
        return 2.0 * float(d.variables["volume_p"].data) / int(d.variables["nfp"].data)


def limits(archives, factor):
    """One iota range (the shown lines) and one pressure range (the kept lines, x 100, padded 5%) over the
    archives, in the units poincare_plot.py draws."""
    iota, p = [], []
    for a in archives:
        z = np.load(a)
        f = str(z["fields"][0])
        iota.append(z[f"{f}_iota"][z[f"{f}_shown"]])
        p += [100 * factor * z[f"{f}_zeta{pl:g}_pressure"][z[f"{f}_keep"]].ravel() for pl in z["planes"]]
    iota, p = np.concatenate(iota), np.concatenate(p)
    lo, hi = np.nanmin(p), np.nanmax(p)
    return f"{iota.min():.6g},{iota.max():.6g}", f"{lo - 0.05 * (hi - lo):.6g},{hi + 0.05 * (hi - lo):.6g}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--records", default="outputs", help="the records root, <records>/<experiment>/<arm> [outputs]")
    ap.add_argument("--out", default="paper/build", help="poincare/ renders and figs/pgf/ pages [paper/build]")
    ap.add_argument("--only", default=",".join(PAGES), help="comma-separated subset of the pages")
    cli = ap.parse_args()
    archive = {page: os.path.join(cli.records, PAGES[page][0]) for page in PAGES}
    shared = None
    for page in cli.only.split(","):
        _, plane, flags = PAGES[page]
        needed = [archive[p] for p in (SHARED if page in SHARED else (page,))]
        missing = [a for a in needed if not os.path.exists(a)]
        if missing:                                 # a queued trace
            print(f"{page}: skipped, no {', '.join(missing)}")
            continue
        if "--no-pressure" not in flags:
            factor = pressure_factor(archive[page])
            flags = flags + ["--pressure-factor", f"{factor:.6g}", "--pressure-label", r"$p_{\mathrm{norm}}$"]
        if page in SHARED:
            shared = shared or limits(needed, factor)
            flags = flags + [f"--iota-lim={shared[0]}", f"--p-lim={shared[1]}"]
        render = os.path.join(cli.out, "poincare", page)
        print(f"=== {page}: {archive[page]} {' '.join(flags)}", flush=True)
        subprocess.run([sys.executable, PLOT, archive[page], "--paper", *flags, "--out", render], check=True)
        stem = f"poincare_zeta{plane:g}"            # the archives hold one field each
        dst = os.path.join(cli.out, "figs", "pgf", page)
        os.makedirs(dst, exist_ok=True)
        page_pgf = os.path.join(render, "pgf", stem)
        for f in [page_pgf + ".pgf"] + glob.glob(page_pgf + "-img*.png"):
            shutil.copy(f, dst)
        print(f"-> figs/pgf/{page}/{stem}.pgf", flush=True)


if __name__ == "__main__":
    main()
