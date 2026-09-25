#!/usr/bin/env python
"""Fig. 8 of Sec. 3.4 (qa_poincare_optimized): the end field of the 10 mm draw 0, remeshed (login node, matplotlib).

    python paper/ad_poincare_page.py [--records DIR] [--out DIR]

Reads <records>/shape_optimization/qa_trace_M10s0_remesh.npz (paper/runs/ad_trace.sh); writes its field ``final`` as
a one-field archive to <out>/poincare/qa_poincare_optimized/trace.npz, renders it with this checkout's
scripts/poincare_plot.py and copies the page, its .pgf and -img*.png rasters, to <out>/figs/pgf/qa_poincare_optimized/
[paper/build], as paper/poincare_pages.py does for the other pages. The PGF needs pdflatex on PATH.

The flags are those of the paper's vacuum page, --paper --no-pressure --planes 0.25 --dot-scale 0.4. The archive holds
LP's field beside the end field; the page of a one-field archive is poincare_zeta0.25, the name the paper imports.
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOT = os.path.join(REPO, "scripts", "poincare_plot.py")
PAGE = "qa_poincare_optimized"
ARCHIVE = "qa_trace_M10s0_remesh.npz"
FLAGS = ["--paper", "--no-pressure", "--planes", "0.25", "--dot-scale", "0.4"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--records", default=os.environ.get("MRX_RECORDS", os.path.join(REPO, "outputs")),
                    help="the records root, <records>/shape_optimization [MRX_RECORDS or outputs]")
    ap.add_argument("--out", default=os.path.join(REPO, "paper", "build"),
                    help="poincare/ renders and figs/pgf/ pages [paper/build]")
    cli = ap.parse_args()
    z = np.load(os.path.join(cli.records, "shape_optimization", ARCHIVE))
    render = os.path.join(cli.out, "poincare", PAGE)
    os.makedirs(render, exist_ok=True)
    archive = os.path.join(render, "trace.npz")
    np.savez(archive, **{k: z[k] for k in z.files if k != "fields" and not k.startswith("start_")},
             fields=np.array(["final"]))
    subprocess.run([sys.executable, PLOT, archive, *FLAGS, "--out", render], check=True)
    dst = os.path.join(cli.out, "figs", "pgf", PAGE)
    os.makedirs(dst, exist_ok=True)
    page = os.path.join(render, "pgf", "poincare_zeta0.25")
    for f in [page + ".pgf"] + glob.glob(page + "-img*.png"):
        shutil.copy(f, dst)
    print(f"-> figs/pgf/{PAGE}/poincare_zeta0.25.pgf", flush=True)


if __name__ == "__main__":
    main()
