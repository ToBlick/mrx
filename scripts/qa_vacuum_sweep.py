"""Sweep :func:`scripts.vacuum_convergence.run_rung` over a resolution list at
one degree p, so a whole p-column is one SLURM job instead of one job per rung.

    python -u scripts/qa_vacuum_sweep.py --geometry data/wout_..._highres.nc \
        --p 3 --ns 8,16,8:12,24,12:16,32,16 --out outputs/qa_vacuum_highres

Then merge and plot the whole directory with
``python scripts/vacuum_convergence.py --plot outputs/qa_vacuum_highres``.
"""
import argparse
import os

# vacuum_convergence exports its --precision (default float64) only from its
# own __main__; this sweep runs the rungs at that default too (the package
# default is float32 since 2026-09-04).
os.environ.setdefault("MRX_DTYPE", "float64")

import vacuum_convergence as vc  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--geometry", required=True)
    ap.add_argument("--p", type=int, required=True)
    ap.add_argument("--ns", required=True, help="colon-separated n_r,n_theta,n_zeta rungs")
    ap.add_argument("--grid", default="48,96,48")
    ap.add_argument("--out", required=True)
    cli = ap.parse_args()
    for chunk in cli.ns.split(":"):
        base = vc.parse_args(["--geometry", cli.geometry, "--ns", chunk,
                              "--p", str(cli.p), "--grid", cli.grid, "--out", cli.out])
        vc.run_rung(base)


if __name__ == "__main__":
    main()
