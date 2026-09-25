#!/usr/bin/env python
"""Figs. 6 and 7 of Sec. 3.4 (qa_paper_qa, qa_paper_shape) from the shape-optimization records (login node, matplotlib).

    python paper/ad_figures.py [--records DIR] [--out DIR]

Reads <records>/shape_optimization/: qa_constrained<tag>.json and qa_recover<tag>.json of the pilot and the four draws
(paper/runs/ad_constrained.sh), qa_baseline_{16,24,32}.json (ad_baseline.sh), qa_boundary_baselines.json (ad_baselines.py).
Writes <out>/figs/<stem>.{pdf,png} and <out>/figs/pgf/<stem>/<stem>.pgf through paper/figures.py's writer [paper/build].

Fig. 6: <Q_QA^2>_{r >= h_r} at every L-BFGS-B iteration (the constrained record's F_path, in units of LP's value),
log-log, with LP's criterion on its interpolated map at 16x32x16, 24x48x24 and 32x64x32 in grey. Fig. 7: the
boundary's d_RMS to LP's over LP's minor radius a at the outer steps (every 100 iterations, the record's rms_mm), log
iteration and linear distance, with a top-right log inset of the run from the device against the same distance of the
spline-fitted boundary to the VMEC boundary at the three meshes. Prints the numbers Sec. 3.4 quotes from the runs.
The PGF needs xelatex on PATH.
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from figures import save  # noqa: E402  (paper/figures.py, next to this file)
from mrx.plotstyle import CYCLE, FS, figsize, house_style  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#: the pilot from the device and the four 10 mm draws (draw 3 fails the fold guard and is not run), in the house
#: (colour, dash) cycle, one marker each on top
RUNS = [("_M0", "device")] + [(f"_M10s{k}", f"draw {k}") for k in (0, 1, 2, 4)]
MARKERS = ("s", "o", "^", "d", "v")
MESHES = (16, 24, 32)


def load(folder, tag):
    """The run's iterations and F_QS (absolute) at every iteration, and its iterations and d_RMS / a at the outer
    steps, the start included."""
    rec = json.load(open(os.path.join(folder, f"qa_constrained{tag}.json")))
    a = json.load(open(os.path.join(folder, f"qa_recover{tag}.json")))["a_lp"]
    path = np.array(rec["F_path"], dtype=float)
    outer = rec["outer"]
    qa = (np.r_[0, path[:, 0]], np.r_[outer[0]["F_qs"], path[:, 1] * rec["units"]["f"]])
    shape = (np.array([o["iterations"] for o in outer], float), np.array([1e-3 * o["rms_mm"] / a for o in outer]))
    return rec, qa, shape


def baselines(ax, values, x, ha, fontsize, pad):
    for n, v in values:
        ax.axhline(v, color="0.5", lw=0.7)
        ax.annotate(r"$%d^3$" % n, (x, v), xytext=(0, pad), textcoords="offset points", ha=ha, va="bottom",
                    fontsize=fontsize, color="0.35")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--records", default=os.environ.get("MRX_RECORDS", os.path.join(REPO, "outputs")),
                    help="the records root, <records>/shape_optimization [MRX_RECORDS or outputs]")
    ap.add_argument("--out", default=os.path.join(REPO, "paper", "build"), help="figs/ in the paper's layout")
    cli = ap.parse_args()
    folder = os.path.join(cli.records, "shape_optimization")
    data = {tag: load(folder, tag) for tag, _ in RUNS}
    qa_base = [(n, json.load(open(os.path.join(folder, f"qa_baseline_{n}.json")))["F_lp"]) for n in MESHES]
    boundary = json.load(open(os.path.join(folder, "qa_boundary_baselines.json")))
    d_base = [(n, boundary[str(n)]["d_rms_over_a"]) for n in MESHES]
    print(f"LP's criterion {', '.join(f'{n}^3 {v:.3e}' for n, v in qa_base)}; rates "
          f"{np.log(qa_base[0][1] / qa_base[1][1]) / np.log(24 / 16):.2f} (16 -> 24), "
          f"{np.log(qa_base[1][1] / qa_base[2][1]) / np.log(32 / 24):.2f} (24 -> 32); interpolated boundary d_RMS / a "
          + ", ".join(f"{n}^3 {v:.3e}" for n, v in d_base))
    for tag, label in RUNS:
        rec, (qi, qv), (si, sv) = data[tag]
        o, end, f_lp = rec["outer"][0], rec["end"], rec["units"]["f"]
        print(f"{label:7s} start: QA {o['F_qs'] / f_lp:.3g} LP, <iotabar>_s - target {o['mean_iota'] - rec['iota_target']:+.4f}, "
              f"d_RMS / a {sv[0]:.3f}; end ({end['reason']}): iteration {int(qi[-1])}, QA {end['qa_own']:.3f} LP, "
              f"d_RMS / a {sv[-1]:.2e}, P {end['P']:.3e}, lambda_i {end['lam_i']:+.4f}, lambda_P {end['lam_P']:.4f}, "
              f"traced - flux ratio <iota>_s {end['iota_traced'] - end['iota_flux_ratio']:+.1e}")

    with house_style():
        fig, ax = plt.subplots(figsize=figsize("column"))
        for (tag, label), (colour, dash), marker in zip(RUNS, CYCLE, MARKERS):
            qi, qv = data[tag][1]
            ax.plot(qi + 1, qv, color=colour, ls=dash, lw=1.0, label=label, marker=marker, ms=3, markevery=0.07)
        baselines(ax, qa_base, 1.2, "left", FS.annot, 1.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"iteration $+\,1$")
        ax.set_ylabel(r"$\langle Q_{\mathrm{QA}}^2 \rangle_{r \geq h_r}$")
        ax.grid(alpha=0.3, which="both")
        ax.legend(loc="upper right")
        save(fig, cli.out, "qa_paper_qa", "qa_paper_qa")

    with house_style():
        fig, ax = plt.subplots(figsize=figsize("column"))
        for (tag, label), (colour, dash), marker in zip(RUNS, CYCLE, MARKERS):
            si, sv = data[tag][2]
            ax.plot(si + 1, sv, color=colour, ls=dash, lw=1.0, label=label, marker=marker, ms=3)
        ax.set_xscale("log")
        ax.set_ylim(bottom=0)
        ax.set_xlabel(r"iteration $+\,1$")
        ax.set_ylabel(r"$d_{\mathrm{RMS}} / \mathsf{a}$")
        ax.grid(alpha=0.3, which="both")
        # top-right inset, log y: the run from the device against the VMEC baselines, which sit on zero in the main axes
        ins = ax.inset_axes([0.56, 0.60, 0.41, 0.36])
        si, sv = data[RUNS[0][0]][2]
        ins.plot(si + 1, sv, color=CYCLE[0][0], ls=CYCLE[0][1], lw=1.0, marker=MARKERS[0], ms=2)
        baselines(ins, d_base, 0.97 * (si[-1] + 1), "right", 6, 1)
        ins.set_xscale("log")
        ins.set_yscale("log")
        ins.tick_params(labelsize=6, length=2)
        ins.grid(alpha=0.3, which="major")
        save(fig, cli.out, "qa_paper_shape", "qa_paper_shape")


if __name__ == "__main__":
    main()
