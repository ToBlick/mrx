#!/usr/bin/env python
"""The logical chart (theta, r) of one plane of a trace archive as a figure of its own (login node, matplotlib only).

    python paper/logical_section.py ARCHIVE --plane 0.5 --pressure-factor F --out DIR

The crossings are coloured as in scripts/poincare_plot.py's logical panel: the rotational transform of their line for
theta < 1/2, the pressure for theta >= 1/2, lost lines grey. The iota and pressure ranges are those of the plotter
over every plane of the archive, so the colours match its pages. Unlike the three-panel page, the iota bar sits on the
left and the (inverted) pressure bar on the right of the chart. The figure is authored at its printed size (--side,
the side of the square chart, labels at --label-size pt) and is meant to be \\import-ed unscaled.
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mrx.plotstyle import FS, PRESSURE_CMAP, SECTION_CMAP, house_style  # noqa: E402
from mrx.plotting import resonant_rationals  # noqa: E402

PGF = {"pgf.texsystem": "pdflatex", "pgf.rcfonts": False, "pgf.preamble": r"\providecommand{\mathdefault}[1]{#1}"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("archive", help="trace.npz of one field")
    ap.add_argument("--plane", type=float, default=0.5)
    ap.add_argument("--pressure-factor", type=float, required=True, help="p_norm = factor * p (poincare_pages.py)")
    ap.add_argument("--pressure-label", default=r"$p_{\mathrm{norm}}$")
    ap.add_argument("--dot-size", type=float, default=0.4, help="marker area in pt^2")
    ap.add_argument("--denom-max", type=int, default=300)
    ap.add_argument("--min-sep", type=float, default=0.05, help="rational ticks: least spacing, in units of the range")
    ap.add_argument("--side", type=float, default=3.4, help="side of the square chart, inches")
    ap.add_argument("--label-size", type=float, default=9.0)
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--out", required=True)
    cli = ap.parse_args()

    z = np.load(cli.archive)
    (f,) = [str(v) for v in z["fields"]]
    nfp = int(z["nfp"])
    keep, shown, iota = z[f"{f}_keep"], z[f"{f}_shown"], z[f"{f}_iota"]
    # the plotter's scales: iota over the shown lines, the weak pressure (gauge 0) over the kept lines of every plane,
    # both x 100 / padded 5% as there
    lo, hi = float(iota[shown].min()), float(iota[shown].max())
    ps = [100.0 * cli.pressure_factor * z[f"{f}_zeta{pl:g}_pressure"][keep] for pl in z["planes"]]
    p0, p1 = min(float(np.nanmin(v)) for v in ps), max(float(np.nanmax(v)) for v in ps)
    p_lim = (p0 - 0.05 * (p1 - p0), p1 + 0.05 * (p1 - p0))
    lr, lth = z[f"{f}_zeta{cli.plane:g}_logr"], z[f"{f}_zeta{cli.plane:g}_logth"]
    press = 100.0 * cli.pressure_factor * z[f"{f}_zeta{cli.plane:g}_pressure"]
    colour = np.broadcast_to(iota[:, None], lr.shape)
    shown2 = np.broadcast_to(shown[:, None], lr.shape)
    left = lth < 0.5

    scale = cli.label_size / FS.label
    with house_style():
        # a square chart of side --side inches, the two bars as tall as it, positions in inches
        side, bar, gap_i, gap_p, x0, y0 = cli.side, 0.1, 0.45, 0.16, 0.55, 0.45
        W, H = x0 + bar + gap_i + side + gap_p + bar + 0.45, y0 + side + 0.08
        fig = plt.figure(figsize=(W, H))
        box = lambda x, w: fig.add_axes((x / W, y0 / H, w / W, side / H))  # noqa: E731
        cax_i, ax, cax_p = box(x0, bar), box(x0 + bar + gap_i, side), box(x0 + bar + gap_i + side + gap_p, bar)
        s = dict(s=cli.dot_size, linewidths=0, rasterized=True)
        sc = ax.scatter(lth[shown2 & left], lr[shown2 & left], c=colour[shown2 & left], vmin=lo, vmax=hi,
                        cmap=SECTION_CMAP, **s)
        psc = ax.scatter(lth[shown2 & ~left], lr[shown2 & ~left], c=press[shown2 & ~left], vmin=p_lim[0],
                         vmax=p_lim[1], cmap=PRESSURE_CMAP, **s)
        lost = np.broadcast_to(~keep[:, None], lr.shape)
        if lost.any():
            ax.scatter(lth[lost], lr[lost], c="0.55", **s)
        ax.axvline(0.5, color="0.35", lw=0.6, ls=":", zorder=1)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel(r"$r$", labelpad=1.0)

        cb_i = fig.colorbar(sc, cax=cax_i)
        cb_i.ax.yaxis.set_ticks_position("left")
        ticks, labels = resonant_rationals(lo, hi, nfp, cli.denom_max, cli.min_sep)
        cb_i.set_ticks(ticks)
        cb_i.set_ticklabels(labels)
        cb_i.ax.set_xlabel(r"$\iota$")
        cb_p = fig.colorbar(psc, cax=cax_p)
        cb_p.ax.invert_yaxis()
        cb_p.ax.set_ylabel(cli.pressure_label + r" $\times$ 100")    # on the right, beside the bar
        cb_p.ax.yaxis.set_label_position("right")

        for a in fig.axes:
            a.tick_params(labelsize=FS.tick * scale)
            a.xaxis.label.set_size(FS.label * scale)
            a.yaxis.label.set_size(FS.label * scale)
        for cb in (cb_i, cb_p):
            cb.ax.tick_params(labelsize=FS.annot * scale)

        os.makedirs(os.path.join(cli.out, "pgf"), exist_ok=True)
        stem = f"logical_zeta{cli.plane:g}"
        fig.savefig(os.path.join(cli.out, stem + ".png"), dpi=cli.dpi, bbox_inches="tight", pad_inches=0.02)
        with matplotlib.rc_context(PGF):
            fig.savefig(os.path.join(cli.out, "pgf", stem + ".pgf"), backend="pgf", dpi=cli.dpi, bbox_inches="tight",
                        pad_inches=0.02)
        print(f"{len(ticks)} rational ticks: {', '.join(labels)}; iota {lo:.4f}..{hi:.4f}, p {p_lim[0]:.3g}..{p_lim[1]:.3g}")
        print("->", os.path.join(cli.out, "pgf", stem + ".pgf"))


if __name__ == "__main__":
    main()
