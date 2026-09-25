#!/usr/bin/env python
"""|J| on one cross section of two states side by side, with their Poincare crossings (login node, matplotlib only).

    python paper/current_section.py --plane 0.5 --out DIR \\
        --state SAMPLE.npz TRACE.npz "title" --state SAMPLE.npz TRACE.npz "title"

SAMPLE.npz is an archive of outputs/current_sheets_2026-09-24/sample_current.py (the weak curl J = curl B pushed
forward on the planes zeta = 0 .. 1/2), TRACE.npz the trace archive of the same state (every kept line drawn, every
--line-stride-th one). One colour scale for both panels, white at |J| = 0. The figure is authored at its printed
width (--width, labels at --label-size pt) and is meant to be \\import-ed unscaled.
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mrx.plotstyle import FS, house_style  # noqa: E402

PGF = {"pgf.texsystem": "pdflatex", "pgf.rcfonts": False, "pgf.preamble": r"\providecommand{\mathdefault}[1]{#1}"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--state", nargs=3, action="append", required=True, metavar=("SAMPLE", "TRACE", "TITLE"))
    ap.add_argument("--plane", type=float, default=0.5)
    ap.add_argument("--cmap", default="Reds", help="sequential, white at 0")
    ap.add_argument("--percentile", type=float, default=99.5, help="upper end of the colour scale over both panels")
    ap.add_argument("--line-stride", type=int, default=6)
    ap.add_argument("--dot-size", type=float, default=0.1, help="marker area in pt^2")
    ap.add_argument("--width", type=float, default=6.5, help="inches")
    ap.add_argument("--label-size", type=float, default=9.0)
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--out", required=True)
    cli = ap.parse_args()

    tag = f"zeta{cli.plane:g}"
    data = [({k: z[k] for k in (f"{tag}_R", f"{tag}_Z", f"{tag}_Jmag")}, np.load(t), title)
            for (s, t, title), z in ((st, np.load(st[0])) for st in cli.state)]
    vmax = np.percentile(np.concatenate([d[f"{tag}_Jmag"].ravel() for d, _, _ in data]), cli.percentile)
    R = np.concatenate([d[f"{tag}_R"].ravel() for d, _, _ in data])
    Z = np.concatenate([d[f"{tag}_Z"].ravel() for d, _, _ in data])
    aspect = (Z.max() - Z.min()) / (R.max() - R.min())

    scale = cli.label_size / FS.label
    with house_style():
        # two equal-aspect panels and a thin colour bar, positions in inches
        x0, y0, gap, bar_gap, bar, right = 0.5, 0.42, 0.3, 0.12, 0.1, 0.55
        w = (cli.width - x0 - gap - bar_gap - bar - right) / 2
        h = w * aspect
        H = y0 + h + 0.25
        fig = plt.figure(figsize=(cli.width, H))
        axes = [fig.add_axes((x / cli.width, y0 / H, w / cli.width, h / H)) for x in (x0, x0 + w + gap)]
        cax = fig.add_axes(((x0 + 2 * w + gap + bar_gap) / cli.width, y0 / H, bar / cli.width, h / H))
        for i, (ax, (d, tr, title)) in enumerate(zip(axes, data)):
            pm = ax.pcolormesh(d[f"{tag}_R"], d[f"{tag}_Z"], d[f"{tag}_Jmag"], vmin=0.0, vmax=vmax, cmap=cli.cmap,
                               shading="gouraud", rasterized=True)
            keep = tr["final_keep"]
            ax.scatter(tr[f"final_{tag}_R"][keep][::cli.line_stride].ravel(),
                       tr[f"final_{tag}_Z"][keep][::cli.line_stride].ravel(), s=cli.dot_size, c="black",
                       linewidths=0, rasterized=True, zorder=3)
            ax.plot(d[f"{tag}_R"][-1], d[f"{tag}_Z"][-1], color="black", lw=0.5, ls="-", zorder=4)
            ax.set_xlim(R.min() - 0.01, R.max() + 0.01)
            ax.set_ylim(Z.min() - 0.01, Z.max() + 0.01)
            ax.set_aspect("equal")
            ax.set_title(title, fontsize=cli.label_size)
            ax.set_xlabel(r"$R$")
            if i == 0:
                ax.set_ylabel(r"$Z$", labelpad=1.0)
            else:
                ax.tick_params(labelleft=False)
        cb = fig.colorbar(pm, cax=cax, extend="max")
        cb.set_label(r"$|J|$")
        for a in fig.axes:
            a.tick_params(labelsize=FS.tick * scale)
            a.xaxis.label.set_size(FS.label * scale)
            a.yaxis.label.set_size(FS.label * scale)

        os.makedirs(os.path.join(cli.out, "pgf"), exist_ok=True)
        stem = f"current_{tag}"
        fig.savefig(os.path.join(cli.out, stem + ".png"), dpi=cli.dpi, bbox_inches="tight", pad_inches=0.02)
        with matplotlib.rc_context(PGF):
            fig.savefig(os.path.join(cli.out, "pgf", stem + ".pgf"), backend="pgf", dpi=cli.dpi, bbox_inches="tight",
                        pad_inches=0.02)
        for d, _, title in data:
            print(f"{title}: |J| max {d[f'{tag}_Jmag'].max():.3f}, 99.5% {np.percentile(d[f'{tag}_Jmag'], 99.5):.3f}")
        print(f"colour scale 0 .. {vmax:.3f}")
        print("->", os.path.join(cli.out, "pgf", stem + ".pgf"))


if __name__ == "__main__":
    main()
