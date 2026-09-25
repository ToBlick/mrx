#!/usr/bin/env python
"""Fig. 7, a schematic of ideal relaxation (Tobias 2026-09-08: "plot the energy as the z-axis, the helicity
isoclines become sheet-shaped, and on the helicity isocline there is a line that is the actual topology
constraint"). Plain matplotlib, no data.

    python scripts/paper_scripts/energy_landscape.py [--out DIR]

Writes figs/energy_landscape.{pdf,png} and figs/pgf/energy_landscape/energy_landscape.pgf under --out
[scripts/paper_scripts/build]; the PGF needs pdflatex on PATH.

Each helicity isocline is a bowl, parabolic at its bottom and deformed further out: the Taylor state at the bottom of
the isocline of B_0, the vacuum at the bottom of the zero-helicity one underneath. The ideal orbit of B_0 is a line
on its sheet that winds around the bowl; the energy along it has kinks (current sheets) and smooth minima, and the
descent from B_0 ends in the nearest minimum, a kink, B_eq, above the Taylor state.
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402
from mpl_toolkits.mplot3d import proj3d  # noqa: E402

from mrx.plotstyle import BLACK, PURPLE, TEAL, figsize, house_style  # noqa: E402

ELEV, AZIM = 35.0, -62.0
E_TAYLOR = 0.9                                             # the energy of the Taylor state above the vacuum
Z_TOP = 2.6                                                # both sheets are cut at this energy
DEFORM = 0.16                                              # the sheets' departure from a paraboloid, O(r^3)
FS = 8                                                     # the font size of the state labels
LABEL = dict(fontsize=FS, zorder=8)


def sheet(z0, r, phi):
    """The energy on one isocline: parabolic at the bottom, deformed further out."""
    return z0 + r ** 2 * (1 + DEFORM * r * np.cos(3 * phi + 0.5))


def sheet_radius(z0, level, phi):
    """The radius of the sheet's contour at the given energy (bisection; the sheet is monotone in r)."""
    lo, hi = np.zeros_like(phi), np.full_like(phi, 2.0)
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        above = sheet(z0, mid, phi) > level
        hi = np.where(above, mid, hi)
        lo = np.where(above, lo, mid)
    return 0.5 * (lo + hi)


def bowl(ax, z0, colour, alpha, rim_colour, zorder):
    """One isocline sheet from its bottom up to Z_TOP, its rim drawn."""
    phi = np.linspace(0, 2 * np.pi, 120)
    levels = z0 + (Z_TOP - z0) * np.linspace(0, 1, 40) ** 2
    LL, PP = np.meshgrid(levels, phi)
    RR = sheet_radius(z0, LL, PP)
    ax.plot_surface(RR * np.cos(PP), RR * np.sin(PP), LL, color=colour, alpha=alpha, linewidth=0, shade=True,
                    zorder=zorder)
    rr = sheet_radius(z0, np.full_like(phi, Z_TOP), phi)
    ax.plot(rr * np.cos(phi), rr * np.sin(phi), np.full_like(phi, Z_TOP), color=rim_colour, lw=0.4, ls="-",
            zorder=zorder + 1)


def kink(s, s_k, a, w):
    """A V-shaped dip of the orbit's radius at s_k: linear at the kink, fading smoothly away from it."""
    u = s - s_k
    return a * np.abs(u) * np.exp(-(u / w) ** 2)


def orbit(s):
    """The orbit in the isocline sheet: it winds around the bowl at a distance from the bottom; its radius has two
    kinks (the current sheets) and smooth dips, so the energy along it has minima of both kinds."""
    r = (0.8 + 0.06 * np.sin(1.7 * s + 0.4) + 0.04 * np.sin(4.3 * s + 2.0)
         + kink(s, -0.5, 1.1, 0.25) + kink(s, 0.2, 0.9, 0.35) - 0.08 * np.exp(-((s - 0.72) / 0.18) ** 2)
         - 0.06 * (1 + np.tanh((s - 1.35) / 0.15)))            # the left end runs down the wall
    return r, 0.5 * np.pi + 0.95 * s


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="scripts/paper_scripts/build", help="figs/ in the paper's layout [scripts/paper_scripts/build]")
    cli = ap.parse_args()

    s = np.linspace(-0.85, 1.65, 6001)                     # the orbit runs from the right around the far wall
    ro, po = orbit(s)
    xo, yo = ro * np.cos(po), ro * np.sin(po)
    Eo = sheet(E_TAYLOR, ro, po)
    mins = [i for i in range(1, len(s) - 1) if Eo[i] < Eo[i - 1] and Eo[i] <= Eo[i + 1]]
    i_b0 = int(np.argmin(np.abs(s - 0.02)))
    i = i_b0
    step = -1 if Eo[i - 1] < Eo[i + 1] else 1
    while 0 < i + step < len(s) - 1 and Eo[i + step] < Eo[i]:
        i += step
    i_eq = i
    assert len(mins) >= 3 and i_eq in mins and Eo[i_eq] > Eo.min() and Eo.max() < Z_TOP, \
        "the orbit does not have the claimed shape"

    with house_style():
        w, h = figsize("column", rows=1, cols=1)
        fig = plt.figure(figsize=(w, 1.3 * h))
        ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
        ax.view_init(elev=ELEV, azim=AZIM)
        ax.set_box_aspect((1, 1, 0.7))
        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-1.4, 1.4)
        ax.set_zlim(0, Z_TOP + 0.1)
        ax.set_axis_off()

        def frac(x, y, z):
            """The axes-fraction position of a 3-D point (for placing 2-D text relative to 3-D content)."""
            xs, ys, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
            return ax.transAxes.inverted().transform(ax.transData.transform((xs, ys)))

        # the ground plane at the vacuum's energy, and the energy axis rising from it
        L, L_FAR = 1.15, 0.85
        ax.plot_surface(np.array([[-L, L], [-L, L]]), np.array([[-L, -L], [L_FAR, L_FAR]]), np.zeros((2, 2)),
                        color="0.92", alpha=0.55, linewidth=0, shade=False, zorder=-1)
        xa, ya = -L + 0.12, -L + 0.12                     # the arrow in the plane's near left corner
        ax.quiver(xa, ya, 0.0, 0.0, 0.0, 0.9, color="0.35", lw=0.9, arrow_length_ratio=0.17, zorder=8)
        ax.text(xa - 0.07, ya, 0.45, r"$\mathcal{E}$", color="0.35", ha="right", va="center", **LABEL)

        # the isocline of zero helicity, the vacuum at its bottom
        bowl(ax, 0.0, "0.9", 0.12, "0.7", zorder=0)
        ax.plot([0], [0], [0], marker="o", ms=3.5, color=BLACK, ls="none", zorder=2)
        ax.text(0.08, -0.06, -0.02, r"$B_{\mathrm{vacuum}}$", ha="left", va="top", **LABEL)
        x_db = 0.805                                      # the delta-B label's column; H = 0 sits straight above it
        ax.text2D(x_db, frac(-0.65, 0.95, Z_TOP + 0.08)[1], r"$\mathcal{H} = 0$", transform=ax.transAxes,
                  fontsize=FS, color="0.35", ha="center", va="bottom", zorder=8)

        # the isocline of B_0, the Taylor state at its bottom
        bowl(ax, E_TAYLOR, "0.82", 0.3, "0.5", zorder=3)
        ax.plot([0], [0], [E_TAYLOR], marker="o", ms=3.5, color=BLACK, ls="none", zorder=6)
        ax.text(0.08, -0.06, E_TAYLOR - 0.02, r"$B_{\mathrm{Taylor}}$", ha="left", va="top", **LABEL)
        ax.text(-0.6, 0.95, sheet(E_TAYLOR, 1.12, 2.13) + 0.05, r"$\mathcal{H} = \mathrm{const.}$", color="0.35",
                ha="center", va="center", **LABEL)

        # the orbit on the sheet (the topology constraint), fading out at both ends: it does not end there
        n_seg, n_fade = 80, 8
        bounds = np.linspace(0, len(s) - 1, n_seg + 1).astype(int)
        for k, (a_, b_) in enumerate(zip(bounds[:-1], bounds[1:])):
            alpha = min(1.0, (k + 1) / n_fade, (n_seg - k) / n_fade)
            ax.plot(xo[a_:b_ + 1], yo[a_:b_ + 1], Eo[a_:b_ + 1], color=TEAL, lw=1.5, ls="-", alpha=alpha,
                    solid_capstyle="round", zorder=5)
        # the delta-B label, shifted right by the B_eq-to-B_0 distance
        dx = frac(xo[i_b0], yo[i_b0], Eo[i_b0])[0] - frac(xo[i_eq], yo[i_eq], Eo[i_eq])[0]
        ax.text2D(x_db + dx, 0.64, r"$\delta B = \mathrm{curl}\,(v \times B)$", transform=ax.transAxes, fontsize=FS,
                  color=TEAL, ha="center", va="center",
                  bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85), zorder=8)
        ax.plot([xo[i_b0]], [yo[i_b0]], [Eo[i_b0]], marker="o", ms=4.5, color="white", mec=BLACK, mew=0.9, ls="none",
                zorder=6)
        ax.text(xo[i_b0], yo[i_b0], Eo[i_b0] + 0.08, "$B_0$", ha="center", va="bottom", **LABEL)
        ax.plot([xo[i_eq]], [yo[i_eq]], [Eo[i_eq]], marker="o", ms=5, color=PURPLE, mec=BLACK, mew=0.6, ls="none",
                zorder=7)
        ax.text(xo[i_eq], yo[i_eq], Eo[i_eq] - 0.13, "$B_{\\mathrm{eq}}$", color=PURPLE, ha="center", va="top", **LABEL)

        # crop to the drawn content: a 3-D axes' own box counts as content for bbox_inches="tight"
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        rows, cols = np.where(np.any(rgba[:, :, :3] < 250, axis=2))
        pad = 0.03 * fig.dpi
        box = Bbox.from_extents((cols.min() - pad) / fig.dpi, (rgba.shape[0] - rows.max() - pad) / fig.dpi,
                                (cols.max() + pad) / fig.dpi, (rgba.shape[0] - rows.min() + pad) / fig.dpi)
        figs, pgf = os.path.join(cli.out, "figs"), os.path.join(cli.out, "figs", "pgf", "energy_landscape")
        os.makedirs(pgf, exist_ok=True)
        fig.savefig(os.path.join(figs, "energy_landscape.png"), dpi=300, bbox_inches=box)
        fig.savefig(os.path.join(figs, "energy_landscape.pdf"), bbox_inches=box)
        with matplotlib.rc_context({"pgf.texsystem": "pdflatex", "pgf.rcfonts": False,
                                    "pgf.preamble": r"\providecommand{\mathdefault}[1]{#1}"}):
            fig.savefig(os.path.join(pgf, "energy_landscape.pgf"), backend="pgf", bbox_inches=box)
        plt.close(fig)
    print(f"wrote {figs}/energy_landscape.pdf, {pgf}/energy_landscape.pgf")


if __name__ == "__main__":
    main()
