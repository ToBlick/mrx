r"""Render the Poincare sections of a trace archive -- the cheap half, no GPU.

Reads the ``trace.npz`` written by ``scripts/poincare_trace.py`` and renders every field and plane it
holds -- or ``--fields`` / ``--planes``, subsets -- with ONE iota and ONE pressure scale over the call
(or the ``--iota-lim`` / ``--p-lim`` given, so separate calls match), each section in the box that
fits it (or ``--window``), as PDF + PNG pages ``poincare[_<field>]_zeta<plane>`` in ``--out``
[``<archive dir>/poincare``], in the publication layout: no title, the house font hierarchy at
``--label-size`` for a ``--page-width`` figure, the crossings rasterised at ``--dpi``; ``--pgf`` adds
the same figure through the pgf backend for ``\input`` (needs a TeX Live on PATH). Plain matplotlib, on
the login node in seconds per page. The pressure is the archived weak pressure normalised by the
field's mean magnetic pressure, ``p_norm = p / <B^2 / 2>`` (``--no-pressure`` leaves it out); the
resonant iota ticks are picked from the iota range shown.

    python scripts/poincare_plot.py outputs/run/trace.npz
"""
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

#: The pressure panel's label: the weak pressure over the field's mean magnetic pressure.
PRESSURE_LABEL = r"$p_{\mathrm{norm}}$"


@dataclass(frozen=True)
class Plot:
    """Render the Poincare sections of a trace archive."""
    archive: str = field(metadata=dict(positional=True, help="trace.npz, or the directory holding it"))
    fields: Optional[str] = field(default=None, metadata=dict(help="comma-separated subset of the archive's fields [all]"))
    planes: Optional[str] = field(default=None, metadata=dict(help="comma-separated subset of the archive's planes [all]"))
    out: Optional[str] = field(default=None, metadata=dict(help="page directory [<archive dir>/poincare]"))
    pressure: bool = field(default=True, metadata=dict(
        help="draw the normalised weak pressure (below the axis in the section and the chart, the profile's right axis)"))
    profile_rays: int = field(default=1, metadata=dict(
        help="poloidal rays on the logical profile (theta = 0.5, then 1/3, 0.2, ...), one marker each"))
    dot_scale: float = field(default=0.15, metadata=dict(
        help="crossing-marker size relative to the house rule; 0.15 for a dense section on a page"))
    label_size: float = field(default=9.0, metadata=dict(
        help="axis-label pt at --page-width, the paper's body size; ticks and legends keep the house hierarchy"))
    page_width: float = field(default=6.5, metadata=dict(help="authored width in inches"))
    dpi: int = field(default=600, metadata=dict(help="resolution of the rasterised crossings"))
    pgf: bool = field(default=False, metadata=dict(help="also the .pgf of each page, under pgf/ (needs a TeX Live)"))
    window: Optional[str] = field(default=None, metadata=dict(
        help="Rmin,Rmax,Zmin,Zmax: pin the section box to this window on every page"))
    iota_lim: Optional[str] = field(default=None, metadata=dict(
        help="LO,HI: fix the iota colour scale and profile axis instead of the call's own range"))
    p_lim: Optional[str] = field(default=None, metadata=dict(
        help="LO,HI: fix the pressure colour scale and profile axis (units of the colour bar, p_norm x 100)"))


def main(cli):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mrx.plotting import paper_fonts, render_section
    from mrx.plotstyle import SectionLimits
    from mrx.poincare import surface_label

    path = cli.archive if cli.archive.endswith(".npz") else os.path.join(cli.archive, "trace.npz")
    z = np.load(path)
    sec = {k: z[k] for k in z.files}
    out = cli.out or os.path.join(os.path.dirname(os.path.abspath(path)), "poincare")
    os.makedirs(out, exist_ok=True)
    all_fields = [str(f) for f in sec["fields"]]
    which = [w.strip() for w in cli.fields.split(",")] if cli.fields else all_fields
    for n in which:
        assert n in all_fields, (n, all_fields)
    planes = [float(v) for v in sec["planes"]]
    if cli.planes:
        planes = [pl for pl in planes if any(abs(pl - float(v)) < 1e-9 for v in cli.planes.split(","))]
    nfp = int(sec["nfp"])
    print(f"[plot] {path}: {str(sec['source'])}; fields {which}, planes {planes}"
          + ("" if cli.pressure else ", no pressure"), flush=True)

    per = {n: {k: np.asarray(sec[f"{n}_{k}"])
               for k in ("iota", "iota_err", "iota_scatter", "seed_r", "keep", "chaotic", "shown")}
           for n in which}
    for n in which:
        print(f"[{n}] {int((~per[n]['keep']).sum())}/{per[n]['keep'].size} lost, "
              f"{int((per[n]['keep'] & per[n]['chaotic']).sum())} chaotic", flush=True)
    # ONE iota scale and ONE p scale over every field and plane of the call
    iota_lim = (min(float(per[m]["iota"][per[m]["shown"]].min()) for m in which if per[m]["shown"].any()),
                max(float(per[m]["iota"][per[m]["shown"]].max()) for m in which if per[m]["shown"].any()))
    if cli.iota_lim:
        iota_lim = tuple(float(v) for v in cli.iota_lim.split(","))
    cuts = {(n, pl): tuple(np.asarray(sec[f"{n}_zeta{pl:g}_{k}"])
                           for k in ("R", "Z", "axisR", "axisZ", "logr", "logth"))
            for n in which for pl in planes}
    # the archive holds the weak pressure at every crossing: normalised here by the field's mean magnetic pressure
    presses = {n: {pl: (np.asarray(sec[f"{n}_zeta{pl:g}_pressure"]) / (0.5 * float(sec[f"{n}_bsq"]))
                        if cli.pressure else None) for pl in planes} for n in which}
    ps = [100.0 * presses[m][pl][per[m]["keep"]] for m in which for pl in planes if presses[m][pl] is not None]
    p_lim = None
    if ps:
        lo_p, hi_p = min(float(np.nanmin(v)) for v in ps), max(float(np.nanmax(v)) for v in ps)
        p_lim = (lo_p - 0.05 * (hi_p - lo_p), hi_p + 0.05 * (hi_p - lo_p))
    if cli.p_lim:
        p_lim = tuple(float(v) for v in cli.p_lim.split(","))
    limits = {}
    if cli.window:
        r0, r1, z0, z1 = (float(v) for v in cli.window.split(","))
        limits["RZ"] = ((r0, r1), (z0, z1))
    for n in which:
        for pl in planes:
            R, Z, aR, aZ, lr, lth = cuts[n, pl]
            a_eff, xlabel = surface_label(R, Z, aR, aZ)
            fig, _ = render_section(
                R, Z, per[n]["iota"], per[n]["iota_err"], per[n]["seed_r"], per[n]["keep"],
                pressure=presses[n][pl], pressure_label=PRESSURE_LABEL,
                axis_RZ=(aR, aZ), axis_marker=False, dot_scale=cli.dot_scale,
                profile_x=a_eff, profile_xlabel=xlabel, nfp=nfp, logical=(lr, lth),
                limits=SectionLimits(iota=iota_lim, p=p_lim, **limits),
                iota_scatter=per[n]["iota_scatter"], profile_rays=cli.profile_rays)
            infix = "" if len(all_fields) == 1 else f"_{n}"
            stem = os.path.join(out, f"poincare{infix}_zeta{pl:g}")
            paper_fonts(fig, label_size=cli.label_size, page_width=cli.page_width)
            # tight: the layout's left margin (colour bars) is cropped here, not by the including document
            tight = dict(bbox_inches="tight", pad_inches=0.02)
            fig.savefig(stem + ".pdf", dpi=cli.dpi, **tight)     # dpi sets the rasterised scatter
            fig.savefig(stem + ".png", dpi=cli.dpi, **tight)     # for viewing
            print(f"  -> {stem}.pdf", flush=True)
            if cli.pgf:
                # The same figure for \input into a document: typeset by pdflatex in the document's own fonts
                # (rcfonts off), the scatter a raster -img*.png beside it, under pgf/.
                pgf_dir = os.path.join(out, "pgf")
                os.makedirs(pgf_dir, exist_ok=True)
                with matplotlib.rc_context({"pgf.texsystem": "pdflatex", "pgf.rcfonts": False,
                                            "pgf.preamble": r"\providecommand{\mathdefault}[1]{#1}"}):
                    fig.savefig(os.path.join(pgf_dir, os.path.basename(stem) + ".pgf"), backend="pgf",
                                dpi=cli.dpi, **tight)
                print(f"  -> {pgf_dir}/{os.path.basename(stem)}.pgf", flush=True)
            plt.close(fig)


if __name__ == "__main__":
    from mrx.cli import parse
    sys.exit(main(parse(Plot, description=__doc__)))
