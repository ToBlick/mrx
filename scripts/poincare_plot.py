r"""Draw the Poincare sections of a trace archive. This is the cheap half and needs no GPU.

The script reads the ``trace.npz`` written by ``scripts/poincare_trace.py`` and draws every field and plane it
holds, or the subsets chosen with ``--fields`` and ``--planes``. All pages of one call share ONE iota colour scale
and ONE pressure scale. ``--iota-lim`` and ``--p-lim`` fix these scales, so that separate calls match. Each section
is drawn in the box that fits it, or in ``--window``.

Each page is written as PDF and PNG, named ``poincare[_<field>]_zeta<plane>``, into ``--out`` (default
``<archive dir>/poincare``). The field name appears only when the archive holds more than one field. The layout
is the publication one: no title, the house font sizes scaled to ``--label-size`` for a figure ``--page-width``
wide, and the crossings rasterised at ``--dpi``. ``--pgf`` also writes the figure through the pgf backend, for
``\input`` into a LaTeX document (this needs TeX Live on the PATH). The script uses plain matplotlib and runs on
the login node in seconds per page.

The pressure shown is the archived weak pressure divided by the field's mean magnetic pressure,
``p_norm = p / <B^2 / 2>``. ``--no-pressure`` leaves it out. The resonant iota values marked on the colour bar are
picked from the iota range shown.

    python scripts/poincare_plot.py outputs/run/trace.npz
"""
import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import tyro

@dataclass(frozen=True)
class Plot:
    """Draw the Poincare sections of a trace archive."""
    archive: tyro.conf.Positional[str]
    """The trace.npz, or the directory holding it."""
    fields: Optional[tuple[str, ...]] = None
    """A subset of the archive's fields. Unset, all of them are drawn."""
    planes: Optional[tuple[float, ...]] = None
    """A subset of the archive's planes. Unset, all of them are drawn."""
    out: Optional[str] = None
    """The directory for the pages. Unset, it is <archive dir>/poincare."""
    pressure: bool = True
    """Draw the normalised weak pressure (below the section and the chart, and on the right axis of the profile)."""
    profile_rays: int = 1
    """The number of poloidal rays in the logical profile (theta = 0.5, then 1/3, 0.2, ...), each with its own marker."""
    dot_scale: float = 0.15
    """The size of the crossing markers relative to the house default. Use 0.15 for a dense section on a page."""
    label_size: float = 9.0
    """The axis-label size in pt at --page-width, the body size of the paper. Ticks and legends are scaled with it."""
    page_width: float = 6.5
    """The width of the figure in inches."""
    dpi: int = 600
    """The resolution of the rasterised crossings."""
    pgf: bool = False
    """Also write the .pgf of each page into pgf/ (needs TeX Live)."""
    window: Optional[tuple[float, float, float, float]] = None
    """Rmin Rmax Zmin Zmax, the box of the section on every page."""
    iota_lim: Optional[tuple[float, float]] = None
    """LO HI, a fixed range for the iota colour scale and profile axis instead of the range of the call."""
    p_lim: Optional[tuple[float, float]] = None
    """LO HI, a fixed range for the pressure colour scale and profile axis, in the units of the colour bar (p_norm x 100)."""


def main(cli):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mrx.diagnostics.plotting import plot_archive

    path = cli.archive if cli.archive.endswith(".npz") else os.path.join(cli.archive, "trace.npz")
    z = np.load(path)
    archive = {k: z[k] for k in z.files}
    print(f"[plot] {path}: {str(archive['source'])}, fields {[str(f) for f in archive['fields']]}, planes "
          f"{[float(v) for v in archive['planes']]}" + ("" if cli.pressure else ", no pressure"), flush=True)
    for n in (cli.fields or [str(f) for f in archive["fields"]]):
        keep, chaotic = archive[f"{n}_keep"], archive[f"{n}_chaotic"]
        print(f"[{n}] {int((~keep).sum())}/{keep.size} lost, {int((keep & chaotic).sum())} chaotic", flush=True)
    window = None
    if cli.window:
        r0, r1, z0, z1 = cli.window
        window = ((r0, r1), (z0, z1))
    figs = plot_archive(archive, cli.out or os.path.join(os.path.dirname(os.path.abspath(path)), "poincare"),
                        fields=cli.fields, planes=cli.planes, pressure=cli.pressure, profile_rays=cli.profile_rays,
                        dot_scale=cli.dot_scale, label_size=cli.label_size, page_width=cli.page_width, dpi=cli.dpi,
                        pgf=cli.pgf, window=window, iota_lim=cli.iota_lim, p_lim=cli.p_lim)
    for fig in figs.values():
        plt.close(fig)


if __name__ == "__main__":
    sys.exit(main(tyro.cli(Plot, description=__doc__)))
