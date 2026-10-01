"""The house matplotlib style of MRX figures.

* :func:`house_style` applies the settings of ``mrx/diagnostics/mrx.mplstyle`` inside a ``with house_style():``
  block or a function decorated with ``@house_style()``: serif fonts, the font sizes :data:`FS`, a colour cycle
  with one dash style per colour, inward ticks and constrained layout. Outside the block the caller's
  matplotlib settings are untouched, so importing mrx never changes them.
* The named colours and colormaps: :data:`LEFT` and :data:`RIGHT` for twin-axis plots, :data:`IOTA_COLOR` and
  :data:`P_COLOR` for section figures, and :data:`FIELD_CMAP`, :data:`PRESSURE_CMAP` and :data:`SECTION_CMAP`.
* :func:`save_figure` saves a figure as a PNG and, when ``xelatex`` is available, as a ``.pgf`` file for LaTeX
  in a ``pgf/`` folder next to it.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass

import matplotlib as mpl

#: The rcParams file behind :func:`house_style`.
STYLE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mrx.mplstyle")

#: Colour of iota in the profile panel of a section figure.
IOTA_COLOR = "black"
#: Colour of the pressure in the profile panel of a section figure.
P_COLOR = "#6a3d9a"
#: Left trace of :func:`mrx.diagnostics.plotting.plot_twin_axis`.
LEFT = dict(color="black", marker="s", linestyle="-", markersize=4)
#: Right trace of :func:`mrx.diagnostics.plotting.plot_twin_axis`.
RIGHT = dict(color="teal", marker="d", linestyle="--", markersize=4)

#: Scalar fields on the torus.
FIELD_CMAP = "plasma"
#: Pressure.
PRESSURE_CMAP = "plasma"
#: iota on a Poincare section. A rainbow map makes neighbouring surfaces differ in hue.
SECTION_CMAP = "gist_rainbow"


@dataclass(frozen=True)
class FontScale:
    """The font sizes of MRX figures, in points. The style sheet ``mrx.mplstyle`` uses the same numbers."""

    title: float = 11.0
    label: float = 11.0
    tick: float = 9.0
    annot: float = 7.5     # in-axes annotations, rational labels, the legend
    big: float = 14.0      # the 3-D torus axis labels


FS = FontScale()


@contextmanager
def house_style(**overrides):
    """Apply the house rcParams for the duration of a ``with`` block or a decorated function. ``overrides``
    change individual rcParams on top of the house style."""
    with mpl.rc_context(rc=overrides, fname=STYLE_FILE):
        yield


@house_style()
def save_figure(fig, png_path, dpi=None):
    """Save ``fig`` as ``png_path`` and as ``pgf/<stem>.pgf`` in the same folder. The ``.pgf`` keeps text and
    lines as vectors, with rasterized parts saved as PNGs next to it. It needs ``xelatex`` on the PATH and is
    skipped with a message otherwise. A LaTeX document that includes it needs ``\\usepackage[strings]{underscore}``
    and ``\\providecommand{\\mathdefault}[1]{#1}`` in its preamble."""
    fig.savefig(png_path, dpi=dpi)
    pgf_dir = os.path.join(os.path.dirname(png_path), "pgf")
    os.makedirs(pgf_dir, exist_ok=True)
    pgf_path = os.path.join(pgf_dir, os.path.splitext(os.path.basename(png_path))[0] + ".pgf")
    try:
        with mpl.rc_context({"pgf.preamble": r"\usepackage[strings]{underscore}\providecommand{\mathdefault}[1]{#1}"}):
            fig.savefig(pgf_path, backend="pgf", dpi=dpi)
    except Exception as exc:      # noqa: BLE001 -- the .pgf is an optional artifact
        if os.path.exists(pgf_path):
            os.remove(pgf_path)   # a half-written .pgf is not a usable file
        print(f"  (pgf skipped -- needs xelatex on PATH: {type(exc).__name__}: {str(exc)[:80]})", flush=True)
