"""House matplotlib style for MRX figures: one palette, one font scale, one dpi.

Every figure the repository writes goes through this module, so that a
tutorial plot, a research-note trace and a publication panel look like they
come from the same place. Use it three ways:

* ``with house_style():`` (or ``@house_style()``) around a figure loads the
  rcParams of ``mrx/mrx.mplstyle`` (fonts, sizes, the line cycle, tick and
  layout settings) for the duration of the block, scoped through
  ``mpl.rc_context`` so importing :mod:`mrx.plotting` never mutates a
  caller's global matplotlib state;
* the named choices are here: the palette (:data:`BLACK`, :data:`TEAL`,
  :data:`PURPLE`, :data:`GREY`; :data:`LEFT` / :data:`RIGHT` for twin axes,
  :data:`IOTA_COLOR` / :data:`P_COLOR` for the section pages), the
  colormaps (:data:`FIELD_CMAP`, :data:`SECTION_CMAP`, :data:`PRESSURE_CMAP`)
  and the figure widths (:func:`figsize`);
* :class:`SectionLimits` pins every scale of a Poincaré
  :func:`mrx.plotting.render_section` so a movie or a side-by-side set is
  comparable frame to frame.

Conventions (settled 2026-09-04, see docs/research/relaxation figure notes):

* **Colours** are black, teal, purple, grey, in that order, never the default
  matplotlib cycle. Black is the run of interest or the left axis, teal the
  reference or the second quantity (dashed), purple the pressure. The line
  cycle pairs each colour with its own dash (``-``, ``--``, ``-.``, ``:``) so
  a greyscale print still separates the arms. Two-factor comparisons encode
  one factor in colour and the other in dash: :func:`arm_style`.
* **Per-step traces** are 100-step block means with a +-1 sd band, in log
  space on log axes (``blocked`` / ``plot_trace`` in
  ``scripts/li383_pub_figures.py``); never raw per-step lines, never running
  means. Events (reconnections, pulses) are dotted grey verticals.
* **Widths**: :data:`COLUMN_WIDTH` (3.4 in, one journal column) and
  :data:`TEXT_WIDTH` (7.0 in, both columns); :func:`figsize` derives the
  height. Legends inside the axes, ``framealpha`` 0.85.
* **Output**: PNG at 200 dpi and, for publication figures, the same figure
  through the pgf backend into a ``pgf/`` subfolder next to the PNG, written
  by the one writer :func:`mrx.plotting.save_figure` (which adds the two
  preamble lines the including document also needs). Per-run figures live in
  the run's own directory, comparison figures in ``outputs/<study>/figures/``.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import matplotlib as mpl

#: The rcParams file behind :func:`house_style`.
STYLE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mrx.mplstyle")

# --- palette ----------------------------------------------------------------
BLACK = "black"
TEAL = "teal"
PURPLE = "#6a3d9a"
GREY = "0.5"
#: The line cycle of the style sheet: (colour, dash) pairs, in order.
CYCLE = ((BLACK, "-"), (TEAL, "--"), (PURPLE, "-."), (GREY, ":"))
#: Dashes for a second factor in :func:`arm_style`.
DASHES = ("-", "--", "-.", ":")
#: iota on a section page and its profile.
IOTA_COLOR = BLACK
#: Pressure on a section page, the beta line, the second profile.
P_COLOR = PURPLE
#: Event markers (reconnections, resistive pulses): dotted grey verticals.
EVENT = dict(color="0.6", linestyle=":", linewidth=0.8)

#: Left trace of :func:`mrx.plotting.plot_twin_axis` (and the iota profile).
LEFT = dict(color=BLACK, marker="s", linestyle="-", markersize=4)
#: Right trace (and the pressure profile).
RIGHT = dict(color=TEAL, marker="d", linestyle="--", markersize=4)


def arm_style(colour: int = 0, dash: int = 0, **kw) -> dict:
    """Line keywords for arm ``(colour, dash)`` of a two-factor comparison.

    ``colour`` indexes the palette (black, teal, purple, grey), ``dash`` the
    dashes (solid, dashed, dash-dot, dotted); a one-factor comparison passes
    the same index to both and follows the style sheet's cycle. ``kw`` adds
    or overrides (``label``, ``lw``, ...).
    """
    return dict(color=CYCLE[colour % len(CYCLE)][0], linestyle=DASHES[dash % len(DASHES)], **kw)


# --- colormaps -------------------------------------------------------------
#: Scalar fields on the torus (a magnitude read against a floor).
FIELD_CMAP = "plasma"
#: Pressure -- a magnitude with a zero, read like the field.
PRESSURE_CMAP = "plasma"
#: iota on a Poincaré section: one hue per nested surface. ``gist_rainbow``
#: rather than a luminance ramp so ADJACENT surfaces separate by hue, which is
#: what the eye follows on a stack of discrete curves.
SECTION_CMAP = "gist_rainbow"

# --- sizes -------------------------------------------------------------------
#: One column of a two-column journal page, inches.
COLUMN_WIDTH = 3.4
#: Both columns (the text width), inches.
TEXT_WIDTH = 7.0
#: Height / width of one panel: the golden ratio.
PANEL_ASPECT = 0.618

DPI = 200


def figsize(width: float | str = "column", rows: int = 1, cols: int = 1,
            aspect: float = PANEL_ASPECT) -> tuple[float, float]:
    """``(width, height)`` in inches for a ``rows x cols`` grid of panels.

    ``width`` is ``"column"``, ``"text"`` or a number of inches; each panel is
    ``aspect`` times as high as it is wide.
    """
    w = {"column": COLUMN_WIDTH, "text": TEXT_WIDTH}.get(width, width)
    return (w, w / cols * aspect * rows)


@dataclass(frozen=True)
class FontScale:
    """One coherent scale, in points; the style sheet carries the same numbers."""

    title: float = 11.0
    label: float = 11.0
    tick: float = 9.0
    annot: float = 7.5     # in-axes annotations, Farey labels, the legend
    big: float = 14.0      # the 3-D torus axis labels, which want more air


FS = FontScale()


@contextmanager
def house_style(**overrides):
    """Apply the house rcParams (``mrx/mrx.mplstyle``) for the duration of a
    ``with`` block; also usable as a decorator.

    Scoped (``mpl.rc_context``) so importing :mod:`mrx.plotting` never mutates a
    caller's global matplotlib state. ``overrides`` patch individual keys.
    """
    with mpl.rc_context(rc=overrides, fname=STYLE_FILE):
        yield


@dataclass
class SectionLimits:
    """Pinned scales for a Poincaré :func:`mrx.plotting.render_section`.

    Any field left ``None`` is fitted from the figure's own data; the ones you
    set are held fixed, which is what makes a movie or a side-by-side set
    comparable (the same iota hue is the same transform in every frame).

    * ``RZ``      -- ``((R0, R1), (Z0, Z1))`` of the section panel;
    * ``z_split`` -- the iota/p dividing line (the magnetic-axis ``Z``);
    * ``x``       -- the profiles' abscissa range;
    * ``p``       -- the pressure panel's ordinate range, in drawn units;
    * ``iota``    -- ``(lo, hi)`` of the iota colour and profile scale.
    """

    RZ: Optional[tuple] = None
    z_split: Optional[float] = None
    x: Optional[tuple] = None
    p: Optional[tuple] = None
    iota: Optional[tuple] = None

    @classmethod
    def coerce(cls, limits=None, iota_lim=None):
        """Normalise the legacy ``(limits dict, iota_lim)`` call into one object."""
        if isinstance(limits, cls):
            lim = cls(limits.RZ, limits.z_split, limits.x, limits.p, limits.iota)
        elif limits:
            lim = cls(**{k: limits[k] for k in ("RZ", "z_split", "x", "p", "iota")
                         if k in limits})
        else:
            lim = cls()
        if iota_lim is not None and lim.iota is None:
            lim.iota = tuple(iota_lim)
        return lim
