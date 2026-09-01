"""House matplotlib style for MRX figures: one palette, one font scale, one dpi.

Centralises the aesthetic choices that were scattered across
:mod:`mrx.plotting` (three colormaps, per-function font sizes, a repeated
``dpi=200``). Use it three ways:

* ``with house_style():`` around a figure sets the shared rcParams (font sizes,
  grid, dpi) so a plotter need not repeat them;
* the exported colormaps (:data:`FIELD_CMAP`, :data:`SECTION_CMAP`,
  :data:`PRESSURE_CMAP`) and line palette (:data:`LEFT`, :data:`RIGHT`) are the
  named choices;
* :class:`SectionLimits` pins every scale of a Poincaré
  :func:`mrx.plotting.render_section` so a movie or a side-by-side set is
  comparable frame to frame -- the movie-pinning that used to be a loose dict.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import matplotlib as mpl

# --- colormaps -------------------------------------------------------------
#: Scalar fields on the torus (a magnitude read against a floor).
FIELD_CMAP = "plasma"
#: Pressure -- a magnitude with a zero, read like the field.
PRESSURE_CMAP = "plasma"
#: iota on a Poincaré section: one hue per nested surface. ``gist_rainbow``
#: rather than a luminance ramp so ADJACENT surfaces separate by hue, which is
#: what the eye follows on a stack of discrete curves.
SECTION_CMAP = "gist_rainbow"

# --- line / twin-axis palette ---------------------------------------------
#: Left trace of :func:`mrx.plotting.plot_twin_axis` (and the iota profile).
LEFT = dict(color="black", marker="s", linestyle="-", markersize=4)
#: Right trace (and the pressure profile).
RIGHT = dict(color="teal", marker="d", linestyle="--", markersize=4)

DPI = 200


@dataclass(frozen=True)
class FontScale:
    """One coherent scale, in points, replacing the per-function sizes."""

    title: float = 11.0
    label: float = 11.0
    tick: float = 9.0
    annot: float = 7.5     # in-axes annotations, Farey labels, the legend
    big: float = 14.0      # the 3-D torus axis labels, which want more air


FS = FontScale()

_RC = {
    "savefig.dpi": DPI,
    "figure.dpi": 110,
    "axes.titlesize": FS.title,
    "axes.labelsize": FS.label,
    "xtick.labelsize": FS.tick,
    "ytick.labelsize": FS.tick,
    "legend.fontsize": FS.annot,
    "legend.framealpha": 0.85,
    "axes.grid": False,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.0,
    "font.size": FS.label,
}


@contextmanager
def house_style(**overrides):
    """Apply the shared rcParams for the duration of a ``with`` block.

    Scoped (``mpl.rc_context``) so importing :mod:`mrx.plotting` never mutates a
    caller's global matplotlib state. ``overrides`` patch individual keys.
    """
    with mpl.rc_context({**_RC, **overrides}):
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
