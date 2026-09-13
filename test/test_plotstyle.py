"""House matplotlib style: palette, sizes, and the scoped rc context."""
from __future__ import annotations

import matplotlib as mpl

from mrx.plotstyle import (
    COLUMN_WIDTH,
    CYCLE,
    DASHES,
    PANEL_ASPECT,
    TEXT_WIDTH,
    FontScale,
    SectionLimits,
    arm_style,
    figsize,
    house_style,
)


def test_arm_style_indexes_the_palette_and_wraps() -> None:
    """``arm_style(1, 2)`` is teal dash-dot; indices wrap past the cycle length."""
    style = arm_style(1, 2, lw=2.0)
    assert style["color"] == CYCLE[1][0]
    assert style["linestyle"] == "-."
    assert style["lw"] == 2.0
    n_colour, n_dash = len(CYCLE), len(DASHES)
    for colour in range(n_colour + 1):
        for dash in range(n_dash + 1):
            got = arm_style(colour, dash)
            assert got["color"] == CYCLE[colour % n_colour][0]
            assert got["linestyle"] == DASHES[dash % n_dash]


def test_figsize_named_and_numeric_widths() -> None:
    w_col, h_col = figsize("column")
    assert w_col == COLUMN_WIDTH
    assert h_col == COLUMN_WIDTH * PANEL_ASPECT
    w_text, _ = figsize("text", rows=2, cols=2)
    assert w_text == TEXT_WIDTH
    w_in, h_in = figsize(5.0, rows=2, cols=1)
    assert w_in == 5.0
    assert h_in == 5.0 * PANEL_ASPECT * 2.0


def test_font_scale_and_section_limits_defaults() -> None:
    fs = FontScale()
    assert fs.title == 11.0 and fs.tick == 9.0
    limits = SectionLimits(iota=(0.3, 1.0), RZ=((0.5, 1.5), (-1.0, 1.0)))
    assert limits.iota == (0.3, 1.0)
    assert limits.z_split is None


def test_house_style_scopes_rcparams() -> None:
    before = mpl.rcParams["font.size"]
    with house_style():
        inside = mpl.rcParams["font.size"]
    assert mpl.rcParams["font.size"] == before
    assert inside > 0
