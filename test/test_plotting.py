"""Plotting contracts: arrays in, assertable figure structure out.

``render_section`` is documented as taking pure arrays so a run can
re-render from its archive. The branch matrix below walks the optional
arguments that hold most of the module's statements. Figures use the Agg
backend and are closed after each assertion.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from mrx.mappings import cylinder_map, toroid_map
from mrx.plotstyle import SectionLimits
from mrx.plotting import (
    _padded,
    get_2d_grids,
    paper_fonts,
    plot_crossections_separate,
    plot_torus,
    plot_twin_axis,
    render_section,
    resonant_rationals,
    save_figure,
    set_axes_equal,
)


def _concentric_section(n_lines: int = 6, n_cross: int = 40) -> dict:
    """Nested circles in ``(R, Z)`` with a monotone iota profile."""
    th = np.linspace(0.0, 2.0 * np.pi, n_cross, endpoint=False)
    rr = np.linspace(0.1, 0.9, n_lines)[:, None]
    keep = np.ones(n_lines, dtype=bool)
    keep[-1] = False
    return dict(
        R=1.0 + rr * np.cos(th),
        Z=rr * np.sin(th),
        iota=0.4 + 0.5 * rr[:, 0] ** 2,
        iota_err=np.full(n_lines, 1e-3),
        seed_r=rr[:, 0],
        keep=keep,
        logical=(np.broadcast_to(rr, (n_lines, n_cross)),
                 np.broadcast_to((th / (2.0 * np.pi))[None, :], (n_lines, n_cross))),
        pressure=np.broadcast_to(np.linspace(0.02, 0.0, n_lines)[:, None],
                                 (n_lines, n_cross)),
        axis_RZ=(np.full(4, 1.0), np.zeros(4)),
        rr=rr,
    )


def test_resonant_rationals_are_nfp_multiples_in_range() -> None:
    iota_min, iota_max, nfp, denom_max, min_sep = 0.2, 1.2, 3, 30, 0.06
    ticks, labels = resonant_rationals(iota_min, iota_max, nfp, denom_max, min_sep)
    span = iota_max - iota_min
    assert ticks == sorted(ticks)
    assert len(ticks) == 13
    for value, lab in zip(ticks, labels):
        n_tor, m_pol = (int(s) for s in lab.split("/"))
        assert n_tor % nfp == 0
        assert 1 <= m_pol <= denom_max
        assert iota_min <= value <= iota_max
        assert abs(n_tor / m_pol - value) < 1e-12
    for left, right in zip(ticks, ticks[1:]):
        assert right - left >= min_sep * span - 1e-12


def test_get_2d_grids_matches_the_map_and_the_ij_layout() -> None:
    """``y == F(x)`` pins the ``indexing='ij'`` order that avoids the star artifact."""
    f = cylinder_map()
    x, y, (y1, y2, y3), _ = get_2d_grids(f, cut_axis=2, cut_value=0.3, nx=6, ny=5, nz=1)
    assert x.shape == (30, 3) and y1.shape == (6, 5)
    np.testing.assert_allclose(np.asarray(y), np.asarray(jax.vmap(f)(x)), atol=1e-12)
    for axis in (0, 1, 2):
        grid = get_2d_grids(f, cut_axis=axis, cut_value=0.3, nx=6, ny=5, nz=4)
        assert grid[0].shape[1] == 3
    flipped = get_2d_grids(f, cut_axis=2, cut_value=0.3, nx=6, ny=5, nz=1, invert_z=True)
    assert flipped[0].shape == (30, 3)


def test_render_section_branch_matrix() -> None:
    s = _concentric_section()
    r, z, iota, err, seed_r, keep = s["R"], s["Z"], s["iota"], s["iota_err"], s["seed_r"], s["keep"]

    fig, axes = render_section(r, z, iota, err, seed_r, keep, title="t",
                               subtitle="s", nfp=3, axis_RZ=s["axis_RZ"])
    assert set(axes) == {"ax", "bx"}
    plt.close(fig)

    limits = SectionLimits(iota=(0.3, 1.0), RZ=((0.5, 1.5), (-1.0, 1.0)),
                           x=(0.0, 1.0), p=(0.0, 3.0), z_split=0.0)
    fig, axes = render_section(
        r, z, iota, err, seed_r, keep, logical=s["logical"], pressure=s["pressure"],
        axis_RZ=s["axis_RZ"], nfp=3, limits=limits, iota_scatter=err)
    assert set(axes) == {"ax", "lx", "bx"}
    plt.close(fig)

    fig, axes = render_section(
        r, z, iota, err, seed_r, keep, logical=s["logical"], axis_RZ=s["axis_RZ"],
        profile_coord="physical", rationals=["3/5", "6/7"], axis_marker=False, title=None)
    assert set(axes) == {"ax", "lx", "bx"}
    plt.close(fig)

    fig, _ = render_section(
        r, z, iota, err, seed_r, keep, logical=s["logical"], pressure=s["pressure"],
        axis_RZ=s["axis_RZ"], profile_rays=5, dot_scale=0.5, title="x")
    plt.close(fig)

    fig, _ = render_section(
        r, z, iota, err, seed_r, keep, nfp=3, axis_RZ=(1.0, 0.0),
        profile_x=np.stack([s["rr"][:, 0], s["rr"][:, 0] + 0.1], axis=1))
    plt.close(fig)

    fig, _ = render_section(
        r, np.zeros_like(z), np.full_like(iota, 0.5), err, seed_r,
        np.ones_like(keep), title="flat")
    plt.close(fig)


def test_render_section_split_requires_pressure_and_axis() -> None:
    s = _concentric_section()
    args = (s["R"], s["Z"], s["iota"], s["iota_err"], s["seed_r"], s["keep"])
    with pytest.raises(ValueError, match="pressure"):
        render_section(*args, split_iota_p=True)
    with pytest.raises(ValueError, match="axis_RZ"):
        render_section(*args, pressure=s["pressure"], split_iota_p=True)


def test_plot_torus_and_cross_sections_on_the_analytic_torus() -> None:
    f = toroid_map()
    grids_pol = [get_2d_grids(f, cut_axis=2, cut_value=z, nx=8, ny=8, nz=1)
                 for z in (0.0, 0.5)]
    grid_surface = get_2d_grids(f, cut_axis=0, cut_value=1.0 - 1e-6, nx=1, ny=8, nz=8)
    def field(x):
        return jnp.linalg.norm(x)
    fig, ax = plot_torus(field, grids_pol, grid_surface, cstride=2,
                         gridlinewidth=0.3, cbar_label="|x|", nfp=1)
    assert ax.name == "3d"
    plt.close(fig)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig, _ = plot_crossections_separate(field, grids_pol, [0.0, 0.5])
    plt.close(fig)


def test_plot_twin_axis_default_x_and_existing_axes() -> None:
    fig, (left, right) = plot_twin_axis(np.arange(1.0, 6.0), np.arange(1.0, 6.0),
                                        num_iters_inner=3)
    np.testing.assert_allclose(left.lines[0].get_xdata(), np.arange(5) * 3)
    plt.close(fig)
    fig, ax = plt.subplots()
    fig2, pair = plot_twin_axis(np.arange(1.0, 4.0), np.arange(1.0, 4.0),
                                left_log=False, right_log=True, grid=False, ax=ax)
    assert fig2 is fig and pair[0] is ax
    paper_fonts(fig)
    plt.close(fig)


def test_save_figure_png_only_and_pgf_skip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fig, _ = plot_twin_axis(np.arange(1.0, 4.0), np.arange(1.0, 4.0))
    png = tmp_path / "only.png"
    save_figure(fig, str(png), pgf=False)
    assert png.exists()
    assert not (tmp_path / "pgf").exists()

    png_b = tmp_path / "both.png"
    original = fig.savefig

    def boom(path, *args, **kwargs):
        if kwargs.get("backend") == "pgf" or str(path).endswith(".pgf"):
            Path(path).write_text("% half-written")
            raise RuntimeError("no xelatex")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(fig, "savefig", boom)
    save_figure(fig, str(png_b), pgf=True)
    assert png_b.exists()
    assert not (tmp_path / "pgf" / "both.pgf").exists()
    plt.close(fig)


def test_set_axes_equal_and_padded() -> None:
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 2.0)
    ax.set_zlim(-1.0, 1.0)
    set_axes_equal(ax)
    xr = np.diff(ax.get_xlim3d())[0]
    yr = np.diff(ax.get_ylim3d())[0]
    zr = np.diff(ax.get_zlim3d())[0]
    assert abs(xr - yr) < 1e-12 and abs(yr - zr) < 1e-12
    plt.close(fig)
    lo, hi = _padded(np.array([0.0, 1.0]), pad=0.0, floor=0.0)
    assert abs(lo) < 1e-12 and abs(hi - 1.0) < 1e-12
