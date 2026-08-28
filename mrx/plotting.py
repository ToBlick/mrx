"""Matplotlib figures of scalar fields on a torus and of relaxation traces.

* :func:`get_2d_grids` samples a map on a logical plane (a poloidal cut at
  fixed zeta, or the boundary surface at fixed r).
* :func:`plot_torus` draws the boundary surface as a wireframe with poloidal
  cuts coloured by a scalar; :func:`plot_crossections_separate` draws the
  same cuts side by side in the ``(R, z)`` plane.
* :func:`plot_twin_axis` draws two traces against a shared x with separate
  y axes -- the standard figure for a force residual next to an energy or a
  helicity.

Poincaré sections live in :mod:`mrx.poincare`; ``scripts/plot_relaxation.py``
makes the figures here from a ``scripts/relax.py`` run.
"""

from typing import Callable, Optional

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

import mrx

#: Colour map of every scalar-field figure in this module.
FIELD_CMAP = "plasma"


def get_2d_grids(
    F: Callable,
    cut_value: float = 0,
    cut_axis: int = 2,
    nx: int = 64,
    ny: int = 64,
    nz: int = 64,
    tol1: float = 1e-6,
    tol2: float = 0,
    tol3: float = 0,
    x_min: float = 0,
    x_max: float = 1,
    y_min: float = 0,
    y_max: float = 1,
    z_min: float = 0,
    z_max: float = 1,
    invert_x: bool = False,
    invert_y: bool = False,
    invert_z: bool = False,
):
    """Sample the map ``F`` on the logical plane ``x_{cut_axis} = cut_value``.

    The other two logical axes are ``linspace(min + tol, max - tol, n)``,
    optionally reversed (``invert_*``) to orient a surface's normal. The
    radial default ``tol1 = 1e-6`` keeps the sample off the polar axis and
    off ``r = 1``, where the spline map's derivative is not defined.

    Returns ``(x, y, (Y1, Y2, Y3), (x1, x2, x3))``: the flat logical points
    ``x`` (``(n1 n2, 3)``), their images ``y = F(x)``, the images reshaped
    to the ``(n1, n2)`` plane for ``plot_surface``/``contourf``, and the
    three 1-D logical axes.
    """
    _x1 = jnp.linspace(x_min + tol1, x_max - tol1, nx)
    _x2 = jnp.linspace(y_min + tol2, y_max - tol2, ny)
    _x3 = jnp.linspace(z_min + tol3, z_max - tol3, nz)
    if invert_x:
        _x1 = _x1[::-1]
    if invert_y:
        _x2 = _x2[::-1]
    if invert_z:
        _x3 = _x3[::-1]
    if cut_axis == 0:
        _x1 = jnp.ones(1) * cut_value
        n1, n2 = ny, nz
    elif cut_axis == 1:
        _x2 = jnp.ones(1) * cut_value
        n1, n2 = nx, nz
    else:  # cut_axis == 2
        _x3 = jnp.ones(1) * cut_value
        n1, n2 = nx, ny
    # indexing="ij" so the flattened point order matches the (n1, n2) reshape
    # of the physical coords below; "xy" swaps the first two axes and scrambles
    # the poloidal-cut surface connectivity whenever n1 != n2 (star artifact).
    _x = jnp.array(jnp.meshgrid(_x1, _x2, _x3, indexing="ij"))
    _x = _x.transpose(1, 2, 3, 0).reshape(n1 * n2, 3)
    _y = jax.lax.map(F, _x, batch_size=mrx.MAP_BATCH_SIZE_INNER)
    _y1 = _y[:, 0].reshape(n1, n2)
    _y2 = _y[:, 1].reshape(n1, n2)
    _y3 = _y[:, 2].reshape(n1, n2)
    return _x, _y, (_y1, _y2, _y3), (_x1, _x2, _x3)


def _values_on_cuts(p_h, grids_pol):
    """``p_h`` on every cut, each reshaped to its ``(n1, n2)`` plane."""
    return np.asarray([
        jax.lax.map(p_h, grid[0], batch_size=mrx.MAP_BATCH_SIZE_INNER)
        .reshape(grid[2][0].shape)
        for grid in grids_pol])


def plot_torus(
    p_h: Callable,
    grids_pol: list,
    grid_surface: tuple,
    figsize: tuple = (12, 8),
    labelsize: int = 20,
    ticksize: int = 16,
    gridlinewidth: float = 0.01,
    cstride: int = 4,
    elev: float = 30,
    azim: float = 140,
    noaxes: bool = False,
    cbar_label: Optional[str] = None,
):
    """The boundary surface as a wireframe with poloidal cuts coloured by ``p_h``.

    ``p_h`` maps a logical point ``(3,)`` to a scalar (a 0-form's
    ``DiscreteFunction``, or a pushed-forward form); ``grids_pol`` are
    :func:`get_2d_grids` cuts at fixed zeta and ``grid_surface`` the
    ``cut_axis=0, cut_value=1`` boundary sample. One colour scale over all
    cuts; ``cbar_label`` adds the colour bar. Returns ``(fig, ax)``.
    """
    vals = _values_on_cuts(p_h, grids_pol)
    vmin, vmax = float(vals.min()), float(vals.max())
    if vmax == vmin:
        vmax = vmin + 1e-12
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(FIELD_CMAP)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    X, Y, Z = grid_surface[2]
    ax.plot_surface(X, Y, Z, edgecolors=(0, 0, 0, 0.2), rstride=cstride,
                    cstride=cstride, shade=True, alpha=0.0, linewidth=gridlinewidth)

    for grid, v in zip(grids_pol, vals):
        X, Y, Z = grid[2]
        ax.plot_surface(X, Y, Z, facecolors=cmap(norm(v)), rstride=1, cstride=1,
                        shade=False, zsort="min", linewidth=0)

    if cbar_label is not None:
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(vals)
        cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.08)
        cbar.set_label(cbar_label, fontsize=labelsize)
        cbar.ax.tick_params(labelsize=ticksize)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    set_axes_equal(ax)

    ax.set_xlabel(r"$x_1$", fontsize=labelsize, labelpad=14)
    ax.set_ylabel(r"$x_2$", fontsize=labelsize, labelpad=14)
    ax.set_zlabel(r"$x_3$", fontsize=labelsize, labelpad=-30)
    for name in ("x", "y", "z"):
        ax.tick_params(axis=name, labelsize=ticksize, pad=6)

    fig.tight_layout()
    ax.view_init(elev=elev, azim=azim)
    if noaxes:
        ax.set_axis_off()
    return fig, ax


def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale."""
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    half = 0.5 * float(np.max(limits[:, 1] - limits[:, 0]))
    mids = limits.mean(axis=1)
    ax.set_xlim3d([mids[0] - half, mids[0] + half])
    ax.set_ylim3d([mids[1] - half, mids[1] + half])
    ax.set_zlim3d([mids[2] - half, mids[2] + half])


def plot_crossections_separate(
    p_h: Callable,
    grids_pol: list,
    zeta_vals: list,
    textsize: int = 16,
    ticksize: int = 16,
    plot_centerline: bool = False,
):
    """The poloidal cuts of :func:`plot_torus` side by side in the ``(R, z)`` plane.

    One filled contour per cut, common axis limits, one shared colour bar,
    a boxed ``zeta`` label per panel and an ``(R, z)`` arrow pair on the
    first. ``plot_centerline`` marks ``R = 1`` (the unit-major-radius
    analytic geometries). Returns ``(fig, axes)``.
    """
    vals = _values_on_cuts(p_h, grids_pol)
    R = [jnp.sqrt(grid[2][0] ** 2 + grid[2][1] ** 2) for grid in grids_pol]
    z = [grid[2][2] for grid in grids_pol]

    fig, axes = plt.subplots(1, len(grids_pol), figsize=(16, 16 / 5), squeeze=False)
    axes = axes.flatten()
    last_c = None
    for ax, Ri, zi, vi, zeta in zip(axes, R, z, vals, zeta_vals):
        last_c = ax.contourf(Ri, zi, vi, 25, cmap=FIELD_CMAP, zorder=2)
        ax.set_axisbelow(False)
        if plot_centerline:
            ax.axvline(1.0, color="k", linestyle=":", linewidth=1.5, zorder=3)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(0.98, 0.98, rf"$\zeta = {float(zeta):.2f}$", transform=ax.transAxes,
                fontsize=textsize, ha="right", va="top", zorder=10,
                bbox=dict(facecolor="white", edgecolor="black",
                          boxstyle="round,pad=0.3", alpha=1.0))

    Rmin, Rmax = float(min(r.min() for r in R)), float(max(r.max() for r in R))
    Zmin, Zmax = float(min(v.min() for v in z)), float(max(v.max() for v in z))
    for ax in axes:
        ax.set_xlim(Rmin, Rmax)
        ax.set_ylim(Zmin, Zmax)

    # (R, z) reference arrows at the bottom-left of the first panel.
    anchor = axes[0]
    x0, y0, arrow_len = -0.01, -0.01, 0.16
    if plot_centerline:
        anchor.text(0.5, 1.02, r"$R = 1$", transform=anchor.transAxes,
                    fontsize=textsize, ha="center", va="bottom", zorder=12,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.2))
    for tip in ((x0, y0 + arrow_len), (x0 + arrow_len, y0)):
        anchor.annotate("", xy=tip, xytext=(x0, y0), xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="->", linewidth=1.5, color="k"))
    anchor.text(x0 - 0.01, y0 + arrow_len + 0.01, r"$z$", transform=anchor.transAxes,
                fontsize=textsize + 2, ha="center", va="bottom")
    anchor.text(x0 + arrow_len + 0.01, y0 - 0.01, r"$R$", transform=anchor.transAxes,
                fontsize=textsize + 2, ha="left", va="center")

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    cbar_ax.tick_params(labelsize=ticksize)
    cbar = fig.colorbar(last_c, cax=cbar_ax,
                        format=mticker.ScalarFormatter(useMathText=True))
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()
    cbar.ax.yaxis.get_offset_text().set_fontsize(ticksize)
    return fig, axes


def plot_twin_axis(
    left_y,
    right_y,
    x_left=None,
    x_right=None,
    left_label: str = "",
    right_label: str = "",
    left_log: bool = True,
    right_log: bool = False,
    left_color: str = "black",
    right_color: str = "teal",
    left_marker: str = "s",
    right_marker: str = "d",
    left_linestyle: str = "-",
    right_linestyle: str = "--",
    left_markersize: int = 4,
    right_markersize: int = 4,
    num_iters_inner: int = 1,
    x_label: str = "iteration",
    figsize: tuple = (8, 3),
    grid: bool = True,
    grid_linestyle: str = "--",
    grid_linewidth: float = 0.5,
    left_plot_kwargs: Optional[dict] = None,
    right_plot_kwargs: Optional[dict] = None,
    ax=None,
):
    """Two traces against a shared x with separate y axes (``twinx``).

    Each side is log (``semilogy``) or linear on its own; the y label and
    ticks take the series colour. Without ``x_*`` the abscissa is
    ``arange(len(y)) * num_iters_inner``. The explicit style arguments are
    defaults that ``left_plot_kwargs``/``right_plot_kwargs`` override.
    With ``ax`` the pair is drawn into that existing axes (a panel of a
    larger figure, whose layout is then the caller's), otherwise into a new
    ``figsize`` figure. Returns ``(fig, (ax_left, ax_right))``.
    """
    if ax is None:
        fig, ax1 = plt.subplots(figsize=figsize)
    else:
        fig, ax1 = ax.figure, ax
    ax2 = ax1.twinx()
    sides = (
        (ax1, left_y, x_left, left_log, left_label,
         {"color": left_color, "linestyle": left_linestyle, "marker": left_marker,
          "markersize": left_markersize, **(left_plot_kwargs or {})}),
        (ax2, right_y, x_right, right_log, right_label,
         {"color": right_color, "linestyle": right_linestyle, "marker": right_marker,
          "markersize": right_markersize, **(right_plot_kwargs or {})}),
    )
    for ax, y, x, log, label, kwargs in sides:
        y = np.asarray(y)
        x = np.arange(len(y)) * int(num_iters_inner) if x is None else np.asarray(x)
        (ax.semilogy if log else ax.plot)(x, y, **kwargs)
        ax.set_ylabel(label, color=kwargs["color"])
        ax.tick_params(axis="y", labelcolor=kwargs["color"])
    ax1.set_xlabel(x_label)
    if grid:
        ax1.grid(True, which="both", linestyle=grid_linestyle, linewidth=grid_linewidth)
    if ax is None:
        fig.tight_layout()
    return fig, (ax1, ax2)
