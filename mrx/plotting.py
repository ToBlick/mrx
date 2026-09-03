"""Matplotlib figures of scalar fields on a torus and of relaxation traces.

* :func:`get_2d_grids` samples a map on a logical plane (a poloidal cut at
  fixed zeta, or the boundary surface at fixed r).
* :func:`plot_torus` draws the boundary surface as a wireframe with poloidal
  cuts coloured by a scalar; :func:`plot_crossections_separate` draws the
  same cuts side by side in the ``(R, z)`` plane.
* :func:`plot_twin_axis` draws two traces against a shared x with separate
  y axes -- the standard figure for a force residual next to an energy or a
  helicity.

Poincaré sections are traced in :mod:`mrx.poincare` and drawn by
:func:`render_section` here; ``scripts/plot_relaxation.py``
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
from mrx.plotstyle import (FIELD_CMAP, FS, LEFT, PRESSURE_CMAP, RIGHT,
                           SECTION_CMAP, SectionLimits, house_style)

# FIELD_CMAP, SECTION_CMAP, PRESSURE_CMAP, FS, LEFT/RIGHT and house_style live in
# mrx.plotstyle now (re-exported here for callers that import them from plotting).
__all__ = ["get_2d_grids", "plot_torus", "plot_crossections_separate",
           "plot_twin_axis", "render_section", "resonant_rationals",
           "set_axes_equal", "FIELD_CMAP", "SECTION_CMAP", "PRESSURE_CMAP",
           "SectionLimits"]


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


@house_style()
def plot_torus(
    p_h: Callable,
    grids_pol: list,
    grid_surface: tuple,
    figsize: tuple = (12, 8),
    labelsize: float = FS.big,
    ticksize: float = FS.tick,
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


@house_style()
def plot_crossections_separate(
    p_h: Callable,
    grids_pol: list,
    zeta_vals: list,
    textsize: float = FS.label,
    ticksize: float = FS.tick,
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


@house_style()
def plot_twin_axis(
    left_y,
    right_y,
    x_left=None,
    x_right=None,
    left_label: str = "",
    right_label: str = "",
    left_log: bool = True,
    right_log: bool = False,
    left_color: str = LEFT["color"],
    right_color: str = RIGHT["color"],
    left_marker: str = LEFT["marker"],
    right_marker: str = RIGHT["marker"],
    left_linestyle: str = LEFT["linestyle"],
    right_linestyle: str = RIGHT["linestyle"],
    left_markersize: int = LEFT["markersize"],
    right_markersize: int = RIGHT["markersize"],
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


# ---------------------------------------------------------------------------
# Poincaré section figure (moved from mrx.poincare 2026-08-28: the tracer
# stays headless, the figure lives with the other matplotlib code)
# ---------------------------------------------------------------------------

def resonant_rationals(iota_min, iota_max, nfp, denom_max=30, min_sep=0.06):
    """Rationals in ``[iota_min, iota_max]`` where an island chain can form.

    An island chain needs a resonant perturbation: ``iota = n/m`` with ``n`` the
    toroidal and ``m`` the poloidal mode number. A field with ``nfp`` field
    periods carries only toroidal harmonics ``n = 0 (mod nfp)``, so the only
    rationals that can open an island are those whose NUMERATOR is a multiple
    of ``nfp``. Every other rational surface is resonance-free and closes on
    itself harmlessly.

    Deduplicated by VALUE, so ``5/6`` is kept and ``10/12`` -- the same
    surface, driven by a weaker harmonic -- is not repeated.

    Which of them to label is a spacing problem: ``denom_max = 30`` on W7-X
    puts ~40 resonances in an iota range of 0.2, and labelling all of them
    overprints, while the two or three lowest orders leave the scale unreadable.
    So the candidates are ranked by poloidal mode number (an ``n/m`` island is
    wider the smaller ``m`` is, and that is also how they are read), and each
    is accepted only if it is at least ``min_sep`` of the range away from every
    label already placed. Low orders always win their slot; higher orders fill
    the gaps until the spacing rule stops them.
    """
    span = max(iota_max - iota_min, 1e-12)
    candidates, seen = [], set()
    for j in range(1, max(denom_max // nfp, 1) + 1):
        n_tor = j * nfp
        for m_pol in range(1, denom_max + 1):
            value = n_tor / m_pol
            if iota_min <= value <= iota_max and value not in seen:
                candidates.append((m_pol, n_tor, value))
                seen.add(value)
    ticks, labels = [], []
    for m_pol, n_tor, value in sorted(candidates):
        if all(abs(value - t) >= min_sep * span for t in ticks):
            ticks.append(value)
            labels.append(f"{n_tor}/{m_pol}")
    order = sorted(range(len(ticks)), key=lambda i: ticks[i])
    return [ticks[i] for i in order], [labels[i] for i in order]


#: Profile-line colours: one per QUANTITY (iota on the left axis, p on the
#: right twin). Rays are told apart by line style, not colour.
IOTA_COLOR = "black"
P_COLOR = "#6a3d9a"

#: Poloidal rays for the logical profile, in order. theta = 0 is the symmetry
#: line where odd island chains have their X-points; theta = 0.5 is where those
#: same chains are fattest (their O-points), so the pair brackets an odd chain;
#: 1/3 catches three-fold structure off both. Further rays fill in.
PROFILE_RAY_THETAS = (0.0, 0.5, 1.0 / 3.0, 1.0 / 6.0, 0.25, 0.75)


def _profile_ray_thetas(n):
    """The first ``n`` profile-ray thetas (see :data:`PROFILE_RAY_THETAS`);
    golden-angle extras beyond the fixed list so a large ``n`` stays spread."""
    n = max(int(n), 1)
    base = list(PROFILE_RAY_THETAS)
    if n <= len(base):
        return base[:n]
    golden = 0.5 * (5.0 ** 0.5 - 1.0)
    return base + [((k + 1) * golden) % 1.0 for k in range(n - len(base))]


def _ray_line(lr, lth, R, Z, pressure, th0):
    """Per-line logical r, physical (R, Z) and p at the crossing nearest the
    poloidal ray ``theta = th0`` (circular nearest, one crossing per line).

    Used to draw the logical-r profile along a ray and to mark that ray on the
    section panels: the crossing nearest th0 is a real point ON the ray, so the
    marked line is F(r, th0) traced through the actual data, not an assumed
    straight radius (logical theta is not the physical poloidal angle).
    """
    dth = np.abs(((lth - th0 + 0.5) % 1.0) - 0.5)      # (nL, nC) circular distance
    k = np.argmin(dth, axis=1)                          # nearest crossing per line
    rows = np.arange(lr.shape[0])
    p_at = None if pressure is None else pressure[rows, k]
    return lr[rows, k], (R[rows, k], Z[rows, k]), p_at


@house_style()
def render_section(R, Z, iota, iota_err, seed_r, keep, *, title, subtitle,
                   axis_RZ=None, profile_x=None,
                   profile_xlabel="seed radius $r$", nfp=None, denom_max=30,
                   logical=None, pressure=None,
                   pressure_label=r"$p$", split_iota_p=None, pressure_scale=100.0,
                   cmap=SECTION_CMAP, iota_lim=None, limits=None, iota_scatter=None,
                   profile_coord="logical", profile_rays=3):
    """The section coloured by iota, with the iota profile and optionally p.

    Pure arrays in, so a run can be re-rendered from its archive without
    rebuilding the map -- which is the expensive half of producing it.

    ``limits`` pins everything else a movie must hold fixed between frames:
    a dict with any of ``RZ`` (``((R0, R1), (Z0, Z1))`` of the section
    panel), ``z_split`` (the split line), ``x`` (the profiles' abscissa) and
    ``p`` (the pressure panel's ordinate, in drawn units).
    ``iota_lim`` pins the colour scale instead of taking it from this figure's
    own lines.  Two sections drawn on limits fitted separately are not
    comparable by colour at all -- the same hue means a different transform in
    each -- so any caller producing a set that is meant to be read side by side
    (two relaxation states, a plane scan) must pass one shared pair.

    ``pressure`` is per-crossing, the same shape as ``R``. It is OPTIONAL
    because the fields this traces are harmonic (vacuum-like) and carry no
    pressure at all; a vacuum run leaves it ``None`` and gets exactly the
    previous figure. When it is given, the pressure PROFILE joins the iota
    profile on the right axis of the same panel (:func:`mrx.plotting.plot_twin_axis`,
    the house twin-axis style): per line, the mean of p over its crossings
    with a one-standard-deviation band, against the same surface label
    (labelled ``pressure_label``). On a flux surface of an equilibrium p is
    constant and the band collapses; on an island chain or a chaotic line it
    is not, and the band width measures how far that line is from
    ``B . grad p = 0``.

    ``pressure_scale`` multiplies p wherever it is drawn (colour and profile),
    and the labels say so.

    Every kept line is drawn and fitted, chaotic ones included: the iota
    profile carries a ribbon, so a line without a rotational transform shows as
    a point with a wide ribbon rather than as a separate category. The ribbon
    is ``iota_scatter`` when given -- the std of iota over K equal ζ-windows
    (:func:`mrx.poincare._iota_window_scatter`), the along-line spread that
    reads like the pressure band -- else ``iota_err`` (the whole-line fit
    RMS/N, see :func:`trace_and_classify`).

    ``split_iota_p`` colours the section by iota ABOVE the magnetic axis and by
    p BELOW it, in one panel; the default is on whenever ``pressure`` is given.
    It needs ``axis_RZ`` and raises without it rather than quietly drawing a
    half-empty panel: 'above' and 'below' are defined against the MAGNETIC
    axis, not ``Z = 0``, which would cut a Shafranov-shifted plasma off-centre.
    """

    if split_iota_p is None:
        split_iota_p = pressure is not None
    if split_iota_p and pressure is None:
        raise ValueError("split_iota_p=True needs a pressure array: the "
                         "below-axis half of the section is coloured by p.")
    if split_iota_p and axis_RZ is None:
        raise ValueError("split_iota_p=True needs axis_RZ: the split is at the "
                         "magnetic axis, not at Z = 0.")

    # Movie/side-by-side pinning: one object, whether the caller passed a
    # SectionLimits, the legacy dict, or a bare iota_lim (all coerced here).
    lim = SectionLimits.coerce(limits, iota_lim)
    has_p = pressure is not None
    # The pressure is drawn at ``pressure_scale`` times its value (the natural
    # scale of p in code units is ~1e-2, and 100 p reads in units).
    p_label = f"{pressure_label} $\\times$ {pressure_scale:g}"
    # Panels, left to right: the section (with its colourbars), the logical
    # chart (when the crossings' logical coordinates are given), and the
    # profiles -- iota on the left axis and, when p is given, p on the right
    # axis of the same panel. The logical chart is where an island chain or
    # an off-centre axis is seen at a glance -- nested surfaces are
    # horizontal bands there -- so it stays in the relaxation figures too
    # (restored 2026-08-28).
    panels = [("ax", 1.45)]
    if logical is not None:
        panels.append(("lx", 0.9))
    panels.append(("bx", 1.15))
    width = {2: 12.0, 3: 16.5}[len(panels)]
    fig = plt.figure(figsize=(width, 4.8), constrained_layout=True)
    axes = dict(zip((name for name, _ in panels),
                    fig.subplots(1, len(panels),
                                 width_ratios=[w for _, w in panels])))
    ax, bx = axes["ax"], axes["bx"]
    lx = axes.get("lx")

    shown = keep
    good = iota[shown][jnp.isfinite(iota[shown])] if shown.any() else iota[:0]
    if lim.iota is not None:
        lo, hi = float(lim.iota[0]), float(lim.iota[1])
    else:
        lo, hi = ((float(jnp.min(good)), float(jnp.max(good)))
                  if good.size else (0.0, 1.0))
    if hi - lo < 1e-9:
        lo, hi = lo - 5e-3, hi + 5e-3

    # One marker per crossing: ~10^4 points want a hairline to show the surface
    # texture, ~10^2 want something you can actually see.
    npts = max(int(keep.sum()) * R.shape[1], 1)
    size = float(jnp.clip(3000.0 / npts, 0.35, 15.0))
    colour = jnp.broadcast_to(iota[:, None], R.shape)

    # The split is per CROSSING, not per line: a surface straddles the axis, so
    # the same line is iota-coloured where it is above and p-coloured below.
    # axis_RZ carries the axis crossing at each save; the dividing line is
    # their mean (the axis wanders by ~1e-3 of the minor radius over a period).
    if split_iota_p:
        z_axis = float(jnp.mean(jnp.asarray(axis_RZ[1])))
        if lim.z_split is not None:
            z_axis = float(lim.z_split)
        upper = Z >= z_axis
    else:
        z_axis = None
        upper = jnp.ones_like(R, dtype=bool)
    # `shown` selects LINES, `upper` selects CROSSINGS.
    shown2 = jnp.broadcast_to(shown[:, None], R.shape)
    sel_iota = shown2 & upper
    sc = ax.scatter(R[sel_iota], Z[sel_iota], c=colour[sel_iota],
                    s=size, vmin=lo, vmax=hi, cmap=cmap, linewidths=0,
                    rasterized=True)
    psc = None
    sel_p = None
    p_range = {}
    if split_iota_p:
        # Chaotic lines keep their grey in BOTH halves: colouring one half of
        # a line and greying the other reads as two different objects.
        sel_p = shown2 & ~upper
        p_range = ({"vmin": lim.p[0], "vmax": lim.p[1]}
                   if lim.p is not None else {})                # pinned in a movie
        if sel_p.any():
            psc = ax.scatter(R[sel_p], Z[sel_p], c=pressure_scale * pressure[sel_p], s=size,
                             cmap=PRESSURE_CMAP, linewidths=0, rasterized=True, **p_range)
    res_ticks, res_labels = (resonant_rationals(lo, hi, int(nfp), denom_max)
                             if nfp else ([], []))
    if (~keep).any():
        ax.scatter(R[~keep], Z[~keep], c="0.55", s=size, linewidths=0,
                   rasterized=True, label=f"lost ({int((~keep).sum())})")
        ax.legend(loc="upper right", fontsize=FS.annot, markerscale=4)
    if axis_RZ is not None:
        # ONE marker at the mean, plus a hairline through the wander. Drawing a
        # "k+" at every save stacked 401 opaque markers into a black blob ~10%
        # of the minor radius across, which reads as a failed line at the axis
        # -- it is neither a line nor a failure, it is the axis doing what a
        # stellarator axis does.
        aR, aZ = jnp.asarray(axis_RZ[0]), jnp.asarray(axis_RZ[1])
        if aR.ndim and aR.size > 1:
            ax.plot(aR, aZ, "-", color="0.35", lw=0.4, alpha=0.6, zorder=4)
        ax.plot(jnp.mean(aR), jnp.mean(aZ), "k+", ms=7, mew=1.2, zorder=5)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    # The label sits BELOW the bar: the Farey tick labels are wide, so a
    # side label was squeezed against the next panel, and above the bar the
    # section's title runs into it whenever the section is narrower than
    # its panel (a bean-shaped cut at equal aspect).
    cbar.ax.set_xlabel(r"$\iota$", fontsize=FS.title)
    if res_ticks:
        # Only the rationals an nfp-periodic field can actually resonate with:
        # everything else on the colorbar is a surface no island can open on.
        cbar.set_ticks(res_ticks)
        cbar.set_ticklabels(res_labels)
    if psc is not None:
        pbar = fig.colorbar(psc, ax=ax, fraction=0.046, pad=0.02)
        # Label BELOW the bar, matching iota: a side label is squeezed against
        # the next panel and the wide tick labels leave no room for it.
        pbar.ax.set_xlabel(pressure_label, fontsize=FS.title)
        pbar.ax.tick_params(labelsize=FS.annot)
        ax.axhline(z_axis, color="0.35", lw=0.6, ls=":", zorder=1)

    # An axisymmetric vacuum field has iota = 0, so every line is a fixed point
    # of the return map and the section collapses onto the midplane.  That is
    # the right answer, but equal aspect renders it as a hairline, so the aspect
    # is only held when the two spans are within a factor of 20.
    xlim = _padded(R[keep])
    # The floor has to be RELATIVE to the other axis: an absolute one is
    # meaningless against whatever units R happens to be in, and leaves a 1e-16
    # Z-range labelled in units of 1e-16.
    ylim = _padded(Z[keep], floor=0.04 * (xlim[1] - xlim[0]))
    if lim.RZ is not None:
        xlim, ylim = (tuple(float(v) for v in pair) for pair in lim.RZ)
    spans = (xlim[1] - xlim[0], ylim[1] - ylim[0])
    to_scale = max(spans) / max(min(spans), 1e-30) < 20.0
    ax.set_aspect("equal" if to_scale else "auto")
    if np.ptp(Z[keep]) < 1e-6 * (xlim[1] - xlim[0]):
        ax.text(0.5, 0.86, "iota = 0: every line is a fixed point of the\n"
                           "return map, so each surface is a single dot",
                transform=ax.transAxes, ha="center", fontsize=FS.annot, color="0.35")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("R")
    ax.set_ylabel("Z")

    if lx is not None:
        # The SAME crossings in the logical chart: r against theta, both in
        # [0,1]. Nested surfaces are horizontal bands here, and anything that
        # is not -- a chain of islands, or surfaces sitting off-centre in r
        # because the magnetic axis is not at r=0 -- shows up immediately,
        # where the physical panel hides it behind the shaping.
        lr, lth = logical
        # Split the same way the physical panel does: iota above the magnetic
        # axis, p below it, per CROSSING (the same ``upper`` / ``sel_p`` masks).
        # The divider is Z = z_axis in physical space, which is not a straight
        # line in the (r, theta) chart, so none is drawn here.
        lx.scatter(lr[sel_iota], lth[sel_iota], c=colour[sel_iota], s=size, vmin=lo,
                   vmax=hi, cmap=cmap, linewidths=0, rasterized=True)
        if split_iota_p and sel_p.any():
            lx.scatter(lr[sel_p], lth[sel_p], c=pressure_scale * pressure[sel_p], s=size,
                       cmap=PRESSURE_CMAP, linewidths=0, rasterized=True, **p_range)
        if (~keep).any():
            lx.scatter(lr[~keep], lth[~keep], c="0.55", s=size, linewidths=0,
                       rasterized=True)
        lx.set_xlim(0.0, 1.0)
        # theta increases DOWNWARD so the pressure half (below the magnetic
        # axis) sits at the bottom, aligned with the physical panel where p is
        # also below the axis -- without the flip p is at the top here and the
        # bottom there.
        lx.set_ylim(1.0, 0.0)
        lx.set_xlabel(r"$r$")
        lx.set_ylabel(r"$\theta$")

    # ---- profile panel: iota (and p) against the surface label -------------
    logical_prof = profile_coord == "logical" and logical is not None
    band = iota_err if iota_scatter is None else iota_scatter
    ribbon_label = (r"$\iota \pm$ fit RMS / $N$" if iota_scatter is None
                    else r"$\iota \pm$ window std")

    if logical_prof:
        # iota (black, left axis) and p (purple, right axis) against LOGICAL r,
        # sampled along ``profile_rays`` poloidal rays (:data:`PROFILE_RAY_THETAS`):
        # theta = 0 is the symmetry line where odd island chains have their
        # X-points, theta = 0.5 is where the same chains are fattest (their
        # O-points), so the pair brackets an odd chain; 1/3 catches three-fold
        # structure off both. Each ray is one LINE STYLE, and is marked in that
        # style on BOTH section panels -- the physical curve F(r, theta0), and
        # theta = theta0 in the chart -- so the reader can place it. Where the
        # rays agree logical r is a faithful surface label; where they fan
        # (edge, islands) it is not, and the fan is the signal.
        styles = ["-", "--", ":", "-."]
        lr_all, lth_all = np.asarray(logical[0]), np.asarray(logical[1])
        Rn, Zn, sn = np.asarray(R), np.asarray(Z), np.asarray(shown)
        iota_n, band_n = np.asarray(iota), np.asarray(band)
        pn = None if pressure is None else np.asarray(pressure)
        pstd = None if pn is None else pressure_scale * np.nanstd(pn, axis=1)
        px = bx.twinx() if has_p else None
        thetas = _profile_ray_thetas(profile_rays)
        for i, th0 in enumerate(thetas):
            ls = styles[i % len(styles)]
            r_line, (Rr, Zr), p_at = _ray_line(lr_all, lth_all, Rn, Zn, pn, th0)
            m = sn & np.isfinite(r_line)
            if not m.any():
                continue
            o = np.argsort(r_line[m])
            rr = r_line[m][o]
            bx.plot(rr, iota_n[m][o], color=IOTA_COLOR, linestyle=ls, lw=1.2,
                    label=rf"$\theta = {th0:.2f}$")
            bx.fill_between(rr, (iota_n - band_n)[m][o], (iota_n + band_n)[m][o],
                            color=IOTA_COLOR, alpha=0.10, lw=0)
            if has_p:
                pm = pressure_scale * p_at
                px.plot(rr, pm[m][o], color=P_COLOR, linestyle=ls, lw=1.2)
                px.fill_between(rr, (pm - pstd)[m][o], (pm + pstd)[m][o],
                                color=P_COLOR, alpha=0.10, lw=0)
            ax.plot(Rr[m][o], Zr[m][o], color="black", linestyle=ls, lw=1.0,
                    alpha=0.85, zorder=6)
            if lx is not None:
                lx.axhline(th0, color="black", linestyle=ls, lw=1.0,
                           alpha=0.85, zorder=6)
        bx.set_xlabel(r"logical $r$")
        bx.set_ylabel(r"$\iota$", color=IOTA_COLOR)
        bx.tick_params(axis="y", labelcolor=IOTA_COLOR)
        if px is not None:
            px.set_ylabel(p_label, color=P_COLOR)
            px.tick_params(axis="y", labelcolor=P_COLOR)
            if lim.p is not None:
                px.set_ylim(*lim.p)
        for value, lab in zip(res_ticks, res_labels):
            bx.axhline(value, color="0.55", lw=0.6, ls="--", zorder=0)
            bx.annotate(lab, (0.995, value), xycoords=("axes fraction", "data"),
                        ha="right", va="bottom", fontsize=FS.annot, color="0.4")
        if lim.iota is not None:
            bx.set_ylim(lo, hi)
        if lim.x is not None:
            bx.set_xlim(*lim.x)
        bx.grid(alpha=0.3)
        bx.legend(loc="upper center", ncol=len(thetas), fontsize=FS.annot,
                  columnspacing=1.0, handlelength=2.4)
    else:
        x = seed_r if profile_x is None else profile_x
        # The abscissa carries both midplane crossings (see midplane_crossings):
        # per-line quantities are tiled to match, a NaN crossing drops that
        # entry, and sorted by abscissa the curve runs inboard -> axis ->
        # outboard as a slice of the section panel.
        X = jnp.asarray(x)
        X = X[:, None] if X.ndim == 1 else X
        xs = X.ravel()

        def per_line(a):
            return jnp.broadcast_to(jnp.asarray(a)[:, None], X.shape).ravel()

        prof = per_line(shown) & jnp.isfinite(xs)
        order = jnp.argsort(xs[prof])
        xo, io, eo = xs[prof][order], per_line(iota)[prof][order], per_line(band)[prof][order]
        left = dict(color=IOTA_COLOR, marker="none", linestyle="-")
        right = dict(color=P_COLOR, marker="none", linestyle="--")
        if has_p:
            p_mean = per_line(jnp.mean(pressure_scale * pressure, axis=1))
            p_std = per_line(jnp.std(pressure_scale * pressure, axis=1))
            mo, so = p_mean[prof][order], p_std[prof][order]
            _, (bx, px) = plot_twin_axis(
                io, mo, x_left=xo, x_right=xo, left_label=r"$\iota$",
                right_label=p_label, left_log=False, right_log=False,
                x_label=profile_xlabel, grid=False, ax=bx,
                left_plot_kwargs=dict(left, lw=0.8),
                right_plot_kwargs=dict(right, lw=0.8))
            px.fill_between(xo, mo - so, mo + so, color=right["color"], alpha=0.2, lw=0,
                            label=r"$p \pm 1$ std over the line")
            if lim.p is not None:
                px.set_ylim(*lim.p)
        else:
            px = None
            bx.plot(xo, io, lw=0.8, **left)
            bx.set_xlabel(profile_xlabel)
            bx.set_ylabel(r"$\iota$")
        bx.fill_between(xo, io - eo, io + eo, color=left["color"], alpha=0.15, lw=0,
                        label=ribbon_label)
        for value, lab in zip(res_ticks, res_labels):
            bx.axhline(value, color="0.55", lw=0.6, ls="--", zorder=0)
            bx.annotate(lab, (0.995, value), xycoords=("axes fraction", "data"),
                        ha="right", va="bottom", fontsize=FS.annot, color="0.4")
        if lim.iota is not None:
            bx.set_ylim(lo, hi)
        bx.grid(alpha=0.3)
        if lim.x is not None:
            bx.set_xlim(*lim.x)
        handles, labels_ = bx.get_legend_handles_labels()
        if px is not None:
            h2, l2 = px.get_legend_handles_labels()
            handles, labels_ = handles + h2, labels_ + l2
        bx.legend(handles, labels_, loc="center")

    # One descriptive title for the whole figure, not a title per panel.
    sup = title if to_scale else f"{title}   —   AXES NOT TO SCALE"
    if subtitle:
        sup = f"{sup}   |   {subtitle}"
    if has_p:
        sup = f"{sup}   |   {p_label}"     # states the p scaling once, here
    fig.suptitle(sup, fontsize=FS.title)

    # Saving is the caller's: render_section is pure, so a run re-renders from
    # its archive and the caller owns the path (and the movie's frame naming).
    return fig, axes


def _padded(v, pad=0.06, floor=0.0):
    lo, hi = float(jnp.nanmin(v)), float(jnp.nanmax(v))
    span = max(hi - lo, floor)
    mid = 0.5 * (lo + hi)
    return mid - 0.5 * span - pad * span, mid + 0.5 * span + pad * span
