"""Matplotlib figures of a mapped torus, of relaxation histories and of Poincare sections.

* :func:`get_2d_grids` samples a map on a logical plane: a poloidal cut at fixed zeta, or the boundary at
  fixed r. Its output feeds the two torus plots.
* :func:`plot_torus` draws the boundary as a wireframe with poloidal cuts coloured by a scalar, and
  :func:`plot_crossections_separate` draws the same cuts side by side in the ``(R, z)`` plane.
* :func:`plot_twin_axis` draws two histories against the iteration on twin y axes, for example a force
  residual and an energy.
* :func:`render_section` draws a Poincare section from :func:`mrx.diagnostics.poincare.poincare`,
  :func:`paper_fonts` resizes a section figure for a printed page, and :func:`plot_archive` draws every
  field and plane of a trace archive in the layout of the paper.

All figures use the house style of :mod:`mrx.diagnostics.plotstyle`.
"""

import os
from typing import Callable

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

import mrx
from mrx.diagnostics.plotstyle import (FIELD_CMAP, FS, IOTA_COLOR, LEFT, P_COLOR, PRESSURE_CMAP, RIGHT, SECTION_CMAP,
                           house_style)


def get_2d_grids(
    Phi: Callable,
    cut_value: float = 0,
    cut_axis: int = 2,
    nx: int = 64,
    ny: int = 64,
    nz: int = 64,
    invert_z: bool = False,
):
    """Sample the logical-to-physical map ``Phi`` on the logical plane
    ``x_{cut_axis} = cut_value``, with ``cut_axis`` 0 (a surface of constant r) or 2 (a poloidal cut at
    constant zeta).

    The other two logical coordinates are sampled uniformly on ``[0, 1]``, except that r stays just inside
    ``(0, 1)`` to avoid the polar axis and the boundary. ``invert_z`` reverses the zeta direction, which flips
    the orientation of a surface. Returns ``(x, y, (Y1, Y2, Y3), (x1, x2, x3))``: the logical points as a flat
    ``(n1 n2, 3)`` array, their images, the three components of the images as ``(n1, n2)`` arrays, and the
    three 1-D logical axes.
    """
    _x1 = jnp.linspace(1e-6, 1.0 - 1e-6, nx)
    _x2 = jnp.linspace(0.0, 1.0, ny)
    _x3 = jnp.linspace(0.0, 1.0, nz)
    if invert_z:
        _x3 = _x3[::-1]
    if cut_axis == 0:
        _x1 = jnp.ones(1) * cut_value
        n1, n2 = ny, nz
    else:
        _x3 = jnp.ones(1) * cut_value
        n1, n2 = nx, ny
    # indexing="ij": the flattened point order matches the (n1, n2) reshape below
    _x = jnp.array(jnp.meshgrid(_x1, _x2, _x3, indexing="ij"))
    _x = _x.transpose(1, 2, 3, 0).reshape(n1 * n2, 3)
    _y = jax.lax.map(Phi, _x, batch_size=mrx.MAP_BATCH_SIZE_INNER)
    _y1 = _y[:, 0].reshape(n1, n2)
    _y2 = _y[:, 1].reshape(n1, n2)
    _y3 = _y[:, 2].reshape(n1, n2)
    return _x, _y, (_y1, _y2, _y3), (_x1, _x2, _x3)


def _values_on_cuts(p_h, grids_pol):
    """The values of ``p_h`` on every cut, each as an ``(n1, n2)`` array."""
    return np.asarray([
        jax.lax.map(p_h, grid[0], batch_size=mrx.MAP_BATCH_SIZE_INNER)
        .reshape(grid[2][0].shape)
        for grid in grids_pol])


@house_style()
def plot_torus(p_h: Callable, grids_pol: list, grid_surface: tuple, cbar_label: str):
    """A 3-D view of the boundary surface as a wireframe, with poloidal cuts coloured by the scalar ``p_h``
    (a function of the logical point).

    ``grids_pol`` are cuts at fixed zeta and ``grid_surface`` is the boundary (``cut_axis=0``), all from
    :func:`get_2d_grids`. All cuts share one colour scale, whose bar is labelled ``cbar_label``. Returns
    ``(fig, ax)``.
    """
    vals = _values_on_cuts(p_h, grids_pol)
    norm = mpl.colors.Normalize(vmin=float(vals.min()), vmax=float(vals.max()))
    cmap = plt.get_cmap(FIELD_CMAP)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(*grid_surface[2], edgecolors=(0, 0, 0, 0.2), rstride=8, cstride=8, shade=True, alpha=0.0,
                    linewidth=0.3)
    for grid, v in zip(grids_pol, vals):
        ax.plot_surface(*grid[2], facecolors=cmap(norm(v)), rstride=1, cstride=1, shade=False, zsort="min",
                        linewidth=0)

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(vals)
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.08)
    cbar.set_label(cbar_label, fontsize=FS.big)
    cbar.ax.tick_params(labelsize=FS.tick)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    set_axes_equal(ax)

    ax.set_xlabel(r"$x_1$", fontsize=FS.big, labelpad=14)
    ax.set_ylabel(r"$x_2$", fontsize=FS.big, labelpad=14)
    ax.set_zlabel(r"$x_3$", fontsize=FS.big, labelpad=-30)
    for name in ("x", "y", "z"):
        ax.tick_params(axis=name, labelsize=FS.tick, pad=6)

    ax.view_init(elev=25, azim=40)
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
def plot_crossections_separate(p_h: Callable, grids_pol: list, zeta_vals: list):
    """The poloidal cuts of :func:`plot_torus` side by side in the ``(R, z)`` plane, as filled contours of
    ``p_h`` with common axis limits and one colour bar. Each panel is labelled with its value from
    ``zeta_vals``. Returns ``(fig, axes)``."""
    textsize, ticksize = FS.label, FS.tick
    vals = _values_on_cuts(p_h, grids_pol)
    R = [jnp.sqrt(grid[2][0] ** 2 + grid[2][1] ** 2) for grid in grids_pol]
    z = [grid[2][2] for grid in grids_pol]

    fig, axes = plt.subplots(1, len(grids_pol), figsize=(16, 16 / 5), squeeze=False)
    axes = axes.flatten()
    last_c = None
    for ax, Ri, zi, vi, zeta in zip(axes, R, z, vals, zeta_vals):
        last_c = ax.contourf(Ri, zi, vi, 25, cmap=FIELD_CMAP, zorder=2)
        ax.set_axisbelow(False)
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
    for tip in ((x0, y0 + arrow_len), (x0 + arrow_len, y0)):
        anchor.annotate("", xy=tip, xytext=(x0, y0), xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="->", linewidth=1.5, color="k"))
    anchor.text(x0 - 0.01, y0 + arrow_len + 0.01, r"$z$", transform=anchor.transAxes,
                fontsize=textsize + 2, ha="center", va="bottom")
    anchor.text(x0 + arrow_len + 0.01, y0 - 0.01, r"$R$", transform=anchor.transAxes,
                fontsize=textsize + 2, ha="left", va="center")

    # the house style lays the figure out itself (constrained layout), so the bar takes its space from the axes
    cbar = fig.colorbar(last_c, ax=list(axes), shrink=0.7, format=mticker.ScalarFormatter(useMathText=True))
    cbar.ax.tick_params(labelsize=ticksize)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()
    cbar.ax.yaxis.get_offset_text().set_fontsize(ticksize)
    return fig, axes


def plot_twin_axis(left_y, right_y, x_right=None, left_label="", right_label="",
                   left_plot_kwargs=None, right_plot_kwargs=None):
    """``left_y`` on a log scale and ``right_y`` on a linear scale against the iteration, on twin y axes. Each
    axis label and its ticks take the colour of its curve.

    The curve styles are :data:`~mrx.diagnostics.plotstyle.LEFT` and :data:`~mrx.diagnostics.plotstyle.RIGHT`,
    updated by ``left_plot_kwargs`` and ``right_plot_kwargs``. ``x_right`` gives the x values of the right curve
    (by default ``0, 1, 2, ...``). Returns ``(fig, (ax_left, ax_right))``."""
    fig, ax1 = plt.subplots(figsize=(8, 3))
    ax2 = ax1.twinx()
    for ax, y, x, plot, label, kwargs in (
            (ax1, left_y, None, ax1.semilogy, left_label, {**LEFT, **(left_plot_kwargs or {})}),
            (ax2, right_y, x_right, ax2.plot, right_label, {**RIGHT, **(right_plot_kwargs or {})})):
        y = np.asarray(y)
        plot(np.arange(len(y)) if x is None else np.asarray(x), y, **kwargs)
        ax.set_ylabel(label, color=kwargs["color"])
        ax.tick_params(axis="y", labelcolor=kwargs["color"])
    ax1.set_xlabel("iteration")
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5)
    return fig, (ax1, ax2)


# ---------------------------------------------------------------------------
# Poincare section figure
# ---------------------------------------------------------------------------

#: Logical poloidal angles of the rays along which :func:`render_section` reads the iota profile, in the order
#: used. theta = 1/2 comes first because it passes through the O-points of chains with odd m. theta = 0 is
#: avoided because the poloidal angle jumps there.
PROFILE_RAY_THETAS = (0.5, 1.0 / 3.0, 0.2, 1.0 / 6.0, 0.25, 0.75)
#: The pressure is drawn multiplied by this factor, and the labels say so.
PRESSURE_SCALE = 100.0


def resonant_rationals(iota_min, iota_max, nfp):
    """Tick positions and labels ``n/m`` for the rationals in ``[iota_min, iota_max]`` at which a field with
    ``nfp`` periods can have island chains (``n`` a multiple of ``nfp``, ``m <= 300``). Rationals with small
    ``m`` are preferred, and a rational is skipped if it lies within 0.12 of the range from one already chosen.
    Returns ``(ticks, labels)`` sorted by value."""
    denom_max, span = 300, iota_max - iota_min
    candidates, seen = [], set()
    for j in range(1, denom_max // nfp + 1):
        n_tor = j * nfp
        for m_pol in range(1, denom_max + 1):
            value = n_tor / m_pol
            if iota_min <= value <= iota_max and value not in seen:
                candidates.append((m_pol, n_tor, value))
                seen.add(value)
    ticks, labels = [], []
    for m_pol, n_tor, value in sorted(candidates):
        if all(abs(value - t) >= 0.12 * span for t in ticks):
            ticks.append(value)
            labels.append(f"{n_tor}/{m_pol}")
    order = sorted(range(len(ticks)), key=lambda i: ticks[i])
    return [ticks[i] for i in order], [labels[i] for i in order]


def _scale_note(axis, scale, color=None, tick=False):
    """Write a small ``x scale`` note next to the label of ``axis``, below an x label and above a y label.
    :func:`paper_fonts` resizes it with the other texts."""
    below = isinstance(axis, mpl.axis.XAxis)
    axis.axes.annotate(f"$\\times$ {scale:g}", xy=(0.5, 0.0) if below else (0.5, 1.0),
                       xycoords=axis.label, xytext=(0, -1) if below else (0, 2),
                       textcoords="offset points", ha="center",
                       va="top" if below else "bottom", rotation=0 if below else 90,
                       fontsize=FS.tick if tick else FS.annot, color=color,
                       gid="scale_note_tick" if tick else None)


def _profile_ray_thetas(n):
    """The first ``n`` profile-ray angles. Beyond :data:`PROFILE_RAY_THETAS` they continue in golden-angle steps."""
    golden = 0.5 * (5.0 ** 0.5 - 1.0)
    base = list(PROFILE_RAY_THETAS)
    return (base + [((k + 1) * golden) % 1.0 for k in range(n - len(base))])[:n]


def _ray_line(lr, lth, th0):
    """Per line, the logical r of its crossing closest to the ray ``theta = th0``. Reading each line at its own
    radius on the ray keeps the radial width of an island chain visible in the profile."""
    dth = np.abs(((lth - th0 + 0.5) % 1.0) - 0.5)      # (nL, nC) circular distance
    k = np.argmin(dth, axis=1)
    return lr[np.arange(lr.shape[0]), k]


def _padded(v, pad=0.06, floor=0.0):
    lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
    span = max(hi - lo, floor)
    mid = 0.5 * (lo + hi)
    return mid - 0.5 * span - pad * span, mid + 0.5 * span + pad * span


@house_style()
def render_section(R, Z, iota, keep, *, logical, axis_RZ, nfp, title=None, subtitle=None, pressure=None,
                   pressure_label=r"$p$", iota_lim=None, p_lim=None, window=None, profile_rays=1,
                   axis_marker=True, dot_scale=1.0, labels=("R", "Z")):
    """A figure of one Poincare section in three panels: the ``(R, Z)`` crossings coloured by the iota of their
    line, the same crossings in the logical ``(theta, r)`` plane, and iota against logical r read along
    ``profile_rays`` poloidal rays.

    The inputs are plain arrays, for example from a section of :func:`mrx.diagnostics.poincare.poincare`.
    ``R``, ``Z`` and ``logical = (r, theta)`` have shape ``(line, crossing)``, ``iota`` and ``keep`` one entry
    per line, and ``axis_RZ`` is the magnetic axis at the plane. Lines with ``keep`` false are drawn in grey.
    The iota colour bar is ticked at the rationals of :func:`resonant_rationals`.

    With ``pressure`` (one value per crossing), the lower half of the section (below the axis, and
    ``theta >= 1/2`` in the logical panel) is coloured by :data:`PRESSURE_SCALE` times the pressure instead,
    and the profile panel shows each line's mean pressure and its spread on a second axis.

    ``iota_lim``, ``p_lim`` (each ``(lo, hi)``) and ``window`` (``((R0, R1), (Z0, Z1))``) fix the scales, so
    that several figures can be compared. ``dot_scale`` multiplies the marker size, ``axis_marker`` draws the
    axis, ``labels`` names the section's axes, and ``title=None`` leaves out the figure title. Returns ``(fig, (ax_section, ax_logical,
    ax_profile))``.
    """
    has_p = pressure is not None
    lr, lth = np.asarray(logical[0]), np.asarray(logical[1])
    fig = plt.figure(figsize=(16.5, 4.8), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.03, hspace=0.02)
    ax, lx, bx = fig.subplots(1, 3, width_ratios=[1.2, 0.9, 1.15])

    good = iota[keep][np.isfinite(iota[keep])]
    if iota_lim is not None:
        lo, hi = float(iota_lim[0]), float(iota_lim[1])
    else:
        lo, hi = (float(np.min(good)), float(np.max(good))) if good.size else (0.0, 1.0)
    if hi - lo < 1e-9:
        lo, hi = lo - 5e-3, hi + 5e-3

    # ~10^4 crossings want a hairline marker, ~10^2 a visible one
    npts = max(int(keep.sum()) * R.shape[1], 1)
    size = dot_scale * float(np.clip(3000.0 / npts, 0.35, 15.0))
    colour = np.broadcast_to(iota[:, None], R.shape)
    shown2 = np.broadcast_to(keep[:, None], R.shape)

    # the split is per crossing, at the mean Z of the magnetic axis
    z_axis = float(np.mean(np.asarray(axis_RZ[1])))
    upper = Z >= z_axis if has_p else np.ones_like(R, dtype=bool)
    sel_iota = shown2 & upper
    sc = ax.scatter(R[sel_iota], Z[sel_iota], c=colour[sel_iota], s=size, vmin=lo, vmax=hi, cmap=SECTION_CMAP,
                    linewidths=0, rasterized=True)
    psc, p_range = None, {}
    if has_p:
        pressure = PRESSURE_SCALE * pressure
        p_range = {"vmin": p_lim[0], "vmax": p_lim[1]} if p_lim is not None else {}
        sel_p = shown2 & ~upper
        if sel_p.any():
            psc = ax.scatter(R[sel_p], Z[sel_p], c=pressure[sel_p], s=size, cmap=PRESSURE_CMAP, linewidths=0,
                             rasterized=True, **p_range)
    res_ticks, res_labels = resonant_rationals(lo, hi, int(nfp))
    if (~keep).any():
        ax.scatter(R[~keep], Z[~keep], c="0.55", s=size, linewidths=0,
                   rasterized=True, label=f"lost ({int((~keep).sum())})")
        ax.legend(loc="upper right", fontsize=FS.annot, markerscale=4)
    if axis_marker:
        # one marker at the mean and a hairline through the axis' wander over the saves
        aR, aZ = np.asarray(axis_RZ[0]), np.asarray(axis_RZ[1])
        ax.plot(aR, aZ, "-", color="0.35", lw=0.4, alpha=0.6, zorder=4)
        ax.plot(np.mean(aR), np.mean(aZ), "k+", ms=7, mew=1.2, zorder=5)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.12 if psc is not None else 0.02)
    # below the bar: beside it the wide Farey labels squeeze it against the next panel
    cbar.ax.set_xlabel(r"$\iota$", fontsize=FS.title)
    if res_ticks:
        cbar.set_ticks(res_ticks)
        cbar.set_ticklabels(res_labels)
    if psc is not None:
        capped = p_lim is not None and float(np.nanmax(pressure[sel_p])) > p_lim[1]
        pbar = fig.colorbar(psc, ax=ax, fraction=0.046, pad=0.02, extend="max" if capped else "neither")
        pbar.ax.set_xlabel(pressure_label, fontsize=FS.title)
        _scale_note(pbar.ax.xaxis, PRESSURE_SCALE)
        pbar.ax.tick_params(labelsize=FS.annot)
        ax.axhline(z_axis, color="0.35", lw=0.6, ls=":", zorder=1)

    # equal aspect only while the two spans are within a factor of 20 (an iota = 0 section is a line)
    xlim = _padded(R[keep])
    ylim = _padded(Z[keep], floor=0.04 * (xlim[1] - xlim[0]))
    if window is not None:
        xlim, ylim = (tuple(float(v) for v in pair) for pair in window)
    spans = (xlim[1] - xlim[0], ylim[1] - ylim[0])
    to_scale = max(spans) / max(min(spans), 1e-30) < 20.0
    ax.set_aspect("equal" if to_scale else "auto")
    if np.ptp(Z[keep]) < 1e-6 * (xlim[1] - xlim[0]):
        ax.text(0.5, 0.86, "iota = 0: every line is a fixed point of the\n"
                           "return map, so each surface is a single dot",
                transform=ax.transAxes, ha="center", fontsize=FS.annot, color="0.35")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])

    # the logical chart, split by its own coordinate: iota on theta < 1/2, p on theta >= 1/2
    top = lth < 0.5 if has_p else np.ones_like(R, dtype=bool)
    csel_iota = shown2 & top
    lx.scatter(lth[csel_iota], lr[csel_iota], c=colour[csel_iota], s=size, vmin=lo, vmax=hi, cmap=SECTION_CMAP,
               linewidths=0, rasterized=True)
    csel_p = shown2 & ~top
    if csel_p.any():
        lx.scatter(lth[csel_p], lr[csel_p], c=pressure[csel_p], s=size, cmap=PRESSURE_CMAP, linewidths=0,
                   rasterized=True, **p_range)
    if (~keep).any():
        lx.scatter(lth[~keep], lr[~keep], c="0.55", s=size, linewidths=0, rasterized=True)
    lx.set_xlim(0.0, 1.0)
    lx.set_ylim(0.0, 1.0)
    lx.set_xlabel(r"$\theta$")
    lx.set_ylabel(r"$r$")

    # the profile: one marker per line and ray, no connecting curve (an island chain is a shelf at n/m)
    markers = ["o", "^", "v", "D"]
    if has_p:
        pmean, pstd = np.nanmean(pressure, axis=1), np.nanstd(pressure, axis=1)
        px = bx.twinx()
    for i, th0 in enumerate(_profile_ray_thetas(profile_rays)):
        mk = markers[i % len(markers)]
        r_line = _ray_line(lr, lth, th0)
        m = keep & np.isfinite(r_line)
        if not m.any():
            continue
        bx.plot(r_line[m], iota[m], linestyle="none", marker=mk, ms=0.9, color=IOTA_COLOR)
        if has_p:
            px.errorbar(r_line[m], pmean[m], yerr=pstd[m], fmt=mk, ms=0.9,
                        color=P_COLOR, ecolor=P_COLOR, elinewidth=0.4, capsize=0)
        lx.axvline(th0, color="black", linestyle=":" if th0 == 0.5 else "-", lw=1.0,
                   alpha=0.85, zorder=6)      # the theta = 1/2 seam dotted, the other rays solid
    bx.set_xlabel(r"$r$")
    bx.set_ylabel(r"$\iota$", color=IOTA_COLOR)
    bx.tick_params(axis="y", labelcolor=IOTA_COLOR)
    if has_p:
        px.set_ylabel(pressure_label, color=P_COLOR)
        _scale_note(px.yaxis, PRESSURE_SCALE, color=P_COLOR, tick=True)
        px.tick_params(axis="y", labelcolor=P_COLOR)
        if p_lim is not None:
            px.set_ylim(*p_lim)
    for value, lab in zip(res_ticks, res_labels):
        bx.axhline(value, color="0.55", lw=0.6, ls="--", zorder=0)
        bx.annotate(lab, (0.995, value), xycoords=("axes fraction", "data"),
                    ha="right", va="bottom", fontsize=FS.annot, color="0.4")
    if iota_lim is not None:
        bx.set_ylim(lo, hi)
    bx.grid(alpha=0.3)

    if title is not None:
        sup = title if to_scale else f"{title}   -   AXES NOT TO SCALE"
        if subtitle:
            sup = f"{sup}   |   {subtitle}"
        if has_p:
            sup = f"{sup}   |   {pressure_label} $\\times$ {PRESSURE_SCALE:g}"
        fig.suptitle(sup, fontsize=FS.title)
    return fig, (ax, lx, bx)


def paper_fonts(fig, *, label_size=9.0, page_width=6.5):
    """Resize a :func:`render_section` figure in place to ``page_width`` inches, with axis labels at
    ``label_size`` points and ticks and annotations scaled in the house proportions of
    :data:`~mrx.diagnostics.plotstyle.FS`."""
    scale = label_size / FS.label
    label_sz, tick_sz, annot_sz = FS.label * scale, FS.tick * scale, FS.annot * scale
    w0, h0 = fig.get_size_inches()
    fig.set_size_inches(page_width, page_width * h0 / w0)
    for a in fig.axes:                              # panels, twins and colour bars alike
        a.tick_params(labelsize=tick_sz)
        a.xaxis.label.set_size(label_sz)
        a.yaxis.label.set_size(label_sz)
        for t in a.texts:                           # rational labels and notes, the axis scale note at tick size
            t.set_fontsize(tick_sz if t.get_gid() == "scale_note_tick" else annot_sz)
        leg = a.get_legend()
        if leg is not None:
            for txt in leg.get_texts():
                txt.set_fontsize(annot_sz)


#: The pressure panel's label of :func:`plot_archive`: the weak pressure over the field's mean magnetic pressure.
PRESSURE_LABEL = r"$p_{\mathrm{norm}}$"


def plot_archive(archive, out, *, fields=None, planes=None, pressure=True, profile_rays=1, dot_scale=0.15,
                 label_size=9.0, page_width=6.5, dpi=600, pgf=False, window=None, iota_lim=None, p_lim=None):
    """Draw the Poincare sections of a trace archive (:func:`mrx.diagnostics.poincare.trace_archive`, or the
    loaded ``trace.npz`` of ``scripts/poincare_trace.py``) in the layout of the paper, and return the figures
    as ``{(field, plane): fig}``, still open.

    Every field and plane of the archive is drawn, or the subsets ``fields`` and ``planes``. All pages share
    ONE iota colour scale and ONE pressure scale, unless ``iota_lim`` and ``p_lim`` fix them. The pressure
    shown is the weak pressure over the field's mean magnetic pressure, and ``pressure=False`` leaves it out.
    Each page is written to ``out`` as PDF and PNG, ``poincare[_<field>]_zeta<plane>``, with the field name only
    when the archive holds more than one field, and with ``pgf`` also as a .pgf in ``out/pgf`` (needs TeX
    Live). The other arguments are those of :func:`render_section` and :func:`paper_fonts`, and ``dpi`` is the
    resolution of the rasterised crossings.
    """
    os.makedirs(out, exist_ok=True)
    all_fields = [str(f) for f in archive["fields"]]
    which = list(fields) if fields else all_fields
    for n in which:
        if n not in all_fields:
            raise ValueError(f"field {n!r} is not in the archive ({all_fields})")
    shown_planes = [float(v) for v in archive["planes"]]
    if planes:
        shown_planes = [pl for pl in shown_planes if any(abs(pl - v) < 1e-9 for v in planes)]
    nfp = int(archive["nfp"])
    # archives traced before 2026-10-06 hold cylindrical sections and no labels
    labels = tuple(str(v) for v in archive.get("section_labels", ("R", "Z")))
    per = {n: {k: np.asarray(archive[f"{n}_{k}"]) for k in ("iota", "keep", "shown")} for n in which}
    if iota_lim is None:
        iota_lim = (min(float(per[m]["iota"][per[m]["shown"]].min()) for m in which if per[m]["shown"].any()),
                    max(float(per[m]["iota"][per[m]["shown"]].max()) for m in which if per[m]["shown"].any()))
    presses = {(n, pl): (np.asarray(archive[f"{n}_zeta{pl:g}_pressure"]) / (0.5 * float(archive[f"{n}_bsq"]))
                         if pressure else None) for n in which for pl in shown_planes}
    ps = [PRESSURE_SCALE * presses[n, pl][per[n]["keep"]] for n in which for pl in shown_planes if pressure]
    if p_lim is None and ps:
        lo_p, hi_p = min(float(np.nanmin(v)) for v in ps), max(float(np.nanmax(v)) for v in ps)
        p_lim = (lo_p - 0.05 * (hi_p - lo_p), hi_p + 0.05 * (hi_p - lo_p))
    figs = {}
    for n in which:
        for pl in shown_planes:
            R, Z, aR, aZ, lr, lth = (np.asarray(archive[f"{n}_zeta{pl:g}_{k}"])
                                     for k in ("R", "Z", "axisR", "axisZ", "logr", "logth"))
            fig, _ = render_section(R, Z, per[n]["iota"], per[n]["keep"], logical=(lr, lth), axis_RZ=(aR, aZ),
                                    nfp=nfp, pressure=presses[n, pl], pressure_label=PRESSURE_LABEL,
                                    axis_marker=False, dot_scale=dot_scale, iota_lim=iota_lim, p_lim=p_lim,
                                    window=window, profile_rays=profile_rays,
                                    labels=labels)
            paper_fonts(fig, label_size=label_size, page_width=page_width)
            stem = os.path.join(out, f"poincare{'' if len(all_fields) == 1 else '_' + n}_zeta{pl:g}")
            # the tight box crops the left margin of the layout (colour bars) here, not in the including document
            tight = dict(bbox_inches="tight", pad_inches=0.02)
            fig.savefig(stem + ".pdf", dpi=dpi, **tight)     # dpi sets the resolution of the rasterised scatter
            fig.savefig(stem + ".png", dpi=dpi, **tight)
            print(f"  -> {stem}.pdf", flush=True)
            if pgf:
                # pdflatex typesets the figure in the document's own fonts (rcfonts off), and the scatter is a
                # raster -img*.png next to it in pgf/
                pgf_dir = os.path.join(out, "pgf")
                os.makedirs(pgf_dir, exist_ok=True)
                with mpl.rc_context({"pgf.texsystem": "pdflatex", "pgf.rcfonts": False,
                                     "pgf.preamble": r"\providecommand{\mathdefault}[1]{#1}"}):
                    fig.savefig(os.path.join(pgf_dir, os.path.basename(stem) + ".pgf"), backend="pgf", dpi=dpi,
                                **tight)
                print(f"  -> {pgf_dir}/{os.path.basename(stem)}.pgf", flush=True)
            figs[n, pl] = fig
    return figs
