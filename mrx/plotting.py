"""Matplotlib figures of scalar fields on a torus, of relaxation traces and
of Poincaré sections.

* :func:`torus_grids` samples a map on poloidal cuts and on the boundary
  surface (:func:`get_2d_grids` is one such plane); :func:`plot_torus`
  draws the surface as a wireframe with the cuts coloured by a scalar,
  :func:`plot_crossections_separate` the same cuts side by side in the
  ``(R, z)`` plane.
* :func:`plot_twin_axis` draws two traces against a shared x with separate
  y axes -- the standard figure for a force residual next to an energy or a
  helicity.
* :func:`render_section` draws a Poincaré section traced by
  :mod:`mrx.poincare`; :func:`section_figure` cuts one plane of a trace and
  draws it.

``scripts/plot_relaxation.py`` and ``scripts/poincare_relax.py`` make the
figures of a ``scripts/relax.py`` run from these.
"""

from typing import Callable, Optional

import os

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

import mrx
from mrx.plotstyle import (FIELD_CMAP, FS, IOTA_COLOR, LEFT, P_COLOR, PRESSURE_CMAP, RIGHT,
                           SECTION_CMAP, SectionLimits, house_style)


def get_2d_grids(
    F: Callable,
    cut_value: float = 0,
    cut_axis: int = 2,
    nx: int = 64,
    ny: int = 64,
    nz: int = 64,
    tol1: float = 1e-6,
    invert_z: bool = False,
):
    """Sample the map ``F`` on the logical plane ``x_{cut_axis} = cut_value``
    (``cut_axis`` 0, the boundary surface at fixed r, or 2, a poloidal cut
    at fixed zeta).

    The other two logical axes are ``linspace`` over ``[0, 1]``, the radial
    one ``[tol1, 1 - tol1]`` (``1e-6`` keeps the sample off the polar axis
    and off ``r = 1``, where the spline map's derivative is not defined);
    ``invert_z`` reverses the toroidal axis to orient a surface's normal.

    Returns ``(x, y, (Y1, Y2, Y3), (x1, x2, x3))``: the flat logical points
    ``x`` (``(n1 n2, 3)``), their images ``y = F(x)``, the images reshaped
    to the ``(n1, n2)`` plane for ``plot_surface``/``contourf``, and the
    three 1-D logical axes.
    """
    _x1 = jnp.linspace(tol1, 1.0 - tol1, nx)
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


def torus_grids(F: Callable, cuts: int, n: int = 48):
    """``(zetas, grids_pol, grid_surface)`` for :func:`plot_torus` and
    :func:`plot_crossections_separate`: ``cuts`` poloidal cuts over one
    field period (the map's logical zeta) at ``n x n`` points, and the
    boundary surface at ``4n x 4n``, sampled with zeta reversed so its
    normal points outward for the wireframe shading."""
    zetas = np.arange(cuts) / cuts
    grids_pol = [get_2d_grids(F, cut_axis=2, cut_value=float(z), nx=n, ny=n, nz=1)
                 for z in zetas]
    grid_surface = get_2d_grids(F, cut_axis=0, cut_value=1.0 - 1e-6,
                                ny=4 * n, nz=4 * n, invert_z=True)
    return zetas, grids_pol, grid_surface


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
    gridlinewidth: float = 0.01,
    cstride: int = 4,
    elev: float = 30,
    azim: float = 140,
    cbar_label: Optional[str] = None,
):
    """The boundary surface as a wireframe with poloidal cuts coloured by ``p_h``.

    ``p_h`` maps a logical point ``(3,)`` to a scalar (a 0-form's
    ``DiscreteFunction``, or a pushed-forward form); ``grids_pol`` and
    ``grid_surface`` are :func:`torus_grids`'s, over ONE field period. One
    colour scale over all cuts; ``cbar_label`` adds the colour bar. Returns
    ``(fig, ax)``.
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

    ax.view_init(elev=elev, azim=azim)
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
    """The poloidal cuts of :func:`plot_torus` side by side in the ``(R, z)`` plane.

    One filled contour per cut, common axis limits, one shared colour bar,
    a boxed ``zeta`` label per panel and an ``(R, z)`` arrow pair on the
    first. Returns ``(fig, axes)``.
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
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(0.98, 0.98, rf"$\zeta = {float(zeta):.2f}$", transform=ax.transAxes,
                fontsize=FS.label, ha="right", va="top", zorder=10,
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
                fontsize=FS.label + 2, ha="center", va="bottom")
    anchor.text(x0 + arrow_len + 0.01, y0 - 0.01, r"$R$", transform=anchor.transAxes,
                fontsize=FS.label + 2, ha="left", va="center")

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    cbar_ax.tick_params(labelsize=FS.tick)
    cbar = fig.colorbar(last_c, cax=cbar_ax,
                        format=mticker.ScalarFormatter(useMathText=True))
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()
    cbar.ax.yaxis.get_offset_text().set_fontsize(FS.tick)
    return fig, axes


@house_style()
def save_figure(fig, png_path, pgf=True, dpi=None):
    """Save a figure as PNG and, with ``pgf``, as ``pgf/<stem>.pgf`` in a
    subfolder beside it: the same figure through matplotlib's pgf backend,
    vector LaTeX for lines and text, rasterized artists as a high-dpi PNG
    that the backend writes next to the .pgf. Needs ``xelatex`` on PATH;
    without it the PNG is still written and the PGF skipped with a message.
    The including document must ``\\usepackage[strings]{underscore}`` (the
    backend writes underscores raw) and ``\\providecommand{\\mathdefault}[1]{#1}``
    (log-axis tick labels); both are put in the pgf preamble so the file's
    own header lists them."""
    fig.savefig(png_path, dpi=dpi)
    if not pgf:
        return
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
    x_label: str = "iteration",
    grid: bool = True,
    left_plot_kwargs: Optional[dict] = None,
    right_plot_kwargs: Optional[dict] = None,
    ax=None,
):
    """Two traces against a shared x with separate y axes (``twinx``).

    Each side is log (``semilogy``) or linear on its own; the y label and
    ticks take the series colour. Without ``x_*`` the abscissa is
    ``arange(len(y))``. The house styles ``LEFT`` and ``RIGHT`` (colour,
    marker, line style, marker size) are the defaults that
    ``left_plot_kwargs``/``right_plot_kwargs`` override. With ``ax`` the
    pair is drawn into that existing axes (a panel of a larger figure, whose
    layout is then the caller's), otherwise into a new figure. Returns
    ``(fig, (ax_left, ax_right))``.
    """
    if ax is None:
        fig, ax1 = plt.subplots(figsize=(8, 3))
    else:
        fig, ax1 = ax.figure, ax
    ax2 = ax1.twinx()
    sides = (
        (ax1, left_y, x_left, left_log, left_label,
         {**{k: LEFT[k] for k in ("color", "linestyle", "marker", "markersize")},
          **(left_plot_kwargs or {})}),
        (ax2, right_y, x_right, right_log, right_label,
         {**{k: RIGHT[k] for k in ("color", "linestyle", "marker", "markersize")},
          **(right_plot_kwargs or {})}),
    )
    for ax, y, x, log, label, kwargs in sides:
        y = np.asarray(y)
        x = np.arange(len(y)) if x is None else np.asarray(x)
        (ax.semilogy if log else ax.plot)(x, y, **kwargs)
        ax.set_ylabel(label, color=kwargs["color"])
        ax.tick_params(axis="y", labelcolor=kwargs["color"])
    ax1.set_xlabel(x_label)
    if grid:
        ax1.grid(True, which="both")
    return fig, (ax1, ax2)


# ---------------------------------------------------------------------------
# Poincaré section figure (the tracer, mrx.poincare, stays headless)
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


#: Poloidal rays for the logical profile, in order (``profile_rays`` of
#: :func:`render_section` takes the first ``n``). theta = 0 is deliberately
#: NOT used: it is the branch point of the poloidal angle, where the ray flips
#: across the midplane (see :func:`_ray_line`). theta = 0.5 is where odd island
#: chains are fattest (their O-points); 1/3 and 0.2 sit off the symmetry line
#: and are non-resonant. Further rays fill in.
PROFILE_RAY_THETAS = (1.0 / 3.0, 0.5, 0.2, 1.0 / 6.0, 0.25, 0.75)

#: The pressure is drawn at this multiple of its value (colour and profile):
#: the natural scale of p in code units is ~1e-2, and 100 p reads in units.
PRESSURE_SCALE = 100.0


def _ray_line(lr, lth, pressure, th0):
    """Per-line logical r and p at the crossing nearest the poloidal ray
    ``theta = th0`` (circular nearest, one crossing per line): r is the radius
    that field line actually sits at where it crosses the ray -- the surface it
    is on -- so a resonant line reads at its own r and an island chain keeps its
    true radial width in the profile.

    Averaging each line over its turns was tried (to smooth a ray drawn on the
    physical section); it collapses island lines to their mean radius and erases
    the resonant shelf, so it is NOT used. Those physical ray-markers are gone,
    and p is sampled the same way as r, so both stay per-line, un-averaged.
    """
    dth = np.abs(((lth - th0 + 0.5) % 1.0) - 0.5)      # (nL, nC) circular distance
    k = np.argmin(dth, axis=1)                          # nearest crossing per line
    rows = np.arange(lr.shape[0])
    p_at = None if pressure is None else pressure[rows, k]
    return lr[rows, k], p_at


@house_style()
def render_section(R, Z, iota, iota_err, keep, *, title, subtitle, axis_RZ, nfp, logical,
                   pressure=None, pressure_label=r"$p$", limits=None, iota_scatter=None,
                   profile_rays=3):
    """The section coloured by iota, the same crossings in the logical chart,
    and the iota (and p) profile.

    Pure arrays in, so a run can be re-rendered from its archive without
    rebuilding the map -- which is the expensive half of producing it:
    ``R``, ``Z`` the crossings per line and plane crossing, ``logical`` their
    ``(r, theta)``, ``axis_RZ`` the magnetic axis at each crossing, ``keep``
    the lines to draw and fit.

    ``limits`` (a :class:`~mrx.plotstyle.SectionLimits`) pins what a movie or
    a side-by-side set must hold fixed between frames: ``iota`` the colour
    and profile scale, ``RZ`` (``((R0, R1), (Z0, Z1))`` of the section
    panel), ``z_split`` (the split line), ``x`` (the profiles' abscissa) and
    ``p`` (the pressure panel's ordinate, in drawn units); any field left
    ``None`` is fitted from this figure's own lines.  Two sections drawn on
    iota limits fitted separately are not comparable by colour at all -- the
    same hue means a different transform in each -- so any caller producing
    a set that is meant to be read side by side (two relaxation states, a
    plane scan) must pass one shared pair.

    ``pressure`` is per-crossing, the same shape as ``R``, drawn at
    :data:`PRESSURE_SCALE` times its value (the labels say so); a harmonic
    (vacuum-like) field leaves it ``None``. When it is given, the section is
    coloured by iota ABOVE the magnetic axis and by p BELOW it (the split is
    at the MAGNETIC axis, not ``Z = 0``, which would cut a Shafranov-shifted
    plasma off-centre), the logical chart by its own coordinate, and the
    pressure profile joins the iota profile on the right axis of the same
    panel with a one-standard-deviation band over each line's crossings: on a
    flux surface of an equilibrium p is constant and the band collapses; on
    an island chain or a chaotic line the band width measures how far that
    line is from ``B . grad p = 0``.

    Every kept line is drawn and fitted, chaotic ones included: the iota
    profile carries a ribbon, so a line without a rotational transform shows as
    a point with a wide ribbon rather than as a separate category. The ribbon
    is ``iota_scatter`` when given -- the std of iota over K equal ζ-windows
    (:func:`mrx.poincare._iota_window_scatter`), the along-line spread that
    reads like the pressure band -- else ``iota_err`` (the whole-line fit
    RMS/N, see :func:`trace_and_classify`).
    """
    lim = SectionLimits() if limits is None else limits
    has_p = pressure is not None
    p_label = f"{pressure_label} $\\times$ {PRESSURE_SCALE:g}"
    # Panels, left to right: the section (with its colourbars), the logical
    # chart (where an island chain or an off-centre axis is seen at a glance:
    # nested surfaces are horizontal bands there), and the profiles -- iota
    # on the left axis and, when p is given, p on the right axis of the same
    # panel.
    fig = plt.figure(figsize=(16.5, 4.8), constrained_layout=True)
    ax, lx, bx = fig.subplots(1, 3, width_ratios=[1.45, 0.9, 1.15])
    axes = {"ax": ax, "lx": lx, "bx": bx}

    good = iota[keep][jnp.isfinite(iota[keep])]
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
    aR, aZ = jnp.asarray(axis_RZ[0]), jnp.asarray(axis_RZ[1])
    if has_p:
        z_axis = float(jnp.mean(aZ)) if lim.z_split is None else float(lim.z_split)
        upper = Z >= z_axis
    else:
        upper = jnp.ones_like(R, dtype=bool)
    # `keep` selects LINES, `upper` selects CROSSINGS.
    keep2 = jnp.broadcast_to(keep[:, None], R.shape)
    sel_iota = keep2 & upper
    sc = ax.scatter(R[sel_iota], Z[sel_iota], c=colour[sel_iota],
                    s=size, vmin=lo, vmax=hi, cmap=SECTION_CMAP, linewidths=0,
                    rasterized=True)
    psc = None
    p_range = {}
    if has_p:
        # Chaotic lines keep their grey in BOTH halves: colouring one half of
        # a line and greying the other reads as two different objects.
        sel_p = keep2 & ~upper
        p_range = ({"vmin": lim.p[0], "vmax": lim.p[1]}
                   if lim.p is not None else {})                # pinned in a movie
        if sel_p.any():
            psc = ax.scatter(R[sel_p], Z[sel_p], c=PRESSURE_SCALE * pressure[sel_p], s=size,
                             cmap=PRESSURE_CMAP, linewidths=0, rasterized=True, **p_range)
    res_ticks, res_labels = resonant_rationals(lo, hi, int(nfp))
    if (~keep).any():
        ax.scatter(R[~keep], Z[~keep], c="0.55", s=size, linewidths=0,
                   rasterized=True, label=f"lost ({int((~keep).sum())})")
        ax.legend(loc="upper right", fontsize=FS.annot, markerscale=4)
    # ONE marker at the mean of the axis, plus a hairline through its wander
    # (a marker at every save stacks into a blob that reads as a failed line).
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

    # The SAME crossings in the logical chart: r against theta, both in
    # [0,1]. The chart splits by ITS coordinate, per crossing: iota on theta <
    # 1/2 (the top half, theta increases downward), p on theta >= 1/2. The
    # physical panel's divider Z = z_axis maps to a theta interval that wraps
    # around on planes away from zeta = 0, so it is not reused here.
    lr, lth = logical
    top = jnp.asarray(lth) < 0.5 if has_p else jnp.ones_like(R, dtype=bool)
    csel_iota = keep2 & top
    lx.scatter(lr[csel_iota], lth[csel_iota], c=colour[csel_iota], s=size, vmin=lo,
               vmax=hi, cmap=SECTION_CMAP, linewidths=0, rasterized=True)
    csel_p = keep2 & ~top
    if csel_p.any():
        lx.scatter(lr[csel_p], lth[csel_p], c=PRESSURE_SCALE * pressure[csel_p], s=size,
                   cmap=PRESSURE_CMAP, linewidths=0, rasterized=True, **p_range)
    if (~keep).any():
        lx.scatter(lr[~keep], lth[~keep], c="0.55", s=size, linewidths=0,
                   rasterized=True)
    lx.set_xlim(0.0, 1.0)
    # theta increases DOWNWARD so the pressure half (below the magnetic
    # axis) sits at the bottom, aligned with the physical panel where p is
    # also below the axis.
    lx.set_ylim(1.0, 0.0)
    lx.set_xlabel(r"$r$")
    lx.set_ylabel(r"$\theta$")

    # ---- profile panel: iota (and p) against logical r -----------------------
    # sampled along ``profile_rays`` poloidal rays (:data:`PROFILE_RAY_THETAS`).
    # Each ray is one LINE STYLE, marked as the theta = theta0 line in the
    # logical chart so the reader can place it. Where the rays agree logical
    # r is a faithful surface label; where they fan (edge, islands) it is
    # not, and the fan is the signal.
    band = iota_err if iota_scatter is None else iota_scatter
    styles = ["-", "--", ":", "-."]
    lr_all, lth_all = np.asarray(lr), np.asarray(lth)
    sn = np.asarray(keep)
    iota_n, band_n = np.asarray(iota), np.asarray(band)
    pn = None if pressure is None else np.asarray(pressure)
    pstd = None if pn is None else PRESSURE_SCALE * np.nanstd(pn, axis=1)
    px = bx.twinx() if has_p else None
    thetas = PROFILE_RAY_THETAS[:max(int(profile_rays), 1)]
    for i, th0 in enumerate(thetas):
        ls = styles[i % len(styles)]
        r_line, p_at = _ray_line(lr_all, lth_all, pn, th0)
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
            pm = PRESSURE_SCALE * p_at
            px.plot(rr, pm[m][o], color=P_COLOR, linestyle=ls, lw=1.2)
            px.fill_between(rr, (pm - pstd)[m][o], (pm + pstd)[m][o],
                            color=P_COLOR, alpha=0.10, lw=0)
        lx.axhline(th0, color="black", linestyle=ls, lw=1.0, alpha=0.85, zorder=6)
    bx.set_xlabel(r"$r$")
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


def section_figure(seq, res, plane, *, title, subtitle, nfp, iota_limits=None):
    """One Poincaré section of a traced field at logical ``plane``: the
    crossings of :func:`mrx.poincare.section_RZ` through
    :func:`render_section`. ``res`` is the dict of
    :func:`mrx.poincare.trace_and_classify` (:func:`mrx.poincare.trace_sections`);
    ``iota_limits`` pins the iota scale of a set of planes or fields.
    Returns ``(fig, axes)``."""
    from mrx.poincare import section_RZ  # noqa: PLC0415  (the tracer stays headless)
    R, Z, aR, aZ, lr, lth = section_RZ(seq, res["ys"], res["axis"], res["saves_per_period"], plane)
    keep = ~(res["escaped"] | ~res["ok"])
    return render_section(
        R, Z, res["iota"], res["iota_err"], keep,
        title=f"{title}  |  $\\zeta = {plane:g}$ -- {R.shape[1]} crossings/line",
        subtitle=subtitle, axis_RZ=(aR, aZ), nfp=nfp, logical=(lr, lth),
        limits=SectionLimits(iota=iota_limits), iota_scatter=res["iota_scatter"])


def _padded(v, pad=0.06, floor=0.0):
    lo, hi = float(jnp.nanmin(v)), float(jnp.nanmax(v))
    span = max(hi - lo, floor)
    mid = 0.5 * (lo + hi)
    return mid - 0.5 * span - pad * span, mid + 0.5 * span + pad * span
