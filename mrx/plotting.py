"""
Plotting utilities for finite element analysis results.

This module provides functions for creating visualizations of convergence plots
and other analysis results using Plotly.
"""
# %%
import os
from typing import Callable, Optional

import diffrax
import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction, Pushforward
from mrx.mappings import cerfon_map

__all__ = ['converge_plot', 'get_3d_grids', 'get_2d_grids',
           'get_1d_grids', 'poincare_plot', 'pressure_plot', 'trace_plot']

# Base marker styles for different data series
base_markers = [
    'circle', 'triangle-down', 'star', 'triangle-left', 'triangle-right',
    'triangle-ne', 'triangle-se', 'triangle-sw', 'triangle-nw',
    'square', 'pentagon', 'triangle-up', 'hexagon', 'hexagon2',
    'cross', 'x', 'diamond', 'diamond-open', 'line-ns', 'line-ew'
]

# Default color scale for plots
colorbar = 'Viridis'


def converge_plot(err: jnp.ndarray, ns: jnp.ndarray, ps: jnp.ndarray, qs: jnp.ndarray):
    """
    Create a convergence plot showing error vs. number of elements for different polynomial orders.

    This function generates a plotly figure showing the convergence behavior of
    numerical solutions for different polynomial orders (p) and quadrature rules (q).

    Args:
        err (numpy.ndarray): Error values of shape (len(ns), len(ps), len(qs))
        ns (numpy.ndarray): Array of number of elements
        ps (numpy.ndarray): Array of polynomial orders
        qs (numpy.ndarray): Array of quadrature rule orders

    Returns:
        plotly.graph_objects.Figure: A plotly figure showing the convergence plot
            with separate markers for polynomial orders and colors for quadrature rules.

    Notes:
        - Each polynomial order (p) is represented by a different marker style
        - Each quadrature rule (q) is represented by a different color
        - The plot includes both lines and markers for better visualization
        - Legend entries are added separately for markers and colors
    """
    import plotly.colors as pc
    import plotly.graph_objects as go
    markers = [base_markers[i % len(base_markers)] for i in range(len(ps))]
    colors = pc.sample_colorscale(colorbar, len(qs))
    fig = go.Figure()

    # Add main traces with both lines and markers
    for j, p in enumerate(ps):
        for k, q in enumerate(qs):
            fig.add_trace(go.Scatter(
                x=ns,
                y=err[:, j, k],
                mode='lines+markers',
                name=f'p={p}',
                marker=dict(symbol=markers[j], size=8, color=colors[k]),
                line=dict(color=colors[k], width=2),
                showlegend=False,
            ))

    # Add legend entries for polynomial orders (markers)
    for j, marker in enumerate(markers):
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(symbol=marker, color=colors[0], size=8),
            name=f'p = {ps[j]}',
            showlegend=True
        ))

    # Add legend entries for quadrature rules (colors)
    for j, color in enumerate(colors):
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            marker=dict(symbol=None, color=color, size=8),
            name=f'q = {qs[j]}',
            showlegend=True
        ))

    return fig


def get_3d_grids(F: Callable,
                 x_min: float = 0, x_max: float = 1,
                 y_min: float = 0, y_max: float = 1,
                 z_min: float = 0, z_max: float = 1,
                 nx: int = 16,
                 ny: int = 16,
                 nz: int = 16):
    """
    Get 3D grids for plotting.

    Parameters
    ----------
    F : callable
        Mapping from logical coordinates to physical coords: (r,theta,zeta)->(x,y,z)
    x_min : float
        Minimum value of the x coordinate.
    x_max : float
        Maximum value of the x coordinate.
    y_min : float
        Minimum value of the y coordinate.
    y_max : float
        Maximum value of the y coordinate.
    z_min : float
        Minimum value of the z coordinate.
    z_max : float
        Maximum value of the z coordinate.
    nx : int
        Number of grid points in the x direction.
    ny : int
        Number of grid points in the y direction.
    nz : int
        Number of grid points in the z direction.

    Returns
    -------
    _x : jnp.ndarray
        Grid points in the x direction.
    _y : jnp.ndarray
        Grid points in the y direction.
    _y1 : jnp.ndarray
        Grid points in the x direction.
    _y2 : jnp.ndarray
        Grid points in the y direction.
    _y3 : jnp.ndarray
        Grid points in the z direction.
    _x1 : jnp.ndarray
        Grid points in the x direction.
    _x2 : jnp.ndarray
        Grid points in the y direction.
    _x3 : jnp.ndarray
        Grid points in the z direction.
    """
    _x1 = jnp.linspace(x_min, x_max, nx)
    _x2 = jnp.linspace(y_min, y_max, ny)
    _x3 = jnp.linspace(z_min, z_max, nz)
    _x = jnp.array(jnp.meshgrid(_x1, _x2, _x3))
    _x = _x.transpose(1, 2, 3, 0).reshape(nx*ny*nz, 3)
    _y = jax.vmap(F)(_x)
    _y1 = _y[:, 0].reshape(nx, ny, nz)
    _y2 = _y[:, 1].reshape(nx, ny, nz)
    _y3 = _y[:, 2].reshape(nx, ny, nz)
    return _x, _y, (_y1, _y2, _y3), (_x1, _x2, _x3)


def get_2d_grids(
        F: Callable, cut_value: float = 0, cut_axis: int = 2,
        nx: int = 64, ny: int = 64, nz: int = 64, tol1: float = 1e-6,
        tol2: float = 0, tol3: float = 0,
        x_min: float = 0, x_max: float = 1,
        y_min: float = 0, y_max: float = 1,
        z_min: float = 0, z_max: float = 1,
        invert_x: bool = False, invert_y: bool = False,
        invert_z: bool = False):
    """
    Get 2D grids for plotting.
    Parameters
    ----------
    F : callable
        Mapping from logical coordinates to physical coords: (r,theta,zeta)->(x,y,z)
    cut_value : float
        Value of the cut to make.
    cut_axis : int
        Axis to cut on.
    nx : int
        Number of grid points in the x direction.
    ny : int
        Number of grid points in the y direction.
    nz : int
        Number of grid points in the z direction.
    tol1 : float
        Tolerance for the x direction.
    tol2 : float
        Tolerance for the y direction.
    tol3 : float
        Tolerance for the z direction.
    x_min : float
        Minimum value of the x coordinate.
    x_max : float
        Maximum value of the x coordinate.
    y_min : float
        Minimum value of the y coordinate.
    y_max : float
        Maximum value of the y coordinate.
    z_min : float
        Minimum value of the z coordinate.
    z_max : float
        Maximum value of the z coordinate.
    invert_x : bool
        Whether to invert the x direction.
    invert_y : bool
        Whether to invert the y direction.
    invert_z : bool
        Whether to invert the z direction.

    Returns
    -------
    _x : jnp.ndarray
        Grid points in the x direction.
    _y : jnp.ndarray
        Grid points in the y direction.
    _y1 : jnp.ndarray
        Grid points in the x direction.
    _y2 : jnp.ndarray
        Grid points in the y direction.
    _y3 : jnp.ndarray
        Grid points in the z direction.
    _x1 : jnp.ndarray
        Grid points in the x direction.
    _x2 : jnp.ndarray
        Grid points in the y direction.
    _x3 : jnp.ndarray
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
    _x = jnp.array(jnp.meshgrid(_x1, _x2, _x3))
    _x = _x.transpose(1, 2, 3, 0).reshape(n1 * n2, 3)
    _y = jax.vmap(F)(_x)
    _y1 = _y[:, 0].reshape(n1, n2)
    _y2 = _y[:, 1].reshape(n1, n2)
    _y3 = _y[:, 2].reshape(n1, n2)
    _y = jax.vmap(F)(_x)
    return _x, _y, (_y1, _y2, _y3), (_x1, _x2, _x3)


def get_1d_grids(
        F: Callable, zeta: float = 0, chi: float = 0,
        nx: int = 64, tol: float = 1e-6):
    """
    Get 1D grids for plotting.
    Parameters
    ----------
    F : callable
        Mapping from logical coordinates to physical coords: (r,theta,zeta)->(x,y,z)
    zeta : float
        Value of the zeta coordinate.
    chi : float
        Value of the chi coordinate.
    nx : int
        Number of grid points in the x direction.
    tol : float
        Tolerance for the grid.

    Returns
    -------
    _x : jnp.ndarray
        Grid points in the x direction.
    _y : jnp.ndarray
        Grid points in the y direction.
    _y1 : jnp.ndarray
        Grid points in the x direction.
    _y2 : jnp.ndarray
        Grid points in the y direction.
    _y3 : jnp.ndarray
        Grid points in the z direction.
    _x1 : jnp.ndarray
        Grid points in the x direction.
    _x2 : jnp.ndarray
        Grid points in the y direction.
    _x3 : jnp.ndarray
        Grid points in the z direction.
    """
    _x1 = jnp.linspace(tol, 1 - tol, nx)
    _x2 = jnp.ones(1) * chi
    _x3 = jnp.ones(1) * zeta
    _x = jnp.array(jnp.meshgrid(_x1, _x2, _x3))
    _x = _x.transpose(1, 2, 3, 0).reshape(nx, 3)
    _y = jax.vmap(F)(_x)
    _y1 = _y[:, 0]
    _y2 = _y[:, 1]
    _y3 = _y[:, 2]
    return _x, _y, (_y1, _y2, _y3), (_x1, _x2, _x3)


def trajectory_plane_intersections_jit(
        trajectories: jnp.ndarray, plane_val: float = 0.5, axis: int = 1):
    """
    Vectorized + jittable intersection with plane x_axis = plane_val.

    Parameters
    ----------
    trajectories : array (N, T, D)
    plane_val    : float
    axis         : int, which coordinate axis (default=1 for x_2).

    Returns
    -------
    intersections : array (N, T-1, D)
        Intersection points for each segment. Non-crossings are filled with NaN.
    mask : bool array (N, T-1)
        True if the corresponding segment contains an intersection.
    """
    x = trajectories[..., axis]                           # (N, T)
    diff = x - plane_val

    # if plane is zero, check for one point very small and another close to one:
    if plane_val == 0:
        diff = jnp.minimum(jnp.abs(diff), jnp.abs(diff - 1))

    # crossings: sign change or exact hit
    mask = (diff[..., :-1] * diff[..., 1:] <= 0)

    # interpolation fraction t in [0,1]
    denom = diff[..., :-1] - diff[..., 1:]
    t = jnp.where(mask, diff[..., :-1] / denom, jnp.nan)  # (N, T-1)

    # shape to broadcast into (N, T-1, 1)
    t = t[..., None]

    # segment start + t * (segment end - start)
    intersections = trajectories[:, :-1, :] + t * \
        (trajectories[:, 1:, :] - trajectories[:, :-1, :])

    return intersections, mask


def plot_crossections_separate(
        p_h: Callable, grids_pol: list, zeta_vals: list,
        textsize: int = 16, ticksize: int = 16,
        plot_centerline: bool = False):
    """
    Plot cross-sections of a function on a list of grids.

    Parameters
    ----------
    p_h : callable
        Function to plot.
    grids_pol : list
    zeta_vals : list
        Values of the zeta coordinate to plot.
    textsize : int
        Size of the text.
    ticksize : int
        Size of the ticks.
    plot_centerline : bool
        Whether to plot the centerline.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    axes : list
        List of axes objects.
    """
    numplots = len(grids_pol)
    fig, axes = plt.subplots(1, numplots, figsize=(16, 16/5))
    axes = axes.flatten()

    last_c = None
    for i, (ax, grid) in enumerate(zip(axes, grids_pol)):
        # # do a PCA to get the planar coordinates
        # mean = jnp.mean(grid[1], axis=0)
        # cov = jnp.cov(grid[1].T)
        # eigvals, eigvecs = jnp.linalg.eigh(cov)
        # # sort eigenvalues and eigenvectors
        # idx = jnp.argsort(eigvals)[::-1]
        # eigvals = eigvals[idx]
        # eigvecs = eigvecs[:, idx]
        # # project points onto the first two principal components
        # centered = grid[1] - mean
        # projected = centered @ eigvecs[:, :2]
        # nu1 = projected[:, 0].reshape(*grid[2][0].shape)
        # nu2 = projected[:, 1].reshape(*grid[2][0].shape)

        R = jnp.sqrt(grid[2][0]**2 + grid[2][1]**2)
        z = grid[2][2]

        vals = jax.vmap(p_h)(grid[0]).reshape(*grid[2][0].shape)

        # draw contour above the guide lines
        last_c = ax.contourf(R, z,
                             vals, 25, cmap="plasma", zorder=2)

        # ensure axis artists (like text/legend) are above the guide lines
        ax.set_axisbelow(False)

        if plot_centerline:
            ax.axvline(1.0, color='k', linestyle=":",
                       linewidth=1.5, zorder=3, clip_on=True)
        # ax.axhline(0.0, color='k', linestyle=":", linewidth=1.5, zorder=3, clip_on=True)

        ax.set_aspect("equal")

        # remove all ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # remove the box (all spines)
        for spine in ax.spines.values():
            spine.set_visible(False)

        try:
            zval = float(zeta_vals[i])
        except Exception:
            zval = float(jnp.asarray(zeta_vals[i]))
        label = rf"$\zeta = {zval:.2f}$"
        # place a small boxed text in the top-right of the axis (no legend handle/whitespace)
        ax.text(0.98, 0.98, label, transform=ax.transAxes,
                fontsize=textsize, ha='right', va='top', zorder=10,
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3', alpha=1.0))

        # force same limits across all subplots
    Rmins, Rmaxs, Zmins, Zmaxs = [], [], [], []
    for grid in grids_pol:
        R = jnp.sqrt(grid[2][0]**2 + grid[2][1]**2)
        z = grid[2][2]
        Rmins.append(R.min())
        Rmaxs.append(R.max())
        Zmins.append(z.min())
        Zmaxs.append(z.max())

    Rmin, Rmax = float(min(Rmins)), float(max(Rmaxs))
    Zmin, Zmax = float(min(Zmins)), float(max(Zmaxs))

    for ax in axes:
        ax.set_xlim(Rmin, Rmax)
        ax.set_ylim(Zmin, Zmax)

    # put ONE shared colorbar on the right, aligned with subplots
    fig.subplots_adjust(right=0.85)  # make space for colorbar
    # [left, bottom, width, height]
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    cbar_ax.tick_params(labelsize=ticksize)

    # add reference axis arrows (R to the right, z upwards) at the bottom-left of the first subplot
    try:
        anchor_ax = axes[0]
    except Exception:
        anchor_ax = axes if hasattr(axes, 'annotate') else None

    if anchor_ax is not None:
        x0, y0 = -0.01, -0.01        # anchor location in axis fraction coordinates
        arrow_len = 0.16           # length of each arrow in axis fraction units

        # upward arrow for z

        # annotate the center (dotted) line at the very top of the axis
        if plot_centerline:
            anchor_ax.text(0.5, 1.02, r"$R = 1$",
                           transform=anchor_ax.transAxes,
                           fontsize=textsize, ha='center', va='bottom', zorder=12,
                           bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0.2))
        anchor_ax.annotate('', xy=(x0, y0 + arrow_len), xytext=(x0, y0),
                           xycoords='axes fraction',
                           arrowprops=dict(arrowstyle='->', linewidth=1.5, color='k'))
        # rightward arrow for R
        anchor_ax.annotate('', xy=(x0 + arrow_len, y0), xytext=(x0, y0),
                           xycoords='axes fraction',
                           arrowprops=dict(arrowstyle='->', linewidth=1.5, color='k'))

        # labels for arrows
        anchor_ax.text(x0 - 0.01, y0 + arrow_len + 0.01, r"$z$",
                       transform=anchor_ax.transAxes, fontsize=textsize+2,
                       ha='center', va='bottom')
        anchor_ax.text(x0 + arrow_len + 0.01, y0 - 0.01, r"$R$",
                       transform=anchor_ax.transAxes, fontsize=textsize+2,
                       ha='left', va='center')

    cbar = fig.colorbar(last_c, cax=cbar_ax,
                        format=mticker.ScalarFormatter(useMathText=True))
    cbar.formatter.set_powerlimits((0, 0))  # always show scientific notation
    cbar.update_ticks()
    cbar.ax.yaxis.get_offset_text().set_fontsize(ticksize)

    # plt.tight_layout(rect=[0, 0, 0.85, 1])  # leave room for cbar
    return fig, axes


def plot_torus(p_h: Callable,
               grids_pol: list,
               grid_surface: list,
               figsize: tuple = (12, 8),
               labelsize: int = 20,
               ticksize: int = 16,
               gridlinewidth: float = 0.01,
               cstride: int = 4,
               elev: float = 30,
               azim: float = 140,
               noaxes: bool = False):
    """
    Plot a torus.

    Parameters
    ----------
    p_h : callable
        Function to plot.
    grids_pol : list
        List of grids to plot.
    grid_surface : list
        List of grid surfaces to plot.
    figsize : tuple
        Size of the figure.
    labelsize : int
        Size of the labels.
    ticksize : int
        Size of the ticks.
    gridlinewidth : float
        Width of the grid lines.
    cstride : int
        Stride for the color map.
    elev : float
        Elevation angle.
    azim : float
        Azimuthal angle.
    noaxes : bool
        Whether to plot the axes.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axes object.
    """
    vals = jnp.array([jax.vmap(p_h)(grid[0]).reshape(grid[2][0].shape)
                     for grid in grids_pol])

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    X = grid_surface[2][0]
    Y = grid_surface[2][1]
    Z = grid_surface[2][2]
    colors = plt.cm.plasma(jnp.zeros_like(X))
    ax.plot_surface(X, Y, Z, edgecolors=(0, 0, 0, 0.2),
                    rstride=cstride, cstride=cstride, shade=True,
                    alpha=0.0, linewidth=gridlinewidth)

    vals_np = np.asarray(vals)
    vals_min = float(vals_np.min())
    vals_max = float(vals_np.max())
    if vals_max == vals_min:
        vals_max = vals_min + 1e-12

    for (i, grid) in enumerate(grids_pol):
        X = grid[2][0]
        Y = grid[2][1]
        Z = grid[2][2]
        v = np.asarray(vals_np[i])
        colors = plt.cm.plasma((v - vals_min) / (vals_max - vals_min))
        ax.plot_surface(X, Y, Z, facecolors=colors, rstride=1,
                        cstride=1, shade=False, zsort='min', linewidth=0)

    # add colorbar
    # norm = mpl.colors.Normalize(vmin=vals_min, vmax=vals_max)
    # sm = mpl.cm.ScalarMappable(cmap='plasma', norm=norm)
    # sm.set_array(vals_np)  # provide data for the colorbar
    # cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.08)
    # cbar.set_label(r'$p$', fontsize=LABEL_SIZE)
    # cbar.ax.tick_params(labelsize=TICK_SIZE)

    # Remove grey background of 3D panes and ensure white background (robust across mpl versions)
    panes = [
        ("xaxis", (1.0, 1.0, 1.0, 1.0)),
        ("yaxis", (1.0, 1.0, 1.0, 1.0)),
        ("zaxis", (1.0, 1.0, 1.0, 1.0)),
    ]

    for name, color in panes:
        axis = getattr(ax, name, None)
        axis.set_pane_color(color)

    set_axes_equal(ax)

    # Move axis labels away from the axes so they don't overlap the ticks
    ax.set_xlabel(r'$x_1$', fontsize=labelsize, labelpad=14)
    ax.set_ylabel(r'$x_2$', fontsize=labelsize, labelpad=14)
    # increase z label padding so it isn't clipped by the figure edge
    ax.set_zlabel(r'$x_3$', fontsize=labelsize, labelpad=-30)

    # Add a bit of padding for tick labels as well
    ax.tick_params(axis='x', labelsize=ticksize, pad=6)
    ax.tick_params(axis='y', labelsize=ticksize, pad=6)
    ax.tick_params(axis='z', labelsize=ticksize, pad=6)

    plt.tight_layout()
    ax.view_init(elev=elev, azim=azim)
    if noaxes:
        ax.set_axis_off()

    return fig, ax


def plot_crossections(f: Callable, grids: list):
    """
    Plot cross-sections of a function on a list of grids.

    Parameters
    ----------
    f : callable
        Function to plot.
    grids : list
        List of grids to plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axes object.
    """
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    for grid in grids:
        X = grid[2][0]
        Y = grid[2][1]
        Z = grid[2][2]
        vals = jax.vmap(f)(grid[0]).reshape(X.shape)

        ax.plot_surface(X, Y, Z, facecolors=plt.cm.plasma(
            (vals - vals.min())/(vals.max()-vals.min())), rstride=1, cstride=1, shade=False)
    set_axes_equal(ax)
    plt.tight_layout()
    return fig, ax


def intersect_with_plane(traj, plane_normal=jnp.array([1., 0., 0.]),
                         plane_offset=0.0, deg=2):
    """
    JAX/vmap/jit-safe: intersections of a 3D trajectory with an arbitrary plane.

    Plane defined by:    n · x = plane_offset
    where n is the normal vector.

    Parameters
    ----------
    traj : (N,3)
        Trajectory points in 3D.
    plane_normal : (3,)
        Plane normal vector (does not need to be normalized).
    plane_offset : float
        Offset in the plane equation n·x = plane_offset.
    deg : int
        Polynomial interpolation degree (1=linear, 2=quadratic, 3=cubic,...)
    """
    N = traj.shape[0]
    pad_size = N
    half = deg // 2

    # signed distance of each point to the plane
    n = plane_normal / jnp.linalg.norm(plane_normal)
    s = traj @ n - plane_offset  # signed distance to plane

    # find sign changes (crossings)
    flip_mask = s[:-1] * s[1:] < 0
    idxs = jnp.where(flip_mask, jnp.arange(N - 1), N)
    idxs = jnp.sort(idxs)
    idxs = jnp.pad(idxs, (0, jnp.maximum(0, pad_size - idxs.size)),
                   constant_values=N)[:pad_size]

    def interp(i):
        valid = (i >= half) & (i < N - half)
        offset = jnp.arange(-half, deg - half + 1)
        idxs_local = jnp.clip(i + offset, 0, N - 1)
        pts_seg = traj[idxs_local]
        s_seg = s[idxs_local]
        t = jnp.arange(deg + 1, dtype=float)

        # fit polynomial s(t) ~ a t^deg + b t^{deg-1} + ... + c
        coeffs_s = jnp.polyfit(t, s_seg, deg=deg)
        roots = jnp.roots(coeffs_s, strip_zeros=False)
        roots_real = jnp.real(roots)
        cond = (jnp.abs(jnp.imag(roots)) <
                1e-8) & (roots_real > 0.0) & (roots_real < deg)
        t_cross = jnp.nanmin(jnp.where(cond, roots_real, jnp.nan))

        # fit each coordinate & evaluate at t_cross
        def eval_coord(y_seg):
            coeffs = jnp.polyfit(t, y_seg, deg=deg)
            return jnp.polyval(coeffs, t_cross)

        pt = jax.vmap(eval_coord)(pts_seg.T)
        return jnp.where(valid, pt, jnp.nan)

    pts = jax.vmap(interp)(idxs)
    return pts, idxs


def trajectory_plane_intersections_list(
        trajectory: jnp.ndarray, plane_point: jnp.ndarray, plane_normal: jnp.ndarray):
    """
    Compute intersections of a 3D trajectory with a general plane.

    Returns a list of intersection points (no NaNs, no masks).

    Parameters
    ----------
    trajectory : ndarray (T, 3)
    plane_point : ndarray (3,)
    plane_normal : ndarray (3,)

    Returns
    -------
    intersections : list of ndarray, each of shape (3,)
    """
    trajectory = np.asarray(trajectory)
    plane_point = np.asarray(plane_point)
    plane_normal = np.asarray(plane_normal)

    intersections = []

    for i in range(len(trajectory)-1):
        seg_start = trajectory[i]
        seg_end = trajectory[i+1]
        seg_vec = seg_end - seg_start

        denom = np.dot(plane_normal, seg_vec)
        if np.abs(denom) < 1e-12:  # parallel segment
            continue

        t = np.dot(plane_normal, plane_point - seg_start) / denom

        if 0 <= t <= 1:
            intersections.append(seg_start + t * seg_vec)

    return intersections
# %%


def poincare_plot(
        outdir: str, vector_field: callable, F: callable, x0: jnp.ndarray,
        n_loop: int, n_batch: int, colors: list, plane_val: float,
        axis: int, final_time: int = 10000, n_saves: int = 20000,
        max_steps: int = 150000, r_tol: float = 1e-7, a_tol: float = 1e-7,
        cylindrical: bool = False, filename: str = ""):
    """
    Plot Poincaré sections of a field line.

    Parameters
    ----------
    outdir : str
        Directory to save the plots.
    vector_field : callable
        Vector field to plot.
    F : callable
        Mapping from logical coordinates to physical coords: (r,theta,zeta)->(x,y,z)
    x0 : jnp.ndarray
        Initial conditions for the field lines.
    n_loop : int
        Number of field lines to plot.
    n_batch : int
        Number of batches of field lines to plot.
    colors : list
        Colors to use for the field lines.
    plane_val : float
        Value of the plane to plot.
    axis : int
        Axis to plot the plane on.
    final_time : int
        Final time to integrate the field lines.
    n_saves : int
        Number of saves to make during the integration.
    max_steps : int
        Maximum number of steps to take during the integration.
    r_tol : float
        Relative tolerance for the integration.
    a_tol : float
        Absolute tolerance for the integration.
    cylindrical : bool
        Whether to plot the Poincaré sections in cylindrical coordinates.
    filename : str
        Name of the plot.
    """
    os.makedirs(outdir, exist_ok=True)

    # --- Figure settings ---
    # FIG_SIZE = (12, 6)
    FIG_SIZE_SQUARE = (8, 8)
    # TITLE_SIZE = 20
    LABEL_SIZE = 20
    TICK_SIZE = 16
    # LINE_WIDTH = 2.5
    # LEGEND_SIZE = 16

    assert x0.shape == (n_batch, n_loop, 3)

    term = diffrax.ODETerm(vector_field)
    solver = diffrax.Dopri5()
    saveat = diffrax.SaveAt(ts=jnp.linspace(0, final_time, n_saves))
    stepsize_controller = diffrax.PIDController(rtol=r_tol, atol=a_tol)
    trajectories = []

    # Compute trajectories
    print("Integrating field lines...")
    for x in x0:
        trajectories.append(jax.vmap(lambda x0: diffrax.diffeqsolve(term, solver,
                                                                    t0=0, t1=final_time, dt0=None,
                                                                    y0=x0,
                                                                    max_steps=max_steps,
                                                                    saveat=saveat, stepsize_controller=stepsize_controller).ys)(x))
    trajectories = jnp.array(trajectories).reshape(
        n_batch * n_loop, n_saves, 3) % 1

    physical_trajectories = jax.vmap(F)(trajectories.reshape(-1, 3))
    physical_trajectories = physical_trajectories.reshape(
        trajectories.shape[0], trajectories.shape[1], 3)

    intersections, mask = trajectory_plane_intersections_jit(
        trajectories, plane_val=plane_val, axis=axis)

    if cylindrical:
        def F_cyl(p):
            x, y, z = F(p)
            r = jnp.sqrt(x**2 + y**2)
            phi = jnp.arctan2(y, x)
            return jnp.array([r, phi, z])
        physical_intersections = jax.vmap(F_cyl)(intersections.reshape(-1, 3))
    else:
        physical_intersections = jax.vmap(F)(intersections.reshape(-1, 3))
    physical_intersections = physical_intersections.reshape(
        intersections.shape[0], intersections.shape[1], 3)

    print("Plotting Poincaré sections...")
    # physical domain
    fig1, ax1 = plt.subplots(figsize=FIG_SIZE_SQUARE)
    for i, t in enumerate(physical_intersections):
        # Cycle through the defined colors
        current_color = colors[i % len(colors)]
        if not cylindrical:
            if axis == 0:
                ax1.scatter(t[:, 1], t[:, 2], s=1, color=current_color)
                ax1.set_xlabel(r'$x_2$', fontsize=LABEL_SIZE)
                ax1.set_ylabel(r'$x_3$', fontsize=LABEL_SIZE)
            elif axis == 1:
                ax1.scatter(t[:, 0], t[:, 2], s=1, color=current_color)
                ax1.set_xlabel(r'$x_1$', fontsize=LABEL_SIZE)
                ax1.set_ylabel(r'$x_3$', fontsize=LABEL_SIZE)
            else:
                ax1.scatter(t[:, 0], t[:, 1], s=1, color=current_color)
                ax1.set_xlabel(r'$x_1$', fontsize=LABEL_SIZE)
                ax1.set_ylabel(r'$x_2$', fontsize=LABEL_SIZE)
        else:  # for cylindrical, always plot (r,z) (axis = 2)
            ax1.scatter(t[:, 0], t[:, 2], s=1, color=current_color)
            ax1.set_xlabel(r'$R$', fontsize=LABEL_SIZE)
            ax1.set_ylabel(r'$z$', fontsize=LABEL_SIZE)
    ax1.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    ax1.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(outdir + filename + "poincare_physical.png",
                dpi=600, bbox_inches='tight')

    # logical domain
    fig1, ax1 = plt.subplots(figsize=FIG_SIZE_SQUARE)
    for i, t in enumerate(intersections):
        current_color = colors[i % len(colors)]
        if axis == 0:
            ax1.scatter(t[:, 1], t[:, 2], s=1, color=current_color)
            ax1.set_xlabel(r'$\theta$', fontsize=LABEL_SIZE)
            ax1.set_ylabel(r'$\zeta$', fontsize=LABEL_SIZE)
        elif axis == 1:
            ax1.scatter(t[:, 0], t[:, 2], s=1, color=current_color)
            ax1.set_xlabel(r'$r$', fontsize=LABEL_SIZE)
            ax1.set_ylabel(r'$\zeta$', fontsize=LABEL_SIZE)
        else:
            ax1.scatter(t[:, 0], t[:, 1], s=1, color=current_color)
            ax1.set_xlabel(r'$r$', fontsize=LABEL_SIZE)
            ax1.set_ylabel(r'$\theta$', fontsize=LABEL_SIZE)
    ax1.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    ax1.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(outdir + filename + "poincare_logical.png",
                dpi=600, bbox_inches='tight')

    print("Plotting field lines...")
    # Also plot a few full physical trajectories
    for (i, t) in enumerate(physical_trajectories[::2]):
        fig = plt.figure(figsize=FIG_SIZE_SQUARE)
        ax = fig.add_subplot(projection='3d')
        ax.plot(t[:, 0], t[:, 1], t[:, 2],
                color="purple",
                alpha=1)
        ax.set_axis_off()
        plt.tight_layout()
        plt.savefig(outdir + filename + "field_line_" +
                    str(i) + ".pdf", bbox_inches='tight')


def pressure_plot(p: jnp.ndarray, Seq: DeRhamSequence, F: Callable, outdir: str, filename: str,
                  resolution: int = 128, zeta: float = 0,
                  tol: float = 1e-3,
                  SQUARE_FIG_SIZE: tuple = (8, 8), LABEL_SIZE: int = 20,
                  TICK_SIZE: int = 16, LINE_WIDTH: float = 2.5):
    """
    Plot the pressure on the physical and logical domains side-by-side.

    Parameters
    ----------
    p : jnp.ndarray
        Pressure values.
    Seq : DeRhamSequence
        DeRham sequence to plot the pressure on.
    F : callable
        Mapping from logical coordinates to physical coords: (r,theta,zeta)->(x,y,z)
    outdir : str
        Directory to save the plot.
    filename : str
        Name of the plot.
    resolution : int
        Resolution of the plot.
    zeta : float
        Value of the zeta coordinate to plot.
    tol : float
        Tolerance for the plot.
    SQUARE_FIG_SIZE : tuple
        Size of the figure.
    LABEL_SIZE : int
        Size of the labels.
    TICK_SIZE : int
        Size of the ticks.
    LINE_WIDTH : float
        Width of the line.
    """
    p_h = DiscreteFunction(p, Seq.Lambda_0, Seq.E0)
    p_h_xyz = Pushforward(p_h, F, 0)

    _s = jax.vmap(F)(jnp.vstack(
        [jnp.ones(256), jnp.linspace(0, 1, 256), jnp.zeros(256)]).T)

    fig, ax = plt.subplots(figsize=SQUARE_FIG_SIZE)

    # Plot the line first
    ax.plot(_s[:, 0], _s[:, 2], 'k--',
            linewidth=LINE_WIDTH, label="trajectory")

    _x, _y, (_y1, _y2, _y3), (_x1, _x2, _x3) = get_2d_grids(
        F, cut_value=zeta, nx=resolution, tol1=tol)

    Z = jax.vmap(p_h_xyz)(_x).reshape(_y1.shape)
    cf = ax.contourf(_y1, _y3, Z, levels=20, cmap="plasma", alpha=0.8)

    ax.set_xlim(jnp.min(_s[:, 0]) - 0.2, jnp.max(_s[:, 0]) + 0.2)
    ax.set_ylim(jnp.min(_s[:, 2]) - 0.2, jnp.max(_s[:, 2]) + 0.2)
    ax.set_aspect('equal')
    ax.set_xlabel("R", fontsize=LABEL_SIZE)
    ax.set_ylabel("z", fontsize=LABEL_SIZE)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Colorbar
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label(r"p", fontsize=LABEL_SIZE)
    cbar.ax.tick_params(labelsize=TICK_SIZE)

    # Save
    plt.tight_layout()
    # Ensure outdir ends with a path separator
    if not outdir.endswith(os.sep) and not outdir.endswith('/'):
        outdir = outdir + os.sep
    plt.savefig(outdir + filename, bbox_inches='tight')
    plt.close()


def plot_scalar_fct_physical_logical(p_h: Callable, Phi: Callable,
                                     n_vis: int = 64,
                                     scale: float = 100.0,
                                     cmap: str = 'viridis',
                                     levels: int = 25,
                                     figsize: tuple = (8, 4),
                                     grid_stride: int = 4,
                                     grid_alpha: float = 0.2,
                                     grid_lw: float = 1.0,
                                     cbar_label: str = None,
                                     cbar_shrink: float = 0.9,
                                     constrained_layout: bool = True,
                                     show: bool = False,
                                     logical_plane: str = 'r_theta',
                                     fixed_theta: float = 0.0,
                                     fixed_r: float = 0.5,
                                     fixed_zeta: float = 0,
                                     colorbar: bool = True):
    """Plot a scalar function on the physical and logical domains side-by-side.

    Parameters
    ----------
    p_h : Callable
        DiscreteFunction-like object mapping logical coordinates -> scalar
        pressure value. Should accept (N,3) or (3,) points via vmap.
    Phi : Callable
        Mapping from logical coordinates to physical coords: (r,theta,zeta)->(x,y,z)
    n_vis : int
        Number of points per logical axis (resolution of the visualization).
    scale : float
        Multiplicative scale applied to `p_h` values for display.
    cmap : str
        Colormap to use.
    levels : int
        Number of levels for the contour plot.
    figsize : tuple
        Size of the figure.
    grid_stride : int
        Stride for the grid lines.
    grid_alpha : float
        Alpha for the grid lines.
    grid_lw : float
        Width for the grid lines.
    cbar_label : str
        Label for the colorbar.
    cbar_shrink : float
        Shrink factor for the colorbar.
    constrained_layout : bool
        Whether to use constrained layout.
    show : bool
        Whether to show the plot.
    logical_plane : str
        Logical plane to plot.
    fixed_theta : float
        Value of the theta coordinate to fix.
    fixed_r : float
        Value of the r coordinate to fix.
    fixed_zeta : float
        Value of the zeta coordinate to fix.
    colorbar : bool
        Whether to show the colorbar.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax_phys : matplotlib.axes.Axes
        Axes object for the physical domain.
    ax_log : matplotlib.axes.Axes
        Axes object for the logical domain.
    """

    # Build logical grid and evaluation points depending on requested plane
    logical_plane = logical_plane.lower()
    if logical_plane in ('r_theta', 'rθ', 'r_θ', 'rtheta', 'rt', 'r_t'):
        # default behavior: vary r and theta, fix zeta at `fixed_zeta`
        _r = jnp.linspace(0, 1, n_vis)
        _θ = jnp.linspace(0, 1, n_vis)
        _ζ = jnp.array([fixed_zeta])
        _rgrid, _θgrid, _ζgrid = jnp.meshgrid(_r, _θ, _ζ, indexing='ij')
        _x_hat = jnp.stack([_rgrid, _θgrid, _ζgrid], axis=-1).reshape(-1, 3)
        logical_x = np.asarray(_rgrid).squeeze()
        logical_y = np.asarray(_θgrid).squeeze()
        xlabel, ylabel = 'r', 'θ'

    elif logical_plane in ('r_zeta', 'r_ζ', 'r_z', 'rzeta', 'rz', 'rζ'):
        # vary r and zeta, fix theta at `fixed_theta`
        _r = jnp.linspace(0, 1, n_vis)
        _ζ = jnp.linspace(0, 1, n_vis)
        _rgrid, _ζgrid = jnp.meshgrid(_r, _ζ, indexing='ij')
        _θgrid = jnp.ones_like(_rgrid) * float(fixed_theta)
        _x_hat = jnp.stack([_rgrid, _θgrid, _ζgrid], axis=-1).reshape(-1, 3)
        logical_x = np.asarray(_rgrid).squeeze()
        logical_y = np.asarray(_ζgrid).squeeze()
        xlabel, ylabel = 'r', 'ζ'

    elif logical_plane in ('theta_zeta', 'θ_ζ', 'theta_z', 'thetazeta', 'tz', 't_z', 'θζ'):
        # vary theta and zeta, fix r at `fixed_r`
        _θ = jnp.linspace(0, 1, n_vis)
        _ζ = jnp.linspace(0, 1, n_vis)
        _θgrid, _ζgrid = jnp.meshgrid(_θ, _ζ, indexing='ij')
        _rgrid = jnp.ones_like(_θgrid) * float(fixed_r)
        _x_hat = jnp.stack([_rgrid, _θgrid, _ζgrid], axis=-1).reshape(-1, 3)
        logical_x = np.asarray(_θgrid).squeeze()
        logical_y = np.asarray(_ζgrid).squeeze()
        xlabel, ylabel = 'θ', 'ζ'

    else:
        raise ValueError(f"Unknown logical_plane '{logical_plane}'."
                         " Use 'rt', 'rz' or 'tz'.")

    # map to physical points and evaluate pressure
    _x = jax.vmap(Phi)(_x_hat)
    R = ((_x[:, 0]**2 + _x[:, 1]**2)**0.5).reshape(n_vis, n_vis)
    Z = _x[:, 2].reshape(n_vis, n_vis)

    p_vals = jax.vmap(p_h)(_x_hat).reshape(n_vis, n_vis) * scale

    # create figure and axes
    if logical_plane in ('r_theta', 'rθ', 'r_x_theta'):
        # keep previous behavior: show both physical (R,z) and logical panels
        fig, (ax_phys, ax_log) = plt.subplots(
            1, 2, figsize=figsize, constrained_layout=constrained_layout)

        # physical domain contour
        ax_phys.contourf(np.asarray(R), np.asarray(Z), np.asarray(p_vals),
                         levels=levels, cmap=cmap)
        if grid_stride > 0:
            for i in range(0, n_vis, grid_stride):
                ax_phys.plot(np.asarray(R[i, :]), np.asarray(
                    Z[i, :]), 'w', lw=grid_lw, alpha=grid_alpha)
            for j in range(0, n_vis, grid_stride):
                ax_phys.plot(np.asarray(R[:, j]), np.asarray(
                    Z[:, j]), 'w', lw=grid_lw, alpha=grid_alpha)
        ax_phys.set(xlabel='R', ylabel='z', aspect='equal')

        # logical domain contour for the selected logical plane
        c2 = ax_log.contourf(logical_x, logical_y, np.asarray(
            p_vals), levels=levels, cmap=cmap)
        if grid_stride > 0:
            for i in range(0, n_vis, grid_stride):
                ax_log.plot(logical_x[i, :], logical_y[i, :], 'w',
                            lw=grid_lw, alpha=grid_alpha)
            for j in range(0, n_vis, grid_stride):
                ax_log.plot(logical_x[:, j], logical_y[:, j], 'w',
                            lw=grid_lw, alpha=grid_alpha)
        ax_log.set(xlabel=xlabel, ylabel=ylabel, aspect='equal')

        # shared colorbar on the right of the logical axis
        if colorbar:
            fig.colorbar(c2, ax=ax_log, label=cbar_label, shrink=cbar_shrink)

        if show:
            plt.show()
        return fig, (ax_phys, ax_log)

    else:
        # Only plot the logical plane (no physical left panel)
        fig, ax_log = plt.subplots(
            1, 1, figsize=figsize, constrained_layout=constrained_layout)

        c2 = ax_log.contourf(logical_x, logical_y, np.asarray(
            p_vals), levels=levels, cmap=cmap)
        if grid_stride > 0:
            for i in range(0, n_vis, grid_stride):
                ax_log.plot(logical_x[i, :], logical_y[i, :], 'w',
                            lw=grid_lw, alpha=grid_alpha)
            for j in range(0, n_vis, grid_stride):
                ax_log.plot(logical_x[:, j], logical_y[:, j], 'w',
                            lw=grid_lw, alpha=grid_alpha)
        ax_log.set(xlabel=xlabel, ylabel=ylabel, aspect='equal')

        # colorbar
        if colorbar:
            fig.colorbar(c2, ax=ax_log, label=cbar_label, shrink=cbar_shrink)

        if show:
            plt.show()

        return fig, (None, ax_log)
# %%


def trace_plot(trace_dict: dict, filename: str,
               FIG_SIZE: tuple = (12, 6), LABEL_SIZE: int = 20, TICK_SIZE: int = 16,
               LINE_WIDTH: float = 2.5, LEGEND_SIZE: int = 16):
    """
    Plot the trace of the energy, force, helicity, divergence, and 
    velocity. 

    Parameters
    ----------
    trace_dict : dict
        Dictionary containing the trace of the energy, force, helicity, divergence, and velocity.
    filename : str
        Name of the file to save the plot.
    FIG_SIZE : tuple
        Size of the figure.
    LABEL_SIZE : int
        Size of the labels.
    TICK_SIZE : int
        Size of the ticks.
    LINE_WIDTH : float
        Width of the lines.
    LEGEND_SIZE : int
        Size of the legend.

    Returns
    ------
    None.
    """

    color1 = 'purple'
    color2 = 'black'
    color3 = 'darkgray'
    # color4 = 'teal'
    color5 = 'orange'

    # # Plot Energy on separate plot
    energy_trace = trace_dict.get("energy_trace")
    if energy_trace is not None:
        energy_trace = jnp.array(energy_trace)
    iterations = trace_dict["iterations"]
    force_trace = jnp.array(trace_dict["force_trace"])
    helicity_trace = jnp.array(trace_dict["helicity_trace"])
    divergence_trace = jnp.array(trace_dict["divergence_trace"])
    velocity_trace = jnp.array(trace_dict["velocity_trace"])
    wall_time_trace = jnp.array(trace_dict["wall_time_trace"])

    # Only plot energy trace if it's not None
    if energy_trace is not None:
        fig1, ax1 = plt.subplots(figsize=FIG_SIZE)
        # ax1 = ax2.twinx()
        ax1.set_xlabel(r'$n$', fontsize=LABEL_SIZE)
        ax1.set_ylabel(r'$\frac{1}{2} \| B \|^2$',
                       color=color1, fontsize=LABEL_SIZE)
        ax1.semilogy(energy_trace[0] - energy_trace,
                     label=r'$\frac{1}{2} \| B \|^2$', color=color1, linestyle='-.', lw=LINE_WIDTH)
        # ax1.plot(jnp.pi * jnp.array(H_trace), label=r'$\pi \, (A, B)$', color=color1, linestyle="--", lw=LINE_WIDTH)
        ax1.tick_params(axis='y', labelcolor=color1, labelsize=TICK_SIZE)
        ax1.tick_params(axis='x', labelsize=TICK_SIZE)  # Set x-tick size
        ax1.set_xscale('log')
        ax1.tick_params(axis='y', labelcolor=color1, labelsize=TICK_SIZE)
        fig1.tight_layout()
        plt.savefig(filename + "energy_trace.pdf", bbox_inches='tight')

    fig2, ax2 = plt.subplots(figsize=FIG_SIZE)

    ax2.set_xlabel(r'$n$', fontsize=LABEL_SIZE)
    ax2.tick_params(axis='y', labelsize=TICK_SIZE)
    ax2.tick_params(axis='x', labelsize=TICK_SIZE)
    ax2.set_yscale('log')
    # make a twin y axis for wall time (top) and iteration(bottom)
    ax2_top = ax2.twiny()
    ax2_top.tick_params(axis='x', labelsize=TICK_SIZE)
    ax2_top.set_xlabel('wall time [s]', fontsize=LABEL_SIZE)

    helicity_change = jnp.abs(jnp.array(helicity_trace - helicity_trace[0]))

    ax2.plot(iterations, force_trace, label=r'$\| \, J \times B - \mathrm{grad} \, p \| / \| \mathrm{grad} p \|$',
             color=color1, lw=LINE_WIDTH, linestyle='-')

    ax2.plot(iterations, velocity_trace, label=r'$\| v \|^2$',
             color=color3, lw=LINE_WIDTH, linestyle='-.')

    ax2.plot(iterations, helicity_change, label=r'$| H - H^0 | $',
             color=color2, linestyle='--', lw=LINE_WIDTH)

    ax2_top.plot(wall_time_trace, divergence_trace - divergence_trace[0], label=r'$ \| \mathrm{div} \, B \|$',
                 color=color5, linestyle='-.', lw=LINE_WIDTH)

    # Set y-limits for better visibility
    # ax2.set_ylim(0.5 * min(min(force_trace), 0.1 * max(helicity_change)),
    #              2 * max(max(force_trace), max(helicity_change)))
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_top.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2,
               loc='best', fontsize=LEGEND_SIZE)
    # ax1.grid(which="major", linestyle="-", color=color1, linewidth=0.5)
    ax2.legend(loc='best', fontsize=LEGEND_SIZE)
    ax2.grid(which="both", linestyle="--", linewidth=0.5)
    fig2.tight_layout()
    if filename.endswith("/"):
        filename = filename[:-1]
    plt.savefig(filename + "/force_trace.pdf", bbox_inches='tight')
# %%


def generate_solovev_plots(filename: str):
    """
    Generate plots for a Solovev equilibrium.

    Parameters
    ----------
    filename : str
        Name of the file to save the plots. If the file is not in the script_outputs/solovev folder,
        the file will be saved in the script_outputs/solovev folder. 

    Returns
    -------
    None.
    """
    from mrx.utils import is_running_in_github_actions
    jax.config.update("jax_enable_x64", True)

    outdir = "script_outputs/solovev/" + filename + "/"
    os.makedirs(outdir, exist_ok=True)

    print("Generating plots for " + filename + "...")

    with h5py.File(outdir + filename + ".h5", "r") as f:
        CONFIG = {k: v for k, v in f["config"].attrs.items()}
        # decode strings back if needed
        CONFIG = {k: v.decode() if isinstance(v, bytes)
                  else v for k, v in CONFIG.items()}

        trace_dict = {}
        for k, v in f.items():
            if k != "config":
                # Convert HDF5 datasets to numpy arrays
                if isinstance(v, h5py.Dataset):
                    trace_dict[k] = v[:]
                else:
                    trace_dict[k] = v
        if CONFIG["save_B"]:
            # B_fields = f["B_fields"][:]
            if "p_fields" in f:
                trace_dict["p_fields"] = f["p_fields"][:]
            if "B_fields" in f:
                trace_dict["B_fields"] = f["B_fields"][:]

    # Step 1: get F
    delta = CONFIG["delta"]
    kappa = CONFIG["kappa"]
    eps = CONFIG["eps"]
    R0 = CONFIG["R_0"]
    alpha = jnp.arcsin(delta)
    F = cerfon_map(eps, kappa, alpha, R0)

    # Step 2: Get the Sequence
    ns = (CONFIG["n_r"], CONFIG["n_theta"], CONFIG["n_zeta"])
    ps = (CONFIG["p_r"], CONFIG["p_theta"], 0
          if CONFIG["n_zeta"] == 1 else CONFIG["p_zeta"])
    q = max(ps)
    types = ("clamped", "periodic",
             "constant" if CONFIG["n_zeta"] == 1 else "periodic")

    Seq = DeRhamSequence(ns, ps, q, types, F, polar=True)

    print("Generating pressure plot...")
    # Plot number one: pressure contour plot of final solution
    pressure_plot(trace_dict["p_final"], Seq, F,
                  outdir, filename="p_final.pdf", zeta=0)
    if CONFIG["save_B"] and "p_fields" in trace_dict:
        for i, p in enumerate(trace_dict["p_fields"]):
            pressure_plot(
                p, Seq, F, outdir, filename=f"p_iter_{i*CONFIG['save_every']:06d}.pdf", zeta=0)

    print("Generating convergence plot...")
    # Figure 2: Energy and Force
    # outdir already ends with '/' and contains filename, so just pass outdir
    trace_plot(trace_dict=trace_dict, filename=outdir)

    # Only plot B_final if it exists
    if "B_final" in trace_dict and not is_running_in_github_actions():
        B_hat = trace_dict["B_final"]
        B_h = DiscreteFunction(B_hat, Seq.Lambda_2, Seq.E2)

        @jax.jit
        def vector_field(t, p, args):
            r, χ, z = p
            r = jnp.clip(r, 1e-6, 1)
            χ = χ % 1.0
            z = z % 1.0
            x = jnp.array([r, χ, z])
            DFx = jax.jacfwd(F)(x)
            norm = ((DFx @ B_h(x)) @ DFx @ B_h(x))**0.5
            return B_h(x) / (norm + 1e-9)

        n_loop = 5
        n_batch = 5

        x0s = jnp.vstack(
            (jnp.linspace(0.05, 0.95, n_loop * n_batch),
             jnp.zeros(n_loop * n_batch),
             jnp.zeros(n_loop * n_batch))
        ).T

        n_cols = x0s.shape[1]
        cm = plt.cm.plasma
        vals = jnp.linspace(0, 1, n_cols + 2)[:-2]

        # Interleave from start and end
        order = jnp.ravel(jnp.column_stack(
            [jnp.arange(n_cols//2), n_cols-1-jnp.arange(n_cols//2)]))
        if n_cols % 2 == 1:
            order = jnp.append(order, n_cols//2)

        colors = cm(vals[order])

        x0s = x0s.T.reshape(n_batch, n_loop, 3)

        if is_running_in_github_actions():
            poincare_plot(outdir, vector_field, F, x0s, n_loop, n_batch, colors,
                          plane_val=0.25, axis=2, final_time=5000, n_saves=20000, cylindrical=True,
                          r_tol=CONFIG["solver_tol"], a_tol=CONFIG["solver_tol"], filename="")

# %%


def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale."""
    X_limits = ax.get_xlim3d()
    Y_limits = ax.get_ylim3d()
    Z_limits = ax.get_zlim3d()

    X_range = X_limits[1] - X_limits[0]
    Y_range = Y_limits[1] - Y_limits[0]
    Z_range = Z_limits[1] - Z_limits[0]
    max_range = max(X_range, Y_range, Z_range)

    X_mid = np.mean(X_limits)
    Y_mid = np.mean(Y_limits)
    Z_mid = np.mean(Z_limits)

    ax.set_xlim3d([X_mid - max_range/2, X_mid + max_range/2])
    ax.set_ylim3d([Y_mid - max_range/2, Y_mid + max_range/2])
    ax.set_zlim3d([Z_mid - max_range/2, Z_mid + max_range/2])


def plot_twin_axis(
    left_y: jnp.ndarray,
    right_y: jnp.ndarray,
    x_left: Optional[jnp.ndarray] = None,
    x_right: Optional[jnp.ndarray] = None,
    left_label: str = "",
    right_label: str = "",
    left_log: bool = True,
    right_log: bool = False,
    left_color: str = "black",
    right_color: str = "teal",
    left_marker: str = 's',
    right_marker: str = 'd',
    left_linestyle: str = '-',
    right_linestyle: str = '--',
    left_markersize: int = 4,
    right_markersize: int = 4,
    num_iters_inner: int = 1,
    x_label: str = 'iteration',
    figsize: tuple = (8, 3),
    grid: bool = True,
    grid_linestyle: str = '--',
    grid_linewidth: float = 0.5,
    left_plot_kwargs: Optional[dict] = None,
    right_plot_kwargs: Optional[dict] = None,
    show: bool = False,
    return_axes: bool = True,
):
    """Plot two series on shared x-axis with separate y-axes (twinx).

    All common plotting options are explicit arguments with sensible
    defaults. Additionally, `left_plot_kwargs` and `right_plot_kwargs`
    may contain any valid matplotlib plotting kwargs which will be
    forwarded to the underlying plotting call and will override the
    corresponding explicit arguments when present.

    Parameters
    ----------
    left_y : jnp.ndarray
        Left y-axis data.
    right_y : jnp.ndarray
        Right y-axis data.
    x_left : Optional[jnp.ndarray]
        Left x-axis data.
    x_right : Optional[jnp.ndarray]
        Right x-axis data.
    left_label : str
        Left y-axis label.
    right_label : str
        Right y-axis label.
    left_log : bool
        Whether to plot the left y-axis on a log scale.
    right_log : bool
        Whether to plot the right y-axis on a log scale.
    left_color : str
        Left color.
    right_color : str
        Right color.
    left_marker : str
        Left marker.
    right_marker : str
        Right marker.
    left_linestyle : str
        Left line style.
    right_linestyle : str
        Right line style.
    left_markersize : int
        Left marker size.
    right_markersize : int
        Right marker size.
    num_iters_inner : int
        Number of iterations per inner data point.
    x_label : str
        X-axis label.
    figsize : tuple
        Figure size.
    grid : bool
        Whether to plot a grid.
    grid_linestyle : str
        Grid line style.
    grid_linewidth : float
        Grid line width.
    left_plot_kwargs : Optional[dict]
        Left plot kwargs.
    right_plot_kwargs : Optional[dict]
        Right plot kwargs.
    show : bool
        Whether to show the plot.
    return_axes : bool
        Whether to return the axes.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax1 : matplotlib.axes.Axes
        Axes object for the left y-axis.
    ax2 : matplotlib.axes.Axes
        Axes object for the right y-axis.
    """

    ly = np.asarray(left_y)
    ry = np.asarray(right_y)

    if x_left is None:
        x_left = np.arange(len(ly)) * int(num_iters_inner)
    else:
        x_left = np.asarray(x_left)

    if x_right is None:
        x_right = np.arange(len(ry)) * int(num_iters_inner)
    else:
        x_right = np.asarray(x_right)

    left_plot_kwargs = {} if left_plot_kwargs is None else dict(
        left_plot_kwargs)
    right_plot_kwargs = {} if right_plot_kwargs is None else dict(
        right_plot_kwargs)

    # Merge explicit style args with kwargs dicts; kwargs win
    base_left = {
        'color': left_color,
        'linestyle': left_linestyle,
        'marker': left_marker,
        'markersize': left_markersize,
    }
    plot_kwargs_left = {**base_left, **left_plot_kwargs}

    base_right = {
        'color': right_color,
        'linestyle': right_linestyle,
        'marker': right_marker,
        'markersize': right_markersize,
    }
    plot_kwargs_right = {**base_right, **right_plot_kwargs}

    fig, ax1 = plt.subplots(figsize=figsize)

    if left_log:
        ax1.semilogy(x_left, ly, **plot_kwargs_left)
    else:
        ax1.plot(x_left, ly, **plot_kwargs_left)

    ax1.set_xlabel(x_label)
    ax1.set_ylabel(left_label, color=plot_kwargs_left.get('color', left_color))
    ax1.tick_params(
        axis='y', labelcolor=plot_kwargs_left.get('color', left_color))

    ax2 = ax1.twinx()
    if right_log:
        ax2.semilogy(x_right, ry, **plot_kwargs_right)
    else:
        ax2.plot(x_right, ry, **plot_kwargs_right)

    ax2.set_ylabel(right_label, color=plot_kwargs_right.get(
        'color', right_color))
    ax2.tick_params(
        axis='y', labelcolor=plot_kwargs_right.get('color', right_color))

    if grid:
        ax1.grid(True, which='both', linestyle=grid_linestyle,
                 linewidth=grid_linewidth)

    fig.tight_layout()
    if show:
        plt.show()
    if return_axes:
        return fig, (ax1, ax2)
    return fig
