"""
Plotting utilities for finite element analysis results.

This module provides functions for creating visualizations of convergence plots
and other analysis results using Plotly.
"""

# %%
from typing import Callable, Optional

import diffrax as dfx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

import mrx
from mrx.differential_forms import DiscreteFunction

# Base marker styles for different data series
base_markers = [
    "circle",
    "triangle-down",
    "star",
    "triangle-left",
    "triangle-right",
    "triangle-ne",
    "triangle-se",
    "triangle-sw",
    "triangle-nw",
    "square",
    "pentagon",
    "triangle-up",
    "hexagon",
    "hexagon2",
    "cross",
    "x",
    "diamond",
    "diamond-open",
    "line-ns",
    "line-ew",
]

# Default color scale for plots
colorbar = "Viridis"

#########
# Grids #
#########


def get_3d_grids(
    F: Callable,
    x_min: float = 0,
    x_max: float = 1,
    y_min: float = 0,
    y_max: float = 1,
    z_min: float = 0,
    z_max: float = 1,
    nx: int = 16,
    ny: int = 16,
    nz: int = 16,
):
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
    _x = _x.transpose(1, 2, 3, 0).reshape(nx * ny * nz, 3)
    _y = jax.lax.map(F, _x, batch_size=mrx.MAP_BATCH_SIZE_INNER)
    _y1 = _y[:, 0].reshape(nx, ny, nz)
    _y2 = _y[:, 1].reshape(nx, ny, nz)
    _y3 = _y[:, 2].reshape(nx, ny, nz)
    return _x, _y, (_y1, _y2, _y3), (_x1, _x2, _x3)


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
    _y = jax.lax.map(F, _x, batch_size=mrx.MAP_BATCH_SIZE_INNER)
    _y1 = _y[:, 0].reshape(n1, n2)
    _y2 = _y[:, 1].reshape(n1, n2)
    _y3 = _y[:, 2].reshape(n1, n2)
    _y = jax.lax.map(F, _x, batch_size=mrx.MAP_BATCH_SIZE_INNER)
    return _x, _y, (_y1, _y2, _y3), (_x1, _x2, _x3)


def get_1d_grids(
    F: Callable, zeta: float = 0, chi: float = 0, nx: int = 64, tol: float = 1e-6
):
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
    _y = jax.lax.map(F, _x, batch_size=mrx.MAP_BATCH_SIZE_INNER)
    _y1 = _y[:, 0]
    _y2 = _y[:, 1]
    _y3 = _y[:, 2]
    return _x, _y, (_y1, _y2, _y3), (_x1, _x2, _x3)

############
# 3D plots #
############


def plot_torus(
    p_h: Callable,
    grids_pol: list,
    grid_surface: list,
    figsize: tuple = (12, 8),
    labelsize: int = 20,
    ticksize: int = 16,
    gridlinewidth: float = 0.01,
    cstride: int = 4,
    elev: float = 30,
    azim: float = 140,
    noaxes: bool = False,
):
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
    vals = jnp.array(
        [jax.lax.map(p_h, grid[0], batch_size=mrx.MAP_BATCH_SIZE_INNER).reshape(grid[2][0].shape)
         for grid in grids_pol]
    )

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    X = grid_surface[2][0]
    Y = grid_surface[2][1]
    Z = grid_surface[2][2]
    colors = plt.cm.plasma(jnp.zeros_like(X))
    ax.plot_surface(
        X,
        Y,
        Z,
        edgecolors=(0, 0, 0, 0.2),
        rstride=cstride,
        cstride=cstride,
        shade=True,
        alpha=0.0,
        linewidth=gridlinewidth,
    )

    vals_np = np.asarray(vals)
    vals_min = float(vals_np.min())
    vals_max = float(vals_np.max())
    if vals_max == vals_min:
        vals_max = vals_min + 1e-12

    for i, grid in enumerate(grids_pol):
        X = grid[2][0]
        Y = grid[2][1]
        Z = grid[2][2]
        v = np.asarray(vals_np[i])
        colors = plt.cm.plasma((v - vals_min) / (vals_max - vals_min))
        ax.plot_surface(
            X,
            Y,
            Z,
            facecolors=colors,
            rstride=1,
            cstride=1,
            shade=False,
            zsort="min",
            linewidth=0,
        )

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
    ax.set_xlabel(r"$x_1$", fontsize=labelsize, labelpad=14)
    ax.set_ylabel(r"$x_2$", fontsize=labelsize, labelpad=14)
    # increase z label padding so it isn't clipped by the figure edge
    ax.set_zlabel(r"$x_3$", fontsize=labelsize, labelpad=-30)

    # Add a bit of padding for tick labels as well
    ax.tick_params(axis="x", labelsize=ticksize, pad=6)
    ax.tick_params(axis="y", labelsize=ticksize, pad=6)
    ax.tick_params(axis="z", labelsize=ticksize, pad=6)

    plt.tight_layout()
    ax.view_init(elev=elev, azim=azim)
    if noaxes:
        ax.set_axis_off()

    return fig, ax


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

    ax.set_xlim3d([X_mid - max_range / 2, X_mid + max_range / 2])
    ax.set_ylim3d([Y_mid - max_range / 2, Y_mid + max_range / 2])
    ax.set_zlim3d([Z_mid - max_range / 2, Z_mid + max_range / 2])

###############
# Trace plots #
###############


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
        "color": left_color,
        "linestyle": left_linestyle,
        "marker": left_marker,
        "markersize": left_markersize,
    }
    plot_kwargs_left = {**base_left, **left_plot_kwargs}

    base_right = {
        "color": right_color,
        "linestyle": right_linestyle,
        "marker": right_marker,
        "markersize": right_markersize,
    }
    plot_kwargs_right = {**base_right, **right_plot_kwargs}

    fig, ax1 = plt.subplots(figsize=figsize)

    if left_log:
        ax1.semilogy(x_left, ly, **plot_kwargs_left)
    else:
        ax1.plot(x_left, ly, **plot_kwargs_left)

    ax1.set_xlabel(x_label)
    ax1.set_ylabel(left_label, color=plot_kwargs_left.get("color", left_color))
    ax1.tick_params(
        axis="y", labelcolor=plot_kwargs_left.get("color", left_color))

    ax2 = ax1.twinx()
    if right_log:
        ax2.semilogy(x_right, ry, **plot_kwargs_right)
    else:
        ax2.plot(x_right, ry, **plot_kwargs_right)

    ax2.set_ylabel(right_label, color=plot_kwargs_right.get(
        "color", right_color))
    ax2.tick_params(
        axis="y", labelcolor=plot_kwargs_right.get("color", right_color))

    if grid:
        ax1.grid(True, which="both", linestyle=grid_linestyle,
                 linewidth=grid_linewidth)

    fig.tight_layout()
    if show:
        plt.show()
    if return_axes:
        return fig, (ax1, ax2)
    return fig


def trace_plot(
    trace_dict: dict,
    filename: str,
    FIG_SIZE: tuple = (12, 6),
    LABEL_SIZE: int = 20,
    TICK_SIZE: int = 16,
    LINE_WIDTH: float = 2.5,
    LEGEND_SIZE: int = 16,
):
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

    color1 = "purple"
    color2 = "black"
    color3 = "darkgray"
    # color4 = 'teal'
    color5 = "orange"

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
        ax1.set_xlabel(r"$n$", fontsize=LABEL_SIZE)
        ax1.set_ylabel(r"$\frac{1}{2} \| B \|^2$",
                       color=color1, fontsize=LABEL_SIZE)
        ax1.semilogy(
            energy_trace[0] - energy_trace,
            label=r"$\frac{1}{2} \| B \|^2$",
            color=color1,
            linestyle="-.",
            lw=LINE_WIDTH,
        )
        # ax1.plot(jnp.pi * jnp.array(H_trace), label=r'$\pi \, (A, B)$', color=color1, linestyle="--", lw=LINE_WIDTH)
        ax1.tick_params(axis="y", labelcolor=color1, labelsize=TICK_SIZE)
        ax1.tick_params(axis="x", labelsize=TICK_SIZE)  # Set x-tick size
        ax1.set_xscale("log")
        ax1.tick_params(axis="y", labelcolor=color1, labelsize=TICK_SIZE)
        fig1.tight_layout()
        plt.savefig(filename + "energy_trace.pdf", bbox_inches="tight")

    fig2, ax2 = plt.subplots(figsize=FIG_SIZE)

    ax2.set_xlabel(r"$n$", fontsize=LABEL_SIZE)
    ax2.tick_params(axis="y", labelsize=TICK_SIZE)
    ax2.tick_params(axis="x", labelsize=TICK_SIZE)
    ax2.set_yscale("log")
    # make a twin y axis for wall time (top) and iteration(bottom)
    ax2_top = ax2.twiny()
    ax2_top.tick_params(axis="x", labelsize=TICK_SIZE)
    ax2_top.set_xlabel("wall time [s]", fontsize=LABEL_SIZE)

    helicity_change = jnp.abs(jnp.array(helicity_trace - helicity_trace[0]))

    ax2.plot(
        iterations,
        force_trace,
        label=r"$\| \, J \times B - \mathrm{grad} \, p \| / \| \mathrm{grad} p \|$",
        color=color1,
        lw=LINE_WIDTH,
        linestyle="-",
    )

    ax2.plot(
        iterations,
        velocity_trace,
        label=r"$\| v \|^2$",
        color=color3,
        lw=LINE_WIDTH,
        linestyle="-.",
    )

    ax2.plot(
        iterations,
        helicity_change,
        label=r"$| H - H^0 | $",
        color=color2,
        linestyle="--",
        lw=LINE_WIDTH,
    )

    ax2_top.plot(
        wall_time_trace,
        divergence_trace - divergence_trace[0],
        label=r"$ \| \mathrm{div} \, B \|$",
        color=color5,
        linestyle="-.",
        lw=LINE_WIDTH,
    )

    # Set y-limits for better visibility
    # ax2.set_ylim(0.5 * min(min(force_trace), 0.1 * max(helicity_change)),
    #              2 * max(max(force_trace), max(helicity_change)))
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_top.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2,
               loc="best", fontsize=LEGEND_SIZE)
    # ax1.grid(which="major", linestyle="-", color=color1, linewidth=0.5)
    ax2.legend(loc="best", fontsize=LEGEND_SIZE)
    ax2.grid(which="both", linestyle="--", linewidth=0.5)
    fig2.tight_layout()
    if filename.endswith("/"):
        filename = filename[:-1]
    plt.savefig(filename + "/force_trace.pdf", bbox_inches="tight")

# %%
##################
# Poincare plots #
##################


def integrate_fieldlines(x0s, B_dof, p_dof, seq, T, N):
    B_h = DiscreteFunction(B_dof, seq.Lambda_2, seq.E2)
    p_h = DiscreteFunction(p_dof, seq.Lambda_0, seq.E0)

    def vector_field(t, x, args):
        # avoid evaluation at x[0] = 0
        x = x.at[0].set(jnp.where(x[0] < 1e-9, 1e-9, x[0]))
        # Ensure periodicity in the last two coordinates
        x = x.at[1:3].set(x[1:3] % 1.0)
        Bx = B_h(x)
        DFx = jax.jacfwd(seq.F)(x)
        return Bx / (jnp.linalg.norm(DFx @ Bx) + 1e-9)

    def integrate_fieldline(x0):
        sol = dfx.diffeqsolve(
            terms=dfx.ODETerm(vector_field),
            solver=dfx.Dopri8(),
            t0=0.0,
            t1=T,
            dt0=1.0,
            y0=x0,
            saveat=dfx.SaveAt(ts=jnp.linspace(0.0, T, N)),
            stepsize_controller=dfx.PIDController(rtol=1e-5, atol=1e-5),
            max_steps=100_000,
            throw=False,
        )
        # On solver failure, fill trajectory with NaN
        failed = sol.result != dfx.RESULTS.successful
        traj = jnp.where(failed, jnp.nan, sol.ys % 1.0)
        p_vals = jax.lax.map(p_h,
            jnp.where(failed, jnp.zeros_like(sol.ys), sol.ys) % 1.0, batch_size=mrx.MAP_BATCH_SIZE_INNER)
        p_vals = jnp.where(failed, jnp.nan, p_vals)
        return traj, p_vals

    # Inner function: map over trajectories in a batch
    def mapped_integrate(x0_batch):
        return jax.lax.map(integrate_fieldline, x0_batch,
                           batch_size=mrx.MAP_BATCH_SIZE_INNER)

    # Loop over 'n_scan' batches sequentially to save memory
    def scan_fn(carry, x0_batch):
        trajs, ps = mapped_integrate(x0_batch)
        return carry, (trajs, ps)

    _, (logical_trajectories, p_values) = jax.lax.scan(scan_fn, None, x0s)

    return logical_trajectories, p_values


def get_periodic_intersections(
    field_line,
    p_values,
    plane_normal,
    plane_point,
    max_intersections=100
):
    # Padding and Masking
    def pad_axis0(arr):
        pads = ((1, 1),) + ((0, 0),) * (arr.ndim - 1)
        return jnp.pad(arr, pads, mode='edge')

    def apply_mask(data, mask):
        mask_reshaped = mask.reshape(mask.shape + (1,) * (data.ndim - 1))
        return jnp.where(mask_reshaped, data, jnp.nan)

    # Find shortest vector in periodic [0, 1] space
    diffs = field_line - plane_point
    dists = jnp.dot(diffs, plane_normal)
    pos_curr = field_line[:-1]
    pos_next_raw = field_line[1:]
    delta = pos_next_raw - pos_curr
    delta_unwrapped = delta - jnp.round(delta)
    pos_next_virt_detect = pos_curr + delta_unwrapped
    # Distance at the "virtual" next point
    dist_next_virt = jnp.dot(pos_next_virt_detect - plane_point, plane_normal)

    # Detect Crossings: A crossing exists if the line segment (curr -> next_virt) crosses zero
    d_curr_detect = dists[:-1]
    valid_crossing = (d_curr_detect * dist_next_virt) < 0.0

    # Interpolation
    pos_pad = pad_axis0(field_line)
    p_pad = pad_axis0(p_values)
    crossing_indices = jnp.nonzero(
        valid_crossing,
        size=max_intersections,
        fill_value=0
    )[0]
    count = jnp.sum(valid_crossing)
    # +1 because we padded the start of the array
    centers = crossing_indices + 1

    # Interpolation Kernel (Quadratic)
    def interpolate_single(idx):
        pos_prev_raw = pos_pad[idx - 1]
        pos_curr = pos_pad[idx]
        pos_next_raw = pos_pad[idx + 1]

        # Unwrap Geometry for the local neighborhood
        delta_prev = pos_prev_raw - pos_curr
        pos_prev_v = pos_curr + (delta_prev - jnp.round(delta_prev))
        delta_next = pos_next_raw - pos_curr
        pos_next_v = pos_curr + (delta_next - jnp.round(delta_next))

        # Recalculate distances on unwrapped local points
        d_prev = jnp.dot(pos_prev_v - plane_point, plane_normal)
        d_curr = jnp.dot(pos_curr - plane_point, plane_normal)
        d_next = jnp.dot(pos_next_v - plane_point, plane_normal)

        # Quadratic coefficients: f(t) = at^2 + bt + c, where t=0 is pos_curr
        c = d_curr
        a = 0.5 * (d_prev - 2.0 * d_curr + d_next)
        b = 0.5 * (d_next - d_prev)

        # Solve at^2 + bt + c = 0
        discriminant = b**2 - 4*a*c
        sqrt_disc = jnp.sqrt(jnp.maximum(discriminant, 0.0))

        div = 2.0 * a + 1e-15
        t1 = (-b + sqrt_disc) / div
        t2 = (-b - sqrt_disc) / div
        t_lin = -c / (b + 1e-15)

        # Choose the root that lies within the segment range [0, 1]
        t_quad = jnp.where((t1 >= 0.0) & (t1 <= 1.0), t1, t2)
        t = jnp.where(jnp.abs(a) > 1e-10, t_quad, t_lin)
        t = jnp.clip(t, 0.0, 1.0)

        # Interpolate Position
        A_pos = 0.5 * (pos_prev_v - 2.0 * pos_curr + pos_next_v)
        B_pos = 0.5 * (pos_next_v - pos_prev_v)
        final_pos = A_pos * (t**2) + B_pos * t + pos_curr

        # Re-apply periodicity to the result
        final_pos = final_pos - jnp.floor(final_pos)

        # Interpolate Scalar Field p
        v0, v1, v2 = p_pad[idx-1], p_pad[idx], p_pad[idx+1]
        Ap = 0.5 * (v0 - 2.0 * v1 + v2)
        Bp = 0.5 * (v2 - v0)
        final_p = Ap * (t**2) + Bp * t + v1

        return final_pos, final_p

    # Vectorize and Mask
    intersections, inter_p = jax.vmap(interpolate_single)(centers)

    _idx = jnp.arange(max_intersections)
    mask = _idx < count

    intersections = apply_mask(intersections, mask)
    inter_p = apply_mask(inter_p, mask)

    return intersections, inter_p, count


def get_iota(c, nfp):
    def toroidal_unwrapped(phi):
        phi_unwrapped = jnp.unwrap(phi)
        total_angle = phi_unwrapped[-1] - phi_unwrapped[0]
        return total_angle / (2 * jnp.pi)

    def poloidal_unwrapped(R, Z, R_center=1.0, Z_center=0.0):
        θ = jnp.arctan2(Z - Z_center, R - R_center)
        θ_unwrapped = jnp.unwrap(θ)
        total_angle = θ_unwrapped[-1] - θ_unwrapped[0]
        return total_angle / (2 * jnp.pi)

    x, y, z = c[:, 0], c[:, 1], c[:, 2]
    R = jnp.sqrt(x**2 + y**2)
    phi = jnp.arctan2(y, x)
    r_mean = jnp.mean(R)
    z_mean = jnp.mean(z)
    m = poloidal_unwrapped(R, z, R_center=r_mean, Z_center=z_mean)
    n = toroidal_unwrapped(phi) / nfp
    return jnp.abs(m / n)


def classify_uniformity(t, ks_thresh=0.05):
    def uniformity_score(t_mod):
        # sort values
        ts = jnp.sort(t_mod)
        n = t_mod.size

        # empirical CDF minus ideal uniform CDF
        ecdf = jnp.arange(1, n + 1) / n
        ucdf = ts  # uniform CDF value at ts

        # KS statistic
        ks = jnp.max(jnp.abs(ecdf - ucdf)) * (n**0.5)
        return ks

    t_mod = t % 1.0
    ks = uniformity_score(t_mod)
    well_winding = ks < ks_thresh
    return well_winding, ks


def get_iota_log(c, nfp, ks_thresh=0.05):
    t = jnp.unwrap(c[:, 1], period=1.0)
    z = jnp.unwrap(c[:, 2], period=1.0)
    total_t_angle = t[-1] - t[0]
    total_z_angle = z[-1] - z[0]
    iota = jnp.abs(total_t_angle / total_z_angle * nfp)
    # Uniformity test
    well_winding, ks = classify_uniformity(t, ks_thresh=ks_thresh)
    # set bad values to nan
    iota = jnp.where(well_winding, iota, jnp.nan)
    return iota, well_winding, ks


def poincare_plot(logical_intersections,
                  physical_intersections,
                  p_values,
                  iota_values,
                  nfp,
                  cmap_iota="berlin",
                  cmap_p="plasma",
                  markersize=0.01,
                  denom_max=15,
                  Rlim=None,
                  zlim=None,
                  p_lim=None,
                  iota_lim=None,
                  rasterized=True,
                  show=False):

    # Separate points based on whether iota is NaN
    valid_mask = ~jnp.isnan(iota_values)
    nan_mask = jnp.isnan(iota_values)

    fig = plt.figure(figsize=(10, 4))
    # Fixed positions: [left, bottom, width, height]
    ax1 = fig.add_axes([0.05, 0.12, 0.35, 0.78])   # left plot
    ax2 = fig.add_axes([0.55, 0.12, 0.35, 0.78])   # right plot
    # Colorbar axes with fixed positions
    cax1 = fig.add_axes([0.41, 0.12, 0.02, 0.78])  # left colorbar
    cax2 = fig.add_axes([0.91, 0.12, 0.02, 0.78])  # right colorbar

    # Plot valid points with color mapping
    if jnp.any(valid_mask):
        # Left plot: color by pressure if available, otherwise by iota
        if p_values is not None:
            s1 = ax1.scatter(  # physical
                (physical_intersections[valid_mask, 0] ** 2 +
                 physical_intersections[valid_mask, 1] ** 2) ** 0.5,
                physical_intersections[valid_mask, 2],
                c=p_values[valid_mask] * 100,
                cmap=cmap_p,
                s=markersize,
                rasterized=rasterized,
                vmin=p_lim[0] * 100 if p_lim is not None else None,
                vmax=p_lim[1] * 100 if p_lim is not None else None
            )
        else:
            s1 = ax1.scatter(  # physical
                (physical_intersections[valid_mask, 0] ** 2 +
                 physical_intersections[valid_mask, 1] ** 2) ** 0.5,
                physical_intersections[valid_mask, 2],
                c=iota_values[valid_mask],
                cmap=cmap_iota,
                s=markersize,
                rasterized=rasterized,
                vmin=iota_lim[0] if iota_lim is not None else None,
                vmax=iota_lim[1] if iota_lim is not None else None
            )
        s2 = ax2.scatter(  # logical
            logical_intersections[valid_mask, 0],
            logical_intersections[valid_mask, 1],
            c=iota_values[valid_mask],
            cmap=cmap_iota,
            s=markersize,
            rasterized=rasterized,
            vmin=iota_lim[0] if iota_lim is not None else None,
            vmax=iota_lim[1] if iota_lim is not None else None
        )

    # Plot NaN points in grey
    if jnp.any(nan_mask):
        ax1.scatter(  # physical
            (physical_intersections[nan_mask, 0] ** 2 +
             physical_intersections[nan_mask, 1] ** 2) ** 0.5,
            physical_intersections[nan_mask, 2],
            c="grey",
            s=markersize,
            rasterized=rasterized,
        )
        ax2.scatter(  # logical
            logical_intersections[nan_mask, 0],
            logical_intersections[nan_mask, 1],
            c="grey",
            s=markersize,
            rasterized=rasterized,
        )

    # Set fixed axis limits - always make the plot square
    # Compute data ranges
    R_data = (physical_intersections[:, 0] ** 2 +
              physical_intersections[:, 1] ** 2) ** 0.5
    z_data = physical_intersections[:, 2]

    # Use provided limits or compute from data
    R_min = Rlim[0] if Rlim is not None else float(jnp.min(R_data))
    R_max = Rlim[1] if Rlim is not None else float(jnp.max(R_data))
    z_min = zlim[0] if zlim is not None else float(jnp.min(z_data))
    z_max = zlim[1] if zlim is not None else float(jnp.max(z_data))

    # Pad to make square
    R_range = R_max - R_min
    z_range = z_max - z_min
    max_range = max(R_range, z_range) * 1.05  # 5% margin
    R_center = (R_min + R_max) / 2
    z_center = (z_min + z_max) / 2
    ax1.set_xlim(R_center - max_range / 2, R_center + max_range / 2)
    ax1.set_ylim(z_center - max_range / 2, z_center + max_range / 2)

    ax1.set(xlabel=r"$R$", ylabel=r"$z$")
    ax2.set(xlabel=r"$r$", ylabel=r"$\theta$", aspect="equal")

    # Only add colorbar if there are valid points
    if jnp.any(valid_mask):
        # Add colorbar to left plot for pressure if available
        if p_values is not None and jnp.any(valid_mask):
            cbar1 = fig.colorbar(s1, cax=cax1, label=r"$p \; [\times 100]$")
        else:
            cax1.set_visible(False)  # Hide left colorbar axis if no pressure

        cbar2 = fig.colorbar(s2, cax=cax2, label=r"$\iota$")

        # Automatically determine rational ticks based on nfp and clipped iota range
        iota_min, iota_max = (
            jnp.nanmin(iota_values[valid_mask]),
            jnp.nanmax(iota_values[valid_mask]),
        )
        rational_ticks = []
        rational_labels = []
        seen_rationals = set()
        for m in range(1, denom_max // nfp + 1):
            m_scaled = m * nfp
            for n in range(1, denom_max):  # reasonable range for denominators
                rational = m_scaled / n
                if iota_min <= rational <= iota_max and rational not in seen_rationals:
                    rational_ticks.append(rational)
                    # Can uncomment this to simplify the fractions
                    g = 1  # jnp.gcd(m_scaled, n)
                    rational_labels.append(
                        f"{int(m_scaled // g)}/{int(n // g)}")
                    seen_rationals.add(rational)
        if rational_ticks:
            cbar2.set_ticks(rational_ticks)
            cbar2.set_ticklabels(rational_labels)

        if show:
            plt.show()
    else:
        # Hide colorbar axes if no valid points
        cax1.set_visible(False)
        cax2.set_visible(False)

    return fig, (ax1, ax2)

# TODO: might be obsolete after the new plotting refactor


def plot_crossections_separate(
    p_h: Callable,
    grids_pol: list,
    zeta_vals: list,
    textsize: int = 16,
    ticksize: int = 16,
    plot_centerline: bool = False,
):
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
    fig, axes = plt.subplots(1, numplots, figsize=(16, 16 / 5))
    axes = axes.flatten()

    last_c = None
    for i, (ax, grid) in enumerate(zip(axes, grids_pol)):

        R = jnp.sqrt(grid[2][0] ** 2 + grid[2][1] ** 2)
        z = grid[2][2]

        vals = jax.lax.map(p_h, grid[0], 
            batch_size=mrx.MAP_BATCH_SIZE_INNER).reshape(*grid[2][0].shape)

        # draw contour above the guide lines
        last_c = ax.contourf(R, z, vals, 25, cmap="plasma", zorder=2)

        # ensure axis artists (like text/legend) are above the guide lines
        ax.set_axisbelow(False)

        if plot_centerline:
            ax.axvline(
                1.0, color="k", linestyle=":", linewidth=1.5, zorder=3, clip_on=True
            )
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
        ax.text(
            0.98,
            0.98,
            label,
            transform=ax.transAxes,
            fontsize=textsize,
            ha="right",
            va="top",
            zorder=10,
            bbox=dict(
                facecolor="white",
                edgecolor="black",
                boxstyle="round,pad=0.3",
                alpha=1.0,
            ),
        )

        # force same limits across all subplots
    Rmins, Rmaxs, Zmins, Zmaxs = [], [], [], []
    for grid in grids_pol:
        R = jnp.sqrt(grid[2][0] ** 2 + grid[2][1] ** 2)
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
        anchor_ax = axes if hasattr(axes, "annotate") else None

    if anchor_ax is not None:
        x0, y0 = -0.01, -0.01  # anchor location in axis fraction coordinates
        arrow_len = 0.16  # length of each arrow in axis fraction units

        # upward arrow for z

        # annotate the center (dotted) line at the very top of the axis
        if plot_centerline:
            anchor_ax.text(
                0.5,
                1.02,
                r"$R = 1$",
                transform=anchor_ax.transAxes,
                fontsize=textsize,
                ha="center",
                va="bottom",
                zorder=12,
                bbox=dict(facecolor="white", edgecolor="none",
                          alpha=0.8, pad=0.2),
            )
        anchor_ax.annotate(
            "",
            xy=(x0, y0 + arrow_len),
            xytext=(x0, y0),
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="->", linewidth=1.5, color="k"),
        )
        # rightward arrow for R
        anchor_ax.annotate(
            "",
            xy=(x0 + arrow_len, y0),
            xytext=(x0, y0),
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="->", linewidth=1.5, color="k"),
        )

        # labels for arrows
        anchor_ax.text(
            x0 - 0.01,
            y0 + arrow_len + 0.01,
            r"$z$",
            transform=anchor_ax.transAxes,
            fontsize=textsize + 2,
            ha="center",
            va="bottom",
        )
        anchor_ax.text(
            x0 + arrow_len + 0.01,
            y0 - 0.01,
            r"$R$",
            transform=anchor_ax.transAxes,
            fontsize=textsize + 2,
            ha="left",
            va="center",
        )

    cbar = fig.colorbar(
        last_c, cax=cbar_ax, format=mticker.ScalarFormatter(useMathText=True)
    )
    cbar.formatter.set_powerlimits((0, 0))  # always show scientific notation
    cbar.update_ticks()
    cbar.ax.yaxis.get_offset_text().set_fontsize(ticksize)

    # plt.tight_layout(rect=[0, 0, 0.85, 1])  # leave room for cbar
    return fig, axes
# %%
