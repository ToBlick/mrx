"""
Plotting utilities for finite element analysis results.
"""
# %%
import os
from typing import Callable
import diffrax
import h5py
import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from mrx.mappings import cerfon_map
from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction, Pushforward

__all__ = []

# Base marker styles for different data series
base_markers = [
    'circle', 'triangle-down', 'star', 'triangle-left', 'triangle-right',
    'triangle-ne', 'triangle-se', 'triangle-sw', 'triangle-nw',
    'square', 'pentagon', 'triangle-up', 'hexagon', 'hexagon2',
    'cross', 'x', 'diamond', 'diamond-open', 'line-ns', 'line-ew'
]

# Default color scale for plots
colorbar = 'plasma'

def get_3d_grids(
    F : Callable, x_min : float = 0.0, x_max: float = 1.0, y_min: float = 0.0, y_max: float = 1.0,
    z_min: float = 0.0, z_max: float = 1.0, nx: int = 16, ny: int = 16, nz: int = 16):
    """
    Get 3D grids for plotting.

    Parameters
    ----------
    F : callable
        Coordinate transformation function.
    x_min : float, default=0.0  
        Minimum value of the x coordinate.  
    x_max : float, default=1.0
        Maximum value of the x coordinate.
    y_min : float, default=0.0
        Minimum value of the y coordinate.
    y_max : float, default=1.0
        Maximum value of the y coordinate.
    z_min : float, default=0.0
        Minimum value of the z coordinate.
    z_max : float, default=1.0
        Maximum value of the z coordinate.
    nx : int, default=16
        Number of grid points in the x direction.
    ny : int, default=16
        Number of grid points in the y direction.
    nz : int, default=16
        Number of grid points in the z direction.

    Returns
    -------
    _x : jnp.ndarray (nx*ny*nz, 3)
    _y : jnp.ndarray (nx*ny*nz, 3)
    _y1 : jnp.ndarray (nx, ny, nz)
    _y2 : jnp.ndarray (nx, ny, nz)
    _y3 : jnp.ndarray (nx, ny, nz)
    _x1 : jnp.ndarray (nx)
    _x2 : jnp.ndarray (ny)
    _x3 : jnp.ndarray (nz)
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
    F : Callable, cut_value: float = 0.0, cut_axis: int = 2, nx: int = 64, ny: int = 64, 
    nz: int = 64, tol1: float = 1e-6, tol2: float = 0.0, tol3: float = 0.0,
    x_min: float = 0.0, x_max: float = 1.0, y_min: float = 0.0, y_max: float = 1.0,
    z_min: float = 0.0, z_max: float = 1.0, invert_x: bool = False, invert_y: bool = False, invert_z: bool = False
    ):
    """
    Get 2D grids for plotting.

    Parameters
    ----------
    F : callable
        Coordinate transformation function.
    cut_value : float, default=0.0
        Value to cut the grid at.
    cut_axis : int, default=2
        Axis to cut the grid at.
    nx : int, default=64
        Number of grid points in the x direction.
    ny : int, default=64
        Number of grid points in the y direction.
    nz : int, default=64
        Number of grid points in the z direction.
    tol1 : float, default=1e-6
        Tolerance for the grid in the x direction.
    tol2 : float, default=0.0
        Tolerance for the grid in the y direction.
    tol3 : float, default=0.0
        Tolerance for the grid in the z direction.
    x_min : float, default=0.0
        Minimum value of the x coordinate.
    x_max : float, default=1.0
        Maximum value of the x coordinate.
    y_min : float, default=0.0
        Minimum value of the y coordinate.
    y_max : float, default=1.0
        Maximum value of the y coordinate.
    z_min : float, default=0.0
        Minimum value of the z coordinate.
    z_max : float, default=1.0
        Maximum value of the z coordinate.

    Returns
    -------
    _x : jnp.ndarray (n1*n2, 3)
    _y : jnp.ndarray (n1*n2, 3)
    _y1 : jnp.ndarray (n1, n2)
    _y2 : jnp.ndarray (n1, n2)
    _y3 : jnp.ndarray (n1, n2)
    _x1 : jnp.ndarray (n1)
    _x2 : jnp.ndarray (n2)
    _x3 : jnp.ndarray (nz)
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


def get_1d_grids(F : Callable, zeta: float = 0.0, chi: float = 0.0, nx: int = 64, tol: float = 1e-6):
    """
    Get 1D grids for plotting.

    Parameters
    ----------
    F : callable
        Coordinate transformation function.
    zeta : float, default=0.0
        Toroidal angle.
    chi : float, default=0.0
        Poloidal angle.
    nx : int, default=64
        Number of grid points.
    tol : float, default=1e-6
        Tolerance for the grid.

    Returns
    -------
    _x : jnp.ndarray (nx, 3)
    _y : jnp.ndarray (nx, 3)
    _y1 : jnp.ndarray (nx)
    _y2 : jnp.ndarray (nx)
    _y3 : jnp.ndarray (nx)
    _x1 : jnp.ndarray (nx)
    _x2 : jnp.ndarray (1)
    _x3 : jnp.ndarray (1)
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


def trajectory_plane_intersections_jit(trajectories: jnp.ndarray, plane_val: float = 0.5, axis: int = 1):
    """
    Vectorized + jittable function for computing intersections of trajectories with a plane.

    Parameters
    ----------
    trajectories : jnp.ndarray (N, T, D)
        Trajectories to intersect with the plane. The shape is (N, T, D), 
        where N is the number of trajectories, T is the number of time steps, 
        and D is the dimension of the space.
    plane_val    : float, default=0.5
        Value of the plane to intersect with.
    axis         : int, default=1
        Coordinate axis to intersect with.

    Returns
    -------
    intersections : jnp.ndarray (N, T-1, D)
        Intersection points for each segment. Non-crossings are filled with NaN.
    mask : jnp.ndarray (N, T-1)
        True if the corresponding segment contains an intersection.
    """
    x = trajectories[..., axis] # (N, T)
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


def plot_crossections_separate(p_h : Callable, grids_pol : list[tuple], zeta_vals : list[float], textsize : int = 16, ticksize : int = 16, plot_centerline : bool = False):
    """
    Plot cross-sections of a function.

    Parameters
    ----------
    p_h : Callable
        Function to plot.
    grids_pol : list[tuple]
        List of tuples containing the grid points and the function values.
    zeta_vals : list[float]
        List of toroidal angles to plot.
    textsize : int, default=16
        Font size for the text.
    ticksize : int, default=16
        Font size for the ticks.
    plot_centerline : bool, default=False
        Whether to plot the centerline.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    axes : list of matplotlib.axes.Axes
        List of axes objects.
    """
    numplots = len(grids_pol)
    fig, axes = plt.subplots(1, numplots, figsize=(16, 16/5))
    axes = axes.flatten()

    last_c = None
    for i, (ax, grid) in enumerate(zip(axes, grids_pol)):
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

    # Set axis limits for all subplots
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
    return fig, axes


def plot_torus(p_h, grids_pol,
               grid_surface,
               figsize : tuple = (12, 8),
               labelsize : int = 20,
               ticksize : int = 16,
               gridlinewidth : float = 0.01,
               cstride : int = 4,
               elev : float = 30,
               azim : float = 140,
               noaxes : bool = False,
               add_colorbar : bool = False):
    """
    Plot a function on a torus.

    Parameters
    ----------
    p_h : callable
        Function to plot.
    grids_pol : list of tuples
        List of tuples containing the grid points and the function values.
    grid_surface : tuple
        Tuple containing the grid points and the function values.
    figsize : tuple, default=(12, 8)
        Size of the figure.
    labelsize : int, default=20
        Font size for the labels.
    ticksize : int, default=16
        Font size for the ticks.
    gridlinewidth : float, default=0.01
        Width of the grid lines.
    cstride : int, default=4
        Stride for the colorbar.
    elev : float, default=30
        Elevation angle.
    azim : float, default=140
        Azimuth angle.
    noaxes : bool, default=False
        Whether to plot the axes.
    add_colorbar : bool, default=False  
        Whether to add a colorbar.

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
    if add_colorbar:
        norm = mpl.colors.Normalize(vmin=vals_min, vmax=vals_max)
        sm = mpl.cm.ScalarMappable(cmap='plasma', norm=norm)
        sm.set_array(vals_np)  # provide data for the colorbar
        cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.08)
        cbar.set_label(r'$p$', fontsize=labelsize)
        cbar.ax.tick_params(labelsize=ticksize)

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


def plot_crossections(f : Callable, grids : list[tuple]):
    """
    Plot cross-sections of a function.

    Parameters
    ----------
    f : callable
        Function to plot.
    grids : list of tuples
        List of tuples containing the grid points and the function values.

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


def trajectory_plane_intersections_list(trajectory: jnp.ndarray, plane_point: jnp.ndarray, plane_normal: jnp.ndarray):
    """
    Compute intersections of a 3D trajectory with a general plane.

    Returns a list of intersection points (no NaNs, no masks).

    Parameters
    ----------
    trajectory : jnp.ndarray (T, 3)
        Trajectory to intersect with the plane. The shape is (T, 3), 
        where T is the number of time steps.
    plane_point : jnp.ndarray (3,)
        Point on the plane.
    plane_normal : jnp.ndarray (3,)
        Normal vector to the plane.

    Returns
    -------
    intersections : list of jnp.ndarray, each of shape (3,)
        Intersection points.
    """
    # TODO: Why convert to np.ndarray?
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


def poincare_plot(
    outdir : str, vector_field : Callable, F : Callable, x0 : jnp.ndarray, 
    n_loop : int, n_batch : int, colors : list[str], plane_val : float = 0.5, axis : int = 1, 
    final_time : float = 10_000, n_saves : int = 20_000, max_steps : int = 150000, r_tol : float = 1e-7, 
    a_tol : float = 1e-7, cylindrical : bool = False, name : str = ""):
    """
    Plot Poincaré sections of a vector field.

    Parameters
    ----------
    outdir : str
        Directory to save the plots.
    vector_field : Callable
        Vector field to plot.
    F : Callable
        Coordinate transformation function.
    x0 : jnp.ndarray (n_batch, n_loop, 3)
        Initial conditions for the trajectories. The shape is (n_batch, n_loop, 3), 
        where n_batch is the number of batches, n_loop is the number of loops, 
        and 3 is the dimension of the space.
    n_loop : int, default=1
        Number of loops.
    n_batch : int, default=1
        Number of batches.
    colors : list of colors
        List of colors for the trajectories.
    plane_val : float, default=0.5
        Value of the plane to intersect with.
    axis : int, default=1
        Coordinate axis to intersect with.
    final_time : float, default=10_000
        Final time.
    n_saves : int, default=20_000
        Number of saves.
    max_steps : int, default=150000
        Maximum number of steps.
    r_tol : float, default=1e-7
        Relative tolerance.
    a_tol : float, default=1e-7
        Absolute tolerance.
    cylindrical : bool, default=False
        Whether to plot in cylindrical coordinates.
    name : str, default=""
        Name of the plot.
    """

    os.makedirs(outdir, exist_ok=True)

    # --- Figure settings ---
    FIG_SIZE_SQUARE = (8, 8)
    LABEL_SIZE = 20
    TICK_SIZE = 16

    assert x0.shape == (n_batch, n_loop, 3)

    # Define the ODE term and solver, use classic Dopri5 solver (Dormand-Prince 4/5)
    term = diffrax.ODETerm(vector_field)
    solver = diffrax.Dopri5()

    # Save at the specified time steps and controller for the stepsize
    saveat = diffrax.SaveAt(ts=jnp.linspace(0, final_time, n_saves))
    stepsize_controller = diffrax.PIDController(rtol=r_tol, atol=a_tol)

    # Initialize an empty list to store the trajectories
    trajectories = []

    # Compute trajectories
    print("Integrating field lines...")

    # Integrate each of the trajectories
    for x in x0:
        trajectories.append(jax.vmap(lambda x0: diffrax.diffeqsolve(term, solver,
                                                                    t0=0, t1=final_time, dt0=None,
                                                                    y0=x0,
                                                                    max_steps=max_steps,
                                                                    saveat=saveat, stepsize_controller=stepsize_controller).ys)(x))
    trajectories = jnp.array(trajectories).reshape(n_batch * n_loop, n_saves, 3) % 1

    physical_trajectories = jax.vmap(F)(trajectories.reshape(-1, 3))
    physical_trajectories = physical_trajectories.reshape(
        trajectories.shape[0], trajectories.shape[1], 3)

    intersections, _ = trajectory_plane_intersections_jit(trajectories, plane_val=plane_val, axis=axis)

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
    _, ax1 = plt.subplots(figsize=FIG_SIZE_SQUARE)
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
    plt.savefig(outdir + name + "poincare_physical.png",
                dpi=600, bbox_inches='tight')

    # logical domain
    _, ax1 = plt.subplots(figsize=FIG_SIZE_SQUARE)
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
    plt.savefig(outdir + name + "poincare_logical.png",
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
        plt.savefig(outdir + name + "field_line_" + str(i) + ".pdf", bbox_inches='tight')


def pressure_plot(p : jnp.ndarray, Seq : DeRhamSequence, F : Callable, outdir : str, name : str,
                  resolution : int = 128, zeta : float = 0.0, tol : float = 1e-3,
                  SQUARE_FIG_SIZE : tuple = (8, 8), LABEL_SIZE : int = 20,
                  TICK_SIZE : int = 16, LINE_WIDTH : float = 2.5):
    """
    Plot a pressure contour plot.

    Parameters
    ----------
    p : jnp.ndarray
        Pressure field.
    Seq : DeRhamSequence
        DeRham sequence.
    F : Callable
        Coordinate transformation function.
    outdir : str
        Directory to save the plot.
    name : str
        Name of the plot.
    resolution : int, default=128
        Resolution of the plot.
    zeta : float, default=0.0
        Toroidal angle.
    tol : float, default=1e-3
        Tolerance for the plot.
    SQUARE_FIG_SIZE : tuple, default=(8, 8)
        Size of the figure.
    LABEL_SIZE : int, default=20
        Font size for the labels.
    TICK_SIZE : int, default=16
        Font size for the ticks.
    LINE_WIDTH : float, default=2.5
        Width of the line.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axes object.
    """

    p_h = DiscreteFunction(p, Seq.Λ0, Seq.E0)
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
    plt.savefig(outdir + name, bbox_inches='tight')
    plt.close()


def trace_plot(
    iterations : jnp.ndarray, force_trace : jnp.ndarray, helicity_trace : jnp.ndarray, divergence_trace : jnp.ndarray, 
    velocity_trace : jnp.ndarray, wall_time_trace : jnp.ndarray, energy_trace : jnp.ndarray = None, 
    outdir : str = './', name : str = '', FIG_SIZE : tuple = (12, 6), LABEL_SIZE : int = 20, 
    TICK_SIZE : int = 16, LINE_WIDTH : float = 2.5, LEGEND_SIZE : int = 16):
    """
    Plot a trace plot.

    Parameters
    ----------
    iterations : jnp.ndarray
        Iterations.
    force_trace : jnp.ndarray
        Force trace.
    energy_trace : jnp.ndarray, default=None
        Energy trace.
    helicity_trace : jnp.ndarray
        Helicity trace.
    divergence_trace : jnp.ndarray
        Divergence trace.
    velocity_trace : jnp.ndarray
        Velocity trace.
    wall_time_trace : jnp.ndarray
        Wall time trace.
    outdir : str, default='./'
        Directory to save the plot.
    name : str, default=''
        Name of the plot.
    FIG_SIZE : tuple, default=(12, 6)
        Size of the figure.
    LABEL_SIZE : int, default=20
        Font size for the labels.
    TICK_SIZE : int, default=16
        Font size for the ticks.
    LINE_WIDTH : float, default=2.5
        Width of the line.
    LEGEND_SIZE : int, default=16
        Font size for the legend.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axes object.
    ax2_top : matplotlib.axes.Axes
        Axes object for the top y-axis.
    """
    fig1, ax2 = plt.subplots(figsize=FIG_SIZE)

    color1 = 'purple'
    color2 = 'black'
    color3 = 'darkgray'
    color4 = 'teal'
    color5 = 'orange'

    # Plot Energy on the left y-axis (ax1)
    if energy_trace is not None:
        ax1 = ax2.twinx()
        ax1.set_xlabel(r'$n$', fontsize=LABEL_SIZE)
        ax1.set_ylabel(r'$\frac{1}{2} \| B \|^2$',
                       color=color4, fontsize=LABEL_SIZE)
        ax1.semilogy(energy_trace[0] - jnp.array(energy_trace),
                 label=r'$\frac{1}{2} \| B \|^2$', color=color4, linestyle='-.', lw=LINE_WIDTH)
        # ax1.plot(jnp.pi * jnp.array(H_trace), label=r'$\pi \, (A, B)$', color=color1, linestyle="--", lw=LINE_WIDTH)
        ax1.tick_params(axis='y', labelcolor=color4, labelsize=TICK_SIZE)
        ax1.tick_params(axis='x', labelsize=TICK_SIZE)  # Set x-tick size
        ax1.set_xscale('log')
        ax1.tick_params(axis='y', labelcolor=color4, labelsize=TICK_SIZE)

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

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_top.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2,
               loc='best', fontsize=LEGEND_SIZE)
    ax2.legend(loc='best', fontsize=LEGEND_SIZE)
    ax2.grid(which="both", linestyle="--", linewidth=0.5)
    fig1.tight_layout()
    plt.savefig(outdir + name, bbox_inches='tight')
    plt.close()


def generate_solovev_plots(name : str):
    """
    Generate all plots for a Solovev configuration.

    Parameters
    ----------
    name : str
        Name of the configuration.
    """

    jax.config.update("jax_enable_x64", True)

    outdir = "script_outputs/solovev/" + name + "/"
    os.makedirs(outdir, exist_ok=True)

    print("Generating plots for " + name + "...")

    with h5py.File("script_outputs/solovev/" + name + ".h5", "r") as f:
        CONFIG = {k: v for k, v in f["config"].attrs.items()}
        # decode strings back if needed
        CONFIG = {k: v.decode() if isinstance(v, bytes)
                  else v for k, v in CONFIG.items()}

        p_final = f["p_final"][:]
        iterations = f["iterations"][:]
        force_trace = f["force_trace"][:]
        velocity_trace = f["velocity_trace"][:]
        helicity_trace = f["helicity_trace"][:]
        energy_trace = f["energy_trace"][:]
        divergence_B_trace = f["divergence_B_trace"][:]
        wall_time_trace = f["wall_time_trace"][:]
        if CONFIG["save_B"]:
            p_fields = f["p_fields"][:]

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
    pressure_plot(p_final, Seq, F, outdir, name="p_final.pdf", zeta=0)
    if CONFIG["save_B"]:
        for i, p in enumerate(p_fields):
            pressure_plot(p,
                          Seq,
                          F,
                          outdir,
                          name=f"p_iter_{i*CONFIG['save_every']:06d}.pdf",
                          zeta=0)

    print("Generating convergence plot...")
    # Figure 2: Energy and Force

    trace_plot(iterations=iterations,
               force_trace=force_trace,
               energy_trace=energy_trace,
               helicity_trace=helicity_trace,
               divergence_trace=divergence_B_trace,
               velocity_trace=velocity_trace,
               wall_time_trace=wall_time_trace,
               outdir=outdir,
               name="force_trace.pdf",
               CONFIG=CONFIG)

def set_axes_equal(ax : plt.Axes):
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
