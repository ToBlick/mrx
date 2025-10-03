"""
Plotting utilities for finite element analysis results.

This module provides functions for creating visualizations of convergence plots
and other analysis results using Plotly.
"""
# %%
import os

import diffrax
import h5py
import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go

from mrx.BoundaryFitting import cerfon_map
from mrx.DeRhamSequence import DeRhamSequence
from mrx.DifferentialForms import DiscreteFunction, Pushforward

__all__ = ['converge_plot']

# Base marker styles for different data series
base_markers = [
    'circle', 'triangle-down', 'star', 'triangle-left', 'triangle-right',
    'triangle-ne', 'triangle-se', 'triangle-sw', 'triangle-nw',
    'square', 'pentagon', 'triangle-up', 'hexagon', 'hexagon2',
    'cross', 'x', 'diamond', 'diamond-open', 'line-ns', 'line-ew'
]

# Default color scale for plots
colorbar = 'Viridis'


def converge_plot(err, ns, ps, qs):
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


def get_3d_grids(F,
                 x_min=0, x_max=1,
                 y_min=0, y_max=1,
                 z_min=0, z_max=1,
                 nx=16,
                 ny=16,
                 nz=16):
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


def get_2d_grids(F, cut_value=0, cut_axis=2, nx=64, ny=64, nz=64, tol1=1e-6, tol2=0, tol3=0,
                 x_min=0, x_max=1,
                 y_min=0, y_max=1,
                 z_min=0, z_max=1, invert_x=False, invert_y=False, invert_z=False):
    _x1 = jnp.linspace(x_min + tol2, x_max - tol2, nx)
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


def get_1d_grids(F, zeta=0, chi=0, nx=64, tol=1e-6):
    tol = 1e-6
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


def trajectory_plane_intersections_jit(trajectories, plane_val=0.5, axis=1):
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


def plot_crossections_separate(p_h, grids_pol, zeta_vals, textsize=16, ticksize=16, plot_centerline=False):

    numplots = len(grids_pol)
    fig, axes = plt.subplots(2, numplots//2, figsize=(12, 6))
    axes = axes.flatten()

    last_c = None
    for i, (ax, grid) in enumerate(zip(axes, grids_pol)):
        R = jnp.sqrt(grid[2][0]**2 + grid[2][1]**2)
        Z = grid[2][2]

        vals = jax.vmap(p_h)(grid[0]).reshape(R.shape)

        # draw contour above the guide lines
        last_c = ax.contourf(R, Z, vals, 25, cmap="plasma", zorder=2)

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
        # place a small boxed text in the bottom-right of the axis (no legend handle/whitespace)
        ax.text(0.98, 0.02, label, transform=ax.transAxes,
                fontsize=textsize, ha='right', va='bottom', zorder=10,
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3', alpha=1.0))

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

    fig.colorbar(last_c, cax=cbar_ax)

    # plt.tight_layout(rect=[0, 0, 0.85, 1])  # leave room for cbar
    return fig, axes


def plot_torus(p_h,
               grids_pol,
               grid_surface,
               figsize=(12, 8),
               labelsize=20,
               ticksize=16,
               gridlinewidth=0.01,
               cstride=4,
               elev=30,
               azim=140):

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
    # ax.set_axis_off()

    return fig, ax


def plot_crossections(f, grids):
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


def trajectory_plane_intersections_list(trajectory, plane_point, plane_normal):
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


def poincare_plot(outdir, vector_field, F, x0, n_loop, n_batch, colors, plane_val, axis, final_time=10_000, n_saves=20_000, max_steps=150000, r_tol=1e-7, a_tol=1e-7, cylindrical=False, name=""):

    os.makedirs(outdir, exist_ok=True)

    # --- Figure settings ---
    FIG_SIZE = (12, 6)
    FIG_SIZE_SQUARE = (8, 8)
    TITLE_SIZE = 20
    LABEL_SIZE = 20
    TICK_SIZE = 16
    LINE_WIDTH = 2.5
    LEGEND_SIZE = 16

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
    plt.savefig(outdir + name + "poincare_physical.png",
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
        plt.savefig(outdir + name + "field_line_" +
                    str(i) + ".pdf", bbox_inches='tight')


def pressure_plot(p, Seq, F, outdir, name,
                  resolution=128, zeta=0, tol=1e-3,
                  SQUARE_FIG_SIZE=(8, 8), LABEL_SIZE=20,
                  TICK_SIZE=16, LINE_WIDTH=2.5):

    p_h = DiscreteFunction(p, Seq.Λ0, Seq.E0_0.matrix())
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
# %%


def trace_plot(iterations,
               force_trace,
               helicity_trace,
               divergence_trace,
               energy_trace,
               velocity_trace,
               wall_time_trace,
               outdir,
               name,
               CONFIG,
               FIG_SIZE=(12, 6), LABEL_SIZE=20, TICK_SIZE=16, LINE_WIDTH=2.5, LEGEND_SIZE=16):
    fig1, ax2 = plt.subplots(figsize=FIG_SIZE)

    # # Plot Energy on the left y-axis (ax1)
    #
    # ax1 = ax2.twinx()
    # ax1.set_xlabel(r'$n$', fontsize=LABEL_SIZE)
    # ax1.set_ylabel(r'$\frac{1}{2} \| B \|^2$',
    #                color=color1, fontsize=LABEL_SIZE)
    # ax1.semilogy(energy_trace[0] - jnp.array(energy_trace),
    #          label=r'$\frac{1}{2} \| B \|^2$', color=color1, linestyle='-.', lw=LINE_WIDTH)
    # # ax1.plot(jnp.pi * jnp.array(H_trace), label=r'$\pi \, (A, B)$', color=color1, linestyle="--", lw=LINE_WIDTH)
    # ax1.tick_params(axis='y', labelcolor=color1, labelsize=TICK_SIZE)
    # ax1.tick_params(axis='x', labelsize=TICK_SIZE)  # Set x-tick size
    # ax1.set_xscale('log')
    # ax1.tick_params(axis='y', labelcolor=color1, labelsize=TICK_SIZE)

    color1 = 'purple'
    color2 = 'black'
    color3 = 'darkgray'
    color4 = 'teal'
    color5 = 'orange'

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
    fig1.tight_layout()
    plt.savefig(outdir + name, bbox_inches='tight')
# %%


def generate_solovev_plots(name):
    jax.config.update("jax_enable_x64", True)

    outdir = "script_outputs/solovev/" + name + "/"
    os.makedirs(outdir, exist_ok=True)

    print("Generating plots for " + name + "...")

    with h5py.File("script_outputs/solovev/" + name + ".h5", "r") as f:
        CONFIG = {k: v for k, v in f["config"].attrs.items()}
        # decode strings back if needed
        CONFIG = {k: v.decode() if isinstance(v, bytes)
                  else v for k, v in CONFIG.items()}

        B_final = f["B_final"][:]
        p_final = f["p_final"][:]
        iterations = f["iterations"][:]
        force_trace = f["force_trace"][:]
        velocity_trace = f["velocity_trace"][:]
        helicity_trace = f["helicity_trace"][:]
        energy_trace = f["energy_trace"][:]
        energy_diff_trace = f["energy_diff_trace"][:]
        divergence_B_trace = f["divergence_B_trace"][:]
        picard_iterations = f["picard_iterations"][:]
        picard_errors = f["picard_errors"][:]
        timesteps = f["timesteps"][:]
        total_time = f["total_time"][0]
        time_setup = f["time_setup"][0]
        time_solve = f["time_solve"][0]
        wall_time_trace = f["wall_time_trace"][:]
        if CONFIG["save_B"]:
            # B_fields = f["B_fields"][:]
            p_fields = f["p_fields"][:]
            B_fields = f["B_fields"][:]

    # Step 1: get F
    delta = CONFIG["delta"]
    kappa = CONFIG["kappa"]
    q_star = CONFIG["q_star"]
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

    # print("Plotting Poincaré sections and field lines...")
    # B_h = DiscreteFunction(B_hat, Seq.Λ2, Seq.E2_0.matrix())

    # @jax.jit
    # def vector_field(t, p, args):
    #     r, χ, z = p
    #     r = jnp.clip(r, 1e-6, 1)
    #     χ = χ % 1.0
    #     z = z % 1.0
    #     x = jnp.array([r, χ, z])
    #     DFx = jax.jacfwd(F)(x)
    #     norm = ((DFx @ B_h(x)) @ DFx @ B_h(x))**0.5
    #     return B_h(x) / (norm + 1e-9)

    # n_loop = 5
    # n_batch = 5

    # x0s = jnp.vstack(
    #     (jnp.linspace(0.05, 0.95, n_loop * n_batch),
    #     jnp.zeros(n_loop * n_batch),
    #     jnp.zeros(n_loop * n_batch))
    # ).T

    # n_cols = x0s.shape[1]
    # cm = plt.cm.plasma
    # vals = jnp.linspace(0, 1, n_cols + 2)[:-2]

    # # Interleave from start and end
    # order = jnp.ravel(jnp.column_stack([jnp.arange(n_cols//2), n_cols-1-jnp.arange(n_cols//2)]))
    # if n_cols % 2 == 1:
    #     order = jnp.append(order, n_cols//2)

    # colors = cm(vals[order])

    # x0s = x0s.T.reshape(n_batch, n_loop, 3)

    # poincare_plot(outdir, vector_field, F, x0s, n_loop, n_batch, colors, plane_val=0.25, axis=2, final_time=5_000, n_saves=20_000, cylindrical=True, r_tol=solver_tol, a_tol=solver_tol)

# %%


def set_axes_equal(ax):
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
