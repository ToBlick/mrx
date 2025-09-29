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
import matplotlib.pyplot as plt
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
                 nx=64,
                 ny=64,
                 nz=64):
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


def get_2d_grids(F, zeta=0, nx=64, tol=1e-6):
    tol = 1e-6
    _x1 = jnp.linspace(tol, 1 - tol, nx)
    _x2 = jnp.linspace(0, 1, nx)
    _x3 = jnp.ones(1) * zeta
    _x = jnp.array(jnp.meshgrid(_x1, _x2, _x3))
    _x = _x.transpose(1, 2, 3, 0).reshape(nx**2, 3)
    _y = jax.vmap(F)(_x)
    _y1 = _y[:, 0].reshape(nx, nx)
    _y2 = _y[:, 1].reshape(nx, nx)
    _y3 = _y[:, 2].reshape(nx, nx)
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

def trajectory_plane_intersections(trajectories, plane_val=0.5, axis=1):
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
    intersections = trajectories[:, :-1, :] + t * (trajectories[:, 1:, :] - trajectories[:, :-1, :])

    return intersections, mask

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
    trajectories = jnp.array(trajectories).reshape(n_batch * n_loop, n_saves, 3) % 1
    
    physical_trajectories = jax.vmap(F)(trajectories.reshape(-1, 3))
    physical_trajectories = physical_trajectories.reshape(trajectories.shape[0], trajectories.shape[1], 3)

    intersections, mask = trajectory_plane_intersections(trajectories, plane_val=plane_val, axis=axis)

    if cylindrical:
        def F_cyl(p):
            x, y, z = F(p)
            r = jnp.sqrt(x**2 + y**2)
            phi = jnp.arctan2(y, x)
            return jnp.array([r, phi, z])
        physical_intersections = jax.vmap(F_cyl)(intersections.reshape(-1, 3))
    else:
        physical_intersections = jax.vmap(F)(intersections.reshape(-1, 3))
    physical_intersections = physical_intersections.reshape(intersections.shape[0], intersections.shape[1], 3)
    
    print("Plotting Poincaré sections...")
    # physical domain
    fig1, ax1 = plt.subplots(figsize=FIG_SIZE_SQUARE)
    for i, t in enumerate(physical_intersections):
        current_color = colors[i % len(colors)]  # Cycle through the defined colors
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
        else: # for cylindrical, always plot (r,z) (axis = 2)
            ax1.scatter(t[:, 0], t[:, 2], s=1, color=current_color)
            ax1.set_xlabel(r'$R$', fontsize=LABEL_SIZE)
            ax1.set_ylabel(r'$z$', fontsize=LABEL_SIZE)
    ax1.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    ax1.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(outdir + name + "poincare_physical.png", dpi=600, bbox_inches='tight')
    
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
    plt.savefig(outdir + name + "poincare_logical.png", dpi=600, bbox_inches='tight')
    
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

def pressure_plot(p, Seq, F, outdir, name,
                   resolution=128, zeta=0, tol=1e-3,
                  SQUARE_FIG_SIZE=(8, 8), LABEL_SIZE=20, TICK_SIZE=16, LINE_WIDTH=2.5):

    p_h = DiscreteFunction(p, Seq.Λ0, Seq.E0_0.matrix())
    p_h_xyz = Pushforward(p_h, F, 0)

    _s = jax.vmap(F)(jnp.vstack(
        [jnp.ones(256), jnp.linspace(0, 1, 256), jnp.zeros(256)]).T)

    fig, ax = plt.subplots(figsize=SQUARE_FIG_SIZE)

    # Plot the line first
    ax.plot(_s[:, 0], _s[:, 2], 'k--',
            linewidth=LINE_WIDTH, label="trajectory")
    
    _x, _y, (_y1, _y2, _y3), (_x1, _x2, _x3) = get_2d_grids(
        F, zeta=zeta, nx=resolution, tol=tol)

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
    
    helicity_change = jnp.abs(jnp.array(helicity_trace - helicity_trace[0]) )

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
