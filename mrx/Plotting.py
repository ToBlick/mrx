"""
Plotting utilities for finite element analysis results.

This module provides functions for creating visualizations of convergence plots
and other analysis results using Plotly.
"""

import os

import h5py
import jax
import jax.numpy as jnp
import matplotlib as plt
import matplotlib.pyplot as plt
import plotly.colors as pc
import plotly.graph_objects as go

from mrx.BoundaryFitting import get_lcfs_F
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

# %%


def generate_solovev_plots(name):
    outdir = "script_outputs/solovev/"
    os.makedirs(outdir, exist_ok=True)

    print("Generating plots for " + name + "...")

    # --- Figure settings ---
    FIG_SIZE = (12, 6)
    SQUARE_FIG_SIZE = (8, 8)
    TITLE_SIZE = 20
    LABEL_SIZE = 20
    TICK_SIZE = 16
    LINE_WIDTH = 2.5

    with h5py.File("script_outputs/solovev/" + name + ".h5", "r") as f:
        B_hat = f["B_hat"][:]
        p_hat = f["p_hat"][:]

        cfg = {k: v for k, v in f["config"].attrs.items()}
        # decode strings back if needed
        cfg = {k: v.decode() if isinstance(v, bytes)
               else v for k, v in cfg.items()}

    R0 = cfg["R_0"]
    aR = cfg["a_R"]
    π = jnp.pi

    # Step 1: Reconstruct F
    if cfg["circular_cross_section"]:
        def F(x):
            r, χ, z = x
            return jnp.ravel(jnp.array(
                [(R0 + aR * r * jnp.cos(2 * π * χ)) * jnp.cos(2 * π * z),
                 -(R0 + aR * r * jnp.cos(2 * π * χ)) * jnp.sin(2 * π * z),
                 aR * r * jnp.sin(2 * π * χ)]))
    else:
        F = get_lcfs_F(cfg["n_chi"], cfg["p_chi"], 2 * cfg["p_chi"],
                       cfg["R_0"], cfg["k_0"], cfg["q_0"], cfg["a_R"])
    # Step 2: Get the Sequence
    ns = (cfg["n_r"], cfg["n_chi"], cfg["n_zeta"])
    ps = (cfg["p_r"], cfg["p_chi"], 0
          if cfg["n_zeta"] == 1 else cfg["p_zeta"])
    q = max(ps)
    types = ("clamped", "periodic",
             "constant" if cfg["n_zeta"] == 1 else "periodic")

    Seq = DeRhamSequence(ns, ps, q, types, F, polar=True)

    # Step 3: get the grids
    _x, _y, (_y1, _y2, _y3), (_x1, _x2, _x3) = get_2d_grids(
        F, zeta=0, nx=64, tol=1e-2)
    _x_1d, _y_1d, (_y1_1d, _y2_1d, _y3_1d), (_x1_1d, _x2_1d,
                                             _x3_1d) = get_1d_grids(F, zeta=0, chi=0, nx=128)

    print("Generating pressure plot...")
    # Plot number one: pressure contour plot
    p_h = DiscreteFunction(p_hat, Seq.Λ0, Seq.E0_0.matrix())
    p_h_xyz = Pushforward(p_h, F, 0)

    _s = jax.vmap(F)(jnp.vstack(
        [jnp.ones(256), jnp.linspace(0, 1, 256), jnp.zeros(256)]).T)

    fig, ax = plt.subplots(figsize=SQUARE_FIG_SIZE)

    # Plot the line first
    ax.plot(_s[:, 0], _s[:, 2], 'k--',
            linewidth=LINE_WIDTH, label="trajectory")

    # Evaluate Z values for contour
    Z = jax.vmap(p_h_xyz)(_x).reshape(_y1.shape)

    # Filled contours for nicer visualization
    cf = ax.contourf(_y1, _y3, Z, levels=20, cmap="plasma", alpha=0.8)

    # Contour lines on top
    # cs = ax.contour(_y1, _y3, Z, levels=10, colors="k", linewidths=LINE_WIDTH)
    # ax.clabel(cs, fmt="%.2f", fontsize=0.5 * LABEL_SIZE)

    # Axes limits and aspect
    ax.set_xlim(jnp.min(_s[:, 0]) - 0.2, jnp.max(_s[:, 0]) + 0.2)
    ax.set_ylim(jnp.min(_s[:, 2]) - 0.2, jnp.max(_s[:, 2]) + 0.2)
    ax.set_aspect('equal')

    # Labels
    ax.set_xlabel("R", fontsize=LABEL_SIZE)
    ax.set_ylabel("z", fontsize=LABEL_SIZE)

    # Optional: grid and title
    ax.grid(True, linestyle="--", alpha=0.5)

    # Colorbar
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label(r"p", fontsize=LABEL_SIZE)
    cbar.ax.tick_params(labelsize=TICK_SIZE)

    # Save
    plt.tight_layout()
    plt.savefig("script_outputs/solovev/" + name + "_pressure.png",
                dpi=400)
    plt.close()

    print("Generating convergence plot...")
