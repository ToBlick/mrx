"""
Plotting utilities for finite element analysis results.

This module provides functions for creating visualizations of convergence plots
and other analysis results using Plotly.
"""

import jax
import jax.numpy as jnp
import matplotlib as plt
import plotly.colors as pc
import plotly.graph_objects as go

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


def get_2d_grids(F, zeta=0, nx=64, tol=1e-6):
    tol = 1e-6
    nx = 64
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
    nx = 64
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
