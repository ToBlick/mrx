# %%
import os
import time
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from mrx.DeRhamSequence import DeRhamSequence
from mrx.DifferentialForms import DiscreteFunction
from mrx.Plotting import get_2d_grids, get_3d_grids
from mrx.Utils import l2_product

# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)

# Create output directory for figures
os.makedirs("script_outputs", exist_ok=True)

p = 2
n = 4

# Set up finite element spaces
q = 2*p
ns = (n, n, n)
ps = (p, p, p)
types = ("clamped", "periodic", "periodic")  # Types
# Domain parameters
π = jnp.pi
ɛ = 1/3
n_turns = 3  # Number of helix turns
h = 1/4      # radius of helix


def _X(r, χ):
    return jnp.ones(1) * (1 + ɛ * r * jnp.cos(2 * π * χ))


def _Y(r, χ):
    return jnp.ones(1) * (1 + ɛ * r * jnp.cos(2 * π * χ))


# For the current construction to work, the tangent line must be parallel to Y at zeta = 0.
# Helical curve
def X(ζ):
    return jnp.array([
        (1 + h * jnp.sin(2 * π * n_turns * ζ)) * jnp.sin(2 * π * ζ),
        (1 + h * jnp.sin(2 * π * n_turns * ζ)) * jnp.cos(2 * π * ζ),
        h * jnp.cos(2 * π * n_turns * ζ)
    ])


def get_frame(ζ):
    dX = jax.jacrev(X)(ζ)
    τ = dX / jnp.linalg.norm(dX)  # Tangent vector

    e = jnp.array([0.0, 0.0, 1.0])
    ν1 = (e - jnp.dot(e, τ) * τ)
    ν1 = ν1 / jnp.linalg.norm(ν1)  # First normal vector
    ν2 = jnp.cross(τ, ν1)         # Second normal vector
    return τ, ν1, ν2


def F(x):
    """Helical coordinate mapping function."""
    r, θ, ζ = x
    τ, ν1, ν2 = get_frame(ζ)
    return X(ζ) + ɛ * r * jnp.cos(2 * π * θ) * ν1 + ɛ * r * jnp.sin(2 * π * θ) * ν2

# %%


def f(x):
    return jnp.ones(1)


# Create DeRham sequence
Seq = DeRhamSequence(ns, ps, q, types, F, polar=True)
# Get stiffness matrix and projector

M0 = Seq.assemble_M0_0()
M1 = Seq.assemble_M1_0()
M2 = Seq.assemble_M2_0()
M3 = Seq.assemble_M3_0()
###
# Operators
###
grad = jnp.linalg.solve(M1, Seq.assemble_grad_0())
curl = jnp.linalg.solve(M2, Seq.assemble_curl_0())
dvg = jnp.linalg.solve(M3, Seq.assemble_dvg_0())
weak_grad = -jnp.linalg.solve(M2, Seq.assemble_dvg_0().T)
weak_curl = jnp.linalg.solve(M1, Seq.assemble_curl_0().T)
weak_dvg = -jnp.linalg.solve(M0, Seq.assemble_grad_0().T)
laplace_0 = Seq.assemble_gradgrad_0()                         # dim ker = 0
laplace_1 = Seq.assemble_curlcurl_0() - M1 @ grad @ weak_dvg  # dim ker = 0 (no voids)
laplace_2 = M2 @ curl @ weak_curl + \
    Seq.assemble_divdiv_0()  # dim ker = 1 (one tunnel)
laplace_3 = - M3 @ dvg @ weak_grad  # dim ker = 1 (constants)

P0 = Seq.P0_0  # Projector for 0-forms


# %%
print("Smallest 3 eigenvalues of Laplace operators:")
print("0-forms (dim ker = 0):", jnp.linalg.eigvalsh(laplace_0)[:3])
print("1-forms (dim ker = 0):", jnp.linalg.eigvalsh(laplace_1)[:3])
print("2-forms (dim ker = 1):", jnp.linalg.eigvalsh(laplace_2)[:3])
print("3-forms (dim ker = 1):", jnp.linalg.eigvalsh(laplace_3)[:3])
# %%
# Solve the system
u_hat = jnp.linalg.solve(laplace_0, P0(f))
u_h = DiscreteFunction(u_hat, Seq.Λ0, Seq.E0_0.matrix())

# Plot solution
grids = [get_2d_grids(F, cut_axis=2, cut_value=v, nx=32)
         for v in jnp.linspace(0, 1, 32, endpoint=False)]
# grids[i] = _x, _y, (_y1, _y2, _y3), (_x1, _x2, _x3)
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


def plot_crossections(f, grids):
    fig = plt.figure()
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


# %%
fig, ax = plot_crossections(u_h, grids)
# %%
cuts = jnp.linspace(0, 1, 8)
grids_pol = [get_2d_grids(F, cut_axis=2, cut_value=v, nx=32) for v in cuts]
grid_surface = get_2d_grids(F, cut_axis=0, cut_value=1.0,
                            nx=32, z_min=cuts[0], z_max=cuts[-1], invert_z=True)


def plot_torus(f, grids_pol, grid_surface):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    X = grid_surface[2][0]
    Y = grid_surface[2][1]
    Z = grid_surface[2][2]
    vals = jax.vmap(f)(grid_surface[0]).reshape(X.shape)
    colors = plt.cm.plasma(vals)
    ax.plot_surface(X, Y, Z, facecolors=colors, rstride=1, cstride=1, shade=False,
                    alpha=0.0, linewidth=0.1,)

    for grid in grids_pol:
        X = grid[2][0]
        Y = grid[2][1]
        Z = grid[2][2]
        vals = jax.vmap(f)(grid[0]).reshape(X.shape)
        colors = plt.cm.plasma((vals - vals.min()) / (vals.max() - vals.min()))
        ax.plot_surface(X, Y, Z, facecolors=colors, rstride=1,
                        cstride=1, shade=False, zsort='min')

    set_axes_equal(ax)
    plt.tight_layout()
    ax.view_init(elev=30, azim=190)
    return fig, ax


fig, ax = plot_torus(u_h, grids_pol, grid_surface)
# %%
