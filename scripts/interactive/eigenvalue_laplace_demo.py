# %%
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# import numpy as np
import scipy as sp

from mrx.DifferentialForms import DifferentialForm, DiscreteFunction
from mrx.LazyMatrices import LazyDerivativeMatrix, LazyMassMatrix
from mrx.Projectors import Projector
from mrx.Quadrature import QuadratureRule
from mrx.Utils import grad, jacobian_determinant, l2_product

# import time
# from functools import partial


# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)

# Create output directory for figures
os.makedirs('script_outputs', exist_ok=True)

n = 16
p = 3
"""
Compute error for mixed Poisson problem.
Args:
    n: Number of elements in each direction
    p: Polynomial degree
Returns:
    float: Relative L2 error of the solution
"""
# Set up finite element spaces
ns = (n, n, 1)
ps = (p, p, 0)
types = ('clamped', 'clamped', 'constant')
# Define exact solution and source term


def F(x):
    return x


def u(x):
    """Exact solution of the Poisson problem."""
    r, χ, z = x
    return jnp.ones(1) * jnp.sin(2 * jnp.pi * r) * jnp.sin(2 * jnp.pi * χ)


def f(x):
    """Source term of the Poisson problem."""
    return 2 * (2*jnp.pi)**2 * u(x)


# Set up differential forms and quadrature
Λ0 = DifferentialForm(0, ns, ps, types)
Λ2 = DifferentialForm(2, ns, ps, types)
Λ3 = DifferentialForm(3, ns, ps, types)
Q = QuadratureRule(Λ0, 3*p)
# Set up operators
D = LazyDerivativeMatrix(Λ2, Λ3, Q).M
M2 = LazyMassMatrix(Λ2, Q).M
M0 = LazyMassMatrix(Λ0, Q).M
M3 = LazyMassMatrix(Λ3, Q).M
# Solve the system
K = D @ jnp.linalg.solve(M2, D.T)
P3 = Projector(Λ3, Q)
u_hat = jnp.linalg.solve(K, P3(f))
u_h = DiscreteFunction(u_hat, Λ3)
# Compute error using Λ3 quadrature
def err(x): return u(x) - u_h(x)


error = (l2_product(err, err, Q) / l2_product(u, u, Q))**0.5
error
# %%
Λv = DifferentialForm(-1, ns, ps, types)

# Derivative matrix


def lazy_gradient_matrix(Λ1, Λ2):

    DF = jax.jacfwd(F)

    def _Λ0(x, i):
        return DF(x).T @ grad(lambda y: Λ1(y, i))(x)

    def _Λv(x, i):
        return DF(x).T @ Λ2(x, i)
    Λ0_ijk = jax.vmap(jax.vmap(_Λ0, (0, None)), (None, 0))(
        Q.x, jnp.arange(Λ1.n))  # n0 x n_q x d
    Λ1_ijk = jax.vmap(jax.vmap(_Λv, (0, None)), (None, 0))(
        Q.x, jnp.arange(Λ2.n))  # n1 x n_q x d
    Jj = jax.vmap(jacobian_determinant(F))(Q.x)  # n_q x 1
    wj = Q.w  # n_q
    return jnp.einsum("ijk,ljk,j,j->li", Λ0_ijk, Λ1_ijk, Jj, wj)


# %%
Dv = lazy_gradient_matrix(Λ0, Λv)
# %%
Mv = LazyMassMatrix(Λv, Q).M
# %%
Kv = Dv.T @ jnp.linalg.solve(Mv, Dv)
P0 = Projector(Λ0, Q)
u_hat_v = jnp.linalg.solve(Kv.at[-1, :].set(1.0), P0(f).at[-1].set(0))
# %%
u_h_v = DiscreteFunction(u_hat_v, Λ0)
def err(x): return u(x) - u_h_v(x)


error = (l2_product(err, err, Q) / l2_product(u, u, Q))**0.5
error
# %%
ɛ = 0
nx = 64
_x1 = jnp.linspace(ɛ, 1 - ɛ, nx)
_x2 = jnp.linspace(0, 1, nx)
_x3 = jnp.zeros(1) / 2
_x = jnp.array(jnp.meshgrid(_x1, _x2, _x3))
_x = _x.transpose(1, 2, 3, 0).reshape(nx * nx * 1, 3)

# # %%
# plt.contourf(_x1, _x2, jax.vmap(u_h)(_x)[:, 0].reshape(nx, nx), levels=20)
# # %%
# plt.contourf(_x1, _x2, jax.vmap(u_h_v)(_x)[:, 0].reshape(nx, nx), levels=20)
# %%
y = jax.vmap(u_h)(_x)
plt.contourf(_x1, _x2, jax.vmap(u_h)(_x)[:, 0].reshape(nx, nx), levels=20)
# %%
plt.contourf(_x1, _x2, jax.vmap(u_h_v)(_x)[:, 0].reshape(nx, nx), levels=20)
# %%
evs, evecs = sp.linalg.eig(Kv, M0)
evs = jnp.real(evs)
evecs = jnp.real(evecs)
sort_indices = jnp.argsort(evs)
evs = evs[sort_indices]
evecs = evecs[:, sort_indices]
# %%
evs_mixed, evecs_mixed = sp.linalg.eig(K, M3)
evs_mixed = jnp.real(evs_mixed)
evecs_mixed = jnp.real(evecs_mixed)
sort_indices = jnp.argsort(evs_mixed)
evs_mixed = evs_mixed[sort_indices]
evecs_mixed = evecs_mixed[:, sort_indices]
# %%
evs_true = jnp.array([k**2 + m**2 for k in range(1, n+1)
                     for m in range(1, n+1)])
sort_indices = jnp.argsort(evs_true)
evs_true = evs_true[sort_indices]
# %%
# --- PLOT SETTINGS FOR SLIDES ---
FIG_SIZE = (12, 6)      # Figure size in inches (width, height)
TITLE_SIZE = 20         # Font size for the plot title
LABEL_SIZE = 20         # Font size for x and y axis labels
TICK_SIZE = 16          # Font size for x and y tick labels
LEGEND_SIZE = 16        # Font size for the legend
LINE_WIDTH = 2.5        # Width of the plot lines
# ---------------------------------


# %% Figure 1: Energy and Force
fig1, ax1 = plt.subplots(figsize=FIG_SIZE)
end = 20
# Plot Energy on the left y-axis (ax1)
color1 = 'purple'
color2 = 'black'
ax1.set_xlabel(r'$k$', fontsize=LABEL_SIZE)
ax1.set_ylabel(r'$\lambda_k / \pi^2$', fontsize=LABEL_SIZE)
ax1.plot(evs_true[:end], label=r'exact', markersize=10,
         color='grey', linestyle="-", lw=LINE_WIDTH)
ax1.plot(evs[:end] / (jnp.pi**2), label=r'$\phi^0, \phi^v$',
         marker='*', markersize=10, color=color2, lw=LINE_WIDTH)
ax1.plot(evs_mixed[:end] / (jnp.pi**2), label=r'$\phi^3, \phi^2$',
         marker='^', markersize=10, color=color1, linestyle="--", lw=LINE_WIDTH)
ax1.tick_params(axis='y', labelsize=TICK_SIZE)
ax1.tick_params(axis='x', labelsize=TICK_SIZE)  # Set x-tick size
ax1.set_yticks(evs_true[:end])
ax1.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(fontsize=LEGEND_SIZE)
fig1.savefig('two_d_poisson_eigenvalues.pdf', bbox_inches='tight')
# %%


def plot_eigenvectors_grid(
    evecs,         # Eigenvectors array, shape (num_dofs, num_eigenvectors)
    Λ1,        # Parameters for DiscreteFunction
    map_input_x,   # Input points for the pushforward map (_x)
    y1_coords,     # y1 coordinates for contourf (_y1)
    y2_coords,     # y2 coordinates for contourf (_y2)
    nx_grid,       # Grid dimension for reshaping (nx)
    num_to_plot=9  # Number of eigenvectors to plot (0 to num_to_plot-1)
):

    if num_to_plot > evecs.shape[1]:
        print(
            f"Warning: Requested {num_to_plot} eigenvectors, but only {evecs.shape[1]} are available. Plotting all available.")
        num_to_plot = evecs.shape[1]

    # Determine grid size
    nrows = int(num_to_plot**0.5)
    ncols = int(num_to_plot**0.5)

    fig, axes = plt.subplots(nrows, ncols, figsize=(
        ncols * 3, nrows * 3))
    axes = axes.flatten()

    for i in range(num_to_plot):
        ax = axes[i]

        # Extract and prepare the degrees of freedom for the i-th eigenvector
        ev_dof = evecs[:, i]
        u_h = DiscreteFunction(ev_dof, Λ1)

        # Vector norm
        _z1_vector_field = jax.vmap(u_h)(map_input_x)
        _z1_reshaped = _z1_vector_field.reshape(nx_grid, nx_grid)
        ax.contourf(y1_coords, y2_coords, _z1_reshaped)

        # No axes
        ax.set_axis_off()
        ax.set_aspect('equal', adjustable='box')  # Maintain aspect ratio

    # Hide any unused subplots if num_to_plot < nrows*ncols
    for j in range(num_to_plot, nrows * ncols):
        fig.delaxes(axes[j])

    # Tight layout
    plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)  # Adjust padding as needed

    return fig


# %%
plot_eigenvectors_grid(
    evecs, Λ0,
    _x, _x1, _x2, nx,
    num_to_plot=25
)

# %%
plot_eigenvectors_grid(
    evecs_mixed, Λ3,
    _x, _x1, _x2, nx,
    num_to_plot=25
)
25  # %%
