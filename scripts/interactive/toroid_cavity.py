# %%
# TODO update this after the refactor
##
# Standing TM waves in a toroidal cavity
##

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import scipy as sp

from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction, Pushforward
from mrx.mappings import toroid_map

# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)

# order of the splines and number of elements in each direction
p = 3
n = 8

# Initialize parameters
ns = (n, n, 1)  # Number of elements in each direction
ps = (p, p, 0)  # Polynomial degree in each direction
types = ('clamped', 'periodic', 'constant')  # Types
bcs = ('dirichlet', 'periodic', 'constant')  # Boundary conditions

# Domain parameters, a = minor radius, R0 = major radius
a = 1
R0 = 2.1
π = jnp.pi
F = toroid_map(epsilon=a, R0=R0)

# Create DeRham sequence
Seq = DeRhamSequence(ns, ps, 2*p, types, F, polar=True)
Seq.evaluate_1d()

# Get extraction operators and mass matrices
E0, E1, E2, E3 = [Seq.E0, Seq.E1, Seq.E2, Seq.E3]
Seq.assemble_M0()
Seq.assemble_M1()
Seq.assemble_M2()
Seq.assemble_M3()
Seq.assemble_d0()
M0, M1, M2, M3 = [Seq.M0, Seq.M1, Seq.M2, Seq.M3]
D0 = Seq.strong_grad  # Gradient operator
O10 = jnp.zeros_like(D0)
Seq.assemble_dd1()
C = Seq.M1 @ (Seq.dd1 + Seq.strong_grad @ Seq.weak_div)  # Double curl matrix

# %%
evs, evecs = sp.linalg.eig(C, M1)
evs = jnp.real(evs)
evecs = jnp.real(evecs)

finite_indices = jnp.isfinite(evs)
evs = evs[finite_indices]
evecs = evecs[:, finite_indices]

sort_indices = jnp.argsort(evs)
evs = evs[sort_indices]
evecs = evecs[:, sort_indices]

# %%
# --- PLOT SETTINGS FOR SLIDES ---
FIG_SIZE = (12, 6)      # Figure size in inches (width, height)
TITLE_SIZE = 20         # Font size for the plot title
LABEL_SIZE = 20         # Font size for x and y axis labels
TICK_SIZE = 16          # Font size for x and y tick labels
LEGEND_SIZE = 16        # Font size for the legend
LINE_WIDTH = 2.5        # Width of the plot lines
end = 40

# %%
fig1, ax1 = plt.subplots(figsize=FIG_SIZE)

color1 = 'purple'
color2 = 'black'
ax1.set_xlabel(r'$k$', fontsize=LABEL_SIZE)
ax1.set_ylabel(r'$\lambda_k / \pi^2$', fontsize=LABEL_SIZE)
ax1.plot(evs[:end], label=r'computed',
         marker='*', ls='', markersize=10, color=color1, lw=LINE_WIDTH)
ax1.tick_params(axis='y', labelsize=TICK_SIZE)
ax1.tick_params(axis='x', labelsize=TICK_SIZE)
# ax1.set_yticks(jnp.unique(true_evs[:end]))
ax1.grid(axis='y', linestyle='--', alpha=0.7)
ax1.legend(fontsize=LEGEND_SIZE)  # Use ax1.legend() for clarity
fig1.savefig('toroid_eigenvalues.pdf', bbox_inches='tight')

# %%
# Plot the first 9 eigenvectors and make a meshgrid in the physical domain
ɛ = 1e-5
nx = 64
_x1 = jnp.linspace(ɛ, 1-ɛ, nx)
_x2 = jnp.linspace(ɛ, 1-ɛ, nx)
_x3 = jnp.zeros(1)
_x = jnp.array(jnp.meshgrid(_x1, _x2, _x3))
_x = _x.transpose(1, 2, 3, 0).reshape(nx*nx*1, 3)
_y = jax.vmap(F)(_x)
_y1 = _y[:, 0].reshape(nx, nx)
_y2 = _y[:, 1].reshape(nx, nx)
_y3 = _y[:, 2].reshape(nx, nx)
_nx = 16
__x1 = jnp.linspace(ɛ, 1-ɛ, _nx)
__x2 = jnp.linspace(ɛ, 1-ɛ, _nx)
__x3 = jnp.zeros(1)
__x = jnp.array(jnp.meshgrid(__x1, __x2, __x3))
__x = __x.transpose(1, 2, 3, 0).reshape(_nx*_nx*1, 3)
__y = jax.vmap(F)(__x)
__y1 = __y[:, 0].reshape(_nx, _nx)
__y2 = __y[:, 1].reshape(_nx, _nx)
__y3 = __y[:, 2].reshape(_nx, _nx)

# %%
def plot_eigenvectors_grid(
    evecs,         # Eigenvectors array, shape (num_dofs, num_eigenvectors)
    M1,            # Matrix used to determine split point for DOFs
    Λ1, E1,        # Parameters for DiscreteFunction
    # The 'F' map for Pushforward (renamed from F to avoid confusion with a potential figure object)
    F_map,
    map_input_x,   # Input points for the pushforward map (_x)
    y1_coords,     # y1 coordinates for contourf (_y1)
    y2_coords,     # y2 coordinates for contourf (_y2)
    nx_grid,       # Grid dimension for reshaping (nx)
    num_to_plot=9  # Number of eigenvectors to plot (0 to num_to_plot-1)
):
    """
    Plots the norm of the pushforward of the first 'num_to_plot' eigenvectors
    on a grid. Assumes num_to_plot <= 9 for a 3x3 grid.

    Args:
        evecs: JAX array of eigenvectors (columns are eigenvectors).
        M1: Object with a .shape[0] attribute for splitting DOFs.
        Λ1, E1: Arguments for DiscreteFunction.
        F_map: The geometric map for Pushforward.
        map_input_x: Input coordinate array for jax.vmap(F_u).
        y1_coords, y2_coords: Meshgrid outputs for plt.contourf.
        nx_grid: Integer dimension for reshaping the output norm.
        num_to_plot: Number of eigenvectors to plot (default is 9 for a 3x3 grid).

    Returns:
        fig: Figure object
    """
    if num_to_plot > evecs.shape[1]:
        print(
            f"Warning: Requested {num_to_plot} eigenvectors, but only {evecs.shape[1]} are available. Plotting all available.")
        num_to_plot = evecs.shape[1]

    nrows = int(num_to_plot**0.5)
    ncols = int(num_to_plot**0.5)

    fig, axes = plt.subplots(nrows, ncols, figsize=(
        ncols * 3, nrows * 3))  # Adjust figsize as needed
    axes = axes.flatten()  # Flatten to easily iterate

    for i in range(num_to_plot):
        ax = axes[i]
        ev_dof = jnp.split(evecs[:, i], (M1.shape[0],))[0]
        u_h = DiscreteFunction(ev_dof, Λ1, E1)
        F_u = Pushforward(u_h, F_map, 1)
        _z1_vector_field = jax.vmap(F_u)(map_input_x)
        _z1_reshaped = _z1_vector_field.reshape(nx_grid, nx_grid, 3)
        _z1_norm = jnp.linalg.norm(_z1_reshaped, axis=2)
        ax.contourf(y1_coords, y2_coords, _z1_norm, levels=25, cmap='plasma')
        ax.set_axis_off()
        ax.set_aspect('equal', adjustable='box')
    for j in range(num_to_plot, nrows * ncols):
        fig.delaxes(axes[j])

    plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
    return fig


fig = plot_eigenvectors_grid(
    evecs, M1, Seq.Λ1, E1, F, _x, _y1, _y3, nx, num_to_plot=25
)
fig.savefig('toroid_eigenvectors.pdf', bbox_inches='tight')
# %%
idx = 3
# The eigenvector from the double curl operator only contains velocity components (1-form)
u_hat, p_hat = jnp.split(evecs[:, idx], (M1.shape[0],))
u_h = DiscreteFunction(u_hat, Seq.Λ1, E1)
F_u = Pushforward(u_h, F, 1)

_z1 = jax.vmap(F_u)(_x).reshape(nx, nx, 3)
_z1_norm = jnp.linalg.norm(_z1, axis=-1)
plt.contourf(_y1, _y3, _z1_norm.reshape(nx, nx))
plt.colorbar()
__z1 = jax.vmap(F_u)(__x).reshape(_nx, _nx, 3)
plt.quiver(__y1, __y3, __z1[:, :, 0], __z1[:, :, 2], color='w', scale=10)
plt.xlabel('X')
plt.ylabel('Z')
plt.show()
# %%
# There is no pressure component in this formulation
# p_h = DiscreteFunction(p_hat, Seq.Λ0, E0)
# F_p = Pushforward(p_h, F, 0)

# _z1 = jax.vmap(F_p)(_x).reshape(nx, nx)
# plt.contourf(_y1, _y3, _z1)
# plt.colorbar()
# plt.xlabel('X')
# plt.ylabel('Z')
# plt.title(f'Pressure field for eigenvalue {evs[idx]:.4f}')
# plt.show()