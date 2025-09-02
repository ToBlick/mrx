# %%

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import scipy as sp

from mrx.DeRhamSequence import DeRhamSequence
from mrx.DifferentialForms import DiscreteFunction, Pushforward

# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)

# Initialize parameters
ns = (15, 15, 1)  # Number of elements in each direction
ps = (3, 3, 0)  # Polynomial degree in each direction
types = ('clamped', 'periodic', 'constant') # Types
bcs = ('dirichlet', 'periodic', 'constant')  # Boundary conditions

a = 1
R0 = 2.1
π = jnp.pi


def _X(r, χ):
    return jnp.ones(1) * (R0 + a * r * jnp.cos(2 * π * χ))


def _Y(r, χ):
    return jnp.ones(1) * (R0 + a * r * jnp.cos(2 * π * χ))


def _Z(r, χ):
    return jnp.ones(1) * a * r * jnp.sin(2 * π * χ)


def F(x):
    """Polar coordinate mapping function."""
    r, χ, z = x
    return jnp.ravel(jnp.array([_X(r, χ) * jnp.cos(2 * π * z),
                                -_Y(r, χ) * jnp.sin(2 * π * z),
                                _Z(r, χ)]))


# Create DeRham sequence 
derham = DeRhamSequence(ns, ps, 8, types, bcs, F, polar=True)

# Get extraction operators and mass matrices 
E0, E1, E2, E3 = [derham.E0.matrix(), derham.E1.matrix(), derham.E2.matrix(), derham.E3.matrix()]
M0, M1, M2, M3 = [derham.assemble_M0(), derham.assemble_M1(), 
                    derham.assemble_M2(), derham.assemble_M3()]


D0 = derham.assemble_grad()  # Gradient matrix 

O10 = jnp.zeros_like(D0)
O0 = jnp.zeros((D0.shape[1], D0.shape[1]))

C = derham.assemble_curlcurl()  # Double curl matrix 

_Q = jnp.block([[C, D0], [D0.T, O0]])
_P = jnp.block([[M1, O10], [O10.T, O0]])

# %%
evs, evecs = sp.linalg.eig(_Q, _P)
evs = jnp.real(evs)
evecs = jnp.real(evecs)

finite_indices = jnp.isfinite(evs)
evs = evs[finite_indices]
evecs = evecs[:, finite_indices]

sort_indices = jnp.argsort(evs)
evs = evs[sort_indices]
evecs = evecs[:, sort_indices]
# %%

# %%
# --- PLOT SETTINGS FOR SLIDES ---
FIG_SIZE = (12, 6)      # Figure size in inches (width, height)
TITLE_SIZE = 20         # Font size for the plot title
LABEL_SIZE = 20         # Font size for x and y axis labels
TICK_SIZE = 16          # Font size for x and y tick labels
LEGEND_SIZE = 16        # Font size for the legend
LINE_WIDTH = 2.5        # Width of the plot lines
# ---------------------------------
end = 40

# %% Figure 1: Energy and Force
fig1, ax1 = plt.subplots(figsize=FIG_SIZE)

color1 = 'purple'
color2 = 'black'
ax1.set_xlabel(r'$k$', fontsize=LABEL_SIZE)
ax1.set_ylabel(r'$\lambda_k / \pi^2$', fontsize=LABEL_SIZE)
ax1.plot(evs[:end], label=r'computed',
         marker='*', ls = '', markersize=10, color=color1, lw=LINE_WIDTH)
ax1.tick_params(axis='y', labelsize=TICK_SIZE)
ax1.tick_params(axis='x', labelsize=TICK_SIZE)
# ax1.set_yticks(jnp.unique(true_evs[:end]))
ax1.grid(axis='y', linestyle='--', alpha=0.7)
ax1.legend(fontsize=LEGEND_SIZE) # Use ax1.legend() for clarity

# %%
fig1.savefig('toroid_eigenvalues.pdf', bbox_inches='tight')
# %%
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
idx = 3
u_hat, p_hat = jnp.split(evecs[:, idx], (M1.shape[0],))
u_h = DiscreteFunction(u_hat, derham.Λ1, E1)
F_u = Pushforward(u_h, F, 1)

_z1 = jax.vmap(F_u)(_x).reshape(nx, nx, 3)
_z1_norm = jnp.linalg.norm(_z1, axis=2)
plt.contourf(_y1, _y3, _z1_norm.reshape(nx, nx))
plt.colorbar()
__z1 = jax.vmap(F_u)(__x).reshape(_nx, _nx, 3)
plt.quiver(
    _y1,
    _y3,
    __z1[:, :, 0],
    __z1[:, :, 2],
    color='w',
    scale=10)
plt.xlabel('X')
plt.ylabel('Z')
# %%
p_h = DiscreteFunction(p_hat, derham.Λ0, E0)
F_p = Pushforward(p_h, F, 0)

_z1 = jax.vmap(F_p)(_x).reshape(nx, nx)
plt.contourf(_y1, _y3, _z1)
plt.colorbar()
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


# %%
# Plot the first 9 eigenvectors
fig = plot_eigenvectors_grid(
    evecs, M1, derham.Λ1, E1, F, _x, _y1, _y3, nx, num_to_plot=25
)
# %%
fig.savefig('toroid_eigenvectors.pdf', bbox_inches='tight')
# %%
