# %%
import os
from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from mrx.DeRhamSequence import DeRhamSequence
from mrx.DifferentialForms import DiscreteFunction

# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)

# Get the absolute path to the script's directory
script_dir = Path(__file__).parent.absolute()
# Create the output directory in the same directory as the script
output_dir = script_dir / 'script_outputs'
os.makedirs(output_dir, exist_ok=True)

# Initialize parameters
ns = (6, 6, 6)  # Number of elements in each direction
ps = (3, 3, 3)  # Polynomial degree in each direction
types = ('clamped', 'clamped', 'clamped')  # Types
bcs = ('dirichlet', 'dirichlet', 'dirichlet')  # Boundary conditions

# Identity mapping for the domain
def F(x): return x

# Create DeRham sequence 
derham = DeRhamSequence(ns, ps, 6, types, bcs, F, polar=False) # Quadrature order 6

# Get boundary operators and mass matrices 
B1 = derham.E1.matrix()  # Boundary operator for 1-forms
B0 = derham.E0.matrix()  # Boundary operator for 0-forms
M1 = derham.assemble_M1()  # Mass matrix for 1-forms
M0 = derham.assemble_M0()  # Mass matrix for 0-forms

D0 = derham.assemble_grad()  # Gradient matrix
O10 = jnp.zeros_like(D0)
O0 = jnp.zeros_like(M0)


C = derham.assemble_curlcurl()  # Double curl matrix 

# %%
Q = jnp.block([[C, D0], [D0.T, O0]])
P = jnp.block([[M1, O10], [O10.T, O0]])

# Eigenvalue of the vector Laplacian
# curl curl E = ω² E
# div E = 0             in Ω
# E x n = 0             on ∂Ω

# mixed formulation:
# sigma = curl u
# curl sigma = f
# div u = 0
# u x n = 0 on ∂Ω
# %%


def generalized_eigh(A, B):
    # Add small value for numerical stability
    L = jnp.linalg.cholesky(B + jnp.eye(B.shape[0]) * 1e-12)
    L_inv = jnp.linalg.inv(L)
    C = L_inv @ A @ L_inv.T
    eigenvalues, eigenvectors_transformed = jnp.linalg.eigh(C)
    eigenvectors_original = L_inv.T @ eigenvectors_transformed
    return eigenvalues, eigenvectors_original


# evs, evecs = sp.linalg.eigh(Q, P)
evs, evecs = generalized_eigh(Q, P)

evs = jnp.real(evs)
evecs = jnp.real(evecs)

finite_indices = evs > 0  # jnp.isfinite(evs)
evs = evs[finite_indices]
evecs = evecs[:, finite_indices]

sort_indices = jnp.argsort(evs)
evs = evs[sort_indices]
evecs = evecs[:, sort_indices]


# %%
def get_true_evs(N_max):
    s_to_multiplicity = defaultdict(int)

    # Iterate over all possible (nx, ny, nz) combinations up to N_max
    # For each combination, determine its contribution to the multiplicity of S.
    for nx in range(N_max + 1):
        for ny in range(N_max + 1):
            for nz in range(N_max + 1):

                num_zeros = 0
                if nx == 0:
                    num_zeros += 1
                if ny == 0:
                    num_zeros += 1
                if nz == 0:
                    num_zeros += 1

                # Degeneracy contribution from this specific (nx,ny,nz)
                current_mode_contribution = 0

                if num_zeros == 3:  # Case (0,0,0)
                    # S=0, E=0 (trivial solution), not a mode of interest for S > 0.
                    continue

                # Case (0,0,k), (0,k,0), or (k,0,0) with k > 0
                elif num_zeros == 2:
                    # These configurations lead to E=0 (trivial solution) because
                    # at least two sine terms in each component of E will be sin(0).
                    continue

                elif num_zeros == 1:  # Case where exactly one index is zero
                    # Valid if the two non-zero indices are >= 1.
                    # Example: (0, ny, nz) requires ny >= 1 and nz >= 1.
                    # These modes have a base E-field degeneracy of 1.
                    if (nx == 0 and ny >= 1 and nz >= 1) or \
                       (ny == 0 and nx >= 1 and nz >= 1) or \
                       (nz == 0 and nx >= 1 and ny >= 1):
                        current_mode_contribution = 1
                    else:
                        # This 'else' handles cases like (0, k, 0) where k was 0,
                        # but those are already handled by num_zeros == 2 or num_zeros == 3.
                        # If num_zeros is truly 1, then two indices must be non-zero.
                        # The conditions ny>=1, nz>=1 (etc.) ensure they are positive non-zero.
                        continue

                # Case where no indices are zero (nx,ny,nz all >= 1)
                elif num_zeros == 0:
                    # Valid if all indices are >= 1 (which they must be if num_zeros is 0 and they are non-negative).
                    # These modes have a base E-field degeneracy of 2.
                    if nx >= 1 and ny >= 1 and nz >= 1:  # Check is slightly redundant but clear
                        current_mode_contribution = 2
                    else:
                        # This path should not be reachable if num_zeros is 0
                        # and indices are non-negative integers.
                        continue

                if current_mode_contribution > 0:
                    s_value = nx**2 + ny**2 + nz**2
                    # s_value must be > 0 here as (0,0,0) is skipped
                    s_to_multiplicity[s_value] += current_mode_contribution

    # Construct the flat list of S values, repeated by multiplicity
    all_s_values_repeated = []

    # Sort by S_value to ensure the final array is sorted
    sorted_unique_s_values = sorted(s_to_multiplicity.keys())

    for s_val in sorted_unique_s_values:
        multiplicity = s_to_multiplicity[s_val]
        all_s_values_repeated.extend([s_val] * multiplicity)

    return np.array(all_s_values_repeated, dtype=int)


# %%
_end = 26
true_evs = get_true_evs(4)[:_end]
# %%
# --- PLOT SETTINGS FOR SLIDES ---
FIG_SIZE = (12, 6)      # Figure size in inches (width, height)
TITLE_SIZE = 20         # Font size for the plot title
LABEL_SIZE = 20         # Font size for x and y axis labels
TICK_SIZE = 16          # Font size for x and y tick labels
LEGEND_SIZE = 16        # Font size for the legend
LINE_WIDTH = 2.5        # Width of the plot lines
# ---------------------------------
end = 64

# %% Figure 1: Energy and Force
fig1, ax1 = plt.subplots(figsize=FIG_SIZE)

color1 = 'purple'
color2 = 'black'
ax1.set_xlabel(r'$k$', fontsize=LABEL_SIZE)
ax1.set_ylabel(r'$\lambda_k / \pi^2$', fontsize=LABEL_SIZE)
ax1.plot(true_evs[:end], label=r'true',
         marker='', ls = ':', markersize=10, color=color2, lw=LINE_WIDTH)
ax1.plot(evs[:end] / (jnp.pi**2), label=r'computed',
         marker='*', ls = '', markersize=10, color=color1, lw=LINE_WIDTH)
ax1.tick_params(axis='y', labelsize=TICK_SIZE)
ax1.tick_params(axis='x', labelsize=TICK_SIZE)
ax1.set_yticks(jnp.unique(true_evs[:end]))
ax1.grid(axis='y', linestyle='--', alpha=0.7)
ax1.legend(fontsize=LEGEND_SIZE) # Use ax1.legend() for clarity

# Now save the figure. The 'tight' layout will be calculated correctly.
fig1.savefig('cube_eigenvalues.pdf', bbox_inches='tight')
# %%
ɛ = 1e-5
nx = 64
_nx = 16

_x1 = jnp.linspace(ɛ, 1-ɛ, nx)
_x2 = jnp.linspace(ɛ, 1-ɛ, nx)
_x3 = jnp.ones(1)/3
_x = jnp.array(jnp.meshgrid(_x1, _x2, _x3))
_x = _x.transpose(1, 2, 3, 0).reshape(nx*nx*1, 3)
__x1 = jnp.linspace(ɛ, 1-ɛ, _nx)
__x2 = jnp.linspace(ɛ, 1-ɛ, _nx)
__x3 = jnp.ones(1)/3
__x = jnp.array(jnp.meshgrid(__x1, __x2, __x3))
__x = __x.transpose(1, 2, 3, 0).reshape(_nx*_nx*1, 3)
__y = jax.vmap(F)(__x)
# %%
u_hat = evecs[:C.shape[0], 0]
u_h = DiscreteFunction(u_hat, derham.Λ1, B1)
_z1 = jax.vmap(u_h)(_x).reshape(nx, nx, 3)
_z1_norm = jnp.linalg.norm(_z1, axis=2)
plt.contourf(_x1, _x2, _z1_norm.reshape(nx, nx), levels=25)
plt.colorbar()
__z1 = jax.vmap(u_h)(__x).reshape(_nx, _nx, 3)
plt.quiver(__x1, __x2, __z1[:, :, 0], __z1[:, :, 1], color='w')

# %%


def plot_eigenvectors_grid(
    evecs,         # Eigenvectors array, shape (num_dofs, num_eigenvectors)
    M1,            # Matrix used to determine split point for DOFs
    Λ1, E1,        # Parameters for DiscreteFunction
    # The 'F' map for Pushforward
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

    # Determine grid size
    nrows = int(num_to_plot**0.5)
    ncols = int(num_to_plot**0.5)

    fig, axes = plt.subplots(nrows, ncols, figsize=(
        ncols * 3, nrows * 3))
    axes = axes.flatten()

    for i in range(num_to_plot):
        ax = axes[i]

        # Extract and prepare the degrees of freedom for the i-th eigenvector
        ev_dof = jnp.split(evecs[:, i], (M1.shape[0],))[0]
        u_h = DiscreteFunction(ev_dof, Λ1, E1)

        # Vector norm
        _z1_vector_field = jax.vmap(u_h)(map_input_x)
        _z1_reshaped = _z1_vector_field.reshape(nx_grid, nx_grid, 3)
        _z1_norm = jnp.linalg.norm(_z1_reshaped, axis=2)
        ax.contourf(y1_coords, y2_coords, _z1_norm, 
                    levels=25, cmap='plasma')

        # No axes
        ax.set_axis_off()
        ax.set_aspect('equal', adjustable='box')  # Maintain aspect ratio

    # Hide any unused subplots if num_to_plot < nrows*ncols
    for j in range(num_to_plot, nrows * ncols):
        fig.delaxes(axes[j])

    # Tight layout
    plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)  # Adjust padding as needed
    plt.show()

    return fig


# %%
fig = plot_eigenvectors_grid(
    evecs, M1, derham.Λ1, B1, F, _x, _x1, _x2, nx, num_to_plot=25
)
# %%
fig.savefig('cube_eigenvectors.pdf', bbox_inches='tight')