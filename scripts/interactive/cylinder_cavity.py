# %%

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction, Pushforward

# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)

# Initialize parameters
ns = (15, 15, 1)  # Number of elements in each direction
ps = (3, 3, 0)  # Polynomial degree in each direction
types = ('clamped', 'periodic', 'constant')  # Types
bcs = ('dirichlet', 'periodic', 'periodic')  # Boundary conditions

a = 1
h = 1


def _R(r, χ):
    return jnp.ones(1) * (a * r * jnp.cos(2 * jnp.pi * χ))


def _Y(r, χ):
    return jnp.ones(1) * (a * r * jnp.sin(2 * jnp.pi * χ))


def F(x):
    r, χ, z = x
    return jnp.ravel(jnp.array([_R(r, χ), _Y(r, χ), h * jnp.ones(1) * z]))


# Create DeRham sequence 
derham = DeRhamSequence(ns, ps, 8, types, bcs, F, polar=True)

# Get extraction operators and mass matrices 
E0, E1, E2, E3 = [derham.E0.matrix(), derham.E1.matrix(), derham.E2.matrix(), derham.E3.matrix()]
M1 = derham.assemble_M1()  # Mass matrix for 1-forms
M0 = derham.assemble_M0()  # Mass matrix for 0-forms


D0 = derham.assemble_grad()  # Gradient matrix 
O10 = jnp.zeros_like(D0)
O0 = jnp.zeros((D0.shape[1], D0.shape[1]))


C = derham.assemble_curlcurl()  # Double curl matrix 

Q = jnp.block([[C, D0], [D0.T, O0]])
P = jnp.block([[M1, O10], [O10.T, O0]])

# %%
evs, evecs = sp.linalg.eig(Q, P)
evs = jnp.real(evs)
evecs = jnp.real(evecs)

finite_indices = jnp.isfinite(evs)
evs = evs[finite_indices]
evecs = evecs[:, finite_indices]

sort_indices = jnp.argsort(evs)
evs = evs[sort_indices]
evecs = evecs[:, sort_indices]

# %%


def calculate_cylindrical_periodic_TE_TM_eigenvalues(
    # List of azimuthal mode indices n (e.g., [0, 1, 2], n >= 0)
    n_values,
    # List of radial mode indices m (e.g., [1, 2, 3], m >= 1)
    m_values,
    # List of axial periodic indices k (e.g., [0, 1, 2], k >= 0)
    k_axial_values,
    radius_a,             # Radius of the cylinder
    period_h              # Periodicity length in z-direction
):
    """
    Calculates the eigenvalues (k^2) for both TE_nmk and TM_nmk modes
    in a cylindrical geometry with periodic boundary conditions in z.

    For TE modes: k^2 = (j'_nm / radius_a)^2 + (2 * k_axial * pi / period_h)^2,
                  where j'_nm is the m-th positive root of J'_n(x) = 0.
    For TM modes: k^2 = (j_nm / radius_a)^2 + (2 * k_axial * pi / period_h)^2,
                  where j_nm is the m-th positive root of J_n(x) = 0.
    """
    if not (radius_a > 0 and period_h > 0):
        raise ValueError("Radius 'a' and period 'h' must be positive.")

    all_eigenvalues_repeated = []

    if not m_values:
        return np.array([])  # No m values, no eigenvalues
    max_m_needed = 0
    if m_values:  # m_values could be empty if user provides empty list
        max_m_needed = max(m_values)
        if max_m_needed <= 0:
            raise ValueError(
                "m_values (radial mode indices) must contain positive integers (m >= 1).")
    else:  # No m_values means no eigenvalues.
        return np.array([])

    if any(k < 0 for k in k_axial_values):
        raise ValueError(
            "k_axial_values (axial periodic indices) must be non-negative integers (k >= 0).")
    if any(n < 0 for n in n_values):
        raise ValueError(
            "n_values (azimuthal mode indices) must be non-negative integers (n >= 0).")

    # --- Calculate TE mode eigenvalues ---
    for n_order in n_values:
        try:
            bessel_prime_zeros_for_n = sp.special.jnp_zeros(
                n_order, max_m_needed)
        except ValueError as e:
            print(
                f"Warning (TE): Error getting Bessel derivative zeros for n={n_order}: {e}. Skipping this n for TE.")
            continue

        if len(bessel_prime_zeros_for_n) < max_m_needed and max_m_needed > 0:
            # This can happen if jnp_zeros doesn't find enough roots, e.g. n_order is very high.
            # Process only the roots found.
            pass  # No warning here, will be handled by m_index check

        for m_index_1_based in m_values:
            if m_index_1_based - 1 < len(bessel_prime_zeros_for_n):
                jprime_nm_root = bessel_prime_zeros_for_n[m_index_1_based - 1]
            else:
                continue  # Not enough roots for this m_index for TE

            for k_axial_index in k_axial_values:
                term1_radial_TE = (jprime_nm_root / radius_a)**2
                term2_axial = (2 * k_axial_index * np.pi / period_h)**2
                eigenvalue_TE = term1_radial_TE + term2_axial

                azimuthal_multiplicity = 1 if n_order == 0 else 2
                axial_multiplicity = 1 if k_axial_index == 0 else 2
                total_multiplicity = azimuthal_multiplicity * axial_multiplicity

                all_eigenvalues_repeated.extend(
                    [eigenvalue_TE] * total_multiplicity)

    # --- Calculate TM mode eigenvalues ---
    for n_order in n_values:
        try:
            # For TM modes, we need zeros of J_n(x)
            bessel_zeros_for_n = sp.special.jn_zeros(n_order, max_m_needed)
        except ValueError as e:
            print(
                f"Warning (TM): Error getting Bessel zeros for n={n_order}: {e}. Skipping this n for TM.")
            continue

        if len(bessel_zeros_for_n) < max_m_needed and max_m_needed > 0:
            # Process only the roots found.
            pass  # No warning here, will be handled by m_index check

        for m_index_1_based in m_values:
            if m_index_1_based - 1 < len(bessel_zeros_for_n):
                j_nm_root = bessel_zeros_for_n[m_index_1_based - 1]
            else:
                continue  # Not enough roots for this m_index for TM

            # For TM_nmk modes, if n=0, k_axial_index=0, the mode TM_0m0 is non-trivial.
            # If n>0 and k_axial_index=0 (TM_nm0, n>0), E_z is proportional to J_n(k_c r).
            # This means H_r and H_phi are zero, but E_z, E_r are not necessarily zero.
            # These are valid modes.

            for k_axial_index in k_axial_values:
                term1_radial_TM = (j_nm_root / radius_a)**2
                term2_axial = (2 * k_axial_index * np.pi /
                               period_h)**2  # Same axial term
                eigenvalue_TM = term1_radial_TM + term2_axial

                azimuthal_multiplicity = 1 if n_order == 0 else 2
                axial_multiplicity = 1 if k_axial_index == 0 else 2
                total_multiplicity = azimuthal_multiplicity * axial_multiplicity

                all_eigenvalues_repeated.extend(
                    [eigenvalue_TM] * total_multiplicity)

    if not all_eigenvalues_repeated:
        return np.array([])

    return np.sort(np.array(all_eigenvalues_repeated))


true_evs = calculate_cylindrical_periodic_TE_TM_eigenvalues(
    range(0, 8), range(1, 8), range(1), a, h)

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
ax1.plot(true_evs[:end], label=r'true',
         marker='', ls = ':', markersize=10, color=color2, lw=LINE_WIDTH)
ax1.plot(evs[:end], label=r'computed',
         marker='*', ls = '', markersize=10, color=color1, lw=LINE_WIDTH)
ax1.tick_params(axis='y', labelsize=TICK_SIZE)
ax1.tick_params(axis='x', labelsize=TICK_SIZE)
# ax1.set_yticks(jnp.unique(true_evs[:end]))
ax1.grid(axis='y', linestyle='--', alpha=0.7)
ax1.legend(fontsize=LEGEND_SIZE) # Use ax1.legend() for clarity

# Now save the figure. The 'tight' layout will be calculated correctly.
fig1.savefig('cylinder_eigenvalues.pdf', bbox_inches='tight')
# %%
# Check that for all EVs in `evs`, there is a corresponding true EV in `true_evs` such that the difference is less than tol:
tol = 1e-5


def dist(ev, true_evs):
    """Calculate the distance between an eigenvalue and the closest true eigenvalue."""
    return jnp.min(jnp.abs(true_evs - ev)/true_evs)


def check_eigenvalues(evs, true_evs, tol=1e-5):
    """Check if all eigenvalues in `evs` are close to some eigenvalue in `true_evs`."""
    return jnp.all(jax.vmap(dist, in_axes=(0, None))(evs, true_evs) < tol)


# %%

# %%
ɛ = 1e-5
nx = 64
_x1 = jnp.linspace(ɛ, 1-ɛ, nx)
_x2 = jnp.linspace(ɛ, 1-ɛ, nx)
_x3 = jnp.ones(1)/2
_x = jnp.array(jnp.meshgrid(_x1, _x2, _x3))
_x = _x.transpose(1, 2, 3, 0).reshape(nx*nx*1, 3)
_y = jax.vmap(F)(_x)
_y1 = _y[:, 0].reshape(nx, nx)
_y2 = _y[:, 1].reshape(nx, nx)
_nx = 16
__x1 = jnp.linspace(ɛ, 1-ɛ, _nx)
__x2 = jnp.linspace(ɛ, 1-ɛ, _nx)
__x3 = jnp.ones(1)/2
__x = jnp.array(jnp.meshgrid(__x1, __x2, __x3))
__x = __x.transpose(1, 2, 3, 0).reshape(_nx*_nx*1, 3)
__y = jax.vmap(F)(__x)
__y1 = __y[:, 0].reshape(_nx, _nx)
_y2 = __y[:, 1].reshape(_nx, _nx)


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
    """
    if num_to_plot > evecs.shape[1]:
        print(
            f"Warning: Requested {num_to_plot} eigenvectors, but only {evecs.shape[1]} are available. Plotting all available.")
        num_to_plot = evecs.shape[1]

    # Determine grid size (aim for roughly square, max 3 columns for 3x3)
    # For a 3x3 grid displaying 9 plots.
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

        ax.contourf(y1_coords, y2_coords, _z1_norm, cmap='plasma', levels=25)

        ax.set_axis_off()
        ax.set_aspect('equal', adjustable='box')  # Maintain aspect ratio

    # Hide any unused subplots if num_to_plot < nrows*ncols
    for j in range(num_to_plot, nrows * ncols):
        fig.delaxes(axes[j])

    plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)  # Adjust padding as needed
    plt.show()

    return fig


# %%
# Plot the first 9 eigenvectors
fig = plot_eigenvectors_grid(
    evecs, M1, derham.Λ1, E1, F, _x, _y1, _y2, nx, num_to_plot=25
)
# %%
fig.savefig('cylinder_eigenmodes.pdf', bbox_inches='tight')
# %%
