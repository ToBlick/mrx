# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from mrx.derham_sequence import DeRhamSequence

# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)

# Initialize parameters
ns = (20, 20, 1)  # Number of elements in each direction
ps = (3, 3, 0)  # Polynomial degree in each direction
types = ('clamped', 'periodic', 'constant')  # Boundary conditions
bcs = ('dirichlet', 'periodic', 'periodic')
q = 8  # Quadrature order

a = 1
h = 1

def F(x):
    """Coordinate transformation from logical to physical coordinates"""
    r, χ, z = x
    R = a * r * jnp.cos(2 * jnp.pi * χ)
    Y = a * r * jnp.sin(2 * jnp.pi * χ)
    Z = h * z
    return jnp.array([R, Y, Z])

# Create DeRham sequence with polar mapping
print("Creating DeRham sequence...")
derham = DeRhamSequence(ns, ps, q, types, bcs, F, polar=True)

# %%
# Assemble mass and stiffness matrices using DeRham sequence
print("Assembling mass matrix...")
M0 = derham.assemble_M0()

print("Assembling stiffness matrix (grad-grad)...")
L = derham.assemble_gradgrad()

# %%
# Solve eigenvalue problem
evs, evecs = sp.linalg.eig(L, M0)
evs = jnp.real(evs)
evecs = jnp.real(evecs)
sort_indices = jnp.argsort(evs)
evs = evs[sort_indices]
evecs = evecs[:, sort_indices]

print(f"Computed {len(evs)} eigenvalues")
print(f"Eigenvalue range: [{evs[0]:.6f}, {evs[-1]:.6f}]")

# %%
def calculate_cylindrical_eigenvalues(
    n_values, m_values, k_values, radius_a, period_h
):
    if not (radius_a > 0 and period_h > 0):
        raise ValueError("Radius 'a' and period 'h' must be positive.")

    all_eigenvalues_repeated = []

    max_m_needed = 0
    if m_values:
        max_m_needed = max(m_values)
        if max_m_needed <= 0:
            raise ValueError(
                "m_values must contain positive integers (m >= 1).")
    else:
        return np.array([])  # No m values, no eigenvalues

    for n_order in n_values:
        if n_order < 0:
            print(
                f"Warning: Order n={n_order} is negative. Physical modes usually have n >= 0. Skipping.")
            # Or take abs(n_order) if desired, but common usage is n>=0
            continue

        try:
            # jn_zeros handles n_order=0 correctly.
            # For n_order > 0, J_n(0) = 0, so the first zero is > 0.
            # For n_order = 0, J_0(0) != 0.
            bessel_zeros_for_n = sp.special.jn_zeros(n_order, max_m_needed)
        except ValueError as e:
            print(
                f"Error getting Bessel zeros for n={n_order}: {e}. Skipping this n.")
            continue

        for m_index_1_based in m_values:  # m_index_1_based is 1, 2, 3,...
            if m_index_1_based <= 0:
                print(
                    f"Warning: m_index={m_index_1_based} is not positive. Skipping.")
                continue

            # m is 1-based index, array is 0-based
            if m_index_1_based - 1 < len(bessel_zeros_for_n):
                J_nm_val = bessel_zeros_for_n[m_index_1_based - 1]
            else:
                print(f"Warning: m={m_index_1_based} is too large for pre-calculated zeros for n={n_order}. "
                      f"Needed {m_index_1_based} zeros, got {len(bessel_zeros_for_n)}. Skipping.")
                continue

            # J_nm_val should not be zero for m >= 1. If it were, it would imply
            # a problem with jn_zeros or the mode definition.

            for k_periodic_index in k_values:  # k_periodic_index is 0, 1, 2,...
                if k_periodic_index < 0:
                    print(
                        f"Warning: k_periodic={k_periodic_index} is negative. Physical modes usually have k_periodic >= 0. Skipping.")
                    continue

                # Calculate the eigenvalue for this specific (n_order, m_index_1_based, k_periodic_index)
                term1_radial = (J_nm_val / radius_a)**2
                term2_axial = (2 * k_periodic_index * np.pi / period_h)**2
                eigenvalue = term1_radial + term2_axial

                # Determine the multiplicity for this eigenvalue
                azimuthal_multiplicity = 1 if n_order == 0 else 2
                axial_multiplicity = 1 if k_periodic_index == 0 else 2

                total_multiplicity = azimuthal_multiplicity * axial_multiplicity

                # Add the eigenvalue 'total_multiplicity' times to the list
                all_eigenvalues_repeated.extend(
                    [eigenvalue] * total_multiplicity)

    return np.sort(np.array(all_eigenvalues_repeated))

# Calculate analytical eigenvalues
true_evs = calculate_cylindrical_eigenvalues(
    range(0, 16), range(1, 16), [0], a, h)

# %%
# Plot comparison
_end = min(32, len(evs), len(true_evs))
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xticks(jnp.arange(1, _end + 1)[::4])
ax.yaxis.grid(True, which='both')
ax.xaxis.grid(True, which='both')
ax.set_ylabel('λ')
ax.set_xlabel('n')

# Plot numerical eigenvalues 
ax.plot(jnp.arange(1, _end + 1),
        evs[:_end], marker='v', label='Numerical', linestyle='-', markersize=6)

# Plot analytical eigenvalues
ax.plot(jnp.arange(1, _end + 1),
        true_evs[:_end], marker='*', label='Analytical', linestyle='', markersize=6)

ax.legend()
ax.set_title('Polar Helmholtz Eigenvalues: Numerical vs Analytical')

plt.tight_layout()
plt.show()

# %%
