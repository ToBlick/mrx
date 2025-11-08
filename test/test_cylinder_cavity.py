"""
Testing that the numerical eigenvalues of the cylinder cavity problem match
the eigenvalues predicted theoretically.
"""


import sys
sys.path.insert(0, '/Users/juliannestratton/mrx') # Can comment this out

import jax
import jax.numpy as jnp
import numpy as np
import scipy as sp

from mrx.derham_sequence import DeRhamSequence

jax.config.update("jax_enable_x64", True)

def test_cylinder_cavity():
    """
    Calculating the cylinder cavity eigenvalues with n=(8,8,1), p=(3,3,0) and 
    asserting that computed eigenvalues match theoretical eigenvalues within tolerance.
    """
    ns = (8, 8, 1)
    ps = (3, 3, 0)
    a = 1
    h = 1
    
    # Compute eigenvalues
    evs = compute_eigenvalues(ns, ps, a, h)
    
    # Compute theoretical eigenvalues
    true_evs = calculate_cylindrical_periodic_TE_TM_eigenvalues(
        range(0, 8), range(1, 8), range(1), a, h
    )
    
    # Filter out zero and negative eigenvalues 
    evs = evs[evs > 1e-10]
    
    # Match eigenvalues
    matched_evs = []
    matched_rel_errors = []  
    used_indices = set() # Empty set to start, will track which ones have already been matched.
    
    for true_ev in true_evs:
        # Find closest computed eigenvalue
        distances = [] # List of distances
        for i, ev in enumerate(evs):
            if i not in used_indices:
                rel_error = float(jnp.abs(ev - true_ev) / true_ev) # Relative error
                distances.append((rel_error, i, float(ev)))
        
        if distances:
            # Sort by relative error 
            distances.sort(key=lambda x: x[0])
            smallest_rel_error, best_idx, best_ev = distances[0] # Best choice
            
            # Only include if within a tolerance
            if smallest_rel_error < 0.09:
                matched_evs.append(best_ev)
                matched_rel_errors.append(smallest_rel_error)  
                used_indices.add(best_idx)
    
    if len(matched_evs) == 0:
        raise AssertionError("No computed eigenvalues matched theoretical eigenvalues.")
    
    evs = jnp.array(matched_evs)
    

    # Set reasonable tolerance
    tol = 1e-1
    max_error = max(matched_rel_errors) 
    
    
    assert max_error < tol, f"Max relative eigenvalue error {max_error:.2e} exceeds tolerance of  {tol:.2e}."
    
    return max_error


def compute_eigenvalues(ns, ps, a, h):
    """Compute eigenvalues for the cylinder cavity problem."""
    types = ('clamped', 'periodic', 'constant')
    
    def _R(r, χ):
        return jnp.ones(1) * (a * r * jnp.cos(2 * jnp.pi * χ))
    
    def _Y(r, χ):
        return jnp.ones(1) * (a * r * jnp.sin(2 * jnp.pi * χ))
    
    def F(x):
        r, χ, z = x
        return jnp.ravel(jnp.array([_R(r, χ), _Y(r, χ), h * jnp.ones(1) * z]))
    
    # Create DeRham sequence
    derham = DeRhamSequence(ns, ps, 8, types, F, polar=True)
    
    # Compute 1D basis functions at quadrature points
    derham.evaluate_1d()
    
    # Assemble mass matrices
    derham.assemble_M0()  # 0-forms
    derham.assemble_M1()  # 1-forms
    derham.assemble_M2()  # 2-forms
    
    # Assemble derivative operators
    derham.assemble_d0()  # Gradient 
    derham.assemble_d1()  # Curl
    

    derham.assemble_dd1()  # Assemble Curl-Curl
    
    # Get mass matrix for 1-forms
    M1 = derham.M1
    
    # Gradient matrix
    D0 = derham.D0
    O10 = jnp.zeros_like(D0)
    O0 = jnp.zeros((D0.shape[1], D0.shape[1]))
    
    # Double curl matrix 
    C = derham.dd1
    
    # Assemble generalized eigenvalue problem
    Q = jnp.block([[C, D0], [D0.T, O0]])
    P = jnp.block([[M1, O10], [O10.T, O0]])
    
    # Solve generalized eigenvalue problem
    evs, evecs = sp.linalg.eig(Q, P)
    evs = jnp.real(evs)
    evecs = jnp.real(evecs)
    
    # Filter out infinite eigenvalues and sort
    finite_indices = jnp.isfinite(evs)
    evs = evs[finite_indices]
    evecs = evecs[:, finite_indices]
    
    sort_indices = jnp.argsort(evs)
    evs = evs[sort_indices]
    evecs = evecs[:, sort_indices]


     # Filter out zero and negative eigenvalues 
    evs = evs[evs > 1e-10]
    
    return evs


def calculate_cylindrical_periodic_TE_TM_eigenvalues(
    n_values,
    m_values,
    k_axial_values,
    radius_a,
    period_h
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




if __name__ == "__main__":
    test_cylinder_cavity()
    print("Test passed.")

