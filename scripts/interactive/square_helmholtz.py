# %%
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from mrx.DeRhamSequence import DeRhamSequence

# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)

# Get the absolute path to the script's directory
script_dir = Path(__file__).parent.absolute()
# Create the output directory in the same directory as the script
output_dir = script_dir / 'script_outputs'
os.makedirs(output_dir, exist_ok=True)

# Initialize parameters
ns = (12,12, 1)  # Number of elements in each direction
ps = (3, 3, 0)  # Polynomial degree in each direction
types = ('clamped', 'clamped', 'constant')  # Boundary conditions
bcs = ('dirichlet', 'dirichlet', 'none')
q = 6  # Quadrature order

def F(x): 
    """Coordinate transformation from logical to physical coordinates"""
    return x * jnp.pi

# Create DeRham sequence (non-polar since this is square domain)
derham = DeRhamSequence(ns, ps, q, types, bcs, F, polar=False)

def generalized_eigh(A, B):
    Q = jnp.linalg.cholesky(B)
    C = jnp.linalg.solve(Q, jnp.linalg.solve(Q, A.T).T)
    eigenvalues, eigenvectors_transformed = jax.scipy.linalg.eigh(C)
    eigenvectors_original = jnp.linalg.solve(Q.T, eigenvectors_transformed)
    return eigenvalues, eigenvectors_original

_end = 26

# %%
for i in range(2):
    if i == 0:
        # k=0 case: Assemble mass and stiffness matrices
        M0 = derham.assemble_M0()
        L = derham.assemble_gradgrad()
        
        # Solve generalized eigenvalue problem
        evs, evecs = generalized_eigh(L, M0)
        
    else:
        
        # k = 3 case: Assemble mass matrices for 2-forms and 3-forms
        M2 = derham.assemble_M2()
        M3 = derham.assemble_M3()
        
        # Assemble divergence matrix from 2-forms to 3-forms
        D2 = derham.assemble_dvg()
        
        # Mixed formulation: L = D2 @ M2^(-1) @ D2^T
        L = D2 @ jnp.linalg.solve(M2, D2.T)
        
        # Solve generalized eigenvalue problem
        evs, evecs = generalized_eigh(L, M3)
    
    # Extract eigenvalues (skip first one)
    evs = evs[1:_end]

    # Analytical eigenvalues
    true_evs = jnp.sort(jnp.array(
        [i**2 + j**2 for i in range(0, ns[0]+1) for j in range(0, ns[1]+1)]))[1:_end]
    
    # Got an error, so here just ensuring we have enough eigenvalues for plotting
    act_end = min(len(evs), len(true_evs), _end)
    if act_end < _end:
        print(f"Careful: Only {act_end} eigenvalues available!")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_yticks((true_evs[:act_end]))
    ax.set_xticks(jnp.arange(1, act_end + 1)[::2])
    ax.yaxis.grid(True, which='both')
    ax.xaxis.grid(True, which='both')
    ax.set_ylabel('λ')
    ax.set_xlabel('n')
    
    # Plot numerical vs analytical eigenvalues
    ax.plot(jnp.arange(1, act_end + 1), evs[:act_end], marker='v', label='Numerical', 
            linestyle='-', markersize=6)
    ax.plot(jnp.arange(1, act_end + 1), true_evs[:act_end], marker='*', label='Analytical', 
            linestyle='', markersize=6)
    
    ax.legend()
    
    if i == 0:
        ax.set_title('Eigenvalues of the Laplace operator: k = 0')
        plt.savefig(
            "script_outputs/Helmholtz_eigenvalues_k=0.png", dpi=300, bbox_inches="tight"
        )
    else:
        ax.set_title('Eigenvalues of the Laplace operator: k = 3')
        plt.savefig(
            "script_outputs/Helmholtz_eigenvalues_k=3.png", dpi=300, bbox_inches="tight"
        )
    
    plt.show()

# %%
# Calculate L1 error for the last case
actual_end = min(len(evs), len(true_evs))
if actual_end > 0:
    l1_error = jnp.sum(jnp.abs(evs[:actual_end] - true_evs[:actual_end])) / jnp.sum(true_evs[:actual_end])
    print(f"L1 error: {l1_error:.6f}")
else:
    l1_error = float('nan')
    print("Warning: No eigenvalues available for error calculation!")

