"""
Two-Dimensional Helicity Analysis

This script analyzes the convergence properties of magnetic helicity calculations
in two dimensions using finite element methods. It demonstrates:

1. Error analysis for vector potential reconstruction
2. Helicity conservation properties
3. Convergence rates for different polynomial orders
4. Computational performance scaling

The script generates several plots:
1. Vector potential error convergence
2. Curl error convergence
3. Helicity error convergence
4. Computational time scaling
5. Singular value spectrum

All plots are saved to the script_outputs/ directory.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path

from mrx.DifferentialForms import DifferentialForm
from mrx.Quadrature import QuadratureRule
from mrx.Projectors import Projector
from mrx.LazyMatrices import LazyMassMatrix, LazyDerivativeMatrix, LazyProjectionMatrix, LazyDoubleCurlMatrix
from mrx.Utils import curl
from functools import partial

# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)

# Create output directory for figures
output_dir = Path("script_outputs")
output_dir.mkdir(parents=True, exist_ok=True)

# frequencies of the sin and cos functions
m1 = 2
m2 = 2

@jax.jit
def A(x):
    """Analytical vector potential function."""
    r, χ, z = x
    a1 = jnp.sin(m1 * jnp.pi * r) * jnp.cos(m2 * jnp.pi * χ) * jnp.sqrt(m2**2/(m2**2 + m1**2))
    a2 = -jnp.cos(m1 * jnp.pi * r) * jnp.sin(m2 * jnp.pi * χ) * jnp.sqrt(m1**2/(m2**2 + m1**2))
    a3 = jnp.sin(m1 * jnp.pi * r) * jnp.sin(m2 * jnp.pi * χ)
    return jnp.array([a1, a2, a3])

@partial(jax.jit, static_argnames=['n', 'p'])
def get_error(n, p):
    """Compute errors in vector potential reconstruction and helicity calculation.
    
    Args:
        n: Number of elements in each direction
        p: Polynomial degree
        
    Returns:
        tuple: (A_err, H_err, curl_A_err)
            A_err: Relative error in vector potential reconstruction
            H_err: Relative error in helicity calculation
            curl_A_err: Relative error in curl calculation
    """
    types = ('periodic', 'periodic', 'constant')

    ns = (n, n, 1)
    ps = (p, p, 0)

    Λ0 = DifferentialForm(0, ns, ps, types)
    Λ1 = DifferentialForm(1, ns, ps, types)
    Λ2 = DifferentialForm(2, ns, ps, types)
    Q = QuadratureRule(Λ0, 10)

    M2 = LazyMassMatrix(Λ2, Q).M
    M1 = LazyMassMatrix(Λ1, Q).M
    C = LazyDoubleCurlMatrix(Λ1, Q).M
    D = LazyDerivativeMatrix(Λ1, Λ2, Q).M
    P2 = Projector(Λ2, Q)
    P1 = Projector(Λ1, Q)

    M12 = LazyProjectionMatrix(Λ1, Λ2, Q).M.T
    B = curl(A)

    A_hat = jnp.linalg.solve(M1, P1(A))
    B_hat = jnp.linalg.solve(M2, P2(B))

    U, S, Vh = jnp.linalg.svd(C)
    S_inv = jnp.where(S/S[0] > 1e-12, 1/S, 0)

    A_hat_recon = Vh.T @ jnp.diag(S_inv) @ U.T @ D.T @ B_hat

    A_err = ((A_hat - A_hat_recon) @ M1 @ (A_hat - A_hat_recon) / (A_hat @ M1 @ A_hat))**0.5
    H_err = (A_hat - A_hat_recon) @ M12 @ B_hat / (A_hat @ M12 @ B_hat)
    curl_A_err = (jnp.linalg.solve(M2, D @ (A_hat - A_hat_recon)) @ M2 @ jnp.linalg.solve(M2, D @ (A_hat - A_hat_recon)) / (jnp.linalg.solve(M2, D @ A_hat) @ M2 @ jnp.linalg.solve(M2, D @ A_hat)))**0.5

    return A_err, H_err, curl_A_err

# Run convergence analysis
print("Running convergence analysis...")
ns = np.arange(4, 15, 2)
ps = np.arange(1, 4)
A_err = np.zeros((len(ns), len(ps)))
H_err = np.zeros((len(ns), len(ps)))
curl_A_err = np.zeros((len(ns), len(ps)))
times = np.zeros((len(ns), len(ps)))

for i, n in enumerate(ns):
    for j, p in enumerate(ps):
        start = time.time()
        _A_err, _H_err, _curl_A_err = get_error(n, p)
        A_err[i, j] = _A_err
        H_err[i, j] = _H_err
        curl_A_err[i, j] = _curl_A_err
        end = time.time()
        times[i, j] = end - start
        print(f"n={n}, p={p}, A_err={A_err[i, j]:.2e}, H_err={H_err[i, j]:.2e}, curl_A_err={curl_A_err[i, j]:.2e}, time={times[i, j]:.2f}s")

# Plot vector potential error convergence
plt.figure(figsize=(10, 6))
plt.plot(ns, A_err[:, 0], label='p=1', marker='o')
plt.plot(ns, A_err[:, 1], label='p=2', marker='*')
plt.plot(ns, A_err[:, 2], label='p=3', marker='s')
plt.plot(ns, A_err[-1, 0] * (ns/ns[-1])**(-2), label='O(n^-2)', linestyle='--')
plt.plot(ns, A_err[-1, 1] * (ns/ns[-1])**(-4), label='O(n^-4)', linestyle='--')
plt.plot(ns, A_err[-1, 2] * (ns/ns[-1])**(-6), label='O(n^-6)', linestyle='--')
plt.loglog()
plt.xlabel('Number of Elements (n)')
plt.ylabel('Relative Error in A')
plt.title('Vector Potential Error Convergence')
plt.grid(True)
plt.legend()
plt.savefig(output_dir / 'vector_potential_error.png', dpi=300, bbox_inches='tight')

# Plot curl error convergence
plt.figure(figsize=(10, 6))
plt.plot(ns, curl_A_err[:, 0], label='p=1', marker='o')
plt.plot(ns, curl_A_err[:, 1], label='p=2', marker='*')
plt.plot(ns, curl_A_err[:, 2], label='p=3', marker='s')
plt.plot(ns, curl_A_err[-1, 0] * (ns/ns[-1])**(-2), label='O(n^-2)', linestyle='--')
plt.plot(ns, curl_A_err[-1, 1] * (ns/ns[-1])**(-4), label='O(n^-4)', linestyle='--')
plt.plot(ns, curl_A_err[-1, 2] * (ns/ns[-1])**(-6), label='O(n^-6)', linestyle='--')
plt.loglog()
plt.xlabel('Number of Elements (n)')
plt.ylabel('Relative Error in curl A')
plt.title('Curl Error Convergence')
plt.grid(True)
plt.legend()
plt.savefig(output_dir / 'curl_error.png', dpi=300, bbox_inches='tight')

# Plot helicity error convergence
plt.figure(figsize=(10, 6))
plt.plot(ns, H_err[:, 0], label='p=1', marker='o')
plt.plot(ns, H_err[:, 1], label='p=2', marker='*')
plt.plot(ns, H_err[:, 2], label='p=3', marker='s')
plt.plot(ns, H_err[-1, 0] * (ns/ns[-1])**(-2), label='O(n^-2)', linestyle='--')
plt.plot(ns, H_err[-1, 1] * (ns/ns[-1])**(-4), label='O(n^-4)', linestyle='--')
plt.plot(ns, H_err[-1, 2] * (ns/ns[-1])**(-6), label='O(n^-6)', linestyle='--')
plt.loglog()
plt.xlabel('Number of Elements (n)')
plt.ylabel('Relative Error in Helicity')
plt.title('Helicity Error Convergence')
plt.grid(True)
plt.legend()
plt.savefig(output_dir / 'helicity_error.png', dpi=300, bbox_inches='tight')

# Plot computational time scaling
plt.figure(figsize=(10, 6))
plt.plot(ns, times[:, 0], label='p=1', marker='o')
plt.plot(ns, times[:, 1], label='p=2', marker='*')
plt.plot(ns, times[:, 2], label='p=3', marker='s')
plt.plot(ns, times[0, 0] * (ns/ns[0])**(4), label='O(n^4)', linestyle='--')
plt.loglog()
plt.xlabel('Number of Elements (n)')
plt.ylabel('Computation Time [s]')
plt.title('Computational Time Scaling')
plt.grid(True)
plt.legend()
plt.savefig(output_dir / 'computation_time.png', dpi=300, bbox_inches='tight')

# Analyze singular value spectrum
print("\nAnalyzing singular value spectrum...")
# Define mesh parameters for singular value analysis
types = ('periodic', 'periodic', 'constant')  # Boundary condition types
nn = 6  # Number of elements for singular value analysis
pp = 3  # Polynomial degree for singular value analysis
nns = (nn, nn, 1)  # Mesh dimensions
pps = (pp, pp, 0)  # Polynomial degrees

# Initialize differential forms and matrices for singular value analysis
Λ0 = DifferentialForm(0, tuple(nns), tuple(pps), types)  # 0-form space
Λ1 = DifferentialForm(1, nns, pps, types)  # 1-form space
Λ2 = DifferentialForm(2, nns, pps, types)  # 2-form space
Q = QuadratureRule(Λ0, 3)  # Quadrature rule with 3 points per dimension

# Assemble mass matrices and derivative operators
M2 = LazyMassMatrix(Λ2, Q).M  # Mass matrix for 2-forms
M1 = LazyMassMatrix(Λ1, Q).M  # Mass matrix for 1-forms
D = LazyDerivativeMatrix(Λ1, Λ2, Q).M  # Derivative operator from 1-forms to 2-forms
P2 = Projector(Λ2, Q)  # Projector for 2-forms
P1 = Projector(Λ1, Q)  # Projector for 1-forms
M12 = LazyProjectionMatrix(Λ1, Λ2, Q).M  # Projection matrix from 1-forms to 2-forms

# Compute double curl operator and its SVD
C = D.T @ jnp.linalg.solve(M2, D)  # Double curl operator
_C = LazyDoubleCurlMatrix(Λ1, Q).M  # Alternative double curl operator

# Compute magnetic field and vector potential
B = curl(A)  # Magnetic field from vector potential
A_hat = jnp.linalg.solve(M1, P1(A))  # Projected vector potential
B_hat = jnp.linalg.solve(M2, P2(B))  # Projected magnetic field

# Compute singular value decomposition of double curl operator
U, S, Vh = jnp.linalg.svd(C)  # SVD of double curl operator

# Plot singular value spectrum
plt.figure(figsize=(10, 6))
plt.plot(S/S[0], marker='o')  # Plot normalized singular values
plt.yscale('log')  # Use logarithmic scale for y-axis
plt.xlabel('Index')
plt.ylabel('Normalized Singular Value')
plt.title('Singular Value Spectrum')
plt.grid(True)
plt.savefig(output_dir / 'singular_values.png', dpi=300, bbox_inches='tight')

# Print final diagnostics
print("\nFinal diagnostics:")
print(f"Vector potential error: {A_err[-1, -1]:.2e}")  # Final vector potential error
print(f"Helicity error: {H_err[-1, -1]:.2e}")  # Final helicity error
print(f"Curl error: {curl_A_err[-1, -1]:.2e}")  # Final curl error
print(f"Computation time: {times[-1, -1]:.2f}s")  # Final computation time

# Compute inverse of double curl operator with regularization
S_inv = jnp.where(S/S[0] > 1e-12, 1/S, 0)  # Regularized inverse singular values
C_inv = Vh.T @ jnp.diag(S_inv) @ U.T  # Regularized inverse of double curl operator

# Reconstruct vector potential and compute errors
A_hat_recon = C_inv @ D.T @ B_hat  # Reconstructed vector potential
A_err = ((A_hat - A_hat_recon) @ M1 @ (A_hat - A_hat_recon) / (A_hat @ M1 @ A_hat))**0.5  # Vector potential error
print("error in A:", A_err)  # Print vector potential error

H_err = (A_hat - A_hat_recon) @ M12 @ B_hat / (A_hat @ M12 @ B_hat)  # Helicity error
print("error in Helicity:", H_err)  # Print helicity error

curl_A_err = (jnp.linalg.solve(M2, D @ (A_hat - A_hat_recon)) @ M2 @ jnp.linalg.solve(M2, D @ (A_hat - A_hat_recon)) / (jnp.linalg.solve(M2, D @ A_hat) @ M2 @ jnp.linalg.solve(M2, D @ A_hat)))**0.5  # Curl error
print("error in curl A:", curl_A_err)  # Print curl error

# Compute residual of double curl equation
residual = ((C @ A_hat - D.T @ B_hat) @ M1 @ (C @ A_hat - D.T @ B_hat))**0.5  # Residual of double curl equation
print("residual of double curl equation:", residual)  # Print residual

# Compute divergence-free constraint violation
v1 = M12 @ B_hat - D.T @ jnp.linalg.solve(M2, M12.T @ A_hat)  # Divergence-free constraint violation
divergence_violation = (v1 @ M1 @ v1)**0.5  # Norm of divergence-free constraint violation
print("divergence-free constraint violation:", divergence_violation)  # Print violation

# Show all plots
plt.show()
