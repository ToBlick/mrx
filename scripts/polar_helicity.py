"""Polar Helicity Analysis Script

This script analyzes the convergence properties of magnetic helicity calculations in a polar geometry.
It computes and visualizes errors in vector potential (A), magnetic helicity (H), and curl of A for
different polynomial degrees and mesh resolutions.

The script demonstrates:
1. Error convergence rates for different polynomial degrees
2. Computational time scaling with mesh resolution
3. Comparison of exact vs reconstructed fields
4. Visualization of error metrics in log-log plots

Key components:
- Differential forms for field representation
- Polar mapping for geometry definition
- Lazy matrices for efficient computation
- Error analysis for different field quantities
"""

# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import time
from pathlib import Path

from mrx.DifferentialForms import DifferentialForm
from mrx.Quadrature import QuadratureRule
from mrx.Projectors import Projector
from mrx.LazyMatrices import LazyMassMatrix, LazyDerivativeMatrix, LazyProjectionMatrix, LazyDoubleCurlMatrix
from mrx.Utils import curl
from mrx.PolarMapping import LazyExtractionOperator, get_xi
from functools import partial
jax.config.update("jax_enable_x64", True)
# %%

# Create output directory for figures
output_dir = Path("scripts_output")
output_dir.mkdir(parents=True, exist_ok=True)


@partial(jax.jit, static_argnames=['n', 'p'])
def get_error(n, p):
    """Compute error metrics for magnetic helicity calculations.

    Args:
        n: Number of elements in each direction
        p: Polynomial degree

    Returns:
        tuple: (A_err, H_err, curl_A_err)
            A_err: Relative L2 error in vector potential reconstruction
            H_err: Relative error in magnetic helicity
            curl_A_err: Relative L2 error in curl of vector potential
    """
    types = ('clamped', 'periodic', 'constant')
    ns = (n, n, 1)
    ps = (p, p, 0)

    Λ0, Λ1, Λ2, Λ3 = [DifferentialForm(i, ns, ps, types) for i in range(4)]

    # Warning: this works with q = 3 but appears NOT to work with q = 10
    Q = QuadratureRule(Λ0, 3)

    ###
    # Mapping definition
    ###
    a = 1
    R0 = 3.0
    Y0 = 0.0

    def θ(x):
        r, χ, z = x
        return 2 * jnp.atan(jnp.sqrt((1 + a*r/R0)/(1 - a*r/R0)) * jnp.tan(jnp.pi * χ))

    def _R(r, χ):
        return jnp.ones(1) * (R0 + a * r * jnp.cos(2 * jnp.pi * χ))

    def _Y(r, χ):
        return jnp.ones(1) * (Y0 + a * r * jnp.sin(2 * jnp.pi * χ))

    def F(x):
        r, χ, z = x
        return jnp.ravel(jnp.array([_R(r, χ) * jnp.cos(2 * jnp.pi * z),
                                    _Y(r, χ),
                                    _R(r, χ) * jnp.sin(2 * jnp.pi * z)]))

    ξ, R_hat, Y_hat, Λ, τ = get_xi(_R, _Y, Λ0, Q)
    Q = QuadratureRule(Λ0, 10)  # Redefine at higher order
    E0, E1, E2, E3 = [LazyExtractionOperator(Λ, ξ, True).M for Λ in [Λ0, Λ1, Λ2, Λ3]]
    M0, M1, M2, M3 = [LazyMassMatrix(Λ, Q, F, E).M for Λ, E in zip([Λ0, Λ1, Λ2, Λ3], [E0, E1, E2, E3])]
    P0, P1, P2, P3 = [Projector(Λ, Q, F, E) for Λ, E in zip([Λ0, Λ1, Λ2, Λ3], [E0, E1, E2, E3])]

    M12 = LazyProjectionMatrix(Λ1, Λ2, Q, F, E1, E2).M.T
    C = LazyDoubleCurlMatrix(Λ1, Q, F, E1).M
    D1 = LazyDerivativeMatrix(Λ1, Λ2, Q, F, E1, E2).M
    # D0 = LazyDerivativeMatrix(Λ0, Λ1, Q, F, E0, E1).M

    def A(x):
        r, χ, z = x
        a1 = jnp.sin(2 * jnp.pi * χ)
        a2 = 1
        a3 = jnp.cos(2 * jnp.pi * χ)
        return jnp.array([a1, a2, a3]) * jnp.sin(jnp.pi * r)**2
    B = curl(A)

    A_hat = jnp.linalg.solve(M1, P1(A))
    B_hat = jnp.linalg.solve(M2, P2(B))

    U, S, Vh = jnp.linalg.svd(C)
    S_inv = jnp.where(S/S[0] > 1e-12, 1/S, 0)

    A_hat_recon = Vh.T @ jnp.diag(S_inv) @ U.T @ D1.T @ B_hat

    A_err = ((A_hat - A_hat_recon) @ M1 @ (A_hat - A_hat_recon) / (A_hat @ M1 @ A_hat))**0.5
    # print("error in A:", A_err)

    H_err = (A_hat - A_hat_recon) @ M12 @ B_hat / (A_hat @ M12 @ B_hat)
    # print("error in Helicity:", H_err)

    curl_A_err = (jnp.linalg.solve(M2, D1 @ (A_hat - A_hat_recon)) @ M2 @ jnp.linalg.solve(M2, D1 @ (A_hat - A_hat_recon)) / (jnp.linalg.solve(M2, D1 @ A_hat) @ M2 @ jnp.linalg.solve(M2, D1 @ A_hat)))**0.5
    # print("error in curl A:", curl_A_err)

    return A_err, H_err, curl_A_err


# %%
ns = np.arange(4, 14, 2)
ps = np.arange(1, 4)
A_err = np.zeros((len(ns), len(ps)))
H_err = np.zeros((len(ns), len(ps)))
curl_A_err = np.zeros((len(ns), len(ps)))
times = np.zeros((len(ns), len(ps)))

# Main analysis loop - first run (includes JIT compilation)
print("\nRunning first analysis (includes JIT compilation)...")
times_first = np.zeros((len(ns), len(ps)))
for i, n in enumerate(ns):
    for j, p in enumerate(ps):
        start = time.time()
        _A_err, _H_err, _curl_A_err = get_error(n, p)
        end = time.time()
        times_first[i, j] = end - start
        A_err[i, j] = _A_err
        H_err[i, j] = _H_err
        curl_A_err[i, j] = _curl_A_err
        print(f"n={n}, p={p}, A_err={A_err[i, j]:.2e}, H_err={H_err[i, j]:.2e}, curl_A_err={curl_A_err[i, j]:.2e}, time={times_first[i, j]:.2f}s")

# Main analysis loop - second run (after JIT compilation)
print("\nRunning second analysis (after JIT compilation)...")
times_second = np.zeros((len(ns), len(ps)))
for i, n in enumerate(ns):
    for j, p in enumerate(ps):
        start = time.time()
        _A_err, _H_err, _curl_A_err = get_error(n, p)
        end = time.time()
        times_second[i, j] = end - start
        print(f"n={n}, p={p}, A_err={_A_err:.2e}, H_err={_H_err:.2e}, curl_A_err={_curl_A_err:.2e}, time={times_second[i, j]:.2f}s")

# Calculate speedup
speedup = times_first / times_second

# Plot and save speedup
plt.figure(figsize=(10, 6))
plt.plot(ns, speedup[:, 0], label='p=1', marker='o')
plt.plot(ns, speedup[:, 1], label='p=2', marker='*')
plt.plot(ns, speedup[:, 2], label='p=3', marker='s')
plt.xlabel('n')
plt.ylabel('Speedup')
plt.title('JIT Compilation Speedup')
plt.grid(True)
plt.legend()
plt.savefig(output_dir / 'jit_speedup.png', dpi=300, bbox_inches='tight')

# Plot and save vector potential error convergence
plt.figure(figsize=(10, 6))
plt.plot(ns, A_err[:, 0], label='p=1', marker='o')
plt.plot(ns, A_err[:, 1], label='p=2', marker='*')
plt.plot(ns, A_err[:, 2], label='p=3', marker='s')
plt.plot(ns, A_err[-1, 0] * (ns/ns[-1])**(-2), label='O(n^-2)', linestyle='--')
plt.plot(ns, A_err[-1, 1] * (ns/ns[-1])**(-4), label='O(n^-4)', linestyle='--')
plt.plot(ns, A_err[-1, 2] * (ns/ns[-1])**(-6), label='O(n^-6)', linestyle='--')
plt.loglog()
plt.xlabel('n')
plt.ylabel('A Error')
plt.title('Vector Potential Error Convergence')
plt.grid(True)
plt.legend()
plt.savefig(output_dir / 'vector_potential_error.png', dpi=300, bbox_inches='tight')

# Plot and save curl error convergence
plt.figure(figsize=(10, 6))
plt.plot(ns, curl_A_err[:, 0], label='p=1', marker='o')
plt.plot(ns, curl_A_err[:, 1], label='p=2', marker='*')
plt.plot(ns, curl_A_err[:, 2], label='p=3', marker='s')
plt.plot(ns, curl_A_err[-1, 0] * (ns/ns[-1])**(-2), label='O(n^-2)', linestyle='--')
plt.plot(ns, curl_A_err[-1, 1] * (ns/ns[-1])**(-4), label='O(n^-4)', linestyle='--')
plt.plot(ns, curl_A_err[-1, 2] * (ns/ns[-1])**(-6), label='O(n^-6)', linestyle='--')
plt.loglog()
plt.xlabel('n')
plt.ylabel('curl A Error')
plt.title('Curl Error Convergence')
plt.grid(True)
plt.legend()
plt.savefig(output_dir / 'curl_error.png', dpi=300, bbox_inches='tight')

# Plot and save helicity error convergence
plt.figure(figsize=(10, 6))
plt.plot(ns, H_err[:, 0], label='p=1', marker='o')
plt.plot(ns, H_err[:, 1], label='p=2', marker='*')
plt.plot(ns, H_err[:, 2], label='p=3', marker='s')
plt.plot(ns, H_err[-1, 0] * (ns/ns[-1])**(-2), label='O(n^-2)', linestyle='--')
plt.plot(ns, H_err[-1, 1] * (ns/ns[-1])**(-4), label='O(n^-4)', linestyle='--')
plt.plot(ns, H_err[-1, 2] * (ns/ns[-1])**(-6), label='O(n^-6)', linestyle='--')
plt.loglog()
plt.xlabel('n')
plt.ylabel('H Error')
plt.title('Helicity Error Convergence')
plt.grid(True)
plt.legend()
plt.savefig(output_dir / 'helicity_error.png', dpi=300, bbox_inches='tight')

# Plot and save computational time scaling
plt.figure(figsize=(10, 6))
plt.plot(ns, times[:, 0], label='p=1', marker='o')
plt.plot(ns, times[:, 1], label='p=2', marker='*')
plt.plot(ns, times[:, 2], label='p=3', marker='s')
plt.plot(ns, times[0, 0] * (ns/ns[0])**(4), label='O(n^4)', linestyle='--')
plt.loglog()
plt.xlabel('n')
plt.ylabel('Time [s]')
plt.title('Computational Time Scaling')
plt.grid(True)
plt.legend()
plt.savefig(output_dir / 'time_scaling.png', dpi=300, bbox_inches='tight')

# Show all plots
plt.show()

print("\nAnalysis completed successfully!")
print(f"Figures saved to: {output_dir}")
print("\nAverage speedup after JIT compilation:")
for j, p in enumerate(ps):
    print(f"p={p}: {np.mean(speedup[:, j]):.2f}x")
