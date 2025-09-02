# %%
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

All plots are saved to the script_outputs/ directory.
"""

import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from mrx.DifferentialForms import DifferentialForm
from mrx.LazyMatrices import (
    LazyDerivativeMatrix,
    LazyDoubleCurlMatrix,
    LazyMassMatrix,
    LazyProjectionMatrix,
)
from mrx.Projectors import Projector
from mrx.Quadrature import QuadratureRule
from mrx.Utils import curl

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
    a1 = (
        jnp.sin(m1 * jnp.pi * r)
        * jnp.cos(m2 * jnp.pi * χ)
        * jnp.sqrt(m2**2 / (m2**2 + m1**2))
    )
    a2 = (
        -jnp.cos(m1 * jnp.pi * r)
        * jnp.sin(m2 * jnp.pi * χ)
        * jnp.sqrt(m1**2 / (m2**2 + m1**2))
    )
    a3 = jnp.sin(m1 * jnp.pi * r) * jnp.sin(m2 * jnp.pi * χ)
    return jnp.array([a1, a2, a3])


@partial(jax.jit, static_argnames=["n", "p"])
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
    types = ("periodic", "periodic", "constant")

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
    S_inv = jnp.where(S / S[0] > 1e-12, 1 / S, 0)

    A_hat_recon = Vh.T @ jnp.diag(S_inv) @ U.T @ D.T @ B_hat

    A_err = (
        (A_hat - A_hat_recon) @ M1 @ (A_hat - A_hat_recon) / (A_hat @ M1 @ A_hat)
    ) ** 0.5
    H_err = (A_hat - A_hat_recon) @ M12 @ B_hat / (A_hat @ M12 @ B_hat)
    curl_A_err = (
        jnp.linalg.solve(M2, D @ (A_hat - A_hat_recon))
        @ M2
        @ jnp.linalg.solve(M2, D @ (A_hat - A_hat_recon))
        / (jnp.linalg.solve(M2, D @ A_hat) @ M2 @ jnp.linalg.solve(M2, D @ A_hat))
    ) ** 0.5

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
        print(
            f"n={n}, p={p}, A_err={A_err[i, j]:.2e}, H_err={H_err[i, j]:.2e}, curl_A_err={curl_A_err[i, j]:.2e}, time={times[i, j]:.2f}s"
        )

# %%
# Plot vector potential error convergence
plt.figure(figsize=(10, 6))
for j, p in enumerate(ps):
    plt.loglog(ns, A_err[:, j], label=f"p={p}", marker="o")
    # Add theoretical convergence rates
    plt.loglog(
        ns,
        A_err[-1, j] * (ns / ns[-1]) ** (-2 * p),
        label=f"O(n^-{2 * p})",
        linestyle="--",
    )
plt.xlabel("Number of Elements (n)")
plt.ylabel("Relative Error in A")
plt.title("Vector Potential (A) Error Convergence")
plt.grid(True)
plt.legend()
plt.savefig(output_dir / "A_error.png", dpi=300, bbox_inches="tight")

# Plot curl error convergence
plt.figure(figsize=(10, 6))
for j, p in enumerate(ps):
    plt.loglog(ns, curl_A_err[:, j], label=f"p={p}", marker="o")
    # Add theoretical convergence rates
    plt.loglog(
        ns,
        curl_A_err[-1, j] * (ns / ns[-1]) ** (-2 * p),
        label=f"O(n^-{2 * p})",
        linestyle="--",
    )
plt.xlabel("Number of Elements (n)")
plt.ylabel("Relative Error in curl A")
plt.title("curl(A) Error Convergence")
plt.grid(True)
plt.legend()
plt.savefig(output_dir / "curl_A_error.png", dpi=300, bbox_inches="tight")

# Plot helicity error convergence
plt.figure(figsize=(10, 6))
for j, p in enumerate(ps):
    plt.loglog(ns, H_err[:, j], label=f"p={p}", marker="o")
    # Add theoretical convergence rates
    plt.loglog(
        ns,
        H_err[-1, j] * (ns / ns[-1]) ** (-2 * p),
        label=f"O(n^-{2 * p})",
        linestyle="--",
    )
plt.xlabel("Number of Elements (n)")
plt.ylabel("Relative Error in H")
plt.title("Magnetic Field (H) Error Convergence")
plt.grid(True)
plt.legend()
plt.savefig(output_dir / "H_error.png", dpi=300, bbox_inches="tight")

# Show all plots
plt.show()
# Clean up
plt.close("all")

# %%
