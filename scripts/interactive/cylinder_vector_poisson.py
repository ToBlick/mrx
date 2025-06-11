# %%
import os
import time
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from mrx.DifferentialForms import DifferentialForm, Flat
from mrx.LazyMatrices import (
    LazyDerivativeMatrix,
    LazyDoubleDivergenceMatrix,
    LazyMassMatrix,
)
from mrx.PolarMapping import LazyExtractionOperator, get_xi
from mrx.Projectors import Projector
from mrx.Quadrature import QuadratureRule

# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)

# Create output directory for figures
os.makedirs("script_outputs", exist_ok=True)


@partial(jax.jit, static_argnames=["n", "p"])
def get_err(n, p):
    # Set up finite element spaces
    q = 2*p
    ns = (n, n, n)
    ps = (p, p, p)

    types = ("clamped", "periodic", "periodic")
    # Domain parameters
    π = jnp.pi

    def _X(r, χ):
        return jnp.ones(1) * r * jnp.cos(2 * π * χ)

    def _Y(r, χ):
        return jnp.ones(1) * r * jnp.sin(2 * π * χ)

    def _Z(r, χ):
        return jnp.ones(1)

    def F(x):
        """Polar coordinate mapping function."""
        r, χ, z = x
        return jnp.ravel(jnp.array([_X(r, χ),
                                    _Y(r, χ),
                                    _Z(r, χ) * z]))
    # Define exact solution and source term

    def u(x):
        """Exact solution of the Poisson problem."""
        r, χ, z = x
        u_theta = r**2 * (1 - r)**2 * jnp.cos(2*π*z)
        return jnp.array([0, u_theta, 0])

    def f(x):
        """Source term of the Poisson problem."""
        r, χ, z = x
        f_theta = (r**2 * (1 - r)**2 * 4*π**2 -
                   (3 - 16*r + 15*r**2)) * jnp.cos(2*π*z)
        return jnp.array([0, f_theta, 0])

    Λ0, Λ1, Λ2, Λ3 = [DifferentialForm(k, ns, ps, types) for k in range(4)]

    # Get polar mapping and set up operators
    Q = QuadratureRule(Λ0, q)
    ξ = get_xi(_X, _Z, Λ0, Q)[0]
    E1 = LazyExtractionOperator(Λ1, ξ, zero_bc=False).M
    E2 = LazyExtractionOperator(Λ2, ξ, zero_bc=True).M

    C = LazyDerivativeMatrix(Λ1, Λ2, Q, F, E1, E2).M
    K = LazyDoubleDivergenceMatrix(Λ2, Q, F, E2).M
    M1 = LazyMassMatrix(Λ1, Q, F, E1).M
    M2 = LazyMassMatrix(Λ2, Q, F, E2).M

    # block_matrix = jnp.block([[K, C], [-C.T, M1]])

    L = C @ jnp.linalg.solve(M1, C.T) + K

    tol = 1e-12
    eigvals, eigvecs = jnp.linalg.eigh(L)
    inv_eigvals = jnp.where(
        jnp.abs(eigvals) > tol,
        1.0 / eigvals,
        0.0
    )
    L_pinv = (eigvecs * inv_eigvals) @ eigvecs.T

    P2 = Projector(Λ2, Q, F, E2)
    u_hat = L_pinv @ P2(Flat(f, F))
    u_hat_analytic = jnp.linalg.solve(M2, P2(Flat(u, F)))
    error = ((u_hat - u_hat_analytic) @ M2 @ (u_hat - u_hat_analytic) /
             (u_hat_analytic @ M2 @ u_hat_analytic))**0.5
    return error


def run_convergence_analysis(ns, ps):
    """Run convergence analysis for different parameters."""

    # Arrays to store results
    err = np.zeros((len(ns), len(ps)))
    times = np.zeros((len(ns), len(ps)))

    # First run (with JIT compilation)
    print("First run (with JIT compilation):")
    for i, n in enumerate(ns):
        for j, p in enumerate(ps):
            start = time.time()
            err[i, j] = get_err(n, p)
            end = time.time()
            times[i, j] = end - start
            print(
                f"n={n}, p={p}, err={err[i, j]:.2e}, time={times[i, j]:.2f}s")

    # Second run (after JIT compilation)
    print("\nSecond run (after JIT compilation):")
    times2 = np.zeros((len(ns), len(ps)))
    for i, n in enumerate(ns):
        for j, p in enumerate(ps):
            start = time.time()
            _ = get_err(n, p)  # We don't need to store the error again
            end = time.time()
            times2[i, j] = end - start
            print(f"n={n}, p={p}, time={times2[i, j]:.2f}s")

    return err, times, times2


def plot_results(err, times, times2, ns, ps):
    """Plot the results of the convergence analysis."""
    # Error convergence plot
    fig1 = plt.figure(figsize=(10, 6))
    for j, p in enumerate(ps):
        plt.loglog(ns, err[:, j],
                   label=f'p={p}',
                   marker='o')
    # Add theoretical convergence rates
    for j, p in enumerate(ps):
        expected_rate = 2 * p - 1 * (p != 1)
        plt.loglog(ns, err[-1, j] * (ns/ns[-1])**(-expected_rate),
                   label=f'O(n^{-expected_rate})', linestyle='--')
    plt.xlabel('Number of elements (n)')
    plt.ylabel('Relative L2 error')
    plt.title('Error Convergence')
    plt.grid(True)
    plt.legend()
    plt.savefig('script_outputs/toroid_vectorpoisson_mixed_error.png',
                dpi=300, bbox_inches='tight')

    return fig1


def main():
    """Main function to run the analysis."""
    # Run convergence analysis
    ns = np.arange(4, 6, 1)
    ps = np.arange(1, 4)
    err, times, times2 = run_convergence_analysis(ns, ps)

    # Plot results
    plot_results(err, times, times2, ns, ps)

    # Show all figures
    plt.show()

    # Clean up
    plt.close('all')


if __name__ == "__main__":
    main()
