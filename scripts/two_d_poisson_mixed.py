"""
2D Mixed Poisson Problem

This script solves a 2D Poisson problem using a mixed finite element formulation.
The problem is defined on a square domain [0,1]^2 with Dirichlet boundary conditions.

The mixed formulation uses:
- Λ2: 2-forms for the flux
- Λ3: 3-forms for the potential
- Λ0: 0-forms for quadrature

The script demonstrates:
1. Solution of the 2D Poisson equation using mixed formulation
2. Convergence analysis with respect to:
   - Number of elements (n)
   - Polynomial degree (p)
3. JIT compilation speedup comparison
4. Error and timing analysis

The exact solution is given by:
u(x,y) = sin(2πx) * sin(2πy)
with source term:
f(x,y) = 2*(2π)^2 * u(x,y)
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time
import os

from mrx.DifferentialForms import DifferentialForm, DiscreteFunction
from mrx.Quadrature import QuadratureRule
from mrx.Projectors import Projector
from mrx.LazyMatrices import LazyMassMatrix, LazyDerivativeMatrix
from mrx.Utils import l2_product
from functools import partial

# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)

# Create output directory for figures
os.makedirs('script_outputs', exist_ok=True)


@partial(jax.jit, static_argnames=["n", "p"])
def get_err(n, p):
    """
    Compute error for mixed Poisson problem.

    Args:
        n: Number of elements in each direction
        p: Polynomial degree

    Returns:
        float: Relative L2 error of the solution
    """
    # Set up finite element spaces
    ns = (n, n, 1)
    ps = (p, p, 0)
    types = ('clamped', 'clamped', 'constant')

    # Define exact solution and source term
    def u(x):
        """Exact solution of the Poisson problem."""
        r, χ, z = x
        return jnp.ones(1) * jnp.sin(2 * jnp.pi * r) * jnp.sin(2 * jnp.pi * χ)

    def f(x):
        """Source term of the Poisson problem."""
        return 2 * (2*jnp.pi)**2 * u(x)

    # Set up differential forms and quadrature
    Λ0 = DifferentialForm(0, ns, ps, types)
    Λ2 = DifferentialForm(2, ns, ps, types)
    Λ3 = DifferentialForm(3, ns, ps, types)
    Q = QuadratureRule(Λ0, 3)

    # Set up operators
    D = LazyDerivativeMatrix(Λ2, Λ3, Q).M
    M2 = LazyMassMatrix(Λ2, Q).M

    # Add small regularization to prevent singular matrices
    reg = 1e-10 * jnp.eye(M2.shape[0])
    M2_reg = M2 + reg

    # Solve the system
    K = D @ jnp.linalg.solve(M2_reg, D.T)
    P3 = Projector(Λ3, Q)
    u_hat = jnp.linalg.solve(K, P3(f))
    u_h = DiscreteFunction(u_hat, Λ3)

    # Compute error using Λ3 quadrature
    def err(x): return u(x) - u_h(x)
    return (l2_product(err, err, Q) / l2_product(u, u, Q))**0.5


def run_convergence_analysis():
    """Run convergence analysis for different parameters."""
    # Parameter ranges
    ns = np.arange(7, 21, 2)
    ps = np.arange(1, 4)

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
    # Create figures
    figures = []

    # Error convergence plot
    fig1 = plt.figure(figsize=(10, 6))
    for j, p in enumerate(ps):
        plt.loglog(ns, err[:, j],
                   label=f'p={p}',
                   marker='o')
    # Add theoretical convergence rates
    plt.loglog(ns, err[-1, 0] * (ns/ns[-1])**(-1),
               label='O(n^-1)', linestyle='--')
    plt.loglog(ns, err[-1, 1] * (ns/ns[-1])**(-2),
               label='O(n^-2)', linestyle='--')
    plt.loglog(ns, err[-1, 2] * (ns/ns[-1])**(-4),
               label='O(n^-4)', linestyle='--')
    plt.xlabel('Number of elements (n)')
    plt.ylabel('Relative L2 error')
    plt.title('Error Convergence')
    plt.grid(True)
    plt.legend()
    figures.append(fig1)
    plt.savefig('script_outputs/2d_poisson_mixed_error.png',
                dpi=300, bbox_inches='tight')

    # Timing plot (first run)
    fig2 = plt.figure(figsize=(10, 6))
    for j, p in enumerate(ps):
        plt.loglog(ns, times[:, j],
                   label=f'p={p}',
                   marker='o')
    plt.loglog(ns, times[0, 0] * (ns/ns[0])**(4),
               label='O(n^4)', linestyle='--')
    plt.xlabel('Number of elements (n)')
    plt.ylabel('Computation time (s)')
    plt.title('Timing (First Run)')
    plt.grid(True)
    plt.legend()
    figures.append(fig2)
    plt.savefig('script_outputs/2d_poisson_mixed_time1.png',
                dpi=300, bbox_inches='tight')

    # Timing plot (second run)
    fig3 = plt.figure(figsize=(10, 6))
    for j, p in enumerate(ps):
        plt.loglog(ns, times2[:, j],
                   label=f'p={p}',
                   marker='o')
    plt.xlabel('Number of elements (n)')
    plt.ylabel('Computation time (s)')
    plt.title('Timing (Second Run)')
    plt.grid(True)
    plt.legend()
    figures.append(fig3)
    plt.savefig('script_outputs/2d_poisson_mixed_time2.png',
                dpi=300, bbox_inches='tight')

    # Speedup plot
    fig4 = plt.figure(figsize=(10, 6))
    for j, p in enumerate(ps):
        speedup = times[:, j] / times2[:, j]
        plt.semilogy(ns, speedup,
                     label=f'p={p}',
                     marker='o')
    plt.xlabel('Number of elements (n)')
    plt.ylabel('Speedup factor')
    plt.title('JIT Compilation Speedup')
    plt.grid(True)
    plt.legend()
    figures.append(fig4)
    plt.savefig('script_outputs/2d_poisson_mixed_speedup.png',
                dpi=300, bbox_inches='tight')

    return figures


def main():
    """Main function to run the analysis."""
    # Run convergence analysis
    err, times, times2 = run_convergence_analysis()

    # Plot results
    ns = np.arange(7, 21, 2)
    ps = np.arange(1, 4)
    plot_results(err, times, times2, ns, ps)

    # Show all figures
    plt.show()

    # Clean up
    plt.close('all')


if __name__ == "__main__":
    main()
