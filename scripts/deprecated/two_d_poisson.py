"""
2D Poisson Problem with Dirichlet Boundary Conditions

This script solves a 2D Poisson problem using finite element methods with Dirichlet boundary conditions.
The problem is defined on a square domain [0,1]^2.

The script demonstrates:
1. Solution of the 2D Poisson equation
2. Convergence analysis with respect to:
   - Number of elements (n)
   - Polynomial degree (p)
   - Quadrature order (q)
3. JIT compilation speedup comparison
4. Error and timing analysis

The exact solution is given by:
u(x,y) = sin(2πx) * sin(2πy)
with source term:
f(x,y) = 2*(2π)^2 * u(x,y)
"""

import os
import time
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction
from mrx.utils import l2_product

# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)

# Create output directory for figures
os.makedirs('script_outputs', exist_ok=True)


@partial(jax.jit, static_argnames=["n", "p", "q"])
def get_err(n, p, q):
    """
    Compute the error in the solution of the 2D Poisson problem.

    Args:
        n: Number of elements in each direction
        p: Polynomial degree
        q: Quadrature order

    Returns:
        float: Relative L2 error of the solution
    """
    # Set up finite element spaces
    ns = (n, n, 1)
    ps = (p, p, 0)
    types = ('clamped', 'clamped', 'constant')
    bcs = ('dirichlet', 'dirichlet', 'none')

    # Define exact solution and source term
    def u(x):
        """Exact solution of the Poisson problem."""
        r, χ, z = x
        return jnp.ones(1) * jnp.sin(2 * jnp.pi * r) * jnp.sin(2 * jnp.pi * χ)

    def f(x):
        """Source term of the Poisson problem."""
        return 2 * (2*jnp.pi)**2 * u(x)

    # Set up operators and solve system
    Seq = DeRhamSequence(ns, ps, q, types, bcs, lambda x: x, polar=False)

    # %%
    # Set up operators and solve system
    K = Seq.assemble_gradgrad()
    # %%
    # Solve the system
    u_hat = jnp.linalg.solve(K, Seq.P0(f))
    u_h = DiscreteFunction(u_hat, Seq.Λ0, Seq.E0.matrix())
    def err(x): return u(x) - u_h(x)
    return (l2_product(err, err, Seq.Q) / l2_product(u, u, Seq.Q)) ** 0.5


def run_convergence_analysis():
    """Run convergence analysis for different parameters."""
    # Parameter ranges
    ns = np.arange(4, 18, 2)
    ps = np.arange(1, 4)
    qs = np.arange(4, 11, 3)

    # Arrays to store results
    err = np.zeros((len(ns), len(ps), len(qs)))
    times = np.zeros((len(ns), len(ps), len(qs)))

    # First run (with JIT compilation)
    print("First run (with JIT compilation):")
    for i, n in enumerate(ns):
        for j, p in enumerate(ps):
            for k, q in enumerate(qs):
                start = time.time()
                err[i, j, k] = get_err(n, p, q)
                end = time.time()
                times[i, j, k] = end - start
                print(
                    f"n={n}, p={p}, q={q}, err={err[i, j, k]:.2e}, time={times[i, j, k]:.2f}s")

    # Second run (after JIT compilation)
    print("\nSecond run (after JIT compilation):")
    times2 = np.zeros((len(ns), len(ps), len(qs)))
    for i, n in enumerate(ns):
        for j, p in enumerate(ps):
            for k, q in enumerate(qs):
                start = time.time()
                _ = get_err(n, p, q)  # We don't need to store the error again
                end = time.time()
                times2[i, j, k] = end - start
                print(f"n={n}, p={p}, q={q}, time={times2[i, j, k]:.2f}s")

    return err, times, times2, ns, ps, qs


def plot_results(err, times, times2, ns, ps, qs):
    """Plot the results of the convergence analysis."""
    # Create figures
    figures = []

    # Error convergence plot
    fig1 = plt.figure(figsize=(10, 6))
    for j, p in enumerate(ps):
        for k, q in enumerate(qs):
            plt.loglog(ns, err[:, j, k],
                       label=f'p={p}, q={q}',
                       marker='o')
    plt.xlabel('Number of elements (n)')
    plt.ylabel('Relative L2 error')
    plt.title('Error Convergence')
    plt.grid(True)
    plt.legend()
    figures.append(fig1)
    plt.savefig('script_outputs/2d_poisson_error.png',
                dpi=300, bbox_inches='tight')

    # Timing plot (first run)
    fig2 = plt.figure(figsize=(10, 6))
    for j, p in enumerate(ps):
        for k, q in enumerate(qs):
            plt.loglog(ns, times[:, j, k],
                       label=f'p={p}, q={q}',
                       marker='o')
    plt.xlabel('Number of elements (n)')
    plt.ylabel('Computation time (s)')
    plt.title('Timing (First Run)')
    plt.grid(True)
    plt.legend()
    figures.append(fig2)
    plt.savefig('script_outputs/2d_poisson_time1.png',
                dpi=300, bbox_inches='tight')

    # Timing plot (second run)
    fig3 = plt.figure(figsize=(10, 6))
    for j, p in enumerate(ps):
        for k, q in enumerate(qs):
            plt.loglog(ns, times2[:, j, k],
                       label=f'p={p}, q={q}',
                       marker='o')
    plt.xlabel('Number of elements (n)')
    plt.ylabel('Computation time (s)')
    plt.title('Timing (Second Run)')
    plt.grid(True)
    plt.legend()
    figures.append(fig3)
    plt.savefig('script_outputs/2d_poisson_time2.png',
                dpi=300, bbox_inches='tight')

    # Speedup plot
    fig4 = plt.figure(figsize=(10, 6))
    for j, p in enumerate(ps):
        for k, q in enumerate(qs):
            speedup = times[:, j, k] / times2[:, j, k]
            plt.semilogy(ns, speedup,
                         label=f'p={p}, q={q}',
                         marker='o')
    plt.xlabel('Number of elements (n)')
    plt.ylabel('Speedup factor')
    plt.title('JIT Compilation Speedup')
    plt.grid(True)
    plt.legend()
    figures.append(fig4)
    plt.savefig('script_outputs/2d_poisson_speedup.png',
                dpi=300, bbox_inches='tight')

    return figures


def main():
    """Main function to run the analysis."""
    # Run convergence analysis
    err, times, times2, ns, ps, qs = run_convergence_analysis()

    # Plot results
    plot_results(err, times, times2, ns, ps, qs)

    # Show all figures
    plt.show()

    # Clean up
    plt.close('all')


if __name__ == "__main__":
    main()
