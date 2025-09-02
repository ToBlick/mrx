# %%
"""
3D Poisson Problem with Dirichlet Boundary Conditions

This script solves a 3D Poisson problem using finite element methods with Dirichlet boundary conditions.
The problem is defined on a cubic domain [0,1]^3.

The script demonstrates:
1. Solution of the 3D Poisson equation
2. Convergence analysis with respect to:
   - Number of elements (n)
   - Polynomial degree (p)
3. JIT compilation speedup comparison
4. Error and timing analysis

The exact solution is given by:
u(x,y,z) = sin(2πx) * sin(2πy) * sin(2πz)
with source term:
f(x,y,z) = 3*(2π)^2 * u(x,y,z)
"""

import os
import time
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from mrx.DeRhamSequence import DeRhamSequence
from mrx.DifferentialForms import DiscreteFunction
from mrx.Utils import l2_product

# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)

# Create output directory for figures
os.makedirs('script_outputs', exist_ok=True)


@partial(jax.jit, static_argnames=['n', 'p'])
def get_err(n, p):
    """
    Compute the error in the solution of the 3D Poisson problem.

    Args:
        n: Number of elements in each direction
        p: Polynomial degree

    Returns:
        float: Relative L2 error of the solution
    """
    # Set up finite element spaces
    ns = (n, n, n)
    ps = (p, p, p)
    types = ('clamped', 'clamped', 'clamped')
    bcs = ('dirichlet', 'dirichlet', 'dirichlet')
    q = p

    # Define exact solution and source term
    def u(x):
        """Exact solution of the Poisson problem."""
        r, χ, z = x
        return jnp.ones(1) * jnp.sin(2 * jnp.pi * r) * jnp.sin(2 * jnp.pi * χ) * jnp.sin(2 * jnp.pi * z)

    def f(x):
        """Source term of the Poisson problem."""
        return 3 * (2*jnp.pi)**2 * u(x)

    # Set up operators and solve system
    Seq = DeRhamSequence(ns, ps, q, types, bcs, lambda x: x, polar=False)
    K = Seq.assemble_gradgrad()

    # Solve the system
    u_hat = jnp.linalg.solve(K, Seq.P0(f))
    u_h = DiscreteFunction(u_hat, Seq.Λ0, Seq.E0.matrix())

    # Compute error
    def err(x): return u(x) - u_h(x)
    return (l2_product(err, err, Seq.Q) / l2_product(u, u, Seq.Q))**0.5


def run_convergence_analysis():
    """Run convergence analysis for different parameters."""
    # Parameter ranges
    # Extended range for higher resolution
    ns = np.arange(6, 11, 2)
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

    return err, times, times2, ns, ps


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
        plt.loglog(ns, err[-1, j] * (ns/ns[-1])**(-2*p),
                   label=f'O(n^-{2*p})', linestyle='--')

    plt.xlabel('Number of elements (n)')
    plt.ylabel('Relative L2 error')
    plt.title('Error Convergence')
    plt.grid(True)
    plt.legend()
    figures.append(fig1)
    plt.savefig('script_outputs/3d_poisson_error.png',
                dpi=300, bbox_inches='tight')

    # Timing plot (first run)
    fig2 = plt.figure(figsize=(10, 6))
    for j, p in enumerate(ps):
        plt.loglog(ns, times[:, j],
                   label=f'p={p}',
                   marker='o')
    plt.loglog(ns, times[-1, 0] * (ns/ns[-1])**(2*3),
               label='O(n^2d)', linestyle='--')
    plt.xlabel('Number of elements (n)')
    plt.ylabel('Computation time (s)')
    plt.title('Timing (First Run)')
    plt.grid(True)
    plt.legend()
    figures.append(fig2)
    plt.savefig('script_outputs/3d_poisson_time1.png',
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
    plt.savefig('script_outputs/3d_poisson_time2.png',
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
    plt.savefig('script_outputs/3d_poisson_speedup.png',
                dpi=300, bbox_inches='tight')

    return figures


def main():
    """Main function to run the analysis."""
    # Run convergence analysis
    err, times, times2, ns, ps = run_convergence_analysis()

    # Plot results
    plot_results(err, times, times2, ns, ps)

    # Show all figures
    plt.show()

    # Clean up
    plt.close('all')


if __name__ == "__main__":
    main()

# %%
