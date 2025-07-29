# %%
"""
2D Poisson Problem in Polar Coordinates

This script solves a 2D Poisson problem in polar coordinates using finite element methods.
The problem is defined on a polar domain with Dirichlet boundary conditions.

The script demonstrates:
1. Solution of the Poisson equation in polar coordinates
2. Convergence analysis with respect to:
   - Number of elements (n)
   - Polynomial degree (p)
   - Quadrature order (q)
3. JIT compilation speedup comparison
4. Error and timing analysis

The exact solution is given by:
u(r, χ) = r³(3log(r) - 2)/27 + 2/27
with source term:
f(r, χ) = -r log(r)

Note that the solution u is not smooth, we only have u ∈ H^s(Ω) for all s < 4. 
This limits the order of convergence we can expect to see.
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
os.makedirs("script_outputs", exist_ok=True)

# %%


@partial(jax.jit, static_argnames=["n", "p", "q"])
def get_err(n, p, q):
    """
    Compute the error in the solution of the Poisson problem.

    Args:
        n: Number of elements in each direction
        p: Polynomial degree
        q: Quadrature order

    Returns:
        float: Relative L2 error of the solution
    """
    # Domain parameters
    a = 1
    R0 = 3.0
    Y0 = 0.0

    def F(x):
        """Polar coordinate mapping function."""
        r, χ, z = x
        return jnp.array([R0 + a * r * jnp.cos(2 * jnp.pi * χ),
                          -z,
                          Y0 + a * r * jnp.sin(2 * jnp.pi * χ)])

    # Define exact solution and source term
    def u(x):
        """Exact solution of the Poisson problem."""
        r, χ, z = x
        return jnp.ones(1) * r**3 * (3 * jnp.log(r) - 2) / 27 + 2 / 27

    def f(x):
        """Source term of the Poisson problem."""
        r, χ, z = x
        return -jnp.ones(1) * r * jnp.log(r)

    # Set up finite element spaces
    ns = (n, n, 1)
    ps = (p, p, 0)
    types = ("clamped", "periodic", "constant")
    bcs = ('dirichlet', 'none', 'none')

    Seq = DeRhamSequence(ns, ps, q, types, bcs, F, polar=True)
    K = Seq.assemble_gradgrad()

    # Solve the system
    u_hat = jnp.linalg.solve(K, Seq.P0(f))
    u_h = DiscreteFunction(u_hat, Seq.Λ0, Seq.E0.matrix())

    # Compute error
    def err(x):
        return u(x) - u_h(x)

    error = (l2_product(err, err, Seq.Q, F) /
             l2_product(u, u, Seq.Q, F)) ** 0.5
    return error


def run_convergence_analysis():
    """Run convergence analysis for different parameters."""
    # Parameter ranges
    ns = np.arange(6, 18, 2)
    ps = np.arange(1, 5)
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
                    f"n={n}, p={p}, q={q}, err={err[i, j, k]:.2e}, time={times[i, j, k]:.2f}s"
                )

    # Second run (after JIT compilation)
    print("\nSecond run (after JIT compilation):")
    times2 = np.zeros((len(ns), len(ps), len(qs)))
    for i, n in enumerate(ns):
        for j, p in enumerate(ps):
            for k, q in enumerate(qs):
                start = time.time()
                err[i, j, k] = get_err(n, p, q)
                end = time.time()
                times2[i, j, k] = end - start
                print(f"n={n}, p={p}, q={q}, time={times2[i, j, k]:.2f}s")

    return err, times, times2


def plot_results(err, times, times2, ns, ps, qs):
    """Plot the results of the convergence analysis."""
    # Create figures
    figures = []

    # Error convergence plot
    fig1 = plt.figure(figsize=(10, 6))
    for j, p in enumerate(ps):
        for k, q in enumerate(qs):
            plt.loglog(ns, err[:, j, k], label=f"p={p}, q={q}", marker="o")
    plt.xlabel("Number of elements (n)")
    plt.ylabel("Relative L2 error")
    plt.title("Error Convergence")
    plt.grid(True)
    plt.legend()
    figures.append(fig1)
    plt.savefig("script_outputs/polar_poisson_error.png",
                dpi=300, bbox_inches="tight")

    # Timing plot (first run)
    fig2 = plt.figure(figsize=(10, 6))
    for j, p in enumerate(ps):
        for k, q in enumerate(qs):
            plt.loglog(ns, times[:, j, k], label=f"p={p}, q={q}", marker="o")
    plt.xlabel("Number of elements (n)")
    plt.ylabel("Computation time (s)")
    plt.title("Timing (First Run)")
    plt.grid(True)
    plt.legend()
    figures.append(fig2)
    plt.savefig("script_outputs/polar_poisson_time1.png",
                dpi=300, bbox_inches="tight")

    # Timing plot (second run)
    fig3 = plt.figure(figsize=(10, 6))
    for j, p in enumerate(ps):
        for k, q in enumerate(qs):
            plt.loglog(ns, times2[:, j, k], label=f"p={p}, q={q}", marker="o")
    plt.xlabel("Number of elements (n)")
    plt.ylabel("Computation time (s)")
    plt.title("Timing (Second Run)")
    plt.grid(True)
    plt.legend()
    figures.append(fig3)
    plt.savefig("script_outputs/polar_poisson_time2.png",
                dpi=300, bbox_inches="tight")

    # Speedup plot
    fig4 = plt.figure(figsize=(10, 6))
    for j, p in enumerate(ps):
        for k, q in enumerate(qs):
            speedup = times[:, j, k] / times2[:, j, k]
            plt.semilogy(ns, speedup, label=f"p={p}, q={q}", marker="o")
    plt.xlabel("Number of elements (n)")
    plt.ylabel("Speedup factor")
    plt.title("JIT Compilation Speedup")
    plt.grid(True)
    plt.legend()
    figures.append(fig4)
    plt.savefig(
        "script_outputs/polar_poisson_speedup.png", dpi=300, bbox_inches="tight"
    )

    return figures


def main():
    """Main function to run the analysis."""
    # Run convergence analysis
    err, times, times2 = run_convergence_analysis()

    # Plot results
    ns = np.arange(6, 18, 2)
    ps = np.arange(1, 5)
    qs = np.arange(4, 11, 3)
    plot_results(err, times, times2, ns, ps, qs)

    # Show all figures
    plt.show()

    # Clean up
    plt.close("all")


if __name__ == "__main__":
    main()
