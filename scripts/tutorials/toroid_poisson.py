# %%
"""
3D Scalar Poisson Problem in Toroidal Coordinates

This script solves a 3D scalar Poisson problem in toroidal coordinates.
The problem is defined on a toroidal domain with Dirichlet boundary conditions.

The exact solution is given by:
u(r, θ, ζ) = (r² - r⁴) cos(2πζ)
with source term:
f(r, θ, ζ) = cos(2πζ) (-4/ɛ² * (1 - 4r²) - 4/(ɛR) (r/2 - r³)cos(2πθ) + (r² - r⁴) / R²)
with R = 1 + ɛ r cos(2πθ).
"""
import os
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction
from mrx.mappings import toroid_map

# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)
# Create output directory for figures
os.makedirs("script_outputs", exist_ok=True)


@partial(jax.jit, static_argnames=["n", "p", "q"])
def get_err(n, p, q):
    """
    Compute the error in the solution of the Poisson problem.
    We define this function that does assembly, solves the system, and computes the error.
    It is JIT-compiled separately for different values of n, p, and q.

    Args:
        n: Number of elements in each direction
        p: Polynomial degree
        q: Quadrature order

    Returns:
        float: Relative L2 error of the solution
    """
    ɛ = 1/3
    π = jnp.pi
    F = toroid_map(epsilon=ɛ)

    # Define exact solution and source term
    def u(x):
        """Exact solution of the Poisson problem in logical coordinates. Solution is independent of θ. Formula is:

            u(r, z) = (r² - r⁴) cos(2πz)

        Args:   
            x: Input logical coordinates (r, χ, z)

        Returns:
            u: Exact solution of the Poisson equation
        """
        r, _, z = x
        return (r**2 - r**4) * jnp.cos(2 * π * z) * jnp.ones(1)

    def f(x):
        """Source term of the Poisson problem in logical coordinates. Formula is:

            f(r, z) = cos(2πz) (-4/ɛ² * (1 - 4r²) - 4/(ɛR) (r/2 - r³)cos(2πθ) + (r² - r⁴) / R²)

        Args:
            x: Input logical coordinates (r, χ, z)

        Returns:
            f: Source term of the Poisson equation
        """
        r, θ, z = x
        R = 1 + ɛ * r * jnp.cos(2 * jnp.pi * θ)
        return jnp.cos(2 * jnp.pi * z) * (-4/ɛ**2 * (1 - 4*r**2) - 4/(ɛ*R) * (r/2 - r**3) * jnp.cos(2 * jnp.pi * θ) + (r**2 - r**4) / R**2) * jnp.ones(1)

    # Set up finite element spaces
    ns = (n, n, n)
    ps = (p, p, p)
    types = ("clamped", "periodic", "periodic")
    Seq = DeRhamSequence(ns, ps, q, types, F, polar=True, dirichlet=True)
    Seq.evaluate_1d()
    Seq.assemble_M0()
    Seq.assemble_dd0()

    # Solve the system
    u_hat = jnp.linalg.solve(Seq.M0 @ Seq.dd0, Seq.P0(f))
    u_h = DiscreteFunction(u_hat, Seq.Lambda_0, Seq.E0)

    # Compute the L2 error
    def diff_at_x(x):
        return u(x) - u_h(x)
    df_at_x = jax.vmap(diff_at_x)(Seq.Q.x)
    f_at_x = jax.vmap(u)(Seq.Q.x)
    L2_df = jnp.einsum('ik,ik,i,i->', df_at_x, df_at_x, Seq.J_j, Seq.Q.w)**0.5
    L2_f = jnp.einsum('ik,ik,i,i->', f_at_x, f_at_x, Seq.J_j, Seq.Q.w)**0.5
    error = L2_df / L2_f
    return error


def run_convergence_analysis(ns, ps):
    """Run convergence analysis for different parameters.

    Args:
        ns: List of number of elements in each direction
        ps: List of polynomial degrees

    Returns:
        err: Array of relative L2 errors
        times: Array of computation times
        times2: Array of computation times for second run
    """
    import time

    # Arrays to store results
    err = np.zeros((len(ns), len(ps)))
    times = np.zeros((len(ns), len(ps)))

    # First run (with JIT compilation)
    print("First run (with JIT compilation):")
    for i, n in enumerate(ns):
        for j, p in enumerate(ps):
            q = p + 2  # Quadrature order
            start = time.time()
            err[i, j] = get_err(n, p, q)
            jax.block_until_ready(err[i, j])
            end = time.time()
            times[i, j] = end - start
            print(
                f"n={n}, p={p}, q={q}, err={err[i, j]:.2e}, time={times[i, j]:.2f}s"
            )

    # Second run (after JIT compilation)
    print("\nSecond run (after JIT compilation):")
    times2 = np.zeros((len(ns), len(ps)))
    for i, n in enumerate(ns):
        for j, p in enumerate(ps):
            q = p + 2  # Quadrature order
            start = time.time()
            err[i, j] = get_err(n, p, q)
            jax.block_until_ready(err[i, j])
            end = time.time()
            times2[i, j] = end - start
            print(f"n={n}, p={p}, q={q}, time={times2[i, j]:.2f}s")

    return err, times, times2


def plot_results(err, times, times2, ns, ps):
    """Plot the results of the convergence analysis.

    Args:
        err: Array of relative L2 errors
        times: Array of computation times
        times2: Array of computation times for second run
        ns: List of number of elements in each direction
        ps: List of polynomial degrees

    Returns:
        figures: List of figures
    """
    # Create figures
    figures = []

    # Error convergence plot
    fig1 = plt.figure(figsize=(10, 6))
    for j, p in enumerate(ps):
        plt.loglog(ns, err[:, j], label=f"p={p}", marker="o")
    plt.xlabel("Number of elements (n)")
    plt.ylabel("Relative L2 error")
    plt.title("Error Convergence")
    plt.grid(True)
    plt.legend()
    figures.append(fig1)
    plt.savefig("script_outputs/toroid_poisson_error.pdf",
                dpi=300, bbox_inches="tight")

    # Timing plot (first run)
    fig2 = plt.figure(figsize=(10, 6))
    for j, p in enumerate(ps):
        plt.loglog(ns, times[:, j], label=f"p={p}", marker="o")
    plt.xlabel("Number of elements (n)")
    plt.ylabel("Computation time (s)")
    plt.title("Timing (First Run)")
    plt.grid(True)
    plt.legend()
    figures.append(fig2)
    plt.savefig("script_outputs/toroid_poisson_time1.pdf",
                dpi=300, bbox_inches="tight")

    # Timing plot (second run)
    fig3 = plt.figure(figsize=(10, 6))
    for j, p in enumerate(ps):
        plt.loglog(ns, times2[:, j], label=f"p={p}", marker="o")
    plt.xlabel("Number of elements (n)")
    plt.ylabel("Computation time (s)")
    plt.title("Timing (Second Run)")
    plt.grid(True)
    plt.legend()
    figures.append(fig3)
    plt.savefig("script_outputs/toroid_poisson_time2.pdf",
                dpi=300, bbox_inches="tight")

    # Speedup plot
    fig4 = plt.figure(figsize=(10, 6))
    for j, p in enumerate(ps):
        speedup = times[:, j] / times2[:, j]
        plt.semilogy(ns, speedup, label=f"p={p}", marker="o")
    plt.xlabel("Number of elements (n)")
    plt.ylabel("Speedup factor")
    plt.title("JIT Compilation Speedup")
    plt.grid(True)
    plt.legend()
    figures.append(fig4)
    plt.savefig(
        "script_outputs/toroid_poisson_speedup.pdf", dpi=300, bbox_inches="tight"
    )

    return figures


def main():
    """Main function to run the analysis."""
    # Run convergence analysis
    ns = np.arange(4, 10, 2)
    ps = np.arange(1, 4)
    err, times, times2 = run_convergence_analysis(ns, ps)
    # Plot results
    plot_results(err, times, times2, ns, ps)
    # Show all figures
    plt.show()

    # Clean up
    plt.close("all")


if __name__ == "__main__":
    main()
