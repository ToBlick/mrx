# %%
"""
2D Scalar Poisson Problem in Polar Coordinates

This script solves a 2D Poisson problem in polar coordinates using mixed finite element methods.
The problem is defined on a polar domain with homogeneous Neumann boundary conditions.

The exact solution is given by:
u(r, θ) = -(1/16)r⁴ + (1/12)r³ + 1/48
such that 
∂u/∂r = (r³ - r²)/4
and u(1, θ) = ∂u/∂r(1, θ) = 0 and
with source term:
f(r, θ) = r² - (3/4)r
"""

import os
import time 
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction, Pushforward
from mrx.mappings import polar_map

# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)

# Create output directory for figures
os.makedirs("script_outputs", exist_ok=True)

# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)
# Create output directory for figures
os.makedirs("script_outputs", exist_ok=True)

@partial(jax.jit, static_argnames=["n", "p", "q"])
def get_err(n, p, q):
    """
    Compute the error in the solution of the Poisson problem.
    We define this function that does assembly, solves the system, 
    and computes the error.
    It is JIT-compiled separately for different values of n, p, and q.

    Args:
        n: Number of elements in each direction
        p: Polynomial degree
        q: Quadrature order

    Returns:
        float: Relative L2 error of the solution
    """
    F = polar_map()

    # Define exact solution and source term
    def u(x):
        """Exact solution of the Poisson problem. Formula is:
        
        u(r, θ, z) = -(1/16)r⁴ + (1/12)r³ + 1/48

        Args:
            x: Input logical coordinates (r, θ, z)

        Returns:
            u: Exact solution of the Poisson equation
        """
        r, _, _ = x  # solution is independent of θ and z
        return -jnp.ones(1) * (r**4/16 - r**3/12 + 1/48)

    def f(x):
        """Source term of the Poisson problem. Formula is:
        
        f(r, θ, z) = r(r - 3/4)

        Args:
            x: Input logical coordinates (r, θ, z)

        Returns:
            f: Source term of the Poisson equation
        """
        r, _, _ = x  # source is independent of θ and z
        return jnp.ones(1) * (r - 3 / 4) * r

    # Set up finite element spaces
    ns = (n, n, 1)
    ps = (p, p, 0)
    types = ("clamped", "periodic", "constant")
    Seq = DeRhamSequence(ns, ps, q, types, F, polar=True, dirichlet=False)
    Seq.evaluate_1d()   # Precompute 1D basis functions at quadrature points
    Seq.assemble_M2()   # Assemble 2-form mass matrix
    Seq.assemble_M3()   # Assemble 3-form mass matrix
    Seq.assemble_d2()   # Assemble strong divergence and weak gradient
    Seq.assemble_dd3()  # Assemble 3-form Laplacian - strong_div o weak_grad

    # Solve the system
    u_dof = jnp.linalg.solve(Seq.M3 @ Seq.dd3, Seq.P3(f))
    # The solution will satisfy u = 0 on the boundary
    u_h = Pushforward(DiscreteFunction(u_dof, Seq.Λ3, Seq.E3), F, 3)

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
    plt.savefig("script_outputs/mixed_polar_poisson_error.pdf",
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
    plt.savefig("script_outputs/mixed_polar_poisson_time1.pdf",
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
    plt.savefig("script_outputs/mixed_polar_poisson_time2.pdf",
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
        "script_outputs/mixed_polar_poisson_speedup.pdf", dpi=300, bbox_inches="tight"
    )

    return figures


def main():
    """Main function to run the analysis."""
    # Run convergence analysis
    ns = np.arange(6, 17, 2)
    ps = np.arange(1, 5)
    err, times, times2 = run_convergence_analysis(ns, ps)
    # Plot results
    plot_results(err, times, times2, ns, ps)
    # Show all figures
    plt.show()

    # Clean up
    plt.close("all")


if __name__ == "__main__":
    main()
