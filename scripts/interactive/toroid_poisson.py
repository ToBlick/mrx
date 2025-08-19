# %%
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


@partial(jax.jit, static_argnames=["n", "p"])
def get_err(n, p):
    # Set up finite element spaces
    q = 2*p
    ns = (n, n, 1)
    ps = (p, p, 0)
    types = ("clamped", "periodic", "constant") # Types
    bcs = ("dirichlet", "periodic", "constant")  # Boundary conditions

    # Domain parameters
    a = 1.13
    R0 = 3.46
    π = jnp.pi

    def _X(r, χ):
        return jnp.ones(1) * (R0 + a * r * jnp.cos(2 * π * χ))

    def _Z(r, χ):
        return jnp.ones(1) * (R0 + a * r * jnp.cos(2 * π * χ))

    def _Y(r, χ):
        return jnp.ones(1) * a * r * jnp.sin(2 * π * χ)

    def F(x):
        """Polar coordinate mapping function."""
        r, χ, z = x
        return jnp.ravel(jnp.array([_X(r, χ) * jnp.cos(2 * π * z),
                                    _Y(r, χ),
                                    _Z(r, χ) * jnp.sin(2 * π * z)]))
    # Define exact solution and source term

    def u(x):
        """Exact solution of the Poisson problem."""
        r, χ, z = x
        # return - jnp.ones(1) * r**2 * (1 - r**2)
        return - jnp.ones(1) * jnp.sin(π * r**2)

    def f(x):
        """Source term of the Poisson problem."""
        r, χ, z = x
        c = jnp.cos(2 * π * χ)
        R = R0 + a * r * c
        # return 1 / (a**2 * R) * (4*R0*(1 - 4*r**2) + 2*a*r*c*(3 - 10*r**2)) * jnp.ones(1)
        return 4 * π / a**2 * (jnp.cos(π*r**2) * (1 + (R - R0) / R / 2)
                               - π * r**2 * jnp.sin(π*r**2)) * jnp.ones(1)
    
    # Create DeRham sequence
    derham = DeRhamSequence(ns, ps, q, types, bcs, F, polar=True)
    
    # Get stiffness matrix and projector
    K = derham.assemble_gradgrad()  # Stiffness matrix 
    P0 = derham.P0  # Projector for 0-forms
    
    # Solve the system
    u_hat = jnp.linalg.solve(K, P0(f))
    u_h = DiscreteFunction(u_hat, derham.Λ0, derham.E0.matrix())
    
    # Compute error
    def err(x):
        return u(x) - u_h(x)
    error = (l2_product(err, err, derham.Q, F) / l2_product(u, u, derham.Q, F)) ** 0.5
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
    plt.savefig('script_outputs/2d_toroid_poisson_mixed_error.png',
                dpi=300, bbox_inches='tight')

    return fig1


def main():
    """Main function to run the analysis."""
    # Run convergence analysis
    ns = np.arange(8, 23, 4)
    ps = np.arange(1, 5)
    err, times, times2 = run_convergence_analysis(ns, ps)

    # Plot results
    plot_results(err, times, times2, ns, ps)

    # Show all figures
    plt.show()

    # Clean up
    plt.close('all')


if __name__ == "__main__":
    main()

# %%
