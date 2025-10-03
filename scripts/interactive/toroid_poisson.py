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
    types = ("clamped", "periodic", "constant")  # Types

    # Domain parameters
    a = 1/3
    R0 = 1.0
    π = jnp.pi

    def _X(r, χ):
        return jnp.ones(1) * (R0 + a * r * jnp.cos(2 * π * χ))

    def _Y(r, χ):
        return jnp.ones(1) * (R0 + a * r * jnp.cos(2 * π * χ))

    def _Z(r, χ):
        return jnp.ones(1) * a * r * jnp.sin(2 * π * χ)

    def F(x):
        """Polar coordinate mapping function."""
        r, χ, z = x
        return jnp.ravel(jnp.array([_X(r, χ) * jnp.cos(2 * π * z),
                                    -_Y(r, χ) * jnp.sin(2 * π * z),
                                    _Z(r, χ)]))

    # Define exact solution and source term
    def u(x):
        """Exact solution of the Poisson problem."""
        r, χ, z = x
        # return - jnp.ones(1) * r**2 * (1 - r**2)
        return - a**2 / (4 * jnp.pi) * jnp.ones(1) * jnp.sin(π * r**2)

    def f(x):
        """Source term of the Poisson problem."""
        r, χ, z = x
        c = jnp.cos(2 * π * χ)
        R = R0 + a * r * c
        # return 1 / (a**2 * R) * (4*R0*(1 - 4*r**2) + 2*a*r*c*(3 - 10*r**2)) * jnp.ones(1)
        return (jnp.cos(π*r**2) * (1 + (R - R0) / R / 2)
                - π * r**2 * jnp.sin(π*r**2)) * jnp.ones(1)

    # Create DeRham sequence
    derham = DeRhamSequence(ns, ps, q, types, F, polar=True)

    # Get stiffness matrix and projector
    K = derham.assemble_gradgrad_0()  # Stiffness matrix
    P0 = derham.P0_0  # Projector for 0-forms

    # Solve the system
    u_hat = jnp.linalg.solve(K, P0(f))
    u_h = DiscreteFunction(u_hat, derham.Λ0, derham.E0_0.matrix())

    # Compute error
    def err(x):
        return u(x) - u_h(x)
    error = (l2_product(err, err, derham.Q, F) /
             l2_product(u, u, derham.Q, F)) ** 0.5
    return error, jnp.linalg.cond(K), jnp.sum(jnp.abs(K) > 1e-12) / K.size


def run_convergence_analysis(ns, ps):
    """Run convergence analysis for different parameters."""

    # Arrays to store results
    err = np.zeros((len(ns), len(ps)))
    times = np.zeros((len(ns), len(ps)))
    sparsities = np.zeros((len(ns), len(ps)))
    conds = np.zeros((len(ns), len(ps)))

    # First run (with JIT compilation)
    print("First run (with JIT compilation):")
    for i, n in enumerate(ns):
        for j, p in enumerate(ps):
            start = time.time()
            _err, cond, sparsity = get_err(n, p)
            err[i, j] = _err
            conds[i, j] = cond
            sparsities[i, j] = sparsity
            end = time.time()
            times[i, j] = end - start
            print(
                f"n={n}, p={p}, err={err[i, j]:.2e}, time={times[i, j]:.2f}s, cond={conds[i, j]:.2e}, sparsity={sparsities[i, j]:.2e}")

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

    return err, times, times2, conds, sparsities


def plot_results(err, times, times2, conds, sparsities, ns, ps):

    # --- Figure settings ---
    FIG_SIZE = (12, 6)
    SQUARE_FIG_SIZE = (8, 8)
    TITLE_SIZE = 20
    LABEL_SIZE = 20
    TICK_SIZE = 16
    LINE_WIDTH = 2.5
    LEGEND_SIZE = 16

    colors = ['purple', 'teal', 'black']

    fig1, ax1 = plt.subplots(figsize=FIG_SIZE)

    ax1.set_title("Error Convergence", fontsize=TITLE_SIZE)
    ax1.set_xlabel(r'$n$', fontsize=LABEL_SIZE)
    ax1.set_ylabel(r'$\| f - f_h \|_{L^2(\Omega)}$',
                   fontsize=LABEL_SIZE)
    for j, p in enumerate(ps):
        ax1.loglog(ns, err[:, j],
                   label=f'p={p}',
                   marker='o')
    # Add theoretical convergence rates
    for j, p in enumerate(ps):
        expected_rate = -(p+1)
        ax1.loglog(ns[-3:], err[-1, j] * (ns[-3:]/ns[-1])**(-expected_rate),
                   label=f'O(n^{-expected_rate})', linestyle='--')
    ax1.set_xlabel(r'$n$')
    ax1.set_ylabel(r'$\| f - f_h \|_{L^2(\Omega)}$')
    ax1.grid(which="both", linestyle="--", linewidth=0.5)
    ax1.legend(fontsize=LEGEND_SIZE)
    ax1.tick_params(axis='y', labelsize=TICK_SIZE)
    ax1.tick_params(axis='x', labelsize=TICK_SIZE)
    fig1.tight_layout()
    plt.savefig(
        'script_outputs/2d_toroid_poisson_mixed_convergence.pdf', bbox_inches='tight')

    # Plot Energy on the left y-axis (ax1)
    color1 = 'purple'
    ax1.set_xlabel(r'$n$', fontsize=LABEL_SIZE)
    ax1.set_ylabel(r'$\frac{1}{2} \| B \|^2$',
                   color=color1, fontsize=LABEL_SIZE)
    for j, p in enumerate(ps):
        ax1.plot(ns, err[:, j],
                 label=f'p={p}',
                 marker='o')
    ax1.set_xscale('log')
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=TICK_SIZE)
    ax1.tick_params(axis='x', labelsize=TICK_SIZE)  # Set x-tick size

    ax1.grid(which="both", linestyle="--", linewidth=0.5)
    fig1.tight_layout()
    plt.show()

    # Timing plot
    fig2 = plt.figure(figsize=(10, 6))
    for j, p in enumerate(ps):
        plt.semilogy(ns, times[:, j],
                     label=f'p={p} (1st run)',
                     marker='o')
        plt.semilogy(ns, times2[:, j],
                     label=f'p={p} (2nd run)',
                     marker='x')
    plt.xlabel('Number of elements (n)')
    plt.ylabel('Time (s)')
    plt.title('Computation Time')
    plt.grid(True)
    plt.legend()
    plt.savefig('script_outputs/2d_toroid_poisson_mixed_time.png',
                dpi=300, bbox_inches='tight')

    # sparsity plot
    fig3 = plt.figure(figsize=(10, 6))
    for j, p in enumerate(ps):
        plt.semilogy(ns, sparsities[:, j],
                     label=f'p={p}',
                     marker='o')
    plt.xlabel('Number of elements (n)')
    plt.ylabel('Sparsity')
    plt.title('Matrix Sparsity')
    plt.grid(True)
    plt.legend()
    plt.savefig('script_outputs/2d_toroid_poisson_mixed_sparsity.png',
                dpi=300, bbox_inches='tight')

    # condition number plot
    fig4 = plt.figure(figsize=(10, 6))
    for j, p in enumerate(ps):
        plt.semilogy(ns, conds[:, j],
                     label=f'p={p}',
                     marker='o')
    plt.xlabel('Number of elements (n)')
    plt.ylabel('Condition Number')
    plt.title('Matrix Condition Number')
    plt.grid(True)
    plt.legend()
    plt.savefig('script_outputs/2d_toroid_poisson_mixed_condition_number.png',
                dpi=300, bbox_inches='tight')

    return fig1


def main():
    """Main function to run the analysis."""
    # Run convergence analysis
    ns = np.arange(8, 16, 2)
    ps = np.arange(1, 4)
    err, times, times2, conds, sparsities = run_convergence_analysis(ns, ps)

    # Plot results
    plot_results(err, times, times2, conds, sparsities, ns, ps)

    # Show all figures
    plt.show()

    # Clean up
    plt.close('all')


if __name__ == "__main__":
    main()

# %%
