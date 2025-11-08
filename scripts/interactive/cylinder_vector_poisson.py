# %%
# TODO: test or delete

import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import cylinder_map

jax.config.update("jax_enable_x64", True)
script_dir = Path(__file__).parent / 'script_outputs'
script_dir.mkdir(parents=True, exist_ok=True)

def get_err(n : int, p : int) -> float:
    """
    Compute the error in the solution of a vector Poisson problem in 3D.
    We define this function that does assembly, solves the system, and computes the error.
    It is JIT-compiled separately for different values of n, p.

    Args:
        n: Number of elements in each direction
        p: Polynomial degree

    Returns:
        float: Relative L2 error of the solution
    """
    # Set up finite element spaces
    q = 2 * p
    ns = (n, n, n)
    ps = (p, p, p)
    types = ("clamped", "periodic", "periodic")

    # Domain parameters
    a = 1
    h = 1
    π = jnp.pi
    F = cylinder_map(a=a, h=h)

    # Define exact solution and source term
    def u(x : jnp.ndarray) -> jnp.ndarray:
        """Exact solution of the Poisson problem. Formula is:
        
        u(r, χ, z) = (0, r² (1 - r)² cos(2πz), 0), 
        and is independent of χ.

        Args:
            x: Input logical coordinates (r, χ, z)

        Returns:
            u: Exact solution of the vector Poisson problem given the source term defined below.
        """
        r, χ, z = x
        u_theta = r**2 * (1 - r)**2 * jnp.cos(2*π*z)
        return jnp.array([0, u_theta, 0])

    def f(x : jnp.ndarray) -> jnp.ndarray:
        """Source term of the Poisson problem. Formula is:
        
        f(r, χ, z) = (0, 4π² r² (1 - r)² cos(2πz) - (3 - 16r + 15r²) cos(2πz), 0),
        and is independent of χ.

        Args:
            x: Input logical coordinates (r, χ, z)

        Returns:
            f: Source term of the vector Poisson problem.
        """
        r, χ, z = x
        f_theta = (r**2 * (1 - r)**2 * 4*π**2 - (3 - 16*r + 15*r**2)) * jnp.cos(2*π*z)
        return jnp.array([0, f_theta, 0])

    # Create DeRham sequence
    derham = DeRhamSequence(ns, ps, q, types, F, polar=False, dirichlet=True)
    derham.evaluate_1d()
    derham.assemble_M0()
    derham.assemble_M1()
    derham.assemble_M2()

    # Curl operator TODO: should this be strong or weak? 
    derham.assemble_d1()
    C = derham.strong_curl

    # Double divergence operator on 2-forms
    derham.assemble_dd2()  # dd2 = divdiv + strong_curl weak_curl
    divdiv = derham.M2 @ (derham.dd2 - derham.strong_curl @ derham.weak_curl)

    # Mass matrix for 1-forms
    derham.assemble_M1()
    M1 = derham.M1

    # Mass matrix for 2-forms
    derham.assemble_M2()
    M2 = derham.M2

    # block_matrix = jnp.block([[K, C], [-C.T, M1]])
    L = C @ jnp.linalg.solve(M1, C.T) + divdiv

    # Solve the generalized eigenvalue problem
    tol = 1e-12
    eigvals, eigvecs = jnp.linalg.eigh(L)
    inv_eigvals = jnp.where(
        jnp.abs(eigvals) > tol,
        1.0 / eigvals,
        0.0
    )
    L_pinv = (eigvecs * inv_eigvals) @ eigvecs.T

    # Project source term onto 2-form space
    f_proj = derham.P2(f)
    u_hat = L_pinv @ f_proj

    # Project exact solution onto 2-form space for error computation
    u_proj = derham.P2(u)

    u_hat_analytic = jnp.linalg.solve(M2, u_proj)
    error = ((u_hat - u_hat_analytic) @ M2 @ (u_hat - u_hat_analytic) /
             (u_hat_analytic @ M2 @ u_hat_analytic))**0.5
    return error


def run_convergence_analysis(ns : list[int], ps : list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    # First run (without JIT compilation)
    print("First run (without JIT compilation):")
    for i, n in enumerate(ns):
        for j, p in enumerate(ps):
            start = time.time()
            err[i, j] = get_err(n, p)
            end = time.time()
            times[i, j] = end - start
            print(f"n={n}, p={p}, err={err[i, j]:.2e}, time={times[i, j]:.2f}s")

    # Second run (after first compilation)
    print("\nSecond run (after first compilation):")
    times2 = np.zeros((len(ns), len(ps)))
    for i, n in enumerate(ns):
        for j, p in enumerate(ps):
            start = time.time()
            _ = get_err(n, p)  # We don't need to store the error again
            end = time.time()
            times2[i, j] = end - start
            print(f"n={n}, p={p}, time={times2[i, j]:.2f}s")

    return err, times, times2


def plot_results(err : np.ndarray, ns : list[int], ps : list[int]) -> plt.Figure:
    """Plot the results of the convergence analysis.
    
    Args:
        err : np.ndarray
            Array of relative L2 errors
        ns : list[int]
            List of number of elements in each direction
        ps : list[int]
            List of polynomial degrees
    
    Returns:
        fig1 : plt.Figure
            Figure object
    """
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
    plt.savefig(script_dir / 'cylinder_vector_poisson_error.png',
                dpi=300, bbox_inches='tight')

    return fig1


def main():
    """Main function to run the analysis."""
    # Run convergence analysis
    ns = np.arange(4, 8, 2)
    ps = np.arange(1, 4)
    err, _, _ = run_convergence_analysis(ns, ps)

    # Plot results
    plot_results(err, ns, ps)

    # Show all figures
    plt.show()

    # Clean up
    plt.close('all')


if __name__ == "__main__":
    main()
