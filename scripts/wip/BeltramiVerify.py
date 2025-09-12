# %%
import os
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from mrx.DeRhamSequence import DeRhamSequence
from mrx.DifferentialForms import DiscreteFunction
from mrx.Utils import l2_product


# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)

# Create output directory for figures
os.makedirs("script_outputs", exist_ok=True)



@partial(jax.jit, static_argnames=["m_mode", "n_mode", "n", "p"])
def get_err(m_mode, n_mode, n, p):
    # Set up 

    q = 2*p
    ns = (n, n, n)
    ps = (p, p, p)
    types = ("periodic", "periodic", "periodic")
    bcs = ("periodic", "periodic", "periodic")

    # Beltrami parameter
    mu_val = jnp.pi * jnp.sqrt(m_mode**2 + n_mode**2)

    # Amplitude factor
    A_0 = 1.0
    
    def u_exact(x):
        """Analytical Beltrami field components."""
        x_1, x_2, x_3 = x
        return jnp.array([
            ((A_0 * n_mode) / (jnp.sqrt(m_mode**2 + n_mode**2))) * jnp.sin(jnp.pi * m_mode * x_1) * jnp.cos(jnp.pi * n_mode * x_2),
            ((A_0 * m_mode * -1) / (jnp.sqrt(m_mode**2 + n_mode**2))) * jnp.cos(jnp.pi * m_mode * x_1) * jnp.sin(jnp.pi * n_mode * x_2),
            A_0*jnp.sin(jnp.pi * m_mode * x_1) * jnp.sin(jnp.pi * n_mode * x_2)
        ])


    # Identity mapping
    def F(x):
        return x


    # Create DeRham sequence
    derham = DeRhamSequence(ns, ps, q, types, bcs, F, polar=False)


    
    def rhs(x):
        """Source term: μu for Beltrami equation ∇×u = μu"""
        return mu_val * u_exact(x)
    
    # Assemble matrices 
    D1 = derham.assemble_curl()     # Curl operator
    M2 = derham.assemble_M2()       # Mass matrix for 2-forms  
    P2 = derham.P2                  # Projector for 2-forms
    
    # Project RHS to 2-form space 
    rhs_projected = P2(rhs)
    
    # Solve: D1*u_hat = rhs_projected 
    # Use regularized version 
    regularization = 1e-10
    lhs_matrix = D1.T @ M2 @ D1 + regularization * derham.assemble_M1()
    rhs_vector = D1.T @ M2 @ rhs_projected
    
    u_hat = jnp.linalg.solve(lhs_matrix, rhs_vector)

    # Create discrete function for u_hat
    u_h = DiscreteFunction(u_hat, derham.Λ1, derham.E1.matrix())


    # Compute error
    def err(x):
        return u_exact(x) - u_h(x)
    error = (l2_product(err, err, derham.Q, F) / l2_product(u_exact, u_exact, derham.Q, F)) ** 0.5
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
            err[i, j] = get_err(1, 1, n, p)  # Use default modes (1,1)
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
            _ = get_err(1, 1, n, p)  
            end = time.time()
            times2[i, j] = end - start
            print(f"n={n}, p={p}, time={times2[i, j]:.2f}s")

    return err, times, times2


def main():
    """Main function to run Beltrami convergence analysis."""
    print("Beltrami Field Convergence Analysis")
    print("=" * 35)
    
    # Run convergence analysis
    ns = np.arange(4, 10,2)
    ps = np.arange(1, 4)
    err, times, times2 = run_convergence_analysis(ns, ps)
    
    print("\nAnalysis complete!")
    print(f"Final errors: {err}")


if __name__ == "__main__":
    main()
