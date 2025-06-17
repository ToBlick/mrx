"""
We solve the Helmholtz equation (∇^2 + λ^2)B = 0. As in Kaptanolglu HW2 question, we assume an infinitely long cylindrical geometry with radius a and assume a form B(r,ϕ,z) = B(r)e^{imϕ+ikz}. Take the boundary condition Br(r = a,ϕ,z) = 0. 

The script includes:
1. Definition of analytical B-field
2. Definition of source term
3. Definition of finite element spaces
4. Definition of operators
5. Solve system
6. Error computation
7. Convergence analysis

This is a work in progress, and is based on cylinder_vector_poisson.py. Has outdated flat operator.
"""

import os
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from mrx.DifferentialForms import DifferentialForm, Flat
from mrx.LazyMatrices import (
    LazyDerivativeMatrix,
    LazyDoubleDivergenceMatrix,
    LazyMassMatrix,
)
from mrx.PolarMapping import LazyExtractionOperator, get_xi
from mrx.Projectors import Projector
from mrx.Quadrature import QuadratureRule

# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)

# Create output directory for figures
os.makedirs("script_outputs", exist_ok=True)

def B(x, m: int, k: float, λ: float, B0: float = 1.0):
    """
    Compute the B-field components in cylindrical coordinates.
    
    Args:
        x: (r, φ, z) coordinates
        m: Azimuthal mode number
        k: Axial wavenumber
        λ: Eigenvalue
        B0: Amplitude constant
        
    Returns:
        Array of [Br, Bφ, Bz] components
    """
    r, φ, z = x
    
    # Compute common terms
    sqrt_term = jnp.sqrt(λ**2 - k**2)
    r_sqrt = r * sqrt_term
    
    # Bessel functions using recurrence relations
    def bessel_jn(n, x):
        """Compute Bessel function of first kind of order n using recurrence relations and series expansion."""
        x_real = jnp.real(x)
        if n == 0:
            return 1 - (x_real**2/4) + (x_real**4/64) - (x_real**6/2304) + (x_real**8/147456)-(x_real**10/14745600)
        elif n == 1:
            return (x_real/2) - (x_real**3/16) + (x_real**5/384) - (x_real**7/18432) + (x_real**9/1474560)-(x_real**11/176947200)
        else:
            J0 = bessel_jn(0, x)
            J1 = bessel_jn(1, x)
            for i in range(1, n):
                J2 = (2*i/x) * J1 - J0
                J0 = J1
                J1 = J2
            return J1
    # Derivative of Bessel function
    def bessel_jn_prime(n, x):
        """Compute derivative of Bessel function using recurrence relation."""
        if n == 0:
            return -bessel_jn(1, x)  # J0' = -J1
        else:
            Jn_minus_1 = bessel_jn(n-1, x)
            Jn_plus_1 = bessel_jn(n+1, x)
            return (Jn_minus_1 - Jn_plus_1)/2
    
    # Get Bessel functions and derivatives
    Jm = bessel_jn(m, r_sqrt)
    Jm_prime = bessel_jn_prime(m, r_sqrt)
    
    # Complex exponential term
    exp_term = jnp.exp(1j * (m * φ + k * z))
    
    # Compute components
    Br = (1j/(λ**2 - k**2)) * (
        (λ * m/r) * Jm + 
        k * sqrt_term * Jm_prime
    ) * B0 * exp_term
    
    Bφ = (-1/(λ**2 - k**2)) * (
        (k * m/r) * Jm + 
        λ * sqrt_term * Jm_prime
    ) * B0 * exp_term
    
    Bz = Jm * B0 * exp_term
    
    return jnp.array([Br, Bφ, Bz])

@partial(jax.jit, static_argnames=['n', 'p'])
def get_err(n, p, m: int = 1, k: float = 2.0, λ: float = 3.0):
    """
    Compute the error in the solution of the vector Helmholtz equation.

    Args:
        n: Number of elements in each direction
        p: Polynomial degree
        m: Mode number
        k: Wavenumber
        λ: Eigenvalue

    Returns:
        float: Relative L2 error of the solution
    """
    # Set up finite element spaces
    q = 2*p
    ns = (n, n, n)
    ps = (p, p, p)
    types = ("clamped", "periodic", "periodic")
    
  

    def _X(r, φ):
        return jnp.ones(1) * r * jnp.cos(φ)

    def _Y(r, φ):
        return jnp.ones(1) * r * jnp.sin(φ)

    def _Z(r, φ):
        return jnp.ones(1)

    def F(x):
        """Cylindricla to cartesian mapping function."""
        r, φ, z = x
        return jnp.ravel(jnp.array([_X(r, φ),
                                    _Y(r, φ),
                                    _Z(r, φ) * z]))

    # Define exact solution and source term
    def B_exact(x):
        """Exact solution of the Helmholtz equation."""
        return B(x, m, k, λ)

    def f(x):
        """Source term of the Helmholtz equation (zero for homogeneous equation)."""
        return jnp.zeros(3)  # Zero source term for (∇² + λ²)B = 0

    # Set up differential forms
    Λ0, Λ1, Λ2, Λ3 = [DifferentialForm(k, ns, ps, types) for k in range(4)]

    # Get polar mapping and set up operators
    Q = QuadratureRule(Λ0, q)
    ξ = get_xi(_X, _Z, Λ0, Q)[0]
    E1 = LazyExtractionOperator(Λ1, ξ, zero_bc=False).M
    E2 = LazyExtractionOperator(Λ2, ξ, zero_bc=True).M

    # Set up operators for mixed formulation
    # C represents the curl operator
    C = LazyDerivativeMatrix(Λ1, Λ2, Q, F, E1, E2).M
    # K represents the divergence operator
    K = LazyDoubleDivergenceMatrix(Λ2, Q, F, E2).M
    # Mass matrices for 1-forms and 2-forms
    M1 = LazyMassMatrix(Λ1, Q, F, E1).M
    M2 = LazyMassMatrix(Λ2, Q, F, E2).M

    # Construct the vector Helmholtz operator
    # For vector fields in cylindrical coordinates:
    # ∇²B = ∇(∇·B) - ∇×(∇×B)
    # The operator C @ jnp.linalg.solve(M1, C.T) represents -∇×(∇×B)
    # The operator K represents ∇(∇·B)
    # The operator M2 represents the mass term
    L = C @ jnp.linalg.solve(M1, C.T) + K + λ**2 * M2

    # Solve the system
    P2 = Projector(Λ2, Q, F, E2)
    
    # Get the exact solution projected onto our space
    B_hat_exact = jnp.linalg.solve(M2, P2(Flat(B_exact, F)))
    
    # For the homogeneous equation, we need to enforce boundary conditions
    # Br = 0 at r = a (outer boundary)
    # All components periodic in φ and z
    # The boundary conditions are already enforced by the extraction operator E2
    # in the construction of L, so we don't need to apply it again
    B_hat = jnp.linalg.solve(L, P2(Flat(f, F)))

    # Compute error using the same form as cylinder_vector_poisson.py
    # Take absolute value to ensure we get a real number
    error = jnp.abs(((B_hat - B_hat_exact) @ M2 @ (B_hat - B_hat_exact) /
             (B_hat_exact @ M2 @ B_hat_exact))**0.5)
    return error

def run_convergence_analysis():
    """Run convergence analysis for different parameters."""
    # Parameter ranges
    ns = np.arange(4, 5)  # number of elements
    ps = np.arange(1, 3)     # polynomial degrees (must be < n)

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
            print(f"n={n}, p={p}, err={err[i, j]:.2e}, time={times[i, j]:.2f}s")

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



def main():
    """Main function to run the analysis."""
    # Run convergence analysis
    err, times, times2 = run_convergence_analysis()

if __name__ == "__main__":
    main()

