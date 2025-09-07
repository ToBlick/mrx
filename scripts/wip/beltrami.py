"""
Beltrami Field Analysis with Homogeneous Dirichlet Boundary Conditions

This script analyzes Beltrami fields in a 3D domain with homogeneous Dirichlet boundary conditions.

The script includes:
1. Definition of Beltrami field components
2. Computation of magnetic helicity
3. Computation of energy


Still to do is adding in relaxation.
"""

import jax.numpy as jnp
import jax
import os
from typing import Tuple
from functools import partial
from mrx.Quadrature import QuadratureRule
from mrx.DeRhamSequence import DeRhamSequence

# Create output directory if it doesn't exist
os.makedirs('script_outputs', exist_ok=True)

# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)

def h(x:jnp.ndarray)-> jnp.ndarray:
    """
    Define h for eta

    Args:
        x: Position vector

    Returns:
        jnp.ndarray:h(x)
    """
    return (x**2)*(1-x)**2

def h_p(x:jnp.ndarray)-> jnp.ndarray:
    """
    Define h' 

    Args:
        x: Position vector

    Returns:
        jnp.ndarray:h'(x)
    """
    return 2*x*(1-x)**2 - 2*(x**2)*(1-x)
def mu(m: int, n: int) -> jnp.ndarray:
    """
    Compute the eigenvalue for the Beltrami field.

    Args:
        m: First mode number
        n: Second mode number

    Returns:
        jnp.ndarray: The eigenvalue mu(m,n)
    """
    return jnp.pi * jnp.sqrt(m**2 + n**2)


def u(A_0: float, x: jnp.ndarray, m: int, n: int) -> jnp.ndarray:
    """
    Compute the Beltrami field components.

    Args:
        A_0: Amplitude factor
        x: Position vector (x1, x2, x3)
        m: First mode number
        n: Second mode number

    Returns:
        jnp.ndarray: Vector field components (u_1, u_2, u_3)
    """
    x_1, x_2, x_3 = x
    return jnp.array([
        ((A_0 * n) / (jnp.sqrt(m**2 + n**2))) * jnp.sin(jnp.pi * m * x_1) * jnp.cos(jnp.pi * n * x_2),
        ((A_0 * m * -1) / (jnp.sqrt(m**2 + n**2))) * jnp.cos(jnp.pi * m * x_1) * jnp.sin(jnp.pi * n * x_2),
        A_0*jnp.sin(jnp.pi * m * x_1) * jnp.sin(jnp.pi * n * x_2)
    ])

def A_prime(A_0: float, x: jnp.ndarray,m_ori: jnp.ndarray, n_ori: jnp.ndarray, m_high: jnp.ndarray, n_high: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the value of A_0'

    Args:
        A_0: Amplitude factor
        m_high: First mode number associated with high energy
        n_high: Second mode number associated with high energy
        m_ori: First mode number
        n_ori: Second mode number
        x: Position vector

    Returns:
        jnp.ndarray: A_0'
    """
    
    return jnp.sqrt((energy_integrand(m_ori,n_ori,x,A_0)*mu(m_ori,n_ori))/(energy_integrand(m_high,n_high,x,A_0)*mu(m_high,n_high)))*A_0

def energy_integrand(m: int, n: int, x: jnp.ndarray, A_0: float) -> jnp.ndarray:

    """
    Computer other relevant integrand for the energy calculation.
    
    """
    x_1,x_2,x_3 = x
    u_e = u(A_0, x, m, n)
    # Get integrand
    return (u_e[1]*h(x_1)*h(x_2)*h_p(x_3)-u_e[2]*h(x_1)*h_p(x_2)*h(x_3))**2 + (u_e[2]*h_p(x_1)*h(x_2)*h(x_3)-u_e[0]*h(x_1)*h(x_2)*h_p(x_3))**2 + (u_e[0]*h(x_1)*h_p(x_2)*h(x_3)-u_e[1]*h_p(x_1)*h(x_2)*h(x_3))**2

def eta(x: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the weight function for the domain.

    Args:
        x: Position vector (x1, x2, x3)

    Returns:
        jnp.ndarray: Weight value at position x
    """
    x_1, x_2, x_3 = x
    return ((x_1**2) * ((1-x_1))**2) * ((x_2**2) * ((1-x_2))**2) * ((x_3**2) * ((1-x_3))**2)


def integrand_helicity(m: int, n: int, x: jnp.ndarray, A_0: float) -> jnp.ndarray:
    """
    Compute the integrand for magnetic helicity calculation.

    Args:
        m: First mode number
        n: Second mode number
        x: Position vector
        A_0: Amplitude factor

    Returns:
        jnp.ndarray: Value of the integrand
    """
    field = u(A_0, x, m, n)
    mu_val = mu(m, n)
    return eta(x)**2 * jnp.dot(field, field) * mu_val

def compute_energy(m: int, n: int, A_0: float, Q: QuadratureRule) -> jnp.ndarray:
    """
    Compute the energy for given modes.

    Args:
        m: First mode number
        n: Second mode number
        A_0: Amplitude factor
        Q: Quadrature rule for integration

    Returns:
        jnp.ndarray: Energy value
    """ 

  # Compute relevant integrand at quadrature points
    integrand_values_energy = jnp.array([energy_integrand(m, n, x, A_0) for x in Q.x])

    # Compute integral using quadrature weights
    integral = jnp.sum(integrand_values_energy * Q.w)

    return  integral+(mu(m,n)/2)*compute_helicity(m, n, A_0,Q)


def compute_helicity(m: int, n: int, A_0: float, Q: QuadratureRule) -> jnp.ndarray:
    """
    Compute the magnetic helicity for given modes.

    Args:
        m: First mode number
        n: Second mode number
        A_0: Amplitude factor
        Q: Quadrature rule for integration

    Returns:
        jnp.ndarray: Magnetic helicity value
    """
    # Compute integrand at quadrature points
    integrand_values = jnp.array([integrand_helicity(m, n, x, A_0) for x in Q.x])

    # Compute integral using quadrature weights
    integral = jnp.sum(integrand_values * Q.w)

    return integral 

def find_extreme_energy_modes(max_m: int, max_n: int, A_0: float,Q: QuadratureRule) -> jnp.ndarray:
    """
    Find the modes (m,n) that produce high and low energy.

    Args:
        max_m: Maximum m value to search
        max_n: Maximum n value to search
        A_0: Amplitude factor

    Returns:
        Lowest mode combination
        Highest mode combination
    """


    # Going to keep energies in a list
    energies = {}
    
    # Compute energies for all mode combinations
    for m in range(1, max_m + 1):
        for n in range(1, max_n + 1):

            E = jnp.float32(compute_energy(m, n, A_0,Q))
            energies[m, n] = E
    
    # Find modes with highest and lowest energy
    high_mode = max(energies.items())[0]
    low_mode = min(energies.items())[0]

    return high_mode, low_mode

@partial(jax.jit, static_argnames=['m', 'n', 'max_iter', 'Q'])
def relaxation(m: int, n: int, A_0: float, x0: jnp.ndarray, Q: QuadratureRule, max_iter: int = 1000, tol: float = 1e-6, dt: float = 0.1) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Perform magnetic relaxation to find minimum energy Beltrami field configuration.
    
    Args:
        m: First mode number
        n: Second mode number
        A_0: Initial amplitude
        x0: Initial point
        Q: Quadrature rule
       
        
    Returns:
    """
    # Initialize field
    field = u(A_0, x0, m, n)
    
    # Initialize arrays
    energies = []
    helicities = []
    
    # Define the energy and helicity functions
    def energy_fn(f):
        return compute_energy(m, n, A_0, Q)
    
    def helicity_fn(f):
        return compute_helicity(m, n, A_0, Q)
    
    
    


def main() -> None:
    """Main function to run the Beltrami field analysis."""
    # Set parameters
    n = 5
    p = 3
    ns = (n, n, n)
    ps =(p, p, p)
    types = ('clamped', 'clamped', 'constant')
    
    # Define identity mapping function
    def F(x):
        return x
    
    # Set boundary conditions 
    bcs = [['dirichlet', 'dirichlet', 'dirichlet'] for _ in range(4)]

    # Initialize DeRham sequence
    DR = DeRhamSequence(ns, ps, 15, types, bcs, F, polar=False)
    Q = DR.Q

    # Set amplitude
    A_0 = 1.0

 

if __name__ == "__main__":
    main()
