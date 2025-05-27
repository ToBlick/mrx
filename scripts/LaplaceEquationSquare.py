import numpy as np
import jax
import jax.numpy as jnp



"""
This is a file intended to solve Laplace's equation on a unit square, with homogeneous dirichlet boundary conditions of u=1. Using separation of variables,
it can be detemrined that a solution can be written as:
u(x, y) = sum_{n=1}^{\infty} a_n sinh(nπx) sin(nπy)
where the coefficients a_n are determined by the boundary conditions.
"""


def laplace_coefficients(N:float):
    """
    Computes the coefficients for the series expansion of the solution to Laplace's equation. We have:
    u(x,1) = 0 = sum_{n=1}^{\infty} a_n sinh(nπx)sin(nπ)
    u(x,0) = 0 = sum_{n=1}^{\infty} a_n sinh(nπx) sin(0)
    u(0,y) = 0 = sum_{n=1}^{\infty} a_n sinh(0) sin(nπy)
    u(1,y) = 0 = sum_{n=1}^{\infty} a_n sinh(nπ) sin(nπy)
    The first three equations are trivial since sin(nπ) = sinh(0)= 0. Thus, using orthogonality, the fourth equation 
    implies that a_n = C/sinh(nπ). I choose C=2.


    Parameters:
    N : float 
        The number of coefficients in the expansion to compute.
        
    Returns:
    a : jnp.ndarray
        The computed coefficients for the series expansion.
    """
    n = jax.numpy.arange(1, N + 1)
    a = 2 * jax.numpy.ones(N) / (jnp.sinh(n * jnp.pi))
    
    return a



def laplace_solution(x:jnp.ndarray,y:jnp.ndarray, a: jnp.ndarray,N:float) -> jnp.ndarray:
    """
    Computes the solution to Laplace's equation on a unit square with homogeneous dirichlet boundary conditions.
    
    Parameters:
    x : x-coordinates
    y :y-coordinates
    a : coefficients of expansion
    N: number of terms in expansion
        
    Returns:
    u : jnp.ndarray
        The solution
    """

    n = jax.numpy.arange(1, N + 1)
    x = x.reshape(-1, 1)  # column vector
    y = y.reshape(-1, 1)  # column vector
    a = a.reshape(1, -1)  # row vector
    n = n.reshape(1, -1)  # row vector
    u = jnp.sum(a * jnp.sinh(n * jnp.pi * x) * jnp.sin(n * jnp.pi * y), axis=1)
    
    return u

print(laplace_solution(jnp.array([0.5, 0.2]), jnp.array([0.5, 0.5]), laplace_coefficients(10),10))