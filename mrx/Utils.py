"""
Utility functions for differential geometry and numerical analysis.

This module provides a collection of mathematical functions commonly used in
differential geometry, finite element analysis, and numerical computations.
It includes functions for computing Jacobians, matrix operations, and
differential operators using JAX for automatic differentiation.
"""

from typing import Any, Callable

import jax
import jax.numpy as jnp

__all__ = ['jacobian', 'inv33', 'div', 'curl', 'grad', 'l2_product']

def jacobian_determinant(f: Callable[[jnp.ndarray], jnp.ndarray]) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Compute the determinant of the Jacobian matrix for a given function.

    Args:
        f: Function mapping from R^n to R^n for which to compute the Jacobian determinant

    Returns:
        Function that computes the Jacobian determinant at a given point
    """
    return lambda x: jnp.linalg.det(jax.jacfwd(f)(x))


def inv33(mat: jnp.ndarray) -> jnp.ndarray:
    """Compute the inverse of a 3x3 matrix using explicit formula.

    This function computes the inverse using the adjugate matrix formula,
    which is more efficient than general matrix inversion for 3x3 matrices.

    Args:
        mat: 3x3 matrix to invert

    Returns:
        The inverse of the input matrix
    """
    m1, m2, m3 = mat[0]
    m4, m5, m6 = mat[1]
    m7, m8, m9 = mat[2]
    det = m1 * (m5 * m9 - m6 * m8) + m4 * \
        (m8 * m3 - m2 * m9) + m7 * (m2 * m6 - m3 * m5)
    # Return zero matrix if determinant is zero
    return jnp.where(
        jnp.abs(det) < 1e-10,  # Careful with this tolerance
        jnp.zeros((3, 3)),
        jnp.array([
            [m5 * m9 - m6 * m8, m3 * m8 - m2 * m9, m2 * m6 - m3 * m5],
            [m6 * m7 - m4 * m9, m1 * m9 - m3 * m7, m3 * m4 - m1 * m6],
            [m4 * m8 - m5 * m7, m2 * m7 - m1 * m8, m1 * m5 - m2 * m4],
        ]) / det
    )


def div(F: Callable[[jnp.ndarray], jnp.ndarray]) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Compute the divergence of a vector field.

    Args:
        F: Vector field function for which to compute the divergence

    Returns:
        Function that computes the divergence at a given point
    """
    def div_F(x: jnp.ndarray) -> jnp.ndarray:
        DF = jax.jacfwd(F)(x)
        return jnp.trace(DF) * jnp.ones(1)
    return div_F


def curl(F: Callable[[jnp.ndarray], jnp.ndarray]) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Compute the curl of a vector field in 3D.

    Args:
        F: Vector field function for which to compute the curl

    Returns:
        Function that computes the curl at a given point
    """
    def curl_F(x: jnp.ndarray) -> jnp.ndarray:
        DF = jax.jacfwd(F)(x)
        return jnp.array([DF[2, 1] - DF[1, 2],
                         DF[0, 2] - DF[2, 0],
                         DF[1, 0] - DF[0, 1]])
    return curl_F


def grad(F: Callable[[jnp.ndarray], jnp.ndarray]) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Compute the gradient of a scalar field.

    Args:
        F: Scalar field function for which to compute the gradient

    Returns:
        Function that computes the gradient at a given point
    """
    def grad_F(x: jnp.ndarray) -> jnp.ndarray:
        DF = jax.jacfwd(F)(x)
        return jnp.ravel(DF)
    return grad_F


def l2_product(f: Callable[[jnp.ndarray], jnp.ndarray],
               g: Callable[[jnp.ndarray], jnp.ndarray],
               Q: Any,
               F: Callable[[jnp.ndarray], jnp.ndarray] = lambda x: x) -> jnp.ndarray:
    """Compute the L2 inner product of two functions over a domain.

    Computes the integral of f·g over the domain defined by the quadrature rule Q,
    with optional coordinate transformation F.

    Args:
        f: First function in the inner product
        g: Second function in the inner product
        Q: Quadrature rule object with attributes x (points) and w (weights)
        F: Optional coordinate transformation function (default is identity)

    Returns:
        The L2 inner product value
    """
    Jj = jax.vmap(jacobian_determinant(F))(Q.x)
    return jnp.einsum("ij,ij,i,i->", jax.vmap(f)(Q.x), jax.vmap(g)(Q.x), Jj, Q.w)
