"""
Iterative solvers for fixed-point problems.

This module provides implementations of iterative solvers for fixed-point problems,
including Picard iteration and Newton's method. These solvers are implemented
using JAX for automatic differentiation and efficient computation.
"""

import jax.numpy as jnp
import jax


def picard_solver(f, z_init, tol=1e-6, max_iter=1000, norm=jnp.linalg.norm):
    """
    Solve a fixed-point problem using Picard iteration.
    
    Args:
        f: Function to find fixed point of, f: R^n -> R^n
        z_init: Initial guess, shape (n,) or scalar
        tol: Tolerance for convergence
        max_iter: Maximum number of iterations
        norm: Norm function to use for convergence check
        
    Returns:
        z_star: Fixed point solution
        
    Raises:
        RuntimeError: If solver fails to converge
    """
    z_init = jnp.asarray(z_init)
    
    def cond_fun(state):
        z_prev, z, i = state
        err = norm(z - z_prev)
        return jnp.logical_and(i < max_iter, err > tol)
    
    def body_fun(state):
        z_prev, z, i = state
        z_next = f(z)
        # Add damping to improve convergence
        z_next = z + 0.8 * (z_next - z)
        return z, z_next, i + 1
    
    # Run Picard iterations
    z_prev, z_star, iter_count = jax.lax.while_loop(cond_fun, body_fun, (z_init, f(z_init), 0))
    
    # Verify solution
    err = norm(f(z_star) - z_star)
    success = err <= tol
    
    return jax.lax.cond(
        success,
        lambda _: z_star,
        lambda _: jnp.full_like(z_star, jnp.nan),
        operand=None
    )


def newton_solver(f, z_init, tol=1e-6, max_iter=100):
    """
    Solve for the fixed point of f using Newton's method.
    
    Args:
        f: The function to find a fixed point for
        z_init: Initial guess
        tol: Tolerance for convergence
        max_iter: Maximum number of iterations
        
    Returns:
        z_final: The fixed point
        z_prev: The previous iterate
        i: Number of iterations taken
    """
    z_init = jnp.asarray(z_init)
    
    def fixed_point_form(z):
        """Convert to root finding form g(z) = f(z) - z"""
        return f(z) - z
    
    def newton_step(z):
        """Compute one Newton step with safeguards"""
        # Compute function value and Jacobian
        val, jac = jax.jvp(fixed_point_form, (z,), (jnp.ones_like(z),))
        
        # Use where to handle zero derivatives safely
        # Add small regularization to avoid division by zero
        safe_jac = jnp.where(jnp.abs(jac) < 1e-10, 1e-10, jac)
        
        # Compute step with damping (careful - this factor is critical)
        damping = 0.5  # Add damping factor for stability
        step = damping * val / safe_jac
        
        # Return new z value
        return z - step
    
    def cond_fn(state):
        """Continue while not converged and under max iterations"""
        z, z_prev, i = state
        not_converged = jnp.any(jnp.abs(z - z_prev) > tol)
        under_max_iter = i < max_iter
        return jnp.logical_and(not_converged, under_max_iter)
    
    def body_fn(state):
        """Perform one iteration"""
        z, z_prev, i = state
        z_new = newton_step(z)
        return z_new, z, i + 1
    
    # Initialize state
    state = (z_init, z_init, 0)
    
    # Run the iteration
    z_final, z_prev, i = jax.lax.while_loop(cond_fn, body_fn, state)
    
    return z_final, z_prev, i
