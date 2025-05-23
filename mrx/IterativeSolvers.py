"""
Iterative solvers for fixed-point problems.

This module provides implementations of iterative solvers for fixed-point problems,
including Picard iteration and Newton's method. These solvers are implemented
using JAX for automatic differentiation and efficient computation.
"""

import warnings

import jax
import jax.numpy as jnp
__all__ = ['picard_solver', 'newton_solver']


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

    # JIT-compile the condition function
    # @jax.jit
    def cond_fun(state):
        z_prev, z, i = state
        err = norm(z - z_prev)
        return jnp.logical_and(i < max_iter, err > tol)

    # JIT-compile the body function
    # @jax.jit
    def body_fun(state):
        z_prev, z, i = state
        z_next = f(z)
        return z, z_next, i + 1

    # Run Picard iterations
    z_prev, z_star, iter_count = jax.lax.while_loop(
        cond_fun, body_fun, (z_init, f(z_init), 0))

    # Verify solution
    err = norm(f(z_star) - z_star)
    success = err <= tol

    def warn_if_failed(success):
        jax.lax.cond(
            jnp.any(~success),
            lambda _: warnings.warn(
                f"Picard solver failed to converge in {max_iter} iterations"),
            lambda _: None,
            operand=None
        )

    warn_if_failed(success)
    return z_star


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

    # JIT-compile the fixed point form function
    @jax.jit
    def fixed_point_form(z):
        """Convert to root finding form g(z) = f(z) - z"""
        return f(z) - z

    # JIT-compile the Newton step function
    @jax.jit
    def newton_step(z):
        """Compute one Newton step with safeguards"""
        # Compute function value and Jacobian
        val, jac = jax.jvp(fixed_point_form, (z,), (jnp.ones_like(z),))

        # Use where to handle zero derivatives safely
        # Add small regularization to avoid division by zero
        safe_jac = jnp.where(jnp.abs(jac) < 1e-10, 1e-10, jac)

        # Return new z value
        return z - val / safe_jac

    # JIT-compile the condition function
    @jax.jit
    def cond_fn(state):
        """Continue while not converged and under max iterations"""
        z, z_prev, i = state
        not_converged = jnp.any(jnp.abs(z - z_prev) > tol)
        under_max_iter = i < max_iter
        return jnp.logical_and(not_converged, under_max_iter)

    # JIT-compile the body function
    @jax.jit
    def body_fn(state):
        """Perform one iteration"""
        z, z_prev, i = state
        z_new = newton_step(z)
        return z_new, z, i + 1

    # Initialize state
    # 0.1 is a dummy scaling to avoid giving z = z_prev which ends the while loop
    state = (z_init, z_init * 0.1, 0)

    # Run the iteration
    z_final, z_prev, i = jax.lax.while_loop(cond_fn, body_fn, state)

    return z_final, z_prev, i
