import jax
import jax.numpy as jnp


def picard_solver(f, z_init, tol=1e-12, max_iter=1000, norm=jnp.linalg.norm):
    """
    Solves a fixed-point problem using the Picard iteration method.

    Parameters:
        f (Callable): A function representing the fixed-point iteration,
                    where the solution satisfies z = f(z).
        z_init (Any): The initial guess for the solution.
        tol (float, optional): The convergence tolerance. Iteration stops
                            when the norm of the difference between
                            successive iterates is less than this value.
                            Default is 1e-12.
        norm (Callable, optional): A function to compute the norm of the
                                    difference between successive iterates.
                                    Default is `jnp.linalg.norm`.

    Returns:
        Any: The fixed-point solution `z_star` such that |z_star - f(z_star)| < tol.
    """

    def cond_fun(state):
        z_prev, z, i = state
        err = norm(z_prev - z)
        # jax.debug.print("err: {err}", err=err)
        return jnp.logical_and(i < max_iter, err > tol)

    def body_fun(state):
        _, z, i = state
        # jax.debug.print("z: {z}", z=z)
        return z, f(z), i + 1

    state = (z_init, f(z_init), 0)
    _, z_star, iters = jax.lax.while_loop(cond_fun, body_fun, state)

    # Verify solution
    err = norm(f(z_star) - z_star)
    # success = err <= tol

    return z_star, err, iters


def newton_solver(f, z_init, tol=1e-12, max_iter=1000, norm=jnp.linalg.norm):
    """
    Solve a fixed-point problem using Newton's method.
    Parameters:
        f (callable): The function for which the fixed point is to be found.
                      It should take a single argument and return a value of the same shape.
        z_init (array-like): The initial guess for the fixed point.
        tol (float, optional): The tolerance for convergence. The iteration stops when the norm
                               of the difference between successive approximations is less than `tol`.
                               Default is 1e-12.
        norm (callable, optional): A function to compute the norm of a vector.
                                   Default is `jnp.linalg.norm`.
    Returns:
        array-like: The computed fixed point of the function `f`.
    Notes:
        - The function `picard_solver` is used internally to perform the iterative process.
    """

    def f_root(z):
        return f(z) - z

    def g(z):
        return z - jnp.linalg.solve(jax.jacrev(f_root)(z), f_root(z))

    return picard_solver(g, jnp.atleast_1d(z_init), tol, max_iter, norm)
