import jax.numpy as jnp
import jax


def picard_solver(f, z_init, tol=1e-12, norm=jnp.linalg.norm):
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

    def cond_fun(carry):
        z_prev, z = carry
        err = norm(z_prev - z)
        # jax.debug.print("err: {err}", err=err)
        return err > tol

    def body_fun(carry):
        _, z = carry
        # jax.debug.print("z: {z}", z=z)
        return z, f(z)

    init_carry = (z_init, f(z_init))
    _, z_star = jax.lax.while_loop(cond_fun, body_fun, init_carry)
    return z_star


def newton_solver(f, z_init, tol=1e-12, norm=jnp.linalg.norm):
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

    return picard_solver(g, z_init, tol, norm)
