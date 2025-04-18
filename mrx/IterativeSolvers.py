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
        Any: The fixed-point solution `z_star` such that z_star ≈ f(z_star).
    """
    def cond_fun(carry):
        z_prev, z = carry
        return norm(z_prev - z) > tol

    def body_fun(carry):
        _, z = carry
        return z, f(z)

    init_carry = (z_init, f(z_init))
    _, z_star = jax.lax.while_loop(cond_fun, body_fun, init_carry)
    return z_star

def newton_solver(f, z_init, tol=1e-12, norm=jnp.linalg.norm):
    f_root = lambda z: f(z) - z
    g = lambda z: z - jnp.linalg.solve(jax.jacobian(f_root)(z), f_root(z))
    return picard_solver(g, z_init, tol, norm)