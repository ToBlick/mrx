import jax
import jax.numpy as jnp
__all__ = ['picard_solver', 'newton_solver']


def aitken_step(z_prev, z_curr, fz, eps=1e-12, inprod=jnp.vdot):
    d1 = z_curr - z_prev
    d2 = fz - z_curr
    num = inprod(d1, d2)
    den = inprod(d2, d2) + eps
    omega = jnp.clip(num / den, 0.0, 1.0)
    z_next = (1.0 - omega) * z_curr + omega * fz
    return z_next, omega
# %%
def picard_solver(f, z_init, tol=1e-12, max_iter=1000, norm=jnp.linalg.norm, inprod=jnp.vdot, debug=False):
    """
    Picard iteration with Aitken acceleration.
    """
    def cond_fun(state):
        # z = (x, (aux1, aux2, ...))
        z_prev, z, i = state
        # Use the residual ||f(z) - z|| as stopping criterion. 
        residual = norm(f(z)[0] - z[0])
        # Continue while either residual or change is above tolerance.
        return jnp.logical_and(i < max_iter, jnp.logical_or(residual > tol, jnp.isnan(residual)))

    def body_fun(state):

        z_prev, z, i = state
        fz = f(z)
        
        alpha = jnp.where(i == 0,
                          1,
                          jnp.clip(norm(fz[0] - z[0]) / (norm(z[0] - z_prev[0]) + 1e-12), 0.0, 1.0)
                          )
        # alpha = jnp.clip(norm(fz[0] - z[0]) / (norm(z[0] - z_prev[0]) + 1e-12), 0.0, 1.0)

        z_next = (alpha * fz[0] + (1 - alpha) * z[0], fz[1])
        return (z, z_next, i + 1)

    state = (z_init, f(z_init), 0)
    _, z_star, iters = jax.lax.while_loop(cond_fun, body_fun, state)

    # If the solver didn't iterate at all (iters == 0), we still want to
    # evaluate f once so the caller receives the updated auxiliary outputs
    # that f computes.
    z_star = jax.lax.cond(iters == 0, lambda z: f(z), lambda z: z, z_star)

    return z_star, norm(f(z_star)[0] - z_star[0]), iters
# %%

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
