import jax
import jax.numpy as jnp
__all__ = ['picard_solver', 'newton_solver']

def picard_solver(f, z_init, tol=1e-12, max_iter=2000, norm=jnp.linalg.norm)->tuple[jnp.ndarray, float, int]:
    """
    Picard solver for fixed-point iteration.

    Parameters
    ----------
    f : callable
        Function to perform the solve on.
    z_init : jnp.ndarray
        Initial guess for the solution.
    tol : float, default=1e-12
        Tolerance for convergence.
    max_iter : int, default=1000
        Maximum number of iterations.
    norm : callable, default=jnp.linalg.norm
        Norm function definition.

    Returns
    -------
    (z_star, residual, iters) : tuple[jnp.ndarray, float, int]
        z_star = (x*, aux*) with x* the fixed point.
        residual = ||f(z_star)[0] - x*||.
        iters = picard iteration count.
    """
    def cond_fun(state):
        """
        Condition function for the while loop. Continue while either residual or change is above tolerance.

        Parameters
        ----------
        state : tuple[jnp.ndarray, jnp.ndarray, int]
            State of the solver.

        Returns
        -------
        bool : Whether to continue the loop.
        """
        # z = (x, (aux1, aux2, ...))
        z_prev, z, i = state
        # Use the residual ||f(z) - z|| as stopping criterion. 
        residual = norm(f(z)[0] - z[0])
        # Continue while either residual or change is above tolerance.
        return jnp.logical_and(i < max_iter, jnp.logical_or(residual > tol, jnp.isnan(residual)))

    def body_fun(state):
        """
        Body function for the while loop.

        Parameters
        ----------
        state : tuple[jnp.ndarray, jnp.ndarray, int]
            State of the solver.

        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray, int] : Next state.
        """
        z_prev, z, i = state
        fz = f(z)
        alpha = jnp.where(i == 0,
                            1,
                            jnp.clip(norm(fz[0] - z[0]) / (norm(z[0] - z_prev[0]) + 1e-12), 0.0, 1.0)
                            )
        z_next = (alpha * fz[0] + (1 - alpha) * z[0], fz[1])
        return (z, z_next, i + 1)

    state = (z_init, f(z_init), 0)
    _, z_star, iters = jax.lax.while_loop(cond_fun, body_fun, state)

    # If the solver didn't iterate at all (iters == 0), we still want to
    # evaluate f once so the caller receives the updated auxiliary outputs
    # that f computes.
    z_star = jax.lax.cond(iters == 0, lambda z: f(z), lambda z: z, z_star)
    return z_star, norm(f(z_star)[0] - z_star[0]), iters

def newton_solver(f, z_init, tol=1e-12, max_iter=2000, norm=jnp.linalg.norm):
    """
    Newton fixed-point solver compatible with picard_solver's (x, aux) state.

    Parameters
    ----------
    f : callable
        Map that takes a state z = (x, aux) and returns (x_new, aux_new).
        The fixed-point equation is x = f((x, aux))[0].
    z_init : jnp.ndarray or tuple
        Initial state (x0, aux0) tuple.
    tol : float, default=1e-12
        Tolerance for convergence.
    max_iter : int, default=1000
        Maximum number of iterations.
    norm : callable, default=jnp.linalg.norm
        Norm function definition.

    Returns
    -------
    (z_star, residual, iters)
        z_star = (x*, aux*) with x* the Newton fixed point.
        residual = ||f(z_star)[0] - x*||.
        iters = picard iteration count applied to the Newton map.
    """
    def g(z):
        """One Newton update on x, threaded aux; returns (x_next, aux_next)."""
        x, aux = z

        # F(x) = f((x, aux))[0] - x  (fixed-point residual on the primary var)
        def F(x_):
            return f((x_, aux))[0] - x_

        Fx = F(x)                             # shape like x
        J = jax.jacrev(F)(x)                  # Jacobian dF/dx at current x
        dx = jnp.linalg.solve(J, Fx)          # Newton step
        x_next = x - dx
        # Update aux consistently
        _, aux_next = f((x_next, aux))
        return (x_next, aux_next)

    # Hand off to picard_solver to iterate the Newton map g
    return picard_solver(g, z_init, tol, max_iter, norm)

