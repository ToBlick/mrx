from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg


def picard_solver(f, z_init, tol=1e-12, max_iter=2000, norm=jnp.linalg.norm) -> tuple[jnp.ndarray, float, int]:
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
                          jnp.clip(
                              norm(fz[0] - z[0]) / (norm(z[0] - z_prev[0]) + 1e-12), 0.0, 1.0)
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


def preconditioned_cg(A_matvec, b, x0=None, M=None, tol=1e-6, maxiter=None):
    """
    Preconditioned Conjugate Gradient with M-norm convergence check.

    Solves A x = b where A is SPD, with optional SPD preconditioner M ≈ A^{-1}.
    Convergence is measured in the preconditioner norm:
        ||r_k||_{M} = sqrt(r_k^T M r_k) < tol * ||b||_{M}

    Uses jax.lax.while_loop for JIT compatibility.

    Args:
        A_matvec: Callable, x -> A @ x (must be SPD).
        b: Right-hand side vector.
        x0: Optional initial guess.
        M: Optional preconditioner callable, x -> M @ x (approx A^{-1}, SPD).
        tol: Relative tolerance in M-norm.
        maxiter: Maximum number of iterations (default: len(b)).

    Returns:
        x: Solution vector.
        info: 0 if converged, >0 = number of iterations if not converged.
    """
    n = b.shape[0]
    if maxiter is None:
        maxiter = n
    if x0 is None:
        x0 = jnp.zeros_like(b)
    if M is None:
        def M(x): return x

    # ||b||_M for relative tolerance
    Mb = M(b)
    bnorm_M = jnp.sqrt(jnp.abs(jnp.dot(b, Mb)))
    bnorm_safe = jnp.where(bnorm_M > 0, bnorm_M, 1.0)

    # Initial residual
    r0 = b - A_matvec(x0)
    z0 = M(r0)
    rz0 = jnp.dot(r0, z0)

    # State: (x, r, z, p, rz, k, converged)
    init_state = (
        x0,
        r0,
        z0,
        z0.copy(),   # p = z0
        rz0,
        0,
        jnp.sqrt(jnp.abs(rz0)) < tol * bnorm_safe,  # check initial
    )

    def cond_fn(state):
        _, _, _, _, _, k, converged = state
        return jnp.logical_and(k < maxiter, ~converged)

    def body_fn(state):
        x, r, z, p, rz, k, _ = state

        Ap = A_matvec(p)
        alpha = rz / jnp.where(jnp.dot(p, Ap) > 0, jnp.dot(p, Ap), 1.0)

        x_new = x + alpha * p
        r_new = r - alpha * Ap
        z_new = M(r_new)
        rz_new = jnp.dot(r_new, z_new)

        beta = rz_new / jnp.where(rz > 0, rz, 1.0)
        p_new = z_new + beta * p

        # Convergence: ||r||_M = sqrt(r^T M r) = sqrt(rz)
        rnorm_M = jnp.sqrt(jnp.abs(rz_new))
        converged_new = rnorm_M < tol * bnorm_safe

        return (x_new, r_new, z_new, p_new, rz_new, k + 1, converged_new)

    final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)
    x_final = final_state[0]
    k_final = final_state[5]
    converged_final = final_state[6]

    info = jnp.where(converged_final, 0, k_final)
    return x_final, info


def solve_singular_cg(A_matvec, b, mass_matvec=None, precond_matvec=lambda x: x, x0=None, vs=[], maxiter=None, tol=1e-6):
    """
    Solve the singular SPSD system for the minimum norm solution using CG.

    Args:
        A_matvec: Callable representing bilinear form (outputs Dual vectors).
        mass_matvec: Callable representing mass matrix.
        b: The right-hand side vector (Dual vector).
        x0: Optional initial guess (Primal vector).
        vs: List of mass-normalized zero eigenvectors (Primal vectors).
        maxiter: Maximum number of CG iterations.
        tol: CG tolerance.
    """
    if mass_matvec is None:
        def mass_matvec(x): return x

    def inner_product(x, y):
        return jnp.dot(x, mass_matvec(y))

    def project_primal(x):
        for v in vs:
            x = x - inner_product(v, x) * v
        return x

    def project_dual(f):
        for v in vs:
            f = f - jnp.dot(v, f) * mass_matvec(v)
        return f

    b_proj = project_dual(b)

    def A_matvec_safe(x):
        x = project_primal(x)
        # Apply the bilinear form (output is Dual)
        Ax = A_matvec(x)
        return project_dual(Ax)

    def precond_matvec_safe(x):
        return project_primal(precond_matvec(project_dual(x)))

    if x0 is None:
        x0 = jnp.zeros_like(b_proj)
    else:
        x0 = project_primal(x0)

    x, info = preconditioned_cg(A_matvec_safe, b_proj, x0=x0,
                                M=precond_matvec_safe, tol=tol, maxiter=maxiter)
    return project_primal(x), info


def minres(A_matvec, b, x0=None, M=None, tol=1e-6, maxiter=None):
    """
    MINRES solver for symmetric (possibly indefinite) linear systems.

    Based on the SOL implementation by Choi, Paige & Saunders (2011).
    Uses jax.lax.while_loop for JIT compatibility.

    Args:
        A_matvec: Callable, x -> A @ x (must be symmetric).
        b: Right-hand side vector.
        x0: Optional initial guess.
        M: Optional preconditioner callable, x -> M^{-1} @ x.
           Must be symmetric positive definite.
        tol: Relative residual tolerance.
        maxiter: Maximum number of iterations (default: len(b)).

    Returns:
        x: Solution vector.
        info: 0 if converged, >0 = number of iterations if not converged.
    """
    n = b.shape[0]
    if maxiter is None:
        maxiter = n
    if x0 is None:
        x0 = jnp.zeros_like(b)
    if M is None:
        def M(x): return x

    # Initial residual
    r0 = b - A_matvec(x0)
    y0 = M(r0)
    beta1 = jnp.sqrt(jnp.abs(jnp.dot(r0, y0)))

    # Use preconditioned norm of b for relative tolerance:
    # ||b||_{M^{-1}} = sqrt(b^T M^{-1} b)
    Mb = M(b)
    bnorm = jnp.sqrt(jnp.abs(jnp.dot(b, Mb)))
    bnorm_safe = jnp.where(bnorm > 0, bnorm, 1.0)

    # State variables following SOL MINRES convention:
    #   x: current solution
    #   y: preconditioned residual (unnormalized; v = y/beta)
    #   r1: previous unpreconditioned Lanczos residual
    #   r2: current unpreconditioned Lanczos residual
    #   beta, oldbeta: current and previous Lanczos betas
    #   cs, sn: last Givens rotation (cs=-1 initially)
    #   dbar, epsln: QR factorization state
    #   phibar: residual norm estimate
    #   w_prev, w_pp: direction vectors (one-back and two-back)
    #   k: iteration count
    #   converged: flag
    init_state = (
        x0,                    # x
        y0,                    # y (preconditioned, divided by beta to get v)
        jnp.zeros_like(b),    # r1
        r0,                    # r2
        beta1,                 # beta
        0.0,                   # oldbeta
        -1.0,                  # cs (SOL convention: initialized to -1)
        0.0,                   # sn
        0.0,                   # dbar
        0.0,                   # epsln
        beta1,                 # phibar
        jnp.zeros_like(b),    # w_prev
        jnp.zeros_like(b),    # w_pp
        0,                     # k
        False,                 # converged
    )

    def cond_fn(state):
        *_, beta, _, _, _, _, _, phibar, _, _, k, converged = state
        return jnp.logical_and(k < maxiter, ~converged)

    def body_fn(state):
        (x, y, r1, r2, beta, oldbeta, cs, sn, dbar, epsln, phibar,
         w_prev, w_pp, k, converged) = state

        # Normalize Lanczos vector: v = y / beta
        safe_beta = jnp.where(beta > 0, beta, 1.0)
        v = y / safe_beta

        # Lanczos step
        y_new = A_matvec(v)

        # 2-term recurrence (avoids storing v_{k-1} explicitly)
        safe_oldbeta = jnp.where(oldbeta > 0, oldbeta, 1.0)
        y_new = y_new - jnp.where(k >= 1, beta / safe_oldbeta, 0.0) * r1

        alfa = jnp.dot(v, y_new)
        y_new = y_new - (alfa / safe_beta) * r2

        # Update Lanczos residual tracking
        r1_new = r2
        r2_new = y_new
        oldbeta_new = beta

        # Precondition and compute next beta
        y_prec = M(y_new)
        beta_new = jnp.sqrt(jnp.abs(jnp.dot(y_new, y_prec)))

        # Apply previous Givens rotation to get QR factorization entries
        oldeps = epsln
        delta = cs * dbar + sn * alfa
        gbar = sn * dbar - cs * alfa
        epsln_new = sn * beta_new
        dbar_new = -cs * beta_new

        # New Givens rotation to eliminate beta_new from column k
        gamma = jnp.sqrt(gbar**2 + beta_new**2)
        safe_gamma = jnp.where(gamma > 0, gamma, 1.0)
        cs_new = gbar / safe_gamma
        sn_new = beta_new / safe_gamma

        # Update residual norm estimate
        phi = cs_new * phibar
        phibar_new = sn_new * phibar

        # Update direction vector and solution
        w_new = (v - oldeps * w_pp - delta * w_prev) / safe_gamma
        x_new = x + phi * w_new

        # Check convergence
        converged_new = jnp.abs(phibar_new) < tol * bnorm_safe

        return (
            x_new,
            y_prec,         # y <- preconditioned for next iteration
            r1_new,
            r2_new,
            beta_new,
            oldbeta_new,
            cs_new,
            sn_new,
            dbar_new,
            epsln_new,
            phibar_new,
            w_new,          # w_prev <- current
            w_prev,         # w_pp <- one-back
            k + 1,
            converged_new,
        )

    final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)
    x_final = final_state[0]
    k_final = final_state[13]
    converged_final = final_state[14]

    info = jnp.where(converged_final, 0, k_final)
    return x_final, info


def solve_saddle_point_minres(
        stiffness_matvec, derivative_matvec, derivative_T_matvec,
        mass_lower_matvec, b_upper, n_upper, n_lower,
        precond_upper=None, precond_lower=None,
        mass_upper_matvec=None,
        vs_upper=None, vs_lower=None,
        x0_upper=None, x0_lower=None,
        tol=1e-6, maxiter=None):
    """
    Solve the saddle-point system using preconditioned MINRES:

        | S    D   | | u |   | f |
        | D^T  -M  | | σ | = | 0 |

    where S is the stiffness (k-form), D is the derivative (k-1 → k),
    M is the mass matrix ((k-1)-form), and σ is the auxiliary (k-1)-form.

    Args:
        stiffness_matvec: u -> S @ u (k-form to k-form dual).
        derivative_matvec: σ -> D @ σ ((k-1)-form to k-form dual).
        derivative_T_matvec: u -> D^T @ u (k-form to (k-1)-form dual).
        mass_lower_matvec: σ -> M @ σ ((k-1)-form to (k-1)-form dual).
        b_upper: RHS for the k-form block (f).
        n_upper: Number of k-form DOFs.
        n_lower: Number of (k-1)-form DOFs.
        precond_upper: Callable, approximate inverse for upper block
            (Schur complement / Hodge Laplacian). Must be linear and SPD.
        precond_lower: Callable, approximate inverse for lower block
            (mass matrix). Must be linear and SPD.
        mass_upper_matvec: u -> M_k @ u (k-form mass, for nullspace projection).
        vs_upper: List of nullspace vectors for the k-form block.
        vs_lower: List of nullspace vectors for the (k-1)-form block.
        x0_upper: Initial guess for u.
        x0_lower: Initial guess for σ.
        tol: MINRES tolerance.
        maxiter: Maximum iterations.

    Returns:
        u: Solution k-form vector.
        sigma: Solution (k-1)-form vector.
        info: 0 if converged, >0 otherwise.
    """
    if vs_upper is None:
        vs_upper = []
    if vs_lower is None:
        vs_lower = []
    if mass_upper_matvec is None:
        def mass_upper_matvec(x): return x

    n_total = n_upper + n_lower

    # --- Nullspace projection helpers ---
    def inner_upper(x, y):
        return jnp.dot(x, mass_upper_matvec(y))

    def inner_lower(x, y):
        return jnp.dot(x, mass_lower_matvec(y))

    def project_primal_upper(x):
        for v in vs_upper:
            x = x - inner_upper(v, x) * v
        return x

    def project_dual_upper(f):
        for v in vs_upper:
            f = f - jnp.dot(v, f) * mass_upper_matvec(v)
        return f

    def project_primal_lower(x):
        for v in vs_lower:
            x = x - inner_lower(v, x) * v
        return x

    def project_dual_lower(f):
        for v in vs_lower:
            f = f - jnp.dot(v, f) * mass_lower_matvec(v)
        return f

    def pack(u, s):
        return jnp.concatenate([u, s])

    def unpack(x):
        return x[:n_upper], x[n_upper:]

    def project_primal(x):
        u, s = unpack(x)
        return pack(project_primal_upper(u), project_primal_lower(s))

    def project_dual(x):
        u, s = unpack(x)
        return pack(project_dual_upper(u), project_dual_lower(s))

    # --- Saddle-point matvec ---
    def A_matvec(x):
        u, s = unpack(x)
        u = project_primal_upper(u)
        s = project_primal_lower(s)
        # Upper block: S @ u + D @ s
        r_upper = stiffness_matvec(u) + derivative_matvec(s)
        # Lower block: D^T @ u - M @ s
        r_lower = derivative_T_matvec(u) - mass_lower_matvec(s)
        result = pack(project_dual_upper(r_upper),
                      project_dual_lower(r_lower))
        return result

    # --- Block-diagonal preconditioner ---
    def precond(x):
        u, s = unpack(x)
        u = project_dual_upper(u)
        s = project_dual_lower(s)
        pu = precond_upper(u) if precond_upper is not None else u
        ps = precond_lower(s) if precond_lower is not None else s
        return pack(project_primal_upper(pu), project_primal_lower(ps))

    # --- RHS ---
    b = pack(project_dual_upper(b_upper), jnp.zeros(n_lower))

    # --- Initial guess ---
    if x0_upper is None:
        x0_upper = jnp.zeros(n_upper)
    if x0_lower is None:
        x0_lower = jnp.zeros(n_lower)
    x0 = pack(project_primal_upper(x0_upper),
              project_primal_lower(x0_lower))

    if maxiter is None:
        maxiter = n_total

    x, info = minres(A_matvec, b, x0=x0, M=precond,
                     tol=tol, maxiter=maxiter)
    u, sigma = unpack(project_primal(x))
    return u, sigma, info
