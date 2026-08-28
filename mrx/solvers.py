"""Matrix-free Krylov solvers: :func:`preconditioned_cg`, :func:`solve_singular_cg`
and :func:`solve_saddle_point_minres`.

Every solver takes a relative tolerance ``tol``.  ``tol=None`` -- the default
everywhere -- resolves to :func:`mrx.precision.sqrt_eps`, the square root of
the working dtype's machine epsilon (1.5e-8 at float64, 3.5e-4 at float32):
the residual of a matrix-free solve is limited by roundoff in the matvec, and
that is the tightest tolerance which does not send the loop to ``maxiter``.
Anything tighter or looser is an ALGORITHMIC choice and is passed explicitly.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from mrx.precision import sqrt_eps


def preconditioned_cg(A_matvec, b, x0=None, M=None, tol=None, maxiter=None):
    """
    Preconditioned Conjugate Gradient with M-norm convergence check.

    Solves A x = b where A is SPD, with optional SPD preconditioner M ≈ A^{-1}.
    Convergence is measured in the preconditioner norm::

        ||r_k||_{M} = sqrt(r_k^T M r_k) < tol * ||b||_{M}

    Uses jax.lax.while_loop for JIT compatibility.

    Args:
        A_matvec: Callable, x -> A @ x (must be SPD).
        b: Right-hand side vector.
        x0: Optional initial guess.
        M: Optional preconditioner callable, x -> M @ x (approx A^{-1}, SPD).
        tol: Relative tolerance in M-norm; ``None`` is ``mrx.sqrt_eps()``.
        maxiter: Maximum number of iterations (default: len(b)).

    Returns:
        x: Solution vector.
        info: ``-k`` if converged after ``k`` iterations, ``+k`` if not.
            (This docstring said "0 if converged" until 2026-08-24; the code
            has always returned the signed iteration count -- see the
            ``jnp.where(converged_final, -k_final, k_final)`` below. The stale
            version caused converged solves to be read as failures.)

    ``M`` must be SPD. That is not checked, and it is not defended against
    either: a non-SPD ``M`` makes ``beta`` the square root of a negative
    number and the whole run turns to NaN, which is the intended outcome.
    """
    n = b.shape[0]
    if tol is None:
        tol = sqrt_eps()
    if maxiter is None:
        maxiter = n
    if x0 is None:
        x0 = jnp.zeros_like(b)
    if M is None:
        def M(x): return x

    # ||b||_M for relative tolerance.
    #
    # No abs(): b^T M^-1 b is an inner product and CG requires M SPD. If it
    # comes out negative the preconditioner is broken and every subsequent
    # number is noise -- NaN is the honest report.
    Mb = M(b)
    bnorm_M = jnp.sqrt(jnp.dot(b, Mb))
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
        z0,          # p = z0
        rz0,
        0,
        jnp.sqrt(rz0) < tol * bnorm_safe,  # check initial
    )

    def cond_fn(state):
        _, _, _, _, _, k, converged = state
        return jnp.logical_and(k < maxiter, ~converged)

    def body_fn(state):
        x, r, z, p, rz, k, _ = state

        Ap = A_matvec(p)
        pAp = jnp.dot(p, Ap)
        # p^T A p <= 0 means A is NOT positive definite, and CG is simply not
        # applicable -- the iteration has no minimisation to perform and the
        # answer is meaningless. Substituting 1.0 turned that into a silently
        # wrong solve that still reports iteration counts. Divide by it.
        alpha = rz / pAp

        x_new = x + alpha * p
        r_new = r - alpha * Ap
        z_new = M(r_new)
        rz_new = jnp.dot(r_new, z_new)

        # rz > 0 for any SPD M and nonzero r; rz == 0 is exact convergence,
        # which cond_fn has already caught, so there is nothing to guard.
        beta = rz_new / rz
        p_new = z_new + beta * p

        # Convergence: ||r||_M = sqrt(r^T M r) = sqrt(rz)
        rnorm_M = jnp.sqrt(rz_new)
        converged_new = rnorm_M < tol * bnorm_safe

        return (x_new, r_new, z_new, p_new, rz_new, k + 1, converged_new)

    final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)
    x_final = final_state[0]
    k_final = final_state[5]
    converged_final = final_state[6]

    # info < 0: converged (|info| = iteration count); info > 0: NOT converged
    info = jnp.where(converged_final, -k_final, k_final)
    return x_final, info


def solve_singular_cg(A_matvec, b, mass_matvec=None, precond_matvec=lambda x: x, x0=None, vs=(), maxiter=None, tol=None):
    """
    Solve the singular SPSD system for the minimum norm solution using CG.

    Args:
        A_matvec: Callable representing bilinear form (outputs Dual vectors).
        mass_matvec: Callable representing mass matrix.
        b: The right-hand side vector (Dual vector).
        x0: Optional initial guess (Primal vector).
        vs: List of mass-normalized zero eigenvectors (Primal vectors).
        maxiter: Maximum number of CG iterations.
        tol: CG tolerance; ``None`` is ``mrx.sqrt_eps()``.
    """
    if mass_matvec is None:
        def mass_matvec(x): return x

    if isinstance(vs, (list, tuple)) and len(vs) == 0:
        vs_stacked = jnp.zeros((0, b.shape[0]), dtype=b.dtype)
    else:
        vs_stacked = jnp.asarray(vs, dtype=b.dtype)
        if vs_stacked.ndim == 1:
            vs_stacked = vs_stacked[None, :]

    def inner_product(x, y):
        return jnp.dot(x, mass_matvec(y))

    if vs_stacked.shape[0] == 0:
        def project_primal(x):
            return x

        def project_dual(f):
            return f
    else:
        mass_vs = jax.vmap(mass_matvec)(vs_stacked)

        def project_primal(x):
            coeffs = vs_stacked @ mass_matvec(x)
            return x - coeffs @ vs_stacked

        def project_dual(f):
            coeffs = vs_stacked @ f
            return f - coeffs @ mass_vs

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


class _MinresState(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    r1: jnp.ndarray
    r2: jnp.ndarray
    beta: float
    oldbeta: float
    cs: float
    sn: float
    dbar: float
    epsln: float
    phibar: float
    w_prev: jnp.ndarray
    w_pp: jnp.ndarray
    k: int
    converged: bool


def minres(A_matvec, b, x0=None, M=None, tol=None, maxiter=None):
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
        tol: Relative residual tolerance; ``None`` is ``mrx.sqrt_eps()``.
        maxiter: Maximum number of iterations (default: len(b)).

    Returns:
        x: Solution vector.
        info: ``-k`` if converged after ``k`` iterations, ``+k`` if not.
            (This docstring said "0 if converged" until 2026-08-25; the code
            has always returned the signed iteration count -- see the
            ``jnp.where(converged_final, -k_final, k_final)`` below. The stale
            version caused a converged solve to be read as a failure.)
    """
    n = b.shape[0]
    if tol is None:
        tol = sqrt_eps()
    if maxiter is None:
        maxiter = n
    if x0 is None:
        x0 = jnp.zeros_like(b)
    if M is None:
        def M(x): return x

    # Initial residual.
    #
    # NO abs() HERE, DELIBERATELY. r0^T M^-1 r0 is an inner product in the
    # M^-1 metric and is positive for any M that MINRES is entitled to be
    # given -- the method REQUIRES an SPD preconditioner. If it is negative,
    # M is not SPD, the Lanczos basis that follows is meaningless, and every
    # number this function returns is noise. An abs() here converts that from
    # a NaN you cannot miss into "it just needs more iterations", which is
    # exactly how a non-SPD preconditioner hides. Let it produce NaN.
    r0 = b - A_matvec(x0)
    y0 = M(r0)
    beta1 = jnp.sqrt(jnp.dot(r0, y0))

    # Use preconditioned norm of b for relative tolerance:
    # ||b||_{M^{-1}} = sqrt(b^T M^{-1} b)
    Mb = M(b)
    bnorm = jnp.sqrt(jnp.dot(b, Mb))
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
    init_state = _MinresState(
        x=x0,
        y=y0,
        r1=jnp.zeros_like(b),
        r2=r0,
        beta=beta1,
        oldbeta=0.0,
        cs=-1.0,
        sn=0.0,
        dbar=0.0,
        epsln=0.0,
        phibar=beta1,
        w_prev=jnp.zeros_like(b),
        w_pp=jnp.zeros_like(b),
        k=0,
        converged=False,
    )

    def cond_fn(state):
        return jnp.logical_and(state.k < maxiter, ~state.converged)

    def body_fn(state):
        (x, y, r1, r2, beta, oldbeta, cs, sn, dbar, epsln, phibar,
         w_prev, w_pp, k, converged) = state

        # Normalize Lanczos vector: v = y / beta
        safe_beta = jnp.where(beta > 0, beta, 1.0)
        v = y / safe_beta

        # Lanczos step
        y_new = A_matvec(v)

        # Compute alpha before any subtraction (SOL convention: alpha = v^T A v,
        # which equals v^T y_new in exact arithmetic since v^T r1 = 0 by
        # M-orthogonality, but computing it first avoids numerical drift at
        # high iteration counts when orthogonality is only approximate).
        alpha = jnp.dot(v, y_new)

        # 2-term recurrence (avoids storing v_{k-1} explicitly)
        old_beta = jnp.where(oldbeta > 0, oldbeta, 1.0)
        y_new = y_new - jnp.where(k >= 1, beta / old_beta, 0.0) * r1
        y_new = y_new - (alpha / safe_beta) * r2

        # Update Lanczos residual tracking
        r1_new = r2
        r2_new = y_new
        oldbeta_new = beta

        # Precondition and compute next beta. No abs(), same reason as beta1:
        # a negative y^T M^-1 y means M is not SPD and the run is void.
        y_prec = M(y_new)
        beta_new = jnp.sqrt(jnp.dot(y_new, y_prec))

        # Apply previous Givens rotation to get QR factorization entries
        oldeps = epsln
        delta = cs * dbar + sn * alpha
        gbar = sn * dbar - cs * alpha
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

        # Check convergence. phibar is a residual-norm estimate built as
        # sn*phibar from phibar_0 = beta1 >= 0 and sn = beta/gamma >= 0, so it
        # is non-negative by construction; abs() here only hid a beta that had
        # gone imaginary.
        converged_new = phibar_new < tol * bnorm_safe

        return _MinresState(
            x=x_new,
            y=y_prec,
            r1=r1_new,
            r2=r2_new,
            beta=beta_new,
            oldbeta=oldbeta_new,
            cs=cs_new,
            sn=sn_new,
            dbar=dbar_new,
            epsln=epsln_new,
            phibar=phibar_new,
            w_prev=w_new,
            w_pp=w_prev,
            k=k + 1,
            converged=converged_new,
        )

    final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)
    x_final = final_state.x
    k_final = final_state.k
    converged_final = final_state.converged

    # info < 0: converged (|info| = iteration count); info > 0: NOT converged
    info = jnp.where(converged_final, -k_final, k_final)
    return x_final, info


def solve_saddle_point_minres(
        stiffness_matvec, derivative_matvec, derivative_T_matvec,
        mass_lower_matvec, b_upper, n_upper, n_lower,
    precond_upper=None, precond_lower=None, precond_matvec=None,
        mass_upper_matvec=None,
        vs_upper=None, vs_lower=None,
        x0_upper=None, x0_lower=None,
        tol=None, maxiter=None):
    """
    Solve the saddle-point system using preconditioned MINRES::

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
        precond_matvec: Callable, approximate inverse for the full saddle
            block. Must be linear and SPD. When supplied, it takes precedence
            over precond_upper / precond_lower.
        mass_upper_matvec: u -> M_k @ u (k-form mass, for nullspace projection).
        vs_upper: List of nullspace vectors for the k-form block.
        vs_lower: List of nullspace vectors for the (k-1)-form block.
        x0_upper: Initial guess for u.
        x0_lower: Initial guess for σ.
        tol: MINRES tolerance; ``None`` is ``mrx.sqrt_eps()``.
        maxiter: Maximum iterations.

    Returns:
        u: Solution k-form vector.
        sigma: Solution (k-1)-form vector.
        info: ``-k`` if converged after ``k`` iterations, ``+k`` if not.
            Forwarded verbatim from :func:`minres`, which has always returned
            the SIGNED iteration count. This docstring said "0 if converged"
            until 2026-08-25 -- the THIRD instance of that claim in this file,
            after the two corrected on 2026-08-24 and 2026-08-25. Reading it
            as written turns a converged solve into a failure.
    """
    if vs_upper is None:
        vs_upper = []
    if vs_lower is None:
        vs_lower = []
    if mass_upper_matvec is None:
        def mass_upper_matvec(x): return x

    n_total = n_upper + n_lower

    # --- Nullspace projection helpers ---
    # Pre-compute mass-applied nullspace vectors with vmap (O(n_null) matvecs),
    # then use matrix ops for projection — matches solve_singular_cg convention.
    if len(vs_upper) == 0:
        def project_primal_upper(x): return x
        def project_dual_upper(f): return f
    else:
        _vs_u = jnp.asarray(vs_upper)
        _mass_vs_u = jax.vmap(mass_upper_matvec)(_vs_u)
        def project_primal_upper(x):
            return x - (_vs_u @ mass_upper_matvec(x)) @ _vs_u
        def project_dual_upper(f):
            return f - (_vs_u @ f) @ _mass_vs_u

    if len(vs_lower) == 0:
        def project_primal_lower(x): return x
        def project_dual_lower(f): return f
    else:
        _vs_l = jnp.asarray(vs_lower)
        _mass_vs_l = jax.vmap(mass_lower_matvec)(_vs_l)
        def project_primal_lower(x):
            return x - (_vs_l @ mass_lower_matvec(x)) @ _vs_l
        def project_dual_lower(f):
            return f - (_vs_l @ f) @ _mass_vs_l

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
        if precond_matvec is not None:
            px = precond_matvec(pack(u, s))
            return project_primal(px)
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


