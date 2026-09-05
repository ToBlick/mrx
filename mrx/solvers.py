"""Matrix-free Krylov solvers: :func:`preconditioned_cg`, :func:`solve_singular_cg`
and :func:`solve_saddle_point_minres`, and :func:`refine`, the iterative
refinement that runs them in mixed precision.

Every solver takes a relative tolerance ``tol``; ``tol=None`` is
:data:`mrx.precision.SOLVE_TOL`. Given the outer operator (``A_res`` /
``saddle_res``: the float64 view of the sequence, or the operator itself
in a plain configuration) and the norm of its residual's space, the
singular CG and the saddle MINRES run under :func:`refine`, the one
stopping criterion of the package: the true residual of the outer
equation in the mass-atom norm, the correction solved by the Krylov
iteration in the working precision, the solution accumulated in the
residual precision and returned in it. Until 2026-09-04 the default was sqrt(eps) of the working
dtype with a docstring claiming it was the tightest tolerance that did not
send the loop to ``maxiter``; measured on li383 (16,32,32) p=3 in float32,
every production solve converges to 1e-7 (``docs/research/velocity_leray_ab_2026-09-04.md``).
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from mrx.precision import DTYPE, MAX_PASSES, RESIDUAL_DTYPE, solve_tol
from mrx.precision import inner_tol as default_inner_tol


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
        tol: Relative tolerance in M-norm; ``None`` is ``mrx.precision.SOLVE_TOL``.
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
        tol = solve_tol()
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


def deflation_projectors(vs, mass_matvec):
    """``(project_primal, project_dual)`` against the ``(m, n)`` rows of ``vs``,
    ``M``-orthonormal kernel vectors: ``x - (vs M x) vs`` on a primal vector,
    ``f - (vs f) (M vs)`` on a dual one, with ``M vs`` formed once. With no
    rows both are the identity."""
    vs = jnp.asarray(vs)
    if vs.ndim != 2:
        raise ValueError(f"vs must be an (m, n) array, got shape {vs.shape}")
    if vs.shape[0] == 0:
        return (lambda x: x), (lambda f: f)
    mass_vs = jax.vmap(mass_matvec)(vs)

    def project_primal(x):
        return x - (vs @ mass_matvec(x)) @ vs

    def project_dual(f):
        return f - (vs @ f) @ mass_vs

    return project_primal, project_dual


def refine(apply_res, solve, b, x0=None, tol=None, project_dual=None, norm=None,
           max_passes=MAX_PASSES, inner_dtype=DTYPE, residual=None):
    """The outer loop of every solve: the true residual of ``A x = b`` in
    the norm of its space, corrected until it is below ``tol``.

    ``apply_res(x)`` applies ``A`` in the residual precision
    (:data:`mrx.precision.RESIDUAL_DTYPE`; the operator itself when the
    working precision is the residual one), ``solve(r)`` solves ``A d = r``
    in the working precision from zero to its own inner tolerance and
    returns ``(d, info)``. The residual ``b - A x`` is evaluated in the
    residual precision and measured by ``norm`` -- the mass-atom norm of
    the residual's space, the h-independent norm of a dual vector (the
    plain 2-norm of the coefficients when no norm is given) -- and the
    correction added, until ``norm(b - A x) <= tol norm(b)`` or
    ``max_passes`` corrections were taken. In mixed precision each pass
    takes the residual down by about the inner tolerance, so a warm start
    with a 1% defect meets 1e-8 in two; in a plain configuration the inner
    solve runs at ``tol`` and the loop is the check that its own criterion,
    a preconditioned norm of some inner operator, did not stop short of the
    true one (measured 2026-09-05: the k=1 Hodge split converged on its hat
    operator at 1e-8 with a true residual of 2e-6). ``project_dual`` removes
    the nullspace component of a singular system's residual. Returns ``(x,
    info)`` with ``x`` in the residual precision and ``info`` the inner
    iterations of all passes, negative when the residual test was met.
    ``inner_dtype`` is the dtype the inner solve runs in: the working dtype,
    or the residual dtype for a solve on the float64 view. A composite
    solve whose residual is not ``b - A x`` of its own unknown -- the
    Hodge Laplacian, measured as the saddle residual of the pair ``(x,
    w)`` -- gives ``residual(x)`` instead of ``apply_res`` (and ``x0``, the
    unknown's shape); ``b`` is then the right-hand side in the residual's
    space, for the norm only, and ``solve`` receives the residual whole.
    """
    tol = solve_tol() if tol is None else tol
    if project_dual is None:
        def project_dual(r): return r
    if norm is None:
        norm = jnp.linalg.norm
    b = b.astype(RESIDUAL_DTYPE)
    if residual is None:
        x = jnp.zeros_like(b) if x0 is None else x0.astype(RESIDUAL_DTYPE)

        def residual(x):
            return project_dual(b - apply_res(x))
    else:
        x = x0.astype(RESIDUAL_DTYPE)
    bnorm = norm(project_dual(b))
    bnorm_safe = jnp.where(bnorm > 0, bnorm, 1.0)

    def cond(carry):
        _, r, k, _ = carry
        return jnp.logical_and(norm(r) > tol * bnorm_safe, k < max_passes)

    def body(carry):
        x, r, k, its = carry
        d, info = solve(r.astype(inner_dtype))
        x = x + d.astype(RESIDUAL_DTYPE)
        return x, residual(x), k + 1, its + jnp.abs(info)

    x, r, _, its = jax.lax.while_loop(cond, body, (x, residual(x), 0, 0))
    converged = norm(r) <= tol * bnorm_safe
    return x, jnp.where(converged, -its, its)


def solve_singular_cg(A_matvec, b, vs, mass_matvec=None, precond_matvec=lambda x: x, x0=None,
                      maxiter=None, tol=None, A_res=None, norm=None, inner_tol=None,
                      inner_dtype=DTYPE):
    """
    Solve the singular SPSD system for the minimum norm solution using CG.

    Args:
        A_matvec: Callable representing bilinear form (outputs Dual vectors).
        mass_matvec: Callable representing mass matrix.
        b: The right-hand side vector (Dual vector).
        x0: Optional initial guess (Primal vector).
        vs: ``(m, n)`` array of M-orthonormal kernel vectors (primal), ``m``
            possibly 0 -- the shape the sequence stores its harmonic forms in
            (:func:`mrx.nullspace.get_nullspace`).
        maxiter: Maximum number of CG iterations.
        tol: the outer tolerance; ``None`` is ``mrx.precision.SOLVE_TOL``.
        A_res: ``A`` in the residual precision. When given the solve runs
            under :func:`refine` -- the true residual in ``norm`` decides --
            and ``x`` comes back in that precision; without it the CG's own
            preconditioned criterion decides.
        norm: the norm of the residual for :func:`refine`.
        inner_tol: the CG's tolerance per pass under :func:`refine`;
            ``None`` is the square root of ``tol`` (:func:`mrx.precision.inner_tol`).
        inner_dtype: the dtype the CG runs in under :func:`refine`.
    """
    if mass_matvec is None:
        def mass_matvec(x): return x

    if A_res is None:
        project_primal, project_dual = deflation_projectors(jnp.asarray(vs, dtype=b.dtype), mass_matvec)

        def A_matvec_safe(x):
            return project_dual(A_matvec(project_primal(x)))

        def precond_matvec_safe(x):
            return project_primal(precond_matvec(project_dual(x)))

        x0 = jnp.zeros_like(b) if x0 is None else project_primal(x0)
        x, info = preconditioned_cg(A_matvec_safe, project_dual(b), x0=x0,
                                    M=precond_matvec_safe, tol=tol, maxiter=maxiter)
        return project_primal(x), info

    # The inner iteration's projectors in its own dtype, the outer loop's in
    # the residual precision.
    project_primal_in, project_dual_in = deflation_projectors(
        jnp.asarray(vs, dtype=inner_dtype), mass_matvec)
    project_primal, project_dual = deflation_projectors(
        jnp.asarray(vs, dtype=RESIDUAL_DTYPE), mass_matvec)

    def A_matvec_safe(x):
        return project_dual_in(A_matvec(project_primal_in(x)))

    def precond_matvec_safe(x):
        return project_primal_in(precond_matvec(project_dual_in(x)))

    tol = solve_tol() if tol is None else tol
    inner = default_inner_tol(tol) if inner_tol is None else inner_tol

    def solve(r):
        return preconditioned_cg(A_matvec_safe, r, M=precond_matvec_safe,
                                 tol=inner, maxiter=maxiter)

    x, info = refine(lambda x: A_res(project_primal(x)), solve, b, x0=x0, tol=tol,
                     project_dual=project_dual, norm=norm, inner_dtype=inner_dtype)
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
        tol: Relative residual tolerance; ``None`` is ``mrx.precision.SOLVE_TOL``.
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
        tol = solve_tol()
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
        precond_upper=None, precond_lower=None,
        mass_upper_matvec=None,
        vs_upper=None,
        x0_upper=None, x0_lower=None,
        tol=None, maxiter=None, saddle_res=None, norm_upper=None, norm_lower=None,
        inner_tol=None, inner_dtype=DTYPE):
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
        mass_upper_matvec: u -> M_k @ u (k-form mass, for nullspace projection).
        vs_upper: ``(m, n_upper)`` M-orthonormal nullspace vectors of the
            k-form block, deflated from ``u``. The lower block needs none:
            a harmonic ``v`` has ``D^T v = 0`` (both halves of ``L_k`` are
            semidefinite), so the saddle matrix's nullspace is ``(v, 0)``.
        x0_upper: Initial guess for u.
        x0_lower: Initial guess for sigma. A guess on ``u`` alone leaves
            ``D^T x0_upper`` in the lower block of the initial residual, so
            the two go together.
        tol: MINRES tolerance; ``None`` is ``mrx.precision.SOLVE_TOL``.
        maxiter: Maximum iterations.
        saddle_res: ``(u, sigma) -> (S u + D sigma, D^T u - M sigma)`` in the
            residual precision. When given the solve runs under
            :func:`refine` -- the true residual decides, in the block norm
            ``sqrt(norm_upper(r_u)^2 + norm_lower(r_l)^2)`` of the two
            residual spaces -- and ``u``, ``sigma`` come back in that
            precision. ``inner_tol`` is MINRES's tolerance per pass;
            ``None`` is the square root of ``tol`` (:func:`mrx.precision.inner_tol`).

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
    if mass_upper_matvec is None:
        def mass_upper_matvec(x): return x

    n_total = n_upper + n_lower
    dtype = b_upper.dtype

    def _rows(vs, n):
        return jnp.zeros((0, n), dtype=dtype) if vs is None or len(vs) == 0 else jnp.asarray(vs)

    project_primal_upper, project_dual_upper = deflation_projectors(_rows(vs_upper, n_upper), mass_upper_matvec)

    def pack(u, s):
        return jnp.concatenate([u, s])

    def unpack(x):
        return x[:n_upper], x[n_upper:]

    def project_primal(x):
        u, s = unpack(x)
        return pack(project_primal_upper(u), s)

    # --- Saddle-point matvec ---
    def A_matvec(x):
        u, s = unpack(x)
        u = project_primal_upper(u)
        # Upper block: S @ u + D @ s
        r_upper = stiffness_matvec(u) + derivative_matvec(s)
        # Lower block: D^T @ u - M @ s
        r_lower = derivative_T_matvec(u) - mass_lower_matvec(s)
        return pack(project_dual_upper(r_upper), r_lower)

    # --- Block-diagonal preconditioner ---
    def precond(x):
        u, s = unpack(x)
        u = project_dual_upper(u)
        pu = precond_upper(u) if precond_upper is not None else u
        ps = precond_lower(s) if precond_lower is not None else s
        return pack(project_primal_upper(pu), ps)

    # --- RHS and initial guess ---
    b = pack(project_dual_upper(b_upper), jnp.zeros(n_lower, dtype=dtype))
    if x0_upper is None:
        x0_upper = jnp.zeros(n_upper, dtype=dtype)
    if x0_lower is None:
        x0_lower = jnp.zeros(n_lower, dtype=dtype)
    x0 = pack(project_primal_upper(x0_upper), x0_lower)

    if maxiter is None:
        maxiter = n_total

    if saddle_res is None:
        x, info = minres(A_matvec, b, x0=x0, M=precond, tol=tol, maxiter=maxiter)
        u, sigma = unpack(project_primal(x))
        return u, sigma, info

    def apply_res(x):
        u, s = unpack(x)
        r_upper, r_lower = saddle_res(project_primal_upper(u), s)
        return pack(project_dual_upper(r_upper), r_lower)

    tol = solve_tol() if tol is None else tol
    inner = default_inner_tol(tol) if inner_tol is None else inner_tol

    def solve(r):
        return minres(A_matvec, r, M=precond, tol=inner, maxiter=maxiter)

    def norm(r):
        r_u, r_l = unpack(r)
        nu = jnp.linalg.norm(r_u) if norm_upper is None else norm_upper(r_u)
        nl = jnp.linalg.norm(r_l) if norm_lower is None else norm_lower(r_l)
        return jnp.sqrt(nu ** 2 + nl ** 2)

    x, info = refine(apply_res, solve, b, x0=x0, tol=tol, norm=norm, inner_dtype=inner_dtype)
    u, sigma = unpack(project_primal(x))
    return u, sigma, info
