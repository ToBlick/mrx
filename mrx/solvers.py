"""Matrix-free Krylov solvers used by all linear solves in MRX.

The solvers take the operator and the preconditioner as functions ``x -> A x``, so they work
inside ``jit`` and never form a matrix. :func:`preconditioned_cg` solves SPD systems, and
:func:`minres` solves symmetric indefinite ones. :func:`minres` can also stop at a direction of
nonpositive curvature, which the Newton-MR relaxation uses. :func:`solve_singular_cg` solves a
semidefinite system with a known kernel, and :func:`solve_saddle_point_minres` solves the
saddle-point form of a Hodge Laplacian. :func:`chebyshev` is a fixed polynomial approximate inverse
with no loop, for use where an inverse sits inside another iteration, and :func:`lanczos_bounds`
estimates the spectral interval it needs. :func:`refine` wraps a solve in iterative refinement: the
residual is computed in the residual precision (float64 in the mixed configuration) and each
correction is solved in the working precision (see :mod:`mrx.precision`).

Every Krylov solver returns a signed iteration count ``info``. A value ``+k`` means converged after ``k``
iterations, and ``-k`` means not converged after ``k`` iterations.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from mrx.precision import DTYPE, MAX_PASSES, RESIDUAL_DTYPE


def preconditioned_cg(A_matvec, b, M, tol, maxiter, x0=None):
    """Solve the SPD system ``A x = b`` by preconditioned CG and return ``(x, info)``.

    ``M`` is an SPD approximation of ``A^{-1}``. The solve stops when ``||r||_M < tol ||b||_M``
    or after ``maxiter`` iterations. If ``A`` or ``M`` is not SPD the result becomes NaN.
    """
    if x0 is None:
        x0 = jnp.zeros_like(b)

    bnorm_M = jnp.sqrt(jnp.dot(b, M(b)))
    bnorm_safe = jnp.where(bnorm_M > 0, bnorm_M, 1.0)

    r0 = b - A_matvec(x0)
    z0 = M(r0)
    rz0 = jnp.dot(r0, z0)

    # state: (x, r, z, p, rz, k, converged)
    init_state = (x0, r0, z0, z0, rz0, 0, jnp.sqrt(rz0) < tol * bnorm_safe)

    def cond_fn(state):
        _, _, _, _, _, k, converged = state
        return jnp.logical_and(k < maxiter, ~converged)

    def body_fn(state):
        x, r, z, p, rz, k, _ = state
        Ap = A_matvec(p)
        alpha = rz / jnp.dot(p, Ap)
        x_new = x + alpha * p
        r_new = r - alpha * Ap
        z_new = M(r_new)
        rz_new = jnp.dot(r_new, z_new)
        p_new = z_new + (rz_new / rz) * p
        # ||r||_M = sqrt(r^T M r) = sqrt(rz)
        return (x_new, r_new, z_new, p_new, rz_new, k + 1, jnp.sqrt(rz_new) < tol * bnorm_safe)

    x_final, _, _, _, _, k_final, converged_final = jax.lax.while_loop(cond_fn, body_fn, init_state)
    return x_final, jnp.where(converged_final, k_final, -k_final)


def chebyshev(A_matvec, b, M, bounds, steps):
    """Return ``x = q(M A) M b``, a fixed polynomial approximation of ``A^{-1} b``.

    ``q`` is the Chebyshev polynomial of degree ``steps`` for the interval ``bounds = (lmin, lmax)``, which
    must contain the spectrum of ``M A`` (``A`` SPD, ``M`` an SPD approximation of ``A^{-1}``). The cost is
    ``steps`` applies of ``A`` and ``steps + 1`` of ``M``, with no loop and no stopping test. Unlike a
    Krylov solve the result is linear in ``b``, and for symmetric ``A`` and ``M`` the map ``b -> x`` is
    symmetric, so it can stand for ``A^{-1}`` inside another symmetric operator. The relative error in the
    ``A``-norm falls like ``2 ((sqrt(kappa) - 1) / (sqrt(kappa) + 1))^steps`` with ``kappa = lmax / lmin``
    (Saad 2003, Algorithm 12.1).
    """
    lmin, lmax = bounds[0].astype(b.dtype), bounds[1].astype(b.dtype)
    theta, delta = (lmax + lmin) / 2, (lmax - lmin) / 2
    sigma = theta / delta
    rho = 1 / sigma
    r = b
    d = M(r) / theta
    x = d
    for _ in range(steps):
        r = r - A_matvec(d)
        rho_new = 1 / (2 * sigma - rho)
        d = rho_new * rho * d + (2 * rho_new / delta) * M(r)
        rho = rho_new
        x = x + d
    return x


def lanczos_bounds(A_matvec, b, M, steps):
    """The extreme Ritz values ``(lmin, lmax)`` of ``M A`` after ``steps`` iterations of PCG on ``A x = b``.

    The Lanczos matrix is read off the PCG coefficients. Both values lie inside the spectrum of ``M A``,
    ``lmax`` close to the top already after a few iterations and ``lmin`` approaching the bottom more
    slowly. The loop has a fixed length, so ``steps`` must stay below the iteration count at which PCG
    converges to round-off.
    """
    z = M(b)
    rz = jnp.dot(b, z)

    def body(carry, _):
        r, z, p, rz = carry
        Ap = A_matvec(p)
        alpha = rz / jnp.dot(p, Ap)
        r = r - alpha * Ap
        z_new = M(r)
        rz_new = jnp.dot(r, z_new)
        beta = rz_new / rz
        return (r, z_new, z_new + beta * p, rz_new), (alpha, beta)

    _, (alpha, beta) = jax.lax.scan(body, (b, z, z, rz), None, length=steps)
    beta_prev = jnp.concatenate([jnp.zeros(1, alpha.dtype), beta[:-1]])
    alpha_prev = jnp.concatenate([jnp.ones(1, alpha.dtype), alpha[:-1]])
    diag = 1 / alpha + beta_prev / alpha_prev
    off = jnp.sqrt(beta[:-1]) / alpha[:-1]
    ritz = jnp.linalg.eigvalsh(jnp.diag(diag) + jnp.diag(off, 1) + jnp.diag(off, -1))
    return ritz[0], ritz[-1]


def deflation_projectors(vs, mass_matvec):
    """Return the projections ``(project_primal, project_dual)`` that remove a kernel.

    The rows of ``vs`` (shape ``(m, n)``) are the kernel vectors, orthonormal in the mass
    inner product ``M``. ``project_primal(x) = x - (vs M x) vs`` acts on coefficient vectors and
    ``project_dual(f) = f - (vs f) (M vs)`` on right-hand sides and residuals. With ``m = 0``
    both are the identity.
    """
    vs = jnp.asarray(vs)
    if vs.shape[0] == 0:
        return (lambda x: x), (lambda f: f)
    mass_vs = jax.vmap(mass_matvec)(vs)

    def project_primal(x):
        return x - (vs @ mass_matvec(x)) @ vs

    def project_dual(f):
        return f - (vs @ f) @ mass_vs

    return project_primal, project_dual


def refine(apply_res, solve, b, *, tol, norm, inner_dtype, x0=None, project_dual=None,
           residual=None):
    """Solve ``A x = b`` by iterative refinement and return ``(x, info)``, ``x`` in the residual precision.

    Each pass computes the residual ``r = b - A x`` with ``apply_res`` (``A`` in the residual
    precision), solves ``A d = r`` approximately with ``solve(r) -> (d, info)`` in ``inner_dtype``
    and adds ``d`` to ``x``. It stops when ``norm(r) <= tol norm(b)`` or after
    :data:`~mrx.precision.MAX_PASSES` passes. ``info`` counts the inner iterations of all passes.

    ``norm`` measures residuals. The callers pass the norm given by the mass preconditioner of the
    residual's space. ``project_dual`` removes the kernel of a singular system from the residual.
    A solve whose residual is not of the form ``b - A x`` passes ``residual(x)`` and ``x0``
    instead of ``apply_res``, and ``b`` is then used only for its norm.
    """
    if project_dual is None:
        def project_dual(r): return r
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
        return jnp.logical_and(norm(r) > tol * bnorm_safe, k < MAX_PASSES)

    def body(carry):
        x, r, k, its = carry
        # The inner solve gets the residual scaled to unit norm, so the range of the working
        # precision never limits the tolerance. s > 0 inside the loop.
        s = norm(r)
        d, info = solve((r / s).astype(inner_dtype))
        x = x + s * d.astype(RESIDUAL_DTYPE)
        return x, residual(x), k + 1, its + jnp.abs(info)

    x, r, _, its = jax.lax.while_loop(cond, body, (x, residual(x), 0, 0))
    converged = norm(r) <= tol * bnorm_safe
    return x, jnp.where(converged, its, -its)


def _compose_parity(projectors, parity):
    """Compose the kernel projections with the parity projections of a half-period sequence (if any)."""
    if parity is None:
        return projectors
    pp, pd = projectors
    par_p, par_d = parity
    return (lambda x: par_p(pp(x))), (lambda f: par_d(pd(f)))


def solve_singular_cg(A_matvec, b, vs, *, precond_matvec, tol, maxiter, mass_matvec=None,
                      x0=None, A_res=None, norm=None, inner_tol=None, inner_dtype=DTYPE,
                      parity=None):
    """Solve the symmetric semidefinite system ``A x = b`` by PCG and return ``(x, info)``.

    The rows of ``vs`` (shape ``(m, n)``, ``m`` may be 0) span the kernel of ``A`` and are
    orthonormal in the mass inner product ``mass_matvec``. The kernel is removed from the
    iterates and the residuals, so ``x`` is the solution orthogonal to it.

    Without ``A_res`` this is a plain PCG that stops at ``tol``. With ``A_res`` (``A`` in the
    residual precision) the PCG runs inside :func:`refine`: each pass solves to ``inner_tol`` in
    ``inner_dtype``, the true residual measured in ``norm`` decides convergence at ``tol``, and
    ``x`` is returned in the residual precision. ``parity``, the parity projections of a
    half-period sequence, keeps the iterates at the parity of ``b``.
    """
    dtype = b.dtype if A_res is None else inner_dtype
    project_primal_in, project_dual_in = _compose_parity(
        deflation_projectors(jnp.asarray(vs, dtype=dtype), mass_matvec), parity)

    def A(x):
        return project_dual_in(A_matvec(project_primal_in(x)))

    def P(x):
        return project_primal_in(precond_matvec(project_dual_in(x)))

    if A_res is None:
        x0 = jnp.zeros_like(b) if x0 is None else project_primal_in(x0)
        x, info = preconditioned_cg(A, project_dual_in(b), P, tol, maxiter, x0=x0)
        return project_primal_in(x), info

    project_primal, project_dual = _compose_parity(deflation_projectors(
        jnp.asarray(vs, dtype=RESIDUAL_DTYPE), mass_matvec), parity)
    x, info = refine(lambda x: A_res(project_primal(x)),
                     lambda r: preconditioned_cg(A, r, P, inner_tol, maxiter),
                     b, x0=x0, tol=tol, project_dual=project_dual, norm=norm,
                     inner_dtype=inner_dtype)
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
    npc: bool


def minres(A_matvec, b, *, M, tol, maxiter, x0=None, npc_exit=False):
    """Solve the symmetric (possibly indefinite) system ``A x = b`` by preconditioned MINRES.

    Returns ``(x, info)``. ``M`` is an SPD approximation of ``A^{-1}``, and the solve stops when
    the residual estimate falls below ``tol ||b||_M`` (Choi, Paige & Saunders 2011).

    With ``npc_exit=True`` the solve also stops at the first residual ``r`` with
    ``<r, A r> <= 0`` and returns ``M r`` in place of the iterate. This is the
    nonpositive-curvature exit of Newton-MR (Liu & Roosta 2022, Algorithm 1), and ``M r`` is a
    descent direction of nonpositive curvature. The return is then ``(x, info, npc)`` with
    ``npc`` true if the exit was taken.
    """
    if x0 is None:
        x0 = jnp.zeros_like(b)

    # A negative r^T M r means M is not SPD. The result then becomes NaN.
    r0 = b - A_matvec(x0)
    y0 = M(r0)
    beta1 = jnp.sqrt(jnp.dot(r0, y0))
    bnorm = jnp.sqrt(jnp.dot(b, M(b)))
    bnorm_safe = jnp.where(bnorm > 0, bnorm, 1.0)

    # The state follows SOL MINRES. y is the preconditioned residual (v = y / beta), r1 and r2
    # are the previous and current Lanczos residuals, (cs, sn) is the last Givens rotation,
    # dbar and epsln are the QR state, phibar is the residual estimate, and w_prev and w_pp
    # are the search directions one and two steps back.
    init_state = _MinresState(
        x=x0, y=y0, r1=jnp.zeros_like(b), r2=r0, beta=beta1, oldbeta=0.0,
        cs=-1.0, sn=0.0, dbar=0.0, epsln=0.0, phibar=beta1,
        w_prev=jnp.zeros_like(b), w_pp=jnp.zeros_like(b), k=0, converged=False, npc=False,
    )

    def cond_fn(state):
        return jnp.logical_and(state.k < maxiter, ~(state.converged | state.npc))

    def body_fn(state):
        (x, y, r1, r2, beta, oldbeta, cs, sn, dbar, epsln, phibar,
         w_prev, w_pp, k, converged, _) = state

        safe_beta = jnp.where(beta > 0, beta, 1.0)
        v = y / safe_beta
        y_new = A_matvec(v)
        # alpha = v^T A v before the subtractions (drift-free at high iteration counts)
        alpha = jnp.dot(v, y_new)
        old_beta = jnp.where(oldbeta > 0, oldbeta, 1.0)
        y_new = y_new - jnp.where(k >= 1, beta / old_beta, 0.0) * r1
        y_new = y_new - (alpha / safe_beta) * r2

        y_prec = M(y_new)
        beta_new = jnp.sqrt(jnp.dot(y_new, y_prec))

        # previous Givens rotation
        oldeps = epsln
        delta = cs * dbar + sn * alpha
        gbar = sn * dbar - cs * alpha
        epsln_new = sn * beta_new
        dbar_new = -cs * beta_new
        # The NPC test on the previous residual, <r, A r> <= 0 read off as c_{t-1} gamma_t >= 0.
        # gamma_t is gbar and c_{t-1} is the rotation before this one (cs = -1 at t = 1, so
        # the test is alpha_1 <= 0).
        npc_new = npc_exit & (cs * gbar >= 0.0)

        # new Givens rotation
        gamma = jnp.sqrt(gbar**2 + beta_new**2)
        safe_gamma = jnp.where(gamma > 0, gamma, 1.0)
        cs_new = gbar / safe_gamma
        sn_new = beta_new / safe_gamma
        phi = cs_new * phibar
        phibar_new = sn_new * phibar

        # The iterate is frozen at the NPC exit, because the tested residual belongs to the previous iterate.
        w_new = (v - oldeps * w_pp - delta * w_prev) / safe_gamma
        x_new = jnp.where(npc_new, x, x + phi * w_new)
        converged_new = phibar_new < tol * bnorm_safe

        return _MinresState(
            x=x_new, y=y_prec, r1=r2, r2=y_new, beta=beta_new, oldbeta=beta,
            cs=cs_new, sn=sn_new, dbar=dbar_new, epsln=epsln_new, phibar=phibar_new,
            w_prev=w_new, w_pp=w_prev, k=k + 1,
            converged=converged_new & ~npc_new, npc=npc_new,
        )

    final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)
    x_final = final_state.x
    info = jnp.where(final_state.converged, final_state.k, -final_state.k)
    if npc_exit:
        # The NPC direction is the preconditioned residual of the frozen iterate.
        d_npc = M(b - A_matvec(x_final))
        return jnp.where(final_state.npc, d_npc, x_final), info, final_state.npc
    return x_final, info


def solve_saddle_point_minres(
        stiffness_matvec, derivative_matvec, derivative_T_matvec,
        mass_lower_matvec, b_upper, n_upper, n_lower, *,
        precond_upper, precond_lower, mass_upper_matvec, vs_upper,
        tol, maxiter, saddle_res, norm_upper, norm_lower, inner_tol, inner_dtype,
        x0_upper=None, x0_lower=None, parity_upper=None, parity_lower=None):
    """Solve the saddle-point system of a Hodge Laplacian by refined MINRES and return ``(u, sigma, info)``::

        | S    D   | | u     |   | f |
        | D^T  -M  | | sigma | = | 0 |

    Here ``S`` is the k-form stiffness matrix, ``D`` the weak derivative of the (k-1)-forms and
    ``M`` the (k-1)-form mass matrix. ``precond_upper`` and ``precond_lower`` are SPD
    preconditioners of the two diagonal blocks. The rows of ``vs_upper`` span the harmonic
    k-forms, orthonormal in ``M_k``, and are removed from ``u``. The lower block has no kernel.

    MINRES runs inside :func:`refine`. ``saddle_res(u, sigma)`` returns the two residual blocks in
    the residual precision, each pass solves to ``inner_tol``, and convergence at ``tol`` is
    judged in the norm ``sqrt(norm_upper^2 + norm_lower^2)``. ``u`` and ``sigma`` are returned in
    the residual precision. An initial guess ``x0_upper`` should come with a matching
    ``x0_lower``, since ``u`` alone leaves ``D^T x0_upper`` in the lower residual.
    """
    dtype = b_upper.dtype
    project_primal_upper, project_dual_upper = _compose_parity(
        deflation_projectors(jnp.asarray(vs_upper), mass_upper_matvec), parity_upper)
    # The lower block has no kernel. parity_lower has the sign of the upper block, since sigma follows u.
    project_primal_lower, project_dual_lower = _compose_parity(((lambda x: x), (lambda f: f)), parity_lower)

    def pack(u, s):
        return jnp.concatenate([u, s])

    def unpack(x):
        return x[:n_upper], x[n_upper:]

    def project_primal(x):
        u, s = unpack(x)
        return pack(project_primal_upper(u), project_primal_lower(s))

    def A_matvec(x):
        u, s = unpack(x)
        u = project_primal_upper(u)
        s = project_primal_lower(s)
        r_upper = stiffness_matvec(u) + derivative_matvec(s)
        r_lower = derivative_T_matvec(u) - mass_lower_matvec(s)
        return pack(project_dual_upper(r_upper), project_dual_lower(r_lower))

    def precond(x):
        u, s = unpack(x)
        pu = precond_upper(project_dual_upper(u))
        ps = precond_lower(project_dual_lower(s))
        return pack(project_primal_upper(pu), project_primal_lower(ps))

    b = pack(project_dual_upper(b_upper), jnp.zeros(n_lower, dtype=dtype))
    if x0_upper is None:
        x0_upper = jnp.zeros(n_upper, dtype=dtype)
    if x0_lower is None:
        x0_lower = jnp.zeros(n_lower, dtype=dtype)
    x0 = pack(project_primal_upper(x0_upper), x0_lower)

    def apply_res(x):
        u, s = unpack(x)
        r_upper, r_lower = saddle_res(project_primal_upper(u), project_primal_lower(s))
        return pack(project_dual_upper(r_upper), project_dual_lower(r_lower))

    def solve(r):
        return minres(A_matvec, r, M=precond, tol=inner_tol, maxiter=maxiter)

    def norm(r):
        r_u, r_l = unpack(r)
        return jnp.sqrt(norm_upper(r_u) ** 2 + norm_lower(r_l) ** 2)

    x, info = refine(apply_res, solve, b, x0=x0, tol=tol, norm=norm, inner_dtype=inner_dtype)
    u, sigma = unpack(project_primal(x))
    return u, sigma, info
