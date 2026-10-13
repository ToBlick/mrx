"""Matrix-free Krylov solvers used by all linear solves in MRX.

The solvers take the operator and the preconditioner as functions ``x -> A x``, so they work
inside ``jit`` and never form a matrix. :func:`preconditioned_cg` solves SPD systems, and
:func:`minres` solves symmetric indefinite ones. :func:`minres` can also stop at a direction of
nonpositive curvature and reorthogonalise its Lanczos vectors, which the Newton solve uses.
:func:`solve_singular_cg` solves a semidefinite system with a known kernel. :func:`lanczos_bounds`
estimates the spectral interval of a preconditioned operator. :func:`refine` is the outer loop of a
solve: it recomputes the true residual after each Krylov solve and corrects the solution until that
residual meets the tolerance. In float32 it carries the solution as a double-single number and forms the
residual in that arithmetic (:mod:`mrx.precision`), which is what lets a float32 Krylov solve reach a tolerance
below its own round-off. The Krylov solvers and the preconditioners only ever see float arrays.

Every Krylov solver returns a signed iteration count ``info``. A value ``+k`` means converged after ``k``
iterations, and ``-k`` means not converged after ``k`` iterations.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

import mrx.double_single as ds
from mrx.precision import DTYPE, MAX_PASSES, PAIRS, lift, pass_floor, to_working


def preconditioned_cg(A_matvec, b, M, tol, maxiter, x0=None, norm=None):
    """Solve the SPD system ``A x = b`` by preconditioned CG and return ``(x, info)``.

    ``M`` is an SPD approximation of ``A^{-1}``. The solve stops when ``norm(r) < tol norm(b)``
    or after ``maxiter`` iterations. ``norm`` is the norm of the residual's space
    (:func:`mrx.operators.dual_norm`, the ``M_k^{-1}`` norm), the one every solve is judged in.
    Without it the residual is measured in the norm of the preconditioner, ``sqrt(r^T M r)``, which
    is the same number when ``M`` is the mass preconditioner. If ``A`` or ``M`` is not SPD the
    result becomes NaN.
    """
    if x0 is None:
        x0 = jnp.zeros_like(b)

    def size(r, rz):
        return jnp.sqrt(rz) if norm is None else norm(r)

    r0 = b - A_matvec(x0)
    z0 = M(r0)
    rz0 = jnp.dot(r0, z0)
    bnorm = jnp.sqrt(jnp.dot(b, M(b))) if norm is None else norm(b)
    bnorm_safe = jnp.where(bnorm > 0, bnorm, 1.0)

    # state: (x, r, z, p, rz, k, converged)
    init_state = (x0, r0, z0, z0, rz0, 0, size(r0, rz0) < tol * bnorm_safe)

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
        return (x_new, r_new, z_new, p_new, rz_new, k + 1, size(r_new, rz_new) < tol * bnorm_safe)

    x_final, _, _, _, _, k_final, converged_final = jax.lax.while_loop(cond_fn, body_fn, init_state)
    return x_final, jnp.where(converged_final, k_final, -k_final)


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
    beta_prev = jnp.concatenate([jnp.zeros(1), beta[:-1]])
    alpha_prev = jnp.concatenate([jnp.ones(1), alpha[:-1]])
    diag = 1 / alpha + beta_prev / alpha_prev
    off = jnp.sqrt(beta[:-1]) / alpha[:-1]
    ritz = jnp.linalg.eigvalsh(jnp.diag(diag) + jnp.diag(off, 1) + jnp.diag(off, -1))
    return ritz[0], ritz[-1]


def deflation_projectors(vs, mass_matvec):
    """Return the projections ``(project_primal, project_dual)`` that remove a kernel.

    The rows of ``vs`` (shape ``(m, n)``) are the kernel vectors, orthonormal in the mass
    inner product ``M``. ``project_primal(x) = x - (vs M x) vs`` acts on coefficient vectors and
    ``project_dual(f) = f - (vs f) (M vs)`` on right-hand sides and residuals. With ``m = 0``
    both are the identity. ``M`` is symmetric, so ``vs M x`` is the product of the stored ``M vs``
    with ``x``, and neither projection applies the mass matrix. Double-single ``vs`` give double-single
    projections.
    """
    if vs.shape[0] == 0:
        return (lambda x: x), (lambda f: f)
    if isinstance(vs, ds.DS):
        rows = [vs[i] for i in range(vs.shape[0])]
        mass_rows = [mass_matvec(v) for v in rows]

        def remove(x, along, against):
            x = ds.lift(x)
            for v, w in zip(along, against):
                x = x - v * ds.dot(w, x)
            return x
        return (lambda x: remove(x, rows, mass_rows)), (lambda f: remove(f, mass_rows, rows))
    vs = jnp.asarray(vs)
    mass_vs = jax.vmap(mass_matvec)(vs)

    def project_primal(x):
        return x - (mass_vs @ x) @ vs

    def project_dual(f):
        return f - (vs @ f) @ mass_vs

    return project_primal, project_dual


#: The weakest reduction of the residual that a pass of :func:`refine` asks for.
PASS_CAP = 1e-2


def refine(apply, solve, b, *, tol, norm, x0=None, project=None, residual=None):
    """Solve ``A x = b`` by repeated corrections and return ``(x, info)``.

    Each pass computes the residual ``r = b - A x`` with ``apply`` (``x -> A x``), solves
    ``A d = r`` approximately with ``solve(r, rtol) -> (d, info)`` and adds ``d`` to ``x``. It stops
    when ``norm(r) <= tol norm(b)`` or after
    :data:`~mrx.precision.MAX_PASSES` passes. ``info`` counts the inner iterations of all passes.

    ``rtol`` aims at half the target: it is ``tol norm(b) / (2 norm(r))``, between
    :func:`~mrx.precision.pass_floor` and :data:`PASS_CAP`. So a pass does not ask for much more than the
    stopping test needs, however good the starting guess: an iteration pushed far past that runs into its
    round-off, where conjugate gradients on a singular system diverge. The cap keeps a pass from asking for
    almost nothing: the composite solves deliver several times less than ``rtol`` in ``norm``, and a pass that
    asks for a factor 10 then hardly moves.

    In float32 ``x`` is a :class:`~mrx.double_single.DS` and so is ``apply(x)``. The residual is formed in
    double-single, where the cancellation of ``b`` against ``A x`` loses nothing, and then rounded: ``solve``,
    ``norm`` and ``project`` see float arrays. ``b`` may be a float array or a :class:`~mrx.double_single.DS`.

    ``norm`` measures residuals. The callers pass the norm given by the mass preconditioner of the
    residual's space. ``project`` is applied to every residual, to keep it in the range of ``A``. It acts on
    the rounded residual, so a part of ``b`` outside that range has to be removed from ``b`` beforehand, to the
    accuracy of ``b``. A solve whose residual is not of the form ``b - A x`` passes ``residual(x)`` and ``x0``
    instead of ``apply``, and ``b`` is then used only for its norm.
    """
    if project is None:
        def project(r): return r
    if residual is None:
        x0 = jnp.zeros(b.shape, dtype=DTYPE) if x0 is None else x0

        def residual(x):
            return b - apply(x)
    x = lift(x0)

    def rounded(x):
        return project(to_working(residual(x)))
    bnorm = norm(project(to_working(b)))
    bnorm_safe = jnp.where(bnorm > 0, bnorm, 1.0)

    def cond(carry):
        _, r, k, _ = carry
        return jnp.logical_and(norm(r) > tol * bnorm_safe, k < MAX_PASSES)

    def body(carry):
        x, r, k, its = carry
        # The inner solve gets the residual scaled to unit norm, so the range of float32 never limits the
        # tolerance. s > 0 inside the loop.
        s = norm(r)
        d, info = solve(r / s, jnp.clip(0.5 * tol * bnorm_safe / s, pass_floor(tol), PASS_CAP))
        x = x + s * d
        return x, rounded(x), k + 1, its + jnp.abs(info)

    x, r, _, its = jax.lax.while_loop(cond, body, (x, rounded(x), 0, 0))
    converged = norm(r) <= tol * bnorm_safe
    return x, jnp.where(converged, its, -its)


def solve_singular_cg(A_matvec, b, vs, *, precond_matvec, tol, maxiter, mass_matvec=None,
                      x0=None, check_residual=False, norm=None):
    """Solve the symmetric semidefinite system ``A x = b`` by PCG and return ``(x, info)``.

    The rows of ``vs`` (shape ``(m, n)``, ``m`` may be 0) span the kernel of ``A`` and are
    orthonormal in the mass inner product ``mass_matvec``. The kernel is removed from the
    iterates and the residuals, so ``x`` is the solution orthogonal to it.

    The residual is measured in ``norm`` (:func:`mrx.operators.dual_norm`), by the PCG as well.
    Without ``norm`` it is measured in the norm of the preconditioner, ``sqrt(r^T P r)``, which the
    PCG has at no cost: pass no ``norm`` when it would be that norm anyway (a mass solve).
    Without ``check_residual`` this is a plain PCG that stops at ``tol`` on the residual of its
    recurrence. With it the PCG runs inside :func:`refine` and the true residual measured in ``norm``
    decides convergence. ``A_matvec`` and ``mass_matvec`` must then follow their argument: a
    :class:`~mrx.double_single.DS` in, a :class:`~mrx.double_single.DS` out.
    """
    project_primal, project_dual = deflation_projectors(to_working(vs), mass_matvec)

    def A(x):
        return project_dual(A_matvec(project_primal(x)))

    def P(x):
        return project_primal(precond_matvec(project_dual(x)))

    def outer_norm(r):
        return jnp.sqrt(r @ P(r)) if norm is None else norm(r)

    if not check_residual:
        x0 = jnp.zeros_like(b) if x0 is None else project_primal(x0)
        x, info = preconditioned_cg(A, project_dual(b), P, tol, maxiter, x0=x0, norm=norm)
        return project_primal(x), info

    # The kernel leaves b and the solution once each, to their own accuracy. The corrections come out of the
    # conjugate gradients in the projected space already.
    exact_primal, exact_dual = deflation_projectors(vs, mass_matvec) if PAIRS else (project_primal, project_dual)
    x, info = refine(A_matvec, lambda r, rtol: preconditioned_cg(A, r, P, rtol, maxiter, norm=norm),
                     exact_dual(lift(b)), x0=None if x0 is None else project_primal(x0), tol=tol,
                     project=project_dual, norm=outer_norm)
    return exact_primal(x), info


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
    res: jnp.ndarray
    k: int
    converged: bool
    npc: bool
    U: jnp.ndarray
    V: jnp.ndarray


def minres(A_matvec, b, *, M, tol, maxiter, norm, npc_exit=False, reorthogonalize=False):
    """Solve the symmetric (possibly indefinite) system ``A x = b`` by preconditioned MINRES from zero.

    Returns ``(x, info)``. ``M`` is an SPD approximation of ``A^{-1}``. The solve stops when
    ``norm(r) < tol norm(b)`` for the residual ``r = b - A x``, which is carried by its recurrence
    ``r_k = s_k^2 r_{k-1} - c_k phibar_k z_{k+1} / beta_{k+1}`` (``z`` the Lanczos vectors before
    the preconditioner, Choi, Paige & Saunders 2011). ``norm`` is the norm of the residual's space
    (:func:`mrx.operators.dual_norm`), the one every solve is judged in.

    With ``npc_exit=True`` the solve also stops at the first residual ``r`` with
    ``<r, A r> <= 0`` and returns ``M r`` in place of the iterate. This is the
    nonpositive-curvature exit of Newton-MR (Liu & Roosta 2022, Algorithm 1), and ``M r`` is a
    descent direction of nonpositive curvature. The return is then ``(x, info, npc)`` with
    ``npc`` true if the exit was taken.

    With ``reorthogonalize=True`` every Lanczos vector is kept, together with its preconditioned
    version, and each new one is made orthogonal to all earlier ones, twice. That costs
    ``2 maxiter`` stored vectors and two products with them per iteration, and ``maxiter`` must be a
    Python int. Without it the three-term recurrence keeps orthogonality only in exact arithmetic.
    Once the extreme Ritz values have converged, the basis drifts, those values are found again and
    the solve repeats work. In the Newton solve of li383 (48,64,24) this happens after about 20
    iterations, and the top Ritz value was found 22 times in 400 (2026-10-14).
    """
    x0 = jnp.zeros_like(b)
    # A negative r^T M r means M is not SPD. The result then becomes NaN.
    r0 = b
    y0 = M(r0)
    beta1 = jnp.sqrt(jnp.dot(r0, y0))
    bnorm = norm(b)
    bnorm_safe = jnp.where(bnorm > 0, bnorm, 1.0)
    # the Lanczos vectors (rows of U) and their preconditioned versions (rows of V): u_i . v_j = delta_ij
    store = (maxiter if reorthogonalize else 0, b.shape[0])

    # The state follows SOL MINRES. y is the preconditioned residual (v = y / beta), r1 and r2
    # are the previous and current Lanczos residuals, (cs, sn) is the last Givens rotation,
    # dbar and epsln are the QR state, phibar is the residual estimate, and w_prev and w_pp
    # are the search directions one and two steps back.
    init_state = _MinresState(
        x=x0, y=y0, r1=jnp.zeros_like(b), r2=r0, beta=beta1, oldbeta=0.0,
        cs=-1.0, sn=0.0, dbar=0.0, epsln=0.0, phibar=beta1,
        w_prev=jnp.zeros_like(b), w_pp=jnp.zeros_like(b), res=r0, k=0, converged=False, npc=False,
        U=jnp.zeros(store, dtype=b.dtype), V=jnp.zeros(store, dtype=b.dtype),
    )

    def cond_fn(state):
        return jnp.logical_and(state.k < maxiter, ~(state.converged | state.npc))

    def body_fn(state):
        (x, y, r1, r2, beta, oldbeta, cs, sn, dbar, epsln, phibar,
         w_prev, w_pp, res, k, converged, _, U, V) = state

        safe_beta = jnp.where(beta > 0, beta, 1.0)
        v = y / safe_beta
        y_new = A_matvec(v)
        # alpha = v^T A v before the subtractions (drift-free at high iteration counts)
        alpha = jnp.dot(v, y_new)
        old_beta = jnp.where(oldbeta > 0, oldbeta, 1.0)
        y_new = y_new - jnp.where(k >= 1, beta / old_beta, 0.0) * r1
        y_new = y_new - (alpha / safe_beta) * r2
        if reorthogonalize:
            U, V = U.at[k].set(r2 / safe_beta), V.at[k].set(v)
            for _ in range(2):
                y_new = y_new - (V @ y_new) @ U

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
        # the residual of the new iterate (frozen with it at the NPC exit)
        safe_beta_new = jnp.where(beta_new > 0, beta_new, 1.0)
        res_new = jnp.where(npc_new, res, sn_new**2 * res - (cs_new * phibar_new / safe_beta_new) * y_new)
        converged_new = norm(res_new) < tol * bnorm_safe

        return _MinresState(
            x=x_new, y=y_prec, r1=r2, r2=y_new, beta=beta_new, oldbeta=beta,
            cs=cs_new, sn=sn_new, dbar=dbar_new, epsln=epsln_new, phibar=phibar_new,
            w_prev=w_new, w_pp=w_prev, res=res_new, k=k + 1,
            converged=converged_new & ~npc_new, npc=npc_new, U=U, V=V,
        )

    final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)
    x_final = final_state.x
    info = jnp.where(final_state.converged, final_state.k, -final_state.k)
    if npc_exit:
        # The NPC direction is the preconditioned residual of the frozen iterate.
        d_npc = M(b - A_matvec(x_final))
        return jnp.where(final_state.npc, d_npc, x_final), info, final_state.npc
    return x_final, info
