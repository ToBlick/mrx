"""The second variation of the magnetic energy along volume-preserving flows, and the Newton direction it defines.

Along the flow of a divergence-free velocity ``u`` the 2-form ``B`` moves as
``B_t = curl(u x B_t)`` (the ideal induction of the relaxation: ``div B`` and
the helicity are kept). To second order in the flow time the pushed-forward
field is ``B + Q + R / 2`` with ``Q = curl(u x B)`` and ``R = curl(u x Q)``,
and the energy along it, ``E(u) = ||B + Q + R / 2||_M^2 / 2``, has the
gradient ``-load(J x B)`` at ``u = 0`` (the force before its Leray
projection, :func:`mrx.relaxation.compute_force`) and the Hessian

    (u, H v) = (Q_u, Q_v)_M + [(B, curl(u x Q_v))_M + (B, curl(v x Q_u))_M] / 2,

symmetric by construction. At an equilibrium ``H v = -load(dJ x B + J x Q_v)``
with ``dJ = curl Q_v``: minus the ideal-MHD force operator at ``p = 0``
(Bernstein et al. 1958), the operator SIESTA linearises (Hirshman et al.
2011); away from one the two halves of the second term differ by the first
variation along the commutator ``[u, v]``. The pressure of the relaxation is
the Leray multiplier and does not enter.

Newton's step on the energy restricted to divergence-free velocities solves
``H u = load(J x B)`` up to a gradient. Here the velocity is written as the
curl of a Dirichlet 1-form, ``u = curl a``, which is divergence-free exactly
(``div curl = 0`` on the incidence matrices) and turns the constrained system
into the symmetric one

    curl^T H curl a = curl^T M_2 F,

consistent by construction (the right-hand side annihilates every ``a`` whose
curl is in the kernel of ``H``; the gauge ``a + grad phi`` is in the kernel of
both sides and the curl removes it from the answer), solved by MINRES with the
harmonic atom as the preconditioner (:func:`harmonic_preconditioner`; the
k=1 Laplacian atom is the alternative). The one divergence-free direction
``curl a`` cannot represent is the
harmonic 2-form of the Dirichlet complex (the net toroidal flux, one DoF).
"""
import jax
import jax.numpy as jnp
import numpy as np

from mrx.operators import _dual_norm
from mrx.precision import RESIDUAL_DTYPE
from mrx.solvers import minres, pcg_steihaug, pcg_steihaug_tr, refine

#: The preconditioners of the Newton solve: the k=1 Laplacian atom, its
#: square (the operator is fourth order in ``a``), the k=1 mass atom, or the
#: harmonic atom (:func:`harmonic_preconditioner`).
PRECONDITIONERS = ("laplacian", "laplacian2", "mass", "harmonic")

#: The floor of the harmonic atom's parallel symbol, in units of
#: ``(2 pi)^2 (h_theta^2 + h_zeta^2)``: stands in for the ``u . grad h`` term
#: the symbol drops, which is what the flat (resonant) modes are left with.
#: ``None`` computes the floor from the strain of the field instead
#: (:func:`harmonic_atom_profiles`).
HARMONIC_FLOOR = 3.0


def _ddx(f, x, axis, periodic):
    """Central difference of ``f`` along ``axis`` on the non-uniform grid ``x``
    (the quadrature points of that axis); a periodic axis wraps on ``[0, 1)``,
    a bounded one is one-sided at its ends."""
    f = jnp.moveaxis(f, axis, 0)
    if periodic:
        df = jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)
        dx = (jnp.roll(x, -1) - jnp.roll(x, 1)) % 1.0
    else:
        df = jnp.concatenate([f[1:2] - f[0:1], f[2:] - f[:-2], f[-1:] - f[-2:-1]])
        dx = jnp.concatenate([x[1:2] - x[0:1], x[2:] - x[:-2], x[-1:] - x[-2:-1]])
    return jnp.moveaxis(df / dx.reshape((-1,) + (1,) * (f.ndim - 1)), 0, axis)


def harmonic_atom_profiles(seq, field):
    """The radial profiles the harmonic atom lumps a 2-form ``field`` to:
    ``(prof_t, prof_z, strain)`` on the radial quadrature points, ``prof_t``
    and ``prof_z`` the angle-averaged logical contravariant ``theta`` and
    ``zeta`` components (reference components over the Jacobian) and
    ``strain`` of shape ``(nq_r, 3)`` the angle average of ``sum_i (d_c
    field^i)^2`` per logical direction ``c``: the diagonal of ``S^T S`` for
    the strain ``S^i_c = d_c field^i`` (central differences on the
    quadrature grid), the size of the dropped ``u . grad field`` on a
    velocity along ``c``. Traceable in ``field``.
    """
    shape = tuple(int(v) for v in seq.quad.shape)
    f_jk = (seq.evaluate_at_quadrature(field, 2, True) / seq.jacobian_j[:, None]).reshape(shape + (3,))
    prof_t = f_jk[..., 1].mean(axis=(1, 2))
    prof_z = f_jk[..., 2].mean(axis=(1, 2))
    grads = [_ddx(f_jk, x, ax, ax > 0) for ax, x in enumerate((seq.quad.x_x, seq.quad.x_y, seq.quad.x_z))]
    strain = jnp.stack([(g ** 2).sum(axis=-1).mean(axis=(1, 2)) for g in grads], axis=-1)
    return prof_t, prof_z, strain


def parallel_penalty_profile(seq, field, alpha):
    """The radial weight of the parallel-flow penalty (:func:`second_variation`)
    on the radial quadrature points: ``alpha (2 pi)^2 (h_theta^2 + h_zeta^2)``
    for a number ``alpha`` (the atom floor's units, ``alpha = kappa`` the floor
    moved into the operator), or for the string ``"strain"`` / ``"c*strain"``
    ``c`` times the strain seen along the field, ``(h_theta^2 s_theta +
    h_zeta^2 s_zeta) / (h_theta^2 + h_zeta^2)`` with ``s`` the lumped strain of
    :func:`harmonic_atom_profiles`: the size of the ``u . grad h`` coupling a
    parallel mode really has, so that the operator lifts the null space to
    ``c`` times what the strain-floored atom models for it. The strain grows
    ~50x from the axis to the edge and its ratio to the floor scale differs
    between devices, while a number ``alpha`` is flat in ``r``; ``c`` measures
    how much the lumped strain undercounts the coupling (a property of the
    lumping, ~2-3 on li383 and W7-X, 2026-09-17), not of the geometry."""
    prof_t, prof_z, strain = harmonic_atom_profiles(seq, field)
    hsq = prof_t ** 2 + prof_z ** 2
    if isinstance(alpha, str):
        scale = float(alpha.split("*")[0]) if "*" in alpha else 1.0
        return scale * (prof_t ** 2 * strain[:, 1] + prof_z ** 2 * strain[:, 2]) / hsq
    return alpha * (2 * np.pi) ** 2 * hsq


def harmonic_preconditioner(seq, field, floor=HARMONIC_FLOOR, shift=0.0):
    """``x -> W P_L W^T x``: the harmonic atom, an approximate inverse of the Newton
    operator ``curl^T H curl`` built from the 2-form ``field`` (the harmonic
    form ``h`` of the sequence, or the current ``B``).

    The Hessian is, to a percent, the Gauss-Newton form ``||curl(u x B)||^2``,
    and ``B`` is mostly harmonic (96% on li383), so ``||curl(u x c h)||^2``
    with ``B = c h + curl A`` is the Hessian to a few percent on every mode
    but the flattest (measured: within 3% above the seventh Ritz value, 0.08
    at the lowest). With ``div u = 0``, ``curl(u x h) = h . grad u - u . grad h``;
    the atom keeps the parallel derivative and lumps it: per component of the
    1-form potential and per radial DoF layer, the symbol

        lambda(r, m, n) = (2 pi)^2 (h_theta(r) m + h_zeta(r) n)^2 + floor(r),

    ``h_theta, h_zeta`` the angle-averaged logical contravariant components
    of the field (:func:`harmonic_atom_profiles`), ``(m, n)`` the Fourier
    frequencies of the DoF grid in the two angles, and ``floor(r)`` for the
    dropped ``u . grad h``: ``floor * (2 pi)^2 (h_theta^2 + h_zeta^2)`` for a
    number ``floor`` (:data:`HARMONIC_FLOOR`), or with ``floor=None`` the
    lumped strain itself, per component the angle average of ``|d_c h|^2``
    (the lumped parallel derivative ``i k I`` is anti-Hermitian and the
    strain of a curl-free field symmetric, so the normal form of ``i k I - S``
    is ``k^2 I + S^T S`` with no cross term: the floor is computed, not
    tuned). The field carries the rotational transform, so the symbol
    vanishes on the resonant modes ``h_theta m + h_zeta n = 0`` and the floor
    is what they see. ``shift`` is added to the whole symbol as
    :func:`parallel_penalty_profile` of ``shift`` (a number in the floor's
    units, or ``"strain"`` / ``"c*strain"``): the
    ``parallel_penalty`` of :func:`second_variation`, so that the atom and
    the operator agree on what a parallel mode sees.
    Traceable in ``field``: built from the current ``B`` inside the step at
    the cost of one quadrature evaluation.

    Inverted as a sandwich of the Laplacian atom ``P_L`` (which approximates
    the inverse of the curl-curl the potential form is quadratic in) with the
    symbol's inverse square root: ``W = E C E^T`` with ``C`` the 2-D Fourier
    scaling by ``(lambda + floor)^{-1/2}`` on the tensor DoF grid of each
    component and ``E`` the Dirichlet 1-form extraction, so that in the bulk
    ``W P_L W^T`` is the Laplacian atom with ``lambda + floor`` multiplied
    into its denominator (the potential-form operator is the curl-curl times
    the parallel symbol) and on the polar rows the extraction lumps it.
    Symmetric positive definite for any ``C``, which is all MINRES needs;
    the quality is the measurement. Two FFTs per component per apply.
    """
    prof_t, prof_z, strain = harmonic_atom_profiles(seq, field)
    penalty = parallel_penalty_profile(seq, field, shift) if isinstance(shift, str) or shift else None
    r_q = seq.quad.x_x
    shapes = [tuple(int(v) for v in s) for s in seq.basis_1.shape]
    scale = []
    for c, (s1, s2, s3) in enumerate(shapes):
        r = (jnp.arange(s1) + 0.5) / s1
        a, b = jnp.interp(r, r_q, prof_t), jnp.interp(r, r_q, prof_z)
        shift_r = jnp.interp(r, r_q, penalty)[:, None, None] if penalty is not None else 0.0
        m = np.fft.fftfreq(s2, d=1.0 / s2)
        nn = np.fft.fftfreq(s3, d=1.0 / s3)
        lam = (2 * np.pi) ** 2 * (a[:, None, None] * m[None, :, None] + b[:, None, None] * nn[None, None, :]) ** 2
        if floor is None:
            flo = jnp.interp(r, r_q, strain[:, c])[:, None, None]
        else:
            flo = floor * (2 * np.pi) ** 2 * (a ** 2 + b ** 2)[:, None, None]
        scale.append((1.0 / jnp.sqrt(lam + flo + shift_r)).astype(seq.dtype))
    E = seq.E(1, True)

    def C(x):
        out, off = [], 0
        for sc, s in zip(scale, shapes):
            n_c = s[0] * s[1] * s[2]
            X = x[off:off + n_c].reshape(s)
            out.append(jnp.fft.ifft2(jnp.fft.fft2(X, axes=(1, 2)) * sc, axes=(1, 2)).real.ravel())
            off += n_c
        return jnp.concatenate(out)

    def apply(x):
        y = E @ C(E.T @ x)
        y = seq.apply_laplacian_preconditioner(y, 1, dirichlet=True)
        return E @ C(E.T @ y)

    return apply


def second_variation(seq, B, J, tol=None, parallel_penalty=0.0):
    """``u -> H u``: the Hessian of the energy along the flow of ``u``, as a dual 2-form.

    ``B`` the 2-form, ``J`` its weak curl (a Dirichlet 1-form, the ``J`` of
    :func:`mrx.relaxation.compute_force`). Three k=1 mass solves per apply,
    each to ``tol`` (the sequence's by default): ``E = M_1^-1 load(u x B)``
    for ``Q = curl E``, the weak curl ``dJ`` of ``Q``, and ``W = M_1^-1
    curl^T load(J x u)``; then

        H u = load(B x dJ) + [load(Q x J) + load(B x W)] / 2.

    ``parallel_penalty = alpha`` adds ``alpha M_par u`` with ``<v, M_par u> =
    int (v . B)(u . B) / |B|^2 J``: the Hessian is exactly null on the
    field-aligned flows ``u = f B`` (``curl(f B x B) = 0``; divergence-free
    wherever ``B . grad f = 0``, so on every flux surface and with a resonant
    ``f`` on every rational surface), and on the mesh that null space is a
    continuum of eigenvalues 1e-4..1e-1 (li383 (16,32,32), measured
    2026-09-17, docs/research/hessian_spectrum_2026-09-17.md) that Newton
    divides the force's round-off components by. The penalty is
    Levenberg-Marquardt damping on the parallel component alone: it lifts
    those modes, leaves the energy descent unchanged (``<F, f B> = 0``) and
    the perpendicular step untouched. The weight is
    :func:`parallel_penalty_profile` of ``alpha`` with the profiles of ``B``:
    a number in the units of the atom's floor, ``(2 pi)^2 (h_theta^2 +
    h_zeta^2)(r)`` (the Hessian's scale is the field's logical gradient
    scale, 35x smaller on W7-X than on li383; ``alpha = kappa`` is exactly the
    atom's floor moved into the operator; li383 optimum 0.075, measured
    2026-09-17), or ``"strain"`` / ``"c*strain"`` for ``c`` times the strain
    along the field, computed. One quadrature load per apply.
    """
    B_jk = seq.evaluate_at_quadrature(B, 2, True)
    J_jk = seq.evaluate_at_quadrature(J, 1, True)
    penalised = isinstance(parallel_penalty, str) or parallel_penalty != 0
    if penalised:
        Bsq_over_J2 = jnp.einsum('qi,qij,qj->q', B_jk, seq.metric_jkl, B_jk) / seq.jacobian_j ** 2
        n_angles = int(seq.quad.shape[1]) * int(seq.quad.shape[2])
        weight = jnp.repeat(parallel_penalty_profile(seq, B, parallel_penalty), n_angles)

    def m1_inv(rhs):
        return seq.apply_inverse_mass_matrix(rhs, 1, dirichlet=True, tol=tol)

    def parallel(u_jk):
        s = jnp.einsum('qi,qij,qj->q', u_jk, seq.metric_jkl, B_jk) / seq.jacobian_j ** 2 / Bsq_over_J2
        return seq._vector_load_values(B_jk * (weight * s)[:, None], 2, 2, True)

    def apply(u):
        u_jk = seq.evaluate_at_quadrature(u, 2, True)
        E = m1_inv(seq.cross_product_load_values(u_jk, B_jk, 1, 2, 2, True))
        Q = seq.apply_incidence_matrix(E, 1, dirichlet_in=True, dirichlet_out=True)
        Q_jk = seq.evaluate_at_quadrature(Q, 2, True)
        dJ = m1_inv(seq.apply_derivative_matrix(Q, 1, dirichlet_in=True, dirichlet_out=True,
                                                transpose=True))
        dJ_jk = seq.evaluate_at_quadrature(dJ, 1, True)
        JxU = seq.cross_product_load_values(J_jk, u_jk, 2, 1, 2, True)
        W = m1_inv(seq.apply_incidence_matrix(JxU, 1, dirichlet_in=True, dirichlet_out=True,
                                              transpose=True))
        W_jk = seq.evaluate_at_quadrature(W, 1, True)
        Hu = (seq.cross_product_load_values(B_jk, dJ_jk, 2, 2, 1, True)
              + 0.5 * (seq.cross_product_load_values(Q_jk, J_jk, 2, 2, 1, True)
                       + seq.cross_product_load_values(B_jk, W_jk, 2, 2, 1, True)))
        return Hu + parallel(u_jk) if penalised else Hu

    return apply


def _preconditioner(seq, name):
    if callable(name):
        return name
    if name == "harmonic":
        return harmonic_preconditioner(seq, seq.nullspace(2, True)[0])
    if name == "laplacian":
        return lambda x: seq.apply_laplacian_preconditioner(x, 1, dirichlet=True)
    if name == "laplacian2":
        return lambda x: seq.apply_laplacian_preconditioner(
            seq.apply_laplacian_preconditioner(x, 1, dirichlet=True), 1, dirichlet=True)
    if name == "mass":
        return lambda x: seq.apply_mass_matrix_preconditioner(x, 1, dirichlet=True)
    raise ValueError(f"newton_precond {name!r} is not one of {PRECONDITIONERS}")


NEWTON_SOLVERS = ("minres", "cg")


def _newton_system(seq, B, J, parallel_penalty):
    """``(curl, curl_t, A)`` of the Newton system ``curl^T H curl a = curl^T M_2 F`` on ``seq``."""
    Hs = second_variation(seq, B.astype(seq.dtype), J.astype(seq.dtype), parallel_penalty=parallel_penalty)

    def curl(a):
        return seq.apply_incidence_matrix(a, 1, dirichlet_in=True, dirichlet_out=True)

    def curl_t(y):
        return seq.apply_incidence_matrix(y, 1, dirichlet_in=True, dirichlet_out=True, transpose=True)

    def A(a):
        return curl_t(Hs(curl(a)))
    return curl, curl_t, A


def newton_direction_tr(seq, B, J, MF, delta, tol=0.03, maxiter=300, precond="harmonic",
                        parallel_penalty=0.0):
    """The trust-region Newton direction (Nocedal & Wright 7.2): the minimiser of the quadratic
    model ``m(a) = -b^T a + a^T A a / 2``, ``b = curl^T M_2 F``, over ``||a||_{P^-1} <= delta`` by
    :func:`mrx.solvers.pcg_steihaug_tr` from a zero guess. Returns ``(u, a, info, hit,
    predicted, delta)`` with ``predicted = -m(a) = b^T a - a^T A a / 2`` the model's energy
    decrease for the step ``dt = 1`` along ``u = curl a``, ``hit`` True when the step is on the
    boundary, and ``delta`` the radius used (``delta = 0`` on entry means the natural unit,
    ``||b||_P``); the caller takes the step, measures the actual decrease, and moves ``delta``.
    """
    curl, curl_t, A = _newton_system(seq, B, J, parallel_penalty)
    P = _preconditioner(seq, precond)
    b = curl_t(MF)
    # a radius of 0 asks for the natural unit: ||b||_P = ||P b||_{P^-1}, the preconditioned
    # gradient's length in the region's norm (the Newton step's when P ~ A^-1); the ball must be
    # finite from the first solve (an infinite one sends a negative-curvature step to infinity)
    delta = jnp.where(delta > 0, delta, jnp.sqrt(b @ P(b)))
    a, info, hit = pcg_steihaug_tr(A, b, delta, M=P, tol=tol, maxiter=maxiter)
    predicted = b @ a - 0.5 * (a @ A(a))
    return curl(a), a, jnp.asarray(info, dtype=jnp.int32), hit, predicted, delta


def newton_direction(seq, B, J, MF, a_guess, tol=0.1, maxiter=300, precond="laplacian",
                     parallel_penalty=0.0, inner_tol=0.0, solver="minres", passes=1):
    """The Newton direction ``u = curl a`` at the field ``B``.

    ``J`` the weak curl of ``B``, ``MF = M_2 F`` the mass times the
    Leray-projected force (``curl^T M_2 F`` is ``curl^T load(J x B)``
    exactly: the gradient part is a ``D_2^T``, and ``D_2 D_1 = 0``),
    ``a_guess`` the previous direction's potential (the warm start of the
    MINRES solve), ``tol`` the relative residual of the solve in the
    preconditioner norm, ``maxiter`` its iteration budget, ``precond`` one
    of :data:`PRECONDITIONERS` or the preconditioner's apply itself (a
    callable), ``parallel_penalty`` the ``alpha`` of :func:`second_variation`,
    ``inner_tol`` the inner solve's own stopping tolerance (below), ``solver``
    one of :data:`NEWTON_SOLVERS`: MINRES (symmetric, indefinite allowed) or
    CG with the Steihaug-Toint negative-curvature exit
    (:func:`mrx.solvers.pcg_steihaug`, the truncated-Newton solve of Nocedal &
    Wright 7.2: minimises the energy of the direction rather than the
    residual, and returns a descent direction when it meets negative
    curvature); the Hessian's mass solves run at the sequence's tolerance.

    The inner solve: from ``a_guess`` up to ``maxiter`` iterations,
    stopping early when its residual estimate is below ``inner_tol`` times
    the right-hand side in the PRECONDITIONER's norm (the forcing term of an
    inexact Newton method, Dembo-Eisenstat-Steihaug); ``inner_tol = 0`` runs
    the whole budget. That norm is not the true residual's: the harmonic
    atom met sqrt(0.1) in it after 62 of 1000 iterations while the true
    residual was far above (2026-09-07), so ``inner_tol`` is calibrated, not
    chosen (2026-09-18, li383 with the parallel penalty: see
    docs/research/hessian_spectrum_2026-09-17.md 7e). Then ONE pass of
    :func:`mrx.solvers.refine` measures the true residual like every solve
    in the code, in the mass-atom norm of the dual 1-forms on the residual
    view, ``tol`` relative to the right-hand side, and reports it. With
    ``passes > 1`` that measurement is the forcing term of the inexact Newton
    method in the code's own convention (the float64 residual, not the
    solver's float32 recurrence): another ``maxiter`` iterations on the
    residual until it is below ``tol`` or the passes are spent, one float64
    operator apply per pass. One pass was the released choice because with the
    parallel penalty the direction stops changing once that residual is ~0.1
    (200 iterations on li383; 400 and 800 give the same relaxation to three
    digits), and without it more iterations put more of the null space into
    the step. Returns ``(u, a, info)`` with ``info`` the iteration count,
    negative when the true residual met ``tol``.
    """
    ops = seq._require_operators(None)
    on = seq if seq.residual is None else seq.residual

    def chain(s):
        Hs = second_variation(s, B.astype(s.dtype), J.astype(s.dtype),
                              parallel_penalty=parallel_penalty)

        def curl(a):
            return s.apply_incidence_matrix(a, 1, dirichlet_in=True, dirichlet_out=True)

        def curl_t(y):
            return s.apply_incidence_matrix(y, 1, dirichlet_in=True, dirichlet_out=True,
                                            transpose=True)

        def A(a):
            u = curl(a)
            return curl_t(Hs(u))
        return curl, curl_t, A

    curl, curl_t, A = chain(seq)
    A_res = chain(on)[2]
    P = _preconditioner(seq, precond)
    if solver not in NEWTON_SOLVERS:
        raise ValueError(f"newton_solver {solver!r} is not one of {NEWTON_SOLVERS}")
    if solver == "cg":
        a, info = refine(A_res, lambda r: pcg_steihaug(A, r, M=P, tol=inner_tol, maxiter=maxiter),
                         curl_t(MF), x0=a_guess, tol=tol, norm=_dual_norm(ops, 1, True),
                         max_passes=passes, inner_dtype=seq.dtype)
        return curl(a.astype(seq.dtype)), a.astype(seq.dtype), jnp.asarray(info, dtype=jnp.int32)
    a, info, npc = newton_mr(A_res, A, P, curl_t(MF), a_guess, tol, maxiter, passes,
                             _dual_norm(ops, 1, True), inner_dtype=seq.dtype, inner_tol=inner_tol)
    a = a.astype(seq.dtype)
    return curl(a), a, jnp.asarray(info, dtype=jnp.int32)


def newton_mr(A_res, A, P, b, x0, tol, maxiter, passes, norm, inner_dtype, inner_tol=0.0):
    """The Newton-MR solve of ``A x = b`` (Liu & Roosta 2022): MINRES on the float64 residual
    in passes of ``maxiter`` iterations from the warm start ``x0``, until ``norm(b - A x) <=
    tol norm(b)`` (the forcing term, in the residual's norm and precision) or ``passes`` are
    spent, with the nonpositive-curvature exit. Each pass solves for the correction from
    zero on the residual at unit norm, as :func:`mrx.solvers.refine` does; Liu & Roosta's
    descent guarantee for an NPC direction holds against the right-hand side that solve
    started from, which is the pass's residual, not ``b`` (measured: with the warm start,
    the NPC direction of the correction solve failed the energy's sign test on every W7-X
    step). So a pass that meets NPC discards its answer and solves ``A x = b`` from zero
    with the exit once more: what that returns -- the NPC direction, or the iterate when
    the curvature was an artefact of the warm start's residual -- is descent-guaranteed
    for ``b`` and is the answer, alone. Returns ``(x, info, npc)`` with ``info`` the inner
    iterations of all passes, negative when the residual test was met."""
    b = b.astype(RESIDUAL_DTYPE)
    x = jnp.zeros_like(b) if x0 is None else x0.astype(RESIDUAL_DTYPE)
    bnorm = norm(b)
    bnorm_safe = jnp.where(bnorm > 0, bnorm, 1.0)

    def cond(carry):
        _, r, k, _, npc = carry
        return jnp.logical_and(jnp.logical_and(norm(r) > tol * bnorm_safe, k < passes), ~npc)

    def body(carry):
        x, r, k, its, _ = carry
        rnorm = norm(r)
        rnorm_safe = jnp.where(rnorm > 0, rnorm, 1.0)
        d, info, npc = minres(A, (r / rnorm_safe).astype(inner_dtype), M=P, tol=inner_tol,
                              maxiter=maxiter, npc_exit=True)
        d = d.astype(RESIDUAL_DTYPE) * rnorm_safe

        def from_zero(_):
            d0, info0, _ = minres(A, (b / bnorm_safe).astype(inner_dtype), M=P, tol=inner_tol,
                                  maxiter=maxiter, npc_exit=True)
            return (d0.astype(RESIDUAL_DTYPE) * bnorm_safe).astype(RESIDUAL_DTYPE), jnp.abs(info0).astype(jnp.int32)

        x_npc, its_npc = jax.lax.cond(npc, from_zero, lambda _: ((x + d).astype(RESIDUAL_DTYPE), jnp.zeros((), jnp.int32)), None)
        x_new = jnp.where(npc, x_npc, x + d)
        r_new = b - A_res(x_new)
        return x_new, r_new, k + 1, (its + jnp.abs(info) + its_npc).astype(jnp.int32), npc

    r0 = b - A_res(x)
    x, r, k, its, npc = jax.lax.while_loop(cond, body, (x, r0, 0, jnp.int32(0), False))
    converged = norm(r) <= tol * bnorm_safe
    return x, jnp.where(converged, -its, its), npc
