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

    curl^T (H + shift M_2) curl a = curl^T M_2 F,

consistent by construction (the right-hand side annihilates every ``a`` whose
curl is in the kernel of ``H``; the gauge ``a + grad phi`` is in the kernel of
both sides and the curl removes it from the answer), solved by MINRES with the
k=1 Laplacian atom as the preconditioner. ``shift`` is the Levenberg-Marquardt
shift in the velocity's L2 metric: 0 is Newton, a large shift is steepest
descent ``u = F / shift`` (the line search removes the scale), and in between
the modes of ``H`` above the shift are solved for and the ones below descended
along. The one divergence-free direction ``curl a`` cannot represent is the
harmonic 2-form of the Dirichlet complex (the net toroidal flux, one DoF).
"""
import jax.numpy as jnp
import numpy as np

from mrx.operators import _dual_norm, _outer
from mrx.solvers import minres, refine

#: The preconditioners of the Newton solve: the k=1 Laplacian atom, its
#: square (the operator is fourth order in ``a``), the k=1 mass atom, or the
#: harmonic atom (:func:`harmonic_preconditioner`).
PRECONDITIONERS = ("laplacian", "laplacian2", "mass", "harmonic")

#: The floor of the harmonic atom's parallel symbol, in units of
#: ``(2 pi)^2 (h_theta^2 + h_zeta^2)``: stands in for the ``u . grad h`` term
#: the symbol drops, which is what the flat (resonant) modes are left with.
HARMONIC_FLOOR = 1e-2


def harmonic_preconditioner(seq, floor=HARMONIC_FLOOR):
    """``x -> W P_L W^T x``: the harmonic atom, an approximate inverse of the Newton
    operator ``curl^T H curl`` built from the harmonic 2-form ``h`` of the sequence.

    The Hessian is, to a percent, the Gauss-Newton form ``||curl(u x B)||^2``,
    and ``B`` is mostly harmonic (96% on li383), so ``||curl(u x c h)||^2``
    with ``B = c h + curl A`` is the Hessian to a few percent on every mode
    but the flattest (measured: within 3% above the seventh Ritz value, 0.08
    at the lowest). With ``div u = 0``, ``curl(u x h) = h . grad u - u . grad h``;
    the atom keeps the parallel derivative and lumps it: per component of the
    1-form potential and per radial DoF layer, the symbol

        lambda(r, m, n) = (2 pi)^2 (h_theta(r) m + h_zeta(r) n)^2 + floor(r),

    ``h_theta, h_zeta`` the angle-averaged logical contravariant components
    of ``h`` (the reference components over the Jacobian), ``(m, n)`` the
    Fourier frequencies of the DoF grid in the two angles, and ``floor(r) =
    floor * (2 pi)^2 (h_theta^2 + h_zeta^2)`` for the dropped ``u . grad h``
    (:data:`HARMONIC_FLOOR`). ``h`` carries the rotational transform of the
    vacuum field inside the boundary, so the symbol vanishes on the resonant
    modes ``h_theta m + h_zeta n = 0`` and the floor is what they see.

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
    h = seq.nullspace(2, True)[0]
    h_jk = np.asarray(seq.evaluate_at_quadrature(h, 2, True)) / np.asarray(seq.jacobian_j)[:, None]
    shape = tuple(int(v) for v in seq.quad.shape)
    prof_t = h_jk[:, 1].reshape(shape).mean(axis=(1, 2))
    prof_z = h_jk[:, 2].reshape(shape).mean(axis=(1, 2))
    r_q = np.asarray(seq.quad.x_x)
    shapes = [tuple(int(v) for v in s) for s in seq.basis_1.shape]
    scale = []
    for s1, s2, s3 in shapes:
        r = (np.arange(s1) + 0.5) / s1
        a, b = np.interp(r, r_q, prof_t), np.interp(r, r_q, prof_z)
        m = np.fft.fftfreq(s2, d=1.0 / s2)
        nn = np.fft.fftfreq(s3, d=1.0 / s3)
        lam = (2 * np.pi) ** 2 * (a[:, None, None] * m[None, :, None] + b[:, None, None] * nn[None, None, :]) ** 2
        flo = floor * (2 * np.pi) ** 2 * (a ** 2 + b ** 2)[:, None, None]
        scale.append(jnp.asarray(1.0 / np.sqrt(lam + flo), dtype=seq.dtype))
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


def second_variation(seq, B, J, tol=None):
    """``u -> H u``: the Hessian of the energy along the flow of ``u``, as a dual 2-form.

    ``B`` the 2-form, ``J`` its weak curl (a Dirichlet 1-form, the ``J`` of
    :func:`mrx.relaxation.compute_force`). Three k=1 mass solves per apply,
    each to ``tol`` (the sequence's by default): ``E = M_1^-1 load(u x B)``
    for ``Q = curl E``, the weak curl ``dJ`` of ``Q``, and ``W = M_1^-1
    curl^T load(J x u)``; then

        H u = load(B x dJ) + [load(Q x J) + load(B x W)] / 2.
    """
    B_jk = seq.evaluate_at_quadrature(B, 2, True)
    J_jk = seq.evaluate_at_quadrature(J, 1, True)

    def m1_inv(rhs):
        return seq.apply_inverse_mass_matrix(rhs, 1, dirichlet=True, tol=tol)

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
        return (seq.cross_product_load_values(B_jk, dJ_jk, 2, 2, 1, True)
                + 0.5 * (seq.cross_product_load_values(Q_jk, J_jk, 2, 2, 1, True)
                         + seq.cross_product_load_values(B_jk, W_jk, 2, 2, 1, True)))

    return apply


def _preconditioner(seq, name):
    if callable(name):
        return name
    if name == "harmonic":
        return harmonic_preconditioner(seq)
    if name == "laplacian":
        return lambda x: seq.apply_laplacian_preconditioner(x, 1, dirichlet=True)
    if name == "laplacian2":
        return lambda x: seq.apply_laplacian_preconditioner(
            seq.apply_laplacian_preconditioner(x, 1, dirichlet=True), 1, dirichlet=True)
    if name == "mass":
        return lambda x: seq.apply_mass_matrix_preconditioner(x, 1, dirichlet=True)
    raise ValueError(f"newton_precond {name!r} is not one of {PRECONDITIONERS}")


def newton_direction(seq, B, J, MF, a_guess, shift=0.0, tol=1e-3, maxiter=100,
                     precond="laplacian", inner_tol=None):
    """The Newton direction ``u = curl a`` at the field ``B``.

    ``J`` the weak curl of ``B``, ``MF = M_2 F`` the mass times the
    Leray-projected force (``curl^T M_2 F`` is ``curl^T load(J x B)``
    exactly: the gradient part is a ``D_2^T``, and ``D_2 D_1 = 0``),
    ``a_guess`` the previous direction's potential (the warm start of the
    MINRES solve), ``shift`` the Levenberg-Marquardt shift, ``tol`` the
    relative residual of the solve in the preconditioner norm, ``maxiter``
    its iteration budget, ``precond`` one of :data:`PRECONDITIONERS` or the
    preconditioner's apply itself (a callable),
    ``inner_tol`` the tolerance of the Hessian's mass solves.

    Stops like every solve in the code (:func:`mrx.solvers.refine`): on the
    true residual of the system in the mass-atom norm of the dual 1-forms,
    evaluated on the residual view, ``tol`` relative to the right-hand side,
    the inner MINRES to the inner tolerance within ``maxiter`` iterations, ONE
    pass: the Newton solve is truncated by design (measured 2026-09-07 on
    li383 (16,32,32) from the step-5000 state: no preconditioner brings the
    true residual below 0.3 in 300 iterations, while the direction's energy
    differs 7x between them -- they differ in which modes they resolve first,
    not in how far they get), so a second pass would only double the cost.
    The outer loop is the criterion: ``tol`` means the same thing whatever
    the preconditioner (MINRES's own test is the residual in the
    preconditioner's norm, not comparable across preconditioners). Returns
    ``(u, a, info)`` with ``info`` the signed inner iteration count (negative
    when the true residual met ``tol``).
    """
    ops = seq._require_operators(None)
    on, inner = _outer(seq, tol)

    def chain(s):
        Hs = second_variation(s, B.astype(s.dtype), J.astype(s.dtype), tol=inner_tol)

        def curl(a):
            return s.apply_incidence_matrix(a, 1, dirichlet_in=True, dirichlet_out=True)

        def curl_t(y):
            return s.apply_incidence_matrix(y, 1, dirichlet_in=True, dirichlet_out=True,
                                            transpose=True)

        def A(a):
            u = curl(a)
            return curl_t(Hs(u) + shift * s.apply_mass_matrix(u, 2, True))
        return curl, curl_t, A

    curl, curl_t, A = chain(seq)
    A_res = chain(on)[2]
    P = _preconditioner(seq, precond)
    a, info = refine(A_res, lambda r: minres(A, r, M=P, tol=inner, maxiter=maxiter),
                     curl_t(MF), x0=a_guess, tol=tol, norm=_dual_norm(ops, 1, True),
                     max_passes=1, inner_dtype=seq.dtype)
    a = a.astype(seq.dtype)
    return curl(a), a, jnp.asarray(info, dtype=jnp.int32)
