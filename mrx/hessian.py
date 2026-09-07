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

from mrx.solvers import minres

#: The preconditioners of the Newton solve: the k=1 Laplacian atom, its
#: square (the operator is fourth order in ``a``), or the k=1 mass atom.
PRECONDITIONERS = ("laplacian", "laplacian2", "mass")


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
    its iteration budget, ``precond`` one of :data:`PRECONDITIONERS`,
    ``inner_tol`` the tolerance of the Hessian's mass solves.

    Returns ``(u, a, info)`` with ``info`` the signed iteration count of
    :func:`mrx.solvers.minres` (negative when converged).
    """
    H = second_variation(seq, B, J, tol=inner_tol)

    def curl(a):
        return seq.apply_incidence_matrix(a, 1, dirichlet_in=True, dirichlet_out=True)

    def curl_t(y):
        return seq.apply_incidence_matrix(y, 1, dirichlet_in=True, dirichlet_out=True,
                                          transpose=True)

    def A(a):
        u = curl(a)
        return curl_t(H(u) + shift * seq.apply_mass_matrix(u, 2, True))

    a, info = minres(A, curl_t(MF), x0=a_guess, M=_preconditioner(seq, precond),
                     tol=tol, maxiter=maxiter)
    return curl(a), a, jnp.asarray(info, dtype=jnp.int32)
