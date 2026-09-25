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
harmonic atom as the preconditioner (:func:`harmonic_preconditioner`) and the
parallel-flow penalty in the operator (:func:`second_variation`): Newton-MR.
The one divergence-free direction
``curl a`` cannot represent is the
harmonic 2-form of the Dirichlet complex (the net toroidal flux, one DoF).
"""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from mrx.operators import _dual_norm, _parity
from mrx.precision import RESIDUAL_DTYPE
from mrx.solvers import minres

#: The Newton configuration (:func:`newton_direction`, the defaults of
#: :class:`mrx.relaxation.TimeStepper` and of ``scripts/relax.py``): the
#: parallel-flow penalty in units of the strain, the forcing term and the
#: MINRES iterations (measured 2026-09-18).
NEWTON_PENALTY = 3.0
NEWTON_TOL = 0.1
NEWTON_MAXITER = 200

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
    f_jk = (seq.odd.evaluate_at_quadrature(field, 2, True) / seq.jacobian_j[:, None]).reshape(shape + (3,))
    prof_t = f_jk[..., 1].mean(axis=(1, 2))
    prof_z = f_jk[..., 2].mean(axis=(1, 2))
    grads = [_ddx(f_jk, x, ax, ax > 0) for ax, x in enumerate((seq.quad.x_x, seq.quad.x_y, seq.quad.x_z))]
    strain = jnp.stack([(g ** 2).sum(axis=-1).mean(axis=(1, 2)) for g in grads], axis=-1)
    return prof_t, prof_z, strain


def parallel_penalty_profile(seq, field, kappa):
    """The radial weight of the parallel-flow penalty (:func:`second_variation`) on the
    radial quadrature points: ``kappa`` times the strain seen along the field,
    ``(h_theta^2 s_theta + h_zeta^2 s_zeta) / (h_theta^2 + h_zeta^2)`` with ``s``
    the lumped strain of :func:`harmonic_atom_profiles`, the size of the
    ``u . grad B`` coupling a field-aligned velocity really has. ``kappa`` is
    the one number of the Newton configuration: it measures how much the
    angle-averaged, direction-diagonal strain undercounts that coupling, a
    property of the lumping, not of the device (3 on li383 and W7-X within
    6 % of the best constant, 2026-09-18; 1 is 2x worse, 0.03 lets the null
    space through)."""
    prof_t, prof_z, strain = harmonic_atom_profiles(seq, field)
    return kappa * (prof_t ** 2 * strain[:, 1] + prof_z ** 2 * strain[:, 2]) / (prof_t ** 2 + prof_z ** 2)


def harmonic_preconditioner(seq, B, kappa):
    """``(seq, x) -> W P_L W^T x`` (a :class:`HarmonicAtom`): the harmonic atom, an approximate
    inverse of the Newton operator ``curl^T H curl`` built from the profiles of the current field ``B``.

    The Hessian is, to a percent, the Gauss-Newton form ``||curl(u x B)||^2``,
    and with ``div u = 0``, ``curl(u x B) = B . grad u - u . grad B``; the atom
    keeps the parallel derivative and lumps it: per component of the 1-form
    potential and per radial DoF layer, the symbol

        lambda(r, m, n) = (2 pi)^2 (h_theta(r) m + h_zeta(r) n)^2 + strain_c(r) + penalty(r),

    ``h_theta, h_zeta`` the angle-averaged logical contravariant components of
    ``B`` and ``strain_c`` the lumped strain of the field along the component's
    direction (:func:`harmonic_atom_profiles`; the lumped parallel derivative
    ``i k I`` is anti-Hermitian and the strain symmetric, so the normal form of
    ``i k I - S`` is ``k^2 I + S^T S`` with no cross term: the floor is
    computed, not tuned), ``(m, n)`` the Fourier frequencies of the DoF grid in
    the two angles, and ``penalty`` the parallel-flow penalty the operator
    carries (:func:`parallel_penalty_profile`), so that the atom and the
    operator agree on what a field-aligned mode sees. The field carries the
    rotational transform, so the symbol vanishes on the resonant modes
    ``h_theta m + h_zeta n = 0`` and the floor is what they see. Rebuilt from
    ``B`` at every step: one quadrature evaluation, traceable.

    Inverted as a sandwich of the Laplacian atom ``P_L`` (which approximates
    the inverse of the curl-curl the potential form is quadratic in) with the
    symbol's inverse square root: ``W = E C E^T`` with ``C`` the 2-D Fourier
    scaling by ``lambda^{-1/2}`` on the tensor DoF grid of each component and
    ``E`` the Dirichlet 1-form extraction, so that in the bulk ``W P_L W^T``
    is the Laplacian atom with the symbol multiplied into its denominator
    and on the polar rows the extraction lumps it. Symmetric positive
    definite for any ``C``, which is all MINRES needs. Two FFTs per component
    per apply.
    """
    prof_t, prof_z, strain = harmonic_atom_profiles(seq, B)
    penalty = parallel_penalty_profile(seq, B, kappa)
    r_q = seq.quad.x_x
    shapes = [tuple(int(v) for v in s) for s in seq.basis_1.shape]
    scale = []
    for c, (s1, s2, s3) in enumerate(shapes):
        r = (jnp.arange(s1) + 0.5) / s1
        a, b = jnp.interp(r, r_q, prof_t), jnp.interp(r, r_q, prof_z)
        floor = jnp.interp(r, r_q, strain[:, c] + penalty)[:, None, None]
        m = np.fft.fftfreq(s2, d=1.0 / s2)
        nn = np.fft.fftfreq(s3, d=1.0 / s3)
        lam = (2 * np.pi) ** 2 * (a[:, None, None] * m[None, :, None] + b[:, None, None] * nn[None, None, :]) ** 2
        scale.append((1.0 / jnp.sqrt(lam + floor)).astype(seq.dtype))
    return HarmonicAtom(scale=tuple(scale), shapes=tuple(shapes))


class HarmonicAtom(eqx.Module):
    """The harmonic atom of :func:`harmonic_preconditioner` as a pytree: the
    per-component Fourier scalings are its arrays, applied through the
    sequence it is CALLED with (``atom(seq, x)``), so that inside a jitted
    function of the sequence nothing is a captured constant (:mod:`mrx.pytree`)."""

    scale: tuple
    shapes: tuple = eqx.field(static=True)

    def _C(self, x):
        out, off = [], 0
        for sc, s in zip(self.scale, self.shapes):
            n_c = s[0] * s[1] * s[2]
            X = x[off:off + n_c].reshape(s)
            out.append(jnp.fft.ifft2(jnp.fft.fft2(X, axes=(1, 2)) * sc, axes=(1, 2)).real.ravel())
            off += n_c
        return jnp.concatenate(out)

    def __call__(self, seq, x):
        E = seq.E(1, True)
        y = E @ self._C(E.T @ x)
        y = seq.apply_laplacian_preconditioner(y, 1, dirichlet=True)
        return E @ self._C(E.T @ y)


def second_variation(seq, B, J, kappa=0.0):
    """``u -> H u + kappa M_par u``: the Hessian of the energy along the flow of ``u`` with the
    parallel-flow penalty, as a dual 2-form.

    ``B`` the 2-form, ``J`` its weak curl (a Dirichlet 1-form, the ``J`` of
    :func:`mrx.relaxation.compute_force`). Three k=1 mass solves per apply,
    each at the sequence's tolerance: ``E = M_1^-1 load(u x B)``
    for ``Q = curl E``, the weak curl ``dJ`` of ``Q``, and ``W = M_1^-1
    curl^T load(J x u)``; then

        H u = load(B x dJ) + [load(Q x J) + load(B x W)] / 2.

    The penalty ``<v, M_par u> = int w(r) (v . B)(u . B) / |B|^2 J`` with the
    weight ``w`` of :func:`parallel_penalty_profile`: the Hessian is exactly
    null on the field-aligned flows ``u = f B`` (``curl(f B x B) = 0``;
    divergence-free wherever ``B . grad f = 0``, so on every flux surface and
    with a resonant ``f`` on every rational surface), and on the mesh that
    null space is a continuum of eigenvalues 1e-4..1e-1 (li383 (16,32,32),
    docs/research/hessian_spectrum_2026-09-17.md) that Newton divides the
    force's round-off components by. The penalty is Levenberg-Marquardt
    damping on the parallel component alone: it lifts those modes, leaves the
    energy descent unchanged (``<F, f B> = 0``) and the perpendicular step
    untouched. One quadrature load per apply; ``kappa = 0`` is the bare Hessian.
    On a half-period sequence every load carries its parity (``u x B`` and
    ``J x u`` odd, the Hessian's image even; :mod:`mrx.symmetry`).
    """
    odd, even = seq.odd, seq.even            # B, J, E, Q, dJ, W odd; u and the Hessian's image even
    B_jk = odd.evaluate_at_quadrature(B, 2, True)
    J_jk = odd.evaluate_at_quadrature(J, 1, True)
    Bsq_over_J2 = jnp.einsum('qi,qij,qj->q', B_jk, seq.metric_jkl, B_jk) / seq.jacobian_j ** 2
    n_angles = int(seq.quad.shape[1]) * int(seq.quad.shape[2])
    weight = jnp.repeat(parallel_penalty_profile(seq, B, kappa), n_angles)

    def m1_inv(rhs):
        return odd.apply_inverse_mass_matrix(rhs, 1, dirichlet=True)

    def apply(u):
        u_jk = even.evaluate_at_quadrature(u, 2, True)
        E = m1_inv(odd.cross_product_load_values(u_jk, B_jk, 1, 2, 2, True))
        Q = odd.apply_incidence_matrix(E, 1, dirichlet_in=True, dirichlet_out=True)
        Q_jk = odd.evaluate_at_quadrature(Q, 2, True)
        dJ = m1_inv(odd.apply_derivative_matrix(Q, 1, dirichlet_in=True, dirichlet_out=True,
                                                transpose=True))
        dJ_jk = odd.evaluate_at_quadrature(dJ, 1, True)
        JxU = odd.cross_product_load_values(J_jk, u_jk, 2, 1, 2, True)
        W = m1_inv(odd.apply_incidence_matrix(JxU, 1, dirichlet_in=True, dirichlet_out=True,
                                              transpose=True))
        W_jk = odd.evaluate_at_quadrature(W, 1, True)
        s = jnp.einsum('qi,qij,qj->q', u_jk, seq.metric_jkl, B_jk) / seq.jacobian_j ** 2 / Bsq_over_J2
        return (even.cross_product_load_values(B_jk, dJ_jk, 2, 2, 1, True)
                + 0.5 * (even.cross_product_load_values(Q_jk, J_jk, 2, 2, 1, True)
                         + even.cross_product_load_values(B_jk, W_jk, 2, 2, 1, True))
                + even.vector_load_values(B_jk * (weight * s)[:, None], 2, 2, True))

    return apply


def newton_direction(seq, B, J, MF, a_guess, kappa=NEWTON_PENALTY, tol=NEWTON_TOL, maxiter=NEWTON_MAXITER):
    """The Newton direction ``u = curl a`` at the field ``B``: Newton-MR.

    ``J`` the weak curl of ``B``, ``MF = M_2 F`` the mass times the
    Leray-projected force (``curl^T M_2 F`` is ``curl^T load(J x B)``
    exactly: the gradient part is a ``D_2^T``, and ``D_2 D_1 = 0``),
    ``a_guess`` the previous direction's potential (the warm start),
    ``kappa`` the parallel-flow penalty of :func:`second_variation` and of
    the atom, ``tol`` the forcing term, ``maxiter`` the MINRES iterations at
    most; the Hessian's mass solves run at the sequence's tolerance.

    The system ``curl^T H curl a = curl^T M_2 F`` is solved by :func:`newton_mr`
    with the harmonic atom of the current field as the preconditioner
    (:func:`harmonic_preconditioner`, rebuilt every step): MINRES from the
    warm start for ``maxiter`` iterations at most, until the residual,
    measured in the residual precision and the mass-atom norm of the dual
    1-forms like every solve in the code, is below ``tol`` of the right-hand
    side (the forcing term of Dembo, Eisenstat & Steihaug). The default of
    ``200`` iterations gives the same relaxation as any tighter solve
    (measured 2026-09-18; with the strain penalty the 0.1 forcing term is met
    only after ~275 iterations on li383, so more iterations cost without gain),
    with the nonpositive-curvature exit of Newton-MR (Liu & Roosta 2022):
    MINRES's iterate is a descent direction as long as no direction of
    nonpositive curvature has appeared in its Krylov space, and when one
    appears the preconditioned residual of a solve from zero is one and is
    taken instead. The second variation is indefinite away from equilibrium
    (W7-X, 2026-09-18), which is why the inner solver is MINRES and not
    conjugate gradients: a CG solve stops at the first direction of negative
    curvature and makes no further progress there. Returns ``(u, a, info)``
    with ``info`` the iteration count, negative when the residual met ``tol``.
    """
    even = seq.even                           # the potential a and the direction u = curl a are even
    ops = even._require_operators(None)
    on = even if even.residual is None else even.residual
    curl, curl_t, A = _newton_system(even, B, J, kappa)
    A_res = _newton_system(on, B, J, kappa)[2]
    atom = harmonic_preconditioner(seq, B, kappa)
    rhs = curl_t(MF)
    # an unreduced half-period sequence: the residual loses the round-off of the other parity
    parity = _parity(even, 1, True, rhs)
    project_dual = None if parity is None else parity[1]
    a, info, _ = newton_mr(A_res, A, lambda x: atom(even, x), rhs, a_guess, tol, maxiter,
                           _dual_norm(ops, 1, True), inner_dtype=seq.dtype, project_dual=project_dual)
    a = a.astype(seq.dtype)
    return curl(a), a, jnp.asarray(info, dtype=jnp.int32)


def _newton_system(seq, B, J, kappa):
    """``(curl, curl_t, A)`` of the Newton system ``curl^T H curl a = curl^T M_2 F`` on the
    even view ``seq`` (the potential's space)."""
    Hs = second_variation(seq, B.astype(seq.dtype), J.astype(seq.dtype), kappa)

    def curl(a):
        return seq.apply_incidence_matrix(a, 1, dirichlet_in=True, dirichlet_out=True)

    def curl_t(y):
        return seq.apply_incidence_matrix(y, 1, dirichlet_in=True, dirichlet_out=True, transpose=True)

    def A(a):
        return curl_t(Hs(curl(a)))
    return curl, curl_t, A


def newton_mr(A_res, A, P, b, x0, tol, maxiter, norm, inner_dtype, project_dual=None):
    """The Newton-MR solve of ``A x = b`` (Liu & Roosta 2022): one MINRES solve of ``maxiter``
    iterations at most on the float64 residual from the warm start ``x0``, skipped when
    ``norm(b - A x0) <= tol norm(b)`` already (the forcing term, in the residual's norm and
    precision), with the nonpositive-curvature exit. The solve is for the correction from
    zero on the residual at unit norm, as :func:`mrx.solvers.refine` does; Liu & Roosta's
    descent guarantee for an NPC direction holds against the right-hand side that solve
    started from, which is the residual, not ``b`` (measured: with the warm start,
    the NPC direction of the correction solve failed the energy's sign test on every W7-X
    step). So a solve that meets NPC discards its answer and solves ``A x = b`` from zero
    with the exit once more: what that returns -- the NPC direction, or the iterate when
    the curvature was an artefact of the warm start's residual -- is descent-guaranteed
    for ``b`` and is the answer, alone. ``project_dual`` removes the other parity's
    round-off from the residual on a half-period sequence (:mod:`mrx.symmetry`). Returns
    ``(x, info, npc)`` with ``info`` the inner iterations of all passes, negative when the
    residual test was met."""
    if project_dual is None:
        def project_dual(r): return r
    b = b.astype(RESIDUAL_DTYPE)
    x = jnp.zeros_like(b) if x0 is None else x0.astype(RESIDUAL_DTYPE)
    bnorm = norm(b)
    bnorm_safe = jnp.where(bnorm > 0, bnorm, 1.0)

    def cond(carry):
        _, r, k, _, npc = carry
        return jnp.logical_and(jnp.logical_and(norm(r) > tol * bnorm_safe, k < 1), ~npc)

    def body(carry):
        x, r, k, its, _ = carry
        rnorm = norm(r)
        rnorm_safe = jnp.where(rnorm > 0, rnorm, 1.0)
        d, info, npc = minres(A, (r / rnorm_safe).astype(inner_dtype), M=P, tol=0.0,
                              maxiter=maxiter, npc_exit=True)
        d = d.astype(RESIDUAL_DTYPE) * rnorm_safe

        def from_zero(_):
            d0, info0, _ = minres(A, (b / bnorm_safe).astype(inner_dtype), M=P, tol=0.0,
                                  maxiter=maxiter, npc_exit=True)
            return d0.astype(RESIDUAL_DTYPE) * bnorm_safe, jnp.abs(info0).astype(jnp.int32)

        x_new, its_npc = jax.lax.cond(npc, from_zero, lambda _: (x + d, jnp.zeros((), jnp.int32)), None)
        r_new = project_dual(b - A_res(x_new))
        return x_new, r_new, k + 1, (its + jnp.abs(info) + its_npc).astype(jnp.int32), npc

    r0 = project_dual(b - A_res(x))
    x, r, k, its, npc = jax.lax.while_loop(cond, body, (x, r0, 0, jnp.int32(0), False))
    converged = norm(r) <= tol * bnorm_safe
    return x, jnp.where(converged, -its, its), npc
