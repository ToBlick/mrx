"""The Newton step of the relaxation: the Hessian of the magnetic energy and the Newton direction.

The relaxation moves the field ``B`` (a 2-form) with a divergence-free velocity ``u``, so that ``B_t = curl(u x
B_t)``. With ``Q = curl(u x B)`` and ``R = curl(u x Q)`` the energy is ``||B + Q + R/2||_M^2 / 2`` to second order
in ``u``. Its gradient is ``-load(J x B)`` and its symmetric Hessian is

    (u, H v) = (Q_u, Q_v)_M + [(B, curl(u x Q_v))_M + (B, curl(v x Q_u))_M] / 2.

At an equilibrium ``H`` is minus the ideal-MHD force operator at zero pressure.

* :func:`second_variation` returns the map ``u -> H u``, optionally with a penalty on flows along the field.
* :func:`newton_direction` computes one Newton step. It writes ``u = curl a``, so that ``u`` is exactly
  divergence-free, and solves ``curl^T H curl a = curl^T M_2 F`` for the potential ``a`` by Newton-MR, a MINRES
  solve that can stop at a direction of nonpositive curvature. The preconditioner is
  :func:`harmonic_preconditioner`.

The module constants are the production defaults of the Newton solve. :class:`mrx.relaxation.config.Newton` uses
them as its defaults.
"""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from mrx.operators import dual_norm, parity_projectors
from mrx.precision import RESIDUAL_DTYPE
from mrx.solvers import minres

#: Production defaults: the weight of the parallel-flow penalty (in units of the field's strain), the relative
#: tolerance of the Newton solve and the maximum number of MINRES iterations per solve.
NEWTON_PENALTY = 3.0
NEWTON_TOL = 0.1
NEWTON_MAXITER = 200


def _ddx(f, x, axis, periodic):
    """Central difference of ``f`` along ``axis`` on the grid ``x``, periodic on ``[0, 1)`` or one-sided at the
    ends."""
    f = jnp.moveaxis(f, axis, 0)
    if periodic:
        df = jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)
        dx = (jnp.roll(x, -1) - jnp.roll(x, 1)) % 1.0
    else:
        df = jnp.concatenate([f[1:2] - f[0:1], f[2:] - f[:-2], f[-1:] - f[-2:-1]])
        dx = jnp.concatenate([x[1:2] - x[0:1], x[2:] - x[:-2], x[-1:] - x[-2:-1]])
    return jnp.moveaxis(df / dx.reshape((-1,) + (1,) * (f.ndim - 1)), 0, axis)


def harmonic_atom_profiles(seq, field):
    """Radial profiles of a 2-form ``field``, on the radial quadrature points, used to build the preconditioner.

    Returns ``(prof_t, prof_z, strain)``. ``prof_t`` and ``prof_z`` are the averages over ``theta`` and ``zeta``
    of the field's logical ``theta`` and ``zeta`` components. ``strain`` has shape ``(nq_r, 3)`` and holds, for
    each logical direction ``c``, the angle average of ``sum_i (d_c field^i)^2``, i.e. how fast the field
    varies along that direction.
    """
    shape = tuple(int(v) for v in seq.quad.shape)
    f_jk = (seq.odd.evaluate_at_quadrature(field, 2) / seq.jacobian_j[:, None]).reshape(shape + (3,))
    prof_t = f_jk[..., 1].mean(axis=(1, 2))
    prof_z = f_jk[..., 2].mean(axis=(1, 2))
    grads = [_ddx(f_jk, x, ax, ax > 0) for ax, x in enumerate((seq.quad.x_x, seq.quad.x_y, seq.quad.x_z))]
    strain = jnp.stack([(g ** 2).sum(axis=-1).mean(axis=(1, 2)) for g in grads], axis=-1)
    return prof_t, prof_z, strain


def parallel_penalty_profile(profiles, kappa):
    """The radial weight ``w(r)`` of the parallel-flow penalty: ``kappa`` times the field's strain along the field.

    With ``(h_t, h_z, s)`` the output of :func:`harmonic_atom_profiles`, the strain along the field is
    ``(h_t^2 s_t + h_z^2 s_z) / (h_t^2 + h_z^2)``.
    """
    prof_t, prof_z, strain = profiles
    return kappa * (prof_t ** 2 * strain[:, 1] + prof_z ** 2 * strain[:, 2]) / (prof_t ** 2 + prof_z ** 2)


def harmonic_preconditioner(seq, profiles, penalty):
    """The preconditioner of the Newton system, a :class:`HarmonicAtom`: an SPD approximate inverse of
    ``curl^T H curl``.

    It is the preconditioner of the 1-form Laplacian, scaled on both sides by ``lambda^{-1/2}``. For each Fourier
    mode ``(m, n)`` in ``(theta, zeta)`` and each radius, ``lambda = (2 pi)^2 (h_t m + h_z n)^2 + s_c + w``
    approximates the Hessian: the first term is the squared derivative along the field (``h_t, h_z`` from
    ``profiles``, see :func:`harmonic_atom_profiles`), ``s_c`` the strain along the component's direction and
    ``w`` the parallel-flow ``penalty``. Build it once per Newton step, since it depends on ``B``.
    """
    prof_t, prof_z, strain = profiles
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
    """The Newton preconditioner of :func:`harmonic_preconditioner`, applied to a dual 1-form ``x`` as
    ``atom(seq, x)``.

    It is a pytree that holds only the per-mode scales, so a new field does not trigger a recompile. The
    sequence is passed at each call.
    """

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
        E = seq.E(1)
        y = E @ self._C(E.T @ x)
        y = seq.L[1].precondition(y)
        return E @ self._C(E.T @ y)


def second_variation(seq, B, J, penalty=None):
    """The map ``u -> H u`` that applies the energy Hessian at ``B`` to a velocity 2-form ``u``.

    ``J`` is the weak curl of ``B``. The result is a dual 2-form. Each application costs three 1-form mass solves.
    Flows along the field, ``u = f B``, do not change ``B`` and form a null space of ``H``. ``penalty`` (the
    weight ``w(r)`` of :func:`parallel_penalty_profile`) removes it by adding
    ``(v, M_par u) = int w(r) (v . B)(u . B) / |B|^2 dx``, which leaves flows across the field unchanged.
    ``penalty=None`` gives the bare Hessian.
    """
    # B, J and the intermediate forms live in the odd parity view, u and H u in the even one
    odd, even = seq.odd, seq.even
    B_jk = odd.evaluate_at_quadrature(B, 2)
    J_jk = odd.evaluate_at_quadrature(J, 1)
    if penalty is not None:
        Bsq_over_J2 = jnp.einsum('qi,qij,qj->q', B_jk, seq.metric_jkl, B_jk) / seq.jacobian_j ** 2
        weight = jnp.repeat(penalty, int(seq.quad.shape[1]) * int(seq.quad.shape[2]))

    m1_inv, curl = odd.M[1].solve, odd.G[1]

    def apply(u):
        u_jk = even.evaluate_at_quadrature(u, 2)
        E = m1_inv(odd.cross_product_load_values(u_jk, B_jk, 1, 2, 2))
        Q = curl @ E
        Q_jk = odd.evaluate_at_quadrature(Q, 2)
        dJ = m1_inv(odd.D[1].T @ Q)
        dJ_jk = odd.evaluate_at_quadrature(dJ, 1)
        JxU = odd.cross_product_load_values(J_jk, u_jk, 2, 1, 2)
        W = m1_inv(curl.T @ JxU)
        W_jk = odd.evaluate_at_quadrature(W, 1)
        Hu = (even.cross_product_load_values(B_jk, dJ_jk, 2, 2, 1)
              + 0.5 * (even.cross_product_load_values(Q_jk, J_jk, 2, 2, 1)
                       + even.cross_product_load_values(B_jk, W_jk, 2, 2, 1)))
        if penalty is None:
            return Hu
        s = jnp.einsum('qi,qij,qj->q', u_jk, seq.metric_jkl, B_jk) / seq.jacobian_j ** 2 / Bsq_over_J2
        return Hu + even.vector_load_values(B_jk * (weight * s)[:, None], 2, 2)

    return apply


def newton_direction(seq, B, J, load, a_guess, kappa=NEWTON_PENALTY, tol=NEWTON_TOL, maxiter=NEWTON_MAXITER):
    """The Newton direction at ``B``. Returns ``(u, a, info)`` with ``u = curl a``.

    ``J`` is the weak curl of ``B`` and ``load`` the Lorentz force ``J x B`` as a dual 2-form. It need not be
    projected: the right-hand side ``curl^T load`` is blind to its gradient and harmonic parts. ``a_guess`` is the warm start for the potential ``a`` (pass the previous step's ``a``) and
    ``kappa`` the weight of the parallel-flow penalty. The solve (:func:`newton_mr`) stops when its residual is
    below ``tol`` times the right-hand side or after ``maxiter`` MINRES iterations. The residual is measured in
    the residual precision, so in mixed precision it is float64. ``info`` is the number of MINRES iterations,
    positive when ``tol`` was met and negative when not. ``u`` and ``a`` are in the sequence's dtype.
    """
    even = seq.even                           # a and u = curl a live in the even parity view
    on = even if even.residual is None else even.residual
    profiles = harmonic_atom_profiles(seq, B)
    penalty = parallel_penalty_profile(profiles, kappa)
    curl, curl_t, A = _newton_system(even, B, J, penalty)
    # the residual operator computes its own penalty from B in the residual precision
    penalty_res = penalty if on is even else parallel_penalty_profile(
        harmonic_atom_profiles(on, B.astype(on.dtype)), kappa)
    A_res = _newton_system(on, B, J, penalty_res)[2]
    atom = harmonic_preconditioner(seq, profiles, penalty)
    rhs = curl_t(load)
    # on a half-period sequence the residual is projected onto the parity of the right-hand side
    parity = parity_projectors(even, 1, rhs)
    project_dual = None if parity is None else parity[1]
    a, info, _ = newton_mr(A_res, A, lambda x: atom(even, x), rhs, a_guess, tol, maxiter,
                           dual_norm(even, 1), inner_dtype=seq.dtype, project_dual=project_dual)
    a = a.astype(seq.dtype)
    return curl(a), a, jnp.asarray(info, dtype=jnp.int32)


def _newton_system(seq, B, J, penalty):
    """The maps ``curl``, ``curl^T`` and ``a -> curl^T H curl a`` of the Newton system on the even view ``seq``."""
    Hs = second_variation(seq, B.astype(seq.dtype), J.astype(seq.dtype), penalty)

    def curl(a):
        return seq.G[1] @ a

    def curl_t(y):
        return seq.G[1].T @ y

    def A(a):
        return curl_t(Hs(curl(a)))
    return curl, curl_t, A


def newton_mr(A_res, A, P, b, x0, tol, maxiter, norm, inner_dtype, project_dual=None):
    """Solve the symmetric ``A x = b`` by Newton-MR (Liu & Roosta 2022). Returns ``(x, info, npc)``.

    If the warm start ``x0`` already meets ``norm(b - A_res x0) <= tol norm(b)`` it is returned unchanged.
    Otherwise one preconditioned MINRES solve of at most ``maxiter`` iterations computes the correction to
    ``x0``. ``A_res`` is the operator in the residual precision and ``A`` the one in ``inner_dtype``. MINRES
    stops early at a direction of nonpositive curvature (``npc`` is then true). Such a direction is a descent
    direction only for the right-hand side it was found with, so the solve is then repeated for ``b`` itself
    from zero. ``project_dual`` removes round-off of the wrong parity from the residual. ``info`` is the number
    of MINRES iterations, positive when the tolerance was met and negative when not.
    """
    if project_dual is None:
        def project_dual(r): return r
    b = b.astype(RESIDUAL_DTYPE)
    x = jnp.zeros_like(b) if x0 is None else x0.astype(RESIDUAL_DTYPE)
    bnorm = norm(b)
    r0 = project_dual(b - A_res(x))

    def solve(_):
        rnorm = norm(r0)
        d, info, npc = minres(A, (r0 / rnorm).astype(inner_dtype), M=P, tol=0.0, maxiter=maxiter, npc_exit=True)

        def from_zero(_):
            d0, info0, _ = minres(A, (b / bnorm).astype(inner_dtype), M=P, tol=0.0, maxiter=maxiter, npc_exit=True)
            return d0.astype(RESIDUAL_DTYPE) * bnorm, jnp.abs(info0).astype(jnp.int32)

        x_new, its_npc = jax.lax.cond(npc, from_zero,
                                      lambda _: (x + d.astype(RESIDUAL_DTYPE) * rnorm, jnp.zeros((), jnp.int32)),
                                      None)
        return x_new, project_dual(b - A_res(x_new)), (jnp.abs(info) + its_npc).astype(jnp.int32), npc

    def skip(_):
        return x, r0, jnp.zeros((), jnp.int32), jnp.zeros((), bool)

    x, r, its, npc = jax.lax.cond(norm(r0) > tol * bnorm, solve, skip, None)
    return x, jnp.where(norm(r) <= tol * bnorm, its, -its), npc
