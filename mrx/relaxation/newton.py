"""The Newton step of the relaxation: the Hessian of the magnetic energy and the Newton direction.

The relaxation moves the field ``B`` (a 2-form) with a divergence-free velocity ``u``, so that ``B_t = curl(u x
B_t)``. With ``Q = curl(u x B)`` and ``R = curl(u x Q)`` the energy is ``||B + Q + R/2||_M^2 / 2`` to second order
in ``u``. Its gradient is ``-load(J x B)`` and its symmetric Hessian is

    (u, H v) = (Q_u, Q_v)_M + [(B, curl(u x Q_v))_M + (B, curl(v x Q_u))_M] / 2.

At an equilibrium ``H`` is minus the ideal-MHD force operator at zero pressure.

* :func:`second_variation` returns the map ``u -> H u``, optionally with a penalty on flows along the field, or
  its Gauss-Newton part without the two terms in the current ``J``.
* :func:`newton_direction` computes one Gauss-Newton step. It writes ``u = curl a``, so that ``u`` is exactly
  divergence-free, and solves ``curl^T H curl a = curl^T M_2 F`` with the Gauss-Newton part of ``H`` for the
  potential ``a`` by Newton-MR, a MINRES solve that keeps and reorthogonalises its Lanczos vectors. The
  preconditioner is :func:`harmonic_preconditioner`. The two mass solves of every Hessian apply run to a
  tolerance inside the operator, which MINRES tolerates once they are accurate enough.

The module constants are the production defaults of the Newton solve. :class:`mrx.relaxation.config.Newton` uses
them as its defaults.
"""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from mrx.operators import _mass_solve
from mrx.solvers import minres

#: Production defaults: the weight of the parallel-flow penalty (in units of the field's strain), the relative
#: tolerance of the Newton solve in the norm of its preconditioner, the maximum number of MINRES iterations per
#: solve and the relative tolerance of the two 1-form mass solves inside every apply of the Gauss-Newton operator.
#: li383 graded (48,64,24), p = 2, from the file's field (2026-10-18): mass solves to 1e-6 and to 1e-8 give the same
#: trajectory, 1e-5 reaches the force residual 1e-8 four steps later and 1e-4 stalls at 2e-7. MINRES is not a
#: flexible method, and the error of the mass solves enters its operator. With tolerance 0.1 and 300 iterations the
#: solve takes the steps of the block Gauss-Newton system under flexible GCR on li383 M, L, p = 3 and W7-X, at the
#: same or 5-11 % less time. Without the reorthogonalisation it needs 1.6-2.6 times the iterations. Tolerance sweep
#: on li383 (48,64,24), (64,80,32) and W7-X (32,64,36), graded, 30 steps (2026-10-19): 0.1 reaches the force residual
#: 1e-8 first or level (17.5, 27.4, 19.7 s). 0.05 and 0.03 take fewer steps at 1.4-1.7 times the cost per step and end
#: 2-5 times lower after 30 steps, 0.2 needs 18-21 steps.
NEWTON_PENALTY = 3.0
NEWTON_TOL = 0.1
NEWTON_MAXITER = 300
NEWTON_MASS_SOLVE_TOL = 1e-6


def _ddx(f, x, axis, periodic):
    """Central difference of ``f`` along ``axis`` on the grid ``x``, periodic on ``[0, 1)`` or one-sided at the
    ends. A half-period grid covers only ``zeta`` in ``[0, 1/2]`` and is not periodic."""
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
    periodic = (False, True, not seq.half_period)
    grads = [_ddx(f_jk, x, ax, periodic[ax]) for ax, x in enumerate((seq.quad.x_x, seq.quad.x_y, seq.quad.x_z))]
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
    # the radial position of a DoF is the Greville point of its radial basis: the derivative splines for the
    # r component, the splines for the other two
    lam_r = seq.basis_0.Lambda[0]
    radial = (seq.basis_0.dLambda[0].s, lam_r, lam_r)
    for c, (s1, s2, s3) in enumerate(shapes):
        r = radial[c].greville_points()
        a, b = jnp.interp(r, r_q, prof_t), jnp.interp(r, r_q, prof_z)
        floor = jnp.interp(r, r_q, strain[:, c] + penalty)[:, None, None]
        m = np.fft.fftfreq(s2, d=1.0 / s2)
        nn = np.fft.fftfreq(s3, d=1.0 / s3)
        lam = (2 * np.pi) ** 2 * (a[:, None, None] * m[None, :, None] + b[:, None, None] * nn[None, None, :]) ** 2
        scale.append(1.0 / jnp.sqrt(lam + floor))
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


def second_variation(seq, B, J=None, penalty=None, mass_tol=None):
    """The map ``u -> H u`` that applies the energy Hessian at ``B`` to a velocity 2-form ``u``.

    ``J`` is the weak curl of ``B``. The result is a dual 2-form. Each application costs two 1-form mass solves,
    to the relative tolerance ``mass_tol`` (``seq.tol`` if ``None``). ``J=None`` leaves out the two terms in ``J``.
    What remains is the Gauss-Newton part ``(u, H v) = (Q_u, Q_v)_M``, which is positive semidefinite.
    Flows along the field, ``u = f B``, do not change ``B`` and form a null space of ``H``. ``penalty`` (the
    weight ``w(r)`` of :func:`parallel_penalty_profile`) removes it by adding
    ``(v, M_par u) = int w(r) (v . B)(u . B) / |B|^2 dx``, which leaves flows across the field unchanged.
    ``penalty=None`` gives the bare Hessian.
    """
    # B, J and the intermediate forms live in the odd parity view, u and H u in the even one
    odd, even = seq.odd, seq.even
    B_jk = odd.evaluate_at_quadrature(B, 2)
    if penalty is not None:
        Bsq_over_J2 = jnp.einsum('qi,qij,qj->q', B_jk, seq.metric_jkl, B_jk) / seq.jacobian_j ** 2
        weight = jnp.repeat(penalty, int(seq.quad.shape[1]) * int(seq.quad.shape[2]))

    curl = odd.G[1]
    if J is not None:
        J_jk = odd.evaluate_at_quadrature(J, 1)

    tol = seq.tol if mass_tol is None else mass_tol

    def m1_inv(x):
        # stopped on the residual of its recurrence, as every solve inside another iteration
        return _mass_solve(odd, x, 1, tol=tol, check_residual=False)

    def apply(u):
        u_jk = even.evaluate_at_quadrature(u, 2)
        E = m1_inv(odd.cross_product_load_values(u_jk, B_jk, 1, 2, 2))
        Q = curl @ E
        if J is None:
            Y = m1_inv(odd.D[1].T @ Q)
            Hu = even.cross_product_load_values(B_jk, odd.evaluate_at_quadrature(Y, 1), 2, 2, 1)
        else:
            Q_jk = odd.evaluate_at_quadrature(Q, 2)
            JxU = odd.cross_product_load_values(J_jk, u_jk, 2, 1, 2)
            # dJ + W / 2 with dJ = M_1^-1 D_1^T Q and W = M_1^-1 curl^T (J x u), in one solve
            Y = m1_inv(odd.D[1].T @ Q + 0.5 * (curl.T @ JxU))
            Y_jk = odd.evaluate_at_quadrature(Y, 1)
            Hu = (even.cross_product_load_values(B_jk, Y_jk, 2, 2, 1)
                  + 0.5 * even.cross_product_load_values(Q_jk, J_jk, 2, 2, 1))
        if penalty is None:
            return Hu
        s = jnp.einsum('qi,qij,qj->q', u_jk, seq.metric_jkl, B_jk) / seq.jacobian_j ** 2 / Bsq_over_J2
        return Hu + even.vector_load_values(B_jk * (weight * s)[:, None], 2, 2)

    return apply


def newton_direction(seq, B, load, a_guess, kappa=NEWTON_PENALTY, tol=NEWTON_TOL, maxiter=NEWTON_MAXITER,
                     mass_tol=NEWTON_MASS_SOLVE_TOL):
    """The Gauss-Newton direction at ``B``. Returns ``(u, a, info)`` with ``u = curl a``.

    ``load`` is the Lorentz force ``J x B`` as a dual 2-form. It need not be projected: the right-hand side
    ``curl^T load`` is blind to its gradient and harmonic parts. ``a_guess`` is the warm start for the potential
    ``a`` (pass the previous step's ``a``) and ``kappa`` the weight of the parallel-flow penalty. The operator is
    the Gauss-Newton part of the Hessian (:func:`second_variation` without ``J``) with its mass solves to
    ``mass_tol``. The solve (:func:`newton_mr`) stops when its residual, in the norm of the preconditioner, is
    below ``tol`` times the right-hand side or after ``maxiter`` MINRES iterations. ``info`` is the number of MINRES
    iterations, positive when ``tol`` was met and negative when not.
    """
    even = seq.even                           # a and u = curl a live in the even parity view
    # Both parity views are taken from a full sequence. From a view the other one is reached through its static
    # base, and its geometry would be compiled in as a constant that a later set_geometry_from_map does not reach.
    profiles = harmonic_atom_profiles(seq, B)
    penalty = parallel_penalty_profile(profiles, kappa)
    curl, curl_t, A = _newton_system(seq, B, penalty, mass_tol)
    atom = harmonic_preconditioner(seq, profiles, penalty)

    def P(x):
        return atom(even, x)

    def norm(r):
        return jnp.sqrt(r @ P(r))
    a, info = newton_mr(A, P, curl_t(load), a_guess, tol, maxiter, norm)
    return curl(a), a, jnp.asarray(info, dtype=jnp.int32)


def _newton_system(seq, B, penalty, mass_tol):
    """The maps ``curl``, ``curl^T`` and ``a -> curl^T H curl a`` of the Gauss-Newton system on the even view of the
    full sequence ``seq``."""
    Hs = second_variation(seq, B, None, penalty, mass_tol)
    even = seq.even

    def curl(a):
        return even.G[1] @ a

    def curl_t(y):
        return even.G[1].T @ y

    def A(a):
        return curl_t(Hs(curl(a)))
    return curl, curl_t, A


def newton_mr(A, P, b, x0, tol, maxiter, norm):
    """Solve the symmetric ``A x = b`` by Newton-MR (Liu & Roosta 2022). Returns ``(x, info)``.

    If the warm start ``x0`` already meets ``norm(b - A x0) <= tol norm(b)`` it is returned unchanged.
    Otherwise one preconditioned MINRES solve computes the correction to ``x0``. It stops when its own residual
    estimate (in the preconditioner's norm) meets the tolerance, or after ``maxiter`` iterations. MINRES
    stops early at a direction of nonpositive curvature. Such a direction is a descent
    direction only for the right-hand side it was found with, so the solve is then repeated for ``b`` itself
    from zero. MINRES keeps its Lanczos vectors and reorthogonalises each new one (``reorthogonalize`` of
    :func:`mrx.solvers.minres`), which stores ``2 maxiter`` vectors. ``info`` is the number of MINRES iterations,
    positive when the tolerance was met and negative when not.
    """
    x = x0
    bnorm = norm(b)
    r0 = b - A(x)

    def solve(_):
        rnorm = norm(r0)
        d, info, npc = minres(A, r0 / rnorm, M=P, tol=tol * bnorm / rnorm, maxiter=maxiter,
                              npc_exit=True, norm=norm, reorthogonalize=True)

        def from_zero(_):
            d0, info0, _ = minres(A, b / bnorm, M=P, tol=tol, maxiter=maxiter, npc_exit=True,
                                  norm=norm, reorthogonalize=True)
            return d0 * bnorm, jnp.abs(info0).astype(jnp.int32)

        x_new, its_npc = jax.lax.cond(npc, from_zero,
                                      lambda _: (x + d * rnorm, jnp.zeros((), jnp.int32)),
                                      None)
        return x_new, b - A(x_new), (jnp.abs(info) + its_npc).astype(jnp.int32)

    def skip(_):
        return x, r0, jnp.zeros((), jnp.int32)

    x, r, its = jax.lax.cond(norm(r0) > tol * bnorm, solve, skip, None)
    return x, jnp.where(norm(r) <= tol * bnorm, its, -its)
