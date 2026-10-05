"""The compressible relaxation with an advected pressure descends ``L = int B^2/2 - p dV`` along its own flow.

For a fixed velocity ``u`` (an even 2-form, not divergence-free) the flow moves the field by ``dB/dt = A_u B`` as in
the incompressible relaxation and the pressure, a free 0-form, by ``dp/dt = T_u p = -M_0^{-1} load(u . grad p)``.
Both are linear, so along the flow

    L'(0) = (B, A_u B)_M - int T_u p,        L''(0) = ||A_u B||^2_M + (B, A_u^2 B)_M - int T_u^2 p.

``L'(0)`` must equal ``-(u, J x B - grad p)`` and ``L''(0)`` must equal ``(u, H u)`` with the pressure term of
:func:`mrx.relaxation.newton.second_variation`, and so must the polarised bilinear form. A short Newton run with the
compressible stepper lowers ``L`` at every step as the line search predicts, and lowers the force.
"""
import jax
import numpy as np

from mrx.precision import DTYPE, eps
from mrx.relaxation.loop import TimeStepper, initial_state, relax
from mrx.relaxation.newton import second_variation
from mrx.relaxation.physics import advection, compute_force, pressure_gradient, pressure_integral

STEPS, CHUNK = 6, 3
BETA = 0.05


def _pressure(seq, b0):
    """``p = p0 (1 - r^2)^2`` at the volume beta ``BETA`` of ``b0``, a free 0-form on the even view."""
    p = seq.even.free.interpolate(lambda x: (1.0 - x[0] ** 2) ** 2, 0, frame='logical')
    return p * (BETA * 0.5 * seq.odd.l2_norm_sq(b0, 2) / pressure_integral(p, seq))


def _velocity(seq, key):
    """A random even 2-form of unit norm, with divergence."""
    even = seq.even
    u = jax.random.normal(key, (even.n(2),), dtype=DTYPE)
    return u / even.l2_norm(u, 2)


def _derivatives(seq, B, p, u):
    """``(L'(0), L''(0))`` along the flow of ``u``."""
    odd, even = seq.odd, seq.even
    u_q = even.evaluate_at_quadrature(u, 2)

    def A(X):
        return odd.G[1] @ odd.M[1].solve(odd.cross_product_load_values(u_q, odd.evaluate_at_quadrature(X, 2),
                                                                     1, 2, 2))

    def T(q):
        return advection(pressure_gradient(q, seq), u_q, seq)
    AB, Tp = A(B), T(p)
    d1 = float(B @ (odd.M[2] @ AB)) - float(pressure_integral(Tp, seq))
    d2 = (float(odd.l2_norm_sq(AB, 2) + B @ (odd.M[2] @ A(AB)))
          - float(pressure_integral(T(Tp), seq)))
    return d1, d2


def test_pressure_hessian_is_the_hessian_along_the_flow(seq, b0):
    """``L'(0) = -(u, J x B - grad p)``, ``L''(0) = (u, H u)`` and the polarised ``(u, H v)``, for random ``u``,
    ``v`` with divergence."""
    even = seq.even
    p = _pressure(seq, b0)
    _, _, J, JxB = compute_force(b0, seq)
    load = even.M[2] @ JxB - even.vector_load_values(pressure_gradient(p, seq), 1, 2)
    H = second_variation(seq, b0, J, p=p)
    k1, k2 = jax.random.split(jax.random.PRNGKey(1))
    u, v = _velocity(seq, k1), _velocity(seq, k2)
    d1u, qu = _derivatives(seq, b0, p, u)
    _, qv = _derivatives(seq, b0, p, v)
    _, quv = _derivatives(seq, b0, p, u + v)
    force = -float(u @ load)
    uHu, uHv = float(u @ H(u)), float(u @ H(v))
    polar = 0.5 * (quv - qu - qv)
    band = 1e3 * max(seq.tol, eps())
    print(f"\n  L'(0) {d1u:+.6e} vs -(u, J x B - grad p) {force:+.6e},  L''(0) {qu:+.6e} vs (u, H u) {uHu:+.6e},  "
          f"polarised {polar:+.6e} vs (u, H v) {uHv:+.6e}")
    assert abs(d1u - force) < band * (abs(d1u) + float(seq.odd.l2_norm(b0, 2)))
    assert abs(qu - uHu) < band * abs(qu)
    assert abs(polar - uHv) < band * (abs(qu) + abs(qv))


def test_compressible_newton_descends(seq, b0):
    """A few compressible Newton steps lower ``L`` at every step, by the amount the line search predicts, and
    lower the force."""
    ts = TimeStepper(seq=seq, newton=True, compressible=True)
    res = relax(initial_state(b0, ts, p=_pressure(seq, b0)), ts, steps=STEPS, chunk=CHUNK, verbose=False)
    dE, dE_ls = np.asarray(res.trace["dE"]), np.asarray(res.trace["dE_ls"])
    F = np.asarray(res.trace["F"])
    print(f"\n  ||F|| {F[0]:.3e} -> {F[-1]:.3e}, dE {dE}, |dE - dE_ls| max {np.abs(dE - dE_ls).max():.2e},  "
          f"MINRES {res.trace['newton_it']},  int p {res.qoi['p_int'][0]:.6e} -> {res.qoi['p_int'][-1]:.6e}")
    assert np.all(dE < eps() * res.E0), f"L not monotone: {dE}"
    assert np.all(np.abs(dE - dE_ls) < 1e2 * max(seq.tol, eps()) * res.E0)
    assert F[-1] < F[0]
