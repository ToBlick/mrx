"""The Newton direction is the second-order model of the energy along the ideal flow, and it descends.

For a fixed divergence-free velocity ``u`` (an even 2-form) the ideal flow of the relaxation is
``dB/dt = A_u B`` with ``A_u B = G_1 M_1^{-1} load(u x B)``, the increment of the time step. It is linear in
``B``, so ``B(t) = sum_n t^n A_u^n B / n!`` and the energy ``E(t) = ||B(t)||^2_M / 2`` has

    E'(0) = (B, A_u B)_M,        E''(0) = ||A_u B||^2_M + (B, A_u^2 B)_M.

The force pairing ``-(u, J x B)_M`` must equal ``E'(0)``, and the second variation ``H`` of
:func:`mrx.relaxation.newton.second_variation` must reproduce ``E''(0) = (u, H u)`` for every ``u``. Both sides
are computed from different code: the left from the stepper's ``A_u`` alone, the right from the current and the
product loads. By polarisation the bilinear form is checked too, ``(u, H v) = (q(u + v) - q(u) - q(v)) / 2``
with ``q(w)`` the ``E''(0)`` of the flow along ``w``. Without the terms in ``J`` the second variation is its
Gauss-Newton part, the first term ``||A_u B||^2_M`` of ``E''(0)``.

A short run with the production stepper takes positive steps, lowers the energy at every step, lowers the force,
keeps ``div B`` at round-off and the helicity up to grid-scale effects, records the best state, and writes a
checkpoint that reads back. A few steps of the gradient descent take positive steps and lower the energy too.

With elastic fields ``S_k`` (:mod:`mrx.relaxation.elastic`) the energy is ``(||B||^2 + sum_k ||S_k||^2) / 2`` with
every field moved by the same velocity, so ``E'(0)`` and ``E''(0)`` are sums over the fields, the force pairing
takes the force of every field and the second variation must reproduce the sum. The tests use the fields across
``B``, the production reference. They are not closed and cross the wall, so each moves by the full transport
``A_u S = G_1 M_1^{-1} load(u x S) - M_2^{-1} load(u div S)`` on the 2-forms without the wall condition. Newton and
gradient steps with them lower the total energy, start from the elastic energy ``lam^2 ||B||^2`` and checkpoint the
fields. The label fields are checked for their norm and their divergence.

The Gauss-Newton operator with lumped or exact elastic terms is a sum of squares, so it is symmetric and positive
semidefinite, and the lumped solve keeps the full force as the right-hand side. The length ``L`` of the parallel-flow
penalty of the elastic fields is checked on a torus, where it is known exactly.
"""
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mrx.operators import _mass_model
from mrx.precision import eps
from mrx.relaxation.loop import (TimeStepper, initial_state, read_checkpoint, relax,
                                 write_checkpoint)
from mrx.relaxation.elastic import (LABELS, field_period_length, frozen, label_fields, newton_terms,
                                    perpendicular_fields)
from mrx.relaxation.newton import (NEWTON_MASS_SOLVE_TOL, NEWTON_MAXITER, _newton_system, divergence_at_quadrature,
                                   newton_direction, second_variation)
from mrx.relaxation.physics import compute_divergence_norm, compute_force

STEPS, CHUNK = 10, 5
GD_STEPS = 4
# ||F||_end / ||F||_0 after 10 Newton steps on li383 (16, 32, 12) p=2: measured 9.1e-3 (2026-10-20).
NEWTON_FORCE_DROP = 0.5
# |H_end - H_0| / (2 E_0) over the 10 steps: grid-scale reconnection, 1.02e-6 in float64 (2026-10-20). 5x that.
HELICITY_DRIFT = 5e-6
#: The fluctuation level of the elastic tests, the top of the range of arXiv:2604.08878.
LAM = 0.1
ELASTIC_STEPS = 4
#: |E_el - lam^2 ||B||^2| / (lam^2 ||B||^2) of the projected fields across B, 7.6e-5 (2026-11-02). 5x that.
PROJECTION_BAND = 4e-4


def _divergence_free(seq, key):
    """A random divergence-free velocity, an even 2-form of unit norm."""
    even = seq.even
    w, _ = even.leray(jax.random.normal(key, (even.n(2),)), k=2)
    return w / even.l2_norm(w, 2)


def _flow(seq, u):
    """``X -> A_u X = G_1 M_1^{-1} load(u x X)`` for odd 2-forms ``X``, the increment of the ideal step."""
    odd, even = seq.odd, seq.even
    u_q = even.evaluate_at_quadrature(u, 2)
    return lambda X: odd.G[1] @ odd.M[1].solve(odd.cross_product_load_values(u_q, odd.evaluate_at_quadrature(X, 2),
                                                                           1, 2, 2))


def _energy_derivatives(seq, B, u):
    """``(E'(0), E''(0))`` of the energy along the flow of ``u``."""
    odd = seq.odd
    A = _flow(seq, u)
    AB = A(B)
    return float(B @ (odd.M[2] @ AB)), float(odd.l2_norm_sq(AB, 2) + B @ (odd.M[2] @ A(AB)))


def test_second_variation_is_the_hessian_along_the_flow(seq, b0):
    """``E'(0) = -(u, J x B)``, ``E''(0) = (u, H u)`` and the polarised ``(u, H v)``, for random ``u``, ``v``."""
    even = seq.even
    _, _, J, JxB = compute_force(b0, seq)
    H = second_variation(seq, b0, J)
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    u, v = _divergence_free(seq, k1), _divergence_free(seq, k2)
    d1u, qu = _energy_derivatives(seq, b0, u)
    _, qv = _energy_derivatives(seq, b0, v)
    _, quv = _energy_derivatives(seq, b0, u + v)
    force = -float(u @ (even.M[2] @ JxB))
    uHu, uHv = float(u @ H(u)), float(u @ H(v))
    polar = 0.5 * (quv - qu - qv)
    band = 1e3 * max(seq.tol, eps())
    print(f"\n  E'(0) {d1u:+.6e} vs -(u, J x B) {force:+.6e},  E''(0) {qu:+.6e} vs (u, H u) {uHu:+.6e},  "
          f"polarised {polar:+.6e} vs (u, H v) {uHv:+.6e}")
    # (B, A_u B) is a cancelling dot product: the solve error of A_u B is tol |A_u B|, so the identity holds to
    # tol |B| |A_u B|, not to tol |E'(0)|.
    assert abs(d1u - force) < band * float(seq.odd.l2_norm(b0, 2) * seq.odd.l2_norm(_flow(seq, u)(b0), 2))
    assert abs(qu - uHu) < band * abs(qu)
    assert abs(polar - uHv) < band * (abs(qu) + abs(qv))


def test_gauss_newton_part_is_the_first_term(seq, b0):
    """Without ``J`` the second variation is the first term of ``E''(0)``, ``(u, H u) = ||A_u B||^2_M``, for a random
    ``u``. The left side comes from the stepper's flow alone."""
    H = second_variation(seq, b0)
    u = _divergence_free(seq, jax.random.PRNGKey(6))
    first = float(seq.odd.l2_norm_sq(_flow(seq, u)(b0), 2))
    uHu = float(u @ H(u))
    print(f"\n  ||A_u B||^2 {first:.6e} vs (u, H u) without J {uHu:.6e}")
    assert abs(first - uHu) < 1e3 * max(seq.tol, eps()) * first


def test_newton_relaxation_descends(seq, b0, tmp_path):
    """Ten Newton steps with the production stepper move along the descent direction (``dt* > 0``), lower the
    energy at every step and the force, conserve ``div B = 0`` and the helicity up to grid-scale effects, record
    the best state, and checkpoint."""
    ts = TimeStepper(seq=seq, newton=True)
    saved = []
    res = relax(initial_state(b0, ts), ts, steps=STEPS, chunk=CHUNK, verbose=False,
                on_chunk=lambda r: saved.append(r.steps))
    dE = np.asarray(res.trace["dE"], dtype=float)
    F = np.asarray(res.trace["F"], dtype=float)
    H = np.asarray(res.qoi["helicity"], dtype=float)
    resid = np.asarray(res.trace["resid"], dtype=float)
    div, E0 = float(res.trace["div"][-1]), res.E0
    print(f"\n  ||F|| {F[0]:.3e} -> {F[-1]:.3e}, dH/2E0 {abs(H[-1] - H[0]) / (2 * E0):.2e}, ||div B|| {div:.1e}")
    assert res.stop == "steps" and saved == [CHUNK, STEPS]
    # a step with dt* <= 0 is clipped to dt = 0 and would pass the energy test below
    assert np.all(np.asarray(res.trace["dt_star"]) > 0.0)
    # dE is formed in the stored precision, so a step whose true change is below an epsilon of E can read +eps
    assert np.all(dE < eps() * E0), f"energy not monotone: {dE}"
    assert F[-1] < NEWTON_FORCE_DROP * F[0]
    assert abs(H[-1] - H[0]) < HELICITY_DRIFT * 2 * E0
    # the topological curl keeps div B at round-off (measured 1.6e-15), where a solve-based curl would leave tol
    assert div < eps(1e4) * np.sqrt(2 * (E0 + dE.sum()))
    assert float(res.state.best.resid) <= resid.min() and int(res.state.best.step) == resid.argmin()

    path = os.path.join(tmp_path, "state.h5")
    write_checkpoint(path, res.state, STEPS, seq)
    state, step = read_checkpoint(path, ts)
    assert step == STEPS and np.array_equal(np.asarray(state.B_n), np.asarray(res.state.B_n))


def test_gradient_descent_descends(seq, b0):
    """A few steps of the gradient descent of the paper (the smoothed force as the velocity) move along the
    descent direction (``dt* > 0``), lower the energy at every step and keep ``div B`` at round-off."""
    ts = TimeStepper(seq=seq, newton=False)
    res = relax(initial_state(b0, ts), ts, steps=GD_STEPS, chunk=GD_STEPS, verbose=False)
    dE = np.asarray(res.trace["dE"], dtype=float)
    print(f"\n  dE {dE.max():+.3e} (max), ||div B|| {float(res.trace['div'][-1]):.1e}")
    assert np.all(np.asarray(res.trace["dt_star"]) > 0.0)
    assert np.all(dE < eps() * res.E0), f"energy not monotone: {dE}"
    assert float(res.trace["div"][-1]) < eps(1e4) * np.sqrt(2 * (res.E0 + dE.sum()))


def _transport(seq, u, view, closed):
    """``X -> A_u X`` for 2-forms ``X`` of ``view``, the increment of the ideal step: ``G_1 M_1^{-1} load(u x X)``, and
    for a field that is not closed minus the drag ``M_2^{-1} load(u div X)``."""
    u_q = seq.even.evaluate_at_quadrature(u, 2)

    def A(X):
        E = view.M[1].solve(view.cross_product_load_values(u_q, view.evaluate_at_quadrature(X, 2), 1, 2, 2))
        AX = view.G[1] @ E
        return AX if closed else AX - view.M[2].solve(view.vector_load_values(u_q * divergence_at_quadrature(view, X),
                                                                              2, 2))
    return A


def _elastic_energy_derivatives(seq, fields, u):
    """``(E'(0), E''(0), sum_f ||f|| ||A_u f||)`` of ``sum_f ||f||^2 / 2`` along the flow of ``u``, for the triples
    ``(view, f, closed)`` of ``fields``. The last entry is the scale of the cancellation in ``E'(0)``."""
    d1 = d2 = scale = 0.0
    for view, f, closed in fields:
        A = _transport(seq, u, view, closed)
        Af = A(f)
        d1 += float(f @ (view.M[2] @ Af))
        d2 += float(view.l2_norm_sq(Af, 2) + f @ (view.M[2] @ A(Af)))
        scale += float(view.l2_norm(f, 2) * view.l2_norm(Af, 2))
    return d1, d2, scale


def _triples(seq, b0, S):
    """``(view, f, closed)`` of ``b0`` and of every elastic field of ``S``."""
    return [(seq.odd, b0, True)] + frozen(seq, S)


def _full_load(seq, b0, S):
    """The full force ``J x B + sum_k (curl S_k x S_k + S_k div S_k)`` of the stepper, a dual 2-form."""
    ts = TimeStepper(seq=seq, newton=True)
    return ts, ts._lorentz(b0, S, None, None)[3]


def test_label_fields(seq, b0):
    """The label fields are two divergence-free odd 2-forms of norm ``lam ||B||``."""
    odd = seq.odd
    S = label_fields(seq, LAM, b0)[LABELS.name]
    B_norm = float(odd.l2_norm(b0, 2))
    norms = [float(odd.l2_norm(S_k, 2)) for S_k in S]
    divs = [float(compute_divergence_norm(S_k, seq)) for S_k in S]
    print(f"\n  ||S_k|| / ||B|| {np.array(norms) / B_norm},  ||div S_k|| {divs}")
    assert np.allclose(norms, LAM * B_norm, rtol=1e3 * eps())
    assert max(divs) < eps(1e4) * LAM * B_norm


def test_second_variation_with_elastic_fields(seq, b0):
    """With the fields across ``B``: ``E'(0) = -(u, load)`` with the full force ``J x B + sum_k (curl S_k x S_k +
    S_k div S_k)``, ``E''(0) = (u, H u)`` and the polarised ``(u, H v)``, with the drag of the density in the flow."""
    S = perpendicular_fields(seq, LAM, b0)
    fields = _triples(seq, b0, S)
    _, load = _full_load(seq, b0, S)
    _, _, J, _ = compute_force(b0, seq)
    H = second_variation(seq, b0, J, frozen=frozen(seq, S))
    k1, k2 = jax.random.split(jax.random.PRNGKey(1))
    u, v = _divergence_free(seq, k1), _divergence_free(seq, k2)
    d1u, qu, scale = _elastic_energy_derivatives(seq, fields, u)
    _, qv, _ = _elastic_energy_derivatives(seq, fields, v)
    _, quv, _ = _elastic_energy_derivatives(seq, fields, u + v)
    force = -float(u @ load)
    uHu, uHv = float(u @ H(u)), float(u @ H(v))
    polar = 0.5 * (quv - qu - qv)
    band = 1e3 * max(seq.tol, eps())
    print(f"\n  E'(0) {d1u:+.6e} vs -(u, load) {force:+.6e},  E''(0) {qu:+.6e} vs (u, H u) {uHu:+.6e},  "
          f"polarised {polar:+.6e} vs (u, H v) {uHv:+.6e}")
    assert abs(d1u - force) < band * scale
    assert abs(qu - uHu) < band * abs(qu)
    assert abs(polar - uHv) < band * (abs(qu) + abs(qv))
    # the elastic terms are there: the Hessian of B alone misses them by about lam^2 of the total
    assert abs(float(u @ second_variation(seq, b0, J)(u)) - uHu) > 0.1 * LAM ** 2 * abs(qu)


@pytest.mark.parametrize("newton", [True, False], ids=["newton", "gradient"])
def test_elastic_relaxation_descends(seq, b0, tmp_path, newton):
    """A few steps with the fields across ``B`` move along the descent direction (``dt* > 0``), lower the total energy
    at every step, move the elastic fields and checkpoint them. The fields start with the elastic energy
    ``lam^2 ||B||^2`` of ``sum_k S_k S_k^T = lam^2 (|B|^2 I - B B^T)``, up to their L2 projection."""
    S = perpendicular_fields(seq, LAM, b0)
    ts = TimeStepper(seq=seq, newton=newton)
    start = initial_state(b0, ts, S=S)
    res = relax(start, ts, steps=ELASTIC_STEPS, chunk=ELASTIC_STEPS, verbose=False)
    dE = np.asarray(res.trace["dE"], dtype=float)
    moved = max(float(view.l2_norm(S_k - S0_k, 2) / view.l2_norm(S0_k, 2))
                for (view, S_k, _), (_, S0_k, _) in zip(frozen(seq, res.state.S), frozen(seq, start.S)))
    E_el = np.asarray(res.qoi["E_el"], dtype=float)
    E_ref = float(LAM * seq.odd.l2_norm(b0, 2)) ** 2
    print(f"\n  dE {dE.max():+.3e} (max), E_el {E_el[0]:.6e} -> {E_el[-1]:.6e}, lam^2 ||B||^2 {E_ref:.6e}, "
          f"S moved {moved:.2e}")
    assert np.all(np.asarray(res.trace["dt_star"]) > 0.0)
    assert np.all(dE < eps() * res.E0), f"energy not monotone: {dE}"
    assert abs(E_el[0] - E_ref) < PROJECTION_BAND * E_ref
    assert moved > 0.0

    path = os.path.join(tmp_path, "state.h5")
    write_checkpoint(path, res.state, ELASTIC_STEPS, seq)
    state, _ = read_checkpoint(path, ts)
    for name in S:
        assert np.array_equal(np.asarray(state.S[name]), np.asarray(res.state.S[name]))


@pytest.mark.parametrize("lumped", [True, False], ids=["lumped", "exact"])
def test_gauss_newton_operator_is_a_sum_of_squares(seq, b0, lumped):
    """Without the current ``(u, H v) = (A_u B, A_v B)_M + sum_k (Q_u S_k, Q_v S_k)_M``. Exact, ``Q_u S = A_u S`` is the
    full transport with the drag of the density. Lumped, ``Q_u S = G_1 P load(u x S)`` with ``P`` the mass
    preconditioner and no drag. So ``H`` is symmetric and positive semidefinite."""
    S = perpendicular_fields(seq, LAM, b0)
    H = second_variation(seq, b0, frozen=frozen(seq, S), lumped=lumped)

    def squares(u):
        """The pairs ``(view, Q)`` of every field along ``u``."""
        u_jk = seq.even.evaluate_at_quadrature(u, 2)

        def Q(view, f, closed):
            if not lumped:
                return _transport(seq, u, view, closed)(f)
            return view.G[1] @ _mass_model(view, 1).apply(
                view.cross_product_load_values(u_jk, view.evaluate_at_quadrature(f, 2), 1, 2, 2))
        return [(seq.odd, _flow(seq, u)(b0))] + [(view, Q(view, f, closed)) for view, f, closed in frozen(seq, S)]

    k1, k2 = jax.random.split(jax.random.PRNGKey(2))
    u, v = _divergence_free(seq, k1), _divergence_free(seq, k2)
    Qu, Qv = squares(u), squares(v)
    uu = sum(float(view.l2_norm_sq(Q, 2)) for view, Q in Qu)
    vv = sum(float(view.l2_norm_sq(Q, 2)) for view, Q in Qv)
    uv = sum(float(Q_u @ (view.M[2] @ Q_v)) for (view, Q_u), (_, Q_v) in zip(Qu, Qv))
    uHu, uHv, vHu = float(u @ H(u)), float(u @ H(v)), float(v @ H(u))
    band = 1e3 * max(seq.tol, eps())
    print(f"\n  squares {uu:.6e} vs (u, H u) {uHu:.6e},  polarised {uv:+.6e} vs (u, H v) {uHv:+.6e}, (v, H u) {vHu:+.6e}")
    assert abs(uu - uHu) < band * uu
    assert abs(uv - uHv) < band * np.sqrt(uu * vv)
    assert abs(uHv - vHu) < band * np.sqrt(uu * vv)


def test_lumped_newton_solve_keeps_the_load(seq, b0):
    """The lumped terms change the matrix of the Newton solve and not its right-hand side, the full force:
    ``(u, load) = (a, A a) > 0`` up to the tolerance of the solve, so ``u`` descends the energy.

    The sign of ``info`` is not tested. It compares the true residual with ``tol`` exactly, and MINRES runs in float32, so
    its recurrence stops following that residual at tight tolerances. At ``tol`` 1e-3 the true residual is 1.04e-3 lumped,
    1.04e-3 exact and 2.4e-3 without elastic fields (2026-11-01). The solve must stop on its tolerance, not on the cap."""
    S = perpendicular_fields(seq, LAM, b0)
    ts, load = _full_load(seq, b0, S)
    tol = 1e-2
    u, a, info = newton_direction(seq, b0, load, jnp.zeros(seq.even.n(1)), tol=tol,
                                  elastic=newton_terms(seq, S, b0, ts.newton_mu, True))
    _, _, A = _newton_system(seq, b0, ts.newton_mu, NEWTON_MASS_SOLVE_TOL, frozen(seq, S), True)
    u_load, aAa = float(u @ load), float(a @ A(a))
    print(f"\n  MINRES {int(info)},  (u, load) {u_load:.6e} vs (a, A a) {aAa:.6e}")
    assert abs(int(info)) < NEWTON_MAXITER
    assert u_load > 0.0
    assert abs(u_load - aAa) < tol * u_load


def test_field_period_length():
    """On a torus of major radius ``R_0`` with ``N`` field periods one period of the axis has the length
    ``2 pi R_0 / N``, so ``L = R_0 / N``."""
    R_0, N = 1.7, 3

    def torus(x):
        phi = 2 * jnp.pi * x[2] / N
        R = R_0 + 0.3 * x[0] * jnp.cos(2 * jnp.pi * x[1])
        return jnp.stack([R * jnp.cos(phi), R * jnp.sin(phi), 0.3 * x[0] * jnp.sin(2 * jnp.pi * x[1])])
    L = float(field_period_length(torus))
    print(f"\n  L {L:.12f} vs R_0 / N {R_0 / N:.12f}")
    assert abs(L - R_0 / N) < 1e3 * eps() * R_0 / N
