"""The Newton direction is the second-order model of the energy along the ideal flow, and it descends.

For a fixed divergence-free velocity ``u`` (an even 2-form) the ideal flow of the relaxation is
``dB/dt = A_u B`` with ``A_u B = G_1 M_1^{-1} load(u x B)``, the increment of the time step. It is linear in
``B``, so ``B(t) = sum_n t^n A_u^n B / n!`` and the energy ``E(t) = ||B(t)||^2_M / 2`` has

    E'(0) = (B, A_u B)_M,        E''(0) = ||A_u B||^2_M + (B, A_u^2 B)_M.

The force pairing ``-(u, J x B)_M`` must equal ``E'(0)``, and the second variation ``H`` of
:func:`mrx.relaxation.newton.second_variation` must reproduce ``E''(0) = (u, H u)`` for every ``u``. Both sides
are computed from different code: the left from the stepper's ``A_u`` alone, the right from the current and the
product loads. By polarisation the bilinear form is checked too, ``(u, H v) = (q(u + v) - q(u) - q(v)) / 2``
with ``q(w)`` the ``E''(0)`` of the flow along ``w``. The parallel-flow penalty adds a positive semidefinite
term. With the Chebyshev mass inverse in place of the mass solves the Hessian stays symmetric.

The Newton direction is divergence-free to round-off and pairs positively with the force, and a short run with
the production stepper lowers the energy at every step, lowers the force, keeps ``div B`` at round-off and the
helicity up to grid-scale effects, records the best state, and writes a checkpoint that reads back. A checkpoint
without the ``angles`` attribute is refused on li383, whose angles are reversed on reading.
"""
import os

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mrx.precision import DTYPE, eps
from mrx.relaxation.loop import (TimeStepper, check_checkpoint, initial_state, read_checkpoint, relax,
                                 write_checkpoint)
from mrx.relaxation.newton import MassChebyshev, newton_direction, second_variation
from mrx.relaxation.physics import compute_force

STEPS, CHUNK = 10, 5
# ||F||_end / ||F||_0 after 10 Newton steps on li383 (8, 12, 12) p=2.
NEWTON_FORCE_DROP = 0.5
# |H_end - H_0| / (2 E_0) over the 10 steps: grid-scale reconnection, about 4e-6 in float64. 5x that.
HELICITY_DRIFT = 2e-5


def _divergence_free(seq, key):
    """A random divergence-free velocity, an even 2-form of unit norm."""
    even = seq.even
    w, _ = even.leray(jax.random.normal(key, (even.n(2),), dtype=DTYPE), k=2)
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


def test_parallel_penalty_is_positive_semidefinite(seq, b0):
    """The penalised Hessian minus the bare one is ``int w (u . B)^2 / |B|^2 dx >= 0`` for a positive weight
    ``w``."""
    _, _, J, _ = compute_force(b0, seq)
    w = jnp.ones(seq.quad.shape[0], dtype=DTYPE)
    H, H_pen = second_variation(seq, b0, J), second_variation(seq, b0, J, penalty=w)
    u = _divergence_free(seq, jax.random.PRNGKey(3))
    extra = float(u @ H_pen(u)) - float(u @ H(u))
    print(f"\n  (u, M_par u) {extra:+.6e}")
    assert extra > 0.0


def test_chebyshev_mass_hessian_is_symmetric(seq, b0):
    """The Chebyshev mass inverse meets its error bound against the mass solve, and the Hessian built with it is
    symmetric, ``(u, H v) = (v, H u)``."""
    odd = seq.odd
    _, _, J, _ = compute_force(b0, seq)
    S = MassChebyshev.build(seq, 0.3)
    lmin, lmax = (float(x) for x in S.bounds)
    rate = (np.sqrt(lmax / lmin) - 1) / (np.sqrt(lmax / lmin) + 1)
    b = odd.M[1] @ jax.random.normal(jax.random.PRNGKey(5), (odd.n(1),), dtype=DTYPE)
    x = odd.M[1].solve(b)
    err = float(odd.l2_norm(S(odd, b) - x, 1) / odd.l2_norm(x, 1))
    H = second_variation(seq, b0, J, mass_inverse=S)
    u, v = _divergence_free(seq, jax.random.PRNGKey(6)), _divergence_free(seq, jax.random.PRNGKey(7))
    uHv, vHu = float(u @ H(v)), float(v @ H(u))
    print(f"\n  bounds [{lmin:.3f}, {lmax:.3f}], error {err:.2e} (degree {S.steps}, bound {2 * rate ** S.steps:.2e}), "
          f"(u, H v) {uHv:+.6e} vs (v, H u) {vHu:+.6e}")
    assert err < 2 * rate ** S.steps < 0.3
    assert abs(uHv - vHu) < 1e3 * eps() * (abs(float(u @ H(u))) + abs(float(v @ H(v))))


def test_newton_direction_is_divergence_free_and_descends(seq, b0):
    """The Newton velocity has zero divergence to round-off and a positive pairing with the force."""
    even = seq.even
    F, _, J, JxB = compute_force(b0, seq)
    MF = even.M[2] @ F
    u, _, info = newton_direction(seq, b0, J, even.M[2] @ JxB, jnp.zeros(even.n(1), dtype=DTYPE), tol=1e-2,
                                  maxiter=40)
    div = float(even.l2_norm(even.G[2] @ u, 3))
    cos = float(u @ MF) / float(even.l2_norm(u, 2) * jnp.sqrt(F @ MF))
    print(f"\n  MINRES info {int(info)}, ||div u|| {div:.2e}, descent cosine {cos:+.4f}")
    assert div < 1e2 * eps() * float(even.l2_norm(u, 2))
    assert cos > 0.0


def test_newton_relaxation_descends(seq, b0, tmp_path):
    """Ten Newton steps with the production stepper lower the energy at every step and the force, conserve
    ``div B = 0`` and the helicity up to grid-scale effects, record the best state, and checkpoint."""
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
    # dE is formed in the stored precision, so a step whose true change is below an epsilon of E can read +eps
    assert np.all(dE < eps() * E0), f"energy not monotone: {dE}"
    assert F[-1] < NEWTON_FORCE_DROP * F[0]
    assert abs(H[-1] - H[0]) < HELICITY_DRIFT * 2 * E0
    assert div < 1e3 * seq.tol * np.sqrt(2 * (E0 + dE.sum()))
    assert float(res.state.best.resid) <= resid.min() and int(res.state.best.step) == resid.argmin()

    path = os.path.join(tmp_path, "state.h5")
    write_checkpoint(path, res.state, STEPS, seq)
    state, step = read_checkpoint(path, ts)
    assert step == STEPS and np.array_equal(np.asarray(state.B_n), np.asarray(res.state.B_n))
    with h5py.File(path, "a") as fh:
        del fh.attrs["angles"]
    with pytest.raises(ValueError, match="left-handed"):
        check_checkpoint(path, seq)
