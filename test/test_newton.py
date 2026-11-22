"""The Newton direction is the second-order model of the energy along the ideal flow, and it descends.

For a fixed divergence-free velocity ``u`` (an even 2-form) the ideal flow of the relaxation is
``dB/dt = A_u B`` with ``A_u B = G_1 M_1^{-1} load(u x H)``, ``H = Pi_1 B`` the L2 projection onto the free 1-forms,
the increment of the time step. It is linear in ``B``, so ``B(t) = sum_n t^n A_u^n B / n!`` and the energy ``E(t) = ||B(t)||^2_M / 2`` has

    E'(0) = (B, A_u B)_M,        E''(0) = ||A_u B||^2_M + (B, A_u^2 B)_M.

The force pairing ``-(u, J x H)_M`` must equal ``E'(0)``, and the second variation ``H`` of
:func:`mrx.relaxation.newton.second_variation` must reproduce ``E''(0) = (u, H u)`` for every ``u``. Both sides
are computed from different code: the left from the stepper's ``A_u`` alone, the right from the current and the
product loads. By polarisation the bilinear form is checked too, ``(u, H v) = (q(u + v) - q(u) - q(v)) / 2``
with ``q(w)`` the ``E''(0)`` of the flow along ``w``. Without the terms in ``J`` the second variation is its
Gauss-Newton part, the first term ``||A_u B||^2_M`` of ``E''(0)``.

MINRES needs a symmetric positive definite preconditioner. The spline atom is ``P = C P_1 C`` with ``P_1`` the
preconditioner of the 1-form Laplacian and ``C`` real and diagonal on the Fourier modes of the raw 1-forms, so
``(x, P x) > 0`` and ``(x, P y) = (y, P x)`` for dual 1-forms ``x``, ``y``. The preconditioners are stored and applied
in float32 (:data:`mrx.precision.PRECOND_DTYPE`), so the symmetry holds to the round-off of float32.

A few steps of the gradient descent of the paper take positive steps, lower the energy and keep ``div B`` at round-off
and the helicity up to grid-scale effects. The ten Newton steps of a run are on the Solov'ev fixture
(test_solovev.py), where the equilibrium is known.

The resistive step ``(M_2 + eps L_2) (B_1 - B_0) = -eps L_2 (B_0 - B_ref)`` lowers the distance from the reference
by its energy identity ``(e_1, M_2 (e_0 - e_1)) = eps ||J(e_1)||^2_M``, ``e = B - B_ref``, ``J`` the weak current.
"""
import jax
import jax.numpy as jnp
import numpy as np

from mrx.precision import PRECOND_DTYPE, eps
from mrx.relaxation.loop import TimeStepper, initial_state, radial_cell_sq, relax
from mrx.relaxation.newton import (NEWTON_PENALTY, atom_profiles, parallel_penalty_profile, second_variation,
                                   spline_preconditioner)
from mrx.relaxation.physics import compute_force, current, resistive_step
from test.conftest import divergence_free, energy_derivatives, ideal_flow

GD_STEPS = 4
#: |H_end - H_0| / (2 E_0) over the gradient-descent steps, grid-scale reconnection: 1.4e-10 measured in float64 on
#: shearedE (2026-11-22).
HELICITY_DRIFT = 3e-10


def test_second_variation_is_the_hessian_along_the_flow(seq, b0):
    """``E'(0) = -(u, J x H)``, ``E''(0) = (u, H u)`` and the polarised ``(u, H v)``, for random ``u``, ``v``."""
    even = seq.even
    _, _, J, JxB = compute_force(b0, seq)
    H = second_variation(seq, b0, J)
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    u, v = divergence_free(seq, k1), divergence_free(seq, k2)
    d1u, qu = energy_derivatives(seq, b0, u)
    _, qv = energy_derivatives(seq, b0, v)
    _, quv = energy_derivatives(seq, b0, u + v)
    force = -float(u @ (even.M[2] @ JxB))
    uHu, uHv = float(u @ H(u)), float(u @ H(v))
    polar = 0.5 * (quv - qu - qv)
    band = 1e3 * max(seq.tol, eps())
    print(f"\n  E'(0) {d1u:+.6e} vs -(u, J x H) {force:+.6e},  E''(0) {qu:+.6e} vs (u, H u) {uHu:+.6e},  "
          f"polarised {polar:+.6e} vs (u, H v) {uHv:+.6e}")
    # (B, A_u B) is a cancelling dot product: the solve error of A_u B is tol |A_u B|, so the identity holds to
    # tol |B| |A_u B|, not to tol |E'(0)|.
    assert abs(d1u - force) < band * float(seq.odd.l2_norm(b0, 2) * seq.odd.l2_norm(ideal_flow(seq, u)(b0), 2))
    assert abs(qu - uHu) < band * abs(qu)
    assert abs(polar - uHv) < band * (abs(qu) + abs(qv))


def test_gauss_newton_part_is_the_first_term(seq, b0):
    """Without ``J`` the second variation is the first term of ``E''(0)``, ``(u, H u) = ||A_u B||^2_M``, for a random
    ``u``. The left side comes from the stepper's flow alone."""
    H = second_variation(seq, b0)
    u = divergence_free(seq, jax.random.PRNGKey(6))
    first = float(seq.odd.l2_norm_sq(ideal_flow(seq, u)(b0), 2))
    uHu = float(u @ H(u))
    print(f"\n  ||A_u B||^2 {first:.6e} vs (u, H u) without J {uHu:.6e}")
    assert abs(first - uHu) < 1e3 * max(seq.tol, eps()) * first


def test_spline_atom_is_symmetric_positive_definite(seq, b0):
    """``(x, P x) > 0`` and ``(x, P y) = (y, P x)`` for the Newton preconditioner ``P`` at the equilibrium's field and
    random dual 1-forms ``x``, ``y``. The defect ``|(x, P y) - (y, P x)|`` over ``sqrt((x, P x) (y, P y))`` is the
    round-off of the float32 application: 2.4e-9 with ``MRX_DTYPE=float64`` and 3.6e-8 with float32 (2026-11-17),
    against the bound of 100 float32 epsilons, 1.2e-5. The Laplacian atom's own arrays applied in float64 are
    symmetric to 1e-17."""
    even = seq.even
    profiles = atom_profiles(seq, b0)
    atom = spline_preconditioner(seq, profiles, parallel_penalty_profile(profiles, NEWTON_PENALTY))
    k1, k2 = jax.random.split(jax.random.PRNGKey(3))
    x, y = jax.random.normal(k1, (even.n(1),)), jax.random.normal(k2, (even.n(1),))
    Px, Py = atom(even, x), atom(even, y)
    xPx, yPy, xPy, yPx = float(x @ Px), float(y @ Py), float(x @ Py), float(y @ Px)
    print(f"\n  (x, P x) {xPx:.6e}, (y, P y) {yPy:.6e}, (x, P y) - (y, P x) {xPy - yPx:+.3e}")
    assert xPx > 0.0 and yPy > 0.0
    assert abs(xPy - yPx) < 1e2 * float(np.finfo(PRECOND_DTYPE).eps) * np.sqrt(xPx * yPy)


def test_gradient_descent_descends(seq, b0):
    """A few steps of the gradient descent of the paper (the smoothed force as the velocity) move along the
    descent direction (``dt* > 0``), lower the energy at every step, keep ``div B`` at round-off (the topological
    curl) and the helicity up to grid-scale reconnection."""
    ts = TimeStepper(seq=seq, newton=False)
    res = relax(initial_state(b0, ts), ts, steps=GD_STEPS, chunk=GD_STEPS, verbose=False)
    dE = np.asarray(res.trace["dE"], dtype=float)
    H = np.asarray(res.qoi["helicity"], dtype=float)
    drift = abs(H[-1] - H[0]) / (2 * res.E0)
    print(f"\n  dE {dE.max():+.3e} (max), ||div B|| {float(res.trace['div'][-1]):.1e}, dH / 2E0 {drift:.2e}")
    assert np.all(np.asarray(res.trace["dt_star"]) > 0.0)
    # dE is formed in the stored precision, so a step whose true change is below an epsilon of E can read +eps
    assert np.all(dE < eps() * res.E0), f"energy not monotone: {dE}"
    assert float(res.trace["div"][-1]) < eps(1e4) * np.sqrt(2 * (res.E0 + dE.sum()))
    assert drift < HELICITY_DRIFT


def test_resistive_step_energy_identity(seq, b0):
    """One resistive step (:func:`mrx.relaxation.physics.resistive_step`) towards a reference ``B_ref = b0 - curl a``,
    ``a`` random, with the dose ``eps`` of the first radial cell squared: ``(e_1, M_2 (e_0 - e_1)) = eps
    ||J(e_1)||^2_M`` with ``e = B - B_ref``, the left side from the step's shifted-stiffness solve and the right from
    the weak current's mass solve, to a multiple of the solve tolerance, so ``||e||`` falls at any dose."""
    odd = seq.odd
    a = jax.random.normal(jax.random.PRNGKey(4), (odd.n(1),))
    curl = odd.G[1] @ a
    B_ref = b0 - 0.1 * curl * odd.l2_norm(b0, 2) / odd.l2_norm(curl, 2)
    dose = jnp.asarray(radial_cell_sq(seq))
    B1, info, _, _ = resistive_step(b0, seq, dose, B_ref)
    e0, e1 = b0 - B_ref, B1 - B_ref
    lhs = float(e1 @ (odd.M[2] @ (e0 - e1)))
    rhs = float(dose) * float(odd.l2_norm_sq(current(e1, seq), 1))
    print(f"\n  (e1, M (e0 - e1)) {lhs:.8e} vs eps |J(e1)|^2 {rhs:.8e}, |e| {float(odd.l2_norm(e0, 2)):.4e} -> "
          f"{float(odd.l2_norm(e1, 2)):.4e}, CG {int(info)}")
    assert int(info) > 0
    assert abs(lhs - rhs) < 1e2 * max(seq.tol, eps()) * rhs
