"""The shells of a prescribed pressure: their velocities have the divergence they should, the shell maps are
transposes of each other, and their rows are the derivative of the energy less the target pressure's work.

A shell velocity ``w`` (:func:`mrx.relaxation.shells.shell_maps`) is explicit given its divergence ``f``, the
projection of the shell function less its mean onto the 3-forms by a mass solve. ``G_2 w = f`` must hold to
round-off: that makes the step's space the one of any other velocity with that divergence. ``f`` is the projection
and the map's transpose its transpose to the tolerance of the mass solve. Along the ideal flow of ``w`` the energy
``W = ||B||^2 / 2 - int p* dV`` changes at the rate ``E'(0) - int p* div w dV``, and the shell's row in
:func:`mrx.relaxation.newton.shell_direction` is ``-(w, J x B) - g_p``. The magnetic part is checked against the
flow of the stepper (as for a divergence-free velocity in ``test_newton.py``), the pressure part against the
divergence.

A user's run with ``--pressure.target`` ends at the target pressure. On the fixture a target a few percent off the
file's pressure is reached by the loop of ``scripts/relax.py``: the flux label, the target on it and the shells
rebuilt before every chunk of Newton steps.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mrx.flux_label import ANISOTROPY
from mrx.precision import DTYPE, eps
from mrx.relaxation.loop import TimeStepper, initial_state, relax
from mrx.relaxation.newton import NEWTON_MASS_SOLVE_TOL
from mrx.relaxation.physics import compute_force, current, weak_pressure
from mrx.relaxation.shells import build_shells, label, load_target, pressure_error, shell_maps, shell_table
from test.test_newton import _energy_derivatives, _flow

#: Newton steps of the prescribed-pressure run, in chunks of relax.py's default length with the shells rebuilt before
#: each.
PRESSURE_STEPS, PRESSURE_CHUNK = 20, 10
#: The changed target's error after them: 2.74e-3 in float64, 2.67e-3 in float32 (2026-11-17).
PRESSURE_ERROR = 5e-3


def test_shell_velocity_and_row(seq, b0):
    """``G_2 w = f`` for the middle shell, ``f`` the projection of ``g - <g>`` and ``velocity_t`` the transpose of
    ``velocity`` to a multiple of the mass solve's tolerance, ``E'(0) = -(w, J x B)`` along its flow, and the pressure
    part ``int p* f dxi`` equal to ``int p* div w dV``."""
    even = seq.even
    r = seq.quad.x[:, 0]
    K = seq.basis_0.Lambda[0].n
    Q = jnp.eye(K)[:, K // 2 - 1:K // 2 + 1]           # two shells, the logical radius as the label, s = r^2
    m = even._scalar_load_values(jnp.ones_like(r), 3)
    e = even.M[3].solve(m)
    tol = NEWTON_MASS_SOLVE_TOL
    div, div_t, velocity, velocity_t = shell_maps(seq, r, Q, e, m, tol)
    c = jnp.array([1.0, -0.5])
    f, w = div(c), velocity(c)
    div_err = float(jnp.linalg.norm(even.G[2] @ w - f) / jnp.linalg.norm(f))
    values, indices = shell_table(seq, r)
    w_e = seq.quad.w * even.evaluate_at_quadrature(e, 3)[:, 0]
    g = jnp.sum(values * (Q @ c)[indices], axis=1)
    proj = even.M[3].solve(even._scalar_load_values(g - jnp.sum(w_e * g) / jnp.sum(w_e), 3))
    proj_err = float(even.l2_norm(f - proj, 3) / even.l2_norm(proj, 3))
    y = jax.random.normal(jax.random.PRNGKey(0), w.shape)
    adjoint = abs(float(y @ w - c @ velocity_t(y))) / float(jnp.linalg.norm(y) * jnp.linalg.norm(w))
    d1, _ = _energy_derivatives(seq, b0, w)
    _, _, _, JxB = compute_force(b0, seq)
    force = -float(w @ (even.M[2] @ JxB))
    # a target that is a 3-form: int p* div w dV = p3 . M_3 G_2 w
    p3 = even.M[3].solve(even._scalar_load_values(1.0 - r ** 2, 3))
    p_q = even.evaluate_at_quadrature(p3, 3)[:, 0] / seq.jacobian_j
    g_p = float(c @ div_t(even._scalar_load_values(p_q, 3)))
    exact = float(p3 @ (even.M[3] @ (even.G[2] @ w)))
    band = 1e3 * max(seq.tol, eps())
    print(f"\n  |G_2 w - f| / |f| {div_err:.1e},  |f - P_3 (g - <g>)| / |P_3 (g - <g>)| {proj_err:.1e},  "
          f"adjoint {adjoint:.1e},  E'(0) {d1:+.6e} vs -(w, J x B) {force:+.6e},  "
          f"g_p {g_p:+.6e} vs p . M_3 G_2 w {exact:+.6e}")
    assert div_err < 1e3 * eps()
    assert proj_err < 10 * max(tol, eps())
    assert adjoint < 10 * max(tol, eps())
    assert abs(d1 - force) < band * float(seq.odd.l2_norm(b0, 2) * seq.odd.l2_norm(_flow(seq, w)(b0), 2))
    assert abs(g_p - exact) < band * float(jnp.linalg.norm(p3) * jnp.linalg.norm(even.M[3] @ f))


def test_prescribed_pressure_reaches_a_changed_target(seq, b0):
    """Twenty Newton steps with the file's pressure profile reshaped by ``1 + 0.05 (1 - 2 s)`` as the target, in two
    chunks with the label and the shells rebuilt before each as ``scripts/relax.py --pressure.target`` does, take
    positive steps, lower the energy less the target pressure's work ``W`` at every step, and end at the target.

    Its error in the measure of the run's record (:func:`mrx.relaxation.shells.pressure_error`) falls from 4.6e-2 to
    2.74e-3 in float64 (2.67e-3 in float32), and to 1.2e-3 after 50 steps, the fixture's floor. The end state's error
    against the unchanged file profile is 2.65e-2, the size of the change (2.5e-2), so a run that ignored the change
    would miss the target by ten times the measured error. The bounds are 5e-3 and a fifth of the error against the
    file's profile (measured 0.103 of it). 190 s on four CPU cores after the fixture (2026-11-17).
    Skipped in float32, where it reaches the same error but a Newton step costs three times as much on the CPU.
    """
    if DTYPE != np.float64:
        pytest.skip("a float32 Newton step costs three times a float64 one on the CPU, the float64 run covers the claim")
    file_p = load_target(seq, "file").profile

    def target(s):
        return file_p(s) * (1.0 + 0.05 * (1.0 - 2.0 * s))

    ts = TimeStepper(seq=seq, newton=True)
    state, T, dW, dt_star, E0 = initial_state(b0, ts), None, [], [], None
    for _ in range(PRESSURE_STEPS // PRESSURE_CHUNK):
        s_q, T, _ = label(seq, state.B_n, ANISOTROPY, T)
        ts = ts.with_shells(build_shells(seq, state.B_n, s_q, target(s_q), ts.newton_penalty,
                                         ts.newton_mass_solve_tol))
        res = relax(state, ts, steps=PRESSURE_CHUNK, chunk=PRESSURE_CHUNK, verbose=False)
        state, E0 = res.state, res.E0 if E0 is None else E0
        dW += list(np.asarray(res.trace["dW"], dtype=float))
        dt_star += list(np.asarray(res.trace["dt_star"], dtype=float))
    s_q, _, _ = label(seq, state.B_n, ANISOTROPY, T)
    p_w = weak_pressure(current(state.B_n, seq), state.B_n, seq)
    err, _ = pressure_error(seq, p_w, s_q, target(s_q), int(seq.ns[0]))
    err_file, _ = pressure_error(seq, p_w, s_q, file_p(s_q), int(seq.ns[0]))
    print(f"\n  |p - p*| / |p*| {err:.3e}, against the file's profile {err_file:.3e}, max dW / E0 {max(dW) / E0:+.1e}")
    assert min(dt_star) > 0.0
    assert max(dW) < eps() * E0, f"W not monotone: {dW}"
    assert err < PRESSURE_ERROR and err < 0.2 * err_file
