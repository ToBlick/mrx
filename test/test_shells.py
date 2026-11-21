"""The shells of a prescribed pressure: their velocities are gradients with the divergence they should have, and
their rows are the derivative of the energy less the target pressure's work.

A shell velocity ``xi`` (:func:`mrx.relaxation.shells.shell_gradients`) is the 2-form part of a mixed Poisson solve
whose right-hand side is the projection ``f`` of the shell function less its mean onto the 3-forms. ``G_2 xi = f``
must hold to the tolerance of that solve: that makes the step's space the one of any other velocity with that
divergence. ``xi`` is a gradient, so it is orthogonal to every curl to round-off. Along the ideal flow of ``xi`` the
energy ``W = ||B||^2 / 2 - int p* dV`` changes at the rate ``E'(0) - int p* div xi dV``, and the shell's row in
:func:`mrx.relaxation.newton.target_direction` is ``-(xi, J x h) - g_p``. The magnetic part is checked against the
flow of the stepper (as for a divergence-free velocity in ``test_newton.py``), the pressure part against the
divergence.
"""
import jax
import jax.numpy as jnp
import numpy as np

from mrx.precision import eps
from mrx.relaxation.newton import NEWTON_MASS_SOLVE_TOL, NEWTON_PENALTY
from mrx.relaxation.physics import compute_force
from mrx.relaxation.shells import PROFILE_CELLS, PROFILE_DEGREE, build_shells, shell_table
from test.test_newton import _energy_derivatives, _flow


def test_shell_velocity_and_row(seq, b0):
    """The shells of :func:`build_shells` with the logical radius as the label: ``G_2 xi_j = f_j`` to a multiple
    of the Poisson tolerance, with ``f_j`` the projection of ``g_j - <g_j>`` for the zero-sum combination ``g_j`` of
    the splines, ``(xi_j, curl a) = 0`` to round-off for a random ``a``, ``E'(0) = -(xi, J x h)`` along the flow of
    the middle shell, and the pressure part ``g_p`` equal to ``int p* div xi_j dV``. The shells' 18 Poisson solves
    take 244 to 369 conjugate gradients. Measured float64 / float32: divergence 1.0e-8 / 6.1e-7 (float32 stores
    ``xi`` rounded to its working precision), orthogonality 8.1e-14 / 1.0e-9, pressure part 4.4e-16 / 8.9e-8. 22 to 29
    s float64, 77 to 150 s float32 on four CPU cores (2026-11-19)."""
    even = seq.even
    r = seq.quad.x[:, 0]
    s = r ** 2
    # a target that is a 3-form: int p* div xi dV = p3 . M_3 G_2 xi
    p3 = even.M[3].solve(even._scalar_load_values(1.0 - s, 3))
    p_q = even.evaluate_at_quadrature(p3, 3)[:, 0] / seq.jacobian_j
    shells, info = build_shells(seq, b0, s, p_q, NEWTON_PENALTY, NEWTON_MASS_SOLVE_TOL)
    Xi = shells.Xi
    K = PROFILE_CELLS + PROFILE_DEGREE
    Q = jnp.asarray(np.linalg.qr(np.eye(K) - 1.0 / K)[0][:, :K - 1])     # the combinations of build_shells
    values, indices = shell_table(K, s)
    e = even.M[3].solve(even._scalar_load_values(jnp.ones_like(s), 3))          # the 3-form of the constant 1
    w_e = seq.quad.w * even.evaluate_at_quadrature(e, 3)[:, 0]
    a = jax.random.normal(jax.random.PRNGKey(0), (even.n(1),))
    curl = even.G[1] @ a
    div_err, orth, pressure = [], [], []
    for j in range(K - 1):
        g = jnp.sum(values * Q[:, j][indices], axis=1)
        f = even.M[3].solve(even._scalar_load_values(g - jnp.sum(w_e * g) / jnp.sum(w_e), 3))
        div_xi = even.G[2] @ Xi[j]
        div_err.append(float(even.l2_norm(div_xi - f, 3) / even.l2_norm(f, 3)))
        orth.append(abs(float(Xi[j] @ (even.M[2] @ curl))) / float(even.l2_norm(Xi[j], 2) * even.l2_norm(curl, 2)))
        exact = float(p3 @ (even.M[3] @ div_xi))
        pressure.append(abs(float(shells.g_p[j]) - exact)
                        / float(jnp.linalg.norm(p3) * jnp.linalg.norm(even.M[3] @ div_xi)))
    xi = Xi[K // 2 - 1]
    d1, _ = _energy_derivatives(seq, b0, xi)
    _, _, _, JxB = compute_force(b0, seq)
    force = -float(xi @ (even.M[2] @ JxB))
    band = 1e3 * max(seq.tol, eps())
    info = np.asarray(info)
    print(f"\n  Poisson CG {info.min()} .. {info.max()},  |G_2 xi - f| / |f| {max(div_err):.1e},  "
          f"|(xi, curl a)| / |xi||curl a| {max(orth):.1e},  E'(0) {d1:+.6e} vs -(xi, J x h) {force:+.6e},  "
          f"|g_p - p . M_3 G_2 xi| {max(pressure):.1e}")
    assert info.min() > 0
    assert max(div_err) < 1e2 * max(seq.tol, eps())
    assert max(orth) < 1e4 * eps()
    assert abs(d1 - force) < band * float(seq.odd.l2_norm(b0, 2) * seq.odd.l2_norm(_flow(seq, xi)(b0), 2))
    assert max(pressure) < band
