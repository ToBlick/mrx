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
"""
import jax
import jax.numpy as jnp

from mrx.precision import eps
from mrx.relaxation.newton import NEWTON_MASS_SOLVE_TOL
from mrx.relaxation.physics import compute_force
from mrx.relaxation.shells import shell_maps, shell_table
from test.test_newton import _energy_derivatives, _flow


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
