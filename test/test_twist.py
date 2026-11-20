"""The poloidal flux slips, the unknowns of the current target.

As unknowns of the Newton step (:class:`mrx.relaxation.twist.Slips`) the twists of the radial spline steps change the
field by a divergence-free form without a ``zeta`` component, their map's transpose is its transpose, the energy's
derivative along each is ``-Z`` times the spline mean of the enclosed current, the same as the weak current's to the
discretisation, and the slip block is the energy's second derivative, positive definite. The run with a current target
is in test_relax_run.py.
"""
import jax.numpy as jnp
import numpy as np

from mrx.flux_label import ANISOTROPY
from mrx.precision import eps
from mrx.relaxation.physics import compute_divergence_norm
from mrx.relaxation.shells import PROFILE_CELLS, label
from mrx.relaxation.twist import (build_slips, current_error, current_on_label, current_target,
                                  enclosed_current)

#: The cells of the profile space on the fixture. Its 14 radial cells resolve 8 cells of the profile space, not the
#: default 16 of finer meshes: with 16 the slip block's condition is 1.1e7 and the moments miss the weak current by
#: 7.4e-3, with 8 the condition is 59 and the miss 1.55e-3 (2026-11-18).
CELLS = 8
#: The moments of the slips against the profile-space means of the weak current: 1.55e-3 of the largest in float64
#: and in float32, on CELLS cells (2026-11-18).
SLIP_WEAK_J = 2e-3


def test_slip_move_and_derivative(seq, b0):
    """The slips on the fixture's field and label, with the target 0.9 times its current: ``T d`` for a random ``d``
    is divergence-free with no ``zeta`` component to round-off and ``move_t`` is the transpose of ``move`` to
    round-off. ``W(d) = E(B + T d) - d . g`` is quadratic in ``d``, so its central difference along ``d`` is the
    assembled ``d . ((B, T_j)_M - g_j)`` and its second difference ``d . D d`` with the block ``D`` up to the
    round-off of the energy over the step. The moments ``-(B, T_j)_M / Z`` are the spline means of the weak
    current's enclosed flux to SLIP_WEAK_J of the largest. The block is positive definite, its eigenvalues 1.24 to
    73.2, the slip at the first spline included. 7 s on four CPU cores after the fixture, 16 s in float32
    (2026-11-18)."""
    _, T, _ = label(seq, b0, ANISOTROPY)
    mu_file = current_on_label(seq, b0, T, CELLS)
    slips, mu = build_slips(seq, b0, T, 0.9 * mu_file)
    move, move_t = slips.maps(seq)
    odd = seq.odd
    M2 = odd.M[2]
    rng = np.random.default_rng(0)
    d = jnp.asarray(rng.standard_normal(mu.size))
    y = jnp.asarray(rng.standard_normal(b0.size))
    Td = move(d)
    adjoint = abs(float(y @ Td - d @ move_t(y))) / float(jnp.linalg.norm(y) * jnp.linalg.norm(Td))
    div = float(compute_divergence_norm(Td, seq) / odd.l2_norm(Td, 2))
    Tdq = odd.evaluate_at_quadrature(Td, 2)
    zeta = float(jnp.abs(Tdq[:, 2]).max() / jnp.abs(Tdq).max())

    def W(t):
        Bt = b0 + t * Td
        return 0.5 * float(Bt @ (M2 @ Bt)) - t * float(d @ slips.g)
    E0 = W(0.0)
    D = np.linalg.inv(np.asarray(slips.D_inv, dtype=float))
    dDd = float(np.asarray(d) @ D @ np.asarray(d))
    h = 1e-2 * (E0 / dDd) ** 0.5                       # the step changes E by about 1e-4 E0
    fd = (W(h) - W(-h)) / (2 * h)
    sd = (W(h) - 2 * E0 + W(-h)) / h ** 2
    grad = float(d @ (move_t(M2 @ b0) - slips.g))
    # the spline means of the weak current inside the label's level sets
    x = np.linspace(0.0, 1.0, 1001)
    s_q, _, _ = label(seq, b0, ANISOTROPY, T)
    I_enc = np.concatenate([[0.0], enclosed_current(seq, b0, np.asarray(s_q), x[1:] ** 2)])
    mu_enc = np.asarray(current_target(lambda s: np.interp(s, x ** 2, I_enc), CELLS))
    weak = float(np.abs(np.asarray(mu) - mu_enc).max() / np.abs(mu_enc).max())
    ev = np.linalg.eigvalsh(D)
    print(f"\n  adjoint {adjoint:.1e}, div {div:.1e}, zeta {zeta:.1e}, first difference {fd:+.10e} vs {grad:+.10e}, "
          f"second {sd:.8e} vs {dDd:.8e}, moments vs weak current {weak:.2e}, block eigenvalues {ev.min():.2e} to "
          f"{ev.max():.2e}")
    assert adjoint < 1e3 * eps()
    assert div < 1e3 * eps()
    assert zeta < 1e3 * eps()
    assert abs(fd - grad) < 1e2 * eps() * E0 / h
    assert abs(sd - dDd) < 1e2 * eps() * E0 / h ** 2
    assert weak < SLIP_WEAK_J
    assert ev.min() > 0.0


def test_current_error_of_a_zero_target():
    """:func:`mrx.relaxation.twist.current_error` is relative to the target, and absolute for a zero target, the
    closure of no net current on any surface: then it is the root mean square of the moments in T m, weighted by the
    widths of the splines. A target twice the moments gives 1/2, a zero target and constant moments their value, both
    to round-off. Pure numpy, milliseconds."""
    mu = np.full(PROFILE_CELLS + 2, 0.25)
    assert abs(current_error(mu, 2.0 * mu) - 0.5) < 10 * np.finfo(float).eps
    assert abs(current_error(mu, np.zeros_like(mu)) - 0.25) < 10 * np.finfo(float).eps
