"""The poloidal flux slips, the unknowns of the current target.

As unknowns of the Newton step (:class:`mrx.relaxation.twist.Slips`) the twists of the radial spline steps change the
field by a divergence-free form without a ``zeta`` component, their map's transpose is its transpose, the energy's
derivative along each is ``-Z`` times the spline mean of the enclosed current, the same as the weak current's to the
discretisation, and the slip block is the energy's second derivative, positive definite.

The target transform's weak residual ``r_j = int (iota - iota*) M_j ds`` (:func:`mrx.relaxation.twist.iota_residual`)
is affine in the field along the slips, which keep the toroidal flux: one update of the slips by its derivative,
``d = -R^-1 r``, with the rest of the field frozen, zeroes it to round-off.
"""
import jax.numpy as jnp
import numpy as np

from mrx.flux_label import ANISOTROPY, label_condition, radius_flux
from mrx.precision import eps
from mrx.relaxation.initial_conditions import toroidal_flux, vacuum_field
from mrx.relaxation.physics import compute_divergence_norm, compute_force
from mrx.relaxation.shells import label, profile_cells
from mrx.relaxation.twist import (_slip_parts, build_slips, current_error, current_on_label, current_start,
                                  current_target, enclosed_current, iota_residual, slip_changes, slip_transform)

#: The moments of the slips against the profile-space means of the weak current, on the sequence's profile space
#: (:func:`mrx.relaxation.shells.profile_cells`, 5 cells on the tokamak): 2.08e-4 of the largest in float64 and float32
#: (2026-11-23).
SLIP_WEAK_J = 4e-4
#: The current start on the tokamak's vacuum: the moments of the slips of ``B_J`` against the target's, as
#: :func:`mrx.relaxation.twist.current_error`, 3.7e-5 in float64 (2026-11-23).
START_MOMENTS = 1e-4


def test_slip_move_and_derivative(seq, b0):
    """The slips on the fixture's field and label, with the target 0.9 times its current: ``T d`` for a random ``d``
    is divergence-free with no ``zeta`` component to round-off and ``move_t`` is the transpose of ``move`` to
    round-off. ``W(d) = E(B + T d) - d . g`` is quadratic in ``d``, so its central difference along ``d`` is the
    assembled ``d . ((B, T_j)_M - g_j)`` and its second difference ``d . D d`` with the block ``D`` up to the
    round-off of the energy over the step. The moments ``-(B, T_j)_M / Z`` are the spline means of the weak
    current's enclosed flux to SLIP_WEAK_J of the largest. The block is positive definite, its eigenvalues 0.40 to
    12.9, the slip at the first spline included. 7 s on four CPU cores after the fixture, 16 s in float32
    (2026-11-18)."""
    _, T, _ = label(seq, b0, ANISOTROPY)
    mu_file = current_on_label(seq, b0, T, profile_cells(seq))
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
    mu_enc = np.asarray(current_target(lambda s: np.interp(s, x ** 2, I_enc), profile_cells(seq)))
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


def test_one_slip_update_zeroes_the_weak_iota_residual(eq, seq, b0):
    """The weak residual of the exact transform as the target on the fixture's field and label, on
    :func:`~mrx.relaxation.shells.profile_cells` cells: with ``R`` its derivatives along the slips (:func:`mrx.relaxation.twist.slip_transform`)
    the field ``B + T d``, ``d = -R^-1 r``, has the residual zero to round-off relative to ``r``. The slips have no
    ``zeta`` component, so the toroidal flux in the residual's denominator does not change: the identity is exact up
    to the condition of ``R``: 2.9e-5 to 5.6e-17 in float64 and 1.2e-8 in float32, the condition of ``R`` 76
    (2026-11-23)."""
    s_q, T, _ = label(seq, b0, ANISOTROPY)
    slips, mu = build_slips(seq, b0, T, current_on_label(seq, b0, T, profile_cells(seq)))
    move = slips.maps(seq)[0]
    residual = iota_residual(seq, b0, s_q, eq.iota, mu.size)
    r0 = np.asarray(residual(b0)[0], dtype=float)
    R = slip_transform(residual, b0, move, mu.size)
    d = np.linalg.solve(R, -r0)
    r1 = np.asarray(residual(b0 + move(jnp.asarray(d, dtype=b0.dtype)))[0], dtype=float)
    print(f"\n  |r| {np.abs(r0).max():.3e} -> {np.abs(r1).max():.3e} after the update, cond R {np.linalg.cond(R):.2e}")
    assert np.abs(r1).max() < 1e4 * eps() * np.abs(r0).max()


def test_current_start(seq):
    """The current start (:func:`mrx.relaxation.twist.current_start`) on the tokamak's vacuum field, the harmonic
    2-form, with the label of the logical surfaces (:func:`mrx.flux_label.radius_flux`) and the target
    ``mu0 I*(s) = s (2 - s)``. ``B_J`` has no toroidal flux to round-off. The slip moments of ``B_J`` on that label,
    ``-(B_J, T_j)_M / Z``, are the target's within START_MOMENTS: the enclosed current of ``B_J`` is ``I*`` in the
    weak sense. ``J = sigma B`` is along the vacuum field, so ``B + eps B_J`` is force free to first order in ``eps``:
    doubling ``eps`` multiplies the force by 4 within 5 %. The vacuum has no poloidal component, so its mean transform
    (:func:`mrx.flux_label.label_condition`) is zero to round-off, and with ``B_J`` it is not."""
    B = vacuum_field(seq, 1.0)
    iota_vac = label_condition(seq, seq.odd.evaluate_at_quadrature(B, 2))[0]
    s_q = radius_flux(seq, seq.odd.evaluate_at_quadrature(B, 2))
    mu_star = current_target(lambda s: s * (2.0 - s), profile_cells(seq))
    B_J, info = current_start(seq, B, s_q, mu_star)
    flux = abs(float(toroidal_flux(seq, B_J)))
    mu = _slip_parts(seq, B_J, slip_changes(seq, s_q, int(mu_star.size) + 1))[0]
    err = current_error(np.asarray(mu), np.asarray(mu_star))
    odd = seq.odd
    eps_ = 1e-2 * float(odd.l2_norm(B, 2) / odd.l2_norm(B_J, 2))
    F1, F2 = (float(seq.even.l2_norm(compute_force(B + e * B_J, seq)[0], 2)) for e in (eps_, 2 * eps_))
    iota_start = label_condition(seq, seq.odd.evaluate_at_quadrature(B + B_J, 2))[0]
    print(f"\n  Hodge solve {int(info)}, toroidal flux of B_J {flux:.1e}, moments {err:.3e}, force ratio {F2 / F1:.4f}, "
          f"mean transform {iota_vac:.1e} -> {iota_start:.3e}")
    assert info > 0
    assert flux < 1e3 * eps()
    assert err < START_MOMENTS
    assert abs(F2 / F1 - 4.0) < 0.2
    assert abs(iota_vac) < 1e3 * eps() < abs(iota_start)
