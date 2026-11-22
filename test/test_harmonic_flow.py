"""The harmonic flow as an unknown of the Newton step.

The incompressible flows tangent to the wall are ``curl a + c h``, with ``h`` the harmonic 2-form of the domain. The
curls are ``M``-orthogonal to ``h``, so a step ``curl a`` cannot move the field along ``h``, and the derivative of the
energy along it, ``h . load(J x Pi_1 B)``, is the harmonic part of the force. With stellarator symmetry ``h`` is odd and
the force even, so that part vanishes, and the even flows of a half-period sequence hold no ``h``. Without the symmetry
the Newton step has the amplitude ``c`` as one more unknown (:func:`mrx.relaxation.newton.target_direction`), and the
bordered step zeroes the harmonic row of the Gauss-Newton model.

Both statements run on the fixture's ``period`` representation: li383 in one field period without the reflection,
once with its own symmetric field and once with the symmetry broken by a smooth curl.
"""
import jax
import jax.numpy as jnp

from mrx.precision import eps
from mrx.relaxation.initial_conditions import potential_two_form
from mrx.relaxation.loop import TimeStepper
from mrx.relaxation.newton import (NEWTON_PENALTY, atom_profiles, parallel_penalty_profile, second_variation,
                                   target_direction)

#: The tolerance of the bordered Newton solve.
TIGHT = 1e-4
#: The size of the symmetry-breaking curl relative to the field.
BREAK = 1e-2
#: The harmonic part of the force of the broken field relative to the force: PENDING measured.
HARMONIC_SHARE = 1e-4


def _harmonic_part(ts, y):
    """``|h . y| / |h|_M`` of the dual 2-form ``y``: the norm of its harmonic part."""
    return float(jnp.linalg.norm(ts.harmonic_flows @ y / jnp.sqrt(ts.harmonic_norm_sq)))


def _force(ts, B):
    """``(load, |F|_M)``: the Lorentz force of ``B`` as a dual 2-form and the mass norm of its projection."""
    _, load = ts._lorentz(B, None)
    F, MF, _ = ts._force(load, jnp.zeros(ts.seq.even.n(1)))
    return load, float(jnp.sqrt(F @ MF))


def test_no_harmonic_force_with_stellarator_symmetry(domains):
    """The half-period flows hold no harmonic form, so the step has no harmonic unknown. On ``period`` the harmonic
    part of the force of the stellarator-symmetric field is 1.9e-12 of the force in float64 (0 in float32,
    2026-11-19), below ten solve tolerances: one force evaluation, no Newton solve."""
    assert TimeStepper(seq=domains("half"), newton=True).harmonic_flows is None
    seq = domains("period")
    ts = TimeStepper(seq=seq, newton=True)
    load, F_norm = _force(ts, potential_two_form(seq)[0])
    F_h = _harmonic_part(ts, load)
    print(f"\n  F_h {F_h:.3e} of |F| {F_norm:.4e}")
    assert F_h < 10 * max(seq.tol, eps()) * F_norm


def test_bordered_step_removes_the_harmonic_force(domains):
    """The li383 field on ``period`` plus the curl of ``A = r (1 - r^2) (cos + sin)(2 pi theta) dzeta``, smooth at the axis and zero on the wall, scaled to
    :data:`BREAK` of the field, which breaks the stellarator symmetry: its force has a harmonic part (above
    :data:`HARMONIC_SHARE` of the force), and the harmonic row of the Gauss-Newton model, ``h . (load - H u)``, after
    the bordered step solved to :data:`TIGHT` is at most 30 times that tolerance relative to the harmonic part."""
    seq = domains("period")
    ts = TimeStepper(seq=seq, newton=True)
    B0 = potential_two_form(seq)[0]

    def A(x):
        t = 2 * jnp.pi * x[1]
        return jnp.array([0.0, 0.0, x[0] * (1.0 - x[0] ** 2) * (jnp.cos(t) + jnp.sin(t))])
    curl = seq.odd.G[1] @ seq.odd.interpolate(jax.jit(A), 1)
    B = B0 + BREAK * curl * seq.odd.l2_norm(B0, 2) / seq.odd.l2_norm(curl, 2)
    load, F_norm = _force(ts, B)
    H = second_variation(seq, B, None, parallel_penalty_profile(atom_profiles(seq, B), NEWTON_PENALTY))
    u, _, _, c_h, _, _, _ = target_direction(seq, B, load, jnp.zeros(seq.even.n(1)), ts.harmonic_flows, tol=TIGHT)
    F_h, after = _harmonic_part(ts, load), _harmonic_part(ts, load - H(u))
    share = float(jnp.linalg.norm(c_h * jnp.sqrt(ts.harmonic_norm_sq)) / seq.even.l2_norm(u, 2))
    print(f"\n  F_h {F_h:.4e} of |F| {F_norm:.4e} ({F_h / F_norm:.2e}), harmonic row after the bordered step "
          f"{after:.4e} ({after / F_h / TIGHT:.1f} TIGHT), |c h| / |u| {share:.3e}")
    assert F_h > HARMONIC_SHARE * F_norm
    assert after < 30 * TIGHT * F_h
