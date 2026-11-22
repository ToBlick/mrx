"""The half-period quadrature of a stellarator-symmetric sequence gives every production functional of a symmetric
state as the full period does.

A half-period sequence integrates over ``zeta`` in ``[0, 1/2]`` with doubled weights, and its fields live on the
parity views ``seq.odd`` (``B``, ``A``, ``J``, ``E``, ``H``) and ``seq.even`` (``u``, ``F``, ``p``). That is exact only for
integrands even under the reflection ``(theta, zeta) -> (-theta, -zeta)``. An odd integrand, or a field on the wrong
view, gives a wrong number silently. So every functional the relaxation, its controls and its diagnostics evaluate is
computed for the projected exact field of shearedE on the tiny mesh twice, with stellarator symmetry (the shared tiny
sequence) and over the whole field period without it, and the two must agree to the tolerance of the solves inside
them. An odd integrand is the negative control: it integrates to zero over the period and not over the half.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from mrx.flux_label import ANISOTROPY, label_spline
from mrx.precision import eps
from mrx.relaxation import mesh_motion as mm
from mrx.relaxation.initial_conditions import potential_two_form, toroidal_flux
from mrx.relaxation.loop import TimeStepper, initial_state
from mrx.relaxation.newton import NEWTON_MASS_SOLVE_TOL, NEWTON_PENALTY
from mrx.relaxation.physics import compute_force, compute_helicity, weak_pressure, beta_vol
from mrx.relaxation.seeding import parallel_seed
from mrx.relaxation.shells import build_shells, label
from mrx.relaxation.twist import build_slips, current_on_label, iota_residual, slip_transform
from test.conftest import TINY, energy_derivatives

#: The cells of the profile space on the tiny mesh.
CELLS = 3
#: The tolerance of the label solve here.
LABEL_TOL = 1e-12


@pytest.fixture(scope="module")
def both(sheared):
    """``(half, full)``: the tiny shearedE sequence with stellarator symmetry and over the whole period."""
    from mrx.geometry import build_sequence
    from mrx.nullspace import compute_nullspaces
    full = build_sequence(sheared[1], TINY, 2, symmetry="field-period", graded=True)
    compute_nullspaces(full, verbose=False)
    return sheared[3], full


def _functionals(eq, seq):
    """Every functional of the projected exact field ``B`` on ``seq``, by name."""
    odd, even = seq.odd, seq.even
    B = potential_two_form(seq)[0]
    out = dict(energy=0.5 * float(odd.l2_norm_sq(B, 2)))
    F, _, J, _ = compute_force(B, seq)
    out["force"] = float(even.l2_norm(F, 2))
    out["resid"] = float(initial_state(B, TimeStepper(seq=seq, newton=True)).best.resid)
    out["first_variation"] = energy_derivatives(seq, B, F / even.l2_norm(F, 2))[0]
    out["helicity"] = float(compute_helicity(B, seq, jnp.zeros(odd.n(1)))[0])
    out["beta"] = float(beta_vol(B, weak_pressure(J, B, seq), seq))
    s_q, T, _ = label(seq, B, ANISOTROPY, None, LABEL_TOL)
    a, b, e = label_spline(seq, T, odd.evaluate_at_quadrature(B, 2))
    out["label_bounds"] = np.array([a, b])
    out["label_moments"] = np.asarray(e)
    mu = current_on_label(seq, B, T, CELLS)
    out["current_moments"] = np.asarray(mu)
    slips = build_slips(seq, B, T, mu)[0]
    residual = iota_residual(seq, B, s_q, eq.iota, mu.size)
    out["iota_residual"] = np.asarray(residual(B)[0])
    out["iota_jacobian"] = slip_transform(residual, B, slips.maps(seq)[0], mu.size)
    shells, _ = build_shells(seq, B, s_q, jnp.asarray(eq.mu0_p_of_s(np.asarray(s_q))), NEWTON_PENALTY,
                             NEWTON_MASS_SOLVE_TOL, CELLS)
    out["shell_means"] = np.asarray(shells.g_p)
    c, _ = mm.magnetic_axis(seq, B)
    out["axis_moments"] = np.asarray(c)
    out["axis_offset"] = np.asarray(mm.axis_offset(seq, c)[0])
    d = np.zeros((seq.basis_0.Lambda[2].n, 2))
    d[:, 0] = 2e-3
    B1, _ = mm.transfer(seq, B, mm.moved(seq.map, mm.tangent(seq, mm.axis_step(seq, d))))
    out["transfer_energy"] = 0.5 * float(odd.l2_norm_sq(B1, 2))
    out["transfer_flux"] = float(toroidal_flux(seq, B1))
    r = seq.quad.x[:, 0]
    dB = parallel_seed(seq, B, 3, 2, r ** 2 * (1.0 - r) ** 2)
    out["seed_model"] = np.array([float(B @ (odd.M[2] @ dB)), float(dB @ (odd.M[2] @ dB))])
    out["seed_energy"] = 0.5 * float(odd.l2_norm_sq(B + dB, 2))
    # the negative control: |B|^2 sin(2 pi (theta - zeta)) is odd under the reflection, its integral over the period
    # is zero
    Bq = odd.evaluate_at_quadrature(B, 2)
    x = seq.quad.x
    weight = seq.quad.w * jnp.sum(Bq * jnp.einsum("qij,qj->qi", seq.metric_jkl, Bq), axis=1) / seq.jacobian_j
    out["odd_control"] = float(jnp.sum(weight * jnp.sin(2 * jnp.pi * (x[:, 1] - x[:, 2]))) / jnp.sum(weight))
    return out


def test_the_half_period_gives_every_functional(both, sheared):
    """Every functional of :func:`_functionals` on the two sequences agrees to ``1e3 max(tol, eps)`` relative to its
    size, the solves' tolerance class. The odd control is zero over the period to round-off and more than 1e-3 over the
    half."""
    half, full = both
    a, b = _functionals(sheared[0], half), _functionals(sheared[0], full)
    bound = 1e3 * max(half.tol, eps())
    bad = []
    for k in a:
        if k == "odd_control":
            continue
        x, y = np.asarray(a[k], dtype=float), np.asarray(b[k], dtype=float)
        rel = float(np.max(np.abs(x - y)) / max(np.max(np.abs(y)), np.finfo(float).tiny))
        print(f"\n  {k}: {rel:.1e}", end="")
        if not rel < bound:
            bad.append(f"{k} {rel:.1e}")
    print(f"\n  odd control: half {a['odd_control']:+.3e}, full {b['odd_control']:+.3e}")
    assert not bad, "; ".join(bad)
    assert abs(b["odd_control"]) < 1e3 * eps() and abs(a["odd_control"]) > 1e-3
