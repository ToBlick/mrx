"""``mrx.initial_conditions`` without data files (ci tier).

The analytic-profile 2-form ``B_hat = (0, Phi'(iota - lam_z), Phi'(1 + lam_c))``
is divergence-free for any ``lambda`` because the mixed partials cancel.
The production route (L2 projection through ``M_2``, then ``leray_clean``)
carries a small discrete divergence before the cleaning -- the projection
does not commute with ``d`` -- which is measured and bounded, and none
after it. The helicity closed form of ``analytic_helicity`` is checked
against a quadrature of eq. (1) it claims to integrate, and the production
helicity diagnostic is pinned on the same field.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import mrx
from mrx.initial_conditions import (
    analytic_helicity,
    divergence_norm,
    leray_clean,
    analytic_profile_form,
    make_lambda,
    make_profiles,
    project_reference_two_form,
)
from mrx.relaxation import compute_helicity

# Production defaults of scripts/relax.py: iota 0.4 -> 0.9 as rho^2, uniform
# toroidal field (flux_exp = 1); one lambda mode so its cancellation is
# exercised (the default run has none).
IOTA0, IOTA1, IOTA_EXP, FLUX_EXP = 0.4, 0.9, 2.0, 1.0
LAMBDA_MODES = [(1, 1, 0.05)]


def analytic_omega(iota0=IOTA0, iota1=IOTA1, modes=LAMBDA_MODES):
    iota, dPhi = make_profiles(iota0, iota1, IOTA_EXP, FLUX_EXP)
    return analytic_profile_form(iota, dPhi, make_lambda(modes))


def test_analytic_ic_projects_and_leray_cleans(tiny_seq):
    """The production route: the L2 projection through ``M_2`` reintroduces a
    small divergence -- measured 2026-08-26 on tiny_seq (see the print):
    ``||div B|| / ||B||_M = 1.13e-3`` before cleaning, 4.0e-11 after, the
    field moved by 3.1e-4 -- and ``leray_clean`` takes it to solver
    tolerance. The bands are ~2x the measured values (3x on the moved
    norm, which a float32 solve at tolerance 3.5e-4 resolves less well)."""
    seq = tiny_seq
    B_raw, norm = project_reference_two_form(seq, analytic_omega())
    assert norm > 0
    div_raw = divergence_norm(seq, B_raw)
    B, moved = leray_clean(seq, B_raw)
    div = divergence_norm(seq, B)
    print(f"\n  projected analytic IC: ||div B|| {div_raw:.2e} -> {div:.2e}, "
          f"moved {moved:.2e}, ||B||_M raw {norm:.4f}")
    assert abs(float(seq.l2_norm(B, 2)) - 1.0) <= 100 * mrx.eps()
    assert div <= 10 * seq.tol
    assert div_raw / norm < 2.5e-3
    assert moved < 1e-3


@pytest.mark.parametrize("params", [(0.4, 0.9, 2.0, 1.0), (0.3, 1.2, 1.0, 2.0),
                                    (0.5, 0.5, 2.0, 1.0), (1.0, 0.2, 3.0, 1.0)])
def test_analytic_helicity_closes_eq1(params):
    """``analytic_helicity`` equals ``int_0^1 Phi^2 (X / Phi)' drho`` with
    ``Phi = int dPhi``, ``X' = iota Phi'`` -- by Gauss quadrature, exact for
    these polynomial profiles (numpy float64, independent of MRX_DTYPE)."""
    iota0, iota1, e, q = params
    x, w = np.polynomial.legendre.leggauss(64)
    r = 0.5 * (x + 1.0)
    w = 0.5 * w
    Phi = r ** (q + 1) / (q + 1)
    # X = int_0^r iota Phi' = iota0 Phi + (iota1 - iota0) r^(q+e+1) / (q+e+1)
    X = iota0 * Phi + (iota1 - iota0) * r ** (q + e + 1) / (q + e + 1)
    dPhi = r ** q
    dX = (iota0 + (iota1 - iota0) * r ** e) * dPhi
    integrand = Phi ** 2 * (dX * Phi - X * dPhi) / Phi ** 2
    H_quad = float(np.sum(w * integrand))
    H_closed = analytic_helicity(iota0, iota1, e, q)
    assert abs(H_quad - H_closed) <= 1e-13 * max(abs(H_closed), 1.0)
    if iota0 == iota1:
        assert H_closed == 0.0


# compute_helicity on the projected analytic IC, measured 2026-08-26 on
# tiny_seq in float64: flat iota 0.6 -> +3.2069e-2, sheared 0.4 -> 0.9 ->
# +3.8762e-2 (eq. (1) gives 0 and +4.4046e-3 for the same fields).
HELICITY_MEASURED = {"flat": 3.2069e-2, "sheared": 3.8762e-2}


def test_computed_helicity_is_pinned(tiny_seq):
    """``compute_helicity`` on the projected analytic IC.

    On a torus the diagnostic and eq. (1) are gauge-related by the harmonic
    1-form (``A`` is solved in the Dirichlet space, the natural gauge
    ``Phi dchi - X dzeta`` is not in it), so they do not agree in value:
    the diagnostic carries a harmonic term of ~3.2e-2 at either shear. What
    the test pins is the diagnostic itself, to 5% of the measured values --
    the 2026-08-25 primal/dual rhs bug moved it by 85x.
    """
    seq = tiny_seq
    A0 = jnp.zeros(seq.n(1, True))
    for tag, (i0, i1) in {"flat": (0.6, 0.6), "sheared": (IOTA0, IOTA1)}.items():
        B, norm = project_reference_two_form(seq, analytic_omega(i0, i1, modes=[]))
        H, _ = compute_helicity(B, seq, A0)
        H_eq1 = analytic_helicity(i0, i1, IOTA_EXP, FLUX_EXP) / norm ** 2
        print(f"\n  {tag}: computed H {float(H):+.4e}, eq.(1) H {H_eq1:+.4e}")
        assert abs(float(H) - HELICITY_MEASURED[tag]) <= 0.05 * HELICITY_MEASURED[tag]
