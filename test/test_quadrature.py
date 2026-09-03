"""Tests for quadrature rules.
"""

import jax.numpy as jnp
import numpy.testing as npt
import pytest

import mrx
from mrx.quadrature import composite_quad, spectral_quad

# Exactness is checked in the working dtype: the nodes come from a float64
# host leggauss, the sums run in mrx.DTYPE. 100 eps is 2.2e-14 in float64
# and 1.2e-5 in float32.
EXACT = mrx.eps(100)

# ── Polynomial exactness ──────────────────────────────────────────────────────

@pytest.mark.parametrize("p", [1, 2, 3, 5, 8])
def test_spectral_exact_for_polynomials(p):
    """A p-point Gauss rule integrates x^k on [0,1] exactly for k <= 2p-1."""
    x, w = spectral_quad(p)
    for deg in range(2 * p):
        npt.assert_allclose(jnp.sum(w * x ** deg), 1.0 / (deg + 1), atol=EXACT)


@pytest.mark.parametrize("p", [1, 2, 3, 5])
@pytest.mark.parametrize("n_intervals", [2, 5])
def test_composite_exact_for_polynomials(p, n_intervals):
    """Composite p-point Gauss rule integrates polynomials of degree <= 2p-1 exactly."""
    T = jnp.linspace(0.0, 1.0, n_intervals + 1)
    x, w = composite_quad(T, p)
    for deg in range(2 * p):
        npt.assert_allclose(jnp.sum(w * x ** deg), 1.0 / (deg + 1), atol=EXACT)


# ── Convergence order ─────────────────────────────────────────────────────────


# ── Non-uniform knots ─────────────────────────────────────────────────────────


# ── Node placement ────────────────────────────────────────────────────────────

