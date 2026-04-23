"""Low-level quadrature tests.

These don't touch the DeRham sequence and run in milliseconds.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from mrx.quadrature import composite_quad, spectral_quad, trapezoidal_quad

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("n", [1, 2, 5, 10, 20])
def test_trapezoidal_weight_sum(n):
    _, w = trapezoidal_quad(n)
    npt.assert_allclose(jnp.sum(w), 1.0, atol=1e-14)


@pytest.mark.parametrize("p", [1, 2, 3, 5, 10])
def test_spectral_weight_sum(p):
    _, w = spectral_quad(p)
    npt.assert_allclose(jnp.sum(w), 1.0, atol=1e-14)


@pytest.mark.parametrize("p", [1, 2, 3, 5])
@pytest.mark.parametrize("n_intervals", [2, 3, 5, 10])
def test_composite_weight_sum(p, n_intervals):
    T = jnp.linspace(0.0, 1.0, n_intervals + 1)
    _, w = composite_quad(T, p)
    npt.assert_allclose(jnp.sum(w), 1.0, atol=1e-14)


@pytest.mark.parametrize("p", [1, 2, 3, 4, 5])
def test_spectral_exact_for_polynomials(p):
    """A Gauss rule with p points integrates polynomials of degree <= 2p-1 exactly."""
    x, w = spectral_quad(p)
    for deg in range(2 * p):
        # True integral of x^deg on [0, 1] is 1 / (deg + 1).
        val = jnp.sum(w * x ** deg)
        npt.assert_allclose(val, 1.0 / (deg + 1), atol=1e-6)


@pytest.mark.parametrize("n_intervals", [2, 4, 8])
def test_composite_refinement_convergence(n_intervals):
    """Composite trapezoidal rule converges for a smooth integrand."""
    T = jnp.linspace(0.0, 1.0, n_intervals + 1)
    x, w = composite_quad(T, 1)
    f = jnp.cos(2 * jnp.pi * x)
    val = jnp.sum(w * f)
    # ∫₀¹ cos(2πx) dx = 0
    assert jnp.abs(val) < 1.0 / n_intervals ** 2
