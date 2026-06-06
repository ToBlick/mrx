"""Tests for quadrature rules.
"""

import numpy as np
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from mrx.quadrature import composite_quad, spectral_quad

# ── Polynomial exactness ──────────────────────────────────────────────────────

@pytest.mark.parametrize("p", [1, 2, 3, 5, 8])
def test_spectral_exact_for_polynomials(p):
    """A p-point Gauss rule integrates x^k on [0,1] exactly for k <= 2p-1."""
    x, w = spectral_quad(p)
    for deg in range(2 * p):
        npt.assert_allclose(jnp.sum(w * x ** deg), 1.0 / (deg + 1), atol=1e-12)


@pytest.mark.parametrize("p", [1, 2, 3, 5])
@pytest.mark.parametrize("n_intervals", [2, 5])
def test_composite_exact_for_polynomials(p, n_intervals):
    """Composite p-point Gauss rule integrates polynomials of degree <= 2p-1 exactly."""
    T = jnp.linspace(0.0, 1.0, n_intervals + 1)
    x, w = composite_quad(T, p)
    for deg in range(2 * p):
        npt.assert_allclose(jnp.sum(w * x ** deg), 1.0 / (deg + 1), atol=1e-12)


# ── Convergence order ─────────────────────────────────────────────────────────

def test_composite_convergence_order():
    """Composite Gauss rule converges at order 2p in the number of intervals.

    Integrand: cos(x) on [0, 1].  Exact value: sin(1).
    For fixed p, halving the interval size should reduce the error by ~2^(2p).
    """
    exact = np.sin(1.0)
    for p in [1, 2, 3]:
        errors = []
        for n in [4, 8, 16]:
            T = jnp.linspace(0.0, 1.0, n + 1)
            x, w = composite_quad(T, p)
            errors.append(abs(float(jnp.sum(w * jnp.cos(x))) - exact))
        # Check that each refinement reduces error by at least factor 2^(2p-1)
        # (conservative: true rate is 2^(2p) but roundoff limits the last digit).
        rate_threshold = 2 ** (2 * p - 1)
        assert errors[0] / errors[1] > rate_threshold, (
            f"p={p}: ratio {errors[0]/errors[1]:.1f} < expected {rate_threshold}"
        )
        assert errors[1] / errors[2] > rate_threshold, (
            f"p={p}: ratio {errors[1]/errors[2]:.1f} < expected {rate_threshold}"
        )


# ── Non-uniform knots ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("p", [1, 2, 3])
def test_composite_non_uniform_knots(p):
    """Polynomial exactness holds on a non-uniform knot vector.

    Uses T = [0, 0.2, 0.65, 1.0] to exercise per-interval rescaling.
    """
    T = jnp.array([0.0, 0.2, 0.65, 1.0])
    x, w = composite_quad(T, p)
    for deg in range(2 * p):
        npt.assert_allclose(jnp.sum(w * x ** deg), 1.0 / (deg + 1), atol=1e-12)


# ── Node placement ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("p", [1, 2, 3, 5])
def test_composite_nodes_in_interval_interiors(p):
    """Every quadrature node lies strictly inside its sub-interval.

    Nodes at breakpoints cause evaluation problems at clamped endpoints.
    """
    T = jnp.array([0.0, 0.2, 0.65, 1.0])
    x, _ = composite_quad(T, p)
    breakpoints = np.array(T)
    x_np = np.array(x)
    for i in range(len(breakpoints) - 1):
        a, b = breakpoints[i], breakpoints[i + 1]
        nodes_in = x_np[(x_np > a - 1e-14) & (x_np < b + 1e-14)]
        assert len(nodes_in) == p, f"interval [{a},{b}]: expected {p} nodes, got {len(nodes_in)}"
        assert np.all(nodes_in > a), f"node(s) at or below left breakpoint {a}"
        assert np.all(nodes_in < b), f"node(s) at or above right breakpoint {b}"
