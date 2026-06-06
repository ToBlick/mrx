"""Tests for SplineBasis, DerivativeSpline, and TensorBasis.
"""

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from mrx.spline_bases import DerivativeSpline, SplineBasis, TensorBasis

jax.config.update("jax_enable_x64", True)

N, P = 10, 3

# Shared evaluation grid, built once at import time.
_XS = jnp.linspace(0.0, 1.0, 51)


def _eval_all(spl, xs=_XS):
    """Return shape (len(xs), spl.n): entry [k, i] = spl(xs[k], i)."""
    return jax.vmap(lambda x: jax.vmap(lambda i: spl(x, i))(spl.ns))(xs)


# Pre-evaluate the main high-degree bases once; tests share these arrays.
_CLAMPED = SplineBasis(N, P, "clamped")
_PERIODIC = SplineBasis(N, P, "periodic")
_CLAMPED_VALS = _eval_all(_CLAMPED)
_PERIODIC_VALS = _eval_all(_PERIODIC)


# ── Partition of unity ────────────────────────────────────────────────────────

def test_partition_of_unity_clamped():
    npt.assert_allclose(jnp.sum(_CLAMPED_VALS, axis=1), 1.0, atol=1e-12)


def test_partition_of_unity_periodic():
    npt.assert_allclose(jnp.sum(_PERIODIC_VALS, axis=1), 1.0, atol=1e-12)


# ── Positivity ────────────────────────────────────────────────────────────────

def test_positivity_clamped():
    assert jnp.all(_CLAMPED_VALS >= -1e-14)


def test_positivity_periodic():
    assert jnp.all(_PERIODIC_VALS >= -1e-14)


# ── Analytic baselines ────────────────────────────────────────────────────────

def test_degree1_clamped_bernstein():
    """Degree-1 clamped on [0,1] with no interior knot: B_0=1-x, B_1=x."""
    vals = _eval_all(SplineBasis(2, 1, "clamped"))
    npt.assert_allclose(vals[:, 0], 1.0 - _XS, atol=1e-12)
    npt.assert_allclose(vals[:, 1], _XS, atol=1e-12)


def test_degree2_clamped_bernstein():
    """Degree-2 clamped on [0,1] with no interior knot: quadratic Bernstein polynomials."""
    vals = _eval_all(SplineBasis(3, 2, "clamped"))
    npt.assert_allclose(vals[:, 0], (1.0 - _XS) ** 2, atol=1e-12)
    npt.assert_allclose(vals[:, 1], 2.0 * _XS * (1.0 - _XS), atol=1e-12)
    npt.assert_allclose(vals[:, 2], _XS ** 2, atol=1e-12)


def test_derivative_basis_of_bernstein_quadratic():
    """Derivative basis of Bernstein quadratic: D_0=2(1-x), D_1=2x (excluding right endpoint)."""
    dspl = DerivativeSpline(SplineBasis(3, 2, "clamped"))
    xs = _XS[:-1]
    vals = jax.vmap(lambda x: jax.vmap(lambda i: dspl(x, i))(dspl.ns))(xs)
    npt.assert_allclose(vals[:, 0], 2.0 * (1.0 - xs), atol=1e-12)
    npt.assert_allclose(vals[:, 1], 2.0 * xs, atol=1e-12)


# ── Greville collocation & de Rham commutation ────────────────────────────────

@pytest.mark.parametrize("typ", ["clamped", "periodic"])
def test_greville_collocation_recovers_coefficients(typ):
    """Collocation at Greville points is an invertible interpolation."""
    spl = SplineBasis(N, P, typ)
    coll = spl.collocation_matrix()
    coeffs = jnp.linspace(-1.0, 1.0, spl.n)
    npt.assert_allclose(jnp.linalg.solve(coll, coll @ coeffs), coeffs, atol=1e-12)


def test_histopolation_de_rham_clamped():
    """Greville histopolation and finite-difference coboundary commute for clamped splines."""
    spl = SplineBasis(N, P, "clamped")
    dspl = DerivativeSpline(spl)
    coll = spl.collocation_matrix()
    hist = dspl.histopolation_matrix()
    coeffs = jnp.linspace(-0.8, 0.9, spl.n)
    # Integrating the derivative over each Greville span equals the endpoint difference.
    span_integrals = (coll @ coeffs)[1:] - (coll @ coeffs)[:-1]
    npt.assert_allclose(
        jnp.linalg.solve(hist, span_integrals),
        coeffs[1:] - coeffs[:-1],
        atol=1e-12,
    )


@pytest.mark.parametrize("typ", ["clamped", "periodic"])
def test_autodiff_agrees_with_derivative_spline(typ):
    """Autodiff derivative of a spline function equals its expansion in the derivative basis.

    For coefficients ``c``, differentiating ``f(x) = c · B(x)`` by autodiff must
    equal ``dc · D(x)`` where ``D`` are the DerivativeSpline basis functions and
    ``dc`` is the coboundary (finite difference) of ``c``.  This is the pointwise
    statement of the de Rham commutation property.
    """
    spl = _CLAMPED if typ == "clamped" else _PERIODIC
    dspl = DerivativeSpline(spl)
    c = jnp.linspace(-1.0, 1.0, spl.n)

    def f(x):
        return jnp.dot(c, jax.vmap(lambda i: spl(x, i))(spl.ns))

    # Coboundary on coefficient space.
    dc = c[1:] - c[:-1] if typ == "clamped" else jnp.roll(c, -1) - c

    def f_deriv(x):
        return jnp.dot(dc, jax.vmap(lambda j: dspl(x, j))(dspl.ns))

    # Interior points only: derivative splines are not required to be evaluable
    # at the clamped boundary (repeated knots cause a genuine kink there).
    xs = _XS[1:-1]
    npt.assert_allclose(jax.vmap(jax.grad(f))(xs), jax.vmap(f_deriv)(xs), atol=1e-10)


# ── TensorBasis ───────────────────────────────────────────────────────────────

def test_tensor_basis_factors():
    """TensorBasis evaluation equals the product of 1-D factor evaluations."""
    tb = TensorBasis([
        SplineBasis(5, 2, "clamped"),
        SplineBasis(4, 2, "periodic"),
        SplineBasis(6, 2, "clamped"),
    ])
    assert tb.n == 5 * 4 * 6
    x = jnp.array([0.5, 0.5, 0.5])
    for lin in (0, 7, 23, tb.n - 1):
        i, j, k = jnp.unravel_index(lin, tb.shape)
        npt.assert_allclose(
            tb.evaluate(x, lin),
            tb.bases[0](x[0], i) * tb.bases[1](x[1], j) * tb.bases[2](x[2], k),
            atol=1e-12,
        )


