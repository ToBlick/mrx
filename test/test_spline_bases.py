"""Tests for SplineBasis, DerivativeSpline, and TensorBasis.
"""

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

import mrx
from mrx.spline_bases import DerivativeSpline, SplineBasis, TensorBasis

# Pointwise identities (partition of unity, Bernstein baselines, the tensor
# factorisation) hold to a few ulp: 100 eps = 2.2e-14 f64 / 1.2e-5 f32.
# Identities that go through a collocation or histopolation SOLVE, and the
# autodiff-vs-derivative-basis comparison, carry the condition number of a
# 10x10 banded system: 1e3 eps = 2.2e-13 f64 / 1.2e-4 f32.
POINTWISE = mrx.eps(100)
SOLVED = mrx.eps(1e3)


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
    npt.assert_allclose(jnp.sum(_CLAMPED_VALS, axis=1), 1.0, atol=POINTWISE)


def test_partition_of_unity_periodic():
    npt.assert_allclose(jnp.sum(_PERIODIC_VALS, axis=1), 1.0, atol=POINTWISE)


# ── Positivity ────────────────────────────────────────────────────────────────

def test_positivity_clamped():
    assert jnp.all(_CLAMPED_VALS >= -POINTWISE)


def test_positivity_periodic():
    assert jnp.all(_PERIODIC_VALS >= -POINTWISE)


# ── Analytic baselines ────────────────────────────────────────────────────────

def test_degree1_clamped_bernstein():
    """Degree-1 clamped on [0,1] with no interior knot: B_0=1-x, B_1=x."""
    vals = _eval_all(SplineBasis(2, 1, "clamped"))
    npt.assert_allclose(vals[:, 0], 1.0 - _XS, atol=POINTWISE)
    npt.assert_allclose(vals[:, 1], _XS, atol=POINTWISE)


def test_degree2_clamped_bernstein():
    """Degree-2 clamped on [0,1] with no interior knot: quadratic Bernstein polynomials."""
    vals = _eval_all(SplineBasis(3, 2, "clamped"))
    npt.assert_allclose(vals[:, 0], (1.0 - _XS) ** 2, atol=POINTWISE)
    npt.assert_allclose(vals[:, 1], 2.0 * _XS * (1.0 - _XS), atol=POINTWISE)
    npt.assert_allclose(vals[:, 2], _XS ** 2, atol=POINTWISE)


def test_derivative_basis_of_bernstein_quadratic():
    """Derivative basis of Bernstein quadratic: D_0=2(1-x), D_1=2x (excluding right endpoint)."""
    dspl = DerivativeSpline(SplineBasis(3, 2, "clamped"))
    xs = _XS[:-1]
    vals = jax.vmap(lambda x: jax.vmap(lambda i: dspl(x, i))(dspl.ns))(xs)
    npt.assert_allclose(vals[:, 0], 2.0 * (1.0 - xs), atol=POINTWISE)
    npt.assert_allclose(vals[:, 1], 2.0 * xs, atol=POINTWISE)


# ── Greville collocation & de Rham commutation ────────────────────────────────

@pytest.mark.parametrize("typ", ["clamped", "periodic"])
def test_greville_collocation_recovers_coefficients(typ):
    """Collocation at Greville points is an invertible interpolation."""
    spl = SplineBasis(N, P, typ)
    coll = spl.collocation_matrix()
    coeffs = jnp.linspace(-1.0, 1.0, spl.n)
    npt.assert_allclose(jnp.linalg.solve(coll, coll @ coeffs), coeffs, atol=SOLVED)


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
        atol=SOLVED,
    )


@pytest.mark.parametrize("p", [2, 3, 4])
def test_histopolation_de_rham_periodic(p):
    """Greville histopolation and the periodic coboundary commute, at BOTH parities.

    The parity matters: periodic Greville points sit ON knots for odd p and
    HALFWAY between knots for even p, so at even p the last sorted span is
    ``[1 - h/2, 1 + h/2]`` and crosses the period seam.  ``SplineBasis.evaluate``
    does not extend periodically past ``x = 1`` (the image of basis function
    ``p - 1`` is missing from the extended knot vector), so the matrix has to
    wrap its quadrature points.  Before it did, the seam row was wrong at even p
    and ``interpolate`` was not a projector at k >= 1.  The right-hand side
    below evaluates the spline itself at WRAPPED span endpoints, so it is the
    periodic ground truth independent of any quadrature.
    """
    spl = SplineBasis(N, p, "periodic")
    dspl = DerivativeSpline(spl)
    hist = dspl.histopolation_matrix()
    spans = dspl.greville_spans()
    coeffs = jnp.cos(2 * jnp.pi * jnp.arange(spl.n) / spl.n) + 0.3
    # d/dx sum_i c_i B_i = sum_j (c_{j+1} - c_j) D_j  (periodic indexing)
    d_coeffs = jnp.roll(coeffs, -1) - coeffs

    def s(x):
        return jnp.sum(jax.vmap(lambda i: spl(jnp.mod(x, 1.0), i))(spl.ns) * coeffs)

    span_integrals = jax.vmap(lambda ab: s(ab[1]) - s(ab[0]))(spans)
    npt.assert_allclose(hist @ d_coeffs, span_integrals, atol=SOLVED)
    npt.assert_allclose(jnp.linalg.solve(hist, span_integrals), d_coeffs, atol=SOLVED)


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
    npt.assert_allclose(jax.vmap(jax.grad(f))(xs), jax.vmap(f_deriv)(xs), atol=SOLVED)


# ── TensorBasis ───────────────────────────────────────────────────────────────

def test_tensor_basis_factors():
    """``contract`` of a one-hot coefficient tensor equals the product of the
    1-D factor evaluations -- the tensor basis function, without a
    per-function evaluator (deleted 2026-08-28; ``evaluate_local``/``contract``
    are the only evaluators)."""
    tb = TensorBasis([
        SplineBasis(5, 2, "clamped"),
        SplineBasis(4, 2, "periodic"),
        SplineBasis(6, 2, "clamped"),
    ])
    assert tb.n == 5 * 4 * 6
    x = jnp.array([0.5, 0.5, 0.5])
    for lin in (0, 7, 23, tb.n - 1):
        i, j, k = jnp.unravel_index(lin, tb.shape)
        one_hot = jnp.zeros(tb.shape).at[i, j, k].set(1.0)
        npt.assert_allclose(
            tb.contract(one_hot, x),
            tb.bases[0](x[0], i) * tb.bases[1](x[1], j) * tb.bases[2](x[2], k),
            atol=POINTWISE,
        )


