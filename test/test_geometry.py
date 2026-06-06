"""Tests for mrx.geometry.

Two groups of fixtures:
- Pure-JAX (_QUAD_X, _C_RAW, etc.): no DeRhamSequence needed.
- Seq-based (_SEQ): (4,8,4) / p=3 / q=6, clamped-periodic-periodic.
  - ``_F_TOROID``: toroid map used for spline geometry tests.
  - ``_F_ANALYTIC``: rotating_ellipse_map (nfp=3) used for R/Z tests.
    Note: Cartesian X, Y of this map are NOT periodic in ζ for nfp>1
    (one field period spans only [0, 1/nfp] in ζ), so it must not be
    Greville-interpolated in Cartesian form.
"""

import numpy as np
import jax
import jax.numpy as jnp
import numpy.testing as npt

jax.config.update("jax_enable_x64", True)

from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction
from mrx.geometry import (
    _coeffs_to_raw_grid,
    _tp_evaluate,
    compute_geometry_terms,
)
from mrx.mappings import rotating_ellipse_map, toroid_map
from mrx.geometry import greville_interpolate_map

# ── Pure-JAX module-level fixtures ───────────────────────────────────────────

_rng = np.random.default_rng(0)
_N_Q = 20
_QUAD_X = jnp.asarray(_rng.uniform(size=(_N_Q, 3)))

_NR, _NT, _NZ = 4, 5, 3
_NQR, _NQT, _NQZ = 6, 7, 8
_C_RAW = jnp.asarray(_rng.standard_normal((3, _NR, _NT, _NZ)))
_M1 = jnp.asarray(_rng.standard_normal((_NR, _NQR)))
_M2 = jnp.asarray(_rng.standard_normal((_NT, _NQT)))
_M3 = jnp.asarray(_rng.standard_normal((_NZ, _NQZ)))

# ── Seq-based module-level fixtures ──────────────────────────────────────────

_TYPES = ("clamped", "periodic", "periodic")

_SEQ = DeRhamSequence(
    (4, 8, 4), (3, 3, 3), 6, _TYPES,
    polar=False,
)
_SEQ.evaluate_1d()

_F_TOROID = toroid_map(epsilon=0.3, kappa=1.2)
_COEFFS = greville_interpolate_map(_F_TOROID, _SEQ)
_SEQ.set_spline_map(_COEFFS)

_SPLINE_MAP = _SEQ.build_spline_map(_COEFFS)

_F_ANALYTIC = rotating_ellipse_map(eps=0.3, kappa=1.2, nfp=3)


# ── _tp_evaluate: sum-factorization identity ──────────────────────────────────

def test_tp_evaluate_matches_brute_force():
    """Sum-factorized result equals the full four-index einsum."""
    expected = jnp.einsum("iabc,aI,bJ,cK->iIJK", _C_RAW, _M1, _M2, _M3)
    got = _tp_evaluate(_C_RAW, _M1, _M2, _M3)
    npt.assert_allclose(got, expected, atol=1e-12)


# ── _coeffs_to_raw_grid: identity extraction ─────────────────────────────────

def test_coeffs_to_raw_grid_identity_extraction():
    """With an identity extraction operator the output is a reshape of the input."""
    n_dof = _NR * _NT * _NZ
    coefficients = jnp.asarray(_rng.standard_normal((3, n_dof)))
    extraction_T = jnp.eye(n_dof)
    result = _coeffs_to_raw_grid(coefficients, extraction_T, _NR, _NT, _NZ)
    npt.assert_allclose(result, coefficients.reshape(3, _NR, _NT, _NZ), atol=1e-14)


# ── compute_geometry_terms: known map ────-───────────────────────────────────

def test_geometry_diagonal_scaling():
    """Diagonal scaling F(x)=(2 x1,3 x2,5 x3) gives metric=diag(4,9,25), det=30."""
    scale = jnp.array([2.0, 3.0, 5.0])

    def F(x):
        return scale * x

    metric, metric_inv, jac = compute_geometry_terms(F, _QUAD_X)

    expected_metric = jnp.diag(scale ** 2)
    npt.assert_allclose(metric, jnp.broadcast_to(expected_metric, (_N_Q, 3, 3)), atol=1e-12)
    eye_check = jnp.einsum("qij,qjk->qik", metric_inv, metric)
    npt.assert_allclose(eye_check, jnp.broadcast_to(jnp.eye(3), (_N_Q, 3, 3)), atol=1e-11)
    npt.assert_allclose(jac, jnp.full(_N_Q, 30.0), atol=1e-11)


# ── Spline-map geometry: two paths agree ─────────────────────────────────────

def test_spline_geometry_matches_jacfwd():
    """Sum-factorized spline metric agrees with jacfwd on the same SplineMap."""
    metric_ref, _, _ = compute_geometry_terms(_SPLINE_MAP, _SEQ.quad.x)
    npt.assert_allclose(_SEQ.metric_jkl, metric_ref, atol=1e-8)


def test_spline_jacobian_positive():
    """Jacobian determinant is positive at every quadrature point."""
    assert jnp.all(_SEQ.jacobian_j > 0), (
        f"Negative Jacobian at {jnp.sum(_SEQ.jacobian_j <= 0)} quadrature point(s)"
    )


def test_spline_metric_symmetric():
    """Metric tensor g_ij = g_ji at every quadrature point."""
    g = _SEQ.metric_jkl
    npt.assert_allclose(g, g.transpose(0, 2, 1), atol=1e-10)


def test_spline_map_approximates_analytic():
    """Greville-interpolated spline map is close to the analytic map pointwise."""
    test_pts = jnp.array([
        [r, t, z]
        for r in [0.2, 0.5, 0.8]
        for t in [0.1, 0.4, 0.75]
        for z in [0.15, 0.5, 0.85]
    ])
    for x in test_pts:
        npt.assert_allclose(_SPLINE_MAP(x), _F_TOROID(x), atol=0.05, rtol=0.05,
                            err_msg=f"map mismatch at x={x}")


def test_spline_jacobian_close_to_analytic():
    """Spline-path Jacobian determinant is within 25% of the analytic value."""
    _, _, jac_analytic = compute_geometry_terms(_F_TOROID, _SEQ.quad.x)
    npt.assert_allclose(_SEQ.jacobian_j, jac_analytic, rtol=0.25)


def test_greville_interpolation_R_Z():
    """R and Z of rotating_ellipse_map(nfp=3) are approximated accurately.

    R and Z are periodic in both angular directions for any nfp.
    """
    def R_fn(x):
        F = _F_ANALYTIC(x)
        return jnp.sqrt(F[0] ** 2 + F[1] ** 2)

    def Z_fn(x):
        return _F_ANALYTIC(x)[2]

    R_dof = _SEQ.interpolate(R_fn, 0)
    Z_dof = _SEQ.interpolate(Z_fn, 0)
    R_h = DiscreteFunction(R_dof, _SEQ.basis_0, _SEQ.e0)
    Z_h = DiscreteFunction(Z_dof, _SEQ.basis_0, _SEQ.e0)

    test_pts = jnp.array([
        [r, t, z]
        for r in [0.2, 0.5, 0.8]
        for t in [0.1, 0.4, 0.75]
        for z in [0.15, 0.5, 0.85]
    ])
    for x in test_pts:
        npt.assert_allclose(float(R_h(x)[0]), float(R_fn(x)), atol=1e-2,
                            err_msg=f"R mismatch at x={x}")
        npt.assert_allclose(float(Z_h(x)[0]), float(Z_fn(x)), atol=1e-2,
                            err_msg=f"Z mismatch at x={x}")
