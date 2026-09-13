"""Native map2disc: exact oracles, invertibility, and comparisons to analytic maps."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from mrx.map2disc import (
    BoundaryCurve,
    boundary_from_fourier,
    boundary_from_samples,
    concentric_nodes,
    double_layer_matrix,
    fit_disc_map,
    harmonic_map,
    invert_harmonic_map,
    map2disc_map,
    zernike_basis,
    zernike_count,
    zernike_indices,
    zernike_radial,
)
from mrx.mappings import (
    rotating_ellipse_map,
    stellarator_symmetry_defect,
    toroid_map,
)
from mrx.precision import DTYPE

ATOL = 1e-5 if np.dtype(DTYPE) == np.dtype(np.float32) else 1e-12


def _ellipse_samples(a: float = 2.0, b: float = 1.0, n: int = 64) -> np.ndarray:
    t = np.arange(n) / n
    return np.stack([a * np.cos(2.0 * np.pi * t), b * np.sin(2.0 * np.pi * t)], axis=1)


def _ellipse_curve(a: float = 2.0, b: float = 1.0, n: int = 64) -> BoundaryCurve:
    return boundary_from_samples(_ellipse_samples(a, b, n))


def test_harmonic_map_matches_the_ellipse_oracle() -> None:
    """Paper eq. (6)--(7): ``g(x, y) = x/a + i y/b`` on an axis-aligned ellipse."""
    a, b = 2.0, 1.0
    g = harmonic_map(_ellipse_curve(a, b))
    pts = jnp.array([[0.0, 0.0], [0.5, 0.2], [1.0, 0.0], [-0.4, 0.3], [0.2, -0.4]])
    pred = jax.vmap(g)(pts)
    exact = pts / jnp.array([a, b])
    np.testing.assert_allclose(np.asarray(pred), np.asarray(exact), atol=ATOL)


def test_zernike_fit_of_an_ellipse_is_exact_at_degree_one() -> None:
    """The ellipse inverse is degree-1 in ``(xi, eta)``, so ``M = 1`` is exact."""
    a, b = 2.0, 1.0
    fh = fit_disc_map(_ellipse_curve(a, b), M=1)
    coeffs = np.asarray(fh.coeffs)
    # Modes are ``(l, m) = (0,0), (1,-1), (1,1)``: ``1``, ``rho sin``, ``rho cos``.
    np.testing.assert_allclose(coeffs[0], [0.0, 0.0, a], atol=ATOL)
    np.testing.assert_allclose(coeffs[1], [0.0, b, 0.0], atol=ATOL)
    for rho in (0.0, 0.25, 0.5, 0.75, 1.0):
        for theta in np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False):
            pred = np.asarray(fh(float(rho), float(theta)))
            exact = np.array([a * rho * np.cos(theta), b * rho * np.sin(theta)])
            np.testing.assert_allclose(pred, exact, atol=ATOL)


def test_unit_circle_is_the_identity_map() -> None:
    """The harmonic map of the unit disc is the identity; so is ``f_h`` at ``M = 1``."""
    g = harmonic_map(_ellipse_curve(1.0, 1.0))
    xy = jnp.array([0.4, 0.3])
    np.testing.assert_allclose(np.asarray(g(xy)), np.asarray(xy), atol=ATOL)
    fh = fit_disc_map(_ellipse_curve(1.0, 1.0), M=1)
    pred = np.asarray(fh(0.7, 1.1))
    exact = np.array([0.7 * np.cos(1.1), 0.7 * np.sin(1.1)])
    np.testing.assert_allclose(pred, exact, atol=ATOL)


def test_boundary_conformity_and_spectral_interpolation() -> None:
    """``f(e^{i theta}) = gamma(e^{i theta})``; Fourier interpolant hits every sample."""
    curve = _ellipse_curve()
    fh = fit_disc_map(curve, M=1)
    t = jnp.arange(curve.n, dtype=DTYPE) / curve.n
    recovered = jax.vmap(curve.interpolate)(t)
    np.testing.assert_allclose(np.asarray(recovered), np.asarray(curve.samples),
                               atol=ATOL)
    for j, tj in enumerate(np.asarray(t[::8])):
        pred = np.asarray(fh(1.0, float(2.0 * np.pi * tj)))
        np.testing.assert_allclose(pred, np.asarray(curve.samples[j * 8]), atol=ATOL)


def test_harmonic_map_round_trips_the_zernike_inverse() -> None:
    """``g(f_h(rho, theta)) = rho e^{i theta}`` on an interior polar grid."""
    curve = _ellipse_curve()
    g = harmonic_map(curve)
    fh = fit_disc_map(curve, M=1)
    # Stay well inside: trapezoidal Nystrom evaluation of ``g`` degrades
    # as the preimage approaches the boundary (documented limitation).
    for rho in (0.2, 0.4, 0.55):
        for theta in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
            xy = fh(float(rho), float(theta))
            pred = np.asarray(g(xy))
            exact = np.array([rho * np.cos(theta), rho * np.sin(theta)])
            np.testing.assert_allclose(pred, exact, atol=1e-6)


def test_concentric_nodes_are_a_square_system() -> None:
    """Rings ``i = 0..floor(M/2)`` total exactly ``(M+1)(M+2)/2`` points in ``[0, 1]``."""
    for M in range(1, 20):
        rho, theta = concentric_nodes(M)
        assert rho.shape == theta.shape == (zernike_count(M),)
        assert float(jnp.min(rho)) >= -1e-15
        assert float(jnp.max(rho)) <= 1.0 + 1e-15
    with pytest.raises(ValueError, match="positive"):
        concentric_nodes(0)


def test_zernike_radial_identities_and_constant_mode() -> None:
    """``R_1^1 = rho``, ``R_2^0 = 2 rho^2 - 1``, ``R_3^1 = 3 rho^3 - 2 rho``; ``Z_0^0 = 1``."""
    rho = jnp.linspace(0.0, 1.0, 11)
    np.testing.assert_allclose(np.asarray(zernike_radial(1, 1, rho)),
                               np.asarray(rho), atol=ATOL)
    np.testing.assert_allclose(np.asarray(zernike_radial(2, 0, rho)),
                               2.0 * np.asarray(rho)**2 - 1.0, atol=ATOL)
    np.testing.assert_allclose(np.asarray(zernike_radial(3, 1, rho)),
                               3.0 * np.asarray(rho)**3 - 2.0 * np.asarray(rho),
                               atol=ATOL)
    Z = zernike_basis(3)(0.4, 0.7)
    assert float(Z[0]) == pytest.approx(1.0)
    ell, m = zernike_indices(3)
    assert int(ell[0]) == 0 and int(m[0]) == 0
    with pytest.raises(ValueError, match="non-negative"):
        zernike_indices(-1)


def test_zernike_modes_are_orthogonal_on_the_disc() -> None:
    """Disc inner product ``int Z_i Z_j rho d rho d theta`` is diagonal at ``M = 4``."""
    M = 4
    K = zernike_count(M)
    xi, w = leggauss(32)
    rho = 0.5 * (xi + 1.0)
    wr = 0.5 * w * rho
    n_th = 64
    theta = np.linspace(0.0, 2.0 * np.pi, n_th, endpoint=False)
    dth = 2.0 * np.pi / n_th
    basis = zernike_basis(M)
    G = np.zeros((K, K))
    for th in theta:
        Z = np.asarray(basis(jnp.asarray(rho), jnp.full_like(rho, th)))
        G += dth * (Z * wr[:, None]).T @ Z
    diag = np.diag(G)
    assert diag[0] == pytest.approx(np.pi, rel=1e-3)
    off = G - np.diag(diag)
    assert np.max(np.abs(off)) < 0.05 * np.min(diag)


def test_fit_error_decreases_with_zernike_degree() -> None:
    """Interior error vs ``g^{-1}`` drops from ``M = 3`` to ``5`` on a smooth wobble.

    ``M >= 6`` puts a concentric ring above ``rho_safe``, where a plain
    Nystrom evaluation of ``g`` is the documented close-evaluation limit.
    """
    t = np.arange(128) / 128
    th = 2.0 * np.pi * t
    rad = 1.0 + 0.12 * np.cos(2.0 * th)
    samples = np.stack([1.3 * rad * np.cos(th), 0.85 * rad * np.sin(th)], axis=1)
    curve = boundary_from_samples(samples)
    g = harmonic_map(curve)
    rhos = (0.3, 0.5, 0.7)
    thetas = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    targets, guesses = [], []
    center = np.mean(np.asarray(curve.samples), axis=0)
    for rho in rhos:
        for theta in thetas:
            targets.append([rho * np.cos(theta), rho * np.sin(theta)])
            bdry = np.asarray(curve.interpolate(theta / (2.0 * np.pi)))
            guesses.append(center + 0.9 * rho * (bdry - center))
    xy_ref = np.asarray(invert_harmonic_map(
        g, jnp.asarray(targets), jnp.asarray(guesses)))
    errors = []
    for M in (3, 4, 5):
        fh = fit_disc_map(curve, M=M)
        pred = np.array([
            np.asarray(fh(float(rho), float(theta)))
            for rho in rhos for theta in thetas
        ])
        errors.append(float(np.max(np.abs(pred - xy_ref))))
    assert errors[1] < errors[0]
    assert errors[2] < errors[1]
    assert errors[2] < 1e-4


def test_bean_boundary_has_positive_jacobian() -> None:
    """A non-convex limaçon still yields ``det f_h > 0`` on a polar grid."""
    t = np.arange(128) / 128
    th = 2.0 * np.pi * t
    rad = 0.75 + 0.28 * np.cos(th) + 0.12 * np.sin(2.0 * th)
    curve = boundary_from_samples(np.stack([rad * np.cos(th), rad * np.sin(th)], 1))
    fh = fit_disc_map(curve, M=6)
    dets = [
        float(fh.jacobian_determinant(float(rho), float(theta)))
        for rho in np.linspace(0.15, 0.85, 5)
        for theta in np.linspace(0.0, 2.0 * np.pi, 10, endpoint=False)
    ]
    assert min(dets) > 0.0
    xy = fh(jnp.array([0.4, 0.2]))
    assert xy.shape == (2,)
    assert np.all(np.isfinite(np.asarray(xy)))


def _poloidal_from_map(F, zeta: float, n: int = 64) -> np.ndarray:
    """``(R, Z)`` samples of the ``r = 1`` cross-section of a 3-D map."""
    pts = []
    for t in np.arange(n) / n:
        xyz = np.asarray(F(jnp.array([1.0, float(t), float(zeta)])))
        pts.append([np.hypot(xyz[0], xyz[1]), xyz[2]])
    return np.asarray(pts)


def test_map2disc_reproduces_toroid_map() -> None:
    """An elliptical axisymmetric boundary recovers ``toroid_map`` at ``M = 1``."""
    donut = toroid_map()
    def _curve(_zeta):
        return boundary_from_samples(_poloidal_from_map(donut, _zeta))

    F = map2disc_map(_curve, nfp=1, M=1, n_zeta=1)
    xs = jnp.array([[0.3, 0.1, 0.2], [0.6, 0.4, 0.7], [0.9, 0.8, 0.15]])
    for x in xs:
        np.testing.assert_allclose(np.asarray(F(x)), np.asarray(donut(x)),
                                   atol=ATOL)
    assert float(stellarator_symmetry_defect(F, xs)) < ATOL


def test_map2disc_reproduces_rotating_ellipse_and_is_symmetric() -> None:
    """Each rotating-ellipse cross-section is an ellipse, exact at ``M = 1``."""
    nfp = 3
    analytic = rotating_ellipse_map(nfp=nfp)
    F = map2disc_map(lambda zeta: _poloidal_from_map(analytic, zeta),
                     nfp=nfp, M=1, n_zeta=8)
    xs = jnp.array([[0.3, 0.1, 0.2], [0.6, 0.4, 0.7],
                    [0.9, 0.8, 0.15], [0.5, 0.25, 0.5]])
    for x in xs:
        np.testing.assert_allclose(np.asarray(F(x)), np.asarray(analytic(x)),
                                   atol=ATOL)
    assert float(stellarator_symmetry_defect(F, xs)) < ATOL


def test_boundary_from_fourier_recovers_an_ellipse() -> None:
    """``R = 2 cos 2 pi t``, ``Z = sin 2 pi t`` matches ``boundary_from_samples``."""
    fourier = boundary_from_fourier(
        rcos=jnp.array([0.0, 2.0]), rsin=jnp.array([0.0, 0.0]),
        zcos=jnp.array([0.0, 0.0]), zsin=jnp.array([0.0, 1.0]),
        n_boundary=64)
    samples = boundary_from_samples(_ellipse_samples(2.0, 1.0, 64))
    np.testing.assert_allclose(np.asarray(fourier.samples),
                               np.asarray(samples.samples), atol=ATOL)
    assert fourier.d2.shape == fourier.samples.shape
    assert fourier.speed.shape == (fourier.n,)
    assert float(jnp.min(fourier.speed)) > 0.0


def test_double_layer_row_sum_is_minus_one_half() -> None:
    """Outward-normal double layer of a constant is ``-1/2`` on the boundary."""
    A = np.asarray(double_layer_matrix(_ellipse_curve(1.0, 1.0, 48)))
    np.testing.assert_allclose(A.sum(axis=1), -0.5, atol=ATOL)


def test_map2disc_rejects_bad_inputs() -> None:
    """Construction-time guards on samples, ``M``, ``nfp`` and ``n_zeta``."""
    with pytest.raises(ValueError, match="shape"):
        boundary_from_samples(jnp.arange(10.0))
    with pytest.raises(ValueError, match="at least 8"):
        boundary_from_samples(jnp.ones((4, 2)))
    with pytest.raises(ValueError, match="positive"):
        fit_disc_map(_ellipse_curve(), M=0)
    with pytest.raises(ValueError, match="nfp"):
        map2disc_map(lambda z: _ellipse_samples(), nfp=0)
    with pytest.raises(ValueError, match="n_zeta"):
        map2disc_map(lambda z: _ellipse_samples(), n_zeta=0)
    with pytest.raises(ValueError, match="BoundaryCurve"):
        map2disc_map(lambda z: jnp.arange(8.0), n_zeta=1)
