"""Native map2disc: exact oracles, invertibility, and comparisons to analytic maps."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from mrx.gvec import StateField, read_equilibrium
from mrx.map2disc import (
    GAP_RATIO,
    N_BOUNDARY_MAX,
    BoundaryCurve,
    _interior_seed,
    _invert_nodes,
    boundary_from_fourier,
    boundary_from_samples,
    concentric_nodes,
    double_layer_matrix,
    fit_disc_map,
    harmonic_map,
    invert_harmonic_map,
    lcfs_boundary,
    map2disc_from_equilibrium,
    map2disc_map,
    nyquist_n_zeta,
    zernike_basis,
    zernike_count,
    zernike_eval,
    zernike_eval_cartesian,
    zernike_indices,
    zernike_radial,
)
from mrx.mappings import (
    rotating_ellipse_map,
    stellarator_symmetry_defect,
    toroid_map,
)
from mrx.precision import DTYPE

FLOAT32 = np.dtype(DTYPE) == np.dtype(np.float32)
ATOL = 1e-5 if FLOAT32 else 1e-12

#: Interior accuracy a refined fit reaches in the working precision. The
#: chain is a dense Nystrom solve, a Newton inversion through it and a
#: Zernike solve, so float32 floors out near 3e-5 however high the
#: boundary resolution and the degree go.
FIT_FLOOR = 5e-5 if FLOAT32 else 1e-5

#: Error ratio a four-degree jump must show, i.e. what "converges
#: spectrally" is worth asserting. The near-boundary blend this module
#: used to apply held the ratio at 1 whatever the degree, which is what
#: these tests exist to catch; in float32 the ratio is capped by
#: :data:`FIT_FLOOR` rather than by the method.
FIT_RATIO = 30.0 if FLOAT32 else 100.0

#: Landreman-Paul QA: nfp = 2, poloidal modes to |m| = 7, toroidal to
#: |n| / nfp = 8. Small enough to fit in the suite, and a real
#: stellarator boundary rather than the ellipses the rest of this file
#: uses -- the ellipse is exactly the class of shape that hid the
#: close-evaluation defect this module used to have.
QA_WOUT = "data/wout_LandremanPaul2021_QA_lowres.nc"

#: A toroidal plane of the QA boundary that is not the elongated
#: ``zeta = 0`` bean. Cheap to resolve, so the per-plane tests use it.
QA_ZETA = 0.25

#: NCSX (li383): nfp = 3, and harder than QA in the one way that matters
#: here -- its cross-sections are crescents rather than beans, so the
#: boundary centroid is not inside the plasma and the domain is not
#: star-shaped about any point.
NCSX_WOUT = "data/wout_li383_low_res_reference.nc"

#: The NCSX plane with the most pronounced crescent, and the one where the
#: centroid sits farthest outside: ``|g(centroid)| = 8.04``.
NCSX_ZETA = 0.0

#: Interior accuracy the NCSX crescent reaches, and the gain from
#: ``M = 3`` to ``M = 6``. Both are far short of :data:`FIT_FLOOR` and
#: :data:`FIT_RATIO`: the crescent's inverse needs a Zernike degree the
#: resolution gate will not let it have. See the test for the measurement
#: that shows this is truncation and not close evaluation.
NCSX_FIT_FLOOR = 5e-3
NCSX_FIT_RATIO = 5.0


def _ellipse_samples(a: float = 2.0, b: float = 1.0, n: int = 64) -> np.ndarray:
    t = np.arange(n) / n
    return np.stack([a * np.cos(2.0 * np.pi * t), b * np.sin(2.0 * np.pi * t)], axis=1)


def _ellipse_curve(a: float = 2.0, b: float = 1.0, n: int = 64) -> BoundaryCurve:
    return boundary_from_samples(_ellipse_samples(a, b, n))


def _bean_curve(n: int = 256) -> BoundaryCurve:
    """A strongly non-convex cross-section: ``f`` is NOT linear in ``rho``."""
    a = 2.0 * np.pi * np.arange(n) / n
    rad = 1.0 + 0.45 * np.cos(a) + 0.25 * np.cos(2.0 * a)
    return boundary_from_samples(np.stack([rad * np.cos(a), 0.7 * rad * np.sin(a)], 1))


def _severe_bean_curve(n: int = 2048) -> BoundaryCurve:
    """A bean so non-convex that the crude Newton start walks out of it.

    Aspect ratio 6.3 between the widest and narrowest radius.
    """
    a = 2.0 * np.pi * np.arange(n) / n
    rad = 1.0 + 0.85 * np.cos(a) + 0.425 * np.cos(2.0 * a)
    return boundary_from_samples(np.stack([rad * np.cos(a), 0.6 * rad * np.sin(a)], 1))


def _crude_start(curve: BoundaryCurve, rho, theta):
    """The radial Newton guess continuation replaces: ``c + 0.9 rho (gamma - c)``."""
    center = jnp.mean(curve.samples, axis=0)
    edge = jax.vmap(curve.interpolate)(theta / (2.0 * jnp.pi))
    return center[None, :] + (0.9 * rho[:, None]) * (edge - center[None, :])


def _roundtrip_error(fh, g, rhos, n_theta: int = 12) -> float:
    """``max |g(f_h(rho, theta)) - (rho, theta)|`` over a polar grid."""
    rho, theta = np.meshgrid(np.asarray(rhos, dtype=float),
                             np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False),
                             indexing="ij")
    rho, theta = jnp.asarray(rho.ravel()), jnp.asarray(theta.ravel())
    got = jax.vmap(g)(jax.vmap(fh)(rho, theta))
    want = jnp.stack([rho * jnp.cos(theta), rho * jnp.sin(theta)], axis=1)
    return float(jnp.max(jnp.abs(got - want)))


@pytest.fixture(scope="session")
def qa_state():
    """The parsed Landreman-Paul QA wout, read once."""
    return read_equilibrium(QA_WOUT)


@pytest.fixture(scope="session")
def qa_curve(qa_state):
    """The QA boundary at :data:`QA_ZETA`, as a :class:`BoundaryCurve`."""
    return lcfs_boundary(qa_state)(QA_ZETA)


@pytest.fixture(scope="session")
def qa_reference(qa_curve):
    """``g`` of the QA plane at a resolution far beyond what any fit uses."""
    return harmonic_map(qa_curve.resample(4096))


@pytest.fixture(scope="session")
def qa_fit(qa_curve):
    """The degree-8 fit of the QA plane."""
    return fit_disc_map(qa_curve, M=8)


@pytest.fixture(scope="session")
def qa_map(qa_state):
    """The full 3-D map2disc map of the QA boundary, at the Nyquist planes."""
    return map2disc_from_equilibrium(qa_state, M=6)


@pytest.fixture(scope="session")
def ncsx_state():
    """The parsed NCSX (li383) wout, read once."""
    return read_equilibrium(NCSX_WOUT)


@pytest.fixture(scope="session")
def ncsx_curve(ncsx_state):
    """The NCSX crescent at :data:`NCSX_ZETA`, as a :class:`BoundaryCurve`."""
    return lcfs_boundary(ncsx_state)(NCSX_ZETA)


@pytest.fixture(scope="session")
def ncsx_fit(ncsx_curve):
    """The degree-6 fit of the NCSX crescent."""
    return fit_disc_map(ncsx_curve, M=6)


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
    fh = fit_disc_map(curve, M=1)
    # ``g`` itself is evaluated by a plain Nystrom rule, which needs the
    # target a few boundary spacings inside; 1024 samples buy that out to
    # rho = 0.9 on this ellipse.
    g = harmonic_map(curve.resample(1024))
    assert _roundtrip_error(fh, g, (0.2, 0.4, 0.55, 0.9), n_theta=8) < FIT_FLOOR


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


def test_a_map2disc_map_survives_jit() -> None:
    """A map MRX can mesh with is one the jitted solvers can trace.

    ``zernike_indices`` used to build its index arrays with ``jnp``, which
    makes them TRACERS when the caller is already inside ``jax.jit``.
    Everything downstream treats them as static -- ``np_int_list``,
    ``powers[int(a)]`` -- so a jitted call raised
    ``TracerArrayConversionError`` and ``--map-source map2disc`` could not
    reach the relaxation's jitted diagnostics at all. ``jax.jacfwd`` alone
    did not catch this: outside ``jit`` the arrays stay concrete.
    """
    ell, m = zernike_indices(3)
    assert isinstance(ell, np.ndarray) and isinstance(m, np.ndarray)

    F = map2disc_map(lambda z: _ellipse_samples(2.0 + 0.3 * np.cos(2.0 * np.pi * z), 1.0),
                     nfp=2, M=3, n_zeta=3)
    x = jnp.array([0.4, 0.2, 0.1])
    np.testing.assert_allclose(np.asarray(jax.jit(F)(x)), np.asarray(F(x)), atol=ATOL)
    assert np.all(np.isfinite(np.asarray(jax.jit(jax.jacfwd(F))(x))))


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
    """The fit converges SPECTRALLY in ``M`` on a smooth non-elliptical wobble.

    The degrees here all put rings inside the old ``rho_safe = 0.85``
    cut, where the fit used to be a linear blend to the boundary. That
    blend is exact for an ellipse and first order for anything else, so
    it pinned the error at ~1e-3 whatever ``M`` and whatever the boundary
    resolution. A merely monotone decrease would not have caught it; the
    two orders of magnitude asserted below do. Measured in float64:
    5.9e-4, 7.5e-6, 1.5e-7 at degrees 3, 5 and 7.
    """
    th = 2.0 * np.pi * np.arange(128) / 128
    rad = 1.0 + 0.12 * np.cos(2.0 * th)
    curve = boundary_from_samples(
        np.stack([1.3 * rad * np.cos(th), 0.85 * rad * np.sin(th)], axis=1))
    g = harmonic_map(curve.resample(2048))
    errors = [_roundtrip_error(fit_disc_map(curve, M=M), g, (0.3, 0.5, 0.7), 8)
              for M in (3, 5)]
    assert errors[0] / errors[1] > 30.0
    assert errors[1] < FIT_FLOOR


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


def test_clockwise_boundary_is_rejected() -> None:
    """The kernel's outward normal ``(y', -x')`` presumes counterclockwise.

    A reversed curve is a perfectly good Jordan curve, so nothing downstream
    fails; it just returns the wrong harmonic map. Caught at construction.
    """
    with pytest.raises(ValueError, match="counterclockwise"):
        boundary_from_samples(_ellipse_samples()[::-1])


def test_resample_is_exact_for_a_band_limited_curve() -> None:
    """Trigonometric resampling adds no error to a finite Fourier boundary.

    This is what lets :func:`fit_disc_map` raise the boundary resolution
    on a curve it was handed rather than demanding a callable.
    """
    coarse = _ellipse_curve(2.0, 1.0, 64)
    np.testing.assert_allclose(np.asarray(coarse.resample(512).samples),
                               _ellipse_samples(2.0, 1.0, 512), atol=ATOL)
    assert coarse.resample(64) is coarse
    with pytest.raises(ValueError, match="at least 8"):
        coarse.resample(4)


def test_cartesian_and_polar_zernike_agree() -> None:
    """The polynomial form reproduces the polar one away from the origin."""
    ell, m = zernike_indices(8)
    rho, theta = 0.63, 1.37
    np.testing.assert_allclose(
        np.asarray(zernike_eval_cartesian(ell, m, rho * np.cos(theta),
                                          rho * np.sin(theta))),
        np.asarray(zernike_eval(ell, m, rho, theta)), atol=ATOL)


def test_jacobian_determinant_is_finite_at_the_axis() -> None:
    """``det Df_h`` at ``rho = 0``, the axis of every MRX map.

    The polar form routes through ``hypot`` and ``atan2``, whose
    derivatives are undefined there, so ``jacfwd`` of it returns NaN.
    """
    fh = fit_disc_map(_ellipse_curve(2.0, 1.0), M=1)
    for theta in (0.0, 1.0, 2.5):
        det = float(fh.jacobian_determinant(0.0, theta))
        assert np.isfinite(det)
        # ``f_h`` is ``(2 xi, eta)`` here, so the determinant is 2 everywhere.
        assert det == pytest.approx(2.0, rel=1e-4)


def test_non_convex_fit_converges_where_a_boundary_blend_cannot(
) -> None:
    """A strongly non-convex bean converges spectrally in ``M``.

    The regression test for the close-evaluation defect. ``f`` is linear
    in ``rho`` for an ellipse, so blending the near-boundary nodes to the
    wall is exact there and every ellipse oracle in this file passed
    while the error on this bean sat at 2.3e-3 for ANY ``M`` and ANY
    boundary resolution. Measured in float64: 1.8e-3 at ``M = 4`` and
    3.1e-7 at ``M = 8``.
    """
    curve = _bean_curve()
    g = harmonic_map(curve.resample(2048))
    coarse = _roundtrip_error(fit_disc_map(curve, M=4), g, (0.3, 0.6, 0.9), 10)
    fine = _roundtrip_error(fit_disc_map(curve, M=8), g, (0.3, 0.6, 0.9), 10)
    assert coarse / fine > FIT_RATIO
    assert fine < FIT_FLOOR


def test_invert_harmonic_map_reports_its_residual() -> None:
    """Newton returns ``(xy, residual)``; the residual is the only convergence signal.

    The loop is fixed-step so that its carry dtype cannot drift from the
    working one, which means a caller that ignores the residual cannot
    tell a converged point from one that ran out of steps.
    """
    a, b = 2.0, 1.0
    curve = _ellipse_curve(a, b, 256)
    g = harmonic_map(curve)
    targets = jnp.array([[0.0, 0.0], [0.4, 0.2], [-0.3, 0.5]])
    xy, residual = invert_harmonic_map(g, targets, jnp.zeros_like(targets))
    assert xy.shape == targets.shape and residual.shape == (3,)
    assert float(jnp.max(residual)) < 1e-6
    # ``g`` is ``(x/a, y/b)`` here, so the inverse is ``(a xi, b eta)``.
    np.testing.assert_allclose(np.asarray(xy),
                               np.asarray(targets) * np.array([a, b]), atol=1e-6)
    _, stalled = invert_harmonic_map(g, targets, jnp.zeros_like(targets), max_iter=0)
    assert float(jnp.max(stalled)) > 1e-6


def test_under_resolved_fit_raises_instead_of_returning_a_folded_map() -> None:
    """Refinement capped below what the degree needs is an error, not a map.

    Left unchecked this returns Zernike coefficients of order ``1e31`` and
    a ``det DF`` that swings through zero -- a map that looks like a map.
    """
    with pytest.raises(ValueError, match="not resolved at n_boundary"):
        fit_disc_map(_bean_curve(), M=12, n_max=64)


def test_continuation_inverts_a_boundary_the_crude_start_cannot() -> None:
    """Degree continuation is what makes a severe bean invertible at all.

    From the crude radial guess the Newton leaves the domain and diverges
    to order ``1e15`` at ``n_boundary = 2048``. Walking the degree up
    ``2, 3, ..., M`` and starting each rung from the previous rung's fit
    keeps every start inside, and converges to the same nodes the crude
    guess only reaches one doubling later.
    """
    curve = _severe_bean_curve()
    rho, theta = concentric_nodes(8)
    inner = np.flatnonzero(np.asarray(rho) < 1.0 - 1e-10)
    targets = jnp.stack([rho[inner] * jnp.cos(theta[inner]),
                         rho[inner] * jnp.sin(theta[inner])], axis=1)
    crude, crude_residual = invert_harmonic_map(
        harmonic_map(curve), targets, _crude_start(curve, rho, theta)[inner])
    assert float(jnp.max(crude_residual)) > 1e-2
    assert float(jnp.max(jnp.abs(crude))) > 1e2

    nodes, residual, _ = _invert_nodes(curve, 8, 25)
    assert residual < FIT_FLOOR
    # Continuation stays inside a boundary whose farthest point is ~2.3.
    assert float(jnp.max(jnp.abs(nodes))) < 4.0


def test_an_unresolved_level_does_not_poison_the_movement_test() -> None:
    """A level that converged but is gap-short is still a valid comparison.

    Refinement accepts a resolution when its nodes have stopped moving
    against the previous one. The previous one only has to have CONVERGED
    -- if it is merely gap-short its nodes are inaccurate, not wrong, and
    the gap is tested on the current level anyway. Requiring the
    predecessor to be gap-resolved too would cost a further doubling and
    push this fit past ``N_BOUNDARY_MAX``; comparing against a DIVERGED
    predecessor would measure a movement of ``1e15`` and reject a fit
    that is right to 2.4e-15.
    """
    curve = _severe_bean_curve()
    _, residual, gap = _invert_nodes(curve, 8, 25)
    assert residual < FIT_FLOOR and gap < GAP_RATIO

    fh = fit_disc_map(curve, M=8, n_max=N_BOUNDARY_MAX)
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, 400, endpoint=False)
    boundary = jax.vmap(lambda t: fh(1.0, t))(theta)
    exact = jax.vmap(curve.interpolate)(theta / (2.0 * jnp.pi))
    assert float(jnp.max(jnp.abs(boundary - exact))) < FIT_FLOOR
    det = jax.vmap(jax.vmap(lambda r, t: fh.jacobian_determinant(r, t)))(
        *jnp.meshgrid(jnp.linspace(0.0, 1.0, 21), theta[::20], indexing="ij"))
    assert float(jnp.min(det)) > 0.0


# ---------------------------------------------------------------------------
# Landreman-Paul QA: a real VMEC boundary
# ---------------------------------------------------------------------------

def test_qa_boundary_is_reproduced_exactly(qa_state, qa_fit) -> None:
    """``f_h(1, theta)`` is the wout's own LCFS, at angles that are not nodes.

    The ``rho = 1`` ring is interpolated from ``2M + 1`` boundary samples,
    so this holds only once ``M`` reaches the boundary's poloidal mode
    number -- 7 for this file. At ``M = 4`` the same check gives 1.5e-4.
    """
    nfp = qa_state["nfp"]
    R_field, Z_field = StateField(qa_state["X1"], nfp), StateField(qa_state["X2"], nfp)
    for t in np.linspace(0.0, 1.0, 37, endpoint=False):
        x = jnp.array([1.0, float(t), QA_ZETA])
        want = np.array([float(R_field(x)), float(Z_field(x))])
        np.testing.assert_allclose(np.asarray(qa_fit(1.0, 2.0 * np.pi * float(t))),
                                   want, atol=ATOL)


def test_qa_fit_converges_with_zernike_degree(qa_curve, qa_reference) -> None:
    """Spectral convergence on a real stellarator cross-section.

        Measured in float64: 1.6e-3 at ``M = 4``, 4.4e-7 at ``M = 8``. Under
    the old near-boundary blend both degrees gave ~5e-3.
    """
    rhos = (0.1, 0.5, 0.8)
    coarse = _roundtrip_error(fit_disc_map(qa_curve, M=4), qa_reference, rhos)
    fine = _roundtrip_error(fit_disc_map(qa_curve, M=8), qa_reference, rhos)
    assert coarse / fine > FIT_RATIO
    assert fine < FIT_FLOOR


def test_qa_fit_is_invertible_including_at_the_axis(qa_fit) -> None:
    """``det Df_h > 0`` everywhere: the paper's invertibility certificate."""
    dets = [float(qa_fit.jacobian_determinant(float(rho), float(theta)))
            for rho in np.linspace(0.0, 1.0, 9)
            for theta in np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)]
    assert np.all(np.isfinite(dets))
    assert min(dets) > 0.0


def test_nyquist_n_zeta_resolves_the_qa_toroidal_content(qa_state) -> None:
    """``n_zeta`` is a Nyquist condition, not a convergence knob.

    QA carries toroidal modes to ``|n| / nfp = 8``, so ``2 * 8 + 1``
    planes are needed; the old default of 8 aliases them.
    """
    n_zeta = nyquist_n_zeta(qa_state)
    assert n_zeta == 17
    n_per = max(abs(int(n)) for n in np.asarray(qa_state["X1"]["n"]))
    assert n_zeta >= 2 * n_per / qa_state["nfp"] + 1


def test_qa_map2disc_map_matches_the_vmec_lcfs(qa_state, qa_map) -> None:
    """The headline comparison: the 3-D map lands on the wout's boundary.

    ``map2disc`` reads only the last closed flux surface, so ``r = 1`` is
    the one place the two maps must agree -- and it agrees to roundoff,
    while the spline projection of ``build_gvec_map`` carries its own
    discretisation error there (4.9e-3 at ns = (6, 8, 8)). Inside, the
    two disagree by construction: map2disc's surfaces are level sets of a
    harmonic map, not the equilibrium's flux surfaces.
    """
    F, info = qa_map
    assert info["sign"] == -1.0
    assert info["n_zeta"] == 17
    assert info["det_range"][0] > 0.0

    nfp = qa_state["nfp"]
    R_field, Z_field = StateField(qa_state["X1"], nfp), StateField(qa_state["X2"], nfp)

    def exact(x):
        R, phi = R_field(x), 2.0 * jnp.pi * x[2] / nfp
        return jnp.array([R * jnp.cos(phi), -R * jnp.sin(phi), Z_field(x)])

    theta, zeta = np.meshgrid(np.linspace(0.0, 1.0, 11, endpoint=False),
                              np.linspace(0.0, 1.0, 9, endpoint=False), indexing="ij")
    pts = jnp.stack([jnp.ones(theta.size), jnp.asarray(theta.ravel()),
                     jnp.asarray(zeta.ravel())], axis=-1)
    np.testing.assert_allclose(np.asarray(jax.vmap(F)(pts)),
                               np.asarray(jax.vmap(exact)(pts)), atol=ATOL)


def test_qa_map2disc_map_is_stellarator_symmetric(qa_map) -> None:
    """A stellarator-symmetric boundary gives a stellarator-symmetric map."""
    F, _ = qa_map
    xs = jnp.array([[0.3, 0.1, 0.2], [0.6, 0.4, 0.7], [0.9, 0.8, 0.15]])
    assert float(stellarator_symmetry_defect(F, xs)) < (1e-3 if FLOAT32 else 1e-8)


# ---------------------------------------------------------------------------
# NCSX (li383): a crescent, where the boundary centroid is not in the plasma
# ---------------------------------------------------------------------------

def test_a_crescent_whose_centroid_is_outside_still_inverts(ncsx_curve,
                                                            ncsx_fit) -> None:
    """The seed bug that made map2disc fail on every NCSX plane, at every degree.

    ``g`` maps the interior onto the unit disc, so ``|g(x)| > 1`` says ``x``
    is outside. The NCSX ``zeta = 0`` centroid measures ``8.04``, and the
    old radial start anchored every Newton there: ``g`` of the start was
    meaningless, the nodes fled to ``1e12`` and the residual pinned at the
    target radius. Moving the anchor to the pole of inaccessibility does
    not help -- a crescent is not star-shaped about any point -- so the
    start has to come from ``g`` itself.
    """
    g = harmonic_map(ncsx_curve)
    centroid = jnp.mean(ncsx_curve.samples, axis=0)
    assert float(jnp.linalg.norm(g(centroid))) > 1.0

    dets = [float(ncsx_fit.jacobian_determinant(float(rho), float(theta)))
            for rho in np.linspace(0.0, 1.0, 9)
            for theta in np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)]
    assert np.all(np.isfinite(dets))
    assert min(dets) > 0.0


def test_interior_seed_beats_the_centroid_start_on_the_crescent(ncsx_curve) -> None:
    """Every seed is inside, and Newton from it converges where the centroid cannot."""
    g = harmonic_map(ncsx_curve)
    rho, theta = concentric_nodes(4)
    inner = np.flatnonzero(np.asarray(rho) < 1.0 - 1e-10)
    targets = jnp.stack([rho[inner] * jnp.cos(theta[inner]),
                         rho[inner] * jnp.sin(theta[inner])], axis=1)

    seed = _interior_seed(ncsx_curve, g, targets)
    assert seed.shape == targets.shape
    assert float(jnp.max(jnp.linalg.norm(jax.vmap(g)(seed), axis=1))) <= 1.0

    _, crude_residual = invert_harmonic_map(
        g, targets, _crude_start(ncsx_curve, rho, theta)[inner])
    assert float(jnp.max(crude_residual)) > 1e-2

    _, residual, _ = _invert_nodes(ncsx_curve, 4, 25)
    assert residual < FIT_FLOOR


def test_ncsx_boundary_is_reproduced_and_the_fit_converges(ncsx_state,
                                                           ncsx_curve,
                                                           ncsx_fit) -> None:
    """``f_h(1, theta)`` is the wout's LCFS, and the interior converges in ``M``.

    The crescent converges far more slowly than the QA bean, and for a
    reason worth recording: the residual is flat in the boundary
    resolution (identical against ``g`` at 2048 and at 4096) and peaks at
    ``rho ~ 0.6``, tens of boundary spacings from the wall, so it is
    Zernike truncation and not close evaluation. Measured in float64:
    3.9e-2 at ``M = 3``, 3.6e-3 at 4, 5.1e-4 at 8. Degree 12 cannot be
    reached at all -- the concentric rings crowd the wall faster than
    :data:`N_BOUNDARY_MAX` can resolve them, and :func:`fit_disc_map`
    raises rather than returning the folded map. The mild ``zeta = 0.25``
    plane of the same file converges spectrally to 2.3e-8 at ``M = 12``,
    so this is the crescent, not the file.
    """
    nfp = ncsx_state["nfp"]
    R_field = StateField(ncsx_state["X1"], nfp)
    Z_field = StateField(ncsx_state["X2"], nfp)
    for t in np.linspace(0.0, 1.0, 25, endpoint=False):
        x = jnp.array([1.0, float(t), NCSX_ZETA])
        np.testing.assert_allclose(np.asarray(ncsx_fit(1.0, 2.0 * np.pi * float(t))),
                                   np.array([float(R_field(x)), float(Z_field(x))]),
                                   atol=ATOL)

    reference = harmonic_map(ncsx_curve.resample(4096))
    rhos = (0.1, 0.5, 0.8)
    coarse = _roundtrip_error(fit_disc_map(ncsx_curve, M=3), reference, rhos)
    fine = _roundtrip_error(ncsx_fit, reference, rhos)
    assert coarse / fine > NCSX_FIT_RATIO
    assert fine < NCSX_FIT_FLOOR
