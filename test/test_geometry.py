"""Analytic maps and geometry-file helpers: no DeRhamSequence required."""
from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mrx.differential_forms import DifferentialForm
from mrx.geometry import (
    build_sequence,
    geometry_kind,
    geometry_nfp,
    grad_1d,
    knot_vector,
    map_jacobian_at,
    parse_knots,
    read_analytic,
)
from mrx.mappings import (
    SplineMap,
    cylinder_map,
    extend_map_half_period,
    invert_map_poloidal,
    rotating_ellipse_map,
    stellarator_symmetric_coefficients,
    stellarator_symmetrize,
    stellarator_symmetry_defect,
    toroid_map,
)
from mrx.precision import eps
from mrx.spline_bases import SplineBasis

WOUT = "data/wout_li383_low_res_reference.nc"


def test_build_sequence_rejects_an_unknown_map_source() -> None:
    """``map_source`` names where an equilibrium's map comes from.

    ``"equilibrium"`` projects the file's own R and Z series;
    ``"map2disc"`` keeps only its boundary and builds the interior as a
    harmonic map. A typo must not fall through to the default.
    """
    with pytest.raises(ValueError, match="map_source"):
        build_sequence(WOUT, (4, 4, 4), 2, map_source="harmonic")


def test_knot_vector_counts_and_rejects_nonmonotone() -> None:
    """Clamped: cells + p splines; periodic: cells. Non-monotone breakpoints raise."""
    bp, p = [0.0, 0.25, 0.5, 0.75, 1.0], 2
    n_cells = len(bp) - 1
    clamped = knot_vector(bp, p, periodic=False)
    periodic = knot_vector(bp, p, periodic=True)
    assert len(clamped) == n_cells + 2 * p + 1
    assert len(periodic) == n_cells + 2 * p + 1
    with pytest.raises(ValueError, match="increase from 0 to 1"):
        knot_vector([0.0, 0.5, 0.4, 1.0], p, False)


def test_parse_knots_empty_is_none() -> None:
    assert parse_knots("") is None
    assert parse_knots("0,0.5,1") == [0.0, 0.5, 1.0]


def test_geometry_kind_and_nfp_of_tracked_files() -> None:
    assert geometry_kind("data/torus.json") == "torus"
    assert geometry_kind("data/cylinder.json") == "cylinder"
    assert geometry_kind("data/rot_ellipse.json") == "rot-ellipse"
    assert geometry_kind(WOUT) == "vmec"
    assert geometry_nfp("data/torus.json") == 1
    rot = read_analytic("data/rot_ellipse.json")
    assert geometry_nfp("data/rot_ellipse.json") == int(rot["map_params"]["nfp"])
    assert geometry_nfp(WOUT) == 3
    assert geometry_nfp(WOUT, nfp=5) == 5
    with pytest.raises(ValueError, match="not a file"):
        geometry_kind("data/does_not_exist.json")
    with pytest.raises(ValueError, match="not a geometry file"):
        geometry_kind("README.md")


def test_read_analytic_rejects_unknown_map(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"map": "klein", "map_params": {}}))
    with pytest.raises(ValueError, match="must be one of"):
        read_analytic(str(path))


def test_map_jacobian_at_is_batched() -> None:
    """``x`` is ``(n, 3)``; a single ``(3,)`` point is not a valid argument."""
    f = cylinder_map(a=2.0, h=3.0)
    x = jnp.array([[0.4, 0.15, 0.6], [0.7, 0.8, 0.2]])
    jac = map_jacobian_at(f, x)
    assert jac.shape == (2, 3, 3)
    np.testing.assert_allclose(np.asarray(jac), np.asarray(jax.vmap(jax.jacfwd(f))(x)),
                               atol=1e-10)
    det = float(jnp.linalg.det(jac[0]))
    assert abs(det - 2.0 * np.pi * (2.0 ** 2) * 3.0 * 0.4) < 1e-5


def test_rotating_ellipse_period_is_a_negative_z_rotation() -> None:
    """One field period: ``F(r, theta, zeta+1) = Rot(-2 pi / nfp) F(...)``.

    The map uses the ``(R cos, -R sin)`` GVEC convention, so the rotation
    about z is negative, not the usual positive ``SO(2)`` generator.
    """
    nfp = 3
    f = rotating_ellipse_map(nfp=nfp)
    x = jnp.array([0.5, 0.3, 0.2])
    a, b = f(x), f(x.at[2].add(1.0))
    delta = 2.0 * np.pi / nfp
    rot = jnp.array([[jnp.cos(delta), jnp.sin(delta), 0.0],
                     [-jnp.sin(delta), jnp.cos(delta), 0.0],
                     [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(np.asarray(b), np.asarray(rot @ a), atol=1e-6)
    with pytest.raises(ValueError, match="nfp"):
        rotating_ellipse_map(nfp=0)
    with pytest.raises(ValueError, match="eps"):
        rotating_ellipse_map(eps=0.0)


def test_toroid_map_is_axisymmetric() -> None:
    f = toroid_map()
    x = jnp.array([0.4, 0.2, 0.1])
    y0, y1 = f(x), f(x.at[2].add(1.0))
    np.testing.assert_allclose(np.asarray(y0), np.asarray(y1), atol=1e-12)


def test_grad_1d_clamped_and_periodic() -> None:
    """Clamped pads then differences; periodic is a circular difference."""
    d = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    clamped = grad_1d(d, "clamped")
    assert clamped.shape[0] == d.shape[0] + 1
    periodic = grad_1d(d, "periodic")
    np.testing.assert_allclose(np.asarray(periodic),
                               np.asarray(jnp.roll(d, 1, axis=0) - d))


_SYMMETRY_POINTS = jnp.array([
    [0.20, 0.10, 0.15],
    [0.45, 0.30, 0.40],
    [0.70, 0.80, 0.65],
    [0.55, 0.05, 0.90],
    [0.35, 0.60, 0.25],
])


def _tilted_toroid(amplitude: float = 0.1):
    """``toroid_map`` plus an even-in-theta ``Z`` tilt, which breaks symmetry."""
    base = toroid_map()

    def F(x):
        r, θ, _ = x
        y = base(x)
        return y.at[2].add(amplitude * jnp.cos(2.0 * jnp.pi * θ))

    return F


def _collocate_map(F, basis_0: DifferentialForm) -> jnp.ndarray:
    """Tensor-product Greville interpolant of ``F`` as ``(3, n_r, n_t, n_z)``."""
    br, bt, bz = basis_0.Λ
    gr, gt, gz = br.greville_points(), bt.greville_points(), bz.greville_points()
    rr, tt, zz = jnp.meshgrid(gr, gt, gz, indexing="ij")
    pts = jnp.stack([rr, tt, zz], axis=-1).reshape(-1, 3)
    vals = jax.vmap(F)(pts).T.reshape(3, br.n, bt.n, bz.n)

    def _solve_axis(matrix, arr, axis):
        moved = jnp.moveaxis(arr, axis, 0)
        solved = jnp.linalg.solve(matrix, moved.reshape(moved.shape[0], -1))
        return jnp.moveaxis(solved.reshape(moved.shape), 0, axis)

    coeffs = vals
    for axis, basis in enumerate((br, bt, bz), start=1):
        coeffs = _solve_axis(basis.collocation_matrix(), coeffs, axis)
    return coeffs


def test_existing_maps_have_zero_stellarator_defect() -> None:
    """``toroid``, ``cylinder`` and ``rotating_ellipse`` are exact fixed points."""
    for F in (toroid_map(), cylinder_map(), rotating_ellipse_map(nfp=3)):
        assert float(stellarator_symmetry_defect(F, _SYMMETRY_POINTS)) < 1e-14


def test_stellarator_symmetrize_is_idempotent_and_fixes_symmetric_maps() -> None:
    """The projector is the identity on already-symmetric maps, and P^2 = P."""
    F = rotating_ellipse_map()
    P = stellarator_symmetrize(F)
    PP = stellarator_symmetrize(P)
    for x in _SYMMETRY_POINTS:
        np.testing.assert_allclose(np.asarray(P(x)), np.asarray(F(x)), atol=1e-14)
        np.testing.assert_allclose(np.asarray(PP(x)), np.asarray(P(x)), atol=1e-14)


def test_stellarator_symmetrize_kills_an_even_z_tilt() -> None:
    """An even-in-theta ``Z`` tilt of size 0.1 is projected out exactly.

    The odd part of the tilted ``Z`` is the original ``toroid_map`` ``Z``.
    """
    base = toroid_map()
    tilted = _tilted_toroid(0.1)
    assert float(stellarator_symmetry_defect(tilted, _SYMMETRY_POINTS)) > 0.05
    P = stellarator_symmetrize(tilted)
    assert float(stellarator_symmetry_defect(P, _SYMMETRY_POINTS)) < 1e-14
    for x in _SYMMETRY_POINTS:
        np.testing.assert_allclose(np.asarray(P(x)), np.asarray(base(x)), atol=1e-12)
        odd_z = 0.5 * (tilted(x)[2] - tilted(jnp.array([x[0], -x[1], -x[2]]))[2])
        assert abs(float(P(x)[2] - odd_z)) < 1e-12


def test_periodic_basis_reflection_is_the_index_permutation() -> None:
    """``B_j(-x) = B_{(p - 1 - j) mod n}(x)`` on a 400-point grid."""
    xs = jnp.linspace(0.0, 1.0, 401)[:-1]
    for n, p in ((8, 2), (8, 3), (7, 1), (12, 2)):
        b = SplineBasis(n, p, "periodic")
        perm = (p - 1 - jnp.arange(n)) % n
        at_x = jax.vmap(lambda t: jax.vmap(lambda i: b(t, i))(jnp.arange(n)))(xs)
        at_minus = jax.vmap(lambda t: jax.vmap(lambda i: b(t, i))(jnp.arange(n)))(
            jnp.mod(-xs, 1.0))
        np.testing.assert_allclose(
            np.asarray(at_minus), np.asarray(at_x[:, perm]), atol=1e-10)


def test_spline_map_stellarator_flag_is_the_coefficient_projector() -> None:
    """The flag is a no-op on a symmetric interpolant and equals the
    pointwise projector on a tilted one."""
    basis_0 = DifferentialForm(
        0, (4, 6, 6), (2, 2, 2), ("clamped", "periodic", "periodic"))
    extraction = jnp.eye(basis_0.n)
    points = _SYMMETRY_POINTS

    # nfp=1: Cartesian components are periodic in logical zeta. For nfp>1
    # they pick up a 2 pi / nfp rotation per period and a periodic spline
    # cannot represent them.
    raw_sym = _collocate_map(rotating_ellipse_map(nfp=1), basis_0)
    plain = SplineMap(raw_sym.reshape(3, -1), extraction, basis_0)
    flagged = SplineMap(raw_sym.reshape(3, -1), extraction, basis_0,
                        stellarator_symmetric=True)
    np.testing.assert_allclose(np.asarray(plain.raw), np.asarray(flagged.raw),
                               atol=1e-10)
    for x in points:
        np.testing.assert_allclose(np.asarray(plain(x)), np.asarray(flagged(x)),
                                   atol=1e-10)

    raw_tilt = _collocate_map(_tilted_toroid(0.1), basis_0)
    unflagged = SplineMap(raw_tilt.reshape(3, -1), extraction, basis_0)
    flagged_tilt = SplineMap(raw_tilt.reshape(3, -1), extraction, basis_0,
                             stellarator_symmetric=True)
    projected = stellarator_symmetrize(unflagged)
    np.testing.assert_allclose(
        np.asarray(flagged_tilt.raw),
        np.asarray(stellarator_symmetric_coefficients(raw_tilt, basis_0)),
        atol=1e-12)
    for x in points:
        np.testing.assert_allclose(
            np.asarray(flagged_tilt(x)), np.asarray(projected(x)), atol=1e-10)
    assert float(stellarator_symmetry_defect(flagged_tilt, points)) < 1e-10


def test_stellarator_symmetric_coefficients_rejects_bad_axes() -> None:
    """Non-periodic angular axes and a shape mismatch raise ``ValueError``."""
    clamped = DifferentialForm(
        0, (4, 6, 6), (2, 2, 2), ("clamped", "clamped", "periodic"))
    raw = jnp.zeros((3,) + clamped.shape[0])
    with pytest.raises(ValueError, match="theta axis"):
        stellarator_symmetric_coefficients(raw, clamped)
    periodic = DifferentialForm(
        0, (4, 6, 6), (2, 2, 2), ("clamped", "periodic", "periodic"))
    with pytest.raises(ValueError, match="shape"):
        stellarator_symmetric_coefficients(jnp.zeros((3, 2, 2, 2)), periodic)


def _poloidal_points() -> jnp.ndarray:
    """Near-axis, mid-radius and near-wall probes, several zeta."""
    return jnp.array([
        [0.04, 0.20, 0.10],
        [0.04, 0.80, 0.70],
        [0.50, 0.30, 0.25],
        [0.50, 0.05, 0.90],
        [0.95, 0.70, 0.40],
        [0.95, 0.15, 0.55],
    ])


def _assert_poloidal_roundtrip(F, inverse, xs, atol: float) -> None:
    """``F(inverse(F(x), x[2]), x[2])`` recovers ``F(x)``, and ``rho`` matches."""
    for x in xs:
        p = F(x)
        rho, theta = inverse(p, x[2])
        recovered = F(jnp.array([rho, theta, x[2]]))
        np.testing.assert_allclose(np.asarray(recovered), np.asarray(p), atol=atol)
        assert abs(float(rho - x[0])) < atol


def test_invert_map_poloidal_round_trips_a_rotating_ellipse() -> None:
    """Analytic nested map: Newton from the axis recovers rho, including the wall."""
    F = rotating_ellipse_map(nfp=3)
    inverse = invert_map_poloidal(F)
    _assert_poloidal_roundtrip(F, inverse, _poloidal_points(), atol=1e3 * float(eps()))


def test_invert_map_poloidal_survives_jit_and_vmap() -> None:
    """The body is a ``lax.fori_loop``; a traced batch must not hit Python."""
    F = rotating_ellipse_map(nfp=2)
    inverse = invert_map_poloidal(F)
    xs = _poloidal_points()
    ps = jax.vmap(F)(xs)

    def _one(p, zeta):
        return inverse(p, zeta)

    rhos, thetas = jax.jit(jax.vmap(_one))(ps, xs[:, 2])
    for i, x in enumerate(xs):
        recovered = F(jnp.array([rhos[i], thetas[i], x[2]]))
        np.testing.assert_allclose(np.asarray(recovered), np.asarray(ps[i]),
                                   atol=1e3 * float(eps()))
        assert abs(float(rhos[i] - x[0])) < 1e3 * float(eps())


def test_invert_map_poloidal_round_trips_the_vmec_map(seq) -> None:
    """The equilibrium interpolant is nested about the axis, so the same seed works."""
    inverse = invert_map_poloidal(seq.map)
    _assert_poloidal_roundtrip(seq.map, inverse, _poloidal_points(),
                               atol=1e-6)


def test_build_sequence_map2disc_installs_an_invertible_map(seq_map2disc) -> None:
    """The happy path the suite was missing: ``map_source="map2disc"`` builds."""
    assert seq_map2disc.map_source == "map2disc"
    dets = jnp.linalg.det(map_jacobian_at(seq_map2disc.map, _poloidal_points()))
    assert bool(jnp.all(jnp.isfinite(dets)))
    assert float(jnp.min(dets)) > 0.0


def test_relax_cli_accepts_map_source() -> None:
    """``scripts/relax.py --map-source map2disc`` parses; the default stays equilibrium.

    The parsed value is what :func:`mrx.geometry.build_sequence` receives
    (``scripts/relax.py`` passes ``cli.map_source`` through). Building the
    sequence is :func:`test_build_sequence_map2disc_installs_an_invertible_map`.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "relax_cli", "scripts/relax.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cli = mod.parse_args([
        "--geometry", WOUT, "--map-source", "map2disc",
        "--ns", "4,6,6", "--p", "2", "--steps", "1", "--chunk", "1",
    ])
    assert cli.map_source == "map2disc"
    assert mod.parse_args(["--geometry", WOUT]).map_source == "equilibrium"


def test_extend_map_half_period_reproduces_a_symmetric_map() -> None:
    """Reflection across ``zeta = 1/2`` rebuilds ``rotating_ellipse_map``."""
    F = rotating_ellipse_map()
    F_ext = extend_map_half_period(F)
    for x in _SYMMETRY_POINTS:
        np.testing.assert_allclose(np.asarray(F_ext(x)), np.asarray(F(x)),
                                   atol=1e-12)
