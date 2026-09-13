"""Analytic maps and geometry-file helpers: no DeRhamSequence required."""
from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mrx.geometry import (
    geometry_kind,
    geometry_nfp,
    grad_1d,
    knot_vector,
    map_jacobian_at,
    parse_knots,
    read_analytic,
)
from mrx.mappings import cylinder_map, rotating_ellipse_map, toroid_map

WOUT = "data/wout_li383_low_res_reference.nc"


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
