"""Poincare tracer: the analytic sheared screw field is an exact iota oracle.

``analytic_profile_form`` with no lambda modes is already a sheared screw
field in logical components, so ``d theta / d zeta = iota(r)`` and
``dr / d zeta = 0``. ``rotational_transform`` multiplies the logical slope
by ``nfp``, which gives a closed-form answer per seed. The integration
error is Tsit5 truncation (the tracer runs in float64 in both working
dtypes), so the tolerances here are absolute, not ``eps``-scaled.
"""
from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from mrx.initial_conditions import analytic_profile_form, make_lambda, make_profiles
from mrx.poincare import (
    BzetaParameterisationError,
    R_MAX,
    _escaped_mask,
    _iota_convergence,
    _iota_window_scatter,
    _step_convergence,
    axis_track,
    logical_field,
    midplane_crossings,
    require_zeta_parameterisation,
    rotational_transform,
    section_RZ,
    section_figure,
    seed_from_axis,
    surface_label,
    to_RZ,
    trace,
    trace_and_classify,
)

NFP = 3
SAVES = 8
IOTA0, IOTA1, IOTA_EXP, FLUX_EXP = 0.4, 0.9, 2.0, 1.0


def _screw_field() -> tuple[Callable, Callable]:
    """The production reference 2-form with no lambda: ``d theta / d zeta = iota(r)``."""
    iota_f, dphi = make_profiles(IOTA0, IOTA1, IOTA_EXP, FLUX_EXP)
    return analytic_profile_form(iota_f, dphi, make_lambda([])), iota_f


def _exact_iota(iota_f: Callable, seeds: jnp.ndarray) -> np.ndarray:
    """``nfp * iota(r)`` at each seed radius."""
    radii = np.asarray(seeds)[:, 0]
    return NFP * np.array([float(iota_f(r)) for r in radii])


def test_iota_matches_the_screw_field_and_converges_like_tsit5() -> None:
    """Absolute error at production ``steps_per_period`` and a factor of 4 per doubling."""
    field, iota_f = _screw_field()
    seeds = jnp.array([[0.01, 0.0], [0.25, 0.0], [0.5, 0.3], [0.75, 0.6]])
    exact = _exact_iota(iota_f, seeds)
    origin = None
    errors = []
    for spp in (16, 32, 64):
        ys, ok = trace(field, seeds, n_periods=64, steps_per_period=spp,
                       saves_per_period=SAVES)
        assert np.all(np.asarray(ok))
        if origin is None:
            origin = jnp.zeros((ys.shape[1], 2))
        iota, _ = rotational_transform(ys, SAVES, NFP, center=origin)
        errors.append(float(np.max(np.abs(np.asarray(iota) - exact))))
    assert errors[1] < 1e-5
    assert errors[0] / errors[1] >= 4.0
    assert errors[1] / errors[2] >= 4.0


def test_screw_field_conserves_radius() -> None:
    field, _ = _screw_field()
    seeds = jnp.array([[0.2, 0.0], [0.5, 0.3], [0.75, 0.6]])
    ys, _ = trace(field, seeds, n_periods=64, steps_per_period=32,
                  saves_per_period=SAVES)
    r = np.hypot(np.asarray(ys)[..., 0], np.asarray(ys)[..., 1])
    assert float(np.max(np.abs(r - r[:, :1]))) < 1e-5


def test_axis_track_is_near_the_origin_and_iota_stays_accurate() -> None:
    field, iota_f = _screw_field()
    seeds = jnp.array([[0.01, 0.0], [0.25, 0.0], [0.5, 0.3], [0.75, 0.6]])
    ys, _ = trace(field, seeds, n_periods=64, steps_per_period=32,
                  saves_per_period=SAVES)
    centre = axis_track(ys, SAVES)
    assert float(jnp.max(jnp.abs(centre))) < 2e-4
    exact = _exact_iota(iota_f, seeds)
    iota, _ = rotational_transform(ys, SAVES, NFP)
    assert float(np.max(np.abs(np.asarray(iota) - exact))) < 1e-5


def test_require_zeta_parameterisation_accepts_the_screw_field() -> None:
    field, _ = _screw_field()
    info = require_zeta_parameterisation(field, name="screw")
    assert 0.7 < info["bz_over_b_min"] < info["bz_over_b_max"] < 1.0
    assert not info["sign_change"]


def test_require_zeta_parameterisation_rejects_a_sign_change() -> None:
    def field(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([0.0, 1.0, jnp.cos(2.0 * jnp.pi * x[2])])

    with pytest.raises(BzetaParameterisationError, match="CHANGES SIGN"):
        require_zeta_parameterisation(field, name="sign")


def test_require_zeta_parameterisation_rejects_a_small_bzeta() -> None:
    def field(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([0.0, 1.0, 0.01])

    with pytest.raises(BzetaParameterisationError, match="comes within"):
        require_zeta_parameterisation(field, name="small")


def test_trace_requires_steps_to_be_a_multiple_of_saves() -> None:
    field, _ = _screw_field()
    with pytest.raises(ValueError, match="multiple"):
        trace(field, jnp.array([[0.3, 0.0]]), n_periods=4,
              steps_per_period=5, saves_per_period=2)


def test_trace_and_classify_marks_the_screw_field_regular() -> None:
    field, _ = _screw_field()
    seeds = seed_from_axis(field, n_seeds=4, saves_per_period=SAVES,
                           n_rays=2, probe_periods=16, steps_per_period=32)
    res = trace_and_classify(field, seeds, NFP, n_periods=32,
                             steps_per_period=32, saves_per_period=SAVES)
    assert not np.any(res["chaotic"])
    assert np.all(res["ok"])
    assert not np.any(res["escaped"])
    assert np.all(np.isfinite(res["iota"]))
    assert res["drift"] < 1e-3
    assert np.all(res["iota_scatter"] < 2e-2)
    ys, _ = trace(field, jnp.array([[0.2, 0.0], [0.5, 0.3]]), 32, 32, SAVES)
    conv = np.asarray(_iota_convergence(ys, SAVES, NFP))
    scatter = np.asarray(_iota_window_scatter(ys, SAVES, NFP))
    assert np.all(conv < 1e-4)
    assert np.all(scatter < 2e-2)


def test_escaped_mask_flags_a_seed_past_r_max() -> None:
    n_saves = 5
    ys = np.zeros((2, n_saves, 2))
    ys[0, :, 0] = 0.3
    ys[1, -1, 0] = float(R_MAX) + 0.01
    mask = np.asarray(_escaped_mask(jnp.asarray(ys)))
    assert mask.tolist() == [False, True]


def test_step_convergence_is_tiny_on_the_screw_field() -> None:
    field, _ = _screw_field()
    seeds = jnp.array([[0.3, 0.0], [0.6, 0.25]])
    drift = _step_convergence(field, seeds, n_periods=8, steps_per_period=16,
                              saves_per_period=SAVES)
    assert drift < 1e-4


def test_midplane_crossings_of_a_circle() -> None:
    a, centre_r, centre_z = 0.3, 1.0, 0.0
    th = np.linspace(0.0, 2.0 * np.pi, 40, endpoint=False)
    r = centre_r + a * np.cos(th)
    z = centre_z + a * np.sin(th)
    crossings = np.asarray(midplane_crossings(
        jnp.asarray(r[None, :]), jnp.asarray(z[None, :]), centre_r, centre_z))
    np.testing.assert_allclose(crossings[0], [centre_r + a, centre_r - a], atol=1e-6)
    nan = np.asarray(midplane_crossings(
        jnp.asarray(r[None, :]), jnp.asarray(z[None, :]),
        centre_r, centre_z, max_gap=1e-6))
    assert np.any(np.isnan(nan))
    label, xlabel = surface_label(r[None, :], z[None, :],
                                  np.array([centre_r]), np.array([centre_z]))
    assert "midplane" in xlabel
    assert np.isfinite(np.asarray(label)).any()


def test_logical_field_rejects_scalar_forms(toroid) -> None:
    with pytest.raises(ValueError, match="k must be 1 or 2"):
        logical_field(toroid, jnp.ones(toroid.n(0, False)), 0, False)


def test_logical_field_to_rz_and_section_figure(toroid) -> None:
    """Piola vs ``g^{-1}`` branches, then one short section on the donut torus."""
    x = jnp.array([0.4, 0.3, 0.2])
    f2 = logical_field(toroid, jnp.ones(toroid.n(2, True)), 2, True)
    f1 = logical_field(toroid, jnp.ones(toroid.n(1, False)), 1, False)
    b2, b1 = f2(x), f1(x)
    assert b2.shape == (3,) and b1.shape == (3,)
    assert jnp.all(jnp.isfinite(b2)) and jnp.all(jnp.isfinite(b1))

    uv = jnp.array([[0.2, 0.0], [0.0, 0.3]])
    r_phys, z_phys = to_RZ(toroid, uv, 0.0)
    assert r_phys.shape == (2,) and z_phys.shape == (2,)
    assert float(jnp.min(r_phys)) > 0.0

    from mrx.initial_conditions import project_reference_two_form

    iota_f, dphi = make_profiles(IOTA0, IOTA1, IOTA_EXP, FLUX_EXP)
    b, _ = project_reference_two_form(
        toroid, analytic_profile_form(iota_f, dphi, make_lambda([])))
    field = logical_field(toroid, b, 2, True)
    seeds = jnp.array([[0.15, 0.0], [0.4, 0.25]])
    ys, _ = trace(field, seeds, n_periods=4, steps_per_period=16,
                  saves_per_period=SAVES)
    axis = axis_track(ys, SAVES)
    r_sec, z_sec, a_r, a_z, *_ = section_RZ(toroid, ys, axis, SAVES, 0.0)
    assert r_sec.shape[0] == 2
    label, _ = surface_label(np.asarray(r_sec), np.asarray(z_sec),
                             np.asarray(a_r), np.asarray(a_z))
    assert np.asarray(label).shape[0] == 2

    fig, res = section_figure(
        toroid, b, nfp=1, plane=0.0, n_seeds=4, n_periods=4,
        steps_per_period=16, saves_per_period=SAVES, n_rays=2, title="toroid")
    assert "iota" in res and fig is not None
    plt.close(fig)
