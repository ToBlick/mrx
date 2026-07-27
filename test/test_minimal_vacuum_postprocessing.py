"""Synthetic tests for minimal-vacuum post-processing helpers."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import os

import pytest

from scripts.minimal_vacuum_problem.compare_push_u_to_simsopt import (
    _classify_resonant_orbit,
    _close_poloidal_slice,
    _exact_section_times,
    _identify_iota_resonances,
    _iota_profile_from_physical_pitch,
    _island_width_consistency_summary,
    _island_width_profile,
    _k2_algebraic_health,
    _local_shear_from_iota_profile,
    _logical_to_phi_zero_rz,
    _parse_fem_resolution_values,
    _parse_resolution_values,
    _parse_section_values,
    _periodic_difference,
    _periodic_section_intersections,
    _physical_phi_zero_slice,
    _physical_sections_to_logical,
    _physical_rotational_transform_profile,
    _plot_component_slices,
    _plot_fem_resolution_sweep,
    _plot_island_diagnostics,
    _plot_poincare_comparison,
    _plot_reference_island_zooms,
    _plot_resolution_sweep,
    _poincare_chunk_size,
    _poloidal_slice_points,
    _radial_error_profile,
    _resonant_normal_error_amplitudes,
    _resonant_normal_error_radial_profile,
    _rotational_transform_profile,
    _simsopt_iota_profile_from_pitch,
    _trace_mrx_poincare,
    _trapped_separatrix_summary,
    pendulum_island_width,
)
from scripts.minimal_vacuum_problem.fem_convergence_robust.aggregate_robust_islands import (
    _isotropic_l2_fit,
)
from scripts.minimal_vacuum_problem.fem_convergence_robust.manuscript_poincare_figure import (
    _panel_marker_style,
    plot_manuscript_poincare_figure,
)


def _synthetic_slice(zeta: float = 0.25) -> dict[str, object]:
    """Return a small, smooth poloidal slice for plotting tests."""
    rho = np.linspace(0.1, 1.0, 4)
    theta = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    rr, tt = np.meshgrid(rho, theta, indexing="ij")
    R = 1.0 + 0.25 * rr * np.cos(tt)
    Z = 0.25 * rr * np.sin(tt)
    base = np.stack([R, Z, R - 1.0], axis=-1)
    return {
        "zeta": zeta,
        "rho": rr,
        "theta": tt / (2.0 * np.pi),
        "R": R,
        "Z": Z,
        "B_mrx": base,
        "B_simsopt": 0.9 * base,
    }


class _ConstantVectorBasis:
    """Three constant Cartesian component basis functions."""

    n = 3

    def __call__(self, _point: jax.Array, index: jax.Array) -> jax.Array:
        """Return one constant vector component."""
        return jax.nn.one_hot(index, 3, dtype=jnp.float64)


class _SyntheticSequence:
    """Minimal sequence exposing a constant toroidal k=2 field."""

    basis_2 = _ConstantVectorBasis()
    e2_dbc = jnp.eye(3)


class _SyntheticHealthSequence:
    """Small exact operator model for k=2 algebraic-health tests."""

    def apply_mass_matrix(
        self,
        vector: jax.Array,
        _k: int,
        *,
        dirichlet: bool,
    ) -> jax.Array:
        """Apply identity mass matrices."""
        assert dirichlet
        return vector

    def apply_inverse_mass_matrix(
        self,
        vector: jax.Array,
        _k: int,
        *,
        dirichlet: bool,
        tol: float,
    ) -> jax.Array:
        """Apply exact identity inverse mass matrices."""
        assert dirichlet
        assert tol > 0.0
        return vector

    def apply_hodge_laplacian(
        self,
        vector: jax.Array,
        _k: int,
        *,
        dirichlet: bool,
    ) -> jax.Array:
        """Apply a diagonal synthetic Hodge Laplacian."""
        assert dirichlet
        return jnp.asarray([vector[0], 4.0 * vector[1]])

    def apply_derivative_matrix(
        self,
        vector: jax.Array,
        k: int,
        *,
        dirichlet_in: bool,
        dirichlet_out: bool,
        transpose: bool = False,
    ) -> jax.Array:
        """Apply divergence or transposed curl factors."""
        assert dirichlet_in and dirichlet_out
        if k == 2 and not transpose:
            return jnp.asarray([vector[0]])
        if k == 1 and transpose:
            return jnp.asarray([0.0, 2.0 * vector[1]])
        raise AssertionError("unexpected synthetic derivative")


class _ConstantSimsoptField:
    """Minimal SIMSOPT-like field returning a constant Cartesian vector."""

    def set_points(self, points: np.ndarray) -> None:
        """Store the requested evaluation points."""
        self.points = np.asarray(points)

    def B(self) -> np.ndarray:
        """Return a nonzero constant reference field."""
        return np.broadcast_to(
            np.asarray([0.0, 1.0, 0.0]),
            self.points.shape,
        )


def test_section_parsing_and_grid_shape() -> None:
    """Section values normalize periodically and grids have expected shape."""
    assert _parse_section_values("0, 0.25, 1.5") == [0.0, 0.25, 0.5]
    assert _parse_resolution_values("8x12, 16X24") == [(8, 12), (16, 24)]
    assert _parse_fem_resolution_values("4x8x4, 6X12X6") == [
        (4, 8, 4),
        (6, 12, 6),
    ]
    points, rho, theta = _poloidal_slice_points(1.25, 4, 6)
    assert points.shape == (24, 3)
    assert rho.shape == theta.shape == (4, 6)
    np.testing.assert_allclose(points[:, 2], 0.25)


def test_close_poloidal_slice_appends_first_column() -> None:
    """Closing a slice appends the first periodic theta column."""
    slab = _synthetic_slice()
    R, Z, field = _close_poloidal_slice(
        np.asarray(slab["R"]),
        np.asarray(slab["Z"]),
        np.asarray(slab["B_mrx"]),
    )
    assert R.shape == Z.shape == (4, 9)
    assert field.shape == (4, 9, 3)
    np.testing.assert_allclose(R[:, -1], R[:, 0])
    np.testing.assert_allclose(field[:, -1], field[:, 0])


def test_periodic_section_intersections_handles_zero_plane() -> None:
    """Periodic shifting detects repeated crossings through zeta zero."""
    toroidal = np.linspace(0.01, 3.99, 400)
    trajectory = np.stack(
        [
            np.full_like(toroidal, 0.5),
            (0.2 + 0.1 * toroidal) % 1.0,
            toroidal % 1.0,
        ],
        axis=1,
    )
    intersections = _periodic_section_intersections(
        trajectory,
        0.0,
        max_intersections=8,
    )
    assert intersections.shape[0] >= 3
    np.testing.assert_allclose(intersections[:, 2], 0.0)


def test_exact_mrx_crossings_and_transit_counts() -> None:
    """Straight toroidal lines hit each requested section once per transit."""
    np.testing.assert_allclose(_exact_section_times(0.0, 3), [1.0, 2.0, 3.0])
    np.testing.assert_allclose(_exact_section_times(0.25, 3), [0.25, 1.25, 2.25])
    sections, _, _, transit_counts = _trace_mrx_poincare(
        _SyntheticSequence(),
        jnp.asarray([0.0, 0.0, 1.0]),
        jax.jit(lambda point: point),
        [0.0, 0.25],
        nlines=2,
        turns=3,
        theta0=0.5,
        tol=1.0e-10,
    )
    assert transit_counts == [3, 3]
    assert all(points.shape == (3, 2) for points in sections[0.0])
    assert all(points.shape == (3, 2) for points in sections[0.25])


def test_radial_error_profile_resolves_shells() -> None:
    """Radial profiles report separate shell-wise relative errors."""
    rho = np.asarray([0.25, 0.25, 0.75, 0.75])
    reference = np.ones((4, 3))
    mrx = reference.copy()
    mrx[:2] += 0.1
    mrx[2:] += 0.2
    profile = _radial_error_profile(rho, mrx, reference)
    assert [record["rho"] for record in profile] == [0.25, 0.75]
    np.testing.assert_allclose(
        [record["relative_error_rms"] for record in profile],
        [0.1, 0.2],
    )


def test_k2_algebraic_health_splits_divergence_and_curl() -> None:
    """Health metrics use the dual residual norm and energy decomposition."""
    health = _k2_algebraic_health(
        _SyntheticHealthSequence(),
        jnp.asarray([1.0, 1.0]),
    )
    np.testing.assert_allclose(health["rayleigh_quotient"], 2.5)
    np.testing.assert_allclose(
        health["relative_residual_m2_inverse"],
        np.sqrt(8.5),
    )
    np.testing.assert_allclose(
        health["relative_divergence_energy"],
        np.sqrt(0.5),
    )
    np.testing.assert_allclose(
        health["relative_weak_curl_energy"],
        np.sqrt(2.0),
    )
    np.testing.assert_allclose(health["energy_identity_relative_error"], 0.0)


def test_iota_resonance_and_island_width_diagnostics() -> None:
    """Synthetic crossings recover iota=3/5 and a finite radial width."""
    crossings = np.arange(40, dtype=np.float64)
    seed_rho = np.asarray([0.35, 0.45, 0.55])
    slopes = [0.18, 0.20, 0.22]
    lines = [
        np.column_stack(
            [
                rho + 0.01 * np.sin(2.0 * np.pi * crossings / 5.0),
                (0.1 + slope * crossings) % 1.0,
            ]
        )
        for rho, slope in zip(seed_rho, slopes)
    ]
    profile = _rotational_transform_profile(lines, seed_rho, nfp=3)
    np.testing.assert_allclose(
        [item["iota"] for item in profile],
        [0.54, 0.60, 0.66],
        atol=1.0e-12,
    )
    resonances = _identify_iota_resonances(
        profile,
        nfp=3,
        rho_min=0.3,
        rho_max=0.6,
        max_poloidal_mode=8,
    )
    three_fifths = [
        item
        for item in resonances
        if item["toroidal_mode"] == 3 and item["poloidal_mode"] == 5
    ]
    assert len(three_fifths) == 1
    np.testing.assert_allclose(three_fifths[0]["rho"], 0.45)
    widths = _island_width_profile(lines, seed_rho)
    assert all(item["width_rho_q05_q95"] > 0.015 for item in widths)


def test_detrended_width_removes_smooth_surface_label_variation() -> None:
    """Low-mode radial variation is removed while a six-lobe signal remains."""
    theta = np.arange(240, dtype=np.float64) / 240.0
    smooth = 0.04 * np.cos(2.0 * np.pi * 2.0 * theta)
    island = 0.006 * np.cos(2.0 * np.pi * 6.0 * theta)
    line = np.column_stack([0.7 + smooth + island, theta])
    width = _island_width_profile([line], np.asarray([0.7]))[0]
    assert width["width_rho_q05_q95"] > 0.07
    np.testing.assert_allclose(
        width["detrended_width_rho_q05_q95"],
        np.quantile(island, 0.95) - np.quantile(island, 0.05),
        atol=1.0e-12,
    )


def test_resonant_phase_classifies_trapped_and_passing_orbits() -> None:
    """Concentrated six-fold phase is trapped; circulating phase is passing."""
    crossings = np.arange(120, dtype=np.float64)
    trapped_theta = 0.2 + 0.005 * np.sin(2.0 * np.pi * crossings / 9.0)
    passing_theta = (0.2 + 0.0137 * crossings) % 1.0
    trapped_line = np.column_stack(
        [0.7 + 0.01 * np.sin(2.0 * np.pi * crossings / 9.0), trapped_theta]
    )
    passing_line = np.column_stack([np.full(crossings.size, 0.76), passing_theta])
    assert _classify_resonant_orbit(trapped_line)["trapped"]
    assert not _classify_resonant_orbit(passing_line)["trapped"]
    summary = _trapped_separatrix_summary(
        [trapped_line, passing_line],
        np.asarray([[0.7, 0.2, 0.0], [0.76, 0.2, 0.0]]),
    )
    assert summary["trapped_line_count"] == 1
    assert summary["trapped_separatrix_width_rho"] > 0.015


def test_island_width_consistency_fits_finest_three_records() -> None:
    """The low-shear square-root model uses only the asymptotic three grids."""
    records = []
    for index, amplitude in enumerate([1.0e-5, 4.0e-5, 9.0e-5, 1.6e-4]):
        predictor = np.sqrt(amplitude / (6.0 * 0.02))
        records.append(
            {
                "ns": [4 + index, 8 + 2 * index, 4 + index],
                "island_diagnostics": {
                    "max_detrended_island_width_rho": 2.5 * predictor,
                    "resonances": [
                        {
                            "rho": 0.7,
                            "toroidal_mode": 3,
                            "poloidal_mode": 6,
                        }
                    ],
                    "resonant_normal_error": [
                        {
                            "toroidal_mode": 3,
                            "poloidal_mode": 6,
                            "normal_error_fourier_relative": amplitude,
                        }
                    ],
                    "iota_profile": [
                        {"rho": 0.6, "iota": 0.498},
                        {"rho": 0.7, "iota": 0.5},
                        {"rho": 0.8, "iota": 0.502},
                    ],
                },
            }
        )
    summary = _island_width_consistency_summary(records)
    np.testing.assert_allclose(summary["constant_C"], 2.5, rtol=1.0e-12)
    assert summary["fit_points_finest_three"] == 3
    np.testing.assert_allclose(
        summary["relative_rms_residual_finest_three"],
        0.0,
        atol=1.0e-12,
    )


def test_physical_iota_uses_full_torus_crossing_cadence() -> None:
    """SIMSOPT physical-plane hits advance once per full toroidal transit."""
    crossings = np.arange(30, dtype=np.float64)
    angle = -2.0 * np.pi * 0.4 * crossings
    line = np.column_stack([1.0 + 0.1 * np.cos(angle), 0.1 * np.sin(angle)])
    profile = _physical_rotational_transform_profile(
        [line],
        np.asarray([0.5]),
        np.asarray([1.0, 0.0]),
        nfp=3,
    )
    np.testing.assert_allclose(profile[0]["iota"], 0.4, atol=1.0e-12)
    np.testing.assert_allclose(
        profile[0]["iota_per_field_period"],
        0.4 / 3.0,
        atol=1.0e-12,
    )


def test_physical_sections_map_back_to_logical_coordinates() -> None:
    """Physical section hits recover their sampled logical radius and angle."""
    slab = _synthetic_slice(zeta=0.0)
    radial_indices = np.asarray([1, 2, 3])
    theta_indices = np.asarray([1, 3, 6])
    physical = np.column_stack(
        [
            np.asarray(slab["R"])[radial_indices, theta_indices],
            np.asarray(slab["Z"])[radial_indices, theta_indices],
        ]
    )
    logical = _physical_sections_to_logical([physical], slab)[0]
    np.testing.assert_allclose(
        logical[:, 0],
        np.asarray(slab["rho"])[radial_indices, theta_indices],
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        logical[:, 1],
        np.asarray(slab["theta"])[radial_indices, theta_indices],
        atol=1.0e-12,
    )


def test_resonant_normal_error_uses_batched_covector_solve() -> None:
    """Normal-error Fourier analysis handles batched map Jacobians."""
    amplitudes = _resonant_normal_error_amplitudes(
        _SyntheticSequence(),
        jnp.asarray([1.0, 0.0, 0.0]),
        jax.jit(lambda point: point),
        _ConstantSimsoptField(),
        [
            {
                "rho": 0.5,
                "iota": 0.5,
                "toroidal_mode": 3,
                "poloidal_mode": 6,
                "field_period_mode": 1,
                "sample_mismatch": 0.0,
            }
        ],
        mrx_scale=1.0,
        ntheta=8,
        nzeta=8,
    )
    assert len(amplitudes) == 1
    assert np.isfinite(amplitudes[0]["normal_error_fourier_relative"])


def test_resonant_normal_error_radial_profile_preserves_radii() -> None:
    """The saved-vector mode projection evaluates every requested surface."""
    rho_values = np.asarray([0.3, 0.55, 0.8])
    profile = _resonant_normal_error_radial_profile(
        _SyntheticSequence(),
        jnp.asarray([1.0, 0.0, 0.0]),
        jax.jit(lambda point: point),
        _ConstantSimsoptField(),
        rho_values,
        mrx_scale=1.0,
        ntheta=8,
        nzeta=8,
    )
    np.testing.assert_allclose(
        [float(item["rho"]) for item in profile],
        rho_values,
    )
    assert all(
        np.isfinite(float(item["normal_error_fourier_relative"]))
        for item in profile
    )


def test_reference_island_zoom_plots_write_files(tmp_path: Path) -> None:
    """Matched SIMSOPT/MRX logical zooms write both PNG artifacts."""
    crossings = np.arange(40, dtype=np.float64)
    seeds = np.asarray(
        [[0.68, 0.5, 0.0], [0.72, 0.55, 0.0]],
        dtype=np.float64,
    )
    lines = [
        np.column_stack(
            [
                seed[0] + 0.01 * np.sin(2.0 * np.pi * crossings / 6.0),
                (seed[1] + 0.01 * crossings) % 1.0,
            ]
        )
        for seed in seeds
    ]
    reference_path, comparison_path = _plot_reference_island_zooms(
        lines,
        lines,
        seeds,
        tmp_path,
        rho_min=0.64,
        rho_max=0.78,
        dpi=50,
    )
    assert reference_path.is_file()
    assert comparison_path.is_file()


def test_postprocessing_plot_files_are_created(tmp_path: Path) -> None:
    """Component and matched Poincare plotting helpers write PNG files."""
    slab = _synthetic_slice()
    component_paths = _plot_component_slices(
        [slab],
        tmp_path,
        mrx_scale=1.1,
        dpi=50,
    )
    assert len(component_paths) == 1
    assert component_paths[0].is_file()

    points_mrx = np.asarray([[0.9, -0.1], [1.1, 0.1]])
    points_sim = np.asarray([[0.91, -0.09], [1.09, 0.09]])
    poincare_path = _plot_poincare_comparison(
        {0.25: [points_mrx]},
        {0.25: [points_sim]},
        [slab],
        tmp_path,
        dpi=50,
    )
    assert poincare_path.is_file()

    resolution_paths = _plot_resolution_sweep(
        [
            {
                "nrho": 8,
                "ntheta": 12,
                "points_per_section": 96,
                "total_points": 384,
                "aligned_rel_l2": 0.03,
                "mrx_evaluation_seconds": 1.0,
            },
            {
                "nrho": 16,
                "ntheta": 24,
                "points_per_section": 384,
                "total_points": 1536,
                "aligned_rel_l2": 0.02,
                "mrx_evaluation_seconds": 3.8,
            },
        ],
        tmp_path,
        dpi=50,
    )
    assert all(path.is_file() for path in resolution_paths)

    fem_paths = _plot_fem_resolution_sweep(
        [
            {
                "ns": [4, 8, 4],
                "n2_dbc": 128,
                "aligned_rel_l2": 0.04,
                "solve_seconds": 2.0,
            },
            {
                "ns": [6, 12, 6],
                "n2_dbc": 512,
                "aligned_rel_l2": 0.02,
                "solve_seconds": 8.0,
            },
        ],
        tmp_path,
        dpi=50,
    )
    assert all(path.is_file() for path in fem_paths)

    diagnostic_records = [
        {
            "ns": [4, 8, 4],
            "aligned_rel_l2": 0.04,
            "island_diagnostics": {
                "iota_profile": [
                    {"rho": 0.3, "iota": 0.55},
                    {"rho": 0.6, "iota": 0.65},
                ],
                "max_island_width_rho": 0.02,
                "max_resonant_normal_error_relative": 0.01,
            },
        },
        {
            "ns": [6, 12, 6],
            "aligned_rel_l2": 0.02,
            "island_diagnostics": {
                "iota_profile": [
                    {"rho": 0.3, "iota": 0.56},
                    {"rho": 0.6, "iota": 0.64},
                ],
                "max_island_width_rho": 0.014,
                "max_resonant_normal_error_relative": 0.005,
            },
        },
    ]
    diagnostic_paths = _plot_island_diagnostics(
        diagnostic_records,
        [{"rho": 0.3, "iota": 0.55}, {"rho": 0.6, "iota": 0.64}],
        tmp_path,
        dpi=50,
    )
    assert all(path.is_file() for path in diagnostic_paths)


def _circular_torus_map(
    point: jax.Array,
    *,
    r0: float = 1.0,
    a: float = 0.25,
    twist: float = 0.0,
) -> jax.Array:
    """Analytic circular-torus map with an optional ``phi(zeta)`` twist."""
    rho = jnp.clip(point[0], 1.0e-8, 1.0 - 1.0e-8)
    theta = point[1] % 1.0
    zeta = point[2] % 1.0
    major = r0 + a * rho * jnp.cos(2.0 * jnp.pi * theta)
    z = a * rho * jnp.sin(2.0 * jnp.pi * theta)
    phi = 2.0 * jnp.pi * zeta + twist * rho * jnp.sin(2.0 * jnp.pi * theta)
    return jnp.asarray(
        [major * jnp.cos(phi), major * jnp.sin(phi), z],
        dtype=jnp.float64,
    )


class _PitchPushforwardField:
    """Stub SIMSOPT field returning a prescribed constant logical pitch."""

    def __init__(
        self,
        map_fn,
        *,
        pitch: float,
        r0: float = 1.0,
        a: float = 0.25,
    ) -> None:
        self.map_fn = map_fn
        self.pitch = float(pitch)
        self.r0 = float(r0)
        self.a = float(a)
        self.points = np.zeros((0, 3), dtype=np.float64)
        self._jacobian = jax.jit(jax.jacfwd(map_fn))

    def set_points(self, points: np.ndarray) -> None:
        """Store the requested evaluation points."""
        self.points = np.asarray(points, dtype=np.float64).reshape(-1, 3)

    def B(self) -> np.ndarray:
        """Push a constant logical pitch field through the analytic Jacobian."""
        values = np.zeros_like(self.points)
        for index, xyz in enumerate(self.points):
            radius = float(np.hypot(xyz[0], xyz[1]))
            height = float(xyz[2])
            rho = float(
                np.hypot((radius - self.r0) / self.a, height / self.a)
            )
            rho = float(np.clip(rho, 1.0e-8, 1.0 - 1.0e-8))
            theta = float(
                np.arctan2(height / self.a, (radius - self.r0) / self.a)
                / (2.0 * np.pi)
                % 1.0
            )
            zeta = float(np.arctan2(xyz[1], xyz[0]) / (2.0 * np.pi) % 1.0)
            logical = jnp.asarray([rho, theta, zeta], dtype=jnp.float64)
            jacobian = np.asarray(self._jacobian(logical), dtype=np.float64)
            logical_field = np.asarray([0.0, self.pitch, 1.0], dtype=np.float64)
            values[index] = jacobian @ logical_field
        return values


def test_pendulum_island_width_closed_form_and_relative_law() -> None:
    """Pendulum width matches the closed form and relative sqrt-amplitude law."""
    amplitude = 2.5e-5
    shear = -0.4
    expected = 4.0 * np.sqrt(amplitude / (2.0 * np.pi * abs(shear)))
    width = pendulum_island_width(amplitude, shear)
    np.testing.assert_allclose(width, expected, rtol=1.0e-12)
    np.testing.assert_allclose(
        pendulum_island_width(amplitude, -shear),
        width,
        rtol=1.0e-12,
    )
    width_half = pendulum_island_width(0.25 * amplitude, shear)
    np.testing.assert_allclose(width_half / width, 0.5, rtol=1.0e-12)
    # Zero shear is floored rather than raising, so the width remains finite.
    assert np.isfinite(pendulum_island_width(amplitude, 0.0))
    assert pendulum_island_width(amplitude, 0.0) > width


def test_local_shear_from_iota_profile_linear_and_windowed() -> None:
    """Local shear recovers an exact slope and respects the half-window."""
    linear = [{"rho": 0.5 + 0.02 * i, "iota": 0.4 + 0.3 * (0.5 + 0.02 * i)} for i in range(11)]
    np.testing.assert_allclose(
        _local_shear_from_iota_profile(linear, 0.6, half_window=4),
        0.3,
        rtol=1.0e-12,
    )
    piecewise = [
        {"rho": float(rho), "iota": float(0.2 * rho if rho < 0.6 else 0.8 * rho)}
        for rho in np.linspace(0.4, 0.8, 21)
    ]
    left = _local_shear_from_iota_profile(piecewise, 0.5, half_window=2)
    right = _local_shear_from_iota_profile(piecewise, 0.7, half_window=2)
    np.testing.assert_allclose(left, 0.2, atol=5.0e-3)
    np.testing.assert_allclose(right, 0.8, atol=5.0e-3)
    with pytest.raises(RuntimeError, match="insufficient iota samples"):
        _local_shear_from_iota_profile(
            [{"rho": 0.5, "iota": 0.5}, {"rho": 0.6, "iota": 0.55}],
            0.55,
            half_window=4,
        )


def test_iota_profile_from_physical_pitch_recovers_constant_pitch() -> None:
    """Flux-surface-averaged pitch recovers a prescribed constant logical pitch."""
    pitch = 0.17
    nfp = 3

    def map_fn(point: jax.Array) -> jax.Array:
        return _circular_torus_map(point, twist=0.0)

    seed_rho = np.asarray([0.3, 0.5, 0.7], dtype=np.float64)
    ntheta, nzeta = 8, 8
    theta = np.arange(ntheta, dtype=np.float64) / ntheta
    zeta = np.arange(nzeta, dtype=np.float64) / nzeta
    theta_grid, zeta_grid = np.meshgrid(theta, zeta, indexing="ij")
    jacobian_fn = jax.jit(jax.jacfwd(map_fn))
    physical_blocks = []
    for rho in seed_rho:
        logical = np.stack(
            [
                np.full(theta_grid.size, float(rho)),
                theta_grid.ravel(),
                zeta_grid.ravel(),
            ],
            axis=1,
        )
        jacobians = np.asarray(
            jax.lax.map(
                jacobian_fn,
                jnp.asarray(logical, dtype=jnp.float64),
                batch_size=64,
            ),
            dtype=np.float64,
        )
        logical_field = np.broadcast_to(
            np.asarray([0.0, pitch, 1.0], dtype=np.float64),
            (logical.shape[0], 3),
        )
        physical_blocks.append(
            np.einsum("nij,nj->ni", jacobians, logical_field)
        )
    physical_field = np.concatenate(physical_blocks, axis=0)
    profile = _iota_profile_from_physical_pitch(
        map_fn,
        physical_field,
        seed_rho,
        nfp=nfp,
        ntheta=ntheta,
        nzeta=nzeta,
    )
    assert len(profile) == seed_rho.size
    for item in profile:
        np.testing.assert_allclose(item["iota"], nfp * pitch, rtol=1.0e-8)
        np.testing.assert_allclose(item["pitch_std"], 0.0, atol=1.0e-10)


def test_simsopt_iota_profile_from_pitch_uses_stub_field() -> None:
    """SIMSOPT pitch helper routes a stub field through the shared pitch estimator."""
    pitch = 0.11
    nfp = 2

    def map_fn(point: jax.Array) -> jax.Array:
        return _circular_torus_map(point, twist=0.0)

    field = _PitchPushforwardField(map_fn, pitch=pitch)
    profile = _simsopt_iota_profile_from_pitch(
        map_fn,
        field,
        np.asarray([0.4, 0.6], dtype=np.float64),
        nfp=nfp,
        ntheta=6,
        nzeta=6,
    )
    assert len(profile) == 2
    for item in profile:
        np.testing.assert_allclose(item["iota"], nfp * pitch, rtol=5.0e-3)


def test_physical_phi_zero_slice_and_embedding_round_trip() -> None:
    """Newton-corrected phi=0 slice is flat and embeds logical nodes exactly."""

    def map_fn(point: jax.Array) -> jax.Array:
        return _circular_torus_map(point, twist=0.08)

    physical_slice = _physical_phi_zero_slice(map_fn, nrho=6, ntheta=12)
    assert float(np.max(np.abs(physical_slice["phi"]))) < 1.0e-12
    # Without the correction, zeta=0 is not the cylindrical plane.
    logical, _, _ = _poloidal_slice_points(0.0, 6, 12)
    mapped = np.asarray(
        jax.lax.map(
            map_fn,
            jnp.asarray(logical, dtype=jnp.float64),
            batch_size=64,
        ),
        dtype=np.float64,
    )
    phi_uncorrected = np.arctan2(mapped[:, 1], mapped[:, 0])
    assert float(np.max(np.abs(phi_uncorrected))) > 1.0e-3

    nodes = np.column_stack(
        [
            np.asarray(physical_slice["rho"]).ravel(),
            np.asarray(physical_slice["theta"]).ravel(),
        ]
    )
    embedded = _logical_to_phi_zero_rz([nodes], physical_slice)[0]
    expected = np.column_stack(
        [
            np.asarray(physical_slice["R"]).ravel(),
            np.asarray(physical_slice["Z"]).ravel(),
        ]
    )
    np.testing.assert_allclose(embedded, expected, atol=1.0e-12)
    np.testing.assert_allclose(
        _periodic_difference(np.asarray([0.1]), np.asarray([0.9]))[0],
        0.2,
        atol=1.0e-12,
    )


def test_poincare_chunk_size_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Chunk size defaults to min(8, n) and honors MRX_POINCARE_CHUNK."""
    monkeypatch.delenv("MRX_POINCARE_CHUNK", raising=False)
    assert _poincare_chunk_size(20) == 8
    assert _poincare_chunk_size(3) == 3
    monkeypatch.setenv("MRX_POINCARE_CHUNK", "1")
    assert _poincare_chunk_size(20) == 1
    monkeypatch.setenv("MRX_POINCARE_CHUNK", "100")
    assert _poincare_chunk_size(20) == 20
    monkeypatch.setenv("MRX_POINCARE_CHUNK", "  ")
    assert _poincare_chunk_size(20) == 8
    assert os.environ.get("MRX_POINCARE_CHUNK") == "  "


def test_manuscript_marker_style_scales_with_density() -> None:
    """Sparse panels get larger markers than dense panels; overrides win."""
    sparse_size, sparse_alpha = _panel_marker_style(5_000)
    dense_size, dense_alpha = _panel_marker_style(200_000)
    assert 0.05 <= dense_size < sparse_size <= 1.2
    assert sparse_alpha == pytest.approx(0.55)
    assert dense_alpha == pytest.approx(0.55)
    override_size, override_alpha = _panel_marker_style(
        200_000, marker_size=0.8, alpha=0.4
    )
    assert override_size == pytest.approx(0.8)
    assert override_alpha == pytest.approx(0.4)


def test_isotropic_l2_fit_recovers_synthetic_order() -> None:
    """Power-law fit recovers ``L2 = C h^4`` on an isotropic ladder."""
    records = []
    for nr in (4, 5, 6, 7, 8, 9, 10):
        h = 1.0 / float(nr)
        records.append(
            {
                "label": f"{nr}x{2 * nr}x{nr}",
                "ns": [nr, 2 * nr, nr],
                "n2_dbc": nr * (2 * nr) * nr,
                "aligned_metrics": {"rel_l2_aligned": 3.0 * h**4},
                "reliability": {"quarantined_nullspace": False},
            }
        )
    fit = _isotropic_l2_fit(records)
    assert fit["order_p"] == pytest.approx(4.0, abs=1e-9)
    assert fit["r_squared"] == pytest.approx(1.0, abs=1e-12)
    assert len(fit["labels"]) == 7


def test_manuscript_poincare_figure_writes_pdf_and_png(tmp_path: Path) -> None:
    """Manuscript figure writer emits both PDF and PNG from synthetic caches."""
    crossings = np.arange(30, dtype=np.float64)
    panels = []
    for title, center in (
        ("SIMSOPT", 0.68),
        ("MRX 8x32x8", 0.69),
        ("MRX 8x36x8", 0.70),
    ):
        lines = [
            np.column_stack(
                [
                    center + 0.01 * np.sin(2.0 * np.pi * crossings / 6.0 + phase),
                    (0.1 * phase + 0.02 * crossings) % 1.0,
                ]
            )
            for phase in (0.0, 0.5)
        ]
        panels.append((title, lines))

    def map_fn(point: jax.Array) -> jax.Array:
        return _circular_torus_map(point, twist=0.05)

    physical_slice = _physical_phi_zero_slice(map_fn, nrho=5, ntheta=10)
    pdf_path, png_path = plot_manuscript_poincare_figure(
        panels,
        physical_slice,
        tmp_path,
        stem="manuscript_test",
        dpi=40,
        marker_size=1.0,
    )
    assert pdf_path.is_file()
    assert png_path.is_file()
    assert pdf_path.suffix == ".pdf"
    assert png_path.suffix == ".png"
