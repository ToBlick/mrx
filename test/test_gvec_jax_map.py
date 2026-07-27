"""Tests for the JAX bridge to GVEC mappings."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from mrx.gvec_jax_map import make_gvec_cartesian_map

jax.config.update("jax_enable_x64", True)


class _FakeGvecState:
    """Minimal GVEC-state API with an analytic cylindrical map."""

    def evaluate_base_list_rtz_all(
        self,
        quantity: str,
        coordinates: np.ndarray,
    ) -> list[np.ndarray]:
        """Return X1=rho or X2=theta for supplied logical coordinates."""
        value = coordinates[0] if quantity == "X1" else coordinates[1]
        return [np.asarray(value, dtype=np.float64)]

    def evaluate_hmap_only(
        self,
        *,
        X1: np.ndarray,
        X2: np.ndarray,
        zeta: np.ndarray,
    ) -> list[np.ndarray]:
        """Map cylindrical values to Cartesian positions."""
        position = np.vstack(
            [
                X1 * np.cos(zeta),
                X1 * np.sin(zeta),
                X2,
            ]
        )
        return [position]


def test_make_gvec_cartesian_map_value_and_jacobian() -> None:
    """The callback map and custom JVP match an analytic reference map."""
    mapping = make_gvec_cartesian_map(
        _FakeGvecState(),
        nfp=2,
        fd_eps=1.0e-6,
    )
    point = jnp.asarray([0.4, 0.25, 0.2])
    angle = np.pi * float(point[2])
    expected = np.asarray(
        [
            float(point[0]) * np.cos(angle),
            float(point[0]) * np.sin(angle),
            2.0 * np.pi * float(point[1]),
        ]
    )
    expected_jacobian = np.asarray(
        [
            [np.cos(angle), 0.0, -np.pi * float(point[0]) * np.sin(angle)],
            [np.sin(angle), 0.0, np.pi * float(point[0]) * np.cos(angle)],
            [0.0, 2.0 * np.pi, 0.0],
        ]
    )

    np.testing.assert_allclose(np.asarray(mapping(point)), expected, rtol=1.0e-10)
    np.testing.assert_allclose(
        np.asarray(jax.jacfwd(mapping)(point)),
        expected_jacobian,
        rtol=2.0e-6,
        atol=2.0e-6,
    )
