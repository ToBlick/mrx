"""JAX-compatible Cartesian mappings backed by a GVEC state."""

from __future__ import annotations

from threading import Lock
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np


def make_gvec_cartesian_map(
    state: Any,
    *,
    nfp: int,
    flip_zeta: bool = False,
    flip_r: bool = False,
    fd_eps: float = 1.0e-7,
) -> Callable[[jax.Array], jax.Array]:
    """Return a differentiable logical-to-Cartesian map for a GVEC state.

    MRX coordinates are normalized to the unit cube. They are converted to
    GVEC coordinates as ``(rho, 2π theta, 2π zeta / nfp)``. GVEC evaluates the
    map through its Fortran bindings, so :func:`jax.pure_callback` bridges the
    evaluation into JAX. A finite-difference Jacobian supplies the custom JVP
    needed by differential-form pushforwards.

    Parameters
    ----------
    state
        Initialized or loadable :class:`gvec.State`.
    nfp
        Number of field periods represented by the normalized toroidal
        coordinate.
    flip_zeta
        Replace normalized ``zeta`` by ``1 - zeta`` before GVEC evaluation.
    flip_r
        Replace ``rho`` by ``1 - rho`` before GVEC evaluation.
    fd_eps
        Step in normalized MRX coordinates used for the Jacobian.

    Returns
    -------
    Callable[[jax.Array], jax.Array]
        Function mapping a logical point with shape ``(3,)`` to Cartesian
        coordinates with shape ``(3,)``.
    """
    if nfp <= 0:
        raise ValueError(f"nfp must be positive, got {nfp}")
    if fd_eps <= 0.0:
        raise ValueError(f"fd_eps must be positive, got {fd_eps}")

    callback_lock = Lock()
    result_spec = (
        jax.ShapeDtypeStruct((3,), jnp.float64),
        jax.ShapeDtypeStruct((3, 3), jnp.float64),
    )

    def evaluate_position(x: np.ndarray) -> np.ndarray:
        """Evaluate one normalized MRX point through the GVEC Fortran API."""
        point = np.asarray(x, dtype=np.float64).reshape(3).copy()
        if flip_r:
            point[0] = 1.0 - point[0]
        if flip_zeta:
            point[2] = 1.0 - point[2]

        rho = float(np.clip(point[0], 0.0, 1.0))
        theta = float(2.0 * np.pi * point[1])
        zeta = float((2.0 * np.pi / nfp) * point[2])
        coordinates = np.asarray([[rho], [theta], [zeta]], dtype=np.float64)

        x1 = float(state.evaluate_base_list_rtz_all("X1", coordinates)[0][0])
        x2 = float(state.evaluate_base_list_rtz_all("X2", coordinates)[0][0])
        position = state.evaluate_hmap_only(
            X1=np.asarray([x1]),
            X2=np.asarray([x2]),
            zeta=np.asarray([zeta]),
        )[0]
        return np.asarray(position, dtype=np.float64).reshape(3)

    def evaluate_with_jacobian(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate the map and its finite-difference Jacobian."""
        point = np.asarray(x, dtype=np.float64).reshape(3)
        with callback_lock:
            position = evaluate_position(point)
            jacobian = np.empty((3, 3), dtype=np.float64)
            for axis in range(3):
                left = point.copy()
                right = point.copy()
                if axis == 0 and point[axis] - fd_eps < 0.0:
                    right[axis] += fd_eps
                    jacobian[:, axis] = (
                        evaluate_position(right) - position
                    ) / fd_eps
                elif axis == 0 and point[axis] + fd_eps > 1.0:
                    left[axis] -= fd_eps
                    jacobian[:, axis] = (
                        position - evaluate_position(left)
                    ) / fd_eps
                else:
                    left[axis] -= fd_eps
                    right[axis] += fd_eps
                    jacobian[:, axis] = (
                        evaluate_position(right) - evaluate_position(left)
                    ) / (2.0 * fd_eps)
        return position, jacobian

    @jax.custom_jvp
    def mapping(x: jax.Array) -> jax.Array:
        point = jnp.asarray(x, dtype=jnp.float64).reshape(3)
        position, _ = jax.pure_callback(
            evaluate_with_jacobian,
            result_spec,
            point,
            vmap_method="sequential",
        )
        return position

    @mapping.defjvp
    def mapping_jvp(
        primals: tuple[jax.Array],
        tangents: tuple[jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        (point,) = primals
        (tangent,) = tangents
        point = jnp.asarray(point, dtype=jnp.float64).reshape(3)
        position, jacobian = jax.pure_callback(
            evaluate_with_jacobian,
            result_spec,
            point,
            vmap_method="sequential",
        )
        return position, jacobian @ tangent

    return mapping
