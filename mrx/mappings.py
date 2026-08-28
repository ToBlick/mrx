"""Analytic logical-to-physical maps and the :class:`SplineMap` wrapper for fitted ones."""
from typing import Any, Callable, Optional

import equinox as eqx
import jax.numpy as jnp
from jax.numpy import cos, pi, sin

from mrx.differential_forms import DifferentialForm, DiscreteFunction


class SplineMap(eqx.Module):
    """A logical-to-physical map represented in the scalar spline basis.

    ``coefficients``, ``extraction``, ``extraction_T`` and ``raw`` are dynamic
    pytree children, so ``SplineMap`` can be passed through ``jit`` /
    ``grad`` / ``vmap`` and its coefficients can be differentiated.
    ``basis_0`` is a static topology object and rides along as aux data.

    ``raw`` is ``E^T`` applied to the three Cartesian coefficient vectors,
    reshaped to the tensor-product grid, ``(3, n_r, n_t, n_z)``; it is what
    :meth:`__call__` evaluates, on the ``prod(p_d + 1)`` basis functions
    that are nonzero at the point.
    """

    coefficients: jnp.ndarray
    extraction: Any
    extraction_T: Optional[Any]
    basis_0: DifferentialForm = eqx.field(static=True)
    raw: jnp.ndarray

    def __init__(self, coefficients, extraction, extraction_T=None, basis_0=None):
        self.coefficients = coefficients
        self.extraction = extraction
        self.extraction_T = extraction_T
        self.basis_0 = basis_0
        coeffs = coefficients.reshape(3, -1)
        self.raw = (extraction.T @ coeffs.T).T.reshape((3,) + basis_0.shape[0])

    def with_coefficients(self, coefficients):
        """Return a new spline map with updated coefficients."""
        return SplineMap(
            coefficients=coefficients,
            extraction=self.extraction,
            extraction_T=self.extraction_T,
            basis_0=self.basis_0,
        )

    def __call__(self, x):
        return self.basis_0.bases[0].contract(self.raw, x)


def rotating_ellipse_map(eps: float = 0.33, kappa: float = 1.2, R0: float = 1.0, nfp: int = 3) -> Callable:
    """Rotating-ellipse map with ``nfp`` field periods.

    Args:
        eps: Minor radius (inverse aspect ratio).
        kappa: Elongation.
        R0: Major radius.
        nfp: Number of field periods.
    """
    if nfp <= 0:
        raise ValueError(f"nfp must be a positive integer, got {nfp}")
    if eps <= 0:
        raise ValueError(f"eps must be a positive number, got {eps}")

    def nu(zeta):
        return 1 + (1 - kappa) * cos(2 * pi * zeta * nfp)

    def F(x):
        r, θ, ζ = x
        if nfp > 0:
            ζ /= nfp  # only model one field period

        R = R0 + eps * nu(ζ) * r * cos(2 * pi * θ)
        if nfp > 0:
            Z = eps * r * nu(ζ + 0.5 / nfp) * sin(2 * pi * θ)
        else:
            Z = eps * nu(ζ) * r * sin(2 * pi * θ)
        return jnp.array([R * cos(2 * pi * ζ),
                          -R * sin(2 * pi * ζ),
                          Z])
    return F


def toroid_map(epsilon: float = 1/3, kappa: float = 1.0, R0: float = 1.0) -> Callable:
    """Simple axisymmetric toroidal map.

    ``F(r, θ, ζ) = (R cos 2πζ, -R sin 2πζ, ε κ r sin 2πθ)``
    where ``R = R0 + ε r cos 2πθ``.

    Args:
        epsilon: Minor radius.
        kappa: Elongation.
        R0: Major radius.
    """
    π = jnp.pi

    def F(x):
        r, θ, ζ = x
        R = R0 + epsilon * r * jnp.cos(2 * π * θ)
        return jnp.array([R * jnp.cos(2 * π * ζ),
                          -R * jnp.sin(2 * π * ζ),
                          epsilon * kappa * r * jnp.sin(2 * π * θ)])
    return F


def cylinder_map(a: float = 1.0, h: float = 1.0) -> Callable:
    """Cylinder map: ``F(r, χ, z) = (a r cos 2πχ, a r sin 2πχ, h z)``.

    Args:
        a: Cylinder radius.
        h: Cylinder height.
    """
    π = jnp.pi

    def F(x):
        r, χ, z = x
        return jnp.array([a * r * jnp.cos(2 * π * χ),
                          a * r * jnp.sin(2 * π * χ),
                          h * z])

    return F


def stellarator_map(R: DiscreteFunction, Z: DiscreteFunction, nfp: int = 3, flip_zeta: bool = False) -> Callable:
    """Stellarator map built from spline R(r,θ,ζ) and Z(r,θ,ζ).

    ``F(r, θ, ζ) = (R cos(2πζ/nfp), -R sin(2πζ/nfp), Z)``

    Args:
        R: Discrete spline for the cylindrical radius.
        Z: Discrete spline for the vertical coordinate.
        nfp: Number of field periods.
        flip_zeta: If ``True``, replace ζ with ``1 - ζ`` before evaluating.
    """
    π_nfp = 2 * jnp.pi / nfp

    def F(x):
        _, _, ζ = x
        if flip_zeta:
            ζ = 1.0 - ζ
        R_x = R(x)[0]
        return jnp.array([R_x * jnp.cos(π_nfp * ζ),
                          -R_x * jnp.sin(π_nfp * ζ),
                          Z(x)[0]])
    return F


def extend_map_nfp(Phi, nfp, sign=-1.0):
    """Extend a one-field-period map to the full ``nfp``-period torus.

    ``Phi(r, theta, zeta)`` covers one period with ``zeta in [0, 1]``; the
    returned map takes ``zeta in [0, 1]`` over the WHOLE device, evaluates
    ``Phi`` on the local period and rotates its output about the z axis by
    ``sign * 2 pi j / nfp`` for period ``j``. It agrees with ``Phi`` exactly on
    the first period -- the toroidal angle is the map's own, not recomputed
    from ``zeta`` (the previous form did that, and was wrong for any map whose
    ``atan2(Y, X)`` is not exactly linear in ``zeta``, e.g. a Cartesian spline
    map).

    ``sign`` is the direction of the toroidal angle with ``zeta``: ``-1`` for
    the ``(R cos, -R sin)`` convention of :func:`stellarator_map` and the GVEC
    maps with ``sign = -1`` (the codebase default), ``+1`` for a map whose
    angle increases with ``zeta``. Made for plotting whole stellarators; the
    solvers never see more than one period.
    """
    def Phi_full_fp(x):
        r, θ, ζ = x
        ξ = ζ * nfp
        j = jnp.floor(ξ)
        X, Y, Z = Phi(jnp.array([r, θ, ξ - j]))
        α = sign * 2 * jnp.pi * j / nfp
        c, s = jnp.cos(α), jnp.sin(α)
        return jnp.array([c * X - s * Y, s * X + c * Y, Z])

    return Phi_full_fp
