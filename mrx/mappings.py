"""Analytic logical-to-physical maps and the :class:`SplineMap` wrapper for fitted ones."""
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.numpy import cos, pi, sin

from mrx.differential_forms import DifferentialForm


class SplineMap(eqx.Module):
    """A logical-to-physical map represented in the scalar spline basis.

    ``raw`` is ``E^T`` applied to the three Cartesian coefficient vectors,
    reshaped to the tensor-product grid, ``(3, n_r, n_t, n_z)``: the one
    dynamic pytree leaf, what :meth:`__call__` evaluates on the ``prod(p_d
    + 1)`` basis functions that are nonzero at the point. ``basis_0`` is a
    static topology object and rides along as aux data.
    """

    basis_0: DifferentialForm = eqx.field(static=True)
    raw: jnp.ndarray

    def __init__(self, coefficients, extraction, basis_0):
        self.basis_0 = basis_0
        coeffs = coefficients.reshape(3, -1)
        self.raw = (extraction.T @ coeffs.T).T.reshape((3,) + basis_0.shape[0])

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
        ζ /= nfp  # only model one field period
        R = R0 + eps * nu(ζ) * r * cos(2 * pi * θ)
        Z = eps * r * nu(ζ + 0.5 / nfp) * sin(2 * pi * θ)
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


# ---------------------------------------------------------------------------
# Stellarator symmetry (the projector of PR #23, akaptano)
# ---------------------------------------------------------------------------
#
# Stellarator symmetry is ``(R, phi, Z) -> (R, -phi, -Z)``, in logical
# coordinates ``(r, theta, zeta) -> (r, -theta, -zeta)`` with ``R`` even and
# ``Z`` odd; on the Cartesian map ``(R cos 2 pi zeta, -R sin 2 pi zeta, Z)``
# it is the reflection ``S = diag(1, -1, -1)``. A map built from a symmetric
# series (a VMEC wout has only ``rmnc`` and ``zmns``) is symmetric to
# roundoff already; the projector makes it so exactly, and measures it.

#: ``S``: the Cartesian reflection stellarator symmetry induces.
STELLARATOR_REFLECTION = jnp.array([1.0, -1.0, -1.0])


def _reflection_permutation(n: int, p: int) -> np.ndarray:
    """Index permutation implementing ``x -> -x`` on a uniform periodic basis:
    ``B_j(-x) = B_{(p - 1 - j) mod n}(x)`` for the ``n`` uniform periodic
    B-splines of degree ``p``."""
    return (p - 1 - np.arange(n)) % n


def _is_uniform_periodic(basis) -> bool:
    """Whether ``basis`` is a uniform periodic B-spline basis on ``[0, 1]``."""
    if getattr(basis, "type", None) != "periodic":
        return False
    unique = np.asarray(basis.T[basis.p:basis.p + basis.n + 1])
    return bool(np.allclose(unique, np.linspace(0.0, 1.0, basis.n + 1), atol=1e-8, rtol=0.0))


def angular_reflection_allowed(basis_0: DifferentialForm) -> bool:
    """Whether the two angular axes of ``basis_0`` are uniform periodic
    bases, on which ``(theta, zeta) -> (-theta, -zeta)`` is an index
    permutation of the coefficients."""
    return all(_is_uniform_periodic(basis_0.Λ[axis]) for axis in (1, 2))


def stellarator_symmetric_scalar(raw: jnp.ndarray, basis_0: DifferentialForm, even: bool) -> jnp.ndarray:
    """Project the ``(n_r, n_t, n_z)`` coefficients of a scalar spline onto
    the part even (``R``) or odd (``Z``) under ``(theta, zeta) -> (-theta,
    -zeta)``, by the index permutation of the two uniform periodic angular
    axes (:func:`angular_reflection_allowed`)."""
    perm_t = _reflection_permutation(basis_0.Λ[1].n, basis_0.Λ[1].p)
    perm_z = _reflection_permutation(basis_0.Λ[2].n, basis_0.Λ[2].p)
    reflected = raw[:, perm_t, :][:, :, perm_z]
    return 0.5 * (raw + reflected) if even else 0.5 * (raw - reflected)


def stellarator_symmetry_defect(F: Callable, x: jnp.ndarray) -> jnp.ndarray:
    """``max |F(r, -t, -z) - S F(r, t, z)|`` over the logical points ``x``
    (``(n, 3)``): zero if and only if ``F`` is stellarator symmetric there."""
    def one(xi):
        r, t, z = xi
        return jnp.max(jnp.abs(F(jnp.array([r, -t, -z])) - STELLARATOR_REFLECTION * F(xi)))

    return jnp.max(jax.vmap(one)(jnp.atleast_2d(jnp.asarray(x))))
