from typing import Any, Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.numpy import cos, pi, sin

from mrx.differential_forms import DifferentialForm, DiscreteFunction
from mrx.geometry import (  # noqa: F401
    greville_interpolate_map,
    greville_interpolate_stellarator_map,
    interpolate_map,
)


class SplineMap(eqx.Module):
    """A logical-to-physical map represented in the scalar spline basis.

    ``coefficients``, ``extraction`` and ``extraction_T`` are dynamic
    pytree children, so ``SplineMap`` can be passed through ``jit`` /
    ``grad`` / ``vmap`` and its coefficients can be differentiated.
    ``basis_0`` is a static topology object and rides along as aux data.
    """

    coefficients: jnp.ndarray
    extraction: Any
    extraction_T: Optional[Any] = None
    basis_0: DifferentialForm = eqx.field(static=True, default=None)

    def with_coefficients(self, coefficients):
        """Return a new spline map with updated coefficients."""
        return SplineMap(
            coefficients=coefficients,
            extraction=self.extraction,
            extraction_T=self.extraction_T,
            basis_0=self.basis_0,
        )

    def __call__(self, x):
        ns = jnp.arange(self.basis_0.n)
        basis_vals = self.extraction @ jax.vmap(self.basis_0, (None, 0))(x, ns)
        basis_vals = jnp.ravel(basis_vals)

        coeffs = self.coefficients
        if coeffs.ndim == 1:
            n_coeff = basis_vals.shape[0]
            coeffs = coeffs.reshape(3, n_coeff)
        return coeffs @ basis_vals


def one_size_fits_all_map(epsilon: float = 0.33, kappa: float = 1.2, alpha: float = 0.0, R0: float = 1.0) -> Callable:
    """Cerfon et al. "One Size Fits All" map (arXiv:1004.3481).

    Args:
        epsilon: Inverse aspect ratio.
        kappa: Elongation.
        alpha: Poloidal tilt angle.
        R0: Major radius.
    """
    π = jnp.pi

    def x_t(t):
        return 1 + epsilon * jnp.cos(2 * π * t + alpha * jnp.sin(2 * π * t))

    def y_t(t):
        return epsilon * kappa * jnp.sin(2 * π * t)

    def _s_from_t(t):
        return jnp.arctan2(kappa * jnp.sin(2 * π * t),
                           jnp.cos(2 * π * t + alpha * jnp.sin(2 * π * t)))

    def s_from_t(t):
        return jnp.where(t > 0.5, _s_from_t(t) + 2 * π, _s_from_t(t))

    def a_from_t(t):
        return jnp.sqrt((x_t(t) - 1)**2 + y_t(t)**2)

    @jax.jit
    def F(x):
        r, χ, z = x
        return jnp.ravel(jnp.array(
            [(R0 + a_from_t(χ) * r * jnp.cos(s_from_t(χ))) * jnp.cos(2 * π * z),
             -(R0 + a_from_t(χ) * r * jnp.cos(s_from_t(χ))) * jnp.sin(2 * π * z),
             a_from_t(χ) * r * jnp.sin(s_from_t(χ))]))
    return F


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
        return jnp.array([R(x)[0] * jnp.cos(π_nfp * ζ),
                          -R(x)[0] * jnp.sin(π_nfp * ζ),
                          Z(x)[0]])
    return F


def approx_inverse_map(y: jnp.ndarray, eps: float, R0: float = 1.0) -> jnp.ndarray:
    """Approximate inverse of ``toroid_map`` for a circular cross-section.

    Args:
        y: Cartesian coordinates ``(X, Y, Z)``.
        eps: Minor radius (same as ``epsilon`` in :func:`toroid_map`).
        R0: Major radius.
    """
    X, Y, Z = y
    R = jnp.sqrt(X**2 + Y**2)
    ζ = (jnp.arctan2(-Y, X) / (2 * pi)) % 1.0
    r = jnp.sqrt(((R - R0) / eps)**2 + (Z / (eps))**2)
    θ = (jnp.arctan2(Z / (eps * r), (R - R0) / (eps * r)) / (2 * pi)) % 1.0
    return jnp.array([r, θ, ζ])


def invert_map(
        f: Callable, y_target: jnp.ndarray,
        x0_fn: Callable, tol: float = 1e-10, max_iter: int = 50) -> jnp.ndarray:
    """Invert ``f`` at ``y_target`` via Newton's method.

    Args:
        f: Map to invert.
        y_target: Target physical coordinates.
        x0_fn: Returns an initial guess ``x0`` given ``y_target``.
        tol: Convergence tolerance on the residual norm.
        max_iter: Maximum Newton iterations.
    """
    def cond_fn(state: tuple[jnp.ndarray, float, int]) -> jnp.ndarray:
        x, err, i = state
        return jnp.logical_and(err > tol, i < max_iter)

    def body_fn(state: tuple[jnp.ndarray, float, int]) -> tuple[jnp.ndarray, float, int]:
        x, _, i = state
        r = f(x) - y_target
        J = jax.jacobian(f)(x)
        dx = jnp.linalg.solve(J, -r)
        x_new = x + dx
        err = jnp.linalg.norm(r)
        return (x_new, err, i + 1)

    x0 = x0_fn(y_target)
    init_state = (x0, jnp.inf, 0)
    x_final, err_final, _ = jax.lax.while_loop(cond_fn, body_fn, init_state)
    return x_final


def extend_map_nfp(Phi, nfp):
    """Extend a single-field-period map to the full ``nfp``-period torus.

    Args:
        Phi: Map covering one field period, ``(r,θ,ζ) -> (x,y,z)`` with
            ``ζ ∈ [0, 1/nfp]``.
        nfp: Number of field periods.
    """
    def Phi_full_fp(x):
        r, θ, ζ = x  # now ζ ∈ [0, 1] should cover the FULL device
        π_nfp = 2 * jnp.pi / nfp
        ξ = ζ * nfp  # in [0, nfp]
        ζ_loc = ξ - jnp.floor(ξ)  # in [0, 1)
        x_loc = jnp.array([r, θ, ζ_loc])
        R = (Phi(x_loc)[0]**2 + Phi(x_loc)[1]**2)**0.5
        Z = Phi(x_loc)[2]
        φ_wedge = π_nfp * ζ_loc  # 0 → 2π/nfp
        φ_shift = 2 * jnp.pi * jnp.floor(ξ) / nfp
        φ = φ_wedge + φ_shift  # total toroidal angle
        return jnp.array([R * jnp.cos(φ), -R * jnp.sin(φ), Z])

    return Phi_full_fp
