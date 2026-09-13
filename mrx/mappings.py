"""Analytic logical-to-physical maps and the :class:`SplineMap` wrapper for fitted ones."""
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.numpy import cos, pi, sin

from mrx.differential_forms import DifferentialForm, DiscreteFunction

#: Cartesian reflection ``(X, Y, Z) -> (X, -Y, -Z)`` induced by stellarator
#: symmetry ``(R, phi, Z) -> (R, -phi, -Z)`` under the GVEC convention
#: ``(R cos 2 pi zeta, -R sin 2 pi zeta, Z)``, for which ``phi = -2 pi zeta``.
STELLARATOR_REFLECTION = jnp.array([1.0, -1.0, -1.0])


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

    def __init__(self, coefficients, extraction, basis_0, stellarator_symmetric: bool = False):
        self.basis_0 = basis_0
        coeffs = coefficients.reshape(3, -1)
        raw = (extraction.T @ coeffs.T).T.reshape((3,) + basis_0.shape[0])
        if stellarator_symmetric:
            raw = stellarator_symmetric_coefficients(raw, basis_0)
        self.raw = raw

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


def one_size_fits_all_map(
        epsilon: float = 0.33, kappa: float = 1.2,
        alpha: float = 0.0, R0: float = 1.0) -> Callable:
    """Cerfon et al. "One Size Fits All" map (arXiv:1004.3481).

    Args:
        epsilon: Inverse aspect ratio.
        kappa: Elongation.
        alpha: Poloidal tilt angle.
        R0: Major radius.

    Returns:
        Logical-to-physical map ``(r, chi, z) -> (X, Y, Z)`` in the GVEC
        convention ``(R cos 2 pi z, -R sin 2 pi z, Z)``.
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


def stellarator_map(
        R: DiscreteFunction, Z: DiscreteFunction,
        nfp: int = 3, flip_zeta: bool = False) -> Callable:
    """Stellarator map built from spline ``R(r, theta, zeta)`` and ``Z``.

    ``F(r, theta, zeta) = (R cos(2 pi zeta / nfp), -R sin(2 pi zeta / nfp), Z)``.

    Args:
        R: Discrete spline for the cylindrical radius. Only the first
            component ``R(x)[0]`` is used.
        Z: Discrete spline for the vertical coordinate. Only ``Z(x)[0]``
            is used.
        nfp: Number of field periods.
        flip_zeta: If ``True``, replace ``zeta`` with ``1 - zeta`` before
            evaluating.

    Returns:
        Logical-to-physical map in the GVEC convention.
    """
    if nfp <= 0:
        raise ValueError(f"nfp must be a positive integer, got {nfp}")
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

    Returns:
        Logical coordinates ``(r, theta, zeta)`` in ``[0, 1]^3`` (angles
        wrapped).
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

    Returns:
        Logical coordinates at which ``f`` equals ``y_target``.
    """
    def cond_fn(state: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
        x, err, i = state
        return jnp.logical_and(err > tol, i < max_iter)

    def body_fn(state: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
                ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        x, _, i = state
        r = f(x) - y_target
        J = jax.jacobian(f)(x)
        dx = jnp.linalg.solve(J, -r)
        x_new = x + dx
        err = jnp.linalg.norm(r)
        return (x_new, err, i + 1)

    x0 = x0_fn(y_target)
    init_state = (x0, jnp.array(jnp.inf), jnp.array(0))
    x_final, _, _ = jax.lax.while_loop(cond_fn, body_fn, init_state)
    return x_final


def extend_map_nfp(Phi: Callable, nfp: int) -> Callable:
    """Extend a single-field-period map to the full ``nfp``-period torus.

    Args:
        Phi: Map covering one field period, ``(r, theta, zeta) -> (x, y, z)``
            with ``zeta`` in ``[0, 1]`` corresponding to one wedge of angle
            ``2 pi / nfp``.
        nfp: Number of field periods. Must be a positive integer.

    Returns:
        A map whose input ``zeta`` in ``[0, 1]`` covers the full device.
    """
    if nfp <= 0:
        raise ValueError(f"nfp must be a positive integer, got {nfp}")

    def Phi_full_fp(x):
        r, θ, ζ = x
        π_nfp = 2 * jnp.pi / nfp
        ξ = ζ * nfp
        ζ_loc = ξ - jnp.floor(ξ)
        x_loc = jnp.array([r, θ, ζ_loc])
        R = (Phi(x_loc)[0]**2 + Phi(x_loc)[1]**2)**0.5
        Z = Phi(x_loc)[2]
        φ_wedge = π_nfp * ζ_loc
        φ_shift = 2 * jnp.pi * jnp.floor(ξ) / nfp
        φ = φ_wedge + φ_shift
        return jnp.array([R * jnp.cos(φ), -R * jnp.sin(φ), Z])

    return Phi_full_fp


def stellarator_symmetrize(F: Callable) -> Callable:
    """Project a map onto the stellarator-symmetric subspace.

    Returns ``(F(r, t, z) + S F(r, -t, -z)) / 2`` with
    ``S = diag(1, -1, -1)``, the reflection induced on
    ``(R cos 2 pi z, -R sin 2 pi z, Z)`` by the physical symmetry
    ``(R, phi, Z) -> (R, -phi, -Z)``.

    Args:
        F: Logical-to-physical map ``(r, theta, zeta) -> (X, Y, Z)``.

    Returns:
        The symmetrised map. Already-symmetric maps are fixed points.
    """
    S = STELLARATOR_REFLECTION

    def F_sym(x):
        r, t, z = x
        return 0.5 * (F(x) + S * F(jnp.array([r, -t, -z])))

    return F_sym


def stellarator_symmetry_defect(F: Callable, x: jnp.ndarray) -> jnp.ndarray:
    """Maximum stellarator-symmetry residual of ``F`` on the points ``x``.

    The residual at a point is ``|F(r, -t, -z) - S F(r, t, z)|`` with
    ``S = diag(1, -1, -1)``. Zero if and only if ``F`` is stellarator
    symmetric on those points.

    Args:
        F: Logical-to-physical map.
        x: One logical point of shape ``(3,)`` or a batch of shape ``(n, 3)``.

    Returns:
        A scalar: the maximum absolute residual over the batch.
    """
    S = STELLARATOR_REFLECTION
    pts = jnp.atleast_2d(jnp.asarray(x))

    def _one(xi):
        r, t, z = xi
        return jnp.max(jnp.abs(F(jnp.array([r, -t, -z])) - S * F(xi)))

    return jnp.max(jax.vmap(_one)(pts))


def extend_map_half_period(F_half: Callable, nfp: int = 1) -> Callable:
    """Extend a half-field-period map to the full period by reflection.

    The input map's ``zeta`` interval ``[0, 1]`` is one field period. For
    ``zeta <= 1/2`` the result equals ``F_half``; for ``zeta > 1/2`` it
    is ``S F_half(r, -theta, -zeta)`` with ``S = diag(1, -1, -1)``. An
    already-symmetric ``F_half`` is reproduced on the whole period.

    Args:
        F_half: Map defined on (at least) the first half-period
            ``zeta in [0, 1/2]``. Periodic evaluation at ``-zeta`` is
            used for the reflected half.
        nfp: Number of field periods of the underlying device. Must be
            a positive integer; the fold is at half a field period in
            the map's own ``zeta``.

    Returns:
        A map defined on the full period ``zeta in [0, 1]``.
    """
    if nfp <= 0:
        raise ValueError(f"nfp must be a positive integer, got {nfp}")
    S = STELLARATOR_REFLECTION

    def F(x):
        r, t, z = x
        z_mod = jnp.mod(z, 1.0)
        x_plus = jnp.array([r, t, z_mod])
        x_minus = jnp.array([r, -t, -z_mod])
        return jnp.where(z_mod <= 0.5, F_half(x_plus), S * F_half(x_minus))

    return F


def _reflection_permutation(n: int, p: int) -> jnp.ndarray:
    """Index permutation implementing ``x -> -x`` on a uniform periodic basis.

    ``B_j(-x) = B_{(p - 1 - j) mod n}(x)`` for every uniform periodic
    B-spline basis of ``n`` functions of degree ``p``.
    """
    return (p - 1 - jnp.arange(n)) % n


def _is_uniform_periodic(basis) -> bool:
    """Whether ``basis`` is a uniform periodic B-spline on ``[0, 1]``."""
    if getattr(basis, "type", None) != "periodic":
        return False
    unique = basis.T[basis.p:basis.p + basis.n + 1]
    expected = jnp.linspace(0.0, 1.0, basis.n + 1, dtype=unique.dtype)
    return bool(jnp.allclose(unique, expected, atol=1e-8, rtol=0.0))


def stellarator_symmetric_coefficients(
        coeffs: jnp.ndarray, basis_0: DifferentialForm) -> jnp.ndarray:
    """Symmetrise ``(3, n_r, n_t, n_z)`` spline coefficients by permutation.

    Reflection ``(theta, zeta) -> (-theta, -zeta)`` acts on each uniform
    periodic angular axis as ``perm[j] = (p - 1 - j) mod n``. The
    Cartesian components are then even (``X``) or odd (``Y``, ``Z``)
    under ``S = diag(1, -1, -1)``, which is the coefficient-space form of
    :func:`stellarator_symmetrize`.

    Args:
        coeffs: Coefficient tensor of shape ``(3, n_r, n_t, n_z)``.
        basis_0: Scalar spline space whose angular axes (1 and 2) must
            be uniform and periodic.

    Returns:
        Symmetrised coefficients of the same shape.

    Raises:
        ValueError: If an angular axis is not periodic with uniform knots,
            or if ``coeffs`` does not match ``basis_0.shape[0]``.
    """
    expected = (3,) + tuple(basis_0.shape[0])
    if tuple(coeffs.shape) != expected:
        raise ValueError(
            f"coeffs must have shape {expected}, got {tuple(coeffs.shape)}")
    for axis, name in ((1, "theta"), (2, "zeta")):
        if not _is_uniform_periodic(basis_0.Λ[axis]):
            raise ValueError(
                f"{name} axis must be a uniform periodic B-spline basis")
    perm_t = _reflection_permutation(basis_0.Λ[1].n, basis_0.Λ[1].p)
    perm_z = _reflection_permutation(basis_0.Λ[2].n, basis_0.Λ[2].p)
    reflected = coeffs[:, :, perm_t, :][:, :, :, perm_z]
    signs = STELLARATOR_REFLECTION.reshape(3, 1, 1, 1)
    return 0.5 * (coeffs + signs * reflected)
