"""Analytic logical-to-physical maps and the :class:`SplineMap` wrapper for fitted ones."""
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.numpy import cos, pi, sin

from mrx.differential_forms import DifferentialForm

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


def extend_map_half_period(F_half: Callable) -> Callable:
    """Extend a half-field-period map to the full period by reflection.

    The input map's ``zeta`` interval ``[0, 1]`` is one field period. For
    ``zeta <= 1/2`` the result equals ``F_half``; for ``zeta > 1/2`` it
    is ``S F_half(r, -theta, -zeta)`` with ``S = diag(1, -1, -1)``. An
    already-symmetric ``F_half`` is reproduced on the whole period. The
    fold is at half a field period in the map's own ``zeta``, so there
    is no separate ``nfp``: a multi-period device already stores one
    period in ``[0, 1]``.

    Args:
        F_half: Map defined on (at least) the first half-period
            ``zeta in [0, 1/2]``. Periodic evaluation at ``-zeta`` is
            used for the reflected half.

    Returns:
        A map defined on the full period ``zeta in [0, 1]``.
    """
    S = STELLARATOR_REFLECTION

    def F(x):
        r, t, z = x
        z_mod = jnp.mod(z, 1.0)
        x_plus = jnp.array([r, t, z_mod])
        x_minus = jnp.array([r, -t, -z_mod])
        return jnp.where(z_mod <= 0.5, F_half(x_plus), S * F_half(x_minus))

    return F


def invert_map_poloidal(F: Callable, iters: int = 20) -> Callable:
    """Invert a logical-to-physical map in the poloidal plane at fixed ``zeta``.

    Both the equilibrium map and a map2disc map of the same boundary use
    the GVEC toroidal convention ``(R cos 2 pi zeta / nfp, sign R sin
    2 pi zeta / nfp, Z)``, so logical ``zeta`` is shared and the
    coordinate change between them is purely poloidal. Returns
    ``inverse(p, zeta) -> (rho, theta)``, a fixed-trip-count Newton
    solving ``F([rho, theta, zeta]) = p``.

    The iteration is in ``(u, v) = rho (cos, sin)(2 pi theta)`` so the
    polar axis is not a degenerate point of the Jacobian, and is seeded
    from the magnetic axis ``F([0, 0, zeta])``. That is the right seed
    for a nested equilibrium map, which is star-shaped about the axis
    (a map2disc crescent is not, and is not inverted here). The body is
    a ``lax.fori_loop``: ``jit``- and ``vmap``-safe, no Python branch on
    a traced value. The Jacobian is the chain rule through ``jacfwd(F)``,
    never through ``atan2``. Steps are capped and the iterate is kept
    inside a slightly enlarged disc: a full Newton step from the axis
    toward a near-wall target can leave the domain, where a clamped
    spline explodes.

    Args:
        F: Logical-to-physical map ``(rho, theta, zeta) -> (X, Y, Z)``.
        iters: Newton steps. A Python ``int``, closed over as a static
            trip count.

    Returns:
        ``inverse(p, zeta) -> (rho, theta)`` with ``p`` a physical point
        of shape ``(3,)`` and ``zeta`` the logical toroidal angle.
    """
    two_pi = 2.0 * jnp.pi

    def _from_uv(uv: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return jnp.hypot(uv[0], uv[1]), jnp.atan2(uv[1], uv[0]) / two_pi

    def inverse(p: jnp.ndarray, zeta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        def residual(uv: jnp.ndarray) -> jnp.ndarray:
            rho, theta = _from_uv(uv)
            return F(jnp.array([rho, theta, zeta])) - p

        def jacobian(uv: jnp.ndarray) -> jnp.ndarray:
            # Chain-rule Jacobian: ``jacfwd(F)`` never sees ``atan2``.
            # ``d(rho, theta)/d(u, v)`` is singular on the axis, where
            # ``dF/dtheta = 0`` and the two columns collapse to
            # ``dF/drho`` at ``theta = 0`` and ``1/4``.
            rho, theta = _from_uv(uv)
            dF = jax.jacfwd(F)(jnp.array([rho, theta, zeta]))[:, :2]
            cut = jnp.sqrt(jnp.finfo(uv.dtype).eps)
            rho_s = jnp.maximum(rho, cut)
            d_rt = jnp.array([[uv[0] / rho_s, uv[1] / rho_s],
                              [-uv[1] / (rho_s ** 2 * two_pi),
                               uv[0] / (rho_s ** 2 * two_pi)]])
            j_off = dF @ d_rt
            j_axis = jnp.stack([
                jax.jacfwd(F)(jnp.array([cut, 0.0, zeta]))[:, 0],
                jax.jacfwd(F)(jnp.array([cut, 0.25, zeta]))[:, 0],
            ], axis=1)
            return jnp.where(rho < cut, j_axis, j_off)

        def body(_: int, uv: jnp.ndarray) -> jnp.ndarray:
            r = residual(uv)
            jac = jacobian(uv)
            gram = jac.T @ jac
            reg = jnp.finfo(uv.dtype).eps * (1.0 + jnp.vdot(gram, gram))
            delta = jnp.linalg.solve(gram + reg * jnp.eye(2, dtype=uv.dtype), jac.T @ r)
            # A full step from the axis toward a near-wall target can
            # leave the disc; a clamped spline then explodes. Cap the
            # step and keep the iterate inside a slightly enlarged disc.
            step = jnp.hypot(delta[0], delta[1])
            delta = jnp.where(step > 0.5, delta * (0.5 / jnp.maximum(step, 1e-30)), delta)
            nxt = uv - delta
            rad = jnp.hypot(nxt[0], nxt[1])
            nxt = jnp.where(rad > 1.05, nxt * (1.05 / rad), nxt)
            return jnp.where(jnp.isfinite(nxt), nxt, uv)

        uv0 = jnp.zeros(2, dtype=jnp.asarray(p).dtype)
        uv = jax.lax.fori_loop(0, iters, body, uv0)
        rho, theta = _from_uv(uv)
        return rho, jnp.mod(theta, 1.0)

    return inverse


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
