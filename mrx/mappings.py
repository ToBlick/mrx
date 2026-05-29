from typing import Any, Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.numpy import cos, pi, sin

from mrx.differential_forms import DifferentialForm, DiscreteFunction
from mrx.io import project_sampled_field


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


def cerfon_map(epsilon: float = 0.33, kappa: float = 1.2, alpha: float = 0.0, R0: float = 1.0) -> Callable:
    """
    Mapping function from "One Size Fits All" paper by Cerfon et al.:
    F(r, t, z) = (X(t), Y(t), z) where X(t) and Y(t) are the Cartesian coordinates, 
    and (r, t, z) are the logical coordinates, with t a poloidal angle.

    Parameters
    ----------
    epsilon : float, default=0.33
        Eccentricity of the ellipse.
    kappa : float, default=1.2
        Aspect ratio of the ellipse.
    alpha : float, default=0.0
        Phase shift of the ellipse.
    R0 : float, default=1.0
        Major radius of the ellipse.

    Returns
    -------
    F : Callable
        Cerfon mapping function.
    """
    π = jnp.pi

    def x_t(t):
        """X coordinate function."""
        return 1 + epsilon * jnp.cos(2 * π * t + alpha * jnp.sin(2 * π * t))

    def y_t(t):
        """Y coordinate function."""
        return epsilon * kappa * jnp.sin(2 * π * t)

    def _s_from_t(t):
        """
        Arc tangent of the ratio of the Y and X coordinates.
        """
        return jnp.arctan2(kappa * jnp.sin(2 * π * t),
                           jnp.cos(2 * π * t + alpha * jnp.sin(2 * π * t)))

    def s_from_t(t):
        """Arc length of the ellipse."""
        return jnp.where(t > 0.5, _s_from_t(t) + 2 * π, _s_from_t(t))

    def a_from_t(t):
        """Radius of the ellipse."""
        return jnp.sqrt((x_t(t) - 1)**2 + y_t(t)**2)

    @jax.jit
    def F(x):
        """Cerfon mapping function."""
        r, χ, z = x
        return jnp.ravel(jnp.array(
            [(R0 + a_from_t(χ) * r * jnp.cos(s_from_t(χ))) * jnp.cos(2 * π * z),
             -(R0 + a_from_t(χ) * r * jnp.cos(s_from_t(χ))) * jnp.sin(2 * π * z),
             a_from_t(χ) * r * jnp.sin(s_from_t(χ))]))
    return F


def helical_map(epsilon: float = 0.33, h: float = 0.25, kappa: float = 1.2, alpha: float = 0.0, nfp: int = 3) -> Callable:
    """
    Helical mapping function:
    F(r, t, ζ) = (X(ζ)) where X(ζ) is the coordinate along the helix.
    and (r, t, ζ) are the logical coordinates, with ζ the toroidal angle.

    Parameters
    ----------
    epsilon : float
        Eccentricity of the helix.
    h : float, default=0.25
        Height of the helix.
    kappa : float, default=1.2
        Aspect ratio of the helix.
    alpha : float, default=0.0
        Phase shift of the helix.
    nfp : int, default=3
        Number of turns of the helix (number of field periods).

    Returns
    -------
    F : Callable
        Helical mapping function.
    """
    π = jnp.pi

    def X(ζ):
        return jnp.array([
            (1 + kappa * h * jnp.cos(2 * π * nfp * ζ + alpha)) * jnp.sin(2 * π * ζ),
            (1 + kappa * h * jnp.cos(2 * π * nfp * ζ + alpha)) * jnp.cos(2 * π * ζ),
            kappa * h * jnp.sin(2 * π * nfp * ζ + alpha)
        ])

    def get_frame(ζ):
        """Get the Frenet frame of the helix."""
        dX = jax.jacrev(X)
        τ = dX(ζ) / jnp.linalg.norm(dX(ζ))  # Tangent vector
        dτ = jax.jacfwd(dX)(ζ)
        ν1 = dτ / jnp.linalg.norm(dτ)
        ν1 = ν1 / jnp.linalg.norm(ν1)  # First normal vector
        ν2 = jnp.cross(τ, ν1)         # Second normal vector
        return τ, ν1, ν2

    def F(x):
        """Helical coordinate mapping function."""
        r, t, ζ = x
        _, ν1, ν2 = get_frame(ζ)
        return (X(ζ)
                + epsilon * r * jnp.cos(2 * π * t) * ν1
                + epsilon * r * jnp.sin(2 * π * t) * ν2)
    return F


def rotating_ellipse_map(eps: float = 0.33, kappa: float = 1.2, R0: float = 1.0, nfp: int = 3) -> Callable:
    """
    Rotating ellipse mapping function.

    Parameters
    ----------
    eps : float, default=0.33
        Eccentricity of the ellipse.
    kappa : float, default=1.2
        Aspect ratio of the ellipse.
    R0 : float, default=1.0
        Major radius of the ellipse.
    nfp : int, default=3
        Number of field periods of the ellipse.

    Returns
    -------
    F : Callable
        Rotating ellipse mapping function.
    """
    # Raise an error if nfp is not a positive integer.
    if nfp <= 0:
        raise ValueError(f"nfp must be a positive integer, got {nfp}")
    if eps <= 0:
        raise ValueError(f"eps must be a positive number, got {eps}")

    def nu(zeta):
        return 1 + (1 - kappa) * cos(2 * pi * zeta * nfp)

    def F(x):
        """Rotating ellipse mapping function."""
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
    """
    Toroidal mapping function:
    F(r, χ, z) = (X, Y, Z) where X, Y, Z are the Cartesian coordinates, 
    and (r, χ, z) are the logical coordinates, with χ the toroidal angle.

    Parameters
    ----------
    epsilon : float
        Eccentricity of the ellipse.
    kappa : float, default=1.0
        Aspect ratio of the ellipse.
    R0 : float, default=1.0
        Major radius of the toroid.
    """
    π = jnp.pi

    def F(x):
        """Toroidal coordinate mapping function. Formula is:

        F(r, θ, z) = (X, Y, Z) = (R(r, θ) cos(2πz), -R(r, θ) sin(2πz), ɛ r sin(2πθ))
        where R(r, θ) = R0 + ɛ r cos(2πθ) is the radial coordinate.
        """
        r, θ, ζ = x
        R = R0 + epsilon * r * jnp.cos(2 * π * θ)
        return jnp.array([R * jnp.cos(2 * π * ζ),
                          -R * jnp.sin(2 * π * ζ),
                          epsilon * kappa * r * jnp.sin(2 * π * θ)])
    return F


def polar_map() -> Callable:
    """
    Polar mapping function:
    F(r, θ, z) = (X, Y, Z) where X, Y, Z are the Cartesian coordinates, 
    and (r, θ, z) are the logical coordinates, with θ the poloidal angle.
    """
    π = jnp.pi

    def F(x):
        """Polar coordinate mapping function."""
        r, θ, z = x
        return jnp.array([r * jnp.cos(2 * π * θ), -z, r * jnp.sin(2 * π * θ)])
    return F


def cylinder_map(a: float = 1.0, h: float = 1.0) -> Callable:
    """
    Cylinder mapping function:
    F(r, χ, z) = (X, Y, Z) where X, Y, Z are the Cartesian coordinates, 
    and (r, χ, z) are the logical coordinates, with χ the toroidal angle.

    Parameters
    ----------
    a : float
        Radius of the cylinder.
    h : float
        Height of the cylinder.

    Returns
    -------
    F : Callable
        Cylinder mapping function.
    """
    π = jnp.pi

    def _X(r, χ):
        """Cylindrical radial coordinate. Formula is:

        X(r, χ) = a r cos(2πχ)
        """
        return jnp.ones(1) * (a * r * jnp.cos(2 * π * χ))

    def _Y(r, χ):
        """Cylindrical vertical coordinate. Formula is:

        Y(r, χ) = a r sin(2πχ)
        """
        return jnp.ones(1) * (a * r * jnp.sin(2 * π * χ))

    def F(x: jnp.ndarray) -> jnp.ndarray:
        """Cylindrical coordinate mapping function. Formula is:

        F(r, χ, z) = (X, Y, Z) = (a r cos(2πχ), a r sin(2πχ), h z)

        Args:   
            x: Input logical coordinates (r, χ, z)

        Returns:
            F: Coordinate mapping function (R, Z, hz)
        """
        r, χ, z = x
        return jnp.ravel(jnp.array([_X(r, χ), _Y(r, χ), h * jnp.ones(1) * z]))

    return F


def stellarator_map(X1_h: DiscreteFunction, X2_h: DiscreteFunction, nfp: int = 3, flip_zeta: bool = False) -> Callable:
    """
    A basic stellarator map:
    F(r, χ, z) = (X, Y, Z) where X, Y, Z are the Cartesian coordinates, 
    and (r, χ, z) are the logical coordinates, with χ the toroidal angle.

    Parameters
    ----------
    X1_h : DiscreteFunction
        Function X1_h(r, θ, ζ) determining the radial coordinate of the GVEC map.
    X2_h : DiscreteFunction
        Function X2_h(r, θ, ζ) determining the vertical coordinate of the GVEC map.
    nfp : int, default=3
        Number of field periods.

    Returns
    -------
    """
    π_nfp = 2 * jnp.pi / nfp

    def F(x):
        _, _, ζ = x
        if flip_zeta:
            ζ = 1.0 - ζ
        return jnp.array([X1_h(x)[0] * jnp.cos(π_nfp * ζ),
                          -X1_h(x)[0] * jnp.sin(π_nfp * ζ),
                          X2_h(x)[0]])
    return F

def greville_interpolate_map(F_analytic: Callable, seq) -> jnp.ndarray:
    """Interpolate an analytic logical-to-physical map to spline coefficients.

    Evaluates each Cartesian component of ``F_analytic`` at the
    tensor-product Greville points and solves the resulting 1-D collocation
    systems, returning a coefficient array suitable for
    :meth:`~mrx.derham_sequence.DeRhamSequence.set_spline_map`.

    This replaces the heavier workflow of sampling the map on a regular
    grid, L²-projecting with the reference-domain mass matrix, and then
    calling ``set_spline_map``.  No mass matrix is required; the only
    prerequisite is :meth:`~mrx.derham_sequence.DeRhamSequence.evaluate_1d`.

    Parameters
    ----------
    F_analytic : callable
        Analytic map ``F: R^3 -> R^3`` mapping logical coordinates
        ``(r, θ, ζ) ∈ [0, 1]^3`` to physical Cartesian coordinates
        ``(X, Y, Z)``.
    seq : DeRhamSequence
        The sequence to interpolate into.  Must have ``evaluate_1d()``
        called.  Currently requires an all-clamped (non-periodic,
        non-polar) sequence; periodic or polar sequences raise
        ``NotImplementedError`` via :meth:`zeroform_interpolation`.

    Returns
    -------
    coefficients : jnp.ndarray of shape ``(3, seq.n0)``
        Spline DOF vectors for the three Cartesian components, stacked
        along axis 0.  Pass directly to ``seq.set_spline_map(coefficients)``.
    """
    component_dofs = [
        seq.p0.zeroform_interpolation(lambda x, i=i: F_analytic(x)[i])
        for i in range(3)
    ]
    return jnp.stack(component_dofs, axis=0)


# Alias for now


def greville_interpolate_stellarator_map(
        F_analytic: Callable, seq, nfp: int, flip_zeta: bool = False) -> Callable:
    """Build a stellarator map by Greville-interpolating R and Z from an analytic map.

    Extracts the cylindrical radius ``R = sqrt(X² + Y²)`` and vertical
    coordinate ``Z`` from ``F_analytic``, interpolates each as a scalar
    0-form via Greville collocation, and wraps the result in
    :func:`stellarator_map`.

    ``R(r, θ, ζ)`` and ``Z(r, θ, ζ)`` are both periodic in ``θ`` and ``ζ``,
    so they are naturally representable in a periodic spline basis without any
    artificial periodicity violation.  This replaces the heavier
    :func:`interpolate_map` workflow (regular-grid sampling + L²-projection
    via the reference-domain mass matrix) with a single collocation step.

    No mass matrix is required; the only prerequisite is
    :meth:`~mrx.derham_sequence.DeRhamSequence.evaluate_1d`.  Works for
    any non-polar sequence (``polar=False``); polar sequences raise
    ``NotImplementedError`` via :meth:`zeroform_interpolation` because the
    polar extraction operator reduces the DOF count.

    Parameters
    ----------
    F_analytic : callable
        Analytic map ``F: R^3 -> R^3`` returning Cartesian ``(X, Y, Z)``.
    seq : DeRhamSequence
        The sequence to use for interpolation.  Must have ``evaluate_1d()``
        called.  Typically built with ``('clamped', 'periodic', 'periodic')``
        boundary conditions and ``polar=False``.
    nfp : int
        Number of field periods.
    flip_zeta : bool, optional
        Passed through to :func:`stellarator_map`.

    Returns
    -------
    Phi : callable
        Stellarator map ``Phi(r, θ, ζ) -> (X, Y, Z)`` built from the
        interpolated spline representations of R and Z.
    """
    def R_fn(x):
        Fxyz = F_analytic(x)
        return jnp.sqrt(Fxyz[0] ** 2 + Fxyz[1] ** 2)

    def Z_fn(x):
        return F_analytic(x)[2]

    R_dof = seq.p0.zeroform_interpolation(R_fn)
    Z_dof = seq.p0.zeroform_interpolation(Z_fn)

    R_h = DiscreteFunction(R_dof, seq.basis_0, seq.e0)
    Z_h = DiscreteFunction(Z_dof, seq.basis_0, seq.e0)

    return stellarator_map(R_h, Z_h, nfp=nfp, flip_zeta=flip_zeta)


def interpolate_map(axes, R_grid, Z_grid, nfp, seq, flip_zeta=False):
    """
    Interpolate a stellarator map from R and Z sampled on a regular grid.

    Uses :func:`project_sampled_field` (L² projection via
    ``RegularGridInterpolator`` + tensor-product integration) to obtain
    the FEM coefficients for *R* and *Z*, then wraps them in a
    :func:`stellarator_map`.

    .. deprecated::
        Prefer :func:`greville_interpolate_stellarator_map` when an analytic
        map is available: it requires no reference-domain mass matrix and no
        sampled grid.

    Parameters
    ----------
    axes : tuple of 1-D arrays
        Grid axes ``(x1, x2, x3)`` spanning the logical domain.
    R_grid : jnp.ndarray
        R values on the grid, shape ``(n1, n2, n3)``.
    Z_grid : jnp.ndarray
        Z values on the grid, shape ``(n1, n2, n3)``.
    nfp : int
        Number of field periods.
    seq : DeRhamSequence
        DeRham sequence to use for interpolation.  Must have
        ``evaluate_1d()`` and ``assemble_reference_mass_matrix()`` called first.
    flip_zeta : bool
        Whether to flip the toroidal angle in the stellarator map.

    Returns
    -------
    Phi : callable
        Stellarator map built from the interpolated R, Z.
    """
    R_dof = project_sampled_field(
        axes, R_grid, seq, k=0, dirichlet=False, reference_domain=True)
    Z_dof = project_sampled_field(
        axes, Z_grid, seq, k=0, dirichlet=False, reference_domain=True)

    R_h = DiscreteFunction(R_dof, seq.basis_0, seq.e0)
    Z_h = DiscreteFunction(Z_dof, seq.basis_0, seq.e0)

    return stellarator_map(R_h, Z_h, nfp=nfp, flip_zeta=flip_zeta)


def approx_inverse_map(y: jnp.ndarray, eps: float, R0: float = 1.0) -> jnp.ndarray:
    """
    Approximate inverse mapping function.

    Parameters
    ----------
    y : jnp.ndarray
        Input coordinates.
    eps : float
        Eccentricity of the ellipse.

    Returns
    -------
    x : jnp.ndarray
        Output coordinates.
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
    """
    Invert a map using Newton's method.

    Parameters
    ----------
    f : Callable
        Map to invert.
    y_target : jnp.ndarray
        Target coordinates.
    x0_fn : Callable
        Function to compute initial guess for x from y_target.
    tol : float, default=1e-10
        Tolerance for convergence.
    max_iter : int, default=50
        Maximum number of iterations.

    Returns
    -------
    x : jnp.ndarray
        Output coordinates such that f(x) = y_target.
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
    """
    Extend the mapping defined by (X1, X2) over one field period
    to the full torus with nfp field periods.

    Parameters
    ----------
    Phi : callable
        Mapping from logical coordinates to physical coords: (r,theta,zeta)->(x,y,z)
    nfp : int
        Number of field periods.

    Returns
    -------
    Phi_full_fp : callable
        Mapping from logical coordinates to physical coords: (r,theta,zeta)->(x,y,z)
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
