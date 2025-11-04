import jax
import jax.numpy as jnp
import optimistix
from jax.numpy import cos, pi, sin

from mrx.differential_forms import DifferentialForm, DiscreteFunction
from mrx.quadrature import QuadratureRule


def lcfs_fit(n_map, p_map, q_map, R0, k0, q0, aR, atol=1e-6, rtol=1e-6, maxiter=100_000)->jnp.ndarray:
    """
    Fit the last closed flux surface (LCFS) shape to the data using a least squares approach solved with Levenberg-Marquardt.

    Flux function: ψ(R, Z) =  (¼ k0² (R² - R0²)² + R²Z² ) / (2 R0² k0 q0)
    where R0 is the major radius, k0 is the toroidal magnetic field strength, 
    q0 is the safety factor at the LCFS, and aR is the radial position of the LCFS.

    Parameters
    ----------
    n_map : int
        Number of basis functions in the mapping.
    p_map : int
        Polynomial degree in the mapping.
    q_map : int
        Quadrature order in the mapping.
    R0 : float
        Major radius of the LCFS.
    k0 : float
        Toroidal magnetic field strength.
    q0 : float
        Safety factor at the LCFS.
    aR : float
        Radial position of the LCFS.
    atol : float, default=1e-6
        Absolute tolerance for the fit.
    rtol : float, default=1e-6
        Relative tolerance for the fit.
    maxiter : int, default=100_000
        Maximum number of iterations for the fit.

    Returns
    -------
    a_hat : jnp.ndarray
        Fitted LCFS shape.
    """
    def psi(R, Z):
        def _psi(R, Z):
            return (k0**2/4*(R**2 - R0**2)**2 + R**2*Z**2) / (2 * R0**2 * k0 * q0)
        return _psi(R, Z) - _psi(R0 + aR, 0)
    π = jnp.pi
    Λ_map = DifferentialForm(0, (n_map, 1, 1), (p_map, 0, 0),
                             ("periodic", "constant", "constant"))
    Q_map = QuadratureRule(Λ_map, q_map)

    def loss(a_hat):
        """
        Loss function to minimize for the LCFS fit.

        Parameters
        ----------
        a_hat : jnp.ndarray
            Fitted LCFS shape.

        Returns
        -------
        loss : float
            Loss value.
        """
        a_h = DiscreteFunction(a_hat, Λ_map)

        def _psi(x):
            a = a_h(x)[0]
            χ = x[0]
            R = R0 + a * jnp.cos(2 * π * χ)
            Z = a * jnp.sin(2 * π * χ)
            return psi(R, Z)**2
        return Q_map.w @ jax.vmap(_psi)(Q_map.x)

    solver = optimistix.LevenbergMarquardt(rtol=rtol, atol=atol)
    sol = optimistix.least_squares(fn=loss,
                                   y0=jnp.ones(n_map) * aR,
                                   solver=solver,
                                   max_steps=maxiter
                                   )
    a_hat = sol.value
    return a_hat


def get_lcfs_F(n_map, p_map, q_map, R0, k0, q0, aR, atol=1e-6, rtol=1e-6, maxiter=20_000):
    """
    Get the LCFS mapping function:
    F(r, χ, z) = (R(r, χ), Z(r, χ), z) where R(r, χ) and Z(r, χ) are the 
    cylindrical radial and Z coordinates, respectively, and (r, χ, z) are the 
    logical coordinates.

    Parameters
    ----------
    n_map : int
        Number of basis functions in the mapping.
    p_map : int
        Polynomial degree in the mapping.
    q_map : int
        Quadrature order in the mapping.
    R0 : float
        Major radius of the LCFS.
    k0 : float
        Toroidal magnetic field strength.
    q0 : float
    aR : float
        Radial position of the LCFS.
    atol : float, default=1e-6
        Absolute tolerance for the fit.
    rtol : float, default=1e-6
        Relative tolerance for the fit.
    maxiter : int, default=20_000
        Maximum number of iterations for the fit.

    Returns
    -------
    F : Callable
        LCFS mapping function.
    """
    π = jnp.pi

    a_hat = lcfs_fit(n_map, p_map, q_map, R0, k0, q0, aR,
                     atol=atol, rtol=rtol, maxiter=maxiter)
    Λ_map = DifferentialForm(0, (n_map, 1, 1), (p_map, 0, 0),
                             ("periodic", "constant", "constant"))
    a_h = DiscreteFunction(a_hat, Λ_map)

    def a(χ):
        """
        Radius function.

        Parameters
        ----------
        χ : float
            Toroidal angle.

        Returns
        -------
        a : float
            Radius.
        """
        return a_h(jnp.array([χ, 0, 0]))[0]

    def _R(r, χ):
        """
        Cylindrical radial coordinate function.

        Parameters
        ----------
        r : float
            Radial coordinate.
        χ : float
            Toroidal angle.

        Returns
        -------
        R : float
            Radial coordinate.
        """
        return jnp.ones(1) * (R0 + a(χ) * r * jnp.cos(2 * π * χ))

    def _Z(r, χ):
        """
        Cylindrical Z coordinate function.

        Parameters
        ----------
        r : float
            Radial coordinate.
        χ : float
            Toroidal angle.

        Returns
        -------
        Z : float
            Z coordinate.
        """
        return jnp.ones(1) * a(χ) * r * jnp.sin(2 * π * χ)

    def F(x):
        """
        Cylindrical mapping function.

        Parameters
        ----------
        x : jnp.ndarray
            Input coordinates.

        Returns
        -------
        F : jnp.ndarray
            Mapping function.
        """
        r, χ, z = x
        return jnp.ravel(jnp.array(
            [_R(r, χ) * jnp.cos(2 * π * z),
             -_R(r, χ) * jnp.sin(2 * π * z),
             _Z(r, χ)]))

    return F


def cerfon_map(epsilon, kappa, alpha, R0=1.0):
    """
    Mapping function from "One Size Fits All" paper by Cerfon et al.:
    F(r, t, z) = (X(t), Y(t), z) where X(t) and Y(t) are the Cartesian coordinates, 
    and (r, t, z) are the logical coordinates, with t a poloidal angle.

    Parameters
    ----------
    epsilon : float
        Eccentricity of the ellipse.
    kappa : float
        Aspect ratio of the ellipse.
    alpha : float
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


def helical_map(epsilon=0.33, h=0.25, n_turns=3):
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
    n_turns : int, default=3
        Number of turns of the helix.

    Returns
    -------
    F : Callable
        Helical mapping function.
    """
    π = jnp.pi

    def X(ζ):
        return jnp.array([
            (1 + h * jnp.cos(2 * π * n_turns * ζ)) * jnp.sin(2 * π * ζ),
            (1 + h * jnp.cos(2 * π * n_turns * ζ)) * jnp.cos(2 * π * ζ),
            h * jnp.sin(2 * π * n_turns * ζ)
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


def rotating_ellipse_map(eps=0.33, kappa=1.2, nfp=3):
    """
    Rotating ellipse mapping function:
    F(r, θ, ζ) = (X, Y, Z) where X, Y, Z are the Cartesian coordinates, 
    and (r, θ, ζ) are the logical coordinates, with ζ the toroidal angle.

    Parameters
    ----------
    eps : float, default=0.33
        Eccentricity of the ellipse.
    kappa : float, default=1.2
        Aspect ratio of the ellipse.
    nfp : int, default=3
        Number of field periods of the ellipse.

    Returns
    -------
    F : Callable
        Rotating ellipse mapping function.
    """
    def nu(zeta):
        return 1 + (1 - kappa) * cos(2 * pi * zeta * nfp)

    def F(x):
        """Rotating ellipse mapping function."""
        r, θ, ζ = x
        if nfp > 0:
            ζ /= nfp  # only model one field period

        R = 1 + eps * nu(ζ) * r * cos(2 * pi * θ)
        if nfp > 0:
            Z = eps * r * nu(ζ + 0.5 / nfp) * sin(2 * pi * θ)
        else:
            Z = eps * nu(ζ) * r * sin(2 * pi * θ)
        return jnp.array([R * cos(2 * pi * ζ),
                          -R * sin(2 * pi * ζ),
                          Z])
    return F
