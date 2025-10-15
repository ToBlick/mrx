import jax
import jax.numpy as jnp
import optimistix

from mrx.DifferentialForms import DifferentialForm, DiscreteFunction
from mrx.Quadrature import QuadratureRule


def lcfs_fit(n_map,
             p_map,
             q_map,
             R0,
             k0,
             q0,
             aR,
             atol=1e-6,
             rtol=1e-6,
             maxiter=100_000):

    ###
    # ψ(R, Z) =  (¼ k₀² (R² - R₀²)² + R²Z² ) / (2 R₀² k₀ q₀)
    ###
    def psi(R, Z):
        def _psi(R, Z):
            return (k0**2/4*(R**2 - R0**2)**2 + R**2*Z**2) / (2 * R0**2 * k0 * q0)
        return _psi(R, Z) - _psi(R0 + aR, 0)
    π = jnp.pi

    Λ_map = DifferentialForm(0, (n_map, 1, 1), (p_map, 0, 0),
                             ("periodic", "constant", "constant"))
    Q_map = QuadratureRule(Λ_map, q_map)

    def loss(a_hat, args):
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


def get_lcfs_F(n_map,
               p_map,
               q_map,
               R0,
               k0,
               q0,
               aR,
               atol=1e-6,
               rtol=1e-6,
               maxiter=20_000):

    π = jnp.pi

    a_hat = lcfs_fit(n_map, p_map, q_map, R0, k0, q0, aR,
                     atol=atol, rtol=rtol, maxiter=maxiter)
    Λ_map = DifferentialForm(0, (n_map, 1, 1), (p_map, 0, 0),
                             ("periodic", "constant", "constant"))
    a_h = DiscreteFunction(a_hat, Λ_map)

    def a(χ):
        return a_h(jnp.array([χ, 0, 0]))[0]

    def _R(r, χ):
        return jnp.ones(1) * (R0 + a(χ) * r * jnp.cos(2 * π * χ))

    def _Z(r, χ):
        return jnp.ones(1) * a(χ) * r * jnp.sin(2 * π * χ)

    def F(x):
        r, χ, z = x
        return jnp.ravel(jnp.array(
            [_R(r, χ) * jnp.cos(2 * π * z),
             -_R(r, χ) * jnp.sin(2 * π * z),
             _Z(r, χ)]))

    return F


def cerfon_map(epsilon, kappa, alpha, R0=1.0):

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


def helical_map(epsilon=0.33, h=0.25, n_turns=3, kappa=1.0, alpha=-0.3):
    π = jnp.pi

    def X(ζ):
        return jnp.array([
            (1 + h * jnp.cos(2 * π * n_turns * ζ)) * jnp.cos(2 * π * ζ),
            (1 + h * jnp.cos(2 * π * n_turns * ζ)) * jnp.sin(2 * π * ζ),
            h * jnp.sin(2 * π * n_turns * ζ)
        ])

    def get_frame(ζ):
        # dX = jax.jacrev(X)
        # τ = dX(ζ) / jnp.linalg.norm(dX(ζ))  # Tangent vector
        # dτ = jax.jacfwd(dX)(ζ)
        # ν1 = dτ / jnp.linalg.norm(dτ)
        # # e = jnp.array([0.0, 0.0, 1.0])
        # # ν1 = (e - jnp.dot(e, τ) * τ)
        # ν1 = ν1 / jnp.linalg.norm(ν1)  # First normal vector
        # ν2 = jnp.cross(τ, ν1)         # Second normal vector
        # return τ, ν1, ν2
        τ = jnp.array([-jnp.sin(2 * π * ζ),
                       jnp.cos(2 * π * ζ),
                       0.0])
        ν1 = jnp.array([jnp.cos(2 * π * ζ),
                        jnp.sin(2 * π * ζ),
                        0.0])

        # Twist angle
        θ = 2 * π * n_turns * ζ

        cosθ, sinθ = jnp.cos(θ), jnp.sin(θ)
        ν1_rot = ν1 * cosθ + jnp.cross(τ, ν1) * \
            sinθ + τ * (jnp.dot(τ, ν1)) * (1 - cosθ)
        ν2_rot = jnp.cross(τ, ν1_rot)

        return τ, ν1_rot, ν2_rot

    def x_t(t):
        return epsilon * jnp.cos(2 * π * t + alpha * jnp.sin(2 * π * t))

    def y_t(t):
        return epsilon * kappa * jnp.sin(2 * π * t)

    def _s_from_t(t):
        return jnp.arctan2(y_t(t), x_t(t))

    def s_from_t(t):
        return jnp.where(t > 0.5, _s_from_t(t) + 2 * π, _s_from_t(t))

    def a_from_t(t):
        return jnp.sqrt(x_t(t)**2 + y_t(t)**2)

    def F(x):
        """Helical coordinate mapping function."""
        r, t, ζ = x
        _, ν1, ν2 = get_frame(ζ)
        return (X(ζ)
                + a_from_t(t) * r * jnp.cos(s_from_t(t)) * ν1
                + a_from_t(t) * r * jnp.sin(s_from_t(t)) * ν2)

    return F


def rotating_ellipse_map(a=0.1, b=0.025, m=5):
    π = jnp.pi

    def Rb(θ, ζ):
        return 1 + a * jnp.cos(2 * π * θ) + b * jnp.cos(2 * π * θ - 2 * π * m * ζ)

    def Zb(θ, ζ):
        return -a * jnp.sin(2 * π * θ) + b * jnp.sin(2 * π * θ - 2 * π * m * ζ)

    def F(x):
        r, θ, ζ = x
        R_val = 1 + r * (Rb(θ, ζ) - 1)
        Z_val = r * Zb(θ, ζ)
        return jnp.ravel(jnp.array([
            R_val * jnp.cos(2 * π * ζ),
            R_val * jnp.sin(2 * π * ζ),
            Z_val
        ]))

    return F
