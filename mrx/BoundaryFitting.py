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


def cerfon_map(epsilon=0.32, kappa=1.72, delta=0.33, R0=1.0):

    π = jnp.pi
    alpha = jnp.arcsin(delta)

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


def helical_map(epsilon=0.33, h=0.25, n_turns=3, kappa=1.0):
    π = jnp.pi

    def X(ζ):
        return jnp.array([
            (1 + h * jnp.cos(2 * π * n_turns * ζ)) * jnp.cos(2 * π * ζ),
            (1 + h * jnp.cos(2 * π * n_turns * ζ)) * jnp.sin(2 * π * ζ),
            h * jnp.sin(2 * π * n_turns * ζ)
        ])

    def get_frame(ζ):
        dX = jax.jacrev(X)
        τ = dX(ζ) / jnp.linalg.norm(dX(ζ))  # Tangent vector
        dτ = jax.jacfwd(dX)(ζ)
        ν1 = dτ / jnp.linalg.norm(dτ)
        # e = jnp.array([0.0, 0.0, 1.0])
        # ν1 = (e - jnp.dot(e, τ) * τ)
        ν1 = ν1 / jnp.linalg.norm(ν1)  # First normal vector
        ν2 = jnp.cross(τ, ν1)         # Second normal vector
        return τ, ν1, ν2

    def F(x):
        """Helical coordinate mapping function."""
        r, θ, ζ = x
        _, ν1, ν2 = get_frame(ζ)
        return X(ζ) + epsilon * r * jnp.cos(2 * π * θ) * ν1 + epsilon * r * kappa * jnp.sin(2 * π * θ) * ν2

    return F


def rotating_ellipse_map(epsilon=0.1, kappa=4, m=5):
    π = jnp.pi
    a = epsilon * kappa
    b = epsilon

    def R(x):
        r, θ, ζ = x
        return 1 + a * r * jnp.cos(2 * π * θ) + b * r * jnp.cos(2 * π * θ - 2 * π * m * ζ)

    def Z(x):
        r, θ, ζ = x
        return -a * r * jnp.sin(2 * π * θ) + b * r * jnp.sin(2 * π * θ - 2 * π * m * ζ)

    def F(x):
        r, θ, ζ = x
        R_val = R(x)
        Z_val = Z(x)
        return jnp.ravel(jnp.array([
            R_val * jnp.cos(2 * π * ζ),
            R_val * jnp.sin(2 * π * ζ),
            Z_val
        ]))

    return F


def w7x_map():
    π = jnp.pi
    Nfp = 5
    R0 = 5.5586

    m_vals = jnp.arange(0, rbc.shape[0])        # [0..6]
    n_vals = jnp.arange(-6, 7)                  # [-6..6]

    def R_axis(ɸ):
        return R0 + 0.35209 * jnp.cos(ɸ)

    def Z_axis(ɸ):
        return -0.29578 * jnp.sin(ɸ)

    # Fourier sums
    def eval_boundary(theta, phi):
        phases = (m_vals[:, None] * theta) - (n_vals[None, :] * Nfp * phi)
        cosvals = jnp.cos(phases)
        sinvals = jnp.sin(phases)
        Rb = jnp.sum(rbc * cosvals)
        Zb = jnp.sum(zbc * sinvals)
        return Rb, Zb

    def F(p):
        r, theta, zeta = p
        phi = -zeta * 2 * π
        theta *= 2 * π

        Rb, Zb = eval_boundary(theta, phi)
        Rc, Zc = R_axis(phi), Z_axis(phi)
        # eps = 0.5
        # # "smooth" the boundary by increasing two mode numbers
        # Rb = Rb + eps * (Rc + jnp.cos(theta))
        # Zb = Zb + eps * (Zc + jnp.sin(theta))

        # interpolate from magnetic axis to outer boundary
        R_pt = Rc + r * (Rb - Rc)
        Z_pt = Zc + r * (Zb - Zc)

        x = R_pt * jnp.cos(phi)
        y = R_pt * jnp.sin(phi)
        z = Z_pt

        return jnp.array([x, y, z]) / R0

    return F


rbc = jnp.array([  # rows are m, cols are n (-6 to 6), modes are cos(m theta - n Nfp phi)
    # m = 0
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.5586, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    # m = 1
    [5.8671e-4, -0.0020214, -0.0032499, -0.0048626, 0.0099524, 0.033555, 0.3 + 0.49093,  # !!!
     -0.25107, 0.0051767, 0.0048235, 3.2065e-4, 1.959e-4, -0.0013768],
    # m = 2
    [5.0917e-4, -6.6977e-4, -6.7572e-4, -0.0014368, 0.0020763, 0.014863, 0.038447,
     0.039485, 0.063998, -0.0079348, 0.0026108, 0.0020567, 5.4194e-4],
    # m = 3
    [-2.123e-4, 4.3695e-5, 3.3644e-4, 6.4011e-4, 0.0014828, 0.001668, -0.007879,
     -0.016151, -0.017821, -0.010231, 0.0020548, -0.0014357, -6.283e-4],
    # m = 4
    [1.6361e-4, -1.6965e-4, 1.6596e-4, 1.9015e-5, -1.6802e-4, -0.0018659, 0.003313,
     -0.0010438, 0.0092226, 7.7604e-4, 5.9064e-5, -5.2259e-5, 3.0372e-4],
    # m = 5
    [-9.7626e-5, -2.1519e-5, 4.0222e-5, -2.9516e-4, -2.6001e-4, 0.001071, 7.9054e-4,
     0.0043576, -8.9611e-4, -6.8915e-4, 0.0016336, -1.3732e-4, 6.2276e-5],
    # m = 6
    [4.932e-5, -1.0468e-4, 3.3192e-5, 1.7365e-4, 3.8511e-4, 1.0762e-4, 2.6971e-5,
     -0.003413, -0.0016615, -0.0010591, -5.3034e-4, -6.5265e-4, 9.7208e-5]
])

zbc = jnp.array([
    # m = 0
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.23754, -
        0.0068221, 0.0031833, 0.0017251, 5.7592e-5, 2.643e-4],
    # m = 1
    [0.0013323, -0.0016138, -0.0042108, -0.0070096, 0.0096053, 0.036669,
     0.3 + 0.61965, 0.17897, 0.0011535, 0.011208, 0.0042871, 0.0022236, -3.1109e-4],  # !!!
    # m = 2
    [-8.5478e-4, 3.8773e-4, -4.941e-4, 1.4908e-4, 0.0013789, 0.011065, -0.0089934,
        0.023219, -0.055638, 0.010888, -5.4241e-4, -0.0010494, -6.6246e-5],
    # m = 3
    [1.434e-4, -6.3872e-6, -1.558e-4, 1.7715e-4, 3.9966e-4, -0.0021165, -
        4.2775e-4, 6.5673e-4, 0.0075688, 0.0094628, -0.0025063, 7.8944e-4, 3.7802e-4],
    # m = 4
    [-1.0181e-4, 1.2157e-4, 4.6167e-5, -2.4091e-4, -1.4719e-4, -4.1381e-4,
        4.3528e-4, 0.0016804, 0.0072835, -0.0060814, 1.6546e-5, 9.278e-4, -2.6709e-4],
    # m = 5
    [-7.3965e-5, 1.3889e-4, -2.0578e-4, 1.2722e-4, 2.1614e-4, 0.0010369,
        0.0015842, -0.0038666, -0.0024331, 6.5004e-4, 5.165e-4, -3.3282e-4, -1.9109e-4],
    # m = 6
    [1.7171e-6, -5.2834e-5, 7.6033e-5, 1.1611e-4, 5.5193e-5, -1.5852e-5, -
        5.8329e-4, -2.3786e-4, -4.2717e-4, -8.1209e-4, -4.0144e-4, 2.4175e-4, -1.894e-4]
])
