# %%
# test_harmonic_fields_hollow_toroid.py
import jax
import jax.numpy as jnp
import scipy as sp

from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction, Pushforward

# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)


def test_harmonic_fields():
    Ip = 1.95
    It = 2.46
    Is = jnp.array([Ip, It])
    n = 6
    p = 3
    q = p + 2
    ɛ = 0.1
    π = jnp.pi
    μ0 = 1.0

    @jax.jit
    def F(x):
        """Hollow toroid. Formula is:
        
        F(r, θ, ζ) = (R cos(2πζ), -R sin(2πζ), ɛ (r + 1)/2 sin(2πθ))
        where R = 1 + ɛ (r + 1)/2 cos(2πθ) is the radial coordinate.

        Args: 
            x: (r, θ, ζ) in logical coordinates

        Returns:
            F: (x, y, z) in physical coordinates
        """
        r, θ, ζ = x
        R = 1 + ɛ * (r + 1)/2 * jnp.cos(2 * π * θ)
        return jnp.array([R * jnp.cos(2 * π * ζ),
                          -R * jnp.sin(2 * π * ζ),
                          ɛ * (r + 1)/2 * jnp.sin(2 * π * θ)])

    # Set up finite element spaces
    ns = (n, n, n)
    ps = (p, p, p)
    types = ("clamped", "periodic", "periodic")
    Seq = DeRhamSequence(ns, ps, q, types, F, polar=False, dirichlet=True)
    Seq.evaluate_1d()
    Seq.assemble_all()

    evs, evecs = sp.linalg.eigh(Seq.M2 @ Seq.dd2, Seq.M2)
    # tolerance is 1e-10 to account for numerical errors
    # since first two eigenvalues are -1e-12 and 1e-11 or so. 
    assert jnp.sum(evs < 1e-10) == 2  # two harmonic fields
    assert jnp.min(evs) > -1e-10  # no negative eigenvalues

    h1_dof = evecs[:, 0]
    h2_dof = evecs[:, 1]

    h1 = jax.jit(DiscreteFunction(h1_dof, Seq.Λ2, Seq.E2))
    h2 = jax.jit(DiscreteFunction(h2_dof, Seq.Λ2, Seq.E2))

    # Compute contour integrals:
    # contour wrapping around the enclosed tunnel poloidally:
    def c1(θ): return jnp.array([1e-6, θ, 0])

    # contour wrapping around the center tunnel toroidally:
    def c2(ζ): return jnp.array([1 - 1e-6, 0.5, ζ])

    def h_dl(twoform, curve):
        def oneform(x):
            DF = jax.jacfwd(F)(x)
            return DF.T @ DF @ twoform(x) / jnp.linalg.det(DF)

        def integrand(χ):
            x = curve(χ)
            dx = jax.jacfwd(curve)(χ).reshape(-1)
            v = oneform(x).reshape(-1)
            return jnp.dot(v, dx)
        return integrand

    # Integrate h1 along contours using trapezoidal rule
    n_q = 256
    _χ = jnp.linspace(0, 1, n_q, endpoint=False)
    _w = jnp.ones(n_q) * (1/n_q)

    # this matrix has entries ∫ h_i · dl_j
    P = jnp.array([
        [jax.vmap(h_dl(h, c))(_χ) @ _w for c in (c1, c2)]
        for h in (h1, h2)
    ])

    # Coefficients of the harmonic fields:
    b = jnp.linalg.solve(P.T, μ0 * Is)
    b_dofs = b[0] * h1_dof + b[1] * h2_dof

    # assert that the solution is indeed harmonic:
    curl_b_dofs = Seq.weak_curl @ b_dofs
    div_b_dofs = Seq.strong_div @ b_dofs
    assert (curl_b_dofs @ Seq.M1 @ curl_b_dofs)**0.5 < 1e-10
    assert (div_b_dofs @ Seq.M3 @ div_b_dofs)**0.5 < 1e-10

    def B_expected(x):
        """Exact magnetic field in Cartesian coordinates.
        In the ɛ ≪ 1 limit, we know that B = μ0 (I_p eφ / 2πR + I_t eθ / 2πd),
        where d is the distance to the centerline of the enclosed tunnel
        and R the distance to the z-axis

        Args: 
            x: (r, θ, ζ) in logical coordinates

        Returns:
            B: (Bx, By, Bz) in Cartesian coordinates
        """
        r, θ, ζ = x
        d = ɛ * (r + 1) / 2
        R = 1 + d * jnp.cos(2 * π * θ)

        sζ, cζ = jnp.sin(2 * π * ζ), jnp.cos(2 * π * ζ)
        sθ, cθ = jnp.sin(2 * π * θ), jnp.cos(2 * π * θ)

        B_ζ = μ0 * It / (2 * π * R)
        B_θ = μ0 * Ip / (2 * π * d)

        Bx = -B_ζ * sζ - B_θ * sθ * cζ
        By = -B_ζ * cζ + B_θ * sθ * sζ
        Bz = B_θ * cθ

        return jnp.array([Bx, By, Bz])

    B_computed = jax.jit(Pushforward(DiscreteFunction(
        b_dofs, Seq.Λ2, Seq.E2), Seq.F, 2))

    # Check the field in the interior
    y = jnp.array([0.5, 0.0, 0.0])
    B_diff = B_computed(y) - B_expected(y)
    # Bx should be very close to 0 because both harmonic
    # fields are orthogonal to this direction at point y
    assert jnp.abs(B_diff[0]) < 1e-12
    # By is due to the toroidal field ~1/R -
    # this error is dominated by resolution
    assert jnp.abs(B_diff/B_expected(y))[1] < (1/n)**p
    # By is due to the poloidal field ~1/d -
    # this error is dominated by the approximation ɛ ≪ 1
    assert jnp.abs(B_diff/B_expected(y))[2] < ɛ
