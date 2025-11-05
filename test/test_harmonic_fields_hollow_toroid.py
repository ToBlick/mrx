# %%
# test_harmonic_fields_hollow_toroid.py
import jax
import jax.numpy as jnp

from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction, Pushforward

# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)


def test_harmonic_fields():
    Ip = 2.31
    It = 1.74
    Is = jnp.array([Ip, It])
    n = 5
    p = 3
    q = p + 2
    ɛ = 0.1
    π = jnp.pi
    μ0 = 1.0

    @jax.jit
    def F(x):
        """Hollow toroid."""
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

    evs, evecs = jnp.linalg.eigh(Seq.M2 @ Seq.dd2)
    assert jnp.sum(evs < 1e-10) == 2  # two harmonic fields
    assert jnp.min(evs) > -1e-10  # no negative eigenvalues

    def m2_orthonormalize(V, M):
        G = V.T @ M @ V              # 2x2 Gram
        R = jnp.linalg.cholesky(G)
        K = V @ jnp.linalg.inv(R)    # columns now M-orthonormal
        return K
    K = m2_orthonormalize(evecs[:, :2], Seq.M2)
    h1_dof = K[:, 0]
    h2_dof = K[:, 1]

    h1 = jax.jit(Pushforward(DiscreteFunction(
        h1_dof, Seq.Λ2, Seq.E2), Seq.F, 2))
    h2 = jax.jit(Pushforward(DiscreteFunction(
        h2_dof, Seq.Λ2, Seq.E2), Seq.F, 2))

    # Compute contour integrals:
    # contour wrapping around the enclosed tunnel poloidally:
    def c1(χ):
        r = jnp.ones_like(χ) * 0.5
        θ = χ
        z = jnp.zeros_like(χ)
        return Seq.F(jnp.array([r, θ, z]))

    # contour wrapping around the center tunnel toroidally:
    def c2(χ):
        r = jnp.ones_like(χ) * 0.5
        θ = jnp.ones_like(χ) * 0.5
        z = χ
        return Seq.F(jnp.array([r, θ, z]))

    def h_dl(function, curve):
        def h_dl(χ):
            return function(curve(χ)) @ jax.jacfwd(curve)(χ)
        return h_dl

    # Integrate h1 along contours using trapezoidal rule
    n_q = 256
    _χ = jnp.linspace(0, 1, n_q, endpoint=False)
    _w = jnp.ones(n_q) * (1/n_q)

    P = jnp.array([
        [jax.vmap(h_dl(h, c))(_χ) @ _w for c in (c1, c2)]
        for h in (h1, h2)
    ])
    # this matrix has entries ∫ h_i · dl_j

    # Coefficients of the harmonic fields:
    b = jnp.linalg.solve(P.T, μ0 * Is)
    b_dofs = b[0] * h1_dof + b[1] * h2_dof

    # assert that the solution is indeed harmonic:
    curl_b_dofs = Seq.weak_curl @ b_dofs
    div_b_dofs = Seq.strong_div @ b_dofs
    assert (curl_b_dofs @ Seq.M1 @ curl_b_dofs)**0.5 < 1e-10
    assert (div_b_dofs @ Seq.M3 @ div_b_dofs)**0.5 < 1e-10

    # check energy in the field
    energy = b_dofs @ Seq.M2 @ b_dofs / (2 * μ0)
    expected_energy = μ0 / 2 * Is @ jnp.linalg.solve(P.T @ P, Is)
    assert jnp.abs(energy - expected_energy) / expected_energy < 1e-6

    # in ɛ ≪ 1 limit, we know that B = μ0 (I_p eφ / 2πR + I_t eθ / 2πd),
    # where d is the distance to the centerline of the enclosed tunnel
    # and R the distance to the z-axis
    def B_expected(x):
        """Expected magnetic field in the thin-torus limit."""
        r, θ, ζ = x
        R = 1 + ɛ * (r + 1)/2 * jnp.cos(2 * π * θ)
        B_φ = μ0 * Ip / (2 * π * R)
        B_θ = μ0 * It / (2 * π * (ɛ * (r + 1)/2))

        sζ = jnp.sin(2 * π * ζ)
        cζ = jnp.cos(2 * π * ζ)
        sθ = jnp.sin(2 * π * θ)
        cθ = jnp.cos(2 * π * θ)

        Bx = -B_φ * sζ - B_θ * sθ * cζ * ɛ * (r + 1)/(2*R)
        By = - B_φ * cζ + B_θ * sθ * sζ * ɛ * (r + 1)/(2*R)
        Bz = B_θ * cθ * ɛ * (r + 1)/(2*R)

        return jnp.array([Bx, By, Bz])

    B_computed = jax.jit(Pushforward(DiscreteFunction(
        b_dofs, Seq.Λ2, Seq.E2), Seq.F, 2))

    return B_computed
