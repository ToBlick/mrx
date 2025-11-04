# %%
"""
Harmonic fields on a hollow toroid.
"""
import os
import time
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction, Pushforward

# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)
# Create output directory for figures
os.makedirs("script_outputs", exist_ok=True)

n = 5
p = 3
q = p + 2
# %%


@partial(jax.jit, static_argnames=["n", "p", "q"])
def get_err(Ip, It, n, p, q, μ0=1.0):
    ɛ = 1/3
    π = jnp.pi

    def F(x):
        """Hollow toroid."""
        r, θ, z = x
        R = 1 + ɛ * (r + 1)/2 * jnp.cos(2 * π * θ)
        return jnp.array([R * jnp.cos(2 * π * z),
                          -R * jnp.sin(2 * π * z),
                          ɛ * (r + 1)/2 * jnp.sin(2 * π * θ)])

    # Set up finite element spaces
    ns = (n, n, n)
    ps = (p, p, p)
    types = ("clamped", "periodic", "periodic")
    Seq = DeRhamSequence(ns, ps, q, types, F, polar=False, dirichlet=True)
    Seq.evaluate_1d()
    Seq.assemble_M1()
    Seq.assemble_M2()
    Seq.assemble_d1()
    Seq.assemble_dd2()

    evs, evecs = jnp.linalg.eigh(Seq.M2 @ Seq.dd2)
    assert jnp.sum(evs < 1e-10) == 2  # two harmonic fields
    assert jnp.min(evs) > -1e-10  # no negative eigenvalues

    def m2_orthonormalize(V, M):
        G = V.T @ M @ V              # 2x2 Gram
        G = 0.5 * (G + G.T)          # symmetrize (numerical hygiene)
        R = jnp.linalg.cholesky(G + 1e-12 * jnp.eye(2))
        K = V @ jnp.linalg.inv(R)    # columns now M-orthonormal
        return K

    K = m2_orthonormalize(evecs[:, :2], Seq.M2)

    h1 = Pushforward(DiscreteFunction(K[0], Seq.Λ2, Seq.E2), Seq.F, 2)
    h2 = Pushforward(DiscreteFunction(K[1], Seq.Λ2, Seq.E2), Seq.F, 2)

    # Next, compute contour integrals:
    # contour wrapping around the tunnel poloidally:
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
    I = jnp.array([Ip, It])

    P = jnp.array([
        [jax.vmap(h_dl(h, c))(_χ) @ _w for c in (c1, c2)]
        for h in (h1, h2)
    ])
    # this matrix has entries ∫ h_i · dl_j

    # Coefficients of the harmonic fields:
    a = jnp.linalg.solve(P.T, μ0 * I)
    b_dofs = a[0] * evecs[:, 0] + a[1] * evecs[:, 1]

    # assert that the solution is indeed harmonic:
    Seq.assemble_M3()
    Seq.assemble_d2()
    curl_b_dofs = Seq.weak_curl @ b_dofs
    div_b_dofs = Seq.strong_div @ b_dofs
    assert (curl_b_dofs @ Seq.M1 @ curl_b_dofs)**0.5 < 1e-10
    assert (div_b_dofs @ Seq.M3 @ div_b_dofs)**0.5 < 1e-10

    # check energy in the field"
    energy = b_dofs @ Seq.M2 @ b_dofs / (2 * μ0)
    expected_energy = 0.5 * μ0 * I @ jnp.linalg.solve(P.T @ P, I)
    err = jnp.abs(energy - expected_energy) / expected_energy
    return


def main():
    """Main function to run the analysis."""


if __name__ == "__main__":
    main()

# %%
