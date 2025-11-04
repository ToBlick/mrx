# %%
# test_helmholtz_decomposition.py
import jax
import jax.numpy as jnp
import numpy.testing as npt

from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import rotating_ellipse_map

jax.config.update("jax_enable_x64", True)


def test_helmholtz_decomposition():
    """
    Test the Helmholtz decomposition using a rotating ellipse mapping.
    """
    eps = 0.5
    kappa = 1.0
    nfp = 3

    Seq = DeRhamSequence(
        (4, 4, 4),
        (3, 3, 3),
        2*3,
        ("clamped", "periodic", "periodic"),
        rotating_ellipse_map(eps, kappa, nfp),
        polar=True,
        dirichlet=True
    )

    Seq.evaluate_1d()
    Seq.assemble_all()
    Seq.assemble_leray_projection()

    def B(x):
        r, θ, _ = x
        x1, x2, _ = Seq.F(x)
        φ = jnp.arctan2(x2, x1)
        R = 1 + r * eps * jnp.cos(2 * jnp.pi * θ)
        return jnp.array([jnp.sin(φ), -jnp.cos(φ), 0]) / R

    # Project B into the discrete space
    B_hat = Seq.P_Leray @ jnp.linalg.solve(Seq.M2, Seq.P2(B))
    B_hat /= (B_hat @ Seq.M2 @ B_hat)**0.5  # normalize
    A_hat = jnp.linalg.solve(Seq.dd1, Seq.weak_curl @ B_hat)
    # A_hat is approx. 0 since B is the harmonic form
    npt.assert_allclose(
        (A_hat @ Seq.M1 @ A_hat)**0.5, 0.0,
        atol=1e-3,
        err_msg="|A| is not approximately zero for harmonic field"
    )

    # Harmonic form is also given by the zero eigenvector of dd2:
    _, eigvecs = jnp.linalg.eigh(Seq.M2 @ Seq.dd2)
    B_harm = eigvecs[:, 0]
    B_harm /= (B_harm @ Seq.M2 @ B_harm)**0.5  # normalize

    # Check that B_hat and B_harm are aligned
    alignment = jnp.abs(B_hat @ Seq.M2 @ B_harm)
    npt.assert_allclose(
        alignment, 1.0,
        atol=1e-6,
        err_msg="Projected field does not match harmonic eigenvector"
    )
