# %%
# test_quadrature.py
import jax
import jax.numpy as jnp
import numpy.testing as npt

from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import rotating_ellipse_map

jax.config.update("jax_enable_x64", True)


def test_quadrature():
    """Check torus volume."""
    eps = 0.5
    kappa = 1.2
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

    # Volume of torus is integral (J dx)
    vol = Seq.Q.w @ Seq.J_j * nfp

    npt.assert_allclose(
        vol, jnp.pi**2 * eps**2 * (2 - (1 - kappa)**2),
        rtol=1e-6,
        err_msg="Torus volume quadrature incorrect"
    )
