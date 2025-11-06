# %%
# test_quadrature.py
import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest
import numpy as np

from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import rotating_ellipse_map
from mrx.quadrature import trapezoidal_quad, composite_quad, spectral_quad

jax.config.update("jax_enable_x64", True)

@pytest.mark.parametrize("n", [6])
@pytest.mark.parametrize("p", [1, 2, 3, 4])
@pytest.mark.parametrize("bc_type", ["clamped", "periodic", "fourier", "constant"])
def test_quadrature(n, p, bc_type):
    """Test the quadrature by verifying the exact torus volume in the rotating ellipse mapping."""
    eps = 0.5
    kappa = 1.2
    nfp = 3
    q = 2*p

    Seq = DeRhamSequence(
        (n, n, n),
        (p, p, p),
        q,
        (bc_type, "periodic", "periodic"),
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

    # TODO: Test the quadrature on more complex tests.


@pytest.mark.parametrize("n", np.arange(1, 21))
def test_trapezoidal_quad_weight_sum(n):
    """Test that trapezoidal quadrature weights sum to 1.0 (length of [0,1] interval)."""
    x_q, w_q = trapezoidal_quad(n)
    weight_sum = jnp.sum(w_q)
    npt.assert_allclose(
        weight_sum, 1.0,
        rtol=1e-7,
        err_msg=f"Trapezoidal quadrature weights for n={n} should sum to 1.0"
    )


@pytest.mark.parametrize("p", np.arange(1, 11))
def test_spectral_quad_weight_sum(p):
    """Test that spectral quadrature weights sum to 1.0 (length of [0,1] interval)."""
    x_q, w_q = spectral_quad(p)
    weight_sum = jnp.sum(w_q)
    npt.assert_allclose(
        weight_sum, 1.0,
        rtol=1e-7,
        err_msg=f"Spectral quadrature weights for p={p} should sum to 1.0"
    )


@pytest.mark.parametrize("p", np.arange(1, 11))
@pytest.mark.parametrize("n_intervals", [2, 3, 5, 10])
def test_composite_quad_weight_sum(p, n_intervals):
    """Test that composite quadrature weights sum to the length of the domain."""
    # Create a knot vector T with n_intervals intervals on [0, 1]
    T = jnp.linspace(0, 1, n_intervals + 1)
    x_q, w_q = composite_quad(T, p)
    weight_sum = jnp.sum(w_q)
    domain_length = T[-1] - T[0]
    npt.assert_allclose(
        weight_sum, domain_length,
        rtol=1e-7,
        err_msg=f"Composite quadrature weights for p={p}, n_intervals={n_intervals} "
                f"should sum to {domain_length}"
    )
