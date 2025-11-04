# %%
# test_spline_bases.py
import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest
from mrx.spline_bases import SplineBasis

jax.config.update("jax_enable_x64", True)

# Helper fixture and parametrization
@pytest.fixture(params=["clamped", "periodic"])
def basis(request):
    """Return a spline basis of given type."""
    n, p = 10, 3
    return SplineBasis(n, p, request.param), request.param

def test_partition_of_unity(basis):
    """Test the partition of unity property of the spline basis."""
    spl, typ = basis
    xs = jnp.linspace(0.0, 1.0, 200)
    i = jnp.arange(spl.n)

    def sum_basis(x):
        vals = jax.vmap(lambda j: spl(x, j))(i)
        return jnp.sum(vals)

    sums = jax.vmap(sum_basis)(xs)
    npt.assert_allclose(
        sums, 1.0, atol=1e-12, err_msg=f"Partition of unity fails ({typ})"
    )


# Positivity
def test_basis_positivity(basis):
    """Test the positivity property of the spline basis."""
    spl, typ = basis
    xs = jnp.linspace(0.0, 1.0, 200)
    i = jnp.arange(spl.n)

    def eval_basis(x):
        return jax.vmap(lambda j: spl(x, j))(i)

    vals = jax.vmap(eval_basis)(xs)  # (nq, n)
    assert jnp.all(vals >= -1e-14), f"Negative basis value detected ({typ})"


# L2 projection of sin(2πx)
def test_spline_projection_sin(basis):
    """Test the L2 projection of the sine function using the spline basis.

    Parameters
    ----------
    basis : tuple
        A tuple containing the spline basis and the type of spline.
    """
    spl, typ = basis
    n = spl.n
    i = jnp.arange(n)

    # Midpoint quadrature on [0,1]
    nq = 256
    xq = (jnp.arange(nq) + 0.5) / nq
    w = jnp.ones(nq) / nq

    # Build basis matrix N_qi
    def basis_at_x(x):
        return jax.vmap(lambda j: spl(x, j))(i)
    N_qi = jax.vmap(basis_at_x)(xq)  # (nq, n)

    # Mass matrix and RHS
    NW = N_qi * w[:, None]
    M = N_qi.T @ NW
    f_q = jnp.sin(2 * jnp.pi * xq)
    b = (f_q * w) @ N_qi

    coeffs = jnp.linalg.solve(M, b)

    # Evaluate projection
    xs = jnp.linspace(0.0, 1.0, 400)
    def spline_eval(x):
        Ni = jax.vmap(lambda j: spl(x, j))(i)
        return jnp.dot(coeffs, Ni)
    f_approx = jax.vmap(spline_eval)(xs)
    f_true = jnp.sin(2 * jnp.pi * xs)

    err = jnp.max(jnp.abs(f_true - f_approx))
    print(err)
    npt.assert_allclose(
        err, 0.0, atol=0.01,
        err_msg=f"L_inf error too large ({typ}): {err}"
    )
