"""Low-level spline-basis tests.

These build small ``SplineBasis`` objects directly — no DeRham sequence.
"""

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from mrx.spline_bases import SplineBasis, TensorBasis

jax.config.update("jax_enable_x64", True)

N, P = 10, 3


@pytest.fixture(params=["clamped", "periodic", "constant", "fourier"])
def basis(request):
    return SplineBasis(N, P, request.param), request.param


def _vals_on_grid(spl, xs):
    i = jnp.arange(spl.n)
    return jax.vmap(lambda x: jax.vmap(lambda j: spl(x, j))(i))(xs)


def test_partition_of_unity(basis):
    spl, typ = basis
    if typ in ("fourier", "constant"):
        pytest.skip(f"{typ} basis is not a partition of unity")
    xs = jnp.linspace(0.0, 1.0, 200)
    vals = _vals_on_grid(spl, xs)
    npt.assert_allclose(jnp.sum(vals, axis=1), 1.0, atol=1e-12)


def test_positivity(basis):
    spl, typ = basis
    if typ == "fourier":
        pytest.skip("fourier basis is not pointwise positive")
    xs = jnp.linspace(0.0, 1.0, 200)
    vals = _vals_on_grid(spl, xs)
    assert jnp.all(vals >= -1e-14), f"negative value for {typ}"


def test_knot_roundtrip(basis):
    """Constructing with the same knot vector reproduces the basis."""
    spl, typ = basis
    spl2 = SplineBasis(spl.n, spl.p, spl.type, spl.T)
    xs = jnp.linspace(0.0, 1.0, 50)
    npt.assert_allclose(_vals_on_grid(spl, xs),
                        _vals_on_grid(spl2, xs), atol=1e-12)


def test_getitem_matches_call(basis):
    spl, _ = basis
    xs = jnp.linspace(0.0, 1.0, 50)
    for j in range(min(5, spl.n)):
        npt.assert_allclose(
            jax.vmap(spl[j])(xs),
            jax.vmap(lambda x, j=j: spl(x, j))(xs),
            atol=1e-12,
        )


def test_bad_init_rejected():
    with pytest.raises(ValueError):
        SplineBasis(N, P, "not-a-type")
    with pytest.raises(ValueError):
        SplineBasis(N, N, "clamped")  # p >= n not allowed
    with pytest.raises(ValueError):
        SplineBasis(N, 2 * N, "fourier")


def test_tensor_basis_evaluate_matches_product():
    tb = TensorBasis([
        SplineBasis(5, 2, "clamped"),
        SplineBasis(4, 2, "periodic"),
        SplineBasis(6, 2, "clamped"),
    ])
    assert tb.n == 5 * 4 * 6
    x = jnp.array([0.5, 0.5, 0.5])
    for lin in (0, 7, 23, tb.n - 1):
        i, j, k = jnp.unravel_index(lin, tb.shape)
        npt.assert_allclose(
            tb.evaluate(x, lin),
            tb.bases[0](x[0], i) * tb.bases[1](x[1], j) * tb.bases[2](x[2], k),
            atol=1e-12,
        )


def test_tensor_basis_wrong_rank_rejected():
    with pytest.raises(ValueError, match="exactly 3 bases"):
        TensorBasis([SplineBasis(5, 2, "clamped")])
