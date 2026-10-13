"""Double-single arithmetic: the error-free transformations are exact under the compiler, and the kernels of
:mod:`mrx.double_single` reproduce a float64 reference computed on the host.

The first test is the one to run on a new device. Everything else in the float32 configuration rests on it.
"""
import jax
import jax.numpy as jnp
import numpy as np

import mrx.double_single as ds


def test_error_free_transformations_survive_the_compiler():
    """``two_sum`` and ``two_prod`` return the exact rounding error of a float32 sum and product on every
    pair, under ``jit``."""
    rng = np.random.default_rng(0)
    a, b = ((rng.standard_normal(100_000) * np.exp(rng.uniform(-8, 8, 100_000))).astype(np.float32)
            for _ in range(2))
    s, e = jax.jit(ds.two_sum)(jnp.asarray(a), jnp.asarray(b))
    assert np.array_equal(ds.to_host(ds.DS(s, e)), a.astype(np.float64) + b.astype(np.float64))
    p, e = jax.jit(ds.two_prod)(jnp.asarray(a), jnp.asarray(b))
    assert np.array_equal(ds.to_host(ds.DS(p, e)), a.astype(np.float64) * b.astype(np.float64))


def test_kernels_against_float64_on_the_host():
    """The dot product and a sparse apply agree with NumPy float64 to 1e-13, where float32 gives 1e-7."""
    rng = np.random.default_rng(1)
    n = 5000
    a, b = rng.standard_normal(n), rng.standard_normal(n)
    dot = float(ds.to_host(jax.jit(ds.dot)(ds.from_host(a), ds.from_host(b))))
    assert abs(dot - np.dot(a, b)) <= 1e-13 * np.linalg.norm(a) * np.linalg.norm(b)
    rows, cols, vals = rng.integers(0, 300, 4000), rng.integers(0, n, 4000), rng.standard_normal(4000)
    dense = np.zeros((300, n))
    np.add.at(dense, (rows, cols), vals)
    y = ds.to_host(ds.Rows.from_coo(*np.nonzero(dense), dense[np.nonzero(dense)], 300) @ ds.from_host(a))
    assert np.linalg.norm(y - dense @ a) <= 1e-13 * np.linalg.norm(dense @ a)
