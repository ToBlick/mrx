"""Double-single arithmetic: the error-free transformations are exact under the compiler, and the kernels of
:mod:`mrx.double_single` reproduce a float64 reference computed on the host.

Run them all on a new device. Everything else in the float32 configuration rests on the first; the others catch a
compiler that breaks the transformations only inside a larger fused kernel, as XLA:CPU on arm64 does when
``two_prod`` has no optimization barrier.
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


def test_sum_factorised_apply_against_float64_on_the_host():
    """The sum-factorised apply (``along`` on each axis and the pointwise ``contract``) agrees with NumPy float64 to
    1e-13 under ``jit``, where float32 gives 1e-7."""
    rng = np.random.default_rng(2)
    n_comp, ns, qs = 3, (6, 8, 5), (9, 12, 7)

    def banded(n, q, p=2):        # (n_comp, N, Q): the nonzero pattern of a degree-p spline table
        t = np.zeros((n_comp, n, q), dtype=np.float32)
        for c in range(n_comp):
            for i in range(n):
                start, stop = max(0, (i - p) * q // n), min(q, (i + 1) * q // n + 1)
                t[c, i, start:stop] = rng.uniform(0.1, 1.0, stop - start)
        return t

    dense = [banded(ns[a], qs[a]) for a in range(3)]
    tables = tuple(ds.Tables.from_dense(t) for t in dense)
    shapes = (ns,) * n_comp
    weights = rng.standard_normal(qs + (n_comp, n_comp)).astype(np.float32)
    x = rng.standard_normal(n_comp * int(np.prod(ns)))
    apply = jax.jit(lambda v: ds.sumfact(tables, tables, shapes, shapes, jnp.asarray(weights), v))
    y = ds.to_host(apply(ds.from_host(x)))
    # the einsum path of mrx.mass.sumfact_apply, in float64
    R, T, Z = (t.astype(np.float64) for t in dense)
    u = np.einsum("cijk,cia,cjb,ckd->abdc", x.reshape((n_comp,) + ns), R, T, Z)
    v = np.einsum("abdrc,abdc->rabd", weights.astype(np.float64), u)
    ref = np.einsum("ria,rjb,rkd,rabd->rijk", R, T, Z, v).reshape(-1)
    assert np.linalg.norm(y - ref) <= 1e-13 * np.linalg.norm(ref)
