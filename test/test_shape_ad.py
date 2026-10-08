"""The vacuum field of :func:`mrx.optimization.shape_ad.vacuum_two_form` and its shape derivative.

On QA at (8, 12, 12) p=2 the field is the harmonic 2-form of :func:`mrx.nullspace.compute_nullspaces` up to
its scale and equals the field of the full Laplacian solve, a gradient added to the potential does not change it,
and the derivative of the mean rotational transform with respect to every map coefficient matches
central differences. Shape gradients need float64 (see tutorial 6), so the test is skipped in float32.
"""
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mrx.precision import DTYPE

pytestmark = pytest.mark.skipif(DTYPE != jnp.float64, reason="shape derivatives need MRX_DTYPE=float64")
QA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "wout_LandremanPaul2021_QA_lowres.nc")


def test_vacuum_field_and_its_shape_derivative():
    import mrx.optimization.shape_ad as sa
    from mrx.equilibria import build_map
    from mrx.geometry import build_sequence
    from mrx.nullspace import compute_nullspaces

    seq, _ = build_sequence(QA, (8, 12, 12), 2)
    seed = sa.flux_seed(seq)
    h, info = sa.vacuum_two_form(seq, seed)
    assert info > 0
    compute_nullspaces(seq)
    harmonic = seq.nullspace(2)[0]
    scale = (h @ harmonic) / (harmonic @ harmonic)
    assert jnp.linalg.norm(h - scale * harmonic) <= 1e-8 * jnp.linalg.norm(h)
    # The same field as the full Hodge-Laplacian solve
    h_L = seed - seq.G[1] @ seq.L[1].solve(seq.D[1].T @ seed)
    assert jnp.linalg.norm(h - h_L) <= 1e-8 * jnp.linalg.norm(h)
    # The S_1 solve leaves the gradient part of a free; it cannot reach h because G_1 G_0 = 0
    phi = jax.random.normal(jax.random.PRNGKey(0), (seq.n(0),), dtype=h.dtype)
    gradient = seq.G[0] @ phi
    assert jnp.linalg.norm(seq.G[1] @ gradient) <= 1e-12 * jnp.linalg.norm(gradient)

    _, info = build_map(seq.equilibrium, seq, stellarator_symmetric=seq.half_period)
    shape = sa.BoundaryShape.from_coefficients(seq, info["raw_R"], info["raw_Z"], seq.nfp, extension="harmonic",
                                               free="all")

    def mean_iota(x):
        R, Z, _, _ = shape.map_coefficients(seq, 1e-2 * x.reshape((2,) + shape.raw_R.shape))
        sq = sa.with_geometry(seq, sa.cylindrical_geometry(seq, R, Z, seq.nfp))
        return sa.mean_iota(sq, sa.vacuum_two_form(sq, seed)[0])

    x = jnp.zeros(2 * shape.raw_R.size)
    d = jnp.asarray(np.random.default_rng(0).standard_normal(x.size))
    f = eqx.filter_jit(mean_iota)
    eps = 1e-3
    fd = (f(x + eps * d) - f(x - eps * d)) / (2 * eps)
    assert abs(jax.grad(mean_iota)(x) @ d - fd) <= 1e-6 * abs(fd)
