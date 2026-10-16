"""The vacuum field of :func:`mrx.optimization.shape_ad.vacuum_two_form` and its shape derivative.

On QA at (12, 20, 12) p=2 the field, which comes from the co-closed solve on ``S_1`` alone, is the harmonic 2-form
of :func:`mrx.nullspace.compute_nullspaces` up to its scale, and the derivative of the mean rotational transform
with respect to every map coefficient matches central differences.
"""
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


QA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "wout_LandremanPaul2021_QA_lowres.nc")


def test_vacuum_field_and_its_shape_derivative():
    import mrx.optimization.shape_ad as sa
    from mrx.equilibria import build_map
    from mrx.geometry import build_sequence
    from mrx.nullspace import compute_nullspaces

    # the finite difference below divides the error of two solves by 2 eps = 2e-3, so they are solved to 1e-10:
    # at the default 1e-8 the comparison sits at its bound of 1e-6 (1.15e-6 and 0.9e-6 in two runs, 2026-10-13)
    seq, _ = build_sequence(QA, (12, 20, 12), 2, tol=1e-10)
    seed = sa.flux_seed(seq)
    h, info = sa.vacuum_two_form(seq, seed)
    assert info > 0
    compute_nullspaces(seq)
    harmonic = seq.nullspace(2)[0]
    scale = (h @ harmonic) / (harmonic @ harmonic)
    # two solves at seq.tol each: measured 1.3 seq.tol at tol 1e-8 (2026-10-12)
    assert jnp.linalg.norm(h - scale * harmonic) <= 1e2 * seq.tol * jnp.linalg.norm(h)

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
