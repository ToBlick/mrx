"""The vacuum field of :func:`mrx.optimization.shape_ad.vacuum_two_form` and its shape derivative.

On QA at (12, 20, 12) p=2 the field, which comes from the co-closed solve on ``S_1`` alone, is harmonic: closed to
round-off (the topological curl) and co-closed to the tolerance of the solve. The derivative of the mean rotational
transform with respect to every map coefficient matches central differences.
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
    from mrx.operators import dual_norm
    from mrx.precision import eps as round_off

    # the finite difference below divides the error of two solves by 2 eps = 2e-3, so they are solved to 1e-10:
    # at the default 1e-8 the comparison sits at its bound of 1e-6 (1.15e-6 and 0.9e-6 in two runs, 2026-10-13)
    seq = build_sequence(QA, (12, 20, 12), 2, tol=1e-10)
    seed = sa.flux_seed(seq)
    h, info = sa.vacuum_two_form(seq, seed)
    assert info > 0
    # closed: h = seed - G_1 a with a closed seed. Co-closed: G_1^T M_2 h is the residual of the solve for a
    closed = float(jnp.linalg.norm(seq.G[2] @ h) / jnp.linalg.norm(h))
    nu = dual_norm(seq, 1)
    coclosed = float(nu(seq.D[1].T @ h) / nu(seq.D[1].T @ seed))
    print(f"\n  |G_2 h| / |h| {closed:.2e}, |G_1^T M_2 h| / |G_1^T M_2 seed| {coclosed:.2e}")
    assert closed <= round_off(1e4)
    assert coclosed <= 1e2 * seq.tol

    _, info = build_map(seq.equilibrium, seq, stellarator_symmetric=seq.half_period)
    shape = sa.BoundaryShape.from_coefficients(seq, info["raw_R"], info["raw_Z"], seq.nfp)

    def mean_iota(x):
        R, Z = shape.map_coefficients(seq, 1e-2 * x.reshape((2,) + shape.raw_R.shape))
        sq = sa.with_geometry(seq, sa.cylindrical_geometry(seq, R, Z, seq.nfp))
        return sa.mean_iota(sq, sa.vacuum_two_form(sq, seed)[0])

    x = jnp.zeros(2 * shape.raw_R.size)
    d = jnp.asarray(np.random.default_rng(0).standard_normal(x.size))
    f = eqx.filter_jit(mean_iota)
    eps = 1e-3
    fd = (f(x + eps * d) - f(x - eps * d)) / (2 * eps)
    assert abs(jax.grad(mean_iota)(x) @ d - fd) <= 1e-6 * abs(fd)
