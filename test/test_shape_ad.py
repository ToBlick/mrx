"""The vacuum field of :func:`mrx.optimization.shape_ad.vacuum_two_form` and its shape derivative.

On the li383 sequence of the suite the field, which comes from the co-closed solve on ``S_1`` alone, is harmonic:
closed to round-off (the topological curl) and co-closed to the tolerance of the solve. The derivative of the mean
rotational transform with respect to every map coefficient matches Richardson-extrapolated central differences. The
parametrisation holds the volume and the aspect ratio exactly, and the hand-derived ``|B|``, ``B . grad|B|`` and
``(B x grad r) . grad|B|`` of the quasi-symmetry residual agree with automatic differentiation of the pushed-forward
field.
"""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import mrx

# derivatives through the solves are taken in float64
pytestmark = pytest.mark.skipif(mrx.DTYPE != jnp.float64, reason="needs MRX_DTYPE=float64")

#: The finite differences below divide the error of two solves by 2 eps, so they are solved to 1e-10: at the default
#: 1e-8 the comparison sat at its bound of 1e-6 on QA (1.15e-6 and 0.9e-6 in two runs, 2026-10-13).
FD_TOL = 1e-10
#: The central difference has an eps^2 error: on li383 (16, 32, 12) it is 3.14e-6 of the derivative at eps = 1e-3 and
#: 7.16e-7 at 5e-4, below that the solves dominate (1.5e-6 at 2.5e-4). Richardson's combination of 1e-3 and 5e-4
#: removes the eps^2 term and leaves 9.1e-8 (2026-10-20), against the bound of 1e-6.
FD_EPS = 1e-3


def test_vacuum_field_and_its_shape_derivative(seq):
    import mrx.optimization.shape_ad as sa
    from mrx.differential_forms import DiscreteFunction, Pushforward
    from mrx.operators import dual_norm
    from mrx.precision import eps as round_off

    seed = sa.flux_seed(seq)
    h, info = sa.vacuum_two_form(seq, seed)
    assert info > 0
    # closed: h = seed - G_1 a with a closed seed. Co-closed: G_1^T M_2 h is the residual of the solve for a
    odd = seq.odd                                   # the field is odd
    closed = float(jnp.linalg.norm(odd.G[2] @ h) / jnp.linalg.norm(h))
    nu = dual_norm(odd, 1)
    coclosed = float(nu(odd.D[1].T @ h) / nu(odd.D[1].T @ seed))
    print(f"\n  |G_2 h| / |h| {closed:.2e}, |G_1^T M_2 h| / |G_1^T M_2 seed| {coclosed:.2e}")
    assert closed <= round_off(1e4)
    assert coclosed <= 1e2 * seq.tol

    shape = sa.BoundaryShape.from_coefficients(seq, seq.map.raw_R, seq.map.raw_Z, seq.nfp)

    def mean_iota(x):
        R, Z = shape.map_coefficients(seq, 1e-2 * x.reshape((2,) + shape.raw_R.shape))
        sq = seq.with_geometry(sa.cylindrical_geometry(seq, R, Z, seq.nfp))
        return sa.mean_iota(sq, sa.vacuum_two_form(sq, seed, tol=FD_TOL)[0])

    x = jnp.zeros(2 * shape.raw_R.size)
    d = jnp.asarray(np.random.default_rng(0).standard_normal(x.size))
    f = eqx.filter_jit(mean_iota)
    def central(eps):
        return (f(x + eps * d) - f(x - eps * d)) / (2 * eps)
    fd = (4 * central(FD_EPS / 2) - central(FD_EPS)) / 3
    assert abs(jax.grad(mean_iota)(x) @ d - fd) <= 1e-6 * abs(fd)

    # the volume and the aspect ratio of a perturbed shape are those asked for, to round-off
    aspect = 6.0
    held = sa.BoundaryShape.from_coefficients(seq, seq.map.raw_R, seq.map.raw_Z, seq.nfp, aspect=aspect)
    beta = 1e-2 * jnp.asarray(np.random.default_rng(2).standard_normal((2,) + held.raw_R.shape))
    R, Z = held.map_coefficients(seq, beta)
    V, _, S = sa.section_moments(seq, R, Z, seq.nfp)
    assert abs(float(V - held.volume)) <= round_off(1e3) * float(V)
    assert abs(float(seq.nfp * V / (2.0 * np.sqrt(np.pi) * S ** 1.5)) - aspect) <= round_off(1e3) * aspect

    # the terms of the quasi-symmetry residual at quadrature points away from the axis, against jax.grad of |B|
    sq = seq.with_geometry(sa.cylindrical_geometry(seq, R, Z, seq.nfp))
    h, _ = sa.vacuum_two_form(sq, seed)
    B_mag, parallel, cross, _, _ = sa._residual_terms(sq, h, R, Z, seq.nfp, sa._tables(sq))
    field = Pushforward(DiscreteFunction(h, sq.odd.basis_2, sq.odd.E(2)), sq.map, 2)
    away = np.flatnonzero(np.asarray(seq.quad.x[:, 0]) > 0.3)
    for q in np.random.default_rng(3).choice(away, 6, replace=False):
        xq = seq.quad.x[q]
        DPhi_T = jax.jacfwd(sq.map)(xq).T
        grad = jnp.linalg.solve(DPhi_T, jax.grad(lambda y: jnp.linalg.norm(field(y)))(xq))   # physical gradients
        grad_r = jnp.linalg.solve(DPhi_T, jnp.array([1.0, 0.0, 0.0]))
        b = field(xq)
        want = (jnp.linalg.norm(b), b @ grad, jnp.cross(b, grad_r) @ grad)
        for got, ref in zip((B_mag[q], parallel[q], cross[q]), want):
            assert abs(float(got - ref)) <= round_off(1e4) * (abs(float(ref)) + float(jnp.linalg.norm(b) ** 2))
