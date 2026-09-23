"""The reverse-mode shape derivative of the vacuum field (:mod:`mrx.shape_ad`).

On the session li383 sequence: the differentiable harmonic 2-form at the
start boundary is the production one, and the gradient of a rotational-
transform objective with respect to the boundary coefficients matches a
central finite difference along one smooth stellarator-symmetric boundary
direction. One compiled program (``value_and_grad``); float64 only, where
the difference resolves the derivative to the solve tolerance.
"""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import mrx
from mrx.gvec import build_gvec_map
from mrx.shape_ad import (BoundaryShape, flux_ratio_iota, flux_seed, flux_surface_radius,
                          vacuum_two_form, with_geometry)


def _objective(beta, seq, shape, seed, s):
    geom, _ = shape.geometry(seq, beta)
    sq = with_geometry(seq, geom)
    h, _ = vacuum_two_form(sq, seed)
    iota, _ = flux_ratio_iota(sq, h, flux_surface_radius(sq, h, s))
    return jnp.sum(iota ** 2), h


_value_and_grad = eqx.filter_jit(jax.value_and_grad(_objective, has_aux=True))


@pytest.mark.skipif(mrx.DTYPE != jnp.float64, reason="the finite difference needs float64")
def test_shape_gradient_matches_finite_difference(seq):
    _, info = build_gvec_map(seq.equilibrium, seq, stellarator_symmetric=seq.half_period)
    shape = BoundaryShape.from_coefficients(seq, info["raw_R"], info["raw_Z"], seq.nfp, info["sign"])
    seed, s = flux_seed(seq), jnp.array([0.25, 0.5, 0.81])
    zero = jnp.zeros((2,) + shape.raw_R.shape[1:])
    (_, h), g = _value_and_grad(zero, seq, shape, seed, s)

    h = h / seq.l2_norm(h, 2)
    h_prod = seq.nullspace(2, True)[0]
    h = h * jnp.sign(h @ seq.apply_mass_matrix(h_prod, 2))
    assert float(seq.l2_norm(h - h_prod, 2)) < 1e-6

    # a smooth boundary change: R even (cosines), Z odd (sines) in (theta, zeta)
    n_t, n_z = zero.shape[1:]
    t, z = np.meshgrid(np.arange(n_t) / n_t, np.arange(n_z) / n_z, indexing="ij")
    d = jnp.asarray(np.stack([np.cos(2 * np.pi * (2 * t - z)) + 0.5 * np.cos(2 * np.pi * t),
                              np.sin(2 * np.pi * (2 * t - z)) - 0.5 * np.sin(2 * np.pi * (3 * t + z))]))
    step = 1e-4 * float(jnp.max(jnp.abs(shape.raw_Z)))          # 1e-4 of the minor radius
    (f_plus, _), _ = _value_and_grad(step * d, seq, shape, seed, s)
    (f_minus, _), _ = _value_and_grad(-step * d, seq, shape, seed, s)
    fd = (float(f_plus) - float(f_minus)) / (2.0 * step)
    ad = float(jnp.sum(g * d))
    print(f"\n  d/dt objective: AD {ad:+.10e}, central FD {fd:+.10e}, rel. diff {abs(fd - ad) / abs(ad):.1e}")
    assert abs(fd - ad) < 1e-5 * abs(ad)
