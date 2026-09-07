"""Pullback undoes pushforward for every degree, on the analytic torus.

``Pushforward(f, F, k)`` is a function of the LOGICAL point: the physical
components of the reference form ``f`` at ``x`` (``DF^{-T}``, ``DF / J``,
``1 / J``). ``Pullback(g, F, k)`` takes a physical form ``g`` -- a function
of the PHYSICAL point -- and evaluates it at ``F(x)`` (``DF^T``,
``adj(DF)``, ``J``). Composed through the inverse map,
``pullback(pushforward(f) o F^-1) = f`` at every logical point. The donut
torus has that inverse in closed form; the forms are random DoF vectors of
the toroid session sequence, the points random away from the axis (where
``DF`` is singular) and off the seams (where the inverse's angles wrap).
"""
import jax.numpy as jnp
import numpy as np

from mrx.differential_forms import DiscreteFunction, Pullback, Pushforward
from test.conftest import TORUS_EPSILON, TORUS_R0


def _torus_inverse(y):
    """``F^-1`` of ``mrx.mappings.toroid_map(epsilon, R0)`` (kappa = 1)."""
    two_pi = 2 * jnp.pi
    R = jnp.sqrt(y[0] ** 2 + y[1] ** 2)
    zeta = (-jnp.arctan2(y[1], y[0]) / two_pi) % 1.0
    a, b = (R - TORUS_R0) / TORUS_EPSILON, y[2] / TORUS_EPSILON
    return jnp.array([jnp.sqrt(a ** 2 + b ** 2), (jnp.arctan2(b, a) / two_pi) % 1.0, zeta])


def test_pullback_inverts_pushforward(toroid, torus_map):
    rng = np.random.default_rng(3)
    pts = jnp.asarray(np.column_stack([rng.uniform(0.15, 0.9, 6), rng.uniform(0.05, 0.95, 6),
                                       rng.uniform(0.05, 0.95, 6)]))
    for k in range(4):
        basis = getattr(toroid, f"basis_{k}")
        E = toroid.E(k, False)
        f = DiscreteFunction(jnp.asarray(rng.standard_normal(E.shape[1])), basis, E)
        pushed = Pushforward(f, torus_map, k)
        back = Pullback(lambda y: pushed(_torus_inverse(y)), torus_map, k)
        got = np.asarray(jnp.stack([back(x) for x in pts]))
        want = np.asarray(jnp.stack([f(x) for x in pts]))
        err = np.abs(got - want).max() / np.abs(want).max()
        assert err < 1e3 * toroid.tol, f"k={k}: pullback(pushforward(f)) off by {err:.2e}"
