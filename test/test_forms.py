"""Pushforward undoes pullback for every degree, on the li383 map.

``Pullback(g, F, k)`` takes a physical form ``g`` -- a function of the
PHYSICAL point -- and evaluates it at ``F(x)`` (``DF^T``, ``J DF^-1``,
``J``); ``Pushforward(f, F, k)`` is a function of the LOGICAL point giving
the physical components of the reference form ``f`` (``DF^-T``, ``DF / J``,
``1 / J``). Composed, ``pushforward(pullback(g))(x) = g(F(x))`` at every
logical point, for any map: no inverse map is needed. The forms are random
polynomials of the physical point, the points random away from the axis
(where ``DF`` is singular).
"""
import jax.numpy as jnp
import numpy as np

from mrx.differential_forms import Pullback, Pushforward
from mrx.precision import eps


def test_pushforward_inverts_pullback(seq):
    rng = np.random.default_rng(3)
    pts = jnp.asarray(np.column_stack([rng.uniform(0.15, 0.9, 6), rng.uniform(0.05, 0.95, 6),
                                       rng.uniform(0.05, 0.95, 6)]))
    for k in range(4):
        n_comp = 1 if k in (0, 3) else 3
        c = jnp.asarray(rng.standard_normal((n_comp, 4)))

        def g(y, c=c, n_comp=n_comp):      # affine in the physical point, one row per component
            v = c[:, 0] + c[:, 1:] @ y
            return v[0] if n_comp == 1 else v
        back = Pullback(g, seq.map, k)
        forth = Pushforward(back, seq.map, k)
        got = np.asarray(jnp.stack([forth(x) for x in pts]))
        want = np.asarray(jnp.stack([g(seq.map(x)) for x in pts]))
        err = np.abs(got - want).max() / np.abs(want).max()
        # DF^-T DF^T and DF DF^-1 / J * J: a 3x3 inverse and two products, in float64 (the map)
        assert err < 1e3 * eps(), f"k={k}: pushforward(pullback(g)) off by {err:.2e}"
