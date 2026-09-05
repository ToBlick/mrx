"""Pullback undoes pushforward for every degree, on the li383 map.

``Pushforward`` carries a k-form's reference components to the physical
frame (``DF^{-T}``, ``DF / J``, ``1 / J``), ``Pullback`` brings a physical
form back (``DF^T``, ``adj(DF)``, ``J``); composed they are the identity at
every logical point, whatever the map. Checked on the session field and its
current, the weak pressure's 0-form and a 3-form, at random logical points
away from the axis and the wall (where ``DF`` of the spline map is not
defined).
"""
import jax.numpy as jnp
import numpy as np

from mrx.differential_forms import DiscreteFunction, Pullback, Pushforward


def test_pullback_inverts_pushforward(seq, b0):
    J = seq.apply_weak_curl(b0, dirichlet=True)
    g = seq.apply_inverse_mass_matrix(seq.magnitude_squared_load(b0), 0, dirichlet=False)
    rho = seq.scalar_product_load(g, g, 3, 0, 0, False, False, False)
    forms = {0: (g, seq.basis_0, seq.E(0, False)), 1: (J, seq.basis_1, seq.E(1, True)),
             2: (b0, seq.basis_2, seq.E(2, True)), 3: (rho, seq.basis_3, seq.E(3, False))}
    rng = np.random.default_rng(3)
    pts = jnp.asarray(np.column_stack([rng.uniform(0.15, 0.9, 6), rng.uniform(0, 1, 6),
                                       rng.uniform(0, 1, 6)]))
    for k, (dof, basis, E) in forms.items():
        f = DiscreteFunction(jnp.asarray(dof), basis, E)
        back = Pullback(Pushforward(f, seq.map, k), seq.map, k)
        got = np.asarray(jnp.stack([back(x) for x in pts]))
        want = np.asarray(jnp.stack([f(x) for x in pts]))
        err = np.abs(got - want).max() / np.abs(want).max()
        assert err < 1e3 * seq.tol, f"k={k}: pullback(pushforward(f)) off by {err:.2e}"
