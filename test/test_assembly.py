"""Matrix-free assembly against a quadrature oracle, on the session geometry.

``mrx.mass`` applies every mass by sum factorisation with the metric weight
formed from ``DF`` inside the kernel. The oracle evaluates the same k-form at
the quadrature points through the basis tables, multiplies by the metric
weight of the space, and integrates back against the basis -- two tensor
contractions that share nothing with the fused kernel. One random vector per
degree: the identity is linear, so one vector tests the whole operator.
"""
import jax.numpy as jnp
import numpy as np
import pytest

import mrx
from mrx.quadrature import integrate_against

# Roundoff identity relative to the size of the result: 1e3 eps
# (2.2e-13 f64 / 1.2e-4 f32).
IDENT = mrx.eps(1e3)


def _weight(seq, k):
    """The mass weight of k-forms at the quadrature points."""
    J = seq.jacobian_j
    if k == 0:
        return J
    if k == 3:
        return 1.0 / J
    if k == 1:
        return seq.metric_inv_jkl * J[:, None, None]
    return seq.metric_jkl / J[:, None, None]


def _oracle_mass(seq, x, k, dirichlet):
    """``M_k x`` by evaluate -> weight -> integrate, through the extraction."""
    u_q = seq.evaluate_at_quadrature(x, k, dirichlet)               # (n_q, d)
    w = _weight(seq, k)
    wu = u_q * w[:, None] if w.ndim == 1 else jnp.einsum('qij,qj->qi', w, u_q)
    comp_info, comp_shapes = seq._form_comp_info(k)
    raw = integrate_against(wu * seq.quad.w[:, None], comp_info, comp_shapes, seq.quad.shape)
    return seq.E(k, dirichlet) @ raw


@pytest.mark.parametrize("k", (0, 1, 2, 3))
def test_mass_apply_matches_quadrature_oracle(seq, k):
    dirichlet = True
    x = jnp.asarray(np.random.default_rng(k).standard_normal(seq.n(k, dirichlet)),
                    dtype=mrx.DTYPE)
    got = seq.apply_mass_matrix(x, k, dirichlet)
    want = _oracle_mass(seq, x, k, dirichlet)
    err = float(jnp.max(jnp.abs(got - want)) / jnp.max(jnp.abs(want)))
    assert err < IDENT, f"k={k}: kernel vs oracle off by {err:.2e}"


@pytest.mark.parametrize(("pair", "partner"), (((2, 1), (1, 2)), ((0, 3), (3, 0))))
def test_projection_pairs_are_transposes(seq, pair, partner):
    """``<P_12 x, y> = <x, P_21 y>`` on the extracted Dirichlet spaces."""
    k_in = {(2, 1): 2, (0, 3): 3}[pair]
    k_out = {(2, 1): 1, (0, 3): 0}[pair]
    rng = np.random.default_rng(7)
    x = jnp.asarray(rng.standard_normal(seq.n(k_in, True)), dtype=mrx.DTYPE)
    y = jnp.asarray(rng.standard_normal(seq.n(k_out, True)), dtype=mrx.DTYPE)
    lhs = float(y @ seq.apply_projection_matrix(x, *pair, True, dirichlet_out=True))
    rhs = float(x @ seq.apply_projection_matrix(y, *partner, True, dirichlet_out=True))
    assert abs(lhs - rhs) < IDENT * abs(lhs), f"{pair}: {lhs:.6e} vs {rhs:.6e}"
