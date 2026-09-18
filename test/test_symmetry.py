"""Stellarator symmetry: the reflection on the spline bases and the parity
projector of the half-period quadrature (``mrx.symmetry``).

Milliseconds: the 1-D identities need no sequence; the projector tests run
on the session ``seq`` (li383, a half-period sequence since the map is
stellarator symmetric) and its own field ``b0``.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mrx.spline_bases import DerivativeSpline, SplineBasis
from mrx.symmetry import reflect, reflection_permutation, reflection_plan, symmetrize


@pytest.mark.parametrize("n,p", [(8, 2), (9, 3), (12, 1)])
def test_reflection_permutes_the_periodic_bases(n, p):
    """``B_j(-x) = B_{perm(j)}(x)`` on the B-spline basis and on its
    derivative basis, the latter with its own ``(n, p - 1)``."""
    x = jax.random.uniform(jax.random.PRNGKey(3), (40,))
    for basis in (SplineBasis(n, p, "periodic"), DerivativeSpline(SplineBasis(n, p, "periodic"))):
        perm = reflection_permutation(basis.n, basis.p)
        left = jax.vmap(jax.vmap(basis, (0, None)), (None, 0))(-x % 1.0, basis.ns)       # (n, 40)
        right = jax.vmap(jax.vmap(basis, (0, None)), (None, 0))(x, jnp.asarray(perm))
        assert float(jnp.max(jnp.abs(left - right))) < 1e-12


def test_reflection_is_an_involution_and_the_projectors_split(seq):
    for k in range(4):
        plan = reflection_plan(seq, k)
        n = sum(int(np.prod(shape)) for *_, shape in plan)
        x = jax.random.normal(jax.random.PRNGKey(k), (n,), dtype=seq.dtype)
        assert float(jnp.max(jnp.abs(reflect(reflect(x, plan), plan) - x))) < 1e-6
        even, odd = symmetrize(x, plan, 1), symmetrize(x, plan, -1)
        assert float(jnp.max(jnp.abs(even + odd - x))) < 1e-6
        assert float(jnp.max(jnp.abs(symmetrize(even, plan, 1) - even))) < 1e-6
        assert float(jnp.max(jnp.abs(symmetrize(odd, plan, 1)))) < 1e-6


def test_the_equilibrium_field_is_odd(seq, b0):
    """The li383 field ``B = dA'`` (a 2-form) and the harmonic forms are odd
    under the reflection: the projector onto the odd fields leaves them
    alone, the even projector annihilates them. This is the check of the
    permutation and the component signs together."""
    for k, x in ((2, seq.E(2, True).T @ b0), (2, seq.E(2, True).T @ seq.nullspace(2, True)[0]),
                 (1, seq.E(1, False).T @ seq.nullspace(1, False)[0])):
        plan = reflection_plan(seq, k)
        scale = float(jnp.max(jnp.abs(x)))
        assert float(jnp.max(jnp.abs(symmetrize(x, plan, -1) - x))) < 1e-5 * scale
        assert float(jnp.max(jnp.abs(symmetrize(x, plan, 1)))) < 1e-5 * scale


def test_half_period_mass_apply_matches_the_full_period(seq, b0):
    """``M_2 B`` on the half-period sequence (its quadrature covers zeta in
    [0, 1/2], the projector combines the mirror images) equals the same apply
    on a full-period sequence of the same mesh and map."""
    from mrx.derham_sequence import DeRhamSequence
    from test.conftest import BETTI, NS, P, TYPES

    assert seq.half_period
    full = DeRhamSequence(NS, (P, P, P), P + 1, TYPES, polar=True, betti_numbers=BETTI,
                          half_period=False)
    full.set_map(seq.map)
    assert not full.half_period
    for k, x, parity in ((2, b0, -1), (0, jnp.ones(seq.n(0, True), dtype=seq.dtype), 1)):
        y_half = seq.apply_mass_matrix(x, k, dirichlet=True)
        y_full = full.apply_mass_matrix(x, k, dirichlet=True)
        tol = 1e3 * float(jnp.finfo(seq.dtype).eps)
        assert float(jnp.max(jnp.abs(y_half - y_full))) < tol * float(jnp.max(jnp.abs(y_full)))
