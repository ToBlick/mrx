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
from mrx.symmetry import parity_extraction, raw_reflection, reflect, reflection_permutation, reflection_plan, symmetrize


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


@pytest.mark.parametrize("k", range(4))
def test_parity_extraction_partitions_the_free_space(seq, k):
    """``E_red = X^T E`` per parity: orthonormal rows, every reduced DoF a raw
    field of that parity (``E_red R = s E_red``, asserted by the builder), and
    the two parities together span the free space, ``n_+ + n_- = n_free``,
    with ``n_-`` about half of it. Checked on the Dirichlet and the free space."""
    perm, sign = raw_reflection(seq.reflection_plan[k])
    for dirichlet in (True, False):
        n_free = seq.n(k, dirichlet)
        sizes = {}
        for s in (1, -1):
            e = parity_extraction(seq, k, dirichlet, s)
            sizes[s] = int(e.forward_shape[0])
            # a random reduced vector expands to a raw field of parity s
            c = jnp.asarray(np.random.default_rng(k).standard_normal(sizes[s]), dtype=seq.dtype)
            raw = e.T @ c
            assert float(jnp.max(jnp.abs(sign * raw[perm] - s * raw))) < 1e-5 * float(jnp.max(jnp.abs(raw)))
        assert sizes[1] + sizes[-1] == n_free, (k, dirichlet, sizes, n_free)
        assert 0.4 * n_free < sizes[-1] < 0.6 * n_free, (k, dirichlet, sizes, n_free)


def test_parity_views_agree_with_the_projected_applies(seq, b0):
    """``seq.odd`` / ``seq.even`` (half the DoFs) apply the mass, the projections
    and the strong derivatives exactly as the unreduced half-period sequence does
    on a field of that parity: ``X^T (M v) = M_red (X^T v)`` for ``v = X X^T v``,
    with ``X`` the view's expansion (verified 2026-09-20 to 2e-7 on every ``(k,
    dirichlet)`` of a torus)."""
    for s, x in ((-1, b0), (1, seq.project_parity(jnp.ones(seq.n(2, True), dtype=seq.dtype), 2, 1))):
        view = seq.parity_view(s)
        X = view.reduction[(2, True)]
        c = X.T @ x
        tol = 1e3 * float(jnp.finfo(seq.dtype).eps)
        for got, want in ((view.apply_mass_matrix(c, 2, True), X.T @ seq.apply_mass_matrix(x, 2, True)),
                          (view.apply_incidence_matrix(c, 2), view.reduction[(3, True)].T @ seq.apply_incidence_matrix(x, 2)),
                          (view.apply_projection_matrix(c, 2, 1, True, True),
                           view.reduction[(1, True)].T @ seq.apply_projection_matrix(x, 2, 1, True, True))):
            assert float(jnp.max(jnp.abs(got - want))) < tol * float(jnp.max(jnp.abs(want)))
        assert 0.4 * seq.n(2, True) < view.n(2, True) < 0.6 * seq.n(2, True)


def test_parity_view_solves_agree_with_the_base(seq, b0):
    """The odd view's solves (mass, Laplacian, Leray) on a field of its parity
    reproduce the unreduced sequence's to the solve tolerance, with the same
    iteration counts: the reduced atoms ``X^T P X`` precondition as the
    projected ones did (a torus (6,8,8) p=2, 2026-09-20: k=2 dbc 61 vs 60
    iterations, k=1 dbc 40 vs 40, k=0 free 32 vs 32; agreement 1e-7..2e-6)."""
    view = seq.odd
    view.build_preconditioners()
    view.compute_nullspaces(verbose=False)
    X = view.reduction[(2, True)]
    c = X.T @ b0
    tol = 1e2 * seq.tol + 1e2 * float(jnp.finfo(seq.dtype).eps)
    rhs_base, rhs_view = seq.apply_laplacian(b0, 2), view.apply_laplacian(c, 2)
    x_base, it_base = seq.apply_inverse_laplacian(rhs_base, 2, return_info=True)
    x_view, it_view = view.apply_inverse_laplacian(rhs_view, 2, return_info=True)
    assert int(it_view) < 0 and abs(int(it_view)) <= 1.5 * abs(int(it_base)), (int(it_view), int(it_base))
    assert float(jnp.max(jnp.abs(x_view - X.T @ x_base))) < tol * float(jnp.max(jnp.abs(x_base)))
    Pb, _ = seq.apply_leray_projection(b0, k=2)
    Pc, _ = view.apply_leray_projection(c, k=2)
    assert float(jnp.max(jnp.abs(Pc - X.T @ Pb))) < tol * float(jnp.max(jnp.abs(Pb)))
    h, hb = view.nullspace(2, True)[0], seq.nullspace(2, True)[0]
    assert abs(abs(float((X @ h) @ seq.apply_mass_matrix(hb, 2, True))) - 1.0) < tol
