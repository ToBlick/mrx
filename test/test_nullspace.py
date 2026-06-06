"""Tests for mrx.nullspace: harmonic nullspace construction and helpers.

All tests that use a sequence reuse the session-scoped ``torus_seq`` fixture
from conftest.py (one full assembly, shared across the entire pytest session).
The stiffness-nullspace bases are pre-computed once in the fixture and stored
on ``torus_seq.stiffness_null[(k, dbc)]``.

Mathematical properties checked
--------------------------------
* ``_n_vectors`` returns the correct harmonic dimensions for solid-torus
  topology ``(b0, b1, b2, b3) = (1, 1, 0, 0)``.
* ``init_nullspaces`` sets every null field to the correct zero-array shape.
* ``get_nullspace`` raises ``ValueError`` when the field has never been set.
* Every stored nullspace vector satisfies ``‖L_k v‖ ≤ seq.tol``.
* The stored null vectors are M-orthonormal (Gram matrix = I).
* The saddle-point lower block satisfies ``M_{k-1} w = D_{k-1}^T v`` for
  each upper/lower pair.
* Every vector returned by ``get_stiffness_nullspace`` lies in ``ker(K_k)``.
* The stiffness nullspace basis is M-orthonormal (Gram matrix = I).
"""

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from test.conftest import BETTI
from mrx.nullspace import (
    _n_vectors,
    _null_field,
    get_nullspace,
    get_saddle_point_nullspaces,
    init_nullspaces,
)
from mrx.operators import SequenceOperators

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# (k, dirichlet) pairs with non-trivial harmonic dimension on the solid torus
# (betti = (1, 1, 0, 0)).
# ---------------------------------------------------------------------------
_NONTRIVIAL = [
    (0, False),   # b0 = 1
    (1, False),   # b1 = 1
    (2, True),    # Dirichlet dual of b1: 1
    (3, True),    # Dirichlet dual of b0: 1
]


# ---------------------------------------------------------------------------
# _n_vectors — pure Python, no sequence needed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k,dbc,expected", [
    (0, False, 1), (1, False, 1), (2, False, 0), (3, False, 0),
    (0, True,  0), (1, True,  0), (2, True,  1), (3, True,  1),
], ids=["k0","k1","k2","k3","k0dbc","k1dbc","k2dbc","k3dbc"])
def test_n_vectors_torus(k, dbc, expected):
    assert _n_vectors((1, 1, 0, 0), k, dbc) == expected


# ---------------------------------------------------------------------------
# init_nullspaces — shapes and zero initialisation
# ---------------------------------------------------------------------------

def test_init_nullspaces_shapes_and_zeros(torus_seq):
    ops = init_nullspaces(torus_seq, torus_seq.operators)
    for k in range(4):
        for dbc in (False, True):
            arr = getattr(ops, _null_field(k, dbc))
            assert arr is not None, f"null_{k}{'_dbc' if dbc else ''} is None after init"
            n_vec = _n_vectors(BETTI, k, dbc)
            n_dof = getattr(torus_seq, f"n{k}_dbc" if dbc else f"n{k}")
            assert arr.shape == (n_vec, n_dof), (
                f"k={k} dbc={dbc}: expected ({n_vec}, {n_dof}), got {arr.shape}"
            )
            npt.assert_array_equal(arr, 0.0)


# ---------------------------------------------------------------------------
# get_nullspace — raises when uninitialised
# ---------------------------------------------------------------------------

def test_get_nullspace_raises_when_uninitialised():
    ops = SequenceOperators()
    with pytest.raises(ValueError, match="not initialised"):
        get_nullspace(ops, 0, False)


# ---------------------------------------------------------------------------
# Quality of the stored nullspaces from the session torus
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k,dbc", _NONTRIVIAL,
                          ids=["k0","k1","k2dbc","k3dbc"])
def test_stored_nullspace_vectors_are_harmonic(torus_seq, k, dbc):
    """Each stored vector v satisfies ‖L_k v‖ ≤ 10 * seq.tol.

    The factor of 10 accounts for the stall-based stopping criterion in
    find_nullspace_vectors: iteration terminates when |res - res_prev| ≤ tol,
    which can leave the final residual marginally above tol itself.
    """
    ops = torus_seq.operators
    vs = get_nullspace(ops, k, dbc)
    atol = 10 * torus_seq.tol
    for i, v in enumerate(vs):
        Lv = torus_seq.apply_hodge_laplacian(v, k, dirichlet=dbc, operators=ops)
        res = float(torus_seq.l2_norm(Lv, k, dirichlet=dbc))
        assert res <= atol, (
            f"k={k} dbc={dbc} vec[{i}]: ‖Lv‖ = {res:.2e} > 10·tol {atol:.2e}"
        )


@pytest.mark.parametrize("k,dbc", _NONTRIVIAL,
                          ids=["k0","k1","k2dbc","k3dbc"])
def test_stored_nullspace_vectors_are_mass_orthonormal(torus_seq, k, dbc):
    """Mass Gram matrix of stored null vectors equals the identity."""
    ops = torus_seq.operators
    vs = get_nullspace(ops, k, dbc)
    n_vec = vs.shape[0]
    mass_vs = jax.vmap(
        lambda v: torus_seq.apply_mass_matrix(v, k, dirichlet=dbc, operators=ops)
    )(vs)
    gram = vs @ mass_vs.T
    npt.assert_allclose(gram, jnp.eye(n_vec), atol=1e-8)


# ---------------------------------------------------------------------------
# get_saddle_point_nullspaces — lower-block consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k,dbc", [(1, False), (2, True)],
                          ids=["k1","k2dbc"])
def test_saddle_point_lower_block_satisfies_mass_equation(torus_seq, k, dbc):
    """Lower block w satisfies M_{k-1} w = D_{k-1}^T v for each pair (v, w)."""
    ops = torus_seq.operators
    vs_upper, vs_lower = get_saddle_point_nullspaces(torus_seq, ops, k, dbc)
    for i, (v, w) in enumerate(zip(vs_upper, vs_lower)):
        Dt_v = torus_seq.apply_derivative_matrix(
            v, k - 1,
            dirichlet_in=dbc, dirichlet_out=dbc,
            transpose=True,
            operators=ops,
        )
        Mw = torus_seq.apply_mass_matrix(w, k - 1, dirichlet=dbc, operators=ops)
        npt.assert_allclose(Mw, Dt_v, atol=1e-8,
                            err_msg=f"k={k} dbc={dbc} vec[{i}]: M_{{k-1}} w ≠ D_{{k-1}}^T v")


# ---------------------------------------------------------------------------
# get_stiffness_nullspace — kernel membership and M-orthonormality
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k,dbc", [(1, False), (2, True)],
                          ids=["k1","k2dbc"])
def test_stiffness_nullspace_is_in_kernel(torus_seq, k, dbc):
    """Every stiffness-nullspace vector satisfies ‖K_k v‖ ≈ 0."""
    ops = torus_seq.operators
    basis = torus_seq.stiffness_null[(k, dbc)]
    for i, v in enumerate(basis):
        Kv = torus_seq.apply_stiffness(v, k, dirichlet=dbc, operators=ops)
        res = float(jnp.linalg.norm(Kv))
        assert res <= 1e-6, (
            f"k={k} dbc={dbc} vec[{i}]: ‖K_k v‖ = {res:.2e}"
        )


@pytest.mark.parametrize("k,dbc", [(1, False), (2, True)],
                          ids=["k1","k2dbc"])
def test_stiffness_nullspace_is_mass_orthonormal(torus_seq, k, dbc):
    """Mass Gram matrix of stiffness-nullspace vectors equals the identity."""
    ops = torus_seq.operators
    basis = torus_seq.stiffness_null[(k, dbc)]
    if basis.shape[0] == 0:
        return
    n_vec = basis.shape[0]
    mass_basis = jax.vmap(
        lambda v: torus_seq.apply_mass_matrix(v, k, dirichlet=dbc, operators=ops)
    )(basis)
    gram = basis @ mass_basis.T
    npt.assert_allclose(gram, jnp.eye(n_vec), atol=1e-8)
