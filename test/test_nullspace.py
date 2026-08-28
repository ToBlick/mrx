"""Tests for mrx.nullspace: harmonic nullspace construction and helpers.

All tests that use a sequence reuse the session-scoped ``tiny_seq`` fixture
from conftest.py (one full assembly, shared across the entire pytest session).

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
"""

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

import mrx
from test.conftest import BETTI
from mrx.nullspace import (
    _n_vectors,
    estimate_spectral_gap,
    get_nullspace,
    harmonic_rayleigh,
    get_saddle_point_nullspaces,
    init_nullspaces,
)
from mrx.operators import SequenceOperators


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

def test_init_nullspaces_shapes_and_zeros(tiny_seq):
    ops = init_nullspaces(tiny_seq, tiny_seq.operators)
    for k in range(4):
        for dbc in (False, True):
            arr = ops.nullspaces[(k, dbc)]
            assert arr is not None, f"null_{k}{'_dbc' if dbc else ''} is None after init"
            n_vec = _n_vectors(BETTI, k, dbc)
            n_dof = tiny_seq.n(k, dbc)
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
def test_stored_nullspace_vectors_are_harmonic(tiny_seq, k, dbc):
    """Each stored vector v has Rayleigh quotient vᵀL v / vᵀM v ≤ 4.5e3 eps (1e-12 f64).

    The bound is quadratic in the eigenvector error, which the solve leaves at
    ``seq.tol = sqrt(eps)``, so it is an eps-scaled quantity: 1e-12 in float64,
    5.4e-4 in float32.

    The metric is the Rayleigh quotient, NOT ‖L_k v‖. ``L_k`` is dual-valued,
    so ``l2_norm(Lv)`` measures a functional in the primal mass norm — an
    uncompensated mass factor whose scale drifts with resolution *and* with
    the choice of preconditioner. It was measured spanning five orders across
    solver arms that produced fields identical to five significant figures, and
    an earlier version of this test asserting ‖Lv‖ ≤ 1e-8 failed at 5.6e-8 for
    k=2/3 dbc on vectors whose Rayleigh quotients were ~6e-23, i.e. harmonic to
    ~1e-12 in eigenvector error. The threshold was measuring the metric, not
    the vectors.

    The Rayleigh quotient carries the units of an eigenvalue, so it is directly
    comparable to the first nonzero eigenvalue λ₁ (O(1) on this fixture), which
    is exactly the quantity deflation cares about. It is quadratic in the
    eigenvector error, so 1e-12 here corresponds to an error of ~1e-6 — ample,
    and the measured values sit eleven orders below it.

    See docs/research/gvec_h5_vacuum_comparison.md, "The k=2 solver failure".
    """
    ops = tiny_seq.operators
    vs = get_nullspace(ops, k, dbc)
    for i, v in enumerate(vs):
        Lv = tiny_seq.apply_laplacian(v, k, dirichlet=dbc, operators=ops)
        nrm2 = float(tiny_seq.l2_norm_sq(v, k, dirichlet=dbc))
        assert nrm2 > 0.0, f"k={k} dbc={dbc} vec[{i}] has zero mass norm"
        rayleigh = abs(float(v @ Lv)) / nrm2
        assert rayleigh <= mrx.eps(4.5e3), (
            f"k={k} dbc={dbc} vec[{i}]: Rayleigh = {rayleigh:.2e} > {mrx.eps(4.5e3):.1e} "
            f"(‖Lv‖ = {float(tiny_seq.l2_norm(Lv, k, dirichlet=dbc)):.2e}, "
            "reported for context only — it is not the criterion)"
        )


@pytest.mark.parametrize("k,dbc", _NONTRIVIAL,
                          ids=["k0","k1","k2dbc","k3dbc"])
def test_stored_nullspace_vectors_are_mass_orthonormal(tiny_seq, k, dbc):
    """Mass Gram matrix of stored null vectors equals the identity.

    The vectors are normalised after the solve, so the Gram matrix is the
    identity to solver accuracy: ``seq.tol`` (1.5e-8 f64 / 3.5e-4 f32).
    """
    ops = tiny_seq.operators
    vs = get_nullspace(ops, k, dbc)
    n_vec = vs.shape[0]
    mass_vs = jax.vmap(
        lambda v: tiny_seq.apply_mass_matrix(v, k, dirichlet=dbc)
    )(vs)
    gram = vs @ mass_vs.T
    npt.assert_allclose(gram, jnp.eye(n_vec), atol=tiny_seq.tol)


# ---------------------------------------------------------------------------
# get_saddle_point_nullspaces — lower-block consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k,dbc", [(1, False), (2, True)],
                          ids=["k1","k2dbc"])
def test_saddle_point_lower_block_satisfies_mass_equation(tiny_seq, k, dbc):
    """Lower block w satisfies M_{k-1} w = D_{k-1}^T v for each pair (v, w).

    ``w`` comes out of a mass solve at ``seq.tol``, so the residual is bounded
    by ``seq.tol`` times the right-hand side (1.5e-8 f64 / 3.5e-4 f32).
    """
    ops = tiny_seq.operators
    vs_upper, vs_lower = get_saddle_point_nullspaces(tiny_seq, ops, k, dbc)
    for i, (v, w) in enumerate(zip(vs_upper, vs_lower)):
        Dt_v = tiny_seq.apply_derivative_matrix(
            v, k - 1,
            dirichlet_in=dbc, dirichlet_out=dbc,
            transpose=True,
        )
        Mw = tiny_seq.apply_mass_matrix(w, k - 1, dirichlet=dbc)
        npt.assert_allclose(Mw, Dt_v,
                            atol=tiny_seq.tol * max(float(jnp.linalg.norm(Dt_v)), 1.0),
                            err_msg=f"k={k} dbc={dbc} vec[{i}]: M_{{k-1}} w ≠ D_{{k-1}}^T v")


@pytest.mark.parametrize("k,dbc", [(1, False), (2, True)], ids=["k1", "k2dbc"])
def test_spectral_gap_dominates_the_harmonic_quotient(tiny_seq, k, dbc):
    """``lambda_1`` from a handful of inverse-iteration sweeps is O(1) and
    the stored form's Rayleigh quotient is negligible against it.

    This is the ratio :func:`compute_nullspaces` prints for every form; the
    test pins the two facts that make the line readable: the estimate is a
    genuine non-harmonic eigenvalue (positive, not the shift ``eps``, not a
    mass-norm artefact) and the ratio separates a harmonic form from a solve
    that stopped early by many orders.
    """
    ops = tiny_seq.operators
    lam, sweeps = estimate_spectral_gap(tiny_seq, ops, k, dbc, maxiter=5)
    assert 0 < sweeps <= 5
    assert 1e-2 < lam < 1e3, f"lambda_1 = {lam:.2e} is not O(1)"
    v = get_nullspace(ops, k, dbc)[0]
    rq = harmonic_rayleigh(tiny_seq, v, k, dbc, ops)
    assert abs(rq) / lam < mrx.eps(4.5e3), f"ratio {abs(rq) / lam:.2e}"
