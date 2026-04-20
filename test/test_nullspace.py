"""Tests for nullspace computation on a rotating ellipse geometry.

Compares three approaches against a shared dense reference:
1. Old method: compute_nullspaces (Leray projection + Hodge decomposition)
2. New method: _compute_nullspaces (inverse iteration with shifted stiffness)
3. Dense reference: scipy.linalg.eigh on dense L_k v = λ M_k v
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
from scipy.linalg import eigh

from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import rotating_ellipse_map

jax.config.update("jax_enable_x64", True)

types = ("clamped", "periodic", "periodic")
# Betti numbers for a solid torus (no cavity): [1, 1, 0, 0]
BETTI = [1, 1, 0, 0]


def _build_dense(matvec, n):
    """Build a dense matrix from a matvec by probing with unit vectors."""
    cols = []
    for i in range(n):
        e = jnp.zeros(n).at[i].set(1.0)
        cols.append(matvec(e))
    return jnp.column_stack(cols)


@pytest.fixture(scope="module")
def seq():
    """Create and assemble a DeRhamSequence on the rotating ellipse."""
    F = rotating_ellipse_map(eps=0.33, kappa=1.2, R0=1.0, nfp=3)
    s = DeRhamSequence((4, 4, 4), (2, 2, 2), 4, types, F,
                       polar=True, tol=1e-12, maxiter=500)
    s.evaluate_1d()
    for k in range(4):
        s.assemble_mass_matrix(k)
    for k in range(3):
        s.assemble_derivative_matrix(k)
    for k in range(4):
        s.assemble_hodge_laplacian(k)
    return s


# --------------------------------------------------------------------------
# Precompute dense eigenpairs and both nullspace methods (once per module)
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dense_results(seq):
    """Dense generalized eigenvalue results for all (k, dirichlet) pairs."""
    results = {}
    cases = [(0, False), (0, True), (1, False), (1, True),
             (2, True), (2, False), (3, True), (3, False)]
    for k, dirichlet in cases:
        suffix = "_dbc" if dirichlet else ""
        n = getattr(seq, f"n{k}{suffix}")
        t0 = time.perf_counter()
        L = np.array(_build_dense(
            lambda v, k=k, d=dirichlet: seq.apply_hodge_laplacian(v, k, dirichlet=d), n))
        M = np.array(_build_dense(
            lambda v, k=k, d=dirichlet: seq.apply_mass_matrix(v, k, dirichlet=d), n))
        eigvals, eigvecs = eigh(L, M)
        dt = time.perf_counter() - t0
        print(f"  [Dense eigh] k={k}, dbc={dirichlet}, n={n}, time={dt:.2f}s, "
              f"smallest eigenvalues: {eigvals[:3]}")
        results[(k, dirichlet)] = (eigvals, eigvecs)
    return results


@pytest.fixture(scope="module")
def old_vecs(seq):
    """Nullspace vectors from the old (Leray) method."""
    t0 = time.perf_counter()
    seq.compute_nullspaces()
    dt = time.perf_counter() - t0
    print(f"  [Old method] compute_nullspaces time={dt:.2f}s")
    vecs = {}
    for k in range(4):
        vecs[(k, True)] = list(getattr(seq, f"null_{k}_dbc"))
        vecs[(k, False)] = list(getattr(seq, f"null_{k}"))
    return vecs


@pytest.fixture(scope="module")
def new_vecs(seq, old_vecs):
    """Nullspace vectors from the new (inverse iteration) method.
    Depends on old_vecs to ensure old method runs first."""
    t0 = time.perf_counter()
    info = seq._compute_nullspaces(BETTI, eps=1e-6)
    dt = time.perf_counter() - t0
    print(f"  [New method] _compute_nullspaces time={dt:.2f}s")
    for (k, dbc), entries in info.items():
        for i, (n_iters, rq) in enumerate(entries):
            print(f"    k={k}, dbc={dbc}, vec {i}: "
                  f"inverse iterations={n_iters}, final Rayleigh quotient={rq:.3e}")
    vecs = {}
    for k in range(4):
        vecs[(k, True)] = list(getattr(seq, f"null_{k}_dbc"))
        vecs[(k, False)] = list(getattr(seq, f"null_{k}"))
    return vecs


# --------------------------------------------------------------------------
# Dense reference: eigenvalues near zero
# --------------------------------------------------------------------------

class TestDenseEigenvalues:
    """Verify that the dense generalized eigensolver finds zero eigenvalues
    matching the expected Betti numbers."""

    @pytest.mark.parametrize("k,dirichlet,n_expected", [
        (0, False, 1),   # betti_0 = 1
        (0, True,  0),   # no harmonic 0-forms with DBC
        (1, False, 1),   # betti_1 = 1
        (1, True,  0),   # betti_2 = 0
        (2, True,  1),   # betti_1 = 1
        (2, False, 0),   # betti_2 = 0
        (3, True,  1),   # betti_0 = 1
        (3, False, 0),   # no harmonic 3-forms without DBC
    ], ids=["k0_no_dbc", "k0_dbc", "k1_no_dbc", "k1_dbc",
            "k2_dbc", "k2_no_dbc", "k3_dbc", "k3_no_dbc"])
    def test_zero_eigenvalue_count(self, dense_results, k, dirichlet, n_expected):
        """Number of near-zero eigenvalues matches predicted Betti number."""
        eigvals, _ = dense_results[(k, dirichlet)]
        n_zero = int(np.sum(np.abs(eigvals) < 1e-6))
        assert n_zero == n_expected, (
            f"k={k}, dbc={dirichlet}: expected {n_expected} zero eigenvalues, "
            f"got {n_zero}. Smallest eigenvalues: {eigvals[:5]}"
        )


# --------------------------------------------------------------------------
# Old and new methods vs dense reference
# --------------------------------------------------------------------------

def _check_vs_dense(seq, method_vecs, dense_results, k, dirichlet, n_expected):
    """Check a set of nullspace vectors against the dense reference."""
    eigvals, eigvecs = dense_results[(k, dirichlet)]
    method_vs = method_vecs[(k, dirichlet)]
    assert len(method_vs) == n_expected

    # Dense reference vectors (M-normalized)
    dense_vs = []
    for i in range(n_expected):
        v = jnp.array(eigvecs[:, i])
        v = v / seq.l2_norm(v, k, dirichlet=dirichlet)
        dense_vs.append(v)

    for idx, v in enumerate(method_vs):
        # Rayleigh quotient should be near zero
        Lv = seq.apply_hodge_laplacian(v, k, dirichlet=dirichlet)
        rq = jnp.abs(v @ Lv)
        print(
            f"    k={k}, dbc={dirichlet}, vec {idx}: Rayleigh quotient = {float(rq):.3e}")
        assert rq < 1e-6, f"Rayleigh quotient = {rq}"

        # M-norm of difference should be small (account for sign flip)
        # ||v - v_dense||_M^2 = 2 - 2*<v, M*v_dense> for M-normalized vectors
        for j, v_dense in enumerate(dense_vs):
            Mv_dense = seq.apply_mass_matrix(v_dense, k, dirichlet=dirichlet)
            overlap = v @ Mv_dense
            m_norm_diff = jnp.sqrt(jnp.abs(2.0 - 2.0 * jnp.abs(overlap)))
            print(f"    k={k}, dbc={dirichlet}, vec {idx} vs dense {j}: "
                  f"M-overlap = {float(overlap):.6f}, ||diff||_M = {float(m_norm_diff):.3e}")
            assert m_norm_diff < 1e-6, (
                f"M-norm of difference = {m_norm_diff}, overlap = {overlap}")


class TestOldMethodVsDense:
    """Compare compute_nullspaces (Leray projection) against dense eigensolver."""

    @pytest.mark.parametrize("k,dirichlet,n_expected", [
        (1, False, 1),
        (2, True, 1),
    ], ids=["k1_no_dbc", "k2_dbc"])
    def test_old_vs_dense(self, seq, old_vecs, dense_results, k, dirichlet, n_expected):
        _check_vs_dense(seq, old_vecs, dense_results, k, dirichlet, n_expected)


class TestNewMethodVsDense:
    """Compare _compute_nullspaces (inverse iteration) against dense eigensolver."""

    @pytest.mark.parametrize("k,dirichlet,n_expected", [
        (1, False, 1),
        (2, True, 1),
    ], ids=["k1_no_dbc", "k2_dbc"])
    def test_new_vs_dense(self, seq, new_vecs, dense_results, k, dirichlet, n_expected):
        _check_vs_dense(seq, new_vecs, dense_results, k, dirichlet, n_expected)


# --------------------------------------------------------------------------
# Correct nullspace dimensions for all k
# --------------------------------------------------------------------------

class TestNullspaceDimensions:
    """Both methods produce the correct number of nullspace vectors."""

    @pytest.mark.parametrize("k,dirichlet,expected_len", [
        (0, False, 1),
        (0, True, 0),
        (1, False, 1),
        (1, True, 0),
        (2, False, 0),
        (2, True, 1),
        (3, False, 0),
        (3, True, 1),
    ], ids=["k0_no_dbc", "k0_dbc", "k1_no_dbc", "k1_dbc",
            "k2_no_dbc", "k2_dbc", "k3_no_dbc", "k3_dbc"])
    def test_dimensions(self, old_vecs, new_vecs, k, dirichlet, expected_len):
        assert len(old_vecs[(k, dirichlet)]) == expected_len
        assert len(new_vecs[(k, dirichlet)]) == expected_len
        assert len(new_vecs[(k, dirichlet)]) == expected_len
