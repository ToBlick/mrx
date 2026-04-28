"""Tests for assembly methods."""

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from mrx.assembly import (assemble_dense_hodge_laplacian,
                          assemble_dense_mass_matrix)
from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import rotating_ellipse_map
from mrx.utils import diag_EAET

jax.config.update("jax_enable_x64", True)

types = ("clamped", "periodic", "periodic")


@pytest.fixture(
    scope="module",
    params=[1, 2, 3],
    ids=[f"p{p}" for p in [1, 2, 3]],
)
def seq_and_p(request):

    """DeRham sequence on a rotating ellipse, parametrised over degree.

    Performs all prerequisite assemblies once so individual tests don't repeat them.
    """
    p = request.param
    n = 4
    ns = (n, n, n)
    ps = (p, p, p)
    F = rotating_ellipse_map(nfp=3)
    seq = DeRhamSequence(ns, ps, 2 * p, types, F, polar=True,
                         tol=1e-12, maxiter=1000)
    seq.evaluate_1d()
    for k in range(4):
        seq.assemble_mass_matrix(k)
    for k in range(3):
        seq.assemble_derivative_matrix(k)
    for k in range(4):
        seq.assemble_hodge_laplacian(k)
    return seq, p


@pytest.fixture(scope="module")
def dense_matrices(seq_and_p):
    """Pre-assembled dense mass matrices and Hodge Laplacians, keyed by (k, dirichlet)."""
    seq, _ = seq_and_p
    masses = {k: assemble_dense_mass_matrix(seq, k, dirichlet=False) for k in range(4)}
    laplacians = {
        (k, d): assemble_dense_hodge_laplacian(seq, k, dirichlet=d)
        for k in range(4)
        for d in (False, True)
    }
    return masses, laplacians


class TestMassMatrixM0:
    """Mass matrix M0: no negative entries and symmetry."""

    def test_m0_symmetry(self, seq_and_p):
        seq, _ = seq_and_p
        M = seq.m0.todense()
        npt.assert_allclose(M, M.T, atol=1e-14)

    def test_m0_no_negative_entries(self, seq_and_p):
        seq, _ = seq_and_p
        assert jnp.all(seq.m0.to_bcoo().data > -1e-10)


class TestMassMatrixM1:
    """Mass matrix M1: symmetry."""

    def test_m1_symmetry(self, seq_and_p):
        seq, _ = seq_and_p
        M = seq.m1.todense()
        npt.assert_allclose(M, M.T, atol=1e-14)


class TestMassMatrixM2:
    """Mass matrix M2: symmetry."""

    def test_m2_symmetry(self, seq_and_p):
        seq, _ = seq_and_p
        M = seq.m2.todense()
        npt.assert_allclose(M, M.T, atol=1e-14)


class TestMassMatrixM3:
    """Mass matrix M3: no negative entries and symmetry."""

    def test_m3_symmetry(self, seq_and_p):
        seq, _ = seq_and_p
        M = seq.m3.todense()
        npt.assert_allclose(M, M.T, atol=1e-14)

    def test_m3_no_negative_entries(self, seq_and_p):
        seq, _ = seq_and_p
        assert jnp.all(seq.m3.to_bcoo().data > -1e-10)


class TestDiagEAET:
    """Test diag_EAET against dense E @ M @ E^T diagonal."""

    @pytest.mark.parametrize("k", [0, 1, 2, 3])
    def test_diag_EAET_vs_dense(self, seq_and_p, k):
        seq, _ = seq_and_p
        E = getattr(seq, f"e{k}")
        M = getattr(seq, f"m{k}_sp")

        diag_fast = diag_EAET(E, M)

        E_dense = E.todense()
        M_dense = M.todense()
        diag_ref = jnp.diag(E_dense @ M_dense @ E_dense.T)

        npt.assert_allclose(diag_fast, diag_ref, atol=1e-12)


class TestMassPreconditioner:
    """Mass matrix Jacobi preconditioner: positivity and spectral properties."""

    @pytest.mark.parametrize("k", [0, 1, 2, 3])
    def test_mass_precond_positive(self, seq_and_p, k):
        seq, _ = seq_and_p
        diaginv = getattr(seq, f"m{k}_sp_diaginv")
        assert jnp.all(diaginv > 0)

    @pytest.mark.parametrize("k", [0, 1, 2, 3])
    def test_mass_precond_dbc_positive(self, seq_and_p, k):
        seq, _ = seq_and_p
        diaginv = getattr(seq, f"m{k}_sp_diaginv_dbc")
        assert jnp.all(diaginv > 0)

    @pytest.mark.parametrize("k", [0, 1, 2, 3])
    def test_preconditioned_mass_eigs_positive(self, seq_and_p, dense_matrices, k):
        seq, _ = seq_and_p
        masses, _ = dense_matrices
        M = masses[k]
        diaginv = getattr(seq, f"m{k}_sp_diaginv")
        sqrtinv = jnp.sqrt(diaginv)
        sPMs = jnp.diag(sqrtinv) @ M @ jnp.diag(sqrtinv)
        eigs = jnp.linalg.eigvalsh(sPMs)
        assert jnp.all(eigs > -1e-12)


class TestHodgeLaplacePreconditioner:
    """Hodge-Laplace Jacobi preconditioner: positivity and spectral properties."""

    @pytest.mark.parametrize("k", [0, 1, 2, 3])
    def test_hodge_precond_positive(self, seq_and_p, k):
        seq, _ = seq_and_p
        diaginv = getattr(seq, f"dd{k}_sp_diaginv")
        assert jnp.all(diaginv > 0)

    @pytest.mark.parametrize("k", [0, 1, 2, 3])
    def test_hodge_precond_dbc_positive(self, seq_and_p, k):
        seq, _ = seq_and_p
        diaginv = getattr(seq, f"dd{k}_sp_diaginv_dbc")
        assert jnp.all(diaginv > 0)

    @pytest.mark.parametrize("k", [0, 1, 2, 3])
    @pytest.mark.parametrize("dirichlet", [False, True], ids=["no_dbc", "dbc"])
    def test_preconditioned_laplace_eigs_positive(self, seq_and_p, dense_matrices, k, dirichlet):
        """Eigenvalues of preconditioned Hodge-Laplace should be non-negative.

        Nullspace dimension depends on BCs:
          no DBC: k=0 (constants) and k=1 (harmonic) each have 1 zero EV
          DBC:    k=2 and k=3 each have 1 zero EV
        """
        seq, _ = seq_and_p
        suffix = "_dbc" if dirichlet else ""
        _, laplacians = dense_matrices
        L = laplacians[k, dirichlet]
        diaginv = getattr(seq, f"dd{k}_sp_diaginv{suffix}")
        sqrtinv = jnp.sqrt(diaginv)
        sPLs = jnp.diag(sqrtinv) @ L @ jnp.diag(sqrtinv)
        eigs = jnp.linalg.eigvalsh(sPLs)
        # All eigenvalues should be non-negative (allow small numerical noise)
        assert jnp.all(eigs > -1e-10)
        # Check nullspace dimension
        if dirichlet:
            has_null = k in (2, 3)
        else:
            has_null = k in (0, 1)
        if has_null:
            n_zero = jnp.sum(eigs < 1e-10)
            assert n_zero == 1, f"Expected 1 zero EV for k={k}, got {n_zero}"
