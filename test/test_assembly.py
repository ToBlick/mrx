"""Tests for assembly methods."""

import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import numpy.testing as npt
import pytest

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
    n = 6
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
    return seq, p


def _bcoo_data(bcsr_matrix):
    """Convert a BCSR matrix to BCOO and return its stored values."""
    return jsparse.BCOO.from_bcsr(bcsr_matrix).data


class TestMassMatrixM0:
    """Mass matrix M0: positivity of all entries and symmetry."""

    def test_m0_symmetry(self, seq_and_p):
        seq, _ = seq_and_p
        M = seq.m0_sp.todense()
        npt.assert_allclose(M, M.T, atol=1e-14)

    def test_m0_positive_entries(self, seq_and_p):
        seq, _ = seq_and_p
        assert jnp.all(_bcoo_data(seq.m0_sp) > 0)


class TestMassMatrixM1:
    """Mass matrix M1: positivity of all entries and symmetry."""

    def test_m1_symmetry(self, seq_and_p):
        seq, _ = seq_and_p
        M = seq.m1_sp.todense()
        npt.assert_allclose(M, M.T, atol=1e-14)

    def test_m1_positive_entries(self, seq_and_p):
        seq, _ = seq_and_p
        assert jnp.all(_bcoo_data(seq.m1_sp) > 0)


class TestMassMatrixM2:
    """Mass matrix M2: positivity of all entries and symmetry."""

    def test_m2_symmetry(self, seq_and_p):
        seq, _ = seq_and_p
        M = seq.m2_sp.todense()
        npt.assert_allclose(M, M.T, atol=1e-14)

    def test_m2_positive_entries(self, seq_and_p):
        seq, _ = seq_and_p
        assert jnp.all(_bcoo_data(seq.m2_sp) > 0)


class TestMassMatrixM3:
    """Mass matrix M3: positivity of all entries and symmetry."""

    def test_m3_symmetry(self, seq_and_p):
        seq, _ = seq_and_p
        M = seq.m3_sp.todense()
        npt.assert_allclose(M, M.T, atol=1e-14)

    def test_m3_positive_entries(self, seq_and_p):
        seq, _ = seq_and_p
        assert jnp.all(_bcoo_data(seq.m3_sp) > 0)


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
