"""
Tests for the LazyBoundaryOperator, including dense and sparse assembly.
"""

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from mrx.extraction_operators import BoundaryOperator
from mrx.differential_forms import DifferentialForm

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NS = (5, 5, 5)
PS = (3, 3, 3)
TYPES = ("clamped", "periodic", "periodic")


@pytest.fixture(params=[0, 1, 2, 3])
def form(request):
    """DifferentialForm for each degree k."""
    return DifferentialForm(request.param, NS, PS, TYPES)


BC_CONFIGS = [
    ("dirichlet", "periodic", "periodic"),
    ("none", "none", "none"),
    ("left", "periodic", "periodic"),
    ("right", "periodic", "periodic"),
    ("dirichlet", "dirichlet", "dirichlet"),
]

# ---------------------------------------------------------------------------
# Dense assembly tests
# ---------------------------------------------------------------------------


class TestDenseAssembly:
    """Test dense matrix assembly via assemble() / matrix()."""

    def test_shape(self, form):
        B = BoundaryOperator(form, ("dirichlet", "periodic", "periodic"))
        M = B.matrix()
        assert M.shape == (B.n, form.n)

    def test_binary_entries(self, form):
        """All entries should be 0 or 1."""
        B = BoundaryOperator(form, ("dirichlet", "periodic", "periodic"))
        M = B.matrix()
        assert jnp.all((M == 0) | (M == 1))

    def test_at_most_one_per_row(self, form):
        """Each row should have at most one non-zero entry."""
        B = BoundaryOperator(form, ("dirichlet", "periodic", "periodic"))
        M = B.matrix()
        row_sums = jnp.sum(M, axis=1)
        assert jnp.all(row_sums <= 1 + 1e-10)

    @pytest.mark.parametrize("bc_types", BC_CONFIGS)
    def test_various_bc_types(self, bc_types):
        """Assembly should succeed for all supported BC configurations."""
        for k in [0, 1, 2, 3]:
            Λ = DifferentialForm(k, NS, PS, TYPES)
            B = BoundaryOperator(Λ, bc_types)
            M = B.matrix()
            assert M.shape == (B.n, Λ.n)


# ---------------------------------------------------------------------------
# Sparse assembly tests
# ---------------------------------------------------------------------------

class TestSparseAssembly:
    """Test sparse BCOO assembly via assemble_sparse() / sparse_matrix()."""

    def test_sparse_matches_dense(self, form):
        """The sparse matrix should be identical to the dense one."""
        B = BoundaryOperator(form, ("dirichlet", "periodic", "periodic"))
        dense = B.assemble()
        sparse = B.assemble_sparse()
        npt.assert_array_equal(sparse.todense(), dense)


# ---------------------------------------------------------------------------
# Specific boundary-condition semantics
# ---------------------------------------------------------------------------

class TestBoundarySemantics:
    """Test that boundary conditions have the expected semantic effect."""

    def test_dirichlet_zeroes_boundary_dofs(self):
        """For 0-forms with Dirichlet BCs, the first and last radial
        basis functions should be absent from the range of E."""
        Λ = DifferentialForm(0, NS, PS, TYPES)
        B = BoundaryOperator(Λ, ("dirichlet", "periodic", "periodic"))
        M = B.matrix()
        # Columns corresponding to i_r = 0 should be zero
        nr, nt, nz = Λ.nr, Λ.nt, Λ.nz
        for j in range(nt):
            for k_idx in range(nz):
                col = int(jnp.ravel_multi_index(
                    jnp.array([0, j, k_idx]), (nr, nt, nz)))
                assert jnp.sum(jnp.abs(M[:, col])) == 0
        # Columns corresponding to i_r = nr-1 should be zero
        for j in range(nt):
            for k_idx in range(nz):
                col = int(jnp.ravel_multi_index(
                    jnp.array([nr - 1, j, k_idx]), (nr, nt, nz)))
                assert jnp.sum(jnp.abs(M[:, col])) == 0

    def test_none_is_identity(self):
        """With no BCs, the boundary operator should be the identity."""
        Λ = DifferentialForm(0, NS, PS, TYPES)
        B = BoundaryOperator(Λ, ("none", "none", "none"))
        M = B.matrix()
        npt.assert_array_equal(M, jnp.eye(Λ.n))

    def test_3form_identity(self):
        """For 3-forms, the boundary operator is always the identity."""
        Λ = DifferentialForm(3, NS, PS, TYPES)
        B = BoundaryOperator(Λ, ("dirichlet", "periodic", "periodic"))
        M = B.matrix()
        npt.assert_array_equal(M, jnp.eye(Λ.n1))
