"""
Tests for the LazyBoundaryOperator, including dense and sparse assembly.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from mrx.boundary import LazyBoundaryOperator
from mrx.differential_forms import DifferentialForm

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NS = (5, 5, 5)
PS = (2, 2, 2)
TYPES = ("clamped", "periodic", "periodic")


@pytest.fixture(params=[0, 1, 2, 3])
def form(request):
    """DifferentialForm for each degree k."""
    return DifferentialForm(request.param, NS, PS, TYPES)


@pytest.fixture
def form_vector():
    """DifferentialForm for vector fields (k = -1)."""
    return DifferentialForm(-1, NS, PS, TYPES)


BC_CONFIGS = [
    ("dirichlet", "periodic", "periodic"),
    ("none", "none", "none"),
    ("left", "periodic", "periodic"),
    ("right", "periodic", "periodic"),
    ("dirichlet", "dirichlet", "dirichlet"),
]


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------

class TestInit:
    """Test LazyBoundaryOperator initialization."""

    def test_form_degree_stored(self, form):
        B = LazyBoundaryOperator(form, ("dirichlet", "periodic", "periodic"))
        assert B.k == form.k

    def test_total_n(self, form):
        B = LazyBoundaryOperator(form, ("dirichlet", "periodic", "periodic"))
        assert B.n == B.n1 + B.n2 + B.n3

    def test_dirichlet_reduces_nr_by_2(self, form):
        B = LazyBoundaryOperator(form, ("dirichlet", "periodic", "periodic"))
        assert B.nr == form.nr - 2

    def test_left_reduces_nr_by_1(self, form):
        B = LazyBoundaryOperator(form, ("left", "periodic", "periodic"))
        assert B.nr == form.nr - 1

    def test_right_reduces_nr_by_1(self, form):
        B = LazyBoundaryOperator(form, ("right", "periodic", "periodic"))
        assert B.nr == form.nr - 1

    def test_none_keeps_dimensions(self, form):
        B = LazyBoundaryOperator(form, ("none", "none", "none"))
        assert B.nr == form.nr
        assert B.nt == form.nt
        assert B.nz == form.nz

    def test_scalar_forms_n2_n3_zero(self):
        for k in [0, 3]:
            Λ = DifferentialForm(k, NS, PS, TYPES)
            B = LazyBoundaryOperator(Λ, ("dirichlet", "periodic", "periodic"))
            assert B.n2 == 0
            assert B.n3 == 0

    def test_vector_forms_all_components_positive(self):
        for k in [1, 2]:
            Λ = DifferentialForm(k, NS, PS, TYPES)
            B = LazyBoundaryOperator(Λ, ("dirichlet", "periodic", "periodic"))
            assert B.n1 > 0
            assert B.n2 > 0
            assert B.n3 > 0

    def test_vector_field_equal_components(self, form_vector):
        B = LazyBoundaryOperator(form_vector, ("dirichlet", "periodic", "periodic"))
        assert B.n1 == B.n2 == B.n3


# ---------------------------------------------------------------------------
# Dense assembly tests
# ---------------------------------------------------------------------------

class TestDenseAssembly:
    """Test dense matrix assembly via assemble() / matrix()."""

    def test_shape(self, form):
        B = LazyBoundaryOperator(form, ("dirichlet", "periodic", "periodic"))
        M = B.matrix()
        assert M.shape == (B.n, form.n)

    def test_binary_entries(self, form):
        """All entries should be 0 or 1."""
        B = LazyBoundaryOperator(form, ("dirichlet", "periodic", "periodic"))
        M = B.matrix()
        assert jnp.all((M == 0) | (M == 1))

    def test_at_most_one_per_row(self, form):
        """Each row should have at most one non-zero entry."""
        B = LazyBoundaryOperator(form, ("dirichlet", "periodic", "periodic"))
        M = B.matrix()
        row_sums = jnp.sum(M, axis=1)
        assert jnp.all(row_sums <= 1 + 1e-10)

    def test_matrix_equals_assemble(self, form):
        B = LazyBoundaryOperator(form, ("dirichlet", "periodic", "periodic"))
        npt.assert_array_equal(B.matrix(), B.assemble())

    def test_array_conversion(self, form):
        B = LazyBoundaryOperator(form, ("dirichlet", "periodic", "periodic"))
        arr = np.array(B)
        assert isinstance(arr, np.ndarray)
        npt.assert_array_equal(arr, np.array(B.matrix()))

    @pytest.mark.parametrize("bc_types", BC_CONFIGS)
    def test_various_bc_types(self, bc_types):
        """Assembly should succeed for all supported BC configurations."""
        for k in [0, 1, 2, 3]:
            Λ = DifferentialForm(k, NS, PS, TYPES)
            B = LazyBoundaryOperator(Λ, bc_types)
            M = B.matrix()
            assert M.shape == (B.n, Λ.n)


# ---------------------------------------------------------------------------
# Sparse assembly tests
# ---------------------------------------------------------------------------

class TestSparseAssembly:
    """Test sparse BCOO assembly via assemble_sparse() / sparse_matrix()."""

    def test_sparse_matches_dense(self, form):
        """The sparse matrix should be identical to the dense one."""
        B = LazyBoundaryOperator(form, ("dirichlet", "periodic", "periodic"))
        dense = B.assemble()
        sparse = B.assemble_sparse()
        npt.assert_array_almost_equal(sparse.todense(), dense)

    def test_sparse_matrix_wrapper(self, form):
        """sparse_matrix() should return the same result as assemble_sparse()."""
        B = LazyBoundaryOperator(form, ("dirichlet", "periodic", "periodic"))
        s1 = B.sparse_matrix()
        s2 = B.assemble_sparse()
        npt.assert_array_equal(s1.todense(), s2.todense())

    def test_sparse_shape(self, form):
        B = LazyBoundaryOperator(form, ("dirichlet", "periodic", "periodic"))
        S = B.assemble_sparse()
        assert S.shape == (B.n, form.n)

    @pytest.mark.parametrize("bc_types", BC_CONFIGS)
    def test_sparse_matches_dense_all_bcs(self, bc_types):
        """Sparse and dense should agree for every BC config and form degree."""
        for k in [0, 1, 2, 3]:
            Λ = DifferentialForm(k, NS, PS, TYPES)
            B = LazyBoundaryOperator(Λ, bc_types)
            dense = B.assemble()
            sparse = B.assemble_sparse()
            npt.assert_array_almost_equal(
                sparse.todense(), dense,
                err_msg=f"Mismatch for k={k}, bc_types={bc_types}",
            )

    def test_sparse_vector_field(self, form_vector):
        B = LazyBoundaryOperator(form_vector, ("dirichlet", "periodic", "periodic"))
        dense = B.assemble()
        sparse = B.assemble_sparse()
        npt.assert_array_almost_equal(sparse.todense(), dense)

    def test_sparse_nnz_bounded(self, form):
        """Number of stored elements should be <= nrows (at most 1 per row)."""
        B = LazyBoundaryOperator(form, ("dirichlet", "periodic", "periodic"))
        S = B.assemble_sparse()
        # BCOO stores exactly nrows entries (max_nnz=1 * nrows)
        assert S.nse <= B.n

    def test_sparse_matvec(self, form):
        """Sparse matvec should match dense matvec."""
        B = LazyBoundaryOperator(form, ("dirichlet", "periodic", "periodic"))
        dense = B.assemble()
        sparse = B.assemble_sparse()
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (form.n,))
        npt.assert_array_almost_equal(sparse @ x, dense @ x)


# ---------------------------------------------------------------------------
# Specific boundary-condition semantics
# ---------------------------------------------------------------------------

class TestBoundarySemantics:
    """Test that boundary conditions have the expected semantic effect."""

    def test_dirichlet_zeroes_boundary_dofs(self):
        """For 0-forms with Dirichlet BCs, the first and last radial
        basis functions should be absent from the range of E."""
        Λ = DifferentialForm(0, NS, PS, TYPES)
        B = LazyBoundaryOperator(Λ, ("dirichlet", "periodic", "periodic"))
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
        B = LazyBoundaryOperator(Λ, ("none", "none", "none"))
        M = B.matrix()
        npt.assert_array_equal(M, jnp.eye(Λ.n))

    def test_3form_identity(self):
        """For 3-forms, the boundary operator is always the identity."""
        Λ = DifferentialForm(3, NS, PS, TYPES)
        B = LazyBoundaryOperator(Λ, ("dirichlet", "periodic", "periodic"))
        M = B.matrix()
        npt.assert_array_equal(M, jnp.eye(Λ.n1))

    def test_dirichlet_smaller_than_none(self):
        Λ = DifferentialForm(0, NS, PS, TYPES)
        B_d = LazyBoundaryOperator(Λ, ("dirichlet", "dirichlet", "dirichlet"))
        B_n = LazyBoundaryOperator(Λ, ("none", "none", "none"))
        assert B_d.n < B_n.n
