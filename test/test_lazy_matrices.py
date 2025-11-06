import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
from mrx.differential_forms import DifferentialForm
from mrx.lazy_matrices import (
    LazyMassMatrix,
    LazyDerivativeMatrix,
    LazyProjectionMatrix,
    LazyDoubleCurlMatrix,
    LazyStiffnessMatrix,
    LazyDoubleDivergenceMatrix,
)
from mrx.polar import LazyExtractionOperator, get_xi
from mrx.quadrature import QuadratureRule
from mrx.boundary import LazyBoundaryOperator

jax.config.update("jax_enable_x64", True)


# Helper fixtures
@pytest.fixture
def simple_form_0():
    """Create a simple 0-form for testing."""
    return DifferentialForm(0, (5, 5, 5), (2, 2, 2), ("clamped", "periodic", "periodic"))


@pytest.fixture
def simple_form_1():
    """Create a simple 1-form for testing."""
    return DifferentialForm(1, (5, 5, 5), (2, 2, 2), ("clamped", "periodic", "periodic"))


@pytest.fixture
def simple_form_2():
    """Create a simple 2-form for testing."""
    return DifferentialForm(2, (5, 5, 5), (2, 2, 2), ("clamped", "periodic", "periodic"))


@pytest.fixture
def simple_form_3():
    """Create a simple 3-form for testing."""
    return DifferentialForm(3, (5, 5, 5), (2, 2, 2), ("clamped", "periodic", "periodic"))


@pytest.fixture
def simple_form_vector():
    """Create a simple vector field (-1-form) for testing."""
    return DifferentialForm(-1, (5, 5, 5), (2, 2, 2), ("clamped", "periodic", "periodic"))


@pytest.fixture
def identity_quadrature(simple_form_0):
    """Create a quadrature rule for testing."""
    return QuadratureRule(simple_form_0, 4)


@pytest.fixture
def identity_map():
    """Create an identity mapping function."""
    return lambda x: x


# Tests for LazyMassMatrix
def test_lazy_mass_matrix_init(simple_form_0, identity_quadrature, identity_map):
    """Test LazyMassMatrix initialization."""
    M = LazyMassMatrix(simple_form_0, identity_quadrature, identity_map)
    
    assert M.Λ0 == simple_form_0, "Λ0 should match input form"
    assert M.Λ1 == simple_form_0, "Λ1 should match input form for mass matrix"
    assert M.Q == identity_quadrature, "Quadrature rule should match"
    assert M.n0 == simple_form_0.n, "n0 should match form size"
    assert M.n1 == simple_form_0.n, "n1 should match form size"


def test_lazy_mass_matrix_0form(simple_form_0, identity_quadrature, identity_map):
    """Test LazyMassMatrix for 0-forms."""
    M = LazyMassMatrix(simple_form_0, identity_quadrature, identity_map)
    M_assembled = M.matrix()
    
    # Check matrix properties
    assert M_assembled.shape == (simple_form_0.n, simple_form_0.n), "Matrix shape should be (n, n)"
    assert jnp.allclose(M_assembled, M_assembled.T), "Mass matrix should be symmetric"
    assert jnp.all(jnp.linalg.eigvals(M_assembled) >= -1e-10), "Mass matrix should be positive semi-definite"


def test_lazy_mass_matrix_1form(simple_form_1, identity_quadrature, identity_map):
    """Test LazyMassMatrix for 1-forms."""
    M = LazyMassMatrix(simple_form_1, identity_quadrature, identity_map)
    M_assembled = M.matrix()
    
    assert M_assembled.shape == (simple_form_1.n, simple_form_1.n), "Matrix shape should be (n, n)"
    assert jnp.allclose(M_assembled, M_assembled.T), "Mass matrix should be symmetric"


def test_lazy_mass_matrix_2form(simple_form_2, identity_quadrature, identity_map):
    """Test LazyMassMatrix for 2-forms."""
    M = LazyMassMatrix(simple_form_2, identity_quadrature, identity_map)
    M_assembled = M.matrix()
    
    assert M_assembled.shape == (simple_form_2.n, simple_form_2.n), "Matrix shape should be (n, n)"
    assert jnp.allclose(M_assembled, M_assembled.T), "Mass matrix should be symmetric"


def test_lazy_mass_matrix_3form(simple_form_3, identity_quadrature, identity_map):
    """Test LazyMassMatrix for 3-forms."""
    M = LazyMassMatrix(simple_form_3, identity_quadrature, identity_map)
    M_assembled = M.matrix()
    
    assert M_assembled.shape == (simple_form_3.n, simple_form_3.n), "Matrix shape should be (n, n)"
    assert jnp.allclose(M_assembled, M_assembled.T), "Mass matrix should be symmetric"


def test_lazy_mass_matrix_vector(simple_form_vector, identity_quadrature, identity_map):
    """Test LazyMassMatrix for vector fields."""
    M = LazyMassMatrix(simple_form_vector, identity_quadrature, identity_map)
    M_assembled = M.matrix()
    
    assert M_assembled.shape == (simple_form_vector.n, simple_form_vector.n), "Matrix shape should be (n, n)"
    assert jnp.allclose(M_assembled, M_assembled.T), "Mass matrix should be symmetric"


def test_lazy_mass_matrix_array_conversion(simple_form_0, identity_quadrature, identity_map):
    """Test LazyMassMatrix __array__ method."""
    M = LazyMassMatrix(simple_form_0, identity_quadrature, identity_map)
    M_array = np.array(M)
    
    assert isinstance(M_array, np.ndarray), "__array__ should return numpy array"
    assert M_array.shape == (simple_form_0.n, simple_form_0.n), "Array shape should match matrix shape"


# Tests for LazyDerivativeMatrix
def test_lazy_derivative_matrix_gradient(simple_form_0, simple_form_1, identity_quadrature, identity_map):
    """Test LazyDerivativeMatrix for gradient (0-form to 1-form)."""
    D = LazyDerivativeMatrix(simple_form_0, simple_form_1, identity_quadrature, identity_map)
    D_assembled = D.matrix()
    
    # Matrix shape is (n1, n0) because einsum returns (l, i) = (n1, n0)
    assert D_assembled.shape == (simple_form_1.n, simple_form_0.n), "Matrix shape should be (n1, n0)"


def test_lazy_derivative_matrix_curl(simple_form_1, simple_form_2, identity_quadrature, identity_map):
    """Test LazyDerivativeMatrix for curl (1-form to 2-form)."""
    D = LazyDerivativeMatrix(simple_form_1, simple_form_2, identity_quadrature, identity_map)
    D_assembled = D.matrix()
    
    # Matrix shape is (n1, n0) because einsum returns (l, i) = (n1, n0)
    assert D_assembled.shape == (simple_form_2.n, simple_form_1.n), "Matrix shape should be (n1, n0)"


def test_lazy_derivative_matrix_div(simple_form_2, simple_form_3, identity_quadrature, identity_map):
    """Test LazyDerivativeMatrix for divergence (2-form to 3-form)."""
    D = LazyDerivativeMatrix(simple_form_2, simple_form_3, identity_quadrature, identity_map)
    D_assembled = D.matrix()
    
    # Matrix shape is (n1, n0) because einsum returns (l, i) = (n1, n0)
    assert D_assembled.shape == (simple_form_3.n, simple_form_2.n), "Matrix shape should be (n1, n0)"


def test_lazy_derivative_matrix_3form(simple_form_3, identity_quadrature, identity_map):
    """Test LazyDerivativeMatrix for 3-forms (should return zero matrix)."""
    D = LazyDerivativeMatrix(simple_form_3, simple_form_3, identity_quadrature, identity_map)
    D_assembled = D.matrix()
    
    assert D_assembled.shape == (simple_form_3.n, simple_form_3.n), "Matrix shape should be (n, n)"
    assert jnp.allclose(D_assembled, 0.0), "Derivative of 3-form should be zero"


# Tests for LazyProjectionMatrix
def test_lazy_projection_matrix(simple_form_0, simple_form_1, identity_quadrature, identity_map):
    """Test LazyProjectionMatrix."""
    P = LazyProjectionMatrix(simple_form_0, simple_form_1, identity_quadrature, identity_map)
    P_assembled = P.matrix()
    
    # Matrix shape is (n1, n0) because einsum returns (l, i) = (n1, n0)
    assert P_assembled.shape == (simple_form_1.n, simple_form_0.n), "Matrix shape should be (n1, n0)"


# Tests for LazyDoubleCurlMatrix
def test_lazy_double_curl_matrix(simple_form_1, identity_quadrature, identity_map):
    """Test LazyDoubleCurlMatrix."""
    C = LazyDoubleCurlMatrix(simple_form_1, identity_quadrature, identity_map)
    C_assembled = C.matrix()
    
    assert C_assembled.shape == (simple_form_1.n, simple_form_1.n), "Matrix shape should be (n, n)"
    assert jnp.allclose(C_assembled, C_assembled.T), "Double curl matrix should be symmetric"


# Tests for LazyStiffnessMatrix
def test_lazy_stiffness_matrix(simple_form_0, identity_quadrature, identity_map):
    """Test LazyStiffnessMatrix."""
    K = LazyStiffnessMatrix(simple_form_0, identity_quadrature, identity_map)
    K_assembled = K.matrix()
    
    assert K_assembled.shape == (simple_form_0.n, simple_form_0.n), "Matrix shape should be (n, n)"
    assert jnp.allclose(K_assembled, K_assembled.T), "Stiffness matrix should be symmetric"
    assert jnp.all(jnp.linalg.eigvals(K_assembled) >= -1e-10), "Stiffness matrix should be positive semi-definite"


# Tests for LazyDoubleDivergenceMatrix
def test_lazy_double_divergence_matrix(simple_form_2, identity_quadrature, identity_map):
    """Test LazyDoubleDivergenceMatrix."""
    D = LazyDoubleDivergenceMatrix(simple_form_2, identity_quadrature, identity_map)
    D_assembled = D.matrix()
    
    assert D_assembled.shape == (simple_form_2.n, simple_form_2.n), "Matrix shape should be (n, n)"
    assert jnp.allclose(D_assembled, D_assembled.T), "Double divergence matrix should be symmetric"


# Tests for LazyExtractionOperator
@pytest.fixture
def simple_extraction_operator_0(simple_form_0):
    """Create a LazyExtractionOperator for 0-forms."""
    ξ = get_xi(simple_form_0.nχ)
    return LazyExtractionOperator(simple_form_0, ξ, zero_bc=False), simple_form_0


@pytest.fixture
def simple_extraction_operator_1(simple_form_1):
    """Create a LazyExtractionOperator for 1-forms."""
    ξ = get_xi(simple_form_1.nχ)
    return LazyExtractionOperator(simple_form_1, ξ, zero_bc=False), simple_form_1


@pytest.fixture
def simple_extraction_operator_2(simple_form_2):
    """Create a LazyExtractionOperator for 2-forms."""
    ξ = get_xi(simple_form_2.nχ)
    return LazyExtractionOperator(simple_form_2, ξ, zero_bc=False), simple_form_2


@pytest.fixture
def simple_extraction_operator_3(simple_form_3):
    """Create a LazyExtractionOperator for 3-forms."""
    ξ = get_xi(simple_form_3.nχ)
    return LazyExtractionOperator(simple_form_3, ξ, zero_bc=False), simple_form_3


def test_lazy_extraction_operator_init(simple_extraction_operator_0):
    """Test LazyExtractionOperator initialization."""
    E, Λ = simple_extraction_operator_0
    
    assert E.k == Λ.k, "Form degree should match"
    assert E.Λ == Λ, "Reference to form should match"
    assert E.nr == Λ.nr, "nr should match"
    assert E.nχ == Λ.nχ, "nχ should match"
    assert E.nζ == Λ.nζ, "nζ should match"


def test_lazy_extraction_operator_0form(simple_extraction_operator_0):
    """Test LazyExtractionOperator for 0-forms."""
    E, Λ = simple_extraction_operator_0
    E_assembled = E.matrix()
    
    assert E_assembled.shape == (E.n, Λ.n), "Matrix shape should be (n_extracted, n_original)"
    assert E.n == E.n1, "For 0-forms, n should equal n1"
    assert E.n2 == 0, "For 0-forms, n2 should be 0"
    assert E.n3 == 0, "For 0-forms, n3 should be 0"


def test_lazy_extraction_operator_1form(simple_extraction_operator_1):
    """Test LazyExtractionOperator for 1-forms."""
    E, Λ = simple_extraction_operator_1
    E_assembled = E.matrix()
    
    assert E_assembled.shape == (E.n, Λ.n), "Matrix shape should be (n_extracted, n_original)"
    assert E.n == E.n1 + E.n2 + E.n3, "Total n should equal sum of components"


def test_lazy_extraction_operator_2form(simple_extraction_operator_2):
    """Test LazyExtractionOperator for 2-forms."""
    E, Λ = simple_extraction_operator_2
    E_assembled = E.matrix()
    
    assert E_assembled.shape == (E.n, Λ.n), "Matrix shape should be (n_extracted, n_original)"
    assert E.n == E.n1 + E.n2 + E.n3, "Total n should equal sum of components"


def test_lazy_extraction_operator_3form(simple_extraction_operator_3):
    """Test LazyExtractionOperator for 3-forms."""
    E, Λ = simple_extraction_operator_3
    E_assembled = E.matrix()
    
    assert E_assembled.shape == (E.n, Λ.n), "Matrix shape should be (n_extracted, n_original)"
    assert E.n == E.n1, "For 3-forms, n should equal n1"
    assert E.n2 == 0, "For 3-forms, n2 should be 0"
    assert E.n3 == 0, "For 3-forms, n3 should be 0"


def test_lazy_extraction_operator_zero_bc(simple_form_0):
    """Test LazyExtractionOperator with zero boundary conditions."""
    ξ = get_xi(simple_form_0.nχ)
    E_zero = LazyExtractionOperator(simple_form_0, ξ, zero_bc=True)
    E_free = LazyExtractionOperator(simple_form_0, ξ, zero_bc=False)
    
    assert E_zero.o == 1, "Zero BC should set o=1"
    assert E_free.o == 0, "Free BC should set o=0"
    assert E_zero.n <= E_free.n, "Zero BC should reduce or keep same size"


def test_lazy_extraction_operator_array_conversion(simple_extraction_operator_0):
    """Test LazyExtractionOperator __array__ method."""
    E, _ = simple_extraction_operator_0
    E_array = np.array(E)
    
    assert isinstance(E_array, np.ndarray), "__array__ should return numpy array"
    assert E_array.shape == E.matrix().shape, "Array shape should match matrix shape"


def test_lazy_extraction_operator_vector_index(simple_extraction_operator_1):
    """Test LazyExtractionOperator _vector_index method."""
    E, _ = simple_extraction_operator_1
    
    # Test first few indices
    for i in range(min(10, E.n)):
        category, index = E._vector_index(i)
        assert category in [0, 1, 2], "Category should be 0, 1, or 2"
        assert index >= 0, "Index should be non-negative"


# Tests for get_xi
def test_get_xi():
    """Test get_xi function."""
    nχ = 8
    ξ = get_xi(nχ)
    
    assert ξ.shape == (3, 2, nχ), f"ξ should have shape (3, 2, {nχ})"
    
    # Check that ξ[0, 0, :] + ξ[1, 0, :] + ξ[2, 0, :] = 1 (partition of unity)
    sum_xi0 = ξ[0, 0, :] + ξ[1, 0, :] + ξ[2, 0, :]
    npt.assert_allclose(sum_xi0, 1.0, atol=1e-12, err_msg="Partition of unity for ξ[*, 0, :] should hold")
    
    # Check that values are in reasonable range
    assert jnp.all(ξ >= -1e-10), "ξ values should be non-negative (within tolerance)"
    assert jnp.all(ξ <= 2.0), "ξ values should be bounded"


def test_get_xi_different_sizes():
    """Test get_xi with different sizes."""
    for nχ in [4, 8, 16, 32]:
        ξ = get_xi(nχ)
        assert ξ.shape == (3, 2, nχ), f"ξ should have shape (3, 2, {nχ}) for nχ={nχ}"


# Tests for matrix properties
def test_mass_matrix_symmetry_identity_map(simple_form_0, identity_quadrature, identity_map):
    """Test that mass matrix is symmetric for identity map."""
    M = LazyMassMatrix(simple_form_0, identity_quadrature, identity_map)
    M_assembled = M.matrix()
    
    npt.assert_allclose(
        M_assembled, M_assembled.T, atol=1e-10,
        err_msg="Mass matrix should be symmetric"
    )


def test_stiffness_matrix_symmetry(simple_form_0, identity_quadrature, identity_map):
    """Test that stiffness matrix is symmetric."""
    K = LazyStiffnessMatrix(simple_form_0, identity_quadrature, identity_map)
    K_assembled = K.matrix()
    
    npt.assert_allclose(
        K_assembled, K_assembled.T, atol=1e-10,
        err_msg="Stiffness matrix should be symmetric"
    )


def test_double_curl_matrix_symmetry(simple_form_1, identity_quadrature, identity_map):
    """Test that double curl matrix is symmetric."""
    C = LazyDoubleCurlMatrix(simple_form_1, identity_quadrature, identity_map)
    C_assembled = C.matrix()
    
    npt.assert_allclose(
        C_assembled, C_assembled.T, atol=1e-10,
        err_msg="Double curl matrix should be symmetric"
    )


def test_double_divergence_matrix_symmetry(simple_form_2, identity_quadrature, identity_map):
    """Test that double divergence matrix is symmetric."""
    D = LazyDoubleDivergenceMatrix(simple_form_2, identity_quadrature, identity_map)
    D_assembled = D.matrix()
    
    npt.assert_allclose(
        D_assembled, D_assembled.T, atol=1e-10,
        err_msg="Double divergence matrix should be symmetric"
    )


# Tests for different mappings
def test_lazy_mass_matrix_with_mapping(simple_form_0, identity_quadrature):
    """Test LazyMassMatrix with a non-identity mapping."""
    # Simple scaling mapping
    def scaling_map(x):
        return 2.0 * x
    
    M = LazyMassMatrix(simple_form_0, identity_quadrature, scaling_map)
    M_assembled = M.matrix()
    
    assert M_assembled.shape == (simple_form_0.n, simple_form_0.n), "Matrix shape should be (n, n)"
    assert jnp.allclose(M_assembled, M_assembled.T), "Mass matrix should be symmetric"


# Tests for extraction operator with different boundary conditions
@pytest.mark.parametrize("zero_bc", [True, False])
def test_extraction_operator_boundary_conditions(simple_form_0, zero_bc):
    """Test LazyExtractionOperator with different boundary conditions."""
    ξ = get_xi(simple_form_0.nχ)
    E = LazyExtractionOperator(simple_form_0, ξ, zero_bc=zero_bc)
    E_assembled = E.matrix()
    
    assert E_assembled.shape == (E.n, simple_form_0.n), "Matrix shape should be correct"
    assert E.o == (1 if zero_bc else 0), "Offset should match boundary condition type"


# Tests for LazyBoundaryOperator
@pytest.fixture
def boundary_operator_0form(simple_form_0):
    """Create a LazyBoundaryOperator for 0-forms."""
    return LazyBoundaryOperator(simple_form_0, ("dirichlet", "periodic", "periodic")), simple_form_0


@pytest.fixture
def boundary_operator_1form(simple_form_1):
    """Create a LazyBoundaryOperator for 1-forms."""
    return LazyBoundaryOperator(simple_form_1, ("dirichlet", "periodic", "periodic")), simple_form_1


@pytest.fixture
def boundary_operator_2form(simple_form_2):
    """Create a LazyBoundaryOperator for 2-forms."""
    return LazyBoundaryOperator(simple_form_2, ("dirichlet", "periodic", "periodic")), simple_form_2


@pytest.fixture
def boundary_operator_3form(simple_form_3):
    """Create a LazyBoundaryOperator for 3-forms."""
    return LazyBoundaryOperator(simple_form_3, ("dirichlet", "periodic", "periodic")), simple_form_3


def test_lazy_boundary_operator_init_0form(simple_form_0):
    """Test LazyBoundaryOperator initialization for 0-forms."""
    B = LazyBoundaryOperator(simple_form_0, ("dirichlet", "periodic", "periodic"))
    
    assert B.k == simple_form_0.k, "Form degree should match"
    assert B.Λ == simple_form_0, "Reference to form should match"
    assert B.nr == simple_form_0.nr - 2, "Dirichlet BC should reduce nr by 2"
    assert B.n == B.n1, "For 0-forms, n should equal n1"
    assert B.n2 == 0, "For 0-forms, n2 should be 0"
    assert B.n3 == 0, "For 0-forms, n3 should be 0"


def test_lazy_boundary_operator_init_1form(simple_form_1):
    """Test LazyBoundaryOperator initialization for 1-forms."""
    B = LazyBoundaryOperator(simple_form_1, ("dirichlet", "periodic", "periodic"))
    
    assert B.k == simple_form_1.k, "Form degree should match"
    assert B.n == B.n1 + B.n2 + B.n3, "Total n should equal sum of components"
    assert B.n1 > 0, "n1 should be positive for 1-forms"
    assert B.n2 > 0, "n2 should be positive for 1-forms"
    assert B.n3 > 0, "n3 should be positive for 1-forms"


def test_lazy_boundary_operator_init_2form(simple_form_2):
    """Test LazyBoundaryOperator initialization for 2-forms."""
    B = LazyBoundaryOperator(simple_form_2, ("dirichlet", "periodic", "periodic"))
    
    assert B.k == simple_form_2.k, "Form degree should match"
    assert B.n == B.n1 + B.n2 + B.n3, "Total n should equal sum of components"


def test_lazy_boundary_operator_init_3form(simple_form_3):
    """Test LazyBoundaryOperator initialization for 3-forms."""
    B = LazyBoundaryOperator(simple_form_3, ("dirichlet", "periodic", "periodic"))
    
    assert B.k == simple_form_3.k, "Form degree should match"
    assert B.n == B.n1, "For 3-forms, n should equal n1"
    assert B.n2 == 0, "For 3-forms, n2 should be 0"
    assert B.n3 == 0, "For 3-forms, n3 should be 0"


def test_lazy_boundary_operator_matrix_0form(boundary_operator_0form):
    """Test LazyBoundaryOperator matrix assembly for 0-forms."""
    B, Λ = boundary_operator_0form
    B_assembled = B.matrix()
    
    assert B_assembled.shape == (B.n, Λ.n), "Matrix shape should be (n_boundary, n_original)"
    # Matrix should be sparse (mostly zeros)
    assert jnp.sum(B_assembled > 0) <= B.n, "Matrix should have at most n non-zero entries per row"


def test_lazy_boundary_operator_matrix_1form(boundary_operator_1form):
    """Test LazyBoundaryOperator matrix assembly for 1-forms."""
    B, Λ = boundary_operator_1form
    B_assembled = B.matrix()
    
    assert B_assembled.shape == (B.n, Λ.n), "Matrix shape should be (n_boundary, n_original)"


def test_lazy_boundary_operator_matrix_2form(boundary_operator_2form):
    """Test LazyBoundaryOperator matrix assembly for 2-forms."""
    B, Λ = boundary_operator_2form
    B_assembled = B.matrix()
    
    assert B_assembled.shape == (B.n, Λ.n), "Matrix shape should be (n_boundary, n_original)"


def test_lazy_boundary_operator_matrix_3form(boundary_operator_3form):
    """Test LazyBoundaryOperator matrix assembly for 3-forms."""
    B, Λ = boundary_operator_3form
    B_assembled = B.matrix()
    
    assert B_assembled.shape == (B.n, Λ.n), "Matrix shape should be (n_boundary, n_original)"
    # For 3-forms, each row should have at most one non-zero entry
    for row in range(B.n):
        row_sum = jnp.sum(B_assembled[row, :])
        assert row_sum <= 1.0 + 1e-10, f"Row {row} should have at most one non-zero entry"


def test_lazy_boundary_operator_array_conversion(boundary_operator_0form):
    """Test LazyBoundaryOperator __array__ method."""
    B, _ = boundary_operator_0form
    B_array = np.array(B)
    
    assert isinstance(B_array, np.ndarray), "__array__ should return numpy array"
    assert B_array.shape == B.matrix().shape, "Array shape should match matrix shape"


def test_lazy_boundary_operator_vector_index(boundary_operator_1form):
    """Test LazyBoundaryOperator _vector_index method."""
    B, _ = boundary_operator_1form
    
    # Test first few indices
    for i in range(min(10, B.n)):
        category, index = B._vector_index(i)
        assert category in [0, 1, 2], "Category should be 0, 1, or 2"
        assert index >= 0, "Index should be non-negative"


def test_lazy_boundary_operator_unravel_index(boundary_operator_0form):
    """Test LazyBoundaryOperator _unravel_index method."""
    B, _ = boundary_operator_0form
    
    # Test first few indices
    for i in range(min(10, B.n)):
        category, i_idx, j_idx, k_idx = B._unravel_index(i)
        assert category == 0, "For 0-forms, category should be 0"
        assert 0 <= i_idx < B.nr, "i index should be in valid range"
        assert 0 <= j_idx < B.nχ, "j index should be in valid range"
        assert 0 <= k_idx < B.nζ, "k index should be in valid range"


@pytest.mark.parametrize("bc_type", ["dirichlet", "left", "right", "periodic"])
def test_lazy_boundary_operator_boundary_types_0form(simple_form_0, bc_type):
    """Test LazyBoundaryOperator with different boundary condition types."""
    B = LazyBoundaryOperator(simple_form_0, (bc_type, "periodic", "periodic"))
    B_assembled = B.matrix()
    
    assert B_assembled.shape == (B.n, simple_form_0.n), "Matrix shape should be correct"
    
    # Check dimension reduction based on BC type
    if bc_type == "dirichlet":
        assert B.nr == simple_form_0.nr - 2, "Dirichlet should reduce by 2"
    elif bc_type == "left" or bc_type == "right":
        assert B.nr == simple_form_0.nr - 1, "Left/right should reduce by 1"
    else:
        assert B.nr == simple_form_0.nr, "No BC should keep original size"


def test_lazy_boundary_operator_dirichlet_reduction(simple_form_0):
    """Test that dirichlet boundary conditions reduce dimensions correctly."""
    B_dirichlet = LazyBoundaryOperator(simple_form_0, ("dirichlet", "dirichlet", "dirichlet"))
    B_none = LazyBoundaryOperator(simple_form_0, ("periodic", "periodic", "periodic"))
    
    assert B_dirichlet.nr == simple_form_0.nr - 2, "Dirichlet should reduce nr by 2"
    assert B_dirichlet.nχ == simple_form_0.nχ - 2, "Dirichlet should reduce nχ by 2"
    assert B_dirichlet.nζ == simple_form_0.nζ - 2, "Dirichlet should reduce nζ by 2"
    assert B_dirichlet.n < B_none.n, "Dirichlet BC should reduce total size"


def test_lazy_boundary_operator_left_right_reduction(simple_form_0):
    """Test that left/right boundary conditions reduce dimensions correctly."""
    B_left = LazyBoundaryOperator(simple_form_0, ("left", "periodic", "periodic"))
    B_right = LazyBoundaryOperator(simple_form_0, ("right", "periodic", "periodic"))
    B_none = LazyBoundaryOperator(simple_form_0, ("periodic", "periodic", "periodic"))
    
    assert B_left.nr == simple_form_0.nr - 1, "Left should reduce nr by 1"
    assert B_right.nr == simple_form_0.nr - 1, "Right should reduce nr by 1"
    assert B_left.n < B_none.n, "Left BC should reduce total size"
    assert B_right.n < B_none.n, "Right BC should reduce total size"


def test_lazy_boundary_operator_vector_field(simple_form_vector):
    """Test LazyBoundaryOperator for vector fields (-1-forms)."""
    B = LazyBoundaryOperator(simple_form_vector, ("dirichlet", "periodic", "periodic"))
    B_assembled = B.matrix()
    
    assert B.k == -1, "Form degree should be -1"
    assert B.n == B.n1 + B.n2 + B.n3, "Total n should equal sum of components"
    assert B.n1 == B.n2 == B.n3, "For vector fields, all components should have same size"
    assert B_assembled.shape == (B.n, simple_form_vector.n), "Matrix shape should be correct"


def test_lazy_boundary_operator_element_computation(boundary_operator_0form):
    """Test LazyBoundaryOperator _element method."""
    B, Λ = boundary_operator_0form
    
    # Test a few elements
    for row in range(min(5, B.n)):
        for col in range(min(5, Λ.n)):
            element = B._element(row, col)
            element_val = float(element) if isinstance(element, jnp.ndarray) else element
            assert element_val in [0, 1], "Element should be 0 or 1 (binary operator)"
            assert isinstance(element, (int, jnp.ndarray)), "Element should be numeric"


def test_lazy_boundary_operator_sparsity(boundary_operator_0form):
    """Test that LazyBoundaryOperator produces sparse matrices."""
    B, Λ = boundary_operator_0form
    B_assembled = B.matrix()
    
    # Each row should have at most one non-zero entry (extraction operator)
    for row in range(B.n):
        row_sum = jnp.sum(B_assembled[row, :])
        assert row_sum <= 1.0 + 1e-10, f"Row {row} should have at most one non-zero entry"

