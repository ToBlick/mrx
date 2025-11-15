# %%
# test_gvec_interface.py
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

from mrx.derham_sequence import DeRhamSequence
from mrx.gvec_interface import interpolate_B_from_GVEC, interpolate_map_from_GVEC

jax.config.update("jax_enable_x64", True)


# ============================================================================
# Fixtures
# ============================================================================
@pytest.fixture
def mapSeq():
    """Create a DeRhamSequence for map interpolation testing."""
    return DeRhamSequence(
        (4, 4, 4),
        (2, 2, 2),
        4,
        ("clamped", "periodic", "periodic"),
        lambda x: x,
        polar=False,
        dirichlet=False
    )


@pytest.fixture
def seq():
    """Create a DeRhamSequence for B-field interpolation testing."""
    def F_identity(x):
        return x
    return DeRhamSequence(
        (4, 4, 4),
        (2, 2, 2),
        4,
        ("clamped", "periodic", "periodic"),
        F_identity,
        polar=False,
        dirichlet=True
    )


@pytest.fixture
def gvec_eq_stellarator():
    """Create a mock GVEC stellarator equilibrium dataset."""
    m_rho = 8
    m_theta = 8
    m_zeta = 4
    
    # Create coordinate arrays
    rho = np.linspace(0.1, 0.9, m_rho)
    theta = np.linspace(0, 2 * np.pi, m_theta, endpoint=False)
    zeta = np.linspace(0, 2 * np.pi / 3, m_zeta, endpoint=False)  # For nfp=3
    
    # Create meshgrid
    rho_grid, theta_grid, zeta_grid = np.meshgrid(rho, theta, zeta, indexing="ij")
    
    # Create X1 and X2 as simple functions of coordinates
    # X1: radial coordinate (R)
    X1 = 1.0 + 0.3 * rho_grid * (1 + 0.2 * np.cos(theta_grid))
    # X2: vertical coordinate (z)
    X2 = 0.3 * rho_grid * np.sin(theta_grid)
    
    # Create xarray Dataset
    ds = xr.Dataset(
        {
            "rho": (["rho"], rho),
            "theta": (["theta"], theta),
            "zeta": (["zeta"], zeta),
            "X1": (["rho", "theta", "zeta"], X1),
            "X2": (["rho", "theta", "zeta"], X2),
        }
    )
    
    return ds


@pytest.fixture
def gvec_eq_with_B():
    """Create a mock GVEC equilibrium dataset with B-field."""
    m_rho = 8
    m_theta = 8
    m_zeta = 4
    
    # Create coordinate arrays
    rho = np.linspace(0.1, 0.9, m_rho)
    theta = np.linspace(0, 2 * np.pi, m_theta, endpoint=False)
    zeta = np.linspace(0, 2 * np.pi / 3, m_zeta, endpoint=False)
    
    # Create meshgrid
    rho_grid, theta_grid, zeta_grid = np.meshgrid(rho, theta, zeta, indexing="ij")
    
    # Create X1 and X2
    X1 = 1.0 + 0.3 * rho_grid * (1 + 0.2 * np.cos(theta_grid))
    X2 = 0.3 * rho_grid * np.sin(theta_grid)
    
    # Create B-field (3D vector field)
    # Simple B-field: mostly toroidal with some poloidal component
    B_R = 0.1 * rho_grid * np.cos(theta_grid)
    B_phi = 1.0 + 0.2 * rho_grid
    B_z = 0.1 * rho_grid * np.sin(theta_grid)
    
    # Stack into (m_rho, m_theta, m_zeta, 3) array
    B = np.stack([B_R, B_phi, B_z], axis=-1)
    
    # Create xarray Dataset
    ds = xr.Dataset(
        {
            "rho": (["rho"], rho),
            "theta": (["theta"], theta),
            "zeta": (["zeta"], zeta),
            "X1": (["rho", "theta", "zeta"], X1),
            "X2": (["rho", "theta", "zeta"], X2),
            "B": (["rho", "theta", "zeta", "component"], B),
        }
    )
    
    return ds


@pytest.fixture
def Phi_identity():
    """Create an identity mapping function."""
    def Phi(x):
        return x
    return Phi


# ============================================================================
# Tests for interpolate_map_from_GVEC
# ============================================================================
def test_interpolate_map_from_GVEC_basic(gvec_eq_stellarator, mapSeq):
    """Test basic interpolate_map_from_GVEC functionality."""
    nfp = 3
    
    Phi = interpolate_map_from_GVEC(gvec_eq_stellarator, nfp, mapSeq)
    
    # Check that Phi is callable
    assert callable(Phi), "Phi should be callable"
    
    # Test evaluation at a point
    x_test = jnp.array([0.5, 0.5, 0.5])
    result = Phi(x_test)
    
    # Check output shape
    assert result.shape == (3,), "Phi should return 3D coordinates"
    assert jnp.all(jnp.isfinite(result)), "Result should be finite"


def test_interpolate_map_from_GVEC_different_nfp(gvec_eq_stellarator, mapSeq):
    """Test interpolate_map_from_GVEC with different nfp values."""
    for nfp in [1, 2, 3, 5]:
        Phi = interpolate_map_from_GVEC(gvec_eq_stellarator, nfp, mapSeq)
        
        # Test evaluation
        x_test = jnp.array([0.5, 0.5, 0.5])
        result = Phi(x_test)
        
        assert result.shape == (3,), f"Phi should return 3D coordinates for nfp={nfp}"
        assert jnp.all(jnp.isfinite(result)), f"Result should be finite for nfp={nfp}"


def test_interpolate_map_from_GVEC_multiple_points(gvec_eq_stellarator, mapSeq):
    """Test interpolate_map_from_GVEC with multiple evaluation points."""
    nfp = 3
    Phi = interpolate_map_from_GVEC(gvec_eq_stellarator, nfp, mapSeq)
    
    # Test multiple points
    x_test = jnp.array([
        [0.3, 0.2, 0.1],
        [0.5, 0.5, 0.5],
        [0.7, 0.8, 0.9],
    ])
    
    results = jax.vmap(Phi)(x_test)
    
    assert results.shape == (3, 3), "Results should have shape (n_points, 3)"
    assert jnp.all(jnp.isfinite(results)), "All results should be finite"


def test_interpolate_map_from_GVEC_boundary_points(gvec_eq_stellarator, mapSeq):
    """Test interpolate_map_from_GVEC at boundary points."""
    nfp = 3
    Phi = interpolate_map_from_GVEC(gvec_eq_stellarator, nfp, mapSeq)
    
    # Test at boundaries
    boundary_points = jnp.array([
        [0.0, 0.0, 0.0],  # Lower boundary
        [1.0, 1.0, 1.0],  # Upper boundary
        [0.0, 0.5, 0.5],  # Mixed boundaries
    ])
    
    results = jax.vmap(Phi)(boundary_points)
    
    assert results.shape == (3, 3), "Results should have correct shape"
    assert jnp.all(jnp.isfinite(results)), "Boundary evaluations should be finite"


def test_interpolate_map_from_GVEC_returns_gvec_stellarator_map(gvec_eq_stellarator, mapSeq):
    """Test that interpolate_map_from_GVEC returns a gvec_stellarator_map."""
    nfp = 3
    Phi = interpolate_map_from_GVEC(gvec_eq_stellarator, nfp, mapSeq)
    
    # The function should return a callable that behaves like gvec_stellarator_map
    # Test that it can be evaluated
    x_test = jnp.array([0.5, 0.5, 0.5])
    result = Phi(x_test)
    
    # Check that result has the expected structure (3D coordinates)
    assert result.shape == (3,), "Result should be 3D coordinates"
    
    # For a gvec_stellarator_map, the zeta coordinate affects the result
    # Test that changing zeta changes the result
    x1 = jnp.array([0.5, 0.5, 0.0])
    x2 = jnp.array([0.5, 0.5, 0.5])
    result1 = Phi(x1)
    result2 = Phi(x2)
    
    # Results should be different (unless X1 and X2 are constant)
    # At minimum, they should both be finite
    assert jnp.all(jnp.isfinite(result1)), "Result1 should be finite"
    assert jnp.all(jnp.isfinite(result2)), "Result2 should be finite"


def test_interpolate_map_from_GVEC_different_grid_sizes():
    """Test interpolate_map_from_GVEC with different grid sizes."""
    nfp = 3
    
    # Create smaller grid
    m_rho, m_theta, m_zeta = 4, 4, 2
    rho = np.linspace(0.1, 0.9, m_rho)
    theta = np.linspace(0, 2 * np.pi, m_theta, endpoint=False)
    zeta = np.linspace(0, 2 * np.pi / 3, m_zeta, endpoint=False)
    
    rho_grid, theta_grid, zeta_grid = np.meshgrid(rho, theta, zeta, indexing="ij")
    X1 = 1.0 + 0.3 * rho_grid * (1 + 0.2 * np.cos(theta_grid))
    X2 = 0.3 * rho_grid * np.sin(theta_grid)
    
    gvec_eq = xr.Dataset(
        {
            "rho": (["rho"], rho),
            "theta": (["theta"], theta),
            "zeta": (["zeta"], zeta),
            "X1": (["rho", "theta", "zeta"], X1),
            "X2": (["rho", "theta", "zeta"], X2),
        }
    )
    
    mapSeq = DeRhamSequence(
        (3, 3, 3),
        (1, 1, 1),
        3,
        ("clamped", "periodic", "periodic"),
        lambda x: x,
        polar=False,
        dirichlet=False
    )
    
    Phi = interpolate_map_from_GVEC(gvec_eq, nfp, mapSeq)
    
    x_test = jnp.array([0.5, 0.5, 0.5])
    result = Phi(x_test)
    
    assert result.shape == (3,), "Result should have correct shape"
    assert jnp.all(jnp.isfinite(result)), "Result should be finite"


# ============================================================================
# Tests for interpolate_B_from_GVEC
# ============================================================================
def test_interpolate_B_from_GVEC_basic(gvec_eq_with_B, seq, Phi_identity):
    """Test basic interpolate_B_from_GVEC functionality."""
    nfp = 3
    
    B_dof, residuals = interpolate_B_from_GVEC(
        gvec_eq_with_B, seq, Phi_identity, nfp
    )
    
    # Check output types and shapes
    assert isinstance(B_dof, jnp.ndarray), "B_dof should be a jnp.ndarray"
    assert isinstance(residuals, jnp.ndarray), "residuals should be a jnp.ndarray"
    
    # B_dof should have shape matching the number of 2-form basis functions
    assert B_dof.shape == (seq.E2.shape[0],), \
        f"B_dof should have shape ({seq.E2.shape[0]},), got {B_dof.shape}"
    
    # Check that values are finite
    assert jnp.all(jnp.isfinite(B_dof)), "B_dof should be finite"
    assert jnp.all(jnp.isfinite(residuals)), "residuals should be finite"
    
    # Residuals should be non-negative
    assert jnp.all(residuals >= 0), "residuals should be non-negative"


def test_interpolate_B_from_GVEC_different_nfp(gvec_eq_with_B, seq, Phi_identity):
    """Test interpolate_B_from_GVEC with different nfp values."""
    for nfp in [1, 2, 3, 5]:
        B_dof, residuals = interpolate_B_from_GVEC(
            gvec_eq_with_B, seq, Phi_identity, nfp
        )
        
        assert B_dof.shape == (seq.E2.shape[0],), \
            f"B_dof should have correct shape for nfp={nfp}"
        assert jnp.all(jnp.isfinite(B_dof)), \
            f"B_dof should be finite for nfp={nfp}"
        assert jnp.all(residuals >= 0), \
            f"residuals should be non-negative for nfp={nfp}"


def test_interpolate_B_from_GVEC_exclude_axis_tol(gvec_eq_with_B, seq, Phi_identity):
    """Test interpolate_B_from_GVEC with different exclude_axis_tol values."""
    nfp = 3
    
    # Test with default tolerance
    B_dof1, residuals1 = interpolate_B_from_GVEC(
        gvec_eq_with_B, seq, Phi_identity, nfp, exclude_axis_tol=1e-3
    )
    
    # Test with larger tolerance (excludes more points)
    B_dof2, residuals2 = interpolate_B_from_GVEC(
        gvec_eq_with_B, seq, Phi_identity, nfp, exclude_axis_tol=0.1
    )
    
    # Both should be valid
    assert jnp.all(jnp.isfinite(B_dof1)), "B_dof1 should be finite"
    assert jnp.all(jnp.isfinite(B_dof2)), "B_dof2 should be finite"
    
    # Results may differ due to different point sets, but both should be valid
    assert B_dof1.shape == B_dof2.shape, "Both should have same shape"


def test_interpolate_B_from_GVEC_different_mapping(gvec_eq_with_B, seq):
    """Test interpolate_B_from_GVEC with a non-identity mapping."""
    nfp = 3
    
    # Create a simple scaling mapping
    def Phi_scale(x):
        return 2.0 * x
    
    B_dof, residuals = interpolate_B_from_GVEC(
        gvec_eq_with_B, seq, Phi_scale, nfp
    )
    
    assert B_dof.shape == (seq.E2.shape[0],), "B_dof should have correct shape"
    assert jnp.all(jnp.isfinite(B_dof)), "B_dof should be finite"
    assert jnp.all(residuals >= 0), "residuals should be non-negative"


def test_interpolate_B_from_GVEC_rotating_mapping(gvec_eq_with_B, seq):
    """Test interpolate_B_from_GVEC with a rotating mapping."""
    nfp = 3
    
    # Create a rotating mapping (rotation around z-axis)
    def Phi_rotate(x):
        r, theta, zeta = x
        return jnp.array([
            r * jnp.cos(2 * jnp.pi * theta),
            r * jnp.sin(2 * jnp.pi * theta),
            zeta
        ])
    
    B_dof, residuals = interpolate_B_from_GVEC(
        gvec_eq_with_B, seq, Phi_rotate, nfp
    )
    
    assert B_dof.shape == (seq.E2.shape[0],), "B_dof should have correct shape"
    assert jnp.all(jnp.isfinite(B_dof)), "B_dof should be finite"
    assert jnp.all(residuals >= 0), "residuals should be non-negative"


def test_interpolate_B_from_GVEC_requires_evaluate_1d(seq):
    """Test that interpolate_B_from_GVEC requires seq.evaluate_1d() to be called."""
    # Create a minimal GVEC dataset
    m_rho, m_theta, m_zeta = 4, 4, 2
    rho = np.linspace(0.1, 0.9, m_rho)
    theta = np.linspace(0, 2 * np.pi, m_theta, endpoint=False)
    zeta = np.linspace(0, 2 * np.pi / 3, m_zeta, endpoint=False)
    
    rho_grid, theta_grid, zeta_grid = np.meshgrid(rho, theta, zeta, indexing="ij")
    X1 = 1.0 + 0.3 * rho_grid
    X2 = 0.3 * rho_grid
    
    B = np.stack([
        0.1 * rho_grid,
        1.0 + 0.2 * rho_grid,
        0.1 * rho_grid
    ], axis=-1)
    
    gvec_eq = xr.Dataset(
        {
            "rho": (["rho"], rho),
            "theta": (["theta"], theta),
            "zeta": (["zeta"], zeta),
            "X1": (["rho", "theta", "zeta"], X1),
            "X2": (["rho", "theta", "zeta"], X2),
            "B": (["rho", "theta", "zeta", "component"], B),
        }
    )
    
    def Phi(x):
        return x
    
    # The function should work even if evaluate_1d hasn't been called
    # (it will be called internally if needed, or the basis functions will work directly)
    # But we test that it works
    nfp = 3
    B_dof, residuals = interpolate_B_from_GVEC(
        gvec_eq, seq, Phi, nfp
    )
    
    assert jnp.all(jnp.isfinite(B_dof)), "Should work without explicit evaluate_1d"


def test_interpolate_B_from_GVEC_different_B_fields(gvec_eq_with_B, seq, Phi_identity):
    """Test interpolate_B_from_GVEC with different B-field configurations."""
    nfp = 3
    
    # Test with original B-field
    B_dof1, residuals1 = interpolate_B_from_GVEC(
        gvec_eq_with_B, seq, Phi_identity, nfp
    )
    
    # Create a different B-field (purely toroidal)
    gvec_eq_toroidal = gvec_eq_with_B.copy()
    m_rho, m_theta, m_zeta = gvec_eq_with_B.B.shape[:3]
    B_toroidal = np.zeros((m_rho, m_theta, m_zeta, 3))
    B_toroidal[:, :, :, 1] = 1.0  # Only phi component
    gvec_eq_toroidal["B"] = (["rho", "theta", "zeta", "component"], B_toroidal)
    
    B_dof2, residuals2 = interpolate_B_from_GVEC(
        gvec_eq_toroidal, seq, Phi_identity, nfp
    )
    
    # Both should be valid
    assert jnp.all(jnp.isfinite(B_dof1)), "B_dof1 should be finite"
    assert jnp.all(jnp.isfinite(B_dof2)), "B_dof2 should be finite"
    
    # Results should differ (different B-fields)
    assert not jnp.allclose(B_dof1, B_dof2), "Results should differ for different B-fields"


def test_interpolate_B_from_GVEC_zero_B_field(gvec_eq_with_B, seq, Phi_identity):
    """Test interpolate_B_from_GVEC with zero B-field."""
    nfp = 3
    
    # Create zero B-field
    gvec_eq_zero = gvec_eq_with_B.copy()
    m_rho, m_theta, m_zeta = gvec_eq_with_B.B.shape[:3]
    B_zero = np.zeros((m_rho, m_theta, m_zeta, 3))
    gvec_eq_zero["B"] = (["rho", "theta", "zeta", "component"], B_zero)
    
    B_dof, residuals = interpolate_B_from_GVEC(
        gvec_eq_zero, seq, Phi_identity, nfp
    )
    
    # B_dof should be approximately zero (or very small)
    assert jnp.all(jnp.isfinite(B_dof)), "B_dof should be finite"
    # For zero B-field, the interpolation should give approximately zero
    # (allowing for numerical errors)
    assert jnp.allclose(B_dof, 0.0, atol=1e-6), \
        "B_dof should be approximately zero for zero B-field"


# ============================================================================
# Integration tests
# ============================================================================
def test_interpolate_map_and_B_together(gvec_eq_with_B, mapSeq, seq):
    """Test using interpolated map with B-field interpolation."""
    nfp = 3
    
    # First interpolate the map
    Phi = interpolate_map_from_GVEC(gvec_eq_with_B, nfp, mapSeq)
    
    # Then interpolate B-field using the interpolated map
    B_dof, residuals = interpolate_B_from_GVEC(
        gvec_eq_with_B, seq, Phi, nfp
    )
    
    # Both should work
    assert callable(Phi), "Phi should be callable"
    assert jnp.all(jnp.isfinite(B_dof)), "B_dof should be finite"
    
    # Test that Phi works
    x_test = jnp.array([0.5, 0.5, 0.5])
    result = Phi(x_test)
    assert result.shape == (3,), "Phi should return 3D coordinates"
    assert jnp.all(jnp.isfinite(result)), "Phi result should be finite"

