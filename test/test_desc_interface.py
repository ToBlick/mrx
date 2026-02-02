# test_desc_interface.py
"""
Tests for DESC equilibrium interface.
"""
import os

import jax
import jax.numpy as jnp
import pytest

from mrx.desc_interface import DESCWrapper, project_desc_equilibrium

jax.config.update("jax_enable_x64", True)

# Path to test DESC file
DESC_PATH = os.path.join(os.path.dirname(
    __file__), "..", "data", "desc_heliotron.h5")


@pytest.fixture
def desc_wrapper():
    """Load DESC equilibrium and create wrapper."""
    import desc.io
    eq_fam = desc.io.load(DESC_PATH)
    eq = eq_fam[-1]
    return DESCWrapper(eq)


# ============================================================================
# DESCWrapper Tests
# ============================================================================

class TestDESCWrapper:
    """Tests for DESCWrapper class."""

    def test_compute_at_points_shapes(self, desc_wrapper):
        """Test that compute_at_points returns correct shapes."""
        n_pts = 10
        points = jnp.array([
            [0.5, 0.25, 0.1],
            [0.3, 0.5, 0.2],
            [0.7, 0.75, 0.3],
        ] + [[0.5, i / n_pts, j / n_pts] for i in range(n_pts) for j in range(n_pts)])
        points = points[:n_pts]  # Take first n_pts

        result = desc_wrapper.compute_at_points(points)

        assert 'R' in result
        assert 'Z' in result
        assert 'B' in result
        assert result['R'].shape == (n_pts,)
        assert result['Z'].shape == (n_pts,)
        assert result['B'].shape == (n_pts, 3)

    def test_compute_at_points_R_positive(self, desc_wrapper):
        """Test that R coordinate is positive (tokamak/stellarator geometry)."""
        points = jnp.array([
            [0.5, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.5, 1.0, 1.0],
        ])

        result = desc_wrapper.compute_at_points(points)

        assert jnp.all(result['R'] > 0), "R should be positive for all points"

    def test_compute_at_points_periodicity(self, desc_wrapper):
        """Test that quantities are periodic in theta and zeta."""
        # Points at theta=0 and theta=1 (2π) should give same values
        points_theta_0 = jnp.array([[0.5, 0.0, 0.5]])
        points_theta_1 = jnp.array([[0.5, 1.0, 0.5]])

        result_0 = desc_wrapper.compute_at_points(points_theta_0)
        result_1 = desc_wrapper.compute_at_points(points_theta_1)

        # Should be approximately equal (periodicity)
        assert jnp.allclose(result_0['R'], result_1['R'], rtol=1e-10)
        assert jnp.allclose(result_0['Z'], result_1['Z'], rtol=1e-10)


# ============================================================================
# project_desc_equilibrium Tests
# ============================================================================

class TestProjectDescEquilibrium:
    """Tests for project_desc_equilibrium function."""

    def test_returns_all_keys(self):
        """Test that function returns all expected keys."""
        result = project_desc_equilibrium(DESC_PATH, n_resolution=4, p=2)

        expected_keys = [
            'X1_h', 'X2_h', 'F_h', 'B_h', 'B_h_xyz',
            'map_seq', 'seq', 'wrapper', 'nfp'
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_geometry_projection_accuracy(self):
        """Test that R and Z projections are accurate."""
        result = project_desc_equilibrium(DESC_PATH, n_resolution=6, p=3)

        X1_h = result['X1_h']
        X2_h = result['X2_h']
        map_seq = result['map_seq']
        wrapper = result['wrapper']

        # Evaluate at quadrature points
        desc_vals = wrapper.compute_at_points(map_seq.Q.x)
        R_exact = desc_vals['R']
        Z_exact = desc_vals['Z']

        R_interp = jax.vmap(X1_h)(map_seq.Q.x).ravel()
        Z_interp = jax.vmap(X2_h)(map_seq.Q.x).ravel()

        # Compute L2 errors
        R_L2_error = jnp.sqrt(jnp.sum((R_exact - R_interp)**2 * map_seq.Q.w) /
                              jnp.sum(R_exact**2 * map_seq.Q.w))
        Z_L2_error = jnp.sqrt(jnp.sum((Z_exact - Z_interp)**2 * map_seq.Q.w) /
                              jnp.sum(Z_exact**2 * map_seq.Q.w))

        # Should be reasonably accurate with n=6, p=3
        assert R_L2_error < 0.05, f"R L2 error too large: {R_L2_error}"
        assert Z_L2_error < 0.05, f"Z L2 error too large: {Z_L2_error}"

    def test_B_projection_accuracy(self):
        """Test that B projection is accurate."""
        result = project_desc_equilibrium(DESC_PATH, n_resolution=6, p=3)

        B_h_xyz = result['B_h_xyz']
        seq = result['seq']
        wrapper = result['wrapper']

        # Evaluate at quadrature points
        desc_vals = wrapper.compute_at_points(seq.Q.x)
        B_exact = desc_vals['B']
        B_interp = jax.vmap(B_h_xyz)(seq.Q.x)

        # Jacobian for proper L2 norm
        J = seq.J_j

        # Compute L2 error
        B_diff_sq = jnp.sum((B_exact - B_interp)**2, axis=1)
        B_exact_sq = jnp.sum(B_exact**2, axis=1)

        B_L2_error = jnp.sqrt(jnp.sum(B_diff_sq * J * seq.Q.w) /
                              jnp.sum(B_exact_sq * J * seq.Q.w))

        # Should be reasonably accurate with n=6, p=3
        assert B_L2_error < 0.05, f"B L2 error too large: {B_L2_error}"

    def test_convergence(self):
        """Test that errors decrease with increasing resolution."""
        errors = []
        for n in [4, 6]:
            result = project_desc_equilibrium(DESC_PATH, n_resolution=n, p=3)

            X1_h = result['X1_h']
            map_seq = result['map_seq']
            wrapper = result['wrapper']

            desc_vals = wrapper.compute_at_points(map_seq.Q.x)
            R_exact = desc_vals['R']
            R_interp = jax.vmap(X1_h)(map_seq.Q.x).ravel()

            R_L2_error = jnp.sqrt(jnp.sum((R_exact - R_interp)**2 * map_seq.Q.w) /
                                  jnp.sum(R_exact**2 * map_seq.Q.w))
            errors.append(float(R_L2_error))

        # Error should decrease with resolution
        assert errors[1] < errors[0], \
            f"Error should decrease: n=4 error={errors[0]}, n=6 error={errors[1]}"

    def test_F_h_callable(self):
        """Test that F_h is a valid callable mapping."""
        result = project_desc_equilibrium(DESC_PATH, n_resolution=4, p=2)
        F_h = result['F_h']

        # Test at a single point
        test_point = jnp.array([0.5, 0.25, 0.1])
        mapped = F_h(test_point)

        assert mapped.shape == (3,), f"Expected shape (3,), got {mapped.shape}"
        assert jnp.isfinite(mapped).all(), "F_h returned non-finite values"

    def test_B_h_xyz_callable(self):
        """Test that B_h_xyz is a valid callable."""
        result = project_desc_equilibrium(DESC_PATH, n_resolution=4, p=2)
        B_h_xyz = result['B_h_xyz']

        # Test at a single point
        test_point = jnp.array([0.5, 0.25, 0.1])
        B_val = B_h_xyz(test_point)

        assert B_val.shape == (3,), f"Expected shape (3,), got {B_val.shape}"
        assert jnp.isfinite(B_val).all(), "B_h_xyz returned non-finite values"


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_projection_workflow(self):
        """Test the complete projection workflow end-to-end."""
        # Project equilibrium
        result = project_desc_equilibrium(DESC_PATH, n_resolution=5, p=3)

        # Verify we can evaluate all quantities
        test_points = jnp.array([
            [0.3, 0.1, 0.2],
            [0.5, 0.5, 0.5],
            [0.7, 0.9, 0.8],
        ])

        # Evaluate F_h
        F_vals = jax.vmap(result['F_h'])(test_points)
        assert F_vals.shape == (3, 3)

        # Evaluate B_h_xyz
        B_vals = jax.vmap(result['B_h_xyz'])(test_points)
        assert B_vals.shape == (3, 3)

        # Compare with DESC
        desc_vals = result['wrapper'].compute_at_points(test_points)

        # B should be close (not exact due to interpolation error)
        B_diff = jnp.linalg.norm(B_vals - desc_vals['B'], axis=1)
        B_norm = jnp.linalg.norm(desc_vals['B'], axis=1)
        rel_error = B_diff / B_norm
        assert jnp.all(rel_error < 0.05), \
            f"B relative error too large: {rel_error}"
