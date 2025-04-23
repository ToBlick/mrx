import unittest
import jax
import jax.numpy as jnp
import numpy.testing as npt
from mrx.SplineBases import SplineBasis, TensorBasis, DerivativeSpline

jax.config.update("jax_enable_x64", True)


class TestSplineBases(unittest.TestCase):
    """Test cases for SplineBasis, TensorBasis, and DerivativeSpline classes."""

    def setUp(self):
        """Set up test fixtures."""
        # Define test parameters for different spline types
        self.n = 8  # Number of basis functions
        self.p = 3  # Polynomial degree
        
        # Create spline bases with different boundary conditions
        self.clamped = SplineBasis(self.n, self.p, 'clamped')
        self.periodic = SplineBasis(self.n, self.p, 'periodic')
        self.constant = SplineBasis(self.n, self.p, 'constant')

    def test_spline_initialization(self):
        """Test initialization of spline bases with different boundary conditions."""
        # Test clamped spline
        self.assertEqual(self.clamped.n, self.n)
        self.assertEqual(self.clamped.p, self.p)
        self.assertEqual(self.clamped.type, 'clamped')
        
        # Test periodic spline
        self.assertEqual(self.periodic.n, self.n)
        self.assertEqual(self.periodic.p, self.p)
        self.assertEqual(self.periodic.type, 'periodic')
        
        # Test constant spline
        self.assertEqual(self.constant.n, self.n)
        self.assertEqual(self.constant.p, self.p)
        self.assertEqual(self.constant.type, 'constant')

    def test_spline_evaluation(self):
        """Test evaluation of spline basis functions."""
        x = jnp.linspace(0, 1, 100)
        
        # Test clamped spline properties
        for i in range(self.n):
            vals = jax.vmap(lambda x: self.clamped(x, i))(x)
            # Values should be non-negative
            self.assertTrue(jnp.all(vals >= 0))
            # Values should be bounded by [0, 1]
            self.assertTrue(jnp.all(vals <= 1))
            # Support should be local (some values should be zero)
            self.assertTrue(jnp.any(vals == 0))
        
        # Test periodic spline properties
        for i in range(self.n):
            vals = jax.vmap(lambda x: self.periodic(x, i))(x)
            # Values at 0 and 1 should be equal for periodic splines
            npt.assert_allclose(self.periodic(0.0, i), self.periodic(1.0, i))
        
        # Test constant spline properties
        for i in range(self.n):
            vals = jax.vmap(lambda x: self.constant(x, i))(x)
            # Should be constant (all values equal)
            self.assertTrue(jnp.allclose(vals, vals[0]))

    def test_partition_of_unity(self):
        """Test that spline basis functions form a partition of unity."""
        x = jnp.linspace(0, 1, 100)
        
        # Test for clamped splines
        sum_clamped = jnp.zeros_like(x)
        for i in range(self.n):
            sum_clamped += jax.vmap(lambda x: self.clamped(x, i))(x)
        npt.assert_allclose(sum_clamped, jnp.ones_like(x), rtol=1e-10)
        
        # Test for periodic splines
        sum_periodic = jnp.zeros_like(x)
        for i in range(self.n):
            sum_periodic += jax.vmap(lambda x: self.periodic(x, i))(x)
        npt.assert_allclose(sum_periodic, jnp.ones_like(x), rtol=1e-10)

    def test_derivative_spline(self):
        """Test derivative of spline basis functions."""
        # Create derivative splines
        d_clamped = DerivativeSpline(self.clamped)
        d_periodic = DerivativeSpline(self.periodic)
        
        # Test derivative spline properties
        self.assertEqual(d_clamped.p, self.p - 1)  # Degree should decrease by 1
        self.assertEqual(d_clamped.n, self.n - 1)  # One less basis function for clamped
        self.assertEqual(d_periodic.n, self.n)      # Same number for periodic
        
        # Test derivative values at boundaries for clamped splines
        for i in range(d_clamped.n):
            self.assertEqual(d_clamped(0.0, i), 0.0)  # Zero at left boundary
            self.assertEqual(d_clamped(1.0, i), 0.0)  # Zero at right boundary

    def test_tensor_basis(self):
        """Test tensor product basis functionality."""
        # Create tensor basis from three 1D bases
        bases = [
            SplineBasis(4, 2, 'clamped'),
            SplineBasis(4, 2, 'periodic'),
            SplineBasis(4, 2, 'constant')
        ]
        tensor_basis = TensorBasis(bases)
        
        # Test tensor basis properties
        self.assertEqual(tensor_basis.n, 4 * 4 * 4)  # Total number of basis functions
        self.assertTrue(jnp.array_equal(tensor_basis.shape, jnp.array([4, 4, 4])))
        
        # Test tensor basis evaluation
        x = jnp.array([0.5, 0.5, 0.5])  # Test point
        for i in range(tensor_basis.n):
            val = tensor_basis(x, i)
            self.assertTrue(jnp.isscalar(val) or val.size == 1)
            self.assertTrue(jnp.isfinite(val))

    def test_spline_getitem(self):
        """Test the __getitem__ functionality of spline bases."""
        x = 0.5
        for i in range(self.n):
            # Test that __getitem__ returns a callable that gives same result as __call__
            self.assertEqual(self.clamped[i](x), self.clamped(x, i))
            self.assertEqual(self.periodic[i](x), self.periodic(x, i))
            self.assertEqual(self.constant[i](x), self.constant(x, i))


if __name__ == '__main__':
    unittest.main() 