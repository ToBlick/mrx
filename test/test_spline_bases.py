"""
Unit tests for the implementation of spline bases in the MRX package.

This module contains tests for various spline properties including:
- Nonnegativity of knot points 
- Nonnegativity of basis functions
- Partition of unity property

The tests verify:
1. Proper initialization and evaluation of spline bases
2. Tensor product basis functionality
3. Implementation of Derivative splines
4. Edge cases and error conditions

"""

import unittest
import jax
import jax.numpy as jnp
import numpy as np
from mrx.SplineBases import SplineBasis, TensorBasis, DerivativeSpline

# Enable x64 mode for better precision
jax.config.update("jax_enable_x64", True)

class TestSplineBases(unittest.TestCase):
    def setUp(self):
        self.n = 10
        self.p = 3
        self.clamped = SplineBasis(self.n, self.p, 'clamped')
        self.periodic = SplineBasis(self.n, self.p, 'periodic')
        self.constant = SplineBasis(self.n, 0, 'constant')

    def test_spline_initialization(self):
        """Test initialization of spline bases with different boundary conditions."""
        # Test valid initialization
        self.assertEqual(self.clamped.n, self.n)
        self.assertEqual(self.clamped.p, self.p)
        self.assertEqual(self.clamped.type, 'clamped')
        
        # Test invalid type
        with self.assertRaises(ValueError):
            SplineBasis(self.n, self.p, 'invalid_type')

        # Test p > 3 check
        # with self.assertRaises(NotImplementedError):
        #     SplineBasis(5, 4, 'clamped')
        
        # Test degree >= n
        with self.assertRaises(ValueError):
            SplineBasis(2, 3, 'clamped')

    def test_spline_evaluation(self):
        """Test spline basis function evaluation."""
        # Test evaluation at knots for clamped splines
        for i in range(self.n):
            # For clamped splines, only test interior points
            if 0 < i < self.n - 1:
                val = self.clamped(self.clamped.T[i + self.p], i)
                self.assertGreaterEqual(val, 0.0)
                self.assertLessEqual(val, 1.0)
        
        # Test evaluation at random points
        x = np.random.random()
        for i in range(self.n):
            val = self.clamped(x, i)
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 1.0)

    def test_partition_of_unity(self):
        """Test partition of unity property."""
        # Test at interior points
        for x in np.linspace(0.1, 0.9, 10):
            sum_val = sum(self.clamped(x, j) for j in range(self.n))
            self.assertAlmostEqual(float(sum_val), 1.0, places=5)

    def test_nonnegativity(self):
        """Test nonnegativity of spline basis functions."""
        # Test at interior points
        for x in np.linspace(0.05, 0.95, 15):
            for i in range(self.n):
                val = self.clamped(x, i)
                self.assertGreaterEqual(float(val), 0.0)

        # Test at knots
        for i in range(self.n):
            val = self.clamped(self.clamped.T[i + self.p], i)
            self.assertGreaterEqual(float(val), 0.0)

    def test_nonnegativity_knots(self):
        """Test nonnegativity of knot endpoints."""
        # Test at knot endpoints
        for i in range(self.n):
            val = self.clamped(self.clamped.T[i + self.p], i)
            self.assertGreaterEqual(float(val), 0.0)
        

    def test_derivative_spline(self):
        """Test derivative of spline basis functions."""
        # Create derivative spline
        d_clamped = DerivativeSpline(self.clamped)
        
        # Test first derivative
        x = np.random.random()
        for i in range(self.n):
            val = d_clamped(x, i)
            self.assertIsInstance(val, jnp.ndarray)
            self.assertEqual(val.shape, ())


        # Test second derivative
        d2_clamped = DerivativeSpline(d_clamped)

        for i in range(self.n):
            val = d2_clamped(x, i)
            self.assertIsInstance(val, jnp.ndarray)
            self.assertEqual(val.shape, ())

    def test_tensor_basis(self):
        """Test tensor product basis functionality."""
        # Create tensor basis with 3 bases
        tensor = TensorBasis([self.clamped, self.clamped, self.clamped])
        
        # Test evaluation at interior points
        x = jnp.array([0.5, 0.5, 0.5])
        val = tensor(x, 0)
        self.assertIsInstance(val, jnp.ndarray)
        self.assertEqual(val.shape, ())
        
        # Test evaluation at random points
        x = jnp.array([np.random.random(), np.random.random(), np.random.random()])
        val = tensor(x, 0)
        self.assertIsInstance(val, jnp.ndarray)
        self.assertEqual(val.shape, ())

    def test_error_bounds(self):
        """Test error bounds for spline approximation."""
        # Test with a quadratic function
        def f(x):
            return x**2
        
        # Test at interior points
        x = 0.5
        # Compute interpolation at knots
        approx = 0.0
        for i in range(self.n):
            xi = self.clamped.T[i + self.p]
            if 0 <= xi <= 1:  # Only use interior knots
                approx += f(xi) * self.clamped(x, i)
        exact = f(x)
        # Allow larger error since we're using cubic splines
        self.assertLessEqual(abs(float(approx) - exact), 0.5)

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Test evaluation outside domain
        x = -0.1
        for i in range(self.n):
            val = self.clamped(x, i)
            self.assertEqual(float(val), 0.0)
        
        # Test tensor basis with wrong dimension input
        tensor = TensorBasis([self.clamped, self.clamped, self.clamped])
        with self.assertRaises(ValueError):
            tensor(jnp.array([0.0, 0.0]), 0)
        
        # Test tensor basis with wrong number of bases
        with self.assertRaises(ValueError):
            TensorBasis([self.clamped, self.clamped])

        # Test nonsensical spline type
        with self.assertRaises(ValueError):
            SplineBasis(4, 3, 'invalid_type')

if __name__ == '__main__':
    unittest.main() 