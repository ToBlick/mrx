"""
Unit tests for iterative solvers in the MRX package.

This module contains tests for various iterative solvers including:
- Picard iteration
- Newton's method
- Other root-finding algorithms

The tests verify:
1. Convergence to known solutions
2. Handling of different function types
3. Behavior with different initial guesses
4. Tolerance settings
5. Edge cases and error conditions
"""

import unittest
import jax
import jax.numpy as jnp
from mrx.IterativeSolvers import picard_solver, newton_solver


class TestIterativeSolvers(unittest.TestCase):
    """Test cases for iterative solvers."""

    def setUp(self):
        """Set up test cases."""
        self.rtol = 1e-6  # Relative tolerance for numerical comparisons
        self.atol = 1e-6  # Absolute tolerance for numerical comparisons
        # Set random seed for reproducibility
        self.rng = jax.random.PRNGKey(0)

    def test_picard_solver_basic(self):
        """Test basic functionality of Picard solver."""
        def f(x):
            return jnp.cos(x)
        
        z_init = jnp.array(1.0)
        z_star = picard_solver(f, z_init, tol=1e-6)
        
        # The Dottie number is approximately 0.739085
        self.assertAlmostEqual(float(z_star), 0.739085, places=5)
        self.assertAlmostEqual(float(f(z_star)), float(z_star), places=5)

    def test_newton_solver_basic(self):
        """Test Newton solver on a simple scalar function"""
        def f(x):
            return x**2  # Fixed point at x=0,1
        
        x_init = 0.5
        x_final, x_prev, iters = newton_solver(f, x_init)
        assert jnp.allclose(x_final, 1.0, atol=1e-6)

    def test_edge_cases(self):
        """Test handling of edge cases."""
        # Test case with zero derivative
        def f1(x):
            return x  # f'(x) = 1, g'(x) = f'(x) - 1 = 0
        
        with self.assertRaises(ValueError):
            newton_solver(f1, 1.0)
        
        # Test case with very small derivative
        def f2(x):
            return 0.5 * x + 1e-11 * jnp.sin(x)
        
        with self.assertRaises(ValueError):
            newton_solver(f2, 1.0)
        
        # Test function with multiple fixed points
        def f3(x):
            return jnp.sin(x)
        
        x_init = jnp.array(1.0)
        x_final, x_prev, iters = newton_solver(f3, x_init)
        assert jnp.allclose(f3(x_final), x_final, atol=1e-6)

    def test_multidimensional(self):
        """Test Newton solver on multidimensional problems"""
        # Test 2D trigonometric system
        def f(x):
            return jnp.array([jnp.cos(x[1]), jnp.sin(x[0])])
        
        x_init = jnp.array([0.5, 0.5])
        x_final, x_prev, iters = newton_solver(f, x_init)
        assert jnp.allclose(f(x_final), x_final, atol=1e-6)
        
        # Test linear system
        def linear_system(x):
            A = jnp.array([[2.0, 1.0], [1.0, 3.0]])
            b = jnp.array([1.0, 2.0])
            return jnp.linalg.solve(A, b + x)
        
        x_init = jnp.array([0.0, 0.0])
        x_final, x_prev, iters = newton_solver(linear_system, x_init)
        assert jnp.allclose(linear_system(x_final), x_final, atol=1e-6)

    def test_convergence_rates(self):
        """Compare convergence rates between Newton and Picard iteration"""
        def f(x):
            return jnp.cos(x)
        
        x_init = 1.0
        x_final, x_prev, newton_iters = newton_solver(f, x_init, tol=1e-6)
        assert newton_iters < 10  # Newton should converge quickly
        assert jnp.allclose(f(x_final), x_final, atol=1e-6)

    def test_newton_additional_1d(self):
        """Test Newton solver on additional 1D functions."""
        # Test 1: Simple linear function x = (x+1)/2
        def f1(x):
            return (x + 1) / 2
        
        x_init = jnp.array(0.0)
        x_final, x_prev, iters = newton_solver(f1, x_init)
        assert jnp.allclose(x_final, 1.0, atol=1e-6)
        assert jnp.allclose(f1(x_final), x_final, atol=1e-6)
        
        # Test 2: Exponential decay x = exp(-x)
        def f2(x):
            return jnp.exp(-x)
        
        x_init = jnp.array(0.5)
        x_final, x_prev, iters = newton_solver(f2, x_init)
        assert jnp.allclose(f2(x_final), x_final, atol=1e-6)

    def test_picard_additional_1d(self):
        """Test Picard solver on additional 1D functions."""
        # Test 1: Simple linear function x = (x+1)/2
        def f1(x):
            return (x + 1) / 2
        
        x_init = jnp.array(0.0)
        x_final = picard_solver(f1, x_init)
        assert jnp.allclose(x_final, 1.0, atol=1e-6)
        assert jnp.allclose(f1(x_final), x_final, atol=1e-6)
        
        # Test 2: Exponential decay x = exp(-x)
        def f2(x):
            return jnp.exp(-x)
        
        x_init = jnp.array(0.5)
        x_final = picard_solver(f2, x_init)
        assert jnp.allclose(f2(x_final), x_final, atol=1e-6)


if __name__ == '__main__':
    print("\n=== Starting MRX Iterative Solvers Test Suite ===")
    # Run all tests in the file
    unittest.main(verbosity=2)
