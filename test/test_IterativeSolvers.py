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

import os
import time
import unittest

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from mrx.IterativeSolvers import newton_solver, picard_solver

# Create output directory if it doesn't exist
os.makedirs('test_outputs', exist_ok=True)
jax.config.update("jax_enable_x64", True)


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

        z_init = 1.0
        z_star, _, _ = picard_solver(f, z_init, tol=1e-6)

        # The Dottie number is approximately 0.739085
        self.assertAlmostEqual(float(z_star), 0.739085, places=5)
        self.assertAlmostEqual(float(f(z_star)), float(z_star), places=5)

    def test_newton_solver_basic(self):
        """Test Newton solver on a simple scalar function"""
        def f(x):
            return x**2  # Fixed point at x=0,1

        x_init = 1.5
        x_final, _, _ = newton_solver(f, x_init)
        assert jnp.allclose(x_final, 1.0, atol=1e-6)

        # Test function with multiple fixed points
        def f3(x):
            return jnp.sin(x)

        x_init = jnp.array(1.0)
        x_final, _, _ = newton_solver(f3, x_init)
        assert jnp.allclose(f3(x_final), x_final, atol=1e-6)

    def test_newton_multidimensional(self):
        """Test Newton solver on multidimensional problems"""
        # Test 2D polynomial
        def f(x):
            return jnp.array([x[0] ** 2, x[1] ** 2 - 1])

        x_init = jnp.array([0.2, 0.9])
        x_final, _, _ = newton_solver(f, x_init)
        # print(x_final, x_prev, iters)
        assert jnp.allclose(f(x_final), x_final, atol=1e-6)

    def test_newton_high_dimensional_scaling(self):
        """Test Newton solver scaling with high-dimensional linear systems."""
        # Test dimensions to try - now with higher dimensions
        dimensions = jnp.logspace(1, 2, 10, dtype=int)
        max_iterations = []
        solve_times_first = []
        solve_times_second = []
        errors = []

        for n in dimensions:
            print(f"\nTesting dimension {n}...")

            # Generate random positive definite matrix A
            self.rng, key = jax.random.split(self.rng)
            A = jax.random.normal(key, (n, n))
            A = A @ A.T  # Make positive definite
            A = A + n * jnp.eye(n)  # Ensure diagonal dominance

            # Generate random vector b
            self.rng, key = jax.random.split(self.rng)
            b = jax.random.normal(key, (n,))

            # Define linear system as fixed point problem
            def linear_system(x):
                return jnp.linalg.solve(A, b + x)

            # Initial guess -- never start with zeros!
            x_init = jnp.ones(n)

            # First run (includes JIT compilation)
            _newton_solver = jax.jit(newton_solver, static_argnames=['f'])
            start_time = time.time()
            x_final, _, _ = _newton_solver(
                linear_system, x_init, tol=1e-8)
            end_time = time.time()
            time_first = end_time - start_time

            # Second run (after JIT compilation)
            start_time = time.time()
            x_final, err, iters = _newton_solver(
                linear_system, x_init, tol=1e-8)
            end_time = time.time()
            time_second = end_time - start_time

            # Store results
            max_iterations.append(iters)
            solve_times_first.append(time_first)
            solve_times_second.append(time_second)

            # Verify solution
            x_exact = jnp.linalg.solve(A - jnp.eye(n), b)
            error = jnp.linalg.norm(x_final - x_exact) / \
                jnp.linalg.norm(x_exact)
            errors.append(error)

            print(f"  Iterations: {iters}")
            print(f"  First run time: {time_first:.3f} seconds")
            print(f"  Second run time: {time_second:.3f} seconds")
            print(f"  JIT speedup: {time_first/time_second:.2f}x")
            print(f"  Relative error: {error:.2e}")

            # Check solution accuracy
            self.assertLess(
                error, 1e-5, f"Solution accuracy degraded for n={n}")

        # Check scaling behavior
        # The number of iterations should not grow too quickly with dimension
        max_iter_growth = max(max_iterations) / min(max_iterations)
        self.assertLess(max_iter_growth, 5.0,
                        "Number of iterations grew too quickly with dimension")

        # Print scaling summary
        print("\nScaling Summary:")
        print("Dimensions:", dimensions)
        print("Iterations:", max_iterations)
        print("First run times (s):", [f"{t:.3f}" for t in solve_times_first])
        print("Second run times (s):", [
              f"{t:.3f}" for t in solve_times_second])

        # Create plots
        plt.figure(figsize=(20, 5))

        # Plot 1: Iterations vs Dimension
        plt.subplot(141)
        plt.plot(dimensions, max_iterations, 'o-')
        plt.xlabel('Dimension')
        plt.ylabel('Number of Iterations')
        plt.title('Iteration Count vs Dimension')
        plt.grid(True)

        # Plot 2: Solve Time vs Dimension (linear scale)
        plt.subplot(142)
        plt.plot(dimensions, solve_times_first, 'o-', label='First run')
        plt.plot(dimensions, solve_times_second, 'o-', label='Second run')
        plt.xlabel('Dimension')
        plt.ylabel('Solve Time (s)')
        plt.title('Solve Time vs Dimension')
        plt.legend()
        plt.grid(True)

        # Plot 3: Solve Time vs Dimension (log-log scale)
        plt.subplot(143)
        plt.loglog(dimensions, solve_times_first, 'o-', label='First run')
        plt.loglog(dimensions, solve_times_second, 'o-', label='Second run')
        plt.xlabel('Dimension')
        plt.ylabel('Solve Time (s)')
        plt.title('Solve Time vs Dimension (log-log)')
        plt.legend()
        plt.grid(True)

        # Plot 4: Error vs Dimension (log scale)
        plt.subplot(144)
        plt.semilogy(dimensions, errors, 'o-')
        plt.xlabel('Dimension')
        plt.ylabel('Relative Error')
        plt.title('Solution Error vs Dimension')
        plt.grid(True)

        # Save plots
        plt.tight_layout()
        plt.savefig('test_outputs/newton_scaling_plots.png')
        plt.close()

    def test_picard_high_dimensional_scaling(self):
        """Test Picard solver scaling with high-dimensional linear systems."""
        # Test dimensions to try
        dimensions = jnp.logspace(1, 2, 10, dtype=int)
        solve_times_first = []
        solve_times_second = []
        errors = []

        for n in dimensions:
            print(f"\nTesting dimension {n}...")

            # Generate random positive definite matrix A with eigenvalues < 1
            # This ensures Picard iteration converges
            self.rng, key = jax.random.split(self.rng)
            A = jax.random.normal(key, (n, n))
            A = A @ A.T  # Make symmetric
            # Scale to ensure eigenvalues < 1 for convergence
            A = A / (2 * jnp.linalg.norm(A, ord=2))

            # Generate random vector b
            self.rng, key = jax.random.split(self.rng)
            b = jax.random.normal(key, (n,))

            # Define linear system as fixed point problem
            # For a linear system Ax = b
            # Rearrange as x = (I-A)x + b
            def linear_system(x):
                return b + (jnp.eye(n) - A) @ x

            # Initial guess
            x_init = jnp.zeros(n)

            # First run (includes JIT compilation)
            start_time = time.time()
            # Picard takes ages to converge for n >> 1
            _picard_solver = jax.jit(picard_solver, static_argnames=['f'])
            x_final, _, _ = _picard_solver(
                linear_system, x_init, tol=1e-14, max_iter=2000000)
            end_time = time.time()
            time_first = end_time - start_time

            # Second run (after JIT compilation)
            start_time = time.time()
            x_final, _, _ = _picard_solver(
                linear_system, x_init, tol=1e-14, max_iter=2000000)
            end_time = time.time()
            time_second = end_time - start_time

            # Store timing results
            solve_times_first.append(time_first)
            solve_times_second.append(time_second)

            # Verify solution
            x_exact = jnp.linalg.solve(A, b)
            error = jnp.linalg.norm(x_final - x_exact) / \
                jnp.linalg.norm(x_exact)
            errors.append(error)

            print(f"  First run time: {time_first:.3f} seconds")
            print(f"  Second run time: {time_second:.3f} seconds")
            print(f"  JIT speedup: {time_first/time_second:.2f}x")
            print(f"  Relative error: {error:.2e}")

        # Print scaling summary
        print("\nScaling Summary:")
        print("Dimensions:", dimensions)
        print("First run times (s):", [f"{t:.3f}" for t in solve_times_first])
        print("Second run times (s):", [
              f"{t:.3f}" for t in solve_times_second])
        print("Relative errors:", [f"{e:.2e}" for e in errors])

        # Create plots
        plt.figure(figsize=(15, 5))

        # Plot 1: Solve Time vs Dimension (linear scale)
        plt.subplot(131)
        plt.plot(dimensions, solve_times_first, 'o-', label='First run')
        plt.plot(dimensions, solve_times_second, 'o-', label='Second run')
        plt.xlabel('Dimension')
        plt.ylabel('Solve Time (s)')
        plt.title('Solve Time vs Dimension')
        plt.legend()
        plt.grid(True)

        # Plot 2: Solve Time vs Dimension (log-log scale)
        plt.subplot(132)
        plt.loglog(dimensions, solve_times_first, 'o-', label='First run')
        plt.loglog(dimensions, solve_times_second, 'o-', label='Second run')
        plt.xlabel('Dimension')
        plt.ylabel('Solve Time (s)')
        plt.title('Solve Time vs Dimension (log-log)')
        plt.legend()
        plt.grid(True)

        # Plot 3: Error vs Dimension
        plt.subplot(133)
        plt.semilogy(dimensions, errors, 'o-')
        plt.xlabel('Dimension')
        plt.ylabel('Relative Error')
        plt.title('Solution Error vs Dimension')
        plt.grid(True)

        # Save plots
        plt.tight_layout()
        plt.savefig('test_outputs/picard_scaling_plots.png')
        plt.close()


if __name__ == '__main__':
    print("\n=== Starting MRX Iterative Solvers Test Suite ===")
    # Run all tests in the file
    unittest.main(verbosity=2)
    plt.show()
