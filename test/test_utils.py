import unittest
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import time
import matplotlib.pyplot as plt
import os
from mrx.Utils import jacobian, inv33, div, curl, grad, l2_product

jax.config.update("jax_enable_x64", True)

# Create output directory if it doesn't exist
os.makedirs('test_outputs', exist_ok=True)

class TestUtils(unittest.TestCase):
    """Test cases for utility functions in Utils.py."""

    def setUp(self):
        """Set up test fixtures."""
        # Define test functions for differential operators
        def f_scalar(x):
            return jnp.sum(x**2)  # Scalar field: f(x) = x₁² + x₂² + x₃²
        
        def f_vector(x):
            return jnp.array([x[1]*x[2], x[0]*x[2], x[0]*x[1]])  # Vector field: F(x) = [x₂x₃, x₁x₃, x₁x₂]
        
        self.f_scalar = f_scalar
        self.f_vector = f_vector
        
        # Define test points
        self.x = jnp.array([0.5, 0.5, 0.5])
        self.x2 = jnp.array([0.25, 0.75, 0.5])
        
        # Define test matrices
        self.A = jnp.array([
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 4.0],
            [5.0, 6.0, 1.0]
        ])
        
        # Near-singular matrix
        self.B = jnp.array([
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0]
        ])
        
        # Orthogonal matrix
        self.C = jnp.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0]
        ])

    def test_jacobian(self):
        """Test the jacobian function."""
        # Test with identity function
        def identity(x):
            return x
        J = jacobian(identity)(self.x)
        npt.assert_allclose(J, 1.0)
        
        # Test with linear function
        def linear(x):
            return 2 * x
        J = jacobian(linear)(self.x)
        npt.assert_allclose(J, 8.0)
        
        # Test with vector field
        J = jacobian(self.f_vector)(self.x)
        npt.assert_allclose(J, 0.25)
        
        # Test with zero function
        def zero(x):
            return jnp.zeros_like(x)
        J = jacobian(zero)(self.x)
        npt.assert_allclose(J, 0.0)
        
        # Test with constant function
        def constant(x):
            return jnp.ones_like(x)
        J = jacobian(constant)(self.x)
        npt.assert_allclose(J, 0.0)

    def test_inv33(self):
        """Test the 3x3 matrix inverse function."""
        # Test with identity matrix
        identity_matrix = jnp.eye(3)
        npt.assert_allclose(inv33(identity_matrix), identity_matrix, rtol=1e-15, atol=1e-15)
        
        # Test with a general 3x3 matrix
        A_inv = inv33(self.A)
        # Verify that A * A_inv ≈ I
        npt.assert_allclose(jnp.dot(self.A, A_inv), jnp.eye(3), rtol=1e-14, atol=1e-14)
        
        # Test with a singular matrix
        singular = jnp.ones((3, 3))
        # The function should return a matrix of zeros for singular input
        A_inv = inv33(singular)
        npt.assert_allclose(A_inv, jnp.zeros((3, 3)), rtol=1e-15, atol=1e-15)
        
        # Test with near-singular matrix
        B_inv = inv33(self.B)
        npt.assert_allclose(B_inv, jnp.zeros((3, 3)), rtol=1e-15, atol=1e-15)
        
        # Test with orthogonal matrix
        C_inv = inv33(self.C)
        npt.assert_allclose(jnp.dot(self.C, C_inv), jnp.eye(3), rtol=1e-14, atol=1e-14)
        
        # Test with random matrices
        rng = jax.random.PRNGKey(0)
        for _ in range(10):
            mat = jax.random.normal(rng, (3, 3))
            mat_inv = inv33(mat)
            if jnp.abs(jnp.linalg.det(mat)) > 1e-10:
                npt.assert_allclose(jnp.dot(mat, mat_inv), jnp.eye(3), rtol=1e-14, atol=1e-14)

    def test_div(self):
        """Test the divergence operator."""
        # Compute divergence of test vector field
        div_F = div(self.f_vector)(self.x)
        npt.assert_allclose(div_F, 0.0, atol=1e-10)
        
        # Test with constant vector field
        def constant(x):
            return jnp.array([1.0, 2.0, 3.0])
        div_const = div(constant)(self.x)
        npt.assert_allclose(div_const, 0.0)
        
        # Test with linear vector field
        def linear(x):
            return x
        div_linear = div(linear)(self.x)
        npt.assert_allclose(div_linear, 3.0)
        
        # Test with quadratic vector field
        def quadratic(x):
            return x**2
        div_quad = div(quadratic)(self.x)
        npt.assert_allclose(div_quad, 2.0 * jnp.sum(self.x))

    def test_curl(self):
        """Test the curl operator."""
        # Compute curl of test vector field
        curl_F = curl(self.f_vector)(self.x)
        npt.assert_allclose(curl_F, jnp.zeros(3), atol=1e-10)
        
        # Test with a non-zero curl field
        def rot_field(x):
            return jnp.array([-x[1], x[0], 0.0])
        curl_rot = curl(rot_field)(self.x)
        expected = jnp.array([0.0, 0.0, 2.0])
        npt.assert_allclose(curl_rot, expected)
        
        # Test with constant vector field
        def constant(x):
            return jnp.array([1.0, 2.0, 3.0])
        curl_const = curl(constant)(self.x)
        npt.assert_allclose(curl_const, jnp.zeros(3))
        
        # Test with linear vector field
        def linear(x):
            return x
        curl_linear = curl(linear)(self.x)
        npt.assert_allclose(curl_linear, jnp.zeros(3))

    def test_grad(self):
        """Test the gradient operator."""
        # Compute gradient of test scalar field
        grad_f = grad(self.f_scalar)(self.x)
        expected = 2 * self.x
        npt.assert_allclose(grad_f, expected)
        
        # Test with linear function
        def linear(x):
            return jnp.sum(x)
        grad_linear = grad(linear)(self.x)
        npt.assert_allclose(grad_linear, jnp.ones(3))
        
        # Test with constant function
        def constant(x):
            return jnp.ones(1)
        grad_const = grad(constant)(self.x)
        npt.assert_allclose(grad_const, jnp.zeros(3))
        
        # Test with exponential function
        def exp_func(x):
            return jnp.exp(jnp.sum(x))
        grad_exp = grad(exp_func)(self.x)
        expected = jnp.exp(jnp.sum(self.x)) * jnp.ones(3)
        npt.assert_allclose(grad_exp, expected)

    def test_l2_product(self):
        """Test the L2 inner product computation."""
        # Create a simple quadrature rule for testing
        class TestQuadrature:
            def __init__(self):
                self.x = jnp.array([[0.25, 0.25, 0.25],
                                  [0.75, 0.75, 0.75]])
                self.w = jnp.array([0.5, 0.5])
        
        Q = TestQuadrature()
        
        # Test with constant functions
        def const_f(x):
            return jnp.array([1.0])
        def const_g(x):
            return jnp.array([2.0])
        product = l2_product(const_f, const_g, Q)
        npt.assert_allclose(product, 2.0)
        
        # Test with identity mapping
        def F(x):
            return x
        product = l2_product(const_f, const_g, Q, F)
        npt.assert_allclose(product, 2.0)
        
        # Test with non-constant functions
        def f(x):
            return jnp.array([jnp.sum(x)])
        def g(x):
            return jnp.array([jnp.prod(x)])
        product = l2_product(f, g, Q)
        self.assertTrue(jnp.isfinite(product))
        
        # Test with zero functions
        def zero(x):
            return jnp.array([0.0])
        product = l2_product(zero, f, Q)
        npt.assert_allclose(product, 0.0)

    def test_performance(self):
        """Test and plot performance of utility functions with increasing problem sizes."""
        # Define problem sizes to test
        n_points_list = np.logspace(2, 8, num=10, dtype=int)
        
        # Initialize timing arrays
        inv_times = []
        grad_times = []
        curl_times = []
        div_times = []
        inv_times_jit = []
        grad_times_jit = []
        curl_times_jit = []
        div_times_jit = []
        
        # Create figure for plotting
        plt.figure(figsize=(12, 8))
        
        # Define colors for each operation
        colors = {
            'inv': '#1f77b4',  # blue
            'grad': '#ff7f0e',  # orange
            'curl': '#2ca02c',  # green
            'div': '#d62728'    # red
        }
        
        for n_points in n_points_list:
            print(f"\nTesting with {n_points} points...")
            
            # Generate random data
            rng = jax.random.PRNGKey(0)
            points = jax.random.normal(rng, (n_points, 3))
            matrices = jax.random.normal(rng, (n_points, 3, 3))
            
            # Create vectorized functions
            vmap_inv33 = jax.vmap(inv33)
            vmap_grad = jax.vmap(grad(self.f_scalar))
            vmap_curl = jax.vmap(curl(self.f_vector))
            vmap_div = jax.vmap(div(self.f_vector))
            
            # Non-JIT timing
            start = time.time()
            _ = vmap_inv33(matrices)
            inv_time = time.time() - start
            inv_times.append(inv_time)
            
            start = time.time()
            _ = vmap_grad(points)
            grad_time = time.time() - start
            grad_times.append(grad_time)
            
            start = time.time()
            _ = vmap_curl(points)
            curl_time = time.time() - start
            curl_times.append(curl_time)
            
            start = time.time()
            _ = vmap_div(points)
            div_time = time.time() - start
            div_times.append(div_time)
            
            # JIT timing
            jit_inv33 = jax.jit(vmap_inv33)
            jit_grad = jax.jit(vmap_grad)
            jit_curl = jax.jit(vmap_curl)
            jit_div = jax.jit(vmap_div)
            
            # Warm up JIT
            _ = jit_inv33(matrices)
            _ = jit_grad(points)
            _ = jit_curl(points)
            _ = jit_div(points)
            
            start = time.time()
            _ = jit_inv33(matrices)
            inv_time_jit = time.time() - start
            inv_times_jit.append(inv_time_jit)
            
            start = time.time()
            _ = jit_grad(points)
            grad_time_jit = time.time() - start
            grad_times_jit.append(grad_time_jit)
            
            start = time.time()
            _ = jit_curl(points)
            curl_time_jit = time.time() - start
            curl_times_jit.append(curl_time_jit)
            
            start = time.time()
            _ = jit_div(points)
            div_time_jit = time.time() - start
            div_times_jit.append(div_time_jit)
            
            print(f"Non-JIT times: inv={inv_time:.4f}, grad={grad_time:.4f}, curl={curl_time:.4f}, div={div_time:.4f}")
            print(f"JIT times: inv={inv_time_jit:.4f}, grad={grad_time_jit:.4f}, curl={curl_time_jit:.4f}, div={div_time_jit:.4f}")
        
        # Plot results with consistent colors
        plt.loglog(n_points_list, inv_times, 'o-', color=colors['inv'], label='inv33 (non-JIT)')
        plt.loglog(n_points_list, grad_times, 'o-', color=colors['grad'], label='grad (non-JIT)')
        plt.loglog(n_points_list, curl_times, 'o-', color=colors['curl'], label='curl (non-JIT)')
        plt.loglog(n_points_list, div_times, 'o-', color=colors['div'], label='div (non-JIT)')
        
        plt.loglog(n_points_list, inv_times_jit, 's--', color=colors['inv'], label='inv33 (JIT)')
        plt.loglog(n_points_list, grad_times_jit, 's--', color=colors['grad'], label='grad (JIT)')
        plt.loglog(n_points_list, curl_times_jit, 's--', color=colors['curl'], label='curl (JIT)')
        plt.loglog(n_points_list, div_times_jit, 's--', color=colors['div'], label='div (JIT)')
        
        plt.xlabel('Number of points')
        plt.ylabel('Computation time (seconds)')
        plt.title('Performance Comparison: JIT vs Non-JIT')
        plt.grid(True, which="both", ls="-")
        plt.legend()
        
        # Save the plot
        plt.savefig('output_scripts/performance_comparison.png')
        plt.show()
        
        print("\nPerformance plot saved as 'output_scripts/performance_comparison.png'")

if __name__ == '__main__':
    unittest.main() 
