"""
Unit tests for testing the Projectors in the MRX package.

For now, we are testing the CurlProjection and GradientProjection operators.

The tests verify:
1. Projection values are finite
2. Projection is not too close to zero (unless it's supposed to)
3. Discrete function values are finite
"""

import unittest

import jax
import jax.numpy as jnp

from mrx.DifferentialForms import DifferentialForm, DiscreteFunction
from mrx.Projectors import Projector, CurlProjection, GradientProjection
from mrx.Quadrature import QuadratureRule

        

class TestProjectorTypes(unittest.TestCase):
    """Test different types of projectors (0-form, 1-form, 2-form, 3-form)."""
    
    def setUp(self):
        """Set up test parameters."""
        # Set up finite element spaces
        self.n = 4 # Number of elements in each direction
        self.p = 3  # Polynomial degree
        self.ns = (self.n, self.n, self.n)
        self.ps = (self.p, self.p, self.p)
        self.types = ('periodic', 'periodic', 'periodic')
        
        # Create differential forms for each degree
        self.Λ0 = DifferentialForm(0, self.ns, self.ps, self.types)  # 0-forms
        self.Λ1 = DifferentialForm(1, self.ns, self.ps, self.types)  # 1-forms 
        self.Λ2 = DifferentialForm(2, self.ns, self.ps, self.types)  # 2-forms
        self.Λ3 = DifferentialForm(3, self.ns, self.ps, self.types)  # 3-forms 
        
        # Create quadrature rule
        self.Q = QuadratureRule(self.Λ0, 5)
        
        # Create projectors for each form
        self.P0 = Projector(self.Λ0, self.Q)
        self.P1 = Projector(self.Λ1, self.Q)
        self.P2 = Projector(self.Λ2, self.Q)
        self.P3 = Projector(self.Λ3, self.Q)
    

    def test_curl_projection(self):
        """Test the curl projection operator."""
        # Define vector fields for testing curl

        # Define one form field A
        def A(x):
            r, χ, z = x
            return jnp.array([
                jnp.cos(2*jnp.pi*r),  # dr component
                jnp.cos(2*jnp.pi*χ),  # dχ component
                jnp.cos(2*jnp.pi*z)   # dz component
            ])
        
        # Define two form field B
        def B(x):
            r, χ, z = x
            return jnp.array([
                jnp.cos(2*jnp.pi*r) + jnp.sin(2*jnp.pi*z),  # dr ∧ dχ component
                jnp.sin(2*jnp.pi*χ) + jnp.cos(2*jnp.pi*r),  # dχ ∧ dz component
                jnp.cos(2*jnp.pi*z) + jnp.sin(2*jnp.pi*χ)   # dz ∧ dr component
            ])
        
        # Create curl projector
        curl_proj = CurlProjection(self.Λ1, self.Q)
        
        # Project the curl
        curl_coeffs = curl_proj(A, B)
        
        # Test result is finite
        self.assertTrue(jnp.all(jnp.isfinite(curl_coeffs)), 
            "Curl projection produced non-finite values")
        
        # Test that the projection is not too close to zero
        self.assertGreater(jnp.linalg.norm(curl_coeffs), 1e-10,
            "Curl projection is too close to zero")
        
        # Create discrete function from coefficients
        q_h = DiscreteFunction(curl_coeffs, self.Λ1)

        # Generate random test point
        key = jax.random.PRNGKey(0)
        key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
            
        # Generate coordinates between 0 and 1 to make it easier
        r = jax.random.uniform(subkey1)
        χ = jax.random.uniform(subkey2)
        z = jax.random.uniform(subkey3)

        x_test = jnp.array([r, χ, z])
        
        # Test discrete function values are finite
        self.assertTrue(jnp.all(jnp.isfinite(q_h(x_test))),
            "Discrete function produced non-finite values at x = {x_test}")


    def test_gradient_projection(self):
        """Test the gradient projection operator."""
        # Define test functions
        def p(x):
            """Scalar function for testing."""
            r, χ, z = x
            return jnp.ones(1) * jnp.sin(2*jnp.pi*r) * jnp.cos(2*jnp.pi*χ) * jnp.exp(z**3)
        
        def u(x):
            """Vector field (2-form) for testing."""
            r, χ, z = x
            # Same from before
            return jnp.array([
                jnp.cos(2*jnp.pi*r) + jnp.sin(2*jnp.pi*z),  # dr ∧ dχ component
                jnp.sin(2*jnp.pi*χ) +jnp.cos(2*jnp.pi*r),  # dχ ∧ dz component
                jnp.cos(2*jnp.pi*z) + jnp.sin(2*jnp.pi*χ)   # dz ∧ dr component
            ])
        
        # Test different values of gamma
        gamma_values = [4/3, 5/3, 2.0]
        
        for gamma in gamma_values:
            print(f"\nTesting GradientProjection with γ = {gamma}")
            
            # Create gradient projector
            grad_proj = GradientProjection(self.Λ0, self.Q, Ɣ=gamma)
            
            # Project the gradient
            grad_coeffs = grad_proj(p, u)
            
            # Test result is finite
            self.assertTrue(jnp.all(jnp.isfinite(grad_coeffs)), 
                f"Gradient projection produced non-finite values for γ = {gamma}")
            
            # Test projection is not too close to zero
            self.assertGreater(jnp.linalg.norm(grad_coeffs), 1e-10,
                f"Gradient projection is too close to zero for γ = {gamma}")
            
            # Create discrete function from coefficients
            p_h = DiscreteFunction(grad_coeffs, self.Λ0)
            
            # Generate random test point
            key = jax.random.PRNGKey(0)
            key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
            
            # Generate coordinates between 0 and 1 to make it easier
            r = jax.random.uniform(subkey1)
            χ = jax.random.uniform(subkey2)
            z = jax.random.uniform(subkey3)
            
            # Create test point 
            x_test = jnp.array([r, χ, z])
            
            # Test discrete function evaluation
            val = p_h(x_test)
            self.assertTrue(jnp.all(jnp.isfinite(val)),
                f"Discrete function produced non-finite values at x = {x_test} for γ = {gamma}")
            
          


if __name__ == "__main__":
    unittest.main()


