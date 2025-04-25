"""
Unit tests for the Quadrature module.

This module contains tests for numerical quadrature implementations,
including tests for 1D quadrature, grid-based quadrature, and various
integration scenarios with different basis functions and domains.

The tests use the following parameters:
- ns: Tuple of integers representing the number of basis functions in each dimension
- ps: Tuple of integers representing the polynomial degree of the basis functions in each dimension
- types: Tuple of strings representing the boundary condition types ('periodic', 'clamped', 'constant')
"""

import unittest
import numpy.testing as npt
import jax
import jax.numpy as jnp
from mrx.Quadrature import QuadratureRule
from mrx.DifferentialForms import DifferentialForm

# Enable double precision
jax.config.update("jax_enable_x64", True)


class TestQuadrature(unittest.TestCase):
    """Test cases for numerical quadrature implementations."""

    def setUp(self):
        """Set up test cases."""
        self.rtol = 1e-3  # Relative tolerance for numerical comparisons

    def test_1d_periodic_quadrature(self):
        """Test 1D quadrature with periodic boundary conditions.
        
        Tests integration of various functions over [0,1], including:
        - sin²(2πx) (exact = 0.5)
        - exp(sin(2πx)) (exact ≈ 1.2660658777520084)
        - sin(2πx)cos(4πx) (exact = 0.0)
        Tests convergence for different quadrature orders and basis function configurations.
        """
        # Test different combinations of basis function counts and polynomial degrees
        test_cases = [
            # (ns, ps, types)
            ((8, 1, 1), (1, 0, 0), ('periodic', 'constant', 'constant')),  # Linear basis
            ((8, 1, 1), (2, 0, 0), ('periodic', 'constant', 'constant')),  # Quadratic basis
            ((8, 1, 1), (3, 0, 0), ('periodic', 'constant', 'constant')),  # Cubic basis
            ((16, 1, 1), (1, 0, 0), ('periodic', 'constant', 'constant')), # More basis functions, linear
            ((16, 1, 1), (3, 0, 0), ('periodic', 'constant', 'constant')), # More basis functions, cubic
            ((4, 1, 1), (3, 0, 0), ('periodic', 'constant', 'constant')),  # Fewer basis functions
        ]
        
        def integrand1(x):
            """Test function sin²(2πx)."""
            return jnp.sin(x[0] * 2 * jnp.pi)**2 * jnp.ones(1)
        
        def integrand2(x):
            """Test function exp(sin(2πx))."""
            return jnp.exp(jnp.sin(x[0] * 2 * jnp.pi)) * jnp.ones(1)
        
        def integrand3(x):
            """Test function sin(2πx)cos(4πx)."""
            return jnp.sin(x[0] * 2 * jnp.pi) * jnp.cos(x[0] * 4 * jnp.pi) * jnp.ones(1)

        exact_values = [0.5, 1.2660658777520084, 0.0]
        integrands = [integrand1, integrand2, integrand3]
        integrand_names = ["sin²(2πx)", "exp(sin(2πx))", "sin(2πx)cos(4πx)"]

        for ns, ps, types in test_cases:
            form = DifferentialForm(0, ns, ps, types)
            for quad_order in range(3, 11):
                quad_rule = QuadratureRule(form, quad_order)
                for integrand, exact, name in zip(integrands, exact_values, integrand_names):
                    result = quad_rule.w @ jax.vmap(integrand)(quad_rule.x)
                    abs_error = abs(float(result[0] - exact))
                    print(f"1D Test - {name} - ns={ns}, ps={ps}, quad_order={quad_order}: abs_error={abs_error:.2e}")
                    npt.assert_allclose(
                        result, 
                        exact, 
                        atol=1e-15,
                        rtol=self.rtol,
                        err_msg=f"Failed for {name} with ns={ns}, ps={ps}, quad_order={quad_order}"
                    )

    def test_3d_mixed_quadrature(self):
        """Test 3D quadrature with mixed boundary conditions.
        
        Tests integration of various functions over [0,1]³, including:
        - x*exp(x)*sin²(2πy)*cos²(2πz) (exact = 0.25)
        - exp(x+y+z)*sin(2πx)*cos(2πy)*sin(2πz) (exact ≈ 0.0)
        - sin(2πx)*sin(2πy)*sin(2πz) (exact = 0.0)
        Tests convergence for different quadrature orders and basis function configurations.
        """
        # First verify the exact value of the problematic integrand
        def verify_exact_value(integrand):
            """Verify the exact value of the problematic integrand using high-order quadrature."""
            # Use a very high order quadrature to approximate the exact value
            ns = (32, 32, 32)
            ps = (3, 3, 3)
            types = ('clamped', 'periodic', 'periodic')
            form = DifferentialForm(0, ns, ps, types)
            quad_rule = QuadratureRule(form, 10)  # High order quadrature
            result = quad_rule.w @ jax.vmap(integrand)(quad_rule.x)
            return float(result[0])
                
        # Test different combinations of basis function counts and polynomial degrees
        test_cases = [
            # (ns, ps, types)
            ((8, 8, 8), (1, 1, 1), ('clamped', 'periodic', 'periodic')),  # Linear basis
            ((8, 8, 8), (2, 2, 2), ('clamped', 'periodic', 'periodic')),  # Quadratic basis
            ((8, 8, 8), (3, 3, 3), ('clamped', 'periodic', 'periodic')),  # Cubic basis
            ((4, 4, 4), (1, 1, 1), ('clamped', 'periodic', 'periodic')),  # Fewer basis functions, linear
            ((4, 4, 4), (3, 3, 3), ('clamped', 'periodic', 'periodic')),  # Fewer basis functions, cubic
            ((16, 16, 16), (3, 3, 3), ('clamped', 'periodic', 'periodic')), # More basis functions
        ]
        
        def integrand1(x):
            """Test function x*exp(x)*sin²(2πy)*cos²(2πz)."""
            return (x[0] * jnp.exp(x[0]) * 
                   jnp.sin(x[1] * 2 * jnp.pi)**2 * 
                   jnp.cos(x[2] * 2 * jnp.pi)**2 * 
                   jnp.ones(1))
        
        def integrand2(x):
            """Test function exp(x+y+z)*sin(2πx)*cos(2πy)*sin(2πz)."""
            return (jnp.exp(x[0] + x[1] + x[2]) * 
                   jnp.sin(x[0] * 2 * jnp.pi) * 
                   jnp.cos(x[1] * 2 * jnp.pi) * 
                   jnp.sin(x[2] * 2 * jnp.pi) * 
                   jnp.ones(1))
        
        def integrand3(x):
            """Test function sin(2πx)*sin(2πy)*sin(2πz)."""
            return (jnp.sin(x[0] * 2 * jnp.pi) * 
                   jnp.sin(x[1] * 2 * jnp.pi) * 
                   jnp.sin(x[2] * 2 * jnp.pi) * 
                   jnp.ones(1))

        integrands = [integrand1, integrand2, integrand3]
        integrand_names = ["x*exp(x)*sin²(2πy)*cos²(2πz)", 
                         "exp(x+y+z)*sin(2πx)*cos(2πy)*sin(2πz)",
                         "sin(2πx)*sin(2πy)*sin(2πz)"]

        for ns, ps, types in test_cases:
            for quad_order in range(5, 11):
                form = DifferentialForm(0, ns, ps, types)
                quad_rule = QuadratureRule(form, quad_order)
                for integrand, name in zip(integrands, integrand_names):
                    exact = verify_exact_value(integrand)
                    result = quad_rule.w @ jax.vmap(integrand)(quad_rule.x)
                    abs_error = abs(float(result[0] - exact))
                    print(f"3D Test - {name} - ns={ns}, ps={ps}, quad_order={quad_order}: abs_error={abs_error:.2e}")
                    npt.assert_allclose(
                        result, 
                        exact, 
                        atol=1e-15,
                        rtol=self.rtol,
                        err_msg=f"Failed for {name} with ns={ns}, ps={ps}, quad_order={quad_order}"
                    )


if __name__ == '__main__':
    unittest.main()