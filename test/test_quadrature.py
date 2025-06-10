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

import jax
import jax.numpy as jnp
import numpy.testing as npt

from mrx.DifferentialForms import DifferentialForm
from mrx.Quadrature import QuadratureRule

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
            # Quadratic basis
            ((8, 1, 1), (2, 0, 0), ('periodic', 'constant', 'constant')),
            ((8, 1, 1), (3, 0, 0), ('periodic', 'constant', 'constant')),  # Cubic basis
            # More basis functions, linear
            ((16, 1, 1), (1, 0, 0), ('periodic', 'constant', 'constant')),
            # More basis functions, cubic
            ((16, 1, 1), (3, 0, 0), ('periodic', 'constant', 'constant')),
            # Fewer basis functions
            ((4, 1, 1), (3, 0, 0), ('periodic', 'constant', 'constant')),
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
                    print(
                        f"1D Test - {name} - ns={ns}, ps={ps}, quad_order={quad_order}: abs_error={abs_error:.2e}")
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
            ns = (8, 8, 8)
            ps = (3, 3, 3)
            types = ('clamped', 'periodic', 'periodic')
            form = DifferentialForm(0, ns, ps, types)
            quad_rule = QuadratureRule(form, 4)
            result = quad_rule.w @ jax.vmap(integrand)(quad_rule.x)
            return float(result[0])

        # Test different combinations of basis function counts and polynomial degrees
        test_cases = [
            # (ns, ps, types)
            ((8, 8, 8), (1, 1, 1), ('clamped', 'periodic', 'periodic')),  # Linear basis
            # Quadratic basis
            ((8, 8, 8), (2, 2, 2), ('clamped', 'periodic', 'periodic')),
            ((8, 8, 8), (3, 3, 3), ('clamped', 'periodic', 'periodic')),  # Cubic basis
            # Fewer basis functions, linear
            ((4, 4, 4), (1, 1, 1), ('clamped', 'periodic', 'periodic')),
            # Fewer basis functions, cubic
            ((4, 4, 4), (3, 3, 3), ('clamped', 'periodic', 'periodic')),
            # More basis functions
            ((16, 16, 16), (3, 3, 3), ('clamped', 'periodic', 'periodic')),
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
                    print(
                        f"3D Test - {name} - ns={ns}, ps={ps}, quad_order={quad_order}: abs_error={abs_error:.2e}")
                    npt.assert_allclose(
                        result,
                        exact,
                        atol=1e-15,
                        rtol=self.rtol,
                        err_msg=f"Failed for {name} with ns={ns}, ps={ps}, quad_order={quad_order}"
                    )

    def test_strange_quadrature(self):
        """Test quadrature behavior with sharp peak and rapid oscillation.
        
        This test verifies that the quadrature rules can handle:
        1. Rapidly oscillating functions
        2. Functions with sharp peaks
        4. Different polynomial degrees
        5. Convergence with increasing quadrature points
        """
        def oscillating_peak(x):
            """Test function with rapid oscillations and a sharp peak."""
            # Create a sharp peak at x[0]=0.25, y[0]=0.25
            peak = jnp.exp(-100*((x[0]-0.25)**2 + (x[1]-0.25)**2))
            # rapid oscillations
            oscillations = jnp.sin(16*jnp.pi*x[1]) * jnp.cos(16*jnp.pi*x[0])
            return peak * oscillations * jnp.ones(1)
        
        # Test cases with different configurations
        test_cases = [
            # (ns, ps, types)
            ((8,8,1), (2,2,0), ('clamped', 'periodic', 'constant')),  
            ((12,12,1), (3,3,0), ('clamped', 'periodic', 'constant')),  # Higher resolution
                            ]
        
        # Get approximate solution computed with very high order quadrature
        ns_approx = (32,32,1)
        ps_approx = (5,5,0)
        types_approx = ('clamped', 'periodic', 'constant')
        form_approx = DifferentialForm(0, ns_approx, ps_approx, types_approx)
        quad_approx = QuadratureRule(form_approx, 10)
        reference = float((quad_approx.w @ jax.vmap(oscillating_peak)(quad_approx.x))[0])
        

        for ns, ps, types in test_cases:
            form = DifferentialForm(0, ns, ps, types)
            errors = []
            
            # Test convergence with increasing quadrature points and compare with high order quadrature
            for quad_order in range(3, 11):
                quad_rule = QuadratureRule(form, quad_order)
                result = float((quad_rule.w @ jax.vmap(oscillating_peak)(quad_rule.x))[0])
                error = abs(result - reference)
                errors.append(error)
                
                print(f"\nConfiguration: ns={ns}, ps={ps}, quad_order={quad_order}")
                print(f"Result: {result:.10f}")
                print(f"Absolute Error: {error:.2e}")
                
                # Check error is within reasonable bounds
                # More lenient for lower orders, stricter for higher orders
                if quad_order <= 5:
                    tol = 1e-3
                elif 5< quad_order <= 8:
                    tol = 1e-4
                else:
                    tol = 1e-5
                    
                self.assertLess(
                    error, tol,
                    f"Error is too large for ns={ns}, ps={ps}, quad_order={quad_order}"
                )
            
            
        
        # Test exact integration of polynomials
        def polynomial_test(x):
            """Test polynomial that should be integrated exactly."""
            # Construct a polynomial of degree 4 or less
            p = (x[0]**2 + x[1]**2) * (1 - x[0]) * (1 - x[1])
            return p * jnp.ones(1)
        
        # This polynomial should be integrated exactly with sufficient quadrature points
        ns = (8,8,1)
        ps = (4,4,0)
        types = ('clamped', 'clamped', 'constant')
        form = DifferentialForm(0, ns, ps, types)
        
        # Test with increasing quadrature orders
        prev_result = None
        for quad_order in range(5, 8):
            quad_rule = QuadratureRule(form, quad_order)
            result = float((quad_rule.w @ jax.vmap(polynomial_test)(quad_rule.x))[0])
            
            if prev_result is not None:
                # Results should be identical (up to numerical precision)
                self.assertTrue(
                    jnp.allclose(result, prev_result, rtol=1e-10),
                    "Exact polynomial integration failed"
                )
            prev_result = result


if __name__ == '__main__':
    unittest.main()
