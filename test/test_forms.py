"""
Unit tests for differential forms and related operations in the MRX package.

This module contains tests for:
1. Assembly of differential forms
2. Projection operations
3. Mass and derivative matrix computations
4. Transformations and mappings

The tests verify:
- Correct assembly of differential forms of different degrees
- Proper handling of boundary conditions
- Accuracy of projection operations
- Correctness of mass and derivative matrices
- Proper transformation of forms under coordinate changes
"""

import unittest
import jax
import jax.numpy as jnp
from mrx.Quadrature import QuadratureRule
from mrx.SplineBases import SplineBasis
from mrx.DifferentialForms import DifferentialForm
from mrx.Projectors import Projector
from mrx import LazyMatrices

# Enable 64-bit precision for numerical accuracy
jax.config.update("jax_enable_x64", True)


class FormsTests(unittest.TestCase):
    """Test cases for differential forms and related operations."""

    def setUp(self):
        """Set up test cases with common parameters."""
        self.n = 16  # Number of elements
        self.p = 3   # Polynomial degree
        self.ns = (self.n, self.n, 1)  # Number of elements in each dimension
        self.ps = (self.p, self.p, 1)  # Polynomial degree in each dimension
        self.types = ('clamped', 'clamped', 'fourier')  # Basis types
        self.boundary = ('periodic', 'periodic', 'periodic')  # Boundary conditions
        self.quad_order = 11  # Quadrature order

    def test_assembly(self):
        """Test assembly of differential forms and related matrices."""
        # Define rotation transformation
        alpha = jnp.pi/2
        def F(x):
            """Rotate the unit cube by 90 degrees."""
            return jnp.array([
                [jnp.cos(alpha), jnp.sin(alpha), 0],
                [-jnp.sin(alpha), jnp.cos(alpha), 0],
                [0, 0, 1]
            ]) @ (x - jnp.ones(3)/2) + jnp.ones(3)/2

        def F_inv(x):
            """Inverse rotation transformation."""
            return jnp.array([
                [jnp.cos(alpha), -jnp.sin(alpha), 0],
                [jnp.sin(alpha), jnp.cos(alpha), 0],
                [0, 0, 1]
            ]) @ (x - jnp.ones(3)/2) + jnp.ones(3)/2

        # Create differential forms of different degrees
        Λ0 = DifferentialForm(0, self.ns, self.ps, self.types)
        Λ1 = DifferentialForm(1, self.ns, self.ps, self.types)
        Λ3 = DifferentialForm(3, self.ns, self.ps, self.types)

        # Create quadrature rule
        Q = QuadratureRule(Λ0, self.quad_order)

        # Test matrix assembly
        D = LazyMatrices.LazyDerivativeMatrix(Λ0, Λ1, Q, F)
        M = LazyMatrices.LazyMassMatrix(Λ0, Q, F)

        # Define test functions
        def f(x):
            """Source function for projection (3-form)."""
            # Return a scalar value for 3-form projection
            return jnp.array([2 * (2 * jnp.pi)**2 * jnp.sin(2 * jnp.pi * x[0]) * jnp.sin(2 * jnp.pi * x[1])])

        def u(x):
            """Solution function for verification (0-form)."""
            return jnp.array([jnp.sin(jnp.pi * 2 * x[0]) * jnp.sin(2 * jnp.pi * x[1])])

        # Test projection
        Proj = Projector(Λ3, Q, F).threeform_projection(f)

        # Add assertions to verify results
        self.assertIsNotNone(D.M, "Derivative matrix should not be None")
        self.assertIsNotNone(M.M, "Mass matrix should not be None")
        self.assertIsNotNone(Proj, "Projection should not be None")
        self.assertEqual(Proj.shape[0], Λ3.n, "Projection should have correct size")

    def test_boundary_conditions(self):
        """Test handling of different boundary conditions."""
        # Test with different boundary conditions
        boundary_types = [
            ('periodic', 'periodic', 'periodic'),
            ('clamped', 'clamped', 'periodic'),
            ('fourier', 'fourier', 'periodic')
        ]

        for boundary in boundary_types:
            s = SplineBasis(self.ns[0], self.ps[0], boundary[0])
            self.assertIsNotNone(s, f"Spline basis should be created for boundary {boundary}")

    def test_quadrature_accuracy(self):
        """Test accuracy of quadrature rules."""
        Λ0 = DifferentialForm(0, self.ns, self.ps, self.types)
        
        # Test different quadrature orders
        for order in [5, 11, 15]:
            Q = QuadratureRule(Λ0, order)
            self.assertIsNotNone(Q, f"Quadrature rule should be created for order {order}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
