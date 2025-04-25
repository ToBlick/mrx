import unittest

import jax
import jax.numpy as jnp

from mrx.DifferentialForms import DifferentialForm
from mrx.Projectors import CurlProjection, Projector
from mrx.Quadrature import QuadratureRule
from mrx.Utils import inv33, jacobian

jax.config.update("jax_enable_x64", True)


class TestProjectors(unittest.TestCase):
    """Test cases for Projector and CurlProjection classes."""

    def setUp(self):
        """Set up test fixtures."""
        # Define test parameters
        self.ns = (8, 8, 1)  # Number of points in each direction
        self.ps = (3, 3, 0)  # Polynomial degrees
        self.types = ('periodic', 'periodic', 'constant')  # Boundary types

        # Create differential forms
        self.Λ0 = DifferentialForm(0, self.ns, self.ps, self.types)  # 0-forms
        self.Λ1 = DifferentialForm(1, self.ns, self.ps, self.types)  # 1-forms
        self.Λ2 = DifferentialForm(2, self.ns, self.ps, self.types)  # 2-forms
        self.Λ3 = DifferentialForm(3, self.ns, self.ps, self.types)  # 3-forms

        # Create quadrature rule
        self.Q = QuadratureRule(self.Λ0, 5)  # Quadrature order 5

        # Identity mapping
        self.F = lambda x: x


if __name__ == '__main__':
    unittest.main()
