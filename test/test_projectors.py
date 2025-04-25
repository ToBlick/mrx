import unittest
import jax
import jax.numpy as jnp

from mrx.DifferentialForms import DifferentialForm
from mrx.Quadrature import QuadratureRule
from mrx.Projectors import Projector, CurlProjection
from mrx.Utils import jacobian, inv33

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

    def _compute_basis_values(self, Λ, x):
        """Compute all basis function values at point x."""
        return jax.vmap(lambda i: Λ(x, i))(jnp.arange(Λ.n))

    def _compute_l2_error(self, f, f_hat, Λ, Q):
        """Compute L2 error between function and its projection."""
        # Vectorize basis function evaluation over quadrature points
        basis_vals = jax.vmap(lambda x: self._compute_basis_values(Λ, x))(Q.x)
        print(f"basis_vals shape: {basis_vals.shape}")
        print(f"f_hat shape: {f_hat.shape}")
        
        # Compute function values at quadrature points
        f_vals = jax.vmap(f)(Q.x)
        print(f"f_vals shape: {f_vals.shape}")
        
        # Compute Jacobian determinant at quadrature points
        Jj = jax.vmap(jacobian(self.F))(Q.x)  # n_q x 1
        
        # Handle vector-valued functions
        if f_vals.ndim > 1:
            # For vector fields, sum over basis functions first, then over vector components
            proj_vals = jnp.einsum('ijk,j->ik', basis_vals, f_hat)
            # For k-forms (k > 0), we need to transform the basis functions
            if Λ.k == 1:
                # For 1-forms, transform with inverse transpose
                DF = jax.jacfwd(self.F)
                def transform_basis(x, vals):
                    return inv33(DF(x)).T @ vals
                proj_vals = jax.vmap(transform_basis)(Q.x, proj_vals)
            elif Λ.k == 2:
                # For 2-forms, transform with DF
                DF = jax.jacfwd(self.F)
                def transform_basis(x, vals):
                    return DF(x) @ vals
                proj_vals = jax.vmap(transform_basis)(Q.x, proj_vals)
        else:
            # For scalar functions, sum over basis functions and squeeze the last dimension
            proj_vals = jnp.squeeze(jnp.einsum('ijk,j->ik', basis_vals, f_hat), axis=-1)
        
        print(f"proj_vals shape: {proj_vals.shape}")
        
        # Compute squared error and integrate with appropriate Jacobian factor
        error_sq = jnp.sum((f_vals - proj_vals)**2, axis=-1)
        if Λ.k == 0 or Λ.k == 1:
            # For 0-forms and 1-forms, multiply by Jacobian
            return jnp.sqrt(jnp.sum(error_sq * Jj * Q.w))
        elif Λ.k == 2:
            # For 2-forms, divide by Jacobian
            return jnp.sqrt(jnp.sum(error_sq * (1/Jj) * Q.w))
        else:  # k == 3
            # For 3-forms, divide by Jacobian
            return jnp.sqrt(jnp.sum(error_sq * (1/Jj) * Q.w))

    def test_zeroform_projection(self):
        """Test projection of scalar functions (0-forms)."""
        # Create projector
        P = Projector(self.Λ0, self.Q)
        
        # Test function: sin(2πx)sin(2πy) - periodic with period 1
        def f(x):
            return jnp.array([jnp.sin(2 * jnp.pi * x[0]) * jnp.sin(2 * jnp.pi * x[1])])
        
        # Project function
        f_hat = P(f)
        
        # Verify projection has correct shape
        self.assertEqual(f_hat.shape, (self.Λ0.n,))
        
        # Verify no NaN values
        self.assertFalse(jnp.any(jnp.isnan(f_hat)))
        
        # Compute L2 error
        error = self._compute_l2_error(f, f_hat, self.Λ0, self.Q)
        print(f"0-form L2 error: {error}")
        self.assertLess(error, 1e-3)

    def test_oneform_projection(self):
        """Test projection of vector fields (1-forms)."""
        # Create projector
        P = Projector(self.Λ1, self.Q)
        
        # Test function: [sin(2πx)cos(2πy), cos(2πx)sin(2πy), 0] - periodic with period 1
        def A(x):
            return jnp.array([
                jnp.sin(2 * jnp.pi * x[0]) * jnp.cos(2 * jnp.pi * x[1]),
                jnp.cos(2 * jnp.pi * x[0]) * jnp.sin(2 * jnp.pi * x[1]),
                0.0
            ])
        
        # Project function
        A_hat = P(A)
        
        # Verify projection has correct shape
        self.assertEqual(A_hat.shape, (self.Λ1.n,))
        
        # Verify no NaN values
        self.assertFalse(jnp.any(jnp.isnan(A_hat)))
        
        # Compute L2 error
        error = self._compute_l2_error(A, A_hat, self.Λ1, self.Q)
        print(f"1-form L2 error: {error}")
        self.assertLess(error, 1e-3)

    def test_twoform_projection(self):
        """Test projection of 2-forms."""
        # Create projector
        P = Projector(self.Λ2, self.Q)
        
        # Test function: [0, 0, sin(2πx)sin(2πy)] - periodic with period 1
        def B(x):
            return jnp.array([
                0.0,
                0.0,
                jnp.sin(2 * jnp.pi * x[0]) * jnp.sin(2 * jnp.pi * x[1])
            ])
        
        # Project function
        B_hat = P(B)
        
        # Verify projection has correct shape
        self.assertEqual(B_hat.shape, (self.Λ2.n,))
        
        # Verify no NaN values
        self.assertFalse(jnp.any(jnp.isnan(B_hat)))
        
        # Compute L2 error
        error = self._compute_l2_error(B, B_hat, self.Λ2, self.Q)
        print(f"2-form L2 error: {error}")
        self.assertLess(error, 1e-3)

    def test_threeform_projection(self):
        """Test projection of volume forms (3-forms)."""
        # Create projector
        P = Projector(self.Λ3, self.Q)
        
        # Test function: sin(2πx)sin(2πy) - periodic with period 1
        def f(x):
            return jnp.array([jnp.sin(2 * jnp.pi * x[0]) * jnp.sin(2 * jnp.pi * x[1])])
        
        # Project function
        f_hat = P(f)
        
        # Verify projection has correct shape
        self.assertEqual(f_hat.shape, (self.Λ3.n,))
        
        # Verify no NaN values
        self.assertFalse(jnp.any(jnp.isnan(f_hat)))
        
        # Compute L2 error
        error = self._compute_l2_error(f, f_hat, self.Λ3, self.Q)
        print(f"3-form L2 error: {error}")
        self.assertLess(error, 1e-3)

    def test_curl_projection(self):
        """Test curl projection between 1-forms and 2-forms."""
        # Create curl projector
        Pc = CurlProjection(self.Λ1, self.Q)
        
        # Test functions - periodic with period 1
        def A(x):
            return jnp.array([
                jnp.sin(2 * jnp.pi * x[0]) * jnp.cos(2 * jnp.pi * x[1]),
                jnp.cos(2 * jnp.pi * x[0]) * jnp.sin(2 * jnp.pi * x[1]),
                0.0
            ])
        
        def B(x):
            return jnp.array([
                0.0,
                0.0,
                jnp.sin(2 * jnp.pi * x[0]) * jnp.sin(2 * jnp.pi * x[1])
            ])
        
        # Project curl
        curl_hat = Pc(A, B)
        
        # Verify projection has correct shape
        self.assertEqual(curl_hat.shape, (self.Λ1.n,))
        
        # Verify no NaN values
        self.assertFalse(jnp.any(jnp.isnan(curl_hat)))
        
        # Compute L2 error
        def curl(x):
            return jnp.cross(A(x), B(x))
        
        error = self._compute_l2_error(curl, curl_hat, self.Λ1, self.Q)
        print(f"Curl L2 error: {error}")
        self.assertLess(error, 1e-3)

    def test_mapped_projection(self):
        """Test projection with coordinate transformation."""
        # Define a simple mapping - periodic with period 1
        def F(x):
            return jnp.array([
                x[0] + 0.1 * jnp.sin(2 * jnp.pi * x[0]) * jnp.sin(2 * jnp.pi * x[1]),  # Small periodic perturbation
                x[1] + 0.1 * jnp.sin(2 * jnp.pi * x[0]) * jnp.sin(2 * jnp.pi * x[1]),  # Small periodic perturbation
                x[2]        # Keep z unchanged
            ])
        
        # Create projector with mapping
        P = Projector(self.Λ0, self.Q, F)
        
        # Test function in physical coordinates - periodic with period 1
        def f(x):
            return jnp.array([jnp.sin(2 * jnp.pi * x[0]) * jnp.sin(2 * jnp.pi * x[1])])
        
        # Project function
        f_hat = P(f)
        
        # Verify projection has correct shape
        self.assertEqual(f_hat.shape, (self.Λ0.n,))
        
        # Verify no NaN values
        self.assertFalse(jnp.any(jnp.isnan(f_hat)))
        
        # Compute L2 error
        def f_phys(x):
            return f(F(x))
        
        error = self._compute_l2_error(f_phys, f_hat, self.Λ0, self.Q)
        print(f"Mapped L2 error: {error}")
        self.assertLess(error, 1e-3)


if __name__ == '__main__':
    unittest.main() 