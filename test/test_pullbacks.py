"""
Unit tests for the pullbacks implemented in Differential Forms.

The tests include:
- Pullback of a scalar is a scalar
- Commutativity of pullback and exterior derivative for a 0-form and 1-form
- Pullback of a 1-form is done correctly
- Linearity
"""

import unittest
import jax
from jax import numpy as jnp
import numpy as np
from mrx.DifferentialForms import Pullback
from mrx.Utils import curl as Curl

# Mapping from cylindrical coordinates to Cartesian coordinates

def _X(r, χ):
        """Compute the X coordinate."""
        return jnp.ones(1) *r *jnp.cos(2 * jnp.pi * χ)

def _Y(r, χ):
        """Compute the Y coordinate."""
        return jnp.ones(1) *  r * jnp.sin(2 * jnp.pi * χ)

def F(x):
        """Cylindrical to cartesian mapping function."""
        r, χ, z = x
        return jnp.ravel(jnp.array([_X(r, χ), _Y(r, χ), jnp.ones(1) * z]))


# Mapping from Cartesian coordinates to cylindrical coordinates
def _R(x_1, x_2):
        """Compute the R coordinate."""
        return jnp.ones(1) * jnp.sqrt((x_1**2) + (x_2**2))
def _χ(x_1, x_2):
        """Compute the χ coordinate."""
        return jnp.ones(1) * jnp.arctan2(x_2, x_1) / (2 * jnp.pi)

def G(x):
        """Cartesian to cylindrical mapping function."""
        x_1, x_2, x_3 = x
        return jnp.ravel(jnp.array([_R(x_1, x_2), _χ(x_1, x_2), jnp.ones(1) * x_3]))



class PullbackTests(unittest.TestCase):
    
    def test_scalar_pullbacks(self):
        """Test that the pullback of a scalar is a scalar."""
        # Define cartesian test point
        x_cart = jnp.array([1.0, 2.0, 3.0])
        
        # Define a 0-form
        def scalar(ζ):
            r, χ, z = ζ
            return r**3 + z*χ

        # Pull it back
        pullback_scalar = Pullback(scalar, G, 0)
        self.assertTrue(
            jnp.isscalar(pullback_scalar(x_cart)),
            "The pullback of a scalar should be a scalar"
        )

    def test_pullback_exterior_zero(self):
        """Test that the pullback commutes with the exterior derivative for a 0-form."""
       
        # Recall that we can define the gradient in terms of an exterior derivative; for a 0-form
        # the gradient is the same as the exterior derivative. Now, we also know that the exterior derivative
        # should commute with the pullback map, so I test that.
        

        # Re-use scalar function from before

        def scalar(ζ):
            r, χ, z = ζ
            return r**3 + z*χ
        
        x_star = jnp.array([1.0, 2.0, 3.0])  # Cartesian test point
    
        # Compute exterior derivative in cylindrical coordinates (recall that this returns a 1-form)
        grad_scalar = jax.grad(scalar)
        
        # Pullback the function and its gradient
        pullback_s = Pullback(scalar, G, 0)
        pullback_grad_s = Pullback(grad_scalar, G, 1)
        
        # Compute gradient of pullback
        grad_pullback_s = jax.grad(lambda x: pullback_s(x))
        
        # Verify commutativity of pullback and exterior derivative
        np.testing.assert_allclose(
            grad_pullback_s(x_star),
            pullback_grad_s(x_star),
            atol=1e-7,
            err_msg="Pullback has to commute with exterior derivative for a zero form"
        )


    def test_pullback_exterior_one(self):
        """Test that the pullback commutes with the exterior derivative for a 1-form, the curl."""
        
        # Define a 1-form in cylindrical coordinates
        def A_hat(ζ):
            r, χ, z = ζ
            return jnp.array([r**2, χ*r, z])
        
        # Test point in cartesian coordinates
        x_star = jnp.array([1.0, 2.0, 3.0])
        
        # Pullback A_hat
        pullback_A_hat = Pullback(A_hat, G, 1)

        # Test that curl commutes with pullback
        curl_pullback = Curl(pullback_A_hat)(x_star)
        pullback_curl = Pullback(Curl(A_hat), G, 2).__call__(x_star)
        np.testing.assert_allclose(
            curl_pullback,
            pullback_curl,
            atol=1e-5,
            err_msg="Curl should commute with pullback"
        )


    def test_pullback_linearity_zero(self):
        """Test linearity of pullbacks for 0-forms."""
        x_star = jnp.array([1.0, 2.0, 3.0])
        # Define two 0-forms
        def f(ζ):
            r, χ, z = ζ
            return r*χ
        def g(ζ):
            r, χ, z = ζ
            return χ + r

        # Make a linear combination of f and g:

        def combo(ζ):
            r, χ, z = ζ
            return  f(ζ) -  g(ζ)

        # Pullback the combo
        pullback_combo = Pullback(combo, G, 0)

        # Pullback the individual forms
        pullback_f = Pullback(f, G, 0)
        pullback_g = Pullback(g, G, 0)

        # Check linearity
        np.testing.assert_allclose(
            pullback_combo(x_star),
            pullback_f(x_star) - pullback_g(x_star),
            atol=1e-8,
            err_msg="Pullback needs to be linear for zero forms"
        )

    def test_pullback_linearity_one(self):
        """Test linearity of pullbacks for 1-forms."""
        x_star = jnp.array([1.0, 2.0, 3.0])
        # Define two 1-forms

        def A(ζ):
            r, χ, z = ζ
            return jnp.array([r*χ, χ + z, z])
        def B(ζ):
            r, χ, z = ζ
            return jnp.array([r + χ, z - r, χ])

        # Make a linear combination of A and B
        def combo(ζ):
            r, χ, z = ζ
            return A(ζ) + 2*B(ζ)
        # Pullback the combo
        pullback_combo = Pullback(combo, G, 1)

        # Pullback the individual forms
        pullback_A = Pullback(A, G, 1)
        pullback_B = Pullback(B, G, 1)
        # Check linearity
        np.testing.assert_allclose(
            pullback_combo(x_star),
            pullback_A(x_star) + 2*pullback_B(x_star),
            atol=1e-7,
            err_msg="Pullback needs to be linear for 1-forms"
        )


    def test_pullback_1form(self):
        """Test implementation of pullback of a 1-form from cylindrical to Cartesian coordinates."""
        # Test point in Cartesian coordinates
        x_star = jnp.array([1.0, 1.0, 1.0])  
        
        # Define a  1-form in cylindrical coordinates
        def A_cyl(ζ):
            r, χ, z = ζ
            return jnp.array([r**2, χ*r, z])
        
        # Pullback A
        pullback_A = Pullback(A_cyl, G, 1)
        
        # Evaluate the pullback at x_star
        A_cart_star = pullback_A(x_star)
        
        # The pullback of A at x_star should match the original form evaluated at the transformed point, multiplied on the left by DG
        ζ_star = G(x_star)  # Transform to cylindrical
        
        # Get the Jacobian of G at x_star
        DG = jax.jacfwd(G)(x_star)
        
        # Original form evaluated at cylindrical point
        A_orig = A_cyl(ζ_star)
        
        # Expected result after pullback (DG^T @ A)
        expected_pullback = DG.T @ A_orig
        
        # Verify the transformation
        np.testing.assert_allclose(
            A_cart_star,
            expected_pullback,
            atol=1e-7,
            err_msg="Pullback of 1-form did not transform as expected"
        )
    

if __name__ == '__main__':
    unittest.main()
