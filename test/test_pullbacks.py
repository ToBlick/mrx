"""
Unit tests for the pullbacks implemented in Differential Forms.

The tests include:
- Pullback of a scalar is a scalar
- Linearity
- Commutativity of pullback and exerior derivative
"""


import unittest
import jax
from jax import numpy as jnp
import numpy as np
from mrx.DifferentialForms import Pullback

# Mappibng from cylindrical coordinates to Cartesian coordinates

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
        return jnp.ones(1) * jnp.sqrt((x_1**2) * (x_2**2))
def _χ(x_1, x_2):
        """Compute the χ coordinate."""
        return jnp.ones(1) * jnp.arctan2(x_2, x_1) / (2 * jnp.pi)

def G(x):
        """Cartesian to cylindrical mapping function."""
        x_1, x_2, x_3 = x
        return jnp.ravel(jnp.array([_R(x_1, x_2), _χ(x_1, x_2), jnp.ones(1) * x_3]))



class PullbackTests(unittest.TestCase):
    
    def test_pullbacks(self):
        # Test pullback of a scalar is a scalar
        x_cart = jnp.array([1.0, 2.0, 3.0])
        
        # Define a 0-form
        def scalar(ζ):
            r, χ, z = ζ
            return r**3 + z*χ
        
        pullback_scalar = Pullback(scalar, G, 0)
        self.assertTrue(
            jnp.isscalar(pullback_scalar(x_cart)),
            "The pullback of a scalar should be a scalar"
        )

    def test_pullback_exterior(self):
        
        # Recall that we can define the gradient in terms of an exterior derivative; for a 0-form
        # the gradient is the same as the exterior derivative. Now, we also know that the exterior derivative
        # should commute with the pullback map, so I test that:

        # Re-use scalar function from before

        def scalar(ζ):
            r, χ, z = ζ
            return r**3 + z*χ
        
        x_star = jnp.array([1.0, 2.0, 3.0])  # Cartesian point
    
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
            err_msg="Pullback has to commute with exterior derivative"
        )

    def test_pullback_linearity(self):
        # Test linearity of pullback
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
            err_msg="Pullback needs to be linear"
        )

if __name__ == '__main__':
    unittest.main()
