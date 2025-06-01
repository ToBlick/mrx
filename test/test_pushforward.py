"""
Unit tests for the pushforwards implemented in Differential Forms.

The tests include:
- Pushforward of a scalar is a scalar
- Linearity
- Pushforward of a 1-form is implemented correctly
"""

import unittest
import jax
from jax import numpy as jnp
import numpy as np
from mrx.DifferentialForms import Pushforward
from mrx.Utils import inv33

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
        return jnp.ones(1) * jnp.sqrt((x_1**2) * (x_2**2))
def _χ(x_1, x_2):
        """Compute the χ coordinate."""
        return jnp.ones(1) * jnp.arctan2(x_2, x_1) / (2 * jnp.pi)

def G(x):
        """Cartesian to cylindrical mapping function."""
        x_1, x_2, x_3 = x
        return jnp.ravel(jnp.array([_R(x_1, x_2), _χ(x_1, x_2), jnp.ones(1) * x_3]))

class PushforwardTests(unittest.TestCase):
    def test_pushforward_scalar(self):
        """Test that the pushforward of a scalar is a scalar."""
        ζ_star = jnp.array([1.0, 0.5, 1.0]) # Test cylindrical point

        # Define a 0-form in cartesian coordinates
        def scalar(x):
            x_1, x_2, x_3 = x
            return x_3 * (x_1 - x_2)
        
        # Pushforward the 0-form
        pushforward_scalar = Pushforward(scalar, F, 0)

        self.assertTrue(
            jnp.isscalar(pushforward_scalar(ζ_star)),
            "The pushforward of a scalar should be a scalar"
        )


    def test_pushforward_linearity_zero(self):
        """Test that the pushforward of a 0-form is linear."""
        ζ_star = jnp.array([1.0, 0.5, 1.0]) # Test point
       
        # Define two scalars in Cartesian coordinates
        def f(x):
            x_1, x_2, x_3 = x
            return x_3 - x_1*x_2
        def g(x):
            x_1, x_2, x_3 = x
            return x_1 + x_2 + x_3

        # Define a linear combination of f and g
        def combo(x):
            return -2*f(x) + 5*g(x)   
        
        # Pushforward the linear combination
        pushforward_combo = Pushforward(combo, F, 0)
        
        # Pushforward f and g
        pushforward_f = Pushforward(f, F, 0)
        pushforward_g = Pushforward(g, F, 0)

        # Check linearity
        np.testing.assert_allclose(pushforward_combo(ζ_star),-2*pushforward_f(ζ_star) + 5*pushforward_g(ζ_star), atol=1e-6, err_msg="Pushforward of a zero form must be linear"
        )

    def test_pushforward_linearity_one(self):
        """Test that the pushforward of a 1-form is linear."""
        
        # Test point
        ζ_star = jnp.array([1.0, 0.5, 1.0])  # Cylindrical
        
        # Define two 1-forms in Cartesian coordinates
        def A_cart(x):
            x_1, x_2, x_3 = x
            return jnp.array([x_1-x_2, x_2*x_1, x_3]) 
            
        def B_cart(x):
            x_1, x_2, x_3 = x
            return jnp.array([x_3-x_2, x_1*x_3, x_2**2]) 
        
        # Define a linear combination of A and B
        def combo(x):
            return -2*A_cart(x) + 5*B_cart(x)
        
        # Pushforward the linear combination
        pushforward_combo = Pushforward(combo, F, 1)

        # Pushforward A and B
        pushforward_A = Pushforward(A_cart, F, 1)
        pushforward_B = Pushforward(B_cart, F, 1) 
        
        # Check linearity
        np.testing.assert_allclose(pushforward_combo(ζ_star),-2*pushforward_A(ζ_star) + 5*pushforward_B(ζ_star), atol=1e-6, err_msg="Pushforward of a one form must be linear"
        )
   
    def test_pushforward_1form(self):
        """Test that the pushforward of a 1-form is implemented correctly."""
        # Test cylindrical point
        ζ_star = jnp.array([1.0, 0.2, 0.0])

        # Define a 1-form in cartesian coordinates
        def A_cart(x):
            x_1, x_2, x_3 = x
            return jnp.array([x_1+x_2, x_3-x_1, x_2])
        
        # Pushforward A
        pushforward_A = Pushforward(A_cart, F, 1)

        # Evaluate the pushforward at ζ_star
        A_cyl_star = pushforward_A(ζ_star)

        # The pushforward of A at ζ_star should match the original form evaluated at the cartesian point, multiplied on the left by inv(DF.T)
        # Get the Jacobian of F at ζ_star (evaluation must be in logical domain)
        DF = jax.jacfwd(F)(ζ_star)

        # Original form evaluated at cartesian point
        A_orig = A_cart(ζ_star)

        # Expected result after pushforward (DF^{-T} @ A)
        expected_pushforward = inv33(DF.T)@A_orig

        # Verify the transformation
        np.testing.assert_allclose(
            A_cyl_star,
            expected_pushforward,
            atol=1e-6,
            err_msg="Pushforward of 1-form  was not implemented correctly"
        )

if __name__ == '__main__':
    unittest.main()