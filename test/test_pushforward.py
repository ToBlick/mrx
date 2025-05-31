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
from mrx.DifferentialForms import Pushforward

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
    def test_pushforward(self):
        # Test that the pushforward of a scalar is a scalar
        ζ_star = jnp.array([1.0, 0.5, 1.0])

        # Test 0-form pushforward
        def scalar(x):
            x_1, x_2, x_3 = x
            return x_3 * (x_1 - x_2)
        
        pushforward_scalar = Pushforward(scalar, F, 0)

        self.assertTrue(
            jnp.isscalar(pushforward_scalar(ζ_star)),
            "The pushforward of a scalar should be a scalar"
        )

    def test_pushforward_linearity(self):
        #Test that the pushforward map is linear
        
        # Test point
        ζ_star = jnp.array([1.0, 0.5, 1.0])  # Cylindrical
        
        # Define two 1-forms in Cartesian coordinates
        def A_cart(x):
            x_1, x_2, x_3 = x
            return jnp.array([x_1-x_2, x_2*x_1, x_3]) 
            
        def B_cart(x):
            x_1, x_2, x_3 = x
            return jnp.array([x_3-x_2, x_1*x_3, x_2**2]) 
        
        # Pushforward A and B
        pushforward_A = Pushforward(A_cart, F, 1)
        pushforward_B = Pushforward(B_cart, F, 1)

        # Verify that the sum of pushforwards is pushforward of the sum

        def sum_AB(x):
            return A_cart(x) + B_cart(x)
        
        pushforward_sum = Pushforward(sum_AB, F, 1)
        sum_star = pushforward_sum(ζ_star)
        
        # Add pushforwards separately 
        separate_star = pushforward_A(ζ_star) + pushforward_B(ζ_star)
        
        # Verify additivity
        np.testing.assert_allclose(sum_star,separate_star, atol=1e-6, err_msg="Pushforward isn't additive"
        )
        
        # Test scalar multiplication

        key = jax.random.PRNGKey(1) #1 as seed
        C = jax.random.uniform(key, minval=-100.0, maxval=100.0) # Random number between -100 and 100

        # Define scaled A
        def scaled_A(x):
            return C * A_cart(x)
        pushforward_A_scaled = Pushforward(scaled_A, F, 1)
        scaled_A_result = pushforward_A_scaled(ζ_star)
        separate_A_scaled = C * pushforward_A(ζ_star)
        
        # Verify scalar multiplication
        np.testing.assert_allclose(scaled_A_result,separate_A_scaled, atol=1e-6, err_msg="Pushforward isn't scalar multiplicative"
        )


if __name__ == '__main__':
    unittest.main()