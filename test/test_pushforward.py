import unittest
import random
import jax
from jax import numpy as jnp
import numpy as np
from jax import jit,grad,vmap
from mrx.Utils import grad as Grad
from mrx.Utils import curl as Curl
from mrx.coordinate_transforms import *
from mrx.DifferentialForms import *
from mrx.Quadrature import *


F = cyl_to_cart
G = cart_to_cyl


class PushforwardTests(unittest.TestCase):
    def test_pushforward(self):
        # Test that the pushforward of a scalar is a scalar
        ζ_star = jnp.array([1.0, jnp.pi, 1.0])

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
        ζ_star = jnp.array([1.0, jnp.pi, 1.0])  # Cylindrical
        
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

        C = random.random()

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