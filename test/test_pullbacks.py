import unittest
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


class PullbackTests(unittest.TestCase):
    
    def test_pullbacks(self):
        # Test pullback of a scalar is a scalar
        x_cart = jnp.array([1.0, 2.0, 3.0])
        
        # Define a 0-form
        def scalar(ζ):
            r, phi, z = ζ
            return r**3 + z*phi
        
        pullback_scalar = Pullback(scalar, G, 0)
        self.assertTrue(
            jnp.isscalar(pullback_scalar(x_cart)),
            "The pullback of a scalar should be a scalar"
        )
        
    def test_pullback_properties_1(self):
        # Test points
        x_star = jnp.array([1.0, 2.0, 3.0])  # Cartesian point
        ζ_star = G(x_star)  # Cylindrical point
        
        # Define a 1-form in cylindrical coordinates
        def A_cyl(ζ):
            r, phi, z = ζ
            return jnp.array([r * jnp.cos(phi), r * jnp.sin(phi), z])
            
        # Test we can pullback from cylindrical to cartesian, and then pushforward back to cylindrical
        # First pullback from cylindrical to Cartesian
        pullback_A = Pullback(A_cyl, G, 1)
        A_cart = pullback_A(x_star)
        
        # Then pushforward back to cylindrical
        pushforward_A_back = Pushforward(lambda x: pullback_A(x), F, 1)
        A_cyl_return =  pushforward_A_back(ζ_star)
        
        # Check we get back the same thing
        np.testing.assert_allclose(A_cyl_return, A_cyl(ζ_star), atol=1e-7)

    def test_pullback_properties_2(self):

        # Test points
        ζ_star = jnp.array([1.0, jnp.pi, 1.0])  # Cylindrical point
        x_star = F(ζ_star)  # Cartesian point

         
        # Define a 1-form in cartesian coordinates
        def A_cart(x):
            x_1,x_2,x_3 = x
            return jnp.array([x_1, x_2, x_3])
        

        # Test we can pushforward from cartesian to cylindrical, and then pullback back to cartesian
        # First pushforward from cartesian to cylindrical
        pushforward_A = Pushforward(A_cart, F, 1)
        A_cyl =  pushforward_A(x_star)
        
        # Then pullback back to cartesian
        pullback_A_back = Pullback(lambda ζ:  pushforward_A(ζ), G, 1)
        A_cart_return =   pullback_A_back(x_star)
        
        # Check we get back the same thing
        np.testing.assert_allclose( A_cart_return, A_cart(x_star), atol=1e-7)

    def test_pullback_exterior(self):
        
        # Recall that we can define the gradient in terms of an exterior derivative; for a 0-form
        # the gradient is the same as the exterior derivative. Now, we also know that the exterior derivative 
        # should commute with the pullback map, so I test that: 

        # Re-use scalar function from before

        def scalar(ζ):
            r, phi, z = ζ
            return r**3 + z*phi
        
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


if __name__ == '__main__':
    unittest.main()