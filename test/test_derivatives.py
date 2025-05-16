import unittest
import jax
from jax import numpy as jnp
from jax import jit,grad,vmap
from mrx.Utils import pullback_1form, grad, div, Jacobian
from mrx.coordinate_transforms import *
from mrx.DifferentialForms import *


jax.config.update("jax_enable_x64", True)


class DerivativesTests(unittest.TestCase):
    def test(self):

        F = cyl_to_cart
        G = cart_to_cyl

        n = 32
        key = jax.random.PRNGKey(0)
        _x_hat = jax.random.uniform(key, (n**3, 3), 
                    minval=jnp.array((0.0, 0.0, 0.0)),
                    maxval=jnp.array((1.0, 2*jnp.pi, 1.0)))
        x = vmap(F)(_x_hat)

        #  Converted x from cylindrical to cartesian coordinates
        # Check if the shape of x is correct
        self.assertEqual(x.shape, (n**3, 3))
    

# From 2014 Variational Integration paper (Zhou), Page 5
        def p_hat(x):
            # x is in cylindrical coordinates
            r,  phi, z = x
            return 0.3  + 8*(jnp.pi**2)*(0.05*(jnp.cos(4*jnp.pi*r*jnp.cos(phi))-jnp.cos(4*jnp.pi*r*jnp.sin(phi))))**2
        
        # Pullback p_hat
        p = pullback_0form(p_hat, G)
        #T ake gradient of pullback
        grad_hat_p_hat = pullback_1form(grad(p), F)
        # Take gradient of p_hat and pull back
        grad_p = pullback_1form(grad(p_hat), G)


        # Check they are equal
        npt.assert_allclose(vmap(grad_hat_p_hat)(_x_hat), vmap(grad(p_hat))(_x_hat), atol=1e-12)
        npt.assert_allclose(vmap(grad_p)(x), vmap(grad(p))(x), atol=1e-12)
        

    