import unittest
import jax
from jax import numpy as jnp
import numpy as np
from jax import jit,grad,vmap
from mrx.Utils import *
from mrx.coordinate_transforms import *
from mrx.DifferentialForms import *
from mrx.Quadrature import *




# From 2014 Variational Integration paper (Zhou), Page 5
def p_hat(ζ):
    # ζ is in cylindrical coordinates
    r,  phi, z = ζ
    return 0.3  + 8*(jnp.pi**2)*(0.05*(jnp.cos(4*jnp.pi*r*jnp.cos(phi))-jnp.cos(4*jnp.pi*r*jnp.sin(phi))))**2
        

def p(x):
    #x is in cartesian coordinates
    x_1,  x_2, x_3 = x
    return 0.3  + 8*(jnp.pi**2)*(0.05*(jnp.cos(4*jnp.pi*x_1)-jnp.cos(4*jnp.pi*x_2)))**2


F = cyl_to_cart
G = cart_to_cyl


# Pullback p_hat   
def pullback_p_hat(x):
    # ζ is in cylindrical coordinates
    x_1,x_2,x_3 = x
    return Pullback(p_hat, G, 0).__call__(x)

# Pushforward p
def pushforward_p(ζ):
    # ζ is in cylindrical coordinates
    r,  phi, z = ζ
    return Pushforward(p, F, 0).__call__(ζ)

# Take gradient of pullback
def grad_pullback(h):
    return jax.grad(h)

# Take gradient of pushforward
def grad_pushforward(h):
    return jax.grad(h)
# Unit test class



class TestDerivatives(unittest.TestCase):

    def test_cyn_to_car(self):
        # Test conversion from cylindrical to cartesian
        ζ_eval = jnp.array([1,jnp.pi,1])
        x_eval = F(ζ_eval)
        self.assertEqual(p(x_eval), p_hat(ζ_eval))

    def test_car_to_cyn(self):

        # Test conversion from cartesian to cylindrical
        x_eval = jnp.array([1,2,3])
        ζ_eval = G(x_eval)
        self.assertEqual(p(x_eval), p_hat(ζ_eval))

    def test_pullback_0form(self):
        # Test pullback 
        x_eval = jnp.array([1,2,3])
        self.assertEqual(pullback_p_hat(x_eval),p(x_eval))

    def test_pushforward_0form(self):
        # Test pushforward
        ζ_eval = jnp.array([1,jnp.pi,1])
        self.assertEqual(pushforward_p(ζ_eval),p_hat(ζ_eval))


    def test_grad_pullback(self):
        # Test gradient of pullback
        x_eval = jnp.array([1.0,2.0,3.0])
        grad_pullback_eval = grad(pullback_p_hat)(x_eval)
        np.testing.assert_array_equal(grad_pullback_eval, jax.grad(p)(x_eval))
    
    def test_grad_pushforward(self):
        # Test gradient of pushforward
        ζ_eval = jnp.array([1.0,jnp.pi,1.0])
        grad_pushforward_eval = grad(pushforward_p)(ζ_eval)
        np.testing.assert_array_equal(grad_pushforward_eval, jax.grad(p_hat)(ζ_eval))

 
if __name__ == '__main__':
    unittest.main()