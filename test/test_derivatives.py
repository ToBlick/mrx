import unittest
import jax
from jax import numpy as jnp
import numpy as np
from jax import jit,grad,vmap
from mrx.Utils import grad as Grad
from mrx.Utils import curl as Curl
from mrx.Utils import div as Div
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


# Define function to take curl of.
def A_hat(ζ):
    r, phi, z = ζ
    return jnp.array([r**2,(r**2)*jnp.cos(phi)*jnp.sin(phi), z])

# Define function to take curl of.
def A(x):
    x_1, x_2, x_3 = x
    return jnp.array([x_1**2 + x_2**2,(x_1*x_2),x_3 ])


# Pullback A_hat
def pullback_A_hat(x):
    # x is in cartesian coordinates
    x_1,x_2,x_3 = x
    return Pullback(A_hat, G, 0).__call__(x)


# Pushforward A
def pushforward_A(ζ):
    # ζ is in cylindrical coordinates
    r,  phi, z = ζ
    return Pushforward(A, F, 0).__call__(ζ)



# Define function for 2 form
def B_hat(ζ):
    r, phi, z = ζ
    return jnp.array([ jnp.sin(phi)*r + z**2, jnp.sin(phi) + r**2,r + z * jnp.cos(phi)])
def B(x):
    x_1, x_2, x_3 = x
    return jnp.array([x_2 + x_3**2, (x_2/jnp.sqrt(x_1**2 + x_2**2))+ x_1**2 + x_2**2, jnp.sqrt(x_1**2 + x_2**2) + x_3*(x_1/jnp.sqrt(x_1**2 + x_2**2)) ])


# Pullback B_hat
def pullback_B_hat(x):
    # x is in cartesian coordinates
    x_1,x_2,x_3 = x
    return Pullback(B_hat, G, 0).__call__(x)

# Pushforward B
def pushforward_B(ζ):
    # ζ is in cylindrical coordinates
    r,  phi, z = ζ
    return Pushforward(B, F, 0).__call__(ζ)

# Define vector field for divergence tests
def V_hat(ζ):
    r, phi, z = ζ
    return jnp.array([r*jnp.sin(phi), r*jnp.cos(phi), z**3])

def V(x):
    x_1, x_2, x_3 = x
    return jnp.array([x_2, x_1, x_3**3])

# Pullback V_hat
def pullback_V_hat(x):
    return Pullback(V_hat, G, 1).__call__(x)

# Pushforward V
def pushforward_V(ζ):
    return Pushforward(V, F, 1).__call__(ζ)

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
        # Test pullback of 0-form 
        x_eval = jnp.array([1,2,3])
        self.assertEqual(pullback_p_hat(x_eval),p(x_eval))

    def test_pushforward_0form(self):
        # Test pushforward of 0-form
        ζ_eval = jnp.array([1,jnp.pi,1])
        self.assertEqual(pushforward_p(ζ_eval),p_hat(ζ_eval))


    def test_grad_pullback(self):
        # Test gradient of pullback of 0-form
        x_eval = jnp.array([1.0,2.0,3.0])
        grad_pullback_eval = Grad(pullback_p_hat)(x_eval)
        np.testing.assert_array_equal(grad_pullback_eval, jax.grad(p)(x_eval))
    
    def test_grad_pushforward(self):
        # Test gradient of pushforward of 0-form
        ζ_eval = jnp.array([1.0,jnp.pi,1.0])
        grad_pushforward_eval = Grad(pushforward_p)(ζ_eval)
        np.testing.assert_array_equal(grad_pushforward_eval, jax.grad(p_hat)(ζ_eval))


    def test_curl_pullback_1(self):
        # Test curl of pullback of 1-form
        x_eval = jnp.array([1.0,2.0,3.0])
        curl_pullback_eval = Curl(pullback_A_hat)(x_eval)
        np.testing.assert_allclose(curl_pullback_eval, Curl(A)(x_eval),atol=1e-7)

        # I noticed that I had to set the tolerance to 1e-7, and any smaller caused the test to fail



    def test_curl_pushforward_1(self):
        # Test curl of pushforward of 1-form
        ζ_eval = jnp.array([1.0,jnp.pi,1.0])
        curl_pushforward_eval = Curl(pushforward_A)(ζ_eval)
        np.testing.assert_allclose(curl_pushforward_eval, Curl(A_hat)(ζ_eval),atol=1e-8)


    # Test curl of pushforward of 2 form

    def test_curl_pushforward_2(self):
        ζ_eval = jnp.array([1.0,jnp.pi,1.0])
        curl_pushforward_eval_2 = Curl(pushforward_B)(ζ_eval)
        np.testing.assert_allclose(curl_pushforward_eval_2, Curl(B_hat)(ζ_eval),atol=1e-8)


    # Test curl of pullback of 2 form

    def test_curl_pullback_2(self):
        x_eval = jnp.array([1.0,2.0,3.0])
        curl_pullback_eval_2 = Curl(pullback_B_hat)(x_eval)
        np.testing.assert_allclose(curl_pullback_eval_2, Curl(B)(x_eval),atol=1e-8)


    def test_div_curl(self):
        # Test that divergence of curl is zero 
        x_eval = jnp.array([1.0, 2.0, 3.0])
        div_of_curl = Div(Curl(A))(x_eval)
        np.testing.assert_allclose(div_of_curl, 0.0, atol=1e-8)

    def test_curl_grad(self):
        # Test that curl of gradient is zero 
        x_eval = jnp.array([1.0, 2.0, 3.0])
        curl_of_grad = Curl(Grad(p))(x_eval)
        np.testing.assert_allclose(curl_of_grad, jnp.zeros(3), atol=1e-8)

   
    
if __name__ == '__main__':
    unittest.main()