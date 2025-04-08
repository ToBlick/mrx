import unittest

from mrx.Quadrature import *
from mrx.SplineBases import *
from mrx.DifferentialForms import *
import numpy.testing as npt
from jax import numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)

class QuadratureTests(unittest.TestCase):
    
    def test_quad(self):

        # Test 1D quadrature
        ns = (8, 1, 1)
        ps = (3, 0, 0)
        types = ('periodic', 'constant', 'constant')
        Λ0 = DifferentialForm(0, ns, ps, types)
        for i in range(3, 11):
            Q = QuadratureRule(Λ0, i)
            f = lambda x: jnp.sin(x[0] * 2 * jnp.pi)**2 * jnp.ones(1)
            print(i)
            print(Q.w @ jax.vmap(f)(Q.x))
            npt.assert_allclose(Q.w @ jax.vmap(f)(Q.x), 0.5, rtol=1e-7)
        
        # # Test spectral quadrature
        # Λ0 = DifferentialForm(0, (8, 1, 1), (3, 1, 1), ('fourier', 'constant', 'constant')) 
        # Q = QuadratureRule(Λ0, 32)
        # x_q, w_q = Q.spectral_quad(31)
        # f = lambda x: jnp.exp(x)
        # npt.assert_allclose(jnp.sum(w_q * vmap(f)(x_q)), jnp.exp(1) - 1, rtol=1e-15)
        
        # Test grid case
        types = ('clamped', 'periodic', 'periodic')
        ns = (8, 8, 8)
        ps = (3, 3, 3)
        for i in range(3, 11):
            Λ0 = DifferentialForm(0, ns, ps, types)
            Q = QuadratureRule(Λ0, i)  
            f = lambda x: x[0] * jnp.exp(x[0]) * jnp.sin(x[1] * 2 * jnp.pi)**2 * jnp.cos(x[2] * 2 * jnp.pi)**2 * jnp.ones(1)
            print(i)
            print(Q.w @ jax.vmap(f)(Q.x))
            npt.assert_allclose(Q.w @ jax.vmap(f)(Q.x), 1/4, rtol=1e-7)