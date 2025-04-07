import unittest

from mrx.Quadrature import *
from mrx.SplineBases import *
from mrx.DifferentialForms import *
import numpy.testing as npt
from jax import numpy as jnp
from jax import jit, vmap
import quadax as quad
import jax
import numpy as np
jax.config.update("jax_enable_x64", True)


class QuadratureTests(unittest.TestCase):
    
    def test_quad(self):

        # Test periodic quadrature
        Λ0 = DifferentialForm(0,8, 3, 'periodic') 
        Q = QuadratureRule(Λ0,32)
        x_q, w_q = Q.composite_quad((0, 1),32)
        f = lambda x: jnp.sin(x * 20 * jnp.pi)**2
        npt.assert_allclose(jnp.sum(w_q * vmap(f)(x_q)), 0.5, rtol=1e-15)
        
        # Test spectral quadrature
        Λ0 = DifferentialForm(0,8, 3, 'spectral') 
        Q = QuadratureRule(Λ0,32)
        x_q, w_q = Q.spectral_quad(31)
        f = lambda x: jnp.exp(x)
        npt.assert_allclose(jnp.sum(w_q * vmap(f)(x_q)), jnp.exp(1) - 1, rtol=1e-15)
        
        # Test grid case
        types = ('clamped,''periodic', 'periodic')
        ns = (8,8,1)
        ps = (3,3,0)
        Λ0 = DifferentialForm(0,ns, ps, types)
        x_q, w_q = quadrature_grid(Q.composite_quad((0,1),31),
                                   Q.composite_quad((0, 2*jnp.pi),16),
                                    Q.composite_quad((0, 2*jnp.pi),16))     
        f = lambda x: x[0] * jnp.exp(x[0]) * jnp.sin(x[1])**2 * jnp.cos(x[2])**2
        npt.assert_allclose(jnp.sum(w_q * vmap(f)(x_q)), jnp.pi**2, rtol=1e-15)