import unittest
import jax
from jax import numpy as jnp
import numpy as np
from mrx.Utils import grad as Grad
from mrx.Utils import curl as Curl
from mrx.SplineBases import SplineBasis
from mrx.coordinate_transforms import *
from mrx.DifferentialForms import *
from mrx.Quadrature import *
from mrx.LazyMatrices import *
import numpy.testing as npt
from jax import vmap, jit, grad, hessian, jacfwd, jacrev

jax.config.update("jax_enable_x64", True)


import matplotlib.pyplot as plt
import time

class AssemblyTests(unittest.TestCase):

# 1D projection
   #def test_assembly(self):

        # Map to physical domain: unit cube to [-3,3]^3
        def F(x):
            return 6*x - jnp.ones_like(x)*3

        def F_inv(x):    
            return (x + jnp.ones_like(x)*3)/6
        
        # Bases
 
        n=2
        p=1
        ns = (2,2,1)
        ps = (1,1,0)
        types = ("clamped", "periodic", "constant")
        # bcs = ('dirichlet', 'dirichlet', 'none')
        Λ0 = DifferentialForm(0, ns, ps, types)

        zero_form_basis = SplineBasis(n,p,"clamped")
        one_form_basis = SplineBasis(n,p,"clamped")


        Λ0 = DifferentialForm(0, ns, ps, types)
        Λ1 = DifferentialForm(1, ns, ps, types)

        # Quadrature order
        q=2

        # Get quadrature points and weights
        Q_0 = QuadratureRule(Λ0, q)
        Q_1 = QuadratureRule(Λ1, q)


        # Assemble matrices


        M = LazyMatrix(Λ0, Λ1, Q_0, Q_1, F, F_inv)
        M_a = LazyMassMatrix(M, Q_0, Q_1, F, F_inv).assemble()
        print(M)


