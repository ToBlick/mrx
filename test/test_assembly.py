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

    def test_assembly(self):
        # Map to physical domain: unit cube to [-3,3]^3
        def F(x):
            return 6*x - jnp.ones_like(x)*3

        def F_inv(x):    
            return (x + jnp.ones_like(x)*3)/6
        

        ns = (2,2,1) # Number of basis functions
        ps = (1,1,0) # Polynomial degree
        types = ("clamped", "periodic", "constant")
        
        Λ0 = DifferentialForm(0, ns, ps, types)
        Λ1 = DifferentialForm(1, ns, ps, types)

        # Quadrature order
        q=2

        # Get quadrature points and weights
        Q_0 = QuadratureRule(Λ0, q)
        Q_1 = QuadratureRule(Λ1, q)

        # Assemble matrices
        M_0 = LazyMassMatrix(Λ0, Q_0, F).assemble()
        M_1 = LazyMassMatrix(Λ1, Q_1, F).assemble()

        # Check matrix dimensions match the number of basis functions
        # Size is the total number of basis functions
        expected_size_0 = Λ0.n
        
        # Size is the sum of the components
        expected_size_1 = Λ1.n
        
        self.assertEqual(M_0.shape, (expected_size_0, expected_size_0), 
                        f"0-form mass matrix has incorrect dimensions.")
        self.assertEqual(M_1.shape, (expected_size_1, expected_size_1), 
                        f"1-form mass matrix has incorrect dimensions.")

    def test_mass_matrix_properties(self):
        # Map to physical domain: unit cube to [0,3]^3 with scaling and translation
        def F(x):
            return 3*x

        def F_inv(x):    
            return x/3
        
        # Define bases with different parameters
        ns = (3,3,1)  #number of basis functions
        ps = (2,2,0)  #polynomial degree
        types = ("clamped", "periodic", "constant")
        
        # Create differential forms
        Λ0 = DifferentialForm(0, ns, ps, types)
        Λ1 = DifferentialForm(1, ns, ps, types)

        # Quadrature order
        q = 3
        
        # Get quadrature rules
        Q_0 = QuadratureRule(Λ0, q)
        Q_1 = QuadratureRule(Λ1, q)

        # Assemble mass matrices
        M_0 = LazyMassMatrix(Λ0, Q_0, F).assemble()
        M_1 = LazyMassMatrix(Λ1, Q_1, F).assemble()

        # Test matrix properties
        # 1. Check symmetry
        npt.assert_array_almost_equal(M_0, M_0.T, decimal=11,
            err_msg="0-form mass matrix is not symmetric")
        npt.assert_array_almost_equal(M_1, M_1.T, decimal=11,
            err_msg="1-form mass matrix is not symmetric")



if __name__ == '__main__':
    unittest.main()
