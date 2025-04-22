
import jax.experimental
import jax.experimental.sparse
from mrx.Quadrature import *
from mrx.SplineBases import *
from mrx.DifferentialForms import *
from mrx.Projectors import *
from mrx.PolarMapping import *


import jax
jax.config.update("jax_enable_x64", True)


# #
# class FormsTests(unittest.TestCase):
# # 1D projection
#     def test_assembly(self):

#         alpha = jnp.pi/2
#         # F maps the logical domain (unit cube) to the physical one by rotating it by 90 degrees
#         def F(x):
#             return jnp.array([ [ jnp.cos(alpha), jnp.sin(alpha), 0],
#                                [-jnp.sin(alpha), jnp.cos(alpha), 0],
#                                [0              , 0             , 1] ]) @ (x - jnp.ones(3)/2) + jnp.ones(3)/2
#         def F_inv(x):
#             return jnp.array([ [jnp.cos(alpha), -jnp.sin(alpha), 0],
#                                [jnp.sin(alpha),  jnp.cos(alpha), 0],
#                                [0             , 0              , 1]]) @ (x - jnp.ones(3)/2) + jnp.ones(3)/2
        
#      # Test 1D projection
#         n = 16
#         p = 3
#         ns = (n, n, 1)
#         ps = (p, p, 1)
#         types = ('clamped', 'clamped', 'fourier')
#         boundary = ('free', 'free', 'periodic')
#         #0-form
#         Λ0 = DifferentialForm(0, ns, ps, types)

#         #1-form
#         Λ1 = DifferentialForm(1, ns, ps, types)

#         #2-form
#         Λ2 = DifferentialForm(2, ns, ps, types)

#         #3-form
#         Λ3 = DifferentialForm(3, ns, ps, types)

          
#         s = SplineBasis(Λ0,0,ns[0], ps[0], types[0], boundary[0])
#         # Quadrature
#         Q = QuadratureRule(Λ0, 11) 

       

#         # Getting divergence matrix
#         D = LazyMatrices.LazyDerivativeMatrix.assemble(Λ0,Λ1, Q,F).M

#         # Getting Mass matrix
#         M = LazyMatrices.LazyMassMatrix(Λ0,Q,F).M
        
#          # Define f and u
#         def f(x):
#             return 2 * (2 * jnp.pi)**2 * jnp.sin(2 * jnp.pi * x[0]) * jnp.sin(2 * jnp.pi * x[1])
#         def u(x):
#             return jnp.sin(jnp.pi * 2 * x[0]) * jnp.sin(2 * jnp.pi * x[1])

#         # Projection
#         Proj = Projectors.Projector(Λ3,Q,F).threeform_projection(f)
