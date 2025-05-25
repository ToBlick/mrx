import jax
import jax.numpy as jnp
import numpy as np
import os
import matplotlib.pyplot as plt
from mrx.Utils import grad, curl, div, jacobian, inv33, l2_product
from mrx.DifferentialForms import DifferentialForm
from mrx.Quadrature import QuadratureRule
from mrx.LazyMatrices import LazyMassMatrix, LazyDerivativeMatrix
from mrx.Projectors import Projector
from functools import partial

# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)

# Create output directory for figures
os.makedirs('script_outputs', exist_ok=True)

def vector_laplace_torus(n, p):
    """
    Solve the vector Laplace equation ∆u = 0 on a torus with Dirichlet boundary conditions.
    Recall that ∆u = (∇(∇⋅u) - ∇ x(∇ x u) = 0.
    
    Args:
        n: Number of elements 
        p: Polynomial degree
        
    Returns:
        tuple: (A_coeffs, error), where A_coeffs are the coefficients of the vector potential
    """
   
    ns = (n, n, n)  # Number of basis functions 
    ps = (p, p, p)  # Polynomial order
    types = ('clamped', 'periodic', 'clamped')  
    
    # Create differential forms
    Λ0 = DifferentialForm(0, ns, ps, types)  # For div u
    Λ1 = DifferentialForm(1, ns, ps, types)  # For u
    Λ2 = DifferentialForm(2, ns, ps, types)  # For curl u
    
    # Create quadrature rule
    Q = QuadratureRule(Λ1, p) 
    
    # Create projectors
    P0 = Projector(Λ0, Q)  # Projector for div u
    P1 = Projector(Λ1, Q)  # Projector for u
    P2 = Projector(Λ2, Q)  # Projector for curl u
    
    # Create mass and derivative matrices
    M0 = LazyMassMatrix(Λ0, Q).zeroform_assemble() 
    M1 = LazyMassMatrix(Λ1, Q).oneform_assembl() 
    M2 = LazyMassMatrix(Λ2, Q).twoform_assemble()
    D01 = LazyDerivativeMatrix(Λ0, Λ1, Q).grad_assemble()   
    D12 = LazyDerivativeMatrix(Λ1, Λ2, Q).curl_assemble()
    
    
    # Project dirichlet boundary conditions
    bc_coeffs = P1.oneform_projection(jnp.zeros(3))
    

