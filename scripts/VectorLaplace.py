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


def vector_laplace_torus(n:float, p:float)-> jnp.ndarray:
    """
    Solve the vector Laplace equation ∆u = 0 on a torus with Dirichlet boundary conditions.
    Recall that ∆u = (∇(∇⋅u) - ∇ x(∇ x u) = 0.
    
    Args:
        n: Number of elements 
        p: Polynomial degree
        
    Returns:
        jnp.ndarray: Solution vector field u
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
    M1 = LazyMassMatrix(Λ1, Q).oneform_assemble() 
    M2 = LazyMassMatrix(Λ2, Q).twoform_assemble()
    D01 = LazyDerivativeMatrix(Λ0, Λ1, Q).grad_assemble()   
    D12 = LazyDerivativeMatrix(Λ1, Λ2, Q).curl_assemble()
    
    
    # Enforce zero boundary conditions by zeroing out first and last row of the mass matrix for u
    M1 = M1.at[0, :].set(0.0)
    M1 = M1.at[-1, :].set(0.0)


def analytic_solution_vector_laplace():
    """
    This is a function to return an analytical solution to the vector laplace equation on a torus.
    Args:
        n: Number of elements 
        p: Polynomial degree
        
    Returns:
        jnp.ndarray: Solution vector field u
    
    """

def plot_laplace_vector_field(u_output:jnp.ndarray, n:float, p:float, save_path='script_outputs/vector_laplace'):
    """
    Plot the vector field solution to Laplace equation on a unit cube
    
    Args:
        u_output: Solution to vector laplace equation
        n: Number of elements 
        p: Polynomial degree
        P1: Projector for the vector field
        save_path: Base path for saving the plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Parameters for torus visualization
    r_major = 1.0  # Major radius of the torus
    r_minor = 0.1  # Minor radius of the torus

    
    # 3D Visualization
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    theta = np.linspace(0, 2*np.pi, 100)
    phi = np.linspace(0, 2*np.pi, 105)
    θ, φ = np.meshgrid(theta, phi)
    
    # Parametric equations of torus
    X = (r_major + r_minor*jnp.cos(φ)) * jnp.cos(θ)
    Y = (r_major + r_minor*jnp.cos(φ)) * jnp.sin(θ)
    Z = r_minor * np.sin(φ)
    
    # Plot torus surface
   # Plot torus surface with improved appearance
    surf = ax.plot_surface(X, Y, Z, alpha=0.3, color='gray', 
                         rstride=2, cstride=2, 
                         linewidth=0.5, edgecolor='k')
    
    ax.set_xlabel(r'$x$', fontsize=12)
    ax.set_ylabel(r'$y$', fontsize=12)
    ax.set_zlabel(r'$z$', fontsize=12)

    
    plt.title('Torus')
    plt.savefig(f'{save_path}.png')
    plt.close()


