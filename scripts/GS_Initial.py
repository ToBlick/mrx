import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
from typing import List, Tuple
from mrx.DifferentialForms import DifferentialForm
from mrx.Quadrature import *


# From Kaptanoglu paper "Grad–Shafranov equilibria via data-free physics informed neural networks"




def fixed_boundary_condition(ɛ: float,κ: float, δ:float) -> jnp.ndarray:
    """Define fixed boundary condition."""
    τ = jnp.linspace(0, 2*jnp.pi, 100)
    Phi = jnp.linspace(0, 2*jnp.pi, 100)
 
    # Boundary conditions:
    R = 1+ɛ*jnp.cos(τ + jnp.arcsin(δ)*jnp.sin(τ))
    Z = ɛ*κ*jnp.sin(τ)



    # Convert to Cartesian coordinates
    X = R * jnp.cos(Phi)
    Y = R * jnp.sin(Phi)


    # Plot the cross section
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_trisurf(X, Y, Z, linewidth=0, antialiased=False)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('3D Toroidal Surface')
    plt.tight_layout()
    plt.show()

# Iter
fixed_boundary_condition(0.32,1.7,0.33)

# NSTX
fixed_boundary_condition(0.78,2.0,0.35)

# Spheromak
fixed_boundary_condition(0.95,1.0,0.2)
