import jax
import jax.numpy as jnp
from mrx.Quadrature import QuadratureRule
from scipy import integrate
import scipy as scipy


# Example from Bressan Thesis, Section A.3.2. Choosing homogeneous Dirichlet boundary conditions.
# Logical domain is [0,1]x[0,1]

# Find ω
def vorticity_gaussian(x,x_0):
    
    x_10,x_20 = x_0
    x_1,x_2 = x
    return jnp.exp((-((x_1-x_10)**2)/0.01)-(((x_2-x_20)**2)/0.07))




def eigenfunctions_Laplace(n,m,x):
    # Find eigenfunctions of the Laplace operator on the unit square
    # with Dirichlet boundary conditions.

        x_1,x_2 = x
        return jnp.sin(n*jnp.pi*x_1)*jnp.sin(m*jnp.pi*x_2)
        #A= 1

n=6
m=4

def vorticity(n,m,x,x_0):

    return  eigenfunctions_Laplace(n,m,x) +vorticity_gaussian(x,x_0)



# Find scalar potential via vorticity = ω =−∆ϕ,the Poisson equation. We use the Green's function.

def potential(v,x,n,m,x_0):
    # Find the potential from the vorticity using the Green's function. The Green's function for the 2D Laplace equation is
    # G(x,y) = 1/(2π) log(|x-y|), and rhe potential is given by: ϕ(x) = -∫∫  G(x,y)ω(y) dy + ϕ_L, where ϕ_L is a solution to Laplace's equation.
    # We will use a Quadrature rule to estimate the 2D integral. For ϕ_L, I will use ϕ_L(x_1, x_2) = x_1^2-x_2^2


        x_1,x_2 = x
        def Integrand(y):
               y_1,y_2 = y
               return (-1/(2*jnp.pi))*jnp.log(jnp.sqrt((x_1-y_1)**2+(x_2-y_2)**2))*vorticity(n,m,x,x_0)

        # For now I use scipy to estimate integral
        In =scipy.integrate.dblquad(Integrand, 0, 1, 0, 1)
        In = In+ x_1**2-x_2**2
       