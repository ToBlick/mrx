# %%
import jax.numpy as jnp
from mrx.DifferentialForms import DifferentialForm
from mrx.Quadrature import QuadratureRule


# Case of Beltrami field with homogenous Dirichlet boundary conditions

# Return mu(m,n)
def mu(m, n):
    return jnp.pi*jnp.sqrt(m**2 + n**2)

# Return u(x)


def u(A_0, x, m, n):
    x_1, x_2, x_3 = x
    return [((A_0*n)/(jnp.sqrt(m**2 + n**2)))*jnp.sin(jnp.pi*m*x_1)*jnp.cos(jnp.pi*n*x_2), ((A_0*m*-1)/(jnp.sqrt(m**2 + n**2)))**jnp.cos(jnp.pi*m*x_1)*jnp.sin(jnp.pi*n*x_2), jnp.sin(jnp.pi*m*x_1)*jnp.sin(jnp.pi*n*x_2)]


# Return eta(x)
def eta(x):
    x_1, x_2, x_3 = x
    return ((x_1**2)*((1-x_1))**2)*((x_2**2)*((1-x_2))**2)*((x_3**2)*((1-x_3))**2)


# Return integrand
def I(m, n, x):
    return (eta(x)**2)*jnp.linalg.norm(u(x))


# Return magnetic helicity
def H(m, n, A_0):
    # Integrate I(m,n,x) over the domain, and multiply by mu

    integral = integral * mu(m, n)
    return integral


# Set n_s and p_s
n = 5
p = 3
ns = (n, n, n)
ps = (p, p, p)
types = ('clamped', 'clamped', 'constant')
# Boundary conditions
bcs = ('dirichlet', 'dirichlet', 'none')
Λ0 = DifferentialForm(0, ns, ps, types)

# 15 points
Q = QuadratureRule(Λ0, 15)
