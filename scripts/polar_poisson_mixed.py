"""
2D Poisson Problem in Polar Coordinates with Mixed Formulation

This script solves the 2D Poisson equation in polar coordinates using a mixed finite element formulation.
The problem is defined on a circular domain with Dirichlet boundary conditions.

The equation being solved is:
    -∇²u = f

where:
    u(r,θ) = -(1/16)r⁴ + (1/12)r³  (exact solution)
    f(r,θ) = r² - (3/4)r           (source term)

The domain is a circle of radius 'a' centered at (R0, Y0).

Key features:
- Uses mixed finite element formulation
- Implements polar mapping
- Demonstrates convergence analysis
- Visualizes solution and error
- Compares different polynomial orders and quadrature rules

The script generates several plots:
1. Solution comparison (exact vs numerical)
2. Source term comparison (exact vs numerical)
3. Convergence analysis (error vs degrees of freedom)
4. Performance analysis (computation time vs degrees of freedom)
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

from mrx.PolarMapping import LazyExtractionOperator, get_xi
from mrx.DifferentialForms import DifferentialForm, DiscreteFunction, Pullback
from mrx.Quadrature import QuadratureRule
from mrx.Projectors import Projector
from mrx.LazyMatrices import LazyMassMatrix, LazyDerivativeMatrix, LazyProjectionMatrix
from mrx.Utils import l2_product
from mrx.Plotting import converge_plot
from functools import partial

# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)


@partial(jax.jit, static_argnames=['n', 'p', 'q'])
def get_err(n, p, q):
    """
    Compute the relative L2 error for the 2D Poisson problem in polar coordinates.
    
    This function solves the Poisson equation using a mixed finite element formulation
    and computes the relative L2 error between the exact and numerical solutions.
    
    Parameters
    ----------
    n : int
        Number of elements in the radial direction
    p : int
        Polynomial degree for the finite element spaces
    q : int
        Quadrature order for numerical integration
        
    Returns
    -------
    float
        Relative L2 error between exact and numerical solutions, computed as:
        ||u - u_h||_L2 / ||u||_L2
        where u is the exact solution and u_h is the numerical solution
        
    Notes
    -----
    - Uses a mixed formulation with H1, H(div), and L2 spaces
    - Implements polar mapping for the circular domain
    - The exact solution is u(r,θ) = -(1/16)r⁴ + (1/12)r³
    - The source term is f(r,θ) = r² - (3/4)r
    - The domain is a unit circle centered at the origin
    """

    def _R(r, χ):
        return jnp.ones(1) * r * jnp.cos(2 * jnp.pi * χ)

    def _Y(r, χ):
        return jnp.ones(1) * r * jnp.sin(2 * jnp.pi * χ)

    def F(p):
        r, χ, z = p
        return jnp.squeeze(jnp.array([_R(r, χ), _Y(r, χ), jnp.ones(1) * z]))

    ns = (n, p, 1)
    ps = (p, p, 0)
    types = ('clamped', 'periodic', 'constant')
    Λ0, Λ2, Λ3 = [DifferentialForm(i, ns, ps, types) for i in [0, 2, 3]]
    Q = QuadratureRule(Λ0, q)
    ξ, R_hat, Y_hat, Λ, τ = get_xi(_R, _Y, Λ0, Q)
    E0, E2, E3 = [LazyExtractionOperator(Λ, ξ, False).M for Λ in [Λ0, Λ2, Λ3]]
    D = LazyDerivativeMatrix(Λ2, Λ3, Q, F, E2, E3).M
    M2 = LazyMassMatrix(Λ2, Q, F, E2).M
    K = D @ jnp.linalg.solve(M2, D.T)

    def u(x):
        r, χ, z = x
        return -jnp.ones(1) * 1/4 * (1/4 * r**4 - 1/3 * r**3 + 1/12)

    def f(x):
        r, χ, z = x
        return jnp.ones(1) * (r - 3/4) * r

    P3 = Projector(Λ3, Q, F, E3)
    u_hat = jnp.linalg.solve(K, P3(f))

    M03 = LazyProjectionMatrix(Λ0, Λ3, Q, F, E0, E3).M
    M0 = LazyMassMatrix(Λ0, Q, F, E0).M
    u_hat = jnp.linalg.solve(M0, M03.T @ u_hat)
    u_h = DiscreteFunction(u_hat, Λ0, E0)

    def err(x): return (u(x) - u_h(x))
    error = (l2_product(err, err, Q, F) / l2_product(u, u, Q, F))**0.5
    return error


# Create output directory
output_dir = Path("script_outputs")
output_dir.mkdir(parents=True, exist_ok=True)

# Problem parameters
n = 8      # Number of elements in radial direction
p = 3      # Polynomial degree
q = 3      # Quadrature order

# Domain parameters
a = 1      # Radius of the circle
R0 = 0.0   # x-coordinate of center
Y0 = 0.0   # y-coordinate of center

# Define the polar mapping functions


def _R(r, χ):
    """Convert polar to Cartesian x-coordinate"""
    return jnp.ones(1) * (R0 + a * r * jnp.cos(2 * jnp.pi * χ))


def _Y(r, χ):
    """Convert polar to Cartesian y-coordinate"""
    return jnp.ones(1) * (Y0 + a * r * jnp.sin(2 * jnp.pi * χ))


def F(x):
    """Full mapping from reference to physical coordinates"""
    r, χ, z = x
    return jnp.ravel(jnp.array([_R(r, χ),
                               _Y(r, χ),
                               jnp.ones(1) * z]))

# Define exact solution and source term


def u(x):
    """Exact solution: u(r,θ) = -(1/16)r⁴ + (1/12)r³"""
    r, χ, z = x
    return -jnp.ones(1) * (1/16 * r**4 - 1/12 * r**3)


def f(x):
    """Source term: f(r,θ) = r² - (3/4)r"""
    r, χ, z = x
    return jnp.ones(1) * (r**2 - 3/4 * r)


# Set up finite element spaces
ns = (n, 4, 1)  # Number of elements in each direction
ps = (p, p, 0)  # Polynomial degrees
types = ('clamped', 'periodic', 'constant')  # Boundary conditions

# Initialize differential forms
Λ0 = DifferentialForm(0, ns, ps, types)  # H1 space
Λ2 = DifferentialForm(2, ns, ps, types)  # H(div) space
Λ3 = DifferentialForm(3, ns, ps, types)  # L2 space
Q = QuadratureRule(Λ0, q)

# Set up extraction operators and quadrature
ξ, R_hat, Y_hat, Λ, τ = get_xi(_R, _Y, Λ0, Q)
Q = QuadratureRule(Λ0, q)
E0 = LazyExtractionOperator(Λ0, ξ, False).M
E2 = LazyExtractionOperator(Λ2, ξ, False).M
E3 = LazyExtractionOperator(Λ3, ξ, False).M

# Assemble matrices
M03 = LazyProjectionMatrix(Λ0, Λ3, Q, F, E0, E3).M
M0 = LazyMassMatrix(Λ0, Q, F, E0).M
D = LazyDerivativeMatrix(Λ2, Λ3, Q, F, E2, E3).M
M2 = LazyMassMatrix(Λ2, Q, F, E2).M
M3 = LazyMassMatrix(Λ3, Q, F, E3).M
K = D @ jnp.linalg.solve(M2, D.T)

# Project and solve
P3 = Projector(Λ3, Q, F, E3)
P0 = Projector(Λ0, Q, F, E0)
f_hat = jnp.linalg.solve(M3, P3(f))
f_hat_0 = jnp.linalg.solve(M0, P0(f))
u_hat = jnp.linalg.solve(K, P3(f))

# Create discrete functions
u_h = DiscreteFunction(u_hat, Λ3, E3)
f_h = DiscreteFunction(f_hat, Λ3, E3)

# Compute error


def err(x): return (u(x) - u_h(x))


error = (l2_product(err, err, Q, F) / l2_product(u, u, Q, F))**0.5

# Set up plotting grid
ɛ = 1e-5  # Small offset from boundaries
nx = 64   # Number of points for plotting
_x1 = jnp.linspace(ɛ, 1-ɛ, nx)
_x2 = jnp.zeros(1)
_x3 = jnp.zeros(1)
_x = jnp.array(jnp.meshgrid(_x1, _x2, _x3))
_x = _x.transpose(1, 2, 3, 0).reshape(nx*1*1, 3)
_y = jax.vmap(F)(_x)

# Plot source term as 3-form
plt.figure(figsize=(10, 6))
F_f_h = Pullback(f_h, F, 3)
F_f = Pullback(f, F, 3)
plt.plot(_y[:, 0], jax.vmap(F_f_h)(_x), label='Numerical (3-form)')
plt.plot(_y[:, 0], jax.vmap(F_f)(_x), label='Exact (3-form)')
plt.xlabel('Radial Coordinate (r)')
plt.ylabel('Source Term (f)')
plt.title('Source Term Comparison (3-form)')
plt.legend()
plt.grid(True)
plt.savefig(output_dir / 'source_term_3form.png', dpi=300, bbox_inches='tight')

# Plot source term as 0-form
plt.figure(figsize=(10, 6))
F_f_h = Pullback(DiscreteFunction(f_hat_0, Λ0, E0), F, 0)
F_f = Pullback(f, F, 0)
plt.plot(_y[:, 0], jax.vmap(F_f_h)(_x), label='Numerical (0-form)')
plt.plot(_y[:, 0], jax.vmap(F_f)(_x), label='Exact (0-form)')
plt.xlabel('Radial Coordinate (r)')
plt.ylabel('Source Term (f)')
plt.title('Source Term Comparison (0-form)')
plt.legend()
plt.grid(True)
plt.savefig(output_dir / 'source_term_0form.png', dpi=300, bbox_inches='tight')

# Plot solution
plt.figure(figsize=(10, 6))
F_u_h = Pullback(u_h, F, 3)
F_u = Pullback(u, F, 3)
plt.plot(_y[:, 0], jax.vmap(F_u_h)(_x), label='Numerical')
plt.plot(_y[:, 0], jax.vmap(F_u)(_x), label='Exact')
plt.xlabel('Radial Coordinate (r)')
plt.ylabel('Solution (u)')
plt.title('Solution Comparison')
plt.legend()
plt.grid(True)
plt.savefig(output_dir / 'solution_comparison.png', dpi=300, bbox_inches='tight')

# Convergence analysis
print("\nRunning convergence analysis...")
ns = np.arange(4, 18, 2)
ps = np.arange(1, 4)
qs = np.arange(4, 5)
err = np.zeros((len(ns), len(ps), len(qs)))
times = np.zeros((len(ns), len(ps), len(qs)))

for i, n in enumerate(ns):
    for j, p in enumerate(ps):
        for k, q in enumerate(qs):
            start = time.time()
            err[i, j, k] = get_err(n, p, q)
            end = time.time()
            times[i, j, k] = end - start
            print(f"n={n}, p={p}, q={q}, err={err[i, j, k]:.2e}, time={times[i, j, k]:.2f}s")

# Plot convergence
plt.figure(figsize=(10, 6))
for j, p in enumerate(ps):
    plt.plot(ns, err[:, j, 0], 'o-', label=f'p={p}')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of Elements (n)')
plt.ylabel('Relative L2 Error')
plt.title('Convergence Analysis')
plt.grid(True)
plt.legend()
plt.savefig(output_dir / 'convergence.png', dpi=300, bbox_inches='tight')

# Plot performance
plt.figure(figsize=(10, 6))
for j, p in enumerate(ps):
    plt.plot(ns, times[:, j, 0], 'o-', label=f'p={p}')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of Elements (n)')
plt.ylabel('Computation Time (s)')
plt.title('Performance Analysis')
plt.grid(True)
plt.legend()
plt.savefig(output_dir / 'performance.png', dpi=300, bbox_inches='tight')

# Print final error
print(f"\nFinal relative L2 error: {error:.2e}")

# Show all plots
plt.show()
