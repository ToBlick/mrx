"""
2D Poisson Problem in Polar Coordinates with Constant Angle

This script solves the 2D Poisson equation in polar coordinates using a finite element method
with a constant angle approximation. The problem is defined on a circular domain with
Dirichlet boundary conditions.

The equation being solved is:
    -∇²u = f

where:
    u(r,θ) = r³(3log(r) - 2)/27 + 2/27  (exact solution)
    f(r,θ) = -r log(r)                   (source term)

The domain is a circle of radius 'a' centered at (R0, Y0).

Key features:
- Uses standard finite element formulation
- Implements polar mapping with constant angle
- Demonstrates convergence analysis
- Visualizes solution and error
- Compares different polynomial orders and quadrature rules

The script generates several plots:
1. Convergence analysis (error vs degrees of freedom)
2. Performance analysis (computation time vs degrees of freedom)
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path

from mrx.DifferentialForms import DifferentialForm, DiscreteFunction
from mrx.Quadrature import QuadratureRule
from mrx.Projectors import Projector
from mrx.LazyMatrices import LazyStiffnessMatrix
from mrx.Utils import l2_product
from mrx.Plotting import converge_plot
from functools import partial

# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)

# Create output directory
output_dir = Path("script_outputs")
output_dir.mkdir(parents=True, exist_ok=True)

@partial(jax.jit, static_argnames=['n', 'p', 'q'])
def get_err(n, p, q):
    """
    Compute the relative L2 error for the 2D Poisson problem in polar coordinates
    with constant angle approximation.
    
    This function solves the Poisson equation using a standard finite element formulation
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
    - Uses standard finite element formulation with H1 space
    - Implements polar mapping with constant angle for the circular domain
    - The exact solution is u(r,θ) = r³(3log(r) - 2)/27 + 2/27
    - The source term is f(r,θ) = -r log(r)
    - The domain is a circle of radius 1 centered at (3,0)
    """
    a = 1
    R0 = 3.0
    Y0 = 0.0

    def _R(r, χ):
        return jnp.ones(1) * (R0 + a * r * jnp.cos(2 * jnp.pi * χ))

    def _Y(r, χ):
        return jnp.ones(1) * (Y0 + a * r * jnp.sin(2 * jnp.pi * χ))

    def F(x):
        r, χ, z = x
        return jnp.ravel(jnp.array([_R(r, χ),
                                    _Y(r, χ),
                                    jnp.ones(1) * z]))
    ns = (n, 1, 1)
    ps = (p, 0, 0)

    def u(x):
        r, χ, z = x
        return jnp.ones(1) * r**3 * (3 * jnp.log(r) - 2) / 27 + 2/27

    def f(x):
        r, χ, z = x
        return -jnp.ones(1) * r * jnp.log(r)
    types = ('clamped', 'constant', 'constant')

    Λ0 = DifferentialForm(0, ns, ps, types)
    E0 = jnp.eye(Λ0.n - 1, Λ0.n)

    Q = QuadratureRule(Λ0, q)
    # M0 = LazyMassMatrix(Λ0, Q, F=F, E=E0).M
    K = LazyStiffnessMatrix(Λ0, Q, F=F, E=E0).M
    P0 = Projector(Λ0, Q, F=F, E=E0)
    u_hat = jnp.linalg.solve(K, P0(f))
    u_h = DiscreteFunction(u_hat, Λ0, E0)
    def err(x): return (u(x) - u_h(x))
    error = (l2_product(err, err, Q, F) / l2_product(u, u, Q, F))**0.5
    return error

# Convergence analysis
print("\nRunning convergence analysis...")
ns = np.arange(4, 20, 2)
ps = np.arange(1, 4)
qs = np.arange(3, 11, 3)
err = np.zeros((len(ns), len(ps), len(qs)))
times = np.zeros((len(ns), len(ps), len(qs)))

for i, n in enumerate(ns):
    for j, p in enumerate(ps):
        for k, q in enumerate(qs):
            start = time.time()
            err[i, j, k] = get_err(n, p, q)
            end = time.time()
            times[i, j, k] = end - start
            print(f"n={n}, p={p}, q={q}, err={err[i,j,k]:.2e}, time={times[i,j,k]:.2f}s")

# Plot and save convergence analysis
# fig = converge_plot(err, ns, ps, qs)
# fig.update_layout(
#     xaxis_type="log",
#     yaxis_type="log",
#     yaxis_tickformat=".1e",
#     xaxis_title='Number of Elements (n)',
#     yaxis_title='Relative L2 Error',
#     title='Convergence Analysis'
# )
plt.figure(figsize=(10, 6))
plt.plot(ns, err[:, 0, 0], 'o-', label='p=1')
plt.plot(ns, err[:, 1, 0], 's-', label='p=2')
plt.plot(ns, err[:, 2, 0], '^-', label='p=3')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of Elements (n)')
plt.ylabel('Relative L2 Error')
plt.title('Convergence Analysis')
plt.grid(True)
plt.legend()
plt.savefig(output_dir / 'convergence.png', dpi=300, bbox_inches='tight')

# Plot and save performance analysis
# fig = converge_plot(times, ns, ps, qs)
# fig.update_layout(
#     xaxis_type="log",
#     yaxis_type="log",
#     yaxis_tickformat=".1e",
#     xaxis_title='Number of Elements (n)',
#     yaxis_title='Computation Time (s)',
#     title='Performance Analysis'
# )
plt.figure(figsize=(10, 6))
plt.plot(ns, times[:, 0, 0], 'o-', label='p=1')
plt.plot(ns, times[:, 1, 0], 's-', label='p=2')
plt.plot(ns, times[:, 2, 0], '^-', label='p=3')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of Elements (n)')
plt.ylabel('Computation Time (s)')
plt.title('Performance Analysis')
plt.grid(True)
plt.legend()
plt.savefig(output_dir / 'performance.png', dpi=300, bbox_inches='tight')

# Show all plots
plt.show()
