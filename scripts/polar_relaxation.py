"""
Magnetic Field Relaxation in Polar Coordinates

This script implements a magnetic field relaxation algorithm in polar coordinates,
preserving magnetic helicity while minimizing the Lorentz force.

Key features:
- Implements polar mapping for toroidal geometry
- Uses mixed finite element formulation
- Preserves magnetic helicity during relaxation
- Computes and visualizes force-free states
- Analyzes convergence and conservation properties

The script demonstrates:
- Initial field setup and perturbation
- Conjugate gradient relaxation
- Force-free state computation
- Conservation of magnetic helicity
- Divergence-free constraint maintenance

The script generates several plots:
1. Initial magnetic field configuration
2. Force field evolution
3. Energy and helicity conservation
4. Divergence evolution
5. Convergence analysis
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from pathlib import Path

from mrx.PolarMapping import LazyExtractionOperator, get_xi
from mrx.DifferentialForms import DifferentialForm, DiscreteFunction
from mrx.Quadrature import QuadratureRule
from mrx.Projectors import Projector, CurlProjection
from mrx.LazyMatrices import LazyMassMatrix, LazyDerivativeMatrix, LazyProjectionMatrix, LazyDoubleCurlMatrix
from mrx.Utils import curl

# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)

# Create output directory for figures
output_dir = Path("script_outputs")
output_dir.mkdir(parents=True, exist_ok=True)

# Define mesh parameters
types = ('clamped', 'periodic', 'constant')  # Boundary condition types
ns = (8, 8, 1)  # Number of elements in each direction
ps = (3, 3, 0)  # Polynomial degrees
Λ0, Λ1, Λ2, Λ3 = [DifferentialForm(i, ns, ps, types) for i in range(4)]  # H1, H(curl), H(div), L2
Q = QuadratureRule(Λ0, 10)  # Quadrature rule

# Define domain parameters
a = 1      # Radius of the torus
R0 = 3.0   # Major radius
Y0 = 0.0   # Vertical offset

# Define polar mapping functions
def θ(x):
    """Convert polar coordinates to toroidal angle"""
    r, χ, z = x
    return 2 * jnp.atan(jnp.sqrt((1 + a*r/R0)/(1 - a*r/R0)) * jnp.tan(jnp.pi * χ))

def _R(r, χ):
    """Convert polar to Cartesian x-coordinate"""
    return jnp.ones(1) * (R0 + a * r * jnp.cos(2 * jnp.pi * χ))

def _Y(r, χ):
    """Convert polar to Cartesian y-coordinate"""
    return jnp.ones(1) * (Y0 + a * r * jnp.sin(2 * jnp.pi * χ))

@jax.jit
def F(x):
    """Full mapping from reference to physical coordinates"""
    r, χ, z = x
    return jnp.ravel(jnp.array([_R(r, χ) * jnp.cos(2 * jnp.pi * z),
                               _Y(r, χ),
                               _R(r, χ) * jnp.sin(2 * jnp.pi * z)]))

# Set up extraction operators and matrices
ξ, R_hat, Y_hat, Λ, τ = get_xi(_R, _Y, Λ0)
E0, E1, E2, E3 = [LazyExtractionOperator(Λ, ξ, True).M for Λ in [Λ0, Λ1, Λ2, Λ3]]
M0, M1, M2, M3 = [LazyMassMatrix(Λ, Q, F, E).M for Λ, E in zip([Λ0, Λ1, Λ2, Λ3], [E0, E1, E2, E3])]
P0, P1, P2, P3 = [Projector(Λ, Q, F, E) for Λ, E in zip([Λ0, Λ1, Λ2, Λ3], [E0, E1, E2, E3])]
M12 = LazyProjectionMatrix(Λ1, Λ2, Q, F, E1, E2).M.T
C = LazyDoubleCurlMatrix(Λ1, Q, F, E1).M
D2 = LazyDerivativeMatrix(Λ2, Λ3, Q, F, E2, E3).M
D1 = LazyDerivativeMatrix(Λ1, Λ2, Q, F, E1, E2).M
D0 = LazyDerivativeMatrix(Λ0, Λ1, Q, F, E0, E1).M
Pc = CurlProjection(Λ1, Q, F, E1)  # Computes (B, A x Λ[i]) given A and B

def l2_product(f, g, Q):
    """Compute the L2 inner product of two functions"""
    return jnp.einsum("ij,ij,i->", jax.vmap(f)(Q.x), jax.vmap(g)(Q.x), Q.w)

# Set up plotting grid
ɛ = 1e-5  # Small offset from boundaries
nx = 64   # High resolution for contour plots
_nx = 16  # Lower resolution for quiver plots

# Create evaluation grids
_x1 = jnp.linspace(ɛ, 1-ɛ, nx)
_x2 = jnp.linspace(ɛ, 1-ɛ, nx)
_x3 = jnp.ones(1)/2
_x = jnp.array(jnp.meshgrid(_x1, _x2, _x3))
_x = _x.transpose(1, 2, 3, 0).reshape(nx*nx*1, 3)
_y = jax.vmap(F)(_x)
_y1 = _y[:, 0].reshape(nx, nx)
_y2 = _y[:, 1].reshape(nx, nx)

# Create quiver plot grid
__x1 = jnp.linspace(ɛ, 1-ɛ, _nx)
__x2 = jnp.linspace(ɛ, 1-ɛ, _nx)
__x3 = jnp.ones(1)/2
__x = jnp.array(jnp.meshgrid(__x1, __x2, __x3))
__x = __x.transpose(1, 2, 3, 0).reshape(_nx*_nx*1, 3)
__y = jax.vmap(F)(__x)
__y1 = __y[:, 0].reshape(_nx, _nx)
__y2 = __y[:, 1].reshape(_nx, _nx)

# Define initial vector potential and magnetic field
def A(x):
    """Initial vector potential"""
    r, χ, z = x
    a1 = jnp.sin(2 * jnp.pi * χ)
    a2 = 1
    a3 = jnp.cos(2 * jnp.pi * χ)
    return jnp.array([a1, a2, a3]) * jnp.sin(jnp.pi * r)**2 * r

B = curl(A)  # Initial magnetic field

# Project initial fields
A_hat = jnp.linalg.solve(M1, P1(A))
A_h = DiscreteFunction(A_hat, Λ1, E1)
B0 = curl(A)
B0_hat = jnp.linalg.solve(M2, P2(B0))
B_h = DiscreteFunction(B0_hat, Λ2, E2)
B0_h = DiscreteFunction(B0_hat, Λ2, E2)

# Compute initial errors
def compute_A_error(x): return A(x) - A_h(x)
def compute_B_error(x): return B0(x) - B_h(x)
def compute_curl_error(x): return curl(A)(x) - curl(A_h)(x)

# Print initial diagnostics
print("Initial field errors:")
print(f"A error: {(l2_product(compute_A_error, compute_A_error, Q) / l2_product(A, A, Q))**0.5:.2e}")
print(f"B error: {(l2_product(compute_B_error, compute_B_error, Q) / l2_product(B0, B0, Q))**0.2e}")
print(f"Curl error: {(l2_product(compute_curl_error, compute_curl_error, Q) / l2_product(curl(A), curl(A), Q))**0.2e}")
print(f"Initial helicity: {A_hat @ M12 @ B0_hat:.2e}")
print(f"Initial energy: {B0_hat @ M2 @ B0_hat / 2:.2e}")

# Plot initial field configuration
plt.figure(figsize=(10, 8))
F_B = Pullback(B0, F, 2)
F_B_h = Pullback(B_h, F, 2)
_z1 = jax.vmap(F_B_h)(_x).reshape(nx, nx, 3)
_z1_norm = jnp.linalg.norm(_z1, axis=2)
_z2 = jax.vmap(F_B)(_x).reshape(nx, nx, 3)
_z2_norm = jnp.linalg.norm(_z2, axis=2)
plt.contourf(_y1, _y2, _z1_norm.reshape(nx, nx))
plt.colorbar(label='Field Magnitude')
plt.contour(_y1, _y2, _z2_norm.reshape(nx, nx), colors='k')
__z1 = jax.vmap(F_B_h)(__x).reshape(_nx, _nx, 3)
plt.quiver(__y1, __y2, __z1[:,:,0], __z1[:,:,1], color='w')
plt.title('Initial Magnetic Field Configuration')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.savefig(output_dir / 'initial_field.png', dpi=300, bbox_inches='tight')

# Calculate SVD for reconstruction
U, S, Vh = jnp.linalg.svd(C)
S_inv = jnp.where(S/S[0] > 1e-11, 1/S, 0)

# Initialize lists for tracking evolution
helicities: List[float] = []
energies: List[float] = []
forces: List[float] = []
critical_as: List[int] = []
divBs: List[float] = []

# %%
@jax.jit
def ẟB_hat(B_hat, B_hat_0, dt):
    """Compute the magnetic field update using the Lorentz force.
    
    Args:
        B_hat: Current magnetic field coefficients
        B_hat_0: Previous magnetic field coefficients
        dt: Time step
        
    Returns:
        tuple: (B_diff, error, B_hat_1, u_hat)
            B_diff: Field difference
            error: L2 error norm
            B_hat_1: Updated field coefficients
            u_hat: Velocity field coefficients
    """
    H_hat = jnp.linalg.solve(M1, M12 @ (B_hat + B_hat_0))/2     # H = Proj(B)
    J_hat = jnp.linalg.solve(M1, D1.T @ (B_hat + B_hat_0))/2    # J = curl H
    H_h = DiscreteFunction(H_hat, Λ1, E1)
    J_h = DiscreteFunction(J_hat, Λ1, E1)

    def JcrossH(x):
        return jnp.cross(J_h(x), H_h(x))
    u_hat = jnp.linalg.solve(M2, P2(JcrossH))           # u = J x H
    u_h = DiscreteFunction(u_hat, Λ2, E2)
    E_hat = jnp.linalg.solve(M1, Pc(H_h, u_h))          # E = u x H
    ẟB_hat = jnp.linalg.solve(M2, D1 @ E_hat)           # ẟB = curl E
    B_hat_1 = B_hat_0 + dt * ẟB_hat
    B_diff = B_hat_1 - B_hat
    error = jnp.array((B_diff @ M2 @ B_diff)**0.5, dtype=jnp.float64)
    return B_diff, error, B_hat_1, u_hat

def rk4(y):
    """Fourth-order Runge-Kutta integration step.
    
    Args:
        y: Current state vector
        
    Returns:
        jnp.ndarray: Updated state vector
    """
    k1 = ẟB_hat(y, y, dt)
    k2 = ẟB_hat(y + k1[0]/2, y, dt)
    k3 = ẟB_hat(y + k2[0]/2, y, dt)
    k4 = ẟB_hat(y + k3[0], y, dt)
    return y + (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])/6

@jax.jit
def integrate(x):
    """Integrate the magnetic field evolution.
    
    Args:
        x: Initial state vector
        
    Returns:
        tuple: (final_state, diagnostics)
    """
    def step(x, _):
        x = rk4(x)
        return x, x
    return jax.lax.scan(step, x, jnp.arange(n_steps))

# Main relaxation loop
n_steps = 1000
dt = 0.01
B_hat = B0_hat
traces = []

print("\nStarting relaxation...")
for i in range(n_steps):
    B_diff, error, B_hat, u_hat = ẟB_hat(B_hat, B_hat, dt)
    
    # Track diagnostics
    A_hat = U @ jnp.diag(S_inv) @ Vh @ D1.T @ B_hat
    helicity = A_hat @ M12 @ B_hat
    energy = B_hat @ M2 @ B_hat / 2
    force = u_hat @ M2 @ u_hat
    divB = (D2 @ B_hat) @ jnp.linalg.solve(M3, D2 @ B_hat)
    
    helicities.append(helicity)
    energies.append(energy)
    forces.append(force)
    divBs.append(divB)
    
    if i % 100 == 0:
        print(f"\nIteration {i}:")
        print(f"Helicity: {helicity:.2e}")
        print(f"Energy: {energy:.2e}")
        print(f"Force: {force:.2e}")
        print(f"Div B: {divB:.2e}")
        print(f"Error: {error:.2e}")

# Plot evolution of quantities
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(energies)
plt.title('Energy Evolution')
plt.xlabel('Iteration')
plt.ylabel('Energy')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(helicities)
plt.title('Helicity Evolution')
plt.xlabel('Iteration')
plt.ylabel('Helicity')
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(forces)
plt.title('Force Evolution')
plt.xlabel('Iteration')
plt.ylabel('Force Norm')
plt.yscale('log')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(divBs)
plt.title('Divergence Evolution')
plt.xlabel('Iteration')
plt.ylabel('Div B')
plt.yscale('log')
plt.grid(True)

plt.tight_layout()
plt.savefig(output_dir / 'evolution.png', dpi=300, bbox_inches='tight')

# Plot final field configuration
plt.figure(figsize=(10, 8))
B_h = DiscreteFunction(B_hat, Λ2, E2)
F_B = Pullback(B0, F, 2)
F_B_h = Pullback(B_h, F, 2)
_z1 = jax.vmap(F_B_h)(_x).reshape(nx, nx, 3)
_z1_norm = jnp.linalg.norm(_z1, axis=2)
_z2 = jax.vmap(F_B)(_x).reshape(nx, nx, 3)
_z2_norm = jnp.linalg.norm(_z2, axis=2)
plt.contourf(_y1, _y2, _z1_norm.reshape(nx, nx))
plt.colorbar(label='Field Magnitude')
plt.contour(_y1, _y2, _z2_norm.reshape(nx, nx), colors='k')
__z1 = jax.vmap(F_B_h)(__x).reshape(_nx, _nx, 3)
plt.quiver(__y1, __y2, __z1[:,:,0], __z1[:,:,1], color='w')
plt.title('Final Magnetic Field Configuration')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.savefig(output_dir / 'final_field.png', dpi=300, bbox_inches='tight')

# Print final diagnostics
print("\nFinal diagnostics:")
print(f"Final helicity: {helicities[-1]:.2e}")
print(f"Final energy: {energies[-1]:.2e}")
print(f"Final force: {forces[-1]:.2e}")
print(f"Final div B: {divBs[-1]:.2e}")
print(f"Helicity conservation: {abs(helicities[-1] - helicities[0])/helicities[0]:.2e}")
print(f"Energy change: {(energies[-1] - energies[0])/energies[0]:.2e}")

# Show all plots
plt.show()
