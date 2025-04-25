"""
Magnetic Field Relaxation in a 3D Domain

This script implements a magnetic field relaxation process in a 3D domain using differential forms
and finite element methods. It includes:

1. Setup of differential forms and operators for H(curl) and H(div) spaces
2. Implementation of magnetic field evolution equations
3. Analysis of magnetic helicity and energy conservation
4. Visualization of field evolution and conservation properties

The script demonstrates:
- Magnetic field relaxation under helicity conservation
- Energy evolution during relaxation
- Divergence-free constraint maintenance
- Force balance analysis
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
from pathlib import Path
from typing import Tuple
from mrx.DifferentialForms import DifferentialForm, DiscreteFunction, Pullback
from mrx.Quadrature import QuadratureRule
from mrx.Projectors import Projector, CurlProjection
from mrx.LazyMatrices import LazyMassMatrix, LazyDerivativeMatrix, LazyProjectionMatrix, LazyDoubleCurlMatrix
from mrx.Utils import curl

# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)

# Get the absolute path to the script's directory
script_dir = Path(__file__).parent.absolute()
# Create the output directory in the same directory as the script
output_dir = script_dir / 'script_outputs'
os.makedirs(output_dir, exist_ok=True)

# Initialize differential forms and operators
ns = (7, 7, 1)  # Number of elements in each direction
ps = (3, 3, 1)  # Polynomial degree in each direction
types = ('periodic', 'periodic', 'constant')  # Boundary conditions

# Define differential forms for different function spaces
Λ0 = DifferentialForm(0, ns, ps, types)  # H1 functions
Λ1 = DifferentialForm(1, ns, ps, types)  # H(curl) vector fields
Λ2 = DifferentialForm(2, ns, ps, types)  # H(div) vector fields
Λ3 = DifferentialForm(3, ns, ps, types)  # L2 densities

# Set up quadrature rule
Q = QuadratureRule(Λ0, 3)

# Identity mapping for the domain


def F(x): return x


# Assemble mass matrices for each space
M0, M1, M2, M3 = [LazyMassMatrix(Λ, Q).M for Λ in [Λ0, Λ1, Λ2, Λ3]]

# Set up projectors
P0, P1, P2, P3 = [Projector(Λ, Q) for Λ in [Λ0, Λ1, Λ2, Λ3]]
Pc = CurlProjection(Λ1, Q)  # Curl projection operator

# Assemble derivative operators
D0, D1, D2 = [LazyDerivativeMatrix(Λk, Λkplus1, Q).M
              for Λk, Λkplus1 in zip([Λ0, Λ1, Λ2], [Λ1, Λ2, Λ3])]

# Set up projection matrices between spaces
M12 = LazyProjectionMatrix(Λ1, Λ2, Q, F).M.T  # H(curl) to H(div)
M03 = LazyProjectionMatrix(Λ0, Λ3, Q, F).M.T  # H1 to L2

# Assemble double curl operator
C = LazyDoubleCurlMatrix(Λ1, Q).M


def l2_product(f, g, Q):
    """Compute L2 inner product of two functions using quadrature."""
    return jnp.einsum("ij,ij,i->", jax.vmap(f)(Q.x), jax.vmap(g)(Q.x), Q.w)


def E(x, m, n):
    """
    Compute the electric field components.

    Args:
        x: Position vector (r, χ, z)
        m: First mode number
        n: Second mode number

    Returns:
        jnp.ndarray: Electric field components
    """
    r, χ, z = x
    h = (1 + 0.0 * jnp.exp(-((r - 0.5)**2 + (χ - 0.5)**2) / 0.3**2))
    a1 = jnp.sin(m * jnp.pi * r) * jnp.cos(n * jnp.pi * χ) * jnp.sqrt(n**2/(n**2 + m**2))
    a2 = -jnp.cos(m * jnp.pi * r) * jnp.sin(n * jnp.pi * χ) * jnp.sqrt(m**2/(n**2 + m**2))
    a3 = jnp.sin(m * jnp.pi * r) * jnp.sin(n * jnp.pi * χ)
    return jnp.array([a1, a2, a3]) * h


def A(x):
    """Compute the vector potential for mode (2,2)."""
    return E(x, 2, 2)


# Compute initial magnetic field and its projection
B0 = curl(A)
B0_hat = jnp.linalg.solve(M2, P2(B0))

# Compute inverse of double curl operator
U, S, Vh = jnp.linalg.svd(C)
S_inv = jnp.where(S > 1e-6 * S[0] * S.shape[0], 1/S, 0)
C_inv = Vh.T @ jnp.diag(S_inv) @ U.T
A_hat_recon = C_inv @ D1.T @ B0_hat


@jax.jit
def force(B_hat):
    """Compute the Lorentz force J × B."""
    H_hat = jnp.linalg.solve(M1, M12 @ B_hat)
    J_hat = jnp.linalg.solve(M1, D1.T @ B_hat)
    J_h = DiscreteFunction(J_hat, Λ1)
    H_h = DiscreteFunction(H_hat, Λ1)

    def JcrossH(x):
        return jnp.cross(J_h(x), H_h(x))
    u_hat = jnp.linalg.solve(M2, P2(JcrossH))
    return u_hat


@jax.jit
def force_residual(B_hat):
    """Compute the L2 norm of the force residual."""
    u_hat = force(B_hat)
    return (u_hat @ M2 @ u_hat)**0.5


@jax.jit
def divergence_residual(B_hat):
    """Compute the L2 norm of the divergence of B."""
    divB = ((D2 @ B_hat) @ jnp.linalg.solve(M3, D2 @ B_hat))**0.5
    return divB


# Print initial conditions
print("Helicity before perturbation: ", A_hat_recon @ M12 @ B0_hat)
print("Energy before perturbation: ", B0_hat @ M2 @ B0_hat / 2)


def u(x):
    """Define a velocity field for perturbation."""
    r, χ, z = x
    a1 = jnp.sin(2 * jnp.pi * r) * jnp.cos(2 * jnp.pi * χ)
    a2 = jnp.cos(2 * jnp.pi * r) * jnp.sin(2 * jnp.pi * χ)
    a3 = jnp.sin(2 * jnp.pi * r) * jnp.cos(2 * jnp.pi * χ)
    return jnp.array([a1, a2, a3])


# Initialize velocity field
u_hat = jnp.linalg.solve(M2, P2(u))
u_h = DiscreteFunction(u_hat, Λ2)
B_hat = B0_hat


@jax.jit
def _perturb_B_hat(B_guess, B_hat_0, dt, u_hat):
    """Compute one step of the magnetic field perturbation."""
    H_hat = jnp.linalg.solve(M1, M12 @ (B_guess + B_hat_0)/2)
    H_h = DiscreteFunction(H_hat, Λ1)
    u_h = DiscreteFunction(u_hat, Λ2)
    E_hat = jnp.linalg.solve(M1, Pc(H_h, u_h))
    ẟB_hat = jnp.linalg.solve(M2, D1 @ E_hat)
    B_hat_1 = B_hat_0 + dt * ẟB_hat
    return B_hat_1


@jax.jit
def perturb_B_hat(B_hat_0, dt, key):
    """Iteratively compute magnetic field perturbation."""
    u_hat = jax.random.normal(key, shape=B_hat_0.shape)

    def cond_fun(B_guess):
        B_hat_1 = _perturb_B_hat(B_guess, B_hat_0, dt, u_hat)
        err = ((B_hat_1 - B_guess) @ M2 @ (B_hat_1 - B_guess))**0.5
        return err > 1e-12

    def body_fun(B_guess):
        B_hat_1 = _perturb_B_hat(B_guess, B_hat_0, dt, u_hat)
        return B_hat_1
    B_hat = jax.lax.while_loop(cond_fun, body_fun, B_hat_0)
    return B_hat


@jax.jit
def f(B_hat, key):
    B_hat = perturb_B_hat(B_hat, 1e-4, key)

    helicity = (C_inv @ D1.T @ B_hat) @ M12 @ B_hat
    energy = B_hat @ M2 @ B_hat / 2
    divB = divergence_residual(B_hat)
    normF = force_residual(B_hat)
    return B_hat, (helicity, energy, divB, normF)


# Initialize random number generator
key = jax.random.PRNGKey(0)
traces = []
BN_hat = B0_hat

# Compute perturbations
for key in jax.random.split(key, 3):
    BN_hat, trace = jax.lax.scan(f, BN_hat, jax.random.split(key, 10))
    traces.append(trace)

# Combine traces
trace = jnp.hstack(jnp.array(traces))
helicity, energy, divB, normF = trace

# Create figures
figures = []

# Energy evolution
fig1 = plt.figure(figsize=(10, 6))
plt.plot(energy - energy[0], label='Energy')
plt.xlabel('Iteration')
plt.ylabel('Energy Change')
plt.title('Energy Evolution')
plt.legend()
plt.grid(True)
figures.append(fig1)
plt.savefig(output_dir / 'energy_evolution.png', dpi=300, bbox_inches='tight')

# Helicity evolution
fig2 = plt.figure(figsize=(10, 6))
plt.plot(helicity - helicity[0], label='Helicity')
plt.xlabel('Iteration')
plt.ylabel('Helicity Change')
plt.title('Helicity Evolution')
plt.legend()
plt.grid(True)
figures.append(fig2)
plt.savefig(output_dir / 'helicity_evolution.png', dpi=300, bbox_inches='tight')

# Divergence evolution
fig3 = plt.figure(figsize=(10, 6))
plt.plot(divB - divB[0], label='|Div B|')
plt.xlabel('Iteration')
plt.ylabel('Divergence Change')
plt.title('Divergence Evolution')
plt.legend()
plt.grid(True)
figures.append(fig3)
plt.savefig(output_dir / 'divergence_evolution.png', dpi=300, bbox_inches='tight')

# Force evolution
fig4 = plt.figure(figsize=(10, 6))
plt.plot(normF, label='|F|')
plt.xlabel('Iteration')
plt.ylabel('Force Magnitude')
plt.title('Force Evolution')
plt.legend()
plt.grid(True)
figures.append(fig4)
plt.savefig(output_dir / 'force_evolution.png', dpi=300, bbox_inches='tight')

# Additional analysis parameters
b = 0.0
dt = 1e-5


@jax.jit
def ẟB_hat(B_guess, B_hat_0, u_hat_0):
    """Compute the change in magnetic field."""
    H_hat = jnp.linalg.solve(M1, M12 @ (B_guess + B_hat_0)/2)  # H = Proj(B)
    J_hat = jnp.linalg.solve(M1, D1.T @ (B_guess + B_hat_0)/2)  # J = curl H
    J_h = DiscreteFunction(J_hat, Λ1)
    H_h = DiscreteFunction(H_hat, Λ1)

    def JcrossH(x):
        return jnp.cross(J_h(x), H_h(x))
    u_hat = jnp.linalg.solve(M2, P2(JcrossH))           # u = J x H
    if b != 0:
        F_1 = u_hat @ M2 @ u_hat
        F_0 = u_hat_0 @ M2 @ u_hat_0
        u_hat += b * F_1/F_0 * u_hat_0
    u_h = DiscreteFunction(u_hat, Λ2)
    E_hat = jnp.linalg.solve(M1, Pc(H_h, u_h))          # E = u x H
    ẟB_hat = jnp.linalg.solve(M2, D1 @ E_hat)           # ẟB = curl E
    B_hat_1 = B_hat_0 + dt * ẟB_hat
    B_diff = B_hat_1 - B_guess
    return B_diff


@jax.jit
def compute_update(B_hat: jnp.ndarray, key: jnp.ndarray) -> Tuple[jnp.ndarray, Tuple[float, float, float, float]]:
    """Compute a single update step."""
    B_hat = B_hat + 0.1 * jax.random.normal(key, shape=B_hat.shape)
    B_hat = B_hat / jnp.linalg.norm(B_hat)
    return B_hat, (0.0, 0.0, 0.0, 0.0)


@jax.jit
def compute_state(B_hat: jnp.ndarray, key: jnp.ndarray) -> Tuple[jnp.ndarray, Tuple[float, float, float, float]]:
    """Compute the state at a single time step."""
    B_hat = B_hat + 0.1 * jax.random.normal(key, shape=B_hat.shape)
    B_hat = B_hat / jnp.linalg.norm(B_hat)
    return B_hat, (0.0, 0.0, 0.0, 0.0)


# Initialize state
x = jnp.concatenate((BN_hat, force(BN_hat)), axis=0)
traces = []

# Compute state evolution
for i in range(1):
    x, trace = jax.lax.scan(compute_state, x, jnp.arange(50))
    B_hat, u_hat = jnp.split(x, 2)
    traces.append(trace)

# Combine traces
trace = jnp.hstack(jnp.array(traces))
__helicity, __energy, __divB, __force_res = trace

# Create additional figures
figures = []

# Energy evolution (log scale)
fig5 = plt.figure(figsize=(10, 6))
plt.plot(__energy - B0_hat @ M2 @ B0_hat / 2)
plt.xlabel('Iteration')
plt.ylabel('Energy - Energy(0)')
plt.yscale('log')
plt.title('Energy Evolution (Log Scale)')
plt.grid(True)
figures.append(fig5)
plt.savefig(output_dir / 'energy_evolution_log.png', dpi=300, bbox_inches='tight')

# Helicity evolution
fig6 = plt.figure(figsize=(10, 6))
plt.plot(__helicity - __helicity[0])
plt.xlabel('Iteration')
plt.ylabel('Helicity - Helicity(0)')
plt.title('Helicity Evolution')
plt.grid(True)
figures.append(fig6)
plt.savefig(output_dir / 'helicity_evolution_2.png', dpi=300, bbox_inches='tight')

# Divergence evolution
fig7 = plt.figure(figsize=(10, 6))
plt.plot(__divB)
plt.xlabel('Iteration')
plt.ylabel('|Div B|')
plt.title('Divergence Evolution')
plt.grid(True)
figures.append(fig7)
plt.savefig(output_dir / 'divergence_evolution_2.png', dpi=300, bbox_inches='tight')

# Force evolution (log scale)
fig8 = plt.figure(figsize=(10, 6))
plt.plot(__force_res)
plt.xlabel('Iteration')
plt.ylabel('| J x B |')
plt.yscale('log')
plt.title('Force Evolution (Log Scale)')
plt.grid(True)
figures.append(fig8)
plt.savefig(output_dir / 'force_evolution_log.png', dpi=300, bbox_inches='tight')

# Print final state information
print("B(0) - B(1): ", ((B0_hat - BN_hat) @ M2 @ (B0_hat - BN_hat) / (B0_hat @ M2 @ B0_hat))**0.5)
print("B(0) - B(T): ", ((B0_hat - B_hat) @ M2 @ (B0_hat - B_hat) / (B0_hat @ M2 @ B0_hat))**0.5)
print("F(0): ", force_residual(B0_hat))
print("F(1): ", force_residual(BN_hat))
print("F(T): ", force_residual(B_hat))
print("E(0): ", B0_hat @ M2 @ B0_hat / 2)
print("E(1): ", B0_hat @ M2 @ BN_hat / 2)
print("E(T): ", B_hat @ M2 @ B_hat / 2)

# Set up grid for field visualization
ɛ = 1e-5
nx = 64
_x1 = jnp.linspace(ɛ, 1-ɛ, nx)
_x2 = jnp.linspace(ɛ, 1-ɛ, nx)
_x3 = jnp.ones(1)/2
_x = jnp.array(jnp.meshgrid(_x1, _x2, _x3))
_x = _x.transpose(1, 2, 3, 0).reshape(nx*nx*1, 3)
_y = jax.vmap(F)(_x)
_y1 = _y[:, 0].reshape(nx, nx)
_y2 = _y[:, 1].reshape(nx, nx)

# Set up grid for quiver plot
_nx = 16
__x1 = jnp.linspace(ɛ, 1-ɛ, _nx)
__x2 = jnp.linspace(ɛ, 1-ɛ, _nx)
__x3 = jnp.ones(1)/2
__x = jnp.array(jnp.meshgrid(__x1, __x2, __x3))
__x = __x.transpose(1, 2, 3, 0).reshape(_nx*_nx*1, 3)
__y = jax.vmap(F)(__x)
__y1 = __y[:, 0].reshape(_nx, _nx)
__y2 = __y[:, 1].reshape(_nx, _nx)

# Plot initial field
fig9 = plt.figure(figsize=(10, 8))
B_h = DiscreteFunction(B0_hat, Λ2)
F_B = Pullback(B0, F, 2)
F_B_h = Pullback(B_h, F, 2)
_z1 = jax.vmap(F_B_h)(_x).reshape(nx, nx, 3)
_z1_norm = jnp.linalg.norm(_z1, axis=2)
_z2 = jax.vmap(F_B)(_x).reshape(nx, nx, 3)
_z2_norm = jnp.linalg.norm(_z2, axis=2)
plt.contourf(_y1, _y2, _z1_norm.reshape(nx, nx))
plt.clim(0, 20)
plt.colorbar()
plt.contour(_y1, _y2, _z2_norm.reshape(nx, nx), colors='k')
__z1 = jax.vmap(F_B_h)(__x).reshape(_nx, _nx, 3)
plt.quiver(__y1, __y2, __z1[:, :, 0], __z1[:, :, 1], color='w')
plt.title('Initial Magnetic Field')
plt.xlabel('x')
plt.ylabel('y')
figures.append(fig9)
plt.savefig(output_dir / 'initial_field.png', dpi=300, bbox_inches='tight')

# Plot perturbed field
fig10 = plt.figure(figsize=(10, 8))
B_h = DiscreteFunction(B0_hat, Λ2)
F_B = Pullback(B0, F, 2)
F_B_h = Pullback(B_h, F, 2)
_z1 = jax.vmap(F_B_h)(_x).reshape(nx, nx, 3)
_z1_norm = jnp.linalg.norm(_z1, axis=2)
_z2 = jax.vmap(F_B)(_x).reshape(nx, nx, 3)
_z2_norm = jnp.linalg.norm(_z2, axis=2)
plt.contourf(_y1, _y2, _z1_norm.reshape(nx, nx))
plt.clim(0, 20)
plt.colorbar()
plt.contour(_y1, _y2, _z2_norm.reshape(nx, nx), colors='k')
__z1 = jax.vmap(F_B_h)(__x).reshape(_nx, _nx, 3)
plt.quiver(__y1, __y2, __z1[:, :, 0], __z1[:, :, 1], color='w')
plt.title('Perturbed Magnetic Field')
plt.xlabel('x')
plt.ylabel('y')
figures.append(fig10)
plt.savefig('script_outputs/perturbed_field.png', dpi=300, bbox_inches='tight')

# Plot final field
fig11 = plt.figure(figsize=(10, 8))
B_h = DiscreteFunction(B_hat, Λ2)
F_B = Pullback(B0, F, 2)
F_B_h = Pullback(B_h, F, 2)
_z1 = jax.vmap(F_B_h)(_x).reshape(nx, nx, 3)
_z1_norm = jnp.linalg.norm(_z1, axis=2)
_z2 = jax.vmap(F_B)(_x).reshape(nx, nx, 3)
_z2_norm = jnp.linalg.norm(_z2, axis=2)
plt.contourf(_y1, _y2, _z1_norm.reshape(nx, nx))
plt.colorbar()
plt.clim(0, 20)
plt.contour(_y1, _y2, _z2_norm.reshape(nx, nx), colors='k')
__z1 = jax.vmap(F_B_h)(__x).reshape(_nx, _nx, 3)
plt.quiver(__y1, __y2, __z1[:, :, 0], __z1[:, :, 1], color='w')
plt.title('Final Magnetic Field')
plt.xlabel('x')
plt.ylabel('y')
figures.append(fig11)
plt.savefig('script_outputs/final_field.png', dpi=300, bbox_inches='tight')

# Plot velocity field
fig12 = plt.figure(figsize=(10, 8))
B_h = DiscreteFunction(u_hat, Λ2)
F_B = Pullback(B0, F, 2)
F_B_h = Pullback(B_h, F, 2)
_z1 = jax.vmap(F_B_h)(_x).reshape(nx, nx, 3)
_z1_norm = jnp.linalg.norm(_z1, axis=2)
_z2 = jax.vmap(F_B)(_x).reshape(nx, nx, 3)
_z2_norm = jnp.linalg.norm(_z2, axis=2)
plt.contourf(_y1, _y2, _z1_norm.reshape(nx, nx))
plt.colorbar()
__z1 = jax.vmap(F_B_h)(__x).reshape(_nx, _nx, 3)
plt.quiver(__y1, __y2, __z1[:, :, 0], __z1[:, :, 1], color='w')
plt.title('Velocity Field')
plt.xlabel('x')
plt.ylabel('y')
figures.append(fig12)
plt.savefig('script_outputs/velocity_field.png', dpi=300, bbox_inches='tight')

# Show all figures at the end
plt.show()

# Clean up
plt.close('all')
