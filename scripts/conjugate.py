"""
Conjugate Gradient Method for Magnetic Field Relaxation

This script implements a conjugate gradient method for relaxing magnetic fields while preserving
magnetic helicity. It includes:

1. Initial field setup and perturbation
2. Conjugate gradient relaxation
3. Error analysis and visualization
4. Conservation properties verification

The script demonstrates:
- Magnetic field relaxation to force-free states
- Helicity conservation during relaxation
- Divergence-free constraint maintenance
- Energy evolution during relaxation

Key components:
- Differential forms for field representation
- Lazy matrices for efficient computation
- Conjugate gradient method for relaxation
- Visualization of field properties
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path

from mrx.DifferentialForms import DifferentialForm, DiscreteFunction, Pullback
from mrx.Quadrature import QuadratureRule
from mrx.Projectors import Projector, CurlProjection
from mrx.LazyMatrices import LazyMassMatrix, LazyDerivativeMatrix, LazyProjectionMatrix, LazyDoubleCurlMatrix, LazyStiffnessMatrix
from mrx.Utils import curl

# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)

# Create output directory for figures
output_dir = Path("script_outputs")
output_dir.mkdir(parents=True, exist_ok=True)

# Define mesh parameters
ns = (7, 7, 1)  # Number of elements in each direction
ps = (3, 3, 1)  # Polynomial degrees
types = ('periodic', 'periodic', 'constant')  # Boundary condition types

# Initialize differential forms
Λ0, Λ1, Λ2, Λ3 = [DifferentialForm(i, ns, ps, types) for i in range(4)]  # H1, H(curl), H(div), L2
Q = QuadratureRule(Λ0, 4)  # Quadrature rule
def F(x): return x  # Identity mapping


# Assemble matrices
M0, M1, M2, M3 = [LazyMassMatrix(Λ, Q).M for Λ in [Λ0, Λ1, Λ2, Λ3]]  # Mass matrices
P0, P1, P2, P3 = [Projector(Λ, Q) for Λ in [Λ0, Λ1, Λ2, Λ3]]  # L2 projectors
Pc = CurlProjection(Λ1, Q)  # Curl projection operator
D0, D1, D2 = [LazyDerivativeMatrix(Λk, Λkplus1, Q).M
              for Λk, Λkplus1 in zip([Λ0, Λ1, Λ2], [Λ1, Λ2, Λ3])]  # Derivative matrices
M12 = LazyProjectionMatrix(Λ1, Λ2, Q, F).M.T  # Projection from H(curl) to H(div)
M03 = LazyProjectionMatrix(Λ0, Λ3, Q, F).M.T  # Projection from H1 to L2
C = LazyDoubleCurlMatrix(Λ1, Q).M  # Double curl operator
K = LazyStiffnessMatrix(Λ0, Q).M  # Stiffness matrix


def l2_product(f, g, Q):
    """Compute the L2 inner product of two functions.

    Args:
        f: First function
        g: Second function
        Q: Quadrature rule

    Returns:
        float: L2 inner product (f,g)
    """
    return jnp.einsum("ij,ij,i->", jax.vmap(f)(Q.x), jax.vmap(g)(Q.x), Q.w)


def E(x, m, n):
    """Compute the electric field components.

    Args:
        x: Position vector (r, χ, z)
        m: First mode number
        n: Second mode number

    Returns:
        jnp.ndarray: Electric field components [E1, E2, E3]
    """
    r, χ, z = x
    h = (1 + 0.0 * jnp.exp(-((r - 0.5)**2 + (χ - 0.5)**2) / 0.3**2))
    a1 = jnp.sin(m * jnp.pi * r) * jnp.cos(n * jnp.pi * χ) * jnp.sqrt(n**2/(n**2 + m**2))
    a2 = -jnp.cos(m * jnp.pi * r) * jnp.sin(n * jnp.pi * χ) * jnp.sqrt(m**2/(n**2 + m**2))
    a3 = jnp.sin(m * jnp.pi * r) * jnp.sin(n * jnp.pi * χ)
    return jnp.array([a1, a2, a3]) * h


def A(x):
    """Compute the vector potential.

    Args:
        x: Position vector (r, χ, z)

    Returns:
        jnp.ndarray: Vector potential components
    """
    return E(x, 2, 2)


# Initialize magnetic field
B0 = curl(A)
B0_hat = jnp.linalg.solve(M2, P2(B0))

# Compute SVD of double curl operator
U, S, Vh = jnp.linalg.svd(C)
S_inv = jnp.where(S > 1e-6 * S[0] * S.shape[0], 1/S, 0)
C_inv = Vh.T @ jnp.diag(S_inv) @ U.T
A_hat_recon = C_inv @ D1.T @ B0_hat


@jax.jit
def force(B_hat):
    """Compute the Lorentz force.

    Args:
        B_hat: Magnetic field coefficients

    Returns:
        jnp.ndarray: Force field coefficients
    """
    H_hat = jnp.linalg.solve(M1, M12 @ B_hat)
    J_hat = jnp.linalg.solve(M1, D1.T @ B_hat)
    J_h = DiscreteFunction(J_hat, Λ1)
    H_h = DiscreteFunction(H_hat, Λ1)

    def JcrossH(x):
        return jnp.cross(J_h(x), H_h(x))
    u_hat = jnp.linalg.solve(M2, P2(JcrossH))  # u = J x H
    return u_hat


@jax.jit
def force_residual(B_hat):
    """Compute the force residual norm.

    Args:
        B_hat: Magnetic field coefficients

    Returns:
        float: Force residual norm
    """
    u_hat = force(B_hat)
    return (u_hat @ M2 @ u_hat)**0.5


@jax.jit
def divergence_residual(B_hat):
    """Compute the divergence residual norm.

    Args:
        B_hat: Magnetic field coefficients

    Returns:
        float: Divergence residual norm
    """
    divB = ((D2 @ B_hat) @ jnp.linalg.solve(M3, D2 @ B_hat))**0.5
    return divB


# Print initial conditions
print("Helicity before perturbation: ", A_hat_recon @ M12 @ B0_hat)
print("Energy before perturbation: ", B0_hat @ M2 @ B0_hat / 2)


def u(x):
    """Compute the velocity field.

    Args:
        x: Position vector (r, χ, z)

    Returns:
        jnp.ndarray: Velocity components
    """
    r, χ, z = x
    a1 = jnp.sin(2 * jnp.pi * r) * jnp.cos(2 * jnp.pi * χ)
    a2 = jnp.cos(2 * jnp.pi * r) * jnp.sin(2 * jnp.pi * χ)
    a3 = jnp.sin(2 * jnp.pi * r) * jnp.cos(2 * jnp.pi * χ)
    return jnp.array([a1, a2, a3])


u_hat = jnp.linalg.solve(M2, P2(u))
u_h = DiscreteFunction(u_hat, Λ2)
B_hat = B0_hat


@jax.jit
def twoformnorm(B):
    """Compute the L2 norm of a 2-form.

    Args:
        B: 2-form coefficients

    Returns:
        float: L2 norm
    """
    return (B @ M2 @ B)**0.5


@jax.jit
def ẟB(B_guess, B_n, u_hat):
    """Compute the magnetic field update.

    Args:
        B_guess: Current guess for magnetic field
        B_n: Previous magnetic field
        u_hat: Velocity field coefficients

    Returns:
        jnp.ndarray: Magnetic field update
    """
    H_hat = jnp.linalg.solve(M1, M12 @ (B_guess + B_n)/2)  # H = Proj(B)
    H_h = DiscreteFunction(H_hat, Λ1)
    u_h = DiscreteFunction(u_hat, Λ2)
    E_hat = jnp.linalg.solve(M1, Pc(H_h, u_h))  # E = u x H
    deltaB = jnp.linalg.solve(M2, D1 @ E_hat)  # ẟB = curl E
    return deltaB


@jax.jit
def advect_B(B_n, u_n, dt):
    """Advect the magnetic field.

    Args:
        B_n: Current magnetic field
        u_n: Velocity field
        dt: Time step

    Returns:
        jnp.ndarray: Advected magnetic field
    """
    # def f(B):
    return B_n + dt * ẟB(B_n, B_n, u_n)
    # return f(B_n)


@jax.jit
def f_perturb(B_hat, key):
    """Perturb the magnetic field while preserving helicity.

    Args:
        B_hat: Magnetic field coefficients
        key: Random key for perturbation

    Returns:
        tuple: (B_hat, (helicity, energy, divB, normF))
    """
    u_hat = jax.random.normal(key, shape=B_hat.shape)
    B_hat = advect_B(B_hat, u_hat, 1e-4)

    helicity = (C_inv @ D1.T @ B_hat) @ M12 @ B_hat
    energy = B_hat @ M2 @ B_hat / 2
    divB = divergence_residual(B_hat)
    normF = force_residual(B_hat)

    return B_hat, (helicity, energy, divB, normF)


# Perform initial perturbations
key = jax.random.PRNGKey(0)
traces = []
BN_hat = B0_hat

for key in jax.random.split(key, 50):
    BN_hat, trace = f_perturb(BN_hat, key)
    traces.append(trace)

trace = jnp.vstack(jnp.array(traces))
helicity, energy, divB, normF = trace.T

# Plot and save results
plt.figure(figsize=(10, 6))
plt.plot(energy - energy[0], label='Energy')
plt.xlabel('Iteration')
plt.ylabel('Energy Change')
plt.title('Energy Evolution')
plt.legend()
plt.grid(True)
plt.savefig(output_dir / 'energy_evolution.png', dpi=300, bbox_inches='tight')

plt.figure(figsize=(10, 6))
plt.plot(helicity - helicity[0], label='Helicity')
plt.xlabel('Iteration')
plt.ylabel('Helicity Change')
plt.title('Helicity Evolution')
plt.legend()
plt.grid(True)
plt.savefig(output_dir / 'helicity_evolution.png', dpi=300, bbox_inches='tight')

plt.figure(figsize=(10, 6))
plt.plot(divB - divB[0], label='|Div B|')
plt.xlabel('Iteration')
plt.ylabel('Divergence Change')
plt.title('Divergence Evolution')
plt.legend()
plt.grid(True)
plt.savefig(output_dir / 'divergence_evolution.png', dpi=300, bbox_inches='tight')

plt.figure(figsize=(10, 6))
plt.plot(normF, label='|F|')
plt.xlabel('Iteration')
plt.ylabel('Force Norm')
plt.title('Force Evolution')
plt.legend()
plt.grid(True)
plt.savefig(output_dir / 'force_evolution.png', dpi=300, bbox_inches='tight')

# Relaxation parameters
b = 0.9
ds = 1e-6
a = 1.0


@jax.jit
def f_relax(x, key):
    """Relax the magnetic field using conjugate gradient method.

    Args:
        x: Tuple of (B_n, u_nminus1, normF_nminus1)
        key: Random key

    Returns:
        tuple: (x, (helicity, energy, divB, normF_n))
    """
    B_n, u_nminus1, normF_nminus1 = x

    F_n = force(B_n)
    normF_n = F_n @ M2 @ F_n

    u_n = F_n + b * normF_n / normF_nminus1 * u_nminus1

    B_s = B_n + ds * ẟB(B_n, B_n, u_n)

    F_s = force(B_s)
    ẟW_s = F_s @ M2 @ u_n
    ẟW_n = F_n @ M2 @ u_n

    dt = - ds * a * ẟW_n / (ẟW_s - ẟW_n)

    B_n = advect_B(B_n, u_n, dt)

    helicity = (C_inv @ D1.T @ B_n) @ M12 @ B_n
    energy = B_n @ M2 @ B_n / 2
    divB = divergence_residual(B_n)

    x = B_n, u_n, normF_n

    return x, (helicity, energy, divB, normF_n)


# First analysis run with fixed timesteps
key = jax.random.PRNGKey(0)
B_hat = BN_hat
traces = []

# Initialize state
x = B_hat, jnp.zeros_like(B_hat), 1.0
# precompile f_relax
x, trace = f_relax(x, 0)

for _ in range(50):  # Remove key splitting since we don't use it
    x, trace = f_relax(x, key)  # Use the updated x from previous iteration
    normF = trace[-1]
    if normF < 1e-12:
        break
    traces.append(trace)

# Process and plot results from first run
trace_array = jnp.vstack(jnp.array(traces))
helicity, energy, divB, normF = trace_array.T
base_energy = B0_hat @ M2 @ B0_hat / 2

# Plot evolution of quantities during first run
plt.figure(figsize=(10, 6))
plt.plot(energy - base_energy, label='Energy')
plt.xlabel('Iteration')
plt.ylabel('Energy Change')
plt.title('Energy Evolution (Fixed Timesteps)')
plt.legend()
plt.grid(True)
plt.savefig(output_dir / 'fixed_timestep_energy.png', dpi=300, bbox_inches='tight')

plt.figure(figsize=(10, 6))
plt.plot(helicity - helicity[0], label='Helicity')
plt.xlabel('Iteration')
plt.ylabel('Helicity Change')
plt.title('Helicity Evolution (Fixed Timesteps)')
plt.legend()
plt.grid(True)
plt.savefig(output_dir / 'fixed_timestep_helicity.png', dpi=300, bbox_inches='tight')

plt.figure(figsize=(10, 6))
plt.plot(divB - divB[0], label='|Div B|')
plt.xlabel('Iteration')
plt.ylabel('Divergence Change')
plt.title('Divergence Evolution (Fixed Timesteps)')
plt.legend()
plt.grid(True)
plt.savefig(output_dir / 'fixed_timestep_divergence.png', dpi=300, bbox_inches='tight')

plt.figure(figsize=(10, 6))
plt.plot(normF, label='|F|')
plt.xlabel('Iteration')
plt.ylabel('Force Norm')
plt.title('Force Evolution (Fixed Timesteps)')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.savefig(output_dir / 'fixed_timestep_force.png', dpi=300, bbox_inches='tight')

# Print diagnostics from first run
print("\nFirst Run Diagnostics (Fixed Timesteps):")
print(f"Initial to Perturbed B difference: {((B0_hat - BN_hat) @ M2 @ (B0_hat - BN_hat) / (B0_hat @ M2 @ B0_hat))**0.5:.2e}")
print(f"Initial to Final B difference: {((B0_hat - B_hat) @ M2 @ (B0_hat - B_hat) / (B0_hat @ M2 @ B0_hat))**0.5:.2e}")
print(f"Initial force: {force_residual(B0_hat):.2e}")
print(f"Perturbed force: {force_residual(BN_hat):.2e}")
print(f"Final force: {force_residual(B_hat):.2e}")
print(f"Initial energy: {B0_hat @ M2 @ B0_hat / 2:.2e}")
print(f"Perturbed energy: {BN_hat @ M2 @ BN_hat / 2:.2e}")
print(f"Final energy: {B_hat @ M2 @ B_hat / 2:.2e}")
print("\nFirst analysis completed successfully!")
print(f"Figures saved to: {output_dir}")

# Second analysis run with non-equispaced timesteps


@jax.jit
def f_relax(B_n, dt):
    """Perform a single relaxation step with given timestep.

    Args:
        B_n: Current magnetic field
        dt: Timestep size

    Returns:
        tuple: (Updated field, (helicity, energy, divergence, force norm))
    """
    F_n = force(B_n)
    B_n = advect_B(B_n, F_n, dt)

    helicity = (C_inv @ D1.T @ B_n) @ M12 @ B_n
    energy = B_n @ M2 @ B_n / 2
    divB = divergence_residual(B_n)
    normF_n = force_residual(B_n)

    return B_n, (helicity, energy, divB, normF_n)


# Setup non-equispaced timesteps using Chebyshev nodes
N = 5000  # Total number of timesteps
k = N//3  # Parameter for Chebyshev nodes
j = jnp.arange(N+1)
nu = k * j
x_n = 1 + jnp.cos((2 * nu - 1) * jnp.pi / (2 * N))  # Chebyshev nodes
dt_0 = 1e-6  # Initial timestep
dt_n = [dt_0] * N  # Fixed timestep array
# dt_n = dt_0 / (x_n + 1/N**2)  # Alternative adaptive timestep formula

# Plot timestep distribution
plt.figure(figsize=(10, 6))
plt.plot(dt_n)
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Timestep Size')
plt.title('Timestep Distribution (Non-Equispaced)')
plt.grid(True)
plt.savefig(output_dir / 'non_equispaced_timesteps.png', dpi=300, bbox_inches='tight')

# Perform relaxation with non-equispaced timesteps
key = jax.random.PRNGKey(0)
B_hat = BN_hat
traces = []
# precompile f_relax
B_hat, trace = f_relax(B_hat, dt_n[0])

for dt in dt_n:
    B_hat, trace = f_relax(B_hat, dt)
    normF = trace[-1]
    if normF < 1e-12:
        break
    traces.append(trace)

# Process and plot results from second run
trace_array = jnp.vstack(jnp.array(traces))
helicity, energy, divB, normF = trace_array.T
base_energy = B0_hat @ M2 @ B0_hat / 2

# Plot evolution of quantities during second run
plt.figure(figsize=(10, 6))
plt.plot(energy - base_energy, label='Energy')
plt.xlabel('Iteration')
plt.ylabel('Energy Change')
plt.title('Energy Evolution (Non-Equispaced Timesteps)')
plt.legend()
plt.grid(True)
plt.savefig(output_dir / 'non_equispaced_energy.png', dpi=300, bbox_inches='tight')

plt.figure(figsize=(10, 6))
plt.plot(helicity - helicity[0], label='Helicity')
plt.xlabel('Iteration')
plt.ylabel('Helicity Change')
plt.title('Helicity Evolution (Non-Equispaced Timesteps)')
plt.legend()
plt.grid(True)
plt.savefig(output_dir / 'non_equispaced_helicity.png', dpi=300, bbox_inches='tight')

plt.figure(figsize=(10, 6))
plt.plot(divB - divB[0], label='|Div B|')
plt.xlabel('Iteration')
plt.ylabel('Divergence Change')
plt.title('Divergence Evolution (Non-Equispaced Timesteps)')
plt.legend()
plt.grid(True)
plt.savefig(output_dir / 'non_equispaced_divergence.png', dpi=300, bbox_inches='tight')

plt.figure(figsize=(10, 6))
plt.plot(normF, label='|F|')
plt.xlabel('Iteration')
plt.ylabel('Force Norm')
plt.title('Force Evolution (Non-Equispaced Timesteps)')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.savefig(output_dir / 'non_equispaced_force.png', dpi=300, bbox_inches='tight')

# Print diagnostics from second run
print("\nSecond Run Diagnostics (Non-Equispaced Timesteps):")
print(f"Initial to Perturbed B difference: {((B0_hat - BN_hat) @ M2 @ (B0_hat - BN_hat) / (B0_hat @ M2 @ B0_hat))**0.5:.2e}")
print(f"Initial to Final B difference: {((B0_hat - B_hat) @ M2 @ (B0_hat - B_hat) / (B0_hat @ M2 @ B0_hat))**0.5:.2e}")
print(f"Initial force: {force_residual(B0_hat):.2e}")
print(f"Perturbed force: {force_residual(BN_hat):.2e}")
print(f"Final force: {force_residual(B_hat):.2e}")
print(f"Initial energy: {B0_hat @ M2 @ B0_hat / 2:.2e}")
print(f"Perturbed energy: {BN_hat @ M2 @ BN_hat / 2:.2e}")
print(f"Final energy: {B_hat @ M2 @ B_hat / 2:.2e}")
print("\nSecond analysis completed successfully!")
print(f"Figures saved to: {output_dir}")

# Field visualization setup
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

# Visualize initial field
B_h = DiscreteFunction(B0_hat, Λ2)
F_B = Pullback(B0, F, 2)
F_B_h = Pullback(B_h, F, 2)
_z1 = jax.vmap(F_B_h)(_x).reshape(nx, nx, 3)
_z1_norm = jnp.linalg.norm(_z1, axis=2)
_z2 = jax.vmap(F_B)(_x).reshape(nx, nx, 3)
_z2_norm = jnp.linalg.norm(_z2, axis=2)

plt.figure(figsize=(10, 8))
plt.contourf(_y1, _y2, _z1_norm.reshape(nx, nx))
plt.clim(0, 20)
plt.colorbar(label='Field Magnitude')
plt.contour(_y1, _y2, _z2_norm.reshape(nx, nx), colors='k')
__z1 = jax.vmap(F_B_h)(__x).reshape(_nx, _nx, 3)
plt.quiver(__y1, __y2, __z1[:, :, 0], __z1[:, :, 1], color='w')
plt.title('Initial Magnetic Field')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig(output_dir / 'initial_field.png', dpi=300, bbox_inches='tight')

# Visualize perturbed field
B_h = DiscreteFunction(BN_hat, Λ2)
F_B = Pullback(B0, F, 2)
F_B_h = Pullback(B_h, F, 2)
_z1 = jax.vmap(F_B_h)(_x).reshape(nx, nx, 3)
_z1_norm = jnp.linalg.norm(_z1, axis=2)
_z2 = jax.vmap(F_B)(_x).reshape(nx, nx, 3)
_z2_norm = jnp.linalg.norm(_z2, axis=2)

plt.figure(figsize=(10, 8))
plt.contourf(_y1, _y2, _z1_norm.reshape(nx, nx))
plt.clim(0, 20)
plt.colorbar(label='Field Magnitude')
plt.contour(_y1, _y2, _z2_norm.reshape(nx, nx), colors='k')
__z1 = jax.vmap(F_B_h)(__x).reshape(_nx, _nx, 3)
plt.quiver(__y1, __y2, __z1[:, :, 0], __z1[:, :, 1], color='w')
plt.title('Perturbed Magnetic Field')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig(output_dir / 'perturbed_field.png', dpi=300, bbox_inches='tight')

# Visualize final field
B_h = DiscreteFunction(B_hat, Λ2)
F_B = Pullback(B0, F, 2)
F_B_h = Pullback(B_h, F, 2)
_z1 = jax.vmap(F_B_h)(_x).reshape(nx, nx, 3)
_z1_norm = jnp.linalg.norm(_z1, axis=2)
_z2 = jax.vmap(F_B)(_x).reshape(nx, nx, 3)
_z2_norm = jnp.linalg.norm(_z2, axis=2)

plt.figure(figsize=(10, 8))
plt.contourf(_y1, _y2, _z1_norm.reshape(nx, nx))
plt.clim(0, 20)
plt.colorbar(label='Field Magnitude')
plt.contour(_y1, _y2, _z2_norm.reshape(nx, nx), colors='k')
__z1 = jax.vmap(F_B_h)(__x).reshape(_nx, _nx, 3)
plt.quiver(__y1, __y2, __z1[:, :, 0], __z1[:, :, 1], color='w')
plt.title('Final Magnetic Field')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig(output_dir / 'final_field.png', dpi=300, bbox_inches='tight')

# Visualize force field
u_hat = force(B_hat)
B_h = DiscreteFunction(u_hat, Λ2)
F_B = Pullback(B0, F, 2)
F_B_h = Pullback(B_h, F, 2)
_z1 = jax.vmap(F_B_h)(_x).reshape(nx, nx, 3)
_z1_norm = jnp.linalg.norm(_z1, axis=2)
_z2 = jax.vmap(F_B)(_x).reshape(nx, nx, 3)
_z2_norm = jnp.linalg.norm(_z2, axis=2)

plt.figure(figsize=(10, 8))
plt.contourf(_y1, _y2, _z1_norm.reshape(nx, nx))
plt.colorbar(label='Force Magnitude')
__z1 = jax.vmap(F_B_h)(__x).reshape(_nx, _nx, 3)
plt.quiver(__y1, __y2, __z1[:, :, 0], __z1[:, :, 1], color='w')
plt.title('Final Force Field')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig(output_dir / 'final_force.png', dpi=300, bbox_inches='tight')

print("\nAll analyses completed successfully!")
print(f"Figures saved to: {output_dir}")
plt.show()
