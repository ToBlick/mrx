"""
Magnetic Relaxation Simulation

This script implements a magnetic relaxation simulation using finite element methods
with JAX for automatic differentiation. The simulation models the evolution of a
magnetic field under the influence of forces while preserving certain invariants
like helicity.

The code uses:
- Differential forms for representing fields
- Finite element discretization
- JAX for numerical computations
- Matplotlib for visualization

Key components:
- Magnetic field evolution
- Helicity preservation
- Energy minimization
- Force calculations
- Divergence cleaning

The simulation outputs include:
- Field evolution plots
- Energy and helicity time series
- Force magnitude plots
- Divergence error plots
"""

# %%
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

from mrx.DifferentialForms import DifferentialForm, DiscreteFunction, Pullback
from mrx.Quadrature import QuadratureRule
from mrx.Projectors import Projector, CurlProjection
from mrx.LazyMatrices import LazyMassMatrix, LazyDerivativeMatrix, LazyProjectionMatrix, LazyDoubleCurlMatrix, LazyStiffnessMatrix
from mrx.Utils import curl

# Ensure output directory exists
output_dir = Path("script_outputs")
output_dir.mkdir(exist_ok=True)

jax.config.update("jax_enable_x64", True)

# %% [markdown]
# ## Setup: Define Differential Forms and Operators
# Initialize the differential forms and operators needed for the simulation
# %%
ns = (8, 8, 1)
ps = (3, 3, 1)
types = ('periodic', 'periodic', 'constant')

Λ0 = DifferentialForm(0, ns, ps, types)  # functions in H1
Λ1 = DifferentialForm(1, ns, ps, types)  # vector fields in H(curl)
Λ2 = DifferentialForm(2, ns, ps, types)  # vector fields in H(div)
Λ3 = DifferentialForm(3, ns, ps, types)  # densities in L2
Q = QuadratureRule(Λ0, 10)              # Quadrature
def F(x): return x                         # identity mapping

# %% [markdown]
# ## Assemble Matrices and Operators
# Construct the necessary matrices for the finite element discretization
# %%
M0, M1, M2, M3 = [LazyMassMatrix(Λ, Q).M
                  for Λ in [Λ0, Λ1, Λ2, Λ3]]                  # assembled mass matries
P0, P1, P2, P3 = [Projector(Λ, Q)
                  for Λ in [Λ0, Λ1, Λ2, Λ3]]                 # L2 projectors
Pc = CurlProjection(Λ1, Q)                      # given A and B, computes (B, A x Λ[i])
D0, D1, D2 = [LazyDerivativeMatrix(Λk, Λkplus1, Q).M
              for Λk, Λkplus1 in zip([Λ0, Λ1, Λ2], [Λ1, Λ2, Λ3])]  # grad, curl, div
M12 = LazyProjectionMatrix(Λ1, Λ2, Q, F).M.T      # L2 projection from H(curl) to H(div)
M03 = LazyProjectionMatrix(Λ0, Λ3, Q, F).M.T      # L2 projection from H1 to L2
C = LazyDoubleCurlMatrix(Λ1, Q).M               # bilinear form (A, E) → (curl A, curl E)
K = LazyStiffnessMatrix(Λ0, Q).M                # bilinear form (q, p) → (grad q, grad p)

# %% [markdown]
# ## Helper Functions
# %%
def l2_product(f, g, Q):
    """Compute the L2 inner product of two functions f and g over the domain.
    
    Args:
        f: First function
        g: Second function
        Q: Quadrature rule
        
    Returns:
        float: L2 inner product
    """
    return jnp.einsum("ij,ij,i->", jax.vmap(f)(Q.x), jax.vmap(g)(Q.x), Q.w)

# %% [markdown]
# ## Initial Field Setup
# Define the initial magnetic field configuration
# %%
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
_nx = 16
__x1 = jnp.linspace(ɛ, 1-ɛ, _nx)
__x2 = jnp.linspace(ɛ, 1-ɛ, _nx)
__x3 = jnp.ones(1)/2
__x = jnp.array(jnp.meshgrid(__x1, __x2, __x3))
__x = __x.transpose(1, 2, 3, 0).reshape(_nx*_nx*1, 3)
__y = jax.vmap(F)(__x)
__y1 = __y[:, 0].reshape(_nx, _nx)
__y2 = __y[:, 1].reshape(_nx, _nx)

def E(x, m, n):
    """Define the initial magnetic field configuration.
    
    Args:
        x: Spatial coordinates (r, χ, z)
        m: Mode number in r direction
        n: Mode number in χ direction
        
    Returns:
        jnp.array: Magnetic field vector
    """
    r, χ, z = x
    h = (1 + 0.0 * jnp.exp(-((r - 0.5)**2 + (χ - 0.5)**2) / 0.3**2))
    a1 = jnp.sin(m * jnp.pi * r) * jnp.cos(n * jnp.pi * χ) * jnp.sqrt(n**2/(n**2 + m**2))
    a2 = -jnp.cos(m * jnp.pi * r) * jnp.sin(n * jnp.pi * χ) * jnp.sqrt(m**2/(n**2 + m**2))
    a3 = jnp.sin(m * jnp.pi * r) * jnp.sin(n * jnp.pi * χ)
    return jnp.array([a1, a2, a3]) * h

def A(x):
    """Initial vector potential.
    
    Args:
        x: Spatial coordinates
        
    Returns:
        jnp.array: Vector potential
    """
    return E(x, 2, 2)

# %% [markdown]
# ## Initial Field Visualization
# %%
print(l2_product(A, curl(A), Q))
# %%
F_A = Pullback(A, F, 1)

_z1 = jax.vmap(F_A)(_x).reshape(nx, nx, 3)
_z1_norm = jnp.linalg.norm(_z1, axis=2)
plt.figure(figsize=(10, 8))
plt.contourf(_y1, _y2, _z1_norm.reshape(nx, nx))
plt.colorbar()
__z1 = jax.vmap(F_A)(__x).reshape(_nx, _nx, 3)
plt.quiver(
    __y1,
    __y2,
    __z1[:, :, 0],
    __z1[:, :, 1],
    color='w')
plt.title('Initial Magnetic Field')
plt.savefig(output_dir / 'initial_field.png')


# %% [markdown]
# ## Field Projection and Error Analysis
# %%
A_hat = jnp.linalg.solve(M1, P1(A))
A_h = DiscreteFunction(A_hat, Λ1)
def compute_A_error(x): return A(x) - A_h(x)

(l2_product(compute_A_error, compute_A_error, Q) / l2_product(A, A, Q))**0.5

# %% [markdown]
# ## Field Evolution Visualization
# %%
F_A_h = Pullback(A_h, F, 1)
_z1 = jax.vmap(F_A_h)(_x).reshape(nx, nx, 3)
_z1_norm = jnp.linalg.norm(_z1, axis=2)
_z2 = jax.vmap(F_A)(_x).reshape(nx, nx, 3)
_z2_norm = jnp.linalg.norm(_z2, axis=2)
plt.figure(figsize=(10, 8))
plt.contourf(_y1, _y2, _z1_norm.reshape(nx, nx))
plt.colorbar()
plt.contour(_y1, _y2, _z2_norm.reshape(nx, nx), colors='k')
__z1 = jax.vmap(F_A_h)(__x).reshape(_nx, _nx, 3)
plt.quiver(
    __y1,
    __y2,
    __z1[:, :, 0],
    __z1[:, :, 1],
    color='w')
plt.title('Field Evolution')
plt.savefig(output_dir / 'field_evolution.png')


# %% [markdown]
# ## Magnetic Field and Error Analysis
# Compute the magnetic field from the vector potential and analyze errors
# %%
B0 = curl(A)  # Compute the initial magnetic field
B0_hat = jnp.linalg.solve(M2, P2(B0))  # Project onto the discrete space
B_h = DiscreteFunction(B0_hat, Λ2)  # Create discrete function
B0_h = DiscreteFunction(B0_hat, Λ2)  # Store initial state

def compute_B_error(x): return B0(x) - B_h(x)  # Compute pointwise error

# Compute relative L2 error in magnetic field
(l2_product(compute_B_error, compute_B_error, Q) / l2_product(B0, B0, Q))**0.5

# %% [markdown]
# ## Magnetic Field Visualization
# Visualize the initial magnetic field configuration
# %%
F_B = Pullback(B0, F, 2)  # Pullback of initial magnetic field
F_B_h = Pullback(B_h, F, 2)  # Pullback of discrete magnetic field
_z1 = jax.vmap(F_B_h)(_x).reshape(nx, nx, 3)
_z1_norm = jnp.linalg.norm(_z1, axis=2)
_z2 = jax.vmap(F_B)(_x).reshape(nx, nx, 3)
_z2_norm = jnp.linalg.norm(_z2, axis=2)
plt.figure(figsize=(10, 8))
plt.contourf(_y1, _y2, _z1_norm.reshape(nx, nx))
plt.colorbar()
plt.contour(_y1, _y2, _z2_norm.reshape(nx, nx), colors='k')
__z1 = jax.vmap(F_B_h)(__x).reshape(_nx, _nx, 3)
plt.quiver(
    __y1,
    __y2,
    __z1[:, :, 0],
    __z1[:, :, 1],
    color='w')
plt.title('Initial Magnetic Field Configuration')
plt.savefig(output_dir / 'initial_magnetic_field.png')


# %% [markdown]
# ## Helicity and Energy Analysis
# Compute initial helicity and energy
# %%
print("Helicity before perturbation: ", A_hat @ M12 @ B0_hat)
print("Energy before perturbation: ", B0_hat @ M2 @ B0_hat / 2)

# %% [markdown]
# ## SVD Analysis
# Perform singular value decomposition for field reconstruction
# %%
U, S, Vh = jnp.linalg.svd(C)  # SVD of curl-curl matrix
S_inv = jnp.where(S > 1e-6 * S[0] * S.shape[0], 1/S, 0)  # Regularized inverse
A_hat_recon = U @ jnp.diag(S_inv) @ Vh @ D1.T @ B0_hat  # Reconstruct vector potential

# Compute reconstruction error
A_err = ((A_hat - A_hat_recon) @ M1 @ (A_hat - A_hat_recon) / (A_hat @ M1 @ A_hat))**0.5
print("error in A:", A_err)

# Verify helicity preservation
A_hat_recon @ M12 @ B0_hat
# %%
A_h = DiscreteFunction(A_hat, Λ1)
F_A_h = Pullback(A_h, F, 1)
_z1 = jax.vmap(F_A_h)(_x).reshape(nx, nx, 3)
_z1_norm = jnp.linalg.norm(_z1, axis=2)
_z2 = jax.vmap(F_A)(_x).reshape(nx, nx, 3)
_z2_norm = jnp.linalg.norm(_z2, axis=2)
plt.figure(figsize=(10, 8))
plt.contourf(_y1, _y2, _z1_norm.reshape(nx, nx))
plt.colorbar()
plt.contour(_y1, _y2, _z2_norm.reshape(nx, nx), colors='k')
__z1 = jax.vmap(F_A_h)(__x).reshape(_nx, _nx, 3)
plt.quiver(
    __y1,
    __y2,
    __z1[:, :, 0],
    __z1[:, :, 1],
    color='w')
# %%
def compute_curl_error(x): return curl(A)(x) - curl(A_h)(x)

# Compute relative L2 error in curl
(l2_product(compute_curl_error, compute_curl_error, Q) / l2_product(curl(A), curl(A), Q))**0.5
# %%
A_h = DiscreteFunction(A_hat, Λ1)
F_A_h = Pullback(curl(A_h), F, 2)
_z1 = jax.vmap(F_A_h)(_x).reshape(nx, nx, 3)
_z1_norm = jnp.linalg.norm(_z1, axis=2)
_z2 = jax.vmap(F_B)(_x).reshape(nx, nx, 3)
_z2_norm = jnp.linalg.norm(_z2, axis=2)
plt.figure(figsize=(10, 8))
plt.contourf(_y1, _y2, _z1_norm.reshape(nx, nx))
plt.colorbar()
plt.contour(_y1, _y2, _z2_norm.reshape(nx, nx), colors='k')
__z1 = jax.vmap(F_A_h)(__x).reshape(_nx, _nx, 3)
plt.quiver(
    __y1,
    __y2,
    __z1[:, :, 0],
    __z1[:, :, 1],
    color='w')
# %%

# %% [markdown]
# ## Perturbation and Evolution
# Define the perturbation field and evolution functions
# %%
print(l2_product(A, curl(A), Q))
# %%
print("Helicity before perturbation: ", A_hat @ M12 @ B0_hat)
print("Energy before perturbation: ", B0_hat @ M2 @ B0_hat / 2)

def u(x):
    """Define the perturbation field.
    
    Args:
        x: Spatial coordinates
        
    Returns:
        jnp.array: Perturbation vector field
    """
    r, χ, z = x
    a1 = jnp.sin(2 * jnp.pi * r) * jnp.cos(2 * jnp.pi * χ)
    a2 = jnp.cos(2 * jnp.pi * r) * jnp.sin(2 * jnp.pi * χ)
    a3 = jnp.sin(2 * jnp.pi * r) * jnp.cos(2 * jnp.pi * χ)
    return jnp.array([a1, a2, a3])

u_hat = jnp.linalg.solve(M2, P2(u))
u_h = DiscreteFunction(u_hat, Λ2)

B_hat = B0_hat
dt = 0.001
max_iterations = int(0.05/dt)

# %% [markdown]
# ## Evolution Loop
# Define the function to compute magnetic field changes
# %%
@jax.jit
def perturb_B_hat(B_hat, B_hat_0, dt):
    """Compute the change in magnetic field.
    
    Args:
        B_hat: Current magnetic field
        B_hat_0: Previous magnetic field
        dt: Time step
        
    Returns:
        tuple: (error, new B_hat, velocity field)
    """
    H_hat_1 = jnp.linalg.solve(M1, M12 @ B_hat)         # H = Proj(B)
    H_hat_0 = jnp.linalg.solve(M1, M12 @ B_hat_0)
    H_h = DiscreteFunction((H_hat_0 + H_hat_1)/2, Λ1)
    u_h = DiscreteFunction(u_hat, Λ2)
    E_hat = jnp.linalg.solve(M1, Pc(H_h, u_h))          # E = u x H
    ẟB_hat = jnp.linalg.solve(M2, D1 @ E_hat)           # ẟB = curl E
    B_hat_1 = B_hat_0 + dt * ẟB_hat
    error = (B_hat_1 - B_hat) @ M2 @ (B_hat_1 - B_hat)
    return error, B_hat_1, u_hat

# %% [markdown]
# ## Main Evolution Loop
# Perform the magnetic field evolution
# %%
for i in range(max_iterations):
    error_val = 1.0
    B_hat_1 = B_hat
    iteration_count = 0
    while error_val > 1e-11:
        error_val, B_hat_1, _u_hat = perturb_B_hat(B_hat_1, B_hat, dt)
        iteration_count += 1
    B_hat = B_hat_1
    print("Iteration: ", i+1)
    print("Magnetic Energy: ", (B_hat @ M2 @ B_hat) / 2)
    print("Force: ", (_u_hat @ M2 @ _u_hat))
    A_hat = U @ jnp.diag(S_inv) @ Vh @ D1.T @ B_hat
    print("Helicity: ", A_hat @ M12 @ B_hat)
    print("Div B: ", (D2 @ B_hat) @ jnp.linalg.solve(M3, D2 @ B_hat))
    print("Picard iterations: ", iteration_count)
    print("dt: ", dt)

B_h = DiscreteFunction(B_hat, Λ2)

# %% [markdown]
# ## Final State Analysis
# Analyze the final state after evolution
# %%
F_B = Pullback(B0, F, 2)
F_B_h = Pullback(B_h, F, 2)
_z1 = jax.vmap(F_B_h)(_x).reshape(nx, nx, 3)
_z1_norm = jnp.linalg.norm(_z1, axis=2)
_z2 = jax.vmap(F_B)(_x).reshape(nx, nx, 3)
_z2_norm = jnp.linalg.norm(_z2, axis=2)
plt.figure(figsize=(10, 8))
plt.contourf(_y1, _y2, _z1_norm.reshape(nx, nx))
plt.colorbar()
plt.contour(_y1, _y2, _z2_norm.reshape(nx, nx), colors='k')
__z1 = jax.vmap(F_B_h)(__x).reshape(_nx, _nx, 3)
plt.quiver(
    __y1,
    __y2,
    __z1[:, :, 0],
    __z1[:, :, 1],
    color='w')
plt.title('Final Magnetic Field Configuration')
plt.savefig(output_dir / 'final_field.png')


# %% [markdown]
# ## Final State Analysis
# Compute final helicity and energy
# %%
A_hat = U @ jnp.diag(S_inv) @ Vh @ D1.T @ B_hat
print("Helicity after perturbation: ", A_hat @ M12 @ B_hat)
print("Energy after perturbation: ", B_hat @ M2 @ B_hat / 2)

BN_hat = B_hat

# %% [markdown]
# ## Force Calculation
# Calculate the force for the final state
# F = J x B = (curl B) x B
# J = curl H
# H = Proj(B)
# %%
helicities = []
energies = []
forces = []
critical_as = []
divBs = []
dts = []

# %% [markdown]
# ## Evolution with Force
# Perform evolution with force calculation
# %%
@jax.jit
def ẟB_hat(B_hat, B_hat_0, dt):
    """Compute the change in magnetic field with force calculation.
    
    Args:
        B_hat: Current magnetic field
        B_hat_0: Previous magnetic field
        dt: Time step
        
    Returns:
        tuple: (error, new B_hat, velocity field)
    """
    H_hat_1 = jnp.linalg.solve(M1, M12 @ B_hat)         # H = Proj(B)
    H_hat_0 = jnp.linalg.solve(M1, M12 @ B_hat_0)
    J_hat_1 = jnp.linalg.solve(M1, D1.T @ B_hat)
    J_hat_0 = jnp.linalg.solve(M1, D1.T @ B_hat_0)      # J = curl H
    H_h = DiscreteFunction((H_hat_0 + H_hat_1)/2, Λ1)
    J_h = DiscreteFunction((J_hat_0 + J_hat_1)/2, Λ1)

    def JcrossH(x):
        return jnp.cross(J_h(x), H_h(x))
    u_hat = jnp.linalg.solve(M2, P2(JcrossH))           # u = J x H
    u_h = DiscreteFunction(u_hat, Λ2)
    E_hat = jnp.linalg.solve(M1, Pc(H_h, u_h))          # E = u x H
    ẟB_hat = jnp.linalg.solve(M2, D1 @ E_hat)           # ẟB = curl E
    B_hat_1 = B_hat_0 + dt * ẟB_hat
    err = (B_hat_1 - B_hat) @ M2 @ (B_hat_1 - B_hat)
    return err, B_hat_1, u_hat

# %% [markdown]
# ## Main Evolution Loop with Force
# Perform the main evolution with force calculation
# %%
dt0 = 0.00005
dt = dt0
B_hat = BN_hat

for i in range(100):
    err = 1
    B_hat_1 = B_hat
    it = 0
    while err > 1e-11:
        err, B_hat_1, _u_hat = ẟB_hat(B_hat_1, B_hat, dt)
        it += 1
    B_hat = B_hat_1

    print("Iteration: ", i+1)
    print("Magnetic Energy: ", (B_hat @ M2 @ B_hat) / 2)
    print("Force: ", (_u_hat @ M2 @ _u_hat))
    A_hat = U @ jnp.diag(S_inv) @ Vh @ D1.T @ B_hat
    print("Helicity: ", A_hat @ M12 @ B_hat)
    print("Div B: ", (D2 @ B_hat) @ jnp.linalg.solve(M3, D2 @ B_hat))
    print("Picard iterations: ", it)
    print("dt: ", dt)

    helicities.append(A_hat @ M12 @ B_hat)
    energies.append((B_hat @ M2 @ B_hat) / 2)
    forces.append((_u_hat @ M2 @ _u_hat))
    critical_as.append(it)
    divBs.append((D2 @ B_hat) @ jnp.linalg.solve(M3, D2 @ B_hat))
    dts.append(dt)

# %% [markdown]
# ## Plot Results
# Visualize the evolution results
# %%
plt.figure(figsize=(10, 6))
plt.plot(np.abs(np.array(energies) - B0_hat @ M2 @ B0_hat / 2), label='Energy - E(0)')
plt.xlabel('Iteration')
plt.ylabel('Energy Difference')
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.title('Energy Evolution')
plt.savefig(output_dir / 'energy_evolution.png')


plt.figure(figsize=(10, 6))
plt.plot(np.abs(np.array(helicities) - helicities[0]), label='|Helicity - H(0)|')
plt.xlabel('Iteration')
plt.ylabel('Helicity Difference')
plt.legend()
plt.yscale('log')
plt.title('Helicity Evolution')
plt.savefig(output_dir / 'helicity_evolution.png')


plt.figure(figsize=(10, 6))
plt.plot(np.array(forces)/np.array(energies), label='force/energy')
plt.xlabel('Iteration')
plt.ylabel('Force/Energy Ratio')
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.title('Force to Energy Ratio')
plt.savefig(output_dir / 'force_energy_ratio.png')


plt.figure(figsize=(10, 6))
plt.plot(np.abs(np.array(divBs) - divBs[0]), label='div B')
plt.xlabel('Iteration')
plt.ylabel('Divergence Error')
plt.legend()
plt.title('Divergence Error Evolution')
plt.savefig(output_dir / 'divergence_error.png')


plt.figure(figsize=(10, 6))
plt.plot(critical_as, label='Picard iterations')
plt.xlabel('Iteration')
plt.ylabel('Number of Picard Iterations')
plt.legend()
plt.title('Picard Iterations per Step')
plt.savefig(output_dir / 'picard_iterations.png')


plt.figure(figsize=(10, 6))
plt.plot(dts, label='adaptive time-step')
plt.xlabel('Iteration')
plt.ylabel('Time Step')
plt.yscale('log')
plt.legend()
plt.title('Adaptive Time Step Evolution')
plt.savefig(output_dir / 'time_step_evolution.png')


# %% [markdown]
# ## Final Field Visualization
# Visualize the final field configuration
# %%
F_B = Pullback(B0_h, F, 2)
_z2 = jax.vmap(F_B)(_x).reshape(nx, nx, 3)
_z2_norm = jnp.linalg.norm(_z2, axis=2)
B_h = DiscreteFunction(B_hat, Λ2)
F_A_h = Pullback(B_h, F, 2)
_z1 = jax.vmap(F_A_h)(_x).reshape(nx, nx, 3)
_z1_norm = jnp.linalg.norm(_z1, axis=2)
plt.figure(figsize=(10, 8))
plt.contourf(_y1, _y2, _z1_norm.reshape(nx, nx))
plt.colorbar()
plt.contour(_y1, _y2, _z2_norm.reshape(nx, nx), colors='k')
__z1 = jax.vmap(F_A_h)(__x).reshape(_nx, _nx, 3)
plt.quiver(
    __y1,
    __y2,
    __z1[:, :, 0],
    __z1[:, :, 1],
    color='w')
plt.title('Final Magnetic Field Configuration')
plt.savefig(output_dir / 'final_field.png')
plt.show()

# %%
print(l2_product(A, curl(A), Q))  # Compute final helicity
