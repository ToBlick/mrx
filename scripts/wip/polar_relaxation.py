# %%
from functools import partial
from pathlib import Path
from typing import List

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from mrx.DifferentialForms import (
    DifferentialForm,
    DiscreteFunction,
    Flat,
    Pullback,
    Pushforward,
    Sharp,
)
from mrx.IterativeSolvers import picard_solver
from mrx.LazyMatrices import (
    LazyDerivativeMatrix,
    LazyDoubleCurlMatrix,
    LazyMassMatrix,
    LazyProjectionMatrix,
)
from mrx.PolarMapping import LazyExtractionOperator, get_xi
from mrx.Projectors import CurlProjection, GradientProjection, Projector
from mrx.Quadrature import QuadratureRule
from mrx.Utils import curl, div, grad

# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)

# Create output directory for figures
output_dir = Path("script_outputs")
output_dir.mkdir(parents=True, exist_ok=True)

# Define mesh parameters
types = ('clamped', 'periodic', 'constant')  # Boundary condition types
ns = (10, 10, 1)  # Number of elements in each direction
ps = (3, 3, 0)  # Polynomial degrees
Λ0, Λ1, Λ2, Λ3 = [DifferentialForm(i, ns, ps, types)
                  for i in range(4)]  # H1, H(curl), H(div), L2
Q = QuadratureRule(Λ0, 6)  # Quadrature rule

# Define domain parameters
a = 1      # Radius of the torus
R0 = 3.0   # Major radius
Y0 = 0.0   # Vertical offset

# Define polar mapping functions


def θ(x):
    r, χ, z = x
    return 2 * jnp.atan(jnp.sqrt((1 + a*r/R0)/(1 - a*r/R0)) * jnp.tan(jnp.pi * χ))


def _R(r, χ):
    return jnp.ones(1) * (R0 + a * r * jnp.cos(2 * jnp.pi * χ))


def _Y(r, χ):
    return jnp.ones(1) * (Y0 + a * r * jnp.sin(2 * jnp.pi * χ))


@jax.jit
def F(x):
    r, χ, z = x
    return jnp.ravel(jnp.array([_R(r, χ) * jnp.cos(2 * jnp.pi * z),
                               _Y(r, χ),
                               _R(r, χ) * jnp.sin(2 * jnp.pi * z)]))


# %%
# Set up extraction operators and matrices
ξ, R_hat, Y_hat, Λ, τ = get_xi(_R, _Y, Λ0, Q)
E0, E1, E2, E3 = [LazyExtractionOperator(
    Λ, ξ, True).M for Λ in [Λ0, Λ1, Λ2, Λ3]]
M0, M1, M2, M3 = [LazyMassMatrix(Λ, Q, F, E).M for Λ, E in zip(
    [Λ0, Λ1, Λ2, Λ3], [E0, E1, E2, E3])]
P0, P1, P2, P3 = [Projector(Λ, Q, F, E)
                  for Λ, E in zip([Λ0, Λ1, Λ2, Λ3], [E0, E1, E2, E3])]
M12 = LazyProjectionMatrix(Λ1, Λ2, Q, F, E1, E2).M.T
C = LazyDoubleCurlMatrix(Λ1, Q, F, E1).M
D2 = LazyDerivativeMatrix(Λ2, Λ3, Q, F, E2, E3).M
D1 = LazyDerivativeMatrix(Λ1, Λ2, Q, F, E1, E2).M
D0 = LazyDerivativeMatrix(Λ0, Λ1, Q, F, E0, E1).M
Pc = CurlProjection(Λ1, Q, F, E1)  # Computes (B, A x Λ[i]) given A and B
Pg = GradientProjection(Λ0, Q, F, E0)


def l2_product(f, g, Q):
    """Compute the L2 inner product of two functions"""
    return jnp.einsum("ij,ij,i->", jax.vmap(f)(Q.x), jax.vmap(g)(Q.x), Q.w)


# %%
# Set up plotting grid
ɛ = 1e-5  # Small offset from boundaries
nx = 64   # High resolution for contour plots
_nx = 16  # Lower resolution for quiver plots

# Create evaluation grids
_x1 = jnp.linspace(ɛ, 1-ɛ, nx)
_x2 = jnp.linspace(ɛ, 1-ɛ, nx)
_x3 = jnp.zeros(1)
_x = jnp.array(jnp.meshgrid(_x1, _x2, _x3))
_x = _x.transpose(1, 2, 3, 0).reshape(nx*nx*1, 3)
_y = jax.vmap(F)(_x)
_y1 = _y[:, 0].reshape(nx, nx)
_y2 = _y[:, 1].reshape(nx, nx)

# Create quiver plot grid
__x1 = jnp.linspace(ɛ, 1-ɛ, _nx)
__x2 = jnp.linspace(ɛ, 1-ɛ, _nx)
__x3 = jnp.zeros(1)
__x = jnp.array(jnp.meshgrid(__x1, __x2, __x3))
__x = __x.transpose(1, 2, 3, 0).reshape(_nx*_nx*1, 3)
__y = jax.vmap(F)(__x)
__y1 = __y[:, 0].reshape(_nx, _nx)
__y2 = __y[:, 1].reshape(_nx, _nx)

# %%


def plot_field_logical(A, vector=True, levels=50):
    """Plot the vector field A."""
    plt.figure(figsize=(8, 6))
    if vector:
        _z1 = jax.vmap(A)(_x).reshape(nx, nx, 3)
        _z1_norm = jnp.linalg.norm(_z1, axis=2)
    else:
        _z1_norm = jax.vmap(A)(_x).reshape(nx, nx)
    plt.contourf(_x1, _x2, _z1_norm.reshape(nx, nx), levels=levels)
    plt.colorbar()
    if vector:
        __z1 = jax.vmap(A)(__x).reshape(_nx, _nx, 3)
        plt.quiver(__x1, __x2, __z1[:, :, 0], __z1[:, :, 1], color='w')
    else:
        plt.contour(_x1, _x2, _z1_norm.reshape(nx, nx), levels=levels //
                    2, colors='w', linestyles=":", linewidths=1.5)
    plt.xlabel('r')
    plt.ylabel('χ')


def plot_field_physical(A, vector=True, levels=50):
    """Plot the vector field A."""
    plt.figure(figsize=(8, 6))
    if vector:
        _z1 = jax.vmap(A)(_x).reshape(nx, nx, 3)
        _z1_norm = jnp.linalg.norm(_z1, axis=2)
    else:
        _z1_norm = jax.vmap(A)(_x).reshape(nx, nx)
    plt.contourf(_y1, _y2, _z1_norm.reshape(nx, nx), levels=levels)
    plt.colorbar()
    if vector:
        __z1 = jax.vmap(A)(__x).reshape(_nx, _nx, 3)
        plt.quiver(__y1, __y2, __z1[:, :, 0], __z1[:, :, 1], color='w')
    else:
        plt.contour(_y1, _y2, _z1_norm.reshape(nx, nx), levels=levels //
                    2, colors='w', linestyles=":", linewidths=1.5)
    plt.xlabel('x')
    plt.ylabel('y')


# # %%
# plot_field_physical(Pushforward(B_h, F, 2))
# plt.title('B-field')

# plot_field_physical(Pushforward(p_h, F, 0), vector=False)
# plt.title('p-field')

# plot_field_physical(Pushforward(u_h, F, 2))
# plt.title('force')
# Define initial vector potential and magnetic field
# %%

# def A(x):
#     """Initial vector potential"""
#     r, χ, z = x
#     a1 = 0
#     a2 = r**2 * 10
#     a3 = 1
#     return jnp.array([a1, a2, a3]) * jnp.exp(-r**2/0.25) / 20

def B(x):
    """Initial magnetic field"""
    r, χ, z = x
    return jnp.array([0, 1, r*(1-r)**2])


def p(x):
    """Initial pressure"""
    r, χ, z = x
    return jnp.atleast_1d(jnp.cos(jnp.pi * r/2)**2)


# A_hat = jnp.linalg.solve(M1, P1(Flat(A, F)))
# A_h = DiscreteFunction(A_hat, Λ1, E1)
# %%
# Plot initial field configuration
# plot_field_logical(A_h)
# plt.title('Initial vector potential')
# # %%
# plot_field_physical(Pushforward(A_h, F, 1))
# plt.title('Initial vector potential')
# %%
H_hat = jnp.linalg.solve(M1, P1(Flat(B, F)))
B_hat = jnp.linalg.solve(M2, M12.T @ H_hat)  # Initial magnetic field
B_h = DiscreteFunction(B_hat, Λ2, E2)
plot_field_logical(B_h)
plt.title('Initial B-field')
# %%
plot_field_physical(Pushforward(B_h, F, 2))
plt.title('Initial B-field')
# %%
p_hat = jnp.linalg.solve(M0, P0(p))
p_h = DiscreteFunction(p_hat, Λ0, E0)
plot_field_logical(p_h, vector=False)
plt.title('Initial p-field')

# %%
plot_field_physical(Pushforward(p_h, F, 0), vector=False)
plt.title('Initial p-field')

# %%
# Calculate SVD for reconstruction
U, S, Vh = jnp.linalg.svd(C)
S_inv = jnp.where(S/S[0] > 1e-11, 1/S, 0)
C_inv = U @ jnp.diag(S_inv) @ Vh

# Compute initial energy
Energy0 = B_hat @ M2 @ B_hat / 2
print(f"Initial energy: {Energy0:.2e}")

# Compute initial helicity
A_hat_recon = C_inv @ D1.T @ B_hat
helicity = A_hat_recon @ M12 @ B_hat
print(f"Initial helicity: {helicity:.2e}")

# print(
#     f"A reconstruction error: {((A_hat_recon - A_hat) @ M1 @ (A_hat_recon - A_hat) /
#                                 ((A_hat) @ M1 @ (A_hat)))**0.5:.2e}")

# %%
# "Divergence-clean" the B field
# K is the 3-form Laplacian
K = D2 @ jnp.linalg.solve(M2, D2.T)
# Compute the Leray projector
𝚷_Leray = jnp.eye(*M2.shape) - jnp.linalg.solve(M2,
                                                D2.T @ jnp.linalg.solve(K, D2))

# %%


@jax.jit
def force(B_hat, p_hat):
    H_hat = jnp.linalg.solve(M1, M12 @ B_hat)
    J_hat = jnp.linalg.solve(M1, D1.T @ B_hat)
    H_h = DiscreteFunction(H_hat, Λ1, E1)
    J_h = DiscreteFunction(J_hat, Λ1, E1)

    grad_p_hat = jnp.linalg.solve(M1, D0 @ p_hat)

    def JcrossH(x):
        return jnp.cross(J_h(x), H_h(x))
    u_hat = jnp.linalg.solve(M2, P2(JcrossH)) - \
        jnp.linalg.solve(M2, M12.T @ grad_p_hat)
    return u_hat


def force_residual(B_hat, p_hat):
    u_hat = force(B_hat, p_hat)
    return (u_hat @ M2 @ u_hat)**0.5


def magnetic_energy(B_hat,):
    return (B_hat @ M2 @ B_hat)/2


def inner_energy(p_hat):
    p_h = DiscreteFunction(p_hat, Λ0, E0)
    Ɣ = 5/3

    def e(x):
        return 1/(Ɣ - 1) * p_h(x)
    return Q.w @ jax.vmap(e)(Q.x)


def rk4(x0, f, dt):
    k1 = f(x0)
    k2 = f(x0 + dt/2 * k1)
    k3 = f(x0 + dt/2 * k2)
    k4 = f(x0 + dt * k3)
    return x0 + dt/6 * (k1 + 2*k2 + 2*k3 + k4)


@partial(jax.jit, static_argnames=['B_h', 'n_steps'])
def fieldline(x0, B_h, dt, n_steps):
    def step(current_x, _):
        next_x = rk4(current_x, B_h, dt)
        next_x = next_x.at[0].set(jnp.clip(next_x[0], 0, 1))
        next_x = next_x.at[1].set(jnp.mod(next_x[1], 1))
        next_x = next_x.at[2].set(jnp.mod(next_x[2], 1))
        return next_x, next_x
    final_x, xs = jax.lax.scan(step, x0, None, length=n_steps)
    return xs


# %%
u_hat = force(B_hat, p_hat)

# %%
# Plot force
u_h = DiscreteFunction(u_hat, Λ2, E2)
plot_field_logical(u_h)
plt.title('Force')
# %%
plot_field_physical(Pushforward(u_h, F, 2))
plt.title('Force')

# %%


@jax.jit
def update(B_hat, p_hat, B_hat_0, dt):
    H_hat = jnp.linalg.solve(M1, M12 @ (B_hat + B_hat_0))/2     # H = Proj(B)
    J_hat = jnp.linalg.solve(M1, D1.T @ (B_hat + B_hat_0))/2    # J = curl H
    H_h = DiscreteFunction(H_hat, Λ1, E1)
    J_h = DiscreteFunction(J_hat, Λ1, E1)
    grad_p_hat = jnp.linalg.solve(M1, D0 @ p_hat)

    def JcrossH(x):
        return jnp.cross(J_h(x), H_h(x))
    u_hat = jnp.linalg.solve(M2, P2(JcrossH)) \
        - jnp.linalg.solve(M2, M12.T @ grad_p_hat)  # u = J x H - grad p
    u_h = DiscreteFunction(u_hat, Λ2, E2)
    E_hat = jnp.linalg.solve(M1, Pc(H_h, u_h))          # E = u x H
    ẟB_hat = jnp.linalg.solve(M2, D1 @ E_hat)           # ẟB = curl E
    B_hat_1 = B_hat_0 + dt * ẟB_hat
    p_h = DiscreteFunction(p_hat, Λ0, E0)
    ẟp_hat = - Pg(p_h, u_h)
    p_hat_1 = p_hat + dt * ẟp_hat
    return B_hat_1, p_hat_1, u_hat


def twoformnorm(x):
    return (x @ M2 @ x)**0.5


def relax(B_hat_0, p_hat_0, dt):
    def step(B_hat_guess, dt):
        B_hat_1, p_hat_1, u_hat = update(B_hat_guess, p_hat_0, B_hat_0, dt)
        return B_hat_1, p_hat_1, u_hat

    B_hat_old = B_hat_0

    # Yolo explicit Euler
    # B_hat, p_hat, u_hat = step(B_hat_old, dt)

    # # Picard by hand
    i = 0
    err = 1
    while err > 1e-6:
        B_hat, p_hat, u_hat = step(B_hat_old, dt)
        err = twoformnorm(B_hat - B_hat_old)
        i += 1
        B_hat_old = B_hat
        if i > 20:
            print("Picard iteration limit reached")
            dt /= 2
            B_hat = B_hat_0
            i = 0
            continue

    if i > 3:
        dt *= 0.8
    if i < 3:
        dt *= 1.2
    return B_hat, p_hat, err, i, u_hat, dt
    # return picard_solver(B_step, B_hat_0, 1e-6, 20, twoformnorm)


# %%
H_hat = jnp.linalg.solve(M1, P1(Flat(B, F)))
B_hat_0 = jnp.linalg.solve(M2, M12.T @ H_hat)  # Initial magnetic field
p_hat_0 = jnp.linalg.solve(M0, P0(p))       # Initial pressure
B_hat = B_hat_0
p_hat = p_hat_0
u_hat = jnp.zeros_like(B_hat)

# %%
# Initialize lists for tracking evolution
i = 0
inner_energy_trace = []
magnetic_energy_trace = []
energy_trace = []
force_trace = []
div_trace = []
picard_iter_trace = []
helicity_trace = []
B_hat_trace = []
p_hat_trace = []
angle_trace = []
u_hat_trace = []
dt_trace = []
dt = 1e-4
# %%

for _ in range(100):
    i += 1
    u_hat_old = u_hat
    B_hat, p_hat, err, iter, u_hat, dt = relax(B_hat, p_hat, dt)

    dt_trace.append(dt)

    picard_iter_trace.append(iter)

    force_trace.append(force_residual(B_hat, p_hat))
    div_trace.append((jnp.linalg.solve(
        M3, D2 @ B_hat) @ D2 @ B_hat)**0.5)

    magnetic_energy_trace.append(magnetic_energy(B_hat))
    inner_energy_trace.append(inner_energy(p_hat))
    energy_trace.append(magnetic_energy_trace[-1] +
                        inner_energy_trace[-1])

    helicity_trace.append(C_inv @ D1.T @ B_hat @ M12 @ B_hat)

    u_angle = u_hat_old @ M2 @ u_hat / \
        (u_hat @ M2 @ u_hat)**0.5 / (u_hat_old @ M2 @ u_hat_old)**0.5
    angle_trace.append(u_angle)
    print("Iteration: ", i)
    print("dt: ", dt)
    print("Picard iterations: ", iter)
    print("L2 norm of div B: ", div_trace[-1])
    print("Force residual: ", force_trace[-1])
    print("Energy: ", energy_trace[-1][0])
    print("Helicity: ", helicity_trace[-1])
    print("Angle: ", angle_trace[-1])
    if i % 1 == 0:
        # print("Saving B_hat...")
        # B_h = DiscreteFunction(B_hat, Λ2, E2)
        # plot_field_physical(Pushforward(B_h, F, 2))
        # plt.title(f'B-field at iteration {i}')
        # plt.savefig(output_dir / f'b_field_{i}.png', dpi=300)
        # plt.close()
        B_hat_trace.append(B_hat)
        p_hat_trace.append(p_hat)
        u_hat_trace.append(u_hat)

    print("")

# %%

# Plot evolution of quantities
plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1)
plt.plot(magnetic_energy_trace, label='Magnetic Energy')
plt.plot(inner_energy_trace, label='Inner Energy')
plt.plot(energy_trace, label='Total Energy')
plt.title('Energy Evolution')
plt.xlabel('Iteration')
plt.ylabel('Energy')
plt.yscale('log')
plt.legend()

plt.subplot(2, 3, 2)
plt.plot(jnp.array(helicity_trace) - helicity_trace[0])
plt.title('Helicity Evolution')
plt.xlabel('Iteration')
plt.ylabel('Helicity')

plt.subplot(2, 3, 3)
plt.plot(force_trace)
plt.title('Force Evolution')
plt.xlabel('Iteration')
plt.ylabel('Force Norm')
plt.yscale('log')

plt.subplot(2, 3, 4)
plt.plot(div_trace)
plt.title('Divergence Evolution')
plt.xlabel('Iteration')
plt.ylabel('Div B')

plt.subplot(2, 3, 5)
plt.plot(picard_iter_trace)
plt.title('Picard Iterations')
plt.xlabel('Iteration')
plt.ylabel('Picard Iterations')

plt.subplot(2, 3, 6)
plt.plot(jnp.arccos(jnp.array(angle_trace)))
plt.title('Change in velocity')
plt.xlabel('Iteration')
plt.ylabel('acos( u · u- / |u| |u-| )')
plt.yscale('log')

plt.tight_layout()

# %%
idx = len(B_hat_trace) - 1
B_h = DiscreteFunction(B_hat_trace[idx], Λ2, E2)
u_h = DiscreteFunction(u_hat_trace[idx], Λ2, E2)
p_h = DiscreteFunction(p_hat_trace[idx], Λ0, E0)

plot_field_physical(Pushforward(B_h, F, 2))
plt.title('B-field')

plot_field_physical(Pushforward(p_h, F, 0), vector=False)
plt.title('p-field')

plot_field_physical(Pushforward(u_h, F, 2))
plt.title('force')

# %%
helicity_trace[0] / magnetic_energy_trace[0]

# %%
H_hat = jnp.linalg.solve(M1, M12 @ B_hat_trace[-1])
(C @ H_hat) @ jnp.linalg.solve(M1, C @ H_hat) / (H_hat @ M1 @ H_hat)

LapH = jnp.linalg.solve(M1, C @ H_hat)
LapH_h = DiscreteFunction(LapH, Λ1, E1)

plot_field_logical(LapH_h)
plt.title('Laplacian of H')

plot_field_physical(Pushforward(LapH_h, F, 1))
plt.title('Laplacian of H')

# %%

key = jax.random.PRNGKey(0)
key_r, key_phi = jax.random.split(key)

# %%
__nx = 16
_ɛ = 1e-1
# ___x1 = jnp.linspace(ɛ, 1-ɛ, __nx)
# ___x2 = jnp.array([0, 0.33, 0.66])
# ___x3 = jnp.zeros(1)
# ___x = jnp.array(jnp.meshgrid(___x1, ___x2, ___x3))
# ___x = ___x.transpose(1, 2, 3, 0).reshape(__nx*3*1, 3)
__z = jnp.zeros(__nx)
__r = jnp.linspace(_ɛ, 1-_ɛ, __nx)
__phi = jax.random.uniform(key_r, shape=(__nx,))
___x = jnp.stack([__r, __phi, __z], axis=-1)
___y = jax.vmap(F)(___x)

H_hat = jnp.linalg.solve(M1, M12 @ B_hat)
H_h = DiscreteFunction(H_hat, Λ1, E1)
H_vec = Sharp(H_h, F)
# FB_h = Pushforward(B_h, F, 2)

# %%
trajectories = jax.vmap(lambda x: fieldline(x, H_vec, 0.05, 20_000))(___x)
physical_trajectories = [jax.vmap(F)(t) for t in trajectories]
# %%
intersections = [t[jnp.abs(t[:, 2] < 1e-2)] for t in trajectories]
physical_intersections = [jax.vmap(F)(intersect)
                          for intersect in intersections]
# %%
# plt.figure(figsize=(8, 6))
# for i in intersections:
#     plt.scatter(i[:, 0], i[:, 1], s=1)
# plt.title('Field line intersections')
# plt.xlabel('r')
# plt.ylabel('phi')

# %%
# plt.figure(figsize=(8, 6))
# for t in physical_trajectories:
#     plt.scatter(t[:, 0], t[:, 1], s=1, alpha = (jnp.abs(t[:, 2])/6)**4)
# plt.title('Field line intersections')
# plt.xlabel('x')
# plt.ylabel('y')
# %%
plt.figure(figsize=(8, 6))
for intersect in physical_intersections:
    plt.scatter(intersect[:, 0], intersect[:, 1], s=1)
plt.title('Field line intersections')
plt.xlabel('x')
plt.ylabel('y')

# %%


def q(x, H_hat):
    r, χ, z = x
    R = R0 + a * r * jnp.cos(2 * jnp.pi * χ)
    H_h = DiscreteFunction(H_hat, Λ1, E1)
    H_vec = Sharp(H_h, F)
    B_val = H_vec(x)
    return r / R * jnp.abs(B_val[2] / B_val[1])


q_vals = jax.vmap(q, (0, None))(_x, H_hat)

plt.figure(figsize=(8, 6))
plt.contour(_y1, _y2, q_vals.reshape(nx, nx), levels=jnp.linspace(0, 10, 100))
plt.colorbar(label='q')
plt.title('q value')
plt.xlabel('x')
plt.ylabel('y')


# %%
# evd = jnp.linalg.eigh(get_hessian(B_hat_trace[-1]))
# print("Number of negative/zero eigenvalues: ", jnp.sum(evd[0] < 1e-9))
# evd[0][:sum(evd[0] < 1e-9)]
# # %%
# ker = evd[1][:, 0]
# ker_h = DiscreteFunction(ker, Λ2, E2)
# plot_field_physical(Pushforward(ker_h, F, 2))
# plt.title('Unstable eigenmode')

# # %%
# ker_h = DiscreteFunction(jnp.zeros_like(B_hat).at[42].set(1), Λ2, E2)
# plot_field_physical(Pushforward(ker_h, F, 2))
# plt.title('Unstable eigenmode')


# %%

# %%
# # %%
# # Main relaxation loop
# n_steps = 1000
# dt = 0.01
# B_hat = B0_hat
# traces = []

# print("\nStarting relaxation...")
# for i in range(n_steps):
#     B_diff, error, B_hat, u_hat = ẟB_hat(B_hat, B_hat, dt)

#     # Track diagnostics
#     A_hat = U @ jnp.diag(S_inv) @ Vh @ D1.T @ B_hat
#     helicity = A_hat @ M12 @ B_hat
#     energy = B_hat @ M2 @ B_hat / 2
#     force = u_hat @ M2 @ u_hat
#     divB = (D2 @ B_hat) @ jnp.linalg.solve(M3, D2 @ B_hat)

#     helicities.append(helicity)
#     energies.append(energy)
#     forces.append(force)
#     divBs.append(divB)

#     if i % 100 == 0:
#         print(f"\nIteration {i}:")
#         print(f"Helicity: {helicity:.2e}")
#         print(f"Energy: {energy:.2e}")
#         print(f"Force: {force:.2e}")
#         print(f"Div B: {divB:.2e}")
#         print(f"Error: {error:.2e}")

# # Plot evolution of quantities
# plt.figure(figsize=(12, 8))
# plt.subplot(2, 2, 1)
# plt.plot(energies)
# plt.title('Energy Evolution')
# plt.xlabel('Iteration')
# plt.ylabel('Energy')
# plt.grid(True)

# plt.subplot(2, 2, 2)
# plt.plot(helicities)
# plt.title('Helicity Evolution')
# plt.xlabel('Iteration')
# plt.ylabel('Helicity')
# plt.grid(True)

# plt.subplot(2, 2, 3)
# plt.plot(forces)
# plt.title('Force Evolution')
# plt.xlabel('Iteration')
# plt.ylabel('Force Norm')
# plt.yscale('log')
# plt.grid(True)

# plt.subplot(2, 2, 4)
# plt.plot(divBs)
# plt.title('Divergence Evolution')
# plt.xlabel('Iteration')
# plt.ylabel('Div B')
# plt.yscale('log')
# plt.grid(True)

# plt.tight_layout()
# plt.savefig(output_dir / 'evolution.png', dpi=300, bbox_inches='tight')

# # Plot final field configuration
# plt.figure(figsize=(10, 8))
# B_h = DiscreteFunction(B_hat, Λ2, E2)
# F_B = Pullback(B0, F, 2)
# F_B_h = Pullback(B_h, F, 2)
# _z1 = jax.vmap(F_B_h)(_x).reshape(nx, nx, 3)
# _z1_norm = jnp.linalg.norm(_z1, axis=2)
# _z2 = jax.vmap(F_B)(_x).reshape(nx, nx, 3)
# _z2_norm = jnp.linalg.norm(_z2, axis=2)
# plt.contourf(_y1, _y2, _z1_norm.reshape(nx, nx))
# plt.colorbar(label='Field Magnitude')
# plt.contour(_y1, _y2, _z2_norm.reshape(nx, nx), colors='k')
# __z1 = jax.vmap(F_B_h)(__x).reshape(_nx, _nx, 3)
# plt.quiver(__y1, __y2, __z1[:, :, 0], __z1[:, :, 1], color='w')
# plt.title('Final Magnetic Field Configuration')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.grid(True)
# plt.savefig(output_dir / 'final_field.png', dpi=300, bbox_inches='tight')

# # Print final diagnostics
# print("\nFinal diagnostics:")
# print(f"Final helicity: {helicities[-1]:.2e}")
# print(f"Final energy: {energies[-1]:.2e}")
# print(f"Final force: {forces[-1]:.2e}")
# print(f"Final div B: {divBs[-1]:.2e}")
# print(
#     f"Helicity conservation: {abs(helicities[-1] - helicities[0])/helicities[0]:.2e}")
# print(f"Energy change: {(energies[-1] - energies[0])/energies[0]:.2e}")

# # Show all plots
# plt.show()

# %%


@jax.jit
def get_hessian(B_hat):
    B_h = DiscreteFunction(B_hat, Λ2, E2)

    def integrand(i, j, x):
        B = B_h(x)
        DF = jax.jacfwd(F)(x)
        J = jnp.linalg.det(DF)
        Du_i = jax.jacfwd(Λ2[i])(x)
        Du_j = jax.jacfwd(Λ2[j])(x)
        term1 = (Du_i @ B) @ (Du_j @ B)
        term2 = - B @ (Du_i @ B) * jnp.trace(Du_j) - \
            B @ (Du_j @ B) * jnp.trace(Du_i)
        term3 = 0.5 * B @ DF.T @ DF @ B * \
            jnp.trace(Du_i) * jnp.trace(Du_j) / J**4
        term4 = 0.5 * B @ DF.T @ DF @ B * jnp.trace(Du_j @ Du_i) / J**2
        return (term1 + term2 + term3 + term4) * J

    def lazy_stab_mat(i, j):
        return jax.vmap(integrand, (None, None, 0))(i, j, Q.x).T @ Q.w

    _ns = jnp.arange(M2.shape[0])
    stab_mat = jax.vmap(jax.vmap(lazy_stab_mat, (0, None)),
                        (None, 0))(_ns, _ns)
    return stab_mat


@jax.jit
def get_approx_hessian(B_hat):
    B_h = DiscreteFunction(B_hat, Λ2, E2)

    def integrand(i, j, x):
        B = B_h(x)
        DF = jax.jacfwd(F)(x)
        J = jnp.linalg.det(DF)
        Du_i = jax.jacfwd(Λ2[i])(x)
        Du_j = jax.jacfwd(Λ2[j])(x)
        term1 = (Du_i @ B) @ (Du_j @ B)
        term2 = 0
        term3 = 0.5 * B @ DF.T @ DF @ B * \
            jnp.trace(Du_i) * jnp.trace(Du_j) / J**2
        term4 = 0
        return (term1 + term2 + term3 + term4) * J

    def lazy_stab_mat(i, j):
        return jax.vmap(integrand, (None, None, 0))(i, j, Q.x).T @ Q.w

    _ns = jnp.arange(M2.shape[0])
    stab_mat = jax.vmap(jax.vmap(lazy_stab_mat, (0, None)),
                        (None, 0))(_ns, _ns)
    return stab_mat
