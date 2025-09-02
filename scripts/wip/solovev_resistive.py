# %%
import time
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optimistix
import scipy as sp
from matplotlib import gridspec

from mrx.BoundaryFitting import solovev_lcfs_fit
from mrx.DifferentialForms import DifferentialForm, DiscreteFunction, Pushforward
from mrx.LazyMatrices import (
    LazyDerivativeMatrix,
    LazyDoubleCurlMatrix,
    LazyDoubleDivergenceMatrix,
    LazyMassMatrix,
    LazyProjectionMatrix,
    LazyStiffnessMatrix,
)
from mrx.Nonlinearities import CrossProductProjection
from mrx.Plotting import get_1d_grids, get_2d_grids
from mrx.PolarMapping import LazyExtractionOperator, get_xi
from mrx.Projectors import Projector
from mrx.Quadrature import QuadratureRule
from mrx.Utils import div, grad, inv33, jacobian_determinant, l2_product

jax.config.update("jax_enable_x64", True)

# %%
R0 = 3.0
μ0 = 1.0
π = jnp.pi
k0 = 1.1
q0 = 1.2
F0 = 3
aR = 0.66

###
# ψ(R, Z) =  (¼ k₀² (R² - R₀²)² + R²Z² ) / (2 R₀² k₀ q₀)
###

p_map = 3
n_map = 8
q_map = 2 * p_map

a_hat = solovev_lcfs_fit(n_map, p_map, q_map, R0, a=aR, k0=k0, q0=q0)

# a_hat = jnp.ones_like(a_hat) * 1.0

Λ_map = DifferentialForm(0, (n_map, 1, 1), (p_map, 0, 0),
                         ("periodic", "constant", "constant"))
a_h = DiscreteFunction(a_hat, Λ_map)


def a(χ):
    """Radius as a function of chi."""
    return a_h(jnp.array([χ, 0, 0]))[0]


_x = jnp.linspace(0, 1, 1024)
plt.plot(_x, jax.vmap(a)(_x))

# %%
γ = 5/3

p = 3
q = 3*p
ns = (8, 8, 4)
ps = (3, 3, 3)
types = ("clamped", "periodic", "periodic")


def _R(r, χ):
    return jnp.ones(1) * (R0 + a(χ) * r * jnp.cos(2 * π * χ))


def _Z(r, χ):
    return jnp.ones(1) * a(χ) * r * jnp.sin(2 * π * χ)


def F(x):
    """Polar coordinate mapping function."""
    r, χ, z = x
    return jnp.ravel(jnp.array(
        [_R(r, χ) * jnp.cos(2 * π * z),
         -_R(r, χ) * jnp.sin(2 * π * z),
         _Z(r, χ)]))

# %%


def psi(p):
    x, y, z = F(p)
    R = (x**2 + y**2)**0.5
    Z = z

    def _psi(R, Z):
        return (k0**2/4*(R**2 - R0**2)**2 + R**2*Z**2) / (2 * R0**2 * k0 * q0)
    return _psi(R, Z) - _psi(R0 + a(0), 0)


# %%
_x, _y, (_y1, _y2, _y3), (_x1, _x2, _x3) = get_2d_grids(F, zeta=0, nx=64)
_x_1d, _y_1d, (_y1_1d, _y2_1d, _y3_1d), (_x1_1d, _x2_1d,
                                         _x3_1d) = get_1d_grids(F, zeta=0, chi=0, nx=128)

# %%
plt.contourf(_y1, _y3, jax.vmap(psi)(_x).reshape(_y1.shape), levels=20)
vals = jax.vmap(psi)(_x_1d)
plt.plot(_y1_1d, vals - vals[0], 'k', label=r'$\psi(r, 0, 0)$')
plt.plot(_y1_1d, jnp.zeros_like(_y2_1d), ':k')
plt.axis('equal')
plt.legend()
plt.colorbar()
plt.xlabel("R")
plt.ylabel("Z")


# %%

Λ0, Λ1, Λ2, Λ3 = [
    DifferentialForm(i, ns, ps, types) for i in range(0, 4)
]  # H1, H(curl), H(div), L2
Q = QuadratureRule(Λ0, q)
ξ, R_hat, Y_hat, _, τ = get_xi(_R, _Z, Λ0, Q)
# %%
E0, E1, E2, E3 = [
    LazyExtractionOperator(Λ, ξ, True) for Λ in [Λ0, Λ1, Λ2, Λ3]
]
E1_f = LazyExtractionOperator(Λ1, ξ, False)
P0, P1, P2, P3 = [
    Projector(Λ, Q, F, E=E) for Λ, E in zip([Λ0, Λ1, Λ2, Λ3], [E0, E1, E2, E3])
]
M0, M1, M2, M3 = [
    LazyMassMatrix(Λ, Q, F, E).matrix() for Λ, E in zip([Λ0, Λ1, Λ2, Λ3], [E0, E1, E2, E3])
]
M1_f = LazyMassMatrix(Λ1, Q, F, E1_f).matrix()
D0, D1, D2 = [
    LazyDerivativeMatrix(Λ0, Λ1, Q, F, E0, E1).matrix() for Λ0, Λ1, E0, E1 in zip([Λ0, Λ1, Λ2], [Λ1, Λ2, Λ3], [E0, E1, E2], [E1, E2, E3])
]
M21 = LazyProjectionMatrix(Λ1, Λ2, Q, F, E1, E2).matrix()
M21_f = LazyProjectionMatrix(Λ1, Λ2, Q, F, E1_f, E2).matrix()
# %%
E0m, E1m, E2m, E3m = [E.matrix() for E in [E0, E1, E2, E3]]
# %%
# only one zero eigenvalue
L_pinv = jnp.linalg.pinv(
    D2 @ jnp.linalg.solve(M2, D2.T))  # one zero eigenvalue
DD = LazyDoubleDivergenceMatrix(Λ2, Q, F, E2).matrix()
K = LazyStiffnessMatrix(Λ0, Q, F, E0).matrix()
L_vec_pinv = jnp.linalg.inv(LazyDoubleCurlMatrix(
    Λ1, Q, F, E1).matrix() + D0 @ jnp.linalg.solve(M0, D0.T))


# %%
jnp.min(jax.vmap(jacobian_determinant(F))(Q.x)), jnp.max(
    jax.vmap(jacobian_determinant(F))(Q.x))
# %%
def p_phys(x):
    return - (k0**2 + 1)/(R0**2 * k0 * q0) * psi(x) * jnp.ones(1)


def B_xyz(p):
    x, y, z = F(p)
    R = (x**2 + y**2)**0.5
    phi = jnp.arctan2(y, x) / (2 * π)

    BR = - R * z / (R0**2 * k0 * q0)
    Bz = (k0**2 * (R**2 - R0**2) + 2*z**2) / (2 * R0**2 * k0 * q0)
    BPhi = 0.1 * F0 / R

    # BR = - z / R
    # Bz = 2 * (R - R0) / R
    # BPhi = F0 / R

    Bx = BR * jnp.cos(2 * π * phi) - BPhi * jnp.sin(2 * π * phi)
    By = BR * jnp.sin(2 * π * phi) + BPhi * jnp.cos(2 * π * phi)

    return jnp.array([Bx, By, Bz])


def twoformprojection(v):
    DF = jax.jacfwd(F)

    def _v(x):
        DFx = DF(x)
        return DFx.T @ v(x)

    def _Λ(x, i):
        return Λ2(x, i)
    Bjk = jax.vmap(_v)(Q.x)  # n_q x d
    Λijk = jax.vmap(jax.vmap(_Λ, (0, None)), (None, 0))(
        Q.x, jnp.arange(Λ2.n))  # n x n_q x d
    # Jj = jax.vmap(jacobian_determinant(F))(Q.x)  # n_q x 1
    wj = Q.w
    return jnp.einsum("ijk,jk,j->i", Λijk, Bjk, wj)


def oneformprojection(v):
    DF = jax.jacfwd(F)

    def _v(x):
        DFx = DF(x)
        return inv33(DFx) @ v(x)

    def _Λ(x, i):
        return Λ1(x, i)
    Bjk = jax.vmap(_v)(Q.x)  # n_q x d
    Λijk = jax.vmap(jax.vmap(_Λ, (0, None)), (None, 0))(
        Q.x, jnp.arange(Λ1.n))  # n x n_q x d
    Jj = jax.vmap(jacobian_determinant(F))(Q.x)  # n_q x 1
    wj = Q.w
    return jnp.einsum("ijk,jk,j,j->i", Λijk, Bjk, Jj, wj)

# %%


# %%
P_JxB = CrossProductProjection(Λ2, Λ1, Λ1, Q, F, En=E2, Em=E1, Ek=E1_f)
P_uxB = CrossProductProjection(Λ1, Λ2, Λ1, Q, F, En=E1, Em=E2, Ek=E1_f)
# %%
# JxB_hat = jnp.linalg.solve(M2, P_JxB(J_hat, H_hat))
# %%
# L_vec = DD + D1 @ jnp.linalg.solve(M1, D1.T)
# Reg = jnp.linalg.pinv(L_vec)
# Reg = (Reg + Reg.T) / 2
# %%
B_hat = jnp.linalg.solve(M2, E2m @ twoformprojection(B_xyz))
print(f"|B|^2: {B_hat @ M2 @ B_hat}")
print(f"|div B|^2: {D2 @ B_hat @ jnp.linalg.solve(M3, D2 @ B_hat)}")
q_hat = - L_pinv @ (D2 @ B_hat)
B_hat += jnp.linalg.solve(M2, D2.T @ q_hat)
print(f"|B|^2: {B_hat @ M2 @ B_hat}")
print(f"|div B|^2: {D2 @ B_hat @ jnp.linalg.solve(M3, D2 @ B_hat)}")

# %%
A_hat = L_vec_pinv @ D1.T @ B_hat
print(f"(A, B): {A_hat @ M21.T @ B_hat}")
# %%
# One step resisitive relaxation
Mat = jnp.block([[M2, 1/ns[0] * D1],
                 [D1.T, -M1]])
# %%
rhs = jnp.block([M2 @ B_hat, jnp.zeros(M1.shape[0])])
res = jnp.linalg.solve(Mat, rhs)

B_hat, J_hat = jnp.split(res, [M2.shape[0]])
print(f"|B|^2: {B_hat @ M2 @ B_hat}")
print(f"|J|^2: {J_hat @ M1 @ J_hat}")
# %%
print(f"|div B|^2: {D2 @ B_hat @ jnp.linalg.solve(M3, D2 @ B_hat)}")
A_hat = L_vec_pinv @ D1.T @ B_hat
B_harm_hat = B_hat - jnp.linalg.solve(M2, D1 @ A_hat)
print(f"(A, B + B_H): {A_hat @ M21.T @ (B_hat + B_harm_hat)}")

# %%
H_hat = jnp.linalg.solve(M1_f, M21_f.T @ B_hat)
JxB_hat = jnp.linalg.solve(M2, P_JxB(J_hat, H_hat))
q_hat = - L_pinv @ (D2 @ JxB_hat)
u_hat = JxB_hat + jnp.linalg.solve(M2, D2.T @ q_hat)

print(f"|div JxB|^2: {(D2 @ JxB_hat) @ jnp.linalg.solve(M3, D2 @ JxB_hat)}")
print(
    f"|div (JxB - grad p)|^2: {D2 @ u_hat @ jnp.linalg.solve(M3, D2 @ u_hat)}")
print(f"|JxB - grad p|^2: {u_hat @ M2 @ u_hat}")

grad_p_hat = jnp.linalg.solve(M2, D2.T @ q_hat)
grad_p_h = DiscreteFunction(grad_p_hat, Λ2, E2m)
grad_p_h_xyz = Pushforward(grad_p_h, F, 2)
# %%
B_hat_0 = B_hat.copy()
J_hat = jnp.zeros(M1.shape[0])
u_trace = []
E_trace = []
H_trace = []

dt = 0.1
eta = 0.00

Mat = jnp.block([[M2,           dt * eta * D1],
                 [0.5 * D1.T,   - M1]])


@jax.jit
def update(B_hat, J_hat):
    JxB_hat = jnp.linalg.solve(M2, P_JxB(J_hat, B_hat))
    q_hat = - L_pinv @ (D2 @ JxB_hat)
    u_hat = JxB_hat + jnp.linalg.solve(M2, D2.T @ q_hat)
    E_hat = jnp.linalg.solve(M1, P_uxB(u_hat, B_hat))
    rhs = jnp.block([M2 @ B_hat + dt * D1 @ E_hat, - 0.5 * D1.T @ B_hat])
    res = jnp.linalg.solve(Mat, rhs)
    B_hat, J_hat = jnp.split(res, [M2.shape[0]])
    return B_hat, J_hat, u_hat, q_hat


@jax.jit
def implicit_update(B_hat_guess, B_hat_0, dt, eta):
    B_hat_star = (B_hat_guess + B_hat_0) / 2
    H_hat = jnp.linalg.solve(M1_f, M21_f.T @ B_hat_star)
    J_hat = jnp.linalg.solve(M1, D1.T @ B_hat_star)

    JxB_hat = jnp.linalg.solve(M2, P_JxB(J_hat, H_hat))
    q_hat = - L_pinv @ (D2 @ JxB_hat)
    u_hat = JxB_hat + jnp.linalg.solve(M2, D2.T @ q_hat)
    E_hat = jnp.linalg.solve(M1, P_uxB(u_hat, H_hat))
    B_hat_1 = B_hat_0 + dt * jnp.linalg.solve(M2, D1 @ E_hat) \
        - eta * dt * jnp.linalg.solve(M2, D1 @ J_hat)
    return B_hat_1, J_hat, u_hat, q_hat


def picard_loop(B_hat, dt, eta, tol):
    B_hat_0 = B_hat
    B_hat_guess = B_hat
    B_hat_1, J_hat, u_hat, q_hat = implicit_update(
        B_hat_guess, B_hat_0, dt, eta)
    delta = (B_hat_1 - B_hat_guess) @ M2 @ (B_hat_1 - B_hat_guess)
    while delta > tol:
        B_hat_guess = B_hat_1
        B_hat_1, J_hat, u_hat, q_hat = implicit_update(
            B_hat_guess, B_hat_0, dt, eta)
        delta = (B_hat_1 - B_hat_guess) @ M2 @ (B_hat_1 - B_hat_guess)
    return B_hat_1, J_hat, u_hat, q_hat


# %%

for i in range(25000):
    start = time.time()
    # B_hat, J_hat, u_hat, q_hat = update(B_hat, J_hat)
    B_hat, J_hat, u_hat, q_hat = picard_loop(
        B_hat, dt=0.1, eta=0*1e-5, tol=1e-12)
    end = time.time()
    # print(f"Iteration {i}, time: {end - start}")
    A_hat = L_vec_pinv @ D1.T @ B_hat
    B_harm_hat = B_hat - jnp.linalg.solve(M2, D1 @ A_hat)
    u_trace.append(u_hat @ M2 @ u_hat)
    E_trace.append(B_hat @ M2 @ B_hat)
    H_trace.append(A_hat @ M21.T @ (B_hat + B_harm_hat))
    if i % 100 == 0:
        print(f"Iteration {i}, u norm: {u_trace[-1]}")


# Island seeding
# %%
# B_h = DiscreteFunction(B_hat_0, Λ2, E2)
# B_h_xyz = Pushforward(B_h, F, 2)

# def A_par(x):
#     r, χ, z = x
#     return jnp.exp(-(r - 0.66)**2 / 0.05) * jnp.cos(χ * 4 * jnp.pi)


# def dE(x):
#     Bx = B_h_xyz(x)
#     return Bx / (Bx @ Bx)**0.5 * A_par(x)


# dE_hat = jnp.linalg.solve(M1, E1 @ oneformprojection(dE))
# dE_h = DiscreteFunction(dE_hat, Λ1, E1)
# dE_h_xyz = Pushforward(dE_h, F, 1)
# # %%
# dB = jnp.linalg.solve(M2, D1 @ dE_hat)
# dB_h = DiscreteFunction(dB, Λ2, E2)
# dB_h_xyz = Pushforward(dB_h, F, 2)

# %%
# plt.contourf(_y1, _y3, jax.vmap(dB_h_xyz)(_x)[:, 1].reshape(nx, nx), levels=20)
# plt.quiver(__y1, __y3,
#            jax.vmap(dB_h_xyz)(__x)[:, 0].reshape(_nx, _nx),
#            jax.vmap(dB_h_xyz)(__x)[:, 2].reshape(_nx, _nx), color="k")
# plt.xlim(R0-1, R0+1)
# plt.ylim(-1, 1)
# plt.colorbar()

# %%
H_hat = jnp.linalg.solve(M1_f, M21_f.T @ B_hat)
JxB_hat = jnp.linalg.solve(M2, P_JxB(J_hat, H_hat))
print(f"|div JxB|^2: {(D2 @ JxB_hat) @ jnp.linalg.solve(M3, D2 @ JxB_hat)}")
print(
    f"|div (JxB - grad p)|^2: {D2 @ u_hat @ jnp.linalg.solve(M3, D2 @ u_hat)}")
print(f"|JxB - grad p|^2: {u_hat @ M2 @ u_hat}")
print(f"|JxB|^2: {JxB_hat @ M2 @ JxB_hat}")
print(f"|B|^2: {B_hat @ M2 @ B_hat}")

# %%
plt.plot(u_trace)
plt.yscale('log')
plt.xlabel("Iteration")
plt.ylabel("||u_h||^2")

# %%
plt.plot(E_trace)
plt.xlabel("Iteration")
plt.ylabel("||B_h||^2")

# %%
plt.plot(np.array(H_trace) - H_trace[0])
plt.xlabel("Iteration")
plt.ylabel("Helicity - H(0)")



# %%

# %%
# Poincare plot
B_h = DiscreteFunction(B_hat, Λ2, E2m)
B_h_xyz = Pushforward(B_h, F, 2)


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


def vector_field(x):
    DFx = jax.jacfwd(F)(x)
    norm = ((DFx @ B_h(x)) @ DFx @ B_h(x))**0.5 / (jnp.linalg.det(DFx) + 1e-12)
    return B_h(x) / (jnp.linalg.det(DFx) * norm + 1e-12)


# %%
x0 = jnp.linspace(5e-2, 1-5e-2, 20)
x0_1 = jnp.array([x0, jnp.zeros_like(x0), jnp.ones_like(x0)/7]).T
x0_2 = jnp.array([x0, jnp.ones_like(x0)/4, jnp.zeros_like(x0)]).T
x0_3 = jnp.array([x0, jnp.ones_like(x0)/2, jnp.zeros_like(x0)]).T
x0_4 = jnp.array([x0, 3*jnp.ones_like(x0)/4, jnp.zeros_like(x0)]).T
# x0 = jnp.concatenate([x0_1, x0_2, x0_3, x0_4], axis=0)
x0 = x0_1
# %%
trajectories = jax.vmap(lambda x: fieldline(
    x, vector_field, 0.025, 20_000))(x0)
# %%
physical_trajectories = jax.vmap(F)(trajectories.reshape(-1, 3))
physical_trajectories = physical_trajectories.reshape(
    trajectories.shape[0], trajectories.shape[1], 3)
# %%

import matplotlib.gridspec as gridspec

# Dummy data for demonstration if 'trajectories' and 'physical_trajectories' are not defined
# In a real scenario, these would come from your calculations.
if 'trajectories' not in locals():
    num_points = 500
    trajectories = [np.random.rand(num_points, 3) * 10 for _ in range(3)]
    trajectories[0][:, 0] = trajectories[0][:, 0] - 5 # Example x values
    trajectories[1][:, 0] = trajectories[1][:, 0] # Example x values
    trajectories[2][:, 0] = trajectories[2][:, 0] + 5 # Example x values


if 'physical_trajectories' not in locals():
    R0 = 5 # Example value for R0
    num_physical_points = 1000
    physical_trajectories = []
    # Create some dummy data that crosses the R0 threshold
    t1 = np.random.rand(num_physical_points // 2, 3) * 2
    t1[:, 0] = t1[:, 0] - (R0 + 1) # Left side
    physical_trajectories.append(t1)

    t2 = np.random.rand(num_physical_points // 2, 3) * 2
    t2[:, 0] = t2[:, 0] + (R0 + 1) # Right side
    physical_trajectories.append(t2)


# --- PLOT SETTINGS FOR SLIDES ---
FIG_SIZE = (12, 6)      # Figure size in inches (width, height)
TITLE_SIZE = 20         # Font size for the plot title
LABEL_SIZE = 20         # Font size for x and y axis labels
TICK_SIZE = 16          # Font size for x and y tick labels
LEGEND_SIZE = 16        # Font size for the legend (not directly used here, but good to keep)
LINE_WIDTH = 2.5        # Width of the plot lines (not directly used here)
# ---------------------------------

# Create a figure with two subplots next to each other
fig = plt.figure(figsize=FIG_SIZE)
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.05)

# Left subplot: x < -2
ax1 = fig.add_subplot(gs[0])
# Right subplot: x > 2
ax2 = fig.add_subplot(gs[1], sharey=ax1)

# Turn off tick labels on right of left plot and left of right plot
ax1.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.yaxis.tick_right()
ax1.yaxis.tick_left()
ax2.tick_params(labelleft=False)

# Define primary and secondary colors for the scatter plots
primary_color = 'purple'
secondary_color = 'teal'
tertiary_color = 'orange'
quaternary_color = 'black'

colors = [primary_color, secondary_color, tertiary_color, quaternary_color]

for i, t in enumerate(physical_trajectories):
    x = np.array(t[:, 0])
    z = np.array(t[:, 2])
    alpha = np.exp(-np.array(t[:, 1])**2 / 0.02)

    mask_left = x < -R0 + 1
    mask_right = x > R0 - 1

    current_color = colors[i % len(colors)] # Cycle through the defined colors

    if np.any(mask_left):
        ax1.scatter(x[mask_left], z[mask_left], s=0.1, alpha=alpha[mask_left], color=current_color)

    if np.any(mask_right):
        ax2.scatter(x[mask_right], z[mask_right], s=0.1, alpha=alpha[mask_right], color=current_color)

# Set labels and titles with specified font sizes
ax1.set_xlabel(r'$x$', fontsize=LABEL_SIZE)
ax2.set_xlabel(r'$x$', fontsize=LABEL_SIZE)
ax1.set_ylabel(r'$z$', fontsize=LABEL_SIZE)
# fig.suptitle(r'Field line intersections', fontsize=TITLE_SIZE)

# Set x limits for both subplots
ax1.set_xlim(-R0 - 0.9, -R0 + 0.9)
ax2.set_xlim(R0 - 0.9, R0 + 0.9)

# Set tick parameters for both axes
ax1.tick_params(axis='x', labelsize=TICK_SIZE)
ax1.tick_params(axis='y', labelsize=TICK_SIZE)
ax2.tick_params(axis='x', labelsize=TICK_SIZE)
ax2.tick_params(axis='y', labelsize=TICK_SIZE) # Although labelleft=False, still good to set size for potential future use

# Adjust layout
fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect to prevent suptitle overlap

# fig.savefig('poincare_solovev_physical.png', bbox_inches='tight', dpi=800)

# %%
plt.figure(figsize=FIG_SIZE)

for i, t in enumerate(trajectories):
    current_color = colors[i % len(colors)] # Cycle through the defined colors
    plt.scatter(t[:, 0], t[:, 1], s=0.1, alpha=jnp.exp(-t[:, 2]**2/0.02), color=current_color)

# plt.title(r'Field line intersections', fontsize=TITLE_SIZE)
plt.xlabel(r'$r$', fontsize=LABEL_SIZE)
plt.ylabel(r'$\chi$', fontsize=LABEL_SIZE)

# Set tick parameters
plt.xticks(fontsize=TICK_SIZE)
plt.yticks(fontsize=TICK_SIZE)

plt.tight_layout() # Adjust layout to prevent labels from overlapping

# plt.savefig('poincare_solovev_logical.png', bbox_inches='tight', dpi=800)

# %% Figure 1: Energy and Force
fig1, ax1 = plt.subplots(figsize=FIG_SIZE)

# Plot Energy on the left y-axis (ax1)
color1 = 'purple'
ax1.set_xlabel(r'$n$', fontsize=LABEL_SIZE)
ax1.set_ylabel(r'$\frac{1}{2} \| B \|^2$', color=color1, fontsize=LABEL_SIZE)
ax1.plot(jnp.array(E_trace), label=r'$\frac{1}{2} \| B \|^2$', color=color1, lw=LINE_WIDTH)
# ax1.plot(jnp.pi * jnp.array(H_trace), label=r'$\pi \, (A, B)$', color=color1, linestyle="--", lw=LINE_WIDTH)
ax1.tick_params(axis='y', labelcolor=color1, labelsize=TICK_SIZE)
ax1.tick_params(axis='x', labelsize=TICK_SIZE) # Set x-tick size

# Create a second y-axis that shares the same x-axis
ax2 = ax1.twinx()

# Plot Force on the right y-axis (ax2)
color2 = 'black'
ax2.set_ylabel(r'$\|J \times B - \nabla p\|^2, \quad | H - H_0 |$', color=color2, fontsize=LABEL_SIZE)
ax2.plot(u_trace, label=r'$\|J \times B - \nabla p \|^2$', color=color2, lw=LINE_WIDTH)
ax2.tick_params(axis='y', labelcolor=color2, labelsize=TICK_SIZE)
ax2.set_ylim(0.5 * min(u_trace), 2 * max(u_trace))  # Set y-limits for better visibility
ax2.set_yscale('log')

relative_helicity_change = jnp.abs(jnp.array(jnp.array(H_trace) - H_trace[0]))
ax2.plot(relative_helicity_change, label=r'$| H - H_0 |$', color='darkgray', linestyle='--', lw=LINE_WIDTH)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=LEGEND_SIZE)

fig1.tight_layout()
plt.show()

# fig1.savefig('solovev_force.pdf', bbox_inches='tight')
# %%