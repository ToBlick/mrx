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
from mrx.DeRhamSequence import DeRhamSequence
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
from mrx.Utils import jacobian_determinant, l2_product

jax.config.update("jax_enable_x64", True)

# %%
R0 = 3.0
μ0 = 1.0
π = jnp.pi
k0 = 1.3
q0 = 1.5
F0 = 3
aR = 0.66

###
# ψ(R, Z) =  (¼ k₀² (R² - R₀²)² + R²Z² ) / (2 R₀² k₀ q₀)
###

p_map = 3
n_map = 8
q_map = 2 * p_map

a_hat = solovev_lcfs_fit(n_map, p_map, q_map, R0, a=0.6, k0=k0, q0=q0)

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
ns = (8, 8, 1)
ps = (3, 3, 0)
types = ("clamped", "periodic", "constant")


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
# Set up finite element spaces
bcs = ('dirichlet', 'none', 'none')

Seq = DeRhamSequence(ns, ps, q, types, F, polar=True)
# DualSeq = DeRhamSequence(ns, ps, q, types, ('dirichlet', 'none', 'none'), F, polar=True, Λ0_ijk = PrimalSeq.Λ0_ijk)

# %%

M0 = Seq.assemble_M0_0()
M1 = Seq.assemble_M1_0()
M2 = Seq.assemble_M2_0()
M3 = Seq.assemble_M3_0()

M1_dual = Seq.assemble_M1()

###
# Operators
###

grad = jnp.linalg.solve(M1, Seq.assemble_grad_0())
curl = jnp.linalg.solve(M2, Seq.assemble_curl_0())
dvg = jnp.linalg.solve(M3, Seq.assemble_dvg_0())
weak_grad = -jnp.linalg.solve(M2, Seq.assemble_dvg_0().T)
weak_curl = jnp.linalg.solve(M1, Seq.assemble_curl_0().T)
weak_dvg = -jnp.linalg.solve(M0, Seq.assemble_grad_0().T)

curlcurl = jnp.linalg.solve(M1, Seq.assemble_curlcurl_0())
graddiv = - jnp.linalg.solve(M2, Seq.assemble_divdiv_0())

laplace_0 = Seq.assemble_gradgrad_0()                        # dim ker = 0
laplace_1 = Seq.assemble_curlcurl_0() - M1 @ grad @ weak_dvg  # dim ker = 0 (no voids)
laplace_2 = M2 @ curl @ weak_curl + \
    Seq.assemble_divdiv_0()  # dim ker = 1 (one tunnel)
laplace_3 = - M3 @ dvg @ weak_grad  # dim ker = 1 (constants)

# from H₀(div) to H(curl)
P1 = jnp.linalg.solve(Seq.assemble_M1(), Seq.assemble_P2().T)

M12 = Seq.assemble_M12_0()


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

    Bx = BR * jnp.cos(2 * π * phi) - BPhi * jnp.sin(2 * π * phi)
    By = BR * jnp.sin(2 * π * phi) + BPhi * jnp.cos(2 * π * phi)

    return jnp.array([Bx, By, Bz])


# %%
P_JxH = CrossProductProjection(
    Seq.Λ2, Seq.Λ1, Seq.Λ1, Seq.Q, Seq.F,
    En=Seq.E2_0, Em=Seq.E1_0, Ek=Seq.E1,
    Λn_ijk=Seq.Λ2_ijk, Λm_ijk=Seq.Λ1_ijk, Λk_ijk=Seq.Λ1_ijk,
    J_j=Seq.J_j, G_jkl=Seq.G_jkl, G_inv_jkl=Seq.G_inv_jkl)
P_JxB = CrossProductProjection(
    Seq.Λ2, Seq.Λ1, Seq.Λ2, Seq.Q, Seq.F,
    En=Seq.E2_0, Em=Seq.E1_0, Ek=Seq.E2_0,
    Λn_ijk=Seq.Λ2_ijk, Λm_ijk=Seq.Λ1_ijk, Λk_ijk=Seq.Λ2_ijk,
    J_j=Seq.J_j, G_jkl=Seq.G_jkl, G_inv_jkl=Seq.G_inv_jkl)
P_uxH = CrossProductProjection(
    Seq.Λ1, Seq.Λ2, Seq.Λ1, Seq.Q, Seq.F,
    En=Seq.E1_0, Em=Seq.E2_0, Ek=Seq.E1,
    Λn_ijk=Seq.Λ1_ijk, Λm_ijk=Seq.Λ2_ijk, Λk_ijk=Seq.Λ1_ijk,
    J_j=Seq.J_j, G_jkl=Seq.G_jkl, G_inv_jkl=Seq.G_inv_jkl)
P_uxB = CrossProductProjection(
    Seq.Λ1, Seq.Λ2, Seq.Λ2, Seq.Q, Seq.F,
    En=Seq.E1_0, Em=Seq.E2_0, Ek=Seq.E2_0,
    Λn_ijk=Seq.Λ1_ijk, Λm_ijk=Seq.Λ2_ijk, Λk_ijk=Seq.Λ2_ijk,
    J_j=Seq.J_j, G_jkl=Seq.G_jkl, G_inv_jkl=Seq.G_inv_jkl)
# %%
P_Leray = jnp.eye(M2.shape[0]) + \
    weak_grad @ jnp.linalg.pinv(laplace_3) @ M3 @ dvg

# %%
# B_hat = jnp.linalg.solve(M2, Seq.P2_0(B_xyz))
# print(f"|B|^2: {B_hat @ M2 @ B_hat}")
# print(f"|div B|^2: {dvg @ B_hat @ M3 @ dvg @ B_hat}")
# B_hat = P_Leray @ B_hat
# print(f"|B|^2: {B_hat @ M2 @ B_hat}")
# print(f"|div B|^2: {dvg @ B_hat @ M3 @ dvg @ B_hat}")

# # %%
# # One step of resisitive relaxation to get J x n = 0 on ∂Ω
# B_hat = jnp.linalg.solve(jnp.eye(M2.shape[0]) + 1e-2 * curl @ weak_curl, B_hat)
# J_hat = weak_curl @ B_hat
# # %%

# # J_h = Pushforward(DiscreteFunction(J_hat, Seq.Λ1, Seq.E1_0.matrix()), F, 1)
# # B_h = Pushforward(DiscreteFunction(B_hat, Seq.Λ2, Seq.E2_0.matrix()), F, 2)
# # plt.plot(_y1_1d, jax.vmap(J_h)(_x_1d)[:, 0], label=r'$J_x$')
# # plt.plot(_y1_1d, jax.vmap(J_h)(_x_1d)[:, 1], label=r'$J_y$')
# # plt.plot(_y1_1d, jax.vmap(J_h)(_x_1d)[:, 2], label=r'$J_z$')
# # plt.plot(_y1_1d, jax.vmap(B_h)(_x_1d)[:, 0], '--', label=r'$B_x$')
# # plt.plot(_y1_1d, jax.vmap(B_h)(_x_1d)[:, 1], '--', label=r'$B_y$')
# # plt.plot(_y1_1d, jax.vmap(B_h)(_x_1d)[:, 2], '--', label=r'$B_z$')
# # plt.legend()

# # %%
# print(f"|B|^2: {B_hat @ M2 @ B_hat}")
# print(f"|J|^2: {J_hat @ M1 @ J_hat}")
# # %%
# print(f"|div B|^2: {dvg @ B_hat @ M3 @ dvg @ B_hat}")
# A_hat = jnp.linalg.solve(laplace_1, M1 @ weak_curl @ B_hat)
# B_harm_hat = B_hat - curl @ A_hat
# print(f"(A, B + B_H): {A_hat @ M12 @ (B_hat + B_harm_hat)}")
# %%
# Set up inital condition
B_hat = P_Leray @ jnp.linalg.solve(M2, Seq.P2_0(B_xyz))
# One step of resisitive relaxation to get J x n = 0 on ∂Ω
B_hat = jnp.linalg.solve(jnp.eye(M2.shape[0]) + 1e-2 * curl @ weak_curl, B_hat)


# A_hat = L_vec_pinv @ M1 @ weak_curl @ B_hat
A_hat = jnp.linalg.solve(laplace_1, M1 @ weak_curl @ B_hat)
B_harm_hat = B_hat - curl @ A_hat
print(f"|div B_harm|^2: {(dvg @ B_harm_hat).T @ M3 @ dvg @ B_harm_hat}")
print(
    f"|curl B_harm|^2: {(weak_curl @ B_harm_hat) @ M1 @ (weak_curl @ B_harm_hat)}")

u_trace = []
E_trace = []
H_trace = []
dvg_trace = []

dt = 0.001
eta = 0.00

# %%


@jax.jit
def update(B_hat):
    J_hat = weak_curl @ B_hat
    JxB_hat = jnp.linalg.solve(M2, P_JxB(J_hat, B_hat))
    u_hat = P_Leray @ JxB_hat
    E_hat = jnp.linalg.solve(M1, P_uxB(u_hat, B_hat)) - eta * J_hat
    B_hat += dt * curl @ E_hat
    return B_hat, J_hat, u_hat


@jax.jit
def implicit_update(B_hat_guess, B_hat_0, dt, eta):
    B_hat_star = (B_hat_guess + B_hat_0) / 2
    J_hat = weak_curl @ B_hat_star

    H_hat = P1 @ B_hat_star
    JxH_hat = jnp.linalg.solve(M2, P_JxH(J_hat, H_hat))
    u_hat = P_Leray @ JxH_hat
    E_hat = jnp.linalg.solve(M1, P_uxH(u_hat, H_hat)) - eta * J_hat

    B_hat_1 = B_hat_0 + dt * curl @ E_hat
    return B_hat_1, J_hat, u_hat


def picard_loop(B_hat, dt, eta, tol):
    B_hat_0 = B_hat
    B_hat_guess = B_hat
    B_hat_1, J_hat, u_hat = implicit_update(
        B_hat_guess, B_hat_0, dt, eta)
    delta = (B_hat_1 - B_hat_guess) @ M2 @ (B_hat_1 - B_hat_guess)
    while delta > tol:
        B_hat_guess = B_hat_1
        B_hat_1, J_hat, u_hat = implicit_update(
            B_hat_guess, B_hat_0, dt, eta)
        delta = (B_hat_1 - B_hat_guess) @ M2 @ (B_hat_1 - B_hat_guess)
    return B_hat_1, J_hat, u_hat


# %%

for i in range(1000):
    # B_hat, J_hat, u_hat = update(B_hat)
    B_hat, J_hat, u_hat = picard_loop(B_hat, dt=0.1, eta=0.00, tol=1e-12)
    A_hat = jnp.linalg.solve(laplace_1, M1 @ weak_curl @ B_hat)
    u_trace.append(u_hat @ M2 @ u_hat)
    E_trace.append(B_hat @ M2 @ B_hat / 2)
    H_trace.append(A_hat @ M12 @ (B_hat + B_harm_hat))
    dvg_trace.append(dvg @ B_hat @ M3 @ dvg @ B_hat)
    if i % 100 == 0:
        print(f"Iteration {i}, u norm: {u_trace[-1]}")

# %%
plt.plot(jnp.sqrt(jnp.array(u_trace)))
plt.xlabel("Iteration")
plt.ylabel("||u_h||")
plt.yscale("log")

# %%
plt.plot(E_trace)
plt.xlabel("Iteration")
plt.ylabel("½||B_h||^2")
plt.yscale("log")

# %%
plt.plot(jnp.array(H_trace))
plt.xlabel("Iteration")
plt.ylabel("Helicity - H(0)")
# %%
plt.plot(jnp.sqrt(jnp.array(dvg_trace)))
plt.xlabel("Iteration")
plt.ylabel("||div B_h||")
plt.yscale("log")

# %%
