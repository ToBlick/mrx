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
# plt.contourf(_y1, _y3, jax.vmap(psi)(_x).reshape(_y1.shape), levels=20)
# vals = jax.vmap(psi)(_x_1d)
# plt.plot(_y1_1d, vals - vals[0], 'k', label=r'$\psi(r, 0, 0)$')
# plt.plot(_y1_1d, jnp.zeros_like(_y2_1d), ':k')
# plt.axis('equal')
# plt.legend()
# plt.colorbar()
# plt.xlabel("R")
# plt.ylabel("Z")


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
JxB_hat = jnp.linalg.solve(M2, P_JxB(J_hat, H_hat))
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

for i in range(100):
    start = time.time()
    # B_hat, J_hat, u_hat, q_hat = update(B_hat, J_hat)
    B_hat, J_hat, u_hat, q_hat = picard_loop(
        B_hat, dt=0.1, eta=0.00, tol=1e-12)
    end = time.time()
    print(f"Iteration {i}, time: {end - start}")
    A_hat = L_vec_pinv @ D1.T @ B_hat
    B_harm_hat = B_hat - jnp.linalg.solve(M2, D1 @ A_hat)
    u_trace.append(u_hat @ M2 @ u_hat)
    E_trace.append(B_hat @ M2 @ B_hat)
    H_trace.append(A_hat @ M21.T @ (B_hat + B_harm_hat))
    if i % 10 == 0:
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
plt.xlabel("Iteration")
plt.ylabel("||u_h||^2")

# %%
plt.plot(E_trace)
plt.xlabel("Iteration")
plt.ylabel("||B_h||^2")

# %%
plt.plot(np.array(H_trace))
plt.xlabel("Iteration")
plt.ylabel("Helicity - H(0)")
# %%
q_h = DiscreteFunction(q_hat, Λ3, E3.matrix())
q_h_xyz = Pushforward(q_h, F, 3)
# %%
tol = 1e-3
nx = 64
_nx = 16
_x1 = jnp.linspace(tol, 1 - tol, nx)
_x2 = jnp.linspace(0, 1, nx)
_x3 = jnp.ones(1) * 0
_x = jnp.array(jnp.meshgrid(_x1, _x2, _x3))
_x = _x.transpose(1, 2, 3, 0).reshape(nx**2, 3)
_y = jax.vmap(F)(_x)
_y1 = _y[:, 0].reshape(nx, nx)
_y3 = _y[:, 2].reshape(nx, nx)
__x1 = jnp.linspace(tol, 1 - tol, _nx)
__x2 = jnp.linspace(0, 1, _nx)
__x3 = jnp.ones(1) * 0
__x = jnp.array(jnp.meshgrid(__x1, __x2, __x3))
__x = __x.transpose(1, 2, 3, 0).reshape(_nx**2, 3)
__y = jax.vmap(F)(__x)
__y1 = __y[:, 0].reshape(_nx, _nx)
__y3 = __y[:, 2].reshape(_nx, _nx)
plt.contourf(_y1, _y3, jax.vmap(q_h_xyz)(_x).reshape(nx, nx), levels=10)
plt.xlim(R0-1, R0+1)
plt.ylim(-1, 1)
plt.xlabel("R")
plt.ylabel("Z")
plt.colorbar()
# %%
u_h = DiscreteFunction(u_hat, Λ2, E2m)
u_h_xyz = Pushforward(u_h, F, 2)

# %%
plt.contourf(_y1, _y3, jax.vmap(u_h_xyz)(
    _x)[:, 1].reshape(nx, nx), levels=20)
plt.quiver(__y1, __y3,
           jax.vmap(u_h_xyz)(__x)[:, 0].reshape(_nx, _nx),
           jax.vmap(u_h_xyz)(__x)[:, 2].reshape(_nx, _nx), color="k")
plt.xlim(R0-1, R0+1)
plt.ylim(-1, 1)
plt.title("u_h")
plt.colorbar()
# %%
B_h = DiscreteFunction(B_hat, Λ2, E2m)
B_h_xyz = Pushforward(B_h, F, 2)
J_h = DiscreteFunction(J_hat, Λ1, E1m)
J_h_xyz = Pushforward(J_h, F, 1)
# %%
plt.contourf(_y1, _y3, jax.vmap(B_h_xyz)(
    _x)[:, 1].reshape(nx, nx), levels=20)
plt.quiver(__y1, __y3, jax.vmap(B_h_xyz)(__x)[:, 0].reshape(_nx, _nx),
           jax.vmap(B_h_xyz)(__x)[:, 2].reshape(_nx, _nx), color="k")
plt.xlim(R0-1, R0+1)
plt.ylim(-1, 1)
plt.title("B_h")
plt.colorbar()
# %%
plt.contourf(_y1, _y3, jax.vmap(J_h_xyz)(
    _x)[:, 1].reshape(nx, nx), levels=20)
plt.quiver(__y1, __y3, jax.vmap(J_h_xyz)(__x)[:, 0].reshape(_nx, _nx),
           jax.vmap(J_h_xyz)(__x)[:, 2].reshape(_nx, _nx), color="k")
plt.xlim(R0-1, R0+1)
plt.ylim(-1, 1)
plt.title("J_h")
plt.colorbar()
# %%
plt.contourf(_y1, _y3, jax.vmap(grad_p_h_xyz)(
    _x)[:, 1].reshape(nx, nx), levels=20)
plt.quiver(__y1, __y3,
           jax.vmap(grad_p_h_xyz)(__x)[:, 0].reshape(_nx, _nx),
           jax.vmap(grad_p_h_xyz)(__x)[:, 2].reshape(_nx, _nx), color="k")
plt.xlim(R0-1, R0+1)
plt.ylim(-1, 1)
plt.title("grad p_h")
plt.colorbar()
# %%
JxB_h = DiscreteFunction(JxB_hat, Λ2, E2)
JxB_h_xyz = Pushforward(JxB_h, F, 2)
# %%
plt.contourf(_y1, _y3, jax.vmap(JxB_h_xyz)(_x)
             [:, 1].reshape(nx, nx), levels=20)
plt.quiver(__y1, __y3, jax.vmap(JxB_h_xyz)(__x)[:, 0].reshape(_nx, _nx),
           jax.vmap(JxB_h_xyz)(__x)[:, 2].reshape(_nx, _nx), color="k")
plt.xlim(R0-1, R0+1)
plt.ylim(-1, 1)
plt.title("JxB_h")
plt.colorbar()

# %%
tol = 1e-3
nx = 64
_nx = 16
_x1 = jnp.linspace(tol, 1 - tol, nx)
_x2 = jnp.ones(1) * 0
_x3 = jnp.ones(1) * 0
_x = jnp.array(jnp.meshgrid(_x1, _x2, _x3))
_x = _x.transpose(1, 2, 3, 0).reshape(nx, 3)
_y = jax.vmap(F)(_x)
_y1 = _y[:, 0]
_y3 = _y[:, 2]

# %%
plt.plot(_y1, jax.vmap(B_h_xyz)(_x)[:, 0], label="Bx")
plt.plot(_y1, jax.vmap(B_h_xyz)(_x)[:, 1], label="By")
plt.plot(_y1, jax.vmap(B_h_xyz)(_x)[:, 2], label="Bz")
plt.plot(_y1, (0.95 * jax.vmap(B_h_xyz)(_x)
         [:, 2]) / (R0 * jax.vmap(B_h_xyz)(_x)[:, 1]), label="r Bz / R By")
plt.plot(_y1, jax.vmap(J_h_xyz)(_x)[:, 0], label="Jx")
plt.plot(_y1, jax.vmap(J_h_xyz)(_x)[:, 1], label="Jy")
plt.plot(_y1, jax.vmap(J_h_xyz)(_x)[:, 2], label="Jz")
plt.plot(_y1, 100 * jax.vmap(q_h_xyz)(_x), label="p")
plt.plot(_y1, jax.vmap(grad_p_h_xyz)(_x)[:, 0], label="grad px")
# plt.plot(_y1, jax.vmap(grad_p_h_xyz)(_x)[:, 1], label="grad py")
# plt.plot(_y1, jax.vmap(grad_p_h_xyz)(_x)[:, 2], label="grad pz")
plt.plot(_y1, jax.vmap(JxB_h_xyz)(_x)[:, 0], label="JxB x")
# plt.plot(_y1, jax.vmap(JxB_h_xyz)(_x)[:, 1], label="JxB y")
# plt.plot(_y1, jax.vmap(JxB_h_xyz)(_x)[:, 2], label="JxB z")
plt.legend()

# %%
# Poincare plot
B_h = DiscreteFunction(B_hat, Λ2, E2)
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
x0 = jnp.linspace(1e-2, 1-1e-2, 7)
x0_1 = jnp.array([x0, jnp.zeros_like(x0), jnp.zeros_like(x0)]).T
x0_2 = jnp.array([x0, jnp.ones_like(x0)/4, jnp.zeros_like(x0)]).T
x0_3 = jnp.array([x0, jnp.ones_like(x0)/2, jnp.zeros_like(x0)]).T
x0_4 = jnp.array([x0, 3*jnp.ones_like(x0)/4, jnp.zeros_like(x0)]).T
x0 = jnp.concatenate([x0_1, x0_2, x0_3, x0_4], axis=0)
# %%
trajectories = jax.vmap(lambda x: fieldline(
    x, vector_field, 0.1, 10_000))(x0)
# %%
physical_trajectories = jax.vmap(F)(trajectories.reshape(-1, 3))
physical_trajectories = physical_trajectories.reshape(
    trajectories.shape[0], trajectories.shape[1], 3)
# %%
# plt.figure(figsize=(8, 6))
# for t in physical_trajectories:
#     plt.scatter(t[:, 0], t[:, 2], s=0.1, alpha=jnp.exp(-t[:, 1]**2/0.01))
# plt.title('Field line intersections')
# plt.xlabel('x')
# plt.ylabel('z')
# %%
plt.figure(figsize=(8, 6))
for t in trajectories:
    plt.scatter(t[:, 0], t[:, 1], s=0.1, alpha=jnp.exp(-t[:, 2]**2/0.01))
plt.title('Field line intersections')
plt.xlabel('r')
plt.ylabel('chi')

# %%


# Create a figure with two subplots next to each other
fig = plt.figure(figsize=(10, 6))
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

# Draw diagonal slashes for broken axis effect
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot([1, 1], [0.1, -0.1], **kwargs)  # slash on right of left plot
kwargs['transform'] = ax2.transAxes
ax2.plot([0, 0], [0.1, -0.1], **kwargs)  # slash on left of right plot

# Now plot the data in the appropriate subplots
for t in physical_trajectories:
    x = np.array(t[:, 0])
    z = np.array(t[:, 2])
    alpha = np.exp(-np.array(t[:, 1])**2 / 0.01)

    mask_left = x < -R0+1
    mask_right = x > R0 - 1
    if np.any(mask_left):
        ax1.scatter(x[mask_left], z[mask_left], s=0.1, alpha=alpha[mask_left])

    if np.any(mask_right):
        ax2.scatter(x[mask_right], z[mask_right],
                    s=0.1, alpha=alpha[mask_right])

# Set labels and titles
ax1.set_xlabel('x')
ax2.set_xlabel('x')
ax1.set_ylabel('z')
fig.suptitle('Field line intersections')

# Set x limits for both subplots
ax1.set_xlim(-R0-1, -R0+1)
ax2.set_xlim(R0 - 1, R0 + 1)

plt.show()

# %%
p_hat = -jnp.linalg.pinv(K) @ M03_f.T @ jnp.linalg.solve(M3, D2 @ JxB_hat)
p_h = DiscreteFunction(p_hat, Λ0, E0_f)
p_h_xyz = Pushforward(p_h, F, 0)

# %%
tol = 1e-3
nx = 64
_nx = 16
_x1 = jnp.linspace(tol, 1 - tol, nx)
_x2 = jnp.linspace(0, 1, nx)
_x3 = jnp.ones(1) * 0
_x = jnp.array(jnp.meshgrid(_x1, _x2, _x3))
_x = _x.transpose(1, 2, 3, 0).reshape(nx**2, 3)
_y = jax.vmap(F)(_x)
_y1 = _y[:, 0].reshape(nx, nx)
_y3 = _y[:, 2].reshape(nx, nx)
__x1 = jnp.linspace(tol, 1 - tol, _nx)
__x2 = jnp.linspace(0, 1, _nx)
__x3 = jnp.ones(1) * 0
__x = jnp.array(jnp.meshgrid(__x1, __x2, __x3))
__x = __x.transpose(1, 2, 3, 0).reshape(_nx**2, 3)
__y = jax.vmap(F)(__x)
__y1 = __y[:, 0].reshape(_nx, _nx)
__y3 = __y[:, 2].reshape(_nx, _nx)
plt.contourf(_y1, _y3, jax.vmap(p_h_xyz)(_x).reshape(nx, nx), levels=20)
plt.xlim(R0-1, R0+1)
plt.ylim(-1, 1)
plt.xlabel("R")
plt.ylabel("Z")
plt.colorbar()
# %%
