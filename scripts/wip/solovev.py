# %%
import os
import time
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from mrx.DifferentialForms import DifferentialForm, DiscreteFunction, Flat, Pushforward
from mrx.LazyMatrices import LazyDerivativeMatrix, LazyMassMatrix, LazyProjectionMatrix
from mrx.PolarMapping import LazyExtractionOperator, get_xi
from mrx.Projectors import CurlProjection, EFieldProjector, Projector
from mrx.Quadrature import QuadratureRule
from mrx.Utils import inv33, jacobian_determinant

# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)

# Create output directory for figures
os.makedirs("script_outputs", exist_ok=True)

a = 1
ɛ = 0.35
q_star = 1.5
β_t = 0.12
B0 = 1.0
μ0 = 1.0

R0 = a / ɛ
π = jnp.pi
ν = β_t * q_star**2 / ɛ


n = 6
p = 3
q = 2*p
ns = (n, n, 1)
ps = (p, p, 0)
types = ("clamped", "periodic", "constant")


def _X(r, χ):
    return jnp.ones(1) * (R0 + a * r * jnp.cos(2 * π * χ))


def _Y(r, χ):
    return jnp.ones(1) * (R0 + a * r * jnp.cos(2 * π * χ))


def _Z(r, χ):
    return jnp.ones(1) * a * r * jnp.sin(2 * π * χ)


def F(x):
    """Polar coordinate mapping function."""
    r, χ, z = x
    return jnp.ravel(jnp.array([_X(r, χ) * jnp.cos(2 * π * z),
                                -_Y(r, χ) * jnp.sin(2 * π * z),
                                _Z(r, χ)]))


# Set up plotting grid
tol = 1e-6
nx = 64
_nx = 16
_x1 = jnp.linspace(tol, 1 - tol, nx)
_x2 = jnp.linspace(0, 1, nx)
_x3 = jnp.zeros(1) / 2
_x = jnp.array(jnp.meshgrid(_x1, _x2, _x3))
_x = _x.transpose(1, 2, 3, 0).reshape(nx * nx * 1, 3)
_y = jax.vmap(F)(_x)
_y1 = _y[:, 0].reshape(nx, nx)
_y2 = _y[:, 2].reshape(nx, nx)
__x1 = jnp.linspace(tol, 1 - tol, _nx)
__x2 = jnp.linspace(0, 1, _nx)
__x3 = jnp.zeros(1) / 2
__x = jnp.array(jnp.meshgrid(__x1, __x2, __x3))
__x = __x.transpose(1, 2, 3, 0).reshape(_nx * _nx * 1, 3)
__y = jax.vmap(F)(__x)
__y1 = __y[:, 0].reshape(_nx, _nx)
__y2 = __y[:, 2].reshape(_nx, _nx)


def p_analytic(x):
    r, χ, z = x
    return 1/μ0 * B0**2 * β_t * (1 - r**2) * (1 + ν * r * jnp.cos(2 * π * χ)) * jnp.ones(1)


def Br(x):
    r, χ, z = x
    return - 0.5 * ν * ɛ / q_star * (r**2 - 1) * jnp.sin(2 * π * χ)


def Bχ(x):
    r, χ, z = x
    return ɛ / q_star * (r + ν / 2 * (3 * r**2 - 1) * jnp.cos(2 * π * χ))


def Bz(x):
    r, χ, z = x
    return - (1 - ɛ * r * jnp.cos(2 * π * χ) - β_t * (1 - r**2) * (1 + ν * r * jnp.cos(2 * π * χ)))


def B_analytic(x):
    r, χ, z = x
    return jnp.array([Br(x), Bχ(x), Bz(x)]) * B0


# %%
Λ0, Λ1, Λ2, Λ3 = [DifferentialForm(k, ns, ps, types) for k in range(4)]
Q = QuadratureRule(Λ0, 8)
ξ = get_xi(_X, _Z, Λ0, Q)[0]
E0 = LazyExtractionOperator(Λ0, ξ, True).M
E1 = LazyExtractionOperator(Λ1, ξ, False).M
E2 = LazyExtractionOperator(Λ2, ξ, True).M
E3 = LazyExtractionOperator(Λ3, ξ, False).M

M0, M1, M2, M3 = [
    LazyMassMatrix(Λ, Q, F, E).M for Λ, E in zip([Λ0, Λ1, Λ2, Λ3], [E0, E1, E2, E3])
]
P0, P1, P2, P3 = [
    Projector(Λ, Q, F, E) for Λ, E in zip([Λ0, Λ1, Λ2, Λ3], [E0, E1, E2, E3])
]
D0, D1, D2 = [
    LazyDerivativeMatrix(Λk, Λkplus1, Q, F, E0, E1).M
    for Λk, Λkplus1, E0, E1 in zip(
        [Λ0, Λ1, Λ2], [Λ1, Λ2, Λ3], [E0, E1, E2], [E1, E2, E3]
    )
]  # grad, curl, div
M03 = LazyProjectionMatrix(Λ0, Λ3, Q, F, E0, E3).M
M12 = LazyProjectionMatrix(Λ1, Λ2, Q, F, E1, E2).M

# %%
H_analytic = Flat(B_analytic, F)
p_hat = jnp.linalg.solve(M0, P0(p_analytic))

B_hat = jnp.linalg.solve(M2, P2(H_analytic))
# %%
H_hat = jnp.linalg.solve(M1, M12.T @ B_hat)
H_h = DiscreteFunction(H_hat, Λ1, E1)
F_H_h = Pushforward(H_h, F, 1)
_z1 = jax.vmap(F_H_h)(_x).reshape(nx, nx, 3)
_z1_norm = jnp.linalg.norm(_z1, axis=2)
plt.contourf(_y1, _y2, _z1_norm.reshape(nx, nx))
plt.colorbar()
__z1 = jax.vmap(F_H_h)(__x).reshape(_nx, _nx, 3)
plt.quiver(__y1, __y2, __z1[:, :, 0], __z1[:, :, 3], color="k")
plt.xlabel('X')
plt.ylabel('Z')
# %%
B_h = DiscreteFunction(B_hat, Λ2, E2)
F_B_h = Pushforward(B_h, F, 2)
_z1 = jax.vmap(F_B_h)(_x).reshape(nx, nx, 3)
_z1_norm = jnp.linalg.norm(_z1, axis=2)

plt.contourf(_y1, _y2, _z1_norm.reshape(nx, nx))
__z1 = jax.vmap(F_B_h)(__x).reshape(_nx, _nx, 3)
plt.colorbar()
plt.quiver(__y1, __y2, __z1[:, :, 0], __z1[:, :, 3], color="k")
plt.xlabel('X')
plt.ylabel('Z')
# %%
p_h = DiscreteFunction(p_hat, Λ0, E0)
F_p_h = Pushforward(p_h, F, 0)
_z1 = jax.vmap(F_p_h)(_x).reshape(nx, nx)
plt.contourf(_y1, _y2, _z1)
plt.colorbar()
# %%
J_hat = jnp.linalg.solve(M1, D1.T @ B_hat)
J_h = DiscreteFunction(J_hat, Λ1, E1)
F_J_h = Pushforward(J_h, F, 1)
_z1 = jax.vmap(F_J_h)(_x).reshape(nx, nx, 3)
_z1_norm = jnp.linalg.norm(_z1, axis=2)

plt.contourf(_y1, _y2, _z1_norm.reshape(nx, nx))
__z1 = jax.vmap(F_J_h)(__x).reshape(_nx, _nx, 3)
plt.colorbar()
plt.quiver(__y1, __y2, __z1[:, :, 0], __z1[:, :, 3], color="k")
plt.xlabel('X')
plt.ylabel('Z')
# %%
Pc = EFieldProjector(Λ1, Q, F, E1)
p_hat = jnp.linalg.solve(M0, P0(p_analytic))
grad_p_hat = jnp.linalg.solve(M1, D0 @ p_hat)

H_h = DiscreteFunction(H_hat, Λ1, E1)
J_h = DiscreteFunction(J_hat, Λ1, E1)
gradp_h = DiscreteFunction(grad_p_hat, Λ1, E1)

E_hat = jnp.linalg.solve(M1, Pc(J_h, H_h, H_h, gradp_h))

# %%
E_h = DiscreteFunction(E_hat, Λ1, E1)
F_E_h = Pushforward(E_h, F, 1)
_z1 = jax.vmap(F_E_h)(_x).reshape(nx, nx, 3)
_z1_norm = jnp.linalg.norm(_z1, axis=2)

plt.contourf(_y1, _y2, _z1_norm.reshape(nx, nx))
plt.colorbar()
__z1 = jax.vmap(F_E_h)(__x).reshape(_nx, _nx, 3)
plt.quiver(__y1, __y2, __z1[:, :, 0], __z1[:, :, 3], color="k")
plt.xlabel('X')
plt.ylabel('Z')
# %%
p_hat = jnp.linalg.solve(M0, P0(p_analytic))
grad_p_hat = -jnp.linalg.solve(M1, D0 @ p_hat)
grad_p_h = DiscreteFunction(grad_p_hat, Λ1, E1)
F_grad_p_h = Pushforward(grad_p_h, F, 1)
_z1 = jax.vmap(F_grad_p_h)(_x).reshape(nx, nx, 3)
_z1_norm = jnp.linalg.norm(_z1, axis=2)
plt.contourf(_y1, _y2, _z1_norm.reshape(nx, nx))
plt.colorbar()
__z1 = jax.vmap(F_grad_p_h)(__x).reshape(_nx, _nx, 3)
plt.quiver(__y1, __y2, __z1[:, :, 0], __z1[:, :, 3], color="k")
plt.xlabel('X')
plt.ylabel('Z')
# %%
deltaB_hat = -jnp.linalg.solve(M2, D1 @ E_hat)
deltaB_h = DiscreteFunction(deltaB_hat, Λ2, E2)
F_deltaB_h = Pushforward(deltaB_h, F, 2)
_z1 = jax.vmap(F_deltaB_h)(_x).reshape(nx, nx, 3)
_z1_norm = (jnp.linalg.norm(_z1, axis=2))
plt.contourf(_y1, _y2, _z1_norm.reshape(nx, nx))
plt.colorbar()
__z1 = jax.vmap(F_deltaB_h)(__x).reshape(_nx, _nx, 3)
plt.quiver(__y1, __y2, __z1[:, :, 0], __z1[:, :, 3], color="k")
plt.xlabel('X')
plt.ylabel('Z')
# %%
Pcurl = CurlProjection(Λ1, Q, F, E1)
F_hat = jnp.linalg.solve(M1, Pcurl(J_h, H_h))  # - grad_p_hat
# %%
jnp.sqrt(F_hat @ M1 @ F_hat / (B_hat @ M2 @ B_hat))
# %%
F_h = DiscreteFunction(F_hat, Λ1, E1)
F_F_h = Pushforward(F_h, F, 1)
_z1 = jax.vmap(F_F_h)(_x).reshape(nx, nx, 3)
_z1_norm = jnp.linalg.norm(_z1, axis=2)
plt.contourf(_y1, _y2, (_z1_norm.reshape(nx, nx)))
plt.colorbar()
__z1 = jax.vmap(F_F_h)(__x).reshape(_nx, _nx, 3)
plt.quiver(__y1, __y2, __z1[:, :, 0], __z1[:, :, 3], color="k", scale=50)
plt.xlabel('X')
plt.ylabel('Z')
# %%
# %%


def JcrossH(x):
    return jnp.cross(J_h(x), H_h(x))


u_hat = jnp.linalg.solve(M2, P2(JcrossH))
u_hat @ M2 @ u_hat, E_hat @ M1 @ E_hat
# %%
u_h = DiscreteFunction(u_hat, Λ2, E2)
F_u_h = Pushforward(u_h, F, 2)
_z1 = jax.vmap(F_u_h)(_x).reshape(nx, nx, 3)
_z1_norm = jnp.linalg.norm(_z1, axis=2)

plt.contourf(_y1, _y2, _z1_norm.reshape(nx, nx))
plt.colorbar()
__z1 = jax.vmap(F_u_h)(__x).reshape(_nx, _nx, 3)
plt.quiver(__y1, __y2, __z1[:, :, 0], __z1[:, :, 3], color="k")
plt.xlabel('X')
plt.ylabel('Z')
# %%
