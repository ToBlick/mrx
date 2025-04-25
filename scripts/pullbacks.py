# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from mrx.DifferentialForms import DifferentialForm, DiscreteFunction, Pullback
from mrx.Quadrature import QuadratureRule
from mrx.Projectors import Projector
from mrx.PolarMapping import get_xi, LazyExtractionOperator
from mrx.LazyMatrices import (
    LazyMassMatrix,
    LazyProjectionMatrix,
)

# %%
n = 8
p = 3


def _R(r, χ):
    return jnp.ones(1) * r * jnp.cos(2 * jnp.pi * χ)


def _Y(r, χ):
    return jnp.ones(1) * r * jnp.sin(2 * jnp.pi * χ)


def F(p):
    r, χ, z = p
    return jnp.squeeze(jnp.array([_R(r, χ), _Y(r, χ), jnp.ones(1) * z]))


def F_inv(p):
    x, y, z = p
    r = jnp.sqrt(x**2 + y**2)
    χ = jnp.arctan2(y, x)  # in [-π, π]
    χ = jnp.where(χ < 0, χ + 2 * jnp.pi, χ) / (2 * jnp.pi)  # in [0, 1]
    return jnp.array([r, χ, z])


t = jnp.array([0.8, 0.25, 0])
F(F_inv(t)), F_inv(F(t))
# %%
ns = (n, n, 1)
ps = (p, p, 0)

types = ("clamped", "periodic", "constant")
Λ0, Λ1, Λ2, Λ3 = [
    DifferentialForm(i, ns, ps, types) for i in range(4)
]  # H1, H(curl), H(div), L2
Q = QuadratureRule(Λ0, 4)

ξ, R_hat, Y_hat, Λ, τ = get_xi(_R, _Y, Λ0, Q)
E0, E1, E2, E3 = [LazyExtractionOperator(Λ, ξ, False).M for Λ in [Λ0, Λ1, Λ2, Λ3]]
M0, M1, M2, M3 = [
    LazyMassMatrix(Λ, Q, F, E).M for Λ, E in zip([Λ0, Λ1, Λ2, Λ3], [E0, E1, E2, E3])
]
P0, P1, P2, P3 = [
    Projector(Λ, Q, F, E) for Λ, E in zip([Λ0, Λ1, Λ2, Λ3], [E0, E1, E2, E3])
]
M03 = LazyProjectionMatrix(Λ0, Λ3, Q, F, E0, E3).M
M12 = LazyProjectionMatrix(Λ1, Λ2, Q, F, E1, E2).M
# %%
ɛ = 1e-6
nx = 64
_x1 = jnp.linspace(ɛ, 1 - ɛ, nx)
_x2 = jnp.linspace(0, 1, nx)
_x3 = jnp.zeros(1) / 2
_x = jnp.array(jnp.meshgrid(_x1, _x2, _x3))
_x = _x.transpose(1, 2, 3, 0).reshape(nx * nx * 1, 3)
_y = jax.vmap(F)(_x)
_y1 = _y[:, 0].reshape(nx, nx)
_y2 = _y[:, 1].reshape(nx, nx)
_nx = 16
__x1 = jnp.linspace(ɛ, 1, _nx)
__x2 = jnp.linspace(0, 1, _nx)
__x3 = jnp.zeros(1) / 2
__x = jnp.array(jnp.meshgrid(__x1, __x2, __x3))
__x = __x.transpose(1, 2, 3, 0).reshape(_nx * _nx * 1, 3)
__y = jax.vmap(F)(__x)
__y1 = __y[:, 0].reshape(_nx, _nx)
__y2 = __y[:, 1].reshape(_nx, _nx)


# %%
def f(x):
    x, y, z = x
    return jnp.ones(1) * jnp.cos(jnp.pi * x)


# def f_logical(x):
#     r, χ, z = x
#     return jnp.ones(1) * r
# f = Pullback(f_logical, F_inv, 0)

f0_logical = Pullback(f, F, 0)

f_hat = jnp.linalg.solve(M0, P0(f0_logical))
f_h = DiscreteFunction(f_hat, Λ0, E0)
plt.contourf(_x1, _x2, jax.vmap(f_h)(_x).reshape(nx, nx))
plt.colorbar()

# %%
F_f_h = Pullback(f_h, F_inv, 0)
plt.contourf(_y1, _y2, jax.vmap(F_f_h)(_y).reshape(nx, nx))
plt.colorbar()
# %%
f3_logical = Pullback(f, F, 3)

f_hat = jnp.linalg.solve(M3, P3(f0_logical))
f_h = DiscreteFunction(f_hat, Λ3, E3)
plt.contourf(_x1, _x2, jax.vmap(f_h)(_x).reshape(nx, nx))
plt.colorbar()

# %%
F_f_h = Pullback(f_h, F_inv, 3)
plt.contourf(_y1, _y2, jax.vmap(F_f_h)(_y).reshape(nx, nx))
plt.colorbar()

# %%
f_hat = jnp.linalg.solve(M3, P3(f0_logical))
f_hat = jnp.linalg.solve(M0, M03.T @ f_hat)
f_h = DiscreteFunction(f_hat, Λ0, E0)
plt.contourf(_x1, _x2, jax.vmap(f_h)(_x).reshape(nx, nx))
plt.colorbar()

# %%
F_f_h = Pullback(f_h, F_inv, 0)
plt.contourf(_y1, _y2, jax.vmap(F_f_h)(_y).reshape(nx, nx))
plt.colorbar()
# %%
# def A(r):
#     x, y, z = r
#     return jnp.array([x, 0, 0])


def A_logical(x):
    r, χ, z = x
    return jnp.array([r, 0, 0])


# A = Pullback(A_logical, F_inv, 1)

# A1_logical = Pullback(A, F, 1)
# A2_logical = Pullback(A, F, 2)

A1_logical = A_logical

# %%
A_hat = jnp.linalg.solve(M1, P1(A1_logical))
# A_hat = jnp.zeros_like(A_hat).at[8].set(1)
A_h = DiscreteFunction(A_hat, Λ1, E1)
_z1 = jax.vmap(A_h)(_x).reshape(nx, nx, 3)
_z1_norm = jnp.linalg.norm(_z1, axis=2)
plt.contourf(_x1, _x2, _z1_norm.reshape(nx, nx))
plt.colorbar()
__z1 = jax.vmap(A_h)(__x).reshape(_nx, _nx, 3)
plt.quiver(__x1, __x2, __z1[:, :, 0], __z1[:, :, 1], color="w")
# %%
F_A_h = Pullback(A_h, F_inv, 1)
_z1 = jax.vmap(F_A_h)(_y).reshape(nx, nx, 3)
_z1_norm = jnp.linalg.norm(_z1, axis=2)
plt.contourf(_y1, _y2, _z1_norm.reshape(nx, nx))
plt.colorbar()
__z1 = jax.vmap(F_A_h)(__y).reshape(_nx, _nx, 3)
plt.quiver(__y1, __y2, __z1[:, :, 0], __z1[:, :, 1], color="w")
# %%
A_hat = jnp.linalg.solve(M2, P2(A1_logical))
A_h = DiscreteFunction(A_hat, Λ2, E2)
# %%
F_A_h = Pullback(A_h, F, 2)
_z1 = jax.vmap(A_h)(_x).reshape(nx, nx, 3)
_z1_norm = jnp.linalg.norm(_z1, axis=2)
plt.contourf(_x1, _x2, _z1_norm.reshape(nx, nx))
plt.colorbar()
__z1 = jax.vmap(A_h)(__x).reshape(_nx, _nx, 3)
plt.quiver(__x1, __x2, __z1[:, :, 0], __z1[:, :, 1], color="w")
# %%
F_A_h = Pullback(A_h, F_inv, 2)
_z1 = jax.vmap(F_A_h)(_y).reshape(nx, nx, 3)
_z1_norm = jnp.linalg.norm(_z1, axis=2)
plt.contourf(_y1, _y2, _z1_norm.reshape(nx, nx))
plt.colorbar()
__z1 = jax.vmap(F_A_h)(__y).reshape(_nx, _nx, 3)
plt.quiver(__y1, __y2, __z1[:, :, 0], __z1[:, :, 1], color="w")

# %%
