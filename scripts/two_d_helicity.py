# %%
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from mrx.DifferentialForms import DifferentialForm, DiscreteFunction, Pullback, Pushforward
from mrx.Quadrature import QuadratureRule
from mrx.Projectors import Projector
from mrx.LazyMatrices import LazyMassMatrix, LazyDerivativeMatrix, LazyProjectionMatrix, LazyDoubleCurlMatrix
from mrx.Utils import l2_product, grad, div, curl
jax.config.update("jax_enable_x64", True)
# %%
n = 10
p = 3

types = ('periodic', 'periodic', 'constant')
bcs = ('none', 'none', 'none')

ns = (n, n, 1)
ps = (p, p, 0)

Λ1 = DifferentialForm(1, ns, ps, types)
Λ2 = DifferentialForm(2, ns, ps, types)
Q = QuadratureRule(Λ1, 10)

M2 = LazyMassMatrix(Λ2, Q).M
M1 = LazyMassMatrix(Λ1, Q).M
C = LazyDoubleCurlMatrix(Λ1, Q).M
D = LazyDerivativeMatrix(Λ1, Λ2, Q).M
P2 = Projector(Λ2, Q)
P1 = Projector(Λ1, Q)

M12 = LazyProjectionMatrix(Λ1, Λ2, Q).M

# %%
m1 = 2
m2 = 2


def A(x):
    r, χ, z = x
    a1 = jnp.sin(m1 * jnp.pi * r) * jnp.cos(m2 * jnp.pi * χ) * jnp.sqrt(m2**2/(m2**2 + m1**2))
    a2 = -jnp.cos(m1 * jnp.pi * r) * jnp.sin(m2 * jnp.pi * χ) * jnp.sqrt(m1**2/(m2**2 + m1**2))
    a3 = jnp.sin(m1 * jnp.pi * r) * jnp.sin(m2 * jnp.pi * χ)
    return jnp.array([a1, a2, a3])


B = curl(A)


# %%
l2_product(A, B, Q)

# %%
A_hat = jnp.linalg.solve(M1, P1(A))
B_hat = jnp.linalg.solve(M2, P2(B))
# %%
A_hat @ M12 @ B_hat
# %%
U, S, Vh = jnp.linalg.svd(C)
plt.plot(S / S[0])
plt.yscale('log')
plt.xlabel('index')
plt.ylabel('singular value')
S_inv = jnp.where(S/S[0] > 1e-12, 1/S, 0)

# %%
A_hat_recon = U @ jnp.diag(S_inv) @ Vh @ D.T @ B_hat

# %%
A_err = ((A_hat - A_hat_recon) @ M1 @ (A_hat - A_hat_recon) / (A_hat @ M1 @ A_hat))**0.5
print("error in A:", A_err)

# %%
print("error in Helicity:", (A_hat - A_hat_recon) @ M12 @ B_hat / (A_hat @ M12 @ B_hat))

# %%
curl_A_err = (jnp.linalg.solve(M2, D @ (A_hat - A_hat_recon)) @ M2 @ jnp.linalg.solve(M2, D @ (A_hat - A_hat_recon)) / (jnp.linalg.solve(M2, D @ A_hat) @ M2 @ jnp.linalg.solve(M2, D @ A_hat)))**0.5
print("error in curl A:", curl_A_err)
# %%
plt.plot(A_hat, label='A')
plt.plot(A_hat_recon, label='A recon')
plt.legend()

# %%
A_h = DiscreteFunction(A_hat, Λ1)
B_h = DiscreteFunction(B_hat, Λ2)
A_h_recon = DiscreteFunction(A_hat_recon, Λ1)
curlA_h_recon = DiscreteFunction(jnp.linalg.solve(M2, D @ A_hat_recon), Λ2)

# %%


def F(x): return x


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

# %%
F_u = Pullback(curlA_h_recon, F, 2)
F_u_h = Pullback(B_h, F, 2)
_z1 = jax.vmap(F_u_h)(_x).reshape(nx, nx, 3)
_z1_norm = jnp.linalg.norm(_z1, axis=2)
_z2 = jax.vmap(F_u)(_x).reshape(nx, nx, 3)
_z2_norm = jnp.linalg.norm(_z2, axis=2)
plt.contourf(_y1, _y2, _z1_norm.reshape(nx, nx))
plt.colorbar()
plt.contour(_y1, _y2, _z2_norm.reshape(nx, nx), colors='k')
__z1 = jax.vmap(F_u_h)(__x).reshape(_nx, _nx, 3)
plt.quiver(
    __y1,
    __y2,
    __z1[:, :, 0],
    __z1[:, :, 1],
    color='w')
# %%
# %%
F_u = Pullback(A_h_recon, F, 2)
F_u_h = Pullback(A_h, F, 2)
_z1 = jax.vmap(F_u_h)(_x).reshape(nx, nx, 3)
_z1_norm = jnp.linalg.norm(_z1, axis=2)
_z2 = jax.vmap(F_u)(_x).reshape(nx, nx, 3)
_z2_norm = jnp.linalg.norm(_z2, axis=2)
plt.contourf(_y1, _y2, _z1_norm.reshape(nx, nx))
plt.colorbar()
plt.contour(_y1, _y2, _z2_norm.reshape(nx, nx), colors='k')
__z1 = jax.vmap(F_u_h)(__x).reshape(_nx, _nx, 3)
plt.quiver(
    __y1,
    __y2,
    __z1[:, :, 0],
    __z1[:, :, 1],
    color='w')

# %%
