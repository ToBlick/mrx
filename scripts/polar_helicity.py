# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from mrx.DifferentialForms import DifferentialForm, DiscreteFunction, Pullback
from mrx.Quadrature import QuadratureRule
from mrx.Projectors import Projector
from mrx.LazyMatrices import LazyMassMatrix, LazyDerivativeMatrix, LazyProjectionMatrix, LazyDoubleCurlMatrix
from mrx.Utils import l2_product, curl
from mrx.PolarMapping import LazyExtractionOperator, get_xi

jax.config.update("jax_enable_x64", True)
# %%
p = 3

types = ('clamped', 'periodic', 'constant')
ns = (12, 8, 1)
ps = (p, p, 0)

Λ0, Λ1, Λ2, Λ3 = [DifferentialForm(i, ns, ps, types) for i in range(4)]
Q = QuadratureRule(Λ0, 10)

# %%
###
# Mapping definition
###
a = 1
R0 = 3.0
Y0 = 0.0


def θ(x):
    r, χ, z = x
    return 2 * jnp.atan(jnp.sqrt((1 + a*r/R0)/(1 - a*r/R0)) * jnp.tan(jnp.pi * χ))


def _R(r, χ):
    return jnp.ones(1) * (R0 + a * r * jnp.cos(2 * jnp.pi * χ))


def _Y(r, χ):
    return jnp.ones(1) * (Y0 + a * r * jnp.sin(2 * jnp.pi * χ))


def F(x):
    r, χ, z = x
    return jnp.ravel(jnp.array([_R(r, χ) * jnp.cos(2 * jnp.pi * z),
                                _Y(r, χ),
                                _R(r, χ) * jnp.sin(2 * jnp.pi * z)]))
    # return jnp.ravel(jnp.array([
    #     _R(r, χ),
    #     _Y(r, χ),
    #     jnp.array([z])]))


# %%
ξ, R_hat, Y_hat, Λ, τ = get_xi(_R, _Y, Λ0)
E0, E1, E2, E3 = [LazyExtractionOperator(Λ, ξ, True).M for Λ in [Λ0, Λ1, Λ2, Λ3]]
M0, M1, M2, M3 = [LazyMassMatrix(Λ, Q, F, E).M for Λ, E in zip([Λ0, Λ1, Λ2, Λ3], [E0, E1, E2, E3])]
P0, P1, P2, P3 = [Projector(Λ, Q, F, E) for Λ, E in zip([Λ0, Λ1, Λ2, Λ3], [E0, E1, E2, E3])]

# %%
M12 = LazyProjectionMatrix(Λ1, Λ2, Q, F, E1, E2).M
C = LazyDoubleCurlMatrix(Λ1, Q, F, E1).M
D1 = LazyDerivativeMatrix(Λ1, Λ2, Q, F, E1, E2).M
D0 = LazyDerivativeMatrix(Λ0, Λ1, Q, F, E0, E1).M
# %%


def A(x):
    r, χ, z = x
    a1 = jnp.sin(2 * jnp.pi * χ)
    a2 = 1
    a3 = jnp.cos(2 * jnp.pi * χ)
    return jnp.array([a1, a2, a3]) * jnp.sin(jnp.pi * r)**2


B = curl(A)

l2_product(A, B, Q)

# %%
A_hat = jnp.linalg.solve(M1, P1(A))
B_hat = jnp.linalg.solve(M2, P2(B))

# %%
A_h = DiscreteFunction(A_hat, Λ1, E1)
B_h = DiscreteFunction(B_hat, Λ2, E2)
def A_err(x): return A(x) - A_h(x)
def B_err(x): return B(x) - B_h(x)


# %%
(l2_product(A_err, A_err, Q)/l2_product(A, A, Q))**0.5
# %%
(l2_product(B_err, B_err, Q)/l2_product(B, B, Q))**0.5
# %%
A_hat @ M12 @ B_hat
# %%
# Op = jnp.block([[C, D0],
#                [D0.T, jnp.zeros((D0.shape[1], D0.shape[1]))]])
# U, S, Vh = jnp.linalg.svd(Op)
# plt.plot(S / S[0])
# plt.yscale('log')
# plt.xlabel('index')
# plt.ylabel('singular value')
# # %%
# S_inv = jnp.where(S > 1e-4, 1/S, 0)
# rhs = jnp.concatenate([D1.T @ B_hat, jnp.zeros(D0.shape[1])])
# Aq_hat = U @ jnp.diag(S_inv) @ Vh @ rhs
# A_hat_recon, q_hat = jnp.split(Aq_hat, [D0.shape[0]])
# %%

U, S, Vh = jnp.linalg.svd(C)
plt.plot(S / S[0])
plt.yscale('log')
plt.xlabel('index')
plt.ylabel('singular value')
S_inv = jnp.where(S/S[0] > 1e-12, 1/S, 0)
# %%
A_hat_recon = U @ jnp.diag(S_inv) @ Vh @ D1.T @ B_hat
# %%
A_err = ((A_hat - A_hat_recon) @ M1 @ (A_hat - A_hat_recon) / (A_hat @ M1 @ A_hat))**0.5
print("error in A:", A_err)
# %%
print("error in Helicity:", (A_hat - A_hat_recon) @ M12 @ B_hat / (A_hat @ M12 @ B_hat))
# %%
curl_A_err = (jnp.linalg.solve(M2, D1 @ (A_hat - A_hat_recon)) @ M2 @ jnp.linalg.solve(M2, D1 @ (A_hat - A_hat_recon)) / (jnp.linalg.solve(M2, D1 @ A_hat) @ M2 @ jnp.linalg.solve(M2, D1 @ A_hat)))**0.5
print("error in curl A:", curl_A_err)
# %%
plt.plot(A_hat, label='A')
plt.plot(A_hat_recon, label='A recon')
plt.legend()
# %%
plt.plot(jnp.linalg.solve(M2, D1 @ A_hat), label='curl A')
plt.plot(jnp.linalg.solve(M2, D1 @ A_hat_recon), label='curl A recon')
plt.legend()
# %%
A_h = DiscreteFunction(A_hat, Λ1, E1)
B_h = DiscreteFunction(B_hat, Λ2, E2)
A_h_recon = DiscreteFunction(A_hat_recon, Λ1, E1)
curlA_h_recon = DiscreteFunction(jnp.linalg.solve(M2, D1 @ A_hat_recon), Λ2, E2)

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
F_u = Pullback(A_h_recon, F, 1)
F_u_h = Pullback(A_h, F, 1)
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
