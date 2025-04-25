# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from mrx.DifferentialForms import DifferentialForm, DiscreteFunction, Pullback
from mrx.Quadrature import QuadratureRule
from mrx.Projectors import Projector
from mrx.LazyMatrices import LazyMassMatrix, LazyDerivativeMatrix
from mrx.Utils import l2_product, grad, div, curl

jax.config.update("jax_enable_x64", True)
# %%
n = 8
p = 3

ns = (n, n, 1)
ps = (p, p, 0)


def q(x):
    r, χ, z = x
    v = ((0.5 - r) ** 2 + (χ - 0.5) ** 2) ** 0.5
    return jnp.ones(1) * jnp.sin(jnp.pi * r) * jnp.sin(jnp.pi * χ) * jnp.exp(-(v**2))


def _w(x):
    r, χ, z = x
    v = ((0.5 - r) ** 2 + (χ - 0.5) ** 2) ** 0.5
    return 10 * jnp.array([0, 0, 1]) * jnp.exp(-(v**2))


w = curl(_w)
types = ("clamped", "clamped", "constant")
bcs = ("dirichlet", "dirichlet", "none")
Λ0 = DifferentialForm(0, ns, ps, types)
Λ2 = DifferentialForm(2, ns, ps, types)
Λ3 = DifferentialForm(3, ns, ps, types)
Q = QuadratureRule(Λ0, 10)
D = LazyDerivativeMatrix(Λ2, Λ3, Q).M
M2 = LazyMassMatrix(Λ2, Q).M
K = D @ jnp.linalg.solve(M2, D.T)
P2 = Projector(Λ2, Q)


def u(x):
    return grad(q)(x) + w(x)


u_hat = jnp.linalg.solve(M2, P2(u))
grad_q_hat_proj = jnp.linalg.solve(M2, P2(grad(q)))
w_hat_proj = jnp.linalg.solve(M2, P2(w))

𝚷_Leray = jnp.eye(Λ2.n) - jnp.linalg.solve(M2, D.T @ jnp.linalg.solve(K, D))

w_hat = 𝚷_Leray @ u_hat
grad_q_hat = u_hat - w_hat

# %%
w_h = DiscreteFunction(w_hat, Λ2)
grad_q_h = DiscreteFunction(grad_q_hat, Λ2)
w_h_proj = DiscreteFunction(w_hat_proj, Λ2)
grad_q_h_proj = DiscreteFunction(grad_q_hat_proj, Λ2)
u_h_proj = DiscreteFunction(u_hat, Λ2)
u_h = DiscreteFunction(grad_q_hat + w_hat, Λ2)


def err_u(x):
    return u(x) - u_h(x)


def err_u_proj(x):
    return u(x) - u_h_proj(x)


def err_grad_q(x):
    return grad(q)(x) - grad_q_h(x)


def err_grad_q_proj(x):
    return grad(q)(x) - grad_q_h_proj(x)


def err_w(x):
    return w(x) - w_h(x)


def err_w_proj(x):
    return w(x) - w_h_proj(x)


# %%
print("error in u:", (l2_product(err_u, err_u, Q) / l2_product(u, u, Q)) ** 0.5)
print(
    "error in grad q:",
    (l2_product(err_grad_q, err_grad_q, Q) / l2_product(grad(q), grad(q), Q)) ** 0.5,
)
print("error in w:", (l2_product(err_w, err_w, Q) / l2_product(w, w, Q)) ** 0.5)
print(
    "error in u (projection):",
    (l2_product(err_u, err_u, Q) / l2_product(u, u, Q)) ** 0.5,
)
print(
    "error in grad q (projection):",
    (l2_product(err_grad_q_proj, err_grad_q_proj, Q) / l2_product(grad(q), grad(q), Q))
    ** 0.5,
)
print(
    "error in w (projection):",
    (l2_product(err_w_proj, err_w_proj, Q) / l2_product(w, w, Q)) ** 0.5,
)
# %%
print("divergence of w:", (l2_product(div(w_h), div(w_h), Q)) ** 0.5)
print("curl of grad q:", (l2_product(curl(grad_q_h), curl(grad_q_h), Q)) ** 0.5)


# %%
def F(x):
    return x


ɛ = 1e-5
nx = 64
_x1 = jnp.linspace(ɛ, 1 - ɛ, nx)
_x2 = jnp.linspace(ɛ, 1 - ɛ, nx)
_x3 = jnp.ones(1) / 2
_x = jnp.array(jnp.meshgrid(_x1, _x2, _x3))
_x = _x.transpose(1, 2, 3, 0).reshape(nx * nx * 1, 3)
_y = jax.vmap(F)(_x)
_y1 = _y[:, 0].reshape(nx, nx)
_y2 = _y[:, 1].reshape(nx, nx)
_nx = 16
__x1 = jnp.linspace(ɛ, 1 - ɛ, _nx)
__x2 = jnp.linspace(ɛ, 1 - ɛ, _nx)
__x3 = jnp.ones(1) / 2
__x = jnp.array(jnp.meshgrid(__x1, __x2, __x3))
__x = __x.transpose(1, 2, 3, 0).reshape(_nx * _nx * 1, 3)
__y = jax.vmap(F)(__x)
__y1 = __y[:, 0].reshape(_nx, _nx)
__y2 = __y[:, 1].reshape(_nx, _nx)

# %%
F_u = Pullback(u, F, 2)
F_u_h = Pullback(u_h, F, 2)
_z1 = jax.vmap(F_u_h)(_x).reshape(nx, nx, 3)
_z1_norm = jnp.linalg.norm(_z1, axis=2)
_z2 = jax.vmap(F_u)(_x).reshape(nx, nx, 3)
_z2_norm = jnp.linalg.norm(_z2, axis=2)
plt.contourf(_y1, _y2, _z1_norm.reshape(nx, nx))
plt.colorbar()
plt.contour(_y1, _y2, _z2_norm.reshape(nx, nx), colors="k")
__z1 = jax.vmap(F_u_h)(__x).reshape(_nx, _nx, 3)
plt.quiver(__y1, __y2, __z1[:, :, 0], __z1[:, :, 1], color="w")
# %%
# %%
F_u = Pullback(grad(q), F, 2)
F_u_h = Pullback(grad_q_h, F, 2)
_z1 = jax.vmap(F_u_h)(_x).reshape(nx, nx, 3)
_z1_norm = jnp.linalg.norm(_z1, axis=2)
_z2 = jax.vmap(F_u)(_x).reshape(nx, nx, 3)
_z2_norm = jnp.linalg.norm(_z2, axis=2)
plt.contourf(_y1, _y2, _z1_norm.reshape(nx, nx))
plt.colorbar()
plt.contour(_y1, _y2, _z2_norm.reshape(nx, nx), colors="k")
__z1 = jax.vmap(F_u_h)(__x).reshape(_nx, _nx, 3)
plt.quiver(__y1, __y2, __z1[:, :, 0], __z1[:, :, 1], color="w")
# %%
F_u = Pullback(w, F, 2)
F_u_h = Pullback(w_h, F, 2)
_z1 = jax.vmap(F_u_h)(_x).reshape(nx, nx, 3)
_z1_norm = jnp.linalg.norm(_z1, axis=2)
_z2 = jax.vmap(F_u)(_x).reshape(nx, nx, 3)
_z2_norm = jnp.linalg.norm(_z2, axis=2)
plt.contourf(_y1, _y2, _z1_norm.reshape(nx, nx))
plt.colorbar()
plt.contour(_y1, _y2, _z2_norm.reshape(nx, nx), colors="k")
__z1 = jax.vmap(F_u_h)(__x).reshape(_nx, _nx, 3)
plt.quiver(__y1, __y2, __z1[:, :, 0], __z1[:, :, 1], color="w")
# %%
𝚷_svd = jnp.linalg.svd(𝚷_Leray)
# %%
plt.plot(𝚷_svd.S / 𝚷_svd.S[0])
plt.yscale("log")
plt.xlabel("index")
plt.ylabel("singular value")
plt.vlines(𝚷_svd.S.shape[0] - Λ3.n, ymax=2, ymin=1e-8, color="k", linestyle="--")
plt.show()
# %%
