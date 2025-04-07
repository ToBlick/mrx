# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from mrx.DifferentialForms import DifferentialForm, DiscreteFunction, Pullback
from mrx.BoundaryConditions import LazyBoundaryOperator

# %%
ns = (4, 4, 4)
ps = (2, 2, 2)
types = ('clamped', 'clamped', 'clamped')
bcs = ('dirichlet', 'dirichlet', 'dirichlet')

Λ0 = DifferentialForm(0, ns, ps, types)
Λ1 = DifferentialForm(1, ns, ps, types)
Λ2 = DifferentialForm(2, ns, ps, types)
Λ3 = DifferentialForm(3, ns, ps, types)
# %%
B0, B1, B2, B3 = [LazyBoundaryOperator(Λ, bcs) for Λ in [Λ0, Λ1, Λ2, Λ3]]

# %%
plt.imshow(B0.M)
# %%
plt.imshow(B1.M)

# %%


def F(x): return x


ɛ = 1e-5
nx = 64
_x1 = jnp.linspace(ɛ, 1-ɛ, nx)
_x2 = jnp.linspace(ɛ, 1-ɛ, nx)
_x3 = jnp.zeros(1)
_x = jnp.array(jnp.meshgrid(_x1, _x2, _x3))
_x = _x.transpose(1, 2, 3, 0).reshape(nx*nx*1, 3)
_y = jax.vmap(F)(_x)
_y1 = _y[:, 0].reshape(nx, nx)
_y2 = _y[:, 1].reshape(nx, nx)
_nx = 16
__x1 = jnp.linspace(ɛ, 1-ɛ, _nx)
__x2 = jnp.linspace(ɛ, 1-ɛ, _nx)
__x3 = jnp.zeros(1)
__x = jnp.array(jnp.meshgrid(__x1, __x2, __x3))
__x = __x.transpose(1, 2, 3, 0).reshape(_nx*_nx*1, 3)
__y = jax.vmap(F)(__x)
__y1 = __y[:, 0].reshape(_nx, _nx)
__y2 = __y[:, 1].reshape(_nx, _nx)

# %%# %%
u_hat = jnp.ones(B0.n)
u_h = DiscreteFunction(u_hat, Λ0, E=B0.M)

_z1 = jax.vmap(Pullback(u_h, F, 0))(_x).reshape(nx, nx, 1)
_z1_norm = jnp.linalg.norm(_z1, axis=2)
plt.contourf(_y1, _y2, _z1_norm)
plt.colorbar()

# %%
A_hat = jnp.ones(B1.n)
A_h = DiscreteFunction(A_hat, Λ1, E=B1.M)

_z1 = jax.vmap(A_h)(_x).reshape(nx, nx, 3)
_z1_norm = jnp.linalg.norm(_z1, axis=2)
plt.contourf(_y1, _y2, _z1_norm.reshape(nx, nx))
plt.colorbar()
__z1 = jax.vmap(A_h)(__x).reshape(_nx, _nx, 3)
plt.quiver(
    __y1,
    __y2,
    __z1[:, :, 0],
    __z1[:, :, 1],
    color='w')
# %%
B_hat = jnp.ones(B2.n)
B_h = DiscreteFunction(B_hat, Λ2, E=B2.M)

_z1 = jax.vmap(B_h)(_x).reshape(nx, nx, 3)
_z1_norm = jnp.linalg.norm(_z1, axis=2)
plt.contourf(_y1, _y2, _z1_norm.reshape(nx, nx))
plt.colorbar()
__z1 = jax.vmap(B_h)(__x).reshape(_nx, _nx, 3)
plt.quiver(
    __y1,
    __y2,
    __z1[:, :, 0],
    __z1[:, :, 1],
    color='w')
# %%
