# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time

from mrx.SplineBases import SplineBasis, DerivativeSpline
from mrx.PolarMapping import LazyExtractionOperator, get_xi
from mrx.DifferentialForms import DifferentialForm, DiscreteFunction, Pullback
from mrx.Quadrature import QuadratureRule
from mrx.Projectors import Projector
from mrx.LazyMatrices import LazyMassMatrix
from mrx.Utils import div, curl, grad

jax.config.update("jax_enable_x64", True)
# %%
ns = (8, 8, 1)
ps = (4, 4, 0)
types = ('clamped', 'periodic', 'constant')
_T = jnp.array([0, 0.2, 0.4, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0])
T = jnp.concatenate([
    _T[-(ps[1]+1):-1] - 1,
    _T,
    _T[1:(ps[1]+1)] + 1
])
Tr = jnp.array([0, 0.3, 0.6, 0.8, 0.9, 1.0])
Tr = jnp.concatenate([
    jnp.zeros(ps[0]),
    Tr,  # jnp.linspace(0, 1, n-p+1),
    jnp.ones(ps[0])
])
Ts = (Tr, T, None)

s = SplineBasis(ns[1], ps[1], 'periodic', T)
d = DerivativeSpline(s)
x = jnp.linspace(0, 1, 1000)

# %%
plt.plot(x, jax.vmap(s, (0, None))(x, 0))
# %%
plt.plot(x, jax.vmap(d, (0, None))(x, 0))
# %%
for i in range(ns[1]):
    plt.plot(x, jax.vmap(s, (0, None))(x, i))
for i in range(ns[1]):
    plt.plot(x, jax.vmap(d, (0, None))(x, i))


# %%

Λ0 = DifferentialForm(0, ns, ps, types, Ts)
Λ1 = DifferentialForm(1, ns, ps, types, Ts)
Λ2 = DifferentialForm(2, ns, ps, types, Ts)
Λ3 = DifferentialForm(3, ns, ps, types, Ts)
Q = QuadratureRule(Λ0, 10)

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

# def F(x):
#     r, χ, z = x
#     return jnp.ravel(jnp.array([
#         _R(r, χ) ,
#         _Y(r, χ),
#         jnp.ones(1) * z]))


ξ, R_hat, Y_hat, Λ, τ = get_xi(_R, _Y, Λ0)
# %%
E0, E1, E2, E3 = [LazyExtractionOperator(Λ, ξ, True).M for Λ in [Λ0, Λ1, Λ2, Λ3]]
M0, M1, M2, M3 = [LazyMassMatrix(Λ, Q, F, E).M for Λ, E in zip([Λ0, Λ1, Λ2, Λ3], [E0, E1, E2, E3])]
P0, P1, P2, P3 = [Projector(Λ, Q, F, E) for Λ, E in zip([Λ0, Λ1, Λ2, Λ3], [E0, E1, E2, E3])]

# E0 = LazyExtractionOperator(Λ0, ξ, True).M
# M0 = LazyMassMatrix(Λ0, Q, F, E0).M
# P0 = Projector(Λ0, Q, F, E0)
# %%


def l2_product(f, g, Q):
    return jnp.einsum("ij,ij,i->", jax.vmap(f)(Q.x), jax.vmap(g)(Q.x), Q.w)


# %%
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


def f(x):
    r, χ, z = x
    return jnp.ones(1) * r**2 * jnp.sin(4 * jnp.pi * χ) * (1 - r)**2


# %%
f_hat = jnp.zeros(E0.shape[0]).at[35:43:2].set(1)
f_h = DiscreteFunction(f_hat, Λ0, E0)
_z1 = jax.vmap(Pullback(f_h, F, 0))(_x).reshape(nx, nx, 1)
_z1_norm = jnp.linalg.norm(_z1, axis=2)
plt.contourf(_y1, _y2, _z1_norm)
plt.colorbar()

# %%
f_hat = jnp.linalg.solve(M0, P0(f))
f_h = DiscreteFunction(f_hat, Λ0, E0)
def err(x): return f(x) - f_h(x)


(l2_product(err, err, Q) / l2_product(f, f, Q))**0.5

# %%
# Λi = jax.vmap(jax.vmap(Λ0, (0, None)), (None, 0))(_x, jnp.arange(Λ0.n))
# Λi_polar = np.einsum('ij,jkm->ikm', E0, Λi)
# _z1_norm = jnp.linalg.norm(Λi_polar, axis=2)
# idx = 0
# _z1 = Λi_polar[idx].reshape(nx, nx, 1)
# plt.contourf(_y1, _y2, _z1_norm[idx].reshape(nx, nx))
# %%
# f_hat = jnp.zeros(E0.shape[0]).at[0:3].set(1)
# f_h = DiscreteFunction(f_hat, Λ0, E0)
_z1 = jax.vmap(Pullback(f_h, F, 0))(_x).reshape(nx, nx, 1)
_z1_norm = jnp.linalg.norm(_z1, axis=2)
_z2 = jax.vmap(Pullback(f, F, 0))(_x).reshape(nx, nx, 1)
_z2_norm = jnp.linalg.norm(_z2, axis=2)
plt.contourf(_y1, _y2, _z1_norm)
plt.colorbar()
plt.contour(_y1, _y2, _z2_norm, colors='k')

# %%
_z1 = jax.vmap(f_h)(_x).reshape(nx, nx, 1)
_z1_norm = jnp.linalg.norm(_z1, axis=2)
_z2 = jax.vmap(f)(_x).reshape(nx, nx, 1)
_z2_norm = jnp.linalg.norm(_z2, axis=2)
plt.contourf(_x1, _x2, _z1_norm)
plt.colorbar()
plt.contour(_x1, _x2, _z2_norm, colors='k')

# %%
A_hat = jnp.ones(E1.shape[0])
B_h = DiscreteFunction(A_hat, Λ1, E1)
F_B_h = Pullback(B_h, F, 1)
_z1 = jax.vmap(F_B_h)(_x).reshape(nx, nx, 3)
_z1_norm = jnp.linalg.norm(_z1, axis=2)
plt.contourf(_y1, _y2, _z1_norm.reshape(nx, nx))
plt.colorbar()
__z1 = jax.vmap(F_B_h)(__x).reshape(_nx, _nx, 3)
plt.quiver(
    __y1,
    __y2,
    __z1[:, :, 0],
    __z1[:, :, 1],
    color='w')

# %%
# def A(x):
#     r, χ, z = x
#     a1 = r * jnp.sin(2 * jnp.pi * χ) * jnp.pi
#     a2 = r**2 * 10
#     a3 = 0
#     return jnp.array([a1, a2, a3])

A = grad(f)
F_A = Pullback(A, F, 1)
A_hat = jnp.linalg.solve(M1, P1(A))
A_h = DiscreteFunction(A_hat, Λ1, E1)
def err(x): return A(x) - A_h(x)


(l2_product(err, err, Q) / l2_product(A, A, Q))**0.5

# %%
F_A_h = Pullback(A_h, F, 1)
# A_h = DiscreteFunction(jnp.zeros_like(A_hat).at[16].set(1), Λ1, E1)
_z1 = jax.vmap(F_A_h)(_x).reshape(nx, nx, 3)
_z1_norm = jnp.linalg.norm(_z1, axis=2)
_z2 = jax.vmap(F_A)(_x).reshape(nx, nx, 3)
_z2_norm = jnp.linalg.norm(_z2, axis=2)
plt.contourf(_y1, _y2, _z1_norm.reshape(nx, nx))
plt.colorbar()
plt.contour(_y1, _y2, _z2_norm.reshape(nx, nx), colors='k')
__z1 = jax.vmap(F_A_h)(__x).reshape(_nx, _nx, 3)
plt.quiver(
    __y1,
    __y2,
    __z1[:, :, 0],
    __z1[:, :, 1],
    color='w')
# %%

# %%
## Exact gradient
grad_fh = jax.grad(lambda x: (f_h)(x).sum())
grad_f_hat = jnp.linalg.solve(M1, P1(grad_fh))
gradf_h = DiscreteFunction(grad_f_hat, Λ1, E1)
def err(x): return grad_fh(x) - gradf_h(x)


(l2_product(err, err, Q) / l2_product(grad_fh, grad_fh, Q))**0.5

# %%
B_hat = jnp.ones(E2.shape[0])
B_h = DiscreteFunction(B_hat, Λ2, E2)
F_B_h = Pullback(B_h, F, 2)
_z1 = jax.vmap(F_B_h)(_x).reshape(nx, nx, 3)
_z1_norm = jnp.linalg.norm(_z1, axis=2)
plt.contourf(_y1, _y2, _z1_norm.reshape(nx, nx))
plt.colorbar()
__z1 = jax.vmap(F_B_h)(__x).reshape(_nx, _nx, 3)
plt.quiver(
    __y1,
    __y2,
    __z1[:, :, 0],
    __z1[:, :, 1],
    color='w')

# %%


def B(x):
    v = A(x)
    return jnp.array([v[1], v[0], v[2]])


B_hat = jnp.linalg.solve(M2, P2(B))
B_h = DiscreteFunction(B_hat, Λ2, E2)
def err(x): return B(x) - B_h(x)


(l2_product(err, err, Q) / l2_product(B, B, Q))**0.5
# %%
F_B = Pullback(B, F, 2)
F_B_h = Pullback(B_h, F, 2)
# A_h = DiscreteFunction(jnp.zeros_like(A_hat).at[16].set(1), Λ1, E1)
_z1 = jax.vmap(F_B_h)(_x).reshape(nx, nx, 3)
_z1_norm = jnp.linalg.norm(_z1, axis=2)
_z2 = jax.vmap(F_B)(_x).reshape(nx, nx, 3)
_z2_norm = jnp.linalg.norm(_z2, axis=2)
plt.contourf(_y1, _y2, _z1_norm.reshape(nx, nx))
plt.colorbar()
plt.contour(_y1, _y2, _z2_norm.reshape(nx, nx), colors='k')
__z1 = jax.vmap(F_B_h)(__x).reshape(_nx, _nx, 3)
plt.quiver(
    __y1,
    __y2,
    __z1[:, :, 0],
    __z1[:, :, 1],
    color='w')
# %%
## exact curl

curl_Ah = curl(A_h)
curl_A_hat = jnp.linalg.solve(M2, P2(curl_Ah))
curlA_h = DiscreteFunction(curl_A_hat, Λ2, E2)
def err(x): return curl_Ah(x) - curlA_h(x)


(l2_product(err, err, Q) / l2_product(curl_Ah, curl_Ah, Q))**0.5

# %%
g = div(B)
g = f
g_hat = jnp.linalg.solve(M3, P3(g))
g_h = DiscreteFunction(g_hat, Λ3, E3)
def err(x): return g(x) - g_h(x)


(l2_product(err, err, Q) / l2_product(g, g, Q))**0.5

# %%
F_g = Pullback(g, F, 3)
F_g_h = Pullback(g_h, F, 3)
# A_h = DiscreteFunction(jnp.zeros_like(A_hat).at[16].set(1), Λ1, E1)
_z1 = jax.vmap(F_g_h)(_x).reshape(nx, nx, 1)
_z1_norm = jnp.linalg.norm(_z1, axis=2)
_z2 = jax.vmap(F_g)(_x).reshape(nx, nx, 1)
_z2_norm = jnp.linalg.norm(_z2, axis=2)
plt.contourf(_y1, _y2, _z1_norm.reshape(nx, nx))
plt.colorbar()
plt.contour(_y1, _y2, _z2_norm.reshape(nx, nx), colors='k')

# %%
# exact divergence
div_Bh = div(B_h)
div_B_hat = jnp.linalg.solve(M3, P3(div_Bh))
divB_h = DiscreteFunction(div_B_hat, Λ3, E3)
def err(x): return div_Bh(x) - divB_h(x)


(l2_product(err, err, Q) / l2_product(div_Bh, div_Bh, Q))**0.5

# %%
# D0 = jnp.linalg.solve(M1, LazyDerivativeMatrix(Λ0, Λ1, Q, F, E0, E1).M)
# D1 = jnp.linalg.solve(M2, LazyDerivativeMatrix(Λ1, Λ2, Q, F, E1, E2).M)
# D2 = jnp.linalg.solve(M3, LazyDerivativeMatrix(Λ2, Λ3, Q, F, E2, E3).M)

# %%
# M03 = LazyProjectionMatrix(Λ0, Λ3, Q, F, E0, E3).M.T
# M12 = LazyProjectionMatrix(Λ1, Λ2, Q, F, E1, E2).M.T

# # %%
# 𝚷30 = jnp.linalg.solve(M3, M03.T)
# 𝚷03 = jnp.linalg.solve(M0, M03)
# 𝚷21 = jnp.linalg.solve(M2, M12.T)
# 𝚷12 = jnp.linalg.solve(M1, M12)

# # %%
# 𝚷f_hat = 𝚷03 @ 𝚷30 @ f_hat
# ((𝚷f_hat - f_hat) @ M0 @ (𝚷f_hat - f_hat) / (f_hat @ M0 @ f_hat))**0.5
# # %%
# 𝚷A_hat = 𝚷12 @ 𝚷21 @ A_hat
# ((𝚷A_hat - A_hat) @ M1 @ (𝚷A_hat - A_hat) / (A_hat @ M1 @ A_hat))**0.5
# # %%
# 𝚷B_hat = 𝚷21 @ 𝚷12 @ B_hat
# ((𝚷B_hat - B_hat) @ M2 @ (𝚷B_hat - B_hat) / (B_hat @ M2 @ B_hat))**0.5
# # %%
# 𝚷g_hat = 𝚷30 @ 𝚷03 @ g_hat
# ((𝚷g_hat - g_hat) @ M3 @ (𝚷g_hat - g_hat) / (g_hat @ M3 @ g_hat))**0.5

# # %%
# jnp.max(D1 @ jnp.linalg.solve(M1, D0)), \
# jnp.max(D2 @ jnp.linalg.solve(M2, D1))


# %%
plt.scatter(R_hat, Y_hat, s=5)
plt.scatter([τ + R0, R0 - τ/2, R0 - τ/2], [0, Y0 + jnp.sqrt(3) * τ/2, Y0 - jnp.sqrt(3) * τ/2], s=10, c='k')
plt.plot([τ + R0, R0 - τ/2, R0 - τ/2, τ + R0], [0, Y0 + jnp.sqrt(3) * τ/2, Y0 - jnp.sqrt(3) * τ/2, 0], 'k:')
# %%

# %%


@jax.jit
def f():
    return [jnp.all(jax.vmap(lambda i: jax.jit(Λ._ravel_index)(*jax.jit(Λ._unravel_index)(i)))(jnp.arange(Λ.n)) == jnp.arange(Λ.n)) for Λ in [Λ0, Λ1, Λ2, Λ3]]


f()
# %%


@jax.jit
def test():
    ns = (4, 8, 1)
    ps = (2, 3, 0)
    types = ('clamped', 'periodic', 'constant')
    Λ0 = DifferentialForm(0, ns, ps, types)
    return Λ0[0](jnp.array([0.5, 0.5, 0.5]))


test()


def f(x):
    return jnp.ones(1) * jnp.sqrt(2 * x[0]**3 * jnp.sin(jnp.pi * x[1]) * jnp.pi)


print(jnp.einsum("ij,ij,i->", jax.vmap(f)(Q.x), jax.vmap(f)(Q.x), Q.w))

# %%


def l2_product(f, g, Q):
    return jnp.einsum("ij,ij,i->", jax.vmap(f)(Q.x), jax.vmap(g)(Q.x), Q.w)


@jax.jit
def get_err():
    ns = (4, 8, 1)
    ps = (2, 3, 0)
    types = ('clamped', 'periodic', 'constant')

    Λ0 = DifferentialForm(0, ns, ps, types)
    Q = QuadratureRule(Λ0, 6)

    def f(x):
        return jnp.ones(1) * jnp.sqrt(2 * x[0]**3 * jnp.sin(jnp.pi * x[1]) * jnp.pi)

    P0 = Projector(Λ0, Q, F)
    M0 = LazyMassMatrix(Λ0, Q, F).M

    f_hat = jnp.linalg.solve(M0, P0(f))
    f_h = DiscreteFunction(f_hat, Λ0)
    def err(x): return f(x) - f_h(x)
    return l2_product(err, err, Q)


# %%
start = time.time()
get_err()
print(time.time() - start)
start = time.time()
get_err()
print(time.time() - start)
for _ in range(100):
    get_err()
print((time.time() - start)/100)
