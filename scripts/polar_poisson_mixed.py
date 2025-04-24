# %%
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from mrx.PolarMapping import LazyExtractionOperator, get_xi
from mrx.DifferentialForms import DifferentialForm, DiscreteFunction, Pullback
from mrx.Quadrature import QuadratureRule
from mrx.Projectors import Projector
from mrx.LazyMatrices import LazyMassMatrix, LazyDerivativeMatrix, LazyProjectionMatrix
from mrx.Utils import l2_product

from mrx.Plotting import converge_plot

jax.config.update("jax_enable_x64", True)
# %%
###
# 2D Poisson problem, Dirichlet BCs
###

###
# Mapping definition
###
n = 8
p = 3
q = 3

a = 1
R0 = 0.0
Y0 = 0.0


def _R(r, χ):
    return jnp.ones(1) * (R0 + a * r * jnp.cos(2 * jnp.pi * χ))


def _Y(r, χ):
    return jnp.ones(1) * (Y0 + a * r * jnp.sin(2 * jnp.pi * χ))


def F(x):
    r, χ, z = x
    return jnp.ravel(jnp.array([_R(r, χ),
                                _Y(r, χ),
                                jnp.ones(1) * z]))


ns = (n, 4, 1)
ps = (p, p, 0)


def u(x):
    r, χ, z = x
    return -jnp.ones(1) * (1/16 * r**4 - 1/12 * r**3)


def f(x):
    r, χ, z = x
    return jnp.ones(1) * (r**2 - 3/4 * r)


types = ('clamped', 'periodic', 'constant')
Λ0 = DifferentialForm(0, ns, ps, types)
Λ2 = DifferentialForm(2, ns, ps, types)
Λ3 = DifferentialForm(3, ns, ps, types)
ξ, R_hat, Y_hat, Λ, τ = get_xi(_R, _Y, Λ0, q)
Q = QuadratureRule(Λ0, q)
E0 = LazyExtractionOperator(Λ0, ξ, False).M
E2 = LazyExtractionOperator(Λ2, ξ, False).M
E3 = LazyExtractionOperator(Λ3, ξ, False).M
M03 = LazyProjectionMatrix(Λ0, Λ3, Q, F, E0, E3).M
M0 = LazyMassMatrix(Λ0, Q, F, E0).M

D = LazyDerivativeMatrix(Λ2, Λ3, Q, F, E2, E3).M
M2 = LazyMassMatrix(Λ2, Q, F, E2).M
M3 = LazyMassMatrix(Λ3, Q, F, E3).M
K = D @ jnp.linalg.solve(M2, D.T)

# %%
P3 = Projector(Λ3, Q, F, E3)
P0 = Projector(Λ0, Q, F, E0)
f_hat = jnp.linalg.solve(M3, P3(f))
f_hat_0 = jnp.linalg.solve(M0, P0(f))
u_hat = jnp.linalg.solve(K, P3(f))
u_h = DiscreteFunction(u_hat, Λ3, E3)
f_h = DiscreteFunction(f_hat, Λ3, E3)
def err(x): return (u(x) - u_h(x))


error = (l2_product(err, err, Q, F) / l2_product(u, u, Q, F))**0.5

# %%
ɛ = 1e-5
nx = 64
_x1 = jnp.linspace(ɛ, 1-ɛ, nx)
_x2 = jnp.zeros(1)
_x3 = jnp.zeros(1)
_x = jnp.array(jnp.meshgrid(_x1, _x2, _x3))
_x = _x.transpose(1, 2, 3, 0).reshape(nx*1*1, 3)
_y = jax.vmap(F)(_x)

F_f_h = Pullback(f_h, F, 3)
F_f = Pullback(f, F, 3)
plt.plot(_y[:, 0], jax.vmap(F_f_h)(_x), label='f_h (as 3form)')
plt.plot(_y[:, 0], jax.vmap(F_f)(_x), label='f (as 3form)')
plt.xlabel('r')
plt.legend()
plt.show()

# %%
F_f_h = Pullback(DiscreteFunction(f_hat_0, Λ0, E0), F, 0)
F_f = Pullback(f, F, 0)
plt.plot(_y[:, 0], jax.vmap(F_f_h)(_x), label='f_h (as 0form)')
plt.plot(_y[:, 0], jax.vmap(F_f)(_x), label='f (as 0form)')
plt.xlabel('r')
plt.legend()
plt.show()


# %%
F_u_h = Pullback(u_h, F, 3)
F_u = Pullback(u, F, 3)

plt.plot(_y[:, 0], jax.vmap(F_u_h)(_x), label='u_h')
plt.plot(_y[:, 0], jax.vmap(F_u)(_x), label='u')
plt.xlabel('r')
plt.legend()

# %%
import time
ns = np.arange(4, 18, 2)
ps = np.arange(1, 4)
qs = np.arange(4, 5)  # np.arange(4, 11, 1)
err = np.zeros((len(ns), len(ps), len(qs)))
times = np.zeros((len(ns), len(ps), len(qs)))
for i, n in enumerate(ns):
    for j, p in enumerate(ps):
        for k, q in enumerate(qs):
            start = time.time()
            err[i, j, k] = get_err(n, p, q)
            end = time.time()
            times[i, j, k] = end - start
            print(f"n={n}, p={p}, q={q}, err={err[i, j, k]}, time={times[i, j, k]}")

# %%
fig = converge_plot(err, ns, ps, qs)
fig.update_layout(
    xaxis_type="log",
    yaxis_type="log",
    yaxis_tickformat=".1e",
    xaxis_title='n',
    yaxis_title='Error',
    # legend_title='Legend'
)
fig.show()
# %%
fig = converge_plot(times, ns, ps, qs)
fig.update_layout(
    xaxis_type="log",
    yaxis_type="log",
    yaxis_tickformat=".1e",
    xaxis_title='n',
    yaxis_title='Time',
    # legend_title='Legend'
)
fig.show()

# %%
