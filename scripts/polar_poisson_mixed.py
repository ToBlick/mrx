# %%
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from mrx.PolarMapping import LazyExtractionOperator, get_xi
from mrx.DifferentialForms import DifferentialForm, DiscreteFunction, Pullback
from mrx.Quadrature import QuadratureRule
from mrx.Projectors import Projector
from mrx.LazyMatrices import LazyStiffnessMatrix, LazyMassMatrix, LazyDerivativeMatrix, LazyProjectionMatrix
from mrx.Utils import l2_product
from functools import partial

from mrx.Plotting import converge_plot

jax.config.update("jax_enable_x64", True)
# %%
###
# 2D Poisson problem, Dirichlet BCs
###

###
# Mapping definition
###
@partial(jax.jit, static_argnames=['n', 'p', 'q'])
def get_err(n, p, q):

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
        χ = jnp.arctan2(y, x) # in [-π, π]
        χ = jnp.where(χ < 0, χ + 2 * jnp.pi, χ) / (2 * jnp.pi) # in [0, 1]
        return jnp.array([r, χ, z])

    ns = (n, p, 1)
    ps = (p, p, 0)
    types = ('clamped', 'periodic', 'constant')
    Λ0, Λ2, Λ3 = [DifferentialForm(i, ns, ps, types) for i in [0, 2, 3]]
    Q = QuadratureRule(Λ0, q)
    ξ, R_hat, Y_hat, Λ, τ = get_xi(_R, _Y, Λ0, Q)
    E0, E2, E3 = [LazyExtractionOperator(Λ, ξ, False).M for Λ in [Λ0, Λ2, Λ3]]
    D = LazyDerivativeMatrix(Λ2, Λ3, Q, F, E2, E3).M
    M2 = LazyMassMatrix(Λ2, Q, F, E2).M
    K = D @ jnp.linalg.solve(M2, D.T)

    def u(x):
        r, χ, z = x
        return -jnp.ones(1) * 1/4 * (1/4 * r**4 - 1/3 * r**3 + 1/12)
    def f(x):
        r, χ, z = x
        return jnp.ones(1) * (r - 3/4) * r

    P3 = Projector(Λ3, Q, F, E3)
    u_hat = jnp.linalg.solve(K, P3(f))
    
    M03 = LazyProjectionMatrix(Λ0, Λ3, Q, F, E0, E3).M
    M0 = LazyMassMatrix(Λ0, Q, F, E0).M
    u_hat = jnp.linalg.solve(M0, M03.T @ u_hat)
    u_h = DiscreteFunction(u_hat, Λ0, E0)
    
    def err(x): return (u(x) - u_h(x))
    error = (l2_product(err, err, Q, F) / l2_product(u, u, Q, F))**0.5
    return error

# %%
import time
ns = np.arange(4, 18, 2)
ps = np.arange(1, 4)
qs = np.arange(3,5) # np.arange(4, 11, 1)
err = np.zeros((len(ns), len(ps), len(qs)))
times = np.zeros((len(ns), len(ps), len(qs)))
for i, n in enumerate(ns):
    for j, p in enumerate(ps):
        for k, q in enumerate(qs):
            start = time.time()
            err[i, j, k] = get_err(n, p, q)
            end = time.time()
            times[i, j,k] = end - start
            print(f"n={n}, p={p}, q={q}, err={err[i,j,k]}, time={times[i,j,k]}")

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
