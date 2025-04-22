# %%
import jax
import jax.numpy as jnp
import numpy as np

from mrx.PolarMapping import LazyExtractionOperator, get_xi
from mrx.DifferentialForms import DifferentialForm, DiscreteFunction
from mrx.Quadrature import QuadratureRule
from mrx.Projectors import Projector
from mrx.LazyMatrices import LazyStiffnessMatrix
from mrx.Utils import l2_product
from functools import partial

from mrx.Plotting import converge_plot

jax.config.update("jax_enable_x64", True)
# %%
###
# 2D Poisson problem, Dirichlet BCs
###


@partial(jax.jit, static_argnames=['n', 'p', 'q'])
def get_err(n, p, q):

    ###
    # Mapping definition
    ###
    a = 1
    R0 = 3.0
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
    ns = (n, n, 1)
    ps = (p, p, 0)

    def u(x):
        r, χ, z = x
        return jnp.ones(1) * r**3 * (3 * jnp.log(r) - 2) / 27 + 2/27

    def f(x):
        r, χ, z = x
        return -jnp.ones(1) * r * jnp.log(r)
    types = ('clamped', 'clamped', 'constant')
    bcs = ('dirichlet', 'dirichlet', 'none')
    Λ0 = DifferentialForm(0, ns, ps, types)
    ξ, R_hat, Y_hat, Λ, τ = get_xi(_R, _Y, Λ0)
    Q = QuadratureRule(Λ0, q)
    E0 = LazyExtractionOperator(Λ0, ξ, True).M
    K = LazyStiffnessMatrix(Λ0, Q, F=F, E=E0).M
    P0 = Projector(Λ0, Q, F=F, E=E0)
    u_hat = jnp.linalg.solve(K, P0(f))
    u_h = DiscreteFunction(u_hat, Λ0, E0)
    def err(x): return (u(x) - u_h(x))
    error = (l2_product(err, err, Q, F) / l2_product(u, u, Q, F))**0.5
    return error


# %%
import time
ns = np.arange(4, 18, 2)
ps = np.arange(1, 4)
qs = np.arange(4, 11, 1)
err = np.zeros((len(ns), len(ps), len(qs)))
times = np.zeros((len(ns), len(ps), len(qs)))
for i, n in enumerate(ns):
    for j, p in enumerate(ps):
        for k, q in enumerate(qs):
            start = time.time()
            err[i, j, k] = get_err(n, p, q)
            end = time.time()
            times[i, j, k] = end - start
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
