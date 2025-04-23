# %%
import jax
import jax.numpy as jnp
import numpy as np
import time

from mrx.DifferentialForms import DifferentialForm, DiscreteFunction
from mrx.Quadrature import QuadratureRule
from mrx.Projectors import Projector
from mrx.LazyMatrices import LazyStiffnessMatrix

from mrx.Utils import l2_product
from mrx.BoundaryConditions import LazyBoundaryOperator
from functools import partial

from mrx.Plotting import converge_plot

jax.config.update("jax_enable_x64", True)
# %%
###
# 2D Poisson problem, Dirichlet BCs
###


@partial(jax.jit, static_argnames=['n', 'p', 'q'])
def get_err(n, p, q):
    ns = (n, n, 1)
    ps = (p, p, 0)

    def u(x):
        r, χ, z = x
        return jnp.ones(1) * jnp.sin(2 * jnp.pi * r) * jnp.sin(2 * jnp.pi * χ)

    def f(x):
        return 2 * (2*jnp.pi)**2 * u(x)

    types = ('clamped', 'clamped', 'constant')
    bcs = ('dirichlet', 'dirichlet', 'none')

    Λ0 = DifferentialForm(0, ns, ps, types)
    Q = QuadratureRule(Λ0, q)

    B0 = LazyBoundaryOperator(Λ0, bcs).M
    K = LazyStiffnessMatrix(Λ0, Q, F=None, E=B0).M

    P0 = Projector(Λ0, Q, E=B0)
    u_hat = jnp.linalg.solve(K, P0(f))
    u_h = DiscreteFunction(u_hat, Λ0, B0)
    def err(x): return u(x) - u_h(x)
    Q_high = QuadratureRule(Λ0, 10)
    return (l2_product(err, err, Q_high) / l2_product(u, u, Q_high))**0.5


# %%
print(get_err(8, 3, 3))
# %%
ns = np.arange(4, 18, 2)
ps = np.arange(1, 4)
qs = np.arange(4, 11, 3)
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
    legend_title='Legend'
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
    legend_title='Legend'
)
fig.show()
# %%
