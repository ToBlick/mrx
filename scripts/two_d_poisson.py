# %%
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from mrx.DifferentialForms import DifferentialForm, DiscreteFunction
from mrx.Quadrature import QuadratureRule
from mrx.Projectors import Projector
from mrx.LazyMatrices import LazyMassMatrix, LazyStiffnessMatrix
from mrx.Utils import l2_product
from mrx.BoundaryConditions import LazyBoundaryOperator
from functools import partial
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
    M0 = LazyMassMatrix(Λ0, Q, F=None, E=B0).M
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
            times[i, j,k] = end - start
            print(f"n={n}, p={p}, q={q}, err={err[i,j,k]}, time={times[i,j,k]}")
# %%
base_markers = [
        'o', 'v', '*', '<', '>', '1', '2', '3', '4',
        's', 'p', '^', 'h', 'H', '+', 'x', 'D', 'd', '|', '_'
    ]
markers = [base_markers[i % len(base_markers)] for i in range(len(ps))]
cm = plt.cm.get_cmap('viridis', len(qs))
for j, p in enumerate(ps):
    for k, q in enumerate(qs):
        if k==0:
            plt.plot(ns, err[:, j, k], label=f'p={p}, q={q}', marker=markers[j], color=cm(k), markersize=8)
        else:
            plt.plot(ns, err[:, j, k], label=None, marker=markers[j], color=cm(k), markersize=8)
    plt.plot(ns, err[-1, j, 0] * (ns/ns[-1])**(-2*p), label=f'O(1/n^{2*p})', linestyle='--', color='k')

plt.loglog()
plt.xlabel('n')
plt.ylabel('Error')
plt.legend()
# %%
cm = plt.cm.get_cmap('viridis', len(qs))
for j, p in enumerate(ps):
    for k, q in enumerate(qs):
        if k==0:
            plt.plot(ns, times[:, j, k], label=f'p={p}', marker=markers[j], color=cm(k), markersize=8)
        else:
            plt.plot(ns, times[:, j, k], label=None, marker=markers[j], color=cm(k), markersize=8)
plt.loglog()
plt.xlabel('n')
plt.ylabel('Time [s]')
plt.legend()
# %%
