# %%
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from mrx.SplineBases import SplineBasis, DerivativeSpline, TensorBasis
from mrx.PolarMapping import LazyExtractionOperator, get_xi
from mrx.DifferentialForms import DifferentialForm, DiscreteFunction, Pullback, Pushforward
from mrx.Quadrature import QuadratureRule
from mrx.Projectors import Projector
from mrx.LazyMatrices import LazyMassMatrix, LazyStiffnessMatrix, LazyProjectionMatrix
from mrx.Utils import l2_product
from mrx.BoundaryConditions import LazyBoundaryOperator
from functools import partial

# %%
###
# 2D Poisson problem, Dirichlet BCs
###
@partial(jax.jit, static_argnames=['n', 'p'])
def get_err(n, p):
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
    Q = QuadratureRule(Λ0, 10)

    B0 = LazyBoundaryOperator(Λ0, bcs).M
    M0 = LazyMassMatrix(Λ0, Q, F=None, E=B0).M
    K = LazyStiffnessMatrix(Λ0, Q, F=None, E=B0).M

    P0 = Projector(Λ0, Q, E=B0)
    u_hat = jnp.linalg.solve(K, P0(f))
    u_h = DiscreteFunction(u_hat, Λ0, B0)
    err = lambda x: u(x) - u_h(x)
    return (l2_product(err, err, Q) / l2_product(u, u, Q))**0.5


# %%
print(get_err(8, 3))
# %%
import time
ns = np.arange(5, 25, 2)
ps = np.arange(1, 4)
err = np.zeros((len(ns), len(ps)))
times = np.zeros((len(ns), len(ps)))
for i, n in enumerate(ns):
    for j, p in enumerate(ps):
        start = time.time()
        err[i,j] = get_err(n, p)
        end = time.time()
        times[i,j] = end - start
        print(f"n={n}, p={p}, err={err[i,j]}, time={times[i,j]}")
# %%
plt.plot(ns, err[:,0], label='p=1', marker='o')
plt.plot(ns, err[:,1], label='p=2', marker='*')
plt.plot(ns, err[:,2], label='p=3', marker='s')
plt.plot(ns, err[0,0] * (ns/ns[0])**(-2), label='O(n^-2)', linestyle='--')
plt.plot(ns, err[0,1] * (ns/ns[0])**(-4), label='O(n^-4)', linestyle='--')
plt.plot(ns, err[0,2] * (ns/ns[0])**(-6), label='O(n^-6)', linestyle='--')
plt.loglog()
plt.xlabel('n')
plt.ylabel('Error')
plt.legend()
# %%
plt.plot(ns, times[:,0], label='p=1', marker='o')
plt.plot(ns, times[:,1], label='p=2', marker='*')
plt.plot(ns, times[:,2], label='p=3', marker='s')
plt.plot(ns, times[0,0] * (ns/ns[0])**(4), label='O(n^4)', linestyle='--')
plt.loglog()
plt.xlabel('n')
plt.ylabel('Time [s]')
plt.legend()
# %%
