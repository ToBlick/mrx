# %%
import os
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import optimistix as optx

# import numpy as np
import scipy as sp

from mrx.DeRhamSequence import DeRhamSequence
from mrx.DifferentialForms import DifferentialForm, DiscreteFunction, Pushforward
from mrx.LazyMatrices import LazyDerivativeMatrix, LazyMassMatrix, LazyStiffnessMatrix
from mrx.PolarMapping import LazyExtractionOperator, get_xi
from mrx.Projectors import Projector
from mrx.Quadrature import QuadratureRule
from mrx.Utils import grad, jacobian_determinant, l2_product

# import time
# from functools import partial


# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)


def generalized_eigh(A, B):
    # Add small value for numerical stability
    L = jnp.linalg.cholesky(B + jnp.eye(B.shape[0]) * 1e-12)
    L_inv = jnp.linalg.inv(L)
    C = L_inv @ A @ L_inv.T
    eigenvalues, eigenvectors_transformed = jnp.linalg.eigh(C)
    eigenvectors_original = L_inv.T @ eigenvectors_transformed
    return eigenvalues, eigenvectors_original

# %%


@partial(jax.jit, static_argnames=["n", "p"])
def get_evs(a_hat, n, p):

    Λmap = DifferentialForm(0, (n, 1, 1), (p, 1, 1),
                            ('periodic', 'constant', 'constant'))

    _a_h = DiscreteFunction(a_hat, Λmap)

    def a_h(x):
        _x = jnp.array([x, 0, 0])
        return _a_h(_x)

    def F(x):
        """Polar coordinate mapping function."""
        r, χ, z = x
        return jnp.array([a_h(χ) * r * jnp.cos(2 * jnp.pi * χ),
                          -z,
                          a_h(χ) * r * jnp.sin(2 * jnp.pi * χ)])

    # Set up finite element spaces
    ns = (n, n, 1)
    ps = (p, p, 0)
    q = 3
    types = ("clamped", "periodic", "constant")
    bcs = ('dirichlet', 'none', 'none')

    Seq = DeRhamSequence(ns, ps, q, types, bcs, F, polar=True)

    K = Seq.assemble_gradgrad()
    M0 = Seq.assemble_M0()
    # Solve the system
    evs, evecs = generalized_eigh(K, M0)
    # finite_indices = evs > 0
    # jnp.ispositive
    # evs = evs[finite_indices]
    return evs, evecs


# %%
n = 8
p = 3

# %%
a = 1.0
e = 0.5

Λmap = DifferentialForm(0, (n, 1, 1), (p, 1, 1),
                        ('periodic', 'constant', 'constant'))
Q = QuadratureRule(Λmap, 3*p)
P_0 = Projector(Λmap, Q)
M0 = LazyMassMatrix(Λmap, Q).matrix()


def radius(chi):
    return jnp.ones(1) * a * (1 + e * jnp.cos(2 * jnp.pi * chi)) / (1 - e**2)**0.5


def _radius(x):
    chi = x[0]
    return radius(chi)


a_target = jnp.linalg.solve(M0, P_0(_radius))

# %%
k = 36
_k = jnp.arange(k)

plt.plot(_k, get_evs(jnp.ones(n), n, p)[0][:k], marker='s', label=r'circle')
plt.plot(_k, get_evs(a_target, n, p)[0][:k], marker='s', label=r'ellipse')
plt.xlabel(r'$k$')
plt.ylabel(r'$\lambda_k$')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()

# %%
a_hat = jnp.ones(n)
target_evs = get_evs(a_target, n, p)[0][:k]


def fit_evs(a_hat, args):
    _k = jnp.arange(k) + 1
    evs = get_evs(a_hat, n, p)[0][:k]
    return jnp.sum((evs/_k - target_evs/_k)**2)


grad_fit = jax.jit(jax.value_and_grad(fit_evs))
# %%
# Momentum gradient descent
key = jax.random.PRNGKey(1)
a_hat = jnp.ones(n) + jax.random.normal(key, (n,)) * 0.5
alpha = 5e-4
beta = 0.9
g = jnp.zeros_like(a_hat)

for i in range(100):
    v, grad_a = grad_fit(a_hat, None)
    print('Objective function: {:.3E}'.format(v))
    a_hat = a_hat - alpha * grad_a + beta * g


# %%
# a_hat = jnp.ones(n)
# solver = optax.adam(learning_rate=1e-4)
# opt_state = solver.init(a_hat)
# for _ in range(100):
#     v, gradf = grad_fit(a_hat, None)
#     updates, opt_state = solver.update(gradf, opt_state, a_hat)
#     a_hat = optax.apply_updates(a_hat, updates)
#     print('Objective function: {:.3E}'.format(v))

# %%
# solver = optx.LevenbergMarquardt(
#     rtol=1e-8, atol=1e-8, verbose=frozenset({"step", "accepted", "loss", "step_size"})
# )

# sol = optx.least_squares(
#     fn=fit_evs,
#     y0=a_hat,
#     solver=solver,
#     max_steps=100
# )

# %%
k_plot = 36
_k_plot = jnp.arange(k_plot)
plt.plot(_k_plot, get_evs(jnp.ones(n), n, p)[0][
         :k_plot], marker='s', label=r'circle')
plt.plot(_k_plot, get_evs(a_target, n, p)[0][
         :k_plot], marker='s', label=r'ellipse')
plt.plot(_k_plot, get_evs(a_hat, n, p)[0][:k_plot], marker='s', label=r'fit')
plt.xlabel(r'$k$')
plt.ylabel(r'$\lambda_k$')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()
# %%

Λmap = DifferentialForm(0, (n, 1, 1), (p, 1, 1),
                        ('periodic', 'constant', 'constant'))
_a_h = DiscreteFunction(a_hat, Λmap)


def a_h(x):
    _x = jnp.array([x, 0, 0])
    return _a_h(_x)


def _R(r, χ):
    """Compute the R coordinate in polar mapping."""
    return jnp.ones(1) * (a_h(χ) * r * jnp.cos(2 * jnp.pi * χ))


def _Y(r, χ):
    """Compute the Y coordinate in polar mapping."""
    return jnp.ones(1) * (a_h(χ) * r * jnp.sin(2 * jnp.pi * χ))


def F(x):
    r, χ, z = x
    return jnp.ravel(jnp.array([_R(r, χ), _Y(r, χ), jnp.ones(1) * z]))


ns = (n, n, 1)
ps = (p, p, 0)
types = ('clamped', 'periodic', 'constant')
Λ0 = DifferentialForm(0, ns, ps, types)
Λ3 = DifferentialForm(3, ns, ps, types)
Q = QuadratureRule(Λ0, 3*p)
ξ, R_hat, Y_hat, Λ, τ = get_xi(_R, _Y, Λ0, Q)
# Set up operators
E3 = LazyExtractionOperator(Λ3, ξ, False).M

# %%
evs, evecs = get_evs(a_hat, n, p)
# %%

ɛ = 1e-6
nx = 64
_nx = 16
_x1 = jnp.linspace(ɛ, 1 - ɛ, nx)
_x2 = jnp.linspace(0, 1, nx)
_x3 = jnp.zeros(1) / 2
_x = jnp.array(jnp.meshgrid(_x1, _x2, _x3))
_x = _x.transpose(1, 2, 3, 0).reshape(nx * nx * 1, 3)
_y = jax.vmap(F)(_x)
_y1 = _y[:, 0].reshape(nx, nx)
_y2 = _y[:, 1].reshape(nx, nx)
# %%
u_h = Pushforward(DiscreteFunction(evecs[:, 0], Λ3, E3), F, 3)
plt.contourf(_y1, _y2, jax.vmap(u_h)(_x).reshape(nx, nx), levels=10)

# %%


def radius_h(x):
    _radius_h = DiscreteFunction(a_hat, Λmap)
    _x = jnp.array([x, 0, 0])
    return _radius_h(_x)


__x = jnp.linspace(0, 1, 100)

plt.plot(__x, jax.vmap(radius_h)(__x), label='fitted radius')
plt.plot(__x, jax.vmap(radius)(__x), label='target radius')
plt.xlabel(r'$\chi$')
plt.ylabel(r'$r(\chi)$')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()
# %%
