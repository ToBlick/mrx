# %%

from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from mrx.DeRhamSequence import DeRhamSequence
from mrx.DifferentialForms import DiscreteFunction, Pushforward

jax.config.update("jax_enable_x64", True)

R0 = 3.0
π = jnp.pi

a1 = 0.8
a2 = 1.2


def a(χ):
    """Radius as a function of chi."""
    return a1 * a2 / jnp.sqrt(a1**2 * jnp.cos(2 * π * χ)**2 + a2**2 * jnp.sin(2 * π * χ)**2)


def _R(r, χ):
    return jnp.ones(1) * (R0 + a(χ) * r * jnp.cos(2 * π * χ))


def _Z(r, χ):
    return jnp.ones(1) * a(χ) * r * jnp.sin(2 * π * χ)


def F(x):
    """Polar coordinate mapping function."""
    r, χ, z = x
    return jnp.ravel(jnp.array(
        [_R(r, χ) * jnp.cos(2 * π * z),
         -_R(r, χ) * jnp.sin(2 * π * z),
         _Z(r, χ)]))


def B_harm(p):
    x, y, z = F(p)
    R = (x**2 + y**2)**0.5
    phi = jnp.arctan2(y, x) / (2 * π)

    BR = 0
    Bz = 0
    BPhi = 1 / R

    Bx = BR * jnp.cos(2 * π * phi) - BPhi * jnp.sin(2 * π * phi)
    By = BR * jnp.sin(2 * π * phi) + BPhi * jnp.cos(2 * π * phi)

    return jnp.array([Bx, By, Bz])


def maybe_flip(B):
    flip = B[-1] > 0
    B = jax.lax.cond(flip, lambda x: -x, lambda x: x, B)
    return B


@partial(jax.jit, static_argnames=("n", "p"))
def run(n, p):
    q = 2 * p

    ns = (n, n, 1)
    ps = (p, p, 0)
    types = ("clamped", "periodic", "constant")

    Seq = DeRhamSequence(ns, ps, q, types, F, polar=True)

    M1 = Seq.assemble_M1_0()
    M2 = Seq.assemble_M2_0()

    curl = jnp.linalg.solve(M2, Seq.assemble_curl_0())
    weak_curl = jnp.linalg.solve(M1, Seq.assemble_curl_0().T)

    laplace_2 = M2 @ curl @ weak_curl + \
        Seq.assemble_divdiv_0()  # dim ker = 1 (one tunnel)

    B_harm_hat = jnp.linalg.eigh(laplace_2)[1][:, 0]
    B_harm_hat /= jnp.sqrt(B_harm_hat @ M2 @ B_harm_hat)

    B_harm_hat = maybe_flip(B_harm_hat)
    B_harm_h = DiscreteFunction(B_harm_hat, Seq.Λ2, Seq.E2_0.matrix())
    B_harm_h = Pushforward(B_harm_h, Seq.F, 2)

    B_harm_norm = jnp.sqrt(jnp.einsum("ij,ij,i,i", jax.vmap(B_harm)(
        Seq.Q.x), jax.vmap(B_harm)(Seq.Q.x), Seq.Q.w, Seq.J_j))

    def squared_difference(p):
        return jnp.sum((B_harm(p)/B_harm_norm - B_harm_h(p))**2)

    error = jnp.sqrt(jnp.einsum("i,i,i", jax.vmap(
        squared_difference)(Seq.Q.x), Seq.Q.w, Seq.J_j))

    return error
# %%


n_scan = np.arange(8, 12, 1)
p_scan = np.arange(1, 6)

errors = np.zeros((len(n_scan), len(p_scan)))

for i, n in enumerate(n_scan):
    for j, p in enumerate(p_scan):
        errors[i, j] = run(n, p)
        print(f"n={n}, p={p}, error={errors[i, j]:.3e}")
# %%

cmap = plt.cm.plasma

colors = [cmap(v) for v in np.linspace(0, 1, len(p_scan) + 1)]

fig1 = plt.figure(figsize=(10, 6))
for j, p in enumerate(p_scan):
    plt.loglog(n_scan, errors[:, j],
               label=f'$p={p_scan[j]}$',
               marker='o', color=colors[j])
    # Add theoretical convergence rates
    plt.loglog(n_scan, errors[-1, j] * (n_scan/n_scan[-1])**(-p),
               #    label=r'$O(n^{-p})$',
               linestyle='--', color='grey', alpha=0.5)

plt.xlabel(r'$n$')
plt.ylabel(r'$\| h - h_h \|_{L^2(\Omega)}$')
plt.grid(which="both", linestyle="--", linewidth=0.5)
plt.grid(True)
plt.legend()
# %%
