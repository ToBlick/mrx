---
title: Harmonic fields in a hollow torus
parent: Tutorials
layout: default
nav_order: 2
---

# Harmonic fields in a hollow torus

This example is one of the tests in MRX. It computes harmonic fields in a hollow toroidal shape and checks that the computed values for the poloidal and toroidal fluxes agree with Ampère’s law.

### Parameters

We begin by defining a number of parameters as well as the mapping function $\Phi$ that maps the logical domain $[0, 1]^3$ to a hollow toroidal shape in $\mathbb{R}^3$.

```python
Ip = 2.31
It = 1.74
Is = jnp.array([Ip, It])
n = 5
p = 3
q = p + 2
ɛ = 1/3
π = jnp.pi
μ0 = 1.0

@jax.jit
def F(x):
    """Hollow toroid."""
    r, θ, ζ = x
    R = 1 + ɛ * (r + 1)/2 * jnp.cos(2 * π * θ)
    return jnp.array([R * jnp.cos(2 * π * ζ),
                      -R * jnp.sin(2 * π * ζ),
                      ɛ * (r + 1)/2 * jnp.sin(2 * π * θ)])
```

### de Rham sequence

Next, we set up the finite element spaces. Using the convenience method `Seq.assemble_all()` is only be marginally more expensive than assembling only the needed operators.

```python
ns = (n, n, n)
ps = (p, p, p)
types = ("clamped", "periodic", "periodic")
Seq = DeRhamSequence(ns, ps, q, types, F, polar=False, dirichlet=True)
Seq.evaluate_1d()
Seq.assemble_all()
```

### Harmonic fields

There are two non-trivial harmonic fields in a hollow torus, one is linked to the current $I_p$ flowing through the hole of the torus toroidally. The other is linked to the current  $I_t$ flowing poloidally around the torus and through the "donut" hole in the middle. This is our first sanity check:
```python
evs, evecs = jnp.linalg.eigh(Seq.M2 @ Seq.dd2)
assert jnp.sum(evs < 1e-10) == 2  # two harmonic fields
assert jnp.min(evs) > -1e-10  # no negative eigenvalues
```
Note that `Seq.M2 @ Seq.dd2` is the bilinear form corresponding to the weak formulation of the Laplace-de Rham operator on 2-forms. `Seq.dd2` is the operator that corresponds to it and acts on the DoFs of 2-forms.

The harmonic fields right now are not `Seq.M2`-orthonormal, so we fix this:
```python
def m2_orthonormalize(V, M):
        G = V.T @ M @ V              # 2x2 Gram
        R = jnp.linalg.cholesky(G)
        K = V @ jnp.linalg.inv(R)    # columns now M-orthonormal
        return K
K = m2_orthonormalize(evecs[:, :2], Seq.M2)
h1_dof = K[:, 0]
h2_dof = K[:, 1]
```

### Ampère’s law

Next, we compute contour integrals of the harmonic fields along two non-contractible loops in the torus: one poloidal ($c_1$) and one toroidal ($c_2$). These integrals give us the fluxes associated with each harmonic field. According to Ampère’s law, we should have
$$
\begin{align}
    \oint_{c_1} h_1 \cdot \mathrm dl_1 &= \mu_0 I_p, \quad &\oint_{c_1} h_2 \cdot \mathrm dl_2 &= \mu_0 I_t.
\end{align}
$$

We compute the line integrals in the physical domain: $\mathrm dl_1$ is nothing else than $\partial_\theta \Phi \, \mathrm d\theta$ where $\theta$ is the poloidal angle and $\mathrm dl_2 = \partial_\zeta \Phi \, \mathrm d\zeta$, where $\zeta$ is the toroidal angle.
```python
# contour wrapping around the enclosed tunnel poloidally:
def c1(χ):
    r = jnp.ones_like(χ) * 0.5
    θ = χ
    z = jnp.zeros_like(χ)
    return Seq.F(jnp.array([r, θ, z]))

# contour wrapping around the center tunnel toroidally:
def c2(χ):
    r = jnp.ones_like(χ) * 0.5
    θ = jnp.ones_like(χ) * 0.5
    z = χ
    return Seq.F(jnp.array([r, θ, z]))

def h_dl(function, curve):
    def h_dl(χ):
        return function(curve(χ)) @ jax.jacfwd(curve)(χ)
    return h_dl

# Integrate h1 along contours using trapezoidal rule
n_q = 256
_χ = jnp.linspace(0, 1, n_q, endpoint=False)
_w = jnp.ones(n_q) * (1/n_q)

P = jnp.array([
    [jax.vmap(h_dl(h, c))(_χ) @ _w for c in (c1, c2)]
    for h in (h1, h2)
])
```
The resulting matrix $\mathbb P$ we have built has entires $\mathbb P_{ij} = \oint_{c_j} h_i \cdot \mathrm dl_j$. We can now solve for the magnetic field using Ampère’s law since it holds that

$$
\begin{align}
    B = b_1 h_1 + b_2 h_2, \quad \text{where} \quad
    \mathbb P^T
    \begin{pmatrix}
        b_1 \\
        b_2
    \end{pmatrix}
    &=
    \mu_0
    \begin{pmatrix}
        I_p \\
        I_t
    \end{pmatrix}.
\end{align}
$$

```python
b = jnp.linalg.solve(P.T, μ0 * Is)
b_dofs = b[0] * h1_dof + b[1] * h2_dof
```

### Final checks

Finally, we can run a few assertions: The field should be divergence-free and curl-free. Furthermore, there are two expressions for the magnetic energy:
$$
\begin{align}
    \mathcal E(B) &= \frac{1}{2 \mu_0} \int_\Omega |B|^2 \, \mathrm dx = \frac{\mu_0}{2} \begin{pmatrix}
        I_p \\
        I_t
    \end{pmatrix}^T 
    (\mathbb P^T \mathbb P)^{-1} 
    \begin{pmatrix}
        I_p \\
        I_t
    \end{pmatrix}.
\end{align}
$$
```python
curl_b_dofs = Seq.weak_curl @ b_dofs
div_b_dofs = Seq.strong_div @ b_dofs
assert (curl_b_dofs @ Seq.M1 @ curl_b_dofs)**0.5 < 1e-10
assert (div_b_dofs @ Seq.M3 @ div_b_dofs)**0.5 < 1e-10

energy = b_dofs @ Seq.M2 @ b_dofs / (2 * μ0)
expected_energy = μ0 / 2 * Is @ jnp.linalg.solve(P.T @ P, Is)
assert jnp.abs(energy - expected_energy) / expected_energy < 1e-10
```