---
title: Harmonic fields in a hollow torus
parent: Tutorials
layout: default
nav_order: 2
---

# Harmonic fields in a hollow torus

This example is one of the tests in MRX. It computes harmonic fields in a hollow toroidal shape and checks that the computed values for the poloidal and toroidal fluxes agree with Ampère’s law and approximate analytical expressions in the thin-torus limit.

### Parameters

We begin by defining a number of parameters as well as the mapping function $\Phi$ that maps the logical domain $[0, 1]^3$ to a hollow toroidal shape in $\mathbb{R}^3$.
```python
Ip = 1.95
It = 2.46
Is = jnp.array([Ip, It])
n = 6
p = 3
q = p + 2
ɛ = 0.1
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

There are two non-trivial harmonic fields in a hollow torus, one is linked to the current $I_p$ flowing through the cavity/tunnel in the torus toroidally. The other is linked to the current  $I_t$ flowing poloidally around the torus and through the "donut" hole in the middle. This is our first sanity check (note that generalized eigenvalue problems are not yet implemented in `jax.numpy.linalg`, so we use `scipy.linalg.eigh` here):
```python
evs, evecs = sp.linalg.eigh(Seq.M2 @ Seq.dd2, Seq.M2)
assert jnp.sum(evs < 1e-11) == 2  # two harmonic fields
assert jnp.min(evs) > -1e-11  # no negative eigenvalues
```
Note that `Seq.M2 @ Seq.dd2` is the bilinear form corresponding to the weak formulation of the Laplace-de Rham operator on 2-forms. `Seq.dd2` is the operator that corresponds to it and acts on the DoFs of 2-forms. In other words, when $u_h = \sum_i \mathtt{u}_i \Lambda_i^2 \in V_2$:
$$
\begin{align}
    (\omega^2 - \Delta) u_h = 0 \quad \text{in } V_2 \quad \Leftrightarrow \quad
    (\omega^2 + \text{dd2}) \mathtt{u} = 0
\end{align}
$$

The harmonic fields are the corresponding `Seq.M2`-orthonormal eigenvectors:
```python
h1_dof = evecs[:, 0]
h2_dof = evecs[:, 1]
h1 = jax.jit(DiscreteFunction(h1_dof, Seq.Λ2, Seq.E2))
h2 = jax.jit(DiscreteFunction(h2_dof, Seq.Λ2, Seq.E2))
```

### Ampère’s law

Next, we compute contour integrals of the harmonic fields along two non-contractible loops in the torus: one poloidal ($c_1$) and one toroidal ($c_2$). These integrals give us the fluxes associated with each harmonic field. According to Ampère’s law, we should have
$$
\begin{align}
    \oint_{c_p} h_p \cdot \mathrm dl_p &= \mu_0 I_p, \quad &\oint_{c_t} h_t \cdot \mathrm dl_t &= \mu_0 I_t.
\end{align}
$$

We compute the line integrals in the logical domain: $\hat c_1$ and $\hat c_2$ have simple expressions:
```python
# contour wrapping around the enclosed tunnel poloidally:
def c1(θ): return jnp.array([1e-6, θ, 0])
# contour wrapping around the center tunnel toroidally:
def c2(ζ): return jnp.array([1 - 1e-6, 0.5, ζ])
```
The small offsets `1e-6` ensure that the contours lie in the interior of the physical domain. If $h_1$ and $h_2$ were one-forms, we could directly evaluate them along the curves in the logical domain since $\int E \cdot \mathrm dl = \int \hat E \cdot \mathrm d \hat l$ for one-forms under pushforward $E = \Phi_*^1 \hat E$.

However, they are two-forms, so instead it holds that
$$
\begin{align}
    \oint_{c} h \cdot \mathrm dl &= \oint_{c} \left( (\Phi^{-1})_*^1 \, \Phi_*^2 \, \hat h \right) \cdot \mathrm d \hat l, \; \text{where} \; \left( (\Phi^{-1})_*^1 \, \Phi_*^2 \, \hat h \right)(\hat x) = \frac{D \Phi(\hat x)^T D \Phi(\hat x) \hat h(\hat x)}{\det(D \Phi(\hat x))}.
\end{align}
$$
In code, the operation $(h, c) \mapsto \oint_c h \cdot \mathrm dl$ is given by:
```python
def h_dl(twoform, curve):
    def oneform(x):
        DF = jax.jacfwd(F)(x)
        return DF.T @ DF @ twoform(x) / jnp.linalg.det(DF)

    def integrand(χ):
        x = curve(χ)
        dx = jax.jacfwd(curve)(χ).reshape(-1)
        v = oneform(x).reshape(-1)
        return jnp.dot(v, dx)

    return integrand

# Integrate h1 along contours using trapezoidal rule
n_q = 256
_χ = jnp.linspace(0, 1, n_q, endpoint=False)
_w = jnp.ones(n_q) * (1/n_q)
```
Next, we build the matrix $\mathbb P$ with entires $\mathbb P_{ij} = \oint_{c_j} h_i \cdot \mathrm dl_j$. We can now solve for the magnetic field using Ampère’s law. It holds that
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
The ordering of $h_1$ and $h_2$ is of course arbitrary, it is this calculation that connects them to $I_p$ and $I_t$. As expected, the matrix $\mathbb P$ only has two entries that are non-zero and these two differ by a multiplicative factor $\varepsilon$.
```python
P = jnp.array([
        [jax.vmap(h_dl(h, c))(_χ) @ _w for c in (c1, c2)]
        for h in (h1, h2)
    ])

b = jnp.linalg.solve(P.T, μ0 * Is)
b_dofs = b[0] * h1_dof + b[1] * h2_dof
```

### Final checks

Finally, we can run a few assertions: The field should be divergence-free and curl-free.
```python
curl_b_dofs = Seq.weak_curl @ b_dofs
div_b_dofs = Seq.strong_div @ b_dofs
assert (curl_b_dofs @ Seq.M1 @ curl_b_dofs)**0.5 < 1e-10
assert (div_b_dofs @ Seq.M3 @ div_b_dofs)**0.5 < 1e-10
```
Furthermore, we know that $h_t = \mu_0 I_t \mathbf e_\phi / (2 \pi R)$ - this is simply the vacuum field of a solid torid with the correct magnitude to match $I_t$. For the poloidal field, $h_p \approx \mu_0 I_p \mathbf e_\theta / (2 \pi d)$, where $d$ is the distance to the centerline of the enclosed tunnel assuming $\varepsilon \ll 1$. At $\hat x = (r, θ, ζ) = (0.5, 0, 0)$, we have $B_r = B_x$, $B_θ = B_z$, and $B_ζ = -B_y$ in Cartesian coordinates. Thus, we can check the values of the computed magnetic field $B$ at this point: The error in $B_e$ should be essentially zero since neither $h_1$ nor $h_2$ have a radial component. The error in $B_ζ$ is determined by the resolution, while that in $B_θ$ is dominated by the thin-torus approximation.
```python
def B_expected(x):
    r, θ, ζ = x
    d = ɛ * (r + 1) / 2
    R = 1 + d * jnp.cos(2 * π * θ)
    sζ, cζ = jnp.sin(2 * π * ζ), jnp.cos(2 * π * ζ)
    sθ, cθ = jnp.sin(2 * π * θ), jnp.cos(2 * π * θ)

    B_ζ = μ0 * It / (2 * π * R)
    B_θ = μ0 * Ip / (2 * π * d)

    Bx = -B_ζ * sζ - B_θ * sθ * cζ
    By = -B_ζ * cζ + B_θ * sθ * sζ
    Bz = B_θ * cθ

    return jnp.array([Bx, By, Bz])

B_computed = jax.jit(Pushforward(DiscreteFunction(
    b_dofs, Seq.Λ2, Seq.E2), Seq.F, 2))
    
y = jnp.array([0.5, 0.0, 0.0])
B_diff = B_computed(y) - B_expected(y)

assert jnp.abs(B_diff[0]) < 1e-12
assert jnp.abs(B_diff/B_expected(y))[1] < (1/n)**p
assert jnp.abs(B_diff/B_expected(y))[2] < ɛ
```