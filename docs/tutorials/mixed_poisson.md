---
title: Poisson Problem - mixed formulation
parent: Tutorials
layout: default
nav_order: 2
---

### Mixed Poisson equation on a square

The Poisson equation can be re-written as
$$
\begin{align}
    -\nabla \cdot \sigma &= f \\
    \nabla u &= \sigma
\end{align}
$$
Note that only one of these can hold in a strong sense, since $\sigma$ cannot be both a two- and a one-form.

When the first equation is fulfilled in strong form and the second weakly, this leads to the system
$$
\begin{align}
    -(\nabla \cdot \sigma^2, \phi^3) &= (f, \phi^3) \\
    -(u^3, \nabla \cdot \psi^2) &= (\sigma^2, \psi^2),
\end{align}
$$
where we use subscripts to denote what forms the variables are interpreted as. This fomrulation automatically implies homogeneous Neumann boundary conditions $\sigma \cdot n = 0$ on $\partial \Omega$.

In matrix-vector form, the equations read
$$
\begin{align}
    -\mathbb D \sigma^2 &= \Pi_3 f \\
    -\mathbb D^T u &= \mathbb M_2 \sigma^2
\end{align}
$$
where $\mathbb D$ denotes the divergence matrix. This can be re-written as
$$
\begin{align}
    \begin{bmatrix}
        \mathbb M_2 & \mathbb D^T \\
        -\mathbb D & 0
    \end{bmatrix}
    \begin{bmatrix}
        \sigma \\ u
    \end{bmatrix}
    = 
    \begin{bmatrix}
        0 \\ \Pi_3 f
    \end{bmatrix}
\end{align}
$$
and back-substituted to get
$$
    \mathbb D \mathbb M_2^{-1} \mathbb D^T u = \Pi_3 f.
$$

Note that 
$$
\begin{align}
    (\Pi_3 f)_i &= \int_{\Omega} f(x) \frac{\Lambda^3 \circ F^{-1} (x)}{\det DF \circ F^{-1}(x)} \, \mathrm d x \\
    &= \int_{\hat \Omega} \hat f(\hat x) \frac{\Lambda^3(\hat x)}{\det DF(\hat x)} \det DF(\hat x) \, \mathrm d \hat x.
\end{align}
$$

In code, the mixed Poisson problem is solved as

```
ns = (n, n, 1)
ps = (p, p, 0)

def u(x):
    r, χ, z = x
    return jnp.ones(1) * jnp.sin(2 * jnp.pi * r) * jnp.sin(2 * jnp.pi * χ)

def f(x):
    return 2 * (2*jnp.pi)**2 * u(x)
types = ('clamped', 'clamped', 'constant')

Λ2 = DifferentialForm(2, ns, ps, types)
Λ3 = DifferentialForm(3, ns, ps, types)
Q = QuadratureRule(Λ0, 3)

D = LazyDerivativeMatrix(Λ2, Λ3, Q).M
M2 = LazyMassMatrix(Λ2, Q).M
K = D @ jnp.linalg.solve(M2, D.T)
P3 = Projector(Λ3, Q)
u_hat = jnp.linalg.solve(K, P3(f))
u_h = DiscreteFunction(u_hat, Λ3)
```