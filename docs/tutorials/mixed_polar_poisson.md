---
title: Mixed Poisson equation on the unit disk
parent: Tutorials
layout: default
nav_order: 2
---

# Mixed Poisson equation on the unit disk

### Mixed formulation

The Poisson equation can be re-written as
$$
\begin{align}
    \nabla \cdot \sigma &= f \\
    -\nabla u &= \sigma
\end{align}
$$
Note that only one of these can hold in a strong sense, since $\sigma$ cannot be both a two- and a one-form.

When the first equation is fulfilled in strong form and the second weakly, this leads to the system
$$
\begin{align}
    (\nabla \cdot \sigma^2, \Lambda^3) &= (f, \Lambda^3) \\
    (u^3, \nabla \cdot \Lambda^2) &= (\sigma^2, \Lambda^2),
\end{align}
$$
where we use superscripts to denote what forms the variables are interpreted as. This formulation automatically implies the boundary condition $u = 0$ on $\partial \Omega$.

### Discrete formulation
In matrix-vector form, the equations read
$$
\begin{align}
    \mathbb D \sigma^2 &= \Pi_3 f \\
    \mathbb D^T u &= \mathbb M_2 \sigma^2
\end{align}
$$
where $\mathbb D$ denotes the divergence matrix. This can be re-written as
$$
\begin{align}
    \begin{bmatrix}
        \mathbb M_2 & - \mathbb D^T \\
        \mathbb D & 0
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
\begin{align}
    \mathbb D \mathbb M_2^{-1} \mathbb D^T u = \Pi_3 f.
\end{align}
$$

### Strong and weak operators
The matrix $\mathbb M_3^{-1} \mathbb D$ corresponds to the strong divergence operator, which maps two-forms to three-forms on the level of their DoFs. The weak gradient operator is given by $-\mathbb M_2^{-1} \mathbb D^T$, it maps discrete two-forms to discrete three-forms. The strong divergence operator coincides point-wise with the analytical divergence operator when applied to discrete two-forms, while the weak gradient operator only does so in a weak $L^2$ sense.

The projector to three-forms is given by
$$
\begin{align}
    (\Pi_3 f)_i &= \int_{\Omega} f(\hat x) \frac{\Lambda^3 \circ \Phi^{-1} (x)}{\det D\Phi \circ \Phi^{-1}(x)} \, \mathrm d x
    = \int_{\hat \Omega} f(\hat x) {\Lambda^3(\hat x)} \, \mathrm d \hat x.
\end{align}
$$

### Manufactured solution
To test this case, we need a solution to Poisson's equation on a disc that satisfies $\int_{\partial \Omega} \nabla u \cdot n \, \mathrm d \sigma = \int_\Omega f \mathrm d x = 0$:

$$
\begin{align}
u(r) &= \frac 1 4 \left( \frac 1 3 r^3 - \frac 1 4 r^4 - \frac 1 {12} \right) \\
f(r) &= r \left( r - \frac 3 4 \right)
\end{align}
$$

The constant in $u$ is chosen such that $u(1) = 0$.

### Code
With all this in place, the code is straightforward. We start by defining the mapping from logical to physical space, get a `deRhamSequence` object (this time without Dirichlet boundary conditions, as the boundary conditions are natural instead of essential), assemble the required matrices, project the source term, and solve for the degrees of freedom of `u_h`.
```
def F(x):
    r, θ, z = x
    return jnp.array([r * jnp.cos(2 * jnp.pi * θ),
                      -z,
                      r * jnp.sin(2 * jnp.pi * θ)])

def u(x):
    r, θ, z = x
    return -jnp.ones(1) * (r**4/16 - r**3/12 + 1/48)

def f(x):
    r, θ, z = x
    return jnp.ones(1) * (r - 3 / 4) * r

ns = (n, n, 1)
ps = (p, p, 0)
types = ("clamped", "periodic", "constant")

Seq = DeRhamSequence(ns, ps, q, types, F, polar=True, dirichlet=False)
Seq.evaluate_1d()   
Seq.assemble_M2()   
Seq.assemble_M3()   
Seq.assemble_d2()
Seq.assemble_dd3()

u_dof = jnp.linalg.solve(Seq.M3 @ Seq.dd3, Seq.P3(f))
```
The assembly calls are done in that order as `Seq.d2` (strong divergence/weak gradient) depends on `Seq.M2`, and `Seq.dd3` (strong gradient $\circ$ weak divergence) depends on `Seq.d2`.

To get a function on the physical domain, we can then use the `Pushforward` operation:
```
u_h = Pushforward(DiscreteFunction(u_dof, Seq.Λ3, Seq.E3), F, 3)
```
The push-forward of a three-form scales the function values by the determinant of the Jacobian of the mapping.
