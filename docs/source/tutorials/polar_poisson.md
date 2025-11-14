---
title: Poisson equation on the unit disk
parent: Tutorials
layout: default
nav_order: 2
---

# Poisson equation on the unit disk

## Mapping

First, we define our mapping function. MRX is written with 3D problems in toroidal geometry in mind, so this map is still three-dimensional as $\Phi : [0, 1]^3 \mapsto \mathbb{R}^3$:
```python
def Phi(x):
    r, θ, z = x
    return jnp.array([r * jnp.cos(2 * jnp.pi * θ),
                      -z,
                      r * jnp.sin(2 * jnp.pi * θ)])
```
## Manufactured solution
We also define the source $f$ and the exact solution $u$ where $-\Delta u = f$:
```python
def f(x):
    r, θ, z = x
    return -jnp.ones(1) * r * jnp.log(r)

def u(x):
    r, θ, z = x
    return jnp.ones(1) * (r**3 * (3 * jnp.log(r) - 2) / 27 + 2 / 27)
```
In MRX, scalar functions are represented as arrays with a single element (or one channel in ML-lingo), hence the use of `jnp.ones(1) * ...`.

## Resolution and periodicity
Next, we set the degree of basis functions in the three spatial dimensions. All code is written in 3D, hence solving a 2D problem is done by setting one of the basis functions constant:
```python
ns = (n, n, 1)
ps = (p, p, 0)
types = ("clamped", "periodic", "constant")
```
The `types` tuple defines the type of basis functions in each spatial dimension. Here, we use clamped B-splines in $r$, periodic B-splines in $\theta$, and constant basis functions in $z$.

## de Rham sequence
We will solve the Poisson problem using zero-forms with Dirichlet boundary conditions. The central object to create is the `DeRhamSequence`:
```python
Seq = DeRhamSequence(ns, ps, q, types, Phi, polar=True, dirichlet=True)
```

The Sequence object is a factory to create all relevant matrices, projectors, and other operators we need. On creation, it pre-computes some useful quantities at all quadrature points such as the (inverse) Metric of the mapping $\Phi$, `Seq.G_jkl` and `Seq.G_inv_jkl` with shape `(n_q, 3, 3)` and its determinant `Seq.J_j` with shape `(n_q,)`.

## Matrix assembly
To assemble the matrices we need, we first evaluate all the 1D basis splines at all quadrature points in each spatial dimension
```python
Seq.evaluate_1d()
```
and then call the assemblers for our mass and stiffness matrices:
```python
Seq.assemble_M0()
Seq.assemble_dd0()
```

## Differential forms

Internally, the `DeRhamSequence` creates `DifferentialForm` objects `Seq.Λ0`, `Seq.Λ1`, ... for zero-forms, one-forms, ... respectively.  `DifferentialForm` objects support indexing and evaluation - to evaluate the i-th basis function at a point `x`, we would call `Λ0[i](x)` where the shape of `x` is `(3,)`.

## Handling the polar singularity and boundary conditions

At the bottom of everything stands a cartesian product of one-dimensional spline bases. However, not all combinations of 1D splines are valid basis functions in physical space. At the polar singularity, only basis functions that are constant in $\theta$ may be non-zero. This introduces a set of linear constraints on the discrete function space, effectively removing some basis functions from the space. The number of constraints depends on the required regularity at the singularity. In MRX, we always enforce $C^1$ regularity at the polar singularity. This removes $2 n_\theta n_\zeta$ basis functions from the cartesian product space and replaces them by $3 n_\zeta$ new basis functions.

Analogously, homogeneous Dirichlet boundary conditions at $r = 1$ removes all basis functions that are non-zero at $r = 1$. Because we are using clamped splines, only a single $r$ basis function is non-zero at the boundary, so $n_\theta \times n_\zeta$ basis functions are removed.

The way that these constraints are implemented is by multiplying the basis functions evaluation with a rectangular matrix. Discrete functions with constraints applied hence have a lower amount of degrees of freedom $\mathring{n} < n = n_r n_\theta n_\zeta$:
\begin{align}
f_h(x) &= \sum_{i=0}^{n-1} \mathtt{f}_i \Lambda_i(x) \quad \text{(no constraints applied)} \\
\mathring{f}_h(x) &= \sum_{j=0}^{\mathring{n}-1} {\mathring{\mathtt{f}}}_j \mathring\Lambda_j(x) = \sum_{j=0}^{\mathring{n}-1} {\mathring{\mathtt{f}}_j} \sum_{i=0}^{n-1} \mathbb E_{ji} \Lambda_i(x).
\end{align}
Analogously, we assemble the stiffness matrix $\mathring {\mathbb K}$. Its $i,j$-th element is
\begin{align}
\mathring{\mathbb K}_{ij} = \int_{\hat \Omega} \hat \nabla \mathring\Lambda_i \cdot (D\Phi)^{-1} (D\Phi)^{-T} \hat \nabla \mathring\Lambda_j \, \det D\Phi \, \mathrm d \hat x
\end{align}
In practice, it is assembled by computing $\mathbb K$ (the stiffness matrix with no constraints applied) and then contracting it on both sides with $\mathbb E$ as $\mathring{\mathbb K} = \mathbb E \mathbb K \mathbb E^T$. The matrix `Seq.dd0` is $\mathring{\mathbb M}_0^{-1} \mathring{\mathbb K}$.

## Pre-computations

Both quadrature grid and spline basis have Cartesian product structure, i.e.
\begin{align}
    x^q_j &= (r^q_{j_r}, \, \theta^q_{j_\theta}, \, \zeta^q_{j_\zeta}) \quad \text{and} \quad \Lambda_i = \lambda_{i_r} \otimes \lambda_{i_\theta} \otimes \lambda_{i_\zeta},
\end{align}
where $x^q_j$ is the $j$-th quadrature point and $\Lambda_i$ is the $i$-th basis function. The indices satisfy: $0 \leq i \leq n = n_r n_\theta n_\zeta$, $0 \leq i_\nu \leq n_\nu$, $0 \leq j \leq n^q = n^q_r n^q_\theta n^q_\zeta$, and $0 \leq j_\nu \leq n^q_\nu$, where $\nu \in \{r, \theta, \zeta\}$. Using this, we can pre-compute the evaluations of the 1D basis functions at the 1D quadrature points. Then, to evaluate both the mass and stiffness matrices, the $i$-th basis function evaluated at the $j$-th quadrature point can be written as
\begin{align}
    \Lambda_i(x^q_j) = \lambda_{i_r}(r^q_j) \, \lambda_{i_\theta}(\theta^q_j) \, \lambda_{i_\zeta}(x^q_j).
\end{align}
We can pre-compute the values of $\lambda_{i_r}(r^q_j)$, $\lambda_{i_\theta}(\theta^q_j)$, and $\lambda_{i_\zeta}(x^q_j)$ at low memory cost. The memory requirement is $\sum_{\nu \in \{r, \theta, \zeta \}} n_\nu n^q_\nu$ as opposed to $\prod_{\nu \in \{r, \theta, \zeta \}} n_\nu n^q_\nu$. We use these pre-computed values to evaluate all basis functions at all quadrature points.

## Matrix solve
To solve the Poisson problem itself, we follow the usual arguments, starting from the weak form
\begin{align}
\sum_{i=0}^{m-1} \mathring{\mathtt{u}}_i \int_{\hat \Omega} \hat \nabla \mathring\Lambda_i \cdot (D\Phi)^{-1} (D\Phi)^{-T} \hat \nabla \mathring\Lambda_j \, \det D\Phi \, \mathrm d \hat x = \int_{\hat \Omega} \hat f \mathring\Lambda_j \, \det D\Phi \, \mathrm d \hat x.
\end{align}
The function $\hat f$ is the pull-back of $f$ into the logical domain where $f$ is treated as a zero-form, i.e., $\hat f(\hat x) = f \circ F(\hat x)$. This right-hand-side is evaluated using a `Projector` object that corresponds to the operation 
\begin{align}
\mathring\Pi_0: \hat f \mapsto \left( \int_{\hat \Omega} \hat f \mathring\Lambda_j \, \det D\Phi \, \mathrm d \hat x \right)_{j = 0}^{\mathring{n}-1}.
\end{align}
The expression $\mathring{\mathbb M}_0^{-1} \mathring\Pi_0(\hat f)$ gives the DoFs of the $L^2$ projection of $\hat f$ onto the discrete zero-form space.

With all this in place, we can solve for the $u$ DoFs and create a `DiscreteFunction` object that supports evaluation as `u_h(x)`:
```python
u_dof = jnp.linalg.solve(Seq.M0 @ Seq.dd0, Seq.P0(f))
u_h = DiscreteFunction(u_dof, Seq.Λ0, Seq.E0)
```
The only thing left to do is to compute the $L^2$ error between the discrete solution `u_h` and the exact solution `u`. This is done by evaluating both functions at `Seq`s quadrature points and computing a weighted sum:
```python
def diff_at_x(x):
    return u(x) - u_h(x)
df_at_x = jax.vmap(diff_at_x)(Seq.Q.x)
f_at_x = jax.vmap(u)(Seq.Q.x)
L2_df = jnp.einsum('ik,ik,i,i->', df_at_x, df_at_x, Seq.J_j, Seq.Q.w)**0.5
L2_f = jnp.einsum('ik,ik,i,i->', f_at_x, f_at_x, Seq.J_j, Seq.Q.w)**0.5
error = L2_df / L2_f
```