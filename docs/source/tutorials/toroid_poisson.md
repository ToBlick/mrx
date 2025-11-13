---
title: Poisson equation on a toroid
parent: Tutorials
layout: default
nav_order: 2
---

# Poisson equation on a toroid

This example is essentially the same as the Poisson equation on the unit disk, but with a different mapping function $\Phi$ that maps the logical domain $[0, 1]^3$ to a toroidal shape in $\mathbb{R}^3$ and uses periodic boundary conditions in both angular directions. This example is what the convergence tests in the MRX paper are done on.

```python
ɛ = 1/3
π = jnp.pi

def F(x):
    """Polar coordinate mapping function."""
    r, θ, z = x
    R = 1 + ɛ * r * jnp.cos(2 * π * θ)
    return jnp.array([R * jnp.cos(2 * π * z),
                      -R * jnp.sin(2 * π * z),
                      ɛ * r * jnp.sin(2 * π * θ)])

# Define exact solution and source term
def u(x):
    """Exact solution of the Poisson problem."""
    r, θ, z = x
    return (r**2 - r**4) * jnp.cos(2 * π * z) * jnp.ones(1)

def f(x):
    """Source term of the Poisson problem."""
    r, χ, z = x
    R = 1 + ɛ * r * jnp.cos(2 * jnp.pi * χ)
    return jnp.cos(2 * jnp.pi * z) * (-4/ɛ**2 * (1 - 4*r**2) - 4/(ɛ*R) * (r/2 - r**3) * jnp.cos(2 * jnp.pi * χ) + (r**2 - r**4) / R**2) * jnp.ones(1)

# Set up finite element spaces
ns = (n, n, n)
ps = (p, p, p)
types = ("clamped", "periodic", "periodic")
Seq = DeRhamSequence(ns, ps, q, types, F, polar=True, dirichlet=True)
Seq.evaluate_1d()
Seq.assemble_M0()
Seq.assemble_dd0()

# Solve the system
u_hat = jnp.linalg.solve(Seq.M0 @ Seq.dd0, Seq.P0(f))
u_h = DiscreteFunction(u_hat, Seq.Λ0, Seq.E0)
```