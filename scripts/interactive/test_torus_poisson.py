# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.scipy.sparse.linalg import cg

from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction
from mrx.mappings import toroid_map

# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)


"""
Computes the error, condition number, and sparsity of the solution to the Poisson equation on a toroidal domain.
Args:
    n: Number of elements in each direction.
    p: Polynomial degree.
Returns:
    error: Error of the solution.
    cond: Condition number of the system.
    sparsity: Sparsity of the system.
"""
# Set up finite element spaces
p = 2
n = 10

q = p
ns = (n, n, n)
ps = (p, p, p)
types = ("clamped", "periodic", "periodic")  # Types

# Domain parameters
a = 1 / 3  # minor radius
π = jnp.pi
F = toroid_map(epsilon=a)


def u(x: jnp.ndarray) -> jnp.ndarray:
    """Exact solution of the Poisson equation. Formula is:
    u(r, χ, z) = 1/4 * (r**2 - r**4) * cos(2πz)
    Args:
        x: Input logical coordinates (r, χ, z)
    Returns:
        u: Exact solution of the Poisson equation
    """
    r, χ, z = x
    return 1/4 * (r**2 - r**4) * jnp.cos(2 * π * z) * jnp.ones(1)


def f(x: jnp.ndarray) -> jnp.ndarray:
    """Source term of the Poisson equation. Formula is:
    f(r, χ, z) = cos(2πz) * (-1/a**2 * (1 - 4r**2) - 1/(a*R) * (r/2 - r**3) * cos(2πχ) + 1/4 * (r**2 - r**4) / R**2 )
    Args:
        x: Input logical coordinates (r, χ, z)
    Returns:
        f: Source term of the Poisson equation
    """
    r, χ, z = x
    R = 1 + a * r * jnp.cos(2 * jnp.pi * χ)
    return jnp.cos(2 * jnp.pi * z) * (-1/a**2 * (1 - 4*r**2) - 1/(a*R) * (r/2 - r**3) * jnp.cos(2 * jnp.pi * χ) + 1/4 * (r**2 - r**4) / R**2) * jnp.ones(1)


# Create DeRham sequence
seq = DeRhamSequence(ns, ps, q, types, F, polar=True, dirichlet=True)
seq.evaluate_1d()
seq.assemble_m0()
seq.assemble_dd0()
seq.assemble_m0_sparse()
seq.assemble_dd0_sparse()
# %%
# Solve the dense system
u_hat = jnp.linalg.solve(seq.m0 @ seq.dd0, seq.p0(f))
u_h = DiscreteFunction(u_hat, seq.basis_0, seq.e0)
# %%
# solve the sparse system
out = jax.lax.custom_linear_solve(seq.apply_dd0_sparse,
                                  seq.p0(f),
                                  lambda mv, b: cg(
                                      mv, b, tol=1e-9, M=seq.apply_dd0_precond)[0],
                                  transpose_solve=None,
                                  symmetric=True,
                                  has_aux=False)
# %%
# do not vmap here because of memory issues


def diff(x):
    return u(x) - u_h(x)


df = jax.lax.map(diff, seq.quad.x, batch_size=20_000)
L2_df = jnp.einsum('ik,ik,i,i->', df, df, seq.jacobian_j, seq.quad.w)
L2_f = jnp.einsum('ik,ik,i,i->', jax.vmap(u)(seq.quad.x),
                  jax.vmap(u)(seq.quad.x), seq.jacobian_j, seq.quad.w)
error = (L2_df / L2_f)**0.5
print(f"Relative L2 error: {error:.2e}")
# %%
