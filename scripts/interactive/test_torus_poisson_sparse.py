# %%
from functools import partial

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


@partial(jax.jit, static_argnames=['n', 'p'])
def compute_error(n, p):
    ns = (n, n, n)
    ps = (p, p, p)
    q = p+1
    # Create DeRham sequence
    seq = DeRhamSequence(ns, ps, q, types, F, polar=True,
                         dirichlet=True, tol=1e-9, maxiter=1000)
    seq.evaluate_1d()
    seq.assemble_m0_sparse()
    seq.assemble_dd0_sparse()

    u_hat = cg(seq.apply_dd0_sparse, seq.P0(f), M=seq.apply_dd0_precond,
               tol=seq.tol, maxiter=seq.maxiter)[0]
    u_h = DiscreteFunction(u_hat, seq.basis_0, seq.e0)

    def diff(x):
        return u(x) - u_h(x)

    df = jax.lax.map(diff, seq.quad.x, batch_size=40_000)
    L2_df = jnp.einsum('ik,ik,i,i->', df, df, seq.jacobian_j, seq.quad.w)
    L2_f = jnp.einsum('ik,ik,i,i->', jax.vmap(u)(seq.quad.x),
                      jax.vmap(u)(seq.quad.x), seq.jacobian_j, seq.quad.w)
    error = (L2_df / L2_f)**0.5
    return error


# %%
# Convergence sweep across n and p
ps_sweep = [1, 2, 3, 4, 5, 6]
ns_sweep = [4, 6, 8, 10, 12]

results = {}  # results[p] = list of (n, error)

for p in ps_sweep:
    results[p] = []
    for n in ns_sweep:
        if n <= p:
            continue
        err = float(compute_error(n, p))
        print(f"p={p}, n={n:>2d}  =>  relative L2 error = {err:.4e}")
        results[p].append((n, err))

# %%
# Convergence plot
fig, ax = plt.subplots(figsize=(7, 5))

markers = ["o", "s", "^", "D", "v", "P"]
for i, p in enumerate(ps_sweep):
    ns_p = jnp.array([r[0] for r in results[p]])
    errs = jnp.array([r[1] for r in results[p]])
    ax.loglog(ns_p, errs, marker=markers[i % len(markers)],
              label=f"$p = {p}$", linewidth=1.5, markersize=6)

# Reference slopes
n_ref = jnp.array([ns_sweep[0], ns_sweep[-1]], dtype=float)
for p in ps_sweep:
    slope = -(p + 1)
    _errs = jnp.array([r[1] for r in results[p]])
    ref = float(_errs[-1]) * (n_ref / n_ref[-1]) ** slope
    ax.loglog(n_ref, ref, "k--", linewidth=0.6, alpha=0.4)
    ax.annotate(f"$n^{{{slope}}}$",
                xy=(float(n_ref[0]), float(ref[0])),
                fontsize=8, color="gray",
                textcoords="offset points", xytext=(4, 4))

ax.set_xlabel("$n$ (elements per direction)")
ax.set_ylabel("Relative $L^2$ error")
ax.set_title("Poisson on torus — $h$-convergence for different $p$")
ax.legend()
ax.grid(True, which="both", linewidth=0.3)
fig.tight_layout()
# plt.savefig("convergence_poisson_torus.png", dpi=200)
plt.show()

# %%
