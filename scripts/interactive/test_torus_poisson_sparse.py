# %%
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.scipy.sparse.linalg import cg

import mrx
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


# @partial(jax.jit, static_argnames=['n', 'p'])
def compute_error(n, p):
    timings = {}
    ns = (n, n, n)
    ps = (p, p, p)
    q = 2*p

    # Create DeRham sequence
    t0 = time.perf_counter()
    seq = DeRhamSequence(ns, ps, q, types, F, polar=True,
                         dirichlet=True, tol=1e-9, maxiter=1000)
    timings['DeRhamSequence.__init__'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    seq.evaluate_1d()
    timings['evaluate_1d'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    seq.assemble_m0_sparse()
    timings['assemble_m0_sparse'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    seq.assemble_dd0_sparse()
    timings['assemble_dd0_sparse'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    rhs = seq.P0(f)
    jax.block_until_ready(rhs)
    timings['P0(f)'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    u_hat = cg(seq.apply_dd0_sparse, rhs, M=seq.apply_dd0_precond,
               tol=seq.tol, maxiter=seq.maxiter)[0]
    jax.block_until_ready(u_hat)
    timings['cg solve'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    u_h = DiscreteFunction(u_hat, seq.basis_0, seq.e0)

    def diff(x):
        return u(x) - u_h(x)

    df = jax.lax.map(diff, seq.quad.x, batch_size=mrx.MAP_BATCH_SIZE_OUTER)
    L2_df = jnp.einsum('ik,ik,i,i->', df, df, seq.jacobian_j, seq.quad.w)
    u_i = jax.lax.map(u, seq.quad.x, batch_size=mrx.MAP_BATCH_SIZE_INNER)
    L2_f = jnp.einsum('ik,ik,i,i->', 
                      u_i,
                      u_i, 
                      seq.jacobian_j, 
                      seq.quad.w)
    jax.block_until_ready(L2_f)
    timings['error_computation'] = time.perf_counter() - t0

    error = (L2_df / L2_f)**0.5

    total = sum(timings.values())
    timings['TOTAL'] = total
    print(f"  --- Timings (n={n}, p={p}) ---")
    for label, dt in timings.items():
        print(f"  {label:.<30s} {dt:8.3f}s")

    return error


# %%
# Convergence sweep across n and p
ps_sweep = [1, 2, 3, 4]
ns_sweep = [8, 16, 32]
results = {}  # results[p] = list of (n, error)
# %%
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
    ax.loglog(n_ref, ref, "k--", linewidth=0.6, alpha=1.0)
    ax.annotate(f"$n^{{{slope}}}$",
                xy=(float(n_ref[0]), float(ref[0])),
                fontsize=8, color="black",
                textcoords="offset points", xytext=(4, 4))

ax.set_xlabel("$n$")
ax.set_ylabel("Relative $L^2$ error")
ax.set_title("Sparse solve (CG)")
ax.legend()
ax.grid(True, which="both", linewidth=0.3)
fig.tight_layout()
# plt.savefig("convergence_poisson_torus_sparse.png", dpi=200)
plt.show()
# %%
