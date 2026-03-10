"""Solve the Poisson equation on a toroidal domain with sparse assembly and CG.
Stores the relative L2 error and all timings to a JSON file.

Usage (run from the repo root on a login node):
    # Single run (local, no SLURM)
    python scripts/config_scripts/test_torus_poisson_sparse.py n=16 p=3

    # Multirun sweep — each (n,p) submitted as a separate SLURM job via submitit
    python scripts/config_scripts/test_torus_poisson_sparse.py -m n=8,16,32 p=1,2,3,4
"""
import json
import os
import time

import hydra
import jax
import jax.numpy as jnp
from hydra.core.hydra_config import HydraConfig
from jax.scipy.sparse.linalg import cg
from omegaconf import DictConfig

import mrx
import mrx.config  # noqa: F401 — register structured configs in ConfigStore
from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction
from mrx.mappings import toroid_map
from mrx.utils import solve_singular_cg

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Problem setup
# ---------------------------------------------------------------------------
types = ("clamped", "periodic", "periodic")
π = jnp.pi


def u(x: jnp.ndarray) -> jnp.ndarray:
    """Exact solution: u(r,χ,z) = 1/4 (r² - r⁴) cos(2πz)."""
    r, χ, z = x
    return 1 / 4 * (r**2 - r**4) * jnp.cos(2 * π * z) * jnp.ones(1)


def make_f(a: float):
    """Return the source term for minor radius *a*."""
    def f(x: jnp.ndarray) -> jnp.ndarray:
        r, χ, z = x
        R = 1 + a * r * jnp.cos(2 * jnp.pi * χ)
        return (
            jnp.cos(2 * jnp.pi * z)
            * (
                -1 / a**2 * (1 - 4 * r**2)
                - 1 / (a * R) * (r / 2 - r**3) * jnp.cos(2 * jnp.pi * χ)
                + 1 / 4 * (r**2 - r**4) / R**2
            )
            * jnp.ones(1)
        )
    return f


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------
def compute_error(n: int, p: int, epsilon: float,
                  cg_tol: float, cg_maxiter: int):
    """Run the sparse Poisson solve and return (error, timings dict)."""
    timings = {}
    ns = (n, n, n)
    ps = (p, p, p)
    q = 2*p
    F = toroid_map(epsilon=epsilon)
    f = make_f(epsilon)

    t0 = time.perf_counter()
    seq = DeRhamSequence(
        ns, ps, q, types, F, polar=True, dirichlet=True,
        tol=cg_tol, maxiter=cg_maxiter,
    )
    timings["DeRhamSequence.__init__"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    seq.evaluate_1d()
    timings["evaluate_1d"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    seq.assemble_m0_sparse()
    timings["assemble_m0_sparse"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    seq.assemble_dd0_sparse()
    timings["assemble_dd0_sparse"] = time.perf_counter() - t0

    # Sparsity diagnostics
    sparsity = {}
    for name, mat in [("m0", seq.m0_sp), ("dd0", seq.grad_grad_sp)]:
        nnz_stored = int(mat.nse)
        nnz_actual = int(jnp.sum(jnp.abs(mat.data) > 1e-12))
        sparsity[f"{name}_nnz_stored"] = nnz_stored
        sparsity[f"{name}_nnz_actual"] = nnz_actual

    t0 = time.perf_counter()
    rhs = seq.p0(f)
    jax.block_until_ready(rhs)
    timings["P0(f)"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    u_hat = solve_singular_cg(
        seq.apply_dd0_sparse, 
        rhs, 
        mass_matvec=seq.apply_m0_sparse,
        precond_matvec=seq.apply_dd0_precond,
        tol=seq.tol, 
        maxiter=seq.maxiter,
    )[0]
    jax.block_until_ready(u_hat)
    timings["cg_solve"] = time.perf_counter() - t0
    
    # Save the conditioning number of the preconditioned operator for diagnostics
    # laplace_dense = seq.e0 @ seq.grad_grad_sp.todense() @ seq.e0.T
    # u_hat_dense = jnp.linalg.solve(laplace_dense, rhs)
    # cond = jnp.linalg.cond(laplace_dense)
    # cond_precond = jnp.linalg.cond(jnp.diag(seq.dd0_sp_diaginv) @ laplace_dense)
    cond = 1.0
    cond_precond = 1.0

    t0 = time.perf_counter()
    u_h = DiscreteFunction(u_hat, seq.basis_0, seq.e0)

    def diff(x):
        return u(x) - u_h(x)

    df = jax.lax.map(diff, seq.quad.x, batch_size=mrx.MAP_BATCH_SIZE_OUTER)
    L2_df = jnp.einsum("ik,ik,i,i->", df, df, seq.jacobian_j, seq.quad.w)
    u_i = jax.lax.map(u, seq.quad.x, batch_size=mrx.MAP_BATCH_SIZE_INNER)
    L2_f = jnp.einsum("ik,ik,i,i->", u_i, u_i, seq.jacobian_j, seq.quad.w)
    jax.block_until_ready(L2_f)
    timings["error_computation"] = time.perf_counter() - t0

    error = float((L2_df / L2_f) ** 0.5)

    timings["TOTAL"] = sum(timings.values())
    return {"n": n, "p": p, "error": error, "timings": timings, "sparsity": sparsity, "cond": float(cond), "cond_precond": float(cond_precond)} 


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------
@hydra.main(config_path="../../conf", config_name="config_poisson_test", version_base=None)
def main(cfg: DictConfig):
    n, p = cfg.n, cfg.p
    mrx.MAP_BATCH_SIZE_INNER = cfg.map_batch_size_inner
    mrx.MAP_BATCH_SIZE_OUTER = cfg.map_batch_size_outer
    print(f"Running sparse Poisson solve: n={n}, p={p}")
    print(f"JAX devices: {jax.devices()}")
    print(f"Batch sizes: inner={mrx.MAP_BATCH_SIZE_INNER}, outer={mrx.MAP_BATCH_SIZE_OUTER}")

    result = compute_error(
        n, p, cfg.epsilon, cfg.cg_tol, cfg.cg_maxiter,
    )

    print(f"\n  --- Timings (n={n}, p={p}) ---")
    for label, dt in result["timings"].items():
        print(f"  {label:.<30s} {dt:8.3f}s")
    print("\n  --- Sparsity ---")
    for label, val in result["sparsity"].items():
        print(f"  {label:.<30s} {val}")
    print(f"\n  Relative L2 error: {result['error']:.6e}")

    # Save results into the Hydra output directory
    output_dir = HydraConfig.get().runtime.output_dir
    outfile = os.path.join(output_dir, "result.json")
    with open(outfile, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"  Results saved to {outfile}")


if __name__ == "__main__":
    main()
