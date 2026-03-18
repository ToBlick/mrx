"""Solve the Poisson equation on a toroidal domain with sparse assembly.
Stores the relative L2 error and all timings to a JSON file.

Usage (run from the repo root on a login node):
    # Single run (local, no SLURM) — loops over all n values for the given p
    python scripts/config_scripts/test_torus_poisson_sparse.py p=3

    # Multirun sweep — one SLURM job per p, each loops over all n values
    python scripts/config_scripts/test_torus_poisson_sparse.py -m p=1,2,3,4

    # Override the n list
    python scripts/config_scripts/test_torus_poisson_sparse.py 'n=[8,16,32,64]' p=2
"""
import json
import os
import time

import hydra
import jax
import jax.numpy as jnp
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

import mrx
import mrx.config  # noqa: F401 — register structured configs in ConfigStore
from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction
from mrx.mappings import toroid_map

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
        ns, ps, q, types, F, polar=True,
        tol=cg_tol, maxiter=cg_maxiter,
    )
    timings["DeRhamSequence.__init__"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    seq.evaluate_1d()
    timings["evaluate_1d"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    seq.assemble_mass_matrix(0)
    timings["assemble_mass_matrix_0"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    seq.assemble_hodge_laplacian(0)
    timings["assemble_hodge_laplacian_0"] = time.perf_counter() - t0

    # Sparsity diagnostics
    sparsity = {}
    for name, mat in [("m0", seq.m0_sp), ("dd0", seq.grad_grad_sp)]:
        nnz_stored = int(mat.nse)
        nnz_actual = int(jnp.sum(jnp.abs(mat.data) > 1e-12))
        sparsity[f"{name}_nnz_stored"] = nnz_stored
        sparsity[f"{name}_nnz_actual"] = nnz_actual

    t0 = time.perf_counter()
    rhs = seq.p0_dbc(f)
    jax.block_until_ready(rhs)
    timings["P0_dbc(f)"] = time.perf_counter() - t0

    # k=0 with DBC has no nullspace; skip expensive compute_nullspaces()
    seq.null_0_dbc = []

    t0 = time.perf_counter()
    u_hat = seq.apply_inverse_hodge_laplacian(rhs, 0, dirichlet=True)
    jax.block_until_ready(u_hat)
    timings["inverse_hodge_laplacian"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    u_h = DiscreteFunction(u_hat, seq.basis_0, seq.e0_dbc)

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
    return {"n": n, "p": p, "error": error, "timings": timings, "sparsity": sparsity}


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------
@hydra.main(config_path="../../conf", config_name="config_poisson_test", version_base=None)
def main(cfg: DictConfig):
    p = cfg.p
    ns = list(cfg.n)
    mrx.MAP_BATCH_SIZE_INNER = cfg.map_batch_size_inner
    mrx.MAP_BATCH_SIZE_OUTER = cfg.map_batch_size_outer
    print(f"Running sparse Poisson solve: n={ns}, p={p}")
    print(f"JAX devices: {jax.devices()}")
    print(
        f"Batch sizes: inner={mrx.MAP_BATCH_SIZE_INNER}, outer={mrx.MAP_BATCH_SIZE_OUTER}")

    results = []
    for n in ns:
        print(f"\n{'='*60}")
        print(f"  n={n}, p={p}")
        print(f"{'='*60}")

        result = compute_error(
            n, p, cfg.epsilon, cfg.cg_tol, cfg.cg_maxiter,
        )
        results.append(result)

        print(f"\n  --- Timings (n={n}, p={p}) ---")
        for label, dt in result["timings"].items():
            print(f"  {label:.<30s} {dt:8.3f}s")
        print("\n  --- Sparsity ---")
        for label, val in result["sparsity"].items():
            print(f"  {label:.<30s} {val}")
        print(f"\n  Relative L2 error: {result['error']:.6e}")

    # Summary table
    print(f"\n{'='*60}")
    print(f"  Summary (p={p})")
    print(f"{'='*60}")
    print(f"  {'n':>5s}  {'error':>12s}  {'total_time':>10s}")
    for r in results:
        print(
            f"  {r['n']:5d}  {r['error']:12.6e}  {r['timings']['TOTAL']:10.3f}s")

    # Save results into the Hydra output directory
    output_dir = HydraConfig.get().runtime.output_dir
    outfile = os.path.join(output_dir, "result.json")
    with open(outfile, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n  Results saved to {outfile}")


if __name__ == "__main__":
    main()
