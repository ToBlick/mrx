"""Solve the Poisson equation on a toroidal domain with sparse assembly.
Stores the relative L2 error and all timings to a JSON file.

Usage (run from the repo root on a login node):
    # Single run (local, no SLURM) — loops over all n values for the given p
    python scripts/config_scripts/test_torus_poisson_sparse.py p=3

    # Multirun sweep — one SLURM job per (p, n) pair, each with a clean GPU
    python scripts/config_scripts/test_torus_poisson_sparse.py -m p=1,2,3,4 n=8,12,16,24,32,48,64

    # Override the n list (single job, loops over all n)
    python scripts/config_scripts/test_torus_poisson_sparse.py 'n=[8,12,16,24,32,48,64]' p=2
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
from mrx.mappings import toroid_map
from mrx.operators import assemble_tensor_laplacian_preconditioner
from mrx.quadrature import evaluate_at_xq

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


def exact_u_at_quad(seq: DeRhamSequence) -> jnp.ndarray:
    """Evaluate the exact scalar solution on the quadrature grid cheaply."""
    u_r = 0.25 * (seq.quad.x_x**2 - seq.quad.x_x**4)
    u_z = jnp.cos(2 * π * seq.quad.x_z)
    values = jnp.ones((seq.quad.ny, 1, 1)) * \
        u_r[None, :, None] * u_z[None, None, :]
    return values.reshape(-1, 1)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------
def compute_error(n: int, p: int, epsilon: float,
                  cg_tol: float, cg_maxiter: int,
                  quad_order: int | None,
                  quad_order_offset: int):
    """Run the sparse Poisson solve and return (error, timings dict).

    Resolution convention: ``ns = (n, 2*n, n)`` (the toroidal direction
    carries twice the angular resolution of the radial / vertical
    directions).
    """
    timings = {}
    ns = (n, 2 * n, n)
    ps = (p, p, p)
    q = 2 * p + quad_order_offset if quad_order is None else quad_order
    if q < 2 * p:
        raise ValueError(
            f"quad_order must satisfy q >= 2*p; got q={q}, p={p}"
        )
    F = toroid_map(epsilon=epsilon)
    f = make_f(epsilon)

    t0 = time.perf_counter()
    seq = DeRhamSequence(
        ns, ps, q, types, polar=True,
        tol=cg_tol, maxiter=cg_maxiter,
    )
    seq.set_map(F)
    timings["DeRhamSequence.__init__"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    seq.evaluate_1d()
    timings["evaluate_1d"] = time.perf_counter() - t0

    # Each assembly / solve is timed twice: the first call includes XLA
    # compilation (``_compile``), the second is execution-only on the cached
    # compiled kernels (``_exec``).

    # K0 = G0^T M1 G0 applies M1 matrix-free (sum factorization); M1 is never
    # assembled or stored. The k=0 tensor Hodge-Laplacian preconditioner below
    # is the ONLY preconditioner built for the solve. Its apply uses only the
    # matrix-free stiffness ``apply_stiffness`` plus the bulk FD / Schur
    # factors it assembles itself; it does NOT consume the tensor *mass*
    # surgery/block data. The tensor mass preconditioner is therefore not
    # assembled here -- the CP fit configuration (rank + CP solver tolerances)
    # is passed directly so the Laplacian builder is self-contained.
    cp_kwargs = {"maxiter": 100, "tol": 1e-9, "ridge": 1e-12}

    # Build the tensor Hodge-Laplacian preconditioner explicitly. This is the
    # ONLY preconditioner built for the solve below: the tensor path does not
    # need (and does not compute) any Jacobi diagonal. `assemble_hodge_laplacian`
    # is intentionally not called here -- operator assembly is decoupled from
    # preconditioner construction.
    t0 = time.perf_counter()
    ops = seq.set_operators(
        assemble_tensor_laplacian_preconditioner(
            seq, seq.get_operators(), ks=(0,), rank=1, cp_kwargs=cp_kwargs,
        )
    )
    jax.block_until_ready(ops)
    timings["build_hodge_preconditioners_0_compile"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    ops = seq.set_operators(
        assemble_tensor_laplacian_preconditioner(
            seq, seq.get_operators(), ks=(0,), rank=1, cp_kwargs=cp_kwargs,
        )
    )
    jax.block_until_ready(ops)
    timings["build_hodge_preconditioners_0_exec"] = time.perf_counter() - t0

    # Sparsity diagnostics. K0 = G0^T M1 G0 is never materialised (the solve
    # applies K via composed matvecs, with M1 applied matrix-free), and M0 is
    # also applied matrix-free, so no mass matrix is stored.
    sparsity = {}

    t0 = time.perf_counter()
    rhs = seq.load(f, 0, dirichlet=True)
    jax.block_until_ready(rhs)
    timings["P0_dbc(f)"] = time.perf_counter() - t0

    # k=0 with DBC has no nullspace (default betti_numbers=(1,1,0,0))
    t0 = time.perf_counter()
    u_hat, cg_info = seq.apply_inverse_hodge_laplacian(
        rhs, 0, dirichlet=True, return_info=True)
    jax.block_until_ready(u_hat)
    timings["inverse_hodge_laplacian_compile"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    u_hat, cg_info = seq.apply_inverse_hodge_laplacian(
        rhs, 0, dirichlet=True, return_info=True)
    jax.block_until_ready(u_hat)
    timings["inverse_hodge_laplacian_exec"] = time.perf_counter() - t0

    cg_info_int = int(cg_info)
    cg_iters = abs(cg_info_int)
    cg_converged = cg_info_int < 0
    residual = seq.apply_hodge_laplacian(u_hat, 0, dirichlet=True) - rhs
    final_rel_residual = float(
        jnp.linalg.norm(residual) / jnp.linalg.norm(rhs))

    t0 = time.perf_counter()
    quad_shape = (seq.quad.ny, seq.quad.nx, seq.quad.nz)
    comp_info_0, comp_shapes_0 = seq._form_comp_info(0)
    u_h_jk = evaluate_at_xq(seq.e0_dbc_T @ u_hat, comp_info_0, comp_shapes_0,
                            quad_shape, 1)
    u_i = exact_u_at_quad(seq)
    df = u_i - u_h_jk
    L2_df = jnp.einsum("ik,ik,i,i->", df, df, seq.jacobian_j, seq.quad.w)
    L2_f = jnp.einsum("ik,ik,i,i->", u_i, u_i, seq.jacobian_j, seq.quad.w)
    jax.block_until_ready(L2_f)
    timings["error_computation"] = time.perf_counter() - t0

    error = float((L2_df / L2_f) ** 0.5)

    timings["TOTAL"] = sum(timings.values())
    return {
        "n": n,
        "p": p,
        "q": q,
        "error": error,
        "cg_iters": cg_iters,
        "cg_converged": cg_converged,
        "final_rel_residual": final_rel_residual,
        "timings": timings,
        "sparsity": sparsity,
    }


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------
@hydra.main(config_path="../../conf", config_name="config_poisson_test", version_base=None)
def main(cfg: DictConfig):
    
    print(f"x64 enabled: {jax.config.jax_enable_x64}")
    print(f"epsilon type: {type(cfg.epsilon).__name__} value: {cfg.epsilon!r}")
    ns = [cfg.n] if isinstance(cfg.n, int) else list(cfg.n)
    print(f"n: {ns!r}")
    print(f"cg_tol type: {type(cfg.cg_tol).__name__} value: {cfg.cg_tol!r}")

    p = cfg.p
    mrx.MAP_BATCH_SIZE_INNER = cfg.map_batch_size_inner
    mrx.MAP_BATCH_SIZE_OUTER = cfg.map_batch_size_outer
    print(f"Running sparse Poisson solve: n={ns}, p={p}")
    if cfg.quad_order is None:
        print(f"Quadrature order: q = 2*p + {cfg.quad_order_offset}")
    else:
        print(f"Quadrature order: q = {cfg.quad_order}")
    print(f"JAX devices: {jax.devices()}")
    print(
        f"Batch sizes: inner={mrx.MAP_BATCH_SIZE_INNER}, outer={mrx.MAP_BATCH_SIZE_OUTER}")

    output_dir = HydraConfig.get().runtime.output_dir
    outfile = os.path.join(output_dir, "result.json")

    results = []
    for n in ns:
        print(f"\n{'='*60}")
        print(f"  n={n}, p={p}")
        print(f"{'='*60}")

        result = compute_error(
            n,
            p,
            cfg.epsilon,
            cfg.cg_tol,
            cfg.cg_maxiter,
            cfg.quad_order,
            cfg.quad_order_offset,
        )
        results.append(result)

        print(f"\n  --- Timings (n={n}, p={p}) ---")
        for label, dt in result["timings"].items():
            print(f"  {label:.<30s} {dt:8.3f}s")
        print("\n  --- Sparsity ---")
        for label, val in result["sparsity"].items():
            print(f"  {label:.<30s} {val}")
        print(f"\n  Relative L2 error: {result['error']:.6e}")
        print(f"  CG iters: {result['cg_iters']}  converged: {result['cg_converged']}"
              f"  final ||K0 u - b||/||b||: {result['final_rel_residual']:.3e}")

        # Write incrementally so results are not lost if a later n OOMs
        with open(outfile, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"  Results saved to {outfile}")

    # Summary table
    print(f"\n{'='*60}")
    print(f"  Summary (p={p})")
    print(f"{'='*60}")
    print(f"  {'n':>5s}  {'error':>12s}  {'total_time':>10s}")
    for r in results:
        print(
            f"  {r['n']:5d}  {r['error']:12.6e}  {r['timings']['TOTAL']:10.3f}s")


if __name__ == "__main__":
    main()
