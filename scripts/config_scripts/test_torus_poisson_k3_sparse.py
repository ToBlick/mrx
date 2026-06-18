"""Convergence study for the k=3 Hodge–Laplacian on a toroidal domain.

Solves  -Δ₃ ω = f  with natural boundary conditions (NBC), where
f and u use the same scalar functions as the k=0 DBC problem.

k=3 NBC is the Hodge dual of k=0 DBC. On a solid torus with
betti_numbers=(1,1,0,0), k=3 NBC has no nullspace (b₃=0).

Preconditioner: tensor mass for the lower block (k=2), tensor Schur inner,
pre-probed Jacobi Schur outer (assemble_schur_jacobi_preconditioner).

Usage (run from the repo root):
    # Single run — loops over all n values for the given p
    python scripts/config_scripts/test_torus_poisson_k3_sparse.py p=3

    # Multirun sweep — one SLURM job per (p, n) pair
    python scripts/config_scripts/test_torus_poisson_k3_sparse.py -m p=1,2,3,4 n=8,12,16,24,32
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
from mrx.nullspace import init_nullspaces
from mrx.operators import (
    assemble_incidence_operators,
    assemble_projection_operators,
    assemble_schur_jacobi_preconditioner,
    assemble_tensor_mass_preconditioner,
)
from mrx.quadrature import evaluate_at_xq

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Problem setup
# ---------------------------------------------------------------------------
types = ("clamped", "periodic", "periodic")
π = jnp.pi

# k=3 NBC, betti=(1,1,0,0): no nullspace (b₃=0 for NBC 3-forms).
BETTI = (1, 1, 0, 0)


def u_exact(x: jnp.ndarray) -> jnp.ndarray:
    """Exact scalar solution: u(r,χ,z) = 1/4 (r²-r⁴) cos(2πz)."""
    r, _chi, z = x
    return 1 / 4 * (r ** 2 - r ** 4) * jnp.cos(2 * π * z) * jnp.ones(1)


def make_f(a: float):
    """Return the manufactured source term for minor radius a."""
    def f(x: jnp.ndarray) -> jnp.ndarray:
        r, chi, z = x
        R = 1.0 + a * r * jnp.cos(2 * π * chi)
        return (
            jnp.cos(2 * π * z)
            * (
                -1.0 / a ** 2 * (1.0 - 4.0 * r ** 2)
                - 1.0 / (a * R) * (r / 2.0 - r ** 3) * jnp.cos(2 * π * chi)
                + 1.0 / 4.0 * (r ** 2 - r ** 4) / R ** 2
            )
            * jnp.ones(1)
        )
    return f


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------
def compute_error(n: int, p: int, epsilon: float,
                  cg_tol: float, cg_maxiter: int,
                  quad_order: int | None,
                  quad_order_offset: int):
    """Run the k=3 DBC saddle-point solve and return (error, timings dict)."""
    timings = {}
    ns = (n, 2 * n, n)
    ps = (p, p, p)
    q = 2 * p + quad_order_offset if quad_order is None else quad_order
    if q < 2 * p:
        raise ValueError(f"quad_order must satisfy q >= 2*p; got q={q}, p={p}")

    F = toroid_map(epsilon=epsilon)
    f = make_f(epsilon)

    t0 = time.perf_counter()
    seq = DeRhamSequence(
        ns, ps, q, types, polar=True,
        tol=cg_tol, maxiter=cg_maxiter,
        betti_numbers=BETTI,
    )
    seq.set_map(F)
    timings["DeRhamSequence.__init__"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    seq.evaluate_1d()
    timings["evaluate_1d"] = time.perf_counter() - t0

    # ---------------------------------------------------------------------------
    # Assembly: tensor mass (k=2,3) + incidence/projection + Schur Jacobi + nullspace
    # ---------------------------------------------------------------------------
    cp_kwargs = {"maxiter": 100, "tol": 1e-9, "ridge": 1e-12}

    t0 = time.perf_counter()
    ops = assemble_incidence_operators(seq)
    ops = assemble_projection_operators(seq, operators=ops)
    # Tensor mass for k=2 (Schur inner) and k=3 (lower block).
    ops = assemble_tensor_mass_preconditioner(seq, ops, ks=(2, 3), rank=1, cp_kwargs=cp_kwargs)
    # Pre-probe the Schur diagonal (D M2_tensor^{-1} D^T) for k=3 NBC.
    ops = assemble_schur_jacobi_preconditioner(seq, ops, ks=(3,), dirichlet_variants=(False,))
    # No nullspace: b₃=0 for k=3 NBC on the solid torus.
    ops = init_nullspaces(seq, ops, BETTI)
    ops = seq.set_operators(ops)
    jax.block_until_ready(ops)
    timings["assembly_compile"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    ops = assemble_incidence_operators(seq)
    ops = assemble_projection_operators(seq, operators=ops)
    ops = assemble_tensor_mass_preconditioner(seq, ops, ks=(2, 3), rank=1, cp_kwargs=cp_kwargs)
    ops = assemble_schur_jacobi_preconditioner(seq, ops, ks=(3,), dirichlet_variants=(False,))
    ops = init_nullspaces(seq, ops, BETTI)
    ops = seq.set_operators(ops)
    jax.block_until_ready(ops)
    timings["assembly_exec"] = time.perf_counter() - t0

    # ---------------------------------------------------------------------------
    # Solve
    # ---------------------------------------------------------------------------
    t0 = time.perf_counter()
    rhs = seq.load(f, 3, dirichlet=False)
    jax.block_until_ready(rhs)
    timings["load_rhs"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    u_hat, info = seq.apply_inverse_laplacian(rhs, 3, dirichlet=False, return_info=True)
    jax.block_until_ready(u_hat)
    timings["solve_compile"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    u_hat, info = seq.apply_inverse_laplacian(rhs, 3, dirichlet=False, return_info=True)
    jax.block_until_ready(u_hat)
    timings["solve_exec"] = time.perf_counter() - t0

    info_int = int(info)
    iters = abs(info_int)
    converged = info_int < 0

    # ---------------------------------------------------------------------------
    # Error: physical L2 via evaluate_at_xq + k=3 pushforward (divide by det(DF)).
    # ---------------------------------------------------------------------------
    t0 = time.perf_counter()
    comp_info, comp_shapes = seq._form_comp_info(3)
    quad_shape = (seq.quad.ny, seq.quad.nx, seq.quad.nz)
    u_h_jk = evaluate_at_xq(
        seq.e3_T @ u_hat, comp_info, comp_shapes, quad_shape, 1)
    # k=3 pushforward: reference value / det(DF)
    u_h_phys = u_h_jk / seq.jacobian_j[:, None]
    u_exact_phys = jax.vmap(u_exact)(seq.quad.x)
    diff = u_h_phys - u_exact_phys
    L2_diff = jnp.einsum("ik,ik,i,i->", diff, diff, seq.jacobian_j, seq.quad.w)
    L2_ref  = jnp.einsum("ik,ik,i,i->", u_exact_phys, u_exact_phys, seq.jacobian_j, seq.quad.w)
    jax.block_until_ready(L2_ref)
    timings["error_computation"] = time.perf_counter() - t0

    error = float(jnp.sqrt(L2_diff / L2_ref))
    timings["TOTAL"] = sum(timings.values())

    return {
        "n": n, "p": p, "q": q,
        "error": error,
        "iters": iters,
        "converged": converged,
        "timings": timings,
    }


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------
@hydra.main(config_path="../../conf", config_name="config_poisson_test", version_base=None)
def main(cfg: DictConfig):
    print(f"x64 enabled: {jax.config.jax_enable_x64}")
    ns = [cfg.n] if isinstance(cfg.n, int) else list(cfg.n)
    p = cfg.p
    mrx.MAP_BATCH_SIZE_INNER = cfg.map_batch_size_inner
    mrx.MAP_BATCH_SIZE_OUTER = cfg.map_batch_size_outer
    print(f"Running k=3 Poisson (NBC) convergence: n={ns}, p={p}, epsilon={cfg.epsilon}")
    print(f"JAX devices: {jax.devices()}")

    output_dir = HydraConfig.get().runtime.output_dir
    outfile = os.path.join(output_dir, "result.json")

    results = []
    for n in ns:
        print(f"\n{'='*60}\n  n={n}, p={p}\n{'='*60}")
        result = compute_error(
            n, p, cfg.epsilon, cfg.cg_tol, cfg.cg_maxiter,
            cfg.quad_order, cfg.quad_order_offset,
        )
        results.append(result)
        print(f"\n  --- Timings (n={n}, p={p}) ---")
        for label, dt in result["timings"].items():
            print(f"  {label:.<30s} {dt:8.3f}s")
        print(f"\n  Relative L2 error: {result['error']:.6e}")
        print(f"  MINRES iters: {result['iters']}  converged: {result['converged']}")
        with open(outfile, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"  Results saved to {outfile}")

    print(f"\n{'='*60}\n  Summary (p={p})\n{'='*60}")
    print(f"  {'n':>5s}  {'error':>12s}  {'total_time':>10s}")
    for r in results:
        print(f"  {r['n']:5d}  {r['error']:12.6e}  {r['timings']['TOTAL']:10.3f}s")


if __name__ == "__main__":
    main()
