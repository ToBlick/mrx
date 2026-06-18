"""Convergence study for the k=0 Hodge–Laplacian with NBC on a toroidal domain.

Exact solution:  u₀ = cos(2πζ)
Source:          f₀ = cos(2πζ) / R²   (= -Δ₀ u₀ on the torus)

k=0 NBC has a 1-dimensional harmonic nullspace spanned by the constant
function.  Deflation is applied via the saddle-point MINRES solve.

Supports both frame='ref' (pass scalar f₀ directly) and frame='phys'
(identical for k=0 scalars, included for API consistency testing).

Diagnostics logged per run:
  - Relative L² error of the scalar solution
  - MINRES iteration count and convergence flag
  - ||D₀ h||₂  (curl of the nullspace constant — should be ≈ 0)
  - Nullspace vector residual  ||L₀ h||₂

Usage (from repo root):
    python scripts/config_scripts/test_torus_poisson_nbc_k0_sparse.py -m p=1,2,3 n=8,12,16,20
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
from mrx.nullspace import get_nullspace, init_nullspaces, _set_null
from mrx.operators import (
    assemble_incidence_operators,
    assemble_projection_operators,
    assemble_tensor_laplacian_preconditioner,
    assemble_tensor_mass_preconditioner,
)
from mrx.quadrature import evaluate_at_xq

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Problem constants
# ---------------------------------------------------------------------------
types = ("clamped", "periodic", "periodic")
π = jnp.pi
BETTI = (1, 1, 0, 0)
K = 0
DIRICHLET = False


# ---------------------------------------------------------------------------
# Source and exact-solution functions
# ---------------------------------------------------------------------------
def make_f0(a: float):
    """Source f₀ = cos(2πζ)/R²; identical in 'ref' and 'phys' frames (k=0 scalar)."""
    def f(x):
        r, chi, z = x
        R = 1.0 + a * r * jnp.cos(2 * π * chi)
        return jnp.cos(2 * π * z) / R**2 * jnp.ones(1)
    return f


def u0_exact(x):
    """Exact scalar solution: cos(2πζ)."""
    return jnp.cos(2 * π * x[2]) * jnp.ones(1)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------
def compute_error(n: int, p: int, epsilon: float,
                  cg_tol: float, cg_maxiter: int,
                  quad_order, quad_order_offset: int,
                  load_frame: str):
    timings = {}
    ns = (n, 2 * n, n)
    ps = (p, p, p)
    q = 2 * p + quad_order_offset if quad_order is None else quad_order

    F = toroid_map(epsilon=epsilon)
    f0 = make_f0(epsilon)
    cp_kwargs = {"maxiter": 100, "tol": 1e-9, "ridge": 1e-12}

    # --- Sequence setup ------------------------------------------------
    t0 = time.perf_counter()
    seq = DeRhamSequence(
        ns, ps, q, types, polar=True,
        tol=cg_tol, maxiter=cg_maxiter,
        betti_numbers=BETTI,
    )
    seq.set_map(F)
    timings["init"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    seq.evaluate_1d()
    timings["evaluate_1d"] = time.perf_counter() - t0

    # --- Assembly (compile pass) ----------------------------------------
    t0 = time.perf_counter()
    ops = assemble_incidence_operators(seq)
    ops = assemble_projection_operators(seq, operators=ops)
    # Tensor mass k=0 (for l2_norm/apply_mass in nullspace bootstrap)
    # and k=1 (needed inside compute_nullspaces_iterative for null_1 NBC).
    ops = assemble_tensor_mass_preconditioner(seq, ops, ks=(0, 1), rank=1, cp_kwargs=cp_kwargs)
    # Tensor Hodge-Laplacian preconditioner for the k=0 solve.
    ops = assemble_tensor_laplacian_preconditioner(seq, ops, ks=(0,), rank=1, cp_kwargs=cp_kwargs)
    ops = seq.set_operators(ops)
    jax.block_until_ready(ops)
    timings["assembly_compile"] = time.perf_counter() - t0

    # --- Assembly (exec pass) ------------------------------------------
    t0 = time.perf_counter()
    ops = assemble_incidence_operators(seq)
    ops = assemble_projection_operators(seq, operators=ops)
    ops = assemble_tensor_mass_preconditioner(seq, ops, ks=(0, 1), rank=1, cp_kwargs=cp_kwargs)
    ops = assemble_tensor_laplacian_preconditioner(seq, ops, ks=(0,), rank=1, cp_kwargs=cp_kwargs)
    ops = seq.set_operators(ops)
    jax.block_until_ready(ops)
    timings["assembly_exec"] = time.perf_counter() - t0

    # --- Nullspace: k=0 NBC constant function (set analytically) -------
    t0 = time.perf_counter()
    ops = init_nullspaces(seq, seq.get_operators(), BETTI)
    const_vec = jnp.ones(seq.n0)
    norm = seq.l2_norm(const_vec, 0, dirichlet=False)
    ops = _set_null(ops, 0, False, (const_vec / norm)[None, :])
    ops = seq.set_operators(ops)
    jax.block_until_ready(ops)
    # Residual diagnostic: ||L₀ h||
    h = get_nullspace(seq.get_operators(), K, DIRICHLET)[0]
    Lh = seq.apply_laplacian(h, K, dirichlet=DIRICHLET)
    null_residual = float(jnp.linalg.norm(Lh))
    # Curl diagnostic: ||D₀ h|| — should be 0 for constant function
    curl = seq.apply_derivative_matrix(h, K, dirichlet_in=DIRICHLET, dirichlet_out=DIRICHLET)
    null_curl_norm = float(jnp.linalg.norm(curl))
    timings["nullspace"] = time.perf_counter() - t0

    # --- RHS -----------------------------------------------------------
    t0 = time.perf_counter()
    rhs = seq.load(f0, K, dirichlet=DIRICHLET, frame=load_frame)
    jax.block_until_ready(rhs)
    timings["load_rhs"] = time.perf_counter() - t0

    # --- Solve (compile + exec) ----------------------------------------
    t0 = time.perf_counter()
    u_hat, info = seq.apply_inverse_laplacian(
        rhs, K, dirichlet=DIRICHLET, return_info=True)
    jax.block_until_ready(u_hat)
    timings["solve_compile"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    u_hat, info = seq.apply_inverse_laplacian(
        rhs, K, dirichlet=DIRICHLET, return_info=True)
    jax.block_until_ready(u_hat)
    timings["solve_exec"] = time.perf_counter() - t0

    iters = abs(int(info))
    converged = int(info) < 0

    # --- Error ---------------------------------------------------------
    t0 = time.perf_counter()
    comp_info, comp_shapes = seq._form_comp_info(K)
    quad_shape = (seq.quad.ny, seq.quad.nx, seq.quad.nz)
    u_h_jk = evaluate_at_xq(seq.e0_T @ u_hat, comp_info, comp_shapes, quad_shape, 1)
    u_ex = jax.vmap(u0_exact)(seq.quad.x)
    diff = u_h_jk - u_ex
    L2_diff = jnp.einsum("ik,ik,i,i->", diff, diff, seq.jacobian_j, seq.quad.w)
    L2_norm = jnp.einsum("ik,ik,i,i->", u_ex, u_ex, seq.jacobian_j, seq.quad.w)
    jax.block_until_ready(L2_norm)
    timings["error"] = time.perf_counter() - t0

    error = float(jnp.sqrt(L2_diff / L2_norm))
    timings["TOTAL"] = sum(timings.values())

    return {
        "n": n, "p": p, "q": q,
        "error": error,
        "iters": iters,
        "converged": converged,
        "null_residual": null_residual,
        "null_curl_norm": null_curl_norm,
        "load_frame": load_frame,
        "timings": timings,
    }


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------
@hydra.main(config_path="../../conf", config_name="config_poisson_test", version_base=None)
def main(cfg: DictConfig):
    print(f"x64 enabled: {jax.config.jax_enable_x64}")
    ns = [cfg.n] if isinstance(cfg.n, int) else list(cfg.n)
    p, load_frame = cfg.p, cfg.load_frame
    mrx.MAP_BATCH_SIZE_INNER = cfg.map_batch_size_inner
    mrx.MAP_BATCH_SIZE_OUTER = cfg.map_batch_size_outer
    print(f"k=0 NBC Poisson | frame={load_frame} | n={ns} p={p} ε={cfg.epsilon}")
    print(f"JAX devices: {jax.devices()}")

    output_dir = HydraConfig.get().runtime.output_dir
    outfile = os.path.join(output_dir, "result.json")

    results = []
    for n in ns:
        print(f"\n{'='*60}\n  n={n}, p={p}\n{'='*60}")
        result = compute_error(
            n, p, cfg.epsilon, cfg.cg_tol, cfg.cg_maxiter,
            cfg.quad_order, cfg.quad_order_offset, load_frame,
        )
        results.append(result)
        print(f"\n  --- Timings (n={n}, p={p}) ---")
        for label, dt in result["timings"].items():
            print(f"  {label:.<32s} {dt:8.3f}s")
        print(f"\n  Relative L2 error     : {result['error']:.6e}")
        print(f"  MINRES iters          : {result['iters']}  converged={result['converged']}")
        print(f"  Nullspace residual    : {result['null_residual']:.3e}  (||L₀ h||)")
        print(f"  Nullspace curl norm   : {result['null_curl_norm']:.3e}  (||D₀ h||)")
        with open(outfile, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"  Saved → {outfile}")

    print(f"\n{'='*60}\n  Summary (p={p}, frame={load_frame})\n{'='*60}")
    print(f"  {'n':>5s}  {'error':>12s}  {'iters':>6s}  {'curl||':>10s}")
    for r in results:
        print(f"  {r['n']:5d}  {r['error']:12.6e}  {r['iters']:6d}  {r['null_curl_norm']:10.3e}")


if __name__ == "__main__":
    main()
