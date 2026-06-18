"""Convergence study for the k=1 Hodge–Laplacian with NBC on a toroidal domain.

Exact solution:  ω₁ = -2π sin(2πζ) dζ  (reference covariant components (0, 0, -2π sin 2πζ))
Source:          f₁ = df₀,  where f₀ = cos(2πζ)/R²
                 Reference covariant components:
                   (f₁)_r    = -2ε cos(2πχ) cos(2πζ) / R³
                   (f₁)_χ    = 4π ε r sin(2πχ) cos(2πζ) / R³
                   (f₁)_ζ    = -2π sin(2πζ) / R²

k=1 NBC has a 1-dimensional harmonic nullspace spanned by the toroidal 1-form
(the Hodge dual of the generator of H¹(T²,ℝ)).

frame='ref'  — pass f₁ as bare reference covariant components (no DF needed).
frame='phys' — pass f₁ as a physical Cartesian vector; load applies DF⁻¹ internally.

Diagnostics logged per run:
  - Relative physical L² error
  - MINRES iteration count and convergence flag
  - Nullspace vector residual  ||L₁ h||₂
  - ||D₁ h||₂   (curl of harmonic 1-form — should be ≈ 0)
  - ||D₀ᵀ h||₂  (divergence — should be ≈ 0)

Usage (from repo root):
    python scripts/config_scripts/test_torus_poisson_nbc_k1_sparse.py -m p=1,2,3 n=8,12,16,20
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
from mrx.nullspace import compute_nullspaces_iterative, get_nullspace
from mrx.operators import (
    assemble_incidence_operators,
    assemble_projection_operators,
    assemble_schur_jacobi_preconditioner,
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
K = 1
DIRICHLET = False


# ---------------------------------------------------------------------------
# Source functions
# ---------------------------------------------------------------------------
def make_f1_ref(a: float):
    """f₁ = df₀ in reference covariant components (no DF transform needed)."""
    def f(x):
        r, chi, z = x
        R = 1.0 + a * r * jnp.cos(2 * π * chi)
        fr   = -2.0 * a * jnp.cos(2 * π * chi) * jnp.cos(2 * π * z) / R**3
        fchi =  4.0 * π * a * r * jnp.sin(2 * π * chi) * jnp.cos(2 * π * z) / R**3
        fzeta = -2.0 * π * jnp.sin(2 * π * z) / R**2
        return jnp.array([fr, fchi, fzeta])
    return f


def make_f1_phys(a: float, F):
    """f₁_phys = DF @ f₁_ref; load applies DF⁻¹, recovering f₁_ref."""
    DF = jax.jacfwd(F)
    f1r = make_f1_ref(a)
    def f(x):
        return DF(x) @ f1r(x)
    return f


# ---------------------------------------------------------------------------
# Exact solution (reference covariant for error)
# ---------------------------------------------------------------------------
def v1_exact_ref(x):
    """ω₁ = -2π sin(2πζ) dζ  →  ref covariant (0, 0, -2π sin 2πζ)."""
    z = x[2]
    return jnp.array([0.0, 0.0, -2.0 * π * jnp.sin(2 * π * z)])


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
    f1 = make_f1_ref(epsilon) if load_frame == 'ref' else make_f1_phys(epsilon, F)
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
    # ks=(0,1,2): k=0,1 for the solve preconditioner and null_1(NBC) iteration;
    # k=2 for null_2(DBC) iteration inside compute_nullspaces_iterative.
    t0 = time.perf_counter()
    ops = assemble_incidence_operators(seq)
    ops = assemble_projection_operators(seq, operators=ops)
    ops = assemble_tensor_mass_preconditioner(seq, ops, ks=(0, 1, 2), rank=1, cp_kwargs=cp_kwargs)
    ops = assemble_schur_jacobi_preconditioner(seq, ops, ks=(1,), dirichlet_variants=(False,))
    ops = seq.set_operators(ops)
    jax.block_until_ready(ops)
    timings["assembly_compile"] = time.perf_counter() - t0

    # --- Assembly (exec pass) ------------------------------------------
    t0 = time.perf_counter()
    ops = assemble_incidence_operators(seq)
    ops = assemble_projection_operators(seq, operators=ops)
    ops = assemble_tensor_mass_preconditioner(seq, ops, ks=(0, 1, 2), rank=1, cp_kwargs=cp_kwargs)
    ops = assemble_schur_jacobi_preconditioner(seq, ops, ks=(1,), dirichlet_variants=(False,))
    ops = seq.set_operators(ops)
    jax.block_until_ready(ops)
    timings["assembly_exec"] = time.perf_counter() - t0

    # --- Nullspace (iterative) -----------------------------------------
    t0 = time.perf_counter()
    ops, null_info = compute_nullspaces_iterative(seq, seq.get_operators(), BETTI)
    ops = seq.set_operators(ops)
    jax.block_until_ready(ops)
    timings["nullspace"] = time.perf_counter() - t0

    h = get_nullspace(seq.get_operators(), K, DIRICHLET)[0]
    Lh = seq.apply_laplacian(h, K, dirichlet=DIRICHLET)
    null_residual = float(jnp.linalg.norm(Lh))
    curl = seq.apply_derivative_matrix(h, K, dirichlet_in=DIRICHLET, dirichlet_out=DIRICHLET)
    null_curl_norm = float(jnp.linalg.norm(curl))
    div_val = seq.apply_derivative_matrix(
        h, K - 1, dirichlet_in=DIRICHLET, dirichlet_out=DIRICHLET, transpose=True)
    null_div_norm = float(jnp.linalg.norm(div_val))
    null_iters = null_info.get((K, DIRICHLET), [(0, 0.0)])[0]

    # --- RHS -----------------------------------------------------------
    t0 = time.perf_counter()
    rhs = seq.load(f1, K, dirichlet=DIRICHLET, frame=load_frame)
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

    # --- Error: physical L² via DF G⁻¹ pushforward ---------------------
    t0 = time.perf_counter()
    comp_info, comp_shapes = seq._form_comp_info(K)
    quad_shape = (seq.quad.ny, seq.quad.nx, seq.quad.nz)
    v_h_log = evaluate_at_xq(seq.e1_T @ u_hat, comp_info, comp_shapes, quad_shape, 3)
    DF_xq = jax.vmap(jax.jacfwd(seq.map))(seq.quad.x)
    v_h_phys = jnp.einsum('qij,qjk,qk->qi', DF_xq, seq.metric_inv_jkl, v_h_log)
    v_ex_ref = jax.vmap(v1_exact_ref)(seq.quad.x)
    v_ex_phys = jnp.einsum('qij,qjk,qk->qi', DF_xq, seq.metric_inv_jkl, v_ex_ref)
    diff = v_h_phys - v_ex_phys
    L2_diff = jnp.einsum('qi,qi,q,q->', diff, diff, seq.jacobian_j, seq.quad.w)
    L2_norm = jnp.einsum('qi,qi,q,q->', v_ex_phys, v_ex_phys, seq.jacobian_j, seq.quad.w)
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
        "null_iters": list(null_iters),
        "null_curl_norm": null_curl_norm,
        "null_div_norm": null_div_norm,
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
    print(f"k=1 NBC Poisson | frame={load_frame} | n={ns} p={p} ε={cfg.epsilon}")
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
        print(f"  Nullspace iters/resid : {result['null_iters']}  (||L₁ h||={result['null_residual']:.3e})")
        print(f"  Nullspace curl norm   : {result['null_curl_norm']:.3e}  (||D₁ h||)")
        print(f"  Nullspace div  norm   : {result['null_div_norm']:.3e}  (||D₀ᵀ h||)")
        with open(outfile, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"  Saved → {outfile}")

    print(f"\n{'='*60}\n  Summary (p={p}, frame={load_frame})\n{'='*60}")
    print(f"  {'n':>5s}  {'error':>12s}  {'iters':>6s}  {'||curl||':>10s}  {'||div||':>10s}")
    for r in results:
        print(f"  {r['n']:5d}  {r['error']:12.6e}  {r['iters']:6d}"
              f"  {r['null_curl_norm']:10.3e}  {r['null_div_norm']:10.3e}")


if __name__ == "__main__":
    main()
