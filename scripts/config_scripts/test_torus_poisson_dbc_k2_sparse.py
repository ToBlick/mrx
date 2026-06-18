"""Convergence study for the k=2 Hodge–Laplacian with DBC on a toroidal domain.

Exact solution:  ω₂ = -(2πε³r sin(2πζ)/R) dr∧dχ
                 Reference 2-form proxy (slot order χζ, rζ, rχ):
                   (ω₂)_χζ = 0,  (ω₂)_rζ = 0,  (ω₂)_rχ = -2πε³r sin(2πζ)/R

Source:          f₂ = ⋆(df₀),  where f₀ = cos(2πζ)/R²
                 Reference 2-form proxy components:
                   (f₂)_χζ = -8π² ε² r cos(2πχ) cos(2πζ) / R²
                   (f₂)_rζ = -4π ε² sin(2πχ) cos(2πζ) / R²
                   (f₂)_rχ = -2π ε³ r sin(2πζ) / R³

k=2 DBC has a 1-dimensional harmonic nullspace (the Hodge dual of the
toroidal 1-form, i.e. ⋆dζ ∝ (ε³r/R) dr∧dχ).

frame='ref'  — pass f₂ as bare reference 2-form components.
frame='phys' — pass f₂ as a physical proxy vector; load applies DFᵀ internally.

Diagnostics logged per run:
  - Relative metric-weighted L² error
  - MINRES iteration count and convergence flag
  - Nullspace vector residual  ||L₂ h||₂
  - ||D₂ h||₂   (curl — should be ≈ 0)
  - ||D₁ᵀ h||₂  (divergence — should be ≈ 0)

Usage (from repo root):
    python scripts/config_scripts/test_torus_poisson_dbc_k2_sparse.py -m p=1,2,3 n=8,12,16,20
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
K = 2
DIRICHLET = True


# ---------------------------------------------------------------------------
# Source functions
# ---------------------------------------------------------------------------
def make_f2_ref(a: float):
    """f₂ = ⋆(df₀) in reference 2-form proxy components (slot order χζ, rζ, rχ)."""
    def f(x):
        r, chi, z = x
        R = 1.0 + a * r * jnp.cos(2 * π * chi)
        fchizeta = -8.0 * π**2 * a**2 * r * jnp.cos(2 * π * chi) * jnp.cos(2 * π * z) / R**2
        frzeta   = -4.0 * π * a**2 * jnp.sin(2 * π * chi) * jnp.cos(2 * π * z) / R**2
        frchi    = -2.0 * π * a**3 * r * jnp.sin(2 * π * z) / R**3
        return jnp.array([fchizeta, frzeta, frchi])
    return f


def make_f2_phys(a: float, F):
    """f₂_phys = DF⁻ᵀ @ f₂_ref; load applies DFᵀ internally, recovering f₂_ref."""
    DF = jax.jacfwd(F)
    f2r = make_f2_ref(a)
    def f(x):
        return jnp.linalg.solve(DF(x).T, f2r(x))
    return f


# ---------------------------------------------------------------------------
# Exact solution (reference 2-form proxy)
# ---------------------------------------------------------------------------
def make_w2_exact_ref(a: float):
    """ω₂ = -(2πε³r sin(2πζ)/R) dr∧dχ  →  ref slot rχ (index 2) only."""
    def w(x):
        r, chi, z = x
        R = 1.0 + a * r * jnp.cos(2 * π * chi)
        w_rchi = -2.0 * π * a**3 * r * jnp.sin(2 * π * z) / R
        return jnp.array([0.0, 0.0, w_rchi])  # slots: (χζ, rζ, rχ)
    return w


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
    f2 = make_f2_ref(epsilon) if load_frame == 'ref' else make_f2_phys(epsilon, F)
    w2_exact = make_w2_exact_ref(epsilon)
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
    # ks=(0,1,2): k=1,2 for the solve preconditioner and null_2(DBC) iteration;
    # k=0 for null_1(NBC) iteration inside compute_nullspaces_iterative.
    t0 = time.perf_counter()
    ops = assemble_incidence_operators(seq)
    ops = assemble_projection_operators(seq, operators=ops)
    ops = assemble_tensor_mass_preconditioner(seq, ops, ks=(0, 1, 2), rank=1, cp_kwargs=cp_kwargs)
    ops = assemble_schur_jacobi_preconditioner(seq, ops, ks=(2,), dirichlet_variants=(True,))
    ops = seq.set_operators(ops)
    jax.block_until_ready(ops)
    timings["assembly_compile"] = time.perf_counter() - t0

    # --- Assembly (exec pass) ------------------------------------------
    t0 = time.perf_counter()
    ops = assemble_incidence_operators(seq)
    ops = assemble_projection_operators(seq, operators=ops)
    ops = assemble_tensor_mass_preconditioner(seq, ops, ks=(0, 1, 2), rank=1, cp_kwargs=cp_kwargs)
    ops = assemble_schur_jacobi_preconditioner(seq, ops, ks=(2,), dirichlet_variants=(True,))
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
    rhs = seq.load(f2, K, dirichlet=DIRICHLET, frame=load_frame)
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

    # --- Error: metric-weighted 2-form L² -------------------------------
    # Slot ordering (χζ=0, rζ=1, rχ=2); weights g^{ii}g^{jj} for pair (i,j).
    t0 = time.perf_counter()
    comp_info, comp_shapes = seq._form_comp_info(K)
    quad_shape = (seq.quad.ny, seq.quad.nx, seq.quad.nz)
    w_h = evaluate_at_xq(seq.e2_dbc_T @ u_hat, comp_info, comp_shapes, quad_shape, 3)
    w_ex = jax.vmap(w2_exact)(seq.quad.x)
    g_inv = seq.metric_inv_jkl  # (nq, 3, 3)
    weights = jnp.stack([
        g_inv[:, 1, 1] * g_inv[:, 2, 2],   # slot 0: ω_χζ
        g_inv[:, 0, 0] * g_inv[:, 2, 2],   # slot 1: ω_rζ
        g_inv[:, 0, 0] * g_inv[:, 1, 1],   # slot 2: ω_rχ
    ], axis=1)
    diff = w_h - w_ex
    L2_diff = jnp.einsum('qi,qi,qi,q->', diff, diff, weights, seq.jacobian_j * seq.quad.w)
    L2_norm = jnp.einsum('qi,qi,qi,q->', w_ex, w_ex, weights, seq.jacobian_j * seq.quad.w)
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
    print(f"k=2 DBC Poisson | frame={load_frame} | n={ns} p={p} ε={cfg.epsilon}")
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
        print(f"  Nullspace iters/resid : {result['null_iters']}  (||L₂ h||={result['null_residual']:.3e})")
        print(f"  Nullspace curl norm   : {result['null_curl_norm']:.3e}  (||D₂ h||)")
        print(f"  Nullspace div  norm   : {result['null_div_norm']:.3e}  (||D₁ᵀ h||)")
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
