"""Convergence study for the k=3 Hodge–Laplacian with DBC on a toroidal domain.

Exact solution:  ω₃ = cos(2πζ) · J dr∧dχ∧dζ  (proxy scalar: cos(2πζ))
Source:          f₃ same proxy scalar as f₀ = cos(2πζ)/R²

k=3 DBC has a 1-dimensional harmonic nullspace spanned by the constant volume
form  J dr∧dχ∧dζ  (Hodge dual of the constant 0-form).

frame='ref'  — pass f₃ as the coefficient A = f₀·J in A dr∧dχ∧dζ.
frame='phys' — pass f₃ as the physical scalar f₀ = cos(2πζ)/R² (same as k=0).

Diagnostics logged per run:
  - Relative physical L² error (proxy scalar comparison)
  - MINRES iteration count and convergence flag
  - Nullspace vector residual  ||L₃ h||₂
  - ||D₂ᵀ h||₂  (co-derivative / divergence — should be ≈ 0)
  (no curl diagnostic: k=3, no D₃)

Usage (from repo root):
    python scripts/config_scripts/test_torus_poisson_dbc_k3_sparse.py -m p=1,2,3 n=8,12,16,20
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
K = 3
DIRICHLET = True


# ---------------------------------------------------------------------------
# Source functions
# ---------------------------------------------------------------------------
def make_f3_phys(a: float):
    """Physical proxy scalar f₀ = cos(2πζ)/R².  Passed with frame='phys'."""
    def f(x):
        r, chi, z = x
        R = 1.0 + a * r * jnp.cos(2 * π * chi)
        return jnp.cos(2 * π * z) / R**2 * jnp.ones(1)
    return f


def make_f3_ref(a: float, F):
    """Ref coefficient A = f₀·J in  A dr∧dχ∧dζ.  Passed with frame='ref'."""
    DF = jax.jacfwd(F)
    f_phys = make_f3_phys(a)
    def f(x):
        J = jnp.linalg.det(DF(x))
        return f_phys(x) * J
    return f


# ---------------------------------------------------------------------------
# Exact solution (proxy scalar)
# ---------------------------------------------------------------------------
def u3_exact(x):
    """Exact 3-form proxy scalar: cos(2πζ)."""
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
    f3 = make_f3_ref(epsilon, F) if load_frame == 'ref' else make_f3_phys(epsilon)
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
    ops = assemble_tensor_mass_preconditioner(seq, ops, ks=(2, 3), rank=1, cp_kwargs=cp_kwargs)
    ops = assemble_schur_jacobi_preconditioner(seq, ops, ks=(3,), dirichlet_variants=(True,))
    ops = seq.set_operators(ops)
    jax.block_until_ready(ops)
    timings["assembly_compile"] = time.perf_counter() - t0

    # --- Assembly (exec pass) ------------------------------------------
    t0 = time.perf_counter()
    ops = assemble_incidence_operators(seq)
    ops = assemble_projection_operators(seq, operators=ops)
    ops = assemble_tensor_mass_preconditioner(seq, ops, ks=(2, 3), rank=1, cp_kwargs=cp_kwargs)
    ops = assemble_schur_jacobi_preconditioner(seq, ops, ks=(3,), dirichlet_variants=(True,))
    ops = seq.set_operators(ops)
    jax.block_until_ready(ops)
    timings["assembly_exec"] = time.perf_counter() - t0

    # --- Nullspace: k=3 DBC constant volume form (set analytically) ----
    # The constant primal 3-form lives in ker(K₃^DBC) exactly;
    # compute M₃⁻¹·1 using the assembled tensor preconditioner and normalise.
    t0 = time.perf_counter()
    ops = init_nullspaces(seq, seq.get_operators(), BETTI)
    ones3 = jnp.ones(seq.n3_dbc)
    v3 = seq.apply_inverse_mass_matrix(ones3, 3, dirichlet=True, preconditioner='tensor')
    norm3 = seq.l2_norm(v3, 3, dirichlet=True)
    ops = _set_null(ops, 3, True, (v3 / norm3)[None, :])
    ops = seq.set_operators(ops)
    jax.block_until_ready(ops)
    # Diagnostics
    h = get_nullspace(seq.get_operators(), K, DIRICHLET)[0]
    Lh = seq.apply_laplacian(h, K, dirichlet=DIRICHLET)
    null_residual = float(jnp.linalg.norm(Lh))
    div_val = seq.apply_derivative_matrix(
        h, K - 1, dirichlet_in=DIRICHLET, dirichlet_out=DIRICHLET, transpose=True)
    null_div_norm = float(jnp.linalg.norm(div_val))
    timings["nullspace"] = time.perf_counter() - t0

    # --- RHS -----------------------------------------------------------
    t0 = time.perf_counter()
    rhs = seq.load(f3, K, dirichlet=DIRICHLET, frame=load_frame)
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

    # --- Error: physical scalar via proxy (ref coeff / J) ---------------
    t0 = time.perf_counter()
    comp_info, comp_shapes = seq._form_comp_info(K)
    quad_shape = (seq.quad.ny, seq.quad.nx, seq.quad.nz)
    u_h_jk = evaluate_at_xq(seq.e3_dbc_T @ u_hat, comp_info, comp_shapes, quad_shape, 1)
    u_h_phys = u_h_jk / seq.jacobian_j[:, None]  # divide by J to get proxy scalar
    u_ex = jax.vmap(u3_exact)(seq.quad.x)
    diff = u_h_phys - u_ex
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
    print(f"k=3 DBC Poisson | frame={load_frame} | n={ns} p={p} ε={cfg.epsilon}")
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
        print(f"  Nullspace residual    : {result['null_residual']:.3e}  (||L₃ h||)")
        print(f"  Nullspace div  norm   : {result['null_div_norm']:.3e}  (||D₂ᵀ h||)")
        with open(outfile, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"  Saved → {outfile}")

    print(f"\n{'='*60}\n  Summary (p={p}, frame={load_frame})\n{'='*60}")
    print(f"  {'n':>5s}  {'error':>12s}  {'iters':>6s}  {'||div||':>10s}")
    for r in results:
        print(f"  {r['n']:5d}  {r['error']:12.6e}  {r['iters']:6d}  {r['null_div_norm']:10.3e}")


if __name__ == "__main__":
    main()
