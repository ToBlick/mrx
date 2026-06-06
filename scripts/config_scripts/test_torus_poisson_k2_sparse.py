"""Convergence study for the k=2 Hodge–Laplacian on a toroidal domain.

Solves  -Δ₂ ω = f  with natural boundary conditions (NBC), where the exact
solution is  ω₂ = dα  with  α = r²(1-r²) dζ.  Hence:

    ω₂ = g'(r) dr ∧ dζ,   g'(r) = 2r(1-2r²)

k=2 NBC on the solid torus has betti number b₂(abs) = 0, so no nullspace.

Since δα = 0 (α has no ζ-dependence and the metric ∂_ζ block is
ζ-independent), the Hodge Laplacian simplifies:

    -Δ₂(dα) = d(-Δ₁α) = d(δdα) = d(f₁)

where f₁ = δω₂ is a k=1 form with only a dζ component.  Derivation using
the diagonal toroidal metric  g = diag(ε², (2πεr)², (2πR)²):

    ∗(dr ∧ dζ) = -(εr/R) dχ          [star of the rζ 2-form]
    d(∗ω₂)     = -ε ∂_r(rg'/R) dr ∧ dχ
    δω₂ = ∗d∗ω₂  →  (f₁)_ζ = -4(1-4r²) + 2r(1-2r²) ε cos(2πχ) / R

RHS construction (coboundary, exactly parallel to k=0→k=1):
    b₂ = D₁ (M₁⁻¹ load(f₁, 1, NBC))

Usage (run from the repo root):
    # Single run — loops over all n values for the given p
    python scripts/config_scripts/test_torus_poisson_k2_sparse.py p=3

    # Multirun sweep — one SLURM job per (p, n) pair
    python scripts/config_scripts/test_torus_poisson_k2_sparse.py -m p=1,2,3,4 n=8,12,16,24,32
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

# k=2 NBC, betti=(1,1,0,0): b₂(abs)=0 → no nullspace.
BETTI = (1, 1, 0, 0)


def make_f1_phys(a: float):
    """Return the k=1 NBC source  f₁ = δω₂ = -Δ₁α  as a physical 3-vector.

    Reference covariant components: (0, 0, f₁ζ) with
        f₁ζ(r,χ) = -4(1-4r²) + 2r(1-2r²) ε cos(2πχ) / R

    seq.load(f, 1, ...) applies DF⁻¹ to the physical vector, so passing
        f_phys = DF @ (0, 0, f₁ζ) = f₁ζ · ∂_ζF = f₁ζ · 2πR(-sin 2πζ, cos 2πζ, 0)
    recovers (0, 0, f₁ζ) as the reference covariant components.
    """
    def f(x: jnp.ndarray) -> jnp.ndarray:
        r, chi, zeta = x
        R = 1.0 + a * r * jnp.cos(2 * π * chi)
        f1z = (
            -4.0 * (1.0 - 4.0 * r**2)
            + 2.0 * r * (1.0 - 2.0 * r**2) * a * jnp.cos(2 * π * chi) / R
        )
        # f_phys = f₁ζ · ∂_ζF,  ∂_ζF = 2πR(-sin 2πζ, cos 2πζ, 0)
        return f1z * 2 * π * R * jnp.array(
            [-jnp.sin(2 * π * zeta), jnp.cos(2 * π * zeta), 0.0]
        )
    return f


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------
def compute_error(n: int, p: int, epsilon: float,
                  cg_tol: float, cg_maxiter: int,
                  quad_order: int | None,
                  quad_order_offset: int):
    """Run the k=2 NBC saddle-point solve and return (error, timings dict)."""
    timings = {}
    ns = (n, 2 * n, n)
    ps = (p, p, p)
    q = 2 * p + quad_order_offset if quad_order is None else quad_order
    if q < 2 * p:
        raise ValueError(f"quad_order must satisfy q >= 2*p; got q={q}, p={p}")

    F = toroid_map(epsilon=epsilon)
    f1_phys = make_f1_phys(epsilon)

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
    # Assembly: tensor mass (k=1,2) + incidence/projection + Schur Jacobi + nullspace
    # ---------------------------------------------------------------------------
    cp_kwargs = {"maxiter": 100, "tol": 1e-9, "ridge": 1e-12}

    t0 = time.perf_counter()
    ops = assemble_incidence_operators(seq)
    ops = assemble_projection_operators(seq, operators=ops)
    # Tensor mass for k=1 (Schur inner) and k=2 (lower block).
    ops = assemble_tensor_mass_preconditioner(seq, ops, ks=(1, 2), rank=1, cp_kwargs=cp_kwargs)
    # Pre-probe the Schur diagonal (D₁ M₁_tensor⁻¹ D₁ᵀ) for k=2 NBC.
    ops = assemble_schur_jacobi_preconditioner(seq, ops, ks=(2,), dirichlet_variants=(False,))
    # No nullspace: b₂(abs)=0 for k=2 NBC on the solid torus.
    ops = init_nullspaces(seq, ops, BETTI)
    ops = seq.set_operators(ops)
    jax.block_until_ready(ops)
    timings["assembly_compile"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    ops = assemble_incidence_operators(seq)
    ops = assemble_projection_operators(seq, operators=ops)
    ops = assemble_tensor_mass_preconditioner(seq, ops, ks=(1, 2), rank=1, cp_kwargs=cp_kwargs)
    ops = assemble_schur_jacobi_preconditioner(seq, ops, ks=(2,), dirichlet_variants=(False,))
    ops = init_nullspaces(seq, ops, BETTI)
    ops = seq.set_operators(ops)
    jax.block_until_ready(ops)
    timings["assembly_exec"] = time.perf_counter() - t0

    # ---------------------------------------------------------------------------
    # RHS: b₂ = D₁ (M₁⁻¹ load(f₁, 1, NBC))
    # ---------------------------------------------------------------------------
    t0 = time.perf_counter()
    b1 = seq.load(f1_phys, 1, dirichlet=False)
    u1_primal = seq.apply_inverse_mass_matrix(b1, 1, dirichlet=False, preconditioner='tensor')
    rhs = seq.apply_derivative_matrix(u1_primal, 1, dirichlet_in=False, dirichlet_out=False)
    jax.block_until_ready(rhs)
    timings["load_rhs"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    u_hat, info = seq.apply_inverse_hodge_laplacian(rhs, 2, dirichlet=False, return_info=True)
    jax.block_until_ready(u_hat)
    timings["solve_compile"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    u_hat, info = seq.apply_inverse_hodge_laplacian(rhs, 2, dirichlet=False, return_info=True)
    jax.block_until_ready(u_hat)
    timings["solve_exec"] = time.perf_counter() - t0

    info_int = int(info)
    iters = abs(info_int)
    converged = info_int < 0

    # ---------------------------------------------------------------------------
    # Error: physical L² via reference 2-form components weighted by metric.
    #
    # _form_comp_info(2) component ordering (output_dim → reference pair):
    #   slot 0  ω_χζ  →  weight g^χχ g^ζζ = metric_inv[1,1] * metric_inv[2,2]
    #   slot 1  ω_rζ  →  weight g^rr g^ζζ = metric_inv[0,0] * metric_inv[2,2]
    #   slot 2  ω_rχ  →  weight g^rr g^χχ = metric_inv[0,0] * metric_inv[1,1]
    #
    # Exact solution: ω₂ = 2r(1-2r²) dr ∧ dζ  →  slot 1 only.
    # ---------------------------------------------------------------------------
    t0 = time.perf_counter()
    comp_info, comp_shapes = seq._form_comp_info(2)
    quad_shape = (seq.quad.ny, seq.quad.nx, seq.quad.nz)
    w_h = evaluate_at_xq(seq.e2_T @ u_hat, comp_info, comp_shapes, quad_shape, 3)

    def w2_exact_fn(x: jnp.ndarray) -> jnp.ndarray:
        r, _chi, _zeta = x
        return jnp.array([0.0, 2.0 * r * (1.0 - 2.0 * r**2), 0.0])

    w_ex = jax.vmap(w2_exact_fn)(seq.quad.x)  # (nq, 3)

    g_inv = seq.metric_inv_jkl  # (nq, 3, 3)
    weights = jnp.stack([
        g_inv[:, 1, 1] * g_inv[:, 2, 2],   # slot 0: ω_χζ
        g_inv[:, 0, 0] * g_inv[:, 2, 2],   # slot 1: ω_rζ
        g_inv[:, 0, 0] * g_inv[:, 1, 1],   # slot 2: ω_rχ
    ], axis=1)  # (nq, 3)

    diff = w_h - w_ex
    L2_diff = jnp.einsum('qi,qi,qi,q->', diff, diff, weights, seq.jacobian_j * seq.quad.w)
    L2_ref  = jnp.einsum('qi,qi,qi,q->', w_ex, w_ex, weights, seq.jacobian_j * seq.quad.w)
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
    print(f"epsilon type: {type(cfg.epsilon).__name__} value: {cfg.epsilon!r}")
    ns = [cfg.n] if isinstance(cfg.n, int) else list(cfg.n)
    p = cfg.p
    mrx.MAP_BATCH_SIZE_INNER = cfg.map_batch_size_inner
    mrx.MAP_BATCH_SIZE_OUTER = cfg.map_batch_size_outer
    print(f"Running k=2 Poisson (NBC) convergence: n={ns}, p={p}, epsilon={cfg.epsilon}")
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
