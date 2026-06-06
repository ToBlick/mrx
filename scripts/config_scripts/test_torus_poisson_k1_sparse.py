"""Convergence study for the k=1 Hodge–Laplacian on a toroidal domain.

Solves  -Δ₁ v = f  with homogeneous Dirichlet BCs (DBC), where
f = d(-Δ₀ u₀) = d(f₀)  and  v = d(u₀),

with u₀ = 1/4 (r² - r⁴) cos(2πz) the k=0 exact solution (vanishes on ∂Ω).
Since v = du₀ is an exact 1-form (coboundary) it automatically satisfies
tangential-trace = 0, i.e. DBC for k=1.  On a solid torus k=1 DBC has no
nullspace (b₂ = 0), so no deflation is needed.

The RHS is assembled as  rhs₁ = D₀ @ (M₀⁻¹ @ load(f₀, 0, dbc=True)),
i.e. a discrete d applied to the L²-projection of the k=0 source.
The exact solution for error comparison is obtained the same way from u₀.

Preconditioner: tensor mass for the lower block (k=0), tensor Schur inner,
pre-probed Jacobi Schur outer (assemble_schur_jacobi_preconditioner).

Usage (run from the repo root):
    # Single run — loops over all n values for the given p
    python scripts/config_scripts/test_torus_poisson_k1_sparse.py p=3

    # Multirun sweep — one SLURM job per (p, n) pair
    python scripts/config_scripts/test_torus_poisson_k1_sparse.py -m p=1,2,3,4 n=8,12,16,24,32
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
from mrx.operators import (
    assemble_incidence_operators,
    assemble_projection_operators,
    assemble_schur_jacobi_preconditioner,
    assemble_tensor_mass_preconditioner,
)
from mrx.nullspace import init_nullspaces
from mrx.quadrature import evaluate_at_xq

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Problem setup
# ---------------------------------------------------------------------------
types = ("clamped", "periodic", "periodic")
π = jnp.pi

# k=1 DBC, betti=(1,1,0,0): no nullspace (b₂=0 for Dirichlet 1-forms).
BETTI = (1, 1, 0, 0)


def u0(x: jnp.ndarray) -> jnp.ndarray:
    """Exact k=0 scalar: u₀(r,χ,z) = 1/4 (r² - r⁴) cos(2πz)."""
    r, _chi, z = x
    return 1 / 4 * (r**2 - r**4) * jnp.cos(2 * π * z) * jnp.ones(1)


def make_f(a: float):
    """Return the k=0 source term f₀ = -Δ₀ u₀ for minor radius a."""
    def f(x: jnp.ndarray) -> jnp.ndarray:
        r, χ, z = x
        R = 1 + a * r * jnp.cos(2 * π * χ)
        return (
            jnp.cos(2 * π * z)
            * (
                -1 / a**2 * (1 - 4 * r**2)
                - 1 / (a * R) * (r / 2 - r**3) * jnp.cos(2 * π * χ)
                + 1 / 4 * (r**2 - r**4) / R**2
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
    """Run the k=1 saddle-point solve and return (error, timings dict)."""
    timings = {}
    ns = (n, 2 * n, n)
    ps = (p, p, p)
    q = 2 * p + quad_order_offset if quad_order is None else quad_order
    if q < 2 * p:
        raise ValueError(f"quad_order must satisfy q >= 2*p; got q={q}, p={p}")

    F = toroid_map(epsilon=epsilon)
    f0 = make_f(epsilon)

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
    # Assembly: tensor mass (k=0,1) + incidence/projection + Schur Jacobi
    # ---------------------------------------------------------------------------
    cp_kwargs = {"maxiter": 100, "tol": 1e-9, "ridge": 1e-12}

    t0 = time.perf_counter()
    ops = assemble_incidence_operators(seq)
    ops = assemble_projection_operators(seq, operators=ops)
    # Tensor mass for k=0 (Schur inner) and k=1 (lower block).
    ops = assemble_tensor_mass_preconditioner(seq, ops, ks=(0, 1), rank=1, cp_kwargs=cp_kwargs)
    # Pre-probe the Schur diagonal (D M0_tensor^{-1} D^T) for k=1 DBC.
    ops = assemble_schur_jacobi_preconditioner(seq, ops, ks=(1,), dirichlet_variants=(True,))
    # Zero-initialize nullspaces (b₂=0, so k=1 DBC has no harmonic forms).
    ops = init_nullspaces(seq, ops, BETTI)
    ops = seq.set_operators(ops)
    jax.block_until_ready(ops)
    timings["assembly_compile"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    ops = assemble_incidence_operators(seq)
    ops = assemble_projection_operators(seq, operators=ops)
    ops = assemble_tensor_mass_preconditioner(seq, ops, ks=(0, 1), rank=1, cp_kwargs=cp_kwargs)
    ops = assemble_schur_jacobi_preconditioner(seq, ops, ks=(1,), dirichlet_variants=(True,))
    ops = init_nullspaces(seq, ops, BETTI)
    ops = seq.set_operators(ops)
    jax.block_until_ready(ops)
    timings["assembly_exec"] = time.perf_counter() - t0

    # ---------------------------------------------------------------------------
    # Solve
    # ---------------------------------------------------------------------------
    t0 = time.perf_counter()
    # rhs₁ = D₀ (M₀⁻¹ load(f₀, 0))  —  dual k=1 DBC load vector
    b0 = seq.load(f0, 0, dirichlet=True)
    u0_primal = seq.apply_inverse_mass_matrix(b0, 0, dirichlet=True, preconditioner='tensor')
    rhs = seq.apply_derivative_matrix(u0_primal, 0, dirichlet_in=True, dirichlet_out=True)
    jax.block_until_ready(rhs)
    timings["load_rhs"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    u_hat, info = seq.apply_inverse_hodge_laplacian(rhs, 1, dirichlet=True, return_info=True)
    jax.block_until_ready(u_hat)
    timings["solve_compile"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    u_hat, info = seq.apply_inverse_hodge_laplacian(rhs, 1, dirichlet=True, return_info=True)
    jax.block_until_ready(u_hat)
    timings["solve_exec"] = time.perf_counter() - t0

    info_int = int(info)
    iters = abs(info_int)
    converged = info_int < 0

    # ---------------------------------------------------------------------------
    # Error: physical L2 via evaluate_at_xq + pushforward
    # v_ref = (du₀)_ref = (∂_r u₀, 0, ∂_z u₀) in reference covariant components.
    # v_phys = DF G⁻¹ v_ref  (contravariant pushforward for a 1-form).
    # ---------------------------------------------------------------------------
    t0 = time.perf_counter()

    def v_ref_fn(x):
        r, _chi, z = x
        return jnp.array([
            r * (1.0 - 2.0 * r**2) / 2.0 * jnp.cos(2 * π * z),
            0.0,
            -π * (r**2 - r**4) / 2.0 * jnp.sin(2 * π * z),
        ])

    comp_info, comp_shapes = seq._form_comp_info(1)
    quad_shape = (seq.quad.ny, seq.quad.nx, seq.quad.nz)
    # Reference covariant components of discrete solution at quadrature points.
    v_h_log = evaluate_at_xq(
        seq.e1_dbc_T @ u_hat, comp_info, comp_shapes, quad_shape, 3)
    # DF at all quadrature points, shape (nq, 3, 3).
    DF_xq = jax.vmap(jax.jacfwd(seq.map))(seq.quad.x)
    # Push to physical Cartesian: v_phys = DF @ (G⁻¹ @ v_cov).
    v_h_phys = jnp.einsum('qij,qjk,qk->qi', DF_xq, seq.metric_inv_jkl, v_h_log)
    v_ex_phys = jnp.einsum('qij,qjk,qk->qi', DF_xq, seq.metric_inv_jkl,
                            jax.vmap(v_ref_fn)(seq.quad.x))
    diff = v_h_phys - v_ex_phys
    L2_diff = jnp.einsum('qi,qi,q,q->', diff, diff, seq.jacobian_j, seq.quad.w)
    L2_ref  = jnp.einsum('qi,qi,q,q->', v_ex_phys, v_ex_phys, seq.jacobian_j, seq.quad.w)
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
    print(f"Running k=1 Poisson (DBC) convergence: n={ns}, p={p}, epsilon={cfg.epsilon}")
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
