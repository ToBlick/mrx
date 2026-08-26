"""Convergence study for the k=1 Hodge Laplacian with natural conditions on a torus.

Solve ``-Δ₁ ω = f`` on the axisymmetric toroid map with natural boundary
conditions (NBC), one resolution after another, and record the relative
physical L² error, the MINRES iteration count and every timing. The
manufactured solution and source are

    ω₁ = -2π sin(2πζ) dζ    reference covariant components (0, 0, -2π sin 2πζ)
    f₁ = d f₀,  f₀ = cos(2πζ)/R²
        (f₁)_r = -2ε cos(2πχ) cos(2πζ) / R³
        (f₁)_χ = 4π ε r sin(2πχ) cos(2πζ) / R³
        (f₁)_ζ = -2π sin(2πζ) / R².

k=1 NBC has a one-dimensional harmonic nullspace spanned by the toroidal
1-form (the Hodge dual of the generator of H¹(T², ℝ)), which the
saddle-point MINRES solve deflates. ``load_frame='ref'`` passes ``f₁`` as
bare reference covariant components; ``load_frame='phys'`` passes a
physical Cartesian vector and ``load`` applies ``DF⁻¹`` itself.

Diagnostics logged per resolution:

- relative physical L² error,
- MINRES iteration count and convergence flag,
- nullspace residual ``||L₁ h||₂``,
- ``||D₁ h||₂`` (curl of the harmonic 1-form, expected ≈ 0),
- ``||D₀ᵀ h||₂`` (divergence, expected ≈ 0).

Configuration:
    Hydra config ``conf/config_poisson_test.yaml``, schema
    ``mrx.config.PoissonTestConfig``. Override any key as ``key=value``.

    n (list[int] | int): Radial resolutions, run one after another; the
        grid is ``ns = (n, 2n, n)``. An int runs a single resolution.
        Default ``[8, 12, 16, 24, 32, 48, 64]``.
    p (int): Spline degree in every direction. Default 3.
    epsilon (float): Minor radius of ``toroid_map`` (major radius 1).
        Default 1/3.
    quad_order (int | None): Gauss quadrature order per direction. ``None``
        selects ``p + 1 + quad_order_offset``. Default ``None``.
    quad_order_offset (int): Offset on ``p + 1``. Dataclass default 4; the
        yaml sets 0.
    cg_maxiter (int): Iteration cap of the Laplacian solve. Dataclass
        default 100000; the yaml sets 50000.
    solver_tol (float | None): Relative residual tolerance of every
        iterative solve in the sequence. ``None`` selects ``sqrt(eps)`` of
        the working precision; the yaml sets 1e-9.
    precision (str): ``float64`` (default) or ``float32``. Read from argv
        and exported as ``MRX_DTYPE`` before ``mrx`` is imported.
    map_batch_size_inner (int): ``mrx.MAP_BATCH_SIZE_INNER``; 0 means
        ``vmap``. Default 0.
    map_batch_size_outer (int | None): ``mrx.MAP_BATCH_SIZE_OUTER``;
        ``None`` means no batching. Default ``None``.
    load_frame (str): ``'ref'`` passes the reference components of the
        source, ``'phys'`` the physical field (see ``mrx.projectors.load``).
        Default ``'ref'``.

Usage:
    Single run, all listed n in one process::

        python -u scripts/config_scripts/test_torus_poisson_nbc_k1_sparse.py p=3
        python -u scripts/config_scripts/test_torus_poisson_nbc_k1_sparse.py p=2 n=16 precision=float32

    Single GPU job through ``slurm/run.sh``::

        SCRIPT=scripts/config_scripts/test_torus_poisson_nbc_k1_sparse.py ARGS="p=3 n=16" \
            JOB_NAME=pois_nbc_k1 MEM_GB=80 TIMEOUT_MIN=120 bash slurm/run.sh

    Multirun, one submitit job per (p, n) pair. Needs ``SLURM_ACCOUNT``,
    ``SLURM_PARTITION`` and ``MRX_ROOT`` exported; the launcher allots one
    GPU, 80 GB and 120 min per job::

        python scripts/config_scripts/test_torus_poisson_nbc_k1_sparse.py -m p=2,3 n=8,16

Runtime:
    One GPU, p=3, ``logs/mergepois_16819379.out`` (2026-08): TOTAL 166 s at
    n=6, 220 s at n=8, 265 s at n=10 per resolution, JIT included. Memory
    not measured; the multirun launcher allots 80 GB.

Output:
    Single run: ``outputs/<date>/<time>/result.json``, a list with one entry
    per n, rewritten after every n so an OOM at a later n keeps the earlier
    results. Multirun: ``multirun/<date>/<time>/<job>/result.json``. Through
    ``slurm/run.sh`` the stdout log is
    ``outputs/<JOB_NAME>/<date>/<time>/<JOB_NAME>.log``.
"""
import json
import os
import time

import sys
# The working precision is chosen before mrx is imported; hydra only hands
# the config over inside main(), so the override is read from argv here.
os.environ["MRX_DTYPE"] = next(
    (a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("precision=")),
    os.environ.get("MRX_DTYPE", "float64"))

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
    assemble_metric_lumping_laplacian_preconditioner,
)
from mrx.quadrature import evaluate_at_xq


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
def make_f1_cov(a: float):
    """f₁ = df₀ in reference COVARIANT components. The physics; not a load."""
    def f(x):
        r, chi, z = x
        R = 1.0 + a * r * jnp.cos(2 * π * chi)
        fr   = -2.0 * a * jnp.cos(2 * π * chi) * jnp.cos(2 * π * z) / R**3
        fchi =  4.0 * π * a * r * jnp.sin(2 * π * chi) * jnp.cos(2 * π * z) / R**3
        fzeta = -2.0 * π * jnp.sin(2 * π * z) / R**2
        return jnp.array([fr, fchi, fzeta])
    return f


# ---------------------------------------------------------------------------
# Frame adapters.  NEITHER frame takes the bare covariant components.
#
# `load` pairs its argument directly against the basis with weight w*J, while
# M₁ = ∫ Λᵀ G⁻¹ Λ J. So recovering a primal covariant ω from M₁⁻¹·load needs
# the load integrand to be G⁻¹ω, not ω.
#
# These two helpers previously INVERTED load's internal pullback (f₁_phys was
# DF @ f₁_ref, so load's DF⁻¹ handed the bare components straight back). That
# made both frames agree with each other and both wrong by one factor of the
# metric -- which does not vanish under refinement, so the study reported a FLAT
# relative L2 error of 3.7256e+01 at n=6/8/10 with MINRES converged=True.
# Measured fix: 8.564e-03 at n=6 and 3.244e-03 at n=8, order ~3.4
# (scripts/debug/poisson_rhs_frame_probe.py, job 16775777). Both corrected
# frames agree to 7 digits, as they must.
# ---------------------------------------------------------------------------
def make_f1_ref(a: float, F):
    """G⁻¹ f₁_cov — what load(frame='ref') pairs against the k=1 basis."""
    DF = jax.jacfwd(F)
    f1c = make_f1_cov(a)
    def f(x):
        dF = DF(x)
        return jnp.linalg.solve(dF.T @ dF, f1c(x))
    return f


def make_f1_phys(a: float, F):
    """DF⁻ᵀ f₁_cov — the true physical proxy of a covariant 1-form.

    load(frame='phys') then forms G⁻¹DFᵀ·DF⁻ᵀf₁_cov = G⁻¹f₁_cov, matching
    make_f1_ref above.
    """
    DF = jax.jacfwd(F)
    f1c = make_f1_cov(a)
    def f(x):
        return jnp.linalg.solve(DF(x).T, f1c(x))
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
                  solver_tol: float, cg_maxiter: int,
                  quad_order, quad_order_offset: int,
                  load_frame: str):
    timings = {}
    ns = (n, 2 * n, n)
    ps = (p, p, p)
    q = p + 1 + quad_order_offset if quad_order is None else quad_order

    F = toroid_map(epsilon=epsilon)
    f1 = make_f1_ref(epsilon, F) if load_frame == 'ref' else make_f1_phys(epsilon, F)

    # --- Sequence setup ------------------------------------------------
    t0 = time.perf_counter()
    seq = DeRhamSequence(
        ns, ps, q, types, polar=True,
        tol=solver_tol, maxiter=cg_maxiter,
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
    ops = assemble_metric_lumping_laplacian_preconditioner(seq, ops, ks=(1,), dirichlets=(False,))
    ops = seq.set_operators(ops)
    jax.block_until_ready(ops)
    timings["assembly_compile"] = time.perf_counter() - t0

    # --- Assembly (exec pass) ------------------------------------------
    t0 = time.perf_counter()
    ops = assemble_incidence_operators(seq)
    ops = assemble_projection_operators(seq, operators=ops)
    ops = assemble_metric_lumping_laplacian_preconditioner(seq, ops, ks=(1,), dirichlets=(False,))
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
    print(f"precision: {mrx.DTYPE}  solver_tol: {cfg.solver_tol}")
    if cfg.precision != str(mrx.DTYPE):
        raise ValueError(f"precision={cfg.precision} but mrx runs in {mrx.DTYPE}; "
                         "MRX_DTYPE was not set before import")
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
            n, p, cfg.epsilon, cfg.solver_tol, cfg.cg_maxiter,
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
