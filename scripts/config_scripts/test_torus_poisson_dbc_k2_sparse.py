"""Convergence study for the k=2 Hodge Laplacian with Dirichlet conditions on a torus.

Solve ``-Δ₂ ω = f`` on the axisymmetric toroid map with homogeneous
Dirichlet (normal-trace) conditions, one resolution after another, and
record the relative metric-weighted L² error, the MINRES iteration count
and every timing. The manufactured solution and source are

    ω₂ = -(2πε³ r sin(2πζ)/R) dr ∧ dχ
        reference 2-form proxy (slot order χζ, rζ, rχ):
        (ω₂)_χζ = 0,  (ω₂)_rζ = 0,  (ω₂)_rχ = -2πε³ r sin(2πζ)/R
    f₂ = ⋆(d f₀),  f₀ = cos(2πζ)/R²
        (f₂)_χζ = -8π² ε² r cos(2πχ) cos(2πζ) / R²
        (f₂)_rζ = -4π ε² sin(2πχ) cos(2πζ) / R²
        (f₂)_rχ = -2π ε³ r sin(2πζ) / R³.

k=2 DBC has a one-dimensional harmonic nullspace, the Hodge dual of the
toroidal 1-form (``⋆dζ ∝ (ε³r/R) dr ∧ dχ``), which the saddle-point MINRES
solve deflates. ``load_frame='ref'`` passes ``f₂`` as bare reference 2-form
components; ``load_frame='phys'`` passes a physical proxy vector and
``load`` applies ``DFᵀ`` itself.

Diagnostics logged per resolution:

- relative metric-weighted L² error,
- MINRES iteration count and convergence flag,
- nullspace residual ``||L₂ h||₂``,
- ``||D₂ h||₂`` (curl, expected ≈ 0),
- ``||D₁ᵀ h||₂`` (divergence, expected ≈ 0).

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
        selects ``2*p + quad_order_offset``. Default ``None``.
    quad_order_offset (int): Offset on ``2*p``. Dataclass default 4; the
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

        python -u scripts/config_scripts/test_torus_poisson_dbc_k2_sparse.py p=3
        python -u scripts/config_scripts/test_torus_poisson_dbc_k2_sparse.py p=2 n=16 precision=float32

    Single GPU job through ``slurm/run.sh``::

        SCRIPT=scripts/config_scripts/test_torus_poisson_dbc_k2_sparse.py ARGS="p=3 n=16" \
            JOB_NAME=pois_dbc_k2 MEM_GB=80 TIMEOUT_MIN=120 bash slurm/run.sh

    Multirun, one submitit job per (p, n) pair. Needs ``SLURM_ACCOUNT``,
    ``SLURM_PARTITION`` and ``MRX_ROOT`` exported; the launcher allots one
    GPU, 80 GB and 120 min per job::

        python scripts/config_scripts/test_torus_poisson_dbc_k2_sparse.py -m p=2,3 n=8,16

Runtime:
    One GPU, p=3, ``logs/mergepois_16819379.out`` (2026-08): TOTAL 167 s at
    n=6, 217 s at n=8, 262 s at n=10 per resolution, JIT included. Memory
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
    assemble_schur_jacobi_preconditioner,
)
from mrx.quadrature import evaluate_at_xq


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
def make_f2_proxy(a: float):
    """f₂ = ⋆(df₀) as the reference 2-form proxy (slot order χζ, rζ, rχ).

    The physics; not a load. These are the primal components ω, i.e. the ones
    a DiscreteFunction evaluates, related to the physical field by
    B_phys = DF ω / J.
    """
    # SIGN, fixed 2026-08-25: frzeta was NEGATIVE. Measured against the working
    # ⋆ in test_torus_poisson_all_k_sparse.py (_hodge_star_1to2_ref, whose
    # cyclic convention (J/g_rr, J/g_χχ, J/g_ζζ) is all-positive), applied to
    # the same f₁ this f₂ is the dual of, the ratio f₂_here / ⋆f₁ was
    # (+ε, −ε, +ε) — exact to 6 decimals at four unrelated points.
    #
    # The uniform ε is harmless: ω₂ below carries the same spurious factor, so
    # it cancels between source and solution, which is why the pair looked
    # self-consistent and why projecting ω₂ alone converges at order 4.5. The
    # MIDDLE SLOT'S SIGN does not cancel -- it is the dr∧dζ vs dζ∧dr
    # orientation -- and a wrong source that does not scale out leaves a fixed,
    # resolution-independent error. That was the residual flat 1.6796e-01 left
    # after the load-frame fix (which had taken it from 1.7818).
    def f(x):
        r, chi, z = x
        R = 1.0 + a * r * jnp.cos(2 * π * chi)
        fchizeta = -8.0 * π**2 * a**2 * r * jnp.cos(2 * π * chi) * jnp.cos(2 * π * z) / R**2
        frzeta   =  4.0 * π * a**2 * jnp.sin(2 * π * chi) * jnp.cos(2 * π * z) / R**2
        frchi    = -2.0 * π * a**3 * r * jnp.sin(2 * π * z) / R**3
        return jnp.array([fchizeta, frzeta, frchi])
    return f


# ---------------------------------------------------------------------------
# Frame adapters.  NEITHER frame takes the bare proxy components.
#
# `load` pairs its argument directly against the basis with weight w (no J at
# k=2), while M₂ = ∫ Λᵀ g Λ / J. So recovering a primal ω from M₂⁻¹·load needs
# the load integrand to be g·ω/J, not ω.
#
# These previously INVERTED load's internal pullback (f₂_phys was DF⁻ᵀf₂_ref,
# so load's DFᵀ handed the bare components straight back), making both frames
# agree with each other and both wrong by one factor of the metric. That does
# not vanish under refinement, so the study reported a FLAT relative L2 error of
# 1.7818/1.7819/1.7819 at n=6/8/10 with MINRES converged=True. The k=1 NBC study
# had the same defect with a different factor (G⁻¹ there, g/J here), which is
# why the two flat constants differed. See
# scripts/debug/poisson_rhs_frame_probe.py for the k=1 measurement.
# ---------------------------------------------------------------------------
def make_f2_ref(a: float, F):
    """g·f₂/J — what load(frame='ref') pairs against the k=2 basis."""
    DF = jax.jacfwd(F)
    f2p = make_f2_proxy(a)
    def f(x):
        dF = DF(x)
        return (dF.T @ dF) @ f2p(x) / jnp.linalg.det(dF)
    return f


def make_f2_phys(a: float, F):
    """DF·f₂/J — the true physical proxy of a 2-form (Piola).

    load(frame='phys') then forms DFᵀ·DF f₂/J = g·f₂/J, matching make_f2_ref.
    """
    DF = jax.jacfwd(F)
    f2p = make_f2_proxy(a)
    def f(x):
        dF = DF(x)
        return dF @ f2p(x) / jnp.linalg.det(dF)
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
                  solver_tol: float, cg_maxiter: int,
                  quad_order, quad_order_offset: int,
                  load_frame: str):
    timings = {}
    ns = (n, 2 * n, n)
    ps = (p, p, p)
    q = 2 * p + quad_order_offset if quad_order is None else quad_order

    F = toroid_map(epsilon=epsilon)
    f2 = make_f2_ref(epsilon, F) if load_frame == 'ref' else make_f2_phys(epsilon, F)
    w2_exact = make_w2_exact_ref(epsilon)

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
    # ks=(0,1,2): k=1,2 for the solve preconditioner and null_2(DBC) iteration;
    # k=0 for null_1(NBC) iteration inside compute_nullspaces_iterative.
    t0 = time.perf_counter()
    ops = assemble_incidence_operators(seq)
    ops = assemble_projection_operators(seq, operators=ops)
    ops = assemble_schur_jacobi_preconditioner(seq, ops, ks=(2,), dirichlet_variants=(True,))
    ops = seq.set_operators(ops)
    jax.block_until_ready(ops)
    timings["assembly_compile"] = time.perf_counter() - t0

    # --- Assembly (exec pass) ------------------------------------------
    t0 = time.perf_counter()
    ops = assemble_incidence_operators(seq)
    ops = assemble_projection_operators(seq, operators=ops)
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
    print(f"precision: {mrx.DTYPE}  solver_tol: {cfg.solver_tol}")
    if cfg.precision != str(mrx.DTYPE):
        raise ValueError(f"precision={cfg.precision} but mrx runs in {mrx.DTYPE}; "
                         "MRX_DTYPE was not set before import")
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
            n, p, cfg.epsilon, cfg.solver_tol, cfg.cg_maxiter,
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
