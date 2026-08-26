"""Convergence study for all eight (k, boundary-condition) Hodge Laplacians on a torus.

Sweep the eight Laplace problems on the axisymmetric toroid map that share
manufactured solutions, one resolution after another, and record per case
the relative error, the iteration count, the nullspace diagnostics and the
``ref``-vs-``phys`` load consistency. Cases pair up under Hodge duality
``⋆: k ↔ (3-k), NBC ↔ DBC`` into four generators:

  generator              cases               exact field / source
  ---------------------  ------------------  -----------------------------------------------
  cos(2πζ)               k0 NBC, k3 DBC      u = cos(2πζ),  f₀ = cos(2πζ)/R²
  cos(πr²/2)             k0 DBC, k3 NBC      u = cos(πr²/2),
                                             f₀ = (2πs + π²r²c)/ε² + πrs cos(2πχ)/(εR)
  cos(2πζ) dζ            k1 NBC, k2 DBC      ω₁ = cos(2πζ) dζ, ω₂ = ⋆ω₁;  f₁ = grad σ, f₂ = ⋆f₁
  cos(πr²/2)cos(2πζ) dζ  k1 DBC, k2 NBC      ω = cos(πr²/2)cos(2πζ) dζ;  f₁ = dσ + curl·curl ω

  (s = sin(πr²/2), c = cos(πr²/2))

Harmonic (nullspace) dimensions per case (Betti numbers (1, 1, 0, 0)):

  k0 NBC: 1 (constant)         k0 DBC: 0
  k1 NBC: 1 (toroidal 1-form)  k1 DBC: 0
  k2 NBC: 0                    k2 DBC: 1 (toroidal 2-form)
  k3 NBC: 0                    k3 DBC: 1 (constant 3-form)

``CASES`` (``test/manufactured.py``, the shared source of these generators
and of ``test/test_poisson.py``) enables all eight pairs. ``ω₁ = cos(2πζ) dζ``
is closed (curl-free), so ``f₁ = L₁ω₁ = grad σ`` with ``σ = -div ω₁``, and
is orthogonal to the harmonic 1-form because ``cos`` has zero mean. The k1
DBC field ``ω = cos(πr²/2)cos(2πζ) dζ`` is not divergence-free in the
interior, but its boundary conditions hold and the source probes all three
covariant slots: ``f₁ = dσ + curl·curl ω``; its tangential trace vanishes
at the wall (``u×n = 0``) and ``σ = 0`` there (both essential k1 DBC
conditions), and k1 DBC has no nullspace. ``ω₂ = ⋆ω₁`` is co-closed with
zero normal trace (the essential k=2 DBC). The Hodge star ``⋆: Ω¹ → Ω²``
uses the framework's cyclic vector-proxy convention (all-positive in the
diagonal metric). See ``docs/source/concepts/manufactured_solutions.md``.

All cases share one ``DeRhamSequence`` and one assembly pass. Both
``frame='ref'`` and ``frame='phys'`` loads are assembled per case and their
difference is reported (expected ≈ 0). k=0 solves use the metric-lumping
Laplacian preconditioner; k≥1 solves are the saddle-point solve of the
sequence.

Configuration:
    Hydra config ``conf/config_poisson_test.yaml``, schema
    ``mrx.config.PoissonTestConfig``. Override any key as ``key=value``.

    n (list[int] | int): Radial resolutions, run one after another; the
        grid is ``ns = (n, 2n, 2n)``. An int runs a single resolution.
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
    load_frame (str): Present in the config, not read by this script.

Usage:
    Single run, all listed n in one process::

        python -u scripts/poisson_study.py p=3
        python -u scripts/poisson_study.py p=2 n=16 precision=float32

    Single GPU job through ``slurm/run.sh``::

        SCRIPT=scripts/poisson_study.py ARGS="p=3 n=16" \
            JOB_NAME=pois_all_k MEM_GB=80 TIMEOUT_MIN=120 bash slurm/run.sh

    Multirun, one submitit job per (p, n) pair. Needs ``SLURM_ACCOUNT``,
    ``SLURM_PARTITION`` and ``MRX_ROOT`` exported; the launcher allots one
    GPU, 80 GB and 120 min per job::

        python scripts/poisson_study.py -m p=2,3 n=8,16

Runtime:
    Not measured. The multirun launcher allots one GPU, 80 GB and 120 min
    per job.

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
from mrx.nullspace import _n_vectors, compute_nullspaces_iterative, get_nullspace
from mrx.operators import (
    assemble_incidence_operators,
    assemble_metric_lumping_laplacian_preconditioner,
)
from test.manufactured import CASES, case_specs, case_tag, relative_l2_error


# ---------------------------------------------------------------------------
# Problem constants
# ---------------------------------------------------------------------------
types = ("clamped", "periodic", "periodic")
BETTI = (1, 1, 0, 0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _log(msg: str):
    """Print with a timestamp and flush immediately."""
    import sys
    ts = time.strftime("%H:%M:%S")
    print(f"  [{ts}] {msg}", flush=True)
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Case registry
# ---------------------------------------------------------------------------
# The eight (k, dirichlet) cases and their manufactured solutions live in
# test/manufactured.py, shared with test/test_poisson.py.


def _null_diag(seq, null_info, k: int, dirichlet: bool):
    """Nullspace diagnostics for one (k, dirichlet) case (handles dim-0 cases)."""
    n_vec = _n_vectors(BETTI, k, dirichlet)
    if n_vec == 0:
        return {
            "null_dim": 0,
            "null_iters": 0,
            "null_final_residual": float("nan"),
            "null_Lh_norm": 0.0,
            "null_curl_norm": 0.0,
            "null_div_norm": 0.0,
        }
    h = get_nullspace(seq.get_operators(), k, dirichlet)[0]
    residual = float(jnp.linalg.norm(seq.apply_laplacian(h, k, dirichlet=dirichlet)))
    iters_res = null_info.get((k, dirichlet), [(0, float("nan"))])[0]
    curl_norm = (float(jnp.linalg.norm(seq.apply_derivative_matrix(
        h, k, dirichlet_in=dirichlet, dirichlet_out=dirichlet))) if k < 3 else 0.0)
    div_norm = (float(jnp.linalg.norm(seq.apply_derivative_matrix(
        h, k - 1, dirichlet_in=dirichlet, dirichlet_out=dirichlet, transpose=True)))
        if k > 0 else 0.0)
    return {
        "null_dim": n_vec,
        "null_iters": iters_res[0],
        "null_final_residual": iters_res[1],
        "null_Lh_norm": residual,
        "null_curl_norm": curl_norm,
        "null_div_norm": div_norm,
    }


def _solve_case(seq, k: int, dirichlet: bool, spec, timings,
                saddle_preconditioner):
    """Load, solve (compile + exec passes), and compute the L² error for one case."""
    tag = case_tag(k, dirichlet)
    _log(f"--- {tag} (k={k}, dirichlet={dirichlet}) ---")

    _log("  Assembling load vectors (ref + phys frame consistency check)...")
    b_ref = seq.load(spec["src_ref"], k, dirichlet=dirichlet, frame='ref')
    b_phys = seq.load(spec["src_phys"], k, dirichlet=dirichlet, frame='phys')
    load_frame_diff = float(jnp.linalg.norm(b_ref - b_phys))
    _log(f"  ||b_ref - b_phys|| = {load_frame_diff:.3e}")

    _log("  Solving (compile pass)...")
    t0 = time.perf_counter()
    preconditioner = 'auto' if k == 0 else saddle_preconditioner
    u_hat, info = seq.apply_inverse_laplacian(
        b_ref, k, dirichlet=dirichlet, preconditioner=preconditioner,
        return_info=True)
    jax.block_until_ready(u_hat)
    timings[f"solve_{tag}_compile"] = time.perf_counter() - t0
    _log(f"  Compile pass done ({timings[f'solve_{tag}_compile']:.2f}s)")

    _log("  Solving (exec pass)...")
    t0 = time.perf_counter()
    u_hat, info = seq.apply_inverse_laplacian(
        b_ref, k, dirichlet=dirichlet, preconditioner=preconditioner,
        return_info=True)
    jax.block_until_ready(u_hat)
    timings[f"solve_{tag}_exec"] = time.perf_counter() - t0
    _log(f"  Solve done: iters={abs(int(info))} converged={int(info) < 0}"
         f" ({timings[f'solve_{tag}_exec']:.2f}s)")

    _log("  Computing L2 error...")
    error = relative_l2_error(seq, k, dirichlet, u_hat, spec["exact"])
    _log(f"  Relative L2 error = {error:.6e}")

    return {
        "k": k,
        "dirichlet": dirichlet,
        "error": error,
        "iters": abs(int(info)),
        "converged": int(info) < 0,
        "load_frame_diff": load_frame_diff,
    }


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------
def compute_all_k(n: int, p: int, epsilon: float,
                  solver_tol: float, cg_maxiter: int,
                  quad_order, quad_order_offset: int):
    timings = {}
    ns = (n, 2 * n, 2 * n)
    ps = (p, p, p)
    q = p + 1 + quad_order_offset if quad_order is None else quad_order

    F = toroid_map(epsilon=epsilon)
    # The production default: block_jacobi mass, the block-Jacobi atom as
    # schur.outer. This was a hand-written spec naming the tensor stack retired
    # on 2026-08-17 and replaced again on 2026-08-22, so the convergence study
    # was not solving what production solves (audit item 3.3).
    saddle_preconditioner = 'auto'

    # --- Sequence setup ------------------------------------------------
    _log(f"Building DeRhamSequence: ns={ns} ps={ps} q={q}")
    t0 = time.perf_counter()
    seq = DeRhamSequence(
        ns, ps, q, types, polar=True,
        tol=solver_tol, maxiter=cg_maxiter,
        betti_numbers=BETTI,
    )
    seq.set_map(F)
    timings["init"] = time.perf_counter() - t0
    _log(f"  DeRhamSequence built: n0={seq.n0} n1={seq.n1} n2={seq.n2} n3={seq.n3}"
         f"  n0_dbc={seq.n0_dbc} n1_dbc={seq.n1_dbc} n2_dbc={seq.n2_dbc} n3_dbc={seq.n3_dbc}"
         f"  ({timings['init']:.2f}s)")

    _log("Evaluating 1D basis functions and geometry...")
    t0 = time.perf_counter()
    seq.evaluate_1d()
    timings["evaluate_1d"] = time.perf_counter() - t0
    _log(f"  evaluate_1d done ({timings['evaluate_1d']:.2f}s)")

    # --- Assembly -------------------------------------------------------
    _log("Assembly: incidence + projection operators...")
    t0 = time.perf_counter()
    ops = assemble_incidence_operators(seq)
    _log("  Assembling the metric-lumping Laplacian preconditioner (k=0..3)...")
    ops = assemble_metric_lumping_laplacian_preconditioner(seq, ops, ks=(0, 1, 2, 3), dirichlets=(True, False))
    _log("  schur.outer = the block-Jacobi atom (production default)...")
    ops = seq.set_operators(ops)
    jax.block_until_ready(ops)
    timings["assembly"] = time.perf_counter() - t0
    _log(f"  Assembly done ({timings['assembly']:.2f}s)")

    # --- Nullspace (iterative for all 4 non-trivial pairs) ---------------
    _log("Computing nullspaces iteratively (k=0 NBC, k=1 NBC, k=2 DBC, k=3 DBC)...")
    t0 = time.perf_counter()
    nullspace_shift = 1.0e-3 / (float(n) ** 2)
    # 1e-8 is sufficient for deflation quality; the analytic initial guesses
    # for k1_nbc (~1.2e-9) and the iterated k2_dbc/k3_dbc modes (~1e-10 to
    # 5e-10) all satisfy this threshold, so most modes will be accepted
    # immediately without burning iterations at maxiter.
    nullspace_abs_tol = 1.0e-8
    # inner_tol only needs to be tight enough for outer iterations to make
    # progress; for shift-and-invert power iteration 1e-6 is sufficient.
    nullspace_inner_tol = 1.0e-6
    _log(f"  Nullspace shift eps={nullspace_shift:.3e} (scaled as 1e-3 / n^2)")
    _log(f"  Nullspace tolerances: abs_tol={nullspace_abs_tol:.1e}, inner_tol={nullspace_inner_tol:.1e}")
    ops, null_info = compute_nullspaces_iterative(seq, seq.get_operators(), BETTI,
                                                   eps=nullspace_shift,
                                                   abs_tol=nullspace_abs_tol,
                                                   inner_tol=nullspace_inner_tol,
                                                   maxiter=100)
    ops = seq.set_operators(ops)
    jax.block_until_ready(ops)
    timings["nullspace"] = time.perf_counter() - t0
    _log(f"  Nullspace iteration done ({timings['nullspace']:.2f}s)")

    # --- Nullspace diagnostics + per-case solves -------------------------
    specs = case_specs(epsilon, F)
    results = {}
    for k, dirichlet in CASES:
        res = _solve_case(
            seq,
            k,
            dirichlet,
            specs[(k, dirichlet)],
            timings,
            saddle_preconditioner,
        )
        nd = _null_diag(seq, null_info, k, dirichlet)
        _log(f"  null {case_tag(k, dirichlet)}: dim={nd['null_dim']}"
             f" iters={nd['null_iters']} ||Lh||={nd['null_Lh_norm']:.3e}"
             f" ||curl||={nd['null_curl_norm']:.3e} ||div||={nd['null_div_norm']:.3e}")
        results[case_tag(k, dirichlet)] = {**res, **nd}

    timings["TOTAL"] = sum(timings.values())
    return {
        "n": n, "p": p, "q": q,
        "cases": [case_tag(k, d) for k, d in CASES],
        "timings": timings,
        **results,
    }


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------
@hydra.main(config_path="../conf", config_name="config_poisson_test", version_base=None)
def main(cfg: DictConfig):
    print(f"precision: {mrx.DTYPE}  solver_tol: {cfg.solver_tol}")
    if cfg.precision != str(mrx.DTYPE):
        raise ValueError(f"precision={cfg.precision} but mrx runs in {mrx.DTYPE}; "
                         "MRX_DTYPE was not set before import")
    ns = [cfg.n] if isinstance(cfg.n, int) else list(cfg.n)
    p = cfg.p
    mrx.MAP_BATCH_SIZE_INNER = cfg.map_batch_size_inner
    mrx.MAP_BATCH_SIZE_OUTER = cfg.map_batch_size_outer
    case_tags = [case_tag(k, d) for k, d in CASES]
    print(f"Hodge–Laplacian convergence | n={ns} p={p} ε={cfg.epsilon}")
    print(f"Cases: {', '.join(case_tags)}")
    print(f"JAX devices: {jax.devices()}")

    output_dir = HydraConfig.get().runtime.output_dir
    outfile = os.path.join(output_dir, "result.json")

    all_results = []
    for n in ns:
        print(f"\n{'='*68}\n  n={n}, p={p}\n{'='*68}")
        result = compute_all_k(
            n, p, cfg.epsilon, cfg.solver_tol, cfg.cg_maxiter,
            cfg.quad_order, cfg.quad_order_offset,
        )
        all_results.append(result)

        print("\n  --- Timings ---")
        for label, dt in result["timings"].items():
            print(f"  {label:.<40s} {dt:8.3f}s")

        print("\n  --- Frame consistency (||b_ref - b_phys||) ---")
        for tag in case_tags:
            print(f"  {tag}: {result[tag]['load_frame_diff']:.3e}")

        print("\n  --- Nullspace diagnostics ---")
        hdr = (f"  {'case':>8s}  {'dim':>3s}  {'iters':>6s}  {'resid':>10s}"
               f"  {'||Lh||':>10s}  {'||curl||':>10s}  {'||div||':>10s}")
        print(hdr)
        for tag in case_tags:
            r = result[tag]
            print(f"  {tag:>8s}  {r['null_dim']:3d}  {r['null_iters']:6d}"
                  f"  {r['null_final_residual']:10.3e}  {r['null_Lh_norm']:10.3e}"
                  f"  {r['null_curl_norm']:10.3e}  {r['null_div_norm']:10.3e}")

        print("\n  --- Convergence ---")
        hdr2 = f"  {'case':>8s}  {'error':>12s}  {'iters':>6s}  {'conv':>5s}"
        print(hdr2)
        for tag in case_tags:
            r = result[tag]
            print(f"  {tag:>8s}  {r['error']:12.6e}  {r['iters']:6d}  {str(r['converged']):>5s}")

        with open(outfile, "w") as fh:
            json.dump(all_results, fh, indent=2)
        print(f"\n  Saved → {outfile}")

    print(f"\n{'='*68}\n  Summary  p={p}\n{'='*68}")
    header = "  " + f"{'n':>5s}" + "".join(f"  {tag + ' err':>14s}" for tag in case_tags)
    print(header)
    for r in all_results:
        row = "  " + f"{r['n']:5d}" + "".join(f"  {r[tag]['error']:14.6e}" for tag in case_tags)
        print(row)


if __name__ == "__main__":
    main()

