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

Usage::

    python -u scripts/poisson_study.py --p 3
    python -u scripts/poisson_study.py --p 2 --n 16 --precision float32
    SCRIPT=scripts/poisson_study.py ARGS="--p 3 --n 16" \
        JOB_NAME=pois_all_k MEM_GB=80 TIMEOUT_MIN=120 bash slurm/run.sh

``--n`` takes one or more radial resolutions, run one after another in one
process; the grid is ``ns = (n, 2n, 2n)``. ``--tol`` is the relative residual
tolerance of every iterative solve (the archived convergence numbers were
measured at ``1e-9``). ``--precision`` is exported as ``MRX_DTYPE`` before
``mrx`` is imported.

Output: ``<out>/result.json``, a list with one entry per n, rewritten after
every n so an OOM at a later n keeps the earlier results.
"""
import argparse
import json
import os
import sys
import time

# The working precision is chosen before mrx is imported.
_ap = argparse.ArgumentParser(description="Hodge-Laplacian convergence study on the toroid")
_ap.add_argument("--n", type=int, nargs="+", default=[8, 12, 16, 24, 32, 48, 64],
                 help="radial resolutions; the grid is (n, 2n, 2n)")
_ap.add_argument("--p", type=int, default=3, help="spline degree in every direction")
_ap.add_argument("--epsilon", type=float, default=1 / 3, help="minor radius of toroid_map (R0 = 1)")
_ap.add_argument("--quad-order", type=int, default=None, help="Gauss order per direction (default p + 1)")
_ap.add_argument("--tol", type=float, default=1e-9, help="relative residual tolerance of every solve")
_ap.add_argument("--maxiter", type=int, default=50_000, help="iteration cap of the Laplacian solves")
_ap.add_argument("--precision", default=os.environ.get("MRX_DTYPE", "float64"),
                 choices=("float32", "float64"))
_ap.add_argument("--out", default="outputs/poisson_study")
cli = _ap.parse_args()
os.environ["MRX_DTYPE"] = cli.precision

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import mrx  # noqa: E402
from mrx.derham_sequence import DeRhamSequence  # noqa: E402
from mrx.mappings import toroid_map  # noqa: E402
from mrx.nullspace import _n_vectors, compute_nullspaces_iterative, get_nullspace  # noqa: E402
from test.manufactured import CASES, case_specs, case_tag, relative_l2_error  # noqa: E402


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
def compute_all_k(n: int, p: int, epsilon: float, solver_tol: float, cg_maxiter: int,
                  quad_order):
    timings = {}
    ns = (n, 2 * n, 2 * n)
    ps = (p, p, p)
    q = p + 1 if quad_order is None else quad_order

    F = toroid_map(epsilon=epsilon)
    # The production default preconditioners, so the study solves what
    # production solves.
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

    # --- Preconditioners -----------------------------------------------
    _log("Building the incidence operators and every preconditioner...")
    t0 = time.perf_counter()
    ops = seq.build_preconditioners()
    jax.block_until_ready(ops)
    timings["assembly"] = time.perf_counter() - t0
    _log(f"  Preconditioners built ({timings['assembly']:.2f}s)")

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
# Entry point
# ---------------------------------------------------------------------------
def main():
    print(f"precision: {mrx.DTYPE}  tol: {cli.tol}")
    ns, p = cli.n, cli.p
    case_tags = [case_tag(k, d) for k, d in CASES]
    print(f"Hodge–Laplacian convergence | n={ns} p={p} ε={cli.epsilon}")
    print(f"Cases: {', '.join(case_tags)}")
    print(f"JAX devices: {jax.devices()}")

    os.makedirs(cli.out, exist_ok=True)
    outfile = os.path.join(cli.out, "result.json")

    all_results = []
    for n in ns:
        print(f"\n{'='*68}\n  n={n}, p={p}\n{'='*68}")
        result = compute_all_k(n, p, cli.epsilon, cli.tol, cli.maxiter, cli.quad_order)
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
