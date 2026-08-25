"""Two routes to the k=1 free harmonic form at p=5, where the direct one is weak.

ROUTE A -- the direct construction, with a better inner solve.
    `compute_nullspaces` builds the form as `v1 = v - weak_curl(L_2^{-1} D_1 v)`
    and its quality IS that inner solve's residual, 1:1. That solve was running
    without the mass preconditioner until 2026-08-24 (the `_tensor_available`
    gate); fixing it bought 280x at p=5 on W7-X. If the rest of the shortfall is
    just budget, tightening tol and raising maxiter on that one solve fixes it,
    and the route stays a PROJECTION -- no iteration, no spectral gap.

ROUTE B -- inverse iteration, with the shift varied.
    `(L + eps M)^{-1} M` converges at roughly `eps/(lambda_1 + eps)` per step,
    so a smaller `eps` converges faster but makes the inner solve harder. If the
    residual floors at the same value for every `eps`, the limit is the inner
    solve and not the iteration; if it tracks `eps`, the shift is the limit.

    python scripts/debug/nullspace_two_routes.py --geometry w7x --p 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mrx.operators as op  # noqa: E402
from mrx.nullspace import (  # noqa: E402
    _commit, _logical_constant_seed, _set_null, find_nullspace_vectors,
    init_nullspaces,
)
from verify_block_jacobi import build_sequence  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry", default="w7x")
    ap.add_argument("--ns", default="12,24,12")
    ap.add_argument("--p", type=int, default=5)
    ap.add_argument("--l2-tols", default="1e-10,1e-12,1e-14")
    ap.add_argument("--l2-maxiters", default="10000,40000")
    ap.add_argument("--eps-ladder", default="1e-2,1e-4,1e-6")
    ap.add_argument("--out", default=None)
    cli = ap.parse_args()
    ns = tuple(int(v) for v in cli.ns.split(","))

    seq, ops = build_sequence(cli.geometry, ns, cli.p, 40000)
    ops = _commit(seq, init_nullspaces(seq, ops, betti_numbers=(1, 1, 0, 0)))
    v0 = jnp.ones(seq.n0)
    ops = _commit(seq, _set_null(ops, 0, False,
                                 (v0 / seq.l2_norm(v0, 0, False))[None, :]))
    seq.set_operators(ops)
    print(f"[setup] {cli.geometry} ns={ns} p={cli.p}", flush=True)

    def rq(v):
        lv = op.apply_hodge_laplacian(seq, ops, v, 1, dirichlet=False)
        mv = op.apply_mass_matrix(seq, ops, v, 1, dirichlet=False)
        return abs(float(v @ lv)) / float(v @ mv)

    seed = _logical_constant_seed(seq, ops, 1, False, (0.0, 0.0, 1.0))
    v, _ = seq.apply_leray_projection(seed, k=1)
    rhs = seq.apply_derivative_matrix(v, 1, dirichlet_in=False,
                                      dirichlet_out=False, operators=ops)
    res = {"geometry": cli.geometry, "p": cli.p, "A": [], "B": []}

    print("\n=== ROUTE A: direct construction, inner L_2 free solve varied ===",
          flush=True)
    print(f"  {'tol':>8}{'maxiter':>9}{'code':>9}{'relL2':>13}{'s':>8}",
          flush=True)
    best = None
    for mi in [int(x) for x in cli.l2_maxiters.split(",")]:
        for tol in [float(x) for x in cli.l2_tols.split(",")]:
            t0 = time.perf_counter()
            out = seq.apply_inverse_hodge_laplacian(
                rhs, 2, dirichlet=False, operators=ops, tol=tol, maxiter=mi,
                return_info=True)
            a, code = (out if isinstance(out, tuple) else (out, None))
            v1 = v - seq.apply_weak_curl(a, False, False)
            v1 = v1 / seq.l2_norm(v1, 1, dirichlet=False)
            q = rq(v1)
            dt = time.perf_counter() - t0
            # minres info: negative == converged, |info| == iterations
            print(f"  {tol:>8.0e}{mi:>9}{int(code):>9}{q ** 0.5:>13.4e}"
                  f"{dt:>8.1f}", flush=True)
            res["A"].append({"tol": tol, "maxiter": mi, "info": int(code),
                             "relL2": q ** 0.5, "s": dt})
            if best is None or q < best[0]:
                best = (q, v1)

    print("\n=== ROUTE B: inverse iteration, shift varied (inner 1e-13) ===",
          flush=True)
    print(f"  {'eps':>8}{'iters':>7}{'residual':>12}{'relL2':>13}{'s':>8}",
          flush=True)
    for eps in [float(x) for x in cli.eps_ladder.split(",")]:
        t0 = time.perf_counter()
        try:
            vecs, infos = find_nullspace_vectors(
                seq, ops, 1, 1, eps, dirichlet=False, x0s=[best[1]],
                inner_tol=1e-13, abs_tol=1e-14, maxiter=100)
            q = rq(vecs[0])
            it, r, _rq = infos[0]
            print(f"  {eps:>8.0e}{int(it):>7}{float(r):>12.2e}"
                  f"{q ** 0.5:>13.4e}{time.perf_counter() - t0:>8.1f}",
                  flush=True)
            res["B"].append({"eps": eps, "iters": int(it),
                             "residual": float(r), "relL2": q ** 0.5})
        except Exception as exc:                              # noqa: BLE001
            print(f"  {eps:>8.0e}  FAILED {str(exc)[:90]}", flush=True)
            res["B"].append({"eps": eps, "error": str(exc)[:200]})

    print("\n=== ROUTE C: direct construction, block atom as schur.outer ===",
          flush=True)
    print("  The inner L_2 free solve above runs on schur.outer='jacobi' (the",
          flush=True)
    print("  library default). S1 measured the block atom at 2.5x on average.",
          flush=True)
    op.assemble_metric_lumping_laplacian_preconditioner(
        seq, ops, ks=(2,), dirichlets=(False,))
    from mrx.operators import _materialize_default_mass_preconditioner  # noqa: PLC0415
    from mrx.preconditioners import (  # noqa: PLC0415
        MassPreconditionerSpec, SaddlePointPreconditionerSpec,
        SchurPreconditionerSpec)
    spec = SaddlePointPreconditionerSpec(
        mass=_materialize_default_mass_preconditioner(seq, ops, k=1),
        schur=SchurPreconditionerSpec(
            inner=MassPreconditionerSpec(kind='metric_lumping'),
            outer=MassPreconditionerSpec(kind='metric_lumping')))
    print(f"  {'tol':>8}{'maxiter':>9}{'code':>9}{'relL2':>13}{'s':>8}",
          flush=True)
    for mi in (10000,):
        for tol in [float(x) for x in cli.l2_tols.split(",")]:
            t0 = time.perf_counter()
            out = seq.apply_inverse_hodge_laplacian(
                rhs, 2, dirichlet=False, operators=ops, tol=tol, maxiter=mi,
                preconditioner=spec, return_info=True)
            a, code = (out if isinstance(out, tuple) else (out, None))
            v1 = v - seq.apply_weak_curl(a, False, False)
            v1 = v1 / seq.l2_norm(v1, 1, dirichlet=False)
            q = rq(v1)
            print(f"  {tol:>8.0e}{mi:>9}{int(code):>9}{q ** 0.5:>13.4e}"
                  f"{time.perf_counter() - t0:>8.1f}", flush=True)
            res.setdefault("C", []).append(
                {"tol": tol, "maxiter": mi, "info": int(code),
                 "relL2": q ** 0.5})

    if cli.out:
        os.makedirs(os.path.dirname(cli.out) or ".", exist_ok=True)
        with open(cli.out, "w") as f:
            json.dump(res, f, indent=1)
        print(f"\nwrote {cli.out}", flush=True)


if __name__ == "__main__":
    main()
