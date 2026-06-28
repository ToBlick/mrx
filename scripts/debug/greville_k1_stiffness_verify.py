"""Validate the greville P_A stiffness atom (k=1 curl-curl, k=2 div-div).

Solves the REAL stiffness system K_k x = b with preconditioned CG and counts
iterations, for three preconditioners:
  - none     : unpreconditioned (baseline lower bound)
  - greville : the WIP greville P_A block  (cp_kwargs={'greville': True})
  - cp       : the existing CP stiffness preconditioner (parity target)

K_k is semidefinite (curl/div nullspace), so the RHS is drawn IN-RANGE
(b = K_k @ random), and P_A deflates the null modes. We measure preconditioner
quality on the range; greville should (a) beat 'none' and (b) match 'cp'.

Mirror of greville_stage4b_verify.py's harness. No mass-solve, geometry-only
build + matrix-free stiffness applies.

Run (per geometry):
  W7X_MAP_BATCH=256 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    .venv/bin/python scripts/debug/greville_k1_stiffness_verify.py --geometry w7x
"""
from __future__ import annotations
import argparse
import os
import sys

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from types import SimpleNamespace  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "benchmark"))
from benchmark_graddiv_k1_preconditioner import build_sequence  # noqa: E402
from mrx.operators import (  # noqa: E402
    apply_mass_matrix,
    apply_stiffness,
    apply_stiffness_tensor_preconditioner,
    assemble_incidence_operators,
    assemble_tensor_stiffness_preconditioner,
)
from mrx.solvers import preconditioned_cg  # noqa: E402


def dof(seq, k, dirichlet):
    return int(getattr(seq, f"n{k}_dbc" if dirichlet else f"n{k}"))


def it_conv(info):
    v = int(info)
    return abs(v), (v <= 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", default="toroid")
    ap.add_argument("--ns", type=int, nargs=3, default=[12, 24, 12])
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--epsilon", type=float, default=1.0 / 3.0)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--r0", type=float, default=1.0)
    ap.add_argument("--nfp", type=int, default=3)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--maxiter", type=int, default=4000)
    ap.add_argument("--shift", type=float, default=0.0,
                    help="optional eps*Mass shift to make K SPD (0 = pure stiffness)")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2])
    args = ap.parse_args()

    cfg = SimpleNamespace(ns=tuple(args.ns), p=args.p, geometry=args.geometry,
                          cg_tol=args.tol, cg_maxiter=args.maxiter, epsilon=args.epsilon,
                          kappa=args.kappa, r0=args.r0, nfp=args.nfp)
    print(f"=== greville P_A stiffness verify  {args.geometry} "
          f"ns={tuple(args.ns)} p={args.p} shift={args.shift} ===", flush=True)
    seq = build_sequence(cfg)
    base = assemble_incidence_operators(seq, seq.get_operators(), ks=(0, 1, 2))

    def make_A(ops, k, d):
        if args.shift > 0:
            return lambda x: (apply_stiffness(seq, ops, x, k, d)
                              + args.shift * apply_mass_matrix(seq, ops, x, k, d))
        return lambda x: apply_stiffness(seq, ops, x, k, d)

    # In-range RHS per (k, dirichlet): b = K @ random  (consistent for singular K).
    rhs = {}
    for k in args.ks:
        for d in (True, False):
            n = dof(seq, k, d)
            v = jax.random.normal(jax.random.PRNGKey(400 + 10 * k + d), (n,), dtype=jnp.float64)
            rhs[(k, d)] = make_A(base, k, d)(v)

    def solve(ops, k, d, M):
        A = make_A(ops, k, d)
        b = rhs[(k, d)]
        x, info = preconditioned_cg(A, b, M=M, tol=args.tol, maxiter=args.maxiter)
        rr = float(jnp.linalg.norm(A(x) - b) / jnp.maximum(jnp.linalg.norm(b), 1e-30))
        it, c = it_conv(info)
        return it, c, rr

    rows = []  # (k, d, variant, it, conv, rr)

    # 1) baseline: unpreconditioned
    for k in args.ks:
        for d in (True, False):
            try:
                rows.append((k, d, "none", *solve(base, k, d, None)))
            except Exception as exc:
                rows.append((k, d, "none", -1, False, repr(exc)[:50]))

    # 2) greville P_A, then 3) CP P_A  (sequential; assemble overwrites in place)
    for variant, kwargs in (("greville", {"greville": True}), ("cp", {})):
        try:
            ops = assemble_tensor_stiffness_preconditioner(
                seq, operators=base, ks=tuple(args.ks), cp_kwargs=kwargs)
        except Exception as exc:
            for k in args.ks:
                for d in (True, False):
                    rows.append((k, d, variant, -1, False, f"BUILD-ERR {repr(exc)[:40]}"))
            continue
        for k in args.ks:
            for d in (True, False):
                def M(v, _ops=ops, _k=k, _d=d):
                    return apply_stiffness_tensor_preconditioner(seq, _ops, v, _k, _d)
                try:
                    rows.append((k, d, variant, *solve(ops, k, d, M)))
                except Exception as exc:
                    rows.append((k, d, variant, -1, False, repr(exc)[:50]))

    # report, grouped by (k, bc) so greville/cp/none sit side by side
    print(f"\n{'k':>2} {'bc':5} {'variant':9} {'it':>6} {'conv':>5} {'rel_res':>11}", flush=True)
    for k in args.ks:
        for d in (True, False):
            for variant in ("none", "greville", "cp"):
                for (rk, rd, rv, it, c, rr) in rows:
                    if (rk, rd, rv) == (k, d, variant):
                        rr_s = f"{rr:.2e}" if isinstance(rr, float) else str(rr)
                        print(f"{k:>2} {'dbc' if d else 'free':5} {variant:9} "
                              f"{it:>6} {str(c):>5} {rr_s:>11}", flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
