"""Quick baseline: plain Jacobi vs greville tensor mass preconditioner (W7-X).

Confirms the greville block-diagonal mass preconditioner (the ~80-iter M1/M2 solver we
are keeping) beats a plain diagonal Jacobi preconditioner on the true mass. k=0/3 scalars
are included for context. One row per (k, bc): iters + wall + ms/it for each, and the
iteration speedup.

  python scripts/debug/mass_jacobi_vs_greville.py --geometry w7x --ns 12 24 24 --nfp 5
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from types import SimpleNamespace

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "benchmark"))
from benchmark_graddiv_k1_preconditioner import build_sequence  # noqa: E402
from mrx.operators import (  # noqa: E402
    apply_mass_matrix,
    assemble_mass_surgery_preconditioner,
    assemble_tensor_mass_preconditioner,
    assemble_mass_jacobi_preconditioner,
    _build_mass_preconditioner_apply,
)
from mrx.solvers import solve_singular_cg  # noqa: E402


def dof(seq, k, dirichlet):
    return int(getattr(seq, f"n{k}_dbc" if dirichlet else f"n{k}"))


def it_conv(info):
    v = int(info)
    return abs(v), (v <= 0)


def timed_solve(A, rhs, precond, mass, tol, maxiter):
    """Jit the whole solve (capturing the operators, compile over rhs) so the timed run
    reuses the compiled CG loop instead of re-lowering it. Warm-up compiles; the second
    call is the honest marginal cost. Returns (info, wall_seconds)."""
    solve = jax.jit(lambda b: solve_singular_cg(
        A, b, mass_matvec=mass, precond_matvec=precond, tol=tol, maxiter=maxiter))
    x, info = solve(rhs)
    jax.block_until_ready((x, info))
    t0 = time.perf_counter()
    x, info = solve(rhs)
    jax.block_until_ready((x, info))
    return info, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", default="w7x")
    ap.add_argument("--ns", type=int, nargs=3, default=[12, 24, 24])
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--epsilon", type=float, default=1.0 / 3.0)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--r0", type=float, default=1.0)
    ap.add_argument("--nfp", type=int, default=5)
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--maxiter", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ks", type=int, nargs="*", default=[0, 1, 2, 3])
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    cfg = SimpleNamespace(
        ns=tuple(args.ns), p=args.p, geometry=args.geometry, cg_tol=args.tol,
        cg_maxiter=args.maxiter, epsilon=args.epsilon, kappa=args.kappa,
        r0=args.r0, nfp=args.nfp,
    )
    print(f"=== jacobi vs greville  {args.geometry}  ns={tuple(args.ns)} p={args.p} "
          f"nfp={args.nfp} ===", flush=True)
    seq = build_sequence(cfg)
    ops = seq.get_operators()
    ops = assemble_mass_surgery_preconditioner(seq, operators=ops, ks=(0, 1, 2))
    t0 = time.perf_counter()
    ops = assemble_tensor_mass_preconditioner(seq, operators=ops, ks=(0, 1, 2, 3),
                                              cp_kwargs={"greville": True})
    ops = assemble_mass_jacobi_preconditioner(seq, operators=ops, ks=(0, 1, 2, 3))
    jax.block_until_ready(jax.tree_util.tree_leaves(ops))
    print(f"assembly (tensor+jacobi): {time.perf_counter() - t0:.1f} s\n", flush=True)

    csv_f = None
    if args.csv:
        new = not os.path.exists(args.csv) or os.path.getsize(args.csv) == 0
        csv_f = open(args.csv, "a")
        if new:
            csv_f.write("geometry,ns,p,nfp,k,bc,n,jac_it,jac_s,jac_ms_per_it,"
                        "grev_it,grev_s,grev_ms_per_it,it_speedup\n")

    ns_str = "x".join(str(v) for v in args.ns)
    print(f"{'k':>2} {'bc':5} {'n':>7}  {'jac_it':>7} {'jac_s':>8} {'ms/it':>7}  "
          f"{'grev_it':>7} {'grev_s':>8} {'ms/it':>7}  {'it_x':>6}", flush=True)
    for k in args.ks:
        for dirichlet in (True, False):
            bc = "dbc" if dirichlet else "free"
            n = dof(seq, k, dirichlet)

            def Mfull(v, _k=k, _d=dirichlet):
                return apply_mass_matrix(seq, ops, v, _k, dirichlet=_d)

            key = jax.random.PRNGKey(args.seed + k + (0 if dirichlet else 100))
            x_true = jax.random.normal(key, (n,), dtype=jnp.float64)
            rhs = Mfull(x_true)

            pj = _build_mass_preconditioner_apply(
                seq, ops, k=k, dirichlet=dirichlet, preconditioner="jacobi", allow_none=True)
            pt = _build_mass_preconditioner_apply(
                seq, ops, k=k, dirichlet=dirichlet, preconditioner="tensor", allow_none=True)

            info_j, sj = timed_solve(Mfull, rhs, pj, Mfull, args.tol, args.maxiter)
            itj, _ = it_conv(info_j)
            info_t, st = timed_solve(Mfull, rhs, pt, Mfull, args.tol, args.maxiter)
            itt, _ = it_conv(info_t)
            msj = 1e3 * sj / max(itj, 1)
            mst = 1e3 * st / max(itt, 1)
            speed = itj / max(itt, 1)

            print(f"{k:>2} {bc:5} {n:>7d}  {itj:>7d} {sj:>8.3f} {msj:>7.2f}  "
                  f"{itt:>7d} {st:>8.3f} {mst:>7.2f}  {speed:>5.1f}x", flush=True)
            if csv_f:
                csv_f.write(f"{args.geometry},{ns_str},{args.p},{args.nfp},{k},{bc},{n},"
                            f"{itj},{sj:.4f},{msj:.4f},{itt},{st:.4f},{mst:.4f},{speed:.2f}\n")
                csv_f.flush()
    if csv_f:
        csv_f.close()


if __name__ == "__main__":
    main()
