"""Ceiling test: is the k=1/2 mass CG blowup entirely the off-diagonal coupling?

The greville mass preconditioner is a per-component BLOCK-DIAGONAL lump (drops the
inter-component off-diagonal metric blocks). On W7-X it needs k=1/2 = ~75 CG iters vs
k=0/3 = ~12. Hypothesis (see memory w7x-k1k2-mass-coupling-diagnosis): the gap is the
dropped coupling, and the lumped block-solves themselves are fine.

Decisive check WITHOUT building anything new: solve, with today's greville
preconditioner, the MODIFIED operator M_bd that has the off-diagonal blocks zeroed:

    M_bd(v) = sum_c P_c M (P_c v)          (P_c = projection onto component c)

If PCG(M_bd, greville) converges like k=0 (~12), the lumped block-diagonal
preconditioner is a near-perfect match to the block-diagonal operator => the whole
75->12 gap is coupling => symmetric block-Gauss-Seidel (which reintroduces exactly the
P_c M P_c' blocks) is the right and sufficient lever. If M_bd still takes ~75, lumping
is also contributing and block-SGS alone won't close it.

M_bd is matrix-free (3 full-mass applies per matvec via component masking); no dense
assembly. Component index sets come from the preconditioner's own surgery slices, so
the split is exactly the one the block preconditioner uses (surgery-aware).

  python scripts/debug/greville_mass_coupling_ceiling.py --geometry w7x --ns 12 24 12 --p 3 --nfp 5
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
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
    _build_mass_preconditioner_apply,
)
from mrx.preconditioners import _surgery_slices_k1, _surgery_slices_k2  # noqa: E402
from mrx.solvers import solve_singular_cg  # noqa: E402


def dof(seq, k, dirichlet):
    return int(getattr(seq, f"n{k}_dbc" if dirichlet else f"n{k}"))


def component_index_sets(seq, k, dirichlet):
    """Disjoint index sets for the r/theta/zeta components of the extracted DOF vector."""
    n = dof(seq, k, dirichlet)
    if k in (0, 3):
        return [np.arange(n)]
    if k == 1:
        s = _surgery_slices_k1(seq, dirichlet)
        r = np.arange(s["r"].start, s["r"].stop)
        th = np.concatenate([np.arange(s["theta_surgery"].start, s["theta_surgery"].stop),
                             np.arange(s["theta_bulk"].start, s["theta_bulk"].stop)])
        ze = np.concatenate([np.arange(s["zeta_surgery"].start, s["zeta_surgery"].stop),
                             np.arange(s["zeta_bulk"].start, s["zeta_bulk"].stop)])
        return [r, th, ze]
    if k == 2:
        s = _surgery_slices_k2(seq, dirichlet)
        r = np.arange(0, s["theta"].start)                  # r_surgery + r_bulk
        th = np.arange(s["theta"].start, s["theta"].stop)
        ze = np.arange(s["zeta"].start, s["zeta"].stop)
        return [r, th, ze]
    raise ValueError(k)


def it_conv(info):
    v = int(info)
    return abs(v), (v <= 0)


def timed_solve(A, rhs, precond, mass, tol, maxiter):
    """Warm-up run (compile) then a timed run. Returns (x, info, wall_seconds)."""
    x, info = solve_singular_cg(A, rhs, mass_matvec=mass, precond_matvec=precond,
                                tol=tol, maxiter=maxiter)
    jax.block_until_ready(x)
    t0 = time.perf_counter()
    x, info = solve_singular_cg(A, rhs, mass_matvec=mass, precond_matvec=precond,
                                tol=tol, maxiter=maxiter)
    jax.block_until_ready(x)
    return x, info, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", default="toroid")
    ap.add_argument("--ns", type=int, nargs=3, default=[12, 24, 12])
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--epsilon", type=float, default=1.0 / 3.0)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--r0", type=float, default=1.0)
    ap.add_argument("--nfp", type=int, default=3)
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--maxiter", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", default=None, help="append results to this CSV (header auto-added)")
    args = ap.parse_args()

    cfg = SimpleNamespace(
        ns=tuple(args.ns), p=args.p, geometry=args.geometry, cg_tol=args.tol,
        cg_maxiter=args.maxiter, epsilon=args.epsilon, kappa=args.kappa,
        r0=args.r0, nfp=args.nfp,
    )
    print(f"=== mass coupling CEILING  {args.geometry}  ns={tuple(args.ns)} p={args.p} "
          f"nfp={args.nfp} ===", flush=True)
    seq = build_sequence(cfg)
    ops = seq.get_operators()
    ops = assemble_mass_surgery_preconditioner(seq, operators=ops, ks=(0, 1, 2))
    t_asm = time.perf_counter()
    ops = assemble_tensor_mass_preconditioner(seq, operators=ops, ks=(0, 1, 2, 3),
                                              cp_kwargs={"greville": True})
    jax.block_until_ready(jax.tree_util.tree_leaves(ops)
                          if hasattr(jax.tree_util, "tree_leaves") else 0)
    asm_s = time.perf_counter() - t_asm
    print(f"\ngreville tensor-mass assembly: {asm_s:.2f} s", flush=True)

    csv_f = None
    if args.csv:
        new = not os.path.exists(args.csv) or os.path.getsize(args.csv) == 0
        csv_f = open(args.csv, "a")
        if new:
            csv_f.write("geometry,ns,p,nfp,k,bc,n,Mfull_it,Mfull_s,Mfull_ms_per_it,"
                        "Mbd_it,Mbd_s,Mbd_ms_per_it,offdiag_ratio,asm_s\n")

    print(f"\n{'k':>2} {'bc':5} {'n':>7} {'Mfull_it':>8} {'Mfull_s':>8} {'ms/it':>7}  "
          f"{'Mbd_it':>6} {'Mbd_s':>8} {'ms/it':>7}  {'offdiag':>9}", flush=True)
    ns_str = "x".join(str(v) for v in args.ns)
    for k in (0, 1, 2, 3):
        for dirichlet in (True, False):
            bc = "dbc" if dirichlet else "free"
            n = dof(seq, k, dirichlet)

            def Mfull(v, _k=k, _d=dirichlet):
                return apply_mass_matrix(seq, ops, v, _k, dirichlet=_d)

            idx = component_index_sets(seq, k, dirichlet)
            covered = np.sort(np.concatenate(idx))
            assert covered.size == n and np.array_equal(covered, np.arange(n)), \
                f"component split does not tile the k={k} {bc} vector ({covered.size} vs {n})"
            masks = [jnp.zeros((n,), dtype=jnp.float64).at[jnp.asarray(ii)].set(1.0) for ii in idx]

            def Mbd(v, _mf=Mfull, _masks=masks):
                out = jnp.zeros_like(v)
                for m in _masks:
                    out = out + m * _mf(m * v)
                return out

            precond = _build_mass_preconditioner_apply(
                seq, ops, k=k, dirichlet=dirichlet, preconditioner="tensor", allow_none=True)

            key = jax.random.PRNGKey(args.seed + k + (0 if dirichlet else 100))
            x_true = jax.random.normal(key, (n,), dtype=jnp.float64)

            # operator-level coupling: ||offdiag x|| / ||diag x|| on a random x
            mfx = Mfull(x_true)
            mbx = Mbd(x_true)
            off = jnp.linalg.norm(mfx - mbx) / jnp.maximum(jnp.linalg.norm(mbx), 1e-30)

            xf, info_f, sf = timed_solve(Mfull, mfx, precond, Mfull, args.tol, args.maxiter)
            itf, _ = it_conv(info_f)
            xb, info_b, sb = timed_solve(Mbd, mbx, precond, Mbd, args.tol, args.maxiter)
            itb, _ = it_conv(info_b)
            msf = 1e3 * sf / max(itf, 1)
            msb = 1e3 * sb / max(itb, 1)

            tag = "  (scalar: Mbd==Mfull)" if k in (0, 3) else ""
            print(f"{k:>2} {bc:5} {n:>7d} {itf:>8d} {sf:>8.3f} {msf:>7.2f}  "
                  f"{itb:>6d} {sb:>8.3f} {msb:>7.2f}  {float(off):>9.3e}{tag}", flush=True)
            if csv_f:
                csv_f.write(f"{args.geometry},{ns_str},{args.p},{args.nfp},{k},{bc},{n},"
                            f"{itf},{sf:.4f},{msf:.4f},{itb},{sb:.4f},{msb:.4f},"
                            f"{float(off):.4e},{asm_s:.4f}\n")
                csv_f.flush()
    if csv_f:
        csv_f.close()


if __name__ == "__main__":
    main()
