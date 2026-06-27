"""REAL k=0 Laplacian solve: surgery-Schur + jitted CG, production apply path.

Head-to-head on the FULL k=0 Hodge Laplacian (not bulk-only): jacobi vs the
production rank-1 FD tensor atom vs the Greville atom -- all three through the
same production surgery-Schur preconditioner envelope and the same jitted
solve_singular_cg. The Greville atom is now wired into the production k=0 Hodge
preconditioner (assemble_tensor_laplacian_preconditioner(..., cp_kwargs={'greville':True})),
so apply_laplacian_preconditioner(kind='tensor') dispatches to it.

This is the apples-to-apples answer to "tensor was faster, why?": both run the
same Schur + jit machinery; only the bulk inverse differs.
"""
from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "benchmark"))

from benchmark_graddiv_k1_preconditioner import build_sequence, time_solve  # noqa: E402
from mrx.operators import (  # noqa: E402
    apply_stiffness,
    apply_mass_matrix,
    apply_laplacian_preconditioner,
    assemble_incidence_operators,
    assemble_laplacian_operators,
    assemble_tensor_laplacian_preconditioner,
    _nullspace_vectors,
    _diagonal_from_matvec,
    _invert_diagonal,
)
from mrx.solvers import solve_singular_cg  # noqa: E402


def time_cg(a_matvec, mass_matvec, precond, n, vs, args):
    # Warm up all lazy host-side index plans (matrix-free mass/stiffness/precond
    # build them via np.asarray on first call) with CONCRETE arrays, so they are
    # not traced inside the jitted solve -- otherwise free-BC nullspace projection
    # first touches mass_matvec under trace -> TracerArrayConversionError.
    z = jnp.zeros((n,), dtype=jnp.float64)
    jax.block_until_ready((a_matvec(z), mass_matvec(z), precond(z)))
    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.n_rhs)
    rhs_batch = jnp.stack(
        [a_matvec(jax.random.normal(kk, (n,), dtype=jnp.float64)) for kk in keys], axis=0)
    jax.block_until_ready(rhs_batch)

    @jax.jit
    def solve(rhs, a=a_matvec, m=mass_matvec, p=precond, vv=vs):
        x, info = solve_singular_cg(a, rhs, mass_matvec=m, precond_matvec=p, vs=vv,
                                    tol=args.cg_tol, maxiter=args.cg_maxiter)
        r = a(x) - rhs
        rel = jnp.linalg.norm(r) / jnp.maximum(jnp.linalg.norm(rhs), 1e-30)
        return x, info, rel

    return time_solve(solve, rhs_batch, rel_tol=args.cg_tol)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", default="toroid", choices=["cylinder", "toroid", "rotating_ellipse", "w7x"])
    ap.add_argument("--ns", type=int, nargs=3, default=[12, 24, 12])
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--epsilon", type=float, default=1.0 / 3.0)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--r0", type=float, default=1.0)
    ap.add_argument("--nfp", type=int, default=3)
    ap.add_argument("--n-rhs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cg-tol", type=float, default=1e-10)
    ap.add_argument("--cg-maxiter", type=int, default=10000)
    ap.add_argument("--d-mode", default="geomean", choices=["geomean", "jac"])
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    cfg = SimpleNamespace(ns=tuple(args.ns), p=args.p, geometry=args.geometry, cg_tol=args.cg_tol,
                          cg_maxiter=args.cg_maxiter, epsilon=args.epsilon, kappa=args.kappa,
                          r0=args.r0, nfp=args.nfp)
    nst = "x".join(str(v) for v in args.ns)
    print(f"=== REAL k=0 Laplacian (Schur+jit)  geometry={args.geometry} ns={tuple(args.ns)} p={args.p} ===", flush=True)
    seq = build_sequence(cfg)
    ops = seq.get_operators()
    ops = assemble_incidence_operators(seq, operators=ops, ks=(0,))
    ops = assemble_laplacian_operators(seq, seq.geometry, operators=ops, ks=(0, 1, 2, 3))

    ops_tensor = assemble_tensor_laplacian_preconditioner(seq, operators=ops, ks=(0,), rank=1, cp_kwargs={})
    # (label, weight_mode, alpha_reduce). Compare arithmetic vs geometric mean for the
    # directional constants on combined and pair_d; radial_dense for reference.
    # alpha-reduction sweep: D fixed geometric (combined), vary the spatial mean
    # used for the three directional constants alpha_a.
    grev_specs = [
        ("comb-arith",   "combined", "arith"),
        ("comb-geom",    "combined", "geom"),
        ("comb-minimax", "combined", "minimax"),
    ]
    grev_modes = {}
    for label, wm, ar in grev_specs:
        grev_modes[label] = assemble_tensor_laplacian_preconditioner(
            seq, operators=ops, ks=(0,),
            cp_kwargs={"greville": True, "greville_d_mode": args.d_mode,
                       "greville_weight_mode": wm, "greville_alpha_reduce": ar})

    rows = []
    print(f"\n{'bc':5} {'method':10} {'avg_it':>7} {'max_it':>7} {'avg_ms':>9} {'max_res':>11} {'nfail':>6}")
    for dirichlet in (True, False):
        bc = "dbc" if dirichlet else "free"
        n = int(getattr(seq, f"n0_dbc" if dirichlet else "n0"))
        a_matvec = lambda v, d=dirichlet: apply_stiffness(seq, ops, v, 0, dirichlet=d)
        mass_matvec = lambda v, d=dirichlet: apply_mass_matrix(seq, ops, v, 0, dirichlet=d)
        vs = _nullspace_vectors(ops, 0, dirichlet)

        diaginv = _invert_diagonal(_diagonal_from_matvec(a_matvec, n))
        methods = [
            ("jacobi", lambda v, di=diaginv: di * v),
            ("tensor", lambda v, o=ops_tensor, d=dirichlet: apply_laplacian_preconditioner(seq, o, v, 0, dirichlet=d, kind="tensor")),
        ]
        for label in grev_modes:
            methods.append(
                (f"grev-{label}",
                 lambda v, o=grev_modes[label], d=dirichlet: apply_laplacian_preconditioner(seq, o, v, 0, dirichlet=d, kind="tensor")))
        for name, precond in methods:
            try:
                st = time_cg(a_matvec, mass_matvec, precond, n, vs, args)
                print(f"{bc:5} {name:10} {st['avg_iters']:>7.1f} {st['max_iters']:>7d} "
                      f"{st['avg_ms']:>9.2f} {st['max_residual']:>11.2e} {st['n_fail']:>6d}", flush=True)
                rows.append((args.geometry, nst, args.p, bc, name, st['avg_iters'], st['max_iters'],
                             st['avg_ms'], st['max_residual'], st['n_fail']))
            except Exception as exc:
                print(f"{bc:5} {name:10}  ERROR: {repr(exc)[:120]}", flush=True)

    if args.csv:
        import csv
        new = not os.path.exists(args.csv)
        with open(args.csv, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["geometry", "ns", "p", "bc", "method", "avg_iters", "max_iters", "avg_ms", "max_residual", "n_fail"])
            w.writerows(rows)
        print(f"-> appended {len(rows)} rows to {args.csv}", flush=True)


if __name__ == "__main__":
    main()
