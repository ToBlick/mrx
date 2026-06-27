"""Stage-1 verification: greville MASS preconditioner through the PRODUCTION path.

Assembles the tensor mass preconditioner with cp_kwargs={'greville': True} and
solves M_k x = b via the production apply_inverse_mass_matrix (surgery-Schur + the
greville bulk sandwich), reporting PCG iters vs jacobi. Should match the debug
greville_bulk_precond.py iteration counts (cyl ~8, toroid ~12-14, w7x k0/k3 ~18-27,
w7x k1/k2 higher) and converge everywhere with bad_D=0.
"""
from __future__ import annotations
import argparse, os, sys
from types import SimpleNamespace
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "benchmark"))
from benchmark_graddiv_k1_preconditioner import build_sequence  # noqa: E402
from mrx.operators import (  # noqa: E402
    apply_mass_matrix, apply_inverse_mass_matrix,
    assemble_mass_surgery_preconditioner, assemble_tensor_mass_preconditioner,
)


def dof(seq, k, dirichlet):
    return int(getattr(seq, f"n{k}_dbc" if dirichlet else f"n{k}"))


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
    args = ap.parse_args()

    cfg = SimpleNamespace(ns=tuple(args.ns), p=args.p, geometry=args.geometry, cg_tol=args.tol,
                          cg_maxiter=args.maxiter, epsilon=args.epsilon, kappa=args.kappa,
                          r0=args.r0, nfp=args.nfp)
    print(f"=== greville MASS (prod path)  {args.geometry} ns={tuple(args.ns)} p={args.p} ===", flush=True)
    seq = build_sequence(cfg)
    ops = seq.get_operators()
    ops = assemble_mass_surgery_preconditioner(seq, operators=ops, ks=(0, 1, 2))
    ops = assemble_tensor_mass_preconditioner(seq, operators=ops, ks=(0, 1, 2, 3),
                                              cp_kwargs={"greville": True})

    # solve_singular_cg info convention: negative => converged (|info| = iters),
    # positive => not converged (= maxiter). iters = abs(info), converged iff info<=0.
    def it_conv(info):
        v = int(info)
        return abs(v), (v <= 0)

    print(f"\n{'k':>2} {'bc':5} {'n':>7} {'jac_it':>7} {'grev_it':>8} {'conv':>5} {'grev_res':>10}", flush=True)
    for k in (0, 1, 2, 3):
        for dirichlet in (True, False):
            bc = "dbc" if dirichlet else "free"
            n = dof(seq, k, dirichlet)
            key = jax.random.PRNGKey(args.seed + k + (0 if dirichlet else 100))
            x_true = jax.random.normal(key, (n,), dtype=jnp.float64)
            rhs = apply_mass_matrix(seq, ops, x_true, k, dirichlet=dirichlet)
            try:
                xg, info_g = apply_inverse_mass_matrix(seq, ops, rhs, k, dirichlet=dirichlet,
                                                       tol=args.tol, maxiter=args.maxiter,
                                                       preconditioner="tensor", return_info=True)
                rg = jnp.linalg.norm(apply_mass_matrix(seq, ops, xg, k, dirichlet=dirichlet) - rhs)
                rg = float(rg / jnp.maximum(jnp.linalg.norm(rhs), 1e-30))
                itg, conv = it_conv(info_g)
            except Exception as exc:
                print(f"{k:>2} {bc:5} {n:>7d}  grev ERROR: {repr(exc)[:90]}", flush=True)
                continue
            try:
                _, info_j = apply_inverse_mass_matrix(seq, ops, rhs, k, dirichlet=dirichlet,
                                                      tol=args.tol, maxiter=args.maxiter,
                                                      preconditioner="jacobi", return_info=True)
                itj, _ = it_conv(info_j)
            except Exception:
                itj = -1
            print(f"{k:>2} {bc:5} {n:>7d} {itj:>7d} {itg:>8d} {str(conv):>5} {rg:>10.2e}", flush=True)


if __name__ == "__main__":
    main()
