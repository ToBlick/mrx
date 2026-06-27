"""Stage-2 verification: k=1,2,3 vector Hodge-Laplacian via greville mass + the
tensor_probe Schur-Jacobi (which auto-uses the greville mass inner inverse).

Assembles greville mass (cp_kwargs={'greville':True}) + surgery + schur-jacobi
(schur_diag_mode='tensor_probe'), then solves L_k x = b through the production
saddle path apply_inverse_hodge_laplacian(preconditioner='auto'). Confirms the
greville-backed probe drives convergence for the vector Laplacians.
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
    apply_hodge_laplacian, apply_inverse_hodge_laplacian,
    assemble_incidence_operators, assemble_laplacian_operators,
    assemble_mass_surgery_preconditioner, assemble_tensor_mass_preconditioner,
    assemble_schur_jacobi_preconditioner,
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
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--maxiter", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = SimpleNamespace(ns=tuple(args.ns), p=args.p, geometry=args.geometry, cg_tol=args.tol,
                          cg_maxiter=args.maxiter, epsilon=args.epsilon, kappa=args.kappa,
                          r0=args.r0, nfp=args.nfp)
    print(f"=== greville VECTOR Laplacian (prod saddle path)  {args.geometry} ns={tuple(args.ns)} p={args.p} ===", flush=True)
    seq = build_sequence(cfg)
    ops = seq.get_operators()
    ops = assemble_incidence_operators(seq, operators=ops, ks=(0, 1, 2))
    ops = assemble_laplacian_operators(seq, seq.geometry, operators=ops, ks=(0, 1, 2, 3))
    ops = assemble_mass_surgery_preconditioner(seq, operators=ops, ks=(0, 1, 2))
    ops = assemble_tensor_mass_preconditioner(seq, operators=ops, ks=(0, 1, 2, 3),
                                              cp_kwargs={"greville": True})
    ops = assemble_schur_jacobi_preconditioner(seq, operators=ops, ks=(1, 2, 3),
                                               dirichlet_variants=(True, False),
                                               schur_diag_mode="tensor_probe")

    def it_conv(info):
        v = int(info)
        return abs(v), (v <= 0)

    print(f"\n{'k':>2} {'bc':5} {'n':>7} {'it':>6} {'conv':>5} {'rel_res':>10}", flush=True)
    for k in (1, 2, 3):
        for dirichlet in (True, False):
            bc = "dbc" if dirichlet else "free"
            n = dof(seq, k, dirichlet)
            key = jax.random.PRNGKey(args.seed + k + (0 if dirichlet else 100))
            x_rand = jax.random.normal(key, (n,), dtype=jnp.float64)
            rhs = apply_hodge_laplacian(seq, ops, x_rand, k, dirichlet=dirichlet)  # in-range
            try:
                x, info = apply_inverse_hodge_laplacian(seq, ops, rhs, k, dirichlet=dirichlet,
                                                        tol=args.tol, maxiter=args.maxiter,
                                                        preconditioner="auto", return_info=True)
                res = apply_hodge_laplacian(seq, ops, x, k, dirichlet=dirichlet) - rhs
                rr = float(jnp.linalg.norm(res) / jnp.maximum(jnp.linalg.norm(rhs), 1e-30))
                it, conv = it_conv(info)
                print(f"{k:>2} {bc:5} {n:>7d} {it:>6d} {str(conv):>5} {rr:>10.2e}", flush=True)
            except Exception as exc:
                print(f"{k:>2} {bc:5} {n:>7d}  ERROR: {repr(exc)[:100]}", flush=True)


if __name__ == "__main__":
    main()
