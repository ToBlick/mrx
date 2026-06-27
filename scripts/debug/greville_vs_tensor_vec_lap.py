"""Greville vs CP-tensor on the VECTOR Hodge-Laplacians (k=1,2,3).

Same production saddle solve (apply_inverse_hodge_laplacian, preconditioner='auto'
= mass + tensor_probe Schur-Jacobi); only the MASS atom feeding the probe differs:
  greville: cp_kwargs={'greville': True}
  tensor:   cp_kwargs={} (CP fit, rank 3)
Reports PCG/MINRES iters + convergence for each, on cyl/toroid/w7x, both BCs.
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
    ap.add_argument("--rank", type=int, default=3)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--maxiter", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = SimpleNamespace(ns=tuple(args.ns), p=args.p, geometry=args.geometry, cg_tol=args.tol,
                          cg_maxiter=args.maxiter, epsilon=args.epsilon, kappa=args.kappa,
                          r0=args.r0, nfp=args.nfp)
    print(f"=== greville vs tensor, VECTOR Laplacian  {args.geometry} ns={tuple(args.ns)} p={args.p} ===", flush=True)
    seq = build_sequence(cfg)
    base = seq.get_operators()
    base = assemble_incidence_operators(seq, operators=base, ks=(0, 1, 2))
    base = assemble_laplacian_operators(seq, seq.geometry, operators=base, ks=(0, 1, 2, 3))
    base = assemble_mass_surgery_preconditioner(seq, operators=base, ks=(0, 1, 2))

    def build(mass_kind):
        if mass_kind == "greville":
            ops = assemble_tensor_mass_preconditioner(seq, operators=base, ks=(0, 1, 2, 3),
                                                      cp_kwargs={"greville": True})
        else:
            ops = assemble_tensor_mass_preconditioner(seq, operators=base, ks=(0, 1, 2, 3),
                                                      rank=args.rank, cp_kwargs={})
        ops = assemble_schur_jacobi_preconditioner(seq, operators=ops, ks=(1, 2, 3),
                                                   dirichlet_variants=(True, False),
                                                   schur_diag_mode="tensor_probe")
        return ops

    results = {}
    for mass_kind in ("greville", "tensor"):
        try:
            ops = build(mass_kind)
        except Exception as exc:
            print(f"[{mass_kind}] ASSEMBLY ERROR: {repr(exc)[:120]}", flush=True)
            continue
        for k in (1, 2, 3):
            for dirichlet in (True, False):
                n = dof(seq, k, dirichlet)
                key = jax.random.PRNGKey(args.seed + k + (0 if dirichlet else 100))
                rhs = apply_hodge_laplacian(seq, ops, jax.random.normal(key, (n,), dtype=jnp.float64),
                                            k, dirichlet=dirichlet)
                try:
                    x, info = apply_inverse_hodge_laplacian(seq, ops, rhs, k, dirichlet=dirichlet,
                                                            tol=args.tol, maxiter=args.maxiter,
                                                            preconditioner="auto", return_info=True)
                    res = apply_hodge_laplacian(seq, ops, x, k, dirichlet=dirichlet) - rhs
                    rr = float(jnp.linalg.norm(res) / jnp.maximum(jnp.linalg.norm(rhs), 1e-30))
                    it, conv = it_conv(info)
                    results[(mass_kind, k, dirichlet)] = (it, conv, rr)
                except Exception as exc:
                    results[(mass_kind, k, dirichlet)] = (None, False, repr(exc)[:60])

    print(f"\n{'k':>2} {'bc':5} {'grev_it':>8} {'grev_c':>6} {'tens_it':>8} {'tens_c':>6}", flush=True)
    for k in (1, 2, 3):
        for dirichlet in (True, False):
            bc = "dbc" if dirichlet else "free"
            g = results.get(("greville", k, dirichlet))
            t = results.get(("tensor", k, dirichlet))
            gi = f"{g[0]}" if g and g[0] is not None else "ERR"
            gc = str(g[1]) if g else "-"
            ti = f"{t[0]}" if t and t[0] is not None else "ERR"
            tc = str(t[1]) if t else "-"
            print(f"{k:>2} {bc:5} {gi:>8} {gc:>6} {ti:>8} {tc:>6}", flush=True)


if __name__ == "__main__":
    main()
