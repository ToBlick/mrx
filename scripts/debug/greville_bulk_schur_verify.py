"""Stage-3 verification: bulk_schur off/on both converge with greville mass (k=1,2).

bulk_schur=off (default) = block-diagonal per-component bulk inverse;
bulk_schur=on  = nested 3x3 Schur across (r,theta,zeta) components.
Both must compose with the greville per-component sandwich. Solves M_k x = b
through the production apply_inverse_mass_matrix(preconditioner='tensor').
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
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--maxiter", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = SimpleNamespace(ns=tuple(args.ns), p=args.p, geometry=args.geometry, cg_tol=args.tol,
                          cg_maxiter=args.maxiter, epsilon=args.epsilon, kappa=args.kappa,
                          r0=args.r0, nfp=args.nfp)
    print(f"=== bulk_schur off/on, greville mass  {args.geometry} ns={tuple(args.ns)} p={args.p} ===", flush=True)
    seq = build_sequence(cfg)
    base = seq.get_operators()
    base = assemble_mass_surgery_preconditioner(seq, operators=base, ks=(0, 1, 2))

    res = {}
    for bs in (False, True):
        ops = assemble_tensor_mass_preconditioner(seq, operators=base, ks=(0, 1, 2, 3),
                                                  cp_kwargs={"greville": True, "bulk_schur": bs})
        for k in (1, 2):
            for dirichlet in (True, False):
                n = dof(seq, k, dirichlet)
                key = jax.random.PRNGKey(args.seed + k + (0 if dirichlet else 100))
                rhs = apply_mass_matrix(seq, ops, jax.random.normal(key, (n,), dtype=jnp.float64),
                                        k, dirichlet=dirichlet)
                try:
                    x, info = apply_inverse_mass_matrix(seq, ops, rhs, k, dirichlet=dirichlet,
                                                        tol=args.tol, maxiter=args.maxiter,
                                                        preconditioner="tensor", return_info=True)
                    rr = float(jnp.linalg.norm(apply_mass_matrix(seq, ops, x, k, dirichlet=dirichlet) - rhs)
                               / jnp.maximum(jnp.linalg.norm(rhs), 1e-30))
                    it, conv = it_conv(info)
                    res[(bs, k, dirichlet)] = (it, conv, rr)
                except Exception as exc:
                    res[(bs, k, dirichlet)] = (None, False, repr(exc)[:70])

    print(f"\n{'k':>2} {'bc':5} {'off_it':>7} {'off_c':>6} {'on_it':>7} {'on_c':>6}", flush=True)
    for k in (1, 2):
        for dirichlet in (True, False):
            bc = "dbc" if dirichlet else "free"
            o = res.get((False, k, dirichlet)); n = res.get((True, k, dirichlet))
            oi = f"{o[0]}" if o and o[0] is not None else f"ERR({o[2] if o else '-'})"
            ni = f"{n[0]}" if n and n[0] is not None else f"ERR({n[2] if n else '-'})"
            print(f"{k:>2} {bc:5} {oi:>7} {str(o[1]) if o else '-':>6} {ni:>7} {str(n[1]) if n else '-':>6}", flush=True)


if __name__ == "__main__":
    main()
