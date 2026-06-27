"""Stage-4b verification: greville is now the DEFAULT/only path for mass + k=0
Laplacian; stiffness CP core kept. Verifies (no 'greville' kwarg anywhere):
  1. k=0 scalar Laplacian solve (greville-only).
  2. mass k=0..3 solves (greville-only default).
  3. vector Laplacian k=1,2,3 (greville mass + tensor_probe Schur-Jacobi).
  4. stiffness preconditioner k=1,2 assembles + applies finite (CP core intact).
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
    apply_mass_matrix, apply_inverse_mass_matrix,
    apply_stiffness_tensor_preconditioner,
    assemble_incidence_operators, assemble_laplacian_operators,
    assemble_mass_surgery_preconditioner, assemble_tensor_mass_preconditioner,
    assemble_schur_jacobi_preconditioner, assemble_tensor_laplacian_preconditioner,
    assemble_tensor_stiffness_preconditioner,
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
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--maxiter", type=int, default=4000)
    args = ap.parse_args()

    cfg = SimpleNamespace(ns=tuple(args.ns), p=args.p, geometry=args.geometry, cg_tol=args.tol,
                          cg_maxiter=args.maxiter, epsilon=args.epsilon, kappa=args.kappa,
                          r0=args.r0, nfp=args.nfp)
    print(f"=== STAGE 4b verify  {args.geometry} ns={tuple(args.ns)} p={args.p} ===", flush=True)
    seq = build_sequence(cfg)
    ops = seq.get_operators()
    ops = assemble_incidence_operators(seq, operators=ops, ks=(0, 1, 2))
    ops = assemble_laplacian_operators(seq, seq.geometry, operators=ops, ks=(0, 1, 2, 3))
    ops = assemble_mass_surgery_preconditioner(seq, operators=ops, ks=(0, 1, 2))
    # NO greville kwarg -> relies on the new default being greville.
    ops = assemble_tensor_mass_preconditioner(seq, operators=ops, ks=(0, 1, 2, 3), cp_kwargs={})
    ops = assemble_tensor_laplacian_preconditioner(seq, operators=ops, ks=(0,))
    ops = assemble_schur_jacobi_preconditioner(seq, operators=ops, ks=(1, 2, 3),
                                               dirichlet_variants=(True, False),
                                               schur_diag_mode="tensor_probe")
    # CP-core stiffness preconditioner must still build.
    try:
        ops = assemble_tensor_stiffness_preconditioner(seq, operators=ops, ks=(1, 2))
        stiff_ok = "built"
    except Exception as exc:
        stiff_ok = f"BUILD-ERR: {repr(exc)[:70]}"
    print(f"stiffness preconditioner (CP core): {stiff_ok}", flush=True)

    print(f"\n{'what':22} {'bc':5} {'it':>6} {'conv':>5} {'rel_res':>10}", flush=True)

    # 1. k=0 scalar Laplacian
    for d in (True, False):
        n = dof(seq, 0, d); key = jax.random.PRNGKey(d * 7)
        rhs = apply_hodge_laplacian(seq, ops, jax.random.normal(key, (n,), dtype=jnp.float64), 0, dirichlet=d)
        try:
            x, info = apply_inverse_hodge_laplacian(seq, ops, rhs, 0, dirichlet=d, tol=args.tol,
                                                    maxiter=args.maxiter, preconditioner="auto", return_info=True)
            rr = float(jnp.linalg.norm(apply_hodge_laplacian(seq, ops, x, 0, dirichlet=d) - rhs)
                       / jnp.maximum(jnp.linalg.norm(rhs), 1e-30))
            it, c = it_conv(info)
            print(f"{'k0 laplacian':22} {'dbc' if d else 'free':5} {it:>6} {str(c):>5} {rr:>10.2e}", flush=True)
        except Exception as exc:
            print(f"{'k0 laplacian':22} {'dbc' if d else 'free':5}  ERR: {repr(exc)[:70]}", flush=True)

    # 2. mass k=0..3
    for k in (0, 1, 2, 3):
        for d in (True, False):
            n = dof(seq, k, d); key = jax.random.PRNGKey(100 + k + d)
            rhs = apply_mass_matrix(seq, ops, jax.random.normal(key, (n,), dtype=jnp.float64), k, dirichlet=d)
            try:
                x, info = apply_inverse_mass_matrix(seq, ops, rhs, k, dirichlet=d, tol=args.tol,
                                                    maxiter=args.maxiter, preconditioner="tensor", return_info=True)
                rr = float(jnp.linalg.norm(apply_mass_matrix(seq, ops, x, k, dirichlet=d) - rhs)
                           / jnp.maximum(jnp.linalg.norm(rhs), 1e-30))
                it, c = it_conv(info)
                print(f"{'mass k'+str(k):22} {'dbc' if d else 'free':5} {it:>6} {str(c):>5} {rr:>10.2e}", flush=True)
            except Exception as exc:
                print(f"{'mass k'+str(k):22} {'dbc' if d else 'free':5}  ERR: {repr(exc)[:70]}", flush=True)

    # 3. vector Laplacian k=1,2,3
    for k in (1, 2, 3):
        for d in (True, False):
            n = dof(seq, k, d); key = jax.random.PRNGKey(200 + k + d)
            rhs = apply_hodge_laplacian(seq, ops, jax.random.normal(key, (n,), dtype=jnp.float64), k, dirichlet=d)
            try:
                x, info = apply_inverse_hodge_laplacian(seq, ops, rhs, k, dirichlet=d, tol=args.tol,
                                                        maxiter=args.maxiter, preconditioner="auto", return_info=True)
                rr = float(jnp.linalg.norm(apply_hodge_laplacian(seq, ops, x, k, dirichlet=d) - rhs)
                           / jnp.maximum(jnp.linalg.norm(rhs), 1e-30))
                it, c = it_conv(info)
                print(f"{'vec laplacian k'+str(k):22} {'dbc' if d else 'free':5} {it:>6} {str(c):>5} {rr:>10.2e}", flush=True)
            except Exception as exc:
                print(f"{'vec laplacian k'+str(k):22} {'dbc' if d else 'free':5}  ERR: {repr(exc)[:70]}", flush=True)

    # 4. stiffness preconditioner apply smoke (CP core)
    for k in (1, 2):
        for d in (True, False):
            n = dof(seq, k, d); key = jax.random.PRNGKey(300 + k + d)
            v = jax.random.normal(key, (n,), dtype=jnp.float64)
            try:
                w = apply_stiffness_tensor_preconditioner(seq, ops, v, k, dirichlet=d)
                finite = bool(jnp.all(jnp.isfinite(w)))
                print(f"{'stiff-precond k'+str(k):22} {'dbc' if d else 'free':5} {'apply':>6} {str(finite):>5} {'(finite)':>10}", flush=True)
            except Exception as exc:
                print(f"{'stiff-precond k'+str(k):22} {'dbc' if d else 'free':5}  ERR: {repr(exc)[:70]}", flush=True)


if __name__ == "__main__":
    main()
