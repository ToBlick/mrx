"""Jitted runtime comparison: jacobi vs tensor(best) vs greville on the MASS bulk.

Fair speed comparison -- all three preconditioners run on the SAME bulk operator
through the SAME jitted solve_singular_cg (lax loop, no host round-trips), so the
wall times are comparable (unlike the unjitted iteration-sweep harness).

  jacobi   : 1 / diag(M_bulk)               (probed once at setup)
  tensor   : CP/NTF rank-r separable inverse (production "best")
  greville : D^{-1/2} (M0^{-1} kron) D^{-1/2}

Reuses the component specs / greville builder / true bulk applies from
greville_bulk_precond.py. Reports avg jitted solve ms, iters, and one-time setup
ms for each method, per (geometry, ns, p, k, component) -> CSV.
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
sys.path.insert(0, HERE)

from benchmark_graddiv_k1_preconditioner import build_sequence, time_solve  # noqa: E402
from greville_bulk_precond import (  # noqa: E402
    component_specs, build_greville_component, true_applies,
)
from mrx.solvers import solve_singular_cg  # noqa: E402
from mrx.operators import assemble_mass_surgery_preconditioner  # noqa: E402
from mrx.preconditioners import (  # noqa: E402
    diag_matvec,
    _build_diagonal_tensor_block_factors,
    _apply_tensor_diagonal_block_preconditioner,
    _k0_bulk_weight_tensor,
    _k3_weight_tensor,
    _k1_diagonal_metric_tensors,
    _k2_diagonal_metric_tensors,
)


def weight_tensor_for(seq, k, name):
    if k == 0:
        return _k0_bulk_weight_tensor(seq)
    if k == 3:
        return _k3_weight_tensor(seq)
    if k == 1:
        return _k1_diagonal_metric_tensors(seq)[{"arr": "alpha_rr", "theta": "alpha_thetatheta", "zeta": "alpha_zetazeta"}[name]]
    return _k2_diagonal_metric_tensors(seq)[{"r": "beta_rr", "theta": "beta_thetatheta", "zeta": "beta_zetazeta"}[name]]


def build_tensor_component(seq, spec, k, rank, true_apply):
    diff = spec["diff"]
    rbas = seq.d_basis_r_jk if diff[0] else seq.basis_r_jk
    tbas = seq.d_basis_t_jk if diff[1] else seq.basis_t_jk
    zbas = seq.d_basis_z_jk if diff[2] else seq.basis_z_jk
    factors = _build_diagonal_tensor_block_factors(
        seq, weight_tensor_for(seq, k, spec["name"]), spec["shape"], rank,
        radial_basis=rbas, theta_basis=tbas, zeta_basis=zbas,
        radial_weights=seq.quad.w_x, theta_weights=seq.quad.w_y, zeta_weights=seq.quad.w_z,
        radial_start=(1 if diff[0] else 2), cp_maxiter=100, cp_tol=1e-9, cp_ridge=1e-12,
        radial_baseline=None, prior_terms=None, chebyshev_steps=0,
        chebyshev_lanczos_iterations=16, chebyshev_lanczos_max_eig_inflation=1.1,
        chebyshev_lanczos_min_eig_deflation=0.85, chebyshev_lanczos_min_eig_floor_fraction=1e-3,
        chebyshev_seed=100, richardson_steps=0, richardson_omega=1.0, true_block_apply=true_apply)
    return lambda v: _apply_tensor_diagonal_block_preconditioner(factors, v)


def timed_setup(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - t0) * 1e3


def jitted_time(a_apply, precond, n, args):
    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.n_rhs)
    rhs_batch = jnp.stack([a_apply(jax.random.normal(kk, (n,), dtype=jnp.float64)) for kk in keys], axis=0)
    jax.block_until_ready(rhs_batch)

    @jax.jit
    def solve(rhs, a=a_apply, p=precond):
        x, info = solve_singular_cg(a, rhs, precond_matvec=p, vs=[], tol=args.tol, maxiter=args.cg_maxiter)
        rel = jnp.linalg.norm(a(x) - rhs) / jnp.maximum(jnp.linalg.norm(rhs), 1e-30)
        return x, info, rel

    return time_solve(solve, rhs_batch, rel_tol=args.tol)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", default="toroid", choices=["cylinder", "toroid", "rotating_ellipse", "w7x"])
    ap.add_argument("--ns", type=int, nargs=3, default=[12, 24, 12])
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--k", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--rank", type=int, default=3)
    ap.add_argument("--epsilon", type=float, default=1.0 / 3.0)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--r0", type=float, default=1.0)
    ap.add_argument("--nfp", type=int, default=3)
    ap.add_argument("--dirichlet", action="store_true")
    ap.add_argument("--n-rhs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--cg-maxiter", type=int, default=10000)
    ap.add_argument("--methods", nargs="+", default=["greville"],
                    choices=["greville", "tensor", "jacobi"],
                    help="which preconditioners to time (default greville only; "
                         "tensor/jacobi already have stored timings)")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    cfg = SimpleNamespace(ns=tuple(args.ns), p=args.p, geometry=args.geometry, cg_tol=args.tol,
                          cg_maxiter=args.cg_maxiter, epsilon=args.epsilon, kappa=args.kappa, r0=args.r0, nfp=args.nfp)
    nst = "x".join(str(v) for v in args.ns)
    print(f"=== SPEED geometry={args.geometry} ns={tuple(args.ns)} p={args.p} rank={args.rank} dirichlet={args.dirichlet} ===", flush=True)
    seq = build_sequence(cfg)
    ops = seq.get_operators()
    surg_ks = tuple(k for k in args.k if k in (0, 1, 2))
    surg = assemble_mass_surgery_preconditioner(seq, operators=ops, ks=surg_ks) if surg_ks else ops

    rows = []
    print(f"\n{'k':>2} {'comp':6} {'n':>6} {'method':9} {'it':>5} {'solve_ms':>9} {'setup_ms':>9} {'max_res':>10}")
    for k in args.k:
        truths = true_applies(seq, surg, k, args.dirichlet)
        for spec in component_specs(seq, k, args.dirichlet):
            nm = spec["name"]
            A_apply, n = truths[nm]
            methods = []
            if "greville" in args.methods:
                (grev, _, _), grev_setup = timed_setup(lambda: build_greville_component(seq, spec))
                methods.append(("greville", grev, grev_setup))
            if "tensor" in args.methods:
                tensor, tensor_setup = timed_setup(lambda: build_tensor_component(seq, spec, k, args.rank, A_apply))
                methods.append(("tensor", tensor, tensor_setup))
            if "jacobi" in args.methods:
                diag, jac_setup = timed_setup(lambda: diag_matvec(A_apply, n))
                methods.append(("jacobi", (lambda v, d=diag: v / d), jac_setup))
            for name, precond, setup_ms in methods:
                try:
                    st = jitted_time(A_apply, precond, n, args)
                    print(f"{k:>2} {nm:6} {n:>6d} {name:9} {st['avg_iters']:>5.0f} {st['avg_ms']:>9.2f} {setup_ms:>9.0f} {st['max_residual']:>10.1e}", flush=True)
                    rows.append((args.geometry, nst, args.p, int(args.dirichlet), k, nm, n, name,
                                 st['avg_iters'], st['avg_ms'], setup_ms, st['max_residual']))
                except Exception as exc:
                    print(f"{k:>2} {nm:6} {n:>6d} {name:9}  ERROR {repr(exc)[:80]}", flush=True)

    if args.csv:
        import csv
        new = not os.path.exists(args.csv)
        with open(args.csv, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["geometry", "ns", "p", "dirichlet", "k", "comp", "n", "method", "avg_iters", "avg_ms", "setup_ms", "max_res"])
            w.writerows(rows)
        print(f"-> appended {len(rows)} rows to {args.csv}", flush=True)


if __name__ == "__main__":
    main()
