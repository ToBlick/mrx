"""Would a looser tolerance fix the k>=1 saddle solves, and is it safe?

Two questions, and they are different:

1. IS IT STAGNATION OR SLOW CONVERGENCE? Relaxing tol only helps if MINRES is
   still descending. Run the real solver with tol effectively unreachable and a
   LADDER of maxiter, then measure the residual of what comes back. That gives
   the convergence curve from the jitted production path, at production speed --
   no instrumented Python loop.

2. IS A LOOSE TOLERANCE SAFE? MINRES's stopping test is on `phibar`, an estimate
   of the residual of the FULL SADDLE system -- upper block and the constraint
   row `D^T u - M s = 0` together. What we actually care about is `L_k u = rhs`.
   Those are not the same number, and the joint one can be much the harder to
   drive down. So report the quantity of interest.

   IN THE DUAL NORM. `r = L_k u - rhs` is a DUAL vector -- a functional, not a
   field -- so its Euclidean norm is not a coherent measure: it weights modes by
   whatever `M^-1` does and carries `||L|| ~ h^-2` (the same trap
   `verify_block_jacobi`'s Rayleigh-quotient comment calls out for `||Lv||/||v||`).
   The natural norm on the dual is

       ||r||_{M^-1} = sqrt(r^T M_k^-1 r),     reported relative to ||rhs||_{M^-1}

   which is the norm of the functional itself and is mesh-independent. `M_k^-1`
   here is a real mass SOLVE, not the preconditioner -- this is a diagnostic and
   can afford it.

   If that is already at 1e-10 while phibar is at 1e-6, the tolerance is simply
   measuring the wrong thing and relaxing it is free rather than a compromise.

    python scripts/debug/minres_tol_curve.py --geometry w7x --k 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mrx.operators as op  # noqa: E402
from mrx.nullspace import compute_nullspaces, get_nullspace  # noqa: E402
from mrx.preconditioners import (  # noqa: E402
    MassPreconditionerSpec, SaddlePointPreconditionerSpec,
    SchurPreconditionerSpec,
)
from verify_block_jacobi import build_sequence  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry", default="w7x")
    ap.add_argument("--ns", default="12,24,12")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--ks", default="1,2,3")
    ap.add_argument("--bcs", default="free,dbc")
    ap.add_argument("--tols", default="1e-6,1e-8,1e-10")
    ap.add_argument("--iter-ladder", default="250,500,1000,2000,4000,8000")
    ap.add_argument("--maxiter", type=int, default=10000)
    ap.add_argument("--out", default=None)
    cli = ap.parse_args()
    ns = tuple(int(v) for v in cli.ns.split(","))

    seq, ops = build_sequence(cli.geometry, ns, cli.p, cli.maxiter)
    ops = compute_nullspaces(seq, ops)
    seq.set_operators(ops)
    print(f"[setup] {cli.geometry} ns={ns} p={cli.p}", flush=True)

    res = {"geometry": cli.geometry, "ns": list(ns), "p": cli.p, "rows": []}
    for k in [int(v) for v in cli.ks.split(",")]:
        for dbc in (False, True):
            if ("dbc" if dbc else "free") not in cli.bcs.split(","):
                continue
            op.assemble_block_jacobi_laplacian_preconditioner(
                seq, ops, ks=(k,), dirichlets=(dbc,))
            spec = SaddlePointPreconditionerSpec(
                mass=op._materialize_default_mass_preconditioner(
                    seq, ops, k=k - 1),
                schur=SchurPreconditionerSpec(
                    inner=MassPreconditionerSpec(kind='raw_kron'),
                    outer=MassPreconditionerSpec(kind='block')))
            n = int(getattr(seq, f"n{k}_dbc" if dbc else f"n{k}"))
            rhs = jax.random.normal(jax.random.PRNGKey(31 * k + dbc), (n,))

            # The solver deflates: solve_saddle_point_minres forms
            # `b = project_dual(b_upper)`, so it solves the system with the
            # nullspace component of the rhs REMOVED. Comparing against the raw
            # rhs therefore charges the solve for a component it was never
            # asked to reproduce -- which shows up as a residual FLOOR on
            # exactly the singular rows (k=1 free, k=2 dbc) at the size of a
            # random vector's projection onto the 1-D nullspace (~3.3e-3 at
            # n=8700, identical across geometries -- the tell). Project the rhs
            # the same way the solver does.
            _vs = jnp.asarray(get_nullspace(ops, k, dbc))

            def project_dual(f, k=k, dbc=dbc, _vs=_vs):
                if _vs.shape[0] == 0:
                    return f
                mass_vs = jax.vmap(lambda v: op.apply_mass_matrix(
                    seq, ops, v, k, dirichlet=dbc))(_vs)
                return f - (_vs @ f) @ mass_vs

            rhs_p = project_dual(rhs)

            def dual_norm(f, k=k, dbc=dbc):
                """||f||_{M^-1} = sqrt(f^T M_k^-1 f), the norm on the DUAL."""
                Minv_f = op.apply_inverse_mass_matrix(
                    seq, ops, f, k, dirichlet=dbc, tol=1e-13)
                return float(jnp.sqrt(jnp.dot(f, Minv_f)))

            nb_dual = dual_norm(rhs_p)

            def true_rel(u, k=k, dbc=dbc):
                """Residual of the DEFLATED problem, in the dual norm."""
                r = op.apply_hodge_laplacian(
                    seq, ops, u, k, dirichlet=dbc) - rhs_p
                return dual_norm(project_dual(r)) / nb_dual

            print(f"\n=== k={k} dbc={dbc} n={n} ===", flush=True)
            print(f"  {'stop':>12}{'iters':>8}{'conv':>6}"
                  f"{'||L u - rhs||_Minv / ||rhs||_Minv':>34}{'s':>8}",
                  flush=True)
            row = {"k": k, "dbc": dbc, "n": n, "tols": [], "ladder": []}

            for tol in [float(t) for t in cli.tols.split(",")]:
                t0 = time.perf_counter()
                u, info = seq.apply_inverse_hodge_laplacian(
                    rhs, k, dirichlet=dbc, operators=ops, tol=tol,
                    maxiter=cli.maxiter, preconditioner=spec, return_info=True)
                code = int(info)
                tr = true_rel(u)
                print(f"  {('tol=%.0e' % tol):>12}{abs(code):>8}"
                      f"{str(code < 0):>6}{tr:>34.4e}"
                      f"{time.perf_counter() - t0:>8.1f}", flush=True)
                row["tols"].append({"tol": tol, "iters": abs(code),
                                    "converged": code < 0, "true_rel": tr})

            # Convergence CURVE: tol unreachable, so each run stops at maxiter.
            for mi in [int(v) for v in cli.iter_ladder.split(",")]:
                t0 = time.perf_counter()
                u, info = seq.apply_inverse_hodge_laplacian(
                    rhs, k, dirichlet=dbc, operators=ops, tol=1e-30,
                    maxiter=mi, preconditioner=spec, return_info=True)
                tr = true_rel(u)
                print(f"  {('it<=%d' % mi):>12}{abs(int(info)):>8}"
                      f"{'-':>6}{tr:>34.4e}"
                      f"{time.perf_counter() - t0:>8.1f}", flush=True)
                row["ladder"].append({"maxiter": mi, "true_rel": tr})
            res["rows"].append(row)

    if cli.out:
        os.makedirs(os.path.dirname(cli.out) or ".", exist_ok=True)
        with open(cli.out, "w") as f:
            json.dump(res, f, indent=1)
        print(f"\nwrote {cli.out}", flush=True)


if __name__ == "__main__":
    main()
