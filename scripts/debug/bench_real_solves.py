"""Benchmark the REAL k>=1 Laplacian solve: `apply_inverse_hodge_laplacian`.

Every bc-alpha number to date came from a bespoke CG in `verify_block_jacobi.py`
iterating on `apply_hodge_laplacian_approx` (S_k + D B D^T, B = one mass-
preconditioner apply). That is not what the library does. The library's k>=1
Laplacian solve is `apply_inverse_hodge_laplacian` -> `solve_saddle_point_minres`
on the EXACT system

    | S_k      D_{k-1} | | u |   | rhs |
    | D^T     -M_{k-1} | | s | = |  0  |

which is symmetric indefinite (MINRES, not CG) and needs no nested mass solve --
sigma is a variable, so M_{k-1} appears as a block and is never inverted. This
script sweeps THAT solve.

The preconditioner is block-diagonal, SPD as MINRES requires, and its upper
block preconditions the Schur complement -- which IS L_k, so the block-Jacobi
atom belongs there. Until 2026-08-24 `schur.outer` accepted only
('none','jacobi','exact_jacobi'): the production Laplacian preconditioner was
unreachable from the production solve. Arms here compare that.

Iteration counts come from the solver itself: `minres` returns
`info = -k` when converged and `+k` when not (`mrx/solvers.py:516`); the
docstring there says "0 if converged" and is stale.

    python scripts/debug/bench_real_solves.py --geometry w7x --ks 1,2,3
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mrx.operators as op  # noqa: E402
from mrx.nullspace import compute_nullspaces  # noqa: E402
from mrx.preconditioners import (  # noqa: E402
    MassPreconditionerSpec, SaddlePointPreconditionerSpec,
    SchurPreconditionerSpec,
)
from verify_block_jacobi import build_sequence  # noqa: E402

#: Arm names: "minres_jacobi" is the jacobi-outer baseline (no block atom);
#: "minres_block_sNNNN" uses the block atom as schur.outer with the natural-BC
#: scale NNNN/1000, so "minres_block_s3000" is the shipped s = 3.
def arm_scale(arm):
    if arm == "minres_jacobi":
        return None
    m = re.match(r"^minres_block_s(\d+)$", arm)
    if not m:
        raise ValueError(f"unknown arm {arm!r}")
    return str(int(m.group(1)) / 1000.0)


def spec_for(arm, seq, ops, k):
    if arm_scale(arm) is None:
        return 'auto'          # the library default: schur.outer = jacobi
    return SaddlePointPreconditionerSpec(
        mass=op._materialize_default_mass_preconditioner(seq, ops, k=k - 1),
        schur=SchurPreconditionerSpec(
            inner=MassPreconditionerSpec(kind='raw_kron'),
            outer=MassPreconditionerSpec(kind='block'),
        ),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry", default="toroid")
    ap.add_argument("--ns", default="12,24,12")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--ks", default="1,2,3")
    ap.add_argument("--bcs", default="free,dbc")
    ap.add_argument("--arms", default="minres_jacobi,minres_block_s3000")
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--maxiter", type=int, default=10000)
    ap.add_argument("--out", default=None)
    cli = ap.parse_args()
    ns = tuple(int(v) for v in cli.ns.split(","))
    arms = cli.arms.split(",")

    seq, ops = build_sequence(cli.geometry, ns, cli.p, cli.maxiter)
    t0 = time.perf_counter()
    ops = compute_nullspaces(seq, ops)
    seq.set_operators(ops)
    print(f"[setup] {cli.geometry} ns={ns} p={cli.p} nullspaces "
          f"{time.perf_counter() - t0:.1f}s", flush=True)

    results = {"geometry": cli.geometry, "ns": list(ns), "p": cli.p,
               "solver": "apply_inverse_hodge_laplacian (saddle MINRES)",
               "tol": cli.tol, "maxiter": cli.maxiter, "rows": []}

    print(f"\n{'k':>2} {'dbc':>5} {'n':>7} " +
          " ".join(f"{a:>26}" for a in arms), flush=True)
    for k in [int(v) for v in cli.ks.split(",")]:
        for dbc in (False, True):
            if ("dbc" if dbc else "free") not in cli.bcs.split(","):
                continue
            n = int(getattr(seq, f"n{k}_dbc" if dbc else f"n{k}"))
            rhs = jax.random.normal(jax.random.PRNGKey(31 * k + dbc), (n,))
            row = {"k": k, "dbc": dbc, "n": n}
            cells = []
            for arm in arms:
                scale = arm_scale(arm)
                if scale is None:
                    os.environ.pop("MRX_BJ_BC_SCALE", None)
                else:
                    os.environ["MRX_BJ_BC_SCALE"] = scale
                # The atom is cached on seq; drop it so each arm rebuilds with
                # its own bc settings rather than inheriting the previous arm's.
                if hasattr(seq, op.BLOCK_JACOBI_CACHE_ATTR):
                    delattr(seq, op.BLOCK_JACOBI_CACHE_ATTR)
                t1 = time.perf_counter()
                build_s = 0.0
                try:
                    if scale is not None:
                        op.assemble_block_jacobi_laplacian_preconditioner(
                            seq, ops, ks=(k,), dirichlets=(dbc,))
                        build_s = time.perf_counter() - t1
                    t2 = time.perf_counter()
                    _, info = seq.apply_inverse_hodge_laplacian(
                        rhs, k, dirichlet=dbc, operators=ops, tol=cli.tol,
                        maxiter=cli.maxiter,
                        preconditioner=spec_for(arm, seq, ops, k),
                        return_info=True)
                    code = int(info)
                    # negative == converged, |code| == iterations
                    ok, iters = code < 0, abs(code)
                    solve_s = time.perf_counter() - t2
                    row[arm] = {"iters": iters, "converged": ok,
                                "build_s": build_s, "solve_s": solve_s}
                    cells.append(f"{iters:>6}{'' if ok else '!':<1}"
                                 f"{solve_s:>7.1f}s{build_s:>7.1f}b")
                except Exception as exc:                      # noqa: BLE001
                    row[arm] = {"error": str(exc)[:200]}
                    cells.append(f"{'ERR':>26}")
            print(f"{k:>2} {dbc!s:>5} {n:>7} " +
                  " ".join(f"{c:>26}" for c in cells), flush=True)
            results["rows"].append(row)

    print("\n! = did not converge within maxiter", flush=True)
    if cli.out:
        os.makedirs(os.path.dirname(cli.out) or ".", exist_ok=True)
        with open(cli.out, "w") as f:
            json.dump(results, f, indent=1)
        print(f"wrote {cli.out}", flush=True)


if __name__ == "__main__":
    main()
