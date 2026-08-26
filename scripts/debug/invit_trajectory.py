"""Per-sweep trajectory of inverse iteration, to see HOW it fails.

The p=5 rerun showed a clean structural split: k=3 dbc improves ~10x on every
geometry in ~43 s, while k=1 free and k=2 dbc degrade -- sometimes by seven
orders -- and burn 4-5x the time. Three shapes would explain that and they are
distinguishable only by looking at the trajectory:

  monotone decay      the iteration is simply diverging (wrong method here)
  converge then drift it reaches a floor and then accumulates inner-solve noise
                      -- consistent with `maxiter` sweeps at an accuracy the
                      shifted solve cannot deliver
  single jump         one bad sweep, e.g. the harmonic coarse correction being
                      rebuilt from a polluted iterate

Rather than instrument inside the library, call it with maxiter = 1, 2, 3, ...
and record the Rayleigh quotient after each. Also reports what the solver
CLAIMS (`n_iters`, its own residual), since `abs_tol` is on ||L_k v|| -- an
unscaled DUAL norm -- and may not track the Rayleigh quotient at all.

    python scripts/debug/invit_trajectory.py --geometry w7x --k 1
"""
from __future__ import annotations

import argparse
import os
import sys
import time



sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mrx.operators as op  # noqa: E402
from mrx.nullspace import (  # noqa: E402
    compute_nullspaces, find_nullspace_vectors, get_nullspace,
)
from verify_block_jacobi import build_sequence  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry", default="w7x")
    ap.add_argument("--ns", default="12,24,12")
    ap.add_argument("--p", type=int, default=5)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--dbc", action="store_true")
    ap.add_argument("--eps", type=float, default=1e-4)
    ap.add_argument("--inner-tol", type=float, default=1e-13)
    ap.add_argument("--sweeps", default="1,2,3,4,6,8,12,20,40,100")
    cli = ap.parse_args()
    ns = tuple(int(v) for v in cli.ns.split(","))
    k, dbc = cli.k, cli.dbc

    seq, ops = build_sequence(cli.geometry, ns, cli.p, 10000)
    ops = compute_nullspaces(seq, ops)
    seq.set_operators(ops)
    v0 = get_nullspace(ops, k, dbc)[0]

    def rq(v):
        lv = op.apply_hodge_laplacian(seq, ops, v, k, dirichlet=dbc)
        mv = op.apply_mass_matrix(seq, ops, v, k, dirichlet=dbc)
        return abs(float(v @ lv)) / float(v @ mv)

    print(f"\n=== {cli.geometry} p={cli.p} k={k} dbc={dbc} eps={cli.eps:g} ===",
          flush=True)
    print(f"  seed (direct):  relL2 = {rq(v0) ** 0.5:.4e}", flush=True)
    print(f"  {'sweeps':>7}{'n_iters':>9}{'solver residual':>18}"
          f"{'relL2':>13}{'s':>8}", flush=True)
    for m in [int(x) for x in cli.sweeps.split(",")]:
        t0 = time.perf_counter()
        vecs, infos = find_nullspace_vectors(
            seq, ops, k, 1, cli.eps, dirichlet=dbc, x0s=[v0],
            inner_tol=cli.inner_tol, maxiter=m)
        # infos entries are (n_iters, residual, rayleigh)
        it, r = infos[0][0], infos[0][1]
        print(f"  {m:>7}{int(it):>9}{float(r):>18.4e}"
              f"{rq(vecs[0]) ** 0.5:>13.4e}{time.perf_counter() - t0:>8.1f}",
              flush=True)


if __name__ == "__main__":
    main()
