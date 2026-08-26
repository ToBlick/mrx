"""S5 gate: is the k=1 free harmonic form good enough to deflate against?

`compute_nullspaces` builds it directly, and the result degrades with p --
W7-X relL2 8.4e-13 / 3.0e-04 / 1.7e-01 at p = 2/3/5 -- because the inner
L_2 free solve it depends on floors. Every k=1-free and k=2-dbc solve deflates
against that vector, so no singular-row iteration count means anything until it
is fixed.

The cure is documented (`tensor_preconditioners.md` §8): inverse iteration
SEEDED with the direct vector, at `inner_tol=1e-8`, two sweeps, landing at
~3e-24 independently of h. An earlier attempt here used a hand-rolled shifted
solve at eps=1e-4 instead of `find_nullspace_vectors` and diverged; use the
library function.

Reports the Rayleigh quotient q = v.Lv / v.Mv (relL2 = sqrt(q)) before and
after, for every (k, dbc) pair that carries a harmonic form.

    python scripts/debug/nullspace_polish_gate.py --geometry w7x --ps 2,3,5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time




sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mrx.operators as op  # noqa: E402
from mrx.nullspace import (  # noqa: E402
    compute_nullspaces, find_nullspace_vectors, get_nullspace,
)
from verify_block_jacobi import build_sequence  # noqa: E402

PAIRS = ((0, False), (1, False), (2, True), (3, True))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry", default="w7x")
    ap.add_argument("--ns", default="12,24,12")
    ap.add_argument("--ps", default="2,3,5")
    ap.add_argument("--eps", type=float, default=1e-4)
    # The inner solve sets a FLOOR on what inverse iteration can deliver: a
    # step cannot produce a residual below its own solve tolerance. inner_tol
    # 1e-8 is why two sweeps plateaued at ~1e-8 and why it moved BACKWARDS on
    # the cells whose direct vector was already at 1e-9. Iterate to
    # convergence, with an inner solve tight enough to allow it.
    ap.add_argument("--inner-tol", type=float, default=1e-13)
    ap.add_argument("--sweeps", type=int, default=100)
    ap.add_argument("--abs-tol", type=float, default=None,
                    help="stop when ||L_k v|| <= this; default lets the "
                         "library pick")
    ap.add_argument("--out", default=None)
    cli = ap.parse_args()
    ns = tuple(int(v) for v in cli.ns.split(","))
    res = {"geometry": cli.geometry, "ns": list(ns), "rows": []}

    for p in [int(v) for v in cli.ps.split(",")]:
        seq, ops = build_sequence(cli.geometry, ns, p, 10000)
        t0 = time.perf_counter()
        ops = compute_nullspaces(seq, ops)
        seq.set_operators(ops)
        t_direct = time.perf_counter() - t0
        print(f"\n=== {cli.geometry} p={p}  (direct {t_direct:.1f}s) ===",
              flush=True)
        print(f"  {'k':>2}{'dbc':>6}{'relL2 direct':>16}"
              f"{'relL2 polished':>17}{'s':>8}", flush=True)

        for k, dbc in PAIRS:
            vs = get_nullspace(ops, k, dbc)
            if vs.shape[0] == 0:
                continue

            def rq(v, k=k, dbc=dbc):
                lv = op.apply_hodge_laplacian(seq, ops, v, k, dirichlet=dbc)
                mv = op.apply_mass_matrix(seq, ops, v, k, dirichlet=dbc)
                return abs(float(v @ lv)) / float(v @ mv)

            q0 = rq(vs[0])
            t1 = time.perf_counter()
            try:
                # returns (vectors, infos); vectors is (n_vectors, n_k)
                polished, infos = find_nullspace_vectors(
                    seq, ops, k, 1, cli.eps, dirichlet=dbc,
                    x0s=[vs[0]], inner_tol=cli.inner_tol,
                    abs_tol=cli.abs_tol, maxiter=cli.sweeps)
                q1 = rq(polished[0])
                err = None
            except Exception as exc:                       # noqa: BLE001
                q1, err = float('nan'), str(exc)[:120]
            dt = time.perf_counter() - t1
            # infos is a list of (n_iters, residual) tuples
            n_it = ""
            try:
                it0, r0_ = infos[0]
                n_it = f"  it={it0} res={float(r0_):.1e}"
            except Exception:                                  # noqa: BLE001
                pass
            print(f"  {k:>2}{str(dbc):>6}{q0 ** 0.5:>16.4e}"
                  f"{q1 ** 0.5:>17.4e}{dt:>8.1f}{n_it}"
                  + (f"   {err}" if err else ""), flush=True)
            res["rows"].append({"p": p, "k": k, "dbc": dbc,
                                "relL2_direct": q0 ** 0.5,
                                "relL2_polished": q1 ** 0.5,
                                "polish_s": dt, "error": err})

    if cli.out:
        os.makedirs(os.path.dirname(cli.out) or ".", exist_ok=True)
        with open(cli.out, "w") as f:
            json.dump(res, f, indent=1)
        print(f"\nwrote {cli.out}", flush=True)


if __name__ == "__main__":
    main()
