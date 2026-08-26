"""What does assembling the block-Jacobi atoms actually cost?

The question behind it: should a sequence assemble them eagerly instead of
leaving every call site to remember? That trades a fixed setup cost for the
class of bug this stack has now shipped twice -- a solve silently taking the
per-DoF diagonal because nobody called the assembler.

Times each (k, BC) separately so the answer is per-slot, not a lump, and reports
the mass preconditioner warm-up beside it for scale.

    python scripts/debug/atom_assembly_cost.py --geometry w7x --ns 12,24,12
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mrx.operators as op  # noqa: E402
from verify_block_jacobi import build_sequence  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry", default="w7x")
    ap.add_argument("--ns", default="12,24,12")
    ap.add_argument("--p", type=int, default=3)
    cli = ap.parse_args()
    ns = tuple(int(v) for v in cli.ns.split(","))

    t0 = time.perf_counter()
    seq, ops = build_sequence(cli.geometry, ns, cli.p, 10000)
    t_build = time.perf_counter() - t0
    print(f"[build] {cli.geometry} ns={ns} p={cli.p}  sequence+geometry "
          f"{t_build:.2f}s", flush=True)

    total = 0.0
    for k in (0, 1, 2, 3):
        for dbc in (False, True):
            n = int(getattr(seq, f"n{k}_dbc" if dbc else f"n{k}"))
            t = time.perf_counter()
            ops = op.assemble_metric_lumping_laplacian_preconditioner(
                seq, ops, ks=(k,), dirichlets=(dbc,))
            jax.block_until_ready(
                getattr(seq, op.METRIC_LUMPING_CACHE_ATTR)[(k, dbc)].apply(
                    jax.numpy.ones(n)))
            dt = time.perf_counter() - t
            total += dt
            print(f"[atom]  k={k} {'dbc ' if dbc else 'free'}  n={n:7d}  "
                  f"{dt:7.2f}s", flush=True)
    print(f"[atom]  ALL EIGHT {total:7.2f}s   "
          f"({100 * total / t_build:.1f}% of sequence build)", flush=True)

    t = time.perf_counter()
    op.warm_mass_preconditioner_cache(seq, ops)
    print(f"[mass]  warm_mass_preconditioner_cache {time.perf_counter() - t:.2f}s",
          flush=True)


if __name__ == "__main__":
    main()
