"""How much did building the discarded Schur apply cost on the production path?

`_build_saddle_preconditioner` used to construct `schur_apply` before branching
on `schur.outer`, but `schur_apply` has exactly one consumer, in the `else`
branch. With the production default `outer='block'` the atom IS the upper-block
inverse, so the whole schur.inner construction -- raw_kron factors included --
was built and thrown away on every k>=1 saddle solve setup.

This times the build both ways. It does NOT test correctness: the value was
discarded, so removing it cannot change any answer. It measures the waste.

    python scripts/debug/schur_inner_waste.py --geometry w7x --ns 12,24,12
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mrx.operators as op  # noqa: E402
from verify_block_jacobi import build_sequence  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry", default="w7x")
    ap.add_argument("--ns", default="12,24,12")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--ks", default="1,2,3")
    ap.add_argument("--repeats", type=int, default=3)
    cli = ap.parse_args()
    ns = tuple(int(v) for v in cli.ns.split(","))

    seq, ops = build_sequence(cli.geometry, ns, cli.p, 10000)
    print(f"[setup] {cli.geometry} ns={ns} p={cli.p}", flush=True)

    for k in [int(v) for v in cli.ks.split(",")]:
        for dbc in (False, True):
            spec = op._materialize_default_saddle_preconditioner(
                seq, ops, k=k, dirichlet=dbc)
            assert spec.schur.outer.kind == 'block', spec.schur.outer.kind

            # The discarded half, timed on its own: this is exactly the call
            # that used to run before the branch.
            t = []
            for _ in range(cli.repeats):
                t0 = time.perf_counter()
                apply_ = op._build_schur_apply_from_saddle_preconditioner(
                    seq, ops, k=k, dirichlet=dbc, eps=0.0,
                    saddle_preconditioner=spec)
                n = int(getattr(seq, f"n{k}_dbc" if dbc else f"n{k}"))
                jax.block_until_ready(apply_(jnp.ones(n)))
                t.append(time.perf_counter() - t0)
            print(f"[waste] k={k} {'dbc ' if dbc else 'free'} "
                  f"schur.inner={spec.schur.inner.kind:10s} "
                  f"build+apply {min(t):7.3f}s (min of {cli.repeats})",
                  flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
