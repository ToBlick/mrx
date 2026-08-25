"""raw_kron vs the metric-lumped atom as the Schur-Jacobi probe: A/B.

`assemble_schur_jacobi_preconditioner` probes and STORES 1/diag(A_k) with
A_k(x) = S_k x + D_{k-1} B_{k-1} D_{k-1}^T x, and B_{k-1} is the schur.inner
inverse. That inner was raw_kron; it is now the block-Jacobi mass atom (the
metric-lumped FD preconditioner that superseded the CP/ALS "tensor").

Three measurements, in the order they can invalidate each other:

1. LIVENESS. The two stored diagonals must DIFFER. A swap that silently does
   not take effect passes every correctness check perfectly and means nothing,
   so this is asserted before anything else is believed.
2. CORRECTNESS. A converged solve is preconditioner-independent, so the two
   arms must reach the SAME solution. |dx|/|x| above solver tolerance means the
   change is wrong regardless of how the iteration counts look.
3. MERIT. Iteration counts, with outer='jacobi' so the probed diagonal is what
   preconditions the solve. Under outer='block' the atom is the upper-block
   inverse directly and the probe is never consulted -- measuring there would
   be measuring nothing.

    python scripts/debug/schur_probe_ab.py --geometry w7x --ns 12,24,12
"""
from __future__ import annotations

import argparse
import os
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mrx.operators as op  # noqa: E402
from mrx.preconditioners import (  # noqa: E402
    MassPreconditionerSpec, SaddlePointPreconditionerSpec,
    SchurPreconditionerSpec,
)
from verify_block_jacobi import build_sequence  # noqa: E402


def spec_for(inner_kind, outer_kind):
    return SaddlePointPreconditionerSpec(
        mass=MassPreconditionerSpec(kind='block_jacobi'),
        schur=SchurPreconditionerSpec(
            inner=MassPreconditionerSpec(kind=inner_kind),
            outer=MassPreconditionerSpec(kind=outer_kind),
        ),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry", default="w7x")
    ap.add_argument("--ns", default="12,24,12")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--ks", default="1,2,3")
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--maxiter", type=int, default=20000)
    cli = ap.parse_args()
    ns = tuple(int(v) for v in cli.ns.split(","))

    seq, ops = build_sequence(cli.geometry, ns, cli.p, cli.maxiter)
    print(f"[setup] {cli.geometry} ns={ns} p={cli.p}", flush=True)

    for k in [int(v) for v in cli.ks.split(",")]:
        for dbc in (False, True):
            n = int(getattr(seq, f"n{k}_dbc" if dbc else f"n{k}"))
            side = "dbc " if dbc else "free"

            # (1) LIVENESS: probe the Schur diagonal both ways.
            diags = {}
            for kind in ("raw_kron", "block_jacobi"):
                apply_ = op._build_schur_apply_from_saddle_preconditioner(
                    seq, ops, k=k, dirichlet=dbc, eps=0.0,
                    saddle_preconditioner=spec_for(kind, 'none'))
                diags[kind] = np.asarray(op._diagonal_from_matvec(apply_, n))
            a, b = diags["raw_kron"], diags["block_jacobi"]
            rel = float(np.linalg.norm(a - b) / np.linalg.norm(a))
            live = rel > 1e-12
            print(f"[diag ] k={k} {side} n={n:7d}  |d_rk - d_atom|/|d_rk| = "
                  f"{rel:.3e}  {'LIVE' if live else '*** IDENTICAL: swap is a '
                                                    'no-op, nothing below means '
                                                    'anything ***'}", flush=True)
            if not live:
                continue

            # (2)+(3) CORRECTNESS and MERIT, with outer='jacobi' so the probed
            # diagonal is what preconditions the solve.
            rhs = jax.random.normal(jax.random.PRNGKey(7 * k + dbc), (n,))
            out = {}
            for kind in ("raw_kron", "block_jacobi"):
                x, info = op.apply_inverse_hodge_laplacian(
                    seq, ops, rhs, k, dirichlet=dbc, tol=cli.tol,
                    maxiter=cli.maxiter, return_info=True,
                    preconditioner=spec_for(kind, 'jacobi'))
                x.block_until_ready()
                out[kind] = (np.asarray(x), int(jnp.abs(jnp.asarray(info))),
                             int(jnp.asarray(info)) < 0)
            xr, ir, okr = out["raw_kron"]
            xb, ib, okb = out["block_jacobi"]
            dx = float(np.linalg.norm(xb - xr) / max(np.linalg.norm(xr), 1e-300))
            print(f"[solve] k={k} {side} raw_kron {ir:6d} it (conv {okr})   "
                  f"atom {ib:6d} it (conv {okb})   |dx|/|x| = {dx:.3e}",
                  flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
