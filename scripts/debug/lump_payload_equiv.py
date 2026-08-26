"""Does the pytree-payload apply produce the SAME numbers as the closure one?

The eqx refactor is a performance change with no intended numerical effect, so
the first question is not "is it faster" but "is it the same". A preconditioner
that changed the answer would show up as a different harmonic form or a
different iteration count long after the change, and be attributed to anything
but this.

Compares the current payload apply against a locally reconstructed closure
apply -- the exact code that ran before -- on the same built factors.

    python scripts/debug/lump_payload_equiv.py --geometry toroid
"""
from __future__ import annotations

import argparse
import os
import sys

import jax


import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mrx.metric_lumping_laplacian import MetricLumpingLaplacian  # noqa: E402
from mrx.operators import _fd_apply_3d  # noqa: E402
from verify_block_jacobi import build_sequence  # noqa: E402


def legacy_apply(pre):
    """The pre-2026-08-25 closure apply, rebuilt verbatim for comparison."""
    blocks = []
    for blk in pre.blocks:
        if blk is None:
            continue
        nr, nt, nz = blk["shape"]
        ir, it, iz = blk["idx"]
        flat = jnp.asarray((ir * nt + it) * nz + iz)
        (v_r, v_t, v_z), (l_r, l_t, l_z), alpha = blk["atom"]
        blocks.append({
            "rows": jnp.asarray(blk["rows"]),
            "vals": jnp.asarray(blk["vals"]),
            "flat": flat, "shape": (nr, nt, nz),
            "v": (v_r, v_t, v_z), "lam": (l_r, l_t, l_z),
            "alpha": tuple(float(a) for a in alpha),
            "dscale": (None if blk["dscale"] is None
                       else jnp.asarray(blk["dscale"])),
        })
    core = jnp.asarray(pre.probe_rows)
    core_inv = jnp.asarray(pre.core_inv)
    has_core = np.asarray(pre.probe_rows).size > 0

    def m_apply(x):
        out = jnp.zeros_like(x)
        for b in blocks:
            buf = jnp.zeros(int(np.prod(b["shape"]))).at[b["flat"]].set(
                b["vals"] * x[b["rows"]]).reshape(b["shape"])
            if b["dscale"] is not None:
                buf = buf * b["dscale"]
            sol = _fd_apply_3d(*b["v"], *b["lam"], b["alpha"], buf)
            if b["dscale"] is not None:
                sol = sol * b["dscale"]
            out = out.at[b["rows"]].set(b["vals"] * sol.reshape(-1)[b["flat"]])
        if has_core:
            out = out.at[core].set(core_inv @ x[core])
        return out

    return jax.jit(m_apply)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry", default="toroid")
    ap.add_argument("--ns", default="8,16,8")
    ap.add_argument("--p", type=int, default=3)
    cli = ap.parse_args()
    ns = tuple(int(v) for v in cli.ns.split(","))

    seq, ops = build_sequence(cli.geometry, ns, cli.p, 2000)
    print(f"[setup] {cli.geometry} ns={ns} p={cli.p}", flush=True)

    worst = 0.0
    for k in (0, 1, 2, 3):
        for dbc in (False, True):
            side = "dbc " if dbc else "free"
            n = int(getattr(seq, f"n{k}_dbc" if dbc else f"n{k}"))
            pre = MetricLumpingLaplacian(seq, ops, k, dbc)
            rng = np.random.default_rng(11)
            v = jnp.asarray(rng.standard_normal(n))
            new = np.asarray(pre.apply(v))
            old = np.asarray(legacy_apply(pre)(v))
            rel = float(np.linalg.norm(new - old)
                        / max(np.linalg.norm(old), 1e-300))
            worst = max(worst, rel)
            flag = "OK" if rel < 1e-13 else "*** DIFFERS ***"
            print(f"[equiv] k={k} {side} n={n:7d}  |new - old|/|old| = "
                  f"{rel:.3e}  {flag}", flush=True)

    print(f"\n[result] worst relative difference {worst:.3e}", flush=True)
    print("[note ] bitwise equality is NOT expected: dscale is now always a "
          "multiply (ones where it used to be skipped) and alpha is an array "
          "rather than Python floats, so the operation ORDER differs slightly. "
          "Anything at round-off is the refactor being numerically inert.",
          flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
