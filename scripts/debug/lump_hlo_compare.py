"""Is the argument-passing apply a SLOWER PROGRAM, or just a slower call?

Two mechanisms can produce the same per-apply regression and they have
opposite consequences:

  (1) PER-CALL DISPATCH -- Python-side marshalling of N array arguments.
      Scales with argument COUNT. Fixable by concatenating leaves.
  (2) LOST CONSTANT-FOLDING -- in the closure version `alpha`, `lam`, `v` and
      `dscale` were compile-time constants, so XLA could fold them: a multiply
      by a known 1.0 vanishes, scalars propagate into fused kernels, strides
      specialise. As runtime arguments they are opaque buffers and none of that
      is available, so the COMPILED PROGRAM is slower however few arguments
      there are. NOT fixable -- an array cannot be a compile-time constant and
      changeable without recompiling. Those are the same property.

Argument count separates them only if the two versions differ in count alone.
This settles it directly instead: compile BOTH programs and count HLO
operations. If the argument version has materially more ops, folding is the
mechanism and no amount of leaf-count engineering will close the gap.

    python scripts/debug/lump_hlo_compare.py --geometry toroid --k 1
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mrx.metric_lumping_laplacian import MetricLumpingLaplacian  # noqa: E402
from mrx.operators import _fd_apply_3d  # noqa: E402
from verify_block_jacobi import build_sequence  # noqa: E402


def closure_apply(pre):
    """The baseline: every array a CLOSURE CONSTANT, so XLA may fold them."""
    blocks = []
    for blk in pre.blocks:
        if blk is None:
            continue
        nr, nt, nz = blk["shape"]
        ir, it, iz = blk["idx"]
        (v_r, v_t, v_z), (l_r, l_t, l_z), alpha = blk["atom"]
        blocks.append({
            "rows": jnp.asarray(blk["rows"]), "vals": jnp.asarray(blk["vals"]),
            "flat": jnp.asarray((ir * nt + it) * nz + iz),
            "shape": (nr, nt, nz), "v": (v_r, v_t, v_z),
            "lam": (l_r, l_t, l_z), "alpha": tuple(float(a) for a in alpha),
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


def hlo_stats(lowered_text):
    ops = len(re.findall(r"^\s+%?\S+ = ", lowered_text, flags=re.M))
    fusions = len(re.findall(r"fusion", lowered_text))
    consts = len(re.findall(r"constant\(", lowered_text))
    return ops, fusions, consts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry", default="toroid")
    ap.add_argument("--ns", default="8,16,8")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--k", type=int, default=1)
    cli = ap.parse_args()
    ns = tuple(int(v) for v in cli.ns.split(","))

    seq, ops_b = build_sequence(cli.geometry, ns, cli.p, 2000)
    k, dbc = cli.k, False
    n = int(getattr(seq, f"n{k}"))
    v = jnp.asarray(np.random.default_rng(0).standard_normal(n))
    pre = MetricLumpingLaplacian(seq, ops_b, k, dbc)
    print(f"[setup] {cli.geometry} ns={ns} p={cli.p} k={k} n={n}", flush=True)

    # ARGUMENT version: the current production path.
    pre.apply(v)
    leaves, jitted = pre._flat
    arg_text = jitted.lower(leaves, v).compile().as_text()

    # CONSTANT version: the pre-refactor closure.
    clo = closure_apply(pre)
    clo_text = clo.lower(v).compile().as_text()

    a_ops, a_fus, a_con = hlo_stats(arg_text)
    c_ops, c_fus, c_con = hlo_stats(clo_text)

    print(f"[hlo  ] CONSTANT (closure) ops={c_ops:5d} fusions={c_fus:4d} "
          f"constants={c_con:5d}", flush=True)
    print(f"[hlo  ] ARGUMENT (payload) ops={a_ops:5d} fusions={a_fus:4d} "
          f"constants={a_con:5d}", flush=True)
    print(f"[hlo  ] delta            ops={a_ops - c_ops:+5d} "
          f"fusions={a_fus - c_fus:+4d} constants={a_con - c_con:+5d}",
          flush=True)
    print(f"[hlo  ] arg-version leaf count = {len(leaves)}", flush=True)

    print("\n[read ] MATERIALLY MORE OPS in the ARGUMENT version means the "
          "COMPILED PROGRAM is slower -- lost constant-folding, mechanism (2), "
          "NOT fixable by reducing leaf count.", flush=True)
    print("[read ] COMPARABLE OPS means the programs are the same and the cost "
          "is per-call dispatch -- mechanism (1), fixable.", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
