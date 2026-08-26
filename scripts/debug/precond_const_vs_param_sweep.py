"""Constants or parameters? The trade as a function of RESOLUTION.

Everything measured so far was at ns=(8,16,8) -- the SMALLEST size we test --
and treated as "the" trade. Tobias reports that the preconditioner payload is
O(n^2) and that XLA already emits CONSTANT-FOLDING WARNINGS at production
resolution. Both cut against the single-point conclusion:

  * the folded literal grows with the payload, so the constants arm's compile
    cost should grow with n -- plausibly worse than the payload does, since the
    literal is embedded in the compiled program;
  * the parameters arm's penalty is per-call marshalling and lost fusion, which
    need not scale the same way at all.

So the question is NOT "which is faster" but "HOW DOES THE CROSSOVER MOVE WITH
n". A conclusion drawn at one point on a curve is the classic thing that
reverses when the x-axis is extended.

Both arms, per resolution:

    payload size (bytes and leaf count)
    CONSTANTS  : recurring compile, steady apply, AND WHETHER XLA WARNS
    PARAMETERS : recurring compile, steady apply
    break-even applies

THE XLA WARNING IS A DATUM, NOT NOISE. It is Tobias's actual symptom, it comes
from C++ on fd 2, and Python-level stderr redirection does not see it -- so
this captures the file descriptor.

    python scripts/debug/precond_const_vs_param_sweep.py --geometry w7x
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys
import tempfile
import time

import jax


import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mrx.metric_lumping_laplacian import MetricLumpingLaplacian  # noqa: E402
from mrx.operators import _fd_apply_3d  # noqa: E402
from verify_block_jacobi import build_sequence  # noqa: E402

_FOLD_MARKERS = ("Constant folding an instruction is taking",
                 "constant folding", "Constant folding")


@contextlib.contextmanager
def capture_fd2():
    """Capture C++-level stderr. absl/XLA warnings do not go through Python."""
    saved = os.dup(2)
    with tempfile.TemporaryFile(mode="w+b") as tmp:
        os.dup2(tmp.fileno(), 2)
        try:
            yield tmp
        finally:
            os.dup2(saved, 2)
            os.close(saved)
            tmp.flush()


def closure_apply(pre):
    """CONSTANTS arm: every array a closure constant, foldable by XLA."""
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


def payload_bytes(pre):
    leaves = jax.tree_util.tree_leaves(pre._build_payload())
    total = sum(int(np.asarray(v).nbytes) for v in leaves)
    return total, len(leaves)


def timed(fn, v, repeats=100):
    jax.block_until_ready(fn(v))
    t0 = time.perf_counter()
    for _ in range(repeats):
        out = fn(v)
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / repeats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry", default="w7x")
    ap.add_argument("--resolutions", default="8,16,8;12,24,12;16,32,16")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--k", type=int, default=1)
    cli = ap.parse_args()
    k, dbc = cli.k, False

    print(f"[note ] {cli.geometry}, k={k}, p={cli.p}. The XLA constant-folding "
          "warning is Tobias's actual symptom and is captured from fd 2.",
          flush=True)

    for spec in cli.resolutions.split(";"):
        ns = tuple(int(v) for v in spec.split(","))
        print(f"\n[res  ] ns={ns}", flush=True)
        try:
            seq, ops = build_sequence(cli.geometry, ns, cli.p, 2000)
            n = int(getattr(seq, f"n{k}"))
            v = jnp.asarray(np.random.default_rng(0).standard_normal(n))
            pre = MetricLumpingLaplacian(seq, ops, k, dbc)
            nbytes, nleaves = payload_bytes(pre)
            print(f"[size ] n={n:8d}  payload {nbytes/1e6:8.2f} MB  "
                  f"{nleaves:3d} leaves", flush=True)

            # --- CONSTANTS arm, watching for the fold warning ---------------
            with capture_fd2() as tmp:
                t0 = time.perf_counter()
                fn = closure_apply(pre)
                jax.block_until_ready(fn(v))
                t_const_compile = time.perf_counter() - t0
                tmp.seek(0)
                err = tmp.read().decode("utf-8", "replace")
            warned = any(m in err for m in _FOLD_MARKERS)
            # POSITIVE CONTROL for the capture itself. "no warning" is only a
            # result if the capture demonstrably captures SOMETHING; an empty
            # buffer cannot distinguish "XLA said nothing" from "fd 2 was not
            # actually redirected", which is the same false-negative class this
            # instrument exists to avoid.
            print(f"[capt ] fd2 captured {len(err)} bytes during compile"
                  + ("" if err else "  *** EMPTY: capture unverified, treat "
                                    "the warning column as UNKNOWN ***"),
                  flush=True)
            t_const = timed(fn, v)

            # --- PARAMETERS arm (v2) ---------------------------------------
            pre2 = MetricLumpingLaplacian(seq, ops, k, dbc)
            t0 = time.perf_counter()
            jax.block_until_ready(pre2.apply(v))
            t_param_compile = time.perf_counter() - t0
            t_param = timed(pre2.apply, v)

            dt = t_param - t_const
            saved = t_const_compile - t_param_compile
            be = (saved / dt) if dt > 0 else float("inf")

            print(f"[const] compile {t_const_compile*1e3:9.1f} ms   apply "
                  f"{t_const*1e6:8.1f} us   XLA fold warning: "
                  f"{'YES' if warned else 'no'}", flush=True)
            print(f"[param] compile {t_param_compile*1e3:9.1f} ms   apply "
                  f"{t_param*1e6:8.1f} us", flush=True)
            print(f"[trade] constants save {dt*1e6:7.1f} us/apply, cost "
                  f"{saved*1e3:9.1f} ms/change  ->  break-even "
                  f"{be:,.0f} applies", flush=True)
            if warned:
                line = next((ln for ln in err.splitlines()
                             if any(m in ln for m in _FOLD_MARKERS)), "")
                print(f"[warn ] {line.strip()[:150]}", flush=True)
        except Exception as exc:                              # noqa: BLE001
            print(f"[res  ] ns={ns} FAILED: {type(exc).__name__}: {exc}",
                  flush=True)

    print("\n[read ] The question is not which is faster at one n, but how the "
          "break-even MOVES with n. A break-even that collapses as n grows "
          "means parameters win everywhere that matters.", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
