#!/usr/bin/env python3
"""Which way of timing one kernel is the one the relaxation pays?

Two published v5e numbers for the same ``mass_core_apply`` k=1 at li383
``(12,24,12)`` p=3 float32, same raw size 10 080, differ by 13x: 0.505 ms and
6.87 ms. The step model was built on the first, so it matters which is real.

The protocols differ in one thing only -- how many applies are in flight --
so this runs all four side by side:

    single      one call, blocked on its output
    batch50     50 calls on the SAME input, blocked once at the end
    batch50_var 50 calls on 50 DIFFERENT inputs, blocked once at the end
    scan50      50 applies chained inside one jitted lax.scan

``batch50`` is the old protocol and ``scan50`` is the shape the relaxation
runs. If ``batch50`` is fast and ``batch50_var`` is not, the old number was
measuring a repeated identical computation. If both are fast and ``scan50``
is not, it is the serial dependency, and only ``scan50`` may be multiplied by
an iteration count.

    python -u dispatch_ab.py --ns 12,24,12 --p 3
"""
from __future__ import annotations

import argparse
import json
import os
import time


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry", default="data/wout_li383_low_res_reference.nc")
    ap.add_argument("--ns", default="12,24,12")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--precision", default="float32")
    ap.add_argument("--n", type=int, default=50, help="applies per batch/scan")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--map-batch-size", type=int, default=None)
    ap.add_argument("--out", default="dispatch_ab.json")
    return ap.parse_args()


def main() -> None:
    cli = parse_args()
    os.environ.setdefault("MRX_DTYPE", cli.precision)

    import jax
    import jax.numpy as jnp

    import mrx
    from mrx.geometry import build_sequence
    from mrx.operators import apply_mass_matrix, mass_core_apply

    if cli.map_batch_size is not None:
        mrx.MAP_BATCH_SIZE_INNER = cli.map_batch_size

    dev = jax.devices()[0]
    print(f"[env] mrx {mrx.DTYPE} on {dev.device_kind} x{len(jax.devices())}",
          flush=True)

    ns = tuple(int(v) for v in cli.ns.split(","))
    t0 = time.perf_counter()
    seq, _ = build_sequence(cli.geometry, ns, cli.p)
    print(f"[setup] {ns} p={cli.p}  {time.perf_counter() - t0:.0f}s", flush=True)

    N = cli.n
    results: dict[str, dict] = {}

    def best(fn, n_applies):
        times = []
        for _ in range(cli.repeats):
            t = time.perf_counter()
            jax.block_until_ready(fn())
            times.append((time.perf_counter() - t) / n_applies)
        return min(times[1:]) if len(times) > 1 else times[0]

    def probe(label, f, x, xs):
        """``f`` timed four ways; ``xs`` is a stack of N distinct inputs."""
        f_jit = jax.jit(f)
        jax.block_until_ready(f_jit(x))

        def single():
            return f_jit(x)

        def batch_same():
            out = None
            for _ in range(N):
                out = f_jit(x)
            return out

        def batch_var():
            out = None
            for i in range(N):
                out = f_jit(xs[i])
            return out

        @jax.jit
        def scan_chain(v):
            def body(carry, _):
                y = f(carry)
                return y / (jnp.linalg.norm(y) + jnp.finfo(mrx.DTYPE).tiny), None
            out, _ = jax.lax.scan(body, v, None, length=N)
            return out

        jax.block_until_ready(scan_chain(x))

        row = {
            "single": best(single, 1),
            "batch50_same_input": best(batch_same, N),
            "batch50_varied_input": best(batch_var, N),
            "scan50": best(lambda: scan_chain(x), N),
        }
        results[label] = row
        print(f"\n  {label}", flush=True)
        for name, val in row.items():
            print(f"    {name:<24} {val * 1e3:9.4f} ms per apply", flush=True)
        ref = row["scan50"]
        print(f"    ratio scan/batch(same)   "
              f"{ref / row['batch50_same_input']:9.2f}x", flush=True)

    key = jax.random.PRNGKey(0)
    for k in (1, 2):
        core = mass_core_apply(seq, k)
        n_raw = int(seq.E(k, False).forward_shape[1])
        x = jax.random.normal(key, (n_raw,), dtype=mrx.DTYPE)
        xs = jax.random.normal(key, (N, n_raw), dtype=mrx.DTYPE)
        probe(f"mass_core_apply k={k} (raw n={n_raw})", core, x, xs)

        n_k = int(seq.n(k, True))
        xk = jax.random.normal(key, (n_k,), dtype=mrx.DTYPE)
        xks = jax.random.normal(key, (N, n_k), dtype=mrx.DTYPE)
        probe(f"apply_mass_matrix k={k} (n={n_k})",
              lambda v, _k=k: apply_mass_matrix(seq, v, _k, dirichlet=True),
              xk, xks)

    with open(cli.out, "w") as fh:
        json.dump({"device": dev.device_kind, "dtype": str(mrx.DTYPE),
                   "ns": list(ns), "p": cli.p, "n": N,
                   "results": results}, fh, indent=2)
    print(f"\nwrote {cli.out}", flush=True)


if __name__ == "__main__":
    main()
