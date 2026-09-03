#!/usr/bin/env python3
"""Is the v5e's 40x deficit the indexed gather, or the arithmetic?

The matvec baseline puts the whole TPU-versus-GPU gap in one place. In the
scan form at li383 ``(12,24,12)`` p=3 float32, per apply:

    mass_core_apply k=1        v5e 6.54 ms   H200 0.135 ms    48x
    metric-lumped mass atom    v5e 0.065 ms  H200 0.039 ms   1.7x
    fast-diagonalisation atom  v5e 0.071 ms  H200 0.084 ms   0.9x

The preconditioner atoms are at parity or better. They are separable tensor
contractions on dof-sized arrays. ``mass_core_apply`` is the same kind of
contraction, but it first expands its input to the quadrature grid through
``x[gather_idx]`` -- 3168 values become 124 416 at k=1 -- and writes the
result back through a structured accumulate.

Arithmetic cannot be the explanation: the k=1 contraction is a few million
flops, which is microseconds at any plausible rate, and setting
``jax_default_matmul_precision`` from 'highest' to 'high' (six bf16 passes
down to three) moved it by 3%.

So this splits the kernel's front half in two and times both at the real
shapes, inside a scan:

    with_gather     _to_quadrature(B, v, gather_idx): gather then 3 einsums
    without_gather  the same 3 einsums on an ALREADY-expanded array

Same einsums, same output shape, same dtype; the only difference is whether
the quadrature values are produced by an indexed read. If ``without_gather``
is dramatically faster on a v5e and roughly the same on a GPU, the deficit is
the indexed read and the fix is a structured, index-free gather -- the mirror
of the ``_structured_accumulate`` that already replaced the scatter.

    python -u gather_cost.py --ns 12,24,12 --p 3
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
    ap.add_argument("--scan-length", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--map-batch-size", type=int, default=None)
    ap.add_argument("--out", default="gather_cost.json")
    return ap.parse_args()


def main() -> None:
    cli = parse_args()
    os.environ.setdefault("MRX_DTYPE", cli.precision)

    import jax
    import jax.numpy as jnp
    import numpy as np

    import mrx
    from mrx.geometry import build_sequence
    from mrx.mass import _flat_dof_plan, _form_bases, _from_quadrature

    if cli.map_batch_size is not None:
        mrx.MAP_BATCH_SIZE_INNER = cli.map_batch_size

    dev = jax.devices()[0]
    print(f"[env] mrx {mrx.DTYPE} on {dev.device_kind} x{len(jax.devices())}",
          flush=True)

    ns = tuple(int(v) for v in cli.ns.split(","))
    t0 = time.perf_counter()
    seq, _ = build_sequence(cli.geometry, ns, cli.p)
    print(f"[setup] {ns} p={cli.p}  {time.perf_counter() - t0:.0f}s", flush=True)

    N = cli.scan_length
    results: dict[str, dict] = {}

    def scan_time(f, x):
        """Per-apply seconds for ``f`` chained N times inside one jitted scan."""
        @jax.jit
        def run(v):
            def body(carry, _):
                y = f(carry)
                return y / (jnp.linalg.norm(y) + jnp.finfo(mrx.DTYPE).tiny), None
            out, _ = jax.lax.scan(body, v, None, length=N)
            return out

        jax.block_until_ready(run(x))
        times = []
        for _ in range(cli.repeats):
            t = time.perf_counter()
            jax.block_until_ready(run(x))
            times.append((time.perf_counter() - t) / N)
        return min(times)

    for k in (1, 2):
        form, comp, n_comp = _form_bases(seq, k)
        Bx, gx, By, gy, Bz, gz = comp[0]
        shape = form.shape[0]
        n_raw = int(np.prod(shape))
        gidx = _flat_dof_plan(gx, gy, gz, shape)
        B = (Bx, By, Bz)

        rng = np.random.default_rng(50 + k)
        x = jnp.asarray(rng.standard_normal(n_raw).astype(mrx.DTYPE))
        expanded = jnp.asarray(x[gidx])
        n_quad = int(expanded.size)

        # Both variants carry the same raw vector, expand it to the same
        # element-local shape, run the same six einsums, and close with the
        # same contiguous slice. The ONLY difference is how the expansion is
        # done: an indexed read against the real gather plan, or a broadcast
        # and reshape, which is the same number of output values written by
        # a structured copy.
        local_shape = tuple(int(s) for s in gidx.shape)
        n_local = int(np.prod(local_shape))
        reps = -(-n_local // n_raw)                      # ceil

        def contract(x_local, _B=B):
            """The six einsums of the kernel, on an already-expanded array."""
            Bx, By, Bz = _B
            t1 = jnp.einsum('xqb,xyzbdf->xyzqdf', Bx, x_local)
            t2 = jnp.einsum('yrd,xyzqdf->xyzqrf', By, t1)
            u = jnp.einsum('zsf,xyzqrf->xyzqrs', Bz, t2)
            return _from_quadrature(_B, u)

        def with_gather(v, _g=gidx):
            return contract(v[_g]).reshape(-1)[:n_raw]

        def without_gather(v):
            tiled = jnp.broadcast_to(v, (reps, n_raw)).reshape(-1)[:n_local]
            return contract(tiled.reshape(local_shape)).reshape(-1)[:n_raw]

        t_with = scan_time(with_gather, x)
        t_without = scan_time(without_gather, x)

        results[f"k={k}"] = {
            "n_raw": n_raw, "n_quadrature": n_quad,
            "expansion": n_quad / n_raw,
            "with_gather_s": t_with, "without_gather_s": t_without,
            "ratio": t_with / t_without if t_without else None,
        }
        print(f"\n  k={k}  component 0, n_raw={n_raw} -> {n_quad} "
              f"({n_quad / n_raw:.0f}x expansion)", flush=True)
        print(f"    gather + einsums   {t_with * 1e3:9.4f} ms per apply",
              flush=True)
        print(f"    einsums only       {t_without * 1e3:9.4f} ms per apply",
              flush=True)
        print(f"    the indexed read costs {t_with / t_without:6.1f}x the "
              f"arithmetic", flush=True)

    with open(cli.out, "w") as fh:
        json.dump({"device": dev.device_kind, "dtype": str(mrx.DTYPE),
                   "ns": list(ns), "p": cli.p, "scan_length": N,
                   "results": results}, fh, indent=2)
    print(f"\nwrote {cli.out}", flush=True)


if __name__ == "__main__":
    main()
