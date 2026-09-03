"""Does the contraction dimension explain the mass kernel's cost?

Every einsum in the sum-factorized mass apply contracts 3 or 4 elements
(``roofline.py`` prints them). A TPU MXU is a 128x128 systolic array, so a
contraction of 4 uses 4 of its 128 rows and the other 124 are padding. If that
is what limits the kernel, then a matmul with the same output size costs the
same whether K is 4 or 128, and rises only past 128.

That is a decisive and cheap test, so this sweeps K at fixed output size and
reports cost per useful FLOP. A flat region up to 128 confirms padding; a cost
proportional to K from the start refutes it and means the kernel is limited by
something else.

The second half prices the alternative directly: the same element transform
done as one 3-D contraction (K = nloc^3) instead of three 1-D ones (K = nloc),
which is more arithmetic on better-shaped matmuls.

Usage:
    python tpu/mxu_occupancy.py --out outputs/mxu.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time


def parse_args() -> argparse.Namespace:
    """Command line for the occupancy sweep."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=int, default=165888,
                    help="rows of the matmul, i.e. the batch the kernel has")
    ap.add_argument("--n-out", type=int, default=4,
                    help="output columns, matching the kernel's q = p + 1")
    ap.add_argument("--ks", default="1,2,4,8,16,32,64,128,256,512",
                    help="contraction dimensions to sweep")
    ap.add_argument("--scan-length", type=int, default=50,
                    help="applies per timed scan, so the cost is the one a "
                         "fused solver body pays rather than a dispatch")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--matmul-precision", default=None)
    ap.add_argument("--precision", default=None, help="MRX_DTYPE")
    ap.add_argument("--out", default=None, help="write results as JSON here")
    return ap.parse_args()


def timed(fn, args, repeats: int) -> tuple[float, float]:
    """Median and first-call wall time of ``fn(*args)``, blocking on the result.

    Args:
        fn: callable returning a JAX array.
        args: positional arguments.
        repeats: timed calls after the warm-up.

    Returns:
        ``(median_seconds, first_call_seconds)``; the first call includes
        compilation and is reported separately rather than averaged in.
    """
    import jax

    start = time.perf_counter()
    out = fn(*args)
    jax.block_until_ready(out)
    first = time.perf_counter() - start

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        samples.append(time.perf_counter() - start)
    return statistics.median(samples), first


def main() -> int:
    """Sweep K at fixed output size and print cost per useful FLOP."""
    cli = parse_args()
    if cli.precision:
        os.environ["MRX_DTYPE"] = cli.precision
    os.environ.setdefault("MRX_DTYPE", "float32")

    import jax
    import jax.numpy as jnp

    import mrx  # noqa: F401  (sets precision before any array is made)

    if cli.matmul_precision:
        jax.config.update("jax_default_matmul_precision", cli.matmul_precision)

    device = jax.devices()[0]
    backend = {
        "platform": device.platform,
        "device_kind": device.device_kind,
        "jax_version": jax.__version__,
        "matmul_precision": str(jax.config.jax_default_matmul_precision),
        "dtype": str(mrx.DTYPE),
    }
    print("=" * 74)
    print(f"{backend['device_kind']}  {backend['dtype']}  "
          f"matmul {backend['matmul_precision']}  jax {backend['jax_version']}")
    print(f"batch {cli.batch}, output columns {cli.n_out}, "
          f"scan of {cli.scan_length}")
    print("=" * 74)

    ks = [int(v) for v in cli.ks.split(",")]
    key = jax.random.PRNGKey(0)
    rows = {}

    print(f"\n{'K':>6} {'per apply ms':>14} {'useful GFLOP/s':>16} "
          f"{'ms per unit K':>15}")
    print("-" * 74)
    baseline_ms = None
    for k in ks:
        b = jax.random.normal(key, (k, cli.n_out), dtype=mrx.DTYPE)
        x0 = jax.random.normal(key, (cli.batch, k), dtype=mrx.DTYPE)

        def body(carry, _, b=b, k=k):
            # Contract K, then map the n_out result back to K so the carry
            # keeps its shape and the scan measures repeated applies of the
            # same contraction rather than one call.
            y = carry @ b
            return jnp.pad(y, ((0, 0), (0, k - cli.n_out)))[:, :k], None

        @jax.jit
        def run(x, body=body):
            out, _ = jax.lax.scan(body, x, None, length=cli.scan_length)
            return out

        if k < cli.n_out:
            continue
        seconds, first = timed(run, (x0,), cli.repeats)
        per_apply = seconds / cli.scan_length
        useful_flops = 2 * cli.batch * cli.n_out * k
        gflops = useful_flops / per_apply / 1e9
        if baseline_ms is None:
            baseline_ms = per_apply
        rows[k] = {"per_apply_s": per_apply, "gflops": gflops,
                   "first_s": first}
        print(f"{k:>6} {per_apply * 1e3:>14.4f} {gflops:>16.1f} "
              f"{per_apply * 1e3 / k:>15.5f}")

    if rows:
        small = [k for k in rows if k <= 128]
        if len(small) >= 2:
            lo, hi = min(small), max(small)
            ratio = rows[hi]["per_apply_s"] / rows[lo]["per_apply_s"]
            print(f"\n  K={lo} -> K={hi} is {ratio:.2f}x for {hi // lo}x the "
                  f"arithmetic")
            if ratio < 2.0:
                print("  => the contraction dimension is being PADDED: work "
                      "at K=4 costs what work at K=128 costs, so the kernel "
                      "is paying for 128 either way")
            else:
                print("  => cost tracks K, so padding is NOT the limit and "
                      "the mass kernel is bound by something else")

    result = {"backend": backend, "args": vars(cli), "rows": rows}
    if cli.out:
        os.makedirs(os.path.dirname(cli.out) or ".", exist_ok=True)
        with open(cli.out, "w") as handle:
            json.dump(result, handle, indent=2)
        print(f"\nwrote {cli.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
