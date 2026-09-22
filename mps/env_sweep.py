"""What the Metal backend's undocumented knobs are worth on a relaxation chunk.

A compiled MRX relaxation chunk is about 21000 StableHLO ops, of which only
1271 are ``dot_general``: the rest is slice, gather, reshape and concatenate,
data movement that XLA fuses away and that MLX may not. A backend that pays
a fixed cost per dispatched op would therefore be slow on MRX for reasons
that have nothing to do with arithmetic, which is what the per-apply floor
in ``docs/source/mps.md`` looks like.

MLX and the plugin both expose knobs for exactly this, none of them
documented upstream; they were read out of the shipped binary:

    MLX_MAX_OPS_PER_BUFFER   ops batched into one Metal command buffer
    MLX_MAX_MB_PER_BUFFER    the same bound by size
    MLX_METAL_FAST_SYNCH     cheaper completion signalling
    MLX_DISABLE_COMPILE      turn MLX's own kernel fusion off (an A/B)
    JAX_MPS_ASYNC_DISPATCH   do not block the host on each execution
    JAX_MPS_NO_OPTIMIZE      turn the plugin's IR passes off (an A/B)
    JAX_MPS_CACHE_LIMIT_BYTES  MLX buffer cache ceiling

Each configuration runs in its own process, because every one of these is
read when the library initialises. The workload is a real
``chunk_runner`` chunk, not a synthetic one, so a knob that helps here
helps a run.

``MLX_ENABLE_TF32`` is never set and is asserted off: MRX sets
``jax_default_matmul_precision='highest'`` because TF32 put a 19% error in
the W7-X map's ``dR/dtheta``, and a knob that buys speed by undoing that is
not a speedup.

    python mps/env_sweep.py                  # the default configurations
    python mps/env_sweep.py --ns 12,24,12 --p 3
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Sequence

#: ``(label, {env})`` of every configuration, baseline first. Keep the
#: baseline first: every ratio is against it.
CONFIGS: list[tuple[str, dict[str, str]]] = [
    ("baseline", {}),
    ("async_dispatch", {"JAX_MPS_ASYNC_DISPATCH": "1"}),
    ("ops_per_buffer=50", {"MLX_MAX_OPS_PER_BUFFER": "50"}),
    ("ops_per_buffer=200", {"MLX_MAX_OPS_PER_BUFFER": "200"}),
    ("ops_per_buffer=1000", {"MLX_MAX_OPS_PER_BUFFER": "1000"}),
    ("fast_synch", {"MLX_METAL_FAST_SYNCH": "1"}),
    ("ops=200+mb=1024", {"MLX_MAX_OPS_PER_BUFFER": "200",
                         "MLX_MAX_MB_PER_BUFFER": "1024"}),
    ("async+ops=200+synch", {"JAX_MPS_ASYNC_DISPATCH": "1",
                             "MLX_MAX_OPS_PER_BUFFER": "200",
                             "MLX_METAL_FAST_SYNCH": "1"}),
    ("no_optimize", {"JAX_MPS_NO_OPTIMIZE": "1"}),
    ("mlx_no_compile", {"MLX_DISABLE_COMPILE": "1"}),
]

#: Timed calls after the compiling one; the minimum is reported.
REPEATS = 3


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command line of the sweep.

    Args:
        argv: Arguments to parse; ``None`` reads :data:`sys.argv`.

    Returns:
        The parsed namespace.
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--geometry", default="data/wout_li383_low_res_reference.nc")
    ap.add_argument("--ns", default="8,12,12")
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=5,
                    help="steps per compiled chunk; small keeps the sweep short "
                         "and the per-step cost is flat in it (mps/scan_scaling.py)")
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--only", default=None,
                    help="run only configurations whose label contains this")
    return ap.parse_args(argv)


def time_chunk(geometry: str, ns: tuple[int, int, int], p: int,
               chunk: int, repeats: int) -> dict[str, float]:
    """Build a sequence and time one compiled relaxation chunk.

    Args:
        geometry: VMEC wout or GVEC state file.
        ns: Resolution of the sequence.
        p: Spline degree.
        chunk: Steps compiled into the scan.
        repeats: Compile-free calls; the fastest is kept.

    Returns:
        ``{"setup": s, "compile": s, "chunk": s, "per_step_ms": ms}``.
    """
    import jax

    from mrx.geometry import build_sequence
    from mrx.initial_conditions import initial_field
    from mrx.nullspace import compute_nullspaces
    from mrx.relaxation import TimeStepper, chunk_runner, initial_state

    assert os.environ.get("MLX_ENABLE_TF32", "0") in ("0", ""), \
        "MLX_ENABLE_TF32 undoes jax_default_matmul_precision='highest'"

    t0 = time.perf_counter()
    seq, _ = build_sequence(geometry, ns, p)
    compute_nullspaces(seq, gap_sweeps=0, verbose=False)
    B0, _ = initial_field(seq)
    ts = TimeStepper(seq=seq, cfl=0.5, history_size=1,
                     velocity_smoothing_order=1, velocity_smoothing_scale=None)
    state0 = initial_state(B0, ts, 1.0)
    jax.block_until_ready(state0.B_n)
    setup = time.perf_counter() - t0

    run = chunk_runner(ts, chunk)
    t0 = time.perf_counter()
    state, _ = run(state0, 0)
    jax.block_until_ready(state.B_n)
    compile_s = time.perf_counter() - t0

    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        state, _ = run(state0, 0)
        jax.block_until_ready(state.B_n)
        best = min(best, time.perf_counter() - t0)

    return {"setup": setup, "compile": compile_s, "chunk": best,
            "per_step_ms": 1e3 * best / chunk}


def main(argv: Sequence[str] | None = None) -> int:
    """Run every configuration in a subprocess and print the comparison.

    Returns:
        Process exit status, 0 unless the baseline itself failed.
    """
    args = parse_args(argv)
    ns = tuple(int(x) for x in args.ns.split(","))

    if os.environ.get("MPS_SWEEP_CHILD"):
        print("RESULT " + json.dumps(
            time_chunk(args.geometry, ns, args.p, args.chunk, args.repeats)))
        return 0

    results: dict[str, dict[str, float]] = {}
    for label, env in CONFIGS:
        if args.only and args.only not in label:
            continue
        child = {**os.environ, "MPS_SWEEP_CHILD": "1", "JAX_PLATFORMS": "mps",
                 "MRX_X64": "0", **env}
        child.pop("MLX_ENABLE_TF32", None)
        done = subprocess.run([sys.executable, "-u", __file__, *(argv or sys.argv[1:])],
                              env=child, capture_output=True, text=True)
        row = next((json.loads(ln[7:]) for ln in done.stdout.splitlines()
                    if ln.startswith("RESULT ")), None)
        if row is None:
            print(f"  {label:<22} FAILED\n{done.stderr[-800:]}", flush=True)
            continue
        results[label] = row
        print(f"  {label:<22} compile {row['compile']:7.2f} s   "
              f"{row['per_step_ms']:8.1f} ms/step", flush=True)

    if "baseline" not in results:
        return 1
    base = results["baseline"]["per_step_ms"]
    print("\n=== against baseline ====================================")
    for label, row in sorted(results.items(), key=lambda kv: kv[1]["per_step_ms"]):
        speedup = base / row["per_step_ms"]
        print(f"  {label:<22} {row['per_step_ms']:8.1f} ms/step   {speedup:5.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
