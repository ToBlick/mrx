"""Does a relaxation step cost more when the chunk around it is longer?

``mrx.relaxation.chunk_runner`` compiles ``n_chunk`` steps into one
``lax.scan``, and the per-step cost ought to be flat in ``n_chunk``: the
work per step is the same and the scan only writes a few scalars of trace.
jax-mps [#215] reports that on the Metal backend the per-iteration cost of
a scan with stacked outputs grows with the trip count, which would make it
quadratic in the chunk and would explain why MRX's relaxation is the one
part of the suite the Apple GPU loses at while its Poisson solves win.

This times one compiled chunk at several lengths on both backends and
prints the per-step cost against the length. A flat CPU column beside a
rising MPS one is the effect; the slope is what a chunk costs per extra
step, and the crossing of the two columns is the chunk size worth running.

Per-step cost is read off the *slope* between successive lengths, the same
way ``scripts/benchmark/relaxation_bench.py`` does it, so the fixed
per-call overhead (dispatch, the carry in and out) does not contaminate it.

    python mps/scan_scaling.py                        # both backends
    python mps/scan_scaling.py --lengths 5,10,25,50   # a shorter sweep
    python mps/scan_scaling.py --backends mps         # one of them

[#215]: https://github.com/tillahoffmann/jax-mps/issues/215
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Sequence

os.environ.setdefault("MRX_X64", "0")

#: Default chunk lengths. Spread over a decade and a half, because the
#: question is the shape of the curve and not any one point.
LENGTHS = (5, 10, 25, 50, 100)

#: Timed calls per length after the compiling one. The minimum is reported:
#: a slow repeat is another process on the machine, never the backend.
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
    ap.add_argument("--ns", default="8,12,12",
                    help="resolution; the default is the test fixture's, which "
                         "compiles fast enough to sweep five lengths twice")
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--lengths", default=",".join(str(n) for n in LENGTHS),
                    help="comma-separated scan trip counts")
    ap.add_argument("--backends", default="cpu,mps",
                    help="comma-separated JAX platforms to sweep")
    ap.add_argument("--repeats", type=int, default=REPEATS)
    return ap.parse_args(argv)


def _sync(x) -> None:
    """Block until ``x`` is on the host, so a timing brackets real work."""
    import jax

    jax.block_until_ready(x)


def sweep_backend(platform: str, geometry: str, ns: tuple[int, int, int],
                  p: int, lengths: Sequence[int], repeats: int) -> dict[int, float]:
    """Time one compiled chunk at each length, on one backend.

    Runs in a fresh process (see :func:`main`), because ``JAX_PLATFORMS`` is
    read when the backend initialises and a second backend in the same
    process would share the compilation cache and the allocator.

    Args:
        platform: The JAX platform, ``"cpu"`` or ``"mps"``.
        geometry: VMEC wout or GVEC state file the sequence is built on.
        ns: Resolution of the sequence.
        p: Spline degree.
        lengths: Scan trip counts to time.
        repeats: Compile-free calls per length; the fastest is kept.

    Returns:
        ``{length: seconds for the whole chunk}``, the steady call.
    """
    import jax  # noqa: F401  (imported for its side effect on the backend)

    from mrx.geometry import build_sequence
    from mrx.initial_conditions import initial_field
    from mrx.nullspace import compute_nullspaces
    from mrx.relaxation import TimeStepper, chunk_runner, initial_state

    print(f"\n=== {platform} ===", flush=True)
    t0 = time.perf_counter()
    seq, _ = build_sequence(geometry, ns, p)
    compute_nullspaces(seq, gap_sweeps=0, verbose=False)
    B0, _ = initial_field(seq)
    # The stepper's own defaults, so the sweep cannot drift from production.
    ts = TimeStepper(seq=seq, cfl=0.5, history_size=1,
                     velocity_smoothing_order=1, velocity_smoothing_scale=None)
    state0 = initial_state(B0, ts, 1.0)
    _sync(state0.B_n)
    print(f"  setup {time.perf_counter() - t0:.0f} s", flush=True)

    out: dict[int, float] = {}
    for n in lengths:
        run = chunk_runner(ts, n)
        t0 = time.perf_counter()
        state, _ = run(state0, 0)
        _sync(state.B_n)
        first = time.perf_counter() - t0
        best = float("inf")
        for _ in range(repeats):
            t0 = time.perf_counter()
            state, _ = run(state0, 0)
            _sync(state.B_n)
            best = min(best, time.perf_counter() - t0)
        out[n] = best
        print(f"  n_chunk {n:>4}   compile {first:7.2f} s   chunk {best:7.3f} s"
              f"   {1e3 * best / n:8.2f} ms/step", flush=True)
    return out


def report(results: dict[str, dict[int, float]], lengths: Sequence[int]) -> None:
    """Print the per-step cost against the trip count, both backends.

    Two per-step numbers per cell, because they answer different questions:
    the ratio ``chunk / n`` is what a run of that chunk actually pays, and
    the slope between successive lengths is the marginal cost of a step with
    the fixed per-call overhead removed. A scan whose per-iteration cost is
    constant has a flat slope; jax-mps #215 predicts a rising one.

    Args:
        results: ``{platform: {length: chunk seconds}}``.
        lengths: The trip counts, in the order swept.
    """
    plats = list(results)
    print("\n=== ms per step, chunk/n (marginal slope) ================")
    header = "  n_chunk" + "".join(f"{p:>26}" for p in plats)
    print(header)
    for i, n in enumerate(lengths):
        row = f"  {n:>7}"
        for plat in plats:
            total = results[plat].get(n)
            if total is None:
                row += f"{'-':>26}"
                continue
            mean = 1e3 * total / n
            if i:
                prev = lengths[i - 1]
                slope = 1e3 * (total - results[plat][prev]) / (n - prev)
                row += f"{mean:>17.2f} ({slope:5.2f})"
            else:
                row += f"{mean:>17.2f} ({'-':>5})"
        print(row)

    if len(plats) == 2:
        a, b = plats
        print(f"\n=== {b} / {a} per step ==========================")
        for n in lengths:
            if n in results[a] and n in results[b]:
                ratio = results[b][n] / results[a][n]
                verdict = f"{b} faster" if ratio < 1 else f"{a} faster"
                print(f"  n_chunk {n:>4}   {ratio:5.2f}x   ({verdict})")


def main(argv: Sequence[str] | None = None) -> int:
    """Sweep every requested backend in its own subprocess and report.

    Returns:
        Process exit status, 0 unless a backend produced nothing.
    """
    import json
    import subprocess
    import sys

    args = parse_args(argv)
    lengths = [int(x) for x in args.lengths.split(",")]
    ns = tuple(int(x) for x in args.ns.split(","))

    if os.environ.get("MPS_SCAN_CHILD"):
        # One backend, this process. The parent reads the JSON tail.
        out = sweep_backend(os.environ["JAX_PLATFORMS"], args.geometry, ns,
                            args.p, lengths, args.repeats)
        print("RESULT " + json.dumps(out))
        return 0

    results: dict[str, dict[int, float]] = {}
    for plat in args.backends.split(","):
        child = {**os.environ, "MPS_SCAN_CHILD": "1", "JAX_PLATFORMS": plat,
                 "MRX_X64": "0"}
        done = subprocess.run([sys.executable, "-u", __file__, *(argv or sys.argv[1:])],
                              env=child, capture_output=True, text=True)
        for line in done.stdout.splitlines():
            if line.startswith("RESULT "):
                results[plat] = {int(k): v for k, v in json.loads(line[7:]).items()}
            else:
                print(line, flush=True)
        if plat not in results:
            print(f"  {plat} produced no result:\n{done.stderr[-2000:]}", flush=True)

    if not results:
        return 1
    report(results, lengths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
