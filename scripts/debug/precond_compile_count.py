"""Does changing the preconditioner payload force an XLA recompilation?

This measures the PREMISE of the proposed eqx.Module refactor before the
refactor is paid for. The claim being tested:

    MetricLumpingLaplacian._build_apply closes over its arrays and returns
    jax.jit(m_apply), so the arrays reach jit as CLOSURE CONSTANTS rather than
    arguments. A new preconditioner object is therefore a new closure and a new
    compilation, no matter where the object is stored. And `alpha` is baked as
    a Python float (`tuple(float(a) for a in alpha)`), which is where bc_scale
    lands -- so changing the BC scale should recompile by construction.

If today's counts are already 1, the premise is FALSE and the refactor buys
nothing. That is a useful answer, not a failed run.

Arms, each reporting compilations triggered by that arm alone. **The build AND
the first apply are both inside the counted region**: MetricLumpingLaplacian
compiles LAZILY on first apply, so counting construction alone measures zero
for every arm and looks exactly like "the premise is false".

  A  first build + apply                       CONTROL, must be > 0
  B  rebuild IDENTICAL config + apply          does a same-config rebuild recompile?
  C  rebuild after a GEOMETRY change + apply   same discretisation, different metric
  D  rebuild with a different bc_scale + apply the user-facing knob

A is a control, not data. A first build and apply MUST compile something, so
A == 0 means the COUNTER is broken and no other number in the run means
anything. The script says so and stops rather than reporting zeros that read
like a result.

WHY THIS CONTROL WORKS, and it is not the control itself: the protocol stated
A's EXPECTED VALUE IN ADVANCE ("first build + apply -> expect >= 1"). That is
what turns a null into a detectable instrument failure. Without the expectation
written down beforehand, A == 0 is just another zero in a column of zeros and
reads as corroboration. A control arm with no pre-stated expected value is one
more number that can quietly be whatever it is.

The first version of this script failed exactly that way: the applies sat
OUTSIDE the counted region, MetricLumpingLaplacian compiles lazily on first
apply, and every arm reported 0 -- which looks like a clean falsification of
the premise and would have cancelled the refactor on no evidence.

AND THE HARD PART: two SECONDARY outputs in that run were genuinely correct --
the identical rebuild agreed to 1.6e-14, and bc_scale=1.7 moved the apply by
2.3e-04, so arm D really was exercising something. A dead primary measurement
surrounded by healthy secondaries is much harder to catch than one that fails
alone, because everything around it says the run worked. Check the number that
CANNOT be zero, not the numbers that look reasonable.

Post-refactor, C and D must be 0 with the payload as leaves. If they are not,
the static/dynamic split is wrong and the refactor has not delivered.

    python scripts/debug/precond_compile_count.py --geometry toroid
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import jax


import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mrx.metric_lumping_laplacian import MetricLumpingLaplacian  # noqa: E402
from verify_block_jacobi import build_sequence  # noqa: E402

_MARKERS = ("Finished XLA compilation", "Compiling ", "Finished tracing",
            "compiling computation")


class CompileCounter(logging.Handler):
    """Count XLA compilation events emitted under jax.log_compiles.

    Attached to the ROOT logger: jax emits from several `jax._src.*` loggers
    and pinning to one name is a good way to count nothing.
    """

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.n = 0
        self.seen = []

    def emit(self, record):
        try:
            msg = record.getMessage()
        except Exception:                                    # noqa: BLE001
            return
        if any(m in msg for m in _MARKERS):
            self.n += 1
            if len(self.seen) < 6:
                self.seen.append(msg[:100])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry", default="toroid")
    ap.add_argument("--geometry-b", default="rot-ellipse",
                    help="second geometry, SAME discretisation")
    ap.add_argument("--ns", default="8,16,8")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--dbc", action="store_true")
    cli = ap.parse_args()
    ns = tuple(int(v) for v in cli.ns.split(","))
    k, dbc = cli.k, cli.dbc

    counter = CompileCounter()
    root = logging.getLogger()
    root.addHandler(counter)
    root.setLevel(logging.DEBUG)
    logging.getLogger("jax").setLevel(logging.DEBUG)

    results = {}

    def arm(label, build_fn, vec):
        """Build and first-apply, counted SEPARATELY.

        The split matters for the go/no-go. The refactor targets the APPLY's
        compilation -- turning closure constants into leaves so a rebuilt
        payload reuses the compiled apply. Build-side compilation is whatever
        `jnp` work the construction does (eigendecompositions and the like) and
        the refactor does not touch it.

        Reporting a single combined number would credit the refactor with
        removing build-side compiles it cannot remove. `apply` is the column
        that decides this.
        """
        before = counter.n
        pre = build_fn()
        n_build = counter.n - before
        before = counter.n
        t0 = time.perf_counter()
        out = pre.apply(vec)
        jax.block_until_ready(out)
        t_first = time.perf_counter() - t0
        n_apply = counter.n - before
        # The wall-clock cost of the apply-side compilation, which is what the
        # count cannot tell you: a recurring compile of 40 ms is a different
        # decision from one of 4 s. Steady state is subtracted so this is the
        # COMPILE, not the compile plus one apply.
        print(f"[arm  ] {label:<46s} build = {n_build:5d}   "
              f"apply = {n_apply:5d}   first-apply = {t_first*1e3:8.1f} ms",
              flush=True)
        results[label[0]] = (n_build, n_apply, t_first)
        return pre, np.asarray(out)

    def timed(pre, vec, repeats=200):
        jax.block_until_ready(pre.apply(vec))
        t0 = time.perf_counter()
        for _ in range(repeats):
            out = pre.apply(vec)
        jax.block_until_ready(out)
        return (time.perf_counter() - t0) / repeats

    with jax.log_compiles(True):
        seqA, opsA = build_sequence(cli.geometry, ns, cli.p, 2000)
        n = int(getattr(seqA, f"n{k}_dbc" if dbc else f"n{k}"))
        v = jnp.asarray(np.random.default_rng(0).standard_normal(n))
        print(f"[setup] {cli.geometry} ns={ns} p={cli.p} k={k} dbc={dbc} "
              f"n={n}", flush=True)

        def build(seq, ops, **kw):
            return MetricLumpingLaplacian(seq, ops, k, dbc, **kw)

        preA, outA = arm("A  first build + apply  (CONTROL)",
                         lambda: build(seqA, opsA), v)

        # Counts only -- NOT sum(results["A"]), which now includes the
        # wall-clock float and is therefore always positive, silently
        # disarming this guard.
        if results["A"][0] + results["A"][1] == 0:
            print("\n[ABORT] arm A compiled NOTHING. A first build and apply "
                  "must compile, so the COUNTER is broken -- not the premise. "
                  "Every other number in this run is meaningless and is not "
                  "reported.", flush=True)
            print(f"[debug] markers matched so far: {counter.seen}", flush=True)
            print("[done]", flush=True)
            return

        tA = timed(preA, v)
        preB, outB = arm("B  rebuild IDENTICAL config + apply",
                         lambda: build(seqA, opsA), v)
        tB = timed(preB, v)

        seqB, opsB = build_sequence(cli.geometry_b, ns, cli.p, 2000)
        nB = int(getattr(seqB, f"n{k}_dbc" if dbc else f"n{k}"))
        vC = jnp.asarray(np.random.default_rng(0).standard_normal(nB))
        preC, _ = arm(f"C  rebuild after geometry change ({cli.geometry_b})",
                      lambda: build(seqB, opsB), vC)
        tC = timed(preC, vC)

        preD, outD = arm("D  rebuild with bc_scale=1.7 only",
                         lambda: build(seqA, opsA, bc_scale=1.7), v)
        tD = timed(preD, v)

    print("\n[result] (build, apply) compilations per arm:", flush=True)
    steady = {"A": tA, "B": tB, "C": tC, "D": tD}
    for key in "ABCD":
        b, a, tf = results[key]
        compile_ms = (tf - steady[key]) * 1e3
        print(f"[result]   {key}: build={b:5d}  apply={a:5d}  "
              f"apply-compile ~{compile_ms:7.1f} ms", flush=True)
    print("[result] THE APPLY COLUMN IS THE ONE THE REFACTOR TARGETS. Build-side "
          "compiles are construction's own jnp work and are not addressed by "
          "moving closure constants to leaves.", flush=True)
    print(f"[result] apply us/call: A={tA*1e6:.1f} B={tB*1e6:.1f} "
          f"C={tC*1e6:.1f} D={tD*1e6:.1f}", flush=True)

    same = float(np.linalg.norm(outA - outB) / max(np.linalg.norm(outA), 1e-300))
    diff = float(np.linalg.norm(outA - outD) / max(np.linalg.norm(outA), 1e-300))
    print(f"[check ] identical rebuild agrees to {same:.3e}", flush=True)
    print(f"[check ] bc_scale change moves the apply by {diff:.3e}", flush=True)
    if diff < 1e-12:
        print("[check ] *** arm D changed NOTHING -- its count is meaningless "
              "***", flush=True)

    # THE HEADLINE NUMBER, MEASURED CLEAN. Everything above ran inside
    # jax.log_compiles(True), which emits ~160k log records and inflates the
    # very tracing it is measuring. The counts need that instrumentation; the
    # WALL CLOCK does not, and quoting a logging-inflated compile time as the
    # cost of a payload change would overstate the case for the refactor.
    print("\n[clean ] re-timing one rebuild with logging OFF:", flush=True)
    logging.getLogger().removeHandler(counter)
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger("jax").setLevel(logging.WARNING)
    t0 = time.perf_counter()
    pre_clean = build(seqA, opsA)
    t_build = time.perf_counter() - t0
    t0 = time.perf_counter()
    jax.block_until_ready(pre_clean.apply(v))
    t_first_clean = time.perf_counter() - t0
    t_steady = timed(pre_clean, v)
    print(f"[clean ] build {t_build*1e3:8.1f} ms | first apply "
          f"{t_first_clean*1e3:8.1f} ms | steady {t_steady*1e6:6.1f} us/call",
          flush=True)
    print(f"[clean ] recurring apply-compile per payload change ~"
          f"{(t_first_clean - t_steady)*1e3:.1f} ms "
          f"= {(t_first_clean - t_steady)/t_steady:,.0f} applies' worth",
          flush=True)

    print("\n[verdict] premise TRUE (refactor can help) if B/C/D > 0; "
          "FALSE (refactor buys nothing) if all are 0.", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
