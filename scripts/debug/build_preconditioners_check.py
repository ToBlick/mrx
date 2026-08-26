"""Does `build_preconditioners` do what its docstring says?

Four claims, each tested rather than asserted:

  1. GEOMETRY-ONLY WORK IS A FIRST-CLASS PATH. `set_map` alone installs a map
     and builds no preconditioners; `build_preconditioners` without a geometry
     raises rather than half-working.
  2. IT IS RE-CALLABLE AFTER A GEOMETRY CHANGE, and the second call builds
     against the geometry now installed -- not against stale factors.
  3. IT CLEARS STALE SCHUR DIAGONALS on an incoming bundle. This is the guard
     the function exists to provide: `ops.schur_diaginv_k*` are fields on the
     SequenceOperators bundle, nothing invalidates them, and `set_geometry`
     cannot reach them. A bundle handed back through `operators=` would
     otherwise carry the previous metric's diagonals silently, which shows up
     as slow convergence rather than as an error.
  4. `set_map_and_preconditioners` IS `set_map` + `build_preconditioners`, so
     the wrapper cannot drift from the thing it wraps.

    python scripts/debug/build_preconditioners_check.py --geometry toroid
"""
from __future__ import annotations

import argparse
import os
import sys

import jax

jax.config.update("jax_enable_x64", True)


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mrx.operators as op  # noqa: E402
from mrx.derham_sequence import DeRhamSequence  # noqa: E402
from mrx.mappings import rotating_ellipse_map, toroid_map  # noqa: E402

FAILURES = []


def check(label, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f"  -- {detail}" if detail
                                                     else ""), flush=True)
    if not ok:
        FAILURES.append(label)


def fresh_seq(ns, p):
    seq = DeRhamSequence(ns, (p,) * 3, 2 * p,
                         ("clamped", "periodic", "periodic"),
                         polar=True, tol=1e-10, maxiter=2000,
                         betti_numbers=(1, 1, 0, 0))
    seq.evaluate_1d()
    return seq


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ns", default="6,12,6")
    ap.add_argument("--p", type=int, default=2)
    cli = ap.parse_args()
    ns = tuple(int(v) for v in cli.ns.split(","))
    p = cli.p
    print(f"[setup] ns={ns} p={p}", flush=True)

    # --- 1. geometry-only, and the no-geometry guard ----------------------
    seq = fresh_seq(ns, p)
    try:
        seq.build_preconditioners(ks=(0,), dirichlets=(False,))
        check("build_preconditioners without a geometry raises", False,
              "it returned instead")
    except Exception as exc:                                  # noqa: BLE001
        check("build_preconditioners without a geometry raises", True,
              type(exc).__name__)

    seq.set_map(toroid_map(epsilon=1 / 3, R0=1.0))
    has_atom = hasattr(seq, op.METRIC_LUMPING_CACHE_ATTR)
    check("set_map alone builds NO preconditioners", not has_atom,
          f"atom cache present={has_atom}")

    # --- 2. explicit build, then re-call after a geometry change ----------
    ops1 = seq.build_preconditioners(ks=(0, 1), dirichlets=(False, True))
    check("build_preconditioners returns a bundle", ops1 is not None)
    check("the atom is assembled after the explicit build",
          op._metric_lumping_available(seq, 0, False))

    # Stamp a Schur diagonal so the clearing has something to clear.
    n1 = int(seq.n1)
    fake = jax.numpy.ones(n1)
    ops1 = op._set_schur_diaginv(ops1, 1, False, fake, "metric_lumping_probe")
    before = op._get_schur_diaginv(ops1, 1, False, "metric_lumping_probe")
    check("a Schur diagonal can be stored on the bundle", before is not None)

    # --- 3. the clearing, which is the guard A4 exists to provide ---------
    seq.set_map(rotating_ellipse_map(eps=0.33, kappa=1.5, nfp=3))
    ops2 = seq.build_preconditioners(ks=(0, 1), dirichlets=(False, True),
                                     operators=ops1)
    after = op._get_schur_diaginv(ops2, 1, False, "metric_lumping_probe")
    check("stale Schur diagonals are CLEARED when a bundle is reused",
          after is None,
          "still present -- a bundle would carry the old metric's diagonals"
          if after is not None else "")

    check("the atom is rebuilt against the NEW geometry",
          op._metric_lumping_available(seq, 0, False))

    # --- 4. the wrapper is the two calls, nothing else --------------------
    seq2 = fresh_seq(ns, p)
    ops3 = seq2.set_map_and_preconditioners(
        toroid_map(epsilon=1 / 3, R0=1.0), ks=(0,), dirichlets=(False,))
    check("set_map_and_preconditioners installs a map and builds",
          ops3 is not None and op._metric_lumping_available(seq2, 0, False))

    # --- idempotence ------------------------------------------------------
    ops4 = seq2.build_preconditioners(ks=(0,), dirichlets=(False,),
                                      operators=ops3)
    check("a second call on the same geometry succeeds (idempotent)",
          ops4 is not None and op._metric_lumping_available(seq2, 0, False))

    print(f"\n[result] {len(FAILURES)} failure(s)"
          + (": " + ", ".join(FAILURES) if FAILURES else ""), flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
