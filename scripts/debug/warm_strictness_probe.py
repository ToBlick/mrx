"""Which (k, BC) pairs does `warm_mass_preconditioner_cache` silently skip?

`warm_mass_preconditioner_cache` wraps each build in `except Exception: pass`,
on the stated grounds that "a degree/BC this kind does not support is not an
error here". A4's `build_preconditioners` is meant to be honest about partial
assembly and raise instead -- but that is only safe if the swallow is currently
catching NOTHING on real geometries. If some pair legitimately fails, strict
warming would break the setup path for every caller.

This measures it rather than assuming either way. It builds each (k, BC) atom
directly and reports the exception, if any.

    python scripts/debug/warm_strictness_probe.py --geometries toroid,w7x
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mrx.operators as op  # noqa: E402
from verify_block_jacobi import build_sequence  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometries", default="toroid,w7x")
    ap.add_argument("--ns", default="8,16,8")
    ap.add_argument("--p", type=int, default=3)
    cli = ap.parse_args()
    ns = tuple(int(v) for v in cli.ns.split(","))

    failures = 0
    for geometry in cli.geometries.split(","):
        geometry = geometry.strip()
        print(f"\n[geom ] {geometry} ns={ns} p={cli.p}", flush=True)
        seq, ops = build_sequence(geometry, ns, cli.p, 2000)
        for k in (0, 1, 2, 3):
            for dbc in (False, True):
                side = "dbc " if dbc else "free"
                try:
                    pre = op._mass_metric_lumping_for(seq, ops, k, dbc)
                    n = int(getattr(seq, f"n{k}_dbc" if dbc else f"n{k}"))
                    v = np.ones(n)
                    out = np.asarray(pre.apply(v))
                    ok = np.all(np.isfinite(out))
                    print(f"[warm ] k={k} {side} OK   n={n:7d} finite={ok}",
                          flush=True)
                except Exception as exc:                     # noqa: BLE001
                    failures += 1
                    print(f"[warm ] k={k} {side} *** FAILS: "
                          f"{type(exc).__name__}: {exc}", flush=True)
                    traceback.print_exc()

    print(f"\n[done] {failures} (k, BC) pair(s) would raise under strict "
          f"warming", flush=True)
    print("[note] zero failures means strict warming is free; any failure means "
          "build_preconditioners must not raise on it.", flush=True)


if __name__ == "__main__":
    main()
