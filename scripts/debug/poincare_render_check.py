"""Validation for the 2026-08-25 plotter changes. Renders, and asserts.

Four things, none of which need a solve -- they run off an archived trace, so
this is presentation-only and cheap:

1. the arbitrary zeta-slice count is a pure driver change and is checked by
   argument parsing, not here;
2. the B^zeta gate RAISES on a field whose toroidal component crosses zero, and
   on one that merely comes close, and passes a healthy one;
3. the section re-renders unchanged in shape with the new colormap;
4. the iota/p split and the p panel draw, exercised with a SYNTHETIC pressure
   -- these traces are harmonic and carry none.

The synthetic pressure is labelled as such on the figure. It exists to prove
the wiring, not to say anything about a device.
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..")))

from mrx.poincare import (  # noqa: E402
    BZETA_MIN_FRACTION, BzetaParameterisationError, CHAOS_TOL,
    SECTION_CMAP, iota_convergence, render_section,
    require_zeta_parameterisation,
)

NFP = {"w7x": 5, "quasr9983": 2, "quasr44970": 3, "hegna": 3}


def check_bzeta_gate():
    """The gate must fire on sign change and on near-zero, and pass sane fields."""
    results = []

    # Healthy: B^zeta dominant, exactly the regime the quasr family measured in.
    def healthy(x):
        return jnp.array([0.05 * jnp.sin(x[1]), 0.10, 1.0])

    info = require_zeta_parameterisation(healthy, n=512, name="healthy")
    assert info["bz_over_b_absmin"] > BZETA_MIN_FRACTION
    results.append(("healthy field passes", True, f"|B^z|/|B| min "
                    f"{info['bz_over_b_absmin']:.3f}"))

    # Sign change: the failure that renders as a chaotic sea.
    def flips(x):
        return jnp.array([0.1, 0.1, jnp.cos(2.0 * jnp.pi * x[0])])

    try:
        require_zeta_parameterisation(flips, n=512, name="sign-flipper")
        results.append(("sign change raises", False, "DID NOT RAISE"))
    except BzetaParameterisationError as exc:
        ok = "CHANGES SIGN" in str(exc)
        results.append(("sign change raises", ok, str(exc)[:90]))

    # Near-zero without a sign change: still refused, because the stiff RHS
    # would surface as refinement-invariant drift, i.e. as chaos.
    def grazing(x):
        return jnp.array([1.0, 1.0, 0.01 + 0.0 * x[0]])

    try:
        require_zeta_parameterisation(grazing, n=512, name="grazer")
        results.append(("near-zero raises", False, "DID NOT RAISE"))
    except BzetaParameterisationError as exc:
        ok = "comes within" in str(exc)
        results.append(("near-zero raises", ok, str(exc)[:90]))

    return results


def synthetic_pressure(R, Z, axis_RZ):
    """A peaked profile in distance from the axis. NOT physics -- wiring only."""
    cR = float(jnp.mean(jnp.asarray(axis_RZ[0])))
    cZ = float(jnp.mean(jnp.asarray(axis_RZ[1])))
    d = jnp.sqrt((R - cR) ** 2 + (Z - cZ) ** 2)
    return jnp.exp(-(d / (0.6 * float(jnp.max(d)) + 1e-30)) ** 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace", help="an archived trace_*.npz")
    ap.add_argument("--geometry", default="w7x")
    ap.add_argument("--plane", type=float, default=0.0)
    ap.add_argument("--out", default="outputs/render_check")
    cli = ap.parse_args()
    os.makedirs(cli.out, exist_ok=True)

    print("=== B^zeta gate ===", flush=True)
    failures = 0
    for name, ok, detail in check_bzeta_gate():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
        failures += not ok

    d = np.load(cli.trace)
    key = f"R_zeta{cli.plane:g}"
    if key not in d.files:
        raise SystemExit(f"{cli.trace} has no {key}; has {sorted(d.files)}")
    R = jnp.asarray(d[key])
    Z = jnp.asarray(d[f"Z_zeta{cli.plane:g}"])
    # Arrays, not scalars: the axis crossing at each save.
    axis = (d[f"axisR_zeta{cli.plane:g}"], d[f"axisZ_zeta{cli.plane:g}"])
    keep = ~(jnp.asarray(d["escaped"]) | ~jnp.asarray(d["ok"]))
    nfp = NFP[cli.geometry]
    chaotic = jnp.asarray(iota_convergence(d["ys"], 8, nfp)) > CHAOS_TOL

    print("\n=== renders ===", flush=True)
    base = os.path.join(cli.out, f"{cli.geometry}_zeta{cli.plane:g}")

    # 3. the existing figure, new colormap.
    render_section(
        R, Z, jnp.asarray(d["iota"]), jnp.asarray(d["resid"]),
        jnp.asarray(d["seeds"])[:, 0], keep,
        title=f"{cli.geometry}  |  $\\zeta={cli.plane:g}$  |  cmap={SECTION_CMAP}",
        subtitle=f"nfp = {nfp}", axis_RZ=axis, nfp=nfp, chaotic=chaotic,
        path=base + "_iota.png")
    print(f"  wrote {base}_iota.png", flush=True)

    # 4/5. pressure panel and the up/down split, on a synthetic p.
    p = synthetic_pressure(R, Z, axis)
    render_section(
        R, Z, jnp.asarray(d["iota"]), jnp.asarray(d["resid"]),
        jnp.asarray(d["seeds"])[:, 0], keep,
        title=f"{cli.geometry}  |  $\\zeta={cli.plane:g}$  |  "
              "SPLIT: $\\iota$ above axis, $p$ below\nSYNTHETIC p -- wiring "
              "check, not physics",
        subtitle=f"nfp = {nfp}", axis_RZ=axis, nfp=nfp, chaotic=chaotic,
        pressure=p, split_iota_p=True, path=base + "_split.png")
    print(f"  wrote {base}_split.png", flush=True)

    # The split must refuse without a pressure, and without an axis.
    for kwargs, want in (
            (dict(pressure=None, axis_RZ=axis), "needs a pressure array"),
            (dict(pressure=p, axis_RZ=None), "needs axis_RZ")):
        try:
            render_section(R, Z, jnp.asarray(d["iota"]), jnp.asarray(d["resid"]),
                           jnp.asarray(d["seeds"])[:, 0], keep, title="x",
                           subtitle="x", nfp=nfp, split_iota_p=True, **kwargs)
            print(f"  [FAIL] split without {want}: DID NOT RAISE", flush=True)
            failures += 1
        except ValueError as exc:
            ok = want in str(exc)
            print(f"  [{'PASS' if ok else 'FAIL'}] split refuses: {str(exc)[:70]}",
                  flush=True)
            failures += not ok

    print(f"\n{'ALL CHECKS PASSED' if not failures else f'{failures} FAILED'}",
          flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
