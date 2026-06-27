"""Diagnostic: angular-mean radial profiles rho_c(r) of the k=0 Laplacian channels.

The whole "r is forced dense" argument rests on the *analytic* claim that the
three stiffness channels diverge oppositely in r:
    alpha_rr = J g^{rr},  alpha_thetatheta = J g^{theta theta},  alpha_zetazeta = J g^{zeta zeta}
    (claimed)  alpha_thetatheta ~ 1/r,   alpha_zetazeta ~ r.
This script MEASURES them instead of asserting. For each geometry it prints, at
the radial quadrature points r:
  - rho_c(r) = <alpha_c>_{theta,zeta}   (the angular-mean radial profile)
  - the angular spread  (max-min)/|mean| and std/|mean| at each r
    -> tells us whether the radial mean is representative, and whether the
       angular part "averages to (near) zero" relative to the mean.
  - a log-log power-law slope of rho_c(r) vs r (labels "~1/r", "~r", "~const")
  - the inter-channel ratios rho_rr/rho_thetatheta and rho_thetatheta/rho_zetazeta
    vs r -> the direct test of "do they diverge oppositely?".

Pure geometry evaluation, no solve. Channels are indexed (theta, r, zeta);
angular mean is over axes (0, 2), leaving the radial axis 1.
"""
from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "benchmark"))

from benchmark_graddiv_k1_preconditioner import build_sequence  # noqa: E402
from mrx.operators import _k0_stiffness_diagonal_metric_tensors  # noqa: E402


def loglog_slope(r, y):
    r = np.asarray(r, float); y = np.asarray(y, float)
    m = (r > 0) & (y > 0)
    if m.sum() < 2:
        return float("nan")
    p = np.polyfit(np.log(r[m]), np.log(y[m]), 1)
    return float(p[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", default="toroid", choices=["cylinder", "toroid", "rotating_ellipse", "w7x"])
    ap.add_argument("--ns", type=int, nargs=3, default=[12, 24, 12])
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--epsilon", type=float, default=1.0 / 3.0)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--r0", type=float, default=1.0)
    ap.add_argument("--nfp", type=int, default=3)
    args = ap.parse_args()

    cfg = SimpleNamespace(ns=tuple(args.ns), p=args.p, geometry=args.geometry, cg_tol=1e-10,
                          cg_maxiter=1, epsilon=args.epsilon, kappa=args.kappa, r0=args.r0, nfp=args.nfp)
    print(f"=== k=0 Laplacian radial profiles  geometry={args.geometry} ns={tuple(args.ns)} p={args.p} ===", flush=True)
    seq = build_sequence(cfg)
    metric = _k0_stiffness_diagonal_metric_tensors(seq)
    r = np.asarray(seq.quad.x_x, dtype=float)  # radial quadrature points (axis 1)

    chans = [("rr", "alpha_rr"), ("thetatheta", "alpha_thetatheta"), ("zetazeta", "alpha_zetazeta")]
    prof = {}
    print(f"\n{'chan':12} {'slope(loglog)':>14}   profile rho_c(r) and angular spread")
    for name, key in chans:
        a = np.asarray(metric[key], dtype=float)            # (theta, r, zeta)
        rho = a.mean(axis=(0, 2))                            # (r,)
        amin = a.min(axis=(0, 2)); amax = a.max(axis=(0, 2))
        astd = a.std(axis=(0, 2))
        prof[name] = rho
        slope = loglog_slope(r, rho)
        print(f"\n{name:12} {slope:>14.3f}")
        print(f"  {'r':>9} {'rho(r)':>12} {'(max-min)/|mean|':>17} {'std/|mean|':>12}")
        for i in range(len(r)):
            den = abs(rho[i]) if abs(rho[i]) > 0 else 1.0
            print(f"  {r[i]:>9.4f} {rho[i]:>12.4e} {(amax[i]-amin[i])/den:>17.3f} {astd[i]/den:>12.3f}", flush=True)

    print(f"\n{'inter-channel ratios':22}")
    print(f"  {'r':>9} {'rr/thetatheta':>14} {'thetatheta/zetazeta':>20} {'rr/zetazeta':>13}")
    for i in range(len(r)):
        tt = prof['thetatheta'][i]; zz = prof['zetazeta'][i]; rr = prof['rr'][i]
        print(f"  {r[i]:>9.4f} {rr/tt if tt else float('nan'):>14.3f} "
              f"{tt/zz if zz else float('nan'):>20.3f} {rr/zz if zz else float('nan'):>13.3f}", flush=True)

    # Verdict aids: do thetatheta and zetazeta share a radial profile?
    tt = prof['thetatheta']; zz = prof['zetazeta']
    ratio = tt / np.where(zz != 0, zz, np.nan)
    rel_spread = (np.nanmax(ratio) - np.nanmin(ratio)) / abs(np.nanmean(ratio))
    print(f"\nrho_thetatheta/rho_zetazeta over r: min={np.nanmin(ratio):.3e} max={np.nanmax(ratio):.3e} "
          f"rel-spread={rel_spread:.3f}", flush=True)
    print("  (rel-spread ~ 0  => angular channels SHARE a radial profile => radially-weighted FD is EXACT.", flush=True)
    print("   rel-spread large => they diverge oppositely => r is forced dense.)", flush=True)


if __name__ == "__main__":
    main()
