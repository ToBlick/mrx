"""Reality check for the W7-X angular atom: is beta_aa = g_aa/J dominated by a
single helical Fourier mode in the (theta, zeta) plane?

If the AC (non-DC) energy of beta_theta-theta / beta_zeta-zeta concentrates in one
(m_theta, m_zeta) mode, a Fourier-banded theta-zeta block atom is cheap (narrow
band). If it spreads over several helical harmonics, the band widens.

Evaluates beta on a uniform periodic (theta, zeta) grid at a few bulk radii and
2-D FFTs each. No solve; map + geometry-term evaluation only.

Run (GPU):
  W7X_MAP_BATCH=256 XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=8 \
    .venv/bin/python scripts/debug/w7x_metric_fft.py --ns 12 24 12 --p 3 --nfp 5
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mrx as _mrx  # noqa: E402
_mrx.MAP_BATCH_SIZE_INNER = int(os.environ.get("W7X_MAP_BATCH", "256"))
from w7x_geometry import build_w7x_map  # noqa: E402
from mrx.geometry import compute_geometry_terms  # noqa: E402


def signed(m, n):
    return int(m) if m <= n // 2 else int(m) - n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs=3, default=[12, 24, 12])
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--nfp", type=int, default=5)
    ap.add_argument("--nth", type=int, default=48)
    ap.add_argument("--nze", type=int, default=48)
    ap.add_argument("--radii", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    args = ap.parse_args()

    print(f"=== W7-X beta=g_aa/J (theta,zeta) FFT  ns={tuple(args.ns)} p={args.p} "
          f"nfp={args.nfp} grid={args.nth}x{args.nze} ===", flush=True)
    map_func, _ = build_w7x_map(map_ns=tuple(args.ns), p=args.p)

    th = np.arange(args.nth) / args.nth
    ze = np.arange(args.nze) / args.nze
    TT, ZZ = np.meshgrid(th, ze, indexing="ij")

    for r in args.radii:
        pts = jnp.asarray(np.stack(
            [np.full(TT.size, r), TT.ravel(), ZZ.ravel()], axis=-1))
        metric, _minv, jac = compute_geometry_terms(map_func, pts)
        metric = np.asarray(metric)
        jac = np.asarray(jac)
        print(f"\n--- r={r}  (J: min={jac.min():.2e} max={jac.max():.2e}) ---", flush=True)
        for name, idx in (("beta_tt", 1), ("beta_zz", 2), ("beta_rr", 0)):
            beta = (metric[:, idx, idx] / jac).reshape(args.nth, args.nze)
            F = np.fft.fft2(beta) / beta.size
            amp = np.abs(F)
            dc = amp[0, 0]
            ac = amp.copy()
            ac[0, 0] = 0.0
            ac_energy = float((ac ** 2).sum())
            rms = float(np.sqrt((ac ** 2).sum()))
            order = np.argsort(ac.ravel())[::-1]
            tops = []
            cum = 0.0
            n90 = 0
            for j, flat in enumerate(order):
                a, b = divmod(int(flat), args.nze)
                v = float(ac.ravel()[flat])
                if j < 5:
                    tops.append((signed(a, args.nth), signed(b, args.nze), v))
                cum += v * v
                if cum < 0.9 * ac_energy:
                    n90 = j + 1
            top_frac = (tops[0][2] ** 2) / ac_energy if ac_energy > 0 else 0.0
            # note: real field -> modes come in +/- conjugate pairs; count pairs
            top_pair_frac = 2 * top_frac if (tops[0][0] or tops[0][1]) else top_frac
            tstr = ", ".join(f"({a},{b}):{v:.2e}" for a, b, v in tops)
            print(f"  {name}: DC={dc:.3e}  AC/DC={rms/dc:.2f}  "
                  f"top1-pair-frac={min(top_pair_frac,1.0):.2f}  n_modes@90%AC={n90}",
                  flush=True)
            print(f"      top modes (mth,mze):amp = {tstr}", flush=True)


if __name__ == "__main__":
    main()
