"""Compare two relax.py runs (half-period vs full-period quadrature): the
per-step force residuals, energies, helicities, the final B, and the wall
time per step. Usage: compare_runs.py <full_dir> <half_dir>"""
import glob
import json
import os
import sys

import h5py
import numpy as np

full, half = sys.argv[1], sys.argv[2]
rf, rh = (json.load(open(os.path.join(d, "relax.json"))) for d in (full, half))
for key in ("F", "resid", "dE"):
    a, b = np.asarray(rf["trace"][key], float), np.asarray(rh["trace"][key], float)
    n = min(a.size, b.size)
    rel = np.abs(a[:n] - b[:n]) / np.maximum(np.abs(a[:n]), 1e-300)
    print(f"{key:4s}: {n} steps, max rel diff {rel.max():.3e} (step {rel.argmax() + 1}), "
          f"final {a[n - 1]:.6e} vs {b[n - 1]:.6e}")
for key in ("helicity", "E"):
    if key in rf.get("qoi", {}):
        a, b = np.asarray(rf["qoi"][key], float), np.asarray(rh["qoi"][key], float)
        n = min(a.size, b.size)
        print(f"{key}: max rel diff {np.max(np.abs(a[:n] - b[:n]) / np.abs(a[:n])):.3e}")
cf = sorted(glob.glob(os.path.join(full, "checkpoints", "state_0*.h5")))[-1]
ch = sorted(glob.glob(os.path.join(half, "checkpoints", "state_0*.h5")))[-1]
with h5py.File(cf) as f, h5py.File(ch) as h:
    Bf, Bh = np.asarray(f["B_n"], float), np.asarray(h["B_n"], float)
print(f"final B ({os.path.basename(cf)}): max |dB| / max |B| = "
      f"{np.max(np.abs(Bf - Bh)) / np.max(np.abs(Bf)):.3e}")
for name, r in (("full", rf), ("half", rh)):
    wall, it = np.asarray(r["qoi"]["wall"], float), np.asarray(r["qoi"]["it"], float)
    print(f"{name}: summary {r['summary']}")
    if wall.size > 1:
        print(f"  wall per step over the last chunk: {(wall[-1] - wall[-2]) / (it[-1] - it[-2]):.3f} s")
