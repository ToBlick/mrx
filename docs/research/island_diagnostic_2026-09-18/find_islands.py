"""``mrx.poincare.islands`` on fields whose islands are known: the three
final states of the seeded-reconnection runs (the paper's widths table:
3/5 chain 0.213 / 0.204 / 0.214, 1/2 chain 0.059 / 0.039 / 0.093, 3/7 chain
0.055 / 0.037 / 0.039 in logical r), the unseeded initial field (no
islands), and the (6,1)-seeded initial field (one chain, 0.164).

    SCRIPT=docs/research/island_diagnostic_2026-09-18/find_islands.py bash slurm/run.sh
"""
import os
import time

import h5py
import jax.numpy as jnp
import numpy as np

from mrx.geometry import build_sequence
from mrx.nullspace import compute_nullspaces
from mrx.experimental.islands import islands
from mrx.poincare import poincare

DEMOS = "/kfs3/scratch/tblickhan/mrx/.claude/worktrees/newton/outputs/newton_demos/seeded_reconnect"

seq, _ = build_sequence("data/wout_li383_1.4m.nc", (16, 32, 32), 2, symmetry="field-period")
compute_nullspaces(seq)

for run, step in (("unseeded", 0), ("seed61", 0), ("unseeded", 300), ("seed51", 300), ("seed61", 300)):
    with h5py.File(os.path.join(DEMOS, run, "checkpoints", f"state_{step:06d}.h5")) as f:
        B = jnp.asarray(np.asarray(f["B_n"]))
    t0 = time.perf_counter()
    res = poincare(seq, B, lines=160, periods=400)
    t1 = time.perf_counter()
    found = islands(seq, B, res)
    t2 = time.perf_counter()
    print(f"=== {run} step {step}: section {t1 - t0:.0f}s, islands {t2 - t1:.0f}s, {len(found)} chains", flush=True)
    for c in found:
        print(f"   ({c['m']:2d},{c['n']}) iota {c['iota']:.4f} at r {c['r_chain']:.3f}: width {c['width']:.4f} "
              f"(ray {c['ray']:.4f}, {c['n_locked']} locked)  R_O {c['residue']:+.4f}  "
              f"{len(c['O'])} O at theta {[round(p[1], 3) for p in c['O']]}  {len(c['X'])} X "
              f"residues {[round(p[2], 4) for p in c['X']]}", flush=True)
