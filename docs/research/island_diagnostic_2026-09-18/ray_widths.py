"""The width of the 3/5 chain measured DIRECTLY: seeds on the radial ray
through the O-point, the extent of the lines locked to the chain.

Tobias 2026-09-18: the three final sections of the seeded-reconnection runs
show no visible difference in the 3/5 chain, while the O-point residues
differ (0.198 / 0.133 / 0.211). The residue is a property of the O-point
(the rotation rate about it); it is a width only through the constant-shear
single-harmonic pendulum, which the final states do not satisfy (the iota
profile flattens and turns over outside the chain). This measures the
separatrix itself: the O-point by Newton, 121 seeds on the ray through it,
300 periods each, locked = fitted iota within 2e-3 of the rational.

    SCRIPT=docs/research/island_diagnostic_2026-09-18/ray_widths.py bash slurm/run.sh
"""
import json
import os

import h5py
import jax.numpy as jnp
import numpy as np

from mrx.geometry import build_sequence
from mrx.nullspace import compute_nullspaces
from mrx.poincare import fixed_points, logical_field, rotational_transform, trace

DEMOS = "/kfs3/scratch/tblickhan/mrx/.claude/worktrees/newton/outputs/newton_demos/seeded_reconnect"
H_R, M, TARGET, TOL = 1.0 / 16.0, 5, 0.6, 2e-3
STEPS, PERIODS = 24, 300

seq, _ = build_sequence("data/wout_li383_1.4m.nc", (16, 32, 32), 2, symmetry="field-period")
compute_nullspaces(seq)
field = logical_field(seq, 2, True)

out = {}
for run, step in (("seed51", 0), ("unseeded", 300), ("seed51", 300), ("seed61", 300)):
    with h5py.File(os.path.join(DEMOS, run, "checkpoints", f"state_{step:06d}.h5")) as f:
        B = jnp.asarray(np.asarray(f["B_n"]))
    fp = fixed_points(seq, B, M, [(0.80, 0.0), (0.80, 0.5 / M)])
    o = int(np.argmax(fp["residue"]))
    r_O, th_O, R = float(fp["r"][o]), float(fp["theta"][o]), float(fp["residue"][o])
    r = np.linspace(max(r_O - 0.25, 0.4), min(r_O + 0.19, 0.985), 121)
    seeds = jnp.asarray(np.stack([r, np.full_like(r, th_O)], axis=1))
    ys, _ = trace(field, B, seeds, PERIODS, STEPS)
    iota, _ = rotational_transform(ys, STEPS, seq.nfp)
    locked = np.abs(np.abs(np.asarray(iota)) - TARGET) < TOL
    # the contiguous locked set that contains the O-point
    i0 = int(np.argmin(np.abs(r - r_O)))
    lo = hi = i0
    while lo > 0 and locked[lo - 1]:
        lo -= 1
    while hi < r.size - 1 and locked[hi + 1]:
        hi += 1
    ray = float(r[hi] - r[lo]) if locked[i0] else 0.0
    # the paper's measure on these seeds: the largest max(r) - min(r) of a locked line over the planes k/8
    rr = np.sqrt(np.sum(np.asarray(ys)[:, ::STEPS // 8, :] ** 2, axis=-1))
    excursion = float(np.max(rr[locked].max(axis=1) - rr[locked].min(axis=1))) if locked.any() else 0.0
    out[f"{run}_{step}"] = dict(r_O=r_O, theta_O=th_O, residue=R, ray=ray, ray_lo=float(r[lo]), ray_hi=float(r[hi]),
                                excursion=excursion, n_locked=int(locked.sum()), dr=float(r[1] - r[0]))
    print(f"{run:9s} step {step:3d}  O at r {r_O:.4f} theta {th_O:.3f}  R {R:+.4f} | ray: locked r in "
          f"[{r[lo]:.4f}, {r[hi]:.4f}] = {ray:.4f} = {ray / H_R:.2f} h_r  ({int(locked.sum())} locked of 121, "
          f"spacing {(r[1] - r[0]) / H_R:.2f} h_r) | largest excursion of a locked line {excursion:.4f} = "
          f"{excursion / H_R:.2f} h_r", flush=True)

json.dump(out, open(os.path.join("outputs", "half_period", "ray_widths.json"), "w"), indent=1)
