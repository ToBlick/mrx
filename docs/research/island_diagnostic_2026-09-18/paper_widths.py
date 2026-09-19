"""The Cary-Hanson numbers for every island width the paper quotes.

The paper's widths are max(r) - min(r) of the lines locked to a chain (the
section measure). Here, for the same fields: the O-point residue of each
chain and the pendulum width from it, with the shear of the run's own
iota profile at that field (locked lines left out) and with the shear of
the unseeded equilibrium. Runs of the newton branch, li383 (16,32,32) p=2:

* the seeded-reconnection table (3/5, 1/2, 3/7 chains; unseeded, (5,1) and
  (6,1) seeds; the initial field, the field before each of the four
  reconnections, the final one),
* the reconnection ladder's 3/5 chain (``ladder3``),
* the seeded table (``seed61_2``, ``seed51_2``: initial, lowest residual, last).

    SCRIPT=docs/research/island_diagnostic_2026-09-18/paper_widths.py bash slurm/run.sh
"""
import json
import os
import sys

import h5py
import numpy as np

from mrx.geometry import build_sequence
from mrx.nullspace import compute_nullspaces
from mrx.experimental.islands import fixed_points, island_width

DEMOS = "/kfs3/scratch/tblickhan/mrx/.claude/worktrees/newton/outputs/newton_demos"
H_R = 1.0 / 16.0
CHAINS = {"3/5": (5, 1), "1/2": (6, 1), "3/7": (7, 1)}
TOL = 2e-3          # the paper's lock criterion on iota


def profile(z, tag):
    """(r, iota) of the regular lines of a traced field: r the median logical
    radius of the line's crossings of the first plane."""
    ok = z[f"{tag}_keep"] & ~z[f"{tag}_chaotic"]
    key = sorted(k for k in z.files if k.startswith(f"{tag}_zeta") and k.endswith("_logr"))[0]
    r = np.nanmedian(z[key], axis=1)
    iota = np.abs(z[f"{tag}_iota"])
    order = np.argsort(r[ok])
    return r[ok][order], iota[ok][order]


def section_width(z, tag, target):
    iota = np.abs(z[f"{tag}_iota"])
    on = z[f"{tag}_keep"] & ~z[f"{tag}_chaotic"] & (np.abs(iota - target) < TOL)
    if not on.any():
        return 0.0
    keys = [k for k in z.files if k.startswith(f"{tag}_zeta") and k.endswith("_logr")]
    lr = np.concatenate([z[k][on] for k in keys], axis=1)
    return float(np.max(np.nanmax(lr, axis=1) - np.nanmin(lr, axis=1)))


def chain_radius_and_shear(r, iota, target, window=0.12):
    """Where the profile crosses the rational (the mean radius of the locked
    lines if there are any) and the slope of a linear fit over +-window
    around it with the locked lines left out."""
    locked = np.abs(iota - target) < TOL
    if locked.any():
        r_chain = float(r[locked].mean())
    else:
        cross = np.flatnonzero(np.diff(np.sign(iota - target)))
        if cross.size == 0:
            return float("nan"), float("nan")
        i = int(cross[0])
        r_chain = float(r[i] + (target - iota[i]) * (r[i + 1] - r[i]) / (iota[i + 1] - iota[i]))
    near = (np.abs(r - r_chain) <= window) & ~locked
    slope = float(np.polyfit(r[near], iota[near], 1)[0]) if near.sum() >= 3 else float("nan")
    return r_chain, slope


def load_B(run, tag, results):
    ck = os.path.join(DEMOS, run, "checkpoints")
    if tag == "best":
        with h5py.File(os.path.join(ck, "best.h5")) as f:
            return np.asarray(f["B_best"] if "B_best" in f else f["B_n"])
    if tag == "ic":
        step = 0
    elif tag == "final":
        step = int(results["summary"]["steps"])
    else:
        step = int(results["reconnect"][int(tag[len("reconnect"):]) - 1]["it"])
    with h5py.File(os.path.join(ck, f"state_{step:06d}.h5")) as f:
        return np.asarray(f["B_n"])


seq, _ = build_sequence("data/wout_li383_1.4m.nc", (16, 32, 32), 2, symmetry="field-period")
compute_nullspaces(seq)

# the unseeded equilibrium's shear at the three rationals
z0 = np.load(os.path.join(DEMOS, "seeded_reconnect", "unseeded", "trace.npz"))
r0, i0 = profile(z0, "ic")
eq = {}
for name, (m, n) in CHAINS.items():
    eq[name] = chain_radius_and_shear(r0, i0, seq.nfp * n / m)
    print(f"equilibrium: chain {name} at r = {eq[name][0]:.4f}, iota' = {eq[name][1]:+.4f}", flush=True)

JOBS = [("seeded_reconnect/unseeded", ("ic", "reconnect1", "reconnect2", "reconnect3", "reconnect4", "final"), CHAINS),
        ("seeded_reconnect/seed51", ("ic", "reconnect1", "reconnect2", "reconnect3", "reconnect4", "final"), CHAINS),
        ("seeded_reconnect/seed61", ("ic", "reconnect1", "reconnect2", "reconnect3", "reconnect4", "final"), CHAINS),
        ("ladder3", ("ic", "reconnect1", "reconnect2", "reconnect3", "reconnect4", "final"), {"3/5": (5, 1)}),
        ("seed61_2", ("ic", "best", "final"), {"1/2": (6, 1)}),
        ("seed51_2", ("ic", "best", "final"), {"3/5": (5, 1)})]

only = set(sys.argv[1:])          # run names to restrict to (a job's time limit: about a minute per evaluation)
out = {}
for run, tags, chains in JOBS:
    if only and run not in only:
        continue
    results = json.load(open(os.path.join(DEMOS, run, "relax.json")))
    z = np.load(os.path.join(DEMOS, run, "trace.npz"))
    traced = set(str(f) for f in z["fields"])
    for tag in tags:
        B = load_B(run, tag, results)
        for name, (m, n) in chains.items():
            target = seq.nfp * n / m
            if tag in traced:
                r, iota = profile(z, tag)
                r_chain, slope = chain_radius_and_shear(r, iota, target)
                w_sec = section_width(z, tag, target)
            else:
                r_chain, slope, w_sec = float("nan"), float("nan"), float("nan")
            if not np.isfinite(r_chain):
                r_chain = eq[name][0]
            fp = fixed_points(seq, B, m, [(r_chain, 0.0), (r_chain, 0.5 / m)])
            o = int(np.argmax(fp["residue"]))                  # the O-point: the larger residue of the two kinds
            R, x = float(fp["residue"][o]), float(fp["residue"][1 - o])
            w_loc = float(island_width(R, m, slope, seq.nfp)) if np.isfinite(slope) and R > 0 else float("nan")
            w_eq = float(island_width(R, m, eq[name][1], seq.nfp)) if R > 0 else 0.0
            rec = dict(r_chain=r_chain, r_O=float(fp["r"][o]), theta_O=float(fp["theta"][o]), residue_O=R, residue_X=x,
                       det=float(fp["det"][o]), defect=float(fp["defect"][o]), shear_local=slope, shear_eq=eq[name][1],
                       width_section=w_sec, width_local=w_loc, width_eq=w_eq)
            out.setdefault(run, {}).setdefault(name, {})[tag] = rec
            print(f"{run:26s} {name} {tag:11s} r_O {rec['r_O']:.4f} th {rec['theta_O']:.3f}  R_O {R:+.4f} R_X {x:+.4f} "
                  f"det {rec['det']:.4f} defect {rec['defect']:.0e} | section {w_sec / H_R:4.1f}  "
                  f"residue(local shear {slope:+.3f}) {w_loc / H_R:4.1f}  residue(eq shear) {w_eq / H_R:4.1f}  [h_r]", flush=True)

dst = os.path.join("outputs", "half_period", "paper_widths" + ("_" + "_".join(sorted(only)) if only else "") + ".json")
json.dump(out, open(dst, "w"), indent=1)
print("wrote", dst)
