"""The Cary-Hanson numbers behind fig:mesh_refinement ("the island width
agrees to 2% across the three meshes"): the runs
``outputs/li383_pulse/reconnect_l5_{h16,h32u,h32r}_p2_g1`` of 2026-09-02,
L-BFGS 10000 steps with one resistive solve at step 5000, on (16,32,32),
(32,32,32) and (32,32,32) radially refined (``--r-refine
0.47:0.62:6,0.68:0.94:15``, rebuilt here with the ``radial_knots`` of that
commit). The field of each run after the solve and at the end; the three
chains; chain radius and shear from the unseeded equilibrium's section.

    SCRIPT=docs/research/island_diagnostic_2026-09-18/paper_widths_mesh.py bash slurm/run.sh
"""
import json
import os

import h5py
import numpy as np

from mrx.geometry import build_sequence
from mrx.nullspace import compute_nullspaces
from mrx.poincare import fixed_points, island_width

PULSE = "/kfs3/scratch/tblickhan/mrx/outputs/li383_pulse"
DEMOS = "/kfs3/scratch/tblickhan/mrx/.claude/worktrees/newton/outputs/newton_demos"
CHAINS = {"3/5": (5, 1), "1/2": (6, 1), "3/7": (7, 1)}
TOL = 2e-3


def radial_breakpoints(n_r, p, windows):
    """The breakpoints of ``mrx.geometry.radial_knots`` at commit 3740157."""
    windows = sorted(windows)
    n_cells = n_r - p
    inside = sum(m for _, _, m in windows)
    gaps, lo = [], 0.0
    for a, b, _ in windows:
        gaps.append((lo, a))
        lo = b
    gaps.append((lo, 1.0))
    gaps = [(a, b) for a, b in gaps if b > a]
    free = n_cells - inside
    length = sum(b - a for a, b in gaps)
    raw = [free * (b - a) / length for a, b in gaps]
    counts = [max(1, int(r)) for r in raw]
    order = sorted(range(len(gaps)), key=lambda i: raw[i] - int(raw[i]), reverse=True)
    for i in order[: free - sum(counts)]:
        counts[i] += 1
    for i in reversed(order):
        if sum(counts) > free and counts[i] > 1:
            counts[i] -= 1
    segments = sorted([(a, b, m) for a, b, m in windows] + [(a, b, c) for (a, b), c in zip(gaps, counts)])
    return [float(v) for v in np.concatenate([np.linspace(a, b, m, endpoint=False) for a, b, m in segments] + [[1.0]])]


def equilibrium_chains():
    z = np.load(os.path.join(DEMOS, "seeded_reconnect", "unseeded", "trace.npz"))
    ok = z["ic_keep"] & ~z["ic_chaotic"]
    key = sorted(k for k in z.files if k.startswith("ic_zeta") and k.endswith("_logr"))[0]
    r, iota = np.nanmedian(z[key], axis=1)[ok], np.abs(z["ic_iota"])[ok]
    order = np.argsort(r)
    r, iota = r[order], iota[order]
    out = {}
    for name, (m, n) in CHAINS.items():
        target = 3 * n / m
        i = int(np.flatnonzero(np.diff(np.sign(iota - target)))[0])
        rc = float(r[i] + (target - iota[i]) * (r[i + 1] - r[i]) / (iota[i + 1] - iota[i]))
        near = np.abs(r - rc) <= 0.12
        out[name] = (rc, float(np.polyfit(r[near], iota[near], 1)[0]))
    return out


eq = equilibrium_chains()
print("equilibrium (r_chain, iota'):", {k: (round(v[0], 4), round(v[1], 4)) for k, v in eq.items()}, flush=True)

ARMS = [("h16", (16, 32, 32), None), ("h32u", (32, 32, 32), None),
        ("h32r", (32, 32, 32), radial_breakpoints(32, 2, [(0.47, 0.62, 6), (0.68, 0.94, 15)]))]
out = {}
for tag, ns, bp in ARMS:
    knots = None if bp is None else (bp, None, None)
    seq, _ = build_sequence("data/wout_li383_1.4m.nc", ns, 2, knots=knots, symmetry="field-period")
    compute_nullspaces(seq)
    h_r = 1.0 / ns[0]
    with h5py.File(os.path.join(PULSE, f"reconnect_l5_{tag}_p2_g1", "B.h5")) as f:
        steps = [int(s) for s in f["snapshot_steps"][:]]
        fields = {"final": np.asarray(f["B_final"])}
        for s in (5000, 5500):
            if s in steps:
                fields[f"step{s}"] = np.asarray(f["B_snapshots"][steps.index(s)])
    assert fields["final"].shape[0] == seq.n(2, True), (fields["final"].shape, seq.n(2, True))
    for fname, B in fields.items():
        for name, (m, n) in CHAINS.items():
            rc, shear = eq[name]
            fp = fixed_points(seq, B, m, [(rc, 0.0), (rc, 0.5 / m)])
            o = int(np.argmax(fp["residue"]))
            R = float(fp["residue"][o])
            w = float(island_width(R, m, shear, seq.nfp)) if R > 0 else 0.0
            out.setdefault(tag, {}).setdefault(fname, {})[name] = dict(
                r_O=float(fp["r"][o]), theta_O=float(fp["theta"][o]), residue_O=R, residue_X=float(fp["residue"][1 - o]),
                det=float(fp["det"][o]), defect=float(fp["defect"][o]), width=w)
            print(f"{tag:5s} {fname:9s} {name}  r_O {fp['r'][o]:.4f} th {fp['theta'][o]:.3f}  R_O {R:+.4f} "
                  f"R_X {fp['residue'][1 - o]:+.4f} det {fp['det'][o]:.4f} defect {fp['defect'][o]:.0e}  "
                  f"width {w:.4f} = {w / h_r:.2f} h_r (h_r = 1/{ns[0]})", flush=True)

dst = os.path.join("outputs", "half_period", "paper_widths_mesh.json")
json.dump(out, open(dst, "w"), indent=1)
print("wrote", dst)
