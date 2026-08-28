"""Summary table of the li383 sweep (outputs/li383_sweep/<arm>/relax.json).

Bookkeeping only (json + sacct), runs on the login node with python3:

    python3 scripts/li383_summary.py [SWEEP_DIR]

Prints a markdown table per arm: |F| start/end, the windowed residual,
steps and stop reason, s/step (descent loop only, setup excluded), the
relaxation time sum(dt), energy removed, helicity change (absolute and
relative), beta_vol, ||J||/||B||, and the GPU-hours of the relaxation job
from sacct (via jobs.tsv), plus the ledger of every job's elapsed time.
"""
import json
import os
import subprocess
import sys


def sacct(jid):
    out = subprocess.run(["sacct", "-j", jid, "-X", "-n", "-P", "-o", "State,ElapsedRaw"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    if not out:
        return "?", 0.0
    state, el = out[0].split("|")[:2]
    return state, float(el or 0) / 3600.0


def main():
    sweep = sys.argv[1] if len(sys.argv) > 1 else "outputs/li383_sweep"
    jobs = []
    ledger = os.path.join(sweep, "jobs.tsv")
    if os.path.isfile(ledger):
        for line in open(ledger):
            f = line.rstrip("\n").split("\t")
            if len(f) >= 6:
                jobs.append(f)
    hours = {}
    total = 0.0
    lines = ["| job | kind | arm | state | GPU-h |", "|---|---|---|---|---|"]
    for jid, kind, name, tmin, log, cmd in jobs:
        state, h = sacct(jid)
        total += h
        if kind == "relax":
            hours[name] = hours.get(name, 0.0) + h
        lines.append(f"| {jid} | {kind} | {name} | {state} | {h:.2f} |")
    lines.append(f"| | | **total** | | **{total:.2f}** |")

    rows = ["| arm | ns | p | g | mu | prec | steps | stop | s/step | t_relax | \\|F\\| 0 -> end | resid win | E0-E | dH abs | dH/H0 | beta_vol | J/B | GPU-h |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in sorted(os.listdir(sweep)):
        fn = os.path.join(sweep, name, "relax.json")
        if not os.path.isfile(fn):
            continue
        d = json.load(open(fn))
        P, tr, q, s = d["params"], d["trace"], d["qoi"], d.get("summary", {})
        n = len(tr["F"])
        wall_setup = q["wall"][0] if q["wall"] else 0.0
        wall = s.get("wall", q["wall"][-1] if q["wall"] else 0.0)
        sps = (wall - wall_setup) / max(n - 1, 1)
        H0, H1 = q["helicity"][0], q["helicity"][-1]
        win = s.get("resid_window_mean", sum(tr["resid"][-100:]) / min(100, n))
        rows.append(
            f"| {name} | {','.join(map(str, P['ns']))} | {P['p']} | {P['velocity_smoothing_order']} | "
            f"{P['velocity_smoothing_scale']:.1e} | {P['precision'][-2:]} | {n} | {s.get('stop', 'RUNNING')} | "
            f"{sps:.2f} | {sum(tr['dt']):.3g} | {tr['F'][0]:.2e} -> {tr['F'][-1]:.2e} | {win:.2e} | "
            f"{tr['E'][0] - tr['E'][-1]:.2e} | {H1 - H0:+.1e} | {(H1 - H0) / H0:+.1e} | "
            f"{q['beta_vol'][-1]:.4f} | {q['JoverB'][-1]:.3f} | {hours.get(name, 0.0):.2f} |")
    print("\n".join(rows))
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
