"""Generate the relaxation results catalogue from the archived run JSONs.

GENERATED, NOT HAND-WRITTEN, on purpose: the campaign is ~40 runs and still
landing, and a hand-maintained table would be stale within the hour and wrong
within the day.  Re-run this and the catalogue is current.

    python scripts/debug/relax_results_table.py \\
        > docs/research/relaxation_results_table.md

THE COLUMNS, AND WHY THESE
--------------------------
``dE``            energy removed.  The ONLY quantity this scheme guarantees to
                  fall, so every ranking is normalised by it.
``|F|``           force residual, reported but never ranked on: F is the
                  gradient of the objective and its norm has no monotonicity
                  guarantee at all.
``|dH|/H per dE`` helicity lost per unit energy removed -- the quantity that
                  correlates with surface destruction.  Comparable WITHIN a
                  resolution and a p; NOT across them, because H itself is not
                  converged (it collapses 5.6x per h-refinement on fmm002,
                  where the field is ~99% harmonic and this gauge's helicity
                  is a small residue).
``div B``         should sit at round-off; it is conserved exactly by the
                  topological curl.
``s/step``        wall clock per step.  A step is NOT a unit of cost -- these
                  arms span 0.23 to 22.6 s/step -- so any ranking that counts
                  steps is ranking on an axis nobody pays in.
``dE/GPU-hr``     energy removed per GPU-hour.  Useful for planning a run;
                  NOT a quality measure and never a reason to prefer one
                  discretisation over another.  A finer arm costing more is
                  expected, not a finding.

NONE OF THESE COLUMNS IS A FLOOR
--------------------------------
The measurable claim of a refinement study is where the run FLOORS -- does
``||F||`` bottom out lower, does the energy settle nearer the true minimum.
No arm in this campaign floored: all are still descending at their last step,
with dissipation rates 1e-9 to 1e-3 against a demonstrated ~1e-16 round-off
floor.  Reading any row here as a verdict on refinement is the error handoff
s32 retracts.
"""
from __future__ import annotations

import json
import os

ROOT = "/scratch/tblickhan/mrx/out/relax_prelim"

#: (directory, label, what varied).  Order is the reading order of the report.
RUNS = [
    ("W1", "W1", "fmm002 clebsch, cg, baseline"),
    ("LR3", "LR3", "w7x_ini clebsch, cg, linesearch"),
    ("W5", "W5", "w7x_ini, FIXED dt=1e-3"),
    ("D1_dt3e3", "D1", "w7x_ini, FIXED dt=3e-3"),
    ("D2_dt3e4", "D2", "w7x_ini, FIXED dt=3e-4"),
    ("D3_dt1e4", "D3", "w7x_ini, FIXED dt=1e-4"),
    ("S13_ini_res12", "S13", "w7x_ini at 12^3"),
    ("S14_ini_res16", "S14", "w7x_ini at 16^3"),
    ("S01_res12", "S01", "fmm002 at 12^3"),
    ("S02_res16", "S02", "fmm002 at 16^3"),
    ("P1", "P1", "fmm002 p=1"),
    ("P2", "P2", "fmm002 p=2"),
    ("S03_p4", "S03", "fmm002 p=4"),
    ("P5", "P5", "fmm002 p=5"),
    ("S04_g1mu3", "S04", "gamma=1 mu=1e-3"),
    ("S05_g1mu2", "S05", "gamma=1 mu=1e-2 (truncated)"),
    ("S06_g2mu3", "S06", "gamma=2 mu=1e-3 (truncated)"),
    ("M1_mu1e4", "M1", "gamma=1 mu=1e-4, NEW precond"),
    ("M2_mu1e3", "M2", "gamma=1 mu=1e-3, NEW precond"),
    ("M3_mu1e2", "M3", "gamma=1 mu=1e-2, NEW precond"),
    ("M4_mu1e1", "M4", "gamma=1 mu=1e-1, NEW precond"),
    ("M5_g2mu3", "M5", "gamma=2 mu=1e-3, NEW precond"),
    ("H1_r12_mu4e4", "H1", "12^3 mu=4.4e-4 (mu~h^2 test)"),
    ("H2_r12_mu1e3", "H2", "12^3 mu=1e-3 (mu~h^2 test)"),
    ("S08_eta4", "S08", "eta=1e-4"),
    ("S09_eta3", "S09", "eta=1e-3"),
    ("S10_eta2", "S10", "eta=1e-2"),
    ("S17_ini_eta3", "S17", "w7x_ini eta=1e-3 (DESIGN ERROR, s23.1)"),
    ("S07_long", "S07", "fmm002, 13018 steps"),
    ("S16_g1_long", "S16", "fmm002 gamma=1, long"),
    ("S15_res12_g1", "S15", "fmm002 12^3 gamma=1"),
    ("S11_opt", "S11", "gradient and lbfgs on fmm002"),
    ("S12_nolam", "S12", "fmm002 --no-lambda"),
    ("DZ", "DZ", "quasr dzeta -> harmonic"),
    ("ETA", "ETA", "quasr eta=1e-4"),
    ("LR1", "LR1", "quasr gamma=1"),
    ("LR2", "LR2", "quasr gamma=0, long"),
    ("LR4", "LR4", "quasr, pressure-shape tracking"),
    ("W2", "W2", "w7x logical (invented) IC"),
    ("W3", "W3", "w7x dzeta"),
    ("W4", "W4", "w7x_ini gamma=1"),
]


def load(tag):
    d = os.path.join(ROOT, tag)
    if not os.path.isdir(d):
        return None
    for f in sorted(os.listdir(d)):
        if not f.endswith(".json"):
            continue
        try:
            with open(os.path.join(d, f)) as fh:
                blob = json.load(fh)
        except Exception:
            continue
        if isinstance(blob, dict) and blob.get("arms"):
            return blob
    return None


def fmt(x, spec=".3e"):
    return "--" if x is None else format(x, spec)


def main():
    print("# Relaxation campaign: results catalogue\n")
    print("GENERATED by `scripts/debug/relax_results_table.py` -- do not hand-edit.")
    print("Re-run it to refresh as jobs land.\n")
    print("Narrative, mechanisms and corrections live in")
    print("`handoff_2026-08-25_relaxation_prelim.md`; this is the numbers.\n")
    print("**NO ARM IN THIS TABLE REACHED A FLOOR.** Every run is still")
    print("descending at its last step (handoff s32.1), and the resolution")
    print("arms were truncated on a budget set from the coarse case, so the")
    print("FINEST arms are furthest from their floors. The point of refining")
    print("h or p is to reach a LOWER floor, not to get there faster -- so")
    print("nothing here is a verdict on refinement, and a finer arm being")
    print("slower is expected rather than a finding.\n")
    print("Cost and quality are SEPARATE columns. `|dH|/H per dE` measures")
    print("reconnection efficiency; `dE/GPU-hr` measures cost. Neither is a")
    print("floor. Do NOT rank refinement on either -- capping dt and refining")
    print("h are not substitutes: the cap reduces time-integration error at")
    print("fixed h, refinement changes where the floor is.\n")
    print("`dE` is energy removed -- the only guaranteed-monotone quantity, and")
    print("what every ranking is normalised by. `|F|` is reported but never")
    print("ranked on: it is the gradient's norm and has no monotonicity")
    print("guarantee. `|dH|/H per dE` compares only WITHIN a fixed (geometry,")
    print("resolution, p) -- H itself is not converged across those.\n")
    hdr = ("| run | what varied | arm | steps | dE | \\|F\\| final | "
           "\\|dH\\|/H | per dE | div B | s/step | dE/GPU-hr |")
    print(hdr)
    print("|" + "---|" * 11)

    missing = []
    for tag, label, what in RUNS:
        blob = load(tag)
        if blob is None:
            missing.append((label, what))
            continue
        for arm, a in blob["arms"].items():
            tr = a.get("trace", {})
            E = tr.get("E") or []
            dE = 0.5 - E[-1] if E else None
            H = tr.get("helicity") or []
            rel = abs((H[-1] - H[0]) / H[0]) if len(H) > 1 and H[0] else None
            per = rel / dE if (rel is not None and dE) else None
            wall, n = a.get("wall"), a.get("steps")
            per_step = wall / n if (wall and n) else None
            per_hour = dE / (wall / 3600.0) if (wall and dE) else None
            print(f"| {label} | {what} | {arm} | {a.get('steps','--')} | "
                  f"{fmt(dE)} | {fmt(a.get('F_final'))} | {fmt(rel)} | "
                  f"{fmt(per, '.4g')} | {fmt(a.get('div_max'))} | "
                  f"{fmt(per_step, '.2f')} | {fmt(per_hour)} |")

    if missing:
        print("\n## Not yet landed\n")
        for label, what in missing:
            print(f"* **{label}** -- {what}")

    print("\n## Figures\n")
    figs = os.path.join(ROOT, "figs")
    if os.path.isdir(figs):
        for f in sorted(os.listdir(figs)):
            print(f"* `figs/{f}`")
    print("\nRegenerate with `scripts/debug/relax_plot_traces.py` "
          "(login node, matplotlib only, no solve).")


if __name__ == "__main__":
    main()
