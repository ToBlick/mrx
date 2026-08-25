"""Plot the relaxation traces that relax_prelim.py archives, with no solve.

Every arm writes its full per-step trace into the run JSON, so the whole
campaign can be re-plotted from disk in seconds.  matplotlib, numpy and json
only -- this runs on a login node, the same arrangement poincare_replot.py
uses for its archived orbits.

WHAT IS PLOTTED, AND WHY THOSE
------------------------------
ENERGY is the only quantity this scheme guarantees, so it goes first and it
goes on a log scale of ``E(0) - E(t)`` -- the energy REMOVED -- because the
absolute values agree to five digits and a linear axis shows nothing.

``||F||`` is a diagnostic and is NOT guaranteed to fall: F is the gradient of
the objective, and a descent method promises the objective decreases, not the
norm of its own gradient.  Excursions in it need no explanation.

HELICITY is shown as the ABSOLUTE change at ``||B||_M = 1``.  The relative
form is actively misleading across these runs: helicity spans three orders of
magnitude between cases, and the arm with the largest relative drift (W1) has
the cleanest surfaces.  Absolute |dH| is what correlated with surface
destruction under a blind classification of every Poincare pair.

    python scripts/debug/relax_plot_traces.py --out out/relax_prelim/figs
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = "/scratch/tblickhan/mrx/out/relax_prelim"


def load(tag, fname=None):
    """Return {arm_name: trace} for one run directory, or {} if absent."""
    d = os.path.join(ROOT, tag)
    if not os.path.isdir(d):
        return {}
    cands = ([fname] if fname else
             [f for f in sorted(os.listdir(d)) if f.endswith(".json")])
    for c in cands:
        path = os.path.join(d, c)
        if not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                blob = json.load(f)
        except Exception:
            continue
        if isinstance(blob, dict) and "arms" in blob:
            return {k: v.get("trace", {}) for k, v in blob["arms"].items()}
    return {}


def series(tr, key):
    v = tr.get(key)
    return np.asarray(v, dtype=float) if v else None


def energy_removed(tr):
    E = series(tr, "E")
    return None if E is None else np.maximum(0.5 - E, 1e-18)


def panel(ax, title, xlabel, ylabel, logy=True):
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    if logy:
        ax.set_yscale("log")
    ax.grid(alpha=0.3, lw=0.5)
    ax.tick_params(labelsize=8)


def plot_group(runs, title, path, hel_absolute=True):
    """One figure per comparison: energy removed, ||F||, |dH|, div B."""
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.0))
    panel(axes[0], f"{title}\nenergy REMOVED (the guaranteed quantity)",
          "step", "E(0) - E(t)")
    panel(axes[1], "force residual ||F||  (NOT guaranteed monotone)",
          "step", "||F||_M")
    panel(axes[2], "|dH| absolute  (relative form misleads here)",
          "step", "|H(t) - H(0)|")
    panel(axes[3], "||div B||", "step", "||div B||")

    any_data = False
    for label, tr in runs:
        if not tr:
            continue
        any_data = True
        dE = energy_removed(tr)
        if dE is not None:
            axes[0].plot(np.arange(1, len(dE) + 1), dE, lw=1.2, label=label)
        F = series(tr, "F")
        if F is not None:
            axes[1].plot(np.arange(1, len(F) + 1), F, lw=1.2, label=label)
        H, it = series(tr, "helicity"), series(tr, "hel_it")
        if H is not None and it is not None and len(H) == len(it):
            dH = np.abs(H - H[0]) if hel_absolute else np.abs((H - H[0]) / H[0])
            axes[2].plot(it, np.maximum(dH, 1e-18), lw=1.2, marker='o',
                         ms=2.5, label=label)
        dv = series(tr, "div")
        if dv is not None:
            axes[3].plot(np.arange(1, len(dv) + 1), np.maximum(dv, 1e-18),
                         lw=1.2, label=label)
    if not any_data:
        plt.close(fig)
        return False
    for a in axes:
        a.legend(fontsize=7, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(ROOT, "figs"))
    cli = ap.parse_args()
    os.makedirs(cli.out, exist_ok=True)

    def one(tag, arm=None):
        arms = load(tag)
        if arm:
            return arms.get(arm, {})
        return next(iter(arms.values()), {}) if arms else {}

    groups = [
        ("dt bracket -- w7x_ini, only the step differs",
         "dt_bracket.png",
         [("linesearch (LR3)", one("LR3")),
          ("fixed 3e-3 (D1)", one("D1_dt3e3")),
          ("fixed 1e-3 (W5)", one("W5")),
          ("fixed 3e-4 (D2)", one("D2_dt3e4")),
          ("fixed 1e-4 (D3)", one("D3_dt1e4"))]),
        ("optimizers -- w7x_fmm002, a case that behaves",
         "optimizers.png",
         [("gradient", one("S11_opt", "gradient")),
          ("lbfgs (fixed)", one("S11_opt", "lbfgs")),
          ("cg", one("W1"))]),
        ("p-refinement at fixed h -- w7x_fmm002",
         "p_sweep.png",
         [("p=1", one("P1")), ("p=2", one("P2")),
          ("p=3", one("W1")), ("p=4", one("S03_p4")),
          ("p=5", one("P5"))]),
        ("h-refinement -- w7x_fmm002",
         "h_sweep.png",
         [("8^3", one("W1")), ("12^3", one("S01_res12")),
          ("16^3", one("S02_res16"))]),
        ("resistivity -- w7x_fmm002",
         "eta_sweep.png",
         [("eta=0", one("W1")), ("eta=1e-4", one("S08_eta4")),
          ("eta=1e-3", one("S09_eta3")), ("eta=1e-2", one("S10_eta2"))]),
        ("hyperregularisation -- w7x_fmm002",
         "gamma_sweep.png",
         [("gamma=0", one("W1")),
          ("gamma=1 mu=1e-3", one("S04_g1mu3")),
          ("gamma=1 mu=1e-2", one("S05_g1mu2")),
          ("gamma=2 mu=1e-3", one("S06_g2mu3"))]),
        ("length -- w7x_fmm002, same settings",
         "length.png",
         [("3000 steps (W1)", one("W1")), ("13018 steps (S07)",
                                           one("S07_long"))]),
    ]
    made = 0
    for title, fname, runs in groups:
        if plot_group(runs, title, os.path.join(cli.out, fname)):
            made += 1
    print(f"\n{made}/{len(groups)} figures written to {cli.out}")


if __name__ == "__main__":
    main()
