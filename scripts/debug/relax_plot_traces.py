"""Plot the relaxation traces that relax.py (formerly relax_prelim.py) archives, with no solve.

Every arm writes its full per-step trace into the run JSON, so the whole
campaign can be re-plotted from disk in seconds.  matplotlib, numpy and json
only -- this runs on a login node, the same arrangement poincare_replot.py
uses for its archived orbits.

WHAT IS PLOTTED, AND WHY THOSE
------------------------------
ENERGY is the only quantity this scheme guarantees, so it goes first -- as the
DISSIPATION RATE ``-dE/dt``, positive by that guarantee.  Not the absolute
energy (the runs agree to five digits, a linear axis shows nothing) and not the
cumulative energy removed either: that curve saturates within a few hundred
steps and every arm then looks like the same flat line.  The rate keeps
resolving, and falls four to five decades over a run, which is the thing worth
seeing -- it says how far from stationary the arm actually is.

``dt`` varies per step under the linesearch, so the rate is
``-dE_meas/dt`` step by step, NOT a difference of the energy trace against a
step index.  Dividing by the actual step is what makes linesearch and
fixed-dt arms comparable on one axis at all.

``||F||`` is a diagnostic and is NOT guaranteed to fall: F is the gradient of
the objective, and a descent method promises the objective decreases, not the
norm of its own gradient.  Excursions in it need no explanation.

HELICITY is shown as the ABSOLUTE change at ``||B||_M = 1``.  The relative
form is actively misleading across these runs: helicity spans three orders of
magnitude between cases, and the arm with the largest relative drift (W1) has
the cleanest surfaces.  Absolute |dH| is what correlated with surface
destruction under a blind classification of every Poincare pair.

THE WALL-CLOCK AXIS
-------------------
``--x wall`` replots everything against GPU-seconds instead of step index,
which is the axis to rank arms on: a step is not a unit of cost, and the
arms here differ by 10x in seconds/step (0.87 at 8^3, 9.03 at 16^3).

Runs from 2026-08-25 onward carry real sampled timing in ``trace['wall']``,
taken at the helicity iterations and NET of the diagnostic block.  For those
the per-step axis is a linear interpolation BETWEEN those samples, which
tracks the real cost drift (warm starts get cheaper as the field settles).

Older runs have only a per-arm total, so their axis is that total spread
uniformly over the steps.  This is exact at the endpoint and an approximation
in between; its known error is JIT compilation, which lands entirely in step 1
and shifts such a curve left near the origin.  Curves reconstructed this way
are drawn DASHED and marked "(uniform)" in the legend, so a uniform-axis arm
is never silently compared against a measured one.

    python scripts/debug/relax_plot_traces.py --out out/relax_prelim/figs
    python scripts/debug/relax_plot_traces.py --x wall
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

#: Line alpha.  These arms differ by design and their traces sit on top of one
#: another for hundreds of steps; opaque lines mean the last one plotted is the
#: only one visible and the overlap reads as agreement of a single curve.
A = 0.5

#: Set by main() from --x.  Module-level because every panel needs it and
#: threading it through four plot calls buys nothing.
WALL = False


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
            return dict(blob["arms"])
    return {}


def xaxis(arm, n):
    """(x values for steps 1..n, label, is_measured).

    Returns the step index unchanged unless a wall-clock axis is asked for.
    """
    if not WALL:
        return np.arange(1, n + 1), "step", True
    tr = arm.get("trace", {})
    w, it = series(tr, "wall"), series(tr, "hel_it")
    idx = np.arange(1, n + 1)
    if w is not None and it is not None and len(w) == len(it) >= 2:
        # Real samples: interpolate between them.  np.interp CLAMPS outside the
        # sample range rather than extrapolating, which would flatten the tail
        # of a budget-truncated arm onto its last sample; extend with the local
        # slope instead.
        x = np.interp(idx, it, w)
        tail = idx > it[-1]
        if tail.any():
            slope = (w[-1] - w[-2]) / max(it[-1] - it[-2], 1.0)
            x[tail] = w[-1] + slope * (idx[tail] - it[-1])
        return x / 3600.0, "GPU-hours", True
    total, steps = arm.get("wall"), arm.get("steps")
    if total and steps:
        return idx * (total / steps) / 3600.0, "GPU-hours", False
    return idx, "step", True


def series(tr, key):
    v = tr.get(key)
    return np.asarray(v, dtype=float) if v else None


def dissipation_rate(tr):
    """-dE/dt per step, positive wherever the descent guarantee holds.

    ``dE_meas`` is the SIGNED energy change and is negative on a healthy step,
    so the rate is its negation.  Non-positive entries are dropped rather than
    clamped: on a log axis a clamp invents a point at the floor, and a step
    that failed to decrease the energy is a real finding that should show up
    as a GAP, not as a spurious data point.
    """
    dE, dt = series(tr, "dE_meas"), series(tr, "dt")
    if dE is None or dt is None or len(dE) != len(dt):
        return None
    return -dE / dt


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
    panel(axes[0], f"{title}\ndissipation rate -dE/dt (the guaranteed sign)",
          "step", "-dE/dt")
    panel(axes[1], "force residual ||F||  (NOT guaranteed monotone)",
          "step", "||F||_M")
    panel(axes[2], "|dH| absolute  (relative form misleads here)",
          "step", "|H(t) - H(0)|")
    panel(axes[3], "||div B||", "step", "||div B||")

    any_data = False
    xlabel = "step"
    for label, arm in runs:
        tr = arm.get("trace", {}) if arm else {}
        if not tr:
            continue
        any_data = True
        n = len(tr.get("E") or [])
        x, xlabel, measured = xaxis(arm, n)
        style = {} if measured else {"ls": "--"}
        if not measured:
            label = f"{label} (uniform)"
        rate = dissipation_rate(tr)
        if rate is not None:
            ok = rate > 0
            if ok.any():
                axes[0].plot(x[:len(rate)][ok], rate[ok],
                             lw=1.2, alpha=A, label=label, **style)
        F = series(tr, "F")
        if F is not None:
            axes[1].plot(x[:len(F)], F, lw=1.2, alpha=A, label=label, **style)
        H, it = series(tr, "helicity"), series(tr, "hel_it")
        if H is not None and it is not None and len(H) == len(it):
            dH = np.abs(H - H[0]) if hel_absolute else np.abs((H - H[0]) / H[0])
            # DROP THE FIRST SAMPLE.  dH(0) = |H(0) - H(0)| is zero by
            # construction, not a measurement, and on a log axis it lands at
            # whatever floor we clamp to -- a dozen decades below every real
            # point, flattening the actual spread into one line.  Any later
            # exact zero is dropped for the same reason.
            keep = dH > 0
            keep[0] = False
            if keep.any():
                # hel_it are STEP indices; map them onto whichever axis is in
                # use rather than plotting steps against hours.
                xh = x[np.clip(it.astype(int) - 1, 0, len(x) - 1)]
                axes[2].plot(xh[keep], dH[keep], lw=1.2, alpha=A, marker='o',
                             ms=2.5, label=label, **style)
        dv = series(tr, "div")
        if dv is not None:
            axes[3].plot(x[:len(dv)], np.maximum(dv, 1e-18),
                         lw=1.2, alpha=A, label=label, **style)
    if not any_data:
        plt.close(fig)
        return False
    for a in axes:
        a.set_xlabel(xlabel, fontsize=9)
        leg = a.legend(fontsize=7, framealpha=0.9)
        for line in leg.get_lines():   # legend keys stay readable at alpha 0.5
            line.set_alpha(1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(ROOT, "figs"))
    ap.add_argument("--x", choices=("step", "wall"), default="step",
                    help="x axis: step index, or GPU-hours (see module docs)")
    cli = ap.parse_args()
    os.makedirs(cli.out, exist_ok=True)
    global WALL
    WALL = cli.x == "wall"
    suffix = "_wall" if WALL else ""

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
        # The mu sweep of s37.  gamma_sweep.png above is the ORIGINAL three
        # points; this is the resolved ladder that located the minimum, and it
        # is kept as a separate figure rather than merged into that one --
        # six arms on four panels is already at the limit of what alpha=0.5
        # keeps readable.
        ("hyperregularisation, resolved -- w7x_fmm002 8^3",
         "mu_sweep.png",
         [("gamma=0", one("W1")),
          ("gamma=1 mu=1e-4", one("M1_mu1e4")),
          ("gamma=1 mu=1e-3", one("M2_mu1e3")),
          ("gamma=1 mu=1e-2", one("M3_mu1e2")),
          ("gamma=1 mu=1e-1 (cut 2000)", one("M4_mu1e1")),
          ("gamma=2 mu=1e-3", one("M5_g2mu3"))]),
        # s40: the mu ORDERING reverses between 8^3 and 12^3.  Both 12^3 arms
        # stopped on wall-clock (1760 and 1200 steps), which is why they end
        # early here; the reversal is read where all four overlap, not at the
        # right-hand end of the longer curves.
        ("mu across resolution -- w7x_fmm002, the ordering REVERSES",
         "mu_h_reversal.png",
         [("8^3  mu=1e-4", one("M1_mu1e4")),
          ("8^3  mu=1e-3", one("M2_mu1e3")),
          ("12^3 mu=4.4e-4 (cut 1760)", one("H1_r12_mu4e4")),
          ("12^3 mu=1e-3 (cut 1200)", one("H2_r12_mu1e3"))]),
    ]
    made = 0
    for title, fname, runs in groups:
        out = os.path.join(cli.out, fname.replace(".png", suffix + ".png"))
        if plot_group(runs, title, out):
            made += 1
    print(f"\n{made}/{len(groups)} figures written to {cli.out}")


if __name__ == "__main__":
    main()
