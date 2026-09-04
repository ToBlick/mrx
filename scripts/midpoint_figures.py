"""Figures and the summary table of the implicit-midpoint study (scripts/midpoint_sweep.sh).

    python scripts/midpoint_figures.py [--root outputs/midpoint_sweep] [--arms a,b,...]

Reads ``<root>/<arm>/relax.json`` and writes ``<root>/figures/*.png`` plus
``<root>/summary.md``. Traces follow the house convention: 100-step block
means with a +-1 sd ribbon, statistics in log space on log axes
(``scripts/li383_pub_figures.blocked``).
"""
import argparse
import json
import os

import numpy as np

from li383_pub_figures import plot_trace, plt  # scripts/ is sys.path[0]; sets the Agg backend

STYLE = {
    "ex_lbfgs": dict(color="k", ls="-", label="explicit, L-BFGS"),
    "mp_lbfgs": dict(color="tab:red", ls="-", label="midpoint, L-BFGS"),
    "ex_lbfgs_f64_Hd": dict(color="k", ls="--", label="explicit, float64, Dirichlet H"),
    "mp_lbfgs_f64_Hd": dict(color="tab:red", ls="--", label="midpoint, float64, Dirichlet H"),
    "ex_small_f64": dict(color="tab:gray", ls="-", label="explicit, float64, (8,16,16) p=2"),
    "mp_small_f64": dict(color="tab:orange", ls="-", label="midpoint, float64, (8,16,16) p=2"),
    "ex_small_f64_Hd": dict(color="tab:gray", ls="--", label="explicit, float64, (8,16,16) p=2, Dirichlet H"),
    "mp_small_f64_Hd": dict(color="tab:orange", ls="--", label="midpoint, float64, (8,16,16) p=2, Dirichlet H"),
}


def load(root, arm):
    path = os.path.join(root, arm, "relax.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def style(arm):
    return STYLE.get(arm, dict(label=arm))


def wall_of_step(j):
    """Wall-clock seconds at every step, interpolated between the qoi samples."""
    q = j["qoi"]
    n = len(j["trace"]["E"])
    it = np.array(q["it"], float)
    w = np.array(q["wall"], float)
    return np.interp(np.arange(1, n + 1), it, w)


def helicity_figure(arms, figdir):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for arm, j in arms:
        h = np.array(j["qoi"]["helicity"])
        it = np.array(j["qoi"]["it"])
        E0 = j["ic"]["E"]
        d = np.abs(h - j["ic"]["H"])
        axes[0].plot(it, np.where(d > 0, d, np.nan), marker=".", ms=3, **style(arm))
        axes[1].plot(it, np.where(d > 0, d / (2 * E0), np.nan), marker=".", ms=3, **style(arm))
    axes[0].set(xlabel="step", ylabel="|H - H_0|", yscale="log", title="helicity drift, absolute (||B||_M = 1)")
    axes[1].set(xlabel="step", ylabel="|H - H_0| / 2E_0", yscale="log", title="relative to the energy")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "helicity_drift.png"), dpi=150)


def descent_figure(arms, figdir):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for arm, j in arms:
        tr = j["trace"]
        E = np.array(tr["E"])
        E0 = j["ic"]["E"]
        st = style(arm)
        plot_trace(axes[0], tr["resid"], log=True, **st)
        w = wall_of_step(j)
        axes[1].plot(w, (E0 - E) / E0, **st)
        axes[2].plot(np.arange(1, len(E) + 1), (E0 - E) / E0, **st)
    axes[0].set(xlabel="step", ylabel="force residual", xscale="log", yscale="log")
    axes[1].set(xlabel="wall-clock [s]", ylabel="(E_0 - E) / E_0", xscale="log", yscale="log")
    axes[2].set(xlabel="step", ylabel="(E_0 - E) / E_0", xscale="log", yscale="log")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "descent.png"), dpi=150)


def picard_figure(arms, figdir):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for arm, j in arms:
        tr = j["trace"]
        st = style(arm)
        if "picard_it" in tr and max(tr["picard_it"]) > 1:
            plot_trace(axes[0], tr["picard_it"], log=False, **st)
            plot_trace(axes[2], np.array(tr["picard_resid"]), log=True, **st)
        plot_trace(axes[1], tr["dt"], log=True, **st)
    axes[0].set(xlabel="step", ylabel="increment evaluations / step")
    axes[1].set(xlabel="step", ylabel="dt", yscale="log")
    axes[2].set(xlabel="step", ylabel="Picard residual (last sweep)", yscale="log")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "picard.png"), dpi=150)


def summary(arms):
    rows = ["| arm | steps | stop | s/step | eval/step | dt halved | unconverged | E removed | resid final (mean last 100) | identity max | dH abs | dH / 2E0 |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm, j in arms:
        tr, s = j["trace"], j["summary"]
        n = s["steps"]
        E0 = j["ic"]["E"]
        h = np.array(j["qoi"]["helicity"])
        dh = h[-1] - j["ic"]["H"]
        pit = np.array(tr.get("picard_it", [1] * n))
        prs = np.array(tr.get("picard_restarts", [0] * n))
        prd = np.array(tr.get("picard_resid", [0.0] * n))
        tol = j["params"].get("picard_tol", 0.0)
        ident = np.abs(np.array(tr["dE_meas"]) - np.array(tr["dE_pred"])) / E0
        rows.append(
            f"| {arm} | {n} | {s['stop']} | {s['wall'] / max(n, 1):.2f} | {pit.mean():.2f} "
            f"| {int((prs > 0).sum())} | {int((prd > tol).sum()) if tol else 0} "
            f"| {(E0 - tr['E'][-1]) / E0:.4%} | {tr['resid'][-1]:.3e} ({np.mean(tr['resid'][-100:]):.3e}) "
            f"| {ident.max():.2e} | {dh:+.3e} | {dh / (2 * E0):+.3e} |")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/midpoint_sweep")
    ap.add_argument("--arms", default=",".join(STYLE))
    cli = ap.parse_args()
    arms = [(a, load(cli.root, a)) for a in cli.arms.split(",")]
    arms = [(a, j) for a, j in arms if j is not None and j.get("trace")]
    figdir = os.path.join(cli.root, "figures")
    os.makedirs(figdir, exist_ok=True)
    helicity_figure(arms, figdir)
    descent_figure(arms, figdir)
    picard_figure(arms, figdir)
    table = summary(arms)
    with open(os.path.join(cli.root, "summary.md"), "w") as fh:
        fh.write(table + "\n")
    print(table)
    print(f"figures in {figdir}")


if __name__ == "__main__":
    main()
