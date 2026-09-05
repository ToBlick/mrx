"""Figures and the summary table of the implicit-midpoint study (scripts/midpoint_sweep.sh).

    python scripts/midpoint_figures.py [--root outputs/midpoint_sweep]

Reads ``<root>/<arm>/relax.json`` and writes ``<root>/figures/*.png`` plus
``<root>/summary.md``. House style (:mod:`mrx.plotstyle`): colour encodes
the route (teal the Dirichlet H, purple B only), dash the scheme (solid
explicit, dashed midpoint); per-step traces are 100-step block means with
a +-1 sd ribbon (``scripts/li383_pub_figures.blocked``). The natural-H
arms are not plotted: with E Dirichlet and H natural the helicity pairing
is not the conserved one (see the note); they stay in the summary table.
"""
import argparse
import json
import os

import numpy as np

from li383_pub_figures import plot_trace, plt  # scripts/ is sys.path[0]; sets the Agg backend
from mrx.plotstyle import arm_style, figsize, house_style

#: plotted arm -> (panel, colour index, dash index, label)
ARMS = {
    "ex_lbfgs_f64_Hd":    ("f64", 1, 0, "explicit, H"),
    "mp_lbfgs_f64_Hd":    ("f64", 1, 1, "midpoint, H"),
    "ex_small_f64_Hd":    ("small", 1, 0, "explicit, H"),
    "mp_small_f64_Hd":    ("small", 1, 1, "midpoint, H"),
    "ex_small_f64_bonly": ("small", 2, 0, "explicit, B only"),
    "mp_small_f64_bonly": ("small", 2, 1, "midpoint, B only"),
    "ex_small_f32_Hd":    ("small32", 1, 0, "explicit, H"),
    "mp_small_f32_Hd":    ("small32", 1, 1, "midpoint, H"),
    "ex_small_f32_bonly": ("small32", 2, 0, "explicit, B only"),
    "mp_small_f32_bonly": ("small32", 2, 1, "midpoint, B only"),
    "ex_r16_f64_Hd":      ("r16", 1, 0, "explicit, H"),
    "mp_r16_f64_Hd":      ("r16", 1, 1, "midpoint, H"),
    "ex_r16_f64_bonly":   ("r16", 2, 0, "explicit, B only"),
    "mp_r16_f64_bonly":   ("r16", 2, 1, "midpoint, B only"),
    "ex_r16_f32_Hd":      ("r16_32", 1, 0, "explicit, H"),
    "mp_r16_f32_Hd":      ("r16_32", 1, 1, "midpoint, H"),
    "ex_r16_f32_bonly":   ("r16_32", 2, 0, "explicit, B only"),
    "mp_r16_f32_bonly":   ("r16_32", 2, 1, "midpoint, B only"),
}
#: every arm of the study, for the summary table
ALL_ARMS = ("ex_lbfgs", "mp_lbfgs", "ex_small_f64", "mp_small_f64",
            "ex_small_f32", "mp_small_f32") + tuple(ARMS)
PANELS = {
    "f64": "(12,24,24) p=3, float64",
    "small": "(8,16,16) p=2, float64",
    "small32": "(8,16,16) p=2, float32",
    "r16": "(16,32,32) p=2, float64",
    "r16_32": "(16,32,32) p=2, float32",
}
#: The 2 x 2 x 2 comparison on (16,32,32) p=2: (precision, route) -> (explicit arm, midpoint arm)
EIGHT = {
    ("float32", "H"): ("ex_r16_f32_Hd", "mp_r16_f32_Hd"),
    ("float32", "B"): ("ex_r16_f32_bonly", "mp_r16_f32_bonly"),
    ("float64", "H"): ("ex_r16_f64_Hd", "mp_r16_f64_Hd"),
    ("float64", "B"): ("ex_r16_f64_bonly", "mp_r16_f64_bonly"),
}
EIGHT_MESH = "(16,32,32) p=2"


def load(root, arm):
    path = os.path.join(root, arm, "relax.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def style(arm):
    _, colour, dash, label = ARMS[arm]
    return arm_style(colour, dash, label=label)


def wall_of_step(j):
    """Wall-clock seconds at every step, interpolated between the qoi samples."""
    q = j["qoi"]
    n = len(j["trace"]["E"])
    return np.interp(np.arange(1, n + 1), np.array(q["it"], float), np.array(q["wall"], float))


def by_panel(arms):
    out = {k: [] for k in PANELS}
    for arm, j in arms:
        out[ARMS[arm][0]].append((arm, j))
    return {k: v for k, v in out.items() if v}


@house_style()
def helicity_figure(arms, figdir):
    panels = by_panel(arms)
    fig, axes = plt.subplots(1, len(panels), figsize=figsize("text", cols=len(panels), aspect=0.8),
                             squeeze=False)
    for ax, (key, group) in zip(axes[0], panels.items()):
        for arm, j in group:
            h = np.array(j["qoi"]["helicity"])
            ax.plot(j["qoi"]["it"], h - j["ic"]["H"], marker=".", **style(arm))
        ax.axhline(0, color="0.7", lw=0.6, zorder=0)
        ax.set_yscale("symlog", linthresh=1e-12)
        ax.set(xlabel="step", title=PANELS[key])
        ax.legend()
    axes[0, 0].set_ylabel(r"$H - H_0$  ($\|B\|_M = 1$)")
    fig.savefig(os.path.join(figdir, "helicity_drift.png"))


@house_style()
def descent_figure(arms, figdir):
    panels = by_panel(arms)
    fig, axes = plt.subplots(2, len(panels), figsize=figsize("text", rows=2, cols=len(panels), aspect=0.8),
                             squeeze=False)
    for col, (key, group) in enumerate(panels.items()):
        for arm, j in group:
            tr = j["trace"]
            E = np.array(tr["E"])
            E0 = j["ic"]["E"]
            plot_trace(axes[0, col], tr["resid"], log=True, **style(arm))
            axes[1, col].plot(wall_of_step(j), (E0 - E) / E0, **style(arm))
        axes[0, col].set(xscale="log", yscale="log", title=PANELS[key])
        axes[1, col].set(xscale="log", yscale="log", xlabel="wall-clock [s]")
        axes[0, col].legend()
    axes[0, 0].set_ylabel("force residual")
    axes[1, 0].set_ylabel(r"$(E_0 - E) / E_0$")
    for ax in axes[0]:
        ax.set_xlabel("step")
    fig.savefig(os.path.join(figdir, "descent.png"))


@house_style()
def picard_figure(arms, figdir):
    fig, axes = plt.subplots(1, 3, figsize=figsize("text", cols=3, aspect=0.8))
    for arm, j in arms:
        tr = j["trace"]
        st = style(arm)
        st["label"] = f"{ARMS[arm][3]}, {ARMS[arm][0]}"
        if max(tr["picard_it"]) > 1:
            plot_trace(axes[0], tr["picard_it"], log=False, **st)
            plot_trace(axes[2], np.array(tr["picard_resid"]), log=True, **st)
        plot_trace(axes[1], tr["dt"], log=True, **st)
    axes[0].set(xlabel="step", ylabel="increment evaluations / step")
    axes[1].set(xlabel="step", ylabel=r"$\Delta t$", yscale="log")
    axes[2].set(xlabel="step", ylabel="Picard defect (last sweep)", yscale="log")
    axes[1].legend()
    fig.savefig(os.path.join(figdir, "picard.png"))


@house_style()
def eight_figure(arms, figdir):
    """Helicity and force traces of (H / B only) x (float32 / float64) x
    (explicit / midpoint) on the ``EIGHT_MESH`` mesh: rows the quantity,
    columns the precision, colour the route, dash the scheme."""
    d = dict(arms)
    fig, axes = plt.subplots(2, 2, figsize=figsize("text", rows=2, cols=2, aspect=0.7))
    for col, prec in enumerate(("float32", "float64")):
        for route, colour in (("H", 1), ("B", 2)):
            for dash, arm in enumerate(EIGHT[(prec, route)]):
                if arm not in d:
                    continue
                j = d[arm]
                st = arm_style(colour, dash, label=f"{'explicit' if dash == 0 else 'midpoint'}, "
                                                   f"{'H' if route == 'H' else 'B only'}")
                h = np.array(j["qoi"]["helicity"])
                axes[0, col].plot(j["qoi"]["it"], h - j["ic"]["H"], marker=".", **st)
                plot_trace(axes[1, col], j["trace"]["resid"], log=True, **st)
        axes[0, col].axhline(0, color="0.7", lw=0.6, zorder=0)
        axes[0, col].set_yscale("symlog", linthresh=1e-12)
        axes[0, col].set(title=f"{EIGHT_MESH}, {prec}", xlabel="step")
        axes[1, col].set(xscale="log", yscale="log", xlabel="step")
        axes[0, col].legend()
    axes[0, 0].set_ylabel(r"$H - H_0$  ($\|B\|_M = 1$)")
    axes[1, 0].set_ylabel("force residual")
    fig.savefig(os.path.join(figdir, "eight.png"))


def summary(arms):
    rows = ["| arm | steps | stop | s/step | eval/step | dt halved | unconverged | E removed | resid final (mean last 100) | identity max | dH abs (from the IC) | dH / H0 |",
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
        tol = j["params"].get("picard_tol") or 0.0
        ident = np.abs(np.array(tr["dE_meas"]) - np.array(tr["dE_pred"])) / E0
        rows.append(
            f"| {arm} | {n} | {s['stop']} | {s['wall'] / max(n, 1):.2f} | {pit.mean():.2f} "
            f"| {int((prs > 0).sum())} | {int((prd > tol).sum()) if tol else 0} "
            f"| {(E0 - tr['E'][-1]) / E0:.4%} | {tr['resid'][-1]:.3e} ({np.mean(tr['resid'][-100:]):.3e}) "
            f"| {ident.max():.2e} | {dh:+.3e} | {dh / j['ic']['H']:+.3e} |")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/midpoint_sweep")
    cli = ap.parse_args()
    every = [(a, load(cli.root, a)) for a in ALL_ARMS]
    every = [(a, j) for a, j in every if j is not None and j.get("trace")]
    arms = [(a, j) for a, j in every if a in ARMS]
    figdir = os.path.join(cli.root, "figures")
    os.makedirs(figdir, exist_ok=True)
    helicity_figure(arms, figdir)
    descent_figure(arms, figdir)
    picard_figure(arms, figdir)
    eight_figure(arms, figdir)
    table = summary(every)
    with open(os.path.join(cli.root, "summary.md"), "w") as fh:
        fh.write(table + "\n")
    print(table)
    print(f"figures in {figdir}")


if __name__ == "__main__":
    main()
