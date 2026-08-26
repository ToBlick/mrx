"""Overlay the traces of several ``scripts/relax.py`` runs.

    python -u scripts/compare_relaxations.py OUT_STEM label=run_dir [label=run_dir ...]

Reads each run's ``relax.json`` and writes the same nine panels
(``||F||_M``, ``E_0 - E``, ``-dE/dt``, ``dH/H_0``, ``dt``, the CFL number
taken, ``||J||/||B||``, ``beta_vol`` and the line-search cosine) three
times: ``OUT_STEM_time.png`` against relaxation time (the descent's own
time, the sum of the ``dt``), ``OUT_STEM_step.png`` against the step number
and ``OUT_STEM_wall.png`` against wall-clock hours (the clock is read at the
QoI steps; in between it is interpolated); and ``OUT_STEM_runtime.png``
(relaxation time reached against wall-clock hours, seconds per step, and
wall hours per unit of relaxation time). Labels containing ``gamma`` or ``mu`` are drawn dashed,
``eta`` dotted, so the descent variant is visible without the legend.

Options
    --smooth K      running mean over K steps for the noisy per-step traces [25]
    --xlog          logarithmic x axes (useful when runs span decades of time)
    --hf            instead of the nine panels: the helicity (absolute value)
                    next to ``||F||_M``, plain labels, ``OUT_STEM_HF_<x>.png``
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def running_mean(x, k):
    if k <= 1 or len(x) < k:
        return np.asarray(x), np.arange(len(x))
    return np.convolve(x, np.ones(k) / k, mode="valid"), np.arange(k - 1, len(x))


def load(path):
    with open(os.path.join(path, "relax.json")) as fh:
        d = json.load(fh)
    tr, q, s = d["trace"], d["qoi"], d["summary"]
    dt = np.asarray(tr["dt"], dtype=float)
    t = np.cumsum(dt)
    t_of = np.concatenate([[0.0], t])                   # relaxation time after step k
    it = np.asarray(q["it"], dtype=int)
    tq = t_of[np.minimum(it, len(t))]
    E = np.asarray(tr["E"], dtype=float)
    wall_q = np.asarray(q["wall"], dtype=float)
    k = np.arange(1, len(dt) + 1)
    wall_of = np.interp(k, it, wall_q)                                   # seconds after step k
    if len(it) > 1:                                                      # past the last reading: last rate
        rate = (wall_q[-1] - wall_q[-2]) / (it[-1] - it[-2])
        wall_of = np.where(k > it[-1], wall_q[-1] + rate * (k - it[-1]), wall_of)
    wall_of = wall_of / 3600.0
    return dict(dt=dt, t=t, t_of=t_of, it=it, tq=tq, E=E, wall_of=wall_of, F=np.asarray(tr["F"], dtype=float),
                cfl=np.asarray(tr["cfl"], dtype=float), cos=np.asarray(tr["cos"], dtype=float),
                H=np.asarray(q["helicity"], dtype=float), JB=np.asarray(q["JoverB"], dtype=float),
                beta=np.asarray(q["beta_vol"], dtype=float), wall_q=wall_q,
                wall=float(s["wall"]), steps=len(dt), stop=s.get("stop", ""))


def style(label):
    lo = label.lower()
    if "eta" in lo:
        return dict(ls=":", lw=1.8)
    if "gamma" in lo or "mu" in lo:
        return dict(ls="--", lw=1.3)
    return dict(ls="-", lw=1.3)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out")
    ap.add_argument("runs", nargs="+", help="label=run_dir")
    ap.add_argument("--smooth", type=int, default=25)
    ap.add_argument("--xlog", action="store_true")
    ap.add_argument("--hf", action="store_true")
    cli = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = []
    for spec in cli.runs:
        label, path = spec.split("=", 1)
        runs.append((label, load(path)))
    colors = plt.cm.tab10(np.arange(len(runs)) % 10)

    if cli.hf:
        for xkind in ("time", "step", "wall"):
            fig, (aH, aF) = plt.subplots(1, 2, figsize=(12, 4.5))
            for (label, r), c in zip(runs, colors):
                st = style(label)
                x_step = {"time": r["t"], "step": np.arange(1, r["steps"] + 1), "wall": r["wall_of"]}[xkind]
                x_q = {"time": r["tq"], "step": r["it"], "wall": r["wall_q"] / 3600.0}[xkind]
                aH.plot(x_q, r["H"], marker="o", ms=2.5, color=c, label=label, **st)
                Fs, i = running_mean(r["F"], cli.smooth)
                aF.semilogy(x_step[i], Fs, color=c, label=label, **st)
            xl = {"time": "relaxation time", "step": "step", "wall": "wall-clock hours"}[xkind]
            for ax, yl in ((aH, r"$H$"), (aF, rf"$\|F\|_M$ ({cli.smooth}-step mean)")):
                ax.set_xlabel(xl)
                ax.set_ylabel(yl)
                ax.grid(alpha=0.3)
                if cli.xlog:
                    ax.set_xscale("log")
            aH.legend(fontsize=7)
            fig.tight_layout()
            path = f"{cli.out}_HF_{xkind}.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print("->", path, flush=True)
        return

    for xkind in ("time", "step", "wall"):
        fig, axes = plt.subplots(3, 3, figsize=(16, 11))
        (aF, aE, adE), (aH, adt, acfl), (aJ, aB, acos) = axes
        for (label, r), c in zip(runs, colors):
            st = style(label)
            x_step = {"time": r["t"], "step": np.arange(1, r["steps"] + 1), "wall": r["wall_of"]}[xkind]
            x_q = {"time": r["tq"], "step": r["it"], "wall": r["wall_q"] / 3600.0}[xkind]
            lab = f"{label} ({r['steps']} steps, t={r['t'][-1]:.1f})"
            Fs, i = running_mean(r["F"], cli.smooth)
            aF.semilogy(x_step[i], Fs, color=c, label=lab, **st)
            aE.semilogy(x_step, r["E"][0] - r["E"], color=c, **st)
            dEdt = -np.gradient(r["E"], x_step) if xkind != "step" else -np.gradient(r["E"])
            ds, i = running_mean(dEdt, cli.smooth)
            adE.semilogy(x_step[i], np.maximum(ds, 1e-300), color=c, **st)
            aH.plot(x_q, (r["H"] - r["H"][0]) / r["H"][0], marker="o", ms=2.5, color=c, **st)
            adt.semilogy(x_step, r["dt"], color=c, **st)
            cs, i = running_mean(r["cfl"], cli.smooth)
            acfl.semilogy(x_step[i], cs, color=c, **st)
            aJ.plot(x_q, r["JB"], marker="o", ms=2.5, color=c, **st)
            aB.plot(x_q, r["beta"], marker="o", ms=2.5, color=c, **st)
            cs, i = running_mean(r["cos"], cli.smooth)
            acos.plot(x_step[i], cs, color=c, **st)
        xl = {"time": "relaxation time", "step": "step", "wall": "wall-clock hours"}[xkind]
        for ax, yl in zip(axes.ravel(), (rf"$\|F\|_M$ ({cli.smooth}-step mean)", r"$E_0 - E$",
                                         rf"$-dE/d{ {'time': 't', 'step': 'n', 'wall': 't_{wall}'}[xkind] }$ ({cli.smooth}-step mean)",
                                         r"$\Delta H / H_0$", r"$dt$ (line search)", f"CFL number taken ({cli.smooth}-step mean)",
                                         r"$\|J\| / \|B\|$", r"$\beta_{vol}$", f"line-search cosine ({cli.smooth}-step mean)")):
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
            ax.grid(alpha=0.3)
            if cli.xlog:
                ax.set_xscale("log")
        aH.axhline(0.0, color="k", lw=0.5)
        aF.legend(fontsize=7)
        fig.tight_layout()
        path = f"{cli.out}_{xkind}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print("->", path, flush=True)

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 4.5))
    labels = [label for label, _ in runs]
    for (label, r), c in zip(runs, colors):
        a1.plot(r["wall_q"] / 3600.0, r["tq"], marker="o", ms=2.5, color=c, label=label, **style(label))
    a1.set_xlabel("wall-clock hours")
    a1.set_ylabel("relaxation time reached")
    a1.grid(alpha=0.3)
    a1.legend(fontsize=7)
    a2.barh(labels, [r["wall"] / r["steps"] for _, r in runs], color=colors)
    a2.set_xlabel("seconds per step")
    a2.invert_yaxis()
    a3.barh(labels, [r["wall"] / 3600.0 / r["t"][-1] for _, r in runs], color=colors)
    a3.set_xlabel("wall hours per unit relaxation time")
    a3.invert_yaxis()
    for ax in (a2, a3):
        ax.grid(alpha=0.3, axis="x")
        ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    path = f"{cli.out}_runtime.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("->", path, flush=True)


if __name__ == "__main__":
    main()
