#!/usr/bin/env python
"""The paper's line figures and tables from the run records (login node, matplotlib only).

    python paper/figures.py [--records DIR] [--out DIR] [--only ITEM,...]

--records is the root that paper/runs/*.sh write, <records>/<experiment>/<arm> [outputs]. --out gets the paper's
layout: tables/<table>.tex, figs/pgf/<figure>/<stem>.pgf, and figs/<stem>.{pdf,png} to look at [paper/build];
copying tables/ and figs/pgf/ into the paper installs them. Every item also prints the numbers the paper's text
quotes from it. The PGF needs xelatex on PATH.

A record is a scripts/relax.py run, relax.json with its continuations in cont/, cont/cont/, ...: the trace per
step, the qoi per chunk. s/step is the steady rate over the chunks after the first, which carries the compile; a
wall-time axis starts at the loop start with that compile placed before the first step.
"""
import argparse
import glob
import json
import os
from fractions import Fraction

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.io  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from mrx.plotstyle import CYCLE, DASHES, FS, figsize, house_style  # noqa: E402

COLOURS = [c for c, _ in CYCLE]
RESID = r"$\|F\|^2_{\mathrm{norm}}$"
#: the house style's pgf backend (xelatex, the document's fonts) and the preamble lines the paper also loads
PGF = {"pgf.preamble": r"\usepackage{amssymb}\usepackage[strings]{underscore}\providecommand{\mathdefault}[1]{#1}"}


# ---------------------------------------------------------------------------------------------------- records
class Record:
    """A run and its continuations, the steps after ``steps`` dropped: ``trace`` per step (plus ``steps``, the
    step numbers), ``qoi`` per chunk (the wall of a continuation runs on from its parent's), ``params`` of the
    first part, ``summary`` of the last."""

    def __init__(self, path, steps=None):
        parts = [json.load(open(os.path.join(path, "relax.json")))]
        while os.path.exists(os.path.join(path, "cont", "relax.json")):
            path = os.path.join(path, "cont")
            parts.append(json.load(open(os.path.join(path, "relax.json"))))
        self.params, self.summary = parts[0]["params"], parts[-1]["summary"]
        tkeys = set.intersection(*(set(p["trace"]) for p in parts))
        qkeys = set.intersection(*(set(p["qoi"]) for p in parts))
        trace, qoi = {k: [] for k in tkeys}, {k: [] for k in qkeys}
        for i, p in enumerate(parts):
            for k in tkeys:
                v = list(p["trace"][k])
                if i and k in ("resid", "F"):
                    # the first step after a restart spikes, 3x to 130x, and is gone the step after (a poorer Newton
                    # direction); Tobias 2026-09-19: the plotted trace carries the step before in its place
                    v[0] = trace[k][-1]
                trace[k] += v
            w0 = qoi["wall"][-1] if i else 0.0
            for k in qkeys:
                v = p["qoi"][k][1:] if i else p["qoi"][k]   # a continuation's first sample repeats the restart
                qoi[k] += [w + w0 for w in v] if k == "wall" else v
        self.trace = {k: np.asarray(v, float) for k, v in trace.items()}
        self.trace["steps"] = self.params["start_step"] + 1 + np.arange(len(self.trace["resid"]))
        self.qoi = {k: np.asarray(v, float) for k, v in qoi.items()}
        if steps is not None:
            self.trace = {k: v[self.trace["steps"] <= steps] for k, v in self.trace.items()}
            self.qoi = {k: v[self.qoi["it"] <= steps] for k, v in self.qoi.items()}

    def seconds_per_step(self):
        it, w = self.qoi["it"], self.qoi["wall"]
        return float((w[-1] - w[1]) / (it[-1] - it[1]))

    def wall_minutes(self):
        """The wall time at every step, minutes: the first chunk's excess over the steady cost is the compile,
        placed before the first step."""
        it, w = self.qoi["it"], self.qoi["wall"].copy()
        w[0] = w[1] - np.median(np.diff(w)[1:] / np.diff(it)[1:]) * (it[1] - it[0])
        return np.interp(self.trace["steps"], it, w) / 60.0

    def first_below(self, threshold):
        """(step, minutes) of the first step below ``threshold``, None if the run never gets there."""
        below = np.nonzero(self.trace["resid"] < threshold)[0]
        return None if not len(below) else (int(self.trace["steps"][below[0]]), float(self.wall_minutes()[below[0]]))

    def helicity_drift(self):
        H = self.qoi["helicity"]
        return float((H[-1] - H[0]) / H[0])


def island_width(archive, iota):
    """The paper's island width at a rational ``iota``: over the traced lines whose fitted |iota| lies within
    2e-3 of it, the largest excursion max - min of the logical radius over every crossing on every plane."""
    z = np.load(archive)
    f = str(z["fields"][0])
    near = np.abs(np.abs(z[f"{f}_iota"]) - float(iota)) < 2e-3
    r = np.concatenate([z[k][near] for k in z.files if k.startswith(f"{f}_zeta") and k.endswith("_logr")], axis=1)
    return float(np.nanmax(r) - np.nanmin(r))


def wout(record, name):
    """A scalar of the VMEC wout the run was built on."""
    with scipy.io.netcdf_file(record.params["geometry_path"], "r", mmap=False) as d:
        return float(d.variables[name].data)


# ---------------------------------------------------------------------------------------------------- output
def sci(v, digits=2, times=r"\times"):
    mant, exp = f"{v:.{digits}e}".split("e")
    return rf"${mant} {times} 10^{{{int(exp)}}}$"


def mesh(ns):
    return r"$\mesh{%d}{%d}{%d}$" % tuple(ns)


def thousands(n):
    return f"{n:,}".replace(",", "{,}")


def row(cells):
    return " & ".join(cells) + r" \\"


def write_table(out, name, spec, head, body, pre=()):
    """tables/<name>.tex: the bare tabular (the paper's table environment holds the caption)."""
    os.makedirs(os.path.join(out, "tables"), exist_ok=True)
    lines = ["% generated by paper/figures.py from the run records; do not edit", *pre,
             rf"\begin{{tabular}}{{{spec}}}", r"\toprule", *head, r"\midrule", *body, r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(out, "tables", name + ".tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def save(fig, out, folder, stem):
    """figs/<stem>.pdf and .png, and figs/pgf/<folder>/<stem>.pgf."""
    pgf = os.path.join(out, "figs", "pgf", folder)
    os.makedirs(pgf, exist_ok=True)
    fig.savefig(os.path.join(out, "figs", stem + ".pdf"))
    fig.savefig(os.path.join(out, "figs", stem + ".png"))
    with matplotlib.rc_context(PGF):
        fig.savefig(os.path.join(pgf, stem + ".pgf"), backend="pgf")
    plt.close(fig)
    print(f"wrote figs/{stem}.pdf, figs/pgf/{folder}/{stem}.pgf")


def residual_axes(ax, x):
    ax.set_yscale("log")
    ax.set_xlabel({"wall": "wall time [min]", "steps": "step"}[x])
    ax.set_ylabel(RESID)
    ax.grid(alpha=0.3, which="both")


def two_legends(ax, colours, colour_labels, dash_labels, loc_colour, loc_dash):
    """One legend keyed by colour, one by dash (the arms of a two-factor comparison)."""
    ax.add_artist(ax.legend([Line2D([], [], color=c, lw=1.2) for c in colours], colour_labels, loc=loc_colour,
                            fontsize=FS.annot))
    ax.legend([Line2D([], [], color="0.25", ls=d, lw=1.2) for d in DASHES[:len(dash_labels)]], dash_labels,
              loc=loc_dash, fontsize=FS.annot)


# ---------------------------------------------------------------------------------------------------- vacuum
def symmetry_vacuum(root, out):
    """Tab. 2: the manufactured vacuum in the three symmetry models."""
    body = []
    for n in (26, 34):
        body += [r"\midrule"] if body else []
        for model, label in (("torus", "whole torus, no symmetry"), ("period", "one field period"),
                             ("half", "half a period, stellarator symmetry")):
            path = os.path.join(root, "vacuum_symmetry", f"{model}_{n}", "analytic_vacuum.json")
            r = json.load(open(path))["records"][0]
            A, C = r["routes"]["A"], r["routes"]["C"]
            # the half-period model counts one stellarator parity class, half the field period's DoFs
            dofs = A["n"] // 2 if r["symmetry"] == "stellarator" else A["n"]
            body.append(row([label, mesh(r["ns"]), f"${thousands(dofs)}$", f"{r['t_setup']:.0f}", f"{A['t']:.0f}",
                             f"{C['t']:.0f}", sci(A["relerr"], times=r"\cdot"), sci(C["relerr"], times=r"\cdot")]))
    write_table(out, "symmetry_vacuum_table_jcp", "l l r r r r c c",
                [r"model & splines & DoFs of $V^1$ & setup [s] & $k=1$ solve [s] & $k=2$ [s] & error $k=1$ & "
                 r"error $k=2$ \\"], body, pre=(r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}"))


def vacuum_convergence(root, out):
    """Figs. 4 and 5: the manufactured vacuum field against the analytic one, k = 1 the scalar-potential route
    (A), k = 2 the vector-potential route (C); the discrete harmonic 2-form against the vacuum field of the low-
    and the high-resolution VMEC equilibrium, each floor the mean over p of the finest mesh."""
    err = {}
    for f in sorted(glob.glob(os.path.join(root, "vacuum_analytic", "*", "analytic_vacuum.json"))):
        for rec in json.load(open(f))["records"]:
            for route, r in rec["routes"].items():
                assert (route, rec["p"], rec["n_elements"]) not in err, (route, rec["p"], rec["n_elements"], f)
                err[route, rec["p"], rec["n_elements"]] = r["relerr"]
    with house_style():
        fig, ax = plt.subplots(figsize=figsize("column"))
        for route, k in (("A", 1), ("C", 2)):
            for p in (1, 2, 3, 4):
                n = np.array(sorted(m for r, q, m in err if (r, q) == (route, p)), float)
                e = np.array([err[route, p, m] for m in n])
                ax.plot(1 / n, e, color=COLOURS[p - 1], ls=DASHES[k - 1], marker="o", ms=3, lw=1.0)
                if k == 1:          # the rate h^p over a factor 2 in h, just below the fine end of the k = 1 curve
                    h = np.array([1 / n[-1], 2 / n[-1]])
                    ax.plot(h, 0.4 * e[-1] * (h * n[-1]) ** p, color="0.35", ls="-", lw=0.7)
                    ax.annotate(rf"$h^{p}$", (h[0], 0.4 * e[-1]), xytext=(4, 0), textcoords="offset points", ha="left",
                                va="center", fontsize=FS.annot, color="0.35")
                print(f"k = {k}, p = {p}: n = {int(n[0])} ... {int(n[-1])}, error {e[0]:.2e} ... {e[-1]:.2e}, "
                      f"order {-np.polyfit(np.log(n), np.log(e), 1)[0]:.2f}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.invert_xaxis()
        ax.set_xlim(right=0.010)
        ax.set_xlabel(r"$h$")
        ax.set_ylabel(r"$\|B - B^*\|_{\mathrm{rel}}$")
        ax.grid(alpha=0.3, which="both")
        two_legends(ax, COLOURS[:4], [f"p = {p}" for p in (1, 2, 3, 4)], [r"$k = 1$", r"$k = 2$"], "lower left",
                    "lower center")
        save(fig, out, "vacuum_convergence_analytic", "analytic_vacuum_convergence")

    D = {}
    for ref in ("lowres", "highres"):
        for f in glob.glob(os.path.join(root, "vacuum_vmec", ref, "rung_*", "result.json")):
            r = json.load(open(f))
            D[ref, r["p"], r["ns"][0] - r["p"]] = r["D"]
    with house_style():
        fig, ax = plt.subplots(figsize=figsize("column"))
        for dash, ref in enumerate(("lowres", "highres")):
            for p in (2, 3, 4):
                n = np.array(sorted(m for r, q, m in D if (r, q) == (ref, p)), float)
                ax.plot(1 / n, [D[ref, p, m] for m in n], color=COLOURS[p - 1], ls=DASHES[dash], marker="o", ms=3,
                        lw=1.0)
            finest = max(m for r, _, m in D if r == ref)
            floor = np.mean([D[ref, p, finest] for p in (2, 3, 4)])
            ax.axhline(floor, color="0.5", ls=DASHES[dash], lw=0.8)
            print(f"{ref}: floor {floor:.2e} (the mean over p = 2, 3, 4 at n = {finest})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.invert_xaxis()
        ax.set_xlabel(r"$h$")
        ax.set_ylabel(r"$\|c\,\mathfrak{h} - B_{\mathrm{VMEC}}\|_{\mathrm{rel}}$")
        ax.grid(alpha=0.3, which="both")
        two_legends(ax, COLOURS[1:4], [f"p = {p}" for p in (2, 3, 4)], ["low-resolution VMEC", "high-resolution VMEC"],
                    "center right", "upper right")
        save(fig, out, "vacuum_convergence_vmec", "vmec_vacuum_convergence")


# ---------------------------------------------------------------------------------------------------- gradient descent
def smoothing(root, out):
    """Fig. 8: gradient descent with (gamma = 1) and without (gamma = 0) velocity smoothing; the two-step cycle that
    Sec. 5.3 quotes."""
    g1 = Record(os.path.join(root, "newton_convergence", "gradient_16"))
    g0 = Record(os.path.join(root, "gradient", "gamma0"))
    for name, r in (("gamma = 1", g1), ("gamma = 0", g0)):
        # the cycle over the last 1000 steps: odd and even steps' medians
        s, dt, x = (r.trace[k][-1000:] for k in ("steps", "dt", "resid"))
        odd = s % 2 == 1
        dto, dte = np.median(dt[odd]), np.median(dt[~odd])
        ro, re = np.median(x[odd]), np.median(x[~odd])
        print(f"{name}: {int(s[-1])} steps, {r.wall_minutes()[-1]:.1f} min, {r.seconds_per_step():.3f} s/step; "
              f"dt odd/even {dto:.3e}/{dte:.3e}, 1/dt1 + 1/dt2 = {1 / dto + 1 / dte:.3e}; resid odd/even "
              f"{ro:.3e}/{re:.3e} ({max(ro, re) / min(ro, re) - 1:.1%} apart)")
    t1 = g1.wall_minutes()[-1]
    print(f"gamma = 1 at its end ({t1:.1f} min): resid {g1.trace['resid'][-1]:.3e}")
    w0, s0, x0 = g0.wall_minutes(), g0.trace["steps"], g0.trace["resid"]
    for parity, branch in ((1, "odd"), (0, "even")):
        m = s0 % 2 == parity
        print(f"gamma = 0, {branch} steps, at {t1:.1f} min: resid {np.interp(t1, w0[m], x0[m]):.3e}")
    with house_style():
        fig, ax = plt.subplots(figsize=figsize("column"))
        ax.plot(g1.wall_minutes(), g1.trace["resid"], color="0.35", ls="-", lw=1.5, label=r"$\gamma = 1$")
        for parity in (1, 0):       # the cycle's two branches as two lines
            m = s0 % 2 == parity
            ax.plot(w0[m], x0[m], color=COLOURS[1], ls="--", lw=1.2, label=r"$\gamma = 0$" if parity else None)
        ax.set_xscale("log")
        residual_axes(ax, "wall")
        ax.legend(loc="lower left", fontsize=FS.annot)
        save(fig, out, "lbfgs_smoothing", "lbfgs_smoothing")


def smoothing_constant(root, out):
    """Sec. 5.3: the smoothing constant c of eps = c <g_rr> h_r^2 swept over two decades, 2000 steps each; the
    c = 0.0737 arm is Tab. 3's gradient run, capped there."""
    arms = [Record(os.path.join(root, "gradient", f"smoothing_c{c}")) for c in ("0.007", "0.02")]
    arms += [Record(os.path.join(root, "newton_convergence", "gradient_16"), steps=2000)]
    arms += [Record(os.path.join(root, "gradient", f"smoothing_c{c}")) for c in ("0.2", "0.7")]
    h_r_sq = arms[0].params["h_r_sq"]      # the mesh's <g_rr> h_r^2, the same in every arm
    tails = [np.median(r.trace["resid"][-100:]) for r in arms]
    for r, tail in zip(arms, tails):
        reach = [r.first_below(t) for t in (1e-6, 1e-7)]
        print(f"c = {r.params['velocity_smoothing_scale'] / h_r_sq:.4f}: {r.seconds_per_step():.3f} s/step, "
              f"resid at step {int(r.trace['steps'][-1])} {tail:.3e} (last-100 median, {tail / min(tails):.2f} x the "
              f"best), steps to 1e-6/1e-7 {[m and m[0] for m in reach]}")


def helicity(root, out):
    """Tab. C.6: the helicity drift of the plain and the corrected step."""
    rec = {(prec, step): Record(os.path.join(root, "gradient", f"helicity_{prec}_{step}"))
           for prec in ("mixed", "float64") for step in ("plain", "corrected")}
    body = [row([prec] + [sci(abs(rec[prec, step].helicity_drift())) for step in ("plain", "corrected")])
            for prec in ("mixed", "float64")]
    write_table(out, "helicity_table", "l c c",
                [r" & \multicolumn{2}{c}{$|\Delta \Helicity / \Helicity|$} \\", r"\cmidrule(lr){2-3}",
                 r"precision & plain step & corrected step \\"], body, pre=(r"\centering",))
    for prec in ("mixed", "float64"):
        a, b = rec[prec, "plain"], rec[prec, "corrected"]
        dEa, dEb = (r.qoi["E"][-1] - r.qoi["E"][0] for r in (a, b))
        # gamma = 0 descends in a two-step cycle: the residual over the last 100 steps holds both of its branches
        xa, xb = (r.trace["resid"][-100:].mean() for r in (a, b))
        print(f"{prec}: corrected against plain, energy removed {abs(dEb - dEa) / abs(dEa):.2%} apart, residual over "
              f"the last 100 steps {abs(xb - xa) / xa:.2%} apart")


def velocity(root, out):
    """Tab. C.7: the potential velocity (Tab. 3's gradient run) against the Leray projection."""
    arms = (("potential", Record(os.path.join(root, "newton_convergence", "gradient_16"))),
            ("Leray", Record(os.path.join(root, "gradient", "leray"))))
    body = []
    for name, r in arms:
        reach = [r.first_below(t) for t in (1e-6, 1e-7, 1e-8)]
        body.append(row([name, f"{r.seconds_per_step():.2f}", f"{1e8 * r.trace['resid'].min():.2f}"]
                        + ["--" if m is None else str(m[0]) for m in reach] + [f"{1e6 * r.helicity_drift():.1f}"]))
    write_table(out, "velocity_table", "l c c c c c c",
                [r"velocity & s/step & $\normnorm{F}^2$ & \multicolumn{3}{c}{steps to $\normnorm{F}^2$} & "
                 r"$\Delta \Helicity / \Helicity$ \\", r"\cmidrule(lr){4-6}",
                 r" & & floor $\times 10^{-8}$ & $10^{-6}$ & $10^{-7}$ & $10^{-8}$ & $\times 10^{-6}$ \\"], body,
                pre=(r"\centering", r"\setlength{\tabcolsep}{3pt}"))
    (_, pot), (_, leray) = arms
    rel = np.abs(pot.trace["resid"] - leray.trace["resid"]) / pot.trace["resid"]
    print(f"per-step relative residual difference: max {rel.max():.2e} (step {np.argmax(rel) + 1}), after the first "
          f"six steps {rel[6:].max():.2e}; the potential route costs "
          f"{1 - pot.seconds_per_step() / leray.seconds_per_step():.0%} less per step")


# ---------------------------------------------------------------------------------------------------- Newton
def newton_convergence(root, out):
    """Tab. 3 and Figs. 9, 10: gradient descent and Newton at five resolutions; the numbers of Sec. 5.4 and of the
    caption of Fig. 11 (the n_r = 48 run)."""
    run = lambda arm, steps=None: Record(os.path.join(root, "newton_convergence", arm), steps)  # noqa: E731
    gd = run("gradient_16")
    newton = {n: run(f"newton_{n}") for n in (16, 24, 32, 48, 64)}
    beta_vmec = wout(gd, "betatotal")      # <p> / <B^2 / 2>, as beta_vol

    def dofs(r):
        """The unknowns of B: the Dirichlet 2-form DoFs of one stellarator parity class, at p = 2 and
        (n, 2n, 2n) splines. With m = n_r + p radial splines the field period has 12 m^3 - 100 m^2 + 260 m - 216,
        the reflection fixes 4 n_r - 12 of them, the odd class holds half the rest (exact against the sequences at
        n_r = 12, 16, 20, 24)."""
        n, p = r.params["ns"][0], r.params["p"]
        assert p == 2 and r.params["ns"] == [n, 2 * n, 2 * n] and r.params["symmetry"] == "stellarator"
        m = n + p
        return (12 * m ** 3 - 100 * m ** 2 + 260 * m - 216 - (4 * n - 12)) // 2

    def floored(x):
        """At the floor: the minimum is not the last step, or the last 20 steps are within 3% of the 20 before
        (Tobias 2026-09-22: a run stopped while still descending gets a <=)."""
        return int(np.argmin(x)) < len(x) - 1 or x[-20:].mean() > 0.97 * x[-40:-20].mean()

    def cells(method, r):
        x = r.trace["resid"]
        reach = [r.first_below(t) for t in (1e-7, 1e-8, 1e-9)]
        floor = sci(x.min()) if floored(x) else r"$\leq " + sci(x.min())[1:]
        return ([method, mesh(r.params["ns"]), thousands(dofs(r)), str(int(r.trace["steps"][-1])),
                 f"{r.seconds_per_step():.2f}", floor]
                + ["--" if m is None else str(m[0]) for m in reach]
                + ["--" if m is None else f"{m[1]:.1f}" for m in reach]
                + [f"{1e6 * r.helicity_drift():.1f}", f"{1e3 * (r.qoi['beta_vol'][-1] - beta_vmec) / beta_vmec:.1f}"])

    def power(pairs):
        """The exponent of n_r fitted through (n_r, value), over the rows that have a value."""
        n, v = np.log(np.array([(n, v) for n, v in pairs if v is not None], float)).T
        return r"$n_r^{%.2f}$" % np.polyfit(n, v, 1)[0]

    body = [row(cells("gradient", gd)), r"\midrule"]
    body += [row(cells("Newton" if i == 0 else "", r)) for i, r in enumerate(newton.values())]
    reach = {n: [r.first_below(t) for t in (1e-7, 1e-8, 1e-9)] for n, r in newton.items()}
    fit = ([power((n, dofs(r)) for n, r in newton.items()), "",
            power((n, r.seconds_per_step()) for n, r in newton.items()), ""]
           + [power((n, m[j] and m[j][0]) for n, m in reach.items()) for j in range(3)]
           + [power((n, m[j] and m[j][1]) for n, m in reach.items()) for j in range(3)]
           + [power((n, abs(r.helicity_drift())) for n, r in newton.items()), ""])
    body += [r"\midrule", row([r"fit $\propto n_r$", ""] + fit)]
    write_table(out, "newton_convergence_table_jcp", "l l r r r c r r r r r r r r",
                [r"method & resolution & DOF count & steps & s/step & $\normnorm{F}^2$ & "
                 r"\multicolumn{3}{c}{steps to $\normnorm{F}^2$} & "
                 r"\multicolumn{3}{c}{time to $\normnorm{F}^2$ [min]} & "
                 r"$\Delta \Helicity / \Helicity$ & $\Delta \betavol / \betavol^{\mathrm{VMEC}}$ \\",
                 r"\cmidrule(lr){7-9}\cmidrule(lr){10-12}",
                 r" & & & & [s] & floor & $10^{-7}$ & $10^{-8}$ & $10^{-9}$ & $10^{-7}$ & $10^{-8}$ & $10^{-9}$ & "
                 r"$\times 10^{-6}$ & $\times 10^{-3}$ \\"], body,
                pre=(r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}"))
    s48, s64 = newton[48].seconds_per_step(), newton[64].seconds_per_step()
    floors = [r.trace["resid"].min() for r in newton.values() if floored(r.trace["resid"])]
    print(f"s/step between n_r = 48 and 64: n_r^{np.log(s64 / s48) / np.log(64 / 48):.2f}; floors of the runs at "
          f"their floor {min(floors):.2e} to {max(floors):.2e}; largest helicity drift "
          f"{max(abs(r.helicity_drift()) for r in newton.values()):.2e}; VMEC beta {beta_vmec:.5f}")
    r48 = newton[48]
    print(f"n_r = 48 (Fig. 11): {int(r48.trace['steps'][-1])} steps, final resid {r48.trace['resid'][-1]:.2e}, "
          f"minimum {r48.trace['resid'].min():.2e}, beta_vol {r48.qoi['beta_vol'][0]:.4f} -> "
          f"{r48.qoi['beta_vol'][-1]:.4f}")

    first = run("newton_16", newton[16].params["steps"])      # the run itself, 100 steps, not its continuation
    with house_style():                     # Fig. 9: Newton over the descent's time range, every step a marker
        fig, ax = plt.subplots(figsize=figsize("column"))
        ax.plot(first.wall_minutes(), first.trace["resid"], color=COLOURS[0], label="Newton", ls="-", lw=0.8,
                marker="s", ms=2.2, zorder=3)
        ax.plot(gd.wall_minutes(), gd.trace["resid"], color=COLOURS[1], label="gradient descent", ls="-", lw=0.6,
                marker="o", ms=1.8, mew=0, alpha=0.7)
        ax.set_xscale("log")
        residual_axes(ax, "wall")
        ax.legend(loc="lower left", fontsize=FS.annot)
        save(fig, out, "newton_vs_lbfgs", "newton_vs_lbfgs")
    with house_style():                     # Fig. 10: colour, dash and marker per resolution
        fig, ax = plt.subplots(figsize=figsize("column"))
        for i, (r, marker) in enumerate(zip(newton.values(), "os^Dv")):
            ax.plot(r.wall_minutes(), r.trace["resid"], color=COLOURS[i], ls=DASHES[i], lw=0.8, marker=marker, ms=2.4,
                    mew=0, label="(%d,%d,%d)" % tuple(r.params["ns"]))
        ax.set_xscale("log")
        residual_axes(ax, "wall")
        ax.legend(loc="upper right", fontsize=FS.annot)
        save(fig, out, "newton_resolution", "newton_resolution")


def newton_order_precision(root, out):
    """Figs. C.16 (degree) and C.17 (precision): 40 Newton steps; the reference arm is Tab. 3's n_r = 16 run
    (p = 2, mixed, solver tolerance 1e-8), capped there. Also the tolerance arms and p = 1, which the text quotes."""
    sweep = lambda arm: Record(os.path.join(root, "newton_sweeps", arm))  # noqa: E731
    ref = Record(os.path.join(root, "newton_convergence", "newton_16"), steps=40)
    figs = {"newton_degree": (("p = 2", ref), ("p = 3", sweep("p3")), ("p = 4", sweep("p4"))),
            "newton_precision": (("float64", sweep("float64")), ("mixed", ref), ("float32", sweep("float32")))}
    for name, arms in figs.items():
        with house_style():
            fig, ax = plt.subplots(figsize=figsize("column"))
            for i, (label, r) in enumerate(arms):
                ax.plot(r.trace["steps"], r.trace["resid"], color=COLOURS[i], ls=DASHES[i], lw=1.5, label=label)
            residual_axes(ax, "steps")
            ax.legend(loc="upper right", fontsize=FS.annot)
            save(fig, out, name, name)
    for label, r in (*figs["newton_degree"], ("p = 1", sweep("p1")), *figs["newton_precision"][::2],
                     ("tol 1e-6", sweep("tol1e-6")), ("tol 1e-10", sweep("tol1e-10"))):
        x = r.trace["resid"]
        print(f"{label}: {r.seconds_per_step():.2f} s/step, floor {x.min():.2e} "
              f"(step {int(r.trace['steps'][np.argmin(x)])}), "
              f"final {x[-1]:.2e}, steps to 1e-6/7/8 {[m and m[0] for m in map(r.first_below, (1e-6, 1e-7, 1e-8))]}")
    f64, f32 = sweep("float64").seconds_per_step(), sweep("float32").seconds_per_step()
    print(f"per step against float64: mixed {f64 / ref.seconds_per_step():.2f}x, float32 {f64 / f32:.2f}x faster")
    for tol in ("1e-6", "1e-10"):
        rel = np.abs(sweep(f"tol{tol}").trace["resid"] - ref.trace["resid"]) / ref.trace["resid"]
        print(f"tol {tol} against 1e-8: per-step relative residual difference at most {rel.max():.2e}")


def newton_sweeps(root, out):
    """Tabs. C.8 and C.9: the parallel-flow penalty and the MINRES budget, 200 Newton steps at n_r = 16."""
    head = [r"%s & $\normnorm{F}^2$ floor & $< 10^{-8}$ & s/step & MINRES \\",
            r" & $\times 10^{-10}$ & at step & & it./step \\"]
    for name, first, arms in (("newton_penalty_table", r"$\kappa$", [(f"${k}$", f"kappa{k}") for k in (0, 1, 3, 10)]),
                              ("newton_inner_table", "MINRES", [(f"${n}$", "kappa3" if n == 200 else f"minres{n}")
                                                                for n in (50, 100, 200, 400)])):
        body = []
        for label, arm in arms:
            r = Record(os.path.join(root, "newton_sweeps", arm))
            reach = r.first_below(1e-8)
            body.append(row([label, f"{1e10 * r.trace['resid'].min():.2f}", "--" if reach is None else str(reach[0]),
                             f"{r.seconds_per_step():.1f}", f"{np.abs(r.trace['newton_it']).mean():.0f}"]))
        write_table(out, name, "l r r r r", [head[0] % first, head[1]], body, pre=(r"\centering",))


# ---------------------------------------------------------------------------------------------------- islands
def seeding(root, out):
    """Tab. 4 and Fig. 13: every resonance seeded at its energy-optimal amplitude on the n_r = 32 reference run,
    then relaxed; the numbers of the caption of Fig. 12."""
    seeds = json.load(open(os.path.join(root, "seeding", "seeded.json")))
    final = os.path.join(root, "seeding", "relax32", "trace.npz")
    nested = os.path.join(root, "newton_convergence", "newton_32", "trace.npz")
    body = []
    for c in sorted(seeds["resonances"], key=lambda c: -c["w"])[:8]:
        iota = Fraction(seeds["nfp"] * c["n"], c["m"])
        # a width is measured where the chain is predicted wider than a radial cell; the +- is the same measure on
        # the unseeded equilibrium
        meas = (rf"${island_width(final, iota):.3f} \pm {island_width(nested, iota):.3f}$"
                if c["w"] > seeds["h_r"] else "--")
        body += [r"\addlinespace[0.15em]"] if body else []
        body.append(row([rf"$({c['m']}, {c['n']})$", rf"$\tfrac{{{iota.numerator}}}{{{iota.denominator}}}$",
                         f"${c['r']:.3f}$", sci(c["dBr"], times=r"\cdot"), f"${c['w']:.3f}$",
                         f"${c['w'] / seeds['h_r']:.2f}$", meas]))
    write_table(out, "seed_selection_table", "l c c c c c c",
                [r"$(m, n)$ & $\iota$ & $r_{mn}$ & $\delta \hat B^r_{mn}$ & $w_{\mathrm{pred}}$ & "
                 r"$w_{\mathrm{pred}} / h_r$ & $w_{\mathrm{meas}}$ \\"], body, pre=(r"\centering",))
    print(f"{len(seeds['resonances'])} resonances, h_r = {seeds['h_r']:.4f}")

    seeded = Record(os.path.join(root, "seeding", "relax32"))
    at = seeded.params["start_step"]
    first = Record(os.path.join(root, "newton_convergence", "newton_32"), steps=at)
    print(f"before the seed (step {at}): resid {first.trace['resid'][-1]:.2e}, H {first.qoi['helicity'][-1]:.5e}, "
          f"beta_vol {first.qoi['beta_vol'][-1]:.5f}; final (step {int(seeded.trace['steps'][-1])}): resid "
          f"{seeded.trace['resid'][-1]:.2e}, H {seeded.qoi['helicity'][-1]:.5e}, "
          f"beta_vol {seeded.qoi['beta_vol'][-1]:.5f} "
          f"({seeded.qoi['beta_vol'][-1] - first.qoi['beta_vol'][-1]:+.1e})")
    with house_style():                     # the relaxation to the floor, the seed (dashed), the relaxation after it
        fig, ax = plt.subplots(figsize=figsize("column"))
        ax.axvline(at, color="0.35", lw=0.9, ls="--", zorder=1)
        steps = np.concatenate([first.trace["steps"], seeded.trace["steps"]])
        ax.plot(steps, np.concatenate([first.trace["resid"], seeded.trace["resid"]]), color="black", lw=0.8, zorder=3)
        residual_axes(ax, "steps")
        ax.set_xlim(0, steps[-1])
        save(fig, out, "seed_trace", "seed_trace_parallel")


def reconnection(root, out):
    """Tab. 5: the three arms of the reconnection demonstration after their initial relaxation, the resistive phase
    and the final ideal relaxation."""
    base = os.path.join(root, "newton_convergence", "newton_32")
    nested = {iota: island_width(os.path.join(base, "trace.npz"), iota) for iota in (Fraction(3, 5), Fraction(1, 2))}
    body = []
    for arm, label in (("unseeded", "none"), ("s51q", r"$(5, 1)$, $\iota = \tfrac35$"),
                       ("s61", r"$(6, 1)$, $\iota = \tfrac12$")):
        body += [r"\midrule"] if body else []
        runs = [base if arm == "unseeded" else os.path.join(root, "reconnection", arm, "ideal")]
        runs += [os.path.join(root, "reconnection", arm, phase) for phase in ("resistive", "ideal_after")]
        res = Record(runs[1])
        # the initial relaxation up to the resistive phase's restart: the reference run is not its continuation
        recs = [Record(runs[0], steps=res.params["start_step"]), res, Record(runs[2])]
        for i, (run, r, phase) in enumerate(zip(runs, recs, ("after initial relaxation", "after resistive phase",
                                                            "after final relaxation"))):
            w = [rf"${island_width(os.path.join(run, 'trace.npz'), iota):.3f} \pm {nested[iota]:.3f}$"
                 for iota in nested]
            body.append(row([label if i == 0 else "", phase, sci(r.trace["resid"][-1], 1),
                             f"${r.qoi['beta_vol'][-1]:.4f}$", f"${1e3 * r.qoi['helicity'][-1]:.4f}$"] + w))
        H, b = res.qoi["helicity"], res.qoi["beta_vol"]
        print(f"{arm}: over the resistive phase H {(H[-1] - H[0]) / H[0]:+.1e}, beta_vol {(b[-1] - b[0]) / b[0]:+.2%}")
    write_table(out, "reconnection_demo_table", "l l c c c c c",
                [r"seed & phase & $\normnorm{F}^2$ & $\betavol$ & $\Helicity \times 10^{3}$ & $w_{3/5}$ & "
                 r"$w_{1/2}$ \\"],
                body, pre=(r"\centering",))


ITEMS = {f.__name__: f for f in (symmetry_vacuum, vacuum_convergence, smoothing, smoothing_constant, newton_convergence,
                                 seeding, reconnection, helicity, velocity, newton_order_precision, newton_sweeps)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--records", default="outputs", help="the records root, <records>/<experiment>/<arm> [outputs]")
    ap.add_argument("--out", default="paper/build", help="tables/ and figs/ in the paper's layout [paper/build]")
    ap.add_argument("--only", default=",".join(ITEMS), help=f"comma-separated subset of {', '.join(ITEMS)}")
    cli = ap.parse_args()
    for name in cli.only.split(","):
        print(f"\n=== {name}")
        try:
            ITEMS[name](cli.records, cli.out)
        except FileNotFoundError as e:      # a queued run has no record yet
            print(f"{name}: skipped, no {e.filename}")


if __name__ == "__main__":
    main()
