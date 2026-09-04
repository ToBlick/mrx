"""Figures and table rows for the li383 publication runs (docs/research/li383_sweep_results_2026-09-02.md).

Reads ``relax.json`` (and ``poincare/sections.npz`` where present) of the
arms in ``outputs/li383_pub`` and the two current-reader arms in
``outputs/li383_axisfix``; writes ``outputs/li383_pub/figures/*.png`` and
``outputs/li383_pub/tables.md`` (the markdown rows of sections 4 and 5 of
the note). Arms that have not finished are skipped.

    python scripts/li383_pub_figures.py [--root outputs]
"""

import argparse
import glob
import math
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mrx.plotstyle import LEFT, RIGHT, house_style  # noqa: E402

NFP = 3
GPU_H = 1.0 / 3600.0


def load(root, sub, arm):
    f = os.path.join(root, sub, arm, "relax.json")
    if not os.path.exists(f):
        return None
    j = json.load(open(f))
    j["arm"] = arm
    j["dir"] = os.path.join(root, sub, arm)
    return j


def sections(j):
    f = os.path.join(j["dir"], "poincare", "sections.npz")
    return np.load(f, allow_pickle=True) if os.path.exists(f) else None


def island_width(z, m, n, tol=2e-3, tag="final"):
    """Width (in logical rho) of the chain at |iota| = nfp n / m in the ``tag``
    sections: the largest max(r) - min(r) over all crossings on all planes of
    any non-chaotic line whose fitted iota sits on the rational to within
    ``tol``. A line inside the island near its separatrix spans the full island
    width; a nested surface spans nothing. 0 when no line is on the rational."""
    target = NFP * n / m
    iota = np.abs(z[f"{tag}_iota"])
    keep = z[f"{tag}_keep"] & ~z[f"{tag}_chaotic"]
    on = keep & (np.abs(iota - target) < tol)
    if not on.any():
        return 0.0
    keys = [k for k in z.files if k.startswith(f"{tag}_zeta") and k.endswith("_logr")]
    lr = np.concatenate([z[k][on] for k in keys], axis=1)
    return float(np.max(np.nanmax(lr, axis=1) - np.nanmin(lr, axis=1)))


def helicity_drift(j):
    h = np.asarray(j["qoi"]["helicity"], dtype=float)
    return float((h[-1] - h[0]) / h[0])


def chaotic(j):
    z = sections(j)
    return int(z["final_chaotic"].sum()) if z is not None else None


def row_common(j):
    p, s, t = j["params"], j["summary"], j["trace"]
    steps = len(t["F"])
    return dict(
        ns=",".join(str(v) for v in p["ns"]),
        p=p["p"],
        g=p["velocity_smoothing_order"],
        prec="f64" if p["precision"] == "float64" else "f32",
        steps=steps,
        stop=s["stop"],
        spst=s["wall"] / steps,
        F0=t["F"][0],
        F1=t["F"][-1],
        dH=helicity_drift(j),
        beta=s["beta_vol"],
        chaotic=chaotic(j),
        gpuh=s["wall"] * GPU_H,
    )


def fmt(v, kind):
    if v is None:
        return "--"
    return {
        "e": f"{v:.2e}",
        "e1": f"{v:.1e}",
        "f2": f"{v:.2f}",
        "f3": f"{v:.3f}",
        "f4": f"{v:.4f}",
        "d": f"{v}",
        "s": str(v),
    }[kind]


def reader_rows(arms):
    out = []
    for j in arms:
        r = row_common(j)
        ref = "ns 49" if "1.4m" in j["params"]["geometry"] else "ns 16"
        out.append(
            "| "
            + " | ".join(
                [
                    j["arm"],
                    r["ns"],
                    str(r["p"]),
                    str(r["g"]),
                    r["prec"],
                    ref,
                    str(r["steps"]),
                    r["stop"],
                    fmt(r["spst"], "f2"),
                    f"{r['F0']:.2e} -> {r['F1']:.2e}",
                    fmt(r["dH"], "e1"),
                    fmt(r["beta"], "f4"),
                    fmt(r["chaotic"], "d"),
                    fmt(r["gpuh"], "f2"),
                ]
            )
            + " |"
        )
    return out


def seeded_rows(arms):
    out = []
    for j in arms:
        r = row_common(j)
        seed = j["params"]["seed"]
        z = sections(j)
        if seed:
            m, n = (int(float(v)) for v in seed.split(",")[:2])
            wtxt = "--" if z is None else f"{island_width(z, m, n):.3f}"
            seedtxt, epstxt = f"({m}, {n})", f"{j['params']['seed_eps']:.0e}"
        else:
            wtxt, seedtxt, epstxt = "--", "--", "0"
        out.append(
            "| "
            + " | ".join(
                [
                    j["arm"],
                    seedtxt,
                    epstxt,
                    str(r["g"]),
                    r["ns"],
                    str(r["steps"]),
                    r["stop"],
                    fmt(r["spst"], "f2"),
                    f"{r['F0']:.2e} -> {r['F1']:.2e}",
                    fmt(r["dH"], "e1"),
                    wtxt,
                    fmt(r["chaotic"], "d"),
                    fmt(r["gpuh"], "f2"),
                ]
            )
            + " |"
        )
    return out


def F(j):
    return np.asarray(j["trace"]["F"])


def it(j):
    return np.arange(1, len(F(j)) + 1)


BLOCK = 100  # steps per block in the trace plots


def blocked(y, log=True, w=BLOCK):
    """Block means of a per-step trace and the +-1 sd band across each block,
    (centre step, mean, lower, upper). ``log`` takes the statistics in log
    space (geometric mean, multiplicative sd) for a log axis. The last
    partial block is kept."""
    y = np.asarray(y, float)
    n = len(y)
    edges = list(range(0, n, w)) + [n]
    x, m, lo, hi = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        blk = y[a:b]
        if log:
            blk = np.log(blk[blk > 0])  # E_0 - E is 0 at step 1
            if len(blk) == 0:
                continue
        mu, sd = blk.mean(), blk.std()
        x.append(0.5 * (a + b + 1))
        m.append(mu)
        lo.append(mu - sd)
        hi.append(mu + sd)
    f = np.exp if log else np.asarray
    return np.asarray(x), f(np.asarray(m)), f(np.asarray(lo)), f(np.asarray(hi))


def plot_trace(ax, y, log=True, **kw):
    """One block-averaged trace with its sd ribbon; ``kw`` goes to the line."""
    x, m, lo, hi = blocked(y, log)
    (line,) = ax.plot(x, m, **kw)
    ax.fill_between(x, lo, hi, color=line.get_color(), alpha=0.2, lw=0)
    return line


def plot_lines(ax, entries):
    for j, lab, st in entries:
        if j is not None:
            plot_trace(ax, F(j), lw=1.0, ls=st, label=lab)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel(r"$\|F\|_M$")
    ax.axhline(1e-3, color="k", lw=0.6, ls=":")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


HSWEEP = ["h8_p2_g1", "h12_p2_g1", "h16_p2_g1", "h24_p2_g1", "h32_p2_g1"]
PSWEEP = ["h16_p1_g1", "h16_p2_g1", "h16_p3_g1", "h16_p4_g1"]


def hsweep_rows(arms, key="ns"):
    """Rows of the h-sweep (``key="ns"``) or the p-sweep (``key="p"``) table."""
    out = []
    for j in arms:
        r = row_common(j)
        f = F(j)
        out.append(
            "| "
            + " | ".join(
                [
                    j["arm"],
                    r["ns"] if key == "ns" else str(r["p"]),
                    fmt(j["params"]["velocity_smoothing_scale"], "e1"),
                    str(r["steps"]),
                    r["stop"],
                    fmt(r["spst"], "f2"),
                    f"{r['F0']:.2e} -> {r['F1']:.2e}",
                    fmt(float(f.min()), "e"),
                    fmt(j["summary"]["resid_window_mean"], "e"),
                    fmt(r["dH"], "e1"),
                    fmt(r["beta"], "f4"),
                    fmt(r["chaotic"], "d"),
                    fmt(r["gpuh"], "f2"),
                ]
            )
            + " |"
        )
    return out


def hsweep_figure(arms, figdir):
    fig, ax = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    plot_lines(
        ax[0, 0],
        [
            (
                j,
                f"({j['params']['ns'][0]},{j['params']['ns'][1]},{j['params']['ns'][2]})",
                "-",
            )
            for j in arms
        ],
    )
    ax[0, 0].set_title(r"p=2, $\gamma=1$: force residual under h-refinement")
    n = np.array([j["params"]["ns"][0] for j in arms], float)
    ax[0, 1].loglog(
        n, [j["ic"]["F"] for j in arms], "s--", label=r"IC $\|F\|_M$ (reference file)"
    )
    ax[0, 1].loglog(n, [float(F(j).min()) for j in arms], "o-", label=r"min $\|F\|_M$")
    ax[0, 1].loglog(
        n,
        [j["summary"]["resid_window_mean"] for j in arms],
        "^-",
        label="window mean at stop",
    )
    for q, st in ((1, ":"), (2, "-.")):
        ax[0, 1].loglog(
            n,
            F(arms[0]).min() * (n[0] / n) ** q,
            color="k",
            lw=0.6,
            ls=st,
            label=f"$n^{{-{q}}}$",
        )
    ax[0, 1].set_xlabel("$n_r$ (mesh $(n, 2n, 2n)$)")
    ax[0, 1].set_ylabel(r"$\|F\|_M$")
    ax[0, 1].set_xticks(n)
    ax[0, 1].set_xticklabels([str(int(v)) for v in n])
    ax[0, 1].tick_params(axis="x", which="minor", labelbottom=False)
    ax[0, 1].grid(alpha=0.3, which="both")
    ax[0, 1].legend(fontsize=8)
    ax[0, 1].set_title("floor versus resolution")
    for j in arms:
        z = sections(j)
        if z is None:
            continue
        keep = z["final_keep"] & ~z["final_chaotic"]
        r, io = z["final_seed_r"][keep], z["final_iota"][keep]
        o = np.argsort(
            r
        )  # the archive stores the two golden-angle rays one after the other
        ax[1, 0].plot(
            r[o], io[o], ".-", ms=3, lw=0.8, label=f"n={j['params']['ns'][0]}"
        )
    z = sections(arms[-1])
    if z is not None:
        keep = z["ic_keep"] & ~z["ic_chaotic"]
        r, io = z["ic_seed_r"][keep], z["ic_iota"][keep]
        o = np.argsort(r)
        ax[1, 0].plot(
            r[o], io[o], "k:", lw=1.0, label=f"IC, n={arms[-1]['params']['ns'][0]}"
        )
    ax[1, 0].set_xlabel(r"$\rho$")
    ax[1, 0].set_ylabel(r"$\iota$ (final)")
    ax[1, 0].grid(alpha=0.3)
    ax[1, 0].legend(fontsize=8)
    ax[1, 0].set_title("rotational transform after relaxation")
    for j in arms:
        E = np.asarray(j["trace"]["E"])
        plot_trace(ax[1, 1], E[0] - E, lw=1.0, label=f"n={j['params']['ns'][0]}")
    ax[1, 1].set_xscale("log")
    ax[1, 1].set_yscale("log")
    ax[1, 1].set_xlabel("step")
    ax[1, 1].set_ylabel("$E_0 - E$")
    ax[1, 1].grid(alpha=0.3)
    ax[1, 1].legend(fontsize=8)
    ax[1, 1].set_title("energy released")
    fig.suptitle(r"li383 (ns = 49 reference), p=2, $\gamma=1$, $\mu = 0.064/n_r^2$")
    fig.savefig(os.path.join(figdir, "hsweep_p2.png"), dpi=150)


ETAS = ["1e-8", "3e-8", "1e-7", "3e-7", "1e-6", "1e-5", "1e-4"]


def ideal_tail(j):
    """Mean ||F|| over the last 500 steps, where the tanh schedule has eta ~ 0."""
    return float(F(j)[-500:].mean())


def eta_rows(arms):
    out = []
    for j in arms:
        r = row_common(j)
        q = j["qoi"]
        seed = j["params"]["seed"]
        if seed:
            m, n = (int(float(v)) for v in seed.split(",")[:2])
            z = sections(j)
            wtxt = "--" if z is None else f"{island_width(z, m, n):.3f}"
        else:
            wtxt = "--"
        h = np.asarray(q["helicity"])
        out.append(
            "| "
            + " | ".join(
                [
                    j["arm"],
                    f"{j['params']['eta_max']:.0e}",
                    str(j["params"]["eta_every"]),
                    "(6, 1) 3e-03" if seed else "--",
                    str(r["steps"]),
                    r["stop"],
                    fmt(r["spst"], "f2"),
                    f"{r['F0']:.2e} -> {r['F1']:.2e}",
                    fmt(float(F(j).min()), "e"),
                    fmt(ideal_tail(j), "e"),
                    f"{q['JoverB'][0]:.3f} -> {q['JoverB'][-1]:.3f}",
                    f"{q['beta_vol'][0]:.4f} -> {q['beta_vol'][-1]:.4f}",
                    fmt(float(h[-1] - h[0]), "e1"),
                    wtxt,
                    fmt(r["chaotic"], "d"),
                    fmt(r["gpuh"], "f2"),
                ]
            )
            + " |"
        )
    return out


def eta_figure(plain, seeded, ideal, figdir):
    fig, ax = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    plot_lines(
        ax[0, 0],
        [(ideal, r"$\eta = 0$", "-")]
        + [(j, rf"$\eta_{{max}}$ = {j['params']['eta_max']:.0e}", "-") for j in plain],
    )
    ax[0, 0].set_title("unseeded: force residual, tanh schedule")
    tw = ax[0, 0].twinx()
    if plain:
        j = plain[-1]
        e = np.asarray(j["trace"]["eta"]) / j["params"]["eta_max"]
        tw.plot(it(j), e, "k:", lw=0.8)
        tw.set_ylabel(r"$\eta / \eta_{max}$ (dotted)")
        tw.set_ylim(0, 1.05)
    for arms, ref, lab, mk in (
        (plain, ideal, "unseeded", "o-"),
        (seeded, None, "seeded (6,1) 3e-3", "s--"),
    ):
        if not arms:
            continue
        e = np.array([j["params"]["eta_max"] for j in arms])
        ax[0, 1].semilogx(
            e, [ideal_tail(j) for j in arms], mk, label=f"{lab}: last 500 steps"
        )
        ax[0, 1].semilogx(
            e, [float(F(j).min()) for j in arms], mk, mfc="none", label=f"{lab}: min"
        )
        if ref is not None:
            ax[0, 1].axhline(
                ideal_tail(ref),
                color="C0" if lab == "unseeded" else "C1",
                lw=0.6,
                ls=":",
            )
    ax[0, 1].set_yscale("log")
    ax[0, 1].set_xlabel(r"$\eta_{max}$")
    ax[0, 1].set_ylabel(r"$\|F\|_M$")
    ax[0, 1].grid(alpha=0.3, which="both")
    ax[0, 1].legend(fontsize=8)
    ax[0, 1].set_title(r"residual after the resistive phase (dotted: $\eta = 0$ runs)")
    for j, lab in [(ideal, r"$\eta = 0$")] + [
        (j, rf"$\eta_{{max}}$ = {j['params']['eta_max']:.0e}") for j in plain
    ]:
        if j is None:
            continue
        q = j["qoi"]
        ax[1, 0].plot(q["it"], q["JoverB"], ".-", ms=3, lw=0.9, label=lab)
    ax[1, 0].set_xlabel("step")
    ax[1, 0].set_ylabel(r"$\|J\| / \|B\|$")
    ax[1, 0].grid(alpha=0.3)
    ax[1, 0].legend(fontsize=8)
    ax[1, 0].set_title("current surviving the resistive phase (unseeded)")
    if seeded:
        e = np.array([j["params"]["eta_max"] for j in seeded])
        ws = [
            island_width(sections(j), 6, 1) if sections(j) is not None else np.nan
            for j in seeded
        ]
        ax[1, 1].semilogx(e, ws, "s-", label="final width")
        z0 = next((sections(j) for j in seeded if sections(j) is not None), None)
        if z0 is not None:
            ax[1, 1].axhline(
                island_width_ic(z0, 6, 1), color="k", lw=0.6, ls=":", label="seeded IC"
            )
        ax[1, 1].set_xlabel(r"$\eta_{max}$")
        ax[1, 1].set_ylabel(r"final island width in $\rho$")
        ax[1, 1].grid(alpha=0.3, which="both")
        ax[1, 1].legend(fontsize=8)
        ax[1, 1].set_title(r"(6, 1) chain at $\iota = 1/2$, seed $\epsilon$ = 3e-3")
    fig.suptitle(
        r"li383 (ns = 49), (16,32,32) p=2 $\gamma=1$: resistivity, tanh schedule over 5000 steps"
    )
    fig.savefig(os.path.join(figdir, "eta_sweep.png"), dpi=150)


def eta_traces(
    arms,
    ideal,
    ideal_seeded,
    figdir,
    title="resistivity traces, tanh schedule over 5000 steps",
):
    """1-D traces of every resistivity arm against the ideal twins: residual,
    ||J||/||B||, beta, helicity and energy released vs step, log x. A floored
    arm (``*_floor1e-5``) is shown only when its floor-0 rerun is missing."""
    fig, ax = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    ax = ax.ravel()
    names = {j["arm"] for j in arms}
    arms = [
        j
        for j in arms
        if not (
            j["arm"].endswith("_floor1e-5") and j["arm"][: -len("_floor1e-5")] in names
        )
    ]
    entries = [
        (ideal, r"$\eta = 0$", "k", "-"),
        (ideal_seeded, r"$\eta = 0$ seeded (p=3)", "k", "--"),
    ]
    colors = {}
    for j in arms:
        colors.setdefault(j["arm"].replace("s61_", ""), f"C{len(colors)}")
        seeded = bool(j["params"]["seed"])
        lab = eta_label(j) + (" seeded" if seeded else "")
        if j["arm"].endswith("_floor1e-5"):
            lab += " (floor 1e-5)"
        entries.append(
            (j, lab, colors[j["arm"].replace("s61_", "")], "--" if seeded else "-")
        )
    for j, lab, c, ls in entries:
        if j is None:
            continue
        q = j["qoi"]
        E = np.asarray(j["trace"]["E"])
        h = np.asarray(q["helicity"])
        plot_trace(ax[0], F(j), color=c, ls=ls, lw=0.9, label=lab)
        ax[1].plot(q["it"], q["JoverB"], ".-", ms=2, color=c, ls=ls, lw=0.9, label=lab)
        ax[2].plot(
            q["it"], q["beta_vol"], ".-", ms=2, color=c, ls=ls, lw=0.9, label=lab
        )
        ax[3].plot(q["it"], h, ".-", ms=2, color=c, ls=ls, lw=0.9, label=lab)
        plot_trace(ax[4], E[0] - E, color=c, ls=ls, lw=0.9, label=lab)
        if j["params"]["eta_max"] > 0:
            ax[5].plot(
                it(j), np.asarray(j["trace"]["eta"]), color=c, ls=ls, lw=0.9, label=lab
            )
    ax[0].set_yscale("log")
    ax[0].set_ylabel(r"$\|F\|_M$")
    ax[1].set_ylabel(r"$\|J\| / \|B\|$")
    ax[2].set_ylabel(r"$\beta_{vol}$")
    ax[3].set_ylabel("helicity")
    ax[4].set_yscale("log")
    ax[4].set_ylabel("$E_0 - E$")
    ax[5].set_yscale("log")
    ax[5].set_ylabel(r"$\eta$ (schedule)")
    for a in ax:
        a.set_xscale("log")
        a.set_xlabel("step")
        a.grid(alpha=0.3, which="both")
    ax[0].legend(fontsize=7)
    fig.suptitle(r"li383 (ns = 49), (16,32,32) p=2 $\gamma=1$: " + title)
    fig.savefig(os.path.join(figdir, "eta_traces.png"), dpi=150)


def eta_islands(plain, seeded, figdir):
    """Final width of the (6, 1) chain at iota = 1/2 and the (5, 1) chain at 3/5
    against eta_max, unseeded and seeded arms of the same mesh and degree
    (eta = 0 twins included; drawn one decade left of the smallest eta). A
    hollow marker at zero means the final profile no longer crosses the
    rational."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    etas = [
        j["params"]["eta_max"] for j in plain + seeded if j["params"]["eta_max"] > 0
    ]
    x0 = min(etas) / 10 if etas else 1e-8

    def xpos(j):
        e = j["params"]["eta_max"]
        return e if e > 0 else x0

    for a, (m, n) in zip(ax, ((6, 1), (5, 1))):
        target = NFP * n / m
        for arms, lab, ls in (
            (plain, "unseeded", "-"),
            (seeded, "seeded (6,1) 3e-3", "--"),
        ):
            pts = []
            for j in arms:
                z = sections(j)
                if z is None:
                    continue
                keep = z["final_keep"] & ~z["final_chaotic"]
                io = np.abs(z["final_iota"][keep])
                present = io.min() <= target <= io.max()
                w = island_width(z, m, n) if present else 0.0
                pts.append((xpos(j), w, present))
            if not pts:
                continue
            pts.sort()
            x = [q[0] for q in pts]
            (l1,) = a.plot(x, [q[1] for q in pts], "s" + ls, label=lab)
            for q in pts:
                if not q[2]:
                    a.plot([q[0]], [0.0], "o", mfc="none", color=l1.get_color(), ms=10)
        if seeded and m == 6:
            z0 = next((sections(j) for j in seeded if sections(j) is not None), None)
            if z0 is not None:
                a.axhline(
                    island_width_ic(z0, 6, 1),
                    color="k",
                    lw=0.6,
                    ls=":",
                    label="seeded IC",
                )
        a.set_xscale("log")
        ticks = sorted(set([x0] + etas))
        a.set_xticks(ticks)
        a.set_xticklabels(["0" if t == x0 else f"{t:.0e}" for t in ticks])
        a.set_xlabel(r"$\eta_{max}$ (tanh schedule)")
        a.set_ylabel(r"final island width in $\rho$")
        a.set_ylim(bottom=-0.005)
        a.grid(alpha=0.3, which="both")
        a.legend(fontsize=8)
        g = math.gcd(NFP * n, m)
        a.set_title(rf"({m}, {n}) chain at $\iota = {NFP * n // g}/{m // g}$")
    fig.suptitle(
        r"li383 (ns = 49), (16,32,32) p=2 $\gamma=1$: island widths after the resistive phase"
    )
    fig.savefig(os.path.join(figdir, "eta_islands.png"), dpi=150)


def psweep_figure(arms, figdir):
    """The p-sweep twin of :func:`hsweep_figure`: degree on the x axis."""
    fig, ax = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    plot_lines(ax[0, 0], [(j, f"p={j['params']['p']}", "-") for j in arms])
    ax[0, 0].set_title(r"(16,32,32), $\gamma=1$: force residual under p-refinement")
    pp = np.array([j["params"]["p"] for j in arms], float)
    ax[0, 1].semilogy(
        pp, [j["ic"]["F"] for j in arms], "s--", label=r"IC $\|F\|_M$ (reference file)"
    )
    ax[0, 1].semilogy(
        pp, [float(F(j).min()) for j in arms], "o-", label=r"min $\|F\|_M$"
    )
    ax[0, 1].semilogy(
        pp,
        [j["summary"]["resid_window_mean"] for j in arms],
        "^-",
        label="window mean at stop",
    )
    ax[0, 1].set_xlabel("$p$")
    ax[0, 1].set_ylabel(r"$\|F\|_M$")
    ax[0, 1].set_xticks(pp)
    ax[0, 1].grid(alpha=0.3, which="both")
    ax[0, 1].legend(fontsize=8)
    ax[0, 1].set_title("floor versus degree")
    for j in arms:
        z = sections(j)
        if z is None:
            continue
        keep = z["final_keep"] & ~z["final_chaotic"]
        r, io = z["final_seed_r"][keep], z["final_iota"][keep]
        o = np.argsort(r)
        ax[1, 0].plot(r[o], io[o], ".-", ms=3, lw=0.8, label=f"p={j['params']['p']}")
    z = sections(arms[-1])
    if z is not None:
        keep = z["ic_keep"] & ~z["ic_chaotic"]
        r, io = z["ic_seed_r"][keep], z["ic_iota"][keep]
        o = np.argsort(r)
        ax[1, 0].plot(
            r[o], io[o], "k:", lw=1.0, label=f"IC, p={arms[-1]['params']['p']}"
        )
    ax[1, 0].set_xlabel(r"$\rho$")
    ax[1, 0].set_ylabel(r"$\iota$ (final)")
    ax[1, 0].grid(alpha=0.3)
    ax[1, 0].legend(fontsize=8)
    ax[1, 0].set_title("rotational transform after relaxation")
    for j in arms:
        E = np.asarray(j["trace"]["E"])
        plot_trace(ax[1, 1], E[0] - E, lw=1.0, label=f"p={j['params']['p']}")
    ax[1, 1].set_xscale("log")
    ax[1, 1].set_yscale("log")
    ax[1, 1].set_xlabel("step")
    ax[1, 1].set_ylabel("$E_0 - E$")
    ax[1, 1].grid(alpha=0.3)
    ax[1, 1].legend(fontsize=8)
    ax[1, 1].set_title("energy released")
    fig.suptitle(
        r"li383 (ns = 49 reference), (16,32,32), $\gamma=1$, $\mu$ = 2.5e-4: p-sweep"
    )
    fig.savefig(os.path.join(figdir, "psweep_p16.png"), dpi=150)


def eta_label(j):
    """Legend text of a resistive arm: the tanh rung by eta_max, a pulse arm by
    eta_max, window and repeat."""
    prm = j["params"]
    e = prm["eta_max"]
    if prm.get("eta_schedule") == "pulse":
        start, width, period = prm["eta_pulse"]
        txt = rf"pulse {e:.1e} $\times$ {width} steps @ {start}"
        return txt + (f" every {period}" if period else "")
    return rf"$\eta_{{max}}$ = {e:.0e}"


def dose(j):
    """The resistive dose actually applied, sum of eta dt over the trace."""
    t = j["trace"]
    return float(np.sum(np.asarray(t["eta"]) * np.asarray(t["dt"])))


def reconnect_rows(j):
    """One row per reconnection of a --reconnect-every arm: the residual's
    chunk mean before the solve, the dose, and what the solve did to |F|,
    helicity, current and beta; widths and chaotic lines from the arm's
    sections, where the fields before each solve are the ``reconnect<k>``
    tags (poincare_relax.py --fields reconnect, traced in one call with ic
    and final)."""
    z = sections(j)
    rows = [
        "| k | step | resid | eps | `||F||` before -> after | H before -> after | dH | "
        "J/B before -> after | beta_vol before -> after | (5,1) width | (6,1) width | chaotic |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for ev in j["reconnect"]:
        tag = f"reconnect{ev['k']}"
        traced = z is not None and f"{tag}_iota" in z.files
        w5 = fmt(island_width(z, 5, 1, tag=tag), "f3") if traced else "-"
        w6 = fmt(island_width(z, 6, 1, tag=tag), "f3") if traced else "-"
        ch = int(z[f"{tag}_chaotic"].sum()) if traced else "-"
        rows.append(
            f"| {ev['k']} | {ev['it']} | {ev['resid']:.2e} | {ev['eps']:.2e} | "
            f"{ev['F_before']:.2e} -> {ev['F_after']:.2e} | "
            f"{ev['helicity_before']:+.4e} -> {ev['helicity_after']:+.4e} | "
            f"{ev['helicity_after'] - ev['helicity_before']:+.2e} | "
            f"{ev['JoverB_before']:.3f} -> {ev['JoverB_after']:.3f} | "
            f"{ev['beta_vol_before']:.4f} -> {ev['beta_vol_after']:.4f} | {w5} | {w6} | {ch} |"
        )
    return rows


def save_figure(fig, png_path, pgf=True):
    """PNG at the house dpi and, with ``pgf``, the same figure through the
    pgf backend under ``pgf/`` beside it (vector LaTeX, needs xelatex on PATH; the
    including document must ``\\usepackage[strings]{underscore}``, as for
    the section pages of scripts/poincare_relax.py)."""
    fig.savefig(png_path)
    if not pgf:
        return
    pgf_dir = os.path.join(os.path.dirname(png_path), "pgf")
    os.makedirs(pgf_dir, exist_ok=True)
    pgf_path = os.path.join(pgf_dir, os.path.splitext(os.path.basename(png_path))[0] + ".pgf")
    try:
        with matplotlib.rc_context({"pgf.preamble": r"\usepackage[strings]{underscore}"}):
            fig.savefig(pgf_path, backend="pgf")
    except Exception as exc:      # noqa: BLE001 -- the .pgf is an optional artifact
        if os.path.exists(pgf_path):
            os.remove(pgf_path)
        print(f"  (pgf skipped -- needs xelatex on PATH: {type(exc).__name__}: {str(exc)[:80]})")


def ladder_figure(j, ideal, figdir):
    """The reconnection ladder against the ideal run of the same rung, six
    panels in the house style: force residual, energy released and the CFL
    number taken (100-step block means, +-1 sd), helicity, ||J||/||B|| and
    beta (the qoi samples); dotted lines mark the reconnections. Written as
    ``ladder_traces.png`` and ``.pgf``."""
    ladder = dict(color=LEFT["color"], ls="-", lw=1.0)
    ref = dict(color=RIGHT["color"], ls="--", lw=1.0)
    with house_style():
        fig, ax = plt.subplots(2, 3, figsize=(11.0, 6.0), constrained_layout=True)
        ax = ax.ravel()
        for run, st, label in ((ideal, ref, "ideal"), (j, ladder, "reconnected every 2000 steps")):
            q = run["qoi"]
            E = np.asarray(run["trace"]["E"])
            h = np.asarray(q["helicity"])
            x = np.asarray(q["it"]) / 1000.0
            plot_trace(ax[0], F(run), label=label, **st)
            plot_trace(ax[1], E[0] - E, **st)
            plot_trace(ax[2], np.asarray(run["trace"]["cfl"]), **st)
            ax[3].plot(x, (h - h[0]) / h[0], **st)
            ax[4].plot(x, q["JoverB"], **st)
            ax[5].plot(x, q["beta_vol"], **st)
        for a in ax[:3]:   # block-mean traces are in steps; label them in thousands like the qoi panels
            a.set_yscale("log")
            a.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:g}"))
        for ev in j.get("reconnect", []):
            for k, a in enumerate(ax):
                a.axvline(ev["it"] if k < 3 else ev["it"] / 1000.0, color="0.6", ls=":", lw=0.6)
        ax[0].set_ylabel(r"$\|F\|_M$")
        ax[1].set_ylabel(r"$E_0 - E$")
        ax[2].set_ylabel(r"CFL number taken")
        ax[3].set_ylabel(r"$(H - H_0) / H_0$")
        ax[4].set_ylabel(r"$\|J\| / \|B\|$")
        ax[5].set_ylabel(r"$\beta_{\mathrm{vol}}$")
        for a in ax:
            a.set_xlabel(r"step / $10^3$")
            a.grid(True)
        ax[0].legend(loc="upper right")
        save_figure(fig, os.path.join(figdir, "ladder_traces.png"))
        plt.close(fig)


def reconnect_figure(arms, ideal, figdir):
    """Residual, ||J||/||B|| and helicity against the step for the reconnect
    arms, the reconnections marked, the ideal run of the same rung for reference."""
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    entries = [(ideal, "ideal", "k")] + [(j, j["arm"], None) for j in arms]
    for j, label, color in entries:
        line = plot_trace(ax[0], F(j), label=label, color=color)
        q = j["qoi"]
        ax[1].plot(q["it"], q["JoverB"], "-", color=line.get_color(), label=label)
        h = np.asarray(q["helicity"])
        ax[2].plot(q["it"], h - h[0], "-", color=line.get_color(), label=label)
        for ev in j.get("reconnect", []):
            for a in ax:
                a.axvline(ev["it"], color=line.get_color(), ls=":", lw=0.8)
    ax[0].set_yscale("log")
    ax[0].set_ylabel(r"$\|F\|$ (100-step mean $\pm$ sd)")
    ax[1].set_ylabel(r"$\|J\| / \|B\|$")
    ax[2].set_ylabel(r"$H - H_0$")
    for a in ax:
        a.set_xlabel("step")
        a.grid(alpha=0.3)
    ax[0].legend()
    fig.suptitle("ideal descent, reconnected every K steps: dotted lines mark the reconnections")
    fig.savefig(os.path.join(figdir, "reconnect_traces.png"), dpi=140)
    plt.close(fig)


def pulse_islands(tanh_arms, pulse_arms, ideal, figdir):
    """Island width at 3/5 and 1/2 against the dose, tanh rungs vs pulses."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for a, (m, n) in zip(ax, ((5, 1), (6, 1))):
        target = NFP * n / m
        for arms, lab, mk in (
            (tanh_arms, "tanh schedule", "o-"),
            (pulse_arms, "pulse after 2000 ideal steps", "s--"),
        ):
            pts = []
            for j in arms:
                z = sections(j)
                if z is None:
                    continue
                keep = z["final_keep"] & ~z["final_chaotic"]
                io = np.abs(z["final_iota"][keep])
                w = island_width(z, m, n) if io.min() <= target <= io.max() else 0.0
                cyc = (
                    j["params"].get("eta_schedule") == "pulse"
                    and j["params"]["eta_pulse"][2] > 0
                )
                pts.append((dose(j), w, cyc))
            single = sorted(q for q in pts if not q[2])
            if single:
                a.semilogx(
                    [q[0] for q in single], [q[1] for q in single], mk, label=lab
                )
            for q in pts:
                if q[2]:
                    a.semilogx(
                        [q[0]],
                        [q[1]],
                        "D",
                        color="C1",
                        label="pulse every 1000 steps (3 pulses)",
                    )
        if ideal is not None and sections(ideal) is not None:
            a.axhline(
                island_width(sections(ideal), m, n),
                color="k",
                lw=0.6,
                ls=":",
                label=r"$\eta = 0$",
            )
        a.set_xlabel(r"dose $\int \eta\, dt$")
        a.set_ylabel(r"final island width in $\rho$")
        a.grid(alpha=0.3, which="both")
        a.legend(fontsize=8)
        g = math.gcd(NFP * n, m)
        a.set_title(rf"({m}, {n}) chain at $\iota = {NFP * n // g}/{m // g}$")
    fig.suptitle(
        r"li383 (ns = 49), (16,32,32) p=2 $\gamma=1$: island width against the resistive dose"
    )
    fig.savefig(os.path.join(figdir, "pulse_islands.png"), dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs")
    cli = ap.parse_args()
    root = cli.root
    figdir = os.path.join(root, "li383_pub", "figures")
    os.makedirs(figdir, exist_ok=True)

    pub = {
        a: load(root, "li383_pub", a)
        for a in [
            "r12_p1_g0",
            "r12_p2_g0",
            "r12_p3_g0",
            "r12_p4_g0",
            "r12_p3_g0_f64",
            "r12_p3_g1",
            "r16_p3_g1",
            "hi_r12_p3_g0",
            "hi_r12x24_p3_g0",
            "s61_e1e-3_g0",
            "s61_e3e-3_g0",
            "s61_e1e-2_g0",
            "s61_e3e-3_g1",
            "s51_e3e-3_g0",
            "s51_e3e-3_g1",
            "r16_s61_e3e-3_g1",
            "hi_r12x24_p3_g0_f4",
            "s61_e3e-3_g0_f4",
            "s61_e3e-3_g1_f4",
        ]
    }
    fix = {a: load(root, "li383_axisfix", a) for a in ["r16_p3_g0", "r24_p3_g0"]}
    hsweep = [j for j in (load(root, "li383_pub", a) for a in HSWEEP) if j is not None]
    ideal16 = load(root, "li383_pub", "h16_p2_g1")   # the reference rung of every comparison
    if hsweep:
        hsweep_figure(hsweep, figdir)
    psweep = [j for j in (load(root, "li383_pub", a) for a in PSWEEP) if j is not None]
    if len(psweep) > 1:
        psweep_figure(psweep, figdir)
    eta_plain = [
        j for j in (load(root, "li383_eta", f"eta{e}") for e in ETAS) if j is not None
    ]
    eta_seeded = [
        j
        for j in (load(root, "li383_eta", f"s61_eta{e}") for e in ETAS)
        if j is not None
    ]
    if eta_plain or eta_seeded:
        etadir = os.path.join(root, "li383_eta", "figures")
        os.makedirs(etadir, exist_ok=True)
        eta_figure(eta_plain, eta_seeded, ideal16, etadir)
        eta_all = [
            j
            for j in (
                load(root, "li383_eta", os.path.basename(d))
                for d in sorted(glob.glob(os.path.join(root, "li383_eta", "*eta1e-*")))
            )
            if j is not None and len(j["trace"]["F"]) > 1
        ]
        eta_islands(
            [ideal16] + eta_plain,
            [j for j in [load(root, "li383_eta", "s61_eta0")] if j is not None]
            + eta_seeded,
            etadir,
        )
        eta_traces(
            eta_all,
            ideal16,
            load(root, "li383_pub", "r16_s61_e3e-3_g1"),
            etadir,
        )
    pulse = [
        j
        for j in (
            load(root, "li383_pulse", os.path.basename(d))
            for d in sorted(glob.glob(os.path.join(root, "li383_pulse", "pulse*")))
        )
        if j is not None and len(j["trace"]["F"]) > 1
    ]
    if pulse:
        pdir = os.path.join(root, "li383_pulse", "figures")
        os.makedirs(pdir, exist_ok=True)
        same_dose = [
            j
            for j in (
                load(root, "li383_eta", f"eta{e}") for e in ("1e-8", "3e-8", "1e-7")
            )
            if j is not None
        ]
        eta_traces(
            same_dose + pulse,
            ideal16,
            None,
            pdir,
            title="resistive pulses after 2000 ideal steps against the tanh rungs of equal dose",
        )
        os.replace(
            os.path.join(pdir, "eta_traces.png"), os.path.join(pdir, "pulse_traces.png")
        )
        pulse_islands(
            [
                j
                for j in (load(root, "li383_eta", f"eta{e}") for e in ETAS)
                if j is not None
            ],
            pulse,
            ideal16,
            pdir,
        )
        reconnect = [
            j
            for j in (
                load(root, "li383_pulse", os.path.basename(d))
                for d in sorted(glob.glob(os.path.join(root, "li383_pulse", "reconnect*")))
            )
            if j is not None and j.get("reconnect")
        ]
        full = [j for j in reconnect if "smoke" not in j["arm"]]
        if full:
            reconnect_figure(full, ideal16, pdir)
        for j in full:
            if "ladder" in j["arm"]:
                ladder_figure(j, ideal16, pdir)
        open(os.path.join(root, "li383_pulse", "tables.md"), "w").write(
            "## pulse rows\n" + "\n".join(eta_rows(pulse)) + "\n"
            + "".join(
                f"\n## reconnections of {j['arm']}\n" + "\n".join(reconnect_rows(j)) + "\n"
                for j in reconnect
            )
        )
        open(os.path.join(root, "li383_eta", "tables.md"), "w").write(
            "## eta sweep rows\n" + "\n".join(eta_rows(eta_plain + eta_seeded)) + "\n"
        )

    # force residual, three panels
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    plot_lines(ax[0], [(pub[f"r12_p{p}_g0"], f"p={p}", "-") for p in (1, 2, 3, 4)])
    ax[0].set_title(r"(12,24,12) $\gamma=0$: degree scan")
    plot_lines(
        ax[1],
        [
            (pub["r12_p3_g0"], "(12,24,12)", "-"),
            (fix["r16_p3_g0"], "(16,32,16)", "-"),
            (fix["r24_p3_g0"], "(24,48,24)", "-"),
        ],
    )
    ax[1].set_title(r"p=3 $\gamma=0$: resolution scan")
    plot_lines(
        ax[2],
        [
            (pub["r12_p3_g0"], r"(12,24,12) $\gamma=0$", "-"),
            (pub["r12_p3_g1"], r"(12,24,12) $\gamma=1$", "--"),
            (fix["r16_p3_g0"], r"(16,32,16) $\gamma=0$", "-"),
            (pub["r16_p3_g1"], r"(16,32,16) $\gamma=1$", "--"),
        ],
    )
    ax[2].set_title(r"p=3: velocity smoothing $\gamma$")
    fig.suptitle("li383 (NCSX) relaxation: force residual")
    fig.savefig(os.path.join(figdir, "force_convergence.png"), dpi=150)

    # energy released and helicity drift
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for j in [
        pub["r12_p3_g0"],
        fix["r16_p3_g0"],
        fix["r24_p3_g0"],
        pub["r12_p3_g1"],
        pub["r16_p3_g1"],
        pub["r12_p3_g0_f64"],
    ]:
        if j is None:
            continue
        E = np.asarray(j["trace"]["E"])
        plot_trace(ax[0], (E[0] - E) / E[0], lw=1.0, label=j["arm"])
        h = np.asarray(j["qoi"]["helicity"])
        ax[1].plot(
            np.asarray(j["qoi"]["it"]),
            (h - h[0]) / h[0],
            "o-",
            ms=3,
            lw=1.0,
            label=j["arm"],
        )
    ax[0].set_yscale("log")
    ax[0].set_title(r"energy released $(E_0 - E)/E_0$")
    ax[1].set_title(r"helicity drift $(H - H_0)/H_0$")
    for a in ax:
        a.set_xlabel("step")
        a.set_xscale("log")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.suptitle(r"li383 relaxation, $\eta = 0$")
    fig.savefig(os.path.join(figdir, "energy_helicity.png"), dpi=150)

    # the reference sets the floor
    fig, ax = plt.subplots(figsize=(6.5, 4.6), constrained_layout=True)
    plot_lines(
        ax,
        [
            (pub["r12_p3_g0"], "reference ns = 16", "-"),
            (pub["hi_r12_p3_g0"], "reference ns = 49", "-"),
        ],
    )
    ax.set_title(r"(12,24,12) p=3 $\gamma=0$: the VMEC reference sets the floor")
    fig.savefig(os.path.join(figdir, "reference_floor.png"), dpi=150)

    # seeded arms: residual traces and island width vs eps
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    plot_lines(
        ax[0],
        [(pub["hi_r12x24_p3_g0"], "no seed", "-")]
        + [
            (pub[a], a, "--" if a.endswith("g1") else "-")
            for a in [
                "s61_e1e-3_g0",
                "s61_e3e-3_g0",
                "s61_e1e-2_g0",
                "s61_e3e-3_g1",
                "s51_e3e-3_g0",
                "s51_e3e-3_g1",
                "r16_s61_e3e-3_g1",
                "hi_r12x24_p3_g0_f4",
                "s61_e3e-3_g0_f4",
                "s61_e3e-3_g1_f4",
            ]
        ],
    )
    ax[0].set_title("seeded arms: force residual")
    eps, w_fin, w_ic = [], [], []
    for a in ["s61_e1e-3_g0", "s61_e3e-3_g0", "s61_e1e-2_g0"]:
        j = pub[a]
        z = sections(j) if j is not None else None
        if z is None:
            continue
        eps.append(j["params"]["seed_eps"])
        w_fin.append(island_width(z, 6, 1))
        w_ic.append(island_width_ic(z, 6, 1))
    if eps:
        e = np.asarray(eps)
        ax[1].loglog(e, w_fin, "s-", label="final")
        ax[1].loglog(e, w_ic, "o--", label="initial condition")
        iota_p = 0.36
        ax[1].loglog(
            e,
            1.6 * np.sqrt(e * NFP / (6 * iota_p)),
            "k:",
            label=r"seed: $1.6\sqrt{\epsilon\, n_{fp} / (m\,\iota')}$",
        )
        ax[1].set_xlabel(r"$\epsilon$")
        ax[1].set_ylabel(r"island width in $\rho$")
        ax[1].set_title(r"(6, 1) chain at $\iota = 1/2$, final, $\gamma = 0$")
        ax[1].grid(alpha=0.3, which="both")
        ax[1].legend(fontsize=8)
    fig.savefig(os.path.join(figdir, "seeded.png"), dpi=150)

    reader = [
        pub[a]
        for a in [
            "r12_p1_g0",
            "r12_p2_g0",
            "r12_p3_g0",
            "r12_p4_g0",
            "r12_p3_g0_f64",
            "r12_p3_g1",
            "r16_p3_g1",
            "hi_r12_p3_g0",
            "hi_r12x24_p3_g0",
        ]
        if pub[a] is not None
    ]
    seeded = [
        pub[a]
        for a in [
            "s61_e1e-3_g0",
            "s61_e3e-3_g0",
            "s61_e1e-2_g0",
            "s61_e3e-3_g1",
            "s51_e3e-3_g0",
            "s51_e3e-3_g1",
            "r16_s61_e3e-3_g1",
            "hi_r12x24_p3_g0_f4",
            "s61_e3e-3_g0_f4",
            "s61_e3e-3_g1_f4",
        ]
        if pub[a] is not None
    ]
    lines = [
        "## section 4 rows",
        *reader_rows(reader),
        "",
        "## section 5 rows",
        *seeded_rows(seeded),
        "",
        "## section 5c rows",
        *hsweep_rows(hsweep),
        "",
        "## section 5d rows",
        *hsweep_rows(psweep, key="p"),
        "",
    ]
    for j in [a for a in seeded if a["params"]["seed"]]:
        z = sections(j)
        if z is None:
            continue
        m, n = (int(float(v)) for v in j["params"]["seed"].split(",")[:2])
        zi = sections(j)
        lines.append(
            f"{j['arm']}: final width {island_width(zi, m, n):.4f}; "
            f"ic width {island_width_ic(zi, m, n):.4f}; chaotic final {int(z['final_chaotic'].sum())}, "
            f"ic {int(z['ic_chaotic'].sum())}"
        )
    open(os.path.join(root, "li383_pub", "tables.md"), "w").write(
        "\n".join(lines) + "\n"
    )
    print("\n".join(lines))


def island_width_ic(z, m, n, tol=2e-3):
    return island_width(z, m, n, tol, tag="ic")


if __name__ == "__main__":
    main()
