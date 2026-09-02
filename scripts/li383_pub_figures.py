"""Figures and table rows for the li383 publication runs (docs/research/li383_sweep_results_2026-09-02.md).

Reads ``relax.json`` (and ``poincare/sections.npz`` where present) of the
arms in ``outputs/li383_pub`` and the two current-reader arms in
``outputs/li383_axisfix``; writes ``outputs/li383_pub/figures/*.png`` and
``outputs/li383_pub/tables.md`` (the markdown rows of sections 4 and 5 of
the note). Arms that have not finished are skipped.

    python scripts/li383_pub_figures.py [--root outputs]
"""

import argparse
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

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


def island_width(z, m, n, tol=2e-3):
    """Two widths (in rho) of the chain at |iota| = nfp n / m from the final
    sections: the extent of the seed radii whose fitted iota sits on the
    rational, and the largest radial excursion (peak to peak of logical r
    over one plane's crossings) of those lines -- a line inside the island
    near its separatrix spans the full island width."""
    target = NFP * n / m
    iota = np.abs(z["final_iota"])
    r = z["final_seed_r"]
    keep = z["final_keep"] & ~z["final_chaotic"]
    on = keep & (np.abs(iota - target) < tol)
    if not on.any():
        return 0.0, 0.0
    dr = np.diff(np.sort(r))[0] if len(r) > 1 else 0.0
    plateau = r[on].max() - r[on].min() + dr
    spans = []
    for key in z.files:
        if key.startswith("final_zeta") and key.endswith("_logr"):
            lr = z[key][on]
            spans.append(np.nanmax(lr, axis=1) - np.nanmin(lr, axis=1))
    return float(plateau), float(np.max(spans)) if spans else 0.0


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
        m, n, rho0, width = (float(v) for v in j["params"]["seed"].split(","))
        z = sections(j)
        w = island_width(z, int(m), int(n)) if z is not None else (None, None)
        wtxt = "--" if w[0] is None else f"{w[0]:.3f} / {w[1]:.3f}"
        out.append(
            "| "
            + " | ".join(
                [
                    j["arm"],
                    f"({int(m)}, {int(n)})",
                    f"{j['params']['seed_eps']:.0e}",
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


def plot_lines(ax, entries):
    for j, lab, st in entries:
        if j is not None:
            ax.plot(it(j), F(j), lw=1.0, ls=st, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel(r"$\|F\|_M$")
    ax.axhline(1e-3, color="k", lw=0.6, ls=":")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


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
        ax[0].plot(it(j), (E[0] - E) / E[0], lw=1.0, label=j["arm"])
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
    eps, w_pl, w_sp = [], [], []
    for a in ["s61_e1e-3_g0", "s61_e3e-3_g0", "s61_e1e-2_g0"]:
        j = pub[a]
        z = sections(j) if j is not None else None
        if z is None:
            continue
        pl, sp = island_width(z, 6, 1)
        eps.append(j["params"]["seed_eps"])
        w_pl.append(pl)
        w_sp.append(sp)
    if eps:
        e = np.asarray(eps)
        ax[1].loglog(e, w_pl, "o-", label="iota plateau extent")
        ax[1].loglog(e, w_sp, "s-", label="max radial excursion")
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
    ]
    for j in seeded:
        z = sections(j)
        if z is None:
            continue
        m, n = (int(float(v)) for v in j["params"]["seed"].split(",")[:2])
        zi = sections(j)
        lines.append(
            f"{j['arm']}: final width (plateau / excursion) {island_width(zi, m, n)}; "
            f"ic width {island_width_ic(zi, m, n)}; chaotic final {int(z['final_chaotic'].sum())}, "
            f"ic {int(z['ic_chaotic'].sum())}"
        )
    open(os.path.join(root, "li383_pub", "tables.md"), "w").write(
        "\n".join(lines) + "\n"
    )
    print("\n".join(lines))


def island_width_ic(z, m, n, tol=2e-3):
    zz = {k.replace("ic_", "final_", 1): z[k] for k in z.files if k.startswith("ic_")}

    class _Z:
        files = list(zz)

        def __getitem__(self, k):
            return zz[k]

    return island_width(_Z(), m, n, tol)


if __name__ == "__main__":
    main()
