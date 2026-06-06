"""Parse submitit log files from a matrix-free matvec benchmark and plot results.

Companion to ``scripts/plot_poisson_sweep.py``; same parsing/plotting style but
for the ``benchmark_matvec_sparse.py`` output (stored-BCSR vs matrix-free mass
matvec timings).

Usage:
    python scripts/plot_matvec_benchmark.py multirun/2026-06-01/13-42-17
    python scripts/plot_matvec_benchmark.py multirun/2026-06-01/13-42-17 --save
"""
import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

# A completed data row from the benchmark table, e.g.
#   16  3  1      24064     138018816    2.208   1.17e-15     49.967      1.248     40.03
_RE_ROW = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+"      # n p k n_dof nnz
    r"([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+"                  # M[GB] relerr
    r"([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s*$",  # bcsr mf speedup
    re.MULTILINE,
)


def parse_log(path: Path) -> list[dict]:
    """Return list of result dicts, one per completed (n, p, k) row."""
    text = path.read_text()
    rows = []
    for m in _RE_ROW.finditer(text):
        n, p, k, n_dof, nnz = (int(m.group(i)) for i in range(1, 6))
        m_gb, relerr, bcsr_ms, mf_ms, speedup = (
            float(m.group(i)) for i in range(6, 11))
        rows.append(dict(n=n, p=p, k=k, n_dof=n_dof, nnz=nnz, m_gb=m_gb,
                         relerr=relerr, bcsr_ms=bcsr_ms, mf_ms=mf_ms,
                         speedup=speedup))
    return rows


def load_multirun(base: Path) -> list[dict]:
    submitit_dir = base / ".submitit"
    rows = []
    for job_dir in sorted(submitit_dir.glob("*_*")):
        for log in sorted(job_dir.glob("*_0_log.out")):
            rows.extend(parse_log(log))
    # deduplicate (same (n,p,k) should not appear twice)
    seen = set()
    unique = []
    for r in rows:
        key = (r["n"], r["p"], r["k"])
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

MARKERS = {1: "o", 2: "s", 3: "^", 4: "D"}
COLORS = {1: "C0", 2: "C1", 3: "C2", 4: "C3"}
K_LABEL = {0: "M0 (0-form)", 1: "M1 (1-form)"}


def _ref_line(ax, xs, ref_x, ref_val, order, label=None, color="gray", ls="--"):
    xs = np.asarray(xs, float)
    ys = ref_val * (xs / ref_x) ** order
    ax.plot(xs, ys, ls=ls, color=color, linewidth=0.8,
            label=label or f"$O(x^{{{order}}})$")


def _by(rows, **filt):
    out = [r for r in rows
           if all(r[key] == val for key, val in filt.items())]
    return sorted(out, key=lambda r: r.get("n", 0))


def plot_all(rows: list[dict], save: bool, base: Path):
    if not rows:
        print("No completed rows found.", file=sys.stderr)
        return

    ks = sorted({r["k"] for r in rows})
    ps = sorted({r["p"] for r in rows})

    # ---- Figure 1: time vs n (per k), bcsr vs mf ------------------------ #
    fig1, axes1 = plt.subplots(1, len(ks), figsize=(6 * len(ks), 4.5),
                               squeeze=False)
    for ax, k in zip(axes1[0], ks):
        for p in ps:
            data = _by(rows, p=p, k=k)
            if not data:
                continue
            ns = [r["n"] for r in data]
            ax.loglog(ns, [r["bcsr_ms"] for r in data],
                      marker=MARKERS.get(p, "o"), color=COLORS.get(p, f"C{p}"),
                      ls="-", label=f"BCSR p={p}")
            ax.loglog(ns, [r["mf_ms"] for r in data],
                      marker=MARKERS.get(p, "o"), color=COLORS.get(p, f"C{p}"),
                      ls="--", mfc="none", label=f"matrix-free p={p}")
        # reference slopes anchored to the highest-p BCSR curve
        ref = _by(rows, p=max(ps), k=k)
        if len(ref) >= 2:
            _ref_line(ax, [r["n"] for r in ref], ref[-1]["n"],
                      ref[-1]["bcsr_ms"], 3, label=r"$O(n^3)$", color="gray")
        ax.set_xlabel("n (grid points per dim)")
        ax.set_ylabel("matvec time (ms)")
        ax.set_title(K_LABEL.get(k, f"k={k}"))
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.legend(fontsize=7, ncol=2)
    fig1.suptitle("Matvec time vs n  (solid = BCSR, dashed = matrix-free)")
    fig1.tight_layout()

    # ---- Figure 2: time vs p (per k), bcsr vs mf ------------------------ #
    ns_all = sorted({r["n"] for r in rows})
    fig2, axes2 = plt.subplots(1, len(ks), figsize=(6 * len(ks), 4.5),
                               squeeze=False)
    for ax, k in zip(axes2[0], ks):
        for n in ns_all:
            data = sorted([r for r in rows if r["n"] == n and r["k"] == k],
                          key=lambda r: r["p"])
            if len(data) < 2:
                continue
            psv = [r["p"] for r in data]
            color = f"C{ns_all.index(n)}"
            ax.loglog(psv, [r["bcsr_ms"] for r in data], marker="o",
                      color=color, ls="-", label=f"BCSR n={n}")
            ax.loglog(psv, [r["mf_ms"] for r in data], marker="o",
                      color=color, ls="--", mfc="none",
                      label=f"matrix-free n={n}")
        ax.set_xlabel("p (polynomial degree)")
        ax.set_ylabel("matvec time (ms)")
        ax.set_title(K_LABEL.get(k, f"k={k}"))
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.legend(fontsize=7, ncol=2)
    fig2.suptitle("Matvec time vs p  (solid = BCSR, dashed = matrix-free)")
    fig2.tight_layout()

    # ---- Figure 3: speedup (BCSR / matrix-free) ------------------------- #
    fig3, axes3 = plt.subplots(1, len(ks), figsize=(6 * len(ks), 4.5),
                               squeeze=False)
    for ax, k in zip(axes3[0], ks):
        for p in ps:
            data = _by(rows, p=p, k=k)
            if not data:
                continue
            ns = [r["n"] for r in data]
            ax.semilogx(ns, [r["speedup"] for r in data],
                        marker=MARKERS.get(p, "o"),
                        color=COLORS.get(p, f"C{p}"), label=f"p={p}")
        ax.axhline(1.0, color="gray", ls="--", lw=0.8, label="parity")
        ax.set_yscale("log")
        ax.set_xlabel("n (grid points per dim)")
        ax.set_ylabel("speedup  (BCSR time / matrix-free time)")
        ax.set_title(K_LABEL.get(k, f"k={k}"))
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.legend(fontsize=8)
    fig3.suptitle("Matrix-free speedup over stored BCSR")
    fig3.tight_layout()

    # ---- output ---------------------------------------------------------- #
    if save:
        out_dir = base / "plots"
        out_dir.mkdir(exist_ok=True)
        fig1.savefig(out_dir / "matvec_time_vs_n.pdf")
        fig2.savefig(out_dir / "matvec_time_vs_p.pdf")
        fig3.savefig(out_dir / "matvec_speedup.pdf")
        print(f"Saved to {out_dir}/")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("multirun_dir", type=Path)
    parser.add_argument("--save", action="store_true",
                        help="Save PDFs to <multirun_dir>/plots/ instead of showing")
    args = parser.parse_args()

    rows = load_multirun(args.multirun_dir)
    print(f"Found {len(rows)} completed (n, p, k) rows:")
    for r in sorted(rows, key=lambda r: (r["k"], r["p"], r["n"])):
        print(f"  k={r['k']} p={r['p']} n={r['n']:3d}  "
              f"bcsr={r['bcsr_ms']:9.3f}ms  mf={r['mf_ms']:7.3f}ms  "
              f"speedup={r['speedup']:7.2f}  relerr={r['relerr']:.1e}")

    plot_all(rows, save=args.save, base=args.multirun_dir)


if __name__ == "__main__":
    main()
