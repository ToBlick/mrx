"""Parse submitit log files from a Poisson convergence sweep and plot results.

Usage:
    python scripts/plot_poisson_sweep.py multirun/2026-06-01/11-06-53
    python scripts/plot_poisson_sweep.py multirun/2026-06-01/11-06-53 --save
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

_RE_TIMING = re.compile(r"^\s+([\w_.()]+)[\s.]+([0-9.]+)s", re.MULTILINE)
_RE_ERROR  = re.compile(r"Relative L2 error:\s+([0-9.e+-]+)")
_RE_ITERS  = re.compile(r"CG iters:\s+(\d+)\s+converged:\s+(\w+)")

# Match a full section: === block header === followed by content up to the
# next === or end of file.  The n=X, p=Y line sits between the two === fences,
# and all timings/errors are in the content that follows the second fence.
_RE_SECTION = re.compile(
    r"={10,}\s*\n\s*n=(\d+), p=(\d+)\s*\n={10,}\s*\n(.*?)(?=\n={10,}|\Z)",
    re.DOTALL,
)


def parse_log(path: Path) -> list[dict]:
    """Return list of result dicts, one per completed (n, p) block."""
    text = path.read_text()
    results = []

    for m in _RE_SECTION.finditer(text):
        n, p = int(m.group(1)), int(m.group(2))
        body = m.group(3)

        m_err = _RE_ERROR.search(body)
        if m_err is None:
            continue  # block started but OOM'd before printing the error line

        error = float(m_err.group(1))

        m_iters = _RE_ITERS.search(body)
        iters     = int(m_iters.group(1))        if m_iters else None
        converged = m_iters.group(2) == "True"   if m_iters else None

        timings = {k.rstrip('.'): float(v) for k, v in _RE_TIMING.findall(body)}

        results.append(dict(n=n, p=p, error=error, iters=iters,
                            converged=converged, timings=timings))
    return results


def load_multirun(base: Path) -> list[dict]:
    submitit_dir = base / ".submitit"
    rows = []
    for job_dir in sorted(submitit_dir.glob("*_*")):
        logs = sorted(job_dir.glob("*_0_log.out"))
        for log in logs:
            rows.extend(parse_log(log))
    # deduplicate (same (n,p) should not appear twice, but just in case)
    seen = set()
    unique = []
    for r in rows:
        key = (r["n"], r["p"])
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

TIMING_KEYS = [
    ("assemble_mass_matrix_0_exec",                  "assemble M0"),
    ("build_hodge_preconditioners_0_exec",           "build tensor-Hodge precond"),
    ("inverse_hodge_laplacian_exec",                 "Hodge-Laplacian solve"),
    ("TOTAL",                                        "total"),
]

MARKERS = {1: "o", 2: "s", 3: "^", 4: "D"}
COLORS  = {1: "C0", 2: "C1", 3: "C2", 4: "C3"}


def _ref_line(ax, ns, ref_n, ref_val, order, label=None, color="gray", ls="--"):
    ys = ref_val * (np.array(ns) / ref_n) ** order
    ax.plot(ns, ys, ls=ls, color=color, linewidth=0.8,
            label=label or f"$O(n^{{{order}}})$")


def plot_all(rows: list[dict], save: bool, base: Path):
    if not rows:
        print("No completed runs found.", file=sys.stderr)
        return

    ps = sorted({r["p"] for r in rows})

    # ---- Figure 1: convergence ------------------------------------------ #
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    for p in ps:
        data = sorted([r for r in rows if r["p"] == p], key=lambda r: r["n"])
        if not data:
            continue
        ns = [r["n"] for r in data]
        errs = [r["error"] for r in data]
        ax1.loglog(ns, errs, marker=MARKERS.get(p, "o"),
                   color=COLORS.get(p, f"C{p}"), label=f"p={p}")

    # reference lines from last two p=1 points
    p1 = sorted([r for r in rows if r["p"] == 1], key=lambda r: r["n"])
    if len(p1) >= 2:
        _ref_line(ax1, [r["n"] for r in p1], p1[-1]["n"], p1[-1]["error"], -2,
                  label=r"$O(n^{-2})$", color="gray")

    ax1.set_xlabel("n")
    ax1.set_ylabel("relative L2 error")
    ax1.set_title("Poisson convergence (Hodge-Laplacian, toroid)")
    ax1.legend()
    ax1.grid(True, which="both", ls=":", alpha=0.4)
    fig1.tight_layout()

    # ---- Figure 2: CG iteration counts ---------------------------------- #
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    for p in ps:
        data = sorted([r for r in rows if r["p"] == p and r["iters"] is not None],
                      key=lambda r: r["n"])
        if not data:
            continue
        ns = [r["n"] for r in data]
        iters = [r["iters"] for r in data]
        ax2.semilogx(ns, iters, marker=MARKERS.get(p, "o"),
                     color=COLORS.get(p, f"C{p}"), label=f"p={p}")

    ax2.set_xlabel("n")
    ax2.set_ylabel("CG iterations")
    ax2.set_title("CG iteration count")
    ax2.legend()
    ax2.grid(True, which="both", ls=":", alpha=0.4)
    fig2.tight_layout()

    # ---- Figure 3: key timings ------------------------------------------ #
    n_panels = len(TIMING_KEYS)
    fig3, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4), sharey=False)
    if n_panels == 1:
        axes = [axes]

    for ax, (key, label) in zip(axes, TIMING_KEYS):
        for p in ps:
            data = sorted(
                [r for r in rows if r["p"] == p and key in r["timings"]],
                key=lambda r: r["n"])
            if not data:
                continue
            ns = [r["n"] for r in data]
            ts = [r["timings"][key] for r in data]
            ax.loglog(ns, ts, marker=MARKERS.get(p, "o"),
                      color=COLORS.get(p, f"C{p}"), label=f"p={p}")

        # n^3 reference anchored to the last data point of p=1 (if present)
        all_data = sorted(
            [r for r in rows if key in r["timings"]], key=lambda r: r["n"])
        if all_data:
            ref = all_data[-1]
            _ref_line(ax, [r["n"] for r in all_data],
                      ref["n"], ref["timings"][key], 3,
                      label=r"$O(n^3)$", color="gray")

        ax.set_xlabel("n")
        ax.set_ylabel("time (s)")
        ax.set_title(label)
        ax.legend(fontsize=7)
        ax.grid(True, which="both", ls=":", alpha=0.4)

    fig3.suptitle("Timings")
    fig3.tight_layout()

    # ---- output ---------------------------------------------------------- #
    if save:
        out_dir = base / "plots"
        out_dir.mkdir(exist_ok=True)
        fig1.savefig(out_dir / "convergence.pdf")
        fig2.savefig(out_dir / "cg_iters.pdf")
        fig3.savefig(out_dir / "timings.pdf")
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
    print(f"Found {len(rows)} completed (n, p) blocks:")
    for r in sorted(rows, key=lambda r: (r["p"], r["n"])):
        conv = "✓" if r["converged"] else "✗"
        print(f"  p={r['p']}  n={r['n']:3d}  error={r['error']:.3e}  "
              f"CG={r['iters']:3d} {conv}")

    plot_all(rows, save=args.save, base=args.multirun_dir)


if __name__ == "__main__":
    main()
