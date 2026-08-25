"""Overlay iota profiles from several runs and quantify how far apart they are.

Two comparisons, one script, because they are the same operation:

* **resolution** -- the same geometry at several ``ns``/``p``.  Does iota settle?
* **perturbation** -- ``quasr44970-c`` against ``pert-axis`` and
  ``pert-interior``.  Both perturbations move the *interior* of the map and
  leave the boundary fixed (the files' own
  ``perturb_boundary_max_abs_dR_m = 0``), so the domain, and therefore the
  harmonic field, is unchanged.  Any iota difference is numerics, and the
  displaced magnetic axis is the thing being stressed.

Profiles are compared against ``a_eff = sqrt(A/pi)``, the enclosed section area
as a length, **not** against the seed radius.  ``r`` is a label in the logical
chart, so it names a different physical surface the moment the map changes --
which is precisely what both comparisons do.  Comparing on ``r`` would report
the relabelling as a physics difference.

    python scripts/debug/poincare_converge.py --label ns8=.../trace_w7x_k2.npz \\
        --label ns12=.../trace_w7x_k2.npz --title w7x --out conv_w7x.png
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def load(path, plane=0.0):
    d = np.load(path)
    keep = ~(d["escaped"] | ~d["ok"])
    key = f"a_eff_zeta{plane:g}"
    if key not in d.files:
        raise KeyError(f"{path} has no {key}; it predates the a_eff archive")
    return {"a": d[key][keep], "iota": d["iota"][keep],
            "resid": d["resid"][keep], "n": int(keep.sum()),
            "lost": int((~keep).sum())}


def interp_diff(ref, other):
    """Max and RMS |d iota| on the a_eff range the two runs share."""
    lo = max(ref["a"].min(), other["a"].min())
    hi = min(ref["a"].max(), other["a"].max())
    if not (hi > lo):
        return np.nan, np.nan, 0.0
    grid = np.linspace(lo, hi, 200)
    # a_eff is monotone in the seed index, but sort anyway -- an island-crossed
    # surface can put one point out of order and np.interp would silently
    # return garbage for it.
    def on(run):
        o = np.argsort(run["a"])
        return np.interp(grid, run["a"][o], run["iota"][o])
    d = np.abs(on(ref) - on(other))
    return float(d.max()), float(np.sqrt(np.mean(d ** 2))), hi - lo


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", action="append", required=True,
                    metavar="NAME=PATH", help="repeatable; first is the reference")
    ap.add_argument("--plane", type=float, default=0.0)
    ap.add_argument("--title", default="")
    ap.add_argument("--out", required=True)
    cli = ap.parse_args()

    runs = []
    for spec in cli.label:
        name, _, path = spec.partition("=")
        if not path:
            raise ValueError(f"--label wants NAME=PATH, got {spec!r}")
        runs.append((name, load(path, cli.plane)))

    ref_name, ref = runs[0]
    print(f"{'run':>14} {'lines':>6} {'lost':>5} {'iota(axis)':>11} "
          f"{'iota(edge)':>11} {'max|d|':>10} {'rms|d|':>10}")
    rows = []
    for name, run in runs:
        o = np.argsort(run["a"])
        axis_iota, edge_iota = run["iota"][o][0], run["iota"][o][-1]
        if name == ref_name:
            dmax = drms = 0.0
        else:
            dmax, drms, _ = interp_diff(ref, run)
        print(f"{name:>14} {run['n']:>6} {run['lost']:>5} {axis_iota:>11.6f} "
              f"{edge_iota:>11.6f} {dmax:>10.3e} {drms:>10.3e}")
        rows.append((name, run, dmax, drms))

    fig = plt.figure(figsize=(11.5, 4.6), constrained_layout=True)
    ax, bx = fig.subplots(1, 2)
    for name, run, _, _ in rows:
        o = np.argsort(run["a"])
        ax.plot(run["a"][o], run["iota"][o], "o-", ms=3, lw=1.0, label=name)
    ax.set_xlabel(r"$a_{\mathrm{eff}} = \sqrt{A/\pi}$  [m]")
    ax.set_ylabel(r"$\iota$")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(f"{cli.title}  |  $\\zeta = {cli.plane:g}$", fontsize=10)

    for name, run, _, _ in rows[1:]:
        lo = max(ref["a"].min(), run["a"].min())
        hi = min(ref["a"].max(), run["a"].max())
        grid = np.linspace(lo, hi, 200)

        def on(r):
            o = np.argsort(r["a"])
            return np.interp(grid, r["a"][o], r["iota"][o])
        bx.semilogy(grid, np.maximum(np.abs(on(ref) - on(run)), 1e-16),
                    lw=1.0, label=f"{name} - {ref_name}")
    bx.set_xlabel(r"$a_{\mathrm{eff}}$  [m]")
    bx.set_ylabel(r"$|\Delta\iota|$")
    bx.grid(alpha=0.3)
    if len(rows) > 1:
        bx.legend(fontsize=8)
    bx.set_title(f"difference against {ref_name}", fontsize=10)

    os.makedirs(os.path.dirname(cli.out) or ".", exist_ok=True)
    fig.savefig(cli.out, dpi=200)
    plt.close(fig)
    print(cli.out)


if __name__ == "__main__":
    main()
