#!/usr/bin/env python
"""Visualize the dense W7-X FEEC system matrices (mass + stiffness).

One plot per picture. For each matrix two standalone PNGs are written:
  * ``<name>_fill.png``      : binary fill pattern, black where |entry| > tol
                               (tol = 1e-12 * max|entry|), white elsewhere.
  * ``<name>_magnitude.png`` : |entry| on a LINEAR greyscale (cmap gray_r, so
                               DARKER = LARGER magnitude), colorbar "|entry|".

Violet block/surgery boundary lines are overlaid on both:
  * Surgery / polar-axis core split (contiguous at the FRONT of the DOF vector):
    3*nzeta for k=0/k=3 forms, 5*nzeta for k=1/k=2 forms.
  * 1-/2-form vector-component block boundaries (measured from the saved
    matrices via the nnz/row structural jump).

Usage:
    .venv/bin/python scripts/plotting/w7x_matrix_fill.py [--bc dbc] [names...]
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGDIR = "outputs/w7x_matrices/figures"
NZETA = 12  # ns[2]

# surgery: index where the contiguous polar-axis core ends.
# components: internal vector-component block boundaries (None for scalar forms).
SPEC = {
    "M0": dict(surgery=3 * NZETA, components=None),
    "K0": dict(surgery=3 * NZETA, components=None),
    "M1": dict(surgery=5 * NZETA, components=[2940, 5532]),
    "K1": dict(surgery=5 * NZETA, components=[2940, 5532]),
    "L1": dict(surgery=5 * NZETA, components=[2940, 5532]),
    # k=2 (2-form): surgery polar-core = first 2*nz; 3 vector components at
    # 2616 / 5496 (from diagonal-magnitude jumps, consistent M2/K2).
    "M2": dict(surgery=2 * NZETA, components=[2616, 5496]),
    "K2": dict(surgery=2 * NZETA, components=[2616, 5496]),
    "L2": dict(surgery=2 * NZETA, components=[2616, 5496]),
    # k=3 (3-form): scalar, no vector components and no surgery core -> no lines.
    "M3": dict(surgery=None, components=None),
    "L3": dict(surgery=None, components=None),
    "K3": dict(surgery=None, components=None),
}

VIOLET = "#8F00FF"
LW = 0.6


def _autospec(name, A):
    """Spec lookup; unknown matrices get no boundary lines."""
    return SPEC.get(name, dict(surgery=None, components=None))


def overlay_boundaries(ax, n, surgery, components):
    if surgery is not None:
        ax.axvline(surgery - 0.5, color=VIOLET, lw=LW)
        ax.axhline(surgery - 0.5, color=VIOLET, lw=LW)
    if components:
        for b in components:
            ax.axvline(b - 0.5, color=VIOLET, lw=LW)
            ax.axhline(b - 0.5, color=VIOLET, lw=LW)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)


def plot_fill(name, A, spec):
    absA = np.abs(A)
    fill = (absA > 1e-12 * absA.max()).astype(np.float64)
    n = A.shape[0]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(fill, cmap="gray_r", interpolation="nearest", aspect="equal")
    overlay_boundaries(ax, n, spec["surgery"], spec["components"])
    fig.tight_layout()
    out = os.path.join(FIGDIR, f"{name}_fill.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {os.path.abspath(out)}")


def plot_magnitude(name, A, spec):
    absA = np.abs(A)
    n = A.shape[0]
    nz = absA[absA > 0]
    floor = np.log10(nz.min()) if nz.size else 0.0
    logmag = np.where(absA > 0, np.log10(np.where(absA > 0, absA, 1.0)), floor)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(logmag, cmap="gray_r", interpolation="nearest", aspect="equal",
                   vmin=floor, vmax=np.log10(absA.max()))
    overlay_boundaries(ax, n, spec["surgery"], spec["components"])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("log10 |entry|")
    fig.tight_layout()
    out = os.path.join(FIGDIR, f"{name}_magnitude.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {os.path.abspath(out)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bc", default="dbc")
    ap.add_argument("names", nargs="*", default=["M0", "M1", "K0", "K1"])
    args = ap.parse_args()
    matdir = f"outputs/w7x_matrices/{args.bc}"
    os.makedirs(FIGDIR, exist_ok=True)

    for name in args.names:
        path = os.path.join(matdir, f"{name}.npy")
        if not os.path.exists(path):
            print(f"skip {name}: {path} not found")
            continue
        A = np.load(path)
        spec = _autospec(name, A)
        plot_fill(name, A, spec)
        plot_magnitude(name, A, spec)
        del A


if __name__ == "__main__":
    main()
