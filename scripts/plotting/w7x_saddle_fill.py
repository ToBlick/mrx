#!/usr/bin/env python
"""Visualize the W7-X k-form Hodge-Laplacian SADDLE-POINT systems.

For k = 1, 2, 3 the saddle matrix is

    saddle_k = [[  S_k        D_{k-1} ]
                [  D_{k-1}^T  -M_{k-1}]]

with S_k the k-form stiffness (K1, K2, K3=0), D_{k-1} the weak derivative
V_{k-1}->V_k (D0, D1, D2), and M_{k-1} the (k-1)-form mass (M0, M1, M2). It is
assembled from the saved dense blocks.

One plot per picture: ``saddle<k>_fill.png`` (binary) and
``saddle<k>_magnitude.png`` (log10 |entry|, gray_r). Violet lines mark the main
2x2 saddle partition (prominent) plus the surgery / vector-component sub-blocks
of each form (thin).

Usage:
    .venv/bin/python scripts/plotting/w7x_saddle_fill.py [--bc dbc] [ks...]
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGDIR = "outputs/w7x_matrices/figures"
NZETA = 12
VIOLET = "#8F00FF"
LW = 0.6
LW_MAIN = 1.6

# saddle_k = [[S_k, D_{k-1}], [D_{k-1}^T, -M_{k-1}]]
SADDLES = {1: ("K1", "D0", "M0"), 2: ("K2", "D1", "M1"), 3: ("K3", "D2", "M2")}


def form_spec(form_degree, A):
    """(surgery, components) for a given form degree."""
    surgery = (3 if form_degree in (0, 3) else 5) * NZETA
    if form_degree in (1, 2):
        n = A.shape[0]
        nnz = (np.abs(A) > 1e-12 * np.abs(A).max()).sum(axis=1)
        comps = []
        for center in (n // 3, 2 * n // 3):
            lo, hi = center - n // 12, center + n // 12
            j = np.abs(np.diff(nnz[lo:hi]))
            comps.append(lo + 1 + int(np.argmax(j)) if j.size else center)
        return surgery, comps
    return surgery, None


def overlay(ax, N, nk):
    # ONLY the main 2x2 saddle partition (no surgery / sub-block lines).
    ax.axvline(nk - 0.5, color=VIOLET, lw=LW_MAIN)
    ax.axhline(nk - 0.5, color=VIOLET, lw=LW_MAIN)
    ax.set_xlim(-0.5, N - 0.5)
    ax.set_ylim(N - 0.5, -0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bc", default="dbc")
    ap.add_argument("ks", nargs="*", type=int, default=[1, 2, 3])
    args = ap.parse_args()
    matdir = f"outputs/w7x_matrices/{args.bc}"
    os.makedirs(FIGDIR, exist_ok=True)

    for k in args.ks:
        sname, dname, mname = SADDLES[k]
        S = np.load(os.path.join(matdir, f"{sname}.npy"))
        D = np.load(os.path.join(matdir, f"{dname}.npy"))   # (n_k, n_{k-1})
        M = np.load(os.path.join(matdir, f"{mname}.npy"))
        nk, nkm1 = S.shape[0], M.shape[0]
        saddle = np.block([[S, D], [D.T, -M]])
        N = nk + nkm1
        print(f"saddle{k}: {sname}({nk}) + {mname}({nkm1}) -> {N}x{N}", flush=True)

        absA = np.abs(saddle)
        amax = absA.max()
        # fill
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        ax.imshow((absA > 1e-12 * amax).astype(np.float64), cmap="gray_r",
                  interpolation="nearest", aspect="equal")
        overlay(ax, N, nk)
        fig.tight_layout()
        out = os.path.join(FIGDIR, f"saddle{k}_fill.png")
        fig.savefig(out, dpi=150); plt.close(fig); print(f"wrote {os.path.abspath(out)}")
        # log magnitude
        nz = absA[absA > 0]
        floor = np.log10(nz.min()) if nz.size else 0.0
        logmag = np.where(absA > 0, np.log10(np.where(absA > 0, absA, 1.0)), floor)
        fig, ax = plt.subplots(figsize=(8.3, 7.5))
        im = ax.imshow(logmag, cmap="gray_r", interpolation="nearest", aspect="equal",
                       vmin=floor, vmax=np.log10(amax))
        overlay(ax, N, nk)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("log10 |entry|")
        fig.tight_layout()
        out = os.path.join(FIGDIR, f"saddle{k}_magnitude.png")
        fig.savefig(out, dpi=150); plt.close(fig); print(f"wrote {os.path.abspath(out)}")
        del S, D, M, saddle, absA, logmag


if __name__ == "__main__":
    main()
