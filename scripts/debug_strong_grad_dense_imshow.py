"""Dense comparison of extracted strong gradient vs extracted incidence on a torus.

Builds a toroidal DeRham sequence, forms dense operators for

    S = (E1 M1 E1^T)^{-1} D0

and

    G = E1 G0 E0^T

then saves imshow plots for S, G, and S-G. This directly visualizes whether the
mismatch is localized ("surgery block") or distributed.

Usage:
    /scratch/tblickhan/mrx/.venv/bin/python scripts/debug_strong_grad_dense_imshow.py

Optional:
    /scratch/tblickhan/mrx/.venv/bin/python scripts/debug_strong_grad_dense_imshow.py \
        --dirichlet dbc --epsilon 0.3333333333333333 --outdir outputs/interactive
"""

from __future__ import annotations

import argparse
import os
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import toroid_map
from mrx.operators import (
    assemble_incidence_operators,
    assemble_mass_operators,
    assemble_tensor_mass_preconditioner,
    dense_derivative_matrix,
    dense_mass_matrix,
)

jax.config.update("jax_enable_x64", True)


def _dense_extracted_incidence(seq: DeRhamSequence, *, dirichlet: bool) -> np.ndarray:
    """Return dense extracted G0 by probing apply_incidence_matrix on basis vectors."""
    n0 = seq.n0_dbc if dirichlet else seq.n0
    eye = jnp.eye(n0, dtype=jnp.float64)
    cols = jax.vmap(
        lambda e: seq.apply_incidence_matrix(
            e, 0, dirichlet_in=dirichlet, dirichlet_out=dirichlet
        )
    )(eye)
    return np.asarray(cols.T)


def _dense_preconditioned_strong_grad(seq: DeRhamSequence, *, dirichlet: bool) -> np.ndarray:
    """Return dense S_pc = P1 D0 with P1 a k=1 mass preconditioner apply."""
    n0 = seq.n0_dbc if dirichlet else seq.n0
    eye = jnp.eye(n0, dtype=jnp.float64)

    def _apply_col(e):
        rhs = seq.apply_derivative_matrix(
            e, 0, dirichlet_in=dirichlet, dirichlet_out=dirichlet
        )
        return seq.apply_mass_matrix_preconditioner(
            rhs, 1, dirichlet=dirichlet, kind="tensor"
        )

    cols = jax.vmap(_apply_col)(eye)
    return np.asarray(cols.T)


def _print_row_diagnostics(A: np.ndarray, B: np.ndarray, *, tag: str, topk: int = 12) -> None:
    """Print row-wise mismatch diagnostics for A-B."""
    diff = A - B
    rel = np.linalg.norm(diff) / max(np.linalg.norm(B), 1e-30)
    row_rel = np.linalg.norm(diff, axis=1) / np.maximum(np.linalg.norm(B, axis=1), 1e-30)
    top = np.argsort(row_rel)[::-1][:topk]
    frac_large = float(np.mean(row_rel > 1e-2))

    print(f"[{tag}] rel||A-B||/||B|| = {rel:.6e}")
    print(f"[{tag}] max|A-B|        = {np.max(np.abs(diff)):.6e}")
    print(f"[{tag}] rows with row_rel>1e-2: {frac_large:.2%}")
    print(f"[{tag}] top row mismatches (idx: row_rel):")
    for i in top:
        print(f"    {int(i)}: {row_rel[i]:.6e}")


def _plot_precond_case(
    S_exact: np.ndarray,
    S_pc: np.ndarray,
    label: str,
    outdir: str,
    tiny_abs: float,
) -> str:
    """Plot exact strong-grad vs preconditioned strong-grad and their mismatch."""
    diff = S_pc - S_exact
    diff_plot = np.where(np.abs(diff) < tiny_abs, 0.0, diff)
    rel = np.linalg.norm(diff) / max(np.linalg.norm(S_exact), 1e-30)
    row_rel = np.linalg.norm(diff, axis=1) / np.maximum(np.linalg.norm(S_exact, axis=1), 1e-30)

    vmax_e = np.percentile(np.abs(S_exact), 99.0)
    vmax_p = np.percentile(np.abs(S_pc), 99.0)
    vmax_d = np.max(np.abs(diff_plot))
    vmax_d = max(vmax_d, 1e-30)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    im0 = axes[0, 0].imshow(
        S_exact, aspect="auto", cmap="viridis", vmin=-vmax_e, vmax=vmax_e
    )
    axes[0, 0].set_title(f"S_exact = M1^-1 D0 ({label})")
    axes[0, 0].set_xlabel("k=0 column")
    axes[0, 0].set_ylabel("k=1 row")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(
        S_pc, aspect="auto", cmap="viridis", vmin=-vmax_p, vmax=vmax_p
    )
    axes[0, 1].set_title(f"S_pc = P1 D0 (tensor rank=3) ({label})")
    axes[0, 1].set_xlabel("k=0 column")
    axes[0, 1].set_ylabel("k=1 row")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[1, 0].imshow(
        diff_plot, aspect="auto", cmap="coolwarm", vmin=-vmax_d, vmax=vmax_d
    )
    axes[1, 0].set_title(f"S_pc - S_exact ({label})")
    axes[1, 0].set_xlabel("k=0 column")
    axes[1, 0].set_ylabel("k=1 row")
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046)

    axes[1, 1].plot(row_rel, lw=1.0)
    axes[1, 1].set_title("Row-wise relative discrepancy ||(S_pc-S)_i|| / ||S_i||")
    axes[1, 1].set_xlabel("k=1 row index")
    axes[1, 1].set_ylabel("relative row error")
    axes[1, 1].grid(alpha=0.3)

    fig.suptitle(
        f"Mass preconditioner apply vs exact inverse ({label})\n"
        f"rel||S_pc-S||/||S||={rel:.3e}, max|S_pc-S|={np.max(np.abs(diff)):.3e}",
        fontsize=12,
    )

    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"strong_grad_precond_vs_exact_{label}.png")
    fig.savefig(outpath, dpi=180)
    plt.close(fig)
    return outpath


def _plot_case(S: np.ndarray, G: np.ndarray, label: str, outdir: str, tiny_abs: float) -> str:
    diff = S - G
    diff_plot = np.where(np.abs(diff) < tiny_abs, 0.0, diff)

    rel = np.linalg.norm(diff) / max(np.linalg.norm(G), 1e-30)
    linf = np.max(np.abs(diff))
    row_rel = np.linalg.norm(diff, axis=1) / np.maximum(np.linalg.norm(G, axis=1), 1e-30)

    vmax_s = np.percentile(np.abs(S), 99.0)
    vmax_g = np.percentile(np.abs(G), 99.0)
    vmax_d = np.max(np.abs(diff_plot))
    vmax_d = max(vmax_d, 1e-30)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    im0 = axes[0, 0].imshow(S, aspect="auto", cmap="viridis", vmin=-vmax_s, vmax=vmax_s)
    axes[0, 0].set_title(f"S = M1^-1 D0 ({label})")
    axes[0, 0].set_xlabel("k=0 column")
    axes[0, 0].set_ylabel("k=1 row")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(G, aspect="auto", cmap="viridis", vmin=-vmax_g, vmax=vmax_g)
    axes[0, 1].set_title(f"G0 extracted ({label})")
    axes[0, 1].set_xlabel("k=0 column")
    axes[0, 1].set_ylabel("k=1 row")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[1, 0].imshow(diff_plot, aspect="auto", cmap="coolwarm", vmin=-vmax_d, vmax=vmax_d)
    axes[1, 0].set_title(f"S - G0 ({label})")
    axes[1, 0].set_xlabel("k=0 column")
    axes[1, 0].set_ylabel("k=1 row")
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046)

    axes[1, 1].plot(row_rel, lw=1.0)
    axes[1, 1].set_title("Row-wise relative discrepancy ||(S-G)_i|| / ||G_i||")
    axes[1, 1].set_xlabel("k=1 row index")
    axes[1, 1].set_ylabel("relative row error")
    axes[1, 1].grid(alpha=0.3)

    fig.suptitle(
        f"Strong grad vs incidence ({label})\n"
        f"rel||S-G||/||G||={rel:.3e}, max|S-G|={linf:.3e}",
        fontsize=12,
    )

    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"strong_grad_vs_incidence_{label}.png")
    fig.savefig(outpath, dpi=180)
    plt.close(fig)
    return outpath


def _run_case(
    seq: DeRhamSequence,
    ops,
    *,
    dirichlet: bool,
    outdir: str,
    tiny_abs: float,
) -> None:
    label = "dbc" if dirichlet else "free"

    t0 = time.perf_counter()
    M1 = np.asarray(dense_mass_matrix(seq, ops, 1, dirichlet=dirichlet))
    D0 = np.asarray(
        dense_derivative_matrix(
            seq,
            ops,
            0,
            dirichlet_in=dirichlet,
            dirichlet_out=dirichlet,
        )
    )
    S = np.linalg.solve(M1, D0)
    S_pc = _dense_preconditioned_strong_grad(seq, dirichlet=dirichlet)
    G = _dense_extracted_incidence(seq, dirichlet=dirichlet)
    dt = time.perf_counter() - t0

    rel = np.linalg.norm(S - G) / max(np.linalg.norm(G), 1e-30)
    linf = np.max(np.abs(S - G))
    print(f"[{label}] build+dense time: {dt:.2f}s")
    print(f"[{label}] rel||S-G||/||G|| = {rel:.6e}")
    print(f"[{label}] max|S-G|        = {linf:.6e}")
    _print_row_diagnostics(S, G, tag=f"{label}: exact-vs-incidence")
    _print_row_diagnostics(S_pc, S, tag=f"{label}: precond-vs-exact")

    outpath = _plot_case(S, G, label, outdir, tiny_abs)
    print(f"[{label}] wrote {outpath}")
    outpath_pc = _plot_precond_case(S, S_pc, label, outdir, tiny_abs)
    print(f"[{label}] wrote {outpath_pc}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=8, help="Radial/toroidal resolution n.")
    parser.add_argument("--p", type=int, default=1, help="Spline order.")
    parser.add_argument("--q", type=int, default=2, help="Quadrature order.")
    parser.add_argument("--epsilon", type=float, default=1.0 / 3.0, help="Toroid epsilon.")
    parser.add_argument(
        "--dirichlet",
        choices=("free", "dbc", "both"),
        default="both",
        help="Which extracted space to analyze.",
    )
    parser.add_argument(
        "--outdir",
        default="outputs/interactive",
        help="Directory for output plots.",
    )
    parser.add_argument(
        "--mass-precond-rank",
        type=int,
        default=3,
        help="Tensor mass-preconditioner rank for k=1 preconditioner apply.",
    )
    parser.add_argument(
        "--plot-zero-threshold",
        type=float,
        default=1e-14,
        help="Set |entry| below this threshold to zero in diff plots.",
    )
    args = parser.parse_args()

    ns = (args.n, 2 * args.n, args.n)
    ps = (args.p, args.p, args.p)
    types = ("clamped", "periodic", "periodic")

    print("JAX devices:", jax.devices())
    print(f"Building DeRhamSequence ns={ns} ps={ps} q={args.q} epsilon={args.epsilon}")

    t0 = time.perf_counter()
    seq = DeRhamSequence(ns, ps, args.q, types, polar=True)
    seq.set_map(toroid_map(epsilon=args.epsilon))
    seq.evaluate_1d()
    ops = assemble_incidence_operators(seq, seq.get_operators(), ks=(0,))
    ops = assemble_mass_operators(seq, seq.geometry, operators=ops, ks=(1,))
    ops = assemble_tensor_mass_preconditioner(
        seq,
        operators=ops,
        ks=(1,),
        rank=args.mass_precond_rank,
    )
    seq.set_operators(ops)
    print(f"Setup done in {time.perf_counter() - t0:.2f}s")

    if args.dirichlet in ("free", "both"):
        _run_case(
            seq,
            ops,
            dirichlet=False,
            outdir=args.outdir,
            tiny_abs=args.plot_zero_threshold,
        )
    if args.dirichlet in ("dbc", "both"):
        _run_case(
            seq,
            ops,
            dirichlet=True,
            outdir=args.outdir,
            tiny_abs=args.plot_zero_threshold,
        )


if __name__ == "__main__":
    main()
