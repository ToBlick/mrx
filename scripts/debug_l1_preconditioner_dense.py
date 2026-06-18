"""Dense diagnostics for a 1-form Laplacian inverse preconditioner on a torus.

Constructs an approximate inverse for the 1-form Hodge Laplacian,

    B = S1_inv
        + M1_inv M21.T M2_inv S2_inv M2_inv M21 M1_inv
        + M1_inv D0 S0_inv D0.T M1_inv

on extracted spaces, then evaluates dense diagnostics for

    BL = B @ L1,

including ||BL-I||, row mismatch statistics, and a partial spectrum estimate.

Notes
-----
- Uses existing MRX applies for all inverse-like pieces (mass/stiffness/hodge
  preconditioner applies), then densifies by probing basis vectors.
- Default is DBC to avoid harmonic nullspace complications in dense diagnostics.

Usage
-----
/scratch/tblickhan/mrx/.venv/bin/python scripts/debug_l1_preconditioner_dense.py

Optional
--------
/scratch/tblickhan/mrx/.venv/bin/python scripts/debug_l1_preconditioner_dense.py \
    --dirichlet free --n 8 --p 1 --q 2 --epsilon 0.3333333333333333 \
    --mass-rank 3 --stiff-rank 3 --k-eigs 24 --outdir outputs/interactive
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
    apply_stiffness_tensor_preconditioner,
    assemble_incidence_operators,
    assemble_mass_operators,
    assemble_projection_operators,
    assemble_tensor_laplacian_preconditioner,
    assemble_tensor_mass_preconditioner,
    assemble_tensor_stiffness_preconditioner,
    dense_derivative_matrix,
    dense_laplacian,
    dense_projection_matrix,
)

jax.config.update("jax_enable_x64", True)


def _dense_from_apply(n_in: int, apply_fn) -> np.ndarray:
    """Materialize a dense matrix by probing apply_fn on canonical basis vectors."""
    eye = np.eye(n_in, dtype=np.float64)
    first = np.asarray(apply_fn(jnp.asarray(eye[:, 0])))
    n_out = int(first.shape[0])
    out = np.empty((n_out, n_in), dtype=np.float64)
    out[:, 0] = first
    for j in range(1, n_in):
        out[:, j] = np.asarray(apply_fn(jnp.asarray(eye[:, j])))
    return out


def _top_rows(A: np.ndarray, B: np.ndarray, topk: int = 12) -> list[tuple[int, float]]:
    diff = A - B
    row_rel = np.linalg.norm(diff, axis=1) / np.maximum(np.linalg.norm(B, axis=1), 1e-30)
    idx = np.argsort(row_rel)[::-1][:topk]
    return [(int(i), float(row_rel[i])) for i in idx]


def _partial_spectrum(BL: np.ndarray, k: int) -> np.ndarray:
    """Return a partial complex spectrum estimate using ARPACK if available."""
    try:
        from scipy.sparse.linalg import eigs
    except Exception:
        return np.array([], dtype=np.complex128)

    k_eff = max(2, min(k, BL.shape[0] - 2))
    vals = []
    for which in ("LM", "SM"):
        try:
            vv = eigs(BL, k=k_eff, which=which, return_eigenvectors=False, tol=1e-8)
            vals.append(np.asarray(vv))
        except Exception:
            pass
    if not vals:
        return np.array([], dtype=np.complex128)
    return np.concatenate(vals)


def _save_imshow(mat: np.ndarray, title: str, path: str, *, tiny_abs: float = 0.0) -> None:
    plot_mat = np.where(np.abs(mat) < tiny_abs, 0.0, mat)
    vmax = max(np.max(np.abs(plot_mat)), 1e-30)
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    im = ax.imshow(plot_mat, cmap="coolwarm", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--q", type=int, default=2)
    parser.add_argument("--epsilon", type=float, default=1.0 / 3.0)
    parser.add_argument("--dirichlet", choices=("dbc", "free"), default="dbc")
    parser.add_argument("--mass-rank", type=int, default=3)
    parser.add_argument("--stiff-rank", type=int, default=3)
    parser.add_argument("--k-eigs", type=int, default=24)
    parser.add_argument("--outdir", default="outputs/interactive")
    parser.add_argument("--plot-zero-threshold", type=float, default=1e-14)
    args = parser.parse_args()

    dirichlet = args.dirichlet == "dbc"
    ns = (args.n, 2 * args.n, args.n)
    ps = (args.p, args.p, args.p)
    types = ("clamped", "periodic", "periodic")

    print("JAX devices:", jax.devices())
    print(
        f"Building sequence ns={ns} ps={ps} q={args.q} eps={args.epsilon} dirichlet={dirichlet}"
    )
    t0 = time.perf_counter()

    seq = DeRhamSequence(ns, ps, args.q, types, polar=True)
    seq.set_map(toroid_map(epsilon=args.epsilon))
    seq.evaluate_1d()

    ops = seq.get_operators()
    ops = assemble_mass_operators(seq, seq.geometry, operators=ops, ks=(0, 1, 2))
    ops = assemble_incidence_operators(seq, operators=ops, ks=(0, 1))
    ops = assemble_projection_operators(seq, operators=ops, pairs=((2, 1),))
    ops = assemble_tensor_mass_preconditioner(
        seq,
        operators=ops,
        ks=(0, 1, 2),
        rank=args.mass_rank,
    )
    ops = assemble_tensor_stiffness_preconditioner(
        seq,
        operators=ops,
        ks=(1, 2),
        rank=args.stiff_rank,
    )
    ops = assemble_tensor_laplacian_preconditioner(seq, operators=ops, ks=(0,))
    seq.set_operators(ops)

    print(f"Setup done in {time.perf_counter() - t0:.2f}s")

    n0 = seq.n0_dbc if dirichlet else seq.n0
    n1 = seq.n1_dbc if dirichlet else seq.n1
    n2 = seq.n2_dbc if dirichlet else seq.n2
    print(f"DoFs: n0={n0} n1={n1} n2={n2}")

    # Dense primal->dual 1-form Laplacian.
    t1 = time.perf_counter()
    L1 = np.asarray(dense_laplacian(seq, ops, 1, dirichlet=dirichlet))
    D0 = np.asarray(
        dense_derivative_matrix(
            seq,
            ops,
            0,
            dirichlet_in=dirichlet,
            dirichlet_out=dirichlet,
        )
    )
    M21 = np.asarray(
        dense_projection_matrix(
            seq,
            ops,
            2,
            1,
            dirichlet_in=dirichlet,
            dirichlet_out=dirichlet,
        )
    )
    print(f"Dense L1/D0/M21 done in {time.perf_counter() - t1:.2f}s")

    # Dense inverse-like building blocks by probing existing applies.
    t2 = time.perf_counter()
    M1_inv = _dense_from_apply(
        n1,
        lambda x: seq.apply_mass_matrix_preconditioner(
            x, 1, dirichlet=dirichlet, kind="tensor"
        ),
    )
    M2_inv = _dense_from_apply(
        n2,
        lambda x: seq.apply_mass_matrix_preconditioner(
            x, 2, dirichlet=dirichlet, kind="tensor"
        ),
    )
    S1_inv = _dense_from_apply(
        n1,
        lambda x: apply_stiffness_tensor_preconditioner(
            seq, ops, x, 1, dirichlet=dirichlet
        ),
    )
    S2_inv = _dense_from_apply(
        n2,
        lambda x: apply_stiffness_tensor_preconditioner(
            seq, ops, x, 2, dirichlet=dirichlet
        ),
    )
    S0_inv = _dense_from_apply(
        n0,
        lambda x: seq.apply_laplacian_preconditioner(
            x, 0, dirichlet=dirichlet, kind="tensor"
        ),
    )
    print(f"Dense inverse-block probes done in {time.perf_counter() - t2:.2f}s")

    # Proposed preconditioner B : V1* -> V1.
    t3 = time.perf_counter()
    term1 = S1_inv
    term2 = M1_inv @ M21.T @ M2_inv @ S2_inv @ M2_inv @ M21 @ M1_inv
    term3 = M1_inv @ D0 @ S0_inv @ D0.T @ M1_inv
    B = term1 + term2 + term3
    BL = B @ L1
    I = np.eye(n1)
    print(f"Dense composition done in {time.perf_counter() - t3:.2f}s")

    rel_BL_I = np.linalg.norm(BL - I) / np.linalg.norm(I)
    max_BL_I = np.max(np.abs(BL - I))
    print(f"rel||BL-I||/||I|| = {rel_BL_I:.6e}")
    print(f"max|BL-I|         = {max_BL_I:.6e}")

    top_rows = _top_rows(BL, I, topk=12)
    print("Top row mismatches for BL vs I (idx: row_rel):")
    for i, v in top_rows:
        print(f"  {i}: {v:.6e}")

    vals = _partial_spectrum(BL, args.k_eigs)
    if vals.size > 0:
        print(f"Partial spectrum size: {vals.size}")
        print(f"  min Re(lambda): {np.min(vals.real):.6e}")
        print(f"  max Re(lambda): {np.max(vals.real):.6e}")
        print(f"  max |Im(lambda)|: {np.max(np.abs(vals.imag)):.6e}")
    else:
        print("Partial spectrum unavailable (scipy/arpack not available).")

    os.makedirs(args.outdir, exist_ok=True)
    tag = f"n{args.n}_p{args.p}_{args.dirichlet}"
    _save_imshow(
        BL,
        f"B @ L1 ({tag})",
        os.path.join(args.outdir, f"l1_precond_BL_{tag}.png"),
        tiny_abs=args.plot_zero_threshold,
    )
    _save_imshow(
        BL - I,
        f"B @ L1 - I ({tag})",
        os.path.join(args.outdir, f"l1_precond_BL_minus_I_{tag}.png"),
        tiny_abs=args.plot_zero_threshold,
    )

    if vals.size > 0:
        fig, ax = plt.subplots(figsize=(6.5, 6.0), constrained_layout=True)
        ax.scatter(vals.real, vals.imag, s=12, alpha=0.8)
        ax.axvline(1.0, color="k", ls="--", lw=1)
        ax.axhline(0.0, color="k", ls="--", lw=1)
        ax.set_title(f"Partial spectrum of B @ L1 ({tag})")
        ax.set_xlabel("Re(lambda)")
        ax.set_ylabel("Im(lambda)")
        ax.grid(alpha=0.3)
        out_spec = os.path.join(args.outdir, f"l1_precond_spectrum_{tag}.png")
        fig.savefig(out_spec, dpi=180)
        plt.close(fig)
        print(f"Wrote {out_spec}")


if __name__ == "__main__":
    main()
