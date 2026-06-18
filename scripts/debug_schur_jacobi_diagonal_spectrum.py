"""Dense Schur-Jacobi diagonal variant spectrum diagnostics.

Compares left-preconditioned spectra of

    A = S1 + D0 M0^{-1} D0.T

for three diagonal Jacobi choices J^{-1} = diag(A_approx)^{-1}:

1) cheap-diagonal:
       diag(S1) + diag(D0 diag(M0)^{-1} D0.T)
2) mass-precond-diagonal:
       diag(S1) + diag(D0 M0_precond^{-1} D0.T)
3) exact-mass-diagonal:
       diag(S1) + diag(D0 M0^{-1} D0.T)

then reports spectral summaries for J^{-1} A.

Defaults target a deliberately small case:
- n = (6, 8, 4)
- p = (3, 3, 3)
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import toroid_map
from mrx.operators import (
    assemble_derivative_operators,
    assemble_laplacian_operators,
    assemble_mass_operators,
    assemble_tensor_mass_preconditioner,
    dense_derivative_matrix,
    dense_mass_matrix,
    dense_stiffness_matrix,
)

jax.config.update("jax_enable_x64", True)


def _dense_from_apply(n_in: int, apply_fn) -> np.ndarray:
    eye = np.eye(n_in, dtype=np.float64)
    first = np.asarray(apply_fn(jnp.asarray(eye[:, 0])))
    n_out = int(first.shape[0])
    out = np.empty((n_out, n_in), dtype=np.float64)
    out[:, 0] = first
    for j in range(1, n_in):
        out[:, j] = np.asarray(apply_fn(jnp.asarray(eye[:, j])))
    return out


def _symmetric_pinv(A: np.ndarray, rtol: float) -> np.ndarray:
    A = 0.5 * (A + A.T)
    w, V = np.linalg.eigh(A)
    scale = max(float(np.max(np.abs(w))), 1.0)
    cut = rtol * scale
    inv_w = np.where(np.abs(w) > cut, 1.0 / w, 0.0)
    return (V * inv_w[None, :]) @ V.T


def _apply_to_columns(mat: np.ndarray, apply_fn) -> np.ndarray:
    out = np.empty_like(mat)
    for j in range(mat.shape[1]):
        out[:, j] = np.asarray(apply_fn(jnp.asarray(mat[:, j])))
    return out


def _solve_mass_columns(M: np.ndarray, rhs_cols: np.ndarray, *, pinv_rtol: float) -> np.ndarray:
    """Solve M X = rhs_cols for multiple RHS with a dense direct solve fallback."""
    try:
        return np.linalg.solve(M, rhs_cols)
    except np.linalg.LinAlgError:
        # Defensive fallback for unexpected singular/ill-conditioned cases.
        Minv = _symmetric_pinv(M, rtol=pinv_rtol)
        return Minv @ rhs_cols


def _diag_dmmdt(D: np.ndarray, Minv: np.ndarray) -> np.ndarray:
    # diag(D Minv D.T)_i = row_i(D) Minv row_i(D)^T
    return np.einsum("ij,jk,ik->i", D, Minv, D)


def _partial_spectrum(M: np.ndarray, k: int) -> np.ndarray:
    try:
        from scipy.sparse.linalg import eigs
    except Exception:
        return np.array([], dtype=np.complex128)

    n = M.shape[0]
    k_eff = max(2, min(k, n - 2))
    vals = []
    for which in ("LM", "SM"):
        try:
            w = eigs(M, k=k_eff, which=which, return_eigenvectors=False, tol=1e-9)
            vals.append(np.asarray(w))
        except Exception:
            pass
    if not vals:
        return np.array([], dtype=np.complex128)
    return np.concatenate(vals)


def _spectrum_report(label: str, M: np.ndarray, *, k_eigs: int) -> None:
    w = _partial_spectrum(M, k=k_eigs)
    if w.size == 0:
        if M.shape[0] <= 900:
            w = np.linalg.eigvals(M)
        else:
            print(label)
            print("  eigen summary unavailable (no scipy eigs and matrix too large)")
            return

    wr = w.real
    wi = np.abs(w.imag)
    print(label)
    print(f"  eig count: {w.shape[0]}")
    print(f"  min Re(lambda): {np.min(wr):.6e}")
    print(f"  max Re(lambda): {np.max(wr):.6e}")
    print(f"  max |Im(lambda)|: {np.max(wi):.6e}")

    pos = wr[wr > 1e-12]
    if pos.size > 0:
        print(f"  cond surrogate (Re+): {float(np.max(pos) / np.min(pos)):.6e}")
    else:
        print("  cond surrogate (Re+): unavailable (no positive real eigs)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, nargs=3, default=(6, 8, 4))
    parser.add_argument("--p", type=int, default=3)
    parser.add_argument("--q", type=int, default=4)
    parser.add_argument("--epsilon", type=float, default=1.0 / 3.0)
    parser.add_argument("--dirichlet", choices=("dbc", "free"), default="dbc")
    parser.add_argument("--mass-rank", type=int, default=3)
    parser.add_argument("--pinv-rtol", type=float, default=1e-11)
    parser.add_argument("--k-eigs", type=int, default=24)
    args = parser.parse_args()

    dirichlet = args.dirichlet == "dbc"
    ns = tuple(int(v) for v in args.n)
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
    ops = assemble_derivative_operators(seq, seq.geometry, operators=ops, ks=(0,))
    ops = assemble_laplacian_operators(seq, seq.geometry, operators=ops, ks=(1,))
    ops = assemble_tensor_mass_preconditioner(
        seq,
        operators=ops,
        ks=(0,),
        rank=args.mass_rank,
    )
    seq.set_operators(ops)
    print(f"Setup done in {time.perf_counter() - t0:.2f}s")

    t1 = time.perf_counter()
    D0 = np.asarray(
        dense_derivative_matrix(
            seq,
            ops,
            0,
            dirichlet_in=dirichlet,
            dirichlet_out=dirichlet,
        )
    )
    S1 = np.asarray(dense_stiffness_matrix(seq, ops, 1, dirichlet=dirichlet))
    M0 = np.asarray(dense_mass_matrix(seq, ops, 0, dirichlet=dirichlet))
    print(f"Dense D0/S1/M0 built in {time.perf_counter() - t1:.2f}s")

    t2 = time.perf_counter()
    Dt = D0.T

    # Exact mass inverse action on D0.T columns (shared by exact A and exact diag term).
    X_exact = _solve_mass_columns(M0, Dt, pinv_rtol=args.pinv_rtol)

    # Preconditioned inverse action only where needed (diag term), no dense matrix materialization.
    X_precond = _apply_to_columns(
        Dt,
        lambda x: seq.apply_mass_matrix_preconditioner(
            x,
            0,
            dirichlet=dirichlet,
            kind="tensor",
        ),
    )
    print(f"Inverse actions on D0.T columns built in {time.perf_counter() - t2:.2f}s")

    # Exact Schur operator for spectrum target.
    A = S1 + D0 @ X_exact

    diag_S1 = np.diag(S1)
    inv_diag_M0 = np.where(np.abs(np.diag(M0)) > 0.0, 1.0 / np.diag(M0), 0.0)
    diag_term_cheap = np.sum((D0 * inv_diag_M0[None, :]) * D0, axis=1)
    diag_term_precond = np.sum(D0 * X_precond.T, axis=1)
    diag_term_exact = np.sum(D0 * X_exact.T, axis=1)

    def _safe_inv_diag(d: np.ndarray) -> np.ndarray:
        return np.where(np.abs(d) > 0.0, 1.0 / d, 0.0)

    invdiag_cheap = _safe_inv_diag(diag_S1 + diag_term_cheap)
    invdiag_precond = _safe_inv_diag(diag_S1 + diag_term_precond)
    invdiag_exact = _safe_inv_diag(diag_S1 + diag_term_exact)

    PA_cheap = invdiag_cheap[:, None] * A
    PA_precond = invdiag_precond[:, None] * A
    PA_exactdiag = invdiag_exact[:, None] * A

    print("\nJacobi diagonal variant spectra for J^{-1} A:")
    _spectrum_report(
        "cheap diagonal: diag(S1) + diag(D0 diag(M0)^{-1} D0.T)",
        PA_cheap,
        k_eigs=args.k_eigs,
    )
    _spectrum_report(
        "diag with M0_precond^{-1}: diag(S1) + diag(D0 M0_precond^{-1} D0.T)",
        PA_precond,
        k_eigs=args.k_eigs,
    )
    _spectrum_report(
        "diag with exact M0^{-1}: diag(S1) + diag(D0 M0^{-1} D0.T)",
        PA_exactdiag,
        k_eigs=args.k_eigs,
    )


if __name__ == "__main__":
    main()
