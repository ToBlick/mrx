"""Dense diagnostics for k=1 Schur-preconditioned spectral equivalence.

We inspect the bulk block of

    C = D0 M0^{-1} D0.T

for extracted k=1 spaces on the toroidal geometry, and compare it against
incidence/stiffness-based surrogates:

    C_hat_full  = G0_b S0_inv G0_b.T
    C_hat_bulk  = G0_bb S0_inv_bb G0_bb.T

where:
- b denotes k=1 bulk rows (from the k=1 stiffness surgery split),
- bb on G0 denotes additionally restricting k=0 columns to k=0 bulk,
- S0_inv is the assembled k=0 tensor Hodge preconditioner apply, densified.

Raw operator equality between coupling terms is not the criterion here.
Instead we form a block preconditioner via the surgery/bulk Schur split and
assess spectral quality of the preconditioned operator.

Given the exact block matrix

    A = [[A_ss, A_sb],
         [A_bs, A_bb]],

and an approximate bulk inverse B_bb (dense diagnostic), define

    S_hat = A_ss - A_sb B_bb A_bs,

then apply

    P(rhs):
      y = B_bb rhs_b
      z = S_hat^{-1} (rhs_s - A_sb y)
      x_b = y - B_bb A_bs z

This script reports spectra/condition surrogates of P @ A for:
- exact bulk inverse (reference),
- additive inverse model B_bb = S_bb^{-1} + G0_b S0_inv G0_b.T,
- additive inverse model B_bb = S_bb^{-1} + G0_bb S0_inv_bb G0_bb.T.

It also reports weak-form mass-wrapped variants:
- B_bb = S_bb^{-1} + M1_bb^{-1} D0_b S0_inv D0_b.T M1_bb^{-1},
- B_bb = S_bb^{-1} + M1_bb^{-1} D0_bb S0_inv_bb D0_bb.T M1_bb^{-1}.
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
    apply_stiffness_tensor_preconditioner,
    assemble_laplacian_operators,
    assemble_incidence_operators,
    assemble_mass_operators,
    assemble_tensor_mass_preconditioner,
    assemble_tensor_laplacian_preconditioner,
    assemble_tensor_stiffness_preconditioner,
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


def _dense_incidence(seq: DeRhamSequence, *, dirichlet: bool) -> np.ndarray:
    n0 = seq.n0_dbc if dirichlet else seq.n0
    return _dense_from_apply(
        n0,
        lambda x: seq.apply_incidence_matrix(
            x,
            0,
            dirichlet_in=dirichlet,
            dirichlet_out=dirichlet,
        ),
    )


def _pinv_sym(A: np.ndarray, rtol: float) -> np.ndarray:
    A = 0.5 * (A + A.T)
    w, V = np.linalg.eigh(A)
    scale = max(float(np.max(np.abs(w))), 1.0)
    cut = rtol * scale
    w_inv = np.where(np.abs(w) > cut, 1.0 / w, 0.0)
    return (V * w_inv[None, :]) @ V.T


def _report(label: str, A: np.ndarray, B: np.ndarray) -> None:
    d = A - B
    rel = np.linalg.norm(d) / max(np.linalg.norm(A), 1e-30)
    rel_b = np.linalg.norm(d) / max(np.linalg.norm(B), 1e-30)
    print(f"{label}:")
    print(f"  rel wrt exact: ||A-B||/||A|| = {rel:.6e}")
    print(f"  rel wrt approx: ||A-B||/||B|| = {rel_b:.6e}")
    print(f"  max |A-B| = {np.max(np.abs(d)):.6e}")


def _dense_restricted_k1_stiff_precond(seq, ops, bulk_idx: np.ndarray, *, dirichlet: bool) -> np.ndarray:
    """Dense restricted map x_b -> (P_stiff(full(x_b)))_b."""
    n1 = seq.n1_dbc if dirichlet else seq.n1
    nb = bulk_idx.shape[0]

    def apply_bulk(xb: jnp.ndarray) -> jnp.ndarray:
        full = jnp.zeros((n1,), dtype=xb.dtype)
        full = full.at[bulk_idx].set(xb)
        yfull = apply_stiffness_tensor_preconditioner(
            seq,
            ops,
            full,
            1,
            dirichlet=dirichlet,
        )
        return yfull[bulk_idx]

    return _dense_from_apply(nb, apply_bulk)


def _dense_restricted_k1_mass_precond(seq, bulk_idx: np.ndarray, *, dirichlet: bool) -> np.ndarray:
    """Dense restricted map x_b -> (P_mass(full(x_b)))_b."""
    n1 = seq.n1_dbc if dirichlet else seq.n1
    nb = bulk_idx.shape[0]

    def apply_bulk(xb: jnp.ndarray) -> jnp.ndarray:
        full = jnp.zeros((n1,), dtype=xb.dtype)
        full = full.at[bulk_idx].set(xb)
        yfull = seq.apply_mass_matrix_preconditioner(
            full,
            1,
            dirichlet=dirichlet,
            kind="tensor",
        )
        return yfull[bulk_idx]

    return _dense_from_apply(nb, apply_bulk)


def _split_blocks(A: np.ndarray, surgery_idx: np.ndarray, bulk_idx: np.ndarray):
    Ass = A[np.ix_(surgery_idx, surgery_idx)]
    Asb = A[np.ix_(surgery_idx, bulk_idx)]
    Abs = A[np.ix_(bulk_idx, surgery_idx)]
    Abb = A[np.ix_(bulk_idx, bulk_idx)]
    return Ass, Asb, Abs, Abb


def _build_block_preconditioner(
    Ass: np.ndarray,
    Asb: np.ndarray,
    Abs: np.ndarray,
    Bbb: np.ndarray,
    *,
    schur_rtol: float,
):
    Shat = 0.5 * ((Ass - Asb @ Bbb @ Abs) + (Ass - Asb @ Bbb @ Abs).T)
    Shat_inv = _pinv_sym(Shat, rtol=schur_rtol)

    n_s = Ass.shape[0]
    n_b = Bbb.shape[0]

    def apply(rhs: np.ndarray) -> np.ndarray:
        rhs_s = rhs[:n_s]
        rhs_b = rhs[n_s : n_s + n_b]
        y = Bbb @ rhs_b
        z = Shat_inv @ (rhs_s - Asb @ y)
        x_b = y - Bbb @ (Abs @ z)
        return np.concatenate([z, x_b])

    return apply


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
            w = eigs(M, k=k_eff, which=which, return_eigenvectors=False, tol=1e-8)
            vals.append(np.asarray(w))
        except Exception:
            pass
    if not vals:
        return np.array([], dtype=np.complex128)
    return np.concatenate(vals)


def _spectrum_summary(label: str, M: np.ndarray, *, rtol_zero: float = 1e-11, k_eigs: int = 24) -> None:
    w = _partial_spectrum(M, k=k_eigs)
    if w.size == 0:
        if M.shape[0] <= 900:
            w = np.linalg.eigvals(M)
        else:
            print(label)
            print("  eigen summary unavailable: scipy eigs missing/failed and matrix is large")
            return
    wr = w.real
    wi = np.abs(w.imag)
    print(label)
    print(f"  eig count: {w.shape[0]}")
    print(f"  min Re(lambda): {np.min(wr):.6e}")
    print(f"  max Re(lambda): {np.max(wr):.6e}")
    print(f"  max |Im(lambda)|: {np.max(wi):.6e}")

    pos = wr[wr > rtol_zero]
    if pos.size > 0:
        kappa = float(np.max(pos) / np.min(pos))
        print(f"  cond surrogate (Re+): {kappa:.6e}")
    else:
        print("  cond surrogate (Re+): unavailable (no positive eigenvalues)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--q", type=int, default=2)
    parser.add_argument("--epsilon", type=float, default=1.0 / 3.0)
    parser.add_argument("--dirichlet", choices=("dbc", "free"), default="dbc")
    parser.add_argument("--stiff-rank", type=int, default=3)
    parser.add_argument("--m0-pinv-rtol", type=float, default=1e-10)
    parser.add_argument("--schur-pinv-rtol", type=float, default=1e-10)
    parser.add_argument("--k-eigs", type=int, default=24)
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
    ops = assemble_mass_operators(seq, seq.geometry, operators=ops, ks=(0, 1))
    ops = assemble_laplacian_operators(seq, seq.geometry, operators=ops, ks=(0, 1))
    ops = assemble_incidence_operators(seq, operators=ops, ks=(0,))
    ops = assemble_tensor_mass_preconditioner(
        seq,
        operators=ops,
        ks=(1,),
        rank=args.stiff_rank,
    )
    ops = assemble_tensor_stiffness_preconditioner(
        seq,
        operators=ops,
        ks=(1,),
        rank=args.stiff_rank,
    )
    ops = assemble_tensor_laplacian_preconditioner(seq, operators=ops, ks=(0,))
    seq.set_operators(ops)

    print(f"Setup done in {time.perf_counter() - t0:.2f}s")

    n0 = seq.n0_dbc if dirichlet else seq.n0
    n1 = seq.n1_dbc if dirichlet else seq.n1

    # Pull k=1 bulk indices from the assembled stiffness surgery split.
    k1_pair = ops.k1_tensor_stiff_precond
    if k1_pair is None:
        raise RuntimeError("k1 tensor stiffness preconditioner is missing")
    k1_payload = k1_pair.dbc if dirichlet else k1_pair.free
    if k1_payload is None:
        raise RuntimeError("k1 tensor stiffness payload missing for requested BC")
    bulk_idx_1 = np.asarray(k1_payload.surgery.bulk_indices)
    surg_idx_1 = np.asarray(k1_payload.surgery.surgery_indices)

    # Pull k=0 bulk indices from the assembled k0 tensor Hodge split.
    k0_pair = ops.k0_tensor_hodge_precond
    if k0_pair is None:
        raise RuntimeError("k0 tensor hodge preconditioner is missing")
    k0_factors = k0_pair.dbc if dirichlet else k0_pair.free
    if k0_factors is None:
        raise RuntimeError("k0 tensor hodge payload missing for requested BC")
    core_size = int(k0_factors.core_size)
    bulk_idx_0 = np.arange(core_size, n0, dtype=np.int32)

    print(f"DoFs: n0={n0} n1={n1}")
    print(f"k1 bulk size: {bulk_idx_1.shape[0]} (surgery={surg_idx_1.shape[0]})")
    print(f"k0 bulk size: {bulk_idx_0.shape[0]} (core={core_size})")

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
    G0 = _dense_incidence(seq, dirichlet=dirichlet)
    M0 = np.asarray(dense_mass_matrix(seq, ops, 0, dirichlet=dirichlet))
    S1 = np.asarray(dense_stiffness_matrix(seq, ops, 1, dirichlet=dirichlet))
    print(f"Dense operators D0/G0/M0/S1 built in {time.perf_counter() - t1:.2f}s")

    t2 = time.perf_counter()
    S0_inv = _dense_from_apply(
        n0,
        lambda x: seq.apply_laplacian_preconditioner(
            x,
            0,
            dirichlet=dirichlet,
            kind="tensor",
        ),
    )
    print(f"Dense S0_inv probe done in {time.perf_counter() - t2:.2f}s")

    t3 = time.perf_counter()
    M0_inv = _pinv_sym(M0, rtol=args.m0_pinv_rtol)

    D0_b = D0[bulk_idx_1, :]
    C_bb = D0_b @ M0_inv @ D0_b.T

    G0_b = G0[bulk_idx_1, :]
    C_hat_full = G0_b @ S0_inv @ G0_b.T

    G0_bb = G0[np.ix_(bulk_idx_1, bulk_idx_0)]
    S0_inv_bb = S0_inv[np.ix_(bulk_idx_0, bulk_idx_0)]
    C_hat_bulk = G0_bb @ S0_inv_bb @ G0_bb.T

    S_bb = S1[np.ix_(bulk_idx_1, bulk_idx_1)]
    A_bb = S_bb + C_bb
    A_hat_full = S_bb + C_hat_full
    A_hat_bulk = S_bb + C_hat_bulk

    print(f"Dense compositions done in {time.perf_counter() - t3:.2f}s")

    print("\nCoupling term comparisons: C_bb = (D0 M0^{-1} D0.T)_bb")
    _report("G0_b S0_inv G0_b.T", C_bb, C_hat_full)
    _report("G0_bb S0_inv_bb G0_bb.T", C_bb, C_hat_bulk)

    print("\nFull bulk operator comparisons: A_bb = S_bb + C_bb")
    _report("S_bb + G0_b S0_inv G0_b.T", A_bb, A_hat_full)
    _report("S_bb + G0_bb S0_inv_bb G0_bb.T", A_bb, A_hat_bulk)

    # Schur-preconditioned spectral diagnostics on the full k1 operator
    A_full = S1 + (D0 @ M0_inv @ D0.T)
    Ass, Asb, Abs, Abb = _split_blocks(A_full, surg_idx_1, bulk_idx_1)

    Sbb_inv_tensor = _dense_restricted_k1_stiff_precond(
        seq,
        ops,
        bulk_idx_1,
        dirichlet=dirichlet,
    )
    M1bb_inv_tensor = _dense_restricted_k1_mass_precond(
        seq,
        bulk_idx_1,
        dirichlet=dirichlet,
    )

    Bbb_exact = _pinv_sym(Abb, rtol=args.m0_pinv_rtol)
    Bbb_hat_full = Sbb_inv_tensor + C_hat_full
    Bbb_hat_bulk = Sbb_inv_tensor + C_hat_bulk
    D0_bb = D0[np.ix_(bulk_idx_1, bulk_idx_0)]
    Bbb_weak_full = Sbb_inv_tensor + M1bb_inv_tensor @ D0_b @ S0_inv @ D0_b.T @ M1bb_inv_tensor
    Bbb_weak_bulk = Sbb_inv_tensor + M1bb_inv_tensor @ D0_bb @ S0_inv_bb @ D0_bb.T @ M1bb_inv_tensor

    P_exact = _build_block_preconditioner(
        Ass,
        Asb,
        Abs,
        Bbb_exact,
        schur_rtol=args.schur_pinv_rtol,
    )
    P_hat_full = _build_block_preconditioner(
        Ass,
        Asb,
        Abs,
        Bbb_hat_full,
        schur_rtol=args.schur_pinv_rtol,
    )
    P_hat_bulk = _build_block_preconditioner(
        Ass,
        Asb,
        Abs,
        Bbb_hat_bulk,
        schur_rtol=args.schur_pinv_rtol,
    )
    P_weak_full = _build_block_preconditioner(
        Ass,
        Asb,
        Abs,
        Bbb_weak_full,
        schur_rtol=args.schur_pinv_rtol,
    )
    P_weak_bulk = _build_block_preconditioner(
        Ass,
        Asb,
        Abs,
        Bbb_weak_bulk,
        schur_rtol=args.schur_pinv_rtol,
    )

    # Form preconditioned operators P A in block ordering [surgery, bulk].
    A_block = A_full[np.ix_(np.concatenate([surg_idx_1, bulk_idx_1]), np.concatenate([surg_idx_1, bulk_idx_1]))]
    n_block = A_block.shape[0]
    Iblk = np.eye(n_block, dtype=np.float64)
    P_exact_dense = np.column_stack([P_exact(Iblk[:, j]) for j in range(n_block)])
    P_hat_full_dense = np.column_stack([P_hat_full(Iblk[:, j]) for j in range(n_block)])
    P_hat_bulk_dense = np.column_stack([P_hat_bulk(Iblk[:, j]) for j in range(n_block)])
    P_weak_full_dense = np.column_stack([P_weak_full(Iblk[:, j]) for j in range(n_block)])
    P_weak_bulk_dense = np.column_stack([P_weak_bulk(Iblk[:, j]) for j in range(n_block)])

    PA_exact = P_exact_dense @ A_block
    PA_hat_full = P_hat_full_dense @ A_block
    PA_hat_bulk = P_hat_bulk_dense @ A_block
    PA_weak_full = P_weak_full_dense @ A_block
    PA_weak_bulk = P_weak_bulk_dense @ A_block

    print("\nSchur-preconditioned spectrum diagnostics (P @ A):")
    _spectrum_summary("reference bulk exact inverse", PA_exact, k_eigs=args.k_eigs)
    _spectrum_summary("B_bb = S_bb^{-1} + G0_b S0_inv G0_b.T", PA_hat_full, k_eigs=args.k_eigs)
    _spectrum_summary("B_bb = S_bb^{-1} + G0_bb S0_inv_bb G0_bb.T", PA_hat_bulk, k_eigs=args.k_eigs)
    _spectrum_summary("B_bb = S_bb^{-1} + M1_bb^{-1} D0_b S0_inv D0_b.T M1_bb^{-1}", PA_weak_full, k_eigs=args.k_eigs)
    _spectrum_summary("B_bb = S_bb^{-1} + M1_bb^{-1} D0_bb S0_inv_bb D0_bb.T M1_bb^{-1}", PA_weak_bulk, k_eigs=args.k_eigs)


if __name__ == "__main__":
    main()
