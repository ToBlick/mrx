"""Diagnose the k=3 Hodge MINRES stall at N=6.

Assembles the full saddle-point matrix densely on the k=3 / k=2 blocks,
checks symmetry, nullspace alignment, and compares the dense pseudo-inverse
solution against ``seq.apply_inverse_laplacian`` (which uses MINRES).

Run:
    .venv/bin/python scripts/debug_k3_saddle.py
"""

from __future__ import annotations

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

from mrx.nullspace import get_saddle_point_nullspaces
from mrx.operators import (apply_laplacian, apply_mass_matrix,
                           apply_stiffness)

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from mrx.nullspace import get_nullspace  # noqa: E402
from mrx.operators import apply_derivative_matrix  # noqa: E402
from scripts.benchmark_preconditioners import build_sequence  # noqa: E402


def assemble_dense(matvec, n_in, n_out):
    I = jnp.eye(n_in)
    cols = jax.vmap(matvec)(I)  # (n_in, n_out)
    return np.asarray(cols.T)  # (n_out, n_in)


def main():
    print("Building sequence (N=6)...")
    seq, ops = build_sequence()
    k, dbc = 3, True
    n_u = seq.n3_dbc  # k=3 DoFs
    n_s = seq.n2_dbc  # k=2 DoFs (lower block)
    print(f"n_upper (k=3) = {n_u}, n_lower (k=2) = {n_s}, total = {n_u + n_s}")

    # --- Dense assembly ---
    print("\nAssembling dense blocks...")
    S = assemble_dense(
        lambda x: apply_stiffness(seq, ops, x, k, dirichlet=dbc), n_u, n_u)
    # k = 3: S should be zero per apply_stiffness (no k=4 mass).  Verify:
    print(f"  ||S_3||_F = {np.linalg.norm(S):.2e}  "
          f"(expected 0 since apply_stiffness returns 0 for k=3)")

    D = assemble_dense(
        lambda s: apply_derivative_matrix(
            seq, ops, s, k - 1, dirichlet_in=dbc, dirichlet_out=dbc),
        n_s, n_u)
    DT = assemble_dense(
        lambda u: apply_derivative_matrix(
            seq, ops, u, k - 1, dirichlet_in=dbc, dirichlet_out=dbc,
            transpose=True),
        n_u, n_s)
    print(f"  ||D - DT.T||_F = {np.linalg.norm(D - DT.T):.2e}  "
          "(should be ~0)")

    M_s = assemble_dense(
        lambda s: apply_mass_matrix(seq, ops, s, k - 1, dirichlet=dbc),
        n_s, n_s)
    M_u = assemble_dense(
        lambda x: apply_mass_matrix(seq, ops, x, k, dirichlet=dbc),
        n_u, n_u)
    print(f"  M_2 sym err = {np.linalg.norm(M_s - M_s.T):.2e}")
    print(f"  M_3 sym err = {np.linalg.norm(M_u - M_u.T):.2e}")

    # --- Hodge Laplacian on k=3 ---
    # L_3 = S_3 + D M_2^{-1} D^T = D M_2^{-1} D^T
    M_s_inv = np.linalg.inv(M_s)
    L3 = S + D @ M_s_inv @ DT
    L3 = 0.5 * (L3 + L3.T)
    w = np.linalg.eigvalsh(L3)
    print(f"\nL_3 spectrum: min={w.min():.3e}  max={w.max():.3e}  "
          f"#(|eig|<1e-10)={int((np.abs(w) < 1e-10).sum())}")

    # --- Nullspace alignment ---
    vs_up = np.asarray(get_nullspace(ops, k, dbc))
    vs_up_s, vs_low_s = get_saddle_point_nullspaces(seq, ops, k, dbc)
    vs_up_s = np.asarray(vs_up_s)
    vs_low_s = np.asarray(vs_low_s)
    print(f"\nvs_upper stored: shape={vs_up.shape}")
    print(f"vs_upper saddle: shape={vs_up_s.shape}")
    print(f"vs_lower saddle: shape={vs_low_s.shape}")
    for i, v in enumerate(vs_up):
        Lv = L3 @ v
        print(f"  ||L_3 v_up[{i}]||/||v|| = "
              f"{np.linalg.norm(Lv)/np.linalg.norm(v):.3e}")
    # Saddle-point kernel check: A @ [v, σ] = 0 where σ = M_2^{-1} D^T v
    if vs_up_s.shape[0] > 0:
        A = np.block([[S, D], [DT, -M_s]])
        for i in range(vs_up_s.shape[0]):
            z = np.concatenate([vs_up_s[i], vs_low_s[i]])
            Az = A @ z
            print(f"  ||A @ [v_up[{i}], v_low[{i}]]||/||z|| = "
                  f"{np.linalg.norm(Az)/np.linalg.norm(z):.3e}")

    # --- Compare dense vs MINRES on random RHS ---
    rng = np.random.default_rng(0)
    b = rng.standard_normal(n_u)
    # Deflate b of the M_3-adjoint nullspace: project onto range(L_3)
    for v in vs_up:
        b = b - (v @ b) * (M_u @ v)  # b -= (v,b) M_3 v
    # Dense reference: pseudo-inverse of L_3
    u_dense, *_ = np.linalg.lstsq(L3, b, rcond=None)
    # MINRES
    u_mr, info = seq.apply_inverse_laplacian(
        jnp.asarray(b), k, dirichlet=dbc, preconditioner='tensor',
        return_info=True)
    u_mr = np.asarray(u_mr)

    # Remove nullspace component from both
    def deflate(x):
        for v in vs_up:
            x = x - (v @ x) * v
        return x
    du = deflate(u_dense - u_mr)
    r_dense = L3 @ u_dense - b
    r_mr = L3 @ u_mr - b
    print(f"\nMINRES info={int(info)}")
    print(f"  ||L u_dense - b|| / ||b|| = "
          f"{np.linalg.norm(r_dense)/np.linalg.norm(b):.2e}")
    print(f"  ||L u_mr    - b|| / ||b|| = "
          f"{np.linalg.norm(r_mr)/np.linalg.norm(b):.2e}")
    print(f"  ||deflate(u_dense - u_mr)|| / ||u_dense|| = "
          f"{np.linalg.norm(du)/max(np.linalg.norm(u_dense),1e-30):.2e}")

    # Dense saddle-point solve for comparison
    A = np.block([[S, D], [DT, -M_s]])
    rhs = np.concatenate([b, np.zeros(n_s)])
    z, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    u_sp, s_sp = z[:n_u], z[n_u:]
    r_sp = L3 @ u_sp - b
    print(f"  dense saddle ||L u_sp - b|| / ||b|| = "
          f"{np.linalg.norm(r_sp)/np.linalg.norm(b):.2e}")


if __name__ == "__main__":
    main()
    main()
