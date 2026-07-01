"""Complete the W7-X dense-matrix set: M2, M3, K2 (and L2, L3); K3 is the zero
matrix (top-form stiffness d_3=0). Saves .npy and reports kappa, mirroring
w7x_dense_matrices.py (M0,M1,K0,K1).

Best preconditioner:
  M2, M3 : greville tensor mass
  K2     : production Schur-jacobi (greville-probed) on condensed L2
  K3     : ZERO matrix -> reported as 0; the k=3 Hodge Laplacian L3 is shown
           with uncond/jacobi (no greville Schur atom for k=3 at klevel=2;
           k=3 transfer precond is known non-viable).

Run (GPU):
  W7X_MAP_BATCH=256 XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=8 \
    .venv/bin/python scripts/debug/w7x_dense_matrices_hi.py --ns 12 24 12 --p 3 --nfp 5
"""
from __future__ import annotations
import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "benchmark"))
from benchmark_graddiv_k1_preconditioner import build_sequence, assemble_operators  # noqa: E402
from w7x_dense_matrices import densify, kappa_sym, kappa_diag, kappa_precond, OUTDIR  # noqa: E402
from mrx.operators import (  # noqa: E402
    apply_mass_matrix,
    apply_stiffness,
    apply_laplacian_approx,
    apply_mass_matrix_preconditioner,
    _get_schur_diaginv,
)


def dinv(A):
    dd = np.diag(A).copy()
    dd[dd == 0] = 1.0
    return 1.0 / dd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs=3, default=[12, 24, 12])
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--nfp", type=int, default=5)
    ap.add_argument("--bcs", nargs="+", default=["dbc", "free"])
    args = ap.parse_args()

    cfg = SimpleNamespace(ns=tuple(args.ns), p=args.p, geometry="w7x",
                          cg_tol=1e-8, cg_maxiter=10, epsilon=1 / 3, kappa=1.0,
                          r0=1.0, nfp=args.nfp)
    print(f"=== W7-X dense matrices (hi: M2,M3,K2,K3,L2,L3)  ns={tuple(args.ns)} "
          f"p={args.p} ===", flush=True)
    seq = build_sequence(cfg)
    ops = assemble_operators(seq, klevel=2, both_bc=True)
    os.makedirs(OUTDIR, exist_ok=True)

    report = []
    for bc in args.bcs:
        d = (bc == "dbc")
        n2 = int(seq.n2_dbc if d else seq.n2)
        n3 = int(seq.n3_dbc if d else seq.n3)
        print(f"\n[{bc}] n2={n2} n3={n3}  densifying...", flush=True)

        M2 = densify(lambda v: apply_mass_matrix(seq, ops, v, 2, dirichlet=d), n2)
        M3 = densify(lambda v: apply_mass_matrix(seq, ops, v, 3, dirichlet=d), n3)
        K2 = densify(lambda v: apply_stiffness(seq, ops, v, 2, dirichlet=d), n2)
        K3 = densify(lambda v: apply_stiffness(seq, ops, v, 3, dirichlet=d), n3)
        L2 = densify(lambda v: apply_laplacian_approx(seq, ops, v, 2, dirichlet=d), n2)
        L3 = densify(lambda v: apply_laplacian_approx(seq, ops, v, 3, dirichlet=d), n3)

        bcdir = os.path.join(OUTDIR, bc)
        os.makedirs(bcdir, exist_ok=True)
        for name, A in (("M2", M2), ("M3", M3), ("K2", K2), ("K3", K3),
                        ("L2", L2), ("L3", L3)):
            np.save(os.path.join(bcdir, f"{name}.npy"), A)
        k3norm = float(np.linalg.norm(K3))
        print(f"[{bc}] saved M2,M3,K2,K3,L2,L3 to {bcdir}/  (||K3||={k3norm:.2e})", flush=True)

        grevM2 = densify(lambda v: apply_mass_matrix_preconditioner(
            seq, ops, v, 2, dirichlet=d, kind="tensor"), n2)
        grevM3 = densify(lambda v: apply_mass_matrix_preconditioner(
            seq, ops, v, 3, dirichlet=d, kind="tensor"), n3)
        schur2 = _get_schur_diaginv(ops, 2, d, "tensor_probe")
        schur2 = None if schur2 is None else np.asarray(schur2)

        rows = [
            ("M2", kappa_sym(M2), kappa_diag(M2, dinv(M2)), kappa_precond(M2, grevM2)),
            ("M3", kappa_sym(M3), kappa_diag(M3, dinv(M3)), kappa_precond(M3, grevM3)),
            ("K2(L2)", kappa_sym(L2), kappa_diag(L2, dinv(L2)),
             (kappa_diag(L2, schur2) if schur2 is not None else float("nan"))),
            # K3 == 0 (top-form stiffness); show the k=3 Hodge Laplacian L3 instead.
            ("K3=0/L3", kappa_sym(L3), kappa_diag(L3, dinv(L3)), float("nan")),
        ]
        print(f"\n[{bc}] {'op':9} {'uncond':>12} {'jacobi':>12} {'best':>12}", flush=True)
        for name, u, j, b in rows:
            print(f"[{bc}] {name:9} {u:12.3e} {j:12.3e} {b:12.3e}", flush=True)
            report.append((bc, name, u, j, b))

    with open(os.path.join(OUTDIR, "kappas_hi.txt"), "w") as f:
        f.write(f"W7-X kappas (hi: k=2,3)  ns={tuple(args.ns)} p={args.p}\n")
        f.write("best: M2/M3=greville-mass, K2=greville-Schur-jacobi(L2); "
                "K3=0 (top-form stiffness), L3 shown with uncond/jacobi only\n")
        f.write(f"{'bc':6} {'op':9} {'uncond':>12} {'jacobi':>12} {'best':>12}\n")
        for bc, name, u, j, b in report:
            f.write(f"{bc:6} {name:9} {u:12.3e} {j:12.3e} {b:12.3e}\n")
    print(f"\nsaved -> {OUTDIR}/kappas_hi.txt", flush=True)


if __name__ == "__main__":
    main()
