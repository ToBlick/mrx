"""Assemble W7-X operators densely (M0, M1, K0, K1, condensed L1), save as .npy,
and report condition numbers: unconditioned, jacobi, and current-best.

Best preconditioner per operator:
  M0, M1 : greville tensor mass        (apply_mass_matrix_preconditioner kind='tensor')
  K0     : greville tensor Hodge       (apply_hodge_laplacian_preconditioner kind='auto')
  K1     : production saddle = Schur-jacobi (greville-probed `schur_diaginv`) on the
           condensed L1 = K1 + D0 M0^{-1}(greville) D0^T; the lower mass block is
           greville too (baked into the probed Schur + the condensed operator).

kappa is EFFECTIVE: ratio of extreme eigenvalues with |lam| > rtol * max|lam|
(K1 curl-curl is singular on gradients; K0/M free are singular on constants).

Run (GPU):
  W7X_MAP_BATCH=256 XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=8 \
    .venv/bin/python scripts/debug/w7x_dense_matrices.py --ns 12 24 12 --p 3
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
from mrx.operators import (  # noqa: E402
    apply_mass_matrix,
    apply_stiffness,
    apply_laplacian_approx,
    apply_mass_matrix_preconditioner,
    apply_hodge_laplacian_preconditioner,
    _get_schur_diaginv,
)

OUTDIR = "outputs/w7x_matrices"


def densify(apply, n, chunk=128):
    """Dense matrix of a linear apply: columns are apply(e_i). Batched via vmap."""
    cols = []
    eye = jnp.eye(n, dtype=jnp.float64)
    vapply = jax.jit(jax.vmap(apply, in_axes=1, out_axes=1))
    for j0 in range(0, n, chunk):
        block = eye[:, j0:j0 + chunk]
        cols.append(np.asarray(vapply(block)))
    return np.concatenate(cols, axis=1)


def kappa_eff(evals, rtol=1e-10):
    a = np.sort(np.abs(np.real(evals)))
    a = a[a > rtol * a[-1]]
    return (a[-1] / a[0]) if a.size else float("nan")


def kappa_sym(A):
    return kappa_eff(np.linalg.eigvalsh(0.5 * (A + A.T)))


def kappa_diag(A, dinv):
    """kappa of diag(dinv) @ A, symmetric form sqrt(dinv) A sqrt(dinv)."""
    s = np.sqrt(np.abs(dinv))
    return kappa_eff(np.linalg.eigvalsh(0.5 * (A + A.T) * np.outer(s, s)))


def kappa_precond(A, Pinv):
    """kappa of Pinv @ A (Pinv ~ A^{-1}, SPD): real eigenvalues."""
    return kappa_eff(np.linalg.eigvals(Pinv @ A))


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
    print(f"=== W7-X dense matrices  ns={tuple(args.ns)} p={args.p} ===", flush=True)
    seq = build_sequence(cfg)
    ops = assemble_operators(seq, klevel=1, both_bc=True)
    os.makedirs(OUTDIR, exist_ok=True)

    report = []
    for bc in args.bcs:
        d = (bc == "dbc")
        n0 = int(seq.n0_dbc if d else seq.n0)
        n1 = int(seq.n1_dbc if d else seq.n1)
        print(f"\n[{bc}] n0={n0} n1={n1}  densifying...", flush=True)

        M0 = densify(lambda v: apply_mass_matrix(seq, ops, v, 0, dirichlet=d), n0)
        M1 = densify(lambda v: apply_mass_matrix(seq, ops, v, 1, dirichlet=d), n1)
        K0 = densify(lambda v: apply_stiffness(seq, ops, v, 0, dirichlet=d), n0)
        K1 = densify(lambda v: apply_stiffness(seq, ops, v, 1, dirichlet=d), n1)
        L1 = densify(lambda v: apply_laplacian_approx(seq, ops, v, 1, dirichlet=d), n1)

        bcdir = os.path.join(OUTDIR, bc)
        os.makedirs(bcdir, exist_ok=True)
        for name, A in (("M0", M0), ("M1", M1), ("K0", K0), ("K1", K1), ("L1", L1)):
            np.save(os.path.join(bcdir, f"{name}.npy"), A)
        print(f"[{bc}] saved M0,M1,K0,K1,L1 to {bcdir}/", flush=True)

        # preconditioner dense inverses (apply to identity)
        grevM0 = densify(lambda v: apply_mass_matrix_preconditioner(
            seq, ops, v, 0, dirichlet=d, kind="tensor"), n0)
        grevM1 = densify(lambda v: apply_mass_matrix_preconditioner(
            seq, ops, v, 1, dirichlet=d, kind="tensor"), n1)
        grevK0 = densify(lambda v: apply_hodge_laplacian_preconditioner(
            seq, ops, v, 0, dirichlet=d, kind="auto"), n0)
        schur1 = np.asarray(_get_schur_diaginv(ops, 1, d, "tensor_probe"))

        def dinv(A):
            dd = np.diag(A).copy()
            dd[dd == 0] = 1.0
            return 1.0 / dd

        rows = [
            ("M0", kappa_sym(M0), kappa_diag(M0, dinv(M0)), kappa_precond(M0, grevM0)),
            ("M1", kappa_sym(M1), kappa_diag(M1, dinv(M1)), kappa_precond(M1, grevM1)),
            ("K0", kappa_sym(K0), kappa_diag(K0, dinv(K0)), kappa_precond(K0, grevK0)),
            ("K1(L1)", kappa_sym(L1), kappa_diag(L1, dinv(L1)), kappa_diag(L1, schur1)),
        ]
        print(f"\n[{bc}] {'op':8} {'uncond':>12} {'jacobi':>12} {'best':>12}", flush=True)
        for name, u, j, b in rows:
            print(f"[{bc}] {name:8} {u:12.3e} {j:12.3e} {b:12.3e}", flush=True)
            report.append((bc, name, u, j, b))

    with open(os.path.join(OUTDIR, "kappas.txt"), "w") as f:
        f.write(f"W7-X kappas  ns={tuple(args.ns)} p={args.p}\n")
        f.write("best: M0/M1=greville-mass, K0=greville-hodge, K1=greville-Schur-jacobi(L1)\n")
        f.write(f"{'bc':6} {'op':8} {'uncond':>12} {'jacobi':>12} {'best':>12}\n")
        for bc, name, u, j, b in report:
            f.write(f"{bc:6} {name:8} {u:12.3e} {j:12.3e} {b:12.3e}\n")
    print(f"\nsaved kappas -> {OUTDIR}/kappas.txt", flush=True)


if __name__ == "__main__":
    main()
