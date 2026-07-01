"""Densify the de Rham derivative coupling blocks D0, D1, D2 for W7-X, so the
saddle-point systems can be assembled for visualization:

    saddle_k = [[ S_k,        D_{k-1}  ],
                [ D_{k-1}^T,  -M_{k-1} ]]   for k = 1, 2, 3

D_{k-1} = M_k G_{k-1} : V_{k-1} -> V_k  (weak derivative), shape (n_k, n_{k-1}).
S_k (=K1,K2; K3=0) and M_{k-1} (=M0,M1,M2) are already saved by the other two
dense scripts. dbc only (the clean saddle).

Run (GPU):
  W7X_MAP_BATCH=256 XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=8 \
    .venv/bin/python scripts/debug/w7x_dense_derivatives.py --ns 12 24 12 --p 3 --nfp 5
"""
from __future__ import annotations
import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "benchmark"))
from benchmark_graddiv_k1_preconditioner import build_sequence, assemble_operators  # noqa: E402
from w7x_dense_matrices import densify, OUTDIR  # noqa: E402
from mrx.operators import apply_derivative_matrix  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs=3, default=[12, 24, 12])
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--nfp", type=int, default=5)
    ap.add_argument("--bcs", nargs="+", default=["dbc"])
    args = ap.parse_args()

    cfg = SimpleNamespace(ns=tuple(args.ns), p=args.p, geometry="w7x",
                          cg_tol=1e-8, cg_maxiter=10, epsilon=1 / 3, kappa=1.0,
                          r0=1.0, nfp=args.nfp)
    print(f"=== W7-X derivative blocks D0,D1,D2  ns={tuple(args.ns)} p={args.p} ===",
          flush=True)
    seq = build_sequence(cfg)
    ops = assemble_operators(seq, klevel=2, both_bc=True)

    for bc in args.bcs:
        d = (bc == "dbc")
        bcdir = os.path.join(OUTDIR, bc)
        os.makedirs(bcdir, exist_ok=True)
        ndof = {0: int(seq.n0_dbc if d else seq.n0),
                1: int(seq.n1_dbc if d else seq.n1),
                2: int(seq.n2_dbc if d else seq.n2),
                3: int(seq.n3_dbc if d else seq.n3)}
        for km1 in (0, 1, 2):
            n_in = ndof[km1]            # V_{k-1}
            D = densify(lambda v, _k=km1: apply_derivative_matrix(
                seq, ops, v, _k, dirichlet_in=d, dirichlet_out=d, transpose=False), n_in)
            np.save(os.path.join(bcdir, f"D{km1}.npy"), D)
            print(f"[{bc}] D{km1}: {D.shape}  (V{km1}->V{km1 + 1})  saved", flush=True)
    print(f"\nsaved D0,D1,D2 -> {OUTDIR}/<bc>/", flush=True)


if __name__ == "__main__":
    main()
