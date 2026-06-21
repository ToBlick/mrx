"""Dense diagnostics for the k=0 (scalar grad-grad) Hodge preconditioner.

k=0 is the bottom of the complex: no projector / P_B, just the tensor Hodge
preconditioner P_0 for L_0 = G_0^T M_1 G_0. So the dense battery is short:

  (A) operator consistency ||apply_stiffness(.,0) - G_0^T M_1 G_0|| / ||.||
      (validates the true-G fix at the grad level; ~0 on a corrected core)
  (C) preconditioned spectrum eig(P_0 . L_0): how well the k=0 tensor Hodge
      preconditioner (the near-exact, kappa~2 scalar atom that the k=1/k=2
      projectors rely on) conditions L_0.

Both BCs (dbc nonsingular; free has the constant nullspace -> excluded from the
spectral ratio). Needs full operator assembly -> GPU/SLURM.
"""

import argparse
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
jax.config.update("jax_enable_x64", True)

from diag_graddiv_subspace_preconditioner import build_sequence, assemble_operators  # noqa: E402
from mrx.operators import (  # noqa: E402
    apply_stiffness, apply_mass_matrix, apply_incidence_matrix,
    apply_laplacian_preconditioner,
)


def build_dense(fn, n_in):
    cols = []
    for j in range(n_in):
        e = jnp.zeros((n_in,), dtype=jnp.float64).at[j].set(1.0)
        cols.append(np.asarray(jax.device_get(fn(e))))
    return np.stack(cols, axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ns", type=str, default="6,12,4")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--geometry", type=str, default="rotating_ellipse")
    ap.add_argument("--epsilon", type=float, default=1.0 / 3.0)
    ap.add_argument("--kappa", type=float, default=1.2)
    ap.add_argument("--r0", type=float, default=1.0)
    ap.add_argument("--nfp", type=int, default=3)
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--cg-tol", type=float, default=1e-10)
    ap.add_argument("--cg-maxiter", type=int, default=2000)
    args = ap.parse_args()
    args.ns = [int(v) for v in args.ns.split(",")]

    t0 = time.perf_counter()
    seq = build_sequence(args)
    ops = assemble_operators(seq, rank=args.rank, klevel=0)
    print(f"[diag] seq+ops built in {time.perf_counter()-t0:.1f}s")

    for DBC in (True, False):
        bc = "dbc" if DBC else "free"
        n0 = int(seq.n0_dbc if DBC else seq.n0)
        print(f"\n========== k=0 {bc} (n0={n0}) ==========")

        # (A) consistency: apply_stiffness(.,0) vs composed G_0^T M_1 G_0
        def composed_L0(v):
            g0 = apply_incidence_matrix(seq, ops, v, 0, dirichlet_in=DBC, dirichlet_out=DBC)
            m1 = apply_mass_matrix(seq, ops, g0, 1, dirichlet=DBC)
            return apply_incidence_matrix(seq, ops, m1, 0,
                                          dirichlet_in=DBC, dirichlet_out=DBC, transpose=True)
        L0 = build_dense(lambda e: apply_stiffness(seq, ops, e, 0, dirichlet=DBC), n0)
        L0c = build_dense(composed_L0, n0)
        relA = np.linalg.norm(L0 - L0c) / max(np.linalg.norm(L0), 1e-30)
        print(f"(A) ||apply_stiffness(.,0) - G_0^T M_1 G_0|| / ||.|| = {relA:.3e}")

        # (C) preconditioned spectrum eig(P_0 . L_0), P_0 = k=0 tensor Hodge precond
        P0 = build_dense(lambda r: apply_laplacian_preconditioner(
            seq, ops, r, 0, dirichlet=DBC, kind="tensor"), n0)
        ev = np.sort(np.real(np.linalg.eigvals(P0 @ L0)))
        pos = ev[ev > 1e-9 * ev.max()]  # drop the constant nullspace (free BC)
        band = np.mean((pos > 0.9) & (pos < 1.1)) * 100.0
        nz = int(np.sum(ev <= 1e-9 * ev.max()))
        print(f"(C) eig(P_0 . L_0): [{pos.min():.3e}, {pos.max():.3e}] "
              f"cond={pos.max()/pos.min():.3e}  in[0.9,1.1]={band:5.1f}%  near0={nz}")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
