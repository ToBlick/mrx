"""Bottom Ritz spectrum of the preconditioned operator smoother . L_0 (k=0
tensor-Hodge surgery preconditioner, production FD bulk inverse), const-deflated,
for rank 1/2/3. Tells us whether the rank>1 kappa blow-up is a SMALL ISOLATED
CLUSTER of near-null modes (deflatable -> deflated Chebyshev viable) or a SMEAR
(nothing to deflate -> rank-1 is the honest answer).

Full-reorthogonalized Lanczos in the A=L_0 inner product (B = sm.L_0 is
A-self-adjoint), returns the whole Ritz spectrum. Prints the bottom 15 + top 3
and the largest relative gap in the bottom quarter.

Usage:
    python scripts/debug/debug_rank_ritz_spectrum.py --geometry toroid --ns 6,12,4 --p 3 --ranks 1,2,3
    python scripts/debug/debug_rank_ritz_spectrum.py --geometry rotating_ellipse --nfp 3 --ns 6,12,4 --p 3 --ranks 1,2,3
"""
import argparse
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
for _p in (ROOT, SCRIPTS, SCRIPTS / "benchmark", SCRIPTS / "debug"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

jax.config.update("jax_enable_x64", True)

from mrx.operators import (  # noqa: E402
    apply_stiffness, apply_mass_matrix, _nullspace_vectors, _core_size,
    _apply_k0_tensor_hodge_bulk_inverse,
    _apply_k0_tensor_hodge_core_block,
    _apply_k0_tensor_hodge_surgery_to_bulk_coupling,
    _apply_k0_tensor_hodge_bulk_to_surgery_coupling,
    _assemble_dense_from_apply, _symmetrize, _symmetric_pseudoinverse,
)
import benchmark_graddiv_k1_preconditioner as dg  # noqa: E402
from benchmark_graddiv_k1_preconditioner import build_sequence, assemble_operators  # noqa: E402

DBC = False


def make_surgery_smoother(seq, ops, core_size, bulk_inv_fn):
    """Full k=0 Hodge preconditioner (Schur block solve) with EXACT core block +
    couplings and the supplied bulk inverse. Mirrors
    _apply_k0_tensor_hodge_preconditioner. (Inlined from the former zfirst probe.)"""
    def core_block(rc):
        return _apply_k0_tensor_hodge_core_block(seq, ops, core_size, rc, dirichlet=DBC)

    def surgery_to_bulk(rc):     # L_bc @ rc   (bulk from core)
        return _apply_k0_tensor_hodge_surgery_to_bulk_coupling(seq, ops, core_size, rc, dirichlet=DBC)

    def bulk_to_surgery(rb):     # L_cb @ rb   (core from bulk)
        return _apply_k0_tensor_hodge_bulk_to_surgery_coupling(seq, ops, core_size, rb, dirichlet=DBC)

    ass = _symmetrize(_assemble_dense_from_apply(core_block, core_size, sequential=True))
    schur = _symmetrize(_assemble_dense_from_apply(
        lambda rc: ass @ rc - bulk_to_surgery(bulk_inv_fn(surgery_to_bulk(rc))),
        core_size, sequential=True))
    schur_inv = _symmetric_pseudoinverse(schur)

    def smoother(rhs):
        rc, rb = rhs[:core_size], rhs[core_size:]
        y = bulk_inv_fn(rb)
        z = schur_inv @ (rc - bulk_to_surgery(y))
        x_b = y - bulk_inv_fn(surgery_to_bulk(z))
        return jnp.concatenate([z, x_b])
    return smoother


def dense_spectrum(a_matvec, precond, n, project):
    """Materialize B = (project . precond . a_matvec) on the n unit vectors and
    return its (real) eigenvalues, sorted ascending. B is A-self-adjoint so its
    eigenvalues are real; we drop the deflated constant's zero eigenvalue."""
    cols = []
    for i in range(n):
        e = jnp.zeros((n,), dtype=jnp.float64).at[i].set(1.0)
        cols.append(project(precond(a_matvec(e))))
    B = np.asarray(jnp.stack(cols, axis=1))
    ev = np.linalg.eigvals(B)
    ev = np.real(ev)
    # the const-deflation leaves an exact zero eigenvalue; drop the near-zeros
    # from deflation only (|ev| below 1e-10 * max), keep the genuine small modes.
    keep = ev[ev > 1e-10 * np.max(np.abs(ev))]
    return np.sort(keep)


def deflated_smoother(seq, ops, cs):
    """const-deflated surgery preconditioner (production FD bulk inverse) + the
    L_0 matvec + the A-PD projector, matching the cheb_L0 setup."""
    sm_raw = make_surgery_smoother(
        seq, ops, cs, lambda rb, _f=ops.k0_tensor_hodge_precond.free:
        _apply_k0_tensor_hodge_bulk_inverse(_f, rb))

    def s_hat(x):
        return apply_stiffness(seq, ops, x, 0, dirichlet=DBC)

    c0 = jnp.asarray(_nullspace_vectors(ops, 0, DBC))
    Mc0 = jnp.stack([apply_mass_matrix(seq, ops, c0[i], 0, dirichlet=DBC)
                     for i in range(c0.shape[0])], axis=0)
    cn = jnp.sqrt(jnp.einsum("ij,ij->i", c0, Mc0))
    c0, Mc0 = c0 / cn[:, None], Mc0 / cn[:, None]

    def Pp(x):
        return x - jnp.einsum("i,ij->j", Mc0 @ x, c0)

    def Pd(b):
        return b - jnp.einsum("i,ij->j", c0 @ b, Mc0)

    def sm(b):
        return Pp(sm_raw(Pd(b)))

    return s_hat, sm, Pp


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry", type=str, default="toroid")
    ap.add_argument("--ranks", type=str, default="1,2,3")
    ap.add_argument("--ns", type=str, default="6,12,4")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--epsilon", type=float, default=1.0 / 3.0)
    ap.add_argument("--kappa", type=float, default=1.2)
    ap.add_argument("--r0", type=float, default=1.0)
    ap.add_argument("--nfp", type=int, default=3)
    ap.add_argument("--cg-tol", type=float, default=1e-10)
    ap.add_argument("--cg-maxiter", type=int, default=600)
    args = ap.parse_args()
    args.ns = [int(v) for v in args.ns.split(",")]
    ranks = [int(r) for r in args.ranks.split(",")]

    dg.DIRICHLET = False
    print(f"[diag] DENSE eig spectrum of smoother.L_0: geometry={args.geometry} ns={args.ns} "
          f"p={args.p} free BC; ranks={ranks}", flush=True)

    t = time.perf_counter()
    seq = build_sequence(args)
    assemble_operators(seq, rank=ranks[0], klevel=0)
    seq._compute_nullspaces(dg.BETTI)
    n0 = int(seq.n0)
    cs = int(_core_size(seq))
    print(f"[diag] build in {(time.perf_counter()-t)*1e3:.1f} ms; n0={n0} core={cs}", flush=True)

    for rank in ranks:
        ops = assemble_operators(seq, rank=rank, klevel=0)
        s_hat, sm, Pp = deflated_smoother(seq, ops, cs)
        ritz = dense_spectrum(s_hat, sm, n0, Pp)
        lo = ritz[:15]
        hi = ritz[-3:]
        # largest relative gap in the bottom quarter (consecutive ratio)
        nb = max(3, len(ritz) // 4)
        bottom = ritz[:nb]
        ratios = bottom[1:] / np.maximum(bottom[:-1], 1e-300)
        gi = int(np.argmax(ratios))
        print(f"\n========== rank {rank}  ({len(ritz)} nonzero eigs) ==========", flush=True)
        print(f"  kappa = lmax/lmin = {ritz[-1]/max(ritz[0],1e-300):.3e}  "
              f"(lmin={ritz[0]:.3e}, lmax={ritz[-1]:.3e})", flush=True)
        print(f"  bottom 15 eigs: " + " ".join(f"{v:.3e}" for v in lo), flush=True)
        print(f"  top 3 eigs:     " + " ".join(f"{v:.3e}" for v in hi), flush=True)
        print(f"  largest gap in bottom {nb}: ritz[{gi}]={bottom[gi]:.3e} -> "
              f"ritz[{gi+1}]={bottom[gi+1]:.3e}  (ratio {ratios[gi]:.2e}); "
              f"=> {gi+1} mode(s) below the gap", flush=True)

    print("\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
