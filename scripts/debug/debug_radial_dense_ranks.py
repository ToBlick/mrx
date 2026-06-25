"""Rank diagnostics for the radial-dense k=0 bulk inverse.

Reports the separable ranks that the radial-dense model relies on:
  - the metric channels' unfolding singular spectra (is θ,ζ really rank-1, and how
    much radial / (r,θ) rank is there?);
  - the assembled dense radial blocks B[j,k] (the "r-blocks"): numerical rank and
    conditioning, and how much they VARY across angular modes (that variation is
    exactly the radial coupling radial-dense captures and the FD denom drops);
  - the (r,θ) "rt-blocks": the separable (Kronecker) rank of the metric in the
    (r,θ) plane (the rank-≥2 we keep dense) and vs ζ (the rank-1 we diagonalise).

Usage (activate venv first: `source .venv/bin/activate`):
    python scripts/debug/debug_radial_dense_ranks.py --geometry toroid --ns 6,12,4 --p 3 --rank 2
    python scripts/debug/debug_radial_dense_ranks.py --geometry rotating_ellipse --nfp 3 --ns 6,12,4 --p 3 --rank 2
"""
import argparse
import sys
from pathlib import Path

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
for _p in (ROOT, SCRIPTS, SCRIPTS / "benchmark", SCRIPTS / "debug"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

jax.config.update("jax_enable_x64", True)

from mrx.operators import (  # noqa: E402
    _k0_stiffness_diagonal_metric_tensors,
    _assemble_k0_stiffness_fd_bulk_factors,
)
import benchmark_graddiv_k1_preconditioner as dg  # noqa: E402
from benchmark_graddiv_k1_preconditioner import build_sequence, assemble_operators  # noqa: E402


def _eff_rank(svals, tols=(1e-2, 1e-4, 1e-6, 1e-8)):
    s = np.sort(np.asarray(svals))[::-1]
    smax = s[0] if s.size else 0.0
    return {t: int(np.sum(s > t * smax)) for t in tols}, s


def _unfolding_svals(tensor, axis):
    """Singular values of the mode-`axis` unfolding of a 3-tensor."""
    t = np.moveaxis(np.asarray(tensor), axis, 0)
    mat = t.reshape(t.shape[0], -1)
    return np.linalg.svd(mat, compute_uv=False)


def _joint_svals(tensor, joint_axes, single_axis):
    """SVD of [joint_axes] x [single_axis] unfolding (e.g. (r,θ) vs ζ)."""
    t = np.asarray(tensor)
    perm = list(joint_axes) + [single_axis]
    t = np.transpose(t, perm)
    mat = t.reshape(-1, t.shape[-1])
    return np.linalg.svd(mat, compute_uv=False)


def report_metric_ranks(name, W):
    nr, nt, nz = W.shape
    print(f"\n--- metric channel {name}  (quad grid {W.shape}) ---", flush=True)
    for ax, label in [(0, "radial r"), (1, "poloidal θ"), (2, "toroidal ζ")]:
        ranks, s = _eff_rank(_unfolding_svals(W, ax))
        top = " ".join(f"{v:.2e}" for v in s[:5])
        print(f"  {label:11s} unfolding rank @[1e-2,1e-4,1e-6,1e-8] = "
              f"{[ranks[t] for t in (1e-2,1e-4,1e-6,1e-8)]}  σ(top5)= {top}", flush=True)
    # joint (r,θ) vs ζ  -> ζ separability ("rt-block rank vs ζ"; expect ~1)
    rz, sz = _eff_rank(_joint_svals(W, (0, 1), 2))
    print(f"  (r,θ)⊗[ζ]    rank @[1e-2,1e-4,1e-6,1e-8] = "
          f"{[rz[t] for t in (1e-2,1e-4,1e-6,1e-8)]}  σ(top5)= "
          + " ".join(f"{v:.2e}" for v in sz[:5]) + "  <- ζ rank-1 check", flush=True)
    # (r) vs (θ) on the dominant ζ-slice -> (r,θ) separable rank ("rt rank"; expect ~2)
    # use the ζ-averaged (r,θ) slice as the representative rt-block
    rt_slice = np.asarray(W).mean(axis=2)            # (nr, nt)
    rrt, srt = _eff_rank(np.linalg.svd(rt_slice, compute_uv=False))
    print(f"  [r]x[θ] (ζ-avg) rank @[1e-2,1e-4,1e-6,1e-8] = "
          f"{[rrt[t] for t in (1e-2,1e-4,1e-6,1e-8)]}  σ(top5)= "
          + " ".join(f"{v:.2e}" for v in srt[:5]) + "  <- (r,θ) rank (keep dense)", flush=True)


def report_radial_block_ranks(bulk_data):
    V_t = np.asarray(bulk_data['bulk_modal_basis_t'])
    V_z = np.asarray(bulk_data['bulk_modal_basis_z'])
    op_r = [np.asarray(a) for a in bulk_data['bulk_term_op_r']]
    op_t = [np.asarray(a) for a in bulk_data['bulk_term_op_t']]
    op_z = [np.asarray(a) for a in bulk_data['bulk_term_op_z']]
    nr, nt, nz = bulk_data['bulk_shape']
    n_terms = len(op_r)

    # rebuild the dense radial blocks B[j,k] (same formula as the production build)
    B = np.zeros((nt, nz, nr, nr))
    for Ar, At, Az in zip(op_r, op_t, op_z):
        d_t = np.einsum("ji,jk,ki->i", V_t, At, V_t)
        d_z = np.einsum("ji,jk,ki->i", V_z, Az, V_z)
        B += d_t[:, None, None, None] * d_z[None, :, None, None] * Ar[None, None, :, :]

    print(f"\n--- radial blocks B[j,k] (the r-blocks): {nt}×{nz} = {nt*nz} blocks of {nr}×{nr} "
          f"(from {n_terms} Kronecker terms) ---", flush=True)
    Bflat = B.reshape(nt * nz, nr, nr)
    conds, ranks_, mineig = [], [], []
    for blk in Bflat:
        sym = 0.5 * (blk + blk.T)
        w = np.linalg.eigvalsh(sym)
        wabs = np.abs(w)
        conds.append(wabs.max() / max(wabs.min(), 1e-300))
        ranks_.append(int(np.sum(wabs > 1e-10 * wabs.max())))
        mineig.append(w.min())
    conds, ranks_, mineig = np.array(conds), np.array(ranks_), np.array(mineig)
    print(f"  numerical rank (eig>1e-10·max): min={ranks_.min()} max={ranks_.max()} "
          f"(full = {nr}); blocks SPD: min eig over all blocks = {mineig.min():.3e}", flush=True)
    print(f"  cond(B[j,k]):  min={conds.min():.2e} median={np.median(conds):.2e} "
          f"max={conds.max():.2e}", flush=True)

    # how much do the blocks VARY across angular modes? (variation == radial coupling
    # radial-dense captures; if blocks were a scalar multiple of one matrix -> FD)
    mean_B = Bflat.mean(axis=0)
    num = np.array([np.linalg.norm(blk - mean_B) for blk in Bflat])
    rel_var = num / (np.linalg.norm(mean_B) + 1e-300)
    print(f"  block variation ‖B[j,k]−mean‖/‖mean‖: min={rel_var.min():.2e} "
          f"median={np.median(rel_var):.2e} max={rel_var.max():.2e}  "
          f"(0 ⇒ radial-dense ≡ FD; larger ⇒ more radial coupling captured)", flush=True)

    # "rt-block rank": stack the blocks as a (nt·nz) × (nr·nr) matrix and take its
    # rank -> how many distinct radial blocks there really are across angular modes
    stack = Bflat.reshape(nt * nz, nr * nr)
    rs, ss = _eff_rank(np.linalg.svd(stack, compute_uv=False))
    print(f"  distinct-radial-block rank (svd of stacked blocks) @[1e-2,1e-4,1e-6,1e-8] = "
          f"{[rs[t] for t in (1e-2,1e-4,1e-6,1e-8)]}  σ(top6)= "
          + " ".join(f"{v:.2e}" for v in ss[:6]), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry", type=str, default="toroid")
    ap.add_argument("--ns", type=str, default="6,12,4")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--rank", type=int, default=2)
    ap.add_argument("--epsilon", type=float, default=1.0 / 3.0)
    ap.add_argument("--kappa", type=float, default=1.2)
    ap.add_argument("--r0", type=float, default=1.0)
    ap.add_argument("--nfp", type=int, default=3)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--cg-tol", type=float, default=1e-10)
    ap.add_argument("--cg-maxiter", type=int, default=600)
    ap.add_argument("--dbc", action="store_true")
    args = ap.parse_args()
    args.ns = [int(v) for v in args.ns.split(",")]

    dg.DIRICHLET = bool(args.dbc)
    print(f"[ranks] geometry={args.geometry} ns={args.ns} p={args.p} "
          f"CP rank={args.rank} {'dbc' if args.dbc else 'free'} BC", flush=True)
    seq = build_sequence(args)
    assemble_operators(seq, rank=args.rank, klevel=0)

    metric = _k0_stiffness_diagonal_metric_tensors(seq)
    for name in ("alpha_rr", "alpha_thetatheta", "alpha_zetazeta"):
        report_metric_ranks(name, np.asarray(metric[name]))

    bulk_data = _assemble_k0_stiffness_fd_bulk_factors(
        seq, dirichlet=bool(args.dbc), rank=args.rank,
        cp_maxiter=100, cp_tol=1e-9, cp_ridge=1e-12, radial_dense=True)
    report_radial_block_ranks(bulk_data)
    print("\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
