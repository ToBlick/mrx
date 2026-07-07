"""Diagnostics for the greville k=1/2 (and k=0/3) MASS lumping quality.

Question being answered: on W7-X the greville vector-mass preconditioner needs
more CG iterations at k=1/2 than at k=0/3. The greville atom is a per-component
diagonal-metric lump `D^{-1/2} (M0_r^{-1}⊗M0_t^{-1}⊗M0_z^{-1}) D^{-1/2}` with the
weight D collocated pointwise at the Greville abscissae (see
`mrx/preconditioners.py:_build_greville_mass_block_factors`). Two independent things
can degrade it:

  (A) DROPPED OFF-DIAGONAL METRIC. The lump keeps only the diagonal of G (k=2) /
      G^{-1} (k=1). We measure how far G and G^{-1} are from their own diagonal,
      pointwise, both as per-pair normalized off-diagonals |g_ij|/sqrt(g_ii g_jj)
      (Cauchy-Schwarz => in [0,1]) and as the spectral error of the diagonal
      approximation  max|eig(D^{-1/2} M D^{-1/2}) - 1|  (relative 2-norm distance of
      the symmetrically-scaled matrix from I; <<1 => coupling negligible).

  (B) SUB-CELL VARIATION OF THE LUMPED WEIGHT. The pointwise Greville sample is a
      good lump only if the weight w_c is ~constant over a basis function's support.
      We have w at every quadrature point, grouped by element (2p Gauss points per
      knot span), so we report the per-cell coefficient of variation std/|mean| of
      each component weight. Large per-cell CoV => the single-sample lump cannot
      represent the integrated weighted mass, regardless of coupling.
      (Caveat: a degree-p basis spans p+1 cells; per-cell CoV is a lower-bound proxy
       for the true per-support variation, but it is monotone and cheap.)

  (B2) GEOMETRIC-MEAN-OF-ENDPOINTS error. The lump replaces the true entry
      ∫ w Φ_I Φ_J with √(D_I D_J)·M0[I,J], i.e. it weights entry (I,J) by the
      GEOMETRIC MEAN of the weight at the two Greville nodes I,J. For degree-p
      splines those nodes overlap whenever |I-J| <= p, so they can sit up to p knot
      spans apart -- and the geometric mean of two far-apart samples can be a poor
      stand-in for the support-averaged weight the true mass applies. We build the
      weight on the Greville grid and, per axis and per lag l=1..p, compare
      √(w_i w_{i+l}) (the lump's effective weight) against the arithmetic mean of the
      intervening Greville samples (a proxy for the true ∫w average). Error growing
      with lag == the "far-apart endpoints" penalty; the lag=p row is the worst
      overlapping pair.

  (C) THE SYMPTOM. CG iteration counts for the mass inversions M_k x = b through the
      production path (greville vs jacobi), k=0..3, DBC and free — copied from
      greville_mass_prod_verify.py.

Weights (natural component order c: 0=rho, 1=theta, 2=zeta):
  k=0: w = J           k=3: w = 1/J
  k=1: w_c = J * g^{cc}   (jac * metric_inv[c,c])
  k=2: w_c = g_cc / J     (metric[c,c] / jac)

Run one geometry per invocation (loop externally for cyl/toroid/w7x). GPU-friendly;
the geometry analysis is pure host numpy over the quad points.

  python scripts/debug/greville_lumping_diagnostics.py --geometry w7x  --ns 12 24 12 --p 3 --nfp 5
  python scripts/debug/greville_lumping_diagnostics.py --geometry toroid --ns 12 24 12 --p 3
  python scripts/debug/greville_lumping_diagnostics.py --geometry cylinder --ns 12 24 12 --p 3 --no-cg
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
from benchmark_graddiv_k1_preconditioner import build_sequence  # noqa: E402
from mrx.operators import (  # noqa: E402
    apply_mass_matrix,
    apply_inverse_mass_matrix,
    assemble_mass_surgery_preconditioner,
    assemble_tensor_mass_preconditioner,
    assemble_mass_jacobi_preconditioner,
)

COMP = ("rho", "theta", "zeta")
PAIRS = ((0, 1, "rho-theta"), (0, 2, "rho-zeta"), (1, 2, "theta-zeta"))


def stats(a) -> str:
    """mean / p50 / p90 / p99 / max over the finite entries."""
    a = np.asarray(a, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return "  (no finite entries)"
    p50, p90, p99, pmax = np.percentile(a, [50, 90, 99, 100])
    return (f"mean={a.mean():.3e}  p50={p50:.3e}  p90={p90:.3e}  "
            f"p99={p99:.3e}  max={pmax:.3e}")


def diag_approx_err(M: np.ndarray) -> np.ndarray:
    """max|eig(D^{-1/2} M D^{-1/2}) - 1| pointwise for a stack of SPD 3x3 (N,3,3).

    Relative 2-norm error of approximating M by diag(M): 0 iff M is diagonal.
    """
    d = np.diagonal(M, axis1=1, axis2=2)                      # (N,3)
    d = np.sqrt(np.clip(d, 1e-300, None))
    Mn = M / (d[:, :, None] * d[:, None, :])                 # unit diagonal
    ev = np.linalg.eigvalsh(Mn)                              # (N,3) ascending
    return np.maximum(np.abs(ev[:, -1] - 1.0), np.abs(ev[:, 0] - 1.0))


def to_grid(seq, field_flat: np.ndarray) -> np.ndarray:
    """Quad-point field (N_q, ...) -> tensor grid in natural (rho, theta, zeta, ...).

    Storage order is (ny, nx, nz) = (theta, rho, zeta) (see
    _quadrature_tensor_shape); moveaxis brings rho (=nx) to the front.
    """
    ny, nx, nz = seq.quad.ny, seq.quad.nx, seq.quad.nz
    f = np.asarray(field_flat).reshape(ny, nx, nz, *field_flat.shape[1:])
    return np.moveaxis(f, 0, 1)                              # (nx, ny, nz, ...) = (rho, theta, zeta)


def axis_elements(seq):
    """Per natural axis (rho, theta, zeta): (n_elements, quad_pts_per_element, size)."""
    quad_pts = (seq.quad.x_x, seq.quad.x_y, seq.quad.x_z)     # x_x=rho, x_y=theta, x_z=zeta
    out = []
    for a in range(3):
        basis = seq.basis_0.Λ[a]
        T = np.asarray(basis.T)
        p = int(basis.p)
        breaks = T[p:T.shape[0] - p]                         # element boundaries used by select_quadrature
        ne = int(breaks.shape[0] - 1)
        nq = int(quad_pts[a].size)
        q = nq // ne
        out.append((ne, q, nq, q * ne == nq))
    return out


def percell_cov(seq, W: np.ndarray, elems) -> np.ndarray:
    """Per-3D-cell coefficient of variation std/|mean| of weight field W (rho,theta,zeta)."""
    (ne0, q0, _, _), (ne1, q1, _, _), (ne2, q2, _, _) = elems
    W6 = W.reshape(ne0, q0, ne1, q1, ne2, q2)
    m = W6.mean(axis=(1, 3, 5))
    s = W6.std(axis=(1, 3, 5))
    return s / np.maximum(np.abs(m), 1e-300)


def greville_grid_weights(seq, r_drop=0):
    """Weights evaluated on the primal Greville tensor grid (nr,nt,nz), natural order.

    Mirrors _build_greville_mass_block_factors' collocation (primal Greville per axis,
    clamped endpoints nudged inward). Drops the inner ``r_drop`` radial rings (surgery).
    Returns (grev_1d_list, {label: W(nr,nt,nz)}).
    """
    from mrx.geometry import compute_geometry_terms  # noqa: PLC0415
    types = seq.basis_0.types
    eps = 1e-7
    grev = []
    for a in range(3):
        g1 = np.asarray(seq.basis_0.Λ[a].greville_points())
        if types[a] == "clamped":
            g1 = np.clip(g1, eps, 1.0 - eps)
        grev.append(g1)
    grev[0] = grev[0][r_drop:]                                # surgery: drop inner radial rings
    rr, tt, zz = np.meshgrid(grev[0], grev[1], grev[2], indexing="ij")
    pts = jnp.asarray(np.stack([rr.ravel(), tt.ravel(), zz.ravel()], axis=-1))
    metric, minv, jac = (np.asarray(v) for v in compute_geometry_terms(seq.map, pts))
    shp = (grev[0].size, grev[1].size, grev[2].size)
    Jg = jac.reshape(shp)
    gg = metric.reshape(*shp, 3, 3)
    gig = minv.reshape(*shp, 3, 3)
    W = {"k0        w=J": Jg, "k3      w=1/J": 1.0 / Jg}
    for c, name in enumerate(COMP):
        W[f"k1 {name:5s} w=J*g^cc"] = Jg * gig[..., c, c]
    for c, name in enumerate(COMP):
        W[f"k2 {name:5s} w=g_cc/J"] = gg[..., c, c] / Jg
    return grev, W


def geomean_endpoint_error(W: np.ndarray, axis: int, lag: int) -> np.ndarray:
    """|sqrt(w_i w_{i+lag}) - mean(w_i..w_{i+lag})| / |mean| along `axis`.

    sqrt(w_i w_{i+lag}) is the lump's effective weight on an overlapping (i,i+lag)
    entry; the arithmetic window mean proxies the true support-averaged weight.
    """
    n = W.shape[axis]
    if lag >= n:
        return np.array([np.nan])

    def sl(s):
        return tuple(s if a == axis else slice(None) for a in range(W.ndim))

    w0 = W[sl(slice(0, n - lag))]
    wl = W[sl(slice(lag, n))]
    geomean = np.sqrt(np.abs(w0 * wl))
    stack = np.stack([W[sl(slice(s, n - lag + s))] for s in range(lag + 1)], axis=0)
    winmean = stack.mean(axis=0)
    return np.abs(geomean - winmean) / np.maximum(np.abs(winmean), 1e-300)


def q3(a) -> str:
    a = np.asarray(a, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return "  n/a"
    p50, p90, pmax = np.percentile(a, [50, 90, 100])
    return f"p50={p50:.2e} p90={p90:.2e} max={pmax:.2e}"


def dof(seq, k, dirichlet):
    return int(getattr(seq, f"n{k}_dbc" if dirichlet else f"n{k}"))


def it_conv(info):
    v = int(info)
    return abs(v), (v <= 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", default="toroid")
    ap.add_argument("--ns", type=int, nargs=3, default=[12, 24, 12])
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--epsilon", type=float, default=1.0 / 3.0)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--r0", type=float, default=1.0)
    ap.add_argument("--nfp", type=int, default=3)
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--maxiter", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-cg", action="store_true", help="skip the CG iteration-count section")
    ap.add_argument("--r-drop", type=int, default=2,
                    help="inner radial rings removed by surgery; excluded from all "
                         "diagnostics (default 2, matching the k>=0 mass blocks)")
    args = ap.parse_args()

    cfg = SimpleNamespace(
        ns=tuple(args.ns), p=args.p, geometry=args.geometry, cg_tol=args.tol,
        cg_maxiter=args.maxiter, epsilon=args.epsilon, kappa=args.kappa,
        r0=args.r0, nfp=args.nfp,
    )
    print(f"=== greville MASS lumping diagnostics  {args.geometry}  "
          f"ns={tuple(args.ns)} p={args.p} nfp={args.nfp} ===", flush=True)
    seq = build_sequence(cfg)

    # ---- geometry at the quadrature points -------------------------------------
    g = np.asarray(seq.geometry.metric_jkl)                  # (N_q,3,3)  G = DF^T DF
    ginv = np.asarray(seq.geometry.metric_inv_jkl)           # (N_q,3,3)  G^{-1}
    J = np.asarray(seq.geometry.jacobian_j)                  # (N_q,)     det DF
    N_q = g.shape[0]
    n_bad_J = int(np.sum(~(np.isfinite(J) & (J > 0))))
    print(f"\nN_q = {N_q}   non-positive/non-finite J points: {n_bad_J}", flush=True)

    # ---- surgery exclusion: drop the inner r_drop radial rings ------------------
    # The inner r_drop radial BASES are removed by surgery, so their near-axis
    # weight blow-up must not enter the bulk-lump diagnostic. Cut by the rho of the
    # first RETAINED radial Greville ring (Greville nodes are 1:1 with bases).
    r_drop = int(args.r_drop)
    grev_full_r = np.asarray(seq.basis_0.Λ[0].greville_points())
    if seq.basis_0.types[0] == "clamped":
        grev_full_r = np.clip(grev_full_r, 1e-7, 1.0 - 1e-7)
    rho_cut = float(grev_full_r[r_drop]) if 0 < r_drop < grev_full_r.size else 0.0
    ny, nx, nz = seq.quad.ny, seq.quad.nx, seq.quad.nz
    rho_flat = np.broadcast_to(np.asarray(seq.quad.x_x)[None, :, None], (ny, nx, nz)).reshape(-1)
    keep = rho_flat >= rho_cut
    print(f"surgery cut: drop inner {r_drop} radial rings -> rho >= {rho_cut:.4f}; "
          f"keep {int(keep.sum())}/{keep.size} quad pts", flush=True)
    gk, gik = g[keep], ginv[keep]

    # ---- (A) off-diagonal metric (bulk only) ----------------------------------
    print("\n--- (A) OFF-DIAGONAL METRIC (does dropping coupling cost anything?) ---")
    dg = np.sqrt(np.clip(np.diagonal(gk, axis1=1, axis2=2), 1e-300, None))
    dgi = np.sqrt(np.clip(np.diagonal(gik, axis1=1, axis2=2), 1e-300, None))
    print("normalized |g_ij|/sqrt(g_ii g_jj)  (G, relevant to k=2):")
    for i, j, name in PAIRS:
        print(f"    {name:11s} {stats(np.abs(gk[:, i, j]) / (dg[:, i] * dg[:, j]))}")
    print("normalized |ginv_ij|/sqrt(ginv_ii ginv_jj)  (G^-1, relevant to k=1):")
    for i, j, name in PAIRS:
        print(f"    {name:11s} {stats(np.abs(gik[:, i, j]) / (dgi[:, i] * dgi[:, j]))}")
    print("diagonal-approx spectral error  max|eig(D^-1/2 M D^-1/2) - 1|:")
    print(f"    G   (k=2)   {stats(diag_approx_err(gk))}")
    print(f"    Ginv(k=1)   {stats(diag_approx_err(gik))}")

    # ---- (B) sub-cell variation of the lumped weight ---------------------------
    print("\n--- (B) SUB-CELL VARIATION of the lumped weight (lump fidelity) ---")
    elems = axis_elements(seq)
    for a, name in enumerate(COMP):
        ne, q, nq, ok = elems[a]
        flag = "" if ok else "  [WARN: nq not divisible by n_elem -> per-cell reshape invalid]"
        print(f"    axis {name:5s}: n_elem={ne:4d}  quad_pts/elem={q:3d}  n_q={nq:4d}{flag}")
    Jg = to_grid(seq, J)
    gg = to_grid(seq, g)
    gig = to_grid(seq, ginv)
    weights = {
        "k0        w=J": Jg,
        "k3      w=1/J": 1.0 / Jg,
    }
    for c, name in enumerate(COMP):
        weights[f"k1 {name:5s} w=J*g^cc"] = Jg * gig[..., c, c]
    for c, name in enumerate(COMP):
        weights[f"k2 {name:5s} w=g_cc/J"] = gg[..., c, c] / Jg
    # radial masks (natural order axis 0 = rho): per-element and per-quad-point
    ne0, q0 = elems[0][0], elems[0][1]
    rho_elem_mean = np.asarray(seq.quad.x_x).reshape(ne0, q0).mean(axis=1)
    rad_elem_keep = rho_elem_mean >= rho_cut
    rad_pt_keep = np.asarray(seq.quad.x_x) >= rho_cut
    print("per-cell CoV = std/|mean| of the weight within each element "
          "(0 => perfectly lumpable), inner rings excluded:")
    for label, W in weights.items():
        cov = percell_cov(seq, np.asarray(W), elems)[rad_elem_keep]     # drop inner radial cells
        rng = np.abs(np.asarray(W)[rad_pt_keep]).ravel()               # drop inner radial quad pts
        rng = rng[np.isfinite(rng) & (rng > 0)]
        dyn = (rng.max() / rng.min()) if rng.size else float("nan")
        print(f"    {label:20s} CoV {stats(cov)}   |  bulk max/min={dyn:.2e}")

    # ---- (B2) geometric-mean-of-endpoints error over the overlap stencil -------
    print("\n--- (B2) GEOMETRIC-MEAN-OF-ENDPOINTS error  sqrt(w_i w_j) vs true avg ---")
    grev, Wg = greville_grid_weights(seq, r_drop=r_drop)
    p = int(seq.basis_0.Λ[0].p)
    print(f"    overlap band = p = {p} knot spans; lag=p is the farthest-apart "
          f"overlapping pair.")
    for a, name in enumerate(COMP):
        sep = np.asarray(grev[a])
        sep = (sep[p:] - sep[:-p]) if sep.size > p else np.array([np.nan])
        print(f"    Greville separation at lag=p on axis {name:5s}: "
              f"max={np.nanmax(sep):.3f} (logical units, ~{p} cells)")
    print("    err = |sqrt(w_i w_j) - windowmean| / windowmean, per axis @ lag=1 and lag=p:")
    for label, W in Wg.items():
        print(f"    {label:20s}")
        for a, name in enumerate(COMP):
            e1 = geomean_endpoint_error(W, a, 1)
            ep = geomean_endpoint_error(W, a, p)
            print(f"        axis {name:5s}: lag1 {q3(e1):38s}  lagp {q3(ep)}")

    # ---- (C) CG iteration counts for the mass inversions -----------------------
    if args.no_cg:
        print("\n(--no-cg) skipping CG section.")
        return
    print("\n--- (C) CG ITERATIONS for mass inversion  M_k x = b  (production path) ---")
    ops = seq.get_operators()
    ops = assemble_mass_surgery_preconditioner(seq, operators=ops, ks=(0, 1, 2))
    ops = assemble_mass_jacobi_preconditioner(seq, operators=ops, ks=(0, 1, 2, 3))
    ops = assemble_tensor_mass_preconditioner(seq, operators=ops, ks=(0, 1, 2, 3),
                                              cp_kwargs={"greville": True})
    print(f"\n{'k':>2} {'bc':5} {'n':>7} {'jac_it':>7} {'grev_it':>8} {'conv':>5} {'grev_res':>10}", flush=True)
    for k in (0, 1, 2, 3):
        for dirichlet in (True, False):
            bc = "dbc" if dirichlet else "free"
            n = dof(seq, k, dirichlet)
            key = jax.random.PRNGKey(args.seed + k + (0 if dirichlet else 100))
            x_true = jax.random.normal(key, (n,), dtype=jnp.float64)
            rhs = apply_mass_matrix(seq, ops, x_true, k, dirichlet=dirichlet)
            try:
                xg, info_g = apply_inverse_mass_matrix(
                    seq, ops, rhs, k, dirichlet=dirichlet, tol=args.tol,
                    maxiter=args.maxiter, preconditioner="tensor", return_info=True)
                rg = jnp.linalg.norm(apply_mass_matrix(seq, ops, xg, k, dirichlet=dirichlet) - rhs)
                rg = float(rg / jnp.maximum(jnp.linalg.norm(rhs), 1e-30))
                itg, conv = it_conv(info_g)
            except Exception as exc:
                print(f"{k:>2} {bc:5} {n:>7d}  grev ERROR: {repr(exc)[:90]}", flush=True)
                continue
            try:
                _, info_j = apply_inverse_mass_matrix(
                    seq, ops, rhs, k, dirichlet=dirichlet, tol=args.tol,
                    maxiter=args.maxiter, preconditioner="jacobi", return_info=True)
                itj, _ = it_conv(info_j)
            except Exception:
                itj = -1
            print(f"{k:>2} {bc:5} {n:>7d} {itj:>7d} {itg:>8d} {str(conv):>5} {rg:>10.2e}", flush=True)


if __name__ == "__main__":
    main()
