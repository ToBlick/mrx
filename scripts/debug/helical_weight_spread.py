"""Gating diagnostic for the helically-sheared smoother atom (fdhel).

The fdbund atom averages the bundled weights W_a = g^{aa} J over the cross
axes; the Laplacian radial-profile diagnostic showed the resulting angular
spread is what bounds the atom (cyl 0% / toroid 24% / W7-X 60% ~ lam_max
1 / 2.2 / 3.7). On W7-X the angular non-separability is hypothesised to be
mostly a helical ROTATION of the cross-section: W(theta, zeta) ~
W(theta - rho*zeta).  If true, the weights are nearly constant along helical
lines and a shear-conjugated separable atom (fdhel) recovers them.

This script measures, per channel a in (rr, tt, zz), the relative spread of
the radially-reduced weight W_a(theta, zeta) = <g^{aa} J>_{r in [xi_1, 1]}
along toroidal lines (pitch rho = 0, what fdbund assumes constant) vs along
helical lines theta - rho*zeta = const, scanning rho in fractions of a full
poloidal turn per logical zeta period.  Spread metric: quad-weighted mean
over lines of (weighted std / weighted mean along the line).

Verdict: min-over-rho helical spread << toroidal spread  =>  fdhel is worth
wiring (expected rho* ~ +-0.5 for l=2 stellarator symmetry: half a poloidal
turn per field period).
"""
import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "benchmark"))

from benchmark_graddiv_k1_preconditioner import build_sequence  # noqa: E402
from mrx.operators import (  # noqa: E402
    _reshape_quadrature_matrix_field,
    _reshape_quadrature_scalar_field,
)
import jax.numpy as jnp  # noqa: E402


def reduced_weights(seq, ring0=1):
    """W_a(q_theta, q_zeta): bundled g^{aa}J, quad-mean over r in [xi_1, 1]."""
    minv = jnp.transpose(
        _reshape_quadrature_matrix_field(seq, seq.geometry.metric_inv_jkl),
        (1, 0, 2, 3, 4))
    jacq = jnp.transpose(
        _reshape_quadrature_scalar_field(seq, seq.geometry.jacobian_j), (1, 0, 2))
    p_r = seq.ps[0]
    xi1 = jnp.asarray(seq.basis_0.Λ[0].T)[p_r + 1 + max(ring0, 0)]
    wx_cut = seq.quad.w_x * (jnp.asarray(seq.quad.x_x) >= xi1)
    sxc = jnp.sum(wx_cut)
    out = []
    for a in range(3):
        w = minv[..., a, a] * jacq
        out.append(np.asarray(jnp.einsum('qrs,q->rs', w, wx_cut) / sxc))
    return out  # each (n_qtheta, n_qzeta)


def line_spread(W, tq, zq, wt, wz, rho):
    """Weighted mean over helical lines (label = theta at zeta=0) of
    std/mean along the line, for pitch rho (theta turns per zeta period)."""
    tq = np.asarray(tq); zq = np.asarray(zq)
    n_t = tq.size
    t_ext = np.concatenate([tq - 1.0, tq, tq + 1.0])
    lines = np.empty_like(W)
    for k in range(zq.size):
        col_ext = np.concatenate([W[:, k]] * 3)
        # value on the helix labelled tq[j]: sample W(theta = label + rho*zeta)
        lines[:, k] = np.interp((tq + rho * zq[k]) % 1.0, t_ext, col_ext)
    wz_n = wz / wz.sum()
    mean = lines @ wz_n
    var = ((lines - mean[:, None]) ** 2) @ wz_n
    rel = np.sqrt(var) / np.abs(mean)
    return float((wt / wt.sum()) @ rel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", default="w7x")
    ap.add_argument("--ns", type=int, nargs=3, default=[12, 24, 24])
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--epsilon", type=float, default=1.0 / 3.0)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--r0", type=float, default=1.0)
    ap.add_argument("--nfp", type=int, default=5)
    ap.add_argument("--r-scale", type=float, default=0.5)
    ap.add_argument("--ring0", type=int, default=1)
    ap.add_argument("--rho-max", type=float, default=1.0)
    ap.add_argument("--rho-step", type=float, default=0.025)
    args = ap.parse_args()

    cfg = SimpleNamespace(ns=tuple(args.ns), p=args.p, geometry=args.geometry,
                          cg_tol=1e-10, cg_maxiter=10, epsilon=args.epsilon,
                          kappa=args.kappa, alpha=args.alpha, r0=args.r0,
                          nfp=args.nfp, r_scale=args.r_scale, polar_order=1,
                          polar_ring1=None)
    seq = build_sequence(cfg)
    Ws = reduced_weights(seq, ring0=args.ring0)
    tq, zq = np.asarray(seq.quad.x_y), np.asarray(seq.quad.x_z)
    wt, wz = np.asarray(seq.quad.w_y), np.asarray(seq.quad.w_z)

    rhos = np.arange(-args.rho_max, args.rho_max + 1e-9, args.rho_step)
    names = ("rr", "tt", "zz")
    print(f"=== helical weight spread  {args.geometry} ns={tuple(args.ns)} "
          f"p={args.p} nfp={args.nfp} kappa={args.kappa} alpha={args.alpha} "
          f"ring0={args.ring0} ===", flush=True)
    curves = {}
    for name, W in zip(names, Ws):
        s = np.array([line_spread(W, tq, zq, wt, wz, r) for r in rhos])
        curves[name] = s
        i = int(np.argmin(s))
        print(f"[{name}] toroidal(rho=0) spread = {s[np.argmin(np.abs(rhos))]:.4f}   "
              f"min helical = {s[i]:.4f} at rho = {rhos[i]:+.3f}   "
              f"reduction x{s[np.argmin(np.abs(rhos))] / max(s[i], 1e-12):.2f}",
              flush=True)
    print("rho curve (tt channel):")
    for r, v in zip(rhos, curves["tt"]):
        print(f"  rho={r:+.3f}  spread={v:.4f}")


if __name__ == "__main__":
    main()
