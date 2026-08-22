"""How much component coupling does block-Jacobi throw away, and where?

The natural-BC coefficient is settled (docs/research/natural_bc_coefficient_handoff.md)
and it wins at k=0 and k=3 -- the degrees with ONE component -- while k=1 and
k=2 free want it damped, more so the more shaped the geometry. The hypothesis:
block-Jacobi is block-diagonal in the COMPONENT index, so at k=1/2 it discards
the inter-component terms; under a free BC those are largest at r=1, exactly
where the boundary term acts.

That coupling is not an operator property that needs probing. Both masses are
local, and their off-diagonal component blocks are weighted by the OFF-DIAGONAL
metric:

    M_1 off-diagonal (a,b)  ~  g^{ab} J        (1-forms)
    M_2 off-diagonal (a,b)  ~  g_{ab} / J      (2-forms)

so the dimensionless strength is the metric correlation coefficient

    corr^{ab} = |g^{ab}| / sqrt(g^{aa} g^{bb})       in [0, 1]

evaluated pointwise and averaged over theta,zeta. NO probes, no solves, no
operator applications -- pure quadrature. Prints the r=1 face value (what the
free BC sees) next to the volume average (what the bulk sees).

PREDICTION under test: the face coupling should order
toroid << rot-ellipse < W7-X, matching the damping each geometry wants at
k=1/2 free (none / 0.16 / 0.072). The toroid map is orthogonal, so its face
coupling must come out at ~0 -- a built-in control.
"""
from __future__ import annotations

import argparse
import os
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mrx  # noqa: E402
from mrx.derham_sequence import DeRhamSequence  # noqa: E402
from mrx.experimental.block_jacobi_laplacian import (  # noqa: E402
    _reshape_quadrature_matrix_field)
from mrx.mappings import rotating_ellipse_map, toroid_map  # noqa: E402

mrx.MAP_BATCH_SIZE_INNER = int(os.environ.get("W7X_MAP_BATCH", "256"))

PAIRS = ((0, 1, "r-t"), (0, 2, "r-z"), (1, 2, "t-z"))


def build_sequence(geometry, ns, p):
    seq = DeRhamSequence(ns, (p,) * 3, 2 * p, ("clamped", "periodic", "periodic"),
                         polar=True, tol=1e-12, maxiter=1000,
                         betti_numbers=(1, 1, 0, 0))
    seq.evaluate_1d()
    if geometry == "toroid":
        seq.set_map(toroid_map(epsilon=1 / 3, R0=1.0))
    elif geometry == "rot-ellipse":
        seq.set_map(rotating_ellipse_map(eps=0.33, kappa=1.5, nfp=3))
    else:
        from w7x_geometry import build_w7x_map  # noqa: PLC0415
        map_func, _ = build_w7x_map(map_ns=ns, p=p)
        seq.set_map(map_func)
    return seq


def corr_field(m):
    """|m_ab| / sqrt(m_aa m_bb) per pair, as (nx, ny, nz) fields."""
    out = {}
    for a, b, name in PAIRS:
        den = jnp.sqrt(jnp.abs(m[..., a, a] * m[..., b, b]))
        out[name] = jnp.abs(m[..., a, b]) / jnp.maximum(den, 1e-300)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry", default="toroid",
                    choices=("toroid", "rot-ellipse", "w7x"))
    ap.add_argument("--ns", default="12,24,12")
    ap.add_argument("--p", type=int, default=3)
    cli = ap.parse_args()

    ns = tuple(int(v) for v in cli.ns.split(","))
    seq = build_sequence(cli.geometry, ns, cli.p)

    ginv = jnp.transpose(_reshape_quadrature_matrix_field(
        seq, seq.geometry.metric_inv_jkl), (1, 0, 2, 3, 4))
    met = jnp.transpose(_reshape_quadrature_matrix_field(
        seq, seq.geometry.metric_jkl), (1, 0, 2, 3, 4))

    wx, wy, wz = seq.quad.w_x, seq.quad.w_y, seq.quad.w_z
    sy, sz = jnp.sum(wy), jnp.sum(wz)

    def face(f):                       # <.> over theta,zeta at the LAST r slice
        return float(jnp.einsum('rs,r,s->', f[-1], wy, wz) / (sy * sz))

    def vol(f):
        return float(jnp.einsum('qrs,q,r,s->', f, wx, wy, wz)
                     / (jnp.sum(wx) * sy * sz))

    print(f"geometry={cli.geometry} ns={ns} p={cli.p}\n", flush=True)
    print("component coupling  |g_ab| / sqrt(g_aa g_bb)   (0 = decoupled)\n",
          flush=True)
    hdr = f"{'space':>6} {'pair':>5} {'FACE r=1':>11} {'volume':>11} {'face/vol':>9}"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    summary = {}
    for label, m in (("k=1", ginv), ("k=2", met)):
        cf = corr_field(m)
        worst = 0.0
        for _, _, name in PAIRS:
            fv, vv = face(cf[name]), vol(cf[name])
            worst = max(worst, fv)
            print(f"{label:>6} {name:>5} {fv:>11.4f} {vv:>11.4f} "
                  f"{fv / max(vv, 1e-300):>9.2f}", flush=True)
        summary[label] = worst
        print(flush=True)
    print(f"WORST face coupling: k=1 {summary['k=1']:.4f}   "
          f"k=2 {summary['k=2']:.4f}", flush=True)

    # The quantity that would give a CLOSED-FORM damping factor.
    #
    # The k=1 boundary pairing is (E u)_j = oint (sum_b g^{rb} J u_b) tau_j, so
    # the component block is w w^T with w_b = g^{rb} J -- rank one, penalising
    # the single direction w. The atom keeps only the (r,r) entry, i.e. it
    # penalises e_r instead. cos^2 phi = w_r^2/|w|^2 is the fraction of the true
    # penalty lying along the direction the atom actually uses.
    #
    # Compare against the empirically optimal damping at k=1/2 free:
    #   toroid ~1.0    rot-ellipse ~0.16    W7-X ~0.072
    # If cos^2 phi tracks those, it is a closed form and needs no fit.
    wr = ginv[..., 0, 0]
    w2 = sum(ginv[..., 0, b] ** 2 for b in range(3))
    cos2 = wr ** 2 / jnp.maximum(w2, 1e-300)
    print(f"\n{'cos^2 phi (k=1)':>22}  face {face(cos2):.4f}   "
          f"volume {vol(cos2):.4f}", flush=True)
    for b, nm in ((1, "t"), (2, "z")):
        rat = jnp.abs(ginv[..., 0, b]) / jnp.maximum(jnp.abs(wr), 1e-300)
        print(f"{'|w_' + nm + '| / |w_r|':>22}  face {face(rat):.4f}   "
              f"volume {vol(rat):.4f}", flush=True)

    # k=2 uses a DIFFERENT metric row per component: the trace is
    # oint [tau_t g_{zb} u^b - tau_z g_{tb} u^b]/J, so the tangential component
    # c carries the row g_{cb}, NOT g_{rb}. (k=3's weight is 1/J with no metric
    # row at all, so its tilt is exactly 1 -- no coupling is possible, matching
    # the single component.)
    for c, nm in ((1, "theta"), (2, "zeta")):
        mc = met[..., c, c]
        m2 = sum(met[..., c, b] ** 2 for b in range(3))
        cc = mc ** 2 / jnp.maximum(m2, 1e-300)
        print(f"{'cos^2 phi (k=2 c=' + nm + ')':>22}  face {face(cc):.4f}   "
              f"volume {vol(cc):.4f}", flush=True)
        for b in range(3):
            if b == c:
                continue
            rat = jnp.abs(met[..., c, b]) / jnp.maximum(jnp.abs(mc), 1e-300)
            print(f"{'  |w_' + 'rtz'[b] + '|/|w_' + 'rtz'[c] + '|':>22}  "
                  f"face {face(rat):.4f}   volume {vol(rat):.4f}", flush=True)
    print("\nPrediction: this orders toroid << rot-ellipse < W7-X, matching the "
          "damping\nwanted at k=1/2 free (none / 0.16 / 0.072). Toroid must be "
          "~0 (orthogonal map).", flush=True)


if __name__ == "__main__":
    main()
