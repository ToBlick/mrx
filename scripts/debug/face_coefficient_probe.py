"""The natural-BC face coefficient, per (k, c), with no solves.

Reports the shipped penalty coefficient `alpha_k` from `_face_alpha` alongside
the exact face block's value `<J g^rr>/h`, which is (k,c)-INDEPENDENT by
construction (`m_k` cancels against `w_comp`). Two numbers matter:

  spread   max/min of alpha_k over (k,c) -- how degree-dependent the penalty is
           on this geometry, i.e. what a single scale has to absorb across k.
  ratio    alpha_k / (<J g^rr>/h) -- how far the penalty sits from the exact
           block. This is what the shipped scale has to make up, and it is
           GEOMETRY dependent: measured 0.0093 (quasr9983) to 0.042 (w7x), a
           factor of 4.5, against a fixed scale that can only match one of them.

Cheap: one geometry build and a face average, no operator applies.

    python scripts/debug/face_coefficient_probe.py --geometry w7x --p 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mrx.metric_lumping_laplacian import (  # noqa: E402
    _face_alpha, _h_last, trace_components, weight_fields,
)
from verify_block_jacobi import build_sequence  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry", default="toroid")
    ap.add_argument("--ns", default="12,24,12")
    ap.add_argument("--ps", default="2,3,5")
    ap.add_argument("--out", default=None)
    cli = ap.parse_args()
    ns = tuple(int(v) for v in cli.ns.split(","))
    res = {"geometry": cli.geometry, "ns": list(ns), "rows": []}

    for p in [int(v) for v in cli.ps.split(",")]:
        seq, _ = build_sequence(cli.geometry, ns, p, 1000)
        fields = weight_fields(seq)
        ginv, jac = fields["ginv_aa"], fields["jac"]
        wy, wz = seq.quad.w_y, seq.quad.w_z
        norm = jnp.sum(wy) * jnp.sum(wz)

        def fm(f):
            return float(jnp.einsum('rs,r,s->', jnp.asarray(f)[-1], wy, wz)
                         / norm)

        # the exact face block, (k,c)-independent: <J g^rr> / h
        exact = fm(jac * ginv[0]) / _h_last(seq)
        # spread of the two fields the penalty averages, as the predictor
        jf = np.asarray(jac)[-1]
        gf = np.asarray(ginv[0])[-1]
        alphas = {}
        for k in (1, 2, 3):
            for c in trace_components(k):
                scalar, amp = _face_alpha(seq, k, c, "diag")
                alphas[(k, c)] = scalar * amp
        vals = list(alphas.values())
        spread = max(vals) / min(vals)
        print(f"\n=== {cli.geometry} p={p} ===", flush=True)
        print(f"  exact face block <J g^rr>/h = {exact:.6e}", flush=True)
        print(f"  J on face  [{jf.min():.3e}, {jf.max():.3e}]  "
              f"spread {jf.max()/jf.min():.3f}", flush=True)
        print(f"  g^rr       [{gf.min():.3e}, {gf.max():.3e}]  "
              f"spread {gf.max()/gf.min():.3f}", flush=True)
        print(f"  {'k':>2}{'c':>3}{'alpha_k':>14}{'/ exact':>12}"
              f"{'/ alpha_(1,*)':>15}", flush=True)
        base = alphas[sorted(alphas)[0]]
        for key in sorted(alphas):
            a = alphas[key]
            print(f"  {key[0]:>2}{key[1]:>3}{a:>14.6e}{a/exact:>12.5f}"
                  f"{a/base:>15.4f}", flush=True)
        print(f"  spread over (k,c) = {spread:.4f}", flush=True)
        res["rows"].append({
            "p": p, "exact": exact, "spread": spread,
            "J_spread": float(jf.max() / jf.min()),
            "grr_spread": float(gf.max() / gf.min()),
            "alpha": {f"{k},{c}": v for (k, c), v in alphas.items()},
            "ratio": {f"{k},{c}": v / exact for (k, c), v in alphas.items()},
        })

    if cli.out:
        os.makedirs(os.path.dirname(cli.out) or ".", exist_ok=True)
        with open(cli.out, "w") as f:
            json.dump(res, f, indent=1)
        print(f"\nwrote {cli.out}", flush=True)


if __name__ == "__main__":
    main()
