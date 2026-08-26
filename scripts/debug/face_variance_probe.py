"""Does the natural-BC scale track the VARIANCE of the face weight?

The rank-one boundary term keeps only the MEAN of the face weight and drops its
angular variation, so the fitted scale is compensating for exactly what the mean
discards -- the variance. `J sqrt(g^rr)` is the natural candidate: it is the
surface element density itself, `dsigma = J sqrt(g^rr) dtheta dzeta`.

This tests that against the alternatives, all dimensionless so they can be
compared with a scale:

    relvar(f)  = <f^2>/<f>^2 - 1        zero iff f is constant on the face
    amhm(f)    = <f> <1/f>              1 iff f is constant (arithmetic/harmonic)
    span(f)    = max f / min f

for f in {J sqrt(g^rr), J g^rr, sqrt(g^rr), J, m_k sqrt(g^rr)}.

No solves. Compare the output against the measured k=3 optima:
    hegna 2.37 | w7x 5.37 | toroid 9.64 | rot-ell 9.05 | quasr44970 19.60 |
    quasr9983 22.63

    python scripts/debug/face_variance_probe.py --geometry w7x
"""
from __future__ import annotations

import argparse
import json
import os
import sys



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
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--out", default=None)
    cli = ap.parse_args()
    ns = tuple(int(v) for v in cli.ns.split(","))
    seq, _ = build_sequence(cli.geometry, ns, cli.p, 1000)

    fields = weight_fields(seq)
    ginv, met, jac = fields["ginv_aa"], fields["met_aa"], fields["jac"]
    wy, wz = np.asarray(seq.quad.w_y), np.asarray(seq.quad.w_z)
    W = np.outer(wy, wz)
    W = W / W.sum()                                # normalised face measure

    def face(f):
        return np.asarray(jnp.asarray(f))[-1]      # r = 1 slice, (qy, qz)

    def mean(f):
        return float((W * f).sum())

    def stats(f):
        m = mean(f)
        return {"mean": m,
                "relvar": mean(f * f) / m ** 2 - 1.0,
                "amhm": m * mean(1.0 / f),
                "span": float(f.max() / f.min())}

    cand = {
        "J*sqrt(grr)": face(jac * jnp.sqrt(ginv[0])),   # the surface element
        "J*grr": face(jac * ginv[0]),                   # the exact face block
        "sqrt(grr)": face(jnp.sqrt(ginv[0])),
        "J": face(jac),
    }
    res = {"geometry": cli.geometry, "p": cli.p, "fields": {}, "alpha": {}}
    print(f"\n=== {cli.geometry} p={cli.p} ===", flush=True)
    print(f"  {'field':<14}{'mean':>13}{'relvar':>12}{'amhm':>10}{'span':>10}",
          flush=True)
    for name, f in cand.items():
        st = stats(f)
        print(f"  {name:<14}{st['mean']:>13.5e}{st['relvar']:>12.4f}"
              f"{st['amhm']:>10.4f}{st['span']:>10.3f}", flush=True)
        res["fields"][name] = st

    # the per-degree integrand the penalty actually averages
    print(f"\n  {'k':>2}{'c':>3}{'relvar(m_k sqrt(grr))':>24}{'amhm':>10}"
          f"{'alpha_5':>13}", flush=True)
    for k in (1, 2, 3):
        for c in trace_components(k):
            m_k = {1: ginv[c] * jac, 2: met[c] / jac, 3: 1.0 / jac}[k]
            f = face(m_k * jnp.sqrt(ginv[0]))
            st = stats(f)
            scalar, amp = _face_alpha(seq, k, c, "diag")
            print(f"  {k:>2}{c:>3}{st['relvar']:>24.4f}{st['amhm']:>10.4f}"
                  f"{scalar * amp:>13.5e}", flush=True)
            res["alpha"][f"{k},{c}"] = {"alpha_5": scalar * amp, **st}
    res["h_last"] = _h_last(seq)

    if cli.out:
        os.makedirs(os.path.dirname(cli.out) or ".", exist_ok=True)
        with open(cli.out, "w") as f:
            json.dump(res, f, indent=1)
        print(f"\nwrote {cli.out}", flush=True)


if __name__ == "__main__":
    main()
