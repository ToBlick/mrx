"""Verify the pieces, THEN the local Nitsche trace-constant eigenproblem.

Two earlier attempts at a global (whole-axis) shortcut failed:
  1. `K_r` under a natural BC is singular and a plain solve returned garbage;
  2. the pseudoinverse fix rested on `e ⊥ null(K_r)`, which is FALSE -- the
     diagnostic reported 12% of `e` in the discarded mode.

The false step: `e` is NOT the derivative of a partition of unity. It is the
DERIVATIVE-SPLINE basis `D` evaluated at r=1 (`_edge_vector` reads
`seq.basis_0.dLambda`), and those are unit-INTEGRAL normalised, so their values
at the boundary do not sum to zero.

So this script verifies before it derives. PART 1 prints the pieces with no
algebra on top. PART 2 attempts the local eigenproblem the literature actually
poses -- on the boundary ELEMENT, not the whole axis -- and refuses to report a
constant unless `e` lies in the range of the local operator, because if it does
not the Rayleigh quotient is unbounded and no finite trace constant exists.

    python scripts/debug/nitsche_local.py --geometry w7x --p 3
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
    _assemble_weighted_1d_mass, _axis_bases, _edge_vector, _face_alpha,
    _h_last, bundled_axis_profiles, component_factors, trace_components,
    weight_fields,
)
from verify_block_jacobi import build_sequence  # noqa: E402


def spectrum(A, tol=1e-10):
    w, V = np.linalg.eigh(0.5 * (A + A.T))
    keep = w > tol * max(w.max(), 1e-300)
    return w, V, keep


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry", default="toroid")
    ap.add_argument("--ns", default="12,24,12")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--out", default=None)
    cli = ap.parse_args()
    ns = tuple(int(v) for v in cli.ns.split(","))
    seq, _ = build_sequence(cli.geometry, ns, cli.p, 1000)
    res = {"geometry": cli.geometry, "p": cli.p, "rows": []}

    # ---------------- PART 1: the pieces, no algebra ----------------------
    e = np.asarray(_edge_vector(seq, 0, None))
    h = _h_last(seq)
    print(f"\n===== PART 1: pieces  ({cli.geometry} p={cli.p}) =====", flush=True)
    print(f"  e = dLambda_r(1):  len={e.shape[0]}  ||e||={np.linalg.norm(e):.6e}",
          flush=True)
    print(f"     sum(e) = {e.sum():.6e}   <-- 0 only if e were d/dr of a "
          f"partition of unity; it is NOT", flush=True)
    print(f"     nonzeros: {int((np.abs(e) > 1e-12 * np.abs(e).max()).sum())}"
          f"   last three: {e[-3:]}", flush=True)
    print(f"  h_last = {h:.6e}", flush=True)

    for k in (1, 2, 3):
        for c in trace_components(k):
            _, stiffs, _ = component_factors(
                seq, k, c, window=None, lumped="diag", bc_entry=False,
                dirichlet=False)
            K = np.asarray(stiffs[0])
            w, V, keep = spectrum(K)
            coeff = V.T @ e
            leak = np.linalg.norm(coeff[~keep]) / np.linalg.norm(coeff)
            print(f"  k={k} c={c}: K_r {K.shape}  trace={np.trace(K):.4e}  "
                  f"null={int((~keep).sum())}  |e in null|/|e| = {leak:.3e}",
                  flush=True)

    # ---------------- PART 2: the LOCAL eigenproblem ----------------------
    print("\n===== PART 2: local trace constant on the boundary element =====",
          flush=True)
    primal, deriv, quad_w = _axis_bases(seq)
    x = np.asarray(seq.quad.x_x)
    mask = jnp.asarray((x >= 1.0 - h - 1e-12).astype(float))
    print(f"  boundary element r in [{1 - h:.4f}, 1]: "
          f"{int(mask.sum())} of {x.size} radial quadrature points", flush=True)
    if not hasattr(seq, "_bj_dd_tables"):
        from mrx.local_assembly import _second_derivative_tables  # noqa: PLC0415
        seq._bj_dd_tables = _second_derivative_tables(seq)

    fields = weight_fields(seq)
    ginv, jac = fields["ginv_aa"], fields["jac"]
    print(f"  {'k':>2}{'c':>3}{'supp':>6}{'null_loc':>10}{'leak_loc':>11}"
          f"{'C_local':>13}{'a_nat/a_5':>12}", flush=True)
    for k in (1, 2, 3):
        for c in trace_components(k):
            m_k = {1: ginv[c] * jac, 2: fields["met_aa"][c] / jac,
                   3: 1.0 / jac}[k]
            prof = bundled_axis_profiles(seq, m_k * ginv[0])[0]
            # element-local stiffness: same assembly, quadrature masked to the
            # last element only
            Kloc = np.asarray(_assemble_weighted_1d_mass(
                seq._bj_dd_tables[0], quad_w[0] * prof * mask))
            supp = np.abs(np.diag(Kloc)) > 1e-12 * np.abs(np.diag(Kloc)).max()
            Ks = Kloc[np.ix_(supp, supp)]
            es = e[supp]
            w, V, keep = spectrum(Ks)
            coeff = V.T @ es
            leak = np.linalg.norm(coeff[~keep]) / max(np.linalg.norm(coeff), 1e-300)
            if leak > 1e-6:
                print(f"  {k:>2}{c:>3}{int(supp.sum()):>6}"
                      f"{int((~keep).sum()):>10}{leak:>11.2e}"
                      f"{'UNBOUNDED':>13}{'--':>12}", flush=True)
                res["rows"].append({"k": k, "c": c, "leak_local": float(leak),
                                    "unbounded": True})
                continue
            C = float(np.sum(coeff[keep] ** 2 / w[keep]))
            a_nat = 1.0 / C
            scalar, amp = _face_alpha(seq, k, c, "diag")
            a5 = scalar * amp
            print(f"  {k:>2}{c:>3}{int(supp.sum()):>6}{int((~keep).sum()):>10}"
                  f"{leak:>11.2e}{C:>13.5e}{a_nat / a5:>12.3f}", flush=True)
            res["rows"].append({"k": k, "c": c, "C_local": C,
                                "predict_s": a_nat / a5,
                                "leak_local": float(leak), "unbounded": False})

    if cli.out:
        os.makedirs(os.path.dirname(cli.out) or ".", exist_ok=True)
        with open(cli.out, "w") as f:
            json.dump(res, f, indent=1)
        print(f"\nwrote {cli.out}", flush=True)


if __name__ == "__main__":
    main()
