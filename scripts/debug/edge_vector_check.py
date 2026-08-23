"""What IS the boundary functional the natural-BC term penalises?

At k=1 the radial factor of `u_r` is the DERIVATIVE basis `basis_0.dΛ[0]`, so
the trace `u_r(1)` is `sum_i c_i dLam_i(1)` -- and the question is how many
basis functions carry it. For a clamped space the derivative basis is a scaled
degree-(p-1) B-spline basis, so exactly ONE should be nonzero at the endpoint
and the trace is a single DOF. Two things make that worth measuring:

* `DerivativeSpline.evaluate` documents "derivative splines cannot be evaluated
  at a clamped boundary", and `_edge_vector` evaluates at `1 - 1e-8`;
* the clamped branch is `s(x, i+1)`, indexing a basis built with `n-1`
  functions, so the LAST derivative spline sits at the top of that shift.

It also reports the penalty strength `alpha e e^T` actually adds to the last
diagonal of the radial stiffness, which is what "penalise it harder" means
quantitatively -- and how far x1e4 (the refuted hard limit) really is.
"""
from __future__ import annotations

import argparse
import os
import sys

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from block_jacobi_spectrum import build_sequence  # noqa: E402
from mrx.block_jacobi_laplacian import (  # noqa: E402
    _edge_vector, _face_metric_scalar, _mesh_amplification, component_factors,
    trace_components)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry", default="toroid",
                    choices=("toroid", "rot-ellipse", "w7x"))
    ap.add_argument("--ns", default="8,16,8")
    ap.add_argument("--p", type=int, default=3)
    cli = ap.parse_args()

    ns = tuple(int(v) for v in cli.ns.split(","))
    seq, ops = build_sequence(cli.geometry, ns, cli.p)
    print(f"geometry={cli.geometry} ns={ns} p={cli.p}", flush=True)

    e = np.asarray(_edge_vector(seq, 0, None))
    big = np.abs(e) > 1e-12 * (np.abs(e).max() + 1e-300)
    print(f"\nradial edge vector e = dLam(1), n = {e.size}", flush=True)
    print(f"  nonzeros: {int(big.sum())} at indices "
          f"{np.flatnonzero(big).tolist()}", flush=True)
    print(f"  values:   {np.array2string(e[big], precision=6)}", flush=True)
    print(f"  |e_2nd|/|e_last| = "
          f"{(np.sort(np.abs(e))[-2] / np.abs(e).max()):.3e}", flush=True)
    print("  (one nonzero => u_r(1) is a SINGLE dof and the penalty is a "
          "clean rank-one on that dof)", flush=True)

    mu0 = _mesh_amplification(seq)
    print(f"\nmesh amplification mu_0 = {mu0:.6e}", flush=True)

    print(f"\n{'k':>2} {'c':>2} {'alpha':>12} {'alpha e_l^2':>13} "
          f"{'K_r[-1,-1]':>13} {'ratio':>10}", flush=True)
    for k in (1, 2, 3):
        for c in trace_components(k):
            shapes = getattr(seq, f"basis_{k}").shape
            if c >= len(shapes):
                continue
            masses, stiffs, _ = component_factors(
                seq, k, c, window=None, ktilde_mode="honest", lumped="diag",
                bc_entry=False, dirichlet=False)
            k_r = np.asarray(stiffs[0])
            alpha = _face_metric_scalar(seq, k, c, "diag") * mu0
            add = alpha * float(e[np.argmax(np.abs(e))]) ** 2
            diag = float(k_r[-1, -1])
            print(f"{k:>2} {c:>2} {alpha:>12.4e} {add:>13.4e} "
                  f"{diag:>13.4e} {add / diag:>10.3f}", flush=True)
    print("\nratio = how much the natural term stiffens the boundary row "
          "relative to the stiffness already there.", flush=True)


if __name__ == "__main__":
    main()
