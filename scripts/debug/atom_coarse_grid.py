"""At which (n, p) does the block-Jacobi atom stop building, and why?

The Poisson convergence study starts at n=4, and repointing it off the retired
tensor stack onto the production atom made it die there:

    component_factors -> np.linalg.eigvals -> "Array must not contain infs or NaNs"

`component_factors` forms ``A^-1 M`` per axis with ``A`` a 1-D mass weighted by
the stiffness profile, and takes the mean eigenvalue. This walks n upward and
reports, per axis, the condition of ``A`` and whether the solve stays finite --
so the answer is "the smallest n that works", not a guess.
"""
from __future__ import annotations

import argparse
import os
import sys



import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mrx.operators as op  # noqa: E402
from mrx.derham_sequence import DeRhamSequence  # noqa: E402
from mrx.mappings import toroid_map  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ns", default="4,5,6,8,10,12")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--ks", default="0,1,2,3")
    cli = ap.parse_args()

    for n in [int(v) for v in cli.ns.split(",")]:
        seq = DeRhamSequence((n, 2 * n, n), (cli.p,) * 3, 2 * cli.p,
                             ("clamped", "periodic", "periodic"), polar=True,
                             betti_numbers=(1, 1, 0, 0))
        seq.evaluate_1d()
        seq.set_map(toroid_map(epsilon=1 / 3, R0=1.0))
        ops = op.assemble_incidence_operators(seq)
        ops = op.assemble_mass_jacobi_preconditioner(seq, ops, ks=(0, 1, 2, 3))
        seq.set_operators(ops)

        jac = np.asarray(seq.geometry.jacobian_j)
        line = (f"[n={n:3d} p={cli.p}] jac in [{jac.min():.3e}, {jac.max():.3e}]"
                f" finite={np.isfinite(jac).all()}")
        for k in [int(v) for v in cli.ks.split(",")]:
            for dbc in (True, False):
                try:
                    op.assemble_metric_lumping_laplacian_preconditioner(
                        seq, ops, ks=(k,), dirichlets=(dbc,))
                    tag = "ok"
                except Exception as exc:                      # noqa: BLE001
                    tag = type(exc).__name__
                line += f"  k{k}{'d' if dbc else 'f'}={tag}"
        print(line, flush=True)


if __name__ == "__main__":
    main()
