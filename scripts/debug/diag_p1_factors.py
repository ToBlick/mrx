"""Why the block-Jacobi atom collapses at p = 1.

The atom's factor on a DERIVATIVE axis is the weak block's radial round trip
``F = M^d G A^-1 G^T M^d`` (:func:`_ktilde_1d`).  At ``p >= 2`` that is modelled
by the honest stiffness of the derivative splines; at ``p = 1`` the derivative
splines are degree 0 and have no stiffness, so a DG-0 jump form stands in
(:func:`_fd_stiffness_degree0`).  This script compares all three against the
round trip in the generalized spectrum ``K v = lam M v`` -- the only thing the
fast diagonalisation sees -- so a pure mis-scale (a constant ratio) is
distinguishable from a structurally wrong operator (a ratio that drifts with
the mode index).

Usage:
    python scripts/debug/diag_p1_factors.py --geometry toroid --ns 8,16,8 --ps 1,2,3
"""
from __future__ import annotations

import argparse
import os
import sys



import numpy as np  # noqa: E402
import scipy.linalg as sla  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mrx.metric_lumping_laplacian import (  # noqa: E402
    _axis_bases, _fd_stiffness_degree0, _ktilde_1d, bundled_axis_profiles,
    component_factors, weight_fields)
from mrx.operators import _assemble_weighted_1d_mass  # noqa: E402
from verify_block_jacobi import build_sequence  # noqa: E402

AXES = "rtz"


def gen_eigs(k, m):
    k = np.asarray(k, dtype=float)
    m = np.asarray(m, dtype=float)
    return np.sort(np.real(sla.eigh(0.5 * (k + k.T), 0.5 * (m + m.T),
                                    eigvals_only=True)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry", default="toroid", choices=("toroid", "w7x"))
    ap.add_argument("--ns", default="8,16,8")
    ap.add_argument("--ps", default="1,2,3")
    cli = ap.parse_args()
    ns = tuple(int(v) for v in cli.ns.split(","))

    for p in (int(v) for v in cli.ps.split(",")):
        seq, ops = build_sequence(cli.geometry, ns, p, 2000)
        primal, deriv, quad_w = _axis_bases(seq)
        fields = weight_fields(seq)
        ginv, jac = fields["ginv_aa"], fields["jac"]
        primal_prof = bundled_axis_profiles(seq, jac)
        degree0 = int(seq.basis_0.Λ[0].p) < 2
        print(f"\n===== p={p} ns={ns} geometry={cli.geometry} "
              f"degree0={degree0}", flush=True)

        for k in (1, 2, 3):
            for c in range(3):
                das = ((c,) if k == 1 else
                       (0, 1, 2) if k == 3 else
                       tuple(a for a in range(3) if a != c))
                for a in das:
                    # Same profiles component_factors uses with lumped="diag".
                    prof = bundled_axis_profiles(seq, ginv[a] * jac)[a]
                    m = np.asarray(_assemble_weighted_1d_mass(
                        deriv[a], quad_w[a]))
                    rt = np.asarray(_ktilde_1d(seq, a, m, primal_prof[a]))
                    lam_rt = gen_eigs(rt, m)
                    if degree0:
                        # The "coef" form (the pre-fix behaviour, under-scaled
                        # by h^2) is gone with its knob -- the value form is
                        # landed and is now the only one. See the memory note
                        # p1-degree0-coefficient-vs-value for what it cost.
                        alt = np.asarray(_fd_stiffness_degree0(seq, a, prof))
                        old = alt
                        lam_old = gen_eigs(old, m)
                        nz0 = lam_rt > 1e-12 * max(lam_rt.max(), 1e-300)
                        r0 = lam_old[nz0] / lam_rt[nz0]
                        print(f"   [coef-basis jump] lam "
                              f"[{lam_old[0]:9.2e},{lam_old[-1]:9.2e}] "
                              f"ratio lo/med/hi {r0[0]:8.2e} "
                              f"{np.median(r0):8.2e} {r0[-1]:8.2e}", flush=True)
                        label = "jump/h2"
                    else:
                        from mrx.local_assembly import (  # noqa: PLC0415
                            _second_derivative_tables)
                        if not hasattr(seq, "_bj_dd_tables"):
                            seq._bj_dd_tables = _second_derivative_tables(seq)
                        alt = np.asarray(_assemble_weighted_1d_mass(
                            seq._bj_dd_tables[a], quad_w[a] * prof))
                        label = "honest"
                    lam_alt = gen_eigs(alt, m)
                    # The atom sees K/M; a constant ratio is a scale bug, a
                    # drifting one is a different operator.
                    nz = lam_rt > 1e-12 * max(lam_rt.max(), 1e-300)
                    ratio = lam_alt[nz] / lam_rt[nz]
                    print(f" k={k} c={AXES[c]} axis={AXES[a]:>1} n={m.shape[0]:>3} "
                          f"| rt lam [{lam_rt[0]:9.2e},{lam_rt[-1]:9.2e}] "
                          f"| {label:>6} lam [{lam_alt[0]:9.2e},{lam_alt[-1]:9.2e}] "
                          f"| ratio lo/med/hi "
                          f"{ratio[0]:8.2e} {np.median(ratio):8.2e} "
                          f"{ratio[-1]:8.2e}", flush=True)

        # What the atom actually assembles, both modes, and their alphas.
        for k in (1, 2, 3):
            for mode in ("honest", "roundtrip"):
                ms, ks, al = component_factors(
                    seq, k, 0, ktilde_mode=mode, lumped="diag",
                    bc_entry="exact", dirichlet=False)
                lam = [gen_eigs(ks[a], ms[a]) for a in range(3)]
                print(f" atom k={k} c=r {mode:>9} alpha={tuple(round(v, 4) for v in al)} "
                      + " ".join(f"{AXES[a]}:[{lam[a][0]:8.1e},{lam[a][-1]:8.1e}]"
                                 for a in range(3)), flush=True)
        del seq, ops


if __name__ == "__main__":
    main()
