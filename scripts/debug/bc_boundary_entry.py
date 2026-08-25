"""What the natural BC adds to the 1-D radial stiffness, measured.

The weak term's exact radial factor is ``F = M^d G A^-1 G^T M^d`` with ``A`` the
LOWER space's radial mass and ``G`` the incidence.  The atom instead uses the
honest stiffness of the derivative splines, ``Ktilde = int (dLam_i)'(dLam_j)' w``.

Integration by parts says the difference is the boundary trace, which lives on
the ``r = 1`` face only -- so ``F - Ktilde`` should be **rank one** and supported
at the last radial index. This script checks that and reads off the coefficient.

Under Dirichlet the trace vanishes, so the same difference computed with the
boundary column of ``G`` removed should be far smaller.
"""
from __future__ import annotations

import argparse
import os
import sys

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mrx  # noqa: E402
import mrx.operators as op  # noqa: E402
from mrx.derham_sequence import DeRhamSequence  # noqa: E402
from mrx.metric_lumping_laplacian import (  # noqa: E402
    _axis_bases, bundled_axis_profiles, weight_fields)
from mrx.mappings import toroid_map  # noqa: E402
from mrx.operators import (  # noqa: E402
    _assemble_weighted_1d_mass, _dense_incidence_1d)

mrx.MAP_BATCH_SIZE_INNER = int(os.environ.get("W7X_MAP_BATCH", "256"))


def build_sequence(geometry, ns, p):
    seq = DeRhamSequence(ns, (p,) * 3, 2 * p, ("clamped", "periodic", "periodic"),
                         polar=True, tol=1e-12, maxiter=1000,
                         betti_numbers=(1, 1, 0, 0))
    seq.evaluate_1d()
    if geometry == "toroid":
        seq.set_map(toroid_map(epsilon=1 / 3, R0=1.0))
    else:
        from w7x_geometry import build_w7x_map  # noqa: PLC0415
        map_func, _ = build_w7x_map(map_ns=ns, p=p)
        seq.set_map(map_func)
    ops = op.assemble_incidence_operators(seq)
    seq.set_operators(op.assemble_mass_jacobi_preconditioner(
        seq, ops, ks=(0, 1, 2, 3)))
    return seq, seq.get_operators()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry", default="toroid", choices=("toroid", "w7x"))
    ap.add_argument("--ns", default="12,24,12")
    ap.add_argument("--p", type=int, default=3)
    cli = ap.parse_args()

    ns = tuple(int(v) for v in cli.ns.split(","))
    seq, _ = build_sequence(cli.geometry, ns, cli.p)
    print(f"geometry={cli.geometry} ns={ns} p={cli.p}\n", flush=True)

    primal, deriv, quad_w = _axis_bases(seq)
    fields = weight_fields(seq)
    ginv, jac = fields["ginv_aa"], fields["jac"]
    types = seq.basis_0.types

    # k=1, r-component: the radial axis is the derivative axis.
    c, a = 0, 0
    prof_k = bundled_axis_profiles(seq, ginv[c] * jac * ginv[a])[a]
    prof_a = bundled_axis_profiles(seq, jac)[a]

    from mrx.local_assembly import _second_derivative_tables  # noqa: PLC0415
    dd = _second_derivative_tables(seq)
    ktilde = np.asarray(_assemble_weighted_1d_mass(dd[a], quad_w[a] * prof_k))

    m_d = np.asarray(_assemble_weighted_1d_mass(
        deriv[a], quad_w[a] * bundled_axis_profiles(seq, ginv[c] * jac)[a]))
    a_mat = np.asarray(_assemble_weighted_1d_mass(
        primal[a], quad_w[a] * prof_a))
    g = np.asarray(_dense_incidence_1d(int(a_mat.shape[0]), types[a]))

    def factor(cols):
        gg = g[:, cols]
        aa = a_mat[np.ix_(cols, cols)]
        return m_d @ (gg @ np.linalg.solve(aa, gg.T)) @ m_d

    n0 = a_mat.shape[0]
    f_free = factor(np.arange(n0))
    f_dbc = factor(np.arange(n0 - 1))          # drop the outer boundary DOF

    for label, f in (("free", f_free), ("dbc", f_dbc)):
        d = f - ktilde
        sv = np.linalg.svd(d, compute_uv=False)
        u, s, vt = np.linalg.svd(d)
        lead = np.abs(u[:, 0])
        lead = lead / lead.max()
        print(f"--- {label}")
        print(f"  ||F - Ktilde|| / ||Ktilde||  = "
              f"{np.linalg.norm(d) / np.linalg.norm(ktilde):.3e}")
        print(f"  top singular values           = "
              f"{np.array2string(sv[:4], precision=3)}")
        print(f"  rank-one-ness  s2/s1          = {sv[1] / sv[0]:.3e}")
        print(f"  alpha (= s1)                  = {sv[0]:.6e}")
        print(f"  leading vector weight on last 3 entries = "
              f"{np.array2string(lead[-3:], precision=3)}")
        print(f"  leading vector weight on first 3        = "
              f"{np.array2string(lead[:3], precision=3)}\n", flush=True)


if __name__ == "__main__":
    main()
