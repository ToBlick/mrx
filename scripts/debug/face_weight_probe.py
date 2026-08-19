"""Measure the natural-BC face operator, instead of deriving it.

The quantity that isolates the trace is the FREE-minus-DIRICHLET difference of
the outer ring block:

    T = E^T L_free E  -  E^T L_dbc E

Both carry the same discretisation on that ring; the natural-BC trace is present
in one and provably absent in the other, so everything else cancels.  Comparing
``E^T(L - A)E`` against the model instead -- as the first version of this script
did -- conflates the trace with every other modelling error on the ring, and
compares a surface integral against a 3-D operator block, whose magnitudes
differ by a radial measure.

Reported against the modelled face operator ``B`` in NORMALISED form, so the
units do not have to line up:

* ``corr``   -- correlation of ``diag(T)`` with ``diag(B)``: is the weight FIELD
  right?  1.0 means the shape is right whatever the constant.
* ``scale``  -- the constant, ``median(diag(T)/diag(B))``.
* ``spread`` -- p10-p90 of that ratio divided by its median: 0 means a pure
  constant offset (right family, wrong factor); large means the wrong family.

k=1 is the control (its weight was derived and works).  k=2 is the question:
the trace is TANGENTIAL there (`w x n` pairs `w_t` with `tau_z`), so it lives on
components t and z, and it comes from the 2-form mass weight `g_ab/J` rather
than the contravariant `g^{aa}J` that k=1 uses.
"""
from __future__ import annotations

import argparse
import os
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mrx  # noqa: E402
import mrx.operators as op  # noqa: E402
from mrx.derham_sequence import DeRhamSequence  # noqa: E402
from mrx.experimental.block_jacobi_laplacian import (  # noqa: E402
    BlockJacobiLaplacian, component_factors, face_operator)
from mrx.mappings import toroid_map  # noqa: E402

mrx.MAP_BATCH_SIZE_INNER = int(os.environ.get("W7X_MAP_BATCH", "256"))


def build_sequence(geometry, ns, p):
    seq = DeRhamSequence(ns, (p,) * 3, 2 * p, ("clamped", "periodic", "periodic"),
                         polar=True, tol=1e-13, maxiter=2000,
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


def atom_forward(seq, k, c, window):
    """Forward action of the bulk atom: the three-term Kronecker sum."""
    masses, stiffs, alpha = component_factors(
        seq, k, c, window=window, ktilde_mode="honest", lumped="diag",
        bc_entry=False, dirichlet=False)
    m = [np.asarray(x) for x in masses]
    s = [np.asarray(x) for x in stiffs]

    def apply(t):
        out = alpha[0] * np.einsum('ij,jkl->ikl', s[0], t)
        out = np.einsum('jk,ikl->ijl', m[1], out)
        out = np.einsum('kl,ijl->ijk', m[2], out)
        t1 = np.einsum('ij,jkl->ikl', m[0], t)
        t1 = alpha[1] * np.einsum('jk,ikl->ijl', s[1], t1)
        out += np.einsum('kl,ijl->ijk', m[2], t1)
        t2 = np.einsum('ij,jkl->ikl', m[0], t)
        t2 = np.einsum('jk,ikl->ijl', m[1], t2)
        out += alpha[2] * np.einsum('kl,ijl->ijk', s[2], t2)
        return out
    return apply


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry", default="toroid", choices=("toroid", "w7x"))
    ap.add_argument("--ns", default="6,12,6")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--ks", default="1,2")
    cli = ap.parse_args()

    ns = tuple(int(v) for v in cli.ns.split(","))
    seq, ops = build_sequence(cli.geometry, ns, cli.p)
    print(f"geometry={cli.geometry} ns={ns} p={cli.p}", flush=True)

    for k in (int(v) for v in cli.ks.split(",")):
        pres = {d: BlockJacobiLaplacian(seq, ops, k, d, ktilde_mode="honest",
                                        lumped="diag", extra_rings=0,
                                        bc_entry=False)
                for d in (False, True)}
        for c in range(len(pres[False].blocks)):
            if pres[False].blocks[c] is None or pres[True].blocks[c] is None:
                continue
            # per-component derivative axes: the trace lives where the RADIAL
            # axis is differentiated, i.e. where delta acts.
            deriv_axes = {0: (), 1: (c,), 3: (0, 1, 2)}.get(
                k, tuple(a for a in range(3) if a != c))
            ring = {}
            for dbc, pre in pres.items():
                blk = pre.blocks[c]
                nr, nt, nz = blk["shape"]
                ir, it, iz = blk["idx"]
                last = np.asarray(ir) == nr - 1
                rows = np.asarray(blk["rows"])[last]
                vals = np.asarray(blk["vals"])[last]
                face = np.asarray(it)[last] * nz + np.asarray(iz)[last]
                order = np.argsort(face)
                rows, vals, face = rows[order], vals[order], face[order]
                n_ext = int(getattr(seq, f"n{k}_dbc" if dbc else f"n{k}"))
                mat = np.zeros((face.size, face.size))
                for j in range(face.size):
                    e = jnp.zeros(n_ext).at[int(rows[j])].set(float(vals[j]))
                    col = np.asarray(op.apply_hodge_laplacian_approx(
                        seq, ops, e, k, dirichlet=dbc))
                    mat[:, j] = col[rows] * vals
                ring[dbc] = (0.5 * (mat + mat.T), face, blk["shape"])

            (t_free, f_free, shp), (t_dbc, f_dbc, _) = ring[False], ring[True]
            common, i_f, i_d = np.intersect1d(f_free, f_dbc,
                                              return_indices=True)
            t = t_free[np.ix_(i_f, i_f)] - t_dbc[np.ix_(i_d, i_d)]
            b = np.asarray(face_operator(seq, k, c, (0, shp[0])))
            b = b[np.ix_(common, common)]

            dt, db = np.diag(t), np.diag(b)
            ok = np.abs(db) > 1e-300
            corr = (np.corrcoef(dt[ok], db[ok])[0, 1] if ok.sum() > 2
                    else float("nan"))
            r = dt[ok] / db[ok]
            med = np.median(r)
            spread = ((np.percentile(r, 90) - np.percentile(r, 10))
                      / abs(med) if med else float("nan"))
            print(f"\nk={k} c={c}  radial axis "
                  f"{'DERIVATIVE -> trace expected' if 0 in deriv_axes else 'primal -> NO trace'}",
                  flush=True)
            print(f"  ||T|| / ||L_free ring||  = "
                  f"{np.linalg.norm(t) / np.linalg.norm(t_free):.3e}")
            print(f"  corr(diag T, diag B)     = {corr:.4f}")
            print(f"  scale  median(T/B)       = {med:.4e}")
            print(f"  spread (p90-p10)/|med|   = {spread:.3f}", flush=True)


if __name__ == "__main__":
    main()
