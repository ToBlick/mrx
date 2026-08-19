"""Dense spectrum diagnostic for the block-Jacobi preconditioner.

Forms ``L_k`` and ``P`` explicitly (one apply per column) and looks at the
spectrum of ``P L``. Small sizes only -- this is a mechanism probe, not a
benchmark.

The question it answers: when the free-BC cases lag, is it

* a handful of OUTLIER eigenvalues -> a few modes the atom misrepresents, and
  taking those rows exactly should fix it;
* a uniformly stretched spectrum -> the bulk model itself is worse; or
* non-positive eigenvalues -> a construction error, not an approximation.

It also reports where the extreme eigenvectors LIVE radially, which is the test
of the natural-BC story: the weak block's natural condition contributes a
boundary trace that the atom omits, so under free BC the bad modes should be
localised at the outer radial boundary. ``S_k`` carries no such term, and
``W_0 = 0``, so k=0 should look the same free and dbc.
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
    BlockJacobiLaplacian)
from mrx.mappings import toroid_map  # noqa: E402

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
    ops = op.assemble_mass_jacobi_preconditioner(seq, ops, ks=(0, 1, 2, 3))
    seq.set_operators(ops)
    return seq, ops


def dense_from_apply(apply, n):
    return np.stack([np.asarray(apply(jnp.zeros(n).at[i].set(1.0)))
                     for i in range(n)], axis=1)


def radial_of_rows(seq, k, dirichlet, n_ext):
    """Radial index of each extracted row, -1 for the coupled/polar rows, and
    the per-component radial extent."""
    e = getattr(seq, f"e{k}_dbc" if dirichlet else f"e{k}")
    rows, cols = np.asarray(e.rows), np.asarray(e.cols)
    counts = np.bincount(rows, minlength=n_ext)
    shapes = [tuple(int(v) for v in sh)
              for sh in getattr(seq, f"basis_{k}").shape]
    starts = np.cumsum([0] + [int(np.prod(sh)) for sh in shapes])
    single = counts[rows] == 1
    r_s, c_s = rows[single], cols[single]
    comp = np.searchsorted(starts[1:], c_s, side="right")
    loc = c_s - starts[comp]
    nt = np.array([sh[1] for sh in shapes])[comp]
    nz = np.array([sh[2] for sh in shapes])[comp]
    nr = np.array([sh[0] for sh in shapes])[comp]
    out_r = np.full(n_ext, -1)
    out_n = np.full(n_ext, -1)
    out_r[r_s] = loc // (nt * nz)
    out_n[r_s] = nr
    return out_r, out_n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry", default="toroid", choices=("toroid", "w7x"))
    ap.add_argument("--ns", default="6,12,6")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--ks", default="0,1")
    ap.add_argument("--rings", type=int, default=3)
    ap.add_argument("--outer", type=int, default=0)
    cli = ap.parse_args()

    ns = tuple(int(v) for v in cli.ns.split(","))
    seq, ops = build_sequence(cli.geometry, ns, cli.p)
    print(f"geometry={cli.geometry} ns={ns} p={cli.p} rings={cli.rings} "
          f"outer={cli.outer}", flush=True)

    for k in (int(v) for v in cli.ks.split(",")):
        for dbc in (False, True):
            n = int(getattr(seq, f"n{k}_dbc" if dbc else f"n{k}"))
            if n > 1600:
                print(f"k={k} dbc={dbc}: n={n} too large, skipped", flush=True)
                continue
            lmat = dense_from_apply(
                lambda x, k=k, dbc=dbc: op.apply_hodge_laplacian_approx(
                    seq, ops, x, k, dirichlet=dbc), n)
            pre = BlockJacobiLaplacian(seq, ops, k, dbc, ktilde_mode="honest",
                                       lumped="diag", extra_rings=cli.rings,
                                       outer_rings=cli.outer)
            pmat = dense_from_apply(pre.apply, n)

            sym = float(np.abs(pmat - pmat.T).max() / np.abs(pmat).max())
            w_p = np.linalg.eigvalsh(0.5 * (pmat + pmat.T))
            w, v = np.linalg.eig(pmat @ lmat)
            w = np.real(w)
            order = np.argsort(w)
            w = w[order]
            v = np.real(v[:, order])

            pos = w[w > 1e-12 * w.max()]
            cond = pos.max() / pos.min()
            # How many eigenvalues sit far from the bulk of the spectrum?
            med = np.median(pos)
            outliers = int(np.sum((pos > 8 * med) | (pos < med / 8)))

            i_r, n_r = radial_of_rows(seq, k, dbc, n)
            def where(vec):
                a = np.abs(vec)
                a = a / (a.max() + 1e-300)
                heavy = a > 0.3
                if not heavy.any():
                    return "--"
                rr, nn = i_r[heavy], n_r[heavy]
                frac_out = float(np.mean((rr >= 0) & (rr >= nn - 2)))
                frac_in = float(np.mean(rr < 0))
                return f"outer={frac_out:.2f} core={frac_in:.2f}"

            print(f"\nk={k} dbc={dbc} n={n}", flush=True)
            print(f"  P symmetry err {sym:.2e}   min eig(P) {w_p.min():.3e}",
                  flush=True)
            print(f"  spec(PL): min {pos.min():.3e}  max {pos.max():.3e}  "
                  f"cond {cond:.3e}  outliers(8x median) {outliers}", flush=True)
            print(f"  smallest 5: {np.array2string(pos[:5], precision=3)}",
                  flush=True)
            print(f"  largest  5: {np.array2string(pos[-5:], precision=3)}",
                  flush=True)
            print(f"  smallest-eig mode lives: {where(v[:, 0])}", flush=True)
            print(f"  largest-eig  mode lives: {where(v[:, -1])}", flush=True)


if __name__ == "__main__":
    main()
