"""Dense spectrum diagnostic for the block-Jacobi preconditioner.

Forms ``L_k`` and ``P`` explicitly (one apply per column) and looks at the
spectrum of ``P L``. Small sizes only -- this is a mechanism probe, not a
benchmark.

The question it answers: when the free-BC cases lag, is it

* a handful of OUTLIER eigenvalues -> a few modes the atom misrepresents, and
  taking those rows exactly should fix it;
* a uniformly stretched spectrum -> the bulk model itself is worse; or
* non-positive eigenvalues -> a construction error, not an approximation.

It reports where the extreme eigenvectors LIVE, in BOTH senses:

* radially -- the natural-BC story says the bad free-BC modes sit on the outer
  radial boundary (``S_k`` carries no boundary term and ``W_0 = 0``, so k=0
  should look the same free and dbc);
* by COMPONENT -- the diagnosis in the natural-BC handoff (§5) is that a scalar
  Laplacian per component cannot carry a VECTOR natural BC: at k=1 free the
  atom asserts ``d_r u_t = 0`` where the operator asks only for
  ``d_r u_t = d_t u_r``.  If that is what the outliers are, they must live on
  the TANGENTIAL components, at the outer boundary, and the ``tg`` penalty --
  which acts exactly there -- must be the knob that moves them.

Several arms can be compared in one build, so the same L is reused and the
outlier counts are directly comparable:

    python scripts/debug/block_jacobi_spectrum.py --geometry rot-ellipse \
        --ns 6,12,6 --ks 1 --bcs free --arms nobc_r3,ibpd_r3,ibpd_r3_tg10
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mrx  # noqa: E402
import mrx.operators as op  # noqa: E402
from mrx.derham_sequence import DeRhamSequence  # noqa: E402
from mrx.experimental.block_jacobi_coarse import (  # noqa: E402
    CoarseCorrectedBlockJacobi,
)
from mrx.experimental.block_jacobi_laplacian import (  # noqa: E402
    BlockJacobiLaplacian)
from mrx.mappings import cylinder_map, rotating_ellipse_map, toroid_map  # noqa: E402

mrx.MAP_BATCH_SIZE_INNER = int(os.environ.get("W7X_MAP_BATCH", "256"))


def build_sequence(geometry, ns, p):
    seq = DeRhamSequence(ns, (p,) * 3, 2 * p, ("clamped", "periodic", "periodic"),
                         polar=True, tol=1e-12, maxiter=1000,
                         betti_numbers=(1, 1, 0, 0))
    seq.evaluate_1d()
    if geometry == "toroid":
        seq.set_map(toroid_map(epsilon=1 / 3, R0=1.0))
    elif geometry == "cylinder":
        # Periodic cylinder F(r,chi,z) = (a r cos2pi chi, a r sin2pi chi, h z).
        # a = 0.33 keeps the minor radius comparable to the toroid's; h = 1.0.
        # ZERO angular metric variation -- the least-coupled geometry there is,
        # and so the zero-coupling end of the s_opt trend (see handoff 17.6).
        seq.set_map(cylinder_map(a=0.33, h=1.0))
    elif geometry == "rot-ellipse":
        # Same parameters as verify_block_jacobi.py.
        seq.set_map(rotating_ellipse_map(eps=0.33, kappa=1.5, nfp=3))
    else:
        from w7x_geometry import build_w7x_map  # noqa: PLC0415
        map_func, _ = build_w7x_map(map_ns=ns, p=p)
        seq.set_map(map_func)
    ops = op.assemble_incidence_operators(seq)
    ops = op.assemble_mass_jacobi_preconditioner(seq, ops, ks=(0, 1, 2, 3))
    seq.set_operators(ops)
    return seq, ops


def make_preconditioner(seq, ops, k, dbc, arm):
    """Same arm grammar as verify_block_jacobi.py, so the two are comparable."""
    if arm == "jacobi":
        d = jnp.asarray(op._hodge_diaginv(seq, ops, k, dbc))
        return lambda v: d * v, None
    pc = re.search(r"bcp(\d+)", arm)
    sc = re.search(r"bcs(\d+)", arm)
    m = re.search(r"_r(\d+)", arm)
    o = re.search(r"_o(\d+)", arm)
    fm = re.search(r"fm(\d+)", arm)
    fr = re.search(r"fr(\d+)", arm)
    ft = re.search(r"ft(\d+)", arm)
    os.environ["MRX_BJ_BC_SCALE"] = (str(int(pc.group(1)) / 100.0) if pc else
                                     sc.group(1) if sc else "1.0")
    kwargs = dict(
        ktilde_mode=("roundtrip" if "rt" in arm else "honest"),
        lumped="diag",
        extra_rings=int(m.group(1)) if m else 0,
        outer_rings=int(o.group(1)) if o else 0,
        # Only the derived term and "off" remain; see verify_block_jacobi.py.
        bc_entry=(False if "nobc" in arm else "ibpd"))
    if fm or fr:
        # `fm` is EXPERIMENTAL and opt-in -- it lives in block_jacobi_coarse,
        # not on the production class. See that module for why.
        pre = CoarseCorrectedBlockJacobi(
            seq, ops, k, dbc,
            coarse_rings=(int(fr.group(1)) if fr else 1),
            coarse_modes=((int(fm.group(1)),) * 2 if fm else (3, 3)),
            coarse_set=("other" if "fso" in arm else
                        "trace" if "fst" in arm else "all"),
            coarse_mode="additive" if "fadd" in arm else "hybrid",
            coarse_trunc=int(ft.group(1)) if ft else 0,
            **kwargs)
    else:
        pre = BlockJacobiLaplacian(seq, ops, k, dbc, **kwargs)
    return pre.apply, pre


def dense_from_apply(apply, n):
    return np.stack([np.asarray(apply(jnp.zeros(n).at[i].set(1.0)))
                     for i in range(n)], axis=1)


def row_labels(seq, k, dirichlet, n_ext):
    """Per extracted row: radial index, that component's radial extent, the
    component index, and the two angular indices.  All -1 for the
    coupled/polar core rows, which are not a single tensor-product basis
    function."""
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
    out_c = np.full(n_ext, -1)
    out_t = np.full(n_ext, -1)
    out_z = np.full(n_ext, -1)
    out_r[r_s] = loc // (nt * nz)
    out_n[r_s] = nr
    out_c[r_s] = comp
    out_t[r_s] = (loc // nz) % nt
    out_z[r_s] = loc % nz
    return out_r, out_n, out_c, out_t, out_z, shapes


def profile(vec, i_r, n_r, i_c, ncomp):
    """Energy fractions of one eigenvector: outer boundary, polar core, and per
    component.  Energy weights, not a threshold -- a threshold hides a mode that
    is spread over many small entries on one component."""
    w = np.asarray(vec) ** 2
    tot = w.sum() + 1e-300
    w = w / tot
    outer = float(w[(i_r >= 0) & (i_r >= n_r - 2)].sum())
    core = float(w[i_r < 0].sum())
    comps = [float(w[i_c == c].sum()) for c in range(ncomp)]
    return outer, core, comps


def mode_content(vec, i_r, i_c, i_t, i_z, shapes):
    """Energy-weighted mean |m| (theta) and |n| (zeta) of one eigenvector.

    The angular bases are uniform and periodic, so coefficient-space Fourier
    content IS mode content -- the same basis the fast diagonalisation uses.
    The question this answers: the natural condition the atom gets wrong is
    ``d_r u_t = d_t u_r``, which at m=0 degenerates to the atom's own
    ``d_r u_t = 0``.  So a penalty on ``u_t(1)`` must be harmless at m=0 and
    grow with |m|; if the modes a constant beta damages are the LOW-|m| ones,
    the penalty has to be mode dependent.
    """
    v = np.asarray(vec)
    e_m = e_n = tot = 0.0
    hist_m = np.zeros(max(sh[1] for sh in shapes) // 2 + 1)
    hist_n = np.zeros(max(sh[2] for sh in shapes) // 2 + 1)
    for c, shape in enumerate(shapes):
        sel = i_c == c
        if not sel.any():
            continue
        arr = np.zeros(shape)
        arr[i_r[sel], i_t[sel], i_z[sel]] = v[sel]
        f = np.fft.fft2(arr, axes=(1, 2))
        w = np.abs(f) ** 2
        m = np.abs(np.fft.fftfreq(shape[1]) * shape[1])
        n = np.abs(np.fft.fftfreq(shape[2]) * shape[2])
        e_m += float(np.einsum('rtz,t->', w, m))
        e_n += float(np.einsum('rtz,z->', w, n))
        tot += float(w.sum())
        np.add.at(hist_m, m.astype(int), np.einsum('rtz->t', w))
        np.add.at(hist_n, n.astype(int), np.einsum('rtz->z', w))
    tot = tot + 1e-300
    # The MEAN is too blunt to design a cutoff with: what `fm` needs to know is
    # "which M captures this mode", i.e. the percentile. m95 = smallest M with
    # 95% of the mode's energy inside |m| <= M -- read it straight as the
    # coarse-space cutoff the mode demands.
    def _pct(hist, frac=0.95):
        c = np.cumsum(hist) / (hist.sum() + 1e-300)
        idx = np.flatnonzero(c >= frac)
        return float(idx[0]) if idx.size else float(len(hist) - 1)
    return (e_m / tot, e_n / tot, _pct(hist_m), _pct(hist_n))


def fmt_profile(outer, core, comps):
    cs = "/".join(f"{c:.2f}" for c in comps)
    return f"outer={outer:.2f} core={core:.2f} comp[{cs}]"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry", default="toroid",
                    choices=("toroid", "rot-ellipse", "w7x", "cylinder"))
    ap.add_argument("--ns", default="6,12,6")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--ks", default="0,1")
    ap.add_argument("--bcs", default="free,dbc")
    ap.add_argument("--arms", default="ibpd_r3")
    ap.add_argument("--nmax", type=int, default=1600)
    ap.add_argument("--show", type=int, default=6,
                    help="how many extreme modes to profile at each end")
    cli = ap.parse_args()

    ns = tuple(int(v) for v in cli.ns.split(","))
    seq, ops = build_sequence(cli.geometry, ns, cli.p)
    arms = cli.arms.split(",")
    want_bc = set(cli.bcs.split(","))
    print(f"geometry={cli.geometry} ns={ns} p={cli.p} arms={arms}", flush=True)

    for k in (int(v) for v in cli.ks.split(",")):
        for dbc in (False, True):
            if ("dbc" if dbc else "free") not in want_bc:
                continue
            n = int(getattr(seq, f"n{k}_dbc" if dbc else f"n{k}"))
            if n > cli.nmax:
                print(f"k={k} dbc={dbc}: n={n} > nmax, skipped", flush=True)
                continue
            lmat = dense_from_apply(
                lambda x, k=k, dbc=dbc: op.apply_hodge_laplacian_approx(
                    seq, ops, x, k, dirichlet=dbc), n)
            i_r, n_r, i_c, i_t, i_z, shapes = row_labels(seq, k, dbc, n)
            ncomp = len(shapes)
            print(f"\n===== k={k} dbc={dbc} n={n} "
                  f"(components {ncomp}, core rows {int((i_r < 0).sum())})",
                  flush=True)

            for arm in arms:
                try:
                    apply, _ = make_preconditioner(seq, ops, k, dbc, arm)
                    pmat = dense_from_apply(apply, n)
                except Exception as exc:  # noqa: BLE001
                    print(f"  {arm}: {type(exc).__name__}: {exc}", flush=True)
                    continue

                sym = float(np.abs(pmat - pmat.T).max() / np.abs(pmat).max())
                w_p = np.linalg.eigvalsh(0.5 * (pmat + pmat.T))
                w, v = np.linalg.eig(pmat @ lmat)
                w, v = np.real(w), np.real(v)
                order = np.argsort(w)
                w, v = w[order], v[:, order]

                pos = w[w > 1e-12 * w.max()]
                cond = pos.max() / pos.min()
                med = np.median(pos)
                hi = (pos > 8 * med)
                lo = (pos < med / 8)
                nout = int(hi.sum() + lo.sum())

                print(f"\n  --- {arm}", flush=True)
                print(f"  P sym err {sym:.2e}  min eig(P) {w_p.min():.3e}",
                      flush=True)
                print(f"  spec(PL): min {pos.min():.3e} max {pos.max():.3e} "
                      f"cond {cond:.3e} median {med:.3e}", flush=True)
                print(f"  outliers(8x median): {nout} of {pos.size}  "
                      f"({int(hi.sum())} high, {int(lo.sum())} low)",
                      flush=True)

                # Where the outliers live, aggregated -- this is the test, not
                # the individual modes.  Energy-weighted mean over the outlier
                # set, so one big mode cannot speak for the group.
                nlow = int(lo.sum())
                nhigh = int(hi.sum())
                off = w.size - pos.size  # nullspace columns sit first
                for tag, idx in (("low outliers", range(off, off + nlow)),
                                 ("high outliers",
                                  range(w.size - nhigh, w.size))):
                    idx = list(idx)
                    if not idx:
                        print(f"  {tag}: none", flush=True)
                        continue
                    rows = []
                    for i in idx:
                        outer, core, comps = profile(v[:, i], i_r, n_r, i_c,
                                                     ncomp)
                        mbar, nbar, m95, n95 = mode_content(
                            v[:, i], i_r, i_c, i_t, i_z, shapes)
                        rows.append([outer, core] + comps
                                    + [mbar, nbar, m95, n95])
                    mean = np.array(rows).mean(axis=0)
                    print(f"  {tag} ({len(idx)}): mean "
                          + fmt_profile(mean[0], mean[1], mean[2:-4])
                          + f" |m|={mean[-4]:.2f} |n|={mean[-3]:.2f}"
                          + f" m95={mean[-2]:.2f} n95={mean[-1]:.2f}",
                          flush=True)

                for i in range(min(cli.show, pos.size)):
                    j = off + i
                    print(f"    lo[{i}] lam {w[j]:.3e}  "
                          + fmt_profile(*profile(v[:, j], i_r, n_r, i_c, ncomp))
                          + "  |m|={:.2f} |n|={:.2f} m95={:.0f} n95={:.0f}".format(
                              *mode_content(v[:, j], i_r, i_c, i_t, i_z,
                                            shapes)), flush=True)
                for i in range(min(cli.show, pos.size)):
                    j = w.size - 1 - i
                    print(f"    hi[{i}] lam {w[j]:.3e}  "
                          + fmt_profile(*profile(v[:, j], i_r, n_r, i_c, ncomp))
                          + "  |m|={:.2f} |n|={:.2f} m95={:.0f} n95={:.0f}".format(
                              *mode_content(v[:, j], i_r, i_c, i_t, i_z,
                                            shapes)), flush=True)


if __name__ == "__main__":
    main()
