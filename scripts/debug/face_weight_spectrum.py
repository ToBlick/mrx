"""How many Fourier modes does the r=1 face weight actually carry?

If the outer-ring / capacitance route is used, the face block `B` is dense as
currently stored -- (n_t n_z)^2 = N^(4/3) memory and (n_t n_z)^3 = N^2 to
factor. Unaffordable at scale.

But in the ANGULAR EIGENBASIS `B_hat = Q^T B Q` couples mode (j,k) to (j',k')
only where the face weight has Fourier content at (j-j', k-k'). So the
bandwidth of `B_hat` is set by the SPECTRUM OF THE WEIGHT, not by the grid. A
stellarator with nfp field periods should put its content at low poloidal m and
toroidal n that are multiples of nfp -- if so, `B_hat` is banded in mode space,
`(I + D B_hat)` is sparse, and the block factors in O(n_t n_z . bw) memory
instead of (n_t n_z)^2.

This measures it directly: 2-D FFT of w(1,theta,zeta), then the number of modes
needed for 90 / 99 / 99.9% of the energy, and the implied half-bandwidths.

NOTE the eigenbasis to use is an explicit FFT basis, not `eigh` eigenvectors of
(K_t, M_t): those are Fourier only up to an arbitrary rotation inside each
degenerate cos/sin pair, which would smear a banded B_hat into a dense one.
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
from mrx.derham_sequence import DeRhamSequence  # noqa: E402
from mrx.experimental.block_jacobi_laplacian import weight_fields  # noqa: E402
from mrx.mappings import rotating_ellipse_map, toroid_map  # noqa: E402

mrx.MAP_BATCH_SIZE_INNER = int(os.environ.get("W7X_MAP_BATCH", "256"))


def build_sequence(geometry, ns, p):
    seq = DeRhamSequence(ns, (p,) * 3, 2 * p, ("clamped", "periodic", "periodic"),
                         polar=True, tol=1e-12, maxiter=1000,
                         betti_numbers=(1, 1, 0, 0))
    seq.evaluate_1d()
    if geometry == "toroid":
        seq.set_map(toroid_map(epsilon=1 / 3, R0=1.0))
    elif geometry == "rot-ellipse":
        seq.set_map(rotating_ellipse_map(eps=0.33, kappa=1.5, nfp=3))
    else:
        from w7x_geometry import build_w7x_map  # noqa: PLC0415
        map_func, _ = build_w7x_map(map_ns=ns, p=p)
        seq.set_map(map_func)
    return seq


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry", default="w7x",
                    choices=("toroid", "rot-ellipse", "w7x"))
    ap.add_argument("--ns", default="12,24,12")
    ap.add_argument("--p", type=int, default=3)
    cli = ap.parse_args()

    ns = tuple(int(v) for v in cli.ns.split(","))
    seq = build_sequence(cli.geometry, ns, cli.p)
    fields = weight_fields(seq)
    ginv, jac = fields["ginv_aa"], fields["jac"]

    # The face weight the boundary term actually uses (§6.1): J g^rr at r=1.
    w = np.asarray(jac * ginv[0])[-1]                 # (n_qt, n_qz)
    print(f"geometry={cli.geometry} ns={ns} p={cli.p}", flush=True)
    print(f"face weight grid {w.shape}, "
          f"min {w.min():.4g} max {w.max():.4g} "
          f"spread {(w.max() - w.min()) / w.mean() * 100:.1f}%\n", flush=True)

    f = np.fft.fft2(w) / w.size
    e = np.abs(f) ** 2
    tot = e.sum()
    order = np.argsort(e.ravel())[::-1]
    cum = np.cumsum(e.ravel()[order]) / tot
    print(f"{'energy':>8} {'modes':>7} {'frac of grid':>13} "
          f"{'m_max':>6} {'n_max':>6}", flush=True)
    nt, nz = w.shape
    for thr in (0.90, 0.99, 0.999, 0.9999):
        cnt = int(np.searchsorted(cum, thr) + 1)
        idx = order[:cnt]
        mm = np.abs(((idx // nz) + nt // 2) % nt - nt // 2).max()
        nn = np.abs(((idx % nz) + nz // 2) % nz - nz // 2).max()
        print(f"{thr:>8.4f} {cnt:>7d} {cnt / w.size:>13.4f} "
              f"{mm:>6d} {nn:>6d}", flush=True)

    print("\nm_max/n_max are the half-bandwidths B_hat would need. Compare with",
          flush=True)
    print(f"the grid: n_t = {ns[1]}, n_z = {ns[2]}. Dense costs "
          f"(n_t n_z)^2 = {(ns[1] * ns[2]) ** 2} entries; banded costs "
          f"~n_t n_z (2 m_max+1)(2 n_max+1).", flush=True)


if __name__ == "__main__":
    main()
