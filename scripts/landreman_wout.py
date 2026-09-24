"""Write the VMEC wout of a Landreman equilibrium (landreman_equilibria.py) and check it.

Checks, at random logical points (R, Z) and at random angles on the half-mesh surfaces (the field): the truncated
series against the closed-form map, and the reference 2-form rebuilt from the series and the profiles in the VMEC
convention, ``B_hat^zeta = phi'(rho) (1 + d_theta lambda)``, ``B_hat^theta = phi'(rho) (iota - d_phi lambda) / nfp``
(``phi' = d phi / d rho``, angles in radians), against ``J DF^-1 B`` of the exact field. Numpy only; run it inside
the job that uses the file (``--ns 201`` is a few minutes on 4 cores). Usage::

    python landreman_wout.py --case sheared --out outputs/.../wout_landreman_sheared.nc
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import landreman_equilibria as le  # noqa: E402

GRIDS = {"iota2": (48, 48), "sheared": (32, 128), "shearedA": (32, 64)}     # (n_theta, n_zeta): series truncation < 1e-11 (tested)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=tuple(le.CASES), required=True)
    ap.add_argument("--ns", type=int, default=201)
    ap.add_argument("--out", required=True)
    cli = ap.parse_args(argv)
    case = le.CASES[cli.case]
    nt, nz = GRIDS[cli.case]
    t0 = time.perf_counter()
    arr, chk = le.build_wout(case, cli.ns, nt, nz)
    le.write_wout(cli.out, **arr)
    chk["t_build"] = time.perf_counter() - t0

    # independent checks at random points
    rng = np.random.default_rng(3)
    n = 300
    s = np.arange(cli.ns) / (cli.ns - 1)
    j = rng.integers(1, cli.ns, n)
    th, ze = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    xm, xn = arr["xm"], arr["xn"]
    arg = le.TWO_PI * (xm[None, :] * th[:, None] - (xn[None, :] / le.NFP) * ze[:, None])
    R = np.sum(arr["rmnc"][j] * np.cos(arg), 1)
    Z = np.sum(arr["zmns"][j] * np.sin(arg), 1)
    X = le.point(case, np.sqrt(s[j]), th, le.TWO_PI * ze / le.NFP)
    chk["series_R_err_max"] = float(np.abs(R - np.hypot(X[:, 0], X[:, 1])).max())
    chk["series_Z_err_max"] = float(np.abs(Z - X[:, 2]).max())
    # the field at half-mesh surfaces
    rho_h = np.sqrt((j - 0.5) / (cli.ns - 1))
    _, _, _, _, Bh = le.logical_field(case, rho_h, th, ze)
    lam = arr["lmns_half"][j]
    dth = np.sum(lam * xm * np.cos(arg), 1)
    dph = np.sum(-lam * xn * np.cos(arg), 1)
    s_h = rho_h ** 2
    phi_s = np.interp(s_h, s, np.gradient(arr["phi"], s, edge_order=2))      # d phi / ds, linear in s
    dphi = 2 * rho_h * phi_s
    iota = np.interp(s_h, s, arr["iotaf"])
    Bz = dphi * (1 + dth)
    Bt = dphi * (iota - dph) / le.NFP
    chk["Bhat_zeta_rel_err_max"] = float(np.abs(Bz - Bh[:, 2]).max() / np.abs(Bh[:, 2]).max())
    chk["Bhat_theta_rel_err_max"] = float(np.abs(Bt - Bh[:, 1]).max() / np.abs(Bh[:, 1]).max())
    chk["Bhat_rho_rel_max"] = float(np.abs(Bh[:, 0]).max() / np.abs(Bh[:, 2]).max())
    chk.update(case=case, ns=cli.ns, n_theta=nt, n_zeta=nz, mnmax=int(len(xm)), out=os.path.abspath(cli.out),
               phi_edge=float(arr["phi"][-1]), iota_axis=float(arr["iotaf"][0]), iota_edge=float(arr["iotaf"][-1]))
    print(json.dumps(chk, indent=1), flush=True)
    with open(os.path.splitext(cli.out)[0] + "_checks.json", "w") as fh:
        json.dump(chk, fh, indent=1)


if __name__ == "__main__":
    main()
