"""Numpy checks of the transcribed Landreman equilibria (landreman_equilibria.py) before any MRX work.

At random points of Omega_delta (drawn through the logical parametrisation, so the map is exercised too):
``div B``, ``curl B x B - grad p`` and ``B . grad psi`` by fourth-order central differences of the closed forms,
relative to ``|dB| |B|`` resp. ``|B| |grad psi|`` scales; ``psi(point(rho, .)) = rho^2 psi_edge``; the reference
2-form's ``B_hat^rho`` (tangency in the logical frame), the zeta-independence of the toroidal flux density, the
flux-ratio iota, ``beta_V = 2 <p>_V / <B^2>_V`` by Gauss quadrature over the logical cube, all against the paper's
numbers. Cheap (login-node numpy). Usage: ``python landreman_check.py [iota2|sheared] [--lam L]``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import landreman_equilibria as le  # noqa: E402


def jac_fd(f, X, h=1e-4):
    """``d f_i / d x_j`` at ``X`` (``(n, 3)``) by fourth-order central differences."""
    cols = []
    for j in range(3):
        e = np.zeros(3)
        e[j] = h
        cols.append((-f(X + 2 * e) + 8 * f(X + e) - 8 * f(X - e) + f(X - 2 * e)) / (12 * h))
    return np.stack(cols, axis=-1)


def check(case, n=400, seed=1):
    rng = np.random.default_rng(seed)
    rho = np.sqrt(rng.uniform(0.0, 1.0, n))
    th, ze = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    X = le.point(case, rho, th, le.TWO_PI * ze / le.NFP)
    B = le.field(case, X)
    dB = jac_fd(lambda Y: le.field(case, Y), X)                   # (n, 3, 3)
    gp = jac_fd(lambda Y: le.pressure(case, Y)[..., None], X)[:, 0, :]
    gpsi = jac_fd(lambda Y: le.psi(case, Y)[..., None], X)[:, 0, :]
    div = np.trace(dB, axis1=1, axis2=2)
    curl = np.stack([dB[:, 2, 1] - dB[:, 1, 2], dB[:, 0, 2] - dB[:, 2, 0], dB[:, 1, 0] - dB[:, 0, 1]], axis=-1)
    force = np.cross(curl, B) - gp
    nB, ndB = np.linalg.norm(B, axis=1), np.linalg.norm(dB.reshape(n, 9), axis=1)
    out = dict(
        div_rel_max=float(np.max(np.abs(div) / ndB)),
        force_rel_max=float(np.max(np.linalg.norm(force, axis=1) / (np.linalg.norm(curl, axis=1) * nB))),
        force_rel_rms_vs_gradp=float(np.sqrt(np.mean(np.sum(force ** 2, 1))) / np.sqrt(np.mean(np.sum(gp ** 2, 1)))),
        Bgradpsi_rel_max=float(np.max(np.abs(np.sum(B * gpsi, 1)) / (nB * np.linalg.norm(gpsi, axis=1)))),
        psi_param_rel_max=float(np.max(np.abs(le.psi(case, X) - rho ** 2 * le.psi_edge(case))) / le.psi_edge(case)),
    )
    # stellarator symmetry and the field period of the exact field on the domain
    Xs = X * np.array([1.0, -1.0, -1.0])
    out["stellsym_B_rel"] = float(np.max(np.abs(le.field(case, Xs) - B * np.array([-1.0, 1.0, 1.0]))) / nB.max())
    Xp = X * np.array([-1.0, -1.0, 1.0])
    out["period_B_rel"] = float(np.max(np.abs(le.field(case, Xp) - B * np.array([-1.0, -1.0, 1.0]))) / nB.max())

    # volume averages over the logical cube (one field period): Gauss in rho, uniform in the angles
    nr, nt, nz = 24, 64, 64
    xg, wg = np.polynomial.legendre.leggauss(nr)
    r = 0.5 * (xg + 1.0)
    Rg, Tg, Zg = np.meshgrid(r, np.arange(nt) / nt, np.arange(nz) / nz, indexing="ij")
    Xq, _, J, Bq, Bh = le.logical_field(case, Rg, Tg, Zg)
    W = (0.5 * wg)[:, None, None] / (nt * nz) * J
    vol = le.NFP * W.sum()
    p_avg = np.sum(W * le.pressure(case, Xq)) / W.sum()
    b2_avg = np.sum(W * np.sum(Bq ** 2, -1)) / W.sum()
    out.update(J_min=float(J.min()), J_max=float(J.max()), volume=float(vol), p_avg=float(p_avg),
               B2_avg=float(b2_avg), beta_V=float(2 * p_avg / b2_avg),
               Bhat_rho_rel=float(np.abs(Bh[..., 0]).max() / np.abs(Bh[..., 2]).max()))
    phi_t = Bh[..., 2].mean(axis=1)                       # d Phi / d rho per zeta plane
    out["dphi_zeta_spread"] = float(((phi_t.max(1) - phi_t.min(1)) / np.abs(phi_t).mean(1)).max())
    out["phi_edge"] = float(np.sum(0.5 * wg * phi_t.mean(1)))
    iota = le.NFP * Bh[..., 1].mean(axis=(1, 2)) / Bh[..., 2].mean(axis=(1, 2))
    out["iota_rho"] = r.tolist()
    out["iota_flux_ratio"] = iota.tolist()

    if case["family"] == "iota2":
        ref = le.iota2_numbers(case["eps"], case["delta"], case["p_edge"])
        out["ref"] = {k: float(v) for k, v in ref.items()}
    else:
        ref = le.sheared_numbers(case["eps"], case["S"], case["k_b"], case["lam"], case["kappa"], case["p_edge"])
        q, a = ref.pop("derivs")(case["k_b"] * r)
        out["iota_landreman"] = (a / q).tolist()
        # his toroidal flux through t = 0: Q = int Q'(k) dk = int (Q'/k) k dk, k = k_b rho
        out["ref"] = dict(ref, phi_edge_landreman=float(np.sum(0.5 * wg * (q * case["k_b"] * r) * case["k_b"])))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("family", choices=tuple(le.CASES))
    ap.add_argument("--lam", type=float, default=None)
    ap.add_argument("--out", default=None)
    cli = ap.parse_args()
    case = dict(le.CASES[cli.family])
    if cli.lam is not None:
        case["lam"] = cli.lam
    res = dict(case=case, **check(case))
    txt = json.dumps(res, indent=1)
    print(txt)
    if cli.out:
        with open(cli.out, "w") as fh:
            fh.write(txt)
