#!/usr/bin/env python
"""Fig. 7's inset (Sec. 3.4): how far LP's boundary represented on the spline mesh lies from the VMEC boundary (numpy).

    python paper/ad_baselines.py [--records DIR]

Reads data/wout_LandremanPaul2021_QA_lowres.nc; writes <records>/shape_optimization/qa_boundary_baselines.json, per
mesh (n, 2n, n), n = 16, 24, 32: d_RMS / a and e_R, e_Z of the boundary (2n, n) spline fit against the VMEC boundary.
Login node, numpy only, a few minutes (OMP_NUM_THREADS=4 on a shared node: more threads only contend).

The fit is a least-squares proxy of the interpolated map's boundary: periodic cubic B-splines on the uniform
(n_theta, n_zeta) = (2n, n) boundary grid of one field period, fitted to the VMEC boundary (its Fourier series) on
a dense grid of midpoints, GRID. d_RMS is the area-weighted RMS over the fitted surface of the distance of its
points to the VMEC surface, along the VMEC normal at the closest point (Gauss-Newton from the same angles, as
surface_distance in paper/ad_recovery.py), over the minor radius a (the wout's Aminor_p); e_R and e_Z are the
relative L2 differences of R and Z at equal angles on GRID.
"""
import argparse
import json
import os

import numpy as np
from scipy.io import netcdf_file

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WOUT = os.path.join(REPO, "data", "wout_LandremanPaul2021_QA_lowres.nc")
MESHES = (16, 24, 32)
#: the fit and evaluation grid over (theta, one field period of zeta): midpoints
GRID = (384, 192)


def cubic(t, derivative):
    """The uniform cubic B-spline on [0, 4] and its derivative at ``t``."""
    pieces = ((lambda u: u ** 3 / 6, lambda u: u ** 2 / 2),
              (lambda u: (-3 * u ** 3 + 12 * u ** 2 - 12 * u + 4) / 6, lambda u: (-9 * u ** 2 + 24 * u - 12) / 6),
              (lambda u: (3 * u ** 3 - 24 * u ** 2 + 60 * u - 44) / 6, lambda u: (9 * u ** 2 - 48 * u + 60) / 6),
              (lambda u: (4 - u) ** 3 / 6, lambda u: -(4 - u) ** 2 / 2))
    out = np.zeros_like(t)
    for k, f in enumerate(pieces):
        m = (t >= k) & (t < k + 1)
        out[m] = f[derivative](t[m])
    return out


def periodic(n, x, derivative=0):
    """``(len(x), n)``: the n periodic cubic B-splines on the uniform grid of [0, 1) (or their x-derivatives) at
    ``x``."""
    return cubic((x[:, None] * n - np.arange(n)[None, :] + 3.0) % n, derivative) * n ** derivative


class Vmec:
    """The VMEC boundary, ``R = sum rmnc cos(2 pi (m theta - n zeta / nfp))``, ``Z`` with ``zmns`` and sin, ``zeta``
    in field periods."""

    def __init__(self, path):
        with netcdf_file(path, "r", mmap=False) as f:
            v = f.variables
            self.m, self.n, self.nfp = v["xm"].data.copy(), v["xn"].data.copy(), int(v["nfp"].data)
            self.rc, self.zs = v["rmnc"].data[-1].copy(), v["zmns"].data[-1].copy()
            self.a = float(v["Aminor_p"].data)

    def __call__(self, theta, zeta):
        """``(R, Z, R_theta, Z_theta, R_zeta, Z_zeta)`` at the scattered angles."""
        angle = 2 * np.pi * (self.m[None, :] * theta[:, None] - self.n[None, :] * zeta[:, None] / self.nfp)
        c, s = np.cos(angle), np.sin(angle)
        dt, dz = 2 * np.pi * self.m, -2 * np.pi * self.n / self.nfp
        return c @ self.rc, s @ self.zs, -(s * dt) @ self.rc, (c * dt) @ self.zs, -(s * dz) @ self.rc, (c * dz) @ self.zs


class Fit:
    """The least-squares periodic cubic spline fit of the VMEC boundary on ``(n_theta, n_zeta)``, separable on
    GRID."""

    def __init__(self, vmec, n_theta, n_zeta, theta, zeta):
        self.n = (n_theta, n_zeta)
        T, Zg = np.meshgrid(theta, zeta, indexing="ij")
        R, Z = (v.reshape(T.shape) for v in vmec(T.ravel(), Zg.ravel())[:2])
        Pt, Pz = np.linalg.pinv(periodic(n_theta, theta)), np.linalg.pinv(periodic(n_zeta, zeta))
        self.cR, self.cZ = Pt @ R @ Pz.T, Pt @ Z @ Pz.T

    def __call__(self, theta, zeta):
        """``(R, Z, R_theta, Z_theta, R_zeta, Z_zeta)`` at the scattered angles."""
        Bt, Bz = periodic(self.n[0], theta), periodic(self.n[1], zeta)
        Dt, Dz = periodic(self.n[0], theta, 1), periodic(self.n[1], zeta, 1)

        def ev(C, A, B):
            return np.sum((A @ C) * B, axis=1)
        return (ev(self.cR, Bt, Bz), ev(self.cZ, Bt, Bz), ev(self.cR, Dt, Bz), ev(self.cZ, Dt, Bz),
                ev(self.cR, Bt, Dz), ev(self.cZ, Bt, Dz))


def cartesian(nfp, zeta, R, Z, Rt, Zt, Rz, Zz):
    """``(X, X_theta, X_zeta)`` of the surface, the toroidal angle ``2 pi zeta / nfp``."""
    a = 2 * np.pi / nfp
    c, s = np.cos(a * zeta), np.sin(a * zeta)
    return (np.stack([R * c, R * s, Z], -1), np.stack([Rt * c, Rt * s, Zt], -1),
            np.stack([Rz * c - a * R * s, Rz * s + a * R * c, Zz], -1))


def d_rms(surface, reference, nfp, theta, zeta, iterations=8):
    """The area-weighted RMS over ``surface`` of the distance of its points at the angles to ``reference``, along
    the reference's normal at the closest point."""
    X, Xt, Xz = cartesian(nfp, zeta, *surface(theta, zeta))
    area = np.linalg.norm(np.cross(Xt, Xz), axis=-1)
    u = np.stack([theta, zeta], -1)
    for _ in range(iterations):
        F, Ft, Fz = cartesian(nfp, u[:, 1], *reference(u[:, 0], u[:, 1]))
        J = np.stack([Ft, Fz], -1)
        u = u + np.linalg.solve(np.einsum("pki,pkj->pij", J, J), np.einsum("pki,pk->pi", J, X - F)[..., None])[..., 0]
    F, Ft, Fz = cartesian(nfp, u[:, 1], *reference(u[:, 0], u[:, 1]))
    normal = np.cross(Ft, Fz)
    d = np.sum((X - F) * normal / np.linalg.norm(normal, axis=-1, keepdims=True), -1)
    return float(np.sqrt(np.sum(area * d ** 2) / np.sum(area)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--records", default=os.environ.get("MRX_RECORDS", os.path.join(REPO, "outputs")),
                    help="the records root, <records>/shape_optimization [MRX_RECORDS or outputs]")
    cli = ap.parse_args()
    vmec = Vmec(WOUT)
    theta, zeta = (np.arange(GRID[0]) + 0.5) / GRID[0], (np.arange(GRID[1]) + 0.5) / GRID[1]
    tt, zz = (g.ravel() for g in np.meshgrid(theta, zeta, indexing="ij"))
    R, Z = vmec(tt, zz)[:2]
    rec = {}
    for n in MESHES:
        fit = Fit(vmec, 2 * n, n, theta, zeta)
        Rh, Zh = fit(tt, zz)[:2]
        rec[str(n)] = dict(boundary_grid=[2 * n, n], d_rms_over_a=d_rms(fit, vmec, vmec.nfp, tt, zz) / vmec.a,
                           e_R=float(np.linalg.norm(Rh - R) / np.linalg.norm(R)),
                           e_Z=float(np.linalg.norm(Zh - Z) / np.linalg.norm(Z)))
        print(f"{n}x{2 * n}x{n}, boundary grid {2 * n} x {n}: d_RMS / a {rec[str(n)]['d_rms_over_a']:.4e}, e_R "
              f"{rec[str(n)]['e_R']:.3e}, e_Z {rec[str(n)]['e_Z']:.3e}", flush=True)
    folder = os.path.join(cli.records, "shape_optimization")
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "qa_boundary_baselines.json"), "w") as fh:
        json.dump(dict(rec, a=vmec.a, grid=list(GRID), wout=os.path.relpath(WOUT, REPO)), fh, indent=1)


if __name__ == "__main__":
    main()
