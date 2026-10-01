"""Using VMEC's ``wout_*.nc`` files as input.

The wout NetCDF file holds:
- Fourier modes ``(xm, xn)`` (``xn`` multiplied by ``nfp``)
- for every radial surface, coefficients of
    - ``R`` (``rmnc``, cosine series)
    - ``Z`` (``zmns``, sine series)
    - lambda (``lmns``, sine series, in radians)
    - toroidal flux (``phi``, divided by 2pi here)
    - rotational transform (``iotaf``)
    - pressure (``presf``)
- with ``lasym = 1`` (no stellarator symmetry) also the other parities ``rmns``, ``zmnc``, ``lmnc``
The series have argument ``m u - n v`` with ``v`` the full-turn toroidal
angle, so this becomes ``2 pi (m theta - (n / nfp) zeta)`` for us.

VMEC's radial label is ``s = Psi / Psi_edge`` and corresponds to MRX'
``r^2``. ``R`` and ``Z`` are defined on the full mesh ``s_j = j / (ns - 1)``.
Lambda lives on the staggered mesh ``s_{j-1/2} = (j - 1/2) / (ns - 1)``.
Its first row is always dropped. The staggered mesh is extended to the axis
and the edge: ``lambda_mn(0) = 0`` for ``m > 0``, the ``m = 0`` modes at
``r = 0`` and all modes at ``r = 1`` are extrapolated linearly in ``s``.
Each mode and each profile is interpolated to a clamped cubic B-spline in
``r = sqrt(s)`` (:mod:`mrx.equilibria.fit`).

Input validation: files older than VMEC 8 are refused.

Files are read through ``scipy.io.netcdf_file``. ``test/test_readers.py`` tests
the read on the li383 wout in the repo.
"""
from __future__ import annotations

import numpy as np
from scipy.io import netcdf_file

from mrx.equilibria.fit import fit_modes, fit_profile

TWO_PI = 2.0 * np.pi
DEG = 3

_VARIABLES = ("ns", "nfp", "mnmax", "xm", "xn", "lasym__logical__", "version_",
              "rmnc", "zmns", "lmns", "phi", "iotaf", "presf")
_ASYMMETRIC = ("rmns", "zmnc", "lmnc")


def read_wout(path):
    """Read a wout file and return its state (:mod:`mrx.equilibria`). The state also holds VMEC's number
    of surfaces ``ns`` and number of modes ``mnmax``."""
    with open(path, "rb") as fh:
        if fh.read(3) != b"CDF":
            raise ValueError(f"{path}: not a NetCDF3 classic wout file")
    with netcdf_file(path, mmap=False) as f:
        names = _VARIABLES + (_ASYMMETRIC if int(f.variables["lasym__logical__"][()]) else ())
        raw = {k: np.array(f.variables[k][()]) for k in names}
    if float(raw["version_"]) < 8.0:
        raise ValueError(f"{path}: VMEC version {float(raw['version_']):g} < 8 stores lambda "
                         "on the full mesh, refused")

    ns, nfp = int(raw["ns"]), int(raw["nfp"])
    m, n = raw["xm"].astype(int), raw["xn"].astype(int)
    zero = np.zeros_like(raw["rmnc"])                  # a symmetric file's missing parities
    r_full = np.sqrt(np.arange(ns) / (ns - 1))
    r_half = np.sqrt((np.arange(1, ns) - 0.5) / (ns - 1))
    s = r_half ** 2
    r_la = np.concatenate([[0.0], r_half, [1.0]])

    def full(key):
        x = raw.get(key, zero).copy()
        x[0, m > 0] = 0.0                              # the m > 0 axis rows are an extrapolation, not data
        return x

    def half(key):
        lm = raw.get(key, zero)[1:]
        axis = np.where(m > 0, 0.0, lm[0] + (lm[1] - lm[0]) * (0.0 - s[0]) / (s[1] - s[0]))
        edge = lm[-1] + (lm[-1] - lm[-2]) * (1.0 - s[-1]) / (s[-1] - s[-2])
        return np.vstack([axis[None, :], lm, edge[None, :]])

    def block(r, cos, sin):
        T, c = fit_modes(r, cos, m, DEG)
        return dict(m=m, n=n, cos=c, sin=fit_modes(r, sin, m, DEG)[1], deg=DEG, T=T)

    return dict(
        nfp=nfp, ns=ns, mnmax=int(raw["mnmax"]),
        X1=block(r_full, full("rmnc"), full("rmns")),
        X2=block(r_full, full("zmnc"), full("zmns")),
        LA=block(r_la, half("lmnc"), half("lmns")),
        profiles=dict(phi=fit_profile(r_full, raw["phi"] / TWO_PI, DEG),
                      iota=fit_profile(r_full, raw["iotaf"], DEG),
                      pressure=fit_profile(r_full, raw["presf"], DEG)))
