"""A DESC output file of a closed-form stellarator-symmetric torus, for the reader tests.

:func:`write_synthetic_desc` writes the HDF5 layout of a DESC ``EquilibriaFamily`` with two members. The
first has a displaced axis, the last is the solution a reader must pick. In DESC's product basis
(``t = theta`` and ``z = zeta`` in radians, ``Z_3^1 = 3 r^3 - 2 r``, ``Z_2^0 = 2 r^2 - 1``,
``Z_2^2 = r^2``) the solution is

    R      = R0 + a r cos t + b Z_3^1 cos t cos(nfp z) + d Z_2^2 sin 2t sin(nfp z)
    Z      = -a r sin t + e Z_2^0 sin(nfp z)
    lambda = A r sin t (1 + 0.3 cos(nfp z)) + f Z_2^2 cos 2t sin(nfp z)
    p      = p0 (1 - r^2),   Psi (toroidal flux, Wb)

and either ``iota = iota0 + iota2 r^2`` or the net toroidal current ``I = I2 r^2 + I4 r^4``. The terms
cover every DESC parity (cos.cos, sin.sin, sin.cos, cos.sin, ``m = 0`` with ``n < 0``) and ``Z = -a r sin t``
makes the Jacobian positive, as in every DESC file. ``asym = (g, h, k)`` adds ``g Z_2^2 cos 2t sin(nfp z)``
to ``R``, ``h r cos t`` to ``Z`` and ``k Z_2^0 cos(nfp z)`` to lambda (``sym = False``). :meth:`SyntheticDESC.fields` evaluates the formulas.
With ``b = d = e = 0`` (circular cross-section, ``g_tz = 0``) the current gives
``iota = mu0 I(r) sqrt(R0^2 - a^2 r^2) / (2 Psi r^2)`` (:meth:`SyntheticDESC.iota`).
"""
from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np
from scipy.constants import mu_0

LA_ZETA_MODULATION = 0.3


@dataclass(frozen=True)
class SyntheticDESC:
    R0: float
    a: float
    b: float
    d: float
    e: float
    A: float
    f: float
    nfp: int
    Psi: float
    p0: float
    iota2: tuple | None       # (iota0, iota2)
    current: tuple | None     # (I2, I4)
    asym: tuple = (0.0, 0.0, 0.0)

    def fields(self, r, t, z):
        """``R, Z, lambda`` at ``(r, theta, zeta)`` (radians, broadcast)."""
        nz = self.nfp * z
        R = (self.R0 + self.a * r * np.cos(t) + self.b * (3 * r ** 3 - 2 * r) * np.cos(t) * np.cos(nz)
             + self.d * r ** 2 * np.sin(2 * t) * np.sin(nz))
        Z = -self.a * r * np.sin(t) + self.e * (2 * r ** 2 - 1) * np.sin(nz)
        lam = (self.A * r * np.sin(t) * (1 + LA_ZETA_MODULATION * np.cos(nz))
               + self.f * r ** 2 * np.cos(2 * t) * np.sin(nz))
        g, h, k = self.asym
        return (R + g * r ** 2 * np.cos(2 * t) * np.sin(nz), Z + h * r * np.cos(t),
                lam + k * (2 * r ** 2 - 1) * np.cos(nz))

    def iota(self, r):
        """The rotational transform per full turn: the stored profile, or the closed form from the current
        (valid for ``b = d = e = 0``)."""
        if self.iota2 is not None:
            return self.iota2[0] + self.iota2[1] * r ** 2
        I2, I4 = self.current
        return mu_0 * (I2 + I4 * r ** 2) * np.sqrt(self.R0 ** 2 - self.a ** 2 * r ** 2) / (2 * self.Psi)

    def pressure(self, r):
        return self.p0 * (1 - r ** 2)


def _power_series(eq, name, powers, params):
    node = eq.create_group(name)
    node["__class__"] = np.bytes_(b"desc.profiles.PowerSeriesProfile")
    node["_params"] = np.asarray(params, dtype=np.float64)
    node.create_group("_basis")["_modes"] = np.array([[p, 0, 0] for p in powers], dtype=np.int64)


def write_synthetic_desc(path, *, R0=1.0, a=1 / 3, b=0.02, d=0.01, e=0.015, A=0.05, f=0.02, nfp=5,
                         Psi=0.35, p0=400.0, iota=(0.9, 0.15), current=None, asym=None):
    """Write the file and return its :class:`SyntheticDESC`. Passing ``current = (I2, I4)`` instead of
    ``iota`` makes the file current-constrained."""
    torus = SyntheticDESC(R0, a, b, d, e, A, f, nfp, Psi, p0, None if current else iota, current,
                          asym or (0.0, 0.0, 0.0))
    fields = {                                  # (l, m, n): coefficient
        "R": {(0, 0, 0): R0, (1, 1, 0): a, (3, 1, 1): b, (2, -2, -1): d},
        "Z": {(1, -1, 0): -a, (2, 0, -1): e},
        "L": {(1, -1, 0): A, (1, -1, 1): LA_ZETA_MODULATION * A, (2, 2, -1): f},
    }
    if asym:
        fields["R"][(2, 2, -1)], fields["Z"][(1, 1, 0)], fields["L"][(2, 0, 1)] = asym
    with h5py.File(path, "w") as fh:
        fh["__class__"] = np.bytes_(b"desc.equilibrium.equilibrium.EquilibriaFamily")
        family = fh.create_group("_equilibria")
        family["__class__"] = np.bytes_(b"list")
        for member in range(2):
            eq = family.create_group(str(member))
            eq["__class__"] = np.bytes_(b"desc.equilibrium.equilibrium.Equilibrium")
            eq["_sym"], eq["_NFP"], eq["_Psi"] = np.bool_(not asym), np.int64(nfp), np.float64(Psi)
            eq["_L"], eq["_M"], eq["_N"] = np.int64(3), np.int64(2), np.int64(1)
            for name, modes in fields.items():
                coef = dict(modes)
                if member == 0 and name == "R":
                    coef[(0, 0, 0)] += a                     # the unconverged member: a displaced axis
                eq[f"_{name}_lmn"] = np.array(list(coef.values()), dtype=np.float64)
                eq.create_group(f"_{name}_basis")["_modes"] = np.array(list(coef), dtype=np.int64)
            _power_series(eq, "_pressure", [0, 2], [p0, -p0])
            eq["_anisotropy"] = np.bytes_(b"None")
            if current:
                eq["_iota"] = np.bytes_(b"None")
                _power_series(eq, "_current", [2, 4], current)
            else:
                _power_series(eq, "_iota", [0, 2], iota)
                eq["_current"] = np.bytes_(b"None")
    return torus
