"""A GVEC-style export of an analytic circular torus, for tests.

:func:`write_synthetic_gvec` writes the flat schema that
:func:`mrx.gvec.build_gvec_map` and :func:`mrx.gvec.load_clebsch` read (the
layout of ``w7x_fmm002_clebsch_mrx.h5``, measured 2026-08-26 and mirrored
here dataset by dataset), but filled from closed formulas: the map is the
circular torus

    R = R0 + a rho cos(theta_G),   Z = a rho sin(theta_G)

and the Clebsch scalars are

    Phi(rho)  = Phi_edge rho^2                         (rho = sqrt(s))
    iota(rho) = iota0 + iota1 rho^2                    (per full toroidal turn)
    chi(rho)  = Phi_edge (iota0 rho^2 + iota1 rho^4 / 2) = int_0^rho iota Phi'
    LA        = lam_amplitude rho sin(theta_G) (1 + LA_ZETA_MODULATION cos(nfp zeta_G))
    p(rho)    = p0 (1 - rho^2),  p0 = beta B0^2 / (2 mu0),  B0 = Phi_edge / (pi a^2)

in GVEC's units: ``theta_G = 2 pi theta`` and ``zeta_G = 2 pi zeta / nfp``
are the radian angles, the stored derivatives are with respect to ``rho``
and to the radian angles, and the evaluation grid is the normalised
``(rho, theta, zeta)`` in [0, 1]. ``lambda`` is stellarator-symmetric (odd
under ``(theta, zeta) -> (-theta, -zeta)``), ``rho^1`` for its ``m = 1``
regularity at the axis, and carries one ``n = nfp`` modulation so both
angular derivatives are exercised.

Grid conventions copied from the real exports:

* ``eval_points`` is ``(N, 3)`` in C order over ``(rho, theta, zeta)``, the
  scalars are flat of length ``N``, ``n_rho``/``n_theta``/``n_zeta``/``nfp``
  are root attributes;
* theta and zeta are sampled half-open, ``i / n`` for ``i < n``; zeta spans
  ONE field period (the real file is stellarator-symmetric under
  ``(theta, zeta) -> (-theta, -zeta)`` on its zeta grid, which only a full
  period admits);
* rho is ``i / (n_rho - 1)`` except the first point, which is
  ``0.1 / (n_rho - 1)``: GVEC does not evaluate on the axis, and every
  export (quasr, W7-X) carries this off-axis first point;
* theta runs counter-clockwise in the ``(R, Z)`` plane from the outboard
  midplane. With ``det DF > 0`` this forces ``Y = -R sin(2 pi zeta / nfp)``,
  the handedness of ``mrx.mappings.toroid_map``, which
  :func:`mrx.gvec.build_gvec_map` measures rather than assumes.

The field on concentric circular surfaces is an equilibrium only in the
large-aspect-ratio, low-beta limit (the analytic map has no Shafranov
shift), so keep ``beta`` small and read the force residual of the projected
field as a property of the choice, not of the code. The pressure is stored
for the schema (``load_clebsch`` returns its surface mean); no solver in
MRX consumes it.
"""
from __future__ import annotations

from dataclasses import dataclass

import h5py
import jax.numpy as jnp
import numpy as np

#: Relative amplitude of the ``cos(nfp zeta_G)`` modulation of lambda.
LA_ZETA_MODULATION = 0.3

MU0 = 4e-7 * np.pi
TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class SyntheticTorus:
    """The closed formulas behind one synthetic export, in normalised
    coordinates ``(rho, theta, zeta)`` in [0, 1] (zeta per field period).

    Every method accepts scalars or arrays and is differentiable with
    ``jax.grad``. Angular derivatives of ``LA`` with respect to GVEC's radian
    angles are ``d/dtheta_G = (1 / 2 pi) d/dtheta`` and
    ``d/dzeta_G = (nfp / 2 pi) d/dzeta``.
    """
    R0: float
    a: float
    nfp: int
    Phi_edge: float
    iota0: float
    iota1: float
    lam_amplitude: float
    beta: float

    def R(self, rho, theta):
        return self.R0 + self.a * rho * jnp.cos(TWO_PI * theta)

    def Z(self, rho, theta):
        return self.a * rho * jnp.sin(TWO_PI * theta)

    def Phi(self, rho):
        return self.Phi_edge * rho ** 2

    def dPhi_dr(self, rho):
        return 2.0 * self.Phi_edge * rho

    def iota(self, rho):
        """Rotational transform per full toroidal turn, ``chi' / Phi'``."""
        return self.iota0 + self.iota1 * rho ** 2

    def chi(self, rho):
        return self.Phi_edge * (self.iota0 * rho ** 2 + 0.5 * self.iota1 * rho ** 4)

    def dchi_dr(self, rho):
        return self.iota(rho) * self.dPhi_dr(rho)

    def LA(self, rho, theta, zeta):
        return (self.lam_amplitude * rho * jnp.sin(TWO_PI * theta)
                * (1.0 + LA_ZETA_MODULATION * jnp.cos(TWO_PI * zeta)))

    @property
    def B0(self):
        """Mean toroidal field ``Phi_edge / (pi a^2)``."""
        return self.Phi_edge / (np.pi * self.a ** 2)

    def pressure(self, rho):
        p0 = self.beta * self.B0 ** 2 / (2.0 * MU0)
        return p0 * (1.0 - rho ** 2)


def synthetic_grid(n_rho, n_theta, n_zeta):
    """The export's axes: off-axis clamped rho, half-open theta and zeta."""
    rho = np.arange(n_rho, dtype=np.float64) / (n_rho - 1)
    rho[0] = 0.1 / (n_rho - 1)
    theta = np.arange(n_theta, dtype=np.float64) / n_theta
    zeta = np.arange(n_zeta, dtype=np.float64) / n_zeta
    return rho, theta, zeta


def write_synthetic_gvec(path, *, R0, a, nfp, n_rho, n_theta, n_zeta, iota,
                         Phi_edge, lam_amplitude, beta):
    """Write the synthetic export to ``path``; returns its :class:`SyntheticTorus`.

    Args:
        path: output file (overwritten).
        R0, a: major and minor radius of the circular torus.
        nfp: field periods; zeta spans one of them.
        n_rho, n_theta, n_zeta: grid sizes (see :func:`synthetic_grid`).
        iota: ``(iota0, iota1)`` of ``iota(rho) = iota0 + iota1 rho^2`` per
            full toroidal turn; negative on W7-X.
        Phi_edge: toroidal flux at ``rho = 1``; ``Phi = Phi_edge rho^2``.
        lam_amplitude: amplitude of ``LA`` (radians); zero switches lambda off.
        beta: on-axis ``2 mu0 p0 / B0^2`` of the stored pressure profile.
    """
    iota0, iota1 = (float(v) for v in iota)
    torus = SyntheticTorus(float(R0), float(a), int(nfp), float(Phi_edge),
                           iota0, iota1, float(lam_amplitude), float(beta))
    rho, theta, zeta = synthetic_grid(n_rho, n_theta, n_zeta)
    RHO, TH, ZE = np.meshgrid(rho, theta, zeta, indexing="ij")
    flat = {
        "R": torus.R(RHO, TH), "Z": torus.Z(RHO, TH),
        "pressure": torus.pressure(RHO),
        "clebsch/Phi": torus.Phi(RHO), "clebsch/chi": torus.chi(RHO),
        "clebsch/dPhi_dr": torus.dPhi_dr(RHO), "clebsch/dchi_dr": torus.dchi_dr(RHO),
        "clebsch/LA": torus.LA(RHO, TH, ZE),
    }
    with h5py.File(path, "w") as h:
        h.attrs["n_rho"], h.attrs["n_theta"], h.attrs["n_zeta"] = n_rho, n_theta, n_zeta
        h.attrs["nfp"] = int(nfp)
        h.attrs["synthetic"] = "mrx.synthetic_gvec.write_synthetic_gvec"
        for name in ("R0", "a", "Phi_edge", "iota0", "iota1", "lam_amplitude", "beta"):
            h.attrs[name] = getattr(torus, name)
        h.attrs["angle_units"] = "radians"
        h.attrs["zeta_convention"] = "per_field_period"
        h.attrs["radial_label"] = "rho = sqrt(s)"
        h.attrs["clebsch_contract"] = ("sqrt(g) B^rho = 0; sqrt(g) B^theta = dchi_dr "
                                       "- dPhi_dr * dLA_dz; sqrt(g) B^zeta = dPhi_dr "
                                       "* (1 + dLA_dt)")
        h["eval_points"] = np.stack([RHO.ravel(), TH.ravel(), ZE.ravel()], axis=1)
        for name, values in flat.items():
            h[name] = np.ascontiguousarray(np.asarray(values, dtype=np.float64).ravel())
    return torus
