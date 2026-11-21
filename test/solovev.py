"""The Solov'ev tokamak, an exact axisymmetric equilibrium, written as a VMEC wout for tests.

The poloidal flux per radian is

    psi(R, Z) = a R^2 Z^2 + b (R^2 - R0^2)^2,    b = kappa^2 a / 4,

and the field is ``B = F grad phi + grad psi x grad phi`` with a constant ``F``. Its Grad-Shafranov operator is
``Delta* psi = (8 b + 2 a) R^2``, so with ``F F' = 0`` the field balances the pressure

    mu0 p = (8 b + 2 a) (psi_b - psi),

which vanishes on the boundary ``psi = psi_b``, and its current is ``mu0 J = curl B = -(8 b + 2 a) R e_phi``. The
magnetic axis is the circle ``R = R0, Z = 0``, where ``psi`` is zero, and the poloidal field ``(-psi_Z e_R + psi_R
e_Z) / R`` turns counter-clockwise in the ``(R, Z)`` plane.

The surfaces are found on rays from the axis, ``R = R0 + ell cos u``, ``Z = ell sin u``, with ``ell(psi, u)`` the root
of ``psi(R0 + ell cos u, ell sin u) = psi``. Along a ray ``d psi / d ell = R B_pol . e_u``, so the toroidal flux per
poloidal flux is the loop integral

    Phi'(psi) = F int_0^2pi ell / (R d psi / d ell) du,

the transform is ``iota = 2 pi / Phi'``, and the straight-field-line angle ``theta*`` grows by
``iota F ell / (R d psi / d ell)`` per unit ``u``. The wout writes every surface in ``theta*``, so its lambda is zero. The
polar angle ``u`` would not do: on the elongated surfaces near the axis it differs from ``theta*`` by a function of
``u`` that does not vanish at the axis, where the reader sets lambda's ``m > 0`` modes to zero. Every integral in
``u`` is a trapezoidal sum of a smooth periodic function and every function of ``psi`` a Chebyshev interpolant, so
the wout's data are exact to round-off up to its radial and Fourier truncation. Like VMEC's, its angles
``(s, theta*, phi)`` are left-handed and MRX reads them reversed.
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial import chebyshev as cheb
from scipy.io import netcdf_file

MU0 = 4e-7 * np.pi
#: Points of the loop integrals in ``u`` and the degree of the interpolants in ``psi``.
N_U, DEG_PSI = 512, 48


class Solovev:
    """The equilibrium with the axis at ``R0``, toroidal field function ``F`` (``F / R0`` the toroidal field on the axis),
    poloidal flux scale ``a`` and elongation ``kappa`` near the axis, bounded by the surface through ``(R0 + x_b, 0)``.
    The defaults give a transform of 0.8 on the axis and 0.65 on the boundary and 10 % beta on the axis."""

    def __init__(self, R0=1.0, F=0.5, a=2.0 / 15.0, kappa=1.5, x_b=0.2):
        self.R0, self.F, self.a = R0, F, a
        self.b = kappa ** 2 * self.a / 4.0
        self.c = 8.0 * self.b + 2.0 * self.a
        self.psi_b = self.psi(R0 + x_b, 0.0)
        self.u = 2.0 * np.pi * np.arange(N_U) / N_U
        dphi = cheb.Chebyshev.interpolate(self._dphi, DEG_PSI, domain=[0.0, self.psi_b])
        self._dphi_c, self._phi_c = dphi, dphi.integ(lbnd=0.0)
        self.phi_b = float(self._phi_c(self.psi_b))

    # closed forms in the cylinder (R, phi, Z)
    def psi(self, R, Z):
        return self.a * R ** 2 * Z ** 2 + self.b * (R ** 2 - self.R0 ** 2) ** 2

    def grad_psi(self, R, Z):
        return (2.0 * self.a * R * Z ** 2 + 4.0 * self.b * R * (R ** 2 - self.R0 ** 2), 2.0 * self.a * R ** 2 * Z)

    def mu0_p(self, R, Z):
        """``mu0 p`` in T^2."""
        return self.c * (self.psi_b - self.psi(R, Z))

    def field(self, X):
        """``B`` at the Cartesian points ``X`` (rows ``x, y, z``), Cartesian."""
        R, Z = np.hypot(X[:, 0], X[:, 1]), X[:, 2]
        cos, sin = X[:, 0] / R, X[:, 1] / R
        pR, pZ = self.grad_psi(R, Z)
        B_R, B_phi, B_Z = -pZ / R, self.F / R, pR / R
        return np.stack([B_R * cos - B_phi * sin, B_R * sin + B_phi * cos, B_Z], axis=1)

    def current(self, X):
        """``mu0 J = curl B`` at the Cartesian points ``X``, Cartesian."""
        R = np.hypot(X[:, 0], X[:, 1])
        J_phi = -self.c * R
        return np.stack([-J_phi * X[:, 1] / R, J_phi * X[:, 0] / R, np.zeros_like(R)], axis=1)

    # the flux coordinates
    def s(self, psi):
        """The normalised toroidal flux of the surface ``psi``."""
        return self._phi_c(psi) / self.phi_b

    def s_at(self, X):
        """The normalised toroidal flux at the Cartesian points ``X``."""
        return self.s(self.psi(np.hypot(X[:, 0], X[:, 1]), X[:, 2]))

    def psi_of_s(self, s):
        """The surface ``psi`` of normalised toroidal flux ``s``, by Newton's method on ``Phi(psi) = s Phi_b``."""
        s = np.asarray(s, dtype=float)
        psi = s * self.psi_b
        for _ in range(60):
            psi = np.clip(psi - (self._phi_c(psi) - s * self.phi_b) / self._dphi_c(psi), 0.0, self.psi_b)
        return psi

    def iota(self, s):
        """The rotational transform per full turn on the surface ``s``, positive: the field lines turn with ``u``."""
        return 2.0 * np.pi / self._dphi_c(self.psi_of_s(s))

    def _ray(self, psi, u=None):
        """``(ell, ell / (d psi / d ell), R)`` on the rays ``u`` (shape ``(len(psi), n)``, by default the uniform ``self.u``
        on every surface) of the surfaces ``psi``."""
        psi = np.atleast_1d(np.asarray(psi, dtype=float))[:, None]
        u = self.u[None, :] if u is None else u
        cu, su = np.cos(u), np.sin(u)
        a, b, R0 = self.a, self.b, self.R0

        def g(ell):
            return a * su ** 2 * (R0 + ell * cu) ** 2 + b * cu ** 2 * (2.0 * R0 + ell * cu) ** 2

        def dg(ell):
            return 2.0 * a * su ** 2 * cu * (R0 + ell * cu) + 2.0 * b * cu ** 3 * (2.0 * R0 + ell * cu)
        ell = np.sqrt(psi / g(0.0))
        for _ in range(60):
            ell = ell - (ell ** 2 * g(ell) - psi) / (2.0 * ell * g(ell) + ell ** 2 * dg(ell) + 1e-300)
        # psi grows along every ray: the surfaces are star-shaped about the axis
        assert np.all(2.0 * g(ell) + ell * dg(ell) > 0.0)
        return ell, 1.0 / (2.0 * g(ell) + ell * dg(ell)), R0 + ell * cu

    def _dphi(self, psi):
        """``Phi'(psi) = F int ell / (R d psi / d ell) du`` (the toroidal flux, not divided by 2 pi, per ``psi``)."""
        _, l_over_dl, R = self._ray(psi)
        return self.F * np.mean(l_over_dl / R, axis=1) * 2.0 * np.pi

    def surfaces(self, s, mpol):
        """``(rmnc, zmns)`` of the surfaces ``s`` in the straight-field-line angle, modes ``m = 0 .. mpol - 1``."""
        s = np.asarray(s, dtype=float)
        psi, iota = self.psi_of_s(s), self.iota(s)
        m = np.arange(mpol)
        k = np.arange(1, N_U // 2)
        # theta*(u) - u as a sine series in u, from its rate on the uniform rays
        _, l_over_dl, R = self._ray(psi)
        rate = iota[:, None] * self.F * l_over_dl / R
        c = 2.0 * (rate - 1.0) @ np.cos(np.outer(self.u, k)) / N_U
        lam = c / k

        def theta_star(u):
            return u + np.einsum("jk,jnk->jn", lam, np.sin(u[..., None] * k))
        # u at uniform theta*, by Newton's method with the rate as derivative
        u = np.broadcast_to(self.u, (s.size, N_U)).copy()
        for _ in range(30):
            _, l_over_dl, R = self._ray(psi, u)
            u = u - (theta_star(u) - self.u[None, :]) / (iota[:, None] * self.F * l_over_dl / R)
        ell, _, R = self._ray(psi, u)
        Z = ell * np.sin(u)
        cos, sin = np.cos(np.outer(self.u, m)), np.sin(np.outer(self.u, m))
        w = np.where(m == 0, 1.0, 2.0) / N_U
        return (R @ cos) * w, (Z @ sin) * w

    def write_wout(self, path, ns=201, mpol=24):
        """Write the wout of the equilibrium on ``ns`` surfaces and ``mpol`` poloidal modes to ``path``."""
        s_full = np.arange(ns) / (ns - 1)
        rmnc, zmns = self.surfaces(s_full, mpol)
        lmns = np.zeros_like(rmnc)
        with netcdf_file(path, "w") as f:
            f.createDimension("radius", ns)
            f.createDimension("mn_mode", mpol)
            for k, v in (("ns", ns), ("nfp", 1), ("mnmax", mpol), ("lasym__logical__", 0)):
                f.createVariable(k, "i4", ())[()] = v
            f.createVariable("version_", "f8", ())[()] = 9.0
            f.createVariable("xm", "f8", ("mn_mode",))[:] = np.arange(mpol)
            f.createVariable("xn", "f8", ("mn_mode",))[:] = 0.0
            for k, v in (("rmnc", rmnc), ("zmns", zmns), ("lmns", lmns)):
                f.createVariable(k, "f8", ("radius", "mn_mode"))[:] = v
            for k, v in (("phi", s_full * self.phi_b), ("iotaf", self.iota(s_full)),
                         ("presf", self.mu0_p(self.R0, 0.0) * (1.0 - self.psi_of_s(s_full) / self.psi_b) / MU0)):
                f.createVariable(k, "f8", ("radius",))[:] = v
        return path
