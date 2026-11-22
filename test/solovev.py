"""Solov'ev tokamaks, exact axisymmetric equilibria with constant ``p'`` and ``F F'``, written as VMEC wouts for tests.

In units of the major radius ``R0 = 1`` the poloidal flux per radian is the particular solution of Cerfon & Freidberg
(Phys. Plasmas 17, 032502, 2010) without their homogeneous terms,

    psi(R, Z) = a R^2 Z^2 + b (R^2 - 1)^2 + d Z^2,    b = kappa^2 / 4,  a = 1 - A (1 + kappa^2),  d = A (1 + kappa^2),

and the field is ``B = F(psi) grad phi + grad psi x grad phi``. Its Grad-Shafranov operator is

    Delta* psi = 2 (1 + kappa^2) ((1 - A) R^2 + A) = -R^2 mu0 p'(psi) - F F'(psi),

Cerfon and Freidberg's right-hand side with their ``A``, the share of ``F F'``. So

    mu0 p = 2 (1 + kappa^2) (1 - A) (psi_b - psi),    F^2 = F0^2 - 4 d psi,
    mu0 J = curl B = -(2 (1 + kappa^2)(1 - A) R + 2 d / R) e_phi + F'(psi) grad psi x grad phi.

``A = 0`` has ``F F' = 0`` (the classical Solov'ev equilibrium, ``solovev20`` of the exact_landreman line with
``kappa = 1.5``, the boundary through ``R = 1.25`` and ``F0 = 6``). The magnetic axis is the circle ``R = 1, Z = 0``,
where ``psi`` is zero, ``kappa`` is the elongation of the surfaces next to it (``a + d = 1``), ``F0 / R`` the toroidal
field there, and the poloidal field turns counter-clockwise in the ``(R, Z)`` plane.

The surfaces are found on elliptic rays from the axis, ``R = 1 + ell cos u``, ``Z = e ell sin u``, with
``ell(psi, u)`` the root of ``psi(1 + ell cos u, e ell sin u) = psi``. The wout's angle is ``u`` with ``e = kappa``, the
elliptic angle of the surfaces next to the axis (``angle="elliptic"``, the default), or the straight-field-line angle
``theta*`` (``angle="straight"``, lambda zero). With ``e`` the Jacobian of ``(ell, u) -> (R, Z)`` is ``e ell``, so the
toroidal flux per poloidal flux is the loop integral

    Phi'(psi) = F(psi) e int_0^2pi ell / (R d psi / d ell) du,

the transform is ``iota = 2 pi / Phi'``, and ``theta*`` grows by ``iota F e ell / (R d psi / d ell)`` per unit ``u``. In
the elliptic angle that rate is constant at the axis, so lambda vanishes there, as the reader assumes, and the map is
smooth there: the ``m != 1`` content of ``(X - X_axis) / r`` falls like ``r``. ``theta*`` is smooth at the axis too, but
on solovev20 its map carries 3.7 times the elliptic angle's ``m != 1`` content, and MRX's projected field has twice its
error and 70 times its force residual at (14,24,10) (2026-11-21). The polar angle (``e = 1``) is not smooth: on
elongated surfaces it differs from ``theta*`` at the axis itself. Every integral in ``u`` is a trapezoidal sum of a
smooth periodic function and every function of ``psi`` a Chebyshev interpolant, so the wout's data are exact to
round-off up to its radial and Fourier truncation. Like VMEC's, its angles are left-handed and MRX reads them reversed.
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial import chebyshev as cheb
from scipy.io import netcdf_file

MU0 = 4e-7 * np.pi
#: Points of the loop integrals in ``u`` and the degree of the interpolants in ``psi``.
N_U, DEG_PSI = 512, 48


class Solovev:
    """The equilibrium with the share ``A`` of ``F F'`` in the current (Cerfon and Freidberg's ``A``), elongation
    ``kappa`` next to the axis, the boundary through ``(1 + x_b, 0)`` and ``F0 = F(0)``, written in the ``angle``
    ``"elliptic"`` or ``"straight"``. The defaults (``A = 0``) give a transform of 0.8 on the axis and 0.65 on the
    boundary and 10 % beta on the axis."""

    def __init__(self, A=0.0, kappa=1.5, x_b=0.2, F0=3.75, angle="elliptic"):
        self.A, self.F0 = float(A), float(F0)
        self.e = {"elliptic": float(kappa), "straight": 1.0}[angle]
        self.straight = angle == "straight"
        self.b = kappa ** 2 / 4.0
        self.a = 1.0 - A * (1.0 + kappa ** 2)
        self.d = A * (1.0 + kappa ** 2)
        self.c = 2.0 * (1.0 + kappa ** 2) * (1.0 - A)           # mu0 p = c (psi_b - psi)
        self.psi_b = self.psi(1.0 + x_b, 0.0)
        assert np.all(self.F(np.array([self.psi_b])) > 0.0)
        self.u = 2.0 * np.pi * np.arange(N_U) / N_U
        dphi = cheb.Chebyshev.interpolate(self._dphi, DEG_PSI, domain=[0.0, self.psi_b])
        self._dphi_c, self._phi_c = dphi, dphi.integ(lbnd=0.0)
        self.phi_b = float(self._phi_c(self.psi_b))

    # closed forms in the cylinder (R, phi, Z)
    def psi(self, R, Z):
        return self.a * R ** 2 * Z ** 2 + self.b * (R ** 2 - 1.0) ** 2 + self.d * Z ** 2

    def grad_psi(self, R, Z):
        return (2.0 * self.a * R * Z ** 2 + 4.0 * self.b * R * (R ** 2 - 1.0),
                2.0 * self.a * R ** 2 * Z + 2.0 * self.d * Z)

    def F(self, psi):
        """The toroidal field function ``R B_phi`` on the surface ``psi``."""
        return np.sqrt(self.F0 ** 2 - 4.0 * self.d * psi)

    def mu0_p(self, R, Z):
        """``mu0 p`` in T^2."""
        return self.c * (self.psi_b - self.psi(R, Z))

    def field(self, X):
        """``B`` at the Cartesian points ``X`` (rows ``x, y, z``), Cartesian."""
        R, Z = np.hypot(X[:, 0], X[:, 1]), X[:, 2]
        cos, sin = X[:, 0] / R, X[:, 1] / R
        pR, pZ = self.grad_psi(R, Z)
        B_R, B_phi, B_Z = -pZ / R, self.F(self.psi(R, Z)) / R, pR / R
        return np.stack([B_R * cos - B_phi * sin, B_R * sin + B_phi * cos, B_Z], axis=1)

    def current(self, X):
        """``mu0 J = curl B`` at the Cartesian points ``X``, Cartesian."""
        R, Z = np.hypot(X[:, 0], X[:, 1]), X[:, 2]
        cos, sin = X[:, 0] / R, X[:, 1] / R
        pR, pZ = self.grad_psi(R, Z)
        dF = -2.0 * self.d / self.F(self.psi(R, Z))                  # F' = -2 d / F
        J_R, J_phi, J_Z = -dF * pZ / R, -(self.c * R + 2.0 * self.d / R), dF * pR / R
        return np.stack([J_R * cos - J_phi * sin, J_R * sin + J_phi * cos, J_Z], axis=1)

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
        """``(ell, ell / (d psi / d ell), R)`` on the rays ``(1 + ell cos u, e ell sin u)`` (``u`` of shape
        ``(len(psi), n)``, by default the uniform ``self.u`` on every surface) of the surfaces ``psi``."""
        psi = np.atleast_1d(np.asarray(psi, dtype=float))[:, None]
        u = self.u[None, :] if u is None else u
        cu, su = np.cos(u), self.e * np.sin(u)
        a, b, d = self.a, self.b, self.d

        def g(ell):                     # psi = ell^2 g(ell) along the ray
            return a * su ** 2 * (1.0 + ell * cu) ** 2 + b * cu ** 2 * (2.0 + ell * cu) ** 2 + d * su ** 2

        def dg(ell):
            return 2.0 * a * su ** 2 * cu * (1.0 + ell * cu) + 2.0 * b * cu ** 3 * (2.0 + ell * cu)
        ell = np.sqrt(psi / g(0.0))
        for _ in range(60):
            ell = ell - (ell ** 2 * g(ell) - psi) / (2.0 * ell * g(ell) + ell ** 2 * dg(ell) + 1e-300)
        # psi grows along every ray: the surfaces are star-shaped about the axis
        assert np.all(2.0 * g(ell) + ell * dg(ell) > 0.0)
        return ell, 1.0 / (2.0 * g(ell) + ell * dg(ell)), 1.0 + ell * cu

    def _dphi(self, psi):
        """``Phi'(psi) = F e int ell / (R d psi / d ell) du`` (the toroidal flux, not divided by 2 pi, per ``psi``)."""
        _, l_over_dl, R = self._ray(psi)
        return self.F(np.atleast_1d(psi)) * self.e * np.mean(l_over_dl / R, axis=1) * 2.0 * np.pi

    def surfaces(self, s, mpol):
        """``(rmnc, zmns, lmns)`` of the surfaces ``s`` in the wout's angle, modes ``m = 0 .. mpol - 1``."""
        s = np.asarray(s, dtype=float)
        psi, iota = self.psi_of_s(s), self.iota(s)
        rate_scale = (iota * self.F(psi) * self.e)[:, None]         # d theta* / du = iota F e ell / (R d psi / d ell)
        m = np.arange(mpol)
        k = np.arange(1, N_U // 2)
        # theta*(u) - u as a sine series in u, from its rate on the uniform rays
        _, l_over_dl, R = self._ray(psi)
        lam = 2.0 * (rate_scale * l_over_dl / R - 1.0) @ np.cos(np.outer(self.u, k)) / N_U / k
        u = np.broadcast_to(self.u, (s.size, N_U)).copy()
        if self.straight:
            # u at uniform theta*, by Newton's method with the rate as derivative
            for _ in range(30):
                _, l_over_dl, R = self._ray(psi, u)
                theta_star = u + np.einsum("jk,jnk->jn", lam, np.sin(u[..., None] * k))
                u = u - (theta_star - self.u[None, :]) / (rate_scale * l_over_dl / R)
        ell, _, R = self._ray(psi, u)
        Z = self.e * ell * np.sin(u)
        cos, sin = np.cos(np.outer(self.u, m)), np.sin(np.outer(self.u, m))
        w = np.where(m == 0, 1.0, 2.0) / N_U
        lmns = np.zeros((s.size, mpol)) if self.straight else np.pad(lam, ((0, 0), (1, 0)))[:, :mpol]
        return (R @ cos) * w, (Z @ sin) * w, lmns

    def point(self, rho, theta, phi, mpol=40):
        """The Cartesian point of the logical coordinates of the wout, ``rho = sqrt(s)``, ``theta`` (the wout's angle
        over 2 pi, counter-clockwise) and ``phi``, each a 1D array of the same length."""
        rho, theta, phi = (np.atleast_1d(np.asarray(v, dtype=float)) for v in (rho, theta, phi))
        rmnc, zmns, _ = self.surfaces(rho ** 2, mpol)
        arg = 2.0 * np.pi * np.outer(theta, np.arange(mpol))
        R, Z = np.sum(np.cos(arg) * rmnc, axis=1), np.sum(np.sin(arg) * zmns, axis=1)
        return np.stack([R * np.cos(phi), R * np.sin(phi), Z], axis=1)

    def write_wout(self, path, ns=201, mpol=40):
        """Write the wout of the equilibrium on ``ns`` surfaces and ``mpol`` poloidal modes to ``path``. ``presf`` is
        the pressure in Pa of a field in tesla."""
        s_full = np.arange(ns) / (ns - 1)
        rmnc, zmns, _ = self.surfaces(s_full, mpol)
        # lambda on VMEC's half mesh, its first row unused
        lmns = self.surfaces(np.clip((np.arange(ns) - 0.5) / (ns - 1), 0.0, 1.0), mpol)[2]
        lmns[0] = 0.0
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
                         ("presf", self.c * (self.psi_b - self.psi_of_s(s_full)) / MU0)):
                f.createVariable(k, "f8", ("radius",))[:] = v
        return path
