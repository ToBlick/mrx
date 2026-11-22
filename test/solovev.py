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

The surfaces are found on elliptic rays from the axis, ``R = 1 + ell cos u``, ``Z = kappa ell sin u``, with
``ell(psi, u)`` the root of ``psi(1 + ell cos u, kappa ell sin u) = psi``, and ``u`` is the wout's angle, the elliptic
angle of the surfaces next to the axis. The Jacobian of ``(ell, u) -> (R, Z)`` is ``kappa ell``, so the toroidal flux
per poloidal flux is the loop integral

    Phi'(psi) = F(psi) kappa int_0^2pi ell / (R d psi / d ell) du,

the transform is ``iota = 2 pi / Phi'``, and the straight-field-line angle ``theta* = u + lambda`` grows by
``iota F kappa ell / (R d psi / d ell)`` per unit ``u``. That rate is constant at the axis, so lambda vanishes there, as
the reader assumes, and the map is smooth there: the ``m != 1`` content of ``(X - X_axis) / r`` falls like ``r``.
``theta*`` itself is no choice for the wout's angle: on the boundary it runs up to 17 times faster in some places than
in others (low aspect ratio, ``F F'``-driven current), and the spline map at (18,32,6) folds there (2026-11-21). The
polar angle is not smooth at the axis: on elongated surfaces it differs from ``theta*`` at the axis itself. Every
integral in ``u`` is a trapezoidal sum of a smooth periodic function and every function of ``psi`` a Chebyshev
interpolant, so the wout's data are exact to round-off up to its radial and Fourier truncation. Like VMEC's, its
angles are left-handed and MRX reads them reversed.
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
    ``kappa`` next to the axis, the boundary through ``(1 + x_b, 0)`` and ``F0 = F(0)``. The defaults (``A = 0``) give a transform of 0.8 on the axis and 0.65 on the
    boundary and 10 % beta on the axis."""

    def __init__(self, A=0.0, kappa=1.5, x_b=0.2, F0=3.75):
        self.A, self.F0 = float(A), float(F0)
        self.kappa = float(kappa)
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
        """``(ell, ell / (d psi / d ell), R)`` on the rays ``(1 + ell cos u, kappa ell sin u)`` (``u`` of shape
        ``(len(psi), n)``, by default the uniform ``self.u`` on every surface) of the surfaces ``psi``."""
        psi = np.atleast_1d(np.asarray(psi, dtype=float))[:, None]
        u = self.u[None, :] if u is None else u
        cu, su = np.cos(u), self.kappa * np.sin(u)
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
        """``Phi'(psi) = F kappa int ell / (R d psi / d ell) du`` (the toroidal flux, not divided by 2 pi, per ``psi``)."""
        _, l_over_dl, R = self._ray(psi)
        return self.F(np.atleast_1d(psi)) * self.kappa * np.mean(l_over_dl / R, axis=1) * 2.0 * np.pi

    def surfaces(self, s, mpol):
        """``(rmnc, zmns, lmns)`` of the surfaces ``s`` in the wout's angle, modes ``m = 0 .. mpol - 1``."""
        s = np.asarray(s, dtype=float)
        psi, iota = self.psi_of_s(s), self.iota(s)
        rate_scale = (iota * self.F(psi) * self.kappa)[:, None]     # d theta* / du = iota F kappa ell / (R d psi / d ell)
        m = np.arange(mpol)
        k = np.arange(1, N_U // 2)
        # lambda = theta*(u) - u as a sine series in u, from its rate on the uniform rays
        ell, l_over_dl, R = self._ray(psi)
        lam = 2.0 * (rate_scale * l_over_dl / R - 1.0) @ np.cos(np.outer(self.u, k)) / N_U / k
        Z = self.kappa * ell * np.sin(self.u)[None, :]
        cos, sin = np.cos(np.outer(self.u, m)), np.sin(np.outer(self.u, m))
        w = np.where(m == 0, 1.0, 2.0) / N_U
        lmns = np.pad(lam, ((0, 0), (1, 0)))[:, :mpol]
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

    def mu0_p_of_s(self, s):
        """``mu0 p`` in T^2 on the surface ``s``."""
        return self.c * (self.psi_b - self.psi_of_s(s))

    def axis(self, phi):
        """The magnetic axis ``R = 1``, ``Z = 0`` at the azimuths ``phi``, Cartesian."""
        phi = np.asarray(phi, dtype=float)
        return np.stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)], axis=-1)

    def write_namelist(self, path, mpol=40, n_profile=41):
        """A VMEC input namelist of the boundary ``s = 1``, its toroidal flux, the exact pressure and the exact transform
        as cubic splines in ``s`` (``NCURR = 0``). Returns ``path``."""
        rmnc, zmns, _ = self.surfaces(np.array([1.0]), mpol)
        s = np.linspace(0.0, 1.0, n_profile)
        row = lambda v: " ".join(f"{float(x):.17g}" for x in v)  # noqa: E731
        lines = ["NFP = 1", f"PHIEDGE = {float(self.phi_b)!r}", "NCURR = 0",
                 "PMASS_TYPE = 'cubic_spline'", f"AM_AUX_S = {row(s)}", f"AM_AUX_F = {row(self.mu0_p_of_s(s) / MU0)}",
                 "PIOTA_TYPE = 'cubic_spline'", f"AI_AUX_S = {row(s)}", f"AI_AUX_F = {row(self.iota(s))}"]
        lines += [f"RBC(0,{m}) = {float(r)!r}  ZBS(0,{m}) = {float(z)!r}" for m, (r, z) in enumerate(zip(rmnc[0], zmns[0]))
                  if abs(r) > 1e-14 or abs(z) > 1e-14]
        with open(path, "w") as fh:
            fh.write("&INDATA\n" + "\n".join(lines) + "\n/\n")
        return path
