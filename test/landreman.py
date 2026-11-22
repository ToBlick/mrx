"""Landreman's exact 3D MHD equilibria with weakly sheared transform (Landreman 2026), the case shearedE, written as a
VMEC wout for the tests.

In units with ``mu0 = 1``, with ``w = x + i y``, ``K = sqrt(conj(w)^2 + eps)`` and ``Xi = w K + pi / 2 - S``:

    B_x + i B_y = i kappa e^{-i lam z} sin(Xi) / (2 K),   B_z = (kappa / lam) Re(e^{-i lam z} cos Xi),
    psi = (sin^2(lam z) + (lam B_z / kappa)^2) / 2,       p = (kappa / lam)^2 (k_b^2 / 2 - psi).

``J x B = grad p`` and ``div B = 0`` hold exactly. The flux surfaces are ``k = sqrt(2 psi) = const``, the plasma is
``k <= k_b``. The logical coordinates of the wout are MRX's: ``rho = k / k_b``, ``theta`` (in turns) the elliptic
angle ``chi / 2 pi`` of the surface, and the geometric toroidal angle ``phi``, two field periods. shearedE (``eps``
0.5, ``S`` 0.65, ``k_b`` 0.2, ``lam`` 6) is Mercier-stable, with the transform 1.3715 on the axis and 1.3387 on the
wall and 0.15 % beta. The magnetic axis is the curve ``k = 0``.

The closed forms and the wout export are those of the JCP paper's verification (branch ``paper``,
``scripts/paper_scripts/landreman_equilibria.py``), cut to this family. The toroidal flux ``Phi(rho)`` and the
transform ``iota(rho)`` are Landreman's one-dimensional quadratures.
"""
from __future__ import annotations

import numpy as np
from scipy.io import netcdf_file

TWO_PI = 2.0 * np.pi
NFP = 2


class Sheared:
    """The sheared equilibrium with the paper's parameters. ``field``, ``psi`` and ``mu0_p`` take an array module
    ``xp`` (numpy or jax.numpy), so the current is ``curl B`` by automatic differentiation."""

    def __init__(self, eps=0.5, S=0.65, k_b=0.2, lam=6.0, kappa=1.0, n_rho=401):
        self.eps, self.S, self.k_b, self.lam, self.kappa = eps, S, k_b, lam, kappa
        # Phi(rho) and iota(rho) on a fine rho grid, s = Phi / Phi_edge, MRX's label
        rho = np.linspace(0.0, 1.0, n_rho)
        q, a = self._derivs(k_b * np.maximum(rho, 1e-12))
        self._rho = rho
        self._iota = a / q
        dphi = q * k_b ** 2 * rho                                  # d Phi / d rho, Phi(0) = 0
        self._phi = np.concatenate([[0.0], np.cumsum(0.5 * (dphi[1:] + dphi[:-1]) * np.diff(rho))])
        self.phi_edge = float(self._phi[-1])
        self._s = self._phi / self.phi_edge

    # the closed forms in Cartesian (x, y, z)
    def field(self, x, xp=np):
        w = x[..., 0] + 1j * x[..., 1]
        wb = xp.conj(w)
        K = wb * xp.sqrt(1 + self.eps / wb ** 2)
        Xi = w * K + np.pi / 2 - self.S
        e = xp.exp(-1j * self.lam * x[..., 2])
        H = 1j * self.kappa * e * xp.sin(Xi) / (2 * K)
        return xp.stack([xp.real(H), xp.imag(H), self.kappa / self.lam * xp.real(e * xp.cos(Xi))], axis=-1)

    def psi(self, x, xp=np):
        return (xp.sin(self.lam * x[..., 2]) ** 2 + (self.lam * self.field(x, xp)[..., 2] / self.kappa) ** 2) / 2

    def mu0_p(self, x, xp=np):
        """``mu0 p``, zero on the wall."""
        return (self.kappa / self.lam) ** 2 * (self.k_b ** 2 / 2 - self.psi(x, xp))

    def current(self, x):
        """``mu0 J = curl B`` at the points ``x`` ``(n, 3)``, by automatic differentiation of the closed form."""
        import jax
        import jax.numpy as jnp
        dB = np.asarray(jax.vmap(jax.jacfwd(lambda y: self.field(y, jnp)))(jnp.asarray(x, dtype=jnp.float64)))
        return np.stack([dB[:, 2, 1] - dB[:, 1, 2], dB[:, 0, 2] - dB[:, 2, 0], dB[:, 1, 0] - dB[:, 0, 1]], axis=-1)

    # the flux coordinates
    def rho_at(self, x):
        return np.sqrt(np.maximum(2.0 * self.psi(x), 0.0)) / self.k_b

    def s_of_rho(self, rho):
        return np.interp(rho, self._rho, self._s)

    def s_at(self, x):
        """The normalised toroidal flux at the Cartesian points ``x``."""
        return self.s_of_rho(self.rho_at(x))

    def iota(self, s):
        """The transform per full turn on the surface ``s``, positive."""
        return np.interp(s, self._s, self._iota)

    def mu0_p_of_s(self, s):
        return (self.kappa / self.lam) ** 2 * self.k_b ** 2 / 2 * (1.0 - np.interp(s, self._s, self._rho) ** 2)

    def _derivs(self, k, n=1024):
        """Landreman's quadratures: ``Q'(k) / k`` and ``A'(k) / k`` of the toroidal and the poloidal flux, ``iota =
        A' / Q'``."""
        eps, S, lam, kap = self.eps, self.S, self.lam, self.kappa
        h = lambda s: np.sqrt(4 * s ** 2 + eps ** 2)       # noqa: E731
        k = np.atleast_1d(k)[:, None]
        u = np.arange(n) * (TWO_PI / n)
        c = np.sqrt(1 - k ** 2 * np.sin(u) ** 2)
        q = kap * np.pi / lam * np.mean(1 / (h(S + np.arcsin(k * np.cos(u) / c)) * c), axis=1)
        v = eps / 2 * np.sin(2 * u)
        sig = S + np.arcsin(k / np.cosh(v))
        g = (h(sig) + eps * np.cos(2 * u)) / 2
        a = TWO_PI * kap / lam * np.mean(g / (h(sig) * np.sqrt(np.cosh(v) ** 2 - k ** 2)), axis=1)
        return q, a

    # the logical parametrisation
    def point(self, rho, theta, phi):
        """The Cartesian point of ``(rho, theta, phi)``, ``theta`` in turns, at the geometric azimuth ``phi``: the
        confocal label ``sigma`` by bisection in its exact bracket ``S +- asin(k)``."""
        rho, theta, phi = np.broadcast_arrays(*(np.asarray(v, dtype=np.float64) for v in (rho, theta, phi)))
        eps = self.eps
        k = self.k_b * rho
        chi = TWO_PI * theta
        P, Y = -k * np.cos(chi), k * np.sin(chi)
        root = np.sqrt(1 - Y ** 2)

        def axes(sigma):
            b = np.sqrt((np.hypot(2 * sigma, eps) + eps) / 2)
            return sigma / b, b

        def residual(sigma):
            a, b = axes(sigma)
            t = np.arctan2(a * np.sin(phi), b * np.cos(phi))
            nu = eps / 2 * np.sin(2 * t)
            return (sigma - self.S - np.arctan(np.tanh(nu) * Y / root)
                    + np.arcsin(P / np.sqrt(np.cosh(nu) ** 2 - Y ** 2))), t

        lo, hi = self.S - np.arcsin(k) - 1e-14, self.S + np.arcsin(k) + 1e-14
        for _ in range(64):
            mid = 0.5 * (lo + hi)
            val, _ = residual(mid)
            lo, hi = np.where(val < 0, mid, lo), np.where(val >= 0, mid, hi)
        sigma = 0.5 * (lo + hi)
        _, t = residual(sigma)
        a, b = axes(sigma)
        return np.stack((a * np.cos(t), b * np.sin(t), -np.arcsin(Y) / self.lam), axis=-1)

    def axis(self, phi):
        """The magnetic axis at the azimuths ``phi``."""
        return self.point(np.zeros_like(phi), np.zeros_like(phi), phi)

    def _logical_2form(self, rho, theta, zeta, h=1e-4):
        """``J DF^-1 B`` of the exact field at logical points (``zeta`` in field periods), ``DF`` by fourth-order
        differences of the point."""
        def P(r, t, z):
            return self.point(r, t, TWO_PI * z / NFP)

        def fd(f, x):
            return (-f(x + 2 * h) + 8 * f(x + h) - 8 * f(x - h) + f(x - 2 * h)) / (12 * h)
        DF = np.stack([fd(lambda r: P(r, theta, zeta), rho), fd(lambda t: P(rho, t, zeta), theta),
                       fd(lambda z: P(rho, theta, z), zeta)], axis=-1)
        B = self.field(P(rho, theta, zeta))
        return np.linalg.det(DF)[..., None] * np.linalg.solve(DF, B[..., None])[..., 0]

    def surfaces(self, rho, n_theta=32, n_zeta=64):
        """``(xm, xn, rmnc, zmns)`` of the surfaces ``rho``: the exact DFT of ``R``, ``Z`` on a uniform grid, ``m <
        n_theta / 2``, ``|n| < n_zeta / 2`` field periods (argument ``m theta - n nfp phi``, radians)."""
        th, ze = np.arange(n_theta) / n_theta, np.arange(n_zeta) / n_zeta
        Rg, Tg, Zg = np.meshgrid(rho, th, ze, indexing="ij")
        X = self.point(Rg, Tg, TWO_PI * Zg / NFP)
        R, Z = np.hypot(X[..., 0], X[..., 1]), X[..., 2]
        modes = [(m, n) for m in range(n_theta // 2) for n in range(-(n_zeta // 2) + 1, n_zeta // 2) if m > 0 or n >= 0]
        xm, xn = (np.array([v[i] for v in modes], float) for i in (0, 1))
        arg = TWO_PI * (xm[:, None, None] * th[None, :, None] - xn[:, None, None] * ze[None, None, :])
        norm = np.where((xm == 0) & (xn == 0), 1.0, 2.0)
        rmnc = np.einsum("rtz,ktz->rk", R, np.cos(arg)) / (n_theta * n_zeta) * norm
        zmns = np.einsum("rtz,ktz->rk", Z, np.sin(arg)) / (n_theta * n_zeta) * norm
        return xm, NFP * xn, rmnc, zmns

    def _lambda(self, rho, xm, xn, n_theta=32, n_zeta=64):
        """``lmns`` (radians) of the surfaces ``rho`` from the exact field: ``B_hat^zeta = Phi' (1 + d lambda / d
        theta)`` for ``m > 0``, the ``m = 0`` modes from the ``theta`` mean of ``B_hat^theta``."""
        th, ze = np.arange(n_theta) / n_theta, np.arange(n_zeta) / n_zeta
        Rg, Tg, Zg = np.meshgrid(rho, th, ze, indexing="ij")
        Bh = self._logical_2form(Rg, Tg, Zg)
        dphi = Bh[..., 2].mean(axis=1)
        iota = NFP * Bh[..., 1].mean(axis=(1, 2)) / Bh[..., 2].mean(axis=(1, 2))
        g = Bh[..., 2] / dphi.mean(axis=1)[:, None, None] - 1.0
        arg = TWO_PI * (xm[:, None, None] * th[None, :, None] - (xn / NFP)[:, None, None] * ze[None, None, :])
        coef = 2.0 * np.einsum("rtz,ktz->rk", g, np.cos(arg)) / (n_theta * n_zeta)
        lmns = np.where(xm[None, :] > 0, coef / np.where(xm > 0, xm, 1.0)[None, :], 0.0)
        h0 = iota[:, None] - NFP * Bh[..., 1].mean(axis=1) / dphi.mean(axis=1)[:, None]
        m0 = (xm == 0) & (xn > 0)
        c0 = 2.0 * np.einsum("rz,kz->rk", h0, np.cos(TWO_PI * (xn[m0] / NFP)[:, None] * ze[None, :])) / n_zeta
        lmns[:, m0] = -c0 / xn[m0][None, :]
        return lmns

    def write_wout(self, path, ns=51, n_theta=32, n_zeta=64):
        """Write the wout of the equilibrium on ``ns`` full-mesh surfaces ``s_j = j / (ns - 1)`` (VMEC's label, the
        paper's ``rho^2``), ``lmns`` on the half mesh, ``phi`` (the toroidal flux), ``iotaf`` and ``presf`` (``p``
        in Pa of a field in tesla is ``mu0 p / mu0``). Returns ``path``."""
        s = np.arange(ns) / (ns - 1)
        rho = np.sqrt(s)
        xm, xn, rmnc, zmns = self.surfaces(rho, n_theta, n_zeta)
        rho_half = np.sqrt((np.arange(1, ns) - 0.5) / (ns - 1))
        lmns = np.vstack([np.zeros((1, len(xm))), self._lambda(rho_half, xm, xn, n_theta, n_zeta)])
        phi = np.interp(rho, self._rho, self._phi)
        q, a = self._derivs(self.k_b * np.maximum(rho, 1e-12))
        iotaf = a / q
        presf = (self.kappa / self.lam) ** 2 * self.k_b ** 2 / 2 * (1.0 - s) / (4e-7 * np.pi)
        with netcdf_file(path, "w") as f:
            f.createDimension("radius", ns)
            f.createDimension("mn_mode", len(xm))

            def var(name, data, dims):
                f.createVariable(name, np.float64 if np.asarray(data).dtype.kind == "f" else np.int32, dims)[...] = data
            var("ns", np.int32(ns), ())
            var("nfp", np.int32(NFP), ())
            var("mnmax", np.int32(len(xm)), ())
            var("lasym__logical__", np.int32(0), ())
            var("version_", np.float64(9.0), ())
            var("xm", xm, ("mn_mode",))
            var("xn", xn, ("mn_mode",))
            var("rmnc", rmnc, ("radius", "mn_mode"))
            var("zmns", zmns, ("radius", "mn_mode"))
            var("lmns", lmns, ("radius", "mn_mode"))
            phipf = np.gradient(phi, s, edge_order=2)
            var("phi", phi, ("radius",))
            var("phipf", phipf, ("radius",))
            var("chipf", iotaf * phipf, ("radius",))
            var("iotaf", iotaf, ("radius",))
            var("presf", presf, ("radius",))
        return path

    def write_namelist(self, path, n_theta=32, n_zeta=64, n_profile=41):
        """A VMEC input namelist of the boundary (the surface ``rho = 1``), its toroidal flux, the exact pressure and
        the exact transform as cubic splines in ``s`` (``PMASS_TYPE`` and ``PIOTA_TYPE``, ``NCURR = 0``)."""
        xm, xn, rmnc, zmns = self.surfaces(np.array([1.0]), n_theta, n_zeta)
        s = np.linspace(0.0, 1.0, n_profile)
        row = lambda v: " ".join(f"{x:.17g}" for x in v)  # noqa: E731
        lines = [f"NFP = {NFP}", f"PHIEDGE = {self.phi_edge!r}", "NCURR = 0",
                 "PMASS_TYPE = 'cubic_spline'", f"AM_AUX_S = {row(s)}",
                 f"AM_AUX_F = {row(self.mu0_p_of_s(s) / (4e-7 * np.pi))}",
                 "PIOTA_TYPE = 'cubic_spline'", f"AI_AUX_S = {row(s)}", f"AI_AUX_F = {row(self.iota(s))}"]
        lines += [f"RBC({int(n) // NFP},{int(m)}) = {r!r}  ZBS({int(n) // NFP},{int(m)}) = {z!r}"
                  for m, n, r, z in zip(xm, xn, rmnc[0], zmns[0]) if abs(r) > 1e-14 or abs(z) > 1e-14]
        with open(path, "w") as fh:
            fh.write("&INDATA\n" + "\n".join(lines) + "\n/\n")
        return path
