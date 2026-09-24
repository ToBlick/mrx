"""Landreman's exact 3-D MHD equilibria (arXiv:2609.26742) in closed form, and their VMEC wout export.

Paper tooling, not production API. Two families, both ``nfp = 2`` and stellarator symmetric, in the paper's
units (``mu_0 = 1``, ``curl B x B = grad p``):

* ``iota2``: uniform ``iota = 2`` (a stretched Solov'ev field). ``a = sqrt(1 + eps)``, ``b = sqrt(1 - eps)``,
  ``s = x^2/a^2 + y^2/b^2``, ``F = sqrt(1 - (1 - s)^2 - 4 z^2)``,
  ``B = ((2 z x - (a/b) F y) / s, (2 z y + (b/a) F x) / s, 1 - s)``,
  ``psi = (x^2 + y^2 + 4 z^2 + |B|^2 - 2 + eps^2) / 4``, ``p = p_a - 2 psi``; the field lines are the ``t``-curves
  of the embedding ``r(u, v, t)`` (``B = dr/dt``) and the flux surfaces are ``psi = (u + eps/2)^2 + v^2``;
  ``Omega_delta = {psi <= delta}``.
* ``sheared``: weakly sheared iota. ``w = x + i y``, ``K = sqrt(conj(w)^2 + eps)``, ``Xi = w K + pi/2 - S``,
  ``B_x + i B_y = i kappa e^{-i lam z} sin(Xi) / (2 K)``, ``B_z = (kappa/lam) Re(e^{-i lam z} cos Xi)``,
  ``psi = (sin^2(lam z) + (lam B_z / kappa)^2) / 2``, ``p = p_a - kappa^2 psi / lam^2``; flux surfaces
  ``X^2 + Y^2 = k^2 = 2 psi`` with ``X = lam B_z / kappa = -k cos chi``, ``Y = -sin(lam z) = k sin chi``, the
  confocal label ``sigma = S + atan(tanh(nu) Y / sqrt(1 - Y^2)) - asin(X / sqrt(cosh^2 nu - Y^2))``,
  ``nu = (eps/2) sin 2t``, ``x = a_c(sigma) cos t``, ``y = b_c(sigma) sin t``, ``z = -asin(Y) / lam``,
  ``a_c b_c = sigma``, ``b_c = sqrt((h + eps)/2)``, ``h = sqrt(4 sigma^2 + eps^2)``; ``Omega_delta = {k <= k_b}``,
  ``delta = k_b^2 / 2``.

The logical coordinates of the export are MRX's: ``rho`` the flux label (``iota2``: ``rho = sqrt(psi/delta)``,
``sheared``: ``rho = k / k_b``), ``theta`` in turns, clockwise in the (R, Z) half plane so that ``(rho, theta,
phi)`` is right-handed and ``iota > 0``, and the GEOMETRIC toroidal angle ``phi`` (the wout's ``v``). For
``iota2`` ``theta = (alpha_s + 2 phi) / 2 pi`` is a straight-field-line angle (``alpha_s`` the surface angle of the
field-line labels, constant on a line), so lambda vanishes identically; for ``sheared`` ``theta = chi / 2 pi``
and lambda comes from the exact field (:func:`logical_field`).
"""
from __future__ import annotations

import numpy as np

TWO_PI = 2.0 * np.pi
NFP = 2

#: the two cases of the verification (the paper's examples; the sheared lambda is ours: the paper leaves it free)
CASES = {
    "iota2": dict(family="iota2", eps=0.5, delta=1.0 / 64.0, p_edge=0.0),
    "sheared": dict(family="sheared", eps=2.0, S=1.0, k_b=0.1, lam=1.0, kappa=1.0, p_edge=0.0),
}


# ---------------------------------------------------------------------------
# iota = 2 family
# ---------------------------------------------------------------------------

def _ab(eps):
    return np.sqrt(1.0 + eps), np.sqrt(1.0 - eps)


def iota2_field(xyz, eps):
    a, b = _ab(eps)
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    s = (x / a) ** 2 + (y / b) ** 2
    F = np.sqrt(1.0 - (1.0 - s) ** 2 - 4.0 * z ** 2)
    return np.stack(((2 * z * x - (a / b) * F * y) / s, (2 * z * y + (b / a) * F * x) / s, 1.0 - s), axis=-1)


def iota2_psi(xyz, eps):
    B = iota2_field(xyz, eps)
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    return (x ** 2 + y ** 2 + 4 * z ** 2 + np.sum(B * B, axis=-1) - 2.0 + eps ** 2) / 4.0


def iota2_embedding(u, v, t, eps):
    """``r(u, v, t)``: ``q^2 = u^2 + v^2``, ``L = sqrt((1 + sqrt(1 - 4 q^2)) / 2)``."""
    a, b = _ab(eps)
    L = np.sqrt((1.0 + np.sqrt(1.0 - 4.0 * (u * u + v * v))) / 2.0)
    ct, st = np.cos(t), np.sin(t)
    return np.stack((a * (L * ct + (u * ct + v * st) / L), b * (L * st + (v * ct - u * st) / L),
                     v * np.cos(2 * t) - u * np.sin(2 * t)), axis=-1)


def iota2_point(rho, theta, phi, eps, delta):
    """The position of the logical point ``(rho, theta, phi)``: the field-line label ``alpha_s = 2 pi theta - 2
    phi`` (``u = -eps/2 + sqrt(delta) rho cos alpha``, ``v = -sqrt(delta) rho sin alpha``: the minus sign makes
    theta clockwise), then the ``t`` on that line with ``atan2(y, x) = phi`` by Newton (``phi(t)`` is monotone)."""
    rho, theta, phi = np.broadcast_arrays(*(np.asarray(v, dtype=np.float64) for v in (rho, theta, phi)))
    al = TWO_PI * theta - 2.0 * phi
    r = np.sqrt(delta) * rho
    u, v = -eps / 2 + r * np.cos(al), -r * np.sin(al)
    t = phi.copy()
    for _ in range(60):
        X = iota2_embedding(u, v, t, eps)
        g = np.angle(np.exp(1j * (np.arctan2(X[..., 1], X[..., 0]) - phi)))
        h = 1e-6
        Xp = iota2_embedding(u, v, t + h, eps)
        dg = np.angle(np.exp(1j * (np.arctan2(Xp[..., 1], Xp[..., 0]) - np.arctan2(X[..., 1], X[..., 0])))) / h
        t = t - g / dg
        if np.max(np.abs(g)) < 1e-15:
            break
    return iota2_embedding(u, v, t, eps)


def iota2_numbers(eps, delta, p_edge=0.0):
    """Closed-form profile numbers: ``Phi_t = pi a b psi`` (toroidal flux), ``iota = 2``, ``beta_V = 2 <p> / <B^2>
    = 2 (p_b + delta) / (1 - eps^2/2 + delta)``, ``V = 2 pi^2 a b delta``."""
    a, b = _ab(eps)
    return dict(phi_edge=np.pi * a * b * delta, iota=2.0, beta_V=2 * (p_edge + delta) / (1 - eps ** 2 / 2 + delta),
                volume=2 * np.pi ** 2 * a * b * delta, p_axis=p_edge + 2 * delta)


# ---------------------------------------------------------------------------
# sheared-iota family
# ---------------------------------------------------------------------------

def sheared_field(xyz, eps, S, lam, kappa=1.0):
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    w, wb = x + 1j * y, x - 1j * y
    K = wb * np.sqrt(1.0 + eps / wb ** 2)        # sqrt(conj(w)^2 + eps), the principal branch of the paper
    Xi = w * K + np.pi / 2 - S
    bxy = 1j * kappa * np.exp(-1j * lam * z) * np.sin(Xi) / (2 * K)
    bz = kappa / lam * np.real(np.exp(-1j * lam * z) * np.cos(Xi))
    return np.stack((bxy.real, bxy.imag, bz), axis=-1)


def sheared_psi(xyz, eps, S, lam, kappa=1.0):
    bz = sheared_field(xyz, eps, S, lam, kappa)[..., 2]
    return (np.sin(lam * xyz[..., 2]) ** 2 + (lam * bz / kappa) ** 2) / 2.0


def _semiaxes(sigma, eps):
    h = np.hypot(2 * sigma, eps)
    b = np.sqrt((h + eps) / 2)
    return sigma / b, b


def sheared_point(rho, theta, phi, eps, S, k_b, lam, **_):
    """Position of ``(rho, theta, phi)``, ``k = k_b rho``, ``chi = 2 pi theta``, at the geometric azimuth ``phi``:
    ``sigma`` by bisection in its exact bracket ``S +- asin(k)`` (the paper's fixed-azimuth inversion)."""
    rho, theta, phi = np.broadcast_arrays(*(np.asarray(v, dtype=np.float64) for v in (rho, theta, phi)))
    k = k_b * rho
    chi = TWO_PI * theta
    P, Y = -k * np.cos(chi), k * np.sin(chi)
    root = np.sqrt(1 - Y ** 2)

    def residual(sigma):
        a, b = _semiaxes(sigma, eps)
        t = np.arctan2(a * np.sin(phi), b * np.cos(phi))
        nu = eps / 2 * np.sin(2 * t)
        return sigma - S - np.arctan(np.tanh(nu) * Y / root) + np.arcsin(P / np.sqrt(np.cosh(nu) ** 2 - Y ** 2)), t

    lo, hi = S - np.arcsin(k) - 1e-14, S + np.arcsin(k) + 1e-14
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        val, _ = residual(mid)
        lo, hi = np.where(val < 0, mid, lo), np.where(val >= 0, mid, hi)
    sigma = 0.5 * (lo + hi)
    _, t = residual(sigma)
    a, b = _semiaxes(sigma, eps)
    return np.stack((a * np.cos(t), b * np.sin(t), -np.arcsin(Y) / lam), axis=-1)


def sheared_numbers(eps, S, k_b, lam, kappa=1.0, p_edge=0.0, n=1024):
    """Landreman's 1-D quadratures (his DESC script): the toroidal flux derivative ``Q'(k)/k`` through ``t = 0``,
    the poloidal ``A'(k)/k``, ``iota = A'/Q'``; the axis value ``iota(0) = h(S) <sech nu>``."""
    h = lambda s: np.sqrt(4 * s ** 2 + eps ** 2)       # noqa: E731

    def derivs(k):
        k = np.atleast_1d(k)[:, None]
        u = np.arange(n) * (TWO_PI / n)
        c = np.sqrt(1 - k ** 2 * np.sin(u) ** 2)
        sig = S + np.arcsin(k * np.cos(u) / c)
        q = kappa * np.pi / lam * np.mean(1 / (h(sig) * c), axis=1)
        v = eps / 2 * np.sin(2 * u)
        sig = S + np.arcsin(k / np.cosh(v))
        g = (h(sig) + eps * np.cos(2 * u)) / 2
        a = TWO_PI * kappa / lam * np.mean(g / (h(sig) * np.sqrt(np.cosh(v) ** 2 - k ** 2)), axis=1)
        return q, a

    t = np.arange(n) * TWO_PI / n
    q0, a0 = derivs(0.0)
    qb, ab = derivs(k_b)
    return dict(iota_axis=float(a0[0] / q0[0]), iota_edge=float(ab[0] / qb[0]),
                iota_axis_formula=float(h(S) * np.mean(1 / np.cosh(eps / 2 * np.sin(2 * t)))),
                p_axis=p_edge + kappa ** 2 * k_b ** 2 / (2 * lam ** 2), derivs=derivs)


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

def field(case, xyz):
    if case["family"] == "iota2":
        return iota2_field(xyz, case["eps"])
    return sheared_field(xyz, case["eps"], case["S"], case["lam"], case["kappa"])


def psi(case, xyz):
    if case["family"] == "iota2":
        return iota2_psi(xyz, case["eps"])
    return sheared_psi(xyz, case["eps"], case["S"], case["lam"], case["kappa"])


def pressure(case, xyz):
    """``p`` with ``p = p_edge`` on the boundary ``psi = delta``."""
    if case["family"] == "iota2":
        return case["p_edge"] + 2.0 * (case["delta"] - iota2_psi(xyz, case["eps"]))
    k2 = case["k_b"] ** 2
    return case["p_edge"] + case["kappa"] ** 2 / case["lam"] ** 2 * (k2 / 2 - psi(case, xyz))


def psi_edge(case):
    return case["delta"] if case["family"] == "iota2" else case["k_b"] ** 2 / 2


def point(case, rho, theta, phi):
    if case["family"] == "iota2":
        return iota2_point(rho, theta, phi, case["eps"], case["delta"])
    return sheared_point(rho, theta, phi, **{k: v for k, v in case.items() if k != "family"})


def rho_of_psi(case, ps):
    return np.sqrt(ps / psi_edge(case))


# ---------------------------------------------------------------------------
# the field in logical coordinates and the wout export
# ---------------------------------------------------------------------------

def _fd(f, x, h):
    """Fourth-order central difference of ``f`` at ``x`` along one argument."""
    return (-f(x + 2 * h) + 8 * f(x + h) - 8 * f(x - h) + f(x - 2 * h)) / (12 * h)


def logical_field(case, rho, theta, zeta, h=(1e-4, 1e-4, 1e-4)):
    """At logical points (``zeta`` in field periods, ``phi = 2 pi zeta / nfp``): the position, ``DF``, ``J = det DF``
    and the reference 2-form components ``B_hat = J DF^-1 B`` of the exact field (MRX's ``sqrt(g) B^i``).
    ``DF`` by fourth-order differences of the closed-form point (the radial one one-sided-safe: ``rho`` must be in
    ``[2 h, 1 + ...]``, the analytic field extends past the wall)."""
    rho, theta, zeta = np.broadcast_arrays(*(np.asarray(v, dtype=np.float64) for v in (rho, theta, zeta)))
    P = lambda r, t, z: point(case, r, t, TWO_PI * z / NFP)    # noqa: E731
    X = P(rho, theta, zeta)
    DF = np.stack([_fd(lambda r: P(r, theta, zeta), rho, h[0]),
                   _fd(lambda t: P(rho, t, zeta), theta, h[1]),
                   _fd(lambda z: P(rho, theta, z), zeta, h[2])], axis=-1)       # (..., 3, 3), columns d/dx_i
    J = np.linalg.det(DF)
    B = field(case, X)
    Bhat = J[..., None] * np.linalg.solve(DF, B[..., None])[..., 0]
    return X, DF, J, B, Bhat


def fourier_rz(case, rho, n_theta=64, n_zeta=64):
    """R and Z on the tensor grid ``rho x theta_j x zeta_k`` (one field period, uniform), and their wout
    coefficients ``rmnc``, ``zmns`` for ``m < n_theta/2``, ``|n| < n_zeta/2`` (argument ``m theta - n nfp phi``,
    ``theta``, ``phi`` in radians): an exact DFT of the grid values, the stellarator symmetry imposed by taking the
    cosine part of R and the sine part of Z."""
    th = np.arange(n_theta) / n_theta
    ze = np.arange(n_zeta) / n_zeta
    Rg, Tg, Zg = np.meshgrid(rho, th, ze, indexing="ij")
    X = point(case, Rg, Tg, TWO_PI * Zg / NFP)
    R, Z = np.hypot(X[..., 0], X[..., 1]), X[..., 2]
    ms = np.arange(n_theta // 2)
    ns = np.arange(-(n_zeta // 2) + 1, n_zeta // 2)
    modes = [(m, n) for m in ms for n in ns if m > 0 or n >= 0]
    arg = TWO_PI * (np.array([m for m, _ in modes])[:, None, None] * th[None, :, None]
                    - np.array([n for _, n in modes])[:, None, None] * ze[None, None, :])
    norm = np.array([1.0 if (m == 0 and n == 0) else 2.0 for m, n in modes])
    rmnc = np.einsum("rtz,ktz->rk", R, np.cos(arg)) / (n_theta * n_zeta) * norm
    zmns = np.einsum("rtz,ktz->rk", Z, np.sin(arg)) / (n_theta * n_zeta) * norm
    xm = np.array([m for m, _ in modes], float)
    xn = NFP * np.array([n for _, n in modes], float)
    return xm, xn, rmnc, zmns, dict(R=R, Z=Z)


def lambda_fourier(case, rho, xm, xn, n_theta=64, n_zeta=64):
    """lambda (radians, sine series) from the exact field on each surface: ``B_hat^zeta = Phi_w' (1 + d lambda /
    d theta_rad)`` with ``Phi_w' = <B_hat^zeta>_theta`` (the toroidal flux density in Wb per unit rho, the
    derivative of the wout ``phi``), so ``lambda_mn`` (``m > 0``) is the ``m``-th sine coefficient of
    ``B_hat^zeta / Phi_w' - 1`` divided by ``m``; the ``m = 0`` modes, invisible to ``B_hat^zeta``, from
    ``<B_hat^theta>_theta = Phi_w' (iota - d_phi lambda_0) / nfp``. Returns ``lmns`` and per surface ``Phi_w'`` and the flux-ratio
    ``iota = nfp <B_hat^theta> / <B_hat^zeta>`` (per full turn), plus the check quantities."""
    th = np.arange(n_theta) / n_theta
    ze = np.arange(n_zeta) / n_zeta
    Rg, Tg, Zg = np.meshgrid(rho, th, ze, indexing="ij")
    _, _, J, _, Bh = logical_field(case, Rg, Tg, Zg)
    dphi = Bh[..., 2].mean(axis=1)                     # (n_rho, n_zeta): must not depend on zeta
    iota = NFP * Bh[..., 1].mean(axis=(1, 2)) / Bh[..., 2].mean(axis=(1, 2))
    g = Bh[..., 2] / dphi.mean(axis=1)[:, None, None] - 1.0
    arg = TWO_PI * (xm[:, None, None] * th[None, :, None] - (xn / NFP)[:, None, None] * ze[None, None, :])
    coef = 2.0 * np.einsum("rtz,ktz->rk", g, np.cos(arg)) / (n_theta * n_zeta)
    lmns = np.where(xm[None, :] > 0, coef / np.where(xm > 0, xm, 1.0)[None, :], 0.0)
    # the m = 0 modes (a zeta-only lambda leaves B_hat^zeta alone): from the theta mean of B_hat^theta,
    # d_phi lambda_0 = iota - nfp <B_hat^theta>_theta / Phi_w', lambda_0n sin(-n nfp phi)
    h0 = iota[:, None] - NFP * Bh[..., 1].mean(axis=1) / dphi.mean(axis=1)[:, None]       # (n_rho, n_zeta)
    m0 = (xm == 0) & (xn > 0)
    c0 = 2.0 * np.einsum("rz,kz->rk", h0, np.cos(TWO_PI * (xn[m0] / NFP)[:, None] * ze[None, :])) / n_zeta
    lmns[:, m0] = -c0 / xn[m0][None, :]
    # the zeta-only part of B_hat^zeta: zero for a divergence-free tangent field (a check)
    check = dict(Bhat_rho_rel=float(np.abs(Bh[..., 0]).max() / np.abs(Bh[..., 2]).max()),
                 dphi_zeta_spread=float((dphi.max(axis=1) - dphi.min(axis=1)).max() / np.abs(dphi).max()),
                 J_min=float(J.min()), J_max=float(J.max()))
    return lmns, dphi.mean(axis=1), iota, check


def write_wout(path, xm, xn, rmnc, zmns, lmns_half, phi, iotaf, presf):
    """A minimal VMEC-8 wout (NetCDF3 classic) with exactly the variables :func:`mrx.vmec.read_wout` reads:
    ``rmnc``, ``zmns`` on the full mesh ``s_j = j/(ns-1)``, ``lmns`` on the half mesh (row 0 junk, zeros here),
    ``phi`` (Wb), ``phipf = dphi/ds``, ``chipf = iotaf phipf``, ``iotaf``, ``presf`` (Pa, i.e. the paper's
    ``p / mu_0``)."""
    from scipy.io import netcdf_file
    ns, mn = rmnc.shape
    f = netcdf_file(path, "w")
    f.createDimension("radius", ns)
    f.createDimension("mn_mode", mn)
    f.createDimension("dim_00001", 1)

    def var(name, data, dims):
        v = f.createVariable(name, np.float64 if np.asarray(data).dtype.kind == "f" else np.int32, dims)
        v[...] = data

    var("ns", np.int32(ns), ())
    var("nfp", np.int32(NFP), ())
    var("mnmax", np.int32(mn), ())
    var("lasym__logical__", np.int32(0), ())
    var("version_", np.float64(9.0), ())
    var("xm", xm, ("mn_mode",))
    var("xn", xn, ("mn_mode",))
    var("rmnc", rmnc, ("radius", "mn_mode"))
    var("zmns", zmns, ("radius", "mn_mode"))
    var("lmns", lmns_half, ("radius", "mn_mode"))
    s = np.arange(ns) / (ns - 1)
    phipf = np.gradient(phi, s, edge_order=2)
    var("phi", phi, ("radius",))
    var("phipf", phipf, ("radius",))
    var("chipf", iotaf * phipf, ("radius",))
    var("iotaf", iotaf, ("radius",))
    var("presf", presf, ("radius",))
    f.close()


def build_wout(case, ns=201, n_theta=48, n_zeta=48):
    """The wout arrays of ``case`` on ``ns`` full-mesh surfaces ``s_j = j/(ns-1)``, ``rho = sqrt(s)``: R, Z by
    :func:`fourier_rz`, lambda on the half mesh by :func:`lambda_fourier`, ``phi`` in closed form (``iota2``:
    ``pi a b delta s``) or by Landreman's toroidal-flux quadrature (``sheared``: ``Q(k)``, ``k = k_b rho``),
    ``iotaf`` the flux ratio of the exact field on the full mesh (``iota2``: 2), ``presf = p`` (``mu_0 = 1``
    units, i.e. Pa for B in tesla up to the ``mu_0`` MRX never uses). Returns ``(arrays, checks)``."""
    s = np.arange(ns) / (ns - 1)
    rho = np.sqrt(s)
    xm, xn, rmnc, zmns, _ = fourier_rz(case, rho, n_theta, n_zeta)
    rho_half = np.sqrt((np.arange(1, ns) - 0.5) / (ns - 1))
    lm, dphi_half, iota_half, chk = lambda_fourier(case, rho_half, xm, xn, n_theta, n_zeta)
    lmns = np.vstack([np.zeros((1, len(xm))), lm])
    pe = psi_edge(case)
    if case["family"] == "iota2":
        num = iota2_numbers(case["eps"], case["delta"], case["p_edge"])
        phi = num["phi_edge"] * s
        iotaf = np.full(ns, 2.0)
        presf = case["p_edge"] + 2.0 * pe * (1.0 - s)
        # d phi / d rho = 2 rho phi_edge against the measured flux density
        chk["dphi_vs_closed_form"] = float(np.abs(dphi_half - 2 * rho_half * num["phi_edge"]).max()
                                           / num["phi_edge"])
        chk["iota_half_minus_2"] = float(np.abs(iota_half - 2.0).max())
    else:
        num = sheared_numbers(case["eps"], case["S"], case["k_b"], case["lam"], case["kappa"], case["p_edge"])
        derivs = num["derivs"]
        xg, wg = np.polynomial.legendre.leggauss(24)
        phi = np.array([0.5 * r * np.sum(wg * (lambda rr: derivs(case["k_b"] * rr)[0] * (case["k_b"] * rr)
                                               * case["k_b"])(0.5 * r * (xg + 1))) for r in rho])
        q, a = derivs(case["k_b"] * rho)
        iotaf = a / q
        iotaf[0] = num["iota_axis"]
        presf = case["p_edge"] + case["kappa"] ** 2 / case["lam"] ** 2 * pe * (1.0 - s)
        q_h, a_h = derivs(case["k_b"] * rho_half)
        chk["dphi_vs_landreman"] = float(np.abs(dphi_half - q_h * case["k_b"] ** 2 * rho_half).max()
                                         / np.abs(dphi_half).max())
        chk["iota_half_vs_landreman"] = float(np.abs(iota_half - a_h / q_h).max())
    chk["lambda_max_rad"] = float(np.abs(lmns).max())
    return dict(xm=xm, xn=xn, rmnc=rmnc, zmns=zmns, lmns_half=lmns, phi=phi, iotaf=iotaf, presf=presf), chk
