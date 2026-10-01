"""Using DESC's ``*.h5`` output files as input.

The file holds an ``Equilibrium``, or an ``EquilibriaFamily`` whose last member is the solution, with:
- ``NFP`` and the toroidal flux through the boundary ``Psi`` (Wb, signed)
- the Fourier-Zernike coefficients of
    - ``R`` (``R_lmn``, cosine symmetry)
    - ``Z`` (``Z_lmn``, sine symmetry)
    - lambda (``L_lmn``, sine symmetry, in radians, with ``theta + lambda`` the straight-field-line angle)
  (without stellarator symmetry, ``sym = False``, of both symmetries) for the modes ``(l, m, n)`` in
  ``*_basis/_modes``. The basis function is ``Z_l^|m|(r) F_m(theta) F_n(NFP zeta)`` with
  ``F_m(x) = cos(|m| x)`` for ``m >= 0`` and ``sin(|m| x)`` for ``m < 0``. ``Z_l^|m|`` is the Zernike
  radial polynomial (``Z(1) = 1``) and ``zeta`` the full-turn toroidal angle.
- the profiles pressure (Pa) and either ``iota`` (per full turn) or ``current`` (the net toroidal current
  in A), each a ``PowerSeriesProfile`` or a ``SplineProfile`` (method ``cubic2``)
DESC's radial label ``rho`` is MRX' ``r``. DESC keeps the Jacobian positive, as MRX does, so a file converted
from a VMEC wout has ``theta = -u``, and lambda and iota change sign with it.

The conversion is exact:
- each product ``F_m(theta) F_n(NFP zeta)`` is the pair ``trig(|m| theta -+ |n| NFP zeta) / 2``, a cosine
  for ``sign(m) = sign(n)`` and a sine otherwise
- each mode's radial function, a polynomial of degree ``L``, is one Bezier segment of degree ``L``
- the profiles are the same functions as ``BSpline``\\ s
A current-constrained file gets ``iota`` from the current by DESC's flux-surface-average formula,
interpolated by a polynomial in ``r^2``.

Input validation: kinetic or anisotropic pressure and other profile classes are refused. Files are read with
``h5py``. ``test/test_readers.py`` tests a synthetic file (``test/synthetic_desc.py``) and the li383 file in
the repo against the wout it was converted from.
"""
from __future__ import annotations

import h5py
import numpy as np
from numpy.polynomial import Chebyshev
from scipy.constants import mu_0
from scipy.interpolate import BSpline, make_interp_spline
from scipy.special import eval_jacobi

TWO_PI = 2.0 * np.pi


def zernike_radial(r, ell, m):
    """DESC's Zernike radial polynomials ``Z_ell^|m|(r)`` as an array ``(len(r), len(ell))``. Columns where
    ``ell - |m|`` is odd are zero."""
    m = np.abs(m)
    k = (ell - m) // 2
    x = np.asarray(r, dtype=np.float64)[:, None]
    return np.where((ell - m) % 2 == 0, (-1.0) ** k * x ** m * eval_jacobi(k, m, 0.0, 1.0 - 2.0 * x ** 2), 0.0)


def _bezier(values, deg):
    """Write the polynomial(s) ``values(x)`` of degree ``<= deg`` on ``[0, 1]`` as one clamped B-spline
    segment (Bezier form). Returns the knots ``T`` and the coefficients."""
    x = 0.5 - 0.5 * np.cos(np.pi * np.arange(deg + 1) / deg)       # collocation is exact for degree deg
    T = np.concatenate([np.zeros(deg + 1), np.ones(deg + 1)])
    return T, np.linalg.solve(BSpline.design_matrix(x, T, deg).toarray(), values(x))


def _block(eq, name, nfp):
    """The Fourier-Zernike field ``name`` (``R``, ``Z`` or ``L``) as a field dict of the state."""
    ell, m, n = np.tile(eq[f"_{name}_basis/_modes"][()], (2, 1)).T     # every DESC mode twice: N = +-|n| nfp
    c = np.tile(eq[f"_{name}_lmn"][()], 2)
    half = len(c) // 2
    N = np.abs(n) * nfp
    N[half:] *= -1
    # F_m(t) F_n(z) = (trig(|m| t - N z) + trig(|m| t + N z)) / 2, a cosine for sign(m) = sign(n) (sign(0) = 1),
    # the first term negated for cos.sin (m >= 0 > n), the second for sin.sin (m, n < 0)
    cos = (m >= 0) == (n >= 0)
    w = np.where((n < 0) & np.concatenate([m[:half] >= 0, m[half:] < 0]), -0.5, 0.5)
    # m = 0: trig(-N z) is +-trig(N z), folded onto N >= 0
    fold = (m == 0) & (N < 0)
    w[fold & ~cos] *= -1.0
    N[fold] *= -1
    keys, col = np.unique(np.column_stack([N, np.abs(m)]), axis=0, return_inverse=True)   # sorted by (n, m)
    deg = max(int(ell.max()), 1)
    parts = []
    for parity in (cos, ~cos):
        A = np.zeros((len(keys), len(c)))
        A[col.ravel()[parity], np.arange(len(c))[parity]] = (w * c)[parity]
        T, coef = _bezier(lambda x, A=A: zernike_radial(x, ell, m) @ A.T, deg)
        parts.append(coef.T)
    return dict(m=keys[:, 1], n=keys[:, 0], cos=parts[0], sin=parts[1], deg=deg, T=T)


def _profile(node, path):
    """A DESC profile as the same function, a ``BSpline`` in ``r``."""
    cls = node["__class__"][()].decode().rsplit(".", 1)[-1]
    params = node["_params"][()]
    if cls == "PowerSeriesProfile":
        powers = node["_basis/_modes"][()][:, 0]
        deg = max(int(powers.max()), 1)
        T, coef = _bezier(lambda x: x[:, None] ** powers @ params, deg)
        return BSpline(T, coef, deg)
    if cls == "SplineProfile" and node["_method"][()] == b"cubic2":
        # interpax's cubic2 is the not-a-knot cubic interpolant, extrapolated by its end pieces
        return make_interp_spline(node["_knots"][()], params, k=3)
    raise NotImplementedError(f"{path}: {node.name} is a {cls}, MRX reads PowerSeriesProfile and "
                              "SplineProfile with method cubic2")


def _iota_from_current(st, current, psi_edge, nfp, deg=20):
    """The ``iota`` profile of a current-constrained state from its net toroidal current ``I(r)``.

    It uses DESC's formula ``iota = (mu0 I / (2 Psi r) + <(lambda_z g_tt - (1 + lambda_t) g_tz) / sqrt g>)
    / <g_tt / sqrt g>``, where ``<.>`` is the average over ``theta, zeta`` and ``Psi`` the boundary flux.
    The first term is ``2 pi mu0 I / psi'`` (``psi = Psi r^2 / 2 pi``) divided by the angular area
    ``4 pi^2``. The result is a polynomial of degree ``deg`` in ``r^2``, returned as a ``BSpline`` in ``r``."""
    m_max = max(int(st[k]["m"].max()) for k in ("X1", "X2", "LA"))
    n_max = max(int(np.abs(st[k]["n"]).max()) for k in ("X1", "X2", "LA")) // nfp
    # One field period suffices. Uniform points integrate the periodic integrands spectrally.
    t = TWO_PI * np.arange(8 * m_max + 16) / (8 * m_max + 16)
    z = TWO_PI / nfp * np.arange(8 * n_max + 16) / (8 * n_max + 16)

    def fields(blk, r):
        """The field ``blk`` and its derivatives in ``r``, ``theta``, ``zeta`` on the grid ``(r, t, z)``."""
        arg = blk["m"][:, None, None] * t[None, :, None] - blk["n"][:, None, None] * z[None, None, :]
        # the cosine and the sine series side by side: the functions and their derivatives in arg
        f, df = np.concatenate([np.cos(arg), np.sin(arg)]), np.concatenate([-np.sin(arg), np.cos(arg)])
        m, n = np.tile(blk["m"], 2)[:, None, None], np.tile(blk["n"], 2)[:, None, None]
        spline = BSpline(blk["T"], np.concatenate([blk["cos"], blk["sin"]]).T, blk["deg"])
        rad, rad_r = spline(r), spline.derivative()(r)
        return (np.einsum("ik,ktz->itz", rad, f), np.einsum("ik,ktz->itz", rad_r, f),
                np.einsum("ik,ktz->itz", rad, m * df), np.einsum("ik,ktz->itz", rad, -n * df))

    def iota(s):
        r = np.sqrt(s)
        (R, R_r, R_t, R_z), (_, Z_r, Z_t, Z_z), (_, _, L_t, L_z) = (fields(st[k], r) for k in ("X1", "X2", "LA"))
        g_tt, g_tz = R_t ** 2 + Z_t ** 2, R_t * R_z + Z_t * Z_z
        sqrt_g = R * (R_t * Z_r - R_r * Z_t)
        num = (L_z * g_tt - (1.0 + L_t) * g_tz) / sqrt_g
        return ((mu_0 * current(r) / (2.0 * psi_edge * r) + num.mean(axis=(1, 2)))
                / (g_tt / sqrt_g).mean(axis=(1, 2)))

    # iota is even in r: interpolated in s = r^2 at the Chebyshev points, all off the axis
    P = Chebyshev.interpolate(iota, deg, domain=[0.0, 1.0])
    T, coef = _bezier(lambda x: P(x ** 2), 2 * deg)
    return BSpline(T, coef, 2 * deg)


def read_desc(path):
    """Read a DESC output file and return its state (:mod:`mrx.equilibria`). The state also holds DESC's
    resolutions ``L``, ``M``, ``N``."""
    with h5py.File(path, "r") as fh:
        cls = fh["__class__"][()].decode() if "__class__" in fh else ""
        if cls.endswith("EquilibriaFamily"):
            family = fh["_equilibria"]
            eq = family[str(sum(k.isdigit() for k in family) - 1)]
        elif cls.endswith("Equilibrium"):
            eq = fh
        else:
            raise ValueError(f"{path}: not a DESC Equilibrium or EquilibriaFamily (class {cls!r})")
        if isinstance(eq["_pressure"], h5py.Dataset):
            raise NotImplementedError(f"{path}: kinetic profiles (densities and temperatures) instead of a pressure")
        if isinstance(eq.get("_anisotropy"), h5py.Group):
            raise NotImplementedError(f"{path}: anisotropic pressure")
        nfp, psi_edge = int(eq["_NFP"][()]), float(eq["_Psi"][()])
        st = dict(nfp=nfp, L=int(eq["_L"][()]), M=int(eq["_M"][()]), N=int(eq["_N"][()]),
                  X1=_block(eq, "R", nfp), X2=_block(eq, "Z", nfp), LA=_block(eq, "L", nfp))
        pressure = _profile(eq["_pressure"], path)
        if isinstance(eq["_iota"], h5py.Dataset):             # the string None: current-constrained
            iota = _iota_from_current(st, _profile(eq["_current"], path), psi_edge, nfp)
        else:
            iota = _profile(eq["_iota"], path)
    phi = BSpline(np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]), np.array([0.0, 0.0, psi_edge / TWO_PI]), 2)
    st["profiles"] = dict(phi=phi, iota=iota, pressure=pressure)
    return st
