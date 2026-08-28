"""VMEC equilibria: ``wout_*.nc`` files read in closed form.

The wout NetCDF file is VMEC's equilibrium output: the Fourier mode table
``(xm, xn)`` (``xn`` already multiplied by ``nfp``) and, per radial surface,
the coefficients of ``R`` (``rmnc``, cosine series), ``Z`` (``zmns``, sine)
and the stream function lambda (``lmns``, sine, radians), plus the profiles
``phi`` (toroidal flux in Wb), ``iotaf`` and ``presf``. The series argument
``m u - n v`` with the full-turn toroidal angle ``v`` is exactly the GVEC
state convention ``2 pi (m theta - (n / nfp) zeta)`` of
:class:`mrx.gvec.StateField`, with the same trig assignment, so a wout file
becomes the *same* block dict :func:`mrx.gvec.read_state` produces and every
consumer downstream -- ``build_gvec_map``, ``load_clebsch``,
``clebsch_potential_form`` -- applies verbatim. Two conversions happen here
and nowhere else:

* **Radial parameterisation.** VMEC's radial label is
  ``s = Phi / Phi_edge``, our ``rho^2``. ``rmnc``/``zmns`` live on the full
  mesh ``s_j = j / (ns - 1)``; ``lmns`` lives on the half mesh
  ``s_{j-1/2} = (j - 1/2) / (ns - 1)`` with the first row junk by VMEC
  convention (always dropped). The half mesh stops short of both ends
  (``rho_1/2 = sqrt(0.5/(ns-1))``, 0.082 at ns = 75), and a spline fit
  through it alone EXTRAPOLATES its end pieces over the axis and the
  edge -- measured 2026-08-28 as the residual of the QA vacuum field
  against the harmonic form concentrating at the axis and GROWING with
  resolution (docs/research/qa_vacuum_convergence_2026-08-28.md). So the
  lambda nodes are augmented with the axis and the edge: ``lambda_mn(0) =
  0`` for ``m > 0`` (the ``rho^m`` behaviour), the ``m = 0`` modes at
  ``rho = 0`` and every mode at ``rho = 1`` extrapolated linearly in ``s``
  from the two nearest half-mesh rows (:func:`_lambda_nodes`). Each mode
  is refit as a clamped
  interpolatory B-spline **in rho = sqrt(s)** through the nodes
  ``rho_j = sqrt(s_j)``: the odd-m ``s^(m/2)`` axis behaviour becomes
  analytic ``rho^m``, so no divide-by-sqrt(s) trick is needed (the DESC
  choice rather than the simsopt one). The knots are data-placed
  (:func:`mrx.gvec.knots_at_data`; the full mesh is edge-refined in rho)
  and carried in the block as ``T``, which :class:`mrx.gvec.StateField`
  prefers over an element grid. The m > 0 axis rows of ``rmnc``/``zmns``
  (an extrapolation in the file, not data) are pinned to 0 so the fit is
  exact at the axis.
* **Flux units.** GVEC profiles store ``Phi / 2 pi`` (flux per radian of
  toroidal angle); VMEC ``phi`` is the flux in Wb. The profile is divided
  by ``2 pi`` here so ``dPhi``/``dchi`` reach the Clebsch consumers in the
  units they expect. Pressure is Pa in both codes; lambda is radians in
  both; ``iotaf`` is per full toroidal turn like GVEC's iota, so the
  downstream ``iota = dchi / (nfp dPhi)`` conversion applies unchanged.

Guards at read time: files older than VMEC 8 store lambda on the full mesh
and are refused; non-stellarator-symmetric files (``lasym = 1``, which add
the cosine partners ``rmns``/``zmnc``/``lmnc``) are not implemented;
``chipf = iotaf * phipf`` is checked against the file's own arrays, which
catches a half/full-mesh or row-0 mistake. The Nyquist-mesh variables
(``gmnc``, ``bmnc``, ``bsup*``, on the *separate* table ``xm_nyq/xn_nyq``)
are not read -- the Clebsch route rebuilds B from the fluxes and lambda.

Files are NetCDF3 classic, read through ``scipy.io.netcdf_file``; no
netCDF4 dependency. ``test/test_vmec.py`` exercises the fit on a synthetic
state and on the simsopt reference files in ``data/``.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline

from mrx.gvec import StateField, knots_at_data

TWO_PI = 2.0 * np.pi

_VARIABLES = ("ns", "nfp", "mnmax", "xm", "xn", "lasym__logical__", "signgs",
              "version_", "rmnc", "zmns", "lmns", "phi", "phipf", "chipf",
              "iotaf", "presf")


def _raw(path):
    """The needed wout variables as numpy arrays (NetCDF3 via scipy)."""
    with open(path, "rb") as fh:
        if fh.read(3) != b"CDF":
            raise ValueError(f"{path}: not a NetCDF3 classic wout file (an "
                             "HDF5-format wout is not supported)")
    from scipy.io import netcdf_file  # noqa: PLC0415  (scipy.io is heavy)
    f = netcdf_file(path, mmap=False)
    try:
        missing = [k for k in _VARIABLES if k not in f.variables]
        if missing:
            raise ValueError(f"{path}: missing wout variables {missing}")
        return {k: np.array(f.variables[k][()]) for k in _VARIABLES}
    finally:
        f.close()


def _fit_block(rho, samples, sin_cos, m, n, deg):
    """A :class:`mrx.gvec.StateField` block from per-surface Fourier rows:
    the clamped interpolatory B-spline through ``samples`` (``(len(rho),
    n_modes)``) at the nodes ``rho``, on data-placed knots carried as
    ``T``."""
    T = np.asarray(knots_at_data(rho, deg, "clamped"))
    A = BSpline.design_matrix(rho, T, deg).toarray()
    coef = np.linalg.solve(A, samples)                    # (n_base, n_modes)
    return dict(m=m, n=n, coef=coef.T, sin_cos=sin_cos, deg=deg, T=T)


def _lambda_nodes(rho_half, lmns, m):
    """``(rho, samples)`` for the lambda fit: the half mesh with the axis and
    the edge added -- ``lambda_mn(0) = 0`` for ``m > 0``, the ``m = 0`` rows at
    the axis and all rows at the edge extrapolated linearly in ``s = rho^2``
    from the two nearest half-mesh rows -- so the spline's domain is [0, 1]
    and nothing is evaluated on an extrapolated end piece."""
    s = rho_half ** 2
    axis = lmns[0] + (lmns[1] - lmns[0]) * (0.0 - s[0]) / (s[1] - s[0])
    axis = np.where(m > 0, 0.0, axis)
    edge = lmns[-1] + (lmns[-1] - lmns[-2]) * (1.0 - s[-1]) / (s[-1] - s[-2])
    rho = np.concatenate([[0.0], rho_half, [1.0]])
    samples = np.vstack([axis[None, :], lmns, edge[None, :]])
    return rho, samples


def _state_from_raw(raw, path="wout"):
    """The :func:`mrx.gvec.read_state`-shaped dict of the raw wout arrays."""
    version = float(raw["version_"])
    if version < 8.0:
        raise ValueError(f"{path}: VMEC version {version:g} < 8 stores lambda "
                         "on the full mesh; refused")
    if int(raw["lasym__logical__"]):
        raise NotImplementedError(f"{path}: lasym = 1 (non-stellarator-"
                                  "symmetric) needs the cosine partners")
    chi_want = raw["iotaf"] * raw["phipf"]
    err = float(np.abs(raw["chipf"] - chi_want).max())
    if err > 1e-8 * max(float(np.abs(chi_want).max()), 1e-300):
        raise ValueError(f"{path}: chipf != iotaf * phipf (max err {err:.3e});"
                         " half/full mesh or row-0 confusion")

    ns, nfp = int(raw["ns"]), int(raw["nfp"])
    m, n = raw["xm"].astype(int), raw["xn"].astype(int)
    rho_full = np.sqrt(np.arange(ns) / (ns - 1))
    rho_half = np.sqrt((np.arange(1, ns) - 0.5) / (ns - 1))
    deg = 3
    rmnc, zmns = raw["rmnc"].copy(), raw["zmns"].copy()
    rmnc[0, m > 0] = 0.0
    zmns[0, m > 0] = 0.0
    return dict(
        nfp=nfp, deg=deg, ns=ns, mnmax=int(raw["mnmax"]),
        signgs=int(raw["signgs"]), version=version,
        X1=_fit_block(rho_full, rmnc, 2, m, n, deg),
        X2=_fit_block(rho_full, zmns, 1, m, n, deg),
        LA=_fit_block(*_lambda_nodes(rho_half, raw["lmns"][1:], m), 1, m, n, deg),
        profiles=dict(rho=rho_full, phi=raw["phi"] / TWO_PI,
                      iota=raw["iotaf"], pressure=raw["presf"]))


def read_wout(path):
    """Parse a wout file into the dict shape of :func:`mrx.gvec.read_state`:
    ``nfp``, ``deg``, the blocks ``X1``, ``X2``, ``LA`` (each carrying its
    knot vector ``T``) and ``profiles`` (``rho``, ``phi``, ``iota``,
    ``pressure`` at the full-mesh nodes, flux in GVEC units)."""
    return _state_from_raw(_raw(path), path)


def read_nfp(path):
    """Just ``nfp``, without fitting anything."""
    from scipy.io import netcdf_file  # noqa: PLC0415
    f = netcdf_file(path, mmap=False)
    try:
        return int(f.variables["nfp"][()])
    finally:
        f.close()


def profile_spline(st, name):
    """The radial spline through a wout profile's full-mesh values, in
    ``rho = sqrt(s)`` (mirror of :func:`mrx.gvec.profile_spline`). ``phi``
    is linear in ``s`` by construction, so its spline is the exact
    quadratic ``Phi_edge rho^2 / 2 pi`` and its derivative vanishes at the
    axis."""
    prof, T, deg = st["profiles"], st["X1"]["T"], st["deg"]
    A = BSpline.design_matrix(prof["rho"], T, deg).toarray()
    return BSpline(T, np.linalg.solve(A, prof[name]), deg)


def load_wout_clebsch(path, n_rho=401):
    """The ``load_clebsch`` dict of a wout file (mirror of
    :func:`mrx.gvec.load_state_clebsch`): profiles on ``n_rho`` uniform
    radii from the rho-splines (``chi' = iota Phi'``) and ``lam_h`` the
    closed-form :class:`mrx.gvec.StateField` of lambda."""
    st = read_wout(path)
    rho = np.linspace(0.0, 1.0, n_rho)
    dPhi = profile_spline(st, "phi").derivative()(rho)
    return dict(nfp=st["nfp"], rho=rho, dPhi=dPhi,
                dchi=profile_spline(st, "iota")(rho) * dPhi,
                p=profile_spline(st, "pressure")(rho), iota_spread=0.0,
                lam_h=StateField(st["LA"], None, st["nfp"]), closed_axes=[])
