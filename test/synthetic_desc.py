"""A DESC output file of an analytic circular torus, for tests.

:func:`write_synthetic_desc` writes the HDF5 layout that
:func:`mrx.desc.read_desc` parses -- the inverse of that parser, group for
group: the ``/_equilibria`` continuation family, each member's
``(l, m, n)`` mode tables and Fourier-Zernike coefficients for ``R``, ``Z``
and lambda, the toroidal flux ``_Psi``, and the ``PowerSeriesProfile``
objects for iota and pressure.

It describes the SAME torus as :mod:`test.synthetic_gvec`, reusing its
:class:`~test.synthetic_gvec.SyntheticTorus` for the closed formulas, so
the two readers can be held against each other as well as against the
formulas:

    R = R0 + a rho cos(theta),   Z = a rho sin(theta)
    LA = lam_amplitude rho sin(theta) (1 + LA_ZETA_MODULATION cos(nfp zeta))
    Phi(rho) = Phi_edge rho^2,  iota(rho) = iota0 + iota1 rho^2
    p(rho) = p0 (1 - rho^2)

Every radial function here is ``1`` or ``rho``, i.e. the Zernike
polynomials ``Z_0^0 = 1`` and ``Z_1^1 = rho``, which the clamped cubic
refit of :func:`mrx.vmec._fit_block` represents exactly; the profiles are
even quadratics, which it also represents exactly. So a reader that parses
the file correctly reproduces the formulas to round-off, and the angular
part exercises every branch of the product-to-sum conversion:

* ``R0`` is ``cos x cos`` with ``m = n = 0``, the doubly degenerate case
* ``a rho cos(theta)`` is ``cos x cos`` with ``n = 0``, where the two split
  modes collapse onto one at full weight
* ``a rho sin(theta)`` is ``sin x cos``, the sine block with ``n = 0``
* the lambda modulation is ``sin x cos`` with ``n = 1``, the genuinely
  split case: one DESC mode becoming the MRX pair ``(1, +-nfp)`` at half
  amplitude each

``LA_ZETA_MODULATION`` and the ``(1, +-nfp)`` half-amplitude split are
shared with the GVEC writer on purpose -- it is the same field.

Units differ from the GVEC file in exactly one place, and it is the easiest
thing to get wrong: DESC stores the toroidal flux in Webers as ``_Psi``
with ``Phi(rho) = Psi rho^2``, whereas GVEC profiles store flux per radian.
So ``_Psi = 2 pi Phi_edge`` here, and :func:`mrx.desc.read_desc` divides it
back out.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from test.synthetic_gvec import LA_ZETA_MODULATION, TWO_PI, SyntheticTorus

#: The ``__version__`` stamp written into every DESC object group. DESC
#: itself records the writing version here; nothing in MRX reads it.
VERSION = "mrx-synthetic"


def _power_series(group: Any, name: str, powers: list[int], params: list[float]) -> None:
    """Write a DESC ``PowerSeriesProfile`` group: ``sum_k params[k] rho^powers[k]``.

    Args:
        group: the parent HDF5 group (the equilibrium).
        name: the dataset name, ``"_pressure"`` or ``"_iota"``.
        powers: the exponents of rho, the first column of the basis modes.
        params: the coefficients, one per power.
    """
    node = group.create_group(name)
    node["__class__"] = np.bytes_(b"desc.profiles.PowerSeriesProfile")
    node["__version__"] = np.bytes_(VERSION.encode())
    node["_name"] = np.bytes_(name.lstrip("_").encode())
    node["_params"] = np.asarray(params, dtype=np.float64)
    basis = node.create_group("_basis")
    basis["__class__"] = np.bytes_(b"desc.basis.PowerSeries")
    basis["_modes"] = np.array([[ell, 0, 0] for ell in powers], dtype=np.int64)


def _field(group: Any, prefix: str, modes: list[tuple[int, int, int]],
           coef: list[float], nfp: int, sym: str) -> None:
    """Write one Fourier-Zernike field: its coefficients and its mode table.

    Args:
        group: the equilibrium HDF5 group.
        prefix: ``"R"``, ``"Z"`` or ``"L"``.
        modes: the ``(l, m, n)`` triples.
        coef: one coefficient per mode.
        nfp: field periods, stamped onto the basis as DESC does.
        sym: ``"cos"`` or ``"sin"``, the basis's symmetry tag.
    """
    group[f"_{prefix}_lmn"] = np.asarray(coef, dtype=np.float64)
    basis = group.create_group(f"_{prefix}_basis")
    basis["__class__"] = np.bytes_(b"desc.basis.FourierZernikeBasis")
    basis["_modes"] = np.asarray(modes, dtype=np.int64)
    basis["_NFP"] = np.int64(nfp)
    basis["_sym"] = np.bytes_(sym.encode())


def write_synthetic_desc(path: str, *, R0: float, a: float, nfp: int,
                         iota: tuple[float, float], Phi_edge: float,
                         lam_amplitude: float, beta: float,
                         sym: bool = True, store_iota: bool = True,
                         n_family: int = 1) -> SyntheticTorus:
    """Write the synthetic DESC output to ``path``; returns its torus.

    Args:
        path: output file (overwritten).
        R0, a: major and minor radius of the circular torus.
        nfp: field periods; DESC's zeta is the PHYSICAL toroidal angle, so
            the lambda modulation is stored as the ``n = 1`` mode, which
            DESC evaluates as ``cos(nfp zeta)``.
        iota: ``(iota0, iota1)`` of ``iota(rho) = iota0 + iota1 rho^2`` per
            full toroidal turn; negative on W7-X.
        Phi_edge: toroidal flux per radian at ``rho = 1``, the GVEC
            convention, stored as ``_Psi = 2 pi Phi_edge`` Webers. Negative
            values are legal and reverse the field (HSX, W7-X).
        lam_amplitude: amplitude of lambda (radians); zero switches it off.
        beta: on-axis ``2 mu0 p0 / B0^2`` of the stored pressure profile.
        sym: ``False`` writes a non-stellarator-symmetric file, which
            :func:`mrx.desc.read_desc` must refuse.
        store_iota: ``False`` writes ``_iota`` as the string ``None`` and an
            iota-shaped ``_current`` instead, the current-constrained layout
            that forces the DESC-backed fallback.
        n_family: members of the continuation family. Only the LAST is the
            converged solution; earlier ones are written with a displaced
            axis so a reader that takes the wrong member is caught.

    Returns:
        The :class:`~test.synthetic_gvec.SyntheticTorus` of the closed
        formulas the file encodes.
    """
    import h5py  # noqa: PLC0415

    iota0, iota1 = (float(v) for v in iota)
    torus = SyntheticTorus(float(R0), float(a), int(nfp), float(Phi_edge),
                           iota0, iota1, float(lam_amplitude), float(beta))
    p0 = float(torus.pressure(0.0))
    # DESC stores the modulation UNSPLIT: read_desc is what halves it onto
    # the (1, +-nfp) pair the GVEC writer spells out by hand.
    modulation = LA_ZETA_MODULATION * lam_amplitude

    with h5py.File(path, "w") as fh:
        fh["__class__"] = np.bytes_(b"desc.equilibrium.EquilibriaFamily")
        fh["__version__"] = np.bytes_(VERSION.encode())
        family = fh.create_group("_equilibria")
        for step in range(n_family):
            # Only the last member carries the true axis; the others are
            # displaced so picking the wrong one changes R on the axis.
            last = step == n_family - 1
            eq = family.create_group(str(step))
            eq["__class__"] = np.bytes_(b"desc.equilibrium.Equilibrium")
            eq["_sym"] = np.bool_(sym)
            eq["_NFP"] = np.int64(nfp)
            eq["_L"], eq["_M"], eq["_N"] = np.int64(2), np.int64(1), np.int64(1)
            eq["_Psi"] = np.float64(TWO_PI * Phi_edge)
            _field(eq, "R", [(0, 0, 0), (1, 1, 0)],
                   [R0 if last else R0 + a, a], nfp, "cos")
            _field(eq, "Z", [(1, -1, 0)], [a], nfp, "sin")
            _field(eq, "L", [(1, -1, 0), (1, -1, 1)],
                   [lam_amplitude, modulation], nfp, "sin")
            _power_series(eq, "_pressure", [0, 2], [p0, -p0])
            if store_iota:
                _power_series(eq, "_iota", [0, 2], [iota0, iota1])
                eq["_current"] = np.bytes_(b"None")
            else:
                eq["_iota"] = np.bytes_(b"None")
                _power_series(eq, "_current", [0, 2], [0.0, 1.0])
    return torus
