"""Solve geometries for the driver scripts, by analytic name or by file path.

:func:`build_sequence` turns a geometry into a polar
:class:`~mrx.derham_sequence.DeRhamSequence` with the map installed and the
operators every solver needs assembled: incidence operators, the Jacobi mass
diagonals and the metric-lumping Laplacian atoms for all ``k`` and both
boundary conditions. Mass matrices, projections and nullspaces are left to the
caller, which knows whether it needs them.

A geometry is either an analytic name (``toroid``, ``cylinder``,
``rot-ellipse``) or the path of a flat-schema GVEC export (``.h5``);
``os.path.isfile`` decides. A GVEC state file (``.dat``) is a file too and
is read in closed form (:mod:`mrx.gvec_state`). Any other string is an
error.
"""
from __future__ import annotations

import os

import h5py
import numpy as np

from mrx.derham_sequence import DeRhamSequence
from mrx.gvec import build_gvec_map
from mrx.mappings import cylinder_map, rotating_ellipse_map, toroid_map

#: Field periods spanned by logical zeta in [0, 1] for the analytic maps.
ANALYTIC_NFP = {"toroid": 1, "cylinder": 1, "rot-ellipse": 3}


def _unknown(geometry):
    return ValueError(
        f"geometry {geometry!r} is neither an analytic name "
        f"({', '.join(ANALYTIC_NFP)}) nor an existing file")


def geometry_nfp(geometry, nfp=None):
    """Field periods of a geometry.

    Args:
        geometry: an analytic name or the path of a GVEC export.
        nfp: overrides the file's ``nfp`` attribute; ignored for the analytic
            names.

    Returns:
        The number of field periods spanned by logical zeta in [0, 1].
    """
    if os.path.isfile(geometry):
        if nfp is not None:
            return int(nfp)
        if geometry.endswith(".dat"):
            from mrx.gvec_state import read_state
            return read_state(geometry)["nfp"]
        with h5py.File(geometry, "r") as h:
            return int(h.attrs["nfp"])
    if geometry in ANALYTIC_NFP:
        return ANALYTIC_NFP[geometry]
    raise _unknown(geometry)


def build_sequence(geometry, ns, p, maxiter=10_000, tol=None, nfp=None):
    """Build the sequence for a geometry and assemble its solver operators.

    Args:
        geometry: 
            an analytic name (``toroid``, ``cylinder``, ``rot-ellipse``),
            the path of a flat-schema GVEC export, 
            or a GVEC state file (read in closed form, ``mrx.gvec_state``).
        ns: ``(n_r, n_theta, n_zeta)``; also the map resolution for a file.
        p: spline degree, all directions; ``p + 1`` Gauss points per knot span.
        maxiter: iteration budget of every solve through the sequence.
        tol: solve tolerance; ``None`` is ``sqrt(eps)`` of working precision.
        nfp: overrides the file's ``nfp`` attribute (see ``mrx.gvec``);
            ignored for the analytic names.

    Returns:
        ``(seq, ops)``: the sequence with its geometry installed and every
        preconditioner built (``seq.operators is ops``).

    Raises:
        ValueError: if ``geometry`` is neither an analytic name nor a file.
        RuntimeError: if the installed map has a non-positive Jacobian at a
            quadrature point.
    """
    seq = DeRhamSequence(ns, (p,) * 3, p + 1, ("clamped", "periodic", "periodic"),
                         polar=True, tol=tol, maxiter=maxiter,
                         betti_numbers=(1, 1, 0, 0))
    if os.path.isfile(geometry):
        map_func, info = build_gvec_map(geometry, map_ns=ns, p=p, nfp=nfp)
        print(f"[geom] {geometry}: nfp={info['nfp']} sign={info['sign']:+.0f} "
              f"det DF in [{info['det_range'][0]:.3e}, "
              f"{info['det_range'][1]:.3e}]", flush=True)
        seq.set_map(map_func)
        jac = np.asarray(seq.geometry.jacobian_j)
        if not np.isfinite(jac).all() or jac.min() <= 0:
            raise RuntimeError(f"{geometry} geometry is degenerate")
    elif geometry == "toroid":
        seq.set_map(toroid_map(epsilon=1/3, R0=1.0))
    elif geometry == "cylinder":
        seq.set_map(cylinder_map(a=1/3, h=1.0))
    elif geometry == "rot-ellipse":
        seq.set_map(rotating_ellipse_map(eps=1/3, kappa=1.3, nfp=3))
    else:
        raise _unknown(geometry)
    return seq, seq.build_preconditioners()
