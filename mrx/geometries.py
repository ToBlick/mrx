"""Named solve geometries for the driver scripts.

:func:`build_sequence` turns a geometry name into a polar
:class:`~mrx.derham_sequence.DeRhamSequence` with the map installed and the
operators every solver needs assembled: incidence operators, the Jacobi mass
diagonals and the metric-lumping Laplacian atoms for all ``k`` and both
boundary conditions. Mass matrices, projections and nullspaces are left to the
caller, which knows whether it needs them.

Analytic names: ``toroid``, ``cylinder``, ``rot-ellipse``. ``w7x`` reads
``W7-X.h5``; every other name is a flat-schema GVEC export listed in
:data:`mrx.gvec.GVEC_GEOMETRIES`.
"""
from __future__ import annotations

import h5py
import numpy as np

import mrx.operators as op
from mrx.derham_sequence import DeRhamSequence
from mrx.gvec import (GVEC_GEOMETRIES, GVEC_NFP_OVERRIDE, NFP_W7X,
                      build_gvec_map, build_w7x_map, gvec_path)
from mrx.mappings import cylinder_map, rotating_ellipse_map, toroid_map

#: Field periods spanned by logical zeta in [0, 1] for the analytic maps.
ANALYTIC_NFP = {"toroid": 1, "cylinder": 1, "rot-ellipse": 3, "w7x": NFP_W7X}


def geometry_nfp(geometry: str) -> int:
    """Field periods of a named geometry, read from the file where there is one."""
    if geometry in GVEC_GEOMETRIES:
        if geometry in GVEC_NFP_OVERRIDE:
            return GVEC_NFP_OVERRIDE[geometry]
        with h5py.File(gvec_path(geometry), "r") as h:
            return int(h.attrs["nfp"])
    return ANALYTIC_NFP[geometry]


def build_sequence(geometry, ns, p, maxiter=10_000, tol=None):
    """Build the sequence for a named geometry and assemble its solver operators.

    Args:
        geometry: one of the names in the module docstring.
        ns: ``(n_r, n_theta, n_zeta)``; also the map resolution for the
            file-based geometries.
        p: spline degree, all directions; quadrature order ``2p``.
        maxiter: iteration budget of every solve through the sequence.
        tol: solve tolerance; ``None`` is ``sqrt(eps)`` of the working
            precision.

    Returns:
        ``(seq, ops)`` with ``seq.set_operators(ops)`` already called.

    Raises:
        RuntimeError: if the installed map has a non-positive Jacobian at a
            quadrature point.
    """
    seq = DeRhamSequence(ns, (p,) * 3, 2 * p, ("clamped", "periodic", "periodic"),
                         polar=True, tol=tol, maxiter=maxiter,
                         betti_numbers=(1, 1, 0, 0))
    seq.evaluate_1d()
    if geometry == "toroid":
        seq.set_map(toroid_map(epsilon=1 / 3, R0=1.0))
    elif geometry == "cylinder":
        # a = 0.33 keeps the minor radius comparable to the toroid's. Zero
        # angular metric variation: the least-coupled geometry there is.
        seq.set_map(cylinder_map(a=0.33, h=1.0))
    elif geometry == "rot-ellipse":
        seq.set_map(rotating_ellipse_map(eps=0.33, kappa=1.5, nfp=3))
    elif geometry == "w7x":
        map_func, _ = build_w7x_map(map_ns=ns, p=p)
        seq.set_map(map_func)
    else:
        map_func, info = build_gvec_map(
            gvec_path(geometry), map_ns=ns, p=p,
            nfp=GVEC_NFP_OVERRIDE.get(geometry))
        print(f"[geom] {geometry}: nfp={info['nfp']} sign={info['sign']:+.0f} "
              f"det DF in [{info['det_range'][0]:.3e}, "
              f"{info['det_range'][1]:.3e}]", flush=True)
        seq.set_map(map_func)
    if geometry not in ("toroid", "cylinder", "rot-ellipse"):
        jac = np.asarray(seq.geometry.jacobian_j)
        if not np.isfinite(jac).all() or jac.min() <= 0:
            raise RuntimeError(f"{geometry} geometry is degenerate")
    ops = op.assemble_incidence_operators(seq)
    ops = op.assemble_mass_jacobi_preconditioner(seq, ops, ks=(0, 1, 2, 3))
    # The k>=1 saddle default requires the metric-lumping atoms; nothing
    # builds them implicitly.
    ops = op.assemble_metric_lumping_laplacian_preconditioner(
        seq, ops, ks=(0, 1, 2, 3), dirichlets=(False, True))
    op.warm_mass_preconditioner_cache(seq, ops)
    seq.set_operators(ops)
    return seq, ops
