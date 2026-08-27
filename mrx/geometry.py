"""Geometry evaluation and map interpolation for mapped de Rham sequences.

Provides two paths for evaluating the map Jacobian ``DF`` at the quadrature
points and reducing it to the stored geometry (metric, inverse metric,
determinant):

- Generic path (:meth:`SequenceGeometry.from_map`): works for any
  differentiable map via ``jax.jacfwd``.
- Sum-factorized fast path (:meth:`SequenceGeometry.from_spline_map`):
  requires the map to be a tensor-product spline.  Avoids the black-box
  ``jacfwd`` pass by exploiting the Kronecker structure of the 1D basis
  evaluations stored on a :class:`~mrx.derham_sequence.DeRhamSequence`.

Also provides utilities for interpolating analytic maps onto the spline basis
of a sequence: :func:`greville_interpolate_map` and
:func:`greville_interpolate_stellarator_map`.
"""

from __future__ import annotations

import os
from typing import Any, Callable

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp

import mrx
from mrx.differential_forms import inv33


def grad_1d(d_basis, boundary_type):
    """Lift a derivative spline basis back to the scalar-basis space.

    Args:
        d_basis: ``(n-1, nq)`` or ``(n, nq)`` derivative basis values.
        boundary_type: ``'clamped'`` or ``'periodic'``.

    Returns:
        ``(n, nq)`` array suitable for contraction with the raw TP coefficient
        grid (same leading dimension as the primal basis).
    """
    if boundary_type == 'clamped':
        padded = jnp.pad(d_basis, ((1, 1), (0, 0)))
        return padded[:-1] - padded[1:]
    else:  # periodic
        return jnp.roll(d_basis, 1, axis=0) - d_basis


# ---------------------------------------------------------------------------
# Generic (jacfwd) path
# ---------------------------------------------------------------------------

def map_jacobian_at(map: Callable, x: jnp.ndarray) -> jnp.ndarray:
    """``DF`` of ``map`` at every point of ``x`` via batched ``jacfwd``.

    This is the only place the raw Jacobian is materialised on a point set.
    :meth:`SequenceGeometry.from_map` reduces it to the stored geometry at
    once; the physical-frame pullbacks (:func:`mrx.projectors.load` with
    ``frame='phys'``, :func:`mrx.io.load_grid_field`) call it on demand at
    load time, since ``DF`` itself is not kept on the sequence.

    Args:
        map: Differentiable logical-to-physical map ``F: R^3 -> R^3``.
        x: Points, shape ``(N, 3)``.

    Returns:
        ``(N, 3, 3)`` with ``DF[q, i, j] = dF_i/dx_j``.
    """
    return jax.lax.map(jax.jacfwd(map), x, batch_size=mrx.MAP_BATCH_SIZE_INNER)


def compute_geometry_terms(map: Callable, quad_x: jnp.ndarray):
    """Compute metric and Jacobian terms for an arbitrary map.

    Args:
        map: Differentiable logical-to-physical map ``F: R^3 -> R^3``.
        quad_x: Quadrature points, shape ``(N_q, 3)``.

    Returns:
        Tuple ``(metric_jkl, metric_inv_jkl, jacobian_j)``:

        - ``metric_jkl``: ``(N_q, 3, 3)`` — metric tensor ``DF^T DF`` at each
          quadrature point.
        - ``metric_inv_jkl``: ``(N_q, 3, 3)`` — inverse metric.
        - ``jacobian_j``: ``(N_q,)`` — Jacobian determinant ``det(DF)``.
    """
    geometry = SequenceGeometry.from_map(map, quad_x)
    return geometry.metric_jkl, geometry.metric_inv_jkl, geometry.jacobian_j


# ---------------------------------------------------------------------------
# SequenceGeometry
# ---------------------------------------------------------------------------

class SequenceGeometry(eqx.Module):
    """Geometry data attached to a de Rham sequence.

    An ``eqx.Module`` so that the quadrature-grid arrays are dynamic pytree
    leaves and can flow through ``jit`` / ``grad``. ``map`` is kept as a
    normal field so that if it is itself a pytree (e.g. a
    :class:`~mrx.mappings.SplineMap`), its coefficient leaves are tracked;
    plain ``Callable`` maps are treated as opaque leaves.

    Stored per quadrature point, built ONCE from ``DF`` by the constructors
    and never recomputed: the metric ``metric_jkl = DF^T DF`` ``(N_q, 3, 3)``,
    its inverse ``metric_inv_jkl`` ``(N_q, 3, 3)`` and the determinant
    ``jacobian_j = det DF`` ``(N_q,)``.  These are what every hot path wants
    -- the mass weights ``J``, ``J G^-1``, ``G/J``, ``1/J`` are elementwise
    products of them, ``cross_product_load`` contracts against ``G`` and
    ``G^-1`` on every force step, and the lumped preconditioners read them in
    the quadrature tensor layout.  ``DF`` itself is discarded: its only
    consumers are the physical-frame pullbacks at load time, which recompute
    it with :func:`map_jacobian_at`.
    """

    map: Any
    metric_jkl: jnp.ndarray = None
    metric_inv_jkl: jnp.ndarray = None
    jacobian_j: jnp.ndarray = None

    @classmethod
    def from_DF(cls, map, DF_jkl: jnp.ndarray) -> "SequenceGeometry":
        """Reduce the Jacobian on the quadrature grid to the stored geometry.

        Args:
            map: The map ``DF_jkl`` was evaluated from.
            DF_jkl: ``(N_q, 3, 3)`` with ``DF[q, i, j] = dF_i/dx_j``.

        Returns:
            A fully populated :class:`SequenceGeometry`.
        """
        metric_jkl = jnp.einsum("qki,qkj->qij", DF_jkl, DF_jkl)
        return cls(map, metric_jkl, jax.vmap(inv33)(metric_jkl),
                   jnp.linalg.det(DF_jkl))

    @classmethod
    def from_map(cls, map: Callable, quad_x: jnp.ndarray) -> "SequenceGeometry":
        """Build geometry by evaluating a map on the quadrature grid.

        Args:
            map: Differentiable logical-to-physical map ``F: R^3 -> R^3``.
            quad_x: Quadrature points, shape ``(N_q, 3)``.

        Returns:
            A fully populated :class:`SequenceGeometry`.
        """
        return cls.from_DF(map, map_jacobian_at(map, quad_x))

    @classmethod
    def from_spline_map(cls, spline_map, seq) -> "SequenceGeometry":
        """Sum-factorized geometry builder for tensor-product spline maps.

        Uses ``seq.basis_{r,t,z}_jk`` / ``seq.d_basis_{r,t,z}_jk`` and needs
        ``spline_map.extraction_T`` set.

        Args:
            spline_map: A :class:`~mrx.mappings.SplineMap` with
                ``coefficients`` and ``extraction_T`` populated.
            seq: A :class:`~mrx.derham_sequence.DeRhamSequence` with.

        Returns:
            A fully populated :class:`SequenceGeometry`.
        """
        if spline_map.extraction_T is None:
            raise ValueError(
                "SplineMap.extraction_T must be set for the sum-factorized "
                "geometry path; construct the map via "
                "seq.build_spline_map(coefficients) or pass seq.E(0).T.")
        _, DF_jkl = spline_map_F_DF_at_quad(
            spline_map.coefficients, spline_map.extraction_T, seq)
        return cls.from_DF(spline_map, DF_jkl)


# ---------------------------------------------------------------------------
# Sum-factorized spline fast path
# ---------------------------------------------------------------------------

def _coeffs_to_raw_grid(coefficients, extraction_T, nr, nt, nz):
    """Undo the extraction operator and reshape to the raw TP grid.

    Args:
        coefficients: ``(3, n_dof)`` Cartesian spline coefficients in the
            extracted basis.
        extraction_T: :class:`~mrx.extraction_operators.MatrixFreeExtraction`
            of shape ``(n_raw, n_dof)`` — the transpose of the extraction
            operator ``E`` (usually ``seq.E(0).T``).
        nr: Raw tensor-product size in the r direction.
        nt: Raw tensor-product size in the t direction.
        nz: Raw tensor-product size in the z direction.

    Returns:
        ``(3, nr, nt, nz)`` array of coefficients in the raw TP spline basis.
    """
    # C_raw = coefficients @ E   (shape (3, n_raw)), written via E^T.
    C_raw_flat = (extraction_T @ coefficients.T).T
    return C_raw_flat.reshape(3, nr, nt, nz)


def _tp_evaluate(C_raw, M1, M2, M3):
    """Sum-factorized evaluation of a tensor-product spline.

    Computes ``sum_{a,b,c} C_raw[i,a,b,c] * M1[a,I] * M2[b,J] * M3[c,K]``
    in three sequential contractions, at cost
    ``O(N_q (n_r + n_t + n_z))`` rather than ``O(N_q n_r n_t n_z)``.

    Args:
        C_raw: ``(3, nr, nt, nz)`` coefficient array.
        M1: ``(nr, nqr)`` basis or derivative-basis matrix in r.
        M2: ``(nt, nqt)`` basis or derivative-basis matrix in t.
        M3: ``(nz, nqz)`` basis or derivative-basis matrix in z.

    Returns:
        ``(3, nqr, nqt, nqz)`` array of evaluated values.
    """
    T1 = jnp.einsum("iabc,aI->iIbc", C_raw, M1)
    T2 = jnp.einsum("iIbc,bJ->iIJc", T1, M2)
    return jnp.einsum("iIJc,cK->iIJK", T2, M3)


def spline_map_F_DF_at_quad(coefficients, extraction_T, seq):
    """Evaluate ``F`` and ``DF`` at the sequence's quadrature grid.

    Uses the precomputed 1D basis / derivative values
    ``seq.basis_{r,t,z}_jk`` and ``seq.d_basis_{r,t,z}_jk`` (built by the
    sequence).

    Args:
        coefficients: ``(3, n_dof)`` spline coefficients of the map.
        extraction_T: Transpose of the extraction operator, shape
            ``(n_raw, n_dof)``.
        seq: :class:`~mrx.derham_sequence.DeRhamSequence` with.

    Returns:
        Tuple ``(F_q, DF_q)``:

        - ``F_q``: ``(N_q, 3)`` physical position at each quadrature point.
        - ``DF_q``: ``(N_q, 3, 3)`` Jacobian of F; axis 1 = Cartesian
          component, axis 2 = logical direction.
    """
    nr, nt, nz = seq.basis_0.shape[0]
    Br, Bt, Bz = seq.basis_r_jk, seq.basis_t_jk, seq.basis_z_jk
    # The stored d_basis_*_jk live in the (p-1)-degree derived space and
    # have a different leading dimension than basis_*_jk; grad_1d lifts
    # them back to the full (nr, nt, nz) scalar-basis space so we can
    # contract against the same raw coefficient grid.
    types = seq.basis_0.types
    Dr = grad_1d(seq.d_basis_r_jk, types[0])
    Dt = grad_1d(seq.d_basis_t_jk, types[1])
    Dz = grad_1d(seq.d_basis_z_jk, types[2])

    C_raw = _coeffs_to_raw_grid(coefficients, extraction_T, nr, nt, nz)

    F = _tp_evaluate(C_raw, Br, Bt, Bz)         # (3, nqr, nqt, nqz)
    dF_dx1 = _tp_evaluate(C_raw, Dr, Bt, Bz)
    dF_dx2 = _tp_evaluate(C_raw, Br, Dt, Bz)
    dF_dx3 = _tp_evaluate(C_raw, Br, Bt, Dz)

    # seq.quad.x is built via jnp.meshgrid(x_x, x_y, x_z) with the default
    # 'xy' indexing, which yields a flat order (iy, ix, iz) -- axis 0 is
    # x_y.  We therefore transpose (nqr, nqt, nqz) -> (nqt, nqr, nqz)
    # before flattening so the per-quad-point arrays align with seq.quad.x.
    def _flatten(X):
        return X.transpose(0, 2, 1, 3).reshape(3, -1)

    F_q = _flatten(F).T                                    # (N_q, 3)
    DF_q = jnp.stack(
        [_flatten(dF_dx1), _flatten(dF_dx2), _flatten(dF_dx3)],
        axis=-1,
    ).transpose(1, 0, 2)                                   # (N_q, 3, 3)
    return F_q, DF_q


# ORPHAN (zero-reference sweep 2026-08-27): nothing in mrx/, scripts/ or test/ calls this.
# ---------------------------------------------------------------------------
# Map interpolation onto spline DOFs
# ---------------------------------------------------------------------------

def greville_interpolate_map(F_analytic: Callable, seq) -> jnp.ndarray:
    """Interpolate an analytic map to spline coefficients via Greville collocation.

    Evaluates each Cartesian component of ``F_analytic`` at the
    tensor-product Greville points and solves the resulting 1-D collocation
    systems, returning a coefficient array suitable for
    :meth:`~mrx.derham_sequence.DeRhamSequence.set_spline_map`.

    Args:
        F_analytic: Analytic map ``F: R^3 -> R^3`` mapping logical coordinates
            ``(r, θ, ζ) ∈ [0, 1]^3`` to physical Cartesian coordinates
            ``(X, Y, Z)``.
        seq: :class:`~mrx.derham_sequence.DeRhamSequence` to interpolate into.
            Polar and periodic sequences
            are supported (the polar rows are restricted conformingly, see
            :func:`mrx.projectors.interpolate`).

    Returns:
        Coefficient array of shape ``(3, seq.n(0))`` — spline DOF vectors for
        the three Cartesian components stacked along axis 0.  Pass directly
        to ``seq.set_spline_map(coefficients)``.
    """
    component_dofs = [
        seq.interpolate(lambda x, i=i: F_analytic(x)[i], 0)
        for i in range(3)
    ]
    return jnp.stack(component_dofs, axis=0)


def greville_interpolate_stellarator_map(
        F_analytic: Callable, seq, nfp: int,
        flip_zeta: bool = False) -> Callable:
    """Build a stellarator map by Greville-interpolating R and Z.

    Extracts the cylindrical radius ``R = sqrt(X² + Y²)`` and vertical
    coordinate ``Z`` from ``F_analytic``, interpolates each as a scalar
    0-form via Greville collocation, and wraps the result in
    :func:`~mrx.mappings.stellarator_map`.

    Args:
        F_analytic: Analytic map ``F: R^3 -> R^3`` returning Cartesian
            ``(X, Y, Z)``.
        seq: :class:`~mrx.derham_sequence.DeRhamSequence` to use for
            interpolation.  Typically
            built with ``('clamped', 'periodic', 'periodic')`` boundary
            conditions.
        nfp: Number of field periods.
        flip_zeta: Passed through to
            :func:`~mrx.mappings.stellarator_map`.

    Returns:
        Stellarator map ``Phi(r, θ, ζ) -> (X, Y, Z)`` built from the
        interpolated spline representations of R and Z.
    """
    # Lazy import to avoid circular dependency (mappings -> geometry -> mappings).
    from mrx.mappings import stellarator_map  # noqa: PLC0415
    from mrx.differential_forms import DiscreteFunction  # noqa: PLC0415

    def R_fn(x):
        Fxyz = F_analytic(x)
        return jnp.sqrt(Fxyz[0] ** 2 + Fxyz[1] ** 2)

    def Z_fn(x):
        return F_analytic(x)[2]

    R_dof = seq.interpolate(R_fn, 0)
    Z_dof = seq.interpolate(Z_fn, 0)

    R_h = DiscreteFunction(R_dof, seq.basis_0, seq.E(0))
    Z_h = DiscreteFunction(Z_dof, seq.basis_0, seq.E(0))

    return stellarator_map(R_h, Z_h, nfp=nfp, flip_zeta=flip_zeta)


# ---------------------------------------------------------------------------
# Solve geometries for the driver scripts, by analytic name or by file path
# ---------------------------------------------------------------------------
#
# A geometry is either an analytic name (``toroid``, ``cylinder``,
# ``rot-ellipse``) or the path of a GVEC export (``.h5``) or state file
# (``.dat``, read in closed form by :mod:`mrx.gvec`); ``os.path.isfile``
# decides. :func:`build_sequence` turns it into a polar sequence with the map
# installed and the preconditioners built; nullspaces are left to the caller.

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
            from mrx.gvec import read_state  # noqa: PLC0415  (imports this module)
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
            or a GVEC state file (read in closed form, ``mrx.gvec``).
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
        ValueError: if ``geometry`` is neither an analytic name nor a file,
            or (from ``set_geometry``) if the map folds.
    """
    from mrx.derham_sequence import DeRhamSequence  # noqa: PLC0415  (imports this module)
    from mrx.gvec import build_gvec_map  # noqa: PLC0415
    from mrx.mappings import cylinder_map, rotating_ellipse_map, toroid_map  # noqa: PLC0415

    seq = DeRhamSequence(ns, (p,) * 3, p + 1, ("clamped", "periodic", "periodic"),
                         polar=True, tol=tol, maxiter=maxiter,
                         betti_numbers=(1, 1, 0, 0))
    if os.path.isfile(geometry):
        map_func, info = build_gvec_map(geometry, map_ns=ns, p=p, nfp=nfp)
        print(f"[geom] {geometry}: nfp={info['nfp']} sign={info['sign']:+.0f} "
              f"det DF in [{info['det_range'][0]:.3e}, "
              f"{info['det_range'][1]:.3e}]", flush=True)
        seq.set_map(map_func)
    elif geometry == "toroid":
        seq.set_map(toroid_map(epsilon=1/3, R0=1.0))
    elif geometry == "cylinder":
        seq.set_map(cylinder_map(a=1/3, h=1.0))
    elif geometry == "rot-ellipse":
        seq.set_map(rotating_ellipse_map(eps=1/3, kappa=1.3, nfp=3))
    else:
        raise _unknown(geometry)
    return seq, seq.build_preconditioners()
