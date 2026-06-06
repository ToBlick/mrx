"""Geometry evaluation and map interpolation for mapped de Rham sequences.

Provides two paths for computing the metric tensor and Jacobian determinant
at quadrature points:

- Generic path (``compute_geometry_terms``): works for any differentiable map
  via ``jax.jacfwd``.
- Sum-factorized fast path (``compute_geometry_terms_from_spline``): requires
  the map to be a tensor-product spline.  Avoids the black-box ``jacfwd``
  pass by exploiting the Kronecker structure of the 1D basis evaluations
  stored on a :class:`~mrx.derham_sequence.DeRhamSequence`.

Both paths return the same three arrays and are consumed by
:class:`SequenceGeometry`.

Also provides utilities for interpolating analytic or sampled maps onto the
spline basis of a sequence: :func:`greville_interpolate_map`,
:func:`greville_interpolate_stellarator_map`, and the deprecated
:func:`interpolate_map`.
"""

from __future__ import annotations

from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp

import mrx
from mrx.differential_forms import inv33, jacobian_determinant


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
    def G(x):
        DF = jax.jacfwd(map)(x)
        return DF.T @ DF

    metric_jkl = jax.lax.map(G, quad_x, batch_size=mrx.MAP_BATCH_SIZE_INNER)
    metric_inv_jkl = jax.lax.map(
        inv33, metric_jkl, batch_size=mrx.MAP_BATCH_SIZE_INNER)
    jacobian_j = jax.lax.map(jacobian_determinant(
        map), quad_x, batch_size=mrx.MAP_BATCH_SIZE_INNER)
    return metric_jkl, metric_inv_jkl, jacobian_j


# ---------------------------------------------------------------------------
# SequenceGeometry
# ---------------------------------------------------------------------------

class SequenceGeometry(eqx.Module):
    """Geometry data attached to a de Rham sequence.

    An ``eqx.Module`` so that the three quadrature-grid arrays
    (``metric_jkl``, ``metric_inv_jkl``, ``jacobian_j``) are dynamic
    pytree leaves and can flow through ``jit`` / ``grad``. ``map`` is
    kept as a normal field so that if it is itself a pytree (e.g. a
    :class:`~mrx.mappings.SplineMap`), its coefficient leaves are tracked;
    plain ``Callable`` maps are treated as opaque leaves.
    """

    map: Any
    metric_jkl: jnp.ndarray = None
    metric_inv_jkl: jnp.ndarray = None
    jacobian_j: jnp.ndarray = None

    @classmethod
    def from_map(cls, map: Callable, quad_x: jnp.ndarray) -> "SequenceGeometry":
        """Build geometry by evaluating a map on the quadrature grid.

        Args:
            map: Differentiable logical-to-physical map ``F: R^3 -> R^3``.
            quad_x: Quadrature points, shape ``(N_q, 3)``.

        Returns:
            A fully populated :class:`SequenceGeometry`.
        """
        metric_jkl, metric_inv_jkl, jacobian_j = compute_geometry_terms(
            map, quad_x)
        return cls(map, metric_jkl, metric_inv_jkl, jacobian_j)

    @classmethod
    def from_spline_map(cls, spline_map, seq) -> "SequenceGeometry":
        """Sum-factorized geometry builder for tensor-product spline maps.

        Requires that ``seq.evaluate_1d()`` has already been called (so
        ``seq.basis_{r,t,z}_jk`` / ``seq.d_basis_{r,t,z}_jk`` are
        populated) and that ``spline_map.extraction_T`` is set.

        Args:
            spline_map: A :class:`~mrx.mappings.SplineMap` with
                ``coefficients`` and ``extraction_T`` populated.
            seq: A :class:`~mrx.derham_sequence.DeRhamSequence` with
                ``evaluate_1d()`` already called.

        Returns:
            A fully populated :class:`SequenceGeometry`.
        """
        if spline_map.extraction_T is None:
            raise ValueError(
                "SplineMap.extraction_T must be set for the sum-factorized "
                "geometry path; construct the map via "
                "seq.build_spline_map(coefficients) or pass seq.e0_T.")
        if not hasattr(seq, "basis_r_jk"):
            raise ValueError(
                "Call seq.evaluate_1d() before constructing a "
                "SequenceGeometry from a SplineMap.")
        metric_jkl, metric_inv_jkl, jacobian_j = \
            compute_geometry_terms_from_spline(
                spline_map.coefficients, spline_map.extraction_T, seq)
        return cls(spline_map, metric_jkl, metric_inv_jkl, jacobian_j)


# ---------------------------------------------------------------------------
# Sum-factorized spline fast path
# ---------------------------------------------------------------------------

def _coeffs_to_raw_grid(coefficients, extraction_T, nr, nt, nz):
    """Undo the extraction operator and reshape to the raw TP grid.

    Args:
        coefficients: ``(3, n_dof)`` Cartesian spline coefficients in the
            extracted basis.
        extraction_T: BCSR matrix of shape ``(n_raw, n_dof)`` — precomputed
            transpose of the extraction operator ``E`` (usually ``seq.e0_T``).
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
    ``seq.basis_{r,t,z}_jk`` and ``seq.d_basis_{r,t,z}_jk`` (populated
    by ``seq.evaluate_1d()``).

    Args:
        coefficients: ``(3, n_dof)`` spline coefficients of the map.
        extraction_T: Transpose of the extraction operator, shape
            ``(n_raw, n_dof)``.
        seq: :class:`~mrx.derham_sequence.DeRhamSequence` with
            ``evaluate_1d()`` already called.

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


def spline_map_jacobian_j_at_quad(coefficients, extraction_T, seq):
    """Return only ``det(DF)`` at the quadrature grid (skips the metric).

    Args:
        coefficients: ``(3, n_dof)`` spline coefficients of the map.
        extraction_T: Transpose of the extraction operator.
        seq: :class:`~mrx.derham_sequence.DeRhamSequence` with
            ``evaluate_1d()`` already called.

    Returns:
        ``(N_q,)`` array of Jacobian determinants.
    """
    _, DF_q = spline_map_F_DF_at_quad(coefficients, extraction_T, seq)
    return jnp.linalg.det(DF_q)


def min_jacobian_from_coeffs(coefficients, extraction_T, seq):
    """Minimum of ``det(DF)`` over the quadrature grid.

    Cheap mesh-folding diagnostic for a tensor-product spline map with
    the given ``coefficients``; evaluates only ``det(DF)`` and no metric
    or operator data.

    Args:
        coefficients: ``(3, n_dof)`` spline coefficients of the map.
        extraction_T: Transpose of the extraction operator.
        seq: :class:`~mrx.derham_sequence.DeRhamSequence` with
            ``evaluate_1d()`` already called.

    Returns:
        Scalar minimum Jacobian determinant over all quadrature points.
    """
    return jnp.min(spline_map_jacobian_j_at_quad(coefficients, extraction_T, seq))


def compute_geometry_terms_from_spline(coefficients, extraction_T, seq):
    """Drop-in replacement for ``compute_geometry_terms`` for spline maps.

    Returns ``(metric_jkl, metric_inv_jkl, jacobian_j)`` with the same
    semantics as :func:`compute_geometry_terms`.

    Args:
        coefficients: ``(3, n_dof)`` spline coefficients of the map.
        extraction_T: Transpose of the extraction operator.
        seq: :class:`~mrx.derham_sequence.DeRhamSequence` with
            ``evaluate_1d()`` already called.

    Returns:
        Tuple ``(metric_jkl, metric_inv_jkl, jacobian_j)``.
    """
    _, DF_q = spline_map_F_DF_at_quad(coefficients, extraction_T, seq)
    metric_jkl = jnp.einsum("qki,qkj->qij", DF_q, DF_q)        # DF^T DF
    metric_inv_jkl = jax.vmap(inv33)(metric_jkl)
    jacobian_j = jnp.linalg.det(DF_q)
    return metric_jkl, metric_inv_jkl, jacobian_j


# ---------------------------------------------------------------------------
# Map interpolation onto spline DOFs
# ---------------------------------------------------------------------------

def greville_interpolate_map(F_analytic: Callable, seq) -> jnp.ndarray:
    """Interpolate an analytic map to spline coefficients via Greville collocation.

    Evaluates each Cartesian component of ``F_analytic`` at the
    tensor-product Greville points and solves the resulting 1-D collocation
    systems, returning a coefficient array suitable for
    :meth:`~mrx.derham_sequence.DeRhamSequence.set_spline_map`.

    No mass matrix is required; the only prerequisite is
    :meth:`~mrx.derham_sequence.DeRhamSequence.evaluate_1d`.

    Args:
        F_analytic: Analytic map ``F: R^3 -> R^3`` mapping logical coordinates
            ``(r, θ, ζ) ∈ [0, 1]^3`` to physical Cartesian coordinates
            ``(X, Y, Z)``.
        seq: :class:`~mrx.derham_sequence.DeRhamSequence` to interpolate into.
            Must have ``evaluate_1d()`` called.  Currently requires an
            all-clamped (non-periodic, non-polar) sequence; periodic or polar
            sequences raise ``NotImplementedError`` via
            :meth:`zeroform_interpolation`.

    Returns:
        Coefficient array of shape ``(3, seq.n0)`` — spline DOF vectors for
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

    No mass matrix is required; the only prerequisite is
    :meth:`~mrx.derham_sequence.DeRhamSequence.evaluate_1d`.

    Args:
        F_analytic: Analytic map ``F: R^3 -> R^3`` returning Cartesian
            ``(X, Y, Z)``.
        seq: :class:`~mrx.derham_sequence.DeRhamSequence` to use for
            interpolation.  Must have ``evaluate_1d()`` called.  Typically
            built with ``('clamped', 'periodic', 'periodic')`` boundary
            conditions and ``polar=False``.
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

    R_h = DiscreteFunction(R_dof, seq.basis_0, seq.e0)
    Z_h = DiscreteFunction(Z_dof, seq.basis_0, seq.e0)

    return stellarator_map(R_h, Z_h, nfp=nfp, flip_zeta=flip_zeta)


def interpolate_map(axes, R_grid, Z_grid, nfp, seq, flip_zeta=False):
    """Interpolate a stellarator map from R and Z sampled on a regular grid.

    Uses :func:`~mrx.io.project_sampled_field` (L² projection via
    ``RegularGridInterpolator`` + tensor-product integration) to obtain
    FEM coefficients for *R* and *Z*, then wraps them in a
    :func:`~mrx.mappings.stellarator_map`.

    .. deprecated::
        Prefer :func:`greville_interpolate_stellarator_map` when an analytic
        map is available: it requires no reference-domain mass matrix and no
        sampled grid.

    Args:
        axes: Tuple of 1-D arrays ``(x1, x2, x3)`` spanning the logical domain.
        R_grid: R values on the grid, shape ``(n1, n2, n3)``.
        Z_grid: Z values on the grid, shape ``(n1, n2, n3)``.
        nfp: Number of field periods.
        seq: :class:`~mrx.derham_sequence.DeRhamSequence` to use.  Must have
            ``evaluate_1d()`` and ``assemble_reference_mass_matrix()`` called.
        flip_zeta: Whether to flip the toroidal angle in the stellarator map.

    Returns:
        Stellarator map built from the interpolated R and Z.
    """
    # Lazy imports to avoid circular dependency.
    from mrx.mappings import stellarator_map  # noqa: PLC0415
    from mrx.differential_forms import DiscreteFunction  # noqa: PLC0415
    from mrx.io import project_sampled_field  # noqa: PLC0415

    R_dof = project_sampled_field(
        axes, R_grid, seq, k=0, dirichlet=False, reference_domain=True)
    Z_dof = project_sampled_field(
        axes, Z_grid, seq, k=0, dirichlet=False, reference_domain=True)

    R_h = DiscreteFunction(R_dof, seq.basis_0, seq.e0)
    Z_h = DiscreteFunction(Z_dof, seq.basis_0, seq.e0)

    return stellarator_map(R_h, Z_h, nfp=nfp, flip_zeta=flip_zeta)
