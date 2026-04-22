"""Sum-factorized geometry evaluation for tensor-product spline maps.

When the logical-to-physical map ``F`` is itself a tensor-product spline
``F_alpha(x) = sum_{a,b,c} C^raw_{alpha, a, b, c}
               lambda^r_a(x_1) lambda^chi_b(x_2) lambda^zeta_c(x_3)``
all derived quantities at the quadrature grid can be computed from the
precomputed 1D basis values stored on a :class:`DeRhamSequence`, without
any black-box ``jacfwd`` / ``lax.map`` passes.

This module provides those sum-factorized evaluators and a
``SequenceGeometry`` builder that short-circuits
``compute_geometry_terms`` in the common case of a
:class:`mrx.mappings.SplineMap`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from mrx.assembly import grad_1d
from mrx.utils import inv33


def _coeffs_to_raw_grid(coefficients, extraction_T, nr, nt, nz):
    """Undo the extraction operator and reshape to the raw TP grid.

    Parameters
    ----------
    coefficients : (3, n_dof) array
        Cartesian spline coefficients in the extracted basis.
    extraction_T : BCSR of shape (n_raw, n_dof)
        Precomputed transpose of the extraction operator ``E``; this
        is usually ``seq.e0_T``.
    nr, nt, nz : int
        Raw tensor-product shape, ``n_raw = nr * nt * nz``.

    Returns
    -------
    (3, nr, nt, nz) array
        Coefficients expressed in the raw TP spline basis.
    """
    # C_raw = coefficients @ E   (shape (3, n_raw)), written via E^T.
    C_raw_flat = (extraction_T @ coefficients.T).T
    return C_raw_flat.reshape(3, nr, nt, nz)


def _tp_evaluate(C_raw, M1, M2, M3):
    """Evaluate sum_{a,b,c} C_raw[i,a,b,c] * M1[a,I] * M2[b,J] * M3[c,K].

    Sum-factorized in three contractions so the cost is
    ``O(N_q (n_r + n_t + n_z))`` rather than ``O(N_q n_r n_t n_z)``.

    Parameters
    ----------
    C_raw : (3, nr, nt, nz) array
    M1 : (nr, nqr) array  (either basis or d_basis in r)
    M2 : (nt, nqt) array
    M3 : (nz, nqz) array

    Returns
    -------
    (3, nqr, nqt, nqz) array
    """
    T1 = jnp.einsum("iabc,aI->iIbc", C_raw, M1)
    T2 = jnp.einsum("iIbc,bJ->iIJc", T1, M2)
    return jnp.einsum("iIJc,cK->iIJK", T2, M3)


def spline_map_F_DF_at_quad(coefficients, extraction_T, seq):
    """Evaluate ``F`` and ``DF`` at the sequence's quadrature grid.

    Uses the precomputed 1D basis / derivative values
    ``seq.basis_{r,t,z}_jk`` and ``seq.d_basis_{r,t,z}_jk`` (populated
    by ``seq.evaluate_1d()``).

    Returns
    -------
    F_q : (N_q, 3) array
        Physical position at each quadrature point.
    DF_q : (N_q, 3, 3) array
        Jacobian of F; axis 1 = Cartesian component, axis 2 = logical
        direction.
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
    """Return only ``det(DF)`` at the quadrature grid (skips the metric)."""
    _, DF_q = spline_map_F_DF_at_quad(coefficients, extraction_T, seq)
    return jnp.linalg.det(DF_q)


def min_jacobian_from_coeffs(coefficients, extraction_T, seq):
    """Minimum of ``det(DF)`` over the quadrature grid.

    Cheap mesh-folding diagnostic for a tensor-product spline map with
    the given ``coefficients``; evaluates only ``det(DF)`` and no metric
    or operator data.
    """
    return jnp.min(spline_map_jacobian_j_at_quad(coefficients, extraction_T, seq))


def compute_geometry_terms_from_spline(coefficients, extraction_T, seq):
    """Drop-in replacement for ``compute_geometry_terms`` for spline maps.

    Returns ``(metric_jkl, metric_inv_jkl, jacobian_j)`` with the same
    semantics as :func:`mrx.derham_sequence.compute_geometry_terms`.
    """
    _, DF_q = spline_map_F_DF_at_quad(coefficients, extraction_T, seq)
    metric_jkl = jnp.einsum("qki,qkj->qij", DF_q, DF_q)        # DF^T DF
    metric_inv_jkl = jax.vmap(inv33)(metric_jkl)
    jacobian_j = jnp.linalg.det(DF_q)
    return metric_jkl, metric_inv_jkl, jacobian_j
