"""The Kronecker mass model and the generalised eigensolve behind the metric-lumping atoms.

:func:`_kron_mass_model_1d` is the separable Kronecker model
``Lam (x)_a A_a Lam`` of the mass matrices that
:class:`~mrx.metric_lumping_laplacian.MetricLumpingMass` inverts on the bulk
DoFs, and :func:`_simultaneous_diagonalize_pair` the generalised eigensolve
the Laplacian atoms' fast diagonalisation is built from. The atoms are the
only preconditioners (``docs/source/concepts/preconditioning.md``).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from mrx.precision import DTYPE


def _symmetrize(matrix: jnp.ndarray) -> jnp.ndarray:
    return 0.5 * (matrix + matrix.T)


def _simultaneous_diagonalize_pair(M: jnp.ndarray, A: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Simultaneously diagonalize an SPD ``M`` and a symmetric ``A``.

    Returns ``(V, lam)`` such that ``V.T @ M @ V = I`` and
    ``V.T @ A @ V = diag(lam)``, via Cholesky ``M = L L.T`` and a
    symmetric eigendecomposition of ``L^{-1} A L^{-T}``. This is the
    per-axis primitive of the Lynch fast-diagonalization (FD) method:
    given a 3D Kronecker sum ``M_r (x) M_t (x) M_z + A_r (x) A_t (x) A_z``
    with all ``M_axis`` SPD, applying the per-axis ``V`` reduces it to
    the diagonal ``1 + lam_r (x) lam_t (x) lam_z`` in the M-orthonormal
    basis. Reusable by the stiffness preconditioner.
    """
    M_sym = _symmetrize(jnp.asarray(M, dtype=DTYPE))
    A_sym = _symmetrize(jnp.asarray(A, dtype=DTYPE))
    L = jnp.linalg.cholesky(M_sym)
    Linv_A = jax.scipy.linalg.solve_triangular(L, A_sym, lower=True)
    B = jax.scipy.linalg.solve_triangular(L, Linv_A.T, lower=True).T
    B = _symmetrize(B)
    lam, U = jnp.linalg.eigh(B)
    V = jax.scipy.linalg.solve_triangular(L.T, U, lower=False)
    return V, lam


def _assemble_weighted_1d_mass(B: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    return (B * weights[None, :]) @ B.T


def _metric_lumping_diff_flags(k: int, c: int) -> tuple:
    """Differentiated-axis flags for component ``c`` of a k-form.

    Mirrors ``_component_axis_bases_k0/k1/k2/k3`` in :mod:`mrx.mass`:
    k=0 differentiates nothing, k=3 everything, k=1 only axis ``c``, and k=2
    every axis *except* ``c``.
    """
    match k:
        case 0:
            return (False, False, False)
        case 3:
            return (True, True, True)
        case 1:
            return tuple(a == c for a in range(3))
        case 2:
            return tuple(a != c for a in range(3))
    raise ValueError("k must be 0, 1, 2 or 3")


def _kron_mass_model_1d(seq, k: int, d_raw=None):
    """1-D factors of the Kronecker model of ``M_k``::

        M_k  ~  (+)_c  Lam_c (A^c_r x A^c_t x A^c_z) Lam_c

    with *unweighted* 1-D masses (degree ``p`` on primal axes, ``p-1`` on each
    differentiated axis) and the diagonal scaling ``Lam_c`` chosen so that the
    model reproduces ``diag(M_k)`` **exactly**: it is the support-averaged
    metric weight, ``sqrt(diag(M_k)_c / diag(A^c_r x A^c_t x A^c_z))``.

    This is the forward half of :func:`build_mass_metric_lumping_factors` -- that one
    inverts the 1-D masses and stores ``1/Lam`` -- and it is also the mass model
    the weak-term diagonal builds on. Measured ``||M~ - M||_F / ||M||_F`` on a
    spline toroid: ``3.1e-2`` at k=1, ``7e-3`` at k=2 and k=3.

    Returns ``(shapes, mass_1d, lam)``: the raw block shapes, the three 1-D
    masses per component, and the 3-D scaling per component.
    """
    from mrx.mass import build_mass_diagonal  # noqa: PLC0415

    form = getattr(seq, f"basis_{k}")
    shapes = [tuple(int(s) for s in sh) for sh in form.shape]

    if d_raw is None:
        d_raw = build_mass_diagonal(seq, k)
    d_raw = jnp.asarray(d_raw)

    primal = (seq.basis_r_jk, seq.basis_t_jk, seq.basis_z_jk)
    deriv = (seq.d_basis_r_jk, seq.d_basis_t_jk, seq.d_basis_z_jk)
    quad_w = (seq.quad.w_x, seq.quad.w_y, seq.quad.w_z)

    mass_1d, lam, start = [], [], 0
    for c, shape in enumerate(shapes):
        diff = _metric_lumping_diff_flags(k, c)
        bases = tuple(deriv[a] if diff[a] else primal[a] for a in range(3))
        m1 = tuple(_assemble_weighted_1d_mass(bases[a], quad_w[a]) for a in range(3))
        for a in range(3):
            if int(m1[a].shape[0]) != shape[a]:
                raise ValueError(
                    f"kron mass model k={k} component {c} axis {a}: 1D mass is "
                    f"{m1[a].shape[0]} but the raw block axis is {shape[a]}"
                )
        kron_diag = jnp.einsum('i,j,l->ijl', jnp.diag(m1[0]),
                               jnp.diag(m1[1]), jnp.diag(m1[2]))
        size = int(np.prod(shape))
        mass_1d.append(m1)
        lam.append(jnp.sqrt(d_raw[start:start + size].reshape(shape) / kron_diag))
        start += size
    return shapes, tuple(mass_1d), tuple(lam)


