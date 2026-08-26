"""Preconditioner specifications, Jacobi and metric-lumping mass preconditioners, and diagonal extraction."""
from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Optional

import os

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import mrx
from mrx.precision import DTYPE, eps

#: Rank / structure cut-offs, in units of the working-dtype epsilon. The
#: float64 values are the ones each site was tuned at; the powers of two
#: reproduce them to within 1.5x (nothing lives near any of them).
PSEUDOINVERSE_TOL = eps(2.0 ** 25)    # 7.5e-9 in f64, was 1e-8
PROJECTOR_SVD_TOL = eps(2.0 ** 19)    # 1.2e-10 in f64, was 1e-10
PROJECTOR_PLANE_TOL = eps(2.0 ** 22)  # 9.3e-10 in f64, was rtol 1e-9 / atol 1e-11
BLOCK_DIAGONAL_TOL = eps(2.0 ** 12)   # 9.1e-13 in f64, was 1e-12

#: Rows per ``lax.map`` batch when probing an operator diagonal row by row.
#: See ``operators._diagonal_from_matvec`` for why the batch stays small.
PROBE_BATCH_SIZE = 8


class BoundaryConditionPair(eqx.Module):
    free: Optional[object] = None
    dbc: Optional[object] = None


class JacobiMassPreconditioner(eqx.Module):
    k0: BoundaryConditionPair = eqx.field(default_factory=BoundaryConditionPair)
    k1: BoundaryConditionPair = eqx.field(default_factory=BoundaryConditionPair)
    k2: BoundaryConditionPair = eqx.field(default_factory=BoundaryConditionPair)
    k3: BoundaryConditionPair = eqx.field(default_factory=BoundaryConditionPair)


class ExtractedMassApplyData(eqx.Module):
    # ``mass_apply`` is the raw-DOF-space matrix-free matvec ``v -> M_k v``
    # (see ``mrx.local_assembly.build_matrixfree_mass_apply``).
    mass_apply: object
    extraction: object
    extraction_t: object
    size: int = eqx.field(static=True)


class MassPreconditioners(eqx.Module):
    jacobi: Optional[JacobiMassPreconditioner] = None


@dataclass(frozen=True)
class MassPreconditionerSpec:
    """A mass preconditioner, named by ``kind``.

    ``'none'`` is the identity, ``'jacobi'`` the diagonal, and
    ``'metric_lumping'`` the production one: a separable Kronecker bulk with
    the polar core probed and inverted DENSELY, scaled by the metric-lumped
    diagonal. The field default is the production kind so that a bare spec
    and :func:`default_mass_preconditioner` agree.
    """
    kind: str = 'metric_lumping'


@dataclass(frozen=True)
class SchurPreconditionerSpec:
    inner: MassPreconditionerSpec = dataclass_field(
        default_factory=MassPreconditionerSpec)
    #: A bare spec must not silently mean SOME preconditioner. 'none' rather
    #: than 'metric_lumping' on purpose: 'none' fails VISIBLY at the first
    #: solve, where 'metric_lumping' would quietly work with something the
    #: caller never asked for. The authoritative answer is
    #: ``operators._materialize_default_saddle_preconditioner``, which needs a
    #: sequence and so cannot live in a field default at all. Every
    #: construction site in mrx/, test/ and scripts/ passes ``outer=``
    #: explicitly.
    outer: MassPreconditionerSpec = dataclass_field(
        default_factory=lambda: MassPreconditionerSpec(kind='none'))


@dataclass(frozen=True)
class SaddlePointPreconditionerSpec:
    mass: MassPreconditionerSpec = dataclass_field(
        default_factory=MassPreconditionerSpec)
    schur: SchurPreconditionerSpec = dataclass_field(
        default_factory=SchurPreconditionerSpec)
    coupled: bool = False


def default_mass_preconditioner() -> MassPreconditionerSpec:
    """The production mass preconditioner: metric_lumping.

    A separable Kronecker bulk with the polar core rows PROBED AND INVERTED
    DENSELY, scaled by the metric-lumped diagonal. The only mass preconditioner
    besides 'jacobi' and 'none'.

    MEASURED against its predecessor, the plain Kronecker model
    (docs/research/production_simplification_plan.md §10), 224 cells over
    four geometries, n = 8..20, p = 2..5:

    * the mass solve itself: median **0.83x** the iterations, and
      **0.70-0.77x** at k=1,2 where the cost is. The advantage HOLDS OR GROWS
      with h and is flat in p. Build time was equal (2.0 vs 2.2 s median).
    * the effect on ``L_k`` -- the mass preconditioner is the weak term's inner
      inverse, so this changes the OPERATOR at k >= 1, not just the solve:
      **median 0.91x, better in 12 of 16 cells**, up to 0.79x on the Dirichlet
      rows. Only regression was cylinder k=1 (1.07x).
    * the natural-BC scale SURVIVES: worst-case penalty against each cell's
      own optimum moves 1.14 -> 1.22, and only on the toroid, where the basin
      is flat. The shaped geometries are unchanged (1.01-1.04).

    Only regression anywhere was ~5% at k=0, on mass solves that take 7-17
    iterations either way.

    As the Schur-Jacobi probe backing it was measured separately -- six
    converged cells, five favouring it by 2.4-16.6%, one at +0.6% inside
    measured noise: docs/research/result_2026-08-25_schur_probe_ab.md.

    THE BUILD IS NOT JIT-SAFE, AND DOES NOT NEED TO BE. Its sparsity
    bookkeeping is host-side numpy and its core probe runs the matrix-free
    apply on concrete vectors, so a COLD cache inside a traced loop dies. The
    apply was made jit-safe in 3bd62aa; the build is warmed OUTSIDE the loop
    by ``operators.warm_mass_preconditioner_cache``. Any new traced entry
    point that solves must warm first.

    CAVEAT ON THE EVIDENCE: the mass A/B covers h = 8..20 and p = 2..5, but the
    effect on ``L_k`` was measured at n=12, p=3 only. The overnight sweep in
    ``outputs/diag_newstack/`` extends that to n = 8..32 and p = 2..5.
    """
    return MassPreconditionerSpec(kind='metric_lumping')


def default_saddle_preconditioner() -> SaddlePointPreconditionerSpec:
    """The k>=1 saddle default, as far as a no-argument function can state it.

    NOT AUTHORITATIVE, and it cannot be: the real outer block depends on whether
    the atom has been assembled for a given ``(k, BC)``, which needs a sequence.
    The authoritative resolver is
    ``operators._materialize_default_saddle_preconditioner``.

    ``outer`` is stated as ``'none'`` here because that is what the real
    resolver actually falls back to when the atom is absent -- it returns
    ``'metric_lumping'`` when assembled and ``'none'`` otherwise, and NEVER ``'jacobi'``.
    This docstring claimed jacobi was the fallback until 2026-08-25, which
    contradicted the very invariant the resolver exists to enforce: substituting
    a jacobi diagonal for a missing preconditioner is how the relaxation loop
    came to run its innermost solve on the diagonal unnoticed. A stale docstring
    naming jacobi as THE fallback is that same failure in prose.

    ``schur.inner`` is metric_lumping, matching the real default.
    """
    return SaddlePointPreconditionerSpec(
        mass=default_mass_preconditioner(),
        schur=SchurPreconditionerSpec(
            inner=MassPreconditionerSpec(kind='metric_lumping'),
            outer=MassPreconditionerSpec(kind='none'),
        ),
    )


def select_boundary_data(pair: BoundaryConditionPair, dirichlet: bool, label: str):
    data = pair.dbc if dirichlet else pair.free
    if data is None:
        side = "dbc" if dirichlet else "free"
        raise ValueError(f"{label} preconditioner is not assembled for {side} BCs")
    return data


def _mass_jacobi_pair(preconds: Optional[MassPreconditioners], k: int) -> Optional[BoundaryConditionPair]:
    if preconds is None or preconds.jacobi is None:
        return None
    match k:
        case 0:
            return preconds.jacobi.k0
        case 1:
            return preconds.jacobi.k1
        case 2:
            return preconds.jacobi.k2
        case 3:
            return preconds.jacobi.k3
    raise ValueError("k must be 0, 1, 2 or 3")


def get_mass_jacobi_diaginv(preconds: Optional[MassPreconditioners], k: int, dirichlet: bool):
    pair = _mass_jacobi_pair(preconds, k)
    if pair is None:
        raise ValueError(f"Jacobi mass preconditioner k={k} is not assembled")
    return select_boundary_data(pair, dirichlet, f"Jacobi mass k={k}")


def set_mass_jacobi_pair(preconds: Optional[MassPreconditioners], k: int, pair: BoundaryConditionPair):
    if preconds is None:
        preconds = MassPreconditioners()
    jacobi = preconds.jacobi if preconds.jacobi is not None else JacobiMassPreconditioner()
    match k:
        case 0:
            jacobi = eqx.tree_at(lambda data: data.k0, jacobi, pair)
        case 1:
            jacobi = eqx.tree_at(lambda data: data.k1, jacobi, pair)
        case 2:
            jacobi = eqx.tree_at(lambda data: data.k2, jacobi, pair)
        case 3:
            jacobi = eqx.tree_at(lambda data: data.k3, jacobi, pair)
        case _:
            raise ValueError("k must be 0, 1, 2 or 3")
    return eqx.tree_at(
        lambda data: data.jacobi,
        preconds,
        jacobi,
        is_leaf=lambda x: x is None,
    )








def _extracted_mass_diagonal(e, d_raw, mass_apply, *, batch_size: int = 16):
    """``diag(E M E^T)`` from the raw diagonal, probing only the coupled rows.

    ``E`` has exactly two kinds of row (verified for k=0,1,2 x both BCs at
    every resolution):

    * **bulk** -- a single nonzero, so ``(E M E^T)_ii = v^2 M_aa`` and the
      closed-form raw diagonal supplies it outright, with no operator apply;
    * **coupled** -- the polar rows, which mix several raw DOFs and therefore
      pick up *off-diagonal* entries of ``M`` that no diagonal can supply.

    Only the coupled rows are probed. There are ``3 n_z / 5 n_z / 2 n_z / 0``
    of them for k=0/1/2/3, so the apply count drops from ``O(N)`` to
    ``O(n_z)`` while the result stays exact -- this is not an approximation of
    the probed diagonal, it agrees with it to floating point.
    """
    rows = np.asarray(e.rows)
    cols = np.asarray(e.cols)
    vals = np.asarray(e.vals)
    n_rows, n_raw = (int(s) for s in e.forward_shape)
    d_raw_np = np.asarray(d_raw)

    counts = np.bincount(rows, minlength=n_rows)
    diag = np.zeros(n_rows)

    # Bulk rows: one nonzero each, so only the raw diagonal is involved.
    single = counts[rows] == 1
    diag[rows[single]] = (vals[single] ** 2) * d_raw_np[cols[single]]

    # Coupled rows: e_i^T M e_i with e_i the (short) raw row of E. The
    # nonzeros of the coupled rows are grouped ONCE, by position among the
    # coupled rows, rather than by scanning ``rows == r`` per row.
    coupled = np.flatnonzero(counts > 1)
    pos = np.full(n_rows, -1)
    pos[coupled] = np.arange(coupled.size)
    nz = np.flatnonzero(pos[rows] >= 0)
    t_all, c_all, v_all = pos[rows[nz]], cols[nz], vals[nz]
    for start in range(0, coupled.size, batch_size):
        blk = coupled[start:start + batch_size]
        sel = (t_all >= start) & (t_all < start + blk.size)
        probe = np.zeros((blk.size, n_raw))
        probe[t_all[sel] - start, c_all[sel]] = v_all[sel]
        probe_j = jnp.asarray(probe, dtype=DTYPE)
        images = jax.vmap(mass_apply)(probe_j)
        diag[blk] = np.asarray(jnp.sum(images * probe_j, axis=1))

    return jnp.asarray(diag, dtype=DTYPE)


def build_mass_jacobi_pair(seq, mass_apply, k: int) -> BoundaryConditionPair:
    """Build a Jacobi (diagonal-inverse) pair for the k-form mass matrix.

    ``mass_apply`` is the raw-DOF-space matvec ``v -> M_k v`` returned by
    :func:`mrx.operators.build_matrixfree_mass_apply`.

    ``diag(E M_k E^T)`` comes from the **closed-form** raw mass diagonal
    (:func:`mrx.local_assembly.build_mass_diagonal` -- one sum-factorized
    contraction against squared basis tables) rather than from ``O(N)``
    canonical-basis probes. Only the ``O(n_z)`` coupled polar rows still need
    an apply. The result is exact, not an estimate.
    """
    from mrx.local_assembly import build_mass_diagonal  # noqa: PLC0415

    d_raw = build_mass_diagonal(seq, k)
    e = getattr(seq, f"e{k}")
    e_dbc = getattr(seq, f"e{k}_dbc")
    return BoundaryConditionPair(
        free=1.0 / _extracted_mass_diagonal(e, d_raw, mass_apply),
        dbc=1.0 / _extracted_mass_diagonal(e_dbc, d_raw, mass_apply),
    )


def _quadrature_tensor_shape(seq) -> tuple[int, int, int]:
    return seq.quad.ny, seq.quad.nx, seq.quad.nz


def _reshape_quadrature_scalar_field(seq, values: jnp.ndarray) -> jnp.ndarray:
    return jnp.asarray(values).reshape(_quadrature_tensor_shape(seq))


def _reshape_quadrature_matrix_field(seq, values: jnp.ndarray) -> jnp.ndarray:
    field = jnp.asarray(values)
    return field.reshape(*_quadrature_tensor_shape(seq), *field.shape[1:])


def _k1_diagonal_metric_tensors(seq) -> dict[str, jnp.ndarray]:
    jacobian = _reshape_quadrature_scalar_field(seq, seq.geometry.jacobian_j)
    metric_inv = _reshape_quadrature_matrix_field(seq, seq.geometry.metric_inv_jkl)
    return {
        "alpha_rr": jacobian * metric_inv[..., 0, 0],
        "alpha_thetatheta": jacobian * metric_inv[..., 1, 1],
        "alpha_zetazeta": jacobian * metric_inv[..., 2, 2],
    }


def _k2_diagonal_metric_tensors(seq) -> dict[str, jnp.ndarray]:
    jacobian = _reshape_quadrature_scalar_field(seq, seq.geometry.jacobian_j)
    metric = _reshape_quadrature_matrix_field(seq, seq.geometry.metric_jkl)
    return {
        "beta_rr": metric[..., 0, 0] / jacobian,
        "beta_thetatheta": metric[..., 1, 1] / jacobian,
        "beta_zetazeta": metric[..., 2, 2] / jacobian,
    }


def _apply_extracted_mass_operator(extraction, extraction_t, mass_apply, x: jnp.ndarray) -> jnp.ndarray:
    raw = extraction_t @ x
    return jnp.asarray(extraction @ mass_apply(raw))


def _apply_extracted_mass_operator_data(data: ExtractedMassApplyData, x: jnp.ndarray) -> jnp.ndarray:
    return _apply_extracted_mass_operator(data.extraction, data.extraction_t, data.mass_apply, x)


def _apply_extracted_submatrix(data: ExtractedMassApplyData, row_indices: jnp.ndarray, col_indices: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    full = jnp.zeros((data.size,), dtype=x.dtype)
    full = full.at[col_indices].set(x)
    return _apply_extracted_mass_operator_data(data, full)[row_indices]


def _symmetric_pseudoinverse(matrix: jnp.ndarray, *,
                             relative_tol: float = PSEUDOINVERSE_TOL) -> jnp.ndarray:
    """PSD (positive-part) pseudoinverse. Both call sites invert Schur
    complements of SPD operators, which are PSD analytically but can dip
    slightly negative when rebuilt through an approximate bulk inverse;
    inverting a near-null eigenvalue by magnitude WITH its roundoff sign
    injects a huge negative Rayleigh direction and stalls CG (observed as
    ~1e-2 residual floors in the 2026-08-13 single-level campaign). Negative
    and sub-cutoff eigenvalues are dropped instead."""
    matrix = _symmetrize(matrix)
    eigvals, eigvecs = jnp.linalg.eigh(matrix)
    scale = jnp.max(jnp.abs(eigvals))
    safe_scale = jnp.where(scale > 0, scale, 1.0)
    cutoff = relative_tol * safe_scale
    inv_eigvals = jnp.where(eigvals > cutoff, 1.0 / eigvals, 0.0)
    return _symmetrize((eigvecs * inv_eigvals[jnp.newaxis, :]) @ eigvecs.T)


















































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

    Mirrors ``_component_axis_bases_k0/k1/k2/k3`` in :mod:`mrx.local_assembly`:
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


def _extraction_gram_inverse(e):
    """``(E E^T)^{-1}`` for the Kronecker mass model, as ``(coupled_rows, inverse)``.

    ``E E^T = diag(C C^T, I)``: bulk rows of ``E`` are orthonormal selectors, and
    the coupled/bulk cross block is exactly zero, so the Gram restricted to the
    coupled rows *is* ``C C^T``. It is block diagonal with blocks of size <= 3,
    so a dense inverse over the ``O(n_z)`` coupled rows reproduces the blocked
    inverse exactly while keeping the construction trivial.

    Returns ``(None, None, 0.0)`` when there are no coupled rows (k=3), where
    the pseudoinverse degenerates to ``E^T`` and the model is a plain tensor
    block.

    The returned ``cross`` is the largest coupled-bulk overlap found, RELATIVE
    to the largest Gram entry; it must be zero for the block structure to
    hold, and the caller asserts that rather than trusting the documented
    invariant.

    Host-side sparsity bookkeeping; the Gram is inverted on device in the
    working dtype.
    """
    rows = np.asarray(e.rows)
    cols = np.asarray(e.cols)
    vals = np.asarray(e.vals)
    n_rows, n_raw = (int(s) for s in e.forward_shape)

    counts = np.bincount(rows, minlength=n_rows)
    coupled = np.flatnonzero(counts > 1)
    if coupled.size == 0:
        return None, None, 0.0

    pos = np.full(n_rows, -1)
    pos[coupled] = np.arange(coupled.size)
    is_cp = pos[rows] >= 0

    # C = the coupled rows of E over the raw columns they touch; gram = C C^T.
    ucol, col_id = np.unique(cols[is_cp], return_inverse=True)
    c_mat = np.zeros((coupled.size, ucol.size))
    np.add.at(c_mat, (pos[rows[is_cp]], col_id), vals[is_cp])
    gram = c_mat @ c_mat.T

    # Largest |v_coupled v_bulk| over raw columns shared by both kinds of row.
    mx_cp, mx_bulk = np.zeros(n_raw), np.zeros(n_raw)
    np.maximum.at(mx_cp, cols[is_cp], np.abs(vals[is_cp]))
    np.maximum.at(mx_bulk, cols[~is_cp], np.abs(vals[~is_cp]))
    cross = float((mx_cp * mx_bulk).max() / np.abs(gram).max())

    gram_inv = jnp.linalg.inv(jnp.asarray(gram, dtype=DTYPE))
    return jnp.asarray(coupled), gram_inv, cross


def build_mass_metric_lumping_factors(seq, k: int, *, dirichlet: bool, d_raw=None):
    """Build the separable Kronecker mass factors for ``M_k``.

    The space is never split. A per-component diagonally-scaled Kronecker
    inverse acts on the full raw grid, and the pseudoinverse
    ``E+ = E^T (E E^T)^{-1}`` moves between raw and extracted coordinates::

        M^-1 ~ (E+)^T [ (+)_c D_c^-1/2 (M_r^-1 x M_t^-1 x M_z^-1)_c D_c^-1/2 ] E+

    with unweighted 1D masses (degree ``p`` on primal axes, ``p-1`` on each
    differentiated axis) and ``D`` the phi^2-weighted support average of the
    metric weight, taken straight from the exact mass diagonal.

    This is the same model class as the Greville-collocation sandwich --
    both are ``M_ab ~ sqrt(v_a v_b) (M_unw)_ab`` -- differing only in whether
    the weight is sampled at a point or averaged over the support. The averaged
    form is what makes it well defined on the innermost rings, where a Greville
    point sits at ``r ~ 0`` and ``J -> 0``.

    **Both sides must carry the full** ``(E E^T)^{-1}``, and this is the single
    easiest thing to get wrong: substituting ``E^T`` for ``E+`` still runs, and
    still looks acceptable on the easiest test case. In the pivot doc's
    pow0/pow1/pow2 ablation sweep -- pow-n = the correction applied n times --
    dropping it entirely (pow0) costs 2.3x the iterations at k=1 and drifts
    upward under refinement, because the mis-scaled subspace has dimension
    ``O(n_z)`` and grows; applying it once (pow1) recovers about half of that.
    These factors are the pow2 arm, i.e. the correction on both sides.

    Returns :class:`MetricLumpingMassFactors`: the separable Kronecker factors
    the metric_lumping atom's weak term is built on.
    """
    e = getattr(seq, f"e{k}_dbc" if dirichlet else f"e{k}")
    shapes, mass_1d, lam = _kron_mass_model_1d(seq, k, d_raw=d_raw)

    inv_1d = [tuple(jnp.linalg.inv(m) for m in mass_1d[c])
              for c in range(len(shapes))]
    inv_sqrt_D = [1.0 / lam_c for lam_c in lam]
    starts = [0]
    for sh in shapes:
        starts.append(starts[-1] + int(np.prod(sh)))

    coupled, gram_inv, cross = _extraction_gram_inverse(e)
    if cross > BLOCK_DIAGONAL_TOL:
        raise ValueError(
            f"metric_lumping k={k} dirichlet={dirichlet}: E E^T is not block diagonal "
            f"(max coupled-bulk overlap {cross:.3e}); the (CC^T, I) split that "
            "the pseudoinverse relies on does not hold here"
        )

    return MetricLumpingMassFactors(
        inv_1d=tuple(inv_1d),
        inv_sqrt_D=tuple(inv_sqrt_D),
        coupled=coupled,
        gram_inv=gram_inv,
        shapes=tuple(shapes),
        starts=tuple(starts),
    )


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
    from mrx.local_assembly import build_mass_diagonal  # noqa: PLC0415

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


class MetricLumpingMassFactors(eqx.Module):
    """Precomputed Kronecker mass-model factors for one ``(k, dirichlet)`` pair.

    Storage is ``O(n_z)``: the 1D inverses are tiny dense blocks and
    ``gram_inv`` covers only the coupled rows. Nothing here depends on the
    element count, and ``gram_inv`` depends only on the *sparsity* of ``E``, so
    it never rebuilds when the geometry changes -- unlike ``coupling_sb``,
    which is ``O(N n_z)`` and metric dependent.
    """
    inv_1d: tuple
    inv_sqrt_D: tuple
    coupled: Optional[jnp.ndarray]
    gram_inv: Optional[jnp.ndarray]
    shapes: tuple = eqx.field(static=True)
    starts: tuple = eqx.field(static=True)


def _restrict_radial_mass(matrix: jnp.ndarray, radial_start: int, nr: int) -> jnp.ndarray:
    radial_stop = radial_start + nr
    if radial_start < 0 or nr < 0 or radial_stop > matrix.shape[0] or radial_stop > matrix.shape[1]:
        raise ValueError(
            f"Invalid radial restriction start={radial_start}, nr={nr} for matrix shape {matrix.shape}"
        )
    return matrix[radial_start:radial_stop, radial_start:radial_stop]


def _core_size(seq) -> int:
    return 3 * seq.basis_0.nz


def _bulk_tensor_shape(seq, dirichlet: bool) -> tuple[int, int, int]:
    nr_bulk = seq.basis_0.nr - 2 - int(dirichlet)
    nt = seq.basis_0.nt
    nz = seq.basis_0.nz
    return nr_bulk, nt, nz


def _k3_extracted_shape(seq) -> tuple[int, int, int]:
    return seq.basis_3.dr - 1, seq.basis_3.dt, seq.basis_3.dz


# ---------------------------------------------------------------------------
# Diagonal probing utilities (matrix-free, probing-based)
# ---------------------------------------------------------------------------

def diag_matvec(A_matvec, n, *, dtype=DTYPE, batch_size=None):
    """Probe ``diag(A)`` from a forward operator on the extracted space.

    The operator is queried on small batches of canonical basis vectors.
    This is the matrix-free-compatible way to extract a diagonal.
    """
    if batch_size is None:
        configured_batch_size = mrx.MAP_BATCH_SIZE_OUTER
        if configured_batch_size is None:
            batch_size = 16
        else:
            batch_size = max(1, min(int(configured_batch_size), 16))
    if n == 0:
        return jnp.zeros((0,), dtype=dtype)
    diag_chunks = []
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        idx = jnp.arange(start, stop)
        basis = jax.nn.one_hot(idx, n, dtype=dtype)
        images = jax.vmap(A_matvec)(basis)
        diag_chunks.append(images[jnp.arange(stop - start), idx])
    return jnp.concatenate(diag_chunks)


def diag_EAET(E, A, E_T=None):
    """Compute ``diag(E @ A @ E^T)`` via probed matvecs (matrix-free)."""
    n = E.shape[0]
    if E_T is None:
        E_T = E.T
    dtype = getattr(A, "dtype", getattr(E, "dtype", DTYPE))
    return diag_matvec(lambda x: E @ (A @ (E_T @ x)), n, dtype=dtype)


def diag_EAET_matvec(E, A_matvec, n, E_T=None):
    """Compute ``diag(E @ A @ E^T)`` with ``A`` given as a matvec (matrix-free)."""
    if E_T is None:
        E_T = E.T
    dtype = getattr(E, "dtype", DTYPE)
    return diag_matvec(lambda x: E @ A_matvec(E_T @ x), n, dtype=dtype)


def diag_schur_complement(apply_DT, diag_inv, n):
    """Compute ``diag(D @ diag(diag_inv) @ D^T)`` via probed matvecs (matrix-free).

    For each row ``i``: ``e_i^T D diag(diag_inv) D^T e_i =
    ||diag_inv^{1/2} D^T e_i||^2``.
    """
    def entry(i):
        e_i = jnp.zeros(n).at[i].set(1.0)
        Dt_ei = apply_DT(e_i)
        return jnp.dot(Dt_ei, diag_inv * Dt_ei)
    return jax.lax.map(entry, jnp.arange(n), batch_size=mrx.MAP_BATCH_SIZE_OUTER)


# --------------------------------------------------------------------------- #
# Closed-form diagonal of the WEAK term of the Hodge Laplacian                 #
# --------------------------------------------------------------------------- #
#
#   L_k = S_k + W_k ,   W_k = D_{k-1} B_{k-1} D_{k-1}^T ,   D_l = E_k M_k G_l E_l^T
#
# with ``B`` the Kronecker mass model standing in for ``M_{k-1}^{-1}``.
# ``diag(S_k)`` is already closed form (:func:`mrx.local_assembly.
# build_stiffness_diagonal`); this is the other half, and it is what forced
# k>=1 Jacobi to probe the Laplacian at O(N) applies.
#
# Substituting ``B = (E+)^T K E+`` and folding the two pseudoinverses into the
# extraction projector ``Pi = E_l^T (E_l E_l^T)^{-1} E_l`` gives
#
#     W_k = E_k [ M_k G_l Pi ] K [ Pi G_l^T M_k ] E_k^T
#
# in which EVERY factor is a sum of Kronecker products.  A Kronecker product's
# diagonal is the outer product of its 1-D diagonals, so the whole thing
# collapses to a handful of small 1-D matrix chains plus one outer product per
# term pair -- O(N), against the O(N) full applies a probe needs.
#
#   M_k, M_{k-1}  ->  the Kronecker model ``Lam (x)_a A_a Lam`` of
#            :func:`_kron_mass_model_1d`, in all three places.
#   G_l  ->  exact, one Kronecker term per (out component, in component) pair.
#   Pi   ->  exact, see :func:`_extraction_projector_kron_terms`.
#
# Note ``K = Sig (x)_a Cinv_a Sig`` with ``Sig = Lam_l^-1`` is EXACTLY the
# inverse of the Kronecker model of ``M_{k-1}``.
#
# The one wrinkle is that ``Lam (x)_a A_a Lam`` is a Kronecker product SANDWICHED
# by a non-separable diagonal, and a diagonal does not push through a Kronecker
# factorization.  Of the six diagonals in ``M_k G M_{k-1}^-1 G^T M_k`` -- two per
# mass -- the two OUTERMOST are free: they multiply the finished diagonal
# pointwise, so they are kept EXACT and cost nothing.  The remaining four are
# interior (they land between a 1-D mass and the incidence) and are split rank-1
# in closed form by :func:`_rank1_diagonal_split`: no iteration, no fit, and no
# extra term pairs.  Measured against the same expansion with every ``Lam`` kept
# exact, the split costs a 2.4e-2 / 5.3e-3 / 2.5e-3 median at k=1/2/3.
#
# Do NOT replace M by its diagonal instead: that discards the mass coupling
# entirely and was rejected on 2026-08-18.
#
# Pi is NOT optional and NOT diagonal.  Both cheap surrogates were measured and
# both fail: masking (Pi ~ the bulk indicator) and the exact leverage diagonal
# (Pi ~ diag(Pi)) each leave ~90% error on the near-axis rows -- p99 8.9e-1 and
# 9.1e-1 against 3.1e-2 for the exact expansion, on a 10x8x6 toroid at k=1.

# Out component -> ((in component, differentiated axis, sign), ...) for G_l.
# Read off ``_apply_incidence_mf``: grad = (d_r, d_t, d_z);
# curl P = -d_z b + d_t c, Q = d_z a - d_r c, R = -d_t a + d_r b;
# div = d_r a + d_t b + d_z c.
_INCIDENCE_KRON_TERMS = {
    0: {0: ((0, 0, 1.0),),
        1: ((0, 1, 1.0),),
        2: ((0, 2, 1.0),)},
    1: {0: ((1, 2, -1.0), (2, 1, 1.0)),
        1: ((0, 2, 1.0), (2, 0, -1.0)),
        2: ((0, 1, -1.0), (1, 0, 1.0))},
    2: {0: ((0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0))},
}


def _raw_block_starts(shapes):
    starts = [0]
    for shape in shapes:
        starts.append(starts[-1] + int(np.prod(shape)))
    return np.asarray(starts)


def _decode_raw_indices(flat, shapes, starts):
    """Flat raw DOF indices -> (component, i_r, i_t, i_z)."""
    flat = np.asarray(flat)
    comp = np.searchsorted(starts, flat, side='right') - 1
    loc = flat - starts[comp]
    shape = np.asarray(shapes)[comp]
    nt, nz = shape[:, 1], shape[:, 2]
    return comp, loc // (nt * nz), (loc // nz) % nt, loc % nz


def _extraction_projector_kron_terms(e, shapes, *, tol=PROJECTOR_SVD_TOL):
    """Exact Kronecker expansion of ``Pi = E^T (E E^T)^{-1} E``.

    ``Pi`` is the identity on the bulk raw DOFs, zero on the raw DOFs that
    ``E`` drops, and a rank-deficient projector on the POLAR RING -- and the
    ring is only ``ring_depth`` radial indices thick, sits at a single zeta
    index per coupled row, and is the same block in every zeta plane.  So

        Pi = (+)_c diag(chi^c_r) x I_t x I_z  +  sum_{c,c'} sum_j F_r x F_t x I_z

    with ``chi^c_r`` the bulk radial indicator.  The ring blocks are split by an
    SVD in the ``(i_r, j_r)`` vs ``(i_t, j_t)`` grouping; that split is EXACT
    and its rank is at most ``ring_depth^2 <= 4`` per component pair, because
    the ring is radially thin.  Hence a handful of terms, not O(n_t).

    Every structural assumption is checked against the actual ``E`` and raises
    rather than silently degrading -- an unnoticed failure here is a ~90% error
    on the near-axis rows, which is exactly where a Jacobi diagonal matters.

    Returns a list of ``(src_component, dst_component, (F_r, F_t, F_z))``.

    Host-side sparsity bookkeeping (the SVD is of a ``ring_depth^2 x n_t^2``
    block of exact rank <= 4); the factors are cast to the working dtype where
    they meet the device, in :func:`build_weak_term_raw_diagonal`.
    """
    rows = np.asarray(e.rows)
    cols = np.asarray(e.cols)
    vals = np.asarray(e.vals)
    n_ext = int(e.forward_shape[0])
    n_raw = int(e.forward_shape[1])
    starts = _raw_block_starts(shapes)

    counts = np.bincount(rows, minlength=n_ext)
    coupled_rows = np.flatnonzero(counts > 1)
    # Nonzeros grouped by row ONCE, so a row's entries are a slice.
    order = np.argsort(rows, kind="stable")
    row_ptr = np.searchsorted(rows[order], np.arange(n_ext + 1))

    def entries(i):
        return order[row_ptr[i]:row_ptr[i + 1]]
    ring_cols = np.unique(cols[np.isin(rows, coupled_rows)])
    touched = np.unique(cols)
    bulk_cols = np.setdiff1d(touched, ring_cols)

    terms = []

    # --- identity part: the bulk radial indicator ---------------------------
    # Pi is exactly the identity on a bulk column and exactly zero on a dropped
    # one, so this part is a 0/1 diagonal; it is separable only if it is
    # constant over (theta, zeta) at fixed (component, i_r).
    if np.bincount(cols[np.isin(cols, bulk_cols)]).max(initial=0) > 1:
        raise ValueError(
            "extraction projector: a bulk raw DOF is shared by more than one "
            "extracted row, so Pi is not the identity there")
    mask = np.zeros(n_raw)
    mask[bulk_cols] = 1.0
    for c, shape in enumerate(shapes):
        block = mask[starts[c]:starts[c + 1]].reshape(shape)
        chi = block[:, 0, 0]
        if not np.array_equal(block, np.broadcast_to(chi[:, None, None], shape)):
            raise ValueError(
                f"extraction projector component {c}: the bulk indicator is not "
                "a radial slab, so the identity part of Pi is not separable")
        terms.append((c, c, (np.diag(chi), np.eye(shape[1]), np.eye(shape[2]))))

    if coupled_rows.size == 0:
        return terms

    # --- ring part ----------------------------------------------------------
    coupled, gram_inv, _ = _extraction_gram_inverse(e)
    coupled = np.asarray(coupled)
    gram_inv = np.asarray(gram_inv)
    pos = {int(r): t for t, r in enumerate(coupled)}

    _, _, _, ring_iz = _decode_raw_indices(ring_cols, shapes, starts)
    row_zeta = {}
    for i in coupled_rows:
        _, _, _, iz = _decode_raw_indices(cols[entries(i)], shapes, starts)
        if len(set(iz.tolist())) != 1:
            raise ValueError(
                f"extraction projector: coupled row {int(i)} spans more than one "
                "zeta index, so Pi is not block diagonal in zeta")
        row_zeta[int(i)] = int(iz[0])

    # Ring index space at one zeta plane: (component, ring radial index, theta).
    ring_comp, ring_ir, _, _ = _decode_raw_indices(ring_cols, shapes, starts)
    ring_radial = {c: np.unique(ring_ir[ring_comp == c])
                   for c in np.unique(ring_comp)}

    def plane_block(z0):
        """Dense ``Pi`` restricted to the ring at zeta index ``z0``."""
        plane_rows = [i for i in coupled_rows if row_zeta[int(i)] == z0]
        offsets, size = {}, 0
        for c, radial in ring_radial.items():
            offsets[c] = size
            size += len(radial) * shapes[c][1]
        cmat = np.zeros((len(plane_rows), size))
        for n, i in enumerate(plane_rows):
            sel = entries(i)
            comp, ir, it, _ = _decode_raw_indices(cols[sel], shapes, starts)
            for c, r, t, v in zip(comp, ir, it, vals[sel]):
                radial = ring_radial[int(c)]
                loc = int(np.searchsorted(radial, r)) * shapes[int(c)][1] + int(t)
                cmat[n, offsets[int(c)] + loc] += float(v)
        idx = np.array([pos[int(i)] for i in plane_rows])
        return cmat.T @ gram_inv[np.ix_(idx, idx)] @ cmat, offsets

    zetas = sorted(set(row_zeta.values()))
    block, offsets = plane_block(zetas[0])
    for z0 in zetas[1:]:
        other, _ = plane_block(z0)
        if np.abs(other - block).max() > PROJECTOR_PLANE_TOL * np.abs(block).max():
            raise ValueError(
                "extraction projector: the polar ring block differs between "
                "zeta planes, so Pi does not factor as (ring block) x I_z")

    for c1, radial1 in ring_radial.items():
        for c2, radial2 in ring_radial.items():
            c1, c2 = int(c1), int(c2)
            nt1, nt2 = shapes[c1][1], shapes[c2][1]
            if shapes[c1][2] != shapes[c2][2]:
                raise ValueError(
                    f"extraction projector: components {c1} and {c2} are coupled "
                    "by the polar ring but have different zeta dimensions")
            sub = block[offsets[c1]:offsets[c1] + len(radial1) * nt1,
                        offsets[c2]:offsets[c2] + len(radial2) * nt2]
            sub = sub.reshape(len(radial1), nt1, len(radial2), nt2)
            flat = sub.transpose(0, 2, 1, 3).reshape(
                len(radial1) * len(radial2), nt1 * nt2)
            u, s, vt = np.linalg.svd(flat, full_matrices=False)
            keep = s > max(tol * s[0], 0.0) if s.size else s
            for j in np.flatnonzero(keep):
                root = np.sqrt(s[j])
                f_r = np.zeros((shapes[c1][0], shapes[c2][0]))
                f_r[np.ix_(radial1, radial2)] = (
                    root * u[:, j]).reshape(len(radial1), len(radial2))
                f_t = (root * vt[j]).reshape(nt1, nt2)
                terms.append((c1, c2, (f_r, f_t, np.eye(shapes[c1][2]))))
    return terms


def _rank1_diagonal_split(tensor, *, mode: str = "geometric"):
    """Closed-form rank-1 separable split of a positive 3-D diagonal.

    Returns ``(f_r, f_t, f_z)`` with ``T ~ f_r x f_t x f_z``.  Each factor is
    the average of ``T`` over the other two axes, rescaled by the global mean so
    the product reproduces ``T`` EXACTLY whenever ``T`` really is separable.
    Three contractions, deterministic, symmetric in the axes -- no iteration and
    no fit to babysit.

    ``mode='geometric'`` averages ``log T`` -- the multiplicative version, and
    the default because ``T`` is a multiplicative scaling: against the unsplit
    raw_kron model of the weak term on a spline toroid it is 2-20x closer than
    the arithmetic split at every k (median 2.4e-2 / 5.3e-3 / 2.5e-3 at
    k=1/2/3, against 4.9e-2 / 4.1e-2 / 6.0e-2).  ``mode='arithmetic'`` averages
    ``T`` itself.  Both are exact on a separable tensor and differ only in how
    they spread a non-separable residual.
    """
    tensor = np.asarray(tensor)
    if mode == "geometric":
        work = np.log(tensor)
    elif mode == "arithmetic":
        work = tensor
    else:
        raise ValueError(f"unknown rank-1 split mode {mode!r}")
    mean = work.mean()
    factors = []
    for a in range(3):
        others = tuple(b for b in range(3) if b != a)
        factors.append(work.mean(axis=others) - (2.0 / 3.0) * mean
                       if mode == "geometric" else
                       work.mean(axis=others) / mean ** (2.0 / 3.0))
    return [np.exp(f) for f in factors] if mode == "geometric" else factors


def _rank1_split_residual(tensor, factors):
    """``T / (f_r x f_t x f_z)`` -- what :func:`_rank1_diagonal_split` missed.

    Positive, and identically 1 wherever ``T`` really is separable, so every
    correction built from it is a no-op on a separable weight.
    """
    return np.asarray(tensor) / np.einsum('i,j,l->ijl', *[np.asarray(f)
                                                          for f in factors])


def _axis_resample(m: int, n: int) -> Optional[np.ndarray]:
    """Linear interpolation from an ``m``-point axis onto an ``n``-point one.

    ``None`` for ``m == n`` (identity).  Only ever called with ``|m - n| = 1``:
    differentiating a clamped axis drops one DOF and a periodic axis none, so
    this is the two-point average between a degree-``p`` and a degree-``p-1``
    grid, written generically because which axis is differentiated depends on
    the component pair.
    """
    if m == n:
        return None
    src = np.linspace(0.0, 1.0, m)
    dst = np.linspace(0.0, 1.0, n)
    idx = np.clip(np.searchsorted(src, dst) - 1, 0, m - 2)
    w = (dst - src[idx]) / (src[idx + 1] - src[idx])
    out = np.zeros((n, m))
    out[np.arange(n), idx] = 1.0 - w
    out[np.arange(n), idx + 1] = w
    return out


def _transfer_tensor(tensor, shape_out) -> np.ndarray:
    """Resample a positive 3-D tensor onto ``shape_out``, axis by axis.

    Multiplicative (works on ``log T``), so the result stays positive and a
    constant tensor transfers to itself.
    """
    work = np.log(np.asarray(tensor))
    for a in range(3):
        T = _axis_resample(work.shape[a], int(shape_out[a]))
        if T is not None:
            work = np.moveaxis(np.tensordot(T, work, axes=([1], [a])), 0, a)
    return np.exp(work)


def _axis_index_delta(n: int, typ: str) -> np.ndarray:
    """``j - i`` on an ``n``-point axis, wrapped to the short way round if the
    axis is periodic.  Only ever multiplied against a banded 1-D mass, so the
    entries that survive are ``|j - i| <= p``.
    """
    d = np.arange(n)[None, :] - np.arange(n)[:, None]
    if typ == "periodic":
        d = (d + n // 2) % n - n // 2
    return d.astype(float)


def _log_index_gradient(tensor, types) -> list:
    """``d(log T) / d(index)`` per axis: central differences on the DOF grid,
    wrapping on periodic axes and one-sided at a clamped end.
    """
    work = np.log(np.asarray(tensor))
    out = []
    for a in range(3):
        if types[a] == "periodic":
            out.append(0.5 * (np.roll(work, -1, axis=a) - np.roll(work, 1, axis=a)))
        else:
            out.append(np.gradient(work, axis=a))
    return out


def _weak_term_kron_terms_raw(seq, k: int, *, dirichlet: bool):
    """Unscaled Kronecker terms of ``X = A^u [Lam] G Pi [Sig]``.

    Each term is ``(c_u, dst, sign, gf)`` with ``gf`` the three 1-D factors of
    ``G Pi`` -- everything BETWEEN the two inner scalings, which is the part
    that is genuinely separable.  What to do about the scalings in brackets is
    left to the caller: split them rank-1, expand them locally, or keep them
    exact.  ``ctx`` carries the 1-D masses, both scalings and the lower inverse.
    """
    from mrx.operators import _dense_incidence_1d  # noqa: PLC0415

    lower = k - 1
    shapes_u, mass_u, lam_u = _kron_mass_model_1d(seq, k)
    factors = build_mass_metric_lumping_factors(seq, lower, dirichlet=dirichlet)
    shapes_l = [tuple(int(s) for s in sh) for sh in factors.shapes]
    inv_l = [tuple(np.asarray(m) for m in factors.inv_1d[c])
             for c in range(len(shapes_l))]
    e_lower = getattr(seq, f"e{lower}_dbc" if dirichlet else f"e{lower}")
    pi_terms = _extraction_projector_kron_terms(e_lower, shapes_l)
    types = seq.basis_0.types

    terms = []
    for c_u, contributions in _INCIDENCE_KRON_TERMS[lower].items():
        for (c_g, axis, sign) in contributions:
            g = np.asarray(_dense_incidence_1d(shapes_l[c_g][axis], types[axis]))
            for a in range(3):
                if a != axis and shapes_u[c_u][a] != shapes_l[c_g][a]:
                    raise ValueError(
                        f"weak-term k={k}: undifferentiated axis {a} has size "
                        f"{shapes_l[c_g][a]} below and {shapes_u[c_u][a]} above")
            if g.shape != (shapes_u[c_u][axis], shapes_l[c_g][axis]):
                raise ValueError(
                    f"weak-term k={k}: incidence on axis {axis} has shape "
                    f"{g.shape}, expected "
                    f"{(shapes_u[c_u][axis], shapes_l[c_g][axis])}")
            for (src, dst, f_axes) in pi_terms:
                if src != c_g:
                    continue
                gf = [np.asarray(g @ f_axes[a]) if a == axis
                      else np.asarray(f_axes[a]) for a in range(3)]
                for a in range(3):
                    if gf[a].shape != (shapes_u[c_u][a], shapes_l[dst][a]):
                        raise ValueError(
                            f"weak-term k={k}: axis {a} factor has shape "
                            f"{gf[a].shape}, expected "
                            f"{(shapes_u[c_u][a], shapes_l[dst][a])}")
                terms.append((c_u, dst, sign, gf))

    ctx = {"shapes_u": shapes_u, "mass_u": mass_u, "lam_u": lam_u,
           "shapes_l": shapes_l, "inv_l": inv_l,
           "sigma": factors.inv_sqrt_D, "types": types}
    return terms, ctx


def _group_terms(terms) -> dict:
    """Terms bucketed by ``(upper component, Pi destination)``."""
    groups: dict = {}
    for (c_u, dst, sign, gf) in terms:
        groups.setdefault((c_u, dst), []).append((sign, gf))
    return groups


def _weak_term_kron_terms(seq, k: int, *, dirichlet: bool, split: str = "geometric",
                          rescale: str = "none"):
    """Kronecker terms of ``X = A^u Lam~^u G_{k-1} Pi Sig~``, grouped by the
    pair ``(upper component, lower component)`` that ``M_{k-1}^{-1}`` couples.

    Each entry is ``(sign, L, L_inv)`` with ``L`` the three 1-D factors and
    ``L_inv = L (A^l)^-1`` pre-multiplied, so a term PAIR costs one row-wise dot
    per axis instead of a matrix chain.

    Also returns a per-group multiplicative CORRECTION, see ``rescale`` in
    :func:`build_weak_term_raw_diagonal`; ``'none'`` gives all-ones.
    """
    terms, ctx = _weak_term_kron_terms_raw(seq, k, dirichlet=dirichlet)
    shapes_u, mass_u, lam_u = ctx["shapes_u"], ctx["mass_u"], ctx["lam_u"]
    shapes_l, inv_l, sigma = ctx["shapes_l"], ctx["inv_l"], ctx["sigma"]

    # The two INNER diagonal scalings, split rank-1 so they fold into the 1-D
    # factors. The upper Lam appears once more on the OUTSIDE, where it is kept
    # exact (see build_weak_term_raw_diagonal).
    lam_split = [_rank1_diagonal_split(lam_u[c], mode=split)
                 for c in range(len(shapes_u))]
    sigma_split = [_rank1_diagonal_split(sigma[c], mode=split)
                   for c in range(len(shapes_l))]

    # Leading-order repair of the two inner splits, off by default. Only the
    # residuals are needed; see build_weak_term_raw_diagonal for the counting.
    if rescale not in ("none", "upper", "both"):
        raise ValueError(f"unknown weak-term rescale mode {rescale!r}")
    resid_u = [_rank1_split_residual(lam_u[c], lam_split[c]) ** 2
               for c in range(len(shapes_u))] if rescale != "none" else None
    resid_l = [_rank1_split_residual(sigma[c], sigma_split[c]) ** 2
               for c in range(len(shapes_l))] if rescale == "both" else None

    groups: dict = {}
    for (c_u, dst, sign, gf) in terms:
        left = [((np.asarray(mass_u[c_u][a]) * lam_split[c_u][a][None, :])
                 @ gf[a]) * sigma_split[dst][a][None, :] for a in range(3)]
        groups.setdefault((c_u, dst), []).append(
            (sign, left, [left[a] @ inv_l[dst][a] for a in range(3)]))

    corr = {}
    for (c_u, dst) in groups:
        if rescale == "none":
            corr[(c_u, dst)] = None
            continue
        factor = resid_u[c_u]
        if resid_l is not None:
            factor = factor * _transfer_tensor(resid_l[dst], shapes_u[c_u])
        corr[(c_u, dst)] = factor
    return groups, shapes_u, lam_u, corr


def _weak_term_taylor_parts(seq, k: int, *, dirichlet: bool,
                            split: str = "geometric"):
    """``split='taylor1'``: expand the inner ``Lam`` LOCALLY instead of fitting
    it globally.

    The rank-1 split is a global fit to a quantity that is only ever sampled
    locally: ``A^u`` has bandwidth ``p``, so ``Lam_j`` enters only for ``j``
    within a few knots of the row ``i``.  Expand about the row instead::

        Lam_j ~ Lam_i (1 + g_i . (j - i)) ,   g = grad log Lam at i

    ``A_ij (j_a - i_a)`` is a first-moment 1-D mass -- still one small dense
    matrix per axis, still separable -- and ``g_a(i)`` is evaluated at the ROW,
    so like the outer ``Lam^2`` it multiplies pointwise and costs nothing.  Each
    term therefore splits into four (no moment, plus one per axis) and the pair
    sum is bucketed by which moments the two sides carry.

    The point of it: the error is now ``O(h^2 |grad^2 log Lam|)``, a local
    SMOOTHNESS assumption that refines away, instead of a global SEPARABILITY
    assumption that does not -- and the measured max error of the rank-1 split
    plateaus under refinement, which is what a global fit residual looks like.
    Positivity survives because the correction stays inside ``X``: the result is
    still ``diag(X~ B X~^T)`` for a genuine ``X~``.

    Only the upper ``Lam`` is expanded.  ``Sig`` sits against the DENSE (though
    exponentially decaying) ``(A^l)^-1``, where the locality argument is much
    weaker, so it keeps the rank-1 split given by ``split``.
    """
    terms, ctx = _weak_term_kron_terms_raw(seq, k, dirichlet=dirichlet)
    shapes_u, mass_u, lam_u = ctx["shapes_u"], ctx["mass_u"], ctx["lam_u"]
    shapes_l, inv_l, sigma = ctx["shapes_l"], ctx["inv_l"], ctx["sigma"]
    types = ctx["types"]

    sigma_split = [_rank1_diagonal_split(sigma[c], mode=split)
                   for c in range(len(shapes_l))]
    grads = [_log_index_gradient(lam_u[c], types) for c in range(len(shapes_u))]
    deltas = [[_axis_index_delta(shapes_u[c][a], types[a]) for a in range(3)]
              for c in range(len(shapes_u))]

    entries: dict = {}
    for (c_u, dst, sign, gf) in terms:
        for v in range(4):  # 0 = plain mass, v = 1..3 = first moment on axis v-1
            fac = []
            for a in range(3):
                m = np.asarray(mass_u[c_u][a])
                if v == a + 1:
                    m = m * deltas[c_u][a]
                fac.append((m @ gf[a]) * sigma_split[dst][a][None, :])
            entries.setdefault((c_u, dst), []).append(
                (sign, v, fac, [fac[a] @ inv_l[dst][a] for a in range(3)]))

    parts = [np.zeros(shape) for shape in shapes_u]
    n_pairs = 0
    for (c_u, dst), group in entries.items():
        signs = np.asarray([g[0] for g in group])
        vs = np.asarray([g[1] for g in group])
        z, w = _pair_products(signs, [g[2] for g in group],
                              [g[3] for g in group])
        n_pairs += (len(group) * (len(group) + 1)) // 2
        vmin, vmax = np.minimum.outer(vs, vs), np.maximum.outer(vs, vs)
        blocks = {}
        for key in set(zip(vmin.ravel().tolist(), vmax.ravel().tolist())):
            mask = (vmin == key[0]) & (vmax == key[1])
            blocks[key] = _pair_sum(z, np.where(mask, w, 0.0))
        total = np.zeros(shapes_u[c_u])
        for (v_i, v_j), block in blocks.items():
            coef = 1.0
            if v_i:
                coef = coef * grads[c_u][v_i - 1]
            if v_j:
                coef = coef * grads[c_u][v_j - 1]
            total += coef * block
        # The Taylor prefactor: Lam_i factors out of the row on both sides. The
        # OUTER Lam^2 is applied by the caller, so the model carries Lam^4 in
        # all -- the same power the split form carries when Lam is constant.
        parts[c_u] += np.asarray(lam_u[c_u]) ** 2 * total
    return parts, lam_u, n_pairs


def _pair_products(signs, left, left_inv):
    """Row-wise dots of every term PAIR, batched over the stacked terms.

    ``left[t][a]`` and ``left_inv[t][a]`` are the ``(m_a, n_a)`` 1-D factors
    of term ``t``. Returns ``(z, w)`` with ``z[a][t, s, i] =
    sum_n left_inv[t][a][i, n] left[s][a][i, n]`` and the pair weights
    ``w[t, s] = sign_t sign_s`` -- the full double sum over ``(t, s)``, which
    equals the ``t <= s`` sum with off-diagonal pairs counted twice.
    """
    z = [np.einsum('tin,sin->tsi', np.stack([li[a] for li in left_inv]),
                   np.stack([lf[a] for lf in left])) for a in range(3)]
    return z, np.outer(signs, signs)


def _pair_sum(z, w):
    """``sum_{t,s} w[t,s] z_r[t,s,:] x z_t[t,s,:] x z_z[t,s,:]`` as one einsum
    over the flattened pair index -- no Python loop over pairs."""
    p = w.size
    return np.einsum('pi,pj,pl->ijl', (w[..., None] * z[0]).reshape(p, -1),
                     z[1].reshape(p, -1), z[2].reshape(p, -1))


def _weak_term_exact_parts(seq, k: int, *, dirichlet: bool):
    """``split='exact'``: the same expansion with BOTH inner scalings kept
    exact -- the oracle that separates the two error sources.

    The closed form carries two independent approximations: the mass model
    (``M~`` vs ``M``, Kronecker 1-D masses) and the rank-1 split of the inner
    scalings.  Against the operator probe they are measured together.  This
    path removes the second one, so ``probe - exact`` is the mass-model error
    and ``exact - closed`` is the split error.

    There is no cheap way to do it -- an exact non-separable diagonal between
    two Kronecker factors is precisely what does not factorize, which is why
    the split exists at all.  So this forms ``X`` DENSELY per group, one
    ``(N_u x N_l)`` matrix, and is a diagnostic at A/B resolution only, never
    production.  ``MRX_WEAK_EXACT_MAXDIM`` (default 2e7 entries, ~160 MB) caps
    it rather than letting it OOM a node.
    """
    terms, ctx = _weak_term_kron_terms_raw(seq, k, dirichlet=dirichlet)
    shapes_u, mass_u, lam_u = ctx["shapes_u"], ctx["mass_u"], ctx["lam_u"]
    shapes_l, inv_l, sigma = ctx["shapes_l"], ctx["inv_l"], ctx["sigma"]

    max_dense = float(os.environ.get("MRX_WEAK_EXACT_MAXDIM", 2e7))
    parts = [np.zeros(shape) for shape in shapes_u]
    a_kron: dict = {}
    for (c_u, dst), group in _group_terms(terms).items():
        n_u = int(np.prod(shapes_u[c_u]))
        n_l = int(np.prod(shapes_l[dst]))
        if n_u * n_l > max_dense or n_l * n_l > max_dense:
            raise MemoryError(
                f"weak-term split='exact' k={k} needs a dense {n_u}x{n_l} "
                f"block ({n_u * n_l:.3g} entries) over the "
                f"{max_dense:.3g} cap. It is a diagnostic oracle; run it at "
                "A/B resolution or raise MRX_WEAK_EXACT_MAXDIM.")
        # X = (x)A^u . D_Lam . [sum_t sign_t (x)gf_t] . D_Sig -- the bracket is
        # the only part that differs between terms, so it is summed first and
        # the two dense products are paid once per group.
        y = np.zeros((n_u, n_l))
        for (sign, gf) in group:
            y += sign * np.kron(np.kron(gf[0], gf[1]), gf[2])
        if c_u not in a_kron:
            a_kron[c_u] = np.kron(np.kron(np.asarray(mass_u[c_u][0]),
                                          np.asarray(mass_u[c_u][1])),
                                  np.asarray(mass_u[c_u][2]))
        x = a_kron[c_u] @ (np.asarray(lam_u[c_u]).reshape(-1)[:, None] * y)
        x *= np.asarray(sigma[dst]).reshape(-1)[None, :]
        v = np.kron(np.kron(inv_l[dst][0], inv_l[dst][1]), inv_l[dst][2])
        parts[c_u] += ((x @ v) * x).sum(axis=1).reshape(shapes_u[c_u])
    return parts, lam_u


def build_weak_term_raw_diagonal(seq, k: int, *, dirichlet: bool,
                                 split: Optional[str] = None,
                                 rescale: Optional[str] = None,
                                 return_info: bool = False):
    """Raw-DOF-space diagonal of the weak term, closed form and O(N).

    ``diag(W)`` for ``W = M_k G Pi M_{k-1}^{-1} Pi G^T M_k`` with every mass
    replaced by the Kronecker model of :func:`_kron_mass_model_1d`.  Each
    term of the expansion is
    then a pure Kronecker product, whose diagonal is the OUTER PRODUCT of its
    three 1-D diagonals::

        diag(term_{t,t'})_i = Lam_i^2  prod_a [ L^t_a (A^l_a)^-1 (L^t'_a)^T ]_{i_a i_a}

    with ``L^t_a = A^u_a lam_a g_a F_a sig_a`` -- upper 1-D mass, the rank-1
    split of the inner scaling, the 1-D incidence, one Kronecker factor of
    ``Pi``, and the rank-1 split of the lower scaling.  The pair sum is
    symmetric, so only ``t <= t'`` is formed.

    The OUTER ``Lam^2`` is the exact, unsplit scaling: it multiplies the
    finished diagonal pointwise, so it costs nothing and needs no separability.
    Only the two inner copies are split.

    **split** -- how the two inner scalings are handled.  Global default from
    ``MRX_LAPLACIAN_DIAG_SPLIT``.

    * ``'geometric'`` (default) / ``'arithmetic'`` -- rank-1 split, see
      :func:`_rank1_diagonal_split`.
    * ``'taylor1'`` -- local first-order expansion about the row instead of a
      global fit, see :func:`_weak_term_taylor_parts`.
    * ``'exact'`` -- no separation at all; dense, diagnostic only, see
      :func:`_weak_term_exact_parts`.
    * ``'codiff'`` -- abandons the expansion entirely for the codifferential
      energy ``||delta phi_i||^2``, by quadrature; k=3 only so far.  See
      :func:`mrx.local_assembly.build_codifferential_diagonal`.

    **rescale** -- leading-order repair of the inner splits, off by default and
    settable globally with ``MRX_LAPLACIAN_DIAG_RESCALE``.  Write the model as

        W = Lam A [Lam] G Pi Sig [A^l]^-1 Sig Pi G^T [Lam] A Lam

    Of the four ``Lam`` the two outer are exact; the two bracketed inner ones
    are split, as are the two ``Sig = Lam_l^-1`` inside ``B``.  ``A`` is a 1-D
    mass, so its bandwidth is ``p``: the inner scaling is sampled at rows ``j``
    a few knots from ``i``, and to leading order in ``h`` it is just
    ``Lam_i / lam_i``.  Two copies, so:

    * ``'upper'`` multiplies each group by ``(Lam / lam)^2`` on the upper grid.
      That ratio IS ``diag(M_k) / diag(M^_k)``: the split breaks the Kronecker
      model's defining property that it reproduces the exact mass diagonal, and
      this restores it.  Note the exponent -- the handoff note said
      ``(diag(M)/diag(M^))^2``, which double counts; there are two inner
      copies, not four.
    * ``'both'`` additionally multiplies by ``(Sig / sig)^2`` for the lower
      component of the group, resampled onto the upper grid (linear per axis,
      in log space).  The transfer is the approximation here: ``[A^l]^-1`` is
      dense, though exponentially decaying, so the locality argument is weaker
      than for the upper half.

    Both are free -- the residuals are byproducts of the split already computed
    -- exact on a separable weight, and positive, so each group's contribution
    stays ``diag(X_g [A^l]^-1 X_g^T) >= 0`` and the total stays positive.
    ``'upper'`` is the zeroth-order term of the ``'taylor1'`` series; measured
    against the probe it helps only where the weak term IS the operator (k=3)
    and costs iterations at k=1/2, so both stay off by default.
    
    KNOWN, MEASURED, UNFIXED -- this closed form is calibrated for a mass
    preconditioner that no longer exists. It models ``D M^-1 D^T`` under the
    plain Kronecker mass model, which used to be the production ``M^-1``.
    Against metric_lumping the model's error versus the exact operator grows from
    ~2-4% median / ~30% max to **22% median / 114% max** (k=1 dbc, spline
    toroid 8,16,8 p=2). The right fix is to model the new mass, not to widen a
    bound.

    Practical cost, measured in ``outputs/diag_masslap`` before the swap:
    ``kind='jacobi'`` iteration counts move by 1-10% (cylinder k=1 free
    262 -> 287, W7-X k=1 free 1658 -> 1668), while the production
    ``kind='metric_lumping'`` gets ~9% better. ``_probed_laplacian_diaginv``
    is exact and unaffected.

    No test pins this bound any more: the one that did was gated on the
    plain Kronecker model being the mass kind, so it became a permanent skip
    and was removed. The measurement it carried lives here instead, and
    re-deriving the bound for metric_lumping is open work.
"""
    if k not in (1, 2, 3):
        raise ValueError("the weak term exists only for k = 1, 2, 3")
    if split is None:
        split = os.environ.get("MRX_LAPLACIAN_DIAG_SPLIT", "geometric")
    if rescale is None:
        rescale = os.environ.get("MRX_LAPLACIAN_DIAG_RESCALE", "none")

    info = {"split": split, "rescale": rescale}
    if split == "codiff":
        # Not an expansion at all: diag(W)_i = ||delta phi_i||^2 by quadrature.
        # No mass model, no Sig, no separability assumption -- see
        # mrx.local_assembly.build_codifferential_diagonal.
        from mrx.local_assembly import (  # noqa: PLC0415
            build_codifferential_diagonal)
        raw = jnp.asarray(build_codifferential_diagonal(seq, k))
        return (raw, info) if return_info else raw
    if split == "exact":
        parts, lam_u = _weak_term_exact_parts(seq, k, dirichlet=dirichlet)
    elif split == "taylor1":
        parts, lam_u, info["term_pairs"] = _weak_term_taylor_parts(
            seq, k, dirichlet=dirichlet)
    else:
        groups, shapes_u, lam_u, corr = _weak_term_kron_terms(
            seq, k, dirichlet=dirichlet, split=split, rescale=rescale)
        parts = [np.zeros(shape) for shape in shapes_u]
        n_pairs = 0
        for key, entries in groups.items():
            c_u = key[0]
            z, w = _pair_products(np.asarray([s for s, _, _ in entries]),
                                  [lf for _, lf, _ in entries],
                                  [li for _, _, li in entries])
            block = _pair_sum(z, w)
            n_pairs += (len(entries) * (len(entries) + 1)) // 2
            # Per GROUP, not per term: the group is one diag(X B_g X^T) with
            # B_g SPD, so it is nonnegative and a positive rescale cannot flip
            # it.
            parts[c_u] += block if corr[key] is None else block * corr[key]
        info["term_pairs"] = n_pairs
        info["terms"] = {key: len(v) for key, v in groups.items()}

    raw = jnp.asarray(np.concatenate([(p * np.asarray(lam_u[c]) ** 2).reshape(-1)
                                      for c, p in enumerate(parts)]),
                      dtype=DTYPE)
    return (raw, info) if return_info else raw


def _weak_term_rows_by_apply(seq, operators, k: int, *, dirichlet: bool, indices):
    """Exact ``diag(W)`` at a handful of extracted rows, by operator applies."""
    from mrx.operators import (apply_derivative_matrix,  # noqa: PLC0415
                               apply_mass_matrix_preconditioner)

    lower = k - 1
    suffix = "_dbc" if dirichlet else ""
    size = int(getattr(seq, f"n{k}{suffix}"))

    def weak_apply(x):
        d_t_x = apply_derivative_matrix(
            seq, operators, x, lower, dirichlet_in=dirichlet,
            dirichlet_out=dirichlet, transpose=True)
        inner = apply_mass_matrix_preconditioner(
            seq, operators, d_t_x, lower, dirichlet=dirichlet, kind='auto')
        return apply_derivative_matrix(
            seq, operators, inner, lower, dirichlet_in=dirichlet,
            dirichlet_out=dirichlet)

    def row(i):
        return weak_apply(jnp.zeros(size, dtype=DTYPE).at[i].set(1.0))[i]

    # Warm the apply on a concrete vector first: the matrix-free mass plan is
    # HOST-built, so building it inside the trace raises
    # TracerArrayConversionError.
    weak_apply(jnp.zeros(size, dtype=DTYPE))
    # lax.map in SMALL batches, never a wide vmap: a batched probe fuses into
    # a transpose kernel that spills registers and crashes ptxas. See
    # _diagonal_from_matvec.
    return np.asarray(jax.lax.map(row, jnp.asarray(indices),
                                  batch_size=PROBE_BATCH_SIZE))


def build_weak_term_diagonal(seq, operators, k: int, *, dirichlet: bool, **kwargs):
    """``diag(E_k W_k E_k^T)``, the weak half of the k>=1 Jacobi Laplacian.

    Bulk extracted rows are pure selectors, so the closed-form raw diagonal
    supplies them directly.  The ``n_polar * n_zeta`` coupled rows would need
    off-diagonal raw entries of ``W``; they are taken EXACTLY instead, by one
    operator apply each.  That is a handful of applies against the probe's one
    per extracted row -- and it puts the exact value on the near-axis rows,
    which is where the Kronecker mass model is least accurate.
    """
    raw = np.asarray(build_weak_term_raw_diagonal(
        seq, k, dirichlet=dirichlet, **kwargs))

    e = getattr(seq, f"e{k}_dbc" if dirichlet else f"e{k}")
    rows = np.asarray(e.rows)
    cols = np.asarray(e.cols)
    vals = np.asarray(e.vals)
    n_ext = int(e.forward_shape[0])
    counts = np.bincount(rows, minlength=n_ext)

    diag = np.zeros(n_ext)
    single = counts[rows] == 1
    diag[rows[single]] = (vals[single] ** 2) * raw[cols[single]]

    coupled = np.flatnonzero(counts > 1)
    if coupled.size:
        diag[coupled] = _weak_term_rows_by_apply(
            seq, operators, k, dirichlet=dirichlet, indices=coupled)
    return jnp.asarray(diag, dtype=DTYPE)


def build_extracted_laplacian_diagonal(seq, operators, k: int, *, dirichlet: bool,
                                       **kwargs):
    """``diag(E_k L_k E_k^T)`` for ``k >= 1``, with no O(N) probe.

    ``L_k = S_k + W_k``.  Both halves are closed form in the raw DOF space --
    :func:`mrx.local_assembly.build_stiffness_diagonal` exactly, and
    :func:`build_weak_term_raw_diagonal` under the Kronecker mass model -- and
    the bulk rows of ``E`` are pure selectors, so the raw diagonal transfers
    straight through.  The ``n_polar * n_zeta`` coupled rows are taken exactly,
    by one apply of ``L_k`` each.

    ``L_0 = S_0`` has no weak term at all and is handled by
    :func:`mrx.local_assembly.build_extracted_stiffness_diagonal_k0`.
    """
    from mrx.local_assembly import build_stiffness_diagonal  # noqa: PLC0415
    from mrx.operators import apply_hodge_laplacian_approx  # noqa: PLC0415

    if k not in (1, 2, 3):
        raise ValueError("use build_extracted_stiffness_diagonal_k0 for k=0")

    raw = (np.asarray(build_stiffness_diagonal(seq, k))
           + np.asarray(build_weak_term_raw_diagonal(
               seq, k, dirichlet=dirichlet, **kwargs)))

    e = getattr(seq, f"e{k}_dbc" if dirichlet else f"e{k}")
    rows = np.asarray(e.rows)
    cols = np.asarray(e.cols)
    vals = np.asarray(e.vals)
    n_ext = int(e.forward_shape[0])
    counts = np.bincount(rows, minlength=n_ext)

    diag = np.zeros(n_ext)
    single = counts[rows] == 1
    diag[rows[single]] = (vals[single] ** 2) * raw[cols[single]]

    coupled = np.flatnonzero(counts > 1)

    # MRX_LAPLACIAN_DIAG_EXACT_RINGS=n also takes the n innermost radial rings
    # exactly. Every closed-form and transfer model measured so far degrades
    # sharply on the ring next to the polar block -- the transfer routes by
    # 20-50x, because V_3's extraction is unitary while V_0's folds that ring
    # into polar rows -- and it is a thin set, O(n_theta n_zeta) applies, the
    # same mechanism the coupled rows already use.
    n_rings = int(os.environ.get("MRX_LAPLACIAN_DIAG_EXACT_RINGS", "0"))
    if n_rings > 0:
        shapes_k = [tuple(int(v) for v in sh)
                    for sh in getattr(seq, f"basis_{k}").shape]
        starts_k = np.cumsum([0] + [int(np.prod(sh)) for sh in shapes_k])
        single_rows, single_cols = rows[single], cols[single]
        comp = np.searchsorted(starts_k[1:], single_cols, side="right")
        loc = single_cols - starts_k[comp]
        nt = np.array([sh[1] for sh in shapes_k])[comp]
        nz = np.array([sh[2] for sh in shapes_k])[comp]
        i_r = loc // (nt * nz)
        coupled = np.union1d(coupled, single_rows[i_r < n_rings])

    if coupled.size:
        size = int(getattr(seq, f"n{k}_dbc" if dirichlet else f"n{k}"))

        def row(i):
            x = jnp.zeros(size, dtype=DTYPE).at[i].set(1.0)
            return apply_hodge_laplacian_approx(
                seq, operators, x, k, dirichlet=dirichlet)[i]

        # Warm the apply outside the trace: its matrix-free mass plan is
        # host-built and cannot be constructed on tracers.
        apply_hodge_laplacian_approx(
            seq, operators, jnp.zeros(size, dtype=DTYPE), k,
            dirichlet=dirichlet)
        # lax.map in small batches, never a wide vmap -- see
        # _diagonal_from_matvec.
        diag[coupled] = np.asarray(jax.lax.map(row, jnp.asarray(coupled),
                                               batch_size=PROBE_BATCH_SIZE))
    return jnp.asarray(diag, dtype=DTYPE)
