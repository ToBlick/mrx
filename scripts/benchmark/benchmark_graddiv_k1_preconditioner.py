"""Diagnostic: additive grad-div subspace preconditioner for the k=1 Hodge Laplacian.

We precondition the statically condensed 1-form operator

    A = K_1 + D_0 M_0^{-1} D_0^T

where ``K_1`` is the discrete curl-curl (1-form stiffness) and
``D_0 M_0^{-1} D_0^T`` is the grad-div penalty obtained by eliminating the
scalar multiplier from the mixed system. To avoid a Krylov-in-Krylov solve the
inner ``M_0^{-1}`` is replaced by a single tensor mass-preconditioner apply
(``apply_laplacian_approx``), giving a linear SPD matvec for ``A``.

Two upper-block preconditioners are compared. The script can run either on
the condensed operator (CG) or on the full saddle-point system (MINRES):

* ``jacobi (diag)`` -- the current production best: the Schur-outer Jacobi
                    preconditioner assembled with ``schur_diag_mode='tensor_probe'``
                    (rank-independent ``diag(M_0)^{-1}`` inner probe). A single
                    stored diagonal multiply. Pairs with the tensor inner that
                    ``A`` already bakes in, i.e. "jacobi outer + tensor inner".
* ``jacobi(K) + P_B`` -- additive upper preconditioner combining the Jacobi
                    inverse on the diagonal of ``K_1`` with the dual-Poisson
                    potential correction block

      P_B = G_0 L_0^{-1} M_0 L_0^{-1} G_0^T

  implemented as the 5-step pipeline below. ``L_0^{-1}`` is the rank-1 tensor
  k=0 Hodge-Laplacian preconditioner; ``G_0`` / ``G_0^T`` are the incidence
  (gradient / weak-divergence) operators; ``M_0`` is the scalar mass matrix.

  Pipeline (input r is a 1-form dual residual):
      y1 = G_0^T r          # V1* -> V0*
      y2 = L_0^{-1} y1      # V0* -> V0
      y3 = M_0 y2           # V0  -> V0*
      y4 = L_0^{-1} y3      # V0* -> V0
      u  = G_0 y4           # V0  -> V1

K_1 alone leaves the gradient subspace (penalised at fourth order by the
grad-div term) badly conditioned; P_B is the inverse of that fourth-order block
on the gradient subspace and should sharply cut iteration counts.

Geometry: axisymmetric toroid map, ns=(6, 12, 4), p=3, BETTI=(1, 1, 0, 0).
Boundary condition: DBC on the 1-form. For BETTI=(1, 1, 0, 0) the k=1 DBC
harmonic count is b2 = 0 and k=0 DBC has no constants, so the saddle system is
nonsingular. This makes it a clean test for upper-preconditioner quality while
keeping the lower block fixed to the rank-1 tensor mass preconditioner.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import NamedTuple
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
for _p in (ROOT, SCRIPTS, SCRIPTS / "benchmark", SCRIPTS / "debug"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import cylinder_map, rotating_ellipse_map, toroid_map
from mrx.differential_forms import safe_inv33
from mrx.operators import (
    _diagonal_from_matvec,
    _invert_diagonal,
    _get_schur_diaginv,
    _nullspace_vectors,
    _build_k1_stiffness_surgery_factors,
    _mass_extraction,
    apply_derivative_matrix,
    apply_laplacian_approx,
    apply_laplacian_preconditioner,
    apply_incidence_matrix,
    apply_mass_matrix,
    apply_mass_matrix_preconditioner,
    apply_projection_matrix,
    apply_stiffness_tensor_preconditioner,
    apply_stiffness,
    assemble_laplacian_operators,
    assemble_incidence_operators,
    assemble_mass_jacobi_preconditioner,
    assemble_projection_operators,
    assemble_schur_jacobi_preconditioner,
    assemble_tensor_laplacian_preconditioner,
    assemble_tensor_mass_preconditioner,
    assemble_tensor_stiffness_preconditioner,
)
from mrx.preconditioners import (
    _apply_extracted_submatrix,
    _apply_k1_bulk_diagonal_preconditioner,
    _apply_k1_bulk_forward_model,
    _apply_k1_bulk_preconditioner,
    _apply_k2_bulk_diagonal_preconditioner,
    _apply_k2_bulk_preconditioner,
    _apply_bulk_to_surgery_coupling,
    _apply_surgery_to_bulk_coupling,
    _apply_tensor_diagonal_block_preconditioner,
    _symmetrize,
)
from mrx.solvers import solve_saddle_point_minres, solve_singular_cg

jax.config.update("jax_enable_x64", True)

TYPES = ("clamped", "periodic", "periodic")
BETTI = (1, 1, 0, 0)
DIRICHLET = True  # DBC 1-form -> zero harmonics for BETTI=(1,1,0,0)
RANK = 1
AUX0_DIRICHLET = False  # prototype choice: all three auxiliary 0-form channels free

# The raw "directly built" incidence E^T sp E. On the POLAR sequence this is NOT
# the true topological derivative (E^T E != I at the axis) and is not nilpotent.
# The TRUE derivative is G = M^{-1} D = Gram^{-1} . (E^T sp E), Gram = E^T E (the
# cheap coefficient Gram, NO mass assembly; block-diagonal: identity bulk + small
# axis block). We keep a handle to the raw operator so projectors can opt into
# the true G via a local name-shadow (see make_apply_routines / *_k2).
_apply_incidence_raw = apply_incidence_matrix


def _build_gram_inv(seq, ops, k: int, dirichlet: bool):
    """Dense (E_k^T E_k)^{-1} for the extraction of space k. Cheap: no mass."""
    e, e_T = _mass_extraction(ops, k, dirichlet)
    n = int(getattr(seq, f"n{k}_dbc" if dirichlet else f"n{k}"))
    cols = []
    for j in range(n):
        ej = jnp.zeros((n,), dtype=jnp.float64).at[j].set(1.0)
        cols.append(np.asarray(jax.device_get(e @ (e_T @ ej))))
    gram = np.stack(cols, axis=1)
    return jnp.asarray(np.linalg.inv(gram))


def _make_true_incidence(gram_invs):
    """Return a drop-in replacement for apply_incidence_matrix that applies the
    TRUE derivative G = Gram_{k+1}^{-1} . (E^T sp E) on the spaces in gram_invs
    (keyed by OUTPUT space index). Falls back to the raw incidence elsewhere.
    G^T = (E^T sp E)^T . Gram_{k+1}^{-1} keeps it a proper adjoint."""
    def true_inc(seq, ops, v, k, dirichlet_in=True, dirichlet_out=True, transpose=False):
        Ginv = gram_invs.get(k + 1)
        if Ginv is None:
            return _apply_incidence_raw(seq, ops, v, k, dirichlet_in=dirichlet_in,
                                        dirichlet_out=dirichlet_out, transpose=transpose)
        if transpose:  # V_{k+1} -> V_k : (E^T sp E)^T applied to Gram^{-1} v
            return _apply_incidence_raw(seq, ops, Ginv @ v, k, dirichlet_in=dirichlet_in,
                                        dirichlet_out=dirichlet_out, transpose=True)
        return Ginv @ _apply_incidence_raw(seq, ops, v, k, dirichlet_in=dirichlet_in,
                                           dirichlet_out=dirichlet_out, transpose=False)
    return true_inc


class K1VectorFDBulkState(NamedTuple):
    surgery: object
    arr_f: object
    theta_f: object
    zeta_f: object
    inv_denom_r: jnp.ndarray
    inv_denom_t: jnp.ndarray
    inv_denom_z: jnp.ndarray
    inv_blocks: jnp.ndarray
    n_r: int
    n_t: int
    n_z: int
    n_common: int


class K1VectorFDTrueBasisBulkState(NamedTuple):
    surgery: object
    vec_r: jnp.ndarray
    vec_t: jnp.ndarray
    vec_z: jnp.ndarray
    inv_eval_r: jnp.ndarray
    inv_eval_t: jnp.ndarray
    inv_eval_z: jnp.ndarray
    inv_blocks: jnp.ndarray
    n_r: int
    n_t: int
    n_z: int
    n_common: int


class K1RadialBandedBulkState(NamedTuple):
    """Modal bulk inverse that keeps the full radial+component coupling per
    angular mode.

    The same-mode 3x3 vector-FD symbol assumes the operator, in the
    per-component fast-diag basis ``V``, only connects mode ``m`` to ``m``.
    The per-axis leakage diagnostic shows the missed energy is ~97% radial:
    ``curl`` contains ``d_r`` which is not diagonal in the radial fast-diag
    basis. This state replaces the diagonal-radial / same-mode-3x3 model by a
    dense block over ``(component, radial mode)`` at each fixed angular mode
    ``(i_t, i_z)`` -- exact in radial+component coupling, block-diagonal (the
    measured ~3% residual) in the angular indices.
    """

    surgery: object
    arr_f: object
    theta_f: object
    zeta_f: object
    nt: int
    nz: int
    r0: int
    r1: int
    r2: int
    bulk_r: int
    bulk_t: int
    bulk_z: int
    inv_blocks: jnp.ndarray


class K1BlockFDPreconditionerState(NamedTuple):
    surgery: object
    factors: object
    schur_inv: jnp.ndarray
    use_inner_schur: bool
    vector_fd_state: object
    vector_fd_true_basis_state: object
    radial_banded_state: object = None


def _build_c_matrix(seq: DeRhamSequence) -> jnp.ndarray:
    """Dense cross-coupling matrix C: extracted V1 (primal) -> extracted W* (dual).

    C[alpha*n0_ext + i, j] = integral phi^0_i(xi) [phi^{1,ref_alpha}_j(xi)] J(xi) dxi

    where alpha = 0,1,2 indexes the reference 1-form component.
    Rows are the three auxiliary 0-form channels (free BCs, AUX0_DIRICHLET=False).
    Columns are the extracted k=1 DBC DOFs.

    P_A = C^T L0^{-3} C  is then symmetric PSD by construction.
    Returns shape (3*n0_ext, n1_ext).
    """
    nq_r = seq.quad.nx
    nq_t = seq.quad.ny
    nq_z = seq.quad.nz

    # Basis arrays: shape (n_basis, n_quad_1d).
    # evaluate_1d populates these as (n_basis, nq_1d) by outer/inner vmap.
    lam_r = seq.basis_r_jk    # (n_r, nq_r)
    lam_t = seq.basis_t_jk    # (n_t, nq_t)
    lam_z = seq.basis_z_jk    # (n_z, nq_z)
    d_lam_r = seq.d_basis_r_jk  # (n_dr, nq_r)
    d_lam_t = seq.d_basis_t_jk  # (n_dt, nq_t)
    d_lam_z = seq.d_basis_z_jk  # (n_dz, nq_z)

    n_r = int(lam_r.shape[0])
    n_t = int(lam_t.shape[0])
    n_z = int(lam_z.shape[0])
    n_dr = int(d_lam_r.shape[0])
    n_dt = int(d_lam_t.shape[0])
    n_dz = int(d_lam_z.shape[0])
    n0_full = n_r * n_t * n_z
    n1_1_full = int(seq.basis_1.n1)  # = n_dr * n_t * n_z
    n1_2_full = int(seq.basis_1.n2)  # = n_r * n_dt * n_z

    # Jacobian: flat shape is (nq_t * nq_r * nq_z) in 'xy' meshgrid ordering.
    # Reshape to (nq_r, nq_t, nq_z) via the same transpose as _split_field.
    J = seq.geometry.jacobian_j.reshape(nq_t, nq_r, nq_z).transpose(1, 0, 2)

    # Combined weight W[qr, qt, qz] = J * w_r x w_t x w_z
    W = (J
         * seq.quad.w_x[:, None, None]
         * seq.quad.w_y[None, :, None]
         * seq.quad.w_z[None, None, :])

    def _block(row_bases, col_bases):
        """Sum-factorized cross block.

        row_bases / col_bases: (B_r, B_t, B_z) each (n_basis, nq_1d).
        Returns shape (n_r0, n_dr1, n_t0, n_dt1, n_z0, n_dz1).
        """
        Br, Bt, Bz = row_bases
        Bcr, Bct, Bcz = col_bases
        A = jnp.einsum('iR,jR,RTZ->ijTZ', Br, Bcr, W)
        B = jnp.einsum('kT,lT,ijTZ->ijklZ', Bt, Bct, A)
        return jnp.einsum('mZ,nZ,ijklZ->ijklmn', Bz, Bcz, B)

    # Component 0: 0-form rows, 1-form component 0 cols (d_r, t, z)
    C0_raw = _block(
        (lam_r, lam_t, lam_z), (d_lam_r, lam_t, lam_z)
    ).transpose(0, 2, 4, 1, 3, 5).reshape(n0_full, n1_1_full)

    # Component 1: 0-form rows, 1-form component 1 cols (r, d_t, z)
    C1_raw = _block(
        (lam_r, lam_t, lam_z), (lam_r, d_lam_t, lam_z)
    ).transpose(0, 2, 4, 1, 3, 5).reshape(n0_full, n1_2_full)

    # Component 2: 0-form rows, 1-form component 2 cols (r, t, d_z)
    n1_3_full = int(seq.basis_1.n3)  # = n_r * n_t * n_dz
    C2_raw = _block(
        (lam_r, lam_t, lam_z), (lam_r, lam_t, d_lam_z)
    ).transpose(0, 2, 4, 1, 3, 5).reshape(n0_full, n1_3_full)

    # Apply extraction: C_ext = e0 @ C_raw @ e1_T_component
    # Materialize extraction operators via unit-vector probing.
    n0_ext = int(seq.n0)   # free 0-form channels (AUX0_DIRICHLET=False)
    n1_ext = int(seq.n1_dbc if DIRICHLET else seq.n1)
    n1_full = int(seq.basis_1.n)

    e0_dense = jax.vmap(
        lambda v: seq.e0 @ v, in_axes=1, out_axes=1
    )(jnp.eye(n0_full, dtype=jnp.float64))  # (n0_ext, n0_full)

    e1_lift = jax.vmap(
        lambda v: (seq.e1_dbc_T if DIRICHLET else seq.e1_T) @ v,
        in_axes=1, out_axes=1,
    )(jnp.eye(n1_ext, dtype=jnp.float64))   # (n1_full, n1_ext)

    e1_lift_c0 = e1_lift[:n1_1_full, :]                        # (n1_1, n1_ext)
    e1_lift_c1 = e1_lift[n1_1_full:n1_1_full + n1_2_full, :]  # (n1_2, n1_ext)
    e1_lift_c2 = e1_lift[n1_1_full + n1_2_full:, :]           # (n1_3, n1_ext)

    C0_ext = e0_dense @ C0_raw @ e1_lift_c0  # (n0_ext, n1_ext)
    C1_ext = e0_dense @ C1_raw @ e1_lift_c1
    C2_ext = e0_dense @ C2_raw @ e1_lift_c2

    return jnp.vstack([C0_ext, C1_ext, C2_ext])  # (3*n0_ext, n1_ext)


def _build_k1_block_fd_preconditioner(
    seq,
    ops,
    *,
    dirichlet: bool,
    pinv_rtol: float = 1e-8,
    use_vector_fd: bool = False,
    use_vector_fd_true_basis: bool = False,
    use_radial_banded: bool = False,
    vector_fd_regularization_rel: float = 1e-2,
    vector_fd_low_mode_exclude: int = 0,
    vector_fd_report_k: int = 8,
    vector_fd_origin_k: int = 4,
    return_profile: bool = False):
    """Build state for matrix-free k=1 tensor bulk preconditioner + surgery Schur."""
    pair = ops.k1_tensor_stiff_precond
    if pair is None:
        raise ValueError("k=1 tensor stiffness preconditioner is not assembled")
    payload = pair.dbc if dirichlet else pair.free
    if payload is None:
        side = "dbc" if dirichlet else "free"
        raise ValueError(f"k=1 tensor stiffness payload missing for {side}")

    surgery = payload.surgery
    factors = payload.factors

    # Matrix-free bulk inverse surrogate from tensor factors.
    if use_vector_fd and use_vector_fd_true_basis:
        raise ValueError(
            "Select at most one of use_vector_fd and use_vector_fd_true_basis."
        )

    vector_fd_state = None
    vector_fd_true_basis_state = None
    radial_banded_state = None
    if use_radial_banded:
        radial_banded_state = _build_k1_radial_banded_bulk_state(
            surgery,
            factors,
            pinv_rtol=pinv_rtol,
            regularization_rel=vector_fd_regularization_rel,
        )
    elif use_vector_fd_true_basis:
        vector_fd_true_basis_state = _build_k1_vector_fd_true_basis_bulk_state(
            surgery,
            factors,
            pinv_rtol=pinv_rtol,
            regularization_rel=vector_fd_regularization_rel,
            low_mode_exclude=vector_fd_low_mode_exclude,
            report_k=vector_fd_report_k,
            origin_k=vector_fd_origin_k,
        )
    elif use_vector_fd:
        vector_fd_state = _build_k1_vector_fd_bulk_state(
            surgery,
            factors,
            pinv_rtol=pinv_rtol,
            regularization_rel=vector_fd_regularization_rel,
        )

    def _bulk_apply(rhs_bulk):
        state = K1BlockFDPreconditionerState(
            surgery=surgery,
            factors=factors,
            schur_inv=jnp.zeros((1, 1), dtype=jnp.float64),
            use_inner_schur=bool(factors.bulk_schur),
            vector_fd_state=vector_fd_state,
            vector_fd_true_basis_state=vector_fd_true_basis_state,
            radial_banded_state=radial_banded_state,
        )
        return _apply_k1_block_fd_bulk_from_state(state, rhs_bulk)

    # Assemble surgery Schur with the chosen bulk surrogate. The surgery
    # coupling applies now route through the production payload's precomputed
    # dense coupling block (``surgery.coupling_sb``, built by default in
    # ``_build_k1_stiffness_surgery_factors``), so no script-local precompute is
    # needed here.
    n_s = int(surgery.surgery_size)
    eye_s = jnp.eye(n_s, dtype=jnp.float64)
    schur_dense = jax.vmap(
        lambda v: surgery.ass @ v - _apply_bulk_to_surgery_coupling(
            surgery, _bulk_apply(_apply_surgery_to_bulk_coupling(surgery, v))
        ),
        in_axes=1,
        out_axes=1,
    )(eye_s)
    schur_dense = _symmetrize(schur_dense)

    # PSD pseudoinverse: invert only sufficiently positive eigenvalues.
    evals, evecs = jnp.linalg.eigh(schur_dense)
    scale = jnp.max(jnp.abs(evals))
    cutoff = jnp.maximum(pinv_rtol * jnp.where(scale > 0, scale, 1.0), 1e-14)
    inv_evals = jnp.where(evals > cutoff, 1.0 / evals, 0.0)
    schur_inv = (evecs * inv_evals[jnp.newaxis, :]) @ evecs.T

    state = K1BlockFDPreconditionerState(
        surgery=surgery,
        factors=factors,
        schur_inv=schur_inv,
        use_inner_schur=bool(factors.bulk_schur),
        vector_fd_state=vector_fd_state,
        vector_fd_true_basis_state=vector_fd_true_basis_state,
        radial_banded_state=radial_banded_state,
    )

    if not return_profile:
        return state
    return state, True


def _build_k2_block_fd_preconditioner(seq, ops, *, dirichlet: bool, pinv_rtol: float = 1e-8):
    """Capped div-div (k=2) P_A, mimicking ``_build_k1_block_fd_preconditioner``.

    Same recipe as the WORKING k=1 block_fd, one degree up: the tensor-factor bulk
    inverse + a surgery Schur that is RE-PROBED densely (consistent with the bulk
    surrogate) and PSD-pseudo-inverted with a relative cutoff -- ``where(evals >
    cutoff, 1/evals, 0)`` drops both the small AND the spurious-negative axis modes.
    This is the cap that makes P_A BOUNDED on its null (the curls), so raw
    ``P_A + P_B`` no longer needs the curl-complement projection. (div-div is the
    EASIER operator vs curl-curl -- whose block_fd already drops all off-diagonal
    couplings yet works -- so this should be at least as good.) Returns a single
    matrix-free apply V2* -> V2; no dense inverse beyond the tiny axis Schur.
    """
    pair = ops.k2_tensor_stiff_precond
    if pair is None:
        raise ValueError("k=2 tensor stiffness preconditioner is not assembled")
    payload = pair.dbc if dirichlet else pair.free
    if payload is None:
        side = "dbc" if dirichlet else "free"
        raise ValueError(f"k=2 tensor stiffness payload missing for {side}")
    surgery = payload.surgery
    factors = payload.factors
    bulk_impl = (_apply_k2_bulk_preconditioner if factors.bulk_schur
                 else _apply_k2_bulk_diagonal_preconditioner)

    def _bulk(rhs_bulk):
        return bulk_impl(surgery, factors.r_bulk, factors.theta, factors.zeta, rhs_bulk)

    n_s = int(surgery.surgery_size)
    eye_s = jnp.eye(n_s, dtype=jnp.float64)
    schur_dense = jax.vmap(
        lambda v: surgery.ass @ v - _apply_bulk_to_surgery_coupling(
            surgery, _bulk(_apply_surgery_to_bulk_coupling(surgery, v))),
        in_axes=1, out_axes=1)(eye_s)
    schur_dense = _symmetrize(schur_dense)
    evals, evecs = jnp.linalg.eigh(schur_dense)
    scale = jnp.max(jnp.abs(evals))
    cutoff = jnp.maximum(pinv_rtol * jnp.where(scale > 0, scale, 1.0), 1e-14)
    inv_evals = jnp.where(evals > cutoff, 1.0 / evals, 0.0)
    schur_inv = (evecs * inv_evals[jnp.newaxis, :]) @ evecs.T

    def apply(v):  # V2* -> V2
        rhs_s = v[surgery.surgery_indices]
        rhs_b = v[surgery.bulk_indices]
        y = _bulk(rhs_b)
        z = schur_inv @ (rhs_s - _apply_bulk_to_surgery_coupling(surgery, y))
        x_b = y - _bulk(_apply_surgery_to_bulk_coupling(surgery, z))
        x = jnp.zeros_like(v)
        x = x.at[surgery.surgery_indices].set(z)
        x = x.at[surgery.bulk_indices].set(x_b)
        return x

    return apply


def _apply_k1_block_fd_bulk_from_state(state: K1BlockFDPreconditionerState, rhs_bulk):
    surgery = state.surgery
    factors = state.factors
    if state.radial_banded_state is not None:
        return _apply_k1_radial_banded_bulk_from_state(
            state.radial_banded_state,
            rhs_bulk,
        )
    if state.vector_fd_true_basis_state is not None:
        return _apply_k1_vector_fd_true_basis_bulk_from_state(
            state.vector_fd_true_basis_state,
            rhs_bulk,
        )
    if state.vector_fd_state is not None:
        return _apply_k1_vector_fd_bulk_from_state(
            state.vector_fd_state,
            rhs_bulk,
        )
    bulk_apply_impl = (
        _apply_k1_bulk_preconditioner
        if factors.bulk_schur
        else _apply_k1_bulk_diagonal_preconditioner
    )
    return bulk_apply_impl(
        surgery,
        factors.arr,
        factors.theta,
        factors.zeta,
        rhs_bulk,
    )


def _apply_k1_block_fd_preconditioner_from_state(state: K1BlockFDPreconditionerState, v):
    surgery = state.surgery
    rhs_s = v[surgery.surgery_indices]
    rhs_b = v[surgery.bulk_indices]
    y = _apply_k1_block_fd_bulk_from_state(state, rhs_b)
    z = state.schur_inv @ (rhs_s - _apply_bulk_to_surgery_coupling(surgery, y))
    x_b = y - _apply_k1_block_fd_bulk_from_state(
        state,
        _apply_surgery_to_bulk_coupling(surgery, z),
    )
    x = jnp.zeros_like(v)
    x = x.at[surgery.surgery_indices].set(z)
    x = x.at[surgery.bulk_indices].set(x_b)
    return x


def _profile_k1_block_fd_preconditioner_from_state(state: K1BlockFDPreconditionerState, v):
    surgery = state.surgery
    rhs_s = v[surgery.surgery_indices]
    rhs_b = v[surgery.bulk_indices]

    t0 = time.perf_counter()
    y = _apply_k1_block_fd_bulk_from_state(state, rhs_b)
    jax.block_until_ready(y)
    t1 = time.perf_counter()

    coup_sb = _apply_bulk_to_surgery_coupling(surgery, y)
    jax.block_until_ready(coup_sb)
    z = state.schur_inv @ (rhs_s - coup_sb)
    jax.block_until_ready(z)
    t2 = time.perf_counter()

    coup_bs = _apply_surgery_to_bulk_coupling(surgery, z)
    jax.block_until_ready(coup_bs)
    corr = _apply_k1_block_fd_bulk_from_state(state, coup_bs)
    jax.block_until_ready(corr)
    t3 = time.perf_counter()

    return {
        "bulk1_ms": (t1 - t0) * 1e3,
        "schur_ms": (t2 - t1) * 1e3,
        "bulk2_ms": (t3 - t2) * 1e3,
        "total_ms": (t3 - t0) * 1e3,
    }


def _accumulate_axis_leakage(acc, out_flat, shape, m):
    """Bucket off-diagonal modal energy of a probe response by tensor axis.

    ``out_flat`` is the modal response in the SAME component as the probe,
    flattened over that component's separable ``(nr, nt, nz)`` mode grid.
    ``m`` is the flat probe mode index (shared-prefix), unravelled C-order in
    ``shape``. Leaked (off-diagonal) energy is split into radial-only /
    poloidal-only / toroidal-only (exactly one tensor index differs from the
    probe) and mixed (two or more indices differ). This measures, in the
    separable fast-diagonalization basis, along which physical direction the
    true operator fails to stay diagonal.
    """
    nr, nt, nz = int(shape[0]), int(shape[1]), int(shape[2])
    e = (jnp.asarray(out_flat).reshape(nr, nt, nz)) ** 2
    i_r = m // (nt * nz)
    rem = m % (nt * nz)
    i_t = rem // nz
    i_z = rem % nz
    diag = float(e[i_r, i_t, i_z])
    radial = float(jnp.sum(e[:, i_t, i_z])) - diag
    poloidal = float(jnp.sum(e[i_r, :, i_z])) - diag
    toroidal = float(jnp.sum(e[i_r, i_t, :])) - diag
    total_off = float(jnp.sum(e)) - diag
    mixed = total_off - radial - poloidal - toroidal
    acc["radial"] += radial
    acc["poloidal"] += poloidal
    acc["toroidal"] += toroidal
    acc["mixed"] += mixed
    acc["total_off"] += total_off


def _build_k1_vector_fd_bulk_state(
        surgery,
        factors,
        *,
    pinv_rtol: float = 1e-8,
    regularization_rel: float = 1e-2):
    """Matrix-free vector-valued FD bulk inverse (diagonal-metric model).

    Keeps the existing per-component FD bases but replaces the scalar modal
    inverse by coupled 3x3 modewise solves on the shared modal prefix.
    Coupling blocks are assembled matrix-free from the existing bulk forward
    model (no dense K_BB assembly).
    """

    if factors.bulk_schur:
        raise ValueError(
            "--pa-block-vector-fd is not compatible with --pa-block-inner-schur; "
            "both add coupling, but on different levels. Disable inner Schur "
            "for vector-FD bulk experiments."
        )

    arr_f = factors.arr
    theta_f = factors.theta
    zeta_f = factors.zeta

    fd_triplets = (
        ("r", arr_f),
        ("theta", theta_f),
        ("zeta", zeta_f),
    )
    for name, fac in fd_triplets:
        if fac.fd_V_r is None or fac.fd_V_t is None or fac.fd_V_z is None or fac.fd_inv_denom is None:
            raise ValueError(
                f"Vector-FD requires fd_V_* and fd_inv_denom on {name} factors; "
                "reassemble tensor stiffness factors with FD data available."
            )

    n_r = int(surgery.rt_r_size)
    n_t = int(surgery.bulk_rt_size - surgery.rt_r_size)
    n_z = int(surgery.bulk_zeta_size)
    n_common = int(min(n_r, n_t, n_z))

    r_slice = slice(0, n_r)
    t_slice = slice(n_r, n_r + n_t)
    z_slice = slice(n_r + n_t, n_r + n_t + n_z)

    def _to_modal(fac, x):
        nr, nt, nz = fac.shape
        modes = jnp.asarray(x).reshape(nr, nt, nz)
        modes = jnp.einsum("ji,jkl->ikl", fac.fd_V_r, modes)
        modes = jnp.einsum("ji,kjl->kil", fac.fd_V_t, modes)
        modes = jnp.einsum("ji,klj->kli", fac.fd_V_z, modes)
        return modes.reshape(-1)

    def _from_modal(fac, xm):
        nr, nt, nz = fac.shape
        modes = jnp.asarray(xm).reshape(nr, nt, nz)
        modes = jnp.einsum("ij,jkl->ikl", fac.fd_V_r, modes)
        modes = jnp.einsum("ij,kjl->kil", fac.fd_V_t, modes)
        modes = jnp.einsum("ij,klj->kli", fac.fd_V_z, modes)
        return modes.reshape(-1)

    inv_denom_r = arr_f.fd_inv_denom.reshape(-1)
    inv_denom_t = theta_f.fd_inv_denom.reshape(-1)
    inv_denom_z = zeta_f.fd_inv_denom.reshape(-1)

    # Assemble coupled 3x3 modal symbols for the shared modal prefix.
    blocks = jnp.zeros((n_common, 3, 3), dtype=jnp.float64)
    leakage_values: list[float] = []
    axis_leak = {"radial": 0.0, "poloidal": 0.0, "toroidal": 0.0, "mixed": 0.0, "total_off": 0.0}
    for m in range(n_common):
        for col in range(3):
            rhs_modal_r = jnp.zeros((n_r,), dtype=jnp.float64)
            rhs_modal_t = jnp.zeros((n_t,), dtype=jnp.float64)
            rhs_modal_z = jnp.zeros((n_z,), dtype=jnp.float64)
            if col == 0:
                rhs_modal_r = rhs_modal_r.at[m].set(1.0)
            elif col == 1:
                rhs_modal_t = rhs_modal_t.at[m].set(1.0)
            else:
                rhs_modal_z = rhs_modal_z.at[m].set(1.0)

            rhs_bulk = jnp.concatenate([
                _from_modal(arr_f, rhs_modal_r),
                _from_modal(theta_f, rhs_modal_t),
                _from_modal(zeta_f, rhs_modal_z),
            ])

            out_bulk = _apply_k1_bulk_forward_model(
                surgery,
                arr_f,
                theta_f,
                zeta_f,
                rhs_bulk,
            )

            out_modal_r = _to_modal(arr_f, out_bulk[r_slice])
            out_modal_t = _to_modal(theta_f, out_bulk[t_slice])
            out_modal_z = _to_modal(zeta_f, out_bulk[z_slice])

            total_modal_energy = (
                float(jnp.dot(out_modal_r, out_modal_r))
                + float(jnp.dot(out_modal_t, out_modal_t))
                + float(jnp.dot(out_modal_z, out_modal_z))
            )
            captured_modal_energy = (
                float(out_modal_r[m] ** 2)
                + float(out_modal_t[m] ** 2)
                + float(out_modal_z[m] ** 2)
            )
            if total_modal_energy > 0.0:
                leakage_values.append(
                    max(0.0, 1.0 - captured_modal_energy / total_modal_energy)
                )

            # Per-axis decomposition of the TRUE extracted operator's response
            # (same component as the probe) in the separable FD basis. This is
            # the honest test of "does the angular direction stay diagonal?":
            # it bypasses the diagonal-metric forward model used to build the
            # coupling blocks and probes the actual bulk stiffness.
            true_out_bulk = _apply_extracted_submatrix(
                surgery.apply_data,
                surgery.bulk_indices,
                surgery.bulk_indices,
                rhs_bulk,
            )
            if col == 0:
                same_out = _to_modal(arr_f, true_out_bulk[r_slice])
                same_shape = arr_f.shape
            elif col == 1:
                same_out = _to_modal(theta_f, true_out_bulk[t_slice])
                same_shape = theta_f.shape
            else:
                same_out = _to_modal(zeta_f, true_out_bulk[z_slice])
                same_shape = zeta_f.shape
            _accumulate_axis_leakage(axis_leak, same_out, same_shape, m)

            blocks = blocks.at[m, 0, col].set(out_modal_r[m])
            blocks = blocks.at[m, 1, col].set(out_modal_t[m])
            blocks = blocks.at[m, 2, col].set(out_modal_z[m])

    if leakage_values:
        leakage_arr = jnp.asarray(leakage_values, dtype=jnp.float64)
        print(
            "[diag] vector-FD modal leakage (fraction of forward-model modal "
            "energy outside the retained same-mode 3x3 block): "
            f"mean={float(jnp.mean(leakage_arr)):.2e} "
            f"p95={float(jnp.percentile(leakage_arr, 95.0)):.2e} "
            f"max={float(jnp.max(leakage_arr)):.2e}"
        )

    if axis_leak["total_off"] > 0.0:
        total_off = axis_leak["total_off"]
        print(
            "[diag] vector-FD true-operator off-diagonal leakage by axis "
            "(separable FD basis; fraction of off-diagonal modal energy): "
            f"radial={axis_leak['radial'] / total_off:.2e} "
            f"poloidal={axis_leak['poloidal'] / total_off:.2e} "
            f"toroidal={axis_leak['toroidal'] / total_off:.2e} "
            f"mixed={axis_leak['mixed'] / total_off:.2e}"
        )

    # Build SPD-clipped inverse blocks once at setup. Near-singular modes are
    # regularized in-coupling (no scalar fallback) so cross-component curl
    # structure is preserved while keeping the local inverse positive.
    block_eigs = jax.vmap(lambda a: jnp.linalg.eigvalsh(_symmetrize(a)))(blocks)
    block_scale = jnp.max(jnp.abs(block_eigs), axis=1)
    block_base_cutoff = jnp.maximum(pinv_rtol * jnp.where(block_scale > 0, block_scale, 1.0), 1e-14)
    block_min = jnp.min(block_eigs, axis=1)
    block_regularized = block_min <= block_base_cutoff
    strong_floor = jnp.asarray(regularization_rel, dtype=jnp.float64) * jnp.maximum(block_scale, 1.0)
    block_cutoff = jnp.where(
        block_regularized,
        jnp.maximum(block_base_cutoff, strong_floor),
        block_base_cutoff,
    )

    def _inv_spd_clipped(a, cutoff):
        evals, evecs = jnp.linalg.eigh(_symmetrize(a))
        clipped = jnp.maximum(evals, cutoff)
        inv_vals = 1.0 / clipped
        return (evecs * inv_vals[jnp.newaxis, :]) @ evecs.T

    inv_blocks = jax.vmap(_inv_spd_clipped)(blocks, block_cutoff)

    n_regularized = int(jnp.sum(block_regularized))
    print(
        "[diag] vector-FD mode regularization: "
        f"spd_clipped={n_common}/{n_common}, regularized={n_regularized}, "
        f"reg_rel={regularization_rel:.2e}"
    )

    return K1VectorFDBulkState(
        surgery=surgery,
        arr_f=arr_f,
        theta_f=theta_f,
        zeta_f=zeta_f,
        inv_denom_r=inv_denom_r,
        inv_denom_t=inv_denom_t,
        inv_denom_z=inv_denom_z,
        inv_blocks=inv_blocks,
        n_r=n_r,
        n_t=n_t,
        n_z=n_z,
        n_common=n_common,
    )


def _apply_k1_vector_fd_bulk_from_state(state: K1VectorFDBulkState, rhs_bulk):
    arr_f = state.arr_f
    theta_f = state.theta_f
    zeta_f = state.zeta_f

    # Per-component bulk sizes span the FULL (nr, nt, nz) tensor grid, not just
    # the leading axis; use the sizes stored at build time (surgery-derived).
    n_r = int(state.n_r)
    n_t = int(state.n_t)
    n_z = int(state.n_z)
    n_common = int(state.n_common)

    r_slice = slice(0, n_r)
    t_slice = slice(n_r, n_r + n_t)
    z_slice = slice(n_r + n_t, n_r + n_t + n_z)

    def _to_modal(fac, x):
        nr, nt, nz = fac.shape
        modes = jnp.asarray(x).reshape(nr, nt, nz)
        modes = jnp.einsum("ji,jkl->ikl", fac.fd_V_r, modes)
        modes = jnp.einsum("ji,kjl->kil", fac.fd_V_t, modes)
        modes = jnp.einsum("ji,klj->kli", fac.fd_V_z, modes)
        return modes.reshape(-1)

    def _from_modal(fac, xm):
        nr, nt, nz = fac.shape
        modes = jnp.asarray(xm).reshape(nr, nt, nz)
        modes = jnp.einsum("ij,jkl->ikl", fac.fd_V_r, modes)
        modes = jnp.einsum("ij,kjl->kil", fac.fd_V_t, modes)
        modes = jnp.einsum("ij,klj->kli", fac.fd_V_z, modes)
        return modes.reshape(-1)

    rhs_r = rhs_bulk[r_slice]
    rhs_t = rhs_bulk[t_slice]
    rhs_z = rhs_bulk[z_slice]

    rhs_modal_r = _to_modal(arr_f, rhs_r)
    rhs_modal_t = _to_modal(theta_f, rhs_t)
    rhs_modal_z = _to_modal(zeta_f, rhs_z)

    x_modal_r = state.inv_denom_r * rhs_modal_r
    x_modal_t = state.inv_denom_t * rhs_modal_t
    x_modal_z = state.inv_denom_z * rhs_modal_z

    if n_common > 0:
        y3 = jnp.stack(
            [rhs_modal_r[:n_common], rhs_modal_t[:n_common], rhs_modal_z[:n_common]],
            axis=1,
        )
        x3 = jnp.einsum("mij,mj->mi", state.inv_blocks, y3)
        x_modal_r = x_modal_r.at[:n_common].set(x3[:, 0])
        x_modal_t = x_modal_t.at[:n_common].set(x3[:, 1])
        x_modal_z = x_modal_z.at[:n_common].set(x3[:, 2])

    return jnp.concatenate([
        _from_modal(arr_f, x_modal_r),
        _from_modal(theta_f, x_modal_t),
        _from_modal(zeta_f, x_modal_z),
    ])


def _build_k1_radial_banded_bulk_state(
        surgery,
        factors,
        *,
    pinv_rtol: float = 1e-8,
    regularization_rel: float = 1e-2):
    """Matrix-free modal bulk inverse with full radial+component coupling.

    The same-mode 3x3 vector-FD symbol only retains the operator's
    ``mode m -> mode m`` action in the per-component fast-diag basis. The
    per-axis leakage diagnostic shows ~97% of the missed energy is *radial*
    (``curl`` carries ``d_r``, which is not diagonal in the radial fast-diag
    basis), with poloidal/toroidal leakage at the few-percent / negligible
    level. This builder therefore keeps the *entire* radial-mode and
    cross-component coupling at each angular mode ``(i_t, i_z)`` and discards
    only the measured small angular leakage.

    Because ``TYPES = (clamped, periodic, periodic)`` keeps the poloidal and
    toroidal *mode counts* identical across the three vector components (only
    the radial count differs), the angular index ``(i_t, i_z)`` is a common
    label for all components. For each such label we assemble the dense block
    ``M^{(i_t,i_z)} = (V^T K V)`` restricted to rows/cols sharing that angular
    index, over the joint ``(component, radial mode)`` space, by probing the
    true extracted bulk operator. Inverting these per-angular blocks (with the
    same SPD-clip regularization as the 3x3 path) yields an exact-in-
    radial+component, block-diagonal-in-angular bulk inverse.
    """

    if factors.bulk_schur:
        raise ValueError(
            "--pa-block-radial-banded is not compatible with --pa-block-inner-schur."
        )

    arr_f = factors.arr
    theta_f = factors.theta
    zeta_f = factors.zeta

    for name, fac in (("r", arr_f), ("theta", theta_f), ("zeta", zeta_f)):
        if fac.fd_V_r is None or fac.fd_V_t is None or fac.fd_V_z is None:
            raise ValueError(
                f"Radial-banded bulk solve requires fd_V_* on {name} factors; "
                "reassemble tensor stiffness factors with FD data available."
            )

    r0, nt, nz = (int(s) for s in arr_f.shape)
    r1, nt1, nz1 = (int(s) for s in theta_f.shape)
    r2, nt2, nz2 = (int(s) for s in zeta_f.shape)
    if not (nt == nt1 == nt2 and nz == nz1 == nz2):
        raise ValueError(
            "Radial-banded bulk solve requires matching poloidal/toroidal mode "
            f"counts across components, got nt=({nt},{nt1},{nt2}) "
            f"nz=({nz},{nz1},{nz2}). This holds for clamped/periodic spline "
            "types; check the de Rham component spaces."
        )

    rsum = r0 + r1 + r2
    n_ang = nt * nz
    bulk_r = r0 * nt * nz
    bulk_t = r1 * nt * nz
    bulk_z = r2 * nt * nz
    r_slice = slice(0, bulk_r)
    t_slice = slice(bulk_r, bulk_r + bulk_t)
    z_slice = slice(bulk_r + bulk_t, bulk_r + bulk_t + bulk_z)

    comps = ((arr_f, r0, 0), (theta_f, r1, r0), (zeta_f, r2, r0 + r1))

    def _to_modal(fac, x):
        nrr, ntt, nzz = fac.shape
        modes = jnp.asarray(x).reshape(nrr, ntt, nzz)
        modes = jnp.einsum("ji,jkl->ikl", fac.fd_V_r, modes)
        modes = jnp.einsum("ji,kjl->kil", fac.fd_V_t, modes)
        modes = jnp.einsum("ji,klj->kli", fac.fd_V_z, modes)
        return modes

    def _from_modal(fac, modes3d):
        modes = jnp.einsum("ij,jkl->ikl", fac.fd_V_r, modes3d)
        modes = jnp.einsum("ij,kjl->kil", fac.fd_V_t, modes)
        modes = jnp.einsum("ij,klj->kli", fac.fd_V_z, modes)
        return modes.reshape(-1)

    def _true_apply(rhs_bulk):
        return _apply_extracted_submatrix(
            surgery.apply_data,
            surgery.bulk_indices,
            surgery.bulk_indices,
            rhs_bulk,
        )

    # Assemble the dense (component, radial) block per angular mode by probing
    # the true extracted bulk operator with one modal unit at a time. Total
    # probes equal the bulk DOF count (setup-only Python loop, like the
    # true-basis path). Each probe yields one column of M^{(i_t,i_z)}.
    blocks = jnp.zeros((n_ang, rsum, rsum), dtype=jnp.float64)
    same_g_energy = jnp.asarray(0.0, dtype=jnp.float64)
    total_energy = jnp.asarray(0.0, dtype=jnp.float64)
    for fac_c, rc, off_c in comps:
        for ir in range(rc):
            for it in range(nt):
                for iz in range(nz):
                    modal_c = jnp.zeros((rc, nt, nz), dtype=jnp.float64)
                    modal_c = modal_c.at[ir, it, iz].set(1.0)
                    bulk_c = _from_modal(fac_c, modal_c)
                    parts = [
                        jnp.zeros((bulk_r,), dtype=jnp.float64),
                        jnp.zeros((bulk_t,), dtype=jnp.float64),
                        jnp.zeros((bulk_z,), dtype=jnp.float64),
                    ]
                    if off_c == 0:
                        parts[0] = bulk_c
                    elif off_c == r0:
                        parts[1] = bulk_c
                    else:
                        parts[2] = bulk_c
                    rhs_bulk = jnp.concatenate(parts)

                    out = _true_apply(rhs_bulk)
                    out_r = _to_modal(arr_f, out[r_slice])
                    out_t = _to_modal(theta_f, out[t_slice])
                    out_z = _to_modal(zeta_f, out[z_slice])

                    g = it * nz + iz
                    col = off_c + ir
                    blocks = blocks.at[g, 0:r0, col].set(out_r[:, it, iz])
                    blocks = blocks.at[g, r0:r0 + r1, col].set(out_t[:, it, iz])
                    blocks = blocks.at[g, r0 + r1:, col].set(out_z[:, it, iz])

                    total_energy = total_energy + (
                        jnp.sum(out_r ** 2)
                        + jnp.sum(out_t ** 2)
                        + jnp.sum(out_z ** 2)
                    )
                    same_g_energy = same_g_energy + (
                        jnp.sum(out_r[:, it, iz] ** 2)
                        + jnp.sum(out_t[:, it, iz] ** 2)
                        + jnp.sum(out_z[:, it, iz] ** 2)
                    )

    total_energy_f = float(total_energy)
    if total_energy_f > 0.0:
        angular_leak = max(0.0, 1.0 - float(same_g_energy) / total_energy_f)
        print(
            "[diag] radial-banded angular leakage (fraction of true modal "
            "energy outside the retained same-(theta,zeta) block; radial and "
            f"component coupling fully retained): {angular_leak:.2e}"
        )

    # SPD-clipped inverse of each per-angular block, mirroring the 3x3 path.
    block_eigs = jax.vmap(lambda a: jnp.linalg.eigvalsh(_symmetrize(a)))(blocks)
    block_scale = jnp.max(jnp.abs(block_eigs), axis=1)
    block_base_cutoff = jnp.maximum(
        pinv_rtol * jnp.where(block_scale > 0, block_scale, 1.0), 1e-14
    )
    block_min = jnp.min(block_eigs, axis=1)
    block_regularized = block_min <= block_base_cutoff
    strong_floor = jnp.asarray(regularization_rel, dtype=jnp.float64) * jnp.maximum(
        block_scale, 1.0
    )
    block_cutoff = jnp.where(
        block_regularized,
        jnp.maximum(block_base_cutoff, strong_floor),
        block_base_cutoff,
    )

    def _inv_spd_clipped(a, cutoff):
        evals, evecs = jnp.linalg.eigh(_symmetrize(a))
        clipped = jnp.maximum(evals, cutoff)
        inv_vals = 1.0 / clipped
        return (evecs * inv_vals[jnp.newaxis, :]) @ evecs.T

    inv_blocks = jax.vmap(_inv_spd_clipped)(blocks, block_cutoff)

    n_regularized = int(jnp.sum(block_regularized))
    print(
        "[diag] radial-banded mode regularization: "
        f"spd_clipped={n_ang}/{n_ang} angular blocks (size {rsum}x{rsum}), "
        f"regularized={n_regularized}, reg_rel={regularization_rel:.2e}"
    )

    return K1RadialBandedBulkState(
        surgery=surgery,
        arr_f=arr_f,
        theta_f=theta_f,
        zeta_f=zeta_f,
        nt=nt,
        nz=nz,
        r0=r0,
        r1=r1,
        r2=r2,
        bulk_r=bulk_r,
        bulk_t=bulk_t,
        bulk_z=bulk_z,
        inv_blocks=inv_blocks,
    )


def _apply_k1_radial_banded_bulk_from_state(state: K1RadialBandedBulkState, rhs_bulk):
    arr_f = state.arr_f
    theta_f = state.theta_f
    zeta_f = state.zeta_f
    # Pull dimensions from the static factor shapes (eqx static fields), not the
    # stored integer leaves: the state is passed as a jit argument, so its int
    # leaves become tracers and cannot drive reshape/slice sizes.
    r0, nt, nz = (int(s) for s in arr_f.shape)
    r1 = int(theta_f.shape[0])
    r2 = int(zeta_f.shape[0])
    n_ang = nt * nz
    bulk_r = r0 * nt * nz
    bulk_t = r1 * nt * nz
    bulk_z = r2 * nt * nz

    r_slice = slice(0, bulk_r)
    t_slice = slice(bulk_r, bulk_r + bulk_t)
    z_slice = slice(bulk_r + bulk_t, bulk_r + bulk_t + bulk_z)

    def _to_modal(fac, x):
        nrr, ntt, nzz = fac.shape
        modes = jnp.asarray(x).reshape(nrr, ntt, nzz)
        modes = jnp.einsum("ji,jkl->ikl", fac.fd_V_r, modes)
        modes = jnp.einsum("ji,kjl->kil", fac.fd_V_t, modes)
        modes = jnp.einsum("ji,klj->kli", fac.fd_V_z, modes)
        return modes

    def _from_modal(fac, modes3d):
        modes = jnp.einsum("ij,jkl->ikl", fac.fd_V_r, modes3d)
        modes = jnp.einsum("ij,kjl->kil", fac.fd_V_t, modes)
        modes = jnp.einsum("ij,klj->kli", fac.fd_V_z, modes)
        return modes.reshape(-1)

    # Modal coefficients per component, shape (Rc, nt, nz).
    rhs_modal_r = _to_modal(arr_f, rhs_bulk[r_slice])
    rhs_modal_t = _to_modal(theta_f, rhs_bulk[t_slice])
    rhs_modal_z = _to_modal(zeta_f, rhs_bulk[z_slice])

    # Stack the (component, radial) coefficients per angular mode -> (n_ang, Rsum).
    vr = jnp.transpose(rhs_modal_r, (1, 2, 0)).reshape(n_ang, r0)
    vt = jnp.transpose(rhs_modal_t, (1, 2, 0)).reshape(n_ang, r1)
    vz = jnp.transpose(rhs_modal_z, (1, 2, 0)).reshape(n_ang, r2)
    v = jnp.concatenate([vr, vt, vz], axis=1)

    x = jnp.einsum("gij,gj->gi", state.inv_blocks, v)

    xr = jnp.transpose(x[:, 0:r0].reshape(nt, nz, r0), (2, 0, 1))
    xt = jnp.transpose(x[:, r0:r0 + r1].reshape(nt, nz, r1), (2, 0, 1))
    xz = jnp.transpose(x[:, r0 + r1:].reshape(nt, nz, r2), (2, 0, 1))

    return jnp.concatenate([
        _from_modal(arr_f, xr),
        _from_modal(theta_f, xt),
        _from_modal(zeta_f, xz),
    ])


def _build_k1_vector_fd_true_basis_bulk_state(
        surgery,
        factors,
        *,
    pinv_rtol: float = 1e-8,
    regularization_rel: float = 1e-2,
    low_mode_exclude: int = 0,
    report_k: int = 8,
    origin_k: int = 4):
    """Vector-valued FD bulk inverse with true (matrix-free probed) modal bases.

    Uses exact extracted bulk diagonal blocks to define per-component modal
    bases and diagonal tails, then assembles same-mode 3x3 coupling symbols by
    probing the true extracted bulk operator. This avoids relying on the
    approximate FD factor bases for coupling assembly.
    """

    if factors.bulk_schur:
        raise ValueError(
            "--pa-block-vector-fd-true-basis is not compatible with --pa-block-inner-schur."
        )

    n_r = int(surgery.rt_r_size)
    n_t = int(surgery.bulk_rt_size - surgery.rt_r_size)
    n_z = int(surgery.bulk_zeta_size)
    n_common = int(min(n_r, n_t, n_z))

    r_slice = slice(0, n_r)
    t_slice = slice(n_r, n_r + n_t)
    z_slice = slice(n_r + n_t, n_r + n_t + n_z)

    def _sym_diag_block(indices):
        n = int(indices.shape[0])
        eye = jnp.eye(n, dtype=jnp.float64)
        block = jax.vmap(
            lambda v: _apply_extracted_submatrix(surgery.apply_data, indices, indices, v),
            in_axes=1,
            out_axes=1,
        )(eye)
        return _symmetrize(block)

    k_rr = _sym_diag_block(surgery.r_indices)
    k_tt = _sym_diag_block(surgery.theta_bulk_indices)
    k_zz = _sym_diag_block(surgery.zeta_bulk_indices)

    eval_r, vec_r = jnp.linalg.eigh(k_rr)
    eval_t, vec_t = jnp.linalg.eigh(k_tt)
    eval_z, vec_z = jnp.linalg.eigh(k_zz)

    def _diag_pinv_from_eval(evals):
        scale = jnp.max(jnp.abs(evals))
        cutoff = jnp.maximum(pinv_rtol * jnp.where(scale > 0, scale, 1.0), 1e-14)
        return jnp.where(evals > cutoff, 1.0 / evals, 0.0)

    inv_eval_r = _diag_pinv_from_eval(eval_r)
    inv_eval_t = _diag_pinv_from_eval(eval_t)
    inv_eval_z = _diag_pinv_from_eval(eval_z)

    def _to_modal(vecs, x):
        return vecs.T @ x

    def _from_modal(vecs, xm):
        return vecs @ xm

    bulk_indices = surgery.bulk_indices

    def _apply_true_bulk(rhs_bulk):
        return _apply_extracted_submatrix(
            surgery.apply_data,
            bulk_indices,
            bulk_indices,
            rhs_bulk,
        )

    blocks = jnp.zeros((n_common, 3, 3), dtype=jnp.float64)
    leakage_values: list[float] = []
    for m in range(n_common):
        for col in range(3):
            rhs_modal_r = jnp.zeros((n_r,), dtype=jnp.float64)
            rhs_modal_t = jnp.zeros((n_t,), dtype=jnp.float64)
            rhs_modal_z = jnp.zeros((n_z,), dtype=jnp.float64)
            if col == 0:
                rhs_modal_r = rhs_modal_r.at[m].set(1.0)
            elif col == 1:
                rhs_modal_t = rhs_modal_t.at[m].set(1.0)
            else:
                rhs_modal_z = rhs_modal_z.at[m].set(1.0)

            rhs_bulk = jnp.concatenate([
                _from_modal(vec_r, rhs_modal_r),
                _from_modal(vec_t, rhs_modal_t),
                _from_modal(vec_z, rhs_modal_z),
            ])

            out_bulk = _apply_true_bulk(rhs_bulk)
            out_modal_r = _to_modal(vec_r, out_bulk[r_slice])
            out_modal_t = _to_modal(vec_t, out_bulk[t_slice])
            out_modal_z = _to_modal(vec_z, out_bulk[z_slice])

            total_modal_energy = (
                float(jnp.dot(out_modal_r, out_modal_r))
                + float(jnp.dot(out_modal_t, out_modal_t))
                + float(jnp.dot(out_modal_z, out_modal_z))
            )
            captured_modal_energy = (
                float(out_modal_r[m] ** 2)
                + float(out_modal_t[m] ** 2)
                + float(out_modal_z[m] ** 2)
            )
            if total_modal_energy > 0.0:
                leakage_values.append(
                    max(0.0, 1.0 - captured_modal_energy / total_modal_energy)
                )

            blocks = blocks.at[m, 0, col].set(out_modal_r[m])
            blocks = blocks.at[m, 1, col].set(out_modal_t[m])
            blocks = blocks.at[m, 2, col].set(out_modal_z[m])

    if leakage_values:
        leakage_arr = jnp.asarray(leakage_values, dtype=jnp.float64)
        print(
            "[diag] vector-FD true-basis modal leakage (fraction of true bulk "
            "modal energy outside retained same-mode 3x3 block): "
            f"mean={float(jnp.mean(leakage_arr)):.2e} "
            f"p95={float(jnp.percentile(leakage_arr, 95.0)):.2e} "
            f"max={float(jnp.max(leakage_arr)):.2e}"
        )

    # Build SPD-clipped inverse blocks once at setup. This keeps coupled
    # cross-component structure even for near-singular low modes.
    block_eigs = jax.vmap(lambda a: jnp.linalg.eigvalsh(_symmetrize(a)))(blocks)
    block_scale = jnp.max(jnp.abs(block_eigs), axis=1)
    block_base_cutoff = jnp.maximum(pinv_rtol * jnp.where(block_scale > 0, block_scale, 1.0), 1e-14)
    block_min = jnp.min(block_eigs, axis=1)
    block_regularized = block_min <= block_base_cutoff
    strong_floor = jnp.asarray(regularization_rel, dtype=jnp.float64) * jnp.maximum(block_scale, 1.0)
    block_cutoff = jnp.where(
        block_regularized,
        jnp.maximum(block_base_cutoff, strong_floor),
        block_base_cutoff,
    )
    if low_mode_exclude > 0:
        n_ex = int(min(low_mode_exclude, n_common))
        force_clip = jnp.arange(n_common) < n_ex
        low_mode_floor = 1e-2 * jnp.maximum(block_scale, 1.0)
        block_cutoff = jnp.where(force_clip, jnp.maximum(block_cutoff, low_mode_floor), block_cutoff)

    if report_k > 0 and n_common > 0:
        k = int(min(report_k, n_common))
        sort_idx = jnp.argsort(block_min)
        lo_idx = sort_idx[:k]
        hi_idx = sort_idx[-k:][::-1]
        lo_idx_str = ",".join(str(int(i)) for i in lo_idx.tolist())
        lo_val_str = ",".join(f"{float(v):.2e}" for v in block_min[lo_idx].tolist())
        hi_idx_str = ",".join(str(int(i)) for i in hi_idx.tolist())
        hi_val_str = ",".join(f"{float(v):.2e}" for v in block_min[hi_idx].tolist())
        print(f"[diag] vector-FD true-basis min-eig(3x3) lowest {k} modes idx: {lo_idx_str}")
        print(f"[diag] vector-FD true-basis min-eig(3x3) lowest {k} values: {lo_val_str}")
        print(f"[diag] vector-FD true-basis min-eig(3x3) highest {k} modes idx: {hi_idx_str}")
        print(f"[diag] vector-FD true-basis min-eig(3x3) highest {k} values: {hi_val_str}")

    def _inv_spd_clipped(a, cutoff):
        evals, evecs = jnp.linalg.eigh(_symmetrize(a))
        clipped = jnp.maximum(evals, cutoff)
        inv_vals = 1.0 / clipped
        return (evecs * inv_vals[jnp.newaxis, :]) @ evecs.T

    inv_blocks = jax.vmap(_inv_spd_clipped)(blocks, block_cutoff)

    n_regularized = int(jnp.sum(block_regularized))
    print(
        "[diag] vector-FD true-basis mode regularization: "
        f"spd_clipped={n_common}/{n_common}, regularized={n_regularized}, "
        f"reg_rel={regularization_rel:.2e}"
    )
    if n_regularized > 0:
        reg_idx = jnp.where(block_regularized)[0]
        reg_min = block_min[reg_idx]
        idx_str = ",".join(str(int(i)) for i in reg_idx.tolist())
        min_str = ",".join(f"{float(v):.2e}" for v in reg_min.tolist())
        print(
            "[diag] vector-FD true-basis regularized modes (shared-modal index): "
            + idx_str
        )
        print(
            "[diag] vector-FD true-basis regularized mode min eig(3x3): "
            + min_str
        )
        if origin_k > 0:
            n_show = int(min(origin_k, reg_idx.shape[0]))
            print(
                "[diag] vector-FD true-basis regularized block origin diagnostics "
                f"(showing {n_show} of {int(reg_idx.shape[0])})"
            )
            for m in reg_idx[:n_show].tolist():
                block = _symmetrize(blocks[m])
                evals, evecs = jnp.linalg.eigh(block)
                min_j = int(jnp.argmin(evals))
                min_vec = evecs[:, min_j]
                diag_norm = float(jnp.linalg.norm(jnp.diag(block)))
                offdiag_norm = float(jnp.linalg.norm(block - jnp.diag(jnp.diag(block))))
                max_eval = float(jnp.max(jnp.abs(evals)))
                min_abs_eval = float(jnp.min(jnp.abs(evals)))
                cond_m = max_eval / max(min_abs_eval, 1e-30)
                eval_str = ",".join(f"{float(v):.2e}" for v in evals.tolist())
                vec_str = ",".join(f"{float(c):+.2e}" for c in min_vec.tolist())
                print(
                    f"[diag]   mode {m}: evals=[{eval_str}] cond_abs={cond_m:.2e} "
                    f"diag_norm={diag_norm:.2e} offdiag_norm={offdiag_norm:.2e} "
                    f"min-evec=[{vec_str}]"
                )

    return K1VectorFDTrueBasisBulkState(
        surgery=surgery,
        vec_r=vec_r,
        vec_t=vec_t,
        vec_z=vec_z,
        inv_eval_r=inv_eval_r,
        inv_eval_t=inv_eval_t,
        inv_eval_z=inv_eval_z,
        inv_blocks=inv_blocks,
        n_r=n_r,
        n_t=n_t,
        n_z=n_z,
        n_common=n_common,
    )


def _apply_k1_vector_fd_true_basis_bulk_from_state(
        state: K1VectorFDTrueBasisBulkState,
        rhs_bulk):
    n_r = int(state.vec_r.shape[0])
    n_t = int(state.vec_t.shape[0])
    n_z = int(state.vec_z.shape[0])
    n_common = int(state.inv_blocks.shape[0])

    r_slice = slice(0, n_r)
    t_slice = slice(n_r, n_r + n_t)
    z_slice = slice(n_r + n_t, n_r + n_t + n_z)

    def _to_modal(vecs, x):
        return vecs.T @ x

    def _from_modal(vecs, xm):
        return vecs @ xm

    rhs_r = rhs_bulk[r_slice]
    rhs_t = rhs_bulk[t_slice]
    rhs_z = rhs_bulk[z_slice]

    rhs_modal_r = _to_modal(state.vec_r, rhs_r)
    rhs_modal_t = _to_modal(state.vec_t, rhs_t)
    rhs_modal_z = _to_modal(state.vec_z, rhs_z)

    x_modal_r = state.inv_eval_r * rhs_modal_r
    x_modal_t = state.inv_eval_t * rhs_modal_t
    x_modal_z = state.inv_eval_z * rhs_modal_z

    if n_common > 0:
        y3 = jnp.stack(
            [rhs_modal_r[:n_common], rhs_modal_t[:n_common], rhs_modal_z[:n_common]],
            axis=1,
        )
        x3 = jnp.einsum("mij,mj->mi", state.inv_blocks, y3)
        x_modal_r = x_modal_r.at[:n_common].set(x3[:, 0])
        x_modal_t = x_modal_t.at[:n_common].set(x3[:, 1])
        x_modal_z = x_modal_z.at[:n_common].set(x3[:, 2])

    return jnp.concatenate([
        _from_modal(state.vec_r, x_modal_r),
        _from_modal(state.vec_t, x_modal_t),
        _from_modal(state.vec_z, x_modal_z),
    ])


def _build_k1_mode3x3_fd_preconditioner(seq, ops, *, dirichlet: bool, pinv_rtol: float = 1e-8):
    """Script-only k=1 prototype: modal per-mode 3x3 solves + surgery Schur.

    This is a debug implementation of the "true modewise 3x3" idea:
    1) assemble dense bulk stiffness, 2) build modal bases per component,
    3) solve coupled 3x3 blocks mode-by-mode (with safe_inv33), 4) transform
    back, 5) wrap with surgery Schur.

    Notes:
    - Extraction/surgery generally gives unequal component sizes; we couple the
      first min(n_r, n_theta, n_zeta) modes and fall back to diagonal modal
      pseudoinverse on unmatched tails.
    - Intended for diagnostics, not production timing.
    """
    surgery = _build_k1_stiffness_surgery_factors(seq, ops, dirichlet=dirichlet)
    n_r = int(surgery.rt_r_size)
    n_theta = int(surgery.bulk_rt_size - surgery.rt_r_size)
    n_zeta = int(surgery.bulk_zeta_size)
    n_bulk = int(surgery.bulk_indices.shape[0])

    eye_bulk = jnp.eye(n_bulk, dtype=jnp.float64)
    k_bulk = jax.vmap(
        lambda v: _apply_extracted_submatrix(
            surgery.apply_data,
            surgery.bulk_indices,
            surgery.bulk_indices,
            v,
        ),
        in_axes=1,
        out_axes=1,
    )(eye_bulk)
    k_bulk = _symmetrize(k_bulk)

    r_slice = slice(0, n_r)
    t_slice = slice(n_r, n_r + n_theta)
    z_slice = slice(n_r + n_theta, n_r + n_theta + n_zeta)

    k_rr = _symmetrize(k_bulk[r_slice, r_slice])
    k_tt = _symmetrize(k_bulk[t_slice, t_slice])
    k_zz = _symmetrize(k_bulk[z_slice, z_slice])

    # Orthonormal modal bases per component.
    eval_r, vec_r = jnp.linalg.eigh(k_rr)
    eval_t, vec_t = jnp.linalg.eigh(k_tt)
    eval_z, vec_z = jnp.linalg.eigh(k_zz)

    # Build transformed bulk operator K_hat = T^T K_bulk T with T block-diagonal.
    tmat = jnp.zeros((n_bulk, n_bulk), dtype=jnp.float64)
    tmat = tmat.at[r_slice, r_slice].set(vec_r)
    tmat = tmat.at[t_slice, t_slice].set(vec_t)
    tmat = tmat.at[z_slice, z_slice].set(vec_z)
    k_hat = _symmetrize(tmat.T @ (k_bulk @ tmat))

    n_common = int(min(n_r, n_theta, n_zeta))

    # Coupled 3x3 blocks for shared modal indices.
    blocks = jnp.stack([
        jnp.stack([
            jnp.array([
                k_hat[m, m],
                k_hat[m, n_r + m],
                k_hat[m, n_r + n_theta + m],
            ]),
            jnp.array([
                k_hat[n_r + m, m],
                k_hat[n_r + m, n_r + m],
                k_hat[n_r + m, n_r + n_theta + m],
            ]),
            jnp.array([
                k_hat[n_r + n_theta + m, m],
                k_hat[n_r + n_theta + m, n_r + m],
                k_hat[n_r + n_theta + m, n_r + n_theta + m],
            ]),
        ])
        for m in range(n_common)
    ], axis=0) if n_common > 0 else jnp.zeros((0, 3, 3), dtype=jnp.float64)

    inv_blocks = jax.vmap(lambda a: safe_inv33(_symmetrize(a), tol=pinv_rtol))(blocks)

    # Tail modes: diagonal pseudoinverse in modal basis.
    def _diag_pinv_from_eval(evals):
        scale = jnp.max(jnp.abs(evals))
        cutoff = jnp.maximum(pinv_rtol * jnp.where(scale > 0, scale, 1.0), 1e-14)
        return jnp.where(evals > cutoff, 1.0 / evals, 0.0)

    inv_eval_r = _diag_pinv_from_eval(eval_r)
    inv_eval_t = _diag_pinv_from_eval(eval_t)
    inv_eval_z = _diag_pinv_from_eval(eval_z)

    def bulk_apply(rhs_bulk):
        y = tmat.T @ rhs_bulk
        y_r = y[r_slice]
        y_t = y[t_slice]
        y_z = y[z_slice]

        x_r = jnp.zeros_like(y_r)
        x_t = jnp.zeros_like(y_t)
        x_z = jnp.zeros_like(y_z)

        if n_common > 0:
            y3 = jnp.stack([y_r[:n_common], y_t[:n_common], y_z[:n_common]], axis=1)
            x3 = jnp.einsum("mij,mj->mi", inv_blocks, y3)
            x_r = x_r.at[:n_common].set(x3[:, 0])
            x_t = x_t.at[:n_common].set(x3[:, 1])
            x_z = x_z.at[:n_common].set(x3[:, 2])

        if n_r > n_common:
            x_r = x_r.at[n_common:].set(inv_eval_r[n_common:] * y_r[n_common:])
        if n_theta > n_common:
            x_t = x_t.at[n_common:].set(inv_eval_t[n_common:] * y_t[n_common:])
        if n_zeta > n_common:
            x_z = x_z.at[n_common:].set(inv_eval_z[n_common:] * y_z[n_common:])

        x_modal = jnp.concatenate([x_r, x_t, x_z])
        return tmat @ x_modal

    # Surgery Schur assembled on top of the modal bulk surrogate.
    n_s = int(surgery.surgery_size)
    eye_s = jnp.eye(n_s, dtype=jnp.float64)
    schur_dense = jax.vmap(
        lambda v: surgery.ass @ v
        - _apply_bulk_to_surgery_coupling(
            surgery,
            bulk_apply(_apply_surgery_to_bulk_coupling(surgery, v)),
        ),
        in_axes=1,
        out_axes=1,
    )(eye_s)
    schur_dense = _symmetrize(schur_dense)
    evals_s, evecs_s = jnp.linalg.eigh(schur_dense)
    scale_s = jnp.max(jnp.abs(evals_s))
    cutoff_s = jnp.maximum(pinv_rtol * jnp.where(scale_s > 0, scale_s, 1.0), 1e-14)
    inv_evals_s = jnp.where(evals_s > cutoff_s, 1.0 / evals_s, 0.0)
    schur_inv = (evecs_s * inv_evals_s[jnp.newaxis, :]) @ evecs_s.T

    def apply(v):
        rhs_s = v[surgery.surgery_indices]
        rhs_b = v[surgery.bulk_indices]
        y = bulk_apply(rhs_b)
        z = schur_inv @ (rhs_s - _apply_bulk_to_surgery_coupling(surgery, y))
        x_b = y - bulk_apply(_apply_surgery_to_bulk_coupling(surgery, z))
        x = jnp.zeros_like(v)
        x = x.at[surgery.surgery_indices].set(z)
        x = x.at[surgery.bulk_indices].set(x_b)
        return x

    return apply


def build_sequence(args) -> DeRhamSequence:
    ns = tuple(int(v) for v in args.ns)
    seq = DeRhamSequence(
        ns,
        (args.p, args.p, args.p),
        2 * args.p,
        TYPES,
        polar=True,
        tol=args.cg_tol,
        maxiter=args.cg_maxiter,
        betti_numbers=BETTI,
    )
    seq.evaluate_1d()
    seq.assemble_reference_mass_matrix()
    if args.geometry == "rotating_ellipse":
        seq.set_map(rotating_ellipse_map(
            eps=args.epsilon, kappa=args.kappa, R0=args.r0, nfp=args.nfp))
    elif args.geometry == "cylinder":
        # Periodic cylinder F(r, chi, z) = (a r cos2pi chi, a r sin2pi chi, h z),
        # periodic in chi and z. a = epsilon*R0 keeps the minor radius comparable
        # to the toroid; h = R0 sets the (periodic) axial length scale.
        seq.set_map(cylinder_map(a=args.epsilon * args.r0, h=args.r0))
    elif args.geometry == "w7x":
        # W7-X stellarator (nfp=5): greville-interpolate R,Z from data/W7-X.h5
        # at the SAME resolution as the solve seq. Bound the geometry jacfwd
        # memory (batched lax.map) -- a full-vmap over all quad points of the
        # recursive-spline map OOMs.
        import mrx as _mrx  # noqa: PLC0415
        _mrx.MAP_BATCH_SIZE_INNER = int(os.environ.get("W7X_MAP_BATCH", "256"))
        _dbg = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "debug")
        if _dbg not in sys.path:
            sys.path.insert(0, _dbg)
        from w7x_geometry import build_w7x_map  # noqa: PLC0415
        map_func, _ = build_w7x_map(map_ns=ns, p=args.p)
        seq.set_map(map_func)
    else:
        seq.set_map(toroid_map(epsilon=args.epsilon, kappa=args.kappa, R0=args.r0))
    return seq


def _sync_pytree(tree):
    """Best-effort device sync for timing (no-op for non-array leaves)."""
    for leaf in jax.tree_util.tree_leaves(tree):
        block = getattr(leaf, "block_until_ready", None)
        if callable(block):
            block()


def assemble_operators(
        seq: DeRhamSequence,
        *,
        rank: int = RANK,
        k1_stiff_inner_schur: bool = False,
        precompute_coupling: bool = True,
        timing_breakdown: bool = False,
        klevel: int = 1,
        both_bc: bool = False):
    def _timed(label, fn, ops_in):
        if not timing_breakdown:
            return fn(ops_in)
        t_step = time.perf_counter()
        out = fn(ops_in)
        _sync_pytree(out)
        dt_ms = (time.perf_counter() - t_step) * 1e3
        print(f"[diag] assembly step {label:<24} {dt_ms:>10.1f} ms")
        return out

    ops = seq.get_operators()
    ops = _timed(
        "mass_jacobi",
        lambda o: assemble_mass_jacobi_preconditioner(seq, operators=o, ks=(0, 1, 2, 3)),
        ops,
    )
    ops = _timed(
        "incidence",
        lambda o: assemble_incidence_operators(seq, operators=o, ks=(0, 1, 2)),
        ops,
    )
    ops = _timed(
        "laplacian",
        lambda o: assemble_laplacian_operators(seq, seq.geometry, operators=o, ks=(0, 1, 2, 3)),
        ops,
    )
    # Tensor preconditioners (rank set by caller; default RANK).
    ops = _timed(
        "tensor_mass",
        lambda o: assemble_tensor_mass_preconditioner(
            seq, operators=o, ks=(0, 1, 2, 3), rank=rank,
            cp_kwargs={"precompute_coupling": precompute_coupling},
        ),
        ops,
    )
    ops = _timed(
        "tensor_laplacian",
        lambda o: assemble_tensor_laplacian_preconditioner(
            seq, operators=o, ks=(0,), rank=rank,
            cp_kwargs={"precompute_coupling": precompute_coupling},
        ),
        ops,
    )
    # Tensor stiffness: k=1 for P_A candidates (k=1 run) AND as the K_1^{-1} atom
    # for the k=2 preconditioner; k=2 div-div P_A is added for the k=2 run. The
    # k=3 unified P_B uses the whole k=2 preconditioner as its inner L_2^{-1}
    # atom, so klevel=3 ALSO needs the k=1+k=2 tensor stiffness. k=0 (nullspace
    # test) needs none (S_3 = 0 itself carries no stiffness).
    if klevel in (1, 2, 3):
        stiff_ks = (1, 2) if klevel in (2, 3) else (1,)
        ops = _timed(
            "tensor_stiffness",
            lambda o: assemble_tensor_stiffness_preconditioner(
                seq,
                operators=o,
                ks=stiff_ks,
                rank=rank,
                cp_kwargs={
                    "bulk_schur": k1_stiff_inner_schur,
                    "precompute_coupling": precompute_coupling,
                },
            ),
            ops,
        )
    # k=3 auxiliary-space transfer needs the V0<->V3 projection (cross-mass)
    # blocks; the k=0 tensor Hodge preconditioner (both BC variants) is already
    # assembled above via tensor_laplacian ks=(0,).
    if klevel == 3:
        ops = _timed(
            "projection_03",
            lambda o: assemble_projection_operators(
                seq, operators=o, pairs=((0, 3), (3, 0))),
            ops,
        )
    # Production baseline: Schur-outer Jacobi diagonal, rank-independent diag mode.
    # k=3 runs the saddle with free BCs (its dual k=0 is dbc), so its baseline
    # jacobi is the no-dbc Schur diagonal. k=0 is a condensed (non-saddle) solve,
    # so no Schur jacobi is needed.
    if klevel in (1, 2, 3):
        # klevel=3's unified P_B nests the whole k=2 preconditioner (which builds
        # the k=1 smoother), and both call _get_schur_diaginv eagerly -> assemble
        # the k=1/k=2/k=3 Schur-Jacobi (all free) for klevel=3.
        schur_ks = {1: (1,), 2: (1, 2), 3: (1, 2, 3)}[klevel]
        schur_bc = (False,) if klevel == 3 else (DIRICHLET,)
        if both_bc:  # k=1 nullspace test needs the free-BC Schur jacobi too
            schur_bc = (True, False)
        ops = _timed(
            "schur_jacobi",
            lambda o: assemble_schur_jacobi_preconditioner(
                seq,
                operators=o,
                ks=schur_ks,
                dirichlet_variants=schur_bc,
                schur_diag_mode='tensor_probe',
            ),
            ops,
        )
    return seq.set_operators(ops)


def make_apply_routines(
    seq: DeRhamSequence,
    ops,
    *,
    pa_mode: str,
    grad_project: bool = False,
    pa_block_vector_fd: bool = False,
    pa_block_vector_fd_true_basis: bool = False,
    pa_block_radial_banded: bool = False,
    pa_block_vector_fd_regularization_rel: float = 1e-2,
    pa_block_vector_fd_low_mode_exclude: int = 0,
    pa_block_vector_fd_report_k: int = 8,
    pa_block_vector_fd_origin_k: int = 4,
    pa_profile: bool = False,
    dirichlet_flag: bool = DIRICHLET,
    true_g: bool = False,
    l0_inv_custom=None,
):
    """Return condensed/saddle applies and upper/lower preconditioners.

    When ``grad_project`` is True the active ``P_A`` is sandwiched between
    gradient-subspace complement projectors so it acts only on the curl-
    dominated complement, leaving the gradient (curl-free) subspace entirely
    to ``P_B``. This removes the additive double-counting where ``P_A`` and
    ``P_B`` both act on the gradient modes.

    ``dirichlet_flag`` selects the boundary condition for the WHOLE routine
    (k=1 dbc by default, or free/no-dbc with a b1=1 harmonic). It locally
    shadows the module ``DIRICHLET`` so all applies below use the chosen BC.
    """
    DIRICHLET = dirichlet_flag  # local shadow: all uses below use this BC

    # Opt into the TRUE polar derivative G_0 = Gram_1^{-1}.(E^T sp E) by shadowing
    # apply_incidence_matrix locally (all P_B / projector uses below pick it up).
    # The k=1 projector only uses G_0 (grad), whose output space is V1.
    if true_g:
        apply_incidence_matrix = _make_true_incidence(
            {1: _build_gram_inv(seq, ops, 1, DIRICHLET)})
    else:
        apply_incidence_matrix = _apply_incidence_raw

    def a_matvec(v):
        # Linear SPD condensed Hodge Laplacian: K_1 + D_0 (M_0-precond) D_0^T.
        return apply_laplacian_approx(seq, ops, v, 1, dirichlet=DIRICHLET)

    def mass_matvec(v):
        return apply_mass_matrix(seq, ops, v, 1, dirichlet=DIRICHLET)

    def stiffness_matvec(v):
        return apply_stiffness(seq, ops, v, 1, dirichlet=DIRICHLET)

    def derivative_matvec(sigma):
        return apply_derivative_matrix(
            seq, ops, sigma, 0,
            dirichlet_in=DIRICHLET, dirichlet_out=DIRICHLET, transpose=False)

    def derivative_t_matvec(u):
        return apply_derivative_matrix(
            seq, ops, u, 0,
            dirichlet_in=DIRICHLET, dirichlet_out=DIRICHLET, transpose=True)

    def mass_lower_matvec(sigma):
        return apply_mass_matrix(seq, ops, sigma, 0, dirichlet=DIRICHLET)

    def lower_tensor_precond(rhs):
        # Fixed lower block: rank-1 tensor mass preconditioner.
        return apply_mass_matrix_preconditioner(
            seq, ops, rhs, 0, dirichlet=DIRICHLET, kind="tensor")

    def l0_inv(x):
        # Rank-1 tensor k=0 Hodge-Laplacian preconditioner: V0* -> V0. The
        # production apply now uses the precomputed dense core<->bulk coupling
        # block (factors.core_coupling, built by default in
        # _assemble_k0_tensor_hodge_preconditioner), so no script-local
        # precompute is needed.
        return apply_laplacian_preconditioner(
            seq, ops, x, 0, dirichlet=DIRICHLET, kind="tensor")

    # Diagnostic hook: inject an exact (dense) L_0^{-1} so P_B and the gradient
    # projectors use it instead of the tensor atom -- the k=1 analog of the k=2
    # script's exact-K_1^{-1} comparison. Overriding l0_inv before the projectors
    # are built routes it everywhere (p_b, Pi_g).
    if l0_inv_custom is not None:
        l0_inv = l0_inv_custom

    def l0_inv_exact(x):
        # Exact (CG) solve of the projector's own k=0 operator L_0 y = x,
        # V0* -> V0. To make Pi = G_0 L_0^{-1} G_0^T M_1 a true (idempotent,
        # M_1-self-adjoint) projector, L_0 must be exactly the operator the
        # projector implies: G_0^T M_1^dbc G_0, built from the SAME DBC
        # incidence and e1-extracted mass applies the projector uses.
        #
        # Note: this differs from apply_stiffness(.,0), which uses the
        # un-extracted core M_1 (extraction only on the 0-form side); that
        # inconsistency is what made the gradient-energy fraction exceed 1.
        # All applies below are matrix-free. Diagnostic-only: never called from
        # inside a preconditioner apply, so this is not a Krylov-in-Krylov nest.
        def l0_matvec(v):
            g0_v = apply_incidence_matrix(
                seq, ops, v, 0,
                dirichlet_in=DIRICHLET, dirichlet_out=DIRICHLET, transpose=False)  # V0  -> V1
            m1_g0_v = apply_mass_matrix(seq, ops, g0_v, 1, dirichlet=DIRICHLET)    # V1  -> V1*
            return apply_incidence_matrix(
                seq, ops, m1_g0_v, 0,
                dirichlet_in=DIRICHLET, dirichlet_out=DIRICHLET, transpose=True)   # V1* -> V0*

        def mass0_matvec(v):
            return apply_mass_matrix(seq, ops, v, 0, dirichlet=DIRICHLET)

        y, _info = solve_singular_cg(
            l0_matvec,
            x,
            mass_matvec=mass0_matvec,
            precond_matvec=l0_inv,
            vs=[],  # k=0 DBC for BETTI=(1,1,0,0): no constants
            tol=seq.tol,
            maxiter=seq.maxiter,
        )
        return y

    k_diaginv = ops.dd1_diaginv_dbc if DIRICHLET else ops.dd1_diaginv
    if k_diaginv is None:
        # Some assembly paths do not populate dd1_diaginv eagerly; recover it
        # by probing the k=1 stiffness diagonal from matrix-free applies.
        size = int(seq.n1_dbc if DIRICHLET else seq.n1)
        k_diag = _diagonal_from_matvec(
            lambda x: apply_stiffness(seq, ops, x, 1, dirichlet=DIRICHLET),
            size,
        )
        k_diaginv = _invert_diagonal(k_diag)

    # Production baseline: Schur-outer Jacobi diagonal (mode diag), a stored
    # diagonal multiply. The tensor inner is already inside `a_matvec`.
    schur_diaginv = _get_schur_diaginv(ops, 1, DIRICHLET, 'diag')
    if schur_diaginv is None:
        raise RuntimeError("Schur jacobi diag preconditioner was not assembled")

    def jacobi_diag(r):
        return schur_diaginv * r

    def p_b(r):
        # Dual-Poisson potential block P_B = G_0 L_0^{-1} M_0 L_0^{-1} G_0^T.
        y1 = apply_incidence_matrix(
            seq, ops, r, 0,
            dirichlet_in=DIRICHLET, dirichlet_out=DIRICHLET, transpose=True)   # V1* -> V0*
        y2 = l0_inv(y1)                                                        # V0* -> V0
        y3 = apply_mass_matrix(seq, ops, y2, 0, dirichlet=DIRICHLET)           # V0  -> V0*
        y4 = l0_inv(y3)                                                        # V0* -> V0
        u = apply_incidence_matrix(
            seq, ops, y4, 0,
            dirichlet_in=DIRICHLET, dirichlet_out=DIRICHLET, transpose=False)  # V0  -> V1
        return u

    if pa_mode == "block_fd":
        # Block-component pseudoinverse preconditioner with surgery Schur.
        # Each bulk component (r, theta, zeta) is inverted by a symmetric
        # pseudoinverse of its dense stiffness block; the surgery rows are
        # handled by an exact Schur complement.  Provably PSD.
        _block_fd_payload = _build_k1_block_fd_preconditioner(
            seq,
            ops,
            dirichlet=DIRICHLET,
            pinv_rtol=1e-8,
            use_vector_fd=pa_block_vector_fd,
            use_vector_fd_true_basis=pa_block_vector_fd_true_basis,
            use_radial_banded=pa_block_radial_banded,
            vector_fd_regularization_rel=pa_block_vector_fd_regularization_rel,
            vector_fd_low_mode_exclude=pa_block_vector_fd_low_mode_exclude,
            vector_fd_report_k=pa_block_vector_fd_report_k,
            vector_fd_origin_k=pa_block_vector_fd_origin_k,
            return_profile=pa_profile,
        )
        if pa_profile:
            _block_fd_state, _ = _block_fd_payload

            def _block_fd_profile(v):
                return _profile_k1_block_fd_preconditioner_from_state(
                    _block_fd_state,
                    v,
                )
        else:
            _block_fd_state = _block_fd_payload
            _block_fd_profile = None

        def p_a_raw(r):
            return _apply_k1_block_fd_preconditioner_from_state(_block_fd_state, r)
    else:
        raise ValueError(
            f"Unsupported pa_mode={pa_mode!r}; expected 'block_fd'"
        )

    # Gradient-subspace complement projectors (M_1-orthogonal).
    #   Pi   = G_0 L_0^{-1} G_0^T M_1   : V1 -> V1   (onto gradient subspace)
    #   Pi^* = M_1 G_0 L_0^{-1} G_0^T   : V1* -> V1* (dual, adjoint of Pi)
    # The complements C = I - Pi and C^* = I - Pi^* keep the curl-dominated
    # part. The preconditioner sandwich uses the tensor l0_inv (cheap, matrix-
    # free); the diagnostic also builds an exact-L_0 variant to separate the
    # inexactness of the tensor solve from the genuine overlap signal.
    def _make_primal_complement(l0_solve):
        def project(u):
            # (I - Pi) u = u - G_0 L_0^{-1} G_0^T M_1 u, maps V1 -> V1.
            y0 = apply_mass_matrix(seq, ops, u, 1, dirichlet=DIRICHLET)            # V1  -> V1*
            y1 = apply_incidence_matrix(
                seq, ops, y0, 0,
                dirichlet_in=DIRICHLET, dirichlet_out=DIRICHLET, transpose=True)   # V1* -> V0*
            y2 = l0_solve(y1)                                                      # V0* -> V0
            y3 = apply_incidence_matrix(
                seq, ops, y2, 0,
                dirichlet_in=DIRICHLET, dirichlet_out=DIRICHLET, transpose=False)  # V0  -> V1
            return u - y3
        return project

    def _make_dual_complement(l0_solve):
        def project(r):
            # (I - Pi^*) r = r - M_1 G_0 L_0^{-1} G_0^T r, maps V1* -> V1*.
            y1 = apply_incidence_matrix(
                seq, ops, r, 0,
                dirichlet_in=DIRICHLET, dirichlet_out=DIRICHLET, transpose=True)   # V1* -> V0*
            y2 = l0_solve(y1)                                                      # V0* -> V0
            y3 = apply_incidence_matrix(
                seq, ops, y2, 0,
                dirichlet_in=DIRICHLET, dirichlet_out=DIRICHLET, transpose=False)  # V0  -> V1
            y4 = apply_mass_matrix(seq, ops, y3, 1, dirichlet=DIRICHLET)           # V1  -> V1*
            return r - y4
        return project

    project_primal_complement = _make_primal_complement(l0_inv)
    project_dual_complement = _make_dual_complement(l0_inv)
    project_primal_complement_exact = _make_primal_complement(l0_inv_exact)

    def p_a_projected(r):
        # Symmetric sandwich C P_A C^*: P_A acts only on the curl complement.
        return project_primal_complement(p_a_raw(project_dual_complement(r)))

    def p_a_with_state(state, r):
        if grad_project:
            return project_primal_complement(
                _apply_k1_block_fd_preconditioner_from_state(
                    state,
                    project_dual_complement(r),
                )
            )
        return _apply_k1_block_fd_preconditioner_from_state(state, r)

    def p_a_raw_with_state(state, r):
        return _apply_k1_block_fd_preconditioner_from_state(state, r)

    def p_a_plus_p_b_with_state(state, r):
        return p_a_with_state(state, r) + p_b(r)

    def jacobi_plus_p_a_plus_p_b_with_state(state, r):
        return jacobi_diag(r) + p_a_with_state(state, r) + p_b(r)

    def jacobi_scaled_plus_p_a_plus_p_b_with_state(state, r, jacobi_scale):
        return jacobi_scale * jacobi_diag(r) + p_a_with_state(state, r) + p_b(r)

    def jacobi_plus_p_a_raw_plus_p_b_with_state(state, r):
        return jacobi_diag(r) + p_a_raw_with_state(state, r) + p_b(r)

    p_a = p_a_projected if grad_project else p_a_raw

    def k_jacobi_plus_p_b(r):
        return k_diaginv * r + p_b(r)

    def projected_jacobi_plus_p_b(r):
        # Symmetric projected Schur-Jacobi on the curl complement plus P_B.
        # This tests whether a cheap projected diagonal model can replace P_A.
        jac = jacobi_diag(project_dual_complement(r))
        return project_primal_complement(jac) + p_b(r)

    def projected_p_a_plus_p_b(r):
        # Canonical projected tensorial block: P^T P_S P + P_B.
        pa_proj = p_a_raw(project_dual_complement(r))
        return project_primal_complement(pa_proj) + p_b(r)

    def projected_p_a_plus_p_b_with_state(state, r):
        # State-threaded version of P^T P_S P + P_B to avoid capturing large
        # bulk payloads as compile-time constants in jitted solve closures.
        pa_proj = p_a_raw_with_state(state, project_dual_complement(r))
        return project_primal_complement(pa_proj) + p_b(r)

    def projected_p_a_plus_p_b_fused_with_state(state, r):
        # Algebraically IDENTICAL to projected_p_a_plus_p_b_with_state
        # (C P_S C^* + P_B) but evaluated with 2 L_0^{-1} solves instead of 4.
        # Two redundancies are removed exactly (no approximation):
        #   1. The dual-projection inner solve and the P_B inner solve are the
        #      same vector q = L_0^{-1} G_0^T r -> computed once.
        #   2. The primal-projection outer solve and the P_B outer solve are
        #      both G_0 L_0^{-1}(.) -> their arguments are summed and solved
        #      once: G_0 L_0^{-1}(M_0 q - G_0^T M_1 w).
        # Symmetry (hence MINRES iteration count) is preserved by construction.
        gtr = apply_incidence_matrix(
            seq, ops, r, 0,
            dirichlet_in=DIRICHLET, dirichlet_out=DIRICHLET, transpose=True)   # G_0^T r
        q = l0_inv(gtr)                                                        # L_0^{-1} G_0^T r
        gq = apply_incidence_matrix(
            seq, ops, q, 0,
            dirichlet_in=DIRICHLET, dirichlet_out=DIRICHLET, transpose=False)  # G_0 q
        pstar_r = r - apply_mass_matrix(seq, ops, gq, 1, dirichlet=DIRICHLET)  # C^* r = r - M_1 G_0 q
        w = p_a_raw_with_state(state, pstar_r)                                 # P_S C^* r
        m0q = apply_mass_matrix(seq, ops, q, 0, dirichlet=DIRICHLET)           # M_0 q
        m1w = apply_mass_matrix(seq, ops, w, 1, dirichlet=DIRICHLET)           # M_1 w
        gtm1w = apply_incidence_matrix(
            seq, ops, m1w, 0,
            dirichlet_in=DIRICHLET, dirichlet_out=DIRICHLET, transpose=True)   # G_0^T M_1 w
        outer = apply_incidence_matrix(
            seq, ops, l0_inv(m0q - gtm1w), 0,
            dirichlet_in=DIRICHLET, dirichlet_out=DIRICHLET, transpose=False)  # G_0 L_0^{-1}(M_0 q - G_0^T M_1 w)
        return w + outer

    def projected_p_b_plus_p_b(r):
        # Same idea as the projected-Jacobi comparator, but using P_B in the
        # projected slot instead of the diagonal Schur approximation.
        pb_proj = p_b(project_dual_complement(r))
        return project_primal_complement(pb_proj) + p_b(r)

    def p_a_plus_p_b(r):
        return p_a_plus_p_b_with_state(_block_fd_state, r)

    def jacobi_plus_p_a_plus_p_b(r):
        return jacobi_plus_p_a_plus_p_b_with_state(_block_fd_state, r)

    def jacobi_scaled_plus_p_a_plus_p_b(r, jacobi_scale):
        return jacobi_scaled_plus_p_a_plus_p_b_with_state(_block_fd_state, r, jacobi_scale)

    return {
        "a_matvec": a_matvec,
        "mass_matvec": mass_matvec,
        "stiffness_matvec": stiffness_matvec,
        "derivative_matvec": derivative_matvec,
        "derivative_t_matvec": derivative_t_matvec,
        "mass_lower_matvec": mass_lower_matvec,
        "lower_tensor_precond": lower_tensor_precond,
        "p_a": p_a,
        "p_a_raw": p_a_raw,
        "p_a_projected": p_a_projected,
        "project_primal_complement": project_primal_complement,
        "project_primal_complement_exact": project_primal_complement_exact,
        "p_a_plus_p_b": p_a_plus_p_b,
        "jacobi_plus_p_a_plus_p_b": jacobi_plus_p_a_plus_p_b,
        "jacobi_scaled_plus_p_a_plus_p_b": jacobi_scaled_plus_p_a_plus_p_b,
        "k_jacobi_plus_p_b": k_jacobi_plus_p_b,
        "projected_p_a_plus_p_b": projected_p_a_plus_p_b,
        "projected_jacobi_plus_p_b": projected_jacobi_plus_p_b,
        "projected_p_b_plus_p_b": projected_p_b_plus_p_b,
        "jacobi_diag": jacobi_diag,
        "schur_diaginv": schur_diaginv,
        "k_diaginv": k_diaginv,
        "p_b": p_b,
        "p_a_profile": _block_fd_profile,
        "p_a_state": _block_fd_state,
        "p_a_with_state": p_a_with_state,
        "p_a_raw_with_state": p_a_raw_with_state,
        "p_a_plus_p_b_with_state": p_a_plus_p_b_with_state,
        "jacobi_plus_p_a_plus_p_b_with_state": jacobi_plus_p_a_plus_p_b_with_state,
        "jacobi_scaled_plus_p_a_plus_p_b_with_state": jacobi_scaled_plus_p_a_plus_p_b_with_state,
        "jacobi_plus_p_a_raw_plus_p_b_with_state": jacobi_plus_p_a_raw_plus_p_b_with_state,
        "projected_p_a_plus_p_b_with_state": projected_p_a_plus_p_b_with_state,
        "projected_p_a_plus_p_b_fused_with_state": projected_p_a_plus_p_b_fused_with_state,
    }


def make_apply_routines_k2(seq: DeRhamSequence, ops, *, grad_project: bool = True,
                           project_atom: bool = True, atom: str = "block_fd",
                           k1_inv_custom=None, true_g: bool = False,
                           atom_cheb_degree: int = 5, atom_cheb_eps=None,
                           atom_cheb_max_degree: int = 100, l0_cheb_eps=None):
    """k=2 Hodge-Laplacian preconditioner applies (degree-shifted from k=1).

    The k=2 Laplacian ``L_2 = S_2 + D_1 M_1^{-1} D_1^T`` has the div-div
    stiffness ``S_2 = G_2^T M_3 G_2`` (singular on curls ``ran(G_1)``) plus a
    curl-handling term. The preconditioner mirrors the validated k=1 path one
    degree up, with two nested projection sandwiches:

    * Outer (curl-complement): the div-div tensor inverse ``P_A`` is sandwiched
      between ``I - Pi_2`` so it acts only on the co-exact complement, leaving
      the curl subspace to ``P_B``. ``Pi_2 = G_1 K_1^{-1} G_1^T M_2``.
    * ``P_B = G_1 K_1^{-1} M_1 K_1^{-1} G_1^T`` is the curl-subspace correction
      (the 1:1 analog of k=1's ``G_0 L_0^{-1} M_0 L_0^{-1} G_0^T``).
    * Atom ``K_1^{-1}`` (``K_1 = G_1^T M_2 G_1`` is the 1-form curl-curl
      stiffness) is the k=1 stiffness tensor preconditioner, but block_fd is
      singular on gradients ``ran(G_0)`` so it is itself sandwiched in an inner
      grad-complement ``I - Pi_g`` built from the cheap, near-exact scalar
      ``L_0^{-1}`` (k=0 Hodge tensor preconditioner). ``Pi_g = G_0 L_0^{-1}
      G_0^T M_1``. The outer ``G_1`` shields the inner ``K_1^{-1}`` apply in
      both ``P_B`` and ``Pi_2``; the inner grad-complement shields the exposed
      outer ``K_1^{-1}`` apply (where ``M_1`` remixes gradient content back in).

    All applies are matrix-free / dense-tensor (no inner Krylov).
    """

    # Opt into the TRUE polar curl G_1 = Gram_2^{-1}.(E^T sp E) by shadowing
    # apply_incidence_matrix locally; the projector/P_B (G_1, output space V2)
    # then use the true curl, so the composed K_1 = G_1^T M_2 G_1 matches
    # apply_stiffness(.,1) -- the operator the atoms invert.
    if true_g:
        apply_incidence_matrix = _make_true_incidence(
            {2: _build_gram_inv(seq, ops, 2, DIRICHLET)})
    else:
        apply_incidence_matrix = _apply_incidence_raw

    # --- forward matvecs for the k=2 saddle system ---
    def a_matvec(v):
        return apply_laplacian_approx(seq, ops, v, 2, dirichlet=DIRICHLET)

    def mass_matvec(v):  # M_2 : V2 -> V2*
        return apply_mass_matrix(seq, ops, v, 2, dirichlet=DIRICHLET)

    def stiffness_matvec(v):  # S_2 : V2 -> V2*
        return apply_stiffness(seq, ops, v, 2, dirichlet=DIRICHLET)

    def derivative_matvec(sigma):  # D_1 : V1 -> V2
        return apply_derivative_matrix(
            seq, ops, sigma, 1,
            dirichlet_in=DIRICHLET, dirichlet_out=DIRICHLET, transpose=False)

    def derivative_t_matvec(u):  # D_1^T : V2 -> V1
        return apply_derivative_matrix(
            seq, ops, u, 1,
            dirichlet_in=DIRICHLET, dirichlet_out=DIRICHLET, transpose=True)

    def mass_lower_matvec(sigma):  # M_1 : V1 -> V1*
        return apply_mass_matrix(seq, ops, sigma, 1, dirichlet=DIRICHLET)

    def lower_tensor_precond(rhs):  # M_1^{-1} (tensor) : V1* -> V1
        return apply_mass_matrix_preconditioner(
            seq, ops, rhs, 1, dirichlet=DIRICHLET, kind="tensor")

    # --- auxiliary inverses ---
    def l0_inv(x):  # k=0 Hodge tensor precond, V0* -> V0 (cheap, near-exact)
        return apply_laplacian_preconditioner(
            seq, ops, x, 0, dirichlet=DIRICHLET, kind="tensor")

    def k1_stiff_inv_raw(x):  # raw k=1 curl-curl tensor precond, V1* -> V1
        return apply_stiffness_tensor_preconditioner(
            seq, ops, x, 1, dirichlet=DIRICHLET)

    # Inner grad-complement projectors on V1 (M_1-orthogonal), identical in form
    # to the k=1 routine's gradient projectors (G_0, L_0^{-1}, M_1).
    def grad_primal_complement(u):  # (I - Pi_g) u : V1 -> V1
        y0 = apply_mass_matrix(seq, ops, u, 1, dirichlet=DIRICHLET)             # V1  -> V1*
        y1 = apply_incidence_matrix(
            seq, ops, y0, 0,
            dirichlet_in=DIRICHLET, dirichlet_out=DIRICHLET, transpose=True)    # V1* -> V0*
        y2 = l0_inv(y1)                                                         # V0* -> V0
        y3 = apply_incidence_matrix(
            seq, ops, y2, 0,
            dirichlet_in=DIRICHLET, dirichlet_out=DIRICHLET, transpose=False)   # V0  -> V1
        return u - y3

    def grad_dual_complement(r):  # (I - Pi_g^*) r : V1* -> V1*
        y1 = apply_incidence_matrix(
            seq, ops, r, 0,
            dirichlet_in=DIRICHLET, dirichlet_out=DIRICHLET, transpose=True)    # V1* -> V0*
        y2 = l0_inv(y1)                                                         # V0* -> V0
        y3 = apply_incidence_matrix(
            seq, ops, y2, 0,
            dirichlet_in=DIRICHLET, dirichlet_out=DIRICHLET, transpose=False)   # V0  -> V1
        y4 = apply_mass_matrix(seq, ops, y3, 1, dirichlet=DIRICHLET)            # V1  -> V1*
        return r - y4

    # Optional: the FULL nonsingular L_1 = K_1 + grad-div as the atom, applied via
    # the validated k=1 Hodge preconditioner (projected P_A + P_B; a single
    # matrix-free apply, NO inner Krylov). On the curl-aux input b = G_1^T M_2 r
    # (which satisfies G_0^T b = 0), the discrete Hodge decomposition gives
    # L_1^{-1} b = K_1^+ b EXACTLY -- and the M_1 in the squared P_B preserves
    # G_0^T(.) = 0 so the exposed outer apply is exact too. But L_1 has no
    # near-kernel, so the cheap apply is robust precisely where the singular
    # block_fd K_1^{-1} blows up. The inner grad-complement sandwich is then
    # unnecessary (L_1 is nonsingular); project_atom is ignored for this atom.
    if atom == "full_l1":
        _k1_applies = make_apply_routines(
            seq, ops, pa_mode="block_fd", grad_project=True, dirichlet_flag=DIRICHLET,
            true_g=true_g)
        _l1_inv = _k1_applies["projected_p_a_plus_p_b"]  # V1* -> V1, approx L_1^{-1}

        def k1_inv(r):  # full-L_1 pseudoinverse surrogate, V1* -> V1
            return _l1_inv(r)
    elif atom == "cheb_tensor":
        # The cheap, scalable, NEAR-EXACT L_1^{-1} atom: a fixed-degree Chebyshev
        # iteration on the k=1 approximate Schur S-hat_1 (~ L_1) with the GOOD k=1
        # Hodge preconditioner (kappa~28) as the inner smoother. All matrix-free
        # (no inner Krylov, no dense pinv) -> the production-legal version of the
        # exact L_1^+ that converged the projected k=2 method.
        #
        # DEFLATION (free BC): L_1 has the b1 cohomology harmonic h (L_1 h = 0). The
        # Chebyshev polynomial p(lambda) ~ 1/lambda BLOWS UP at lambda ~ 0, so an
        # un-deflated atom amplifies h and wrecks the projector Pi_2. We
        # M_1-orthogonally project h out of the atom's input (dual) and output
        # (primal), turning it into a proper deflated PSEUDO-inverse L_1^+. dbc has
        # no harmonic -> no-op. h from _nullspace_vectors (the same harmonic the k=1
        # free saddle deflates). Interval via A-inner-product Lanczos on the
        # DEFLATED smoother (so lmin is the bulk min, not the ~0 harmonic).
        # NESTED near-exact inner L_0^{-1}. The rough single-apply tensor l0 was the
        # DOMINANT limiter on the smoother quality: with an EXACT l0, block_fd P_A
        # already gives cond(P_hodge.L_1)=9 and NO near-null cluster, vs 152 + a
        # 17-mode cluster with the single rough tensor l0 (job 14603363). So nest a
        # near-exact CHEB-L_0 atom -- a Chebyshev on L_0=apply_stiffness(.,0) (exact,
        # matrix-free) with the CONSTANT-DEFLATED k=0 tensor precond as smoother,
        # degree from Lanczos kappa -- as the l0 used inside the k=1 smoother. This
        # recurses the same template one degree down and bottoms out at the
        # near-exact (kappa~6) scalar k=0. Constant deflation (free k=0 null) is
        # mandatory: a Chebyshev amplifies the constant null otherwise.
        _n0 = int(seq.n0_dbc if DIRICHLET else seq.n0)
        _c0 = jnp.asarray(_nullspace_vectors(ops, 0, DIRICHLET))   # (nc, n0)
        if _c0.shape[0] > 0:
            _Mc0 = jnp.stack(
                [apply_mass_matrix(seq, ops, _c0[i], 0, dirichlet=DIRICHLET)
                 for i in range(_c0.shape[0])], axis=0)
            _c0n = jnp.sqrt(jnp.einsum("ij,ij->i", _c0, _Mc0))
            _c0 = _c0 / _c0n[:, None]
            _Mc0 = _Mc0 / _c0n[:, None]

            def _defl0_primal(x):  # V0 -> V0: M_0-orth projection off the constant
                return x - jnp.einsum("i,ij->j", _Mc0 @ x, _c0)

            def _defl0_dual(b):    # V0* -> V0*
                return b - jnp.einsum("i,ij->j", _c0 @ b, _Mc0)
        else:
            def _defl0_primal(x):
                return x
            _defl0_dual = _defl0_primal

        def _l0_smoother(b):       # const-deflated k=0 tensor precond ~ L_0^{-1}
            return _defl0_primal(apply_laplacian_preconditioner(
                seq, ops, _defl0_dual(b), 0, dirichlet=DIRICHLET, kind="tensor"))

        def _s_hat0(x):            # L_0 = apply_stiffness(.,0): exact, matrix-free
            return apply_stiffness(seq, ops, x, 0, dirichlet=DIRICHLET)

        _lmin0, _lmax0 = _lanczos_extremal_eigs_precond(
            _s_hat0, _l0_smoother, _n0, steps=30, seed=0, project=_defl0_primal)
        _lmin0 = max(_lmin0, _lmax0 * 1e-5)
        _kap0 = _lmax0 / max(_lmin0, 1e-300)
        # Inner-l0 accuracy: default TIED to the outer atom eps (eps=1e-2 inner is
        # overkill when the outer atom is rougher). Explicit l0_cheb_eps overrides.
        _l0_eps = (l0_cheb_eps if l0_cheb_eps is not None
                   else (atom_cheb_eps if atom_cheb_eps is not None else 1e-2))
        _deg0 = int(min(max(int(np.ceil(0.5 * np.sqrt(_kap0) * np.log(2.0 / _l0_eps))),
                            1), atom_cheb_max_degree))
        print(f"[atom] inner cheb_L0: kappa={_kap0:.3e} interval=[{_lmin0:.3e},"
              f"{_lmax0:.3e}] eps={_l0_eps} -> degree={_deg0}", flush=True)
        _cheb0 = make_chebyshev_upper(_s_hat0, _l0_smoother, _lmin0, _lmax0, _deg0)

        def _l0_inv_cdefl(r):      # near-exact L_0^{-1} (cheb-L_0), V0* -> V0
            return _defl0_primal(_cheb0(_defl0_dual(r)))

        _k1_applies = make_apply_routines(
            seq, ops, pa_mode="block_fd", grad_project=True, dirichlet_flag=DIRICHLET,
            true_g=true_g, l0_inv_custom=_l0_inv_cdefl)
        _s_hat1 = _k1_applies["a_matvec"]                       # ~ L_1 : V1 -> V1*
        # Use the FUSED k=1 smoother (2 L_0^{-1} solves instead of 4; algebraically
        # identical) -- halves the nested cheb-L_0 cost per smoother apply.
        _smoother_raw = (lambda r, _a=_k1_applies:
                         _a["projected_p_a_plus_p_b_fused_with_state"](_a["p_a_state"], r))
        _n1 = int(seq.n1_dbc if DIRICHLET else seq.n1)

        _H = jnp.asarray(_nullspace_vectors(ops, 1, DIRICHLET))  # (n_harm, n1), primal
        if _H.shape[0] > 0:
            _MH = jnp.stack(
                [apply_mass_matrix(seq, ops, _H[i], 1, dirichlet=DIRICHLET)
                 for i in range(_H.shape[0])], axis=0)           # M_1 h_i (dual reps)
            _nrm = jnp.sqrt(jnp.einsum("ij,ij->i", _H, _MH))     # ||h_i||_{M_1}
            _H = _H / _nrm[:, None]
            _MH = _MH / _nrm[:, None]

            def _defl_primal(x):   # V1 -> V1: M_1-orth projection onto h^perp
                return x - jnp.einsum("i,ij->j", _MH @ x, _H)

            def _defl_dual(b):     # V1* -> V1*: adjoint of _defl_primal
                return b - jnp.einsum("i,ij->j", _H @ b, _MH)

            def _smoother(b):
                return _defl_primal(_smoother_raw(_defl_dual(b)))
        else:
            def _defl_primal(x):
                return x

            _defl_dual = _defl_primal
            _smoother = _smoother_raw

        _lmin1, _lmax1 = _lanczos_extremal_eigs_precond(
            _s_hat1, _smoother, _n1, steps=50, seed=0, project=_defl_primal)
        _lmin1 = max(_lmin1, _lmax1 * 1e-5)  # low floor: keep the bulk small modes in-interval
        # Degree: read it off the Lanczos kappa via the Chebyshev bound
        # d ~ 0.5 sqrt(kappa) ln(2/eps) when atom_cheb_eps is given, else fixed.
        _kap1 = _lmax1 / max(_lmin1, 1e-300)
        if atom_cheb_eps is not None:
            _deg = int(np.ceil(0.5 * np.sqrt(_kap1) * np.log(2.0 / atom_cheb_eps)))
            _deg = int(min(max(_deg, 1), atom_cheb_max_degree))
        else:
            _deg = atom_cheb_degree
        print(f"[atom] cheb_tensor: kappa={_kap1:.3e} interval=[{_lmin1:.3e},"
              f"{_lmax1:.3e}] eps={atom_cheb_eps} -> degree={_deg}", flush=True)
        _cheb = make_chebyshev_upper(_s_hat1, _smoother, _lmin1, _lmax1, _deg)

        def k1_inv(r):  # deflated cheb-tensor near-exact L_1^+ (pseudo-inverse), V1* -> V1
            return _defl_primal(_cheb(_defl_dual(r)))
    elif atom == "block_fd":
        def k1_inv(r):  # (projected) pseudoinverse of curl-curl K_1, V1* -> V1
            # When project_atom is True, sandwich the inexact block_fd between
            # grad-complement projectors so its gradient-nullspace gain never
            # escapes (the exposed-outer-apply fix). When False, raw block_fd.
            if project_atom:
                return grad_primal_complement(k1_stiff_inv_raw(grad_dual_complement(r)))
            return k1_stiff_inv_raw(r)
    elif atom == "custom":
        # Diagnostic hook: inject an externally-built atom (e.g. a dense exact
        # L_1^{-1} from a Cholesky factor -- a FIXED LINEAR operator, so the
        # outer Krylov stays linear). Used by the dense exact-atom diagnostic to
        # test whether a kappa~1 inner inverse makes Pi_2 idempotent / the k=2
        # preconditioner converge.
        if k1_inv_custom is None:
            raise ValueError("atom='custom' requires k1_inv_custom")
        k1_inv = k1_inv_custom
    else:
        raise ValueError(
            f"Unsupported atom={atom!r}; expected 'block_fd', 'full_l1', or 'custom'")

    # --- P_B: curl-subspace correction, V2* -> V2 ---
    def p_b(r):
        y1 = apply_incidence_matrix(
            seq, ops, r, 1,
            dirichlet_in=DIRICHLET, dirichlet_out=DIRICHLET, transpose=True)    # V2* -> V1*  (G_1^T)
        y2 = k1_inv(y1)                                                         # V1* -> V1
        y3 = apply_mass_matrix(seq, ops, y2, 1, dirichlet=DIRICHLET)            # V1  -> V1*  (M_1)
        y4 = k1_inv(y3)                                                         # V1* -> V1
        return apply_incidence_matrix(
            seq, ops, y4, 1,
            dirichlet_in=DIRICHLET, dirichlet_out=DIRICHLET, transpose=False)   # V1  -> V2   (G_1)

    # --- div-div tensor block P_A, V2* -> V2 ---
    def p_a_raw(r):
        return apply_stiffness_tensor_preconditioner(
            seq, ops, r, 2, dirichlet=DIRICHLET)

    # CAPPED div-div P_A (block_fd mimic of the working k=1 P_A): PSD-pinv on the
    # surgery Schur -> BOUNDED on the curl null. Unlike the uncapped tensor P_A
    # (which leaks onto curls -> raw diverges -> needs the projection), this should
    # let RAW P_A + P_B converge with no projector and no near-exact atom.
    _p_a_capped_fn = _build_k2_block_fd_preconditioner(seq, ops, dirichlet=DIRICHLET)

    def p_a_capped(r):
        return _p_a_capped_fn(r)

    # Outer curl-complement projectors on V2 (M_2-orthogonal), built from K_1^{-1}.
    def curl_primal_complement(u):  # (I - Pi_2) u : V2 -> V2
        y0 = apply_mass_matrix(seq, ops, u, 2, dirichlet=DIRICHLET)             # V2  -> V2*  (M_2)
        y1 = apply_incidence_matrix(
            seq, ops, y0, 1,
            dirichlet_in=DIRICHLET, dirichlet_out=DIRICHLET, transpose=True)    # V2* -> V1*  (G_1^T)
        y2 = k1_inv(y1)                                                         # V1* -> V1
        y3 = apply_incidence_matrix(
            seq, ops, y2, 1,
            dirichlet_in=DIRICHLET, dirichlet_out=DIRICHLET, transpose=False)   # V1  -> V2   (G_1)
        return u - y3

    def curl_dual_complement(r):  # (I - Pi_2^*) r : V2* -> V2*
        y1 = apply_incidence_matrix(
            seq, ops, r, 1,
            dirichlet_in=DIRICHLET, dirichlet_out=DIRICHLET, transpose=True)    # V2* -> V1*  (G_1^T)
        y2 = k1_inv(y1)                                                         # V1* -> V1
        y3 = apply_incidence_matrix(
            seq, ops, y2, 1,
            dirichlet_in=DIRICHLET, dirichlet_out=DIRICHLET, transpose=False)   # V1  -> V2   (G_1)
        y4 = apply_mass_matrix(seq, ops, y3, 2, dirichlet=DIRICHLET)            # V2  -> V2*  (M_2)
        return r - y4

    def p_a_projected(r):
        return curl_primal_complement(p_a_raw(curl_dual_complement(r)))

    p_a = p_a_projected if grad_project else p_a_raw

    def raw_p_a_plus_p_b(r):
        return p_a_raw(r) + p_b(r)

    def _projected_plus_pb_fused(p_a_fn, r):
        # Algebraically IDENTICAL to
        #   curl_primal_complement(p_a_fn(curl_dual_complement(r))) + p_b(r)
        # but evaluated with 2 k1_inv solves instead of 4 (the projector and P_B
        # share the L_1 inverse). Two exact redundancies removed:
        #   1. curl_dual_complement's inner solve and p_b's inner solve are the
        #      SAME y = k1_inv(G_1^T r) -> computed once.
        #   2. curl_primal_complement's outer solve and p_b's outer solve are both
        #      G_1 k1_inv(.) -> arguments summed and solved once:
        #      result = pa + G_1 k1_inv(M_1 y - G_1^T M_2 pa).
        # Symmetry (hence MINRES iteration count) preserved by construction.
        g1t_r = apply_incidence_matrix(seq, ops, r, 1, dirichlet_in=DIRICHLET,
                                       dirichlet_out=DIRICHLET, transpose=True)    # G_1^T r
        y = k1_inv(g1t_r)                                                          # solve 1
        g1_y = apply_incidence_matrix(seq, ops, y, 1, dirichlet_in=DIRICHLET,
                                      dirichlet_out=DIRICHLET, transpose=False)    # G_1 y
        cdc = r - apply_mass_matrix(seq, ops, g1_y, 2, dirichlet=DIRICHLET)        # C^* r  (V2*)
        pa = p_a_fn(cdc)                                                           # P_A C^* r  (V2)
        m1_y = apply_mass_matrix(seq, ops, y, 1, dirichlet=DIRICHLET)             # M_1 y  (V1*)
        m2_pa = apply_mass_matrix(seq, ops, pa, 2, dirichlet=DIRICHLET)           # M_2 pa (V2*)
        g1t_m2_pa = apply_incidence_matrix(seq, ops, m2_pa, 1, dirichlet_in=DIRICHLET,
                                           dirichlet_out=DIRICHLET, transpose=True)  # G_1^T M_2 pa
        outer = apply_incidence_matrix(seq, ops, k1_inv(m1_y - g1t_m2_pa), 1,      # solve 2
                                       dirichlet_in=DIRICHLET, dirichlet_out=DIRICHLET,
                                       transpose=False)                            # G_1 k1_inv(...)
        return pa + outer

    def projected_p_a_plus_p_b(r):
        return _projected_plus_pb_fused(p_a_raw, r)

    # Route (b): CAPPED P_A, no projection. The capped div-div P_A is bounded on
    # curls (mimics k=1 block_fd), so raw should converge without the projector.
    def raw_p_a_capped_plus_p_b(r):
        return p_a_capped(r) + p_b(r)

    def projected_p_a_capped_plus_p_b(r):
        return _projected_plus_pb_fused(p_a_capped, r)

    # Production baseline: Schur-outer Jacobi diagonal for k=2 (stored multiply).
    # This is the WHOLE-space diagonal: 1/diag(S_2 + D_1 diag(M_1)^{-1} D_1^T),
    # i.e. it already includes the curl term -> overlaps/double-counts with P_B.
    schur_diaginv = _get_schur_diaginv(ops, 2, DIRICHLET, 'diag')

    def jacobi_diag(r):
        return schur_diaginv * r

    # Stiffness-only Jacobi: 1/diag(S_2) (div-div). S_2 annihilates curls
    # (ran G_1), so this is ZERO on the curl subspace and does NOT overlap P_B
    # -> a clean co-exact/curl split (the k=2 analog of k=1 jacobi(S)+P_B).
    k_diaginv = ops.dd2_diaginv_dbc if DIRICHLET else ops.dd2_diaginv
    if k_diaginv is None:
        size = int(seq.n2_dbc if DIRICHLET else seq.n2)
        k_diag = _diagonal_from_matvec(
            lambda x: apply_stiffness(seq, ops, x, 2, dirichlet=DIRICHLET), size)
        k_diaginv = _invert_diagonal(k_diag)

    def jacobi_stiff(r):
        return k_diaginv * r

    return {
        "a_matvec": a_matvec,
        "mass_matvec": mass_matvec,
        "stiffness_matvec": stiffness_matvec,
        "derivative_matvec": derivative_matvec,
        "derivative_t_matvec": derivative_t_matvec,
        "mass_lower_matvec": mass_lower_matvec,
        "lower_tensor_precond": lower_tensor_precond,
        "p_b": p_b,
        "p_a_raw": p_a_raw,
        "p_a_projected": p_a_projected,
        "p_a": p_a,
        "k1_inv": k1_inv,
        "curl_primal_complement": curl_primal_complement,
        "curl_dual_complement": curl_dual_complement,
        "jacobi_diag": (jacobi_diag if schur_diaginv is not None else None),
        "jacobi_stiff": jacobi_stiff,
        "raw_p_a_plus_p_b": raw_p_a_plus_p_b,
        "projected_p_a_plus_p_b": projected_p_a_plus_p_b,
        "p_a_capped": p_a_capped,
        "raw_p_a_capped_plus_p_b": raw_p_a_capped_plus_p_b,
        "projected_p_a_capped_plus_p_b": projected_p_a_capped_plus_p_b,
    }


def make_apply_routines_k3(seq: DeRhamSequence, ops):
    """k=3 Hodge-Laplacian preconditioner via the auxiliary-space transfer to k=0.

    L_3 = D_2 M_2^{-1} D_2^T has NO stiffness term, so it is solved as a saddle
    with a zero upper block (MINRES). The exact M_2^{-1} lives in the lower
    block; it is never forward-applied.

    Two upper-block preconditioners (approximations of L_3^{-1}) are compared:
      - ``jacobi``: 1/diag(L_3) (stored Schur diagonal probe).
      - ``transfer`` P_3 = T_{0->3} L_0^{-1} T_{3->0}: map the V3 residual
        SIDEWAYS to the dual scalar space k=0 (NOT via the derivative, which
        would kill subspaces) using the Galerkin transfer T = M^{-1} C, with the
        cross-mass C = projection block ("primal->dual") and M^{-1} done by the
        (excellent) tensor mass preconditioner ("dual->primal"); invert with the
        near-exact k=0 Hodge preconditioner; map back.

    BC SWAP (key): we run the WHOLE k=3 problem with dirichlet=False, so its dual
    k=0 carries dbc=True -- which is exactly the BC variant for which the working
    tensor k=0 Hodge preconditioner is built. (Dual to k=3-no-dbc is k=0-dbc.)
    So ``k3_dbc=False`` for the k=3 saddle, ``aux_dbc=True`` for the auxiliary
    k=0 inverse and the k=0 ends of both transfers.

    P_3 = T_{0->3} L_0^{-1} (T_{0->3})^* is SPD (T_{3->0} is the adjoint of
    T_{0->3}); all applies are matrix-free / single tensor applies (no inner
    Krylov).
    """
    k3_dbc = False    # run the k=3 problem with free BCs ...
    aux_dbc = True    # ... so the dual k=0 is dbc -> working tensor L_0^{-1}

    # --- forward saddle matvecs for L_3 (upper block S_3 = 0) ---
    def a_matvec(v):
        return apply_laplacian_approx(seq, ops, v, 3, dirichlet=k3_dbc)

    def mass_matvec(v):  # M_3 : V3 -> V3*
        return apply_mass_matrix(seq, ops, v, 3, dirichlet=k3_dbc)

    def stiffness_matvec(v):  # S_3 = 0
        return apply_stiffness(seq, ops, v, 3, dirichlet=k3_dbc)

    def derivative_matvec(sigma):  # D_2 : V2 -> V3
        return apply_derivative_matrix(
            seq, ops, sigma, 2,
            dirichlet_in=k3_dbc, dirichlet_out=k3_dbc, transpose=False)

    def derivative_t_matvec(u):  # D_2^T : V3 -> V2
        return apply_derivative_matrix(
            seq, ops, u, 2,
            dirichlet_in=k3_dbc, dirichlet_out=k3_dbc, transpose=True)

    def mass_lower_matvec(sigma):  # M_2 : V2 -> V2*
        return apply_mass_matrix(seq, ops, sigma, 2, dirichlet=k3_dbc)

    def lower_tensor_precond(rhs):  # M_2^{-1} (tensor) : V2* -> V2
        return apply_mass_matrix_preconditioner(
            seq, ops, rhs, 2, dirichlet=k3_dbc, kind="tensor")

    # --- transfer preconditioner P_3 = T_{0->3} L_0^{-1} T_{3->0} ---
    def mass3_inv(r):  # M_3^{-1} via tensor mass precond, V3* -> V3 (no-dbc)
        return apply_mass_matrix_preconditioner(
            seq, ops, r, 3, dirichlet=k3_dbc, kind="tensor")

    def l0_inv(x):  # k=0 Hodge tensor precond, DBC (working), V0* -> V0
        return apply_laplacian_preconditioner(
            seq, ops, x, 0, dirichlet=aux_dbc, kind="tensor")

    # NOTE: apply_projection_matrix(v, a, b) maps space b -> space a (the pair is
    # (output, input); dirichlet_in is the INPUT-space BC, dirichlet_out the
    # OUTPUT-space BC). So V3->V0 uses (0, 3) and V0->V3 uses (3, 0).
    def transfer_3_to_0(r):  # T^* = C^T M_3^{-1} : V3*(free) -> V0*(dbc)
        return apply_projection_matrix(
            seq, ops, mass3_inv(r), 0, 3,
            dirichlet_in=k3_dbc, dirichlet_out=aux_dbc)

    def transfer_0_to_3(x):  # T = M_3^{-1} C : V0(dbc) -> V3(free)
        return mass3_inv(apply_projection_matrix(
            seq, ops, x, 3, 0,
            dirichlet_in=aux_dbc, dirichlet_out=k3_dbc))

    def p3_transfer(r):
        return transfer_0_to_3(l0_inv(transfer_3_to_0(r)))

    # --- jacobi baseline (whole-space diag(L_3)) ---
    schur_diaginv = _get_schur_diaginv(ops, 3, k3_dbc, 'diag')

    def jacobi_diag(r):
        return schur_diaginv * r

    return {
        "a_matvec": a_matvec,
        "mass_matvec": mass_matvec,
        "stiffness_matvec": stiffness_matvec,
        "derivative_matvec": derivative_matvec,
        "derivative_t_matvec": derivative_t_matvec,
        "mass_lower_matvec": mass_lower_matvec,
        "lower_tensor_precond": lower_tensor_precond,
        "jacobi_diag": (jacobi_diag if schur_diaginv is not None else None),
        "p3_transfer": p3_transfer,
    }


def build_rhs_batch(seq, ops, a_matvec, *, n_rhs: int, seed: int, rhs_kind: str):
    n = int(seq.n1_dbc)
    n0 = int(seq.n0_dbc)
    keys = jax.random.split(jax.random.PRNGKey(seed), n_rhs)
    rhs = []
    for key in keys:
        if rhs_kind == "random":
            x_true = jax.random.normal(key, (n,), dtype=jnp.float64)
        elif rhs_kind == "gradient":
            phi = jax.random.normal(key, (n0,), dtype=jnp.float64)
            x_true = apply_incidence_matrix(
                seq,
                ops,
                phi,
                0,
                dirichlet_in=DIRICHLET,
                dirichlet_out=DIRICHLET,
                transpose=False,
            )
        else:
            raise ValueError(
                f"Unsupported rhs_kind {rhs_kind!r}; expected 'random' or 'gradient'"
            )
        rhs.append(a_matvec(x_true))  # guarantees b in range(A)
    return jnp.stack(rhs, axis=0)


def make_solve(
        a_matvec,
        mass_matvec,
        precond,
        *,
        tol: float,
        maxiter: int,
        precond_state=None):
    if precond_state is None:
        @jax.jit
        def solve(rhs):
            x, info = solve_singular_cg(
                a_matvec,
                rhs,
                mass_matvec=mass_matvec,
                precond_matvec=precond,
                vs=[],  # k=1 DBC: no harmonics
                tol=tol,
                maxiter=maxiter,
            )
            r = a_matvec(x) - rhs
            r_norm = jnp.linalg.norm(r)
            b_norm = jnp.linalg.norm(rhs)
            rel = r_norm / jnp.where(b_norm > 0.0, b_norm, 1.0)
            return x, info, rel

        return solve

    @jax.jit
    def solve(rhs, state):
        def stateful_precond(vec):
            return precond(state, vec)

        x, info = solve_singular_cg(
            a_matvec,
            rhs,
            mass_matvec=mass_matvec,
            precond_matvec=stateful_precond,
            vs=[],  # k=1 DBC: no harmonics
            tol=tol,
            maxiter=maxiter,
        )
        r = a_matvec(x) - rhs
        r_norm = jnp.linalg.norm(r)
        b_norm = jnp.linalg.norm(rhs)
        rel = r_norm / jnp.where(b_norm > 0.0, b_norm, 1.0)
        return x, info, rel

    return solve


def make_saddle_solve(
        stiffness_matvec,
        derivative_matvec,
        derivative_t_matvec,
        mass_lower_matvec,
        precond_upper,
        precond_lower,
        *,
        n_upper: int,
        n_lower: int,
        tol: float,
        maxiter: int,
        precond_upper_state=None,
        vs_upper=None,
        mass_upper_matvec=None):
    if precond_upper_state is None:
        @jax.jit
        def solve(rhs_upper):
            u, sigma, info = solve_saddle_point_minres(
                stiffness_matvec,
                derivative_matvec,
                derivative_t_matvec,
                mass_lower_matvec,
                rhs_upper,
                n_upper,
                n_lower,
                precond_upper=precond_upper,
                precond_lower=precond_lower,
                vs_upper=vs_upper,
                mass_upper_matvec=mass_upper_matvec,
                tol=tol,
                maxiter=maxiter,
            )
            r_upper = stiffness_matvec(u) + derivative_matvec(sigma) - rhs_upper
            r_lower = derivative_t_matvec(u) - mass_lower_matvec(sigma)
            b_norm = jnp.linalg.norm(rhs_upper)
            rel = jnp.sqrt(jnp.dot(r_upper, r_upper) + jnp.dot(r_lower, r_lower)) / jnp.where(
                b_norm > 0.0, b_norm, 1.0
            )
            return u, sigma, info, rel

        return solve

    @jax.jit
    def solve(rhs_upper, state):
        def stateful_upper(vec):
            return precond_upper(state, vec)

        u, sigma, info = solve_saddle_point_minres(
            stiffness_matvec,
            derivative_matvec,
            derivative_t_matvec,
            mass_lower_matvec,
            rhs_upper,
            n_upper,
            n_lower,
            precond_upper=stateful_upper,
            precond_lower=precond_lower,
            vs_upper=vs_upper,
            mass_upper_matvec=mass_upper_matvec,
            tol=tol,
            maxiter=maxiter,
        )
        r_upper = stiffness_matvec(u) + derivative_matvec(sigma) - rhs_upper
        r_lower = derivative_t_matvec(u) - mass_lower_matvec(sigma)
        b_norm = jnp.linalg.norm(rhs_upper)
        rel = jnp.sqrt(jnp.dot(r_upper, r_upper) + jnp.dot(r_lower, r_lower)) / jnp.where(
            b_norm > 0.0, b_norm, 1.0
        )
        return u, sigma, info, rel

    return solve


def time_solve(solve, rhs_batch, *, solve_state=None, rel_tol: float = 1e-10):
    if solve_state is None:
        x, info, rel = solve(rhs_batch[0])
    else:
        x, info, rel = solve(rhs_batch[0], solve_state)
    jax.block_until_ready((x, info, rel))

    iters: list[int] = []
    infos: list[int] = []
    times_ms: list[float] = []
    residuals: list[float] = []
    for rhs in rhs_batch:
        t0 = time.perf_counter()
        if solve_state is None:
            x, info, rel = solve(rhs)
        else:
            x, info, rel = solve(rhs, solve_state)
        jax.block_until_ready((x, info, rel))
        times_ms.append((time.perf_counter() - t0) * 1e3)
        info_i = int(info)
        infos.append(info_i)
        iters.append(abs(info_i))
        residuals.append(float(rel))

    n_fail = sum(
        1
        for info_i, rel_i in zip(infos, residuals)
        if (info_i > 0) or (rel_i > rel_tol)
    )

    return {
        "avg_iters": float(jnp.mean(jnp.asarray(iters))),
        "max_iters": int(max(iters)),
        "min_iters": int(min(iters)),
        "std_iters": float(jnp.std(jnp.asarray(iters, dtype=jnp.float64))),
        "avg_ms": float(jnp.mean(jnp.asarray(times_ms))),
        "std_ms": float(jnp.std(jnp.asarray(times_ms))),
        "max_residual": float(max(residuals)),
        "n_fail": int(n_fail),
        "n_total": int(len(infos)),
    }


def _lanczos_extremal_eigs_precond(a_matvec, precond, n, *, steps: int = 30, seed: int = 0,
                                   project=None):
    """Estimate (lambda_min, lambda_max) of ``P @ A`` for SPD ``A`` (``a_matvec``)
    and SPD preconditioner ``P`` (``precond``), via Lanczos in the A-inner product.

    ``B = P A`` is self-adjoint w.r.t. <u,v>_A = u^T A v (A,P symmetric), so its
    real eigenvalues are the Ritz values of an A-inner-product Lanczos. Used to
    size the Chebyshev interval for a NON-diagonal (tensor) smoother, where the
    standard symmetric-Lanczos form D^{-1/2} A D^{-1/2} is unavailable. Costs ~2
    A-applies + 1 P-apply per step (setup-time only).

    ``project`` (optional): when ``A`` is only SEMI-definite (e.g. free-BC L_1 has
    the harmonic null h with ||h||_A = 0), the A-inner-product degenerates and the
    Lanczos blows up (||w||_A -> 0 -> NaN). Pass the M-orthogonal projector onto
    the complement of ker(A); applied to the start vector AND every basis vector,
    it confines the recursion to the subspace where A is positive definite. dbc ->
    pass None (no null).
    """
    if project is None:
        def project(v):
            return v
    key = jax.random.PRNGKey(seed)
    q = project(jax.random.normal(key, (n,), dtype=jnp.float64))
    Aq = a_matvec(q)
    qAq = float(jnp.dot(q, Aq))
    if not np.isfinite(qAq) or qAq <= 1e-30:
        return 0.0, 1.0  # degenerate; caller floors the interval anyway
    nrm = float(np.sqrt(qAq))
    q = q / nrm
    Aq = Aq / nrm
    q_prev = jnp.zeros_like(q)
    beta = 0.0
    alphas: list[float] = []
    betas: list[float] = []
    for _ in range(steps):
        w = precond(Aq)                      # B q = P (A q)
        alpha = float(jnp.dot(w, Aq))        # <Bq, q>_A = w^T A q
        w = project(w - alpha * q - beta * q_prev)  # keep in the A-PD subspace
        alphas.append(alpha)
        Aw = a_matvec(w)
        wAw = float(jnp.dot(w, Aw))          # ||w||_A^2
        if not np.isfinite(wAw) or wAw < 1e-12:
            break
        beta = float(np.sqrt(wAw))
        betas.append(beta)
        q_prev, q = q, w / beta
        Aq = Aw / beta
    if not alphas:
        return 0.0, 1.0
    T = np.diag(np.asarray(alphas))
    off = betas[:len(alphas) - 1]
    if off:
        b = np.asarray(off)
        T = T + np.diag(b, 1) + np.diag(b, -1)
    eigs = np.linalg.eigvalsh(T)
    return float(eigs[0]), float(eigs[-1])


def _lanczos_extremal_eigs(matvec, n, *, steps: int = 24, seed: int = 0):
    """Estimate (lambda_min, lambda_max) of a symmetric operator via Lanczos.

    ``matvec`` must be the SYMMETRIC operator whose spectrum we want (for a
    jacobi-preconditioned Schur this is M = D^{-1/2} S-hat D^{-1/2}). Returns the
    extremal Ritz values; cheap setup-time estimate (a handful of applies).
    """
    key = jax.random.PRNGKey(seed)
    v = jax.random.normal(key, (n,), dtype=jnp.float64)
    v = v / jnp.linalg.norm(v)
    v_prev = jnp.zeros_like(v)
    beta = 0.0
    alphas: list[float] = []
    betas: list[float] = []
    for _ in range(steps):
        w = matvec(v)
        alpha = float(jnp.dot(v, w))
        w = w - alpha * v - beta * v_prev
        alphas.append(alpha)
        beta = float(jnp.linalg.norm(w))
        if beta < 1e-10:
            break
        betas.append(beta)
        v_prev, v = v, w / beta
    T = np.diag(np.asarray(alphas))
    # The m x m tridiagonal has m-1 off-diagonals; the final beta connects to an
    # unused (m+1)-th vector, so trim.
    off = betas[:len(alphas) - 1]
    if off:
        b = np.asarray(off)
        T = T + np.diag(b, 1) + np.diag(b, -1)
    eigs = np.linalg.eigvalsh(T)
    return float(eigs[0]), float(eigs[-1])


def make_chebyshev_upper(s_hat, jac, lmin: float, lmax: float, degree: int):
    """Degree-``degree`` Chebyshev iteration approximating S-hat^{-1}, inner
    (diagonal) preconditioner ``jac`` ~ diag(S-hat)^{-1}, spectrum of jac*S-hat
    in [lmin, lmax]. Fixed degree -> a linear, symmetric operator (valid MINRES
    upper preconditioner; no Krylov-in-Krylov). ``s_hat`` is the APPROXIMATE
    Schur (M replaced by its preconditioner)."""
    theta = 0.5 * (lmax + lmin)
    delta = 0.5 * (lmax - lmin)
    # Degenerate interval (lmin ~ lmax, or a collapsed Lanczos estimate) -> the
    # recurrence divides by delta -> NaN. Fall back to a single smoother apply.
    if not np.isfinite(delta) or delta <= 1e-300 or degree < 1:
        def apply(b):
            return jac(b)
        return apply
    sigma1 = theta / delta

    def apply(b):
        d_vec = jac(b) / theta
        x = d_vec
        rho_prev = 1.0 / sigma1
        for _ in range(degree - 1):
            r = b - s_hat(x)
            rho = 1.0 / (2.0 * sigma1 - rho_prev)
            d_vec = rho * rho_prev * d_vec + (2.0 * rho / delta) * jac(r)
            x = x + d_vec
            rho_prev = rho
        return x

    return apply


def make_richardson_upper(s_hat, jac, lmin: float, lmax: float, degree: int):
    """Degree-``degree`` (fixed-omega) Richardson iteration approximating
    S-hat^{-1} with inner preconditioner ``jac``; omega = 2/(lmin+lmax)
    (optimal for the symmetric case). Linear, symmetric operator."""
    omega = 2.0 / (lmin + lmax)

    def apply(b):
        x = jnp.zeros_like(b)
        for _ in range(degree):
            x = x + omega * jac(b - s_hat(x))
        return x

    return apply


def time_saddle_solve(solve, rhs_batch, *, solve_state=None, rel_tol: float = 1e-10):
    if solve_state is None:
        u, sigma, info, rel = solve(rhs_batch[0])
    else:
        u, sigma, info, rel = solve(rhs_batch[0], solve_state)
    jax.block_until_ready((u, sigma, info, rel))

    iters: list[int] = []
    infos: list[int] = []
    times_ms: list[float] = []
    residuals: list[float] = []
    for rhs in rhs_batch:
        t0 = time.perf_counter()
        if solve_state is None:
            u, sigma, info, rel = solve(rhs)
        else:
            u, sigma, info, rel = solve(rhs, solve_state)
        jax.block_until_ready((u, sigma, info, rel))
        times_ms.append((time.perf_counter() - t0) * 1e3)
        info_i = int(info)
        infos.append(info_i)
        iters.append(abs(info_i))
        residuals.append(float(rel))

    n_fail = sum(
        1
        for info_i, rel_i in zip(infos, residuals)
        if (info_i > 0) or (rel_i > rel_tol)
    )

    return {
        "avg_iters": float(jnp.mean(jnp.asarray(iters))),
        "max_iters": int(max(iters)),
        "min_iters": int(min(iters)),
        "avg_ms": float(jnp.mean(jnp.asarray(times_ms))),
        "max_residual": float(max(residuals)),
        "n_fail": int(n_fail),
        "n_total": int(len(infos)),
    }


def make_scaled_saddle_solve(
        stiffness_matvec,
        derivative_matvec,
        derivative_t_matvec,
        mass_lower_matvec,
        scaled_precond_upper,
        precond_lower,
        *,
        n_upper: int,
        n_lower: int,
        tol: float,
        maxiter: int,
        scaled_precond_state=None):
    if scaled_precond_state is None:
        @jax.jit
        def solve(rhs_upper, jacobi_scale):
            def precond_upper(vec):
                return scaled_precond_upper(vec, jacobi_scale)

            u, sigma, info = solve_saddle_point_minres(
                stiffness_matvec,
                derivative_matvec,
                derivative_t_matvec,
                mass_lower_matvec,
                rhs_upper,
                n_upper,
                n_lower,
                precond_upper=precond_upper,
                precond_lower=precond_lower,
                tol=tol,
                maxiter=maxiter,
            )
            r_upper = stiffness_matvec(u) + derivative_matvec(sigma) - rhs_upper
            r_lower = derivative_t_matvec(u) - mass_lower_matvec(sigma)
            b_norm = jnp.linalg.norm(rhs_upper)
            rel = jnp.sqrt(jnp.dot(r_upper, r_upper) + jnp.dot(r_lower, r_lower)) / jnp.where(
                b_norm > 0.0, b_norm, 1.0
            )
            return u, sigma, info, rel

        return solve

    @jax.jit
    def solve(rhs_upper, jacobi_scale, state):
        def precond_upper(vec):
            return scaled_precond_upper(state, vec, jacobi_scale)

        u, sigma, info = solve_saddle_point_minres(
            stiffness_matvec,
            derivative_matvec,
            derivative_t_matvec,
            mass_lower_matvec,
            rhs_upper,
            n_upper,
            n_lower,
            precond_upper=precond_upper,
            precond_lower=precond_lower,
            tol=tol,
            maxiter=maxiter,
        )
        r_upper = stiffness_matvec(u) + derivative_matvec(sigma) - rhs_upper
        r_lower = derivative_t_matvec(u) - mass_lower_matvec(sigma)
        b_norm = jnp.linalg.norm(rhs_upper)
        rel = jnp.sqrt(jnp.dot(r_upper, r_upper) + jnp.dot(r_lower, r_lower)) / jnp.where(
            b_norm > 0.0, b_norm, 1.0
        )
        return u, sigma, info, rel

    return solve


def time_scaled_saddle_solve(
    solve,
    rhs_batch,
    jacobi_scale: float,
    *,
    solve_state=None,
    rel_tol: float = 1e-10):
    alpha = jnp.asarray(jacobi_scale, dtype=jnp.float64)
    if solve_state is None:
        u, sigma, info, rel = solve(rhs_batch[0], alpha)
    else:
        u, sigma, info, rel = solve(rhs_batch[0], alpha, solve_state)
    jax.block_until_ready((u, sigma, info, rel))

    iters: list[int] = []
    infos: list[int] = []
    times_ms: list[float] = []
    residuals: list[float] = []
    for rhs in rhs_batch:
        t0 = time.perf_counter()
        if solve_state is None:
            u, sigma, info, rel = solve(rhs, alpha)
        else:
            u, sigma, info, rel = solve(rhs, alpha, solve_state)
        jax.block_until_ready((u, sigma, info, rel))
        times_ms.append((time.perf_counter() - t0) * 1e3)
        info_i = int(info)
        infos.append(info_i)
        iters.append(abs(info_i))
        residuals.append(float(rel))

    n_fail = sum(
        1
        for info_i, rel_i in zip(infos, residuals)
        if (info_i > 0) or (rel_i > rel_tol)
    )

    return {
        "avg_iters": float(jnp.mean(jnp.asarray(iters))),
        "max_iters": int(max(iters)),
        "min_iters": int(min(iters)),
        "avg_ms": float(jnp.mean(jnp.asarray(times_ms))),
        "max_residual": float(max(residuals)),
        "n_fail": int(n_fail),
        "n_total": int(len(infos)),
    }


def time_pa_profile(profile_apply, rhs_batch):
    rows = [profile_apply(rhs) for rhs in rhs_batch]
    return {
        "bulk1_ms": float(jnp.mean(jnp.asarray([r["bulk1_ms"] for r in rows]))),
        "schur_ms": float(jnp.mean(jnp.asarray([r["schur_ms"] for r in rows]))),
        "bulk2_ms": float(jnp.mean(jnp.asarray([r["bulk2_ms"] for r in rows]))),
        "total_ms": float(jnp.mean(jnp.asarray([r["total_ms"] for r in rows]))),
    }


def diagnose_pa_operator(pa_matvec, *, n: int, seed: int, tol: float):
    """Return symmetry/positivity diagnostics for P_A.

    Includes random bilinear probes and exhaustive unit-vector probing via
    dense matrix assembly A[:,i] = P_A(e_i).
    """
    key = jax.random.PRNGKey(seed)
    key_x, key_y, _ = jax.random.split(key, 3)

    x = jax.random.normal(key_x, (n,), dtype=jnp.float64)
    y = jax.random.normal(key_y, (n,), dtype=jnp.float64)
    p_x = pa_matvec(x)
    p_y = pa_matvec(y)
    jax.block_until_ready((p_x, p_y))

    bxy = float(jnp.dot(x, p_y))
    byx = float(jnp.dot(y, p_x))
    bilinear_sym_abs = abs(bxy - byx)
    bilinear_sym_rel = bilinear_sym_abs / max(1.0, abs(bxy), abs(byx))
    rayleigh_x = float(jnp.dot(x, p_x))

    # Exhaustive probe on unit vectors: build dense matrix column-by-column.
    eye = jnp.eye(n, dtype=jnp.float64)
    a_dense = jax.vmap(pa_matvec, in_axes=1, out_axes=1)(eye)
    jax.block_until_ready(a_dense)

    skew = a_dense - a_dense.T
    skew_inf = float(jnp.max(jnp.abs(skew)))
    a_inf = float(jnp.max(jnp.abs(a_dense)))
    sym_inf_rel = skew_inf / max(1.0, a_inf)

    diag_vals = jnp.diag(a_dense)
    min_diag = float(jnp.min(diag_vals))
    max_diag = float(jnp.max(diag_vals))

    eigvals = jnp.linalg.eigvalsh(0.5 * (a_dense + a_dense.T))
    min_eig = float(jnp.min(eigvals))
    max_eig = float(jnp.max(eigvals))
    max_abs_eig = float(jnp.max(jnp.abs(eigvals)))

    return {
        "bilinear_sym_abs": bilinear_sym_abs,
        "bilinear_sym_rel": bilinear_sym_rel,
        "rayleigh_x": rayleigh_x,
        "dense_sym_inf_abs": skew_inf,
        "dense_sym_inf_rel": sym_inf_rel,
        "unit_min_diag": min_diag,
        "unit_max_diag": max_diag,
        "sym_part_min_eig": min_eig,
        "sym_part_max_eig": max_eig,
        "sym_part_max_abs_eig": max_abs_eig,
        "is_symmetric": sym_inf_rel <= tol,
        "is_psd": min_eig >= -tol,
    }


def diagnose_grad_subspace_overlap(
        p_a_raw,
        p_a_projected,
        p_b,
        project_primal_complement,
        project_primal_complement_exact,
        mass1_matvec,
        *,
        n: int,
        seed: int,
        n_probe: int = 8):
    """Quantify how much P_A leaks into the gradient subspace that P_B owns.

    For random dual residuals r we measure, in the M_1 inner product:

    * ``grad_frac`` -- fraction of the M_1 energy of ``P_A r`` that lives in
      the gradient (curl-free) subspace. This is the part P_B is responsible
      for; a large value means P_A and P_B fight over the same modes. The
      fraction is reported using both the cheap tensor projector (same one the
      preconditioner sandwich uses) and an exact matrix-free L_0 projector. A
      true M_1-orthogonal projector gives ``grad_frac`` in [0, 1]; values above
      1 from the tensor projector are an artifact of the approximate L_0^{-1},
      not genuine energy, which the exact column isolates.
    * ``pa_pb_cos`` -- the M_1 cosine between ``P_A r`` and ``P_B r``. This is
      projector-independent and therefore trustworthy. Large magnitude means
      the two blocks produce aligned corrections (additive double-counting);
      near zero means they target complementary subspaces.

    Both quantities are reported for the raw and the gradient-projected P_A so
    a single run shows the effect of the projection.
    """
    keys = jax.random.split(jax.random.PRNGKey(seed), n_probe)
    tiny = 1e-300

    def m_norm_sq(u):
        return jnp.dot(u, mass1_matvec(u))

    def m_dot(u, v):
        return jnp.dot(u, mass1_matvec(v))

    def grad_fraction(u, projector):
        # gradient part = u - (I - Pi) u = Pi u
        u_curl = projector(u)
        u_grad = u - u_curl
        return m_norm_sq(u_grad) / jnp.maximum(m_norm_sq(u), tiny)

    def pa_pb_cosine(ua, ub):
        denom = jnp.sqrt(jnp.maximum(m_norm_sq(ua) * m_norm_sq(ub), tiny))
        return m_dot(ua, ub) / denom

    raw_grad: list[float] = []
    raw_grad_exact: list[float] = []
    proj_grad_exact: list[float] = []
    raw_cos: list[float] = []
    proj_cos: list[float] = []
    for key in keys:
        r = jax.random.normal(key, (n,), dtype=jnp.float64)
        ua_raw = p_a_raw(r)
        ua_proj = p_a_projected(r)
        ub = p_b(r)
        jax.block_until_ready((ua_raw, ua_proj, ub))
        raw_grad.append(float(grad_fraction(ua_raw, project_primal_complement)))
        raw_grad_exact.append(
            float(grad_fraction(ua_raw, project_primal_complement_exact)))
        proj_grad_exact.append(
            float(grad_fraction(ua_proj, project_primal_complement_exact)))
        raw_cos.append(float(pa_pb_cosine(ua_raw, ub)))
        proj_cos.append(float(pa_pb_cosine(ua_proj, ub)))

    arr = jnp.asarray
    return {
        "n_probe": n_probe,
        "raw_grad_frac_tensor_mean": float(jnp.mean(arr(raw_grad))),
        "raw_grad_frac_tensor_max": float(jnp.max(arr(raw_grad))),
        "raw_grad_frac_exact_mean": float(jnp.mean(arr(raw_grad_exact))),
        "raw_grad_frac_exact_max": float(jnp.max(arr(raw_grad_exact))),
        "proj_grad_frac_exact_mean": float(jnp.mean(arr(proj_grad_exact))),
        "proj_grad_frac_exact_max": float(jnp.max(arr(proj_grad_exact))),
        "raw_pa_pb_cos_mean": float(jnp.mean(jnp.abs(arr(raw_cos)))),
        "raw_pa_pb_cos_max": float(jnp.max(jnp.abs(arr(raw_cos)))),
        "proj_pa_pb_cos_mean": float(jnp.mean(jnp.abs(arr(proj_cos)))),
        "proj_pa_pb_cos_max": float(jnp.max(jnp.abs(arr(proj_cos)))),
    }


def _parse_pa_spectrum_modes(option: str) -> list[str]:
    value = option.strip().lower()
    if value == "none":
        return []
    if value == "all":
        return ["active", "jacobi", "pinv"]
    modes = [m.strip().lower() for m in option.split(",") if m.strip()]
    valid = {"active", "jacobi", "pinv"}
    unknown = [m for m in modes if m not in valid]
    if unknown:
        raise ValueError(
            "Unknown --pa-stiffness-spectrum mode(s): "
            + ", ".join(unknown)
            + ". Valid choices: none, all, active, jacobi, pinv"
        )
    deduped: list[str] = []
    for mode in modes:
        if mode not in deduped:
            deduped.append(mode)
    return deduped


def diagnose_pa_times_stiffness_spectrum(
        seq: DeRhamSequence,
        ops,
        stiffness_matvec,
        *,
    active_pa_apply,
    active_pa_name: str,
    mass1_matvec,
    project_primal_complement_exact,
        modes: list[str],
        k_diaginv,
        pinv_rcond: float,
    curl_rel_cutoff: float = 1e-8,
    tail_k: int = 4):
    """Build dense ``S`` and report spectrum of ``P_A S`` for selected ``P_A`` variants."""
    if not modes:
        return None

    n = int(seq.n1_dbc if DIRICHLET else seq.n1)
    eye = jnp.eye(n, dtype=jnp.float64)
    s_dense = jax.vmap(stiffness_matvec, in_axes=1, out_axes=1)(eye)
    s_dense = 0.5 * (s_dense + s_dense.T)
    jax.block_until_ready(s_dense)
    tiny = 1e-300

    def m_norm_sq(u):
        return jnp.dot(u, mass1_matvec(u))

    def _summarize(pa_name: str, pa_apply):
        pa_dense = jax.vmap(pa_apply, in_axes=1, out_axes=1)(eye)
        pa_s = pa_dense @ s_dense
        eigvals = jnp.linalg.eigvals(pa_s)
        eigvals_right, eigvecs_right = jnp.linalg.eig(pa_s)
        real = jnp.real(eigvals)
        imag = jnp.imag(eigvals)
        abs_eigs = jnp.abs(eigvals)
        min_abs = float(jnp.min(abs_eigs))
        max_abs = float(jnp.max(abs_eigs))
        cond_abs = max_abs / max(min_abs, 1e-30)
        # Curl-subspace-restricted conditioning: discard the gradient nullspace
        # of S = K_1 (eigenvalues that are ~0) using a relative cutoff, then
        # report lambda_max / lambda_min over the surviving (curl) modes. This
        # is the metric that reflects how well P_A preconditions K_1 on the
        # subspace it owns; unlike cond_abs it is not polluted by nullspace
        # zeros that land at machine precision.
        cutoff = curl_rel_cutoff * max(max_abs, 1e-30)
        nonzero = abs_eigs[abs_eigs > cutoff]
        n_nonzero = int(nonzero.shape[0])
        if n_nonzero > 0:
            min_nonzero = float(jnp.min(nonzero))
            cond_curl = max_abs / max(min_nonzero, 1e-30)
            pct_levels = jnp.asarray([5.0, 25.0, 50.0, 75.0, 95.0], dtype=jnp.float64)
            pct_vals = jnp.percentile(nonzero, pct_levels)
            curl_percentiles = {
                "p05": float(pct_vals[0]),
                "p25": float(pct_vals[1]),
                "p50": float(pct_vals[2]),
                "p75": float(pct_vals[3]),
                "p95": float(pct_vals[4]),
            }
            tail_counts = {
                "gt2": int(jnp.sum(nonzero > 2.0)),
                "gt5": int(jnp.sum(nonzero > 5.0)),
                "gt10": int(jnp.sum(nonzero > 10.0)),
                "gt20": int(jnp.sum(nonzero > 20.0)),
            }
        else:
            min_nonzero = float("nan")
            cond_curl = float("nan")
            curl_percentiles = {
                "p05": float("nan"),
                "p25": float("nan"),
                "p50": float("nan"),
                "p75": float("nan"),
                "p95": float("nan"),
            }
            tail_counts = {
                "gt2": 0,
                "gt5": 0,
                "gt10": 0,
                "gt20": 0,
            }

        if tail_k > 0 and abs_eigs.shape[0] > 0:
            tail_idx = jnp.argsort(abs_eigs)[-tail_k:][::-1]
            print(f"[diag] {pa_name} top-{int(min(tail_k, abs_eigs.shape[0]))} |eig| outliers:")
            for idx in tail_idx.tolist():
                val = eigvals_right[idx]
                vec = jnp.real(eigvecs_right[:, idx])
                vec_norm = jnp.linalg.norm(vec)
                if float(vec_norm) <= 0.0:
                    vec = jnp.imag(eigvecs_right[:, idx])
                    vec_norm = jnp.linalg.norm(vec)
                vec = vec / jnp.where(vec_norm > 0, vec_norm, 1.0)
                grad_part = vec - project_primal_complement_exact(vec)
                grad_frac = float(m_norm_sq(grad_part) / jnp.maximum(m_norm_sq(vec), tiny))
                top_idx = jnp.argsort(jnp.abs(vec))[-5:][::-1]
                top_idx_str = ",".join(str(int(i)) for i in top_idx.tolist())
                top_val_str = ",".join(f"{float(vec[i]):+.2e}" for i in top_idx.tolist())
                print(
                    f"[diag]   idx={int(idx):>4d} eig={float(jnp.real(val)):+.2e}"
                    f"+{float(jnp.imag(val)):.2e}j |eig|={float(jnp.abs(val)):.2e} "
                    f"grad_frac={grad_frac:.2e} top_idx=[{top_idx_str}] top_val=[{top_val_str}]"
                )
        return {
            "name": pa_name,
            "real_min": float(jnp.min(real)),
            "real_max": float(jnp.max(real)),
            "imag_max_abs": float(jnp.max(jnp.abs(imag))),
            "eig_min_abs": min_abs,
            "eig_max_abs": max_abs,
            "eig_cond_abs": float(cond_abs),
            "eig_min_nonzero": min_nonzero,
            "n_nonzero": n_nonzero,
            "cond_curl": float(cond_curl),
            "curl_percentiles": curl_percentiles,
            "curl_tail_counts": tail_counts,
        }

    results = []
    for mode in modes:
        if mode == "active":
            results.append(_summarize(f"active:{active_pa_name}", active_pa_apply))
        elif mode == "jacobi":
            pa_apply = lambda x, d=k_diaginv: d * x
            results.append(_summarize("jacobi", pa_apply))
        elif mode == "pinv":
            s_pinv = jnp.linalg.pinv(s_dense, rcond=pinv_rcond, hermitian=True)
            pa_apply = lambda x, p=s_pinv: p @ x
            results.append(_summarize("dense pinv", pa_apply))
        else:
            raise ValueError(f"Unsupported P_A spectrum mode {mode!r}")

    return {
        "n": n,
        "results": results,
    }


def run_k2_benchmark(seq, ops, args, *, report_rel_tol: float) -> None:
    """k=2 Hodge-Laplacian saddle benchmark: raw vs projected div-div P_A + P_B.

    Mirrors the k=1 saddle table one degree up. The scientific question is
    whether the block_fd-quality, grad-complement-projected ``K_1^{-1}`` atom is
    accurate enough to make the curl-complement projected div-div preconditioner
    converge (and beat the Schur-Jacobi baseline).
    """
    # Two atom variants: K_1^{-1} with the inner grad-complement sandwich ON
    # (production design) and OFF (ablation: raw block_fd, gradient nullspace
    # exposed at the outer K_1^{-1} apply).
    tg = bool(getattr(args, "true_g", False))
    if tg:
        print("[diag] k=2: using TRUE polar curl G_1 = Gram_2^-1 . (E^T sp E) in projector/P_B")
    ap_atom_on = make_apply_routines_k2(seq, ops, grad_project=True, project_atom=True, true_g=tg)
    ap_atom_off = make_apply_routines_k2(seq, ops, grad_project=True, project_atom=False, true_g=tg)
    # Full-L_1 atom: P_B (and the curl-complement projector) use the validated
    # k=1 Hodge preconditioner as the inner inverse instead of the rough,
    # singular block_fd curl-curl K_1^{-1}. This targets the documented k=2 root
    # cause (the near-kernel of K_1) by regularizing it with the grad-div term.
    ap_full = make_apply_routines_k2(seq, ops, grad_project=True, atom="full_l1", true_g=tg)

    n_upper = int(seq.n2_dbc if DIRICHLET else seq.n2)
    n_lower = int(seq.n1_dbc if DIRICHLET else seq.n1)
    print(f"[diag] k=2 Hodge-Laplacian path (n2_dbc={n_upper}, n1_dbc={n_lower})")
    print("[diag] 2x2 projection ablation: outer = curl-complement sandwich "
          "around div-div P_A; inner = grad-complement sandwich inside the "
          "K_1^{-1} atom (used by P_B and the outer projector).")

    # Consistent RHS: A x_true for random x_true in V2 (guarantees b in range A).
    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.n_rhs)
    rhs_batch = jnp.stack(
        [ap_atom_on["a_matvec"](jax.random.normal(k, (n_upper,), dtype=jnp.float64))
         for k in keys],
        axis=0,
    )
    jax.block_until_ready(rhs_batch)

    # Additive HX/ADS-style combinations: rough smoother (jacobi) + auxiliary
    # corrections, NO projectors, NO exact atoms. The solo run showed P_B
    # (curl-aux) is healthy (0.43%) while P_A and the K_1^{-1}-based projectors
    # are destructive. ADS philosophy: let the smoother cover the co-exact part
    # and the outer Krylov clean up the rough single applies.
    jac = ap_atom_on["jacobi_diag"]       # WHOLE-space diag(L_2) -> overlaps P_B on curls
    jac_s = ap_atom_on["jacobi_stiff"]    # diag(S_2) only -> ZERO on curls, clean split
    p_b_raw = ap_atom_off["p_b"]          # curl-aux, raw block_fd atom (cheaper, no worse)
    p_a_raw = ap_atom_on["p_a_raw"]       # div-div co-exact block
    methods = {}
    # Clean co-exact/curl splits (no double-counting): stiffness-only jacobi on
    # the co-exact part, P_B on curls. The k=2 analog of k=1 jacobi(S)+P_B.
    methods["jacobi(S) + P_B (clean split)"] = lambda r: jac_s(r) + p_b_raw(r)
    if jac is not None:
        methods["jacobi(whole) + P_B (overlaps)"] = lambda r: jac(r) + p_b_raw(r)
    methods["P_B (curl-aux) only"] = p_b_raw
    # Full-L_1 atom variants: the central comparison for the "use full L_1 in
    # P_B" hypothesis. P_B-only isolates the atom swap; the projected and
    # jacobi(S)-split forms test whether the better atom rescues the full
    # preconditioner / additive blend that block_fd could not.
    p_b_full = ap_full["p_b"]
    methods["P_B (full L_1, curl-aux) only"] = p_b_full
    methods["jacobi(S) + P_B (full L_1)"] = lambda r: jac_s(r) + p_b_full(r)
    methods["projected P_A + P_B (full L_1)"] = ap_full["projected_p_a_plus_p_b"]
    # ROUTE (b): CAPPED div-div P_A (block_fd mimic of k=1), bounded on curls.
    # The test: does raw P_A(capped) + P_B converge with NO projector (and beat
    # jacobi)? If so, k=2 needs neither the projection nor a near-exact atom.
    methods["P_A(capped) only"] = ap_full["p_a_capped"]
    methods["raw P_A(capped) + P_B (full L_1)"] = ap_full["raw_p_a_capped_plus_p_b"]
    methods["projected P_A(capped) + P_B (full L_1)"] = ap_full["projected_p_a_capped_plus_p_b"]
    # CHEB-TENSOR atom (the keep-projection path) is DISABLED: it produces NaN in
    # the squared P_B on the singular free-L_1 even with deflation, and route (b)
    # (capped P_A, no projection) makes it unnecessary. Re-enable by setting
    # K2_CHEB_TENSOR=1 if revisiting the deflation bug.
    if os.environ.get("K2_CHEB_TENSOR", "0") not in ("0", "", "false", "False"):
        for d in (3, 5, 8):
            ap_ct = make_apply_routines_k2(
                seq, ops, grad_project=True, atom="cheb_tensor",
                atom_cheb_degree=d, true_g=tg)
            methods[f"P_B (cheb-tensor-{d}) only"] = ap_ct["p_b"]
            methods[f"projected P_A + P_B (cheb-tensor-{d})"] = ap_ct["projected_p_a_plus_p_b"]
    if ap_atom_on["jacobi_diag"] is not None:
        methods = {"jacobi (diag)": ap_atom_on["jacobi_diag"], **methods}
    else:
        print("[diag] note: k=2 Schur-Jacobi baseline unavailable (not assembled)")

    applies = ap_atom_on  # forward matvecs/lower precond are atom-independent
    print()
    print("[diag] saddle MINRES (upper varies, lower fixed=tensor mass M_1)")
    header = (
        f"{'upper precond':<42} {'avg_it':>8} {'max_it':>7} "
        f"{'avg_ms':>9} {'max_res':>11} {'fails':>7}"
    )
    print(header)
    print("-" * len(header))
    for name, precond_upper in methods.items():
        solve = make_saddle_solve(
            applies["stiffness_matvec"],
            applies["derivative_matvec"],
            applies["derivative_t_matvec"],
            applies["mass_lower_matvec"],
            precond_upper,
            applies["lower_tensor_precond"],
            n_upper=n_upper,
            n_lower=n_lower,
            tol=args.cg_tol,
            maxiter=args.cg_maxiter,
        )
        stats = time_saddle_solve(solve, rhs_batch, rel_tol=report_rel_tol)
        print(f"{name:<42} {stats['avg_iters']:>8.1f} {stats['max_iters']:>7d} "
              f"{stats['avg_ms']:>9.1f} {stats['max_residual']:>11.2e} "
              f"{stats['n_fail']:>7d}/{stats['n_total']:<d}")


def run_k1_both_bc_benchmark(seq, ops, args, *, report_rel_tol: float) -> None:
    """k=1 Hodge-Laplacian saddle benchmark for BOTH BCs: dbc (nonsingular) and
    free (b1=1 harmonic nullspace, deflated). Compares jacobi vs the production
    tensor preconditioner (projected P_A + P_B), testing the vector tensor
    preconditioner's robustness to a nullspace.
    """
    print("[diag] k=1 Hodge-Laplacian saddle (jacobi vs tensor P.T P_S P + P_B); "
          "dbc = nonsingular, free = b1 harmonic (deflated).")
    header = (f"{'bc / upper precond':<34} {'avg_it':>8} {'max_it':>7} "
              f"{'avg_ms':>9} {'max_res':>11} {'fails':>7}")
    tg = bool(getattr(args, "true_g", False))
    if tg:
        print("[diag] k=1: using TRUE polar grad G_0 = Gram_1^-1 . (E^T sp E) in projector/P_B")
    # P_A bulk model from the CLI flags (default block_fd). Honoring these here
    # lets --k1-both-bc compare block_fd vs vector-fd vs radial-banded P_A.
    if getattr(args, "pa_block_radial_banded", False):
        pa_model = "radial_banded"
    elif getattr(args, "pa_block_vector_fd_true_basis", False):
        pa_model = "vector_fd_true_basis"
    elif getattr(args, "pa_block_vector_fd", False):
        pa_model = "vector_fd"
    else:
        pa_model = "block_fd"
    print(f"[diag] k=1: P_A bulk model = {pa_model}")
    for dirichlet in (True, False):
        bc = "dbc" if dirichlet else "free"
        applies = make_apply_routines(
            seq, ops, pa_mode="block_fd", grad_project=True, dirichlet_flag=dirichlet,
            true_g=tg,
            pa_block_vector_fd=getattr(args, "pa_block_vector_fd", False),
            pa_block_vector_fd_true_basis=getattr(
                args, "pa_block_vector_fd_true_basis", False),
            pa_block_radial_banded=getattr(args, "pa_block_radial_banded", False),
            pa_block_vector_fd_regularization_rel=getattr(
                args, "pa_block_vector_fd_regularization_rel", 1e-2),
            pa_block_vector_fd_low_mode_exclude=getattr(
                args, "pa_block_vector_fd_low_mode_exclude", 0),
        )
        n_upper = int(seq.n1_dbc if dirichlet else seq.n1)
        n_lower = int(seq.n0_dbc if dirichlet else seq.n0)
        vs_upper = _nullspace_vectors(ops, 1, dirichlet)
        n_harm = int(jnp.asarray(vs_upper).shape[0])

        def mass_upper(v, d=dirichlet):
            return apply_mass_matrix(seq, ops, v, 1, dirichlet=d)

        keys = jax.random.split(jax.random.PRNGKey(args.seed), args.n_rhs)
        rhs_batch = jnp.stack(
            [applies["a_matvec"](jax.random.normal(k, (n_upper,), dtype=jnp.float64))
             for k in keys], axis=0)
        jax.block_until_ready(rhs_batch)
        print()
        print(f"[diag] k=1 {bc}: n_upper={n_upper}, n_lower={n_lower}, harmonics={n_harm}")
        print(header)
        print("-" * len(header))

        methods = {
            "jacobi (diag)": (applies["jacobi_diag"], None),
            f"P.T P_A P + P_B [{pa_model}]": (
                applies["projected_p_a_plus_p_b_with_state"], applies["p_a_state"]),
        }
        for name, (precond_upper, precond_state) in methods.items():
            solve = make_saddle_solve(
                applies["stiffness_matvec"],
                applies["derivative_matvec"],
                applies["derivative_t_matvec"],
                applies["mass_lower_matvec"],
                precond_upper,
                applies["lower_tensor_precond"],
                n_upper=n_upper, n_lower=n_lower,
                tol=args.cg_tol, maxiter=args.cg_maxiter,
                precond_upper_state=precond_state,
                vs_upper=(vs_upper if n_harm > 0 else None),
                mass_upper_matvec=(mass_upper if n_harm > 0 else None),
            )
            stats = time_saddle_solve(
                solve, rhs_batch, solve_state=precond_state, rel_tol=report_rel_tol)
            print(f"{bc + ' / ' + name:<34} {stats['avg_iters']:>8.1f} "
                  f"{stats['max_iters']:>7d} {stats['avg_ms']:>9.1f} "
                  f"{stats['max_residual']:>11.2e} "
                  f"{stats['n_fail']:>7d}/{stats['n_total']:<d}")


def run_k0_benchmark(seq, ops, args, *, report_rel_tol: float) -> None:
    """k=0 Hodge-Laplacian (stiffness L_0) condensed CG: tensor vs jacobi
    preconditioner, for dbc (nonsingular) AND free (constant nullspace).

    Tests whether the tensor Hodge preconditioner stays effective when the
    operator is SINGULAR (free BCs -> constant nullspace, deflated in the CG via
    the stored nullspace vectors). L_0 = G_0^T M_1 G_0 is applied directly (no
    inner mass inverse), so this is a plain scalar CG, not a saddle solve.
    """
    print("[diag] k=0 stiffness L_0 condensed CG (tensor vs jacobi); "
          "dbc = nonsingular, free = constant nullspace (deflated).")
    header = (f"{'bc / precond':<26} {'avg_it':>8} {'max_it':>7} "
              f"{'avg_ms':>9} {'max_res':>11} {'fails':>7}")
    print(header)
    print("-" * len(header))
    for dirichlet in (True, False):
        n = int(seq.n0_dbc if dirichlet else seq.n0)
        vs = _nullspace_vectors(ops, 0, dirichlet)
        bc = "dbc" if dirichlet else "free"

        def a_matvec(v, d=dirichlet):
            return apply_stiffness(seq, ops, v, 0, dirichlet=d)

        def mass_matvec(v, d=dirichlet):
            return apply_mass_matrix(seq, ops, v, 0, dirichlet=d)

        keys = jax.random.split(jax.random.PRNGKey(args.seed), args.n_rhs)
        rhs_batch = jnp.stack(
            [a_matvec(jax.random.normal(k, (n,), dtype=jnp.float64)) for k in keys],
            axis=0)
        jax.block_until_ready(rhs_batch)
        print(f"[diag] k=0 {bc}: n={n}, nullspace dim={int(jnp.asarray(vs).shape[0])}")

        # Precompute diag(L_0)^{-1} ONCE for a fair jacobi (apply_laplacian_
        # preconditioner(kind='jacobi') re-probes the diagonal on every call,
        # which gets traced into the CG loop -> bogus wall times).
        l0_diaginv = _invert_diagonal(_diagonal_from_matvec(a_matvec, n))

        def jacobi_precond(v, di=l0_diaginv):
            return di * v

        def tensor_precond(v, d=dirichlet):
            return apply_laplacian_preconditioner(
                seq, ops, v, 0, dirichlet=d, kind="tensor")

        for kind, precond in (("jacobi", jacobi_precond), ("tensor", tensor_precond)):
            @jax.jit
            def solve(rhs, a=a_matvec, m=mass_matvec, p=precond, vv=vs):
                x, info = solve_singular_cg(
                    a, rhs, mass_matvec=m, precond_matvec=p, vs=vv,
                    tol=args.cg_tol, maxiter=args.cg_maxiter)
                r = a(x) - rhs
                rel = jnp.linalg.norm(r) / jnp.maximum(jnp.linalg.norm(rhs), 1e-30)
                return x, info, rel

            stats = time_solve(solve, rhs_batch, rel_tol=report_rel_tol)
            print(f"{bc + ' / ' + kind:<26} {stats['avg_iters']:>8.1f} "
                  f"{stats['max_iters']:>7d} {stats['avg_ms']:>9.1f} "
                  f"{stats['max_residual']:>11.2e} "
                  f"{stats['n_fail']:>7d}/{stats['n_total']:<d}")


def run_k3_benchmark(seq, ops, args, *, report_rel_tol: float) -> None:
    """k=3 Hodge-Laplacian saddle benchmark: jacobi vs auxiliary-space transfer.

    Proof-of-concept for the "map sideways" idea on the simplest (scalar) dual
    pair k=3<->k=0: precondition L_3 by transferring to k=0, applying the
    near-exact k=0 Hodge preconditioner, and transferring back. If the transfer
    P_3 beats jacobi here, the same construction (with vector V1<->V2 transfer
    and the k=1 preconditioner) is the route for k=2.
    """
    applies = make_apply_routines_k3(seq, ops)

    # k=3 runs with FREE BCs (its dual k=0 is dbc -> working tensor L_0^{-1}).
    n_upper = int(seq.n3)
    n_lower = int(seq.n2)
    print(f"[diag] k=3 Hodge-Laplacian path, FREE BCs (n3={n_upper}, n2={n_lower})")
    print("[diag] auxiliary-space transfer to k=0 dbc (BC swap: k=3 no-dbc <-> "
          "k=0 dbc); M^-1 via tensor mass precond, L_0^-1 via the working dbc "
          "k=0 tensor Hodge precond.")

    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.n_rhs)
    rhs_batch = jnp.stack(
        [applies["a_matvec"](jax.random.normal(k, (n_upper,), dtype=jnp.float64))
         for k in keys],
        axis=0,
    )
    jax.block_until_ready(rhs_batch)

    methods = {"transfer P_3 = T L_0^-1 T* (aux-space)": applies["p3_transfer"]}
    if applies["jacobi_diag"] is not None:
        jac = applies["jacobi_diag"]
        tr = applies["p3_transfer"]
        # HX/ADS completion: transfer is rank-deficient (T: V0 -> V3 is not
        # surjective -- different-order bases), leaving a subspace unpreconditioned
        # -> hard stall. Add the jacobi smoother to cover that remainder. At unit
        # weight the near-exact transfer and whole-space jacobi double-count on
        # the transfer's range; sweep the smoother weight alpha to down-weight it
        # so the transfer dominates where it is good and jacobi only fills the
        # complement.
        for a in (0.05, 0.1, 0.25, 0.5, 1.0):
            methods[f"{a:g}*jacobi + transfer"] = (lambda r, a=a: a * jac(r) + tr(r))
        methods = {"jacobi (diag)": jac, **methods}
    else:
        print("[diag] note: k=3 Schur-Jacobi baseline unavailable (not assembled)")

    print()
    print("[diag] saddle MINRES (upper varies, lower fixed=tensor mass M_2)")
    header = (
        f"{'upper precond':<42} {'avg_it':>8} {'max_it':>7} "
        f"{'avg_ms':>9} {'max_res':>11} {'fails':>7}"
    )
    print(header)
    print("-" * len(header))
    for name, precond_upper in methods.items():
        solve = make_saddle_solve(
            applies["stiffness_matvec"],
            applies["derivative_matvec"],
            applies["derivative_t_matvec"],
            applies["mass_lower_matvec"],
            precond_upper,
            applies["lower_tensor_precond"],
            n_upper=n_upper,
            n_lower=n_lower,
            tol=args.cg_tol,
            maxiter=args.cg_maxiter,
        )
        stats = time_saddle_solve(solve, rhs_batch, rel_tol=report_rel_tol)
        print(f"{name:<42} {stats['avg_iters']:>8.1f} {stats['max_iters']:>7d} "
              f"{stats['avg_ms']:>9.1f} {stats['max_residual']:>11.2e} "
              f"{stats['n_fail']:>7d}/{stats['n_total']:<d}")


def main() -> None:
    global DIRICHLET
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ns", type=str, default="6,12,4",
                        help="Comma-separated (n_r,n_theta,n_zeta).")
    parser.add_argument("--p", type=int, default=3)
    parser.add_argument("--epsilon", type=float, default=1.0 / 3.0)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--r0", type=float, default=1.0)
    parser.add_argument(
        "--geometry", choices=("toroid", "rotating_ellipse", "cylinder", "w7x"),
        default="toroid",
        help="Mapping: axisymmetric toroid (default), symmetry-breaking "
             "rotating ellipse, periodic cylinder, or W7-X stellarator "
             "(nfp=5, from data/W7-X.h5).")
    parser.add_argument(
        "--nfp", type=int, default=3,
        help="Field periods for the rotating-ellipse map (ignored for toroid).")
    parser.add_argument(
        "--leakage-only", action="store_true",
        help="Stop after P_A assembly + leakage diagnostics; skip property/"
             "overlap/spectrum checks, RHS build, and the MINRES table.")
    parser.add_argument("--n-rhs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cg-tol", type=float, default=1e-10)
    parser.add_argument("--cg-maxiter", type=int, default=2000)
    parser.add_argument(
        "--report-rel-tol",
        type=float,
        default=None,
        help=(
            "Residual threshold used only for fail counting in summary tables. "
            "Defaults to --cg-tol when omitted."
        ),
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=RANK,
        help=(
            "CP rank for the tensor preconditioners used by block_fd."
        ),
    )
    parser.add_argument(
        "--k1-both-bc",
        action="store_true",
        help=(
            "Run the k=1 saddle benchmark for BOTH boundary conditions (dbc, "
            "nonsingular; and free, with a b1=1 harmonic nullspace deflated), "
            "comparing jacobi vs the tensor preconditioner, then exit."
        ),
    )
    parser.add_argument(
        "--true-g",
        action="store_true",
        help=(
            "Use the TRUE polar derivative G = Gram^{-1}.(E^T sp E) (Gram = E^T E, "
            "no mass assembly) in the projector/P_B, instead of the directly-built "
            "apply_incidence which is non-nilpotent on the polar axis. Applies to "
            "the k=1 and k=2 benchmarks."
        ),
    )
    parser.add_argument(
        "--klevel",
        type=int,
        default=1,
        choices=(0, 1, 2, 3),
        help=(
            "Form degree of the Hodge Laplacian to precondition. 0 = k=0 "
            "stiffness condensed-CG nullspace test (tensor vs jacobi, dbc vs "
            "free). 1 (default) = the k=1 grad-div path. 2 = the k=2 div-div "
            "path. 3 = the k=3 auxiliary-space transfer to k=0. "
            "0, 2, 3 run their dedicated benchmark and exit."
        ),
    )
    parser.add_argument(
        "--pa-check-tol",
        type=float,
        default=1e-10,
        help="Tolerance for P_A symmetry/PSD diagnostics.",
    )
    parser.add_argument(
        "--pa-profile",
        action="store_true",
        default=False,
        help="Profile P_A apply phases (bulk1/schur/bulk2) outside Krylov.",
    )
    parser.add_argument(
        "--compact-output",
        action="store_true",
        default=False,
        help="Print compact diagnostic summaries instead of full verbose blocks.",
    )
    parser.add_argument(
        "--assembly-breakdown",
        action="store_true",
        default=False,
        help="Print per-substep timings during operator/preconditioner assembly.",
    )
    parser.add_argument(
        "--pa-mode",
        type=str,
        default="block_fd",
        choices=("block_fd",),
        help=(
            "Choose P_A implementation (current supported mode: block_fd)."
        ),
    )
    parser.add_argument(
        "--pa-block-inner-schur",
        action="store_true",
        default=False,
        help=(
            "Enable inner bulk Schur coupling when --pa-mode=block_fd. "
            "Ignored for other pa modes."
        ),
    )
    parser.add_argument(
        "--pa-block-vector-fd",
        action="store_true",
        default=False,
        help=(
            "Experimental: use vector-valued modewise 3x3 FD coupling in the "
            "block_fd bulk inverse (diagonal-metric model). Not compatible "
            "with --pa-block-inner-schur."
        ),
    )
    parser.add_argument(
        "--pa-block-vector-fd-true-basis",
        action="store_true",
        default=False,
        help=(
            "Experimental: vector-valued modewise 3x3 FD bulk coupling with "
            "modal bases/symbols probed from the true extracted bulk operator. "
            "Not compatible with --pa-block-inner-schur or --pa-block-vector-fd."
        ),
    )
    parser.add_argument(
        "--pa-block-radial-banded",
        action="store_true",
        default=False,
        help=(
            "Root-cause fix for the k=1 low-mode defect: keep the full "
            "radial-mode and cross-component coupling per angular mode "
            "(i_t, i_z) instead of the same-mode 3x3 symbol. Exact in "
            "radial+component coupling, block-diagonal in the (measured ~3%%) "
            "angular leakage. Not compatible with --pa-block-inner-schur, "
            "--pa-block-vector-fd, or --pa-block-vector-fd-true-basis."
        ),
    )
    parser.add_argument(
        "--pa-block-vector-fd-regularization-rel",
        type=float,
        default=1e-2,
        help=(
            "Relative spectral floor used to regularize near-singular 3x3 "
            "vector-FD modal blocks (applied only to flagged blocks)."
        ),
    )
    parser.add_argument(
        "--pa-block-vector-fd-low-mode-exclude",
        type=int,
        default=0,
        help=(
            "Force stronger regularization floor for the first N shared modal "
            "indices in vector-FD true-basis mode (diagnostic stability knob)."
        ),
    )
    parser.add_argument(
        "--pa-block-vector-fd-report-k",
        type=int,
        default=8,
        help=(
            "How many lowest/highest per-mode min-eig(3x3) entries to print "
            "for vector-FD true-basis diagnostics."
        ),
    )
    parser.add_argument(
        "--pa-block-vector-fd-origin-k",
        type=int,
        default=4,
        help=(
            "How many flagged regularized 3x3 blocks to print in detail when "
            "diagnosing the mathematical origin of the low modes."
        ),
    )
    parser.add_argument(
        "--no-precompute-coupling",
        action="store_true",
        default=False,
        help=(
            "Disable the production precompute of the dense surgery<->bulk "
            "coupling blocks (k=1 curl-curl C and k=0 Hodge core coupling C0). "
            "Precompute is ON by default (set on the preconditioner payload at "
            "construction); this flag restores the matrix-free coupling path "
            "for A/B comparison."
        ),
    )
    parser.add_argument(
        "--mass-benchmark",
        action="store_true",
        default=False,
        help=(
            "Additionally benchmark pure k=0 mass inversion M_0^{-1} via CG, "
            "comparing jacobi vs the tensor preconditioner with the dense "
            "surgery coupling precomputed ON vs OFF (built in the same run). "
            "Reports a separate table alongside the saddle/condensed results."
        ),
    )
    parser.add_argument(
        "--pa-grad-project",
        action="store_true",
        default=False,
        help=(
            "Sandwich the active P_A between gradient-subspace complement "
            "projectors so it acts only on the curl-dominated complement, "
            "leaving the gradient (curl-free) subspace to P_B. Run two jobs "
            "(with and without) to compare iteration counts and overlap."
        ),
    )
    parser.add_argument(
        "--pa-stiffness-spectrum",
        type=str,
        default="none",
        help=(
            "Dense spectrum check for P_A S (k=1 stiffness). "
            "Choose one of: none, all, or comma list from active,jacobi,pinv."
        ),
    )
    parser.add_argument(
        "--pa-stiffness-spectrum-pinv-rcond",
        type=float,
        default=1e-12,
        help="Relative cutoff for dense pseudoinverse used in P_A S pinv spectral check.",
    )
    parser.add_argument(
        "--pa-stiffness-spectrum-curl-cutoff",
        type=float,
        default=1e-8,
        help=(
            "Relative |eig| cutoff that separates the gradient nullspace of "
            "K_1 from the curl modes when computing cond_curl in the P_A S "
            "spectral check."
        ),
    )
    parser.add_argument(
        "--pa-stiffness-spectrum-tail-k",
        type=int,
        default=4,
        help=(
            "How many largest-|eig| P_A S modes to print with tail-origin "
            "diagnostics."
        ),
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="all",
        help=(
            "Comma-separated method names to run, or 'all'. "
            "Example: 'jacobi (diag),jacobi(K)+P_B'"
        ),
    )
    parser.add_argument(
        "--jacobi-scale-sweep",
        type=str,
        default="",
        help=(
            "Comma-separated sweep for alpha in alpha*jacobi(diag)+P_A+P_B. "
            "Runs through one jitted solve with alpha passed at runtime. "
            "Example: '0.5,1,2,4'"
        ),
    )
    parser.add_argument(
        "--rhs-kind",
        type=str,
        default="random",
        choices=("random", "gradient"),
        help="Build consistent RHS from a random 1-form or a pure gradient trial vector.",
    )
    parser.add_argument(
        "--system",
        type=str,
        default="saddle",
        choices=("condensed", "saddle", "both"),
        help="Run condensed CG, full saddle MINRES, or both.",
    )
    args = parser.parse_args()
    args.ns = tuple(int(s) for s in args.ns.split(","))
    report_rel_tol = args.cg_tol if args.report_rel_tol is None else float(args.report_rel_tol)
    pa_spectrum_modes = _parse_pa_spectrum_modes(args.pa_stiffness_spectrum)
    jacobi_scale_sweep = [
        float(token.strip())
        for token in args.jacobi_scale_sweep.split(",")
        if token.strip()
    ]
    rank = int(args.rank)

    if args.pa_block_vector_fd and args.pa_block_inner_schur:
        raise ValueError(
            "--pa-block-vector-fd is not compatible with --pa-block-inner-schur. "
            "Disable one of them."
        )
    if args.pa_block_vector_fd_true_basis and args.pa_block_inner_schur:
        raise ValueError(
            "--pa-block-vector-fd-true-basis is not compatible with --pa-block-inner-schur. "
            "Disable one of them."
        )
    if args.pa_block_vector_fd and args.pa_block_vector_fd_true_basis:
        raise ValueError(
            "--pa-block-vector-fd and --pa-block-vector-fd-true-basis are mutually exclusive. "
            "Pick one experimental vector-FD mode."
        )
    if args.pa_block_radial_banded and (
        args.pa_block_inner_schur
        or args.pa_block_vector_fd
        or args.pa_block_vector_fd_true_basis
    ):
        raise ValueError(
            "--pa-block-radial-banded is not compatible with --pa-block-inner-schur, "
            "--pa-block-vector-fd, or --pa-block-vector-fd-true-basis. Pick one bulk model."
        )

    # k=2 L_2 = S_2 + D_1 M_1^{-1} D_1^T is SINGULAR under DBC (b_1 harmonic via the
    # BC flip) and nonsingular under free BC; the established dense reference
    # (exact L_1^+ = 91 it) is free. Force free BC for the k=2 benchmark so the
    # Schur-jacobi baseline is assembled for the matching BC.
    if args.klevel == 2:
        DIRICHLET = False
        print("[diag] k=2: forcing free BC (L_2 nonsingular; matches dense reference)")

    print(f"[diag] building sequence ns={args.ns} p={args.p} {args.geometry} "
          f"epsilon={args.epsilon:.4f} kappa={args.kappa} R0={args.r0} nfp={args.nfp}")
    t0 = time.perf_counter()
    seq = build_sequence(args)
    print(f"[diag] sequence built in {(time.perf_counter() - t0) * 1e3:.1f} ms "
          f"(n1_dbc={int(seq.n1_dbc)}, n0_dbc={int(seq.n0_dbc)})")

    t0 = time.perf_counter()
    use_k1_inner_schur = args.pa_block_inner_schur
    ops = assemble_operators(
        seq,
        rank=rank,
        k1_stiff_inner_schur=use_k1_inner_schur,
        precompute_coupling=not args.no_precompute_coupling,
        timing_breakdown=args.assembly_breakdown,
        klevel=args.klevel,
        both_bc=args.k1_both_bc,
    )
    print(f"[diag] operators + rank-{rank} tensor preconditioners assembled in "
          f"{(time.perf_counter() - t0) * 1e3:.1f} ms")

    if args.k1_both_bc:
        run_k1_both_bc_benchmark(seq, ops, args, report_rel_tol=report_rel_tol)
        return

    if args.klevel == 0:
        run_k0_benchmark(seq, ops, args, report_rel_tol=report_rel_tol)
        return

    if args.klevel == 2:
        run_k2_benchmark(seq, ops, args, report_rel_tol=report_rel_tol)
        return

    if args.klevel == 3:
        run_k3_benchmark(seq, ops, args, report_rel_tol=report_rel_tol)
        return

    applies = make_apply_routines(
        seq,
        ops,
        pa_mode=args.pa_mode,
        grad_project=args.pa_grad_project,
        pa_block_vector_fd=args.pa_block_vector_fd,
        pa_block_vector_fd_true_basis=args.pa_block_vector_fd_true_basis,
        pa_block_radial_banded=args.pa_block_radial_banded,
        pa_block_vector_fd_regularization_rel=args.pa_block_vector_fd_regularization_rel,
        pa_block_vector_fd_low_mode_exclude=args.pa_block_vector_fd_low_mode_exclude,
        pa_block_vector_fd_report_k=args.pa_block_vector_fd_report_k,
        pa_block_vector_fd_origin_k=args.pa_block_vector_fd_origin_k,
        pa_profile=args.pa_profile,
        true_g=bool(getattr(args, "true_g", False)),
    )

    print(f"[diag] using P_A mode: {args.pa_mode}")
    if args.pa_mode == "block_fd":
        if args.pa_block_radial_banded:
            print("[diag] P_A detail: block_fd + radial-banded bulk coupling "
                  "(full radial+component per angular mode; root-cause fix)")
            print(
                "[diag] P_A detail: radial-banded regularization "
                f"rel_floor={float(args.pa_block_vector_fd_regularization_rel):.2e}"
            )
        elif args.pa_block_vector_fd_true_basis:
            print("[diag] P_A detail: block_fd + vector-valued modewise 3x3 FD bulk coupling (true-basis experimental)")
            print(
                "[diag] P_A detail: true-basis vector-FD regularization "
                f"rel_floor={float(args.pa_block_vector_fd_regularization_rel):.2e} "
                f"low-mode-force-N={int(args.pa_block_vector_fd_low_mode_exclude)}"
            )
        elif args.pa_block_vector_fd:
            print("[diag] P_A detail: block_fd + vector-valued modewise 3x3 FD bulk coupling (experimental)")
        elif args.pa_block_inner_schur:
            print("[diag] P_A detail: block-component pseudoinverse + surgery Schur + inner bulk Schur")
        else:
            print("[diag] P_A detail: block-component pseudoinverse with surgery Schur (new)")
    if args.pa_grad_project:
        print("[diag] P_A gradient-subspace projection: ON "
              "(P_A acts on curl complement; gradient subspace left to P_B)")
    else:
        print("[diag] P_A gradient-subspace projection: OFF (raw additive P_A)")

    if args.no_precompute_coupling:
        print("[diag] dense coupling precompute: OFF "
              "(k=1 surgery + k=0 Hodge core couplings use matrix-free applies)")
    else:
        print("[diag] dense coupling precompute: ON (production default) "
              "(k=1 surgery C and k=0 Hodge core C0 are dense matvecs)")

    if args.leakage_only:
        print("[diag] leakage-only run: skipping property/overlap/spectrum "
              "checks, RHS build, and MINRES table.")
        return

    t0 = time.perf_counter()
    rhs_batch = build_rhs_batch(
        seq,
        ops,
        applies["a_matvec"],
        n_rhs=args.n_rhs,
        seed=args.seed,
        rhs_kind=args.rhs_kind,
    )
    jax.block_until_ready(rhs_batch)
    print(f"[diag] {args.n_rhs} consistent RHS ({args.rhs_kind}) built in "
          f"{(time.perf_counter() - t0) * 1e3:.1f} ms")

    methods = {
        "jacobi (diag)": (applies["jacobi_diag"], None),
        "raw P_S + P_B": (
            lambda state, r: applies["p_a_raw_with_state"](state, r) + applies["p_b"](r),
            applies["p_a_state"],
        ),
        "jacobi + P.T P_S P + P_B": (
            applies["jacobi_plus_p_a_plus_p_b_with_state"],
            applies["p_a_state"],
        ),
        "jacobi + raw P_S + P_B": (
            applies["jacobi_plus_p_a_raw_plus_p_b_with_state"],
            applies["p_a_state"],
        ),
        "jacobi(K)+P_B": (applies["k_jacobi_plus_p_b"], None),
        "P.T P_S P + P_B": (applies["projected_p_a_plus_p_b_with_state"], applies["p_a_state"]),
        "P.T P_S P + P_B (fused)": (applies["projected_p_a_plus_p_b_fused_with_state"], applies["p_a_state"]),
        "P.T jacobi(S) P + P_B": (applies["projected_jacobi_plus_p_b"], None),
    }

    # Richardson / Chebyshev acting on the APPROXIMATE Schur S-hat = a_matvec
    # (the M->preconditioner approximation, same operator the jacobi diagonal is
    # probed from), inner smoother = jacobi diag(S-hat)^{-1}. Spectrum of the
    # jacobi-preconditioned S-hat estimated once via Lanczos on the symmetric
    # M = D^{-1/2} S-hat D^{-1/2}. Fixed degree -> linear SPD upper precond.
    s_hat = applies["a_matvec"]
    jac = applies["jacobi_diag"]
    sqrt_di = jnp.sqrt(jnp.abs(applies["schur_diaginv"]))

    def _m_sym(v):
        return sqrt_di * s_hat(sqrt_di * v)

    lmin_raw, lmax = _lanczos_extremal_eigs(
        _m_sym, int(seq.n1_dbc if DIRICHLET else seq.n1), steps=30, seed=args.seed)
    lmin = max(lmin_raw, lmax * 1e-3)  # defensive floor for the Chebyshev interval
    print(f"[diag] approx-Schur jacobi-precond spectrum: raw[{lmin_raw:.3e},{lmax:.3e}] "
          f"used lmin={lmin:.3e} (cond~{lmax / lmin:.1f})")
    for d in (2, 3, 5, 8):
        methods[f"richardson-{d}"] = (make_richardson_upper(s_hat, jac, lmin, lmax, d), None)
        methods[f"chebyshev-{d}"] = (make_chebyshev_upper(s_hat, jac, lmin, lmax, d), None)

    # Polynomial methods with the GOOD TENSOR PRECONDITIONER as the inner smoother
    # (the k=1 Hodge precond ~ L_1^{-1}, kappa~28, NOT jacobi). This is the right
    # way to turn the kappa~28 tensor preconditioner into a near-exact L_1^{-1}:
    # a few Chebyshev iterations should drop the outer MINRES to ~10-20 iters.
    # That accelerated apply IS the cheap, scalable, near-exact atom k=2 needs.
    # Interval via A-inner-product Lanczos (the smoother is non-diagonal).
    tensor_smoother = applies["projected_p_a_plus_p_b"]   # k=1 precond ~ L_1^{-1}
    lmin_t, lmax_t = _lanczos_extremal_eigs_precond(
        s_hat, tensor_smoother, int(seq.n1_dbc if DIRICHLET else seq.n1),
        steps=30, seed=args.seed)
    lmin_t = max(lmin_t, lmax_t * 1e-3)
    print(f"[diag] approx-Schur TENSOR-precond spectrum: eig[{lmin_t:.3e},{lmax_t:.3e}] "
          f"(cond~{lmax_t / lmin_t:.1f}) -> Chebyshev/Richardson smoother")
    for d in (1, 2, 3, 5, 8, 10):
        methods[f"cheb-tensor-{d}"] = (
            make_chebyshev_upper(s_hat, tensor_smoother, lmin_t, lmax_t, d), None)
        methods[f"rich-tensor-{d}"] = (
            make_richardson_upper(s_hat, tensor_smoother, lmin_t, lmax_t, d), None)

    if args.methods.strip().lower() != "all":
        requested = [m.strip() for m in args.methods.split(",") if m.strip()]
        unknown = [m for m in requested if m not in methods]
        if unknown:
            raise ValueError(
                "Unknown method(s): "
                + ", ".join(unknown)
                + ". Available: "
                + ", ".join(methods.keys())
            )
        methods = {name: methods[name] for name in requested}

    print()
    print("[diag] P_A property checks (random + exhaustive unit-vector probes)")
    t0 = time.perf_counter()
    pa_diag = diagnose_pa_operator(
        applies["p_a"],
        n=int(seq.n1_dbc if DIRICHLET else seq.n1),
        seed=args.seed,
        tol=args.pa_check_tol,
    )
    print(f"[diag] P_A checks finished in {(time.perf_counter() - t0) * 1e3:.1f} ms")
    if args.compact_output:
        print(
            f"[diag] P_A summary: sym_rel={pa_diag['dense_sym_inf_rel']:.2e} "
            f"eig[min,max]=[{pa_diag['sym_part_min_eig']:.2e},{pa_diag['sym_part_max_eig']:.2e}] "
            f"diag[min,max]=[{pa_diag['unit_min_diag']:.2e},{pa_diag['unit_max_diag']:.2e}]"
        )
    else:
        print(f"[diag] bilinear symmetry: abs={pa_diag['bilinear_sym_abs']:.2e} "
              f"rel={pa_diag['bilinear_sym_rel']:.2e}")
        print(f"[diag] random Rayleigh x^T P_A x = {pa_diag['rayleigh_x']:.2e}")
        print(f"[diag] dense symmetry ||A-A^T||_inf={pa_diag['dense_sym_inf_abs']:.2e} "
              f"rel={pa_diag['dense_sym_inf_rel']:.2e}")
        print(f"[diag] unit-vector diag range: "
              f"[{pa_diag['unit_min_diag']:.2e}, {pa_diag['unit_max_diag']:.2e}]")
        print(f"[diag] sym(A) min eigenvalue: {pa_diag['sym_part_min_eig']:.2e}")
        print(f"[diag] sym(A) max eigenvalue: {pa_diag['sym_part_max_eig']:.2e}")
        print(f"[diag] sym(A) spectral radius: {pa_diag['sym_part_max_abs_eig']:.2e}")
    print(f"[diag] symmetry(pass={pa_diag['is_symmetric']}) "
          f"psd(pass={pa_diag['is_psd']}) with tol={args.pa_check_tol:.1e}")

    print()
    print("[diag] gradient-subspace overlap (raw vs gradient-projected P_A)")
    t0 = time.perf_counter()
    overlap = diagnose_grad_subspace_overlap(
        applies["p_a_raw"],
        applies["p_a_projected"],
        applies["p_b"],
        applies["project_primal_complement"],
        applies["project_primal_complement_exact"],
        applies["mass_matvec"],
        n=int(seq.n1_dbc if DIRICHLET else seq.n1),
        seed=args.seed,
    )
    print(f"[diag] overlap checks finished in {(time.perf_counter() - t0) * 1e3:.1f} ms "
          f"(n_probe={overlap['n_probe']})")
    if args.compact_output:
        print(
            f"[diag] overlap summary: grad_exact raw={overlap['raw_grad_frac_exact_mean']:.2e} "
            f"proj={overlap['proj_grad_frac_exact_mean']:.2e}; "
            f"|cos| raw={overlap['raw_pa_pb_cos_mean']:.2e} proj={overlap['proj_pa_pb_cos_mean']:.2e}"
        )
    else:
        print(f"[diag] raw P_A gradient-energy fraction (tensor proj): "
              f"mean={overlap['raw_grad_frac_tensor_mean']:.2e} "
              f"max={overlap['raw_grad_frac_tensor_max']:.2e} "
              f"(>1 = inexact-L0 artifact)")
        print(f"[diag] raw P_A gradient-energy fraction (exact proj):  "
              f"mean={overlap['raw_grad_frac_exact_mean']:.2e} "
              f"max={overlap['raw_grad_frac_exact_max']:.2e} "
              f"(true gradient energy, in [0,1])")
        print(f"[diag] projected P_A gradient-energy fraction (exact proj): "
              f"mean={overlap['proj_grad_frac_exact_mean']:.2e} "
              f"max={overlap['proj_grad_frac_exact_max']:.2e}")
        print(f"[diag] |M-cosine(P_A, P_B)| (projector-free): raw "
              f"mean={overlap['raw_pa_pb_cos_mean']:.2e} max={overlap['raw_pa_pb_cos_max']:.2e} "
              f"-> projected mean={overlap['proj_pa_pb_cos_mean']:.2e} "
              f"max={overlap['proj_pa_pb_cos_max']:.2e}")

    if args.pa_profile and applies["p_a_profile"] is not None:
        print()
        print("[diag] P_A apply phase profile (outside Krylov)")
        t0 = time.perf_counter()
        pa_prof = time_pa_profile(applies["p_a_profile"], rhs_batch)
        print(f"[diag] P_A profiling finished in {(time.perf_counter() - t0) * 1e3:.1f} ms")
        print(
            f"[diag] P_A profile avg_ms: bulk1={pa_prof['bulk1_ms']:.2f} "
            f"schur={pa_prof['schur_ms']:.2f} bulk2={pa_prof['bulk2_ms']:.2f} "
            f"total={pa_prof['total_ms']:.2f}"
        )

    if pa_spectrum_modes:
        print()
        print("[diag] dense stiffness spectrum checks for P_A S")
        t0 = time.perf_counter()
        pa_spectrum = diagnose_pa_times_stiffness_spectrum(
            seq,
            ops,
            applies["stiffness_matvec"],
            active_pa_apply=applies["p_a"],
            active_pa_name=args.pa_mode,
            mass1_matvec=applies["mass_matvec"],
            project_primal_complement_exact=applies["project_primal_complement_exact"],
            modes=pa_spectrum_modes,
            k_diaginv=applies["k_diaginv"],
            pinv_rcond=args.pa_stiffness_spectrum_pinv_rcond,
            curl_rel_cutoff=args.pa_stiffness_spectrum_curl_cutoff,
            tail_k=args.pa_stiffness_spectrum_tail_k,
        )
        print(f"[diag] P_A S checks finished in {(time.perf_counter() - t0) * 1e3:.1f} ms")
        print("[diag] cond_curl = lambda_max / lambda_min over nonzero (curl) "
              "modes; nnz = count of those modes. cond_abs is nullspace-polluted "
              "and not meaningful for K_1.")
        header = (
            f"{'P_A':<12} {'real_min':>11} {'real_max':>11} "
            f"{'|imag|max':>11} {'|eig|min':>11} {'|eig|max':>11} {'cond_abs':>11} "
            f"{'nnz':>5} {'min_nz':>11} {'cond_curl':>11}"
        )
        print(header)
        print("-" * len(header))
        for row in pa_spectrum["results"]:
            print(
                f"{row['name']:<12} "
                f"{row['real_min']:>11.2e} {row['real_max']:>11.2e} "
                f"{row['imag_max_abs']:>11.2e} {row['eig_min_abs']:>11.2e} "
                f"{row['eig_max_abs']:>11.2e} {row['eig_cond_abs']:>11.2e} "
                f"{row['n_nonzero']:>5d} {row['eig_min_nonzero']:>11.2e} "
                f"{row['cond_curl']:>11.2e}"
            )
            cp = row["curl_percentiles"]
            ct = row["curl_tail_counts"]
            print(
                " " * 13
                + f"curl |eig| percentiles: p05={cp['p05']:.2e} p25={cp['p25']:.2e} "
                + f"p50={cp['p50']:.2e} p75={cp['p75']:.2e} p95={cp['p95']:.2e}"
            )
            print(
                " " * 13
                + f"curl tail counts: >2={ct['gt2']} >5={ct['gt5']} >10={ct['gt10']} >20={ct['gt20']}"
            )

    if args.system in ("condensed", "both"):
        print()
        print("[diag] condensed CG (single-block preconditioner test)")
        header = (
            f"{'preconditioner':<38} {'avg_it':>8} {'max_it':>7} "
            f"{'avg_ms':>9} {'max_res':>11} {'fails':>7}"
        )
        print(header)
        print("-" * len(header))
        for name, (precond, precond_state) in methods.items():
            solve = make_solve(
                applies["a_matvec"], applies["mass_matvec"], precond,
                tol=args.cg_tol,
                maxiter=args.cg_maxiter,
                precond_state=precond_state,
            )
            stats = time_solve(
                solve,
                rhs_batch,
                solve_state=precond_state,
                rel_tol=report_rel_tol,
            )
            print(f"{name:<38} {stats['avg_iters']:>8.1f} {stats['max_iters']:>7d} "
                  f"{stats['avg_ms']:>9.1f} {stats['max_residual']:>11.2e} "
                  f"{stats['n_fail']:>7d}/{stats['n_total']:<d}")

    if args.system in ("saddle", "both"):
        print()
        print("[diag] saddle MINRES (upper varies, lower fixed=tensor mass)")
        header = (
            f"{'upper precond':<38} {'avg_it':>8} {'max_it':>7} "
            f"{'avg_ms':>9} {'max_res':>11} {'fails':>7}"
        )
        print(header)
        print("-" * len(header))
        for name, (precond_upper, precond_state) in methods.items():
            solve = make_saddle_solve(
                applies["stiffness_matvec"],
                applies["derivative_matvec"],
                applies["derivative_t_matvec"],
                applies["mass_lower_matvec"],
                precond_upper,
                applies["lower_tensor_precond"],
                n_upper=int(seq.n1_dbc),
                n_lower=int(seq.n0_dbc),
                tol=args.cg_tol,
                maxiter=args.cg_maxiter,
                precond_upper_state=precond_state,
            )
            stats = time_saddle_solve(
                solve,
                rhs_batch,
                solve_state=precond_state,
                rel_tol=report_rel_tol,
            )
            print(f"{name:<38} {stats['avg_iters']:>8.1f} {stats['max_iters']:>7d} "
                  f"{stats['avg_ms']:>9.1f} {stats['max_residual']:>11.2e} "
                  f"{stats['n_fail']:>7d}/{stats['n_total']:<d}")

        if jacobi_scale_sweep:
            print()
            print("[diag] alpha sweep for alpha*jacobi(diag) + P_A + P_B (single-compile path)")
            header = (
                f"{'alpha':<10} {'avg_it':>8} {'max_it':>7} "
                f"{'avg_ms':>9} {'max_res':>11} {'fails':>7}"
            )
            print(header)
            print("-" * len(header))
            solve = make_scaled_saddle_solve(
                applies["stiffness_matvec"],
                applies["derivative_matvec"],
                applies["derivative_t_matvec"],
                applies["mass_lower_matvec"],
                applies["jacobi_scaled_plus_p_a_plus_p_b_with_state"],
                applies["lower_tensor_precond"],
                n_upper=int(seq.n1_dbc if DIRICHLET else seq.n1),
                n_lower=int(seq.n0_dbc if DIRICHLET else seq.n0),
                tol=args.cg_tol,
                maxiter=args.cg_maxiter,
                scaled_precond_state=applies["p_a_state"],
            )
            for alpha in jacobi_scale_sweep:
                stats = time_scaled_saddle_solve(
                    solve,
                    rhs_batch,
                    alpha,
                    solve_state=applies["p_a_state"],
                    rel_tol=report_rel_tol,
                )
                print(f"{alpha:<10.3g} {stats['avg_iters']:>8.1f} {stats['max_iters']:>7d} "
                      f"{stats['avg_ms']:>9.1f} {stats['max_residual']:>11.2e} "
                      f"{stats['n_fail']:>7d}/{stats['n_total']:<d}")

    if args.mass_benchmark:
        print()
        print("[diag] pure mass inversion M_k^{-1} (CG; dense surgery coupling "
              "on/off vs jacobi; k=3 has no surgery, so no coupling precompute)")
        # Rebuild the surgery-bearing mass preconditioners (k=0,1,2) with the
        # dense coupling precompute OFF on a functional copy of ``ops`` so both
        # variants live in one run. Bit-identical to ON (only the surgery
        # coupling apply path differs), so iteration counts must match; only
        # wall time should move. k=3 has no surgery split (nothing to densify).
        ops_mass_off = assemble_tensor_mass_preconditioner(
            seq, operators=ops, ks=(0, 1, 2), rank=rank,
            cp_kwargs={"precompute_coupling": False},
        )

        header = (
            f"{'mass precond':<40} {'avg_it':>8} {'it_std':>7} "
            f"{'avg_ms':>9} {'ms_std':>8} {'max_res':>11} {'fails':>7}"
        )
        print(header)
        print("-" * len(header))
        for k in (0, 1, 2, 3):
            n_k = int(getattr(seq, f"n{k}_dbc" if DIRICHLET else f"n{k}"))

            def _mk_matvec(v, k=k):
                return apply_mass_matrix(seq, ops, v, k, dirichlet=DIRICHLET)

            methods_k = [
                (f"k={k} jacobi (diag)",
                 lambda rhs, k=k: apply_mass_matrix_preconditioner(
                     seq, ops, rhs, k, dirichlet=DIRICHLET, kind="jacobi")),
                (f"k={k} tensor (coupling precompute ON)",
                 lambda rhs, k=k: apply_mass_matrix_preconditioner(
                     seq, ops, rhs, k, dirichlet=DIRICHLET, kind="tensor")),
            ]
            if k != 3:
                methods_k.append(
                    (f"k={k} tensor (coupling precompute OFF)",
                     lambda rhs, k=k: apply_mass_matrix_preconditioner(
                         seq, ops_mass_off, rhs, k, dirichlet=DIRICHLET, kind="tensor")))

            # b = M_k x_true keeps the RHS well-scaled (M_k is SPD, non-singular).
            mass_keys = jax.random.split(jax.random.PRNGKey(args.seed + k), args.n_rhs)
            rhs_batch_k = jnp.stack(
                [_mk_matvec(jax.random.normal(key, (n_k,), dtype=jnp.float64))
                 for key in mass_keys],
                axis=0,
            )
            for name, precond in methods_k:
                solve = make_solve(
                    _mk_matvec, _mk_matvec, precond,
                    tol=args.cg_tol, maxiter=args.cg_maxiter,
                )
                stats = time_solve(solve, rhs_batch_k, rel_tol=report_rel_tol)
                print(f"{name:<40} {stats['avg_iters']:>8.1f} {stats['std_iters']:>7.2f} "
                      f"{stats['avg_ms']:>9.1f} {stats['std_ms']:>8.2f} "
                      f"{stats['max_residual']:>11.2e} "
                      f"{stats['n_fail']:>7d}/{stats['n_total']:<d}")


if __name__ == "__main__":
    main()
