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
                    preconditioner assembled with ``schur_diag_mode='diag'``
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
import sys
import time
from typing import NamedTuple
from pathlib import Path

import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import rotating_ellipse_map, toroid_map
from mrx.differential_forms import safe_inv33
from mrx.operators import (
    _diagonal_from_matvec,
    _invert_diagonal,
    _get_schur_diaginv,
    _build_k1_stiffness_surgery_factors,
    apply_derivative_matrix,
    apply_laplacian_approx,
    apply_laplacian_preconditioner,
    apply_incidence_matrix,
    apply_mass_matrix,
    apply_mass_matrix_preconditioner,
    apply_stiffness_tensor_preconditioner,
    apply_stiffness,
    assemble_laplacian_operators,
    assemble_incidence_operators,
    assemble_mass_jacobi_preconditioner,
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
    _apply_k1_bulk_to_surgery_coupling,
    _apply_k1_surgery_to_bulk_coupling,
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
            use_inner_schur=bool(factors.use_inner_schur),
            vector_fd_state=vector_fd_state,
            vector_fd_true_basis_state=vector_fd_true_basis_state,
            radial_banded_state=radial_banded_state,
        )
        return _apply_k1_block_fd_bulk_from_state(state, rhs_bulk)

    # Assemble surgery Schur with the chosen bulk surrogate.
    n_s = int(surgery.surgery_size)
    eye_s = jnp.eye(n_s, dtype=jnp.float64)
    schur_dense = jax.vmap(
        lambda v: surgery.ass @ v
        - _apply_k1_bulk_to_surgery_coupling(
            surgery,
            _bulk_apply(_apply_k1_surgery_to_bulk_coupling(surgery, v)),
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
        use_inner_schur=bool(factors.use_inner_schur),
        vector_fd_state=vector_fd_state,
        vector_fd_true_basis_state=vector_fd_true_basis_state,
        radial_banded_state=radial_banded_state,
    )

    if not return_profile:
        return state
    return state, True


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
        if state.use_inner_schur
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
    z = state.schur_inv @ (rhs_s - _apply_k1_bulk_to_surgery_coupling(surgery, y))
    x_b = y - _apply_k1_block_fd_bulk_from_state(
        state,
        _apply_k1_surgery_to_bulk_coupling(surgery, z),
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

    coup_sb = _apply_k1_bulk_to_surgery_coupling(surgery, y)
    jax.block_until_ready(coup_sb)
    z = state.schur_inv @ (rhs_s - coup_sb)
    jax.block_until_ready(z)
    t2 = time.perf_counter()

    coup_bs = _apply_k1_surgery_to_bulk_coupling(surgery, z)
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

    if factors.use_inner_schur:
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

    if factors.use_inner_schur:
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

    if factors.use_inner_schur:
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
        - _apply_k1_bulk_to_surgery_coupling(
            surgery,
            bulk_apply(_apply_k1_surgery_to_bulk_coupling(surgery, v)),
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
        z = schur_inv @ (rhs_s - _apply_k1_bulk_to_surgery_coupling(surgery, y))
        x_b = y - bulk_apply(_apply_k1_surgery_to_bulk_coupling(surgery, z))
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
        timing_breakdown: bool = False):
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
        lambda o: assemble_tensor_mass_preconditioner(seq, operators=o, ks=(0, 1, 2, 3), rank=rank),
        ops,
    )
    ops = _timed(
        "tensor_laplacian",
        lambda o: assemble_tensor_laplacian_preconditioner(seq, operators=o, ks=(0,), rank=rank),
        ops,
    )
    # Always assemble k=1 tensor stiffness for P_A candidates.
    ops = _timed(
        "tensor_stiffness",
        lambda o: assemble_tensor_stiffness_preconditioner(
            seq,
            operators=o,
            ks=(1,),
            rank=rank,
            cp_kwargs={"k1_inner_schur": k1_stiff_inner_schur},
        ),
        ops,
    )
    # Production baseline: Schur-outer Jacobi diagonal, rank-independent diag mode.
    ops = _timed(
        "schur_jacobi",
        lambda o: assemble_schur_jacobi_preconditioner(
            seq,
            operators=o,
            ks=(1,),
            dirichlet_variants=(DIRICHLET,),
            schur_diag_mode='diag',
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
):
    """Return condensed/saddle applies and upper/lower preconditioners.

    When ``grad_project`` is True the active ``P_A`` is sandwiched between
    gradient-subspace complement projectors so it acts only on the curl-
    dominated complement, leaving the gradient (curl-free) subspace entirely
    to ``P_B``. This removes the additive double-counting where ``P_A`` and
    ``P_B`` both act on the gradient modes.
    """

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
        # Rank-1 tensor k=0 Hodge-Laplacian preconditioner: V0* -> V0.
        return apply_laplacian_preconditioner(
            seq, ops, x, 0, dirichlet=DIRICHLET, kind="tensor")

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
        precond_upper_state=None):
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
        "avg_ms": float(jnp.mean(jnp.asarray(times_ms))),
        "max_residual": float(max(residuals)),
        "n_fail": int(n_fail),
        "n_total": int(len(infos)),
    }


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ns", type=str, default="6,12,4",
                        help="Comma-separated (n_r,n_theta,n_zeta).")
    parser.add_argument("--p", type=int, default=3)
    parser.add_argument("--epsilon", type=float, default=1.0 / 3.0)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--r0", type=float, default=1.0)
    parser.add_argument(
        "--geometry", choices=("toroid", "rotating_ellipse"), default="toroid",
        help="Mapping: axisymmetric toroid (default) or symmetry-breaking "
             "rotating ellipse.")
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
        timing_breakdown=args.assembly_breakdown,
    )
    print(f"[diag] operators + rank-{rank} tensor preconditioners assembled in "
          f"{(time.perf_counter() - t0) * 1e3:.1f} ms")

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
        "P.T jacobi(S) P + P_B": (applies["projected_jacobi_plus_p_b"], None),
    }

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


if __name__ == "__main__":
    main()
