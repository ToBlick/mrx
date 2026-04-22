from __future__ import annotations

from typing import Optional, Sequence

import equinox as eqx
import jax.experimental.sparse as jsparse
import jax.numpy as jnp

from mrx.assembly import (assemble_scalar_tp, assemble_stiffness_scalar_tp,
                          assemble_vectorial_tp)
from mrx.solvers import solve_saddle_point_minres, solve_singular_cg
from mrx.utils import diag_EAET, diag_schur_complement


class SequenceOperators(eqx.Module):
    """Dynamic operator bundle for a de Rham sequence.

    Stores geometry-dependent operator data explicitly so it can be carried
    through JAX transforms while the sequence object remains a static topology
    shell.
    """

    m0_sp: Optional[jsparse.BCSR] = None
    m1_sp: Optional[jsparse.BCSR] = None
    m2_sp: Optional[jsparse.BCSR] = None
    m3_sp: Optional[jsparse.BCSR] = None
    m0_sp_diaginv: Optional[object] = None
    m1_sp_diaginv: Optional[object] = None
    m2_sp_diaginv: Optional[object] = None
    m3_sp_diaginv: Optional[object] = None
    m0_sp_diaginv_dbc: Optional[object] = None
    m1_sp_diaginv_dbc: Optional[object] = None
    m2_sp_diaginv_dbc: Optional[object] = None
    m3_sp_diaginv_dbc: Optional[object] = None
    d0_sp: Optional[jsparse.BCSR] = None
    d0_sp_T: Optional[jsparse.BCSR] = None
    d1_sp: Optional[jsparse.BCSR] = None
    d1_sp_T: Optional[jsparse.BCSR] = None
    d2_sp: Optional[jsparse.BCSR] = None
    d2_sp_T: Optional[jsparse.BCSR] = None
    grad_grad_sp: Optional[jsparse.BCSR] = None
    curl_curl_sp: Optional[jsparse.BCSR] = None
    div_div_sp: Optional[jsparse.BCSR] = None
    dd0_sp_diaginv: Optional[object] = None
    dd1_sp_diaginv: Optional[object] = None
    dd2_sp_diaginv: Optional[object] = None
    dd3_sp_diaginv: Optional[object] = None
    dd0_sp_diaginv_dbc: Optional[object] = None
    dd1_sp_diaginv_dbc: Optional[object] = None
    dd2_sp_diaginv_dbc: Optional[object] = None
    dd3_sp_diaginv_dbc: Optional[object] = None
    p21_sp: Optional[jsparse.BCSR] = None
    p12_sp: Optional[jsparse.BCSR] = None
    p03_sp: Optional[jsparse.BCSR] = None
    p30_sp: Optional[jsparse.BCSR] = None

    def todense(self, seq, operator: str, k, dirichlet: bool = True,
                transpose: bool = False):
        """Return a dense matrix for one assembled operator block."""
        match operator:
            case "mass":
                return dense_mass_matrix(seq, self, k, dirichlet=dirichlet)
            case "derivative":
                return dense_derivative_matrix(
                    seq, self, k,
                    dirichlet_in=dirichlet,
                    dirichlet_out=dirichlet,
                    transpose=transpose,
                )
            case "stiffness":
                return dense_stiffness_matrix(seq, self, k, dirichlet=dirichlet)
            case "hodge_laplacian":
                return dense_hodge_laplacian(seq, self, k, dirichlet=dirichlet)
            case "projection":
                if not isinstance(k, tuple) or len(k) != 2:
                    raise ValueError("Projection dense conversion expects k=(k_in, k_out)")
                return dense_projection_matrix(
                    seq, self, k[0], k[1],
                    dirichlet_in=dirichlet,
                    dirichlet_out=dirichlet,
                )
        raise ValueError(
            "operator must be one of 'mass', 'derivative', 'stiffness', 'hodge_laplacian', or 'projection'"
        )


def _assemble_mass_block(seq, geometry, k):
    quad_shape = (seq.quad.ny, seq.quad.nx, seq.quad.nz)

    match k:
        case 0:
            W_flat = geometry.jacobian_j * seq.quad.w
            sp = assemble_scalar_tp(
                seq.basis_r_jk, seq.basis_t_jk, seq.basis_z_jk,
                seq.basis_r_jk, seq.basis_t_jk, seq.basis_z_jk,
                W_flat, quad_shape, seq.basis_0.shape[0],
                seq.basis_0.pr, seq.basis_0.pt, seq.basis_0.pz)
            sp = jsparse.BCSR.from_bcoo(sp)
            diaginv = 1.0 / diag_EAET(seq.e0, sp, seq.e0_T)
            diaginv_dbc = 1.0 / diag_EAET(seq.e0_dbc, sp, seq.e0_dbc_T)
        case 1:
            W_3x3 = geometry.metric_inv_jkl * \
                (geometry.jacobian_j * seq.quad.w)[:, None, None]
            terms = [
                [(0, seq.d_basis_r_jk, seq.basis_t_jk, seq.basis_z_jk, +1)],
                [(1, seq.basis_r_jk, seq.d_basis_t_jk, seq.basis_z_jk, +1)],
                [(2, seq.basis_r_jk, seq.basis_t_jk, seq.d_basis_z_jk, +1)],
            ]
            sp = assemble_vectorial_tp(
                terms, terms, W_3x3, quad_shape,
                list(seq.basis_1.shape), seq.basis_1.pr)
            sp = jsparse.BCSR.from_bcoo(sp)
            diaginv = 1.0 / diag_EAET(seq.e1, sp, seq.e1_T)
            diaginv_dbc = 1.0 / diag_EAET(seq.e1_dbc, sp, seq.e1_dbc_T)
        case 2:
            W_3x3 = geometry.metric_jkl * \
                (1 / geometry.jacobian_j * seq.quad.w)[:, None, None]
            terms = [
                [(0, seq.basis_r_jk, seq.d_basis_t_jk, seq.d_basis_z_jk, +1)],
                [(1, seq.d_basis_r_jk, seq.basis_t_jk, seq.d_basis_z_jk, +1)],
                [(2, seq.d_basis_r_jk, seq.d_basis_t_jk, seq.basis_z_jk, +1)],
            ]
            sp = assemble_vectorial_tp(
                terms, terms, W_3x3, quad_shape,
                list(seq.basis_2.shape), seq.basis_2.pr)
            sp = jsparse.BCSR.from_bcoo(sp)
            diaginv = 1.0 / diag_EAET(seq.e2, sp, seq.e2_T)
            diaginv_dbc = 1.0 / diag_EAET(seq.e2_dbc, sp, seq.e2_dbc_T)
        case 3:
            W_flat = (1 / geometry.jacobian_j) * seq.quad.w
            sp = assemble_scalar_tp(
                seq.d_basis_r_jk, seq.d_basis_t_jk, seq.d_basis_z_jk,
                seq.d_basis_r_jk, seq.d_basis_t_jk, seq.d_basis_z_jk,
                W_flat, quad_shape, seq.basis_3.shape[0],
                seq.basis_3.pr, seq.basis_3.pt, seq.basis_3.pz)
            sp = jsparse.BCSR.from_bcoo(sp)
            diaginv = 1.0 / diag_EAET(seq.e3, sp, seq.e3_T)
            diaginv_dbc = 1.0 / diag_EAET(seq.e3_dbc, sp, seq.e3_dbc_T)
        case _:
            raise ValueError("k must be 0, 1, 2 or 3")

    return sp, diaginv, diaginv_dbc


def update_mass_operator(seq, geometry, operators: Optional[SequenceOperators], k: int):
    """Return an operator bundle with the k-th mass operator updated."""
    sp, diaginv, diaginv_dbc = _assemble_mass_block(seq, geometry, k)
    if operators is None:
        operators = SequenceOperators()

    match k:
        case 0:
            return eqx.tree_at(
                lambda ops: (ops.m0_sp, ops.m0_sp_diaginv, ops.m0_sp_diaginv_dbc),
                operators,
                (sp, diaginv, diaginv_dbc),
                is_leaf=lambda x: x is None,
            )
        case 1:
            return eqx.tree_at(
                lambda ops: (ops.m1_sp, ops.m1_sp_diaginv, ops.m1_sp_diaginv_dbc),
                operators,
                (sp, diaginv, diaginv_dbc),
                is_leaf=lambda x: x is None,
            )
        case 2:
            return eqx.tree_at(
                lambda ops: (ops.m2_sp, ops.m2_sp_diaginv, ops.m2_sp_diaginv_dbc),
                operators,
                (sp, diaginv, diaginv_dbc),
                is_leaf=lambda x: x is None,
            )
        case 3:
            return eqx.tree_at(
                lambda ops: (ops.m3_sp, ops.m3_sp_diaginv, ops.m3_sp_diaginv_dbc),
                operators,
                (sp, diaginv, diaginv_dbc),
                is_leaf=lambda x: x is None,
            )
    raise ValueError("k must be 0, 1, 2 or 3")


def assemble_mass_operators(seq, geometry, operators: Optional[SequenceOperators] = None,
                            ks: Sequence[int] = (0, 1, 2, 3)):
    """Assemble mass operators for the requested form degrees."""
    for k in ks:
        operators = update_mass_operator(seq, geometry, operators, k)
    return operators


def _mass_components(operators: SequenceOperators, k: int):
    match k:
        case 0:
            return operators.m0_sp, operators.m0_sp_diaginv, operators.m0_sp_diaginv_dbc
        case 1:
            return operators.m1_sp, operators.m1_sp_diaginv, operators.m1_sp_diaginv_dbc
        case 2:
            return operators.m2_sp, operators.m2_sp_diaginv, operators.m2_sp_diaginv_dbc
        case 3:
            return operators.m3_sp, operators.m3_sp_diaginv, operators.m3_sp_diaginv_dbc
    raise ValueError("k must be 0, 1, 2 or 3")


def _mass_diaginv(operators: SequenceOperators, k: int, dirichlet: bool):
    _, diaginv, diaginv_dbc = _mass_components(operators, k)
    selected = diaginv_dbc if dirichlet else diaginv
    if selected is None:
        raise ValueError(f"Mass preconditioner k={k} is not assembled")
    return selected


def _hodge_components(operators: SequenceOperators, k: int):
    match k:
        case 0:
            return operators.grad_grad_sp, operators.dd0_sp_diaginv, operators.dd0_sp_diaginv_dbc
        case 1:
            return operators.curl_curl_sp, operators.dd1_sp_diaginv, operators.dd1_sp_diaginv_dbc
        case 2:
            return operators.div_div_sp, operators.dd2_sp_diaginv, operators.dd2_sp_diaginv_dbc
        case 3:
            return None, operators.dd3_sp_diaginv, operators.dd3_sp_diaginv_dbc
    raise ValueError("k must be 0, 1, 2 or 3")


def _hodge_diaginv(operators: SequenceOperators, k: int, dirichlet: bool):
    _, diaginv, diaginv_dbc = _hodge_components(operators, k)
    selected = diaginv_dbc if dirichlet else diaginv
    if selected is None:
        raise ValueError(f"Hodge preconditioner k={k} is not assembled")
    return selected


def _derivative_components(operators: SequenceOperators, k: int):
    match k:
        case 0:
            return operators.d0_sp, operators.d0_sp_T
        case 1:
            return operators.d1_sp, operators.d1_sp_T
        case 2:
            return operators.d2_sp, operators.d2_sp_T
    raise ValueError("k must be 0, 1 or 2")


def _projection_components(operators: SequenceOperators, k_in: int, k_out: int):
    match (k_in, k_out):
        case (2, 1):
            return operators.p21_sp
        case (1, 2):
            return operators.p12_sp
        case (0, 3):
            return operators.p03_sp
        case (3, 0):
            return operators.p30_sp
    raise ValueError(
        "Only (k_in, k_out) = (1, 2), (2, 1), (0, 3), or (3, 0) supported"
    )


def _assemble_derivative_block(seq, geometry, k):
    quad_shape = (seq.quad.ny, seq.quad.nx, seq.quad.nz)
    types = seq.basis_0.types
    gr = seq._grad_1d(seq.d_basis_r_jk, types[0])
    gt = seq._grad_1d(seq.d_basis_t_jk, types[1])
    gz = seq._grad_1d(seq.d_basis_z_jk, types[2])

    match k:
        case 0:
            W_3x3 = geometry.metric_inv_jkl * \
                (geometry.jacobian_j * seq.quad.w)[:, None, None]
            row_terms = [
                [(0, seq.d_basis_r_jk, seq.basis_t_jk, seq.basis_z_jk, +1)],
                [(1, seq.basis_r_jk, seq.d_basis_t_jk, seq.basis_z_jk, +1)],
                [(2, seq.basis_r_jk, seq.basis_t_jk, seq.d_basis_z_jk, +1)],
            ]
            col_terms = [
                [(0, gr, seq.basis_t_jk, seq.basis_z_jk, +1),
                 (1, seq.basis_r_jk, gt, seq.basis_z_jk, +1),
                 (2, seq.basis_r_jk, seq.basis_t_jk, gz, +1)],
            ]
            sp = assemble_vectorial_tp(
                row_terms, col_terms, W_3x3, quad_shape,
                list(seq.basis_1.shape), seq.basis_1.pr,
                col_comp_shapes=list(seq.basis_0.shape))
        case 1:
            W_3x3 = geometry.metric_jkl * \
                (1 / geometry.jacobian_j * seq.quad.w)[:, None, None]
            dR = seq.d_basis_r_jk
            dT = seq.d_basis_t_jk
            dZ = seq.d_basis_z_jk
            R = seq.basis_r_jk
            T = seq.basis_t_jk
            Z = seq.basis_z_jk
            row_terms = [
                [(0, R, dT, dZ, +1)],
                [(1, dR, T, dZ, +1)],
                [(2, dR, dT, Z, +1)],
            ]
            col_terms = [
                [(1, dR, T, gz, +1),
                 (2, dR, gt, Z, -1)],
                [(0, R, dT, gz, -1),
                 (2, gr, dT, Z, +1)],
                [(0, R, gt, dZ, +1),
                 (1, gr, T, dZ, -1)],
            ]
            sp = assemble_vectorial_tp(
                row_terms, col_terms, W_3x3, quad_shape,
                list(seq.basis_2.shape), seq.basis_2.pr,
                col_comp_shapes=list(seq.basis_1.shape))
        case 2:
            W_scalar = (1 / geometry.jacobian_j) * seq.quad.w
            W_1x1 = W_scalar.reshape(-1, 1, 1)
            row_terms = [
                [(0, seq.d_basis_r_jk, seq.d_basis_t_jk, seq.d_basis_z_jk, +1)],
            ]
            col_terms = [
                [(0, gr, seq.d_basis_t_jk, seq.d_basis_z_jk, +1)],
                [(0, seq.d_basis_r_jk, gt, seq.d_basis_z_jk, +1)],
                [(0, seq.d_basis_r_jk, seq.d_basis_t_jk, gz, +1)],
            ]
            sp = assemble_vectorial_tp(
                row_terms, col_terms, W_1x1, quad_shape,
                list(seq.basis_3.shape), seq.basis_3.pr,
                col_comp_shapes=list(seq.basis_2.shape))
        case _:
            raise ValueError("k must be 0, 1 or 2")

    sp_T = jsparse.BCSR.from_bcoo(sp.T)
    sp = jsparse.BCSR.from_bcoo(sp)
    return sp, sp_T


def update_derivative_operator(seq, geometry, operators: Optional[SequenceOperators], k: int):
    """Return an operator bundle with the k-th weak derivative updated."""
    sp, sp_T = _assemble_derivative_block(seq, geometry, k)
    if operators is None:
        operators = SequenceOperators()

    match k:
        case 0:
            return eqx.tree_at(
                lambda ops: (ops.d0_sp, ops.d0_sp_T),
                operators,
                (sp, sp_T),
                is_leaf=lambda x: x is None,
            )
        case 1:
            return eqx.tree_at(
                lambda ops: (ops.d1_sp, ops.d1_sp_T),
                operators,
                (sp, sp_T),
                is_leaf=lambda x: x is None,
            )
        case 2:
            return eqx.tree_at(
                lambda ops: (ops.d2_sp, ops.d2_sp_T),
                operators,
                (sp, sp_T),
                is_leaf=lambda x: x is None,
            )
    raise ValueError("k must be 0, 1 or 2")


def assemble_derivative_operators(seq, geometry, operators: Optional[SequenceOperators] = None,
                                  ks: Sequence[int] = (0, 1, 2)):
    """Assemble weak derivative operators for the requested form degrees."""
    for k in ks:
        operators = update_derivative_operator(seq, geometry, operators, k)
    return operators


def _assemble_projection_block(seq, k_in: int, k_out: int):
    quad_shape = (seq.quad.ny, seq.quad.nx, seq.quad.nz)
    dR = seq.d_basis_r_jk
    dT = seq.d_basis_t_jk
    dZ = seq.d_basis_z_jk
    R = seq.basis_r_jk
    T = seq.basis_t_jk
    Z = seq.basis_z_jk

    match (k_in, k_out):
        case (2, 1):
            W_3x3 = seq.quad.w[:, None, None] * jnp.eye(3)
            row_terms = [
                [(0, dR, T, Z, +1)],
                [(1, R, dT, Z, +1)],
                [(2, R, T, dZ, +1)],
            ]
            col_terms = [
                [(0, R, dT, dZ, +1)],
                [(1, dR, T, dZ, +1)],
                [(2, dR, dT, Z, +1)],
            ]
            sp = assemble_vectorial_tp(
                row_terms, col_terms, W_3x3, quad_shape,
                list(seq.basis_1.shape), seq.basis_1.pr,
                col_comp_shapes=list(seq.basis_2.shape))
            return jsparse.BCSR.from_bcoo(sp)
        case (1, 2):
            W_3x3 = seq.quad.w[:, None, None] * jnp.eye(3)
            row_terms = [
                [(0, dR, T, Z, +1)],
                [(1, R, dT, Z, +1)],
                [(2, R, T, dZ, +1)],
            ]
            col_terms = [
                [(0, R, dT, dZ, +1)],
                [(1, dR, T, dZ, +1)],
                [(2, dR, dT, Z, +1)],
            ]
            sp = assemble_vectorial_tp(
                row_terms, col_terms, W_3x3, quad_shape,
                list(seq.basis_1.shape), seq.basis_1.pr,
                col_comp_shapes=list(seq.basis_2.shape))
            return jsparse.BCSR.from_bcoo(sp.T)
        case (0, 3):
            W_1x1 = seq.quad.w.reshape(-1, 1, 1)
            row_terms = [
                [(0, R, T, Z, +1)],
            ]
            col_terms = [
                [(0, dR, dT, dZ, +1)],
            ]
            sp = assemble_vectorial_tp(
                row_terms, col_terms, W_1x1, quad_shape,
                list(seq.basis_0.shape), seq.basis_0.pr,
                col_comp_shapes=list(seq.basis_3.shape))
            return jsparse.BCSR.from_bcoo(sp)
        case (3, 0):
            W_1x1 = seq.quad.w.reshape(-1, 1, 1)
            row_terms = [
                [(0, R, T, Z, +1)],
            ]
            col_terms = [
                [(0, dR, dT, dZ, +1)],
            ]
            sp = assemble_vectorial_tp(
                row_terms, col_terms, W_1x1, quad_shape,
                list(seq.basis_0.shape), seq.basis_0.pr,
                col_comp_shapes=list(seq.basis_3.shape))
            return jsparse.BCSR.from_bcoo(sp.T)
    raise ValueError(
        "Only (k_in, k_out) = (1, 2), (2, 1), (0, 3), or (3, 0) supported"
    )


def update_projection_operator(seq, operators: Optional[SequenceOperators],
                               k_in: int, k_out: int):
    """Return an operator bundle with the requested projection updated."""
    sp = _assemble_projection_block(seq, k_in, k_out)
    if operators is None:
        operators = SequenceOperators()

    match (k_in, k_out):
        case (2, 1):
            return eqx.tree_at(
                lambda ops: ops.p21_sp,
                operators,
                sp,
                is_leaf=lambda x: x is None,
            )
        case (1, 2):
            return eqx.tree_at(
                lambda ops: ops.p12_sp,
                operators,
                sp,
                is_leaf=lambda x: x is None,
            )
        case (0, 3):
            return eqx.tree_at(
                lambda ops: ops.p03_sp,
                operators,
                sp,
                is_leaf=lambda x: x is None,
            )
        case (3, 0):
            return eqx.tree_at(
                lambda ops: ops.p30_sp,
                operators,
                sp,
                is_leaf=lambda x: x is None,
            )
    raise ValueError(
        "Only (k_in, k_out) = (1, 2), (2, 1), (0, 3), or (3, 0) supported"
    )


def assemble_projection_operators(seq, operators: Optional[SequenceOperators] = None,
                                  pairs: Sequence[tuple[int, int]] = ((2, 1), (1, 2), (0, 3), (3, 0))):
    """Assemble projection operators for the requested degree pairs."""
    for k_in, k_out in pairs:
        operators = update_projection_operator(seq, operators, k_in, k_out)
    return operators


def _assemble_hodge_block(seq, geometry, operators: SequenceOperators, k):
    quad_shape = (seq.quad.ny, seq.quad.nx, seq.quad.nz)
    types = seq.basis_0.types
    gr = seq._grad_1d(seq.d_basis_r_jk, types[0])
    gt = seq._grad_1d(seq.d_basis_t_jk, types[1])
    gz = seq._grad_1d(seq.d_basis_z_jk, types[2])

    match k:
        case 0:
            W_3x3 = geometry.metric_inv_jkl * \
                (geometry.jacobian_j * seq.quad.w)[:, None, None]
            grad_basis_1d = [
                (gr, seq.basis_t_jk, seq.basis_z_jk),
                (seq.basis_r_jk, gt, seq.basis_z_jk),
                (seq.basis_r_jk, seq.basis_t_jk, gz),
            ]
            sp = assemble_stiffness_scalar_tp(
                grad_basis_1d, grad_basis_1d, W_3x3, quad_shape,
                seq.basis_0.shape[0],
                seq.basis_0.pr, seq.basis_0.pt, seq.basis_0.pz)
            sp = jsparse.BCSR.from_bcoo(sp)
            diaginv = 1.0 / diag_EAET(seq.e0, sp, seq.e0_T)
            diaginv_dbc = 1.0 / diag_EAET(seq.e0_dbc, sp, seq.e0_dbc_T)
        case 1:
            if operators.m0_sp_diaginv is None or operators.m0_sp_diaginv_dbc is None:
                raise ValueError("Assemble mass operator k=0 before Hodge operator k=1")
            W_3x3 = geometry.metric_jkl * \
                (1 / geometry.jacobian_j * seq.quad.w)[:, None, None]
            dR = seq.d_basis_r_jk
            dT = seq.d_basis_t_jk
            dZ = seq.d_basis_z_jk
            R = seq.basis_r_jk
            T = seq.basis_t_jk
            Z = seq.basis_z_jk
            curl_terms = [
                [(1, dR, T, gz, +1),
                 (2, dR, gt, Z, -1)],
                [(0, R, dT, gz, -1),
                 (2, gr, dT, Z, +1)],
                [(0, R, gt, dZ, +1),
                 (1, gr, T, dZ, -1)],
            ]
            sp = assemble_vectorial_tp(
                curl_terms, curl_terms, W_3x3, quad_shape,
                list(seq.basis_1.shape), seq.basis_1.pr)
            sp = jsparse.BCSR.from_bcoo(sp)
            d_stiff = diag_EAET(seq.e1, sp, seq.e1_T)
            d_schur = diag_schur_complement(
                lambda v: seq.e0 @ (operators.d0_sp_T @ (seq.e1_T @ v)),
                operators.m0_sp_diaginv, seq.n1)
            diaginv = 1.0 / (d_stiff + d_schur)
            d_stiff_dbc = diag_EAET(seq.e1_dbc, sp, seq.e1_dbc_T)
            d_schur_dbc = diag_schur_complement(
                lambda v: seq.e0_dbc @ (operators.d0_sp_T @ (seq.e1_dbc_T @ v)),
                operators.m0_sp_diaginv_dbc, seq.n1_dbc)
            diaginv_dbc = 1.0 / (d_stiff_dbc + d_schur_dbc)
        case 2:
            if operators.m1_sp_diaginv is None or operators.m1_sp_diaginv_dbc is None:
                raise ValueError("Assemble mass operator k=1 before Hodge operator k=2")
            W_scalar = (1 / geometry.jacobian_j) * seq.quad.w
            W_3x3 = W_scalar[:, None, None] * jnp.ones((1, 3, 3))
            div_terms = [
                [(0, gr, seq.d_basis_t_jk, seq.d_basis_z_jk, +1)],
                [(1, seq.d_basis_r_jk, gt, seq.d_basis_z_jk, +1)],
                [(2, seq.d_basis_r_jk, seq.d_basis_t_jk, gz, +1)],
            ]
            sp = assemble_vectorial_tp(
                div_terms, div_terms, W_3x3, quad_shape,
                list(seq.basis_2.shape), seq.basis_2.pr)
            sp = jsparse.BCSR.from_bcoo(sp)
            d_stiff = diag_EAET(seq.e2, sp, seq.e2_T)
            d_schur = diag_schur_complement(
                lambda v: seq.e1 @ (operators.d1_sp_T @ (seq.e2_T @ v)),
                operators.m1_sp_diaginv, seq.n2)
            diaginv = 1.0 / (d_stiff + d_schur)
            d_stiff_dbc = diag_EAET(seq.e2_dbc, sp, seq.e2_dbc_T)
            d_schur_dbc = diag_schur_complement(
                lambda v: seq.e1_dbc @ (operators.d1_sp_T @ (seq.e2_dbc_T @ v)),
                operators.m1_sp_diaginv_dbc, seq.n2_dbc)
            diaginv_dbc = 1.0 / (d_stiff_dbc + d_schur_dbc)
        case 3:
            if operators.m2_sp_diaginv is None or operators.m2_sp_diaginv_dbc is None:
                raise ValueError("Assemble mass operator k=2 before Hodge operator k=3")
            sp = None
            diaginv = 1.0 / diag_schur_complement(
                lambda v: seq.e2 @ (operators.d2_sp_T @ (seq.e3_T @ v)),
                operators.m2_sp_diaginv, seq.n3)
            diaginv_dbc = 1.0 / diag_schur_complement(
                lambda v: seq.e2_dbc @ (operators.d2_sp_T @ (seq.e3_dbc_T @ v)),
                operators.m2_sp_diaginv_dbc, seq.n3_dbc)
        case _:
            raise ValueError("k must be 0, 1, 2, or 3")

    return sp, diaginv, diaginv_dbc


def update_hodge_operator(seq, geometry, operators: Optional[SequenceOperators], k: int):
    """Return an operator bundle with the k-th Hodge/stiffness data updated."""
    if operators is None:
        operators = SequenceOperators()
    sp, diaginv, diaginv_dbc = _assemble_hodge_block(seq, geometry, operators, k)

    match k:
        case 0:
            return eqx.tree_at(
                lambda ops: (ops.grad_grad_sp, ops.dd0_sp_diaginv, ops.dd0_sp_diaginv_dbc),
                operators,
                (sp, diaginv, diaginv_dbc),
                is_leaf=lambda x: x is None,
            )
        case 1:
            return eqx.tree_at(
                lambda ops: (ops.curl_curl_sp, ops.dd1_sp_diaginv, ops.dd1_sp_diaginv_dbc),
                operators,
                (sp, diaginv, diaginv_dbc),
                is_leaf=lambda x: x is None,
            )
        case 2:
            return eqx.tree_at(
                lambda ops: (ops.div_div_sp, ops.dd2_sp_diaginv, ops.dd2_sp_diaginv_dbc),
                operators,
                (sp, diaginv, diaginv_dbc),
                is_leaf=lambda x: x is None,
            )
        case 3:
            return eqx.tree_at(
                lambda ops: (ops.dd3_sp_diaginv, ops.dd3_sp_diaginv_dbc),
                operators,
                (diaginv, diaginv_dbc),
                is_leaf=lambda x: x is None,
            )
    raise ValueError("k must be 0, 1, 2, or 3")


def assemble_hodge_operators(seq, geometry, operators: Optional[SequenceOperators] = None,
                             ks: Sequence[int] = (0, 1, 2, 3)):
    """Assemble Hodge/stiffness operators for the requested form degrees."""
    for k in ks:
        operators = update_hodge_operator(seq, geometry, operators, k)
    return operators


def assemble_all_operators(seq, geometry,
                           operators: Optional[SequenceOperators] = None):
    """Assemble all geometry-dependent mass, derivative, and Hodge operators."""
    operators = assemble_mass_operators(seq, geometry, operators=operators)
    operators = assemble_derivative_operators(
        seq, geometry, operators=operators)
    operators = assemble_hodge_operators(seq, geometry, operators=operators)
    operators = assemble_projection_operators(seq, operators=operators)
    return operators


def operators_from_coeffs(seq, coeffs,
                          ks: Sequence[int] = (0,),
                          kinds: Sequence[str] = ("mass", "derivative", "hodge")):
    """Build operators from spline-map coefficients.

    Routes ``coeffs`` through :meth:`DeRhamSequence.geometry_from_spline_map`
    and assembles only the requested operator ``kinds`` for the requested
    form degrees ``ks``. Useful as a pure function of ``coeffs`` for
    adjoint / shape-derivative workflows, where assembling the full
    operator bundle on every gradient call is wasteful.

    Parameters
    ----------
    seq : DeRhamSequence
    coeffs : (3, n_dof) array
        Cartesian spline coefficients defining the physical map.
    ks : sequence of int
        Form degrees to assemble (subset of ``(0, 1, 2, 3)``).
    kinds : sequence of str
        Any subset of ``("mass", "derivative", "hodge")``.

    Returns
    -------
    (operators, geometry) : (SequenceOperators, SequenceGeometry)
    """
    geometry = seq.geometry_from_spline_map(coeffs)
    ops: Optional[SequenceOperators] = None
    if "mass" in kinds:
        ops = assemble_mass_operators(seq, geometry, operators=ops, ks=ks)
    if "derivative" in kinds:
        ops = assemble_derivative_operators(seq, geometry, operators=ops, ks=ks)
    if "hodge" in kinds:
        ops = assemble_hodge_operators(seq, geometry, operators=ops, ks=ks)
    return ops, geometry


def _mass_extraction(seq, k: int, dirichlet: bool):
    match k:
        case 0:
            return (seq.e0_dbc, seq.e0_dbc_T) if dirichlet else (seq.e0, seq.e0_T)
        case 1:
            return (seq.e1_dbc, seq.e1_dbc_T) if dirichlet else (seq.e1, seq.e1_T)
        case 2:
            return (seq.e2_dbc, seq.e2_dbc_T) if dirichlet else (seq.e2, seq.e2_T)
        case 3:
            return (seq.e3_dbc, seq.e3_dbc_T) if dirichlet else (seq.e3, seq.e3_T)
    raise ValueError("k must be 0, 1, 2 or 3")


def _derivative_extraction(seq, k: int, dirichlet_in: bool, dirichlet_out: bool):
    match k:
        case 0:
            e_in = seq.e0_dbc if dirichlet_in else seq.e0
            e_in_T = seq.e0_dbc_T if dirichlet_in else seq.e0_T
            e_out = seq.e1_dbc if dirichlet_out else seq.e1
            e_out_T = seq.e1_dbc_T if dirichlet_out else seq.e1_T
        case 1:
            e_in = seq.e1_dbc if dirichlet_in else seq.e1
            e_in_T = seq.e1_dbc_T if dirichlet_in else seq.e1_T
            e_out = seq.e2_dbc if dirichlet_out else seq.e2
            e_out_T = seq.e2_dbc_T if dirichlet_out else seq.e2_T
        case 2:
            e_in = seq.e2_dbc if dirichlet_in else seq.e2
            e_in_T = seq.e2_dbc_T if dirichlet_in else seq.e2_T
            e_out = seq.e3_dbc if dirichlet_out else seq.e3
            e_out_T = seq.e3_dbc_T if dirichlet_out else seq.e3_T
        case _:
            raise ValueError("k must be 0, 1 or 2")
    return e_in, e_in_T, e_out, e_out_T


def dense_mass_matrix(seq, operators: SequenceOperators, k: int,
                      dirichlet: bool = True):
    """Return the dense extracted mass matrix for degree k."""
    sp, _, _ = _mass_components(operators, k)
    if sp is None:
        raise ValueError(f"Mass operator k={k} is not assembled")
    e, e_T = _mass_extraction(seq, k, dirichlet)
    return e.todense() @ sp.todense() @ e_T.todense()


def dense_derivative_matrix(seq, operators: SequenceOperators, k: int,
                            dirichlet_in: bool = True,
                            dirichlet_out: bool = True,
                            transpose: bool = False):
    """Return the dense extracted weak derivative matrix for degree k."""
    sp, sp_T = _derivative_components(operators, k)
    if sp is None or sp_T is None:
        raise ValueError(f"Derivative operator k={k} is not assembled")
    e_in, e_in_T, e_out, e_out_T = _derivative_extraction(
        seq, k, dirichlet_in, dirichlet_out)
    if transpose:
        return e_in.todense() @ sp_T.todense() @ e_out_T.todense()
    return e_out.todense() @ sp.todense() @ e_in_T.todense()


def dense_stiffness_matrix(seq, operators: SequenceOperators, k: int,
                           dirichlet: bool = True):
    """Return the dense extracted stiffness matrix for degree k."""
    if k == 3:
        n = seq.n3_dbc if dirichlet else seq.n3
        return jnp.zeros((n, n))
    sp, _, _ = _hodge_components(operators, k)
    if sp is None:
        raise ValueError(f"Hodge operator k={k} is not assembled")
    e, e_T = _mass_extraction(seq, k, dirichlet)
    return e.todense() @ sp.todense() @ e_T.todense()


def dense_hodge_laplacian(seq, operators: SequenceOperators, k: int,
                          dirichlet: bool = True):
    """Return the dense extracted Hodge Laplacian for degree k."""
    match k:
        case 0:
            return dense_stiffness_matrix(seq, operators, 0, dirichlet=dirichlet)
        case 1:
            stiffness = dense_stiffness_matrix(seq, operators, 1, dirichlet=dirichlet)
            derivative = dense_derivative_matrix(
                seq, operators, 0,
                dirichlet_in=dirichlet,
                dirichlet_out=dirichlet,
            )
            mass = dense_mass_matrix(seq, operators, 0, dirichlet=dirichlet)
            return stiffness + derivative @ jnp.linalg.solve(mass, derivative.T)
        case 2:
            stiffness = dense_stiffness_matrix(seq, operators, 2, dirichlet=dirichlet)
            derivative = dense_derivative_matrix(
                seq, operators, 1,
                dirichlet_in=dirichlet,
                dirichlet_out=dirichlet,
            )
            mass = dense_mass_matrix(seq, operators, 1, dirichlet=dirichlet)
            return stiffness + derivative @ jnp.linalg.solve(mass, derivative.T)
        case 3:
            derivative = dense_derivative_matrix(
                seq, operators, 2,
                dirichlet_in=dirichlet,
                dirichlet_out=dirichlet,
            )
            mass = dense_mass_matrix(seq, operators, 2, dirichlet=dirichlet)
            return derivative @ jnp.linalg.solve(mass, derivative.T)
    raise ValueError("k must be 0, 1, 2 or 3")


def dense_projection_matrix(seq, operators: SequenceOperators, k_in: int, k_out: int,
                            dirichlet_in: bool = True,
                            dirichlet_out: bool = True):
    """Return the dense extracted projection matrix for the requested degrees."""
    sp = _projection_components(operators, k_in, k_out)
    if sp is None:
        raise ValueError(
            f"Projection operator ({k_in}, {k_out}) is not assembled"
        )

    match (k_in, k_out):
        case (2, 1):
            e_in = seq.e2_dbc if dirichlet_in else seq.e2
            e_in_T = seq.e2_dbc_T if dirichlet_in else seq.e2_T
            e_out = seq.e1_dbc if dirichlet_out else seq.e1
        case (1, 2):
            e_in = seq.e1_dbc if dirichlet_in else seq.e1
            e_in_T = seq.e1_dbc_T if dirichlet_in else seq.e1_T
            e_out = seq.e2_dbc if dirichlet_out else seq.e2
        case (0, 3):
            e_in = seq.e3_dbc if dirichlet_in else seq.e3
            e_in_T = seq.e3_dbc_T if dirichlet_in else seq.e3_T
            e_out = seq.e0_dbc if dirichlet_out else seq.e0
        case (3, 0):
            e_in = seq.e0_dbc if dirichlet_in else seq.e0
            e_in_T = seq.e0_dbc_T if dirichlet_in else seq.e0_T
            e_out = seq.e3_dbc if dirichlet_out else seq.e3
        case _:
            raise ValueError(
                "Only (k_in, k_out) = (1, 2), (2, 1), (0, 3), or (3, 0) supported"
            )
    return e_out.todense() @ sp.todense() @ e_in_T.todense()


def apply_mass_matrix(seq, operators: SequenceOperators, v, k: int, dirichlet: bool = True):
    """Apply a mass matrix from an explicit operator bundle."""
    sp, _, _ = _mass_components(operators, k)
    if sp is None:
        raise ValueError(f"Mass operator k={k} is not assembled")

    match k:
        case 0:
            e = seq.e0_dbc if dirichlet else seq.e0
            e_T = seq.e0_dbc_T if dirichlet else seq.e0_T
        case 1:
            e = seq.e1_dbc if dirichlet else seq.e1
            e_T = seq.e1_dbc_T if dirichlet else seq.e1_T
        case 2:
            e = seq.e2_dbc if dirichlet else seq.e2
            e_T = seq.e2_dbc_T if dirichlet else seq.e2_T
        case 3:
            e = seq.e3_dbc if dirichlet else seq.e3
            e_T = seq.e3_dbc_T if dirichlet else seq.e3_T
        case _:
            raise ValueError("k must be 0, 1, 2 or 3")
    return e @ (sp @ (e_T @ v))


def apply_projection_matrix(seq, operators: SequenceOperators, v,
                            k_in: int, k_out: int,
                            dirichlet_in: bool = True,
                            dirichlet_out: bool = True):
    """Apply a projection matrix from an explicit operator bundle."""
    sp = _projection_components(operators, k_in, k_out)
    if sp is None:
        raise ValueError(
            f"Projection operator ({k_in}, {k_out}) is not assembled"
        )

    match (k_in, k_out):
        case (2, 1):
            e_in = seq.e2_dbc if dirichlet_in else seq.e2
            e_in_T = seq.e2_dbc_T if dirichlet_in else seq.e2_T
            e_out = seq.e1_dbc if dirichlet_out else seq.e1
        case (1, 2):
            e_in = seq.e1_dbc if dirichlet_in else seq.e1
            e_in_T = seq.e1_dbc_T if dirichlet_in else seq.e1_T
            e_out = seq.e2_dbc if dirichlet_out else seq.e2
        case (0, 3):
            e_in = seq.e3_dbc if dirichlet_in else seq.e3
            e_in_T = seq.e3_dbc_T if dirichlet_in else seq.e3_T
            e_out = seq.e0_dbc if dirichlet_out else seq.e0
        case (3, 0):
            e_in = seq.e0_dbc if dirichlet_in else seq.e0
            e_in_T = seq.e0_dbc_T if dirichlet_in else seq.e0_T
            e_out = seq.e3_dbc if dirichlet_out else seq.e3
        case _:
            raise ValueError(
                "Only (k_in, k_out) = (1, 2), (2, 1), (0, 3), or (3, 0) supported"
            )

    return e_out @ (sp @ (e_in_T @ v))


def apply_derivative_matrix(seq, operators: SequenceOperators, v, k: int,
                            dirichlet_in: bool = True,
                            dirichlet_out: bool = True,
                            transpose: bool = False):
    """Apply a weak derivative matrix from an explicit operator bundle."""
    sp, sp_T = _derivative_components(operators, k)
    if sp is None or sp_T is None:
        raise ValueError(f"Derivative operator k={k} is not assembled")

    match k:
        case 0:
            e_in = seq.e0_dbc if dirichlet_in else seq.e0
            e_in_T = seq.e0_dbc_T if dirichlet_in else seq.e0_T
            e_out = seq.e1_dbc if dirichlet_out else seq.e1
            e_out_T = seq.e1_dbc_T if dirichlet_out else seq.e1_T
        case 1:
            e_in = seq.e1_dbc if dirichlet_in else seq.e1
            e_in_T = seq.e1_dbc_T if dirichlet_in else seq.e1_T
            e_out = seq.e2_dbc if dirichlet_out else seq.e2
            e_out_T = seq.e2_dbc_T if dirichlet_out else seq.e2_T
        case 2:
            e_in = seq.e2_dbc if dirichlet_in else seq.e2
            e_in_T = seq.e2_dbc_T if dirichlet_in else seq.e2_T
            e_out = seq.e3_dbc if dirichlet_out else seq.e3
            e_out_T = seq.e3_dbc_T if dirichlet_out else seq.e3_T
        case _:
            raise ValueError("k must be 0, 1 or 2")

    if transpose:
        return e_in @ (sp_T @ (e_out_T @ v))
    return e_out @ (sp @ (e_in_T @ v))


def apply_mass_matrix_preconditioner(seq, operators: SequenceOperators, v, k: int,
                                     dirichlet: bool = True):
    """Apply the Jacobi preconditioner from an explicit operator bundle."""
    _, diaginv, diaginv_dbc = _mass_components(operators, k)
    if diaginv is None or diaginv_dbc is None:
        raise ValueError(f"Mass preconditioner k={k} is not assembled")
    return (diaginv_dbc if dirichlet else diaginv) * v


def apply_inverse_mass_matrix(seq, operators: SequenceOperators, v, k: int,
                              dirichlet: bool = True, guess=None,
                              tol: Optional[float] = None,
                              maxiter: Optional[int] = None):
    """Solve with the inverse mass matrix from an explicit operator bundle."""
    tol = seq.tol if tol is None else tol
    maxiter = seq.maxiter if maxiter is None else maxiter
    return solve_singular_cg(
        lambda x: apply_mass_matrix(seq, operators, x, k, dirichlet=dirichlet),
        v,
        mass_matvec=lambda x: apply_mass_matrix(seq, operators, x, k, dirichlet=dirichlet),
        precond_matvec=lambda x: apply_mass_matrix_preconditioner(
            seq, operators, x, k, dirichlet=dirichlet),
        x0=guess,
        tol=tol,
        maxiter=maxiter,
    )[0]


def apply_stiffness(seq, operators: SequenceOperators, v, k: int, dirichlet: bool = True):
    """Apply a stiffness matrix from an explicit operator bundle."""
    sp, _, _ = _hodge_components(operators, k)
    if k == 3:
        return jnp.zeros_like(v)
    if sp is None:
        raise ValueError(f"Hodge operator k={k} is not assembled")

    match k:
        case 0:
            e = seq.e0_dbc if dirichlet else seq.e0
            e_T = seq.e0_dbc_T if dirichlet else seq.e0_T
        case 1:
            e = seq.e1_dbc if dirichlet else seq.e1
            e_T = seq.e1_dbc_T if dirichlet else seq.e1_T
        case 2:
            e = seq.e2_dbc if dirichlet else seq.e2
            e_T = seq.e2_dbc_T if dirichlet else seq.e2_T
        case _:
            raise ValueError("k must be 0, 1, 2 or 3")
    return e @ (sp @ (e_T @ v))


def apply_hodge_laplacian_preconditioner(seq, operators: SequenceOperators, v, k: int,
                                         dirichlet: bool = True):
    """Apply the Hodge-Laplacian Jacobi preconditioner from an operator bundle."""
    return _hodge_diaginv(operators, k, dirichlet) * v


def apply_inverse_shifted_stiffness(seq, operators: SequenceOperators, v, k: int,
                                    eps: float, dirichlet: bool = True, guess=None,
                                    tol: Optional[float] = None,
                                    maxiter: Optional[int] = None):
    """Solve with the inverse of S_k + eps M_k using an explicit operator bundle."""
    tol = seq.tol if tol is None else tol
    maxiter = seq.maxiter if maxiter is None else maxiter
    stiffness_diaginv = _hodge_diaginv(operators, k, dirichlet)
    mass_diaginv = _mass_diaginv(operators, k, dirichlet)
    shifted_diaginv = 1.0 / (1.0 / stiffness_diaginv + eps / mass_diaginv)

    if k == 0:
        vs = seq._get_nullspace(0, dirichlet) if eps == 0 else []
        return solve_singular_cg(
            lambda x: apply_stiffness(seq, operators, x, 0, dirichlet=dirichlet)
            + eps * apply_mass_matrix(seq, operators, x, 0, dirichlet=dirichlet),
            v,
            mass_matvec=(
                lambda x: apply_mass_matrix(seq, operators, x, 0, dirichlet=dirichlet)
            ) if eps == 0 else None,
            precond_matvec=lambda x: shifted_diaginv * x,
            x0=guess,
            vs=vs,
            tol=tol,
            maxiter=maxiter,
        )[0]

    vs_upper, vs_lower = seq._get_saddle_point_nullspaces(
        k, dirichlet) if eps == 0 else ([], [])
    mass_lower_diaginv = _mass_diaginv(operators, k - 1, dirichlet)
    suffix = "_dbc" if dirichlet else ""
    n_upper = getattr(seq, f"n{k}{suffix}")
    n_lower = getattr(seq, f"n{k-1}{suffix}")

    def precond_upper(x):
        return shifted_diaginv * x

    def precond_lower(x):
        return mass_lower_diaginv * x

    u, sigma, info = solve_saddle_point_minres(
        stiffness_matvec=lambda x: apply_stiffness(seq, operators, x, k, dirichlet=dirichlet)
        + eps * apply_mass_matrix(seq, operators, x, k, dirichlet=dirichlet),
        derivative_matvec=lambda s: apply_derivative_matrix(
            seq, operators, s, k - 1, dirichlet_in=dirichlet, dirichlet_out=dirichlet),
        derivative_T_matvec=lambda u: apply_derivative_matrix(
            seq, operators, u, k - 1, dirichlet_in=dirichlet,
            dirichlet_out=dirichlet, transpose=True),
        mass_lower_matvec=lambda s: apply_mass_matrix(
            seq, operators, s, k - 1, dirichlet=dirichlet),
        b_upper=v,
        n_upper=n_upper,
        n_lower=n_lower,
        precond_upper=precond_upper,
        precond_lower=precond_lower,
        mass_upper_matvec=lambda x: apply_mass_matrix(
            seq, operators, x, k, dirichlet=dirichlet),
        vs_upper=vs_upper,
        vs_lower=vs_lower,
        x0_upper=guess,
        tol=tol,
        maxiter=maxiter,
    )
    return u


def apply_inverse_diffusion(seq, operators: SequenceOperators, v, k: int,
                            alpha: float, dirichlet: bool = True, guess=None,
                            tol: Optional[float] = None,
                            maxiter: Optional[int] = None):
    """Solve with the inverse of M_k + alpha L_k using an explicit operator bundle."""
    tol = seq.tol if tol is None else tol
    maxiter = seq.maxiter if maxiter is None else maxiter

    if k == 0:
        mass_diaginv = _mass_diaginv(operators, 0, dirichlet)
        stiffness_diaginv = _hodge_diaginv(operators, 0, dirichlet)
        diaginv = 1.0 / (1.0 / mass_diaginv + alpha / stiffness_diaginv)
        return solve_singular_cg(
            lambda x: apply_mass_matrix(seq, operators, x, 0, dirichlet=dirichlet)
            + alpha * apply_stiffness(seq, operators, x, 0, dirichlet=dirichlet),
            v,
            precond_matvec=lambda x: diaginv * x,
            x0=guess,
            tol=tol,
            maxiter=maxiter,
        )[0]

    mass_diaginv = _mass_diaginv(operators, k, dirichlet)
    mass_lower_diaginv = _mass_diaginv(operators, k - 1, dirichlet)
    suffix = "_dbc" if dirichlet else ""
    n_upper = getattr(seq, f"n{k}{suffix}")
    n_lower = getattr(seq, f"n{k-1}{suffix}")

    def precond_lower(x):
        return (1.0 / alpha) * mass_lower_diaginv * x

    def precond_upper(x):
        return mass_diaginv * x

    u, sigma, info = solve_saddle_point_minres(
        stiffness_matvec=lambda x: apply_mass_matrix(seq, operators, x, k, dirichlet=dirichlet)
        + alpha * apply_stiffness(seq, operators, x, k, dirichlet=dirichlet),
        derivative_matvec=lambda s: alpha * apply_derivative_matrix(
            seq, operators, s, k - 1, dirichlet_in=dirichlet, dirichlet_out=dirichlet),
        derivative_T_matvec=lambda u: alpha * apply_derivative_matrix(
            seq, operators, u, k - 1, dirichlet_in=dirichlet,
            dirichlet_out=dirichlet, transpose=True),
        mass_lower_matvec=lambda s: alpha * apply_mass_matrix(
            seq, operators, s, k - 1, dirichlet=dirichlet),
        b_upper=v,
        n_upper=n_upper,
        n_lower=n_lower,
        precond_upper=precond_upper,
        precond_lower=precond_lower,
        x0_upper=guess,
        tol=tol,
        maxiter=maxiter,
    )
    return u


def apply_hodge_laplacian(seq, operators: SequenceOperators, v, k: int,
                          dirichlet: bool = True, guess=None,
                          tol: Optional[float] = None,
                          maxiter: Optional[int] = None):
    """Apply the Hodge Laplacian using explicit operator data.

    This uses bundled mass, weak derivative, and stiffness operators.
    """
    match k:
        case 0:
            return apply_stiffness(seq, operators, v, 0, dirichlet=dirichlet)
        case 1:
            Dt_v = apply_derivative_matrix(
                seq, operators,
                v, 0, dirichlet_in=dirichlet, dirichlet_out=dirichlet, transpose=True)
            Minv_Dt_v = apply_inverse_mass_matrix(
                seq, operators, Dt_v, 0, dirichlet=dirichlet,
                guess=guess, tol=tol, maxiter=maxiter)
            return apply_stiffness(seq, operators, v, 1, dirichlet=dirichlet) + \
                apply_derivative_matrix(
                    seq, operators,
                    Minv_Dt_v, 0, dirichlet_in=dirichlet, dirichlet_out=dirichlet)
        case 2:
            Dt_v = apply_derivative_matrix(
                seq, operators,
                v, 1, dirichlet_in=dirichlet, dirichlet_out=dirichlet, transpose=True)
            Minv_Dt_v = apply_inverse_mass_matrix(
                seq, operators, Dt_v, 1, dirichlet=dirichlet,
                guess=guess, tol=tol, maxiter=maxiter)
            return apply_stiffness(seq, operators, v, 2, dirichlet=dirichlet) + \
                apply_derivative_matrix(
                    seq, operators,
                    Minv_Dt_v, 1, dirichlet_in=dirichlet, dirichlet_out=dirichlet)
        case 3:
            Dt_v = apply_derivative_matrix(
                seq, operators,
                v, 2, dirichlet_in=dirichlet, dirichlet_out=dirichlet, transpose=True)
            Minv_Dt_v = apply_inverse_mass_matrix(
                seq, operators, Dt_v, 2, dirichlet=dirichlet,
                guess=guess, tol=tol, maxiter=maxiter)
            return apply_derivative_matrix(
                seq, operators,
                Minv_Dt_v, 2, dirichlet_in=dirichlet, dirichlet_out=dirichlet)
        case _:
            raise ValueError("k must be 0, 1, 2 or 3")