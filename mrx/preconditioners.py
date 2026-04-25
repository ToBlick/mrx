from __future__ import annotations

from typing import Optional

import equinox as eqx
import jax.numpy as jnp

from mrx.utils import diag_EAET


class BoundaryConditionPair(eqx.Module):
    free: Optional[object] = None
    dbc: Optional[object] = None


class JacobiMassPreconditioner(eqx.Module):
    k0: BoundaryConditionPair = eqx.field(default_factory=BoundaryConditionPair)
    k1: BoundaryConditionPair = eqx.field(default_factory=BoundaryConditionPair)
    k2: BoundaryConditionPair = eqx.field(default_factory=BoundaryConditionPair)
    k3: BoundaryConditionPair = eqx.field(default_factory=BoundaryConditionPair)


class KroneckerMassPreconditioner(eqx.Module):
    m1d_inv_p_r: Optional[jnp.ndarray] = None
    m1d_inv_p_t: Optional[jnp.ndarray] = None
    m1d_inv_p_z: Optional[jnp.ndarray] = None
    m1d_inv_d_r: Optional[jnp.ndarray] = None
    m1d_inv_d_t: Optional[jnp.ndarray] = None
    m1d_inv_d_z: Optional[jnp.ndarray] = None
    k0_scale: Optional[jnp.ndarray] = None
    k1_scale: Optional[jnp.ndarray] = None
    k2_scale: Optional[jnp.ndarray] = None
    k3_scale: Optional[jnp.ndarray] = None


class ScalarRtZBlockInverseFactors(eqx.Module):
    k: int
    dirichlet: bool
    extracted_size: int
    full_shape: tuple[int, int, int]
    rt_size: int
    z_size: int
    z_basis: jnp.ndarray
    z_eigenvalues: jnp.ndarray
    rt_blocks: tuple[jnp.ndarray, ...]
    rt_block_inverses: tuple[jnp.ndarray, ...]


class ExtractedBulkRtZBlockInverseFactors(eqx.Module):
    dirichlet: bool
    core_size: int
    bulk_size: int
    bulk_shape: tuple[int, int, int]
    matrix: jnp.ndarray
    acb: jnp.ndarray
    abc: jnp.ndarray
    schur_inv: jnp.ndarray
    z_basis: jnp.ndarray
    z_eigenvalues: jnp.ndarray
    rt_blocks: tuple[jnp.ndarray, ...]
    rt_block_inverses: tuple[jnp.ndarray, ...]


class RtZBlockMassPreconditioner(eqx.Module):
    k0: BoundaryConditionPair = eqx.field(default_factory=BoundaryConditionPair)
    k3: BoundaryConditionPair = eqx.field(default_factory=BoundaryConditionPair)


class MassPreconditioners(eqx.Module):
    jacobi: Optional[JacobiMassPreconditioner] = None
    kronecker: Optional[KroneckerMassPreconditioner] = None
    rtzblock: Optional[RtZBlockMassPreconditioner] = None


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


def set_mass_kronecker(preconds: Optional[MassPreconditioners], data: KroneckerMassPreconditioner):
    if preconds is None:
        preconds = MassPreconditioners()
    return eqx.tree_at(
        lambda payload: payload.kronecker,
        preconds,
        data,
        is_leaf=lambda x: x is None,
    )


def set_mass_rtzblock_factor(preconds: Optional[MassPreconditioners], k: int, dirichlet: bool, factor_data):
    if preconds is None:
        preconds = MassPreconditioners()
    rtzblock = preconds.rtzblock if preconds.rtzblock is not None else RtZBlockMassPreconditioner()
    pair = rtzblock.k0 if k == 0 else rtzblock.k3
    pair = eqx.tree_at(
        lambda boundary_pair: boundary_pair.dbc if dirichlet else boundary_pair.free,
        pair,
        factor_data,
        is_leaf=lambda x: x is None,
    )
    if k == 0:
        rtzblock = eqx.tree_at(lambda data: data.k0, rtzblock, pair)
    elif k == 3:
        rtzblock = eqx.tree_at(lambda data: data.k3, rtzblock, pair)
    else:
        raise ValueError("rt-zblock mass preconditioner only supports k=0 and k=3")
    return eqx.tree_at(
        lambda payload: payload.rtzblock,
        preconds,
        rtzblock,
        is_leaf=lambda x: x is None,
    )


def invalidate_mass_rtzblock(preconds: Optional[MassPreconditioners], k: int):
    if preconds is None or preconds.rtzblock is None:
        return preconds
    empty_pair = BoundaryConditionPair()
    if k == 0:
        rtzblock = eqx.tree_at(lambda data: data.k0, preconds.rtzblock, empty_pair)
    elif k == 3:
        rtzblock = eqx.tree_at(lambda data: data.k3, preconds.rtzblock, empty_pair)
    else:
        return preconds
    return eqx.tree_at(lambda payload: payload.rtzblock, preconds, rtzblock)


def build_mass_jacobi_pair(seq, mass_sp, k: int) -> BoundaryConditionPair:
    e = getattr(seq, f"e{k}")
    e_t = getattr(seq, f"e{k}_T")
    e_dbc = getattr(seq, f"e{k}_dbc")
    e_t_dbc = getattr(seq, f"e{k}_dbc_T")
    return BoundaryConditionPair(
        free=1.0 / diag_EAET(e, mass_sp, e_t),
        dbc=1.0 / diag_EAET(e_dbc, mass_sp, e_t_dbc),
    )


def _assemble_1d_mass_inverse(B, w):
    M = (B * w[None, :]) @ B.T
    M_inv = jnp.linalg.inv(M)
    return 0.5 * (M_inv + M_inv.T)


def _kron_geometric_scales(seq, k: int):
    geometry = seq.geometry
    w = seq.quad.w
    w_sum = jnp.sum(w)
    J = geometry.jacobian_j
    match k:
        case 0:
            return jnp.array([jnp.sum(J * w) / w_sum])
        case 1:
            g_inv = geometry.metric_inv_jkl
            return jnp.array([
                jnp.sum(J * g_inv[:, i, i] * w) / w_sum for i in range(3)
            ])
        case 2:
            g = geometry.metric_jkl
            return jnp.array([
                jnp.sum(g[:, i, i] / J * w) / w_sum for i in range(3)
            ])
        case 3:
            return jnp.array([jnp.sum(w / J) / w_sum])
    raise ValueError("k must be 0, 1, 2 or 3")


def build_mass_kronecker_preconditioner(seq) -> KroneckerMassPreconditioner:
    return KroneckerMassPreconditioner(
        m1d_inv_p_r=_assemble_1d_mass_inverse(seq.basis_r_jk, seq.quad.w_x),
        m1d_inv_p_t=_assemble_1d_mass_inverse(seq.basis_t_jk, seq.quad.w_y),
        m1d_inv_p_z=_assemble_1d_mass_inverse(seq.basis_z_jk, seq.quad.w_z),
        m1d_inv_d_r=_assemble_1d_mass_inverse(seq.d_basis_r_jk, seq.quad.w_x),
        m1d_inv_d_t=_assemble_1d_mass_inverse(seq.d_basis_t_jk, seq.quad.w_y),
        m1d_inv_d_z=_assemble_1d_mass_inverse(seq.d_basis_z_jk, seq.quad.w_z),
        k0_scale=_kron_geometric_scales(seq, 0),
        k1_scale=_kron_geometric_scales(seq, 1),
        k2_scale=_kron_geometric_scales(seq, 2),
        k3_scale=_kron_geometric_scales(seq, 3),
    )


def _kron_component_specs(seq, k: int):
    nr_p = seq.basis_r_jk.shape[0]
    nt_p = seq.basis_t_jk.shape[0]
    nz_p = seq.basis_z_jk.shape[0]
    nr_d = seq.d_basis_r_jk.shape[0]
    nt_d = seq.d_basis_t_jk.shape[0]
    nz_d = seq.d_basis_z_jk.shape[0]
    if k == 0:
        return [((nr_p, nt_p, nz_p), ("p", "p", "p"))]
    if k == 1:
        return [
            ((nr_d, nt_p, nz_p), ("d", "p", "p")),
            ((nr_p, nt_d, nz_p), ("p", "d", "p")),
            ((nr_p, nt_p, nz_d), ("p", "p", "d")),
        ]
    if k == 2:
        return [
            ((nr_p, nt_d, nz_d), ("p", "d", "d")),
            ((nr_d, nt_p, nz_d), ("d", "p", "d")),
            ((nr_d, nt_d, nz_p), ("d", "d", "p")),
        ]
    if k == 3:
        return [((nr_d, nt_d, nz_d), ("d", "d", "d"))]
    raise ValueError("k must be 0, 1, 2 or 3")


def _kron_inv_table(data: KroneckerMassPreconditioner):
    return {
        ("p", "r"): data.m1d_inv_p_r,
        ("p", "t"): data.m1d_inv_p_t,
        ("p", "z"): data.m1d_inv_p_z,
        ("d", "r"): data.m1d_inv_d_r,
        ("d", "t"): data.m1d_inv_d_t,
        ("d", "z"): data.m1d_inv_d_z,
    }


def mass_kronecker_available(seq, preconds: Optional[MassPreconditioners], k: int) -> bool:
    if preconds is None or preconds.kronecker is None:
        return False
    table = _kron_inv_table(preconds.kronecker)
    needed = set()
    for _, kinds in _kron_component_specs(seq, k):
        for axis, kind in zip("rtz", kinds):
            needed.add((kind, axis))
    return all(table[key] is not None for key in needed)


def _kron_scale_for_k(data: KroneckerMassPreconditioner, k: int):
    match k:
        case 0:
            return data.k0_scale
        case 1:
            return data.k1_scale
        case 2:
            return data.k2_scale
        case 3:
            return data.k3_scale
    raise ValueError("k must be 0, 1, 2 or 3")


def _kron_apply_3d(Mr_inv, Mt_inv, Mz_inv, x):
    x = jnp.einsum("ij,jkl->ikl", Mr_inv, x)
    x = jnp.einsum("ij,kjl->kil", Mt_inv, x)
    x = jnp.einsum("ij,klj->kli", Mz_inv, x)
    return x


def _kron_apply_full(seq, data: KroneckerMassPreconditioner, v_full, k: int):
    table = _kron_inv_table(data)
    specs = _kron_component_specs(seq, k)
    scales = _kron_scale_for_k(data, k)
    parts = []
    offset = 0
    for i, (shape, kinds) in enumerate(specs):
        size = shape[0] * shape[1] * shape[2]
        x = v_full[offset:offset + size].reshape(shape)
        Mr_inv = table[(kinds[0], "r")]
        Mt_inv = table[(kinds[1], "t")]
        Mz_inv = table[(kinds[2], "z")]
        y = _kron_apply_3d(Mr_inv, Mt_inv, Mz_inv, x)
        if scales is not None:
            y = y / scales[i]
        parts.append(y.reshape(-1))
        offset += size
    return jnp.concatenate(parts) if len(parts) > 1 else parts[0]


def apply_mass_kronecker_preconditioner(seq, preconds: Optional[MassPreconditioners], v, k: int, dirichlet: bool = True):
    if not mass_kronecker_available(seq, preconds, k):
        raise ValueError(f"Kronecker mass preconditioner not assembled for k={k}")
    if dirichlet:
        e = getattr(seq, f"e{k}_dbc")
        e_t = getattr(seq, f"e{k}_dbc_T")
    else:
        e = getattr(seq, f"e{k}")
        e_t = getattr(seq, f"e{k}_T")
    v_full = e_t @ v
    y_full = _kron_apply_full(seq, preconds.kronecker, v_full, k)
    return e @ y_full


def _symmetrize(matrix: jnp.ndarray) -> jnp.ndarray:
    return 0.5 * (matrix + matrix.T)


def _quadrature_weight_tensor(seq, k: int) -> jnp.ndarray:
    if k == 0:
        weight_flat = seq.geometry.jacobian_j
    elif k == 3:
        weight_flat = 1.0 / seq.geometry.jacobian_j
    else:
        raise ValueError("Scalar helpers only support k=0 and k=3")
    quad_shape = (seq.quad.ny, seq.quad.nx, seq.quad.nz)
    return weight_flat.reshape(quad_shape).transpose(1, 0, 2)


def _scalar_basis_triplet(seq, k: int):
    if k == 0:
        return seq.basis_r_jk, seq.basis_t_jk, seq.basis_z_jk
    if k == 3:
        return seq.d_basis_r_jk, seq.d_basis_t_jk, seq.d_basis_z_jk
    raise ValueError("Scalar helpers only support k=0 and k=3")


def _assemble_weighted_1d_mass(B: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    return (B * weights[None, :]) @ B.T


def _scalar_outer_z_basis_from_weight(seq, k: int):
    basis_z = _scalar_basis_triplet(seq, k)[2]
    weight_tensor = _quadrature_weight_tensor(seq, k)
    nr_q, nt_q, nz_q = weight_tensor.shape
    outer_matrix = weight_tensor.reshape(nr_q * nt_q, nz_q)
    _, _, vh = jnp.linalg.svd(outer_matrix, full_matrices=False)
    z_terms = []
    for idx in range(vh.shape[0]):
        z_terms.append(_assemble_weighted_1d_mass(basis_z, seq.quad.w_z * vh[idx, :]))
    z_average = _symmetrize(sum(z_terms))
    evals, evecs = jnp.linalg.eigh(z_average)
    order = jnp.argsort(evals)[::-1]
    return evals[order], evecs[:, order]


def _scalar_rt_z_tensor_from_matrix(matrix: jnp.ndarray, full_shape: tuple[int, int, int]) -> jnp.ndarray:
    nr, nt, nz = full_shape
    rt_size = nr * nt
    return matrix.reshape(nr, nt, nz, nr, nt, nz).transpose(0, 1, 3, 4, 2, 5).reshape(
        rt_size, rt_size, nz, nz
    )


def _apply_rt_z_block_inverse(rt_block_inverses: tuple[jnp.ndarray, ...], z_basis: jnp.ndarray, rhs: jnp.ndarray, rt_size: int) -> jnp.ndarray:
    x = rhs.reshape(rt_size, z_basis.shape[0])
    x_hat = x @ z_basis
    y_hat = jnp.stack([block_inv @ x_hat[:, idx] for idx, block_inv in enumerate(rt_block_inverses)], axis=1)
    y = y_hat @ z_basis.T
    return y.reshape(-1)


def build_mass_rtzblock_k3_factors(seq, full_matrix: jnp.ndarray, dirichlet: bool = False) -> ScalarRtZBlockInverseFactors:
    full_shape = seq.basis_3.shape[0]
    rt_size = full_shape[0] * full_shape[1]
    z_size = full_shape[2]
    tensor = _scalar_rt_z_tensor_from_matrix(full_matrix, full_shape)
    z_eigenvalues, z_basis = _scalar_outer_z_basis_from_weight(seq, 3)

    rt_blocks = []
    rt_block_inverses = []
    for idx in range(z_size):
        q = z_basis[:, idx]
        block = _symmetrize(jnp.einsum("ijab,a,b->ij", tensor, q, q))
        rt_blocks.append(block)
        rt_block_inverses.append(jnp.linalg.inv(block))

    e = getattr(seq, "e3_dbc" if dirichlet else "e3")
    return ScalarRtZBlockInverseFactors(
        k=3,
        dirichlet=dirichlet,
        extracted_size=e.shape[0],
        full_shape=full_shape,
        rt_size=rt_size,
        z_size=z_size,
        z_basis=z_basis,
        z_eigenvalues=z_eigenvalues,
        rt_blocks=tuple(rt_blocks),
        rt_block_inverses=tuple(rt_block_inverses),
    )


def _core_size(seq) -> int:
    return 3 * seq.basis_0.nz


def _bulk_tensor_shape(seq, dirichlet: bool) -> tuple[int, int, int]:
    nr_bulk = seq.basis_0.nr - 2 - int(dirichlet)
    nt = seq.basis_0.nt
    nz = seq.basis_0.nz
    return nr_bulk, nt, nz


def _split_blocks(matrix: jnp.ndarray, core_size: int):
    acc = matrix[:core_size, :core_size]
    acb = matrix[:core_size, core_size:]
    abc = matrix[core_size:, :core_size]
    abb = matrix[core_size:, core_size:]
    return acc, acb, abc, abb


def build_mass_rtzblock_k0_factors(seq, full_matrix: jnp.ndarray, dirichlet: bool = False) -> ExtractedBulkRtZBlockInverseFactors:
    e = getattr(seq, "e0_dbc" if dirichlet else "e0").todense()
    e_t = getattr(seq, "e0_dbc_T" if dirichlet else "e0_T").todense()
    matrix = e @ full_matrix @ e_t
    core_size = _core_size(seq)
    acc, acb, abc, abb = _split_blocks(matrix, core_size)
    bulk_shape = _bulk_tensor_shape(seq, dirichlet)
    rt_size = bulk_shape[0] * bulk_shape[1]
    tensor = _scalar_rt_z_tensor_from_matrix(abb, bulk_shape)

    z_average = _symmetrize(jnp.einsum("ijab->ab", tensor) / rt_size)
    z_eigenvalues, z_basis = jnp.linalg.eigh(z_average)
    order = jnp.argsort(z_eigenvalues)[::-1]
    z_eigenvalues = z_eigenvalues[order]
    z_basis = z_basis[:, order]

    rt_blocks = []
    rt_block_inverses = []
    for idx in range(bulk_shape[2]):
        q = z_basis[:, idx]
        block = _symmetrize(jnp.einsum("ijab,a,b->ij", tensor, q, q))
        rt_blocks.append(block)
        rt_block_inverses.append(jnp.linalg.inv(block))

    u_cols = [
        _apply_rt_z_block_inverse(tuple(rt_block_inverses), z_basis, abc[:, j], rt_size)
        for j in range(abc.shape[1])
    ]
    u = jnp.stack(u_cols, axis=1)
    schur = acc - acb @ u
    schur_inv = jnp.linalg.inv(schur)

    return ExtractedBulkRtZBlockInverseFactors(
        dirichlet=dirichlet,
        core_size=core_size,
        bulk_size=matrix.shape[0] - core_size,
        bulk_shape=bulk_shape,
        matrix=matrix,
        acb=acb,
        abc=abc,
        schur_inv=schur_inv,
        z_basis=z_basis,
        z_eigenvalues=z_eigenvalues,
        rt_blocks=tuple(rt_blocks),
        rt_block_inverses=tuple(rt_block_inverses),
    )


def _apply_scalar_rt_z_block_inverse_factors(seq, factors: ScalarRtZBlockInverseFactors, rhs: jnp.ndarray) -> jnp.ndarray:
    e = getattr(seq, f"e{factors.k}_dbc" if factors.dirichlet else f"e{factors.k}").todense()
    e_t = getattr(seq, f"e{factors.k}_dbc_T" if factors.dirichlet else f"e{factors.k}_T").todense()
    full_rhs = e_t @ rhs
    return e @ _apply_rt_z_block_inverse(factors.rt_block_inverses, factors.z_basis, full_rhs, factors.rt_size)


def _apply_extracted_bulk_rt_z_block_inverse(factors: ExtractedBulkRtZBlockInverseFactors, rhs: jnp.ndarray) -> jnp.ndarray:
    rhs_c = rhs[:factors.core_size]
    rhs_b = rhs[factors.core_size:]
    rt_size = factors.bulk_shape[0] * factors.bulk_shape[1]
    y = _apply_rt_z_block_inverse(factors.rt_block_inverses, factors.z_basis, rhs_b, rt_size)
    z = factors.schur_inv @ (rhs_c - factors.acb @ y)
    x_b = y - _apply_rt_z_block_inverse(factors.rt_block_inverses, factors.z_basis, factors.abc @ z, rt_size)
    return jnp.concatenate([z, x_b])


def apply_mass_rtzblock_preconditioner(seq, preconds: Optional[MassPreconditioners], rhs: jnp.ndarray, k: int, dirichlet: bool = True) -> jnp.ndarray:
    if preconds is None or preconds.rtzblock is None:
        raise ValueError("rt-zblock mass preconditioner is not assembled")
    if k == 0:
        factors = select_boundary_data(preconds.rtzblock.k0, dirichlet, "rt-zblock mass k=0")
        return _apply_extracted_bulk_rt_z_block_inverse(factors, rhs)
    if k == 3:
        factors = select_boundary_data(preconds.rtzblock.k3, dirichlet, "rt-zblock mass k=3")
        return _apply_scalar_rt_z_block_inverse_factors(seq, factors, rhs)
    raise ValueError("rt-zblock mass preconditioner only supports k=0 and k=3")