# %% [markdown]
# # k=0 Polar Mass Surgery Debug
#
# This interactive script mirrors the `k=1` workflow on the scalar `k=0`
# extracted mass matrix.
#
# The relevant extracted split is
#
#     A = E_0 M_{0,raw} E_0^T = [[A_cc, A_cb], [A_bc, A_bb]],
#
# where `A_cc` is the small fused polar core and `A_bb` is the extracted bulk.
#
# The immediate questions are:
#
# 1. how well the `rt|z` inverse class approximates `A_bb^{-1}`,
# 2. how much core-bulk coupling matters,
# 3. and how accurate the resulting Schur model is on the full extracted matrix.

# %%
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import rotating_ellipse_map, toroid_map
from mrx.operators import assemble_mass_operators, dense_mass_matrix

jax.config.update("jax_enable_x64", True)


# %% Configuration
@dataclass(frozen=True)
class ExperimentConfig:
    ns: tuple[int, int, int] = (8, 8, 8)
    p: int = 2
    tol: float = 1e-10
    maxiter: int = 2000
    betti: tuple[int, int, int, int] = (1, 1, 0, 0)
    map_kind: str = "rotating_ellipse"
    torus_epsilon: float = 1.0 / 3.0
    torus_r0: float = 1.0
    rotating_eps: float = 0.2
    rotating_kappa: float = 1.1
    rotating_r0: float = 1.0
    rotating_nfp: int = 2


@dataclass
class RtZInverseDiagnostics:
    relative_operator_error: float
    relative_solve_error: float
    relative_residual: float
    min_eigenvalue: float
    max_eigenvalue: float


@dataclass
class CoupledBlockDiagnostics:
    offdiag_relative_frobenius: float
    exact_blockdiag_operator_error: float
    mixed_blockdiag_operator_error: float
    exact_blockdiag_solve_error: float
    mixed_blockdiag_solve_error: float
    exact_blockdiag_residual: float
    mixed_blockdiag_residual: float


@dataclass
class CoupledSchurDiagnostics:
    exact_schur_operator_error: float
    mixed_schur_operator_error: float
    exact_schur_solve_error: float
    mixed_schur_solve_error: float
    exact_schur_residual: float
    mixed_schur_residual: float


@dataclass
class CoreSchurCorrectionDiagnostics:
    exact_correction_relative_to_acc: float
    mixed_correction_relative_to_acc: float
    mixed_vs_exact_correction_error: float
    exact_schur_relative_to_acc: float
    mixed_schur_vs_exact_error: float


CONFIG = ExperimentConfig()
SEQ = None
OPERATORS = None
BUILT_CONFIG = None


# %% Helpers
def _build_map(config: ExperimentConfig):
    if config.map_kind == "toroidal":
        return toroid_map(epsilon=config.torus_epsilon, R0=config.torus_r0)
    if config.map_kind == "rotating_ellipse":
        return rotating_ellipse_map(
            eps=config.rotating_eps,
            kappa=config.rotating_kappa,
            R0=config.rotating_r0,
            nfp=config.rotating_nfp,
        )
    raise ValueError(f"Unsupported map kind: {config.map_kind}")


def build_case(config: ExperimentConfig = CONFIG):
    seq = DeRhamSequence(
        config.ns,
        (config.p, config.p, config.p),
        2 * config.p,
        ("clamped", "periodic", "periodic"),
        lambda x: x,
        polar=True,
        tol=config.tol,
        maxiter=config.maxiter,
        betti_numbers=config.betti,
    )
    seq.evaluate_1d()
    seq.assemble_reference_mass_matrix()
    seq.set_map(_build_map(config))
    operators = assemble_mass_operators(seq, seq.geometry, ks=(0,))
    return seq, operators


def ensure_built(config: ExperimentConfig = CONFIG, rebuild: bool = False):
    global SEQ, OPERATORS, BUILT_CONFIG
    if rebuild or SEQ is None or OPERATORS is None or BUILT_CONFIG != config:
        SEQ, OPERATORS = build_case(config)
        BUILT_CONFIG = config
    return SEQ, OPERATORS


def dense_extraction_matrix(seq, dirichlet: bool) -> jnp.ndarray:
    e = seq.e0_dbc if dirichlet else seq.e0
    return jnp.asarray(e.todense())


def dense_raw_mass_matrix(operators) -> jnp.ndarray:
    return jnp.asarray(operators.m0_sp.todense())


def dense_extracted_mass_matrix(seq, operators, dirichlet: bool) -> jnp.ndarray:
    return jnp.asarray(dense_mass_matrix(seq, operators, 0, dirichlet=dirichlet))


def verify_extracted_mass_identity(seq, operators, dirichlet: bool):
    e = dense_extraction_matrix(seq, dirichlet)
    m_raw = dense_raw_mass_matrix(operators)
    a_ref = dense_extracted_mass_matrix(seq, operators, dirichlet)
    a_sandwich = e @ m_raw @ e.T
    diff = jnp.max(jnp.abs(a_ref - a_sandwich))
    return a_ref, a_sandwich, float(diff)


def block_norm_table(matrix: jnp.ndarray, slices: dict[str, slice]):
    labels = list(slices.keys())
    total = float(jnp.linalg.norm(matrix))
    table = []
    for row_label in labels:
        row = []
        for col_label in labels:
            block = matrix[slices[row_label], slices[col_label]]
            fro = float(jnp.linalg.norm(block))
            row.append((fro, fro / total if total > 0 else 0.0))
        table.append(row)
    return labels, table, total


def print_block_norm_table(labels, table, total_norm: float):
    print("-" * 112)
    print(f"total Frobenius norm: {total_norm:.6e}")
    header = " " * 18 + " ".join(f"{label:>18}" for label in labels)
    print(header)
    for row_label, row in zip(labels, table):
        entries = " ".join(f"{fro:8.2e}/{rel:7.3f}" for fro, rel in row)
        print(f"{row_label:<18} {entries}")


def _symmetrize(matrix: jnp.ndarray) -> jnp.ndarray:
    return 0.5 * (matrix + matrix.T)


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


def _rt_z_tensor_from_matrix(matrix: jnp.ndarray, full_shape: tuple[int, int, int]) -> jnp.ndarray:
    nr, nt, nz = full_shape
    rt_size = nr * nt
    return matrix.reshape(nr, nt, nz, nr, nt, nz).transpose(0, 1, 3, 4, 2, 5).reshape(
        rt_size, rt_size, nz, nz
    )


def _apply_rt_z_block_inverse(
    rt_block_inverses: tuple[jnp.ndarray, ...],
    z_basis: jnp.ndarray,
    rhs: jnp.ndarray,
    rt_size: int,
) -> jnp.ndarray:
    x = rhs.reshape(rt_size, z_basis.shape[0])
    x_hat = x @ z_basis
    y_hat = jnp.stack(
        [block_inv @ x_hat[:, idx] for idx, block_inv in enumerate(rt_block_inverses)],
        axis=1,
    )
    y = y_hat @ z_basis.T
    return y.reshape(-1)


def build_bulk_rt_z_inverse(matrix: jnp.ndarray, full_shape: tuple[int, int, int]):
    tensor = _rt_z_tensor_from_matrix(matrix, full_shape)
    rt_size = full_shape[0] * full_shape[1]

    z_average = _symmetrize(jnp.einsum("ijab->ab", tensor) / rt_size)
    z_eigenvalues, z_basis = jnp.linalg.eigh(z_average)
    order = jnp.argsort(z_eigenvalues)[::-1]
    z_eigenvalues = z_eigenvalues[order]
    z_basis = z_basis[:, order]

    rt_blocks = []
    rt_block_inverses = []
    for idx in range(full_shape[2]):
        q = z_basis[:, idx]
        block = _symmetrize(jnp.einsum("ijab,a,b->ij", tensor, q, q))
        rt_blocks.append(block)
        rt_block_inverses.append(jnp.linalg.inv(block))

    return z_basis, z_eigenvalues, tuple(rt_blocks), tuple(rt_block_inverses)


def apply_bulk_rt_z_inverse(
    rt_block_inverses: tuple[jnp.ndarray, ...],
    z_basis: jnp.ndarray,
    rhs: jnp.ndarray,
    full_shape: tuple[int, int, int],
) -> jnp.ndarray:
    rt_size = full_shape[0] * full_shape[1]
    return _apply_rt_z_block_inverse(rt_block_inverses, z_basis, rhs, rt_size)


def bulk_rt_z_diagnostics(matrix: jnp.ndarray, full_shape: tuple[int, int, int]) -> RtZInverseDiagnostics:
    z_basis, _, _, rt_block_inverses = build_bulk_rt_z_inverse(matrix, full_shape)
    exact_inv = jnp.linalg.inv(matrix)
    eye = jnp.eye(matrix.shape[0], dtype=matrix.dtype)
    approx_inv = jnp.stack(
        [apply_bulk_rt_z_inverse(rt_block_inverses, z_basis, eye[:, idx], full_shape) for idx in range(matrix.shape[0])],
        axis=1,
    )

    key = jax.random.PRNGKey(0)
    rhs = jax.random.normal(key, (matrix.shape[0],), dtype=matrix.dtype)
    exact_sol = exact_inv @ rhs
    approx_sol = apply_bulk_rt_z_inverse(rt_block_inverses, z_basis, rhs, full_shape)
    residual = matrix @ approx_sol - rhs
    eigvals = jnp.linalg.eigvalsh(_symmetrize(matrix))
    return RtZInverseDiagnostics(
        relative_operator_error=float(jnp.linalg.norm(approx_inv - exact_inv) / jnp.linalg.norm(exact_inv)),
        relative_solve_error=float(jnp.linalg.norm(approx_sol - exact_sol) / jnp.linalg.norm(exact_sol)),
        relative_residual=float(jnp.linalg.norm(residual) / jnp.linalg.norm(rhs)),
        min_eigenvalue=float(jnp.min(eigvals)),
        max_eigenvalue=float(jnp.max(eigvals)),
    )


def _build_inverse_from_apply(apply_fn, size: int, dtype) -> jnp.ndarray:
    eye = jnp.eye(size, dtype=dtype)
    return jnp.stack([apply_fn(eye[:, idx]) for idx in range(size)], axis=1)


def _block_diag_apply(core_inv: jnp.ndarray, bulk_apply, rhs: jnp.ndarray, block_slices: dict[str, slice]) -> jnp.ndarray:
    rhs_c = rhs[block_slices["core"]]
    rhs_b = rhs[block_slices["bulk"]]
    x_c = core_inv @ rhs_c
    x_b = bulk_apply(rhs_b)
    return jnp.concatenate([x_c, x_b])


def coupled_block_diagnostics(
    block: jnp.ndarray,
    block_slices: dict[str, slice],
    bulk_exact_inv: jnp.ndarray,
    bulk_rt_z_apply,
) -> CoupledBlockDiagnostics:
    acc = block[block_slices["core"], block_slices["core"]]
    acb = block[block_slices["core"], block_slices["bulk"]]
    abc = block[block_slices["bulk"], block_slices["core"]]
    core_inv = jnp.linalg.inv(acc)
    exact_inv = jnp.linalg.inv(block)

    offdiag = jnp.sqrt(jnp.linalg.norm(acb) ** 2 + jnp.linalg.norm(abc) ** 2)
    offdiag_relative = float(offdiag / jnp.linalg.norm(block))

    eye = jnp.eye(block.shape[0], dtype=block.dtype)
    exact_blockdiag_inv = jnp.stack(
        [_block_diag_apply(core_inv, lambda rhs: bulk_exact_inv @ rhs, eye[:, idx], block_slices) for idx in range(block.shape[0])],
        axis=1,
    )
    mixed_blockdiag_inv = jnp.stack(
        [_block_diag_apply(core_inv, bulk_rt_z_apply, eye[:, idx], block_slices) for idx in range(block.shape[0])],
        axis=1,
    )

    key = jax.random.PRNGKey(11)
    rhs = jax.random.normal(key, (block.shape[0],), dtype=block.dtype)
    exact_sol = exact_inv @ rhs
    exact_blockdiag_sol = _block_diag_apply(core_inv, lambda vec: bulk_exact_inv @ vec, rhs, block_slices)
    mixed_blockdiag_sol = _block_diag_apply(core_inv, bulk_rt_z_apply, rhs, block_slices)
    exact_blockdiag_residual = block @ exact_blockdiag_sol - rhs
    mixed_blockdiag_residual = block @ mixed_blockdiag_sol - rhs

    return CoupledBlockDiagnostics(
        offdiag_relative_frobenius=offdiag_relative,
        exact_blockdiag_operator_error=float(jnp.linalg.norm(exact_blockdiag_inv - exact_inv) / jnp.linalg.norm(exact_inv)),
        mixed_blockdiag_operator_error=float(jnp.linalg.norm(mixed_blockdiag_inv - exact_inv) / jnp.linalg.norm(exact_inv)),
        exact_blockdiag_solve_error=float(jnp.linalg.norm(exact_blockdiag_sol - exact_sol) / jnp.linalg.norm(exact_sol)),
        mixed_blockdiag_solve_error=float(jnp.linalg.norm(mixed_blockdiag_sol - exact_sol) / jnp.linalg.norm(exact_sol)),
        exact_blockdiag_residual=float(jnp.linalg.norm(exact_blockdiag_residual) / jnp.linalg.norm(rhs)),
        mixed_blockdiag_residual=float(jnp.linalg.norm(mixed_blockdiag_residual) / jnp.linalg.norm(rhs)),
    )


def schur_core_bulk_apply(block: jnp.ndarray, block_slices: dict[str, slice], bulk_apply):
    acc = block[block_slices["core"], block_slices["core"]]
    acb = block[block_slices["core"], block_slices["bulk"]]
    abc = block[block_slices["bulk"], block_slices["core"]]

    u_cols = [bulk_apply(abc[:, idx]) for idx in range(abc.shape[1])]
    u = jnp.stack(u_cols, axis=1)
    schur = acc - acb @ u
    schur_inv = jnp.linalg.inv(schur)

    def apply(rhs: jnp.ndarray) -> jnp.ndarray:
        rhs_c = rhs[block_slices["core"]]
        rhs_b = rhs[block_slices["bulk"]]
        y = bulk_apply(rhs_b)
        z = schur_inv @ (rhs_c - acb @ y)
        x_b = y - bulk_apply(abc @ z)
        return jnp.concatenate([z, x_b])

    return apply, schur


def coupled_schur_diagnostics(
    block: jnp.ndarray,
    block_slices: dict[str, slice],
    bulk_exact_inv: jnp.ndarray,
    bulk_rt_z_apply,
) -> CoupledSchurDiagnostics:
    exact_inv = jnp.linalg.inv(block)
    exact_apply, _ = schur_core_bulk_apply(block, block_slices, lambda rhs: bulk_exact_inv @ rhs)
    mixed_apply, _ = schur_core_bulk_apply(block, block_slices, bulk_rt_z_apply)
    exact_schur_inv = _build_inverse_from_apply(exact_apply, block.shape[0], block.dtype)
    mixed_schur_inv = _build_inverse_from_apply(mixed_apply, block.shape[0], block.dtype)

    key = jax.random.PRNGKey(17)
    rhs = jax.random.normal(key, (block.shape[0],), dtype=block.dtype)
    exact_sol = exact_inv @ rhs
    exact_schur_sol = exact_apply(rhs)
    mixed_schur_sol = mixed_apply(rhs)
    exact_schur_residual = block @ exact_schur_sol - rhs
    mixed_schur_residual = block @ mixed_schur_sol - rhs

    return CoupledSchurDiagnostics(
        exact_schur_operator_error=float(jnp.linalg.norm(exact_schur_inv - exact_inv) / jnp.linalg.norm(exact_inv)),
        mixed_schur_operator_error=float(jnp.linalg.norm(mixed_schur_inv - exact_inv) / jnp.linalg.norm(exact_inv)),
        exact_schur_solve_error=float(jnp.linalg.norm(exact_schur_sol - exact_sol) / jnp.linalg.norm(exact_sol)),
        mixed_schur_solve_error=float(jnp.linalg.norm(mixed_schur_sol - exact_sol) / jnp.linalg.norm(exact_sol)),
        exact_schur_residual=float(jnp.linalg.norm(exact_schur_residual) / jnp.linalg.norm(rhs)),
        mixed_schur_residual=float(jnp.linalg.norm(mixed_schur_residual) / jnp.linalg.norm(rhs)),
    )


def core_schur_correction_diagnostics(
    block: jnp.ndarray,
    block_slices: dict[str, slice],
    bulk_exact_inv: jnp.ndarray,
    bulk_rt_z_apply,
) -> CoreSchurCorrectionDiagnostics:
    acc = block[block_slices["core"], block_slices["core"]]
    acb = block[block_slices["core"], block_slices["bulk"]]
    abc = block[block_slices["bulk"], block_slices["core"]]

    exact_correction = acb @ bulk_exact_inv @ abc
    mixed_u = jnp.stack([bulk_rt_z_apply(abc[:, idx]) for idx in range(abc.shape[1])], axis=1)
    mixed_correction = acb @ mixed_u
    exact_schur = acc - exact_correction
    mixed_schur = acc - mixed_correction

    acc_norm = jnp.linalg.norm(acc)
    exact_correction_norm = jnp.linalg.norm(exact_correction)

    return CoreSchurCorrectionDiagnostics(
        exact_correction_relative_to_acc=float(exact_correction_norm / acc_norm),
        mixed_correction_relative_to_acc=float(jnp.linalg.norm(mixed_correction) / acc_norm),
        mixed_vs_exact_correction_error=float(jnp.linalg.norm(mixed_correction - exact_correction) / exact_correction_norm),
        exact_schur_relative_to_acc=float(jnp.linalg.norm(exact_schur) / acc_norm),
        mixed_schur_vs_exact_error=float(jnp.linalg.norm(mixed_schur - exact_schur) / jnp.linalg.norm(exact_schur)),
    )


# %% Build once
SEQ, OPERATORS = ensure_built(CONFIG, rebuild=True)
print(
    f"built k=0 scalar case: ns={CONFIG.ns}, p={CONFIG.p}, map_kind={CONFIG.map_kind}, "
    f"eps={CONFIG.rotating_eps}, kappa={CONFIG.rotating_kappa}, nfp={CONFIG.rotating_nfp}"
)


# %% Verify A = E M_raw E^T
K0_MASS_IDENTITY = {}
for dirichlet in (False, True):
    a_ref, a_sandwich, diff = verify_extracted_mass_identity(SEQ, OPERATORS, dirichlet)
    K0_MASS_IDENTITY[dirichlet] = {"matrix": a_ref, "sandwich": a_sandwich, "diff": diff}
    print(f"dirichlet={dirichlet}: max |A - E M_raw E^T| = {diff:.3e}")


# %% Split the extracted matrix into fused core and bulk
K0_BLOCK_NORMS = {}
for dirichlet in (False, True):
    matrix = K0_MASS_IDENTITY[dirichlet]["matrix"]
    core_size = _core_size(SEQ)
    slices = {"core": slice(0, core_size), "bulk": slice(core_size, matrix.shape[0])}
    labels, table, total_norm = block_norm_table(matrix, slices)
    K0_BLOCK_NORMS[dirichlet] = {
        "matrix": matrix,
        "slices": slices,
        "labels": labels,
        "table": table,
        "total_norm": total_norm,
    }
    print("=" * 112)
    print(f"extracted k=0 core/bulk block norms: dirichlet={dirichlet}")
    print_block_norm_table(labels, table, total_norm)


# %% Try the scalar rt-z bulk strategy on A_bb
K0_BULK_RT_Z = {}
for dirichlet in (False, True):
    matrix = K0_MASS_IDENTITY[dirichlet]["matrix"]
    core_size = _core_size(SEQ)
    _, _, _, abb = _split_blocks(matrix, core_size)
    full_shape = _bulk_tensor_shape(SEQ, dirichlet)
    expected_size = full_shape[0] * full_shape[1] * full_shape[2]
    if abb.shape != (expected_size, expected_size):
        raise ValueError(
            f"A_bb shape mismatch for dirichlet={dirichlet}: got {abb.shape}, expected {(expected_size, expected_size)}"
        )
    diagnostics = bulk_rt_z_diagnostics(abb, full_shape)
    K0_BULK_RT_Z[dirichlet] = {"matrix": abb, "shape": full_shape, "diagnostics": diagnostics}
    print("=" * 112)
    print(f"A_bb rt-z diagnostics: dirichlet={dirichlet}, shape={full_shape}, size={abb.shape[0]}")
    print(f"relative operator error : {diagnostics.relative_operator_error:.3e}")
    print(f"relative solve error    : {diagnostics.relative_solve_error:.3e}")
    print(f"relative residual       : {diagnostics.relative_residual:.3e}")
    print(f"eigenvalue range        : [{diagnostics.min_eigenvalue:.3e}, {diagnostics.max_eigenvalue:.3e}]")


# %% Check whether the next issue is core-bulk coupling rather than A_bb itself
K0_CORE_BULK = {}
for dirichlet in (False, True):
    block = K0_MASS_IDENTITY[dirichlet]["matrix"]
    block_slices = K0_BLOCK_NORMS[dirichlet]["slices"]
    bulk = K0_BULK_RT_Z[dirichlet]["matrix"]
    full_shape = K0_BULK_RT_Z[dirichlet]["shape"]
    bulk_exact_inv = jnp.linalg.inv(bulk)
    z_basis, _, _, rt_block_inverses = build_bulk_rt_z_inverse(bulk, full_shape)
    bulk_rt_z_apply = lambda rhs, rb=rt_block_inverses, zb=z_basis, fs=full_shape: apply_bulk_rt_z_inverse(
        rb, zb, rhs, fs
    )

    diagnostics = coupled_block_diagnostics(block, block_slices, bulk_exact_inv, bulk_rt_z_apply)
    K0_CORE_BULK[dirichlet] = {"matrix": block, "slices": block_slices, "diagnostics": diagnostics}
    print("=" * 112)
    print(f"(core, bulk) coupled-block diagnostics: dirichlet={dirichlet}, size={block.shape[0]}")
    print(f"offdiag relative Frobenius     : {diagnostics.offdiag_relative_frobenius:.3e}")
    print(f"exact blockdiag operator error: {diagnostics.exact_blockdiag_operator_error:.3e}")
    print(f"mixed blockdiag operator error: {diagnostics.mixed_blockdiag_operator_error:.3e}")
    print(f"exact blockdiag solve error   : {diagnostics.exact_blockdiag_solve_error:.3e}")
    print(f"mixed blockdiag solve error   : {diagnostics.mixed_blockdiag_solve_error:.3e}")
    print(f"exact blockdiag residual      : {diagnostics.exact_blockdiag_residual:.3e}")
    print(f"mixed blockdiag residual      : {diagnostics.mixed_blockdiag_residual:.3e}")


# %% Promote the (core, bulk) block to a Schur model
K0_CORE_BULK_SCHUR = {}
for dirichlet in (False, True):
    block = K0_CORE_BULK[dirichlet]["matrix"]
    block_slices = K0_CORE_BULK[dirichlet]["slices"]
    bulk = K0_BULK_RT_Z[dirichlet]["matrix"]
    full_shape = K0_BULK_RT_Z[dirichlet]["shape"]
    bulk_exact_inv = jnp.linalg.inv(bulk)
    z_basis, _, _, rt_block_inverses = build_bulk_rt_z_inverse(bulk, full_shape)
    bulk_rt_z_apply = lambda rhs, rb=rt_block_inverses, zb=z_basis, fs=full_shape: apply_bulk_rt_z_inverse(
        rb, zb, rhs, fs
    )

    diagnostics = coupled_schur_diagnostics(block, block_slices, bulk_exact_inv, bulk_rt_z_apply)
    K0_CORE_BULK_SCHUR[dirichlet] = {"matrix": block, "slices": block_slices, "diagnostics": diagnostics}
    print("=" * 112)
    print(f"(core, bulk) Schur diagnostics: dirichlet={dirichlet}, size={block.shape[0]}")
    print(f"exact Schur operator error: {diagnostics.exact_schur_operator_error:.3e}")
    print(f"mixed Schur operator error: {diagnostics.mixed_schur_operator_error:.3e}")
    print(f"exact Schur solve error   : {diagnostics.exact_schur_solve_error:.3e}")
    print(f"mixed Schur solve error   : {diagnostics.mixed_schur_solve_error:.3e}")
    print(f"exact Schur residual      : {diagnostics.exact_schur_residual:.3e}")
    print(f"mixed Schur residual      : {diagnostics.mixed_schur_residual:.3e}")


# %% Measure how much the core Schur correction changes A_cc
K0_CORE_SCHUR_CORRECTION = {}
for dirichlet in (False, True):
    block = K0_CORE_BULK[dirichlet]["matrix"]
    block_slices = K0_CORE_BULK[dirichlet]["slices"]
    bulk = K0_BULK_RT_Z[dirichlet]["matrix"]
    full_shape = K0_BULK_RT_Z[dirichlet]["shape"]
    bulk_exact_inv = jnp.linalg.inv(bulk)
    z_basis, _, _, rt_block_inverses = build_bulk_rt_z_inverse(bulk, full_shape)
    bulk_rt_z_apply = lambda rhs, rb=rt_block_inverses, zb=z_basis, fs=full_shape: apply_bulk_rt_z_inverse(
        rb, zb, rhs, fs
    )

    diagnostics = core_schur_correction_diagnostics(block, block_slices, bulk_exact_inv, bulk_rt_z_apply)
    K0_CORE_SCHUR_CORRECTION[dirichlet] = diagnostics
    print("=" * 112)
    print(f"core Schur correction diagnostics: dirichlet={dirichlet}")
    print(f"||Acb Abb^-1 Abc|| / ||Acc||               : {diagnostics.exact_correction_relative_to_acc:.3e}")
    print(f"||Acb Atilde_bb^-1 Abc|| / ||Acc||         : {diagnostics.mixed_correction_relative_to_acc:.3e}")
    print(f"relative correction error (mixed vs exact) : {diagnostics.mixed_vs_exact_correction_error:.3e}")
    print(f"||Schur_core|| / ||Acc||                   : {diagnostics.exact_schur_relative_to_acc:.3e}")
    print(f"relative Schur error from A_bb approx      : {diagnostics.mixed_schur_vs_exact_error:.3e}")


# %% Optional: visualize the extracted k=0 matrix with core separator
DIRICHLET_TO_PLOT = False
A_TO_PLOT = K0_MASS_IDENTITY[DIRICHLET_TO_PLOT]["matrix"]
CORE_SIZE = _core_size(SEQ)

fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(jnp.log10(jnp.abs(A_TO_PLOT) + 1e-16), cmap="viridis", origin="lower")
ax.axhline(CORE_SIZE - 0.5, color="white", linewidth=1.0)
ax.axvline(CORE_SIZE - 0.5, color="white", linewidth=1.0)
ax.set_title(f"log10 |E0 M0 E0^T|, dirichlet={DIRICHLET_TO_PLOT}")
fig.colorbar(im, ax=ax, shrink=0.8)
plt.show()
# %%