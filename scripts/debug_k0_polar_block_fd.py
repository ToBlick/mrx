# %% [markdown]
# # RT-Z Scalar Mass Preconditioner
#
# This interactive script keeps only the pieces that matter for the current
# scalar-mass preconditioner story.
#
# The conclusions from the earlier experiments are:
#
# 1. The old single-`M_z` bulk model is not good enough on coupled geometry.
# 2. The scalar geometry factors are nevertheless strongly compressible under
#    an `rt|z` split.
# 3. For `k=3`, a richer `rt-z` inverse class beats the production mass
#    preconditioners.
# 4. For `k=0`, the polar axis fusion matters, so the right object is the
#    extracted block matrix
#
#        A = E M_raw E^T = [[A_cc, A_cb], [A_bc, A_bb]].
#
#    The current `rt-zblock` preconditioner treats the core and coupling blocks
#    explicitly through a Schur complement and approximates only the extracted
#    bulk inverse `A_bb^{-1}`.
#
# The script below keeps only:
#
# - sequence / operator construction,
# - the production scalar mass preconditioners (`jacobi`, `kronecker`),
# - the new `rt-zblock` preconditioner,
# - storage / memory estimates,
# - and direct CG benchmarks.
#
# All heavier tensor-factorization experiments were removed.

# %%
from __future__ import annotations

import time
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp

from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import rotating_ellipse_map, toroid_map
from mrx.operators import (
    apply_mass_matrix,
    apply_mass_matrix_preconditioner,
    assemble_kron_mass_preconditioner,
    assemble_mass_operators,
    dense_mass_matrix,
)
from mrx.solvers import solve_singular_cg

jax.config.update("jax_enable_x64", True)


# %% [markdown]
# ## Geometry And Factorization View
#
# For the scalar mass matrices we care about the weights
#
# - `W^0 = J` for `k=0`,
# - `W^3 = 1/J` for `k=3`.
#
# On donut-like geometries these weights are well described by an outer `rt|z`
# split: the toroidal direction `z` is the natural outer mode, while the `r-theta`
# plane carries the denser local structure.
#
# The production `kronecker` preconditioner ignores almost all of that structure.
# It uses a single reference inverse sandwiched by extraction:
#
#     E (M_r^{-1} ⊗ M_t^{-1} ⊗ M_z^{-1}) E^T,
#
# up to simple scalar geometry scales.
#
# The new `rt-zblock` idea is richer:
#
# 1. choose a dense basis in the `z` direction,
# 2. transform the operator into that basis,
# 3. keep only the diagonal `z` blocks,
# 4. store a dense `rt` inverse for each retained `z` block.
#
# For `k=3`, the extracted space is still tensor-product-like, so this is done
# directly on the extracted scalar mass matrix.
#
# For `k=0`, the polar axis fusion is non-unitary. So we do **not** invert a raw
# sandwiched model. Instead, we work directly with the extracted block matrix,
# keep the small fused core explicit, and use an `rt-z` block inverse only for the
# extracted bulk block.


# %% Configuration
@dataclass(frozen=True)
class ExperimentConfig:
    ns: tuple[int, int, int] = (20, 20, 20)
    p: int = 3
    tol: float = 1e-9
    maxiter: int = 2000
    betti: tuple[int, int, int, int] = (1, 1, 0, 0)
    map_kind: str = "rotating_ellipse"
    torus_epsilon: float = 1.0 / 3.0
    torus_r0: float = 1.0
    rotating_eps: float = 0.33
    rotating_kappa: float = 1.2
    rotating_r0: float = 1.0
    rotating_nfp: int = 3


@dataclass
class ScalarRtZBlockInverseFactors:
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


@dataclass
class ExtractedBulkRtZBlockInverseFactors:
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


@dataclass
class ScalarMassPreconditionerBenchmarkReport:
    k: int
    dirichlet: bool
    label: str
    n_rhs: int
    avg_iters: float
    std_iters: float
    max_iters: int
    avg_time_ms: float
    std_time_ms: float
    max_time_ms: float


@dataclass
class RtZStorageReport:
    k: int
    dirichlet: bool
    n_rt_blocks: int
    rt_size: int
    z_size: int
    total_matrix_entries: int
    approx_mebibytes: float


CONFIG = ExperimentConfig()
seed = 0
SEQ: DeRhamSequence | None = None
OPERATORS = None
BUILT_CONFIG: ExperimentConfig | None = None


# %% Helpers

def _symmetrize(matrix: jnp.ndarray) -> jnp.ndarray:
    return 0.5 * (matrix + matrix.T)


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


def _build_sequence(config: ExperimentConfig):
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

    operators = None
    operators = assemble_mass_operators(seq, seq.geometry, operators=operators, ks=(0, 3))
    operators = assemble_kron_mass_preconditioner(seq, operators=operators)
    return seq, operators


def ensure_built(config: ExperimentConfig = CONFIG, rebuild: bool = False):
    global SEQ, OPERATORS, BUILT_CONFIG
    if rebuild or SEQ is None or OPERATORS is None or BUILT_CONFIG != config:
        SEQ, OPERATORS = _build_sequence(config)
        BUILT_CONFIG = config
    return SEQ, OPERATORS


def _scalar_extraction_pair(seq, k: int, dirichlet: bool):
    if k == 0:
        if dirichlet:
            return seq.e0_dbc.todense(), seq.e0_dbc_T.todense()
        return seq.e0.todense(), seq.e0_T.todense()
    if k == 3:
        if dirichlet:
            return seq.e3_dbc.todense(), seq.e3_dbc_T.todense()
        return seq.e3.todense(), seq.e3_T.todense()
    raise ValueError("Scalar helpers only support k=0 and k=3")


def _scalar_sparse_operator(seq, operators, k: int):
    if k == 0:
        return operators.m0_sp
    if k == 3:
        return operators.m3_sp
    raise ValueError("Scalar helpers only support k=0 and k=3")


def _scalar_full_shape(seq, k: int) -> tuple[int, int, int]:
    if k == 0:
        return seq.basis_0.shape[0]
    if k == 3:
        return seq.basis_3.shape[0]
    raise ValueError("Scalar helpers only support k=0 and k=3")


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


def _kron_forward_3d(Mr: jnp.ndarray, Mt: jnp.ndarray, Mz: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    y = jnp.einsum("ij,jkl->ikl", Mr, x)
    y = jnp.einsum("ij,kjl->kil", Mt, y)
    y = jnp.einsum("ij,klj->kli", Mz, y)
    return y


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


def _apply_rt_z_block_inverse(rt_block_inverses: tuple[jnp.ndarray, ...], z_basis: jnp.ndarray, rhs: jnp.ndarray, rt_size: int) -> jnp.ndarray:
    x = rhs.reshape(rt_size, z_basis.shape[0])
    x_hat = x @ z_basis
    y_hat = jnp.stack([block_inv @ x_hat[:, idx] for idx, block_inv in enumerate(rt_block_inverses)], axis=1)
    y = y_hat @ z_basis.T
    return y.reshape(-1)


# %% [markdown]
# ## New Preconditioner
#
# The `rt-zblock` preconditioner has two variants.
#
# ### `k = 3`
#
# We work directly on the extracted scalar mass matrix.  We reshape it as an
# `rt-z` tensor, choose a dense `z` basis from the low-rank structure of the
# scalar weight `1/J`, and keep the diagonal `z` blocks only.  Each retained
# block is a dense `rt` matrix whose inverse is stored explicitly.
#
# ### `k = 0`
#
# We work on the extracted block matrix
#
#     A = [[A_cc, A_cb], [A_bc, A_bb]].
#
# The small fused core `A_cc` and the coupling blocks are kept explicitly.  The
# only approximation is in `A_bb^{-1}`, which is modeled by the same dense
# `rt-z` block inverse idea as above.  The final preconditioner applies the
# extracted-space Schur complement.
#
# This is the key difference from the production `kronecker` preconditioner:
# the new `k=0` prototype respects the polar fused axis at the level of the
# extracted matrix itself.
#
# ## Memory Footprint
#
# This preconditioner is more expensive than production `kronecker`.
# Production stores only a handful of 1D inverses and scalar scales.
#
# The `rt-zblock` prototype stores:
#
# - one dense `z` basis of size `n_z x n_z`,
# - `n_z` dense `rt` blocks,
# - `n_z` dense `rt` block inverses,
# - and for `k=0` also the small extracted core / coupling Schur data.
#
# If `n_r ~ n_t ~ n_z ~ n`, this is roughly `O(n_z (n_r n_t)^2) = O(n^5)`
# storage if all `z` blocks are kept.  That is still far below a full dense 3D
# inverse (`O(n^6)`), but it is much larger than the production Kronecker model.
#
# So this script answers the question “is the richer inverse class worth it?”
# It does **not** yet claim that the current storage pattern is the final
# production form.


# %% RT-Z inverse builds

def build_scalar_rt_z_block_inverse_factors(config: ExperimentConfig = CONFIG, k: int = 3, dirichlet: bool = False) -> ScalarRtZBlockInverseFactors:
    seq, operators = ensure_built(config)
    full_matrix = jnp.asarray(_scalar_sparse_operator(seq, operators, k).todense())
    full_shape = _scalar_full_shape(seq, k)
    rt_size = full_shape[0] * full_shape[1]
    z_size = full_shape[2]
    tensor = _scalar_rt_z_tensor_from_matrix(full_matrix, full_shape)
    z_eigenvalues, z_basis = _scalar_outer_z_basis_from_weight(seq, k)

    rt_blocks = []
    rt_block_inverses = []
    for idx in range(z_size):
        q = z_basis[:, idx]
        block = _symmetrize(jnp.einsum("ijab,a,b->ij", tensor, q, q))
        rt_blocks.append(block)
        rt_block_inverses.append(jnp.linalg.inv(block))

    e, _ = _scalar_extraction_pair(seq, k, dirichlet)
    return ScalarRtZBlockInverseFactors(
        k=k,
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


def _apply_scalar_rt_z_block_inverse_factors(seq, factors: ScalarRtZBlockInverseFactors, rhs: jnp.ndarray) -> jnp.ndarray:
    e, e_t = _scalar_extraction_pair(seq, factors.k, factors.dirichlet)
    full_rhs = e_t @ rhs
    return e @ _apply_rt_z_block_inverse(factors.rt_block_inverses, factors.z_basis, full_rhs, factors.rt_size)


def build_extracted_bulk_rt_z_block_inverse_factors(config: ExperimentConfig = CONFIG, dirichlet: bool = False) -> ExtractedBulkRtZBlockInverseFactors:
    seq, operators = ensure_built(config)
    matrix = dense_mass_matrix(seq, operators, 0, dirichlet=dirichlet)
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


def _apply_extracted_bulk_rt_z_block_inverse(factors: ExtractedBulkRtZBlockInverseFactors, rhs: jnp.ndarray) -> jnp.ndarray:
    rhs_c = rhs[:factors.core_size]
    rhs_b = rhs[factors.core_size:]
    rt_size = factors.bulk_shape[0] * factors.bulk_shape[1]
    y = _apply_rt_z_block_inverse(factors.rt_block_inverses, factors.z_basis, rhs_b, rt_size)
    z = factors.schur_inv @ (rhs_c - factors.acb @ y)
    x_b = y - _apply_rt_z_block_inverse(factors.rt_block_inverses, factors.z_basis, factors.abc @ z, rt_size)
    return jnp.concatenate([z, x_b])


# %% Storage and benchmark helpers

def build_rt_z_storage_report(config: ExperimentConfig = CONFIG, k: int = 0, dirichlet: bool = False) -> RtZStorageReport:
    if k == 0:
        factors = build_extracted_bulk_rt_z_block_inverse_factors(config=config, dirichlet=dirichlet)
        rt_size = factors.bulk_shape[0] * factors.bulk_shape[1]
        z_size = factors.bulk_shape[2]
        n_rt_blocks = len(factors.rt_blocks)
        extra_entries = (
            factors.acb.size + factors.abc.size + factors.schur_inv.size
        )
    else:
        factors = build_scalar_rt_z_block_inverse_factors(config=config, k=k, dirichlet=dirichlet)
        rt_size = factors.rt_size
        z_size = factors.z_size
        n_rt_blocks = len(factors.rt_blocks)
        extra_entries = 0

    total_entries = 2 * n_rt_blocks * rt_size * rt_size + z_size * z_size + extra_entries
    return RtZStorageReport(
        k=k,
        dirichlet=dirichlet,
        n_rt_blocks=n_rt_blocks,
        rt_size=rt_size,
        z_size=z_size,
        total_matrix_entries=total_entries,
        approx_mebibytes=float(total_entries * 8 / 1024**2),
    )


def _build_scalar_mass_preconditioner_matvec(seq, operators, k: int, dirichlet: bool, label: str, config: ExperimentConfig):
    if label in ("jacobi", "kronecker"):
        return lambda x: apply_mass_matrix_preconditioner(seq, operators, x, k, dirichlet=dirichlet, kind=label)
    if label == "rt-zblock":
        if k == 0:
            factors = build_extracted_bulk_rt_z_block_inverse_factors(config=config, dirichlet=dirichlet)
            return lambda x: _apply_extracted_bulk_rt_z_block_inverse(factors, x)
        factors = build_scalar_rt_z_block_inverse_factors(config=config, k=k, dirichlet=dirichlet)
        return lambda x: _apply_scalar_rt_z_block_inverse_factors(seq, factors, x)
    raise ValueError(f"Unsupported scalar mass preconditioner label: {label}")


def benchmark_scalar_mass_preconditioners(config: ExperimentConfig = CONFIG, k: int = 0, dirichlet: bool = False, seed: int = 0, n_rhs: int = 8, labels: tuple[str, ...] = ("jacobi", "kronecker", "rt-zblock")):
    seq, operators = ensure_built(config)
    e, _ = _scalar_extraction_pair(seq, k, dirichlet)
    rhs_batch = jax.random.normal(jax.random.PRNGKey(seed + 100 * k + int(dirichlet)), (n_rhs, e.shape[0]), dtype=jnp.float64)

    def A_mv(x):
        return apply_mass_matrix(seq, operators, x, k, dirichlet=dirichlet)

    reports = []
    for label in labels:
        M_mv = _build_scalar_mass_preconditioner_matvec(seq, operators, k, dirichlet, label, config)

        @jax.jit
        def solve(rhs):
            x, info = solve_singular_cg(
                A_mv,
                rhs,
                mass_matvec=A_mv,
                precond_matvec=M_mv,
                tol=config.tol,
                maxiter=config.maxiter,
            )
            return x, jnp.abs(info)

        x0, _ = solve(rhs_batch[0])
        jax.block_until_ready(x0)

        iters = []
        times_ms = []
        for rhs in rhs_batch:
            t0 = time.perf_counter()
            x, info = solve(rhs)
            jax.block_until_ready(x)
            times_ms.append((time.perf_counter() - t0) * 1e3)
            iters.append(int(info))
        iters = jnp.asarray(iters)
        times_ms = jnp.asarray(times_ms)
        reports.append(
            ScalarMassPreconditionerBenchmarkReport(
                k=k,
                dirichlet=dirichlet,
                label=label,
                n_rhs=n_rhs,
                avg_iters=float(jnp.mean(iters)),
                std_iters=float(jnp.std(iters)),
                max_iters=int(jnp.max(iters)),
                avg_time_ms=float(jnp.mean(times_ms)),
                std_time_ms=float(jnp.std(times_ms)),
                max_time_ms=float(jnp.max(times_ms)),
            )
        )
    return reports


def _print_storage_report(report: RtZStorageReport):
    print("-" * 96)
    print(
        f"rt-zblock storage: k={report.k} dirichlet={report.dirichlet} "
        f"n_rt_blocks={report.n_rt_blocks} rt_size={report.rt_size} z_size={report.z_size}"
    )
    print(f"total_matrix_entries : {report.total_matrix_entries:d}")
    print(f"approx_mebibytes     : {report.approx_mebibytes:.2f}")


def _print_scalar_mass_preconditioner_benchmark_report(report: ScalarMassPreconditionerBenchmarkReport):
    print("-" * 96)
    print(
        f"scalar mass preconditioner benchmark: k={report.k} dirichlet={report.dirichlet} "
        f"label={report.label} n_rhs={report.n_rhs}"
    )
    print(f"avg_iters : {report.avg_iters:.2f}")
    print(f"std_iters : {report.std_iters:.2f}")
    print(f"max_iters : {report.max_iters:d}")
    print(f"avg_ms    : {report.avg_time_ms:.2f}")
    print(f"std_ms    : {report.std_time_ms:.2f}")
    print(f"max_ms    : {report.max_time_ms:.2f}")


# %% [markdown]
# ## Build Once
#
# This cell is the expensive step.  Everything below can be rerun without
# rebuilding the sequence and sparse operators.

# %% Build or rebuild the experiment once
SEQ, OPERATORS = ensure_built(CONFIG, rebuild=True)
print(
    f"built experiment: map_kind={CONFIG.map_kind}, ns={CONFIG.ns}, p={CONFIG.p}, "
    f"torus_epsilon={CONFIG.torus_epsilon}, torus_r0={CONFIG.torus_r0}"
)


# %% [markdown]
# ## Storage Report
#
# This reports only the stored dense matrices for the current debug prototype.
# It does not include JAX/XLA runtime buffers or sparse operator storage.

# %% Report rt-zblock storage footprint
RTZ_STORAGE_REPORTS = {}
for k in (0, 3):
    case_reports = {}
    for dirichlet in (False, True):
        report = build_rt_z_storage_report(config=CONFIG, k=k, dirichlet=dirichlet)
        _print_storage_report(report)
        case_reports[dirichlet] = report
    RTZ_STORAGE_REPORTS[k] = case_reports


# %% [markdown]
# ## Benchmark Against Production Baselines
#
# The only comparison kept here is the one that matters for the new
# preconditioner:
#
# - `jacobi`
# - production `kronecker`
# - `rt-zblock`
#
# The benchmark solves the scalar mass system with CG and swaps only the
# preconditioner callback.

# %% Compare scalar mass preconditioners: jacobi vs production kronecker vs rt-zblock
SCALAR_MASS_PRECONDITIONER_BENCHMARKS = {}
for k in (0, 3):
    case_reports = {}
    for dirichlet in (False, True):
        reports = benchmark_scalar_mass_preconditioners(
            config=CONFIG,
            k=k,
            dirichlet=dirichlet,
            seed=seed,
            n_rhs=8,
            labels=("jacobi", "kronecker", "rt-zblock"),
        )
        for report in reports:
            _print_scalar_mass_preconditioner_benchmark_report(report)
        case_reports[dirichlet] = reports
    SCALAR_MASS_PRECONDITIONER_BENCHMARKS[k] = case_reports

# %%
