# %% [markdown]
# # k=3 Polar Mass Surgery Debug
#
# This interactive script mirrors the `k=1` inverse-quality diagnostics on the
# scalar `k=3` extracted mass matrix.
#
# Here there is no special surgery split to discover: the main question is how
# well the scalar `rt|z` inverse class approximates the extracted `k=3` mass
# inverse itself.

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
class ExtractionSupportSummary:
    label: str
    n_rows: int
    total_min: int
    total_max: int
    total_mean: float


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
    operators = assemble_mass_operators(seq, seq.geometry, ks=(3,))
    return seq, operators


def ensure_built(config: ExperimentConfig = CONFIG, rebuild: bool = False):
    global SEQ, OPERATORS, BUILT_CONFIG
    if rebuild or SEQ is None or OPERATORS is None or BUILT_CONFIG != config:
        SEQ, OPERATORS = build_case(config)
        BUILT_CONFIG = config
    return SEQ, OPERATORS


def dense_extraction_matrix(seq, dirichlet: bool) -> jnp.ndarray:
    e = seq.e3_dbc if dirichlet else seq.e3
    return jnp.asarray(e.todense())


def dense_raw_mass_matrix(operators) -> jnp.ndarray:
    return jnp.asarray(operators.m3_sp.todense())


def dense_extracted_mass_matrix(seq, operators, dirichlet: bool) -> jnp.ndarray:
    return jnp.asarray(dense_mass_matrix(seq, operators, 3, dirichlet=dirichlet))


def verify_extracted_mass_identity(seq, operators, dirichlet: bool):
    e = dense_extraction_matrix(seq, dirichlet)
    m_raw = dense_raw_mass_matrix(operators)
    a_ref = dense_extracted_mass_matrix(seq, operators, dirichlet)
    a_sandwich = e @ m_raw @ e.T
    diff = jnp.max(jnp.abs(a_ref - a_sandwich))
    return a_ref, a_sandwich, float(diff)


def summarize_extraction_support(seq, dirichlet: bool):
    e = dense_extraction_matrix(seq, dirichlet)
    total = jnp.count_nonzero(e, axis=1)
    return ExtractionSupportSummary(
        label="k3",
        n_rows=e.shape[0],
        total_min=int(jnp.min(total)) if e.shape[0] else 0,
        total_max=int(jnp.max(total)) if e.shape[0] else 0,
        total_mean=float(jnp.mean(total)) if e.shape[0] else 0.0,
    )


def print_support_summary(report: ExtractionSupportSummary):
    print("-" * 96)
    print(f"label            : {report.label}")
    print(f"n_rows           : {report.n_rows}")
    print(f"nnz per row min  : {report.total_min}")
    print(f"nnz per row max  : {report.total_max}")
    print(f"nnz per row mean : {report.total_mean:.1f}")


def _symmetrize(matrix: jnp.ndarray) -> jnp.ndarray:
    return 0.5 * (matrix + matrix.T)


def _quadrature_weight_tensor(seq) -> jnp.ndarray:
    weight_flat = 1.0 / seq.geometry.jacobian_j
    quad_shape = (seq.quad.ny, seq.quad.nx, seq.quad.nz)
    return weight_flat.reshape(quad_shape).transpose(1, 0, 2)


def _assemble_weighted_1d_mass(basis: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    return (basis * weights[None, :]) @ basis.T


def _outer_z_basis_from_weight(seq):
    basis_z = seq.d_basis_z_jk
    weight_tensor = _quadrature_weight_tensor(seq)
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


def scalar_extraction_pair(seq, dirichlet: bool):
    if dirichlet:
        return jnp.asarray(seq.e3_dbc.todense()), jnp.asarray(seq.e3_dbc_T.todense())
    return jnp.asarray(seq.e3.todense()), jnp.asarray(seq.e3_T.todense())


def build_k3_rt_z_inverse(seq, operators, dirichlet: bool):
    full_matrix = jnp.asarray(operators.m3_sp.todense())
    full_shape = seq.basis_3.shape[0]
    rt_size = full_shape[0] * full_shape[1]
    z_eigenvalues, z_basis = _outer_z_basis_from_weight(seq)
    tensor = _scalar_rt_z_tensor_from_matrix(full_matrix, full_shape)

    rt_blocks = []
    rt_block_inverses = []
    for idx in range(full_shape[2]):
        q = z_basis[:, idx]
        block = _symmetrize(jnp.einsum("ijab,a,b->ij", tensor, q, q))
        rt_blocks.append(block)
        rt_block_inverses.append(jnp.linalg.inv(block))
    return full_shape, z_basis, z_eigenvalues, tuple(rt_blocks), tuple(rt_block_inverses), rt_size


def apply_k3_rt_z_inverse(seq, dirichlet: bool, rt_block_inverses, z_basis: jnp.ndarray, rt_size: int, rhs: jnp.ndarray) -> jnp.ndarray:
    e, e_t = scalar_extraction_pair(seq, dirichlet)
    full_rhs = e_t @ rhs
    full_sol = _apply_rt_z_block_inverse(rt_block_inverses, z_basis, full_rhs, rt_size)
    return e @ full_sol


def k3_rt_z_diagnostics(seq, operators, dirichlet: bool) -> RtZInverseDiagnostics:
    matrix = dense_extracted_mass_matrix(seq, operators, dirichlet)
    _, z_basis, _, _, rt_block_inverses, rt_size = build_k3_rt_z_inverse(seq, operators, dirichlet)
    exact_inv = jnp.linalg.inv(matrix)
    eye = jnp.eye(matrix.shape[0], dtype=matrix.dtype)
    approx_inv = jnp.stack(
        [apply_k3_rt_z_inverse(seq, dirichlet, rt_block_inverses, z_basis, rt_size, eye[:, idx]) for idx in range(matrix.shape[0])],
        axis=1,
    )

    key = jax.random.PRNGKey(0)
    rhs = jax.random.normal(key, (matrix.shape[0],), dtype=matrix.dtype)
    exact_sol = exact_inv @ rhs
    approx_sol = apply_k3_rt_z_inverse(seq, dirichlet, rt_block_inverses, z_basis, rt_size, rhs)
    residual = matrix @ approx_sol - rhs
    eigvals = jnp.linalg.eigvalsh(_symmetrize(matrix))
    return RtZInverseDiagnostics(
        relative_operator_error=float(jnp.linalg.norm(approx_inv - exact_inv) / jnp.linalg.norm(exact_inv)),
        relative_solve_error=float(jnp.linalg.norm(approx_sol - exact_sol) / jnp.linalg.norm(exact_sol)),
        relative_residual=float(jnp.linalg.norm(residual) / jnp.linalg.norm(rhs)),
        min_eigenvalue=float(jnp.min(eigvals)),
        max_eigenvalue=float(jnp.max(eigvals)),
    )


# %% Build once
SEQ, OPERATORS = ensure_built(CONFIG, rebuild=True)
print(
    f"built k=3 scalar case: ns={CONFIG.ns}, p={CONFIG.p}, map_kind={CONFIG.map_kind}, "
    f"eps={CONFIG.rotating_eps}, kappa={CONFIG.rotating_kappa}, nfp={CONFIG.rotating_nfp}"
)


# %% Dense extraction support summary
for dirichlet in (False, True):
    print("=" * 96)
    print(f"dense extraction support summary: dirichlet={dirichlet}")
    print_support_summary(summarize_extraction_support(SEQ, dirichlet))


# %% Verify A = E M_raw E^T
K3_MASS_IDENTITY = {}
for dirichlet in (False, True):
    a_ref, a_sandwich, diff = verify_extracted_mass_identity(SEQ, OPERATORS, dirichlet)
    K3_MASS_IDENTITY[dirichlet] = {"matrix": a_ref, "sandwich": a_sandwich, "diff": diff}
    print(f"dirichlet={dirichlet}: max |A - E M_raw E^T| = {diff:.3e}")


# %% Try the scalar rt-z inverse on the full extracted k=3 matrix
K3_RT_Z = {}
for dirichlet in (False, True):
    diagnostics = k3_rt_z_diagnostics(SEQ, OPERATORS, dirichlet)
    matrix = K3_MASS_IDENTITY[dirichlet]["matrix"]
    K3_RT_Z[dirichlet] = {"matrix": matrix, "diagnostics": diagnostics}
    print("=" * 112)
    print(f"k=3 rt-z diagnostics: dirichlet={dirichlet}, size={matrix.shape[0]}")
    print(f"relative operator error : {diagnostics.relative_operator_error:.3e}")
    print(f"relative solve error    : {diagnostics.relative_solve_error:.3e}")
    print(f"relative residual       : {diagnostics.relative_residual:.3e}")
    print(f"eigenvalue range        : [{diagnostics.min_eigenvalue:.3e}, {diagnostics.max_eigenvalue:.3e}]")


# %% Optional: visualize the extracted k=3 matrix
DIRICHLET_TO_PLOT = False
A_TO_PLOT = K3_MASS_IDENTITY[DIRICHLET_TO_PLOT]["matrix"]

fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(jnp.log10(jnp.abs(A_TO_PLOT) + 1e-16), cmap="viridis", origin="lower")
ax.set_title(f"log10 |E3 M3 E3^T|, dirichlet={DIRICHLET_TO_PLOT}")
fig.colorbar(im, ax=ax, shrink=0.8)
plt.show()
# %%