# %% [markdown]
# Stiffness Tensor Forward Diagnostics
#
# Notebook-style diagnostics for the extracted stiffness forward models and
# the associated tensor-preconditioner block data for `k = 0, 1, 2`.
#
# Intended workflow in VS Code:
#
# - Run the configuration and helper cells.
# - Run the build cell once on a GPU node.
# - Then inspect the per-k cells interactively.

# %%
from __future__ import annotations

import jax
import jax.numpy as jnp

from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import rotating_ellipse_map
from mrx.operators import (
    _apply_k0_tensor_hodge_forward_model,
    _assemble_dense_from_apply,
    _reshape_quadrature_scalar_field,
    apply_stiffness,
    apply_stiffness_tensor_forward_model,
    assemble_hodge_operators,
    assemble_incidence_operators,
    assemble_mass_operators,
    assemble_tensor_mass_preconditioner,
    assemble_tensor_stiffness_models,
    assemble_tensor_stiffness_preconditioner,
    update_hodge_operator,
    _reshape_quadrature_matrix_field
)
from mrx.preconditioners import (
    _apply_k1_bulk_diagonal_preconditioner,
    _apply_k1_bulk_preconditioner,
    _surgery_slices_k1,
    select_boundary_data,
)

jax.config.update("jax_enable_x64", True)


# %% Configuration
NS = (6, 8, 4)
P = 3
DIRICHLET = True
RANK = 1
N_VECTORS = 8
SEED = 0

ROTATING_EPS = 0.33
ROTATING_KAPPA = 1.4
ROTATING_R0 = 1.0
ROTATING_NFP = 3

CP_KWARGS = {
    "maxiter": 100,
    "tol": 1e-9,
    "ridge": 1e-12,
    "surgery_schur_pinv_tol": 1e-8,
    "bulk_block_pinv_tol": 1e-8,
}


# %% Helpers
def _build_sequence() -> DeRhamSequence:
    seq = DeRhamSequence(
        NS,
        (P, P, P),
        2 * P,
        ("clamped", "periodic", "periodic"),
        polar=True,
        tol=1e-9,
        maxiter=2000,
        betti_numbers=(1, 1, 0, 0),
    )
    seq.evaluate_1d()
    seq.assemble_reference_mass_matrix()
    seq.set_map(
        rotating_ellipse_map(
            eps=ROTATING_EPS,
            kappa=ROTATING_KAPPA,
            R0=ROTATING_R0,
            nfp=ROTATING_NFP,
        )
    )
    return seq


def _symmetrize(matrix: jnp.ndarray) -> jnp.ndarray:
    return 0.5 * (matrix + matrix.T)


def _assemble_unweighted_1d_mass(basis: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    return _symmetrize((basis * weights[None, :]) @ basis.T)


def _restrict_radial_mass(matrix: jnp.ndarray, radial_start: int, nr: int) -> jnp.ndarray:
    return matrix[radial_start:radial_start + nr, radial_start:radial_start + nr]


def _relative_error(y_model: jnp.ndarray, y_exact: jnp.ndarray) -> jnp.ndarray:
    return jnp.linalg.norm(y_model - y_exact) / jnp.maximum(jnp.linalg.norm(y_exact), 1e-30)


def random_error_stats(exact_apply, model_apply, size: int, *, n_vectors: int = N_VECTORS, seed: int = SEED):
    keys = jax.random.split(jax.random.PRNGKey(seed), n_vectors)

    def one(key):
        x = jax.random.normal(key, (size,), dtype=jnp.float64)
        return _relative_error(model_apply(x), exact_apply(x))

    rel_errors = jax.vmap(one)(keys)
    return {
        "mean": float(jnp.mean(rel_errors)),
        "std": float(jnp.std(rel_errors)),
        "max": float(jnp.max(rel_errors)),
        "values": rel_errors,
    }


def dense_comparison(exact_apply, model_apply, size: int):
    exact = _assemble_dense_from_apply(exact_apply, size)
    model = _assemble_dense_from_apply(model_apply, size)
    return {
        "exact": exact,
        "model": model,
        "fro_relative_error": float(
            jnp.linalg.norm(model - exact) / jnp.maximum(jnp.linalg.norm(exact), 1e-30)
        ),
        "exact_symmetry_defect": float(jnp.linalg.norm(exact - exact.T)),
        "model_symmetry_defect": float(jnp.linalg.norm(model - model.T)),
    }


def eig_summary(matrix: jnp.ndarray):
    eigs = jnp.linalg.eigvalsh(_symmetrize(matrix))
    abs_eigs = jnp.abs(eigs)
    return {
        "eigs": eigs,
        "min": float(jnp.min(eigs)),
        "max": float(jnp.max(eigs)),
        "min_abs": float(jnp.min(abs_eigs)),
        "count_nonpositive": int(jnp.sum(eigs <= 0.0)),
    }


def summarize_modal_inverse(block_factors):
    inv_denom = block_factors.fd_inv_denom
    if inv_denom is None:
        return {
            "shape": block_factors.shape,
            "has_modal_inverse": False,
            "cp_relative_error": block_factors.cp_relative_error,
            "cp_final_delta": block_factors.cp_final_delta,
        }

    positive_mask = inv_denom > 0.0
    positive_values = inv_denom[positive_mask]
    zero_count = int(inv_denom.size - positive_values.size)
    denom_positive = jnp.where(positive_mask, 1.0 / inv_denom, jnp.inf)

    return {
        "shape": block_factors.shape,
        "has_modal_inverse": True,
        "cp_relative_error": block_factors.cp_relative_error,
        "cp_final_delta": block_factors.cp_final_delta,
        "zero_count": zero_count,
        "positive_count": int(positive_values.size),
        "min_inv_denom": float(jnp.min(positive_values)) if positive_values.size else None,
        "max_inv_denom": float(jnp.max(positive_values)) if positive_values.size else None,
        "min_positive_denom": float(jnp.min(denom_positive[positive_mask])) if positive_values.size else None,
        "max_positive_denom": float(jnp.max(denom_positive[positive_mask])) if positive_values.size else None,
        "lam_r": block_factors.fd_lam_r,
        "lam_t": block_factors.fd_lam_t,
        "lam_z": block_factors.fd_lam_z,
        "inv_denom": inv_denom,
    }


def modal_finiteness_summary(block_factors):
    def summarize(name, values):
        if values is None:
            return {"name": name, "present": False}
        values = jnp.asarray(values)
        return {
            "name": name,
            "present": True,
            "shape": tuple(values.shape),
            "all_finite": bool(jnp.all(jnp.isfinite(values))),
            "nan_count": int(jnp.sum(jnp.isnan(values))),
            "inf_count": int(jnp.sum(jnp.isinf(values))),
            "min": float(jnp.nanmin(values)),
            "max": float(jnp.nanmax(values)),
        }

    return {
        "lam_r": summarize("lam_r", block_factors.fd_lam_r),
        "lam_t": summarize("lam_t", block_factors.fd_lam_t),
        "lam_z": summarize("lam_z", block_factors.fd_lam_z),
        "inv_denom": summarize("inv_denom", block_factors.fd_inv_denom),
    }


def canonical_k1_reference_masses(seq, *, dirichlet: bool):
    arr_shape = k1_payload.factors.arr.shape if dirichlet == DIRICHLET else None
    theta_shape = k1_payload.factors.theta.shape if dirichlet == DIRICHLET else None
    zeta_shape = k1_payload.factors.zeta.shape if dirichlet == DIRICHLET else None
    if arr_shape is None or theta_shape is None or zeta_shape is None:
        raise ValueError("Run the build cell first for the selected boundary condition")
    return {
        "arr": {
            "r": _restrict_radial_mass(_assemble_unweighted_1d_mass(seq.d_basis_r_jk, seq.quad.w_x), 1, arr_shape[0]),
            "t": _assemble_unweighted_1d_mass(seq.basis_t_jk, seq.quad.w_y),
            "z": _assemble_unweighted_1d_mass(seq.basis_z_jk, seq.quad.w_z),
        },
        "theta": {
            "r": _restrict_radial_mass(_assemble_unweighted_1d_mass(seq.basis_r_jk, seq.quad.w_x), 2, theta_shape[0]),
            "t": _assemble_unweighted_1d_mass(seq.d_basis_t_jk, seq.quad.w_y),
            "z": _assemble_unweighted_1d_mass(seq.basis_z_jk, seq.quad.w_z),
        },
        "zeta": {
            "r": _restrict_radial_mass(_assemble_unweighted_1d_mass(seq.basis_r_jk, seq.quad.w_x), 2, zeta_shape[0]),
            "t": _assemble_unweighted_1d_mass(seq.basis_t_jk, seq.quad.w_y),
            "z": _assemble_unweighted_1d_mass(seq.d_basis_z_jk, seq.quad.w_z),
        },
    }


def canonical_k2_reference_masses(seq, *, dirichlet: bool):
    r_bulk_shape = k2_payload.factors.r_bulk.shape if dirichlet == DIRICHLET else None
    theta_shape = k2_payload.factors.theta.shape if dirichlet == DIRICHLET else None
    zeta_shape = k2_payload.factors.zeta.shape if dirichlet == DIRICHLET else None
    if r_bulk_shape is None or theta_shape is None or zeta_shape is None:
        raise ValueError("Run the build cell first for the selected boundary condition")
    return {
        "r_bulk": {
            "r": _restrict_radial_mass(_assemble_unweighted_1d_mass(seq.basis_r_jk, seq.quad.w_x), 2, r_bulk_shape[0]),
            "t": _assemble_unweighted_1d_mass(seq.d_basis_t_jk, seq.quad.w_y),
            "z": _assemble_unweighted_1d_mass(seq.d_basis_z_jk, seq.quad.w_z),
        },
        "theta": {
            "r": _restrict_radial_mass(_assemble_unweighted_1d_mass(seq.d_basis_r_jk, seq.quad.w_x), 1, theta_shape[0]),
            "t": _assemble_unweighted_1d_mass(seq.basis_t_jk, seq.quad.w_y),
            "z": _assemble_unweighted_1d_mass(seq.d_basis_z_jk, seq.quad.w_z),
        },
        "zeta": {
            "r": _restrict_radial_mass(_assemble_unweighted_1d_mass(seq.d_basis_r_jk, seq.quad.w_x), 1, zeta_shape[0]),
            "t": _assemble_unweighted_1d_mass(seq.d_basis_t_jk, seq.quad.w_y),
            "z": _assemble_unweighted_1d_mass(seq.basis_z_jk, seq.quad.w_z),
        },
    }


def reference_mass_summary(reference_dict):
    return {
        block_name: {
            axis_name: eig_summary(axis_matrix)
            for axis_name, axis_matrix in axes.items()
        }
        for block_name, axes in reference_dict.items()
    }


def compare_random_vector(exact_apply, model_apply, size: int, *, seed: int = SEED):
    x = jax.random.normal(jax.random.PRNGKey(seed), (size,), dtype=jnp.float64)
    y_exact = exact_apply(x)
    y_model = model_apply(x)
    return {
        "x": x,
        "y_exact": y_exact,
        "y_model": y_model,
        "relative_error": float(_relative_error(y_model, y_exact)),
    }


# %% Build once
seq = _build_sequence()

ops = None
ops = assemble_mass_operators(seq, seq.geometry, operators=ops, ks=(0, 1, 2, 3))
ops = assemble_incidence_operators(seq, operators=ops, ks=(0, 1, 2))

ops = assemble_tensor_mass_preconditioner(
    seq,
    operators=ops,
    ks=(0,),
    rank=RANK,
    cp_kwargs=CP_KWARGS,
)
ops = update_hodge_operator(seq, seq.geometry, ops, 0)

ops = assemble_tensor_stiffness_models(
    seq,
    operators=ops,
    ks=(1, 2),
    rank=RANK,
    cp_kwargs=CP_KWARGS,
)
ops = assemble_tensor_stiffness_preconditioner(
    seq,
    operators=ops,
    ks=(1, 2),
    rank=RANK,
    cp_kwargs=CP_KWARGS,
)

seq.operators = ops
dirichlet = DIRICHLET

print(f"Built sequence with ns={NS}, p={P}, rank={RANK}, dirichlet={DIRICHLET}")


# %% k = 0 forward-model diagnostics
k0_size = seq.n0_dbc if dirichlet else seq.n0

def exact_apply_k0(x):
    return apply_stiffness(seq, ops, x, 0, dirichlet=dirichlet)


def model_apply_k0(x):
    return _apply_k0_tensor_hodge_forward_model(seq, ops, x, dirichlet=dirichlet)


k0_random = random_error_stats(exact_apply_k0, model_apply_k0, k0_size)
k0_dense = dense_comparison(exact_apply_k0, model_apply_k0, k0_size)
k0_exact_spec = eig_summary(k0_dense["exact"])
k0_model_spec = eig_summary(k0_dense["model"])
k0_factors = select_boundary_data(ops.k0_tensor_hodge_precond, dirichlet, "Tensor Hodge k=0")
k0_bulk_denom = k0_factors.bulk_modal_denom

k0_random, k0_dense["fro_relative_error"], k0_exact_spec["min"], k0_model_spec["min"]


# %% k = 0 modal / denominator inspection
k0_bulk_denom.shape, float(jnp.min(k0_bulk_denom)), float(jnp.max(k0_bulk_denom))


# %% k = 1 forward-model diagnostics
k1_size = seq.n1_dbc if dirichlet else seq.n1

def exact_apply_k1(x):
    return apply_stiffness(seq, ops, x, 1, dirichlet=dirichlet)


def model_apply_k1(x):
    return apply_stiffness_tensor_forward_model(seq, ops, x, 1, dirichlet=dirichlet)


k1_random = random_error_stats(exact_apply_k1, model_apply_k1, k1_size)
k1_dense = dense_comparison(exact_apply_k1, model_apply_k1, k1_size)
k1_exact_spec = eig_summary(k1_dense["exact"])
k1_model_spec = eig_summary(k1_dense["model"])
k1_payload = select_boundary_data(ops.k1_tensor_stiff_precond, dirichlet, "Tensor stiffness k=1")
k1_arr = k1_payload.factors.arr
k1_theta = k1_payload.factors.theta
k1_zeta = k1_payload.factors.zeta

k1_random, k1_dense["fro_relative_error"], k1_exact_spec["min"], k1_model_spec["min"]


# %% k = 1 block modal data
k1_arr_summary = summarize_modal_inverse(k1_arr)
k1_theta_summary = summarize_modal_inverse(k1_theta)
k1_zeta_summary = summarize_modal_inverse(k1_zeta)

k1_arr_summary, k1_theta_summary, k1_zeta_summary


# %% k = 1 modal finiteness check
k1_arr_finite = modal_finiteness_summary(k1_arr)
k1_theta_finite = modal_finiteness_summary(k1_theta)
k1_zeta_finite = modal_finiteness_summary(k1_zeta)

k1_arr_finite, k1_theta_finite, k1_zeta_finite


# %% k = 1 canonical reference masses
k1_reference_masses = canonical_k1_reference_masses(seq, dirichlet=dirichlet)
k1_reference_summary = reference_mass_summary(k1_reference_masses)

k1_reference_summary


# %% k = 1 arrays to inspect directly
k1_arr_lam_r = k1_arr.fd_lam_r
k1_arr_lam_t = k1_arr.fd_lam_t
k1_arr_lam_z = k1_arr.fd_lam_z
k1_arr_inv_denom = k1_arr.fd_inv_denom

k1_theta_lam_r = k1_theta.fd_lam_r
k1_theta_lam_t = k1_theta.fd_lam_t
k1_theta_lam_z = k1_theta.fd_lam_z
k1_theta_inv_denom = k1_theta.fd_inv_denom

k1_zeta_lam_r = k1_zeta.fd_lam_r
k1_zeta_lam_t = k1_zeta.fd_lam_t
k1_zeta_lam_z = k1_zeta.fd_lam_z
k1_zeta_inv_denom = k1_zeta.fd_inv_denom


# %% k = 2 forward-model diagnostics
k2_size = seq.n2_dbc if dirichlet else seq.n2

def exact_apply_k2(x):
    return apply_stiffness(seq, ops, x, 2, dirichlet=dirichlet)


def model_apply_k2(x):
    return apply_stiffness_tensor_forward_model(seq, ops, x, 2, dirichlet=dirichlet)


k2_random = random_error_stats(exact_apply_k2, model_apply_k2, k2_size)
k2_dense = dense_comparison(exact_apply_k2, model_apply_k2, k2_size)
k2_exact_spec = eig_summary(k2_dense["exact"])
k2_model_spec = eig_summary(k2_dense["model"])
k2_payload = select_boundary_data(ops.k2_tensor_stiff_precond, dirichlet, "Tensor stiffness k=2")
k2_r_bulk = k2_payload.factors.r_bulk
k2_theta = k2_payload.factors.theta
k2_zeta = k2_payload.factors.zeta

k2_random, k2_dense["fro_relative_error"], k2_exact_spec["min"], k2_model_spec["min"]


# %% k = 2 block modal data
k2_r_bulk_summary = summarize_modal_inverse(k2_r_bulk)
k2_theta_summary = summarize_modal_inverse(k2_theta)
k2_zeta_summary = summarize_modal_inverse(k2_zeta)

k2_r_bulk_summary, k2_theta_summary, k2_zeta_summary


# %% k = 2 modal finiteness check
k2_r_bulk_finite = modal_finiteness_summary(k2_r_bulk)
k2_theta_finite = modal_finiteness_summary(k2_theta)
k2_zeta_finite = modal_finiteness_summary(k2_zeta)

k2_r_bulk_finite, k2_theta_finite, k2_zeta_finite


# %% k = 2 canonical reference masses
k2_reference_masses = canonical_k2_reference_masses(seq, dirichlet=dirichlet)
k2_reference_summary = reference_mass_summary(k2_reference_masses)

k2_reference_summary


# %% Bulk-only dense checks
k1_bulk_indices = k1_payload.surgery.bulk_indices
k2_bulk_indices = k2_payload.surgery.bulk_indices
k0_bulk_indices = jnp.arange(k0_factors.core_size, k0_size)


def restrict_apply(full_size: int, indices: jnp.ndarray, apply_fn):
    def restricted(x):
        full = jnp.zeros((full_size,), dtype=x.dtype)
        full = full.at[indices].set(x)
        return apply_fn(full)[indices]

    return restricted


k0_bulk_dense = dense_comparison(
    restrict_apply(k0_size, k0_bulk_indices, exact_apply_k0),
    restrict_apply(k0_size, k0_bulk_indices, model_apply_k0),
    int(k0_bulk_indices.shape[0]),
)
k1_bulk_dense = dense_comparison(
    restrict_apply(k1_size, k1_bulk_indices, exact_apply_k1),
    restrict_apply(k1_size, k1_bulk_indices, model_apply_k1),
    int(k1_bulk_indices.shape[0]),
)
k2_bulk_dense = dense_comparison(
    restrict_apply(k2_size, k2_bulk_indices, exact_apply_k2),
    restrict_apply(k2_size, k2_bulk_indices, model_apply_k2),
    int(k2_bulk_indices.shape[0]),
)

{
    "k0_bulk_fro": k0_bulk_dense["fro_relative_error"],
    "k1_bulk_fro": k1_bulk_dense["fro_relative_error"],
    "k2_bulk_fro": k2_bulk_dense["fro_relative_error"],
}


# %% Convenience one-vector comparisons
k0_cmp = compare_random_vector(exact_apply_k0, model_apply_k0, k0_size, seed=SEED)
k1_cmp = compare_random_vector(exact_apply_k1, model_apply_k1, k1_size, seed=SEED)
k2_cmp = compare_random_vector(exact_apply_k2, model_apply_k2, k2_size, seed=SEED)

k0_cmp["relative_error"], k1_cmp["relative_error"], k2_cmp["relative_error"]

# %% Inspect canonical axis operators and generalized spectra

from mrx.operators import _assemble_unweighted_1d_stiffness
from mrx.preconditioners import _simultaneous_diagonalize_pair

def matrix_summary(name, A):
    A = _symmetrize(jnp.asarray(A, dtype=jnp.float64))
    eigs = jnp.linalg.eigvalsh(A)
    return {
        "name": name,
        "shape": A.shape,
        "min_eig": float(jnp.min(eigs)),
        "max_eig": float(jnp.max(eigs)),
        "count_negative": int(jnp.sum(eigs < -1e-12)),
        "count_near_zero": int(jnp.sum(jnp.abs(eigs) <= 1e-12)),
        "count_positive": int(jnp.sum(eigs > 1e-12)),
    }

def generalized_summary(name, M, A):
    _, lam = _simultaneous_diagonalize_pair(M, A)
    return {
        "name": name,
        "min_lam": float(jnp.min(lam)),
        "max_lam": float(jnp.max(lam)),
        "count_negative": int(jnp.sum(lam < -1e-12)),
        "count_near_zero": int(jnp.sum(jnp.abs(lam) <= 1e-12)),
        "count_positive": int(jnp.sum(lam > 1e-12)),
        "lam": lam,
    }

k1_model = ops.k1_tensor_stiff_model
k2_model = ops.k2_tensor_stiff_model

# k = 1 canonical active-axis operators
k1_full_stiff_r = _assemble_unweighted_1d_stiffness(
    seq.basis_r_jk,
    seq.d_basis_r_jk,
    seq.quad.w_x,
    k1_model.g_r,
)
k1_arr_op_t = _assemble_unweighted_1d_stiffness(
    seq.basis_t_jk,
    seq.d_basis_t_jk,
    seq.quad.w_y,
    k1_model.g_t,
)
k1_arr_op_z = _assemble_unweighted_1d_stiffness(
    seq.basis_z_jk,
    seq.d_basis_z_jk,
    seq.quad.w_z,
    k1_model.g_z,
)

k1_theta_op_r = _restrict_radial_mass(k1_full_stiff_r, 2, k1_theta.shape[0])
k1_theta_op_z = k1_arr_op_z

k1_zeta_op_r = _restrict_radial_mass(k1_full_stiff_r, 2, k1_zeta.shape[0])
k1_zeta_op_t = k1_arr_op_t

# k = 2 canonical active-axis operators
k2_full_stiff_r = _assemble_unweighted_1d_stiffness(
    seq.basis_r_jk,
    seq.d_basis_r_jk,
    seq.quad.w_x,
    k2_model.g_r,
)
k2_r_bulk_op_r = _restrict_radial_mass(k2_full_stiff_r, 2, k2_r_bulk.shape[0])
k2_theta_op_t = _assemble_unweighted_1d_stiffness(
    seq.basis_t_jk,
    seq.d_basis_t_jk,
    seq.quad.w_y,
    k2_model.g_t,
)
k2_zeta_op_z = _assemble_unweighted_1d_stiffness(
    seq.basis_z_jk,
    seq.d_basis_z_jk,
    seq.quad.w_z,
    k2_model.g_z,
)

axis_operator_checks = {
    "k1_arr_op_t": matrix_summary("k1_arr_op_t", k1_arr_op_t),
    "k1_arr_op_z": matrix_summary("k1_arr_op_z", k1_arr_op_z),
    "k1_theta_op_r": matrix_summary("k1_theta_op_r", k1_theta_op_r),
    "k1_theta_op_z": matrix_summary("k1_theta_op_z", k1_theta_op_z),
    "k1_zeta_op_r": matrix_summary("k1_zeta_op_r", k1_zeta_op_r),
    "k1_zeta_op_t": matrix_summary("k1_zeta_op_t", k1_zeta_op_t),
    "k2_r_bulk_op_r": matrix_summary("k2_r_bulk_op_r", k2_r_bulk_op_r),
    "k2_theta_op_t": matrix_summary("k2_theta_op_t", k2_theta_op_t),
    "k2_zeta_op_z": matrix_summary("k2_zeta_op_z", k2_zeta_op_z),
}

axis_operator_checks
# %%
# %% Compare those assembled operators against the generalized spectra you see in fd_lam_*

k1_generalized_checks = {
    "arr_t": generalized_summary("arr_t", k1_reference_masses["arr"]["t"], k1_arr_op_t),
    "arr_z": generalized_summary("arr_z", k1_reference_masses["arr"]["z"], k1_arr_op_z),
    "theta_r": generalized_summary("theta_r", k1_reference_masses["theta"]["r"], k1_theta_op_r),
    "theta_z": generalized_summary("theta_z", k1_reference_masses["theta"]["z"], k1_theta_op_z),
    "zeta_r": generalized_summary("zeta_r", k1_reference_masses["zeta"]["r"], k1_zeta_op_r),
    "zeta_t": generalized_summary("zeta_t", k1_reference_masses["zeta"]["t"], k1_zeta_op_t),
}

k2_generalized_checks = {
    "r_bulk_r": generalized_summary("r_bulk_r", k2_reference_masses["r_bulk"]["r"], k2_r_bulk_op_r),
    "theta_t": generalized_summary("theta_t", k2_reference_masses["theta"]["t"], k2_theta_op_t),
    "zeta_z": generalized_summary("zeta_z", k2_reference_masses["zeta"]["z"], k2_zeta_op_z),
}

k1_generalized_checks, k2_generalized_checks
# %%
# %% Recompute raw CP factors and inspect sign patterns

from mrx.operators import _k2_divdiv_weight_tensor

from mrx.preconditioners import _cp_als_3tensor

def factor_sign_summary(name, vec):
    vec = jnp.asarray(vec)
    return {
        "name": name,
        "min": float(jnp.min(vec)),
        "max": float(jnp.max(vec)),
        "all_nonnegative": bool(jnp.all(vec >= 0.0)),
        "all_nonpositive": bool(jnp.all(vec <= 0.0)),
        "count_negative": int(jnp.sum(vec < 0.0)),
        "count_positive": int(jnp.sum(vec > 0.0)),
        "count_zero": int(jnp.sum(vec == 0.0)),
        "values": vec,
    }

def cp_tensor_sign_report(name, tensor, rank=RANK):
    weights, factors, rel_err, final_delta, n_iter = _cp_als_3tensor(
        tensor,
        rank,
        maxiter=CP_KWARGS["maxiter"],
        tol=CP_KWARGS["tol"],
        ridge=CP_KWARGS["ridge"],
    )
    factor_theta, factor_r, factor_z = factors
    report = {
        "name": name,
        "weights": weights,
        "relative_error": rel_err,
        "final_delta": final_delta,
        "n_iter": n_iter,
        "theta": [],
        "r": [],
        "z": [],
    }
    for j in range(rank):
        report["theta"].append(factor_sign_summary(f"{name}.theta[{j}]", factor_theta[:, j]))
        report["r"].append(factor_sign_summary(f"{name}.r[{j}]", factor_r[:, j]))
        report["z"].append(factor_sign_summary(f"{name}.z[{j}]", factor_z[:, j]))
    return report

# k=1 uses the diagonal k=2 metric tensor entries J g^{ii}
metric2 = _reshape_quadrature_matrix_field(seq, seq.geometry.metric_jkl)
jac = _reshape_quadrature_scalar_field(seq, seq.geometry.jacobian_j)
k1_beta_rr = jac * metric2[..., 0, 0]
k1_beta_tt = jac * metric2[..., 1, 1]
k1_beta_zz = jac * metric2[..., 2, 2]

k2_weight = _k2_divdiv_weight_tensor(seq)

cp_reports = {
    "k1_beta_rr": cp_tensor_sign_report("k1_beta_rr", k1_beta_rr, rank=RANK),
    "k1_beta_tt": cp_tensor_sign_report("k1_beta_tt", k1_beta_tt, rank=RANK),
    "k1_beta_zz": cp_tensor_sign_report("k1_beta_zz", k1_beta_zz, rank=RANK),
    "k2_divdiv_weight": cp_tensor_sign_report("k2_divdiv_weight", k2_weight, rank=RANK),
}

cp_reports
# %%
# %% Check whether assembled 1D CP term matrices have negative eigenvalues / entries

def entry_and_spectrum_summary(name, A):
    A = _symmetrize(jnp.asarray(A, dtype=jnp.float64))
    eigs = jnp.linalg.eigvalsh(A)
    return {
        "name": name,
        "min_entry": float(jnp.min(A)),
        "max_entry": float(jnp.max(A)),
        "all_entries_nonnegative": bool(jnp.all(A >= 0.0)),
        "min_eig": float(jnp.min(eigs)),
        "max_eig": float(jnp.max(eigs)),
        "count_negative_eigs": int(jnp.sum(eigs < -1e-12)),
    }

assembled_term_checks = {
    "k1_tt_mass_t_terms": [entry_and_spectrum_summary(f"k1_tt_mass_t[{j}]", A) for j, A in enumerate(k1_model.tt_mass_t_terms)],
    "k1_zz_mass_t_terms": [entry_and_spectrum_summary(f"k1_zz_mass_t[{j}]", A) for j, A in enumerate(k1_model.zz_mass_t_terms)],
    "k1_tt_mass_z_terms": [entry_and_spectrum_summary(f"k1_tt_mass_z[{j}]", A) for j, A in enumerate(k1_model.tt_mass_z_terms)],
    "k2_mass_r_terms": [entry_and_spectrum_summary(f"k2_mass_r[{j}]", A) for j, A in enumerate(k2_model.mass_r_terms)],
    "k2_mass_t_terms": [entry_and_spectrum_summary(f"k2_mass_t[{j}]", A) for j, A in enumerate(k2_model.mass_t_terms)],
    "k2_mass_z_terms": [entry_and_spectrum_summary(f"k2_mass_z[{j}]", A) for j, A in enumerate(k2_model.mass_z_terms)],
}

assembled_term_checks
# %%
# %% Check whether assembled 1D CP term matrices have negative eigenvalues / entries

def entry_and_spectrum_summary(name, A):
    A = _symmetrize(jnp.asarray(A, dtype=jnp.float64))
    eigs = jnp.linalg.eigvalsh(A)
    return {
        "name": name,
        "min_entry": float(jnp.min(A)),
        "max_entry": float(jnp.max(A)),
        "all_entries_nonnegative": bool(jnp.all(A >= 0.0)),
        "min_eig": float(jnp.min(eigs)),
        "max_eig": float(jnp.max(eigs)),
        "count_negative_eigs": int(jnp.sum(eigs < -1e-12)),
    }

assembled_term_checks = {
    "k1_tt_mass_t_terms": [entry_and_spectrum_summary(f"k1_tt_mass_t[{j}]", A) for j, A in enumerate(k1_model.tt_mass_t_terms)],
    "k1_zz_mass_t_terms": [entry_and_spectrum_summary(f"k1_zz_mass_t[{j}]", A) for j, A in enumerate(k1_model.zz_mass_t_terms)],
    "k1_tt_mass_z_terms": [entry_and_spectrum_summary(f"k1_tt_mass_z[{j}]", A) for j, A in enumerate(k1_model.tt_mass_z_terms)],
    "k2_mass_r_terms": [entry_and_spectrum_summary(f"k2_mass_r[{j}]", A) for j, A in enumerate(k2_model.mass_r_terms)],
    "k2_mass_t_terms": [entry_and_spectrum_summary(f"k2_mass_t[{j}]", A) for j, A in enumerate(k2_model.mass_t_terms)],
    "k2_mass_z_terms": [entry_and_spectrum_summary(f"k2_mass_z[{j}]", A) for j, A in enumerate(k2_model.mass_z_terms)],
}

assembled_term_checks
# %%
# %% Deflated tensor stiffness solve smoke test

from mrx.nullspace import get_stiffness_nullspace
from mrx.operators import apply_mass_matrix, apply_stiffness_tensor_preconditioner
from mrx.solvers import solve_singular_cg


def _project_primal(x, vs, mass_matvec):
    if vs.shape[0] == 0:
        return x
    coeffs = vs @ mass_matvec(x)
    return x - coeffs @ vs


def _project_dual(f, vs, mass_matvec):
    if vs.shape[0] == 0:
        return f
    mass_vs = jax.vmap(mass_matvec)(vs)
    coeffs = vs @ f
    return f - coeffs @ mass_vs


def _primal_kernel_overlap(x, vs, mass_matvec):
    if vs.shape[0] == 0:
        return {
            "coeff_norm": 0.0,
            "relative_coeff_norm": 0.0,
            "projected_relative_change": 0.0,
        }
    coeffs = vs @ mass_matvec(x)
    projected = _project_primal(x, vs, mass_matvec)
    return {
        "coeff_norm": float(jnp.linalg.norm(coeffs)),
        "relative_coeff_norm": float(
            jnp.linalg.norm(coeffs) / jnp.maximum(jnp.linalg.norm(x), 1e-30)
        ),
        "projected_relative_change": float(
            jnp.linalg.norm(x - projected) / jnp.maximum(jnp.linalg.norm(x), 1e-30)
        ),
    }


def _dual_kernel_overlap(f, vs, mass_matvec):
    if vs.shape[0] == 0:
        return {
            "coeff_norm": 0.0,
            "relative_coeff_norm": 0.0,
            "projected_relative_change": 0.0,
        }
    coeffs = vs @ f
    projected = _project_dual(f, vs, mass_matvec)
    return {
        "coeff_norm": float(jnp.linalg.norm(coeffs)),
        "relative_coeff_norm": float(
            jnp.linalg.norm(coeffs) / jnp.maximum(jnp.linalg.norm(f), 1e-30)
        ),
        "projected_relative_change": float(
            jnp.linalg.norm(f - projected) / jnp.maximum(jnp.linalg.norm(f), 1e-30)
        ),
    }


def run_deflated_tensor_stiffness_solve(
    k,
    *,
    operators_override=None,
    dirichlet_case: bool | None = None,
    seed=0,
    tol=1e-8,
    maxiter=500,
):
    operators_bundle = ops if operators_override is None else operators_override
    dirichlet_case = dirichlet if dirichlet_case is None else dirichlet_case
    size = getattr(seq, f"n{k}_dbc" if dirichlet_case else f"n{k}")

    def M(x):
        return apply_mass_matrix(seq, operators_bundle, x, k, dirichlet=dirichlet_case)

    def K(x):
        return apply_stiffness(seq, operators_bundle, x, k, dirichlet=dirichlet_case)

    P = lambda x: apply_stiffness_tensor_preconditioner(seq, operators_bundle, x, k, dirichlet=dirichlet_case)
    vs = get_stiffness_nullspace(seq, operators_bundle, k, dirichlet_case)

    key = jax.random.PRNGKey(seed)
    x_true_raw = jax.random.normal(key, (size,), dtype=jnp.float64)
    x_true = _project_primal(x_true_raw, vs, M)
    rhs = K(x_true)
    rhs_projected = _project_dual(rhs, vs, M)

    x, info = solve_singular_cg(
        K,
        rhs,
        mass_matvec=M,
        precond_matvec=P,
        vs=vs,
        tol=tol,
        maxiter=maxiter,
    )

    residual = _project_dual(rhs - K(x), vs, M)
    err = _project_primal(x - x_true, vs, M)

    return {
        "k": k,
        "dirichlet": bool(dirichlet_case),
        "raw_info": int(info),
        "converged": bool(info < 0),
        "iterations": int(abs(info)),
        "nullspace_dim": int(vs.shape[0]),
        "relative_residual": float(
            jnp.linalg.norm(residual) / jnp.maximum(jnp.linalg.norm(rhs), 1e-30)
        ),
        "relative_solution_error": float(
            jnp.linalg.norm(err) / jnp.maximum(jnp.linalg.norm(x_true), 1e-30)
        ),
        "x_true_raw_kernel_overlap": _primal_kernel_overlap(x_true_raw, vs, M),
        "x_true_kernel_overlap": _primal_kernel_overlap(x_true, vs, M),
        "rhs_kernel_overlap": _dual_kernel_overlap(rhs, vs, M),
        "rhs_projected_relative_change": float(
            jnp.linalg.norm(rhs - rhs_projected) / jnp.maximum(jnp.linalg.norm(rhs), 1e-30)
        ),
        "error_kernel_overlap": _primal_kernel_overlap(x - x_true, vs, M),
        "residual_kernel_overlap": _dual_kernel_overlap(rhs - K(x), vs, M),
    }


harmonic_free_stiffness_solves = {
    "k1_dbc": run_deflated_tensor_stiffness_solve(1, dirichlet_case=True, seed=SEED),
    "k2_free": run_deflated_tensor_stiffness_solve(2, dirichlet_case=False, seed=SEED),
}

harmonic_free_stiffness_solves
# %%

# %% Dense stiffness nullspace diagnostics

from mrx.nullspace import get_nullspace
from mrx.preconditioners import _simultaneous_diagonalize_pair


def _mass_orthonormalize_rows(rows: jnp.ndarray, mass_matrix: jnp.ndarray, tol: float = 1e-10):
    rows = jnp.asarray(rows, dtype=jnp.float64)
    if rows.size == 0:
        return jnp.zeros((0, mass_matrix.shape[0]), dtype=jnp.float64)
    gram = _symmetrize(rows @ mass_matrix @ rows.T)
    eigvals, eigvecs = jnp.linalg.eigh(gram)
    scale = jnp.max(jnp.abs(eigvals)) if eigvals.size else jnp.asarray(0.0, dtype=jnp.float64)
    threshold = tol * jnp.maximum(scale, 1.0)
    keep = eigvals > threshold
    if not bool(jnp.any(keep)):
        return jnp.zeros((0, mass_matrix.shape[0]), dtype=jnp.float64)
    kept_vals = eigvals[keep]
    kept_vecs = eigvecs[:, keep]
    return (kept_vecs / jnp.sqrt(kept_vals)[None, :]).T @ rows


def _stiffness_exact_image_basis(
    k: int,
    mass_matrix: jnp.ndarray,
    *,
    dirichlet_case: bool,
    tol: float = 1e-10,
):
    n_prev = getattr(seq, f"n{k - 1}_dbc" if dirichlet_case else f"n{k - 1}")
    basis_prev = jnp.eye(n_prev, dtype=jnp.float64)
    if k == 1:
        image = jax.vmap(
            lambda v: seq.apply_strong_grad(v, dirichlet_in=dirichlet_case, dirichlet_out=dirichlet_case)
        )(basis_prev).T
    elif k == 2:
        image = jax.vmap(
            lambda v: seq.apply_strong_curl(v, dirichlet_in=dirichlet_case, dirichlet_out=dirichlet_case)
        )(basis_prev).T
    else:
        raise ValueError(f"stiffness exact-image basis only supports k=1 or k=2 (got {k})")
    return _mass_orthonormalize_rows(image.T, mass_matrix, tol=tol)


def _dense_stiffness_generalized_nullspace(stiffness_matrix: jnp.ndarray, mass_matrix: jnp.ndarray, tol: float = 1e-10):
    basis, lam = _simultaneous_diagonalize_pair(mass_matrix, stiffness_matrix)
    lam = jnp.asarray(lam, dtype=jnp.float64)
    scale = jnp.max(jnp.abs(lam)) if lam.size else jnp.asarray(0.0, dtype=jnp.float64)
    threshold = tol * jnp.maximum(scale, 1.0)
    keep = jnp.abs(lam) <= threshold
    return basis[:, keep].T, lam, float(threshold)


def _subspace_projection_defect(source_rows: jnp.ndarray, target_rows: jnp.ndarray, mass_matrix: jnp.ndarray) -> float:
    if source_rows.shape[0] == 0:
        return 0.0
    if target_rows.shape[0] == 0:
        return float(jnp.sqrt(jnp.trace(source_rows @ mass_matrix @ source_rows.T)))
    coeffs = source_rows @ mass_matrix @ target_rows.T
    residual = source_rows - coeffs @ target_rows
    defect = _symmetrize(residual @ mass_matrix @ residual.T)
    return float(jnp.sqrt(jnp.maximum(jnp.trace(defect), 0.0)))


def dense_stiffness_nullspace_summary(
    k: int,
    *,
    operators_override=None,
    dirichlet_case: bool | None = None,
    tol: float = 1e-10,
):
    operators_bundle = ops if operators_override is None else operators_override
    dirichlet_case = dirichlet if dirichlet_case is None else dirichlet_case
    if k not in (1, 2):
        raise ValueError(f"dense stiffness nullspace diagnostics only support k=1 or k=2 (got {k})")
    size = getattr(seq, f"n{k}_dbc" if dirichlet_case else f"n{k}")
    stiffness_matrix = _symmetrize(_assemble_dense_from_apply(
        lambda x: apply_stiffness(seq, operators_bundle, x, k, dirichlet=dirichlet_case),
        size,
    ))

    mass_matrix = _symmetrize(_assemble_dense_from_apply(
        lambda x: apply_mass_matrix(seq, operators_bundle, x, k, dirichlet=dirichlet_case),
        size,
    ))
    dense_null_rows, lam, lam_tol = _dense_stiffness_generalized_nullspace(
        stiffness_matrix,
        mass_matrix,
        tol=tol,
    )
    exact_rows = _stiffness_exact_image_basis(k, mass_matrix, dirichlet_case=dirichlet_case, tol=tol)
    harmonic_rows = _mass_orthonormalize_rows(get_nullspace(operators_bundle, k, dirichlet_case), mass_matrix, tol=tol)
    deflated_rows = _mass_orthonormalize_rows(get_stiffness_nullspace(seq, operators_bundle, k, dirichlet_case), mass_matrix, tol=tol)

    def _basis_residual(rows):
        if rows.shape[0] == 0:
            return 0.0
        residual = stiffness_matrix @ rows.T
        return float(jnp.linalg.norm(residual))

    cross_exact_harm = exact_rows @ mass_matrix @ harmonic_rows.T if harmonic_rows.shape[0] else jnp.zeros((exact_rows.shape[0], 0), dtype=jnp.float64)
    positive_lam = lam[lam > lam_tol]

    return {
        "k": k,
        "dirichlet": bool(dirichlet_case),
        "dense_null_dim": int(dense_null_rows.shape[0]),
        "exact_dim": int(exact_rows.shape[0]),
        "harmonic_dim": int(harmonic_rows.shape[0]),
        "deflated_dim": int(deflated_rows.shape[0]),
        "dense_lam_tol": lam_tol,
        "max_dense_null_lam_abs": float(jnp.max(jnp.abs(lam[jnp.abs(lam) <= lam_tol]))) if bool(jnp.any(jnp.abs(lam) <= lam_tol)) else None,
        "min_positive_lam": float(jnp.min(positive_lam)) if positive_lam.size else None,
        "exact_harmonic_mass_overlap": float(jnp.linalg.norm(cross_exact_harm)) if cross_exact_harm.size else 0.0,
        "dense_basis_stiffness_residual": _basis_residual(dense_null_rows),
        "exact_basis_stiffness_residual": _basis_residual(exact_rows),
        "harmonic_basis_stiffness_residual": _basis_residual(harmonic_rows),
        "deflated_basis_stiffness_residual": _basis_residual(deflated_rows),
        "dense_to_deflated_defect": _subspace_projection_defect(dense_null_rows, deflated_rows, mass_matrix),
        "deflated_to_dense_defect": _subspace_projection_defect(deflated_rows, dense_null_rows, mass_matrix),
        "exact_to_dense_defect": _subspace_projection_defect(exact_rows, dense_null_rows, mass_matrix),
        "harmonic_to_dense_defect": _subspace_projection_defect(harmonic_rows, dense_null_rows, mass_matrix),
        "deflated_mass_orthogonality_defect": float(jnp.linalg.norm(deflated_rows @ mass_matrix @ deflated_rows.T - jnp.eye(deflated_rows.shape[0], dtype=jnp.float64))) if deflated_rows.shape[0] else 0.0,
        "dense_null_lam": lam[jnp.abs(lam) <= lam_tol],
    }


harmonic_free_nullspace_debug = {
    "k1_dbc": dense_stiffness_nullspace_summary(1, dirichlet_case=True),
    "k2_free": dense_stiffness_nullspace_summary(2, dirichlet_case=False),
}

harmonic_free_nullspace_debug
# %%

# %% Dense preconditioned stiffness diagnostics on the positive subspace

def dense_preconditioned_stiffness_summary(
    k: int,
    *,
    operators_override=None,
    dirichlet_case: bool,
    tol: float = 1e-10,
):
    operators_bundle = ops if operators_override is None else operators_override
    size = getattr(seq, f"n{k}_dbc" if dirichlet_case else f"n{k}")

    def K_apply(x):
        return apply_stiffness(seq, operators_bundle, x, k, dirichlet=dirichlet_case)

    def M_apply(x):
        return apply_mass_matrix(seq, operators_bundle, x, k, dirichlet=dirichlet_case)

    def P_apply(x):
        return apply_stiffness_tensor_preconditioner(seq, operators_bundle, x, k, dirichlet=dirichlet_case)

    K_dense = _symmetrize(_assemble_dense_from_apply(K_apply, size))
    M_dense = _symmetrize(_assemble_dense_from_apply(M_apply, size))
    P_dense = _assemble_dense_from_apply(P_apply, size)
    P_sym = _symmetrize(P_dense)
    V, lam = _simultaneous_diagonalize_pair(M_dense, K_dense)
    lam = jnp.asarray(lam, dtype=jnp.float64)
    lam_scale = jnp.max(jnp.abs(lam)) if lam.size else jnp.asarray(0.0, dtype=jnp.float64)
    lam_tol = tol * jnp.maximum(lam_scale, 1.0)
    keep = lam > lam_tol
    V_pos = V[:, keep]

    if V_pos.shape[1] == 0:
        return {
            "k": k,
            "dirichlet": bool(dirichlet_case),
            "positive_dim": 0,
        }

    def preconditioned_column(v):
        return P_apply(K_apply(v))

    PKV = jax.vmap(preconditioned_column, in_axes=1, out_axes=1)(V_pos)
    PK_coeff = V_pos.T @ (M_dense @ PKV)
    identity = jnp.eye(PK_coeff.shape[0], dtype=jnp.float64)
    eigs = jnp.linalg.eigvals(PK_coeff)
    eigs_real = jnp.real(eigs)
    eigs_imag = jnp.imag(eigs)
    positive_real = eigs_real[eigs_real > 1e-12]

    P_pos = V_pos.T @ (M_dense @ (P_sym @ M_dense)) @ V_pos
    P_pos_eigs = jnp.linalg.eigvalsh(_symmetrize(P_pos))

    return {
        "k": k,
        "dirichlet": bool(dirichlet_case),
        "positive_dim": int(V_pos.shape[1]),
        "pk_minus_identity_norm": float(jnp.linalg.norm(PK_coeff - identity)),
        "pk_relative_identity_error": float(
            jnp.linalg.norm(PK_coeff - identity) / jnp.maximum(jnp.linalg.norm(identity), 1e-30)
        ),
        "pk_symmetry_defect": float(jnp.linalg.norm(PK_coeff - PK_coeff.T)),
        "pk_max_imag_eig": float(jnp.max(jnp.abs(eigs_imag))),
        "pk_min_real_eig": float(jnp.min(eigs_real)),
        "pk_max_real_eig": float(jnp.max(eigs_real)),
        "pk_condition_estimate": float(jnp.max(positive_real) / jnp.min(positive_real)) if positive_real.size else None,
        "p_symmetry_defect": float(jnp.linalg.norm(P_dense - P_dense.T)),
        "p_min_sym_eig_on_pos": float(jnp.min(P_pos_eigs)),
        "p_max_sym_eig_on_pos": float(jnp.max(P_pos_eigs)),
        "positive_lam": lam[keep],
        "pk_eigs": eigs,
    }


def _block_submatrix(matrix: jnp.ndarray, row_indices: jnp.ndarray, col_indices: jnp.ndarray) -> jnp.ndarray:
    return matrix[row_indices][:, col_indices]


def _transpose_defect_summary(matrix: jnp.ndarray, row_indices: jnp.ndarray, col_indices: jnp.ndarray):
    forward = _block_submatrix(matrix, row_indices, col_indices)
    reverse = _block_submatrix(matrix, col_indices, row_indices)
    defect = jnp.linalg.norm(forward - reverse.T)
    scale = jnp.maximum(jnp.maximum(jnp.linalg.norm(forward), jnp.linalg.norm(reverse)), 1e-30)
    return {
        "shape": tuple(forward.shape),
        "absolute": float(defect),
        "relative": float(defect / scale),
    }


def _dense_local_preconditioned_summary(
    K_block: jnp.ndarray,
    M_block: jnp.ndarray,
    P_block: jnp.ndarray,
    *,
    tol: float = 1e-10,
):
    K_block = _symmetrize(K_block)
    M_block = _symmetrize(M_block)
    P_sym = _symmetrize(P_block)
    V, lam = _simultaneous_diagonalize_pair(M_block, K_block)
    lam = jnp.asarray(lam, dtype=jnp.float64)
    lam_scale = jnp.max(jnp.abs(lam)) if lam.size else jnp.asarray(0.0, dtype=jnp.float64)
    lam_tol = tol * jnp.maximum(lam_scale, 1.0)
    keep = lam > lam_tol
    V_pos = V[:, keep]

    summary = {
        "size": int(K_block.shape[0]),
        "positive_dim": int(V_pos.shape[1]),
        "k_symmetry_defect": float(jnp.linalg.norm(K_block - K_block.T)),
        "m_symmetry_defect": float(jnp.linalg.norm(M_block - M_block.T)),
        "p_symmetry_defect": float(jnp.linalg.norm(P_block - P_block.T)),
        "min_positive_lam": float(jnp.min(lam[keep])) if bool(jnp.any(keep)) else None,
        "max_positive_lam": float(jnp.max(lam[keep])) if bool(jnp.any(keep)) else None,
    }
    if V_pos.shape[1] == 0:
        return summary

    PK_coeff = V_pos.T @ (M_block @ (P_block @ (K_block @ V_pos)))
    identity = jnp.eye(PK_coeff.shape[0], dtype=jnp.float64)
    eigs = jnp.linalg.eigvals(PK_coeff)
    eigs_real = jnp.real(eigs)
    eigs_imag = jnp.imag(eigs)
    positive_real = eigs_real[eigs_real > 1e-12]
    P_pos = V_pos.T @ (M_block @ (P_sym @ M_block)) @ V_pos
    P_pos_eigs = jnp.linalg.eigvalsh(_symmetrize(P_pos))

    summary.update({
        "pk_minus_identity_norm": float(jnp.linalg.norm(PK_coeff - identity)),
        "pk_relative_identity_error": float(
            jnp.linalg.norm(PK_coeff - identity) / jnp.maximum(jnp.linalg.norm(identity), 1e-30)
        ),
        "pk_symmetry_defect": float(jnp.linalg.norm(PK_coeff - PK_coeff.T)),
        "pk_max_imag_eig": float(jnp.max(jnp.abs(eigs_imag))),
        "pk_min_real_eig": float(jnp.min(eigs_real)),
        "pk_max_real_eig": float(jnp.max(eigs_real)),
        "pk_condition_estimate": float(jnp.max(positive_real) / jnp.min(positive_real)) if positive_real.size else None,
        "p_min_sym_eig_on_pos": float(jnp.min(P_pos_eigs)),
        "p_max_sym_eig_on_pos": float(jnp.max(P_pos_eigs)),
    })
    return summary


def _symmetric_pseudoinverse_local(matrix: jnp.ndarray, *, relative_tol: float = 1e-8) -> jnp.ndarray:
    matrix = _symmetrize(matrix)
    eigvals, eigvecs = jnp.linalg.eigh(matrix)
    scale = jnp.max(jnp.abs(eigvals)) if eigvals.size else jnp.asarray(0.0, dtype=matrix.dtype)
    safe_scale = jnp.where(scale > 0, scale, 1.0)
    cutoff = relative_tol * safe_scale
    inv_eigvals = jnp.where(jnp.abs(eigvals) > cutoff, 1.0 / eigvals, 0.0)
    return _symmetrize((eigvecs * inv_eigvals[jnp.newaxis, :]) @ eigvecs.T)


def _relative_fro_error_matrix(model: jnp.ndarray, exact: jnp.ndarray) -> float:
    return float(jnp.linalg.norm(model - exact) / jnp.maximum(jnp.linalg.norm(exact), 1e-30))


def _k1_bulk_inverse_dense(
    operators_bundle,
    *,
    dirichlet_case: bool,
) -> tuple[object, object, jnp.ndarray]:
    payload = select_boundary_data(
        operators_bundle.k1_tensor_stiff_precond,
        dirichlet_case,
        "Tensor stiffness k=1",
    )
    surgery = payload.surgery
    factors = payload.factors
    bulk_apply = _apply_k1_bulk_preconditioner if factors.use_inner_schur else _apply_k1_bulk_diagonal_preconditioner
    bulk_inv = _assemble_dense_from_apply(
        lambda x: bulk_apply(surgery, factors.arr, factors.theta, factors.zeta, x),
        int(surgery.bulk_indices.shape[0]),
    )
    return surgery, factors, bulk_inv


def dense_k1_bulk_inverse_summary(
    *,
    operators_override=None,
    dirichlet_case: bool,
    tol: float = 1e-10,
):
    operators_bundle = ops if operators_override is None else operators_override
    size = getattr(seq, f"n1_dbc" if dirichlet_case else f"n1")

    def K_apply(x):
        return apply_stiffness(seq, operators_bundle, x, 1, dirichlet=dirichlet_case)

    def M_apply(x):
        return apply_mass_matrix(seq, operators_bundle, x, 1, dirichlet=dirichlet_case)

    K_dense = _symmetrize(_assemble_dense_from_apply(K_apply, size))
    M_dense = _symmetrize(_assemble_dense_from_apply(M_apply, size))
    surgery, factors, bulk_inv = _k1_bulk_inverse_dense(operators_bundle, dirichlet_case=dirichlet_case)
    bulk_indices = surgery.bulk_indices
    K_bulk = _block_submatrix(K_dense, bulk_indices, bulk_indices)
    M_bulk = _block_submatrix(M_dense, bulk_indices, bulk_indices)
    exact_bulk_pinv = _symmetric_pseudoinverse_local(K_bulk, relative_tol=1e-8)

    summary = _dense_local_preconditioned_summary(K_bulk, M_bulk, bulk_inv, tol=tol)
    summary.update({
        "use_inner_schur": bool(factors.use_inner_schur),
        "bulk_inverse_vs_exact_pinv_relative_error": _relative_fro_error_matrix(bulk_inv, exact_bulk_pinv),
    })
    return summary


def _schur_split_block_summary(matrix: jnp.ndarray, theta_indices: jnp.ndarray, zeta_indices: jnp.ndarray):
    theta_theta = _block_submatrix(matrix, theta_indices, theta_indices)
    zeta_zeta = _block_submatrix(matrix, zeta_indices, zeta_indices)
    theta_zeta = _block_submatrix(matrix, theta_indices, zeta_indices)
    return {
        "theta_theta": eig_summary(theta_theta),
        "zeta_zeta": eig_summary(zeta_zeta),
        "theta_zeta_norm": float(jnp.linalg.norm(theta_zeta)),
    }


def _schur_correction_split_summary(
    correction: jnp.ndarray,
    exact_correction: jnp.ndarray,
    theta_indices: jnp.ndarray,
    zeta_indices: jnp.ndarray,
):
    correction = _symmetrize(correction)
    exact_correction = _symmetrize(exact_correction)
    theta_theta = _block_submatrix(correction, theta_indices, theta_indices)
    zeta_zeta = _block_submatrix(correction, zeta_indices, zeta_indices)
    theta_zeta = _block_submatrix(correction, theta_indices, zeta_indices)
    exact_theta_theta = _block_submatrix(exact_correction, theta_indices, theta_indices)
    exact_zeta_zeta = _block_submatrix(exact_correction, zeta_indices, zeta_indices)
    exact_theta_zeta = _block_submatrix(exact_correction, theta_indices, zeta_indices)
    return {
        "full_relative_error": _relative_fro_error_matrix(correction, exact_correction),
        "theta_theta": eig_summary(theta_theta),
        "zeta_zeta": eig_summary(zeta_zeta),
        "theta_zeta_norm": float(jnp.linalg.norm(theta_zeta)),
        "theta_theta_relative_error": _relative_fro_error_matrix(theta_theta, exact_theta_theta),
        "zeta_zeta_relative_error": _relative_fro_error_matrix(zeta_zeta, exact_zeta_zeta),
        "theta_zeta_relative_error": _relative_fro_error_matrix(theta_zeta, exact_theta_zeta),
    }


def dense_k1_surgery_schur_summary(
    *,
    operators_override=None,
    dirichlet_case: bool,
):
    operators_bundle = ops if operators_override is None else operators_override
    size = getattr(seq, f"n1_dbc" if dirichlet_case else f"n1")

    def K_apply(x):
        return apply_stiffness(seq, operators_bundle, x, 1, dirichlet=dirichlet_case)

    K_dense = _symmetrize(_assemble_dense_from_apply(K_apply, size))
    surgery, factors, bulk_inv = _k1_bulk_inverse_dense(operators_bundle, dirichlet_case=dirichlet_case)
    surgery_indices = surgery.surgery_indices
    bulk_indices = surgery.bulk_indices
    ass = _block_submatrix(K_dense, surgery_indices, surgery_indices)
    asb = _block_submatrix(K_dense, surgery_indices, bulk_indices)
    abs_ = _block_submatrix(K_dense, bulk_indices, surgery_indices)
    abb = _block_submatrix(K_dense, bulk_indices, bulk_indices)

    exact_bulk_pinv = _symmetric_pseudoinverse_local(abb, relative_tol=1e-8)
    exact_schur = _symmetrize(ass - asb @ exact_bulk_pinv @ abs_)
    approx_schur = _symmetrize(ass - asb @ bulk_inv @ abs_)
    schur_from_inv = None if factors.schur_inv is None else _symmetric_pseudoinverse_local(factors.schur_inv, relative_tol=1e-8)

    slices = _surgery_slices_k1(seq, dirichlet_case)
    theta_size = slices["theta_surgery"].stop - slices["theta_surgery"].start
    zeta_size = slices["zeta_surgery"].stop - slices["zeta_surgery"].start
    theta_local = jnp.arange(theta_size)
    zeta_local = jnp.arange(theta_size, theta_size + zeta_size)

    summary = {
        "use_inner_schur": bool(factors.use_inner_schur),
        "surgery_size": int(surgery_indices.shape[0]),
        "theta_surgery_size": int(theta_size),
        "zeta_surgery_size": int(zeta_size),
        "raw_ass": eig_summary(ass),
        "exact_schur": eig_summary(exact_schur),
        "approx_schur": eig_summary(approx_schur),
        "approx_vs_exact_relative_error": _relative_fro_error_matrix(approx_schur, exact_schur),
        "ass_vs_exact_relative_error": _relative_fro_error_matrix(ass, exact_schur),
        "theta_zeta_split": {
            "raw_ass": _schur_split_block_summary(ass, theta_local, zeta_local),
            "exact_schur": _schur_split_block_summary(exact_schur, theta_local, zeta_local),
            "approx_schur": _schur_split_block_summary(approx_schur, theta_local, zeta_local),
            "theta_block_relative_error": _relative_fro_error_matrix(
                _block_submatrix(approx_schur, theta_local, theta_local),
                _block_submatrix(exact_schur, theta_local, theta_local),
            ),
            "zeta_block_relative_error": _relative_fro_error_matrix(
                _block_submatrix(approx_schur, zeta_local, zeta_local),
                _block_submatrix(exact_schur, zeta_local, zeta_local),
            ),
            "theta_zeta_block_relative_error": _relative_fro_error_matrix(
                _block_submatrix(approx_schur, theta_local, zeta_local),
                _block_submatrix(exact_schur, theta_local, zeta_local),
            ),
        },
    }
    if schur_from_inv is not None:
        summary["stored_inverse_vs_approx_schur_relative_error"] = _relative_fro_error_matrix(schur_from_inv, approx_schur)
        summary["stored_inverse_schur"] = eig_summary(schur_from_inv)
    return summary


def dense_k1_schur_correction_summary(
    *,
    operators_override=None,
    dirichlet_case: bool,
):
    operators_bundle = ops if operators_override is None else operators_override
    size = getattr(seq, f"n1_dbc" if dirichlet_case else f"n1")

    def K_apply(x):
        return apply_stiffness(seq, operators_bundle, x, 1, dirichlet=dirichlet_case)

    K_dense = _symmetrize(_assemble_dense_from_apply(K_apply, size))
    surgery, factors, bulk_inv = _k1_bulk_inverse_dense(operators_bundle, dirichlet_case=dirichlet_case)
    surgery_indices = surgery.surgery_indices
    bulk_indices = surgery.bulk_indices
    asb = _block_submatrix(K_dense, surgery_indices, bulk_indices)
    abs_ = _block_submatrix(K_dense, bulk_indices, surgery_indices)
    abb = _block_submatrix(K_dense, bulk_indices, bulk_indices)
    exact_bulk_pinv = _symmetric_pseudoinverse_local(abb, relative_tol=1e-8)

    exact_correction = _symmetrize(asb @ exact_bulk_pinv @ abs_)
    approx_correction = _symmetrize(asb @ bulk_inv @ abs_)

    slices = _surgery_slices_k1(seq, dirichlet_case)
    theta_size = slices["theta_surgery"].stop - slices["theta_surgery"].start
    zeta_size = slices["zeta_surgery"].stop - slices["zeta_surgery"].start
    theta_local = jnp.arange(theta_size)
    zeta_local = jnp.arange(theta_size, theta_size + zeta_size)

    return {
        "use_inner_schur": bool(factors.use_inner_schur),
        "exact_correction": _schur_correction_split_summary(
            exact_correction,
            exact_correction,
            theta_local,
            zeta_local,
        ),
        "approx_correction": _schur_correction_split_summary(
            approx_correction,
            exact_correction,
            theta_local,
            zeta_local,
        ),
        "correction_gap": _schur_correction_split_summary(
            approx_correction - exact_correction,
            exact_correction,
            theta_local,
            zeta_local,
        ),
    }


def _k1_bulk_response_component_norms(surgery, response: jnp.ndarray):
    r = response[:surgery.rt_r_size]
    theta = response[surgery.rt_r_size:surgery.bulk_rt_size]
    zeta = response[surgery.bulk_rt_size:surgery.bulk_rt_size + surgery.bulk_zeta_size]
    return {
        "total": float(jnp.linalg.norm(response)),
        "r": float(jnp.linalg.norm(r)),
        "theta_bulk": float(jnp.linalg.norm(theta)),
        "zeta_bulk": float(jnp.linalg.norm(zeta)),
    }


def _k1_bulk_response_component_relative_errors(surgery, approx_response: jnp.ndarray, exact_response: jnp.ndarray):
    approx_r = approx_response[:surgery.rt_r_size]
    exact_r = exact_response[:surgery.rt_r_size]
    approx_theta = approx_response[surgery.rt_r_size:surgery.bulk_rt_size]
    exact_theta = exact_response[surgery.rt_r_size:surgery.bulk_rt_size]
    approx_zeta = approx_response[surgery.bulk_rt_size:surgery.bulk_rt_size + surgery.bulk_zeta_size]
    exact_zeta = exact_response[surgery.bulk_rt_size:surgery.bulk_rt_size + surgery.bulk_zeta_size]
    return {
        "r": _relative_error(approx_r, exact_r),
        "theta_bulk": _relative_error(approx_theta, exact_theta),
        "zeta_bulk": _relative_error(approx_zeta, exact_zeta),
    }


def _summarize_response_columns(columns):
    if not columns:
        return {
            "count": 0,
        }

    relative_errors = jnp.asarray([entry["relative_error"] for entry in columns], dtype=jnp.float64)
    r_errors = jnp.asarray([entry["block_relative_errors"]["r"] for entry in columns], dtype=jnp.float64)
    theta_errors = jnp.asarray([entry["block_relative_errors"]["theta_bulk"] for entry in columns], dtype=jnp.float64)
    zeta_errors = jnp.asarray([entry["block_relative_errors"]["zeta_bulk"] for entry in columns], dtype=jnp.float64)
    return {
        "count": len(columns),
        "mean_relative_error": float(jnp.mean(relative_errors)),
        "max_relative_error": float(jnp.max(relative_errors)),
        "mean_block_relative_errors": {
            "r": float(jnp.mean(r_errors)),
            "theta_bulk": float(jnp.mean(theta_errors)),
            "zeta_bulk": float(jnp.mean(zeta_errors)),
        },
        "max_block_relative_errors": {
            "r": float(jnp.max(r_errors)),
            "theta_bulk": float(jnp.max(theta_errors)),
            "zeta_bulk": float(jnp.max(zeta_errors)),
        },
        "worst_column": max(columns, key=lambda entry: entry["relative_error"]),
        "columns": columns,
    }


def dense_k1_surgery_response_summary(
    *,
    operators_override=None,
    dirichlet_case: bool,
):
    operators_bundle = ops if operators_override is None else operators_override
    size = getattr(seq, f"n1_dbc" if dirichlet_case else f"n1")

    def K_apply(x):
        return apply_stiffness(seq, operators_bundle, x, 1, dirichlet=dirichlet_case)

    K_dense = _symmetrize(_assemble_dense_from_apply(K_apply, size))
    surgery, factors, bulk_inv = _k1_bulk_inverse_dense(operators_bundle, dirichlet_case=dirichlet_case)
    surgery_indices = surgery.surgery_indices
    bulk_indices = surgery.bulk_indices
    abs_ = _block_submatrix(K_dense, bulk_indices, surgery_indices)
    abb = _block_submatrix(K_dense, bulk_indices, bulk_indices)
    exact_bulk_pinv = _symmetric_pseudoinverse_local(abb, relative_tol=1e-8)

    slices = _surgery_slices_k1(seq, dirichlet_case)
    theta_size = slices["theta_surgery"].stop - slices["theta_surgery"].start
    zeta_size = slices["zeta_surgery"].stop - slices["zeta_surgery"].start
    theta_columns = []
    zeta_columns = []

    for source_local_index in range(abs_.shape[1]):
        bulk_rhs = abs_[:, source_local_index]
        exact_response = exact_bulk_pinv @ bulk_rhs
        approx_response = bulk_inv @ bulk_rhs
        entry = {
            "source_local_index": int(source_local_index),
            "source_component": "theta_surgery" if source_local_index < theta_size else "zeta_surgery",
            "source_component_local_index": int(source_local_index if source_local_index < theta_size else source_local_index - theta_size),
            "rhs_norm": float(jnp.linalg.norm(bulk_rhs)),
            "relative_error": float(_relative_error(approx_response, exact_response)),
            "exact_norms": _k1_bulk_response_component_norms(surgery, exact_response),
            "approx_norms": _k1_bulk_response_component_norms(surgery, approx_response),
            "block_relative_errors": _k1_bulk_response_component_relative_errors(surgery, approx_response, exact_response),
        }
        if source_local_index < theta_size:
            theta_columns.append(entry)
        else:
            zeta_columns.append(entry)

    return {
        "use_inner_schur": bool(factors.use_inner_schur),
        "theta_surgery_size": int(theta_size),
        "zeta_surgery_size": int(zeta_size),
        "theta_sources": _summarize_response_columns(theta_columns),
        "zeta_sources": _summarize_response_columns(zeta_columns),
    }


def dense_k1_block_preconditioned_stiffness_summary(
    *,
    operators_override=None,
    dirichlet_case: bool,
    tol: float = 1e-10,
):
    operators_bundle = ops if operators_override is None else operators_override
    size = getattr(seq, f"n1_dbc" if dirichlet_case else f"n1")

    def K_apply(x):
        return apply_stiffness(seq, operators_bundle, x, 1, dirichlet=dirichlet_case)

    def M_apply(x):
        return apply_mass_matrix(seq, operators_bundle, x, 1, dirichlet=dirichlet_case)

    def P_apply(x):
        return apply_stiffness_tensor_preconditioner(seq, operators_bundle, x, 1, dirichlet=dirichlet_case)

    K_dense = _symmetrize(_assemble_dense_from_apply(K_apply, size))
    M_dense = _symmetrize(_assemble_dense_from_apply(M_apply, size))
    P_dense = _assemble_dense_from_apply(P_apply, size)
    surgery = select_boundary_data(
        operators_bundle.k1_tensor_stiff_precond,
        dirichlet_case,
        "Tensor stiffness k=1",
    ).surgery
    block_indices = {
        "arr": surgery.r_indices,
        "theta": surgery.theta_bulk_indices,
        "zeta": surgery.zeta_bulk_indices,
        "rt": surgery.rt_indices,
        "bulk": surgery.bulk_indices,
        "surgery": surgery.surgery_indices,
    }
    coupling_pairs = {
        "arr_theta": (surgery.r_indices, surgery.theta_bulk_indices),
        "rt_zeta": (surgery.rt_indices, surgery.zeta_bulk_indices),
        "bulk_surgery": (surgery.bulk_indices, surgery.surgery_indices),
    }

    return {
        "dirichlet": bool(dirichlet_case),
        "block_sizes": {name: int(indices.shape[0]) for name, indices in block_indices.items()},
        "block_pk": {
            name: _dense_local_preconditioned_summary(
                _block_submatrix(K_dense, indices, indices),
                _block_submatrix(M_dense, indices, indices),
                _block_submatrix(P_dense, indices, indices),
                tol=tol,
            )
            for name, indices in block_indices.items()
        },
        "coupling_transpose_defects": {
            name: {
                "k": _transpose_defect_summary(K_dense, left, right),
                "p": _transpose_defect_summary(P_dense, left, right),
            }
            for name, (left, right) in coupling_pairs.items()
        },
    }


harmonic_free_preconditioned_stiffness_debug = {
    "k1_dbc": dense_preconditioned_stiffness_summary(1, dirichlet_case=True),
    "k2_free": dense_preconditioned_stiffness_summary(2, dirichlet_case=False),
}

harmonic_free_preconditioned_stiffness_debug
# %%

# %% Compare k=1 inner-Schur toggle on the no-harmonic DBC case

ops_k1_inner_schur = assemble_tensor_stiffness_preconditioner(
    seq,
    operators=ops,
    ks=(1,),
    rank=RANK,
    cp_kwargs={**CP_KWARGS, "k1_inner_schur": True},
)

k1_inner_schur_compare = {
    "default_solve": run_deflated_tensor_stiffness_solve(1, operators_override=ops, dirichlet_case=True, seed=SEED),
    "inner_schur_solve": run_deflated_tensor_stiffness_solve(1, operators_override=ops_k1_inner_schur, dirichlet_case=True, seed=SEED),
    "default_pk": dense_preconditioned_stiffness_summary(1, operators_override=ops, dirichlet_case=True),
    "inner_schur_pk": dense_preconditioned_stiffness_summary(1, operators_override=ops_k1_inner_schur, dirichlet_case=True),
}

k1_inner_schur_compare
# %%

# %% Localize k = 1 symmetry loss to blocks and couplings

k1_block_local_compare = {
    "default": dense_k1_block_preconditioned_stiffness_summary(
        operators_override=ops,
        dirichlet_case=True,
    ),
    "inner_schur": dense_k1_block_preconditioned_stiffness_summary(
        operators_override=ops_k1_inner_schur,
        dirichlet_case=True,
    ),
}

k1_block_local_compare
# %%

# %% Audit k = 1 bulk inverse before outer surgery completion

k1_bulk_inverse_compare = {
    "default": dense_k1_bulk_inverse_summary(
        operators_override=ops,
        dirichlet_case=True,
    ),
    "inner_schur": dense_k1_bulk_inverse_summary(
        operators_override=ops_k1_inner_schur,
        dirichlet_case=True,
    ),
}

k1_bulk_inverse_compare
# %%

# %% Audit k = 1 outer surgery Schur, split into theta- and zeta-surgery modes

k1_surgery_schur_compare = {
    "default": dense_k1_surgery_schur_summary(
        operators_override=ops,
        dirichlet_case=True,
    ),
    "inner_schur": dense_k1_surgery_schur_summary(
        operators_override=ops_k1_inner_schur,
        dirichlet_case=True,
    ),
}

k1_surgery_schur_compare
# %%

# %% Decompose k = 1 Schur correction A_sb P_bulk A_bs by theta/zeta surgery blocks

k1_schur_correction_compare = {
    "default": dense_k1_schur_correction_summary(
        operators_override=ops,
        dirichlet_case=True,
    ),
    "inner_schur": dense_k1_schur_correction_summary(
        operators_override=ops_k1_inner_schur,
        dirichlet_case=True,
    ),
}

k1_schur_correction_compare
# %%

# %% Compare exact vs approximate bulk responses for each k = 1 surgery source column

k1_surgery_response_compare = {
    "default": dense_k1_surgery_response_summary(
        operators_override=ops,
        dirichlet_case=True,
    ),
    "inner_schur": dense_k1_surgery_response_summary(
        operators_override=ops_k1_inner_schur,
        dirichlet_case=True,
    ),
}

k1_surgery_response_compare
# %%
