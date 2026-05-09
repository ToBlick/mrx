"""Inspect whether the k=1 stiffness bulk blocks and their FD surrogates are invertible."""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "True")

import jax
import jax.numpy as jnp

from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import rotating_ellipse_map
from mrx.operators import (
    _assemble_dense_from_apply,
    _build_k1_stiffness_surgery_factors,
    _restrict_radial_mass,
    _stiffness_axis_from_mass_term,
    _symmetrize,
    _theta_bulk_shape_k1,
    _zeta_bulk_shape_k1,
    _arr_shape_k1,
    assemble_incidence_operators,
    assemble_mass_operators,
    assemble_tensor_stiffness_models,
)
from mrx.preconditioners import _apply_extracted_submatrix, _build_kron_sum_fd_factors

jax.config.update("jax_enable_x64", True)

DIRICHLET = True
RANK = 2
TOL = 1e-10


def build_case():
    ns = (6, 12, 4)
    p = 3
    seq = DeRhamSequence(
        ns, (p, p, p), 2 * p,
        ("clamped", "periodic", "periodic"),
        polar=True,
        tol=1e-9,
        maxiter=2000,
        betti_numbers=(1, 1, 0, 0),
    )
    seq.evaluate_1d()
    seq.assemble_reference_mass_matrix()
    seq.set_map(rotating_ellipse_map(eps=0.3, kappa=1.2, R0=1.0, nfp=3))
    ops = assemble_mass_operators(seq, seq.geometry, ks=(1, 2))
    ops = assemble_incidence_operators(seq, operators=ops, ks=(0, 1))
    ops = assemble_tensor_stiffness_models(
        seq,
        operators=ops,
        ks=(1,),
        rank=RANK,
        cp_kwargs={"k1_rank": RANK},
    )
    return seq, ops


def dense_submatrix(apply_data, indices: jnp.ndarray) -> jnp.ndarray:
    size = int(indices.shape[0])
    return _symmetrize(_assemble_dense_from_apply(
        lambda x, data=apply_data, idx=indices: _apply_extracted_submatrix(data, idx, idx, x),
        size,
    ))


def kron3(a: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
    return jnp.kron(c, jnp.kron(b, a))


def summarize_matrix(label: str, matrix: jnp.ndarray, tol: float) -> None:
    evals = jnp.linalg.eigvalsh(_symmetrize(matrix))
    abs_evals = jnp.abs(evals)
    nullity = int(jnp.sum(abs_evals <= tol))
    rank = int(matrix.shape[0] - nullity)
    print(label)
    print(f"  shape={matrix.shape}")
    print(f"  min_eig={float(jnp.min(evals)):.6e}")
    print(f"  max_eig={float(jnp.max(evals)):.6e}")
    print(f"  min_abs_eig={float(jnp.min(abs_evals)):.6e}")
    print(f"  nullity@tol={tol:.1e}: {nullity}")
    print(f"  invertible@tol={tol:.1e}: {rank == matrix.shape[0]}")


def summarize_axes(label: str, terms: tuple[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], ...]) -> None:
    print(label)
    for idx, (mass_r, mass_t, mass_z) in enumerate(terms[:2]):
        min_r = float(jnp.min(jnp.linalg.eigvalsh(_symmetrize(mass_r))))
        min_t = float(jnp.min(jnp.linalg.eigvalsh(_symmetrize(mass_t))))
        min_z = float(jnp.min(jnp.linalg.eigvalsh(_symmetrize(mass_z))))
        print(
            f"  term{idx}: min_eigs r={min_r:.6e}  t={min_t:.6e}  z={min_z:.6e}"
        )


def check_fd(label: str, terms: tuple[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], ...]) -> None:
    if len(terms) < 2:
        print(f"{label}\n  FD check skipped: fewer than two terms")
        return
    try:
        _build_kron_sum_fd_factors(*terms[0], *terms[1])
    except Exception as exc:
        print(f"{label}\n  FD check: FAIL ({exc})")
        return
    print(f"{label}\n  FD check: OK")


def build_arr_terms(model, arr_shape):
    terms = []
    for mass_r, mass_t, mass_z in zip(model.tt_mass_r_terms, model.tt_mass_t_terms, model.tt_mass_z_terms):
        terms.append((
            _restrict_radial_mass(mass_r, 1, arr_shape[0]),
            mass_t,
            _stiffness_axis_from_mass_term(mass_z, model.g_z),
        ))
    for mass_r, mass_t, mass_z in zip(model.zz_mass_r_terms, model.zz_mass_t_terms, model.zz_mass_z_terms):
        terms.append((
            _restrict_radial_mass(mass_r, 1, arr_shape[0]),
            _stiffness_axis_from_mass_term(mass_t, model.g_t),
            mass_z,
        ))
    return tuple(terms)


def build_theta_terms(model, theta_shape):
    terms = []
    for mass_r, mass_t, mass_z in zip(model.rr_mass_r_terms, model.rr_mass_t_terms, model.rr_mass_z_terms):
        terms.append((
            _restrict_radial_mass(mass_r, 2, theta_shape[0]),
            mass_t,
            _stiffness_axis_from_mass_term(mass_z, model.g_z),
        ))
    for mass_r, mass_t, mass_z in zip(model.zz_mass_r_terms, model.zz_mass_t_terms, model.zz_mass_z_terms):
        terms.append((
            _restrict_radial_mass(_stiffness_axis_from_mass_term(mass_r, model.g_r), 2, theta_shape[0]),
            mass_t,
            mass_z,
        ))
    return tuple(terms)


def build_zeta_terms(model, zeta_shape):
    terms = []
    for mass_r, mass_t, mass_z in zip(model.rr_mass_r_terms, model.rr_mass_t_terms, model.rr_mass_z_terms):
        terms.append((
            _restrict_radial_mass(mass_r, 2, zeta_shape[0]),
            _stiffness_axis_from_mass_term(mass_t, model.g_t),
            mass_z,
        ))
    for mass_r, mass_t, mass_z in zip(model.tt_mass_r_terms, model.tt_mass_t_terms, model.tt_mass_z_terms):
        terms.append((
            _restrict_radial_mass(_stiffness_axis_from_mass_term(mass_r, model.g_r), 2, zeta_shape[0]),
            mass_t,
            mass_z,
        ))
    return tuple(terms)


def surrogate_matrix(terms: tuple[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], ...]) -> jnp.ndarray:
    matrix = jnp.zeros((terms[0][0].shape[0] * terms[0][1].shape[0] * terms[0][2].shape[0],) * 2, dtype=jnp.float64)
    for mass_r, mass_t, mass_z in terms:
        matrix = matrix + kron3(mass_r, mass_t, mass_z)
    return _symmetrize(matrix)


def main() -> None:
    seq, ops = build_case()
    surgery = _build_k1_stiffness_surgery_factors(seq, ops, dirichlet=DIRICHLET)
    model = ops.k1_tensor_stiff_model
    if model is None:
        raise ValueError("Tensor stiffness model k=1 is not assembled")

    arr_shape = _arr_shape_k1(seq, DIRICHLET)
    theta_shape = _theta_bulk_shape_k1(seq, DIRICHLET)
    zeta_shape = _zeta_bulk_shape_k1(seq, DIRICHLET)

    arr_true = dense_submatrix(surgery.apply_data, surgery.r_indices)
    theta_true = dense_submatrix(surgery.apply_data, surgery.theta_bulk_indices)
    zeta_true = dense_submatrix(surgery.apply_data, surgery.zeta_bulk_indices)

    arr_terms = build_arr_terms(model, arr_shape)
    theta_terms = build_theta_terms(model, theta_shape)
    zeta_terms = build_zeta_terms(model, zeta_shape)

    arr_surrogate = surrogate_matrix(arr_terms)
    theta_surrogate = surrogate_matrix(theta_terms)
    zeta_surrogate = surrogate_matrix(zeta_terms)

    print(f"k=1 stiffness FD invertibility diagnostic  dirichlet={DIRICHLET}  rank={RANK}")
    print(f"n1={seq.n1}  n1_dbc={seq.n1_dbc}")
    print()

    summarize_matrix("true arr block", arr_true, TOL)
    summarize_axes("arr surrogate axis minima", arr_terms)
    summarize_matrix("rank-2 arr surrogate", arr_surrogate, TOL)
    check_fd("arr surrogate FD prerequisites", arr_terms)
    print()

    summarize_matrix("true theta_bulk block", theta_true, TOL)
    summarize_axes("theta surrogate axis minima", theta_terms)
    summarize_matrix("rank-2 theta surrogate", theta_surrogate, TOL)
    check_fd("theta surrogate FD prerequisites", theta_terms)
    print()

    summarize_matrix("true zeta_bulk block", zeta_true, TOL)
    summarize_axes("zeta surrogate axis minima", zeta_terms)
    summarize_matrix("rank-2 zeta surrogate", zeta_surrogate, TOL)
    check_fd("zeta surrogate FD prerequisites", zeta_terms)


if __name__ == "__main__":
    main()