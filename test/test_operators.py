"""Mass-operator and mass-preconditioner tests for ``mrx.operators``.

These tests keep the geometry genuinely 3D by using a small rotating ellipse,
assemble the full mass-operator bundle once, then reuse it for dense/sparse
consistency checks and solver-facing mass-inverse tests.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from test.conftest import build_dense

from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import rotating_ellipse_map
from mrx.operators import (
    _build_mass_preconditioner_apply,
    apply_mass_matrix,
    apply_mass_tensor_preconditioner_ops,
    apply_inverse_mass_matrix,
    assemble_mass_operators,
    assemble_tensor_mass_preconditioner,
    dense_mass_matrix,
)
from mrx.preconditioners import MassPreconditionerSpec

jax.config.update("jax_enable_x64", True)

ALL_K = (0, 1, 2, 3)
ALL_DBC = (False, True)
N_PROBES = 4
N_SOLVER_PROBES = 4
WARM_START_TOL = 1e-10
WARM_START_LOOSE_TOL = 1e-8
MASS_SOLVE_COMPARE_ATOL = 1e-6
MASS_SOLVE_COMPARE_RTOL = 1e-7
MASS_PRECONDITIONERS = {
    "jacobi": MassPreconditionerSpec(kind="jacobi"),
    "richardson-4": MassPreconditionerSpec(
        kind="richardson",
        steps=4,
        power_iterations=8,
        damping_safety=0.8,
        smoother=MassPreconditionerSpec(kind="tensor"),
    ),
    "chebyshev-4": MassPreconditionerSpec(
        kind="chebyshev",
        steps=4,
        power_iterations=8,
        min_eig_fraction=1e-3,
        smoother=MassPreconditionerSpec(kind="tensor"),
    ),
    "tensor": MassPreconditionerSpec(kind="tensor", surgery_schur=True),
}
SPD_PRECONDITIONERS = ("jacobi", "richardson-4", "tensor")
K2_SCHUR_PRECONDITIONERS = {
    "none/schur/tensor": MassPreconditionerSpec(
        kind="none",
        surgery_schur=True,
        smoother=MassPreconditionerSpec(kind="tensor"),
    ),
    "richardson/schur/tensor": MassPreconditionerSpec(
        kind="richardson",
        surgery_schur=True,
        steps=4,
        power_iterations=8,
        damping_safety=0.8,
        smoother=MassPreconditionerSpec(kind="tensor"),
    ),
}
K2_INVALID_SCHUR_PRECONDITIONERS = {
    "none/schur/jacobi": MassPreconditionerSpec(
        kind="none",
        surgery_schur=True,
        smoother=MassPreconditionerSpec(kind="jacobi"),
    ),
    "none/schur/richardson": MassPreconditionerSpec(
        kind="none",
        surgery_schur=True,
        smoother=MassPreconditionerSpec(
            kind="richardson",
            steps=4,
            power_iterations=8,
            damping_safety=0.8,
            smoother=MassPreconditionerSpec(kind="tensor"),
        ),
    ),
    "none/schur/chebyshev": MassPreconditionerSpec(
        kind="none",
        surgery_schur=True,
        smoother=MassPreconditionerSpec(
            kind="chebyshev",
            steps=4,
            power_iterations=8,
            min_eig_fraction=1e-3,
            smoother=MassPreconditionerSpec(kind="tensor"),
        ),
    ),
    "richardson/schur/jacobi": MassPreconditionerSpec(
        kind="richardson",
        surgery_schur=True,
        steps=4,
        power_iterations=8,
        damping_safety=0.8,
        smoother=MassPreconditionerSpec(kind="jacobi"),
    ),
    "richardson/schur/richardson": MassPreconditionerSpec(
        kind="richardson",
        surgery_schur=True,
        steps=4,
        power_iterations=8,
        damping_safety=0.8,
        smoother=MassPreconditionerSpec(
            kind="richardson",
            steps=4,
            power_iterations=8,
            damping_safety=0.8,
            smoother=MassPreconditionerSpec(kind="tensor"),
        ),
    ),
    "richardson/schur/chebyshev": MassPreconditionerSpec(
        kind="richardson",
        surgery_schur=True,
        steps=4,
        power_iterations=8,
        damping_safety=0.8,
        smoother=MassPreconditionerSpec(
            kind="chebyshev",
            steps=4,
            power_iterations=8,
            min_eig_fraction=1e-3,
            smoother=MassPreconditionerSpec(kind="tensor"),
        ),
    ),
}


def _random_vectors(n: int, seed: int, count: int = N_PROBES) -> np.ndarray:
    return np.asarray(
        jax.random.normal(jax.random.PRNGKey(seed), (count, n), dtype=jnp.float64)
    )


def _solve_dense_spd(cholesky_factor: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    y = np.linalg.solve(cholesky_factor, rhs)
    return np.linalg.solve(cholesky_factor.T, y)


@pytest.fixture(scope="module")
def rotating_mass_case():
    seq = DeRhamSequence(
        (5, 5, 5),
        (3, 3, 3),
        6,
        ("clamped", "periodic", "periodic"),
        polar=True,
        tol=1e-10,
        maxiter=1000,
        betti_numbers=(1, 1, 0, 0),
    )
    seq.evaluate_1d()
    seq.assemble_reference_mass_matrix()
    seq.set_map(rotating_ellipse_map(eps=0.33, kappa=1.2, R0=1.0, nfp=3))

    operators = assemble_mass_operators(seq, seq.geometry, ks=ALL_K)
    operators = assemble_tensor_mass_preconditioner(
        seq,
        operators=operators,
        ks=ALL_K,
        rank=3,
        cp_kwargs={"tol": 1e-8, "maxiter": 200},
    )

    dense_mass_cache = {}
    dense_mass_cholesky_cache = {}
    dense_preconditioner_cache = {}

    def get_dense_mass(k: int, dirichlet: bool) -> np.ndarray:
        key = (k, dirichlet)
        if key not in dense_mass_cache:
            dense_mass_cache[key] = np.asarray(
                dense_mass_matrix(seq, operators, k, dirichlet=dirichlet)
            )
        return dense_mass_cache[key]

    def get_dense_preconditioner(k: int, dirichlet: bool, label: str) -> np.ndarray:
        key = (k, dirichlet, label)
        if key not in dense_preconditioner_cache:
            precond_apply = _build_mass_preconditioner_apply(
                seq,
                operators,
                k=k,
                dirichlet=dirichlet,
                preconditioner=MASS_PRECONDITIONERS[label],
                allow_none=False,
            )
            n = get_dense_mass(k, dirichlet).shape[0]
            dense_preconditioner_cache[key] = np.asarray(build_dense(precond_apply, n))
        return dense_preconditioner_cache[key]

    def get_dense_mass_cholesky(k: int, dirichlet: bool) -> np.ndarray:
        key = (k, dirichlet)
        if key not in dense_mass_cholesky_cache:
            dense_mass_cholesky_cache[key] = np.linalg.cholesky(get_dense_mass(k, dirichlet))
        return dense_mass_cholesky_cache[key]

    return {
        "seq": seq,
        "operators": operators,
        "dense_mass": get_dense_mass,
        "dense_mass_cholesky": get_dense_mass_cholesky,
        "dense_preconditioner": get_dense_preconditioner,
    }


@pytest.mark.parametrize("k", ALL_K)
@pytest.mark.parametrize("dirichlet", ALL_DBC)
def test_dense_mass_matches_sparse_probe(rotating_mass_case, k, dirichlet):
    seq = rotating_mass_case["seq"]
    operators = rotating_mass_case["operators"]
    dense_mass = rotating_mass_case["dense_mass"](k, dirichlet)

    for vector in _random_vectors(dense_mass.shape[0], seed=10 + 13 * k + int(dirichlet)):
        sparse_apply = np.asarray(
            apply_mass_matrix(
                seq,
                operators,
                jnp.asarray(vector),
                k,
                dirichlet=dirichlet,
            )
        )
        dense_apply = dense_mass @ vector
        npt.assert_allclose(
            sparse_apply,
            dense_apply,
            atol=1e-10,
            rtol=1e-10,
            err_msg=f"dense/sparse mass mismatch for k={k} dirichlet={dirichlet}",
        )


@pytest.mark.parametrize("k", ALL_K)
@pytest.mark.parametrize("dirichlet", ALL_DBC)
def test_mass_symmetry_and_positivity_by_probing(rotating_mass_case, k, dirichlet):
    seq = rotating_mass_case["seq"]
    operators = rotating_mass_case["operators"]
    vectors = _random_vectors(
        rotating_mass_case["dense_mass"](k, dirichlet).shape[0],
        seed=100 + 17 * k + int(dirichlet),
    )

    for left, right in zip(vectors[:-1], vectors[1:]):
        mass_left = np.asarray(
            apply_mass_matrix(seq, operators, jnp.asarray(left), k, dirichlet=dirichlet)
        )
        mass_right = np.asarray(
            apply_mass_matrix(seq, operators, jnp.asarray(right), k, dirichlet=dirichlet)
        )
        lhs = float(left @ mass_right)
        rhs = float(right @ mass_left)
        scale = max(np.linalg.norm(left) * np.linalg.norm(mass_right), 1.0)
        assert abs(lhs - rhs) < 1e-10 * scale, (
            f"mass symmetry probe failed for k={k} dirichlet={dirichlet}: "
            f"lhs={lhs}, rhs={rhs}"
        )

    for vector in vectors:
        mass_vector = np.asarray(
            apply_mass_matrix(seq, operators, jnp.asarray(vector), k, dirichlet=dirichlet)
        )
        quadratic_form = float(vector @ mass_vector)
        assert quadratic_form > 1e-10, (
            f"mass positivity probe failed for k={k} dirichlet={dirichlet}: "
            f"x^T M x = {quadratic_form}"
        )


@pytest.mark.parametrize("k", ALL_K)
@pytest.mark.parametrize("dirichlet", ALL_DBC)
def test_dense_mass_is_spd(rotating_mass_case, k, dirichlet):
    dense_mass = rotating_mass_case["dense_mass"](k, dirichlet)
    npt.assert_allclose(dense_mass, dense_mass.T, atol=1e-10)
    eigvals = np.linalg.eigvalsh(dense_mass)
    assert eigvals.min() > 1e-10, (
        f"dense mass matrix is not SPD for k={k} dirichlet={dirichlet}: "
        f"lambda_min={eigvals.min()}"
    )


@pytest.mark.parametrize("label", tuple(MASS_PRECONDITIONERS))
@pytest.mark.parametrize("k", ALL_K)
@pytest.mark.parametrize("dirichlet", ALL_DBC)
def test_mass_preconditioner_is_symmetric(rotating_mass_case, label, k, dirichlet):
    dense_preconditioner = rotating_mass_case["dense_preconditioner"](k, dirichlet, label)
    npt.assert_allclose(
        dense_preconditioner,
        dense_preconditioner.T,
        atol=1e-10,
        err_msg=f"mass preconditioner {label} is not symmetric for k={k} dirichlet={dirichlet}",
    )


@pytest.mark.parametrize("label", SPD_PRECONDITIONERS)
@pytest.mark.parametrize("k", ALL_K)
@pytest.mark.parametrize("dirichlet", ALL_DBC)
def test_mass_preconditioner_is_spd(rotating_mass_case, label, k, dirichlet):
    dense_preconditioner = rotating_mass_case["dense_preconditioner"](k, dirichlet, label)
    eigvals = np.linalg.eigvalsh(dense_preconditioner)
    assert eigvals.min() > 1e-12, (
        f"mass preconditioner {label} is not SPD for k={k} dirichlet={dirichlet}: "
        f"lambda_min={eigvals.min()}"
    )


@pytest.mark.parametrize("dirichlet", ALL_DBC)
def test_direct_k0_tensor_apply_matches_routed_tensor_preconditioner(rotating_mass_case, dirichlet):
    seq = rotating_mass_case["seq"]
    operators = rotating_mass_case["operators"]
    dense_mass = rotating_mass_case["dense_mass"](0, dirichlet)
    direct_tensor = np.asarray(
        build_dense(
            lambda x: apply_mass_tensor_preconditioner_ops(
                seq,
                operators,
                x,
                0,
                dirichlet=dirichlet,
            ),
            dense_mass.shape[0],
        )
    )
    routed_tensor = rotating_mass_case["dense_preconditioner"](0, dirichlet, "tensor")
    npt.assert_allclose(
        direct_tensor,
        routed_tensor,
        atol=1e-10,
        rtol=1e-10,
        err_msg=f"direct k=0 tensor apply does not match routed tensor preconditioner for dirichlet={dirichlet}",
    )


@pytest.mark.parametrize("dirichlet", ALL_DBC)
def test_direct_k1_tensor_apply_matches_routed_tensor_preconditioner(rotating_mass_case, dirichlet):
    seq = rotating_mass_case["seq"]
    operators = rotating_mass_case["operators"]
    dense_mass = rotating_mass_case["dense_mass"](1, dirichlet)
    direct_tensor = np.asarray(
        build_dense(
            lambda x: apply_mass_tensor_preconditioner_ops(
                seq,
                operators,
                x,
                1,
                dirichlet=dirichlet,
            ),
            dense_mass.shape[0],
        )
    )
    routed_tensor = rotating_mass_case["dense_preconditioner"](1, dirichlet, "tensor")
    npt.assert_allclose(
        direct_tensor,
        routed_tensor,
        atol=1e-10,
        rtol=1e-10,
        err_msg=f"direct k=1 tensor apply does not match routed tensor preconditioner for dirichlet={dirichlet}",
    )


@pytest.mark.parametrize("label", tuple(MASS_PRECONDITIONERS))
@pytest.mark.parametrize("k", ALL_K)
@pytest.mark.parametrize("dirichlet", ALL_DBC)
def test_inverse_mass_matches_dense_solve(rotating_mass_case, label, k, dirichlet):
    seq = rotating_mass_case["seq"]
    operators = rotating_mass_case["operators"]
    dense_mass = rotating_mass_case["dense_mass"](k, dirichlet)
    dense_mass_cholesky = rotating_mass_case["dense_mass_cholesky"](k, dirichlet)

    for rhs in _random_vectors(
        dense_mass.shape[0],
        seed=200 + 19 * k + 100 * int(dirichlet),
        count=N_SOLVER_PROBES,
    ):
        x, info = apply_inverse_mass_matrix(
            seq,
            operators,
            jnp.asarray(rhs),
            k,
            dirichlet=dirichlet,
            preconditioner=MASS_PRECONDITIONERS[label],
            tol=1e-10,
            maxiter=2000,
            return_info=True,
        )
        x = np.asarray(x)
        x_dense = _solve_dense_spd(dense_mass_cholesky, rhs)
        npt.assert_allclose(
            x,
            x_dense,
            atol=MASS_SOLVE_COMPARE_ATOL,
            rtol=MASS_SOLVE_COMPARE_RTOL,
            err_msg=f"inverse mass solve mismatch for {label} k={k} dirichlet={dirichlet}",
        )
        npt.assert_allclose(
            dense_mass @ x,
            rhs,
            atol=MASS_SOLVE_COMPARE_ATOL,
            rtol=MASS_SOLVE_COMPARE_RTOL,
            err_msg=f"mass round-trip failed for {label} k={k} dirichlet={dirichlet}",
        )
        assert int(info) <= 0, (
            f"inverse mass solve did not converge for {label} k={k} "
            f"dirichlet={dirichlet} (info={int(info)})"
        )


@pytest.mark.parametrize("label", tuple(MASS_PRECONDITIONERS))
@pytest.mark.parametrize("k", ALL_K)
@pytest.mark.parametrize("dirichlet", ALL_DBC)
def test_mass_preconditioner_reduces_cg_iterations(rotating_mass_case, label, k, dirichlet):
    seq = rotating_mass_case["seq"]
    operators = rotating_mass_case["operators"]
    rhs_batch = _random_vectors(
        rotating_mass_case["dense_mass"](k, dirichlet).shape[0],
        seed=300 + 23 * k + 100 * int(dirichlet),
        count=N_SOLVER_PROBES,
    )

    none_iters = []
    precond_iters = []
    for rhs in rhs_batch:
        _, none_info = apply_inverse_mass_matrix(
            seq,
            operators,
            jnp.asarray(rhs),
            k,
            dirichlet=dirichlet,
            preconditioner="none",
            tol=1e-10,
            maxiter=2000,
            return_info=True,
        )
        _, precond_info = apply_inverse_mass_matrix(
            seq,
            operators,
            jnp.asarray(rhs),
            k,
            dirichlet=dirichlet,
            preconditioner=MASS_PRECONDITIONERS[label],
            tol=1e-10,
            maxiter=2000,
            return_info=True,
        )
        none_iters.append(abs(int(none_info)))
        precond_iters.append(abs(int(precond_info)))

    assert np.mean(precond_iters) < np.mean(none_iters), (
        f"mass preconditioner {label} did not reduce CG iterations for "
        f"k={k} dirichlet={dirichlet}: none={none_iters}, precond={precond_iters}"
    )


@pytest.mark.parametrize("label", tuple(MASS_PRECONDITIONERS))
@pytest.mark.parametrize("k", ALL_K)
@pytest.mark.parametrize("dirichlet", ALL_DBC)
def test_mass_inverse_warm_start_helps_with_tighter_tolerance(
    rotating_mass_case, label, k, dirichlet
):
    seq = rotating_mass_case["seq"]
    operators = rotating_mass_case["operators"]
    dense_mass = rotating_mass_case["dense_mass"](k, dirichlet)
    dense_mass_cholesky = rotating_mass_case["dense_mass_cholesky"](k, dirichlet)
    rhs = _random_vectors(
        dense_mass.shape[0],
        seed=400 + 29 * k + 100 * int(dirichlet),
        count=1,
    )[0]

    x_loose, loose_info = apply_inverse_mass_matrix(
        seq,
        operators,
        jnp.asarray(rhs),
        k,
        dirichlet=dirichlet,
        preconditioner=MASS_PRECONDITIONERS[label],
        tol=WARM_START_LOOSE_TOL,
        maxiter=2000,
        return_info=True,
    )
    x_cold, cold_info = apply_inverse_mass_matrix(
        seq,
        operators,
        jnp.asarray(rhs),
        k,
        dirichlet=dirichlet,
        preconditioner=MASS_PRECONDITIONERS[label],
        tol=WARM_START_TOL,
        maxiter=2000,
        return_info=True,
    )
    x_warm, warm_info = apply_inverse_mass_matrix(
        seq,
        operators,
        jnp.asarray(rhs),
        k,
        dirichlet=dirichlet,
        guess=x_loose,
        preconditioner=MASS_PRECONDITIONERS[label],
        tol=WARM_START_TOL,
        maxiter=2000,
        return_info=True,
    )

    x_dense = _solve_dense_spd(dense_mass_cholesky, rhs)
    npt.assert_allclose(
        np.asarray(x_cold),
        x_dense,
        atol=MASS_SOLVE_COMPARE_ATOL,
        rtol=MASS_SOLVE_COMPARE_RTOL,
        err_msg=f"cold tight solve mismatch for {label} k={k} dirichlet={dirichlet}",
    )
    npt.assert_allclose(
        np.asarray(x_warm),
        x_dense,
        atol=MASS_SOLVE_COMPARE_ATOL,
        rtol=MASS_SOLVE_COMPARE_RTOL,
        err_msg=f"warm tight solve mismatch for {label} k={k} dirichlet={dirichlet}",
    )
    assert int(loose_info) <= 0, (
        f"loose warm-start seed solve did not converge for {label} k={k} "
        f"dirichlet={dirichlet} (info={int(loose_info)})"
    )
    assert int(cold_info) <= 0, (
        f"cold tight solve did not converge for {label} k={k} "
        f"dirichlet={dirichlet} (info={int(cold_info)})"
    )
    assert int(warm_info) <= 0, (
        f"warm tight solve did not converge for {label} k={k} "
        f"dirichlet={dirichlet} (info={int(warm_info)})"
    )
    assert abs(int(warm_info)) <= abs(int(cold_info)), (
        f"warm start did not help for {label} k={k} dirichlet={dirichlet}: "
        f"cold={abs(int(cold_info))}, warm={abs(int(warm_info))}"
    )


@pytest.mark.parametrize("label", tuple(K2_SCHUR_PRECONDITIONERS))
@pytest.mark.parametrize("dirichlet", ALL_DBC)
def test_k2_schur_inverse_mass_matches_dense_solve(rotating_mass_case, label, dirichlet):
    seq = rotating_mass_case["seq"]
    operators = rotating_mass_case["operators"]
    dense_mass = rotating_mass_case["dense_mass"](2, dirichlet)
    dense_mass_cholesky = rotating_mass_case["dense_mass_cholesky"](2, dirichlet)
    rhs = _random_vectors(
        dense_mass.shape[0],
        seed=500 + 100 * int(dirichlet) + tuple(K2_SCHUR_PRECONDITIONERS).index(label),
        count=1,
    )[0]

    x, info = apply_inverse_mass_matrix(
        seq,
        operators,
        jnp.asarray(rhs),
        2,
        dirichlet=dirichlet,
        preconditioner=K2_SCHUR_PRECONDITIONERS[label],
        tol=1e-10,
        maxiter=2000,
        return_info=True,
    )
    x = np.asarray(x)
    x_dense = _solve_dense_spd(dense_mass_cholesky, rhs)

    npt.assert_allclose(
        x,
        x_dense,
        atol=MASS_SOLVE_COMPARE_ATOL,
        rtol=MASS_SOLVE_COMPARE_RTOL,
        err_msg=f"k=2 Schur inverse mass solve mismatch for {label} dirichlet={dirichlet}",
    )
    npt.assert_allclose(
        dense_mass @ x,
        rhs,
        atol=MASS_SOLVE_COMPARE_ATOL,
        rtol=MASS_SOLVE_COMPARE_RTOL,
        err_msg=f"k=2 Schur mass round-trip failed for {label} dirichlet={dirichlet}",
    )
    assert int(info) <= 0, (
        f"k=2 Schur inverse mass solve did not converge for {label} "
        f"dirichlet={dirichlet} (info={int(info)})"
    )


@pytest.mark.parametrize("label", tuple(K2_INVALID_SCHUR_PRECONDITIONERS))
@pytest.mark.parametrize("dirichlet", ALL_DBC)
def test_k2_invalid_non_tensor_inner_schur_rejected(rotating_mass_case, label, dirichlet):
    seq = rotating_mass_case["seq"]
    operators = rotating_mass_case["operators"]
    rhs = jnp.asarray(_random_vectors(
        rotating_mass_case["dense_mass"](2, dirichlet).shape[0],
        seed=700 + 100 * int(dirichlet) + tuple(K2_INVALID_SCHUR_PRECONDITIONERS).index(label),
        count=1,
    )[0])

    with pytest.raises(ValueError, match="only supports kind='tensor' as the inner smoother"):
        apply_inverse_mass_matrix(
            seq,
            operators,
            rhs,
            2,
            dirichlet=dirichlet,
            preconditioner=K2_INVALID_SCHUR_PRECONDITIONERS[label],
            tol=1e-10,
            maxiter=2000,
        )