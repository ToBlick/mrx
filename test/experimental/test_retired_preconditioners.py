"""Tests for the RETIRED preconditioner paths (mrx/experimental/).

Moved out of test/test_sequence.py and test/test_preconditioners.py on
2026-08-18, when production became raw_kron masses + Jacobi Laplacians. The
code these cover is kept -- the surgery/Schur mass path and the k=0
tensor-Hodge Laplacian preconditioner are both better than the defaults on
axisymmetric geometry -- so the tests are kept too, just out of the default run.

This directory is in norecursedirs; run it explicitly:

    pytest test/experimental
"""
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from test.conftest import build_dense, n_dofs

ALL_DBC = (False, True)
_dof = n_dofs
from mrx.operators import (apply_inverse_mass_matrix, apply_mass_matrix,
                           apply_mass_tensor_preconditioner_ops)
from mrx.preconditioners import (MassPreconditionerSpec,
                                 SaddlePointPreconditionerSpec,
                                 SchurPreconditionerSpec)

jax.config.update("jax_enable_x64", True)

_ALL_K = (0, 1, 2, 3)
_ALL_DBC = (False, True)
_N_PROBES = 4
_TENSOR_SPEC = MassPreconditionerSpec(kind="tensor", surgery_schur=True)


@pytest.mark.parametrize("dbc", _ALL_DBC)
@pytest.mark.parametrize("k", _ALL_K)
def test_tensor_preconditioner_is_symmetric(torus_seq_tensor, precond_jit_tensor, k, dbc):
    """|u^T P v - v^T P u| small for the retired tensor mass preconditioner."""
    P = precond_jit_tensor[("tensor", k, dbc)]
    n = n_dofs(torus_seq_tensor, k, dbc)
    rng = np.random.default_rng(k + 10 * dbc)
    for _ in range(_N_PROBES):
        u = jnp.asarray(rng.standard_normal(n))
        v = jnp.asarray(rng.standard_normal(n))
        a, b = float(u @ P(v)), float(v @ P(u))
        npt.assert_allclose(a, b, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("dbc", _ALL_DBC)
@pytest.mark.parametrize("k", _ALL_K)
def test_tensor_preconditioner_is_spd(torus_seq_tensor, precond_jit_tensor, k, dbc):
    P = precond_jit_tensor[("tensor", k, dbc)]
    n = n_dofs(torus_seq_tensor, k, dbc)
    rng = np.random.default_rng(100 + k + 10 * dbc)
    for _ in range(_N_PROBES):
        v = jnp.asarray(rng.standard_normal(n))
        assert float(v @ P(v)) > 0.0


@pytest.mark.parametrize("dbc", _ALL_DBC)
@pytest.mark.parametrize("k", _ALL_K)
def test_tensor_inverse_mass_roundtrip(torus_seq_tensor, k, dbc):
    """M (M^-1 b) == b through the retired tensor path."""
    seq = torus_seq_tensor
    n = n_dofs(seq, k, dbc)
    rng = np.random.default_rng(200 + k + 10 * dbc)
    b = jnp.asarray(rng.standard_normal(n))
    x = apply_inverse_mass_matrix(seq, seq.operators, b, k, dirichlet=dbc,
                                  preconditioner=_TENSOR_SPEC, tol=1e-12)
    r = apply_mass_matrix(seq, seq.operators, x, k, dirichlet=dbc) - b
    assert float(jnp.linalg.norm(r)) / float(jnp.linalg.norm(b)) < 1e-8


# ---------------------------------------------------------------------------
# Fast-diagonalisation Hodge preconditioner (k = 0)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dirichlet", ALL_DBC)
def test_fd_hodge_preconditioner_spd(torus_seq_tensor, dirichlet):
    """``apply_hodge_laplacian_preconditioner(kind='tensor')`` is SPD for k=0."""
    seq = torus_seq_tensor
    n = _dof(seq, 0, dirichlet)
    P = np.asarray(build_dense(
        lambda v: seq.apply_hodge_laplacian_preconditioner(
            v, k=0, dirichlet=dirichlet, kind='tensor'),
        n))
    npt.assert_allclose(P, P.T, atol=1e-9,
                        err_msg="FD Hodge precond not symmetric")
    eigs = np.linalg.eigvalsh(P)
    assert eigs.min() > -1e-9, (
        f"FD Hodge precond k=0 dirichlet={dirichlet} has large negative "
        f"eigenvalue {eigs.min()}"
    )


@pytest.mark.parametrize("dirichlet", ALL_DBC)
def test_fd_hodge_preconditioner_accelerates_cg(torus_seq_tensor, dirichlet):
    """Tensor Hodge preconditioner runs and converges at k=0."""
    seq = torus_seq_tensor
    from mrx.nullspace import get_nullspace
    from mrx.solvers import solve_singular_cg
    n = _dof(seq, 0, dirichlet)
    key = jax.random.PRNGKey(7)
    b = jax.random.normal(key, (n,))

    vs = get_nullspace(seq.get_operators(), 0, dirichlet)

    def matvec(x):
        return seq.apply_hodge_laplacian(x, k=0, dirichlet=dirichlet)

    def mass_matvec(x):
        return seq.apply_mass_matrix(x, k=0, dirichlet=dirichlet)

    _, info = solve_singular_cg(
        matvec, b, mass_matvec=mass_matvec,
        precond_matvec=lambda x: seq.apply_hodge_laplacian_preconditioner(
            x, k=0, dirichlet=dirichlet, kind='tensor'),
        vs=vs, tol=1e-8, maxiter=2000,
    )
    assert int(info) <= 0, (
        f"Tensor Hodge precond k=0 dirichlet={dirichlet} did not converge "
        f"(info={int(info)})"
    )


# ---------------------------------------------------------------------------
# Fast-diagonalisation Hodge preconditioner (k = 3)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Fast-diagonalisation Hodge preconditioner (k = 3)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("coupled_preconditioner", [False, True])
def test_jacobi_outer_k3_solve_converges(torus_seq_tensor, coupled_preconditioner):
    """k=3 inverse Hodge solve with the production jacobi Schur outer.

    (Chebyshev outer removed 2026-08-14 -- see mrx/experimental/chebyshev.py.)
    """
    seq = torus_seq_tensor
    dirichlet = False
    n = _dof(seq, 3, dirichlet)
    rhs = jax.random.normal(jax.random.PRNGKey(23), (n,))
    # NOTE: the schur INNER must not use surgery_schur (validator forbids
    # nesting surgery inside the Schur smoother; the old chebyshev-era test
    # carried that invalid spec and failed since before 2026-08).
    preconditioner = SaddlePointPreconditionerSpec(
        mass=MassPreconditionerSpec(kind='tensor', surgery_schur=True),
        schur=SchurPreconditionerSpec(
            inner=MassPreconditionerSpec(kind='tensor'),
            outer=MassPreconditionerSpec(kind='jacobi'),
        ),
        coupled=coupled_preconditioner,
    )

    _, info = seq.apply_inverse_hodge_laplacian(
        rhs,
        k=3,
        dirichlet=dirichlet,
        preconditioner=preconditioner,
        return_info=True,
    )

    assert int(info) <= 0, (
        "jacobi-outer k=3 inverse Hodge solve did not converge "
        f"(coupled={coupled_preconditioner}, info={int(info)})"
    )


# (3, 'jacobi') dropped 2026-08-15: an ALL-jacobi k=3 saddle is not a
# shipped configuration (production = tensor masses + jacobi Schur outer,
# covered by test_jacobi_outer_k3_solve_converges) and needs >1000 MINRES
# iterations on this fixture.
