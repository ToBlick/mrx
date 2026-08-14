"""Tests on the shared session-scoped DeRham sequence.

These all reuse the ``torus_seq`` fixture so the expensive assembly runs
exactly once. Each test builds a dense view of whatever operator it needs by
probing the sparse matvec with unit vectors. At the session's (n, p) this is
cheap (a few hundred columns at most) and lets us verify global spectral
properties with ``scipy.linalg.eigh``.
"""

from test.conftest import build_dense

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction
from mrx.preconditioners import (
    MassPreconditionerSpec,
    SaddlePointPreconditionerSpec,
    SchurPreconditionerSpec,
)

ALL_K = (0, 1, 2, 3)
ALL_DBC = (False, True)


def _dof(seq, k, dirichlet):
    return getattr(seq, f"n{k}_dbc" if dirichlet else f"n{k}")


def test_zeroform_greville_interpolation_recovers_discrete_function():
    seq = DeRhamSequence(
        (5, 4, 4),
        (3, 2, 2),
        6,
        ("clamped", "periodic", "periodic"),
        polar=False,
        tol=1e-12,
        maxiter=200,
        betti_numbers=(1, 1, 0, 0),
    )
    coeffs = jnp.linspace(-0.75, 0.5, seq.n0)
    discrete = DiscreteFunction(coeffs, seq.basis_0, seq.e0)
    recovered = seq.interpolate(discrete, 0)
    npt.assert_allclose(recovered, coeffs, atol=1e-12)


def test_polar_zeroform_greville_interpolation_recovers_discrete_function():
    seq = DeRhamSequence(
        (5, 4, 4),
        (3, 2, 2),
        6,
        ("clamped", "periodic", "periodic"),
        polar=True,
        tol=1e-12,
        maxiter=200,
        betti_numbers=(1, 1, 0, 0),
    )
    seq.set_map(lambda x: x)
    coeffs = jnp.linspace(-0.6, 0.7, seq.n0)
    discrete = DiscreteFunction(coeffs, seq.basis_0, seq.e0)
    recovered = seq.interpolate(discrete, 0)
    npt.assert_allclose(recovered, coeffs, atol=1e-12)


@pytest.fixture(scope="module")
def identity_clamped_seq():
    seq = DeRhamSequence(
        (5, 4, 4),
        (3, 2, 2),
        6,
        ("clamped", "clamped", "clamped"),
        polar=False,
        tol=1e-12,
        maxiter=200,
        betti_numbers=(1, 1, 0, 0),
    )
    seq.set_map(lambda x: x)
    return seq


def test_twoform_histopolation_recovers_discrete_function(identity_clamped_seq):
    seq = identity_clamped_seq
    coeffs = jnp.linspace(-0.5, 0.75, seq.n2)
    discrete = DiscreteFunction(coeffs, seq.basis_2, seq.e2)
    recovered = seq.interpolate(discrete, 2)
    npt.assert_allclose(recovered, coeffs, atol=1e-11)


def test_oneform_histopolation_recovers_discrete_function(identity_clamped_seq):
    seq = identity_clamped_seq
    coeffs = jnp.linspace(-0.4, 0.6, seq.n1)
    discrete = DiscreteFunction(coeffs, seq.basis_1, seq.e1)
    recovered = seq.interpolate(discrete, 1)
    npt.assert_allclose(recovered, coeffs, atol=1e-11)


def test_threeform_histopolation_recovers_discrete_function(identity_clamped_seq):
    seq = identity_clamped_seq
    coeffs = jnp.linspace(-0.3, 0.4, seq.n3)
    discrete = DiscreteFunction(coeffs, seq.basis_3, seq.e3)
    recovered = seq.interpolate(discrete, 3)
    npt.assert_allclose(recovered, coeffs, atol=1e-11)


def test_polar_oneform_histopolation_recovers_discrete_function():
    seq = DeRhamSequence(
        (5, 4, 4),
        (3, 2, 2),
        6,
        ("clamped", "periodic", "periodic"),
        polar=True,
        tol=1e-12,
        maxiter=200,
        betti_numbers=(1, 1, 0, 0),
    )
    seq.set_map(lambda x: x)
    coeffs = jnp.linspace(-0.35, 0.55, seq.n1)
    discrete = DiscreteFunction(coeffs, seq.basis_1, seq.e1)
    recovered = seq.interpolate(discrete, 1)
    npt.assert_allclose(recovered, coeffs, atol=1e-11)


# ---------------------------------------------------------------------------
# Derivatives annihilate stored harmonic forms (they are closed AND coclosed)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k,dirichlet", [(0, False), (1, False), (1, True), (2, True)])
def test_harmonic_forms_closed(torus_seq, k, dirichlet):
    """For a harmonic k-form v, d v = 0 in the dual sense."""
    seq = torus_seq
    vs = getattr(seq, f"null_{k}_dbc" if dirichlet else f"null_{k}")
    if vs.shape[0] == 0:
        pytest.skip("no harmonic forms for this (k, dirichlet)")
    for v in vs:
        dv = seq.apply_derivative_matrix(
            v, k, dirichlet_in=dirichlet, dirichlet_out=dirichlet)
        # normalise by the mass of v to get a scale-invariant tolerance.
        v_mass = float(seq.l2_norm(v, k, dirichlet=dirichlet))
        assert jnp.linalg.norm(dv) < 1e-6 * max(v_mass, 1.0), (
            f"harmonic k={k} dbc={dirichlet} is not closed: ||dv|| = {jnp.linalg.norm(dv)}")


# ---------------------------------------------------------------------------
# Hodge Laplacian solve: L_k u = f has u as its solution on the non-kernel part
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k,dirichlet", [(0, True), (3, True)])
def test_hodge_laplacian_solve_roundtrip(torus_seq, k, dirichlet):
    """L_k u = L_k u_0  =>  apply_inverse returns u_0 (up to kernel)."""
    seq = torus_seq
    n = _dof(seq, k, dirichlet)
    key = jax.random.PRNGKey(100 + k)
    u = jax.random.normal(key, (n,))
    f = seq.apply_hodge_laplacian(u, k, dirichlet=dirichlet)
    u_hat = seq.apply_inverse_hodge_laplacian(f, k, dirichlet=dirichlet)

    # Remove the kernel component (M-orthogonal projection) from both sides.
    vs = getattr(seq, f"null_{k}_dbc" if dirichlet else f"null_{k}")

    def deflate(x):
        for w in vs:
            coeff = w @ seq.apply_mass_matrix(x, k, dirichlet=dirichlet)
            x = x - coeff * w
        return x
    diff = float(seq.l2_norm(
        deflate(u) - deflate(u_hat), k, dirichlet=dirichlet))
    u_mass = float(seq.l2_norm(deflate(u), k, dirichlet=dirichlet))
    assert diff < 1e-5 * max(u_mass, 1.0), (
        f"L_{k} solve round-trip residual {diff} too large (|u|_M = {u_mass})")


# ---------------------------------------------------------------------------
# Fast-diagonalisation Hodge preconditioner (k = 0)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dirichlet", ALL_DBC)
def test_fd_hodge_preconditioner_spd(torus_seq, dirichlet):
    """``apply_hodge_laplacian_preconditioner(kind='tensor')`` is SPD for k=0."""
    seq = torus_seq
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
def test_fd_hodge_preconditioner_accelerates_cg(torus_seq, dirichlet):
    """Tensor Hodge preconditioner runs and converges at k=0."""
    seq = torus_seq
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

@pytest.mark.parametrize("coupled_preconditioner", [False, True])
def test_jacobi_outer_k3_solve_converges(torus_seq, coupled_preconditioner):
    """k=3 inverse Hodge solve with the production jacobi Schur outer.

    (Chebyshev outer removed 2026-08-14 -- see mrx/experimental/chebyshev.py.)
    """
    seq = torus_seq
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


@pytest.mark.parametrize(
    ("k", "preconditioner"),
    [
        (0, 'jacobi'),
        (0, 'tensor'),
        (3, 'jacobi'),
        (3, 'tensor'),
    ],
)
def test_diffusion_solver_default_preconditioners_converge(
        torus_seq, k, preconditioner):
    """Diffusion solve accepts Jacobi and tensor out of the box."""
    seq = torus_seq
    dirichlet = False
    eps = 1e-2
    n = _dof(seq, k, dirichlet)
    rhs = jax.random.normal(jax.random.PRNGKey(123 + 17 * k), (n,))

    _, info = seq.apply_inverse_mass_plus_eps_laplace_matrix(
        rhs,
        k=k,
        eps=eps,
        dirichlet=dirichlet,
        preconditioner=preconditioner,
        return_info=True,
    )

    assert int(info) <= 0, (
        "Diffusion solve did not converge with built-in preconditioner "
        f"{preconditioner!r} for k={k} (info={int(info)})"
    )


