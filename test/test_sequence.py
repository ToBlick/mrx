"""Tests on the shared session-scoped DeRham sequence.

These all reuse the ``torus_seq`` fixture so the expensive assembly runs
exactly once. Each test builds a dense view of whatever operator it needs by
probing the sparse matvec with unit vectors. At the session's (n, p) this is
cheap (a few hundred columns at most) and lets us verify global spectral
properties with ``scipy.linalg.eigh``.
"""

from test.conftest import BETTI, TORUS_EPSILON, build_dense

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
from scipy.linalg import eigh

from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction
from mrx.preconditioners import (
    MassPreconditionerSpec,
    SaddlePointPreconditionerSpec,
    SchurPreconditionerSpec,
)
from mrx.projectors import Projector

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
    projector = Projector(seq, 0, dirichlet=False)
    coeffs = jnp.linspace(-0.75, 0.5, seq.n0)
    discrete = DiscreteFunction(coeffs, seq.basis_0, seq.e0)
    recovered = projector.zeroform_interpolation(discrete)
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
    projector = Projector(seq, 0, dirichlet=False)
    coeffs = jnp.linspace(-0.6, 0.7, seq.n0)
    discrete = DiscreteFunction(coeffs, seq.basis_0, seq.e0)
    recovered = projector.zeroform_interpolation(discrete)
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
    projector = Projector(seq, 2, dirichlet=False)
    coeffs = jnp.linspace(-0.5, 0.75, seq.n2)
    discrete = DiscreteFunction(coeffs, seq.basis_2, seq.e2)
    recovered = projector.twoform_histopolation(discrete)
    npt.assert_allclose(recovered, coeffs, atol=1e-11)


def test_oneform_histopolation_recovers_discrete_function(identity_clamped_seq):
    seq = identity_clamped_seq
    projector = Projector(seq, 1, dirichlet=False)
    coeffs = jnp.linspace(-0.4, 0.6, seq.n1)
    discrete = DiscreteFunction(coeffs, seq.basis_1, seq.e1)
    recovered = projector.oneform_histopolation(discrete)
    npt.assert_allclose(recovered, coeffs, atol=1e-11)


def test_threeform_histopolation_recovers_discrete_function(identity_clamped_seq):
    seq = identity_clamped_seq
    projector = Projector(seq, 3, dirichlet=False)
    coeffs = jnp.linspace(-0.3, 0.4, seq.n3)
    discrete = DiscreteFunction(coeffs, seq.basis_3, seq.e3)
    recovered = projector.threeform_histopolation(discrete)
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
    projector = Projector(seq, 1, dirichlet=False)
    coeffs = jnp.linspace(-0.35, 0.55, seq.n1)
    discrete = DiscreteFunction(coeffs, seq.basis_1, seq.e1)
    recovered = projector.oneform_histopolation(discrete)
    npt.assert_allclose(recovered, coeffs, atol=1e-11)


# ---------------------------------------------------------------------------
# Mass matrix: symmetric positive definite
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", ALL_K)
@pytest.mark.parametrize("dirichlet", ALL_DBC)
def test_mass_matrix_spd(torus_seq, k, dirichlet):
    n = _dof(torus_seq, k, dirichlet)
    M = np.asarray(build_dense(
        lambda v: torus_seq.apply_mass_matrix(v, k, dirichlet=dirichlet), n))
    npt.assert_allclose(M, M.T, atol=1e-10, err_msg="M_k not symmetric")
    eigs = np.linalg.eigvalsh(M)
    assert eigs.min(
    ) > 0, f"M_{k} dirichlet={dirichlet} has non-positive eigenvalue {eigs.min()}"


# ---------------------------------------------------------------------------
# Stiffness matrix: symmetric positive semidefinite
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", (0, 1, 2))  # k=3 stiffness is zero
@pytest.mark.parametrize("dirichlet", ALL_DBC)
def test_stiffness_psd(torus_seq, k, dirichlet):
    n = _dof(torus_seq, k, dirichlet)
    S = np.asarray(build_dense(
        lambda v: torus_seq.apply_stiffness(v, k, dirichlet=dirichlet), n))
    npt.assert_allclose(S, S.T, atol=1e-9, err_msg="S_k not symmetric")
    eigs = np.linalg.eigvalsh(S)
    assert eigs.min() > - \
        1e-9, f"S_{k} dirichlet={dirichlet} has large negative eigenvalue {eigs.min()}"


# ---------------------------------------------------------------------------
# Hodge Laplacian: symmetric positive semidefinite, kernel has expected dim
# ---------------------------------------------------------------------------

def _expected_null_dim(k, dirichlet, betti):
    b0, b1, b2, _b3 = betti
    if dirichlet:
        return (0, b2, b1, b0)[k]
    return (b0, b1, b2, 0)[k]


@pytest.mark.parametrize("k", ALL_K)
@pytest.mark.parametrize("dirichlet", ALL_DBC)
def test_hodge_laplacian_psd_and_kernel(torus_seq, k, dirichlet):
    seq = torus_seq
    n = _dof(seq, k, dirichlet)

    # L_k v = λ M_k v, so solve the generalised eigenproblem.
    L = np.asarray(build_dense(
        lambda v: seq.apply_hodge_laplacian(v, k, dirichlet=dirichlet), n))
    M = np.asarray(build_dense(
        lambda v: seq.apply_mass_matrix(v, k, dirichlet=dirichlet), n))

    npt.assert_allclose(L, L.T, atol=1e-8, err_msg="L_k not symmetric")

    eigvals = eigh(L, M, eigvals_only=True)
    assert eigvals.min() > -1e-8, (
        f"L_{k} dirichlet={dirichlet} has large negative generalised eigenvalue "
        f"{eigvals.min()}")

    # Number of numerical zeros matches the expected harmonic-space dimension.
    expected = _expected_null_dim(k, dirichlet, BETTI)
    lam_max = max(eigvals.max(), 1.0)
    n_zero = int(np.sum(eigvals < 1e-6 * lam_max))
    assert n_zero == expected, (
        f"L_{k} dirichlet={dirichlet}: expected {expected} zero eigenvalues, "
        f"got {n_zero}. Smallest eigenvalues: {eigvals[:max(3, expected + 1)]}")


# ---------------------------------------------------------------------------
# Derivative chain d∘d = 0 (via the strong operators)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dirichlet", ALL_DBC)
def test_curl_grad_zero(torus_seq, dirichlet):
    seq = torus_seq
    n0 = _dof(seq, 0, dirichlet)
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (n0,))
    grad_x = seq.apply_strong_grad(
        x, dirichlet_in=dirichlet, dirichlet_out=dirichlet)
    cg = seq.apply_strong_curl(
        grad_x, dirichlet_in=dirichlet, dirichlet_out=dirichlet)
    l2_sq = float(cg @ seq.apply_mass_matrix(cg, k=2, dirichlet=dirichlet))
    assert l2_sq < 1e-12


@pytest.mark.parametrize("dirichlet", ALL_DBC)
def test_div_curl_zero(torus_seq, dirichlet):
    seq = torus_seq
    n1 = _dof(seq, 1, dirichlet)
    key = jax.random.PRNGKey(1)
    x = jax.random.normal(key, (n1,))
    curl_x = seq.apply_strong_curl(
        x, dirichlet_in=dirichlet, dirichlet_out=dirichlet)
    dc = seq.apply_strong_div(
        curl_x, dirichlet_in=dirichlet, dirichlet_out=dirichlet)
    l2_sq = float(dc @ seq.apply_mass_matrix(dc, k=3, dirichlet=dirichlet))
    assert l2_sq < 1e-12


# ---------------------------------------------------------------------------
# Topological incidence matrix: structural ±1 entries and equivalence with
# the weak-form derivative through the mass matrix.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", (0, 1, 2))
def test_incidence_entries_are_topological(torus_seq, k):
    """Gk stored on the operators bundle has entries in {-1, 0, +1}."""
    ops = torus_seq.get_operators()
    sp = getattr(ops, f"g{k}_sp")
    assert sp is not None, f"g{k}_sp was not assembled"
    data = np.asarray(sp.data)
    # All nonzero values must be exactly ±1 (up to floating-point equality).
    assert np.all(np.isin(data, [-1.0, 0.0, 1.0])), (
        f"g{k} has non-{{-1,0,+1}} entries: unique values "
        f"{np.unique(data)}"
    )


@pytest.mark.parametrize("k", (0, 1, 2))
def test_strong_derivative_matches_weak_through_mass(torus_seq, k):
    """On the FULL DoF grid, ``D_k = M_{k+1} G_k`` by construction.

    ``apply_derivative_matrix`` applies this composition without
    materialising ``D_k``.  We verify agreement between the lazy matvec
    path and the explicit dense product, sampled via the non-Dirichlet
    extraction sandwich.
    """
    seq = torus_seq
    ops = seq.get_operators()
    g_sp = getattr(ops, f"g{k}_sp")
    m_sp = getattr(ops, f"m{k + 1}_sp")
    # Compare D_ext produced by apply_derivative_matrix (column-by-column)
    # against the explicit sandwich of (M_{k+1} G_k).
    e_in_T = getattr(seq, f"e{k}_T").todense()
    e_out = getattr(seq, f"e{k + 1}").todense()
    d_full = m_sp.todense() @ g_sp.todense()
    d_ext_ref = e_out @ d_full @ e_in_T
    n_in = d_ext_ref.shape[1]
    # Matvec each unit vector through apply_derivative_matrix
    from mrx.operators import apply_derivative_matrix
    cols = [
        apply_derivative_matrix(
            seq, ops, jnp.eye(n_in)[:, j], k,
            dirichlet_in=False, dirichlet_out=False)
        for j in range(n_in)
    ]
    d_ext_apply = jnp.stack(cols, axis=1)
    num = float(jnp.linalg.norm(d_ext_ref - d_ext_apply))
    den = float(jnp.linalg.norm(d_ext_ref))
    assert num < 1e-6 * max(den, 1.0), (
        f"D{k} matvec ≠ M{k+1} G{k} sandwich: rel err = "
        f"{num / max(den, 1.0):.3e}"
    )


# ---------------------------------------------------------------------------
# Stored harmonic forms are actual nullspace vectors of L_k
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", ALL_K)
@pytest.mark.parametrize("dirichlet", ALL_DBC)
def test_stored_nullspace_is_harmonic(torus_seq, k, dirichlet):
    seq = torus_seq
    vs = getattr(seq, f"null_{k}_dbc" if dirichlet else f"null_{k}")
    # Stored dimension matches topology.
    assert vs.shape[0] == _expected_null_dim(k, dirichlet, BETTI)
    for v in vs:
        Lv = seq.apply_hodge_laplacian(v, k, dirichlet=dirichlet)
        # Rayleigh quotient v^T L v / v^T M v must be numerically zero.
        Mv = seq.apply_mass_matrix(v, k, dirichlet=dirichlet)
        rq = float(jnp.abs(v @ Lv) / jnp.abs(v @ Mv))
        assert rq < 1e-6, f"harmonic k={k} dbc={dirichlet} has Rayleigh quotient {rq}"


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
# Analytical Poisson benchmark on the donut torus
# ---------------------------------------------------------------------------
#
# On the donut torus with minor radius ``a = TORUS_EPSILON``, the function
#
#     u(r, chi, z) = (1/4) (r^2 - r^4) cos(2 pi z)
#
# vanishes at r = 0 and r = 1 and therefore satisfies homogeneous Dirichlet
# BCs on the full boundary. It is manufactured so that ``-Delta u = f`` with
# ``f`` given by ``_poisson_source`` below.

def _poisson_exact(x):
    r, _chi, z = x
    pi = jnp.pi
    return 1 / 4 * (r ** 2 - r ** 4) * jnp.cos(2 * pi * z) * jnp.ones(1)


def _poisson_source(x):
    r, chi, z = x
    pi = jnp.pi
    a = TORUS_EPSILON
    radius = 1.0 + a * r * jnp.cos(2 * pi * chi)
    return (
        jnp.cos(2 * pi * z)
        * (
            -1.0 / a ** 2 * (1.0 - 4.0 * r ** 2)
            - 1.0 / (a * radius) * (r / 2.0 - r ** 3) * jnp.cos(2 * pi * chi)
            + 1.0 / 4.0 * (r ** 2 - r ** 4) / radius ** 2
        )
        * jnp.ones(1)
    )


def test_poisson_k0_matches_analytical(torus_seq):
    """Solve -Delta u = f with homogeneous DBC; check L2 error vs the known u."""
    seq = torus_seq
    rhs = seq.p0_dbc(_poisson_source)
    u_hat = seq.apply_inverse_hodge_laplacian(rhs, k=0, dirichlet=True)
    u_h = DiscreteFunction(u_hat, seq.basis_0, seq.e0_dbc)

    diff_vals = jax.lax.map(
        lambda x: _poisson_exact(x) - u_h(x), seq.quad.x, batch_size=20_000,
    )
    u_vals = jax.vmap(_poisson_exact)(seq.quad.x)
    l2_diff_sq = jnp.einsum(
        "ik,ik,i,i->", diff_vals, diff_vals, seq.jacobian_j, seq.quad.w)
    l2_u_sq = jnp.einsum(
        "ik,ik,i,i->", u_vals, u_vals, seq.jacobian_j, seq.quad.w)
    rel_err = float(jnp.sqrt(l2_diff_sq / l2_u_sq))
    assert rel_err < 1e-1, (
        f"k=0 Poisson relative L2 error too large on the spline-projected "
        f"donut torus: {rel_err:.3e}")


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

def test_fd_hodge_preconditioner_spd_k3(torus_seq):
    """``apply_hodge_laplacian_preconditioner(kind='tensor')`` is SPD for k=3."""
    seq = torus_seq
    dirichlet = True
    n = _dof(seq, 3, dirichlet)
    P = np.asarray(build_dense(
        lambda v: seq.apply_hodge_laplacian_preconditioner(
            v, k=3, dirichlet=dirichlet, kind='tensor'),
        n))
    npt.assert_allclose(P, P.T, atol=1e-9,
                        err_msg="Tensor Hodge precond k=3 not symmetric")
    eigs = np.linalg.eigvalsh(P)
    assert eigs.min() > -1e-9, (
        f"Tensor Hodge precond k=3 dirichlet={dirichlet} has large negative "
        f"eigenvalue {eigs.min()}"
    )


@pytest.mark.parametrize("coupled_preconditioner", [False, True])
def test_chebyshev_preconditioned_k3_solve_converges(torus_seq, coupled_preconditioner):
    """k=3 inverse Hodge solve accepts the production Chebyshev path."""
    seq = torus_seq
    dirichlet = False
    n = _dof(seq, 3, dirichlet)
    rhs = jax.random.normal(jax.random.PRNGKey(23), (n,))
    preconditioner = SaddlePointPreconditionerSpec(
        mass=MassPreconditionerSpec(kind='tensor', surgery_schur=True),
        schur=SchurPreconditionerSpec(
            inner=MassPreconditionerSpec(kind='tensor', surgery_schur=True),
            outer=MassPreconditionerSpec(
                kind='chebyshev',
                steps=4,
                power_iterations=8,
                min_eig_fraction=1e-3,
            ),
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
        "Chebyshev k=3 inverse Hodge solve did not converge "
        f"(coupled={coupled_preconditioner}, info={int(info)})"
    )


@pytest.mark.parametrize(
    ("k", "preconditioner"),
    [
        (0, 'jacobi'),
        (0, 'tensor'),
        (0, MassPreconditionerSpec(
            kind='chebyshev',
            steps=4,
            power_iterations=8,
            min_eig_fraction=1e-3,
        )),
        (3, 'jacobi'),
        (3, 'tensor'),
        (3, MassPreconditionerSpec(
            kind='chebyshev',
            steps=4,
            power_iterations=8,
            min_eig_fraction=1e-3,
        )),
    ],
)
def test_diffusion_solver_default_preconditioners_converge(
        torus_seq, k, preconditioner):
    """Diffusion solve accepts Jacobi, tensor, and Chebyshev out of the box."""
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
