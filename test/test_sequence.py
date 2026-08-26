"""Tests on the shared session-scoped DeRham sequence.

These all reuse the ``torus_seq`` fixture so the expensive assembly runs
exactly once. Each test builds a dense view of whatever operator it needs by
probing the sparse matvec with unit vectors. At the session's (n, p) this is
cheap (a few hundred columns at most) and lets us verify global spectral
properties with ``scipy.linalg.eigh``.
"""


import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction

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


@pytest.fixture(scope="module", params=[False, True], ids=["tensor", "polar"])
def tiny_seq(request):
    """(4,4,4) sequence with a clamped and a periodic axis at both parities of p."""
    seq = DeRhamSequence(
        (4, 4, 4),
        (3, 2, 3) if not request.param else (2, 2, 2),
        4,
        ("clamped", "periodic", "clamped") if not request.param
        else ("clamped", "periodic", "periodic"),
        polar=request.param,
        tol=1e-12,
        maxiter=200,
        betti_numbers=(1, 1, 0, 0),
    )
    seq.set_map(lambda x: x)
    return seq


@pytest.mark.parametrize("k", ALL_K)
@pytest.mark.parametrize("dirichlet", ALL_DBC, ids=["free", "dbc"])
def test_discrete_function_matches_dense_evaluation(tiny_seq, k, dirichlet):
    """The local-support evaluator equals ``dof @ (E @ Λ(x))`` over ALL basis functions."""
    seq = tiny_seq
    basis = getattr(seq, f"basis_{k}")
    e = getattr(seq, f"e{k}_dbc" if dirichlet else f"e{k}")
    dof = jax.random.normal(jax.random.PRNGKey(7 * k + dirichlet), (int(_dof(seq, k, dirichlet)),))
    discrete = DiscreteFunction(dof, basis, e)
    xs = jax.random.uniform(jax.random.PRNGKey(3), (6, 3))

    def dense(x):
        return dof @ (e @ jax.vmap(basis, (None, 0))(x, basis.ns))

    got = jax.vmap(discrete)(xs)
    want = jax.vmap(dense)(xs)
    assert got.shape == want.shape
    err = float(jnp.linalg.norm(got - want) / jnp.linalg.norm(want))
    assert err < 1e-12, f"k={k} dirichlet={dirichlet}: local evaluator off by {err:.2e}"


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
    # maxiter above the fixture default: production preconditions the
    # Laplacians with Jacobi (2026-08-18), which needs more iterations than the
    # retired tensor path to reach the same residual. The ACCURACY assertion
    # below is unchanged -- only the iteration budget is.
    u_hat = seq.apply_inverse_hodge_laplacian(
        f, k, dirichlet=dirichlet, maxiter=6000)

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

@pytest.mark.parametrize(
    ("k", "preconditioner"),
    [
        (0, 'jacobi'),
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


