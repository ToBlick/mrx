"""Interpolation identities, the local evaluator, and the Hodge Laplacian
solve on the session ``tiny_seq``.

Each test either builds its own small identity-map sequence or reuses the
session fixture; nothing here depends on resolution.
"""


import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

import mrx
from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction

ALL_K = (0, 1, 2, 3)
ALL_DBC = (False, True)

# Greville interpolation and histopolation recover the coefficients of a
# function already in the space up to the roundoff of the collocation solves:
# 1e4 eps = 2.2e-12 f64 / 1.2e-3 f32 (the histopolation of a 1-form goes
# through a solve per histopolated axis, the polar extraction adds its
# pseudo-inverse). The local evaluator against the dense sum is a pure
# roundoff identity: 1e3 eps.
RECOVER = mrx.eps(1e4)
IDENT = mrx.eps(1e3)


def _dof(seq, k, dirichlet):
    return getattr(seq, f"n{k}_dbc" if dirichlet else f"n{k}")


def test_zeroform_greville_interpolation_recovers_discrete_function():
    seq = DeRhamSequence(
        (5, 4, 4),
        (3, 2, 2),
        6,
        ("clamped", "periodic", "periodic"),
        polar=False,
        maxiter=200,
        betti_numbers=(1, 1, 0, 0),
    )
    coeffs = jnp.linspace(-0.75, 0.5, seq.n0)
    discrete = DiscreteFunction(coeffs, seq.basis_0, seq.e0)
    recovered = seq.interpolate(discrete, 0)
    npt.assert_allclose(recovered, coeffs, atol=RECOVER)


@pytest.fixture(scope="module", params=[False, True], ids=["tensor", "polar"])
def evaluator_seq(request):
    """(4,4,4) sequence with a clamped and a periodic axis at both parities of p."""
    seq = DeRhamSequence(
        (4, 4, 4),
        (3, 2, 3) if not request.param else (2, 2, 2),
        4,
        ("clamped", "periodic", "clamped") if not request.param
        else ("clamped", "periodic", "periodic"),
        polar=request.param,
        maxiter=200,
        betti_numbers=(1, 1, 0, 0),
    )
    seq.set_map(lambda x: x)
    return seq


@pytest.mark.parametrize("k", ALL_K)
def test_discrete_function_matches_dense_evaluation(evaluator_seq, k):
    """The local-support evaluator equals ``dof @ (E @ Λ(x))`` over ALL basis functions.

    The dense sweep over every raw basis function is the expensive half and
    does not depend on the extraction, so it is done once and both the free
    and the Dirichlet extraction are checked against it.
    """
    seq = evaluator_seq
    basis = getattr(seq, f"basis_{k}")
    xs = jax.random.uniform(jax.random.PRNGKey(3), (6, 3))
    raw = jax.vmap(lambda x: jax.vmap(basis, (None, 0))(x, basis.ns))(xs)
    for dirichlet in ALL_DBC:
        e = getattr(seq, f"e{k}_dbc" if dirichlet else f"e{k}")
        dof = jax.random.normal(jax.random.PRNGKey(7 * k + dirichlet),
                                (int(_dof(seq, k, dirichlet)),))
        discrete = DiscreteFunction(dof, basis, e)
        got = jax.vmap(discrete)(xs)
        want = jax.vmap(lambda lam, dof=dof, e=e: dof @ (e @ lam))(raw)
        assert got.shape == want.shape
        err = float(jnp.linalg.norm(got - want) / jnp.linalg.norm(want))
        assert err < IDENT, f"k={k} dirichlet={dirichlet}: local evaluator off by {err:.2e}"


def test_polar_zeroform_greville_interpolation_recovers_discrete_function():
    seq = DeRhamSequence(
        (5, 4, 4),
        (3, 2, 2),
        6,
        ("clamped", "periodic", "periodic"),
        polar=True,
        maxiter=200,
        betti_numbers=(1, 1, 0, 0),
    )
    seq.set_map(lambda x: x)
    coeffs = jnp.linspace(-0.6, 0.7, seq.n0)
    discrete = DiscreteFunction(coeffs, seq.basis_0, seq.e0)
    recovered = seq.interpolate(discrete, 0)
    npt.assert_allclose(recovered, coeffs, atol=RECOVER)


@pytest.fixture(scope="module")
def identity_clamped_seq():
    seq = DeRhamSequence(
        (5, 4, 4),
        (3, 2, 2),
        6,
        ("clamped", "clamped", "clamped"),
        polar=False,
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
    npt.assert_allclose(recovered, coeffs, atol=RECOVER)


def test_oneform_histopolation_recovers_discrete_function(identity_clamped_seq):
    seq = identity_clamped_seq
    coeffs = jnp.linspace(-0.4, 0.6, seq.n1)
    discrete = DiscreteFunction(coeffs, seq.basis_1, seq.e1)
    recovered = seq.interpolate(discrete, 1)
    npt.assert_allclose(recovered, coeffs, atol=RECOVER)


def test_threeform_histopolation_recovers_discrete_function(identity_clamped_seq):
    seq = identity_clamped_seq
    coeffs = jnp.linspace(-0.3, 0.4, seq.n3)
    discrete = DiscreteFunction(coeffs, seq.basis_3, seq.e3)
    recovered = seq.interpolate(discrete, 3)
    npt.assert_allclose(recovered, coeffs, atol=RECOVER)


def test_polar_oneform_histopolation_recovers_discrete_function():
    seq = DeRhamSequence(
        (5, 4, 4),
        (3, 2, 2),
        6,
        ("clamped", "periodic", "periodic"),
        polar=True,
        maxiter=200,
        betti_numbers=(1, 1, 0, 0),
    )
    seq.set_map(lambda x: x)
    coeffs = jnp.linspace(-0.35, 0.55, seq.n1)
    discrete = DiscreteFunction(coeffs, seq.basis_1, seq.e1)
    recovered = seq.interpolate(discrete, 1)
    npt.assert_allclose(recovered, coeffs, atol=RECOVER)


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
        tiny_seq, k, preconditioner):
    """Diffusion solve accepts Jacobi out of the box."""
    seq = tiny_seq
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


