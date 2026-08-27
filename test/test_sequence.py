"""The local evaluator and the diffusion solve of ``DeRhamSequence``.

The interpolation round trips that used to live here handed a
``DiscreteFunction`` of the target space straight to ``interpolate``, which
returns its DOFs through ``projectors._matching_discrete_dofs`` without
interpolating anything; the identity they meant is
``test_projectors.test_interpolation_reproduces_its_own_space`` (polar, both
parities of p).
"""


import jax
import jax.numpy as jnp
import pytest

import mrx
from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction

ALL_K = (0, 1, 2, 3)
ALL_DBC = (False, True)

# The local evaluator against the dense sum is a pure roundoff identity.
IDENT = mrx.eps(1e3)


def _dof(seq, k, dirichlet):
    return seq.n(k, dirichlet)


@pytest.fixture(scope="module")
def evaluator_seq():
    """(4,4,4) p=2 polar sequence on the identity map."""
    seq = DeRhamSequence(
        (4, 4, 4),
        (2, 2, 2),
        4,
        ("clamped", "periodic", "periodic"),
        polar=True,
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
        e = seq.E(k, dirichlet)
        dof = jax.random.normal(jax.random.PRNGKey(7 * k + dirichlet),
                                (int(_dof(seq, k, dirichlet)),))
        discrete = DiscreteFunction(dof, basis, e)
        got = jax.vmap(discrete)(xs)
        want = jax.vmap(lambda lam, dof=dof, e=e: dof @ (e @ lam))(raw)
        assert got.shape == want.shape
        err = float(jnp.linalg.norm(got - want) / jnp.linalg.norm(want))
        assert err < IDENT, f"k={k} dirichlet={dirichlet}: local evaluator off by {err:.2e}"


# ---------------------------------------------------------------------------
# Fast-diagonalisation Hodge preconditioner (k = 0)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("k", "preconditioner"),
    [
        (0, 'auto'),
    ],
)
def test_diffusion_solver_default_preconditioners_converge(
        tiny_seq, k, preconditioner):
    """The diffusion solve converges with its default preconditioner."""
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


