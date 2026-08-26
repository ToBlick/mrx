"""The local evaluator and the diffusion solve of ``DeRhamSequence``.

The interpolation round trips that used to live here handed a
``DiscreteFunction`` of the target space straight to ``interpolate``, which
returns its DOFs through ``projectors._matching_discrete_dofs`` without
interpolating anything; the identity they meant is
``test_projectors.test_interpolation_reproduces_its_own_space`` (polar, both
parities of p) and ``test_pi_full_is_idempotent`` (full tensor space).
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
    return getattr(seq, f"n{k}_dbc" if dirichlet else f"n{k}")


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


