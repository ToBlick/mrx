"""The harmonic forms of a solid torus, the null spaces of the discrete Hodge Laplacians ``L_k``.

The Laplacian and Leray solves on a sequence use these forms to remove the null space of ``L_k``.
Run :func:`compute_nullspaces` once the preconditioners are built. The forms depend on the geometry, so
installing a new map drops them together with the preconditioners, and both must be built again.
:func:`harmonic_rayleigh` checks how harmonic a given form is.

In float32 the forms are computed and stored as :class:`~mrx.double_single.DS`. A form rounded to float32 is
1e-7 from the kernel, and a solve that removes it does not get below that.
"""

import equinox as eqx
import jax
import jax.numpy as jnp

import mrx
import mrx.double_single as ds
import mrx.operators as op
from mrx.differential_forms import DERIV_AXES
from mrx.extraction_operators import conforming_restriction
from mrx.precision import PAIRS, inner, lift, to_working
from mrx.pytree import compiled_when_eager


#: The co-closed solve of the harmonic 2-form stops this factor below the tolerance of the sequence. The Rayleigh
#: quotient of the form is quadratic in the error of that solve times an eigenvalue of the stiffness, which grows
#: with the mesh.
HARMONIC_TOL = 0.1


def _commit(seq, operators):
    """Install ``operators`` on ``seq`` and return it. The next solve then removes the forms stored so far."""
    seq.operators = operators
    return operators


def harmonic_rayleigh(seq, v, k):
    """The Rayleigh quotient ``v^T L_k v / v^T M_k v`` (``M_k`` the mass matrix) of a ``k``-form ``v``
    in the space of ``seq``. It is zero for a harmonic form and of the order of the smallest nonzero
    eigenvalue otherwise. It takes a mass solve, so it is meant as a diagnostic.
    """
    v = lift(v)
    return inner(v, seq.L[k] @ v) / inner(v, seq.M[k] @ v)


@compiled_when_eager
def _normalised(space, v, k):
    """``v`` scaled to unit norm in ``M_k``. The projections that remove a form assume that norm to the
    accuracy of the form."""
    if PAIRS:
        return v * ds.rsqrt(ds.dot(v, space.M[k] @ v))
    return v / space.l2_norm(v, k)


def _constant_seed(seq, k):
    """The 1-form ``d zeta`` (k = 1) or the 2-form ``dr ^ dtheta`` (k = 2) in the space of ``seq``.

    Its third component is the constant 1 and the others vanish. The coefficients of a constant are 1 along an
    axis with the spline basis and the cell integrals of the unit-integral splines along an axis with the
    derivative basis. They are written down rather than interpolated, so that along the spline axes they are
    exactly equal and the form is closed to the round-off of the derivative, in float32 as well.
    """
    seed = conforming_restriction(seq.E(k), lift(_constant_raw(seq, k)), seq.core_rows(k))
    # The cell integrals come from the knots, which in float32 are uniform only to their rounding, so the seed
    # has the parity of the form only that far. Its part of that parity is closed as well, since the reflection
    # commutes with the derivative, and it is what the parity views reduce exactly.
    projector = seq.free_projector(k)
    if projector is not None:
        seed = _own_parity(projector, seed)
    closed = float(seq.l2_norm(to_working(seq.G[k] @ seed), k + 1) / seq.l2_norm(to_working(seed), k))
    if closed > mrx.sqrt_eps():
        raise RuntimeError(
            f"compute_nullspaces: the k={k} seed is not closed (|G seed| / |seed| = {closed:.2e})")
    return seed


@compiled_when_eager
def _normalised_row(space, v, k):
    """:func:`_normalised` as a row of the harmonic forms."""
    return _normalised(space, v, k)[None, :]


@compiled_when_eager
def _reduced_form(space, X, forms, k):
    """The first of the harmonic ``forms`` of the full sequence reduced by the expansion ``X`` of a parity view,
    normalised on its ``space``."""
    return _normalised(space, X.T @ forms[0], k)


@eqx.filter_jit
def _constant_raw(seq, k: int):
    """The raw tensor-product coefficients of the seed of :func:`_constant_seed`."""
    form = (seq.basis_0, seq.basis_1, seq.basis_2, seq.basis_3)[k]
    shapes = [tuple(int(n) for n in s) for s in form.shape]
    factors = []
    for a in range(3):
        if a in DERIV_AXES[k][2]:
            d = seq.basis_0.dLambda[a]
            factors.append(1.0 / jax.vmap(d._scale)(jnp.arange(d.n)))
        else:
            factors.append(jnp.ones(shapes[2][a]))
    third = jnp.einsum('i,j,k->ijk', *factors).ravel()
    return jnp.concatenate([jnp.zeros(shapes[0][0] * shapes[0][1] * shapes[0][2]),
                            jnp.zeros(shapes[1][0] * shapes[1][1] * shapes[1][2]), third])


@eqx.filter_jit
def _own_parity(projector, v):
    """``v`` projected onto its own stellarator parity."""
    return projector.post(v, projector.parity(v))


@eqx.filter_jit
def _plus_exact(space, seed, a, k: int, subtract: bool = False):
    """``seed + G_{k-1} a``, or ``seed - G_{k-1} a`` if ``subtract``."""
    exact = space.G[k - 1] @ a
    return seed - exact if subtract else seed + exact


def compute_nullspaces(seq, *, verbose=True):
    """Compute the harmonic forms of ``seq``, install them on ``seq.operators`` and return the operators.

    Call it after :meth:`~mrx.derham_sequence.DeRhamSequence.build_preconditioners`. The domain is a solid
    torus (Betti numbers ``(1, 1, 0, 0)``, of which the odd parity view keeps ``b1`` and the even view
    ``b0``). The forms are, in order:

    - k = 3 with Dirichlet conditions: ``M_3^{-1}`` applied to the constant.
    - k = 2 with Dirichlet conditions: the toroidal flux form ``dr ^ dtheta`` minus its exact part, found
      with one curl-curl solve.
    - k = 0 without boundary conditions: the constant.
    - k = 1 without boundary conditions: the Leray projection of ``d zeta``, found with one ``L_0`` solve.

    The parity views of ``seq`` get their forms as well. With ``verbose`` the
    :func:`harmonic_rayleigh` of each form is printed.
    """
    operators = _commit(seq, op.init_nullspaces(seq, seq._require_operators()))

    def store(space, k, v):
        return _commit(seq, op.set_nullspace(operators, k, space.dirichlet, _normalised_row(space, lift(v), k)))

    if op.n_vectors(seq.betti_numbers, 3, True):
        operators = store(seq, 3, seq.M[3].solve(jnp.ones(seq.n(3)), accurate=True))

    if op.n_vectors(seq.betti_numbers, 2, True):
        # the closed seed has no coexact part, so removing its exact part (the image of curl, one
        # curl-curl solve) leaves the harmonic form
        seed2 = _constant_seed(seq, 2)
        a = seq.L[1].solve(seq.G[1].T @ (seq.M[2] @ seed2), part="coexact", gauge=False,
                           tol=HARMONIC_TOL * seq.tol, accurate=True)
        operators = store(seq, 2, _plus_exact(seq, seed2, a, 2, subtract=True))

    # seq.free is taken anew after every store: a view copies the operators it sees
    if op.n_vectors(seq.betti_numbers, 0, False):
        operators = store(seq.free, 0, jnp.ones(seq.free.n(0)))

    if op.n_vectors(seq.betti_numbers, 1, False):
        # curl-free since the seed is closed, weakly divergence-free by the Leray projection
        free = seq.free
        seed1 = _constant_seed(free, 1)
        q = free.L[0].solve(-(free.D[0].T @ seed1), accurate=True)
        operators = store(free, 1, _plus_exact(free, seed1, q, 1))

    if verbose:
        for space, k in ((seq, 3), (seq, 2), (seq.free, 0), (seq.free, 1)):
            forms = op._harmonic(space, k)
            for i in range(forms.shape[0]):
                print(f"[nullspace] k={k} {'dbc' if space.dirichlet else 'free'} "
                      f"form {i}: v^T L v / v^T M v = {harmonic_rayleigh(space, forms[i], k):.2e}", flush=True)

    if seq.half_period and seq.parity is None:
        for view in (seq.odd, seq.even):
            _reduce_to_view(seq, view, verbose)
    return operators


def _reduce_to_view(seq, view, verbose):
    """Install on a parity view the forms of its parity (the odd view holds the 1- and 2-forms, the even
    view the constants), reduced from the full sequence's forms by the view's expansion ``X``."""
    # X^T v is exact for a field of the view's parity. Building the forms on the view itself is not: a
    # vector of ones is not the constant there, since a mirror pair of DoFs carries the weight 1 / sqrt 2
    # and a DoF the reflection maps to itself the weight 1.
    operators = _commit(view, op.init_nullspaces(view, view._require_operators()))
    for k, dirichlet in ((3, True), (2, True), (0, False), (1, False)):
        if not op.n_vectors(view.betti_numbers, k, dirichlet):
            continue
        base, space = (seq, view) if dirichlet else (seq.free, view.free)
        v = _reduced_form(space, view.reduction[(k, dirichlet)], op._harmonic(base, k), k)
        operators = _commit(view, op.set_nullspace(operators, k, dirichlet, v[None, :]))
        if verbose:
            print(f"[nullspace] {'odd' if view.parity == -1 else 'even'} view k={k} "
                  f"{'dbc' if dirichlet else 'free'}: v^T L v / v^T M v = "
                  f"{harmonic_rayleigh(view if dirichlet else view.free, v, k):.2e}", flush=True)
