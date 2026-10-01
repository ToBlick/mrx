"""The harmonic forms of a solid torus, the null spaces of the discrete Hodge Laplacians ``L_k``.

The Laplacian and Leray solves on a sequence use these forms to remove the null space of ``L_k``.
Run :func:`compute_nullspaces` once the preconditioners are built. The forms depend on the geometry, so
installing a new map drops them together with the preconditioners, and both must be built again.
:func:`harmonic_rayleigh` checks how harmonic a given form is.
"""

import jax.numpy as jnp

import mrx
import mrx.operators as op
from mrx.precision import RESIDUAL_DTYPE


def _builder(seq):
    """The sequence the forms are built and checked on, which is its float64 view when it has one."""
    return seq.residual if seq.residual is not None else seq


def _commit(seq, operators):
    """Install ``operators`` on ``seq`` and its float64 view and return it. The next solve then removes
    the forms stored so far."""
    seq.operators = operators
    if seq.residual is not None:
        seq.residual.operators = operators
    return operators


def harmonic_rayleigh(seq, v, k):
    """The Rayleigh quotient ``v^T L_k v / v^T M_k v`` (``M_k`` the mass matrix) of a ``k``-form ``v``
    in the space of ``seq``. It is zero for a harmonic form and of the order of the smallest nonzero
    eigenvalue otherwise.

    It is evaluated in the residual precision, since ``L_k v`` of a nearly harmonic ``v`` cancels two
    large terms. It takes a mass solve, so it is meant as a diagnostic.
    """
    on = _builder(seq)
    v = jnp.asarray(v).astype(RESIDUAL_DTYPE)
    lv = op.laplacian_with(on, v, k, lambda w, j: seq.M[j].solve(w, dtype=RESIDUAL_DTYPE))
    mv = on.M[k] @ v
    return float(jnp.dot(v, lv) / jnp.dot(v, mv))


def _closed_seed(build, components, k):
    """The form with constant logical components ``components``, checked to be closed up to round-off."""
    components = jnp.asarray(components, dtype=mrx.DTYPE)
    seed = build.interpolate(lambda x_hat: components, k, frame='logical')
    closed = float(build.l2_norm(build.G[k] @ seed, k + 1) / build.l2_norm(seed, k))
    if closed > mrx.sqrt_eps():
        raise RuntimeError(
            f"compute_nullspaces: the k={k} seed is not closed (|G seed| / |seed| = {closed:.2e})")
    return seed


def compute_nullspaces(seq, *, verbose=True):
    """Compute the harmonic forms of ``seq``, install them on ``seq.operators`` and return the operators.

    Call it after :meth:`~mrx.derham_sequence.DeRhamSequence.build_preconditioners`. The domain is a solid
    torus (Betti numbers ``(1, 1, 0, 0)``, of which the odd parity view keeps ``b1`` and the even view
    ``b0``). The forms are, in order:

    - k = 3 with Dirichlet conditions: ``M_3^{-1}`` applied to the constant.
    - k = 2 with Dirichlet conditions: the toroidal flux form ``dr ^ dtheta`` minus its exact part, found
      with one ``L_1`` solve.
    - k = 0 without boundary conditions: the constant.
    - k = 1 without boundary conditions: the Leray projection of ``d zeta``, found with one ``L_0`` solve.

    In the mixed precision configuration they are computed in float64 and stored in the working dtype.
    The float64 view and the parity views of ``seq`` get their forms as well. With ``verbose`` the
    :func:`harmonic_rayleigh` of each form is printed.
    """
    operators = _commit(seq, op.init_nullspaces(seq, seq._require_operators()))
    build = _builder(seq)

    def store(space, k, v):
        v = v / space.l2_norm(v, k)
        return _commit(seq, op.set_nullspace(operators, k, space.dirichlet, v.astype(mrx.DTYPE)[None, :]))

    if op.n_vectors(seq.betti_numbers, 3, True):
        operators = store(build, 3, build.M[3].solve(jnp.ones(build.n(3), dtype=mrx.DTYPE)))

    if op.n_vectors(seq.betti_numbers, 2, True):
        # the closed seed has no coexact part, so removing its exact part (the image of curl, one L_1
        # Dirichlet solve) leaves the harmonic form
        seed2 = _closed_seed(build, (0.0, 0.0, 1.0), 2)
        a = build.L[1].solve(build.D[1].T @ seed2)
        operators = store(build, 2, seed2 - build.G[1] @ a)

    # build.free is taken anew after every store: a view copies the operators it sees
    if op.n_vectors(seq.betti_numbers, 0, False):
        operators = store(build.free, 0, jnp.ones(build.free.n(0), dtype=mrx.DTYPE))

    if op.n_vectors(seq.betti_numbers, 1, False):
        # curl-free since the seed is closed, weakly divergence-free by the projection
        v1, _ = build.free.leray(_closed_seed(build.free, (0.0, 0.0, 1.0), 1), k=1)
        operators = store(build.free, 1, v1)

    if verbose:
        for space, k in ((seq, 3), (seq, 2), (seq.free, 0), (seq.free, 1)):
            for i, v in enumerate(space.nullspace(k)):
                print(f"[nullspace] k={k} {'dbc' if space.dirichlet else 'free'} "
                      f"form {i}: v^T L v / v^T M v = {harmonic_rayleigh(space, v, k):.2e}", flush=True)

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
        v = view.reduction[(k, dirichlet)].T @ base.nullspace(k)[0]
        v = v / space.l2_norm(v, k)
        operators = _commit(view, op.set_nullspace(operators, k, dirichlet, v.astype(mrx.DTYPE)[None, :]))
        if verbose:
            print(f"[nullspace] {'odd' if view.parity == -1 else 'even'} view k={k} "
                  f"{'dbc' if dirichlet else 'free'}: v^T L v / v^T M v = "
                  f"{harmonic_rayleigh(view if dirichlet else view.free, v, k):.2e}", flush=True)
