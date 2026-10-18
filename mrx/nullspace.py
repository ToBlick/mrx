"""The harmonic forms of a solid torus, the null spaces of the discrete Hodge Laplacians ``L_k``.

The Laplacian and Leray solves on a sequence use these forms to remove the null space of ``L_k``.
Run :func:`compute_nullspaces` once the preconditioners are built. The forms depend on the geometry, so
installing a new map drops them together with the preconditioners, and both must be built again.
:func:`harmonic_rayleigh` checks how harmonic a given form is.
"""

import jax.numpy as jnp

import mrx
import mrx.operators as op


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
    v = jnp.asarray(v)
    lv = seq.L[k] @ v
    mv = seq.M[k] @ v
    return float(jnp.dot(v, lv) / jnp.dot(v, mv))


def _closed_seed(seq, components, k):
    """The form with constant logical components ``components``, checked to be closed up to round-off."""
    components = jnp.asarray(components)
    seed = seq.interpolate(lambda x_hat: components, k)
    closed = float(seq.l2_norm(seq.G[k] @ seed, k + 1) / seq.l2_norm(seed, k))
    if closed > mrx.sqrt_eps():
        raise RuntimeError(
            f"compute_nullspaces: the k={k} seed is not closed (|G seed| / |seed| = {closed:.2e})")
    return seed


def _constant(space, k):
    """The constant ``k``-form (k = 0, 3) of ``space``: the vector of ones, reduced on a parity view, where a
    mirror pair of DoFs carries the weight ``1 / sqrt 2``."""
    if space.parity is None:
        return jnp.ones(space.n(k))
    X = space.reduction[(k, space.dirichlet)]
    return X.T @ jnp.ones(X.forward_shape[0])


def compute_nullspaces(seq, *, verbose=True):
    """Compute the harmonic forms of ``seq``, install them on its operators and return ``seq.operators``.

    Call it after :meth:`~mrx.derham_sequence.DeRhamSequence.build_preconditioners`. The domain is a solid
    torus (Betti numbers ``(1, 1, 0, 0)``). On a half-period sequence the forms are computed on the parity
    views, which hold the fields: the odd view the 1- and 2-forms, the even view the constants. The full
    half-period sequence is not solved on and keeps zero forms. The forms are:

    - k = 3 with Dirichlet conditions: ``M_3^{-1}`` applied to the constant.
    - k = 2 with Dirichlet conditions: the toroidal flux form ``dr ^ dtheta`` minus its exact part, found
      with one curl-curl solve.
    - k = 0 without boundary conditions: the constant.
    - k = 1 without boundary conditions: the Leray projection of ``d zeta``, found with one ``L_0`` solve.

    With ``verbose`` the :func:`harmonic_rayleigh` of each form is printed.
    """
    for space in ((seq.odd, seq.even) if seq.half_period and seq.parity is None else (seq,)):
        _compute(space, verbose)
    return seq.operators


def _compute(space, verbose):
    """The harmonic forms of ``space``, a full-period sequence or a parity view, installed on its operators."""
    _commit(space, op.init_nullspaces(space, space._require_operators()))

    def store(sp, k, v):
        # sp.free is taken anew after every store: a view copies the operators it sees
        v = v / sp.l2_norm(v, k)
        _commit(space, op.set_nullspace(space.operators, k, sp.dirichlet, v[None, :]))
        if verbose:
            name = "" if space.parity is None else ("odd " if space.parity == -1 else "even ")
            print(f"[nullspace] {name}k={k} {'dbc' if sp.dirichlet else 'free'}: v^T L v / v^T M v = "
                  f"{harmonic_rayleigh(sp, v, k):.2e}", flush=True)

    betti = space.betti_numbers
    if op.n_vectors(betti, 3, True):
        store(space, 3, space.M[3].solve(_constant(space, 3)))
    if op.n_vectors(betti, 2, True):
        # the closed seed has no coexact part, so removing its exact part (the image of curl, one
        # curl-curl solve) leaves the harmonic form
        seed2 = _closed_seed(space, (0.0, 0.0, 1.0), 2)
        a = space.L[1].solve(space.G[1].T @ (space.M[2] @ seed2), part="coexact", gauge=False,
                             tol=HARMONIC_TOL * space.tol)
        store(space, 2, seed2 - space.G[1] @ a)
    if op.n_vectors(betti, 0, False):
        store(space.free, 0, _constant(space.free, 0))
    if op.n_vectors(betti, 1, False):
        # curl-free since the seed is closed, weakly divergence-free by the projection
        v1, _ = space.free.leray(_closed_seed(space.free, (0.0, 0.0, 1.0), 1), k=1)
        store(space.free, 1, v1)
