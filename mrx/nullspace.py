"""Harmonic nullspaces of the discrete Hodge Laplacians.

The nullspace DoF vectors are stored on the dynamic ``SequenceOperators``
pytree (one stacked array per ``(k, dirichlet)`` pair). Their **shapes** are
topology-determined (from the Betti numbers passed to ``DeRhamSequence``) so
they are fixed across JAX traces, while the actual DoFs are dynamic and may
change when the geometry is updated.
"""

import equinox as eqx
import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Shape helpers
# ---------------------------------------------------------------------------


def _n_vectors(betti_numbers, k, dirichlet):
    """Number of harmonic ``k``-forms for the given Betti numbers."""
    b0, b1, b2, _b3 = betti_numbers
    if dirichlet:
        return (0, b2, b1, b0)[k]
    return (b0, b1, b2, 0)[k]


def _dof_count(seq, k, dirichlet):
    return getattr(seq, f"n{k}_dbc" if dirichlet else f"n{k}")


def _null_field(k, dirichlet):
    return f"null_{k}_dbc" if dirichlet else f"null_{k}"


# ---------------------------------------------------------------------------
# Initialisation and accessors
# ---------------------------------------------------------------------------

def init_nullspaces(seq, operators, betti_numbers=None):
    """Return ``operators`` with all eight nullspace arrays set to zeros.

    Shapes are derived from ``betti_numbers`` (or ``seq.betti_numbers`` when
    that argument is ``None``) and from the sequence's DoF counts. The DoFs
    are set to zero so that until the vectors are filled in, deflation is a
    no-op (projecting against a zero vector does nothing).
    """
    if betti_numbers is None:
        betti_numbers = seq.betti_numbers

    replacements = {}
    for k in range(4):
        for dirichlet in (False, True):
            n_vec = _n_vectors(betti_numbers, k, dirichlet)
            n_dof = _dof_count(seq, k, dirichlet)
            replacements[_null_field(k, dirichlet)] = jnp.zeros((n_vec, n_dof))

    return eqx.tree_at(
        lambda ops: tuple(getattr(ops, name) for name in replacements),
        operators,
        tuple(replacements.values()),
        is_leaf=lambda x: x is None,
    )


def get_nullspace(operators, k, dirichlet):
    """Return the stacked nullspace array for the k-th Hodge Laplacian.

    Returns an array of shape ``(n_vectors, n_k)``. Iterating over it yields
    the individual nullspace vectors.
    """
    vs = getattr(operators, _null_field(k, dirichlet))
    if vs is None:
        raise ValueError(
            f"Nullspace for k={k}, dirichlet={dirichlet} is not initialised. "
            "Call init_nullspaces(seq, operators) or one of the "
            "compute_nullspaces* functions first.")
    return vs


def get_saddle_point_nullspaces(seq, operators, k, dirichlet):
    """Nullspace vectors for the saddle-point system.

    If ``v`` lies in ``ker(S_k + D_{k-1} M_{k-1}^{-1} D_{k-1}^T)``, then
    ``[v, M_{k-1}^{-1} D_{k-1}^T v]`` lies in the nullspace of the full
    saddle-point matrix. Returned as two stacked arrays.
    """
    vs_upper = get_nullspace(operators, k, dirichlet)
    if k == 0 or vs_upper.shape[0] == 0:
        n_lower = _dof_count(seq, k - 1, dirichlet) if k >= 1 else 0
        return vs_upper, jnp.zeros((vs_upper.shape[0], n_lower))

    def _lower(v):
        Dt_v = seq.apply_derivative_matrix(
            v, k - 1,
            dirichlet_in=dirichlet, dirichlet_out=dirichlet,
            transpose=True,
            operators=operators,
        )
        return seq.apply_inverse_mass_matrix(
            Dt_v, k - 1, dirichlet=dirichlet, operators=operators)

    vs_lower = jnp.stack([_lower(v) for v in vs_upper])
    return vs_upper, vs_lower


def _set_null(operators, k, dirichlet, values):
    """Return ``operators`` with a single nullspace field replaced."""
    name = _null_field(k, dirichlet)
    return eqx.tree_at(
        lambda ops: getattr(ops, name),
        operators,
        values,
        is_leaf=lambda x: x is None,
    )


def _overwrite_nullspace_vector(operators, k, dirichlet, idx, value):
    """Return ``operators`` with one stored nullspace vector overwritten."""
    values = get_nullspace(operators, k, dirichlet)
    return _set_null(operators, k, dirichlet, values.at[idx].set(value))


def _commit(seq, operators):
    """Set ``seq.operators`` to ``operators`` so fallback lookups see the
    latest null fields, and return the bundle unchanged.
    """
    seq.operators = operators
    return operators


def _bootstrap_nullspace_guesses(seq, operators, k, dirichlet, guesses):
    """Store normalised bootstrap guesses in the nullspace field for ``(k, dirichlet)``.

    This lets shifted preconditioners read a stable coarse vector from the
    operator bundle while inverse iteration is still constructing the true
    nullspace.
    """
    n_vec = len(guesses)
    n_dof = _dof_count(seq, k, dirichlet)
    values = jnp.zeros((n_vec, n_dof))
    stored = []

    for idx, guess in enumerate(guesses):
        if guess is None:
            continue
        work = guess
        for u in stored:
            work = work - (u @ seq.apply_mass_matrix(
                work, k, dirichlet=dirichlet, operators=operators)) * u
        norm = seq.l2_norm(work, k, dirichlet=dirichlet)
        if float(norm) <= 0.0:
            continue
        work = work / norm
        stored.append(work)
        values = values.at[idx].set(work)

    return _commit(seq, _set_null(operators, k, dirichlet, values))


# ---------------------------------------------------------------------------
# Closed-form nullspace construction (contractible domain)
# ---------------------------------------------------------------------------

def compute_nullspaces(seq, operators=None):
    """Closed-form harmonic forms for a contractible domain.

    Assumes ``betti = (1, 0, 0, 0)``. For more general topology use
    :func:`compute_nullspaces_iterative`.

    Returns the updated ``SequenceOperators`` bundle with the eight
    ``null_*`` fields populated.
    """
    if operators is None:
        operators = seq._require_operators()

    operators = _commit(seq, init_nullspaces(
        seq, operators, betti_numbers=(1, 0, 0, 0)))

    # k = 3, Dirichlet: lift the constant 1-vector via M^{-1}.
    v3 = seq.apply_inverse_mass_matrix(
        jnp.ones(seq.n3_dbc), 3, dirichlet=True, operators=operators)
    v3 = v3 / seq.l2_norm(v3, 3, dirichlet=True)
    operators = _commit(seq, _set_null(operators, 3, True, v3[None, :]))

    # k = 2, Dirichlet: Leray-project 1, then subtract its curl contribution.
    v, _ = seq.apply_leray_projection(jnp.ones(seq.n2_dbc), k=2)
    curl_v_dual = seq.apply_derivative_matrix(
        v, 1, dirichlet_in=True, dirichlet_out=True, transpose=True,
        operators=operators)
    a = seq.apply_inverse_hodge_laplacian(
        curl_v_dual, 1, dirichlet=True, operators=operators)
    curl_a = seq.apply_strong_curl(a, True, True)
    v2 = v - curl_a
    v2 = v2 / seq.l2_norm(v2, 2, dirichlet=True)
    operators = _commit(seq, _set_null(operators, 2, True, v2[None, :]))

    # k = 0, no Dirichlet BC: the constant function.
    v0 = jnp.ones(seq.n0)
    v0 = v0 / seq.l2_norm(v0, 0, dirichlet=False)
    operators = _commit(seq, _set_null(operators, 0, False, v0[None, :]))

    # k = 1, no Dirichlet BC: Leray-project 1, subtract its curl contribution.
    v, _ = seq.apply_leray_projection(jnp.ones(seq.n1), k=1)
    curl_v_dual = seq.apply_derivative_matrix(
        v, 1, dirichlet_in=False, dirichlet_out=False, operators=operators)
    a = seq.apply_inverse_hodge_laplacian(
        curl_v_dual, 2, dirichlet=False, operators=operators)
    curl_a = seq.apply_weak_curl(a, False, False)
    v1 = v - curl_a
    v1 = v1 / seq.l2_norm(v1, 1, dirichlet=False)
    operators = _commit(seq, _set_null(operators, 1, False, v1[None, :]))

    return operators


# ---------------------------------------------------------------------------
# Iterative nullspace construction (arbitrary topology)
# ---------------------------------------------------------------------------

def _toroidal_vacuum_field(seq):
    """Return ``B(x) = (1/R) e_zeta`` in Cartesian physical coordinates.

    ``x`` is the logical coordinate. ``e_zeta`` is the unit tangent along the
    third logical coordinate direction, i.e. the normalized third column of
    ``DF(x)``. ``R = sqrt(X^2 + Y^2)`` uses physical coordinates
    ``F(x) = (X, Y, Z)``.

    Used as the analytic initial guess for the harmonic 1-form (no BC) and
    2-form (DBC) on toroidal geometries.
    """
    DF = jax.jacfwd(seq.map)

    def B(x_hat):
        x, y, _ = seq.map(x_hat)
        dF = DF(x_hat)
        dzeta = dF[:, 2]
        dzeta_norm = jnp.linalg.norm(dzeta)
        R = jnp.sqrt(x**2 + y**2)
        return dzeta / (dzeta_norm * R)

    return B


def _initial_guesses(seq, operators, k, dirichlet, n_vec):
    """Return a length-``n_vec`` list of analytic initial guesses (or ``None``).

    Cases covered (the four non-trivial harmonic spaces on a
    ``betti = (1, 1, 0, 0)`` solid torus):

    * ``k = 0, no DBC``: the constant scalar field ``1``.
    * ``k = 1, no DBC``: ``1/R * e_zeta`` projected to a 1-form.
    * ``k = 2, DBC``  : ``1/R * e_zeta`` projected to a 2-form.
    * ``k = 3, DBC``  : the constant, lifted via ``M_3^{-1}``.

    Any remaining slots are ``None`` (fall back to the random init).
    """
    if n_vec == 0:
        return []
    guesses = [None] * n_vec
    if k == 0 and not dirichlet:
        guesses[0] = jnp.ones(seq.n0)
    elif k == 3 and dirichlet:
        guesses[0] = seq.apply_inverse_mass_matrix(
            jnp.ones(seq.n3_dbc), 3, dirichlet=True, operators=operators)
    elif k == 1 and not dirichlet:
        guesses[0] = seq.apply_inverse_mass_matrix(
            seq.p1(_toroidal_vacuum_field(seq)), 1, dirichlet=False, operators=operators)
    elif k == 2 and dirichlet:
        guesses[0] = seq.apply_inverse_mass_matrix(
            seq.p2_dbc(_toroidal_vacuum_field(seq)), 2, dirichlet=True, operators=operators)
    return guesses


def compute_nullspaces_iterative(seq, operators=None, betti_numbers=None,
                                 eps=1e-6, abs_tol=None):
    """Compute harmonic forms via shift-and-invert iteration.

    Each ``(k, dirichlet)`` pair with a non-zero harmonic dimension is
    seeded with an analytic initial guess when available (see
    :func:`_initial_guesses`). If that guess already satisfies
    ``||L_k v|| <= abs_tol`` we accept it directly without running inverse
    iteration. Otherwise the guess is used as the starting point for
    inverse iteration, which also terminates on ``||L_k v|| <= abs_tol``.

    Parameters
    ----------
    seq : DeRhamSequence
    operators : SequenceOperators, optional
        Bundle to update. Defaults to ``seq._require_operators()``.
    betti_numbers : tuple of 4 ints, optional
        ``(b0, b1, b2, b3)``. Defaults to ``seq.betti_numbers``. Must have
        ``b0 == 1`` and ``b3 == 0``.
    eps : float
        Shift used to regularise the stiffness block.
    abs_tol : float, optional
        Absolute tolerance on the Hodge-Laplacian residual ``||L_k v||``.
        Defaults to ``seq.tol``.

    Returns
    -------
    operators : SequenceOperators
        Updated bundle with the eight ``null_*`` fields populated.
    info : dict
        Per ``(k, dirichlet)`` key: a list of ``(n_iters, residual)`` tuples,
        one per converged eigenvector, where ``residual = ||L_k v||``.
        ``n_iters == 0`` indicates the initial guess was accepted without
        iteration.
    """
    if operators is None:
        operators = seq._require_operators()
    if betti_numbers is None:
        betti_numbers = seq.betti_numbers
    if abs_tol is None:
        abs_tol = seq.tol
    assert len(betti_numbers) == 4, "betti_numbers must have length 4"
    assert betti_numbers[0] == 1, "betti_numbers[0] must be 1"
    assert betti_numbers[3] == 0, "betti_numbers[3] must be 0"

    operators = _commit(seq, init_nullspaces(
        seq, operators, betti_numbers=betti_numbers))
    info = {}

    for k in range(4):
        for dirichlet in (False, True):
            n_vec = _n_vectors(betti_numbers, k, dirichlet)
            if n_vec == 0:
                continue
            x0s = _initial_guesses(seq, operators, k, dirichlet, n_vec)
            operators = _bootstrap_nullspace_guesses(
                seq, operators, k, dirichlet, x0s)
            vs, iters = find_nullspace_vectors(
                seq, operators, k, n_vec, eps, dirichlet=dirichlet,
                x0s=x0s, abs_tol=abs_tol)
            operators = _commit(seq, _set_null(operators, k, dirichlet, vs))
            info[(k, dirichlet)] = iters

    return operators, info


def find_nullspace_vectors(seq, operators, k, n_vectors, eps, dirichlet=True,
                           x0s=None, abs_tol=None):
    """Find ``n_vectors`` harmonic ``k``-forms via inverse iteration.

    Each vector is found by repeatedly applying ``(S_k + eps M_k)^{-1} M_k``
    with M-orthogonalisation against the previously found vectors. Uses
    ``jax.lax.while_loop`` so the inner iteration is JIT-compatible.

    Parameters
    ----------
    x0s : list of optional arrays, length ``n_vectors``
        Per-vector initial guesses. Entries that are ``None`` fall back to
        a deterministic random initialisation.
    abs_tol : float
        Absolute tolerance on the residual ``||L_k v||``. If the normalised
        initial guess already satisfies it (after M-orthogonalisation
        against previously-found vectors), it is accepted directly and no
        inverse iteration is run for that slot. The inner iteration also
        terminates once ``||L_k v||`` falls below ``abs_tol`` or stalls.

    Returns
    -------
    vs : jnp.ndarray
        Stacked array of shape ``(n_vectors, n_k)``. Empty shape
        ``(0, n_k)`` when ``n_vectors == 0``.
    iters : list of (int, float)
        ``(n_iters, final_residual_norm)`` per vector, where the residual
        norm is ``||L_k v||``. ``n_iters == 0`` means the initial guess
        was accepted without iteration.
    """
    if abs_tol is None:
        abs_tol = seq.tol
    n = _dof_count(seq, k, dirichlet)
    if n_vectors == 0:
        return jnp.zeros((0, n)), []

    found = []
    iters = []

    for idx in range(n_vectors):
        if x0s is not None and x0s[idx] is not None:
            v0 = x0s[idx]
        else:
            v0 = jax.random.normal(jax.random.PRNGKey(idx), (n,))
        # M-orthogonalise against already-found vectors.
        for u in found:
            v0 = v0 - (u @ seq.apply_mass_matrix(
                v0, k, dirichlet=dirichlet, operators=operators)) * u
        v0 = v0 / seq.l2_norm(v0, k, dirichlet=dirichlet)

        # Early exit if the initial guess is already harmonic to tolerance.
        # Metric: ``||L v||`` (residual norm of the Hodge Laplacian). For
        # k = 3 the stiffness block is zero, so the Rayleigh quotient
        # ``v^T S v`` gives 0 regardless -- we must use the full L.
        Lv0 = seq.apply_hodge_laplacian(
            v0, k, dirichlet=dirichlet, operators=operators)
        res_init = float(seq.l2_norm(Lv0, k, dirichlet=dirichlet))
        if res_init <= abs_tol:
            found.append(v0)
            iters.append((0, res_init))
            continue

        def body_fn(state):
            v, res, _, i = state
            Mv = seq.apply_mass_matrix(
                v, k, dirichlet=dirichlet, operators=operators)
            w = seq.apply_inverse_shifted_hodge_laplacian(
                Mv, k, eps, dirichlet=dirichlet, guess=v,
                operators=operators)
            for u in found:
                w = w - (u @ seq.apply_mass_matrix(
                    w, k, dirichlet=dirichlet, operators=operators)) * u
            w = w / seq.l2_norm(w, k, dirichlet=dirichlet)
            Lw = seq.apply_hodge_laplacian(
                w, k, dirichlet=dirichlet, operators=operators)
            res_new = seq.l2_norm(Lw, k, dirichlet=dirichlet)
            return w, res_new, res, i + 1

        def cond_fn(state):
            _, res, res_prev, i = state
            converged = (res <= abs_tol) | (
                jnp.abs(res - res_prev) <= abs_tol)
            return ~converged & (i < seq.maxiter)

        init_state = (v0, jnp.inf, jnp.inf, 0)
        v_final, res_final, _, n_iters = jax.lax.while_loop(
            cond_fn, body_fn, init_state)
        found.append(v_final)
        iters.append((int(n_iters), float(res_final)))

    return jnp.stack(found), iters
