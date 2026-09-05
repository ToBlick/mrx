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

import mrx
from mrx.solvers import deflation_projectors

# ---------------------------------------------------------------------------
# Shape helpers
# ---------------------------------------------------------------------------


def _n_vectors(betti_numbers, k, dirichlet):
    """Number of harmonic ``k``-forms for the given Betti numbers.

    ``betti_numbers`` are the ABSOLUTE Betti numbers, which is what the free
    (natural-BC) branch reads directly. The Dirichlet branch needs the RELATIVE
    ones, and they are not the same list -- they are the reversed one::

        b_k^rel = b_{3-k}^abs

    which is what the ``(0, b2, b1, b0)[k]`` reversal below is. Poincare-Lefschetz
    duality on a 3-manifold with boundary; nothing in any call signature says so.

    WORKED, because the default matters and the reversal is easy to miss.
    ``betti_numbers=(1, 1, 0, 0)`` -- the DeRhamSequence default -- gives

        free (absolute)   k=0: 1   k=1: 1   k=2: 0   k=3: 0
        dbc  (relative)   k=0: 0   k=1: 0   k=2: 1   k=3: 1

    so harmonic forms exist at exactly ``(0, free)``, ``(1, free)``,
    ``(2, dbc)`` and ``(3, dbc)`` -- and NOT at ``(1, dbc)`` or ``(2, free)``.

    THIS HAS COST TWO PEOPLE ONE DAY (2026-08-25), from opposite directions:
    once by reading absolute ``(1,1,0,0)`` as "no harmonic 2-forms anywhere"
    when ``compute_helicity`` works in the Dirichlet complex where there IS
    one; and once by A/B-ing a preconditioner on ``(1, dbc)`` and ``(2, free)``,
    where BOTH arms returned the same non-harmonic vector because there was no
    harmonic form to find -- which reads as reassuring agreement rather than as
    a measurement of nothing.

    If you are choosing ``(k, dirichlet)`` cells to test, check this function
    first. A cell with zero harmonic forms still RETURNS a vector.
    """
    b0, b1, b2, _b3 = betti_numbers
    if dirichlet:
        return (0, b2, b1, b0)[k]
    return (b0, b1, b2, 0)[k]


def _dof_count(seq, k, dirichlet):
    return seq.n(k, dirichlet)


# ---------------------------------------------------------------------------
# Initialisation and accessors
# ---------------------------------------------------------------------------

def init_nullspaces(seq, operators, betti_numbers=None):
    """Return ``operators`` with every nullspace set to zeros.

    ``operators.nullspaces[(k, dirichlet)]`` is ``(n_vectors, n_k)`` with the
    vector count from ``betti_numbers`` (or ``seq.betti_numbers``). Zeros mean
    deflation is a no-op until the vectors are computed.
    """
    if betti_numbers is None:
        betti_numbers = seq.betti_numbers
    spaces = {(k, dirichlet): jnp.zeros((_n_vectors(betti_numbers, k, dirichlet),
                                         _dof_count(seq, k, dirichlet)), dtype=mrx.DTYPE)
              for k in range(4) for dirichlet in (False, True)}
    return eqx.tree_at(lambda ops: ops.nullspaces, operators, spaces,
                       is_leaf=lambda x: x is None or isinstance(x, dict))


def get_nullspace(operators, k, dirichlet):
    """The stacked nullspace array ``(n_vectors, n_k)`` of the k-th Hodge Laplacian."""
    try:
        return operators.nullspaces[(int(k), bool(dirichlet))]
    except (KeyError, TypeError):
        raise ValueError(
            f"nullspace for k={k}, dirichlet={dirichlet} is not initialised; "
            "seq.build_preconditioners() creates the bundle with zero "
            "nullspaces, compute_nullspaces fills them") from None


def _set_null(operators, k, dirichlet, values):
    """Return ``operators`` with the nullspace of ``(k, dirichlet)`` replaced."""
    spaces = dict(operators.nullspaces)
    spaces[(int(k), bool(dirichlet))] = values
    return eqx.tree_at(lambda ops: ops.nullspaces, operators, spaces,
                       is_leaf=lambda x: isinstance(x, dict))


def _commit(seq, operators):
    """Set ``seq.operators`` to ``operators`` so fallback lookups see the
    latest null fields, and return the bundle unchanged.
    """
    seq.operators = operators
    return operators


# ---------------------------------------------------------------------------
# Direct (Hodge-decomposition) nullspace construction
# ---------------------------------------------------------------------------

def direct_construction_unsupported_reason(betti_numbers):
    """Why the direct route cannot run on this topology, or ``None`` if it can.

    The direct construction obtains each harmonic form by stripping the exact
    and coexact parts off a seed, which costs two Hodge solves in neighbouring
    degrees.  Those solves are themselves singular whenever the neighbouring
    harmonic space is non-trivial, so they need deflation vectors -- and the
    route is only self-sufficient when every kernel it needs has already been
    built by an earlier step.

    Walking the construction order (k=3 DBC, k=2 DBC, k=0 NBC, k=1 NBC);
    every seed is exactly closed, so there is no coexact part to strip and
    each form costs one solve:

    ==================  ==============================
    construction        the one solve, and its kernel
    ==================  ==============================
    k=3 DBC             closed form (no solve)
    k=2 DBC             ``L_1`` DBC, kernel ``b2``
    k=0 NBC             closed form (no solve)
    k=1 NBC             ``L_0`` NBC, kernel ``b0``
    ==================  ==============================

    The ``b0`` kernel is the constants, produced in closed form by the step
    immediately before.  The ``b2`` kernel (harmonic 1-forms of the
    Dirichlet complex) is not produced at all, so ``b2 > 0`` breaks the
    route -- circularly: building it would need ``L_2`` DBC, whose kernel is
    ``b1``, the harmonic 2-form under construction.  No ordering fixes it.

    The k=3 Hodge-split solve (:func:`mrx.operators.apply_inverse_laplacian_hodge`)
    deflates against the k=2 DBC form, so nothing here may call it before
    that form is stored -- which is why the k=2 seed is closed by
    construction rather than Leray-projected (a k=3 solve).
    """
    b0, b1, b2, b3 = (int(b) for b in betti_numbers)
    if b0 != 1:
        return f"b0 = {b0}, expected 1 (connected domain)"
    if b3 != 0:
        return f"b3 = {b3}, expected 0"
    if b2 != 0:
        return (
            f"b2 = {b2} > 0 (betti = {(b0, b1, b2, b3)}). The direct route "
            "strips the exact part of the k=2 seed by inverting L_1 with "
            "Dirichlet BCs, whose kernel has dimension b2 and is not "
            "available. Building it the same way needs L_2 (DBC), whose "
            "kernel is the very form under construction -- the dependency is "
            "circular, so that Krylov solve cannot be given the deflation "
            "vectors it needs. Use the iterative route (direct=False): its "
            "shift removes the singularity, so it needs no prior nullspace."
        )
    return None


def harmonic_rayleigh(seq, v, k, dirichlet=True, operators=None):
    """``v^T L_k v / v^T M_k v`` -- how far ``v`` is from being harmonic.

    Zero for a true harmonic form, ``O(lambda_1)`` for anything else.  This is
    the number that tells you whether :func:`compute_nullspaces` succeeded: the
    construction is a chain of Hodge solves with a fixed iteration budget and no
    gate of its own, so a solve that runs out of iterations returns a
    non-harmonic vector, every deflated solve downstream deflates against it,
    and nothing says a word.

    ``L_k`` is applied EXACTLY (nested mass solve).  That shape is banned inside
    a Krylov solve; a diagnostic evaluated once per vector is the one place it
    is legitimate.

    Quote it against ``lambda_1`` from :func:`estimate_spectral_gap` -- the
    quotient is not dimensionless, so a raw value carries the units of the
    geometry and means nothing on its own; the ratio is the eigenvector error
    squared.  :func:`compute_nullspaces` prints both for every form it builds.
    """
    lv = seq.apply_laplacian(v, k, dirichlet=dirichlet,
                                   operators=operators)
    mv = seq.apply_mass_matrix(v, k, dirichlet=dirichlet)
    return float(jnp.dot(v, lv) / jnp.dot(v, mv))


def compute_nullspaces(seq, operators=None, betti_numbers=None, *,
                       gap_sweeps=0, verbose=True):
    """Harmonic forms by direct Hodge decomposition (no inverse iteration).

    Each form is built from an exactly closed seed (``d zeta`` at k = 1,
    ``dr^dchi`` at k = 2, both histopolated constants) by removing its exact
    part with ONE Hodge solve -- no coexact part to remove, no shift, no
    outer loop, and no dependence on a spectral gap.  Requires ``b2 == 0``;
    see
    :func:`direct_construction_unsupported_reason`.  For anything else use
    :func:`compute_nullspaces_iterative`.

    ``betti_numbers`` defaults to ``seq.betti_numbers``.  (It used to be
    hard-wired to ``(1, 0, 0, 0)``, which contradicted the function's own
    output: that tuple allocates *zero* rows for the k=1 NBC and k=2 DBC
    slots, and the code then wrote one row into each.  What it actually
    computes is the ``b1``-driven pair, so the Betti numbers must come from
    the sequence.)

    The construction has no gate of its own -- a Hodge solve that runs out of
    iterations returns a non-harmonic vector and nothing downstream says a
    word -- so every form is reported (``verbose``) with its Rayleigh
    quotient :func:`harmonic_rayleigh` and, for the two forms that are built
    by solves (k = 1 free, k = 2 Dirichlet), the first non-harmonic eigenvalue
    ``lambda_1`` from :func:`estimate_spectral_gap` in ``gap_sweeps`` sweeps
    of inverse iteration when asked (the default ``0`` skips it: it is a
    check for the console, ~17 s at W7-X (12,24,12) p=3 for 5 sweeps).
    The ratio of the two is the squared relative error of the form, i.e.
    ``O(seq.tol^2)`` when the solves converged -- measured 2e-4 / 4e-5 in
    float32 (tol 3.5e-4) on W7-X, 1e-14-ish in float64; ``1e-1`` is a solve
    that stopped early.

    Returns the updated ``SequenceOperators`` bundle.
    """
    if operators is None:
        operators = seq._require_operators()
    if betti_numbers is None:
        betti_numbers = seq.betti_numbers
    betti_numbers = tuple(int(b) for b in betti_numbers)

    reason = direct_construction_unsupported_reason(betti_numbers)
    if reason is not None:
        raise ValueError(
            "compute_nullspaces: the direct construction does not support "
            f"this topology: {reason}"
        )

    # The construction below is two Hodge-Laplacian solves -- L_1 DBC for the
    # k=2 form, L_0 FREE for the k=1 form -- and each takes the metric-lumped
    # atom, which is REQUIRED rather than consulted.
    #
    # This function does NOT build it. It used to, and that was a setup step
    # hidden inside a solve routine: the caller could not tell which
    # preconditioners existed, and a geometry change afterwards left them stale.
    # Build them explicitly, or in one call with
    # ``seq.set_map_and_preconditioners(map)``. A missing atom raises here with
    # a message naming the assembler.
    operators = _commit(seq, init_nullspaces(
        seq, operators, betti_numbers=betti_numbers))

    # Order is load-bearing: each solve below is deflated against a kernel
    # that a previous step has already stored (see the table in
    # direct_construction_unsupported_reason).
    if _n_vectors(betti_numbers, 3, True):
        # k = 3, Dirichlet: lift the constant 1-vector via M^{-1}.
        v3 = seq.apply_inverse_mass_matrix(
            jnp.ones(seq.n(3, True), dtype=mrx.DTYPE), 3, dirichlet=True, operators=operators)
        v3 = v3 / seq.l2_norm(v3, 3, dirichlet=True)
        operators = _commit(seq, _set_null(operators, 3, True, v3[None, :]))

    if _n_vectors(betti_numbers, 2, True):
        # k = 2, Dirichlet: the seed is the flux 2-form dr^dchi -- primal
        # proxy (0, 0, 1), histopolated (a direct solve), constant
        # coefficients, so its incidence divergence is zero to round-off.  A
        # closed seed has no coexact part, so the only work is removing the
        # im(curl) part with one L_1 DBC solve (k=1 Dirichlet has no harmonic
        # form: no dependency on anything stored later).  The seed is NOT the
        # harmonic form -- that is the Hodge star of d zeta, metric-weighted,
        # never in V^2 -- it only has to be closed; the solve does the metric.
        #
        # It used to Leray-project an M_2^{-1}-seed first (a k=3 solve).  The
        # k=3 Hodge-split solve deflates its L^_2 solves against THIS form,
        # which at that point is still zero, and returned a form with
        # |D h_2| / |h_2| ~ 2 on QA (co-closed, not closed).
        flux = jnp.asarray((0.0, 0.0, 1.0), dtype=mrx.DTYPE)
        seed2 = seq.interpolate(lambda x_hat: flux, 2, dirichlet=True, frame='ref')
        div_seed = seq.apply_incidence_matrix(
            seed2, 2, dirichlet_in=True, dirichlet_out=True)
        closed = float(seq.l2_norm(div_seed, 3, dirichlet=True)
                       / seq.l2_norm(seed2, 2, dirichlet=True))
        # A histopolated seed is closed to the round-off of its working-
        # precision incidence apply, not to the solves' residual tolerance.
        if closed > mrx.sqrt_eps():
            raise RuntimeError(
                "compute_nullspaces: the dr^dchi seed is not closed "
                f"(|D seed| / |seed| = {closed:.2e} > sqrt(eps) {mrx.sqrt_eps():.1e}); the "
                "k=2 Dirichlet harmonic form would carry that divergence")
        curl_dual = seq.apply_derivative_matrix(
            seed2, 1, dirichlet_in=True, dirichlet_out=True, transpose=True)
        a = seq.apply_inverse_laplacian(
            curl_dual, 1, dirichlet=True, operators=operators)
        v2 = seed2 - seq.apply_strong_curl(a, True, True)
        v2 = v2 / seq.l2_norm(v2, 2, dirichlet=True)
        operators = _commit(seq, _set_null(operators, 2, True, v2[None, :]))

    if _n_vectors(betti_numbers, 0, False):
        # k = 0, no Dirichlet BC: the constant function.
        v0 = jnp.ones(seq.n(0), dtype=mrx.DTYPE)
        v0 = v0 / seq.l2_norm(v0, 0, dirichlet=False)
        operators = _commit(seq, _set_null(operators, 0, False, v0[None, :]))

    if _n_vectors(betti_numbers, 1, False):
        # k = 1, no BC: the seed is d zeta itself -- covariant (0, 0, 1), which
        # V^1 contains exactly; histopolated (a direct collocation solve), so
        # its incidence curl is zero to round-off.  A closed seed has no
        # coexact part, so the Leray projection (one L_0 NBC solve, deflated
        # by the constants just stored) IS the harmonic form: curl-free by
        # C G = 0, weakly divergence-free by the solve.  No L_2 solve.
        #
        # It used to seed with M_1^{-1} of the (0, 0, 1) dual load, i.e. the
        # form with CONTRAVARIANT (0, 0, 1) -- |curl|/|v| ~ 2 on QA -- and
        # remove that curl with an L_2 free solve.  At p = 4, n >= 24 that
        # solve exhausts its budget and the leftover curl (4e-3 -> 4e-2 from
        # n = 24 to 32) was the whole Route-A convergence stall of
        # scripts/analytic_vacuum.py.  The k = 2 Dirichlet form above cannot
        # take this shortcut: it is the Hodge star of d zeta, metric-weighted,
        # never in V^2, so its L_1 solve is doing real work.
        dzeta = jnp.asarray((0.0, 0.0, 1.0), dtype=mrx.DTYPE)
        seed1 = seq.interpolate(lambda x_hat: dzeta, 1, dirichlet=False, frame='ref')
        curl_seed = seq.apply_incidence_matrix(
            seed1, 1, dirichlet_in=False, dirichlet_out=False)
        closed = float(seq.l2_norm(curl_seed, 2, dirichlet=False)
                       / seq.l2_norm(seed1, 1, dirichlet=False))
        # A histopolated seed is closed to the round-off of its working-
        # precision incidence apply, not to the solves' residual tolerance.
        if closed > mrx.sqrt_eps():
            raise RuntimeError(
                "compute_nullspaces: the d zeta seed is not closed "
                f"(|C seed| / |seed| = {closed:.2e} > sqrt(eps) {mrx.sqrt_eps():.1e}); the "
                "k=1 free harmonic form would carry that curl unremoved")
        v1, _ = seq.apply_leray_projection(seed1, k=1)
        v1 = v1 / seq.l2_norm(v1, 1, dirichlet=False)
        operators = _commit(seq, _set_null(operators, 1, False, v1[None, :]))

    if verbose:
        for k, dirichlet in ((3, True), (2, True), (0, False), (1, False)):
            for i, v in enumerate(get_nullspace(operators, k, dirichlet)):
                rq = harmonic_rayleigh(seq, v, k, dirichlet, operators)
                line = (f"[nullspace] k={k} {'dbc' if dirichlet else 'free'} "
                        f"form {i}: v^T L v / v^T M v = {rq:.2e}")
                if gap_sweeps and k in (1, 2):
                    lam, sweeps = estimate_spectral_gap(
                        seq, operators, k, dirichlet, maxiter=gap_sweeps)
                    # The gap sweep is one shifted saddle solve; in float32 that
                    # solve can diverge for a near-singular block (e.g. QA k=2),
                    # returning a non-finite estimate. Report that plainly instead
                    # of printing NaN; the estimate needs float64 to be reliable.
                    if jnp.isfinite(lam):
                        line += (f",  lambda_1 ~ {lam:.2e} ({sweeps} sweeps)"
                                 f"  ->  ratio {rq / lam:.1e}")
                    else:
                        line += (f",  lambda_1 gap sweep did not converge "
                                 f"({sweeps} sweeps) -- float32 shifted solve; "
                                 f"use float64 for the gap")
                print(line, flush=True)

    return operators


# ---------------------------------------------------------------------------
# Iterative nullspace construction (arbitrary topology)
# ---------------------------------------------------------------------------

def _logical_constant_seed(seq, operators, k, dirichlet, components):
    """``M_k^{-1}`` of the dual load with constant reference *integrand*.

    ``components`` go straight into :func:`load` with ``frame='ref'``, i.e.
    they are the vector the basis is paired against, not the primal
    coefficients of a form: for k=1 the result is the form with
    CONTRAVARIANT components ``components`` (covariant ``g . components``),
    for k=2 the one with primal proxy ``J g^{-1} . components``.  Both are
    metric-weighted and neither is closed nor co-closed -- on QA the k=1
    seed has ``|curl| / |v| ~ 2``.  That is fine for what this is: a
    topologically right *guess* for inverse iteration, built without ``1/R``,
    a physical frame, or a Cartesian sample (so no zeta = 0 seam exposure).
    It is NOT an exact representative; :func:`compute_nullspaces` seeds with
    the histopolated ``d zeta`` / ``dr^dchi`` instead, which are exactly
    closed.
    """
    comps = jnp.asarray(components, dtype=mrx.DTYPE)
    return seq.apply_inverse_mass_matrix(
        seq.load(lambda x_hat: comps, k, dirichlet=dirichlet, frame='ref'),
        k, dirichlet=dirichlet, operators=operators)


def _initial_guesses(seq, operators, k, dirichlet, n_vec):
    """Return a length-``n_vec`` list of analytic initial guesses (or ``None``).

    Every vector-valued case uses the constant reference-frame form of the
    matching degree (see :func:`_logical_constant_seed`):

    * ``k = 0, no DBC``: the constant scalar field ``1``           (``b0``).
    * ``k = 1, no DBC``: ``dzeta``          -> ``(0, 0, 1)``       (``b1``).
    * ``k = 2, DBC``   : ``dr ^ dchi``      -> ``(0, 0, 1)``       (``b1``).
    * ``k = 3, DBC``   : the constant, lifted via ``M_3^{-1}``     (``b0``).

    The k=1/k=2 pair are the toroidal-loop and toroidal-flux classes on a solid
    torus.  ``b2 > 0`` topologies (a shell, e.g. betti ``(1, 1, 1, 0)``) also
    populate ``k = 2, no DBC`` and ``k = 1, DBC``; the natural seeds there are
    the *poloidal-surface* classes ``dchi ^ dzeta`` and ``dr``, i.e.
    ``(1, 0, 0)`` in both cases.  Those are left as ``None`` (random init)
    until a b2 > 0 sequence is actually exercised -- a seed that happens to be
    M-orthogonal to the harmonic space is worse than a random one.

    Any remaining slots are ``None`` (fall back to the random init).
    """
    if n_vec == 0:
        return []
    guesses = [None] * n_vec
    if k == 0 and not dirichlet:
        guesses[0] = jnp.ones(seq.n(0), dtype=mrx.DTYPE)
    elif k == 3 and dirichlet:
        guesses[0] = seq.apply_inverse_mass_matrix(
            jnp.ones(seq.n(3, True), dtype=mrx.DTYPE), 3, dirichlet=True, operators=operators)
    elif k == 1 and not dirichlet:
        guesses[0] = _logical_constant_seed(
            seq, operators, 1, False, (0.0, 0.0, 1.0))
    elif k == 2 and dirichlet:
        # dr ^ dchi: zero flux through the rho = const boundary, so the seed
        # already lives in the DBC space instead of being projected into it.
        guesses[0] = _logical_constant_seed(
            seq, operators, 2, True, (0.0, 0.0, 1.0))
    return guesses


def compute_nullspaces_iterative(seq, operators=None, betti_numbers=None,
                                 eps=1e-4, abs_tol=None, inner_tol=1e-6,
                                 maxiter=100):
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
        Shift used to regularise the stiffness block.  A *fixed* constant, not
        a mesh-dependent one: in FEEC the discrete harmonic space lies exactly
        in ``ker(L_k)`` (its dimension is the Betti number), so there is no
        h-dependent near-null floor to chase.  The only requirement is
        ``eps << lambda_1``, and ``lambda_1`` -- the first non-harmonic
        eigenvalue -- is a continuum quantity, independent of the mesh and
        O(1) once lengths are normalised to a unit major radius.  Shrinking
        ``eps`` with h therefore buys no outer convergence (the ratio is
        already ~1e-4 per sweep) while making the shifted solve
        correspondingly worse conditioned.  Verify the margin with
        :func:`estimate_spectral_gap` rather than assuming it.
    abs_tol : float, optional
        Absolute tolerance on the Hodge-Laplacian residual ``||L_k v||``.
        Defaults to ``seq.tol``.  Keep it comfortably above ``inner_tol``;
        the outer loop's stall guard catches the rest.
    inner_tol : float
        Tolerance for the inner shifted MINRES solve at each power-iteration
        step.  This sets an accuracy floor on the outer iteration (see
        :func:`find_nullspace_vectors`); 1e-6 is the measured-good value on
        W7-X and 1e-3 is not.

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
            n_vectors = _n_vectors(betti_numbers, k, dirichlet)
            guesses = _initial_guesses(seq, operators, k, dirichlet, n_vectors)
            vectors, iters = find_nullspace_vectors(
                seq,
                operators,
                k,
                n_vectors,
                eps,
                dirichlet=dirichlet,
                x0s=guesses,
                abs_tol=abs_tol,
                inner_tol=inner_tol,
                maxiter=maxiter,
            )
            operators = _commit(seq, _set_null(operators, k, dirichlet, vectors))
            info[(k, dirichlet)] = iters

    return operators, info


def find_nullspace_vectors(seq, operators, k, n_vectors, eps, dirichlet=True,
                           x0s=None, abs_tol=None, inner_tol=1e-6,
                           maxiter=100, stall_ratio=0.9, known=None):
    """Find ``n_vectors`` harmonic ``k``-forms via inverse iteration.

    Each vector is found by repeatedly applying ``(S_k + eps M_k)^{-1} M_k``
    with M-orthogonalisation against the previously found vectors. Uses
    ``jax.lax.while_loop`` so the inner iteration is JIT-compatible.

    Parameters
    ----------
    known : array ``(m, n_k)``, optional
        M-orthonormal vectors that are already known and only DEFLATED
        against, never re-solved or returned -- the stored harmonic forms
        when :func:`estimate_spectral_gap` aims a slot outside the kernel.
    x0s : list of optional arrays, length ``n_vectors``
        Per-vector initial guesses. Entries that are ``None`` fall back to
        a deterministic random initialisation.
    abs_tol : float
        Tolerance on ``sqrt(v^T L_k v / v^T M_k v)`` -- the relative L2 error
        of the form, which is what callers judge the vector by. NOT on
        ``||L_k v||``: that measures a dual vector in the primal mass norm, so
        it carries ``||L|| ~ h^-2`` and a fixed bound on it moves the stopping
        point with resolution and degree. If the normalised
        initial guess already satisfies it (after M-orthogonalisation
        against previously-found vectors), it is accepted directly and no
        inverse iteration is run for that slot.
    inner_tol : float
        Tolerance for the inner shifted saddle solve.  The outer loop tests
        the *true* residual of whatever vector comes back, so a loose inner
        solve cannot make the exit criterion lie -- but it does set an
        accuracy FLOOR: the perturbed iteration's fixed point is displaced by
        O(inner_tol), so the outer residual plateaus there and no number of
        sweeps gets below it.  Measured on W7-X: 1e-3 plateaus far above the
        harmonic vector (30% field error), 1e-6 converges.  Do not loosen this
        without re-measuring -- and note it interacts with ``stall_ratio``,
        which will happily accept the plateau as convergence.
    stall_ratio : float
        Outer-loop stall guard.  Iteration stops once a sweep fails to reduce
        the residual by at least this factor.  Without it a request for an
        ``abs_tol`` below what ``inner_tol`` can deliver silently burns all
        ``maxiter`` saddle solves.  With it, an ``inner_tol`` that is too loose
        turns a slow grind into an early wrong answer, so the two must be set
        together.
    Returns
    -------
    vs : jnp.ndarray
        Stacked array of shape ``(n_vectors, n_k)``. Empty shape
        ``(0, n_k)`` when ``n_vectors == 0``.
    iters : list of (int, float, float)
        ``(n_iters, residual, rayleigh)`` per vector: the Hodge-Laplacian
        residual ``||L_k v||`` and the Rayleigh quotient
        ``v^T L_k v / v^T M_k v``.  ``n_iters == 0`` means the initial guess
        was accepted without iteration.  The Rayleigh quotient is the
        interpretable one -- it carries the units of an eigenvalue, so it is
        directly comparable to the first non-harmonic eigenvalue (see
        :func:`estimate_spectral_gap`), whereas ``||L_k v||`` measures a dual
        vector in the primal mass norm and its scale drifts with resolution.
    """
    if abs_tol is None:
        abs_tol = seq.tol
    n = _dof_count(seq, k, dirichlet)
    if n_vectors == 0:
        return jnp.zeros((0, n), dtype=mrx.DTYPE), []

    found = [] if known is None else [jnp.asarray(v, dtype=mrx.DTYPE) for v in known]
    n_known = len(found)
    iters = []
    for idx in range(n_vectors):
        seeded = x0s is not None and idx < len(x0s) and x0s[idx] is not None
        if seeded:
            v0 = x0s[idx]
        else:
            v0 = jax.random.normal(jax.random.PRNGKey(idx), (n,))
        # M-orthogonalise against already-found vectors: one mass apply for
        # all of them (they are M-orthonormal, so the projections commute).
        found_stacked = jnp.stack(found) if found else None

        project_out = (lambda v: v) if found_stacked is None else deflation_projectors(
            found_stacked, lambda v: seq.apply_mass_matrix(v, k, dirichlet=dirichlet))[0]

        v0 = project_out(v0)
        v0 = v0 / seq.l2_norm(v0, k, dirichlet=dirichlet)

        # Early exit if the initial guess is already harmonic to tolerance,
        # on the SAME criterion the loop uses: the Rayleigh quotient of the
        # full L (not of the stiffness block alone -- at k = 3 that block is
        # zero and v^T S v would read 0 for any vector). v0 is M-normalised
        # just above, so v0 @ Lv0 is the quotient.
        Lv0 = seq.apply_laplacian(
            v0, k, dirichlet=dirichlet, operators=operators)
        res_init = float(seq.l2_norm(Lv0, k, dirichlet=dirichlet))
        rq_init = float(v0 @ Lv0)
        if rq_init <= abs_tol ** 2:
            found.append(v0)
            iters.append((0, res_init, rq_init))
            continue

        def body_fn(state):
            v, rq, _rq_prev, i = state
            Mv = seq.apply_mass_matrix(
                v, k, dirichlet=dirichlet)
            w = seq.apply_inverse_shifted_laplacian(
                Mv, k, eps, dirichlet=dirichlet, guess=v,
                operators=operators,
                tol=inner_tol)
            w = project_out(w)
            # M is SPD so w^T M w >= 0 exactly; in float32 the deflated
            # cancellation can round it slightly negative, which would make
            # l2_norm's sqrt return NaN and kill the gap estimate. Guard the
            # sqrt and the division so a single-precision sweep stays finite.
            wgram = seq.l2_norm_sq(w, k, dirichlet=dirichlet)
            wnorm = jnp.sqrt(jnp.maximum(wgram, 0.0))
            w = w / jnp.where(wnorm > 0.0, wnorm, 1.0)
            Lw = seq.apply_laplacian(
                w, k, dirichlet=dirichlet, operators=operators)
            # w is M-normalised on the line above, so w @ Lw IS the Rayleigh
            # quotient w^T L w / w^T M w.
            return w, w @ Lw, rq, i + 1

        def cond_fn(state):
            _, rq, rq_prev, i = state
            progressing = rq < stall_ratio * rq_prev
            # i == 0 forces the first sweep; rq/rq_prev are inf until then.
            # Terminate on the RAYLEIGH QUOTIENT, not on ||L v||. The latter
            # measures a dual vector in the primal mass norm, so its scale
            # drifts with resolution (this function's own docstring says so)
            # and comparing it to a fixed tolerance means the stopping point
            # moves with h and p. `rq` is a generalized eigenvalue, directly
            # comparable to lambda_1, and `sqrt(rq)` is the relative L2 error
            # in the form -- the quantity every caller actually judges the
            # vector by. abs_tol is a bound on sqrt(rq).
            return (i == 0) | ((rq > abs_tol ** 2) & (i < maxiter)
                               & progressing)

        init_state = (v0, jnp.inf, jnp.inf, 0)
        v_final, rq_final, _, n_iters = jax.lax.while_loop(
            cond_fn, body_fn, init_state)
        found.append(v_final)
        res_final = float(seq.l2_norm(
            seq.apply_laplacian(
                v_final, k, dirichlet=dirichlet, operators=operators),
            k, dirichlet=dirichlet))
        iters.append((int(n_iters), res_final, float(rq_final)))

    return jnp.stack(found[n_known:]), iters


def estimate_spectral_gap(seq, operators, k, dirichlet, eps=1e-4, *,
                          maxiter=5, **kwargs):
    """Estimate the first non-harmonic eigenvalue ``lambda_1`` of ``L_k``.

    One slot of inverse iteration, M-orthogonalised against the harmonic
    forms stored on ``operators``, so it converges to the lowest eigenvector
    outside the kernel and its Rayleigh quotient to ``lambda_1`` -- from
    above, since the problem is symmetric, and it never reaches ``abs_tol``,
    so it exits on ``maxiter`` or the stall guard.  A handful of sweeps is
    the point: after ``n`` sweeps from a random start the quotient is a
    weighted mean of the lowest few eigenvalues with weights
    ``(lambda_1 / lambda_j)^{2n}``, i.e. the right order of magnitude after
    the first sweep and within the lowest cluster after five, which is all
    that reading a harmonic form's quotient against it needs.  Convergence
    to the digit is slow when the spectrum is dense (``lambda_2 / lambda_1``
    near 1) and not worth a solve per digit.

    Each sweep is one shifted saddle solve at ``inner_tol`` (``kwargs`` go
    to :func:`find_nullspace_vectors`).

    The number certifies the shift of the iterative route: inverse iteration
    on ``(S_k + eps M_k)^{-1} M_k`` reduces non-harmonic content by
    ``eps / (lambda_1 + eps)`` per sweep, and the residual-to-error
    conversion for the harmonic vectors themselves is also governed by
    ``lambda_1``.  Both are assumptions about a quantity nothing else
    measures.

    Returns ``(lambda_1, n_sweeps)``.
    """
    known = get_nullspace(operators, k, dirichlet)
    _, iters = find_nullspace_vectors(
        seq, operators, k, 1, eps, dirichlet=dirichlet, x0s=[None],
        maxiter=maxiter, known=known, **kwargs)
    n_sweeps, _, lam = iters[0]
    return lam, n_sweeps
