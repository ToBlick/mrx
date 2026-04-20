import jax
import jax.numpy as jnp


def get_nullspace(seq, k, dirichlet):
    """Return the list of nullspace vectors for the k-th Hodge Laplacian."""
    attr = f"null_{k}_dbc" if dirichlet else f"null_{k}"
    return getattr(seq, attr)


def get_saddle_point_nullspaces(seq, k, dirichlet):
    """
    Compute nullspace vectors for the saddle-point system from the
    Schur complement nullspace vectors.

    If v is in null(S_k + D_{k-1} M_{k-1}^{-1} D_{k-1}^T), then
    [v, M_{k-1}^{-1} D_{k-1}^T v] is in the nullspace of the
    saddle-point system.
    """
    vs_upper = get_nullspace(seq, k, dirichlet)
    vs_lower = []
    if k >= 1:
        for v in vs_upper:
            Dt_v = seq.apply_derivative_matrix(
                v, k - 1, dirichlet_in=dirichlet,
                dirichlet_out=dirichlet, transpose=True)
            s = seq.apply_inverse_mass_matrix(
                Dt_v, k - 1, dirichlet=dirichlet)
            vs_lower.append(s)
    return vs_upper, vs_lower


def compute_nullspaces(seq):
    """
    Compute the nullspace of the k-th Hodge Laplacian.
    TODO: For now this only handles the case where the nullspace is 1-dim.
    """
    seq.null_0_dbc = []
    seq.null_1_dbc = []
    v3 = seq.apply_inverse_mass_matrix(
        jnp.ones(seq.n3_dbc), 3, dirichlet=True)
    v3 /= seq.l2_norm(v3, 3, dirichlet=True)
    seq.null_3_dbc = [v3]
    v, _ = seq.apply_leray_projection(
        jnp.ones(seq.n2_dbc), k=2)
    curl_v_dual = seq.apply_derivative_matrix(
        v, 1, dirichlet_in=True, dirichlet_out=True, transpose=True)
    a = seq.apply_inverse_hodge_laplacian(curl_v_dual, 1, dirichlet=True)
    curl_a = seq.apply_strong_curl(a, True, True)
    v2 = v - curl_a
    v2 /= seq.l2_norm(v2, 2, dirichlet=True)
    seq.null_2_dbc = [v2]

    # no Dirichlet BCs (all defaults to False)
    v0 = jnp.ones(seq.n0)
    v0 /= seq.l2_norm(v0, 0, False)
    seq.null_0 = [v0]
    seq.null_2 = []
    v, _ = seq.apply_leray_projection(
        jnp.ones(seq.n1), k=1)
    curl_v_dual = seq.apply_derivative_matrix(
        v, 1, dirichlet_in=False, dirichlet_out=False)
    a = seq.apply_inverse_hodge_laplacian(curl_v_dual, 2, dirichlet=False)
    curl_a = seq.apply_weak_curl(a, False, False)
    v1 = v - curl_a
    v1 /= seq.l2_norm(v1, 1, False)
    seq.null_1 = [v1]
    seq.null_3 = []


def compute_nullspaces_iterative(seq, betti_numbers, eps=1e-6):
    """
    Compute the nullspaces of the Hodge Laplacians for the provided Betti numbers using inverse power iterations.

    Args:
        seq: DeRhamSequence instance.
        betti_numbers (list of ints): A list containing the Betti numbers for k=0,1,2,3. This is, in order:
        - the number of connected components (k=0) - this is always 1
        - the number of tunnels/handles (k=1)
        - the number of voids/cavities (k=2)
        - zero (k=3)
        eps (float): offset for the power iteration.
    """
    assert len(
        betti_numbers) == 4, "betti_numbers must be a list of 4 integers"
    assert betti_numbers[0] == 1, "betti_numbers[0] must be 1 (number of connected components)"
    assert betti_numbers[3] == 0, "betti_numbers[3] must be 0"

    # k = 0:
    seq.null_0_dbc = []  # no nullspace for 0-forms with Dirichlet BCs
    v0 = jnp.ones(seq.n0)
    v0 /= seq.l2_norm(v0, 0, False)
    # we have betti_0=1 nullspace vectors for 0-forms without Dirichlet BCs (the constant function)
    seq.null_0 = [v0]

    # k = 1:
    # betti_2 nullspace vectors for 1-forms with Dirichlet BCs
    seq.null_1_dbc, info_1_dbc = find_nullspace_vectors(
        seq, 1, betti_numbers[2], eps, dirichlet=True)
    # betti_1 nullspace vectors for 1-forms without Dirichlet BCs
    seq.null_1, info_1 = find_nullspace_vectors(
        seq, 1, betti_numbers[1], eps, dirichlet=False)

    # k = 2:
    # betti_1 nullspace vectors for 2-forms with Dirichlet BCs
    seq.null_2_dbc, info_2_dbc = find_nullspace_vectors(
        seq, 2, betti_numbers[1], eps, dirichlet=True)
    # betti_2 nullspace vectors for 2-forms without Dirichlet BCs
    seq.null_2, info_2 = find_nullspace_vectors(
        seq, 2, betti_numbers[2], eps, dirichlet=False)

    info = {
        (1, True): info_1_dbc,
        (1, False): info_1,
        (2, True): info_2_dbc,
        (2, False): info_2,
    }

    # k = 3:
    v3 = seq.apply_inverse_mass_matrix(
        jnp.ones(seq.n3_dbc), 3, dirichlet=True)
    v3 /= seq.l2_norm(v3, 3, dirichlet=True)
    # we have betti_0=1 nullspace vectors for 3-forms with Dirichlet BCs (the constant function)
    seq.null_3_dbc = [v3]
    # no nullspace for 3-forms without Dirichlet BCs
    seq.null_3 = []

    return info


def find_nullspace_vectors(seq, k, n_vectors, eps, dirichlet=True):
    """
    Find n_vectors nullspace vectors of the k-th Hodge Laplacian
    using inverse iteration with shift eps.

    Each eigenvector is found by repeated application of
    (S_k + eps * M_k)^{-1} M_k, with M-orthogonalization against
    previously found vectors. Uses jax.lax.while_loop for JIT
    compatibility.
    """
    if n_vectors == 0:
        return [], []

    n = getattr(seq, f"n{k}_dbc" if dirichlet else f"n{k}")
    found = []
    iters = []

    for _ in range(n_vectors):
        key = jax.random.PRNGKey(len(found))
        v0 = jax.random.normal(key, (n,))
        # M-orthogonalize against already found vectors
        for u in found:
            v0 = v0 - (u @ seq.apply_mass_matrix(v0,
                       k, dirichlet=dirichlet)) * u
        v0 = v0 / seq.l2_norm(v0, k, dirichlet=dirichlet)

        def body_fn(state):
            v, rq, _, i = state
            # apply (S_k + eps M_k)^{-1} M_k
            Mv = seq.apply_mass_matrix(v, k, dirichlet=dirichlet)
            w = seq.apply_inverse_shifted_stiffness(
                Mv, k, eps, dirichlet=dirichlet, guess=v)
            # M-orthogonalize against found vectors
            for u in found:
                w = w - (u @ seq.apply_mass_matrix(w,
                         k, dirichlet=dirichlet)) * u
            # M-normalize
            w = w / seq.l2_norm(w, k, dirichlet=dirichlet)
            # Rayleigh quotient: v^T S_k v (should -> 0 for nullspace)
            Sv = seq.apply_stiffness(w, k, dirichlet=dirichlet)
            rq_new = jnp.abs(w @ Sv)
            return w, rq_new, rq, i + 1

        def cond_fn(state):
            _, rq, rq_prev, i = state
            converged = (rq <= seq.tol) & (
                jnp.abs(rq - rq_prev) <= seq.tol)
            return ~converged & (i < seq.maxiter)

        init_state = (v0, jnp.inf, jnp.inf, 0)
        v_final, rq_final, _, n_iters = jax.lax.while_loop(
            cond_fn, body_fn, init_state)
        found.append(v_final)
        iters.append((int(n_iters), float(rq_final)))

    return found, iters
