"""One solve per degree, on the session geometry.

The vacuum field ``B* = grad Psi`` (``test/manufactured.py``) is recovered
through the k = 0, 1, 2 solves, each the production solve of its degree with
its production preconditioner; the k = 3 solve is the Leray projection, which
must remove a discrete co-exact perturbation of the equilibrium field exactly:

    k = 0   Route A   S_0 f = G^T load_1(B*)            grad f_h -> B*
    k = 1   Route C   L_1 A = C^T load_2(B*)            curl A_h -> B*     (the Hodge split)
    k = 2   the shifted solve (M_2 + eps L_2) u = load_2(B*)    u_h -> B*  (the split identity)
    k = 3   Leray(B_0 + grad_w q) = B_0                            (the k=3 Laplacian)

The k = 2 row holds because a curl-free, div-free field has no weak boundary
terms, so ``M_2 B* + eps L_2 B* = M_2 B*`` weakly; the discrete solution is
the L2 projection smoothed by ``(I + eps M^-1 L)^-1``, which costs a factor
``~ 1 + eps lambda`` on the projection error at the smoothing ``eps``. The
bands are 1.25x the values measured at ``(8, 12, 12)`` p=3 on li383 (see the
print): one-resolution consistency of operator, preconditioner and boundary
handling, not a convergence study (``scripts/analytic_vacuum.py`` is that).
The k = 3 check is exact to solver tolerance.
"""
import jax.numpy as jnp
import numpy as np
import pytest

import mrx
from test.conftest import NS
from test.manufactured import make_b_star, relative_error

# Relative M-norm error against B*, li383 (8, 12, 12) p=3, measured 2026-09-02,
# band 1.25x: 1.07e-1 (k=0), 4.83e-2 (k=1), 3.13e-1 (k=2). The k=2 value is the
# L2-projection error 4.76e-2 smoothed by (I + eps M^-1 L)^-1 at the production
# eps = 1e-3; it tends to 4.8e-2 as eps -> 0 (measured 6.6e-2 at 1e-5, 1.7e-1 at
# 1e-4, 5.8e-1 at 1e-2), so the band checks the shifted operator at the eps it
# is used with, not the approximation.
BAND = {0: 0.134, 1: 0.0604, 2: 0.391}
EPS_SMOOTHING = 0.064 / NS[0] ** 2


def _route_a(seq, b_star):
    load1 = seq.load(b_star, 1, dirichlet=False)
    rhs = seq.apply_incidence_matrix(load1, 0, dirichlet_in=False, dirichlet_out=False,
                                     transpose=True)
    f, info = seq.apply_inverse_laplacian(rhs, 0, dirichlet=False, return_info=True)
    return seq.apply_strong_grad(f, dirichlet_in=False, dirichlet_out=False), 1, info


def _route_c(seq, b_star):
    load2 = seq.load(b_star, 2, dirichlet=False)
    rhs = seq.apply_incidence_matrix(load2, 1, dirichlet_in=False, dirichlet_out=False,
                                     transpose=True)
    A, info = seq.apply_inverse_laplacian(rhs, 1, dirichlet=False, return_info=True)
    return seq.apply_strong_curl(A, dirichlet_in=False, dirichlet_out=False), 2, info


def _shifted(seq, b_star):
    load2 = seq.load(b_star, 2, dirichlet=False)
    u, info = seq.apply_inverse_mass_plus_eps_laplace_matrix(
        load2, 2, EPS_SMOOTHING, dirichlet=False, return_info=True)
    return u, 2, info


@pytest.mark.parametrize("k", (0, 1, 2))
def test_vacuum_field_is_recovered(seq, k):
    b_star = make_b_star(seq.map)
    u_h, k_form, info = {0: _route_a, 1: _route_c, 2: _shifted}[k](seq, b_star)
    assert int(info) <= 0, f"k={k}: the solve did not converge ({int(info)} iterations)"
    relerr = relative_error(seq, u_h, k_form, False, b_star)
    print(f"\n  k={k}: relerr {relerr:.3e} ({abs(int(info))} iterations)")
    assert np.isfinite(relerr) and relerr < BAND[k], f"k={k}: relerr {relerr:.3e} > {BAND[k]:.3e}"


def test_leray_projection_removes_the_co_exact_part(seq, b0):
    """``Leray(B_0 + grad_w q) = B_0`` to solver tolerance for a random 3-form
    ``q``: ``B_0`` is exactly div-free and ``grad_w q`` is exactly co-exact, so
    the k=3 solve must find ``q`` and remove it entirely."""
    q = jnp.asarray(np.random.default_rng(3).standard_normal(seq.n(3, True)), dtype=mrx.DTYPE)
    w = seq.apply_weak_grad(q, True)
    w = w * (seq.l2_norm(b0, 2) / seq.l2_norm(w, 2))          # a perturbation of size ||B_0||
    b_h, _ = seq.apply_leray_projection(b0 + w, k=2)
    rel = float(seq.l2_norm(b_h - b0, 2) / seq.l2_norm(w, 2))

    def div_norm(v):
        return float(seq.l2_norm(seq.apply_incidence_matrix(v, 2, True, True), 3))
    # The divergence left over, relative to the divergence that was removed:
    # div amplifies the solve residual by the operator norm, so ||div B_h|| /
    # ||B_h|| alone would compare a solve residual with a field norm.
    div = div_norm(b_h) / div_norm(w)
    print(f"\n  Leray: residual perturbation {rel:.2e}, ||div B_h|| / ||div w|| {div:.1e}")
    assert rel < 100 * seq.tol, f"Leray left {rel:.2e} of the co-exact perturbation"
    assert div < 100 * seq.tol, f"Leray left {div:.2e} of the divergence"
